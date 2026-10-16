from tqdm import tqdm
import numpy as np
from scipy.stats import cauchy, combine_pvalues
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
from pathlib import Path

import joblib

from safetycage.safetycage import SafetyCage

# Gaussian Mixture Model size limits, see _fit_gaussian_mixture
GMM_MAX_COMPONENTS = 3
GMM_SAMPLES_PER_COMPONENT = 10

# Constructor settings that change how a fitted cage behaves, stored by save_cage
SAVED_SETTINGS = (
    "s_statistic_source",
    "pvalue_method",
    "calibration_fraction",
    "test_type_between_layers",
    "cauchy_weights_per_layer",
    "minimum_sample_size",
)

class SPARDACUS(SafetyCage):
    """
    SPARDACUS Safety Cage Method.

    The SPARDACUS safety cage detects misclassified samples by comparing how a sample’s 
    internal neural network activations differ from correctly and incorrectly classified 
    training samples.

    The method models this by learning a projection that separates correct and 
    incorrect activations for each class and layer. During prediction after training,
    it projects a sample’s activation and evaluates how likely it is under both the 
    correct and incorrect distributions. This comparison is then converted into a
    p-value using fitted density estimators and ECDFs on the likelihood values.

    By default (``pvalue_method="monte_carlo"``) the p-value follows the paper: the
    ECDF is built from Monte Carlo draws of the fitted Gaussian mixture. As an opt-in
    alternative, ``pvalue_method="conformal"`` holds out ``calibration_fraction`` of
    the reference samples, fits on the rest, and computes the p-value from the held-out
    samples' scores. Its p-values stay calibrated even when the Gaussian mixture fits
    the data poorly, at the cost of fitting on fewer samples.

    The smaller the p-value, the more likely the sample is to be misclassified. When 
    multiple layers are used, a global p-value is found by combining the each layer's 
    p-value using either Fisher’s method or the Cauchy combination test. The optimal
    threshold to compare the resulting p-values is given to the threshold attribute.

    NOTE: This method **only works for neural network models**, as it relies
    on intermediate layer activations and learned representations.

    See the below research paper for a thorough explanation of the SPARDACUS method.

    **Reference:**
        P. V. Johnsen and F. Remonato. “SPARDACUS SafetyCage: A new misclassification detector”.
        https://proceedings.mlr.press/v265/johnsen25a.html

    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
        classes (dict): Mapping of class indices to class labels.
        layer_params (dict): Stores parameters including such as projection vectors, density 
        estimators, and ECDFs for each layer and class from training.
        unreliable_classes (set): Set of class labels for which density estimation
            failed or was unreliable due to insufficient amount of correct/incorrect predictions.

        s_statistic_source (str): Determines which distribution is used to
            compute p-values ("correctly" or "incorrectly").
        threshold (float | None): Significance threshold used for flagging.
        cauchy_weights_per_layer (list): Weights used for Cauchy combination test.
        test_type_between_layers (str): Method for combining p-values across layers.
        minimum_sample_size (int): Minimum number of samples required to fit
            Gaussian Mixture Models.
        pvalue_method (str): How p-values are computed ("monte_carlo" or "conformal").
        calibration_fraction (float): Share of reference samples held out for
            conformal p-values.
    """
    def __init__(self, model_module, data_module, **kwargs):
        """
        Initialize the SPARDACUS safety cage.

        Args:
            model_module: Reference to model module object for making predictions.
            data_module: Reference to data module object for handling data.
            s_statistic_source (str, optional): Source used to compute p-values ("correctly" or "incorrectly"). (default: "correctly")
            threshold (float): Significance threshold for flagging samples.
            test_type_between_layers (str, optional): Method for combining p-values ("fisher" or "cauchy"). (default: "cauchy", as in the paper)
            cauchy_weights_per_layer (list[float]): Weights for the Cauchy combination test.
            minimum_sample_size (int, optional): Minimum number of samples required to fit density models. (default: 3)
            pvalue_method (str, optional): "monte_carlo" builds the ECDF from draws of the fitted Gaussian
                mixture, as in the paper; "conformal" uses held-out calibration samples instead. (default: "monte_carlo")
            calibration_fraction (float, optional): Share of the reference samples (correct ones for
                s_statistic_source="correctly", incorrect ones for "incorrectly") held out for conformal
                p-values. Only used when pvalue_method="conformal". (default: 0.3)
        """
        super(SPARDACUS, self).__init__(model_module, data_module, **kwargs)

        self.s_statistic_source = kwargs.get("s_statistic_source", "correctly")
        if self.s_statistic_source not in ("correctly", "incorrectly"):
            raise ValueError(f"s_statistic_source must be \"correctly\" or \"incorrectly\". Provided: {self.s_statistic_source!r}")
        self.cauchy_weights_per_layer = kwargs.get("cauchy_weights_per_layer")
        self.test_type_between_layers = kwargs.get("test_type_between_layers", "cauchy")

        # For the Gaussian Mixture Model fitting: classes with fewer samples are flagged as unreliable
        # (see _fit_gaussian_mixture). Must be at least 3, as fewer points cannot support a meaningful fit.
        # The default is that floor: accurate models often leave only a handful of incorrect samples per
        # class (MNIST: 2-9 for most classes), and a single Gaussian fitted to them still ranks mistakes well.
        if "minimum_sample_size" in kwargs and kwargs["minimum_sample_size"] < 3:
            raise ValueError(f"Minimum_sample_size must be at least 3. Provided: {kwargs['minimum_sample_size']}")
        self.minimum_sample_size = kwargs.get("minimum_sample_size", 3)

        self.pvalue_method = kwargs.get("pvalue_method", "monte_carlo")
        if self.pvalue_method not in ("monte_carlo", "conformal"):
            raise ValueError(f"pvalue_method must be \"monte_carlo\" or \"conformal\". Provided: {self.pvalue_method!r}")
        self.calibration_fraction = kwargs.get("calibration_fraction", 0.3)
        if not 0 < self.calibration_fraction < 1:
            raise ValueError(f"calibration_fraction must be between 0 and 1 (exclusive). Provided: {self.calibration_fraction}")

        self.classes = data_module.classes
        self.unreliable_classes = set()
    
    @property
    def name(self):
        """Return the name of the safety cage method."""
        return "SPARDACUS"

    def fit(
        self,
        x=None,
        y=None,
        y_pred=None,
        find_threshold=False,
        metric_fn=None,
        x_val=None,
        y_val=None,
        search="uniform",
        num_thresholds=1000,
        greater_is_better=True,
    ) -> None:
        """
        Train the SPARDACUS safety cage.

        Separates training samples into correctly and incorrectly classified groups and
        computes parameters for each layer and class, which are later used to evaluate
        new samples during prediction.

        Args:
            x: Input data. If None, the training data is loaded from the data module.
            y: True labels for x. Required when x is given.
            y_pred: Model predictions. If None, computed using the model module.
            find_threshold (bool, optional): If True, also choose ``self.threshold``
                on validation data after fitting -- see ``SafetyCage.fit_threshold``
                for what the remaining keyword arguments do. (default: False)
            metric_fn (callable, optional): Metric to maximize (default: MCC).
            x_val: Validation inputs. If None, loaded from the data module.
            y_val: Validation labels. If None, loaded from the data module.
            search (str, optional): Threshold sweep strategy (default: "uniform").
            num_thresholds (int, optional): Number of candidate thresholds (default: 1000).
            greater_is_better (bool, optional): Whether greater metric values are better (default: True).
        """

        if x is None:
            x, y = self.data_module.data_train
        elif y is None:
            raise ValueError("y must be given together with x: the training labels belong to other samples.")
        if y_pred is None:
            y_pred = self.model_module._get_predictions(x)

        # Start from a clean slate so classes flagged by an earlier fit don't stick
        self.unreliable_classes = set()

        if self.model_module.use_onehot_encoder:
            predicted_classes = np.argmax(y_pred, axis=1)
            mask = predicted_classes == np.argmax(y, axis=1)
        else:
            predicted_classes = np.asarray(y_pred)
            mask = y_pred == y

        if isinstance(x, dict):
            x_correct = {key: val[mask] for key, val in x.items()}
            x_incorrect = {key: val[~mask] for key, val in x.items()}
        else:
            x_correct = x[mask]
            x_incorrect = x[~mask]
            
        # Group by the class the model predicted, not the true class: predict()
        # is called with the model's predictions, so the incorrect reference for
        # class i must be samples wrongly predicted as i. (For correct samples
        # the two are the same.)
        predicted_correct = predicted_classes[np.asarray(mask)]
        predicted_incorrect = predicted_classes[~np.asarray(mask)]

        # Get layer activations
        layers_activations = {
            "correct": self.model_module._get_activations(x_correct),
            "incorrect": self.model_module._get_activations(x_incorrect)
        }

        # Initialize parameters dictionary
        selected_layers = self.model_module.selected_layers
        self.layer_params = {
            layer: {class_index: {} for class_index in self.classes}
            for layer in selected_layers
        }
        # Process each layer and class
        for layer in selected_layers:
            for class_key, class_label in tqdm(self.classes.items()):
                
                # Get class-specific activations
                class_activations_correct = self._get_class_activations(
                    layers_activations["correct"], layer, predicted_correct, class_key
                )
                class_activations_incorrect = self._get_class_activations(
                    layers_activations["incorrect"], layer, predicted_incorrect, class_key
                )
                
                # Process layer and class
                self.layer_params[layer][class_label] = self._process_layer_class(
                    class_activations_correct, class_activations_incorrect, class_label
                )

        if find_threshold:
            self.fit_threshold(
                metric_fn=metric_fn,
                x_val=x_val,
                y_val=y_val,
                search=search,
                num_thresholds=num_thresholds,
                greater_is_better=greater_is_better,
            )


    def _get_class_activations(self, layer_activations: dict, layer: str,
                            predicted_classes: np.ndarray, class_index: int) -> np.ndarray:
        """
        Extract activations at a given layer for the samples predicted as a specific
        class (given by class_index).

        Args:
            predicted_classes (numpy.ndarray): Predicted class index of each sample
                (already reduced from one-hot or probability rows by ``fit``).

        Returns:
            numpy.ndarray: Activations for the specified class and layer
        """
        return layer_activations[layer][predicted_classes == class_index, :]


    def _process_layer_class(self, class_activations_correct: np.ndarray, class_activations_incorrect: np.ndarray, class_label: str) -> dict:
        """
        Process activations for a single layer and class.

        Learns a projection that separates correctly and incorrectly classified samples,
        fits density models to the projected values, and prepares statistics used to compute p-values.

        Args:
            class_activations_correct (numpy.ndarray): Activations of correctly classified samples
            class_activations_incorrect (numpy.ndarray): Activations of incorrectly classified samples
            class_label (str): Class label

        Returns:
            dict: Dictionary containing projection vectors, density models, and ECDFs (or, for
            pvalue_method="conformal", sorted calibration scores) for the class and layer
        """
        # For conformal p-values, hold out part of the reference samples before anything is fitted,
        # so the calibration scores come from samples the projection and densities never saw
        calibration_activations = None
        if self.pvalue_method == "conformal":
            if self.s_statistic_source == "correctly":
                class_activations_correct, calibration_activations = self._split_calibration(class_activations_correct)
            else:
                class_activations_incorrect, calibration_activations = self._split_calibration(class_activations_incorrect)

        # Double check if both class_activations_correct and class_activations_incorrect have a positive number of values
        if len(class_activations_incorrect) == 0 or len(class_activations_correct) == 0:
            warnings.warn(f"No incorrect and/or correct samples for class \"{class_label}\" exist in layer activations. This class will be flagged as unreliable and the results "
                          "for this class are unreliable. We recommend using a different safetycage method or ensuring some incorrect and/or correct samples exist in this class.")
            self.unreliable_classes.add(class_label)
            return {
                "ecdf_correct": None,
                "ecdf_incorrect": None,
                "beta_hat": None,
                "density_correct": None,
                "density_incorrect": None,
                "calibration_scores": None
            }
        
        # Run fastSPARDA
        beta_hat, _, _, _ = self.fastSPARDA(
            X_samples = class_activations_correct, 
            Y_samples = class_activations_incorrect
            )
        
        # Get projected samples
        predicted_samples_correct = np.dot(class_activations_correct, beta_hat)
        predicted_samples_incorrect = np.dot(class_activations_incorrect, beta_hat)
        
        # Fit density estimators
        density_correct = self._fit_gaussian_mixture(predicted_samples_correct, "correct", class_label)
        density_incorrect = self._fit_gaussian_mixture(predicted_samples_incorrect, "incorrect", class_label)
        
        # Check if fit_gaussian_mixture failed
        if density_correct == None or density_incorrect == None:
            # Assume warning messages and adding to self.unreliable_classes was taken care of in _fit_gaussian_mixture
            return {
                "ecdf_correct": None,
                "ecdf_incorrect": None,
                "beta_hat": None,
                "density_correct": None,
                "density_incorrect": None,
                "calibration_scores": None
            }

        if self.pvalue_method == "conformal":
            # Same statistic as in prediction, ln f_incorrect - ln f_correct, on the held-out samples
            projected_calibration = np.dot(calibration_activations, beta_hat).reshape(-1, 1)
            calibration_scores = np.sort(
                density_incorrect.score_samples(projected_calibration)
                - density_correct.score_samples(projected_calibration)
            )
            return {
                "ecdf_correct": None,
                "ecdf_incorrect": None,
                "beta_hat": beta_hat,
                "density_correct": density_correct,
                "density_incorrect": density_incorrect,
                "calibration_scores": calibration_scores
            }

        # Compute log PDFs
        pdf_results = self._compute_log_pdfs(density_correct, density_incorrect)
        
        # Initialize statistics
        score_statistic_correct = None
        score_statistic_incorrect = None

        # Compute relevant statistics based on configuration
        if self.s_statistic_source == "correctly":
            score_statistic_correct = pdf_results["ln_pdf_h1_correct"] - pdf_results["ln_pdf_h0_correct"]
            
        if self.s_statistic_source == "incorrectly":
            score_statistic_incorrect = pdf_results["ln_pdf_h1_incorrect"] - pdf_results["ln_pdf_h0_incorrect"]
        
        # Compute ECDFs
        ecdf_correct = ECDF(score_statistic_correct) if score_statistic_correct is not None else None
        ecdf_incorrect = ECDF(score_statistic_incorrect) if score_statistic_incorrect is not None else None
        
        return {
            "ecdf_correct": ecdf_correct,
            "ecdf_incorrect": ecdf_incorrect,
            "beta_hat": beta_hat,
            "density_correct": density_correct,
            "density_incorrect": density_incorrect,
            "calibration_scores": None
        }


    def _split_calibration(self, activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly split activations into a fitting part and a held-out calibration part
        of calibration_fraction (at least one sample) for conformal p-values.

        Returns:
            tuple: (fit activations, calibration activations)
        """
        n = len(activations)
        n_calibration = min(n, max(1, int(round(self.calibration_fraction * n))))
        order = np.random.permutation(n)
        return activations[order[n_calibration:]], activations[order[:n_calibration]]


    def _empirical_pvalue(self, reference_scores: np.ndarray, statistic: float) -> float:
        """
        Empirical p-value of a statistic against sorted reference scores: the held-out
        calibration scores (pvalue_method="conformal") or the Monte Carlo draws
        (pvalue_method="monte_carlo").

        For s_statistic_source="correctly" (right-sided), counts reference scores >= statistic;
        for "incorrectly" (left-sided), counts scores <= statistic. The +1 in numerator and
        denominator counts the new sample itself, so the p-value is never 0 and, for samples
        drawn like the reference ones, P(p <= alpha) <= alpha.

        Args:
            reference_scores (numpy.ndarray): Sorted reference statistics
            statistic (float): Statistic of the new sample

        Returns:
            float: p-value in [1 / (n + 1), 1]
        """
        n = len(reference_scores)
        if self.s_statistic_source == "correctly":
            count = n - np.searchsorted(reference_scores, statistic, side="left")
        else:
            count = np.searchsorted(reference_scores, statistic, side="right")
        return (1 + count) / (n + 1)


    def _fit_gaussian_mixture(self, samples: np.ndarray, correctness: str, class_label: str) -> GaussianMixture | None:
        """
        Fit a Gaussian Mixture Model to the given projected samples.

        The number of components (1-3) is chosen by the lowest BIC, with every
        candidate fitted on all samples. BIC already penalizes extra components,
        so no cross-validation split is needed. To keep fits stable, a component
        is only allowed for every GMM_SAMPLES_PER_COMPONENT samples: fewer than 20
        samples give a single Gaussian, 20-29 up to two, and 30 or more up to three.

        **WARNING:** With fewer than minimum_sample_size samples (default 3; this happens
        when a well-performing model makes almost no mistakes for a class), no model is fitted: a warning is raised,
        the class is added to self.unreliable_classes, and None is returned. In such
        cases, we recommend using a different safetycage method or ensuring enough
        correct and incorrect samples exist for each class.

        Args:
            samples (numpy.ndarray): Projected samples
            correctness (str): whether we are fitting predictions that were "correct" or "incorrect"
            class_label (str): Class label

        Returns:
            GaussianMixture | None: Fitted model, or None if there were too few samples
        """

        if len(samples) < self.minimum_sample_size:
            warnings.warn(
                f"[Gaussian Mixture Model WARNING] There are not enough {correctness} samples in the \"{class_label}\" class to fit a Gaussian Mixture Model. "
                f"Provided: {len(samples)}. "
                f"Minimum required: {self.minimum_sample_size}. "
                f"This class will be flagged as unreliable. We recommend that you use a different safetycage method.",
                UserWarning
            )
            self.unreliable_classes.add(class_label)
            return None

        samples = samples.reshape(-1, 1)
        max_components = max(1, min(GMM_MAX_COMPONENTS, len(samples) // GMM_SAMPLES_PER_COMPONENT))

        best_density, best_bic = None, np.inf
        for n_components in range(1, max_components + 1):
            density = GaussianMixture(n_components=n_components, covariance_type="full").fit(samples)
            bic = density.bic(samples)
            if bic < best_bic:
                best_density, best_bic = density, bic

        return best_density


    def _compute_log_pdfs(self, density_correct: GaussianMixture, density_incorrect: GaussianMixture, n_samples: int = int(1e6)) -> dict:
        """
        Compute log-likelihood values for correct and incorrect distributions.

        Args:
            density_correct (GaussianMixture): Model for correct samples
            density_incorrect (GaussianMixture): Model for incorrect samples
            n_samples (int): Number of samples to draw

        Returns:
            dict: Sampled points log-likelihood values
        """

        samples_correct = density_correct.sample(n_samples)[0]
        samples_incorrect = density_incorrect.sample(n_samples)[0]
        
        ln_pdf_h0_correct = density_correct.score_samples(samples_correct)
        ln_pdf_h1_correct = density_incorrect.score_samples(samples_correct)
        ln_pdf_h0_incorrect = density_correct.score_samples(samples_incorrect)
        ln_pdf_h1_incorrect = density_incorrect.score_samples(samples_incorrect)
        
        return {
            "samples_correct": samples_correct,
            "samples_incorrect": samples_incorrect,
            "ln_pdf_h0_correct": ln_pdf_h0_correct,
            "ln_pdf_h1_correct": ln_pdf_h1_correct,
            "ln_pdf_h0_incorrect": ln_pdf_h0_incorrect,
            "ln_pdf_h1_incorrect": ln_pdf_h1_incorrect
        }


    def predict(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Tests cage on given data.

        Computes per-layer p-values for each sample and combines them into a global
        p-value using the configured method.
        
        Args:
            x: Input features array
            y: Target values array
            
        Returns:
            np.ndarray: Vector of global combined p-values per sample. Shape depends on s_statistic_source.
        """

        pvalue = self._compute_statistics(x, y)
        
        return self._combine_layer_pvalues(pvalue, len(y), self.test_type_between_layers)


    def _compute_statistics(self, x, y):
        """
        Compute p-values for each sample and layer based on the fitted density estimators and ECDFs.

        Evaluates how likely each sample is under the correct and incorrect distributions.
        Samples predicted as an unreliable class cannot be scored; they get the most
        suspicious p-value (0 for s_statistic_source="correctly", 1 for "incorrectly")
        so they are always flagged for review and stay in the metrics.

        Args:
            x: Input data samples
            y: True labels

        Returns:
            numpy.ndarray: Matrix of p-values with shape (num_samples, num_layers)
        """
        selected_layers = self.model_module.selected_layers

        num_samples = len(y)
        num_layers = len(selected_layers)
        
        pvalue = np.full(
            shape = (num_samples, num_layers),
            fill_value = np.inf,
            dtype = np.float64
            )
        
        activations = self.model_module._get_activations(x)

        # Most suspicious p-value, given to samples that cannot be scored
        unscored_pvalue = 0.0 if self.s_statistic_source == "correctly" else 1.0

        predicted = np.argmax(y, axis=1) if self.model_module.use_onehot_encoder else np.asarray(y)
        unseen = sorted(set(predicted.tolist()) - set(self.classes))
        if unseen:
            warnings.warn(f"Samples predicted as the classes {unseen} cannot be scored, as these classes were not seen in the training data. "
                        f"They get the most suspicious p-value ({unscored_pvalue}) so they are always flagged.")

        if self.unreliable_classes:
            warnings.warn(f"Samples predicted as the classes {self.unreliable_classes} cannot be scored, due to insufficient data to fit the Gaussian Mixture Models. "
                        f"They get the most suspicious p-value ({unscored_pvalue}) so they are always flagged. Consider using a different safetycage method.")
        
        for layer_index, layer in enumerate(selected_layers): # for all layers
            for sample_index, y_sample, in enumerate(y): # for all predictions to be tested
                
                # Compute p-values of each sample per layer using ECDF function
                class_label = self.classes.get(predicted[sample_index])

                if class_label is None or class_label in self.unreliable_classes:
                    pvalue[sample_index,layer_index] = unscored_pvalue
                    continue

                ## Get the projection vector beta hat and the actication for the sample
                activation = activations[layer][sample_index]
                beta_hat = self.layer_params[layer][class_label]["beta_hat"]
                
                # Compute observed value with respect to beta_hat_i projection for predicted class y[sample_index]:
                activation_projected = np.dot(activation, beta_hat).reshape(1,-1)
                
                # Get the density functions of correctly and incorrectly predicted samples, for the layer
                density_correct = self.layer_params[layer][class_label]["density_correct"]
                density_incorrect = self.layer_params[layer][class_label]["density_incorrect"]
                
                # Compute the s statistic for the sample
                # since -ln(a/b) = ln(b)-ln(a)
                statistic = np.subtract(
                    density_incorrect.score_samples(activation_projected),
                    density_correct.score_samples(activation_projected)
                    )
                
                # Get the ECDF functions for the layer
                ecdf_correct = self.layer_params[layer][class_label]["ecdf_correct"]
                ecdf_incorrect = self.layer_params[layer][class_label]["ecdf_incorrect"]
                
                if self.pvalue_method == "conformal":
                    reference_scores = self.layer_params[layer][class_label]["calibration_scores"]
                else:
                    # The ECDF stores the sorted Monte Carlo scores after a leading -inf
                    ecdf = ecdf_correct if self.s_statistic_source == "correctly" else ecdf_incorrect
                    reference_scores = ecdf.x[1:]

                # Right-sided for "correctly" (small p-value: likely misclassified), left-sided for
                # "incorrectly" (small p-value: likely correct). The +1 keeps p-values above 0.
                pvalue[sample_index,layer_index] = self._empirical_pvalue(reference_scores, statistic[0])
                    
        return pvalue


    def _combine_layer_pvalues(self, pvalues: np.ndarray, y_len: int, test_type: str | None = None) -> np.ndarray:
        """
        Combine p-values across layers into a global p-value using one of the specified methods:
        - Fisher’s method
        - the Cauchy combination test

        If just one layer of p-values is given, the function simply returns the p-values for that layer.?

        Args:
            pvalues (numpy.ndarray): Per-layer p-values
            y_len (int): Number of samples
            test_type (str): Combination method

        Returns:
            numpy.ndarray: Combined p-values per sample
        """
        num_layers = pvalues.shape[1]
            
        if test_type is None and num_layers > 1:
            raise ValueError("test_type_between_layers cannot be None when combining p-values between several layers")
            
        if num_layers == 1:
            return pvalues[:, 0]
        
        if test_type == 'fisher':
            return np.array([
                combine_pvalues(
                    pvalues = pvalues[i, :],
                    method = "fisher"
                    )[1]
                for i in range(y_len)
            ])
        
        if test_type == 'cauchy':
            return np.array([
                self.CauchyCombinationTest(
                    p_values = pvalues[i, :],
                    weights = self.cauchy_weights_per_layer
                    )
                for i in range(y_len)
            ])
        
        raise ValueError(f"Unknown test type: {test_type}")


    def flag(self, statistics, threshold=None) -> float | np.ndarray:
        """
        Flag samples with probability less than or equal (self.s_statistic_source == "correctly") to threshold
        or probability more than or equal (self.s_statistic_source == "incorrectly") to threshold as incorrect.

        Args:
            statistics (numpy.ndarray): Computed p-values
            threshold (float): Threshold for flagging samples

        Returns:
            numpy.ndarray: Boolean array indicating flagged samples. A NaN statistic (a sample
            that could not be scored) is always flagged.
        """
        # Check priority of threshold parameter
        if threshold is None:
            # If not provided as input, try to use self.threshold
            if hasattr(self, 'threshold') and self.threshold is not None:
                threshold = self.threshold
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing threshold parameter: must be provided as input or set as class attribute")

        if self.s_statistic_source == "correctly":
            # If threshold argument to flag() function not none, use this and not the one in config-file
            flags = (statistics <= threshold)

        # Small p-value indicates sample is correctly classified. Make sure flag = 1 means prediction is deemed to be wrong
        elif self.s_statistic_source == "incorrectly":
            flags = statistics > threshold

        # A sample that could not be scored goes to review under either setting
        return flags | np.isnan(statistics)
    
    def remove_nan_values(self, statistics, y_true) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove samples with NaN p-values from the statistics and corresponding true labels.

        Typically use after calling flag() to remove samples with NaN p-values before
        further computing (ex. calculating metrics, plotting, etc.).

        Args:
            statistics (numpy.ndarray): Computed p-values
            y_true (numpy.ndarray): True labels

        Returns:
            tuple: Cleaned statistics and corresponding true labels without NaN values
        """
        flag_nan_idx = np.flatnonzero(np.isnan(statistics))
        y_true_clean = np.delete(y_true, flag_nan_idx)
        statistics_clean = np.delete(statistics, flag_nan_idx)

        return statistics_clean, y_true_clean

    def save_cage(self, path):
        """
        Save the trained cage, including the settings that change how it computes p-values.

        Args:
            path (str or Path): File path to save to (should end in .joblib).

        Raises:
            ValueError: If the cage has not been trained (threshold not set).
        """
        if getattr(self, "threshold", None) is None:
            raise ValueError("Cannot save: cage has not been trained (threshold is not set).")

        parameters = {
            "threshold": self.threshold,
            "layer_params": getattr(self, "layer_params", None),
            "unreliable_classes": self.unreliable_classes,
            "settings": {name: getattr(self, name) for name in SAVED_SETTINGS},
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(parameters, path)

    @classmethod
    def load_cage(cls, path, model_module, data_module, **kwargs):
        """
        Load a trained cage saved with save_cage.

        The cage is rebuilt with the settings it was saved with. Keyword arguments
        override saved settings. Files saved before settings were stored fall back to
        the defaults (with a warning) unless the settings are passed here.

        Args:
            path (str or Path): Path to the saved .joblib file.
            model_module: The model module to use with the loaded cage.
            data_module: The data module to use with the loaded cage.
            **kwargs: Settings that override the saved ones (e.g. test_type_between_layers).

        Returns:
            SPARDACUS: An instance with trained parameters and settings restored.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No saved cage found at {path}")

        parameters = joblib.load(path)

        if "settings" not in parameters:
            warnings.warn(
                f"{path} was saved without its settings (s_statistic_source, pvalue_method, ...). "
                "Defaults are used for any setting not passed to load_cage; re-save the cage to store them.",
                UserWarning
            )
        settings = {**parameters.get("settings", {}), **kwargs}

        instance = cls(model_module=model_module, data_module=data_module, **settings)
        instance.threshold = parameters["threshold"]
        instance.layer_params = parameters.get("layer_params")
        instance.unreliable_classes = set(parameters.get("unreliable_classes") or ())

        return instance

    @staticmethod
    def CauchyCombinationTest(p_values, weights=None):
        """
        Combine p-values using the Cauchy combination test.

        This method combines multiple p-values into a single global p-value.
        If no weights are provided, all p-values are given equal weight.

        Args:
            p_values (numpy.ndarray): P-values to combine.
            weights (numpy.ndarray, optional): Non-negative weights, one per p-value. They are
                rescaled to sum to 1, as the test requires, so e.g. [1, 3] means [0.25, 0.75]. (default: None).

        Returns:
            float: Combined p-value.
        """
        p_values = np.asarray(p_values, dtype=float)

        # If weights is None, put equal weight to each p-value:
        if weights is None or len(weights) == 0:
            weights = np.ones(len(p_values))
        weights = np.asarray(weights, dtype=float)

        if weights.shape != p_values.shape:
            raise ValueError(f"Expected one Cauchy weight per p-value ({len(p_values)}), got {len(weights)}.")
        if np.any(weights < 0) or weights.sum() == 0:
            raise ValueError(f"Cauchy weights must be non-negative and not all zero. Provided: {weights.tolist()}")
        weights = weights / weights.sum()

        # Compute Cauchy statistic:
        C = np.sum(weights*np.tan((0.5-p_values)*np.pi))

        # If p-value are uniformly distributed, C has a standard Cauchy distribution
        # Small p-values indicate discrepancies from H_0, which will give large C.
        # Compute one-sided right-tailed p-value:
        # (survival function: 1 - cdf without losing precision for large C)
        p_value_combined_cauchy = cauchy.sf(C, loc=0, scale=1)

        return(p_value_combined_cauchy)

    @staticmethod
    def fastSPARDA(X_samples, Y_samples, **kwargs):
        """
        Find a projection that separates two sample sets using the fastSPARDA method.

        This method searches for a projection vector that maximizes the projected
        Wasserstein distance between X_samples and Y_samples.

        Args:
            X_samples (numpy.ndarray): First set of samples.
            Y_samples (numpy.ndarray): Second set of samples.
            **kwargs: Optional parameters controlling optimization and cross-validation:
                lambdas (list[float]): l1 penalties on the projection. One value is used as is;
                    several are chosen between by cross-validation (see _cross_validate_lambda).
                    (default: [0], no penalty, as in the SPARDACUS paper)
                num_folds (int): Number of cross-validation folds when several lambdas are given. (default: 5)
                max_iter (int): Maximum number of hill-climbing iterations. (default: 1000)
                eps (float): Minimum improvement required to continue. (default: 1e-8)
                learning_rate (float): Base learning rate. (default: 1)
                print_update (int): Print progress every print_update iterations; 0 disables printing. (default: 0)

        Returns:
            tuple: Projection vector, Wasserstein distance, optimization cost, and best lambda value.
        """
        #This method directly solves the original nonconvex formulation using subgradient
        # hill-climbing (with l1 penalty on the projection vector).
        # Thus it is more efficient, but results may heavily depend on
        # initialization if the underlying distributions induce nonconvexity in
        # the SPARDA objective function.

        # Parse input arguments
        lambdas_default = [0]
        num_folds_default = 5
        max_iter_default = 1000
        eps_default = 1e-8
        learning_rate_default = 1
        print_update_default = 0  # progress printing is opt-in: pass print_update=N to print every N iterations

        lambdas = kwargs.get('lambdas', lambdas_default)
        num_folds = kwargs.get('num_folds', num_folds_default)
        max_iter = kwargs.get('max_iter', max_iter_default)
        eps = kwargs.get('eps', eps_default)
        learning_rate = kwargs.get('learning_rate', learning_rate_default)
        print_update = kwargs.get('print_update', print_update_default)

        if len(lambdas) == 0:
            raise ValueError('lambdas not correctly formatted')

        if X_samples.shape[0] < Y_samples.shape[0]:#ensure n >= m with dim(X)=n,dim(y)=m by swapping
            X_samples, Y_samples = Y_samples, X_samples

        # Starting direction: the best of random directions and the coordinate axes
        prev_beta, _, _ = SPARDACUS.randomProjectionSearch(X_samples, Y_samples, max_iter=max(100, int(np.ceil(max_iter / 10))))

        if len(lambdas) == 1:
            best_lambda = lambdas[0]
        else:
            best_lambda, prev_beta = SPARDACUS._cross_validate_lambda(
                X_samples, Y_samples, lambdas, num_folds, prev_beta, max_iter, eps, learning_rate, print_update
            )

        beta_hat, cost = SPARDACUS.l1SPARDA(X_samples, Y_samples, best_lambda, max_iter, eps, learning_rate, print_update, prev_beta)
        beta_norm = np.linalg.norm(beta_hat)
        if beta_norm == 0:
            raise ValueError(f"lambda={best_lambda} shrank the projection to zero; use smaller lambdas.")
        beta_hat = beta_hat/beta_norm # always re-scale to unit norm.
        wass_dist = SPARDACUS.projectedWasserstein(X_samples, Y_samples, beta_hat)

        return beta_hat, wass_dist, cost, best_lambda

    @staticmethod
    def _cross_validate_lambda(X_samples, Y_samples, lambdas, num_folds, start_beta, max_iter, eps, learning_rate, print_update):
        """
        Choose the l1 penalty lambda for fastSPARDA by K-fold cross-validation.

        Both sample sets are split into num_folds shuffled folds. For each fold, the projection
        is fitted on the other folds for every lambda in increasing order, each fit warm-started
        from the previous one, and scored by the projected Wasserstein distance on the held-out
        fold. The score uses the unit-length direction, so larger penalties (which also shrink
        beta) are judged on the direction they find, not marked down for being shorter. Once a
        projection uses fewer than two features, larger lambdas are skipped for that fold, as
        they can only be sparser. The lambda with the highest average held-out distance wins.

        Args:
            X_samples (numpy.ndarray): Larger set of samples.
            Y_samples (numpy.ndarray): Smaller set of samples.
            lambdas (list[float]): Candidate l1 penalties.
            num_folds (int): Number of folds.
            start_beta (numpy.ndarray): Starting direction.
            max_iter, eps, learning_rate, print_update: Passed on to l1SPARDA.

        Returns:
            tuple: (best lambda, unpenalised full-data direction to warm-start the final fit)
        """
        lambdas = sorted(lambdas)
        if len(Y_samples) < num_folds:
            raise ValueError(f"Cross-validating lambdas needs at least num_folds={num_folds} samples in each set; the smaller set has {len(Y_samples)}.")

        # Unpenalised solution on all samples: warm start for every fold and for the final fit
        first_beta, _ = SPARDACUS.l1SPARDA(X_samples, Y_samples, 0, max_iter, eps, learning_rate, print_update, start_beta)

        x_folds = KFold(n_splits=num_folds, shuffle=True).split(X_samples)
        y_folds = KFold(n_splits=num_folds, shuffle=True).split(Y_samples)
        lambda_scores = np.zeros(len(lambdas))

        for fold, ((x_train, x_test), (y_train, y_test)) in enumerate(zip(x_folds, y_folds), start=1):
            prev_beta = first_beta
            for index, lambda_val in enumerate(lambdas):
                beta, cost = SPARDACUS.l1SPARDA(X_samples[x_train], Y_samples[y_train], lambda_val, max_iter, eps, learning_rate, print_update, prev_beta)
                beta_norm = np.linalg.norm(beta)
                if beta_norm == 0:
                    break
                prev_beta = beta / beta_norm

                heldout_wass = SPARDACUS.projectedWasserstein(X_samples[x_test], Y_samples[y_test], prev_beta)
                lambda_scores[index] += heldout_wass
                cardinality = np.sum(np.abs(beta) > 0)

                if print_update > 0:
                    print(f'lambda: {lambda_val}  fold: {fold}')
                    print(f'training cost: {cost}')
                    print(f'heldout cost: {heldout_wass}')
                    print(f'projection cardinality: {cardinality}')

                if cardinality < 2:
                    break

        lambda_scores /= num_folds
        return lambdas[int(np.argmax(lambda_scores))], first_beta

    @staticmethod
    def l1SPARDA(xs, ys, _lambda, max_iter, eps, learning_rate, print_update, beta0=None):
        """
        Optimize the SPARDA objective with an l1 penalty.

        This method updates the projection vector using subgradient hill-climbing
        and optional soft-thresholding for sparsity.

        Args:
            xs (numpy.ndarray): First set of samples.
            ys (numpy.ndarray): Second set of samples.
            _lambda (float): l1 regularization strength.
            max_iter (int): Maximum number of iterations.
            eps (float): Minimum improvement required to continue.
            learning_rate (float): Base learning rate.
            print_update (int): Print progress every print_update iterations; 0 disables printing.
            beta0 (numpy.ndarray, optional): Initial projection vector. (default: None).

        Returns:
            tuple: Optimized projection vector and final objective value.
        """

        n, d = xs.shape
        m = ys.shape[0]
        iter_val = 0

        if beta0 is None:
            beta0 = np.sqrt(d) / d / 2 * np.ones(d)

        # Copy so the in-place updates below never touch the caller's array
        beta = np.array(beta0, dtype=float)
        last_cost = -np.inf
        cost = SPARDACUS.projectedWasserstein(xs, ys, beta) - _lambda * np.sum(np.abs(beta))
        grad = np.zeros(d)

        while iter_val < max_iter and cost - last_cost >= eps:
            last_cost = cost
            last_beta = beta.copy()
            iter_val += 1
            step_size = learning_rate / np.sqrt(iter_val)

            if print_update > 0 and iter_val % print_update == 0:
                print(f'iter: {iter_val}   cost: {cost}  grad-norm: {np.linalg.norm(grad)}    beta_norm: {np.linalg.norm(beta)}  Step-size: {step_size}')

            # Compute gradient:
            projected_xs, x_order = np.sort(xs @ beta), np.argsort(xs @ beta)
            projected_ys, y_order = np.sort(ys @ beta), np.argsort(ys @ beta)
            grad = np.zeros(d)
            quant_x = 0
            quant_y = 0
            last_quant = 0
            NUMERIC_FACTOR = 1e-6
            delta = NUMERIC_FACTOR / (n * m)

            x_index = 0
            y_index = 0

            while quant_x < 1 - delta or quant_y < 1 - delta:
                next_quant_x = quant_x + 1 / n
                next_quant_y = quant_y + 1 / m
                proj_x = projected_xs[x_index]
                proj_y = projected_ys[y_index]

                while next_quant_x < next_quant_y - delta:
                    grad += 2 * (proj_x - proj_y) * (xs[x_order[x_index], :] - ys[y_order[y_index], :]).T * (next_quant_x - last_quant)
                    quant_x = next_quant_x
                    last_quant = quant_x
                    next_quant_x = quant_x + 1 / n
                    x_index = min(x_index + 1, n - 1)
                    proj_x = projected_xs[x_index]

                if quant_x < 1 - delta or quant_y < 1 - delta:
                    if abs(next_quant_x - next_quant_y) < delta:
                        grad += 2 * (proj_x - proj_y) * (xs[x_order[x_index], :] - ys[y_order[y_index], :]).T * (next_quant_x - last_quant)
                        quant_x = next_quant_x
                        quant_y = next_quant_y
                        x_index = min(x_index + 1, n - 1)
                        y_index = min(y_index + 1, m - 1)
                        #x_index = x_index + 1
                        #y_index = y_index + 1
                        last_quant = quant_x
                    else:
                        grad += 2 * (proj_x - proj_y) * (xs[x_order[x_index], :] - ys[y_order[y_index], :]).T * (next_quant_y - last_quant)
                        quant_y = next_quant_y
                        y_index = min(y_index + 1, m - 1)
                        last_quant = quant_y

            MAX_BACKTRACKING = 10
            backtrack_tries = 0

            while cost - last_cost < eps and backtrack_tries <= MAX_BACKTRACKING:
                step_size = learning_rate / np.log(iter_val + 1) * 2 ** (-backtrack_tries)
                # Each backtracking try restarts from last_beta with a smaller step
                beta = last_beta + step_size * grad

                if beta[0] < 0:
                    beta = -beta

                if _lambda > 0:
                    beta = np.sign(beta) * np.maximum(np.abs(beta) - _lambda * step_size, 0)

                beta_norm = np.linalg.norm(beta)

                if beta_norm > 1:
                    beta /= beta_norm

                cost = SPARDACUS.projectedWasserstein(xs, ys, beta) - _lambda * np.sum(np.abs(beta))
                backtrack_tries += 1

            if cost < last_cost:
                cost = last_cost
                beta = last_beta
                break

        if iter_val >= max_iter:
            warnings.warn(
                f"fastSPARDA optimization stopped after max_iter={max_iter} iterations without converging "
                "(likely max_iter or learning_rate is too small); the projection may be slightly suboptimal.",
                UserWarning
            )

        return beta, cost

    @staticmethod
    def randomProjectionSearch(X_samples, Y_samples, **kwargs):
        """
        Search for a good projection using random directions.

        This method evaluates random projection vectors and keeps the one that
        maximizes the projected Wasserstein distance.

        Args:
            X_samples (numpy.ndarray): First set of samples.
            Y_samples (numpy.ndarray): Second set of samples.
            **kwargs: Optional parameters controlling the search.

        Returns:
            tuple: Best projection vector, best Wasserstein distance, and parameter settings.
        """
        # Simple random search to find projection BETA which maximizes projected Wasserstein distance.
        # Optional arguments: max_iter = the number of random projections to try.

        if X_samples.shape[1] != Y_samples.shape[1]:
            raise ValueError('X and Y must have the same dimension')

        # Parse input arguments
        max_iter_default = 100
        max_iter = kwargs.get('max_iter', max_iter_default)

        best_dist = -np.inf
        best_beta = np.nan

        for i in range(1, max_iter + 1):
            # standard_normal always returns an array, even for a single feature
            # (multivariate_normal.rvs returned a plain float then)
            beta = np.random.standard_normal(X_samples.shape[1])
            beta /= np.linalg.norm(beta)

            if beta[0] < 0:
                beta = -beta

            d_beta = SPARDACUS.projectedWasserstein(X_samples, Y_samples, beta)

            if d_beta > best_dist:
                best_dist = d_beta
                best_beta = beta

        # Also check basis directions:
        for l in range(X_samples.shape[1]):
            beta = np.zeros(X_samples.shape[1])
            beta[l] = 1
            d_beta = SPARDACUS.projectedWasserstein(X_samples, Y_samples, beta)

            if d_beta > best_dist:
                best_dist = d_beta
                best_beta = beta

        param_settings = {'max_iter': max_iter}

        return best_beta, best_dist, param_settings

    @staticmethod
    def projectedWasserstein(X_samples, Y_samples, beta):
        """
        Compute the projected squared Wasserstein distance between two sample sets.

        This method projects both sample sets onto the direction beta and computes
        the empirical squared Wasserstein distance between the projected values.

        Args:
            X_samples (numpy.ndarray): First set of samples.
            Y_samples (numpy.ndarray): Second set of samples.
            beta (numpy.ndarray): Projection vector.

        Returns:
            float: Projected squared Wasserstein distance.
        """
        # Projected SQUARED empirical Wasserstein distance in direction beta
        # between X_samples and Y_samples.

        projected_xs = np.sort(X_samples.dot(beta))
        projected_ys = np.sort(Y_samples.dot(beta))

        n = len(projected_xs)
        m = len(projected_ys)

        # Both empirical quantile functions are step functions. On the integer
        # grid t = quantile * n * m, x steps every m units and y every n units,
        # so merging the two sets of step points gives every interval where
        # both are constant. Integers avoid the drift that summing 1/n and 1/m
        # step by step accumulates, which could run past the end of the data.
        breaks = np.union1d(np.arange(n + 1) * m, np.arange(m + 1) * n)
        starts = breaks[:-1]
        widths = np.diff(breaks)

        diffs = projected_xs[starts // m] - projected_ys[starts // n]

        return np.sum(diffs ** 2 * widths) / (n * m)

