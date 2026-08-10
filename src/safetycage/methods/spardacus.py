from tqdm import tqdm
import numpy as np
from scipy.stats import cauchy, combine_pvalues, multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

from ..safetycage import SafetyCage

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

    The smaller the p-value, the more likely the sample is to be misclassified. When 
    multiple layers are used, a global p-value is found by combining the each layer's 
    p-value using either Fisher’s method or the Cauchy combination test. The optimal 
    threshold to compare the resulting p-values is given to the alpha attribute.

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
        alpha (float | None): Significance threshold used for flagging.
        cauchy_weights_per_layer (list): Weights used for Cauchy combination test.
        test_type_between_layers (str): Method for combining p-values across layers.
        minimum_sample_size (int): Minimum number of samples required to fit
            Gaussian Mixture Models.
    """
    def __init__(self, model_module, data_module, **kwargs):
        """
        Initialize the SPARDACUS safety cage.

        Args:
            model_module: Reference to model module object for making predictions.
            data_module: Reference to data module object for handling data.
            s_statistic_source (str): Source used to compute p-values ("correctly" or "incorrectly").
            alpha (float): Significance threshold for flagging samples.
            test_type_between_layers (str): Method for combining p-values ("fisher" or "cauchy").
            cauchy_weights_per_layer (list[float]): Weights for the Cauchy combination test.
            minimum_sample_size (int, optional): Minimum number of samples required to fit density models. (default: 10)
        """
        super(SPARDACUS, self).__init__(model_module, data_module, **kwargs)

        self.s_statistic_source = kwargs.get("s_statistic_source")
        self.alpha = kwargs.get("alpha", None)
        self.cauchy_weights_per_layer = kwargs.get("cauchy_weights_per_layer")
        self.test_type_between_layers = kwargs.get("test_type_between_layers")

        # For the Gaussian Mixture Model fitting. Must be at least 3 based on current implementation (see _fit_gaussian_mixture). Default value is 10.
        if "minimum_sample_size" in kwargs and kwargs["minimum_sample_size"] < 3:
            raise ValueError(f"Minimum_sample_size must be at least 3. Provided: {kwargs['minimum_sample_size']}")
        self.minimum_sample_size = kwargs.get("minimum_sample_size", 10)

        self.classes = data_module.classes
        self.unreliable_classes = set()
    
    @property
    def name(self):
        """Return the name of the safety cage method."""
        return "SPARDACUS"

    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        """
        Train the SPARDACUS safety cage.

        Separates training samples into correctly and incorrectly classified groups and
        computes parameters for each layer and class, which are later used to evaluate 
        new samples during prediction.

        Args:
            x: Tuple of (x_correct, x_incorrect) input data
            y: Tuple of (y_correct, y_incorrect) labels
            y_pred: Model predictions
        """
        
        if x is None:
            x, y = self.data_module.data_train
        if y is None:
            _, y = self.data_module.data_train
        if y_pred is None:
            y_pred = self.model_module._get_predictions(x)

        if self.model_module.use_onehot_encoder:
            mask = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
        else:
            mask = y_pred == y
        
        if isinstance(x, dict):
            x_correct = {key: val[mask] for key, val in x.items()}
            x_incorrect = {key: val[~mask] for key, val in x.items()}
        else:
            x_correct = x[mask]
            x_incorrect = x[~mask]
            
        y_correct = y[mask]
        y_incorrect = y[~mask]
        
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
                    layers_activations["correct"], layer, y_correct, class_key
                )
                class_activations_incorrect = self._get_class_activations(
                    layers_activations["incorrect"], layer, y_incorrect, class_key
                )
                
                # Process layer and class
                self.layer_params[layer][class_label] = self._process_layer_class(
                    class_activations_correct, class_activations_incorrect, class_label
                )


    def _get_class_activations(self, layer_activations: dict, layer: str, 
                            y_data: np.ndarray, class_index: int) -> np.ndarray:
        """
        Extract activations for a specific class (given by class_index) at a given layer.
        Supports both one-hot encoded and integer labels.

        Returns:
            numpy.ndarray: Activations for the specified class and layer
        """
        if self.model_module.use_onehot_encoder:
            return layer_activations[layer][y_data[:, class_index] == 1, :]
        
        return layer_activations[layer][y_data == class_index, :]


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
            dict: Dictionary containing projection vectors, density models, and ECDFs for the class and layer
        """
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
                "density_incorrect": None
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
                "density_incorrect": None
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
            "density_incorrect": density_incorrect
        }


    def _fit_gaussian_mixture(self, samples: np.ndarray, correctness: str, class_label: str) -> GaussianMixture:
        """
        Notes:
        - Throws an error if there are not enough samples to fit the model, but catches this error and throws a warning instead, returning the best estimator found by GridSearchCV even if it was not properly fitted.

        Fit a Gaussian Mixture Model to the given samples using GridSearchCV for hyperparameter tuning. 
        Use cross-validation with 2-fold CV to select the number of components (1-3) and takes the BIC
        average score of all folds to select the best model. 
        
        **WARNING:** In cases where there are too few samples (often by too few incorrect samples for 
        well-performing models), the Gaussian Mixture Model may fail to fit properly (since it does not 
        make sense to fit a gaussian to too few samples). In such cases, we recommend using a different 
        safetycage method or ensuring enough incorrect and incorrect samples exist for each class.

        For most warning cases the error/warning will be handled below, and a clear warning statement is 
        sent to output. In cases where there are many warnings, it is often a result of sklearn's 
        GridSearchCV struggling to fit.

        Consider the following warning cases:
            - The provided number of samples is less than or minimum_sample_size (which itself has a minimum
            value of 3 since we consider at most 3 components).
            - All Gaussian fits fail (all 6, since there are 3 components to try and 2-fold CV). This is 
            captured as an error and throws a warning. Gaussian fitting does not occur, the given class
            is saved to self.unreliable_classes, and the method returns None.
            - Not all fits failed, but all BIC scores are NaN values (meaning for all component options, 
            at least 1 NaN value occured in the 2-fold CV). The given class is saved to self.unreliable_classes.
            It returns the best estimator found by GridSearchCV, but the found parameter is simply the first one
            available.
            - Not all fits failed, but some BIC scores are NaN values. A warning is thrown that model selection 
            may be unreliable, but the best estimator found by GridSearchCV is returned.

        Args:
            samples (numpy.ndarray): Projected samples
            correctness (str): whethering we are fitting predictions that were "correct" or "incorrect"
            class_label (str): Class label

        Returns:
            GaussianMixture | None: Fitted model or None if fitting failed
        """

        if len(samples) < self.minimum_sample_size:
            warnings.warn(
                f"[Gaussian Mixture Model WARNING] There are not enough {correctness} samples in the \"{class_label}\" class to fit a Gaussian Mixture Model. "
                f"Provided: {len(samples)}. "
                f"Minimum required: {self.minimum_sample_size}. "
                f"We recommend that you use a different safetycage method.",
                UserWarning
        )
        
        param_grid = {
            "n_components": range(1, 4),
            "covariance_type": ["full"],
        }
        
        grid_search = GridSearchCV(
            estimator=GaussianMixture(),
            param_grid=param_grid,
            scoring=self.gmm_bic_score,
            cv=2
        )
        
        # Try Catch to catch ValueError thrown by sklearn when all fits fail, ignore sklearn warning when some fits fail
        try:
            # If fits fail, sklearn throws a long FitFailedWarning, catch this and let the warning message be provided by the if statement below.
            with warnings.catch_warnings():
                from sklearn.exceptions import FitFailedWarning
                warnings.filterwarnings("ignore", category=FitFailedWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="One or more of the test scores are non-finite:.*",
                    category=UserWarning,
                    module="sklearn.model_selection._search"
                )
                grid_search.fit(samples.reshape(-1, 1))

        except ValueError as e:
            warnings.warn(
                f"[Gaussian Mixture Model WARNING] All GMM fits failed for "
                f"{correctness} samples in class \"{class_label}\". "
                f"This class will be flagged as unreliable.",
                UserWarning
            )
            self.unreliable_classes.add(class_label)
            return None

        # Address the cases where all/some GMM fits fail during CV.
        scores = grid_search.cv_results_["mean_test_score"]

        if np.isnan(scores).any():
            warnings.warn(f"[Gaussian Mixture Model WARNING] There are not enough {correctness} samples in the \"{class_label}\" class to fit a Gaussian Mixture Model.")

            if np.all(np.isnan(scores)):
                # If all scores are NaN .best_estimator_ automatically chooses the first parameter (n_components=1, covariance_type='full').
                warnings.warn(f"All mean BIC scores are NaN values. Hence, model selection is invalid. The results for the \"{class_label}\" class are unreliable and this class will be flagged as unreliable.")
                self.unreliable_classes.add(class_label)
            else:
                warnings.warn(f"{np.sum(np.isnan(scores))} BIC score(s) are NaN values. Model selection for the \"{class_label}\" class may be unreliable.")
        
        return grid_search.best_estimator_


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
        Samples belonging to unreliable classes are assigned NaN values.

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

        if self.unreliable_classes:
            warnings.warn(f"The p-value for the classes {self.unreliable_classes} have been set to NaN due to insufficient data to fit the Gaussian Mixture Models. "
                        f"Consider using a different safetycage method.")
        
        for layer_index, layer in enumerate(selected_layers): # for all layers
            for sample_index, y_sample, in enumerate(y): # for all predictions to be tested
                
                # Compute p-values of each sample per layer using ECDF function
                if self.model_module.use_onehot_encoder:
                    class_label = self.classes[np.argmax(y_sample)]
                else:
                    class_label = self.classes[y_sample]
                
                if class_label in self.unreliable_classes:
                    pvalue[sample_index,layer_index] = np.nan
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
                
                if self.s_statistic_source == "correctly": 
                    # Right-sided test. Small p-value indicates sample is incorrectly classfied                           
                    pvalue[sample_index,layer_index] = 1 - ecdf_correct(statistic)
                    
                elif self.s_statistic_source == "incorrectly":
                    # Left-sided test. Small p-value indicates sample is correctly classified.
                    pvalue[sample_index,layer_index] = ecdf_incorrect(statistic)
                    
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


    def flag(self, statistics, alpha=None) -> float | np.ndarray:
        """
        Flag samples with probability less than or equal (self.s_statistic_source == "correctly") to alpha 
        or probability more than or equal (self.s_statistic_source == "incorrectly") to alpha as incorrect.

        Args:
            statistics (numpy.ndarray): Computed p-values
            alpha (float): Threshold for flagging samples

        Returns:
            numpy.ndarray: Boolean array indicating flagged samples with NaNs preserved for missing statistics.
        """
        # Check priority of alpha parameter
        if alpha is None:
            # If not provided as input, try to use self.alpha
            if hasattr(self, 'alpha') and self.alpha is not None:
                alpha = self.alpha
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing alpha parameter: must be provided as input or set as class attribute")
            
        if self.s_statistic_source == "correctly":
            # If alpha argument to flag() function not none, use this and not the one in config-file
            flags = (statistics <= alpha)
            
        # Small p-value indicates sample is correctly classified. Make sure flag = 1 means prediction is deemed to be wrong
        elif self.s_statistic_source == "incorrectly":
            flags = ~(statistics <= alpha)

        return flags
    
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

    @staticmethod
    def CauchyCombinationTest(p_values, weights=None):
        """
        Combine p-values using the Cauchy combination test.

        This method combines multiple p-values into a single global p-value.
        If no weights are provided, all p-values are given equal weight.

        Args:
            p_values (numpy.ndarray): P-values to combine.
            weights (numpy.ndarray, optional): Weights for each p-value. (default: None).

        Returns:
            float: Combined p-value.
        """

        # If weights is None, put equal weight to each p-value:
        if weights is None or weights == []:
            weights = np.ones(len(p_values))/len(p_values)

        # Compute Cauchy statistic:
        C = np.sum(weights*np.tan((0.5-p_values)*np.pi))

        # If p-value are uniformly distributed, C has a standard Cauchy distribution
        # Small p-values indicate discrepancies from H_0, which will give large C.
        # Compute one-sided right-tailed p-value:
        p_value_combined_cauchy = 1 - cauchy.cdf(C, loc=0, scale=1)

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
            **kwargs: Optional parameters controlling optimization and cross-validation.

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
        print_update_default = 100

        lambdas = kwargs.get('lambdas', lambdas_default)
        num_folds = kwargs.get('num_folds', num_folds_default)
        max_iter = kwargs.get('max_iter', max_iter_default)
        eps = kwargs.get('eps', eps_default)
        learning_rate = kwargs.get('learning_rate', learning_rate_default)
        print_update = kwargs.get('print_update', print_update_default)

        if X_samples.shape[0] < Y_samples.shape[0]:#ensure n >= m with dim(X)=n,dim(y)=m by swapping
            X_samples, Y_samples = Y_samples, X_samples

        n, d = X_samples.reshape(-1), X_samples.shape[1]
        m = Y_samples.shape[0]

        x_foldsize = n // num_folds
        y_foldsize = m // num_folds
        prev_beta = np.zeros(d)
        prev_beta[0] = 0.5

        if len(lambdas) > 1:
            lambdas = sorted(lambdas)
            lambda_scores = np.zeros(len(lambdas))
            first_beta, first_cost = SPARDACUS.l1SPARDA(X_samples, Y_samples, 0, max_iter, eps, learning_rate, print_update, prev_beta)

            for fold in range(1, num_folds + 1):

                x_foldindex = (fold-1) * x_foldsize + 1
                y_foldindex = (fold-1) * y_foldsize + 1
                if fold < num_folds:
                    xs_test = X_samples[x_foldindex:(x_foldindex+x_foldsize-1),:]
                    xs_train = X_samples[1:(x_foldindex-1) (x_foldindex+x_foldsize):n]
                    ys_test = Y_samples[y_foldindex:(y_foldindex+y_foldsize-1),:]
                    ys_train = Y_samples[1:(y_foldindex-1) (y_foldindex+y_foldsize):m]
                else:
                    xs_test = X_samples[x_foldindex:n,:]
                    xs_train = X_samples[1:(x_foldindex-1),:]
                    ys_test = Y_samples[y_foldindex:m,:]
                    ys_train = Y_samples[1:(y_foldindex-1),:]



                prev_beta = first_beta
                for l in range(len(lambdas)):
                    lambda_val = lambdas[l]
                    beta, cost = SPARDACUS.l1SPARDA(xs_train, ys_train, lambda_val, max_iter, eps, learning_rate, print_update, prev_beta)

                    if np.linalg.norm(beta) > 0:
                        prev_beta = beta / np.linalg.norm(beta)

                    heldout_wass = SPARDACUS.projectedWasserstein(xs_test, ys_test, beta)
                    lambda_scores[l] += heldout_wass

                    cardinality = np.sum(np.abs(beta) > 0)

                    if print_update < np.inf:
                        print(f'lambda: {lambda_val}  fold: {fold}')
                        print(f'training cost: {cost}')
                        print(f'heldout cost: {heldout_wass}')
                        print(f'projection cardinality: {cardinality}')

                    if cardinality < 2:
                        break

            lambda_scores /= num_folds
            best_indices = np.where(lambda_scores == np.max(lambda_scores))[0]
            best_lambda = lambdas[best_indices[0]]
        elif len(lambdas) == 1:
            best_lambda = lambdas[0]
            prev_beta, _, _ = SPARDACUS.randomProjectionSearch(X_samples, Y_samples, max_iter=max(100, int(np.ceil(max_iter / 10))))
        else:
            raise ValueError('lambdas not correctly formatted')

        beta_hat, cost = SPARDACUS.l1SPARDA(X_samples, Y_samples, best_lambda, max_iter, eps, learning_rate, print_update, prev_beta)
        beta_hat = beta_hat/np.linalg.norm(beta_hat) # always re-scale to unit norm.
        wass_dist = SPARDACUS.projectedWasserstein(X_samples, Y_samples, beta_hat)

        return beta_hat, wass_dist, cost, best_lambda

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
            print_update (int): Frequency of progress updates.
            beta0 (numpy.ndarray, optional): Initial projection vector. (default: None).

        Returns:
            tuple: Optimized projection vector and final objective value.
        """

        n, d = xs.shape
        m = ys.shape[0]
        iter_val = 0

        if beta0 is None:
            beta0 = np.sqrt(d) / d / 2 * np.ones(d)

        beta = beta0
        last_cost = -np.inf
        cost = SPARDACUS.projectedWasserstein(xs, ys, beta) - _lambda * np.sum(np.abs(beta))
        grad = np.zeros(d)

        while iter_val < max_iter and cost - last_cost >= eps:
            last_cost = cost
            last_beta = beta
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
                beta += step_size * grad

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
            print('fastSPARDA optimization failed to converge (likely max_iter or learning_rate is too small)')

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
            beta = multivariate_normal.rvs(mean=np.zeros(X_samples.shape[1]), cov=np.eye(X_samples.shape[1]))
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

        NUMERIC_FACTOR = 1e6  # used in ceil function to get the index of the quantile in the projected data vector.

        projected_xs = np.sort(X_samples.dot(beta))
        projected_ys = np.sort(Y_samples.dot(beta))

        if len(projected_xs) < len(projected_ys):
            projected_xs, projected_ys = projected_ys, projected_xs

        n = len(projected_xs)
        m = len(projected_ys)
        eps = 1 / (n * m * NUMERIC_FACTOR)
        dist = 0
        quant_x = 0
        quant_y = 0
        last_quant = 0
        x_index = 0
        y_index = 0
        while quant_x < 1 - eps or quant_y < 1 - eps:
            next_quant_x = quant_x + 1 / n
            next_quant_y = quant_y + 1 / m
            proj_x = projected_xs[x_index] if x_index < n else float('inf')
            proj_y = projected_ys[y_index] if y_index < m else float('inf')

            while next_quant_x < next_quant_y - eps:
                dist += (proj_x - proj_y) ** 2 * (next_quant_x - last_quant)
                quant_x = next_quant_x
                last_quant = quant_x
                next_quant_x = quant_x + 1 / n
                x_index = min(x_index + 1, n)
                proj_x = projected_xs[x_index] if x_index < n else float('inf')

            if quant_x < 1 - eps or quant_y < 1 - eps:
                if abs(next_quant_x - next_quant_y) < eps:
                    dist += (proj_x - proj_y) ** 2 * (next_quant_x - last_quant)
                    quant_x = next_quant_x
                    quant_y = next_quant_y
                    x_index += 1
                    y_index += 1
                    last_quant = quant_x
                else:
                    dist += (proj_x - proj_y) ** 2 * (next_quant_y - last_quant)
                    quant_y = next_quant_y
                    y_index = min(y_index + 1, m)
                    last_quant = quant_y

        return dist

    @staticmethod
    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

if __name__ == "__main__":
    SPARDACUS(None, None, None)