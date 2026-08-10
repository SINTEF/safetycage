import os
import joblib
import numpy as np
from numpy import linalg
from scipy.stats import cauchy, chi2, norm, f, combine_pvalues
from statsmodels.distributions.empirical_distribution import ECDF

from ..safetycage import SafetyCage

class Mahalanobis(SafetyCage):
    """
    Mahalanobis-based Safety Cage Method.

    The Mahalanobis safety cage detects misclassified samples by measuring how far 
    a sample’s internal neural network activations differ from the distribution of 
    correctly classified training samples.
    
    The method models the distribution of layer-wise activations for each class
    using a multivariate Gaussian approximation. During prediction after training,
    it computes a Mahalanobis distance between a sample’s activation and
    the corresponding class distribution. This distance is then converted into
    a p-value using either:

    - Asymptotic χ² approximation (chi-squared test),
    - Hotelling’s T² distribution (finite-sample correction),
    - Empirical distribution of distances from training data.

    The smaller the p-value, the more likely the sample is to be misclassified or 
    out-of-distribution. When multiple layers are used, a global p-value is found by 
    combining the per-layer p-valyes using either Fisher’s method or the Cauchy 
    combination test. The optimal threshold to compare the resulting p-values is given 
    to the alpha attribute.

    See the below research paper for a thorough explanation of the Mahalanobis method.

    **Reference:**
        P. V. Johnsen and F. Remonato. “SafetyCage: A misclassification detector for 
        feed-forward neural networks”. https://proceedings.mlr.press/v233/johnsen24a.html.

    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
        selected_layers (list): List of model layers used for feature extraction.
            Retrieved from the model module.
        last_layer (str): Name of the final layer.
            Retrieved from the model module.
        classes (dict): Mapping of class indices to class labels.
            Retrieved from the data module.
        layer_params (dict): Stores per-layer, per-class distribution parameters
            (mean, covariance, ECDF, etc.).
        leq (bool): Indicates comparison direction for downstream thresholding
            (smaller p-values indicate higher uncertainty).

        empirical (bool): Whether to use empirical distributions instead of
            parametric Gaussian assumptions.
        use_preactivations (bool): Whether to use pre-activation values.
        test_type_within_layer (str): Statistic used per layer
            ("chi2", "t2", or "mahalanobis").
        test_type_between_layers (str): Method for combining p-values across layers
            ("fisher" or "cauchy").
        cauchy_weights_per_layer (list): Weights for Cauchy combination test.
    """
    def __init__(self, model_module, data_module,**kwargs):
        """
        Initialize the Mahalanobis safety cage.

        Args:
            model_module: Reference to model module object for making predictions.
            data_module: Reference to data module object for handling data.
            empirical (bool): If True, p-values are computed  using an empirical distribution
                of Mahalanobis distances. If False, parametric distributions (χ² or Hotelling’s T²) are used.
            use_preactivations (bool): Whether to use pre-activation values.
            test_type_within_layer (str): Method used to compute p-values per layer. 
                Must be one of the following:
                    - "chi2": asymptotic chi-squared approximation
                    - "t2": Hotelling’s T² (finite-sample correction)
                    - "mahalanobis": empirical distribution
            test_type_between_layers (str): Method used to combine p-values across layers. Must be one of:
                    - "fisher"
                    - "cauchy"
            cauchy_weights_per_layer (list[float]): Weights used when combining p-values with the Cauchy method.
        """
        super(Mahalanobis, self).__init__(model_module, data_module, **kwargs)
        
        self.leq = True

        self.empirical = kwargs.get("empirical")
        self.use_preactivations = kwargs.get("use_preactivations")
        self.cauchy_weights_per_layer = kwargs.get("cauchy_weights_per_layer")
        self.test_type_between_layers = kwargs.get("test_type_between_layers")
        self.test_type_within_layer = kwargs.get("test_type_within_layer")
        
        self.selected_layers = self.model_module.selected_layers
        self.last_layer = self.model_module.last_layer
        self.classes = data_module.classes
            
        self.test_type_fn_dict = {
            "chi2": self.chi2_statistic,
            "t2": self.t2_statistic,
            "mahalanobis": self.mahalanobis_statistic
        }
        
    @property
    def name(self):
        """Return the name of the safety cage method."""
        return "mahalanobis"

        
    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        """
        For each layer and class, train_cage estimates the mean and covariance of activations,
        and optionally constructs an empirical distribution of Mahalanobis distances.

        Args:
            x: Input data. If None, loaded from the data module.
            y: Ground-truth labels. If None, loaded from the data module.
            y_pred: Model predictions. If None, computed using the model module's _get_predictions method.
        """
        if x is None:
            x, y = self.data_module.data_train
        if y is None:
            _, y = self.data_module.data_train
        if y_pred is None:
            y_pred = self.model_module._get_predictions(x)
        
        # Mahalanobis distance is used to compute the p-value

        if self.model_module.use_onehot_encoder:
            mask = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
        else:
            mask = y_pred == y

        x_correct = x[mask]
        y_correct = y[mask]


        layer_activations =  self.model_module._get_pre_activations(x_correct)
        
        self.layer_params = {
            layer: {class_index: {} for class_index in self.classes}
            for layer in self.selected_layers
        }
        
        # Process each layer and class
        for layer in self.selected_layers:
            for class_key, class_label in self.classes.items():
                    
                if self.model_module.use_onehot_encoder:
                    num_observations = np.sum(y_correct[:, class_key] == 1)
                else:
                    num_observations = np.sum(y_correct == class_key)
                    
                class_activations = self._get_class_activations(
                    layer_activations = layer_activations,
                    layer = layer,
                    y_data = y_correct,
                    class_index = class_key
                    )
                
                self.layer_params[layer][class_label] = self.compute_empirical_distribution(
                    class_activations = class_activations,
                    empirical = self.empirical,
                    num_observations = num_observations
                    )


    def _get_class_activations(self, layer_activations: dict, layer: str, 
                            y_data: np.ndarray, class_index: int) -> np.ndarray:
        """
        Extract and returns class-specific activations based on labels.

        Supports both one-hot encoded and integer label formats.
        """
        if self.model_module.use_onehot_encoder:
            return layer_activations[layer][y_data[:, class_index] == 1, :]
        
        return layer_activations[layer][y_data == class_index, :]
    

    def compute_empirical_distribution(self, class_activations:np.ndarray, empirical:bool, num_observations:int):
        """
        Computes mean and covariance, and optionally an empirical distribution of Mahalanobis distances on the
        given class activations.

        Args:
            class_activations (numpy.ndarray): Activations for a class.
            empirical (bool): Whether to compute ECDF.
            num_observations (int): Number of samples.
        """
        # Compute activation statistic moments 
        sample_mean = np.mean(class_activations, axis = 0)
        sample_var = np.cov(class_activations, rowvar=False) if len(class_activations) > 1 else np.zeros_like(class_activations)

        if not empirical:
            empirical_distribution_statistics = {
                "mean": sample_mean,
                "variance": sample_var,
                "num_observations": num_observations
            }
        
        else:
            mahalanobis_distances = []
            for activation in class_activations:
                
                # Compute difference between activation and sample mean
                diff = activation - sample_mean
                
                # Reshape diff to match the dimensions of sample_var
                diff_reshaped = diff[np.newaxis, ...]
                
                # Solve the linear system of equations
                solved = linalg.solve(sample_var, diff[..., np.newaxis])
                
                # Compute Mahalanobis distance
                distance = np.matmul(diff_reshaped, solved)[0][0]
                mahalanobis_distances.append(distance)
            
            # Compute empirical distribution
            empirical_distribution_statistics = {
                "mean": sample_mean,
                "variance": sample_var,
                "ECDF": ECDF(mahalanobis_distances)
            }
            
        return empirical_distribution_statistics


    def predict(self, x, y) -> np.ndarray:
        """Compute p-values for input data using Mahalanobis distance.
        
        Computes p-values per layer for each sample and combines them into a global p-value.
        Note: For activation values, nodes that are never activated need special handling.
        
        Args:
            x (numpy.ndarray): Input data samples
            y (numpy.ndarray): True labels
            
        Returns:
            combined_pvalue (numpy.ndarray): Global p-value per sample
        """
        
        # Calculate p-values for each layer
        pvalue = self._compute_statistics(x, y)

        # Combine p-values across layers into global p-values per sample
        combined_pvalue = self._combine_layer_pvalues(
            pvalues = pvalue, 
            num_samples = len(y), 
            test_type = self.test_type_between_layers
        )
        
        return combined_pvalue
    

    def _compute_statistics(self, x, y):
        """
        Compute per-layer p-values for each sample given in y.

        Args:
            x: Input data samples.
            y: Ground-truth labels.
        
        Returns:
            pvalue (numpy.ndarray): Array of shape (num_samples, num_layers) containing p-values 
                for each sample and layer.
        """
        num_samples = len(y)
        num_layers = len(self.selected_layers)
        
        pvalue = np.full(
            shape = (num_samples, num_layers),
            fill_value = np.inf,
            dtype  = np.float64
        )
                
        activations = self.model_module._get_pre_activations(x)

        test_type = self.test_type_fn_dict[self.test_type_within_layer]

        # Compute p-value per layer using the mahalanobis distance. 
        # See https://stats.stackexchange.com/questions/416198/calculate-p-value-of-multivariate-normal-distribution
        
        # for all predictions to be tested
        for sample_index, y_sample  in enumerate(y):

            # get the class index
            if self.model_module.use_onehot_encoder:
                class_label = self.classes[np.argmax(y_sample)]
            else:
                class_label = self.classes[y_sample]
            
            # for all layers ...
            for layer_index, layer in enumerate(self.selected_layers):
                
                # get the activations of the sample for the layer
                activation = activations[layer][sample_index]
                
                # If we are not at the last layer:
                if layer != self.last_layer:
                    pvalue[sample_index, layer_index] = test_type(activation, class_label, layer)

                else:
                    
                    # Multivariate approach using chi2
                    if self.model_module.use_onehot_encoder: 
                        pvalue[sample_index, layer_index] = self.chi2_statistic(activation, class_label, layer)
                    
                    # Assume univariate normal distribution, and do a two sided-test:
                    else:
                        pvalue[sample_index, layer_index] = self.two_sided_test(activation, class_label, layer)

        # Return p-value
        return(pvalue)


    def _combine_layer_pvalues(self, pvalues: np.ndarray, num_samples: int, test_type: str | None = None) -> np.ndarray:
        """
        Combine per-layer p-values into a global p-value per sample using one of the specified methods:
        - Fisher's method ("fisher")
        - Cauchy combination test ("cauchy")

        If only one layer is provided, this method returns the p-value for that layer. 

        Args:
            pvalues (numpy.ndarray): Per-layer p-values.
            num_samples (int): Number of samples.
            test_type (str): Combination method ("fisher" or "cauchy").
        
        Returns:
            numpy.ndarray: Combined p-values per sample.
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
                for i in range(num_samples)
            ])
        
        if test_type == 'cauchy':
            return np.array([
                self.CauchyCombinationTest(
                    p_values = pvalues[i, :],
                    weights = self.cauchy_weights_per_layer
                    )
                for i in range(num_samples)
            ])
        
        raise ValueError(f"Unknown test type: {test_type}")

    
    def chi2_statistic(self, activation, class_index, layer):
        """
        Compute p-value using Mahalanobis distance with χ² approximation.

        Assumes activations follow a multivariate Gaussian distribution.

        Args:
            activation (numpy.ndarray): Activation vector.
            class_index (int): Class index.
            layer (str): Layer name.
        
        Returns:
            result (float): p-value indicating how typical the activation is under the class distribution.
        """
        # Asymptotic assumption => chi2-distribution

        mean = self.layer_params[layer][class_index]["mean"]
        variance = self.layer_params[layer][class_index]["variance"]
        
        activation_centered = (activation - mean)[..., np.newaxis]
        
        inv_var_mean = linalg.solve(variance, activation_centered)
        
        distance = np.matmul(activation_centered.T, inv_var_mean).item()
        
        result = chi2.sf(distance, df=len(activation))
        
        return result
    
    def t2_statistic(self, activation, class_index, layer):
        """
        Compute a p-value using Hotelling’s T² statistic.

        Applies a finite-sample correction by mapping the statistic to an F-distribution.

        Returns:
            result (float): p-value indicating how typical the activation is under the class distribution.
        """
        # Using exact distribution, the Hotelling's T^2 distribution:

        # Number of observations for particular class during training
        n = self.layer_params[layer][class_index]["ECDF"]
        
        # The dimension of the random vector oif layer 
        p = np.shape(activation)[0]
        
        part_1 = (activation-self.layer_params[layer][class_index][0])[np.newaxis, ...]
        part_2 = linalg.solve(
            self.layer_params[layer][class_index][1],
            (activation-self.layer_params[layer][class_index][0])[..., np.newaxis]
            )
        
        f_obs = ((n-p)/(p*(n-1))) * np.matmul(part_1, part_2)[0][0]
        
        result = f.sf(f_obs, dfn=p, dfd=n-p)
        return result
    
    def mahalanobis_statistic(self, activation, class_index, layer):
        """
        Compute a p-value using the empirical distribution of Mahalanobis distances.

        Uses the ECDF of training distances to estimate how extreme the current sample is.

        Returns:
            result (float): p-value based on the empirical distribution.
        """

        part_1 = (activation-self.layer_params[layer][class_index]["mean"])[np.newaxis, ...]

        part_2 = linalg.solve(
            a = self.layer_params[layer][class_index]["variance"],
            b = (activation-self.layer_params[layer][class_index]["mean"])[..., np.newaxis]
            )
        
        product = np.matmul(part_1, part_2)[0][0]

        # Compute p-value using empirical distribution of Mahalanobis distance of correcly classified samples for the particular class:
        result = 1 - self.layer_params[layer][class_index]["ECDF"](product)
        
        return result
    
    def two_sided_test(self, activation, class_index, layer):
        """
        Compute a two-sided p-value under a univariate normal assumption.

        Used for the final layer when activations are treated as scalar values.

        Returns:
            float: p-value measuring deviation from the class mean.
        """

        mean = self.layer_params[layer][class_index]["mean"]
        variance = self.layer_params[layer][class_index]["variance"]

        # Calculate values for upper and lower tail probabilities
        upper_bound = activation if activation > mean else 2*mean - activation
        lower_bound = 2 * mean - activation if activation > mean else activation

        # Calculate p-value using the same formula in both cases
        upper_tail_prob = norm.sf(
            x = upper_bound,
            loc = mean,
            scale = variance
            )
        
        lower_tail_prob = norm.cdf(
            x = lower_bound,
            loc = mean,
            scale = variance
            )
        
        return upper_tail_prob + lower_tail_prob

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

if __name__ == "__main__":
    Mahalanobis(None, None, None)