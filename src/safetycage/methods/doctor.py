import numpy as np
from safetycage.safetycage import SafetyCage

class DOCTOR(SafetyCage):
    """
    DOCTOR Safety Cage Method.
    
    This class implements a safety cage based on estimating the probability of misclassification from 
    the model’s predicted class probability distribution. The method derives an error-related 
    uncertainty score (e.g., using 1 - max(p) or 1 - sum(p²)) and flags predictions as potentially incorrect 
    when this estimated error probability is above a certain threshold.
    
    **Reference:**
        Granese, F., Romanelli, M., Gorla, D., Palamidessi, C., & Piantanida, P. (2021).
        DOCTOR: A Simple Method for Detecting Misclassification Errors.
        https://arxiv.org/pdf/2306.01710

    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
        method (str): Method for calculating error probability ('max' or 'sum').
    """

    def __init__(self, model_module, data_module, **kwargs):
        """
        Initialize the DOCTOR safety cage method.

        Args:
            model_module: Reference to model module object for making predictions.
            data_module: Reference to data module object for handling data.
            method (str): Method for calculating error probability ('max' or 'sum').
            **kwargs: Additional keyword arguments.
        """
        super(DOCTOR, self).__init__(model_module, data_module, **kwargs)
        self.method = kwargs.get("method")
        self.leq = False

    @property
    def name(self):
        """Return the name of the safety cage method."""
        return "DOCTOR"
    
    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        """
        Estimate empirical misclassification rate from training data.

        Args:
            x: Input data
            y: Tuple containing (correct_predictions, incorrect_predictions)
        """

        if y is None:
            x, y = self.data_module.data_train
        if y_pred is None:
            y_pred = self.model_module._get_predictions(x)

        num_incorrect = (y != y_pred).sum()
        total_samples = len(y) 
        
        # Calculate empirical error probability
        self.PE_1 = num_incorrect / total_samples


    def predict(self, x, y) -> np.ndarray:
        """
        Compute statistical metrics based on model predictions.

        Args:
            x (numpy.ndarray): Input data to be processed by the model.
            y (numpy.ndarray): True labels
        
        Returns:
            numpy.ndarray: Array of maximum probabilities for each input sample.
        """
        statistics =  self._compute_statistics(x, y)

        return statistics
    
    def _compute_statistics(self, x, y):
        """
        Compute uncertainty statistics based on softmax probabilities.
        
        Args:
            x: Input data
            y: Ground truth labels
            
        Returns:
            numpy.ndarray: Uncertainty score Pe(x)/(1-Pe(x)) values
            
        Raises:
            ValueError: If method is not 'max' or 'sum'
        """
        # Get prediction probabilities from model
        probs = self.model_module._get_probabilities(x)
        
        # Calculate error probability based on selected method
        if self.method == "max":
            # Use maximum probability
            error_prob = 1 - np.max(probs, axis=1)
        elif self.method == "sum":
            # Use sum of squared probabilities
            error_prob = 1 - np.sum(probs**2, axis=1) 
        else:
            raise ValueError(f"Invalid method '{self.method}'. Must be 'max' or 'sum'")

        # Return uncertainty ratio
        return error_prob / (1 - error_prob)

if __name__ == "__main__":
    DOCTOR(None, None)