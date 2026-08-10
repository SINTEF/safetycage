import numpy as np
from safetycage.safetycage import SafetyCage

class MSP(SafetyCage):
    """
    Maximum Softmax Probability (MSP) Safety Cage Method.
    
    This class implements a simple safety cage based on maximum softmax probability thresholding
    for detecting uncertain predictions in neural network classifiers. The approach flags predictions
    as potentially incorrect when the maximum softmax probability falls below a specified threshold. Hence,
    the method works for any classifier that outputs class probabilities, not just neural networks.
    
    **Reference:**
        Hendrycks, D., & Gimpel, K. (2016). A Baseline for Detecting Misclassified and Out-of-Distribution
        Examples in Neural Networks.
        https://arxiv.org/abs/1610.02136
    
    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
    """

    def __init__(self, model_module, data_module, **kwargs):
        """
        Initialize the MSP safety cage method.

        Args:
            model_module: Reference to the model module.
            data_module: Reference to the data module.
            **kwargs: Additional keyword arguments.
        """
        super(MSP, self).__init__(model_module, data_module, **kwargs)
        self.leq = True
    
    @property
    def name(self):
        """Return the name of the safety cage method."""
        return "MSP"
    
    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        """
    No training step is required for MSP.

    Args:
        x: Unused. Included for API consistency.
        y: Unused. Included for API consistency.
        y_pred: Unused. Included for API consistency.
    """
        pass

    def predict(self, x, y) -> np.ndarray:
        """
        Compute misclassification detection statistic based on model predictions.

        Args:
            x (numpy.ndarray): Input data to be processed by the model.
        
        Returns:
            numpy.ndarray: Array of maximum probabilities for each input sample.
        """
        
        statistics = self._compute_statistics(x)

        return statistics
    
    def _compute_statistics(self, x):
        """
        Compute misclassification detection statistic based on model predictions.
        This method processes input data through the model to get softmax probabilities.
        and returns the maximum probability for each sample.

        Args:
            x (numpy.ndarray): Input data to be processed by the model.
            y (numpy.ndarray): Target labels (not used in current implementation).
        
        Returns:
            numpy.ndarray: Array of maximum probabilities for each input sample.
        """
        
        # Get softmax probabilities from model
        probabilities = self.model_module._get_probabilities(x)
        
        # Get maximum probability for each sample
        max_probabilities = np.max(probabilities, axis=1)
        
        return max_probabilities

if __name__ == "__main__":
    MSP(None, None)