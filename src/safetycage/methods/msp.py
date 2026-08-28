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
    No training step is required for MSP.

    Args:
        x: Unused. Included for API consistency.
        y: Unused. Included for API consistency.
        y_pred: Unused. Included for API consistency.
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
        if find_threshold:
            self.fit_threshold(
                metric_fn=metric_fn,
                x_val=x_val,
                y_val=y_val,
                search=search,
                num_thresholds=num_thresholds,
                greater_is_better=greater_is_better,
            )

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