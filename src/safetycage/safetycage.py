from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path
from safetycage.modelmodule import ModelModule
from safetycage.datamodule import DataModule
class SafetyCage(ABC):
    """
    Abstract base class for safety cage methods.

    A safety cage detects misclassification in classification tasks. Safety cage methods
    evaluate predictive models by computing statistics that indicate whether a sample is 
    likely to be misclassified and comparing to some optimal threshold statistic, called "alpha". 
    Concrete base classes define how these statistics are computed, how predictions are 
    evaluated, and how to find the optimal alpha value.

    Subclasses must implement training, prediction, and the method to compute the statistic.
    Implementations on how to flag misclassifications, find the best threshold, and save/load the 
    safetycage are provided for simplicity, but should be overridden if they do not meet the requirements
    of the specific safety cage method.

    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
        num_classes (int): Number of classes. Retrieved from the data module.
        selected_classes (list): List of classes. Retrieved from the data module.
        alpha (float): Threshold statistic value used for flagging misclassifications.
        layer_params (dict, optional): Dictionary to store parameters for specific layers, if needed by 
            safety cage.
        leq (bool, optional): If True, samples with statistic less than or equal to alpha are flagged as 
            misclassified. Only required if the default flag method is used.
    """

    def __init__(
        self,
        model_module: ModelModule,
        data_module: DataModule,
        **kwargs
        ) -> None:
        """
        Initialize the safety cage.

        Stores references to the model and data modules and initializes shared attributes 
        used across all safety cage methods.

        Args:
            model_module: Model module used for predictions and activations.
            data_module: Data module providing datasets and class information.
        """

        self.model_module = model_module
        self.data_module = data_module
        
        self.num_classes = data_module.num_classes
        self.selected_classes = data_module.classes
        
        self.alpha = None
        
    #Train the parameters of the specified SafetyCage
    @abstractmethod
    def train_cage(self) -> None:
        """
        Train the safety cage.

        Learns parameters from training data that are later used to evaluate
        whether predictions are reliable.

        Returns:
            None
        """
        pass
    
    #Apply the SafetyCage on unseen test samples
    @abstractmethod
    def predict(self, x, y) -> Optional[np.ndarray]:
        """
        Evaluate input samples using the trained safety cage.

        Computes statistics that indicate how likely each prediction is a misclassification.

        Args:
            x: Input data samples
            y: True labels

        Returns:
            numpy.ndarray: Computed statistics for each sample
        """
        pass

    #Compute the statistics to evaluate whether each test sample is wrongly predicted
    @abstractmethod
    def _compute_statistics(self, x, y):
        """
        Compute per-sample statistics used to evaluate prediction reliability.

        Args:
            x: Input data samples
            y: True labels

        Returns:
            numpy.ndarray: Per-sample statistics
        """
        pass

    #Flag predictions as being correct (0) or wrong (1)
    def flag(self, statistics: float | np.ndarray, alpha: float | None = None) -> float | np.ndarray:
        """
        Flag samples with probability less than or equal (safetycage.leq = True) to alpha as incorrect
        or probability more than or equal (safetycage.leq = False) to alpha as incorrect.
        
        This method identifies samples where the maximum/minimum probability is below/above a
        specified threshold (alpha), marking them as potentially incorrect classifications.

        If some statistics are np.NaN values (as a result of unreliable_classes), the corresponding flag
        will be set to np.NaN as well.

        *Requires safetycage.leq to be defined, not None.*

        Args:
            statistics (numpy.ndarray): Array of probability values to evaluate
            alpha (float): Threshold value for flagging samples (0 to 1)
        Returns:
            numpy.ndarray: Boolean array where True indicates probabilities below/above the alpha threshold
                depending on safetycage.leq. There are NaN values given for when the statistic is NaN.
        """
        if self.leq is None:
            raise ValueError("safetycage.leq is not defined. Define safetycage.leq to use the default flag method.")

        # Check priority of alpha parameter
        if alpha is None:
            # If not provided as input, try to use self.alpha
            if hasattr(self, 'alpha') and self.alpha is not None:
                alpha = self.alpha
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing alpha parameter: must be provided as input or set as class attribute")
        
        if self.leq:
            flags = statistics <= alpha
        elif not self.leq and self.leq is not None:
            flags = statistics >= alpha

        return flags

    def find_best_threshold(self, y_true, y_probs, metric_fn, greater_is_better=True) -> float | np.ndarray:
        """
        Find the optimal threshold for flagging samples by calling self.flag().

        Evaluates thresholds t (from 1000 samples between min to max) and selects the one that maximizes 
        the given metric.

        Args:
            y_true (numpy.ndarray): Ground-truth misclassification labels
            y_probs (numpy.ndarray): Computed statistics or probabilities
            metric_fn (callable): Function to evaluate performance
            greater_is_better (bool, optional): Whether greater metric values are better (default: True)

        Returns:
            dict: Dictionary containing the optimal threshold and the corresponding best metric value
        """
        if greater_is_better:
            compare = lambda x, y: x > y
            best_metric = -np.inf 
        else:
            compare = lambda x, y: x < y
            best_metric = np.inf 

        best_alpha = -np.inf
        for t in np.linspace(min(y_probs), max(y_probs), num=1000):

            flag = self.flag(y_probs, t)

            # Flag is true when misclassification occurs
            tps = np.sum(flag & y_true)
            fps = np.sum(flag & (1 - y_true))
            
            total_pos = y_true.sum()
            total_neg = y_true.size - total_pos
            
            fns = total_pos - tps
            tns = total_neg - fps
            
            metric = metric_fn(TP=tps, TN=tns, FP=fps, FN=fns)
            
            if compare(metric, best_metric):
                best_metric = metric
                best_alpha = t
                
        return {
            "alpha_opt": best_alpha,
            "metric_max": best_metric,
        }
    
    def save_cage(self, path):
        """Save trained cage parameters to a joblib file.

        Args:
            path (str or Path): File path to save to (should end in .joblib).

        Raises:
            ValueError: If the cage has not been trained (alpha not set).
        """
        if getattr(self, "alpha", None) is None:
            raise ValueError("Cannot save: cage has not been trained (alpha is not set).")

        parameters = {"alpha": self.alpha}

        if getattr(self, "layer_params", None) is not None:
            parameters["layer_params"] = self.layer_params

        if getattr(self, "unreliable_classes", None):
            parameters["unreliable_classes"] = self.unreliable_classes

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(parameters, path)
    
    @classmethod
    def load_cage(cls, path, model_module, data_module):
        """Load a trained cage from a saved file.

        Args:
            path (str or Path): Path to the saved .joblib file.
            model_module: The model module to use with the loaded cage.
            data_module: The data module to use with the loaded cage.

        Returns:
            An instance of the cage class with trained parameters restored.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No saved cage found at {path}")

        parameters = joblib.load(path)

        instance = cls(model_module=model_module, data_module=data_module)
        instance.alpha = parameters["alpha"]

        if "layer_params" in parameters:
            instance.layer_params = parameters["layer_params"]

        if "unreliable_classes" in parameters:
            instance.unreliable_classes = parameters["unreliable_classes"]

        return instance
                
if __name__ == "__main__":
    SafetyCage(None, None, None)