from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
import joblib
from pathlib import Path
from safetycage.modelmodule import ModelModule
from safetycage.datamodule import DataModule
from safetycage.utils.metrics import calculate_confusion_rates, recall, specificity
class SafetyCage(ABC):
    """
    Abstract base class for safety cage methods.

    A safety cage detects misclassification in classification tasks. Safety cage methods
    evaluate predictive models by computing statistics that indicate whether a sample is 
    likely to be misclassified and comparing to some optimal threshold. 
    Concrete base classes define how these statistics are computed, how predictions are 
    evaluated, and how to find the optimal threshold.

    Subclasses must implement training, prediction, and the method to compute the statistic.
    Implementations on how to flag misclassifications, find the best threshold, and save/load the 
    safetycage are provided for simplicity, but should be overridden if they do not meet the requirements
    of the specific safety cage method.

    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
        num_classes (int): Number of classes. Retrieved from the data module.
        selected_classes (list): List of classes. Retrieved from the data module.
        threshold (float): Threshold statistic value used for flagging misclassifications.
        layer_params (dict, optional): Dictionary to store parameters for specific layers, if needed by 
            safety cage.
        leq (bool, optional): If True, samples with statistic less than or equal to the threshold are flagged as 
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
        
        self.threshold = None
        
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
    def flag(self, statistics: float | np.ndarray, threshold: float | None = None) -> float | np.ndarray:
        """
        Flag samples with probability less than or equal (safetycage.leq = True) to threshold as incorrect
        or probability more than or equal (safetycage.leq = False) to threshold as incorrect.
        
        This method identifies samples where the maximum/minimum probability is below/above a
        specified threshold (threshold), marking them as potentially incorrect classifications.

        If some statistics are np.NaN values (as a result of unreliable_classes), the corresponding flag
        will be set to np.NaN as well.

        *Requires safetycage.leq to be defined, not None.*

        Args:
            statistics (numpy.ndarray): Array of probability values to evaluate
            threshold (float): Threshold value for flagging samples (0 to 1)
        Returns:
            numpy.ndarray: Boolean array where True indicates probabilities below/above the threshold
                depending on safetycage.leq. There are NaN values given for when the statistic is NaN.
        """
        if self.leq is None:
            raise ValueError("safetycage.leq is not defined. Define safetycage.leq to use the default flag method.")

        # Check priority of threshold parameter
        if threshold is None:
            # If not provided as input, try to use self.threshold
            if hasattr(self, 'threshold') and self.threshold is not None:
                threshold = self.threshold
            else:
                # If neither source is available, raise an error
                raise ValueError("Missing threshold parameter: must be provided as input or set as class attribute")

        if self.leq:
            flags = statistics <= threshold
        elif not self.leq and self.leq is not None:
            flags = statistics >= threshold

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

        thresholds = np.linspace(min(y_probs), max(y_probs), num=1000)
        metrics = []
        for t in thresholds:

            flag = self.flag(y_probs, t)

            # Flag is true when misclassification occurs
            tps = np.sum(flag & y_true)
            fps = np.sum(flag & (1 - y_true))
            
            total_pos = y_true.sum()
            total_neg = y_true.size - total_pos
            
            fns = total_pos - tps
            tns = total_neg - fps
            
            metric = metric_fn(TP=tps, TN=tns, FP=fps, FN=fns)
            metrics.append(metric)

        optimal_metric_index = np.argmax(metrics) if greater_is_better else np.argmin(metrics)
        best_threshold = thresholds[optimal_metric_index]
        best_metric = metrics[optimal_metric_index]
        
        return {
            "threshold_opt": best_threshold,
            "metric_max": best_metric,
            "thresholds": thresholds,
            "metrics": metrics,
        }

    def roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute the ROC curve by sweeping self.flag() across every threshold.

        Going through self.flag() means this works whichever direction a method
        flags in, and for methods that override flag() entirely.

        Thresholds are the unique finite statistics, padded with -inf and +inf so
        the curve reaches both (0, 0) and (1, 1) either way round. NaN statistics
        stay in the data — they simply never compare true — but are not used as
        thresholds, since a comparison against NaN is always False and would add
        a spurious "nothing flagged" point.

        Args:
            y_true (numpy.ndarray): Ground-truth misclassification labels
            y_pred (numpy.ndarray): Computed statistics or probabilities

        Returns:
            dict: Dictionary containing
                - fpr (numpy.ndarray): False positive rates
                - tpr (numpy.ndarray): True positive rates
                - thresholds (numpy.ndarray): Threshold values used
        """

        y_pred = np.asarray(y_pred, dtype=float)

        finite = np.unique(y_pred[np.isfinite(y_pred)])
        thresholds = np.concatenate(([-np.inf], finite, [np.inf]))

        tpr_values = []
        fpr_values = []

        for threshold in thresholds:
            flags = self.flag(y_pred, threshold)
            confusion_rates = calculate_confusion_rates(y=y_true, y_pred=flags)

            tpr_values.append(recall(**confusion_rates))
            fpr_values.append(1.0 - specificity(**confusion_rates))

        return {
            "fpr": np.array(fpr_values),
            "tpr": np.array(tpr_values),
            "thresholds": thresholds,
        }

    def auroc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the area under the ROC curve produced by self.roc_curve().

        Args:
            y_true (numpy.ndarray): Ground-truth misclassification labels
            y_pred (numpy.ndarray): Computed statistics or probabilities

        Returns:
            float: The area under the ROC curve
        """

        curve = self.roc_curve(y_true, y_pred)

        # Integrate left to right: the sweep visits thresholds in either
        # direction depending on how the method flags, and np.trapezoid needs a
        # monotone x. Break ties by TPR so a vertical run ends on the top of the
        # step — ordering it arbitrarily makes the trapezoid cut the corner and
        # understates the area.
        order = np.lexsort((curve["tpr"], curve["fpr"]))
        auc_value = np.trapezoid(y=curve["tpr"][order], x=curve["fpr"][order])

        return float(auc_value)

    def save_cage(self, path):
        """Save trained cage parameters to a joblib file.

        Args:
            path (str or Path): File path to save to (should end in .joblib).

        Raises:
            ValueError: If the cage has not been trained (alpha not set).
        """
        if getattr(self, "threshold", None) is None:
            raise ValueError("Cannot save: cage has not been trained (threshold is not set).")

        parameters = {"threshold": self.threshold}

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
        instance.threshold = parameters["threshold"]

        if "layer_params" in parameters:
            instance.layer_params = parameters["layer_params"]

        if "unreliable_classes" in parameters:
            instance.unreliable_classes = parameters["unreliable_classes"]

        return instance
                
if __name__ == "__main__":
    SafetyCage(None, None, None)