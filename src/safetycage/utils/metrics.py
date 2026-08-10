"""
Classification metrics for misclassification detection.

Everything here works on labels alone: ``y`` is the ground-truth
misclassification label and ``y_pred`` the flag a safety cage raised. Nothing in
this module needs a fitted cage, and nothing here sweeps thresholds — for that,
see :meth:`~safetycage.safetycage.SafetyCage.roc_curve` and
:meth:`~safetycage.safetycage.SafetyCage.auroc`, which need to know which
direction the cage flags in.

The individual metrics take confusion counts, which
:func:`calculate_confusion_rates` produces from labels; :func:`calculate_metrics`
chains the two for the common case.
"""

import numpy as np

def precision(TP, TN, FP, FN):
    """Compute precision (positive predictive value) from confusion matrix components."""
    denom = TP + FP
    return np.divide(TP, denom, out=np.zeros_like(TP, dtype=float), where=denom > 0)

def recall(TP, TN, FP, FN):
    """Compute recall (sensitivity) from confusion matrix components."""
    denom = TP + FN
    return np.divide(TP, denom, out=np.zeros_like(TP, dtype=float), where=denom > 0)

def specificity(TP, TN, FP, FN):
    """Compute specificity (true negative rate) from confusion matrix components."""
    denom = TN + FP
    return np.divide(TN, denom, out=np.zeros_like(TN, dtype=float), where=denom > 0)

def NPV(TP, TN, FP, FN):
    """Compute negative predictive value (NPV) from confusion matrix components."""
    denom = TN + FN
    return np.divide(TN, denom, out=np.zeros_like(TN, dtype=float), where=denom > 0)

def MCC(TP, TN, FP, FN):
    """
    Compute Matthews Correlation Coefficient (MCC) from confusion matrix components.

    Vectorized version of MCC calculation.
    TP, TN, FP, FN are numpy arrays of the same length.
    """

    # Calculate numerator and denominator arrays
    numerator = (TP * TN) - (FP * FN)
    # The four terms of the denominator
    d1, d2, d3, d4 = (TP + FP), (TP + FN), (TN + FP), (TN + FN)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_denom = 0.5 * (np.log(d1) + np.log(d2) + np.log(d3) + np.log(d4))
        denom = np.exp(log_denom) # Initialize output array mcc = np.zeros_like(numerator)

    # Ensure output is float to avoid UFuncTypeError
    out_arr = np.zeros_like(numerator, dtype=float)
    mcc = np.divide(numerator, denom, out=out_arr, where=denom != 0)

    return mcc

def accuracy(TP, TN, FP, FN):
    """Compute accuracy from confusion matrix components."""

    numerator = TP + TN
    denom = TP + TN + FP + FN

    return np.divide(numerator, denom, out=np.zeros_like(TN, dtype=float), where=denom > 0)

def f1_score(TP, TN, FP, FN):
    """Compute F1-score from confusion matrix components."""
    p = precision(TP, TN, FP, FN)
    r = recall(TP, TN, FP, FN)

    numerator = 2 * p * r
    denom = p + r

    return np.divide(numerator, denom, out=np.zeros_like(TN, dtype=float), where=denom > 0)



def calculate_confusion_rates(y: np.ndarray, y_pred: np.ndarray):
    """
    Calculates confusion rates (TP, TN, FP, FN) from true and predicted labels

    y represents the true misclassification labels and y_pred represents the predicted
    misclassification labels.

    Args:
        y (numpy.ndarray): Ground truth labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        dict: Dictionary containing TP, TN, FP, and FN counts.
    """

    return {
        "TP": np.sum((y == 1) & (y_pred == 1)).item(),
        "TN": np.sum((y == 0) & (y_pred == 0)).item(),
        "FP": np.sum((y == 0) & (y_pred == 1)).item(),
        "FN": np.sum((y == 1) & (y_pred == 0)).item()
    }

#: Metrics reported by :func:`calculate_metrics`, keyed by display name.
METRIC_FUNCTIONS = {
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity,
    "NPV": NPV,
    "MCC": MCC,
    "Accuracy": accuracy,
    "F1-score": f1_score,
}

def calculate_metrics(y: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate every metric in :data:`METRIC_FUNCTIONS` from true and predicted
    misclassification labels.

    This computes confusion matrix components once and applies each metric
    function to them.

    y represents the true misclassification labels and y_pred represents the predicted
    misclassification labels.

    For a single metric, or one that is not in the registry, apply it to the
    confusion counts directly::

        MCC(**calculate_confusion_rates(y, y_pred))

    Args:
        y (numpy.ndarray): Ground truth labels.
        y_pred (numpy.ndarray): Predicted labels.

    Returns:
        dict: Dictionary mapping metric names to metric values.
    """

    confusion_rates = calculate_confusion_rates(
        y=y,
        y_pred=y_pred,
    )

    metrics_dict = {}
    for name, func in METRIC_FUNCTIONS.items():
        # Force the result to a standard Python float
        metrics_dict[name] = float(func(**confusion_rates))

    return metrics_dict
