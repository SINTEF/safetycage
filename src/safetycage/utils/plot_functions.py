# This script contains the library of functions used in the Exaigon safety_cage development
from mailbox import linesep
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from typing import Literal

from ..utils.evaluate import calculate_roc_curve

def annotate_text_box(
    x: float,
    y: float,
    metric_name: str,
    colour: str,
    label_prefix: str = "Best",
    side: str = "auto",          # "left" | "right" | "auto"
    vertical: str = "auto",      # "top" | "bottom" | "auto"
    # layout parameters
    dx: float = 0.0,
    dy: float = 0.0,
    point: bool = True,
    point_size: int = 3,
    zorder: int = 3,
    label_x_offset: float = 0.08,   # horizontal distance from point
    label_y_offset: float = 0.05,   # vertical distance from point
    pad_fraction_x: float = 0.02,   # padding as fraction of axis width
    pad_fraction_y: float = 0.01,   # padding as fraction of axis height
    bbox_pad: float = 0.3,          # value passed to boxstyle round,pad=
    bbox_lw: float = 1.0,
    arrow_lw: float = 1.5,
    ):
    """
    Annotate a point (x,y) with a text box and arrow.

    This is designed specifically for annotating the alpha curve.
    Enforce relative label positions by using side and vertical arguments.
    Hardcode label positions by altering direction factor.

    Args:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        metric_name (str): Name of the metric shown in the label.
        colour (str): Colour used for the annotation box and arrow.
        label_prefix (str, optional): Prefix shown in the label. (default: "Best").
        side (str, optional): Horizontal side of the label ("left", "right", or "auto"). (default: "auto").
        vertical (str, optional): Vertical side of the label ("top", "bottom", or "auto"). (default: "auto").
        dx (float, optional): Extra horizontal offset. (default: 0.0).
        dy (float, optional): Extra vertical offset. (default: 0.0).
        point (bool, optional): Whether to draw the point marker. (default: True).
        point_size (int, optional): Size of the point marker. (default: 3).
        zorder (int, optional): Drawing order for the annotation. (default: 3).
        label_x_offset (float, optional): Horizontal distance from the point. (default: 0.08).
        label_y_offset (float, optional): Vertical distance from the point. (default: 0.05).
        pad_fraction_x (float, optional): Horizontal padding as a fraction of axis width. (default: 0.02).
        pad_fraction_y (float, optional): Vertical padding as a fraction of axis height. (default: 0.01).
        bbox_pad (float, optional): Padding inside the text box. (default: 0.3).
        bbox_lw (float, optional): Line width of the text box. (default: 1.0).
        arrow_lw (float, optional): Line width of the arrow. (default: 1.5).

    Further Explanations of specific arguments:

    side:
      - "right": label to the right of point
      - "left":  label to the left of point
      - "auto":  choose based on x location (keeps label inside axes)

    vertical:
      - "top":    label above point
      - "bottom": label below point
      - "auto":   choose based on y location
    
    Improvements to Make:
      - FIX: currently, the only function that calls this is plot_alpha_metric_curve, which currently
        has test metrics labels go upward, and val downward.
      - Test for all cases, it's a bit weirdly hardcoded where it works for most cases we'll see, but not all possible cases.
    """

    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Choose side/vertical to determine which side of the point that the label is on
    if side == "auto":
        side = "right" if x < (xmin + xmax) / 2 else "left"
    if vertical == "auto":
        vertical = "top" if y < (ymin + ymax) / 2 else "bottom"

    # Signed offsets depending on side/vertical
    sx = +1 if side == "right" else -1
    sy = +1 if vertical == "top" else -1

    # Calculate better position for annotation text to ensure it's inside the figure
    text_x = x + sx * (label_x_offset + dx)
    text_y = y + sy * (label_y_offset + dy)

    pad_x = pad_fraction_x * (xmax - xmin)
    pad_y = pad_fraction_y * (ymax - ymin)
    text_x = min(max(text_x, xmin + pad_x), xmax - pad_x) # 0.02 * (xmax - xmin) <= x <= x + sx * (0.08 + dx)
    text_y = min(max(text_y, ymin + pad_y), ymax - pad_y)
    
    # Align text box in terms relative to the point
    ha = "left" if side == "right" else "right"
    va = "bottom" if vertical == "top" else "top"

    ax.annotate(
        text=f"{label_prefix} alpha: {x:.3f}\nMax {metric_name}: {y:.3f}",
        xy=(x, y),
        xytext=(text_x, text_y),
        bbox=dict(boxstyle=f"round,pad={bbox_pad}", fc="white", ec=colour, lw=bbox_lw),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color=colour, lw=arrow_lw),
        ha=ha, va=va, zorder=zorder,
    )

    if point:
        ax.plot(x, y, "o", color=colour, markersize=point_size, zorder=zorder)


def plot_alpha_metric_curve(
    alpha_test: np.ndarray,
    metric_test: np.ndarray,
    alpha_val: np.ndarray,
    metric_val: np.ndarray,
    max_alpha_test: float,
    max_metric_test: float,
    max_alpha_val: float,
    max_metric_val: float,
    metric_name: str,
    statistics_val: np.ndarray,
    statistics_test: np.ndarray,
    output_path: str,
    # layout parameters
    xlim: tuple[float, float] | None = (0,1),
    ylim_metric: tuple[float, float] | None = None,
    ylim_distribution: tuple[float, float] | None = None,
    save: bool = True,
    test_label_offset: tuple[float, float] = (0.0, 0.0),
    val_label_offset: tuple[float, float] = (0.0, 0.0),
    zorder: int = -1,
    test_colour: str = "blue",
    val_colour: str = "red",
    test_relative_vertical: str = "top",
    val_relative_vertical: str = "bottom",
    n_bins_histogram: int = 100,
    save_dpi: int = 300,
    ) -> None:
    """
    Plot a metric curve as a function of alpha.

    This method plots the metric values for test ("alpha_opt" and "metric_max") and validation 
    ("alpha_val" and "metric_val") thresholds and annotates the best test and validation points.

    Args:
        alpha_test (numpy.ndarray): Alpha values for the test curve.
        metric_test (numpy.ndarray): Metric values for the test curve.
        alpha_val (numpy.ndarray): Alpha values for the validation curve.
        metric_val (numpy.ndarray): Metric values for the validation curve.
        max_alpha_test (float): Best test alpha value.
        max_metric_test (float): Best test metric value.
        max_alpha_val (float): Best validation alpha value.
        max_metric_val (float): Best validation metric value.
        metric_name (str): Name of the metric being plotted.
        output_path (str): Directory where the figure should be saved.
        xlim (tuple[float, float] | None, optional): X-axis limits. (default: (0,1)).
        ylim_metric (tuple[float, float] | None, optional): Y-axis limits for the metric. Automatically fits to the data. (default: None).
        ylim_distribution (tuple[float, float] | None, optional): Y-axis limits for the alpha distribution. Automatically fits to the data. (default: None). 
        save (bool, optional): Whether to save the figure. (default: True).
        test_label_offset (tuple[float, float], optional): Offset for the test label. (default: (0.0, 0.0)).
        val_label_offset (tuple[float, float], optional): Offset for the validation label. (default: (0.0, 0.0)).
        zorder (int, optional): Drawing order of the metric curves. (default: -1).
        y_padding (float, optional): Padding factor for the y-axis. (default: 10.0).
        test_colour (str, optional): Colour of the test annotation. (default: "blue").
        val_colour (str, optional): Colour of the validation annotation. (default: "red").
        test_relative_vertical (str, optional): Vertical placement of the test annotation. (default: "top").
        val_relative_vertical (str, optional): Vertical placement of the validation annotation. (default: "bottom").
        n_bins_histogram (int, optional): Number of bins in the historgram. (default: 100).
        save_dpi (int, optional): Resolution used when saving the figure. (default: 300).
    """
    fig, ax1 = plt.subplots()

    # --- Line plots ---
    ax1.plot(alpha_test, metric_test, label="Test", zorder=zorder)
    ax1.plot(alpha_val, metric_val, label="Validation", zorder=zorder)
    ax1.set_ylabel(metric_name)

    dx, dy = test_label_offset
    annotate_text_box(
        max_alpha_test,
        max_metric_test,
        metric_name,
        label_prefix="Test",
        colour=test_colour,
        vertical=test_relative_vertical,
        dx=dx,
        dy=dy,
    )

    dx, dy = val_label_offset
    annotate_text_box(
        max_alpha_val,
        max_metric_val,
        metric_name,
        label_prefix="Val",
        colour=val_colour,
        vertical=val_relative_vertical,
        dx=dx,
        dy=dy,
    )

    if ylim_metric is not None:
        ax1.set_ylim(*ylim_metric)

    # --- Second axis for histograms ---
    ax2 = ax1.twinx()

    ax2.hist(
        statistics_val,
        bins=n_bins_histogram,
        alpha=0.3,
        color=val_colour,
        density=True,
        label="Validation Alpha Distribution"
    )

    ax2.hist(
        statistics_test,
        bins=n_bins_histogram,
        alpha=0.3,
        color=test_colour,
        density=True,
        label="Test Alpha Distribution"
    )

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2)

    if ylim_distribution is not None:
        ax2.set_ylim(*ylim_distribution)

    plt.xlim(*xlim)

    ax1.set_xlabel("Alpha")
    ax2.set_ylabel("Statistics Density")
    if save:
        path = os.path.join(output_path, f"alpha_{metric_name}_curve.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=save_dpi)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    normalize: Literal['true', 'pred', 'all'], 
    output_path: str,
    cmap = plt.cm.Blues,
    save:bool = True,
    ):
    """
    Plot a confusion matrix for predicted misclassification labels.

    Optionally saves the figure to the given output directory.

    Args:
        y_true (numpy.ndarray): True misclassification labels.
        y_pred (numpy.ndarray): Predicted misclassification labels.
        normalize (Literal['true', 'pred', 'all']): Normalization mode.
        output_path (str): Directory where the figure should be saved.
        cmap (matplotlib.colors.Colormap, optional): Colour map for the plot. (default: plt.cm.Blues).
        save (bool, optional): Whether to save the figure. (default: True).
    """

    # Return confusion matrix for the best threshold
    cm = metrics.confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        normalize=normalize,
    )

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Correct prediction", "Incorrect prediction"])
    plt.yticks(tick_marks, ["Correct prediction", "Incorrect prediction"])

    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=cmap, ax=plt.gca())

    plt.ylabel("True Misclassification")
    plt.xlabel("Predicted Misclassification")

    if save is True:
        path = os.path.join(output_path, "confusion_matrix.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    plt.close()


def plot_roc_curve(
    tpr, fpr, thresholds,
    output_path: str,
    save:bool = True,
    fig_dimensions = (6,6),
    curve_color: str = "blue",
    fill_in_curve: bool = False,
    fill_in_opacity: float = 0.3,
    diag_color: str = "red",
    diag_linestyle: str = "--",
    xlim: tuple[float, float] = [0.0, 1.0],
    ylim: tuple[float, float] = [0.0, 1.0],
    save_dpi: int = 300
    ):
    """
    Plot a receiver operating characteristic (ROC) curve.

    Plots the true positive rate against the false positive rate and optionally saves the figure to disk.

    Args:
        tpr (numpy.ndarray): True positive rates.
        fpr (numpy.ndarray): False positive rates.
        thresholds (numpy.ndarray): Threshold values used to compute the curve.
        output_path (str): Directory where the figure should be saved.
        save (bool, optional): Whether to save the figure. (default: True).
        fig_dimensions (tuple, optional): Figure size. (default: (6, 6)).
        curve_color (str, optional): Colour of the ROC curve. (default: "blue").
        fill_in_curve (bool, optional): Whether to shade the area under the curve. (default: False).
        fill_in_opacity (float, optional): Opacity of the shaded area. (default: 0.3).
        diag_color (str, optional): Colour of the diagonal reference line. (default: "red").
        diag_linestyle (str, optional): Line style of the diagonal reference line. (default: "--").
        xlim (tuple[float, float], optional): X-axis limits. (default: [0.0, 1.0]).
        ylim (tuple[float, float], optional): Y-axis limits. (default: [0.0, 1.0]).
        save_dpi (int, optional): Resolution used when saving the figure. (default: 300).
    """
    
    plt.figure(figsize=fig_dimensions)

    # Plot the ROC curve with correct axes
    plt.plot(fpr, tpr, color=curve_color)

     # [OPTIONALLY: Default = False] Add shaded area under the curve
    if fill_in_curve:
        plt.fill_between(fpr, tpr, alpha=fill_in_opacity, color=curve_color)


    plt.plot(xlim, ylim, color=diag_color, linestyle=diag_linestyle)  # Diagonal line
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    
    if save is True:
        path = os.path.join(output_path, "roc_curve.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=save_dpi)
    plt.close()