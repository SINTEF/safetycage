"""
Plotting helpers for choosing a safety cage's decision threshold.

Both functions follow the matplotlib axis-level convention: one series per
call, an optional `ax` to draw on, the `Axes` returned, and nothing saved or
closed. Overlay by passing the same axes twice::

    ax = plot_metric_vs_threshold(thresholds_val, f1_val, label="Validation")
    plot_metric_vs_threshold(thresholds_test, f1_test, ax=ax, label="Test")
    ax.set_ylabel("F1")

To read the threshold against the spread of the statistic it cuts, put the
distributions on a twin axis and merge the legends::

    density_ax = ax.twinx()
    plot_statistic_distribution(statistics_val, ax=density_ax, label="Validation")
    plot_statistic_distribution(statistics_test, ax=density_ax, label="Test")
    ax.legend(
        *[a + b for a, b in zip(ax.get_legend_handles_labels(),
                                density_ax.get_legend_handles_labels())]
    )
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def plot_metric_vs_threshold(
    thresholds: np.ndarray,
    metrics: np.ndarray,
    *,
    label: str | None = None,
    ax: Axes | None = None,
    mark_best: bool = True,
    **plot_kwargs,
) -> Axes:
    """
    Plot a metric against the decision threshold.

    The threshold is the cut on the cage's statistic above (or below, depending
    on `SafetyCage.leq`) which a sample is flagged as a likely misclassification
    — the one-dimensional decision boundary this curve is used to choose.

    Args:
        thresholds (numpy.ndarray): Threshold values, the x-axis.
        metrics (numpy.ndarray): Metric value at each threshold. Assumed to be a
            metric where higher is better.
        label (str, optional): Legend label for the series. Omit to leave the
            series out of the legend.
        ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure is
            created when omitted.
        mark_best (bool, optional): Mark the metric-maximising threshold and
            report it in the legend label. (default: True).
        **plot_kwargs: Passed through to `matplotlib.axes.Axes.plot`, so
            `color`, `linestyle` and friends work as usual.

    Returns:
        matplotlib.axes.Axes: The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    thresholds = np.asarray(thresholds)
    metrics = np.asarray(metrics)
    (line,) = ax.plot(thresholds, metrics, **plot_kwargs)

    # Report the optimum in the series' own legend entry, so matplotlib handles
    # placement rather than hand-tuned text offsets, and one series costs one
    # legend row rather than two.
    if mark_best and metrics.size and not np.all(np.isnan(metrics)):
        best = int(np.nanargmax(metrics))
        ax.plot(thresholds[best], metrics[best], "o", color=line.get_color())

    if label:
        line.set_label(label)

    ax.set_xlabel("Decision threshold")
    return ax


def plot_statistic_distribution(
    statistics: np.ndarray,
    *,
    label: str | None = None,
    ax: Axes | None = None,
    bins: int = 100,
    **hist_kwargs,
) -> Axes:
    """
    Histogram the cage's statistic, for context behind a threshold curve.

    Draw this on `ax.twinx()` of a `plot_metric_vs_threshold` axes to see the
    chosen threshold against the spread of the values it separates.

    NaN statistics are dropped. Some methods (SPARDACUS in particular) emit NaN
    for samples they cannot score, and matplotlib refuses a histogram whose
    range is not finite.

    Args:
        statistics (numpy.ndarray): Statistic values, one per sample.
        label (str, optional): Legend label for the distribution.
        ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure is
            created when omitted.
        bins (int, optional): Number of histogram bins. (default: 100).
        **hist_kwargs: Passed through to `matplotlib.axes.Axes.hist`. Defaults
            to a translucent density, since this is background context.

    Returns:
        matplotlib.axes.Axes: The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots()

    statistics = np.asarray(statistics, dtype=float)
    statistics = statistics[~np.isnan(statistics)]

    # `alpha` here is matplotlib's opacity, unrelated to the threshold.
    hist_kwargs.setdefault("alpha", 0.3)
    hist_kwargs.setdefault("density", True)
    ax.hist(statistics, bins=bins, label=label, **hist_kwargs)

    ax.set_ylabel("Statistic density")
    return ax
