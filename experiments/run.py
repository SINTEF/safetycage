"""Run misclassification-detection experiments across pipelines and safety cage methods.

Fits every (pipeline, safety cage) combination requested with
``fit(find_threshold=True)`` -- training on the pipeline's cage-train split
and picking a threshold on its cage-val split in one call -- then evaluates
once on the held-out cage-test split. Each combination appends a row to
``experiments/results.json`` (upserted by (pipeline, method), so re-running
a combination replaces its own row). Every run's console output is also
written to a timestamped file under ``experiments/logs/``.

Usage:
    python -m experiments.run --pipeline mnist --safetycage msp doctor red spardacus mahalanobis random_baseline
    python -m experiments.run --pipeline mnist cifar10 --safetycage msp random_baseline

A failed combination (e.g. a missing checkpoint) is reported and skipped
rather than aborting the whole sweep.
"""
import argparse
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

from experiments.registry import PIPELINES, SAFETYCAGES
from experiments.results_store import flatten_metrics, save_result
from safetycage.utils.metrics import MCC

LOGS_DIR = Path(__file__).resolve().parent / "logs"
METRIC_FUNCTIONS = {"MCC": MCC}

logger = logging.getLogger("experiments")


def _setup_logging():
    """Log to the console and to a fresh timestamped file under experiments/logs/."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    for handler in (logging.FileHandler(log_path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return log_path


def classifier_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def run_experiment(pipeline, safetycage, device="cpu", metric_fn=MCC, cage_kwargs=None, pipeline_kwargs=None):
    """Fit one safety cage on one pipeline and evaluate it on the held-out test split.

    Args:
        pipeline (str): Key into ``experiments.registry.PIPELINES``.
        safetycage (str): Key into ``experiments.registry.SAFETYCAGES``.
        device (str): Torch device to load the pipeline's model onto.
        metric_fn (callable): Metric maximized when picking the threshold.
        cage_kwargs (dict, optional): Overrides for the cage's default constructor kwargs.
        pipeline_kwargs (dict, optional): Extra kwargs for the pipeline factory,
            e.g. ``{"retrain": True}`` for ``mnist`` to train a fresh model
            instead of loading its checkpoint.

    Returns:
        dict: The record appended to ``experiments/results.json``.
    """
    model_module, data_module = PIPELINES[pipeline](device=device, **(pipeline_kwargs or {}))

    cage_cls, default_kwargs = SAFETYCAGES[safetycage]
    kwargs = {**default_kwargs, **(cage_kwargs or {})}
    cage = cage_cls(model_module=model_module, data_module=data_module, **kwargs)

    cage.fit(find_threshold=True, metric_fn=metric_fn)

    x_test, y_test = data_module.data_test
    y_pred_test = model_module._get_predictions(x_test)
    statistics_test = np.asarray(cage.predict(x_test, y_pred_test), dtype=float)

    # Methods that mark some classes "unreliable" (e.g. SPARDACUS) return NaN
    # statistics for them -- drop those samples before scoring, same as the
    # example notebooks do by hand.
    valid = ~np.isnan(statistics_test)
    if not valid.all():
        y_test, y_pred_test, statistics_test = y_test[valid], y_pred_test[valid], statistics_test[valid]

    incorrect_test = (y_pred_test != y_test).astype(int)
    flagged_test = cage.flag(statistics_test)
    auroc = cage.auroc(y_true=incorrect_test, y_pred=statistics_test)

    caught_mask = flagged_test & (incorrect_test == 1)
    missed_mask = (~flagged_test) & (incorrect_test == 1)
    kept = ~flagged_test

    no_cage_metrics = classifier_metrics(y_test, y_pred_test)
    kept_metrics = classifier_metrics(y_test[kept], y_pred_test[kept]) if kept.any() else no_cage_metrics

    corrected_test = y_pred_test.copy()
    corrected_test[caught_mask] = y_test[caught_mask]
    corrected_metrics = classifier_metrics(y_test, corrected_test)

    # Points gained per fraction of the test set flagged (i.e. sent for
    # review) -- lets you compare methods that flag very different amounts.
    # Undefined (NaN) if nothing was flagged.
    flagged_fraction = flagged_test.mean()
    gain_per_review_metrics = {
        name: (corrected_metrics[name] - no_cage_metrics[name]) / flagged_fraction
        if flagged_fraction > 0 else float("nan")
        for name in no_cage_metrics
    }

    record = {
        "threshold": float(cage.threshold),
        "auroc": float(auroc),
        "test_samples": int(len(y_test)),
        "total_errors": int(incorrect_test.sum()),
        "flagged": int(flagged_test.sum()),
        "errors_caught": int(caught_mask.sum()),
        "errors_missed": int(missed_mask.sum()),
        "coverage": float(kept.mean()),
        **flatten_metrics("no_cage", no_cage_metrics),
        **flatten_metrics("cage_kept", kept_metrics),
        **flatten_metrics("with_review", corrected_metrics),
        **flatten_metrics("gain_per_review", gain_per_review_metrics),
    }

    save_result(example=pipeline, method=type(cage).__name__, **record)
    return record


# (record key suffix, display label) -- mirrors the metric table the example
# notebooks print for each scenario.
METRIC_LABELS = [
    ("accuracy", "accuracy"),
    ("precision_macro", "precision (macro)"),
    ("recall_macro", "recall (macro)"),
    ("f1_macro", "f1 (macro)"),
    ("MCC", "MCC"),
]


def format_summary(record):
    """Format both notebook scenarios against the no-cage baseline.

    Scenario 1, "cage-kept": abstain on everything flagged, trust the rest.
    Scenario 2, "with-review": an expert reviews every flagged sample and
    corrects the ones that were really wrong. "gain/review" is the
    with-review gain over no-cage, normalized by the fraction flagged --
    lets you compare methods that flag very different amounts.
    """
    lines = [
        f"  AUROC={record['auroc']:.4f}  threshold={record['threshold']:.4g}",
        f'  {"metric":<20}{"no-cage":>12}{"cage-kept":>12}{"with-review":>12}{"gain/review":>13}',
    ]
    for key, label in METRIC_LABELS:
        lines.append(
            f'  {label:<20}{record[f"no_cage_{key}"]:>12.4f}'
            f'{record[f"cage_kept_{key}"]:>12.4f}'
            f'{record[f"with_review_{key}"]:>12.4f}'
            f'{record[f"gain_per_review_{key}"]:>13.4f}'
        )
    lines.append(
        f"  flagged={record['flagged']}/{record['test_samples']}"
        f"  errors caught={record['errors_caught']}/{record['total_errors']}"
        f"  errors missed={record['errors_missed']}/{record['total_errors']}"
        f"  coverage={record['coverage']:.2%}"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pipeline", nargs="+", required=True,
        help=f"Pipeline name(s), or 'all'. Registered: {sorted(PIPELINES)}",
    )
    parser.add_argument(
        "--safetycage", nargs="+", required=True,
        help=f"Safety cage name(s), or 'all'. Registered: {sorted(SAFETYCAGES)}",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric", default="MCC", choices=sorted(METRIC_FUNCTIONS))
    parser.add_argument(
        "--retrain", action="store_true",
        help="Train a fresh model from scratch instead of loading its checkpoint.",
    )
    parser.add_argument(
        "--cage-kwargs", default=None,
        help="JSON dict of constructor kwarg overrides applied to every requested "
        "safety cage, e.g. red's epoch count: "
        "'{\"training_iterations\": 20, \"num_inducing_points\": 100}'. "
        "Only override kwargs the requested method(s) actually accept -- mixing "
        "--safetycage red msp with red-only kwargs will fail msp's constructor.",
    )
    args = parser.parse_args()

    pipelines = sorted(PIPELINES) if args.pipeline == ["all"] else args.pipeline
    safetycages = sorted(SAFETYCAGES) if args.safetycage == ["all"] else args.safetycage

    for flag, names, registry in (("--pipeline", pipelines, PIPELINES), ("--safetycage", safetycages, SAFETYCAGES)):
        unknown = [n for n in names if n not in registry]
        if unknown:
            parser.error(f"unknown {flag} value(s) {unknown}; registered: {sorted(registry)}")

    if args.cage_kwargs is not None:
        try:
            cage_kwargs = json.loads(args.cage_kwargs)
        except json.JSONDecodeError as e:
            parser.error(f"--cage-kwargs is not valid JSON: {e}")
        if not isinstance(cage_kwargs, dict):
            parser.error("--cage-kwargs must be a JSON object, e.g. '{\"training_iterations\": 20}'")
    else:
        cage_kwargs = None

    args.pipeline, args.safetycage = pipelines, safetycages
    metric_fn = METRIC_FUNCTIONS[args.metric]
    pipeline_kwargs = {"retrain": True} if args.retrain else None
    combinations = [(p, s) for p in args.pipeline for s in args.safetycage]
    succeeded = 0

    log_path = _setup_logging()
    logger.info(f"Logging to {log_path}")

    for pipeline, safetycage in combinations:
        logger.info(f"=== {pipeline} / {safetycage} ===")
        try:
            record = run_experiment(
                pipeline, safetycage, device=args.device, metric_fn=metric_fn,
                cage_kwargs=cage_kwargs, pipeline_kwargs=pipeline_kwargs,
            )
            logger.info(format_summary(record))
            succeeded += 1
        except Exception:
            logger.info("  FAILED:")
            logger.info(traceback.format_exc())

    logger.info(f"\n{succeeded} of {len(combinations)} combinations succeeded.")
    logger.info(f"Results: {Path(__file__).resolve().parent / 'results.json'}")
    logger.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
