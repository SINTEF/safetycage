"""Local result storage for experiments/, decoupled from examples/results.json.

Same upsert-by-(example, method) shape as examples/utils.py's save_result,
kept as a separate small file so experiments/ doesn't need to reach into
examples/ just to record its own results.
"""
import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    """Flatten a ``{"metric name": value}`` dict into flat, dataframe-friendly keys.

    ``{"precision (macro)": 0.9}`` with ``prefix="cage_kept"`` becomes
    ``{"cage_kept_precision_macro": 0.9}``.
    """
    return {
        f"{prefix}_{name.replace(' ', '_').replace('(', '').replace(')', '')}": value
        for name, value in metrics.items()
    }


def save_result(example: str, method: str, **fields) -> Path:
    """Upsert one result record into ``experiments/results.json``, keyed by (example, method).

    Re-running a combination replaces its own record rather than duplicating
    it. Load the table later with
    ``pandas.DataFrame(json.loads(Path("experiments/results.json").read_text()))``.
    """
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    results = [r for r in results if not (r["example"] == example and r["method"] == method)]
    results.append({"example": example, "method": method, **fields})

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    return RESULTS_PATH
