"""Shared helpers for the safetycage examples."""
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src" / "safetycage").is_dir():
            return candidate

    raise FileNotFoundError("Could not locate the repository root from the current directory.")


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    """Flatten a ``{"metric name": value}`` dict into dataframe-friendly columns.

    ``{"precision (macro)": 0.9}`` with ``prefix="baseline"`` becomes
    ``{"baseline_precision_macro": 0.9}`` -- nested dicts survive a JSON
    round-trip fine, but turn into object columns in a dataframe instead of
    one column per metric.
    """
    return {
        f"{prefix}_{name.replace(' ', '_').replace('(', '').replace(')', '')}": value
        for name, value in metrics.items()
    }


def save_result(example: str, method: str, **fields) -> Path:
    """Upsert one result record into ``examples/results.json``, keyed by (example, method).

    Every example notebook can call this to contribute a row to a shared
    comparison table across misclassification-detection methods. Re-running
    a notebook replaces its own record rather than duplicating it. Load the
    table later with ``pandas.DataFrame(json.loads(Path("examples/results.json").read_text()))``.
    """
    path = _repo_root() / "examples" / "results.json"

    results = json.loads(path.read_text()) if path.exists() else []
    results = [r for r in results if not (r["example"] == example and r["method"] == method)]
    results.append({"example": example, "method": method, **fields})

    path.write_text(json.dumps(results, indent=2))
    return path


def fix_pythonpath_if_working_locally():
    """Make ``import safetycage`` resolve to this repository's ``src/``.

    ``uv sync`` installs safetycage in editable mode, so this is a no-op for
    anyone following the README. It matters for the reader who opens a
    notebook without syncing while a released safetycage is installed
    elsewhere — without it they would silently exercise the published version
    instead of the working tree.

    Walks up from the current directory rather than checking for a fixed
    directory name, so it works from ``examples/``, ``examples/01-mnist/``, or
    anywhere else inside the clone.

    Returns:
        The path inserted onto ``sys.path``, or None if no checkout was found.
    """
    try:
        root = _repo_root()
    except FileNotFoundError:
        return None

    src = str(root / "src")
    sys.path.insert(0, src)
    return src
