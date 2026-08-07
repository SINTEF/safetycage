"""Shared helpers for the safetycage examples."""
import sys
from pathlib import Path


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
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src" / "safetycage").is_dir():
            src = str(candidate / "src")
            sys.path.insert(0, src)
            return src

    return None
