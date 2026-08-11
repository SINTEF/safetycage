"""Sphinx configuration for the safetycage documentation."""
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Document the working tree rather than any installed release.
sys.path.insert(0, str(REPO_ROOT / "src"))

with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
    _pyproject = tomllib.load(fh)["project"]

# -- Project information -----------------------------------------------------

project = "safetycage"
author = ", ".join(a["name"] for a in _pyproject["authors"])
copyright = "2026, SINTEF"

# Read from pyproject so a release bump never leaves the docs stale.
release = _pyproject["version"]
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # reads the Google-style docstrings used throughout
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",  # Markdown pages (superset of myst_parser) plus rendered notebooks
]

# The MNIST tutorial notebook is already committed with its outputs; re-running
# it here would need torch/torchvision in the docs build for no benefit.
nb_execution_mode = "off"

exclude_patterns = [
    "_build",
    # Working documents for agentic development, not user documentation.
    "superpowers/**",
]

# RED needs torch and gpytorch (behind the `red` extra); SPARDACUS needs
# sklearn/scipy/statsmodels/tqdm and Mahalanobis needs scipy/statsmodels
# (behind the `spardacus`/`mahalanobis` extras). Mocking all of them keeps
# the docs buildable without installing any extra — Read the Docs only
# installs the `docs` dependency group.
autodoc_mock_imports = ["torch", "gpytorch", "sklearn", "scipy", "statsmodels", "tqdm"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    # The methods safetycage actually calls on a ModelModule are underscore
    # -prefixed (_get_probabilities, _get_activations). They are part of the
    # integration contract, so they belong in the reference.
    "private-members": "_get_predictions,_get_probabilities,_get_activations,_get_pre_activations",
}
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"safetycage {release}"
