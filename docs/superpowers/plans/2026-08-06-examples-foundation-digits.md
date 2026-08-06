# Examples Restructure — Plan 1: Foundation + digits

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `examples/` in safetycage-pypi with a complete, tested digits example that proves the structure before any PyTorch porting begins.

**Architecture:** Each example is a directory holding an importable `modules.py` (the `DataModule`/`ModelModule` subclasses — the integration contract being taught), a `train.py` (model fitting boilerplate, kept out of the notebook), and a notebook carrying narration plus safetycage usage. A shared `examples/utils.py` makes `import safetycage` resolve to the working tree. Tests import `modules.py` by file path and exercise it end-to-end.

**Tech Stack:** Python 3.11.7, scikit-learn (already a core dependency), pytest, jupyter, uv.

## Why digits and not iris

The spec named iris. Implementation measurement showed iris cannot demonstrate
what this example exists to demonstrate: with 150 samples it yields a
24-sample validation split containing **1 misclassification**, so threshold
selection optimises a metric computed from a single positive. The plan was
changed to `sklearn.datasets.load_digits` on that evidence:

| dataset | samples | classes | val samples | val errors |
| --- | --- | --- | --- | --- |
| iris | 150 | 3 | 24 | 1 |
| wine | 178 | 3 | 29 | 0 |
| digits | 1797 | 10 | 288 | 33 |

digits ships inside scikit-learn exactly as iris does, so this costs no
dependency. It is also 8×8 handwritten digits with 10 classes, which makes it a
progression into the mnist example rather than a jump.

## Global Constraints

- Target package version is **0.0.54**. Do not pin examples to older versions.
- **digits adds no new runtime dependency.** scikit-learn, numpy and matplotlib
  are already core dependencies, and `load_digits()` is bundled and offline.
- Notebook tooling goes in a **PEP 735 `[dependency-groups]` table**, never
  `[project.optional-dependencies]`. Extras are advertised in the published
  wheel; dependency groups are not.
- **No `pyrootutils`, no `.project-root`.** Those exist only because the
  tutorials lived outside the package repo.
- `DataModule.__init__` calls `Path(data_dir)` unconditionally, so
  **`data_dir=None` raises `TypeError`**. Every concrete `DataModule` must
  supply a real default.
- `SafetyCage.find_best_threshold(y_true, y_probs, metric_fn, greater_is_better=True)`
  is a **method that takes no `leq` argument** — it reads `self.leq` via
  `self.flag()`. The separate module-level
  `safetycage.utils.evaluate.find_best_threshold(...)` *does* take `leq`. Do not
  confuse them.
- `pvalues_` is a **notebook variable naming convention, not a package
  attribute.** It does not exist in `src/`.
- The classifier is deliberately constrained to `max_depth=3`. An unconstrained
  RandomForest reaches 97.6% and leaves only 7 validation errors, which is too
  few to threshold on. This is stated plainly in the notebook rather than hidden.
- CI is out of scope. Tests are run manually with `pytest`.

**Measured reference values** (RandomForest, `n_estimators=100`, `max_depth=3`,
`random_state=42`, splits 1149/288/360): validation accuracy 88.5%, 33
validation errors, `alpha_opt` ≈ 0.235, MCC ≈ 0.442, and on test it catches 34
of 47 misclassifications while flagging 89 samples. Use these to sanity-check
your run; small deviations from library version differences are fine, but a
wildly different number means something is wrong.

---

### Task 1: Repair the red test baseline

The existing suite fails before any of our work starts, which makes every later
"expect PASS" step unverifiable.

**Files:**
- Modify: `tests/test_example.py:9-11`

**Interfaces:**
- Consumes: nothing.
- Produces: a green `pytest tests/` baseline that all later tasks depend on.

- [ ] **Step 1: Reproduce the failure**

Run: `uv run --with pytest pytest tests/ -q`
Expected: FAIL — `assert '0.0.54' == '0.0.2'` in `test_check_package_version`.

- [ ] **Step 2: Make the assertion track pyproject instead of a frozen literal**

Replace the body of `test_check_package_version` in `tests/test_example.py`:

```python
def test_check_package_version():
    """The installed version must match the version declared in pyproject.toml."""
    import tomllib
    from importlib.metadata import version
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]

    assert version("safetycage") == declared
```

This stops the test needing a manual edit on every release, which is why it
rotted in the first place.

- [ ] **Step 3: Verify it passes**

Run: `uv run --with pytest pytest tests/ -q`
Expected: PASS — 2 passed, 1 skipped, 0 failed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_example.py
git commit -m "Assert package version against pyproject rather than a literal

The test pinned '0.0.2' while the package had moved to 0.0.54, so the
suite was red. Reading the declared version removes the manual edit
that was being forgotten each release."
```

---

### Task 2: `examples/utils.py` path helper

**Files:**
- Create: `examples/utils.py`
- Create: `tests/test_examples.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `fix_pythonpath_if_working_locally() -> str | None` — inserts the
  repo's `src/` at `sys.path[0]` and returns the inserted path, or returns
  `None` if no `src/safetycage` was found in any parent. Every example notebook
  calls this in its first cell.

- [ ] **Step 1: Write the failing test**

Create `tests/test_examples.py`:

```python
"""Tests for the worked examples under examples/.

These load each example's modules.py by file path, because examples/ is
deliberately not an importable package - each example directory must be
copyable on its own.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"


def load_module_by_path(path: Path, name: str):
    """Import a .py file that is not on sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "start",
    ["examples", "examples/digits", "."],
    ids=["from-examples", "from-example-subdir", "from-repo-root"],
)
def test_fix_pythonpath_resolves_src_from_any_depth(start, monkeypatch):
    """The helper must work regardless of how deeply the notebook is nested.

    This is the exact bug in the darts original it was adapted from: its
    `basename(cwd) == "examples"` guard silently no-ops one level down.
    """
    utils = load_module_by_path(EXAMPLES / "utils.py", "examples_utils")

    target = REPO_ROOT / start
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(target)
    monkeypatch.setattr(sys, "path", list(sys.path))

    result = utils.fix_pythonpath_if_working_locally()

    assert result == str(REPO_ROOT / "src")
    assert sys.path[0] == str(REPO_ROOT / "src")


def test_fix_pythonpath_returns_none_outside_a_clone(tmp_path, monkeypatch):
    """Outside a checkout there is no src/safetycage, so it must not guess."""
    utils = load_module_by_path(EXAMPLES / "utils.py", "examples_utils")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))

    assert utils.fix_pythonpath_if_working_locally() is None
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run --with pytest pytest tests/test_examples.py -q`
Expected: FAIL — `FileNotFoundError`, because `examples/utils.py` does not exist yet.

- [ ] **Step 3: Write the helper**

Create `examples/utils.py`:

```python
"""Shared helpers for the safetycage examples."""
import sys
from pathlib import Path


def fix_pythonpath_if_working_locally():
    """Make ``import safetycage`` resolve to this repository's ``src/``.

    Running ``uv sync`` installs safetycage in editable mode, so this is a
    no-op for anyone following the README. It matters for the reader who
    opens a notebook without syncing while a released safetycage is
    installed elsewhere - without it they would silently exercise the
    published version instead of the working tree.

    Walks up from the current directory rather than checking for a fixed
    directory name, so it works from ``examples/``, ``examples/digits/``, or
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
```

- [ ] **Step 4: Verify it passes**

Run: `uv run --with pytest pytest tests/test_examples.py -q`
Expected: PASS — 4 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/utils.py tests/test_examples.py
git commit -m "Add examples path helper resolving safetycage to the working tree

Adapted from darts, corrected for this repo's src layout and for
notebooks nested one level below examples/. Tested at all three depths a
notebook can realistically run from."
```

---

### Task 3: digits `DataModule`

Structure ported from `safetycage-tutorials/iris/modules/sklearn_iris_datamodule.py`,
with `pyrootutils` removed, a working `data_dir` default, the npz caching
dropped (`load_digits()` is bundled and offline, so the cache was ceremony), and
the dataset changed to digits for the reason given at the top of this plan.

**Files:**
- Create: `examples/digits/modules.py`
- Modify: `tests/test_examples.py` (append)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `DigitsDataModule(data_dir=None, from_cache=True, batch_size=32, val_split=0.2, test_split=0.2, use_onehot_encoder=False, standardize=True, random_state=42, device="cpu")` with attributes `data_train`, `data_val`, `data_test`, each a `(x, y)` tuple of `np.ndarray`; properties `classes -> dict[int, str]`, `num_classes -> int`, `dataset_name -> str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_examples.py`:

```python
DIGITS = EXAMPLES / "digits"


@pytest.fixture(scope="module")
def digits_modules():
    return load_module_by_path(DIGITS / "modules.py", "digits_modules")


def test_datamodule_builds_three_disjoint_splits(digits_modules):
    dm = digits_modules.DigitsDataModule()

    assert dm.num_classes == 10
    assert dm.dataset_name == "digits"
    assert set(dm.classes) == set(range(10))

    sizes = [len(split[0]) for split in (dm.data_train, dm.data_val, dm.data_test)]
    assert sum(sizes) == 1797, "digits has 1797 samples; splits must partition it"
    assert sizes == [1149, 288, 360]

    for x, y in (dm.data_train, dm.data_val, dm.data_test):
        assert x.ndim == 2 and x.shape[1] == 64, "8x8 images flattened to 64 features"
        assert y.ndim == 1, "labels stay 1D unless one-hot is explicitly requested"
        assert len(x) == len(y)


def test_datamodule_does_not_require_a_data_dir(digits_modules):
    """DataModule.__init__ calls Path(data_dir) unconditionally, so a None
    default would raise TypeError. The subclass must supply a real path."""
    dm = digits_modules.DigitsDataModule(data_dir=None)
    assert dm.data_dir.is_dir()


def test_datamodule_onehot_is_opt_in(digits_modules):
    dm = digits_modules.DigitsDataModule(use_onehot_encoder=True)
    y = dm.data_train[1]
    assert y.ndim == 2 and y.shape[1] == 10
    assert (y.sum(axis=1) == 1).all()


def test_datamodule_scaler_is_fitted_on_training_data_only(digits_modules):
    """Fitting the scaler before splitting would leak test statistics into
    training. Training data must therefore be the split centred on zero."""
    dm = digits_modules.DigitsDataModule()
    train_mean = abs(dm.data_train[0].mean())
    test_mean = abs(dm.data_test[0].mean())
    assert train_mean < 1e-9, "training split should be centred by its own scaler"
    assert test_mean > train_mean, "test split must not have been used to fit"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run --with pytest pytest tests/test_examples.py -q`
Expected: FAIL — `examples/digits/modules.py` does not exist.

- [ ] **Step 3: Write `examples/digits/modules.py`**

```python
"""Data and model modules for the digits example.

This file is the integration contract. Adopting safetycage means writing
your own version of these two classes for your dataset and model:

  * ``DataModule``  - declares how data is loaded, transformed and split,
    and what the classes are.
  * ``ModelModule`` - declares how to get predictions, probabilities and
    intermediate activations out of your model.

Everything safetycage needs, it gets through these two objects.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from safetycage.datamodule import DataModule
from safetycage.modelmodule import ModelModule

# DataModule.__init__ calls Path(data_dir) with no None check, so we must
# supply a concrete default rather than passing None through.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


class DigitsDataModule(DataModule):
    """DataModule for the 8x8 handwritten digits bundled with scikit-learn."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        from_cache: bool = True,
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.2,
        use_onehot_encoder: bool = False,
        standardize: bool = True,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(data_dir or DEFAULT_DATA_DIR, from_cache, batch_size, device)

        self.val_split = val_split
        self.test_split = test_split
        self.use_onehot_encoder = use_onehot_encoder
        self.standardize = standardize
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.image_shape = (8, 8)

        self.setup()

    @property
    def classes(self) -> Dict[int, str]:
        return {digit: str(digit) for digit in range(10)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def dataset_name(self) -> str:
        return "digits"

    def setup(self) -> None:
        """Load, split, then transform.

        Order matters. The scaler is fitted on the training split only, so
        that validation and test statistics never leak into it. Splitting
        first is what makes that possible.
        """
        x, y = self._load_data(self.data_dir / self.dataset_name)

        x_train_val, y_train_val, x_test, y_test = self._split(x, y, self.test_split)
        x_train, y_train, x_val, y_val = self._split(x_train_val, y_train_val, self.val_split)

        x_train, y_train = self._transform(x_train, y_train, fit_scaler=True)
        x_val, y_val = self._transform(x_val, y_val, fit_scaler=False)
        x_test, y_test = self._transform(x_test, y_test, fit_scaler=False)

        self.data_train = (x_train, y_train)
        self.data_val = (x_val, y_val)
        self.data_test = (x_test, y_test)

    def _load_data(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """digits ships inside scikit-learn, so there is nothing to download
        or cache. ``filepath`` is accepted to satisfy the base class."""
        dataset = load_digits()
        return dataset.data.astype(np.float64), dataset.target.astype(np.int64)

    def _split(self, x, y, split):
        """Stratified split. Labels are still 1D here - see _transform."""
        x_a, x_b, y_a, y_b = train_test_split(
            x, y, stratify=y, random_state=self.random_state, test_size=split
        )
        return x_a, y_a, x_b, y_b

    def _transform(self, x, y, fit_scaler: bool = False):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if self.standardize:
            if fit_scaler or self.scaler is None:
                self.scaler = StandardScaler()
                x = self.scaler.fit_transform(x)
            else:
                x = self.scaler.transform(x)

        if self.use_onehot_encoder:
            return x, np.eye(self.num_classes, dtype=np.float64)[y]

        return x, y

    def set_predictions(self, predictions: Dict[str, np.ndarray]) -> None:
        """Attach model predictions, widening each split to (x, y, y_pred)."""
        required = ("y_pred_train", "y_pred_val", "y_pred_test")
        missing = [key for key in required if predictions.get(key) is None]
        if missing:
            raise ValueError(f"predictions is missing required keys: {missing}")

        self.data_train = (*self.data_train[:2], predictions["y_pred_train"])
        self.data_val = (*self.data_val[:2], predictions["y_pred_val"])
        self.data_test = (*self.data_test[:2], predictions["y_pred_test"])

    def _default_joblib_path(self) -> Path:
        return (self.data_dir / f"{self.dataset_name}_data_module").with_suffix(".joblib")

    def to_joblib(self, path: Optional[str] = None):
        path = Path(path or self._default_joblib_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def from_joblib(self, path: Optional[str] = None):
        return joblib.load(Path(path or self._default_joblib_path()))
```

- [ ] **Step 4: Verify it passes**

Run: `uv run --with pytest pytest tests/test_examples.py -q`
Expected: PASS — 8 passed.

- [ ] **Step 5: Ignore the generated data directory**

Append to `.gitignore`:

```
examples/*/data/
```

- [ ] **Step 6: Commit**

```bash
git add examples/digits/modules.py tests/test_examples.py .gitignore
git commit -m "Add digits DataModule to examples

Structure ported from the tutorials' iris module with pyrootutils
removed, a concrete data_dir default (the base class calls Path(data_dir)
with no None check), and the npz cache dropped since load_digits() is
bundled and offline. One-hot is now opt-in rather than on by default."
```

---

### Task 4: digits `ModelModule`

**Files:**
- Modify: `examples/digits/modules.py` (append)
- Modify: `tests/test_examples.py` (append)

**Interfaces:**
- Consumes: `DigitsDataModule` from Task 3.
- Produces: `DigitsModelModule(selected_layers, use_onehot_encoder, model, **kwargs)` implementing `_get_probabilities(x) -> np.ndarray`, `_get_predictions(x) -> np.ndarray`, `_get_activations(x) -> Dict[str, np.ndarray]`, `_get_pre_activations(x) -> Dict[str, np.ndarray]`, `_calc_model_shape() -> Dict[str, int]`. Valid `selected_layers` values: `"input"`, `"probabilities"`, `"log_probabilities"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_examples.py`:

```python
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="module")
def fitted_digits(digits_modules):
    """A DataModule plus a ModelModule wrapping a fitted RandomForest."""
    dm = digits_modules.DigitsDataModule()
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(*dm.data_train)
    mm = digits_modules.DigitsModelModule(
        selected_layers=["probabilities"],
        use_onehot_encoder=False,
        model=clf,
    )
    return dm, mm


def test_modelmodule_probabilities_are_a_simplex(fitted_digits):
    dm, mm = fitted_digits
    x_val = dm.data_val[0]

    probs = mm._get_probabilities(x_val)

    assert probs.shape == (len(x_val), 10)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_modelmodule_predictions_are_class_indices(fitted_digits):
    dm, mm = fitted_digits
    preds = mm._get_predictions(dm.data_val[0])

    assert preds.ndim == 1
    assert set(np.unique(preds)) <= set(range(10))


def test_modelmodule_activations_return_only_selected_layers(fitted_digits):
    dm, mm = fitted_digits
    activations = mm._get_activations(dm.data_val[0][:5])

    assert set(activations) == {"probabilities"}
    assert activations["probabilities"].shape == (5, 10)


def test_modelmodule_log_probabilities_are_finite(fitted_digits, digits_modules):
    """A zero probability would give -inf without clipping, and a constrained
    forest produces plenty of zeros across 10 classes."""
    dm, _ = fitted_digits
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(*dm.data_train)
    mm = digits_modules.DigitsModelModule(
        selected_layers=["log_probabilities"],
        use_onehot_encoder=False,
        model=clf,
    )

    logs = mm._get_activations(dm.data_val[0])["log_probabilities"]
    assert np.isfinite(logs).all()


def test_modelmodule_rejects_unknown_layers(digits_modules, fitted_digits):
    _, mm = fitted_digits
    with pytest.raises(ValueError, match="Unsupported selected_layers"):
        digits_modules.DigitsModelModule(
            selected_layers=["conv1"],
            use_onehot_encoder=False,
            model=mm.model,
        )
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run --with pytest pytest tests/test_examples.py -q`
Expected: FAIL — `AttributeError: module 'digits_modules' has no attribute 'DigitsModelModule'`.

- [ ] **Step 3: Append `DigitsModelModule` to `examples/digits/modules.py`**

```python
class DigitsModelModule(ModelModule):
    """ModelModule for a scikit-learn classifier on digits.

    A tree ensemble has no hidden layers, so "activations" here means the
    representations safetycage can actually threshold: the inputs, the
    class probabilities, and their logs. A neural network's ModelModule
    would return real layer outputs instead - see the mnist example.
    """

    AVAILABLE_LAYERS = ("input", "probabilities", "log_probabilities")

    def __init__(
        self,
        selected_layers: List[str],
        use_onehot_encoder: bool,
        model: Any,
        **kwargs,
    ):
        super().__init__(selected_layers, use_onehot_encoder, model, **kwargs)

        invalid = [n for n in self.selected_layers if n not in self.AVAILABLE_LAYERS]
        if invalid:
            raise ValueError(
                f"Unsupported selected_layers: {invalid}. "
                f"Choose from {sorted(self.AVAILABLE_LAYERS)}."
            )

        self.model_shape = self._calc_model_shape()
        self.last_layer = kwargs.get("last_layer", self.selected_layers[-1])

    @staticmethod
    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x.reshape(1, -1) if x.ndim == 1 else x

    def _get_probabilities(self, x: np.ndarray) -> np.ndarray:
        """Not abstract on the base class, but MSP calls it - so any
        ModelModule used with MSP must provide it."""
        return np.asarray(
            self.model.predict_proba(self._ensure_2d(x)), dtype=np.float64
        )

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        predicted = np.asarray(self.model.predict(self._ensure_2d(x)), dtype=np.int64)

        if self.use_onehot_encoder:
            return np.eye(len(self.model.classes_), dtype=np.float64)[predicted]

        return predicted

    def _get_activations(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        x = self._ensure_2d(x)
        probabilities = self._get_probabilities(x)

        available = {
            "input": x,
            "probabilities": probabilities,
            # clipped so a zero probability becomes a large negative number
            # rather than -inf; with 10 classes and a shallow forest, zeros
            # are common
            "log_probabilities": np.log(np.clip(probabilities, 1e-12, 1.0)),
        }

        return {name: available[name] for name in self.selected_layers}

    def _get_pre_activations(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """A tree ensemble has no pre-activations. Returning the same
        representation keeps the interface satisfied."""
        return self._get_activations(x)

    def _calc_model_shape(self) -> Dict[str, int]:
        n_features = int(getattr(self.model, "n_features_in_", 64))
        n_classes = len(getattr(self.model, "classes_", tuple(range(10))))

        return {
            "input": n_features,
            "probabilities": n_classes,
            "log_probabilities": n_classes,
        }
```

- [ ] **Step 4: Verify it passes**

Run: `uv run --with pytest pytest tests/test_examples.py -q`
Expected: PASS — 13 passed.

- [ ] **Step 5: Commit**

```bash
git add examples/digits/modules.py tests/test_examples.py
git commit -m "Add digits ModelModule to examples

Wraps a scikit-learn classifier. Documents why a tree ensemble's
'activations' are inputs/probabilities/log-probabilities rather than
layer outputs, and that _get_probabilities must exist for MSP even
though the base class does not declare it abstract."
```

---

### Task 5: digits `train.py` and the end-to-end MSP check

**Files:**
- Create: `examples/digits/train.py`
- Modify: `tests/test_examples.py` (append)

**Interfaces:**
- Consumes: `DigitsDataModule`, `DigitsModelModule` from Tasks 3–4.
- Produces: `train_model(data_module, n_estimators=100, max_depth=3, random_state=42) -> RandomForestClassifier` and `build_modules(**data_module_kwargs) -> tuple[DigitsDataModule, DigitsModelModule]`. The notebook calls `build_modules()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_examples.py`:

```python
def test_train_module_builds_a_usable_pair():
    train = load_module_by_path(DIGITS / "train.py", "digits_train")
    dm, mm = train.build_modules()

    predictions = mm._get_predictions(dm.data_val[0])
    accuracy = (predictions == dm.data_val[1]).mean()
    errors = int((predictions != dm.data_val[1]).sum())

    assert 0.80 < accuracy < 0.95, (
        f"{accuracy:.1%}: the model is deliberately constrained so that it "
        "makes mistakes worth detecting. Too high means max_depth was raised."
    )
    assert errors >= 20, (
        f"only {errors} validation errors: too few to select a threshold on"
    )


def test_msp_end_to_end_separates_correct_from_incorrect():
    """The whole point of the example: MSP scores should be systematically
    lower for predictions the model got wrong."""
    from safetycage.methods.msp import MSP
    from safetycage.utils.evaluate import MCC

    train = load_module_by_path(DIGITS / "train.py", "digits_train")
    dm, mm = train.build_modules()

    msp = MSP(model_module=mm, data_module=dm)
    msp.train_cage()

    x_val, y_val = dm.data_val
    statistics = msp.predict(x_val, y_val)
    incorrect = (mm._get_predictions(x_val) != y_val).astype(int)

    assert statistics.shape == (len(x_val),)
    assert ((statistics >= 0) & (statistics <= 1)).all(), "MSP returns max softmax"

    assert statistics[incorrect == 1].mean() < statistics[incorrect == 0].mean(), (
        "MSP is uninformative here: wrong predictions are not less confident"
    )

    # find_best_threshold is a METHOD here and takes no leq argument; it
    # reads self.leq via self.flag(). The module-level helper in
    # safetycage.utils.evaluate is a different function that does take leq.
    result = msp.find_best_threshold(
        y_true=incorrect,
        y_probs=statistics,
        metric_fn=MCC,
    )

    assert set(result) == {"alpha_opt", "metric_max"}
    assert 0.0 <= result["alpha_opt"] <= 1.0
    assert float(result["metric_max"]) > 0.25, (
        f"MCC {float(result['metric_max']):.3f} is near chance; expected ~0.44"
    )


def test_msp_threshold_catches_most_test_errors():
    """Applying the validation-chosen threshold to held-out data is the only
    honest estimate of how the cage will behave."""
    from safetycage.methods.msp import MSP
    from safetycage.utils.evaluate import MCC

    train = load_module_by_path(DIGITS / "train.py", "digits_train")
    dm, mm = train.build_modules()

    msp = MSP(model_module=mm, data_module=dm)
    msp.train_cage()

    x_val, y_val = dm.data_val
    incorrect_val = (mm._get_predictions(x_val) != y_val).astype(int)
    alpha = msp.find_best_threshold(
        y_true=incorrect_val,
        y_probs=msp.predict(x_val, y_val),
        metric_fn=MCC,
    )["alpha_opt"]

    x_test, y_test = dm.data_test
    flagged = msp.flag(msp.predict(x_test, y_test), alpha)
    incorrect_test = (mm._get_predictions(x_test) != y_test).astype(int)

    caught = int((flagged & (incorrect_test == 1)).sum())
    recall = caught / incorrect_test.sum()

    assert recall > 0.5, f"caught only {caught} of {incorrect_test.sum()} test errors"
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `uv run --with pytest pytest tests/test_examples.py -q`
Expected: FAIL — `examples/digits/train.py` does not exist.

- [ ] **Step 3: Write `examples/digits/train.py`**

```python
"""Model training for the digits example.

Deliberately separate from modules.py. Nothing here is specific to
safetycage - it is ordinary scikit-learn - which is exactly why it is kept
out of the notebook.
"""
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier

from modules import DigitsDataModule, DigitsModelModule

# An unconstrained forest reaches ~97.6% on digits, leaving only 7 errors in
# the validation split - too few to choose a threshold on. Capping the depth
# gives a model that is wrong often enough for misclassification detection to
# be worth demonstrating.
DEFAULT_MAX_DEPTH = 3


def train_model(
    data_module: DigitsDataModule,
    n_estimators: int = 100,
    max_depth: int = DEFAULT_MAX_DEPTH,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit a RandomForest on the training split."""
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    classifier.fit(*data_module.data_train)
    return classifier


def build_modules(**data_module_kwargs) -> Tuple[DigitsDataModule, DigitsModelModule]:
    """Build a fitted (DataModule, ModelModule) pair ready for safetycage."""
    data_module = DigitsDataModule(**data_module_kwargs)
    classifier = train_model(data_module)

    model_module = DigitsModelModule(
        selected_layers=["probabilities"],
        use_onehot_encoder=data_module.use_onehot_encoder,
        model=classifier,
    )

    return data_module, model_module


if __name__ == "__main__":
    dm, mm = build_modules()
    predictions = mm._get_predictions(dm.data_val[0])
    accuracy = (predictions == dm.data_val[1]).mean()
    errors = int((predictions != dm.data_val[1]).sum())
    print(f"validation accuracy : {accuracy:.1%}")
    print(f"validation errors   : {errors} of {len(dm.data_val[1])}")
```

Note: `from modules import ...` works because `load_module_by_path` and Jupyter
both put the file's own directory on `sys.path`. Task 6's notebook relies on the
same behaviour.

- [ ] **Step 4: Verify it passes**

Run: `uv run --with pytest pytest tests/test_examples.py -q`
Expected: PASS — 16 passed.

- [ ] **Step 5: Verify the script runs standalone**

Run: `cd examples/digits && uv run python train.py`
Expected:
```
validation accuracy : 88.5%
validation errors   : 33 of 288
```

- [ ] **Step 6: Commit**

```bash
git add examples/digits/train.py tests/test_examples.py
git commit -m "Add digits training entrypoint and end-to-end MSP tests

train.py holds the scikit-learn boilerplate so the notebook does not, and
caps tree depth so the model errs often enough for detection to be worth
demonstrating. Tests assert MSP separates correct from incorrect
predictions and that the validation threshold generalises to test."
```

---

### Task 6: digits notebook

Built with `nbformat` rather than by hand so the result is reproducible and
reviewable as code.

**Files:**
- Create: `examples/digits/build_notebook.py` (temporary, deleted in Step 5)
- Create: `examples/digits/digits_example.ipynb`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `build_modules` from Task 5, `fix_pythonpath_if_working_locally` from Task 2.
- Produces: `examples/digits/digits_example.ipynb` with committed outputs.

- [ ] **Step 1: Add the dependency group**

Append to `pyproject.toml`, after the `[project.optional-dependencies]` table:

```toml
[dependency-groups]
examples = [
    "ipykernel>=7",
    "jupyter>=1.1",
    "nbformat>=5.10",
]
```

A PEP 735 dependency group, not an extra: `red` is an extra because RED is a
runtime package feature needing torch, whereas notebook tooling is local-only
and must stay out of the published wheel's metadata.

Run: `uv sync --group examples`
Expected: resolves and installs without adding anything to `[project] dependencies`.

- [ ] **Step 2: Write the notebook builder**

Create `examples/digits/build_notebook.py`:

```python
"""Generate digits_example.ipynb. Run once, then delete."""
import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Detecting misclassifications with safetycage\n"
        "\n"
        "A classifier that is wrong and *knows* it is wrong is far more useful\n"
        "than one that is wrong confidently. safetycage flags predictions a\n"
        "model is likely to have got wrong, so they can be reviewed rather\n"
        "than trusted.\n"
        "\n"
        "This example uses the 8x8 handwritten digits bundled with\n"
        "scikit-learn, so it needs no dependency beyond safetycage's own and\n"
        "downloads nothing.\n"
        "\n"
        "**What you have to write to adopt safetycage** is two classes: a\n"
        "`DataModule` describing your data, and a `ModelModule` describing how\n"
        "to get predictions and probabilities out of your model. Both live in\n"
        "`modules.py` next to this notebook, and we read them below."
    ),
    nbf.v4.new_code_cell(
        "from utils import fix_pythonpath_if_working_locally\n"
        "\n"
        "# Resolves `import safetycage` to this repo's src/ if you opened this\n"
        "# notebook without running `uv sync` first.\n"
        "fix_pythonpath_if_working_locally()"
    ),
    nbf.v4.new_markdown_cell("## The data"),
    nbf.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "from sklearn.datasets import load_digits\n"
        "\n"
        "raw = load_digits()\n"
        "fig, axes = plt.subplots(2, 8, figsize=(10, 3))\n"
        "for ax, image, label in zip(axes.ravel(), raw.images, raw.target):\n"
        "    ax.imshow(image, cmap='gray_r')\n"
        "    ax.set_title(str(label))\n"
        "    ax.axis('off')\n"
        "fig.suptitle('digits: 1797 samples, 8x8 pixels, 10 classes')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## The integration contract\n"
        "\n"
        "`DataModule` declares nine abstract methods and `ModelModule` four.\n"
        "Implementing them is the entire cost of adopting safetycage. Rather\n"
        "than paraphrase, here is the actual source you would be writing."
    ),
    nbf.v4.new_code_cell(
        "import inspect\n"
        "import modules\n"
        "\n"
        "print(inspect.getsource(modules.DigitsDataModule))"
    ),
    nbf.v4.new_markdown_cell(
        "Note the ordering in `setup()`: the data is split **before** the\n"
        "scaler is fitted, and the scaler is fitted on the training split\n"
        "only. Fitting it on everything first would leak validation and test\n"
        "statistics into training, and would quietly flatter every number\n"
        "that follows.\n"
        "\n"
        "Now the model side. A RandomForest has no hidden layers, so the\n"
        "representations safetycage can threshold are the inputs, the class\n"
        "probabilities, and their logs. A neural network would expose real\n"
        "layer activations instead."
    ),
    nbf.v4.new_code_cell("print(inspect.getsource(modules.DigitsModelModule))"),
    nbf.v4.new_markdown_cell(
        "## Training a model worth guarding\n"
        "\n"
        "Training is ordinary scikit-learn and teaches nothing about\n"
        "safetycage, so it lives in `train.py`.\n"
        "\n"
        "One deliberate choice there: the forest is capped at `max_depth=3`.\n"
        "Unconstrained it reaches about 97.6%, which leaves only 7 mistakes in\n"
        "the validation split - too few to choose a threshold on, and too few\n"
        "to tell whether the cage is working. A weaker model makes the\n"
        "demonstration honest."
    ),
    nbf.v4.new_code_cell(
        "from train import build_modules\n"
        "\n"
        "data_module, model_module = build_modules()\n"
        "\n"
        "x_val, y_val = data_module.data_val\n"
        "predictions = model_module._get_predictions(x_val)\n"
        "incorrect = (predictions != y_val).astype(int)\n"
        "\n"
        "print(f'validation samples : {len(x_val)}')\n"
        "print(f'accuracy           : {(predictions == y_val).mean():.1%}')\n"
        "print(f'misclassified      : {incorrect.sum()}')"
    ),
    nbf.v4.new_markdown_cell(
        "## Maximum Softmax Probability\n"
        "\n"
        "MSP is the simplest safety cage: it treats the largest class\n"
        "probability as a confidence score and flags anything below a\n"
        "threshold. It needs no training, which is why `train_cage()` does\n"
        "nothing here.\n"
        "\n"
        "Hendrycks & Gimpel (2016), https://arxiv.org/abs/1610.02136"
    ),
    nbf.v4.new_code_cell(
        "from safetycage.methods.msp import MSP\n"
        "\n"
        "msp = MSP(model_module=model_module, data_module=data_module)\n"
        "msp.train_cage()\n"
        "\n"
        "statistics_val = msp.predict(x_val, y_val)\n"
        "\n"
        "print(f'mean confidence, correct   : {statistics_val[incorrect == 0].mean():.3f}')\n"
        "print(f'mean confidence, incorrect : {statistics_val[incorrect == 1].mean():.3f}')"
    ),
    nbf.v4.new_markdown_cell(
        "The gap between those two numbers is the entire premise of the\n"
        "method. If it were absent, MSP would have nothing to work with here."
    ),
    nbf.v4.new_code_cell(
        "plt.hist(statistics_val[incorrect == 0], bins=40, alpha=0.6, label='correct')\n"
        "plt.hist(statistics_val[incorrect == 1], bins=40, alpha=0.6, label='incorrect')\n"
        "plt.yscale('log')\n"
        "plt.xlabel('maximum predicted probability')\n"
        "plt.ylabel('count (log scale)')\n"
        "plt.title('Confidence separates correct from incorrect predictions')\n"
        "plt.legend()\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## Choosing a threshold\n"
        "\n"
        "`find_best_threshold` sweeps candidate thresholds and keeps the one\n"
        "maximising the metric you pass. We use MCC, which stays honest under\n"
        "class imbalance - and misclassifications are the minority here.\n"
        "\n"
        "This is the *method* on the cage object. It takes no `leq` argument;\n"
        "it knows from `msp.leq` that low scores mean 'flag this'. There is a\n"
        "separate module-level `safetycage.utils.evaluate.find_best_threshold`\n"
        "that does take `leq` - they are different functions."
    ),
    nbf.v4.new_code_cell(
        "from safetycage.utils.evaluate import MCC\n"
        "\n"
        "result = msp.find_best_threshold(\n"
        "    y_true=incorrect,\n"
        "    y_probs=statistics_val,\n"
        "    metric_fn=MCC,\n"
        ")\n"
        "\n"
        "print(f\"optimal threshold : {result['alpha_opt']:.4f}\")\n"
        "print(f\"MCC at threshold  : {float(result['metric_max']):.4f}\")"
    ),
    nbf.v4.new_markdown_cell(
        "## Applying it to held-out data\n"
        "\n"
        "The threshold was chosen on validation data. Applying it to the test\n"
        "split is the only honest estimate of how it will behave in use."
    ),
    nbf.v4.new_code_cell(
        "x_test, y_test = data_module.data_test\n"
        "\n"
        "statistics_test = msp.predict(x_test, y_test)\n"
        "flagged = msp.flag(statistics_test, result['alpha_opt'])\n"
        "incorrect_test = (model_module._get_predictions(x_test) != y_test).astype(int)\n"
        "\n"
        "caught = int((flagged & (incorrect_test == 1)).sum())\n"
        "total_errors = int(incorrect_test.sum())\n"
        "\n"
        "print(f'test samples           : {len(x_test)}')\n"
        "print(f'actually misclassified : {total_errors}')\n"
        "print(f'flagged by the cage    : {int(flagged.sum())}')\n"
        "print(f'errors caught          : {caught} of {total_errors} ({caught / total_errors:.0%})')"
    ),
    nbf.v4.new_markdown_cell(
        "Read that last block carefully: the cage flags more samples than\n"
        "there are errors. That is the trade-off. Catching most mistakes means\n"
        "sending some correct predictions for review too, and where you set\n"
        "the threshold decides which of those costs you would rather pay.\n"
        "MCC picked this point; a different metric would pick a different one.\n"
        "\n"
        "## Where to go next\n"
        "\n"
        "MSP only needs class probabilities, so it works with any classifier.\n"
        "The other methods - Doctor, Spardacus, Mahalanobis - read intermediate\n"
        "activations, which means a model with hidden layers. See the mnist\n"
        "example for a `ModelModule` that extracts them from a neural network."
    ),
]

nbf.write(nb, "digits_example.ipynb")
print("wrote digits_example.ipynb")
```

- [ ] **Step 3: Generate the notebook**

Run: `cd examples/digits && uv run --group examples python build_notebook.py`
Expected: prints `wrote digits_example.ipynb`.

- [ ] **Step 4: Execute it and commit the outputs**

Run:
```bash
cd examples/digits && uv run --group examples jupyter nbconvert \
  --to notebook --execute --inplace digits_example.ipynb
```
Expected: completes with no error, every cell carrying output. The accuracy cell
should read 88.5% with 33 misclassified.

If it fails, read the traceback before changing anything — a failure here means
an earlier task's code is wrong, not that the notebook needs a workaround.

- [ ] **Step 5: Delete the builder**

```bash
rm examples/digits/build_notebook.py
```

It has served its purpose; leaving it would imply the notebook should not be
edited directly, which is not the intent.

- [ ] **Step 6: Commit**

```bash
git add examples/digits/digits_example.ipynb pyproject.toml
git commit -m "Add digits example notebook and examples dependency group

Notebook renders both module subclasses inline with inspect.getsource so
the integration contract is visible without duplicating it, and is
explicit about the flagging trade-off rather than only reporting recall.
Jupyter tooling goes in a PEP 735 dependency group so it stays out of the
published wheel."
```

---

### Task 7: READMEs

**Files:**
- Create: `examples/README.md`
- Create: `examples/digits/README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the documented entry point for readers.

- [ ] **Step 1: Write `examples/README.md`**

````markdown
# safetycage examples

Worked examples showing how to wrap your own data and model so safetycage
can flag likely misclassifications.

Start with **[digits](digits/)**. It is the smallest complete integration
and needs no dependency beyond safetycage's own.

| Example | Model | Methods | Extra dependencies |
| --- | --- | --- | --- |
| [digits](digits/) | scikit-learn RandomForest | MSP | none |

## Running them

```bash
uv sync --group examples
uv run --group examples jupyter lab examples/digits/digits_example.ipynb
```

`uv sync` installs safetycage in editable mode, so the examples exercise
the source in `src/` rather than a released version.

## How each example is laid out

Every example directory contains the same three things:

- `modules.py` — the `DataModule` and `ModelModule` subclasses. **This is
  the part you would write for your own project.** Copy the directory and
  adapt this file.
- `train.py` — model fitting. Ordinary scikit-learn or PyTorch, nothing
  specific to safetycage, kept out of the notebook.
- `*_example.ipynb` — narration and safetycage usage, with the module
  source rendered inline so you can read it without opening another file.

Each directory is self-contained on purpose: examples deliberately repeat
some code rather than share a base class, so that copying one directory
gives you a complete working reference.
````

- [ ] **Step 2: Write `examples/digits/README.md`**

````markdown
# digits + MSP

The smallest complete safetycage integration.

A RandomForest classifies 8×8 images of handwritten digits. safetycage's
Maximum Softmax Probability method then flags the predictions the model is
least confident about, so they can be reviewed instead of trusted.

**No extra dependencies.** scikit-learn is already required by safetycage,
and the digits dataset ships inside it, so nothing is downloaded.

## Run it

```bash
uv sync --group examples
uv run --group examples jupyter lab digits_example.ipynb
```

Or without the notebook:

```bash
cd examples/digits && uv run python train.py
```

## Files

| File | What it is |
| --- | --- |
| `modules.py` | `DigitsDataModule` and `DigitsModelModule` — the integration contract |
| `train.py` | Fits the RandomForest; no safetycage in it |
| `digits_example.ipynb` | The walkthrough |

## What to read first

`modules.py`. Adopting safetycage means writing your own version of those
two classes — `DataModule` declares nine abstract methods and
`ModelModule` four, and implementing them is the whole cost of adoption.

Three details are easy to get wrong:

- `setup()` splits the data **before** fitting the scaler, and fits it on
  the training split only. Fitting first would leak test statistics into
  training.
- `_get_probabilities` is not declared abstract on the base class, but MSP
  calls it. Any `ModelModule` used with MSP must define it.
- `log_probabilities` clips before taking the log. With 10 classes and a
  shallow forest, exact zero probabilities are common, and `log(0)` is
  `-inf`.

## Why the model is deliberately weak

`train.py` caps the forest at `max_depth=3`, giving about 88.5% accuracy.
Unconstrained it reaches 97.6%, which leaves only 7 mistakes in the
validation split — too few to choose a threshold on, and too few to tell
whether the cage is doing anything. Misclassification detection is only
demonstrable on a model that misclassifies.
````

- [ ] **Step 3: Verify the documented commands actually work**

Run every command in both READMEs from a clean shell. Any that fails must be
fixed in the README, not left aspirational.

Run: `uv sync --group examples && cd examples/digits && uv run python train.py`
Expected: prints validation accuracy 88.5% and 33 errors.

- [ ] **Step 4: Commit**

```bash
git add examples/README.md examples/digits/README.md
git commit -m "Document the examples layout and the digits example

Records the copy-one-directory convention, the three details in
modules.py that are easy to get wrong, and why the classifier is
deliberately constrained."
```

---

## Done when

- [ ] `uv run --with pytest pytest tests/ -q` is green (18 passed).
- [ ] `examples/digits/digits_example.ipynb` has committed outputs in every cell.
- [ ] `uv sync --group examples` does not alter `[project] dependencies`.
- [ ] `git diff --stat origin/main -- pyproject.toml` shows only the
      `[dependency-groups]` addition.
- [ ] Every command in both READMEs has been run and works.
- [ ] No `pyrootutils` import and no `.project-root` file anywhere in `examples/`.

## Follow-on plans

- **Plan 2 — mnist:** Keras MLP to PyTorch, activation extraction via forward
  hooks, covering MSP, Doctor and Spardacus. digits leads into this naturally:
  same task, same ten classes, a real neural network instead of a forest.
- **Plan 3 — cifar10:** Keras CNN to PyTorch, covering Mahalanobis and
  Spardacus. Conv activations are `(N, C, H, W)` where an MLP's are `(N, F)`,
  so `_calc_model_shape` and any flattening the methods assume need real
  attention rather than a copy of the mnist implementation.

## Deferred, from the spec

- Git history handling for the migration (subtree merge vs clean commit).
- The fate of the `safetycage-tutorials` repository afterwards.
- Wiring `.gitlab-ci.yml`, and whether to relax the exact
  `requires-python = "==3.11.7"` pin to make runner images easier to source.
