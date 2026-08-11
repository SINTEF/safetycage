# ArrayDataModule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `ArrayDataModule`, a ready-made `DataModule` for train/val/test splits the caller already has in memory as numpy arrays or pandas DataFrame/Series, so the common "I already split my data" case needs zero adapter code.

**Architecture:** One new class in a new subpackage, `src/safetycage/datamodules/array_datamodule.py`, mirroring the existing `src/safetycage/modelmodules/` subpackage's shape and conventions. Pandas input is normalized to numpy via a duck-typed `.to_numpy()` check — no new dependency, no format-specific subclass, no dispatcher (there is nothing to dispatch between; a DataFrame and a numpy array behave identically once normalized).

**Tech Stack:** Python 3.13, numpy, joblib (already core dependencies); pandas is exercised only via duck typing in tests, never imported by the shipped module itself.

## Global Constraints

- `ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test, **kwargs)` — all six required. No `AutoDataModule`, no auto-splitting, no `Bunch` support: out of scope per the spec.
- No `data_dir` argument on the public constructor — internally calls `super().__init__(data_dir=".", **kwargs)`, matching the existing `how-it-works.md` sketch.
- Must accept numpy arrays or pandas DataFrame/Series for any of the six arguments, normalizing to numpy via a duck-typed check (`hasattr(x, "to_numpy")`), never importing `pandas` itself.
- `classes`/`num_classes` derived from `np.unique(y_train)` post-normalization: `{c: str(c) for c in np.unique(y_train)}`.
- `dataset_name` returns the literal string `"custom"`.
- `setup`, `_load_data`, `_split` are no-ops; `_transform` returns `(x, y)` unchanged.
- `to_joblib`/`from_joblib` delegate to `joblib.dump`/`joblib.load` on `self`.
- File location: `src/safetycage/datamodules/array_datamodule.py`, no `__init__.py` in the new `datamodules/` directory (matches the existing `src/safetycage/modelmodules/` and `src/safetycage/methods/` namespace-package convention in this repo).
- Test command for this project: `uv run --with pytest --all-extras pytest <path> -v`. `pandas` is already present in this dev environment (a transitive dependency), so pandas-specific tests can run directly; guard them with `pytest.importorskip("pandas")` regardless, for correctness on environments where it's absent.

---

### Task 1: Implement `ArrayDataModule`

**Files:**
- Create: `src/safetycage/datamodules/array_datamodule.py`
- Test: `tests/test_array_datamodule.py`

**Interfaces:**
- Consumes: `safetycage.datamodule.DataModule` (base class, `__init__(self, data_dir=None, from_cache=False, batch_size=32, device="cpu")`).
- Produces: `ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test, **kwargs)` — a `DataModule` subclass with `data_train`/`data_val`/`data_test` as `(np.ndarray, np.ndarray)` tuples, `classes`/`num_classes`/`dataset_name` properties, and `to_joblib`/`from_joblib`. No later task in this plan consumes it directly (Task 2 only documents it), but it is a public API other work may import from `safetycage.datamodules.array_datamodule`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_array_datamodule.py`:
```python
"""Unit tests for ArrayDataModule.

Run with: uv run --with pytest --all-extras pytest tests/test_array_datamodule.py -v
"""
import os
import tempfile

import numpy as np
import pytest

from safetycage.datamodules.array_datamodule import ArrayDataModule


def _synthetic_splits():
    rng = np.random.RandomState(0)
    x_train = rng.randn(20, 3)
    y_train = rng.randint(0, 3, size=20)
    x_val = rng.randn(10, 3)
    y_val = rng.randint(0, 3, size=10)
    x_test = rng.randn(10, 3)
    y_test = rng.randint(0, 3, size=10)
    return x_train, y_train, x_val, y_val, x_test, y_test


def test_wraps_numpy_arrays():
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()
    module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)

    np.testing.assert_array_equal(module.data_train[0], x_train)
    np.testing.assert_array_equal(module.data_train[1], y_train)
    np.testing.assert_array_equal(module.data_val[0], x_val)
    np.testing.assert_array_equal(module.data_val[1], y_val)
    np.testing.assert_array_equal(module.data_test[0], x_test)
    np.testing.assert_array_equal(module.data_test[1], y_test)


def test_classes_and_num_classes_from_y_train():
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()
    module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)

    expected = {c: str(c) for c in np.unique(y_train)}
    assert module.classes == expected
    assert module.num_classes == len(expected)


def test_dataset_name_is_custom():
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()
    module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)
    assert module.dataset_name == "custom"


def test_accepts_pandas_dataframe_and_series():
    pd = pytest.importorskip("pandas")
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()

    module = ArrayDataModule(
        pd.DataFrame(x_train),
        pd.Series(y_train),
        pd.DataFrame(x_val),
        pd.Series(y_val),
        pd.DataFrame(x_test),
        pd.Series(y_test),
    )

    for split in (module.data_train, module.data_val, module.data_test):
        assert isinstance(split[0], np.ndarray)
        assert isinstance(split[1], np.ndarray)

    np.testing.assert_array_equal(module.data_train[0], x_train)
    np.testing.assert_array_equal(module.data_train[1], y_train)


def test_to_joblib_from_joblib_roundtrip():
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()
    module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data_module.joblib")
        module.to_joblib(path)
        loaded = module.from_joblib(path)

    np.testing.assert_array_equal(loaded.data_train[0], x_train)
    np.testing.assert_array_equal(loaded.data_test[1], y_test)
    assert loaded.classes == module.classes
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pytest --all-extras pytest tests/test_array_datamodule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'safetycage.datamodules'`.

- [ ] **Step 3: Implement `ArrayDataModule`**

Create `src/safetycage/datamodules/array_datamodule.py`:
```python
"""ArrayDataModule: wraps train/val/test splits already loaded in memory.

Accepts numpy arrays or pandas DataFrame/Series for any split. Pandas
input is normalized to numpy via a duck-typed to_numpy() check, so this
module never imports pandas itself.
"""
from typing import Any, Dict

import joblib
import numpy as np

from safetycage.datamodule import DataModule


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


class ArrayDataModule(DataModule):
    """Wraps train/val/test splits that are already loaded in memory.

    ``train`` fits the cage itself (used by RED/SPARDACUS/Mahalanobis;
    MSP/DOCTOR need no fitting). ``val`` estimates the optimal threshold.
    ``test`` evaluates that threshold on held-out data.
    """

    def __init__(
        self,
        x_train: Any,
        y_train: Any,
        x_val: Any,
        y_val: Any,
        x_test: Any,
        y_test: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_dir=".", **kwargs)

        y_train = _to_numpy(y_train)
        self.data_train = (_to_numpy(x_train), y_train)
        self.data_val = (_to_numpy(x_val), _to_numpy(y_val))
        self.data_test = (_to_numpy(x_test), _to_numpy(y_test))
        self._classes = {c: str(c) for c in np.unique(y_train)}

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def classes(self) -> Dict[Any, str]:
        return self._classes

    @property
    def dataset_name(self) -> str:
        return "custom"

    def setup(self) -> None:
        pass  # Splits are already provided in __init__.

    def _load_data(self, filepath: str) -> None:
        pass  # Not needed: data arrives pre-loaded.

    def _transform(self, x, y):
        return x, y

    def _split(self, x, y, split):
        pass  # Not needed: splits are already provided.

    def to_joblib(self, path: str) -> None:
        joblib.dump(self, path)

    def from_joblib(self, path: str) -> Any:
        return joblib.load(path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --with pytest --all-extras pytest tests/test_array_datamodule.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `uv run --with pytest --all-extras pytest -q`
Expected: all prior passing tests still pass, plus these 5 new ones; the 3 pre-existing unrelated `test_red.py` failures (from an earlier `alpha`→`threshold` rename) are still present and are not this task's concern.

- [ ] **Step 6: Commit**

```bash
git add src/safetycage/datamodules/array_datamodule.py tests/test_array_datamodule.py
git commit -m "Add ArrayDataModule for pre-split in-memory data"
```

---

### Task 2: Document `ArrayDataModule`

**Files:**
- Modify: `docs/how-it-works.md`
- Create: `docs/api/datamodules.md`
- Modify: `docs/api/index.md`

**Interfaces:**
- Consumes: `safetycage.datamodules.array_datamodule.ArrayDataModule` (Task 1).
- Produces: nothing consumed by later tasks — this is the last task in the plan.

- [ ] **Step 1: Replace the hand-rolled example in `docs/how-it-works.md`**

Replace this block (the current `## The DataModule` section, up to but not
including `## The ModelModule`) — shown below delimited with 4-backtick
fences purely so its own internal ```python fence displays correctly; the
fences themselves are not part of the file content:

````markdown
## The DataModule

At minimum, a `DataModule` exposes the `classes` and `num_classes`
properties, and populates `data_train`/`data_val`/`data_test` as `(x, y)`
tuples of numpy arrays. If your data is already loaded and split, `setup`,
`_load_data` and `_split` have nothing to do:

```python
import numpy as np
from safetycage.datamodule import DataModule


class ArrayDataModule(DataModule):
    """Wraps data that is already loaded as train/val/test numpy arrays."""

    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test, **kwargs):
        super().__init__(data_dir=".", **kwargs)
        self.data_train = (x_train, y_train)
        self.data_val = (x_val, y_val)
        self.data_test = (x_test, y_test)
        self._classes = {c: str(c) for c in np.unique(y_train)}

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def classes(self):
        return self._classes

    @property
    def dataset_name(self):
        return "custom"

    def setup(self):
        pass  # Splits are already provided in __init__.

    def _load_data(self, filepath):
        pass  # Not needed: data arrives pre-loaded.

    def _transform(self, x, y):
        return x, y

    def _split(self, x, y, split):
        pass  # Not needed: splits are already provided.

    def to_joblib(self, path):
        import joblib
        joblib.dump(self, path)

    def from_joblib(self, path):
        import joblib
        return joblib.load(path)
```
````

with (again shown with 4-backtick outer fences only for display purposes):

````markdown
## The DataModule

At minimum, a `DataModule` exposes the `classes` and `num_classes`
properties, and populates `data_train`/`data_val`/`data_test` as `(x, y)`
tuples of numpy arrays: `train` fits the cage itself (used by
RED/SPARDACUS/Mahalanobis; MSP/DOCTOR need no fitting), `val` estimates the
optimal threshold, and `test` evaluates that threshold on held-out data.

If your data is already loaded and split, safetycage ships a ready-made
`DataModule` for exactly that:

```python
from safetycage.datamodules.array_datamodule import ArrayDataModule

data_module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)
```

Accepts numpy arrays or pandas DataFrame/Series for any split — pandas
input is normalized to numpy internally, so `data_train`/`data_val`/`data_test`
are always numpy arrays regardless of what you passed in. Write a custom
`DataModule` only if your data isn't already split, or needs its own
loading/caching logic (see the MNIST example).
````

(i.e. the new content keeps the `## The DataModule` heading, replaces the
40-line class sketch with the two paragraphs and short usage snippet above,
then the file continues unchanged into `## The ModelModule`.)

- [ ] **Step 2: Create the API reference page for the new subpackage**

Create `docs/api/datamodules.md` (shown with a 4-backtick outer fence only
so its internal ```{eval-rst} fence displays correctly):

````markdown
# Data modules

Ready-made {py:class}`~safetycage.datamodule.DataModule` implementations
for common data shapes. See [How it works](../how-it-works) for usage.

## ArrayDataModule

```{eval-rst}
.. automodule:: safetycage.datamodules.array_datamodule
```
````

- [ ] **Step 3: Add the new page to the API toctree**

In `docs/api/index.md`, change (outer 4-backtick fence for display only):

````markdown
```{toctree}
:maxdepth: 2

core
methods
modelmodules
utils
```
````

to:

````markdown
```{toctree}
:maxdepth: 2

core
datamodules
methods
modelmodules
utils
```
````

- [ ] **Step 4: Build the docs and check the new page and cross-links resolve**

Run: `uv run --group docs sphinx-build -b html docs docs/_build/html -q`
Expected: exits with no errors or warnings about `datamodules` or the
`../how-it-works` cross-link. Then open `docs/_build/html/api/datamodules.html`
and `docs/_build/html/how-it-works.html` and visually confirm the class
renders with its docstring and the rewritten section reads correctly.

- [ ] **Step 5: Commit**

```bash
git add docs/how-it-works.md docs/api/datamodules.md docs/api/index.md
git commit -m "Document ArrayDataModule"
```
