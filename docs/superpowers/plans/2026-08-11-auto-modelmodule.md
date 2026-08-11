# Built-in ModelModule implementations (Torch/Sklearn/Auto) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `SklearnModelModule`, `TorchModelModule` and `AutoModelModule` so MSP/DOCTOR work against a plain scikit-learn-style classifier or PyTorch model with near-zero user code, and remove the dead `_calc_model_shape` abstract method these new classes would otherwise need to stub out pointlessly.

**Architecture:** New subpackage `src/safetycage/modelmodules/` (no `__init__.py`, matching the existing `src/safetycage/methods/` namespace-package convention) holding three modules, each a thin `ModelModule` subclass. `AutoModelModule` is a factory (`__new__` returns a `TorchModelModule`/`SklearnModelModule` instance directly) that dispatches on the model's type.

**Tech Stack:** Python 3.13, numpy, optional `torch` (new lightweight extra), optional `scikit-learn`-style duck typing (no new dependency), pytest.

**Spec:** `docs/superpowers/specs/2026-08-11-auto-modelmodule-design.md`

## Global Constraints

- Scope is `_get_predictions`/`_get_probabilities` only. `_get_activations`/`_get_pre_activations` raise `NotImplementedError` on all three new classes — no activation auto-discovery in this plan.
- No `use_onehot_encoder` parameter on `SklearnModelModule` or `TorchModelModule` — they always return integer class-index predictions.
- `_calc_model_shape` is removed entirely from `ModelModule` and its only implementation (`examples/01-mnist/mlp_modelmodule.py`) — not stubbed on the new classes.
- `TorchModelModule` requires `torch>=1.13`, installed via a new `torch` extra (`pip install safetycage[torch]`), independent of the heavier `red` extra (`torch` + `gpytorch`).
- `SklearnModelModule` must not import `sklearn` — duck-typed on `.predict`/`.predict_proba` so XGBoost/LightGBM/CatBoost work too.
- Importing `safetycage.modelmodules.torch_modelmodule` must succeed even without torch installed (guarded `try/except ImportError`, same pattern as `src/safetycage/methods/red.py`); only instantiating `TorchModelModule` without torch raises `ImportError`.
- Error message conventions (from the spec's error-handling table):
  - `TorchModelModule` without torch installed → `ImportError` naming `safetycage[torch]`.
  - `SklearnModelModule` model lacking `predict_proba` → `AttributeError` at `_get_probabilities` call time, naming the missing method.
  - `AutoModelModule` given an unrecognized model → `TypeError` describing what was checked.
  - Any activation method call on the new classes → `NotImplementedError` pointing to `docs/how-it-works.md`.
- Test command for this project: `uv run --with pytest --all-extras pytest <path> -v` (there is no `pytest`/`sklearn` in the base env; `--all-extras` pulls in `scikit-learn`, and `--with pytest` adds the runner itself). Note: `tests/test_red.py` has 3 pre-existing failures unrelated to this work (from an earlier `alpha`→`threshold` rename) — do not try to fix them as part of this plan.

---

### Task 1: Remove the dead `_calc_model_shape` abstract method

**Files:**
- Modify: `src/safetycage/modelmodule.py`
- Modify: `examples/01-mnist/mlp_modelmodule.py`
- Modify: `docs/conf.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ModelModule` no longer declares `_calc_model_shape` as abstract — later tasks' `SklearnModelModule`/`TorchModelModule` do not need to implement it.

- [ ] **Step 1: Remove `_calc_model_shape` from the `ModelModule` ABC**

In `src/safetycage/modelmodule.py`, remove the unused `Dict` import (it is only used by the method being deleted) and the method itself.

Change the import line:
```python
from typing import Tuple, Union, List, Any, Dict
```
to:
```python
from typing import Tuple, Union, List, Any
```

Remove this block (currently lines 72–79, immediately before the `if __name__ == '__main__':` block):
```python

    @abstractmethod
    def _calc_model_shape(self) -> Dict[str,int]:
        """
        Get the shape of each layer in the model.
        Returns: List of integers representing the number of neurons in each layer
        """
        raise NotImplementedError("Implement based on your model architecture")
```
so that `_get_pre_activations` is immediately followed by the (untouched) `if __name__ == '__main__':` block.

- [ ] **Step 2: Remove its implementation from the MNIST example**

In `examples/01-mnist/mlp_modelmodule.py`, remove the last method of `MLPModelModule` (currently lines 177–184):
```python

    def _calc_model_shape(self) -> Dict[str, int]:
        """Units per block. Nothing here uses it, but ``ModelModule`` declares
        it abstract, so the class cannot be instantiated without it."""
        return {
            block: self._modules_by_name[LAYER_BLOCKS[block][0]].out_features
            for block in self.AVAILABLE_LAYERS
            if LAYER_BLOCKS[block][0] in self._modules_by_name
        }
```
`Dict` is still used elsewhere in this file's type hints (e.g. `_get_activations` returns `Dict[str, np.ndarray]`), so leave its import alone.

- [ ] **Step 3: Remove it from the Sphinx autodoc config**

In `docs/conf.py`, change:
```python
    "private-members": "_get_predictions,_get_probabilities,_get_activations,_get_pre_activations,_calc_model_shape",
```
to:
```python
    "private-members": "_get_predictions,_get_probabilities,_get_activations,_get_pre_activations",
```

- [ ] **Step 4: Run the existing test suite to confirm nothing else depended on it**

Run: `uv run --with pytest --all-extras pytest -q`
Expected: same baseline as before this change — 14 passed, 3 failed (the pre-existing, unrelated `test_red.py` failures noted in Global Constraints). If any *different* test fails, stop and investigate before continuing.

- [ ] **Step 5: Confirm the MNIST example still imports**

Run: `uv run --group examples python -c "import sys; sys.path.insert(0, 'examples/01-mnist'); from mlp_modelmodule import MLPModelModule; print('ok')"`
Expected: prints `ok` with no traceback.

- [ ] **Step 6: Rebuild the docs to confirm the autodoc config change doesn't break the build**

Run: `uv run --group docs sphinx-build -b html docs docs/_build/html -q`
Expected: exits with no errors printed.

- [ ] **Step 7: Commit**

```bash
git add src/safetycage/modelmodule.py examples/01-mnist/mlp_modelmodule.py docs/conf.py
git commit -m "Remove unused _calc_model_shape abstract method from ModelModule"
```

---

### Task 2: Add `SklearnModelModule`

**Files:**
- Create: `src/safetycage/modelmodules/sklearn_modelmodule.py`
- Test: `tests/test_sklearn_modelmodule.py`

**Interfaces:**
- Consumes: `safetycage.modelmodule.ModelModule` (base class, `__init__(self, selected_layers, use_onehot_encoder, model, **kwargs)`).
- Produces: `SklearnModelModule(model, **kwargs)` — a `ModelModule` subclass with `_get_predictions`, `_get_probabilities` calling `model.predict`/`model.predict_proba`; `_get_activations`/`_get_pre_activations` raising `NotImplementedError`. Later tasks (`AutoModelModule`) import this class from `safetycage.modelmodules.sklearn_modelmodule`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sklearn_modelmodule.py`:
```python
"""Unit tests for SklearnModelModule.

Run with: uv run --with pytest --all-extras pytest tests/test_sklearn_modelmodule.py -v
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression

from safetycage.modelmodules.sklearn_modelmodule import SklearnModelModule


class _PredictOnlyModel:
    """Stands in for a classifier that only implements predict(), no predict_proba."""

    def predict(self, x):
        return np.zeros(len(x), dtype=int)


@pytest.fixture
def fitted_logistic_regression():
    rng = np.random.RandomState(0)
    x = rng.randn(50, 4)
    y = (x[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(x, y)
    return model, x


def test_get_predictions_matches_model(fitted_logistic_regression):
    model, x = fitted_logistic_regression
    module = SklearnModelModule(model)
    np.testing.assert_array_equal(module._get_predictions(x), model.predict(x))


def test_get_probabilities_matches_model(fitted_logistic_regression):
    model, x = fitted_logistic_regression
    module = SklearnModelModule(model)
    np.testing.assert_allclose(module._get_probabilities(x), model.predict_proba(x))


def test_get_probabilities_without_predict_proba_raises():
    module = SklearnModelModule(_PredictOnlyModel())
    with pytest.raises(AttributeError, match="predict_proba"):
        module._get_probabilities(np.zeros((3, 1)))


def test_activations_not_implemented(fitted_logistic_regression):
    model, x = fitted_logistic_regression
    module = SklearnModelModule(model)
    with pytest.raises(NotImplementedError):
        module._get_activations(x)
    with pytest.raises(NotImplementedError):
        module._get_pre_activations(x)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pytest --all-extras pytest tests/test_sklearn_modelmodule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'safetycage.modelmodules'`.

- [ ] **Step 3: Implement `SklearnModelModule`**

Create `src/safetycage/modelmodules/sklearn_modelmodule.py`:
```python
"""SklearnModelModule: wraps any classifier with predict/predict_proba.

Duck-typed — does not import scikit-learn itself, so this also covers
XGBoost, LightGBM and CatBoost classifiers, which follow the same
convention.
"""
from typing import Any

import numpy as np

from safetycage.modelmodule import ModelModule


class SklearnModelModule(ModelModule):
    """Wraps a classifier implementing predict()/predict_proba().

    Supports MSP and DOCTOR only. SPARDACUS, Mahalanobis and RED need
    hidden-layer activations, which this class does not provide — write a
    custom ModelModule for those (see docs/how-it-works.md).
    """

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(selected_layers=[], use_onehot_encoder=False, model=model, **kwargs)

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def _get_probabilities(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"{type(self.model).__name__} has no predict_proba method. "
                "MSP and DOCTOR need class probabilities; a model exposing "
                "only predict() cannot be used with SklearnModelModule for "
                "those methods."
            )
        return self.model.predict_proba(x)

    def _get_activations(self, x: np.ndarray):
        raise NotImplementedError(
            "SklearnModelModule exposes no intermediate layers. SPARDACUS, "
            "Mahalanobis and RED need hidden-layer activations - write a "
            "custom ModelModule (see docs/how-it-works.md)."
        )

    def _get_pre_activations(self, x: np.ndarray):
        raise NotImplementedError(
            "SklearnModelModule exposes no intermediate layers. SPARDACUS, "
            "Mahalanobis and RED need hidden-layer pre-activations - write "
            "a custom ModelModule (see docs/how-it-works.md)."
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --with pytest --all-extras pytest tests/test_sklearn_modelmodule.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/safetycage/modelmodules/sklearn_modelmodule.py tests/test_sklearn_modelmodule.py
git commit -m "Add SklearnModelModule for predict/predict_proba-style models"
```

---

### Task 3: Add `TorchModelModule` and the `torch` extra

**Files:**
- Create: `src/safetycage/modelmodules/torch_modelmodule.py`
- Modify: `pyproject.toml`
- Test: `tests/test_torch_modelmodule.py`

**Interfaces:**
- Consumes: `safetycage.modelmodule.ModelModule` (same base class as Task 2).
- Produces: `TorchModelModule(model, device="cpu", output_is_probabilities=False, **kwargs)` and module-level `HAS_TORCH: bool` in `safetycage.modelmodules.torch_modelmodule` — `AutoModelModule` (Task 4) imports both.

- [ ] **Step 1: Add the `torch` extra to `pyproject.toml`**

Change:
```toml
[project.optional-dependencies]
red = ["gpytorch>=1.9", "torch>=1.13"]
spardacus = ["statsmodels>=0.14.6", "scikit-learn==1.9.0", "scipy", "tqdm"]
mahalanobis = ["statsmodels>=0.14.6", "scipy"]
```
to:
```toml
[project.optional-dependencies]
red = ["gpytorch>=1.9", "torch>=1.13"]
torch = ["torch>=1.13"]
spardacus = ["statsmodels>=0.14.6", "scikit-learn==1.9.0", "scipy", "tqdm"]
mahalanobis = ["statsmodels>=0.14.6", "scipy"]
```

Run: `uv lock`
Expected: `uv.lock` updates to add the new `torch` extra entry; exits 0.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_torch_modelmodule.py`:
```python
"""Unit tests for TorchModelModule.

Run with: uv run --with pytest --all-extras pytest tests/test_torch_modelmodule.py -v
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from safetycage.modelmodules.torch_modelmodule import TorchModelModule


@pytest.fixture
def logits_model():
    torch.manual_seed(0)
    return nn.Linear(4, 3)


def test_get_probabilities_applies_softmax(logits_model):
    module = TorchModelModule(logits_model)
    x = np.random.RandomState(0).randn(5, 4).astype(np.float32)

    with torch.no_grad():
        expected = torch.softmax(logits_model(torch.as_tensor(x)), dim=-1).numpy()

    np.testing.assert_allclose(module._get_probabilities(x), expected, atol=1e-6)


def test_get_probabilities_sum_to_one(logits_model):
    module = TorchModelModule(logits_model)
    x = np.random.RandomState(1).randn(5, 4).astype(np.float32)
    probs = module._get_probabilities(x)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-6)


def test_get_predictions_is_argmax(logits_model):
    module = TorchModelModule(logits_model)
    x = np.random.RandomState(2).randn(5, 4).astype(np.float32)
    np.testing.assert_array_equal(
        module._get_predictions(x), np.argmax(module._get_probabilities(x), axis=1)
    )


def test_output_is_probabilities_flag_skips_softmax():
    torch.manual_seed(0)

    class AlreadySoftmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, x):
            return torch.softmax(self.linear(x), dim=-1)

    model = AlreadySoftmax()
    module = TorchModelModule(model, output_is_probabilities=True)
    x = np.random.RandomState(3).randn(5, 4).astype(np.float32)

    with torch.no_grad():
        expected = model(torch.as_tensor(x)).numpy()

    np.testing.assert_allclose(module._get_probabilities(x), expected, atol=1e-6)


def test_activations_not_implemented(logits_model):
    module = TorchModelModule(logits_model)
    x = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        module._get_activations(x)
    with pytest.raises(NotImplementedError):
        module._get_pre_activations(x)


def test_import_error_when_torch_missing(monkeypatch):
    """Simulates torch being unavailable, regardless of the test environment."""
    import safetycage.modelmodules.torch_modelmodule as torch_modelmodule

    monkeypatch.setattr(torch_modelmodule, "HAS_TORCH", False)
    with pytest.raises(ImportError, match=r"safetycage\[torch\]"):
        torch_modelmodule.TorchModelModule(model=object())
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --with pytest --all-extras pytest tests/test_torch_modelmodule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'safetycage.modelmodules.torch_modelmodule'`.

- [ ] **Step 4: Implement `TorchModelModule`**

Create `src/safetycage/modelmodules/torch_modelmodule.py`:
```python
"""TorchModelModule: wraps a plain torch.nn.Module.

Requires the lightweight `torch` extra (pip install safetycage[torch]), not
the heavier `red` extra (torch + gpytorch).
"""
from typing import Any

import numpy as np

from safetycage.modelmodule import ModelModule

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TorchModelModule(ModelModule):
    """Wraps a torch.nn.Module for MSP/DOCTOR.

    Assumes the model outputs raw logits (the usual torch convention of
    ending in a Linear layer) and applies softmax to get probabilities.
    Pass output_is_probabilities=True if the model already ends in a
    softmax/sigmoid.

    Supports MSP and DOCTOR only. SPARDACUS, Mahalanobis and RED need
    hidden-layer activations, which this class does not provide — write a
    custom ModelModule for those (see docs/how-it-works.md and
    examples/01-mnist/mlp_modelmodule.py).
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        output_is_probabilities: bool = False,
        **kwargs: Any,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "TorchModelModule requires torch. Install it with "
                "`pip install safetycage[torch]`."
            )
        super().__init__(selected_layers=[], use_onehot_encoder=False, model=model, **kwargs)
        self.device = torch.device(device)
        self.output_is_probabilities = output_is_probabilities
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, x: np.ndarray):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

    def _get_probabilities(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            output = self.model(self._to_tensor(x))
            if not self.output_is_probabilities:
                output = torch.softmax(output, dim=-1)
        return output.cpu().numpy()

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self._get_probabilities(x), axis=1)

    def _get_activations(self, x: np.ndarray):
        raise NotImplementedError(
            "TorchModelModule exposes no intermediate layers. SPARDACUS, "
            "Mahalanobis and RED need hidden-layer activations - write a "
            "custom ModelModule (see docs/how-it-works.md)."
        )

    def _get_pre_activations(self, x: np.ndarray):
        raise NotImplementedError(
            "TorchModelModule exposes no intermediate layers. SPARDACUS, "
            "Mahalanobis and RED need hidden-layer pre-activations - write "
            "a custom ModelModule (see docs/how-it-works.md)."
        )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --with pytest --all-extras pytest tests/test_torch_modelmodule.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/safetycage/modelmodules/torch_modelmodule.py tests/test_torch_modelmodule.py
git commit -m "Add TorchModelModule and the torch extra"
```

---

### Task 4: Add `AutoModelModule`

**Files:**
- Create: `src/safetycage/modelmodules/auto_modelmodule.py`
- Test: `tests/test_auto_modelmodule.py`

**Interfaces:**
- Consumes: `TorchModelModule`, `HAS_TORCH` from `safetycage.modelmodules.torch_modelmodule` (Task 3); `SklearnModelModule` from `safetycage.modelmodules.sklearn_modelmodule` (Task 2); `ModelModule` from `safetycage.modelmodule`.
- Produces: `AutoModelModule(model, **kwargs) -> ModelModule` (a class whose `__new__` returns a `TorchModelModule` or `SklearnModelModule` instance).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_auto_modelmodule.py`:
```python
"""Unit tests for AutoModelModule.

Run with: uv run --with pytest --all-extras pytest tests/test_auto_modelmodule.py -v
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression

from safetycage.modelmodules.auto_modelmodule import AutoModelModule
from safetycage.modelmodules.sklearn_modelmodule import SklearnModelModule
from safetycage.modelmodules.torch_modelmodule import TorchModelModule


def test_dispatches_to_torch_for_nn_module():
    module = AutoModelModule(nn.Linear(4, 3))
    assert isinstance(module, TorchModelModule)


def test_dispatches_to_sklearn_for_predict_object():
    rng = np.random.RandomState(0)
    x = rng.randn(20, 4)
    y = (x[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(x, y)

    module = AutoModelModule(model)
    assert isinstance(module, SklearnModelModule)


def test_raises_type_error_for_unrecognized_model():
    with pytest.raises(TypeError, match="ModelModule"):
        AutoModelModule(object())


def test_kwargs_forwarded_to_torch_module():
    module = AutoModelModule(nn.Linear(4, 3), output_is_probabilities=True)
    assert module.output_is_probabilities is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pytest --all-extras pytest tests/test_auto_modelmodule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'safetycage.modelmodules.auto_modelmodule'`.

- [ ] **Step 3: Implement `AutoModelModule`**

Create `src/safetycage/modelmodules/auto_modelmodule.py`:
```python
"""AutoModelModule: picks TorchModelModule or SklearnModelModule for you."""
from typing import Any

from safetycage.modelmodule import ModelModule
from safetycage.modelmodules.sklearn_modelmodule import SklearnModelModule
from safetycage.modelmodules.torch_modelmodule import HAS_TORCH, TorchModelModule

if HAS_TORCH:
    import torch


class AutoModelModule:
    """Factory that dispatches to TorchModelModule or SklearnModelModule.

    AutoModelModule(model, **kwargs) returns a ModelModule instance
    directly; AutoModelModule is never itself instantiated.
    """

    def __new__(cls, model: Any, **kwargs: Any) -> ModelModule:
        if HAS_TORCH and isinstance(model, torch.nn.Module):
            return TorchModelModule(model, **kwargs)
        if hasattr(model, "predict"):
            return SklearnModelModule(model, **kwargs)
        raise TypeError(
            f"Could not determine a ModelModule for {type(model).__name__}: "
            "it is not a torch.nn.Module and has no predict() method. Write "
            "a custom ModelModule (see docs/how-it-works.md)."
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --with pytest --all-extras pytest tests/test_auto_modelmodule.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `uv run --with pytest --all-extras pytest -q`
Expected: same baseline as Task 1 Step 4, plus the new passing tests from Tasks 2–4 (14 + 4 + 6 + 4 = 28 passed, 3 pre-existing unrelated failures).

- [ ] **Step 6: Commit**

```bash
git add src/safetycage/modelmodules/auto_modelmodule.py tests/test_auto_modelmodule.py
git commit -m "Add AutoModelModule to dispatch between Torch and Sklearn model modules"
```

---

### Task 5: Document the new classes

**Files:**
- Modify: `docs/how-it-works.md`
- Create: `docs/api/modelmodules.md`
- Modify: `docs/api/index.md`

**Interfaces:**
- Consumes: `safetycage.modelmodules.sklearn_modelmodule.SklearnModelModule`, `safetycage.modelmodules.torch_modelmodule.TorchModelModule`, `safetycage.modelmodules.auto_modelmodule.AutoModelModule` (Tasks 2–4).
- Produces: nothing consumed by later tasks — this is the last task in the plan.

- [ ] **Step 1: Replace the hand-rolled example in `docs/how-it-works.md` and add the new-classes section**

Replace this block (the current `## The ModelModule` section, up to but not
including `## Running a safety cage method`) — shown below delimited with
4-backtick fences purely so its own internal ```python fences display
correctly; the fences themselves are not part of the file content:

````markdown
## The ModelModule

At minimum, a `ModelModule` implements `_get_predictions`. MSP and DOCTOR
also need `_get_probabilities`; SPARDACUS, Mahalanobis and RED need
`_get_activations`/`_get_pre_activations` from a network's hidden layers, so
they only work with models that expose intermediate layers (see the MNIST
example). A model that only exposes a `predict_proba`-style output, such as
a scikit-learn classifier, is enough for MSP or DOCTOR:

```python
from safetycage.modelmodule import ModelModule


class SklearnModelModule(ModelModule):
    """Wraps any scikit-learn-style classifier with `predict_proba`."""

    def __init__(self, model, **kwargs):
        super().__init__(selected_layers=[], use_onehot_encoder=False, model=model, **kwargs)

    def _get_predictions(self, x):
        return self.model.predict(x)

    def _get_probabilities(self, x):
        return self.model.predict_proba(x)

    def _get_activations(self, x):
        raise NotImplementedError("This model exposes no intermediate layers.")

    def _get_pre_activations(self, x):
        raise NotImplementedError("This model exposes no intermediate layers.")

    def _calc_model_shape(self):
        return {}
```
````

with (again shown with 4-backtick outer fences only for display purposes):

````markdown
## The ModelModule

At minimum, a `ModelModule` implements `_get_predictions`. MSP and DOCTOR
also need `_get_probabilities`; SPARDACUS, Mahalanobis and RED need
`_get_activations`/`_get_pre_activations` from a network's hidden layers, so
they only work with models that expose intermediate layers (see the MNIST
example). For the common case of a scikit-learn-style classifier or a plain
PyTorch model, safetycage ships ready-made `ModelModule`s — see the next
section. Write a custom one only when your model doesn't fit either shape,
or you need SPARDACUS/Mahalanobis/RED's hidden-layer access.

## Common models: TorchModelModule, SklearnModelModule, AutoModelModule

For MSP or DOCTOR, a plain scikit-learn-style classifier needs no adapter
code at all:

```python
from safetycage.modelmodules.sklearn_modelmodule import SklearnModelModule

model_module = SklearnModelModule(model)  # model has .predict / .predict_proba
```

Works with any object following the same convention — XGBoost, LightGBM and
CatBoost classifiers included.

The equivalent for a plain PyTorch model:

```python
from safetycage.modelmodules.torch_modelmodule import TorchModelModule

model_module = TorchModelModule(model, device="cpu")
```

Requires the `torch` extra (`pip install safetycage[torch]`). Assumes
`model` outputs raw logits (the usual convention of ending in a `Linear`
layer) and applies softmax internally; pass `output_is_probabilities=True`
if your model already ends in softmax/sigmoid.

{py:class}`~safetycage.modelmodules.auto_modelmodule.AutoModelModule` picks
between the two for you:

```python
from safetycage.modelmodules.auto_modelmodule import AutoModelModule

model_module = AutoModelModule(model)  # torch.nn.Module or a .predict-style object
```

None of these three expose hidden-layer activations — SPARDACUS, Mahalanobis
and RED still need a hand-written `ModelModule` (see the MNIST example).
````

(i.e. the new content keeps the `## The ModelModule` heading and its
shortened paragraph, then adds the new `## Common models: ...` section, then
the file continues unchanged into `## Running a safety cage method`.)

- [ ] **Step 2: Create the API reference page for the new subpackage**

Create `docs/api/modelmodules.md` (shown with a 4-backtick outer fence only
so its internal ```{eval-rst} fences display correctly):

````markdown
# Model modules

Ready-made {py:class}`~safetycage.modelmodule.ModelModule` implementations
for common model types. See [How it works](../how-it-works) for usage.

## SklearnModelModule

```{eval-rst}
.. automodule:: safetycage.modelmodules.sklearn_modelmodule
```

## TorchModelModule

Requires the `torch` extra (`pip install safetycage[torch]`).

```{eval-rst}
.. automodule:: safetycage.modelmodules.torch_modelmodule
```

## AutoModelModule

```{eval-rst}
.. automodule:: safetycage.modelmodules.auto_modelmodule
```
````

- [ ] **Step 3: Add the new page to the API toctree**

In `docs/api/index.md`, change (outer 4-backtick fence for display only):

````markdown
```{toctree}
:maxdepth: 2

core
methods
utils
```
````

to:

````markdown
```{toctree}
:maxdepth: 2

core
methods
modelmodules
utils
```
````

- [ ] **Step 4: Build the docs and check the new page and cross-links resolve**

Run: `uv run --group docs sphinx-build -b html docs docs/_build/html -q`
Expected: exits with no errors or warnings about `modelmodules` or the `../how-it-works` cross-link. Then open `docs/_build/html/api/modelmodules.html` and `docs/_build/html/how-it-works.html` and visually confirm the three classes render with their docstrings and the new section reads correctly.

- [ ] **Step 5: Commit**

```bash
git add docs/how-it-works.md docs/api/modelmodules.md docs/api/index.md
git commit -m "Document TorchModelModule, SklearnModelModule and AutoModelModule"
```
