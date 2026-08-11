# Design: built-in ModelModule implementations for common frameworks

Date: 2026-08-11

## Problem

Adopting safetycage requires writing a `ModelModule` subclass implementing
`_get_predictions`, `_get_activations`, `_get_pre_activations` and
`_calc_model_shape` (see `docs/how-it-works.md`). For the two class-probability
methods (MSP, DOCTOR), this is pure boilerplate for the common cases of a
scikit-learn-style classifier or a plain PyTorch model: the user has to
re-derive predictions/probabilities from `model.predict_proba` or a forward
pass every time, by hand.

This design adds ready-made `ModelModule` implementations for those common
cases, so the two easiest-to-support methods (MSP, DOCTOR) work with near-zero
integration code for the majority of models.

## Scope

**In scope:** `_get_predictions` / `_get_probabilities` for scikit-learn-style
and PyTorch models.

**Out of scope (explicit follow-on, not detailed here):** auto-discovery of
`_get_activations` / `_get_pre_activations` for SPARDACUS, Mahalanobis and RED.
Automating "which layers, what counts as pre-activation vs activation" is
architecture-specific (see `examples/01-mnist/mlp_modelmodule.py`'s
`LAYER_BLOCKS`) and needs its own design once this phase lands.

**Considered and rejected for this phase:**
- TensorFlow/Keras support — de-prioritized as rarely used by the target
  users today. Can be added later following the same pattern as
  `TorchModelModule` if that changes.
- A generic `FunctionModelModule(fn)` wrapper around a bare callable — too
  little boilerplate saved over just subclassing `ModelModule` directly (the
  documented fallback) to justify a new public class.
- `DataModule` automation — out of scope; data loading/splitting is
  inherently project-specific and not addressed here.

## Architecture

New subpackage, mirroring the existing `methods/` package layout:

```
src/safetycage/modelmodules/
    __init__.py
    torch_modelmodule.py    -> TorchModelModule
    sklearn_modelmodule.py  -> SklearnModelModule
    auto_modelmodule.py     -> AutoModelModule
```

`src/safetycage/modelmodule.py` (the `ModelModule` ABC) is unchanged. All
three new classes ultimately produce a `ModelModule` instance.

## TorchModelModule

```python
TorchModelModule(
    model: torch.nn.Module,
    use_onehot_encoder: bool = False,
    device: str = "cpu",
    output_is_probabilities: bool = False,
)
```

- `_get_probabilities(x)`: converts `x` to a tensor on `device`, runs
  `model(x)` under `torch.no_grad()`, applies `softmax(dim=-1)` unless
  `output_is_probabilities=True` (torch models conventionally end in a
  `Linear` layer and output logits, not probabilities).
- `_get_predictions(x)`: `argmax` of `_get_probabilities(x)`; returns one-hot
  rows when `use_onehot_encoder=True`, matching the base class convention
  already used by `MLPModelModule` in the MNIST example.
- `_get_activations`, `_get_pre_activations`, `_calc_model_shape`: raise
  `NotImplementedError` with a message pointing to `docs/how-it-works.md` and
  the MNIST example for methods that need hidden-layer access.
- Import guarded the same way `methods/red.py` guards `torch`/`gpytorch`:
  ```python
  try:
      import torch
      HAS_TORCH = True
  except ImportError:
      HAS_TORCH = False
  ```
  Importing `safetycage.modelmodules` never fails without torch installed;
  instantiating `TorchModelModule` without it raises a clear `ImportError`
  naming `pip install safetycage[torch]`.

## SklearnModelModule

```python
SklearnModelModule(model, use_onehot_encoder: bool = False)
```

- Purely duck-typed — does **not** import `sklearn`. Works with any object
  implementing the sklearn convention: scikit-learn, XGBoost, LightGBM,
  CatBoost classifiers all qualify for free.
- `_get_predictions(x)` → `model.predict(x)`.
- `_get_probabilities(x)` → `model.predict_proba(x)`. If the model lacks
  `predict_proba` (e.g. an `SVC` without `probability=True`), raise
  `AttributeError` naming the missing method at call time — no
  `decision_function` fallback in this phase.
- Same `NotImplementedError` stubs for activation methods as
  `TorchModelModule`.

## AutoModelModule

```python
AutoModelModule(model, **kwargs) -> ModelModule
```

Implemented as a dispatching factory (`__new__` returns a concrete instance,
so `AutoModelModule` is never itself instantiated as a real object):

1. If torch is importable and `isinstance(model, torch.nn.Module)` →
   `TorchModelModule(model, **kwargs)`.
2. Elif `hasattr(model, "predict")` → `SklearnModelModule(model, **kwargs)`.
3. Else → raise `TypeError` stating what was checked (torch `nn.Module`,
   presence of `.predict`) and pointing to writing a custom `ModelModule`.

`**kwargs` are forwarded verbatim to whichever concrete class is chosen, so
`TorchModelModule`-only arguments like `device` or `output_is_probabilities`
are simply unused/invalid when a sklearn-style model is passed (documented,
not defended against).

## Packaging (`pyproject.toml`)

```toml
[project.optional-dependencies]
red = ["torch>=1.13", "gpytorch>=1.9"]
torch = ["torch>=1.13"]
spardacus = ["statsmodels>=0.14.6", "scikit-learn==1.9.0", "scipy", "tqdm"]
mahalanobis = ["statsmodels>=0.14.6", "scipy"]
```

`red`'s pin is unchanged in effect (still `torch` + `gpytorch`); the new
`torch` extra lets `TorchModelModule`-only users skip `gpytorch`.
`SklearnModelModule` needs no new extra — no import, no new dependency.

## Error handling summary

| Situation | Behavior |
| --- | --- |
| `TorchModelModule` used without `torch` installed | `ImportError` naming `safetycage[torch]` |
| `SklearnModelModule.model` lacks `predict_proba` | `AttributeError` at `_get_probabilities` call time, naming the missing method |
| `AutoModelModule` given a model matching neither branch | `TypeError` describing what was checked and pointing to a custom `ModelModule` |
| Any of the three used with SPARDACUS/Mahalanobis/RED | `NotImplementedError` on the activation methods, pointing to `docs/how-it-works.md` |

## Testing

- `TorchModelModule`: a tiny `nn.Sequential` (e.g. `Linear(4, 3)`), checking
  predictions and probabilities against a manual softmax computation; a case
  covering `output_is_probabilities=True`; a case covering
  `use_onehot_encoder=True`.
- `SklearnModelModule`: `sklearn.linear_model.LogisticRegression` fit on a toy
  dataset; a case where `predict_proba` is absent, asserting the
  `AttributeError`.
- `AutoModelModule`: dispatch test for a torch model, dispatch test for a
  sklearn-style model, and the `TypeError` case for an unrecognized object.
- A torch-not-installed path is only meaningfully testable in an environment
  without torch; note this in the test file rather than attempting to fake it
  under CI, and skip/mark it if torch is present.

## Docs impact

- `docs/how-it-works.md`: add a section after the two adapter classes,
  "Common models: TorchModelModule, SklearnModelModule, AutoModelModule",
  showing the near-zero-code path for MSP/DOCTOR against a plain sklearn or
  torch model. The current hand-rolled `SklearnModelModule` example in that
  file is replaced by a pointer to the real class (it exists now).
- `docs/index.md`: the "Choosing a method" table and worked example are
  unaffected; the "How it works" pointer paragraph stays as-is.
- `docs/api/core.md` (or a new `docs/api/modelmodules.md`, matching how
  `methods.md` documents the `methods/` package): document the three new
  classes via `autodoc`.

## Open questions for the implementation plan

- Exact wording/format of the `NotImplementedError` messages (should point at
  a concrete doc anchor, not just "see the docs").
- Whether `docs/conf.py`'s `autodoc_mock_imports = ["torch", "gpytorch"]`
  needs any adjustment so `TorchModelModule` still documents cleanly without
  torch installed in the docs build environment (likely no change needed,
  same mechanism already covers `red.py`).
