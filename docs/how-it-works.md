# How it works

Adopting safetycage means writing two adapter classes for your own project:

- {py:class}`~safetycage.datamodule.DataModule` — declares how your data is
  loaded, transformed and split, and what the classes are.
- {py:class}`~safetycage.modelmodule.ModelModule` — declares how to get
  predictions, probabilities and intermediate activations out of your model.

Everything safetycage needs, it gets through those two objects. This page
shows the minimal shape of each one; for a full integration against a real
PyTorch model, see the [MNIST example](examples/mnist).

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
if your model already ends in softmax/sigmoid. Constructing
`TorchModelModule` moves `model` to `device` and switches it to eval mode in
place, not on a copy, so pass in a copy if you need the original to stay on
its device or in train mode.

Passing `selected_layers` (names from `model.named_modules()`) additionally
unlocks SPARDACUS: each selected submodule is hooked and its raw forward
output is returned as the activation. Mahalanobis and RED still need a
custom `ModelModule`, since they need a genuine pre-nonlinearity signal that
a generic named-submodule hook can't provide.

{py:class}`~safetycage.modelmodules.auto_modelmodule.AutoModelModule` picks
between the two for you:

```python
from safetycage.modelmodules.auto_modelmodule import AutoModelModule

model_module = AutoModelModule(model)  # torch.nn.Module or a .predict-style object
```

Any `**kwargs` `AutoModelModule` receives that the dispatched class's
constructor doesn't recognize are silently absorbed with no effect — e.g.
passing `device=` for a model that dispatches to `SklearnModelModule` does
nothing, no error.

None of these three expose hidden-layer activations — SPARDACUS, Mahalanobis
and RED still need a hand-written `ModelModule` (see the MNIST example).

## Running a safety cage method

A safety cage method consumes the two objects:

```python
from safetycage.methods.msp import MSP
from safetycage.utils.metrics import MCC

msp = MSP(model_module=model_module, data_module=data_module)
msp.train_cage()

# Confidence statistic per sample; low means "likely wrong".
statistics = msp.predict(x_val, y_val)

# Pick a threshold on validation data...
incorrect = (model_module._get_predictions(x_val) != y_val).astype(int)
result = msp.find_best_threshold(y_true=incorrect, y_probs=statistics, metric_fn=MCC)

# ...then apply it to held-out data.
flagged = msp.flag(msp.predict(x_test, y_test), result["alpha_opt"])
```
