# safetycage

Misclassification detection for predictive models.

A classifier that is wrong and *knows* it is wrong is far more useful than one
that is wrong confidently. safetycage wraps a trained model and flags the
predictions it is likely to have got wrong, so they can be reviewed rather
than trusted.

## Installation

```bash
pip install safetycage
```

The RED method additionally needs `torch` and `gpytorch`:

```bash
pip install safetycage[red]
```

```{warning}
safetycage currently declares `requires-python = "==3.11.7"`, an exact pin.
Installation will fail on any other Python version, including 3.11.6 and
3.11.9.
```

## How it works

Adopting safetycage means writing two adapter classes for your own project:

- {py:class}`~safetycage.datamodule.DataModule` — declares how your data is
  loaded, transformed and split, and what the classes are.
- {py:class}`~safetycage.modelmodule.ModelModule` — declares how to get
  predictions, probabilities and intermediate activations out of your model.

Everything safetycage needs, it gets through those two objects. A safety cage
method then consumes them:

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

## Choosing a method

| Method | Needs | Works with |
| --- | --- | --- |
| {py:class}`~safetycage.methods.msp.MSP` | class probabilities | any classifier |
| {py:class}`~safetycage.methods.doctor.DOCTOR` | class probabilities | any classifier |
| {py:class}`~safetycage.methods.spardacus.SPARDACUS` | layer activations | neural networks only |
| {py:class}`~safetycage.methods.mahalanobis.Mahalanobis` | layer pre-activations | neural networks only |
| {py:class}`~safetycage.methods.red.RED` | layer activations | neural networks only, needs `[red]` |

Start with MSP. It needs no fitting and works with anything that produces
class probabilities.

## Worked examples

The [`examples/`](https://github.com/SINTEF/safetycage/tree/main/examples)
directory contains complete integrations. `examples/01-mnist` wraps a PyTorch
MLP and guards it with MSP, including a runnable notebook.

```{toctree}
:maxdepth: 2
:caption: Contents

api/index
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
```
