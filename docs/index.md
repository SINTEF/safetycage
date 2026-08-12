# Safetycage
Predictive models, especially neural networks, are increasingly deployed in
real-world and safety-critical settings. A well-documented failure mode is
that they fail silently: a model can be highly confident about a prediction
and still be wrong, and that confidence does not reliably drop just because
the input differs from what the model was trained on. Measuring accuracy once
on a held-out set says nothing about whether any single prediction made after
deployment can be trusted.

**Misclassification detection** addresses this at the level of individual
predictions. Given a model's output on a new input, a misclassification
detector estimates whether that specific prediction is likely correct or
likely wrong, without access to the true label, since none is available at
deployment time. This is a different problem from out-of-distribution (OOD)
detection: an input can be entirely in-distribution and still be
misclassified, and in safety-critical applications catching that case is
often what matters most. Predictions flagged as untrustworthy can then be
routed to a human reviewer, which is exactly the kind of error-resilience and
human-oversight support called for by regulation such as the EU AI Act.

safetycage collects several misclassification detection methods behind a
common interface: softmax-based baselines (MSP, DOCTOR), statistical tests
over hidden-layer activations (SPARDACUS, Mahalanobis), and an
uncertainty-aware residual model (RED). Swapping between them only requires
implementing the two adapter classes described below.

## Installation

```bash
pip install safetycage
# or: uv add safetycage
```

Some methods need extra dependencies, installed via `pip install
safetycage[extra]` (or `uv add "safetycage[extra]"`):

| Extra | Uses | Adds |
| --- | --- | --- |
| `red` | {py:class}`~safetycage.methods.red.RED` | `torch`, `gpytorch` |
| `spardacus` | {py:class}`~safetycage.methods.spardacus.SPARDACUS` | `statsmodels`, `scikit-learn`, `scipy`, `tqdm` |
| `mahalanobis` | {py:class}`~safetycage.methods.mahalanobis.Mahalanobis` | `statsmodels`, `scipy` |
| `torch` | {py:class}`~safetycage.modelmodules.torch_modelmodule.TorchModelModule` | `torch` |

## How it works

Adopting safetycage means writing two adapter classes for your own project: a
{py:class}`~safetycage.datamodule.DataModule` and a
{py:class}`~safetycage.modelmodule.ModelModule`. See [How it works](how-it-works)
for the minimal shape of each and a worked-through safety cage method run.

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
MLP and guards it with MSP, including a runnable notebook, rendered below.

```{toctree}
:maxdepth: 1
:caption: Examples

examples/mnist
```

```{toctree}
:maxdepth: 2
:caption: Contents

how-it-works
api/index
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
```
