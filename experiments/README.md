# Experiments

Systematic sweeps across (dataset, model) pipelines and safety cage methods.
This is the counterpart to `examples/`: the notebooks in `examples/` are a
guided walkthrough of *one* pipeline and *one* method at a time; this folder
is for cross-checking many combinations against each other with as little
boilerplate as possible, reusing `fit(find_threshold=True)` so each
combination trains and picks its own threshold in one call.

## Quickstart

Each pipeline reuses the dataset + trained model checkpoint from its
matching `examples/` notebook, so run that notebook once first if the
checkpoint doesn't exist yet (`examples/01-mnist/mlp_mnist.pt` for `mnist`,
`examples/03-cifar10-cnn/cnn_cifar10.pt` for `cifar10`).

```bash
# one pipeline, one method
uv run python -m experiments.run --pipeline mnist --safetycage msp

# every registered method on both pipelines
uv run python -m experiments.run \
    --pipeline mnist cifar10 \
    --safetycage msp doctor red spardacus mahalanobis random_baseline

# 'all' expands to every registered name, so this is equivalent and stays
# correct as pipelines/methods get added later
uv run python -m experiments.run --pipeline all --safetycage all
```

There's no separate "dataset" and "model" axis to cross freely — a pipeline
*is* a dataset+model pair (see "Adding a pipeline" below), because models
here are architecturally tied to one dataset's input shape. `--pipeline`
picks among the valid pairs; only `--safetycage` is a genuinely free axis to
sweep against every pipeline.

Run it from the repository root (`python -m experiments.run`, not
`python experiments/run.py`) so the `experiments` package import resolves.

`red`'s registered defaults (`training_iterations=100`,
`num_inducing_points=500`) took ~10 minutes on CPU for the `mnist` pipeline
(~7.6k cage-train samples) -- for fast iteration while exploring, override a
cage's constructor kwargs with `--cage-kwargs` (a JSON object) from the CLI:

```bash
uv run python -m experiments.run --pipeline mnist --safetycage red \
    --cage-kwargs '{"training_iterations": 20, "num_inducing_points": 100}'
```

or the same thing from Python:

```python
from experiments.run import run_experiment
run_experiment("mnist", "red", cage_kwargs={"training_iterations": 20, "num_inducing_points": 100})
```

`--cage-kwargs` applies to *every* `--safetycage` name in that invocation, so
keep it scoped to one method at a time -- mixing `--safetycage red msp` with
RED-only kwargs like `training_iterations` will fail MSP's constructor for
that combination (logged, not fatal to the rest of the sweep).

Each `(pipeline, safetycage)` combination that succeeds appends one row to
`experiments/results.json` (upserted by `(pipeline, method)`, so re-running a
combination replaces its own row rather than duplicating it). Run
`uv run python experiments/compare.py` afterward to regenerate
`experiments/compare.md`, a flat table across everything you've run — same
idea as `examples/build_compare.py`, kept as a separate file since these are
sweep runs, not the one curated result per method each notebook records.
A combination that raises (missing checkpoint, a method that can't fit this
pipeline, ...) is logged and skipped rather than aborting the whole sweep.

Every invocation also writes its full console output to a timestamped file
under `experiments/logs/` (already covered by the repo's `**/logs/` /
`*.log` gitignore rules, so nothing here gets committed) — useful for
looking back at exactly what a past sweep run did or why a combination
failed, without having to re-run it.

Each metric is reported for three columns — `no-cage` (the raw model),
`cage-kept` (Scenario 1: abstain on everything flagged), `with-review`
(Scenario 2: an expert reviews every flagged sample and corrects the wrong
ones) — plus a fourth, `gain/review`: the with-review gain over no-cage,
divided by the fraction of the test set flagged. That normalization is what
lets you compare a cage that flags 5% of samples against one that flags 40%
on equal footing, rather than the raw gain rewarding whichever one just
flagged more.

Both `mnist` and `cifar10` accept `--retrain`, training a fresh model from
scratch on the model-half split (same split, architecture, optimizer, and
hyperparameters as the matching notebook's own training cell) instead of
loading its checkpoint. Not persisted to disk — retrained again on every
call. Useful for checking a cage method against a newly initialized model
rather than the one checkpoint already on disk.

## Adding a pipeline

A "pipeline" bundles a dataset with the model trained on it, because the two
aren't independent here — a CNN expects `(N, 3, 32, 32)` CIFAR-10 images,
not flat MNIST vectors. Add a new one in `registry.py`:

1. Write a function shaped like `_mnist_pipeline`: load your `DataModule`'s
   cage-half split (disjoint from whatever trained the model), load the
   trained model into its `ModelModule`, return `(model_module, data_module)`.
2. Register it in `PIPELINES` under a short name.
3. If a new safety cage method needs configuring for it, add a
   `(SafetyCageClass, default_kwargs)` entry to `SAFETYCAGES` in the same
   file — every registered pipeline is offered every registered method, so
   this is the one place to add a new method for the whole grid.

## Why no Hydra

The actual axis of variation here is small and shaped like two flat
registries (`PIPELINES`, `SAFETYCAGES`) crossed against each other — a
config-composition framework buys the most when there are many independent
config groups, nested overrides, and cluster-submission sweeps to manage.
`argparse` + `itertools`-style looping covers this without a new dependency
or a second config language to learn on top of the code itself. If the sweep
grid grows into genuinely many independent axes (augmentation configs,
multiple hyperparameter sets per method, cluster submission), Hydra becomes
worth it — `registry.py`'s two dicts would translate directly into Hydra
config groups at that point.

## Why the model/data code stays in examples/

`registry.py` reaches *into* `examples/01-mnist/` and `examples/03-cifar10-cnn/`
for their `DataModule`/`ModelModule`/model classes and trained checkpoints,
rather than those living in `experiments/` with the notebooks importing them
back out. That direction is deliberate: each notebook under `examples/` is
meant to be a complete, self-contained tutorial you can open and read top to
bottom without jumping to another folder -- that's the whole point of it
being a tutorial. Moving the model code into `experiments/` would break that
for the sake of tidying up a dependency arrow that only exists because this
folder was added *after* and *on top of* `examples/`, as a secondary tool
for systematic sweeps. There's also no real duplication to eliminate by
moving anything: `mnist`'s MLP/DataModule and `cifar10`'s CNN/DataModule
share no code with each other today, so relocating them wouldn't shrink
anything, just relocate the one existing coupling (`registry.py` -> `examples/`)
without removing it.

## Known limitations

- **Mahalanobis** currently only works when the model's `last_layer` (the
  final entry of `selected_layers`, per `ModelModule`) produces a *scalar*
  value per class. For an ordinary hidden layer (e.g. MNIST's `dense_1`,
  256-dim) with `use_onehot_encoder=False`, `Mahalanobis._compute_statistics`
  routes into `two_sided_test`, which assumes a scalar activation and raises
  `ValueError: The truth value of an array with more than one element is
  ambiguous` for a vector one. This is a limitation in `Mahalanobis` itself
  (`src/safetycage/methods/mahalanobis.py`), not the experiment harness --
  it's registered here so the failure is visible per-pipeline rather than
  silently skipped.
