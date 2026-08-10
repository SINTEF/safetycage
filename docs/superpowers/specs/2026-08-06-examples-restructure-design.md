# Examples restructure: migrate tutorials into `examples/` and port to PyTorch

Date: 2026-08-06
Status: Partly superseded — see the amendment below

## Amendment, 2026-08-10: the zero-dependency lead example was dropped

This spec opens with **iris** as a zero-dependency lead example, ahead of mnist
and cifar10. That lead example no longer exists in any form, and the plan that
would have built it has been deleted.

The reasoning is recorded here because the measurements are worth keeping. iris
has 150 samples, yielding a 24-sample validation split with **one**
misclassification, so threshold selection would optimise a metric computed from
a single positive — it could not demonstrate the thing it existed to
demonstrate. wine (178 samples) is worse, at zero validation errors. digits was
adopted instead: 1797 samples, 10 classes, and a 288-sample validation split
with 33 errors under a depth-capped RandomForest.

digits was then dropped too. `examples/01-mnist/` was built directly and covers
the same ground with a real neural network, making a scikit-learn warm-up
redundant.

**What still stands:** the layout convention, the split between `modules.py`,
`train.py` and the notebook, the dependency-group strategy, `examples/utils.py`,
and the testing approach. **What does not:** the `iris/` → `mnist/` → `cifar10/`
progression below, and the flat directory names — the actual convention is
numbered (`01-mnist/`). Read the layout section with that substitution.

## Problem

The worked examples for safetycage live in a separate repository,
`github.com/safety-cage/safetycage-tutorials`. Keeping them there has three costs.

**They drift silently.** The package has moved to 0.0.54 while the tutorials pin
`safetycage==0.0.3`. Two API changes have already broken them without anyone
noticing at the time: `find_best_threshold` moved onto the `SafetyCage` object,
and `max_probs_*` was renamed `pvalues_*` in v0.0.5. Nothing catches this until
a human opens a notebook.

**They carry a stale fork of the library.** `cifar10/mahalanobis.py` (481 lines)
and `mnist/msp.py` (94 lines) are near-identical copies of the package's own
`methods/mahalanobis.py` (478) and `methods/msp.py` (91). The examples are
demonstrating a private fork of the code they exist to demonstrate.

**They are hard to install.** The tutorials pin `tensorflow==2.15.0` plus
`tensorflow-io-gcs-filesystem==0.31.0`, which has no arm64 macOS wheel. A plain
`uv run` on an Apple Silicon machine fails outright.

`safetycage-pypi/examples/` already exists and is empty. This design fills it.

## Goals

- Examples live beside the code they document, so API changes and example
  updates land in the same commit.
- Examples are structured to be executable and testable, not merely readable.
- The published package gains no new runtime dependency.
- A reader can see what adopting safetycage actually costs them.

## Non-goals

- Setting up CI. The structure makes automated testing possible; wiring
  `.gitlab-ci.yml` is deliberately deferred. Tests run manually via `pytest`
  until then.
- Deciding git-history handling for the migration, or the fate of the
  `safetycage-tutorials` repository afterwards. Both deferred.
- Demonstrating the `red` method. The torch port makes it feasible for the first
  time, but it is out of scope here.

## Design

### Layout

```
safetycage-pypi/examples/
├── README.md                     # index: start with iris → mnist → cifar10
├── utils.py                      # fix_pythonpath_if_working_locally()
├── iris/
│   ├── README.md
│   ├── iris_example.ipynb        # MSP
│   ├── modules.py                # IrisDataModule + IrisModelModule
│   └── train.py                  # RandomForest fit
├── mnist/
│   ├── README.md
│   ├── mnist_example.ipynb       # MSP → Doctor → Spardacus
│   ├── model.py                  # torch MLP
│   ├── modules.py                # MNISTDataModule + TorchModelModule
│   └── train.py
└── cifar10/
    ├── README.md
    ├── cifar10_example.ipynb     # Mahalanobis → Spardacus
    ├── model.py                  # torch CNN
    ├── modules.py                # CIFAR10DataModule + TorchModelModule
    └── train.py
```

### Splitting code between notebook and modules

Code is divided by whether it teaches safetycage, not by class.

`modules.py` holds the `DataModule` and `ModelModule` subclasses. This is the
lesson. `DataModule` declares 9 abstract methods and `ModelModule` declares 4;
implementing them is the entire integration contract a user must satisfy to
adopt safetycage. The file is the artifact a reader copies into their own
project, so it stays tight and heavily commented.

`train.py` holds model fitting. This is boilerplate that teaches nothing about
safetycage, and it is kept out of the notebook so the notebook does not spend
dozens of cells on a training loop.

The notebook carries narration and safetycage usage. It imports from
`modules.py` and renders the subclass source inline with `inspect.getsource`, so
a reader on the web sees the contract without the code being duplicated in two
places. The file remains the single source of truth: importable, testable, and
copy-pasteable.

**Each example's `modules.py` is self-contained.** The mnist and cifar10
`ModelModule` subclasses share most of their hook logic but are deliberately not
factored into a shared base class or a common `examples/_shared.py`. A reader
must be able to copy one directory and have a complete, working reference; a
shared base would mean copying two files and mentally subtracting the parts that
belong to the other example. The cost is real duplication between the two, which
is accepted: these are teaching artifacts, not production code, and the
duplication is bounded at two files that change rarely. If a third torch example
is ever added, revisit this.

Rejected alternatives: importing the subclasses without displaying them hides
the integration contract behind an import, which is the one thing a prospective
user most needs to see. Defining the subclasses inline in notebook cells makes
them readable but not importable, so they cannot be tested.

### One notebook per dataset

Each dataset gets a single notebook that sets up data and model once, then walks
through each method on that shared setup. Today mnist has three notebooks
(doctor, msp, spardacus) at roughly 200 code-lines each that repeat the same
preamble. Consolidating removes the duplication and lets a reader compare
methods on identical data.

### Resolving `import safetycage` to the working tree

`examples/utils.py` provides a single helper, called at the top of each notebook:

```python
import sys
from pathlib import Path


def fix_pythonpath_if_working_locally():
    """Make `import safetycage` resolve to this repo's src/ when run from a clone."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src" / "safetycage").is_dir():
            sys.path.insert(0, str(candidate / "src"))
            return
```

```python
from utils import fix_pythonpath_if_working_locally

fix_pythonpath_if_working_locally()
```

This is adapted from the equivalent helper in the `darts` project. **It is
deliberately not a copy of theirs, and the differences are load-bearing.** The
darts version reads:

```python
if basename(cwd) == "examples":
    sys.path.insert(0, dirname(cwd))
```

Transplanted unchanged it would fail here in two independent ways, both
silently:

1. **darts uses a flat layout; this repo uses a src layout.** In darts the
   package sits at the repository root, so inserting `dirname(cwd)` is enough.
   Here the package is at `src/safetycage/`, so `dirname(cwd)` points at a
   directory containing no `safetycage/` package and the insert does nothing.

2. **darts keeps notebooks directly in `examples/`; this design nests them one
   level deeper.** The guard `basename(cwd) == "examples"` is false for
   `examples/mnist/`, so the function would return without acting.

Walking up the parents until a `src/safetycage/` directory is found fixes both,
and works regardless of how deeply a notebook is nested.

The helper is a safety net, not the supported path. `uv sync` installs the
project editable — verified to resolve `import safetycage` to
`src/safetycage/` — so the documented workflow in each README is:

```bash
uv sync --group examples
uv run jupyter lab examples/mnist/mnist_example.ipynb
```

The helper matters for the case `uv sync` does not cover: someone who opens a
notebook without syncing, with a released safetycage already installed
elsewhere, would otherwise silently exercise the published version instead of
the working tree.

This is unrelated to importing each example's own `modules.py`. Jupyter places
the notebook's directory on `sys.path`, so a notebook in `examples/mnist/`
imports its sibling `modules.py` without help. Only `pytest` needs assistance
there, which the testing section covers.

### Dependencies

```toml
[dependency-groups]
examples = ["ipykernel>=7", "jupyter", "torch>=2.0", "torchvision>=0.15"]
```

A PEP 735 dependency group, not a `[project.optional-dependencies]` extra. The
distinction matters: `red` is correctly an extra because RED is a runtime
feature of the package that requires torch. Examples are not a feature anyone
installs — they are local tooling. A dependency group keeps these out of the
published wheel's metadata entirely, where an extra would advertise them.

`torchvision` supplies the MNIST and CIFAR-10 loaders, replacing
`keras.datasets` and the hand-rolled `_download_data` / npz caching.

Install with `uv sync --group examples`.

### PyTorch port

The package core is framework-agnostic — it contains no TensorFlow or Keras
references, and `torch` appears only as a guarded import inside
`methods/red.py`. `SafetyCage` reads only `data_module.num_classes` and
`data_module.classes`; it never consumes `tf.data.Dataset`. The port is
therefore confined to the examples.

`ModelModule.selected_layers` is already typed as layer-name strings, which maps
directly onto `dict(model.named_modules())`. Activations come from **forward
hooks** rather than `keras.backend.function`.

Layers are declared with the nonlinearity separate:

```python
nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(784, 256)), ("relu1", nn.ReLU()),
    ("fc2", nn.Linear(256, 128)), ("relu2", nn.ReLU()),
    ("fc3", nn.Linear(128, 10)),
]))
```

Pre-activation is then a hook on `fc1` and post-activation a hook on `relu1`.
Under Keras, `Dense(activation="relu")` fuses the linear operation with the
nonlinearity, which is why the current `_get_pre_activations` has to reconstruct
the linear part by hand from layer inputs. In torch the distinction is
structural, and that method collapses to a single hook registration.

`_calc_model_shape` reads `.out_features` instead of `layer.output_shape[1]`.

Because `nn.CrossEntropyLoss` takes class indices rather than one-hot vectors,
`use_onehot_encoder` is `False` throughout the ported examples.

Torch also has native arm64 macOS wheels with MPS support, so `device="mps"` is
correct on Apple Silicon rather than a workaround, and the platform-marker
complexity around `tensorflow-macos` / `tensorflow-metal` disappears.

**cifar10 is the one non-mechanical part.** Convolutional activations are shaped
`(N, C, H, W)` where the MLP's are `(N, F)`. The CNN's `_calc_model_shape`, and
any flattening the methods assume, need genuine attention rather than a copy of
the MLP implementation. This may expose shape assumptions in the methods that
the MLP examples never exercise.

### Testing

`tests/test_examples.py` loads each example's `modules.py`, builds the
datamodule, fits or loads a model, runs one method, and asserts the shape and
range of `pvalues_`. Run manually with `pytest`.

iris needs no dependency beyond the package core — it uses
`sklearn.datasets.load_iris` (bundled, no download), `RandomForestClassifier`and `StandardScaler`, and scikit-learn is already a core dependency. It is
therefore the cheapest example to check and the natural first target once CI
exists.

### Removed in the migration

- The forked `cifar10/mahalanobis.py` and `mnist/msp.py`.
- `pyrootutils` bootstrapping and `.project-root`, unnecessary once the examples
  sit inside the package repository.
- The Keras `metadata/` cache format (226MB, already gitignored).
- Duplicated per-method notebook preambles.

## Risks

**This is a rewrite, not a move.** Porting three examples to torch and rewriting
the narration is substantially more work than relocating files. Estimating it as
a file move will underestimate it.

**The existing narration is thin.** Each notebook currently carries 19–24
markdown lines across roughly 25 cells, which is closer to a script than a
tutorial. Migrating as-is would carry that over. The structure in this design
only pays off if the notebooks explain why each abstract method exists and what
a reader must do in their own project. This is the main source of effort after
the port itself.

**cifar10 is the long pole**, for the CNN port and for the conv-shaped
activation question above.

**Committed notebook outputs will drift.** With no CI, nothing detects that a
notebook's committed outputs no longer match what the code produces. This is
accepted for now and is the strongest argument for revisiting CI.

## Open questions

- Git history: preserve via subtree merge, or land as a clean commit.
- What happens to `safetycage-tutorials` afterwards: archive with a pointer, or
  delete.
- When to wire up `.gitlab-ci.yml`, and whether to relax the exact
  `requires-python = "==3.11.7"` pin to make runner images easier to source.
