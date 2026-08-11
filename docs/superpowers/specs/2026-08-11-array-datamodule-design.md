# Design: a built-in DataModule for pre-split array/DataFrame data

Date: 2026-08-11

## Problem

Adopting safetycage requires writing a `DataModule` subclass. For the common
case where the user already has train/val/test splits sitting in memory as
numpy arrays or pandas objects (e.g. after calling `sklearn.datasets.fetch_*`
and `train_test_split` themselves), this is pure boilerplate: `how-it-works.md`
already sketches this exact case as `ArrayDataModule`, but it is docs-only —
there is no shipped class, and the base `DataModule.__init__` forces a
`data_dir` argument that makes no sense for in-memory data (the sketch works
around it with `data_dir="."`).

This design ships that sketch as a real class, extended to also accept
pandas DataFrame/Series input, so the common "I already have my splits"
case needs zero adapter code.

## Scope

**In scope:** one class, `ArrayDataModule`, wrapping caller-provided
train/val/test splits (`x_train, y_train, x_val, y_val, x_test, y_test`) as
numpy arrays or pandas DataFrame/Series, normalized to numpy internally.

**Out of scope:**

- **Auto-splitting.** The class does not split a single `X, y` blob itself.
  Splitting stays the caller's job (e.g. `sklearn.model_selection.train_test_split`),
  matching the existing `examples/02-xgboost` sketch, which already splits
  before touching safetycage.
- **`AutoDataModule` / format-specific subclasses.** Unlike
  `TorchModelModule` vs `SklearnModelModule`, numpy arrays and pandas
  DataFrame/Series need no different *behavior* here — a DataFrame is just
  normalized to a numpy array once in `__init__`. There is nothing to
  dispatch between, so no factory class is introduced. (Models split
  cleanly by *framework*; this kind of data does not split cleanly by
  *format* in a way that changes behavior.)
- **A `Bunch`-shaped input** (raw `sklearn.utils.Bunch` from `load_*`/`fetch_*`,
  before the caller has pulled out `.data`/`.target`). Not requested; the
  caller already does this unpacking before calling `train_test_split` in
  the existing example.

## ArrayDataModule

```python
ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test, **kwargs)
```

- All six positional/keyword arguments are required — matching the
  train/estimate-threshold/evaluate three-way split every safety cage
  workflow in `how-it-works.md` and the MNIST example already uses: `train`
  fits the cage itself (needed by RED/SPARDACUS/Mahalanobis, which actually
  train something; MSP/DOCTOR ignore it), `val` estimates the optimal
  threshold (`find_best_threshold`), `test` evaluates that threshold on
  held-out data.
- Each of `x_train`, `y_train`, `x_val`, `y_val`, `x_test`, `y_test` may be a
  numpy array or a pandas `DataFrame`/`Series`. Pandas input is converted via
  `.to_numpy()` in `__init__`, so `self.data_train`/`data_val`/`data_test`
  are always `(np.ndarray, np.ndarray)` tuples regardless of what was passed
  in — callers and safety cage methods never need to care which format the
  caller originally had.
- No `data_dir` argument on the public constructor. Internally calls
  `super().__init__(data_dir=".", **kwargs)` exactly like the existing
  `how-it-works.md` sketch, since the base class requires *some* path but
  this class never reads or writes to it.
- `classes`/`num_classes` derived from `np.unique(y_train)` (post-normalization),
  matching the existing sketch: `{c: str(c) for c in np.unique(y_train)}`.
- `dataset_name` returns `"custom"` (matching the sketch) — there is no
  dataset-name metadata to derive this from generically.
- `setup`/`_load_data`/`_transform`/`_split` are no-ops (`pass`), since data
  arrives pre-loaded and pre-split — matching the sketch exactly.
- `to_joblib`/`from_joblib` delegate to `joblib.dump`/`joblib.load` on
  `self`, matching the sketch.

## Architecture

New subpackage, mirroring `src/safetycage/modelmodules/` (no `__init__.py`,
same namespace-package convention already established there):

```
src/safetycage/datamodules/
    array_datamodule.py    -> ArrayDataModule
```

`src/safetycage/datamodule.py` (the `DataModule` ABC) is unchanged.

## Docs impact

- `docs/how-it-works.md`: replace the current hand-rolled `ArrayDataModule`
  example under "The DataModule" with a pointer to the real class, the same
  way the `SklearnModelModule` example there was replaced by a pointer to
  the real class in the earlier ModelModule work.
- New `docs/api/datamodules.md`, mirroring `docs/api/modelmodules.md`'s
  shape, with an `automodule` directive for `safetycage.datamodules.array_datamodule`.
- `docs/api/index.md`'s toctree gains `datamodules` alongside `core`,
  `methods`, `modelmodules`, `utils`.

## Testing

- Construction from plain numpy arrays: `data_train`/`data_val`/`data_test`
  round-trip correctly, `classes`/`num_classes` match `np.unique(y_train)`.
- Construction from pandas `DataFrame`/`Series` input: same assertions,
  confirming the `.to_numpy()` normalization — `data_train[0]` etc. must be
  `np.ndarray`, not a DataFrame, after construction.
- `to_joblib`/`from_joblib` round-trip preserves the six arrays.
- No `data_dir` argument required to construct the class.
