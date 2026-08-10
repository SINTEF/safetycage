# Changelog

## v0.0.57 (10/08/2026)

No code changes — this release exists to correct the package metadata, which
PyPI freezes per release and cannot be edited after upload.

### Feature

- Documentation is now published at <https://safetycage.readthedocs.io/> and
  linked from the PyPI project page.

### Fix

- Project URLs now point at <https://github.com/SINTEF/safetycage>. Releases up
  to v0.0.56 linked to the `safety-cage` account, which no longer hosts the
  repository, so the Homepage and Issues links on PyPI were dead.

## v0.0.56 (07/08/2026)

> Versions 0.0.6 through 0.0.55 were published without changelog entries.
> This entry covers everything that changed since v0.0.5.

### Breaking

- Removed the module-level `safetycage.utils.evaluate.find_best_threshold()`.
  Use the `SafetyCage.find_best_threshold()` method instead. Note the method
  takes no `leq` argument — it reads `self.leq` via `self.flag()` — so callers
  passing `leq=` explicitly need to drop it.
- `SafetyCage.save_cage()` / `.load_cage()` now use a single joblib file rather
  than the previous multi-file format, and `load_cage()` is a classmethod.
  Cages saved by earlier versions need to be re-fitted and saved again.

### Feature

- Added the RED method (`safetycage.methods.red`). It needs `gpytorch` and
  `torch`, available through the `red` extra: `pip install safetycage[red]`.
- Added `examples/`, starting with a worked MNIST example in
  `examples/01-mnist`: a PyTorch MLP wrapped in a `DataModule` and
  `ModelModule` and guarded by MSP. Notebook and example tooling live in a
  PEP 735 `examples` dependency group, so none of it reaches the published
  wheel.

### Fix

- Corrected type annotations on `DataModule` and `ModelModule`.

## v0.0.5 (24/04/2026)

### Fix

- Cleaning up the parameter names in plot_alpha_metric_curve()

### Feature

- Added the alpha distributions to the plot_alpha_metric_curve()
- Added self.unreliable_classes to be optionally saved or loaded in safetycage.save_cage() and .load_cage()
- Add remove_nan_values() method to SPARDACUS.

<!-- 
Commented out to reserve an unedited version for reference.

### Fix

- Changed confusing error message in plotting.plot_words()

### Feature

- Added a "stop_words" argument to pycounts.count_words()

### Documentation

- Added new usage examples
- Now hosting documentation on Read the Docs
-->


## v0.0.4 (21/04/2026)

- Simple updates to readme to test publishing by different authors

### Fix

- Removing ABC folder

## v0.0.2 and v0.0.2 (20/04/2026)

- Quick edits while doing a live testing session

## v0.1.0 (24/08/2021)

- First release of `safetycage`. Stable but may contain bugs