# Changelog

## v0.0.60 (11/08/2026)

### Feature

- The MNIST tutorial notebook is now rendered on the documentation site
  under "Examples", not just linked to on GitHub.
- Three new `ModelModule` implementations for models that don't need
  hand-written adapters: `safetycage.modelmodules.sklearn_modelmodule.SklearnModelModule`
  wraps any classifier with `.predict`/`.predict_proba` (scikit-learn,
  XGBoost, LightGBM, CatBoost and friends);
  `safetycage.modelmodules.torch_modelmodule.TorchModelModule` wraps a plain
  `torch.nn.Module` and requires the new `torch` extra
  (`pip install safetycage[torch]`);
  `safetycage.modelmodules.auto_modelmodule.AutoModelModule` inspects the
  model and dispatches to whichever of the two applies. All three support
  MSP and DOCTOR only — none of them expose hidden-layer activations, so
  SPARDACUS, Mahalanobis and RED still need a hand-written `ModelModule`.

### Fix

- `SafetyCage.roc_curve()` and `SafetyCage.auroc()` renamed their second
  parameter from `statistics` to `y_pred`, matching the rest of the metrics
  API.
- SPARDACUS and Mahalanobis's API reference pages were silently empty on
  the published docs site: Read the Docs only installs the `docs`
  dependency group, so autodoc's import of `safetycage.methods.spardacus`
  (needs `sklearn`/`scipy`/`statsmodels`/`tqdm`) and `.mahalanobis` (needs
  `scipy`/`statsmodels`) failed and Sphinx skipped their content with only
  a build-log warning. Added them to `autodoc_mock_imports` alongside the
  existing `torch`/`gpytorch` mocks for RED.

### Breaking

- `tqdm` moved out of the package's core dependencies into the `spardacus`
  extra, since only SPARDACUS uses it. Anyone relying on `tqdm` being
  installed transitively via plain `pip install safetycage` now needs
  `pip install safetycage[spardacus]` or their own `tqdm` dependency.
- `scikit-learn` dropped from the `mahalanobis` extra — Mahalanobis never
  imported it. Anyone relying on `scikit-learn` being installed transitively
  via `pip install safetycage[mahalanobis]` now needs
  `pip install safetycage[spardacus]` or their own `scikit-learn` dependency.

## v0.0.58 (10/08/2026)

### Breaking

- Removed `safetycage.utils.evaluate`. Its contents split by what they need:
  - The seven metric functions, `calculate_confusion_rates` and
    `calculate_metrics` moved to `safetycage.utils.metrics`, which works on
    labels alone. Importing them from `evaluate` used to work by accident —
    they were re-exported — so import from `safetycage.utils.metrics` instead.
  - `AUROC` and `calculate_roc_curve` became `SafetyCage.auroc()` and
    `SafetyCage.roc_curve()`. A threshold sweep has to know which direction a
    method flags in, which is cage state, so it belongs on the cage. Both now
    call `self.flag()`, which means they follow a subclass that overrides
    `flag()` — SPARDACUS, for one — with nothing extra to configure.
  - `calculate_roc_curve` also lost its unused `num_thresholds`,
    `threshold_min` and `threshold_max` arguments. The sweep visits the unique
    statistics, so there is nothing left to tune.
- `calculate_metrics` no longer takes a `metric_functions` argument. It always
  reports the registry, now named `METRIC_FUNCTIONS` (was `metric_functions`).
  To compute one metric, or one outside the registry, apply it to the confusion
  counts directly: `MCC(**calculate_confusion_rates(y, y_pred))`.
- Removed `safetycage.utils.functions_library`. Its contents moved onto the
  method classes as static methods, under the same names:
  - `CauchyCombinationTest` → `SPARDACUS.CauchyCombinationTest` *and*
    `Mahalanobis.CauchyCombinationTest`. Both methods used it, so each class
    now carries its own copy.
  - `fastSPARDA`, `l1SPARDA`, `randomProjectionSearch`,
    `projectedWasserstein`, `gmm_bic_score` → `SPARDACUS`.
- Removed `safetycage.utils.plot_functions`, replaced by
  `safetycage.utils.visualise`. `plot_confusion_matrix`, `plot_roc_curve` and
  `annotate_text_box` are gone — use scikit-learn's `ConfusionMatrixDisplay`
  and `RocCurveDisplay` instead. `plot_alpha_metric_curve` is now
  `plot_metric_vs_threshold`, which takes one series rather than a hardcoded
  validation/test pair, accepts an `ax` and returns it, and no longer saves or
  closes figures. Overlay by passing the same `ax` twice. The statistic
  histogram it used to draw on a twin axis is now `plot_statistic_distribution`.
- `requires-python` is now `>=3.13`, up from an exact `==3.11.7` pin. The
  package no longer installs on 3.11 or 3.12. Most dependencies relaxed to
  floors in exchange; `numpy` and `scikit-learn` stay pinned for now, since the
  methods are sensitive to both.

### Fix

- SPARDACUS no longer raises `AttributeError` on any dataset with an unreliable
  class. It assigned `np.NaN` to those p-values, and numpy 2 removed that
  spelling — it is `np.nan`.
- `AUROC` no longer raises either. It called `np.trapz`, also removed in
  numpy 2, and now uses `np.trapezoid`. Its integration additionally broke ties
  by FPR alone, which cut the corner off every vertical step and understated
  the area. `SafetyCage.auroc()` now matches `sklearn.metrics.roc_auc_score`
  exactly.
- The ROC sweep pads its thresholds with ±inf, so the curve reaches both (0, 0)
  and (1, 1) whichever direction the cage flags in, and no longer uses NaN
  statistics as thresholds. A comparison against NaN is always False, which
  added a spurious "nothing flagged" point.
- `plot_statistic_distribution` drops NaN statistics. SPARDACUS emits NaN for
  samples it cannot score, and matplotlib rejects a histogram whose range is
  not finite, so the old code raised on real SPARDACUS output.

## v0.0.57 (10/08/2026)

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