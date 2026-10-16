# Changelog

## v0.0.65 (16/10/2026)

### Fix

- `SPARDACUS.fastSPARDA()` set the sample count with
  `n = X_samples.reshape(-1)`, which flattens the whole activation matrix
  into one array instead of counting its rows. It now reads
  `n, d = X_samples.shape`. The default single-`lambdas` path never used
  `n`, so results there are unchanged; only the cross-validation fold
  sizes (more than one `lambdas` value) depended on it.

## v0.0.64 (16/10/2026)

### License

- Relicensed from MIT to the Apache License 2.0. `LICENSE` now holds the
  full Apache 2.0 text, and the copyright notice moved to a new `NOTICE`
  file, which ships in the wheel alongside `LICENSE`. Releases up to and
  including v0.0.63 remain available under MIT.

## v0.0.63 (28/08/2026)

### Breaking

- `SafetyCage.train_cage()` renamed to `fit()` across the base class and
  every method (MSP, DOCTOR, RED, SPARDACUS, Mahalanobis, RandomBaseline).
  Callers need to switch `cage.train_cage()` to `cage.fit()`.
- `RED` and `SPARDACUS` each carried their own `alpha` attribute, separate
  from the base class's `threshold` — `flag(alpha=...)`, `self.alpha`, and
  a `find_best_threshold()` override on `RED` that renamed its result key
  from `threshold_opt` to `alpha_opt`. All of that is gone: both methods
  now use `self.threshold` like every other method, and `RED.flag()` /
  `RED.find_best_threshold()` are fully inherited from the base class.
  Anyone setting `cage.alpha`, calling `cage.flag(alpha=...)`, or reading
  `result["alpha_opt"]` needs to switch to `threshold`. `RED.save_cage()` /
  `load_cage()` now persist `threshold`, not `alpha` — cages saved by
  earlier versions need to be re-fitted and saved again.

### Feature

- `SafetyCage.fit_threshold()`: runs the predict → `find_best_threshold()`
  pipeline every example notebook was writing by hand against the
  validation split, and sets `self.threshold`. Every `fit()` gained
  matching opt-in keyword arguments (`find_threshold=False`, `metric_fn`,
  `x_val`, `y_val`, `search`, `num_thresholds`, `greater_is_better`) that
  route into it, so `cage.fit(find_threshold=True)` trains and picks a
  threshold in one call. Default behavior (`find_threshold=False`) is
  unchanged, and `find_best_threshold()` remains fully usable on its own.
- `safetycage.methods.random_baseline.RandomBaseline`: a misclassification
  baseline that flags samples at random, weighted by how often the model
  is wrong on the predicted class (learned in `fit()`, from the training
  split). Lets other methods be judged against a chance-adjusted baseline
  at the same review budget, rather than only against a do-nothing
  baseline.
- `examples/03-cifar10-cnn`: added a "confusion matrix as a graph" section
  (treats the confusion matrix as a directed adjacency matrix and
  visualizes it with networkx/nxviz), a `RandomBaseline` comparison
  alongside the real cage in both deployment scenarios, and a
  gain-per-review metric — `(with-review metric − no-cage metric) /
  fraction flagged` — that normalizes a method's payoff by its review
  cost, so methods that flag very different fractions of the test set can
  be compared on equal footing.
- New `experiments/` module: systematic sweeps across (dataset, model)
  pipelines and safety cage methods, complementing `examples/`'s one
  guided walkthrough per notebook. `python -m experiments.run --pipeline
  <name> --safetycage <name>` fits and evaluates any registered
  combination, with `--retrain` (train a fresh model from scratch instead
  of loading its checkpoint) and `--cage-kwargs` (JSON constructor
  overrides, e.g. RED's `training_iterations`) for per-run tuning. Results
  are upserted into `experiments/results.json`, summarized to
  `experiments/compare.md`, and every invocation is logged to
  `experiments/logs/`.

### Fix

- `find_best_threshold()` now casts `threshold_opt` and `metric_max` to
  native `float`. Previously, a `float32` statistic array — RED's, for
  instance — could produce a `threshold_opt` of type `np.float32`, which
  (unlike `np.float64`) is not a subclass of Python's `float`; RED's own
  `find_best_threshold()` override happened to mask this by casting its
  renamed `alpha_opt` explicitly, which is gone now that the override is.
- `docs/how-it-works.md` and `paper/paper.md` still showed `train_cage()`
  and a `result["alpha_opt"]` / `cage.find_best_threshold_flag()` API that
  no longer exists (the latter never existed under that name). Updated to
  the current `fit()` / `threshold_opt` API.

## v0.0.62 (19/08/2026)

### Feature

- `safetycage.datamodules.array_datamodule.ArrayDataModule`: a ready-made
  `DataModule` for train/val/test splits you already have in memory as
  numpy arrays or pandas DataFrame/Series — no adapter code needed for the
  common case of data that's already loaded and split. Accepts either
  format for any split, normalizing pandas input to numpy internally.
- `TorchModelModule` now accepts `selected_layers` (names from
  `model.named_modules()`) and implements `_get_activations`, unlocking
  SPARDACUS without a hand-written `ModelModule`. Mahalanobis and RED still
  need a genuine pre-nonlinearity signal this class can't provide, so
  `_get_pre_activations` remains unimplemented.
- `examples/03-cifar10-cnn`: a CNN trained on CIFAR-10, guarded with RED.
  Compares two deployment scenarios (abstain on what's flagged vs. review
  and correct what's flagged) and records results to a shared
  `examples/results.json`, so runs across different methods and examples
  can be compared side by side.
- `examples/04-activity-monitor`: a GRU trained on the UCI Daily and
  Sports Activities dataset (19 activities from 5 body-worn IMU sensors),
  comparing MSP against SPARDACUS side by side. The first example to wrap
  its model with the generic `TorchModelModule` directly instead of a
  hand-written `ModelModule`.
- `MNISTDataModule` now accepts pre-split `x_train`/`y_train`/etc. arrays
  directly, matching `ArrayDataModule`/`CIFAR10DataModule`'s constructor
  pattern — lets a notebook split data once (e.g. into disjoint
  model-training and cage-training halves) without re-deriving
  `MNISTDataModule`'s own split logic.
- All three example notebooks (MNIST, CIFAR-10 CNN, activity-monitor) are
  now rendered on the documentation site under "Examples", not just
  linked to on GitHub.

### Fix

- `xgboost` was listed as a core dependency in `pyproject.toml`, but
  nothing in `src/safetycage` imports it — only `examples/02-xgboost`
  does. Moved to the `examples` dependency-group so installing
  `safetycage` doesn't pull in a heavy, unused dependency.

- `RED.flag()` was missing entirely: `RED` inherited `SafetyCage.flag()`,
  which reads `self.threshold`, but `RED.save_cage`/`load_cage` only ever
  set `self.alpha`, so flagging a trained `RED` cage raised `ValueError:
  Missing threshold parameter`. `RED` now defines its own `flag()` and
  `find_best_threshold()`, mirroring the self-contained `alpha`-based
  pattern `SPARDACUS` already uses.

## v0.0.61 (11/08/2026)

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
- Added a project logo, shown in the documentation site's sidebar
  (`docs/_static/logo-2.png`, wired up via `html_logo` in `docs/conf.py`).

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