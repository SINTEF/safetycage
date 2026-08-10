# Utilities

## Metrics

Confusion counts and classification metrics, computed from labels alone at a
single, fixed threshold.

```{note}
Threshold *sweeps* live on the cage, not here: see `SafetyCage.roc_curve()` and
`SafetyCage.auroc()` under [Core](core.md). They need to know which direction a
method flags in, which is cage state.
```

```{eval-rst}
.. automodule:: safetycage.utils.metrics
```

## Plotting

```{eval-rst}
.. automodule:: safetycage.utils.visualise
```
