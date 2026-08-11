"""SklearnModelModule: wraps any classifier with predict/predict_proba.

Duck-typed — does not import scikit-learn itself, so this also covers
XGBoost, LightGBM and CatBoost classifiers, which follow the same
convention.
"""
from typing import Any, List

import numpy as np

from safetycage.modelmodule import ModelModule


class SklearnModelModule(ModelModule):
    """Wraps a classifier implementing predict()/predict_proba().

    Supports MSP and DOCTOR only. SPARDACUS, Mahalanobis and RED need
    hidden-layer activations, which this class does not provide — write a
    custom ModelModule for those (see docs/how-it-works.md).
    """

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(selected_layers=[], use_onehot_encoder=False, model=model, **kwargs)

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def _get_probabilities(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"{type(self.model).__name__} has no predict_proba method. "
                "MSP and DOCTOR need class probabilities; a model exposing "
                "only predict() cannot be used with SklearnModelModule for "
                "those methods."
            )
        return self.model.predict_proba(x)

    def _get_activations(self, x: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError(
            "SklearnModelModule exposes no intermediate layers. SPARDACUS, "
            "Mahalanobis and RED need hidden-layer activations - write a "
            "custom ModelModule (see docs/how-it-works.md)."
        )

    def _get_pre_activations(self, x: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError(
            "SklearnModelModule exposes no intermediate layers. SPARDACUS, "
            "Mahalanobis and RED need hidden-layer pre-activations - write "
            "a custom ModelModule (see docs/how-it-works.md)."
        )
