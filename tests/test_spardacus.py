"""Unit tests for SPARDACUS.

Run with: uv run --with pytest --all-extras pytest tests/test_spardacus.py -v
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
statsmodels = pytest.importorskip("statsmodels")
scipy = pytest.importorskip("scipy")

from safetycage.methods.spardacus import SPARDACUS


class MockModelModule:
    """Minimal ModelModule stand-in exposing fixed, separable activations.

    ``x`` is used purely as an index into precomputed activations/predictions
    arrays, so the mock never needs real model internals.
    """

    def __init__(self, activations, predictions):
        self.use_onehot_encoder = False
        self.selected_layers = ["layer1"]
        self._activations = activations
        self._predictions = predictions

    def _get_predictions(self, x):
        return self._predictions[x]

    def _get_activations(self, x):
        return {"layer1": self._activations[x]}


class MockDataModule:
    def __init__(self, x_train, y_train, num_classes):
        self._x_train = x_train
        self._y_train = y_train
        self._num_classes = num_classes

    @property
    def data_train(self):
        return self._x_train, self._y_train

    @property
    def classes(self):
        return {i: str(i) for i in range(self._num_classes)}

    @property
    def num_classes(self):
        return self._num_classes


@pytest.fixture
def separable_cage():
    """Two classes, well-separated correct/incorrect activation clusters.

    20 samples per (true_class, correct/incorrect) bucket -- comfortably
    above SPARDACUS's default minimum_sample_size=10 for GMM fitting.
    """
    rng = np.random.RandomState(0)
    n_per_bucket = 20
    n_features = 5

    x_indices, y_true, predictions, activations = [], [], [], []

    idx = 0
    for true_class in (0, 1):
        for pred_class, center in [(true_class, 0.0), (1 - true_class, 5.0)]:
            for _ in range(n_per_bucket):
                activations.append(rng.normal(loc=center, scale=0.5, size=n_features))
                y_true.append(true_class)
                predictions.append(pred_class)
                x_indices.append(idx)
                idx += 1

    model_module = MockModelModule(np.array(activations), np.array(predictions))
    data_module = MockDataModule(np.array(x_indices), np.array(y_true), num_classes=2)

    return model_module, data_module, np.array(x_indices), np.array(y_true)


def test_predict_after_train_cage_does_not_raise(separable_cage):
    """Regression test for a numpy 2.x incompatibility: ECDF(...) returns a
    shape-(1,) array, and numpy 2.x no longer allows assigning that into a
    scalar array slot (``arr[i, j] = np.array([x])``), raising
    "setting an array element with a sequence." This used to work under
    numpy 1.x's implicit squeeze.
    """
    model_module, data_module, x_indices, y_true = separable_cage

    spardacus = SPARDACUS(
        model_module=model_module, data_module=data_module, s_statistic_source="correctly"
    )
    spardacus.train_cage()

    statistics = spardacus.predict(x_indices, y_true)

    assert statistics.shape == (len(y_true),)
    assert np.all(np.isfinite(statistics) | np.isnan(statistics))
