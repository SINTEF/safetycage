"""Integration tests wiring SklearnModelModule into real safety cage methods.

The unit tests for SklearnModelModule/TorchModelModule/AutoModelModule call
_get_predictions/_get_probabilities directly. These tests instead run a
fitted classifier through MSP and DOCTOR end to end, the way a user would.

Run with:
    uv run --with pytest --all-extras pytest tests/test_modelmodules_integration.py -v
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression

from safetycage.methods.doctor import DOCTOR
from safetycage.methods.msp import MSP
from safetycage.modelmodules.sklearn_modelmodule import SklearnModelModule


class MockDataModule:
    """Minimal stand-in for a DataModule (see tests/test_red.py)."""

    def __init__(self, x_train, y_train, num_classes=3):
        self._x_train = x_train
        self._y_train = y_train
        self._num_classes = num_classes

    @property
    def data_train(self):
        return self._x_train, self._y_train

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def classes(self):
        return {i: str(i) for i in range(self._num_classes)}


@pytest.fixture
def modules():
    """Fit a LogisticRegression on synthetic data and wrap/mock it."""
    rng = np.random.RandomState(0)
    n_samples = 200
    n_features = 5
    n_classes = 3

    x = rng.randn(n_samples, n_features)
    y = rng.randint(0, n_classes, size=n_samples)

    model = LogisticRegression().fit(x, y)
    model_module = SklearnModelModule(model)
    data_module = MockDataModule(x, y, num_classes=n_classes)
    return model_module, data_module, x, y


def test_msp_predict_via_sklearn_modelmodule(modules):
    model_module, data_module, x, y = modules
    msp = MSP(model_module=model_module, data_module=data_module)
    msp.train_cage()
    statistics = msp.predict(x, y)

    assert statistics.shape == (len(x),)
    assert np.all(np.isfinite(statistics))


def test_doctor_predict_via_sklearn_modelmodule(modules):
    model_module, data_module, x, y = modules
    doctor = DOCTOR(model_module=model_module, data_module=data_module, method="max")
    doctor.train_cage()
    statistics = doctor.predict(x, y)

    assert statistics.shape == (len(x),)
    assert np.all(np.isfinite(statistics))
