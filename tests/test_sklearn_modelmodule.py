"""Unit tests for SklearnModelModule.

Run with: uv run --with pytest --all-extras pytest tests/test_sklearn_modelmodule.py -v
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression

from safetycage.modelmodules.sklearn_modelmodule import SklearnModelModule


class _PredictOnlyModel:
    """Stands in for a classifier that only implements predict(), no predict_proba."""

    def predict(self, x):
        return np.zeros(len(x), dtype=int)


@pytest.fixture
def fitted_logistic_regression():
    rng = np.random.RandomState(0)
    x = rng.randn(50, 4)
    y = (x[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(x, y)
    return model, x


def test_get_predictions_matches_model(fitted_logistic_regression):
    model, x = fitted_logistic_regression
    module = SklearnModelModule(model)
    np.testing.assert_array_equal(module._get_predictions(x), model.predict(x))


def test_get_probabilities_matches_model(fitted_logistic_regression):
    model, x = fitted_logistic_regression
    module = SklearnModelModule(model)
    np.testing.assert_allclose(module._get_probabilities(x), model.predict_proba(x))


def test_get_probabilities_without_predict_proba_raises():
    module = SklearnModelModule(_PredictOnlyModel())
    with pytest.raises(AttributeError, match="predict_proba"):
        module._get_probabilities(np.zeros((3, 1)))


def test_activations_not_implemented(fitted_logistic_regression):
    model, x = fitted_logistic_regression
    module = SklearnModelModule(model)
    with pytest.raises(NotImplementedError):
        module._get_activations(x)
    with pytest.raises(NotImplementedError):
        module._get_pre_activations(x)
