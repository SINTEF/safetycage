"""
Unit tests for the RED (Residual-based Error Detection) safety cage method.

Uses lightweight mock ModelModule and DataModule objects so no real trained
model or dataset is needed.  Only requires gpytorch and torch to be installed
(pip install safetycage[red]).

Run with:
    uv run pytest tests/test_red.py -v
or:
    python -m pytest tests/test_red.py -v
"""

import numpy as np
import pytest
import tempfile
import os

# ---------------------------------------------------------------------------
# Skip the entire module if gpytorch / torch are not installed
# ---------------------------------------------------------------------------
gpytorch = pytest.importorskip("gpytorch")
torch = pytest.importorskip("torch")

from safetycage.methods.red import RED


# ---------------------------------------------------------------------------
# Mock ModelModule and DataModule
# ---------------------------------------------------------------------------
class MockModelModule:
    """Minimal stand-in for a ModelModule that wraps a simple numpy classifier."""

    def __init__(self, num_classes=3):
        self.use_onehot_encoder = False
        self.selected_layers = ["layer1"]
        self.last_layer = "layer1"
        self.model = None
        self._num_classes = num_classes
        # Build a trivial weight matrix once so predictions are deterministic
        self._rng = np.random.RandomState(0)
        self._W = self._rng.randn(num_classes)  # not used directly

    def _get_predictions(self, x):
        probs = self._get_probabilities(x)
        return np.argmax(probs, axis=1)

    def _get_probabilities(self, x):
        # Deterministic softmax-like probabilities derived from x
        logits = x[:, :self._num_classes] if x.shape[1] >= self._num_classes else np.hstack([x, np.zeros((len(x), self._num_classes - x.shape[1]))])
        logits = logits[:, :self._num_classes]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)


class MockDataModule:
    """Minimal stand-in for a DataModule."""

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_data():
    """Create a small synthetic dataset with 3 classes and 5 features."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_features = 5
    n_classes = 3

    x = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, size=n_samples)

    return x, y, n_classes


@pytest.fixture
def modules(synthetic_data):
    """Create mock model and data modules from synthetic data."""
    x, y, n_classes = synthetic_data
    model_module = MockModelModule(num_classes=n_classes)
    data_module = MockDataModule(x, y, num_classes=n_classes)
    return model_module, data_module


@pytest.fixture
def trained_red(modules):
    """Return a RED instance that has been trained on synthetic data."""
    model_module, data_module = modules
    red = RED(
        model_module, data_module,
        num_inducing_points=50,
        training_iterations=20,
        learning_rate=0.05,
        batch_size=64,
    )
    red.train_cage()
    return red


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestREDInit:
    def test_leq_is_true(self, modules):
        model_module, data_module = modules
        red = RED(model_module, data_module)
        assert red.leq is True

    def test_name(self, modules):
        model_module, data_module = modules
        red = RED(model_module, data_module)
        assert red.name == "RED"

    def test_default_kwargs(self, modules):
        model_module, data_module = modules
        red = RED(model_module, data_module)
        assert red.num_inducing_points == 500
        assert red.training_iterations == 100
        assert red.learning_rate == 0.01
        assert red.batch_size == 256

    def test_custom_kwargs(self, modules):
        model_module, data_module = modules
        red = RED(model_module, data_module, num_inducing_points=100, training_iterations=10)
        assert red.num_inducing_points == 100
        assert red.training_iterations == 10


class TestREDTraining:
    def test_train_populates_layer_params(self, trained_red):
        assert hasattr(trained_red, "layer_params")
        assert "gp_model" in trained_red.layer_params
        assert "likelihood" in trained_red.layer_params
        assert "input_dim" in trained_red.layer_params

    def test_input_dim_matches_features(self, trained_red, synthetic_data):
        x, _, _ = synthetic_data
        assert trained_red.layer_params["input_dim"] == x.shape[1]

    def test_train_with_explicit_data(self, modules, synthetic_data):
        """Train with explicitly passed x, y, y_pred instead of data_module defaults."""
        model_module, data_module = modules
        x, y, _ = synthetic_data
        y_pred = model_module._get_predictions(x)

        red = RED(model_module, data_module, num_inducing_points=30, training_iterations=5)
        red.train_cage(x=x, y=y, y_pred=y_pred)

        assert "gp_model" in red.layer_params


class TestREDPrediction:
    def test_predict_shape(self, trained_red, synthetic_data):
        x, y, _ = synthetic_data
        # Use a subset as "test" data
        x_test, y_test = x[:50], y[:50]
        y_pred_test = trained_red.model_module._get_predictions(x_test)

        scores = trained_red.predict(x_test, y_pred_test)
        assert scores.shape == (50,)

    def test_scores_are_finite(self, trained_red, synthetic_data):
        x, y, _ = synthetic_data
        y_pred = trained_red.model_module._get_predictions(x[:30])
        scores = trained_red.predict(x[:30], y_pred)
        assert np.all(np.isfinite(scores))

    def test_uncertainty_stored(self, trained_red, synthetic_data):
        x, y, _ = synthetic_data
        y_pred = trained_red.model_module._get_predictions(x[:30])
        trained_red.predict(x[:30], y_pred)

        assert trained_red._last_uncertainty is not None
        assert trained_red._last_uncertainty.shape == (30,)
        assert np.all(trained_red._last_uncertainty >= 0)


class TestREDFlagging:
    def test_flag_produces_boolean_array(self, trained_red, synthetic_data):
        x, y, _ = synthetic_data
        y_pred = trained_red.model_module._get_predictions(x[:50])
        scores = trained_red.predict(x[:50], y_pred)

        trained_red.alpha = 0.5
        flags = trained_red.flag(scores)
        assert flags.dtype == bool
        assert flags.shape == (50,)

    def test_flag_with_explicit_alpha(self, trained_red, synthetic_data):
        x, y, _ = synthetic_data
        y_pred = trained_red.model_module._get_predictions(x[:50])
        scores = trained_red.predict(x[:50], y_pred)

        flags = trained_red.flag(scores, alpha=0.5)
        # With leq=True, flagged means score <= alpha
        assert np.all(flags == (scores <= 0.5))

    def test_find_best_threshold(self, trained_red, synthetic_data):
        x, y, _ = synthetic_data
        y_pred = trained_red.model_module._get_predictions(x)
        scores = trained_red.predict(x, y_pred)

        # Binary misclassification labels
        y_true_misclassified = (y_pred != y).astype(int)

        def simple_accuracy(TP, TN, FP, FN):
            total = TP + TN + FP + FN
            return (TP + TN) / total if total > 0 else 0

        result = trained_red.find_best_threshold(
            y_true=y_true_misclassified,
            y_probs=scores,
            metric_fn=simple_accuracy,
        )
        assert "alpha_opt" in result
        assert "metric_max" in result
        assert isinstance(result["alpha_opt"], float)


class TestREDSaveLoad:
    def test_save_and_load_roundtrip(self, trained_red, synthetic_data, modules):
        model_module, data_module = modules
        x, y, _ = synthetic_data
        y_pred = trained_red.model_module._get_predictions(x[:30])

        # Get predictions before save
        trained_red.alpha = 0.5
        scores_before = trained_red.predict(x[:30], y_pred)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "red_cage.joblib")
            trained_red.save_cage(save_path)

            # Verify files created
            assert os.path.exists(save_path)
            pt_path = save_path.replace(".joblib", ".pt")
            assert os.path.exists(pt_path)

            # Load and compare
            loaded_red = RED.load_cage(save_path, model_module, data_module)
            assert loaded_red.alpha == 0.5
            scores_after = loaded_red.predict(x[:30], y_pred)

            np.testing.assert_allclose(scores_before, scores_after, atol=1e-5)

    def test_save_without_alpha_raises(self, trained_red):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "red_cage.joblib")
            with pytest.raises(ValueError, match="alpha is not set"):
                trained_red.save_cage(save_path)
