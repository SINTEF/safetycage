"""Unit tests for TorchModelModule.

Run with: uv run --with pytest --all-extras pytest tests/test_torch_modelmodule.py -v
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from safetycage.modelmodules.torch_modelmodule import TorchModelModule


@pytest.fixture
def logits_model():
    torch.manual_seed(0)
    return nn.Linear(4, 3)


def test_get_probabilities_applies_softmax(logits_model):
    module = TorchModelModule(logits_model)
    x = np.random.RandomState(0).randn(5, 4).astype(np.float32)

    with torch.no_grad():
        expected = torch.softmax(logits_model(torch.as_tensor(x)), dim=-1).numpy()

    np.testing.assert_allclose(module._get_probabilities(x), expected, atol=1e-6)


def test_get_probabilities_sum_to_one(logits_model):
    module = TorchModelModule(logits_model)
    x = np.random.RandomState(1).randn(5, 4).astype(np.float32)
    probs = module._get_probabilities(x)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-6)


def test_get_predictions_is_argmax(logits_model):
    module = TorchModelModule(logits_model)
    x = np.random.RandomState(2).randn(5, 4).astype(np.float32)
    np.testing.assert_array_equal(
        module._get_predictions(x), np.argmax(module._get_probabilities(x), axis=1)
    )


def test_output_is_probabilities_flag_skips_softmax():
    torch.manual_seed(0)

    class AlreadySoftmax(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, x):
            return torch.softmax(self.linear(x), dim=-1)

    model = AlreadySoftmax()
    module = TorchModelModule(model, output_is_probabilities=True)
    x = np.random.RandomState(3).randn(5, 4).astype(np.float32)

    with torch.no_grad():
        expected = model(torch.as_tensor(x)).numpy()

    np.testing.assert_allclose(module._get_probabilities(x), expected, atol=1e-6)


def test_activations_not_implemented_without_selected_layers(logits_model):
    module = TorchModelModule(logits_model)
    x = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        module._get_activations(x)


@pytest.fixture
def block_model():
    """Linear -> ReLU, named so both can be selected as activation layers."""
    torch.manual_seed(0)
    model = nn.Sequential()
    model.add_module("linear", nn.Linear(4, 3))
    model.add_module("relu", nn.ReLU())
    return model


def test_pre_activations_not_implemented(block_model):
    module = TorchModelModule(block_model, selected_layers=["linear"])
    x = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        module._get_pre_activations(x)


def test_get_activations_captures_named_submodule_output(block_model):
    module = TorchModelModule(block_model, selected_layers=["linear", "relu"])
    x = np.random.RandomState(4).randn(5, 4).astype(np.float32)

    with torch.no_grad():
        linear_out = block_model.linear(torch.as_tensor(x)).numpy()
        relu_out = block_model.relu(torch.as_tensor(linear_out)).numpy()

    activations = module._get_activations(x)
    np.testing.assert_allclose(activations["linear"], linear_out, atol=1e-6)
    np.testing.assert_allclose(activations["relu"], relu_out, atol=1e-6)


def test_unknown_selected_layer_raises(logits_model):
    with pytest.raises(ValueError, match="no submodule"):
        TorchModelModule(logits_model, selected_layers=["nonexistent"])


def test_import_error_when_torch_missing(monkeypatch):
    """Simulates torch being unavailable, regardless of the test environment."""
    import safetycage.modelmodules.torch_modelmodule as torch_modelmodule

    monkeypatch.setattr(torch_modelmodule, "HAS_TORCH", False)
    with pytest.raises(ImportError, match=r"safetycage\[torch\]"):
        torch_modelmodule.TorchModelModule(model=object())
