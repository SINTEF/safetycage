"""Unit tests for AutoModelModule.

Run with: uv run --with pytest --all-extras pytest tests/test_auto_modelmodule.py -v
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression

from safetycage.modelmodules.auto_modelmodule import AutoModelModule
from safetycage.modelmodules.sklearn_modelmodule import SklearnModelModule
from safetycage.modelmodules.torch_modelmodule import TorchModelModule


def test_dispatches_to_torch_for_nn_module():
    module = AutoModelModule(nn.Linear(4, 3))
    assert isinstance(module, TorchModelModule)


def test_dispatches_to_sklearn_for_predict_object():
    rng = np.random.RandomState(0)
    x = rng.randn(20, 4)
    y = (x[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(x, y)

    module = AutoModelModule(model)
    assert isinstance(module, SklearnModelModule)


def test_raises_type_error_for_unrecognized_model():
    with pytest.raises(TypeError, match="ModelModule"):
        AutoModelModule(object())


def test_kwargs_forwarded_to_torch_module():
    module = AutoModelModule(nn.Linear(4, 3), output_is_probabilities=True)
    assert module.output_is_probabilities is True
