"""TorchModelModule: wraps a plain torch.nn.Module.

Requires the lightweight `torch` extra (pip install safetycage[torch]), not
the heavier `red` extra (torch + gpytorch).
"""
from typing import Any

import numpy as np

from safetycage.modelmodule import ModelModule

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TorchModelModule(ModelModule):
    """Wraps a torch.nn.Module for MSP/DOCTOR.

    Assumes the model outputs raw logits (the usual torch convention of
    ending in a Linear layer) and applies softmax to get probabilities.
    Pass output_is_probabilities=True if the model already ends in a
    softmax/sigmoid.

    Supports MSP and DOCTOR only. SPARDACUS, Mahalanobis and RED need
    hidden-layer activations, which this class does not provide — write a
    custom ModelModule for those (see docs/how-it-works.md and
    examples/01-mnist/mlp_modelmodule.py).
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        output_is_probabilities: bool = False,
        **kwargs: Any,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "TorchModelModule requires torch. Install it with "
                "`pip install safetycage[torch]`."
            )
        super().__init__(selected_layers=[], use_onehot_encoder=False, model=model, **kwargs)
        self.device = torch.device(device)
        self.output_is_probabilities = output_is_probabilities
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, x: np.ndarray):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)

    def _get_probabilities(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            output = self.model(self._to_tensor(x))
            if not self.output_is_probabilities:
                output = torch.softmax(output, dim=-1)
        return output.cpu().numpy()

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self._get_probabilities(x), axis=1)

    def _get_activations(self, x: np.ndarray):
        raise NotImplementedError(
            "TorchModelModule exposes no intermediate layers. SPARDACUS, "
            "Mahalanobis and RED need hidden-layer activations - write a "
            "custom ModelModule (see docs/how-it-works.md)."
        )

    def _get_pre_activations(self, x: np.ndarray):
        raise NotImplementedError(
            "TorchModelModule exposes no intermediate layers. SPARDACUS, "
            "Mahalanobis and RED need hidden-layer pre-activations - write "
            "a custom ModelModule (see docs/how-it-works.md)."
        )
