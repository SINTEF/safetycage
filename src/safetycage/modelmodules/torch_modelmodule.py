"""TorchModelModule: wraps a plain torch.nn.Module.

Requires the lightweight `torch` extra (pip install safetycage[torch]), not
the heavier `red` extra (torch + gpytorch).
"""
from typing import Any, Dict, List, Union

import numpy as np

from safetycage.modelmodule import ModelModule

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TorchModelModule(ModelModule):
    """Wraps a torch.nn.Module for MSP/DOCTOR, and SPARDACUS given selected_layers.

    Assumes the model outputs raw logits (the usual torch convention of
    ending in a Linear layer) and applies softmax to get probabilities.
    Pass output_is_probabilities=True if the model already ends in a
    softmax/sigmoid.

    Supports MSP and DOCTOR out of the box. Passing ``selected_layers``
    (names from ``model.named_modules()``) additionally unlocks SPARDACUS by
    hooking those submodules and returning their raw forward output as the
    "activation" — since this class wraps an arbitrary ``nn.Module`` with no
    known block structure, there is no separate pre-activation/activation
    split the way there is for an architecture-specific ModelModule like
    ``examples/01-mnist/mlp_modelmodule.py``. Mahalanobis and RED need that
    genuine pre-nonlinearity signal, which this class still does not
    provide — write a custom ModelModule for those (see
    docs/how-it-works.md).

    Construction moves ``model`` to ``device`` and switches it to eval mode
    in place — it is not copied — so a caller who needs the model to stay on
    its current device, or to remain in train mode afterward, should pass in
    a copy instead.

    ``_get_predictions``'s ``np.argmax(..., axis=1)`` assumes a multi-class
    output (2+ classes); a binary model with a single sigmoid logit of shape
    ``(N, 1)`` is not supported and will silently produce all-zero
    predictions.

    Input arrays are cast to ``float32`` in ``_to_tensor`` regardless of the
    model's actual dtype — a double-precision model will raise a dtype
    mismatch from torch.
    """

    def __init__(
        self,
        model: Any,
        device: str = "cpu",
        output_is_probabilities: bool = False,
        selected_layers: Union[str, List[str]] = [],
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: The wrapped ``torch.nn.Module``.
            device: Where to run the forward passes. (default: "cpu")
            output_is_probabilities: Whether ``model`` already ends in
                softmax/sigmoid, skipping the internal softmax. (default: False)
            selected_layers: Names from ``model.named_modules()`` whose forward
                output ``_get_activations`` should capture. Leave empty to use
                MSP/DOCTOR only. (default: [])
            batch_size: Forward-pass batch size when capturing activations.
                (default: 512)
        """
        if not HAS_TORCH:
            raise ImportError(
                "TorchModelModule requires torch. Install it with "
                "`pip install safetycage[torch]`."
            )
        super().__init__(
            selected_layers=selected_layers, use_onehot_encoder=False, model=model, **kwargs
        )
        self.device = torch.device(device)
        self.output_is_probabilities = output_is_probabilities
        self.batch_size = kwargs.get("batch_size", 512)
        self.model.to(self.device)
        self.model.eval()

        # Keyed on the leaf name, so a selected name resolves regardless of nesting.
        self._modules_by_name = {
            name.rsplit(".", 1)[-1]: module
            for name, module in self.model.named_modules()
            if name
        }

        missing = [name for name in self.selected_layers if name not in self._modules_by_name]
        if missing:
            raise ValueError(
                f"Model has no submodule(s) named {missing}. Pass selected_layers "
                "matching names from model.named_modules()."
            )

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

    @torch.no_grad()
    def _forward_capture(self, x: Any, module_names: List[str]) -> Dict[str, np.ndarray]:
        """Run the model, capturing the output of each named submodule.

        Forward hooks rather than a rebuilt sub-network, so the captured values
        are the ones the real model produces.
        """
        x = self._to_tensor(x)

        captured: Dict[str, List[torch.Tensor]] = {name: [] for name in module_names}
        handles = []

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                captured[name].append(output.detach().flatten(start_dim=1).cpu())

            return hook

        for name in module_names:
            handles.append(self._modules_by_name[name].register_forward_hook(make_hook(name)))

        try:
            for start in range(0, len(x), self.batch_size):
                self.model(x[start : start + self.batch_size])
        finally:
            for handle in handles:
                handle.remove()

        return {
            name: torch.cat(batches).numpy().astype(np.float64)
            for name, batches in captured.items()
        }

    def _get_activations(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Raw forward-hook output of each selected submodule, keyed by name.

        SPARDACUS indexes these as ``activations[layer][mask, :]``, so each
        value must be a 2D array aligned with the input rows.
        """
        if not self.selected_layers:
            raise NotImplementedError(
                "TorchModelModule needs selected_layers to expose hidden-layer "
                "activations - pass names from model.named_modules(), e.g. "
                "TorchModelModule(model, selected_layers=['relu1'])."
            )
        return self._forward_capture(x, self.selected_layers)

    def _get_pre_activations(self, x: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError(
            "TorchModelModule exposes no pre-activations. Mahalanobis and RED "
            "need a genuine pre-nonlinearity signal - write a custom "
            "ModelModule (see docs/how-it-works.md)."
        )
