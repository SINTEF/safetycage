"""ModelModule for the MNIST MLP.

This is the integration contract on the model side: safetycage reaches the
network only through these methods. The methods differ in what they need —
MSP and Doctor want class probabilities, SPARDACUS wants activations, and
Mahalanobis wants pre-activations plus ``last_layer`` — so all of it is
implemented here.

Layer names are the *blocks* of the MLP, not the individual ``nn.Module``
children. For ``dense_1`` the pre-activation is the ``Linear`` output and the
activation is the ``ReLU`` output; for the final ``dense_3`` the
pre-activation is the logits and the activation is the softmax.
"""
from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn

from safetycage.modelmodule import ModelModule

# block name -> (module producing the pre-activation, module producing the activation)
LAYER_BLOCKS = OrderedDict(
    [
        ("dense_1", ("dense_1", "relu_1")),
        ("dense_2", ("dense_2", "relu_2")),
        ("dense_3", ("dense_3", "softmax")),
    ]
)


class MLPModelModule(ModelModule):
    """Wraps the PyTorch MLP so safetycage can read predictions and activations."""

    AVAILABLE_LAYERS = tuple(LAYER_BLOCKS)

    def __init__(
        self,
        selected_layers: Union[str, List[str]],
        use_onehot_encoder: bool,
        model: nn.Module,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            selected_layers: Any of ``dense_1``, ``dense_2``, ``dense_3``.
            use_onehot_encoder: Whether ``_get_predictions`` returns one-hot rows.
                Must match the DataModule, since SPARDACUS compares them directly.
            model: The ``MLP``.
            device: Where to run the forward passes. (default: "cpu")
            batch_size: Forward-pass batch size. Activations are held in memory
                for every sample passed in, so this only bounds peak activation
                memory on the GPU, not the returned arrays. (default: 512)
            last_layer: Block Mahalanobis treats as the output.
                (default: the last of ``selected_layers``)
        """
        super().__init__(selected_layers, use_onehot_encoder, model, **kwargs)

        invalid = [n for n in self.selected_layers if n not in self.AVAILABLE_LAYERS]
        if invalid:
            raise ValueError(
                f"Unsupported selected_layers: {invalid}. "
                f"Choose from {list(self.AVAILABLE_LAYERS)}."
            )

        self.device = torch.device(kwargs.get("device", "cpu"))
        self.batch_size = kwargs.get("batch_size", 512)

        self.model = model.to(self.device)
        self.model.eval()

        # Keyed on the leaf name, so "dense_1" resolves whether the modules sit
        # directly on the model or nested under an nn.Sequential attribute.
        self._modules_by_name = {
            name.rsplit(".", 1)[-1]: module
            for name, module in self.model.named_modules()
            if name
        }

        missing = [
            name
            for block in self.selected_layers
            for name in LAYER_BLOCKS[block]
            if name not in self._modules_by_name
        ]
        if missing:
            raise ValueError(
                f"Model has no submodule(s) named {missing}. This ModelModule "
                "expects the named layers built by examples/01-mnist/mlp.py."
            )

        self.last_layer = kwargs.get("last_layer", self.selected_layers[-1])

    def _to_tensor(self, x: Any) -> torch.Tensor:
        """Accept numpy or torch, single sample or batch."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x, dtype=np.float32))

        x = x.to(device=self.device, dtype=torch.float32)

        return x.unsqueeze(0) if x.ndim == 1 else x

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

    def _get_probabilities(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities, shape ``(N, num_classes)``.

        Not abstract on the base class, but MSP and Doctor both call it.
        The model ends in ``Softmax``, so this is simply its output.
        """
        return self._forward_capture(x, ["softmax"])["softmax"]

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        """Predicted classes, as indices or one-hot rows."""
        probabilities = self._get_probabilities(x)
        predicted = np.argmax(probabilities, axis=1).astype(np.int64)

        if self.use_onehot_encoder:
            return np.eye(probabilities.shape[1], dtype=np.float64)[predicted]

        return predicted

    def _get_activations(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Post-nonlinearity output of each selected block, keyed by block name.

        SPARDACUS indexes these as ``activations[layer][mask, :]``, so each
        value must be a 2D array aligned with the input rows.
        """
        wanted = {block: LAYER_BLOCKS[block][1] for block in self.selected_layers}
        captured = self._forward_capture(x, list(wanted.values()))

        return {block: captured[name] for block, name in wanted.items()}

    def _get_pre_activations(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Linear output of each selected block, before the nonlinearity.

        For ``dense_3`` these are the logits — Mahalanobis fits Gaussians to
        these, and softmax outputs would be a poor fit for that.
        """
        wanted = {block: LAYER_BLOCKS[block][0] for block in self.selected_layers}
        captured = self._forward_capture(x, list(wanted.values()))

        return {block: captured[name] for block, name in wanted.items()}

    def _calc_model_shape(self) -> Dict[str, int]:
        """Units per block. Nothing here uses it, but ``ModelModule`` declares
        it abstract, so the class cannot be instantiated without it."""
        return {
            block: self._modules_by_name[LAYER_BLOCKS[block][0]].out_features
            for block in self.AVAILABLE_LAYERS
            if LAYER_BLOCKS[block][0] in self._modules_by_name
        }