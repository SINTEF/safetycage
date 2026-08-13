"""ModelModule for the CIFAR-10 CNN.

This is the integration contract on the model side: safetycage reaches the
network only through these methods. SPARDACUS wants activations, and
Mahalanobis wants pre-activations plus ``last_layer``; MSP and DOCTOR want
class probabilities, and RED only needs predictions/probabilities too -- it
never touches hidden-layer activations, see ``safetycage.methods.red.RED``'s
docstring. All four methods work through this one class.

Layer names are the *blocks* of the CNN, not the individual ``nn.Module``
children. For ``conv1``/``conv2`` the pre-activation is the BatchNorm output
(before the ReLU) and the activation is the pooled, post-ReLU output that
actually feeds the next block. For ``fc1`` the pre-activation is the Linear
output and the activation is the ReLU output. For the final ``fc2`` the
pre-activation is the logits; there is no activation module to hook (the
model outputs raw logits, the usual torch convention), so its "activation"
is produced by applying softmax manually instead.
"""
from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn

from safetycage.modelmodule import ModelModule

# block name -> (module producing the pre-activation, module producing the
# activation). fc2's activation is None: handled specially below, since
# there is no softmax submodule to hook.
LAYER_BLOCKS = OrderedDict(
    [
        ("conv1", ("conv1_bn", "pool1")),
        ("conv2", ("conv2_bn", "pool2")),
        ("fc1", ("fc1", "relu_fc1")),
        ("fc2", ("fc2", None)),
    ]
)


class CNNModelModule(ModelModule):
    """Wraps the CIFAR-10 CNN so safetycage can read predictions and activations."""

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
            selected_layers: Any of ``conv1``, ``conv2``, ``fc1``, ``fc2``.
            use_onehot_encoder: Whether ``_get_predictions`` returns one-hot rows.
                Must match the DataModule, since SPARDACUS compares them directly.
            model: The ``CNN``.
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

        # Keyed on the leaf name, so block names resolve regardless of nesting.
        self._modules_by_name = {
            name.rsplit(".", 1)[-1]: module
            for name, module in self.model.named_modules()
            if name
        }

        missing = [
            name
            for block in self.selected_layers
            for name in LAYER_BLOCKS[block]
            if name is not None and name not in self._modules_by_name
        ]
        if missing:
            raise ValueError(
                f"Model has no submodule(s) named {missing}. This ModelModule "
                "expects the named layers built by examples/03-cifar10-cnn/cnn.py."
            )

        self.last_layer = kwargs.get("last_layer", self.selected_layers[-1])

    def _to_tensor(self, x: Any) -> torch.Tensor:
        """Accept numpy or torch, single sample or batch of (3, 32, 32) images."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x, dtype=np.float32))

        x = x.to(device=self.device, dtype=torch.float32)

        return x.unsqueeze(0) if x.ndim == 3 else x

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

        The model ends in ``fc2`` with no softmax layer (raw logits, the
        usual torch convention), so softmax is applied here explicitly.
        """
        logits = self._forward_capture(x, ["fc2"])["fc2"]
        return torch.softmax(torch.as_tensor(logits), dim=-1).numpy()

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
        value must be a 2D array aligned with the input rows. ``fc2`` has no
        activation submodule to hook -- its "activation" is the softmax
        output, computed via ``_get_probabilities`` instead.
        """
        hooked = {
            block: LAYER_BLOCKS[block][1]
            for block in self.selected_layers
            if LAYER_BLOCKS[block][1] is not None
        }
        captured = self._forward_capture(x, list(hooked.values()))
        result = {block: captured[name] for block, name in hooked.items()}

        if "fc2" in self.selected_layers:
            result["fc2"] = self._get_probabilities(x)

        return result

    def _get_pre_activations(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Linear/conv output of each selected block, before the nonlinearity.

        For ``fc2`` these are the logits -- Mahalanobis fits Gaussians to
        these, and softmax outputs would be a poor fit for that.
        """
        wanted = {block: LAYER_BLOCKS[block][0] for block in self.selected_layers}
        captured = self._forward_capture(x, list(wanted.values()))

        return {block: captured[name] for block, name in wanted.items()}
