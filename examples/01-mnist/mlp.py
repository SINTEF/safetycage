from collections import OrderedDict

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Three-layer MLP over flattened 28x28 images.

    Layers are named so forward hooks can pull activations out by name.
    """

    def __init__(
        self,
        input_features: int = 28 * 28 * 1,
        num_classes: int = 10,
    ) -> None:
        """
        Args:
            input_features: Flattened input size. Keras' ``input_shape`` only
                declares this; PyTorch takes it as the first ``Linear`` argument.
            num_classes: Output size.
            softmax: Whether to append the final ``Softmax``. Keep True to match
                Keras; set False when training with ``nn.CrossEntropyLoss``,
                which applies log-softmax internally and would softmax twice.
        """
        super().__init__()
        layers = OrderedDict(
            [
                ("dense_1", nn.Linear(input_features, 256)),
                ("relu_1", nn.ReLU()),
                ("dense_2", nn.Linear(256, 128)),
                ("relu_2", nn.ReLU()),
                ("dense_3", nn.Linear(128, num_classes)),
                ("softmax", nn.Softmax(dim=1)),
            ]
        )

        self.mlp = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flattens anything past the batch dimension, so (N, 1, 28, 28) works."""
        return self.mlp(x.flatten(start_dim=1))
