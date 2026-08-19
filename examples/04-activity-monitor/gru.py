import torch
import torch.nn as nn


class GRUActivityClassifier(nn.Module):
    """GRU over the 125-timestep, 45-channel sensor window, plus a linear head.

    The GRU itself is not a ``selected_layers`` target for
    ``TorchModelModule``: its forward returns a ``(output, h_n)`` tuple, not a
    single tensor, so SPARDACUS instead hooks ``relu`` in the head, which is a
    plain 2D tensor.
    """

    def __init__(
        self,
        num_channels: int = 45,
        hidden_size: int = 64,
        num_classes: int = 19,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(num_channels, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits -- no final softmax, matching the usual torch convention.

        Args:
            x: ``(N, 125, 45)`` sensor windows.
        """
        _, h_n = self.gru(x)
        h = h_n[-1]  # (N, hidden_size), the final layer's last hidden state
        h = self.bn(h)
        h = self.relu(self.fc1(h))
        return self.fc2(h)
