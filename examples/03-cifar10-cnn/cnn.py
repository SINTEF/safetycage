import torch
import torch.nn as nn


class CNN(nn.Module):
    """Two-conv-block CNN over 32x32x3 CIFAR-10 images, plus two linear layers.

    Layers are named so forward hooks can pull activations out by name, same
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits -- no final softmax, matching the usual torch convention."""
        x = self.pool1(self.relu1(self.conv1_bn(self.conv1(x))))
        x = self.pool2(self.relu2(self.conv2_bn(self.conv2(x))))
        x = x.flatten(start_dim=1)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x
