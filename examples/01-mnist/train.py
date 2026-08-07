"""Train the MLP on MNIST.

Ordinary PyTorch, nothing safetycage-specific — which is why it is kept out
of the notebook.

One wrinkle: ``MLP`` ends in ``Softmax``, so it outputs probabilities rather
than logits. ``nn.CrossEntropyLoss`` applies log-softmax internally and would
softmax twice, so the loss here is ``NLLLoss`` over the log of the model
output instead.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mlp import MLP
from mnist_datamodule import MNISTDataModule

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
CHECKPOINT = Path(__file__).resolve().parent / "mlp_mnist.pt"


def as_loader(split, batch_size=BATCH_SIZE, shuffle=False):
    """Wrap an (x, y) numpy split as a DataLoader."""
    x, y = split

    return DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long()),
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Return (mean loss, accuracy) over a loader."""
    model.eval()
    loss_fn = nn.NLLLoss()

    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        probabilities = model(x)

        total_loss += loss_fn(torch.log(probabilities.clamp_min(1e-12)), y).item() * len(y)
        correct += (probabilities.argmax(dim=1) == y).sum().item()
        seen += len(y)

    return total_loss / seen, correct / seen


def train(
    model=None,
    data_module=None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    device="cpu",
    seed=42,
):
    """Fit the MLP and return (model, data_module)."""
    torch.manual_seed(seed)
    device = torch.device(device)

    data_module = data_module or MNISTDataModule()
    model = (model or MLP()).to(device)

    train_loader = as_loader(data_module.data_train, batch_size, shuffle=True)
    val_loader = as_loader(data_module.data_val, batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Applied to log() of the model's softmax output; see the module docstring.
    loss_fn = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = loss_fn(torch.log(model(x).clamp_min(1e-12)), y)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}/{epochs}  val_loss {val_loss:.4f}  val_acc {val_accuracy:.4f}")

    return model, data_module


if __name__ == "__main__":
    model, data_module = train()

    test_loss, test_accuracy = evaluate(model, as_loader(data_module.data_test))
    print(f"\ntest_loss {test_loss:.4f}  test_acc {test_accuracy:.4f}")

    torch.save(model.state_dict(), CHECKPOINT)
    print(f"saved {CHECKPOINT}")
