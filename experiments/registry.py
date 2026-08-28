"""Registry of runnable (dataset, model) pipelines and safety cage methods.

A "pipeline" bundles a dataset with the model trained on it -- the two
aren't independent in this repo (a CNN expects (N, 3, 32, 32) CIFAR-10
images, not flat MNIST vectors), so ``PIPELINES`` is keyed by pipeline name
rather than crossing separate dataset x model registries. Each pipeline
reuses the dataset/model/checkpoint already built for the matching
``examples/`` notebook, so run that notebook once first if the checkpoint
doesn't exist yet.

Safety cage methods *are* independent of the pipeline (any ``SafetyCage``
subclass works with any ``ModelModule`` that exposes what it needs), so
``SAFETYCAGES`` is a flat registry crossed against every pipeline.

Each pipeline's ``_<name>_split`` function re-derives the model/cage split
with the exact ``train_test_split`` call the matching notebook used to
create it in the first place -- it doesn't load a persisted partition, it
recomputes it deterministically. That's only safe as long as the two stay
parameter-identical, so tests/test_experiments_registry.py pins the split's
exact output: if the notebook's split logic ever changes without this file
following, that test fails loudly instead of the cage split silently
starting to overlap with whatever trained the model.

To add a new pipeline: write a split function and a pipeline function with
the same shape as ``_mnist_split``/``_mnist_pipeline`` -- load your
DataModule's cage-half split, load your trained model into its
ModelModule -- and add it to ``PIPELINES``.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from safetycage.methods.doctor import DOCTOR
from safetycage.methods.mahalanobis import Mahalanobis
from safetycage.methods.msp import MSP
from safetycage.methods.random_baseline import RandomBaseline
from safetycage.methods.red import RED
from safetycage.methods.spardacus import SPARDACUS

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _mnist_split(example_dir):
    """Model/cage split for MNIST. Same call as examples/01-mnist/tutorial.ipynb.

    Returns:
        (x_model, x_cage, y_model, y_cage)
    """
    dataset = datasets.MNIST(root=str(example_dir / "data"), train=True, download=True)
    y = np.array(dataset.targets)
    return train_test_split(dataset.data, y, test_size=0.2, random_state=42, stratify=y)


def _train_mlp(x_model, y_model, device="cpu", epochs=10, batch_size=64, learning_rate=1e-3):
    """Train a fresh MLP from scratch on the model-half MNIST split.

    Mirrors examples/01-mnist/tutorial.ipynb's training cell exactly -- same
    model/val split, architecture, optimizer, hyperparameters, and
    preprocessing via MNISTDataModule -- so a from-scratch run is comparable
    to the notebook's own saved checkpoint. Not persisted to disk.
    """
    from mlp import MLP
    from mnist_datamodule import MNISTDataModule

    x_model_train, x_model_temp, y_model_train, y_model_temp = train_test_split(
        x_model, y_model, test_size=0.2, random_state=42, stratify=y_model
    )
    x_model_val, x_model_test, y_model_val, y_model_test = train_test_split(
        x_model_temp, y_model_temp, test_size=0.2, random_state=42, stratify=y_model_temp
    )
    data_module_model = MNISTDataModule(
        x_train=x_model_train, y_train=y_model_train,
        x_val=x_model_val, y_val=y_model_val,
        x_test=x_model_test, y_test=y_model_test,
        device=device,
    )

    def as_loader(split, shuffle=False):
        x, y = split
        return DataLoader(
            TensorDataset(torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)),
            batch_size=batch_size, shuffle=shuffle,
        )

    train_loader = as_loader(data_module_model.data_train, shuffle=True)
    val_loader = as_loader(data_module_model.data_val)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = sum((model(x).argmax(dim=1) == y).sum().item() for x, y in val_loader)
            seen = len(data_module_model.data_val[1])
        print(f"  [retrain mlp] epoch {epoch:2d}/{epochs}  val_acc {correct / seen:.4f}", flush=True)

    return model


def _mnist_pipeline(device="cpu", retrain=False):
    """MLP on MNIST. Mirrors examples/01-mnist/tutorial.ipynb.

    Args:
        device: Torch device to train/load the model onto.
        retrain (bool): If True, train a fresh MLP from scratch on the
            model-half split instead of loading
            examples/01-mnist/mlp_mnist.pt -- useful for checking a cage
            method against a newly initialized model rather than the one
            checkpoint already on disk. Not persisted to disk, and
            retrained again on every call.
    """
    example_dir = EXAMPLES_DIR / "01-mnist"
    sys.path.insert(0, str(example_dir))
    from mlp import MLP
    from mlp_modelmodule import MLPModelModule
    from mnist_datamodule import MNISTDataModule

    x_model, x_cage, y_model, y_cage = _mnist_split(example_dir)

    if retrain:
        model = _train_mlp(x_model, y_model, device=device)
    else:
        checkpoint = example_dir / "mlp_mnist.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{checkpoint} not found -- run examples/01-mnist/tutorial.ipynb once "
                "to download the data and train + save the model, or pass retrain=True."
            )
        model = MLP()
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    model_module = MLPModelModule(
        selected_layers="dense_1", use_onehot_encoder=False, model=model, device=device
    )

    x_cage_train, x_cage_temp, y_cage_train, y_cage_temp = train_test_split(
        x_cage, y_cage, test_size=0.2, random_state=42
    )
    x_cage_val, x_cage_test, y_cage_val, y_cage_test = train_test_split(
        x_cage_temp, y_cage_temp, test_size=0.2, random_state=42
    )
    data_module = MNISTDataModule(
        x_train=x_cage_train, y_train=y_cage_train,
        x_val=x_cage_val, y_val=y_cage_val,
        x_test=x_cage_test, y_test=y_cage_test,
        device=device,
    )
    return model_module, data_module


def _cifar10_split(example_dir):
    """Model/cage split for CIFAR-10. Same call as examples/03-cifar10-cnn/tutorial.ipynb.

    Returns:
        (x_model, x_cage, y_model, y_cage)
    """
    dataset = datasets.CIFAR10(root=str(example_dir / "data"), train=True, download=True)
    y = np.array(dataset.targets)
    return train_test_split(dataset.data, y, test_size=0.5, random_state=42, stratify=y)


def _train_cnn(x_model, y_model, device="cpu", epochs=10, batch_size=64, learning_rate=1e-3):
    """Train a fresh CNN from scratch on the model-half CIFAR-10 split.

    Mirrors examples/03-cifar10-cnn/tutorial.ipynb's training cell exactly --
    same model/val split, architecture, optimizer, hyperparameters, and
    preprocessing via CIFAR10DataModule -- so a from-scratch run is
    comparable to the notebook's own saved checkpoint. Not persisted to disk.
    """
    from cnn import CNN
    from cifar10_datamodule import CIFAR10DataModule

    x_model_train, x_model_temp, y_model_train, y_model_temp = train_test_split(
        x_model, y_model, test_size=0.2, random_state=42, stratify=y_model
    )
    x_model_val, x_model_test, y_model_val, y_model_test = train_test_split(
        x_model_temp, y_model_temp, test_size=0.2, random_state=42, stratify=y_model_temp
    )
    data_module_model = CIFAR10DataModule(
        x_train=x_model_train, y_train=y_model_train,
        x_val=x_model_val, y_val=y_model_val,
        x_test=x_model_test, y_test=y_model_test,
        device=device,
    )

    def as_loader(split, shuffle=False):
        x, y = split
        return DataLoader(
            TensorDataset(torch.from_numpy(x).float().to(device), torch.from_numpy(y).long().to(device)),
            batch_size=batch_size, shuffle=shuffle,
        )

    train_loader = as_loader(data_module_model.data_train, shuffle=True)
    val_loader = as_loader(data_module_model.data_val)

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = sum((model(x).argmax(dim=1) == y).sum().item() for x, y in val_loader)
            seen = len(data_module_model.data_val[1])
        print(f"  [retrain cnn] epoch {epoch:2d}/{epochs}  val_acc {correct / seen:.4f}", flush=True)

    return model


def _cifar10_pipeline(device="cpu", retrain=False):
    """CNN on CIFAR-10. Mirrors examples/03-cifar10-cnn/tutorial.ipynb.

    Args:
        device: Torch device to train/load the model onto.
        retrain (bool): If True, train a fresh CNN from scratch on the
            model-half split instead of loading
            examples/03-cifar10-cnn/cnn_cifar10.pt -- useful for checking a
            cage method against a newly initialized model rather than the
            one checkpoint already on disk. Not persisted to disk, and
            retrained again on every call.
    """
    example_dir = EXAMPLES_DIR / "03-cifar10-cnn"
    sys.path.insert(0, str(example_dir))
    from cnn import CNN
    from cnn_modelmodule import CNNModelModule
    from cifar10_datamodule import CIFAR10DataModule

    x_model, x_cage, y_model, y_cage = _cifar10_split(example_dir)

    if retrain:
        model = _train_cnn(x_model, y_model, device=device)
    else:
        checkpoint = example_dir / "cnn_cifar10.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{checkpoint} not found -- run examples/03-cifar10-cnn/tutorial.ipynb once "
                "to download the data and train + save the model, or pass retrain=True."
            )
        model = CNN()
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)

    model_module = CNNModelModule(
        selected_layers="fc2", use_onehot_encoder=False, model=model, device=device
    )

    x_cage_train, x_cage_temp, y_cage_train, y_cage_temp = train_test_split(
        x_cage, y_cage, test_size=0.2, random_state=42
    )
    x_cage_val, x_cage_test, y_cage_val, y_cage_test = train_test_split(
        x_cage_temp, y_cage_temp, test_size=0.2, random_state=42
    )
    data_module = CIFAR10DataModule(
        x_train=x_cage_train, y_train=y_cage_train,
        x_val=x_cage_val, y_val=y_cage_val,
        x_test=x_cage_test, y_test=y_cage_test,
        device=device,
    )
    return model_module, data_module


# name -> factory(device, **kwargs) -> (model_module, data_module)
PIPELINES = {
    "mnist": _mnist_pipeline,
    "cifar10": _cifar10_pipeline,
}

# name -> (SafetyCage subclass, default constructor kwargs)
SAFETYCAGES = {
    "msp": (MSP, {}),
    "doctor": (DOCTOR, {"method": "max"}),
    "red": (RED, {"training_iterations": 20, "batch_size": 32}),
    "spardacus": (SPARDACUS, {"s_statistic_source": "correctly"}),
    "mahalanobis": (
        Mahalanobis,
        {"empirical": False, "test_type_within_layer": "chi2", "test_type_between_layers": None},
    ),
    "random_baseline": (RandomBaseline, {}),
}
