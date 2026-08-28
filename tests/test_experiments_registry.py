"""Regression test pinning the model/cage split derived in experiments/registry.py.

experiments/registry.py re-derives the cage-half split with the exact same
train_test_split() call the matching examples/ notebook used to create the
model/cage split in the first place -- it doesn't load a saved partition, it
recomputes it deterministically. That's only safe as long as the two stay
parameter-identical; if a notebook's split ever changes without registry.py
following, the cage split would silently start overlapping with whatever
trained the model, with no error raised anywhere.

This test pins the exact split verified correct when it was written (cage
sizes cross-checked against the notebooks' own printed partition summaries),
so any future drift in either split function fails loudly here instead.
"""
import hashlib

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")

from experiments.registry import EXAMPLES_DIR, _cifar10_split, _mnist_split


def _sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_mnist_model_cage_split_is_pinned():
    x_model, x_cage, y_model, y_cage = _mnist_split(EXAMPLES_DIR / "01-mnist")

    assert len(x_model) == 48000
    assert len(x_cage) == 12000
    assert _sha256(y_model) == "271599a81e262c05078170bb9e4b39101c546b0bd39639cd413b7f4bb3325d9d"
    assert _sha256(y_cage) == "e5351d19f9ad88b4cb197468b02a8a5836d62dcba8aa63992e9eeb90c3e0ea85"
    assert _sha256(np.asarray(x_cage)) == "961b6180e120135c0b607dca6ca3a1dfec3b3b6dd8ee15355ad060e03fd2436f"


def test_cifar10_model_cage_split_is_pinned():
    x_model, x_cage, y_model, y_cage = _cifar10_split(EXAMPLES_DIR / "03-cifar10-cnn")

    assert len(x_model) == 25000
    assert len(x_cage) == 25000
    assert _sha256(y_model) == "b8a6062d8bac703511bd56b8d8d87c5e889eadc556833af681fe6b68bf75f1fb"
    assert _sha256(y_cage) == "c1a2927f565cf7cf2bac8720303099ceb0e862309e0dbec366c088dd1058b52f"
    assert _sha256(np.asarray(x_cage)) == "c1236c3a6d569531010883481ef1060ed00d6435ef7e5e9469666b238ea4cab1"
