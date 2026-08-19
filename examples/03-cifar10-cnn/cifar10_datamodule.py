"""DataModule for CIFAR-10.

This is the integration contract on the data side. safetycage reads three
things from it: the ``data_train``/``data_val``/``data_test`` splits, the
``classes`` mapping (SPARDACUS iterates it), and ``num_classes``.

Each split is an ``(x, y)`` tuple of numpy arrays, where ``x`` is
``(N, 3, 32, 32)`` float32 and ``y`` is ``(N,)`` int64 -- or ``(N, 10)``
one-hot when ``use_onehot_encoder=True``.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from torchvision import datasets

from safetycage.datamodule import DataModule

# DataModule.__init__ calls Path(data_dir) with no None check, so a None
# default would raise TypeError. Supply a concrete one.
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Standard CIFAR-10 per-channel statistics, computed over the training split.
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


class CIFAR10DataModule(DataModule):
    """CIFAR-10 as (N, 3, 32, 32) float32 tensors, split train/val/test."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        from_cache: bool = True,
        batch_size: int = 32,
        val_split: float = 0.1,
        use_onehot_encoder: bool = False,
        normalize: bool = True,
        random_state: int = 42,
        device: str = "cpu",
        x_train: Optional[Any] = None,
        y_train: Optional[Any] = None,
        x_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        x_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
    ) -> None:
        """
        Args:
            data_dir: Where torchvision downloads CIFAR-10. (default: ``./data``)
            from_cache: Passed to the base class. torchvision skips the
                download whenever the files are already present regardless.
            batch_size: Stored on the base class; unused here, since the splits
                are handed over as whole arrays.
            val_split: Fraction of the 50k training images held out for
                validation. The 10k test images are CIFAR-10's own test split
                and are never touched by this. Ignored when x_train etc. are
                given -- there is nothing left to split.
            use_onehot_encoder: Whether labels come back one-hot. Must match
                the ModelModule, since SPARDACUS compares the two directly.
            normalize: Standardize with the CIFAR-10 per-channel mean/std
                rather than leaving pixels in [0, 1].
            random_state: Seed for the train/val shuffle.
            device: Stored on the base class; the arrays stay on the CPU and
                the ModelModule moves them.
            x_train, y_train, x_val, y_val, x_test, y_test: Already-loaded and split data
        """
        super().__init__(data_dir or DEFAULT_DATA_DIR, from_cache, batch_size, device)

        self.val_split = val_split
        self.use_onehot_encoder = use_onehot_encoder
        self.normalize = normalize
        self.random_state = random_state
        self.image_shape = (3, 32, 32)

        provided = (x_train, y_train, x_val, y_val, x_test, y_test)
        if any(p is not None for p in provided) and not all(p is not None for p in provided):
            raise ValueError(
                "Pass all six of x_train/y_train/x_val/y_val/x_test/y_test, or none of them."
            )
        self._provided_splits = provided if provided[0] is not None else None

        self.setup()

    @property
    def classes(self) -> Dict[int, str]:
        """SPARDACUS iterates this as ``.items()`` and indexes it by label."""
        return dict(enumerate(CIFAR10_CLASSES))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def dataset_name(self) -> str:
        return "cifar10"

    def setup(self) -> None:
        """Load CIFAR-10's own train/test split, then carve validation out of train.

        Skipped entirely if train/val/test arrays were passed to __init__ --
        those are used directly instead, with no download or re-split.
        """
        if self._provided_splits is not None:
            x_train, y_train, x_val, y_val, x_test, y_test = self._provided_splits
        else:
            x_train_val, y_train_val = self._load_data(self.data_dir, train=True)
            x_test, y_test = self._load_data(self.data_dir, train=False)

            x_train, y_train, x_val, y_val = self._split(
                x_train_val, y_train_val, self.val_split
            )

        self.data_train = self._transform(x_train, y_train)
        self.data_val = self._transform(x_val, y_val)
        self.data_test = self._transform(x_test, y_test)

    def _load_data(self, filepath: str, train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch one CIFAR-10 split as raw uint8 arrays, channel-last (N, 32, 32, 3).

        torchvision downloads on first call and reuses the files afterwards.
        No transform is passed: the tensor conversion and normalization belong
        in ``_transform``, so that this returns raw pixels.
        """
        dataset = datasets.CIFAR10(root=str(filepath), train=train, download=True)

        return np.array(dataset.data), np.array(dataset.targets)

    def _split(self, x, y, split):
        """Shuffle, then take the last ``split`` fraction as the held-out set.

        Stratification is not needed: CIFAR-10's classes are perfectly
        balanced and 5000 held-out samples leave every class well represented.
        """
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(x))

        cut = len(x) - int(round(len(x) * split))
        keep, held_out = order[:cut], order[cut:]

        return x[keep], y[keep], x[held_out], y[held_out]

    def _transform(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Channel-last uint8 (N, 32, 32, 3) -> channel-first float32 (N, 3, 32, 32)."""
        x = np.asarray(x, dtype=np.float32).transpose(0, 3, 1, 2) / 255.0

        if self.normalize:
            x = (x - CIFAR10_MEAN.reshape(1, 3, 1, 1)) / CIFAR10_STD.reshape(1, 3, 1, 1)

        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if self.use_onehot_encoder:
            return x, np.eye(self.num_classes, dtype=np.float64)[y]

        return x, y

    def print_partition_summary(self) -> None:
        """Print the number of samples in each split."""
        for name in ["train", "val", "test"]:
            x, y = getattr(self, f"data_{name}")
            print(f"{name:5s} x={x.shape} y={y.shape}")

    def plot_samples(self, n_samples_per_class: int = 5, fig_scale: float = 1.0) -> None:
        """Plot sample images from the training split.

        Args:
            n_samples_per_class: Number of samples per class to plot. Defaults to 5.
        """
        import matplotlib.pyplot as plt

        x_train, y_train = self.data_train

        if self.use_onehot_encoder:
            y_labels = np.argmax(y_train, axis=1)
        else:
            y_labels = y_train.flatten()

        fig, axes = plt.subplots(
            n_samples_per_class,
            self.num_classes,
            figsize=(self.num_classes * fig_scale, n_samples_per_class * fig_scale),
            squeeze=True,
        )

        for class_id, class_name in self.classes.items():
            idx = np.where(y_labels == class_id)[0]
            selected_idx = idx[:n_samples_per_class]

            for i in range(n_samples_per_class):
                ax = axes[i, class_id]
                if i < len(selected_idx):
                    img = x_train[selected_idx[i]]
                    if self.normalize:
                        img = img * CIFAR10_STD.reshape(3, 1, 1) + CIFAR10_MEAN.reshape(3, 1, 1)
                    img = np.clip(img.transpose(1, 2, 0), 0, 1)
                    ax.imshow(img)

                    if i == 0:
                        ax.set_title(class_name, fontsize=8)
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    def _default_joblib_path(self) -> Path:
        return (self.data_dir / f"{self.dataset_name}_data_module").with_suffix(".joblib")

    def to_joblib(self, path: Optional[str] = None) -> None:
        path = Path(path or self._default_joblib_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def from_joblib(self, path: Optional[str] = None) -> Any:
        return joblib.load(Path(path or self._default_joblib_path()))
