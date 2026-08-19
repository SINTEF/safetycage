"""DataModule for MNIST.

This is the integration contract on the data side. safetycage reads three
things from it: the ``data_train``/``data_val``/``data_test`` splits, the
``classes`` mapping (SPARDACUS iterates it), and ``num_classes``.

Each split is an ``(x, y)`` tuple of numpy arrays, where ``x`` is
``(N, 784)`` float32 and ``y`` is ``(N,)`` int64 — or ``(N, 10)`` one-hot
when ``use_onehot_encoder=True``.
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

# The usual MNIST statistics, computed over the training split.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MNISTDataModule(DataModule):
    """MNIST as flat 784-dimensional vectors, split train/val/test."""

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
            data_dir: Where torchvision downloads MNIST. (default: ``./data``)
            from_cache: Passed to the base class. torchvision skips the
                download whenever the files are already present regardless.
            batch_size: Stored on the base class; unused here, since the splits
                are handed over as whole arrays.
            val_split: Fraction of the 60k training images held out for
                validation. The 10k test images are MNIST's own test split and
                are never touched by this.
            use_onehot_encoder: Whether labels come back one-hot. Must match
                the ModelModule, since SPARDACUS compares the two directly.
            normalize: Standardize with the MNIST mean/std rather than leaving
                pixels in [0, 1].
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
        self.image_shape = (28, 28)

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
        return {digit: str(digit) for digit in range(10)}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def dataset_name(self) -> str:
        return "mnist"

    def setup(self) -> None:
        """Load MNIST's own train/test split, then carve validation out of train.

        Normalization uses fixed published constants rather than statistics
        fitted on the data, so there is no split-order leakage to worry about
        the way there would be with a fitted scaler.
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
        """Fetch one MNIST split as raw uint8 arrays.

        torchvision downloads on first call and reuses the files afterwards.
        No transform is passed: the tensor conversion and normalization belong
        in ``_transform``, so that this returns raw pixels.
        """
        dataset = datasets.MNIST(root=str(filepath), train=train, download=True)

        return dataset.data.numpy(), dataset.targets.numpy()

    def _split(self, x, y, split):
        """Shuffle, then take the last ``split`` fraction as the held-out set.

        Stratification is not needed: MNIST's classes are near-balanced and
        6000 held-out samples leave every digit well represented.
        """
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(x))

        cut = len(x) - int(round(len(x) * split))
        keep, held_out = order[:cut], order[cut:]

        return x[keep], y[keep], x[held_out], y[held_out]

    def _transform(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten to (N, 784), scale to [0, 1], optionally standardize."""
        x = np.asarray(x, dtype=np.float32).reshape(len(x), -1) / 255.0

        if self.normalize:
            x = (x - MNIST_MEAN) / MNIST_STD

        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if self.use_onehot_encoder:
            return x, np.eye(self.num_classes, dtype=np.float64)[y]

        return x, y

    def print_partition_summary(self) -> None:
        """Print the number of samples in each split."""
        for name in ["train", "val", "test"]:
            x, y = getattr(self, f"data_{name}")
            print(f"{name:5s} x={x.shape} y={y.shape}")
            
            
    def plot_samples(self, n_samples_per_class: int = 5, fig_scale:int = .5, cmap:str = "gray_r") -> None:
        """Plot sample images from the dataset.

        Args:
            n_samples_per_class  (int, optional): Number of samples per class to plot. Defaults to 5.
        """
        import matplotlib.pyplot as plt

        x_train, y_train = self.data_train[:2]
        
        # Handle one-hot encoded labels or (N, 1) shape
        if self.use_onehot_encoder:
            y_labels = np.argmax(y_train, axis=1)
        else:
            y_labels = y_train.flatten()

        fig, axes = plt.subplots(n_samples_per_class, len(self.classes), figsize=(len(self.classes) * fig_scale, n_samples_per_class * fig_scale), squeeze=True)
        
        for class_id, class_name in self.classes.items():
            # Find indices for this class
            idx = np.where(y_labels == class_id)[0]
            selected_idx = idx[:n_samples_per_class]
            
            for i in range(n_samples_per_class):
                ax = axes[i, class_id]
                if i < len(selected_idx):
                    img_idx = selected_idx[i]
                    img = x_train[img_idx].reshape(28,28)
                    ax.imshow(img, cmap=cmap)
                
                    if i == 0:
                        ax.set_title(class_name)
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
