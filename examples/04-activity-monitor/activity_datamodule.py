"""DataModule for the UCI Daily and Sports Activities dataset.

This is the integration contract on the data side. safetycage reads three
things from it: the ``data_train``/``data_val``/``data_test`` splits, the
``classes`` mapping (SPARDACUS iterates it), and ``num_classes``.

Each split is an ``(x, y)`` tuple of numpy arrays, where ``x`` is
``(N, 125, 45)`` float32 -- 5-second windows at 25 Hz from 5 body-worn
Xsens units (torso, right/left arm, right/left leg), each contributing a
3-axis accelerometer + gyroscope + magnetometer (9 channels x 5 units = 45)
-- and ``y`` is ``(N,)`` int64, or ``(N, 19)`` one-hot when
``use_onehot_encoder=True``.

Unlike MNIST/CIFAR-10, this dataset ships with no official train/test
split and no published per-channel normalization constants, so both are
handled differently here: ``setup()`` carves out test and validation
fractions itself (stratified by activity), and no normalization is
applied at all -- the accelerometer/gyroscope/magnetometer channels sit on
very different scales, and a model consuming this data is expected to
normalize internally (e.g. a BatchNorm layer, as the CIFAR-10 CNN example
already does) rather than have this DataModule fit and bake in scaling
statistics that would leak across whatever split it's asked to produce.

Source: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities
(CC BY 4.0). Primary reference: Altun, Barshan & Tuncel (2010), "Comparative
study on classifying human activities with miniature inertial and magnetic
sensors", Pattern Recognition, 43(10), 3605-3620.

Physical sensor placement -- chest, wrists, sides of the knees, not the hip
and shoulder anchors ``plot_pose_animation``'s schematic skeleton assumes --
is documented in the follow-up study reusing the same sensor setup: Barshan
& Yuksek (2014), "Recognizing Daily and Sports Activities in Two Open Source
Machine Learning Environments Using Body-Worn Sensor Units", The Computer
Journal, 57(11), 1649-1667.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from safetycage.datamodule import DataModule

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

SEGMENT_SHAPE = (125, 45)
SAMPLE_RATE_HZ = 25

# Column layout: 5 units x 9 channels (acc xyz, gyro xyz, mag xyz) each, in
# this order -- https://archive.ics.uci.edu/dataset/256. Physical placement
# per Barshan & Yuksek (2014): torso = chest, arm units = wrists (not upper
# arm), leg units = sides of the knees (not thigh or ankle).
SENSOR_UNITS = ["torso", "right-arm", "left-arm", "right-leg", "left-leg"]
CHANNELS_PER_UNIT = 9

# a01-a19, in order -- https://archive.ics.uci.edu/dataset/256
ACTIVITIES = [
    "sitting",
    "standing",
    "lying-on-back",
    "lying-on-right-side",
    "ascending-stairs",
    "descending-stairs",
    "standing-in-elevator-still",
    "moving-in-elevator",
    "walking-in-parking-lot",
    "walking-on-treadmill-flat",
    "walking-on-treadmill-inclined",
    "running-on-treadmill",
    "exercising-on-stepper",
    "exercising-on-cross-trainer",
    "cycling-horizontal",
    "cycling-vertical",
    "rowing",
    "jumping",
    "playing-basketball",
]


def load_raw_activity_data(data_dir: Path, from_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Read every ``aXX/pY/sZZ.txt`` segment under ``data_dir`` into (x, y).

    Parsing all 9,120 segment files from disk takes tens of seconds, unlike
    the near-instant torchvision cache checks the MNIST/CIFAR-10 examples
    rely on, so the result is cached alongside the raw files as a joblib
    dump and reused on subsequent calls whenever ``from_cache`` is True.

    Returns:
        x: ``(N, 125, 45)`` float32 raw sensor windows.
        y: ``(N,)`` int64 activity indices (0-18, matching ``ACTIVITIES``).
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / "raw_activity_data.joblib"
    if from_cache and cache_path.exists():
        return joblib.load(cache_path)

    segment_files = sorted(data_dir.glob("a*/p*/s*.txt"))
    if not segment_files:
        raise FileNotFoundError(
            f"No aXX/pY/sZZ.txt segment files found under {data_dir}. Download the "
            "dataset from https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities "
            "and extract it there."
        )

    x = np.stack([np.loadtxt(f, delimiter=",") for f in segment_files]).astype(np.float32)
    y = np.array([int(f.parent.parent.name[1:]) - 1 for f in segment_files], dtype=np.int64)

    joblib.dump((x, y), cache_path)
    return x, y


class ActivityDataModule(DataModule):
    """Wraps the Daily and Sports Activities dataset as (N, 125, 45) sensor windows."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        from_cache: bool = True,
        batch_size: int = 32,
        val_split: float = 0.1,
        test_split: float = 0.2,
        use_onehot_encoder: bool = False,
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
            data_dir: Where the extracted ``aXX/pY/sZZ.txt`` tree lives. (default: ``./data``)
            from_cache: Reuse the joblib-cached parse of the raw segment files
                instead of re-reading all 9,120 of them from disk. (default: True)
            batch_size: Stored on the base class; unused here, since the splits
                are handed over as whole arrays.
            val_split: Fraction of the non-test pool held out for validation.
                Ignored when x_train etc. are given.
            test_split: Fraction of the full dataset held out for testing --
                this dataset has no official test split, unlike MNIST/CIFAR-10.
                Ignored when x_train etc. are given.
            use_onehot_encoder: Whether labels come back one-hot. Must match
                the ModelModule, since SPARDACUS compares the two directly.
            random_state: Seed for the stratified train/val/test split.
            device: Stored on the base class; the arrays stay on the CPU and
                the ModelModule moves them.
            x_train, y_train, x_val, y_val, x_test, y_test: Already-loaded
                and split data. If any one of the six is given, all six are
                required, and the raw segment files are not read at all.
                Each x_* must be raw ``(N, 125, 45)`` sensor windows -- the
                same format ``load_raw_activity_data`` returns -- not already
                transformed.
        """
        super().__init__(data_dir or DEFAULT_DATA_DIR, from_cache, batch_size, device)

        self.val_split = val_split
        self.test_split = test_split
        self.use_onehot_encoder = use_onehot_encoder
        self.random_state = random_state
        self.segment_shape = SEGMENT_SHAPE

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
        return dict(enumerate(ACTIVITIES))

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def dataset_name(self) -> str:
        return "daily-sports-activities"

    def setup(self) -> None:
        """Load every segment, then carve out test and validation splits.

        Skipped entirely if train/val/test arrays were passed to __init__ --
        those are used directly instead, with no file reads or re-split.
        """
        if self._provided_splits is not None:
            x_train, y_train, x_val, y_val, x_test, y_test = self._provided_splits
        else:
            x, y = self._load_data(self.data_dir)
            x_train, y_train, x_test, y_test = self._split(x, y, self.test_split)
            x_train, y_train, x_val, y_val = self._split(x_train, y_train, self.val_split)

        self.data_train = self._transform(x_train, y_train)
        self.data_val = self._transform(x_val, y_val)
        self.data_test = self._transform(x_test, y_test)

    def _load_data(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Fetch the full (unsplit) dataset as raw float32 windows."""
        return load_raw_activity_data(filepath, from_cache=self.from_cache)

    def _split(self, x, y, split):
        """Stratified split off a ``split`` fraction of (x, y), by activity."""
        x_keep, x_held_out, y_keep, y_held_out = train_test_split(
            x, y, test_size=split, random_state=self.random_state, stratify=y
        )
        return x_keep, y_keep, x_held_out, y_held_out

    def _transform(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Cast dtypes; no normalization (see module docstring for why)."""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if self.use_onehot_encoder:
            return x, np.eye(self.num_classes, dtype=np.float64)[y]

        return x, y

    def print_partition_summary(self) -> None:
        """Print the number of samples in each split."""
        for name in ["train", "val", "test"]:
            x, y = getattr(self, f"data_{name}")
            print(f"{name:5s} x={x.shape} y={y.shape}")

    def plot_samples(self, channel: int = 0, n_samples_per_class: int = 3) -> None:
        """Plot one sensor channel's trace over time, a few segments per activity.

        Args:
            channel: Which of the 45 sensor columns to plot (default: 0, the
                torso accelerometer's x-axis).
            n_samples_per_class: Number of segments to overlay per activity.
        """
        import matplotlib.pyplot as plt

        x_train, y_train = self.data_train[:2]
        y_labels = np.argmax(y_train, axis=1) if self.use_onehot_encoder else y_train.flatten()

        fig, axes = plt.subplots(
            self.num_classes, 1, figsize=(6, self.num_classes * 0.8), sharex=True
        )

        for class_id, class_name in self.classes.items():
            idx = np.where(y_labels == class_id)[0][:n_samples_per_class]
            ax = axes[class_id]

            for i in idx:
                ax.plot(x_train[i, :, channel], linewidth=0.8)

            ax.set_ylabel(class_name, fontsize=7, rotation=0, ha="right", va="center")
            ax.set_yticks([])

        axes[-1].set_xlabel("timestep")
        plt.tight_layout()
        plt.show()



    def plot_window_timeseries(
        self, index: int = 0, split: str = "train", title: Optional[str] = None
    ) -> None:
        """Plot one window's raw sensor channels, faceted by unit and sensor type.

        A grid of subplots: one row per sensor unit (torso, right/left
        arm, right/left leg), one column per sensor type (accelerometer,
        gyroscope, magnetometer), each showing that unit-sensor's x/y/z
        channels over the window's 125 timesteps. No orientation
        estimation, no animation: just the numbers as measured.

        Args:
            index: Which segment (window) to plot.
            split: Which split to pull it from ("train", "val", or "test").
            title: Figure title. Defaults to the window's activity label.
        """
        import matplotlib.pyplot as plt

        x, y = getattr(self, f"data_{split}")
        window = x[index]  # (125, 45)
        label = title or self.classes[int(np.argmax(y[index]) if self.use_onehot_encoder else y[index])]

        sensor_types = ["accelerometer", "gyroscope", "magnetometer"]
        axis_names = ["x", "y", "z"]
        axis_colors = ["tab:red", "tab:green", "tab:blue"]
        timesteps = np.arange(window.shape[0]) / SAMPLE_RATE_HZ

        fig, axes = plt.subplots(
            len(SENSOR_UNITS), len(sensor_types), figsize=(11, 12), sharex=True
        )
        for row, unit in enumerate(SENSOR_UNITS):
            unit_data = window[:, row * CHANNELS_PER_UNIT : (row + 1) * CHANNELS_PER_UNIT]
            for col, sensor_type in enumerate(sensor_types):
                ax = axes[row, col]
                sensor_data = unit_data[:, col * 3 : (col + 1) * 3]
                for axis_index, axis_name in enumerate(axis_names):
                    ax.plot(
                        timesteps,
                        sensor_data[:, axis_index],
                        color=axis_colors[axis_index],
                        linewidth=1,
                        label=axis_name,
                    )
                if row == 0:
                    ax.set_title(sensor_type, fontsize=9)
            axes[row, 0].set_ylabel(unit, fontsize=8, rotation=0, ha="right", va="center")

        axes[0, 0].legend(fontsize=6, loc="upper right")
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        fig.suptitle(label)
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
