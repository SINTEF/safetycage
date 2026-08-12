"""ArrayDataModule: wraps train/val/test splits already loaded in memory.

Accepts numpy arrays or pandas DataFrame/Series for any split. Pandas
input is normalized to numpy via a duck-typed to_numpy() check, so this
module never imports pandas itself.
"""
from typing import Any, Dict

import joblib
import numpy as np

from safetycage.datamodule import DataModule


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


class ArrayDataModule(DataModule):
    """Wraps train/val/test splits that are already loaded in memory.

    ``train`` fits the cage itself (used by RED/SPARDACUS/Mahalanobis;
    MSP/DOCTOR need no fitting). ``val`` estimates the optimal threshold.
    ``test`` evaluates that threshold on held-out data.
    """

    def __init__(
        self,
        x_train: Any,
        y_train: Any,
        x_val: Any,
        y_val: Any,
        x_test: Any,
        y_test: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(data_dir=".", **kwargs)

        x_train_arr, y_train_arr = _to_numpy(x_train), _to_numpy(y_train)
        x_val_arr, y_val_arr = _to_numpy(x_val), _to_numpy(y_val)
        x_test_arr, y_test_arr = _to_numpy(x_test), _to_numpy(y_test)

        for split_name, (x, y) in (
            ("train", (x_train_arr, y_train_arr)),
            ("val", (x_val_arr, y_val_arr)),
            ("test", (x_test_arr, y_test_arr)),
        ):
            if len(x) != len(y):
                raise ValueError(
                    f"x_{split_name} has {len(x)} samples but y_{split_name} has {len(y)}"
                )

        self.data_train = (x_train_arr, y_train_arr)
        self.data_val = (x_val_arr, y_val_arr)
        self.data_test = (x_test_arr, y_test_arr)
        self._classes = {c: str(c) for c in np.unique(y_train_arr)}

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def classes(self) -> Dict[Any, str]:
        return self._classes

    @property
    def dataset_name(self) -> str:
        return "custom"

    def setup(self) -> None:
        pass  # Splits are already provided in __init__.

    def _load_data(self, filepath: str) -> None:
        pass  # Not needed: data arrives pre-loaded.

    def _transform(self, x, y):
        return x, y

    def _split(self, x, y, split):
        pass  # Not needed: splits are already provided.

    def to_joblib(self, path: str) -> None:
        joblib.dump(self, path)

    def from_joblib(self, path: str) -> Any:
        return joblib.load(path)
