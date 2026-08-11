"""Unit tests for ArrayDataModule.

Run with: uv run --with pytest --all-extras pytest tests/test_array_datamodule.py -v
"""
import os
import tempfile

import numpy as np
import pytest

from safetycage.datamodules.array_datamodule import ArrayDataModule


def _synthetic_splits():
    rng = np.random.RandomState(0)
    x_train = rng.randn(20, 3)
    y_train = rng.randint(0, 3, size=20)
    x_val = rng.randn(10, 3)
    y_val = rng.randint(0, 3, size=10)
    x_test = rng.randn(10, 3)
    y_test = rng.randint(0, 3, size=10)
    return x_train, y_train, x_val, y_val, x_test, y_test


def test_wraps_numpy_arrays():
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()
    module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)

    np.testing.assert_array_equal(module.data_train[0], x_train)
    np.testing.assert_array_equal(module.data_train[1], y_train)
    np.testing.assert_array_equal(module.data_val[0], x_val)
    np.testing.assert_array_equal(module.data_val[1], y_val)
    np.testing.assert_array_equal(module.data_test[0], x_test)
    np.testing.assert_array_equal(module.data_test[1], y_test)


def test_classes_and_num_classes_from_y_train():
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()
    module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)

    expected = {c: str(c) for c in np.unique(y_train)}
    assert module.classes == expected
    assert module.num_classes == len(expected)


def test_dataset_name_is_custom():
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()
    module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)
    assert module.dataset_name == "custom"


def test_accepts_pandas_dataframe_and_series():
    pd = pytest.importorskip("pandas")
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()

    module = ArrayDataModule(
        pd.DataFrame(x_train),
        pd.Series(y_train),
        pd.DataFrame(x_val),
        pd.Series(y_val),
        pd.DataFrame(x_test),
        pd.Series(y_test),
    )

    for split in (module.data_train, module.data_val, module.data_test):
        assert isinstance(split[0], np.ndarray)
        assert isinstance(split[1], np.ndarray)

    np.testing.assert_array_equal(module.data_train[0], x_train)
    np.testing.assert_array_equal(module.data_train[1], y_train)


def test_to_joblib_from_joblib_roundtrip():
    x_train, y_train, x_val, y_val, x_test, y_test = _synthetic_splits()
    module = ArrayDataModule(x_train, y_train, x_val, y_val, x_test, y_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data_module.joblib")
        module.to_joblib(path)
        loaded = module.from_joblib(path)

    np.testing.assert_array_equal(loaded.data_train[0], x_train)
    np.testing.assert_array_equal(loaded.data_test[1], y_test)
    assert loaded.classes == module.classes
