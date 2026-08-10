import safetycage
import numpy as np

# pytest convention that all files in ./tests begin with "test_"

def test_add_one_basic():
    assert 1+1 == 2

def test_check_package_version():
    """The installed version must match the version declared in pyproject.toml."""
    import tomllib
    from importlib.metadata import version
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]

    assert version("safetycage") == declared

def main():
    test_add_one_basic()
    test_check_package_version()
    print("Hello from safetycage-tutorials!")

    leq = True
    compare = np.less_equal if leq else np.greater_equal

    N = 10001
    y_true = np.random.randint(0, 2, size=N)

    statistics = np.random.rand(N)
    M = 1000
    nan_idx = np.random.choice(N, M, replace=False)
    statistics[nan_idx] = np.NaN
    print("statistics:", statistics)
    alpha = 0.5
    flags = compare(statistics, alpha)
    print("flags:", flags)
    flags = np.where(np.isnan(statistics), np.nan, flags)
    print("flags with NaNs:", flags)

    flag_nan_idx = np.flatnonzero(np.isnan(statistics))
    print("flag_nan_idx:", flag_nan_idx)
    y_true_clean = np.delete(y_true, flag_nan_idx)
    flag_clean = np.delete(flags, flag_nan_idx)
    flag_clean = flag_clean.astype(bool)
    print("y_true_clean:", y_true_clean)
    print("flag_clean:", flag_clean)

    tps = np.sum(flag_clean & y_true_clean)
    fps = np.sum(flag_clean & (1 - y_true_clean))

    total_pos = y_true_clean.sum()
    total_neg = y_true_clean.size - total_pos

    fns = total_pos - tps
    tns = total_neg - fps

    print(f"TP: {tps}, FP: {fps}, FN: {fns}, TN: {tns}")
    print(f"Accuracy: {(tps + tns) / y_true_clean.size:.2f}")


if __name__ == "__main__":
    main()


# THIS IS NOT CONFIGURED PROPERLY RIGHT NOW


### create simple tests here, install the python package in the venv to quickly test,
# then use safetycage_tutorials to see a more end/user perspective
