"""Unit tests for SPARDACUS.

Run with: uv run --with pytest --all-extras pytest tests/test_spardacus.py -v
"""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")
statsmodels = pytest.importorskip("statsmodels")
scipy = pytest.importorskip("scipy")

from safetycage.methods.spardacus import SPARDACUS


class MockModelModule:
    """Minimal ModelModule stand-in exposing fixed, separable activations.

    ``x`` is used purely as an index into precomputed activations/predictions
    arrays, so the mock never needs real model internals.
    """

    def __init__(self, activations, predictions):
        self.use_onehot_encoder = False
        self.selected_layers = ["layer1"]
        self._activations = activations
        self._predictions = predictions

    def _get_predictions(self, x):
        return self._predictions[x]

    def _get_activations(self, x):
        return {"layer1": self._activations[x]}


class MockDataModule:
    def __init__(self, x_train, y_train, num_classes):
        self._x_train = x_train
        self._y_train = y_train
        self._num_classes = num_classes

    @property
    def data_train(self):
        return self._x_train, self._y_train

    @property
    def classes(self):
        return {i: str(i) for i in range(self._num_classes)}

    @property
    def num_classes(self):
        return self._num_classes


@pytest.fixture
def separable_cage():
    """Two classes, well-separated correct/incorrect activation clusters.

    20 samples per (true_class, correct/incorrect) bucket -- comfortably
    above SPARDACUS's default minimum_sample_size=10 for GMM fitting.
    """
    rng = np.random.RandomState(0)
    n_per_bucket = 20
    n_features = 5

    x_indices, y_true, predictions, activations = [], [], [], []

    idx = 0
    for true_class in (0, 1):
        for pred_class, center in [(true_class, 0.0), (1 - true_class, 5.0)]:
            for _ in range(n_per_bucket):
                activations.append(rng.normal(loc=center, scale=0.5, size=n_features))
                y_true.append(true_class)
                predictions.append(pred_class)
                x_indices.append(idx)
                idx += 1

    model_module = MockModelModule(np.array(activations), np.array(predictions))
    data_module = MockDataModule(np.array(x_indices), np.array(y_true), num_classes=2)

    return model_module, data_module, np.array(x_indices), np.array(y_true)


def test_predict_after_fit_does_not_raise(separable_cage):
    """Regression test for a numpy 2.x incompatibility: ECDF(...) returns a
    shape-(1,) array, and numpy 2.x no longer allows assigning that into a
    scalar array slot (``arr[i, j] = np.array([x])``), raising
    "setting an array element with a sequence." This used to work under
    numpy 1.x's implicit squeeze.
    """
    model_module, data_module, x_indices, y_true = separable_cage

    spardacus = SPARDACUS(
        model_module=model_module, data_module=data_module, s_statistic_source="correctly"
    )
    spardacus.fit()

    statistics = spardacus.predict(x_indices, y_true)

    assert statistics.shape == (len(y_true),)
    assert np.all(np.isfinite(statistics) | np.isnan(statistics))


def test_projected_wasserstein_hand_computed():
    """x = {0, 1} and y = {0, 1, 2}: the quantile functions differ by 1 on
    (1/3, 1/2) and (2/3, 1), so the squared distance is 1/6 + 1/3 = 0.5."""
    xs = np.array([[0.0], [1.0]])
    ys = np.array([[0.0], [1.0], [2.0]])

    assert SPARDACUS.projectedWasserstein(xs, ys, np.ones(1)) == pytest.approx(0.5)
    assert SPARDACUS.projectedWasserstein(ys, xs, np.ones(1)) == pytest.approx(0.5)


def test_projected_wasserstein_equal_sizes_matches_sorted_difference():
    """With equal sample sizes the squared distance is the mean squared
    difference between the two sorted samples."""
    rng = np.random.default_rng(0)
    xs, ys = rng.normal(size=(50, 3)), rng.normal(loc=1.0, size=(50, 3))
    beta = np.array([0.6, 0.0, 0.8])

    expected = np.mean((np.sort(xs @ beta) - np.sort(ys @ beta)) ** 2)

    assert SPARDACUS.projectedWasserstein(xs, ys, beta) == pytest.approx(expected)


@pytest.mark.parametrize("n, m", [(3140, 3000), (3175, 3000), (3999, 3581)])
def test_projected_wasserstein_is_finite_for_awkward_sizes(n, m):
    """Regression test: adding 1/n and 1/m step by step drifted past the end
    of the sorted samples for some sizes, giving inf or nan."""
    rng = np.random.default_rng(0)
    xs, ys = rng.normal(size=(n, 1)), rng.normal(size=(m, 1))

    assert np.isfinite(SPARDACUS.projectedWasserstein(xs, ys, np.ones(1)))


def _relu_activations(seed, d=8):
    rng = np.random.default_rng(seed)
    xs = np.maximum(rng.normal(size=(120, d)), 0) * 3
    ys = np.maximum(rng.normal(size=(40, d)) + rng.normal(scale=0.5, size=d), 0) * 3
    return xs, ys


def test_l1sparda_does_not_modify_start_direction():
    xs, ys = _relu_activations(seed=2)
    beta0 = np.ones(xs.shape[1]) / np.sqrt(xs.shape[1])
    beta0_before = beta0.copy()

    SPARDACUS.l1SPARDA(xs, ys, 0, 1000, 1e-8, 1, 0, beta0)

    np.testing.assert_array_equal(beta0, beta0_before)


@pytest.mark.parametrize("seed", [0, 2, 6])
def test_l1sparda_returned_cost_matches_returned_direction(seed):
    """Regression test: ``last_beta = beta`` kept a reference, not a copy, so
    when a step made things worse the "undo" kept the bad direction while
    reporting the old, better cost."""
    xs, ys = _relu_activations(seed)
    beta0 = np.ones(xs.shape[1]) / np.sqrt(xs.shape[1])

    beta, cost = SPARDACUS.l1SPARDA(xs, ys, 0, 1000, 1e-8, 1, 0, beta0)

    assert cost == pytest.approx(SPARDACUS.projectedWasserstein(xs, ys, beta), rel=1e-12)
    assert cost >= SPARDACUS.projectedWasserstein(xs, ys, beta0)


@pytest.mark.parametrize("use_onehot_encoder", [False, True])
def test_fit_groups_incorrect_samples_by_predicted_class(monkeypatch, use_onehot_encoder):
    """The "incorrect" reference for class i must be samples the model wrongly
    *predicted* as i, because ``predict`` is called with the model's
    predictions. Grouping by the true class instead used samples that truly
    were i but were predicted as something else.

    Wrong predictions get activations that depend on the predicted class
    (predicted 0 -> around +5, predicted 1 -> around -5), so the two
    groupings pick out clearly different samples.
    """
    rng = np.random.RandomState(0)
    n_features = 5
    activations, y_true, predictions = [], [], []
    for true_class in (0, 1):
        for pred_class in (0, 1):
            center = 0.0 if pred_class == true_class else (5.0 if pred_class == 0 else -5.0)
            for _ in range(20):
                activations.append(rng.normal(loc=center, scale=0.5, size=n_features))
                y_true.append(true_class)
                predictions.append(pred_class)
    activations, y_true, predictions = np.array(activations), np.array(y_true), np.array(predictions)

    model_module = MockModelModule(activations, predictions)
    y_train = y_true
    if use_onehot_encoder:
        model_module.use_onehot_encoder = True
        y_train = np.eye(2)[y_true]
        # Probabilities rather than exact one-hot rows, as a softmax output would give
        model_module._predictions = np.eye(2)[predictions] * 0.8 + 0.1
    data_module = MockDataModule(np.arange(len(y_true)), y_train, num_classes=2)

    captured = {}

    def capture(self, correct, incorrect, class_label):
        captured[class_label] = (correct, incorrect)
        return {}

    monkeypatch.setattr(SPARDACUS, "_process_layer_class", capture)
    SPARDACUS(model_module, data_module, s_statistic_source="correctly").fit()

    for class_index, class_label in data_module.classes.items():
        correct, incorrect = captured[class_label]
        is_correct = predictions == y_true
        np.testing.assert_array_equal(correct, activations[is_correct & (predictions == class_index)])
        np.testing.assert_array_equal(incorrect, activations[~is_correct & (predictions == class_index)])


def test_s_statistic_source_defaults_to_correctly(separable_cage):
    model_module, data_module, _, _ = separable_cage

    assert SPARDACUS(model_module, data_module).s_statistic_source == "correctly"


def test_invalid_s_statistic_source_raises(separable_cage):
    model_module, data_module, _, _ = separable_cage

    with pytest.raises(ValueError, match="s_statistic_source"):
        SPARDACUS(model_module, data_module, s_statistic_source="correct")


def test_fit_clears_unreliable_classes_from_previous_fit(separable_cage):
    model_module, data_module, _, _ = separable_cage
    spardacus = SPARDACUS(model_module, data_module)
    spardacus.unreliable_classes.add("0")

    spardacus.fit()

    assert spardacus.unreliable_classes == set()


def test_cauchy_combination_accepts_numpy_weights():
    p_values = np.array([0.1, 0.4])

    from_array = SPARDACUS.CauchyCombinationTest(p_values, np.array([0.25, 0.75]))
    from_list = SPARDACUS.CauchyCombinationTest(p_values, [0.25, 0.75])

    assert from_array == pytest.approx(from_list)
    assert SPARDACUS.CauchyCombinationTest(p_values, np.array([])) == pytest.approx(
        SPARDACUS.CauchyCombinationTest(p_values)
    )


def test_random_projection_search_handles_one_feature():
    rng = np.random.default_rng(0)
    xs, ys = rng.normal(size=(50, 1)), rng.normal(loc=2.0, size=(30, 1))

    beta, dist, _ = SPARDACUS.randomProjectionSearch(xs, ys)

    assert beta.shape == (1,)
    assert np.isfinite(dist)


def _bimodal(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(-5, 0.5, n // 2), rng.normal(5, 0.5, n - n // 2)])


def test_gmm_with_too_few_samples_marks_class_unreliable(separable_cage):
    model_module, data_module, _, _ = separable_cage
    spardacus = SPARDACUS(model_module, data_module)

    with pytest.warns(UserWarning, match="unreliable"):
        density = spardacus._fit_gaussian_mixture(_bimodal(9), "incorrect", "aspen")

    assert density is None
    assert "aspen" in spardacus.unreliable_classes


@pytest.mark.parametrize("n_samples, max_components", [(10, 1), (19, 1), (25, 2)])
def test_gmm_components_are_capped_by_sample_size(separable_cage, n_samples, max_components):
    """At least 10 samples per component: even clearly two-humped data gets a
    single Gaussian when there are fewer than 20 samples."""
    model_module, data_module, _, _ = separable_cage
    spardacus = SPARDACUS(model_module, data_module)

    density = spardacus._fit_gaussian_mixture(_bimodal(n_samples), "correct", "aspen")

    assert 1 <= density.n_components <= max_components
    assert spardacus.unreliable_classes == set()


def test_gmm_picks_components_by_bic_on_all_samples(separable_cage):
    model_module, data_module, _, _ = separable_cage
    spardacus = SPARDACUS(model_module, data_module)

    density = spardacus._fit_gaussian_mixture(_bimodal(400), "correct", "aspen")

    assert density.n_components == 2


def test_pvalue_method_defaults_to_monte_carlo(separable_cage):
    model_module, data_module, _, _ = separable_cage

    assert SPARDACUS(model_module, data_module).pvalue_method == "monte_carlo"


@pytest.mark.parametrize("kwargs", [{"pvalue_method": "exact"}, {"calibration_fraction": 0.0}, {"calibration_fraction": 1.0}])
def test_invalid_conformal_settings_raise(separable_cage, kwargs):
    model_module, data_module, _, _ = separable_cage

    with pytest.raises(ValueError):
        SPARDACUS(model_module, data_module, **kwargs)


@pytest.mark.parametrize("source, expected", [("correctly", 3 / 5), ("incorrectly", 4 / 5)])
def test_conformal_pvalue_counts_calibration_scores(separable_cage, source, expected):
    """Calibration scores [1, 2, 3, 4] and a new score of 3:
    "correctly" counts scores >= 3 (two of them) -> (1 + 2) / (4 + 1);
    "incorrectly" counts scores <= 3 (three of them) -> (1 + 3) / (4 + 1)."""
    model_module, data_module, _, _ = separable_cage
    spardacus = SPARDACUS(model_module, data_module, s_statistic_source=source, pvalue_method="conformal")

    assert spardacus._conformal_pvalue(np.array([1.0, 2.0, 3.0, 4.0]), 3.0) == pytest.approx(expected)


@pytest.fixture
def skewed_cage():
    """Correct activations follow a skewed (exponential) distribution that a
    3-component Gaussian mixture can only approximate; incorrect ones are
    shifted. Rows from n_train onwards are fresh, correctly classified test
    samples that the cage never sees during fit."""
    rng = np.random.RandomState(1)
    n_correct, n_incorrect, n_test, n_features = 1500, 150, 4000, 3
    activations, y_true, predictions = [], [], []
    for cls in (0, 1):
        activations.append(rng.exponential(1.0, size=(n_correct, n_features)))
        activations.append(rng.normal(3.0, 1.0, size=(n_incorrect, n_features)))
        y_true += [cls] * n_correct + [1 - cls] * n_incorrect
        predictions += [cls] * (n_correct + n_incorrect)
    n_train = len(y_true)
    for cls in (0, 1):
        activations.append(rng.exponential(1.0, size=(n_test, n_features)))
        y_true += [cls] * n_test
        predictions += [cls] * n_test

    model_module = MockModelModule(np.vstack(activations), np.array(predictions))
    data_module = MockDataModule(np.arange(n_train), np.array(y_true[:n_train]), num_classes=2)
    test_indices = np.arange(n_train, len(y_true))
    return model_module, data_module, test_indices, np.array(predictions)[test_indices]


def test_conformal_fit_skips_monte_carlo_and_holds_out_calibration(skewed_cage, monkeypatch):
    model_module, data_module, _, _ = skewed_cage

    def no_monte_carlo(*args, **kwargs):
        raise AssertionError("conformal p-values must not draw Monte Carlo samples")

    monkeypatch.setattr(SPARDACUS, "_compute_log_pdfs", no_monte_carlo)
    spardacus = SPARDACUS(model_module, data_module, pvalue_method="conformal", calibration_fraction=0.3)
    np.random.seed(0)
    spardacus.fit()

    for params in spardacus.layer_params["layer1"].values():
        if params:
            assert len(params["calibration_scores"]) == round(0.3 * 1500)


def test_conformal_pvalues_are_calibrated_on_fresh_correct_samples(skewed_cage):
    """For new correct predictions, P(p <= alpha) should be close to alpha,
    and p is never 0 (smallest possible value is 1 / (n_calibration + 1))."""
    model_module, data_module, test_indices, test_predictions = skewed_cage
    spardacus = SPARDACUS(model_module, data_module, pvalue_method="conformal")
    np.random.seed(0)
    spardacus.fit()

    pvalues = spardacus.predict(test_indices, test_predictions)

    assert pvalues.min() >= 1 / (round(0.3 * 1500) + 1)
    for alpha in (0.05, 0.1, 0.2):
        assert np.mean(pvalues <= alpha) == pytest.approx(alpha, abs=0.02)


SAVED_SETTINGS = {
    "s_statistic_source": "incorrectly",
    "pvalue_method": "conformal",
    "calibration_fraction": 0.25,
    "test_type_between_layers": "cauchy",
    "cauchy_weights_per_layer": [1.0],
    "minimum_sample_size": 5,
}


def test_save_and_load_keep_settings_and_predictions(separable_cage, tmp_path):
    model_module, data_module, x_indices, y_true = separable_cage
    spardacus = SPARDACUS(model_module, data_module, **SAVED_SETTINGS)
    np.random.seed(0)
    spardacus.fit()
    spardacus.threshold = 0.05
    path = tmp_path / "spardacus.joblib"

    spardacus.save_cage(path)
    loaded = SPARDACUS.load_cage(path, model_module, data_module)

    for name, value in SAVED_SETTINGS.items():
        assert getattr(loaded, name) == value
    assert loaded.threshold == 0.05
    np.testing.assert_array_equal(loaded.predict(x_indices, y_true), spardacus.predict(x_indices, y_true))


def test_load_cage_arguments_override_saved_settings(separable_cage, tmp_path):
    model_module, data_module, _, _ = separable_cage
    spardacus = SPARDACUS(model_module, data_module, test_type_between_layers="cauchy")
    spardacus.fit()
    spardacus.threshold = 0.05
    path = tmp_path / "spardacus.joblib"
    spardacus.save_cage(path)

    loaded = SPARDACUS.load_cage(path, model_module, data_module, test_type_between_layers="fisher")

    assert loaded.test_type_between_layers == "fisher"


def test_load_cage_without_saved_settings_warns_and_uses_defaults(separable_cage, tmp_path):
    """Files written before settings were saved hold only threshold and layer_params."""
    import joblib

    model_module, data_module, x_indices, y_true = separable_cage
    spardacus = SPARDACUS(model_module, data_module)
    spardacus.fit()
    path = tmp_path / "old.joblib"
    joblib.dump({"threshold": 0.05, "layer_params": spardacus.layer_params}, path)

    with pytest.warns(UserWarning, match="settings"):
        loaded = SPARDACUS.load_cage(path, model_module, data_module)

    assert loaded.s_statistic_source == "correctly"
    assert loaded.pvalue_method == "monte_carlo"
    assert np.all(np.isfinite(loaded.predict(x_indices, y_true)))
