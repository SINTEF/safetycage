import numpy as np
from safetycage.safetycage import SafetyCage


class RandomBaseline(SafetyCage):
    """
    Prevalence-weighted random flagging baseline.

    This is not a misclassification detector -- it flags samples at random,
    with no knowledge of the specific input. It exists so other safety cage
    methods can be judged against "flagging informed only by how often each
    predicted class is wrong", rather than against pure chance or against a
    do-nothing baseline. A cage that cannot beat this at the same review
    budget is not adding value beyond the class-level prior.

    ``fit`` estimates, per predicted class, how often the model is
    wrong on the training split (i.e. the empirical error rate conditioned
    on the predicted label). ``predict`` then draws one uniform random
    number per sample and divides it by its predicted class's error rate:
    samples from classes the model gets wrong more often end up with smaller
    values more often, so they are more likely to fall below any given
    threshold. Within a class, flagging is still purely random.

    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
        random_state (int): Seed for the random number generator.
        class_error_rate (numpy.ndarray): Per-class empirical error rate,
            learned by ``fit``.
    """

    def __init__(self, model_module, data_module, **kwargs):
        """
        Initialize the random baseline.

        Args:
            model_module: Reference to model module object for making predictions.
            data_module: Reference to data module object for handling data.
            random_state (int): Seed for the random number generator (default: 42).
            **kwargs: Additional keyword arguments.
        """
        super(RandomBaseline, self).__init__(model_module, data_module, **kwargs)
        self.leq = True
        self.random_state = kwargs.get("random_state", 42)
        self.rng = np.random.default_rng(self.random_state)
        self.class_error_rate = None

    @property
    def name(self):
        """Return the name of the safety cage method."""
        return "RandomBaseline"

    def fit(
        self,
        x=None,
        y=None,
        y_pred=None,
        find_threshold=False,
        metric_fn=None,
        x_val=None,
        y_val=None,
        search="uniform",
        num_thresholds=1000,
        greater_is_better=True,
    ) -> None:
        """
        Estimate the empirical error rate of each predicted class.

        For every class c, this is P(y_pred != y | y_pred == c) on the
        training split -- how often the model is wrong when it predicts c.
        Classes never predicted on the training split fall back to the
        overall error rate.

        Args:
            x: Input data. If None, loaded from the data module.
            y: Ground-truth labels. If None, loaded from the data module.
            y_pred: Model predictions. If None, computed using the model module.
            find_threshold (bool, optional): If True, also choose ``self.threshold``
                on validation data after fitting -- see ``SafetyCage.fit_threshold``
                for what the remaining keyword arguments do. (default: False)
            metric_fn (callable, optional): Metric to maximize (default: MCC).
            x_val: Validation inputs. If None, loaded from the data module.
            y_val: Validation labels. If None, loaded from the data module.
            search (str, optional): Threshold sweep strategy (default: "uniform").
            num_thresholds (int, optional): Number of candidate thresholds (default: 1000).
            greater_is_better (bool, optional): Whether greater metric values are better (default: True).
        """
        if x is None:
            x, y = self.data_module.data_train
        if y is None:
            _, y = self.data_module.data_train
        if y_pred is None:
            y_pred = self.model_module._get_predictions(x)

        incorrect = (y_pred != y).astype(np.float64)

        class_error_rate = np.full(self.num_classes, incorrect.mean(), dtype=np.float64)
        for c in range(self.num_classes):
            predicted_c = y_pred == c
            if predicted_c.any():
                class_error_rate[c] = incorrect[predicted_c].mean()

        # Clip away from 0 so a class that was never wrong doesn't produce
        # an infinite (and therefore un-thresholdable) statistic.
        self.class_error_rate = np.clip(class_error_rate, 1e-6, None)

        if find_threshold:
            self.fit_threshold(
                metric_fn=metric_fn,
                x_val=x_val,
                y_val=y_val,
                search=search,
                num_thresholds=num_thresholds,
                greater_is_better=greater_is_better,
            )

    def predict(self, x, y) -> np.ndarray:
        """
        Draw a prevalence-weighted random statistic for each sample.

        Args:
            x: Input data. Unused -- flagging depends only on the predicted class.
            y (numpy.ndarray): Predicted class labels, used to look up each
                class's empirical error rate from ``fit``.

        Returns:
            numpy.ndarray: Random statistics; lower values are more likely
                to be flagged as misclassifications.
        """
        statistics = self._compute_statistics(x, y)

        return statistics

    def _compute_statistics(self, x, y):
        """
        Compute per-sample statistics as u / class_error_rate[y_pred].

        Args:
            x: Input data. Unused.
            y (numpy.ndarray): Predicted class labels.

        Returns:
            numpy.ndarray: Per-sample random statistics.
        """
        if self.class_error_rate is None:
            raise ValueError("RandomBaseline has not been trained. Call fit() first.")

        y_pred = np.asarray(y)
        u = self.rng.uniform(low=1e-12, high=1.0, size=len(y_pred))
        statistics = u / self.class_error_rate[y_pred]

        return statistics


if __name__ == "__main__":
    RandomBaseline(None, None)
