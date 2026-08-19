import numpy as np
from tqdm import tqdm

from safetycage.safetycage import SafetyCage

try:
    import torch
    import gpytorch
    from gpytorch.models import ApproximateGP
    from gpytorch.variational import (
        CholeskyVariationalDistribution,
        VariationalStrategy,
    )
    from gpytorch.kernels import ScaleKernel, RBFKernel, Kernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import VariationalELBO
    from torch.utils.data import TensorDataset, DataLoader

    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False

DEVICE = torch.device("cpu") if HAS_GPYTORCH else None


# ---------------------------------------------------------------------------
# GPyTorch components (defined at module level, guarded by HAS_GPYTORCH)
# ---------------------------------------------------------------------------
if HAS_GPYTORCH:

    class IOKernel(Kernel):
        """I/O kernel from the RED paper (Qiu & Miikkulainen, 2022).

        Splits the concatenated feature vector [x, sigma] at ``input_dim`` and
        applies independent ScaleKernel(RBFKernel) to each part.  The final
        kernel value is the sum of the two.

        k((x_i, sigma_i), (x_j, sigma_j)) = k_in(x_i, x_j) + k_out(sigma_i, sigma_j)
        """

        def __init__(self, input_dim, **kwargs):
            super().__init__(**kwargs)
            self.input_dim = input_dim
            self.k_in = ScaleKernel(RBFKernel(ard_num_dims=input_dim))
            self.k_out = ScaleKernel(RBFKernel())

        def forward(self, x1, x2, diag=False, **params):
            x1_in, x1_out = x1[:, :self.input_dim], x1[:, self.input_dim:]
            x2_in, x2_out = x2[:, :self.input_dim], x2[:, self.input_dim:]

            k_in_val = self.k_in(x1_in, x2_in, diag=diag, **params)
            k_out_val = self.k_out(x1_out, x2_out, diag=diag, **params)

            return k_in_val + k_out_val

    class SVGPModel(ApproximateGP):
        """Stochastic Variational GP for residual prediction in RED."""

        def __init__(self, inducing_points, input_dim):
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = IOKernel(input_dim=input_dim)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RED(SafetyCage):
    """
    Residual-based Error Detection (RED) Safety Cage Method.

    RED builds a Gaussian Process (GP) error detector on top of a base
    classifier. It learns to predict the residual between a binary correctness
    indicator (1 = correct, 0 = incorrect) and the classifier's maximum class
    probability. The calibrated detection score separates correct from incorrect
    predictions better than raw softmax confidence.

    The GP uses an I/O kernel that combines an input-space RBF kernel with an
    output-space RBF kernel operating on the full softmax probability vector.
    A Stochastic Variational GP (SVGP) is used for scalability.

    The detection score is ``c_hat + r_mean`` where higher values indicate a
    prediction is more likely correct. Flagging uses ``leq=True``: samples with
    score <= alpha are flagged as potential misclassifications.

    After calling ``predict()``, the GP variance is available as
    ``self._last_uncertainty`` for advanced use (e.g. OOD detection).

    NOTE: Requires ``gpytorch`` and ``torch``. Install via:
        ``pip install safetycage[red]``

    **Reference:**
        Qiu, X. & Miikkulainen, R. (2022). Detecting Misclassification Errors
        in Neural Networks with a Gaussian Process Model. AAAI 2022.

    Attributes:
        model_module: Reference to model module object for making predictions.
        data_module: Reference to data module object for handling data.
        num_inducing_points (int): Number of inducing points for the SVGP.
        training_iterations (int): Number of training iterations for the SVGP.
        learning_rate (float): Learning rate for Adam optimizer.
        batch_size (int): Mini-batch size for SVGP training.
        random_state (int): Random seed for reproducibility.
    """

    def __init__(self, model_module, data_module, **kwargs):
        """
        Initialize the RED safety cage method.

        Args:
            model_module: Reference to model module object for making predictions.
            data_module: Reference to data module object for handling data.
            num_inducing_points (int): Number of inducing points for SVGP (default: 500).
            training_iterations (int): Number of training iterations (default: 100).
            learning_rate (float): Learning rate for Adam optimizer (default: 0.01).
            batch_size (int): Mini-batch size for training (default: 256).
            random_state (int): Random seed (default: 42).
            **kwargs: Additional keyword arguments.
        """
        if not HAS_GPYTORCH:
            raise ImportError(
                "RED requires gpytorch and torch. "
                "Install them with: pip install safetycage[red]"
            )

        super(RED, self).__init__(model_module, data_module, **kwargs)
        self.leq = True

        self.num_inducing_points = kwargs.get("num_inducing_points", 500)
        self.training_iterations = kwargs.get("training_iterations", 100)
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.batch_size = kwargs.get("batch_size", 256)
        self.random_state = kwargs.get("random_state", 42)
        self.alpha = kwargs.get("alpha", None)

        self._last_uncertainty = None

    @property
    def name(self):
        """Return the name of the safety cage method."""
        return "RED"

    def train_cage(self, x=None, y=None, y_pred=None) -> None:
        """
        Train the RED safety cage.

        Computes residuals between binary correctness targets and maximum class
        probabilities, then trains a SVGP with an I/O kernel to predict these
        residuals.

        Args:
            x: Input data. If None, loaded from the data module.
            y: Ground-truth labels. If None, loaded from the data module.
            y_pred: Model predictions. If None, computed using the model module.
        """
        if x is None:
            x, y = self.data_module.data_train
        if y is None:
            _, y = self.data_module.data_train
        if y_pred is None:
            y_pred = self.model_module._get_predictions(x)

        # Step 1: target detection score c_i = delta(y_i, y_hat_i)
        if self.model_module.use_onehot_encoder:
            correct = (np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)).astype(np.float32)
        else:
            correct = (y_pred == y).astype(np.float32)

        # Step 2: softmax probabilities and max class probability
        sigma = self.model_module._get_probabilities(x)
        c_hat = np.max(sigma, axis=1).astype(np.float32)

        # Step 3: residuals
        residuals = correct - c_hat

        # Step 4: composite features [x, sigma]
        x_float = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
        sigma_float = np.asarray(sigma, dtype=np.float32)
        features = np.hstack([x_float, sigma_float])
        input_dim = x_float.shape[1]

        # Step 5: select inducing points (random subset)
        n = features.shape[0]
        n_inducing = min(self.num_inducing_points, n)
        rng = np.random.RandomState(self.random_state)
        inducing_idx = rng.choice(n, n_inducing, replace=False)

        features_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)
        residuals_tensor = torch.tensor(residuals, dtype=torch.float32, device=DEVICE)
        inducing_points = features_tensor[inducing_idx].clone()

        # Step 6: build SVGP model
        model = SVGPModel(inducing_points, input_dim).to(DEVICE)
        likelihood = GaussianLikelihood().to(DEVICE)

        # Step 7: train
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=self.learning_rate,
        )
        mll = VariationalELBO(likelihood, model, num_data=n)

        dataset = TensorDataset(features_tensor, residuals_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.loss_history = []

        progress = tqdm(range(self.training_iterations), desc="Training RED")
        for _ in progress:
            epoch_losses = []
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            epoch_loss = float(np.mean(epoch_losses))
            self.loss_history.append(epoch_loss)
            progress.set_postfix(loss=epoch_loss)

        model.eval()
        likelihood.eval()

        # Store for persistence and prediction
        self.layer_params = {
            "gp_model": model,
            "likelihood": likelihood,
            "input_dim": input_dim,
            "num_inducing_points": n_inducing,
        }

    def predict(self, x, y) -> np.ndarray:
        """
        Compute RED detection scores for input samples.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray): Predicted labels (used for API consistency).

        Returns:
            numpy.ndarray: Detection scores (higher = more likely correct).
        """
        statistics = self._compute_statistics(x, y)
        return statistics

    def _compute_statistics(self, x, y):
        """
        Compute detection scores as c_hat + r_mean.

        Also stores the GP variance in ``self._last_uncertainty``.

        Args:
            x: Input data samples.
            y: Labels (not used directly, included for API consistency).

        Returns:
            numpy.ndarray: Detection scores per sample.
        """
        sigma = self.model_module._get_probabilities(x)
        c_hat = np.max(sigma, axis=1)

        x_float = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
        sigma_float = np.asarray(sigma, dtype=np.float32)
        features = np.hstack([x_float, sigma_float])
        features_tensor = torch.tensor(features, dtype=torch.float32, device=DEVICE)

        model = self.layer_params["gp_model"]
        likelihood = self.layer_params["likelihood"]

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(features_tensor))
            r_mean = pred.mean.cpu().numpy()
            r_var = pred.variance.cpu().numpy()

        self._last_uncertainty = r_var

        detection_scores = c_hat + r_mean
        return detection_scores

    def flag(self, statistics, alpha=None) -> float | np.ndarray:
        """
        Flag samples with detection score <= alpha as potential misclassifications.

        Args:
            statistics (numpy.ndarray): Detection scores from ``predict()``.
            alpha (float, optional): Threshold for flagging samples. Falls back
                to ``self.alpha`` if not provided.

        Returns:
            numpy.ndarray: Boolean array where True indicates a flagged sample.
        """
        if alpha is None:
            if getattr(self, "alpha", None) is not None:
                alpha = self.alpha
            else:
                raise ValueError("Missing alpha parameter: must be provided as input or set as class attribute")

        return statistics <= alpha

    def find_best_threshold(self, y_true, y_probs, metric_fn, **kwargs) -> dict:
        """
        Find the optimal alpha for flagging samples.

        Thin wrapper over the base implementation that renames the result key
        from ``threshold_opt`` to ``alpha_opt`` to match RED's own terminology.

        Returns:
            dict: Same as the base implementation, with ``alpha_opt`` instead
                of ``threshold_opt``.
        """
        result = super().find_best_threshold(y_true, y_probs, metric_fn, **kwargs)
        result["alpha_opt"] = float(result.pop("threshold_opt"))
        return result

    def save_cage(self, path):
        """Save trained RED cage to disk.

        Saves the GP model state (via torch) and metadata (via joblib) to
        separate files sharing the same base path.

        Args:
            path (str or Path): File path (should end in .joblib). A companion
                .pt file will be created for the torch model state.
        """
        from pathlib import Path as _Path
        import joblib

        if getattr(self, "alpha", None) is None:
            raise ValueError("Cannot save: cage has not been trained (alpha is not set).")

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save torch model + likelihood state
        pt_path = path.with_suffix(".pt")
        torch.save(
            {
                "model_state": self.layer_params["gp_model"].state_dict(),
                "likelihood_state": self.layer_params["likelihood"].state_dict(),
                "input_dim": self.layer_params["input_dim"],
                "num_inducing_points": self.layer_params["num_inducing_points"],
            },
            pt_path,
        )

        # Save alpha and other metadata via joblib
        parameters = {"alpha": self.alpha, "has_pt_file": True}
        joblib.dump(parameters, path)

    @classmethod
    def load_cage(cls, path, model_module, data_module, **kwargs):
        """Load a trained RED cage from saved files.

        Args:
            path (str or Path): Path to the .joblib file.
            model_module: The model module to use with the loaded cage.
            data_module: The data module to use with the loaded cage.

        Returns:
            RED: An instance with trained parameters restored.
        """
        from pathlib import Path as _Path
        import joblib

        path = _Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No saved cage found at {path}")

        parameters = joblib.load(path)

        # Load torch state
        pt_path = path.with_suffix(".pt")
        if not pt_path.is_file():
            raise FileNotFoundError(f"No torch model file found at {pt_path}")

        checkpoint = torch.load(pt_path, map_location=DEVICE, weights_only=False)
        input_dim = checkpoint["input_dim"]
        n_inducing = checkpoint["num_inducing_points"]

        # Reconstruct model architecture with dummy inducing points
        dummy_inducing = torch.zeros(n_inducing, input_dim + data_module.num_classes, device=DEVICE)
        model = SVGPModel(dummy_inducing, input_dim).to(DEVICE)
        likelihood = GaussianLikelihood().to(DEVICE)

        model.load_state_dict(checkpoint["model_state"])
        likelihood.load_state_dict(checkpoint["likelihood_state"])
        model.eval()
        likelihood.eval()

        instance = cls(model_module=model_module, data_module=data_module, **kwargs)
        instance.alpha = parameters["alpha"]
        instance.layer_params = {
            "gp_model": model,
            "likelihood": likelihood,
            "input_dim": input_dim,
            "num_inducing_points": n_inducing,
        }

        return instance


if __name__ == "__main__":
    RED(None, None)
