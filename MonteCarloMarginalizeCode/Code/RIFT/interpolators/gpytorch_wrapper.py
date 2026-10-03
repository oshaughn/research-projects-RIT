"""Deterministic, double-precision exact GPs for intrinsic log likelihoods.

Coordinates and targets are standardized once, using the training set. Reported
``y_errors`` are standard deviations in the original target units: their squared,
normalized values enter a fixed heteroscedastic likelihood, in addition to a
learned diagonal noise term. ``evaluate`` returns the latent posterior mean,
never a likelihood draw or an uncertainty-adjusted value.

This is an O(N**3) exact GP, not a scalable full-grid approximation. The explicit
training-size guard must be raised deliberately for larger fits. Prediction
caches the training solve and forms only bounded query-by-training kernel blocks.
"""
from contextlib import contextmanager
import hashlib
import json
import platform

import gpytorch
import numpy as np
import torch


DEFAULT_MAX_TRAIN_POINTS = 8000
FORMAT_VERSION = 1
FIXED_NOISE_FLOOR = 1e-10


def _array(value, name, ndim):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError("{} must be a finite {}-dimensional array".format(name, ndim))
    return result


def _vector(value, name, n):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape == (n, 1):
        result = result[:, 0]
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError("{} must have shape (N,) or (N, 1), with finite values".format(name))
    return result


def _device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(value)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported for float64 GP fits")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return selected


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


class ExactGPModel(gpytorch.models.ExactGP):
    """Normalized-target ARD Matérn-5/2 GP, with zero prior mean.

    Kernel/noise bounds match the independent Matérn diagnostic's conventions.
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=train_x.shape[1],
                lengthscale_constraint=gpytorch.constraints.Interval(0.03, 30.0),
            ),
            outputscale_constraint=gpytorch.constraints.Interval(0.01, 100.0),
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class Interpolator:
    """An exact GP with the legacy ``train``/``evaluate`` interface.

    ``input`` is (N, D); targets/errors may be (N,) or (N, 1). Input arrays are
    never modified. ``device='auto'`` selects CUDA when available, otherwise CPU
    (MPS cannot provide the required float64). The training cap is an error,
    never an implicit subsampling operation.
    """

    def __init__(self, input, target, epochs=100, learning_rate=1e-1,
                 betas=(0.9, 0.99), eps=1e-2, weight_decay=1e-6, *,
                 y_errors=None, device="auto", prediction_batch_size=1024,
                 max_train_points=DEFAULT_MAX_TRAIN_POINTS,
                 feature_names=None, provenance=None):
        x = _array(input, "input", 2)
        n, self.n_features = x.shape
        self.max_train_points = int(max_train_points)
        self._check_size(n)
        if self.n_features == 0:
            raise ValueError("At least one input feature is required")
        y = _vector(target, "target", n)
        errors = np.zeros(n) if y_errors is None else _vector(y_errors, "y_errors", n)
        if np.any(errors < 0):
            raise ValueError("y_errors must be nonnegative standard deviations")
        self.epochs = int(epochs)
        if self.epochs < 0:
            raise ValueError("epochs must be nonnegative")
        self.prediction_batch_size = int(prediction_batch_size)
        if self.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        self.device = _device(device)
        self.input_mu = x.mean(axis=0)
        self.input_sigma = x.std(axis=0)
        self.input_sigma[self.input_sigma == 0] = 1.0
        self.target_mu = float(y.mean())
        self.target_sigma = float(y.std()) or 1.0
        self.feature_names = None if feature_names is None else list(feature_names)
        if self.feature_names is not None and (
                len(self.feature_names) != self.n_features or
                not all(isinstance(name, str) for name in self.feature_names)):
            raise ValueError("feature_names must contain one string per input column")
        # Restrict provenance to portable JSON values, rather than pickled objects.
        self.provenance = json.loads(json.dumps(provenance or {}, allow_nan=False))
        self.provenance.update({
            "training_input_sha256": _digest(x),
            "training_target_sha256": _digest(y),
            "training_error_sha256": _digest(errors),
            "reported_errors_supplied": y_errors is not None,
        })
        self.optimizer_options = dict(lr=float(learning_rate), betas=tuple(betas),
                                      eps=float(eps), weight_decay=float(weight_decay))
        self.input_train = self._tensor((x-self.input_mu)/self.input_sigma)
        self.target_train = self._tensor((y-self.target_mu)/self.target_sigma)
        self.fixed_noise = self._tensor(np.maximum(
            (errors/self.target_sigma)**2, FIXED_NOISE_FLOOR))
        self.loss_history = []
        self.gp_init()
        self.optim_init(learning_rate, betas, eps, weight_decay)

    def _check_size(self, n):
        if self.max_train_points < 2 or n < 2:
            raise ValueError("Exact GP requires at least two training points and a cap >=2")
        if n > self.max_train_points:
            raise ValueError(
                "Exact gp-torch training has {} points, exceeding the explicit limit {}. "
                "Use --cap-points to select a bounded set or deliberately raise "
                "--gp-torch-max-train-points after assessing O(N^3) compute and O(N^2) memory."
                .format(n, self.max_train_points))

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float64, device=self.device).clone()

    @contextmanager
    def _exact_settings(self):
        # Avoid stochastic log-determinant estimates/iterative solves in this
        # bounded exact backend. No process-global dtype/device setting changes.
        with gpytorch.settings.max_cholesky_size(self.max_train_points+1), \
                gpytorch.settings.fast_computations(
                    covar_root_decomposition=False, log_prob=False, solves=False):
            yield

    def gp_init(self):
        with gpytorch.settings.min_fixed_noise(double_value=FIXED_NOISE_FLOOR):
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=self.fixed_noise, learn_additional_noise=True,
                noise_constraint=gpytorch.constraints.Interval(1e-5, 0.1),
            ).to(device=self.device, dtype=torch.float64)
        self.likelihood.second_noise = 1e-3
        self.model = ExactGPModel(self.input_train, self.target_train, self.likelihood)
        self.model = self.model.to(device=self.device, dtype=torch.float64)
        self.model.covar_module.base_kernel.lengthscale = 1.0
        self.model.covar_module.outputscale = 1.0
        self._mean_weights = None

    def optim_init(self, learning_rate, betas, eps, weight_decay):
        self.optimizer_options = dict(lr=float(learning_rate), betas=tuple(betas),
                                      eps=float(eps), weight_decay=float(weight_decay))
        self.optim = torch.optim.Adam(self.model.parameters(), **self.optimizer_options)

    def train(self):
        self._mean_weights = None
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        with self._exact_settings():
            for epoch in range(self.epochs):
                self.optim.zero_grad()
                loss = -mll(self.model(self.input_train), self.target_train)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite gp-torch training loss at epoch {}".format(epoch))
                loss.backward()
                self.optim.step()
                self.loss_history.append(float(loss.detach().cpu()))
        self.model.eval()
        self.likelihood.eval()
        return self

    def _prepare_mean(self):
        if self._mean_weights is None:
            self.model.eval()
            self.likelihood.eval()
            with torch.no_grad(), self._exact_settings():
                prior = self.model.forward(self.input_train)
                observations = self.likelihood(prior)
                rhs = (self.target_train-prior.mean).unsqueeze(-1)
                self._mean_weights = observations.lazy_covariance_matrix.solve(rhs).detach()

    def _cross_covariance(self, query):
        # GPyTorch's Matérn forward centers by the *query batch* mean before
        # computing distances. A distant distractor can then spoil precision
        # for nearby queries. Direct coordinate differences avoid that batch
        # dependence and the cancellation in a squared-norm matrix identity.
        lengthscale = self.model.covar_module.base_kernel.lengthscale
        distance = torch.cdist(query/lengthscale, self.input_train/lengthscale,
                               compute_mode="donot_use_mm_for_euclid_dist")
        scaled = np.sqrt(5.0)*distance
        return self.model.covar_module.outputscale * (
            1.0+scaled+scaled.square()/3.0) * torch.exp(-scaled)

    def evaluate(self, input, batch_size=None):
        """Return deterministic latent means as a float64 NumPy vector.

        Only B-by-N kernel blocks are formed, where B <= batch_size; no query
        covariance or posterior samples are constructed. Errors from training
        observations affect the cached solve, never query cross-covariances.
        """
        x = _array(input, "input", 2)
        if x.shape[1] != self.n_features:
            raise ValueError("Query feature count differs from the training feature count")
        size = self.prediction_batch_size if batch_size is None else int(batch_size)
        if size < 1:
            raise ValueError("batch_size must be positive")
        if len(x) == 0:
            return np.empty(0, dtype=np.float64)
        self._prepare_mean()
        result = np.empty(len(x), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(x), size):
                query = self._tensor((x[start:start+size]-self.input_mu)/self.input_sigma)
                cross = self._cross_covariance(query)
                mean = self.model.mean_module(query) + (cross @ self._mean_weights).squeeze(-1)
                result[start:start+size] = mean.cpu().numpy()*self.target_sigma+self.target_mu
        return result

    def save(self, filename):
        """Save portable tensors and explicit model/scaler metadata at filename.

        No arbitrary model object, optimizer object, CUDA cache or pickle-defined
        class is serialized. Load with ``Interpolator.load`` (weights_only=True).
        """
        state = {
            "format": "RIFT.gpytorch.exact", "format_version": FORMAT_VERSION,
            "kernel": "ScaleKernel(ARD-Matern-2.5)", "mean": "zero-normalized-target",
            "dtype": "float64", "feature_names": self.feature_names,
            "input_mu": torch.from_numpy(self.input_mu.copy()),
            "input_sigma": torch.from_numpy(self.input_sigma.copy()),
            "target_mu": self.target_mu, "target_sigma": self.target_sigma,
            "train_x": self.input_train.detach().cpu(),
            "train_y": self.target_train.detach().cpu(),
            "fixed_noise": self.fixed_noise.detach().cpu(),
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "epochs": self.epochs, "loss_history": self.loss_history,
            "optimizer_options": self.optimizer_options,
            "prediction_batch_size": self.prediction_batch_size,
            "max_train_points": self.max_train_points,
            "provenance": self.provenance,
            "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "gpytorch": str(gpytorch.__version__), "numpy": str(np.__version__)},
        }
        torch.save(state, filename)

    @classmethod
    def load(cls, filename, *, device="auto", prediction_batch_size=None,
             max_train_points=DEFAULT_MAX_TRAIN_POINTS, expected_feature_names=None):
        """Restore explicit state, optionally checking native coordinate order."""
        state = torch.load(filename, map_location="cpu", weights_only=True)
        if (state.get("format") != "RIFT.gpytorch.exact" or
                state.get("format_version") != FORMAT_VERSION or
                state.get("kernel") != "ScaleKernel(ARD-Matern-2.5)" or
                state.get("mean") != "zero-normalized-target" or state.get("dtype") != "float64"):
            raise ValueError("Unsupported gp-torch checkpoint format, kernel, mean or dtype")
        names = state["feature_names"]
        if expected_feature_names is not None and names != list(expected_feature_names):
            raise ValueError("Saved gp-torch feature names/order do not match native fit coordinates")
        obj = cls.__new__(cls)
        obj.device = _device(device)
        obj.max_train_points = int(max_train_points)
        tx = _array(state["train_x"].numpy(), "saved train_x", 2)
        obj._check_size(len(tx))
        obj.n_features = tx.shape[1]
        ty = _vector(state["train_y"].numpy(), "saved train_y", len(tx))
        noise = _vector(state["fixed_noise"].numpy(), "saved fixed_noise", len(tx))
        obj.input_mu = _vector(state["input_mu"].numpy(), "saved input_mu", obj.n_features)
        obj.input_sigma = _vector(state["input_sigma"].numpy(), "saved input_sigma", obj.n_features)
        obj.target_mu, obj.target_sigma = float(state["target_mu"]), float(state["target_sigma"])
        if (np.any(obj.input_sigma <= 0) or np.any(noise < FIXED_NOISE_FLOOR) or
                not np.isfinite([obj.target_mu, obj.target_sigma]).all() or obj.target_sigma <= 0):
            raise ValueError("Invalid scaler or noise in gp-torch checkpoint")
        obj.input_train, obj.target_train, obj.fixed_noise = obj._tensor(tx), obj._tensor(ty), obj._tensor(noise)
        obj.feature_names = names
        obj.provenance = state["provenance"]
        obj.loaded_software = state["software"]
        obj.epochs = int(state["epochs"])
        obj.loss_history = list(state["loss_history"])
        obj.prediction_batch_size = int(state["prediction_batch_size"] if prediction_batch_size is None else prediction_batch_size)
        if obj.prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        obj.gp_init()
        obj.model.load_state_dict(state["model_state"], strict=True)
        opt = state["optimizer_options"]
        obj.optim_init(opt["lr"], opt["betas"], opt["eps"], opt["weight_decay"])
        obj.model.eval()
        obj.likelihood.eval()
        return obj
