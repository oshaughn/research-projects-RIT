"""Exact bounded-batch means from an already fitted sklearn Matérn GP.

This is a prediction adapter, never a fitter. CuPy is imported and GPU memory
allocated only on the first nonempty prediction. Cross-covariance with a
WhiteKernel is zero, including queries equal to training rows. Target and
feature normalization are frozen at training time; batches are never centered.
"""
import numpy as np


class CachedMaternMean:
    """Predict-compatible single-target Matérn-5/2 mean, CPU or CUDA float64.

    Use :func:`from_sklearn` to validate a fitted GP or two-step
    StandardScaler/GaussianProcessRegressor pipeline. Predictive covariance and
    standard deviation are deliberately unsupported. Training noise is already
    represented in the learned ``alpha_``; it must not be added at prediction.
    """

    def __init__(self, *, x_train, alpha, feature_mean, feature_scale,
                 length_scale, constant, target_mean, target_scale,
                 backend="cupy", batch_size=4096):
        if backend not in ("numpy", "cupy"):
            raise ValueError("backend must be numpy or cupy")
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.backend, self.batch_size = backend, int(batch_size)
        self.x_train = np.array(x_train, dtype=np.float64, copy=True)
        if self.x_train.ndim != 2 or not self.x_train.shape[0] or not self.x_train.shape[1]:
            raise ValueError("x_train must be a nonempty two-dimensional array")
        self.n_features_in_ = self.x_train.shape[1]
        self.alpha = np.array(alpha, dtype=np.float64, copy=True)
        if self.alpha.shape != (len(self.x_train),):
            raise ValueError("only one-dimensional single-target alpha is supported")
        self.feature_mean = self._vector(feature_mean, "feature_mean")
        self.feature_scale = self._vector(feature_scale, "feature_scale", positive=True)
        self.length_scale = self._vector(length_scale, "length_scale", positive=True)
        self.constant = self._scalar(constant, "constant", positive=True)
        self.target_mean = self._scalar(target_mean, "target_mean")
        self.target_scale = self._scalar(target_scale, "target_scale", positive=True)
        if not np.all(np.isfinite(self.x_train)) or not np.all(np.isfinite(self.alpha)):
            raise ValueError("training coordinates and alpha must be finite")
        self._device_state = None

    def _vector(self, value, name, positive=False):
        v = np.asarray(value, dtype=np.float64)
        if v.ndim == 0:
            v = np.full(self.n_features_in_, v, dtype=np.float64)
        if v.shape != (self.n_features_in_,) or not np.all(np.isfinite(v)):
            raise ValueError(name + " must be scalar or a finite feature vector")
        if positive and np.any(v <= 0):
            raise ValueError(name + " must be positive")
        return v.copy()

    @staticmethod
    def _scalar(value, name, positive=False):
        v = np.asarray(value, dtype=np.float64)
        if v.size != 1 or not np.all(np.isfinite(v)):
            raise ValueError(name + " must be a finite single-target scalar")
        out = float(v.reshape(-1)[0])
        if positive and out <= 0:
            raise ValueError(name + " must be positive")
        return out

    def _state(self):
        if self._device_state is None:
            if self.backend == "cupy":
                import cupy as xp
                if xp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("CuPy prediction requires an actual CUDA device")
            else:
                xp = np
            train = xp.asarray(self.x_train) / xp.asarray(self.length_scale)
            self._device_state = (xp, train, xp.sum(train * train, axis=1),
                                  xp.asarray(self.alpha), xp.asarray(self.feature_mean),
                                  xp.asarray(self.feature_scale), xp.asarray(self.length_scale))
        return self._device_state

    def predict(self, x, return_std=False, return_cov=False):
        if return_std or return_cov:
            raise NotImplementedError("cached adapter computes means only")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.all(np.isfinite(x)):
            raise ValueError("queries must be a finite (rows, n_features) array")
        if len(x) == 0:
            return np.empty(0, dtype=np.float64)
        xp, train, norms, alpha, mean, scale, lengths = self._state()
        result = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), self.batch_size):
            stop = min(start + self.batch_size, len(x))
            query = (xp.asarray(x[start:stop]) - mean) / scale / lengths
            squared = xp.sum(query * query, axis=1)[:, None] + norms[None, :] - 2 * (query @ train.T)
            xp.maximum(squared, 0, out=squared)
            scaled_r = np.sqrt(5.0) * xp.sqrt(squared)
            kernel = self.constant * (1 + scaled_r + (5.0 / 3.0) * squared) * xp.exp(-scaled_r)
            prediction = (kernel @ alpha) * self.target_scale + self.target_mean
            result[start:stop] = xp.asnumpy(prediction) if self.backend == "cupy" else prediction
        return result


def from_sklearn(model, *, backend="cupy", batch_size=4096):
    """Freeze a supported fitted sklearn GP without refitting or changing it.

    Accepts a fitted GaussianProcessRegressor directly, or exactly a fitted
    StandardScaler followed by a GaussianProcessRegressor. Supported kernel is
    ConstantKernel * Matern(nu=2.5), optionally plus one WhiteKernel. Unsupported
    transformers, kernels, targets or missing fitted state fail explicitly.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, Product, Sum, WhiteKernel
    scaler = None
    gp = model
    if isinstance(model, Pipeline):
        if len(model.steps) != 2 or not isinstance(model.steps[0][1], StandardScaler):
            raise ValueError("only StandardScaler/GP two-step pipelines are supported")
        scaler, gp = model.steps[0][1], model.steps[1][1]
    if not isinstance(gp, GaussianProcessRegressor):
        raise TypeError("expected a fitted sklearn GaussianProcessRegressor")
    for attr in ("kernel_", "X_train_", "alpha_", "_y_train_mean", "_y_train_std"):
        if not hasattr(gp, attr):
            raise ValueError("GP is not fitted: missing " + attr)
    kernel = gp.kernel_
    if isinstance(kernel, Sum):
        if isinstance(kernel.k1, WhiteKernel):
            kernel = kernel.k2
        elif isinstance(kernel.k2, WhiteKernel):
            kernel = kernel.k1
        else:
            raise ValueError("only a single optional additive WhiteKernel is supported")
    if not isinstance(kernel, Product):
        raise ValueError("expected ConstantKernel * Matern")
    constant, matern = kernel.k1, kernel.k2
    if isinstance(constant, Matern) and isinstance(matern, ConstantKernel):
        constant, matern = matern, constant
    if not isinstance(constant, ConstantKernel) or not isinstance(matern, Matern) or matern.nu != 2.5:
        raise ValueError("only ConstantKernel * Matérn(nu=2.5) is supported")
    features = np.asarray(gp.X_train_).shape[1]
    mean, scale = np.zeros(features), np.ones(features)
    if scaler is not None:
        if not hasattr(scaler, "n_features_in_") or scaler.n_features_in_ != features:
            raise ValueError("scaler is unfitted or has different feature dimension")
        if scaler.with_mean:
            mean = scaler.mean_
        if scaler.with_std:
            scale = scaler.scale_
    return CachedMaternMean(x_train=gp.X_train_, alpha=gp.alpha_, feature_mean=mean,
                            feature_scale=scale, length_scale=matern.length_scale,
                            constant=constant.constant_value, target_mean=gp._y_train_mean,
                            target_scale=gp._y_train_std, backend=backend, batch_size=batch_size)
