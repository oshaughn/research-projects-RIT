"""Gaussian-process fit with a LINEAR mean function and a real posterior std.

Why this exists (and why the mean function is not zero)
------------------------------------------------------
The production default fit is the random forest (`_rf.py`). A forest is
piecewise-constant: outside the convex hull of the training points every tree
returns its boundary leaf, so the surrogate is exactly FLAT there
(`smooth_gradient = False`). When the lnL peak is clipped against a box edge --
the grid was drawn too narrow and the likelihood is still rising as it leaves
the sampled region -- a flat surrogate gives placement nothing to chase, and
the next iteration re-piles points on the wall.

A zero-mean GP is no better: away from data it relaxes to its prior mean, so
the surrogate falls back to 0 instead of following the trend. That failure mode
is already documented in CIP's own `--lnL-shift-prevent-overflow` help text
("If you shift the result to be below zero, because the GP relaxes to 0, you
will get crazy answers").

This fit therefore uses a LINEAR mean function: the GP kernel explains local
structure inside the sampled region, while the fitted hyperplane carries the
global trend outward. Extrapolation past the training hull follows that trend
rather than flattening, so UCB / SMC placement can chase a peak that lies
outside the region sampled so far. `mean="const"` is available for the
conservative behaviour (revert to a flat prior away from data); it is the
right choice when the trend is not believed and you would rather explore by
posterior variance alone.

The GP also supplies a calibrated posterior variance, so this is the fit the
UCB sampler asks for (`_base.FitBase.predict_with_std`): sigma is small where
data constrains the surface and grows to the signal amplitude out in the
unsampled frontier.

Ported from the R3 kilonova-placement study (`placement/propose_gp_resample.py`,
class `LinearMeanGP`), where the same construction was introduced to recover a
lnL peak clipped at the v_outer box edge. numpy-only: no sklearn or scipy
dependency, so this fit is usable in the same minimal environments the rest of
the tracer engine runs in.

Cost is the usual dense-GP O(n^3) factorization / O(n^2) memory in the number
of training points, which is fine at the tracer's design size (10^2 - 10^3
points per iteration) but is NOT a drop-in replacement for the forest on very
large unions; a warning is emitted past `_N_WARN`.
"""
import sys

import numpy as np

from ._base import FitBase

# Above this training-set size the dense Cholesky starts to dominate the
# per-iteration cost of the placement tool; warn rather than refuse.
_N_WARN = 2000


def _sqdist(A, B):
    """Pairwise squared Euclidean distance |a|^2 + |b|^2 - 2 a.b.

    Written as three 2-D products rather than a (len(A), len(B), d) broadcast
    so the candidate pools UCB hands us (~2e4 rows) stay in cache.
    """
    d2 = (np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :]
          - 2.0 * (A @ B.T))
    return np.clip(d2, 0.0, None)


class LinearMeanGPFit(FitBase):
    """RBF-kernel GP with a linear (or constant) mean, fit in a standardized basis.

    Parameters
    ----------
    X : (n, d) array
        Training coordinates, in the sampler's coordinate basis.
    Y : (n,) array
        Training lnL values. Must be finite -- see `--tracer-lnl-floor-delta`
        (fits.build's `lnl_floor_delta`) for the supported way to tame
        catastrophic-fit outliers, which also maps -inf onto the floor.
    sigma : (n,) array, optional
        Per-point lnL uncertainty, used as heteroscedastic observation noise.
        `None` means "use `sigma_floor` everywhere" (a small nugget).
    length_scale : float, optional
        RBF length scale in the standardized basis. Default: median pairwise
        distance / sqrt(2), the usual scale-free heuristic.
    mean : {"linear", "const"}
        Mean function. "linear" extrapolates the global trend past the data
        edge (chases a clipped peak, but bets on the trend continuing);
        "const" reverts to a flat prior away from data (conservative).
    sigma_floor : float
        Observation-noise floor, in lnL units. lnL uncertainties below this are
        not meaningful in RIFT and drive the kernel matrix towards singularity.
    jitter : float
        Initial diagonal jitter added before the Cholesky. Escalated by
        factors of 10 if the factorization fails.
    """

    has_uncertainty = True
    smooth_gradient = True

    def __init__(self, X, Y, sigma=None, length_scale=None, mean="linear",
                 sigma_floor=1e-2, jitter=1e-8):
        if mean not in ("linear", "const"):
            raise ValueError(f"LinearMeanGPFit: mean must be 'linear' or "
                             f"'const', got {mean!r}")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float).ravel()
        if len(X) != len(Y):
            raise ValueError(f"LinearMeanGPFit: X has {len(X)} rows but Y has "
                             f"{len(Y)} entries")
        if len(X) < 2:
            raise ValueError("LinearMeanGPFit: need at least 2 training points")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError(
                "LinearMeanGPFit: non-finite value in X or Y. Catastrophic-fit "
                "lnL outliers should be tamed with fits.build(..., "
                "lnl_floor_delta=...) (--tracer-lnl-floor-delta), which clamps "
                "them to max(lnL) - delta instead of discarding them.")
        n, self.d = X.shape

        if n > _N_WARN:
            sys.stderr.write(
                f"fits._gp_linmean: fitting a dense GP to {n} points "
                f"(O(n^3) factorization, O(n^2) memory). Consider "
                f"--tracer-fit-method rf for large unions.\n")

        # --- standardized basis: makes one isotropic length scale defensible
        self._mu_x = X.mean(axis=0)
        self._sd_x = X.std(axis=0)
        self._sd_x[self._sd_x == 0] = 1.0
        Xs = (X - self._mu_x) / self._sd_x

        # --- mean function
        A = np.column_stack([np.ones(n), Xs])
        self._beta = np.linalg.lstsq(A, Y, rcond=None)[0]
        if mean == "const":
            self._beta = np.zeros_like(self._beta)
            self._beta[0] = float(Y.mean())
        self.mean_kind = mean
        resid = Y - A @ self._beta

        # --- kernel hyperparameters
        d2 = _sqdist(Xs, Xs)
        if length_scale is None:
            iu = np.triu_indices(n, 1)
            med = float(np.median(np.sqrt(d2[iu]))) if len(iu[0]) else 1.0
            length_scale = max(med / np.sqrt(2.0), 1e-2)
        self.length_scale = float(length_scale)
        # Signal variance is the residual scatter about the mean function. This
        # is exactly where a lnL FLOOR beats a lnL CUT: floored known-bad points
        # stay in the fit as anchors and keep sf2 (and the length scale) honest,
        # where cutting them throws that geometry away.
        self.sf2 = max(float(np.var(resid)), 1e-6)

        if sigma is None:
            noise_var = np.full(n, sigma_floor ** 2)
        else:
            s = np.asarray(sigma, dtype=float).ravel()
            s = np.where(np.isfinite(s), s, sigma_floor)
            noise_var = np.maximum(s, sigma_floor) ** 2

        # --- Cholesky, with escalating jitter on failure
        K0 = self.sf2 * np.exp(-0.5 * d2 / self.length_scale ** 2)
        self._L = None
        for k in range(6):
            K = K0.copy()
            K[np.diag_indices_from(K)] += noise_var + jitter * (10.0 ** k)
            try:
                self._L = np.linalg.cholesky(K)
                break
            except np.linalg.LinAlgError:
                continue
        if self._L is None:
            raise np.linalg.LinAlgError(
                "LinearMeanGPFit: kernel matrix not positive-definite even "
                f"with jitter {jitter * 1e5:g}; check for duplicate training "
                "points or a degenerate coordinate.")

        # Form L^{-1} once. Every predict_with_std / grad call then costs
        # O(m n^2) instead of re-solving (and re-factorizing) per call, which
        # matters because samplers.ucb polishes each selected point one at a
        # time. This is the same Cholesky solve, just staged.
        self._Linv = np.linalg.solve(self._L, np.eye(n))
        self._alpha = self._Linv.T @ (self._Linv @ resid)
        self._Xs = Xs

        self.train_rms = float(np.sqrt(np.mean((self.predict(X) - Y) ** 2)))

    # ------------------------------------------------------------------ #

    def _standardize(self, Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        return (Z - self._mu_x) / self._sd_x

    def _kstar(self, Zs):
        return self.sf2 * np.exp(-0.5 * _sqdist(Zs, self._Xs)
                                 / self.length_scale ** 2)

    def _mean_from(self, Zs, ks):
        return (self._beta[0] + Zs @ self._beta[1:]) + ks @ self._alpha

    def predict(self, Z):
        Zs = self._standardize(Z)
        return self._mean_from(Zs, self._kstar(Zs))

    def predict_with_std(self, Z):
        """Return (mean, std): the GP posterior mean and standard deviation.

        std -> ~0 at well-constrained training points and -> sqrt(sf2) far from
        any data, which is the behaviour samplers.ucb needs from
        `mu + kappa * sigma`.
        """
        Zs = self._standardize(Z)
        mean = np.empty(len(Zs))
        var = np.empty(len(Zs))
        # Chunked so the (n, chunk) intermediate stays bounded for the ~2e4
        # candidate pools UCB evaluates in one shot.
        chunk = 2048
        for i0 in range(0, len(Zs), chunk):
            Zc = Zs[i0:i0 + chunk]
            ks = self._kstar(Zc)
            mean[i0:i0 + chunk] = self._mean_from(Zc, ks)
            v = self._Linv @ ks.T
            var[i0:i0 + chunk] = self.sf2 - np.sum(v * v, axis=0)
        return mean, np.sqrt(np.maximum(var, 1e-12))

    def grad(self, Z, eps=None):
        """Analytic gradient of the posterior mean (eps is ignored)."""
        Zs = self._standardize(Z)
        ks = self._kstar(Zs)
        # d/dZs_j [ks @ alpha] = -(1/ls^2) sum_i alpha_i ks_ij (Zs_j - Xs_ij)
        Aa = ks * self._alpha[None, :]
        term = Zs * Aa.sum(axis=1)[:, None] - Aa @ self._Xs
        g = self._beta[1:][None, :] - term / self.length_scale ** 2
        return g / self._sd_x
