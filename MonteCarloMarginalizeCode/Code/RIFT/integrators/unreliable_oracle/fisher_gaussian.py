"""
fisher_gaussian.py

A Fisher-matrix / Gaussian oracle for the RIFT portfolio integrator.

The paper's remark is that a normalizing-flow oracle, trained per-inference, in
effect just supplies a fast local approximation to the posterior -- and that a
Fisher matrix gives the same thing essentially for free.  This oracle is that
cheap substitute: given a mean and a covariance (or a Fisher matrix Gamma, cov =
Gamma^{-1}), it proposes points from the box-truncated Gaussian N(mean, cov), and
can optionally refresh (mean, cov) from the weighted history it is shown.

Oracles only PROPOSE points; the portfolio evaluates the true likelihood at them
and folds them into the training data for the other integrators.  A proposal can
therefore never bias the integral -- at worst it wastes a few evaluations -- so
an approximate Fisher is a safe, robust way to inject known posterior shape.

Interface matches MCSamplerGeneric: setup(), update_sampling_prior(), draw_simplified().
Backend-agnostic: all math is host numpy (proposals are cheap and small); the
portfolio moves arrays to the active backend as needed.
"""
import numpy as np

from RIFT.integrators.mcsampler_generic import MCSamplerGeneric


class FisherGaussianOracle(MCSamplerGeneric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reference_mean = None
        self.reference_cov = None
        self._chol = None
        # if True, refit (mean,cov) from weighted history on each update
        self.adapt = False
        # inflate the covariance when proposing, to stay broad (avoid collapse)
        self.cov_inflate = 1.0

    def add_parameter(self, params, pdf, **kwargs):
        super().add_parameter(params, pdf, **kwargs)

    def setup(self, mean=None, cov=None, fisher=None, adapt=False,
              cov_inflate=1.0, params=None, **kwargs):
        """mean/cov describe the proposal; alternatively pass a Fisher matrix
        (cov = fisher^{-1}).  `params` names the columns of mean/cov if they are
        not already in this oracle's params_ordered order.  adapt=True refreshes
        (mean,cov) from weighted history on each update_sampling_prior call."""
        super().setup(**kwargs)
        self.adapt = adapt
        self.cov_inflate = float(cov_inflate)
        if cov is None and fisher is not None:
            cov = np.linalg.inv(np.atleast_2d(np.asarray(fisher, dtype=float)))
        if mean is not None and cov is not None:
            mean = np.asarray(mean, dtype=float)
            cov = np.atleast_2d(np.asarray(cov, dtype=float))
            if params is not None and self.params_ordered:
                order = [list(params).index(p) for p in self.params_ordered]
                mean = mean[order]
                cov = cov[np.ix_(order, order)]
            self._set_gaussian(mean, cov)

    def _set_gaussian(self, mean, cov):
        cov = 0.5 * (cov + cov.T)
        # regularize to SPD
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 1e-12 * max(np.max(w), 1e-300), None)
        cov = (V * w) @ V.T
        self.reference_mean = mean
        self.reference_cov = cov * self.cov_inflate
        self._chol = np.linalg.cholesky(self.reference_cov)

    def update_sampling_prior(self, ln_weights, n_history, lnw_cut=-10,
                              external_rvs=None, verbose=False, **kwargs):
        """Optionally refit (mean, cov) from the recent weighted history.  If
        adapt is False (default), the oracle keeps the supplied Fisher/Gaussian
        and this is a no-op -- the robust behaviour when a trustworthy Fisher is
        available.  With adapt=True it becomes a self-refining Gaussian proposal."""
        if not self.adapt:
            return
        rvs_here = external_rvs if external_rvs else self._rvs
        if rvs_here is None or len(self.params_ordered) == 0:
            return
        p0 = self.params_ordered[0]
        n_avail = len(rvs_here[p0])
        n_use = int(min(n_history, n_avail))
        if ln_weights is not None:
            n_use = int(min(n_use, len(ln_weights)))
        if n_use < len(self.params_ordered) + 2:
            return
        X = np.empty((n_use, len(self.params_ordered)))
        for j, p in enumerate(self.params_ordered):
            X[:, j] = np.asarray(rvs_here[p])[-n_use:]
        w = None
        if ln_weights is not None:
            lw = np.asarray(ln_weights)[-n_use:].astype(float)
            lw = lw - np.max(lw)
            if lnw_cut is not None:
                keep = lw > lnw_cut
                if np.sum(keep) >= len(self.params_ordered) + 2:
                    X = X[keep]
                    lw = lw[keep]
            w = np.exp(lw)
            w = w / np.sum(w)
        mean = np.average(X, axis=0, weights=w)
        cov = np.cov(X.T, aweights=w)
        cov = np.atleast_2d(cov)
        self._set_gaussian(mean, cov)
        if verbose:
            print(" oracle - fisher_gaussian - refit mean", mean)

    def _bounds(self):
        lo = np.array([self.llim[p] for p in self.params_ordered], dtype=float)
        hi = np.array([self.rlim[p] for p in self.params_ordered], dtype=float)
        return lo, hi

    def draw_simplified(self, n_samples, *args, **kwargs):
        """Draw n_samples from the box-truncated Gaussian.  Returns
        (p_s, p_prior, rv) with rv shape (n_samples, ndim); p_s is the Gaussian
        proposal density (so the draws can be used with correct importance
        weights if desired), p_prior is left None (the portfolio supplies it)."""
        if self.reference_mean is None:
            raise Exception("FisherGaussianOracle: setup(mean=,cov=/fisher=) required before draw")
        d = len(self.params_ordered)
        lo, hi = self._bounds()
        out = np.empty((n_samples, d))
        logdens = np.empty(n_samples)
        n_out = 0
        cov = self.reference_cov
        inv = np.linalg.inv(cov)
        logdet = np.linalg.slogdet(cov)[1]
        lognorm = -0.5 * (d * np.log(2 * np.pi) + logdet)
        guard = 0
        while n_out < n_samples and guard < 1000:
            guard += 1
            batch = np.random.multivariate_normal(self.reference_mean, cov, size=2 * n_samples)
            inside = np.all((batch >= lo) & (batch <= hi), axis=1)
            batch = batch[inside]
            if len(batch) == 0:
                continue
            take = min(len(batch), n_samples - n_out)
            b = batch[:take]
            out[n_out:n_out + take] = b
            dx = b - self.reference_mean
            logdens[n_out:n_out + take] = lognorm - 0.5 * np.einsum('ij,jk,ik->i', dx, inv, dx)
            n_out += take
        if n_out < n_samples:
            # fall back to filling the remainder uniformly in-box (keeps coverage)
            rem = n_samples - n_out
            out[n_out:] = np.random.uniform(lo, hi, size=(rem, d))
            logdens[n_out:] = -np.sum(np.log(hi - lo))
        return np.exp(logdens), None, out
