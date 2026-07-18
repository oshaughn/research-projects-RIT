#!/usr/bin/env python
"""
benchmark_integrators.py

A reusable, quantitative benchmark harness for RIFT Monte-Carlo integrators.

Goal
----
Provide a *testable* definition of "a better integrator", matching the existing
RIFT sampler API (add_parameter / setup / integrate[_log]), on analytic targets
with known truth.  Supports:

  * multiple analytic targets with known integral & known 1-D marginals:
        - CorrelatedGaussian(ndim)     (the CI 3-D target, generalized to any D)
        - Rosenbrock2D                 (true log-evidence -5.804)
        - GaussianMixture(ndim,ncomp)  (the FinerNet multigauss high-D stress test)
  * a uniform adapter over the heterogeneous sampler call/return conventions
        (default mcsampler, AC/mcsamplerGPU, GMM/mcsamplerEnsemble, AV, NF, portfolio)
  * the paper's quality metrics:
        - integral bias      ln(I_hat) - ln(Z_true)
        - fractional MC error   sqrt(var)/I
        - n_eff  (RIFT: sum p / max p)   and   n_ESS = (sum w)^2 / sum w^2  (Kish)
        - EFFICIENCY  eff = n_eff / N_eval        <-- headline scaling metric
        - N_eval and wallclock consumed to reach a target n_eff
        - Jensen-Shannon divergence of recovered vs true 1-D marginals (nats)
  * cold-vs-warm comparison: an optional `warm_start` hook that seeds a sampler
    with prior information before integrate() (used by the bootstrappable-AV work).

Backend (CPU vs GPU / xpy) is selected by the *caller's* environment exactly as
in production ILE:  set CUDA_VISIBLE_DEVICES="" for CPU/numpy, or to an idle GPU
index for cupy.  The harness records which backend each sampler actually used.

This file is import-safe (no side effects) and has a CLI at the bottom.
"""
from __future__ import print_function

import sys
import time
import json

import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.special import logsumexp, erf


# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------
class Target(object):
    """Analytic integrand with known truth.

    Contract:
      name          : str
      ndim          : int
      params        : list[str]          (ordered parameter names)
      llim, rlim    : arrays length ndim (integration box)
      lnL(X)        : X is (N, ndim) -> (N,) log-integrand (the 'likelihood')
      true_lnZ      : ln of the true integral of exp(lnL) over the box
      true_marginal_pdf(dim, x) : analytic 1-D marginal density of the
                                  *normalized posterior* (integrand/Z), or None
    """
    name = "abstract"

    def lnL(self, X):
        raise NotImplementedError

    # convenience: a callable of positional scalars/arrays, matching RIFT's
    # `no_protect_names=True` integrand signature f(x0, x1, ...).
    def as_lnfunc(self):
        def ln_f(*cols):
            X = np.array(cols, dtype=float).T
            return self.lnL(np.atleast_2d(X))
        return ln_f

    def as_func(self):
        ln_f = self.as_lnfunc()
        def f(*cols):
            return np.exp(ln_f(*cols))
        return f

    def true_marginal_pdf(self, dim, x):
        return None


class CorrelatedGaussian(Target):
    """Scaled multivariate normal with one narrow dimension and off-diagonal
    correlation.  Generalizes the CI test (test_mcsamplerEnsemble_extended.py)
    to arbitrary dimension.  With mu in the middle of the box and small widths,
    the box captures essentially all the mass, so the true integral over the box
    equals `scale` (the Gaussian integrates to 1)."""
    def __init__(self, ndim=3, width=10.0, scale=100.0, seed=123456, rho=-0.1,
                 narrow=0.05):
        self.name = "corrgauss_d{}".format(ndim)
        self.ndim = ndim
        self.width = width
        self.scale = scale
        self.params = [str(i) for i in range(ndim)]
        self.llim = -0.5 * width * np.ones(ndim)
        self.rlim = 0.5 * width * np.ones(ndim)
        rng = np.random.RandomState(seed)
        self.mu = rng.uniform(-width / 4.0, width / 4.0, ndim)
        cov = np.identity(ndim)
        cov[ndim - 1][ndim - 1] = narrow            # one narrow dimension
        cov[0][ndim - 1] = rho                       # a correlation
        cov[ndim - 1][0] = rho
        # keep SPD in high-D: shrink off-diagonals if needed
        while np.min(np.linalg.eigvalsh(cov)) <= 1e-6:
            cov[0][ndim - 1] *= 0.5
            cov[ndim - 1][0] *= 0.5
        self.cov = cov
        self._mvn = multivariate_normal(self.mu, self.cov)
        # The samplers return  I = \int L * p_prior dx , with p_prior the
        # normalized uniform density (1/width per dim).  Over the box the
        # Gaussian integrates to `scale`, so the returned quantity's truth is
        # ln(scale) minus the uniform-prior log-normalization.
        self.true_lnZ = np.log(scale) - np.sum(np.log(self.rlim - self.llim))

    def lnL(self, X):
        return np.log(self.scale * self._mvn.pdf(X) + 1e-300)

    def true_marginal_pdf(self, dim, x):
        return norm.pdf(x, loc=self.mu[dim], scale=np.sqrt(self.cov[dim][dim]))


class Rosenbrock2D(Target):
    """2-D Rosenbrock likelihood distributed with RIFT.  Known true log-evidence
    over the box [-5,5]^2 is -5.804 (see FinerNet neff_linear demo)."""
    def __init__(self, box=5.0, lnL_offset=0.0):
        self.name = "rosenbrock2d"
        self.ndim = 2
        self.params = ["0", "1"]
        self.llim = -box * np.ones(2)
        self.rlim = box * np.ones(2)
        self.lnL_offset = lnL_offset
        self.true_lnZ = -5.804 + lnL_offset

    def lnL(self, X):
        x1 = X[:, 0]; x2 = X[:, 1]
        minus = (1.0 - x1) ** 2 + 100.0 * (x2 - x1 ** 2) ** 2
        return self.lnL_offset - minus

    def true_marginal_pdf(self, dim, x):
        if dim == 0:
            # exact 1-D marginal in x1 (unnormalized then normalized by exp(true_lnZ))
            box = self.rlim[1]
            m = (1.0 / 20.0) * np.sqrt(np.pi) * (erf(10 * (box - x ** 2)) +
                                                 erf(10 * (box + x ** 2))) * np.exp(-(1 - x) ** 2)
            return m / np.exp(self.true_lnZ - self.lnL_offset)
        return None


class GaussianMixture(Target):
    """Superposition of `ncomp` multivariate normals with random weights, means
    and Wishart-drawn covariances (the FinerNet multigauss stress test that
    exposes AV's high-dimensional degradation).  Normalized so the true integral
    over the box equals `scale`."""
    def __init__(self, ndim=4, ncomp=3, width=10.0, scale=100.0, seed=42,
                 sigma_1d=0.7, scale_x0=3.0):
        self.name = "gaussmix_d{}_n{}".format(ndim, ncomp)
        self.ndim = ndim
        self.ncomp = ncomp
        self.width = width
        self.scale = scale
        self.params = [str(i) for i in range(ndim)]
        self.llim = -0.5 * width * np.ones(ndim)
        self.rlim = 0.5 * width * np.ones(ndim)
        import scipy.stats as ss
        rng = np.random.RandomState(seed)
        wt = rng.uniform(size=ncomp) + 0.1
        self.wt = wt / np.sum(wt)
        self.means = []
        self.covs = []
        self._rvs = []
        for k in range(ncomp):
            x0 = rng.uniform(-scale_x0 / np.sqrt(ndim), scale_x0 / np.sqrt(ndim), ndim)
            Sig = (sigma_1d ** 2) * np.diag(rng.uniform(1.0, 2.0, ndim))
            Sig = ss.wishart.rvs(df=ndim, scale=Sig / ndim, random_state=rng) / 1.25
            Sig = np.atleast_2d(Sig)
            self.means.append(x0)
            self.covs.append(Sig)
            self._rvs.append(multivariate_normal(x0, Sig, allow_singular=True))
        # returned I = \int L p_prior dx ; mixture integrates to `scale` over box
        self.true_lnZ = np.log(scale) - np.sum(np.log(self.rlim - self.llim))

    def lnL(self, X):
        val = np.zeros(len(X))
        for k in range(self.ncomp):
            val += self.wt[k] * self._rvs[k].pdf(X)
        return np.log(self.scale * val + 1e-300)

    def true_marginal_pdf(self, dim, x):
        p = np.zeros_like(x, dtype=float)
        for k in range(self.ncomp):
            p += self.wt[k] * norm.pdf(x, loc=self.means[k][dim],
                                       scale=np.sqrt(self.covs[k][dim][dim]))
        return p

    def sample_truth(self, n, seed=0):
        rng = np.random.RandomState(seed)
        counts = rng.multinomial(n, self.wt)
        out = []
        for k in range(self.ncomp):
            out.append(rng.multivariate_normal(self.means[k], self.covs[k], counts[k]))
        return np.vstack(out)


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
def _asnumpy(a):
    try:
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def log_weights_from_rvs(rvs):
    """Return per-sample ln(weight) = ln L + ln p_prior - ln p_sampling, from a
    sampler's _rvs cache, tolerating the heterogeneous storage conventions:
    log-keyed (AV/NF/portfolio) vs linear-keyed (default/AC), and GMM which
    stores 'integrand' as *log* L (negative values) alongside linear priors."""
    # integrand -> ln L
    if "log_integrand" in rvs:
        lnL = _asnumpy(rvs["log_integrand"]).astype(float)
    elif "integrand" in rvs:
        L = _asnumpy(rvs["integrand"]).astype(float)
        # a genuine integrand/density is >=0; negative values mean it is
        # already stored as ln L (GMM under return_lnI).
        lnL = L if np.nanmin(L) < 0 else np.log(L + 1e-300)
    else:
        raise KeyError("no integrand in _rvs; run with save_intg=True")
    # prior and sampling prior, each possibly log- or linear-keyed
    def _get_log(logkey, linkey, n):
        if logkey in rvs:
            return _asnumpy(rvs[logkey]).astype(float)
        if linkey in rvs:
            return np.log(_asnumpy(rvs[linkey]).astype(float) + 1e-300)
        return np.zeros(n)
    n = len(lnL)
    lnp = _get_log("log_joint_prior", "joint_prior", n)
    lnps = _get_log("log_joint_s_prior", "joint_s_prior", n)
    return lnL + lnp - lnps


def n_ess_kish(ln_wt):
    ln_wt = ln_wt - np.max(ln_wt)
    w = np.exp(ln_wt)
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def marginal_js(target, rvs, ln_wt, dim, nbins=60):
    """Jensen-Shannon divergence (nats) between the weighted-sample 1-D marginal
    and the analytic true marginal, over the box for `dim`."""
    tp = target.true_marginal_pdf(dim, np.array([0.0]))
    if tp is None:
        return float("nan")
    name = target.params[dim]
    x = _asnumpy(rvs[name]).astype(float).flatten()
    w = np.exp(ln_wt - np.max(ln_wt))
    lo, hi = target.llim[dim], target.rlim[dim]
    edges = np.linspace(lo, hi, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist, _ = np.histogram(x, bins=edges, weights=w)
    if hist.sum() <= 0:
        return float("nan")
    q = hist / hist.sum()
    tpdf = target.true_marginal_pdf(dim, centers)
    p = tpdf / tpdf.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / (b[mask] + 1e-300)))
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


# ----------------------------------------------------------------------------
# Sampler adapter
# ----------------------------------------------------------------------------
def build_sampler(kind, target, n_chunk=10000):
    """Instantiate + add_parameter a sampler of the requested kind for `target`.
    Returns (sampler, backend_str)."""
    from RIFT.integrators import (mcsampler, mcsamplerEnsemble, mcsamplerGPU,
                                  mcsamplerAdaptiveVolume)

    def uniform_pdf(d):
        w = target.rlim[d] - target.llim[d]
        return np.vectorize(lambda x, w=w: 1.0 / w)

    if kind == "default":
        s = mcsampler.MCSampler(); backend = "cpu"
    elif kind in ("AC", "adaptive_cartesian_gpu"):
        s = mcsamplerGPU.MCSampler(); backend = "gpu" if mcsamplerGPU.cupy_ok else "cpu"
    elif kind in ("GMM", "gmm"):
        s = mcsamplerEnsemble.MCSampler(); backend = "cpu"
    elif kind == "AV":
        s = mcsamplerAdaptiveVolume.MCSampler(n_chunk=n_chunk)
        backend = "gpu" if mcsamplerAdaptiveVolume.cupy_ok else "cpu"
    elif kind == "NF":
        from RIFT.integrators import mcsamplerNFlow
        s = mcsamplerNFlow.MCSampler(); backend = "cpu(torch)"
    else:
        raise ValueError("unknown sampler kind %r" % kind)

    for d, p in enumerate(target.params):
        s.add_parameter(p, uniform_pdf(d), prior_pdf=uniform_pdf(d),
                        left_limit=float(target.llim[d]), right_limit=float(target.rlim[d]),
                        adaptive_sampling=True)
    return s, backend


def run(kind, target, nmax=200000, neff=1000, n_chunk=10000, tempering_exp=0.1,
        n_adapt=100, warm_start=None, verbose=False, seed=None):
    """Run one sampler on one target and return a metrics dict.

    warm_start : optional callable(sampler, target) invoked after setup() and
                 before integrate(), used to seed prior information (cold-vs-warm).
    """
    if seed is not None:
        np.random.seed(seed)
    s, backend = build_sampler(kind, target, n_chunk=n_chunk)
    ln_f = target.as_lnfunc()
    f = target.as_func()
    params = target.params
    extra = dict(n=n_chunk, n_adapt=n_adapt, floor_level=0.0,
                 tempering_exp=tempering_exp, neff=neff, nmax=nmax,
                 save_intg=True, verbose=verbose)

    # setup + optional warm start
    if hasattr(s, "setup"):
        try:
            s.setup()
        except TypeError:
            pass
    if warm_start is not None:
        warm_start(s, target)

    t0 = time.time()
    if kind == "default":
        I, var, eff, _ = s.integrate(f, *params, no_protect_names=True, **extra)
        lnI = np.log(I); ln_relerr = np.log(np.sqrt(var) / I)
    elif kind in ("AC", "adaptive_cartesian_gpu"):
        lnI, logvar, eff, _ = s.integrate(ln_f, *params, no_protect_names=True,
                                          use_lnL=True, **extra)
        lnI = float(_asnumpy(lnI)); ln_relerr = float(_asnumpy(logvar)) / 2 - lnI
    elif kind in ("GMM", "gmm"):
        n_iters = int(nmax / n_chunk)
        lnI, logvar, eff, _ = s.integrate(ln_f, *params, min_iter=n_iters,
                                          max_iter=n_iters, correlate_all_dims=True,
                                          n_comp=1, use_lnL=True, return_lnI=True, **extra)
        lnI = float(_asnumpy(lnI)); ln_relerr = float(_asnumpy(logvar)) / 2 - lnI
    elif kind == "AV":
        lnI, logvar, eff, _ = s.integrate_log(ln_f, *params, no_protect_names=True, **extra)
        lnI = float(_asnumpy(lnI)); ln_relerr = 0.5 * (float(_asnumpy(logvar)) - 2 * lnI)
    elif kind == "NF":
        lnI, logvar, eff, _ = s.integrate_log(ln_f, *params, no_protect_names=True, **extra)
        lnI = float(_asnumpy(lnI)); ln_relerr = 0.5 * (float(_asnumpy(logvar)) - 2 * lnI)
    else:
        raise ValueError(kind)
    wall = time.time() - t0
    eff = float(_asnumpy(eff))
    n_eval = int(getattr(s, "ntotal", 0)) or int(nmax)

    # sample-based metrics
    ln_wt = log_weights_from_rvs(s._rvs)
    ness = n_ess_kish(ln_wt)
    js = []
    for d in range(target.ndim):
        js.append(marginal_js(target, s._rvs, ln_wt, d))
    js = [x for x in js if x == x]  # drop nan
    js_mean = float(np.mean(js)) if js else float("nan")

    return dict(
        kind=kind, target=target.name, backend=backend,
        ndim=target.ndim, n_eval=n_eval, wallclock=wall,
        lnI=lnI, true_lnZ=float(target.true_lnZ),
        bias_ln=lnI - float(target.true_lnZ),
        rel_err=float(np.exp(ln_relerr)),
        n_eff=eff, n_ess=ness,
        efficiency=eff / max(n_eval, 1),
        js_marginal=js_mean,
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
_TARGETS = {
    "corrgauss3": lambda: CorrelatedGaussian(ndim=3),
    "corrgauss5": lambda: CorrelatedGaussian(ndim=5),
    "corrgauss8": lambda: CorrelatedGaussian(ndim=8),
    "rosenbrock": lambda: Rosenbrock2D(),
    "gaussmix4": lambda: GaussianMixture(ndim=4, ncomp=3),
    "gaussmix8": lambda: GaussianMixture(ndim=8, ncomp=3),
}


def _fmt(r):
    return ("{kind:>8s} {target:>14s} [{backend:>7s}]  N={n_eval:>8d}  "
            "t={wallclock:6.1f}s  lnI-lnZ={bias_ln:+7.3f}  relerr={rel_err:7.4f}  "
            "neff={n_eff:9.1f}  nESS={n_ess:10.1f}  eff={efficiency:.2e}  "
            "JS={js_marginal:.4f}").format(**r)


def main():
    import optparse
    p = optparse.OptionParser()
    p.add_option("--target", default="corrgauss3")
    p.add_option("--samplers", default="default,AC,GMM,AV")
    p.add_option("--nmax", type=int, default=200000)
    p.add_option("--neff", type=int, default=1000)
    p.add_option("--n-chunk", type=int, default=10000)
    p.add_option("--seed", type=int, default=123456)
    p.add_option("--json", default=None, help="write results as JSON to this path")
    p.add_option("--verbose", action="store_true")
    opts, _ = p.parse_args()

    tgt = _TARGETS[opts.target]()
    print("# target: {}  ndim={}  true_lnZ={:.4f}".format(tgt.name, tgt.ndim, tgt.true_lnZ))
    results = []
    for kind in opts.samplers.split(","):
        kind = kind.strip()
        try:
            r = run(kind, tgt, nmax=opts.nmax, neff=opts.neff, n_chunk=opts.n_chunk,
                    verbose=opts.verbose, seed=opts.seed)
            results.append(r)
            print(_fmt(r))
        except Exception as e:
            import traceback
            print("  {:>8s}  FAILED: {}".format(kind, e))
            if opts.verbose:
                traceback.print_exc()
    if opts.json:
        with open(opts.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print("# wrote", opts.json)


if __name__ == "__main__":
    main()
