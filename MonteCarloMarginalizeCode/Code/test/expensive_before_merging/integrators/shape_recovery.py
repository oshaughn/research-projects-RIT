#!/usr/bin/env python
"""
shape_recovery.py -- posterior SHAPE-recovery merge gate for RIFT MC integrators.

Motivation
----------
The fast CI gate (.travis/test-integrate.sh -> test/test_mcsamplerEnsemble_extended.py)
checks only the *integral* on a single 3-D correlated Gaussian.  Integrals are
easy: importance-sampling estimators of Z are unbiased under very weak
conditions, while the recovered *posterior shape* (the weighted sample cloud
used downstream by CIP / fairdraws) can be subtly biased -- wrong widths,
clipped tails, missing mixture components, distorted correlations -- without
the integral moving outside its error bars.  This suite is the strong,
expensive check run before confirming a merge (see ../README.md).

Method (follows RIFT-FinerNet demos/integrators/multigauss_direct, Wagner et al):
  * Targets: random Gaussian mixtures over a range of dimensions.  Weights
    ~U(0.1,1.1) normalized, means uniform in a central sub-box, covariances
    Wishart-drawn (random orientation + condition number).  Mixture recipe is
    seeded, so every branch under test sees the *identical* targets.
  * Truth: 10^6 exact fair draws per target by rejection sampling inside the
    integration box (also yields the in-box mass for the true evidence).
  * Recovery: each integrator runs through its production API
    (add_parameter / setup / integrate[_log]) with save_intg=True; the weighted
    posterior cloud is read back from sampler._rvs exactly as ILE/CIP consume it.
  * Shape metrics, each dimension:
      - JS divergence (nats) of the weighted 1-D marginal histogram vs the
        truth-pool histogram;
      - mean pull  (weighted mean - true mean)/true sigma;
      - width ratio  weighted sigma / true sigma;
    plus max |Delta corr(i,j)| over dimension pairs, evidence bias
    lnI - lnZ_true, n_eff and Kish n_ESS.
  * Self-calibrating JS pass threshold: the sampling floor for each dim is
    measured by subsampling the truth pool down to the run's own n_ESS and
    computing JS(subsample, pool) -- i.e. the JS a *perfect* sampler with the
    same effective sample count would score.  PASS requires
        JS  <  JS_MULT * floor + JS_ABS_MIN.
    This keeps one threshold meaningful across samplers/dimensions/branches.

Policy
------
Strict (hard-fail) samplers default to AV + GMM; NF and portfolio default to
warn-only (they are known-weaker in older code lines, e.g. rift_O4c).  Override
with --strict-samplers / --samplers.  Exit code 1 iff any strict run fails.

Usage
-----
    # environment: any venv with numpy/scipy (torch+nflows only needed for NF),
    # PYTHONPATH pointing at the checkout under test:
    export PYTHONPATH=/path/to/checkout/MonteCarloMarginalizeCode/Code:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=""      # CPU: deterministic merge-gate default

    python shape_recovery.py --preset quick        # ~minutes, smoke
    python shape_recovery.py --preset standard --jobs 8 --json results.json

This file is self-contained on purpose: it must run unmodified against ANY
branch (including historical ones that lack test/integrators helpers).
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

# production-scale constant lnL offset (a modest-SNR detection), as in the
# FinerNet multigauss demo: keeps us honest about lnL-vs-L overflow handling.
LNL_OFFSET = 100.0
BOX_HALF_WIDTH = 5.0
TRUTH_POOL_N = 1000000
JS_NBINS = 50
JS_MULT = 3.0      # pass if JS < JS_MULT*floor + JS_ABS_MIN
JS_ABS_MIN = 0.004
MIN_NEFF_FOR_SHAPE = 100.0
NF_NMAX_CAP = 400000   # NF trains a flow per chunk; cap its budget (warn-only sampler)


# ----------------------------------------------------------------------------
# Target: seeded random Gaussian mixture with exact truth
# ----------------------------------------------------------------------------
class MixtureTarget(object):
    """Random `ncomp`-component Gaussian mixture in `ndim` dimensions on the
    box [-BOX_HALF_WIDTH, BOX_HALF_WIDTH]^ndim, FinerNet multigauss recipe."""

    def __init__(self, ndim, ncomp, seed, sigma_1d=0.7, scale_x0=3.0, offset=0.0):
        self.ndim = int(ndim)
        self.ncomp = int(ncomp)
        self.seed = int(seed)
        self.name = "mix_d{}_n{}_s{}".format(ndim, ncomp, seed)
        # `offset` TRANSLATES the whole mixture.  Two targets built with the same seed and
        # offset=-a / +a are the SAME shape displaced -- which is what the sequential cases need
        # and what different seeds cannot guarantee (random means typically overlap).  It is
        # applied AFTER the rng.uniform draw below, so the RNG stream and every existing target
        # are bit-identical at the default offset=0.
        self.offset = np.zeros(int(ndim)) + np.asarray(offset, dtype=float)
        if np.any(self.offset):
            self.name += "_o{:+.2f}".format(float(np.mean(self.offset)))
        self.params = ["x{}".format(i) for i in range(ndim)]
        self.llim = -BOX_HALF_WIDTH * np.ones(ndim)
        self.rlim = BOX_HALF_WIDTH * np.ones(ndim)
        rng = np.random.RandomState(seed)
        wt = rng.uniform(size=ncomp) + 0.1
        self.wt = wt / np.sum(wt)
        import scipy.stats as ss
        self.means, self.covs, self._mvns = [], [], []
        for k in range(ncomp):
            x0 = rng.uniform(-scale_x0 / np.sqrt(ndim), scale_x0 / np.sqrt(ndim), ndim) + self.offset
            Sig = (sigma_1d ** 2) * np.diag(rng.uniform(1.0, 2.0, ndim))
            Sig = ss.wishart.rvs(df=ndim, scale=Sig / ndim, random_state=rng) / 1.25
            Sig = np.atleast_2d(Sig)
            self.means.append(x0)
            self.covs.append(Sig)
            self._mvns.append(multivariate_normal(x0, Sig, allow_singular=True))
        self._pool = None
        self._box_mass = None

    def lnL(self, X):
        X = np.atleast_2d(X)
        terms = np.empty((len(X), self.ncomp))
        for k in range(self.ncomp):
            terms[:, k] = self._mvns[k].logpdf(X) + np.log(self.wt[k])
        return LNL_OFFSET + logsumexp(terms, axis=1)

    def as_lnfunc(self):
        def ln_f(*cols):
            return self.lnL(np.array([np.asarray(c, dtype=float) for c in cols]).T)
        return ln_f

    def as_func(self):
        ln_f = self.as_lnfunc()
        return lambda *cols: np.exp(ln_f(*cols))

    def _build_pool(self):
        """Exact fair draws of the box-truncated posterior, by rejection."""
        rng = np.random.RandomState(self.seed + 7)
        kept, n_tot, n_in = [], 0, 0
        while n_in < TRUTH_POOL_N:
            counts = rng.multinomial(200000, self.wt)
            chunks = [rng.multivariate_normal(self.means[k], self.covs[k], counts[k])
                      for k in range(self.ncomp) if counts[k] > 0]
            draw = np.vstack(chunks)
            rng.shuffle(draw)   # multinomial blocks are ordered by component
            n_tot += len(draw)
            inside = np.all((draw > self.llim) & (draw < self.rlim), axis=1)
            draw = draw[inside]
            n_in += len(draw)
            kept.append(draw)
        self._pool = np.vstack(kept)[:TRUTH_POOL_N]
        self._box_mass = float(n_in) / n_tot

    @property
    def pool(self):
        if self._pool is None:
            self._build_pool()
        return self._pool

    @property
    def true_lnZ(self):
        """Truth for the sampler-returned integral \\int L p_prior dx with
        normalized uniform prior: LNL_OFFSET + ln(in-box mass) - sum ln(width)."""
        if self._box_mass is None:
            self._build_pool()
        return (LNL_OFFSET + np.log(self._box_mass)
                - float(np.sum(np.log(self.rlim - self.llim))))


# ----------------------------------------------------------------------------
# Reading back the weighted posterior cloud (tolerant of _rvs conventions)
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
    """ln(weight) = lnL + ln p_prior - ln p_sampling per stored sample, across
    the heterogeneous _rvs conventions (log-keyed AV/NF/portfolio, linear-keyed
    default/AC, GMM storing lnL under 'integrand' when return_lnI is set)."""
    if "log_integrand" in rvs:
        lnL = _asnumpy(rvs["log_integrand"]).astype(float)
    elif "integrand" in rvs:
        L = _asnumpy(rvs["integrand"]).astype(float)
        lnL = L if np.nanmin(L) < 0 else np.log(L + 1e-300)
    else:
        raise KeyError("no integrand in _rvs; run with save_intg=True")

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


# ----------------------------------------------------------------------------
# Shape metrics
# ----------------------------------------------------------------------------
def _js_from_hists(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / (b[mask] + 1e-300))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def shape_metrics(target, X, ln_wt, rng):
    """Compare weighted cloud (X, ln_wt) against the target truth pool.

    Returns dict with per-dim JS + matched-n_ESS JS floors, mean pulls, width
    ratios, and max correlation-coefficient discrepancy."""
    pool = target.pool
    w = np.exp(ln_wt - np.max(ln_wt))
    w = w / np.sum(w)
    ness = n_ess_kish(ln_wt)

    mu_true = pool.mean(axis=0)
    sd_true = pool.std(axis=0)
    mu_w = np.sum(w[:, None] * X, axis=0)
    var_w = np.sum(w[:, None] * (X - mu_w) ** 2, axis=0)
    sd_w = np.sqrt(var_w)

    js, js_floor = [], []
    n_sub = int(min(max(ness, 50), len(pool) // 10))
    for d in range(target.ndim):
        edges = np.linspace(target.llim[d], target.rlim[d], JS_NBINS + 1)
        h_pool, _ = np.histogram(pool[:, d], bins=edges)
        h_run, _ = np.histogram(X[:, d], bins=edges, weights=w)
        js.append(_js_from_hists(h_run.astype(float), h_pool.astype(float)))
        # JS floor: perfect sampler at the same effective sample size
        f = []
        for _ in range(5):
            sub = pool[rng.choice(len(pool), size=n_sub, replace=False), d]
            h_sub, _ = np.histogram(sub, bins=edges)
            f.append(_js_from_hists(h_sub.astype(float), h_pool.astype(float)))
        js_floor.append(float(np.mean(f) + 2.0 * np.std(f)))

    # correlation matrices (guard zero-width dims)
    corr_diff = 0.0
    if target.ndim > 1:
        cov_w = np.einsum("i,ij,ik->jk", w, X - mu_w, X - mu_w)
        corr_w = cov_w / np.outer(sd_w, sd_w)
        corr_t = np.corrcoef(pool.T)
        corr_diff = float(np.max(np.abs(corr_w - corr_t)
                                 [np.triu_indices(target.ndim, 1)]))

    return dict(
        n_ess=ness,
        js=[float(x) for x in js],
        js_floor=js_floor,
        mean_pull=[float(x) for x in (mu_w - mu_true) / sd_true],
        width_ratio=[float(x) for x in sd_w / sd_true],
        corr_diff_max=corr_diff,
    )


# ----------------------------------------------------------------------------
# Sampler adapter (production API, tolerant of older branches)
# ----------------------------------------------------------------------------
KNOWN_SAMPLERS = ("AV", "GMM", "NF", "portfolio", "AC", "default")


def _gpu_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _force_cpu(s):
    """Mirror production's --sampler-xpy numpy instance override.  Needed for
    GMM: mcsamplerEnsemble sets cupy_ok on *import* success without a device
    probe, so on a cupy-installed GPU-less node it crashes in setup()."""
    s.xpy = np
    s.identity_convert = lambda x: x
    s.identity_convert_togpu = lambda x: x
    return s


_CPU_PATCHED = False


def _force_cpu_modules():
    """The GMM stack (mcsamplerEnsemble -> MonteCarloEnsemble ->
    gaussian_mixture_model) selects cupy at *module* level on import success
    with no device probe, and the inner integrator has no xpy argument; on a
    cupy-installed GPU-less node the only recourse is patching the module
    globals to numpy (what production sees when cupy is absent)."""
    global _CPU_PATCHED
    if _CPU_PATCHED:
        return
    import importlib
    import scipy.special as _sp
    for name in ("mcsampler", "mcsamplerEnsemble", "MonteCarloEnsemble",
                 "gaussian_mixture_model", "mcsamplerGPU",
                 "mcsamplerAdaptiveVolume", "mcsamplerPortfolio",
                 "mcsamplerNFlow"):
        try:
            mod = importlib.import_module("RIFT.integrators." + name)
        except Exception:
            continue
        for attr, val in (("xpy_default", np), ("cupy_ok", False),
                          ("xpy_special_default", _sp),
                          ("identity_convert", lambda x: x),
                          ("identity_convert_togpu", lambda x: x)):
            if hasattr(mod, attr):
                setattr(mod, attr, val)
    _CPU_PATCHED = True


def build_sampler(kind, target, n_chunk):
    if not _gpu_available():
        _force_cpu_modules()
    from RIFT.integrators import mcsampler

    def uniform_pdf(d):
        wdt = target.rlim[d] - target.llim[d]
        return np.vectorize(lambda x, wdt=wdt: 1.0 / wdt)

    if kind == "default":
        s = mcsampler.MCSampler()
    elif kind == "AC":
        from RIFT.integrators import mcsamplerGPU
        s = mcsamplerGPU.MCSampler()
    elif kind == "GMM":
        from RIFT.integrators import mcsamplerEnsemble
        s = mcsamplerEnsemble.MCSampler()
        if not _gpu_available():
            _force_cpu(s)
    elif kind == "AV":
        from RIFT.integrators import mcsamplerAdaptiveVolume
        try:
            s = mcsamplerAdaptiveVolume.MCSampler(n_chunk=n_chunk)
        except TypeError:   # older signature
            s = mcsamplerAdaptiveVolume.MCSampler()
    elif kind == "NF":
        from RIFT.integrators import mcsamplerNFlow
        s = mcsamplerNFlow.MCSampler()
    elif kind == "portfolio":
        from RIFT.integrators import (mcsamplerPortfolio,
                                      mcsamplerAdaptiveVolume, mcsamplerEnsemble)
        try:
            m1 = mcsamplerAdaptiveVolume.MCSampler(n_chunk=n_chunk)
        except TypeError:
            m1 = mcsamplerAdaptiveVolume.MCSampler()
        m2 = mcsamplerEnsemble.MCSampler()
        if not _gpu_available():
            _force_cpu(m1)
            _force_cpu(m2)
        s = mcsamplerPortfolio.MCSampler(portfolio=[m1, m2])
        if not _gpu_available():
            _force_cpu(s)
    else:
        raise ValueError("unknown sampler kind %r" % kind)

    for d, p in enumerate(target.params):
        s.add_parameter(p, uniform_pdf(d), prior_pdf=uniform_pdf(d),
                        left_limit=float(target.llim[d]),
                        right_limit=float(target.rlim[d]),
                        adaptive_sampling=True)
    return s


def run_one(kind, target, nmax, neff, n_chunk=10000, seed=987654, verbose=False):
    """Run one sampler on one target; return metrics dict (never raises)."""
    t0 = time.time()
    if kind == "NF":
        nmax = min(nmax, NF_NMAX_CAP)
    out = dict(kind=kind, target=target.name, ndim=target.ndim,
               ncomp=target.ncomp, target_seed=target.seed, nmax=int(nmax))
    try:
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
        except Exception:
            pass
        s = build_sampler(kind, target, n_chunk)
        ln_f = target.as_lnfunc()
        params = target.params
        extra = dict(n=n_chunk, n_adapt=100, floor_level=0.0, tempering_exp=0.1,
                     neff=neff, nmax=int(nmax), save_intg=True, verbose=verbose)
        if hasattr(s, "setup"):
            try:
                s.setup()
            except TypeError:
                pass

        if kind == "default":
            f = target.as_func()
            I, var, eff, _ = s.integrate(f, *params, no_protect_names=True, **extra)
            lnI = float(np.log(I))
            relerr = float(np.sqrt(var) / I)
        elif kind == "AC":
            lnI, logvar, eff, _ = s.integrate(ln_f, *params, no_protect_names=True,
                                              use_lnL=True, **extra)
            lnI = float(_asnumpy(lnI))
            relerr = float(np.exp(float(_asnumpy(logvar)) / 2 - lnI))
        elif kind == "GMM":
            n_iters = max(2, int(nmax / n_chunk))
            lnI, logvar, eff, _ = s.integrate(ln_f, *params, min_iter=n_iters,
                                              max_iter=n_iters, correlate_all_dims=True,
                                              n_comp=max(1, target.ncomp),
                                              use_lnL=True, return_lnI=True, **extra)
            lnI = float(_asnumpy(lnI))
            relerr = float(np.exp(float(_asnumpy(logvar)) / 2 - lnI))
        else:  # AV, NF, portfolio: integrate_log
            lnI, logvar, eff, _ = s.integrate_log(ln_f, *params,
                                                  no_protect_names=True, **extra)
            lnI = float(_asnumpy(lnI))
            logvar = float(_asnumpy(logvar))
            relerr = float(np.exp(0.5 * logvar - lnI)) if np.isfinite(logvar) else float("nan")

        eff = float(_asnumpy(eff))
        ln_wt = log_weights_from_rvs(s._rvs)
        X = np.column_stack([_asnumpy(s._rvs[p]).astype(float).flatten()
                             for p in params])
        rng = np.random.RandomState(seed + 1)
        out.update(shape_metrics(target, X, ln_wt, rng))
        out.update(lnI=lnI, true_lnZ=float(target.true_lnZ),
                   bias_ln=lnI - float(target.true_lnZ), rel_err=relerr,
                   n_eff=eff, n_eval=int(getattr(s, "ntotal", 0)) or int(nmax),
                   wallclock=time.time() - t0, error=None)
    except Exception as e:
        import traceback
        out.update(error="{}: {}".format(type(e).__name__, e),
                   traceback=traceback.format_exc(), wallclock=time.time() - t0)
    return out


# ----------------------------------------------------------------------------
# Warm-start / sequential-reuse cases
# ----------------------------------------------------------------------------
# These guard two portfolio defects that are INVISIBLE to the ordinary matrix, because both
# produce a wrong answer with no exception and (for the first) no statistical signature at all:
#
#   (a) a warm-start seed that is accepted but never installed on the draw path.  Measured: with
#       the AV install disabled, 12/12 runs still PASS and n_eff is 4692-5972 versus 3240-5941
#       for the correct code -- the broken config often scores BETTER, because the portfolio's
#       GMM member supplies nearly the whole warm-start win.  No black-box assertion can see
#       this; only a direct check that the AV member's live volume actually contracted.
#
#   (b) state leaking between sequential points.  mcsamplerPortfolio.integrate_log does not call
#       self.setup(), so a member's contracted live volume survives into the next integral.  If
#       the next point's support lies outside it, lnZ is biased low with a healthy-looking n_eff.
#       This needs no warm-start feature at all -- it bites any --n-events-to-analyze > 1 run.
WARM_KINDS = ("portfolio_warm", "portfolio_seq", "portfolio_seq_nobs", "AV_seq")
# For these kinds a starved run is a FAILURE, not "untestable": the whole point of the case is
# that correct code comfortably clears the floor (measured margins >= 8x).
STARVE_IS_FAIL = ("portfolio_warm", "portfolio_seq", "portfolio_seq_nobs")
WARM_NEFF_FLOOR = 1000.0   # case W A3; measured warm 2426-5941, cold 3.8-91.7
WARM_V_MAX = 0.9           # case W A1; measured installed 0.031-0.129, inert exactly 1.000
WARM_BOX_MULT = 3.0        # case W A2; measured on AV-member draws only -- see run_warm_case
SEQ_OFFSET = 2.0           # +-2 with sigma_1d=0.7 -> mean separation 4.0, both inside [-5,5]

# CASE-LIST NOTE, from directly reintroducing each bug and re-running (not from reasoning):
#   * portfolio_seq_nobs is the DISCRIMINANT for the leak.  With clear_warm_state() no-op'd it
#     gives n_eff 1.0 / 1.0 / 9.9 and lnZ bias -22.8 / -59.7 / -0.56 at ts 101/202/303, versus
#     n_eff ~2094 and bias +0.019 when the reset works.
#   * portfolio_seq (which re-bootstraps on point B) does NOT catch it: measured PASS with
#     n_eff 5979 while the bug was active, because the fresh B seed overwrites the stale grid.
#     It is kept as ONE row only, and only because it covers the reseed-after-reset path.
WARM_CASES = [   # (kind, ndim, ncomp, target_seed, nmax, neff, extra)
    ("portfolio_warm",     6, 1, 101, 600000, 3000, {}),
    ("portfolio_warm",     6, 1, 202, 600000, 3000, {}),
    ("portfolio_warm",     6, 1, 303, 600000, 3000, {}),
    ("portfolio_seq_nobs", 2, 1, 101, 100000, 2000, dict(scale_x0=1.0)),
    ("portfolio_seq_nobs", 2, 1, 202, 100000, 2000, dict(scale_x0=1.0)),
    ("portfolio_seq_nobs", 2, 1, 303, 100000, 2000, dict(scale_x0=1.0)),
    # covers reseed-after-reset; NOT a leak discriminant (see note above)
    ("portfolio_seq",      2, 1, 101, 100000, 2000, dict(scale_x0=1.0)),
    # negative control: standalone AV must be unaffected (it reruns its own setup).  Warn-only.
    ("AV_seq",             2, 1, 101, 100000, 2000, dict(scale_x0=1.0)),
]


def _warm_seed_cloud(target, n=3000):
    """Fair draws from the target's own truth pool -- a PERFECT seed, so any shortfall is the
    warm-start machinery, not a bad proposal."""
    pool = target.pool
    rng = np.random.RandomState(target.seed + 13)
    idx = rng.choice(len(pool), size=min(int(n), len(pool)), replace=False)
    return np.asarray(pool[idx], dtype=float)


def _finish_record(out, target, s, lnI, logvar, eff, nmax, seed, t0):
    """Shared tail of run_one: shape metrics + lnZ bookkeeping from a finished sampler."""
    eff = float(_asnumpy(eff))
    ln_wt = log_weights_from_rvs(s._rvs)
    X = np.column_stack([_asnumpy(s._rvs[p]).astype(float).flatten() for p in target.params])
    rng = np.random.RandomState(seed + 1)
    out.update(shape_metrics(target, X, ln_wt, rng))
    lnI = float(_asnumpy(lnI))
    logvar = float(_asnumpy(logvar))
    relerr = float(np.exp(0.5 * logvar - lnI)) if np.isfinite(logvar) else float("nan")
    out.update(lnI=lnI, true_lnZ=float(target.true_lnZ), bias_ln=lnI - float(target.true_lnZ),
               rel_err=relerr, n_eff=eff, n_eval=int(getattr(s, "ntotal", 0)) or int(nmax),
               wallclock=time.time() - t0, error=None)
    return out


def run_warm_case(target, nmax, neff, n_chunk=10000, seed=987654, verbose=False):
    """Case W: does a warm-start seed reach the AV member's ACTIVE draw state?

    Two samplers: a cheap PROBE that is seeded and drawn once (white-box: reads the AV member's
    live volume and bin count), then a fresh one that is seeded and integrated (black-box).
    The probe is what catches the inert-seed bug; the integral catches "all warm channels died"."""
    t0 = time.time()
    out = dict(kind="portfolio_warm", target=target.name, ndim=target.ndim,
               ncomp=target.ncomp, target_seed=target.seed, nmax=int(nmax))
    try:
        np.random.seed(seed)
        cloud = _warm_seed_cloud(target)
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)

        probe = build_sampler("portfolio", target, n_chunk)
        try:
            probe.setup()
        except TypeError:
            pass
        probe.bootstrap_from_samples(cloud, cover_frac=0.0)
        probe.draw(n_chunk)
        av = probe.portfolio_realizations[0]
        out["warm_V"] = float(_asnumpy(getattr(av, "V", 1.0)))
        out["warm_bins"] = int(len(getattr(av, "binunique", [0])))
        # Measure concentration on the AV MEMBER'S OWN draws, not the portfolio mixture.  Measured:
        # with the AV install disabled the MIXTURE still concentrates 28x in the seed box, because
        # the GMM member is separately warm-started -- so a mixture-level ratio is not a
        # discriminant for this bug at all.  Drawing from the member isolates the path under test.
        _ps, _p, rv_av = av.draw_simplified(n_chunk)   # rv_av is (ndim, n)
        Xp = np.asarray(_asnumpy(rv_av), dtype=float).T
        out["warm_box_frac"] = float(np.mean(np.all((Xp >= lo) & (Xp <= hi), axis=1)))
        # what a UNIFORM (cold) proposal would put in the same box -- the null this must beat
        out["warm_box_frac_uniform"] = float(np.prod((hi - lo) / (target.rlim - target.llim)))

        s = build_sampler("portfolio", target, n_chunk)
        try:
            s.setup()
        except TypeError:
            pass
        s.bootstrap_from_samples(cloud, cover_frac=0.0)
        extra = dict(n=n_chunk, n_adapt=100, floor_level=0.0, tempering_exp=0.1,
                     neff=neff, nmax=int(nmax), save_intg=True, verbose=verbose)
        lnI, logvar, eff, _ = s.integrate_log(target.as_lnfunc(), *target.params,
                                              no_protect_names=True, **extra)
        _finish_record(out, target, s, lnI, logvar, eff, nmax, seed, t0)
    except Exception as e:
        import traceback
        out.update(error="{}: {}".format(type(e).__name__, e),
                   traceback=traceback.format_exc(), wallclock=time.time() - t0)
    return out


def run_seq_case(kind, target_b, target_a, nmax, neff, n_chunk=10000, seed=987654, verbose=False):
    """Cases S / S-nobs / AV_seq: integrate DISPLACED target A then target B on ONE sampler.

    Scores point B only.  If the sampler carries A's contracted live volume into B, B's mass sits
    outside it and lnZ collapses.  Uses clear_warm_state() when present and falls back to the old
    `_warm = None` otherwise, so the case RUNS on a base branch without the API -- and fails there,
    which is the point."""
    t0 = time.time()
    out = dict(kind=kind, target=target_b.name, ndim=target_b.ndim, ncomp=target_b.ncomp,
               target_seed=target_b.seed, nmax=int(nmax))
    try:
        np.random.seed(seed)
        s = build_sampler("AV" if kind == "AV_seq" else "portfolio", target_a, n_chunk)
        # PRODUCTION SHAPE: supply gmm_dict.  Production always does, and it is not an inert spec --
        # monte_carlo.integrator stores the caller's dict without copying and writes trained models
        # into it, so this is the configuration in which stale-proposal aliasing can occur.  A gate
        # that only ever ran the gmm_dict=None branch could not see that class of defect at all.
        _setup_kw = {}
        if kind != "AV_seq":
            _dims = tuple(range(len(target_a.params)))
            _setup_kw = dict(n_comp={_dims: 2}, gmm_dict={_dims: None}, correlate_all_dims=True)
        try:
            s.setup(**_setup_kw)
        except TypeError:
            s.setup()
        extra = dict(n=n_chunk, n_adapt=100, floor_level=0.0, tempering_exp=0.1,
                     neff=neff, nmax=int(nmax), save_intg=True, verbose=verbose)
        if kind != "portfolio_seq_nobs":
            s.bootstrap_from_samples(_warm_seed_cloud(target_a), cover_frac=0.0)
        s.integrate_log(target_a.as_lnfunc(), *target_a.params,
                        no_protect_names=True, **extra)
        # record that point A ACTUALLY trained something -- otherwise "cleared" could pass
        # vacuously on a run where the GMM never fitted at all
        _trained_before_reset = False
        if kind != "AV_seq":
            try:
                _trained_before_reset = any(
                    v is not None for v in s.portfolio_realizations[1].integrator.gmm_dict.values())
            except Exception as _e_tr:
                out["seq_gmm_error"] = "pre-reset inspection failed: {}: {}".format(
                    type(_e_tr).__name__, _e_tr)

        # ---- the transition the driver performs between two points ----
        s._rvs = {}
        if hasattr(s, "clear_warm_state"):
            s.clear_warm_state()
        else:
            s._warm = None
        out["used_clear_api"] = bool(hasattr(s, "clear_warm_state"))
        # WHITE-BOX: did the reset actually clear the GMM member's TRAINED PROPOSAL?  Measured, the
        # statistical rows cannot answer this: with the trained proposal leaking, n_eff is 196-1700
        # and |lnZ bias| <= 0.010 -- degraded but passing, because a stale GMM proposal is merely a
        # bad proposal (the AV member still covers the support), unlike the AV grid leak which
        # removes support and does bias.  So the leak must be observed directly.
        if kind != "AV_seq":
            try:
                _gd = s.portfolio_realizations[1].integrator.gmm_dict
                out["seq_gmm_trained_before_reset"] = bool(_trained_before_reset)
                out["seq_gmm_cleared"] = all(v is None for v in _gd.values())
            except Exception as _e_gd:
                out["seq_gmm_cleared"] = None
                out["seq_gmm_error"] = "{}: {}".format(type(_e_gd).__name__, _e_gd)

        if kind != "portfolio_seq_nobs":
            s.bootstrap_from_samples(_warm_seed_cloud(target_b), cover_frac=0.0)
        lnI, logvar, eff, _ = s.integrate_log(target_b.as_lnfunc(), *target_b.params,
                                              no_protect_names=True, **extra)
        _finish_record(out, target_b, s, lnI, logvar, eff, nmax, seed, t0)
    except Exception as e:
        import traceback
        out.update(error="{}: {}".format(type(e).__name__, e),
                   traceback=traceback.format_exc(), wallclock=time.time() - t0)
    return out


# ----------------------------------------------------------------------------
# Pass/fail policy
# ----------------------------------------------------------------------------
def evaluate(r):
    """Return (status, reasons) for one run record.

    status: "PASS" | "FAIL" | "STARVED" | "ERROR".
    STARVED (n_eff below the shape-testability floor at this budget) is NOT an
    absolute failure: high-dimensional mixtures legitimately exhaust production
    budgets (the FinerNet high-D degradation), so starved rows gate only
    *differentially* -- a candidate that starves where its base was healthy is
    a regression (see compare_shape_results.py)."""
    if r.get("error"):
        return "ERROR", ["ERROR " + r["error"]]
    if r["kind"] in STARVE_IS_FAIL and r["n_eff"] < MIN_NEFF_FOR_SHAPE:
        # NOT "untestable": correct code clears this floor by >= 8x on these cases (measured
        # 799-2103 vs 0.0-12.8 when state leaks), so starvation here IS the defect.
        return "FAIL", ["n_eff={:.0f} < {:.0f}: warm/sequential case must not starve".format(
            r["n_eff"], MIN_NEFF_FOR_SHAPE)]
    if r["n_eff"] < MIN_NEFF_FOR_SHAPE:
        return "STARVED", ["n_eff={:.0f} < {:.0f}: shape untestable at this budget".format(
            r["n_eff"], MIN_NEFF_FOR_SHAPE)]
    reasons = []
    if r["kind"] == "portfolio_warm":
        # A1: white-box, and deliberately so.  An inert seed leaves V at exactly 1.000 with a
        # single live bin; a working one contracted to 0.031-0.129 with 489-656 bins in every
        # measured run.  No RNG enters either quantity, so the margin is categorical.
        if not (r.get("warm_V", 1.0) < WARM_V_MAX and r.get("warm_bins", 1) > 1):
            reasons.append("warm seed NOT installed on the draw path: AV member V={:.3f}, "
                           "live bins={} (cold state)".format(r.get("warm_V", float("nan")),
                                                              r.get("warm_bins", -1)))
        # A2: behavioural confirmation -- draws must actually concentrate in the seed box.
        if r.get("warm_box_frac", 0.0) < WARM_BOX_MULT * r.get("warm_box_frac_uniform", 1.0):
            reasons.append("warm draws not concentrated in the seed box: {:.3f} < {:.1f}x{:.4f}"
                           .format(r.get("warm_box_frac", float("nan")), WARM_BOX_MULT,
                                   r.get("warm_box_frac_uniform", float("nan"))))
        # A3: feature level -- catches "every warm-start channel went inert", which A1/A2 (AV
        # member only) would miss.
        if r["n_eff"] < WARM_NEFF_FLOOR:
            reasons.append("warm n_eff {:.0f} < {:.0f}".format(r["n_eff"], WARM_NEFF_FLOOR))
    if r["kind"] in ("portfolio_seq", "portfolio_seq_nobs"):
        # Require BOTH observations to be affirmatively True.  Testing only for `cleared is False`
        # let three distinct failures read as success: training never happened (so "cleared" is
        # trivially true and proves nothing), the inspection raised (cleared is None), or the
        # fields were absent entirely.  A check that cannot run is a failed check, not a pass.
        _trained = r.get("seq_gmm_trained_before_reset")
        _cleared = r.get("seq_gmm_cleared")
        if _trained is not True:
            reasons.append("GMM clearing check could not run: trained_before_reset={!r} -- the "
                           "member never trained, so a 'cleared' verdict would be vacuous{}".format(
                               _trained,
                               " [" + r["seq_gmm_error"] + "]" if r.get("seq_gmm_error") else ""))
        elif _cleared is not True:
            reasons.append("reset did NOT clear the GMM member's trained proposal (cleared={!r}): "
                           "point B inherits point A's fitted components (setup-arg aliasing){}".format(
                               _cleared,
                               " [" + r["seq_gmm_error"] + "]" if r.get("seq_gmm_error") else ""))
    ness = max(r["n_ess"], 1.0)
    for d, (js, floor) in enumerate(zip(r["js"], r["js_floor"])):
        thresh = JS_MULT * floor + JS_ABS_MIN
        if js > thresh:
            reasons.append("JS[{}]={:.4f} > {:.4f} (floor {:.4f})".format(
                d, js, thresh, floor))
    tol_mean = max(5.0 / np.sqrt(ness), 0.05)
    for d, pull in enumerate(r["mean_pull"]):
        if abs(pull) > tol_mean:
            reasons.append("mean_pull[{}]={:+.3f} > {:.3f}".format(d, pull, tol_mean))
    tol_wid = max(5.0 / np.sqrt(2.0 * ness), 0.05)
    for d, wr in enumerate(r["width_ratio"]):
        if abs(wr - 1.0) > tol_wid:
            reasons.append("width_ratio[{}]={:.3f} (tol {:.3f})".format(d, wr, tol_wid))
    tol_corr = max(8.0 / np.sqrt(ness), 0.08)
    if r["corr_diff_max"] > tol_corr:
        reasons.append("corr_diff_max={:.3f} > {:.3f}".format(
            r["corr_diff_max"], tol_corr))
    relerr = r["rel_err"] if np.isfinite(r.get("rel_err", float("nan"))) else 0.05
    tol_lnZ = max(4.0 * relerr, 0.10)
    if abs(r["bias_ln"]) > tol_lnZ:
        reasons.append("lnZ bias {:+.3f} > {:.3f}".format(r["bias_ln"], tol_lnZ))
    return ("FAIL" if reasons else "PASS"), reasons


# ----------------------------------------------------------------------------
# Matrix presets + CLI
# ----------------------------------------------------------------------------
# PER-CELL BUDGET OVERRIDES.  The matrix budget is nmax_per_dim*d for EVERY cell, and strictness
# is per-SAMPLER, so a single mis-budgeted cell can be neither re-budgeted nor exempted without a
# hook like this one.  (WARM_CASES already carries per-case budgets; same idea.)
#
# GMM mix_d6_n3_s303 sits on the n_eff=100 starvation floor at the matrix budget, which makes it a
# coin flip as a merge-BLOCKING row -- measured over 8 fresh seeds:
#
#     budget   n_eff min/med/max   clears 100   median |bias|
#      x1        59 / 105 / 159       5/8          0.014
#      x2       105 / 169 / 214       8/8          0.010
#      x4       209 / 293 / 473       8/8          0.012
#
# The bias is flat, so this is purely threshold margin, not a defect.  x2 clears 8/8 but its
# MINIMUM (105) is 5% above the floor -- not a margin worth trusting for a row that has already
# swung between 66 and 119 on unchanged code.  x4 gives min 209 (2.1x margin) and costs ~4% of the
# gate's total evaluations, since it is one cell of ~96.
#
# DO NOT add ("GMM", 4, 2, 101) here.  That cell -- the `quick` preset's d=4 GMM row, which skips as
# STARVED under `RIFT_RUN_EXPENSIVE=1 pytest` -- looks like the same problem and is not.  Measured
# over 8 fresh run seeds at each budget:
#
#     budget      n_eff min/med/max   clears 100   PASS   median width_ratio[1]
#      x1 200k        11 /  50 / 101      1/8       0/8         0.989
#      x4 800k        28 /  44 / 179      2/8       1/8         0.919
#     x32 6.4M        28 / 139 / 217      6/8       1/8         0.889
#
# n_eff grows as ~nmax**0.3, so no budget clears the floor: at x32 (8x the STANDARD preset's own d=4
# budget) it still starves 2/8.  And the failures are not starvation -- width_ratio[1] degrades
# MONOTONICALLY with budget while the other three dims stay at 1.00, so a bigger budget converts a
# documented skip into a merge-blocking FAIL.  AV on the identical target passes 8/8 at every budget
# with width_ratio 1.000, so the target and the thresholds are sound; this is a GMM defect.
# FOLLOWUPS.md items 5 (why the skip is deliberate) and 6 (the defect itself).
CELL_BUDGET_MULT = {
    ("GMM", 6, 3, 303): 4,
}


def cell_budget(kind, ndim, ncomp, tseed, nmax_per_dim, apply_overrides=True):
    """THE budget for one matrix cell: nmax_per_dim*ndim, times any per-cell override.

    Single entry point because there are two of them.  `main()` builds jobs, and
    test_shape_recovery.py parametrizes the same matrix under pytest; computing the budget
    separately in each left the override applied on one path and not the other, so the cell this
    table exists to fix stayed starved under `RIFT_SHAPE_PRESET=standard pytest`.

    `apply_overrides=False` returns the plain contract value.  Callers pass it when the budget was
    named EXPLICITLY (`--nmax-per-dim`), because the CLI documents `nmax = this * ndim` and silently
    scaling that would corrupt exactly the controlled x1/x2/x4 comparisons this table was derived
    from.
    """
    base = int(nmax_per_dim) * int(ndim)
    if not apply_overrides:
        return base
    return base * int(CELL_BUDGET_MULT.get((kind, int(ndim), int(ncomp), int(tseed)), 1))

PRESETS = {
    # (dims, ncomps, target_seeds, nmax_per_dim, neff)
    "quick": (dict(dims=[2, 4], ncomps=[2], seeds=[101], nmax_per_dim=50000, neff=2000)),
    "standard": (dict(dims=[2, 4, 6, 8], ncomps=[1, 3], seeds=[101, 202, 303],
                      nmax_per_dim=200000, neff=3000)),
}


def _worker(job):
    kind, tgt_args, nmax, neff, seed = job[:5]
    extra = job[5] if len(job) > 5 else {}
    if kind == "portfolio_warm":
        return run_warm_case(MixtureTarget(*tgt_args, **extra), nmax, neff, seed=seed)
    if kind in ("portfolio_seq", "portfolio_seq_nobs", "AV_seq"):
        # SAME mixture, translated: A at -SEQ_OFFSET, B at +SEQ_OFFSET.  Different seeds would
        # give random, typically overlapping means and would not test displacement at all.
        a = MixtureTarget(*tgt_args, offset=-SEQ_OFFSET, **extra)
        b = MixtureTarget(*tgt_args, offset=+SEQ_OFFSET, **extra)
        return run_seq_case(kind, b, a, nmax, neff, seed=seed)
    return run_one(kind, MixtureTarget(*tgt_args, **extra), nmax, neff, seed=seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--preset", default="standard", choices=sorted(PRESETS))
    ap.add_argument("--samplers", default="AV,GMM,NF,portfolio",
                    help="comma list from: " + ",".join(KNOWN_SAMPLERS))
    ap.add_argument("--warm-cases", default="auto", choices=("auto", "on", "off"),
                    help="run the warm-start/sequential-reuse cases "
                         "(auto = on for --preset standard, off for quick)")
    ap.add_argument("--strict-samplers", default="AV,GMM",
                    help="samplers whose failures set exit code 1 (others warn)")
    ap.add_argument("--dims", default=None, help="override preset, e.g. 2,4,8")
    ap.add_argument("--ncomps", default=None)
    ap.add_argument("--target-seeds", default=None)
    ap.add_argument("--nmax-per-dim", type=int, default=None,
                    help="nmax = this * ndim (exactly; passing this disables per-cell overrides)")
    ap.add_argument("--no-cell-budget-mult", action="store_true",
                    help="ignore CELL_BUDGET_MULT even at preset defaults (for controlled "
                         "budget comparisons)")
    ap.add_argument("--neff", type=int, default=None)
    ap.add_argument("--run-seed", type=int, default=987654)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--json", default=None, help="write full records here")
    ap.add_argument("--verbose", action="store_true")
    opts = ap.parse_args(argv)

    cfg = dict(PRESETS[opts.preset])
    if opts.dims:
        cfg["dims"] = [int(x) for x in opts.dims.split(",")]
    if opts.ncomps:
        cfg["ncomps"] = [int(x) for x in opts.ncomps.split(",")]
    if opts.target_seeds:
        cfg["seeds"] = [int(x) for x in opts.target_seeds.split(",")]
    # An EXPLICIT --nmax-per-dim means the caller is controlling the budget; honour the documented
    # contract (nmax = this * ndim) rather than silently multiplying it.  --no-cell-budget-mult
    # disables the table even for preset defaults.
    _apply_cell_overrides = not bool(opts.no_cell_budget_mult)
    if opts.nmax_per_dim:
        cfg["nmax_per_dim"] = opts.nmax_per_dim
        _apply_cell_overrides = False
        if CELL_BUDGET_MULT:
            print("# --nmax-per-dim given explicitly: per-cell budget overrides DISABLED "
                  "({} cell(s) affected at preset defaults)".format(len(CELL_BUDGET_MULT)))
    if opts.neff:
        cfg["neff"] = opts.neff

    samplers = [x.strip() for x in opts.samplers.split(",") if x.strip()]
    strict = set(x.strip() for x in opts.strict_samplers.split(",") if x.strip())

    jobs = []
    for d in cfg["dims"]:
        for nc in cfg["ncomps"]:
            for ts in cfg["seeds"]:
                for kind in samplers:
                    jobs.append((kind, (d, nc, ts),
                                 cell_budget(kind, d, nc, ts, cfg["nmax_per_dim"],
                                             apply_overrides=_apply_cell_overrides),
                                 cfg["neff"], opts.run_seed))
    n_matrix = len(jobs)
    want_warm = (opts.warm_cases == "on" or
                 (opts.warm_cases == "auto" and opts.preset == "standard"))
    if want_warm:
        for kind, d, nc, ts, nmax, neff, extra in WARM_CASES:
            jobs.append((kind, (d, nc, ts), nmax, neff, opts.run_seed, dict(extra)))
    print("# shape_recovery: {} runs ({} targets x {} samplers){}, preset={}".format(
        len(jobs), n_matrix // len(samplers), len(samplers),
        " + {} warm/sequential cases".format(len(jobs) - n_matrix) if want_warm else "",
        opts.preset))
    sys.stdout.flush()

    if opts.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(opts.jobs) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    n_fail_strict, n_fail_warn, n_starved = 0, 0, 0
    print("\n{:<10s} {:<16s} {:>9s} {:>9s} {:>7s} {:>7s} {:>8s} {:>7s}  {}".format(
        "sampler", "target", "n_eff", "n_ESS", "JSmax", "|pull|", "widthdev",
        "lnZbias", "verdict"))
    for r in results:
        status, reasons = evaluate(r)
        if status == "PASS":
            tag = "PASS"
        elif status == "STARVED":
            tag = "STARVED"   # non-blocking; gates differentially vs base
            n_starved += 1
        elif r["kind"] in strict:
            tag = "FAIL"
            n_fail_strict += 1
        else:
            tag = "WARN"
            n_fail_warn += 1
        if r.get("error"):
            print("{:<10s} {:<16s} {:>9s} {:>9s} {:>7s} {:>7s} {:>8s} {:>7s}  {} {}".format(
                r["kind"], r["target"], "-", "-", "-", "-", "-", "-", tag, r["error"]))
            continue
        print("{:<10s} {:<16s} {:>9.0f} {:>9.0f} {:>7.4f} {:>7.3f} {:>8.3f} {:>+7.3f}  {}{}".format(
            r["kind"], r["target"], r["n_eff"], r["n_ess"], max(r["js"]),
            max(abs(p) for p in r["mean_pull"]),
            max(abs(w - 1) for w in r["width_ratio"]), r["bias_ln"], tag,
            ("  [" + "; ".join(reasons) + "]") if reasons else ""))
        sys.stdout.flush()

    if opts.json:
        with open(opts.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print("# wrote", opts.json)
    print("# strict failures: {}   warn-only failures: {}   starved (non-blocking): {}".format(
        n_fail_strict, n_fail_warn, n_starved))
    return 1 if n_fail_strict else 0


if __name__ == "__main__":
    sys.exit(main())
