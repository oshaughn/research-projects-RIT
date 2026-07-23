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

    def __init__(self, ndim, ncomp, seed, sigma_1d=0.7, scale_x0=3.0):
        self.ndim = int(ndim)
        self.ncomp = int(ncomp)
        self.seed = int(seed)
        self.name = "mix_d{}_n{}_s{}".format(ndim, ncomp, seed)
        self.params = ["x{}".format(i) for i in range(ndim)]
        self.llim = -BOX_HALF_WIDTH * np.ones(ndim)
        self.rlim = BOX_HALF_WIDTH * np.ones(ndim)
        rng = np.random.RandomState(seed)
        wt = rng.uniform(size=ncomp) + 0.1
        self.wt = wt / np.sum(wt)
        import scipy.stats as ss
        self.means, self.covs, self._mvns = [], [], []
        for k in range(ncomp):
            x0 = rng.uniform(-scale_x0 / np.sqrt(ndim), scale_x0 / np.sqrt(ndim), ndim)
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
    if r["n_eff"] < MIN_NEFF_FOR_SHAPE:
        return "STARVED", ["n_eff={:.0f} < {:.0f}: shape untestable at this budget".format(
            r["n_eff"], MIN_NEFF_FOR_SHAPE)]
    reasons = []
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
PRESETS = {
    # (dims, ncomps, target_seeds, nmax_per_dim, neff)
    "quick": (dict(dims=[2, 4], ncomps=[2], seeds=[101], nmax_per_dim=50000, neff=2000)),
    "standard": (dict(dims=[2, 4, 6, 8], ncomps=[1, 3], seeds=[101, 202, 303],
                      nmax_per_dim=200000, neff=3000)),
}


def _worker(job):
    kind, tgt_args, nmax, neff, seed = job
    target = MixtureTarget(*tgt_args)
    return run_one(kind, target, nmax, neff, seed=seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--preset", default="standard", choices=sorted(PRESETS))
    ap.add_argument("--samplers", default="AV,GMM,NF,portfolio",
                    help="comma list from: " + ",".join(KNOWN_SAMPLERS))
    ap.add_argument("--strict-samplers", default="AV,GMM",
                    help="samplers whose failures set exit code 1 (others warn)")
    ap.add_argument("--dims", default=None, help="override preset, e.g. 2,4,8")
    ap.add_argument("--ncomps", default=None)
    ap.add_argument("--target-seeds", default=None)
    ap.add_argument("--nmax-per-dim", type=int, default=None,
                    help="nmax = this * ndim")
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
    if opts.nmax_per_dim:
        cfg["nmax_per_dim"] = opts.nmax_per_dim
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
                                 cfg["nmax_per_dim"] * d, cfg["neff"], opts.run_seed))
    print("# shape_recovery: {} runs ({} targets x {} samplers), preset={}".format(
        len(jobs), len(jobs) // len(samplers), len(samplers), opts.preset))
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
