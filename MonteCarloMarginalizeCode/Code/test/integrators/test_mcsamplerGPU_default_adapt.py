#!/usr/bin/env python
"""
test_mcsamplerGPU_default_adapt.py

mcsamplerGPU adapts whenever n_adapt > 0, but it only caches the integrand history when
tempering_exp > 0.  With the default tempering_exp = 0 the adaptation block therefore ran
with no cache to read, and crashed before adapting anything: integrate() raised
NameError('int_vals') and integrate_log() raised KeyError('log_weights').  Both were
reachable from a bare MCSampler().add_parameter(..., adaptive_sampling=True) followed by
integrate(), which is what test/demo_mcsampler_foridiots.py does.

Two things sat behind those.  A sampler reused for a second integrate() keeps _rvs[p] from
the first pass while the integrand record restarts, so the two are different lengths: that
reached numpy.bincount as a length mismatch, and the end-of-pass cleanup would have paired
each likelihood with the wrong row.  And the weights the fallback uses have to be the
IMPORTANCE weights, not the raw integrand, or the histogram estimates p_s * f instead of
prior * f.

The last point is what most of this file measures, because a wrong weight rule still
produces plausible integrals -- the crash was the loud half of the defect.
"""
from __future__ import print_function

import numpy as np
import pytest

from RIFT.integrators import mcsamplerGPU

LO, HI = -1.5, 1.0
SIG = 0.3
# Normalized prior 1/(HI-LO), so integrate() returns the prior-weighted average.
TRUTH = np.sqrt(2 * np.pi) * SIG / (HI - LO)


def _sampler():
    s = mcsamplerGPU.MCSampler()
    s.add_parameter(
        'x',
        pdf=np.vectorize(lambda x: 1.0 / (HI - LO)),
        cdf_inv=None,
        prior_pdf=np.vectorize(lambda x: 1.0 / (HI - LO)),
        left_limit=LO, right_limit=HI,
        adaptive_sampling=True,
    )
    return s


def _gaussian(x):
    return np.exp(-x ** 2 / (2 * SIG ** 2))


def _lngaussian(x):
    return -x ** 2 / (2 * SIG ** 2)


# ---------------------------------------------------------------------------
# A proposal deliberately unequal to the prior, so that f, f*prior/p_s and a flat
# weight are three different vectors and the tests below can tell them apart.
# p_s is a linear ramp TILT:1 high on the left; the prior stays uniform.
# ---------------------------------------------------------------------------
SLO, SHI, TILT = -1.0, 1.0, 10.0
_SNORM = (SHI - SLO) * (TILT + 1.0) / 2.0


def _skew_pdf(x):
    t = (np.asarray(x, dtype=float) - SLO) / (SHI - SLO)
    return (TILT - (TILT - 1.0) * t) / _SNORM


def _skew_cdf_inv(u):
    u = np.asarray(u, dtype=float)
    a, b = -(TILT - 1.0) / 2.0, TILT
    t = (-b + np.sqrt(b * b + 4 * a * (_SNORM * u / (SHI - SLO)))) / (2 * a)
    return SLO + t * (SHI - SLO)


def _flat_prior(x):
    return np.ones_like(np.asarray(x, dtype=float)) / (SHI - SLO)


def _skew_sampler():
    s = mcsamplerGPU.MCSampler()
    s.add_parameter('x', pdf=_skew_pdf, cdf_inv=_skew_cdf_inv, prior_pdf=_flat_prior,
                    left_limit=SLO, right_limit=SHI, adaptive_sampling=True)
    return s


# ---------------------------------------------------------------------------
# The crashes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_integrate_default_tempering_runs_and_is_unbiased(seed):
    """integrate() with the default tempering_exp=0: used to raise NameError('int_vals')."""
    np.random.seed(seed)
    ret, var, neff, _ = _sampler().integrate(
        np.vectorize(_gaussian), 'x', n=500, nmax=10000, neff=1e9, full_output=True)
    ret = float(ret)
    assert np.isfinite(ret)
    # 4x the sampler's own error estimate: this is a bias check, not a precision check.
    assert abs(ret - TRUTH) < 4 * np.sqrt(float(var)) + 0.05 * TRUTH
    assert float(neff) > 1


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_integrate_log_default_tempering_runs(seed):
    """integrate_log() with the default tempering_exp=0: used to raise KeyError('log_weights')."""
    np.random.seed(seed)
    res = _sampler().integrate_log(
        np.vectorize(_lngaussian), 'x', n=500, nmax=10000, neff=1e9)
    ln_ret = float(res[0])
    assert np.isfinite(ln_ret)
    assert abs(np.exp(ln_ret) - TRUTH) < 0.2 * TRUTH


# ---------------------------------------------------------------------------
# The weight rule.  These are the tests that fail for a plausible WRONG fix.
# ---------------------------------------------------------------------------

def test_adaptation_uses_importance_weights_not_the_raw_integrand():
    """One adaptation step on a CONSTANT integrand must return the proposal to the prior.

    Points come from p_s and the weights are f*prior/p_s, so the weighted histogram
    estimates f*prior -- here a constant times a uniform prior, i.e. flat -- no matter how
    skewed p_s was.  Weighting by f alone (or by any flat vector) instead estimates p_s*f
    and hands back the tilt it started with.

    nmax == n so that exactly ONE histogram update happens: past the first step the
    zero-bin behaviour of compute_hist dominates and swamps the weight rule.

    Measured over these 8 seeds: correct weights give tilt 0.97-1.02; weighting by fval,
    or by a flat vector, gives 3.38-3.51.
    """
    grid = np.linspace(SLO + 0.1, SHI - 0.1, 201)
    n_side = len(grid) // 4
    for seed in range(8):
        np.random.seed(seed)
        s = _skew_sampler()
        s.integrate(lambda x: np.ones(len(np.asarray(x))), 'x',
                    n=20000, nmax=20000, neff=1e9, full_output=True)
        v = np.asarray(s.pdf['x'](grid), dtype=float)
        tilt = float(np.mean(v[:n_side]) / max(np.mean(v[-n_side:]), 1e-30))
        assert 0.85 < tilt < 1.2, (
            "seed %d: adapted proposal still tilted %.2f (p_s was %.0f:1). The weighted "
            "histogram is estimating p_s*f, not prior*f -- the weights are not the "
            "importance weights." % (seed, tilt, TILT))


def test_integrate_log_fallback_adapts_on_the_likelihood():
    """The integrate_log fallback must carry lnL.

    It is only reachable at tempering_exp == 0, where log_weights = 0*lnL + ln p - ln p_s
    has no likelihood in it at all and the histogram just replays the current proposal.
    log_integrand = lnL + ln p - ln p_s is the log-space twin of integrate()'s int_val.

    Target sits at +0.6 while p_s piles up on the left, so an adapted proposal that found
    the likelihood is far denser on the right.  Measured: log_integrand gives a ratio of at
    least 178 on every one of these seeds; log_weights gives as little as 0.02.
    """
    x0, sig = 0.6, 0.15
    grid = np.linspace(SLO + 1e-6, SHI - 1e-6, 401)
    i_hi = int(np.argmin(np.abs(grid - x0)))
    i_lo = int(np.argmin(np.abs(grid + x0)))
    ratios = []
    for seed in range(8):
        np.random.seed(seed)
        s = _skew_sampler()
        s.integrate_log(lambda x: -(np.asarray(x) - x0) ** 2 / (2 * sig ** 2), 'x',
                        n=400, nmax=4000, neff=1e9)
        v = np.asarray(s.pdf['x'](grid), dtype=float)
        ratios.append(float(v[i_hi] / max(v[i_lo], 1e-30)))
    assert min(ratios) > 25.0, (
        "adapted proposal is not concentrated on the likelihood: per-seed density ratio "
        "at the peak vs its mirror was %r" % (ratios,))


# ---------------------------------------------------------------------------
# The reused sampler
# ---------------------------------------------------------------------------

def test_reused_sampler_adapts_without_length_mismatch():
    """Second integrate() on the same sampler: weights and points must line up.

    The first pass leaves len(_rvs['x']) == nmax with no "integrand" record; the second
    starts that record from zero.  Slicing both to n_history handed numpy.bincount 2*n
    points against n weights ("The weights and list don't have the same length").
    """
    np.random.seed(0)
    s = _sampler()
    s.integrate(np.vectorize(_gaussian), 'x', n=500, nmax=2000, neff=1e9,
                full_output=True)                       # no cache: save_intg stays False
    assert len(s._rvs['x']) >= 2000, "first pass should leave a parameter record behind"
    s.reset_sampling('x')
    ret, var, neff, _ = s.integrate(np.vectorize(_gaussian), 'x', n=500, nmax=4000,
                                    neff=1e9, full_output=True, tempering_exp=0.2)
    ret = float(ret)
    assert np.isfinite(ret)
    assert abs(ret - TRUTH) < 0.25 * TRUTH


@pytest.mark.parametrize("log_space", [False, True])
def test_reused_sampler_keeps_rvs_rows_aligned(log_space):
    """_rvs rows must still describe the same draw after a reuse pass.

    The end-of-pass cleanup reindexes EVERY _rvs key with one index list built from the
    length of the integrand record.  On a carried-over parameter record that selects its
    FIRST rows, so each likelihood ends up beside a different draw's parameter value.  The
    lengths agree afterwards, so nothing downstream can notice.

    An injective integrand makes the pairing checkable: recomputing it from the stored x
    must reproduce the stored integrand row by row.
    """
    def f(x):
        return np.exp(0.7 * np.asarray(x, dtype=float)) + 0.11

    np.random.seed(3)
    s = _sampler()
    s.integrate(f, 'x', n=500, nmax=2000, neff=1e9, full_output=True)   # leaves no cache
    s.reset_sampling('x')
    if log_space:
        s.integrate_log(lambda x: np.log(f(x)), 'x', n=500, nmax=2000, neff=1e9,
                        tempering_exp=0.2)
        stored = np.exp(np.asarray(s._rvs["log_integrand"], dtype=float))
    else:
        s.integrate(f, 'x', n=500, nmax=2000, neff=1e9, full_output=True,
                    tempering_exp=0.2)
        stored = np.asarray(s._rvs["integrand"], dtype=float)
    xs = np.asarray(s._rvs['x'], dtype=float)
    assert len(xs) == len(stored)
    bad = int(np.count_nonzero(np.abs(f(xs) - stored) > 1e-9 * np.abs(stored)))
    assert bad == 0, "%d of %d _rvs rows pair x with another draw's integrand" % (bad, len(xs))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
