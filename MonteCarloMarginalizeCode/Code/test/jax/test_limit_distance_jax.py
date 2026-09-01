#!/usr/bin/env python
"""
--limit-distance on the differentiable (JAX) arm.

The batchmode ILE reaches distance through a SAMPLER (mcsampler add_parameter);
the JAX driver reaches it through a QUADRATURE GRID (make_distance_grid ->
JAXDist*MargLikelihood) or, in the explicitly-6-D modes, through sample_prior /
log_prior.  Both arms must accept the same box, or the cross-sampler lnZ
comparison this option exists to enable is not a fair one.

WHAT MUST HOLD.  Narrowing the box changes what is INTEGRATED, never how the
prior is NORMALIZED.  On this arm the trap has the same shape as on the other:
make_distance_grid ended with

    w = w / np.sum(w)               # normalize the distance average

which normalizes the prior onto whatever range the grid happens to span, so a
narrowed grid renormalizes onto itself and the marginal comes back looking
"unchanged" while the evidence scale has moved by the prior mass outside the box
(here: several nats).  d_prior_range= splits the two roles.

WHAT DOES NOT HOLD EXACTLY, AND WHY.  Unlike the batchmode arm -- where the
narrowed and full-range estimators integrate the SAME function and agree to the
truncated likelihood mass -- the JAX marginalized arm changes the QUADRATURE
RESOLUTION when it changes the range: n_grid nodes over a narrow box resolve the
distance integrand better than n_grid nodes over [d_min,d_max].  So narrowed and
full-range lnL agree only to the FULL-RANGE grid's own discretization error, and
the narrowed one is the more accurate of the two (that being the point).  The
tests below therefore compare BOTH against a converged reference rather than
asserting they agree with each other to a tolerance neither of them earns.
"""

import os
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from RIFT.likelihood.jax_ile.core import (                     # noqa: E402
    make_distance_grid, make_distance_grid_adaptive)
from RIFT.likelihood.jax_ile.wrapper import (                  # noqa: E402
    JAXDistanceMarginalizedLikelihood)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_angle_marg_exact import make_synth                   # noqa: E402

D_MIN, D_MAX = 1.0, 20000.0


###
### 1. No existing default may move
###

@pytest.mark.parametrize('d_prior', ['euclidean', 'uniform'])
def test_uniform_grid_default_is_bitwise_unchanged(d_prior):
    """d_prior_range == (d_min,d_max) must reproduce the historical
    `w = w/np.sum(w)` branch bit for bit -- the driver now passes it always."""
    a = make_distance_grid(D_MIN, D_MAX, 256, d_prior)
    b = make_distance_grid(D_MIN, D_MAX, 256, d_prior, d_prior_range=(D_MIN, D_MAX))
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_adaptive_grid_default_is_bitwise_unchanged():
    kw = dict(d_peak=500.0, sigma_d=25.0, d_prior='euclidean')
    a = make_distance_grid_adaptive(D_MIN, D_MAX, **kw)
    b = make_distance_grid_adaptive(D_MIN, D_MAX, d_prior_range=(D_MIN, D_MAX), **kw)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_full_range_grid_weights_still_sum_to_one():
    _, log_w = make_distance_grid(D_MIN, D_MAX, 256, d_prior_range=(D_MIN, D_MAX))
    assert float(np.sum(np.exp(np.asarray(log_w)))) == pytest.approx(1.0, rel=1e-9)


###
### 2. A narrowed grid carries the prior mass of the box, not unity
###

def test_narrowed_grid_weights_are_the_box_prior_mass():
    lo, hi = 800.0, 3200.0
    _, log_w = make_distance_grid(lo, hi, 256, d_prior_range=(D_MIN, D_MAX))
    got = float(np.sum(np.exp(np.asarray(log_w))))
    analytic = (hi ** 3 - lo ** 3) / (D_MAX ** 3 - D_MIN ** 3)
    # 1e-2 relative: the numerator and denominator use the same n_grid rectangle
    # rule over ranges of different width, so they do not cancel exactly.
    assert got == pytest.approx(analytic, rel=1e-2)
    assert got < 0.05                       # emphatically NOT renormalized to 1


def test_narrowed_grid_without_the_prior_range_renormalizes_the_defect():
    """Power check: the historical call signature, given a narrow range, produces
    unit weight -- i.e. it moved the prior.  This is what d_prior_range prevents."""
    lo, hi = 800.0, 3200.0
    _, log_w = make_distance_grid(lo, hi, 256)          # no d_prior_range
    assert float(np.sum(np.exp(np.asarray(log_w)))) == pytest.approx(1.0, rel=1e-9)


def test_narrowed_grid_nodes_are_inside_the_box():
    """The disconnected-flag check at grid level: the nodes must actually move."""
    lo, hi = 800.0, 3200.0
    x, _ = make_distance_grid(lo, hi, 256, d_prior_range=(D_MIN, D_MAX))
    d = 1.0 / np.asarray(x)                # x = distMpcRef/d, up to distMpcRef
    assert d.min() / d.max() == pytest.approx(lo / hi, rel=1e-9)


###
### 3. Through the real marginalized likelihood
###
# The acceptance quantity here is the EVIDENCE, not the per-angle marginal.  That
# distinction is not pedantic: at an angle where the signal is weak the
# distance-marginalized lnL is carried by the prior's own d^2 mass at large d (L -> 1
# there), so narrowing the box moves it by many nats -- correctly.  What must not move
# is lnZ, and it does not, because at high amplitude the peak angle carries the whole
# integral.  So: a fixed prior-drawn angle cloud and lnZ = logsumexp(lnL) - ln N, which
# is exactly the driver's --mode prior-mc estimator.
#
# MEASURED on the configuration below (rho_mf 17.4, d* = 625 Mpc, prior [1,20000] Mpc,
# x8 multiplicative box [78,4999] = 1.6% of the prior mass), narrowed minus full-range
# lnZ at equal n_grid:
#
#     n_grid    256      512      1000      2000      4000
#     dlnZ   -3.0e-02  -4.8e-06  +2.8e-14  +0.0e+00  +2.8e-14
#
# i.e. EXACTLY ZERO once the full-range grid resolves its own integrand at all.  The
# n=256 entry is not a failure of the option: there the full-range grid is the wrong
# one (it sits +2.5e-02 from a converged reference while the box sits -5.8e-03), which
# is the resolution the narrowing exists to buy back.  Both grids retain a shared
# O(1/n_grid) offset from the discrete normalization (-6.6e-04 at n=2000), which
# cancels identically between them -- that is why the equal-n comparison is the sharp
# one and the converged-reference comparison is the loose one.

_NG_ACCEPT = 2000        # both calculations; the equal-n comparison is exact
_NG_CONVERGED = 8000     # converged-reference comparison, O(1/n) normalization offset


def _cloud(n=16, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n)),
            rng.uniform(0, np.pi, n), np.arccos(rng.uniform(-1, 1, n)),
            rng.uniform(0, 2 * np.pi, n))


def _lnZ(lnL):
    from scipy.special import logsumexp
    return float(logsumexp(np.asarray(lnL)) - np.log(len(lnL)))


@pytest.fixture(scope='module')
def _loud():
    """A high-amplitude synthetic, plus the box its own precompute implies.

    The box is derived, not guessed: the distance integrand per (angle, time bin) is
    exp(K x - R x^2/2) with x = d_ref/d, so the best-fit distance at the dominant
    sample is d_ref R/K.  One _accumulate_unit call, no gradient ascent -- cheap and
    reproducible.
    """
    from RIFT.likelihood.jax_ile.core import _accumulate_unit, JAX_INTERP_DEFAULT
    data = make_synth(scale=20.0, kappa_boost=20.0)
    a5 = _cloud()
    K, R = _accumulate_unit(data, *a5, JAX_INTERP_DEFAULT, False)
    K = np.asarray(K.real)
    R = np.maximum(np.asarray(R), 1e-30)
    snr2 = np.where(K > 0, K * K / R, -np.inf)
    i, j = np.unravel_index(int(np.argmax(snr2)), snr2.shape)
    d_star = float(data.distMpcRef) * R[i, j] / K[i, j]
    lo, hi = max(D_MIN, d_star / 8.0), min(D_MAX, d_star * 8.0)
    return data, a5, (lo, hi)


def _like(data, lo, hi, n_grid):
    return JAXDistanceMarginalizedLikelihood(
        data, lo, hi, n_grid=n_grid, d_prior_range=(D_MIN, D_MAX))


def test_the_box_is_actually_a_narrowing(_loud):
    """Guard on the fixture: if the x8 window ever clips to the full range, every
    test below would pass while measuring nothing."""
    _, _, (lo, hi) = _loud
    mass = (hi ** 3 - lo ** 3) / (D_MAX ** 3 - D_MIN ** 3)
    assert mass < 0.05


def test_ACCEPTANCE_narrowing_leaves_the_evidence_alone(_loud):
    """THE acceptance test on this arm: same n_grid, narrowed vs full range."""
    data, a5, (lo, hi) = _loud
    lnz_full = _lnZ(_like(data, D_MIN, D_MAX, _NG_ACCEPT).log_likelihood(*a5))
    lnz_box = _lnZ(_like(data, lo, hi, _NG_ACCEPT).log_likelihood(*a5))
    assert abs(lnz_box - lnz_full) < 1e-10, \
        "narrowing moved lnZ by {:.3e} nats".format(lnz_box - lnz_full)


def test_ACCEPTANCE_narrowed_evidence_matches_a_converged_reference(_loud):
    """And the shared residual is the discrete normalization, not the box: a
    converged full-range calculation is within 1e-3 nats of the narrowed one."""
    data, a5, (lo, hi) = _loud
    lnz_ref = _lnZ(_like(data, D_MIN, D_MAX, _NG_CONVERGED).log_likelihood(*a5))
    lnz_box = _lnZ(_like(data, lo, hi, _NG_ACCEPT).log_likelihood(*a5))
    assert abs(lnz_box - lnz_ref) < 1e-3


def test_the_renormalizing_call_moves_the_evidence_by_the_prior_mass(_loud):
    """Power check: the same box through the PRE-CHANGE call signature (no
    d_prior_range) shifts every lnL, and hence lnZ, by ln(1/prior mass of the box) --
    MEASURED +4.1594 nats here.  Without this, the acceptance test above could be
    passing on an implementation that renormalized and cancelled its own error."""
    data, a5, (lo, hi) = _loud
    good = np.asarray(_like(data, lo, hi, _NG_ACCEPT).log_likelihood(*a5))
    bad = np.asarray(JAXDistanceMarginalizedLikelihood(
        data, lo, hi, n_grid=_NG_ACCEPT).log_likelihood(*a5))
    _, log_w = make_distance_grid(lo, hi, _NG_ACCEPT, d_prior_range=(D_MIN, D_MAX))
    expected = -np.log(float(np.sum(np.exp(np.asarray(log_w)))))
    assert np.allclose(bad - good, expected, atol=1e-8)
    assert expected > 1.0                # several nats, not a rounding difference
    assert abs(_lnZ(bad) - _lnZ(good) - expected) < 1e-8


###
### 4. Driver wiring
###

_DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'bin', 'integrate_likelihood_extrinsic_jax')


@pytest.mark.skipif(not os.path.exists(_DRIVER), reason='JAX driver not in this tree')
def test_driver_resolves_and_forwards_the_box():
    """The driver must (a) define the option, (b) hand the marginalized
    likelihoods the SAMPLED range as d_min/d_max and the PHYSICAL range as
    d_prior_range, and (c) not leave --limit-distance in the accepted-but-ignored
    set, which is how a flag ends up parsing and doing nothing here."""
    with open(_DRIVER) as f:
        src = f.read()
    assert 'g.add_option("--limit-distance"' in src
    assert 'def resolve_distance_limit(opts):' in src
    assert src.count('d_prior_range=(opts.d_min, opts.d_max)') == 4
    assert 'like_data, d_lo, d_hi' in src
    assert '"--d-prior", "--limit-distance",' in src      # in the `implemented` set


@pytest.mark.skipif(not os.path.exists(_DRIVER), reason='JAX driver not in this tree')
def test_driver_prior_normalization_stays_on_d_min_d_max():
    """log_prior() restricts SUPPORT to the box but must keep normalizing on
    [d_min,d_max]; run_prior_mc() must correct for its restricted proposal."""
    with open(_DRIVER) as f:
        src = f.read()
    assert 'inb = inb & (dist >= d_lo) & (dist <= d_hi)' in src
    assert 'dmin3, dmax3 = opts.d_min ** 3, opts.d_max ** 3' in src
    assert 'logw = lnL - log_distance_box_correction(opts, with_distance)' in src


def test_box_correction_is_exactly_zero_without_the_option():
    """The historical prior-MC path must be untouched: `lnL - 0.0` is bitwise lnL."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(_DRIVER)))
    import importlib.util
    spec = importlib.util.spec_from_loader('_jaxdrv', loader=None)
    mod = importlib.util.module_from_spec(spec)
    with open(_DRIVER) as f:
        src = f.read()
    # exec only the two functions under test, with their numpy dependency
    ns = {'np': np}
    start = src.index('def resolve_distance_limit(opts):')
    end = src.index('def sample_prior(n, opts, rng, with_distance):')
    exec(compile(src[start:end], _DRIVER, 'exec'), ns)          # noqa: S102
    mod.__dict__.update(ns)

    class _O(object):
        d_min, d_max, limit_distance = 1.0, 20000.0, None
    assert ns['resolve_distance_limit'](_O()) == (1.0, 20000.0)
    corr = ns['log_distance_box_correction'](_O(), True)
    assert corr == 0.0 and isinstance(corr, float)
    lnL = np.array([-3.0, 0.0, 12.5, -np.inf])
    assert np.array_equal(lnL - corr, lnL) or np.all(
        (lnL - corr == lnL) | np.isnan(lnL))

    _O.limit_distance = '800,3200'
    assert ns['resolve_distance_limit'](_O()) == (800.0, 3200.0)
    assert ns['log_distance_box_correction'](_O(), True) == pytest.approx(
        np.log((20000.0 ** 3 - 1.0 ** 3) / (3200.0 ** 3 - 800.0 ** 3)))
    # inert when distance is marginalized out (no explicit distance proposal)
    assert ns['log_distance_box_correction'](_O(), False) == 0.0
