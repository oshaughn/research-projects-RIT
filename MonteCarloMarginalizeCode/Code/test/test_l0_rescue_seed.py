#!/usr/bin/env python
"""
Regression tests for the L0 auto-rescue's WARM SEED
(RIFT/integrators/mcsamplerAdaptiveVolume.py, bin/integrate_likelihood_extrinsic_batchmode).

Background (the two defects these tests lock down).  When a high-amplitude extrinsic pass
collapses to n_eff ~ 1, --sampler-warmstart-retry-neff re-runs it warm, seeded from that
same pass's own highest-likelihood samples.  Measured on zero-noise injections at a fixed
intrinsic point, the seed it built was not fit to define a live volume:

    rho_net 102.8   seed 5 points, affine rank 2-4 of 6   ->  V ~ 3e-06,  5/12 replicates
    rho_net 146.8   seed 2 points, affine rank 0 of 6     ->  V ~ 9e-36, ESS ~ 1

  1. THE GUARD WAS A COUNT.  `if len(_seed) < 2: puff` -- but a 2-to-5 point seed passes
     that and is still rank-deficient in 6 dimensions, so the warm start contracts onto a
     degenerate subspace, terminates in one cycle, and reports an excellent n_eff while lnZ
     is a lower bound.  The rank was already being computed and printed by the [AV COLLAPSE]
     report; nothing acted on it.  n points span at most n-1 affine dimensions, so rank
     subsumes the count.

  2. THE POINTS WERE THERE ALL ALONG.  integrate_log's fair draw takes
     n_extr = min(n_extr, 1.5*eff_samp, 1.5*neff) rows WITH REPLACEMENT and rebinds every
     self._rvs key to that subset -- a resample built for EXPORT -- and the rescue read
     _rvs.  On a collapsed pass eff_samp ~ 1, so the seed was one row ("Fairdraw size : 1"
     in the rho_net 146.8 logs) while the live set held 1000.  Sampling with replacement is
     also why a "2-point seed" had affine rank 0: two copies of one point.  This is the
     reason widening --sampler-sequential-warmstart-deltalnL did nothing even at 20x -- no
     window can admit rows that are no longer there.

  3. THE PUFF WIDTH KNEW NOTHING ABOUT THE POSTERIOR.  It was a hardcoded 1/200 of each
     PRIOR range, while the posterior narrows as 1/rho.  On a known-lnZ 6-D target that
     truncated by 0.8-8.3 nats with a healthy-looking ESS.

The requirement is not "does not crash": a seed that cannot span the space must be repaired
BEFORE it is handed to the sampler, and a seed that can span it must be left alone.
"""

import os

import numpy as np
import pytest

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV
from RIFT.integrators.mcsamplerAdaptiveVolume import (
    build_warm_seed,
    live_volume_collapse_verdict,
    seed_affine_rank,
    warm_seed_scale_from_finite_points,
)

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
NDIM = len(NAMES)
LO = np.zeros(NDIM)
HI = np.ones(NDIM)
AX = list(range(NDIM))

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')


def _sampler(n_chunk=10000, limits=None):
    """Bound to the ACTIVE backend exactly as the ILE does; see test_av_empty_live_volume."""
    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = mcsamplerAV.xpy_default
    s.identity_convert = mcsamplerAV.identity_convert
    for indx, name in enumerate(NAMES):
        lo, hi = (0.0, 1.0) if limits is None else limits[indx]
        s.add_parameter(name, pdf=None, left_limit=lo, right_limit=hi,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    return s


def _peaked(rho, x0=None, widths=None):
    """6-D Gaussian at lnL scale rho^2/2, with the float64 underflow of the real code."""
    x0 = 0.5 * np.ones(NDIM) if x0 is None else np.asarray(x0, dtype=float)
    w = (0.5 / rho) * np.ones(NDIM) if widths is None else np.asarray(widths, dtype=float)
    lnLmax = 0.5 * rho ** 2

    def lnL(*args, **kwargs):
        x = np.array([np.asarray(a, dtype=float).ravel() for a in args]).T
        out = lnLmax - 0.5 * np.sum(((x - x0) / w) ** 2, axis=-1)
        return np.where(out > lnLmax - 745.0, out, -np.inf)
    return lnL


###
### 1. seed_affine_rank -- the ONE definition the guard and the diagnostic share
###

def test_duplicated_points_have_affine_rank_zero():
    """The rho_net 146.8 seed: 'two points' that the fair draw drew twice from one row."""
    p = np.full((2, NDIM), 0.5)
    assert seed_affine_rank(p, LO, HI, AX)[0] == 0


def test_collinear_points_span_only_a_line_however_many_there_are():
    t = np.linspace(0.3, 0.7, 500)
    p = np.full((500, NDIM), 0.5)
    p[:, 2] = t
    assert seed_affine_rank(p, LO, HI, AX)[0] == 1


@pytest.mark.parametrize('n,expect', [(2, 1), (4, 3), (NDIM, NDIM - 1), (NDIM + 1, NDIM)])
def test_n_points_span_at_most_n_minus_one_affine_dimensions(n, expect):
    """Which is why rank subsumes the count rule it replaces."""
    rng = np.random.RandomState(7)
    p = 0.5 + 0.05 * rng.randn(n, NDIM)
    assert seed_affine_rank(p, LO, HI, AX)[0] == expect


def test_rank_ignores_out_of_box_rows():
    """The grid only spans the box, and an unclipped puff does reach the builder."""
    t = np.linspace(0.4, 0.6, 40)
    inbox = np.full((40, NDIM), 0.5)
    inbox[:, 0] = t                                   # a LINE in the box: rank 1
    outside = np.random.RandomState(11).uniform(1.5, 2.5, size=(40, NDIM))
    rank, n_in = seed_affine_rank(np.vstack([inbox, outside]), LO, HI, AX)
    assert (rank, n_in) == (1, 40)


def test_rank_is_unit_free_across_wildly_different_parameter_scales():
    """A distance in Mpc and an angle in radians must not get different tolerances."""
    lo = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
    hi = np.array([6.3, 3.1, 6.3, 3.1, 3.1, 5000.0])
    rng = np.random.RandomState(3)
    box = hi - lo
    p = lo + box * (0.5 + 0.01 * rng.randn(200, NDIM))
    assert seed_affine_rank(p, lo, hi, AX)[0] == NDIM
    flat = p.copy()
    flat[:, 5] = lo[5] + 0.5 * box[5]                 # kill the Mpc direction only
    assert seed_affine_rank(flat, lo, hi, AX)[0] == NDIM - 1


def test_the_grid_builder_records_the_same_rank_the_guard_tests():
    """One definition, two callers: a guard that puffed on a different rank than the
    report measures would either puff a healthy seed or ship a flagged one."""
    s = _sampler()
    s.setup()
    rng = np.random.RandomState(4)
    plane = np.full((300, NDIM), 0.5)
    plane[:, :2] = 0.5 + 0.02 * rng.randn(300, 2)     # rank 2 in 6 dimensions
    warm = s.bootstrap_from_samples(plane, cover_frac=0.0)
    assert warm['n_seed_rank'] == seed_affine_rank(plane, LO, HI, AX)[0] == 2
    assert warm['n_seed_dim'] == NDIM


###
### 2. build_warm_seed -- rank, not count, decides; and the real points are KEPT
###

def _core_and_lnL(pts, lnLmax=10700.0, spread=1.0):
    """A points/lnL pair whose top `len(pts)` rows are the seed core."""
    lnL = lnLmax - spread * np.arange(len(pts), dtype=float)
    return np.asarray(pts, dtype=float), lnL


@pytest.mark.parametrize('n_core,rank_in', [(1, 0), (2, 0), (2, 1), (5, 2), (5, 4)])
def test_a_rank_deficient_core_is_puffed_to_full_rank(n_core, rank_in):
    """The measured failures: 1-5 points at rank 0-4 of 6.  Every one passed `len < 2`."""
    rng = np.random.RandomState(5)
    core = np.full((n_core, NDIM), 0.5)
    if rank_in:                                        # spread over exactly rank_in axes
        core[:, :rank_in] += 0.01 * rng.randn(n_core, rank_in)
        core[rank_in:] = core[rank_in:]                # (no-op; keeps the intent explicit)
    core = core[:n_core]
    assert seed_affine_rank(core, LO, HI, AX)[0] <= rank_in
    pts, lnL = _core_and_lnL(core, spread=0.5)         # all within deltalnL of the peak
    seed, info = build_warm_seed(pts, lnL, LO, HI, AX, deltalnL=15.0)
    assert info['puffed'] is True
    assert info['rank_core'] < NDIM
    assert info['rank_final'] == NDIM, info
    # ... and the repaired seed is not flagged by the very report that named the defect
    collapsed, reasons = live_volume_collapse_verdict(
        5000, NDIM, ess=40.0, khat=0.5, n_warm_seed=info['n_seed'],
        n_warm_seed_rank=info['rank_final'], n_warm_seed_dim=NDIM)
    assert collapsed is False, reasons


def test_the_old_count_rule_would_have_passed_every_one_of_those():
    """Pins WHY the guard had to change: `len(_seed) < 2` sees none of these."""
    for n_core in (2, 3, 5):
        core = np.full((n_core, NDIM), 0.5)
        core[:, 0] += 1e-3 * np.arange(n_core)         # a line: rank 1 in 6 dimensions
        assert len(core) >= 2, 'the old rule declines to puff'
        assert seed_affine_rank(core, LO, HI, AX)[0] < NDIM, 'yet it cannot define a volume'


def test_a_full_rank_core_is_left_completely_alone():
    """The guard must not cry wolf: d+1 independent points DO define a volume in d."""
    rng = np.random.RandomState(6)
    core = 0.5 + 0.01 * rng.randn(NDIM + 1, NDIM)
    pts, lnL = _core_and_lnL(core, spread=0.5)
    seed, info = build_warm_seed(pts, lnL, LO, HI, AX, deltalnL=15.0)
    assert info['puffed'] is False
    assert info['rank_core'] == NDIM
    assert len(seed) == NDIM + 1
    assert np.allclose(np.sort(seed, axis=0), np.sort(core, axis=0))


def test_points_outside_the_deltalnL_window_are_not_part_of_the_core():
    rng = np.random.RandomState(8)
    pts = 0.5 + 0.01 * rng.randn(50, NDIM)
    lnL = np.full(50, 100.0)
    lnL[3:] = 0.0                                       # only 3 rows within 15 nats
    _, info = build_warm_seed(pts, lnL, LO, HI, AX, deltalnL=15.0)
    assert info['n_core'] == 3


def test_the_puff_AUGMENTS_the_seed_rather_than_replacing_it():
    """The real points are the only direct evidence of where the peak is; a real point
    outside the puff must be able to widen the seeded volume, and VARAHA can only
    contract afterwards."""
    core = np.full((3, NDIM), 0.5)
    core[:, 0] = [0.20, 0.50, 0.80]                     # rank 1, and WIDE
    pts, lnL = _core_and_lnL(core, spread=0.5)
    seed, info = build_warm_seed(pts, lnL, LO, HI, AX, deltalnL=15.0,
                                 puff_scale='fixed', puff_width_frac=1e-3, puff_factor=1.0)
    assert info['puffed'] is True
    for row in core:
        assert np.any(np.all(np.isclose(seed, row), axis=1)), \
            'the original seed point {} was discarded'.format(row)
    # and the union, not the puff, sets the extent the grid is built from
    assert seed[:, 0].min() <= 0.20 and seed[:, 0].max() >= 0.80


def test_the_puff_is_clipped_into_the_box():
    """An unclipped Gaussian about a peak near an edge loses most of its rows in the grid
    builder, so the seed the sampler sees is not the one measured for rank here."""
    core = np.full((2, NDIM), 0.02)                     # hard against the lower edge
    pts, lnL = _core_and_lnL(core, spread=0.5)
    seed, info = build_warm_seed(pts, lnL, LO, HI, AX, deltalnL=15.0,
                                 puff_scale='fixed', puff_width_frac=0.05)
    assert info['puffed'] is True
    assert np.all(seed >= LO) and np.all(seed <= HI)
    assert seed_affine_rank(seed, LO, HI, AX)[1] == len(seed), 'rows were lost out of box'


def test_a_puffed_seed_builds_a_live_volume_instead_of_a_sliver():
    """End to end against the measured failure: rank 0 of 6 gave V ~ 9e-36."""
    s_bad, s_good = _sampler(), _sampler()
    s_bad.setup(); s_good.setup()
    core = np.full((2, NDIM), 0.5)                      # the duplicated-row seed
    pts, lnL = _core_and_lnL(core, spread=0.5)
    warm_bad = s_bad.bootstrap_from_samples(core, cover_frac=0.0)
    seed, info = build_warm_seed(pts, lnL, LO, HI, AX, deltalnL=15.0,
                                 puff_scale='fixed', puff_width_frac=0.005, puff_factor=2.0)
    warm_good = s_good.bootstrap_from_samples(seed, cover_frac=0.0)
    assert warm_bad['n_seed_rank'] == 0 and warm_good['n_seed_rank'] == NDIM
    assert warm_good['V'] > 1e6 * warm_bad['V'], \
        'V {:.3e} -> {:.3e}'.format(warm_bad['V'], warm_good['V'])


###
### 3. the puff WIDTH must track the posterior, which the prior range cannot
###

def test_the_scale_estimator_recovers_the_posterior_width_from_the_underflow_shell():
    """cov(finite points) * (d+2)/(2D) inverts the uniform-in-ellipsoid level set.

    Recovered sigma/true was 1.01-1.23 per axis on the 6-replicate study behind the
    default; this pins the mechanism at a tolerance that leaves room for MC scatter.
    """
    rng = np.random.RandomState(20260811)
    sig = np.array([0.004, 0.006, 0.003, 0.008, 0.005, 0.004])
    x0 = 0.5 * np.ones(NDIM)
    n = 1000000
    X = rng.uniform(0.0, 1.0, size=(n, NDIM))
    lnLmax = 10700.0
    lnL = lnLmax - 0.5 * np.sum(((X - x0) / sig) ** 2, axis=1)
    lnL = np.where(lnL > lnLmax - 745.0, lnL, -np.inf)
    assert np.sum(np.isfinite(lnL)) > 50, 'the test target must actually underflow'
    cov = warm_seed_scale_from_finite_points(X, lnL, LO, HI, AX)
    assert cov is not None
    est = np.sqrt(np.diag(cov))                          # box is unit, so scaled == raw
    assert np.all(est / sig > 0.6) and np.all(est / sig < 1.7), (est / sig).tolist()


def test_the_scale_estimator_declines_rather_than_guess_from_too_few_points():
    rng = np.random.RandomState(9)
    X = rng.uniform(0.4, 0.6, size=(5, NDIM))
    assert warm_seed_scale_from_finite_points(X, np.arange(5.0), LO, HI, AX) is None


def test_auto_falls_back_to_the_fixed_width_and_still_reaches_full_rank():
    core = np.full((2, NDIM), 0.5)
    pts, lnL = _core_and_lnL(core, spread=0.5)
    seed, info = build_warm_seed(pts, lnL, LO, HI, AX, deltalnL=15.0, puff_scale='auto')
    assert info['puff_scale'] == 'fixed', 'two points cannot yield a 6-D covariance'
    assert info['rank_final'] == NDIM


def test_the_fixed_width_still_reproduces_the_historical_puff_exactly():
    """A knob that cannot restore the previous behaviour is not a knob."""
    core = np.full((1, NDIM), 0.5)
    pts, lnL = _core_and_lnL(core)
    seed, info = build_warm_seed(pts, lnL, LO, HI, AX, deltalnL=15.0, n_puff=2000,
                                 puff_scale='fixed', puff_width_frac=1.0 / 200,
                                 puff_factor=1.0, seed=0)
    assert info['n_puff'] == 2000
    ref = np.random.RandomState(0).normal(core[0], (HI - LO) / 200.0, size=(2000, NDIM))
    # same width, to within the sampling scatter of 2000 draws
    assert np.allclose(seed[1:].std(axis=0), ref.std(axis=0), rtol=0.15)


def test_widening_the_puff_without_limit_is_not_safe():
    """Documents the measured non-monotonicity: a seed an order of magnitude too wide is a
    cold start in all but name (V -> O(1)), which is the failure the rescue exists to fix."""
    s_ok, s_wide = _sampler(), _sampler()
    s_ok.setup(); s_wide.setup()
    core = np.full((2, NDIM), 0.5)
    pts, lnL = _core_and_lnL(core, spread=0.5)
    seed_ok, _ = build_warm_seed(pts, lnL, LO, HI, AX, puff_scale='fixed',
                                 puff_width_frac=0.005, puff_factor=2.0)
    seed_wide, _ = build_warm_seed(pts, lnL, LO, HI, AX, puff_scale='fixed',
                                   puff_width_frac=0.005, puff_factor=32.0)
    V_ok = s_ok.bootstrap_from_samples(seed_ok, cover_frac=0.0)['V']
    V_wide = s_wide.bootstrap_from_samples(seed_wide, cover_frac=0.0)['V']
    assert V_wide > 100 * V_ok, 'V {:.3e} -> {:.3e}'.format(V_ok, V_wide)


###
### 4. the fair draw must not be able to starve the seed
###

def test_the_retained_points_survive_a_fair_draw_that_keeps_one_row():
    """The root cause: _rvs is REBOUND to min(n_extr, 1.5*eff_samp, 1.5*neff) rows drawn
    WITH REPLACEMENT, so on the collapsed pass the rescue exists for it holds one row --
    while the live set holds a thousand."""
    np.random.seed(20260811)
    s = _sampler(20000)
    res = s.integrate_log(_peaked(60.0), *NAMES, nmax=400000, neff=8, n=20000,
                          no_protect_names=True, verbose=False,
                          igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    n_rvs = len(np.asarray(mcsamplerAV.identity_convert(s._rvs['log_integrand'])).ravel())
    reserve = s._warm_seed_reserve
    assert reserve is not None
    assert reserve['n_retained'] >= n_rvs
    assert len(reserve['lnL']) == len(reserve['X']) >= n_rvs
    assert list(reserve['params_ordered']) == list(s.params_ordered)
    assert np.all(np.isfinite(reserve['lnL'])), 'the reserve is the RETAINED (finite) set'


def test_the_reserve_carries_the_peak_the_seed_is_built_around():
    """A uniform subsample can drop the best row, and the seed is defined relative to it."""
    np.random.seed(20260811)
    s = _sampler(20000)
    s.n_warm_seed_reserve = 50                     # force the subsample path
    s.integrate_log(_peaked(40.0), *NAMES, nmax=200000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    r = s._warm_seed_reserve
    assert r is not None and len(r['lnL']) <= 51   # cap, plus the appended peak row
    assert len(r['lnL']) >= NDIM + 1


def test_a_reserve_from_a_previous_point_cannot_leak_into_the_next():
    """Cleared on ENTRY, so 'present' always means 'this pass wrote it'.  Otherwise a pass
    that raises leaves the previous point's peak for the next point's rescue to seed from."""
    np.random.seed(20260811)
    s = _sampler(20000)
    s.integrate_log(_peaked(40.0), *NAMES, nmax=200000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    assert s._warm_seed_reserve is not None
    stale = s._warm_seed_reserve

    def _explode(*args, **kwargs):
        raise RuntimeError('waveform generation failed')
    with pytest.raises(Exception):
        s.integrate_log(_explode, *NAMES, nmax=200000, neff=8, n=20000,
                        no_protect_names=True, verbose=False)
    assert s._warm_seed_reserve is not stale
    assert s._warm_seed_reserve is None


###
### 4c. the PORTFOLIO must build its own reserve -- no member ever will
###

def _portfolio(n_chunk=10000):
    """An AV+GMM portfolio built the way the ILE builds one: member INSTANCES, then
    add_parameter on the portfolio, which forwards to every member in the same order."""
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPF
    import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
    members = [mcsamplerAV.MCSampler(n_chunk=n_chunk), mcsamplerEnsemble.MCSampler()]
    s = mcsamplerPF.MCSampler(portfolio=members)
    pdf = np.vectorize(lambda x: 1.0)
    for name in NAMES:
        s.add_parameter(name, pdf, prior_pdf=pdf, left_limit=0.0, right_limit=1.0,
                        adaptive_sampling=True)
    s.setup()          # initializes portfolio_breakpoints/weights; integrate_log assumes it
    return s


def test_the_portfolio_never_routes_through_a_member_integrate_log():
    """The premise of the portfolio reserve, pinned so it cannot silently stop being true.

    mcsamplerPortfolio drives members through draw_simplified(); if it ever started calling
    member.integrate_log() the members would build their own reserves and the portfolio-level
    one could be reconsidered.  Until then the portfolio is the ONLY place its aggregate
    retained set exists.
    """
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPF
    import inspect
    src = inspect.getsource(mcsamplerPF.MCSampler.integrate_log)
    assert 'draw_simplified' in src
    assert 'member.integrate_log' not in src and '.integrate_log(' not in src.replace(
        'self.integrate_log(', ''), 'members are now driven through integrate_log'


def test_a_portfolio_pass_leaves_a_reserve_of_its_aggregate_retained_points():
    """Without this the L0 rescue on a portfolio reads the pruned + fair-drawn _rvs."""
    np.random.seed(20260811)
    s = _portfolio(20000)
    s.integrate_log(_peaked(40.0), *NAMES, nmax=300000, neff=8, n=20000,
                    no_protect_names=True, verbose=False, save_intg=True,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    r = s._warm_seed_reserve
    assert r is not None, 'the portfolio built no reserve; the rescue would starve on _rvs'
    n_rvs = len(np.asarray(mcsamplerAV.identity_convert(s._rvs['log_integrand'])).ravel())
    assert r['n_retained'] >= n_rvs
    assert len(r['lnL']) == len(r['X']) >= n_rvs, \
        'reserve ({}) is no larger than the fair-drawn _rvs ({})'.format(len(r['lnL']), n_rvs)
    assert list(r['params_ordered']) == list(s.params_ordered)
    assert r['X'].shape[1] == NDIM


def test_the_portfolio_reserve_is_taken_before_pruning_and_the_fair_draw():
    """Order matters: taken after either step it would carry the same starved subset."""
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPF
    import inspect
    src = inspect.getsource(mcsamplerPF.MCSampler.integrate_log)
    i_res = src.index('make_warm_seed_reserve')
    assert i_res < src.index("Clean out the _rvs arrays"), 'reserve taken after pruning'
    assert i_res < src.index('if bFairdraw'), 'reserve taken after the fair draw'


def test_a_portfolio_reserve_cannot_leak_from_one_point_to_the_next():
    np.random.seed(20260811)
    s = _portfolio(20000)
    s.integrate_log(_peaked(40.0), *NAMES, nmax=300000, neff=8, n=20000,
                    no_protect_names=True, verbose=False, save_intg=True,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    assert s._warm_seed_reserve is not None

    def _explode(*args, **kwargs):
        raise RuntimeError('waveform generation failed')
    with pytest.raises(Exception):
        s.integrate_log(_explode, *NAMES, nmax=300000, neff=8, n=20000,
                        no_protect_names=True, verbose=False, save_intg=True)
    assert s._warm_seed_reserve is None


###
### 5. the ILE must actually use all of this
###

@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_ile_no_longer_guards_the_puff_with_a_row_count():
    with open(_ILE) as f:
        src = f.read()
    # the CODE, not the comment that explains why it went (which quotes it verbatim)
    assert 'if len(_seed) < 2:' not in src, 'the count rule is back'
    assert 'build_warm_seed' in src, 'the rescue does not go through the rank-tested builder'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_ile_prefers_the_reserve_over_the_fair_drawn_rvs():
    with open(_ILE) as f:
        src = f.read()
    i = src.index('build_warm_seed')
    block = src[max(0, i - 2500):i]
    assert '_warm_seed_reserve' in block, \
        'the rescue still seeds from _rvs, which the fair draw has already truncated'
    assert 'params_ordered' in block, 'the reserve is used without checking its column order'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_puff_width_is_configurable_and_defaults_are_the_measured_ones():
    with open(_ILE) as f:
        src = f.read()
    for opt in ('--sampler-l0-rescue-puff-scale', '--sampler-l0-rescue-puff-width-frac',
                '--sampler-l0-rescue-puff-factor'):
        assert opt in src, 'missing {}'.format(opt)
    i = src.index('--sampler-l0-rescue-puff-factor')
    assert 'default=2.0' in src[i:i + 200], 'the measured optimum is not the default'
