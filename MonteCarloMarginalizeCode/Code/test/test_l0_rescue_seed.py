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


def test_a_second_portfolio_pass_reserve_contains_only_second_pass_draws():
    """A warm retry reuses member proposals, not the cold pass's aggregate sample cache."""
    np.random.seed(20260812)
    s = _portfolio(256)

    def _marked(value):
        return lambda *args: np.full(np.asarray(args[0]).shape, value, dtype=float)

    kwargs = dict(nmax=256, neff=1, n=256, no_protect_names=True,
                  verbose=False, save_intg=True)
    s.integrate_log(_marked(11.0), *NAMES, **kwargs)
    assert np.all(s._warm_seed_reserve['lnL'] == 11.0)

    s.integrate_log(_marked(22.0), *NAMES, **kwargs)
    reserve = s._warm_seed_reserve
    assert reserve['n_retained'] == s.ntotal == 256
    assert np.all(reserve['lnL'] == 22.0), \
        'the warm reserve contains cold-pass rows from the retained portfolio cache'


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
### 4d. the cap must not throw away the thing it is capping
###

def test_the_cap_keeps_every_finite_row_when_they_are_rare():
    """The portfolio's _rvs holds EVERY draw, and on a collapsed pass almost all are -inf.

    Uniformly subsampling all rows keeps the finite ones in proportion -- which is to say
    almost none.  Measured before the fix: 10 finite rows among 1,000,000 at a 20,000 cap
    survived as 2 (the forced peak plus one lucky draw), leaving build_warm_seed a rank-0
    core and the scale estimator nothing to work with.
    """
    from RIFT.integrators.mcsamplerAdaptiveVolume import make_warm_seed_reserve
    n = 1000000
    lnL = np.full(n, -np.inf)
    idx = np.random.RandomState(1).choice(n, 10, replace=False)
    lnL[idx] = 10700.0 - 0.5 * np.arange(10)
    X = np.random.RandomState(2).uniform(size=(n, NDIM))
    r = make_warm_seed_reserve(X, lnL, NAMES, n_max=20000)
    assert int(np.sum(np.isfinite(r['lnL']))) == 10, 'finite rows were thinned by the cap'
    assert r['n_retained'] == n and r['n_finite'] == 10
    # and what the rescue builds from it is now usable
    seed, info = build_warm_seed(r['X'], r['lnL'], LO, HI, AX, deltalnL=15.0)
    assert info['n_core'] >= 10 or info['rank_final'] == NDIM


def test_the_cap_still_binds_and_still_keeps_the_peak_on_a_healthy_run():
    from RIFT.integrators.mcsamplerAdaptiveVolume import make_warm_seed_reserve
    n = 200000
    lnL = 10700.0 - np.random.RandomState(3).exponential(50.0, size=n)
    X = np.random.RandomState(4).uniform(size=(n, NDIM))
    r = make_warm_seed_reserve(X, lnL, NAMES, n_max=5000)
    assert 5000 <= len(r['lnL']) <= 5001, 'cap not honoured'
    assert np.isclose(r['lnL'].max(), lnL.max()), 'the peak row was dropped'


def test_building_the_reserve_does_not_disturb_the_global_random_stream():
    """It is built unconditionally, including when --sampler-warmstart-retry-neff is unset.

    Drawing its subsample from the global numpy stream advanced that stream before the fair
    draw, the exported posterior, and every later event -- so an opt-in rescue that is
    switched OFF changed a seeded run's output.
    """
    from RIFT.integrators.mcsamplerAdaptiveVolume import make_warm_seed_reserve
    n = 60000                                   # over any sane cap, so the draw really happens
    lnL = 10700.0 - np.random.RandomState(5).exponential(50.0, size=n)
    X = np.random.RandomState(6).uniform(size=(n, NDIM))
    np.random.seed(1234)
    expected = np.random.rand(3)
    np.random.seed(1234)
    r = make_warm_seed_reserve(X, lnL, NAMES, n_max=20000)
    assert len(r['lnL']) > 20000 - 1, 'the subsample path did not run; test proves nothing'
    assert np.array_equal(np.random.rand(3), expected), \
        'the reserve consumed the global RNG'


def test_the_reserve_subsample_is_reproducible():
    """A private stream is only an improvement if it is deterministic."""
    from RIFT.integrators.mcsamplerAdaptiveVolume import make_warm_seed_reserve
    n = 60000
    lnL = 10700.0 - np.random.RandomState(7).exponential(50.0, size=n)
    X = np.random.RandomState(8).uniform(size=(n, NDIM))
    a = make_warm_seed_reserve(X, lnL, NAMES, n_max=20000)
    b = make_warm_seed_reserve(X, lnL, NAMES, n_max=20000)
    assert np.array_equal(a['lnL'], b['lnL'])


###
### 4f. the EXACT finite-population total, captured before the cap can perturb it
###

def test_the_reserve_records_the_exact_pre_cap_weight_total():
    """A capped reserve is a Horvitz-Thompson sample: unbiased in the LINEAR total, but its
    logarithm moves in discrete jumps depending on whether the subsample caught the rows that
    carry the weight.  The exact total is the one thing a bounded record cannot rebuild
    afterwards, so it is captured at build time."""
    from RIFT.integrators.mcsamplerAdaptiveVolume import make_warm_seed_reserve
    n = 200000
    lnL = np.full(n, -50.0)
    lnL[[7, 9999]] = 0.0                       # two EQUAL dominant rows
    X = np.zeros((n, NDIM))
    zeros = np.zeros(n)
    r = make_warm_seed_reserve(X, lnL, NAMES, n_max=2000,
                               log_joint_prior=zeros, log_joint_s_prior=zeros)
    exact = np.log(np.sum(np.exp(lnL)))
    assert r['ln_sum_w_finite'] is not None
    assert abs(r['ln_sum_w_finite'] - exact) < 1e-9, \
        'the recorded total is not the full finite-population total'
    assert len(r['lnL']) <= 2001, 'the cap did not actually bind; test proves nothing'


def test_the_exact_total_does_not_move_with_the_cap():
    """The failure this prevents: same population, different cap -> same lnZ."""
    from RIFT.integrators.mcsamplerAdaptiveVolume import make_warm_seed_reserve
    n = 200000
    lnL = np.full(n, -50.0)
    lnL[[7, 9999]] = 0.0
    X = np.zeros((n, NDIM))
    zeros = np.zeros(n)
    uncapped = make_warm_seed_reserve(X, lnL, NAMES, n_max=0,
                                      log_joint_prior=zeros, log_joint_s_prior=zeros)
    capped = make_warm_seed_reserve(X, lnL, NAMES, n_max=2000,
                                    log_joint_prior=zeros, log_joint_s_prior=zeros)
    assert abs(uncapped['ln_sum_w_finite'] - capped['ln_sum_w_finite']) < 1e-9
    # ... whereas the capped SUBSAMPLE misses the second dominant row, which is the log(2)
    # error that would have driven the gate
    lw_kept = capped['lnL'] + capped['log_joint_prior'] - capped['log_joint_s_prior']
    n_dom_kept = int(np.sum(lw_kept > -1.0))
    assert n_dom_kept < 2, 'this cap happened to keep both; the scenario is not exercised'


def test_the_exact_total_is_absent_rather_than_wrong_without_the_prior_components():
    from RIFT.integrators.mcsamplerAdaptiveVolume import make_warm_seed_reserve
    r = make_warm_seed_reserve(np.zeros((10, NDIM)), np.zeros(10), NAMES, n_max=0)
    assert r['ln_sum_w_finite'] is None


###
### 4b. the reject gate must not read a fair-draw artifact as evidence of lost mass
###

def test_a_fair_drawn_lnZ_sits_above_the_retained_set_lnZ_by_a_predictable_amount():
    """WHY the reject gate had to stop reading _rvs.

    _lnZ_of_rvs forms logsumexp(w) - log(n).  The fair draw resamples n rows proportional to
    w, so its rows cluster at the TOP of the weight distribution and the estimate lands near
    max(w) rather than mean(w) -- high by about log(n_retained / eff_samp).  The gate compared
    a 1-row cold reading against a 5-row warm one and read the difference as lost mass.
    """
    rng = np.random.RandomState(20260811)
    n = 1000
    lw = np.concatenate([[0.0], -30.0 - 5.0 * rng.rand(n - 1)])   # one dominant weight
    w = np.exp(lw - lw.max())
    eff = w.sum() / w.max()
    lnZ_all = np.log(np.mean(np.exp(lw)))
    drawn = rng.choice(n, size=1, replace=True, p=w / w.sum())
    lnZ_fair = np.log(np.mean(np.exp(lw[drawn])))
    assert lnZ_fair > lnZ_all, 'the fair-drawn reading must be the HIGH one'
    predicted = np.log(n / eff)
    assert abs((lnZ_fair - lnZ_all) - predicted) < 0.5, \
        'gap {:.2f} should track log(n/eff_samp) = {:.2f}'.format(lnZ_fair - lnZ_all, predicted)


def test_the_reserve_carries_what_is_needed_to_rebuild_the_weight():
    """lnZ needs the two prior components, not just lnL."""
    np.random.seed(20260811)
    s = _sampler(20000)
    s.integrate_log(_peaked(60.0), *NAMES, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    r = s._warm_seed_reserve
    for k in ('lnL', 'log_joint_prior', 'log_joint_s_prior'):
        assert k in r, 'reserve omits {}'.format(k)
        assert len(r[k]) == len(r['X']), '{} is not aligned with the points'.format(k)


###
### 4e. the reserve's lnZ must be normalized by the DRAWS, not by what survived filtering
###

def _reserve(lnL, lp=None, ls=None, n_retained=None, n_finite=None):
    from RIFT.integrators.mcsamplerAdaptiveVolume import make_warm_seed_reserve
    lnL = np.asarray(lnL, dtype=float)
    X = np.zeros((len(lnL), NDIM))
    r = make_warm_seed_reserve(
        X, lnL, NAMES, n_max=0,
        log_joint_prior=np.zeros(len(lnL)) if lp is None else lp,
        log_joint_s_prior=np.zeros(len(lnL)) if ls is None else ls)
    if n_retained is not None:
        r['n_retained'] = n_retained
    if n_finite is not None:
        r['n_finite'] = n_finite
    return r


def test_reserve_lnZ_divides_by_the_draws_made_not_by_the_finite_survivors():
    """A -inf draw is a real draw contributing a real zero; dropping it must not renormalize.

    Averaging over the stored rows overestimates by log(n_retained/n_finite) -- ~11 nats for
    a portfolio whose finite fraction on a collapsed pass is ~1e-5.
    """
    from RIFT.integrators.mcsamplerAdaptiveVolume import lnZ_from_reserve
    finite = np.array([10.0, 9.0, 8.0])
    n_draws = 300000
    lnL = np.concatenate([finite, np.full(n_draws - len(finite), -np.inf)])
    r = _reserve(lnL)
    assert r['n_retained'] == n_draws and r['n_finite'] == len(finite)
    expected = np.log(np.sum(np.exp(finite))) - np.log(n_draws)
    assert abs(lnZ_from_reserve(r) - expected) < 1e-9


def test_reserve_lnZ_is_unchanged_for_a_sampler_whose_rows_are_all_finite():
    """AV retains only finite rows, so the correction is exactly zero there."""
    from RIFT.integrators.mcsamplerAdaptiveVolume import lnZ_from_reserve
    lw = np.array([3.0, 2.0, 1.0, 0.5])
    r = _reserve(lw)
    assert r['n_retained'] == r['n_finite'] == 4
    expected = np.log(np.mean(np.exp(lw)))
    assert abs(lnZ_from_reserve(r) - expected) < 1e-9


def test_the_fallback_reading_accounts_for_the_uniform_cap():
    """The FALLBACK path only -- a reserve with no recorded exact total (an older writer, or
    one built without the prior components).  With m rows kept out of n_finite, the sum must
    be scaled back up by n_finite/m.  Where the exact total IS recorded it wins, because this
    estimate's logarithm carries cap sampling error; see section 4g."""
    from RIFT.integrators.mcsamplerAdaptiveVolume import lnZ_from_reserve
    lw = np.zeros(100)                      # w = 1 each, so the arithmetic is exact
    r = dict(lnL=lw, log_joint_prior=np.zeros(100), log_joint_s_prior=np.zeros(100),
             n_retained=10000, n_finite=1000, params_ordered=NAMES)
    assert 'ln_sum_w_finite' not in r
    # 100 kept of 1000 finite of 10000 drawn -> Z = (1000/100)*100*1 / 10000 = 0.1
    assert abs(lnZ_from_reserve(r) - np.log(0.1)) < 1e-9


def test_the_normalization_error_does_not_cancel_between_two_passes():
    """WHY this is a gate bug and not just a wrong number: the fractions differ.

    A cold pass with a 1e-5 finite fraction and a warm pass with 1e-2, on IDENTICAL finite
    weights, must give the same lnZ ordering as their draw counts imply -- not an artefact
    of how much each one underflowed.
    """
    from RIFT.integrators.mcsamplerAdaptiveVolume import lnZ_from_reserve
    finite = np.array([10.0, 9.5, 9.0, 8.0])
    cold = _reserve(finite, n_retained=1000000, n_finite=len(finite))
    warm = _reserve(finite, n_retained=1000, n_finite=len(finite))
    gap = lnZ_from_reserve(warm) - lnZ_from_reserve(cold)
    assert abs(gap - np.log(1000000.0 / 1000.0)) < 1e-9, \
        'the draw-count difference is not being carried into lnZ'
    # and the naive row-average would have made them IDENTICAL, hiding a 6.9-nat difference
    naive = np.log(np.mean(np.exp(finite)))
    assert abs(naive - (lnZ_from_reserve(cold) + np.log(1000000.0 / len(finite)))) < 1e-9


def test_a_portfolio_pass_with_underflowed_draws_reports_a_draw_normalized_lnZ():
    """End to end on the sampler that actually holds -inf rows in _rvs."""
    from RIFT.integrators.mcsamplerAdaptiveVolume import lnZ_from_reserve
    np.random.seed(20260812)
    s = _portfolio(20000)
    # a peak sharp enough that most draws underflow, so n_finite << n_retained
    s.integrate_log(_peaked(90.0), *NAMES, nmax=200000, neff=8, n=20000,
                    no_protect_names=True, verbose=False, save_intg=True,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    r = s._warm_seed_reserve
    assert r is not None
    assert r['n_finite'] < r['n_retained'], \
        'this target did not underflow; the test would prove nothing'
    v = lnZ_from_reserve(r)
    assert v is not None and np.isfinite(v)
    # it is the EXACT pre-cap total over the draws made -- not an average of stored rows
    assert abs(v - (r['ln_sum_w_finite'] - np.log(r['n_retained']))) < 1e-9
    # and the naive row-average, which is what reading the reserve like an _rvs would give,
    # is higher.  The gap is log(n_retained/n_finite) only when the cap did not bind; with a
    # cap the stored rows are a subsample too, so assert the direction, not the exact size.
    lw = r['lnL'] + r['log_joint_prior'] - r['log_joint_s_prior']
    lw = lw[np.isfinite(lw)]
    naive = np.log(np.sum(np.exp(lw - lw.max()))) + lw.max() - np.log(len(r['lnL']))
    assert naive > v, 'the draw normalization did not lower the estimate'
    if len(r['lnL']) >= r['n_finite']:      # uncapped: the exact relation holds
        assert abs((naive - v) - np.log(r['n_retained'] / float(r['n_finite']))) < 1e-6


###
### 4g. the gate's reading must not depend on whether the cap happened to bind
###

def test_capping_does_not_move_the_reserve_lnZ():
    """The reviewer's scenario, end to end.

    Two equally dominant rows among 200,000.  Uncapped, both are in the reserve; capped at
    2,000 only the force-appended peak is certain and the other is missed ~99% of the time.
    Estimating lnZ from the kept rows puts the capped reading log(2) = 0.69 nats low --
    above the 0.5-nat default reject threshold -- so a cold pass that fits under the cap
    would reject an otherwise identical warm pass that does not, on subsample luck alone.
    """
    from RIFT.integrators.mcsamplerAdaptiveVolume import (
        make_warm_seed_reserve, lnZ_from_reserve)
    n = 200000
    lnL = np.full(n, -50.0)
    lnL[[7, 9999]] = 0.0
    X = np.zeros((n, NDIM))
    z = np.zeros(n)
    cold = make_warm_seed_reserve(X, lnL, NAMES, n_max=0,
                                  log_joint_prior=z, log_joint_s_prior=z)
    warm = make_warm_seed_reserve(X, lnL, NAMES, n_max=2000,
                                  log_joint_prior=z, log_joint_s_prior=z)
    assert len(warm['lnL']) < len(cold['lnL']), 'the cap did not bind; test proves nothing'
    gap = abs(lnZ_from_reserve(cold) - lnZ_from_reserve(warm))
    assert gap < 1e-9, 'cap sampling error of {:.3f} nats reached the gate'.format(gap)


def test_the_exact_reading_is_the_draw_normalized_one():
    from RIFT.integrators.mcsamplerAdaptiveVolume import (
        make_warm_seed_reserve, lnZ_from_reserve)
    n = 50000
    lnL = np.full(n, -np.inf)
    lnL[:4] = np.array([1.0, 0.5, 0.25, 0.0])
    X = np.zeros((n, NDIM))
    z = np.zeros(n)
    r = make_warm_seed_reserve(X, lnL, NAMES, n_max=0,
                               log_joint_prior=z, log_joint_s_prior=z)
    expected = np.log(np.sum(np.exp(lnL[:4]))) - np.log(n)
    assert abs(lnZ_from_reserve(r) - expected) < 1e-9


def test_the_fallback_still_works_for_a_reserve_without_the_exact_total():
    """An older writer, or one that had no prior components at build time."""
    from RIFT.integrators.mcsamplerAdaptiveVolume import lnZ_from_reserve
    lw = np.array([1.0, 0.5, 0.25])
    r = dict(lnL=lw, log_joint_prior=np.zeros(3), log_joint_s_prior=np.zeros(3),
             n_retained=300, n_finite=3, params_ordered=NAMES)
    expected = np.log(np.sum(np.exp(lw))) - np.log(300)
    assert abs(lnZ_from_reserve(r) - expected) < 1e-9


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


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_reject_gate_reads_both_sides_from_the_same_record():
    with open(_ILE) as f:
        src = f.read()
    assert '_lnZ_of_reserve_or_rvs' in src, 'the gate still reads lnZ straight out of _rvs'
    i = src.index('_evidence_of_loss = (')
    block = src[max(0, i - 2000):i]
    assert '_cold_src != _warm_src' in block, \
        'nothing stops the gate comparing a retained-set lnZ against a fair-drawn one'
    assert 'lnZ_from_reserve' in src, \
        'the reserve reading still averages over stored rows instead of over the draws made'
    # the cold reserve must be snapshotted before the warm pass overwrites it.  Asserted on
    # the WHOLE source between the two anchors rather than inside a fixed-size window: the
    # window version started failing when unrelated lines were added between them, which reads
    # as a regression in the gate rather than in the test.
    _i_res = src.index('_cold_reserve_l0')
    _i_int = src.index('sampler.integrate(', _i_res)
    assert _i_res < _i_int, \
        'the cold reserve is read after the warm pass has already replaced it'
