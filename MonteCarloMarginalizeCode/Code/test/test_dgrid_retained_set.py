#!/usr/bin/env python
"""
Regression tests for the .dgrid exporter reading the RETAINED set instead of the fair draw
(RIFT/integrators/*, RIFT/misc/distance_grid.py, bin/integrate_likelihood_extrinsic_batchmode).

Background.  `--export-marginal-distance-grid` built its likelihood-vs-distance table from
`sampler._rvs`.  By then every integrator has REPLACED `_rvs` with an export resample:
n_extr = min(--fairdraw-extrinsic-output-n-max (5 by default), 1.5*eff_samp, 1.5*neff) rows
drawn WITH REPLACEMENT.  Measured on the ILE-GPU-Paper demo at GMM n_eff 5.9, that is FIVE
rows holding THREE distinct distances -- and the exporter wrote a five-row curve from them.

Fixing the binder (test_distance_grid_degenerate_bins.py) stops that table being degenerate.
It does not make it a curve: three points is three points.  The rows the pass actually kept
were still in `_rvs` one statement earlier.

  1. ONLY TWO SAMPLERS KEPT THEM.  `make_warm_seed_reserve` was built for the L0 rescue, so
     mcsamplerAdaptiveVolume and mcsamplerPortfolio kept a bounded copy of the retained rows
     and mcsampler / mcsamplerGPU (both entry points) / mcsamplerEnsemble / mcsamplerNFlow
     passed `reserve=getattr(self,'_warm_seed_reserve',None)` -- always None -- into their
     records.  For an L0 rescue that was the honest answer.  For anything that EXPORTS A
     SHAPE it was a gap, and the sampler in the bug report (GMM = Ensemble) was one of them.

  2. THE RESERVE IS NOT EQUAL WEIGHT.  A fair-drawn record must NOT be re-weighted by w
     (ln_weights_for_posterior exists for that reason).  The retained rows are the opposite
     case: they carry real importance weights, and both prior components ride along in the
     reserve precisely so a consumer can rebuild them.  Reading the reserve and then
     weighting it uniformly would be the same defect with the sign flipped.

  3. MORE ROWS IS NOT MORE INFORMATION.  The retained set has thousands of distinct
     distances at the same n_eff, so the duplicate-distance warning goes quiet exactly when
     the table stops LOOKING broken.  The resolution check is therefore effective sample
     size against bin count, not distinct values alone.

The requirement: every integrator that fair-draws leaves the retained rows reachable, the
exporter uses them with their own weights, and a starved pass is still called starved.
"""

import ast
import os

import numpy as np
import pytest

import RIFT.integrators.mcsampler as MC
import RIFT.integrators.mcsamplerAdaptiveVolume as AV
import RIFT.integrators.mcsamplerEnsemble as ENS
from RIFT.integrators.mcsamplerAdaptiveVolume import (make_reserve_from_rvs,
                                                     make_warm_seed_reserve)
from RIFT.integrators.rvs_record import RvsRecord
from RIFT.misc.distance_grid import (
    _ess,
    build_distance_grid,
    distance_grid_inputs,
    distance_grid_resolution_warning,
    reconstruct_marginal_lnL,
    reserve_distance_and_ln_weights,
)

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
_HERE = os.path.dirname(os.path.abspath(__file__))
_ILE = os.path.join(_HERE, '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')
_INTEGRATORS = os.path.join(_HERE, '..', 'RIFT', 'integrators')

FAIRDRAW_MAX = 5          # the --fairdraw-extrinsic-output-n-max default


def _sharp(lnLmax=50.0, width=0.01, log=True):
    """A peak narrow enough that the fair draw really does collapse, as it does at the
    production SNR this defect was found at."""
    def f(*args, **kwargs):
        x = np.array([np.asarray(a, dtype=float).ravel() for a in args]).T
        v = -0.5 * np.sum(((x - 0.5) / width) ** 2, axis=-1) + lnLmax
        return v if log else np.exp(v)
    return f


def _av():
    s = AV.MCSampler(n_chunk=5000)
    s.xpy = AV.xpy_default
    s.identity_convert = AV.identity_convert
    for n in NAMES:
        s.add_parameter(n, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    return s


def _vectorized(mod):
    def build():
        s = mod.MCSampler()
        v = np.vectorize(lambda x: 1.0)
        for n in NAMES:
            s.add_parameter(n, v, prior_pdf=v, left_limit=0.0, right_limit=1.0,
                            adaptive_sampling=True)
        return s
    return build


def _backends():
    """The three backends that can be driven in-process here.  mcsamplerGPU and
    mcsamplerNFlow need cupy / torch, so their sites are covered by the source sweep
    below rather than left unchecked."""
    return [
        ('AV', _av, 'integrate_log', _sharp(log=True)),
        ('Ensemble', _vectorized(ENS), 'integrate', _sharp(log=False)),
        ('mcsampler', _vectorized(MC), 'integrate', _sharp(log=False)),
    ]


def _run(build, method, target):
    np.random.seed(11)
    s = build()
    getattr(s, method)(target, *NAMES, nmax=40000, n=4000, neff=30, save_intg=True,
                       no_protect_names=True, verbose=False,
                       igrand_fairdraw_samples=True,
                       igrand_fairdraw_samples_max=FAIRDRAW_MAX)
    return s


def _reserve_of(sampler):
    rec = sampler.samples() if hasattr(sampler, 'samples') else None
    if rec is not None and getattr(rec, 'reserve', None) is not None:
        return rec.reserve
    return getattr(sampler, '_warm_seed_reserve', None)


###
### 1. every fair-drawing backend leaves the retained rows reachable
###

@pytest.mark.parametrize('label,build,method,target', _backends())
def test_the_fair_draw_no_longer_takes_the_retained_rows_with_it(label, build, method, target):
    s = _run(build, method, target)

    n_rvs = len(np.asarray(s._rvs['distance']).ravel())
    assert n_rvs <= FAIRDRAW_MAX, \
        '{}: the draw did not collapse, so this run proves nothing'.format(label)

    reserve = _reserve_of(s)
    assert reserve is not None, '{}: the retained rows were discarded with no record'.format(label)
    assert len(reserve['X']) > n_rvs, \
        '{}: reserve holds {} rows, no more than the {}-row draw'.format(
            label, len(reserve['X']), n_rvs)
    for key in ('lnL', 'log_joint_prior', 'log_joint_s_prior', 'params_ordered'):
        assert key in reserve, '{}: reserve cannot rebuild a weight without {}'.format(label, key)


@pytest.mark.parametrize('label,build,method,target', _backends())
def test_the_grid_built_from_the_reserve_resolves_where_the_draw_cannot(label, build, method, target):
    s = _run(build, method, target)
    got = reserve_distance_and_ln_weights(_reserve_of(s))
    assert got is not None, '{}: the exporter cannot read this reserve'.format(label)
    d_res, ln_w_res = got

    d_draw = np.asarray(s._rvs['distance'], dtype=float).ravel()
    assert len(np.unique(d_res)) > len(np.unique(d_draw)), \
        '{}: the reserve resolves no better than the fair draw'.format(label)

    grid = build_distance_grid(d_res, ln_w_res, 12.0, 0.0, {},
                               ln_prior_d_at_samples=np.zeros(len(d_res)), n_grid=50)
    assert len(grid) == 50, '{}: got {} rows from {} distinct distances'.format(
        label, len(grid), len(np.unique(d_res)))
    assert np.all(np.diff(grid['dist']) > 0)
    # NOT asserted here: reconstruct_marginal_lnL(grid) == 12.0.  That identity holds for
    # any partition whatsoever (lnL carries -log(width) and the reconstruction adds it back),
    # so phrasing it as "swapping in the retained set moved the evidence" claimed a
    # detection it cannot make.  What this test can honestly say is that the retained rows
    # carry MORE DISTINCT DISTANCES than the draw -- which is resolution in d, and is not
    # the same thing as information; the module docstring's third point is exactly that.


###
### 2. the reserve is weighted by w; the fair draw is not
###

def test_the_reserve_carries_real_importance_weights_not_uniform_ones():
    """If these rows were handed to the exporter and then weighted uniformly -- the correct
    treatment for a FAIR-DRAWN record -- the grid would be the wrong posterior."""
    names = ['distance', 'psi']
    lnL = np.array([0.0, 3.0, -2.0, 1.0])
    ln_prior = np.array([0.5, 0.5, 0.5, 0.5])
    ln_s_prior = np.array([0.1, 0.2, 0.3, 0.4])
    reserve = dict(X=np.array([[100.0, 0.1], [200.0, 0.2], [300.0, 0.3], [400.0, 0.4]]),
                   lnL=lnL, log_joint_prior=ln_prior, log_joint_s_prior=ln_s_prior,
                   params_ordered=names)

    d, ln_w = reserve_distance_and_ln_weights(reserve)

    assert np.array_equal(d, [100.0, 200.0, 300.0, 400.0])
    assert np.allclose(ln_w, lnL + ln_prior - ln_s_prior)
    assert not np.allclose(ln_w, ln_w[0]), 'the weights came back uniform'


def _no_reserve():          return None
def _empty():               return {}
def _no_distance_column():  return dict(X=np.zeros((3, 2)), lnL=np.zeros(3),
                                        log_joint_prior=np.zeros(3), log_joint_s_prior=np.zeros(3),
                                        params_ordered=['psi', 'phi_orb'])
def _no_rows():             return dict(X=np.zeros((0, 2)), lnL=np.zeros(0),
                                        log_joint_prior=np.zeros(0), log_joint_s_prior=np.zeros(0),
                                        params_ordered=['distance', 'psi'])
def _prior_missing():       return dict(X=np.zeros((3, 2)), lnL=np.zeros(3),
                                        log_joint_s_prior=np.zeros(3),
                                        params_ordered=['distance', 'psi'])
def _ragged():              return dict(X=np.zeros((3, 2)), lnL=np.zeros(4),
                                        log_joint_prior=np.zeros(3), log_joint_s_prior=np.zeros(3),
                                        params_ordered=['distance', 'psi'])


@pytest.mark.parametrize('make_broken,why', [
    (_no_reserve, 'no reserve at all'),
    (_empty, 'empty reserve'),
    (_no_distance_column, 'no distance column'),
    (_no_rows, 'no rows'),
    (_prior_missing, 'prior component missing'),
    (_ragged, 'ragged columns'),
])
def test_an_unusable_reserve_returns_none_so_the_caller_falls_back(make_broken, why):
    """None means "fall back to _rvs", which still works.  A raise here would take down an
    export that has already finished the expensive part.

    Each case is BUILT PER TEST rather than shared in the parametrize list: the list is
    evaluated once at collection, so a shared dict of numpy arrays is cross-test state, and
    this test was seen failing once in nine full-gate runs and never in twelve file-only
    runs -- the signature of order dependence rather than of the function under test."""
    assert reserve_distance_and_ln_weights(make_broken()) is None, why


###
### 3. the builder must read either integrand convention
###

@pytest.mark.parametrize('columns,is_log', [
    ({'log_integrand': np.array([0.0, 1.0, 2.0]),
      'log_joint_prior': np.array([0.1, 0.1, 0.1]),
      'log_joint_s_prior': np.array([0.2, 0.2, 0.2])}, None),
    ({'integrand': np.array([0.0, 1.0, 2.0]),
      'joint_prior': np.exp(np.array([0.1, 0.1, 0.1])),
      'joint_s_prior': np.exp(np.array([0.2, 0.2, 0.2]))}, True),
    ({'integrand': np.exp(np.array([0.0, 1.0, 2.0])),
      'joint_prior': np.exp(np.array([0.1, 0.1, 0.1])),
      'joint_s_prior': np.exp(np.array([0.2, 0.2, 0.2]))}, False),
])
def test_the_reserve_builder_recovers_the_same_weights_on_either_convention(columns, is_log):
    """`integrand` is lnL on some backends and linear L on others, and only the sampler knows
    which.  Reading it under the wrong one is not a rounding error: it is a different posterior."""
    cols = dict(columns)
    cols['distance'] = np.array([100.0, 200.0, 300.0])
    cols['psi'] = np.array([0.1, 0.2, 0.3])

    reserve = make_reserve_from_rvs(cols, ['distance', 'psi'], integrand_is_log=is_log)
    d, ln_w = reserve_distance_and_ln_weights(reserve)

    order = np.argsort(d)
    assert np.allclose(d[order], [100.0, 200.0, 300.0])
    assert np.allclose(ln_w[order] - ln_w[order][0], [0.0, 1.0, 2.0]), \
        'the weights differ by more than an additive constant: the convention was misread'


def test_a_zero_weight_row_is_dropped_rather_than_becoming_minus_inf_ballast():
    cols = {'integrand': np.array([1.0, 0.0, 2.0]),
            'joint_prior': np.ones(3), 'joint_s_prior': np.ones(3),
            'distance': np.array([100.0, 200.0, 300.0])}
    reserve = make_reserve_from_rvs(cols, ['distance'], integrand_is_log=False)
    d, ln_w = reserve_distance_and_ln_weights(reserve)
    assert np.all(np.isfinite(ln_w))
    assert 200.0 not in set(d.tolist())


###
### 3b. the bounded reserve is a SAMPLE here, not a seed
###

def _skewed_rvs(n=4000, seed=0):
    """Rows whose importance weights have a real tail, so the peak row matters."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(100.0, 900.0, size=n)
    ln_w = -0.5 * ((d - 400.0) / 40.0) ** 2
    return dict(distance=d, psi=rng.uniform(0, 1, n),
                log_integrand=ln_w, log_joint_prior=np.zeros(n),
                log_joint_s_prior=np.zeros(n))


def test_the_export_reserve_does_not_force_the_peak_row_in():
    """A seed wants the peak unconditionally -- it defines the seed's centre.  A WEIGHTED
    SAMPLE must not have it: forcing it gives the largest weight probability 1 where
    uniform subsampling gives it n_max/n_finite, and that one row then carries the export.
    """
    cols = _skewed_rvs()
    peak = float(cols['distance'][int(np.argmax(cols['log_integrand']))])

    seen_forced, seen_free = 0, 0
    for seed in range(30):
        X = np.vstack([cols['distance'], cols['psi']]).T
        common = dict(n_max=100, log_joint_prior=cols['log_joint_prior'],
                      log_joint_s_prior=cols['log_joint_s_prior'])
        forced = make_warm_seed_reserve(X, cols['log_integrand'], ['distance', 'psi'],
                                        rng=np.random.RandomState(seed), force_peak=True, **common)
        free = make_warm_seed_reserve(X, cols['log_integrand'], ['distance', 'psi'],
                                      rng=np.random.RandomState(seed), force_peak=False, **common)
        seen_forced += peak in set(forced['X'][:, 0].tolist())
        seen_free += peak in set(free['X'][:, 0].tolist())

    assert seen_forced == 30, 'force_peak=True must always keep the peak (the seed needs it)'
    assert seen_free < 30, 'force_peak=False still admits the peak every time'


def _fair_draw(uniq, counts):
    """An _rvs column as the fair draw leaves it: each surviving distance repeated."""
    uniq = np.asarray(uniq, dtype=float)
    distance = np.repeat(uniq, np.asarray(counts, dtype=int))
    return distance, 2.0 * np.log(distance), np.zeros(len(distance))


def _spiked_population(n=8000, seed=3, boost=0.0):
    """Broad importance weights with one row optionally lifted above the rest."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(100.0, 900.0, n)
    ln_w = -0.5 * ((d - 400.0) / 150.0) ** 2
    if boost:
        ln_w = ln_w.copy()
        ln_w[int(np.argmax(ln_w))] += boost
    X = np.vstack([d, rng.uniform(0, 1, n)]).T
    return X, ln_w


def _median_reserve_ess(X, ln_w, force, n_max=800, seeds=9):
    out = []
    for seed in range(seeds):
        r = make_warm_seed_reserve(X, ln_w, ['distance', 'psi'], n_max=n_max,
                                   log_joint_prior=np.zeros(len(ln_w)),
                                   log_joint_s_prior=np.zeros(len(ln_w)),
                                   rng=np.random.RandomState(seed), force_peak=force)
        out.append(_ess(reserve_distance_and_ln_weights(r)[1]))
    return float(np.median(out))


def _raw_reserve_weights(r):
    """The reserve's weights WITHOUT the exporter's inclusion-probability correction."""
    return (np.asarray(r["lnL"], float) + np.asarray(r["log_joint_prior"], float)
            - np.asarray(r["log_joint_s_prior"], float))


def test_the_export_is_insensitive_to_whether_the_reserve_forced_its_peak():
    """AV and mcsamplerPortfolio build their own reserve for the L0 rescue, which WANTS the
    peak row, and never go through make_reserve_from_rvs -- so asking that adapter for
    force_peak=False fixed nothing on the two samplers that actually emit a .dgrid.  The
    exporter therefore divides the forced row by its own inclusion probability instead.

    Both directions are asserted: the correction must close the gap, and the gap must be
    there to close, or this test passes on an exporter that does nothing."""
    X, spiked = _spiked_population(boost=6.0)
    kw = dict(n_max=800, log_joint_prior=np.zeros(len(spiked)),
              log_joint_s_prior=np.zeros(len(spiked)))

    population = _ess(spiked)
    raw, corrected = [], []
    for seed in range(9):
        r = make_warm_seed_reserve(X, spiked, ['distance', 'psi'],
                                   rng=np.random.RandomState(seed), force_peak=True, **kw)
        raw.append(_ess(_raw_reserve_weights(r)))
        corrected.append(_ess(reserve_distance_and_ln_weights(r)[1]))
    raw, corrected = float(np.median(raw)), float(np.median(corrected))

    assert raw < 0.2 * population, \
        'the forced peak no longer dominates the RAW weights, so this test is vacuous'
    assert 0.5 * population < corrected < 2.0 * population, \
        ('corrected ESS {:.1f} is not the population {:.1f}: the exporter still reads a '
         'forced row as though it came in at p=1'.format(corrected, population))


def test_the_correction_needs_the_inclusion_probability_to_be_recorded():
    """n_subsample is the one number the kept rows cannot reveal, so it is recorded next to
    n_finite for the same reason ln_sum_w_finite and ess_finite are."""
    X, spiked = _spiked_population(boost=6.0)
    r = make_warm_seed_reserve(X, spiked, ['distance', 'psi'], n_max=800,
                               log_joint_prior=np.zeros(len(spiked)),
                               log_joint_s_prior=np.zeros(len(spiked)),
                               rng=np.random.RandomState(0), force_peak=True)
    assert r['capped'] is True and r['force_peak'] is True
    assert r['n_subsample'] == 800
    assert r['n_finite'] == len(spiked)

    d, ln_w = reserve_distance_and_ln_weights(r)
    raw = _raw_reserve_weights(r)
    k = int(np.argmax(np.asarray(r['lnL'], float)))
    shifted = np.flatnonzero(~np.isclose(ln_w, raw))
    assert list(shifted) == [k], 'the correction touched rows other than the forced peak'
    assert np.isclose(ln_w[k] - raw[k], np.log(800.0 / r['n_finite']))

    uncapped = make_warm_seed_reserve(X, spiked, ['distance', 'psi'], n_max=len(spiked) * 2,
                                      log_joint_prior=np.zeros(len(spiked)),
                                      log_joint_s_prior=np.zeros(len(spiked)),
                                      force_peak=True)
    assert uncapped['n_subsample'] is None, 'an uncapped reserve has no inclusion probability'
    assert np.allclose(reserve_distance_and_ln_weights(uncapped)[1],
                       _raw_reserve_weights(uncapped)), 'corrected a reserve that was never capped'


def test_the_reserve_records_the_population_ess_the_subsample_cannot_show():
    """The other half of the same problem, and the reason dropping the forced peak is not
    on its own enough: an unbiased subsample that misses the dominant row reports a healthy
    effective sample size for a pass that has none."""
    X, spiked = _spiked_population(boost=9.0)
    population = _ess(spiked)

    # MEDIAN over seeds, not RandomState(0): the subsample contains the dominant row about
    # 10% of the time (800 of 8000), so a single pinned seed is one n_max change away from a
    # confusing red.  Measured 4 of 40 seeds fail the inequality individually.
    subs, ess_finites = [], []
    for seed in range(9):
        r = make_warm_seed_reserve(X, spiked, ['distance', 'psi'], n_max=800,
                                   log_joint_prior=np.zeros(len(spiked)),
                                   log_joint_s_prior=np.zeros(len(spiked)),
                                   rng=np.random.RandomState(seed), force_peak=False)
        subs.append(_ess(reserve_distance_and_ln_weights(r)[1]))
        ess_finites.append(r['ess_finite'])
    subsample = float(np.median(subs))

    assert all(e is not None for e in ess_finites), 'the pre-cap ESS was not recorded'
    assert np.allclose(ess_finites, population, rtol=1e-9), \
        'ess_finite describes the subsample, not the population'
    assert subsample > 10.0 * population, \
        'the typical subsample did not miss the dominant row, so it does not show the problem'


def test_the_warning_prefers_the_population_ess_over_the_subsample_it_was_handed():
    rng = np.random.default_rng(8)
    d = rng.uniform(100.0, 900.0, size=2000)
    ln_w = -0.5 * ((d - 500.0) / 300.0) ** 2          # a healthy-looking subsample

    assert distance_grid_resolution_warning(d, n_grid=50, ln_weights=ln_w) is None
    warned = distance_grid_resolution_warning(d, n_grid=50, ln_weights=ln_w,
                                              n_eff_population=2.1)
    assert warned is not None, 'the pass had n_eff 2.1 and the check believed the subsample'
    assert 'n_eff=2.1' in warned


def test_the_adapter_records_everything_the_correction_needs():
    """Every reserve in the tree is built the same way, so what the adapter owes the
    exporter is the bookkeeping, not a different sampling rule."""
    cols = _skewed_rvs(n=3000)
    reserve = make_reserve_from_rvs(cols, ['distance', 'psi'], n_max=250)
    assert reserve['capped'] is True
    assert reserve['n_finite'] == 3000, 'the pre-cap population was not recorded'
    assert reserve['n_subsample'] == 250, 'the inclusion probability is not recoverable'
    assert reserve['ess_finite'] is not None
    assert len(reserve['X']) <= 251        # n_max, plus the forced peak


def test_the_default_keeps_the_peak_row():
    """The default is what every reserve in the tree is built with, and the exporter's
    inclusion-probability correction is written against it.  Flipping it to False survived
    the whole suite, because the one test that looks at the peak compares lnL values around
    10700 with np.isclose, whose default rtol is 0.107 nats there."""
    X, spiked = _spiked_population(boost=6.0)
    r = make_warm_seed_reserve(X, spiked, ['distance', 'psi'], n_max=800,
                               log_joint_prior=np.zeros(len(spiked)),
                               log_joint_s_prior=np.zeros(len(spiked)),
                               rng=np.random.RandomState(3))
    assert r['force_peak'] is True, 'the default no longer keeps the peak row'
    assert float(np.max(r['lnL'])) == float(np.max(spiked)), \
        'the peak row is absent from a reserve built with the default'


def test_a_tuple_parameter_still_yields_a_usable_reserve():
    """--skymap-file registers ('declination','right_ascension') as ONE parameter, so
    _rvs[key] is (2, N).  A plain ravel makes the vstack ragged and the whole reserve is
    lost -- silently, since the builder only prints and returns None."""
    n = 400
    rng = np.random.default_rng(2)
    cols = {
        'distance': rng.uniform(100.0, 900.0, n),
        ('declination', 'right_ascension'): np.vstack([rng.uniform(-1, 1, n),
                                                       rng.uniform(0, 6, n)]),
        'log_integrand': -0.5 * rng.normal(size=n) ** 2,
        'log_joint_prior': np.zeros(n),
        'log_joint_s_prior': np.zeros(n),
    }
    params = ['distance', ('declination', 'right_ascension')]

    reserve = make_reserve_from_rvs(cols, params)

    assert reserve['params_ordered'] == ['distance', 'declination', 'right_ascension'], \
        'the tuple key was not flattened to its component coordinates'
    assert reserve['X'].shape == (n, 3)
    # THE VALUES, not just the shape.  reshape(-1, 2).T has the same shape and interleaves
    # declination with right ascension; the warm-seed rescue builds a covariance from these
    # columns, so scrambled sky values corrupt a seed with nothing to show for it.
    for col, want in ((0, cols['distance']),
                      (1, cols[('declination', 'right_ascension')][0]),
                      (2, cols[('declination', 'right_ascension')][1])):
        assert np.allclose(reserve['X'][:, col], want), \
            'column {} is not the values that were handed in'.format(col)
    d, ln_w = reserve_distance_and_ln_weights(reserve)
    assert np.allclose(d, cols['distance'])
    assert reserve_distance_and_ln_weights(reserve, param='declination') is not None


def test_a_row_that_cannot_carry_weight_is_dropped_not_turned_into_nan():
    """RvsRecord.log_weights() records why: computed term by term, a row with a zero prior
    AND a zero sampling prior gives -inf - (-inf) = NaN rather than 'no weight', and one NaN
    propagates through _logsumexp to take the WHOLE export down."""
    from RIFT.integrators.rvs_record import RvsRecord
    n = 50
    rng = np.random.default_rng(1)
    cols = {'distance': rng.uniform(100.0, 900.0, n),
            'integrand': np.exp(-0.5 * rng.normal(size=n) ** 2),
            'joint_prior': np.ones(n), 'joint_s_prior': np.ones(n)}
    cols['joint_prior'][7] = 0.0
    cols['joint_s_prior'][7] = 0.0

    reserve = make_reserve_from_rvs(cols, ['distance'], integrand_is_log=False)
    d, ln_w = reserve_distance_and_ln_weights(reserve)
    assert np.all(np.isfinite(ln_w)), 'a NaN weight survived into the export'
    assert cols['distance'][7] not in set(d.tolist()), 'the zero-weight row was kept'

    # and the surviving weights are the ones the canonical implementation gives
    want = np.asarray(RvsRecord.retained(dict(cols), integrand_is_log=False).log_weights(),
                      dtype=float)
    keep = np.isfinite(want)
    assert np.allclose(np.sort(ln_w - np.max(ln_w)),
                       np.sort(want[keep] - np.max(want[keep]))), \
        'the adapter derives a different weight than RvsRecord.log_weights()'


def test_a_collapsed_pass_is_flagged_even_when_its_weights_look_uniform():
    """The integrand column can bottom out at its underflow floor, which makes the derived
    weights UNIFORM and every sample-side check report a healthy grid.  Measured on a
    collapsed Ensemble pass: ESS 4000 of 4000 rows, exported curve flat to 0.13 nats across
    a true 1130-nat span.  The sampler's own n_eff is the input that cannot be faked."""
    rng = np.random.default_rng(9)
    d = rng.uniform(100.0, 900.0, size=4000)
    ln_w = np.zeros(len(d))                       # saturated column -> uniform weights

    assert distance_grid_resolution_warning(d, n_grid=20, ln_weights=ln_w) is None, \
        'the sample-side checks already catch this, so the new clause is untested'
    warned = distance_grid_resolution_warning(d, n_grid=20, ln_weights=ln_w, n_eff_sampler=1.4)
    assert warned is not None, 'a pass the sampler reported as n_eff 1.4 was called fine'
    assert 'sampler reported n_eff=1.4' in warned
    assert distance_grid_resolution_warning(d, n_grid=20, ln_weights=ln_w,
                                            n_eff_sampler=500.0) is None


def test_an_uncapped_reserve_is_not_marked_as_a_subsample():
    cols = _skewed_rvs(n=300)
    reserve = make_reserve_from_rvs(cols, ['distance', 'psi'], n_max=20000)
    assert reserve['capped'] is False
    assert len(reserve['X']) == 300


###
### 4. a starved pass is still called starved once the rows look healthy
###

def test_more_distinct_distances_at_the_same_neff_does_not_silence_the_warning():
    """This is the trap in reading the retained set: thousands of distinct distances, the
    same handful of effective samples.  The duplicate-distance clause goes quiet; the
    effective-sample-size clause must not."""
    rng = np.random.default_rng(5)
    d = rng.uniform(100.0, 900.0, size=20000)          # all distinct
    ln_w = np.full(len(d), -700.0)
    ln_w[:6] = 0.0                                      # n_eff ~ 6

    warn = distance_grid_resolution_warning(d, n_grid=50, ln_weights=ln_w)

    assert warn is not None, 'a 50-bin grid from n_eff ~ 6 was reported as fine'
    assert 'distinct distance' not in warn, 'the wrong clause fired: the distances are distinct'
    assert 'n_eff' in warn


def test_the_warning_describes_the_rows_the_BUILDER_will_bin():
    """The helper and _weighted_blocks must filter the same way.

    _weighted_blocks drops every row whose normalized probability is not strictly
    positive.  A helper that filters only on isfinite is describing a different sample
    set, and the divergence is one-sided: the pass where a handful of weights survive and
    the rest underflow is the genuinely starved one, and it was the silent one.
    """
    ln_pi = np.zeros(5)

    # (a) three rows underflow to p == 0 exactly, and the three that survive share one
    #     distance -- so the builder cannot make a grid at all.
    d = np.array([100.0, 100.0, 100.0, 400.0, 500.0])
    ln_w = np.array([0.0, 0.0, 0.0, -1e4, -1e4])
    assert np.sum(np.exp(ln_w - np.max(ln_w)) > 0) == 3, 'the setup no longer underflows'
    with pytest.raises(ValueError):
        build_distance_grid(d, ln_w, 0.0, 0.0, {}, ln_prior_d_at_samples=ln_pi, n_grid=3)
    warn = distance_grid_resolution_warning(d, n_grid=3, ln_weights=ln_w)
    assert warn is not None, 'the builder refused this grid and the check called it fine'
    assert 'distinct distance' in warn

    # (b) the other direction: two usable rows, two bins, one effective sample per bin.
    #     Counting the three discarded rows instead reports a five-bin grid that is not
    #     what gets written.
    d = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    ln_w = np.array([0.0, 0.0, -1e4, -1e4, -1e4])
    grid = build_distance_grid(d, ln_w, 0.0, 0.0, {}, ln_prior_d_at_samples=ln_pi, n_grid=5)
    assert len(grid) == 2, 'the builder kept rows this test assumes it drops'
    assert distance_grid_resolution_warning(d, n_grid=5, ln_weights=ln_w) is None, \
        'flagged a two-row grid built from two effective samples'


def test_the_message_names_the_row_count_that_will_actually_be_written():
    """4b: the ESS threshold and the message both use the bin count the builder will
    produce, which is capped at the distinct-distance count.  Using the REQUESTED count
    instead makes the message self-contradictory ("carries 4 row(s), not the 4 requested")
    and moves the threshold, and no existing case drove n_unique < n_requested WITH
    weights supplied."""
    distance, ln_pi, _ = _fair_draw([120.0, 305.0, 900.0, 1500.0], [3, 2, 4, 1])
    ln_w = np.zeros(len(distance))

    grid = build_distance_grid(distance, ln_w, 0.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=500)
    warn = distance_grid_resolution_warning(distance, n_grid=500, ln_weights=ln_w)

    assert len(grid) == 4
    assert warn is not None
    assert 'carries {} row(s), not the {} requested'.format(len(grid), len(distance)) in warn, \
        'message says: {!r}'.format(warn)


def test_a_healthy_weighted_sample_draws_no_warning():
    rng = np.random.default_rng(6)
    d = rng.uniform(100.0, 900.0, size=20000)
    ln_w = -0.5 * ((d - 500.0) / 300.0) ** 2           # n_eff of order the sample size
    assert distance_grid_resolution_warning(d, n_grid=50, ln_weights=ln_w) is None


def test_both_clauses_report_together_when_both_apply():
    d = np.repeat([100.0, 200.0], 3)
    ln_w = np.array([0.0, -700.0, -700.0, -700.0, -700.0, -700.0])
    warn = distance_grid_resolution_warning(d, n_grid=6, ln_weights=ln_w)
    assert warn is not None and 'distinct distance' in warn and 'n_eff' in warn


###
### 4b. the export DECISION, run rather than read
###

def _fake_rvs(n=5, d0=900.0):
    """An _rvs as the fair draw leaves it: few rows, distances nowhere near the reserve's."""
    return {'distance': np.full(n, d0), 'psi': np.linspace(0.1, 0.5, n),
            'integrand': np.ones(n), 'joint_prior': np.ones(n), 'joint_s_prior': np.ones(n)}


def _fake_reserve(n=400, capped=True, ess=None):
    rng = np.random.default_rng(12)
    d = rng.uniform(100.0, 300.0, n)
    ln_w = -0.5 * ((d - 200.0) / 40.0) ** 2
    r = dict(X=np.vstack([d, rng.uniform(0, 1, n)]).T, lnL=ln_w,
             log_joint_prior=np.zeros(n), log_joint_s_prior=np.zeros(n),
             params_ordered=['distance', 'psi'], capped=capped, force_peak=False,
             n_finite=(4000 if capped else n), n_subsample=(n if capped else None),
             ess_finite=(ess if ess is not None else _ess(ln_w)))
    return r


def _uniform_fallback(rvs):
    return lambda: np.zeros(len(np.asarray(rvs['distance']).ravel()))


def test_a_fairdrawn_record_with_a_reserve_exports_the_RETAINED_rows():
    """The whole of commit 2, as one call.  Inverting the branch, or reading the reserve and
    then re-reading _rvs anyway, both leave every source-level assertion satisfied."""
    rvs, reserve = _fake_rvs(), _fake_reserve()
    rec = RvsRecord.fair_draw(rvs, reserve=reserve, integrand_is_log=False)

    d, ln_w, _lnpi, notes, warn = distance_grid_inputs(rec, rvs, _uniform_fallback(rvs), n_grid=20)

    assert len(d) == len(reserve['X']), 'the fair-drawn rows were exported, not the reserve'
    assert np.allclose(np.sort(d), np.sort(reserve['X'][:, 0]))
    assert 900.0 not in set(d.tolist()), 'an _rvs distance leaked into the export'
    assert not np.allclose(ln_w, ln_w[0]), 'the retained rows came back equal-weight'
    assert any('RETAINED set' in n for n in notes)
    assert any('uniform subsample of 4000' in n for n in notes), 'the cap was not disclosed'


def test_the_prior_comes_back_evaluated_at_the_rows_that_were_chosen():
    """The prior and the rows cannot be separated any more, because they used to be: taking
    it at the fair draw's distances while the curve came from the retained set moved the
    exported lnL by 2.87 nats and flattened ln_prior_d_sampling, with every test green."""
    rvs, reserve = _fake_rvs(), _fake_reserve()
    volumetric = lambda d: np.asarray(d, dtype=float) ** 2

    d, _, ln_pi, _, _ = distance_grid_inputs(
        RvsRecord.fair_draw(rvs, reserve=reserve, integrand_is_log=False), rvs,
        _uniform_fallback(rvs), n_grid=20, prior_pdf=volumetric)

    assert len(ln_pi) == len(d), 'the prior has a different length than the rows'
    assert np.allclose(ln_pi, np.log(volumetric(d))), \
        'the prior was evaluated somewhere other than the exported distances'
    # and it is emphatically NOT the fair draw's single repeated distance
    assert not np.allclose(ln_pi, np.log(volumetric(np.full(len(d), 900.0))))


def test_the_prior_follows_the_fall_back_rows_too():
    rvs, reserve = _fake_rvs(), _fake_reserve()
    volumetric = lambda d: np.asarray(d, dtype=float) ** 2
    d, _, ln_pi, notes, _ = distance_grid_inputs(
        RvsRecord.retained(rvs, reserve=reserve, integrand_is_log=False), rvs,
        _uniform_fallback(rvs), n_grid=20, prior_pdf=volumetric)
    assert notes == [] and np.allclose(d, 900.0)
    assert np.allclose(ln_pi, np.log(volumetric(d)))


def test_an_uncorrectable_capped_reserve_is_declined_rather_than_used():
    """A peak-forced capped reserve with no recorded n_subsample predates the field, so the
    forced row cannot be divided by its inclusion probability.  Using it anyway reinstates
    exactly the bias this path removes."""
    r = _fake_reserve()
    r['force_peak'] = True
    r.pop('n_subsample')
    assert reserve_distance_and_ln_weights(r) is None
    r['n_subsample'] = None
    assert reserve_distance_and_ln_weights(r) is None
    # and a reserve that CAN be corrected is still used
    r['n_subsample'] = 400
    assert reserve_distance_and_ln_weights(r) is not None


def test_equal_weights_are_exactly_n_effective_samples():
    """1/sum(p^2) over n equal weights lands a few ULP below n, which read as "fewer than
    one effective sample per bin" for a perfectly flat set at n = 3, 5, 6, 8, 9, 10."""
    for n in range(2, 16):
        assert _ess(np.zeros(n)) == float(n), 'n={} gave {!r}'.format(n, _ess(np.zeros(n)))
    d = np.linspace(100.0, 900.0, 5)
    assert distance_grid_resolution_warning(d, n_grid=5, ln_weights=np.zeros(5)) is None, \
        'five equal-weight rows over five bins was called starved'


def test_a_sampler_that_could_not_measure_its_own_neff_is_not_called_fine():
    d = np.linspace(100.0, 900.0, 40)
    ln_w = np.zeros(40)
    warned = distance_grid_resolution_warning(d, n_grid=20, ln_weights=ln_w,
                                              n_eff_sampler=float('nan'))
    assert warned is not None, 'a NaN n_eff passed the check silently'
    assert 'sampler reported' in warned


def test_the_exported_column_is_the_one_asked_for():
    """`param='psi'` exports psi values labelled as distance, and nothing in the source text
    can tell the difference."""
    rvs, reserve = _fake_rvs(), _fake_reserve()
    rec = RvsRecord.fair_draw(rvs, reserve=reserve, integrand_is_log=False)
    d, _, _, _, _ = distance_grid_inputs(rec, rvs, _uniform_fallback(rvs), n_grid=20)
    d_psi, _, _, _, _ = distance_grid_inputs(rec, rvs, _uniform_fallback(rvs), n_grid=20,
                                          param='psi')
    assert np.allclose(np.sort(d), np.sort(reserve['X'][:, 0]))
    assert np.allclose(np.sort(d_psi), np.sort(reserve['X'][:, 1]))
    assert not np.allclose(np.sort(d), np.sort(d_psi))


@pytest.mark.parametrize('make_record,why', [
    (lambda rvs, res: RvsRecord.retained(rvs, reserve=res, integrand_is_log=False),
     'a retained record is already the rows; it must not be swapped for the reserve'),
    (lambda rvs, res: None,
     'no record means the record does not describe these rows, so its reserve cannot either'),
    (lambda rvs, res: RvsRecord.pooled(rvs, resampled_blocks=[True, True], block_sizes=[3, 2],
                                       reserve=res, integrand_is_log=False),
     'a pooled record is not one sampler draws; one replica reserve is not its retained set'),
    (lambda rvs, res: RvsRecord.fair_draw(rvs, reserve=None, integrand_is_log=False),
     'no reserve at all'),
    (lambda rvs, res: RvsRecord.fair_draw(rvs, reserve={'X': np.zeros((0, 2))},
                                          integrand_is_log=False),
     'an unusable reserve'),
])
def test_the_fall_back_to_rvs_is_taken_when_it_should_be(make_record, why):
    rvs, reserve = _fake_rvs(), _fake_reserve()
    d, ln_w, _lnpi, notes, _ = distance_grid_inputs(make_record(rvs, reserve), rvs,
                                             _uniform_fallback(rvs), n_grid=20)
    assert len(d) == 5 and np.allclose(d, 900.0), why
    assert np.allclose(ln_w, 0.0), 'the fall-back weighting was not the one supplied'
    assert notes == [], 'a fall-back claimed to have used the retained set'


def test_the_population_ess_reaches_the_check_and_the_subsample_does_not():
    """A capped reserve whose kept rows look healthy but whose pass did not converge."""
    rvs = _fake_rvs()
    healthy = _fake_reserve(ess=None)
    starved = _fake_reserve(ess=1.2)

    _, _, _, _, ok = distance_grid_inputs(RvsRecord.fair_draw(rvs, reserve=healthy), rvs,
                                       _uniform_fallback(rvs), n_grid=20)
    _, _, _, _, bad = distance_grid_inputs(RvsRecord.fair_draw(rvs, reserve=starved), rvs,
                                        _uniform_fallback(rvs), n_grid=20)
    assert ok is None, 'a healthy retained set was flagged'
    assert bad is not None and 'n_eff=1.2' in bad, \
        'ess_finite never reached the resolution check'


def test_the_bin_count_the_check_sees_is_the_one_that_will_be_written():
    rvs, reserve = _fake_rvs(), _fake_reserve(ess=30.0)
    rec = RvsRecord.fair_draw(rvs, reserve=reserve, integrand_is_log=False)
    assert distance_grid_inputs(rec, rvs, _uniform_fallback(rvs), n_grid=5)[4] is None
    wide = distance_grid_inputs(rec, rvs, _uniform_fallback(rvs), n_grid=200)[4]
    assert wide is not None and 'n_eff=30.0' in wide


def test_the_sampler_neff_is_carried_through():
    rvs, reserve = _fake_rvs(), _fake_reserve()
    rec = RvsRecord.fair_draw(rvs, reserve=reserve, integrand_is_log=False)
    assert distance_grid_inputs(rec, rvs, _uniform_fallback(rvs), n_grid=20,
                                n_eff_sampler=500.0)[4] is None
    warned = distance_grid_inputs(rec, rvs, _uniform_fallback(rvs), n_grid=20,
                                  n_eff_sampler=1.4)[4]
    assert warned is not None and 'sampler reported n_eff=1.4' in warned


###
### 5. every site, including the two that need a GPU or torch to run
###

def _enclosing_function(tree, node):
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if fn.lineno <= node.lineno <= (getattr(fn, 'end_lineno', fn.lineno) or fn.lineno):
                yield fn


def _fairdraw_calls(tree):
    """Every in-place fair draw: a random.choice(..., replace=True)."""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, 'attr', None) == 'choice'
            and any(kw.arg == 'replace' and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True for kw in n.keywords)]


_RESERVE_BUILDERS = ('keep_reserve_from_rvs', 'make_reserve_from_rvs',
                     'make_warm_seed_reserve')


def _reserve_calls_in(fn):
    # Name OR Attribute: the portfolio calls mcsamplerAdaptiveVolume.make_warm_seed_reserve.
    return [n for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and (getattr(n.func, 'id', None) in _RESERVE_BUILDERS
                 or getattr(n.func, 'attr', None) in _RESERVE_BUILDERS)]


def _called_names(node):
    """Every function/method NAME called anywhere under this AST node."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                out.add(n.func.id)
            elif isinstance(n.func, ast.Attribute):
                out.add(n.func.attr)
    return out


def _keywords_of_call(node, name):
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == name:
            return set(kw.arg for kw in n.keywords)
    return set()


def _dgrid_export_block(src):
    """The AST subtree of the ILE's .dgrid export, or None.  Parsed, not sliced: a window
    of source text matches the import line and the comments that quote the names."""
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.If):
            continue
        if 'export_marginal_distance_grid' not in ast.dump(node.test):
            continue
        if 'build_distance_grid' in _called_names(node):
            return node
    return None


@pytest.mark.parametrize('module', ['mcsampler.py', 'mcsamplerGPU.py', 'mcsamplerEnsemble.py',
                                    'mcsamplerNFlow.py', 'mcsamplerAdaptiveVolume.py',
                                    'mcsamplerPortfolio.py'])
def test_every_fair_draw_site_keeps_the_retained_rows_first(module):
    """The sweep, because the four new sites were wired by one patcher: a single mistake
    would be replicated, and two of them (GPU, NFlow) cannot be driven on this host.

    PER SITE and PARSED.  Asking whether the text `keep_reserve_from_rvs(` appears anywhere
    earlier in the FILE passes when the call is commented out, and passes for
    mcsamplerGPU's second fair-draw site because its first site's call is earlier in the
    file -- both measured.  A call node in the same function at a lower line is the claim.
    """
    src = open(os.path.join(_INTEGRATORS, module)).read()
    tree = ast.parse(src)
    draws = _fairdraw_calls(tree)
    assert draws, '{}: no fair draw found; this sweep is looking at the wrong thing'.format(module)

    for draw in draws:
        kept = False
        for fn in _enclosing_function(tree, draw):
            if any(c.lineno < draw.lineno for c in _reserve_calls_in(fn)):
                kept = True
        assert kept, \
            '{}: the fair draw at line {} discards the retained rows with no reserve ' \
            'taken earlier in the same function'.format(module, draw.lineno)


def test_the_portfolio_builds_its_reserve_through_the_shared_adapter():
    """It used to build X with its own vstack, which carried both defects the adapter was
    given: a tuple parameter makes _rvs[key] (2,N) so the ravel goes ragged and the reserve
    is lost, and a term-by-term weight turns a zero-prior row into NaN instead of into no
    weight.  Two implementations of one thing drift; this pins the one."""
    src = open(os.path.join(_INTEGRATORS, 'mcsamplerPortfolio.py')).read()
    tree = ast.parse(src)
    builders = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = getattr(n.func, 'id', None) or getattr(n.func, 'attr', None)
            if name in _RESERVE_BUILDERS:
                builders.add(name)
    assert 'make_reserve_from_rvs' in builders, \
        'the portfolio builds its reserve some other way again'
    assert 'make_warm_seed_reserve' not in builders, \
        'the portfolio calls the raw builder directly, bypassing the adapter fixes'
    assert 'numpy.vstack' not in src.split('Clean out the _rvs arrays')[0][-2000:], \
        'the hand-rolled sample matrix is back'


@pytest.mark.parametrize('module', ['mcsampler.py', 'mcsamplerGPU.py', 'mcsamplerEnsemble.py',
                                    'mcsamplerNFlow.py'])
def test_each_site_declares_the_same_integrand_convention_as_its_record(module):
    """`integrand` is lnL on some backends and linear L on others, and only the sampler
    knows which.  Reading it under the wrong one exports a curve weighted by exp(L): the
    module docstring calls that "a different posterior", and flipping the flag at any of
    the three linear sites left the whole suite green.
    """
    src = open(os.path.join(_INTEGRATORS, module)).read()
    tree = ast.parse(src)
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == 'keep_reserve_from_rvs']
    assert sites, '{}: no reserve site to check'.format(module)

    for site in sites:
        mine = [kw.value for kw in site.keywords if kw.arg == 'integrand_is_log']
        assert mine, '{}:{} does not state the convention at all'.format(module, site.lineno)
        for fn in _enclosing_function(tree, site):
            records = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                       and getattr(n.func, 'attr', None) in ('retained', 'fair_draw')
                       and getattr(getattr(n.func, 'value', None), 'id', None) == 'RvsRecord']
            for rec in records:
                theirs = [kw.value for kw in rec.keywords if kw.arg == 'integrand_is_log']
                if not theirs:
                    # The record leaves it unstated because this pass writes a
                    # `log_integrand` column, which the reserve builder prefers outright --
                    # so the flag is inert here.  Check that column really is written.
                    assert "log_integrand'] =" in src or 'log_integrand"] =' in src, \
                        ('{}:{} states a convention its record leaves unstated, and the '
                         'module writes no log_integrand column'.format(module, site.lineno))
                    continue
                assert ast.dump(mine[0]) == ast.dump(theirs[0]), \
                    ('{}:{} tells the reserve builder {} while its RvsRecord says {}'
                     .format(module, site.lineno, ast.unparse(mine[0]), ast.unparse(theirs[0])))


def _bound_before(tree, fn, block):
    """Everything a name inside `block` could legally resolve to AT THAT POINT.

    Order matters, which is the whole point: UnboundLocalError is about a local assigned
    LATER, not about a name that is missing.  `params_out` is also built by the .dslice
    block further down the same function, so a presence-only check called it bound and
    passed while the .dgrid block read it before it existed.
    """
    import builtins
    lo = block.lineno
    hi = getattr(block, 'end_lineno', lo) or lo
    bound = set(dir(builtins))
    for n in ast.walk(tree):                       # module scope, any line
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split('.')[0] for a in n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
    for n in ast.iter_child_nodes(tree):
        for t in ast.walk(n) if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                   ast.ClassDef)) else []:
            if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                bound.add(t.id)
    for n in ast.walk(fn):                         # locals, but only those already bound
        earlier = n.lineno < lo if hasattr(n, 'lineno') else False
        inside = hasattr(n, 'lineno') and lo <= n.lineno <= hi
        if not (earlier or inside):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split('.')[0] for a in n.names)
    for a in ast.walk(fn):                         # parameters, wherever they appear
        if isinstance(a, ast.arg):
            bound.add(a.arg)
    return bound


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_every_name_the_export_block_reads_is_actually_bound():
    """Nothing in this suite EXECUTES the driver, so an unbound local in the export block is
    invisible to all of it.  Measured: refactoring the block deleted the `params_out` dict
    and the distance-prior columns it passes to build_distance_grid, every test stayed
    green, and the first thing that noticed was a real ILE run failing at the very end with
    UnboundLocalError -- after the integration was done.  A name-resolution pass is cheap
    and catches exactly that.
    """
    src = open(_ILE).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == 'analyze_event')
    block = _dgrid_export_block(src)
    assert block is not None, 'no .dgrid export block found; this test is looking at nothing'

    bound = _bound_before(tree, fn, block)
    loaded = set(n.id for n in ast.walk(block)
                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    missing = sorted(loaded - bound)
    assert not missing, 'the .dgrid export block reads unbound name(s): {}'.format(missing)


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_ile_delegates_the_decision_and_prints_what_comes_back():
    """All that is left in the driver is the call and the printing, so that is all this
    asserts.  The decision itself is tested by running it (section 4b) -- it used to live
    inline, where the only available test was reading the source, and nine separate ways of
    getting it wrong left that test green.
    """
    block = _dgrid_export_block(open(_ILE).read())
    assert block is not None, 'no .dgrid export block found; this test is looking at nothing'
    called = _called_names(block)

    assert 'distance_grid_inputs' in called, \
        'the exporter decides for itself again instead of calling the tested function'
    for gone in ('reserve_distance_and_ln_weights', 'distance_grid_resolution_warning'):
        assert gone not in called, \
            'the exporter re-implements {} inline, where it cannot be run'.format(gone)
    kw = _keywords_of_call(block, 'distance_grid_inputs')
    for need in ('convert', 'n_grid', 'n_eff_sampler'):
        assert need in kw, 'the decision is made without {}'.format(need)

    dumped = ast.dump(block)
    # the results must actually reach the log
    prints = [n for n in ast.walk(block)
              if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'print']
    assert any('_dgrid_notes' in ast.dump(n) or '_note' in ast.dump(n)
               for n in ast.walk(block) if isinstance(n, ast.For)), \
        'the notes the decision returns are never printed'
    assert any('_dgrid_warn' in ast.dump(n) and 'WARNING' in ast.dump(n) for n in prints), \
        'the warning is computed and never printed'

    # A retry that succeeds must clear the previous attempt's marker, or the census the
    # handler exists to support counts a point that was in fact exported.
    assert dumped.count("'.dgrid.skipped'") >= 2, \
        'the sidecar path is built once, so nothing clears a stale one'
    assert any(isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'remove'
               for n in ast.walk(block)), \
        'a successful export never removes a stale .dgrid.skipped sidecar'
