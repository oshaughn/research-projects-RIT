#!/usr/bin/env python
"""
Regression tests for DOUBLE-WEIGHTING a fair-drawn `_rvs` record
(bin/integrate_likelihood_extrinsic_batchmode, RIFT/integrators/*).

Found by the mechanical `_rvs` audit
(test/expensive_before_merging/integrators/audit_rvs_fairdraw.py).

THE DEFECT.  At the end of integrate_log the fair draw replaces every `_rvs` key with rows
resampled WITH REPLACEMENT proportional to the importance weight `w`.  Those rows are then an
EQUAL-WEIGHT draw from the posterior.  Three consumers went on to weight them by `w` again --
so the product follows `w^2` and is over-concentrated:

  * the `--extrinsic-proposal-output` GMM breadcrumb, whose proposal is handed to the NEXT
    iteration, so the truncation compounds across iterations;
  * the `.dgrid` distance-posterior exporter;
  * the `.dslice` reweight core (a different shape: it double-counts pi_Omega/q_Omega and
    takes N from the resample, and cannot be corrected after the fact -- so it is routed to
    the exact all-fresh path instead).

`_pool_replica_rvs` has guarded against exactly this since the replica work, via
`already_resampled`.  None of these three had an equivalent.

THE PREDICATE MATTERS.  `opts.fairdraw_extrinsic_output` is NOT the same question as "did the
draw fire": the samplers skip it when it would not shrink the record, and then the rows still
carry real importance weights that must be applied.  So the samplers mark the rebind itself.
"""

import os

import numpy as np
import pytest

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
NDIM = len(NAMES)

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')
_INTEGRATORS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'RIFT', 'integrators')

SAMPLER_FILES = ['mcsampler.py', 'mcsamplerAdaptiveVolume.py', 'mcsamplerEnsemble.py',
                 'mcsamplerGPU.py', 'mcsamplerNFlow.py', 'mcsamplerPortfolio.py']


def _load_ile_helpers():
    """Exec the weight helpers out of the ILE script (not importable: it parses argv)."""
    src = open(_ILE).read()
    start = src.index("def ln_weights_from_rvs")
    end = src.index("def _pool_replica_rvs")
    ns = {"numpy": np, "np": np, "_rvs_lnL_convention": lambda x=None: bool(x)}
    exec(compile(src[start:end], "ile_weight_helpers", "exec"), ns)
    return ns


H = _load_ile_helpers()
ln_weights_from_rvs = H["ln_weights_from_rvs"]
ln_weights_for_posterior = H["ln_weights_for_posterior"]
_rvs_is_export_resample = H["_rvs_is_export_resample"]


class _FakeSampler(object):
    def __init__(self, fairdrawn):
        self._rvs_is_fairdraw = fairdrawn


def _record(n=200, seed=3):
    rng = np.random.default_rng(seed)
    lnL = rng.normal(0.0, 3.0, size=n)
    return {"log_integrand": lnL,
            "log_joint_prior": np.zeros(n),
            "log_joint_s_prior": np.zeros(n),
            "x": rng.normal(size=n)}


def _sampler(n_chunk=20000):
    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = mcsamplerAV.xpy_default
    s.identity_convert = mcsamplerAV.identity_convert
    for name in NAMES:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    return s


def _peaked(rho):
    x0 = 0.5 * np.ones(NDIM)
    w = (0.5 / rho) * np.ones(NDIM)
    lnLmax = 0.5 * rho ** 2

    def lnL(*args, **kwargs):
        x = np.array([np.asarray(a, dtype=float).ravel() for a in args]).T
        out = lnLmax - 0.5 * np.sum(((x - x0) / w) ** 2, axis=-1)
        return np.where(out > lnLmax - 745.0, out, -np.inf)
    return lnL


###
### 1. the helper itself
###

def test_a_fair_drawn_record_gets_uniform_weights():
    r = _record()
    lw = ln_weights_for_posterior(r, _FakeSampler(True))
    assert lw.shape == (len(r["log_integrand"]),)
    assert np.allclose(lw, 0.0), 'a fair draw is already equal-weight; w must not be reapplied'


def test_a_retained_record_still_gets_its_importance_weights():
    """The other direction is just as wrong: flattening a record the draw never touched."""
    r = _record()
    lw = ln_weights_for_posterior(r, _FakeSampler(False))
    assert np.allclose(lw, ln_weights_from_rvs(r)), \
        'a record that was NOT resampled must keep its importance weights'
    assert np.std(lw) > 1.0, 'these weights are not degenerate; flattening them loses the shape'


def test_the_helper_defaults_to_importance_weights_when_the_flag_is_absent():
    """An older sampler, or one hand-built in a test, must not be silently flattened."""
    class _Bare(object):
        pass
    r = _record()
    assert not _rvs_is_export_resample(_Bare())
    assert np.allclose(ln_weights_for_posterior(r, _Bare()), ln_weights_from_rvs(r))


###
### 2. the numbers: re-weighting a fair draw shifts the answer
###

def test_reweighting_a_fair_draw_biases_a_posterior_mean():
    """The measurement behind the fix, at small scale: on a coordinate correlated with the
    weight, applying w twice moves the posterior mean well outside its own MC error."""
    rng = np.random.default_rng(11)
    n = 20000
    lnL = rng.normal(0.0, 4.0, size=n)
    x = 0.5 * lnL + 0.5 * rng.normal(size=n)
    rvs = {"log_integrand": lnL, "log_joint_prior": np.zeros(n),
           "log_joint_s_prior": np.zeros(n), "x": x}
    w = np.exp(lnL - lnL.max()); w /= w.sum()
    truth = float(np.sum(w * x))

    idx = rng.choice(np.arange(n), size=4000, replace=True, p=w)   # the rebind
    drawn = {k: np.asarray(v)[idx] for k, v in rvs.items()}

    correct = float(np.mean(ln_weights_for_posterior(drawn, _FakeSampler(True)) * 0
                            + drawn["x"]))
    lw = ln_weights_from_rvs(drawn)
    ww = np.exp(lw - lw.max()); ww /= ww.sum()
    doubled = float(np.sum(ww * drawn["x"]))

    assert abs(correct - truth) < 0.1, 'the unweighted fair draw should recover the truth'
    assert doubled - truth > 0.5, \
        'expected a clear upward bias from w^2, got {:+.3f}'.format(doubled - truth)


###
### 3. the samplers mark the rebind, and only when it fires
###

def test_the_sampler_marks_the_record_when_the_fair_draw_fires():
    np.random.seed(20260813)
    s = _sampler()
    s.integrate_log(_peaked(100.0), *NAMES, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    assert _rvs_is_export_resample(s), 'the fair draw fired but left no marker'


def test_the_sampler_does_not_mark_the_record_when_no_fair_draw_was_asked_for():
    np.random.seed(20260813)
    s = _sampler()
    s.integrate_log(_peaked(100.0), *NAMES, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False)
    assert not _rvs_is_export_resample(s)


def test_the_marker_is_reset_per_pass_rather_than_latching():
    """Samplers are reused across events (--n-events-to-analyze), so a latched True would
    flatten the weights of every later pass that did not resample."""
    np.random.seed(20260813)
    s = _sampler()
    s.integrate_log(_peaked(100.0), *NAMES, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    assert _rvs_is_export_resample(s)
    s.integrate_log(_peaked(100.0), *NAMES, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False)
    assert not _rvs_is_export_resample(s), 'the marker latched across passes'


@pytest.mark.parametrize('fname', SAMPLER_FILES)
def test_every_sampler_with_a_fair_draw_marks_it(fname):
    """Fix the twin: all seven rebind sites, not just the one that was being edited."""
    src = open(os.path.join(_INTEGRATORS, fname)).read()
    if 'bFairdraw' not in src:
        pytest.skip('{} has no fair-draw block'.format(fname))
    n_anchor = src.count('bFairdraw  = kwargs[') + src.count('bFairdraw = kwargs[')
    assert src.count('self._rvs_is_fairdraw = True') == n_anchor, \
        '{}: {} fair-draw block(s) but {} marker(s) -- a rebind is unmarked'.format(
            fname, n_anchor, src.count('self._rvs_is_fairdraw = True'))
    assert src.count('self._rvs_is_fairdraw = False') == n_anchor, \
        '{}: a fair-draw block never resets the marker, so it latches across passes'.format(fname)


###
### 4. the consumers actually use it
###

@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('anchor', [
    '_ext = _ehmod.fit_extrinsic_proposal',        # extrinsic-proposal breadcrumb
    'dgrid = build_distance_grid(',                # .dgrid
])
def test_the_weighted_exporters_ask_for_posterior_weights(anchor):
    src = open(_ILE).read()
    i = src.index(anchor)
    block = src[max(0, i - 2500):i]
    assert 'ln_weights_for_posterior' in block, \
        'this exporter still applies importance weights to a possibly fair-drawn record'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_dslice_reweight_core_refuses_a_fair_drawn_record():
    src = open(_ILE).read()
    i = src.index('all_fresh = bool(getattr(opts, "distance_slice_all_fresh"')
    block = src[i:i + 1800]
    assert '_rvs_is_export_resample(sampler)' in block, \
        'the .dslice reweight core still runs on the fair-draw export'
    assert 'all_fresh = True' in block, 'it does not route to the exact fresh path'


###
### 5. pooled n_eff after replica pooling
###

def _block_kish(rep_lnZ, rep_neff):
    """The formula the ILE now uses when the pooled export is fair-drawn."""
    l = np.asarray(rep_lnZ, float) - float(np.max(rep_lnZ))
    Z = np.exp(l)
    n = np.asarray(rep_neff, float)
    ok = np.isfinite(Z) & np.isfinite(n) & (n > 0)
    return float(np.sum(Z[ok]) ** 2 / np.sum(Z[ok] ** 2 / n[ok]))


def test_pooled_neff_reduces_to_the_sum_when_replicas_agree():
    """The property the original comment asked for, and the sanity check on the formula."""
    for K, ne in ((3, 40.0), (5, 12.5), (2, 100.0)):
        assert _block_kish([7.0] * K, [ne] * K) == pytest.approx(K * ne, rel=1e-9)


def test_pooled_neff_falls_below_the_sum_when_replicas_disagree():
    """Disagreement is the whole reason the replicas are run; it must show up here."""
    agree = _block_kish([7.0, 7.0, 7.0], [40.0] * 3)
    disagree = _block_kish([7.0, 7.0, 11.0], [40.0] * 3)   # one replica 4 nats high
    assert disagree < agree
    assert disagree < 0.5 * agree, 'a 4-nat outlier barely moved the pooled n_eff'


def test_pooled_neff_is_not_the_export_row_count():
    """The defect: Kish of the FLATTENED pooled record is just its row count.

    _pool_replica_rvs forces equal weights within each block when the input is fair-drawn, so
    the Kish n_eff of that record equals the number of exported rows -- which is
    K*min(n_max, 1.5*eff_samp, 1.5*neff), the size of the export, not the quality of the
    integral.  At the default --fairdraw-extrinsic-output-n-max 5 that is 5K.
    """
    K, n_k = 4, 5                      # 5 exported rows per replica, the default cap
    lnZ_k, neff_k = 7.0, 60.0
    flat_lw = np.concatenate([
        np.full(n_k, lnZ_k - np.log(K) - np.log(n_k)) for _ in range(K)])
    w = np.exp(flat_lw - flat_lw.max())
    kish_of_export = float(np.sum(w) ** 2 / np.sum(w ** 2))
    assert kish_of_export == pytest.approx(K * n_k), 'flat weights: Kish IS the row count'
    assert kish_of_export == pytest.approx(20.0)
    assert _block_kish([lnZ_k] * K, [neff_k] * K) == pytest.approx(K * neff_k)
    assert _block_kish([lnZ_k] * K, [neff_k] * K) > 10 * kish_of_export, \
        'the export row count understates the pooled n_eff by more than an order of magnitude'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_ile_uses_the_block_form_only_for_a_fair_drawn_export():
    """A record that was never resampled still has real per-row weights, and the pooled Kish
    over them is finer-grained than the block form -- so the switch must be conditional."""
    src = open(_ILE).read()
    i = src.index('_neff_pooled')
    block = src[i - 1500:i + 2000]
    assert '_rvs_is_export_resample(sampler)' in block, 'the switch is unconditional'
    assert '_kish_neff_of_rvs(sampler._rvs)' in block, \
        'the non-fairdraw path no longer uses the pooled Kish'


###
### 6. the L0 reject threshold, pinned to its measurement
###

@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_l0_reject_threshold_is_the_measured_default():
    """0.5 was strictly dominated and is not a safe value to drift back to.

    Measured over 160 known-lnZ passes (L0_REJECT_DLNZ_MEASUREMENT.md): the gate caught 0 of
    55 genuinely truncated warm passes at EVERY threshold from 0.25 to 4.0 nats, on both AV
    and portfolio.  Meanwhile at 0.5 it rejected 25% of GOOD portfolio warm passes, keeping a
    collapsed cold result instead.  So 0.5 bought no detection and cost one good pass in four;
    3.0 costs ~0% and buys the same nothing, while still catching a genuinely large gap.
    """
    src = open(_ILE).read()
    i = src.index('--sampler-l0-rescue-reject-dlnZ')
    decl = src[i:i + 400]
    assert 'default=3.0' in decl, \
        'the L0 reject threshold moved off its measured value; re-measure before changing it'
    assert 'default=0.5' not in decl


###
### 7. COMPOSITION: the marker and the reserve must agree with the record beside them
###
### Both defects below were found in review, and both are the same shape: a fix that is
### correct in isolation and wrong once another code path runs after it.  The tests above
### exercise the pieces separately and would pass with both bugs present.
###
### NOTE ON WHAT CARRIES THE WEIGHT HERE.  The helpers below (_pool_replica_rvs,
### _snapshot_pass_state / _restore_pass_state) are correct in isolation and were correct
### with both bugs present -- the bugs were at the CALL SITES, in analyze_event, which needs
### data, PSDs and a waveform to run and cannot be exercised from a unit test.  So the
### behavioural tests pin the contracts the call sites depend on, and the source-level tests
### pin the wiring.  Verified by reverting each fix: only the wiring tests fail.  If a future
### change makes analyze_event callable in pieces, promote these.
###

def _load_pool_helpers():
    """Exec ln_weights_* plus _pool_replica_rvs and its dependencies."""
    src = open(_ILE).read()
    start = src.index("def ln_weights_from_rvs")
    end = src.index("def _warm_seed_geometry")
    ns = {"numpy": np, "np": np, "_rvs_lnL_convention": lambda x=None: bool(x),
          "mcsamplerAdaptiveVolume": mcsamplerAV}
    exec(compile(src[start:end], "ile_pool_helpers", "exec"), ns)
    return ns


P = _load_pool_helpers()


class _ConvSampler(object):
    """Minimal stand-in for the sampler interface _pool_replica_rvs uses."""
    def __init__(self):
        self._rvs_is_fairdraw = True

    @staticmethod
    def identity_convert(x):
        return x


def _fairdrawn_block(n, seed):
    """An equal-weight posterior draw: the rows a fair draw leaves behind."""
    rng = np.random.default_rng(seed)
    lnL = rng.normal(0.0, 2.0, size=n)
    return {"log_integrand": lnL,
            "log_joint_prior": np.zeros(n),
            "log_joint_s_prior": np.zeros(n),
            "x": rng.normal(size=n)}


def test_a_pooled_record_is_not_globally_equal_weight():
    """P1: each block is equal-weight WITHIN itself, but blocks differ by their evidences.

    Treating the pooled record as a fair draw makes .dgrid and the proposal breadcrumb mix the
    replicas by exported ROW COUNT instead of by evidence -- discarding exactly the replica
    disagreement the replicas were run to measure.
    """
    rep_lnZ = [7.0, 9.0]                       # a 2-nat disagreement, i.e. e^2 in evidence
    reps = [_fairdrawn_block(60, 1), _fairdrawn_block(60, 2)]
    pooled = P["_pool_replica_rvs"](reps, _ConvSampler(), rep_lnZ=rep_lnZ,
                                    already_resampled=True, use_lnL=False)
    lw = P["ln_weights_from_rvs"](pooled)
    assert np.ptp(lw) > 1.0, 'the pooled record came out globally flat; the evidences are gone'
    a, b = lw[:60], lw[60:]
    assert np.allclose(a, a[0]) and np.allclose(b, b[0]), 'within a block weights must be equal'
    assert b[0] - a[0] == pytest.approx(rep_lnZ[1] - rep_lnZ[0], abs=1e-9), \
        'the between-block offset must be exactly the evidence difference'


def test_the_fairdraw_marker_is_cleared_once_the_record_is_pooled():
    """...so ln_weights_for_posterior reads those reconstructed block weights."""
    src = open(_ILE).read()
    i = src.index('_pool_replica_rvs(_rep_rvs')
    block = src[i:i + 2200]
    assert '_rvs_is_fairdraw = False' in block, \
        'the marker survives pooling; the pooled record would be treated as equal-weight'
    assert 'is _r for _r in _rep_rvs' in block, \
        'the marker must only be cleared when pooling actually happened -- every fallback in ' \
        '_pool_replica_rvs returns an INPUT record, which is still a fair draw'


def test_posterior_weights_of_a_pooled_record_keep_the_replica_evidences():
    """End to end through the helper the exporters actually call."""
    rep_lnZ = [7.0, 9.0]
    reps = [_fairdrawn_block(40, 5), _fairdrawn_block(40, 6)]
    s = _ConvSampler()
    pooled = P["_pool_replica_rvs"](reps, s, rep_lnZ=rep_lnZ,
                                    already_resampled=True, use_lnL=False)
    s._rvs_is_fairdraw = False              # what the ILE now does after pooling
    lw = P["ln_weights_for_posterior"](pooled, s)
    assert lw[40] - lw[0] == pytest.approx(rep_lnZ[1] - rep_lnZ[0], abs=1e-9)
    # and the bug: had the marker survived, every row would weigh the same
    s._rvs_is_fairdraw = True
    assert np.allclose(P["ln_weights_for_posterior"](pooled, s), 0.0)


def test_a_fallback_pool_keeps_the_marker():
    """One replica, or a record with no sampling-prior column, comes back unchanged and is
    still the fair draw it arrived as."""
    one = [_fairdrawn_block(30, 9)]
    out = P["_pool_replica_rvs"](one, _ConvSampler(), rep_lnZ=[7.0],
                                 already_resampled=True, use_lnL=False)
    assert out is one[0], 'a single replica must come back as the same object'


def test_rejecting_the_warm_pass_restores_the_cold_reserve():
    """P1: the reject path put back _rvs, the estimate and the diagnostics, but left
    _warm_seed_reserve holding the REJECTED warm cloud -- which --sampler-sequential-warmstart
    then seeds the next intrinsic point from.  Snapshot and restore must move together."""
    ns = {}
    src = open(_ILE).read()
    start = src.index("def _snapshot_pass_state")
    end = src.index("def _warm_seed_geometry")
    exec(compile(src[start:end], "ile_state_helpers", "exec"), ns)

    class _S(object):
        pass
    s = _S()
    s._rvs = {"x": np.arange(5)}
    s._warm_seed_reserve = {"tag": "cold"}
    s._rvs_is_fairdraw = True
    s.portfolio_realizations = []
    state = ns["_snapshot_pass_state"](s, 1.0, 2.0, 3.0, {"d": "cold"})

    # the warm pass runs and overwrites everything in place
    s._rvs = {"x": np.arange(2)}
    s._warm_seed_reserve = {"tag": "warm"}
    s._rvs_is_fairdraw = False

    res, var, neff, dd = ns["_restore_pass_state"](s, state)
    assert (res, var, neff, dd) == (1.0, 2.0, 3.0, {"d": "cold"})
    assert s._warm_seed_reserve == {"tag": "cold"}, \
        'the rejected warm reserve survived; the next point would seed from it'
    assert s._rvs_is_fairdraw is True, 'the marker must describe the restored record'
    assert ns["_warm_seed_reserve_for"] is not None


def test_the_restore_reaches_portfolio_member_reserves_too():
    """_warm_seed_reserve_for falls through to portfolio_realizations, so restoring only the
    aggregate would leave that fallback pointing at the rejected warm pass."""
    ns = {}
    src = open(_ILE).read()
    start = src.index("def _snapshot_pass_state")
    end = src.index("def _warm_seed_geometry")
    exec(compile(src[start:end], "ile_state_helpers", "exec"), ns)

    class _S(object):
        pass
    m = _S(); m._warm_seed_reserve = {"tag": "cold-member"}
    s = _S(); s._rvs = {}; s._warm_seed_reserve = None
    s._rvs_is_fairdraw = False; s.portfolio_realizations = [m]
    state = ns["_snapshot_pass_state"](s, 0, 0, 0, {})
    m._warm_seed_reserve = {"tag": "warm-member"}
    ns["_restore_pass_state"](s, state)
    assert m._warm_seed_reserve == {"tag": "cold-member"}


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_both_l0_restore_paths_go_through_the_shared_helper():
    """The reject path and the exception handler must not drift in WHAT they put back."""
    src = open(_ILE).read()
    assert src.count('_restore_pass_state(sampler, _cold_state_l0)') == 2, \
        'expected the reject path and the failure handler to share one restore'
    assert 'sampler._rvs, res, var, neff, dict_return = _cold_state_l0' not in src, \
        'the old partial tuple restore is back'
    # and no hand-rolled partial restore alongside it: the reject branch must not poke _rvs
    # directly, which is precisely what left the reserve describing the rejected warm pass.
    i = src.index('keeping the COLD (full-support) result')
    branch = src[i:i + 1200]
    assert 'sampler._rvs =' not in branch, \
        'the reject branch assigns _rvs directly again; the reserve and marker will not follow'
