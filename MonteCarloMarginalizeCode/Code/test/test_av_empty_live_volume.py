#!/usr/bin/env python
"""
Regression tests for the adaptive-volume EMPTY LIVE VOLUME crash
(RIFT/integrators/mcsamplerAdaptiveVolume.py).

Background (the bug these tests lock down).  At high network SNR the production
extrinsic likelihood underflows: exp() of a lnL more than ~745 nats below the peak
returns 0, so the likelihood hands back -inf.  Over a cold extrinsic prior at
rho_net ~ 147 that is ~99.996% of draws (measured: 99996 of 100000), and the
adaptive-volume live set is therefore born holding a handful of samples, sometimes
one.  Two things then went wrong, in sequence:

  1. get_likelihood_threshold returned a threshold >= max(lkl).  With a small live
     set the `len(lkl) > nsel` branch falls through to lkl_stop_thr = lkl[-1] (the
     array MINIMUM), while prob_stop_thr saturates at the MAXIMUM because every
     other weight underflows to zero, so the discard_prob quantile IS the top
     sample.  min(min, max) is then the live set's own minimum -- and with only one
     sample left, its maximum.
  2. integrate_log applies that threshold as a STRICT `allloglkl > loglkl_thr`, so
     it discarded at least one sample per cycle regardless of merit, ratcheting the
     live set down to 1 and then to 0.

The empty array then reached `lw = allloglkl - xpy_here.max(allloglkl)` and raised

    ValueError: zero-size array to reduction operation CUPY_CUB_MAX which has no identity

which bin/integrate_likelihood_extrinsic_batchmode reported as "Probable reasons:
SEOB nyquist or starting frequency limit or signal duration".  Measured crash rate
by network SNR on zero-noise injections at a fixed intrinsic point:
rho 51.4 -> 0/12, rho 72.1 -> 4/129, rho 102.8 -> 5/12, rho 146.8 -> 11/12.

NOT CUPY-SPECIFIC.  The traceback names CUPY_CUB_MAX only because production runs
on the GPU; numpy raises the identical ValueError ("zero-size array to reduction
operation maximum which has no identity") from the same line.  These tests run on
whichever backend the sampler picked, and the cupy path is exercised as well when a
GPU is present.

The requirement is not merely "does not crash": a degenerate contraction must be
REPORTED (dict_return['live_volume_collapsed']) rather than silently exporting the
one surviving sample as if it were a posterior.
"""

import numpy as np
import pytest

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV
from RIFT.integrators.mcsamplerAdaptiveVolume import (
    LiveVolumeCollapse,
    get_likelihood_threshold,
    live_volume_collapse_verdict,
)

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
NDIM = len(NAMES)

xpy = mcsamplerAV.xpy_default
to_backend = mcsamplerAV.identity_convert_togpu
to_host = mcsamplerAV.identity_convert


def _sampler(n_chunk=10000):
    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    # Bind the sampler to the ACTIVE backend exactly as bin/integrate_likelihood_extrinsic_
    # batchmode does (`sampler.xpy = xpy_default; sampler.identity_convert = ...`).
    # MCSampler.__init__ defaults self.xpy to numpy, so on a GPU host an unconfigured
    # sampler mixes cupy arrays with numpy calls and dies in the fairdraw block for
    # reasons that have nothing to do with what is under test here.
    s.xpy = xpy
    s.identity_convert = to_host
    for name in NAMES:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)),
                        adaptive_sampling=True)
    return s


def _integrate(fn, nmax=200000, neff=8, n_chunk=10000):
    s = _sampler(n_chunk)
    res = s.integrate_log(fn, *NAMES, nmax=nmax, neff=neff, n=n_chunk,
                          no_protect_names=True, verbose=False,
                          igrand_fairdraw_samples=True,
                          igrand_fairdraw_samples_max=200)
    return s, res


###
### 1. get_likelihood_threshold must never return a threshold that empties the set
###
# This is the root defect.  The threshold is consumed as a STRICT `lkl > thr`, so a
# threshold at or above max(lkl) encloses zero probability -- which contradicts the
# enc_prob = 0.999 the function exists to maintain.

@pytest.mark.parametrize('n', [1, 2, 3, 10, 999])
def test_threshold_never_discards_the_entire_live_volume(n):
    """The regression: small live set + one dominant weight -> thr was max(lkl)."""
    # lnL values spread far enough apart that exp(lkl-max) underflows for all but the top,
    # which is exactly the high-SNR condition that saturates prob_stop_thr at the maximum.
    lkl_host = 10000.0 + 1000.0 * np.arange(n)
    lkl = to_backend(lkl_host)
    thr, truncp = get_likelihood_threshold(lkl, -1e15, 1000, 1e-3, xpy_here=xpy)
    assert float(thr) < float(lkl_host.max()), \
        "threshold {} >= max {}: strict `>` would empty the live volume".format(thr, lkl_host.max())
    assert int(np.sum(to_host(lkl) > thr)) >= 1


def test_threshold_survives_an_all_equal_live_volume():
    """No contraction is possible when every sample has the same lnL; keep them all."""
    lkl = to_backend(np.full(5, 123.5))
    thr, _ = get_likelihood_threshold(lkl, -1e15, 1000, 1e-3, xpy_here=xpy)
    assert float(thr) < 123.5
    assert int(np.sum(to_host(lkl) > thr)) == 5


def test_threshold_on_a_single_sample_keeps_it():
    lkl = to_backend(np.array([42.0]))
    thr, _ = get_likelihood_threshold(lkl, -1e15, 1000, 1e-3, xpy_here=xpy)
    assert float(thr) < 42.0


def test_threshold_on_an_empty_live_volume_raises_a_named_error():
    """Not a bare 'zero-size array to reduction operation ...' from inside a reduction."""
    with pytest.raises(LiveVolumeCollapse):
        get_likelihood_threshold(to_backend(np.array([])), -1e15, 1000, 1e-3, xpy_here=xpy)


@pytest.mark.parametrize('scale', [1.0, 5.0])
def test_clamp_is_inert_when_the_live_set_is_well_populated(scale):
    """The clamp must not move the threshold in the regime production actually runs in.

    Reference value is the PRE-FIX formula, reproduced here verbatim, so this test fails
    if the clamp ever starts engaging on a healthy live set (which would silently shift
    every production lnZ).
    """
    rng = np.random.RandomState(20260810)
    lkl_host = rng.normal(0.0, scale, size=20000)
    nsel, discard_prob = 1000, 1e-3

    # --- the original (unclamped) threshold, verbatim from the pre-fix implementation
    w = np.exp(lkl_host - np.max(lkl_host))
    prob = w / np.sum(w)
    idx = np.argsort(prob)
    ecdf = np.cumsum(prob[idx])
    prob_stop_thr = lkl_host[idx][ecdf >= discard_prob][0]
    srt = np.sort(lkl_host)[::-1]
    lkl_stop_thr = srt[nsel] if len(srt) > nsel else srt[-1]
    expected = min(lkl_stop_thr, prob_stop_thr)
    # ---

    thr, _ = get_likelihood_threshold(to_backend(lkl_host), -1e15, nsel, discard_prob, xpy_here=xpy)
    assert float(thr) == pytest.approx(float(expected)), 'clamp engaged on a healthy live set'
    assert int(np.sum(lkl_host > float(thr))) > 1


###
### 2. integrate_log must not crash on the two routes measured in production
###

def _lone_survivor(*args):
    """Exactly one finite draw per chunk: live set of size 1 on cycle 1.

    Signature in production: the crash arrives BEFORE any per-cycle line is printed
    (run/logs/snr140_cold_801.log, rho_net 146.8).
    """
    x = np.array(args).T
    out = np.full(len(x), -np.inf)
    out[0] = 100.0
    return out


def _ratchet_to_one(*args):
    """A few distinct finite values, never more, so `>` sheds one sample per cycle.

    Signature in production: int_var 0.7071 (2 samples) -> 0.0 (1 sample) -> crash
    (run/logs/cold_125.log, rho_net 72.1).
    """
    x = np.array(args).T
    out = np.full(len(x), -np.inf)
    k = min(3, len(x))
    out[:k] = 100.0 + np.arange(k)
    return out


@pytest.mark.parametrize('fn,label', [(_lone_survivor, 'lone survivor'),
                                      (_ratchet_to_one, 'ratchet to one')])
def test_degenerate_live_volume_does_not_raise_an_empty_reduction(fn, label):
    try:
        s, res = _integrate(fn)
    except ValueError as e:                     # the exact regression
        if 'zero-size array' in str(e) or 'no identity' in str(e):
            pytest.fail("empty-live-volume crash returned ({}): {}".format(label, e))
        raise
    assert np.isfinite(float(res[0])), "lnZ must be a real number, got {}".format(res[0])
    assert len(s._rvs['log_integrand']) >= 1


def test_no_finite_sample_anywhere_raises_a_named_error_not_a_reduction_error():
    """Nothing finite ever: there IS no integral, so fail -- but say why."""
    def all_underflowed(*args):
        return np.full(len(np.array(args).T), -np.inf)

    with pytest.raises(LiveVolumeCollapse) as excinfo:
        _integrate(all_underflowed, nmax=30000)
    msg = str(excinfo.value).lower()
    assert 'underflow' in msg or 'no finite value' in msg
    # the misattribution this whole investigation chased down must not come back: the
    # message may MENTION nyquist, but only to rule it out.
    assert 'not a waveform nyquist' in msg


###
### 3. A degenerate contraction must be REPORTED, not silently exported
###
# The failure mode that survives the crash fix is worse than the crash: an export
# built from one sample, indistinguishable in the output from a converged one.

def _peaked(rho, underflow=True):
    """6-D Gaussian at lnL scale rho^2/2, with the float64 underflow of the real code."""
    x0 = 0.5 * np.ones(NDIM)
    width = 0.5 / rho
    lnLmax = 0.5 * rho ** 2

    def lnL(*args):
        x = np.array(args).T
        out = lnLmax - 0.5 * np.sum(((x - x0) / width) ** 2, axis=-1)
        if underflow:
            out = np.where(out > lnLmax - 745.0, out, -np.inf)
        return out
    return lnL


def test_collapse_is_flagged_in_dict_return_at_high_snr():
    """rho ~ 147 with a production-sized chunk: a few finite draws, then a degenerate
    contraction.  The run COMPLETES -- and must say that its answer is degenerate."""
    np.random.seed(20260810)
    s, res = _integrate(_peaked(146.8), nmax=300000, neff=8, n_chunk=100000)
    dd = res[3]
    assert dd.get('live_volume_collapsed') is True, dd
    assert dd.get('collapse_reason')
    assert dd.get('n_live_final') is not None


@pytest.mark.parametrize('n_live,ess,khat,expect', [
    # (live samples, ESS, k-hat) -> should the verdict call this collapsed?
    (1,    1.0,  None, True),    # rho 146.8 seed 801: one sample exported
    (2,    1.7,  21.5, True),    # rho 146.8 seed 809: slipped an earlier ESS<1.5-only rule
    (6,    3.0,  40.0, True),    # live set no larger than the dimension
    (4000, 16.8, 1.03, False),   # rho 51.4, measured healthy
    (4000, 20.2, 1.50, False),   # rho 51.4, measured healthy
    (4000, 4.0,  1.20, False),   # hard but not degenerate: low ESS, well-behaved tail
])
def test_collapse_verdict_matches_the_measured_regimes(n_live, ess, khat, expect):
    """Pin the decision boundary against the regimes measured on the real problem."""
    collapsed, reasons = live_volume_collapse_verdict(n_live, NDIM, ess=ess, khat=khat)
    assert collapsed is expect, reasons
    assert bool(reasons) is expect


def test_collapse_verdict_reports_loop_evidence_even_when_the_stats_look_fine():
    collapsed, reasons = live_volume_collapse_verdict(4000, NDIM, ess=18.0, khat=1.2,
                                                      n_empty_cycles=3)
    assert collapsed is True
    assert 'no finite in-volume sample' in '; '.join(reasons)


def test_a_healthy_run_is_not_flagged_as_collapsed():
    """The guard must not cry wolf on the regime that already works (rho ~ 51)."""
    np.random.seed(20260810)
    s, res = _integrate(_peaked(20.0), nmax=400000, neff=20, n_chunk=20000)
    dd = res[3]
    assert dd.get('live_volume_collapsed') is False, dd.get('collapse_reason')
    assert dd['n_live_final'] > 2


###
### 4. The fix must be inert on well-conditioned integrals
###
# The clamp changes the threshold only when the threshold would have emptied the live
# volume; anything else would silently move production lnZ values.

def test_known_gaussian_integral_is_recovered():
    """int over [0,1]^6 of exp(-|x-x0|^2/2w^2) = (w sqrt(2pi))^6 for w << 1."""
    w = 0.08
    x0 = 0.5 * np.ones(NDIM)

    def lnL(*args):
        x = np.array(args).T
        return -0.5 * np.sum(((x - x0) / w) ** 2, axis=-1)

    np.random.seed(20260810)
    s, res = _integrate(lnL, nmax=600000, neff=30, n_chunk=20000)
    expected = NDIM * np.log(w * np.sqrt(2 * np.pi))
    # Tolerance is loose on purpose.  At this seed AV lands ~0.20 nats high (-9.4367 vs
    # -9.6407), which is a PRE-EXISTING property of the estimator on this problem, not an
    # effect of the collapse fix: the pinned pre-fix tree returns the identical value to
    # every digit (verified against ba2b38da, same seed, ntotal 60292 both).  The purpose
    # of this test is to catch a gross regression in the integral, not to grade AV's bias.
    assert float(res[0]) == pytest.approx(expected, abs=0.35)
    assert res[3].get('live_volume_collapsed') is False


###
### 5. Backend coverage
###
# The reported traceback is the cupy flavour (CUPY_CUB_MAX).  The tests above run on
# whatever backend is active; when a GPU is present, pin the cupy path explicitly so a
# CPU-only CI run can never be mistaken for coverage of the reported configuration.

@pytest.mark.skipif(not mcsamplerAV.cupy_ok, reason='no cupy/GPU on this host')
def test_threshold_clamp_on_the_cupy_backend():
    import cupy
    lkl = cupy.asarray(np.array([10000.0, 11000.0, 12000.0]))
    thr, _ = get_likelihood_threshold(lkl, -1e15, 1000, 1e-3, xpy_here=cupy)
    assert float(thr) < 12000.0
    assert int(cupy.sum(lkl > thr).get()) >= 1


@pytest.mark.skipif(not mcsamplerAV.cupy_ok, reason='no cupy/GPU on this host')
def test_degenerate_live_volume_on_the_cupy_backend():
    try:
        s, res = _integrate(_lone_survivor)
    except ValueError as e:
        if 'CUPY_CUB_MAX' in str(e) or 'zero-size array' in str(e):
            pytest.fail("the reported cupy crash is back: {}".format(e))
        raise
    assert np.isfinite(float(res[0]))


###
### 6. Wiring: the ILE script must not re-attribute an integrator collapse to the waveform
###

import os

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_does_not_blame_the_waveform_unconditionally():
    with open(_ILE) as f:
        src = f.read()
    # match the PRINT, not the prose: the surrounding comment quotes the same text
    i_hint = src.find('print( " Probable reasons: SEOB nyquist')
    assert i_hint > 0, 'hint text moved; update this test'

    # The hint must live inside a conditional branch, not fire for every exception.
    # Check the region between the handler that catches the failure and the hint itself.
    i_handler = src.rfind('except Exception as exception_failure:', 0, i_hint)
    assert i_handler > 0, 'handler moved; update this test'
    handler_body = src[i_handler:i_hint]
    assert 'LiveVolumeCollapse' in handler_body, \
        'the SEOB-nyquist hint is no longer guarded by a cause check: an integrator ' \
        'collapse would again be reported as a waveform Nyquist/duration failure'

    # ...and the hint must be nested INSIDE that branch, i.e. indented deeper than the
    # statements the handler runs unconditionally (such as the FAILED ANALYSIS banner).
    def _indent(needle):
        i = src.index(needle)
        return i - (src.rfind('\n', 0, i) + 1)

    assert _indent('print( " Probable reasons: SEOB nyquist') > _indent('print( "  ===> FAILED ANALYSIS'), \
        'the SEOB-nyquist hint sits at the handler top level again: it would be printed ' \
        'for every exception, including an integrator live-volume collapse'


###
### 7. The L0 WARM-START RESCUE: a second integrate() on the same sampler
###
# Observed on all 12 replicates of a rho_net=146.8 rescue campaign: the cold pass
# collapsed, the ILE re-ran a warm pass on the SAME sampler object, and that pass died
# partway through with
#
#     Implicit conversion to a NumPy array is not allowed. Please use `.get()` ...
#
# reported only as "[L0 auto-rescue] skipped (...)".  Cause: integrate() writes
# _rvs['integrand'] AFTER integrate_log has moved everything to the host and (if a fair
# draw ran) truncated it, so on the next pass that key is stale in both length and
# backend -- and the fair-draw gather indexed it with a device index array.  The cold
# pass survived the same code only because a 1-sample live set makes n_extr < len(...)
# False, skipping the gather entirely.
#
# The damage was not the exception: `res, var, neff, dict_return = sampler.integrate(...)`
# never completed, so the ILE reported the COLD diagnostics beside the WARM export.


def test_a_second_integrate_with_a_fairdraw_does_not_die_on_the_stale_integrand_key():
    """The regression.  Two integrate() calls, both fair-drawing, on one sampler."""
    np.random.seed(20260810)
    s = _sampler(20000)
    kw = dict(nmax=200000, neff=8, n=20000, no_protect_names=True, verbose=False,
              igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=50)
    s.integrate(_peaked(20.0), *NAMES, use_lnL=True, **kw)
    assert 'integrand' in s._rvs, 'setup no longer reproduces the stale key'
    # Pre-fix this raised TypeError("Implicit conversion to a NumPy array is not allowed")
    # from the fair-draw gather on a GPU host, and IndexError on a CPU host (the stale
    # key still carries the FIRST pass's fair-draw length).
    res = s.integrate(_peaked(20.0), *NAMES, use_lnL=True, **kw)
    assert res[0] is not None
    n = len(np.asarray(to_host(s._rvs['log_integrand'])))
    for k, v in s._rvs.items():
        assert len(np.asarray(to_host(v))) == n, \
            'key {} kept a stale length from the previous integral'.format(k)


def test_the_fairdraw_gather_leaves_no_device_typed_entry_behind():
    """The gather must land on the host for EVERY key, not only the ones it knows.

    The gather indexes each stored array with the index array `random.choice` returned.
    Those need not share a backend -- on a GPU host the index array is a cupy array while
    a key written outside integrate_log is host-typed, and numpy raises rather than
    converting.  Converting each array to the host first makes the gather backend-blind;
    this pins that nothing device-typed survives it.  Inert on a CPU host, which is why
    the original bug reached production unnoticed by this suite.
    """
    np.random.seed(20260810)
    s = _sampler(20000)
    kw = dict(nmax=200000, neff=8, n=20000, no_protect_names=True, verbose=False,
              igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=50)
    s.integrate(_peaked(20.0), *NAMES, use_lnL=True, **kw)
    s.integrate(_peaked(20.0), *NAMES, use_lnL=True, **kw)
    for k, v in s._rvs.items():
        assert isinstance(v, np.ndarray), \
            'key {} came back on the device: a later host-side consumer will raise'.format(k)


###
### 8. An OVER-CONTRACTED warm start must be reported
###
# The other failure direction, and the one none of the section-3 rules see: seeded from
# too few points the grid contracts onto a sliver, the integrand is flat across it, and
# the pass terminates in ONE cycle looking excellent.  Measured over the 12 rescue
# replicates (see live_volume_collapse_verdict): eleven seeded from 2000 puffed points
# warm-started at V = 7.5e-9..1.5e-8 (351-684 bins) and returned ln(Z/Lmax) = -27.0..-30.6;
# the twelfth seeded from 2 points, warm-started at V = 9.192e-36 (13 bins), and returned
# -80.68 with eff_samp 9789 of 10010 samples -- ~50 nats low, and every existing rule
# passes it: n_live 10010 >> ndim, ESS ~ n, k-hat small, no empty cycles.


def test_the_existing_rules_alone_do_not_see_an_over_contracted_warm_start():
    """Pins WHY a new rule was needed: seed 9134's warm pass looks healthy on the stats."""
    collapsed, _ = live_volume_collapse_verdict(10010, NDIM, ess=9788.7, khat=0.4)
    assert collapsed is False


@pytest.mark.parametrize('n_seed,expect', [
    (2, True),          # seed 9134: V = 9.192e-36, lnZ ~50 nats low
    (NDIM + 1, True),   # a simplex is the smallest cloud that spans NDIM dimensions
    (NDIM + 2, False),
    (2000, False),      # the eleven healthy replicates (the caller's puffed seed)
])
def test_collapse_verdict_flags_a_warm_seed_too_small_to_define_a_volume(n_seed, expect):
    collapsed, reasons = live_volume_collapse_verdict(10010, NDIM, ess=9788.7, khat=0.4,
                                                      n_warm_seed=n_seed)
    assert collapsed is expect, reasons
    if expect:
        assert 'seed point' in '; '.join(reasons)


def test_a_cold_pass_is_never_flagged_for_its_warm_seed():
    """n_warm_seed=None (cold) and 0 (grid of unknown provenance) both skip the rule."""
    for val in (None, 0):
        collapsed, reasons = live_volume_collapse_verdict(10010, NDIM, ess=9788.7,
                                                          khat=0.4, n_warm_seed=val)
        assert collapsed is False, reasons


def test_the_warm_seed_size_reaches_the_verdict_from_bootstrap_from_samples():
    """End to end: bootstrap_from_samples -> _warm['n_seed'] -> dict_return + collapse."""
    np.random.seed(20260810)
    s = _sampler(20000)
    s.setup()
    peak = 0.5 * np.ones(NDIM)
    s.bootstrap_from_samples(peak + 1e-3 * np.array([[-1.0] * NDIM, [1.0] * NDIM]),
                             cover_frac=0.0)
    assert s._warm['n_seed'] == 2
    res = s.integrate_log(_peaked(20.0), *NAMES, nmax=100000, neff=8, n=20000,
                          no_protect_names=True, verbose=False)
    dd = res[3]
    assert dd['n_warm_seed'] == 2
    assert dd['live_volume_collapsed'] is True
    assert 'seed point' in dd['collapse_reason']


def test_a_well_seeded_warm_start_is_not_flagged():
    """The guard must not cry wolf on the eleven replicates that worked."""
    np.random.seed(20260810)
    s = _sampler(20000)
    s.setup()
    peak = 0.5 * np.ones(NDIM)
    s.bootstrap_from_samples(peak + 0.02 * np.random.randn(2000, NDIM), cover_frac=0.0)
    assert s._warm['n_seed'] > NDIM + 1
    res = s.integrate_log(_peaked(20.0), *NAMES, nmax=400000, neff=20, n=20000,
                          no_protect_names=True, verbose=False)
    dd = res[3]
    assert dd.get('live_volume_collapsed') is False, dd.get('collapse_reason')


###
### 9. Wiring: the ILE must not report one pass's diagnostics beside another's samples
###


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_l0_rescue_restores_the_cold_pass_when_the_warm_pass_raises():
    with open(_ILE) as f:
        src = f.read()
    i = src.find('[L0 auto-rescue]')
    assert i > 0, 'rescue block moved; update this test'
    block = src[i:src.find('# Persist adapted state', i)]

    # _rvs is repopulated IN PLACE, so an aliasing capture would restore the warm samples.
    assert '_cold_rvs = dict(sampler._rvs)' in block, \
        '_cold_rvs must be a snapshot: `= sampler._rvs` aliases the dict integrate() mutates'

    # A warm pass that raises mid-assignment leaves res/var/neff/dict_return on the COLD
    # pass while sampler._rvs already holds the WARM samples.  The handler must undo that.
    j = block.find('except Exception as _e_l0')
    assert j > 0, 'handler moved; update this test'
    handler = block[j:]
    assert 'sampler._rvs' in handler and '_cold_state_l0' in handler, \
        'the L0 handler no longer restores the cold pass: the ILE would report cold ' \
        'k-hat/ESS/lnZ beside a warm export'


###
### 8. Empty chunks must not move the answer  (PR #63 review, finding 2)
###
# The empty-chunk guard originally tested `ninj`, the CUMULATIVE live-set size, which is
# only zero for LEADING empty chunks.  Once one sample had survived, a later chunk
# contributing nothing sailed past it and re-thresholded the recycled live set: measured
# before the fix, 20 live points decaying 19, 18, 17, ... with ln V falling -0.05, -0.11,
# -0.16, -0.22, ... over chunks that each returned zero finite samples.  Contraction is an
# inference FROM the chunk, so an empty chunk must license none of it.

class _FiniteFirstChunkOnly(object):
    """Finite on the first chunk, all -inf afterwards. Nothing is learned after chunk 1."""

    def __init__(self, n_finite=20):
        self.calls = 0
        self.n_finite = n_finite

    def __call__(self, *args):
        x = np.array(args).T
        self.calls += 1
        out = np.full(len(x), -np.inf)
        if self.calls == 1:
            k = min(self.n_finite, len(x))
            out[:k] = 100.0 + np.arange(k) * 0.5
        return out


def _run_first_chunk_only(nmax, n_chunk=5000):
    np.random.seed(7)
    s = _sampler(n_chunk)
    fn = _FiniteFirstChunkOnly()
    res = s.integrate_log(fn, *NAMES, nmax=nmax, neff=8, n=n_chunk,
                          no_protect_names=True, verbose=False,
                          igrand_fairdraw_samples=True,
                          igrand_fairdraw_samples_max=50)
    return res, fn.calls


def test_empty_chunks_after_a_successful_one_do_not_change_the_result():
    """The invariant: extra chunks that contribute nothing must be a no-op."""
    short, n_short = _run_first_chunk_only(nmax=10000)     # ~2 chunks
    long_, n_long = _run_first_chunk_only(nmax=60000)      # ~12 chunks
    assert n_long > n_short, 'the long run must actually evaluate more chunks'

    # same evidence, same live set, same weights -- the empty chunks taught us nothing
    assert float(long_[0]) == pytest.approx(float(short[0]), rel=1e-12), \
        'lnZ moved on chunks that contributed no finite sample'
    assert long_[3]['n_live_final'] == short[3]['n_live_final'], \
        'the live set was eroded by chunks that contributed nothing'


def test_empty_chunks_are_counted_and_reported():
    res, n_calls = _run_first_chunk_only(nmax=60000)
    dd = res[3]
    assert dd['n_empty_cycles'] >= n_calls - 2, \
        'chunks contributing no finite sample were not recognised as empty'
    assert dd['live_volume_collapsed'] is True
    assert 'no finite in-volume sample' in dd['collapse_reason']


def test_the_live_set_does_not_shrink_across_empty_chunks():
    """Directly: the surviving count after N empty chunks equals the count after zero."""
    two, _ = _run_first_chunk_only(nmax=10000)
    twelve, _ = _run_first_chunk_only(nmax=60000)
    assert two[3]['n_live_final'] > 1, 'setup should leave a real live set to erode'
    assert twelve[3]['n_live_final'] == two[3]['n_live_final']


###
### 9. The caller must consume the verdict  (PR #63 review, finding 1)
###
# Before this branch a collapsed run CRASHED, which at least kept it out of the posterior.
# Now that it completes, an unconsumed verdict would let a known-degenerate export enter
# downstream assembly as an ordinary row.

@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_ile_consumes_the_collapse_verdict():
    with open(_ILE) as f:
        src = f.read()
    assert "dict_return.get('live_volume_collapsed'" in src, \
        'the ILE never reads the sampler collapse verdict'
    assert 'LIVE VOLUME COLLAPSED' in src, 'the collapse is not surfaced in the ILE log'
    # and it must be actionable, not merely printed
    assert '--reject-collapsed-live-volume' in src and 'reject_collapsed_live_volume' in src, \
        'no way to keep a collapsed event out of the posterior'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_collapse_can_trigger_the_existing_replication_machinery():
    with open(_ILE) as f:
        src = f.read()
    i = src.find('_trigger_reasons = []')
    assert i > 0, 'trigger block moved; update this test'
    block = src[i:i + 1500]
    assert 'live volume collapsed' in block, \
        'a collapsed live volume does not trigger --mc-error-replicas replication'


###
### 10. The legacy string classifier must be corroborated  (PR #63 review, finding 3)
###
# The enclosing handler covers waveform generation, data conditioning and the whole
# likelihood stack; any of those could reduce over an empty array for unrelated reasons.

@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_legacy_empty_reduction_classifier_requires_traceback_corroboration():
    with open(_ILE) as f:
        src = f.read()
    i = src.find("'zero-size array' in str_err")
    assert i > 0, 'legacy classifier moved; update this test'
    clause = src[i:i + 260]
    assert 'mcsamplerAdaptiveVolume' in clause, \
        'the bare empty-reduction string is still enough to be labelled an AV collapse, ' \
        'even for an exception raised in waveform generation'
    # the named exception remains the primary, isinstance-based route
    assert 'isinstance(exception_failure' in src, \
        'the named LiveVolumeCollapse is no longer matched by type'
