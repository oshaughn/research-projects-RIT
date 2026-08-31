"""Tests for the peak-local time-marginalization quadrature.

Companion to ``test_time_marginalization_quadrature.py``, which covers the dense
band-limited path this one is measured against.  The fixtures are deliberately the
same construction -- ``kappa`` built as a sum of complex exponentials strictly below
Nyquist, so the continuous function and its integral are known in CLOSED FORM -- and
several tests here compare against that analytic truth rather than against the dense
path, so a shared bias in the two implementations cannot pass.

Three things this file has to establish that the dense path did not need:

* the LOCAL EVALUATOR reconstructs the same interpolant the dense upsample does, at
  arbitrary points and at every production ``npts`` (the odd-``npts`` spectrum split
  is a live trap: exact at the samples, wrong between them);
* MERGING is load-bearing, not tidiness -- the un-merged variant is reproduced here
  and shown to be wrong;
* the TRUNCATION is bounded rather than hoped for, including when the enumeration is
  sabotaged.
"""

from __future__ import print_function, division

import os
import re
import sys

import numpy as np
import pytest
from scipy import integrate

from RIFT.likelihood import time_marginalization_quadrature as tmq
from RIFT.likelihood import time_marginalization_peak_local as pl
from RIFT.likelihood import factored_likelihood as fl

simpson = getattr(integrate, 'simpson', None) or integrate.simps

SRATE = 4096.0
DELTAT = 1.0 / SRATE
NPTS = 614                       # marginalization_time_grid(0.075, 1/4096)
RHO_SQ = 1000.0

#: The production ``npts`` values, plus their odd neighbours and three tiny sizes.
#: ``marginalization_time_grid(0.075, 1/srate)`` is ODD at three of the five
#: production sample rates -- 153 at 1024, 307 at 2048, 2457 at 16384 -- so odd is
#: the common case and not a corner.
NPTS_CASES = [153, 307, 613, 614, 1228, 2457, 8, 9, 3]


# ----------------------------------------------------------------- helpers

def _lnL(kappa_term, rho_sq):
    """The production default helper, spelled out so the test does not depend on a
    private name.  Affine and INCREASING in the kappa term, which is the property
    the enumeration relies on."""
    return kappa_term - 0.5 * rho_sq


def _lnL_distmarg_like(kappa_term, rho_sq):
    """A distance-marginalization-SHAPED callback: nonlinear, still monotone
    increasing in the kappa term, and ``-inf`` outside a table-like domain.

    The real distmarg callback is a 2-D table interpolation; what matters for THIS
    callback is one property -- monotone but NOT affine in the kappa term -- because
    the production default helper is affine and an implementation that quietly
    assumed linearity would pass every test that used only it.  The ``-inf`` branch is
    present for shape but is far enough out that it does not fire on these fixtures;
    the ``-inf`` domain edge is exercised separately by ``domain_limited`` and by
    ``test_unmeasurable_row_falls_back_and_is_counted``.
    """
    x = kappa_term - 0.5 * rho_sq
    return np.where(x > -1e6, x + 0.05 * np.log1p(np.abs(x)) * np.sign(x), -np.inf)


def _log_trapz(v, dx):
    m = v.max()
    w = np.full(v.size, dx)
    w[0] *= 0.5
    w[-1] *= 0.5
    return m + np.log(np.sum(w * np.exp(v - m)))


def _log_simps(v, dx):
    m = v.max()
    return m + np.log(simpson(np.exp(v - m), dx=dx))


class BandLimited(object):
    """kappa(t) = sum_m c_m exp(2 pi i m t / T), every |f| < Nyquist.

    Identical construction to the dense path's suite, so the two are measured on the
    same object.  ``n_period == NPTS`` gives a window that is exactly periodic;
    ``n_period > NPTS`` gives a window cut from a longer signal, the realistic case.
    """

    def __init__(self, amp, peak_sample, n_period=NPTS, m_hi=None, seed=7,
                 background=0.0, extra_peaks=()):
        self.T = n_period * DELTAT
        self.j0 = (n_period - NPTS) // 2
        scale = n_period / float(NPTS)
        m_hi = int(200 * scale) if m_hi is None else m_hi
        assert m_hi < n_period // 2, "would exceed Nyquist"
        ms = np.arange(1, m_hi + 1)
        env = 1.0 / (1.0 + (ms / (120.0 * scale)) ** 2)
        c = np.exp(-2j * np.pi * ms * (self.j0 + peak_sample) * DELTAT / self.T) * env
        for where, rel in extra_peaks:
            c = c + rel * np.exp(
                -2j * np.pi * ms * (self.j0 + where) * DELTAT / self.T) * env
        if background:
            rng = np.random.default_rng(seed)
            c = c + background * ((rng.normal(size=m_hi) + 1j * rng.normal(size=m_hi))
                                  / (1.0 + (ms / (40.0 * scale)) ** 2))
        self.ms, self.c = ms, amp * c

    def at(self, ts, chunk=40000):
        ts = np.asarray(ts, dtype=float)
        out = np.empty(ts.size, dtype=complex)
        for i in range(0, ts.size, chunk):
            t = ts[i:i + chunk]
            out[i:i + chunk] = np.exp(2j * np.pi * np.outer(t, self.ms) / self.T) @ self.c
        return out

    def samples(self):
        return self.at((self.j0 + np.arange(NPTS)) * DELTAT)

    def truth(self, refine, callback=_lnL):
        """log int exp(lnL) dt over the SAME closed domain [t_0, t_{NPTS-1}].

        ``refine`` is REQUIRED and is passed explicitly at every call site, because a
        default that under-resolves the peak turns the "analytic truth" into another
        under-resolved estimate and the comparison into a coincidence.  The rule used
        below is ``deltaT/refine <= sigma_t/8``; ``sigma_t/deltaT`` is ~0.10 at
        amp 1, 0.047 at 5, 0.017 at 40 and 0.0074 at 200 for this fixture.  It also
        bounds the cost: this is an O(n_modes * n_points) closed-form evaluation.
        """
        n = (NPTS - 1) * refine + 1
        td = self.j0 * DELTAT + np.arange(n) * (DELTAT / refine)
        return _log_trapz(callback(self.at(td).real, RHO_SQ), DELTAT / refine)


def _peak_local(kappa_row, callback=_lnL, **kw):
    k = np.asarray(kappa_row)[None, :]
    r = np.full(k.shape, RHO_SQ)
    return float(pl.time_marginalize_peak_local(k, r, DELTAT, callback, **kw)[0])


def _bandlimited(kappa_row, callback=_lnL):
    k = np.asarray(kappa_row)[None, :]
    r = np.full(k.shape, RHO_SQ)
    return float(tmq.time_marginalize_bandlimited(k, r, DELTAT, callback)[0])


def _simpson_value(kappa_row, callback=_lnL):
    return _log_simps(callback(np.asarray(kappa_row).real, RHO_SQ), DELTAT)


def _random_bandlimited(n, seed=3):
    """A row whose spectrum fills EVERY bin strictly below Nyquist, both signs.

    Filling every bin is the point: a fixture with an empty top positive bin cannot
    see the odd-``npts`` split bug at all.  The exact Nyquist bin (even ``n`` only) is
    left EMPTY on purpose -- that component aliases onto its own conjugate on the
    samples, so no interpolant can recover it and a fixture that fills it would fail
    spuriously.  It is empty in rholm data anyway.
    """
    rng = np.random.default_rng(seed)
    n_pos = (n - 1) // 2
    X = np.zeros(n, dtype=complex)
    X[0] = rng.normal()
    for k in range(1, n_pos + 1):
        X[k] = rng.normal() + 1j * rng.normal()
        X[n - k] = rng.normal() + 1j * rng.normal()
    return np.fft.ifft(X)


# ------------------------------------------------------- the local evaluator

@pytest.mark.parametrize("n", NPTS_CASES)
def test_local_evaluator_matches_the_dense_upsample(n):
    """The peak-local path evaluates the interpolant at arbitrary local times; the
    dense path evaluates it on a uniform refinement by zero-padded FFT.  They must
    be THE SAME FUNCTION, or the two quadratures are not comparable and the A/B
    below means nothing.

    Parametrised over the real production sizes because the frequency split is where
    this breaks: putting the top positive bin at a negative frequency (which a split
    at ``n//2`` does for odd ``n``) leaves the reconstruction exact AT the samples
    and wrong between them, so a round-trip test cannot see it.
    """
    x = _random_bandlimited(n)[None, :]
    factor = 4
    dense = tmq.bandlimited_upsample(x, factor)[0]
    Xw, fk = pl.bandlimited_spectrum(x)
    got = pl.eval_bandlimited_uniform(Xw, fk, np.array([0.0]),
                                      np.array([DELTAT / factor]), n * factor,
                                      n * DELTAT)[0]
    scale = np.max(np.abs(dense))
    assert np.max(np.abs(got - dense)) / scale < 1e-11, n


@pytest.mark.parametrize("n", NPTS_CASES)
def test_local_evaluator_reproduces_the_coarse_samples(n):
    """Necessary but NOT sufficient -- this is exactly the check that stays green
    when the spectrum is split at the wrong index -- so it is here as a floor under
    the test above, not as a substitute for it."""
    x = _random_bandlimited(n, seed=11)[None, :]
    Xw, fk = pl.bandlimited_spectrum(x)
    got = pl.eval_bandlimited_uniform(Xw, fk, np.array([0.0]), np.array([DELTAT]),
                                      n, n * DELTAT)[0]
    assert np.allclose(got, x[0], atol=1e-11 * np.max(np.abs(x)))


def test_local_evaluator_is_accurate_far_along_a_long_grid():
    """The evaluator advances the phase by a recurrence, which drifts as ``m * eps``
    because ``exp(ix)`` is not exactly unit modulus; it is re-anchored periodically.
    Deleting the re-anchor has to fail, so evaluate FAR out -- 8x the re-anchor
    interval -- where an un-anchored recurrence would have accumulated visibly."""
    n = 2457
    x = _random_bandlimited(n, seed=5)[None, :]
    m = 8 * pl._RECURRENCE_REANCHOR
    Xw, fk = pl.bandlimited_spectrum(x)
    h = DELTAT / 4.0
    got = pl.eval_bandlimited_uniform(Xw, fk, np.array([0.0]), np.array([h]), m,
                                      n * DELTAT)[0]
    ref = tmq.bandlimited_upsample(x, 4)[0][:m]
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


def test_local_evaluator_honours_a_per_row_grid():
    """Rows are batched with DIFFERENT interval starts and spacings; a version that
    used row 0's grid for every row would still pass a single-row test."""
    n = 307
    x = np.stack([_random_bandlimited(n, seed=s) for s in (1, 2, 3)])
    Xw, fk = pl.bandlimited_spectrum(x)
    t0 = np.array([0.0, 13.0 * DELTAT, 101.5 * DELTAT])
    h = np.array([DELTAT / 3, DELTAT / 7, DELTAT / 11])
    got = pl.eval_bandlimited_uniform(Xw, fk, t0, h, 20, n * DELTAT)
    for r in range(3):
        ts = t0[r] + np.arange(20) * h[r]
        want = pl.eval_bandlimited_uniform(Xw[r:r + 1], fk, t0[r:r + 1], h[r:r + 1],
                                           20, n * DELTAT)[0]
        assert np.allclose(got[r], want, atol=1e-12 * np.max(np.abs(want)))


# --------------------------------------------- accuracy against analytic truth

@pytest.mark.parametrize("amp,refine", [(1.0, 256), (5.0, 512)])
@pytest.mark.parametrize("phase", [0.0, 0.25, 0.5, 0.3125, 0.28125])
def test_exact_on_a_periodic_window(amp, phase, refine):
    """Against a CLOSED-FORM truth, not against the dense path.

    Grid PHASE is swept as well as amplitude: the defect this whole line of work
    exists to remove is that the answer depends on where the sample grid happens to
    fall relative to the peak, so a fixture pinned to one phase can be exactly wrong
    and look exactly right.

    THE PHASES ABOVE ARE NOT ALL MULTIPLES OF 1/PEAK_ENUM_FACTOR, and that is the whole
    point.  The first version of this suite swept 0.0 / 0.25 / 0.5 only -- all exact
    multiples of 1/8 -- so the crest landed EXACTLY on an enumeration sample in every
    single test, and a bug that only appears between samples was invisible to all of
    them.  The docstring above was already there when that happened.  0.3125 puts the
    crest at half a sample; it is the phase that found the bug.
    """
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2 + phase)
    got, want = _peak_local(sig.samples()), sig.truth(refine)
    assert abs(got - want) < 5e-5, (amp, phase, got, want)


def test_exact_where_simpson_is_hundreds_of_nats_wrong():
    """rho ~ 220, sigma_t/deltaT = 0.0074.  The truth needs a 1024x refinement to be
    a truth at all, which is why this is one case and not a sweep."""
    sig = BandLimited(amp=200.0, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()
    want = sig.truth(1024)
    assert abs(_peak_local(k) - want) < 1e-3, (_peak_local(k), want)
    assert abs(_simpson_value(k) - want) > 100.0


@pytest.mark.parametrize("amp,refine", [(1.0, 256), (5.0, 512), (40.0, 2048)])
def test_accurate_on_a_non_periodic_window(amp, refine):
    """The realistic case: a window cut from a longer band-limited signal, so the
    periodic interpolant genuinely rings at the wrap and the window's own samples do
    NOT determine the true continuous function."""
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2, n_period=2 * NPTS,
                      background=0.12)
    got, want = _peak_local(sig.samples()), sig.truth(refine)
    assert abs(got - want) < 1e-3, (amp, got, want)


@pytest.mark.parametrize("amp", [1.0, 5.0, 40.0, 200.0, 2000.0])
def test_agrees_with_the_dense_bandlimited_path(amp):
    """The A/B this PR exists to make possible: the same rows, the same domain, two
    different placements of the refined grid."""
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()
    assert abs(_peak_local(k) - _bandlimited(k)) < 1e-4, amp


def test_beats_simpson_where_the_peak_is_under_resolved():
    """The whole point, stated as an inequality against the analytic truth rather
    than as a claim about which is prettier."""
    sig = BandLimited(amp=40.0, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()
    ref = sig.truth(2048)
    assert abs(_peak_local(k) - ref) < 1e-4
    assert abs(_simpson_value(k) - ref) > 1.0


def test_the_crest_is_localised_between_enumeration_samples():
    """F1 REGRESSION -- the bug this module shipped with, and the sharpest test here.

    An enumerated extremum is a grid INDEX.  The crest it stands for can be up to
    ``h_enum/2`` away, while the interval half-width is ``W_SIGMA * sigma_t``, so once
    ``W_SIGMA * sigma_t < h_enum/2`` -- any row with a derived factor of 512 or more --
    the peak falls entirely OUTSIDE its own interval and its mass is silently dropped.

    Centring on the sample instead of the crest measured, on this fixture at
    ``sigma_t/deltaT = 0.0024``: **0.00 nats at offset 0, -6.52 at h_enum/4, -164.93 at
    h_enum/2**, always negative.  The sweep below walks the crest across a full
    enumeration cell, so the worst case is inside it by construction rather than by
    luck, and the reference is the dense path (exact here, and independently checked
    against a closed-form truth in the control at the end).
    """
    F = pl.PEAK_ENUM_FACTOR
    ref_sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2)
    k0 = ref_sig.samples()[None, :]
    sigma, _, _ = tmq.peak_width_from_lnL(_lnL(k0.real, RHO_SQ), DELTAT)
    assert pl.W_SIGMA * float(sigma[0]) < 0.5 * DELTAT / F, (
        "fixture is not in the regime this test exists for", float(sigma[0]) / DELTAT)

    for off in np.linspace(0.0, 1.0, 9):            # a full enumeration cell
        phase = off / F
        sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + phase)
        k = sig.samples()
        assert abs(_peak_local(k) - _bandlimited(k)) < 1e-3, (off, phase)

    # control: at the phase that produced -164.93, both paths hit the analytic truth
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.5 / F)
    k, want = sig.samples(), None
    want = sig.truth(2048)
    assert abs(_peak_local(k) - want) < 1e-3
    assert abs(_bandlimited(k) - want) < 1e-3


def test_a_block_with_uniform_arrival_times_is_unbiased():
    """The production statement of the same thing, and the one that matters.

    A real block's arrival times bear no relation to the sample grid, so the
    grid-quantisation error is drawn uniformly across an enumeration cell.  Before the
    localisation fix this block measured **median -1.15 nats, worst -131 nats, 56% of
    rows wrong by more than 0.01 nats, and all of them ACCEPTED** -- a bias, not noise,
    because the sign is always negative: mass is dropped, never added.

    Asserted on the WHOLE distribution rather than on a summary, because a median-only
    check passes while a tail deletes extrinsic samples from the marginalization.
    """
    rng = np.random.default_rng(20260828)
    phases = rng.uniform(0.0, 1.0, 48)
    k = np.stack([BandLimited(amp=2000.0, peak_sample=NPTS // 2 + p).samples()
                  for p in phases])
    r = np.full(k.shape, RHO_SQ)
    got = np.asarray(pl.time_marginalize_peak_local(k, r, DELTAT, _lnL))
    ref = np.asarray(tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL))
    d = got - ref
    assert np.max(np.abs(d)) < 1e-3, (np.median(d), d[np.argmax(np.abs(d))])
    assert abs(np.median(d)) < 1e-6, np.median(d)
    assert pl.last_report()['n_peak_local_rows'] == len(phases)


def test_a_failed_localisation_sends_the_row_to_the_dense_path(monkeypatch):
    """Fail closed.  If the crest cannot be PLACED, the row is not approximated.

    Sabotaged by making the localiser report non-convergence, which is what a
    pathological integrand would do.  The value must still be right -- it comes from
    the dense path -- and the row must be counted, not silently absorbed.
    """
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.3125)
    k = sig.samples()
    real = pl.localise_peaks

    def never_converges(*a, **kw):
        t, q, ok = real(*a, **kw)
        return t, q, np.zeros_like(ok)

    monkeypatch.setattr(pl, 'localise_peaks', never_converges)
    got = _peak_local(k)
    rep = pl.last_report()
    assert rep['n_dense_fallback_localise'] == 1, rep
    assert rep['n_peak_local_rows'] == 0, rep
    assert got == _bandlimited(k)


def test_the_containment_check_catches_a_mis_placed_interval(monkeypatch):
    """The a-posteriori half of the F1 fix, tested by re-introducing F1 exactly.

    Forcing the crest back onto the enumeration sample is precisely the old behaviour.
    The interval then misses the peak, and the run must NOT report the truncated value:
    the local grid fails to attain the localised crest's ``lnL`` and the row is handed
    to the dense path.

    Note the check compares against ``lnL`` at the LOCALISED crest.  Against the
    enumeration SAMPLE it would pass here -- the sample is already ~87 nats below the
    crest in this configuration -- which is why the sample cannot be the reference.
    """
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.5 / pl.PEAK_ENUM_FACTOR)
    k = sig.samples()
    real = pl.localise_peaks

    def snap_back_to_the_grid(Xw, fk, rows, t_grid, h_enum, tol, period, **kw):
        t, q, ok = real(Xw, fk, rows, t_grid, h_enum, tol, period, **kw)
        return t_grid, q, ok            # crest value kept, position quantised: old bug

    monkeypatch.setattr(pl, 'localise_peaks', snap_back_to_the_grid)
    got = _peak_local(k)
    rep = pl.last_report()
    assert rep['n_dense_fallback_containment'] == 1, rep
    assert rep['n_peak_local_rows'] == 0, rep
    assert got == _bandlimited(k)


def test_a_nonlinear_distance_marginalization_style_callback():
    """The default helper is AFFINE in the kappa term, so an implementation that
    accidentally assumed linearity would pass everything above.  This callback is
    monotone but not affine, and has a ``-inf`` domain edge."""
    sig = BandLimited(amp=40.0, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()
    got = _peak_local(k, callback=_lnL_distmarg_like)
    assert abs(got - sig.truth(2048, callback=_lnL_distmarg_like)) < 1e-3
    assert abs(got - _bandlimited(k, callback=_lnL_distmarg_like)) < 1e-3


# ------------------------------------------------- property 1: the merge

def _unmerged_value(kappa_row, callback=_lnL):
    """The NAIVE variant: one interval per enumerated peak, integrated and summed
    WITHOUT merging.  Reproduced here rather than described, because "we merge" is
    the kind of statement that survives the code that implements it being deleted.
    Mirrors ``~/tmarg_harness/peaklocal.py``."""
    k = np.asarray(kappa_row)[None, :]
    F = pl.PEAK_ENUM_FACTOR
    h = DELTAT / F
    last = (NPTS - 1) * F
    up = tmq.reflected_bandlimited_upsample(k, F)[0][:last + 1]
    v = callback(up.real, RHO_SQ)
    idx = np.where((v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]))[0] + 1
    idx = idx[v[idx] > v[idx].max() - pl.PEAK_KEEP_NATS]
    parts = []
    for i in idx:
        if i < 1 or i >= v.size - 1:
            continue
        d2 = (v[i + 1] - 2 * v[i] + v[i - 1]) / h ** 2
        if d2 >= 0:
            continue
        s = 1.0 / np.sqrt(-d2)
        a = max(0.0, i * h - pl.W_SIGMA * s)
        b = min(last * h, i * h + pl.W_SIGMA * s)
        n_loc = max(3, int(np.ceil((b - a) / min(s / tmq.UPSAMPLE_SAFETY, h))) + 1)
        tl = np.linspace(a, b, n_loc)
        Xw, fk = pl.bandlimited_spectrum(
            np.concatenate((k, np.flip(k, axis=-1)), axis=-1))
        kl = pl.eval_bandlimited_uniform(Xw, fk, np.array([tl[0]]),
                                         np.array([tl[1] - tl[0]]), n_loc,
                                         2.0 * NPTS * DELTAT)[0]
        parts.append(_log_trapz(callback(kl.real, RHO_SQ), tl[1] - tl[0]))
    if not parts:
        return np.nan
    m = max(parts)
    return m + np.log(sum(np.exp(p - m) for p in parts))


def test_unmerged_intervals_double_count():
    """Merging is CORRECTNESS, not tidiness.

    Two overlapping windows integrated separately both contain the shared region, so
    the log-sum-exp of the parts counts it twice.  On a broad integrand -- where many
    enumerated peaks sit within a few sigma of each other -- the prototype measured
    **+1.6 nats at rho ~ 6**.  Here the merged value is exact against the analytic
    truth and the un-merged one is not, by a margin no rounding can explain.  Without
    this test, deleting the merge leaves every accuracy test above still green,
    because on a sharply peaked row the intervals do not overlap at all.
    """
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.25)     # rho ~ 6
    k = sig.samples()
    ref = sig.truth(128)
    naive = _unmerged_value(k)
    assert np.isfinite(naive)
    assert naive - ref > 0.5, (naive, ref)              # over-counts, and upward
    assert abs(_peak_local(k) - ref) < 1e-3


def test_two_peaks_merge_continuously_as_they_approach():
    """No regime switch and no threshold.

    Two peaks of equal height are walked together.  Far apart the rule builds TWO
    disjoint intervals; close together they overlap and the merge collapses them to
    ONE.  Both extremes must occur in the sweep, the count must never go UP as the
    peaks approach, and -- the part that matters -- the value must track the analytic
    truth right through the transition, because a regime switch would show up as a
    step there.
    """
    counts, seps = [], (200, 100, 40, 16, 6, 2)
    for sep in seps:
        sig = BandLimited(amp=200.0, peak_sample=NPTS // 2 - sep / 2.0,
                          extra_peaks=[(NPTS // 2 + sep / 2.0, 1.0)])
        got, want = _peak_local(sig.samples()), sig.truth(1024)
        assert abs(got - want) < 1e-2, (sep, got, want)
        counts.append(pl.last_report()['n_intervals_total'])
    assert counts[0] == 2 and counts[-1] == 1, counts
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts


def test_a_broad_integrand_degenerates_into_the_dense_grid():
    """The other end of the same continuum: when the peaks crowd, the union grows to
    the whole window, the local grid stops being cheaper than the dense one, and the
    row is simply handed over.  That is the degeneration completing -- not a special
    case being detected -- and the answer is the dense path's, unchanged."""
    sig = BandLimited(amp=0.5, peak_sample=NPTS // 2 + 0.25)       # rho ~ 11
    k = sig.samples()
    assert _peak_local(k) == _bandlimited(k)
    rep = pl.last_report()
    assert rep['n_peak_local_rows'] == 0 and rep['n_dense_fallback_cost'] == 1, rep

    sharp = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.25)  # rho ~ 700
    _peak_local(sharp.samples())
    assert pl.last_report()['n_peak_local_rows'] == 1


# ------------------------------- property 2: enumeration and the tail bound

def test_two_separated_peaks_are_both_found():
    """The anti-#201 test, stated behaviourally.

    RIFT PR #201 seeded a Newton solve at guessed points and missed genuine maxima,
    returning ``-inf`` for a finite integral.  Here two well-separated peaks of
    comparable height are built; a seed-and-hope implementation converges to one of
    them and reports roughly half the integral, i.e. ``log 2 = 0.69`` nats low.  The
    tolerance is far tighter than that, so this cannot pass by luck.
    """
    sig = BandLimited(amp=200.0, peak_sample=NPTS // 3 + 0.3125,
                      extra_peaks=[(2 * NPTS // 3 + 0.40625, 1.0)])
    k = sig.samples()
    got, want = _peak_local(k), sig.truth(1024)
    assert abs(got - want) < 1e-3, (got, want)
    assert pl.last_report()['n_peaks_total'] >= 2
    assert pl.last_report()['n_intervals_total'] >= 2


def test_a_sabotaged_enumeration_is_caught_by_the_tail_bound(monkeypatch):
    """COMPLETENESS BUYS SPEED; THE BOUND BUYS CORRECTNESS.

    The argument for truncating is that the mass outside the intervals is BOUNDED,
    and the bound is computed from a grid that resolves ``kappa`` -- not from the
    assumption that the enumeration found everything.  So break the enumeration
    deliberately: keep only the single highest maximum in each row.  On the two-peak
    integrand that discards half the mass, and the module must NOT report the
    truncated value.  It detects the shortfall and hands the row to the dense path,
    which returns the right answer.

    If this test fails, the whole rigour claim in the module docstring is void.
    """
    # OFF-GRID on purpose.  Both peaks previously sat at NPTS//3 and 2*NPTS//3, which
    # are exact enumeration samples -- so the outside maximum was sampled right on the
    # discarded crest and the bound worked for a reason that does not generalise.  Move
    # them off the grid and the bound has to work on its own merits.
    sig = BandLimited(amp=200.0, peak_sample=NPTS // 3 + 0.3125,
                      extra_peaks=[(2 * NPTS // 3 + 0.40625, 1.0)])
    k = sig.samples()
    truth = sig.truth(1024)

    real_enumerate = pl.enumerate_peak_indices

    def only_the_best(q, xpy=np):
        mask = real_enumerate(q, xpy=xpy)
        out = np.zeros_like(mask)
        for r in range(q.shape[0]):
            cand = np.where(mask[r])[0]
            if cand.size:
                out[r, cand[np.argmax(q[r, cand + 1])]] = True
        return out

    monkeypatch.setattr(pl, 'enumerate_peak_indices', only_the_best)
    got = _peak_local(k)
    rep = pl.last_report()
    assert rep['n_dense_fallback_tail'] == 1, rep
    assert rep['n_peak_local_rows'] == 0, rep
    assert abs(got - truth) < 1e-3, (got, truth)


def test_the_reported_tail_bound_is_actually_below_the_tolerance():
    """The diagnostic has to be load-bearing, not decorative: a row this rule
    ACCEPTS must carry a bound strictly under the tolerance it claims to enforce."""
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.25)
    _peak_local(sig.samples())
    rep = pl.last_report()
    assert rep['n_peak_local_rows'] == 1
    assert rep['tail_bound_worst'] < pl.TAIL_LOG_TOL, rep


def test_peak_positions_do_not_depend_on_distance_or_callback():
    """The invariant that licenses enumerating on ``kappa`` instead of on ``lnL``.

    Every shipped callback is monotone increasing in ``Re kappa`` (or ``|kappa|``) at
    fixed ``rho_sq``, and ``rho_sq`` is time-independent on this path, so a monotone
    map cannot move a maximum.  Distance enters only as a positive rescaling of the
    exponent's argument.  Swept over three decades in 1/D and across an affine, a
    nonlinear distmarg-shaped, and a log-shaped callback: the enumerated peak set
    must be IDENTICAL, and equal to the peaks of ``Re kappa``.

    This is what keeps the likelihood callback -- a table interpolation in
    production -- off the full time axis.
    """
    sig = BandLimited(amp=40.0, peak_sample=NPTS // 2 + 0.25, background=0.05,
                      n_period=2 * NPTS)
    k = sig.samples()[None, :]
    up = tmq.reflected_bandlimited_upsample(
        k, pl.PEAK_ENUM_FACTOR)[0][:(NPTS - 1) * pl.PEAK_ENUM_FACTOR + 1]
    ref = np.where(pl.enumerate_peak_indices(up.real[None, :])[0])[0]
    assert ref.size > 3, "fixture must have several maxima for this to mean anything"

    # Every callback must be STRICTLY increasing and numerically safe over the whole
    # range of `up.real`.  A softplus was tried first and is wrong: `exp` overflows to
    # `inf` across most of the range, the callback goes FLAT, and the test then fails
    # for a reason that has nothing to do with the invariant it is checking.  `cbrt`
    # is strictly increasing, continuous, genuinely nonlinear, and cannot overflow.
    callbacks = [_lnL, _lnL_distmarg_like, lambda x, r: np.cbrt(x - 0.5 * r)]
    for one_over_d in (0.05, 1.0, 40.0):
        for cb in callbacks:
            # Through the SHIPPED enumerator on both sides.  A hand-inlined copy of the
            # comparison here silently stopped matching when the enumerator started
            # including endpoints, which made this test fail for a reason that had
            # nothing to do with the invariant it is about.
            v = cb(one_over_d * up.real, RHO_SQ)
            got = np.where(pl.enumerate_peak_indices(v[None, :])[0])[0]
            assert np.array_equal(got, ref), (one_over_d, cb)


def test_the_enumeration_factor_finds_the_same_peaks_as_a_much_finer_grid():
    """``PEAK_ENUM_FACTOR`` is justified by the band limit -- ``kappa``'s narrowest
    possible lobe is a half-cycle of width ``deltaT`` -- but that is an argument, and
    arguments are cheap.  Check it: every peak found at factor 64 that carries
    representable mass must also be found at the shipped factor, to within one
    coarse sample."""
    sig = BandLimited(amp=40.0, peak_sample=NPTS // 2 + 0.25, background=0.20,
                      n_period=2 * NPTS)
    k = sig.samples()[None, :]

    def peaks_at(F):
        up = tmq.reflected_bandlimited_upsample(k, F)[0][:(NPTS - 1) * F + 1].real
        i = np.where(pl.enumerate_peak_indices(up[None, :])[0])[0] + 1
        i = i[up[i] > up[i].max() - pl.PEAK_KEEP_NATS]
        return np.sort(i * (DELTAT / F))

    # To within one ENUMERATION sample, not one coarse sample.  The old bound was
    # `DELTAT` = 8 * h_enum, ~290x the interval half-width it was meant to justify, so
    # it would have accepted an enumeration that missed by far more than the interval
    # is wide.
    coarse, fine = peaks_at(pl.PEAK_ENUM_FACTOR), peaks_at(64)
    h_enum = DELTAT / pl.PEAK_ENUM_FACTOR
    for t in fine:
        assert np.min(np.abs(coarse - t)) <= h_enum, (t, coarse)


# ------------------------------------------- inherited invariants (PR #203)

def test_edge_proximity_is_reported_but_selects_no_quadrature():
    """The edge guard is DIAGNOSTIC, and this test used to assert the opposite.

    It once routed a near-edge row to the caller's Simpson rule, because the periodic
    reconstruction rang at the window wrap.  rift_O4d e4ed25c7 removed that wrap by even
    reflection and demoted the guard: crossing an arbitrary threshold must not silently
    change likelihood quality.  This module inherited the old routing and kept it across
    the rebase, so it returned a SIMPSON value where the dense path returns a refined
    one -- 3.79 nats apart on a row with peaks at both ends, with every fallback counter
    reading zero because the row never entered the rule.

    So: still reported, and refined anyway, and agreeing with the dense path.
    """
    guard = max(1, int(NPTS * tmq.EDGE_GUARD_FRACTION))
    for j in (guard - 1, NPTS - guard):
        sig = BandLimited(amp=40.0, peak_sample=j)
        k = sig.samples()
        got = _peak_local(k)
        rep = pl.last_report()
        assert rep['n_wrap_exposed_rows'] == 1, (j, rep)
        assert rep['n_refined_rows'] == 1, (j, rep)
        assert got != _simpson_value(k), (j, "edge proximity still selects Simpson")
        assert abs(got - _bandlimited(k)) < 1e-3, (j, got, _bandlimited(k))


def test_row_classification_matches_the_dense_path_exactly():
    """The invariant the rebase broke, asserted so it cannot break silently again.

    peak-local's contract is that it changes WHERE the refined grid is placed and
    nothing about WHICH rows get one.  Three clauses of the dense path's classification
    had drifted out of this module -- the reflected reconstruction, the demotion of the
    edge guard, and `boundary_unresolved` -- and each drift showed up as peak-local
    silently returning a lower-accuracy value than `time_marginalize_bandlimited` for
    the same row.  Compare the CLASSIFICATION, not just the values: a value check passes
    whenever the two rules happen to agree, which on most rows they are designed to.
    """
    rows = [np.zeros(NPTS, dtype=complex),                                  # flat
            BandLimited(amp=40.0, peak_sample=3).samples(),                 # edge
            BandLimited(amp=40.0, peak_sample=NPTS - 4).samples(),          # far edge
            BandLimited(amp=0.5, peak_sample=NPTS // 2 + 0.25).samples(),   # broad
            BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.25).samples()]
    k = np.stack(rows)
    r = np.full(k.shape, RHO_SQ)
    pl.time_marginalize_peak_local(k, r, DELTAT, _lnL)
    pr = dict(pl.last_report())
    tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL)
    br = dict(tmq.last_report())
    for key in ('n_refined_rows', 'n_flat_rows', 'n_unmeasurable_rows',
                'n_wrap_exposed_rows'):
        assert pr[key] == br[key], (key, pr[key], br[key])


def test_the_edge_guard_covers_the_RIGHT_edge_too():
    """Pin BOTH boundaries at the sample.  An off-by-one in the upper term leaves
    exactly one row's worth of the right guard band open, and every peak-placement
    fixture elsewhere is far enough inside that both spellings agree."""
    guard = max(1, int(NPTS * tmq.EDGE_GUARD_FRACTION))

    def row_peaking_at(j):
        t = np.arange(NPTS, dtype=float)
        return (np.exp(-0.5 * ((t - j) / 0.35) ** 2) * 40.0).astype(complex)

    for j, expect_exposed in ((guard - 1, True), (guard, False),
                              (NPTS - 1 - guard, False), (NPTS - guard, True)):
        _peak_local(row_peaking_at(j))
        rep = pl.last_report()
        assert (rep['n_wrap_exposed_rows'] == 1) == expect_exposed, (j, rep)
        # The guard REPORTS; it no longer selects a rule, so all four are refined.
        assert rep['n_refined_rows'] == 1, (j, rep)


def test_flat_and_signal_free_rows_are_not_refined_and_not_reported_as_exposed():
    k = np.zeros(NPTS, dtype=complex)
    assert _peak_local(k) == _simpson_value(k)
    rep = pl.last_report()
    assert rep['n_flat_rows'] == 1 and rep['n_wrap_exposed_rows'] == 0, rep
    assert rep['n_refined_rows'] == 0 and rep['n_peak_local_rows'] == 0, rep


def test_an_all_minus_inf_row_returns_minus_inf_and_not_nan():
    """A per-row log-sum-exp offset would compute ``-inf - (-inf) = NaN`` and feed a
    NaN into the sampler weights."""
    k = np.zeros((2, NPTS), dtype=complex)
    k[0] = BandLimited(amp=40.0, peak_sample=NPTS // 2).samples()
    r = np.full(k.shape, RHO_SQ)

    def domain_limited(term, rho):
        """``-inf`` outside a table domain, the shape the distance-marginalization
        callback actually has.  Row 1 has zero kappa, so it lands outside everywhere
        and its ``lnL(t)`` is ``-inf`` for the whole window.

        Note this callback is written as a function of its ARGUMENTS only.  A first
        version blanked "row 1" by index, which is wrong: the callback is also invoked
        on peak stencils and local grids whose leading axis is not the row axis, so it
        indexed into the wrong thing and raised."""
        return np.where(term > 100.0, term - 0.5 * rho, -np.inf)

    got = pl.time_marginalize_peak_local(k, r, DELTAT, domain_limited)
    assert np.isfinite(got[0]) and got[1] == -np.inf, got
    assert not np.any(np.isnan(np.asarray(got)))


def test_unmeasurable_row_falls_back_and_is_counted():
    """``lnL(t)`` non-finite around its maximum at every stencil half-width: no width
    can be justified, so none is guessed at."""
    k = BandLimited(amp=40.0, peak_sample=NPTS // 2).samples()[None, :]
    r = np.full(k.shape, RHO_SQ)

    def holed(term, rho):
        v = np.array(_lnL(term, rho), dtype=float, copy=True)
        j = int(np.argmax(v[0]))
        for d in (1, 2, 4, 8):
            for s in (-1, 1):
                idx = j + s * d
                if 0 <= idx < v.shape[-1]:
                    v[0, idx] = -np.inf
        return v

    out = pl.time_marginalize_peak_local(k, r, DELTAT, holed)
    rep = pl.last_report()
    assert rep['n_unmeasurable_rows'] == 1 and rep['n_refined_rows'] == 0, rep
    assert np.isfinite(out[0])


def test_time_dependent_rho_sq_is_refused():
    k = BandLimited(amp=40.0, peak_sample=NPTS // 2).samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    r[0, 3] += 1.0
    with pytest.raises(NotImplementedError):
        pl.time_marginalize_peak_local(k, r, DELTAT, _lnL)


def test_a_nan_self_term_does_not_abort_the_run():
    """A NaN self-term is NORMAL -- the defensive proposal component draws
    physically-extreme points where the likelihood is NaN.  A bare ``==`` in the
    time-independence tripwire makes ``nan != nan`` fire and kills the ILE process,
    blaming a rotating-response path that is not in use."""
    k = np.zeros((2, NPTS), dtype=complex)
    k[0] = BandLimited(amp=40.0, peak_sample=NPTS // 2).samples()
    r = np.full(k.shape, RHO_SQ)
    r[1] = np.nan
    out = pl.time_marginalize_peak_local(k, r, DELTAT, _lnL)
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_simps_is_required_for_a_non_numpy_backend():
    """scipy's Simpson rule RAISES on a device array, and the fallback rows must use
    the rule the caller's own likelihood uses.  A default here is how every
    ``--vectorized --gpu`` run of the sibling option once crashed."""
    class FakeXpy(object):
        def __getattr__(self, name):
            return getattr(np, name)

    k = BandLimited(amp=40.0, peak_sample=NPTS // 2).samples()[None, :]
    with pytest.raises(ValueError):
        pl.time_marginalize_peak_local(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL,
                                       xpy=FakeXpy())


def test_the_ceiling_fails_closed_for_the_SHARPEST_rows_not_just_broad_ones():
    """A row whose derived factor exceeds the ceiling must RAISE, and the route to that
    must not depend on the cost gate declining it.

    This is F3, and it was fail-OPEN.  The cost gate compares the local cost against the
    dense cost, so the sharper the row the more certainly peak-local keeps it -- and a
    row past the ceiling is the sharpest kind there is.  At a derived factor of 8192 the
    dense path raised, as designed, while peak-local returned -24451 nats and reported
    ``tail_bound_worst = -2721``.

    The earlier version of this test only passed because it set
    ``UPSAMPLE_FACTOR_MAX = 2``, which is broad enough that the COST gate declined the
    row first; the ceiling was never what routed it.  Here the ceiling is lowered but
    left well inside the regime where the cost gate would happily keep the row, so the
    ceiling check is the only thing that can produce the raise -- and the control below
    confirms the row is one peak-local would otherwise have taken.
    """
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.3125)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)

    # control: at the shipped ceiling this row is handled by peak-local
    pl.time_marginalize_peak_local(k, r, DELTAT, _lnL)
    assert pl.last_report()['n_peak_local_rows'] == 1

    old_max = tmq.UPSAMPLE_FACTOR_MAX
    try:
        tmq.UPSAMPLE_FACTOR_MAX = 256
        sigma, _, _ = tmq.peak_width_from_lnL(_lnL(k.real, r), DELTAT)
        assert int(tmq.required_upsample_factors(sigma, DELTAT)[0]) > 256
        with pytest.raises(RuntimeError):
            pl.time_marginalize_peak_local(k, r, DELTAT, _lnL)
    finally:
        tmq.UPSAMPLE_FACTOR_MAX = old_max


def test_a_cost_fallback_row_gets_the_DENSE_value_not_an_approximation():
    """A row this rule declines must come back with the reviewed dense
    implementation's number, not Simpson's and not a truncated local estimate."""
    sig = BandLimited(amp=0.5, peak_sample=NPTS // 2 + 0.25)       # rho ~ 11
    k = sig.samples()
    got = _peak_local(k)
    rep = pl.last_report()
    assert rep['n_dense_fallback_rows'] == 1 and rep['n_peak_local_rows'] == 0, rep
    assert got == _bandlimited(k)
    assert got != _simpson_value(k)


def test_a_mixed_block_gives_every_row_its_own_treatment():
    """Rows are independent, and a block that mixes flat, exposed, dense-fallback and
    peak-local rows must give each the value it would have got alone.  A shared
    offset or a shared factor would show up here and nowhere else."""
    rows = [np.zeros(NPTS, dtype=complex),
            BandLimited(amp=40.0, peak_sample=3).samples(),
            BandLimited(amp=0.5, peak_sample=NPTS // 2 + 0.25).samples(),
            BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.25).samples()]
    singles = [_peak_local(r) for r in rows]
    k = np.stack(rows)
    block = pl.time_marginalize_peak_local(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    rep = pl.last_report()
    # 2, not 1: the near-edge row (peak_sample=3) is now refined like any other, since
    # the edge guard became diagnostic.  The flat row and the broad row account for the
    # rest -- the broad one is declined on cost and gets the dense value.
    assert rep['n_rows'] == 4 and rep['n_peak_local_rows'] == 2, rep
    for i, (a, b) in enumerate(zip(singles, np.asarray(block))):
        assert a == b or abs(a - b) < 1e-9, (i, a, b)


def test_phase_marginalization_is_refused_by_the_library():
    """A DELIBERATE SCOPE CUT, refused rather than silently ignored.

    Production marginalizes over distance, not phase.  Under phase marginalization
    the time peak's Laplace width carries an ``(I1/I0)(|kappa|/D)`` factor that does
    not reduce, so the local spacing stops being derivable from ``rho_sq`` and the
    curvature alone -- the derived-not-configured property this whole line of work
    rests on.  Refusing keeps the option from being run, believed, and compared
    against.
    """
    sig = BandLimited(amp=200.0, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    with pytest.raises(NotImplementedError):
        pl.time_marginalize_peak_local(k, r, DELTAT, _lnL, phase_marginalization=True)


def test_the_bandlimited_path_still_supports_phase_marginalization():
    """The cut applies to the NEW rule only.  'bandlimited' is the reviewed reference
    implementation and must not regress -- and it is what a caller who needs phase
    marginalization is told to use."""
    sig = BandLimited(amp=200.0, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    a = float(tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL,
                                               phase_marginalization=True)[0])
    b = float(tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL,
                                               phase_marginalization=False)[0])
    assert np.isfinite(a) and a != b, (a, b)


def test_the_memory_chunking_path_assembles_its_result():
    """The extrinsic axis is chunked so one dense temporary stays inside a working-set
    budget.  Rows are independent, so chunking must not change the ANSWER -- and that,
    not the physics, is what this test is for: it compares the chunked result against
    the unchunked one on identical inputs.

    Bit-identity is deliberately NOT the bar, because it is not available and asserting
    it fails.  MEASURED on these three rows: the chunked and unchunked values differ by
    **0, 0 and 2 ULPs** (2.4e-16 relative).  Chunking changes the leading dimension of
    the FFT and of the reduction inside the local evaluator, and both numpy's FFT and
    its pairwise summation reassociate with batch shape.  A tolerance of a few ULPs is
    still an extremely sharp instrument for what this test is actually guarding --
    a dropped, duplicated or mis-ordered chunk moves a row by nats, not by ULPs.

    Comparing against an analytic truth instead would be both slower (the truth needs a
    4096x refinement for the sharpest row) and weaker: a tolerance loose enough to
    absorb the reference's own error is loose enough to hide an assembly bug.
    """
    amps = (40.0, 200.0, 2000.0)
    k = np.stack([BandLimited(amp=a, peak_sample=NPTS // 2 + 0.25).samples()
                  for a in amps])
    r = np.full(k.shape, RHO_SQ)
    whole = np.asarray(pl.time_marginalize_peak_local(k, r, DELTAT, _lnL))
    assert pl.last_report()['n_peak_local_rows'] == len(amps)

    old = pl._CHUNK_BYTES
    try:
        pl._CHUNK_BYTES = 1                       # forces one row per chunk
        chunked = np.asarray(pl.time_marginalize_peak_local(k, r, DELTAT, _lnL))
    finally:
        pl._CHUNK_BYTES = old
    ulps = np.abs(chunked - whole) / np.spacing(np.abs(whole))
    assert np.all(ulps <= 8), (whole, chunked, ulps)
    assert pl.last_report()['n_peak_local_rows'] == len(amps)


def test_return_peaks_exposes_t_star_and_the_local_width():
    """``t_star`` and the local curvature are first-class OUTPUTS, not internal
    temporaries.  They are distance- and callback-independent (see the invariance
    test above), which is what a time-first reordering of the marginalizations would
    need, so they are exposed deliberately rather than incidentally."""
    # OFF the enumeration grid, and with no background, so the true crest is known in
    # closed form: this fixture's kernel is symmetric about `peak_sample`.
    peak_sample = NPTS // 2 + 0.3125
    sig = BandLimited(amp=2000.0, peak_sample=peak_sample)
    k = sig.samples()[None, :]
    out, peaks = pl.time_marginalize_peak_local(
        k, np.full(k.shape, RHO_SQ), DELTAT, _lnL, return_peaks=True)
    assert peaks[0] is not None
    t_star, sigma_star = peaks[0]

    # Against the TRUE crest, not against the coarse argmax.  Measuring the distance to
    # the nearest coarse sample tests nothing -- it is 0.3125*deltaT here by
    # construction, whether or not any localisation happened.  The bar is a small
    # fraction of the ENUMERATION spacing, which is what "sub-sample" has to mean when
    # this is offered as the t_star a time-first reordering would build on.
    h_enum = DELTAT / pl.PEAK_ENUM_FACTOR
    err = np.min(np.abs(t_star - peak_sample * DELTAT))
    assert err < 0.01 * h_enum, (err / h_enum, "localisation is not sub-sample")
    assert np.all(sigma_star > 0) and np.all(np.isfinite(sigma_star))


def test_the_tuned_constants_are_pinned_to_their_measured_values():
    """These are not free parameters.  Each is justified in the module docstring or
    in DESIGN_time_marginalization_peak_local.md, and the suite otherwise pins them
    only loosely -- so changing one could pass CI while invalidating the argument
    behind it.  Changing a value here is the deliberate act of also updating that
    record."""
    assert pl.PEAK_ENUM_FACTOR == 8
    assert pl.W_SIGMA == 12.0
    assert pl.PEAK_KEEP_NATS == 60.0
    assert pl.TAIL_LOG_TOL == -23.0
    assert pl.MAX_INTERVALS == 32
    assert pl._RECURRENCE_REANCHOR == 64
    assert pl.LOCALISE_SAFETY == 0.25
    assert pl.LOCALISE_MAX_ITER == 16
    assert pl.CONTAINMENT_SLACK_NATS == 0.5

    # The relation that USED to be an unstated precondition.  `erfc(W/sqrt2)` bounds the
    # truncation of a Gaussian about its CREST, so centring on the enumeration sample
    # silently required `W_SIGMA * sigma_t >= h_enum/2` -- which couples W_SIGMA to
    # PEAK_ENUM_FACTOR, is violated by every sharp row, and was nowhere asserted.
    # Localisation discharges it: the interval is centred on the crest and widened by
    # the localisation residual, so what has to hold is only that the residual is small
    # against the half-width.  That is checkable, so check it.
    assert pl.LOCALISE_SAFETY < 0.1 * pl.W_SIGMA, (
        "the localisation residual must be negligible against the interval half-width")
    assert pl.CONTAINMENT_SLACK_NATS < 1.0, (
        "the containment slack must be far below the smallest miss F1 produced (6.5 nats)")
    # inherited, and the peak-local path derives its LOCAL spacing from this one
    assert tmq.UPSAMPLE_SAFETY == 2.0
    assert tmq.EDGE_GUARD_FRACTION == 0.125


def test_peak_local_is_a_recognised_quadrature_name():
    assert tmq.validate_time_quadrature('peak-local') == 'peak-local'
    assert 'peak-local' in tmq.TIME_QUADRATURE_CHOICES
    with pytest.raises(ValueError):
        tmq.validate_time_quadrature('peaklocal')          # sic
    with pytest.raises(ValueError):
        tmq.validate_time_quadrature('peak_local')         # sic


def test_a_secondary_crest_between_samples_is_not_dropped_by_the_keep_filter():
    """G1 REGRESSION -- the same defect as F1, at a different site.

    The keep filter compared `q_up[rows_p, cols_p]`, the SAMPLE value, against the
    highest sample.  A crest a distance d from its sample reads `(d/sigma)^2/2` nats
    low, so a peak between samples was measured against a peak on a sample and dropped.
    MEASURED on this fixture at rho ~ 700: the secondary crest is 1.003 nats below the
    dominant crest but its SAMPLE is 70.99 nats below, past `PEAK_KEEP_NATS = 60`, and
    the answer came back **exactly -log(2) = -0.693147** low: one of two equal peaks
    silently deleted, with no fallback triggered and `tail_bound_worst = -120`.

    THE ASYMMETRY IS THE POINT, and it is the cell the suite was missing.  Every
    multi-peak fixture here ran at `amp=200` (just above the threshold), every sharp
    fixture was single-peak, and the updated F1 tests moved BOTH peaks off-grid by
    similar amounts -- so the two deficits cancelled.  The defect needs one peak near a
    sample and one between, which is the generic production case.
    """
    F = pl.PEAK_ENUM_FACTOR
    for off in (0.0, 0.25, 0.5, 0.75):
        sig = BandLimited(amp=2000.0, peak_sample=NPTS // 3,
                          extra_peaks=[(2 * NPTS // 3 + off / F, 1.0)])
        k = sig.samples()
        assert abs(_peak_local(k) - _bandlimited(k)) < 1e-3, off
        rep = pl.last_report()
        assert rep['n_peaks_total'] == 2, (off, rep)      # neither peak dropped
        assert rep['n_peak_local_rows'] == 1, (off, rep)  # and not rescued by fallback


def test_the_keep_threshold_is_justified_without_the_tail_bound():
    """G2.  `PEAK_KEEP_NATS` and `TAIL_LOG_TOL` are NOT independent, and the docstring
    that said a dropped peak "enters the tail bound" was vacuous: since `q_out_max` is
    at least the dropped peak's own sample, the bound accepts the row unless
    `sigma_t < 2.6e-18 s`, while `UPSAMPLE_FACTOR_MAX` bounds the sharpest legal row ten
    orders of magnitude above that. For every row this module can legally handle, a
    keep-filter drop is automatically accepted.

    So the filter has to stand on its own magnitude argument, and that argument ties it
    to `UPSAMPLE_FACTOR_MAX`. Assert the inequality, not the constants.
    """
    t_window = 0.075
    sigma_min = DELTAT / tmq.UPSAMPLE_FACTOR_MAX
    log_rel_mass = -pl.PEAK_KEEP_NATS + np.log(t_window / sigma_min)
    assert log_rel_mass < np.log(1e-16), (
        "a dropped peak's relative mass is not below double precision", log_rel_mass)
    # and the relation that makes the tail bound unable to backstop it, recorded so a
    # future change to either constant has to confront it
    assert pl.PEAK_KEEP_NATS > -pl.TAIL_LOG_TOL


def test_W_SIGMA_gives_the_tail_bound_its_structural_slack():
    """WHY the sampled tail bound holds, rather than "we could not break it".

    An independent attempt to break `q_out_max` -- 24 accepted rows, honest supremum
    recomputed on a 4096x grid -- failed, worst honest margin -63.42 against
    `TAIL_LOG_TOL = -23`.  But the reason is NOT that the sampling is adequate: it is
    that the outside supremum sits at an interval EDGE, already `W_SIGMA**2/2 = 72` nats
    below the crest, which leaves ~40 nats of structural slack.  Drop `W_SIGMA` below
    ~8-9 and that slack vanishes and the bound is silently invalid, with nothing
    reporting it.

    So the inequality gets the same treatment as `PEAK_KEEP_NATS` vs
    `UPSAMPLE_FACTOR_MAX`: asserted, not admired.  Requiring

        W_SIGMA**2 / 2  >  |TAIL_LOG_TOL| + log(T_out / (sqrt(2 pi) sigma_min))

    ties `W_SIGMA` to `TAIL_LOG_TOL` and to `UPSAMPLE_FACTOR_MAX`, and none of the three
    can now be moved alone.
    """
    t_window = 0.075
    sigma_min = DELTAT / tmq.UPSAMPLE_FACTOR_MAX
    need = -pl.TAIL_LOG_TOL + np.log(t_window / (np.sqrt(2 * np.pi) * sigma_min))
    have = pl.W_SIGMA ** 2 / 2.0
    assert have > need, ("W_SIGMA no longer gives the tail bound its slack", have, need)
    # the margin is large, but assert the inequality rather than the margin: it is the
    # inequality that a future edit has to preserve
    assert have - need > 10.0, (have, need)


def test_the_localised_crest_stays_inside_its_bracket():
    """G4/M4.  `localise_peaks` brackets at +/- h_enum and accepts anything strictly
    inside, so the displacement bound is h_enum, NOT h_enum/2 -- and it is approached:
    0.959*h_enum was observed over 14,182 peaks. The gate interval must therefore widen
    by h_enum, and an escaped iterate must not be accepted."""
    rng = np.random.default_rng(7)
    k = np.stack([BandLimited(amp=2000.0, peak_sample=NPTS // 2 + x).samples()
                  for x in rng.uniform(0, 1, 24)])
    r = np.full(k.shape, RHO_SQ)
    _, peaks = pl.time_marginalize_peak_local(k, r, DELTAT, _lnL, return_peaks=True)
    h_enum = DELTAT / pl.PEAK_ENUM_FACTOR
    worst = 0.0
    for pk in peaks:
        if pk is None:
            continue
        d = np.abs(pk[0] / h_enum - np.round(pk[0] / h_enum))
        worst = max(worst, float(np.max(d)))
    assert worst <= 1.0, worst           # the bracket really does bound it
    # ...and the gate's widening must cover that bound, or its "superset" claim is false
    assert 1.0 * h_enum >= worst * h_enum


def test_a_row_with_a_near_edge_secondary_peak_is_answered_correctly():
    """Near-edge secondary peaks, end to end.

    NAMED FOR WHAT IT ACTUALLY CHECKS.  An earlier version of this was called
    `..._still_gets_its_own_curvature` and claimed to cover the G5 stencil-centre fix.
    It does not, and the mutation sweep said so: re-clipping the stencil centre leaves
    every assertion here green, changing the answer only at the 1e-12 level.  The reason
    is that on these fixtures the edge peak is >`PEAK_KEEP_NATS` below the dominant
    crest and is dropped by the keep filter regardless (`n_peaks_total == 1`), so the
    stencil never runs on it.  Making an edge peak both near-edge AND within 60 nats
    needs an amplitude regime where the row is declined on cost instead -- so the shape
    is not reachable through the public entry point with the fixtures available here.

    That gap is recorded in DESIGN_time_marginalization_peak_local.md rather than
    papered over. The reviewer who found G5 measured it directly (-0.3124 nats), and
    the fix is applied; what is missing is a test of mine that exercises it.
    """
    for edge in (0.05, 0.3, 1.0):
        sig = BandLimited(amp=40.0, peak_sample=NPTS // 2 + 0.3125,
                          extra_peaks=[(edge, 0.9)])
        k = sig.samples()
        assert abs(_peak_local(k) - _bandlimited(k)) < 1e-3, edge


def test_the_reported_tail_bound_matches_an_independent_recomputation():
    """R2-F2.  Asserting only `bound < TAIL_LOG_TOL` pins nothing: the measured margins
    are -440 / -1791 / -4495 against a tolerance of -23, so any error in the bound
    smaller than ~400 nats is invisible, and deleting `log(T_outside)` entirely, or
    marking extra samples covered, both survived. Recompute the bound from the module's
    own reported peaks and require agreement."""
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.3125)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    out, peaks = pl.time_marginalize_peak_local(k, r, DELTAT, _lnL, return_peaks=True)
    rep = pl.last_report()
    assert rep['n_peak_local_rows'] == 1 and peaks[0] is not None

    F = pl.PEAK_ENUM_FACTOR
    h_enum = DELTAT / F
    t_last = (NPTS - 1) * DELTAT
    t_star, sigma = peaks[0]
    half = pl.W_SIGMA * sigma + pl.LOCALISE_SAFETY * sigma
    lo = np.maximum(t_star - half, 0.0)
    hi = np.minimum(t_star + half, t_last)
    starts, stops, _ = (lambda o: (o[3], o[4], o[1]))(
        pl.merge_intervals_by_row(np.zeros(len(lo), dtype=np.int64), lo, hi, t_last))

    up = tmq.reflected_bandlimited_upsample(k, F)[0][:(NPTS - 1) * F + 1].real
    covered = np.zeros(up.size, dtype=bool)
    for a, b in zip(starts, stops):
        i0, i1 = int(np.ceil(a / h_enum)), int(np.floor(b / h_enum))
        if i1 >= i0:
            covered[max(i0, 0):i1 + 1] = True
    T_out = t_last - float(np.sum(stops - starts))
    want = np.log(T_out) + _lnL(np.max(up[~covered]), RHO_SQ) - float(out[0])
    assert abs(rep['tail_bound_worst'] - want) < 1e-6, (rep['tail_bound_worst'], want)


def test_merge_keeps_a_fully_contained_interval_inside_its_enclosure():
    """R2-F3.  The running maximum in `merge_intervals_by_row` is what stops a CONTAINED
    interval truncating the one enclosing it: [0,10] then [1,3] must give [0,10], and
    with a plain `hi_s` it gives [0,3], silently dropping (3,10]. The row-boundary half
    of that trick is covered elsewhere; this is the containment half, and it is a unit
    test so it does not depend on a fixture reaching the code."""
    rows = np.array([0, 0, 0], dtype=np.int64)
    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([10.0, 3.0, 4.0])
    _, _, g_row, g_lo, g_hi = pl.merge_intervals_by_row(rows, lo, hi, 100.0)
    assert g_row.size == 1, (g_row, g_lo, g_hi)
    assert g_lo[0] == 0.0 and g_hi[0] == 10.0, (g_lo, g_hi)

    # two rows, each with a nested pair, to keep the row-reset and the running maximum
    # exercised together rather than one masking the other
    rows = np.array([0, 0, 1, 1], dtype=np.int64)
    lo = np.array([0.0, 1.0, 5.0, 6.0])
    hi = np.array([10.0, 3.0, 20.0, 7.0])
    _, _, g_row, g_lo, g_hi = pl.merge_intervals_by_row(rows, lo, hi, 100.0)
    assert list(g_row) == [0, 1] and list(g_hi) == [10.0, 20.0], (g_row, g_lo, g_hi)


def test_intervals_are_clipped_to_the_integration_domain():
    """R2-F4.  Without the clip an interval can start below 0 or end past t_last, and
    the evaluator is PERIODIC -- it returns the value from the opposite end of the
    window, wrapping mass from outside the integration domain into the answer. More
    reachable since endpoints became enumerable, not less."""
    t_last = (NPTS - 1) * DELTAT
    sig = BandLimited(amp=5.0, peak_sample=NPTS // 2 + 0.3125,
                      extra_peaks=[(0.2, 0.98), (NPTS - 1.2, 0.98)])
    k = sig.samples()
    assert abs(_peak_local(k) - _bandlimited(k)) < 1e-3
    _, peaks = pl.time_marginalize_peak_local(
        k[None, :], np.full((1, NPTS), RHO_SQ), DELTAT, _lnL, return_peaks=True)
    if peaks[0] is not None:
        t_star, sigma = peaks[0]
        half = pl.W_SIGMA * sigma + pl.LOCALISE_SAFETY * sigma
        assert np.all(np.maximum(t_star - half, 0.0) >= 0.0)
        assert np.all(np.minimum(t_star + half, t_last) <= t_last)


def test_a_plateau_yields_exactly_one_enumerated_maximum():
    """R2-F5.  The `>=` / `>` asymmetry is stated as a deliberate property in the
    docstring and had no detector: swapping one way gives a plateau NO peak, the other
    gives it EVERY sample. Both are wrong and both were invisible."""
    q = np.array([[0.0, 1.0, 1.0, 1.0, 0.0]])
    m = pl.enumerate_peak_indices(q)
    assert int(np.sum(m)) == 1, m
    assert int(np.where(m[0])[0][0]) == 3, m          # the LAST index of the plateau

    q = np.array([[0.0, 1.0, 0.0, 2.0, 0.0]])         # two isolated maxima
    assert int(np.sum(pl.enumerate_peak_indices(q))) == 2


def test_the_local_trapezoid_uses_endpoint_half_weights():
    """R2-F6.  Benign at `W_SIGMA = 12`, where the endpoints are `exp(-72)` relative --
    but the "trapezoid, not Simpson" argument had no detector at all, and it stops being
    benign the moment `W_SIGMA` moves. Pinned on a constant integrand, where the
    trapezoid rule is exact and the half-weights are the whole difference."""
    n, h = 11, 0.25
    lnL = np.zeros((1, n))
    got = float(pl._log_trapz_local(lnL, np.array([h]))[0])
    assert abs(np.exp(got) - (n - 1) * h) < 1e-12, (np.exp(got), (n - 1) * h)


def test_the_pre_enumeration_cost_gate_actually_fires():
    """R2-F7.  Gate 1 is the fix this PR credits with removing a 0.43x regression, and
    it was removable with the suite green because both gates shared one counter --
    nothing could tell which had fired. Split, and asserted here on a broad row that
    must be declined BEFORE any enumeration is done for it."""
    sig = BandLimited(amp=0.02, peak_sample=NPTS // 2 + 0.3125)     # broad, factor ~4
    k = sig.samples()
    assert _peak_local(k) == _bandlimited(k)
    rep = pl.last_report()
    assert rep['n_dense_fallback_cost_pregate'] == 1, rep
    assert rep['n_peak_local_rows'] == 0, rep

    # control: a sharp row must NOT be declined by the pre-gate, or the gate is simply
    # rejecting everything and the assertion above means nothing
    sharp = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.3125)
    _peak_local(sharp.samples())
    assert pl.last_report()['n_dense_fallback_cost_pregate'] == 0
    assert pl.last_report()['n_peak_local_rows'] == 1


def test_a_row_with_more_structure_than_MAX_INTERVALS_is_declined():
    """R2-F7.  `MAX_INTERVALS` is a fail-closed guard and was removable with the suite
    green.

    Tested by lowering the constant rather than by building a 33-peak fixture, and the
    reason is worth stating: a comb of nearly-equal peaks does not survive contact with
    the rest of the algorithm -- at any amplitude tried, interference between the bumps
    spread their crests by more than `PEAK_KEEP_NATS`, so all but one were dropped and
    the row arrived at the guard with a single interval. Lowering the constant tests the
    ROUTING, which is what the guard is; the control below shows the same row is handled
    by peak-local when the guard is not in the way, so the assertion is not vacuous.
    """
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 3 + 0.3125,
                      extra_peaks=[(2 * NPTS // 3 + 0.25 / pl.PEAK_ENUM_FACTOR, 1.0)])
    k = sig.samples()

    # control: two intervals, handled
    assert abs(_peak_local(k) - _bandlimited(k)) < 1e-3
    assert pl.last_report()['n_intervals_total'] == 2, pl.last_report()
    assert pl.last_report()['n_peak_local_rows'] == 1

    old_max = pl.MAX_INTERVALS
    try:
        pl.MAX_INTERVALS = 1
        got = _peak_local(k)
        rep = pl.last_report()
    finally:
        pl.MAX_INTERVALS = old_max
    # NOTE, from the mutation sweep: this is killed by the PROVISIONAL structure gate,
    # not the final one.  Disabling the final `too_much` check alone leaves this green,
    # because the provisional gate declines the row first.  The final check is therefore
    # a belt-and-braces guard that no fixture here reaches -- it is kept because the
    # final interval count CAN exceed the provisional one (narrower intervals merge less
    # readily), but that shape is not exercised, and saying so is more useful than a
    # contrived fixture that pretends otherwise.
    assert rep['n_dense_fallback_structure'] == 1, rep
    assert rep['n_peak_local_rows'] == 0, rep
    assert got == _bandlimited(k)


def test_the_curvature_stencil_widens_over_a_hole_on_the_ENUMERATION_grid():
    """R2-F7.  The widening ladder is tested for the coarse-grid width estimator but was
    untested for the peak stencil, where the callback's `-inf` domain edge also lands.
    A hole at the immediate neighbours must be stepped over, not read as "no curvature"
    -- which would drop the peak."""
    sig = BandLimited(amp=200.0, peak_sample=NPTS // 2 + 0.3125)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    ref = float(pl.time_marginalize_peak_local(k, r, DELTAT, _lnL)[0])
    n_ref = pl.last_report()['n_peak_local_rows']

    peak_q = np.max(k.real)

    def holed_near_the_crest(term, rho):
        v = _lnL(term, rho)
        # blank a thin shell just inside the crest: the d=1 stencil straddles it, the
        # wider half-widths step over it
        band = (term < peak_q * 0.99999) & (term > peak_q * 0.9999)
        return np.where(band, -np.inf, v)

    got = float(pl.time_marginalize_peak_local(k, r, DELTAT, holed_near_the_crest)[0])
    assert np.isfinite(got)
    assert pl.last_report()['n_peak_local_rows'] == n_ref, pl.last_report()


def test_the_localiser_reports_non_convergence_when_it_does_not_converge():
    """M3 (my own sweep).  Dropping the convergence test from `localise_peaks` survived,
    because the only test of the flag monkeypatched the localiser and so exercised the
    CONSUMER, never the producer. Starve the iteration count instead: the real localiser
    must then report not-converged, and the row must fall back rather than be accepted
    on an unconverged crest."""
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.4)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    old_iter, old_tol = pl.LOCALISE_MAX_ITER, pl.LOCALISE_SAFETY
    try:
        pl.LOCALISE_MAX_ITER = 1
        pl.LOCALISE_SAFETY = 1e-12          # unreachable in one step
        got = float(pl.time_marginalize_peak_local(k, r, DELTAT, _lnL)[0])
        rep = pl.last_report()
    finally:
        pl.LOCALISE_MAX_ITER, pl.LOCALISE_SAFETY = old_iter, old_tol
    assert rep['n_dense_fallback_localise'] == 1, rep
    assert rep['n_peak_local_rows'] == 0, rep
    assert got == _bandlimited(k[0])


def test_the_report_sub_counts_reconcile():
    """R2-F9.  An operator reading `last_report()` should not find an unexplained
    residual: every declined row must appear in exactly one sub-count."""
    rows = [np.zeros(NPTS, dtype=complex),
            BandLimited(amp=0.02, peak_sample=NPTS // 2 + 0.3125).samples(),
            BandLimited(amp=0.5, peak_sample=NPTS // 2 + 0.3125).samples(),
            BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.3125).samples()]
    k = np.stack(rows)
    pl.time_marginalize_peak_local(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    r = pl.last_report()
    subs = (r['n_dense_fallback_cost_pregate'] + r['n_dense_fallback_cost']
            + r['n_dense_fallback_tail'] + r['n_dense_fallback_structure']
            + r['n_dense_fallback_ceiling'] + r['n_dense_fallback_localise']
            + r['n_dense_fallback_containment'] + r['n_dense_fallback_nopeak'])
    assert subs == r['n_dense_fallback_rows'], (subs, r)
    assert r['n_peak_local_rows'] + r['n_dense_fallback_rows'] == r['n_refined_rows'], r


def _two_equal_kernels(H, off, m0=120.0):
    """Two kernels of IDENTICAL height, one on an enumeration sample and one displaced
    by `off` cells.  The asymmetry is the point: it is the shape that three rounds of
    review kept finding, and that fixtures moving both peaks by similar amounts cannot
    produce.  Built directly in the spectrum so the sharpness is set by H alone."""
    M = (NPTS - 1) // 2
    ms = np.arange(1, M + 1)
    h = DELTAT / pl.PEAK_ENUM_FACTOR
    e = np.exp(-0.5 * (ms / m0) ** 2)
    S = 2 * e.sum()
    c = 2.0 * e * ((H / S) * np.exp(-2j * np.pi * ms * (204 * DELTAT) / (NPTS * DELTAT))
                   + (H / S) * np.exp(-2j * np.pi * ms
                                      * (409 * DELTAT + off * h) / (NPTS * DELTAT)))
    return (np.exp(2j * np.pi * np.outer(np.arange(NPTS), ms) / NPTS) @ c)[None, :]


def _exact_reference(kappa, refine=16384):
    """The integral of the exact band-limited interpolant -- the same object the module
    integrates -- at a refinement far beyond any factor the module will derive.

    REFLECTED, because that is what the module integrates since rift_O4d e4ed25c7: the
    raw periodic interpolant is a different function near the window ends, and using it
    here would hold peak-local to a reference its own dense fallback does not meet."""
    up = tmq.reflected_bandlimited_upsample(kappa, refine)[0, :(NPTS - 1) * refine + 1].real
    m = up.max()
    w = np.ones(up.size)
    w[0] = w[-1] = 0.5
    return m + np.log(np.sum(np.exp(up - m) * w)) + np.log(DELTAT / refine)


@pytest.mark.parametrize("H,want_factor", [(160000.0, 1024), (2.56e6, 4096)])
@pytest.mark.parametrize("off", [0.0, 0.25, 0.5])
def test_the_keep_decision_is_exact_at_the_sharpest_legal_rows(H, want_factor, off):
    """THE THIRD ROUND ON ONE CLASS, pinned at the sharp end of the LEGAL range.

    Comparing SAMPLES dropped a peak once `(delta/sigma)^2/2 > PEAK_KEEP_NATS`.  The
    parabolic vertex that replaced them bought about 8x in `sigma_t/deltaT` and then
    failed the same way: its under-read of the crest at half-cell phase is 1.17 / 4.66 /
    18.66 / **74.63** nats at factors 512 / 1024 / 2048 / 4096, and
    `UPSAMPLE_FACTOR_MAX = 4096` makes the last one LEGAL.  End-to-end that was
    **-0.693147** -- `-log 2`, one of two equal peaks deleted -- with the tail bound and
    the containment check both silent.

    The lesson this test exists to pin: **every approximation substituted for the crest
    fails the same way one octave further out.**  The keep decision is now taken on
    `lnL_star`, which is the crest by construction, so the reach of the fix is not a
    function of sharpness at all -- which is exactly what this parametrisation checks.
    """
    kappa = _two_equal_kernels(H, off)
    rho = np.zeros_like(kappa.real)
    sigma, _, _ = tmq.peak_width_from_lnL(_lnL(kappa.real, rho), DELTAT)
    factor = int(tmq.required_upsample_factors(sigma, DELTAT)[0])
    assert factor == want_factor, (factor, want_factor, float(sigma[0]) / DELTAT)

    got = float(pl.time_marginalize_peak_local(kappa, rho, DELTAT, _lnL)[0])
    rep = pl.last_report()
    assert abs(got - _exact_reference(kappa)) < 1e-3, (H, off, rep)
    # and BOTH crests were kept -- the value alone cannot distinguish "both peaks" from
    # "one peak plus a compensating error"
    assert rep['n_peaks_total'] == 2, (H, off, rep)
    assert rep['n_peak_local_rows'] == 1, (H, off, rep)


def test_an_endpoint_maximum_can_obtain_a_finite_width():
    """Door 2, asserted on the MECHANISM rather than on a value.

    Both crest estimators read `stencil[m-d]` and `stencil[m+d]`; with the centre left
    at `cols_p == 0` the entire left half is out of range at every half-width, `d2` is
    NaN throughout and `sigma = inf`, so the peak is dropped before anything else runs.
    That made revision 2's endpoint enumeration DEAD CODE -- 22 endpoint maxima
    enumerated, zero able to obtain a width -- and it is invisible to any test that only
    checks the returned value, because the dense fallback supplies a correct answer.

    A value test is exactly what I wrote the first time, and it passed while the feature
    did nothing.  This one asks the estimator directly.
    """
    n_enum = 64
    h = DELTAT / pl.PEAK_ENUM_FACTOR
    # a clean quadratic maximum sitting ON index 0, decaying to the right
    idx = np.arange(n_enum, dtype=float)
    q = (-0.5 * (idx / 3.0) ** 2)[None, :]

    mask = pl.enumerate_peak_indices(q)
    assert bool(mask[0, 0]), "an endpoint maximum must be enumerated at all"

    maxd = max(tmq.CURVATURE_STENCIL_HALFWIDTHS)
    cols = np.array([0])
    c_st = np.clip(cols, 1, n_enum - 2)
    take = c_st[:, None] + np.arange(-maxd, maxd + 1)[None, :]
    valid = (take >= 0) & (take < n_enum)
    st = np.where(valid, q[0][np.clip(take, 0, n_enum - 1)], np.nan)
    sigma = pl._peak_curvature_sigma(st, h)
    assert np.isfinite(sigma[0]) and sigma[0] > 0, (
        "an endpoint maximum still cannot obtain a finite width", sigma)


def test_the_integration_domain_is_exactly_the_analysis_window():
    """The clip to `[0, t_last]` is not cosmetic: the evaluator is PERIODIC, so an
    interval running past either end returns values from the opposite end of the window,
    wrapping mass from outside the integration domain into the answer.

    It fires for 138 of 2682 peaks on a realistic block -- I previously recorded it as
    "no fixture reaches it", which was false -- and it is what makes this path's domain
    exactly the dense path's.  Asserted on the intervals themselves, not on a value:
    the clipped region carries `e^-72`, so a value test cannot see it.
    """
    t_last = (NPTS - 1) * DELTAT
    # The clip fires for a SECONDARY peak near an edge in a row whose DOMINANT peak is
    # mid-window.  A row whose own argmax is near an edge is wrap-exposed and never
    # reaches this code at all -- which is what a first version of this fixture produced,
    # and the vacuity guard below caught it.  Moderate sharpness, so W_SIGMA*sigma is a
    # fraction of a sample rather than a thousandth of one.
    # Measured, not guessed: these three reach the clip (`overhang = 1` each), while a
    # secondary peak at `rel = 0.9` is dropped by the keep filter and a broad row is
    # declined on cost, so neither reaches this code at all.
    rows = [BandLimited(amp=5.0, peak_sample=NPTS // 2 + 0.3125,
                        extra_peaks=[(e, rel)]).samples()
            for e, rel in ((0.2, 0.95), (0.5, 0.95), (0.5, 0.99))]
    k = np.stack(rows)
    _, peaks = pl.time_marginalize_peak_local(
        k, np.full(k.shape, RHO_SQ), DELTAT, _lnL, return_peaks=True)
    seen = 0
    for pk in peaks:
        if pk is None:
            continue
        t_star, sigma = pk
        half = pl.W_SIGMA * sigma + pl.LOCALISE_SAFETY * sigma
        lo = np.maximum(t_star - half, 0.0)
        hi = np.minimum(t_star + half, t_last)
        assert np.all(lo >= 0.0) and np.all(hi <= t_last)
        seen += int(np.sum((t_star - half < 0.0) | (t_star + half > t_last)))
    assert seen > 0, "fixture no longer reaches the clip; the assertion is vacuous"


# --------------------------------------------------------------- the wiring

N_BUFFER = 4096


def _fake_likelihood_inputs(kappa_buffer):
    """Minimal inputs that drive the SHIPPED NoLoop function on the numpy backend.

    One detector, one (l,m) pair and zero cross terms, so the self-term is constant
    and kappa reduces to the supplied rholm buffer times a fixed response factor.
    The point is to exercise the argument plumbing; the physics is covered above
    against analytic truth.
    """
    import lal
    import RIFT.lalsimutils as lsu

    rholm = np.asarray(kappa_buffer, dtype=complex)[None, :]
    P = lsu.ChooseWaveformParams()
    P.deltaT = DELTAT
    P.tref = 1000000000.0
    for name, val in [('phi', 0.0), ('theta', 0.0), ('phiref', 0.0),
                      ('incl', 0.0), ('psi', 0.0)]:
        setattr(P, name, np.zeros(1) + val)
    P.dist = np.full(1, fl.distMpcRef * 1e6 * lal.PC_SI)
    return (P, {'H1': rholm}, {'H1': np.array([[2, 2]])},
            {'H1': np.zeros((1, 1), dtype=complex)}, {'H1': P.tref - 0.5})


def _buffer_signal(amp, roll=0):
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2, n_period=N_BUFFER, m_hi=1400,
                      background=0.12)
    return np.roll(sig.at(np.arange(N_BUFFER) * DELTAT), int(roll))


def _shipped(tvals, args, **kw):
    P, rholms, lookupNK, ct, epochs = args
    return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, lookupNK, rholms, ct, ct, epochs, Lmax=2, xpy=np, **kw)


def _tuned_inputs(tvals, sigma_target_over_dt=0.05):
    """Build inputs whose lnL(t) is genuinely under-resolved, by MEASURING what the
    shipped function produces rather than assuming it: the response factor and the
    gather offset are the code's business, not the test's."""
    amp, roll = 1.0, 0
    for _ in range(8):
        args = _fake_likelihood_inputs(_buffer_signal(amp, roll))
        lnL_t = np.asarray(_shipped(tvals, args, return_lnLt=True))
        sigma, jmax, _ = tmq.peak_width_from_lnL(lnL_t, DELTAT)
        roll += int(NPTS // 2 - int(jmax[0]))
        if np.isfinite(sigma[0]):
            amp *= (float(sigma[0]) / (sigma_target_over_dt * DELTAT)) ** 2
    args = _fake_likelihood_inputs(_buffer_signal(amp, roll))
    lnL_t = np.asarray(_shipped(tvals, args, return_lnLt=True))
    sigma, jmax, _ = tmq.peak_width_from_lnL(lnL_t, DELTAT)
    return args, float(sigma[0]) / DELTAT, int(jmax[0])


def test_the_option_reaches_the_shipped_likelihood_and_changes_the_answer():
    """THE WIRING, NOT THE HELPER.

    A flag computed correctly and then never delivered to the likelihood is a
    documented failure mode in this repo -- a whole comparison campaign has been run
    against an inert option here before.  So set the module default exactly the way
    the driver sets it, call the SHIPPED function, and require the number to move.
    """
    pytest.importorskip('RIFT.lalsimutils')
    tvals = fl.marginalization_time_grid(0.075, DELTAT)
    assert len(tvals) == NPTS

    args, sigma_over_dt, jmax = _tuned_inputs(tvals)
    assert sigma_over_dt < 0.15, sigma_over_dt              # under-resolved
    guard = int(NPTS * tmq.EDGE_GUARD_FRACTION)
    assert guard < jmax < NPTS - 1 - guard, jmax

    assert fl.TIME_QUADRATURE_DEFAULT == 'simpson', "default must not have moved"
    base = float(np.asarray(_shipped(tvals, args))[0])
    old = fl.TIME_QUADRATURE_DEFAULT
    try:
        fl.TIME_QUADRATURE_DEFAULT = 'peak-local'          # exactly what the driver does
        new = float(np.asarray(_shipped(tvals, args))[0])
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old
    assert abs(new - base) > 1e-3, (base, new)
    dense = float(np.asarray(_shipped(tvals, args, time_quadrature='bandlimited'))[0])
    assert abs(new - dense) < 1e-3, (new, dense)

    # AND THAT IT WAS THIS RULE THAT PRODUCED IT.  The assertion above cannot tell:
    # peak-local is DESIGNED to agree with bandlimited, so rewiring the branch in the
    # shipped likelihood to call `time_marginalize_bandlimited` instead leaves every
    # number in this file unchanged and the whole suite green -- while the entire cost
    # benefit, the only reason this PR exists, silently disappears.  That is exactly the
    # "a comparison campaign has been run against an inert option here before" hazard.
    # `_LAST_REPORT` is a module global that ONLY the peak-local module writes, so it
    # distinguishes the two rules where the returned value provably cannot.
    pl._LAST_REPORT.clear()
    fl.TIME_QUADRATURE_DEFAULT = 'peak-local'
    try:
        _shipped(tvals, args)
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old
    rep_ = pl.last_report()
    assert rep_.get('n_peak_local_rows', 0) >= 1, (
        "the shipped branch did not reach time_marginalize_peak_local", rep_)

    kw = float(np.asarray(_shipped(tvals, args, time_quadrature='peak-local'))[0])
    assert kw == new
    fl.TIME_QUADRATURE_DEFAULT = 'peak-local'
    try:
        assert float(np.asarray(_shipped(tvals, args, time_quadrature='simpson'))[0]) == base
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old


def test_unsupported_combinations_refuse_rather_than_silently_using_simpson():
    pytest.importorskip('RIFT.lalsimutils')
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2)
    tvals = fl.marginalization_time_grid(0.075, DELTAT)
    P, rholms, lookupNK, ct, epochs = _fake_likelihood_inputs([sig.samples()])
    common = dict(Lmax=2, xpy=np, time_quadrature='peak-local')
    for extra in ({'n_cal': 2}, {'return_lnLt': True}, {'return_cal_components': True},
                  {'phase_marginalization': True}):
        kw = dict(common)
        kw.update(extra)
        with pytest.raises(NotImplementedError):
            fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
                tvals, P, lookupNK, rholms, ct, ct, epochs, **kw)


@pytest.mark.parametrize("module_name,func_name", [
    ('RIFT.likelihood.factored_likelihood_with_rotation',
     'DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation'),
    ('RIFT.likelihood.factored_likelihood_freqresponse',
     'DiscreteFactoredLogLikelihoodFreqResponseNoLoop'),
])
def test_excluded_paths_refuse_the_global_default(module_name, func_name):
    """These likelihoods have a time-DEPENDENT rho_sq, so neither refined quadrature
    applies.  Setting the option globally must make them RAISE, not quietly run
    Simpson -- otherwise the exclusion is invisible at the point of use."""
    mod = pytest.importorskip(module_name)
    func = getattr(mod, func_name)
    old = fl.TIME_QUADRATURE_DEFAULT
    try:
        fl.TIME_QUADRATURE_DEFAULT = 'peak-local'
        with pytest.raises(NotImplementedError):
            func(None, None, None, None, None, None, None, None)
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old


def test_return_lnLt_still_works_when_the_module_default_is_peak_local():
    """The group's standard extrinsic stage (``--add-extrinsic
    --add-extrinsic-time-resampling``) calls this function with ``return_lnLt=True``
    and no explicit quadrature.  Raising on the INHERITED default there once broke
    that stage after the whole integration had run."""
    pytest.importorskip('RIFT.lalsimutils')
    tvals = fl.marginalization_time_grid(0.075, DELTAT)
    args = _fake_likelihood_inputs(_buffer_signal(1.0))
    old = fl.TIME_QUADRATURE_DEFAULT
    try:
        fl.TIME_QUADRATURE_DEFAULT = 'peak-local'
        got = np.asarray(_shipped(tvals, args, return_lnLt=True))
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old
    assert got.shape == (1, NPTS)


# ------------------------------------------------------------- the driver CLI

def _run_driver(extra_args):
    """Invoke the ILE driver in a SUBPROCESS and return (returncode, output).

    Deliberately a subprocess: the option's whole job is to travel from a command
    line into the likelihood, and the guard that stops it being silently inert lives
    in the driver's startup, not in the library -- so a test that imports the library
    cannot see it.  The driver exits long before any data is needed.
    """
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    driver = os.path.join(root, 'bin', 'integrate_likelihood_extrinsic_batchmode')
    env = dict(os.environ)
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    env['OMP_NUM_THREADS'] = '1'
    proc = subprocess.run([sys.executable, driver] + extra_args, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=900)
    return proc.returncode, proc.stdout.decode('utf-8', 'replace')


_HONOURED = ['--time-marginalization', '--vectorized', '--gpu', '--force-xpy']


def _quadrature_banner(out):
    """The quadrature banner line, matched SPECIFICALLY.

    The pre-existing ``--interpolate-time`` banner carries the identical phrase
    "honoured by this configuration", so a bare substring test matches whichever line
    happens to say what you were hoping for -- a mutation that made the quadrature
    banner claim ``True`` unconditionally survived exactly that way once.  Anchor the
    identifying prefix and the value together, and require exactly one match.
    """
    m = re.findall(r'^\s*Time-marginalization quadrature: (\S+) '
                   r'\(from --time-marginalization-quadrature (.+?)\); '
                   r'honoured by this configuration: (True|False)\s*$',
                   out, re.MULTILINE)
    assert len(m) == 1, ("expected exactly one quadrature banner line, got %d:\n%s"
                         % (len(m), out[-3000:]))
    return m[0][0], m[0][2]


def test_driver_announces_peak_local_and_actually_puts_it_in_force():
    """The banner prints the value READ BACK OUT of the module, so this assertion is
    load-bearing: deleting the assignment leaves the flag inert AND changes the
    printed line."""
    rc, out = _run_driver(['--time-marginalization-quadrature', 'peak-local'] + _HONOURED)
    assert _quadrature_banner(out) == ('peak-local', 'True'), out[-2000:]
    rc, out = _run_driver(_HONOURED)
    assert _quadrature_banner(out) == ('simpson', 'True'), out[-2000:]


def test_driver_rejects_a_misspelled_peak_local():
    """A misspelled stencil name was once absorbed as "not truthy" and silently ran a
    different likelihood here.  A typo in this option has to be loud."""
    rc, out = _run_driver(['--time-marginalization-quadrature', 'peaklocal'])
    assert rc != 0
    assert 'peaklocal' in out and 'peak-local' in out


def test_driver_refuses_configurations_that_cannot_honour_peak_local():
    """Refuse, do not ignore.  Each of these would otherwise run the historical
    Simpson quadrature while the banner said otherwise."""
    for missing in ([], ['--time-marginalization'],
                    ['--time-marginalization', '--vectorized'],
                    _HONOURED + ['--rotation-slow'],
                    _HONOURED + ['--freqresponse']):
        rc, out = _run_driver(['--time-marginalization-quadrature', 'peak-local'] + missing)
        assert rc != 0, (missing, out[-2000:])
        assert 'cannot honour it' in out, (missing, out[-2000:])


def _phase_marg_lookup(tmp_path, value):
    """A minimal distance-marginalization lookup table carrying only the key the
    startup guard reads.  Phase marginalization is NOT a CLI boolean -- it is a
    property of that table -- so a guard that checked an ``opts`` attribute would
    silently never fire."""
    path = str(tmp_path / ('lookup_%s.npz' % value))
    np.savez(path, phase_marginalization=np.array(bool(value)))
    return path


def test_driver_refuses_peak_local_under_phase_marginalization_AT_STARTUP(tmp_path):
    """Refused before the run, not deep inside it.

    The library refuses this too, but by then the integration is under way -- and
    raising late is its own bug on this option: an over-broad ``return_lnLt`` guard
    once let the standard extrinsic stage run the ENTIRE integration and then die at
    the export step.  So the guard has to be at startup, and this test invokes the
    driver to prove it is.

    The three controls are the point: the same table with ``bandlimited`` must be
    HONOURED (that path supports phase marginalization and must not regress), and
    ``peak-local`` without phase marginalization must be honoured too.  A guard that
    simply refused whenever a lookup table was present would pass a bare refusal test
    and fail all three.
    """
    with_phase = _phase_marg_lookup(tmp_path, True)
    without = _phase_marg_lookup(tmp_path, False)
    dm = ['--distance-marginalization', '--distance-marginalization-lookup-table']

    rc, out = _run_driver(['--time-marginalization-quadrature', 'peak-local']
                          + _HONOURED + dm + [with_phase])
    assert rc != 0, out[-2000:]
    assert 'cannot honour it' in out and 'phase marginalization' in out, out[-2000:]

    rc, out = _run_driver(['--time-marginalization-quadrature', 'bandlimited']
                          + _HONOURED + dm + [with_phase])
    assert _quadrature_banner(out) == ('bandlimited', 'True'), out[-2000:]

    rc, out = _run_driver(['--time-marginalization-quadrature', 'peak-local']
                          + _HONOURED + dm + [without])
    assert _quadrature_banner(out) == ('peak-local', 'True'), out[-2000:]


# --------------------------------------------------------------- GPU parity

def _cupy_or_skip():
    if os.environ.get('RIFT_CI_REQUIRE_GPU', '0') != '1':
        cupy = pytest.importorskip('cupy')
    else:
        import cupy
    try:
        cupy.zeros(1) + 1
    except Exception as e:                                  # pragma: no cover
        if os.environ.get('RIFT_CI_REQUIRE_GPU', '0') == '1':
            raise
        pytest.skip("cupy present but no usable device: %s" % e)
    return cupy


def test_peak_local_runs_on_the_gpu_backend_and_matches_numpy():
    """xpy-generic code that has never run on a device is broken until proven
    otherwise.  This is the whole path -- FFT upsample, enumeration, the ragged
    host-side merge, the batched local evaluation and the tail bound -- on cupy,
    against the numpy answer on identical inputs."""
    cupy = _cupy_or_skip()
    from RIFT.likelihood import optimized_gpu_tools

    rows = [BandLimited(amp=a, peak_sample=NPTS // 2 + 0.25).samples()
            for a in (40.0, 200.0, 2000.0)]
    rows.append(np.zeros(NPTS, dtype=complex))
    k = np.stack(rows)
    r = np.full(k.shape, RHO_SQ)

    cpu = np.asarray(pl.time_marginalize_peak_local(k, r, DELTAT, _lnL))
    gpu = cupy.asnumpy(pl.time_marginalize_peak_local(
        cupy.asarray(k), cupy.asarray(r), DELTAT, _lnL,
        simps=optimized_gpu_tools.simps, xpy=cupy))
    fin = np.isfinite(cpu)
    assert np.max(np.abs(gpu[fin] - cpu[fin])) < 1e-6, (cpu, gpu)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
