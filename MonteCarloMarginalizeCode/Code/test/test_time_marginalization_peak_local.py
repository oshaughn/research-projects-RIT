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
@pytest.mark.parametrize("phase", [0.0, 0.25, 0.5])
def test_exact_on_a_periodic_window(amp, phase, refine):
    """Against a CLOSED-FORM truth, not against the dense path.

    Grid PHASE is swept as well as amplitude: the defect this whole line of work
    exists to remove is that the answer depends on where the sample grid happens to
    fall relative to the peak, so a fixture pinned to one phase can be exactly wrong
    and look exactly right.
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
    up = tmq.bandlimited_upsample(k, F)[0][:last + 1]
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
        Xw, fk = pl.bandlimited_spectrum(k)
        kl = pl.eval_bandlimited_uniform(Xw, fk, np.array([tl[0]]),
                                         np.array([tl[1] - tl[0]]), n_loc,
                                         NPTS * DELTAT)[0]
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
    sig = BandLimited(amp=200.0, peak_sample=NPTS // 3,
                      extra_peaks=[(2 * NPTS // 3, 1.0)])
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
    sig = BandLimited(amp=200.0, peak_sample=NPTS // 3,
                      extra_peaks=[(2 * NPTS // 3, 1.0)])
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
    up = tmq.bandlimited_upsample(k, pl.PEAK_ENUM_FACTOR)[0][:(NPTS - 1) * pl.PEAK_ENUM_FACTOR + 1]
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
            v = cb(one_over_d * up.real, RHO_SQ)
            got = np.where((v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:]))[0]
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
        up = tmq.bandlimited_upsample(k, F)[0][:(NPTS - 1) * F + 1].real
        i = np.where(pl.enumerate_peak_indices(up[None, :])[0])[0] + 1
        i = i[up[i] > up[i].max() - pl.PEAK_KEEP_NATS]
        return np.sort(i * (DELTAT / F))

    coarse, fine = peaks_at(pl.PEAK_ENUM_FACTOR), peaks_at(64)
    for t in fine:
        assert np.min(np.abs(coarse - t)) <= DELTAT, (t, coarse)


# ------------------------------------------- inherited invariants (PR #203)

def test_wrap_exposed_rows_fall_back_to_simpson_exactly():
    """The edge guard is inherited unchanged, and it must still route rows to the
    CALLER'S rule bit-for-bit -- the wrap contaminates the kappa upsample this path
    enumerates on just as much as the one the dense path integrates on."""
    guard = max(1, int(NPTS * tmq.EDGE_GUARD_FRACTION))
    for j in (guard - 1, NPTS - guard):
        sig = BandLimited(amp=40.0, peak_sample=j)
        k = sig.samples()
        assert _peak_local(k) == _simpson_value(k), j
        rep = pl.last_report()
        assert rep['n_wrap_exposed_rows'] == 1 and rep['n_refined_rows'] == 0, (j, rep)


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
        assert (rep['n_refined_rows'] == 1) == (not expect_exposed), (j, rep)


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


def test_the_ceiling_still_fails_closed_through_the_fallback():
    """A row too sharp for the derivation is not silently under-resolved.  Here the
    peak-local rule declines it on cost and hands it to the dense path, which raises
    at its ceiling -- so the fail-closed behaviour survives the delegation."""
    old = tmq.UPSAMPLE_FACTOR_MAX
    try:
        tmq.UPSAMPLE_FACTOR_MAX = 2
        k = BandLimited(amp=2000.0, peak_sample=NPTS // 2).samples()[None, :]
        r = np.full(k.shape, RHO_SQ)
        with pytest.raises(RuntimeError):
            pl.time_marginalize_peak_local(k, r, DELTAT, _lnL)
    finally:
        tmq.UPSAMPLE_FACTOR_MAX = old


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
    assert rep['n_rows'] == 4 and rep['n_peak_local_rows'] == 1, rep
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
    old = pl._CHUNK_BYTES
    try:
        pl._CHUNK_BYTES = 1
        rows = [BandLimited(amp=a, peak_sample=NPTS // 2 + 0.25).samples()
                for a in (40.0, 200.0, 2000.0)]
        k = np.stack(rows)
        got = pl.time_marginalize_peak_local(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    finally:
        pl._CHUNK_BYTES = old
    for i, a in enumerate((40.0, 200.0, 2000.0)):
        assert abs(float(got[i]) - BandLimited(amp=a, peak_sample=NPTS // 2 + 0.25)
                   .truth(4096)) < 1e-3, i


def test_return_peaks_exposes_t_star_and_the_local_width():
    """``t_star`` and the local curvature are first-class OUTPUTS, not internal
    temporaries.  They are distance- and callback-independent (see the invariance
    test above), which is what a time-first reordering of the marginalizations would
    need, so they are exposed deliberately rather than incidentally."""
    sig = BandLimited(amp=2000.0, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()[None, :]
    out, peaks = pl.time_marginalize_peak_local(
        k, np.full(k.shape, RHO_SQ), DELTAT, _lnL, return_peaks=True)
    assert peaks[0] is not None
    t_star, sigma_star = peaks[0]
    j = int(np.argmax(_lnL(k[0].real, RHO_SQ)))
    assert np.min(np.abs(t_star - j * DELTAT)) < DELTAT
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
