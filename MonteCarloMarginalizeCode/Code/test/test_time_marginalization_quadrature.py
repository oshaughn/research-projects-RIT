#!/usr/bin/env python
"""Gate for the band-limited time-marginalization quadrature.

WHAT IS BEING TESTED, AND AGAINST WHAT
--------------------------------------
The claim is that the samples the likelihood already computes determine the
continuous time integrand exactly, because kappa(t) is band-limited below
Nyquist and rho_sq is time-independent.  So the reference here is NOT another
numerical estimate of the same thing, and it is not a stored number: it is an
ANALYTIC continuous function.  kappa(t) is built as a sum of complex exponentials
with every frequency below Nyquist, which is band-limited by construction, and
the truth is that same closed form evaluated directly (a dense complex-exponential
sum -- not an FFT, so it shares no machinery with the code under test) and
integrated at a density where the quadrature error is analytically negligible.

Two regimes are covered on purpose:
  * exactly periodic on the window  -> the interpolation is exact, so the only
    error left is the quadrature's, and it should vanish to machine precision;
  * a segment cut from a LONGER band-limited function -> not periodic on the
    window, so the periodic interpolant rings at the wrap.  This is the realistic
    case and it is where the edge guard has to earn its place.

The wiring test drives the SHIPPED likelihood function rather than the helper: an
accuracy option that computes the right number but never reaches the likelihood
is the failure mode this repo has been bitten by before.
"""
from __future__ import print_function, division

import os
import sys

import numpy as np
import pytest
from scipy import integrate

from RIFT.likelihood import time_marginalization_quadrature as tmq
from RIFT.likelihood import factored_likelihood as fl

simpson = getattr(integrate, 'simpson', None) or integrate.simps

SRATE = 4096.0
DELTAT = 1.0 / SRATE
NPTS = 614                       # marginalization_time_grid(0.075, 1/4096)
RHO_SQ = 1000.0


# ----------------------------------------------------------------- helpers

def _lnL(kappa_term, rho_sq):
    """The production default helper, spelled out so the test does not depend on
    a private name."""
    return kappa_term - 0.5 * rho_sq


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

    ``n_period`` sets the period in samples.  n_period == NPTS gives a window that
    is exactly periodic; n_period > NPTS gives a window cut from a longer signal,
    which is the realistic, non-periodic case.
    """

    def __init__(self, amp, peak_sample, n_period=NPTS, m_hi=None, seed=7,
                 background=0.0):
        self.T = n_period * DELTAT
        self.j0 = (n_period - NPTS) // 2
        scale = n_period / float(NPTS)
        m_hi = int(200 * scale) if m_hi is None else m_hi
        assert m_hi < n_period // 2, "would exceed Nyquist"
        ms = np.arange(1, m_hi + 1)
        t_peak = (self.j0 + peak_sample) * DELTAT
        c = np.exp(-2j * np.pi * ms * t_peak / self.T) / (1.0 + (ms / (120.0 * scale)) ** 2)
        if background:
            rng = np.random.default_rng(seed)
            c = c + background * ((rng.normal(size=m_hi) + 1j * rng.normal(size=m_hi))
                                  / (1.0 + (ms / (40.0 * scale)) ** 2))
        self.ms, self.c = ms, amp * c

    def at(self, ts, chunk=40000):
        out = np.empty(np.size(ts), dtype=complex)
        ts = np.asarray(ts)
        for i in range(0, ts.size, chunk):
            t = ts[i:i + chunk]
            out[i:i + chunk] = np.exp(2j * np.pi * np.outer(t, self.ms) / self.T) @ self.c
        return out

    def samples(self):
        return self.at((self.j0 + np.arange(NPTS)) * DELTAT)

    def truth(self, refine=128):
        """log int exp(lnL) dt over the SAME closed domain [t_0, t_{NPTS-1}]."""
        n = (NPTS - 1) * refine + 1
        td = self.j0 * DELTAT + np.arange(n) * (DELTAT / refine)
        return _log_trapz(_lnL(self.at(td).real, RHO_SQ), DELTAT / refine)


def _bandlimited(kappa_row):
    k = np.asarray(kappa_row)[None, :]
    r = np.full(k.shape, RHO_SQ)
    return float(tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL)[0])


def _simpson_value(kappa_row):
    return _log_simps(_lnL(np.asarray(kappa_row).real, RHO_SQ), DELTAT)


# ------------------------------------------------- the band-limited identity

def test_upsample_is_exact_on_a_band_limited_sequence():
    """The upsample must REPRODUCE the analytic function between the samples, not
    merely pass through them.  Interpolating exactly at the input samples is the
    weak check every interpolant passes; the strong one is the values in between,
    which is the whole claim."""
    sig = BandLimited(amp=1.0, peak_sample=NPTS // 2)
    factor = 8
    up = tmq.bandlimited_upsample(sig.samples()[None, :], factor)[0]
    t_dense = np.arange(NPTS * factor) * (DELTAT / factor)
    exact = sig.at(t_dense)
    assert np.allclose(up, exact, atol=1e-10, rtol=0), np.abs(up - exact).max()
    # and the coarse samples land on dense indices j*factor (power-of-two design)
    assert np.allclose(up[::factor], sig.samples(), atol=1e-12, rtol=0)


def test_peak_width_estimator_is_exact_for_a_gaussian_at_any_grid_phase():
    """The width estimator is what makes the refinement DERIVED rather than
    guessed, and its whole job is to stay honest when the peak is under-resolved
    and sitting at an arbitrary phase relative to the grid.  A Gaussian lnL has a
    known width; recover it from grids that resolve it badly."""
    for sigma_over_dt in (4.0, 1.0, 0.3, 0.05):
        sigma = sigma_over_dt * DELTAT
        for phase in (0.0, 0.25, 0.5, 0.75):
            t = (np.arange(NPTS) - NPTS // 2 + phase) * DELTAT
            lnL = -0.5 * (t / sigma) ** 2
            got, _, meas = tmq.peak_width_from_lnL(lnL[None, :], DELTAT)
            assert bool(meas[0])
            assert np.isclose(float(got[0]), sigma, rtol=1e-9), (sigma_over_dt, phase, got)


def test_flat_integrand_derives_no_refinement():
    """A well-resolved integrand must cost nothing: the derivation has to return
    factor 1 rather than paying for resolution it does not need."""
    sig = BandLimited(amp=0.002, peak_sample=NPTS // 2)
    lnL = _lnL(sig.samples().real, RHO_SQ)[None, :]
    sigma, _, meas = tmq.peak_width_from_lnL(lnL, DELTAT)
    assert bool(meas[0]) and float(sigma[0]) > DELTAT, sigma
    assert int(tmq.required_upsample_factors(sigma, DELTAT)[0]) == 1


# --------------------------------------------- accuracy against analytic truth

@pytest.mark.parametrize("amp,phase", [(a, p) for a in (0.02, 0.17, 1.0, 5.0)
                                       for p in (0.0, 0.25, 0.5)])
def test_exact_on_a_periodic_window(amp, phase):
    """Exactly-periodic window: interpolation is exact, so the band-limited value
    must match the analytic truth to well below any level Simpson achieves."""
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2 + phase)
    ref = sig.truth()
    k = sig.samples()
    assert abs(_bandlimited(k) - ref) < 1e-6


@pytest.mark.parametrize("amp,phase", [(a, p) for a in (0.02, 0.17, 1.0, 5.0)
                                       for p in (0.0, 0.25, 0.5)])
def test_accurate_on_a_non_periodic_window(amp, phase):
    """The realistic case: the window is a segment of a longer band-limited
    signal, so the periodic interpolant rings at the wrap.  With the peak
    centred, the residual must still be far below Simpson's error."""
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2 + phase,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    ref = sig.truth()
    k = sig.samples()
    assert abs(_bandlimited(k) - ref) < 1e-3


def test_beats_simpson_where_the_peak_is_under_resolved():
    """The defect itself.  Sweeping the peak across one sample must move the
    Simpson answer by of order a nat while leaving the band-limited answer put --
    that grid-phase sensitivity IS the bug, and insensitivity to it is the fix."""
    sig0 = BandLimited(amp=0.02, peak_sample=NPTS // 2,
                       n_period=8 * NPTS, m_hi=1400, background=0.12)
    sigma, _, _ = tmq.peak_width_from_lnL(_lnL(sig0.samples().real, RHO_SQ)[None, :], DELTAT)
    assert 0.15 < float(sigma[0]) / DELTAT < 0.45, "not the under-resolved regime"

    s_err, b_err = [], []
    for phase in (0.0, 0.25, 0.5, 0.75):
        sig = BandLimited(amp=0.02, peak_sample=NPTS // 2 + phase,
                          n_period=8 * NPTS, m_hi=1400, background=0.12)
        ref = sig.truth()
        k = sig.samples()
        s_err.append(_simpson_value(k) - ref)
        b_err.append(_bandlimited(k) - ref)

    assert max(s_err) - min(s_err) > 0.5, s_err       # Simpson swings by ~2 nats
    assert max(np.abs(b_err)) < 1e-3, b_err
    assert max(np.abs(b_err)) < 0.01 * max(np.abs(s_err))


# ----------------------------------------------------------- the edge guard

def test_wrap_exposed_rows_fall_back_to_simpson_exactly():
    """A peak parked near the window edge is where the periodic interpolant is
    least trustworthy -- unguarded it was measured +88 nats HIGH, an upward bias
    in the evidence, which is the dangerous direction.  Such rows must be handed
    back the historical value bit-for-bit, so enabling the option can never make
    a row worse than the status quo."""
    for peak in (0.3, 2.3, 30.3):
        sig = BandLimited(amp=1.0, peak_sample=peak, n_period=8 * NPTS,
                          m_hi=1400, background=0.12)
        k = sig.samples()
        assert _bandlimited(k) == _simpson_value(k), peak
        assert tmq.last_report()['n_wrap_exposed_rows'] == 1


def test_one_exposed_row_does_not_contaminate_its_block():
    """The guard is per row.  A mis-centred row must fall back WITHOUT dragging a
    healthy row in the same block onto the Simpson path, and without inflating the
    refinement the healthy rows pay for."""
    bad = BandLimited(amp=1.0, peak_sample=1.3, n_period=8 * NPTS,
                      m_hi=1400, background=0.12)
    good = BandLimited(amp=1.0, peak_sample=NPTS // 2 + 0.25, n_period=8 * NPTS,
                       m_hi=1400, background=0.12, seed=11)
    k = np.stack([bad.samples(), good.samples()])
    out = tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    assert tmq.last_report()['n_wrap_exposed_rows'] == 1
    assert float(out[0]) == _simpson_value(bad.samples())
    assert abs(float(out[1]) - good.truth()) < 1e-3


def test_a_sharp_row_does_not_degrade_a_flat_row_sharing_its_block():
    """One refinement factor serves a whole block, so a flat row gets interpolated
    at a factor its own integrand never asked for.  That must not hurt it."""
    flat = BandLimited(amp=0.0012, peak_sample=NPTS // 2 + 0.3, n_period=8 * NPTS,
                       m_hi=1400, background=0.12, seed=11)
    sharp = BandLimited(amp=5.0, peak_sample=NPTS // 2 + 0.3, n_period=8 * NPTS,
                        m_hi=1400, background=0.12)
    k = np.stack([flat.samples(), sharp.samples()])
    out = tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    hist = tmq.last_report()['factor_histogram']
    assert tmq.last_report()['upsample_factor'] > 8
    assert abs(float(out[0]) - flat.truth()) < 1e-4
    # and the flat row must NOT have been dragged onto the sharp row's grid: the
    # factor is derived per row precisely so the broad majority stop paying for
    # the sharpest few.
    assert len(hist) == 2, hist
    assert min(hist) * 8 <= max(hist), hist


# ------------------------------------------------------------- fail-closed

def test_time_dependent_rho_sq_is_refused():
    """The precondition is checked, not trusted.  A time-dependent self-term (the
    banded / rotating-response path) would give a confident wrong number."""
    sig = BandLimited(amp=1.0, peak_sample=NPTS // 2)
    k = sig.samples()[None, :]
    rho = np.full(k.shape, RHO_SQ)
    rho[0, NPTS // 3] += 1e-9
    with pytest.raises(NotImplementedError):
        tmq.time_marginalize_bandlimited(k, rho, DELTAT, _lnL)


def test_ceiling_raises_rather_than_truncating_resolution():
    """Running out of refinement must be an error, never a silently coarser grid."""
    old = tmq.UPSAMPLE_FACTOR_MAX
    tmq.UPSAMPLE_FACTOR_MAX = 4
    try:
        sig = BandLimited(amp=40.0, peak_sample=NPTS // 2)
        k = sig.samples()[None, :]
        with pytest.raises(RuntimeError):
            tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    finally:
        tmq.UPSAMPLE_FACTOR_MAX = old


def test_unknown_quadrature_name_is_rejected():
    with pytest.raises(ValueError):
        tmq.validate_time_quadrature('bandlimted')     # sic


# --------------------------------------------------------------- the wiring

N_BUFFER = 4096


def _fake_likelihood_inputs(kappa_buffer):
    """Minimal inputs that drive the SHIPPED NoLoop function on the numpy backend.

    One detector, one (l,m) pair and zero cross terms, so the self-term is a
    constant and kappa reduces to the supplied rholm buffer times a fixed
    response factor.  The point is to exercise the argument plumbing; the physics
    is covered above against analytic truth.  The buffer is band-limited AND
    periodic on its own length, so whatever integer window the code gathers is a
    genuine band-limited segment -- the test does not need to predict ``ifirst``.
    """
    import lal
    import RIFT.lalsimutils as lsu

    det = 'H1'
    rholm = np.asarray(kappa_buffer, dtype=complex)[None, :]
    P = lsu.ChooseWaveformParams()
    P.deltaT = DELTAT
    P.tref = 1000000000.0
    for name, val in [('phi', 0.0), ('theta', 0.0), ('phiref', 0.0),
                      ('incl', 0.0), ('psi', 0.0)]:
        setattr(P, name, np.zeros(1) + val)
    P.dist = np.full(1, fl.distMpcRef * 1e6 * lal.PC_SI)
    # Put the window well inside the buffer: the epoch offset sets ifirst, and a
    # window running off the front would be zero-extended rather than gathered.
    return (P, {det: rholm}, {det: np.array([[2, 2]])},
            {det: np.zeros((1, 1), dtype=complex)}, {det: P.tref - 0.5})


def _buffer_signal(amp, roll=0):
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2, n_period=N_BUFFER,
                      m_hi=1400, background=0.12)
    ts = np.arange(N_BUFFER) * DELTAT
    return np.roll(sig.at(ts), int(roll))


def _shipped(tvals, args, **kw):
    P, rholms, lookupNK, ct, epochs = args
    return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, lookupNK, rholms, ct, ct, epochs, Lmax=2, xpy=np, **kw)


def _tuned_inputs(tvals, sigma_target_over_dt=0.25):
    """Build likelihood inputs whose lnL(t) actually sits in the under-resolved
    regime, by MEASURING what the shipped function produces rather than assuming
    it: the response factor and the gather offset are the code's business, not the
    test's.  Centres the peak in the window (an integer roll of a periodic
    band-limited buffer is still band-limited) and scales the amplitude using
    sigma ~ 1/sqrt(amp)."""
    amp, roll = 1.0, 0
    for _ in range(6):
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


def test_driver_flag_reaches_the_likelihood_and_changes_the_answer():
    """The wiring, not the helper.

    A flag that is computed correctly and then never reaches the likelihood is a
    documented failure mode in this repo -- a whole comparison campaign has been
    run against an inert stencil option here before.  So: set the module default
    the way the driver sets it, call the SHIPPED function, and require the number
    to actually move on an under-resolved peak.
    """
    pytest.importorskip('RIFT.lalsimutils')
    tvals = fl.marginalization_time_grid(0.075, DELTAT)
    assert len(tvals) == NPTS

    args, sigma_over_dt, jmax = _tuned_inputs(tvals)
    assert 0.1 < sigma_over_dt < 0.6, sigma_over_dt        # under-resolved, as intended
    guard = int(NPTS * tmq.EDGE_GUARD_FRACTION)
    assert guard < jmax < NPTS - 1 - guard, jmax           # and not wrap-exposed

    assert fl.TIME_QUADRATURE_DEFAULT == 'simpson', "default must not have moved"
    base = float(np.asarray(_shipped(tvals, args))[0])
    old = fl.TIME_QUADRATURE_DEFAULT
    try:
        fl.TIME_QUADRATURE_DEFAULT = 'bandlimited'      # exactly what the driver does
        new = float(np.asarray(_shipped(tvals, args))[0])
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old
    assert tmq.last_report()['upsample_factor'] > 1
    assert tmq.last_report()['n_wrap_exposed_rows'] == 0
    assert abs(new - base) > 1e-3, (base, new)

    # the explicit kwarg must override the module default, in both directions
    kw = float(np.asarray(_shipped(tvals, args, time_quadrature='bandlimited'))[0])
    assert kw == new
    fl.TIME_QUADRATURE_DEFAULT = 'bandlimited'
    try:
        assert float(np.asarray(_shipped(tvals, args, time_quadrature='simpson'))[0]) == base
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old


def test_unsupported_combinations_refuse_rather_than_silently_using_simpson():
    pytest.importorskip('RIFT.lalsimutils')
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2)
    tvals = fl.marginalization_time_grid(0.075, DELTAT)
    P, rholms, lookupNK, ct, epochs = _fake_likelihood_inputs([sig.samples()])
    common = dict(Lmax=2, xpy=np, time_quadrature='bandlimited')
    for extra in ({'n_cal': 2}, {'return_lnLt': True}, {'return_cal_components': True}):
        kw = dict(common); kw.update(extra)
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
    """These likelihoods have a time-DEPENDENT rho_sq, so the band-limited
    argument does not hold for them.  Enabling the option globally must make them
    RAISE, not quietly run Simpson -- otherwise the exclusion is invisible at the
    point of use.  Behavioural, not a source grep: the guard is the first thing
    the function does, so junk arguments must still produce NotImplementedError
    rather than a TypeError from further in."""
    mod = pytest.importorskip(module_name)
    func = getattr(mod, func_name)
    old = fl.TIME_QUADRATURE_DEFAULT
    try:
        fl.TIME_QUADRATURE_DEFAULT = 'bandlimited'
        with pytest.raises(NotImplementedError):
            func(None, None, None, None, None, None, None, None)
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old


# ------------------------------------------- non-finite lnL(t) (input space)

def test_minus_inf_next_to_the_peak_does_not_read_as_a_flat_integrand():
    """``lnL_t`` genuinely contains ``-inf`` in production: the distance-
    marginalization callback returns ``-inf`` outside its interpolation table.
    A three-point stencil that straddles that hole computes
    ``(-inf) - 2*(-inf) + (-inf) = NaN``, and ``NaN < 0`` is False -- so the row
    would report "no peak", derive a factor of 1 and be SILENTLY under-resolved,
    which is the exact failure this change exists to remove.  This cannot be
    caught by mutating the code (it is a missing case, not a wrong constant), so
    it is tested from the input side."""
    t = (np.arange(NPTS) - NPTS // 2) * DELTAT
    sigma_true = 0.05 * DELTAT
    base = -0.5 * (t / sigma_true) ** 2

    for label, hole in [("tails", (slice(0, 20), slice(-20, None))),
                        ("adjacent to the peak", (NPTS // 2 - 1,)),
                        ("both sides of the peak", (NPTS // 2 - 1, NPTS // 2 + 1))]:
        lnL = base.copy()
        for h in hole:
            lnL[h] = -np.inf
        sigma, _, meas = tmq.peak_width_from_lnL(lnL[None, :], DELTAT)
        assert bool(meas[0]), label
        assert np.isclose(float(sigma[0]), sigma_true, rtol=1e-9), (label, sigma)
        assert int(tmq.required_upsample_factors(sigma, DELTAT)[0]) > 1, label


def test_a_signal_free_row_is_reported_as_flat_not_as_wrap_exposed():
    """A row with no signal in it -- an extrinsic sample in an antenna null, where
    kappa is numerically zero -- has a constant lnL(t) and therefore an argmax of
    0 by convention.  Applying the edge guard to it would report it as
    wrap-exposed, which in a production log reads as a mis-centred window rather
    than as a row with nothing in it.  The edge guard is only meaningful for rows
    that HAVE a peak."""
    sig = BandLimited(amp=1.0, peak_sample=NPTS // 2)
    k = np.stack([np.zeros(NPTS, dtype=complex), sig.samples()])
    out = tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    rep = tmq.last_report()
    assert rep['n_flat_rows'] == 1, rep
    assert rep['n_wrap_exposed_rows'] == 0, rep
    assert rep['n_unmeasurable_rows'] == 0, rep
    # and it is still integrated correctly: a constant integrand over the window
    expect = _lnL(0.0, RHO_SQ) + np.log((NPTS - 1) * DELTAT)
    assert abs(float(out[0]) - expect) < 1e-12, (out[0], expect)


def test_unmeasurable_row_falls_back_and_is_counted():
    """A row whose curvature cannot be evaluated at ANY stencil half-width must be
    counted and given the historical value -- never silently assigned factor 1,
    which is indistinguishable from a genuinely flat integrand."""
    sig = BandLimited(amp=1.0, peak_sample=NPTS // 2)
    k = np.stack([np.zeros(NPTS, dtype=complex), sig.samples()])

    def lnL_with_hole(kappa_term, rho_sq):
        out = _lnL(kappa_term, rho_sq)
        out = np.where(np.abs(np.asarray(kappa_term)) > 0, out, -np.inf)
        return out

    out = tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT,
                                           lnL_with_hole)
    rep = tmq.last_report()
    assert rep['n_unmeasurable_rows'] == 1, rep
    assert rep['n_refined_rows'] == 1, rep
    # `exposed` is gated on `has_peak`, which already implies `measurable`, so an
    # unmeasurable row can never also be exposed -- asserting the two do not
    # overlap is vacuous.  What is worth pinning is that the counters PARTITION
    # the batch, so no row can fall through a gap between them and be invisible.
    assert (rep['n_refined_rows'] + rep['n_wrap_exposed_rows']
            + rep['n_unmeasurable_rows'] + rep['n_flat_rows']
            + _n_resolved(rep)) == rep['n_rows'], rep
    # zero likelihood over the whole window integrates to zero: the answer is
    # -inf, which is what the historical global-offset path returns.  NaN here
    # would propagate into the sampler weights.
    assert float(out[0]) == -np.inf, out[0]
    assert abs(float(out[1]) - sig.truth()) < 1e-6


def test_a_row_changes_if_and_only_if_it_was_under_resolved():
    """The guarantee, stated so it can be checked rather than argued.

    Every row that is NOT refined -- wrap-exposed, unmeasurable, or already
    resolved -- must come back with the historical Simpson value, so enabling
    this option cannot make any row worse than the status quo.  Letting an
    unrefined row fall through to a coarse trapezoid instead is numerically a
    non-event, but it changes the rule for rows this option was never meant to
    touch and forfeits exactly this property.
    """
    rows, expect_refined = [], []
    # resolved (no refinement warranted)
    rows.append(BandLimited(amp=0.002, peak_sample=NPTS // 2).samples()); expect_refined.append(False)
    # signal-free
    rows.append(np.zeros(NPTS, dtype=complex)); expect_refined.append(False)
    # wrap-exposed
    rows.append(BandLimited(amp=1.0, peak_sample=2.3, n_period=8 * NPTS,
                            m_hi=1400, background=0.12).samples()); expect_refined.append(False)
    # genuinely under-resolved
    rows.append(BandLimited(amp=1.0, peak_sample=NPTS // 2 + 0.25, n_period=8 * NPTS,
                            m_hi=1400, background=0.12).samples()); expect_refined.append(True)

    k = np.stack(rows)
    out = tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    rep = tmq.last_report()
    assert rep['n_refined_rows'] == sum(expect_refined), rep

    for i, should_change in enumerate(expect_refined):
        historical = _simpson_value(rows[i])
        if should_change:
            assert float(out[i]) != historical, i
        else:
            assert float(out[i]) == historical, (i, out[i], historical)


def test_remeasure_on_the_dense_grid_repairs_an_under_derived_factor():
    """The remeasure-and-double step is what makes the derivation an assertion
    rather than a guess.  Force the derivation to hand back a factor that is far
    too small and require the refinement loop to notice on the dense grid and
    recover the right answer anyway."""
    sig = BandLimited(amp=5.0, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()[None, :]
    rho = np.full(k.shape, RHO_SQ)
    honest = tmq.time_marginalize_bandlimited(k, rho, DELTAT, _lnL)
    honest_factor = tmq.last_report()['upsample_factor']
    assert honest_factor >= 16

    real = tmq.required_upsample_factors
    tmq.required_upsample_factors = lambda sigma, dx, xpy=np: real(sigma, dx, xpy=xpy) // 8
    try:
        got = tmq.time_marginalize_bandlimited(k, rho, DELTAT, _lnL)
    finally:
        tmq.required_upsample_factors = real
    rep = tmq.last_report()
    assert rep['n_refinements'] > 0, rep
    assert rep['upsample_factor'] == honest_factor, rep
    assert abs(float(got[0]) - float(honest[0])) < 1e-9
    assert abs(float(got[0]) - sig.truth()) < 1e-6




# ------------------------------------------------------- the driver CLI

def _run_driver(extra_args):
    """Invoke the ILE driver and return (returncode, combined output).

    A subprocess, deliberately.  The option's whole job is to travel from a
    command line into the likelihood, and the guard that stops it being silently
    inert lives in the driver's startup, not in the library -- so a test that
    imports the library cannot see it.  The driver exits long before any data is
    needed, so this costs one interpreter start.
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


def test_driver_rejects_a_misspelled_quadrature_name():
    """A misspelled stencil name was once absorbed as "not truthy" and silently
    ran a different likelihood here.  A typo in this option has to be loud."""
    rc, out = _run_driver(['--time-marginalization-quadrature', 'bandlimted'])
    assert rc != 0
    assert 'bandlimted' in out and 'simpson' in out


def test_driver_refuses_configurations_that_cannot_honour_the_option():
    """Refuse, do not ignore.  Each of these would otherwise run the historical
    Simpson quadrature while the startup banner said otherwise -- which is how a
    comparison campaign gets run against an inert flag."""
    for missing in ([], ['--time-marginalization'],
                    ['--time-marginalization', '--vectorized'],
                    _HONOURED + ['--rotation-slow'],
                    _HONOURED + ['--freqresponse']):
        rc, out = _run_driver(['--time-marginalization-quadrature', 'bandlimited'] + missing)
        assert rc != 0, (missing, out[-2000:])
        assert 'cannot honour it' in out, (missing, out[-2000:])


def test_driver_announces_the_quadrature_it_will_actually_use():
    rc, out = _run_driver(['--time-marginalization-quadrature', 'bandlimited'] + _HONOURED)
    assert 'Time-marginalization quadrature: bandlimited' in out
    assert 'honoured by this configuration: True' in out
    # and the default stays put when the option is not given
    rc, out = _run_driver(_HONOURED)
    assert 'Time-marginalization quadrature: simpson' in out



# --------------------------------------------------------------- GPU parity

def _cupy_or_skip():
    """cupy, or skip -- unless the GPU gate demands a device, in which case FAIL.

    `RIFT_CI_REQUIRE_GPU=1` is how `.travis/test-integrate.sh` says "this job runs
    on hardware".  A skip under that flag would be a GPU gate reporting green
    without having touched a GPU, which is the failure mode this whole file is
    written against.
    """
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("cupy imported but reports zero CUDA devices")
        return cupy
    except Exception as exc:
        if os.environ.get('RIFT_CI_REQUIRE_GPU') == '1':
            pytest.fail("RIFT_CI_REQUIRE_GPU=1 but cupy/GPU unavailable: %s" % exc)
        pytest.skip("cupy/GPU unavailable: %s" % exc)


def test_bandlimited_runs_on_the_gpu_backend_and_matches_numpy():
    """The backend-generic code must actually RUN on cupy, not merely look like it.

    This path shipped once already having never been executed on a GPU: the
    likelihood omitted the caller's `simps`, the module defaulted to scipy's, and
    scipy raises `TypeError: Implicit conversion to a NumPy array is not allowed`
    on a cupy array -- so EVERY `--vectorized --gpu` run of the option crashed
    while all 46 CPU tests stayed green.  Reading the cupy API is not a substitute
    for executing it.
    """
    cupy = _cupy_or_skip()
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    flat = BandLimited(amp=0.002, peak_sample=NPTS // 2)
    edge = BandLimited(amp=1.0, peak_sample=2.3, n_period=8 * NPTS,
                       m_hi=1400, background=0.12)
    k = np.stack([sig.samples(), flat.samples(), edge.samples(),
                  np.zeros(NPTS, dtype=complex)])
    r = np.full(k.shape, RHO_SQ)

    simps_cpu = simpson
    out_np = np.asarray(tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL,
                                                         simps=simps_cpu, xpy=np))
    rep_np = tmq.last_report()

    from RIFT.likelihood import optimized_gpu_tools
    out_cp = cupy.asnumpy(tmq.time_marginalize_bandlimited(
        cupy.asarray(k), cupy.asarray(r), DELTAT, _lnL,
        simps=optimized_gpu_tools.simps, xpy=cupy))
    rep_cp = tmq.last_report()

    # Same classification and the same derived factors on both backends.
    for key in ('upsample_factor', 'factor_histogram', 'n_refined_rows',
                'n_wrap_exposed_rows', 'n_unmeasurable_rows', 'n_flat_rows'):
        assert rep_np[key] == rep_cp[key], (key, rep_np[key], rep_cp[key])
    assert rep_np['n_refined_rows'] >= 1

    # The REFINED rows integrate with trapezoid on the dense grid, which has no
    # even/odd Simpson ambiguity, so the two backends must agree to round-off.
    # (Rows that fall back use each backend's OWN Simpson rule, and those two
    # rules genuinely differ for even npts -- see the CPU/GPU note below.)
    # Only the REFINED rows are required to agree: they integrate with trapezoid
    # on the dense grid, which has no even/odd Simpson ambiguity.  Rows that fall
    # back use each backend's OWN Simpson rule, and those two rules genuinely
    # differ for even npts -- asserting agreement over all rows would be
    # asserting the absence of a divergence this file documents as real.
    refined = np.array([f > 1 for f in _row_factors(k, r)])
    assert refined.any()
    fin = np.isfinite(out_np) & np.isfinite(out_cp) & refined
    assert np.max(np.abs(out_np[fin] - out_cp[fin])) < 1e-6


def test_the_likelihood_hands_the_bandlimited_path_its_own_simpson_rule():
    """Rows that fall back must reproduce what THIS run would have returned.

    `factored_likelihood` integrates with scipy on CPU and with
    `optimized_gpu_tools.simps` on GPU, and the two are NOT interchangeable: the
    vendored GPU copy is an old scipy with `even='avg'` while modern scipy uses
    the Cartwright correction, so for EVEN npts -- production is 614 at srate
    4096 -- they disagree.  A private scipy copy inside the module would be
    bit-for-bit on CPU and quietly wrong on GPU.
    """
    # Behavioural, not a source grep: a non-numpy backend with no `simps` must
    # REFUSE, because the scipy default cannot consume a device array and that
    # default is precisely how every GPU run of this option crashed.
    class _FakeDevice(object):
        """Minimal stand-in for a non-numpy backend, so the guard is exercised
        without needing a GPU."""
        def __getattr__(self, name):
            return getattr(np, name)

    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    with pytest.raises(ValueError):
        tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL, xpy=_FakeDevice())
    # ... and is satisfied once a rule is supplied
    tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL, simps=simpson,
                                     xpy=_FakeDevice())

    # The likelihood must hand its OWN rule over.  Parse the real call, not a
    # prefix of it: slicing at the first ')' truncates inside `float(deltaT)`.
    import ast, inspect
    tree = ast.parse(inspect.getsource(
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop).lstrip())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, 'attr', None) == 'time_marginalize_bandlimited']
    assert calls, "the likelihood no longer calls time_marginalize_bandlimited"
    for c in calls:
        kw = {k2.arg: k2.value for k2 in c.keywords}
        assert 'simps' in kw and getattr(kw['simps'], 'id', None) == 'simps', \
            "call site does not forward the caller's Simpson rule"
        assert 'lnL_coarse' in kw, \
            "call site does not forward the already-computed coarse lnL"


def _row_factors(k, r):
    """Per-row derived factor, as the integrator computes it."""
    lnL = _lnL(np.asarray(k).real, np.asarray(r))
    sigma, jmax, meas = tmq.peak_width_from_lnL(lnL, DELTAT)
    guard = max(1, int(k.shape[-1] * tmq.EDGE_GUARD_FRACTION))
    ok = meas & np.isfinite(sigma) & (jmax >= guard) & (jmax <= k.shape[-1] - 1 - guard)
    f = np.maximum(tmq.required_upsample_factors(sigma, DELTAT), 1)
    return np.where(ok, f, 1)


@pytest.mark.parametrize("n", [153, 307, 613, 614, 1228, 2457, 8, 9, 3])
def test_upsample_is_exact_for_odd_npts_too(n):
    """ODD npts is the COMMON case in production, not an exotic one.

    `marginalization_time_grid(0.075, 1/srate)` gives npts = 153 / 307 / 614 /
    1228 / 2457 at srate 1024 / 2048 / 4096 / 8192 / 16384 -- odd at THREE of the
    five, including 16384, the low-mass rate.  An earlier split at `h = n//2`
    placed the highest positive frequency at a negative frequency for odd n:
    still exact AT the original samples (so a "reproduces the input" check passes)
    and wrong everywhere between them, by 0.41 at n=613 and 0.54 at n=307 against
    an analytic truth of order unity.
    """
    R = 4
    rng = np.random.default_rng(1)
    ms = np.arange(1, (n - 1) // 2 + 1)          # fill EVERY bin up to Nyquist
    c = (rng.normal(size=ms.size) + 1j * rng.normal(size=ms.size)) / (1 + ms / 50.0)
    t = np.arange(n) / float(n)
    td = np.arange(n * R) / float(n * R)
    x = np.exp(2j * np.pi * np.outer(t, ms)) @ c
    exact = np.exp(2j * np.pi * np.outer(td, ms)) @ c
    up = tmq.bandlimited_upsample(x[None, :], R)[0]
    assert np.allclose(up, exact, atol=1e-9, rtol=0), np.abs(up - exact).max()


def test_which_rows_change_relative_to_the_SHIPPED_historical_expression():
    """The guarantee, checked against the historical GLOBAL-offset expression.

    An earlier version of this test compared against a per-row-offset Simpson
    helper -- the same expression the code under test uses for its fallback rows
    -- so it was common-mode with the thing it was meant to check and could not
    fail.  The shipped path offsets by a SINGLE GLOBAL maximum over the whole
    block, so a multi-row batch with production-scale dynamic range is required
    to see the difference at all.
    """
    def historical(kappa_rows, rho):
        lnL_t = _lnL(np.asarray(kappa_rows).real, rho)
        lnLmax = lnL_t.max()                       # GLOBAL, as the shipped path does
        return lnLmax + np.log(simpson(np.exp(lnL_t - lnLmax), dx=DELTAT, axis=-1))

    loud = BandLimited(amp=1.0, peak_sample=NPTS // 2 + 0.25, n_period=8 * NPTS,
                       m_hi=1400, background=0.12).samples()
    quiet = BandLimited(amp=0.002, peak_sample=NPTS // 2).samples() * 1e-3
    k = np.stack([loud, quiet])
    r = np.full(k.shape, RHO_SQ)

    hist = historical(k, r)
    new = np.asarray(tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL))
    rep = tmq.last_report()

    # The QUADRATURE changed for exactly the under-resolved row.
    assert rep['n_refined_rows'] == 1, rep
    assert _row_factors(k, r)[0] > 1 and _row_factors(k, r)[1] == 1

    # The refined row moved, as intended.
    assert abs(new[0] - hist[0]) > 1e-3

    # And the documented second change: the unrefined row underflowed to -inf
    # under the shared global offset and now comes back finite.  This is NOT
    # "unchanged"; pin it so the PR text and the code cannot drift apart.
    span = _lnL(k.real, r).max() - _lnL(k.real, r)[1].max()
    assert span > 745, span               # the underflow threshold for exp()
    assert hist[1] == -np.inf
    assert np.isfinite(new[1])


def test_return_lnLt_still_works_when_the_module_default_is_bandlimited():
    """The group's standard extrinsic stage must not die at the export step.

    `--add-extrinsic --add-extrinsic-time-resampling` maps to
    `--resample-time-marginalization`, whose `resample_samples()` calls the
    likelihood with `return_lnLt=True` and no explicit quadrature.  Raising on
    the INHERITED default made that configuration run the entire integration and
    then crash with no output.  `return_lnLt` returns lnL(t) on the original grid
    and takes no time integral, so the quadrature is inapplicable, not ignored --
    but asking for it EXPLICITLY there is still a caller error.
    """
    pytest.importorskip('RIFT.lalsimutils')
    tvals = fl.marginalization_time_grid(0.075, DELTAT)
    args = _fake_likelihood_inputs(_buffer_signal(1.0))
    old = fl.TIME_QUADRATURE_DEFAULT
    try:
        fl.TIME_QUADRATURE_DEFAULT = 'bandlimited'
        lnLt = np.asarray(_shipped(tvals, args, return_lnLt=True))
        assert lnLt.shape[-1] == NPTS
        with pytest.raises(NotImplementedError):
            _shipped(tvals, args, return_lnLt=True, time_quadrature='bandlimited')
    finally:
        fl.TIME_QUADRATURE_DEFAULT = old


def test_a_nan_self_term_does_not_abort_the_run():
    """NaN rows are NORMAL -- the defensive proposal component deliberately draws
    physically-extreme points where the likelihood is NaN, and the historical path
    returns NaN for that row and moves on.  A bare `rho_sq == rho_sq[...,:1]`
    tripwire makes `nan != nan` abort the whole ILE process, blaming a
    rotating-response path that is not in use."""
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2)
    k = np.stack([sig.samples(), sig.samples()])
    r = np.full(k.shape, RHO_SQ)
    r[1, :] = np.nan
    out = tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL)
    assert np.isfinite(float(out[0]))
    assert np.isnan(float(out[1]))
    # a genuinely time-DEPENDENT self-term must still be refused
    r2 = np.full(k.shape, RHO_SQ); r2[0, NPTS // 3] += 1e-9
    with pytest.raises(NotImplementedError):
        tmq.time_marginalize_bandlimited(k, r2, DELTAT, _lnL)


def _n_resolved(rep):
    """Rows with a real peak that simply needed no refinement."""
    return (rep['n_rows'] - rep['n_refined_rows'] - rep['n_wrap_exposed_rows']
            - rep['n_unmeasurable_rows'] - rep['n_flat_rows'])


def test_the_edge_guard_covers_the_RIGHT_edge_too():
    """Both ends, not just the one the first test happened to use.

    The guard is `(jmax < g) | (jmax > npts-1-g)`.  Dropping the second term, or
    an off-by-one in it, leaves the left edge covered and the right edge wide
    open -- and a peak parked at the last sample then returns +88.8 nats ABOVE
    truth, the evidence-inflating direction the guard exists to stop.  Every
    fixture in the original suite parked peaks near sample 0.
    """
    for peak in (NPTS - 1.3, NPTS - 3.3, NPTS - 31.3):
        sig = BandLimited(amp=1.0, peak_sample=peak, n_period=8 * NPTS,
                          m_hi=1400, background=0.12)
        k = sig.samples()
        assert _bandlimited(k) == _simpson_value(k), peak
        assert tmq.last_report()['n_wrap_exposed_rows'] == 1, peak
    # and a peak just INSIDE the right guard is still refined, so the guard is
    # not merely swallowing everything on that side
    inside = BandLimited(amp=1.0, peak_sample=NPTS // 2 + 0.25, n_period=8 * NPTS,
                         m_hi=1400, background=0.12)
    _bandlimited(inside.samples())
    assert tmq.last_report()['n_wrap_exposed_rows'] == 0


@pytest.mark.parametrize("phase_marg", [False, True])
def test_phase_marginalization_reaches_the_new_path(phase_marg):
    """`--distance-marginalization --phase-marginalization` is the standard
    production call site, and it passes `phase_marginalization=True` with a
    NONLINEAR callback.  Every original fixture used the affine helper with
    `kappa.real`, for which lnL(t) is itself exactly band-limited -- so dropping
    the `abs()` entirely changed nothing any test could see."""
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    term = (lambda z: np.abs(z)) if phase_marg else (lambda z: z.real)

    got = float(tmq.time_marginalize_bandlimited(
        k, r, DELTAT, _lnL, phase_marginalization=phase_marg)[0])

    # analytic truth for THIS integrand
    n = (NPTS - 1) * 128 + 1
    td = sig.j0 * DELTAT + np.arange(n) * (DELTAT / 128)
    ref = _log_trapz(_lnL(term(sig.at(td)), RHO_SQ), DELTAT / 128)
    assert abs(got - ref) < 1e-3, (phase_marg, got - ref)

    # the two settings must actually differ, or the parametrisation proves nothing
    other = float(tmq.time_marginalize_bandlimited(
        k, r, DELTAT, _lnL, phase_marginalization=not phase_marg)[0])
    assert abs(got - other) > 1e-3, "abs() vs real() made no difference"


def test_a_nonlinear_distance_marginalization_style_callback():
    """The production callback is a table interpolation, not `kappa - rho_sq/2`.
    For the affine helper lnL(t) is itself band-limited, which is a much easier
    problem than the one production actually poses."""
    def distmarg_like(x, rho_sq):
        # monotone, nonlinear, and -inf outside a table range, like the real one
        z = np.asarray(x) / np.sqrt(np.asarray(rho_sq) + 1.0)
        out = np.where(z > -3.0, np.log1p(np.exp(np.clip(z, -50, 50))) * 40.0, -np.inf)
        return out
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    got = float(tmq.time_marginalize_bandlimited(k, r, DELTAT, distmarg_like)[0])
    n = (NPTS - 1) * 128 + 1
    td = sig.j0 * DELTAT + np.arange(n) * (DELTAT / 128)
    ref = _log_trapz(distmarg_like(sig.at(td).real, RHO_SQ), DELTAT / 128)
    simp = _log_simps(distmarg_like(k[0].real, RHO_SQ), DELTAT)
    assert abs(got - ref) < 1e-2, got - ref
    assert abs(got - ref) < 0.05 * abs(simp - ref)


def test_the_memory_chunking_path_assembles_its_result():
    """Production runs `--n-chunk 10000`, so EVERY real call chunks; the suite's
    largest batch is 4 rows, so the assembly branch never ran.  Dropping all but
    the first chunk was invisible."""
    rows = [BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.1 * i,
                        n_period=8 * NPTS, m_hi=1400, background=0.12,
                        seed=7 + i).samples() for i in range(6)]
    k = np.stack(rows)
    r = np.full(k.shape, RHO_SQ)
    whole = np.asarray(tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL))
    old = tmq._DENSE_CHUNK_BYTES
    try:
        tmq._DENSE_CHUNK_BYTES = 4096          # force several chunks per group
        chunked = np.asarray(tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL))
    finally:
        tmq._DENSE_CHUNK_BYTES = old
    assert chunked.shape == whole.shape == (6,)
    assert np.array_equal(chunked, whole), np.abs(chunked - whole).max()


def test_the_one_ulp_factor_bump_is_exercised():
    """`2**ceil(log2(need))` can land one power of two SHORT when log2 rounds down
    for a `need` a hair above a power of two -- erring LOW, i.e. silently
    under-resolving.  A log-spaced sweep never lands there; this does."""
    dx = DELTAT
    for kexp in range(0, 12):
        need = 2.0 ** kexp
        sigma = tmq.UPSAMPLE_SAFETY * dx / np.nextafter(need, np.inf)
        f = int(tmq.required_upsample_factors(np.array([sigma]), dx)[0])
        assert f >= tmq.UPSAMPLE_SAFETY * dx / sigma, (kexp, f)
        assert dx / f <= sigma / tmq.UPSAMPLE_SAFETY, (kexp, f)


def test_argmax_ignores_non_finite_bins():
    """`argmax` over a raw array containing NaN returns the NaN's index, which
    would put the whole width measurement on a bin that carries no likelihood."""
    t = (np.arange(NPTS) - NPTS // 2) * DELTAT
    lnL = -0.5 * (t / (0.3 * DELTAT)) ** 2
    lnL[10] = np.nan
    sigma, jmax, meas = tmq.peak_width_from_lnL(lnL[None, :], DELTAT)
    assert int(jmax[0]) == NPTS // 2, jmax
    assert bool(meas[0]) and np.isclose(float(sigma[0]), 0.3 * DELTAT, rtol=1e-9)


def test_report_sigma_t_min_is_the_width_that_was_resolved():
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.25)
    k = sig.samples()[None, :]
    tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    rep = tmq.last_report()
    coarse, _, _ = tmq.peak_width_from_lnL(_lnL(k.real, RHO_SQ), DELTAT)
    assert np.isfinite(rep['sigma_t_min'])
    assert abs(rep['sigma_t_min'] - float(coarse[0])) < 0.2 * float(coarse[0]), rep
    assert DELTAT / rep['upsample_factor'] <= rep['sigma_t_min'] / tmq.UPSAMPLE_SAFETY


def test_the_tuned_constants_are_pinned_to_their_measured_values():
    """These are not free parameters.  Each is justified by a measured table in
    DESIGN_time_marginalization_quadrature.md, and the suite otherwise pins them
    only to within a factor of ~10 -- so changing one could pass CI while
    invalidating the argument behind it.  Changing a value here is the deliberate
    act of also updating that table."""
    assert tmq.UPSAMPLE_SAFETY == 2.0
    assert tmq.EDGE_GUARD_FRACTION == 0.125
    assert tmq.UPSAMPLE_FACTOR_MAX == 4096
    assert tmq.CURVATURE_STENCIL_HALFWIDTHS == (1, 2, 4, 8)


def test_driver_banner_reports_what_is_ACTUALLY_IN_FORCE():
    """The driver's single load-bearing line is the assignment to
    `factored_likelihood.TIME_QUADRATURE_DEFAULT`.  Deleting it, or hardcoding
    'simpson' there, leaves the flag inert -- and every banner-inspecting test
    stays green if the banner is built from `opts`.  The banner therefore prints
    the value READ BACK out of the module, and this asserts on that."""
    rc, out = _run_driver(['--time-marginalization-quadrature', 'bandlimited'] + _HONOURED)
    assert 'Time-marginalization quadrature: bandlimited' in out, out[-2000:]
    import re
    m = re.search(r'Time-marginalization quadrature: (\S+) \(from', out)
    assert m and m.group(1) == 'bandlimited', out[-2000:]
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'bin',
        'integrate_likelihood_extrinsic_batchmode')).read()
    assert 'factored_likelihood.TIME_QUADRATURE_DEFAULT,' in src, \
        "the banner no longer reads the value back out of the module"


def test_driver_does_not_refuse_an_ordinary_default_run():
    """The refuse-guard is gated on the option being non-default.  Dropping that
    gate makes the driver reject EVERY ordinary ILE run that is not
    `--time-marginalization --vectorized --gpu` -- and no driver test ever
    launched a default configuration, so nothing noticed."""
    for args in ([], ['--time-marginalization'], ['--vectorized']):
        rc, out = _run_driver(args)
        assert 'cannot honour it' not in out, (args, out[-2000:])
        assert 'Time-marginalization quadrature: simpson' in out, (args, out[-2000:])


def _quadrature_banner(out):
    """The quadrature banner line, matched SPECIFICALLY.

    The pre-existing `--interpolate-time` banner carries the identical phrase
    "honoured by this configuration", so a bare substring test matches whichever
    line happens to say what you were hoping for.  A mutation making the
    quadrature banner claim `True` unconditionally survived exactly that way:
    the stencil line still said `False` and the assertion passed.
    """
    import re
    m = re.search(r'^\s*Time-marginalization quadrature: (\S+) '
                  r'\(from --time-marginalization-quadrature (.+?)\); '
                  r'honoured by this configuration: (True|False)\s*$',
                  out, re.MULTILINE)
    assert m is not None, "no quadrature banner line found:\n" + out[-3000:]
    return m.group(1), m.group(3)


def test_driver_banner_does_not_claim_to_honour_what_it_cannot():
    quad, honoured = _quadrature_banner(_run_driver(['--time-marginalization'])[1])
    assert (quad, honoured) == ('simpson', 'False')
    quad, honoured = _quadrature_banner(_run_driver(_HONOURED)[1])
    assert (quad, honoured) == ('simpson', 'True')
    quad, honoured = _quadrature_banner(
        _run_driver(['--time-marginalization-quadrature', 'bandlimited'] + _HONOURED)[1])
    assert (quad, honoured) == ('bandlimited', 'True')


def test_the_edge_guard_band_is_exactly_the_outer_fraction():
    """Pin both boundaries to the sample, not merely "near the edge".

    An off-by-one in the upper term -- `jmax > npts - guard` instead of
    `npts - 1 - guard` -- leaves exactly one row's worth of the right guard band
    open, and every peak-placement fixture is far enough inside that both spellings
    agree.  Driving the argmax to a chosen bin makes the boundary itself the
    subject.
    """
    guard = max(1, int(NPTS * tmq.EDGE_GUARD_FRACTION))

    def row_peaking_at(j):
        t = np.arange(NPTS, dtype=float)
        return (np.exp(-0.5 * ((t - j) / 0.35) ** 2) * 40.0).astype(complex)

    # The last EXPOSED index and the first ACCEPTED one, at both ends.  These four
    # are what an off-by-one in either term moves, and every peak-placement
    # fixture elsewhere is far enough inside that both spellings agree.
    for j, expect_exposed in ((guard - 1, True), (guard, False),
                              (NPTS - 1 - guard, False), (NPTS - guard, True)):
        k = row_peaking_at(j)[None, :]
        r = np.full(k.shape, RHO_SQ)
        sigma, jmax, meas = tmq.peak_width_from_lnL(_lnL(k.real, r), DELTAT)
        assert int(jmax[0]) == j and np.isfinite(sigma[0]), (j, jmax, sigma)
        tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL)
        rep = tmq.last_report()
        assert (rep['n_wrap_exposed_rows'] == 1) == expect_exposed, (j, guard, rep)
        # accepted rows here are sharp enough to be refined, so the guard is
        # deciding something rather than being masked by a factor of 1
        assert (rep['n_refined_rows'] == 1) == (not expect_exposed), (j, rep)

    # A peak on the very first or last SAMPLE is a documented corner: the
    # curvature stencil is clipped inward, so it measures a positive second
    # difference and the row classifies as FLAT rather than wrap-exposed.  That
    # under-states the window-centring problem in the diagnostic, but it is safe
    # -- what matters is that such a row is never refined, so it gets the
    # historical value either way.
    for j in (0, NPTS - 1):
        k = row_peaking_at(j)[None, :]
        r = np.full(k.shape, RHO_SQ)
        out = tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL)
        rep = tmq.last_report()
        assert rep['n_refined_rows'] == 0, (j, rep)
        assert rep['n_flat_rows'] == 1, (j, rep)
        assert float(out[0]) == _simpson_value(k[0]), j


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
