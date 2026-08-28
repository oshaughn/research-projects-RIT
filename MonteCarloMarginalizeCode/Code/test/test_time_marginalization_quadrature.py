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
    assert rep['n_fallback_rows'] == 1, rep
    # counted unconditionally: an all -inf row also has argmax 0, so a counter
    # written as "unmeasurable AND not exposed" would hide it behind the guard
    assert rep['n_wrap_exposed_rows'] == 0, rep
    # zero likelihood over the whole window integrates to zero: the answer is
    # -inf, which is what the historical global-offset path returns.  NaN here
    # would propagate into the sampler weights.
    assert float(out[0]) == -np.inf, out[0]
    assert abs(float(out[1]) - sig.truth()) < 1e-6


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



if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
