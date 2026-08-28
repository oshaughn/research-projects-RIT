#!/usr/bin/env python3
"""Tests for the band-limited time-marginalization quadrature.

Two layers, deliberately:

* the module in isolation, against integrands whose exact value is known in
  closed form -- no LAL, no waveforms, deterministic, seconds to run;
* the WIRING, i.e. that flipping ``time_quadrature`` on the shipped likelihood
  function actually changes the number it returns.  A silently-inert opt-in
  flag is a known failure mode in this codebase, and a test that only exercises
  the helper would not catch one.
"""
import math

import numpy as np
import pytest

from RIFT.likelihood import time_marginalization_quadrature as tq


# --------------------------------------------------------------------------
# A synthetic integrand whose continuous form is known exactly.
#
# kappa(t) = sum_k c_k exp(2 pi i f_k t), every f_k an exact multiple of
# 1/(n*deltaT) with |f_k| < Nyquist.  Then kappa is band-limited AND periodic
# on the sample window, so the DFT interpolation of its samples is not merely
# accurate but exact -- which is precisely the claim under test.
# --------------------------------------------------------------------------
def _synthetic_kappa(n, deltaT, amp, t0=None, seed=7, spectrum="gaussian",
                     n_modes=6):
    """A band-limited, window-periodic kappa(t) whose peak sits exactly at t0.

    Every frequency is an exact multiple of 1/(n*deltaT) and below Nyquist, so
    kappa is band-limited AND periodic on the sample window: the DFT
    interpolation of its samples is then not merely accurate but exact, which
    is the claim under test.  The mode coefficients are real and positive, so
    Re kappa peaks at exactly t=t0 with curvature known in closed form.

    `amp` plays the role of rho^2: larger means a narrower exp(lnL(t)) against
    the same fixed grid, i.e. exactly what a louder event does.

    spectrum:
      'gaussian' -- a smooth spectral taper over every mode, giving a single
          dominant pulse with small side lobes.  This is the physical
          matched-filter shape: a slowly varying envelope times a carrier.
      'comb' -- a handful of random high frequencies and no envelope, so the
          row is full of comparable spikes and the SAMPLED maximum need not sit
          anywhere near the true one.  Deliberately adversarial; it exists to
          pin the behaviour the argmax-only width rule got wrong.
    """
    if t0 is None:
        t0 = 0.5 * (n - 1) * deltaT
    rng = np.random.default_rng(seed)
    kmax = (n - 1) // 2
    if spectrum == "gaussian":
        ks = np.arange(1, kmax + 1)
        a = np.exp(-0.5 * (ks / (0.35 * kmax)) ** 2)
    elif spectrum == "comb":
        ks = rng.choice(np.arange(kmax // 2, kmax + 1), size=n_modes,
                        replace=False)
        a = rng.uniform(0.5, 1.5, size=n_modes)
    else:
        raise ValueError(spectrum)
    a = a * (amp / a.sum())                 # Re kappa(t0) == amp, exactly
    freqs = ks / (n * deltaT)

    def kappa_of_t(t):
        t = np.asarray(t, dtype=float)
        ph = 2j * np.pi * freqs[:, None] * (t.ravel()[None, :] - t0)
        return (a[:, None] * np.exp(ph)).sum(axis=0).reshape(t.shape)

    kappa_of_t.curvature = -float((a * (2 * np.pi * freqs) ** 2).sum())
    kappa_of_t.t0 = t0
    kappa_of_t.amp = amp
    return kappa_of_t


def _helper(kappa_real, rho_sq):
    """The shipped default loglikelihood callback."""
    return kappa_real - 0.5 * rho_sq


def _log_trapz(t, lnL):
    m = lnL.max()
    return m + math.log(np.trapz(np.exp(lnL - m), t))


def _reference_log_integral(kappa_of_t, rho_sq, t_grid, oversample=257,
                            lnL_of=None):
    """log int exp(lnL) dt over [t_grid[0], t_grid[-1]], from the ANALYTIC kappa.

    Independent of the module: it never touches an FFT, it evaluates the closed
    form on a grid `oversample` times finer (a prime factor, so it shares no
    nodes with any power-of-two factor the module might pick).

    Returns ``(value, err)``, where `err` is the reference's OWN convergence
    estimate -- the shift between this resolution and half of it.  Tests
    compare against `err` rather than a hardcoded tolerance, so a test can
    never be tighter than the truth it is checking against.
    """
    if lnL_of is None:
        lnL_of = lambda t: kappa_of_t(t).real - 0.5 * rho_sq
    t = np.linspace(t_grid[0], t_grid[-1], (len(t_grid) - 1) * oversample + 1)
    fine = _log_trapz(t, lnL_of(t))
    coarse = _log_trapz(t[::2], lnL_of(t[::2]))
    return fine, abs(fine - coarse)


def _simpson_log_integral(kappa_of_t, rho_sq, t_grid):
    """What the historical path computes: Simpson at dx=deltaT on the samples."""
    from scipy import integrate
    simps = getattr(integrate, "simpson", None) or integrate.simps
    lnL = kappa_of_t(t_grid).real - 0.5 * rho_sq
    m = lnL.max()
    return m + math.log(simps(np.exp(lnL - m), dx=t_grid[1] - t_grid[0]))


# --------------------------------------------------------------------------
# bandlimited_upsample
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n", [64, 65, 128, 127])
@pytest.mark.parametrize("factor", [2, 4, 8])
def test_upsample_is_exact_on_a_band_limited_signal(n, factor):
    """The whole fix rests on this: the samples already determine the continuum."""
    deltaT = 1.0 / 512
    kap = _synthetic_kappa(n, deltaT, amp=3.0)
    t = np.arange(n) * deltaT
    dense = tq.bandlimited_upsample(kap(t)[None, :], factor)[0]
    t_dense = np.arange(n * factor) * (deltaT / factor)
    assert np.allclose(dense, kap(t_dense), rtol=0, atol=1e-10 * np.abs(kap(t)).max())


@pytest.mark.parametrize("n", [64, 65])
def test_upsample_preserves_the_original_samples_at_stride_factor(n):
    deltaT = 1.0 / 512
    kap = _synthetic_kappa(n, deltaT, amp=3.0)
    x = kap(np.arange(n) * deltaT)
    dense = tq.bandlimited_upsample(x[None, :], 8)[0]
    assert np.allclose(dense[::8], x, rtol=0, atol=1e-12 * np.abs(x).max())


def test_upsample_factor_one_is_a_no_op():
    x = np.arange(10, dtype=complex)[None, :]
    assert tq.bandlimited_upsample(x, 1) is x


def test_upsample_rejects_a_nonpositive_factor():
    with pytest.raises(ValueError):
        tq.bandlimited_upsample(np.zeros((1, 8), dtype=complex), 0)


# --------------------------------------------------------------------------
# peak_width_from_lnL -- the estimator the derived factor rests on
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sigma_over_h", [4.0, 1.0, 0.3, 0.1, 0.03])
@pytest.mark.parametrize("offset", [0.0, 0.17, 0.5, -0.41])
def test_peak_width_is_exact_for_a_gaussian_at_any_resolution_and_phase(
        sigma_over_h, offset):
    """A quadratic's second difference is its second derivative at ANY spacing.

    This is why a peak the grid cannot resolve can still report its own width
    honestly, which is what lets the factor be derived rather than guessed.
    The badly under-resolved cases (0.1, 0.03) are the ones that matter: they
    are the production regime this change exists to fix.
    """
    n, dx = 201, 1.0
    sigma = sigma_over_h * dx
    t = (np.arange(n) - n // 2) * dx
    lnL = -0.5 * ((t - offset * dx) / sigma) ** 2
    sig, _ = tq.peak_width_from_lnL(lnL[None, :], dx)
    assert sig[0] == pytest.approx(sigma, rel=1e-10)


def test_peak_width_is_infinite_for_a_flat_integrand():
    lnL = np.zeros((1, 51))
    sig, _ = tq.peak_width_from_lnL(lnL, 1.0)
    assert not np.isfinite(sig[0])


def test_peak_width_handles_a_peak_at_the_array_edge():
    """argmax at index 0 must not index out of bounds."""
    lnL = -np.arange(31.0)[None, :]
    sig, j = tq.peak_width_from_lnL(lnL, 1.0)
    assert j[0] == 0 and np.isfinite(sig).size == 1


def test_peak_width_rejects_too_few_samples():
    with pytest.raises(ValueError):
        tq.peak_width_from_lnL(np.zeros((1, 2)), 1.0)


# --------------------------------------------------------------------------
# required_upsample_factor -- derived, power of two, and it refuses rather
# than truncating
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sigma_over_h", [8.0, 1.0, 0.25, 0.05, 0.01])
def test_required_factor_meets_the_criterion_and_is_a_power_of_two(sigma_over_h):
    n, dx = 201, 1.0
    sigma = sigma_over_h * dx
    t = (np.arange(n) - n // 2) * dx
    lnL = -0.5 * (t / sigma) ** 2
    factor, sigma_min = tq.required_upsample_factor(lnL[None, :], dx)
    assert factor & (factor - 1) == 0, "factor must be a power of two"
    assert dx / factor <= sigma_min / tq.UPSAMPLE_SAFETY
    # and not wastefully large: one step down must FAIL the criterion
    if factor > 1:
        assert dx / (factor // 2) > sigma_min / tq.UPSAMPLE_SAFETY


def test_required_factor_is_one_when_nothing_needs_resolving():
    factor, sigma = tq.required_upsample_factor(np.zeros((3, 51)), 1.0)
    assert factor == 1 and not np.isfinite(sigma)


def test_required_factor_is_set_by_the_SHARPEST_row():
    n, dx = 201, 1.0
    t = (np.arange(n) - n // 2) * dx
    broad = -0.5 * (t / (4.0 * dx)) ** 2
    sharp = -0.5 * (t / (0.05 * dx)) ** 2
    f_broad, _ = tq.required_upsample_factor(broad[None, :], dx)
    f_both, _ = tq.required_upsample_factor(np.stack([broad, sharp]), dx)
    f_sharp, _ = tq.required_upsample_factor(sharp[None, :], dx)
    assert f_both == f_sharp > f_broad


def test_required_factor_raises_rather_than_truncating():
    """A cap that silently clamps would hand back a knowingly wrong integral."""
    n, dx = 201, 1.0
    t = (np.arange(n) - n // 2) * dx
    lnL = -0.5 * (t / (1.0e-5 * dx)) ** 2
    with pytest.raises(ValueError, match="UPSAMPLE_FACTOR_MAX"):
        tq.required_upsample_factor(lnL[None, :], dx)


# --------------------------------------------------------------------------
# validate_time_quadrature
# --------------------------------------------------------------------------
def test_validate_accepts_the_documented_choices_and_rejects_others():
    for choice in tq.TIME_QUADRATURE_CHOICES:
        assert tq.validate_time_quadrature(choice) == choice
    for bad in ("Simpson", "trapezoid", "bandlimited ", None, 8):
        with pytest.raises(ValueError):
            tq.validate_time_quadrature(bad)


# --------------------------------------------------------------------------
# time_marginalize_bandlimited against a closed-form integrand
# --------------------------------------------------------------------------
@pytest.mark.parametrize("amp", [4.0, 40.0, 400.0])
def test_bandlimited_beats_simpson_against_the_analytic_integral(amp):
    """The headline claim, at three peak sharpnesses.

    `amp` plays the role of rho^2: the larger it is, the narrower exp(lnL(t))
    becomes against the fixed grid, and the worse the historical Simpson rule
    does.  The band-limited path must stay accurate where Simpson does not.
    Tolerance is the REFERENCE's own convergence estimate, not a constant.
    """
    n, deltaT = 128, 1.0 / 4096
    kap = _synthetic_kappa(n, deltaT, amp=amp)
    t = np.arange(n) * deltaT
    rho_sq = amp
    truth, ref_err = _reference_log_integral(kap, rho_sq, t)
    simpson = _simpson_log_integral(kap, rho_sq, t)
    band = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), rho_sq), deltaT, _helper)[0]

    tol = max(10 * ref_err, 1e-9)
    assert abs(band - truth) < tol, (band, truth, ref_err)
    assert abs(band - truth) < abs(simpson - truth)


def test_synthetic_peak_width_matches_the_closed_form_curvature():
    """Guards the fixture: if its peak is not where and as sharp as it claims,
    every tolerance built on it is meaningless."""
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    kap = _synthetic_kappa(n, deltaT, amp=amp)
    t = np.arange(n) * deltaT
    lnL = kap(t).real - 0.5 * amp
    assert t[np.argmax(lnL)] == pytest.approx(kap.t0, abs=deltaT)
    sigma_exact = 1.0 / math.sqrt(-kap.curvature)
    sig, _ = tq.peak_width_from_lnL(lnL[None, :], deltaT)
    assert 0.5 < sig[0] / sigma_exact < 2.0, (sig[0], sigma_exact)


def test_width_is_taken_from_the_sharpest_feature_not_the_tallest_sample():
    """The width used must be the sharpest feature's, not the argmax's.

    On a comb the true peak can fall midway between samples, so the SAMPLED
    maximum need not sit near it.  Measured honestly, the two rules agree on
    every case constructed here -- an A/B over both spectra, three amplitudes,
    25 seeds and five sub-sample offsets found no difference in the resulting
    integral.  So this pins the conservative property (the width used is never
    larger than the argmax reading, hence the grid is never coarser) and the
    outcome that actually matters (the integral is right), without claiming the
    argmax rule was observed to fail.
    """
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    kap = _synthetic_kappa(n, deltaT, amp=amp, spectrum="comb")
    t = np.arange(n) * deltaT
    lnL = kap(t).real - 0.5 * amp
    # the premise: the sampled argmax really is not the true peak here
    assert abs(t[np.argmax(lnL)] - kap.t0) > deltaT

    j = int(np.argmax(lnL))
    d2_at_argmax = (lnL[j - 1] - 2 * lnL[j] + lnL[j + 1]) / deltaT ** 2
    sigma_argmax = 1.0 / math.sqrt(-d2_at_argmax)
    sigma_used, _ = tq.peak_width_from_lnL(lnL[None, :], deltaT)
    assert sigma_used[0] <= sigma_argmax

    truth, ref_err = _reference_log_integral(kap, amp, t)
    band = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, _helper)[0]
    assert abs(band - truth) < max(10 * ref_err, 1e-9), (band, truth, ref_err)


def test_bandlimited_is_insensitive_to_grid_phase_where_simpson_is_not():
    """The defect's signature is that the answer moves when the PEAK moves.

    Sliding the peak by up to two samples about the window centre cannot change
    the continuous integral -- and that is asserted here on the analytic
    reference first, so the test cannot pass by comparing two broken things.
    Simpson's answer moves anyway; the band-limited answer must not.
    """
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    t = np.arange(n) * deltaT
    centre = 0.5 * (n - 1) * deltaT
    ref_vals, simps_vals, band_vals = [], [], []
    for shift in np.linspace(-deltaT, deltaT, 9):
        kap = _synthetic_kappa(n, deltaT, amp=amp, t0=centre + shift)
        ref, _ = _reference_log_integral(kap, amp, t)
        ref_vals.append(ref)
        simps_vals.append(_simpson_log_integral(kap, amp, t))
        band_vals.append(tq.time_marginalize_bandlimited(
            kap(t)[None, :], np.full((1, n), amp), deltaT, _helper)[0])
    span = lambda v: max(v) - min(v)
    assert span(ref_vals) < 1e-6, span(ref_vals)     # the truth IS invariant
    assert span(simps_vals) > 1.0, span(simps_vals)  # the defect is present
    assert span(band_vals) < 1e-6, span(band_vals)   # and the fix removes it


def test_a_resolved_row_comes_back_byte_equal_to_simpson():
    """The property the whole design now rests on: a row's value differs from
    the historical one IF AND ONLY IF its integrand was under-resolved.

    A row the grid already resolves derives factor 1, is not refined, and is
    returned untouched -- not re-integrated with a different rule.
    """
    n, deltaT, amp = 65, 1.0, 0.05
    kap = _synthetic_kappa(n, deltaT, amp=amp)      # deliberately very broad
    t = np.arange(n) * deltaT
    K, R = kap(t)[None, :], np.full((1, n), amp)
    got = tq.time_marginalize_bandlimited(K, R, deltaT, _helper)
    rep = tq.last_report()
    assert rep["n_refined_rows"] == 0
    assert np.array_equal(got, _simpson_rows((K[0].real - 0.5 * amp)[None, :], deltaT))


def test_bandlimited_integrates_the_same_domain_as_simpson():
    """Domain, not just spacing: the dense grid must span [t_0, t_{npts-1}].

    An edge taper would shrink it instead, which is invisible for a sharp
    interior peak and a real error for a broad one.
    """
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    kap = _synthetic_kappa(n, deltaT, amp=amp)      # sharp enough to be refined
    t = np.arange(n) * deltaT
    tq.time_marginalize_bandlimited(kap(t)[None, :], np.full((1, n), amp),
                                    deltaT, _helper)
    rep = tq.last_report()
    assert rep["n_refined_rows"] == 1 and rep["n_wrap_exposed_rows"] == 0
    assert rep["npts_dense"] == (rep["npts"] - 1) * rep["factor"] + 1


# --------------------------------------------------------------------------
# the wrap guard: no row may come out worse than the status quo
# --------------------------------------------------------------------------
def _simpson_rows(lnL_rows, deltaT):
    from scipy import integrate
    simps = getattr(integrate, "simpson", None) or integrate.simps
    m = lnL_rows.max()
    return m + np.log(simps(np.exp(lnL_rows - m), dx=deltaT, axis=-1))


def test_wrap_exposed_rows_get_the_historical_value_bit_for_bit():
    """The FFT interpolant is periodic, so a peak beside the wrap is not merely
    inaccurate but wrong HIGH -- which would importance-weight that sample into
    dominance.  Such rows fall back, exactly, rather than being repaired."""
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    t = np.arange(n) * deltaT
    kap = _synthetic_kappa(n, deltaT, amp=amp, t0=2 * deltaT)   # hard against the edge
    lnL = (kap(t).real - 0.5 * amp)[None, :]
    got = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, _helper)
    assert tq.last_report()["n_wrap_exposed_rows"] == 1
    assert np.array_equal(got, _simpson_rows(lnL, deltaT))


def test_the_guard_uses_the_callers_simpson_and_offset():
    """Bit-for-bit means the CALLER's rule and offset, not a private copy: the
    GPU build integrates with optimized_gpu_tools.simps."""
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    t = np.arange(n) * deltaT
    kap = _synthetic_kappa(n, deltaT, amp=amp, t0=2 * deltaT)
    calls = []

    def spy(x, dx, axis=-1):
        from scipy import integrate
        simps = getattr(integrate, "simpson", None) or integrate.simps
        calls.append(dx)
        return simps(x, dx=dx, axis=axis)

    tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, _helper, simps=spy)
    assert calls == [deltaT]


def test_exposed_rows_do_not_inflate_the_factor_for_healthy_rows():
    """One mis-centred row must not make every healthy row pay for refinement
    it does not need."""
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    t = np.arange(n) * deltaT
    centre = _synthetic_kappa(n, deltaT, amp=30.0)                 # broad, interior
    edge = _synthetic_kappa(n, deltaT, amp=amp, t0=1 * deltaT)     # sharp, exposed
    K = np.stack([centre(t), edge(t)])
    R = np.stack([np.full(n, 30.0), np.full(n, amp)])
    tq.time_marginalize_bandlimited(K, R, deltaT, _helper)
    rep = tq.last_report()
    assert rep["n_wrap_exposed_rows"] == 1
    alone = tq.required_upsample_factor(
        (centre(t).real - 15.0)[None, :], deltaT)[0]
    assert rep["factor"] == alone


def test_the_guard_also_covers_the_no_refinement_path():
    """If every row is exposed there is no dense grid at all, and the rows must
    still come back as SIMPSON -- not as a coarse trapezoid, which would be a
    different rule and measurably worse on an under-resolved peak."""
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    t = np.arange(n) * deltaT
    kaps = [_synthetic_kappa(n, deltaT, amp=amp, t0=j * deltaT) for j in (1, 2)]
    K = np.stack([k(t) for k in kaps])
    R = np.full((2, n), amp)
    lnL = np.stack([k(t).real - 0.5 * amp for k in kaps])
    got = tq.time_marginalize_bandlimited(K, R, deltaT, _helper)
    rep = tq.last_report()
    assert rep["n_wrap_exposed_rows"] == 2 and rep["npts_dense"] == 0
    assert np.array_equal(got, _simpson_rows(lnL, deltaT))


# --------------------------------------------------------------------------
# numerical robustness in the regime the fix exists for
# --------------------------------------------------------------------------
def test_no_overflow_when_the_dense_peak_towers_over_the_coarse_one():
    """Under-resolution MEANS the dense maximum can exceed the coarse one by
    thousands of nats.  Offsetting by the coarse maximum would overflow to +inf
    exactly where this option is supposed to help."""
    n, deltaT = 256, 1.0 / 4096
    amp = 4000.0
    kap = _synthetic_kappa(n, deltaT, amp=amp)
    t = np.arange(n) * deltaT
    coarse_max = (kap(t).real - 0.5 * amp).max()
    got = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, _helper)[0]
    assert np.isfinite(got)
    assert got > coarse_max - 50, (got, coarse_max)


def test_rows_far_below_the_block_maximum_do_not_underflow_to_minus_inf():
    """A single global offset makes every row more than ~745 nats down return
    log(0) = -inf.  Per-row offsets do not."""
    n, deltaT = 128, 1.0 / 4096
    loud = _synthetic_kappa(n, deltaT, amp=2000.0)
    quiet = _synthetic_kappa(n, deltaT, amp=3.0, seed=5)
    t = np.arange(n) * deltaT
    K = np.stack([loud(t), quiet(t)])
    R = np.array([np.full(n, 2000.0), np.full(n, 3.0)])
    got = tq.time_marginalize_bandlimited(K, R, deltaT, _helper)
    assert np.all(np.isfinite(got)), got


def test_minus_inf_in_lnL_does_not_silently_collapse_the_factor():
    """distmarg_loglikelihood returns -inf outside its table, and
    (-inf) - 2(-inf) + (-inf) is NaN.  A plain max() over that is NaN, which
    reads as 'no peak' and would leave the row UNDER-resolved with no warning.
    """
    n, dx = 201, 1.0
    t = (np.arange(n) - n // 2) * dx
    lnL = -0.5 * (t / (0.05 * dx)) ** 2
    lnL[:20] = -np.inf
    lnL[-20:] = -np.inf
    sigma, _ = tq.peak_width_from_lnL(lnL[None, :], dx)
    assert np.isfinite(sigma[0])
    assert sigma[0] == pytest.approx(0.05 * dx, rel=1e-6)


def test_tail_noise_cannot_drive_the_factor():
    """The curvature scan must see only bins that can affect the integral.

    Otherwise numerical noise 300 nats down -- where the integrand is e^-300 --
    sets the resolution for the whole group, and can reach the raising ceiling
    on a row that contributes nothing.
    """
    n, dx = 401, 1.0
    t = (np.arange(n) - n // 2) * dx
    lnL = -0.5 * (t / (2.0 * dx)) ** 2                  # broad, well resolved
    rng = np.random.default_rng(0)
    lnL[:50] += -1.0e3 + 1.0e2 * rng.normal(size=50)    # violent junk in the tail
    factor, _ = tq.required_upsample_factor(lnL[None, :], dx)
    assert factor == 1, factor


def test_bandlimited_refuses_a_time_dependent_rho_sq():
    """The band-limited argument is about kappa(t) only.

    rift_O4c has no banded / slow-rotation path today, which is exactly why
    this guard matters: it is what stops a future backport whose rho_sq DOES
    vary in time from being interpolated as though it did not.
    """
    n = 32
    rho = np.tile(np.arange(n, dtype=float), (2, 1))
    with pytest.raises(NotImplementedError, match="time-independent"):
        tq.time_marginalize_bandlimited(
            np.zeros((2, n), dtype=complex), rho, 1.0, _helper)


def test_bandlimited_accepts_a_nonlinear_loglikelihood_callback():
    """Production runs pass distmarg_loglikelihood, not the affine helper.

    The callback is applied AFTER upsampling, so a nonlinear one is handled
    exactly rather than approximated -- and the resolution is derived from
    lnL(t), which already includes it.  A rule built on kappa's curvature would
    be right only for the affine helper.
    """
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    kap = _synthetic_kappa(n, deltaT, amp=amp)
    t = np.arange(n) * deltaT
    # a deliberately nonlinear, monotone callback
    nl = lambda k, r: np.log1p(np.exp(np.clip(k - 0.5 * r, -700, 700)))
    truth, ref_err = _reference_log_integral(
        kap, amp, t, lnL_of=lambda tt: nl(kap(tt).real, amp))
    band = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, nl)[0]
    assert abs(band - truth) < max(10 * ref_err, 1e-9), (band, truth, ref_err)


def test_bandlimited_handles_phase_marginalization_absolute_value():
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    kap = _synthetic_kappa(n, deltaT, amp=amp)
    t = np.arange(n) * deltaT
    truth, ref_err = _reference_log_integral(
        kap, amp, t, lnL_of=lambda tt: np.abs(kap(tt)) - 0.5 * amp)
    band = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, _helper,
        phase_marginalization=True)[0]
    assert abs(band - truth) < max(10 * ref_err, 1e-9), (band, truth, ref_err)


def test_chunking_cannot_change_the_answer(monkeypatch):
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    kap = _synthetic_kappa(n, deltaT, amp=amp)
    t = np.arange(n) * deltaT
    K = np.repeat(kap(t)[None, :], 9, axis=0)
    R = np.full((9, n), amp)
    whole = tq.time_marginalize_bandlimited(K, R, deltaT, _helper)
    assert tq.last_report()["n_chunks"] == 1
    monkeypatch.setattr(tq, "_DENSE_CHUNK_BYTES", 1)
    chunked = tq.time_marginalize_bandlimited(K, R, deltaT, _helper)
    assert tq.last_report()["n_chunks"] == 9
    assert np.allclose(whole, chunked, rtol=0, atol=1e-12)


def test_report_counts_rows_whose_peak_abuts_the_window_edge():
    """Reported, not repaired: a mis-centred window is a different defect."""
    n = 64
    ramp = -np.arange(n, dtype=float)
    interior = -((np.arange(n) - n // 2) ** 2).astype(float)
    tq.time_marginalize_bandlimited(
        np.zeros((2, n), dtype=complex), np.zeros((2, n)), 1.0, _helper,
        lnL_t=np.stack([ramp, interior]))
    assert tq.last_report()["n_edge_peak_rows"] == 1


def test_per_row_factor_meets_the_criterion_across_the_whole_usable_range():
    """Exhaustive sweep, because this is the one number that decides accuracy.

    The factor comes from `2**ceil(log2(need))`, and ceil/log2 can land one step
    short when `need` sits just under a power of two.  Erring LOW there would
    silently under-resolve, so the implementation re-checks the criterion and
    bumps.  Half a million widths spanning the entire range the ceiling allows,
    plus the no-peak case.
    """
    dx = 1.0 / 4096
    lo = tq.UPSAMPLE_SAFETY * dx / tq.UPSAMPLE_FACTOR_MAX
    sigma = np.concatenate([np.logspace(np.log10(lo * 1.001), -1, 500000),
                            np.full(10, np.inf)])
    factor = tq._factor_per_row(sigma, dx)
    finite = np.isfinite(sigma)
    with np.errstate(divide="ignore"):
        assert np.all((dx / factor <= sigma / tq.UPSAMPLE_SAFETY) | ~finite)
        # never more than one power of two above what is needed, either
        assert not np.any(finite & (factor > 1) &
                          (dx / (factor // 2) <= sigma / tq.UPSAMPLE_SAFETY))
    assert np.all(factor & (factor - 1) == 0)
    assert factor.max() <= tq.UPSAMPLE_FACTOR_MAX
    assert np.all(factor[~finite] == 1)


def test_the_ceiling_is_tested_against_the_factor_not_the_raw_requirement():
    """`need` just above the ceiling rounds UP to twice it, so a check on `need`
    would pass a factor that exceeds the ceiling."""
    dx = 1.0
    just_over = tq.UPSAMPLE_SAFETY * dx / (tq.UPSAMPLE_FACTOR_MAX + 1.0)
    with pytest.raises(ValueError, match="UPSAMPLE_FACTOR_MAX"):
        tq._factor_per_row(np.array([just_over]), dx)
    # and the largest width that IS representable is accepted
    ok = tq.UPSAMPLE_SAFETY * dx / tq.UPSAMPLE_FACTOR_MAX
    assert int(tq._factor_per_row(np.array([ok]), dx)[0]) == tq.UPSAMPLE_FACTOR_MAX


def test_accepts_the_input_shapes_its_contract_implies():
    """1-D (single row), 2-D, and a rho_sq given as one column rather than
    broadcast across time must all give the same answer.

    The 1-D branch exists in the code, so it is either supported or it is dead;
    this makes it the former.  The single-column rho_sq is what a caller would
    naturally pass once it knows the self-term is time-independent.
    """
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    kap = _synthetic_kappa(n, deltaT, amp=amp)
    t = np.arange(n) * deltaT
    one_d = tq.time_marginalize_bandlimited(kap(t), np.full(n, amp), deltaT, _helper)
    two_d = tq.time_marginalize_bandlimited(kap(t)[None, :], np.full((1, n), amp),
                                            deltaT, _helper)
    rho_col = tq.time_marginalize_bandlimited(kap(t)[None, :], np.full((1, 1), amp),
                                              deltaT, _helper)
    assert np.shape(one_d) == () and np.shape(two_d) == (1,)
    assert float(one_d) == float(two_d[0]) == float(rho_col[0])


# --------------------------------------------------------------------------
# Why the wrap guard exists, measured on THIS line rather than cited.
#
# `_synthetic_kappa` is exactly periodic on the window by construction (every
# frequency an exact multiple of 1/(n*deltaT)), which is what makes the DFT
# interpolation exact -- and also means it can NEVER exhibit wrap ringing.  A
# guard test built on it would pass without demonstrating the guard is needed.
# This fixture is band-limited but NOT periodic: incommensurate frequencies
# under a Gaussian envelope, so a peak near one edge leaves the interpolant
# genuinely discontinuous across the wrap.
# --------------------------------------------------------------------------
_NONPERIODIC_ENV = 6.0e-3


def _nonperiodic_kappa(n, deltaT, amp, t0, seed=4, n_modes=6):
    rng = np.random.default_rng(seed)
    f = rng.uniform(200.0, 1500.0, size=n_modes)      # not multiples of 1/(n*dt)
    a = rng.uniform(0.5, 1.5, size=n_modes)
    a = a * (amp / a.sum())

    def kappa_of_t(t):
        t = np.asarray(t, dtype=float)
        d = t.ravel() - t0
        z = (a[:, None] * np.exp(2j * np.pi * f[:, None] * d[None, :])).sum(axis=0)
        return (z * np.exp(-0.5 * (d / _NONPERIODIC_ENV) ** 2)).reshape(t.shape)

    return kappa_of_t


@pytest.mark.parametrize("bins_from_edge,bound", [(64, 1e-6), (40, 1e-6), (32, 1e-2)])
def test_wrap_contamination_is_negligible_OUTSIDE_the_guard_band(
        bins_from_edge, bound, monkeypatch):
    """What justifies EDGE_GUARD_FRACTION: outside it the ringing does not matter.

    Measured with the guard disabled, against the analytic integrand.  The
    guard band is 0.125*256 = 32 bins here, so 32 is the boundary case.
    """
    n, deltaT, amp = 256, 1.0 / 4096, 400.0
    t = np.arange(n) * deltaT
    kap = _nonperiodic_kappa(n, deltaT, amp, t0=bins_from_edge * deltaT)
    truth, ref_err = _reference_log_integral(kap, amp, t)
    monkeypatch.setattr(tq, "EDGE_GUARD_FRACTION", 0.0)
    got = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, _helper)[0]
    assert tq.last_report()["n_wrap_exposed_rows"] == 0
    assert abs(got - truth) < max(bound, 10 * ref_err), (got, truth)


def test_wrap_contamination_is_large_and_HIGH_inside_the_guard_band(monkeypatch):
    """And inside it, it does matter -- and errs in the dangerous direction.

    A spuriously HIGH lnL biases the evidence up and importance-weights that
    sample into dominance, which is why these rows fall back rather than being
    trusted.  (The companion rift_O4d measurement, with a sharper peak, reached
    +88.8 nats; the magnitude depends on the peak, the sign does not.)
    """
    n, deltaT, amp = 256, 1.0 / 4096, 400.0
    t = np.arange(n) * deltaT
    kap = _nonperiodic_kappa(n, deltaT, amp, t0=2 * deltaT)
    truth, _ = _reference_log_integral(kap, amp, t)
    monkeypatch.setattr(tq, "EDGE_GUARD_FRACTION", 0.0)
    unguarded = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, _helper)[0]
    assert unguarded - truth > 0.1, unguarded - truth        # large, and HIGH
    monkeypatch.undo()
    guarded = tq.time_marginalize_bandlimited(
        kap(t)[None, :], np.full((1, n), amp), deltaT, _helper)
    assert tq.last_report()["n_wrap_exposed_rows"] == 1
    assert np.array_equal(guarded,
                          _simpson_rows((kap(t).real - 0.5 * amp)[None, :], deltaT))


def test_a_row_that_is_minus_inf_everywhere_returns_minus_inf_not_nan():
    """Per-row offsets make `-inf - (-inf)` reachable, and NaN is far worse than
    -inf here: -inf means zero weight, NaN propagates through the sampler.

    The distance-marginalization callback returns -inf outside its table, so a
    row entirely outside it is not hypothetical.  The historical global-offset
    path returned -inf for such a row; so must this one.
    """
    n, deltaT = 64, 1.0
    dead = np.full(n, -np.inf)
    live = -0.01 * (np.arange(n) - n // 2).astype(float) ** 2
    lnL = np.stack([dead, live])
    # a callback that reproduces lnL from a zero kappa, so the dense grid is
    # -inf on row 0 as well
    cb = lambda k, r: np.where(np.asarray(r) > 0.5, -np.inf, np.asarray(k).real)
    K = np.zeros((2, n), dtype=complex)
    K[1] = live
    R = np.stack([np.ones(n), np.zeros(n)])
    got = tq.time_marginalize_bandlimited(K, R, deltaT, cb, lnL_t=lnL)
    assert not np.any(np.isnan(got)), got
    assert np.isneginf(got[0])
    assert np.isfinite(got[1])


def test_report_splits_the_unrefined_population():
    """`n_refined_rows` is exactly the set of rows whose value moved.  The rest
    are unrefined for three distinct reasons that a bare factor histogram cannot
    tell apart -- and two of them are "this row was skipped" wearing the costume
    of "this row was cheap".
    """
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    t = np.arange(n) * deltaT
    dead = np.full(n, -np.inf)                                  # unmeasurable
    edge = _synthetic_kappa(n, deltaT, amp=amp, t0=1 * deltaT)  # wrap exposed
    sharp = _synthetic_kappa(n, deltaT, amp=amp)                # refined
    broad = _synthetic_kappa(n, deltaT, amp=0.05, seed=3)       # already resolved
    lnL = np.stack([dead,
                    edge(t).real - 0.5 * amp,
                    sharp(t).real - 0.5 * amp,
                    broad(t).real - 0.5 * 0.05])
    K = np.stack([np.zeros(n, dtype=complex), edge(t), sharp(t), broad(t)])
    R = np.stack([np.zeros(n), np.full(n, amp), np.full(n, amp), np.full(n, 0.05)])
    got = tq.time_marginalize_bandlimited(K, R, deltaT, _helper, lnL_t=lnL)
    rep = tq.last_report()
    assert rep["n_nonfinite_rows"] == 1
    assert rep["n_wrap_exposed_rows"] == 1        # the edge row ONLY
    assert rep["n_refined_rows"] == 1             # the sharp row ONLY
    assert not np.any(np.isnan(got))
    # every unrefined row must be byte-equal to the historical value
    hist = _simpson_rows(lnL, deltaT)
    for k in (0, 1, 3):
        assert got[k] == hist[k] or (np.isneginf(got[k]) and np.isneginf(hist[k]))
    assert got[2] != hist[2]


def test_a_signal_free_row_is_not_reported_as_wrap_exposed():
    """A constant row has argmax 0.  A blanket guard would claim it and report
    "wrap exposed", which in a production log reads as "your window is
    mis-centred" when it means "these samples have no signal"."""
    n, deltaT = 128, 1.0
    lnL = np.full((1, n), 3.0)
    got = tq.time_marginalize_bandlimited(
        np.full((1, n), 3.0 + 0j), np.zeros((1, n)), deltaT, _helper, lnL_t=lnL)
    rep = tq.last_report()
    assert rep["n_wrap_exposed_rows"] == 0
    assert rep["n_flat_rows"] == 1 and rep["n_refined_rows"] == 0
    assert np.array_equal(got, _simpson_rows(lnL, deltaT))


# --------------------------------------------------------------------------
# Written because a mutation sweep showed the suite did NOT catch these.  Each
# corresponds to a mutation that previously survived: the safety machinery was
# present, argued for in comments, and completely unexercised.
# --------------------------------------------------------------------------
def test_the_factor_is_not_one_power_of_two_short_at_a_boundary():
    """`2**ceil(log2(need))` can land one step SHORT when `need` sits a few ulp
    above a power of two, because log2 rounds down to the integer.  Erring low
    silently under-resolves.  A log-spaced sweep never lands there; this does,
    by construction, at every boundary the ceiling allows.
    """
    dx = 1.0 / 4096
    for k in range(0, 12):
        need = np.nextafter(float(2 ** k), np.inf)
        sigma = np.array([tq.UPSAMPLE_SAFETY * dx / need])
        factor = tq._factor_per_row(sigma, dx)
        assert dx / factor[0] <= sigma[0] / tq.UPSAMPLE_SAFETY, (k, need, factor[0])
        assert factor[0] >= need, (k, need, factor[0])


def test_trapezoid_returns_minus_inf_for_an_all_minus_inf_row():
    """Unit test of `_trapezoid_log`, deliberately NOT through the integrator.

    A per-row offset makes `-inf - (-inf)` reachable in principle, and NaN is
    far worse than -inf here (NaN propagates into the sampler weights; -inf is
    just zero weight).  Being honest about reachability: the integrator cannot
    currently get here, because such a row has no resolvable peak, is never
    refined, and takes the historical Simpson path.  This is defence in depth,
    and testing it through the integrator would only re-test that routing.
    """
    got = tq._trapezoid_log(np.full((1, 32), -np.inf), 1.0, np.array([-np.inf]), np)
    assert np.isneginf(got[0]), got
    assert not np.any(np.isnan(got))


def test_a_minus_inf_beside_the_peak_does_not_produce_a_zero_width():
    """`distmarg_loglikelihood` returns -inf outside its table.  A -inf bin next
    to a contributing one makes the three-point second difference -inf, hence an
    apparent curvature of +inf and a measured width of ZERO -- "unresolvable at
    any factor", the opposite of "flat", and it must not be silently mapped onto
    a small factor.
    """
    n, dx = 201, 1.0
    t = (np.arange(n) - n // 2) * dx
    lnL = -0.5 * (t / (0.3 * dx)) ** 2
    lnL[n // 2 - 2] = -np.inf          # table edge right beside the peak
    sigma, _ = tq.peak_width_from_lnL(lnL[None, :], dx)
    assert sigma[0] > 0 and np.isfinite(sigma[0])
    assert sigma[0] == pytest.approx(0.3 * dx, rel=1e-6)     # the TRUE width
    assert int(tq._factor_per_row(sigma, dx)[0]) > 1


def test_a_zero_width_is_refused_rather_than_guessed_at():
    """If a zero width ever does reach the factor rule it must raise.  Folding it
    in with sigma=inf (both are non-finite after the division) would hand back a
    small factor for an unresolvable row -- the exact failure this module exists
    to prevent.
    """
    with pytest.raises(ValueError, match="non-positive measured width"):
        tq._factor_per_row(np.array([0.0]), 1.0)


def test_the_dense_remeasurement_recovers_an_under_derived_factor(monkeypatch):
    """The refinement loop re-measures on the grid it actually integrated and
    doubles until the criterion holds there too.  Force the derivation to
    under-shoot and require the loop to notice and still land on the truth.
    """
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    kap = _synthetic_kappa(n, deltaT, amp=amp)
    t = np.arange(n) * deltaT
    K, R = kap(t)[None, :], np.full((1, n), amp)
    truth, ref_err = _reference_log_integral(kap, amp, t)

    honest = tq.time_marginalize_bandlimited(K, R, deltaT, _helper)[0]
    good_factor = tq.last_report()["factor"]
    assert good_factor >= 8

    real_rule = tq._factor_per_row
    monkeypatch.setattr(
        tq, "_factor_per_row",
        lambda sigma, dx, xpy=np: np.minimum(real_rule(sigma, dx, xpy=xpy), 2))
    got = tq.time_marginalize_bandlimited(K, R, deltaT, _helper)[0]
    rep = tq.last_report()
    assert rep["factor_initial"] == 2, rep          # the derivation under-shot
    assert rep["factor"] >= good_factor, rep        # the re-measurement recovered
    assert abs(got - truth) < max(10 * ref_err, 1e-9), (got, truth)
    assert got == pytest.approx(honest, abs=1e-9)


def test_the_log_offset_is_per_row_not_per_block():
    """Two rows in ONE block, separated by more than ~745 nats.  A single
    block-wide offset underflows the quieter one to exp()=0 and then log(0).

    Identical kappa in both rows, so identical curvature, so the same derived
    factor and therefore the same block -- that is the whole point.  They are
    separated purely by the time-independent self-term, which shifts lnL by
    0.5*rho_sq without touching the peak width.  Separating them by AMPLITUDE
    instead lets them straddle a power-of-two factor boundary and land in
    different blocks, which makes a block-wide offset accidentally per-row --
    exactly how an earlier version of this test passed against a mutant.
    """
    n, deltaT, amp = 256, 1.0 / 4096, 400.0
    kap = _synthetic_kappa(n, deltaT, amp=amp, seed=11)
    t = np.arange(n) * deltaT
    K = np.stack([kap(t), kap(t)])
    R = np.array([np.zeros(n), np.full(n, 3000.0)])
    got = tq.time_marginalize_bandlimited(K, R, deltaT, _helper)
    rep = tq.last_report()
    assert len(rep["factor_histogram"]) == 1, rep["factor_histogram"]   # one block
    assert rep["n_refined_rows"] == 2, rep
    assert got[0] - got[1] == pytest.approx(1500.0, abs=1e-6), got
    assert np.all(np.isfinite(got)), got


def test_the_report_buckets_partition_the_batch():
    """refined + resolved + wrap-exposed + flat + nonfinite == every row.

    Without this, a row can fall through a gap between the counters and be
    invisible in a production log -- which is how "this row was skipped" gets
    mistaken for "this row was cheap".
    """
    n, deltaT, amp = 128, 1.0 / 4096, 200.0
    t = np.arange(n) * deltaT
    rows = [
        np.zeros(n, dtype=complex),                              # nonfinite lnL
        _synthetic_kappa(n, deltaT, amp=amp, t0=1 * deltaT)(t),  # wrap exposed
        _synthetic_kappa(n, deltaT, amp=amp)(t),                 # refined
        _synthetic_kappa(n, deltaT, amp=0.05, seed=3)(t),        # already resolved
        np.full(n, 2.0 + 0j),                                    # flat / no signal
    ]
    lnL = np.stack([np.full(n, -np.inf),
                    rows[1].real - 0.5 * amp,
                    rows[2].real - 0.5 * amp,
                    rows[3].real - 0.5 * 0.05,
                    np.full(n, 2.0)])
    R = np.stack([np.zeros(n), np.full(n, amp), np.full(n, amp),
                  np.full(n, 0.05), np.zeros(n)])
    tq.time_marginalize_bandlimited(np.stack(rows), R, deltaT, _helper, lnL_t=lnL)
    rep = tq.last_report()
    buckets = ("n_refined_rows", "n_resolved_rows", "n_wrap_exposed_rows",
               "n_flat_rows", "n_nonfinite_rows")
    assert sum(rep[k] for k in buckets) == rep["n_rows"] == 5, \
        {k: rep[k] for k in buckets + ("n_rows",)}
    for k in buckets:
        assert rep[k] == 1, (k, {b: rep[b] for b in buckets})


# --------------------------------------------------------------------------
# Written after the companion rift_O4d line found a wrong-answer bug in its own
# spectrum split for ODD n.  This line's split is different and correct, and the
# parametrised exactness test above does catch the broken form -- but only
# because its fixture happens to populate the top bin.  These pin the class
# directly, at the npts production actually produces and with a spectrum that
# fills EVERY bin below Nyquist, so a fixture with a spectral gap at the top
# cannot hide it.
# --------------------------------------------------------------------------
#: npts = int(2*0.075/deltaT) at srate 1024 / 2048 / 4096 / 8192 / 16384.
#: THREE OF FIVE ARE ODD, including 16384 -- the batch-mode driver's default.
PRODUCTION_NPTS = [153, 307, 614, 1228, 2457]


@pytest.mark.parametrize("n", PRODUCTION_NPTS)
@pytest.mark.parametrize("factor", [2, 4, 8])
def test_upsample_is_exact_at_every_production_npts(n, factor):
    """Every DFT bin strictly below Nyquist populated, odd and even n alike.

    Two things hide this class and both are avoided here: the reconstruction is
    exact at the ORIGINAL samples however the spectrum is split, so a
    "reproduces its input" assertion cannot see it; and a fixture whose spectrum
    stops short of Nyquist leaves the one misplaced bin empty.

    Strictly below Nyquist is not a convenience: for even n a component exactly
    at Nyquist is genuinely NOT determined by the samples (it aliases onto its
    own conjugate), so no interpolant can recover it and there is nothing to
    test.  Real rholm data is band-limited to fmax < srate/2, so that bin is
    empty in practice.
    """
    rng = np.random.default_rng(n * 100 + factor)
    kmax = (n - 1) // 2                     # excludes the Nyquist bin when n is even
    ks = np.arange(1, kmax + 1)
    a = rng.uniform(0.5, 1.5, ks.size) + 1j * rng.uniform(-1, 1, ks.size)
    deltaT = 1.0 / 4096
    f = ks / (n * deltaT)

    def kappa_of_t(t):
        t = np.asarray(t, dtype=float)
        return (a[:, None] * np.exp(2j * np.pi * f[:, None] * t[None, :])).sum(axis=0)

    t = np.arange(n) * deltaT
    dense = tq.bandlimited_upsample(kappa_of_t(t)[None, :], factor)[0]
    t_dense = np.arange(n * factor) * (deltaT / factor)
    scale = np.abs(kappa_of_t(t)).max()
    assert np.abs(dense - kappa_of_t(t_dense)).max() < 1e-10 * scale


@pytest.mark.parametrize("sigma", [1e-20, 1e-30, 1e-300])
def test_an_absurdly_sharp_width_raises_rather_than_wrapping_to_one(sigma):
    """The ceiling is tested on the FLOAT factor, before the int64 cast.

    `2**ceil(log2(need))` above 2**63 wraps to a large NEGATIVE int64 on the
    cast, and a `maximum(.., 1)` downstream would turn that into 1 -- so
    "unresolvable at any factor" would come back as "nothing to refine", which
    is the same conflation as the sigma<=0 case and just as silent.
    """
    with pytest.raises(ValueError, match="UPSAMPLE_FACTOR_MAX"):
        tq._factor_per_row(np.array([sigma]), 1.0 / 4096)
