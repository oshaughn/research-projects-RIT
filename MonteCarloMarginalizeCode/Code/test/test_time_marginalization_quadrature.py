"""Matched numerical regressions from rift_O4d 94840482 for the O4c port.
Driver and shipped-likelihood wiring are in separate O4c test files.
"""
import warnings
import numpy as np
import pytest
from scipy import integrate
from RIFT.likelihood import time_marginalization_quadrature as tmq
simpson = getattr(integrate, 'simpson', None) or integrate.simps
SRATE = 4096.0
DELTAT = 1.0 / SRATE
NPTS = 614
RHO_SQ = 1000.0

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

    def at(self, ts, chunk=1000):
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


def test_piecewise_linear_draw_has_no_output_lattice_and_handles_flat_density():
    lnL = np.zeros((2, 3))
    uniforms = np.array([[0.25, 0.5], [0.75, 0.2]])
    times, at_draw = tmq.draw_piecewise_linear_log_posterior(
        lnL, 1.0, t0=-1.0, uniforms=uniforms)
    # Equal interval masses: the first uniforms choose intervals 0 and 1.  A
    # flat density has the exact uniform conditional limit inside each interval.
    np.testing.assert_allclose(times, [-0.5, 0.2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(at_draw, [0.0, 0.0], rtol=0, atol=0)
    phase = (times + 1.0) / 1.0
    assert np.all(phase != np.round(phase))


def test_piecewise_linear_draw_inverts_the_density_not_log_density():
    # One interval with density rising linearly from 1 to 3.  At conditional
    # quantile r=1/4, p(u)^2 = 1 + r*(9-1) = 3.
    lnL = np.log(np.array([[1.0, 3.0]]))
    times, at_draw = tmq.draw_piecewise_linear_log_posterior(
        lnL, 2.0, t0=4.0, uniforms=np.array([[0.0, 0.25]]))
    frac = (np.sqrt(3.0) - 1.0) / 2.0
    assert times[0] == pytest.approx(4.0 + 2.0 * frac, abs=2e-15)
    assert at_draw[0] == pytest.approx(0.5 * np.log(3.0), abs=2e-15)


def test_bandlimited_draw_uses_the_validated_export_representation():
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    k = sig.samples()[None, :]
    r = np.full(k.shape, RHO_SQ)
    uniforms = np.array([[0.3712345, 0.6180339]])

    integral, time_draw, lnL_draw = tmq.time_marginalize_bandlimited(
        k, r, DELTAT, _lnL, return_time_draw=True,
        draw_uniforms=uniforms, t0=-0.075)
    draw_report = tmq.last_report()
    integral_only = tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL)
    np.testing.assert_allclose(integral, integral_only, rtol=0, atol=0)

    factor = max(draw_report['export_factor_histogram'])
    assert factor > 1
    dense_k = tmq.reflected_bandlimited_upsample(k, factor)
    dense_lnL = _lnL(dense_k.real, RHO_SQ)
    dx = DELTAT / factor
    phase = (float(time_draw[0]) + 0.075) / dx
    j = int(np.floor(phase))
    frac = phase - j
    density = np.exp(dense_lnL[0] - np.max(dense_lnL[0]))
    density_at_draw = density[j] + frac * (density[j + 1] - density[j])
    expected_lnL = np.max(dense_lnL[0]) + np.log(density_at_draw)
    assert float(lnL_draw[0]) == pytest.approx(expected_lnL, abs=2e-10)
    assert abs(phase - round(phase)) > 1e-6, "draw snapped to the dense FFT grid"


def test_continuous_draw_accepts_minus_infinity_nodes_but_not_invalid_rows():
    times, lnL = tmq.draw_piecewise_linear_log_posterior(
        np.array([[0.0, -np.inf, -1.0]]), 1.0,
        uniforms=np.array([[0.1, 0.4]]))
    assert np.isfinite(times[0]) and np.isfinite(lnL[0])
    # The exact RNG endpoint must skip a leading zero-mass plateau instead of
    # selecting it through searchsorted's equality convention.
    times, lnL = tmq.draw_piecewise_linear_log_posterior(
        np.array([[-np.inf, -np.inf, 0.0]]), 1.0,
        uniforms=np.array([[0.0, 0.0]]))
    assert 1.0 < times[0] < 2.0
    assert np.isfinite(lnL[0])
    with pytest.raises(ValueError, match="no finite positive mass"):
        tmq.draw_piecewise_linear_log_posterior(
            np.full((1, 3), -np.inf), 1.0,
            uniforms=np.array([[0.1, 0.4]]))
    with pytest.raises(ValueError, match="NaN or \\+inf"):
        tmq.draw_piecewise_linear_log_posterior(
            np.array([[0.0, np.nan, -1.0]]), 1.0,
            uniforms=np.array([[0.1, 0.4]]))


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


def test_reflected_upsample_reproduces_the_finite_row_exactly():
    """The non-periodic boundary construction must not move supplied samples."""
    sig = BandLimited(amp=1.0, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    k = sig.samples()
    factor = 8
    up = tmq.reflected_bandlimited_upsample(k[None, :], factor)[0]
    assert up.size == (NPTS - 1) * factor + 1
    assert np.allclose(up[::factor], k, atol=1e-11, rtol=0)


@pytest.mark.parametrize("n,factor", [(614, 64), (1228, 32), (2457, 16)])
def test_retained_fft_is_the_same_reflected_sinc_interpolant(n, factor):
    """Pruning outputs must not change the reconstruction being evaluated.

    Random complex rows populate every bin, including Nyquist, so this compares
    the half-bin convention too.  The production 22 and higher-mode window sizes
    are explicit rather than hidden behind a toy power-of-two transform.
    """
    rng = np.random.default_rng(90210 + n)
    rows = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
    reference = tmq.reflected_bandlimited_upsample(rows, factor)
    retained = tmq._reflected_bandlimited_upsample_retained(rows, factor)
    assert retained.shape == reference.shape == (2, (n - 1) * factor + 1)
    np.testing.assert_allclose(retained, reference, rtol=0, atol=2e-11)
    np.testing.assert_allclose(retained[..., ::factor], rows, rtol=0, atol=2e-11)


def test_retained_fft_removes_the_production_padding_mismatch():
    period, factor = 2 * NPTS, 64
    plan = tmq._retained_fft_plan(period, factor, np.complex128)
    assert plan['n_out'] == (NPTS - 1) * factor + 1 == 39233
    assert period * factor == 78592
    # The optimized convolution is close to the retained half, not the discarded
    # full reflected period.  Do not pin scipy's exact next-fast-length policy.
    assert plan['n_out'] <= plan['n_fft'] < 0.53 * period * factor


def test_transform_decline_retries_full_sinc_and_reports_provenance(monkeypatch):
    """An optimization decline is not a failed waveform point.

    Warning-as-error is included because a diagnostic warning must not undo the
    successful reference-path retry in production environments with strict
    warning filters.
    """
    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    k = sig.samples()[None, :]
    rho = np.full(k.shape, RHO_SQ)
    retained = tmq.time_marginalize_bandlimited(k, rho, DELTAT, _lnL)
    assert tmq.last_report()['bandlimited_fft_strategy'] == 'retained-grid-zoomfft'

    def decline(*args, **kwargs):
        raise tmq._RetainedFFTUnsupported('forced unsupported transform')

    monkeypatch.setattr(tmq, '_reflected_bandlimited_upsample_retained', decline)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fallback = tmq.time_marginalize_bandlimited(k, rho, DELTAT, _lnL)
    report = tmq.last_report()
    assert np.isfinite(float(fallback[0]))
    np.testing.assert_allclose(fallback, retained, rtol=0, atol=1e-9)
    assert report['bandlimited_fft_strategy'] == 'full-padding-fallback'
    assert report['retained_fft_batches'] == 0
    assert report['full_fft_fallback_batches'] >= 1
    assert report['full_fft_fallback_rows'] >= 1
    assert report['full_fft_fallback_reasons'] == {
        '_RetainedFFTUnsupported: forced unsupported transform':
        report['full_fft_fallback_rows']}


def test_unsupported_factor_falls_back_to_full_sinc_without_changing_values():
    rng = np.random.default_rng(19)
    rows = rng.normal(size=(2, 17)) + 1j * rng.normal(size=(2, 17))
    report = tmq._new_transform_report()
    with pytest.warns(RuntimeWarning, match='full-padding sinc reconstruction'):
        got = tmq._reflected_upsample_for_integration(
            rows, 3, {}, report, xpy=np)
    reference = tmq.reflected_bandlimited_upsample(rows, 3)
    np.testing.assert_array_equal(got, reference)
    assert report['retained_fft_batches'] == 0
    assert report['full_fft_fallback_batches'] == 1
    assert '_RetainedFFTUnsupported' in next(iter(
        report['full_fft_fallback_reasons']))


def test_low_factor_numpy_reference_is_selected_not_mislabeled_as_failure():
    rng = np.random.default_rng(23)
    rows = rng.normal(size=(2, 17)) + 1j * rng.normal(size=(2, 17))
    report = tmq._new_transform_report()
    got = tmq._reflected_upsample_for_integration(
        rows, 4, {}, report, xpy=np)
    reference = tmq.reflected_bandlimited_upsample(rows, 4)
    np.testing.assert_array_equal(got, reference)
    assert report['full_fft_selected_batches'] == 1
    assert report['full_fft_selected_rows'] == 2
    assert report['full_fft_fallback_batches'] == 0
    assert report['full_fft_fallback_reasons'] == {}
    assert 'measured retained-FFT crossover' in next(iter(
        report['full_fft_selected_reasons']))


def test_likelihood_failure_is_not_mislabeled_as_transform_fallback(monkeypatch):
    """Only the transform is guarded; callback failures retain their identity."""
    class LikelihoodFailure(RuntimeError):
        pass

    sig = BandLimited(amp=0.17, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    k = sig.samples()[None, :]
    rho = np.full(k.shape, RHO_SQ)

    def fail_on_dense(kappa_term, rho_sq):
        if kappa_term.shape[-1] == NPTS:
            return _lnL(kappa_term, rho_sq)
        raise LikelihoodFailure('callback, not FFT')

    def fallback_must_not_run(*args, **kwargs):
        pytest.fail('a likelihood exception was incorrectly retried as an FFT decline')

    monkeypatch.setattr(tmq, 'reflected_bandlimited_upsample', fallback_must_not_run)
    with pytest.raises(LikelihoodFailure, match='callback, not FFT'):
        tmq.time_marginalize_bandlimited(k, rho, DELTAT, fail_on_dense)


def test_forward_backward_reflection_blocks_the_endpoint_gibbs_counterexample():
    """A decayed integrand does not imply a periodic kappa slice.

    This centred row has negligible edge likelihood but a large coherent
    endpoint mismatch.  Periodizing the raw slice biases the integral by more
    than 100 nats; the literal 2N reflection must stay sub-millinat.  This pins
    the adversarial case that invalidated the tail-decay guard.
    """
    sig = BandLimited(amp=1.0, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    tone_amp, tone_mode, tone_phase = 930.745, 620, 2.3532
    n_period = 8 * NPTS
    js = sig.j0 + np.arange(NPTS)
    tone = lambda j: tone_amp * np.exp(1j * (2 * np.pi * tone_mode * j / n_period
                                             + tone_phase))
    k = sig.samples() + tone(js)
    lnL = _lnL(k.real, RHO_SQ)
    guard = int(NPTS * tmq.EDGE_GUARD_FRACTION)
    assert guard < np.argmax(lnL) < NPTS - 1 - guard
    assert lnL.max() - lnL[:guard].max() > 30
    assert lnL.max() - lnL[-guard:].max() > 30
    assert abs(k[0] - k[-1]) > 1000

    refine = 128
    jd = sig.j0 + np.arange((NPTS - 1) * refine + 1) / float(refine)
    exact = sig.at(jd * DELTAT, chunk=4000) + tone(jd)
    ref = _log_trapz(_lnL(exact.real, RHO_SQ), DELTAT / refine)
    sigma, _, _ = tmq.peak_width_from_lnL(lnL[None, :], DELTAT)
    factor = int(tmq.required_upsample_factors(sigma, DELTAT)[0])

    raw = tmq.bandlimited_upsample(k[None, :], factor)[0]
    raw_value = _log_trapz(_lnL(raw[:(NPTS - 1) * factor + 1].real, RHO_SQ),
                           DELTAT / factor)
    reflected = tmq.reflected_bandlimited_upsample(k[None, :], factor)[0]
    reflected_value = _log_trapz(_lnL(reflected.real, RHO_SQ), DELTAT / factor)
    assert raw_value - ref > 100
    assert abs(reflected_value - ref) < 1e-3


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


@pytest.mark.parametrize("amp,phase", [(a, p) for a in (0.3, 0.5, 1.0, 5.0)
                                       for p in (0.0, 0.25, 0.5)])
def test_exact_on_a_periodic_window(amp, phase):
    """Exactly-periodic window: interpolation is exact, so the band-limited value
    must match the analytic truth to well below any level Simpson achieves."""
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2 + phase)
    ref = sig.truth()
    k = sig.samples()
    assert abs(_bandlimited(k) - ref) < 1e-6


@pytest.mark.parametrize("amp,phase", [(a, p) for a in (0.05, 0.17, 1.0, 5.0)
                                       for p in (0.0, 0.25, 0.5)])
def test_accurate_on_a_non_periodic_window(amp, phase):
    """The realistic case: the window is a segment of a longer band-limited
    signal, so a raw periodic interpolant rings at the wrap.  With the peak
    centred, the reflected residual must still be far below Simpson's error."""
    sig = BandLimited(amp=amp, peak_sample=NPTS // 2 + phase,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    ref = sig.truth()
    k = sig.samples()
    assert abs(_bandlimited(k) - ref) < 1e-3


def test_beats_simpson_where_the_peak_is_under_resolved():
    """The defect itself.  Sweeping the peak across one sample must move the
    Simpson answer by of order a nat while leaving the band-limited answer put --
    that grid-phase sensitivity IS the bug, and insensitivity to it is the fix."""
    sig0 = BandLimited(amp=0.05, peak_sample=NPTS // 2,
                       n_period=8 * NPTS, m_hi=1400, background=0.12)
    sigma, _, _ = tmq.peak_width_from_lnL(_lnL(sig0.samples().real, RHO_SQ)[None, :], DELTAT)
    assert 0.08 < float(sigma[0]) / DELTAT < 0.30, "not the under-resolved regime"

    s_err, b_err = [], []
    for phase in (0.0, 0.25, 0.5, 0.75):
        sig = BandLimited(amp=0.05, peak_sample=NPTS // 2 + phase,
                          n_period=8 * NPTS, m_hi=1400, background=0.12)
        ref = sig.truth()
        k = sig.samples()
        s_err.append(_simpson_value(k) - ref)
        b_err.append(_bandlimited(k) - ref)

    assert max(s_err) - min(s_err) > 0.5, s_err       # Simpson swings by ~2 nats
    assert max(np.abs(b_err)) < 1e-3, b_err
    assert max(np.abs(b_err)) < 0.01 * max(np.abs(s_err))


def test_centered_row_does_not_switch_rules_at_a_tail_threshold():
    """Tail height does not select a Simpson fallback after reflection."""
    sig = BandLimited(amp=0.02, peak_sample=NPTS // 2 + 0.25,
                      n_period=8 * NPTS, m_hi=1400, background=0.12)
    k = sig.samples()
    lnL = _lnL(k.real, RHO_SQ)
    guard = max(1, int(NPTS * tmq.EDGE_GUARD_FRACTION))
    jmax = int(np.argmax(lnL))
    assert guard <= jmax <= NPTS - 1 - guard
    assert lnL.max() - lnL[:guard].max() < 30
    assert lnL.max() - lnL[-guard:].max() < 30

    got = _bandlimited(k)
    assert got != _simpson_value(k)
    assert abs(got - sig.truth()) < 1e-3
    rep = tmq.last_report()
    assert rep['n_refined_rows'] == 1, rep


def test_boundary_diagnostic_does_not_select_simpson():
    """Crossing the diagnostic boundary must not change quadrature rules."""
    for peak in (0.3, 2.3, 30.3):
        sig = BandLimited(amp=1.0, peak_sample=peak, n_period=8 * NPTS,
                          m_hi=1400, background=0.12)
        k = sig.samples()
        assert _bandlimited(k) != _simpson_value(k), peak
        assert tmq.last_report()['n_wrap_exposed_rows'] == 1
        assert tmq.last_report()['n_refined_rows'] == 1


def test_one_exposed_row_does_not_contaminate_its_block():
    """The boundary diagnostic is per row and does not alter reconstruction."""
    bad = BandLimited(amp=1.0, peak_sample=1.3, n_period=8 * NPTS,
                      m_hi=1400, background=0.12)
    good = BandLimited(amp=1.0, peak_sample=NPTS // 2 + 0.25, n_period=8 * NPTS,
                       m_hi=1400, background=0.12, seed=11)
    k = np.stack([bad.samples(), good.samples()])
    out = tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    assert tmq.last_report()['n_wrap_exposed_rows'] == 1
    assert tmq.last_report()['n_refined_rows'] == 2
    assert float(out[0]) != _simpson_value(bad.samples())
    assert abs(float(out[1]) - good.truth()) < 1e-3


def test_rows_sharing_a_block_keep_their_individual_resolution():
    """Each row still derives and pays for its own reconstruction factor."""
    flat = BandLimited(amp=0.0012, peak_sample=NPTS // 2 + 0.3, n_period=8 * NPTS,
                       m_hi=1400, background=0.12, seed=11)
    sharp = BandLimited(amp=5.0, peak_sample=NPTS // 2 + 0.3, n_period=8 * NPTS,
                        m_hi=1400, background=0.12)
    k = np.stack([flat.samples(), sharp.samples()])
    out = tmq.time_marginalize_bandlimited(k, np.full(k.shape, RHO_SQ), DELTAT, _lnL)
    rep = tmq.last_report()
    hist = rep['factor_histogram']
    assert rep['upsample_factor'] > 8
    assert rep['n_refined_rows'] == 2, rep
    assert float(out[0]) != _simpson_value(flat.samples())
    assert abs(float(out[0]) - flat.truth()) < 1e-4
    assert sum(hist.values()) == 2, hist


def test_simpson_fallback_is_evaluated_only_for_unrefined_rows():
    """Dense rows must not also pay for a coarse integration that is discarded."""
    sharp = BandLimited(amp=1.0, peak_sample=NPTS // 2 + 0.25,
                        n_period=8 * NPTS, m_hi=1400, background=0.12)
    flat = np.full(NPTS, 0.12 + 0.0j)
    k = np.stack((sharp.samples(), flat))
    calls = []

    def recording_simps(y, dx, axis):
        calls.append(np.asarray(y).shape)
        return simpson(y, dx=dx, axis=axis)

    got = tmq.time_marginalize_bandlimited(
        k, np.full(k.shape, RHO_SQ), DELTAT, _lnL, simps=recording_simps)
    assert got.shape == (2,)
    assert tmq.last_report()['n_refined_rows'] == 1
    assert calls == [(1, NPTS)], calls


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
    assert (rep['n_refined_rows']
            + rep['n_unmeasurable_rows']
            + rep['n_flat_rows']
            + _n_resolved(rep)) == rep['n_rows'], rep
    # zero likelihood over the whole window integrates to zero: the answer is
    # -inf, which is what the historical global-offset path returns.  NaN here
    # would propagate into the sampler weights.
    assert float(out[0]) == -np.inf, out[0]
    assert abs(float(out[1]) - sig.truth()) < 1e-6


def test_a_row_changes_if_and_only_if_it_was_under_resolved():
    """The guarantee, stated so it can be checked rather than argued.

    Every row that is NOT refined -- unmeasurable, flat, or already resolved --
    must come back with the historical Simpson value, so
    enabling this option cannot make any row worse than the status quo.  Letting
    an unrefined row fall through to a coarse trapezoid instead is numerically a
    non-event, but it changes the rule for rows this option was never meant to
    touch and forfeits exactly this property.
    """
    rows, expect_refined = [], []
    # resolved (no refinement warranted)
    rows.append(BandLimited(amp=0.002, peak_sample=NPTS // 2).samples()); expect_refined.append(False)
    # signal-free
    rows.append(np.zeros(NPTS, dtype=complex)); expect_refined.append(False)
    # boundary-diagnostic row: still refined
    rows.append(BandLimited(amp=1.0, peak_sample=2.3, n_period=8 * NPTS,
                            m_hi=1400, background=0.12).samples()); expect_refined.append(True)
    # centred and sharp with non-negligible coarse tails: still refined
    rows.append(BandLimited(amp=0.02, peak_sample=NPTS // 2 + 0.25,
                            n_period=8 * NPTS, m_hi=1400,
                            background=0.12).samples()); expect_refined.append(True)
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


def test_dense_remeasurement_refines_only_the_rows_that_still_need_it():
    """One pathological row must not impose its extra FFT octaves on a group."""
    signals = [
        BandLimited(amp=0.2, peak_sample=NPTS // 2 + 0.25),
        BandLimited(amp=5.0, peak_sample=NPTS // 2 + 0.25),
    ]
    k = np.stack([sig.samples() for sig in signals])
    rho = np.full(k.shape, RHO_SQ)
    honest = tmq.time_marginalize_bandlimited(k, rho, DELTAT, _lnL)

    real = tmq.required_upsample_factors
    tmq.required_upsample_factors = lambda sigma, dx, xpy=np: 2 * xpy.ones(
        np.asarray(sigma).shape, dtype=np.int64)
    try:
        got = tmq.time_marginalize_bandlimited(k, rho, DELTAT, _lnL)
    finally:
        tmq.required_upsample_factors = real
    rep = tmq.last_report()
    assert rep['n_refinements'] > 0, rep
    assert len(rep['factor_histogram']) == 2, rep
    assert sum(rep['factor_histogram'].values()) == 2, rep
    assert np.allclose(got, honest, rtol=0, atol=1e-9), (got, honest, rep)


def _row_factors(k, r):
    """Per-row derived factor, as the integrator computes it."""
    lnL = _lnL(np.asarray(k).real, np.asarray(r))
    sigma, _, meas = tmq.peak_width_from_lnL(lnL, DELTAT)
    ok = meas & np.isfinite(sigma)
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
    return (rep['n_rows'] - rep['n_refined_rows']
            - rep['n_unmeasurable_rows'] - rep['n_flat_rows'])


def test_the_boundary_diagnostic_covers_the_RIGHT_edge_too():
    """Both physical integration boundaries are reported, not rule switches."""
    for peak in (NPTS - 1.3, NPTS - 3.3, NPTS - 31.3):
        sig = BandLimited(amp=1.0, peak_sample=peak, n_period=8 * NPTS,
                          m_hi=1400, background=0.12)
        k = sig.samples()
        assert _bandlimited(k) != _simpson_value(k), peak
        assert tmq.last_report()['n_wrap_exposed_rows'] == 1, peak
        assert tmq.last_report()['n_refined_rows'] == 1, peak
    # A central peak is reconstructed by the same rule without the diagnostic.
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
    # NOT bit-identity.  The batch shape reaches numpy's FFT and its pairwise
    # summation, so a differently-chunked run reassociates and can differ in the
    # last bit or two -- the companion peak-local implementation measured 0, 0
    # and 2 ULPs on the equivalent test.  This fixture happens to come out
    # bit-identical, which is exactly why asserting it would be a latent flake:
    # it would pass here and fail on a different row count or chunk boundary.
    # Assert what is actually guaranteed, at a bound far below anything that
    # could hide a real assembly bug (dropping a chunk moves rows by nats).
    tol = 64 * np.spacing(np.abs(whole).max())
    assert np.allclose(chunked, whole, rtol=0, atol=tol), np.abs(chunked - whole).max()


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
    sig = BandLimited(amp=0.3, peak_sample=NPTS // 2 + 0.25)
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
        # All four rows are sharp enough to be refined; the boundary changes
        # only the diagnostic count, never the quadrature rule.
        assert rep['n_refined_rows'] == 1, (j, rep)

    # A peak on the very first or last SAMPLE is a documented corner: the
    # curvature stencil is clipped inward and initially measures positive
    # curvature away from the maximum.  A strongly varying row must not be
    # confused with a genuinely flat antenna null; it receives a seed factor and
    # lets dense remeasurement derive the eventual resolution.
    for j in (0, NPTS - 1):
        k = row_peaking_at(j)[None, :]
        r = np.full(k.shape, RHO_SQ)
        out = tmq.time_marginalize_bandlimited(k, r, DELTAT, _lnL)
        rep = tmq.last_report()
        assert rep['n_refined_rows'] == 1, (j, rep)
        assert rep['n_wrap_exposed_rows'] == 1, (j, rep)
        assert rep['n_flat_rows'] == 0, (j, rep)
        assert rep['upsample_factor'] >= 16, (j, rep)
        assert float(out[0]) != _simpson_value(k[0]), j


def test_export_minimum_rate_preserves_integrals_and_refines_resolved_rows():
    # A flat row and a broad but varying row normally remain Simpson fallbacks.
    n = 33
    dt = 0.01
    x = np.arange(n) * dt
    k = np.stack([np.zeros(n), -0.1 * (x - x.mean())**2]).astype(complex)
    r = np.zeros_like(k.real)
    uniforms = np.array([[0.173, 0.617], [0.619, 0.231]])
    base = tmq.time_marginalize_bandlimited(k, r, dt, _lnL)
    out, times, lnL = tmq.time_marginalize_bandlimited(
        k, r, dt, _lnL, return_time_draw=True, draw_uniforms=uniforms,
        time_draw_minimum_srate=350.0, t0=-0.16)
    np.testing.assert_array_equal(out, base)
    report = tmq.last_report()
    assert report['n_refined_rows'] == 0
    assert report['export_minimum_factor'] == 4
    assert report['export_factor_histogram'] == {4: 2}
    dense = tmq.reflected_bandlimited_upsample(k, 4)
    expected_t, expected_l = tmq.draw_piecewise_linear_log_posterior(
        _lnL(dense.real, 0), dt / 4, t0=-0.16, uniforms=uniforms)
    np.testing.assert_allclose(times, expected_t, atol=1e-14, rtol=0)
    np.testing.assert_allclose(lnL, expected_l, atol=1e-14, rtol=0)
    # A minimum export rate cannot affect integral-only evaluation, even when
    # that irrelevant rate exceeds the supported ceiling.
    ignored = tmq.time_marginalize_bandlimited(
        k, r, dt, _lnL, time_draw_minimum_srate=1e100)
    np.testing.assert_array_equal(ignored, base)


@pytest.mark.parametrize('rate', [0, -1, np.nan, np.inf])
def test_export_minimum_rate_refuses_invalid_rates(rate):
    k = np.zeros((1, 5), dtype=complex)
    with pytest.raises(ValueError, match='finite and positive'):
        tmq.time_marginalize_bandlimited(
            k, k.real, 0.01, _lnL, return_time_draw=True,
            time_draw_minimum_srate=rate)


def test_export_minimum_rate_obeys_refinement_ceiling():
    k = np.zeros((1, 5), dtype=complex)
    with pytest.raises(RuntimeError, match='UPSAMPLE_FACTOR_MAX'):
        tmq.time_marginalize_bandlimited(
            k, k.real, 0.01, _lnL, return_time_draw=True,
            time_draw_minimum_srate=(tmq.UPSAMPLE_FACTOR_MAX + 1) / 0.01)


def test_export_width_safety_preserves_integral_and_resolves_point_density():
    # Exact reflected Fourier mode, with narrow peaks at the window ends. Unlike a
    # Gaussian cut out of a finite window this primitive has no boundary seam.
    from scipy.integrate import cumulative_trapezoid
    n, dt, amp = 129, 0.01, 500.0
    x = np.arange(n) * dt
    primitive = amp * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)
    k = primitive[None, :].astype(complex)
    r = np.zeros_like(k.real)
    integral = tmq.time_marginalize_bandlimited(k, r, dt, _lnL)
    baseline_many = tmq.time_marginalize_bandlimited(
        np.repeat(k, 5, axis=0), np.repeat(r, 5, axis=0), dt, _lnL)
    probabilities = np.array([0.005, 0.05, 0.5, 0.95, 0.995])
    sigma = tmq.peak_width_from_lnL(k.real, dt)[0]
    factor = int(tmq.required_upsample_factors(
        sigma, dt, safety=tmq.EXPORT_UPSAMPLE_SAFETY)[0])
    dense = tmq.reflected_bandlimited_upsample(k, factor).real[0]
    weights = np.exp(dense - dense.max())
    cdf = cumulative_trapezoid(weights, dx=dt/factor, initial=0)
    cdf /= cdf[-1]
    bins = np.minimum(np.searchsorted(cdf, probabilities, side='right')-1,
                      len(cdf)-2)
    second = (probabilities-cdf[bins])/(cdf[bins+1]-cdf[bins])
    uniforms = np.column_stack([probabilities, second])
    out, draws, lnL = tmq.time_marginalize_bandlimited(
        np.repeat(k, 5, axis=0), np.repeat(r, 5, axis=0), dt, _lnL,
        return_time_draw=True, draw_uniforms=uniforms)
    np.testing.assert_array_equal(out, baseline_many)
    truth = amp * np.cos(2*np.pi*(draws/dt+0.5)/n)
    assert np.max(abs(lnL-truth)) < 0.01
    fine_x = np.linspace(x[0], x[-1], 65537)
    fine_truth = amp * np.cos(2*np.pi*(fine_x/dt+0.5)/n)
    fine_cdf = cumulative_trapezoid(np.exp(fine_truth-amp), fine_x, initial=0)
    fine_cdf /= fine_cdf[-1]
    assert np.max(abs(np.interp(draws, fine_x, fine_cdf)-probabilities)) < 2e-4
    report = tmq.last_report()
    assert report['export_width_safety'] == 16.0
    assert report['export_factor_histogram'] == {factor: 5}
    assert dt/factor <= report['export_sigma_t_min']/16.0
    # Raising the export-only rate floor cannot perturb any integral bytes.
    higher, _, _ = tmq.time_marginalize_bandlimited(
        k, r, dt, _lnL, return_time_draw=True,
        draw_uniforms=uniforms[:1], time_draw_minimum_srate=2*factor/dt)
    np.testing.assert_array_equal(higher, integral)
    assert tmq.UPSAMPLE_SAFETY == 2.0


def test_export_width_remeasures_and_doubles_without_changing_integral(monkeypatch):
    n, dt = 33, 0.01
    k = (-0.1*(np.arange(n)-16)**2)[None, :].astype(complex)
    r = np.zeros_like(k.real)
    baseline = tmq.time_marginalize_bandlimited(k, r, dt, _lnL)
    original = tmq.peak_width_from_lnL
    calls = []

    def optimistic_coarse(values, spacing, xpy=np):
        sigma, jmax, measurable = original(values, spacing, xpy=xpy)
        calls.append(spacing)
        if spacing == dt:
            sigma = np.full_like(sigma, 0.16)  # Seed factor 1 optimistically.
        elif spacing == dt/2:
            sigma = np.full_like(sigma, 0.04)  # Requires another doubling.
        return sigma, jmax, measurable

    monkeypatch.setattr(tmq, 'peak_width_from_lnL', optimistic_coarse)
    out, _, _ = tmq.time_marginalize_bandlimited(
        k, r, dt, _lnL, return_time_draw=True,
        draw_uniforms=np.array([[0.5, 0.5]]),
        time_draw_minimum_srate=2/dt)
    np.testing.assert_array_equal(out, baseline)
    assert dt/2 in calls and dt/4 in calls
    assert min(tmq.last_report()['export_factor_histogram']) >= 4
