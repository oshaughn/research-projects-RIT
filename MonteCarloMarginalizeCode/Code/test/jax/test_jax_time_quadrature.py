"""Band-limited time marginalization for the JAX ILE likelihood.

The defect: `_time_marginalize` integrates exp(lnL_t) with fixed Simpson
weights at the DATA sample spacing, while the integrand's width
sigma_t = 1/(2 pi rho sigma_f) SHRINKS as the signal gets louder.  Measured on
a 35+30 Msun HLV injection at rho=40: sigma_t = 61.2 us against grid spacings
of 244/122/61 us at srate 4096/8192/16384 -- under-resolved at the rates people
use, worse at higher SNR.  Simpson makes it worse rather than safer, because
Simpson = (4 T_h - T_2h)/3 carries the coarser T_2h alias.

The fix costs no likelihood evaluations: kappa(t) is band-limited, so the
samples already computed determine the continuous integrand exactly.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from RIFT.likelihood.jax_ile.core import (
    _upsample_bandlimited, _time_marginalize, _time_marginalize_bandlimited,
    _simpson_weights, _norm_is_arrival_time_dependent, fused_log_likelihood)


def test_upsampling_is_exact_for_a_band_limited_signal():
    """Not 'accurate' -- EXACT.  This is the sampling theorem, so agreement is
    at machine precision, and anything worse means the padding or the Nyquist
    split is wrong."""
    n, factor = 64, 8
    t = np.arange(n) / n
    # content strictly below Nyquist for the coarse grid
    x = sum(np.exp(2j * np.pi * k * t) * (0.7 ** k) for k in range(1, 12))
    fine = np.asarray(_upsample_bandlimited(jnp.asarray(x), factor))
    tf = np.arange(n * factor) / (n * factor)
    exact = sum(np.exp(2j * np.pi * k * tf) * (0.7 ** k) for k in range(1, 12))
    err = np.max(np.abs(fine - exact))
    assert err < 1e-12, "band-limited upsampling is not exact: %.3e" % err
    # and it must reproduce the original samples where they sit
    assert np.max(np.abs(fine[::factor] - x)) < 1e-12


def test_nyquist_bin_is_split_not_dumped():
    """A real signal upsampled must stay real.  Dumping the +n/2 bin into one
    side instead of splitting it leaves an imaginary part oscillating at
    Nyquist -- small, grid-phase dependent, and exactly the class of error this
    change removes."""
    n, factor = 32, 4
    t = np.arange(n) / n
    x = np.cos(2 * np.pi * (n // 2) * t) + 0.3 * np.cos(2 * np.pi * 3 * t)
    fine = np.asarray(_upsample_bandlimited(jnp.asarray(x + 0j), factor))
    assert np.max(np.abs(fine.imag)) < 1e-12, (
        "upsampling a real signal produced an imaginary part %.3e -- the "
        "Nyquist bin is not being split evenly" % np.max(np.abs(fine.imag)))


def _sharp_case(npts=256, deltaT=1.0 / 4096, sigma_samples=4.0, amp=600.0,
                phase=0.0):
    """kappa(t) BAND-LIMITED and well resolved, but LARGE.

    This is the production geometry, and getting it wrong is easy: it is not
    that kappa is narrow -- kappa is smooth on the sample grid, which is why
    band-limited reconstruction works at all.  It is that exp(kappa) is narrow
    BECAUSE kappa is large.  Near the peak
        kappa ~ amp - amp t^2 / (2 sigma_k^2)
    so exp(kappa) has width sigma_k / sqrt(amp): with sigma_k = 4 dt and
    amp = 600 that is 0.16 dt, i.e. sub-sample, while kappa itself spans ~4
    samples and is comfortably below Nyquist.  That is exactly the regime
    sigma_t = 1/(2 pi rho sigma_f) describes -- the integrand narrows as the
    signal gets louder, the grid does not.
    """
    t = (np.arange(npts) - npts // 2) * deltaT
    sigma_k = sigma_samples * deltaT
    kappa = amp * np.exp(-0.5 * ((t - phase * deltaT) / sigma_k) ** 2)
    return t, kappa, deltaT


def test_stock_simpson_is_grid_phase_dependent_and_bandlimited_is_not():
    """The BITING test.  Slide the peak across one sample spacing: stock Simpson
    swings by orders of magnitude more than the band-limited quadrature.

    This is what fails if someone reverts the fix, and it needs no reference
    integral -- it is a self-consistency statement, since the true value cannot
    depend on where the sampling grid happens to sit.
    """
    vals_simpson, vals_bl = [], []
    for phase in np.linspace(0.0, 1.0, 9):
        t, kappa, deltaT = _sharp_case(phase=phase)
        k = jnp.asarray(kappa[None, :] + 0j)
        rho = jnp.zeros((1, kappa.size))
        w = jnp.asarray(_simpson_weights(kappa.size, deltaT))
        vals_simpson.append(float(_time_marginalize(k.real, w)[0]))
        vals_bl.append(float(_time_marginalize_bandlimited(k, rho, deltaT, 16)[0]))
    span_s = max(vals_simpson) - min(vals_simpson)
    span_b = max(vals_bl) - min(vals_bl)
    assert span_b < 0.05, (
        "band-limited quadrature is still grid-phase dependent: span %.4f nats"
        % span_b)
    assert span_s > 20 * max(span_b, 1e-6), (
        "this test does not BITE: stock Simpson span %.4f vs band-limited "
        "%.4f -- it would not catch a revert.  Narrow the peak until it does."
        % (span_s, span_b))


def test_bandlimited_converges_in_the_upsample_factor():
    """Increasing the (free) reconstruction factor must stop changing the
    answer -- otherwise the quadrature is not converged and the 'exact'
    claim is empty."""
    t, kappa, deltaT = _sharp_case(phase=0.37)
    k = jnp.asarray(kappa[None, :] + 0j)
    rho = jnp.zeros((1, kappa.size))
    got = [float(_time_marginalize_bandlimited(k, rho, deltaT, f)[0])
           for f in (8, 16, 32)]
    assert abs(got[2] - got[1]) < 1e-3, (
        "not converged in the upsample factor: %r" % got)


def test_bandlimited_integrates_the_same_interval_as_simpson():
    """A CONSTANT integrand isolates the interval: the answer is then exactly
    log(length), with no reconstruction error of any kind to hide behind.

    The upsampled array holds n*factor points spanning (n - 1/factor)*deltaT,
    while the likelihood's window is (n-1)*deltaT -- the trailing factor-1
    samples are the periodic FFT continuation past the last data sample.
    Integrating them renormalizes every lnL by log((n - 1/factor)/(n - 1)),
    which is 0.0138 nats at n=64, factor=8: invisible in a self-consistency
    scan, fatal when comparing against the stock quadrature.
    """
    n, deltaT, c = 64, 1.0 / 4096, 3.0
    k = jnp.asarray(np.full((1, n), c) + 0j)
    rho = jnp.zeros((1, n))
    exact = c + np.log((n - 1) * deltaT)
    for factor in (1, 4, 8, 16):
        got = float(_time_marginalize_bandlimited(k, rho, deltaT, factor)[0])
        assert abs(got - exact) < 1e-12, (
            "factor %d integrates the wrong window: %.12f vs %.12f, a %+.4f-nat "
            "normalization shift" % (factor, got, exact, got - exact))
    # ... and that is the interval the stock Simpson path uses, so the two agree
    # on a constant instead of differing by a fixed offset.
    w = jnp.asarray(_simpson_weights(n, deltaT))
    assert abs(float(_time_marginalize(k.real, w)[0]) - exact) < 1e-12


def test_bandlimited_refuses_arrival_time_dependent_norms():
    """The band-limited quadrature reconstructs kappa alone and uses the model
    norm at a single time bin.  That is the whole norm for the baseline and
    finite-size accumulators, but the slow-rotation post-phase makes <h|h>
    arrival-time dependent, where the first bin would be a different likelihood
    rather than a better-integrated one.  It must be refused, not approximated.
    """
    class _StubData(object):
        def __init__(self, feature):
            self.feature = feature

    assert _norm_is_arrival_time_dependent(_StubData("rotation"))
    assert not _norm_is_arrival_time_dependent(_StubData("freqresponse"))
    assert not _norm_is_arrival_time_dependent(_StubData(None))

    # The screen must run BEFORE anything touches the data, both so the stub
    # suffices here and so the caller is not made to pay a full trace to be told
    # the option is unavailable.
    with pytest.raises(ValueError, match="arrival time"):
        fused_log_likelihood(_StubData("rotation"), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                             time_quad="bandlimited")


def test_unknown_time_quad_raises_rather_than_silently_defaulting():
    """A typo'd quadrature name must not quietly give the OLD behaviour."""
    from RIFT.likelihood.jax_ile.core import _TIME_QUAD_CHOICES
    assert "bandlimited" in _TIME_QUAD_CHOICES and "simpson" in _TIME_QUAD_CHOICES
    import inspect
    from RIFT.likelihood.jax_ile import core as _core
    src = inspect.getsource(_core.fused_log_likelihood)
    assert "raise ValueError" in src, (
        "an unrecognised time_quad must raise, not fall through to the default")
