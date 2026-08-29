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

What the reconstruction does cost is GUARD SAMPLES.  The window is a crop of a
longer correlation buffer, its two ends do not join, and the FFT that does the
reconstruction has no choice but to treat them as though they did -- so the
seam's fictitious jump rings into the inserted samples, invisibly, since every
retained sample stays exact.  The guard samples put that seam outside the
integrated window; the tests below pin both halves of that (the defect on a
non-periodic crop, and its removal), because an exactness check written only
with modes that fit a whole number of periods in the crop passes either way.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from RIFT.likelihood.jax_ile.core import (
    _upsample_bandlimited, _time_marginalize, _time_marginalize_bandlimited,
    _simpson_weights, _norm_is_arrival_time_dependent, fused_log_likelihood,
    _guarded_window, default_time_guard)


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


_TONE_N, _TONE_K0, _TONE_START = 1024, 17, 256


def _cropped_tone(npts, guard, factor):
    """A CROP of a globally band-limited tone, and the exact continuous function.

    exp(2j pi k0 t / N) with k0/N = 0.0166 cycles/sample is band-limited far
    below Nyquist, so nothing here is about resolution: the whole difficulty is
    that the crop is a window cut out of a longer buffer, exactly as the
    accumulators' window is cut out of the rholm correlation buffer.  k0*npts/N
    is not an integer, so the crop's ends do not join, which is the ordinary case
    -- and the one a periodic reconstruction gets wrong.

    Returns ``(coarse_samples, exact_fine, sl)``: ``npts + 2*guard`` coarse
    samples, the true function on the fine grid of the KEPT window, and the slice
    of the upsampled array that window occupies -- the same arithmetic
    ``_time_marginalize_bandlimited`` does.
    """
    w = 2.0 * np.pi * _TONE_K0 / _TONE_N
    origin = _TONE_START - guard
    coarse = origin + np.arange(npts + 2 * guard)
    sl = slice(guard * factor, guard * factor + (npts - 1) * factor + 1)
    fine = origin + np.arange(sl.start, sl.stop) / float(factor)
    return np.exp(1j * w * coarse), np.exp(1j * w * fine), sl


def test_periodic_upsampling_of_a_crop_is_wrong_between_the_samples():
    """THE DEFECT the guard exists for, and it hides from the obvious check.

    Crop a band-limited tone out of a longer buffer and upsample it: every
    RETAINED sample is still exact -- the interpolant passes through its inputs
    -- while the INSERTED samples, the ones the quadrature actually integrates,
    carry the ringing from the seam the FFT invents between the two ends.  So
    `fine[::factor] == x` proves nothing, and neither does an exactness test
    built from modes that fit a whole number of periods in the window.
    """
    npts, factor = 64, 8
    x, exact, sl = _cropped_tone(npts, 0, factor)
    fine = np.asarray(_upsample_bandlimited(jnp.asarray(x), factor))
    assert np.max(np.abs(fine[::factor] - x)) < 1e-12, (
        "the reconstruction no longer reproduces its own input samples")
    err = np.max(np.abs(fine[sl] - exact))
    assert err > 1e-2, (
        "this test does not BITE: unguarded reconstruction of a NON-PERIODIC "
        "crop is off by only %.3e, so it would not catch the guard being "
        "dropped.  Check the crop really is non-periodic (k0*npts/N must not be "
        "an integer)." % err)


def test_guard_samples_move_the_seam_out_of_the_reconstructed_window():
    """The fix: reconstruct from a window widened by guard samples and keep only
    the middle.  The seam error falls off like 1/(distance from the seam), so the
    guard buys accuracy in the region that is actually integrated -- the
    regression this file was missing."""
    npts, factor, guard = 64, 8, 128
    x0, exact0, sl0 = _cropped_tone(npts, 0, factor)
    err0 = np.max(np.abs(
        np.asarray(_upsample_bandlimited(jnp.asarray(x0), factor))[sl0] - exact0))
    xg, exactg, slg = _cropped_tone(npts, guard, factor)
    errg = np.max(np.abs(
        np.asarray(_upsample_bandlimited(jnp.asarray(xg), factor))[slg] - exactg))
    assert errg < 1e-2, (
        "guarded reconstruction is still %.3e off inside the integrated window"
        % errg)
    assert errg < err0 / 5.0, (
        "guard samples bought almost nothing: %.3e guarded vs %.3e unguarded"
        % (errg, err0))


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

    This case is deliberately GUARD-FREE: the Gaussian is 32 sigma_k from either
    end, so the window's ends join to the precision of exp(-512) and the periodic
    seam has nothing to ring on.  That is what makes it a clean probe of the
    quadrature -- and also why it cannot substitute for the cropped-tone tests
    below, which are where the seam actually bites.
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
        # guard=0: legitimate here, and only here -- see _sharp_case.
        vals_bl.append(
            float(_time_marginalize_bandlimited(k, rho, deltaT, 16, 0)[0]))
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
    got = [float(_time_marginalize_bandlimited(k, rho, deltaT, f, 0)[0])
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
        got = float(_time_marginalize_bandlimited(k, rho, deltaT, factor, 0)[0])
        assert abs(got - exact) < 1e-12, (
            "factor %d integrates the wrong window: %.12f vs %.12f, a %+.4f-nat "
            "normalization shift" % (factor, got, exact, got - exact))
    # ... and that is the interval the stock Simpson path uses, so the two agree
    # on a constant instead of differing by a fixed offset.
    w = jnp.asarray(_simpson_weights(n, deltaT))
    assert abs(float(_time_marginalize(k.real, w)[0]) - exact) < 1e-12


def test_guard_samples_are_support_and_are_never_integrated():
    """A CONSTANT integrand isolates the interval again, now with a guard: the
    answer must stay log((npts-1)*deltaT) whatever the guard is.  Integrating the
    guard samples instead of using them as support would renormalize every lnL by
    log((npts-1+2*guard)/(npts-1)) -- 1.4 nats at npts=64, guard=32 -- and the
    grid-phase scan above would not notice."""
    npts, deltaT, c = 64, 1.0 / 4096, 3.0
    exact = c + np.log((npts - 1) * deltaT)
    for guard in (0, 1, 5, 32):
        n = npts + 2 * guard
        k = jnp.asarray(np.full((1, n), c) + 0j)
        rho = jnp.zeros((1, n))
        for factor in (1, 4, 8):
            got = float(
                _time_marginalize_bandlimited(k, rho, deltaT, factor, guard)[0])
            assert abs(got - exact) < 1e-12, (
                "guard %d, factor %d integrates the wrong window: %.12f vs "
                "%.12f, a %+.4f-nat normalization shift"
                % (guard, factor, got, exact, got - exact))
    # and a guard that would leave nothing to integrate is an error, not a
    # silently empty window
    k = jnp.asarray(np.full((1, 8), c) + 0j)
    with pytest.raises(ValueError, match="guard"):
        _time_marginalize_bandlimited(k, jnp.zeros((1, 8)), deltaT, 4, 4)


def test_bandlimited_quadrature_has_no_default_guard():
    """No default, on purpose: guard=0 is the periodic-seam defect on any real
    (cropped) window, so a caller that forgets it must get a TypeError rather
    than a quietly wrong likelihood."""
    import inspect
    sig = inspect.signature(_time_marginalize_bandlimited)
    assert sig.parameters["guard"].default is inspect.Parameter.empty, (
        "guard acquired a default; an unguarded call must be impossible to make "
        "by accident")
    with pytest.raises(TypeError):
        _time_marginalize_bandlimited(jnp.zeros((1, 8), dtype=jnp.complex128),
                                      jnp.zeros((1, 8)), 1.0 / 4096, 4)


def test_both_accumulators_take_the_guarded_window_from_one_place():
    """The guard widens the window by shifting the gather offsets to
    [-guard, npts+guard).  An off-by-one there misplaces the arrival time of
    every guarded evaluation, and a second copy of the rule in the banded
    accumulator is how that off-by-one would arrive, so neither accumulator is
    allowed to build its own offsets."""
    import inspect
    from RIFT.likelihood.jax_ile import core as _core

    class _Stub(object):
        npts = 8

    npts, off = _guarded_window(_Stub(), 0)
    assert npts == 8 and np.array_equal(np.asarray(off), np.arange(8)), (
        "guard=0 must be the production window, unchanged")
    npts, off = _guarded_window(_Stub(), 3)
    assert npts == 14 and np.array_equal(np.asarray(off), np.arange(-3, 11)), (
        "guarded window is not centred on the production window")
    with pytest.raises(ValueError, match="guard"):
        _guarded_window(_Stub(), -1)

    for fn in (_core._accumulate_unit, _core._accumulate_unit_banded):
        src = inspect.getsource(fn)
        assert "_guarded_window(data, guard)" in src, (
            "%s builds its own time offsets instead of taking them from "
            "_guarded_window" % fn.__name__)


def test_fused_bandlimited_widens_the_window_and_forwards_the_guard(monkeypatch):
    """The plumbing, which is where this defect could quietly come back: asking
    for the band-limited quadrature must make the ACCUMULATOR gather guard
    samples and hand the same number to the quadrature.  Reconstructing an
    unwidened window would be the original bug with a guard argument bolted on.
    """
    from RIFT.likelihood.jax_ile import core as _core

    class _StubData(object):
        feature = None
        npts = 48
        deltaT = 1.0 / 4096
        distMpcRef = 1000.0
        w_t = jnp.asarray(_simpson_weights(48, 1.0 / 4096))

    seen = {}

    def _fake_accumulate(data, ra, dec, psi, incl, phiref, interp,
                         phase_marginalization, guard=0):
        seen["guard"] = int(guard)
        n = data.npts + 2 * int(guard)
        t = np.arange(n) - 0.5 * n
        kappa = (7.0 * np.exp(-0.5 * (t / 3.0) ** 2) + 0.5)[None, :] + 0j
        return jnp.asarray(kappa), jnp.zeros((1, n))

    monkeypatch.setattr(_core, "_accumulate_unit", _fake_accumulate)
    z = np.zeros(1)
    dist = np.full(1, _StubData.distMpcRef)   # invDist == 1, so kappa is as built

    got = float(_core.fused_log_likelihood(
        _StubData(), z, z, z, z, z, dist, time_quad="bandlimited",
        time_upsample=4)[0])
    assert seen["guard"] == default_time_guard(_StubData.npts), (
        "the band-limited path did not widen the accumulation window by the "
        "default guard (got %r)" % (seen["guard"],))
    k, rho = _fake_accumulate(_StubData(), None, None, None, None, None, None,
                              False, guard=seen["guard"])
    want = float(_time_marginalize_bandlimited(
        k, rho, _StubData.deltaT, 4, seen["guard"])[0])
    assert abs(got - want) < 1e-12, (
        "fused_log_likelihood did not pass its guard to the quadrature: "
        "%.12f vs %.12f" % (got, want))

    _core.fused_log_likelihood(_StubData(), z, z, z, z, z, dist,
                               time_quad="bandlimited", time_upsample=4,
                               time_guard=3)
    assert seen["guard"] == 3, "explicit time_guard ignored (got %r)" % (
        seen["guard"],)

    _core.fused_log_likelihood(_StubData(), z, z, z, z, z, dist,
                               time_quad="simpson")
    assert seen["guard"] == 0, (
        "the stock Simpson path must keep gathering exactly npts bins; it asked "
        "for a guard of %r" % (seen["guard"],))


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
