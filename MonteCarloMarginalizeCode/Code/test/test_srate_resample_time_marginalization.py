#!/usr/bin/env python3
"""
Unit tests for the time-marginalisation upsampling used by
``--srate-resample-time-marginalization`` in
``bin/integrate_likelihood_extrinsic_batchmode``.

Before the fix, the option was effectively a boolean: whenever the requested
rate exceeded --srate, the internal time grid was refined by a hardcoded factor
of two and the requested value was discarded.  With the O4c production settings
(--srate 4096, --data-integration-window-half 0.075) asking for 16384 Hz
delivered ~8173 Hz, and the exported geocentre times inherited that resolution.

These tests exercise a transcription of the shipped block, kept in sync by
``test_source_matches_reference_implementation`` below.
"""

import os
import re

import numpy as np
import pytest

# RIFT defaults exercised by the O4c production configuration.
SRATE = 4096.0
WINDOW_HALF = 75e-3  # --data-integration-window-half default
REQUESTED = 16384

ILE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "bin",
    "integrate_likelihood_extrinsic_batchmode",
)


def rift_tvals(srate=SRATE, window_half=WINDOW_HALF):
    """The internal time grid built by analyze_event_extrinsic_export."""
    n_points = int(2 * window_half / (1.0 / srate))
    return np.linspace(-window_half, window_half, n_points)


def upsample(tvals, lnLt, requested, fsample=SRATE):
    """Reference implementation, mirroring the shipped code."""
    from scipy.interpolate import CubicSpline

    if not (requested and requested > fsample):
        return tvals, lnLt
    dt_target = 1.0 / requested
    n_dense = int(np.floor((tvals[-1] - tvals[0]) / dt_target)) + 1
    tvals_denser = tvals[0] + dt_target * np.arange(n_dense)
    lnLt_new = np.zeros((lnLt.shape[0], n_dense))
    for index in range(lnLt.shape[0]):
        lnLt_new[index] = CubicSpline(tvals, lnLt[index])(tvals_denser)
    return tvals_denser, lnLt_new


def output_spacing(tvals):
    """The one spacing of the (uniform) output grid."""
    diffs = np.diff(tvals)
    assert np.allclose(diffs, diffs[0], rtol=0, atol=1e-15), "grid is not uniform"
    return diffs[0]


def effective_rate(tvals):
    return 1.0 / output_spacing(tvals)


@pytest.fixture
def toy_lnl():
    """A smooth, sharply peaked lnL(t): Gaussians of ~1 ms width."""
    tvals = rift_tvals()
    peak = np.array([[-0.7e-3], [0.0], [1.3e-3]])
    return tvals, -0.5 * ((tvals[None, :] - peak) / 1.0e-3) ** 2


def test_internal_grid_is_slightly_coarser_than_srate():
    """
    linspace(-W, W, N) with N = int(2*W*fS) spans the closed interval with N
    points, so the spacing is deltaT*N/(N-1) - about 0.2% coarser than 1/fS.
    The refinement factor must therefore be derived from the grid spacing, not
    from fSample, or the result lands just short of the requested rate.
    """
    tvals = rift_tvals()
    assert len(tvals) == 614
    assert tvals[1] - tvals[0] > 1.0 / SRATE
    assert effective_rate(tvals) == pytest.approx(4086.67, rel=1e-4)


@pytest.mark.parametrize("requested", [8192, 16384, 32768, 65536])
def test_recovers_the_exact_requested_rate(toy_lnl, requested):
    """
    The whole point of the fix: the output rate must equal the requested rate,
    not merely reach or exceed it.  The requested rates are powers of two, so
    1/requested is exactly representable in float64 and consecutive output
    times differ by exactly that step, to the bit.
    """
    tvals, lnl = toy_lnl
    dense, _ = upsample(tvals, lnl, requested)
    spacing = output_spacing(dense)
    assert spacing == 1.0 / requested            # bit-exact, not approx
    assert effective_rate(dense) == float(requested)


@pytest.mark.parametrize("requested", [16384, 32768])
def test_output_times_lie_on_the_requested_grid(toy_lnl, requested):
    """
    Every output time is tvals[0] + k/requested for integer k, i.e. the
    exported geocenter time is quantized at exactly 1/requested seconds.
    """
    tvals, lnl = toy_lnl
    dense, _ = upsample(tvals, lnl, requested)
    k = (dense - dense[0]) * requested
    np.testing.assert_allclose(k, np.round(k), rtol=0, atol=1e-9)


def test_scales_with_the_request(toy_lnl):
    """Doubling the request exactly halves the output spacing."""
    tvals, lnl = toy_lnl
    s16 = output_spacing(upsample(tvals, lnl, 16384)[0])
    s32 = output_spacing(upsample(tvals, lnl, 32768)[0])
    assert s16 == 2.0 * s32


def test_does_not_extrapolate_outside_the_original_grid(toy_lnl):
    """
    The dense grid must stay within [tvals[0], tvals[-1]] so the cubic spline
    never extrapolates (the old grid ran half a sample past tvals[-1]).  We
    floor the point count, so the far edge is left short by < 1/requested s.
    """
    tvals, lnl = toy_lnl
    dense, _ = upsample(tvals, lnl, REQUESTED)
    assert dense[0] == tvals[0]
    assert dense[-1] <= tvals[-1]
    assert (tvals[-1] - dense[-1]) < 1.0 / REQUESTED


def test_recovers_the_peak_to_the_requested_resolution(toy_lnl):
    """
    The exported geocentre time is drawn from this grid, so the grid spacing
    floors the achievable time resolution.
    """
    tvals, lnl = toy_lnl
    truth = np.array([-0.7e-3, 0.0, 1.3e-3])
    dense, lnl_dense = upsample(tvals, lnl, REQUESTED)
    error = np.abs(dense[np.argmax(lnl_dense, axis=1)] - truth).max()
    assert error < 1.0 / REQUESTED


def test_switch_is_off_at_or_below_fsample(toy_lnl):
    tvals, lnl = toy_lnl
    for requested in (None, 0, 2048, int(SRATE)):
        dense, lnl_dense = upsample(tvals, lnl, requested)
        np.testing.assert_array_equal(dense, tvals)
        np.testing.assert_array_equal(lnl_dense, lnl)


def test_source_matches_reference_implementation():
    """
    Guard against the shipped block and this reference drifting apart - the
    tests above are only meaningful if they describe the real code.
    """
    if not os.path.exists(ILE_SCRIPT):
        pytest.skip("ILE script not found next to the test directory")
    with open(ILE_SCRIPT) as handle:
        source = handle.read()

    block = re.search(
        r"(?:if|elif) opts\.srate_resample_time_marginalization and .*?lnLt_norm = "
        r"scipy\.special\.logsumexp\(lnLt,axis=-1\)",
        source,
        re.S,
    )
    assert block, "could not locate the upsampling block"
    text = block.group(0)

    # The output step must be exactly 1/requested, so the requested rate is
    # recovered exactly rather than snapped to a multiple of the grid.
    assert "dt_target = 1.0/opts.srate_resample_time_marginalization" in text
    assert "dt_target * np.arange(n_dense)" in text
    # ...and the old hardcoded doubling and the integer-factor upsample are gone.
    assert "np.arange(2*len(tvals))" not in text
    assert "lnLt.shape[1]*2" not in text
    assert "n_upsample" not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
