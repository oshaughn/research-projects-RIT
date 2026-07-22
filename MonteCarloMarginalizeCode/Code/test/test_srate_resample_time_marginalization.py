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
    deltaT_orig = tvals[1] - tvals[0]
    n_upsample = max(2, int(np.ceil(requested * deltaT_orig)))
    n_dense = n_upsample * (len(tvals) - 1) + 1
    tvals_denser = tvals[0] + (deltaT_orig / n_upsample) * np.arange(n_dense)
    lnLt_new = np.zeros((lnLt.shape[0], n_dense))
    for index in range(lnLt.shape[0]):
        lnLt_new[index] = CubicSpline(tvals, lnLt[index])(tvals_denser)
    return tvals_denser, lnLt_new


def effective_rate(tvals):
    return 1.0 / np.diff(tvals).min()


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


def test_reaches_the_requested_rate(toy_lnl):
    tvals, lnl = toy_lnl
    dense, _ = upsample(tvals, lnl, REQUESTED)
    assert effective_rate(dense) >= REQUESTED


def test_scales_with_the_request(toy_lnl):
    """Regression guard for the old behaviour, which ignored the value."""
    tvals, lnl = toy_lnl
    rate_16k = effective_rate(upsample(tvals, lnl, 16384)[0])
    rate_32k = effective_rate(upsample(tvals, lnl, 32768)[0])
    assert rate_16k >= 16384
    assert rate_32k >= 32768
    assert rate_32k > 1.5 * rate_16k


def test_does_not_extrapolate_outside_the_original_grid(toy_lnl):
    """
    The previous grid, tvals[0] + (dt/2)*arange(2N), ended half a sample past
    tvals[-1], where CubicSpline extrapolates.
    """
    tvals, lnl = toy_lnl
    dense, _ = upsample(tvals, lnl, REQUESTED)
    assert dense[0] == pytest.approx(tvals[0])
    assert dense[-1] == pytest.approx(tvals[-1])


def test_preserves_the_original_nodes(toy_lnl):
    """Refinement, not re-derivation: lnL at the original times is unchanged."""
    tvals, lnl = toy_lnl
    dense, lnl_dense = upsample(tvals, lnl, REQUESTED)
    factor = max(2, int(np.ceil(REQUESTED * (tvals[1] - tvals[0]))))
    np.testing.assert_allclose(dense[::factor], tvals, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(lnl_dense[:, ::factor], lnl, rtol=1e-9, atol=1e-9)


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
        r"if opts\.srate_resample_time_marginalization and .*?lnLt_norm = "
        r"scipy\.special\.logsumexp\(lnLt,axis=-1\)",
        source,
        re.S,
    )
    assert block, "could not locate the upsampling block"
    text = block.group(0)

    # The requested rate must actually be used, not just tested for truthiness.
    assert "np.ceil(opts.srate_resample_time_marginalization" in text
    # ...and the old hardcoded doubling must be gone.
    assert "np.arange(2*len(tvals))" not in text
    assert "lnLt.shape[1]*2" not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
