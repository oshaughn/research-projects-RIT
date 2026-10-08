#!/usr/bin/env python3
"""
The exported time grid must be labelled with the spacing the likelihood steps by.

``bin/integrate_likelihood_extrinsic_batchmode`` built the export grid in
``resample_samples()`` as

    tvals = linspace(-t_ref_wind, t_ref_wind, int(t_ref_wind*2/P.deltaT))

whose spacing is ``2W/(N-1)``.  ``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop``
does not step by that: it uses only ``tvals[0]`` (``tfirst = t_det + tvals[0]``) and
then reads ``lnLt[k]`` from the precomputed Q_lm buffer at index ``ifirst + k``,
i.e. at detector time ``tfirst + k*deltaT``.  Labelling those values with a
closed-interval linspace stretches the exported time about the START of the window,

    t_reported = t_true + (t_true - (event_time - W)) * delta,
    delta      = (1 + f)/(N - 1),   with  2*W*srate = N + f

which at the production settings (srate 4096, W = 75 ms) is +171 us at the window
centre and +0.23% on the width of any exported time posterior.
"""

import os
import re

import numpy as np
import pytest

SRATE = 4096.0
WINDOW_HALF = 75e-3          # --data-integration-window-half default

ILE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bin",
    "integrate_likelihood_extrinsic_batchmode",
)


def npts(srate=SRATE, window_half=WINDOW_HALF):
    return int(window_half*2/(1.0/srate))


def tvals_fixed(srate=SRATE, window_half=WINDOW_HALF):
    """The corrected grid: same first sample and same length, spacing tied to deltaT."""
    return -window_half + (1.0/srate)*np.arange(npts(srate, window_half))


def tvals_legacy(srate=SRATE, window_half=WINDOW_HALF):
    """The construction that shipped through O4c."""
    return np.linspace(-window_half, window_half, npts(srate, window_half))


def label_bias(t, srate=SRATE, window_half=WINDOW_HALF):
    """Predicted legacy label error at a time t relative to the event time."""
    x = 2*window_half*srate
    n = int(x)
    return (t + window_half)*(1 + (x - n))/(n - 1)


def test_corrected_grid_is_spaced_by_deltaT():
    assert np.allclose(np.diff(tvals_fixed()), 1.0/SRATE, rtol=0, atol=1e-15)


def test_the_fix_changes_only_the_labels():
    """
    The likelihood consumes tvals[0] and len(tvals) and nothing else, so a fix that
    preserves both cannot move lnL, lnZ, or which data samples are integrated.
    """
    leg, fix = tvals_legacy(), tvals_fixed()
    assert len(leg) == len(fix)
    assert leg[0] == fix[0] == -WINDOW_HALF


@pytest.mark.parametrize("t_true", [-40e-3, -20e-3, 0.0, 20e-3, 40e-3])
def test_legacy_bias_matches_the_closed_form(t_true):
    """
    Index the two grids identically -- which is what the likelihood does, since it
    walks the Q buffer one sample at a time from tvals[0] -- and the difference in
    the labels IS the bias.
    """
    k = int(round((t_true + WINDOW_HALF)*SRATE))
    fix = tvals_fixed()
    err = tvals_legacy()[k] - fix[k]
    # evaluate the closed form AT THE GRID NODE, not at the requested time: k is
    # rounded, so the node can sit up to half a sample away from t_true.
    assert err == pytest.approx(label_bias(fix[k]), abs=1e-9)


@pytest.mark.parametrize("srate,expect_us", [(4096, 171.3), (8192, 110.0), (16384, 48.9)])
def test_bias_at_the_window_centre(srate, expect_us):
    """
    delta is (1+f)/(N-1), not 1/srate: it depends on the fractional part of
    2*W*srate, so the progression is not a clean factor of two.
    """
    assert label_bias(0.0, srate=srate)*1e6 == pytest.approx(expect_us, abs=0.5)


def test_width_is_stretched_by_delta():
    """A posterior spanning [a, b] was reported 0.2284% wider at srate 4096."""
    ka = int(round((-20e-3 + WINDOW_HALF)*SRATE))
    kb = int(round((+20e-3 + WINDOW_HALF)*SRATE))
    leg, fix = tvals_legacy(), tvals_fixed()
    stretch = (leg[kb] - leg[ka])/(fix[kb] - fix[ka]) - 1
    x = 2*WINDOW_HALF*SRATE
    n = int(x)
    assert stretch == pytest.approx((1 + (x - n))/(n - 1), rel=1e-6)


def _export_grid_expression():
    """The right-hand side of the export-path grid, as the driver spells it today.

    Anchored on ``P.phi = identity_convert_togpu(...)``, the line that immediately
    follows it in resample_samples(): that is the grid whose labels become the
    exported 't_ref'.
    """
    with open(ILE_SCRIPT) as f:
        src = f.read()
    m = re.search(r"tvals\s*=\s*([^\n]*)\n\s*P\.phi\s*=\s*identity_convert_togpu", src)
    assert m, "could not find the export-path time grid in %s" % ILE_SCRIPT
    return m.group(1).split("#")[0].strip()


def test_export_path_grid_is_not_a_closed_interval_linspace():
    """
    Drift guard on the one line this patch changes.  A closed-interval linspace over
    the window silently reintroduces the +171 us bias in the exported time.
    """
    expr = _export_grid_expression()
    assert "linspace" not in expr, \
        "export time grid is a closed-interval linspace again: %r" % expr
    assert "arange" in expr and "deltaT" in expr, \
        "export time grid no longer steps by deltaT: %r" % expr
    assert "-t_ref_wind" in expr, \
        "export time grid no longer starts at -t_ref_wind: %r" % expr


def test_source_grid_reproduces_the_reference_grid():
    """Evaluate the shipped expression itself, so this test cannot drift from it."""
    expr = _export_grid_expression()
    for srate in (1024, 2048, 4096, 8192, 16384):
        ns = {"t_ref_wind": WINDOW_HALF, "xpy_default": np,
              "P": type("P", (), {"deltaT": 1.0/srate})()}
        got = eval(expr, ns)
        # nothing is injected but the window and deltaT, so the shipped expression
        # must supply its own npts: a wrong point count fails here, not silently.
        assert len(got) == npts(srate), (
            "srate %d: shipped grid has %d points, expected int(2*W/deltaT) = %d"
            % (srate, len(got), npts(srate)))
        np.testing.assert_allclose(got, tvals_fixed(srate), rtol=0, atol=1e-15)
