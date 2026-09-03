# RIFT-CI-GATE: q-window-stencil
# ^ registers this file with .travis/test-q-window-stencil.sh, run by ci.yml's
#   q-window-stencil-check job.  Membership lives here, in the test file, so that
#   adding a test needs no edit to any shared list.  Do not reword the line above.
# ---------------------------------------------------------------------------------
# WHY THIS FILE IS IN q-window-stencil-check.  Moved verbatim from the comment block
# above that job's hand-maintained file list in .github/workflows/ci.yml; it lives
# here now so that registering a test needs no edit to a shared file.
#
# test_noloop_time_interp pins the cubic Q_lm window against the nearest-sample window at
# integer offsets, against linear interpolation at midpoints, and against an exact cubic
# polynomial.  It was registered by this PR: it matched no job in ci.yml before, so its
# three tests had never run in CI -- the same silent loss this gate exists to stop.
# ---------------------------------------------------------------------------------
import os

os.environ.setdefault("RIFT_LOWLATENCY", "1")

import numpy as np

from RIFT.likelihood.factored_likelihood import (
    _cubic_Q_window_numpy,
    _nearest_Q_window_numpy,
)


def test_cubic_q_window_matches_nearest_at_integer_samples():
    q = (np.arange(40) + 1j * np.arange(40, 80)).reshape(20, 2)
    start = np.asarray([2, 7, 12], dtype=np.int32)

    nearest = _nearest_Q_window_numpy(q, start, 5)
    cubic = _cubic_Q_window_numpy(q, start, np.zeros(len(start)), 5)

    assert np.allclose(cubic, nearest)


def test_cubic_q_window_reproduces_linear_midpoints():
    q = (np.arange(40) + 1j * np.arange(40, 80)).reshape(20, 2)
    start = np.asarray([3], dtype=np.int32)

    cubic = _cubic_Q_window_numpy(q, start, np.asarray([0.5]), 4)
    expected = 0.5 * (q[3:7] + q[4:8])

    assert np.allclose(cubic[0], expected)


def test_cubic_q_window_reproduces_cubic_polynomial():
    times = np.arange(30, dtype=float)
    values = times**3 - 2.0 * times**2 + 0.5 * times - 7.0
    q = np.column_stack([values, values + 1j * values])
    start = np.asarray([10], dtype=np.int32)
    frac = np.asarray([0.37])

    cubic = _cubic_Q_window_numpy(q, start, frac, 3)
    target_times = start[0] + frac[0] + np.arange(3)
    expected_values = target_times**3 - 2.0 * target_times**2 + 0.5 * target_times - 7.0
    expected = np.column_stack([expected_values, expected_values + 1j * expected_values])

    assert np.allclose(cubic[0], expected)
