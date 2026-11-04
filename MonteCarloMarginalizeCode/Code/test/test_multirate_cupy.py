"""cupy legs of the multirate backend tests: cupy against numpy for the per-point preparation and
for the device templates.  Needs a usable GPU, so no CI runner can run it (rostered GPU in
.travis/ci_roster.txt).  Without a device it skips; with RIFT_CI_REQUIRE_GPU=1 it fails instead."""
import os

import numpy as np
import pytest

from RIFT.likelihood import factored_likelihood_multirate as flm
from test_multirate_precompute import inputs   # noqa: F401 (fixture)
from test_multirate_xpy import _run, _assert_close, test_xp_waveform_matches_lal_pieces as _xp_wf_check


def _cupy():
    if os.environ.get("RIFT_CI_REQUIRE_GPU"):
        return flm._select_xpy("cupy")   # raises without a device: a failure, not a skip
    xp = flm._select_xpy()
    if xp is np:
        pytest.skip("no usable GPU")
    return xp


def test_backend_cupy_matches_numpy(tmp_path, inputs):
    _cupy()
    a = _run(tmp_path, inputs, multirate_xpy="numpy")
    b = _run(tmp_path, inputs, multirate_xpy="cupy")
    _assert_close(a, b, 1e-10)


def test_xp_waveform_matches_lal_pieces_cupy():
    _xp_wf_check(_cupy())
