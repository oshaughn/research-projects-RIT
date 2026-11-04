"""Subprocess regression for the TEOBResumS-DALI near-aligned segfault.

The backend is a native extension, so a regression must fail this test rather
than terminate pytest itself.  CI hosts without the optional backend skip; a
host with the backend treats every later import/generation failure as real.
"""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest


def test_near_aligned_gwsignal_call_cannot_terminate_python():
    if importlib.util.find_spec("EOBRun_module") is None:
        pytest.skip("TEOBResumSDALI needs EOBRun_module, which is not installed here")

    child_code = textwrap.dedent(
        r"""
        import numpy as np
        import lal
        import lalsimulation as lalsim
        from RIFT import lalsimutils
        from RIFT.physics import GWSignal

        P = lalsimutils.ChooseWaveformParams()
        P.m1, P.m2 = 50 * lal.MSUN_SI, 30 * lal.MSUN_SI
        P.s1x, P.s1y, P.s1z = 1e-5, 0.0, 0.2
        P.s2x, P.s2y, P.s2z = 0.0, 0.0, -0.1
        P.dist = 400e6 * lal.PC_SI
        P.incl = 0.4
        P.phiref = P.psi = 0.0
        P.fmin = P.fref = 20.0
        P.deltaT, P.deltaF = 1.0 / 4096, 1.0 / 16
        P.eccentricity = P.meanPerAno = 0.0
        P.taper = lalsim.SIM_INSPIRAL_TAPER_NONE
        P.approx = lalsim.IMRPhenomXPHM

        modes = GWSignal.hlmoft(P, Lmax=4, approx_string="TEOBResumSDALI")
        assert modes
        assert all(np.isfinite(mode.data.data).all() for mode in modes.values())
        print("near-aligned-safe")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=120,
    )

    assert completed.returncode == 0, (
        "TEOBResumSDALI near-aligned child failed (a native crash is usually "
        "return code -11/139):\nstdout:\n{}\nstderr:\n{}".format(
            completed.stdout, completed.stderr
        )
    )
    assert "near-aligned-safe" in completed.stdout
