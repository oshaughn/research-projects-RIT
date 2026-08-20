import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import RIFT.lalsimutils as lalsimutils


class _FakeEOBRunModule:
    last_parameters = None

    @staticmethod
    def EOBRunPy(parameters):
        _FakeEOBRunModule.last_parameters = parameters
        amplitude = np.array([0.0, 1.0, 0.2])
        phase = np.zeros_like(amplitude)
        return np.arange(3), None, None, {"1": [amplitude, phase]}, None


def test_hyperbolic_classification_uses_both_component_masses(monkeypatch):
    monkeypatch.setenv("RIFT_TEOBRESUMS_SKIP_PROBE", "1")
    monkeypatch.setattr(lalsimutils, "EOBRun_module", _FakeEOBRunModule, raising=False)
    parameters = lalsimutils.ChooseWaveformParams(
        m1=30 * lalsimutils.lal.MSUN_SI,
        m2=20 * lalsimutils.lal.MSUN_SI,
        dist=1e6 * lalsimutils.lal.PC_SI,
        E0=1.02,
        p_phi0=4.1,
    )

    assert parameters.extract_param("hypclass") == "scatter"
    assert _FakeEOBRunModule.last_parameters["arg_out"] == "yes"
    assert _FakeEOBRunModule.last_parameters["nqc"] == "no"
    assert "LambdaAl2" in _FakeEOBRunModule.last_parameters
    assert "LambdaBl2" in _FakeEOBRunModule.last_parameters
    assert _FakeEOBRunModule.last_parameters["nqc_coefs_hlm"] == "none"
    assert _FakeEOBRunModule.last_parameters["nqc_coefs_flx"] == "none"
    assert _FakeEOBRunModule.last_parameters["use_geometric_units"] == "no"
    assert _FakeEOBRunModule.last_parameters["interp_uniform_grid"] == "yes"
    assert _FakeEOBRunModule.last_parameters["output_hpc"] == "no"


def test_real_hyperbolic_classification_when_teob_is_available(monkeypatch):
    eobrun_module = pytest.importorskip("EOBRun_module")
    monkeypatch.setattr(lalsimutils, "EOBRun_module", eobrun_module, raising=False)
    parameters = lalsimutils.ChooseWaveformParams(
        m1=30 * lalsimutils.lal.MSUN_SI,
        m2=30 * lalsimutils.lal.MSUN_SI,
        dist=1e6 * lalsimutils.lal.PC_SI,
        E0=1.0027,
        p_phi0=4.0,
        fmin=20,
        deltaT=1 / 4096,
    )

    assert parameters.extract_param("hypclass") in {
        "scatter",
        "plunge",
        "zoomwhirl",
        "meaningless",
    }


def _run_clean_ile(tmp_path, option, row):
    input_path = tmp_path / "ile.dat"
    input_path.write_text(" ".join(str(value) for value in row) + "\n")
    script = Path(__file__).parents[1] / "bin" / "util_CleanILE.py"
    result = subprocess.run(
        [sys.executable, str(script), option, str(input_path)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return result.stdout.split()


def test_clean_ile_keeps_hyperbolic_columns(tmp_path):
    row = [-1, 30, 20, 0, 0, 0, 0, 0, 0, 1.02, 4.1, 12, 0.1, 100, 30]
    output = _run_clean_ile(tmp_path, "--hyperbolic", row)

    assert len(output) == 15
    assert output[9:11] == ["1.02", "4.1"]


def test_clean_ile_does_not_treat_a6c_as_distance(tmp_path):
    row = [-1, 30, 20, 0, 0, 0, 0, 0, 0, -55, 12, 0.1, 100, 30]
    output = _run_clean_ile(tmp_path, "--a6c", row)

    assert len(output) == 14
    assert output[9] == "-55.0"


def test_clean_ile_keeps_tidal_a6c_columns(tmp_path):
    row = [-1, 2, 1.4, 0, 0, 0, 0, 0, 0, 400, 800, -55, 12, 0.1, 100, 30]
    output = _run_clean_ile(tmp_path, "--a6c", row)

    assert len(output) == 16
    assert output[9:12] == ["400.0", "800.0", "-55.0"]
