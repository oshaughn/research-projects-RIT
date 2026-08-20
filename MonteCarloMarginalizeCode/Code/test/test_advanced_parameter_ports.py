import os
import inspect
import ast
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


def test_hyperbolic_classification_is_distance_invariant(monkeypatch):
    monkeypatch.setenv("RIFT_TEOBRESUMS_SKIP_PROBE", "1")
    monkeypatch.setattr(lalsimutils, "EOBRun_module", _FakeEOBRunModule, raising=False)

    outcomes = []
    for distance_mpc in (1e-3, 1e24):
        parameters = lalsimutils.ChooseWaveformParams(
            m1=30 * lalsimutils.lal.MSUN_SI,
            m2=20 * lalsimutils.lal.MSUN_SI,
            dist=distance_mpc * 1e6 * lalsimutils.lal.PC_SI,
            E0=1.02,
            p_phi0=4.1,
        )
        outcomes.append(parameters.extract_param("hypclass"))

    assert outcomes == ["scatter", "scatter"]


def test_hyperbolic_mode_generation_has_no_absolute_strain_classifier():
    source = inspect.getsource(lalsimutils.hlmoft)
    assert "1e-26" not in source


def test_hyperbolic_endpoint_prefers_radial_dynamics():
    amplitude = np.array([0.0, 1.0, 0.5])
    assert lalsimutils._hyperbolic_endpoint_outcome(
        {"Prstar": np.array([-0.2])}, amplitude
    ) == "plunge"
    assert lalsimutils._hyperbolic_endpoint_outcome(
        {"Prstar": np.array([0.2])}, amplitude
    ) == "scatter"


def test_zero_mode_data_zeros_every_mode():
    class Data:
        def __init__(self, values):
            self.data = np.asarray(values, dtype=np.complex128)

    class Series:
        def __init__(self, values):
            self.data = Data(values)

    modes = {
        (2, 2): Series([1.0, 2.0]),
        (2, -2): Series([3.0, 4.0]),
    }
    lalsimutils._zero_mode_data(modes)
    assert all(np.count_nonzero(series.data.data) == 0 for series in modes.values())


def test_hyperbolic_convert_arguments_do_not_truncate_other_exports():
    script = Path(__file__).parents[1] / "bin" / "helper_LDG_Events.py"
    tree = ast.parse(script.read_text())
    hyperbolic_writes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_source = ast.unparse(node.test)
        if test_source != "opts.assume_hyperbolic":
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id != "open" or len(call.args) < 2:
                continue
            if isinstance(call.args[0], ast.Constant) and call.args[0].value == "helper_convert_args.txt":
                hyperbolic_writes.append(call.args[1].value)
    assert hyperbolic_writes == ["a"]


def test_nrhybsur_tidal_routing_checks_the_approximant_family():
    script = Path(__file__).parents[1] / "bin" / "util_RIFT_pseudo_pipe.py"
    source = script.read_text()
    assert "('NRHybSur' and" not in source
    assert source.count("'NRHybSur' in opts.approx") >= 2


def test_cip_rejects_hyperbolic_class_filter_without_hyperbolic_mode(tmp_path):
    script = (
        Path(__file__).parents[1]
        / "bin"
        / "util_ConstructIntrinsicPosterior_GenericCoordinates.py"
    )
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, str(script), "--force-scatter"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "require --use-hyperbolic" in result.stderr


def test_cip_preserves_base_rf_pca_and_rbf_fit_methods():
    script = (
        Path(__file__).parents[1]
        / "bin"
        / "util_ConstructIntrinsicPosterior_GenericCoordinates.py"
    )
    tree = ast.parse(script.read_text())
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    fit_method_values = {
        node.comparators[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and isinstance(node.left, ast.Attribute)
        and isinstance(node.left.value, ast.Name)
        and node.left.value.id == "opts"
        and node.left.attr == "fit_method"
    }
    assert {"fit_rf_pca", "fit_rbf"} <= function_names
    assert {"rf_pca", "rbf"} <= fit_method_values


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
