import os
import inspect
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def test_helper_defines_a6c_range_without_ini():
    script = Path(__file__).parents[1] / "bin" / "helper_LDG_Events.py"
    tree = ast.parse(script.read_text())
    unconditional = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "a6c_range_str"
            for target in node.targets
        )
    ]
    # --use-EOB-parameters must not require an [engine] a6c_min entry
    assert unconditional


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


def _run_clean_ile_lines(tmp_path, rows, *options):
    input_path = tmp_path / "ile.dat"
    input_path.write_text(
        "".join(" ".join(str(value) for value in row) + "\n" for row in rows)
    )
    script = Path(__file__).parents[1] / "bin" / "util_CleanILE.py"
    result = subprocess.run(
        [sys.executable, str(script), *options, str(input_path)],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    return [line.split() for line in result.stdout.strip().splitlines()]


def _run_clean_ile(tmp_path, option, row):
    return _run_clean_ile_lines(tmp_path, [row], option)[0]


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


def test_clean_ile_keeps_hyperbolic_a6c_columns(tmp_path):
    rows = [
        [-1, 30, 20, 0, 0, 0, 0, 0, 0, -55, 1.02, 4.1, 12, 0.1, 100, 30],
        [-1, 30, 20, 0, 0, 0, 0, 0, 0, -35, 1.02, 4.1, 11, 0.1, 100, 30],
    ]
    lines = _run_clean_ile_lines(tmp_path, rows, "--hyperbolic", "--a6c")

    # distinct a6c values are distinct intrinsic points, not repeated evaluations
    assert len(lines) == 2
    assert all(len(line) == 16 for line in lines)
    assert sorted(line[9] for line in lines) == ["-35.0", "-55.0"]
    assert all(line[10:12] == ["1.02", "4.1"] for line in lines)


def test_clean_ile_keeps_every_combined_advanced_column(tmp_path):
    row = [-1, 30, 20, 0, 0, 0, 0, 0, 0, -55, 1.02, 4.1, 0.3, 1.1, 12, 0.1, 100, 30]
    output = _run_clean_ile_lines(
        tmp_path, [row], "--a6c", "--hyperbolic", "--eccentricity", "--meanPerAno"
    )[0]

    assert len(output) == 18
    assert output[9] == "-55.0"
    assert output[10:12] == ["1.02", "4.1"]
    assert output[12:14] == ["0.3", "1.1"]


def test_clean_ile_keeps_tidal_columns_alongside_advanced_groups(tmp_path):
    row = [-1, 2, 1.4, 0, 0, 0, 0, 0, 0, 400, 800, -55, 1.02, 4.1, 0.3, 12, 0.1, 100, 30]
    output = _run_clean_ile_lines(
        tmp_path, [row], "--a6c", "--hyperbolic", "--eccentricity"
    )[0]

    assert len(output) == 19
    assert output[9:11] == ["400.0", "800.0"]
    assert output[11] == "-55.0"
    assert output[12:14] == ["1.02", "4.1"]
    assert output[14] == "0.3"


def test_dag_postprocess_forwards_every_advanced_flag():
    script = Path(__file__).parents[1] / "bin" / "util_ILEdagPostprocess.sh"
    source = script.read_text()
    # every flag after the directory/label arguments reaches util_CleanILE.py
    assert '"${CLEAN_FLAGS[@]}"' in source
    # ... instead of a mutually exclusive dispatch on the first flag only
    assert "'--eccentricity'" not in source
    assert "'--hyperbolic'" not in source


def _extract_ile_output_block():
    """Source of the ILE .dat writer (hyperpipeline branch + legacy branch)."""
    script = Path(__file__).parents[1] / "bin" / "integrate_likelihood_extrinsic_batchmode"
    tree = ast.parse(script.read_text())
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(node, field, None)
            if not isinstance(statements, list):
                continue
            for index, statement in enumerate(statements):
                if not (
                    isinstance(statement, ast.If)
                    and ast.unparse(statement.test) == "_hpio.is_active()"
                ):
                    continue
                start = index
                while start > 0 and not isinstance(statements[start - 1], ast.ImportFrom):
                    start -= 1
                assert start > 0, "hyperpipeline_io import not found above the writer"
                return "\n".join(
                    ast.unparse(entry) for entry in statements[start - 1:index + 1]
                )
    raise AssertionError("ILE output-format block not found")


class _CapturingNumpy:
    def __init__(self):
        self.rows = None

    @staticmethod
    def array(values):
        return values

    def savetxt(self, fname, rows):
        self.rows = rows


def _legacy_ile_row(monkeypatch, lambda1=0.0, lambda2=0.0, **flags):
    """Run the ILE .dat writer in legacy mode and return the row it emits."""
    monkeypatch.delenv("RIFT_HYPERPIPELINE_FORMAT", raising=False)
    options = SimpleNamespace(
        save_eccentricity=False,
        save_meanPerAno=False,
        save_EOB_parameters=False,
        save_hyperbolic=False,
        export_eos_index=False,
        pin_distance_to_sim=False,
    )
    for name, value in flags.items():
        assert hasattr(options, name)
        setattr(options, name, value)
    parameters = SimpleNamespace(
        s1x=0.1, s1y=0.2, s1z=0.3, s2x=0.4, s2y=0.5, s2z=0.6,
        lambda1=lambda1, lambda2=lambda2,
        eccentricity=0.3, meanPerAno=1.1,
        a6c=-55.0, E0=1.02, p_phi0=4.1, eos_table_index=7,
    )
    recorder = _CapturingNumpy()
    namespace = {
        "opts": options,
        "P": parameters,
        "numpy": recorder,
        "event_id": -1,
        "m1": 30.0,
        "m2": 20.0,
        "log_res": 12.0,
        "manual_avoid_overflow_logarithm": 0.0,
        "sqrt_var_over_res": 0.1,
        "sampler": SimpleNamespace(ntotal=100),
        "neff": 30,
        "pinned_params": {"distance": 410.0},
        "fname_output_txt": str(Path("unused.dat")),
    }
    exec(compile(_extract_ile_output_block(), "<ile-writer>", "exec"), namespace)
    assert recorder.rows is not None
    return list(recorder.rows[0])


def test_ile_hyperbolic_output_retains_eob_parameter(monkeypatch):
    row = _legacy_ile_row(monkeypatch, save_EOB_parameters=True, save_hyperbolic=True)

    assert len(row) == 9 + 1 + 2 + 4
    assert row[9] == -55.0
    assert row[10:12] == [1.02, 4.1]


def test_ile_legacy_row_keeps_every_enabled_group(monkeypatch):
    row = _legacy_ile_row(
        monkeypatch,
        lambda1=400.0,
        lambda2=800.0,
        save_eccentricity=True,
        save_meanPerAno=True,
        save_EOB_parameters=True,
        save_hyperbolic=True,
    )

    # lambda1 lambda2 | a6c | E0 p_phi0 | eccentricity meanPerAno, in CIP order
    assert len(row) == 9 + 2 + 1 + 2 + 2 + 4
    assert row[9:11] == [400.0, 800.0]
    assert row[11] == -55.0
    assert row[12:14] == [1.02, 4.1]
    assert row[14:16] == [0.3, 1.1]
    assert row[16] == 12.0  # lnL still lands 4 columns from the end


def test_ile_legacy_row_preserves_unflagged_layout(monkeypatch):
    assert len(_legacy_ile_row(monkeypatch)) == 13
    assert len(_legacy_ile_row(monkeypatch, pin_distance_to_sim=True)) == 14
    assert len(_legacy_ile_row(monkeypatch, lambda1=400.0, lambda2=800.0)) == 15
    assert len(
        _legacy_ile_row(
            monkeypatch, lambda1=400.0, lambda2=800.0, export_eos_index=True
        )
    ) == 16
    assert len(_legacy_ile_row(monkeypatch, save_eccentricity=True)) == 14
