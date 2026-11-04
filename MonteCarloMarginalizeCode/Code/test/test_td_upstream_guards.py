"""New upstream FD-only paths must refuse incompatible TD requests at startup."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize("flags,message", [
    (["--rotation-slow"], "td-toeplitz does not support --rotation-slow"),
    (["--freqresponse"], "td-toeplitz does not support --rotation-slow"),
    (["--log-noise-evidence-only"], "td-toeplitz does not support FD"),
    (["--log-noise-evidence-output", "noise.json"], "td-toeplitz does not support FD"),
])
def test_refuse_fd_only_path_with_td(flags, message, tmp_path):
    code = Path(__file__).resolve().parents[1]
    env = dict(os.environ, GW_SURROGATE="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(code)
    proc = subprocess.run([sys.executable, str(code / "bin/integrate_likelihood_extrinsic_batchmode"),
                           "--internal-ile-inner-product", "td-toeplitz", *flags],
                          cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120)
    assert proc.returncode != 0
    assert message in proc.stdout, proc.stdout[-3000:]
    assert not (tmp_path / "noise.json").exists()
    assert not (tmp_path / "log_noise_evidence.json").exists()


@pytest.mark.parametrize("flags", [
    ["--td-analysis-start", "0"], ["--td-analysis-end", ".1"],
    ["--td-acf-duration", "1"],
    ["--td-solver", "pcg"], ["--glitch-basis", "H1:0,.1,40,100"],
    ["--glitch-basis-file", "H1:unused.npz"],
])
def test_td_requests_are_not_ignored_by_fd(flags, tmp_path):
    code = Path(__file__).resolve().parents[1]
    env = dict(os.environ, GW_SURROGATE="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(code))
    proc = subprocess.run([sys.executable, str(code / "bin/integrate_likelihood_extrinsic_batchmode"),
                           *flags], cwd=tmp_path, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    assert proc.returncode != 0
    assert "--internal-ile-inner-product td-toeplitz" in proc.stdout, proc.stdout[-3000:]


@pytest.mark.parametrize("same_source,osg,existing_aux,refused", [
    (False, True, False, True),
    (True, True, False, False),
    (False, True, True, True),
    (False, False, False, False),
])
def test_pipeline_basis_transfer_names(tmp_path, same_source, osg, existing_aux, refused):
    """Execute the actual forwarding block without launching a pipeline."""
    import ast
    from types import SimpleNamespace
    code = Path(__file__).resolve().parents[1]
    tree = ast.parse((code / "bin/util_RIFT_pseudo_pipe.py").read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_as_list")
    block = next(n for n in tree.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Attribute) and n.test.attr == "internal_ile_inner_product")
    paths = []
    for detector in ("H1", "L1"):
        path = tmp_path / detector / "basis.npz"
        path.parent.mkdir()
        path.touch()
        paths.append(path)
    if same_source:
        paths[1] = paths[0]
    opts = SimpleNamespace(internal_ile_inner_product="td-toeplitz",
        internal_ile_td_solver=None, internal_ile_td_acf_duration=None,
        internal_ile_td_analysis_start=None, internal_ile_td_analysis_end=None,
        internal_ile_td_gap=None, internal_ile_glitch_basis=None,
        internal_ile_glitch_sigma=None, use_osg=osg,
        internal_ile_td_mref=None, internal_ile_td_piece=None,
        internal_ile_td_ifo_time_offset=None, internal_ile_td_acf_file=None,
        internal_ile_td_dense_max_n=None, internal_ile_td_u_stride=None,
        internal_ile_td_u_method=None, internal_ile_td_u_svd_tol=None, internal_ile_td_highpass=None,
        internal_ile_td_subsample=None,
        internal_ile_td_acf_cond=None, internal_ile_td_calibration_correction_type=None,
        internal_ile_td_calibration_fir_half_length=None,
        internal_ile_td_whiteness_check=False,
        ile_additional_files_to_transfer=str(paths[0]) if existing_aux else None,
        internal_ile_glitch_basis_file=(["L1:" + str(paths[1])] if existing_aux else
                                      [ifo + ":" + str(path) for ifo, path in zip(("H1", "L1"), paths)]))
    namespace = dict(opts=opts, os=os, sys=sys, line="")
    module = ast.fix_missing_locations(ast.Module(body=[helper, block], type_ignores=[]))
    if refused:
        with pytest.raises(SystemExit) as exc:
            exec(compile(module, "pipeline_td_forwarding", "exec"), namespace)
        assert exc.value.code == 1
    else:
        exec(compile(module, "pipeline_td_forwarding", "exec"), namespace)
        assert "--glitch-basis-file" in namespace["line"]



# --------------------------------------------------------------------------- --td-gap refusal
@pytest.mark.parametrize("inner_product", ["fd-diagonal", "td-toeplitz", "td-gs"])
def test_ile_refuses_td_gap(inner_product, tmp_path):
    """The gap-marginalized td-toeplitz likelihood is regularization-dependent: refuse it."""
    code = Path(__file__).resolve().parents[1]
    env = dict(os.environ, GW_SURROGATE="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(code))
    proc = subprocess.run([sys.executable, str(code / "bin/integrate_likelihood_extrinsic_batchmode"),
                           "--internal-ile-inner-product", inner_product, "--td-gap", "-.1,.1"],
                          cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120)
    assert proc.returncode != 0
    assert "--td-gap is disabled" in proc.stdout, proc.stdout[-3000:]
    assert "--internal-ile-inner-product td-gs" in proc.stdout
    assert "--td-mref" in proc.stdout


def test_pseudo_pipe_refuses_td_gap_on_the_command_line(tmp_path):
    code = Path(__file__).resolve().parents[1]
    env = dict(os.environ, GW_SURROGATE="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(code))
    proc = subprocess.run([sys.executable, str(code / "bin/util_RIFT_pseudo_pipe.py"),
                           "--internal-ile-td-gap=-.1,.1"],
                          cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120)
    assert proc.returncode == 1
    assert "--internal-ile-td-gap is disabled" in proc.stdout, proc.stdout[-3000:]
    assert "--internal-ile-td-piece" in proc.stdout


def test_pseudo_pipe_refuses_td_gap_from_an_ini():
    """An ini sets opts after parsing, so the ILE-line builder must refuse too."""
    import ast
    code = Path(__file__).resolve().parents[1]
    tree = ast.parse((code / "bin/util_RIFT_pseudo_pipe.py").read_text())
    msg = next(n for n in tree.body if isinstance(n, ast.Assign)
               and getattr(n.targets[0], "id", None) == "_TD_GAP_REFUSAL")
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_as_list")
    block = next(n for n in tree.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Attribute) and n.test.attr == "internal_ile_inner_product")

    class _Opts(object):
        internal_ile_inner_product = "td-toeplitz"
        internal_ile_td_gap = "-.1,.1"          # an ini's eval() of a quoted scalar
        internal_ile_td_whiteness_check = False

        def __getattr__(self, name):
            return None

    namespace = dict(opts=_Opts(), os=os, sys=sys, line="")
    module = ast.fix_missing_locations(ast.Module(body=[msg, helper, block], type_ignores=[]))
    with pytest.raises(SystemExit) as exc:
        exec(compile(module, "pipeline_td_gap", "exec"), namespace)
    assert exc.value.code == 1
    assert "--td-gap" not in namespace["line"]

# --------------------------------------------------------------------------- td-gs startup checks
def _run_batchmode(flags, tmp_path, keep_pythonpath=False):
    code = Path(__file__).resolve().parents[1]
    env = dict(os.environ, GW_SURROGATE="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # keep_pythonpath: the td-gs checks import td_gs_precompute, i.e. tdanalysis, which may
    # come from PYTHONPATH rather than from the environment.
    env["PYTHONPATH"] = (str(code) + os.pathsep + os.environ.get("PYTHONPATH", "")
                         if keep_pythonpath else str(code))
    return subprocess.run([sys.executable, str(code / "bin/integrate_likelihood_extrinsic_batchmode"),
                           *flags], cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120)


@pytest.mark.parametrize("inner_product", [[], ["--internal-ile-inner-product", "td-toeplitz"]])
@pytest.mark.parametrize("flags", [
    ["--td-mref", "60"], ["--td-acf-file", "H1:acf.npy"], ["--td-u-method", "batched"],
    ["--td-whiteness-check"], ["--td-subsample", "4"],
    ["--td-calibration-correction-type", "bilby"],
])
def test_td_gs_settings_require_td_gs(flags, inner_product, tmp_path):
    proc = _run_batchmode(inner_product + flags, tmp_path)
    assert proc.returncode != 0
    assert "TD-GS settings require --internal-ile-inner-product td-gs" in proc.stdout, proc.stdout[-3000:]


def test_td_gs_requires_time_marginalization(tmp_path):
    proc = _run_batchmode(["--internal-ile-inner-product", "td-gs", "--vectorized"], tmp_path)
    assert proc.returncode != 0
    assert "td-gs requires --time-marginalization" in proc.stdout, proc.stdout[-3000:]


def _tdanalysis_or_skip():
    try:
        from RIFT.likelihood import td_gs_precompute  # noqa: F401
    except ImportError as exc:
        pytest.skip("td_gs_precompute not importable (tdanalysis >= 1.1): %r" % (exc,))


@pytest.mark.parametrize("flags", [
    ["--approximant", "IMRPhenomD"], ["--approximant", "IMRPhenomXPHM"],
    ["--use-gwsignal", "--approximant", "SEOBNRv5PHM"], ["--rom-group", "x"], ["--nr-lookup"],
])
def test_td_gs_refuses_non_native_td_generators(flags, tmp_path):
    _tdanalysis_or_skip()
    proc = _run_batchmode(["--internal-ile-inner-product", "td-gs", "--time-marginalization",
                           "--vectorized", *flags], tmp_path, keep_pythonpath=True)
    assert proc.returncode != 0
    assert "td-gs requires an untapered native TD waveform generator" in proc.stdout, proc.stdout[-3000:]


def test_td_gs_native_td_generator_passes_the_startup_checks(tmp_path):
    # TaylorT4 (the default) and NRSur7dq4 are time-domain models: the startup checks let
    # them through (the run then stops later, for want of data).
    _tdanalysis_or_skip()
    for flags in ([], ["--approximant", "NRSur7dq4"]):
        proc = _run_batchmode(["--internal-ile-inner-product", "td-gs", "--time-marginalization",
                               "--vectorized", *flags], tmp_path, keep_pythonpath=True)
        assert "untapered native TD waveform generator" not in proc.stdout, proc.stdout[-3000:]
        assert "td-gs requires --time-marginalization" not in proc.stdout
        assert "TD-GS settings require" not in proc.stdout


def _td_gs_block_statements(names):
    """Statements of the driver's `if use_td_gs:` block that assign or fill `names`."""
    import ast
    code = Path(__file__).resolve().parents[1]
    tree = ast.parse((code / "bin/integrate_likelihood_extrinsic_batchmode").read_text())
    block = next(n for n in tree.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == "use_td_gs")
    picked = []
    for stmt in block.body:
        targets = []
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
        elif isinstance(stmt, ast.For):   # the loop that fills names[...]
            targets = [node.value.id for node in ast.walk(stmt)
                       if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
                       and isinstance(node.value, ast.Name)]
        if any(t in names for t in targets):
            picked.append(stmt)
    return ast.Module(body=picked, type_ignores=[])


@pytest.mark.parametrize("specs,ok", [
    (["H1:0.01", "L1:-0.002"], True),
    (["V1:0.01"], False),          # no such channel
    (["h1:0.01"], False),          # names are case sensitive
    (["H1:0.01", "H1:0.02"], False),
])
def test_td_ifo_time_offset_detectors(specs, ok):
    from types import SimpleNamespace
    code = compile(_td_gs_block_statements({"td_ifo_offset"}), "batchmode", "exec")
    ns = dict(opts=SimpleNamespace(td_ifo_time_offset=specs), data_dict={"H1": None, "L1": None})
    if ok:
        exec(code, ns)
        assert ns["td_ifo_offset"] == {s.split(":")[0]: float(s.split(":")[1]) for s in specs}
    else:
        with pytest.raises(ValueError, match="unknown or duplicate --td-ifo-time-offset detector"):
            exec(code, ns)


@pytest.mark.parametrize("ifos,ok", [(["H1", "L1"], True), (["V1"], False), (["L1", "L1"], False)])
def test_td_acf_file_detectors(ifos, ok, tmp_path):
    from types import SimpleNamespace
    acf = tmp_path / "acf.npy"
    acf.touch()
    code = compile(_td_gs_block_statements({"td_acf_paths"}), "batchmode", "exec")
    ns = dict(opts=SimpleNamespace(td_acf_file=["%s:%s" % (ifo, acf) for ifo in ifos]),
              data_dict={"H1": None, "L1": None}, os=os, sys=sys)
    if ok:
        exec(code, ns)
        assert sorted(ns["td_acf_paths"]) == sorted(ifos)
    else:
        with pytest.raises(ValueError, match="unknown or duplicate --td-acf-file detector"):
            exec(code, ns)


# --------------------------------------------------------------------------- td-gs distmarg wrapper
def _distmarg_loglikelihood(time_interp, global_norm=False):
    """The driver's distmarg_loglikelihood, with the distance-marginalization table stubbed
    out (lnI = 0, exponent_max = x0 * rho_sq / 2)."""
    import ast
    from types import SimpleNamespace
    import numpy as np
    code = Path(__file__).resolve().parents[1]
    tree = ast.parse((code / "bin/integrate_likelihood_extrinsic_batchmode").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "distmarg_loglikelihood")
    ns = dict(ile_td_gs_contexts={"H1": None}, xpy_default=np,
              opts=SimpleNamespace(_noloop_time_interp=time_interp,
                                   calibration_envelope_directory="cal" if global_norm else None,
                                   calibration_global_norm=global_norm),
              x0_to_s=lambda x0: x0, b_to_t=lambda b: b, smin=-np.inf, smax=np.inf, tmax=np.inf,
              intp=lambda s, t: 0. * s, exponent_max=lambda x0, rho_sq: 0.5 * x0 * x0 * rho_sq)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "batchmode", "exec"), ns)
    return ns["distmarg_loglikelihood"]


def test_distmarg_wrapper_td_gs_norm_checks():
    import numpy as np
    f = _distmarg_loglikelihood("nearest")
    kappa = np.array([[2., 0., 1.]])
    rho = np.array([[4., 0., 1.]])
    np.testing.assert_array_equal(f(kappa, rho), [[0.5, 0., 0.5]])   # zero norm: lnL = 0
    with pytest.raises(ValueError, match="negative template norm"):
        f(kappa, np.array([[4., 0., -1e-3]]))
    with pytest.raises(ValueError, match="zero template norm has nonzero data overlap"):
        f(np.array([[2., 1e-3, 1.]]), rho)
    # a negative norm at rounding level is not an error; that bin counts as no template
    np.testing.assert_array_equal(f(kappa, np.array([[4., -1e-15, 1.]])), [[0.5, 0., 0.5]])


def test_distmarg_wrapper_td_gs_cubic_keeps_the_mask():
    # The cubic stencil can take an interpolated norm below zero (at the zero extension past
    # the buffer edge); those bins count as no template, as before.
    import numpy as np
    f = _distmarg_loglikelihood("cubic")
    np.testing.assert_array_equal(f(np.array([[2., 1e-3, 1.]]), np.array([[4., -1e-3, 1.]])),
                                  [[0.5, 0., 0.5]])


def test_distmarg_wrapper_td_gs_global_norm_calibration():
    # --calibration-global-norm: the norm is the uncalibrated template's, the overlap the
    # FIR-filtered template's, so a zero norm may meet a nonzero overlap; a negative norm is
    # still an error.
    import numpy as np
    f = _distmarg_loglikelihood("nearest", global_norm=True)
    np.testing.assert_array_equal(f(np.array([[2., 1e-3, 1.]]), np.array([[4., 0., 1.]])),
                                  [[0.5, 0., 0.5]])
    with pytest.raises(ValueError, match="negative template norm"):
        f(np.array([[2., 0., 1.]]), np.array([[4., 0., -1e-3]]))


@pytest.mark.parametrize("route,override,expected", [
    ("td-gs", None, {"H1": "template", "V1": "template"}),
    ("fd-diagonal", None, {"H1": "data", "V1": "template"}),
    ("fd-diagonal", "template", {"H1": "template", "V1": "template"}),
])
def test_calibration_prior_conversion_happens_once(route, override, expected):
    """Execute the ILE resolver with the shared loader: TD FIR gets raw priors."""
    import ast
    from types import SimpleNamespace
    from RIFT.calmarg.calibration import correction_types_for_ifos
    code = Path(__file__).resolve().parents[1]
    tree = ast.parse((code / "bin/integrate_likelihood_extrinsic_batchmode").read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_calibration_envelope_types")
    ns = dict(correction_types_for_ifos=correction_types_for_ifos)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "calibration_route", "exec"), ns)
    opts = SimpleNamespace(internal_ile_inner_product=route, calibration_correction_type=override)
    assert ns["_calibration_envelope_types"](opts, ["H1", "V1"]) == expected


def test_td_gs_refuses_fd_calibration_convention_flag(tmp_path):
    proc = _run_batchmode(["--internal-ile-inner-product", "td-gs",
                          "--calibration-correction-type", "data"], tmp_path)
    assert proc.returncode != 0
    assert "td-gs uses --td-calibration-correction-type" in proc.stdout, proc.stdout[-3000:]


@pytest.mark.parametrize("td_gs,conjugate,expected", [
    (True, True, True), (True, False, False), (False, True, False),
])
def test_calibration_phase_transport_respects_route(td_gs, conjugate, expected):
    import ast
    from types import SimpleNamespace
    code = Path(__file__).resolve().parents[1]
    tree = ast.parse((code / "bin/integrate_likelihood_extrinsic_batchmode").read_text())
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                           and t.value.id == "extra_kwargs" and isinstance(t.slice, ast.Constant)
                           and t.slice.value == "calibration_conjugate" for t in n.targets)]
    assert len(assignments) == 1
    ns = dict(use_td_gs=td_gs, opts=SimpleNamespace(calibration_conjugate_phase=conjugate), extra_kwargs={})
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "calibration_phase", "exec"), ns)
    assert ns["extra_kwargs"]["calibration_conjugate"] is expected
