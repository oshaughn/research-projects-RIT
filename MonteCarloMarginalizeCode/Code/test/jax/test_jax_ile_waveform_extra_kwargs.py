"""JAX ILE forwards the waveform backend options numpy ILE forwards.

``--internal-waveform-extra-kwargs``, ``--internal-waveform-extra-lalsuite-args``
and ``--use-gwsignal-lmax-nyquist`` act inside the shared precompute.  The JAX
driver accepted them as compatibility no-ops, so a ``--use-jax-ile`` run with the
production SEOBNRv5PHM antisymmetric-mode settings generated waveforms without
them.  Each case compares the JAX driver's ``extra_waveform_kwargs`` with the one
numpy ILE assembles for the same command line (extracted from the numpy driver by
``test_gwsignal_ile_options.assembled_options``).
"""
import ast
import contextlib
import importlib.machinery
import importlib.util
import io
import os

import pytest

lal = pytest.importorskip("lal")
pytest.importorskip("lalsimulation.gwsignal")

from RIFT import lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood.jax_ile import core as core_mod
from RIFT.physics import GWSignal as rgws

_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_DRIVER = os.path.join(_CODE, "bin", "integrate_likelihood_extrinsic_jax")
_NUMPY_ASSEMBLY = os.path.join(_CODE, "test", "test_gwsignal_ile_options.py")
_NUMPY_ILE = os.path.join(_CODE, "bin", "integrate_likelihood_extrinsic_batchmode")

PRODUCTION = ("{'antisymmetric_modes_hm': True, 'enable_antisymmetric_modes': True, "
              "'lmax_nyquist': 1}")


def _load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load("_jax_waveform_extra_driver", _DRIVER)


@pytest.fixture(scope="module")
def numpy_ile():
    return _load("_numpy_ile_waveform_assembly", _NUMPY_ASSEMBLY)


@pytest.fixture
def restore_gh_state(monkeypatch):
    # check_critical_and_report applies the distance-GH node count globally.
    monkeypatch.delenv("JAX_ILE_DISTMARG_GH", raising=False)
    saved = core_mod._DISTMARG_GH_N
    yield
    core_mod._DISTMARG_GH_N = saved


def _jax_extra(driver, argv):
    opts, _ = driver.build_parser().parse_args(list(argv))
    return opts, driver._waveform_precompute_kwargs(opts)


def _check(driver, argv):
    """Return (refused, text) for parsing plus check_critical_and_report."""
    optp = driver.build_parser()
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            opts, _ = optp.parse_args(list(argv))
            driver.check_critical_and_report(opts, optp)
        except SystemExit:
            return True, out.getvalue()
    return False, out.getvalue()


@pytest.mark.parametrize("double_quoted", [False, True])
@pytest.mark.parametrize("direct_lmax", [None, 0, 5])
def test_gwsignal_route_matches_numpy_ile(driver, numpy_ile, double_quoted,
                                          direct_lmax):
    value = repr(PRODUCTION) if double_quoted else PRODUCTION
    argv = ["--use-gwsignal", "--approximant", "SEOBNRv5PHM",
            "--internal-waveform-fd-L-frame",
            "--internal-waveform-extra-kwargs", value]
    if direct_lmax is not None:
        argv += ["--use-gwsignal-lmax-nyquist", str(direct_lmax)]
    _, got = _jax_extra(driver, argv)
    _, want = numpy_ile.assembled_options(argv)
    assert got["extra_waveform_kwargs"] == want
    assert got["extra_waveform_kwargs"]["gwsignal_extra_kwargs"] == {
        "antisymmetric_modes_hm": True, "enable_antisymmetric_modes": True,
        "lmax_nyquist": 1}
    assert got["use_gwsignal"] is True
    assert got["use_gwsignal_approx"] == "SEOBNRv5PHM"


def test_production_options_reach_gwsignal_generator(driver, monkeypatch):
    opts, got = _jax_extra(driver, [
        "--use-gwsignal", "--approximant", "SEOBNRv5PHM",
        "--internal-waveform-extra-kwargs", PRODUCTION,
        "--use-gwsignal-lmax-nyquist", "5"])
    captured = {}

    class AtGenerator(Exception):
        pass

    def capture(params, generator):
        captured.update(params)
        raise AtGenerator

    monkeypatch.setattr(rgws.wfm, "GenerateTDModes", capture)
    monkeypatch.setattr(rgws.gws.models, "gwsignal_get_waveform_generator",
                        lambda name: name)
    monkeypatch.setattr(fl, "has_GWS", True)
    monkeypatch.setattr(fl, "rgws", rgws, raising=False)
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = 60 * lal.MSUN_SI, 40 * lal.MSUN_SI
    p.fmin = p.fref = 30
    p.deltaT, p.deltaF = 1 / 2048, 1 / 4
    p.dist = 500e6 * lal.PC_SI
    with pytest.raises(AtGenerator):
        fl.internal_hlm_generator(
            p, 4, use_gwsignal=got["use_gwsignal"],
            use_gwsignal_approx=got["use_gwsignal_approx"],
            extra_waveform_kwargs=got["extra_waveform_kwargs"], quiet=True)
    assert captured["antisymmetric_modes_hm"] is True
    assert captured["enable_antisymmetric_modes"] is True
    assert captured["lmax_nyquist"] == 1  # the explicit dictionary wins


def test_non_gwsignal_dictionary_merges_like_numpy_ile(driver, numpy_ile):
    argv = ["--approximant", "IMRPhenomXPHM",
            "--internal-waveform-extra-kwargs", "{'fd_standoff_factor': 0.8}",
            "--use-gwsignal-lmax-nyquist", "2"]
    _, got = _jax_extra(driver, argv)
    _, want = numpy_ile.assembled_options(argv)
    assert got["extra_waveform_kwargs"] == want
    assert got["extra_waveform_kwargs"]["fd_standoff_factor"] == 0.8
    assert got["extra_waveform_kwargs"]["lmax_nyquist"] == 2


def test_non_gwsignal_dictionary_lmax_beats_flag_like_numpy_ile(driver, numpy_ile):
    argv = ["--approximant", "IMRPhenomXPHM",
            "--internal-waveform-extra-kwargs", "{'lmax_nyquist': 1}",
            "--use-gwsignal-lmax-nyquist", "3"]
    _, got = _jax_extra(driver, argv)
    _, want = numpy_ile.assembled_options(argv)
    assert got["extra_waveform_kwargs"] == want
    assert got["extra_waveform_kwargs"]["lmax_nyquist"] == 1


@pytest.mark.parametrize("flag", ["--internal-waveform-extra-kwargs",
                                  "--internal-waveform-extra-lalsuite-args"])
def test_empty_dictionary_string_is_unset_like_numpy_ile(driver, numpy_ile,
                                                        restore_gh_state, flag):
    argv = ["--approximant", "IMRPhenomXPHM", flag, ""]
    refused, text = _check(driver, argv)
    assert not refused, text
    _, got = _jax_extra(driver, argv)
    _, want = numpy_ile.assembled_options(argv)
    assert got["extra_waveform_kwargs"] == want


def test_dictionary_may_use_numpy_like_numpy_ile(driver, restore_gh_state):
    # numpy ILE evaluates in a module namespace with ``numpy`` imported.
    argv = ["--approximant", "IMRPhenomXPHM",
            "--internal-waveform-extra-kwargs", "{'f_cut': numpy.inf}"]
    refused, text = _check(driver, argv)
    assert not refused, text
    _, got = _jax_extra(driver, argv)
    assert got["extra_waveform_kwargs"]["f_cut"] == float("inf")


def test_lalsuite_args_forwarded_like_numpy_ile(driver, numpy_ile):
    argv = ["--approximant", "IMRPhenomXPHM",
            "--internal-waveform-extra-lalsuite-args", "{'PhenomXHMReleaseVersion': 122019}"]
    _, got = _jax_extra(driver, argv)
    _, want = numpy_ile.assembled_options(argv)
    assert got["extra_waveform_kwargs"] == want
    assert got["extra_waveform_kwargs"]["extra_waveform_args"] == {
        "PhenomXHMReleaseVersion": 122019}


def test_lalsuite_args_with_gwsignal_refused(driver, restore_gh_state):
    refused, text = _check(driver, [
        "--use-gwsignal", "--approximant", "SEOBNRv5PHM",
        "--internal-waveform-extra-lalsuite-args", "{'a': 1}"])
    assert refused
    assert "--internal-waveform-extra-lalsuite-args cannot be used with --use-gwsignal" in text


@pytest.mark.parametrize("flag", ["--internal-waveform-extra-kwargs",
                                  "--internal-waveform-extra-lalsuite-args"])
@pytest.mark.parametrize("value", ["['lmax_nyquist', 1]", "{'unterminated': "])
def test_bad_dictionary_refused_before_data(driver, restore_gh_state, flag, value):
    refused, text = _check(driver, ["--approximant", "IMRPhenomXPHM", flag, value])
    assert refused
    assert flag in text


@pytest.mark.parametrize("value", ["1.5", "abc"])
def test_non_integer_lmax_nyquist_refused_at_parse(driver, value):
    # numpy ILE declares type=int, so optparse refuses these values there too.
    with pytest.raises(SystemExit), \
            contextlib.redirect_stderr(io.StringIO()):
        driver.build_parser().parse_args(
            ["--use-gwsignal-lmax-nyquist", value])


def _numpy_value_types():
    """Long option -> (type, choices) for numpy ILE's single-value options."""
    tree = ast.parse(open(_NUMPY_ILE).read())
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_option"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant)
                 and isinstance(a.value, str) and a.value.startswith("--")]
        kw = {k.arg: k.value for k in node.keywords}
        action = kw.get("action")
        if not names or (action is not None and
                         ast.literal_eval(action) != "store"):
            continue
        typ = kw.get("type")
        if typ is None:
            typ = "string"
        elif isinstance(typ, ast.Name):  # type=int / type=float
            typ = typ.id
        else:
            typ = ast.literal_eval(typ)
        choices = (sorted(ast.literal_eval(kw["choices"]))
                   if "choices" in kw else None)
        out[names[0]] = ({"str": "string"}.get(typ, typ), choices)
    return out


# Options this driver defines itself with a wider or narrower value set than
# numpy ILE, on purpose.  float --srate accepts every integer numpy ILE does.
_NATIVE_TYPE_EXCEPTIONS = {"--srate", "--time-marginalization-quadrature"}


def test_option_value_types_match_numpy_ile(driver):
    """Hot swap: each option parses the same values in both drivers."""
    numpy_types = _numpy_value_types()
    assert len(numpy_types) > 100  # the AST walk found numpy ILE's options
    mismatched = []
    for option in driver.build_parser()._get_all_options():
        if option.action != "store":
            continue
        for name in option._long_opts:
            if name not in numpy_types or name in _NATIVE_TYPE_EXCEPTIONS:
                continue
            got = (option.type or "string",
                   sorted(option.choices) if option.choices else None)
            if got != numpy_types[name]:
                mismatched.append((name, numpy_types[name], got))
    assert not mismatched, mismatched


@pytest.mark.parametrize("argv", [["--e-freq", "1.0"], ["--srate-internal", "4096.5"],
                                  ["--time-posterior-export", "bogus"]])
def test_values_numpy_ile_refuses_are_refused_at_parse(driver, argv):
    with pytest.raises(SystemExit), \
            contextlib.redirect_stderr(io.StringIO()):
        driver.build_parser().parse_args(argv)


def test_compat_numbers_reach_the_driver_as_numbers(driver, restore_gh_state):
    # "1" as a string failed the "not in (None, 1)" calibration check, and
    # the ignore threshold reached the precompute as a string.
    refused, text = _check(driver, ["--approximant", "IMRPhenomXPHM",
                                    "--calibration-n-realizations", "1",
                                    "--internal-precompute-ignore-threshold", "1e-4"])
    assert not refused, text
    _, got = _jax_extra(driver, ["--approximant", "IMRPhenomXPHM",
                                 "--internal-precompute-ignore-threshold", "1e-4"])
    assert got["ignore_threshold"] == 1e-4


@pytest.mark.parametrize("argv", [
    ["--use-gwsignal", "--approximant", "SEOBNRv5PHM",
     "--internal-waveform-extra-kwargs", PRODUCTION,
     "--use-gwsignal-lmax-nyquist", "1"],
    ["--approximant", "IMRPhenomXPHM",
     "--internal-waveform-extra-lalsuite-args", "{'a': 1}"],
], ids=["gwsignal", "lalsuite"])
def test_forwarded_options_not_reported_ignored(driver, restore_gh_state, argv):
    refused, text = _check(driver, argv)
    assert not refused, text
    for flag in ("--internal-waveform-extra-kwargs",
                 "--internal-waveform-extra-lalsuite-args",
                 "--use-gwsignal-lmax-nyquist"):
        assert flag not in text


@pytest.mark.parametrize("policy", [None, "legacy", "native"])
def test_conditioning_policy_matches_numpy_ile(driver, numpy_ile, policy):
    argv = ["--use-gwsignal", "--approximant", "SEOBNRv5PHM"]
    if policy is not None:
        argv += ["--gwsignal-mode-conditioning", policy]
    _, got = _jax_extra(driver, argv)
    _, want = numpy_ile.assembled_options(argv)
    assert got["extra_waveform_kwargs"] == want
    assert want["gwsignal_mode_conditioning"] == (policy or "legacy")


def test_native_conditioning_without_gwsignal_refused(driver, restore_gh_state):
    refused, text = _check(driver, ["--gwsignal-mode-conditioning", "native"])
    assert refused
    assert "native requires --use-gwsignal" in text


def test_native_unconditioned_request_refused(driver, restore_gh_state):
    refused, text = _check(driver, ["--use-gwsignal", "--gwsignal-mode-conditioning", "native",
                                  "--internal-waveform-fd-no-condition"])
    assert refused
    assert "conflicts with --internal-waveform-fd-no-condition" in text
