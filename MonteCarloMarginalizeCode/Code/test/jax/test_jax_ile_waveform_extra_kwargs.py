"""JAX ILE forwards the waveform backend options numpy ILE forwards.

``--internal-waveform-extra-kwargs``, ``--internal-waveform-extra-lalsuite-args``
and ``--use-gwsignal-lmax-nyquist`` act inside the shared precompute.  The JAX
driver accepted them as compatibility no-ops, so a ``--use-jax-ile`` run with the
production SEOBNRv5PHM antisymmetric-mode settings generated waveforms without
them.  Each case compares the JAX driver's ``extra_waveform_kwargs`` with the one
numpy ILE assembles for the same command line (extracted from the numpy driver by
``test_gwsignal_ile_options.assembled_options``).
"""
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
    optp = driver.build_parser()
    opts, _ = optp.parse_args(list(argv))
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
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


def test_forwarded_options_not_reported_ignored(driver, restore_gh_state):
    refused, text = _check(driver, [
        "--use-gwsignal", "--approximant", "SEOBNRv5PHM",
        "--internal-waveform-extra-kwargs", PRODUCTION,
        "--use-gwsignal-lmax-nyquist", "1"])
    assert not refused, text
    assert "--internal-waveform-extra-kwargs" not in text
    assert "--use-gwsignal-lmax-nyquist" not in text
