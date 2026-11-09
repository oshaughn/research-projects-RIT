"""Run the pseudo-pipelines' ILE dispatch without staging detector data."""

import ast
import configparser
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

lalsim = pytest.importorskip("lalsimulation")
from RIFT import lalsimutils


BIN = Path(__file__).resolve().parents[1] / "bin"
SCRIPTS = ["util_RIFT_pseudo_pipe.py", "util_RIFT_pseudo_pipe_lowlatency.py"]


def run_block(script, marker, scope):
    path = BIN / script
    tree = ast.parse(path.read_text())
    block = next(node for node in tree.body if isinstance(node, ast.If)
                 and any(isinstance(child, ast.Constant) and child.value == marker
                         for child in ast.walk(node)))
    exec(compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec"), scope)


def ile_args(script, name, *, surrogate=False, gwsignal=False, manual_gwsignal=False):
    options = SimpleNamespace(approx=name, use_gwsurrogate=surrogate, use_gwsignal=gwsignal)
    scope = dict(opts=options, line="", sur_location_prefix="/models",
                 lalsimutils=lalsimutils, sys=sys, manual_gwsignal=manual_gwsignal)
    run_block(script, " Unknown approx ", scope)
    return shlex.split(scope["line"])


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("name", ["IMRPhenomXPHM", "IMRPhenomXPNR", "IMRPhenomPv2_NRTidalv2",
                                  "NRSur7dq4", "TaylorF2threePointFivePN"])
def test_defined_native_waveform_is_forwarded(script, name):
    if name == "IMRPhenomXPNR" and not hasattr(lalsim, name):
        pytest.skip("installed LALSimulation predates IMRPhenomXPNR")
    assert ile_args(script, name) == ["--approx", name]


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("name", ["NotAWaveform", "IMRPhenomXPHMtypo", "SEOBNRv4typo"])
def test_unknown_native_waveform_is_rejected(script, name):
    with pytest.raises(SystemExit) as error:
        ile_args(script, name)
    assert error.value.code == 1


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("name", ["NRHybSur3dq8", "NRSur7dq2", "NRSur7dq4"])
def test_explicit_surrogate_keeps_its_rom_arguments(script, name):
    assert ile_args(script, name, surrogate=True) == [
        "--rom-group", "/models", "--rom-param", name + ".h5", "--approx", name,
    ]


def test_explicit_gwsignal_keeps_precedence():
    assert ile_args(SCRIPTS[0], "NRSur7dq4", gwsignal=True, surrogate=True) == [
        "--use-gwsignal", "--approx", "NRSur7dq4",
    ]


@pytest.mark.parametrize("script", SCRIPTS)
def test_manually_selected_gwsignal_accepts_backend_name(script):
    assert ile_args(script, "SEOBNRv5PHM", manual_gwsignal=True) == ["--approx", "SEOBNRv5PHM"]


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("name,manual_gwsignal,expected", [
    ('"TaylorF2threePointFivePN"', False, "TaylorF2"),
    ("TaylorF2bogus", False, None),
    ("SEOBNRv5PHM", True, "SEOBNRv5PHM"),
])
def test_ini_uses_the_selected_backend(script, name, manual_gwsignal, expected, tmp_path):
    path = tmp_path / "run.ini"
    path.write_text(f"[engine]\napprox = {name}\n")
    config = configparser.ConfigParser()
    config.read(path)
    scope = dict(opts=SimpleNamespace(approx=None, use_ini=str(path), use_gwsignal=False), config=config,
                 ConfigParser=configparser, lalsim=lalsim, sys=sys, manual_gwsignal=manual_gwsignal)
    if expected is None:
        with pytest.raises(RuntimeError):
            run_block(script, " Approximant provided in ini file: ", scope)
    else:
        run_block(script, " Approximant provided in ini file: ", scope)
        assert scope["opts"].approx == expected
