"""Exercise real CIP startup, beyond --help, without a costly inference run.

A deliberately absent input file is the boundary: ordinary BBH arguments must
reach data loading, while invalid EOS options must fail with their own errors.
Do not mock the parser/validator: their integration broke O4c 0.0.17.14rc2.
"""
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "util_ConstructIntrinsicPosterior_GenericCoordinates.py"


def run_cip(tmp_path, extra):
    missing = tmp_path / "intentionally-absent-composite.dat"
    assert not missing.exists()
    env = dict(os.environ, GW_SURROGATE="", MPLBACKEND="Agg",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fname", str(missing),
         "--parameter", "mc", "--parameter", "eta", "--no-plots", *extra],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    return result, missing


@pytest.mark.parametrize("extra", [
    [],
    ["--use-precessing", "--fit-method", "rf", "--sampler-method", "AV",
     "--internal-use-lnL", "--not-worker"],
])
def test_bbh_startup_reaches_data_loading(tmp_path, extra):
    result, missing = run_cip(tmp_path, extra)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "FileNotFoundError:" in result.stderr, output
    assert str(missing) in result.stderr, output
    assert "AttributeError:" not in result.stderr, output


@pytest.mark.parametrize("extra,message", [
    (["--using-eos-branch", "1"],
     "--using-eos-branch also requires --using-eos"),
    (["--using-eos-dirty-phase-transitions"],
     "require --using-eos lalsim_file:<path>"),
    (["--using-eos-extended-family"],
     "require --using-eos lalsim_file:<path>"),
])
def test_invalid_eos_options_fail_before_data_loading(tmp_path, extra, message):
    result, _ = run_cip(tmp_path, extra)
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "ValueError:" in result.stderr and message in result.stderr, output
    assert "FileNotFoundError:" not in result.stderr, output
    assert "AttributeError:" not in result.stderr, output


def test_o4d_hyperprior_option_remains_unsupported(tmp_path):
    result, _ = run_cip(tmp_path, ["--using-eos-for-prior"])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "unrecognized arguments: --using-eos-for-prior" in result.stderr
