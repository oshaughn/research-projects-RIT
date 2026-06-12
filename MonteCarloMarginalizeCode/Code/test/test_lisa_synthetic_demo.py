"""Tests for the LISA synthetic-data demo analysis surface."""

import os
import subprocess

import numpy as np

from RIFT.misc import hyperpipeline_io


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEMO_DIR = os.path.join(REPO_ROOT, "MonteCarloMarginalizeCode", "Code", "demo", "rift", "lisa")
MAKE_INPUTS = os.path.join(DEMO_DIR, "make_synthetic_lisa_inputs.py")
RUN_ILE = os.path.join(DEMO_DIR, "run_lisa_synthetic_ile.sh")


def _read_env_file(path):
    values = {}
    with open(path) as inp:
        for line in inp:
            key, value = line.strip().split("=", 1)
            values[key] = value
    return values


def test_lisa_synthetic_input_builder_writes_analysis_products(tmp_path):
    subprocess.run(
        [
            MAKE_INPUTS,
            "--output-directory",
            os.fspath(tmp_path),
            "--duration",
            "1024",
            "--deltaT",
            "4",
        ],
        check=True,
    )

    expected = {
        "A-fake_strain-1000000-10000.h5",
        "E-fake_strain-1000000-10000.h5",
        "T-fake_strain-1000000-10000.h5",
        "A_psd.xml.gz",
        "E_psd.xml.gz",
        "T_psd.xml.gz",
        "lisa.cache",
        "synthetic-params.env",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}

    env = _read_env_file(tmp_path / "synthetic-params.env")
    assert env["SRATE"] == "0.25"
    assert env["DELTA_T"] == "4.0"
    assert env["DURATION"] == "1024.0"


def test_lisa_synthetic_demo_wires_real_analysis_arguments(tmp_path):
    env = os.environ.copy()
    env["RIFT_LISA_WORKDIR"] = os.fspath(tmp_path)
    env["RIFT_LISA_RUN_ILE"] = "0"
    subprocess.run([RUN_ILE, "--duration", "1024"], check=True, env=env)

    _, columns = hyperpipeline_io.read_table(os.fspath(tmp_path / "proposed-grid.dat"))
    assert "ecliptic_longitude" in columns
    assert "ecliptic_latitude" in columns

    ile_args = (tmp_path / "args_ile.txt").read_text()
    assert "--zero-likelihood" not in ile_args
    assert "--time-marginalization" in ile_args
    assert "--cache-file {}".format(tmp_path / "lisa.cache") in ile_args
    assert "--psd-file A={}".format(tmp_path / "A_psd.xml.gz") in ile_args
    assert "--psd-file A=A_psd.xml.gz" not in ile_args
    assert "--srate 0.25" in ile_args
    assert "--data-integration-window-half 8.0" in ile_args

    transfer_files = (tmp_path / "helper_transfer_files.txt").read_text().splitlines()
    assert os.fspath(tmp_path / "lisa.cache") in transfer_files
    assert os.fspath(tmp_path / "A_psd.xml.gz") in transfer_files


def test_lisa_synthetic_demo_runs_real_ile(tmp_path):
    env = os.environ.copy()
    env["RIFT_LISA_WORKDIR"] = os.fspath(tmp_path)
    subprocess.run([RUN_ILE, "--duration", "1024"], check=True, env=env)

    output = np.loadtxt(tmp_path / "lisa_ile_0_.dat")
    assert output.shape == (15,)
    assert np.isfinite(output[11])
    assert output[13] > 0
