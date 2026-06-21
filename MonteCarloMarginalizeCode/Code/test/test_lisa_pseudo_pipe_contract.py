#!/usr/bin/env python
"""Contract tests for the LISA known-sky pseudo_pipe surface."""

import os
import subprocess
import sys

import pytest

from RIFT.misc import hyperpipeline_io


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "MonteCarloMarginalizeCode", "Code")
PSEUDO_PIPE = os.path.join(CODE_DIR, "bin", "util_RIFT_pseudo_pipe.py")


def test_lisa_known_sky_pseudo_pipe_renders_cepp_surface(tmp_path):
    rundir = tmp_path / "pseudo_lisa"
    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = os.path.join(CODE_DIR, "bin") + os.pathsep + env.get("PATH", "")

    cmd = [
        sys.executable,
        PSEUDO_PIPE,
        "--lisa-known-sky",
        "--use-rundir",
        os.fspath(rundir),
        "--approx",
        "IMRPhenomD",
        "--event-time",
        "1234.5",
        "--ecliptic-longitude",
        "1.25",
        "--ecliptic-latitude",
        "-0.4",
        "--lisa-cache-file",
        os.fspath(tmp_path / "lisa.cache"),
        "--lisa-channel-name",
        "A=SYNTH",
        "--lisa-channel-name",
        "E=SYNTH",
        "--lisa-channel-name",
        "T=SYNTH",
        "--lisa-psd-file",
        "A={}".format(tmp_path / "A_psd.xml.gz"),
        "--lisa-psd-file",
        "E={}".format(tmp_path / "E_psd.xml.gz"),
        "--lisa-psd-file",
        "T={}".format(tmp_path / "T_psd.xml.gz"),
        "--lisa-srate",
        "0.25",
        "--lisa-fmin-template",
        "0.001",
        "--lisa-fmax",
        "0.125",
        "--lisa-grid-size",
        "1",
        "--lisa-n-iterations",
        "1",
        "--lisa-n-samples-per-job",
        "1",
        "--internal-ile-request-memory",
        "1024",
        "--internal-cip-request-memory",
        "1024",
    ]
    subprocess.run(cmd, check=True, env=env)

    assert (rundir / "args_ile.txt").exists()
    assert (rundir / "args_cip_list.txt").exists()
    assert (rundir / "helper_transfer_files.txt").exists()
    assert (rundir / "ILE.sub").exists()
    assert (rundir / "CIP.sub").exists()

    grid, columns = hyperpipeline_io.read_table(os.fspath(rundir / "proposed-grid.dat"))
    assert grid.shape == (1,)
    assert "ecliptic_longitude" in columns
    assert "ecliptic_latitude" in columns
    assert grid["ecliptic_longitude"][0] == 1.25
    assert grid["ecliptic_latitude"][0] == -0.4

    ile_args = (rundir / "args_ile.txt").read_text()
    assert "--LISA" in ile_args
    assert "--lisa-fixed-sky 1" in ile_args
    assert "--ecliptic-longitude 1.25" in ile_args
    assert "--ecliptic-latitude -0.4" in ile_args
    assert "--srate 0.25" in ile_args
    assert "--channel-name A=SYNTH" in ile_args
    assert "A=fake_strain" not in ile_args
    assert "--psd-file A={}".format(tmp_path / "A_psd.xml.gz") in ile_args
    assert "A=A_psd.xml.gz" not in ile_args

    transfer_files = (rundir / "helper_transfer_files.txt").read_text().splitlines()
    assert os.fspath(tmp_path / "lisa.cache") in transfer_files
    assert os.fspath(tmp_path / "A_psd.xml.gz") in transfer_files


def test_lisa_variable_sky_pseudo_pipe_leaves_sky_intrinsic(tmp_path):
    rundir = tmp_path / "pseudo_lisa_variable_sky"
    env = os.environ.copy()
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = os.path.join(CODE_DIR, "bin") + os.pathsep + env.get("PATH", "")

    cmd = [
        sys.executable,
        PSEUDO_PIPE,
        "--lisa-known-sky",
        "--lisa-vary-sky",
        "--use-rundir",
        os.fspath(rundir),
        "--approx",
        "IMRPhenomD",
        "--event-time",
        "1234.5",
        "--ecliptic-longitude",
        "1.25",
        "--ecliptic-latitude",
        "-0.4",
        "--lisa-cache-file",
        os.fspath(tmp_path / "lisa.cache"),
        "--lisa-psd-file",
        "A={}".format(tmp_path / "A_psd.xml.gz"),
        "--lisa-psd-file",
        "E={}".format(tmp_path / "E_psd.xml.gz"),
        "--lisa-psd-file",
        "T={}".format(tmp_path / "T_psd.xml.gz"),
        "--lisa-srate",
        "0.25",
        "--lisa-grid-size",
        "3",
        "--lisa-sky-grid-width",
        "0.01",
        "--lisa-n-iterations",
        "1",
        "--lisa-n-samples-per-job",
        "1",
        "--internal-ile-request-memory",
        "1024",
        "--internal-cip-request-memory",
        "1024",
    ]
    subprocess.run(cmd, check=True, env=env)

    grid, _ = hyperpipeline_io.read_table(os.fspath(rundir / "proposed-grid.dat"))
    assert grid.shape[0] >= 3
    assert len(set(grid["ecliptic_longitude"])) > 1
    assert len(set(grid["ecliptic_latitude"])) > 1
    # off-lattice sky jitter within +/- sky_grid_width (see pp_surface test for
    # the collinearity rationale); assert it varies and stays in the width box.
    assert all(abs(x - 1.25) <= 0.01 + 1e-9 for x in grid["ecliptic_longitude"])
    assert all(abs(x + 0.40) <= 0.01 + 1e-9 for x in grid["ecliptic_latitude"])

    ile_args = (rundir / "args_ile.txt").read_text()
    assert "--LISA" in ile_args
    # vary-sky: ILE uses the per-row grid sky (--lisa-fixed-sky 1), no hardcode.
    assert "--lisa-fixed-sky" in ile_args
    assert "--ecliptic-longitude" not in ile_args
    assert "--ecliptic-latitude" not in ile_args

    cip_args = (rundir / "args_cip_list.txt").read_text()
    # CIP fits sky as phi/theta; hyperpipeline aliases the ecliptic_longitude/
    # latitude NAMED columns into P.phi/P.theta on read (no positional all.net).
    assert "--parameter phi" in cip_args
    assert "--parameter theta" in cip_args
