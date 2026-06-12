"""Tests for the lightweight LISA PP-style surface."""

import os
import subprocess

import RIFT.lalsimutils as lalsimutils

from RIFT.misc import hyperpipeline_io


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CODE_DIR = os.path.join(REPO_ROOT, "MonteCarloMarginalizeCode", "Code")
PP_LISA_DRIVER = os.path.join(CODE_DIR, "test", "pp_lisa", "run_pp_lisa_known_sky.sh")
MAKE_PSDS = os.path.join(CODE_DIR, "demo", "rift", "lisa", "make_lisa_psds.py")


def test_lisa_demo_psd_generator_writes_channel_xml(tmp_path):
    subprocess.run(
        [
            MAKE_PSDS,
            "--output-directory",
            os.fspath(tmp_path),
            "--fmax",
            "0.125",
            "--npts",
            "129",
            "--write-ascii",
        ],
        check=True,
    )

    for channel in ["A", "E", "T"]:
        psd_path = tmp_path / "{}_psd.xml.gz".format(channel)
        assert psd_path.exists()
        psd = lalsimutils.get_psd_series_from_xmldoc(os.fspath(psd_path), channel)
        assert psd.data.length == 129
        assert psd.deltaF > 0
        assert psd.data.data[1] > 0

    assert (tmp_path / "LISA_psd.txt").exists()


def test_lisa_pp_known_sky_surface_builds_bundle_and_dag(tmp_path):
    env = os.environ.copy()
    env["RIFT_PP_LISA_WORKDIR"] = os.fspath(tmp_path)
    env["RIFT_PP_LISA_RUN_ILE"] = "0"
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = os.path.join(CODE_DIR, "bin") + os.pathsep + env.get("PATH", "")

    subprocess.run([PP_LISA_DRIVER], check=True, env=env)

    bundle_dir = tmp_path / "event_0"
    rundir = tmp_path / "analysis_event_0"

    expected_bundle = {
        "A-fake_strain-1000000-10000.h5",
        "E-fake_strain-1000000-10000.h5",
        "T-fake_strain-1000000-10000.h5",
        "A_psd.xml.gz",
        "E_psd.xml.gz",
        "T_psd.xml.gz",
        "LISA_psd.txt",
        "lisa.cache",
        "synthetic-params.env",
    }
    assert expected_bundle <= {path.name for path in bundle_dir.iterdir()}

    assert (rundir / "proposed-grid.dat").exists()
    assert (rundir / "args_ile.txt").exists()
    assert (rundir / "helper_transfer_files.txt").exists()
    assert (rundir / "ILE.sub").exists()
    assert (rundir / "CIP.sub").exists()

    _, columns = hyperpipeline_io.read_table(os.fspath(rundir / "proposed-grid.dat"))
    assert "ecliptic_longitude" in columns
    assert "ecliptic_latitude" in columns

    ile_args = (rundir / "args_ile.txt").read_text()
    assert "--lisa-fixed-sky 1" in ile_args
    assert "--cache-file {}".format(bundle_dir / "lisa.cache") in ile_args
    assert "--psd-file A={}".format(bundle_dir / "A_psd.xml.gz") in ile_args
    assert "--srate 0.25" in ile_args

    transfer_files = (rundir / "helper_transfer_files.txt").read_text().splitlines()
    assert os.fspath(bundle_dir / "lisa.cache") in transfer_files
    assert os.fspath(bundle_dir / "A_psd.xml.gz") in transfer_files


def test_lisa_pp_variable_sky_surface_builds_intrinsic_sky_grid(tmp_path):
    env = os.environ.copy()
    env["RIFT_PP_LISA_WORKDIR"] = os.fspath(tmp_path)
    env["RIFT_PP_LISA_RUN_ILE"] = "0"
    env["RIFT_PP_LISA_VARY_SKY"] = "1"
    env["PYTHONPATH"] = CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = os.path.join(CODE_DIR, "bin") + os.pathsep + env.get("PATH", "")

    subprocess.run([PP_LISA_DRIVER], check=True, env=env)

    rundir = tmp_path / "analysis_event_0"
    grid, _ = hyperpipeline_io.read_table(os.fspath(rundir / "proposed-grid.dat"))
    assert grid.shape == (3,)
    assert len(set(grid["ecliptic_longitude"])) == 3
    assert len(set(grid["ecliptic_latitude"])) == 3

    ile_args = (rundir / "args_ile.txt").read_text()
    assert "--lisa-fixed-sky" not in ile_args
    assert "--ecliptic-longitude" not in ile_args
    assert "--ecliptic-latitude" not in ile_args
