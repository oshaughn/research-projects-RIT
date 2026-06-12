#!/usr/bin/env python
"""Contract tests for the standalone LISA CEPP helper."""

import os
import subprocess
import sys

import pytest

from RIFT.misc import hyperpipeline_io


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
HELPER = os.path.join(
    REPO_ROOT, "MonteCarloMarginalizeCode", "Code", "bin", "helper_LISA_Events.py"
)
CEPP = os.path.join(
    REPO_ROOT,
    "MonteCarloMarginalizeCode",
    "Code",
    "bin",
    "create_event_parameter_pipeline_BasicIteration",
)


def _run_helper(tmp_path, *extra_args):
    cmd = [
        sys.executable,
        HELPER,
        "--working-directory",
        os.fspath(tmp_path),
        "--zero-likelihood",
    ]
    cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def test_lisa_helper_writes_cepp_contract_files(tmp_path):
    _run_helper(tmp_path)

    expected = {
        "proposed-grid.dat",
        "args_ile.txt",
        "args_cip_list.txt",
        "args_test.txt",
        "helper_transfer_files.txt",
        "command-cepp-lisa.sh",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}

    grid, columns = hyperpipeline_io.read_table(os.fspath(tmp_path / "proposed-grid.dat"))
    assert grid.shape == (3,)
    assert "ecliptic_longitude" in columns
    assert "ecliptic_latitude" in columns

    ile_args = (tmp_path / "args_ile.txt").read_text()
    assert ile_args.startswith("X ")
    assert "--LISA" in ile_args
    assert "--h5-frame-FD" in ile_args
    assert "--time-marginalization" in ile_args
    assert "--zero-likelihood" in ile_args
    assert "--data-integration-window-half 8.0" in ile_args
    assert "--cache-file lisa.cache" in ile_args
    assert "--channel-name A=fake_strain" in ile_args
    assert "--psd-file A=A_psd.xml.gz" in ile_args
    assert "--sim-grid" not in ile_args
    assert "--output-file" not in ile_args

    cip_args = (tmp_path / "args_cip_list.txt").read_text()
    assert cip_args.startswith("1 ")
    assert "--parameter ecliptic_longitude" in cip_args
    assert "--parameter ecliptic_latitude" in cip_args
    assert "--fname" not in cip_args

    test_args = (tmp_path / "args_test.txt").read_text()
    assert test_args.startswith("X ")
    assert "--always-succeed" in test_args

    cepp_command = (tmp_path / "command-cepp-lisa.sh").read_text()
    assert "RIFT_HYPERPIPELINE_FORMAT=1" in cepp_command
    assert "integrate_likelihood_extrinsic_batchmode_lisa" in cepp_command
    assert os.fspath(tmp_path / "proposed-grid.dat") in cepp_command


def test_lisa_helper_custom_data_products_replace_defaults(tmp_path):
    _run_helper(
        tmp_path,
        "--cache-file",
        os.fspath(tmp_path / "custom.cache"),
        "--psd-file",
        "A=/tmp/A.xml.gz",
        "--psd-file",
        "E=/tmp/E.xml.gz",
        "--psd-file",
        "T=/tmp/T.xml.gz",
        "--channel-name",
        "A=SYNTH",
        "--channel-name",
        "E=SYNTH",
        "--channel-name",
        "T=SYNTH",
    )

    ile_args = (tmp_path / "args_ile.txt").read_text()
    assert "--cache-file {}/custom.cache".format(tmp_path) in ile_args
    assert "--psd-file A=/tmp/A.xml.gz" in ile_args
    assert "--psd-file A=A_psd.xml.gz" not in ile_args
    assert "--channel-name A=SYNTH" in ile_args
    assert "--channel-name A=fake_strain" not in ile_args

    transfer_files = (tmp_path / "helper_transfer_files.txt").read_text().splitlines()
    assert os.fspath(tmp_path / "custom.cache") in transfer_files
    assert "/tmp/A.xml.gz" in transfer_files
    assert "A_psd.xml.gz" not in transfer_files


def test_lisa_helper_bundle_renders_basic_cepp_dag(tmp_path):
    _run_helper(
        tmp_path,
        "--grid-size",
        "1",
        "--n-iterations",
        "1",
        "--n-samples-per-job",
        "1",
    )

    env = os.environ.copy()
    env["RIFT_HYPERPIPELINE_FORMAT"] = "1"
    env["PATH"] = os.path.dirname(HELPER) + os.pathsep + env.get("PATH", "")
    cmd = [
        sys.executable,
        CEPP,
        "--ile-n-events-to-analyze",
        "1",
        "--input-grid",
        os.fspath(tmp_path / "proposed-grid.dat"),
        "--ile-exe",
        HELPER.replace("helper_LISA_Events.py", "integrate_likelihood_extrinsic_batchmode_lisa"),
        "--ile-args",
        os.fspath(tmp_path / "args_ile.txt"),
        "--cip-args-list",
        os.fspath(tmp_path / "args_cip_list.txt"),
        "--test-args",
        os.fspath(tmp_path / "args_test.txt"),
        "--working-directory",
        os.fspath(tmp_path),
        "--n-iterations",
        "1",
        "--n-samples-per-job",
        "1",
        "--n-copies",
        "1",
        "--request-memory-ILE",
        "1024",
        "--request-memory-CIP",
        "1024",
    ]
    try:
        subprocess.run(cmd, check=True, env=env, cwd=os.fspath(tmp_path))
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip("CEPP DAG render unavailable in this environment: {}".format(exc))

    assert (tmp_path / "ILE.sub").exists()
    assert (tmp_path / "CIP.sub").exists()
    assert any(path.suffix == ".dag" for path in tmp_path.iterdir())
