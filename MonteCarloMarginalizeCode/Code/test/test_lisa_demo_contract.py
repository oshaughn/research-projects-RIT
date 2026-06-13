"""Contract test for the checked-in LISA zero-likelihood demo."""

import os
import subprocess

from RIFT.misc import hyperpipeline_io


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEMO = os.path.join(
    REPO_ROOT,
    "MonteCarloMarginalizeCode",
    "Code",
    "demo",
    "rift",
    "lisa",
    "run_lisa_zero_likelihood_cepp.sh",
)


def test_lisa_zero_likelihood_demo_renders_cepp_bundle(tmp_path):
    env = os.environ.copy()
    env["RIFT_LISA_WORKDIR"] = os.fspath(tmp_path)
    subprocess.run([DEMO], check=True, env=env)

    expected = {
        "proposed-grid.dat",
        "args_ile.txt",
        "args_cip_list.txt",
        "args_test.txt",
        "helper_transfer_files.txt",
        "command-cepp-lisa.sh",
        "ILE.sub",
        "CIP.sub",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert any(path.suffix == ".dag" for path in tmp_path.iterdir())

    grid, columns = hyperpipeline_io.read_table(os.fspath(tmp_path / "proposed-grid.dat"))
    assert grid.shape == (1,)
    assert "ecliptic_longitude" in columns
    assert "ecliptic_latitude" in columns

    ile_args = (tmp_path / "args_ile.txt").read_text()
    assert "--zero-likelihood" in ile_args
    assert "--LISA" in ile_args
    assert "--cache-file lisa.cache" in ile_args

    cip_args = (tmp_path / "args_cip_list.txt").read_text()
    assert "--parameter mc" in cip_args
    assert "--parameter eta" in cip_args
    assert "--parameter ecliptic_longitude" not in cip_args
    assert "--parameter ecliptic_latitude" not in cip_args
