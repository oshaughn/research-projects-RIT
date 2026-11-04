"""
Live integration test: invoke util_RIFT_hyperpipe.py as a subprocess with
the real hydra-core / omegaconf installed by pixi.

Verifies that the emitted command line and per-stage args/exe/event/nchunk
files match what create_eos_posterior_pipeline expects, without actually
running the DAG-builder (general.dry-run=true).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


def _ensure_hydra_available() -> None:
    try:
        import hydra  # noqa: F401
        import omegaconf  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"hydra/omegaconf not importable in this env: {exc}")


def test_util_rift_hyperpipe_dry_run(rift_root, rift_py, tmp_path):
    _ensure_hydra_available()

    base = tmp_path / "base"
    run = tmp_path / "run"
    base.mkdir()
    real_exe = rift_py / "bin" / "util_HyperMargGaussian.py"
    if not real_exe.exists():
        pytest.skip(f"in-tree {real_exe} missing")

    grid_path = base / "blind_gaussian_plus_minus.dat"
    grid_path.write_text(
        "# lnL sigma_lnL x y z\n"
        + "\n".join(f"0 0 {i} {i} {i}" for i in range(10))
        + "\n"
    )

    overrides = [
        f"general.rundir={run}",
        "general.dry-run=true",
        "general.use-osg=true",
        "general.use-singularity=true",
        "general.condor-local-nonworker=true",
        "general.condor-local-nonworker-igwn-prefix=true",
        "general.retries=5",
        "general.request-disk=2G",
        "arch.n-iterations=20",
        "arch.n-samples-per-job=1000",
        "arch.explode-marg-jobs=5",
        f"init.file={grid_path}",
        'marg-list=[{name:gaussian, exe:'
        + shlex.quote(str(real_exe))
        + ', args:"--outdir Gaussian_example --conforming-output-name", '
        'event-file:null, n-chunk:100, coord-module:null, extra-args:""}]',
    ]

    cmd = [sys.executable, str(rift_py / "bin" / "util_RIFT_hyperpipe.py"), *overrides]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(rift_py) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=base)
    print("STDOUT:", proc.stdout)
    print("STDERR:", proc.stderr)
    assert proc.returncode == 0, (
        f"util_RIFT_hyperpipe.py exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    for fname in (
        "args_marg_eos.txt",
        "args_marg_eos_exe.txt",
        "args_eos_post.txt",
        "args_puff.txt",
        "args_test.txt",
        "event-0.net",
        "event_nchunk.txt",
        "initial_grid.dat",
        "transfer_file_list.txt",
    ):
        path = run / fname
        assert path.exists(), f"expected {path} to be written"

    assert (run / "event-0.net").read_text().strip() == "empty_event_file"
    assert (run / "event_nchunk.txt").read_text().strip() == "100"

    marg_args = (run / "args_marg_eos.txt").read_text().strip()
    assert marg_args == "--outdir Gaussian_example --conforming-output-name", marg_args

    post = (run / "args_eos_post.txt").read_text()
    for needle in (
        "--parameter x", "--parameter y", "--parameter z",
        "--integration-parameter-range x:[-8,8]",
        "--integration-parameter-range y:[-8,8]",
        "--integration-parameter-range z:[-8,8]",
    ):
        assert needle in post, (needle, post)

    full = proc.stdout
    for needle in (
        "create_eos_posterior_pipeline",
        "--n-samples-per-job 1000",
        "--n-iterations 20",
        "--marg-event-exe-list-file",
        "--marg-event-args-list-file",
        "--marg-event-nchunk-list-file",
        "--eos-post-args",
        "--eos-post-exe",
        "--puff-exe",
        "--puff-args",
        "--test-args",
        "--test-exe convergence_test_samples",
        "--use-osg",
        "--use-singularity",
        "--condor-local-nonworker",
        "--condor-local-nonworker-igwn-prefix",
        "--general-retries 5",
        "--general-request-disk 2G",
        "--request-memory-marg 16384",
        "--eos-post-explode-jobs 5",
        "--transfer-file-list",
        "--event-file",
    ):
        assert needle in full, f"missing flag in dry-run command line: {needle}"
