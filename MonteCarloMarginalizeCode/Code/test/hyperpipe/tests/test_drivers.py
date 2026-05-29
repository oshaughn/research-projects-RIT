"""
Tests for the toy Gaussian marg driver.

Exercises the full in-tree code path (real env has scipy + lalsuite).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_base_driver_grid_roundtrip(hp_modules, tmp_path):
    """Read -> mutate lnL/sigma -> write should round-trip without truncation."""
    grid = tmp_path / "grid.dat"
    grid.write_text(
        "# lnL sigma_lnL x y z\n"
        "0 0 1.0 2.0 3.0\n"
        "0 0 -1.0 -2.0 -3.0\n"
    )
    rows, cols = hp_modules.drivers_base.read_grid(f"file:{grid}")
    assert cols == ["x", "y", "z"]

    rows[0, 0] = "-3.1415926535"
    rows[0, 1] = "1.234567e-3"
    out = hp_modules.drivers_base.write_marg_output(
        rows[:1], cols,
        fname_output_integral="f.txt",
        outdir=str(tmp_path / "out"),
        fname=None,
        conforming_output_name=True,
    )
    assert Path(out).name == "f.txt+annotation.dat"
    text = Path(out).read_text()
    assert "-3.1415926535" in text
    assert "1.234567e-3" in text
    assert "lnL" in text and "sigma_lnL" in text and "x y z" in text


def test_gaussian_driver_end_to_end(rift_root, rift_py, tmp_path):
    """
    Actually run util_HyperMargGaussian.py against a tiny grid and verify
    the output file shape and that the mode point at (4, 0, 0) outranks
    the far point at (10, 10, 10).
    """
    grid = tmp_path / "grid.dat"
    points = [
        ( 4.0,  0.0,  0.0),
        (-4.0,  0.0,  0.0),
        ( 0.0,  0.0,  0.0),
        (10.0, 10.0, 10.0),
        ( 4.0,  0.5, -0.5),
    ]
    with open(grid, "w") as f:
        f.write("# lnL sigma_lnL x y z\n")
        for x, y, z in points:
            f.write(f"0 0 {x} {y} {z}\n")

    outdir = tmp_path / "out"

    cmd = [
        sys.executable,
        str(rift_py / "bin" / "util_HyperMargGaussian.py"),
        "--using-eos", f"file:{grid}",
        "--eos_start_index", "0",
        "--eos_end_index", "5",
        "--outdir", str(outdir),
        "--fname-output-integral", "lnL.txt",
        "--conforming-output-name",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(rift_py) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, (
        f"util_HyperMargGaussian.py exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    out_file = outdir / "lnL.txt+annotation.dat"
    assert out_file.exists(), f"expected {out_file} to exist"

    lines = [ln for ln in out_file.read_text().splitlines() if ln.strip()]
    assert lines[0].lstrip("#").split() == ["lnL", "sigma_lnL", "x", "y", "z"]
    data_lines = lines[1:]
    assert len(data_lines) == 5
    for ln in data_lines:
        assert len(ln.split()) == 5

    arr = np.array([[float(c) for c in ln.split()] for ln in data_lines])
    lnL = arr[:, 0]
    assert lnL[0] > lnL[3], (
        f"mode point (4,0,0) should have higher lnL than (10,10,10); got {lnL}"
    )
    assert lnL[1] > lnL[3]
