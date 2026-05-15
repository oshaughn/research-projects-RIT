"""
Unit tests for RIFT.hyperpipe.marg_list.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _make_fake_exe(p: Path) -> None:
    p.write_text("#!/usr/bin/env python\n")
    os.chmod(p, 0o755)


def test_mono_gaussian_assembly(hp_modules, tmp_path):
    base = tmp_path / "base"
    run = tmp_path / "run"
    base.mkdir()
    run.mkdir()
    _make_fake_exe(base / "example_gaussian.py")

    cfg = {
        "marg-list": [
            {
                "name": "gaussian",
                "exe": "example_gaussian.py",
                "args": "--outdir Gaussian_example --conforming-output-name",
                "event-file": None,
                "n-chunk": 100,
                "coord-module": None,
            }
        ]
    }
    marg = hp_modules.marg_list.assemble_marg_list(
        cfg, base_dir=str(base), run_dir=str(run)
    )

    assert marg.names == ["gaussian"]
    assert marg.n_chunks == [100]
    assert marg.args_lines == ["--outdir Gaussian_example --conforming-output-name"]
    assert Path(marg.exe_paths[0]).name == "example_gaussian.py"
    assert (run / "event-0.net").read_text().strip() == "empty_event_file"

    marg.write_args_file(str(run / "args_marg_eos.txt"))
    marg.write_exe_file(str(run / "args_marg_eos_exe.txt"))
    marg.write_nchunk_file(str(run / "event_nchunk.txt"))
    assert (run / "args_marg_eos.txt").read_text().strip() == (
        "--outdir Gaussian_example --conforming-output-name"
    )
    assert (run / "event_nchunk.txt").read_text().strip() == "100"


def test_heterogeneous_nicer_plus_gw(hp_modules, tmp_path):
    """Two drivers, different batch sizes, per-driver coord override."""
    base = tmp_path / "base"
    run = tmp_path / "run"
    base.mkdir()
    run.mkdir()
    _make_fake_exe(base / "reading_NICER_MR.py")
    _make_fake_exe(base / "util_ConstructIntrinsicPosterior_GenericCoordinates.py")
    (base / "my_event_B.net").write_text("# lnL sigma_lnL m1 m2\n0 0 1.4 1.4\n")

    cfg = {
        "marg-list": [
            {
                "name": "nicer",
                "exe": "reading_NICER_MR.py",
                "args": "--j0740 --j0030 --conforming-output-name",
                "event-file": None,
                "n-chunk": 5,
            },
            {
                "name": "gw",
                "exe": "util_ConstructIntrinsicPosterior_GenericCoordinates.py",
                "args": "--eos-param spectral --aligned-prior alignedspin-zprior",
                "event-file": "my_event_B.net",
                "n-chunk": 1,
                "coord-module": "rift_default",
            },
        ]
    }
    marg = hp_modules.marg_list.assemble_marg_list(
        cfg, base_dir=str(base), run_dir=str(run)
    )
    assert marg.n_chunks == [5, 1]
    assert marg.names == ["nicer", "gw"]
    assert "--supplementary-coordinate-code rift_default" in marg.args_lines[1]
    assert "--supplementary-coordinate-code" not in marg.args_lines[0]
    assert (run / "event-0.net").read_text().strip() == "empty_event_file"
    assert "m1 m2" in (run / "event-1.net").read_text()


def test_missing_exe_raises(hp_modules, tmp_path):
    cfg = {"marg-list": [{"name": "x"}]}
    with pytest.raises(ValueError, match="missing required key 'exe'"):
        hp_modules.marg_list.assemble_marg_list(
            cfg, base_dir=str(tmp_path), run_dir=str(tmp_path)
        )
