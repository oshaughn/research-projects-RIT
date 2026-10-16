import os
from pathlib import Path
import shutil
import subprocess

import pytest


BIN_DIR = Path(__file__).resolve().parents[1] / "bin"
SANITIZED_PATH = "/usr/bin:/bin"


def _copy_wrapper(tmp_path, name):
    wrapper = tmp_path / name
    shutil.copy2(BIN_DIR / name, wrapper)
    return wrapper


def _write_helper(directory, name, body):
    helper = directory / name
    helper.write_text("#!/bin/sh\n" + body)
    helper.chmod(0o755)
    return helper


def _run_ile(wrapper, tmp_path):
    input_dir = tmp_path / "ile"
    input_dir.mkdir()
    (input_dir / "CME_test.dat").write_text(
        "0 1 2 3 4 5 6 7 8 9 10 11 12\n"
    )
    (input_dir / "test.psd.xml.gz").write_bytes(b"psd")
    (input_dir / "command-single.sh").write_text("command\n")
    base_out = tmp_path / "consolidated"
    env = os.environ.copy()
    env["PATH"] = SANITIZED_PATH
    result = subprocess.run(
        [str(wrapper), str(input_dir), str(base_out)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    return result, base_out


def _run_nr(wrapper, tmp_path):
    input_dir = tmp_path / "nr"
    input_dir.mkdir()
    (input_dir / "CME_test.dat").write_text(
        "0 1 2 3 4 5 6 7 8 9 10 11 12\n"
    )
    (input_dir / "test.psd.xml.gz").write_bytes(b"psd")
    (input_dir / "command-single.sh").write_text("command\n")
    (input_dir / "integrate.sub").write_text("queue 1\n")
    base_out = tmp_path / "consolidated_nr"
    env = os.environ.copy()
    env["PATH"] = SANITIZED_PATH
    result = subprocess.run(
        [str(wrapper), str(input_dir), str(base_out), "Sequence-RIT-All"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    return result, base_out


@pytest.mark.parametrize("name", ["util_ILEdagPostprocess.sh", "util_NRdagPostprocess.sh"])
def test_postprocess_fails_when_cleaner_is_unavailable(tmp_path, name):
    assert shutil.which("util_CleanILE.py", path=SANITIZED_PATH) is None
    wrapper = _copy_wrapper(tmp_path, name)
    env = os.environ.copy()
    env["PATH"] = SANITIZED_PATH
    args = [str(wrapper), "missing-input", str(tmp_path / "out")]
    if name == "util_NRdagPostprocess.sh":
        args.append("Sequence-RIT-All")
    result = subprocess.run(args, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "unable to locate required helper util_CleanILE.py" in result.stderr


def test_ile_postprocess_resolves_sibling_cleaner_with_sanitized_path(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", 'cat "$1"\n')
    result, base_out = _run_ile(wrapper, tmp_path)
    assert result.returncode == 0, result.stderr
    assert base_out.with_suffix(".composite").stat().st_size > 0


def test_ile_postprocess_rejects_empty_composite(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", "exit 0\n")
    result, base_out = _run_ile(wrapper, tmp_path)
    assert result.returncode != 0
    assert "produced an empty composite" in result.stderr
    assert not base_out.with_suffix(".composite").exists()


def test_ile_postprocess_propagates_cleaner_failure(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", "exit 42\n")
    result, base_out = _run_ile(wrapper, tmp_path)
    assert result.returncode == 42
    assert not base_out.with_suffix(".composite").exists()


def test_nr_postprocess_rejects_empty_relabel_output(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_NRdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", 'cat "$1"\n')
    _write_helper(tmp_path, "util_NRRelabelILE.py", "exit 0\n")
    result, base_out = _run_nr(wrapper, tmp_path)
    assert result.returncode != 0
    assert "NR relabeling failed" in result.stderr
    assert not base_out.with_suffix(".indexed").exists()
