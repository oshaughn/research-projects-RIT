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


def _run(wrapper, tmp_path, subdir, extra_args=(), shards=True, extra_env=None):
    input_dir = tmp_path / subdir
    input_dir.mkdir()
    if shards:
        (input_dir / "CME_test.dat").write_text(
            "0 1 2 3 4 5 6 7 8 9 10 11 12\n"
        )
    (input_dir / "test.psd.xml.gz").write_bytes(b"psd")
    (input_dir / "command-single.sh").write_text("command\n")
    (input_dir / "integrate.sub").write_text("queue 1\n")
    base_out = tmp_path / ("consolidated_" + subdir)
    env = os.environ.copy()
    env.pop("RIFT_HYPERPIPELINE_FORMAT", None)
    env.update(extra_env or {})
    env["PATH"] = SANITIZED_PATH
    result = subprocess.run(
        [str(wrapper), str(input_dir), str(base_out), *extra_args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    return result, base_out


def _run_ile(wrapper, tmp_path, **kwargs):
    return _run(wrapper, tmp_path, "ile", **kwargs)


def _run_nr(wrapper, tmp_path):
    return _run(wrapper, tmp_path, "nr", extra_args=["Sequence-RIT-All"])


@pytest.mark.parametrize("name", ["util_ILEdagPostprocess.sh", "util_NRdagPostprocess.sh"])
def test_postprocess_fails_when_cleaner_is_unavailable(tmp_path, name):
    assert shutil.which("util_CleanILE.py", path=SANITIZED_PATH) is None
    wrapper = _copy_wrapper(tmp_path, name)
    run = _run_nr if name == "util_NRdagPostprocess.sh" else _run_ile
    result, base_out = run(wrapper, tmp_path)
    assert result.returncode == 127
    assert "unable to locate required helper util_CleanILE.py" in result.stderr
    assert not base_out.with_suffix(".composite").exists()


def test_ile_postprocess_without_shards_keeps_jumpstart_behaviour(tmp_path):
    # --first-iteration-jumpstart: the first consolidate node has no ILE
    # parents and no POST check, so an absent shard set must still exit 0.
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    result, base_out = _run_ile(wrapper, tmp_path, shards=False)
    assert result.returncode == 0, result.stderr
    assert base_out.with_suffix(".composite").stat().st_size == 0
    assert base_out.with_suffix(".tgz").exists()


def test_ile_postprocess_resolves_sibling_cleaner_with_sanitized_path(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", 'cat "$1"\n')
    result, base_out = _run_ile(wrapper, tmp_path)
    assert result.returncode == 0, result.stderr
    assert base_out.with_suffix(".composite").stat().st_size > 0


def test_ile_postprocess_rejects_cleaner_output_without_rows(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", "exit 0\n")
    result, base_out = _run_ile(wrapper, tmp_path)
    assert result.returncode == 1
    assert "produced no usable rows" in result.stderr
    assert not base_out.with_suffix(".composite").exists()


def test_ile_postprocess_propagates_cleaner_failure(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", "exit 42\n")
    result, base_out = _run_ile(wrapper, tmp_path)
    assert result.returncode == 42
    assert not base_out.with_suffix(".composite").exists()


HYPER = {"RIFT_HYPERPIPELINE_FORMAT": "1"}


def test_ile_hyperpipeline_resolves_sibling_cleaner(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    # --output FILE [flags] shards: write the shards to FILE
    _write_helper(tmp_path, "util_CleanILE_hyperpipeline.py",
                  'out=$2; shift 2; cat "$@" > "$out"\n')
    result, base_out = _run_ile(wrapper, tmp_path, extra_env=HYPER)
    assert result.returncode == 0, result.stderr
    assert base_out.with_suffix(".composite").stat().st_size > 0


def test_ile_hyperpipeline_propagates_cleaner_failure(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE_hyperpipeline.py",
                  'echo partial > "$2"; exit 3\n')
    result, base_out = _run_ile(wrapper, tmp_path, extra_env=HYPER)
    assert result.returncode == 3
    assert not base_out.with_suffix(".composite").exists()


def test_ile_hyperpipeline_rejects_empty_composite(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_ILEdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE_hyperpipeline.py", ': > "$2"\n')
    result, base_out = _run_ile(wrapper, tmp_path, extra_env=HYPER)
    assert result.returncode == 1
    assert "produced an empty composite" in result.stderr
    assert not base_out.with_suffix(".composite").exists()


def test_nr_postprocess_propagates_cleaner_failure(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_NRdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", "exit 7\n")
    _write_helper(tmp_path, "util_NRRelabelILE.py", "exit 0\n")
    result, base_out = _run_nr(wrapper, tmp_path)
    assert result.returncode == 7
    assert not base_out.with_suffix(".composite").exists()


def test_nr_postprocess_propagates_relabel_failure(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_NRdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", 'cat "$1"\n')
    _write_helper(tmp_path, "util_NRRelabelILE.py", "echo -1 row; exit 5\n")
    result, base_out = _run_nr(wrapper, tmp_path)
    assert result.returncode == 5
    assert "NR relabeling failed" in result.stderr
    assert not base_out.with_suffix(".indexed").exists()


def test_nr_postprocess_rejects_empty_relabel_output(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_NRdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", 'cat "$1"\n')
    _write_helper(tmp_path, "util_NRRelabelILE.py", "exit 0\n")
    result, base_out = _run_nr(wrapper, tmp_path)
    assert result.returncode == 1
    assert "produced an empty index" in result.stderr
    assert not base_out.with_suffix(".indexed").exists()


def test_nr_postprocess_succeeds_with_sibling_helpers(tmp_path):
    wrapper = _copy_wrapper(tmp_path, "util_NRdagPostprocess.sh")
    _write_helper(tmp_path, "util_CleanILE.py", 'cat "$1"\n')
    _write_helper(tmp_path, "util_NRRelabelILE.py", "echo -1 row\n")
    result, base_out = _run_nr(wrapper, tmp_path)
    assert result.returncode == 0, result.stderr
    assert base_out.with_suffix(".indexed").read_text() == "-1 row\n"
    assert base_out.with_suffix(".tgz").exists()
