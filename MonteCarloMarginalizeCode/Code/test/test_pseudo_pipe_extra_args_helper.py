"""--extra-args-helper reaches the standard helper_LDG_Events.py call (build only, never submit)."""
import re

import pytest

from test_rf_transverse_pipeline_build import _build, _opt_in_ini


def _helper_cip_lines(tmp_path, value):
    ini = _opt_in_ini(tmp_path, extra=['extra-args-helper="{}"'.format(value)])
    ini.write_text(re.sub(r'(?m)^rf-transverse-spin-coordinates=.*\n', '', ini.read_text()))
    proc, rundir = _build(tmp_path, ini)
    assert proc.returncode == 0, proc.stdout[-3000:]
    return (rundir / "helper_cip_arg_list.txt").read_text().splitlines()


def test_extra_args_helper_file_reaches_standard_helper(tmp_path):
    pytest.importorskip("lal")
    # A helper-only flag: without it the helper puts --eta-range on every CIP line.
    # Relative path, resolved before pseudo_pipe changes into the run directory.
    (tmp_path / "extra_helper.txt").write_text("--no-propose-limits\n")
    lines = _helper_cip_lines(tmp_path, "extra_helper.txt")
    assert lines and not any("--eta-range" in line for line in lines), lines


def test_extra_args_helper_inline_reaches_standard_helper(tmp_path):
    # Existing callers pass the arguments themselves rather than a file name.
    pytest.importorskip("lal")
    lines = _helper_cip_lines(tmp_path, " --no-propose-limits ")
    assert lines and not any("--eta-range" in line for line in lines), lines
