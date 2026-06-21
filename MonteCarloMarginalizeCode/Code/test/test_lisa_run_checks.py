"""Tests for LISA run diagnostics."""

import json
import os
import subprocess
import sys

import numpy as np

from RIFT.LISA.run_checks import plot_RIFT


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CHECK_SCRIPT = os.path.join(
    REPO_ROOT,
    "MonteCarloMarginalizeCode",
    "Code",
    "RIFT",
    "LISA",
    "run_checks",
    "plot_RIFT.py",
)


def _write_lisa_table(path):
    rows = np.array(
        [
            [0, 1.0e5, 8.0e4, 0, 0, 0.1, 0, 0, -0.1, 1.0, 0.3, 10.0, 0.2, 40, 12],
            [0, 1.1e5, 7.5e4, 0, 0, 0.0, 0, 0, 0.0, 1.1, 0.2, 14.0, 0.5, 50, 20],
            [0, 0.9e5, 8.5e4, 0, 0, 0.2, 0, 0, 0.1, 0.9, 0.4, 13.0, 0.1, 60, 30],
        ]
    )
    np.savetxt(path, rows)


def test_lisa_run_summary_identifies_best_point(tmp_path):
    table = tmp_path / "all.net"
    _write_lisa_table(table)

    summary = plot_RIFT.summarize_lisa_ile(os.fspath(table), lnL_window=2.0, error_threshold=0.4)
    assert summary["n_rows"] == 3
    assert summary["max_index"] == 1
    assert summary["max_lnL"] == 14.0
    assert summary["high_lnL_points"] == 2
    assert summary["high_lnL_low_error_points"] == 1
    assert summary["best"]["ecliptic_longitude"] == 1.1
    assert summary["best"]["ecliptic_latitude"] == 0.2


def test_lisa_run_summary_cli_json(tmp_path):
    table = tmp_path / "lisa_ile_0_.dat"
    _write_lisa_table(table)

    output = subprocess.check_output(
        [sys.executable, CHECK_SCRIPT, os.fspath(table), "--json"],
        text=True,
    )
    summary = json.loads(output)
    assert summary["max_lnL"] == 14.0
    assert summary["best"]["n_eff"] == 20.0
