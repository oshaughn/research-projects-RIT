"""Regression for the multi-worker CIP evidence summary (upstream PR 156)."""

from pathlib import Path
import subprocess
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "util_CIPDirSummarizeEvidence.py"


def test_multi_worker_summary_uses_every_worker(tmp_path):
    workers = [(10.0, 1.0), (20.0, 2.0), (30.0, 4.0)]
    for index, (lnl, sigma) in enumerate(workers):
        stem = tmp_path / f"overlap-grid-0-{index}"
        (stem.with_name(stem.name + "+annotation.dat")).write_text(
            f"sigmaL neff\n{sigma} 100\n"
        )
        (stem.with_name(stem.name + "_withpriorchange+annotation.dat")).write_text(
            f"lnL neff\n{lnl} 100\n"
        )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--cip-dir", str(tmp_path), "--stream-output"],
        capture_output=True, text=True, check=True,
    )
    observed_lnl, observed_sigma = map(float, result.stdout.split())
    values = np.array([item[0] for item in workers])
    sigmas = np.array([item[1] for item in workers])
    expected_lnl = np.average(values, weights=1 / sigmas**2)
    expected_sigma = max(np.sqrt(np.mean(sigmas**2) / len(workers)), np.std(values))
    assert np.isclose(observed_lnl, expected_lnl)
    assert np.isclose(observed_sigma, expected_sigma)
    assert not np.isclose(observed_lnl, workers[-1][0])
