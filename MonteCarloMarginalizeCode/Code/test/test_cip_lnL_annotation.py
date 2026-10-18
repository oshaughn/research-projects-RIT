"""CIP evidence outputs with --internal-use-lnL (AV, portfolio, NFlow, GMM in lnL mode).

In that mode sampler.integrate returns (ln Z, ln Var[Z]).  CIP must write ln Z itself,
not ln(ln Z), to integral.dat and +annotation.dat, and a relative error built from
ln Var[Z], not sqrt(ln Var)/ln Z.  The CIP run uses a synthetic quadratic lnL whose
maximum is lnLmax=40: ln Z must lie a few units below it, while ln(ln Z) is ~3.6.
"""
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from RIFT.misc.mc_error import relative_mc_error


SCRIPT = Path(__file__).resolve().parents[1] / "bin" / "util_ConstructIntrinsicPosterior_GenericCoordinates.py"
LNL_MAX = 40.0


def test_log_space_error_matches_av_variance_convention():
    # mcsamplerAdaptiveVolume.integrate_log returns ln Var[Z] = ln(rel_var) + 2 ln Z.
    log_z, rel_sigma = 348.2, 0.015
    log_var = math.log(rel_sigma**2) + 2 * log_z
    assert relative_mc_error(log_z, log_var, log_space=True) == pytest.approx(rel_sigma)


def _write_synthetic_grid(path, n=400, seed=1):
    rng = np.random.default_rng(seed)
    mc = rng.uniform(10.0, 12.0, n)
    eta = rng.uniform(0.15, 0.2499, n)
    lnL = LNL_MAX - 0.5 * ((mc - 11.0) / 0.2) ** 2 - 0.5 * ((eta - 0.22) / 0.02) ** 2
    mtot = mc / eta ** 0.6
    delta = np.sqrt(1 - 4 * eta)
    m1, m2 = mtot * (1 + delta) / 2, mtot * (1 - delta) / 2
    zeros = np.zeros(n)
    rows = np.column_stack([-np.ones(n), m1, m2, zeros, zeros, zeros, zeros, zeros, zeros,
                            lnL, 0.01 * np.ones(n), 1000 * np.ones(n), -np.ones(n)])
    np.savetxt(path, rows)


def test_cip_writes_lnZ_not_log_lnZ(tmp_path):
    grid = tmp_path / "all.net"
    _write_synthetic_grid(grid)
    out = tmp_path / "out"
    env = dict(os.environ, GW_SURROGATE="", MPLBACKEND="Agg",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [sys.executable, str(SCRIPT), "--fname", str(grid), "--fit-method", "quadratic",
           "--parameter", "mc", "--parameter", "eta", "--mc-range", "[10,12]",
           "--eta-range", "[0.15,0.2499]", "--sampler-method", "AV", "--internal-use-lnL",
           "--n-output-samples", "100", "--n-eff", "100", "--n-max", "200000", "--no-plots",
           "--fname-output-samples", str(out), "--fname-output-integral", str(out)]
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    ln_z, sigma, _neff = np.loadtxt(str(out) + "+annotation.dat")[:3]
    assert LNL_MAX - 20 < ln_z < LNL_MAX, ln_z
    assert 0 < sigma < 0.5, sigma
    assert float(np.loadtxt(str(out) + ".dat")) == pytest.approx(ln_z)
    ln_z_ess, sigma_ess = np.loadtxt(str(out) + "+annotation_ESS.dat")[:2]
    assert (ln_z_ess, sigma_ess) == pytest.approx((ln_z, sigma))
