"""util_ConstructEOSPosterior.py reports the evidence of the likelihood, not of the shifted likelihood.

The fits evaluate lnL - lnL_shift, with lnL_shift from --lnL-shift-prevent-overflow or set automatically
when every lnL is negative. The driver must add the shift back to --fname-output-integral. Known
answer: lnL = A - r^2/(2 s^2) on the box [-1,1]^2. The driver's default uniform prior has unit density
(it integrates L dx, not L dx / volume), so Z = exp(A) 2 pi sx sy; the Gaussian is well inside the box.
Before the fix, the explicit shift moved the reported evidence by -30 and the automatic shift (A = -20)
by +80.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
DRIVER = os.path.join(CODE, "bin", "util_ConstructEOSPosterior.py")
SX, SY = 0.25, 0.2


def _evidence(tmp, A, extra=(), tag="run"):
    fname = os.path.join(tmp, "grid_%s.dat" % tag)
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, (2000, 2))
    lnL = A - 0.5 * ((x[:, 0] / SX) ** 2 + (x[:, 1] / SY) ** 2) + rng.normal(0, 0.05, len(x))
    np.savetxt(fname, np.column_stack([lnL, 0.05 * np.ones(len(x)), x]), header=" lnL sigma_lnL xx yy")
    env = dict(os.environ, PYTHONPATH=CODE + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
               OMP_NUM_THREADS="1", MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="",
               XDG_CACHE_HOME=os.path.join(tmp, "cache"), MPLCONFIGDIR=os.path.join(tmp, "mpl"))
    out = os.path.join(tmp, "evid_" + tag)
    cmd = [sys.executable, DRIVER, "--fname", fname, "--parameter", "xx", "--parameter", "yy",
           "--integration-parameter-range", "xx:[-1,1]", "--integration-parameter-range", "yy:[-1,1]",
           "--fit-method", "rf", "--sampler-method", "AV", "--internal-use-lnL",
           "--n-max", "400000", "--n-eff", "500", "--n-output-samples", "200",
           "--fname-output-samples", "post_" + tag, "--fname-output-integral", out, "--no-plots"] + list(extra)
    proc = subprocess.run(cmd, cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:]
    return float(np.loadtxt(out))


def _truth(A):
    return A + np.log(2 * np.pi * SX * SY)


@pytest.mark.parametrize("A,extra,tag", [(20.0, (), "plain"),
                                         (20.0, ("--lnL-shift-prevent-overflow", "30"), "explicit"),
                                         (-20.0, (), "automatic")])
def test_reported_evidence_is_unshifted(tmp_path, A, extra, tag):
    z = _evidence(str(tmp_path), A, extra, tag)
    assert abs(z - _truth(A)) < 0.3, (tag, z, _truth(A))
