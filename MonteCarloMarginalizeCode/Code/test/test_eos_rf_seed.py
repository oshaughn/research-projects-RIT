"""--rf-seed in bin/util_ConstructEOSPosterior.py: the same seed regrows the same ExtraTrees forest.

The driver prints an RF-FOREST line (tree and node counts, threshold and leaf-value sums) when a seed
is given. Two runs with one seed must print the same line; a different seed must change the
thresholds. Predictions of a regrown forest agree to float roundoff, not bit for bit: sklearn adds
the trees' predictions in thread-completion order.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
DRIVER = os.path.join(CODE, "bin", "util_ConstructEOSPosterior.py")


def _run(tmp, seed):
    fname = os.path.join(tmp, "grid.dat")
    if not os.path.exists(fname):
        rng = np.random.default_rng(5)
        x = rng.uniform(-1, 1, (800, 2))
        lnL = 20 - 0.5 * np.sum(((x - 0.1) / 0.3) ** 2, axis=1) + rng.normal(0, 0.2, len(x))
        np.savetxt(fname, np.column_stack([lnL, 0.2 * np.ones(len(x)), x]), header=" lnL sigma_lnL xx yy")
    env = dict(os.environ, PYTHONPATH=CODE + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
               OMP_NUM_THREADS="1", MPLBACKEND="Agg", XDG_CACHE_HOME=os.path.join(tmp, "cache"),
               MPLCONFIGDIR=os.path.join(tmp, "mpl"), CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, DRIVER, "--fname", fname, "--parameter", "xx", "--parameter", "yy",
           "--integration-parameter-range", "xx:[-1,1]", "--integration-parameter-range", "yy:[-1,1]",
           "--fit-method", "rf", "--rf-seed", str(seed), "--sampler-method", "AV", "--internal-use-lnL",
           "--n-max", "200000", "--n-eff", "200", "--n-output-samples", "200",
           "--fname-output-samples", "post_%d" % seed, "--no-plots"]
    proc = subprocess.run(cmd, cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:]
    lines = re.findall(r"RF-FOREST .*", proc.stdout)
    assert len(lines) == 1, proc.stdout[-3000:]
    return lines[0]


def test_same_seed_same_forest_other_seed_differs(tmp_path):
    tmp = str(tmp_path)
    a, b, c = _run(tmp, 11), _run(tmp, 11), _run(tmp, 12)
    assert a == b
    thr = lambda line: re.search(r"thr_sum=(\S+)", line).group(1)
    assert thr(a) != thr(c)


def test_sklearn_seeded_forest_predictions_agree_to_roundoff():
    from sklearn.ensemble import ExtraTreesRegressor
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (2000, 4)); y = np.sin(3 * X[:, 0]) + rng.normal(0, 0.1, 2000)
    q = rng.uniform(-1, 1, (1000, 4))
    p = [ExtraTreesRegressor(50, n_jobs=4, random_state=11).fit(X, y).predict(q) for _ in range(2)]
    assert np.max(np.abs(p[0] - p[1])) < 1e-12
