"""Known-answer test for `--fit-method gp-matern` in bin/util_ConstructEOSPosterior.py.

A noisy 2-D Gaussian lnL on a random grid, uniform prior on the box: the posterior is the same
Gaussian (truncation by the box is negligible at these widths), so the recovered mean and width
are known. The noise (0.3 nats, reported in the sigma column) is what the GP's per-row alpha
should average away.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
DRIVER = os.path.join(CODE, "bin", "util_ConstructEOSPosterior.py")

MU = np.array([0.2, -0.1])
SIG = np.array([0.25, 0.15])
NOISE = 0.3


def _write_grid(path, n=1500, center=MU):
    rng = np.random.default_rng(20261003)
    x = rng.uniform(-1.0, 1.0, (n, 2))
    lnL = 20.0 - 0.5 * np.sum(((x - center) / SIG) ** 2, axis=1) + rng.normal(0, NOISE, n)
    np.savetxt(path, np.column_stack([lnL, NOISE * np.ones(n), x]), header=" lnL sigma_lnL xx yy")


def _run(tmp_path, extra=()):
    fname = os.path.join(str(tmp_path), "grid.dat")
    _write_grid(fname)
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(OMP_NUM_THREADS="1", MPLBACKEND="Agg",
               XDG_CACHE_HOME=os.path.join(str(tmp_path), "cache"),
               MPLCONFIGDIR=os.path.join(str(tmp_path), "mpl"))
    cmd = [sys.executable, DRIVER, "--fname", fname,
           "--parameter", "xx", "--parameter", "yy",
           "--integration-parameter-range", "xx:[-1,1]", "--integration-parameter-range", "yy:[-1,1]",
           "--fit-method", "gp-matern", "--gp-matern-max-train-points", "600",
           "--sampler-method", "AV", "--internal-use-lnL",
           "--n-max", "400000", "--n-eff", "1000", "--n-output-samples", "2000",
           "--fname-output-samples", "post", "--no-plots"] + list(extra)
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
    return proc, os.path.join(str(tmp_path), "post.dat")


@pytest.fixture(scope="module")
def matern_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("eos_matern"))


def test_gp_matern_completes_and_records_the_fit(matern_run):
    proc, _ = matern_run
    assert proc.returncode == 0, proc.stdout[-3000:]
    assert "GP-MATERN-RECORD" in proc.stdout
    assert '"training_rows": 600' in proc.stdout


def test_gp_matern_recovers_the_known_posterior(matern_run):
    proc, out = matern_run
    post = np.genfromtxt(out, names=True)
    x = np.column_stack([post["xx"], post["yy"]])
    # Tolerances: 2000 weighted draws give ~0.01 sampling error on the mean; the widths carry the
    # fit error, which is the thing under test, so 10% is the bar.
    assert np.all(np.abs(x.mean(0) - MU) < 0.03), x.mean(0)
    assert np.all(np.abs(x.std(0) / SIG - 1) < 0.10), x.std(0)


def test_cupy_backend_requires_gp_matern(tmp_path):
    proc, _ = _run(tmp_path, ["--fit-method", "rf", "--gp-predict-backend", "cupy"])
    assert proc.returncode != 0
    assert "--gp-predict-backend cupy requires --fit-method gp-matern" in proc.stdout


_NAN_PLUGIN = '''
import numpy as np
def convert_coordinates(x_in, coord_names, low_level_coord_names, **kwargs):
    x_in = np.atleast_2d(np.asarray(x_in, dtype=float))
    src = {n: x_in[:, i] for i, n in enumerate(low_level_coord_names)}
    if "xx" in src:
        src["ux"] = np.where(src["xx"] > 0.9, np.nan, src["xx"])   # a converter that fails on part of the box
        src["uy"] = src["yy"]
    return np.column_stack([src[n] for n in coord_names])
'''


def test_nonfinite_conversion_gets_the_floor(tmp_path):
    """Rows the converter cannot map must carry no posterior weight (the -500 fill is a floor).

    The likelihood peaks at the box edge xx=1 here, so a fill that acted as a ceiling (or any
    value near the peak) would put most samples in the NaN strip xx>0.9."""
    plugin = os.path.join(str(tmp_path), "nan_plugin.py")
    open(plugin, "w").write(_NAN_PLUGIN)
    fname = os.path.join(str(tmp_path), "grid.dat")
    rng = np.random.default_rng(4)
    x = rng.uniform(-1.0, 1.0, (800, 2))
    lnL = 20.0 - 0.5 * (((x[:, 0] - 0.85) / 0.3) ** 2 + (x[:, 1] / 0.3) ** 2)
    np.savetxt(fname, np.column_stack([lnL, 0.05 * np.ones(len(x)), x]), header=" lnL sigma_lnL xx yy")
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(OMP_NUM_THREADS="1", MPLBACKEND="Agg", XDG_CACHE_HOME=os.path.join(str(tmp_path), "cache"),
               MPLCONFIGDIR=os.path.join(str(tmp_path), "mpl"))
    cmd = [sys.executable, DRIVER, "--fname", fname,
           "--parameter-nofit", "xx", "--parameter-nofit", "yy",
           "--parameter-implied", "ux", "--parameter-implied", "uy",
           "--supplementary-coordinate-code", plugin,
           "--integration-parameter-range", "xx:[-1,1]", "--integration-parameter-range", "yy:[-1,1]",
           "--fit-method", "gp-matern", "--gp-matern-max-train-points", "400",
           "--sampler-method", "AV", "--internal-use-lnL",
           "--n-max", "300000", "--n-eff", "300", "--n-output-samples", "1000",
           "--fname-output-samples", "post", "--no-plots"]
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:]
    post = np.genfromtxt(os.path.join(str(tmp_path), "post.dat"), names=True)
    assert np.mean(post["xx"] > 0.9) < 0.01, np.mean(post["xx"] > 0.9)


def test_reload_with_a_different_lnL_shift(tmp_path):
    """A saved fit reloaded under another --lnL-shift-prevent-overflow returns the same likelihood. The posterior
    must agree. A constant offset cannot change the posterior, so the evidence is checked too: it must agree
    as well (the driver restores the shift); a wrong-sign reload offset would move it by ~10."""
    proc, out = _run(tmp_path, ["--fit-save-gp", os.path.join(str(tmp_path), "fit"),
                                "--fname-output-integral", "evid_fresh"])
    assert proc.returncode == 0, proc.stdout[-3000:]
    fresh = np.genfromtxt(out, names=True)
    proc2, out2 = _run(tmp_path, ["--fit-load-gp", os.path.join(str(tmp_path), "fit.pkl"),
                                  "--lnL-shift-prevent-overflow", "5", "--fname-output-integral", "evid_reload"])
    assert proc2.returncode == 0, proc2.stdout[-3000:]
    e1 = float(np.loadtxt(os.path.join(str(tmp_path), "evid_fresh")))
    e2 = float(np.loadtxt(os.path.join(str(tmp_path), "evid_reload")))
    assert abs(e1 - e2) < 0.5, (e1, e2)
    again = np.genfromtxt(out2, names=True)
    for c in ("xx", "yy"):
        assert abs(fresh[c].mean() - again[c].mean()) < 0.03
        assert abs(fresh[c].std() / again[c].std() - 1) < 0.1


def test_cupy_fit_backend_through_the_driver(tmp_path):
    try:
        import cupy as cp
        cp.zeros(1) + 1
    except Exception as exc:
        pytest.skip("no usable CUDA device: %s" % exc)
    proc, out = _run(tmp_path, ["--gp-matern-fit-backend", "cupy", "--gp-predict-backend", "cupy"])
    assert proc.returncode == 0, proc.stdout[-3000:]
    post = np.genfromtxt(out, names=True)
    x = np.column_stack([post["xx"], post["yy"]])
    assert np.all(np.abs(x.mean(0) - MU) < 0.03), x.mean(0)
    assert np.all(np.abs(x.std(0) / SIG - 1) < 0.10), x.std(0)
