"""Rows the coordinate conversion cannot map must get the lnL FLOOR from the rf fit, not a ceiling.

fit_rf (CIP and EOSPosterior) and fit_xg (CIP) fill nonfinite or |x|>1e37 query rows with
lnL_default_large_negative = -500.  A sign error made the fill +500, so any nonfinite row the
sampler gives prior weight to took the whole posterior.  Two drivers, two in-box routes:

* EOSPosterior: a coordinate plugin that returns NaN on part of the box (xx > 0.9).
* CIP: --downselect-enforce-kerr, which sets a Kerr-violating row to -inf.  With s1x,s1y,s1z
  sampled on [-1,1]^3 about half the box violates the bound.  chi1 as a fit coordinate is what
  routes the conversion through the per-row Kerr test.

In both, lnL peaks at 20, so a +500 fill shows up as ln Z near 500 and every sample in the
unmappable region.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
EOS_DRIVER = os.path.join(CODE, "bin", "util_ConstructEOSPosterior.py")
CIP_DRIVER = os.path.join(CODE, "bin", "util_ConstructIntrinsicPosterior_GenericCoordinates.py")

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


def _run(cmd, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(OMP_NUM_THREADS="1", MPLBACKEND="Agg", XDG_CACHE_HOME=os.path.join(cwd, "cache"),
               MPLCONFIGDIR=os.path.join(cwd, "mpl"))
    return subprocess.run([sys.executable] + cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)


def test_eos_rf_nonfinite_conversion_gets_the_floor(tmp_path):
    """The likelihood peaks at xx=0.85, next to the NaN strip, so a ceiling fill would put most
    samples in xx > 0.9.  Training rows stay off the strip: fit_rf cannot train on a NaN row."""
    d = str(tmp_path)
    plugin = os.path.join(d, "nan_plugin.py")
    with open(plugin, "w") as f:
        f.write(_NAN_PLUGIN)
    fname = os.path.join(d, "grid.dat")
    rng = np.random.default_rng(4)
    x = rng.uniform(-1.0, 1.0, (800, 2))
    x[:, 0] = rng.uniform(-1.0, 0.9, len(x))
    lnL = 20.0 - 0.5 * (((x[:, 0] - 0.85) / 0.3) ** 2 + (x[:, 1] / 0.3) ** 2)
    np.savetxt(fname, np.column_stack([lnL, 0.05 * np.ones(len(x)), x]), header=" lnL sigma_lnL xx yy")
    proc = _run([EOS_DRIVER, "--fname", fname,
                 "--parameter-nofit", "xx", "--parameter-nofit", "yy",
                 "--parameter-implied", "ux", "--parameter-implied", "uy",
                 "--supplementary-coordinate-code", plugin,
                 "--integration-parameter-range", "xx:[-1,1]", "--integration-parameter-range", "yy:[-1,1]",
                 "--fit-method", "rf", "--sampler-method", "AV", "--internal-use-lnL",
                 "--n-max", "300000", "--n-eff", "300", "--n-output-samples", "1000",
                 "--fname-output-samples", "post", "--no-plots"], d)
    assert proc.returncode == 0, proc.stdout[-3000:]
    post = np.genfromtxt(os.path.join(d, "post.dat"), names=True)
    assert np.mean(post["xx"] > 0.9) < 0.01, np.mean(post["xx"] > 0.9)


def test_cip_rf_kerr_violating_rows_get_the_floor(tmp_path):
    d = str(tmp_path)
    rng = np.random.default_rng(7)
    n = 400
    m1 = rng.uniform(25.0, 45.0, n)
    m2 = np.minimum(rng.uniform(15.0, 30.0, n), m1)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    s = rng.uniform(-0.5, 0.5, (n, 3))
    lnL = 20.0 - 0.5 * ((mc - 28.0) / 1.5) ** 2
    z = np.zeros(n)
    np.savetxt(os.path.join(d, "ile.dat"),
               np.column_stack([np.arange(n), m1, m2, s, z, z, z, lnL,
                                np.full(n, 0.01), np.full(n, 1000.0), np.full(n, 100.0)]))
    proc = _run([CIP_DRIVER, "--fname", "ile.dat",
                 "--parameter", "mc", "--parameter", "delta_mc", "--parameter-implied", "chi1",
                 "--parameter-nofit", "s1z", "--parameter-nofit", "s1x", "--parameter-nofit", "s1y",
                 "--mc-range", "[24,32]", "--chi-max", "1", "--downselect-enforce-kerr",
                 "--fit-method", "rf", "--sampler-method", "AV",
                 "--n-max", "200000", "--n-eff", "500", "--n-output-samples", "500",
                 "--fname-output-samples", "post", "--no-plots"], d)
    # lnZ first: with the +500 fill CIP writes lnZ ~ 502 and then exits 1 ("no export data"),
    # so checking the exit code first would fail on that crash, not on the fill.
    fname_lnZ = os.path.join(d, "integral_result.dat")
    assert os.path.exists(fname_lnZ), proc.stdout[-3000:]
    lnZ = float(np.loadtxt(fname_lnZ).ravel()[0])
    assert lnZ < 25.0, lnZ    # fixed: 20.8-21.5 over 7 data seeds; +500 fill: ~502
    assert proc.returncode == 0, proc.stdout[-3000:]
    sys.path.insert(0, CODE)
    from RIFT import lalsimutils
    P = lalsimutils.xml_to_ChooseWaveformParams_array(os.path.join(d, "post.xml.gz"))
    chi1 = np.array([np.sqrt(p.s1x ** 2 + p.s1y ** 2 + p.s1z ** 2) for p in P])
    assert len(chi1) > 0 and np.all(chi1 <= 1.0), chi1.max()
