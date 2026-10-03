"""CIP --n-events-to-analyze > 1, --chunk-save and --save-hyperfile-only, run end to end.

create_eos_posterior_pipeline hands CIP a chunk of N consecutive grid rows per job
(--using-eos-index <first> --n-events-to-analyze N).  CIP used to evaluate only the first
row, so (N-1)/N of the grid was silently skipped.  The contract checked here is that one
job with N rows writes the same files, under the same names, with statistically the same
integrals, as N separate jobs with N=1 -- including a chunk that starts at row 0 and one
that runs past the end of the grid.

The problem is synthetic and has a known answer.  lnL(mc) is exactly quadratic, so
--fit-method quadratic reproduces it.  A supplementary-likelihood plugin adds a Gaussian in
mc centred on the grid row's hyperparameter `mc_center`, and reports a row-dependent
offset, so every row has a different evidence.  The mc prior is proportional to mc and
separable from delta_mc, so differences of ln Z between rows follow from 1-D quadrature.
"""
import os
import glob
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
DRIVER = os.path.join(CODE, "bin", "util_ConstructIntrinsicPosterior_GenericCoordinates.py")
HYPERCOMBINE = os.path.join(CODE, "bin", "util_HyperCombine.py")

LNL_PEAK, MC0, MC_W = 20.0, 28.0, 1.5
PLUGIN_W = 1.0
MC_RANGE = (24.0, 32.0)
N_GRID = 12
CENTERS = np.linspace(26.0, 30.0, N_GRID)

PLUGIN = '''
import os
import numpy as np
_state = {}
def initialize_me(input_line=None, param_names=None, cip_param_names=None, **kwargs):
    _state["center"] = float(input_line[list(param_names).index("mc_center")])
    if os.environ.get("CIP_ROWTEST_FAIL_CENTER") == repr(_state["center"]):
        raise RuntimeError("synthetic failure for this row")
def prepare_lnL_hyper(config=None, coords=None):
    _state["i_mc"] = list(coords).index("mc")
def lnL_hyper(*x):
    mc = np.asarray(x[_state["i_mc"]], dtype=float)
    return -0.5*((mc - _state["center"])/%r)**2
def lnL_hyper_offset():
    return 0.1*_state["center"]
''' % PLUGIN_W


def _lnL_of_mc(mc):
    return LNL_PEAK - 0.5 * ((np.asarray(mc, dtype=float) - MC0) / MC_W) ** 2


def _expected_lnZ_up_to_constant(center):
    mc = np.linspace(MC_RANGE[0], MC_RANGE[1], 20001)
    f = mc * np.exp(_lnL_of_mc(mc) - LNL_PEAK - 0.5 * ((mc - center) / PLUGIN_W) ** 2)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return np.log(trapezoid(f, mc)) + 0.1 * center


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cip_multi_row")
    rng = np.random.default_rng(7)
    n = 300
    m1 = rng.uniform(25.0, 45.0, n)
    m2 = np.minimum(rng.uniform(15.0, 30.0, n), m1)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    z = np.zeros(n)
    np.savetxt(str(d / "ile.dat"), np.column_stack([np.arange(n), m1, m2, z, z, z, z, z, z,
                                                    _lnL_of_mc(mc), np.full(n, 0.01),
                                                    np.full(n, 1000.0), np.full(n, 100.0)]))
    np.savetxt(str(d / "grid.dat"), np.column_stack([np.zeros(N_GRID), np.zeros(N_GRID), CENTERS]),
               header="lnL sigma_lnL mc_center")
    (d / "cip_rowtest_plugin.py").write_text(PLUGIN)
    return d


def _run(workdir, name, extra, expect_ok=True, env_extra=None):
    run_dir = workdir / name
    run_dir.mkdir()
    env = dict(os.environ)
    env.update(env_extra or {})
    env["PYTHONPATH"] = os.pathsep.join([CODE, str(workdir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["OMP_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["XDG_CACHE_HOME"] = str(run_dir / "cache")
    env["MPLCONFIGDIR"] = str(run_dir / "mpl")
    cmd = [sys.executable, DRIVER, "--fname", str(workdir / "ile.dat"),
           "--parameter", "mc", "--parameter", "delta_mc", "--mc-range", str(list(MC_RANGE)),
           "--fit-method", "quadratic", "--n-max", "200000", "--n-eff", "1000",
           "--n-output-samples", "50", "--no-plots"] + extra
    proc = subprocess.run(cmd, cwd=str(run_dir), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=900)
    if expect_ok:
        assert proc.returncode == 0, "driver exited %d:\n%s" % (proc.returncode, proc.stdout[-4000:])
    return proc, run_dir


def _grid_args(workdir, index, n, prefix):
    return ["--using-eos", "file:" + str(workdir / "grid.dat"), "--using-eos-for-prior",
            "--using-eos-index", str(index), "--n-events-to-analyze", str(n),
            "--supplementary-likelihood-factor-code", "cip_rowtest_plugin",
            "--supplementary-likelihood-factor-function", "lnL_hyper",
            "--fname-output-integral", prefix, "--fname-output-samples", prefix]


def _products(run_dir):
    return sorted(os.path.basename(f) for f in glob.glob(str(run_dir / "MARG-0-*")))


def _annotation(path):
    with open(path) as f:
        header = f.readline()
    return header, np.atleast_2d(np.loadtxt(path))


@pytest.fixture(scope="module")
def single_rows(workdir):
    """N=1 reference jobs for rows 0-3 and 9-12 (12 is past the end of the 12-row grid)."""
    out = {}
    for row in list(range(4)) + list(range(9, 13)):
        out[row] = _run(workdir, "single_%d" % row, _grid_args(workdir, row, 1, "MARG-0-%d" % row))[1]
    return out


@pytest.mark.parametrize("first", [0, 9])
def test_chunk_matches_single_row_jobs(workdir, single_rows, first):
    proc, run_dir = _run(workdir, "chunk_from_%d" % first, _grid_args(workdir, first, 4, "MARG-0-%d" % first))
    rows = range(first, first + 4)
    expected_files = sorted(f for row in rows for f in _products(single_rows[row]))
    assert _products(run_dir) == expected_files
    assert not os.path.exists(str(run_dir / "MARG-0-12+annotation.dat"))   # past the end, as for N=1
    for row in rows:
        if row >= N_GRID:
            continue
        h1, a1 = _annotation(str(single_rows[row] / ("MARG-0-%d+annotation.dat" % row)))
        h4, a4 = _annotation(str(run_dir / ("MARG-0-%d+annotation.dat" % row)))
        assert h1 == h4
        np.testing.assert_array_equal(a1[0, 2:], a4[0, 2:])   # this row's hyperparameters, exactly
        assert a4[0, 2] == CENTERS[row]
        tol = 5 * np.hypot(a1[0, 1], a4[0, 1]) + 0.02
        assert abs(a1[0, 0] - a4[0, 0]) < tol, (row, a1[0, :2], a4[0, :2])


def test_rows_have_the_known_relative_evidence(workdir, single_rows):
    """Each row's ln Z, relative to row 0's, against quadrature: proves rows are not one integral."""
    a = {row: _annotation(str(single_rows[row] / ("MARG-0-%d+annotation.dat" % row)))[1][0]
         for row in range(4)}
    for row in range(1, 4):
        got = a[row][0] - a[0][0]
        want = _expected_lnZ_up_to_constant(CENTERS[row]) - _expected_lnZ_up_to_constant(CENTERS[0])
        assert abs(got - want) < 5 * np.hypot(a[row][1], a[0][1]) + 0.02, (row, got, want)


def test_chunk_save_writes_one_file(workdir, single_rows):
    proc, run_dir = _run(workdir, "chunk_save", _grid_args(workdir, 0, 4, "MARG-0-0") + ["--chunk-save"])
    # the glob create_eos_posterior_pipeline's consolidation job uses sees exactly one file
    assert [os.path.basename(f) for f in glob.glob(str(run_dir / "MARG*[0-9]+annotation.dat"))] == \
        ["MARG-0-0+annotation.dat"]
    for row in range(1, 4):
        assert not os.path.exists(str(run_dir / ("MARG-0-%d.dat" % row)))
    header, chunk = _annotation(str(run_dir / "MARG-0-0+annotation.dat"))
    assert header.split() == ["#", "lnL", "sigma_lnL", "mc_center"]
    assert chunk.shape == (4, 3)
    np.testing.assert_array_equal(chunk[:, 2], CENTERS[:4])
    assert np.loadtxt(str(run_dir / "MARG-0-0.dat")).shape == (4,)
    for row in range(4):
        a1 = _annotation(str(single_rows[row] / ("MARG-0-%d+annotation.dat" % row)))[1][0]
        assert abs(a1[0] - chunk[row, 0]) < 5 * np.hypot(a1[1], chunk[row, 1]) + 0.02
    # the hyperpipeline's consolidation step reads it
    comb = subprocess.run([sys.executable, HYPERCOMBINE, str(run_dir / "MARG-0-0+annotation.dat")],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                          env=dict(os.environ, PYTHONPATH=CODE), timeout=300)
    assert comb.returncode == 0, comb.stderr
    lines = [l for l in comb.stdout.splitlines() if l.strip() and not l.startswith("#")]
    assert len(lines) == 4


def test_save_hyperfile_only(workdir):
    proc, run_dir = _run(workdir, "hyperfile_only", _grid_args(workdir, 0, 2, "MARG-0-0") + ["--save-hyperfile-only"])
    assert _products(run_dir) == ["MARG-0-0+annotation.dat", "MARG-0-1+annotation.dat"]


def test_non_grid_single_run_still_works(workdir):
    proc, run_dir = _run(workdir, "plain", ["--fname-output-integral", "out_int", "--fname-output-samples", "out"])
    for name in ["out_int.dat", "out_int+annotation.dat", "out_int_withpriorchange+annotation.dat",
                 "out.xml.gz", "out_lnL.dat"]:
        assert os.path.exists(str(run_dir / name)), name


@pytest.mark.parametrize("extra", [["--n-events-to-analyze", "4"], ["--chunk-save"]],
                         ids=["n4", "chunk_save"])
def test_rows_without_a_grid_are_refused(workdir, extra):
    proc, run_dir = _run(workdir, "refuse_" + extra[0].strip("-"), extra, expect_ok=False)
    assert proc.returncode == 2
    assert "--using-eos file:<grid>" in proc.stdout


def test_failed_row_is_reported_and_other_rows_still_run(workdir):
    """A row that fails exits the job nonzero, as its own N=1 job would; the others are written."""
    proc, run_dir = _run(workdir, "one_row_fails", _grid_args(workdir, 0, 3, "MARG-0-0") + ["--save-hyperfile-only"],
                         expect_ok=False, env_extra={"CIP_ROWTEST_FAIL_CENTER": repr(float(CENTERS[1]))})
    assert proc.returncode == 1, proc.stdout[-3000:]
    assert "synthetic failure for this row" in proc.stdout
    assert _products(run_dir) == ["MARG-0-0+annotation.dat", "MARG-0-2+annotation.dat"]
