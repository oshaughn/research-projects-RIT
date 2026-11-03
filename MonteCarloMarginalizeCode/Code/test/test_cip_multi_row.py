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


def test_source_is_read_before_the_fit():
    """Row mode re-runs the post-fit source; it must be the source read at startup, not a file
    that a git pull may have changed during the fit."""
    import ast
    with open(DRIVER) as f:
        tree = ast.parse(f.read())
    first_import = next(i for i, n in enumerate(tree.body) if isinstance(n, (ast.Import, ast.ImportFrom)))
    reads = [i for i, n in enumerate(tree.body) if isinstance(n, ast.With)
             and "_OWN_SOURCE" in ast.dump(n)]
    assert reads and reads[0] < first_import
    assert "open(__file__)" not in ast.get_source_segment(open(DRIVER).read(),
                                                          next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_run_row_passes"))

# The injected factory isolates the proposed-EOS constructor boundary without
# depending on which native EOS coefficient sets a particular LAL build accepts.
EOS_FACTORY_PATCH = '''
import os
if os.environ.get("CIP_ROWTEST_CONSTRUCTOR"):
    import numpy as np
    import RIFT.physics.EOSManager as manager
    class Model:
        mMinMsun = .1
        mMaxMsun = 1000.
        calls = 0
        evaluation = None
        def lambda_from_m(self, mass):
            self.calls += 1
            if self.evaluation == "lambda_nan" or (self.evaluation == "late_valid" and self.calls <= 100):
                return np.nan
            return 0.
        def lambda_from_m_vector(self, masses):
            order = np.arange(self.calls+1, self.calls+np.size(masses)+1)
            self.calls += np.size(masses)
            if self.evaluation == "lambda_nan":
                return np.full_like(masses, np.nan, dtype=float)
            if self.evaluation == "late_valid":
                return np.where(order <= 100, np.nan, 0.).reshape(np.shape(masses))
            return np.zeros_like(masses, dtype=float)
    def construct(name=None, spec_params=None, **kwargs):
        invalid = os.environ.get("CIP_ROWTEST_BAD_EOS") == "all" or spec_params["gamma1"] == 2.
        mode = os.environ.get("CIP_ROWTEST_BAD_EOS")
        if invalid and mode == "raise":
            raise RuntimeError("synthetic proposed EOS constructor failure")
        if invalid and mode == "oserror":
            raise OSError("synthetic proposed EOS I/O failure")
        if invalid and mode == "importerror":
            raise ImportError("synthetic proposed EOS implementation failure")
        if invalid and mode == "all":
            raise RuntimeError("synthetic proposed EOS constructor failure")
        model = Model()
        if invalid and mode in ("lambda_nan", "late_valid"):
            model.evaluation = mode
        if invalid and mode == "nanmax":
            model.mMaxMsun = float("nan")
        if invalid and mode == "nanmin":
            model.mMinMsun = float("nan")
        return model
    manager.EOSLindblomSpectral = construct
    if os.environ.get("CIP_ROWTEST_SAMPLER_ERROR"):
        import RIFT.integrators.mcsampler as sampler_module
        def fail_sampler(self, function, *args, **kwargs):
            function(np.array([30.]), np.array([.5]))
            raise Exception("synthetic unrelated sampler failure")
        sampler_module.MCSampler.integrate = fail_sampler
'''


def _constructor_args(workdir, count=3):
    (workdir / "sitecustomize.py").write_text(EOS_FACTORY_PATCH)
    np.savetxt(workdir / "eos_proposals.dat", [[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 3, 0]],
               header="lnL sigma_lnL gamma1 gamma2")
    return ["--using-eos", "file:" + str(workdir / "eos_proposals.dat"),
            "--eos-param", "spectral", "--using-eos-index", "0",
            "--n-events-to-analyze", str(count), "--fname-output-integral", "MARG-0-0",
            "--fname-output-samples", "MARG-0-0", "--save-hyperfile-only"]


@pytest.mark.parametrize("mode", ["raise", "nanmax", "nanmin", "oserror", "importerror"])
def test_proposed_eos_constructor_failure_rejects_only_its_row(workdir, mode):
    args = _constructor_args(workdir)
    _, run_dir = _run(workdir, "eos_constructor_"+mode, args,
        env_extra={"CIP_ROWTEST_CONSTRUCTOR": "1", "CIP_ROWTEST_BAD_EOS": mode})
    rows = [_annotation(run_dir / ("MARG-0-%d+annotation.dat" % row))[1][0] for row in range(3)]
    assert np.isfinite(rows[0][0]) and np.isfinite(rows[2][0])
    assert np.isneginf(rows[1][0]) and rows[1][1] == 0
    np.testing.assert_array_equal(np.array(rows)[:, 2:], [[1, 0], [2, 0], [3, 0]])
    assert not list(run_dir.glob("*.xml*"))
    # Same physical fit/prior for both valid proposals: no stale state or floor.
    assert abs(rows[0][0]-rows[2][0]) < 5*np.hypot(rows[0][1], rows[2][1]) + .02


@pytest.mark.parametrize("chunk", [False, True])
def test_all_invalid_proposed_eos_rows_keep_zero_evidence(workdir, chunk):
    args = _constructor_args(workdir)
    args = [arg for arg in args if arg != "--save-hyperfile-only"]
    if chunk:
        args += ["--chunk-save"]
    _, run_dir = _run(workdir, "eos_all_invalid_"+str(chunk), args,
        env_extra={"CIP_ROWTEST_CONSTRUCTOR": "1", "CIP_ROWTEST_BAD_EOS": "all"})
    if chunk:
        header, rows = _annotation(run_dir / "MARG-0-0+annotation.dat")
        assert header.split() == ["#", "lnL", "sigma_lnL", "gamma1", "gamma2"]
        np.testing.assert_array_equal(rows[:, 2:], [[1, 0], [2, 0], [3, 0]])
        assert np.isneginf(np.loadtxt(run_dir / "MARG-0-0.dat")).all()
    else:
        rows = np.array([_annotation(run_dir / ("MARG-0-%d+annotation.dat" % row))[1][0] for row in range(3)])
        assert all(np.isneginf(np.loadtxt(run_dir / ("MARG-0-%d.dat" % row))) for row in range(3))
    assert np.isneginf(rows[:, 0]).all() and (rows[:, 1] == 0).all()
    assert not list(run_dir.glob("*.xml*"))


def test_single_proposed_eos_constructor_failure_is_controlled(workdir):
    args = _constructor_args(workdir, count=1)
    _, run_dir = _run(workdir, "eos_single_invalid", args,
        env_extra={"CIP_ROWTEST_CONSTRUCTOR": "1", "CIP_ROWTEST_BAD_EOS": "all"})
    _, rows = _annotation(run_dir / "MARG-0-0+annotation.dat")
    assert np.isneginf(rows[0, 0]) and rows[0, 1] == 0
    assert not list(run_dir.glob("*.xml*"))


def test_hypercombine_rejected_replicas_preserve_finite_average_and_product(workdir):
    def combine(name, rows, mode="average"):
        path = workdir / (name+".dat")
        np.savetxt(path, rows, header="lnL sigma_lnL model")
        result = subprocess.run([sys.executable, HYPERCOMBINE, str(path), "--combination", mode],
            capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=CODE), timeout=60)
        assert result.returncode == 0, result.stderr
        from io import StringIO
        return np.atleast_2d(np.loadtxt(StringIO(result.stdout)))
    finite = [[3., .1, 1.], [4., .2, 1.], [8., .3, 2.]]
    baseline = combine("combine_finite", finite)
    mixed = combine("combine_mixed", finite+[[-np.inf, 0., 1.], [-np.inf, 0., 3.]])
    np.testing.assert_array_equal(mixed[:2], baseline)
    np.testing.assert_array_equal(mixed[2], [-np.inf, 0., 3.])
    rejected = combine("combine_rejected", [[-np.inf, 0., 3.], [-np.inf, 0., 3.]])
    np.testing.assert_array_equal(rejected, [[-np.inf, 0., 3.]])
    product = combine("combine_product", [[3., .1, 1.], [-np.inf, 0., 1.]], "product")
    assert np.isneginf(product[0, 0]) and product[0, 2] == 1.


def test_hyperprior_eos_constructor_rejection_keeps_later_proposals(workdir):
    plugin = PLUGIN + '''
class EOS:
    mMinMsun = .1
    mMaxMsun = 1000.
    def lambda_from_m(self, mass):
        return 0.
    def lambda_from_m_vector(self, mass):
        return np.zeros_like(mass, dtype=float)
def retrieve_eos(**kwargs):
    if _state["center"] == %r:
        raise RuntimeError("invalid proposed hyperprior EOS")
    return EOS()
''' % float(CENTERS[1])
    (workdir / "cip_rowtest_eos_plugin.py").write_text(plugin)
    args = _grid_args(workdir, 0, 3, "MARG-0-0") + ["--save-hyperfile-only"]
    args[args.index("cip_rowtest_plugin")] = "cip_rowtest_eos_plugin"
    _, run_dir = _run(workdir, "hyperprior_bad_eos", args)
    rows = np.array([_annotation(run_dir / ("MARG-0-%d+annotation.dat" % row))[1][0] for row in range(3)])
    assert np.isfinite(rows[[0, 2], 0]).all()
    assert np.isneginf(rows[1, 0]) and rows[1, 1] == 0
    np.testing.assert_array_equal(rows[:, 2], CENTERS[:3])
    expected = _expected_lnZ_up_to_constant(CENTERS[2])-_expected_lnZ_up_to_constant(CENTERS[0])
    assert abs(rows[2, 0]-rows[0, 0]-expected) < 5*np.hypot(rows[0, 1], rows[2, 1])+.02


def test_fixed_eos_actual_fit_branch_rejects_bad_conversion_without_floor():
    """Execute the driver's actual EOS branch, including its unconditional guard."""
    import ast
    import types
    from RIFT import lalsimutils
    from RIFT.physics.lalsim_eos_compat import mass_in_eos_support
    source = open(DRIVER).read()
    tree = ast.parse(source)
    block = next(node for node in tree.body if isinstance(node, ast.If)
        and "using_eos" in ast.dump(node.test) and any(
            isinstance(item, ast.FunctionDef) and item.name == "convert_coords" for item in node.orelse))
    class EOS:
        mMinMsun = 1.
        mMaxMsun = 3.
        def lambda_from_m(self, mass):
            mass = mass/lalsimutils.lal.MSUN_SI if mass > 1e15 else mass
            if np.isclose(mass, 1.5):
                raise RuntimeError("bad proposed star")
            if np.isclose(mass, 1.6):
                return np.nan
            return 100.
        def lambda_from_m_vector(self, masses):
            return np.array([self.lambda_from_m(mass) for mass in masses])
    seen = []
    def predictor(rows):
        seen.append(rows.copy())
        # A legitimate tiny finite likelihood must not become a rejected row.
        return -1e6-100*rows[:, 0]
    namespace = dict(np=np, opts=types.SimpleNamespace(using_eos="test", no_matter1=False,
        no_matter2=False, downselect_enforce_kerr=False, fref=20), fake_eos=False,
        my_eos=EOS(), my_fit=predictor, coord_names=['mc', 'eta'],
        low_level_coord_names=['m1', 'm2'], lalsimutils=lalsimutils,
        mass_in_eos_support=mass_in_eos_support, source_redshift=0.)
    top_guard = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "protect_fit_against_out_of_support"]
    exec(compile(ast.Module(body=top_guard+[block], type_ignores=[]), DRIVER, 'exec'), namespace)
    points = np.array([[1.8, 1.2], [1.5, 1.2], [1.6, 1.2], [1.9, 1.2]])
    converted = namespace['convert_coords'](points)
    result = namespace['my_fit'](converted)
    assert np.isneginf(result[1:3]).all()
    expected = lalsimutils.convert_waveform_coordinates(points[[0, 3]],
        coord_names=['mc', 'eta'], low_level_coord_names=['m1', 'm2'])
    np.testing.assert_array_equal(result[[0, 3]], -1e6-100*expected[:, 0])
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], expected)
    # Execute the actual unified likelihood wrapper as well: scalar shortcuts
    # must not bypass the EOS conversion that rejects an in-range failing star.
    namespace['opts'].eccentric_phase_coordinates = False
    namespace['my_log_prior_scale'] = lambda rows: np.zeros(len(rows))
    unified = next(node for node in tree.body if isinstance(node, ast.If)
        and 'eccentric_phase_coordinates' in ast.dump(node.test) and any(
            isinstance(item, ast.FunctionDef) and item.name == 'log_likelihood_function'
            for item in node.body))
    exec(compile(ast.Module(body=[unified], type_ignores=[]), DRIVER, 'exec'), namespace)
    for index in (0, 1, 2, 3):
        value = namespace['log_likelihood_function'](*map(float, points[index]))
        assert np.ndim(value) == 0
        assert value == result[index]
    assert namespace['likelihood_function'](*map(float, points[1])) == 0.
    vector = namespace['log_likelihood_function'](points[:, 0], points[:, 1])
    np.testing.assert_array_equal(vector, result)


@pytest.mark.parametrize('method, parameters', [('unknown', [1., 0.]), ('spectral', [1., 0., 0.])])
def test_invalid_eos_grid_configuration_is_not_a_rejected_proposal(method, parameters):
    import ast
    import types
    tree = ast.parse(open(DRIVER).read())
    names = {'_RejectedEOSProposal', '_construct_eos_proposal', '_eos_from_grid_row'}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = dict(np=np, opts=types.SimpleNamespace(eos_param=method))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), DRIVER, 'exec'), namespace)
    with pytest.raises(ValueError):
        namespace['_eos_from_grid_row']('file:proposal.dat', np.array([0., 0.]+parameters))


def test_missing_native_constructor_api_is_configuration_error():
    import ast
    import types
    tree = ast.parse(open(DRIVER).read())
    names = {'_RejectedEOSProposal', '_construct_eos_proposal', '_eos_from_grid_row'}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = dict(np=np, opts=types.SimpleNamespace(eos_param='spectral', no_use_lal_eos=False),
                     lalsim=types.SimpleNamespace())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), DRIVER, 'exec'), namespace)
    with pytest.raises(RuntimeError, match='lacks required EOS API'):
        namespace['_eos_from_grid_row']('file:proposal.dat', np.array([0., 0., 1., 0.]))
    for bounds in ({}, {'mMinMsun': None, 'mMaxMsun': None}, {'mMinMsun': 0., 'mMaxMsun': 3.}):
        eos = types.SimpleNamespace(**bounds)
        assert namespace['_construct_eos_proposal'](lambda: eos) is eos


def test_incomplete_native_family_api_is_configuration_error():
    import ast
    import types
    tree = ast.parse(open(DRIVER).read())
    names = {'_RejectedEOSProposal', '_construct_eos_proposal', '_eos_from_grid_row'}
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    incomplete = types.SimpleNamespace(SimNeutronStarEOS4ParameterSpectralDecomposition=lambda *a: None,
                                      SimNeutronStarFamMinMassPerBranch=lambda *a: None)
    namespace = dict(np=np, opts=types.SimpleNamespace(eos_param='spectral', no_use_lal_eos=False),
                     lalsim=incomplete)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), DRIVER, 'exec'), namespace)
    with pytest.raises(RuntimeError, match='API is incomplete'):
        namespace['_eos_from_grid_row']('file:proposal.dat', np.array([0., 0., 1., 0.]))


@pytest.mark.parametrize('error_type', [OSError, ImportError])
def test_proposed_factory_exception_types_are_sample_local(error_type):
    import ast
    tree = ast.parse(open(DRIVER).read())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in {'_RejectedEOSProposal', '_construct_eos_proposal'}]
    namespace = dict(np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), DRIVER, 'exec'), namespace)
    def factory():
        raise error_type('failure inside proposed EOS factory')
    with pytest.raises(namespace['_RejectedEOSProposal'], match='inside proposed EOS factory'):
        namespace['_construct_eos_proposal'](factory)


@pytest.mark.parametrize('missing', ['grid', 'plugin'])
def test_static_eos_inputs_are_not_rejected_proposals(workdir, missing):
    args = _grid_args(workdir, 0, 1, 'MARG-0-0') + ['--save-hyperfile-only']
    if missing == 'grid':
        args[args.index('file:'+str(workdir/'grid.dat'))] = 'file:'+str(workdir/'missing-grid.dat')
    else:
        args[args.index('cip_rowtest_plugin')] = 'missing_cip_eos_test_plugin'
    result, run_dir = _run(workdir, 'static_missing_'+missing, args, expect_ok=False)
    assert result.returncode != 0
    assert not list(run_dir.glob('*annotation.dat'))


@pytest.mark.parametrize('evaluation', ['lambda_nan', 'late_valid'])
def test_eos_sampling_budget_rejects_allbad_model_but_allows_later_valid_batch(workdir, evaluation):
    args = _constructor_args(workdir) + ['--n-max', '2000', '--n-eff', '10', '--n-chunk', '50']
    result, run_dir = _run(workdir, 'eos_sampling_'+evaluation, args,
        env_extra={'CIP_ROWTEST_CONSTRUCTOR': '1', 'CIP_ROWTEST_BAD_EOS': evaluation})
    rows = np.array([_annotation(run_dir / ('MARG-0-%d+annotation.dat' % row))[1][0] for row in range(3)])
    assert np.isfinite(rows[[0, 2], 0]).all()
    np.testing.assert_array_equal(rows[:, 2:], [[1, 0], [2, 0], [3, 0]])
    assert 'No contribution to integral, skipping.' in result.stdout
    if evaluation == 'lambda_nan':
        assert np.isneginf(rows[1, 0]) and rows[1, 1] == 0.
        assert result.stdout.count('No contribution to integral, skipping.') >= 10
    else:
        assert np.isfinite(rows[1, 0])
        assert 'Rejecting proposed EOS row' not in result.stdout


def test_unrelated_sampler_failure_after_eos_rejections_is_not_hidden(workdir):
    args = _constructor_args(workdir, count=1)
    args[args.index('--using-eos-index')+1] = '1'
    result, run_dir = _run(workdir, 'unrelated_sampler_failure', args, expect_ok=False,
        env_extra={'CIP_ROWTEST_CONSTRUCTOR': '1', 'CIP_ROWTEST_BAD_EOS': 'lambda_nan',
                   'CIP_ROWTEST_SAMPLER_ERROR': '1'})
    assert result.returncode != 0
    assert 'synthetic unrelated sampler failure' in result.stdout
    assert not list(run_dir.glob('*annotation.dat'))


def test_only_known_exhausted_all_eos_rejection_signal_is_translated():
    import ast
    import types
    tree = ast.parse(open(DRIVER).read())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == '_exhausted_rejected_eos_proposal')
    namespace = dict(opts=types.SimpleNamespace(sampler_method='adaptive_cartesian',
        using_eos='file:proposals.dat', using_eos_index=0))
    exec(compile(ast.Module(body=[node], type_ignores=[]), DRIVER, 'exec'), namespace)
    signal = 'mcsampler: Too many iterations with no contribution to integral, hard fail'
    namespace['_eos_sampling_status'] = dict(seen=20, all_rejected=True)
    assert namespace['_exhausted_rejected_eos_proposal'](Exception(signal))
    assert not namespace['_exhausted_rejected_eos_proposal'](RuntimeError(signal))
    assert not namespace['_exhausted_rejected_eos_proposal'](Exception('unrelated'))
    for state in (dict(seen=0, all_rejected=True), dict(seen=20, all_rejected=False)):
        namespace['_eos_sampling_status'] = state
        assert not namespace['_exhausted_rejected_eos_proposal'](Exception(signal))
