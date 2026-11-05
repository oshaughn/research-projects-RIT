"""O4c proposed-EOS rejection and actual fixed-EOS fit boundary."""
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pytest

CODE = str(Path(__file__).resolve().parents[1])
DRIVER = os.path.join(CODE, "bin", "util_ConstructIntrinsicPosterior_GenericCoordinates.py")
HYPERCOMBINE = os.path.join(CODE, "bin", "util_HyperCombine.py")

def test_fixed_eos_actual_fit_branch_rejects_bad_conversion_without_floor():
    """Execute the driver's actual EOS branch, including its unconditional guard."""
    import ast
    import types
    from RIFT import lalsimutils
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
        source_redshift=0.)
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


EOS_FACTORY_PATCH = '''
import os
if os.environ.get("CIP_ROWTEST_CONSTRUCTOR"):
    import numpy as np
    # O4c's old quadratic fitter uses this spelling on every platform. This
    # subprocess-only alias avoids the pre-existing macOS dtype-name gap; it
    # changes neither the EOS behavior nor the production/shared environment.
    if not hasattr(np, "float128"):
        np.float128 = np.longdouble
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


def test_hypercombine_rejected_replicas_preserve_finite_average_and_product(tmp_path):
    def combine(name, rows, mode="average"):
        path = tmp_path / (name+".dat")
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



@pytest.mark.parametrize("mode", ["raise", "nanmax", "nanmin", "oserror", "importerror"])
def test_single_proposed_eos_failure_writes_zero_evidence(tmp_path, mode):
    (tmp_path/'sitecustomize.py').write_text(EOS_FACTORY_PATCH)
    grid = tmp_path/'proposals.dat'
    np.savetxt(grid, [[0., 0., 2., 0.]], header='lnL sigma_lnL gamma1 gamma2')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([CODE, str(tmp_path)]),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        MPLBACKEND='Agg', CIP_ROWTEST_CONSTRUCTOR='1', CIP_ROWTEST_BAD_EOS=mode)
    result = subprocess.run([sys.executable, DRIVER, '--fname', 'unneeded-invalid-eos-input.dat',
        '--using-eos', 'file:'+str(grid), '--eos-param', 'spectral', '--using-eos-index', '0',
        '--fname-output-integral', 'rejected', '--fname-output-samples', 'unused', '--no-plots'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+result.stderr
    row = np.atleast_2d(np.loadtxt(tmp_path/'rejected+annotation.dat'))[0]
    np.testing.assert_array_equal(row, [-np.inf, 0., 2., 0.])
    assert np.isneginf(np.loadtxt(tmp_path/'rejected.dat'))
    assert not list(tmp_path.glob('*.xml*'))


def test_missing_proposal_file_is_still_a_configuration_error(tmp_path):
    result = subprocess.run([sys.executable, DRIVER, '--using-eos', 'file:missing.dat',
        '--eos-param', 'spectral', '--using-eos-index', '0', '--no-plots'],
        cwd=tmp_path, env=dict(os.environ, PYTHONPATH=CODE), capture_output=True,
        text=True, timeout=90)
    assert result.returncode != 0
    assert not list(tmp_path.glob('*annotation.dat'))


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


@pytest.mark.parametrize('evaluation', ['lambda_nan', 'late_valid', 'unrelated'])
def test_eos_sampling_exhaustion_is_sample_local_and_waits_for_later_batches(tmp_path, evaluation):
    (tmp_path/'sitecustomize.py').write_text(EOS_FACTORY_PATCH)
    grid = tmp_path/'proposals.dat'
    np.savetxt(grid, [[0., 0., 2., 0.]], header='lnL sigma_lnL gamma1 gamma2')
    rng = np.random.default_rng(7)
    m1 = rng.uniform(25., 45., 300)
    m2 = np.minimum(rng.uniform(15., 30., 300), m1)
    mc = (m1*m2)**.6/(m1+m2)**.2
    z = np.zeros(300)
    np.savetxt(tmp_path/'ile.dat', np.column_stack([np.arange(300), m1, m2, z, z, z, z, z, z,
        20.-.5*((mc-28.)/1.5)**2, np.full(300, .01), np.full(300, 1000.), np.full(300, 100.)]))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([CODE, str(tmp_path)]), OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg',
        XDG_CACHE_HOME=str(tmp_path/'cache'), MPLCONFIGDIR=str(tmp_path/'mpl'),
        CIP_ROWTEST_CONSTRUCTOR='1', CIP_ROWTEST_BAD_EOS='lambda_nan' if evaluation == 'unrelated' else evaluation)
    if evaluation == 'unrelated':
        env['CIP_ROWTEST_SAMPLER_ERROR'] = '1'
    result = subprocess.run([sys.executable, DRIVER, '--fname', str(tmp_path/'ile.dat'),
        '--using-eos', 'file:'+str(grid), '--eos-param', 'spectral', '--using-eos-index', '0',
        '--parameter', 'mc', '--parameter', 'delta_mc', '--mc-range', '[24.,32.]',
        '--fit-method', 'quadratic', '--n-max', '2000', '--n-eff', '10', '--n-chunk', '50',
        '--fname-output-integral', 'result', '--no-plots', '--no-save-samples'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    if evaluation == 'unrelated':
        assert result.returncode != 0
        assert 'synthetic unrelated sampler failure' in result.stdout+result.stderr
        assert not (tmp_path/'result+annotation.dat').exists()
    else:
        assert result.returncode == 0, result.stdout+result.stderr
        row = np.atleast_2d(np.loadtxt(tmp_path/'result+annotation.dat'))[0]
        assert 'No contribution to integral, skipping.' in result.stdout+result.stderr
        if evaluation == 'lambda_nan':
            np.testing.assert_array_equal(row, [-np.inf, 0., 2., 0.])
            assert np.isneginf(np.loadtxt(tmp_path/'result.dat'))
        else:
            assert np.isfinite(row[0])
            assert 'Rejecting proposed EOS row' not in result.stdout
