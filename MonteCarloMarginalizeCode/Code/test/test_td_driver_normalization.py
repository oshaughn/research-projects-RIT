"""The real NoLoop driver callback must retain Gaussian normalization in exports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('constant', [0.0, -30.0, 5.0, 1000.0])
@pytest.mark.parametrize('log_return', [False, True])
def test_glitch_constant_in_integral_and_saved_likelihood(constant, log_return):
    # Extract the driver's actual callback and initialization block, without executing
    # the CLI's unrelated data loading. Stub only the underlying NoLoop statistic:
    # this test concerns driver accounting, not the separately tested quadratic form.
    source = Path(__file__).parents[1] / 'bin/integrate_likelihood_extrinsic_batchmode'
    tree = ast.parse(source.read_text())
    analyze = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze_event')
    start = next(i for i, n in enumerate(analyze.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_glitch_lnL_constant' for t in n.targets))
    init = analyze.body[start:start+3]
    callback = next(n for n in ast.walk(analyze) if isinstance(n, ast.FunctionDef)
                    and n.name == 'likelihood_function' and len(n.args.args) == 6
                    and any(isinstance(x, ast.Attribute) and x.attr ==
                    'DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop' for x in ast.walk(n)))
    raw = np.array([-2., -1.])
    flags = ['declination_cosine_sampler', 'inclination_cosine_sampler',
             'd_prior_redshift', 'internal_reparam_dl_incl',
             'internal_sky_network_coordinates', 'internal_rotate_phase',
             'rotation_slow', 'freqresponse']
    opts = SimpleNamespace(**dict.fromkeys(flags, False), zero_likelihood=False,
                           l_max=2, _noloop_time_interp='nearest')
    namespace = dict(np=np, numpy=np, xpy_default=np, opts=opts,
        P=SimpleNamespace(deltaT=1/1024), fiducial_epoch=0., t_ref_wind=.1,
        factored_likelihood=SimpleNamespace(marginalization_time_grid=lambda *a, **kw: np.arange(4),
            DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop=lambda *a, **kw: raw.copy()),
        lalsimutils=SimpleNamespace(lsu_PC=3.085677581491367e16),
        lookupNKDict={}, rholmArrayDict={}, ctUArrayDict={}, ctVArrayDict={}, epochDict={},
        n_cal_for_likelihood=1, use_fused_calmarg=False, calibration_log_weights=None,
        td_subsample_for_likelihood=1,
        ctUArrayDict_cal=None, ctVArrayDict_cal=None, supplemental_ln_likelihood=None,
        return_lnL=log_return, manual_avoid_overflow_logarithm=17.,
        identity_convert_lnL=lambda x:x,
        ile_ip_factory=SimpleNamespace(glitch_delta_lnZ_total=lambda: constant))
    code = ast.fix_missing_locations(ast.Module(body=[*init, callback], type_ignores=[]))
    exec(compile(code, str(source), 'exec'), namespace)
    x = np.array([0.2, 0.3])
    value = namespace['likelihood_function'](x, x, x, x, x, np.array([100., 200.]))
    # A thousand-nat data constant must not overflow the sampler's linear integrand.
    assert np.all(np.isfinite(value))
    sampler_log = value if log_return else np.log(value)
    np.testing.assert_allclose(sampler_log, raw-17., atol=1e-12, rtol=0)
    # This is the same offset restoration used by evidence and sample-export writers.
    restored = sampler_log + namespace['manual_avoid_overflow_logarithm']
    np.testing.assert_allclose(restored, raw+constant, atol=1e-12, rtol=0)


@pytest.mark.parametrize('constant', [0.0, -30.0, 5.0, 1000.0])
def test_arrival_resampling_restores_same_normalization(constant):
    source = Path(__file__).parents[1] / 'bin/integrate_likelihood_extrinsic_batchmode'
    tree = ast.parse(source.read_text())
    analyze = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze_event')
    branch = next(n for n in ast.walk(analyze) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Attribute)
                  and n.test.attr == 'resample_time_marginalization'
                  and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                          and x.func.id == 'resample_samples' for x in ast.walk(n)))
    raw, times = np.array([-2., -1.]), np.array([-.01, .02])
    point = object()
    def draw(samples, *args, **kwargs):
        assert kwargs['P'] is point   # the CURRENT point's P, not the module-global P_list[0]
        return dict(lnL_raw=raw.copy(), t_ref=times.copy())
    ns = dict(samples={}, resample_samples=draw, _glitch_lnL_constant=constant,
              lookupNKDict={}, rholmArrayDict={}, ctUArrayDict={}, ctVArrayDict={},
              epochDict={}, n_cal_for_likelihood=1, calibration_log_weights=None,
              ctUArrayDict_cal=None, ctVArrayDict_cal=None, P=point)
    # Execute the actual driver export, before unrelated calibration exports.
    block = ast.fix_missing_locations(ast.Module(body=branch.body[:3], type_ignores=[]))
    exec(compile(block, str(source), 'exec'), ns)
    np.testing.assert_array_equal(ns['samples']['t_ref'], times)
    np.testing.assert_allclose(ns['samples']['lnL_raw'], raw+constant, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ns['samples']['loglikelihood'], raw+constant, rtol=0, atol=1e-12)
