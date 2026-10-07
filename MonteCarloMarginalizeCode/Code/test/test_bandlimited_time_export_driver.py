"""Exercise shipped driver export function without running startup/data loading."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from RIFT.misc import xmlutils

import test_time_marginalization_wiring as fixture

DRIVER = Path(__file__).resolve().parents[1] / 'bin/integrate_likelihood_extrinsic_batchmode'


@pytest.mark.parametrize('rate', [None, 16384])
@pytest.mark.parametrize('interp', ['nearest', 'cubic'])
def test_production_export_function_returns_continuous_times_and_likelihoods(rate, interp):
    tree = ast.parse(DRIVER.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resample_samples')
    ctx = fixture._build(fixture._rholm_functions())
    p = fixture._P_vec(ctx['deltaT'])
    samples = dict(longitude=np.copy(p.phi), right_ascension=np.copy(p.phi),
                   declination=np.copy(p.theta), coa_phase=np.copy(p.phiref),
                   inclination=np.copy(p.incl), psi=np.copy(p.psi), distance=np.full(3, 1000.))
    namespace = dict(np=np, xpy_default=np, P=p, cupy_success=False,
                     identity_convert=lambda x: x, identity_convert_togpu=lambda x: x,
                     fiducial_epoch=fixture.EPOCH, fiducial_epoch_seconds=int(fixture.EPOCH),
                     fiducial_epoch_nanoseconds=0, xmlutils=xmlutils,
                     t_ref_wind=fixture.WINDOW_HALF,
                     fSample=fixture.BASE_SRATE, factored_likelihood=fixture.fl,
                     lalsimutils=fixture.lalsimutils,
                     opts=SimpleNamespace(vectorized=True, distance_marginalization=False,
                         l_max=2, time_marginalization_quadrature='bandlimited',
                         _noloop_time_interp=interp, srate_resample_time_marginalization=rate))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(DRIVER), 'exec'), namespace)
    np.random.seed(182)
    got = namespace['resample_samples'](samples, ctx['lookupNKDict'], ctx['rholmArrayDict'],
                                        ctx['ctUArrayDict'], ctx['ctVArrayDict'], ctx['epochDict'])
    assert got is samples
    t = samples['t_ref'] - fixture.EPOCH
    assert np.all(np.isfinite(samples['lnL_raw']))
    assert np.all(np.abs(t) <= fixture.WINDOW_HALF)
    # GPS float64 loses ~0.1 us, still enough to distinguish lattice draws.
    fractional = (t + fixture.WINDOW_HALF) * (rate or fixture.BASE_SRATE)
    assert np.any(np.abs(fractional - np.rint(fractional)) > .001)
    np.random.seed(182)
    expected_t, expected_lnL = fixture.fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        fixture._tvals(p.deltaT), p, ctx['lookupNKDict'], ctx['rholmArrayDict'],
        ctx['ctUArrayDict'], ctx['ctVArrayDict'], ctx['epochDict'], Lmax=2,
        xpy=np, time_interp=interp, time_quadrature='bandlimited',
        return_time_draw=True, time_draw_minimum_srate=rate)
    np.testing.assert_array_equal(samples['t_ref'], fixture.EPOCH + expected_t)
    np.testing.assert_array_equal(samples['lnL_raw'], expected_lnL)

    secs, ns = xmlutils.gps_add_seconds_exact(int(fixture.EPOCH), 0, expected_t)
    np.testing.assert_array_equal(samples['t_ref_gps_seconds'], secs)
    np.testing.assert_array_equal(samples['t_ref_gps_nanoseconds'], ns)
