"""Fit-only effective phase: seams, invertibility and native conversion contract."""
import importlib.util
from pathlib import Path
import ast
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('eccentric_phase', ROOT/'RIFT/misc/eccentric_phase.py')
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)


def test_closed_phase_and_identity_frequency():
    m1, m2, e = 18., 18., .1
    eta = .25
    x0 = (np.pi*36*ep.MTSUN*10)**(2/3)
    xa = (np.pi*36*ep.MTSUN*20)**(2/3)
    expected = (x0**(-2.5)-xa**(-2.5))/(32*eta) - 785/(8704*eta)*e**2*x0**(-2.5)*(1-(xa/x0)**(-17/3))
    assert ep.phase_advance(m1,m2,e) == pytest.approx(expected, rel=2e-15)
    assert ep.phase_advance(m1,m2,e,10,10) == 0


def test_inverse_changes_masses_e_and_periodic_seam():
    rng = np.random.default_rng(314)
    m1,m2 = rng.uniform(5,70,(2,1000))
    e = rng.uniform(0,.4,1000)
    theta = rng.uniform(0,2*np.pi,1000)
    angle = ep.inverse(m1,m2,e,theta)
    delta = ep.forward(m1,m2,e,angle)-theta
    assert np.max(abs(np.angle(np.exp(1j*delta)))) < 5e-12
    np.testing.assert_allclose(ep.features(m1,m2,e,angle), ep.features(m1,m2,e,angle+2*np.pi), atol=5e-12)
    # theta crosses its seam smoothly in the regression embedding.
    angles = ep.inverse(18,18,.1,np.array([1e-9,2*np.pi-1e-9]))
    assert np.linalg.norm(np.diff(ep.features(18,18,.1,angles),axis=0)) < 3e-9


def test_fit_names_never_change_sampling_or_prior_coordinates():
    for eccentricity in ep.ECCENTRICITY_NAMES:
        sampled = ['mc','eta',eccentricity,'meanPerAno']
        original = sampled.copy()
        fit = ep.fitting_names(sampled,sampled)
        assert sampled == original
        assert fit == ['mc','eta',eccentricity,*ep.FEATURE_NAMES]
    for names in (['mc','meanPerAno'], ['mc','eccentricity'], ['eccentricity','eccentricity_ln','meanPerAno']):
        with pytest.raises(ValueError):
            ep.fitting_names(names,names)


def test_converter_passthrough_redshift_invalid_and_no_mutation():
    names=['m1','m2','eccentricity','meanPerAno']
    x=np.array([[18.,18.,.1,1.], [18.,18.,.2,2.], [18.,18.,1.1,2.]])
    before=x.copy()
    calls=[]
    def native(x, coord_names, low_level_coord_names, **kwargs):
        calls.append(kwargs)
        out=np.column_stack([x[:,names.index(n)] if n in names else np.zeros(len(x)) for n in coord_names])
        for i,n in enumerate(coord_names):
            if n in ('m1','m2'): out[:,i]*=1+kwargs['source_redshift']
        out[1,:]=-np.inf
        return out
    fit=ep.fitting_names(names,names)
    result=ep.convert(x,fit,names,native,source_redshift=.2,enforce_kerr=True)
    np.testing.assert_array_equal(x,before)
    np.testing.assert_allclose(result[0,-2:],ep.features(21.6,21.6,.1,1.))
    np.testing.assert_allclose(result[0,:3],[21.6,21.6,.1])
    assert np.isneginf(result[1:]).all()
    assert calls == [dict(source_redshift=.2,enforce_kerr=True)]*2


@pytest.mark.parametrize('kwargs',[dict(reference_frequency=0),dict(target_frequency=np.nan)])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError): ep.forward(18,18,.1,0,**kwargs)


def test_native_scalar_vector_projection_redshift_and_aliases():
    lal=pytest.importorskip('lal')
    from RIFT import lalsimutils
    for eccentricity,value in [('eccentricity',.1),('eccentricity_ln',np.log(.1)),('eccentricity_squared',.01)]:
        names=['m1','m2',eccentricity,'meanPerAno']
        x=np.array([[15.,10.,value,.4],[20.,12.,value,6.2]])
        fit=ep.fitting_names(names,names)
        actual=ep.convert(x,fit,names,lalsimutils.convert_waveform_coordinates,source_redshift=.3)
        expected=[]
        for row in x:
            p=lalsimutils.ChooseWaveformParams()
            for n,v in zip(names,row): p.assign_param(n,v*lal.MSUN_SI if n in ('m1','m2') else v)
            p.m1*=1.3; p.m2*=1.3
            expected.append([ep.extract(p,n)/(lal.MSUN_SI if n in ('m1','m2') else 1) for n in fit])
        np.testing.assert_allclose(actual,expected,atol=3e-12)


def test_cip_scalar_and_batch_likelihood_share_projection():
    tree=ast.parse((ROOT/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py').read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='opts.eccentric_phase_coordinates' and any(isinstance(c,ast.FunctionDef) and c.name=='log_likelihood_function' for c in n.body))
    ns={'np':np,'convert_coords':lambda x:np.column_stack((x,np.sin(x[:,0]))), 'my_fit':lambda x:np.sum(x,axis=1),'my_log_prior_scale':lambda x: np.zeros(len(x))}
    for node in block.body: exec(compile(ast.Module(body=[node],type_ignores=[]),'cip-hook','exec'),ns)
    scalar=ns['log_likelihood_function'](.2,.3)
    batch=ns['log_likelihood_function'](np.array([.2,.4]),np.array([.3,.5]))
    assert scalar[0] == batch[0]
    assert scalar[0] == pytest.approx(.2+.3+np.sin(.2))


def test_cip_opt_in_layout_and_unsupported_modes():
    from types import SimpleNamespace
    tree=ast.parse((ROOT/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py').read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='opts.eccentric_phase_coordinates')
    defaults=dict(fit_method="rf",integrate_prior=False,use_eccentricity=True,use_meanPerAno=True,using_eos=False,input_tides=False,input_eos_index=False,tabular_eos_file=None,rf_transverse_spin_coordinates=None,fit_load_gp=None,fit_load_quadratic=None,contingency_unevolved_neff=None,report_best_point=False,eccentric_phase_reference_frequency=10.,eccentric_phase_target_frequency=20.)
    def run(**overrides):
        opts=SimpleNamespace(**dict(defaults,**overrides))
        names=['mc','eta','eccentricity_ln','meanPerAno']
        ns=dict(opts=opts,eccentric_phase=ep,coord_names=names.copy(),low_level_coord_names=names.copy())
        for node in block.body: exec(compile(ast.Module(body=[node],type_ignores=[]),'cip-config','exec'),ns)
        return ns
    result=run()
    assert result['low_level_coord_names']==['mc','eta','eccentricity_ln','meanPerAno']
    assert result['coord_names']==['mc','eta','eccentricity_ln',*ep.FEATURE_NAMES]
    for overrides in (dict(use_meanPerAno=False),dict(using_eos=True),dict(fit_load_gp='old.pkl'),dict(contingency_unevolved_neff='quadpuff'),dict(report_best_point=True),dict(eccentric_phase_target_frequency=-10)):
        with pytest.raises(ValueError): run(**overrides)


def test_native_kerr_rejection_does_not_abort_good_rows():
    pytest.importorskip('lal')
    from RIFT import lalsimutils
    names=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z','eccentricity','meanPerAno']
    x=np.array([[18,18,.1,.2,.3,0,0,.1,.1,1.], [18,18,1.2,0,0,0,0,0,.1,1.]])
    fit=ep.fitting_names(names,names)
    out=ep.convert(x,fit,names,lalsimutils.convert_waveform_coordinates,enforce_kerr=True)
    assert np.isfinite(out[0]).all()
    assert np.isneginf(out[1]).all()


@pytest.mark.parametrize('alias,values', [('eccentricity_ln',[np.log(.1),1000.]), ('eccentricity_squared',[.01,-.01])])
def test_alias_projection_rejects_bad_rows_without_scalar_fallback(alias,values):
    names=['m1','m2',alias,'meanPerAno']
    x=np.array([[18.,18.,values[0],1.],[18.,18.,values[1],1.]])
    saved=x.copy()
    calls=[]
    def native(x,coord_names,low_level_coord_names,**kwargs):
        calls.append(list(low_level_coord_names))
        return np.column_stack([x[:,low_level_coord_names.index(p)] for p in coord_names])
    out=ep.convert(x,ep.fitting_names(names,names),names,native)
    assert calls==[names,['m1','m2','eccentricity','meanPerAno']]
    assert np.isfinite(out[0]).all()
    assert np.isneginf(out[1]).all()
    np.testing.assert_array_equal(x,saved)



def test_cip_refuses_singular_effective_quadratic_fit(monkeypatch, tmp_path, capsys):
    import runpy
    import sys
    code = Path(__file__).resolve().parents[1]
    script = code/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py'
    monkeypatch.chdir(tmp_path)
    # One row is sufficient: rejection must precede data loading and fitting.
    np.savetxt('grid.dat', [[0,20,15,0,0,0,0,0,0,.1,1.,100,.1,100,1000]])
    args = ['--fname','grid.dat','--use-eccentricity','--use-meanPerAno',
            '--fit-method','quadratic','--eccentric-phase-coordinates','--no-plots']
    for name in ['mc','delta_mc','eccentricity','meanPerAno']:
        args += ['--parameter',name]
    monkeypatch.setattr(sys,'argv',[str(script)]+args)
    with pytest.raises(ValueError, match='quadratic.*constrained circle'):
        runpy.run_path(str(script),run_name='__main__')


@pytest.mark.parametrize('alias', ['eccentricity', 'eccentricity_squared', 'eccentricity_ln'])
def test_actual_cip_training_matches_native_prediction(monkeypatch, tmp_path, alias):
    import runpy
    import sys
    root = Path(__file__).resolve().parents[1]
    script = root/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py'
    rng = np.random.default_rng(914)
    n = 180
    m1 = rng.uniform(19, 21, n)
    m2 = rng.uniform(16, 18, n)
    e = rng.uniform(.07, .14, n)
    angle = rng.uniform(0, 2*np.pi, n)
    y = 100 - 3*(e-.1)**2 - (1-np.cos(angle))
    grid = np.c_[np.arange(n), m1, m2, np.zeros((n, 6)), e, angle,
                 y, np.full(n, .1), np.full(n, 100), np.full(n, 1000)]
    monkeypatch.chdir(tmp_path)
    np.savetxt('grid.dat', grid)
    args = ['--fname', 'grid.dat', '--fit-method', 'rf', '--no-plots',
            '--use-eccentricity', '--use-meanPerAno', '--eccentric-phase-coordinates',
            '--source-redshift', '.3']
    for name in ['mc', 'eta', alias, 'meanPerAno']:
        args += ['--parameter', name]
    monkeypatch.setattr(sys, 'argv', [str(script)] + args)
    boundary = next(i for i, line in enumerate(script.read_text().splitlines(), 1)
                    if line == 'oracle_realizations = None')
    class FitChecked(Exception):
        pass
    def trace(frame, event, arg):
        if frame.f_code.co_filename == str(script) and event == 'line' and frame.f_lineno == boundary:
            g = frame.f_globals
            native = g['dat_out_low_level_coord_names'].copy()
            native[:, g['low_level_coord_names'].index('mc')] /= 1.3
            predicted = g['convert_coords'](native)
            np.testing.assert_allclose(predicted, g['X'], rtol=2e-5, atol=3e-6)
            assert predicted.shape[1] == 5 and native.shape[1] == 4
            assert np.isfinite(g['my_fit'](predicted)).all()
            raise FitChecked
        return trace
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(FitChecked):
            runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.settrace(previous)
