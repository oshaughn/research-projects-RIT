"""Bound NoLoop's coarse workspace without changing likelihood or seeded draws."""
from types import SimpleNamespace
import numpy as np
import pytest
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood import time_marginalization_quadrature as tmq
import test_time_marginalization_wiring as wiring


def _evaluate(fun,ctx,P,**kwargs):
    return fun(wiring._tvals(ctx['deltaT']),P,ctx['lookupNKDict'],
               ctx['rholmArrayDict'],ctx['ctUArrayDict'],ctx['ctVArrayDict'],
               ctx['epochDict'],Lmax=2,xpy=np,**kwargs)


@pytest.mark.parametrize('interp',['nearest','cubic'])
@pytest.mark.parametrize('phase',[False,True])
@pytest.mark.parametrize('nonlinear',[False,True])
@pytest.mark.parametrize('draw',[False,True])
def test_bounded_rows_match_original_and_keep_order(monkeypatch,interp,phase,nonlinear,draw):
    ctx=wiring._build(wiring._rholm_functions(),amp=30)
    P=wiring._P_vec(ctx['deltaT'],n=7)
    P.dist=P.dist*np.geomspace(.2,3.,7)
    saved={key:np.array(getattr(P,key),copy=True) for key in ('phi','theta','phiref','incl','psi','dist')}
    cb=(lambda k,r:np.logaddexp(0.,k-.5*r)) if nonlinear else fl._factored_lnL_helper
    kw=dict(time_quadrature='bandlimited',time_interp=interp,
            phase_marginalization=phase,loglikelihood=cb,
            return_time_draw=draw,time_draw_minimum_srate=16384)
    if draw:kw['time_draw_uniforms']=np.random.default_rng(14).random((7,2))
    expected=_evaluate(fl._DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked,ctx,P,**kw)
    baseline=tmq.last_report()
    original=fl._DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked
    seen=[]
    def traced(*args,**kwargs):
        view=args[1];seen.append(len(view.phi))
        assert view.tref==P.tref and view.deltaT==P.deltaT
        return original(*args,**kwargs)
    monkeypatch.setattr(fl,'_bandlimited_noloop_chunk_rows',lambda *args:2)
    monkeypatch.setattr(fl,'_DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked',traced)
    result=_evaluate(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop,ctx,P,**kw)
    np.testing.assert_allclose(result,expected,rtol=0,atol=2e-10)
    assert seen==[2,2,2,1]
    for key,value in saved.items():np.testing.assert_array_equal(getattr(P,key),value)
    report=tmq.last_report()
    assert report['noloop_row_chunks']==4
    assert report['noloop_row_limit']==2
    assert report['noloop_input_rows']==report['n_rows']==7
    for key in ('factor_histogram','export_factor_histogram','n_refined_rows','n_flat_rows','upsample_factor'):
        assert report[key]==baseline[key]
    np.testing.assert_allclose(report['export_sigma_t_min'],
                               baseline['export_sigma_t_min'], rtol=1e-9)


def test_generated_uniforms_follow_original_seed_order(monkeypatch):
    ctx=wiring._build(wiring._rholm_functions(),amp=30)
    P=wiring._P_vec(ctx['deltaT'],n=7)
    kw=dict(time_quadrature='bandlimited',return_time_draw=True)
    np.random.seed(140182)
    expected=_evaluate(fl._DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked,ctx,P,**kw)
    expected_next=np.random.random(4)
    np.random.seed(140182)
    monkeypatch.setattr(fl,'_bandlimited_noloop_chunk_rows',lambda *args:2)
    result=_evaluate(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop,ctx,P,**kw)
    np.testing.assert_allclose(result,expected,rtol=0,atol=2e-10)
    np.testing.assert_array_equal(np.random.random(4),expected_next)


def test_row_view_preserves_scalar_and_singleton_broadcast_fields():
    P=SimpleNamespace(phi=np.arange(7.),theta=np.arange(7.)/10,
                      phiref=0.,incl=np.array([.7]),psi=.3,dist=np.arange(7.)+10,
                      tref=object(),deltaT=.125,intrinsic=object())
    view=fl._noloop_extrinsic_row_view(P,2,4,7)
    np.testing.assert_array_equal(view.phi,[2,3])
    np.testing.assert_array_equal(view.dist,[12,13])
    assert view.phiref==P.phiref and view.psi==P.psi
    assert view.incl is P.incl
    assert view.tref is P.tref and view.intrinsic is P.intrinsic
    assert len(P.phi)==7


@pytest.mark.parametrize('kw',[dict(time_quadrature='simpson'),dict(time_quadrature='bandlimited',return_lnLt=True)])
def test_simpson_and_coarse_timeseries_do_not_use_row_planner(monkeypatch,kw):
    ctx=wiring._build(wiring._rholm_functions());P=wiring._P_vec(ctx['deltaT'])
    def forbidden(*args):raise AssertionError('untouched path reached planner')
    monkeypatch.setattr(fl,'_bandlimited_noloop_chunk_rows',forbidden)
    expected=_evaluate(fl._DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked,ctx,P,**kw)
    result=_evaluate(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop,ctx,P,**kw)
    np.testing.assert_array_equal(result,expected)


def test_gpu_planner_bounds_coarse_work_before_allocation():
    device=SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda:(4*1024**3,24*1024**3))))
    n=74287;n_time=614;n_modes=21
    rows=fl._bandlimited_noloop_chunk_rows(n,n_time,n_modes,device)
    assert 1<=rows<=4096
    assert rows*(160*n_time+128*n_modes**2)<=min(fl._NOLOOP_BANDLIMITED_COARSE_BYTES,4*1024**3//8)
    assert fl._bandlimited_noloop_chunk_rows(n,n_time,n_modes,np)==n


@pytest.mark.parametrize('free_bytes',[32*1024**2,4*1024**3,24*1024**3])
def test_gpu_planner_respects_available_memory_at_production_internal_rate(free_bytes):
    device=SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda:(free_bytes,24*1024**3))))
    n_time=2457;n_modes=21
    rows=fl._bandlimited_noloop_chunk_rows(74287,n_time,n_modes,device)
    assert 1<=rows<=fl._NOLOOP_BANDLIMITED_MAX_ROWS
    assert rows*(160*n_time+128*n_modes**2)<=min(fl._NOLOOP_BANDLIMITED_COARSE_BYTES,free_bytes//8)


def test_report_aggregation_preserves_input_and_late_larger_transform():
    reports=[dict(n_rows=2,factor_histogram={4:2},upsample_factor=4,sigma_t_min=.1,max_dense_factor=4,max_reference_full_fft_length=100,n_retained_fft_plans=1),dict(n_rows=1,factor_histogram={16:1},upsample_factor=16,sigma_t_min=.01,max_dense_factor=16,max_reference_full_fft_length=400,n_retained_fft_plans=2)]
    merged=fl._combine_noloop_chunk_reports(reports,3,2)
    assert merged['factor_histogram']=={4:2,16:1}
    assert reports[0]['factor_histogram']=={4:2}
    assert merged['max_dense_factor']==merged['upsample_factor']==16
    assert merged['max_reference_full_fft_length']==400
    assert merged['sigma_t_min']==.01
    assert merged['n_retained_fft_plans']==3


def test_chunked_callback_output_dtype_matches_unchunked(monkeypatch):
    ctx=wiring._build(wiring._rholm_functions());P=wiring._P_vec(ctx['deltaT'],n=7)
    callback=lambda k,r:(k-.5*r).astype(np.float32)
    kwargs=dict(time_quadrature='bandlimited',loglikelihood=callback)
    expected=_evaluate(fl._DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked,ctx,P,**kwargs)
    monkeypatch.setattr(fl,'_bandlimited_noloop_chunk_rows',lambda *args:2)
    actual=_evaluate(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop,ctx,P,**kwargs)
    assert actual.dtype==expected.dtype==np.float32
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-6)
