"""Adapted from O4d 7a42bf08f tests: factor2 and independent integration spacing."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import weakref
import numpy as np
import pytest
from RIFT.likelihood import factored_likelihood as fl

@pytest.mark.parametrize("n",[31,32])
@pytest.mark.parametrize("factor",[1,2,4,8])
def test_roundtrip_and_retained_storage(n,factor):
    rng=np.random.default_rng(811)
    coarse=rng.normal(size=(3,n))+1j*rng.normal(size=(3,n))
    fine,report=fl.build_reflected_q_pregrid(coarse,factor=factor)
    assert fine.shape==(3,(n-1)*factor+1)
    np.testing.assert_allclose(fine[...,::factor],coarse,rtol=5e-13,atol=5e-13)
    if factor==1:
        assert fine is coarse
    else:
        assert fine.flags.owndata and fine.base is None
        assert report["retained_bytes"]==fine.nbytes

def test_factor2_transfer_failure_rolls_back_all_detectors_and_releases_storage():
    original={"H1":np.ones((2,9)),"L1":np.ones((2,9))*2}
    calls=[];cleaned=[];refs=[]
    def transfer(value):
        calls.append(value.shape[-1])
        if value.shape[-1]==17:refs.append(weakref.ref(value))
        if calls==[17,17]:raise MemoryError("forced")
        return value.copy()
    got,reports,failure=fl.prepare_reflected_q_pregrid(
        original,factor=2,transfer=transfer,cleanup=lambda:cleaned.append(True))
    assert failure["type"]=="MemoryError" and not reports and cleaned==[True]
    assert calls==[17,17,9,9]
    for det in original:np.testing.assert_array_equal(got[det],original[det])
    gc.collect();assert all(ref() is None for ref in refs)

def test_stride2_preserves_integration_nodes_and_fractional_polynomial():
    starts,fracs,per_time,stride=fl._q_sample_positions(
        np.array([10.25]),np.arange(7)*.25-.5,.25,.125,"cubic",False)
    assert not per_time and stride==2
    np.testing.assert_allclose(starts+fracs,(10.25-.5)/.125)
    x=np.arange(64,dtype=float)
    q=(x**3-2*x+1).astype(complex)[:,None]
    got=fl._cubic_Q_window_numpy(q,np.array([10]),np.array([.25]),7,time_stride=2)
    target=10.25+np.arange(7)*2
    np.testing.assert_allclose(got[0,:,0],target**3-2*target+1,rtol=2e-13)

@pytest.mark.parametrize("qdt",[0,-1,np.nan,np.inf,.3])
def test_invalid_separate_spacing_rejected_even_without_detectors(qdt):
    p=SimpleNamespace(deltaT=1.,phi=np.array([0.]))
    with pytest.raises(ValueError):
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            np.arange(2.),p,{},{},{},{},{},time_interp="cubic",q_deltaT=qdt)

def test_separate_spacing_refuses_nearest():
    p=SimpleNamespace(deltaT=1.,phi=np.array([0.]))
    with pytest.raises(ValueError):
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            np.arange(2.),p,{},{},{},{},{},time_interp="nearest",q_deltaT=.5)

@pytest.mark.parametrize("stride",[1,2])
def test_phase_cubic_stride_matches_direct_complex_contraction(stride):
    grid=np.arange(80.)
    rows=np.vstack((np.exp(.08j*grid),(1+.01*grid)*np.exp(-.05j*grid)))
    qdt=1./stride;start=8;fraction=.25
    p=SimpleNamespace(deltaT=1.,phi=np.array([.1]),theta=np.array([.2]),
        phiref=np.array([.3]),incl=np.array([.4]),psi=np.array([.5]),
        dist=np.array([fl.distMpcRef*1e6*fl.lal.PC_SI]),tref=0.)
    harmonics=np.array([[1.2+.4j,-.7+.2j]]);response=np.array([.8-.3j])
    tvals=np.arange(3.)
    lookup={"H1":np.array([[2,2],[2,-2]])}
    zero={"H1":np.zeros((2,2),complex)}
    with patch.object(fl,"_detector_geometry",return_value=(None,None)), \
         patch.object(fl,"SourcePolarizationBasis",return_value=(None,None)), \
         patch.object(fl,"SourcePropagationDirection",return_value=None), \
         patch.object(fl,"ComputeDetAMResponsePrecomputed",return_value=response), \
         patch.object(fl,"TimeDelayFromEarthCenterPrecomputed",return_value=np.array([(start+fraction)*qdt])), \
         patch.object(fl,"SphericalHarmonicsVectorized",return_value=harmonics.copy()):
        got=fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals,p,lookup,{"H1":rows},zero,zero,{"H1":0.},Lmax=2,xpy=np,
            return_lnLt=True,phase_marginalization=True,time_interp="cubic",q_deltaT=qdt)
    block=np.column_stack((rows[0],np.conj(rows[1])))
    sampled=fl._cubic_Q_window_numpy(block,np.array([start]),np.array([fraction]),3,time_stride=stride)[0]
    h=harmonics.copy();h[:,1]=np.conj(h[:,1])
    a=np.array([[response[0],np.conj(response[0])]])*h
    expected=np.abs(np.einsum("ti,i->t",sampled,np.conj(a[0])))
    np.testing.assert_allclose(got[0],expected,rtol=2e-13,atol=2e-13)

def test_chunk_wrapper_forwards_independent_q_spacing_to_every_row_chunk():
    p=SimpleNamespace(deltaT=1.,phi=np.arange(5.),theta=np.zeros(5),
        phiref=np.zeros(5),incl=np.zeros(5),psi=np.zeros(5),dist=np.ones(5))
    calls=[]
    def evaluate(tvals,p,*args,**kwargs):
        calls.append((p.phi.copy(),p.deltaT,kwargs["q_deltaT"],np.array(tvals)))
        return p.phi+10
    nodes=np.arange(4.)
    with patch.object(fl,"_bandlimited_noloop_chunk_rows",return_value=2), \
         patch.object(fl,"_DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopUnchunked",side_effect=evaluate), \
         patch.object(fl.time_quad,"last_report",return_value={}):
        got=fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            nodes,p,{"H1":np.array([[2,2]])},{},{},{},{},
            time_interp="cubic",time_quadrature="bandlimited",q_deltaT=.5)
    np.testing.assert_array_equal(got,np.arange(5.)+10)
    assert [len(c[0]) for c in calls]==[2,2,1]
    assert all(c[1:3]==(1.,.5) for c in calls)
    for c in calls:np.testing.assert_array_equal(c[3],nodes)

def test_export_passes_q_spacing_explicitly_and_pregrid_failure_is_not_swallowed():
    driver=Path(fl.__file__).parents[2]/"bin/integrate_likelihood_extrinsic_batchmode"
    tree=ast.parse(driver.read_text())
    resample=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="resample_samples")
    assert "q_deltaT" in [a.arg for a in resample.args.args]
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=="resample_samples"]
    assert calls and all(any(k.arg=="q_deltaT" for k in n.keywords) for n in calls)
    handlers=[n for n in ast.walk(tree) if isinstance(n,ast.ExceptHandler)
              and isinstance(n.type,ast.Name) and n.type.id=="Exception"]
    assert any(any(isinstance(x,ast.If) and "QTimePregridError" in ast.unparse(x.test)
                   and any(isinstance(y,ast.Raise) for y in x.body) for x in n.body)
               for n in handlers)
