"""O4d native factor-2 refinement: physical nodes and fatal-request semantics.

Numerical fixtures adapted from the validated rc11 port; O4d retains P.q_deltaT
transport, time-indexed self-term exclusions, and its existing NoLoop evaluator.
"""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import weakref
import numpy as np
import pytest
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood.q_time_pregrid import QTimePregridError

CODE = Path(__file__).resolve().parents[1]
DRIVER = CODE / "bin/integrate_likelihood_extrinsic_batchmode"

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

@pytest.mark.parametrize("stride",[1,2,4,8])
def test_phase_cubic_stride_matches_direct_complex_contraction(stride):
    grid=np.arange(80.)
    rows=np.vstack((np.exp(.08j*grid),(1+.01*grid)*np.exp(-.05j*grid)))
    qdt=1./stride;start=8;fraction=.25
    p=SimpleNamespace(deltaT=1.,phi=np.array([.1]),theta=np.array([.2]),
        phiref=np.array([.3]),incl=np.array([.4]),psi=np.array([.5]),
        dist=np.array([fl.distMpcRef*1e6*fl.lal.PC_SI]),tref=0.)
    p.q_deltaT=qdt
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
            return_lnLt=True,phase_marginalization=True,time_interp="cubic")
    block=np.column_stack((rows[0],np.conj(rows[1])))
    sampled=fl._cubic_Q_window_numpy(block,np.array([start]),np.array([fraction]),3,time_stride=stride)[0]
    h=harmonics.copy();h[:,1]=np.conj(h[:,1])
    a=np.array([[response[0],np.conj(response[0])]])*h
    expected=np.abs(np.einsum("ti,i->t",sampled,np.conj(a[0])))
    np.testing.assert_allclose(got[0],expected,rtol=2e-13,atol=2e-13)

@pytest.mark.parametrize("failure_stage", ["coarse_transfer", "cleanup", "spacing"])
def test_driver_pregrid_preparation_failures_are_fatal(failure_stage):
    """Execute the driver's actual block, including the helper recovery path."""
    from RIFT.likelihood.q_time_pregrid import QTimePregridError
    driver=DRIVER
    tree=ast.parse(driver.read_text())
    analyze=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="analyze_event")
    blocks=[n for n in ast.walk(analyze) if isinstance(n,ast.If)
            and ast.unparse(n.test)=="opts.q_time_pregrid_factor > 1"]
    assert len(blocks)==1
    code=compile(ast.fix_missing_locations(ast.Module(body=blocks,type_ignores=[])),str(driver),"exec")
    transfers=[];cleanups=[]
    def transfer(value):
        transfers.append(value.shape[-1])
        raise MemoryError("forced transfer")
    def cleanup():
        cleanups.append(True)
        if failure_stage=="cleanup":raise RuntimeError("forced cleanup")
    fake_cupy=SimpleNamespace(asarray=transfer,
        get_default_memory_pool=lambda:SimpleNamespace(free_all_blocks=cleanup))
    gpu=failure_stage!="spacing"
    ns=dict(opts=SimpleNamespace(q_time_pregrid_factor=2,gpu=gpu),
        np=np,xpy_default=object() if gpu else np,cupy=fake_cupy,
        rholmArrayDict={"H1":np.ones((2,9),complex)},factored_likelihood=fl,
        P=SimpleNamespace(deltaT=None if failure_stage=="spacing" else 1/4096),
        QTimePregridError=QTimePregridError)
    with pytest.raises(QTimePregridError) as caught:
        exec(code,ns)
    assert isinstance(caught.value.__cause__,{"coarse_transfer":MemoryError,
        "cleanup":RuntimeError,"spacing":TypeError}[failure_stage])
    if failure_stage=="coarse_transfer":
        assert transfers==[17,9] and cleanups==[True]
    assert "q_deltaT" not in ns


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_actual_driver_preserves_physical_grid_and_q_epochs(factor):
    tree = ast.parse(DRIVER.read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "opts.q_time_pregrid_factor > 1"
                 and "prepare_reflected_q_pregrid" in ast.unparse(node))
    transport = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                     and any(ast.unparse(target) == "P.q_deltaT" for target in node.targets))
    coarse = {det: np.ones((2, 17), complex) * scale for det, scale in [("H1", 1), ("L1", 2)]}
    epochs = {"H1": 1000000000., "L1": 1000000000.125}
    nodes = np.arange(-4, 5) / 4096
    p = SimpleNamespace(deltaT=1/4096, deltaF=1/8)
    ns = dict(opts=SimpleNamespace(q_time_pregrid_factor=factor, gpu=False),
              np=np, xpy_default=np, rholmArrayDict=coarse,
              factored_likelihood=fl, P=p, QTimePregridError=QTimePregridError,
              epochDict=epochs.copy(), tvals=nodes.copy())
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block, transport], type_ignores=[])), str(DRIVER), "exec"), ns)
    assert p.deltaT == 1/4096 and p.deltaF == 1/8
    assert p.q_deltaT == p.deltaT/factor
    assert ns["opts"].q_time_pregrid_factor == factor
    np.testing.assert_array_equal(ns["tvals"], nodes)
    assert ns["epochDict"] == epochs
    for det in coarse:
        refined = ns["rholmArrayDict"][det]
        assert refined.shape == (2, 16*factor+1)
        np.testing.assert_allclose(refined[:, ::factor], coarse[det], atol=1e-12)
        assert refined.flags.owndata


def test_second_detector_transfer_failure_cannot_run_coarse_likelihood():
    tree = ast.parse(DRIVER.read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "opts.q_time_pregrid_factor > 1"
                 and "prepare_reflected_q_pregrid" in ast.unparse(node))
    calls, cleanups = [], []
    def transfer(values):
        calls.append(values.shape[-1])
        if calls == [17, 17]:
            raise MemoryError("forced second detector transfer")
        return values.copy()
    cupy = SimpleNamespace(asarray=transfer, get_default_memory_pool=lambda:
                           SimpleNamespace(free_all_blocks=lambda: cleanups.append(True)))
    p = SimpleNamespace(deltaT=1/4096)
    ns = dict(opts=SimpleNamespace(q_time_pregrid_factor=2, gpu=True), np=np,
              xpy_default=object(), cupy=cupy, P=p, factored_likelihood=fl,
              QTimePregridError=QTimePregridError,
              rholmArrayDict={"H1": np.ones((2,9), complex), "L1": np.ones((2,9), complex)})
    with pytest.raises(QTimePregridError, match="Requested Q refinement failed"):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), str(DRIVER), "exec"), ns)
    assert calls == [17, 17, 9, 9] and cleanups == [True]
    assert ns["opts"].q_time_pregrid_factor == 2 and p.deltaT == 1/4096
    assert "q_deltaT" not in ns


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_actual_outer_handler_cannot_hide_refinement_failure(tmp_path, factor):
    import subprocess
    import sys
    tree = ast.parse(DRIVER.read_text())
    handler = next(node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
                   and node.name == "exception_failure")
    failing = ast.parse('raise QTimePregridError("requested CUDA Q refinement failed")').body
    guarded = ast.Try(body=failing, handlers=[handler], orelse=[], finalbody=[])
    loop = ast.For(target=ast.Name(id="indx", ctx=ast.Store()),
                   iter=ast.List(elts=[ast.Constant(0), ast.Constant(1)], ctx=ast.Load()),
                   body=[guarded], orelse=[])
    actual = ast.unparse(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])))
    script = ('import sys\nfrom types import SimpleNamespace\n'
              'from RIFT.likelihood.q_time_pregrid import QTimePregridError\n'
              f'opts=SimpleNamespace(q_time_pregrid_factor={factor}, internal_make_empty_file_on_error=True, '
              'internal_soft_fail_on_cuda_error=True, internal_hard_fail_on_error=False, '
              f'output_file={str(tmp_path / "point")!r})\n' + actual)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "QTimePregridError" in proc.stderr
    assert not list(tmp_path.iterdir()), "fatal Q failure must not create an empty success artifact"
