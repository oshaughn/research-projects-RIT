"""Geometric4 dimension and physical conversion contracts; no inference jobs."""
import importlib.util
from pathlib import Path
import shlex
import numpy as np
import pytest
CODE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rf',CODE/'RIFT/misc/rf_transverse_spin.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)

@pytest.mark.parametrize('masses',[(20.,10.),(20.,20.),(1000.,1000.)])
@pytest.mark.parametrize('phase_excess',[False,True])
def test_roundtrip(masses,phase_excess):
    rng=np.random.default_rng(103)
    s1=rng.uniform(-.5,.5,(500,3));s2=rng.uniform(-.5,.5,(500,3))
    s1[0]=0;s2[0]=0
    s2[1,:2]=-s1[1,:2]*(masses[0]/masses[1])**2
    features=f.geometric4(*masses,s1,s2,35.,phase_excess=phase_excess)
    assert features.shape==(500,4) and np.isfinite(features).all()
    a,b=f.geometric4_inverse(*masses,s1[:,2],s2[:,2],features,35.,phase_excess=phase_excess)
    np.testing.assert_allclose(a,s1,atol=5e-15)
    np.testing.assert_allclose(b,s2,atol=5e-15)
    assert features[0,1]==0 and features[1,1]==0


def test_negative_parallel_J_is_phase_excess_not_Q():
    s=np.array([.2,.1,-1.])
    g=f.geometry(1000.,1000.,s,s,100.)
    assert g['D']<0
    H=f.geometric4(1000.,1000.,s,s,100.,phase_excess=True)[0]
    np.testing.assert_allclose(H,(g['J']-abs(g['D']))/g['L'])
    assert not np.isclose(H,g['JminusD']/g['L'])


@pytest.mark.parametrize('mode',f.GEOMETRIC4_MODES)
def test_native_training_and_prediction_redshift(mode):
    import lal
    from RIFT import lalsimutils
    p=lalsimutils.ChooseWaveformParams();p.m1=22*lal.MSUN_SI;p.m2=11*lal.MSUN_SI
    p.s1x=.3;p.s1y=-.2;p.s1z=.1;p.s2x=-.1;p.s2y=.3;p.s2z=-.2;p.fref=35
    names=list(f.NATIVE_FEATURES[:4])+list(f.geometric_names(mode))
    physical=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    for z in [0.,.3]:
        row=np.array([[p.extract_param(n) for n in physical]])
        row[0,:2]/=lal.MSUN_SI*(1+z)
        out=f.convert(row,names,physical,35,lalsimutils.convert_waveform_coordinates,source_redshift=z)
        expected=np.array([f.extract(p,n) for n in names])
        np.testing.assert_allclose(out[0],expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('mode',f.GEOMETRIC4_MODES)
def test_stage_preserves_raw_ranges_and_waveform_dictionary(mode):
    line='1 --fit-method rf --use-precessing '+ ' '.join('--parameter-implied '+n for n in f.NATIVE_FEATURES)
    line+=' --downselect-parameter chi1 --downselect-parameter-range [0,0.9] --fref=10'
    line+=' --approx IMRPhenomXPHM --lalsim-extra-waveform-args "{\'PhenomXPrecVersion\': 223}"'
    active=f.stage_arguments(line,mode,30,True,35)
    argv=shlex.split(active)
    assert argv[argv.index('--rf-transverse-spin-coordinates')+1]==mode
    assert argv[argv.index('--fref')+1]=='35.0'
    assert '[0,0.9]' in argv and active.count('[0,0.9]')==1
    assert argv[argv.index('--lalsim-extra-waveform-args')+1]=="{'PhenomXPrecVersion': 223}"
    assert f.revalidate_stage(active)==active
    with pytest.raises(ValueError):f.revalidate_stage(active.replace('mu1','eta'))

@pytest.mark.parametrize('extra',['phi1','s1x'])
@pytest.mark.parametrize('mode',f.GEOMETRIC4_MODES)
def test_build_rejects_redundant_basis(extra,mode):
    line='1 --fit-method rf --use-precessing '+' '.join('--parameter-implied '+n for n in f.NATIVE_FEATURES)
    with pytest.raises(ValueError,match='exactly the eight'):
        f.stage_arguments(line+' --parameter-implied '+extra,mode,10,True,35)
    active=f.stage_arguments(line,mode,10,True,35)
    with pytest.raises(ValueError,match='exactly the eight'):
        f.revalidate_stage(active+' --parameter-implied '+extra)


def test_actual_pipeline_forwarding_and_postrewrite_guard():
    import ast,types
    source=(CODE/'bin/util_RIFT_pseudo_pipe.py').read_text()
    blocks=[n for n in ast.parse(source).body if isinstance(n,ast.If) and ast.unparse(n.test)=='opts.rf_transverse_spin_coordinates']
    forward=next(n for n in blocks if 'cmd +=' in ast.unparse(n))
    ns={'opts':types.SimpleNamespace(rf_transverse_spin_coordinates='geometric4'),'cmd':'helper_LDG_Events.py --approx IMRPhenomXPHM'}
    exec(compile(ast.Module(body=[forward],type_ignores=[]),'actual_pipeline_forward','exec'),ns)
    argv=shlex.split(ns['cmd']);assert argv[argv.index('--rf-transverse-spin-coordinates')+1]=='geometric4'
    guard=next(n for n in blocks if 'revalidate_stage' in ast.unparse(n))
    line='1 --fit-method rf --use-precessing '+' '.join('--parameter-implied '+n for n in f.NATIVE_FEATURES)
    ns['lines']=[f.stage_arguments(line,'geometric4',10,True,35)]
    exec(compile(ast.Module(body=[guard],type_ignores=[]),'actual_pipeline_guard','exec'),ns)
    ns['lines'][0]+=' --parameter-implied phi1'
    with pytest.raises(ValueError,match='exactly the eight'):
        exec(compile(ast.Module(body=[guard],type_ignores=[]),'actual_pipeline_guard','exec'),ns)

@pytest.mark.parametrize('spins',[np.zeros(2),np.array([np.nan,0,0])])
def test_invalid_public_spin_input(spins):
    with pytest.raises(ValueError,match='finite three-component'):
        f.geometric4(20,10,spins,np.zeros(3))


def test_radius_and_phase_variant_are_distinct():
    s1=np.array([.2,.3,.1]);s2=np.array([-.1,.4,-.2])
    g=f.geometry(20.,10.,s1,s2,35.)
    raw=f.geometric4(20.,10.,s1,s2,35.)
    phase=f.geometric4(20.,10.,s1,s2,35.,phase_excess=True)
    np.testing.assert_allclose(raw[0],np.linalg.norm(g['T']))
    assert not np.isclose(raw[0],phase[0])
    np.testing.assert_array_equal(raw[1:],phase[1:])
    with pytest.raises(ValueError,match='Do not mix'):
        f.convert(np.zeros((1,8)),list(f.GEOMETRIC4_NAMES)+['rf_phase_excess'],[],35,None)
