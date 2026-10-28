"""Geometric4 dimension and physical conversion contracts; no inference jobs."""
import importlib.util
from pathlib import Path
import shlex
import numpy as np
import pytest
CODE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rf',CODE/'RIFT/misc/rf_transverse_spin.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
NATIVE=('delta_mc','mu1','mu2','chiMinus')+f.TRANSVERSE
LINE='1 --fit-method rf --use-precessing '+' '.join('--parameter-implied '+n for n in NATIVE) + ' ' + ' '.join('--parameter-nofit '+n for n in ('mc','delta_mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2'))

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
    names=['delta_mc','mu1','mu2','chiMinus']+list(f.geometric_names(mode))
    physical=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    for z in [0.,.3]:
        row=np.array([[p.extract_param(n) for n in physical]])
        row[0,:2]/=lal.MSUN_SI*(1+z)
        out=lalsimutils.convert_waveform_coordinates(row,names,physical,source_redshift=z,fref=35)
        expected=np.array([p.extract_param(n) for n in names])
        np.testing.assert_allclose(out[0],expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('mode',f.GEOMETRIC4_MODES)
def test_stage_preserves_raw_ranges_and_waveform_dictionary(mode):
    line=LINE
    line+=' --downselect-parameter chi1 --downselect-parameter-range [0,0.9] --fref=10'
    line+=' --approx IMRPhenomXPHM --lalsim-extra-waveform-args "{\'PhenomXPrecVersion\': 223}"'
    active=f.stage_arguments(line,mode,30,True,35)
    argv=shlex.split(active)
    assert '--rf-transverse-spin-coordinates' not in argv
    implied=[argv[i+1] for i,t in enumerate(argv[:-1]) if t=='--parameter-implied']
    assert implied==['delta_mc','mu1','mu2','chiMinus']+list(f.geometric_names(mode))
    assert argv[argv.index('--fref')+1]=='35.0' and argv.count('--fref')==1
    assert '[0,0.9]' in argv and active.count('[0,0.9]')==1
    assert argv[argv.index('--lalsim-extra-waveform-args')+1]=="{'PhenomXPrecVersion': 223}"

@pytest.mark.parametrize('mode',('auto',)+f.GEOMETRIC4_MODES)
def test_stage_swaps_only_a_complete_transverse_set(mode):
    # Extra fit coordinates are allowed; a repeated or missing transverse name leaves the stage alone.
    names=list(f.geometric_names('geometric4' if mode=='auto' else mode))
    active=f.stage_arguments(LINE+' --parameter-implied phi1',mode,10,True,35)
    assert all('--parameter-implied '+n in active for n in names) and '--parameter-implied phi1' in active
    assert not any('--parameter-implied '+n+' ' in active+' ' for n in f.TRANSVERSE)
    for line in (LINE+' --parameter-implied s1x', LINE.replace('--parameter-implied s2y','')):
        assert f.stage_arguments(line,mode,10,True,35)==line


def test_actual_pipeline_forwarding_and_retired_guard():
    import ast,types
    source=(CODE/'bin/util_RIFT_pseudo_pipe.py').read_text()
    blocks=[n for n in ast.parse(source).body if isinstance(n,ast.If) and ast.unparse(n.test)=='opts.rf_transverse_spin_coordinates']
    forward=next(n for n in blocks if 'cmd +=' in ast.unparse(n))
    ns={'opts':types.SimpleNamespace(rf_transverse_spin_coordinates='geometric4'),'cmd':'helper_LDG_Events.py --approx IMRPhenomXPHM'}
    exec(compile(ast.Module(body=[forward],type_ignores=[]),'actual_pipeline_forward','exec'),ns)
    argv=shlex.split(ns['cmd']);assert argv[argv.index('--rf-transverse-spin-coordinates')+1]=='geometric4'
    assert 'revalidate_stage' not in source and 'stage_problem' not in source
    guard=next(n for n in ast.parse(source).body if isinstance(n,ast.For) and 'retired_problem' in ast.unparse(n))
    ns['retired_problem']=f.retired_problem
    ns['lines']=[f.stage_arguments(LINE,'geometric4',10,True,35)]
    exec(compile(ast.Module(body=[guard],type_ignores=[]),'actual_pipeline_guard','exec'),ns)
    ns['lines'][0]+=' --rf-transverse-spin-coordinates=physics3'
    with pytest.raises(ValueError,match='retired'):
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


@pytest.mark.parametrize('style',['flag','implied'])
@pytest.mark.parametrize('mode',f.GEOMETRIC4_MODES)
def test_quadpuff_contingency_puffs_sampling_coordinates(tmp_path,mode,style):
    # quadpuff cannot assign RF features; it must jitter the sampling coordinates and still write a physical grid.
    pytest.importorskip('lal')
    import os,subprocess,sys
    from igwn_ligolw import utils,lsctables
    rng=np.random.default_rng(5);n=160
    dat=np.c_[np.arange(n),rng.uniform(20,25,n),rng.uniform(10,15,n),rng.uniform(-.4,.4,(n,6)),
              3000-rng.uniform(0,3000,n),np.full(n,.1),np.full(n,100),np.full(n,1000)]  # sharply peaked: low n_eff
    np.savetxt(tmp_path/'grid.dat',dat)
    args=['--fname','grid.dat','--fit-method','rf','--use-precessing','--no-plots','--fref','35',
          '--parameter','delta_mc',
          '--contingency-unevolved-neff','quadpuff','--sampler-method','AV','--n-eff','1000','--n-max','20000',
          '--n-output-samples','100','--fname-output-samples','puff']
    if style=='flag': args+=['--rf-transverse-spin-coordinates',mode]
    for p in ['mu1','mu2','chiMinus']+(list(f.TRANSVERSE) if style=='flag' else list(f.geometric_names(mode))):args+=['--parameter-implied',p]
    for p in ['mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']:args+=['--parameter-nofit',p]
    env=dict(os.environ,PYTHONPATH=str(CODE)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1',GW_SURROGATE='')
    proc=subprocess.run([sys.executable,str(CODE/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py')]+args,
        cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=600)
    assert proc.returncode==0, proc.stdout[-3000:]
    assert 'Contingency: quadpuff' in proc.stdout
    rows=lsctables.SimInspiralTable.get_table(utils.load_filename(str(tmp_path/'puff.xml.gz')))
    assert len(rows)>0
    m=np.array([[r.mass1,r.mass2] for r in rows]);s1=np.array([[r.spin1x,r.spin1y,r.spin1z] for r in rows])
    assert (m>0).all() and (np.linalg.norm(s1,axis=1)<=1).all()
    assert not np.allclose(m[:,0],dat[:len(m),1])  # actually jittered
