"""Deployment contracts for redundant L-frame RF features; no likelihood calls."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rf_transverse_spin', ROOT/'RIFT/misc/rf_transverse_spin.py')
rf=importlib.util.module_from_spec(spec);spec.loader.exec_module(rf)
FULL='3 --fit-method rf --use-precessing --parameter mc --parameter delta_mc --parameter-implied mu1 --parameter-implied mu2 --parameter-implied chiMinus --parameter-implied s1x --parameter-implied s1y --parameter-implied s2x --parameter-implied s2y --parameter-nofit chi1 --parameter-nofit chi2 --parameter-nofit cos_theta1 --parameter-nofit cos_theta2 --parameter-nofit phi1 --parameter-nofit phi2'

@pytest.mark.parametrize('mc',[None,float('nan'),float('inf'),-1,0,20,21,'not a mass'])
def test_auto_unknown_and_boundary_off(mc):
    assert rf.stage_arguments(FULL,'auto',mc,True,20)==FULL

@pytest.mark.parametrize('mc',[1,19.999,'10'])
def test_auto_lowmass_fullstage(mc):
    line=rf.stage_arguments(FULL,'auto',mc,True,40)
    assert line==FULL+' --rf-transverse-spin-coordinates physics3 --fref 40.0'
    assert '--parameter-nofit phi2' in line

@pytest.mark.parametrize('line',[FULL.replace('fit-method rf','fit-method gp'),FULL.replace('fit-method rf','fit-method quadratic'),FULL.replace('--parameter-implied s2y','--parameter-nofit s2y')])
def test_reduced_or_other_interpolator_unchanged(line):
    assert rf.stage_arguments(line,'physics3',10,True,20)==line

def test_override_and_applicability():
    assert rf.stage_arguments(FULL,'off',10,True,20)==FULL
    assert rf.stage_arguments(FULL,'auto',10,False,20)==FULL
    with pytest.raises(ValueError):rf.stage_arguments(FULL,'physics3',10,False,20)
    assert 'physics3' in rf.stage_arguments(FULL,'physics3',25,True,20)

def test_frequency_replaces_existing_without_mixing_sampling():
    out=rf.stage_arguments(FULL+' --fref 11','physics3',10,True,30)
    assert out.count('--fref')==1 and '--fref 30.0' in out
    with pytest.raises(ValueError):rf.stage_arguments(FULL,'physics3',10,True,float('nan'))

def test_exact_frozen_scalar_formulas_and_zero_axis():
    rng=np.random.default_rng(131)
    m1=rng.uniform(10,50,500);m2=m1*rng.uniform(.1,1,500)
    s1=rng.uniform(-.5,.5,(500,3));s2=rng.uniform(-.5,.5,(500,3))
    x=rf.scalar_features(m1,m2,s1,s2,20)
    q=m2/m1;eta=q/(1+q)**2;v=(np.pi*(m1+m2)*rf.MTSUN*20)**(1/3)
    w1=1/(1+q)**2;w2=q*q*w1;L=eta/v;D=L+w1*s1[:,2]+w2*s2[:,2]
    T=w1[:,None]*s1[:,:2]+w2[:,None]*s2[:,:2];G=(2+1.5*q)[:,None]*w1[:,None]*s1[:,:2]+(2+1.5/q)[:,None]*w2[:,None]*s2[:,:2]
    T2=(T*T).sum(axis=1);J=np.sqrt(D*D+T2)
    expected=np.stack([T2/(D*D+(.1*L)**2),T2/(J+D)/L,(G*G).sum(axis=1)/(eta*v*v)**2],axis=1)
    np.testing.assert_allclose(x,expected,rtol=1e-14,atol=1e-15)
    np.testing.assert_array_equal(rf.scalar_features(m1,m2,np.zeros_like(s1),np.zeros_like(s2)),0)
    assert np.isfinite(x).all()
    assert not np.array_equal(x,rf.scalar_features(m1,m2,s1,s2,40))

def test_vector_conversion_native_columns_preserved():
    physical=np.array([[20,10,.3,.2,-.1,.1,-.2,.4],[20,20,0,0,.2,0,0,.3]])
    names=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    def converter(x,coord_names,low_level_coord_names,**kw):
        return x[:,[low_level_coord_names.index(n) for n in coord_names]]
    out=rf.convert(physical,names+list(rf.FEATURE_NAMES),names,35,converter)
    np.testing.assert_array_equal(out[:,:8],physical)
    np.testing.assert_array_equal(out[:,8:],rf.scalar_features(physical[:,0],physical[:,1],physical[:,2:5],physical[:,5:8],35))

def test_component_mirror_is_symmetric():
    s1=np.array([.2,.3,-.5]);s2=np.array([-.3,.1,.4])
    np.testing.assert_allclose(rf.scalar_features(30,10,s1,s2),rf.scalar_features(10,30,s2,s1),rtol=2e-15)

def test_invalid_physical_rows_do_not_abort_valid_batch():
    physical=np.array([[20,10,.3,.2,-.1,.1,-.2,.4],[-np.inf]*8])
    names=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    def converter(x,coord_names,low_level_coord_names,**kw):
        return x[:,[low_level_coord_names.index(n) for n in coord_names]]
    out=rf.convert(physical,names+list(rf.FEATURE_NAMES),names,20,converter,enforce_kerr=True)
    assert np.isfinite(out[0]).all() and np.isneginf(out[1]).all()

@pytest.mark.parametrize('frequency',[float('nan'),float('inf'),0,-1])
def test_scalar_frequency_rejects_nonphysical_values(frequency):
    with pytest.raises(ValueError):rf.scalar_features(20,10,np.zeros(3),np.zeros(3),frequency)

def test_actual_native_spherical_conversion_and_source_redshift():
    # LAL is optional for lightweight CI; deployment validation runs this on real LAL.
    pytest.importorskip('lal')
    import sys
    sys.path.insert(0,str(ROOT))
    from RIFT import lalsimutils
    names=['mc','delta_mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    low=np.array([[10,.2,.5,.7,-.2,.4,1,2],[15,.1,0,0,1,-1,0,0]])
    physical_names=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    for redshift in [0,.4]:
        physical=lalsimutils.convert_waveform_coordinates(low,coord_names=physical_names,low_level_coord_names=names,source_redshift=redshift)
        out=rf.convert(low,physical_names+list(rf.FEATURE_NAMES),names,35,lalsimutils.convert_waveform_coordinates,source_redshift=redshift)
        np.testing.assert_allclose(out[:,:8],physical,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(out[:,8:],rf.scalar_features(physical[:,0],physical[:,1],physical[:,2:5],physical[:,5:8],35),rtol=1e-14,atol=1e-14)
        import lal
        p=lalsimutils.ChooseWaveformParams();p.fref=35;p.m1=physical[0,0]*lal.MSUN_SI;p.m2=physical[0,1]*lal.MSUN_SI
        p.s1x,p.s1y,p.s1z=physical[0,2:5];p.s2x,p.s2y,p.s2z=physical[0,5:8]
        np.testing.assert_allclose([rf.extract(p,n) for n in rf.FEATURE_NAMES],out[0,8:],rtol=1e-14)

def test_bool_mass_does_not_activate_auto():
    assert rf.stage_arguments(FULL,'auto',True,True,20)==FULL

def test_actual_native_kerr_rejection_mixed_batch():
    pytest.importorskip('lal')
    import sys
    sys.path.insert(0,str(ROOT))
    from RIFT import lalsimutils
    names=['mc','delta_mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    low=np.array([[10,.2,.5,.7,-.2,.4,1,2],[10,.2,1.1,.7,-.2,.4,1,2]])
    fit=['delta_mc','mu1','mu2','chiMinus','s1x','s1y','s2x','s2y']+list(rf.FEATURE_NAMES)
    out=rf.convert(low,fit,names,35,lalsimutils.convert_waveform_coordinates,enforce_kerr=True)
    assert np.isfinite(out[0]).all() and np.isneginf(out[1]).all()

def test_requires_tested_native_phase_basis():
    line=FULL.replace('--parameter-implied mu1','--parameter-implied xi')
    assert rf.stage_arguments(line,'physics3',10,True,20)==line

CIP_FIT=['--parameter','delta_mc']+[a for p in ['mu1','mu2','chiMinus','s1x','s1y','s2x','s2y'] for a in ('--parameter-implied',p)]

@pytest.mark.parametrize('extra',[
    ['--fit-load-gp','my_fit.pkl'],
    ['--fit-method','gp'],
    ['--fref','nan'],
    'no-mu1',
])
def test_cip_refuses_physics3_without_a_fresh_native_rf_fit(tmp_path,extra):
    # Executes CIP up to its option check; a worker given --fit-load-gp must fail, not fit.
    pytest.importorskip('lal')
    import os,subprocess,sys
    args=list(CIP_FIT)
    if extra=='no-mu1':
        args[args.index('mu1')]='xi'; extra=[]
    (tmp_path/'g.dat').write_text('0 20 10 0 0 0 0 0 0 10 0.1 100 1000\n')
    cmd=[sys.executable,str(ROOT/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py'),'--fname','g.dat',
         '--fit-method','rf','--use-precessing','--no-plots','--fref','35','--rf-transverse-spin-coordinates','physics3']+args+extra
    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1')
    proc=subprocess.run(cmd,cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=300)
    assert proc.returncode!=0
    assert ('physics3 requires a fresh RF fit' in proc.stdout) or ('RF reference frequency' in proc.stdout), proc.stdout[-2000:]

def test_default_and_off_never_enter_the_policy():
    # Helper behavior under each mode is executed in test_rf_transverse_helper_generation.py.
    assert not rf.enabled(None,10,True) and not rf.enabled('off',10,True)


def test_advertised_packages_survive_wheel_discovery():
    # Source-tree namespace imports can hide a package omitted from the wheel.
    import setuptools
    names = setuptools.find_packages(str(ROOT))
    advertised = ast.literal_eval(ast.parse((ROOT/'RIFT/__init__.py').read_text()).body[0].value)
    assert 'plot_utilities' in advertised
    assert all('RIFT.'+name in names for name in advertised)
    assert 'RIFT.asimov' in names and 'RIFT.misc' in names


def test_timing_validator_finds_installed_driver(tmp_path, monkeypatch):
    """A wheel has no sibling source-tree bin; inspect its installed entry script."""
    import os, sysconfig
    source=ROOT/'RIFT/likelihood/q_time_pregrid.py'
    legacy=ROOT/'bin/integrate_likelihood_extrinsic_batchmode'
    driver=tmp_path/legacy.name
    driver.write_text("parser.add_option('--vectorized')\nparser.add_option('--q-time-pregrid-factor')\n")
    original=os.path.isfile
    monkeypatch.setattr(os.path,'isfile',lambda p: False if Path(p)==legacy else original(p))
    monkeypatch.setattr(sysconfig,'get_path',lambda name: str(tmp_path) if name=='scripts' else None)
    spec=importlib.util.spec_from_file_location('installed_q_time_pregrid',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module._DRIVER_PATH==str(driver)
    assert module._driver_long_option_names()=={'--vectorized','--q-time-pregrid-factor'}


@pytest.mark.parametrize('values,streams,extra',[
    ({}, True, {}),
    ({'RIFT_NOSTREAM_LOG':'1'}, True, {}),
    ({'RIFT_NOSTREAM_LOG_CIP':'1'}, False, {}),
    ({'RIFT_CIP_FLOCK_LOCAL':'true','RIFT_CIP_POOLS':'IGWN,CIT'}, True, {'MY.flock_local':'true','MY.POOLS':'"IGWN,CIT"'}),
    ({'RIFT_CIP_FLOCK_LOCAL':'false'}, True, {}),
])
@pytest.mark.parametrize('module_name',['dag_utils','dag_utils_generic'])
def test_cip_submit_transport_policy(tmp_path,monkeypatch,values,streams,extra,module_name):
    # CIP streams unless RIFT_NOSTREAM_LOG_CIP is set; RIFT_NOSTREAM_LOG alone does not stop it.
    import importlib,os
    monkeypatch.syspath_prepend(str(ROOT))
    module=pytest.importorskip('RIFT.misc.'+module_name)
    for key in ['RIFT_NOSTREAM_LOG','RIFT_NOSTREAM_LOG_CIP','RIFT_CIP_FLOCK_LOCAL','RIFT_CIP_POOLS']:
        monkeypatch.delenv(key,raising=False)
    for key,value in values.items():
        monkeypatch.setenv(key,value)
    job,_=module.write_CIP_sub(exe='/bin/true',log_dir=None,arg_str='--fit-method rf',out_dir=str(tmp_path))
    cmds=job.get_condor_cmds() if hasattr(job,'get_condor_cmds') else job.condor_cmds
    cmds={k:v for k,v in dict(cmds).items()}
    assert (cmds.get('stream_error')=='True' and cmds.get('stream_output')=='True')==streams
    if not streams:
        assert 'stream_error' not in cmds and 'stream_output' not in cmds
    for key,value in extra.items():
        assert cmds[key]==value
    assert ('MY.flock_local' in cmds)==('MY.flock_local' in extra)


def test_asimov_psd_staging_targets_rundir_from_foreign_cwd(tmp_path,monkeypatch):
    import ast, shutil
    from types import SimpleNamespace,MethodType
    source=(ROOT/'RIFT/asimov/rift.py').read_text()
    tree=ast.parse(source)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_stage_xml_psds')
    imports=[n for n in tree.body if isinstance(n,ast.Import) and any(a.name=='shutil' for a in n.names)]
    assert imports
    namespace={'Path':Path,'shutil':shutil}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'psd-staging','exec'),namespace)
    repository=tmp_path/'repository with spaces';repository.mkdir()
    source=repository/'psd_H1.xml.gz';source.write_bytes(b'exact frozen PSD')
    rundir=tmp_path/'run with spaces';rundir.mkdir()
    other=tmp_path/'foreign cwd';other.mkdir();monkeypatch.chdir(other)
    worker=SimpleNamespace(production=SimpleNamespace(rundir=str(rundir)),_get_psds=lambda kind:[str(source)],_detector_for_psd=lambda path:'H1')
    worker._stage_xml_psds=MethodType(namespace['_stage_xml_psds'],worker)
    worker._stage_xml_psds(dryrun=True)
    assert list(rundir.iterdir())==[]
    worker._stage_xml_psds()
    assert (rundir/'H1-psd.xml.gz').read_bytes()==source.read_bytes()
    assert (rundir/source.name).read_bytes()==source.read_bytes()
    assert list(other.iterdir())==[]
    worker._get_psds=lambda kind:[str(rundir/'H1-psd.xml.gz')]
    worker._stage_xml_psds()  # same-file sources remain safe
    worker.production.rundir='relative/run'
    worker._stage_xml_psds(rundir=str(rundir))  # build resolves before changing cwd
    build=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='build_dag')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_stage_xml_psds' and any(k.arg=='rundir' for k in n.keywords) for n in ast.walk(build))
    for name in ['build_dag','submit_dag']:
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_stage_xml_psds' for n in ast.walk(fn))
