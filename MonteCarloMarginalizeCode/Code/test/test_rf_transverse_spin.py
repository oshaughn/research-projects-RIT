"""geometric4 as implied CIP coordinates: helper policy and conversion contracts; no likelihood calls."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rf_transverse_spin', ROOT/'RIFT/misc/rf_transverse_spin.py')
rf=importlib.util.module_from_spec(spec);spec.loader.exec_module(rf)
FULL='3 --fit-method rf --use-precessing --parameter-nofit mc --parameter delta_mc --parameter-implied mu1 --parameter-implied mu2 --parameter-implied chiMinus --parameter-implied s1x --parameter-implied s1y --parameter-implied s2x --parameter-implied s2y --parameter-nofit chi1 --parameter-nofit chi2 --parameter-nofit cos_theta1 --parameter-nofit cos_theta2 --parameter-nofit phi1 --parameter-nofit phi2'
PHYSICAL=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']

def chart_line(line, mode='geometric4'):
    for old,new in zip(rf.TRANSVERSE, rf.geometric_names(mode)):
        line=line.replace('--parameter-implied '+old+' ','--parameter-implied '+new+' ')
    return line

def _identity(x,coord_names,low_level_coord_names,**kw):
    return x[:,[low_level_coord_names.index(n) for n in coord_names]]

def _lalsimutils():
    pytest.importorskip('lal')
    import sys
    sys.path.insert(0,str(ROOT))
    from RIFT import lalsimutils
    return lalsimutils

@pytest.mark.parametrize('mc',[None,float('nan'),float('inf'),-1,0,20,21,'not a mass'])
def test_auto_unknown_and_boundary_off(mc):
    assert rf.stage_arguments(FULL,'auto',mc,True,20)==FULL

@pytest.mark.parametrize('mc',[1,19.999,'10'])
def test_auto_lowmass_fullstage(mc):
    line=rf.stage_arguments(FULL,'auto',mc,True,40)
    assert line==chart_line(FULL)+' --fref 40.0'
    assert '--parameter-nofit phi2' in line and 's1x' not in line

@pytest.mark.parametrize('mode',rf.GEOMETRIC4_MODES)
def test_explicit_mode_swaps_the_four_transverse_names(mode):
    line=rf.stage_arguments(FULL,mode,25,True,20)
    assert line==chart_line(FULL,mode)+' --fref 20.0'
    assert '--rf-transverse-spin-coordinates' not in line

@pytest.mark.parametrize('line',[FULL.replace('fit-method rf','fit-method gp'),FULL.replace('fit-method rf','fit-method quadratic'),FULL.replace('--parameter-implied s2y','--parameter-nofit s2y'),FULL+' --parameter-implied s1x'])
def test_reduced_or_other_interpolator_unchanged(line):
    assert rf.stage_arguments(line,'geometric4',10,True,20)==line

def test_override_and_applicability():
    assert rf.stage_arguments(FULL,'off',10,True,20)==FULL
    assert rf.stage_arguments(FULL,'auto',10,False,20)==FULL
    with pytest.raises(ValueError):rf.stage_arguments(FULL,'geometric4',10,False,20)

def test_frequency_replaces_existing_without_mixing_sampling():
    out=rf.stage_arguments(FULL+' --fref 11','geometric4',10,True,30)
    assert out.count('--fref')==1 and '--fref 30.0' in out
    with pytest.raises(ValueError):rf.stage_arguments(FULL,'geometric4',10,True,float('nan'))

def test_bool_mass_does_not_activate_auto():
    assert rf.stage_arguments(FULL,'auto',True,True,20)==FULL

def test_any_mass_and_aligned_basis_is_accepted():
    # The chart needs only physical masses and spins, so the rest of the basis is the helper's choice.
    line=FULL.replace('--parameter-implied mu1','--parameter-implied xi')
    assert rf.stage_arguments(line,'geometric4',10,True,20)==chart_line(line)+' --fref 20.0'

SAMPLED='3 --fit-method rf --parameter-nofit mc --parameter delta_mc --parameter-implied mu1 --parameter-implied mu2 --parameter-implied chiMinus --parameter-nofit s1z --parameter-nofit s2z --use-precessing --parameter s1x --parameter s1y --parameter s2x  --parameter s2y'

@pytest.mark.parametrize('mode',rf.GEOMETRIC4_MODES)
def test_sampled_transverse_stay_sampled_and_chart_is_fit(mode):
    # Volumetric-spin stages sample and fit s1x..s2y with --parameter.
    import shlex
    tokens=shlex.split(rf.stage_arguments(SAMPLED,mode,10,True,20))
    def values(flag): return [tokens[i+1] for i,t in enumerate(tokens[:-1]) if t==flag]
    assert not set(rf.TRANSVERSE)&set(values('--parameter'))
    assert set(rf.TRANSVERSE)<=set(values('--parameter-nofit'))
    assert values('--parameter-implied')==['mu1','mu2','chiMinus']+list(rf.geometric_names(mode))
    # CIP samples --parameter plus --parameter-nofit: the same set as before.
    before=shlex.split(SAMPLED)
    def sampled(t): return sorted(t[i+1] for i,x in enumerate(t[:-1]) if x in ('--parameter','--parameter-nofit'))
    assert sampled(tokens)==sampled(before)
    assert values('--fref')==['20.0']

@pytest.mark.parametrize('line',[SAMPLED+' --parameter-implied s1x', SAMPLED.replace('--parameter s2y','--parameter-nofit s2y')])
def test_sampled_transverse_needs_each_component_once(line):
    assert rf.stage_arguments(line,'geometric4',10,True,20)==line

def test_mixed_sampled_and_implied_components():
    line=FULL.replace('--parameter-implied s2y','--parameter s2y')
    out=rf.stage_arguments(line,'geometric4',10,True,20)
    assert '--parameter-nofit s2y --parameter-implied rf_residual_perpendicular' in out
    assert '--parameter-implied rf_total_perp' in out and ' s1x ' not in out.replace('--parameter-nofit s1x','')

def test_vector_conversion_native_columns_preserved():
    physical=np.array([[20,10,.3,.2,-.1,.1,-.2,.4],[20,20,0,0,.2,0,0,.3]])
    out=rf.convert(physical,PHYSICAL+list(rf.GEOMETRIC4_NAMES),PHYSICAL,35,_identity)
    np.testing.assert_array_equal(out[:,:8],physical)
    np.testing.assert_array_equal(out[:,8:],rf.geometric4(physical[:,0],physical[:,1],physical[:,2:5],physical[:,5:8],35))

@pytest.mark.parametrize('frequency',[None,float('nan'),float('inf'),0,-1])
def test_conversion_requires_a_reference_frequency(frequency):
    with pytest.raises(ValueError,match='fref'):
        rf.convert(np.zeros((1,8)),list(rf.GEOMETRIC4_NAMES),PHYSICAL,frequency,_identity)

def test_invalid_physical_rows_do_not_abort_valid_batch():
    physical=np.array([[20,10,.3,.2,-.1,.1,-.2,.4],[-np.inf]*8])
    out=rf.convert(physical,PHYSICAL+list(rf.GEOMETRIC4_NAMES),PHYSICAL,20,_identity,enforce_kerr=True)
    assert np.isfinite(out[0]).all() and np.isneginf(out[1]).all()

@pytest.mark.parametrize('mode',rf.GEOMETRIC4_MODES)
def test_lalsimutils_spherical_conversion_matches_extract_param(mode):
    lalsimutils=_lalsimutils()
    import lal
    names=['mc','delta_mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    low=np.array([[10,.2,.5,.7,-.2,.4,1,2],[15,.1,0,0,1,-1,0,0]])
    chart=list(rf.geometric_names(mode))
    for redshift in [0,.4]:
        physical=lalsimutils.convert_waveform_coordinates(low,coord_names=PHYSICAL,low_level_coord_names=names,source_redshift=redshift)
        out=lalsimutils.convert_waveform_coordinates(low,coord_names=PHYSICAL+chart,low_level_coord_names=names,source_redshift=redshift,fref=35)
        np.testing.assert_allclose(out[:,:8],physical,rtol=1e-14,atol=1e-14)
        expected=rf.geometric4(physical[:,0],physical[:,1],physical[:,2:5],physical[:,5:8],35,phase_excess=(mode!='geometric4'))
        np.testing.assert_allclose(out[:,8:],expected,rtol=1e-14,atol=1e-14)
        p=lalsimutils.ChooseWaveformParams();p.fref=35;p.m1=physical[0,0]*lal.MSUN_SI;p.m2=physical[0,1]*lal.MSUN_SI
        p.s1x,p.s1y,p.s1z=physical[0,2:5];p.s2x,p.s2y,p.s2z=physical[0,5:8]
        np.testing.assert_allclose([p.extract_param(n) for n in chart],out[0,8:],rtol=1e-14)
    with pytest.raises(ValueError,match='fref'):
        lalsimutils.convert_waveform_coordinates(low,coord_names=chart,low_level_coord_names=names)

def test_extract_param_requires_fref():
    lalsimutils=_lalsimutils()
    p=lalsimutils.ChooseWaveformParams()
    assert p.fref==0
    with pytest.raises(ValueError,match='reference frequency'):
        p.extract_param('rf_total_perp')

def test_lalsimutils_kerr_rejection_mixed_batch():
    lalsimutils=_lalsimutils()
    names=['mc','delta_mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    low=np.array([[10,.2,.5,.7,-.2,.4,1,2],[10,.2,1.1,.7,-.2,.4,1,2]])
    fit=['delta_mc','mu1','mu2','chiMinus']+list(rf.GEOMETRIC4_NAMES)
    out=lalsimutils.convert_waveform_coordinates(low,coord_names=fit,low_level_coord_names=names,enforce_kerr=True,fref=35)
    assert np.isfinite(out[0]).all() and np.isneginf(out[1]).all()

def test_chart_names_are_registered_parameters():
    lalsimutils=_lalsimutils()
    assert set(rf.GEOMETRIC4_NAMES+rf.PHASE_EXCESS_NAMES)==set(lalsimutils.rf_transverse_param_names)
    # Grid readers assign every valid_params column; the chart cannot be assigned.
    assert not set(lalsimutils.rf_transverse_param_names)&set(lalsimutils.valid_params)
    assert set(lalsimutils.rf_transverse_param_names)<=set(lalsimutils.tex_dictionary)

CIP_FIT=['--parameter','delta_mc']+[a for p in ['mu1','mu2','chiMinus','s1x','s1y','s2x','s2y'] for a in ('--parameter-implied',p)]

CIP_SAMPLED=[a for p in ['mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2'] for a in ('--parameter-nofit',p)]

CIP_BASE=['--fit-method','rf','--use-precessing','--fref','35','--rf-transverse-spin-coordinates','geometric4']+CIP_FIT+CIP_SAMPLED

def test_physics3_is_refused():
    with pytest.raises(ValueError,match='retired'):rf.enabled('physics3',10,True)
    with pytest.raises(ValueError,match='retired'):rf.stage_arguments(FULL,'physics3',10,True,20)


def _run_cip(tmp_path,args):
    pytest.importorskip('lal')
    import os,subprocess,sys
    (tmp_path/'g.dat').write_text('0 20 10 0 0 0 0 0 0 10 0.1 100 1000\n')
    cmd=[sys.executable,str(ROOT/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py'),'--fname','g.dat','--no-plots']+args
    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='1')
    return subprocess.run(cmd,cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=300)


def test_cip_refuses_physics3(tmp_path):
    proc=_run_cip(tmp_path,[a if a!='geometric4' else 'physics3' for a in CIP_BASE])
    assert proc.returncode!=0 and 'physics3 is retired' in proc.stdout, proc.stdout[-2000:]


def test_cip_alias_requires_the_four_transverse_names(tmp_path):
    args=[a for a in CIP_BASE]
    args[args.index('s2y')-1:args.index('s2y')+1]=[]
    proc=_run_cip(tmp_path,args)
    assert proc.returncode!=0 and 'needs s1x, s1y, s2x, s2y' in proc.stdout, proc.stdout[-2000:]


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


def test_retired_problem_finds_any_occurrence_in_any_spelling():
    # pseudo_pipe applies this to every final CIP line, including --manual-extra-cip-args.
    base=' '.join(CIP_BASE)
    assert rf.retired_problem(base) is None
    for extra in [' --rf-transverse-spin-coordinates=physics3',' --rf-transverse-spin-coordinates physics3']:
        assert 'retired' in rf.retired_problem(base+extra)
        assert 'retired' in rf.retired_problem(extra+' '+base)
