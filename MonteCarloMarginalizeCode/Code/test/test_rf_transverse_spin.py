"""RF coordinate contract: native features/physical priors remain authoritative."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rf_features', CODE/'RIFT/misc/rf_transverse_spin.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)

FULL = '1 --parameter delta_mc --parameter-implied mu1 --parameter-implied mu2 --parameter-implied chiMinus --fit-method rf --use-precessing --parameter s1x --parameter s1y --parameter-implied s2x --parameter-implied s2y --parameter-nofit chi1 --parameter-nofit chi2'

def test_frozen_physics3_parity():
    # Frozen scalar values independently checked against the investigation implementation.
    actual = f.scalar_features(20.,10.,np.array([.2,.3,-.1]),np.array([-.1,.4,.2]),20.)
    expected = np.array([.0347815756,.0170543684,3969.05598])
    np.testing.assert_allclose(actual,expected,rtol=2e-9)

@pytest.mark.parametrize('mass', [20,21,None,np.nan,-1])
def test_auto_conservative_mass(mass):
    assert f.stage_arguments(FULL,'auto',mass,True,20)==FULL

@pytest.mark.parametrize('mode,expected', [('auto','geometric4'),('physics3','physics3'),
                                         ('geometric4','geometric4'),
                                         ('geometric4-phase-excess','geometric4-phase-excess')])
def test_opt_in_keeps_native_sampling_and_reference(mode,expected):
    line=f.stage_arguments(FULL+' --fref 10',mode,19.9,True,25)
    assert '--parameter-nofit chi1 --parameter-nofit chi2' in line
    assert '--fref 10' not in line
    assert '--fref 25.0' in line
    assert '--rf-transverse-spin-coordinates '+expected in line
    assert f.stage_arguments(FULL,None,10,True,20)==FULL
    assert f.stage_arguments(FULL,'off',10,True,20)==FULL

def test_reduced_or_non_rf_stages_are_unchanged():
    for line in [FULL.replace('--parameter-implied s2y',''),FULL.replace('fit-method rf','fit-method gp')]:
        assert f.stage_arguments(line,'physics3',10,True,20)==line
    assert f.stage_arguments(FULL,'auto',10,False,20)==FULL
    with pytest.raises(ValueError): f.stage_arguments(FULL,'physics3',10,False,20)

def test_rotation_and_two_spin_cancellation():
    s1=np.array([.2,.3,-.1]);s2=np.array([-.1,.4,.2]);a=.4
    rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    np.testing.assert_allclose(f.scalar_features(20,10,s1,s2),f.scalar_features(20,10,rot@s1,rot@s2),rtol=2e-15)
    out=f.scalar_features(10,10,np.array([.2,.3,0]),np.array([-.2,-.3,0]))
    assert np.all(np.isfinite(out)) and np.all(out==0)

def test_native_converter_scalar_vector_and_no_feature_loss():
    pytest.importorskip('lal')
    from RIFT import lalsimutils
    import lal
    rng=np.random.default_rng(742)
    low=['delta_mc','mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    x=np.column_stack([rng.uniform(.05,.7,60),rng.uniform(5,19,60),rng.uniform(0,.8,(60,2)),rng.uniform(-1,1,(60,2)),rng.uniform(0,2*np.pi,(60,2))])
    base=['delta_mc','mu1','mu2','chiMinus','s1x','s1y','s2x','s2y']
    native=lalsimutils.convert_waveform_coordinates(x,base,low)
    enhanced=f.convert(x,base+list(f.FEATURE_NAMES),low,30,lalsimutils.convert_waveform_coordinates)
    assert np.array_equal(native,enhanced[:,:8])
    physical=lalsimutils.convert_waveform_coordinates(x,['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z'],low)
    for row,vals in zip(physical,enhanced[:,8:]):
        p=lalsimutils.ChooseWaveformParams();p.m1=row[0]*lal.MSUN_SI;p.m2=row[1]*lal.MSUN_SI;p.fref=30
        p.s1x,p.s1y,p.s1z,p.s2x,p.s2y,p.s2z=row[2:]
        np.testing.assert_allclose(vals,[f.extract(p,n) for n in f.FEATURE_NAMES],rtol=3e-15,atol=1e-12)

def test_mirror_rows_and_invalid_proposals():
    s1=np.array([.2,.3,-.1]);s2=np.array([-.1,.4,.2])
    np.testing.assert_allclose(f.scalar_features(20,10,s1,s2),f.scalar_features(10,20,s2,s1),rtol=2e-15)
    columns=['m1','m2','s1x','s1y','s1z','s2x','s2y','s2z']
    x=np.array([[20,10,*s1,*s2],[-np.inf]*8])
    def converter(x,coord_names,low_level_coord_names,**kwargs):
        return x[:,[low_level_coord_names.index(p) for p in coord_names]]
    out=f.convert(x,columns+list(f.FEATURE_NAMES),columns,20,converter,enforce_kerr=True)
    assert np.isfinite(out[0]).all() and np.isneginf(out[1]).all()

def test_cli_pseudo_forwarding_and_asimov_template():
    import ast
    for name in ['helper_LDG_Events.py','util_RIFT_pseudo_pipe.py','util_ConstructIntrinsicPosterior_GenericCoordinates.py']:
        ast.parse((CODE/'bin'/name).read_text())
    helper=(CODE/'bin/helper_LDG_Events.py').read_text()
    assert helper.count('event_dict["rf_mass_is_placeholder"] = True')==2
    assert "None if event_dict.get('rf_mass_is_placeholder', False)" in helper
    pseudo=(CODE/'bin/util_RIFT_pseudo_pipe.py').read_text()
    assert pseudo.index('cmd = " helper_LDG_Events.py')<pseudo.index('if opts.rf_transverse_spin_coordinates:\n    cmd +=')
    template=(CODE/'RIFT/asimov/rift.ini').read_text()
    assert "rf_mode = sampler['cip']['transverse spin coordinates']" in template
    cip=(CODE/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py').read_text()
    assert 'coord_names = list(coord_names)' in cip
    assert '.extract_param(coord_names[' not in cip

@pytest.mark.parametrize('frequency',[np.nan,np.inf,0,-1])
def test_bad_fref_is_rejected(frequency):
    with pytest.raises(ValueError): f.scalar_features(20,10,np.zeros(3),np.zeros(3),frequency)

def test_actual_native_fast_kerr_guard():
    pytest.importorskip('lal')
    from RIFT import lalsimutils
    low=['delta_mc','mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    x=np.array([[.2,10,.4,.3,.2,-.2,.3,.8],[.2,10,1.2,.3,.2,-.2,.3,.8]])
    cols=['delta_mc','mu1','mu2','chiMinus','s1x','s1y','s2x','s2y']+list(f.FEATURE_NAMES)
    out=f.convert(x,cols,low,20,lalsimutils.convert_waveform_coordinates,enforce_kerr=True)
    assert np.isfinite(out[0]).all() and np.isneginf(out[1]).all()

MISSING=object()
@pytest.mark.parametrize('value,expected',[(MISSING,'auto'),(None,'auto'),(False,'off'),(True,'physics3'),('off','off'),('auto','auto'),('physics3','physics3')])
def test_real_liquid_asimov_override(value,expected):
    liquid=pytest.importorskip('liquid')
    template=(CODE/'RIFT/asimov/rift.ini').read_text()
    start=template.index("{% assign rf_mode = sampler['cip']['transverse spin coordinates'] %}")
    end=template.index('cip-sampler-method=',start)
    cip={} if value is MISSING else {'transverse spin coordinates':value}
    rendered=liquid.Liquid(template[start:end],from_file=False,mode='standard').render(sampler={'cip':cip})
    assert rendered.count('rf-transverse-spin-coordinates=')==1
    assert 'rf-transverse-spin-coordinates="'+expected+'"' in rendered

@pytest.mark.parametrize('missing',['delta_mc','mu1','mu2','chiMinus'])
def test_only_tested_native_basis_is_activated(missing):
    import shlex
    words=shlex.split(FULL)
    i=words.index(missing); del words[i-1:i+1]
    line=' '.join(shlex.quote(p) for p in words)
    assert f.stage_arguments(line,'physics3',10,True,20)==line

def test_basis_policy_is_resolved_after_initial_grid_before_strategy():
    src=(CODE/'bin/helper_LDG_Events.py').read_text()
    gate=src.index('# The single top-level option selects')
    assert src.index('Executing grid command')<gate
    assert gate<src.index('if opts.propose_fit_strategy:\n    puff_max_it= 0')
    assert 'opts.internal_use_aligned_phase_coordinates = True' in src[gate:]
    assert not f.enabled('auto',True,True)
    assert f.enabled('physics3',None,True)


def test_advertised_packages_survive_wheel_discovery():
    # Source-tree namespace imports can hide a package omitted from the wheel.
    import setuptools, ast
    names = setuptools.find_packages(str(CODE))
    advertised = ast.literal_eval(ast.parse((CODE/'RIFT/__init__.py').read_text()).body[0].value)
    assert 'plot_utilities' in advertised
    assert all('RIFT.'+name in names for name in advertised)
    assert 'RIFT.asimov' in names and 'RIFT.misc' in names

def test_calibration_waveform_kwargs_survive_shell_transport():
    import ast, shlex
    source=(CODE/'bin/util_RIFT_pseudo_pipe.py').read_text()
    tree=ast.parse(source)
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.AugAssign) and
           isinstance(n.target,ast.Name) and n.target.id=='cmd' and
           isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and
           isinstance(n.value.func.value,ast.Constant) and
           n.value.func.value.value==' --calibration-reweighting-initial-extra-args={} ']
    assert len(nodes)==1
    value=' --extra-waveform-kwargs "{\'fd_alignment_postevent_time\': None, \'fd_centering_factor\': 0.75}" --fref 20 --internal-waveform-fd-L-frame '
    namespace={'cmd':'builder','my_extra_string':value,'shlex':shlex}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pipeline-calibration-transport','exec'),namespace)
    assert shlex.split(namespace['cmd'])==['builder','--calibration-reweighting-initial-extra-args=  '+value]


@pytest.mark.parametrize('values,expected',[
    ({}, {'stream_error':'True','stream_output':'True'}),
    ({'RIFT_NOSTREAM_LOG':'1'}, {}),
    ({'RIFT_NOSTREAM_LOG':'1','RIFT_CIP_FLOCK_LOCAL':'true','RIFT_CIP_POOLS':'IGWN,CIT'}, {'MY.flock_local':'true','MY.POOLS':'"IGWN,CIT"'}),
    ({'RIFT_CIP_FLOCK_LOCAL':'false'}, {'stream_error':'True','stream_output':'True'}),
])
@pytest.mark.parametrize('module_name',['dag_utils.py','dag_utils_generic.py'])
def test_cip_explicit_transport_policy(values,expected,module_name):
    import ast
    from types import SimpleNamespace
    tree=ast.parse((CODE/'RIFT/misc'/module_name).read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='write_CIP_sub')
    nodes=[n for n in function.body if isinstance(n,ast.If) and any(k in ast.unparse(n.test) for k in ['RIFT_NOSTREAM_LOG','RIFT_CIP_FLOCK_LOCAL','RIFT_CIP_POOLS'])]
    result={}
    job=SimpleNamespace(add_condor_cmd=lambda k,v:result.update({k:v}))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'cip-transport','exec'),{'os':SimpleNamespace(environ=values),'ile_job':job,'use_osg':False,'use_singularity':False})
    assert result==expected


def test_calibration_condor_value_has_no_literal_shell_quotes():
    import ast, shlex
    from types import SimpleNamespace
    source=(CODE/'bin/create_event_parameter_pipeline_BasicIteration').read_text()
    tree=ast.parse(source)
    imports=[n for n in tree.body if isinstance(n,ast.Import) and any(a.name=='shlex' for a in n.names)]
    assert imports, 'The executed builder requires a module-level shlex import'
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='opts.calibration_reweighting_initial_extra_args']
    assert len(nodes)==1
    module=ast.parse((CODE/'RIFT/misc/dag_utils_generic.py').read_text())
    definitions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['_double_up_quotes','quote_arguments']]
    namespace={};exec(compile(ast.Module(body=definitions,type_ignores=[]),'condor-quote','exec'),namespace)
    value=" --extra-waveform-kwargs \"{'fd_alignment_postevent_time': None, 'fd_centering_factor': 0.75}\" --fref 20 --internal-waveform-fd-L-frame "
    args=[]
    environment={'opts':SimpleNamespace(calibration_reweighting_initial_extra_args=value),'shlex':shlex,'dag_utils':SimpleNamespace(quote_arguments=namespace['quote_arguments']),'calibration_job':SimpleNamespace(add_arg=args.append)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'calibration-emission','exec'),environment)
    assert args==["--extra-waveform-kwargs '{''fd_alignment_postevent_time'': None, ''fd_centering_factor'': 0.75}' --fref 20 --internal-waveform-fd-L-frame"]


def test_asimov_submission_priority_is_explicit_and_validated():
    import ast, re
    from types import SimpleNamespace
    tree=ast.parse((CODE/'RIFT/asimov/rift.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='submit_dag')
    nodes=[n for n in method.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='priority' for t in n.targets)) or (isinstance(n,ast.If) and ast.unparse(n.test)=='priority is not None')]
    assert len(nodes)==2
    code=compile(ast.Module(body=nodes,type_ignores=[]),'priority-contract','exec')
    for value in [None,900,'900']:
        env={'self':SimpleNamespace(production=SimpleNamespace(meta={'scheduler':{'priority':value}})),'re':re,'command':['condor_submit_dag','workflow.dag']}
        exec(code,env)
        assert env['command']==(['condor_submit_dag','workflow.dag'] if value is None else ['condor_submit_dag','-priority','900','workflow.dag'])
    for value in [True,'900;bad','1.5']:
        env={'self':SimpleNamespace(production=SimpleNamespace(meta={'scheduler':{'priority':value}})),'re':re,'command':[]}
        with pytest.raises(ValueError):exec(code,env)


def test_asimov_psd_staging_targets_rundir_from_foreign_cwd(tmp_path,monkeypatch):
    import ast, shutil
    from types import SimpleNamespace,MethodType
    source=(CODE/'RIFT/asimov/rift.py').read_text()
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


@pytest.mark.parametrize('value,expected', [
    (True, ['--example']), ('true', ['--example']), ('True', ['--example']),
    (False, []), ('false', []), ('False', []),
    (1.0, ['--example=1.0']), (0.0, ['--example=0.0']),
    (1, ['--example=1']), (0, ['--example=0']),
])
def test_asimov_pipeline_numeric_values_are_not_boolean_flags(value, expected):
    import ast
    tree = ast.parse((CODE/'RIFT/asimov/rift.py').read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.BoolOp)
                  and 'value is True' in ast.unparse(n.test))
    namespace = {'value': value, 'key': 'example', 'command': []}
    exec(compile(ast.Module(body=[branch], type_ignores=[]),
                 'pipeline-value-transport', 'exec'), namespace)
    assert namespace['command'] == expected


@pytest.mark.parametrize('module_name', ['dag_utils', 'dag_utils_generic'])
@pytest.mark.parametrize('image,shared', [
    ('/scratch/review.sif', True), ('osdf:///review/test.sif', False),
])
def test_calibration_shared_image_filesystem_match(module_name, image, shared,
                                                  tmp_path, monkeypatch):
    import importlib
    pytest.importorskip('glue.pipeline')
    module = importlib.import_module('RIFT.misc.' + module_name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('RIFT_REQUIRE_NONWORKER', 'EPNFS')
    monkeypatch.setenv('RIFT_NOSTREAM_LOG', '1')
    job, sub = module.write_calibration_uncertainty_reweighting_sub(
        tag=module_name, exe='/usr/bin/calibration_reweighting.py', log_dir='./',
        pickle_file='dump.pickle', posterior_file='posterior.dat',
        use_osg=True, use_singularity=True, singularity_image=image,
        transfer_files=[])
    job.write_sub_file()
    body = Path(sub).read_text()
    requirements = next(line for line in body.splitlines()
                        if line.lower().startswith('requirements'))
    assert ('EPNFS' in requirements) == shared
    assert 'universe = local' not in body and 'universe = scheduler' not in body
    assert 'stream_output = True' not in body and 'stream_error = True' not in body


ACTIVE = '2 ' + FULL[2:] + ' --mc-range [9.8,10.3] --fref 10'

@pytest.mark.parametrize('rewrite', [
    ('parameter delta_mc', 'parameter eta'),  # --cip-internal-use-eta-in-sampler
    ('parameter delta_mc', 'parameter-implied eta --parameter-nofit delta_mc'),  # --use-quadratic-early
])
def test_pipeline_rewrite_cannot_strand_activated_stage(rewrite):
    line = f.stage_arguments(ACTIVE, 'auto', 10, True, 20)
    assert f.revalidate_stage(line) == line
    with pytest.raises(ValueError):
        f.revalidate_stage(line.replace(*rewrite))
    assert f.revalidate_stage(ACTIVE.replace(*rewrite)) == ACTIVE.replace(*rewrite)

def test_fref_replacement_keeps_range_literals():
    line = f.stage_arguments(ACTIVE, 'physics3', 10, True, 25)
    assert '--mc-range [9.8,10.3]' in line and "'" not in line
    assert '--fref 10' not in line and line.endswith('--fref 25.0')

def test_pseudo_pipe_revalidates_after_its_rewrites():
    import ast, types
    src = (CODE/'bin/util_RIFT_pseudo_pipe.py').read_text()
    block = next(n for n in ast.parse(src).body if isinstance(n, ast.If)
                 and 'revalidate_stage' in ast.unparse(n))
    lineno = lambda text: src[:src.index(text)].count('\n') + 1
    for rewrite in ["line.replace('parameter delta_mc','parameter eta')",
                    "line.replace('parameter delta_mc', 'parameter-implied eta"]:
        assert lineno(rewrite) < block.lineno
    assert block.lineno < lineno('with open("args_cip_list.txt"')
    code = compile(ast.Module(body=[block], type_ignores=[]), 'pseudo_pipe_block', 'exec')
    active = f.stage_arguments(ACTIVE, 'auto', 10, True, 20)
    for mode in ('auto', 'physics3'):
        ns = {'opts': types.SimpleNamespace(rf_transverse_spin_coordinates=mode), 'lines': [active]}
        exec(code, ns)
        assert ns['lines'] == [active]
        ns['lines'] = [active.replace('parameter delta_mc', 'parameter eta')]
        with pytest.raises(ValueError):
            exec(code, ns)

@pytest.mark.parametrize('z', [0., .3])
def test_prediction_matches_training_with_source_redshift(z):
    # CIP trains on detector-frame P from ILE and predicts from source-frame samples
    pytest.importorskip('lal')
    import lal
    from RIFT import lalsimutils
    low=['mc','delta_mc','chi1','chi2','cos_theta1','cos_theta2','phi1','phi2']
    cols=['delta_mc','mu1','mu2','chiMinus','s1x','s1y','s2x','s2y']+list(f.FEATURE_NAMES)
    rng=np.random.default_rng(3); n=20
    x=np.c_[rng.uniform(9.8,10.3,n),rng.uniform(0,.9,n),rng.uniform(0,.99,(n,2)),
            rng.uniform(-1,1,(n,2)),rng.uniform(0,2*np.pi,(n,2))]
    out=f.convert(x,cols,low,20.,lalsimutils.convert_waveform_coordinates,source_redshift=z)
    for row,xi in zip(out,x):
        P=lalsimutils.ChooseWaveformParams(); P.fref=20.
        for name,value in zip(low,xi):
            P.assign_param(name,value*lal.MSUN_SI if name=='mc' else value)
        P.m1*=1+z; P.m2*=1+z
        np.testing.assert_allclose(row,[f.extract(P,c) for c in cols],rtol=1e-9,atol=1e-12)

def test_liquid_null_cip_block_renders_default():
    # base template rendered with `sampler: {cip: null}`; keep that working
    liquid=pytest.importorskip('liquid')
    text=(CODE/'RIFT/asimov/rift.ini').read_text()
    start=text.index('cip-fit-method=')
    end=text.index('cip-explode-jobs=',start)
    rendered=liquid.Liquid(text[start:end],from_file=False).render(sampler={'cip':None})
    assert 'rf-transverse-spin-coordinates="auto"' in rendered
    assert 'cip-fit-method="rf"' in rendered
