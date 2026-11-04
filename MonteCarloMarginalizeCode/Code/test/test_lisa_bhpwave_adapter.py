"""BHPWave contracts and explicit opt-in native CPU adapter checks."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from RIFT.LISA import bhpwave_adapter as ba


def parameters():
    return ba.BHPWaveParameters(1e6,200,0,12,1,1,2,.7,.3,.2)


def config():
    return ba.BHPWaveConfig(1024,10,0,'seconds from declared mission t=0')


def test_contract_roundtrip_and_unused_parameters_refused():
    p,c=parameters(),config()
    assert ba.BHPWaveParameters.from_dict(p.to_dict())==p
    assert ba.BHPWaveConfig.from_dict(c.to_dict())==c
    with pytest.raises(TypeError):
        ba.BHPWaveParameters.from_dict(dict(p.to_dict(),e0=.1))
    with pytest.raises(TypeError):
        ba.BHPWaveConfig.from_dict(dict(c.to_dict(),backend='gpu'))


@pytest.mark.parametrize('field,value',[('primary_msun',True),('secondary_msun',0),
    ('primary_msun',100),('a',1),('r0',0),('distance_gpc',np.nan),('qS',-1),
    ('phiK',2*np.pi),('Phi_phi0',float('inf')),('qK','0.5')])
def test_parameter_guards(field,value):
    with pytest.raises(ValueError):
        replace(parameters(),**{field:value})


@pytest.mark.parametrize('field,value',[('n_samples',True),('n_samples',1024.),
    ('n_samples',1),('n_samples',ba.MAX_SAMPLES+1),('dt_seconds',0),
    ('dt_seconds',float('inf')),('epoch_start_seconds',False),('epoch_reference','')])
def test_sampling_guards(field,value):
    with pytest.raises(ValueError):
        replace(config(),**{field:value})


def fake_adapter(monkeypatch, *, count=1024, merger=1e7, omega=.068,
                 output=None, bad_class=False, check_radius_error=False, sampling=None):
    calls={}
    class Native:
        def time_step_number(self,*args):
            calls['count_args']=args
            return count
    class Data:
        base_class=SimpleNamespace(max_orbital_frequency=lambda a:omega)
        def time_to_merger(self,*args):
            return merger
    class Evolution:
        def __init__(self,**kwargs):
            calls['evolution_init']=kwargs
        def check_radius(self,*args):
            calls['radius']=args
            if check_radius_error:
                raise ValueError('outside trajectory table')
    class Generator:
        waveform_generator=SimpleNamespace() if bad_class else Native()
        def __init__(self,**kwargs):
            calls['generator_init']=kwargs
        def __call__(self,*args,**kwargs):
            calls['evaluation']=(args,kwargs)
            return output if output is not None else [np.ones(count),-2*np.ones(count)]
    monkeypatch.setattr(ba,'_load_backend',lambda path:(
        SimpleNamespace(KerrCircularWaveform=Generator),
        SimpleNamespace(TrajectoryData=Data,InspiralGenerator=Evolution),
        SimpleNamespace(yr_MKS=100,Modot_GC1_to_S=5e-6),
        SimpleNamespace(WaveformGeneratorPy=Native),{'test_fixture':True}))
    return ba.BHPWaveAdapter(config() if sampling is None else sampling,'unused-fixture-lock'),calls


def test_native_api_mapping_and_output_metadata(monkeypatch):
    adapter,calls=fake_adapter(monkeypatch)
    result=adapter.generate(parameters())
    args,kwargs=calls['evaluation']
    assert args==(1e6,200,0,12,1,1,2,.7,.3,.2)
    assert kwargs['dt']==10 and kwargs['return_list'] is True
    assert kwargs['pad_output'] is False
    assert kwargs['select_modes']==[(2,2)] and kwargs['include_negative_m'] is True
    assert 10230<kwargs['T']*100<10230+1e-7
    assert calls['generator_init']['num_threads']==1
    assert result.polarizations.shape==(2,1024)
    assert np.all(result.polarizations[1]==-2)
    assert not result.polarizations.flags.writeable
    assert result.metadata['carrier_nyquist_ratio']<1
    assert result.epoch_reference==config().epoch_reference
    assert result.metadata['parameters']==parameters().to_dict()
    json.dumps(result.metadata,allow_nan=False)


@pytest.mark.parametrize('change,match',[
    ({'count':1023},'sample count'),({'merger':100},'evolution ends'),
    ({'omega':10},'Nyquist'),({'check_radius_error':True},'trajectory table'),
    ({'output':[np.ones(1023),np.ones(1023)]},'finite real'),
    ({'output':[np.ones(1024,dtype=complex),np.ones(1024)]},'finite real'),
    ({'output':[np.full(1024,np.nan),np.ones(1024)]},'finite real'),
])
def test_backend_support_and_output_refusals(monkeypatch,change,match):
    adapter,_=fake_adapter(monkeypatch,**change)
    with pytest.raises(ValueError,match=match):
        adapter.generate(parameters())


def test_mismatched_native_class_is_refused(monkeypatch):
    with pytest.raises(RuntimeError,match='native CPU class'):
        fake_adapter(monkeypatch,bad_class=True)


def lock_fixture(tmp_path,monkeypatch):
    data={'bhpwave/data/fixture.txt':hashlib.sha256(b'fixture').hexdigest()}
    monkeypatch.setattr(ba,'DATA_INDEX_SHA256',hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    monkeypatch.setattr(ba.platform,'system',lambda:'Linux')
    monkeypatch.setattr(ba.platform,'machine',lambda:'test_arch')
    lock={'schema_version':1,'source_revision':ba.SOURCE_REVISION,
          'python_abi':ba.sys.implementation.cache_tag,'architecture':'test_arch',
          'data_files':data,'native_module_sha256':'a'*64,
          'native_libraries':{'fixture':{'resolved':'/fixture','sha256':'b'*64}},
          'build_preparation':{},'producer_result_sha256':'c'*64}
    path=tmp_path/'lock.json';path.write_text(json.dumps(lock))
    return path,lock


def test_explicit_runtime_lock_is_read_without_backend_imports(tmp_path,monkeypatch):
    path,lock=lock_fixture(tmp_path,monkeypatch)
    actual,checksum=ba._read_lock(path)
    assert actual==lock and checksum==ba._sha256(path)


@pytest.mark.parametrize('field,value',[('schema_version',True),('source_revision','other'),
    ('python_abi','other'),('architecture','other'),('data_files',{}),
    ('native_module_sha256','invalid'),('native_libraries',{}),
    ('native_libraries',{'x':{}}),('native_libraries',{'x':{'resolved':'/a','sha256':'bad'}}),
    ('native_libraries',{'x':{'foo':1}}),('native_libraries',{'x':{'resolved':'/a','foo':1}}),
    ('native_libraries',{'x':{'sha256':'b'*64,'foo':1}})])
def test_runtime_lock_identity_guards(tmp_path,monkeypatch,field,value):
    path,lock=lock_fixture(tmp_path,monkeypatch)
    lock[field]=value;path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        ba._read_lock(path)


def test_library_guard_checks_process_mapping_and_content(tmp_path,monkeypatch):
    library=tmp_path/'libfixture.so';library.write_bytes(b'fixture')
    entry={'fixture':{'resolved':str(library),'sha256':ba._sha256(library)}}
    original=Path.read_text
    def read(path,*args,**kwargs):
        if str(path)=='/proc/self/maps':
            return '0000-1000 r-xp 0000 00:00 1 '+str(library)+'\n'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    assert ba._verify_native_libraries(entry)==entry
    library.write_bytes(b'changed')
    with pytest.raises(ValueError,match='mapped artifact'):
        ba._verify_native_libraries(entry)


@pytest.mark.skipif(os.environ.get('RIFT_RUN_BHPWAVE_REAL')!='1',
                    reason='requires the explicitly staged pinned BHPWave CPU runtime')
def test_actual_cpu_adapter_distance_and_coverage():
    manifest=os.environ['RIFT_BHPWAVE_RUNTIME_LOCK']
    adapter=ba.BHPWaveAdapter(config(),manifest)
    base=adapter.generate(parameters())
    distant=adapter.generate(replace(parameters(),distance_gpc=2))
    np.testing.assert_allclose(distant.polarizations,base.polarizations/2,rtol=2e-12,atol=0)
    assert base.n_samples==config().n_samples
    assert base.metadata['time_to_merger_seconds']>base.metadata['sample_span_seconds']
    assert base.metadata['carrier_nyquist_ratio']<1


@pytest.mark.skipif(os.environ.get('RIFT_RUN_BHPWAVE_REAL')!='1',
                    reason='requires the explicitly staged pinned BHPWave CPU runtime')
@pytest.mark.parametrize('a', [0., .5, -.5])
def test_actual_cpu_orbit_toward_observer_is_right_handed_in_response_basis(a):
    # The response takes hp + 1j*hc in a (u, v) basis with u x v along propagation.
    # An orbit whose angular momentum (qK, phiK) points at the observer then
    # rotates counterclockwise: hp*dhc/dt - hc*dhp/dt > 0. Avoid the exactly
    # face-on direction, where the in-plane azimuth is degenerate.
    adapter=ba.BHPWaveAdapter(config(),os.environ['RIFT_BHPWAVE_RUNTIME_LOCK'])
    qS,phiS=1.,2.
    def circulation(qK,phiK):
        hp,hc=adapter.generate(replace(parameters(),a=a,qS=qS,phiS=phiS,qK=qK,phiK=phiK)).polarizations
        dhp,dhc=np.gradient(hp),np.gradient(hc)
        return np.mean(hp*dhc-hc*dhp)/np.sqrt(np.mean(hp**2+hc**2)*np.mean(dhp**2+dhc**2))
    toward=circulation(np.pi-qS+1e-2,(phiS+np.pi)%(2*np.pi))
    away=circulation(qS+1e-2,phiS)
    assert toward>0.5 and away<-0.5


def test_verified_loader_rejects_changed_source_data_and_binary(tmp_path,monkeypatch):
    lock_path,lock=lock_fixture(tmp_path,monkeypatch)
    root=tmp_path/'runtime'
    relatives=['bhpwave/waveform.py','bhpwave/constants.py',
               'bhpwave/trajectory/inspiral.py','bhpwave/trajectory/geodesic.py']
    for name in relatives:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# fixture '+name+'\n')
    monkeypatch.setattr(ba,'SOURCE_FILES',{name:ba._sha256(root/name) for name in relatives})
    data_path=root/'bhpwave/data/fixture.txt';data_path.parent.mkdir()
    data_path.write_bytes(b'fixture')
    native_path=root/'bhpwaveformcy.so';native_path.write_bytes(b'compiled fixture bytes')
    lock['native_module_sha256']=ba._sha256(native_path)
    lock_path.write_text(json.dumps(lock))
    modules={
        'bhpwave.waveform':SimpleNamespace(__file__=str(root/relatives[0])),
        'bhpwave.constants':SimpleNamespace(__file__=str(root/relatives[1])),
        'bhpwave.trajectory.inspiral':SimpleNamespace(__file__=str(root/relatives[2])),
        'bhpwave.trajectory.geodesic':SimpleNamespace(__file__=str(root/relatives[3])),
        'bhpwaveformcy':SimpleNamespace(__file__=str(native_path)),
    }
    monkeypatch.setattr(ba.importlib,'import_module',lambda name:modules[name])
    monkeypatch.setattr(ba,'_verify_native_libraries',lambda entries:entries)
    assert ba._load_backend(lock_path)[-1]['source_revision']==ba.SOURCE_REVISION
    data_path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='data mismatch'):
        ba._load_backend(lock_path)
    data_path.write_bytes(b'fixture')
    native_path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='native module differs'):
        ba._load_backend(lock_path)
    native_path.write_bytes(b'compiled fixture bytes')
    (root/relatives[0]).write_text('# changed scientific source')
    with pytest.raises(ValueError,match='source mismatch'):
        ba._load_backend(lock_path)


@pytest.mark.parametrize('cause,expected',[
    (ModuleNotFoundError("No module named 'bhpwave'",name='bhpwave'),'missing BHPWave runtime'),
    (ModuleNotFoundError("No module named 'dependency'",name='dependency'),"'dependency'"),
    (ImportError("libm.so.6: version `GLIBC_2.38' not found"),'native ABI'),
])
def test_backend_import_failure_preserves_actionable_cause(tmp_path,monkeypatch,cause,expected):
    path,_=lock_fixture(tmp_path,monkeypatch)
    def fail_import(name):
        raise cause
    monkeypatch.setattr(ba.importlib,'import_module',fail_import)
    with pytest.raises(ImportError,match=expected) as failure:
        ba._load_backend(path)
    assert failure.value.__cause__ is cause
    if not isinstance(cause,ModuleNotFoundError):
        assert str(cause) in str(failure.value)


@pytest.mark.parametrize('count,dt',[(26216,5.),(131076,1.)])
def test_bounded_observation_and_fine_reference_counts_reach_generator(monkeypatch,count,dt):
    sampling=ba.BHPWaveConfig(count,dt,0.,'seconds from declared mission t=0')
    adapter,calls=fake_adapter(monkeypatch,count=count,sampling=sampling)
    result=adapter.generate(parameters())
    assert result.polarizations.shape==(2,count)
    assert result.metadata['config']==sampling.to_dict()
    assert result.metadata['sample_span_seconds']==131075.
    assert calls['evaluation'][1]['pad_output'] is False
    assert calls['count_args'][-2]==dt


@pytest.mark.parametrize('count,dt,native_dt,stride',[(26216,5.,1.,5),(131076,1.,.5,2)])
def test_explicit_generation_grid_reaches_native_and_restricts_genuine_samples(monkeypatch,count,dt,native_dt,stride):
    sampling=ba.BHPWaveConfig(count,dt,0.,'seconds from declared mission t=0',generation_dt_seconds=native_dt)
    native_count=(count-1)*stride+1
    raw=np.array([np.arange(native_count,dtype=float),-2*np.arange(native_count,dtype=float)])
    adapter,calls=fake_adapter(monkeypatch,count=native_count,sampling=sampling,output=raw)
    result=adapter.generate(parameters())
    np.testing.assert_array_equal(result.polarizations,raw[:,::stride])
    assert result.polarizations.shape==(2,count) and result.polarizations.flags.c_contiguous
    assert result.polarizations.base is None and not result.polarizations.flags.writeable
    assert calls['count_args'][-2]==native_dt and calls['evaluation'][1]['dt']==native_dt
    assert calls['evaluation'][1]['pad_output'] is False
    assert result.metadata['native_n_samples']==native_count
    assert result.metadata['native_stride']==stride and result.metadata['native_dt_seconds']==native_dt
    assert result.metadata['sample_span_seconds']==131075.


@pytest.mark.parametrize('generation_dt',[True,0.,-1.,float('inf'),float('nan'),'1',20.,3.])
def test_generation_grid_refuses_invalid_or_noncommensurate_steps(generation_dt):
    with pytest.raises(ValueError):replace(config(),generation_dt_seconds=generation_dt)


def test_generation_grid_bounds_internal_allocation_before_backend_loading():
    with pytest.raises(ValueError,match='internal native sample count'):
        ba.BHPWaveConfig(131076,1.,0.,'test seconds',generation_dt_seconds=.25)
    assert ba.BHPWaveConfig(131076,1.,0.,'test seconds',generation_dt_seconds=.5).n_samples==131076
    assert ba.BHPWaveConfig.from_dict(config().to_dict()).generation_dt_seconds is None
