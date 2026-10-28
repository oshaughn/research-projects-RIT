"""Native cropped frames must be readable by ILE before cache publication."""
from pathlib import Path
import os, subprocess, sys
import numpy as np
import pytest
CODE=Path(__file__).resolve().parents[1]
BIN=CODE/'bin'
ENV=dict(os.environ,PATH=str(BIN)+os.pathsep+os.environ['PATH'])

def run(script,args,cwd,check=True):
    result=subprocess.run([sys.executable,str(BIN/script),*map(str,args)],cwd=cwd,env=ENV,text=True,capture_output=True,timeout=120)
    if check:assert result.returncode==0,result.stdout+result.stderr
    return result

def stage_frames(tmp_path, missing=False, gap=False):
    from gwpy.timeseries import TimeSeries
    from gwpy.timeseries import TimeSeriesDict
    paths=[]
    for first,last in [(100,106),(106 if not gap else 107,112)]:
        filename=tmp_path/f'H-TEST-{first}-{last-first}.gwf'
        data=TimeSeries(np.arange(first*16,last*16,dtype=float),t0=first,sample_rate=16,channel='H1:TEST-STRAIN',name='H1:TEST-STRAIN')
        data.write(str(filename));paths.append(filename)
    cache=tmp_path/'local.cache'
    cache.write_text(''.join(f'H TEST {100+i*6} 6 {p.as_uri()}\n' for i,p in enumerate(paths)))
    original=cache.read_bytes()
    channels='--channel-name H1=TEST-STRAIN'+(' --channel-name L1=missing' if missing else '')
    (tmp_path/'args_ile.txt').write_text(f'--data-start-time 103.2 --data-end-time 108.4 {channels}')
    return original


def test_real_frame_crop_and_cache(tmp_path):
    from gwpy.timeseries import TimeSeries
    original=stage_frames(tmp_path)
    result=subprocess.run([str(BIN/'util_ForOSG_MakeTruncatedLocalFramesDir.sh'),str(tmp_path)],env=ENV,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'local_orig.cache').read_bytes()==original
    rows=(tmp_path/'local.cache').read_text().split()
    assert rows[:4]==['H1','TEST_STRAIN','102','8']
    assert rows[4].startswith('file://localhost/')
    import lal, lalframe
    stream = lalframe.FrStreamCacheOpen(lal.CacheImport(str(tmp_path/'local.cache')))
    native_data = lalframe.FrStreamReadREAL8TimeSeries(stream, 'H1:TEST-STRAIN', lal.LIGOTimeGPS(102), 8, 0)
    np.testing.assert_array_equal(native_data.data.data, np.arange(102*16,110*16,dtype=float))
    frame=next((tmp_path/'frames_dir').glob('*.gwf'))
    data=TimeSeries.read(str(frame),'H1:TEST-STRAIN')
    np.testing.assert_array_equal(data.value,np.arange(102*16,110*16,dtype=float))
    assert float(data.t0.value)==102 and float(data.dt.value)==1/16
    # The real LAL cache import/read above validates the worker contract,
    # including URI authority, rather than a deprecated path2cache wrapper.
    from lal.utils import CacheEntry
    entry=CacheEntry(str((tmp_path/'local.cache').read_text().strip()))
    np.testing.assert_array_equal(TimeSeries.read(entry.path,'H1:TEST-STRAIN').value,data.value)


@pytest.mark.parametrize('kind',['absent_detector','absent_channel','gap','empty_cache'])
def test_frame_failure_preserves_input(tmp_path,kind):
    original=stage_frames(tmp_path,missing=kind=='absent_detector',gap=kind=='gap')
    if kind=='empty_cache':
        (tmp_path/'local.cache').write_bytes(b'')
        original=b''
    if kind=='absent_channel':
        p=tmp_path/'args_ile.txt';p.write_text(p.read_text().replace('TEST-STRAIN','NOT_PRESENT'))
    result=run('util_ForOSG_MakeTruncatedLocalFramesDir.py',[tmp_path],tmp_path,False)
    assert result.returncode!=0
    assert (tmp_path/'local.cache').read_bytes()==original
    assert not (tmp_path/'frames_dir').exists()
    assert not list(tmp_path.glob('.frames-staging-*'))


def test_frame_cache_publish_failure_rolls_back(tmp_path,monkeypatch):
    import importlib.util
    original=stage_frames(tmp_path)
    script=BIN/'util_ForOSG_MakeTruncatedLocalFramesDir.py'
    spec=importlib.util.spec_from_file_location('framecrop',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(sys,'argv',[str(script),str(tmp_path)])
    def failure(*args):
        raise OSError('injected cache publish failure')
    monkeypatch.setattr(module.os,'replace',failure)
    with pytest.raises(OSError,match='injected'):
        module.main()
    assert (tmp_path/'local.cache').read_bytes()==original
    assert not (tmp_path/'frames_dir').exists()
    assert not list(tmp_path.glob('.cache-*'))


@pytest.mark.parametrize('case',['space_path','duplicate','endpoint','condor_fallback'])
def test_frame_edges(tmp_path,case):
    from gwpy.timeseries import TimeSeries
    from RIFT.misc.pipeline_arguments import format_submit_arguments
    if case=='space_path':
        tmp_path=tmp_path/'path with spaces';tmp_path.mkdir()
    original=stage_frames(tmp_path)
    argsfile=tmp_path/'args_ile.txt'
    if case=='duplicate':
        argsfile.write_text(argsfile.read_text()+' --channel-name H1=OTHER')
    elif case=='endpoint':
        argsfile.write_text(argsfile.read_text().replace('108.4','113.4'))
    elif case=='condor_fallback':
        (tmp_path/'ILE.sub').write_text('arguments = '+format_submit_arguments(argsfile.read_text().split())+'\nqueue\n')
        argsfile.unlink()
    result=run('util_ForOSG_MakeTruncatedLocalFramesDir.py',[tmp_path],tmp_path,False)
    assert (result.returncode==0)==(case == 'condor_fallback'),result.stdout+result.stderr
    if result.returncode:
        assert (tmp_path/'local.cache').read_bytes()==original
        assert not (tmp_path/'frames_dir').exists()
    else:
        frame=next((tmp_path/'frames_dir').glob('*.gwf'))
        np.testing.assert_array_equal(TimeSeries.read(str(frame),'H1:TEST-STRAIN').value,np.arange(102*16,110*16))


def test_frame_crop_accepts_input_filenames_with_spaces(tmp_path):
    stage_frames(tmp_path)
    rows=[]
    for path in sorted(tmp_path.glob('H-TEST-*.gwf')):
        target=path.with_name('input with spaces '+path.name)
        path.rename(target)
        first=int(path.name.split('-')[-2])
        rows.append(f'H TEST {first} 6 {target.as_uri()}\n')
    (tmp_path/'local.cache').write_text(''.join(rows))
    result=run('util_ForOSG_MakeTruncatedLocalFramesDir.py',[tmp_path],tmp_path,False)
    assert result.returncode==0,result.stdout+result.stderr
    import lal, lalframe
    stream=lalframe.FrStreamCacheOpen(lal.CacheImport(str(tmp_path/'local.cache')))
    data=lalframe.FrStreamReadREAL8TimeSeries(stream,'H1:TEST-STRAIN',lal.LIGOTimeGPS(102),8,0)
    np.testing.assert_array_equal(data.data.data,np.arange(102*16,110*16,dtype=float))


def test_frame_reader_failure_preserves_unpublished_cache(tmp_path,monkeypatch):
    import importlib.util
    original=stage_frames(tmp_path)
    script=BIN/'util_ForOSG_MakeTruncatedLocalFramesDir.py'
    spec=importlib.util.spec_from_file_location('framecrop_reader_failure',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(sys,'argv',[str(script),str(tmp_path)])
    native_open=module.lalframe.FrFileOpenURL
    def failure(url,*args):
        if '.frames-staging-' in str(url):
            raise RuntimeError('injected ILE reader failure')
        return native_open(url,*args)
    monkeypatch.setattr(module.lalframe,'FrFileOpenURL',failure)
    with pytest.raises(RuntimeError,match='injected ILE reader failure'):
        module.main()
    assert (tmp_path/'local.cache').read_bytes()==original
    assert not (tmp_path/'frames_dir').exists()
    assert not list(tmp_path.glob('.frames-staging-*'))
