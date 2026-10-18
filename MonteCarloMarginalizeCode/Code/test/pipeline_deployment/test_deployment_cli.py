"""Actual CLI/generated-DAG and frame IO checks; no jobs are submitted."""
from pathlib import Path
import os
import re
import subprocess
import sys
import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[2]
BIN = CODE / 'bin'
INPUTS = CODE / 'test/backends/inputs'
ENV = dict(os.environ, PATH=str(BIN)+os.pathsep+os.environ['PATH'],
           PYTHONPATH=str(CODE), RIFT_DAG_BACKEND='htcondor',
           LIGO_ACCOUNTING='ligo.sim.o4.cbc.pe.rift', LIGO_USER_NAME='test',
           SINGULARITY_RIFT_IMAGE='/test/container.sif', SINGULARITY_BASE_EXE_DIR='/test/bin', GW_SURROGATE='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')


def run(script, args, cwd, check=True):
    result = subprocess.run([sys.executable, str(BIN / script), *map(str, args)],
                            cwd=cwd, env=ENV, text=True, capture_output=True, timeout=120)
    (cwd / (script+'.log')).write_text(result.stdout+result.stderr)
    if check:
        assert result.returncode == 0, result.stdout+result.stderr
    return result


@pytest.fixture
def grid(tmp_path):
    import lal
    from RIFT import lalsimutils as lsu
    points = [lsu.ChooseWaveformParams(m1=(30+i)*lal.MSUN_SI, m2=20*lal.MSUN_SI) for i in range(7)]
    filename = tmp_path / 'grid'
    lsu.ChooseWaveformParams_array_to_xml(points, str(filename))
    return filename.with_suffix('.xml.gz')


@pytest.mark.parametrize('argv,count', [
    (['--foo', "{'x': 1}", '--n-events-to-analyze', '2'], 4),
    (['--foo', "{'x': '--n-events-to-analyze 9'}", '--n-events-to-analyze=2'], 4),
    (['--foo=prefix--n-events-to-analyze', '9'], 7),
    (['--unrelated', 'true'], 7),
    (['--n-events-to-analyze', '2', '--n-events-to-analyze=2'], 4),
])
def test_subdag_arguments(tmp_path, grid, argv, count):
    from RIFT.misc.pipeline_arguments import format_submit_arguments
    args = format_submit_arguments(argv)
    sub = tmp_path / 'ILE.sub'
    sub.write_text('executable = /bin/true\ntransfer_executable = False\narguments = '+args+'\nqueue\n')
    result = run('create_ile_sub_dag.py', ['--sim-xml',grid,'--submit-script',sub,'--macroiteration',0,
                                         '--target-dir',tmp_path,'--output-suffix','main'],tmp_path)
    dag = (tmp_path/'iteration_0_main.dag').read_text()
    assert len(re.findall(r'^JOB ',dag,re.M)) == count
    assert 'exe is /bin/true' in result.stdout


@pytest.mark.parametrize('args', ['--n-events-to-analyze 0','--n-events-to-analyze -2',
                                 '--n-events-to-analyze x','--n-events-to-analyze',
                                 '--n-events-to-analyze 2 --n-events-to-analyze=3'])
def test_subdag_bad_arguments(tmp_path, grid, args):
    sub=tmp_path/'ILE.sub';sub.write_text(f'executable = /bin/true\narguments = "{args}"\nqueue\n')
    result=run('create_ile_sub_dag.py',['--sim-xml',grid,'--submit-script',sub,'--macroiteration',0],tmp_path,False)
    assert result.returncode != 0
    assert not list(tmp_path.glob('*.dag'))


@pytest.mark.parametrize('cap,expected', [(3,[2,1]),(1,[1]),(4,[2,2]),(100,[2,2,2,1])])
def test_exact_point_cap(tmp_path,grid,cap,expected):
    sub=tmp_path/'ILE.sub';original='executable = /bin/true\narguments = "--n-events-to-analyze 2"\nqueue\n';sub.write_text(original)
    run('create_ile_sub_dag.py',['--sim-xml',grid,'--submit-script',sub,'--macroiteration',0,
                              '--cap-points',cap,'--output-suffix','main'],tmp_path)
    dag=(tmp_path/'iteration_0_main.dag').read_text()
    if min(cap,7)%2:
        assert list(map(int,re.findall(r'macrobatchsize="(\d+)"',dag))) == expected
    else:
        assert len(re.findall(r'^JOB ',dag,re.M)) == len(expected)
    assert sub.read_text() == original


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
    frame=next((tmp_path/'frames_dir').glob('*.gwf'))
    data=TimeSeries.read(str(frame),'H1:TEST-STRAIN')
    np.testing.assert_array_equal(data.value,np.arange(102*16,110*16,dtype=float))
    assert float(data.t0.value)==102 and float(data.dt.value)==1/16
    pathcache=subprocess.run(['lalapps_path2cache'],input=str(frame)+'\n',text=True,capture_output=True,env=ENV,check=True)
    assert len(pathcache.stdout.split())==5
    from urllib.parse import urlparse, unquote
    native=pathcache.stdout.split()
    assert native[:4]==rows[:4]
    assert unquote(urlparse(native[4]).path)==unquote(urlparse(rows[4]).path)
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


@pytest.mark.parametrize('builder',['BasicIteration','AlternateIteration'])
def test_real_default_builder(tmp_path,grid,builder):
    ile=tmp_path/'ile.args';ile.write_text('X --time-marginalization --vectorized --gpu --force-xpy\n')
    cip=tmp_path/'cip.args';cip.write_text('X --parameter mc --parameter eta --fit-method rf\n')
    run('create_event_parameter_pipeline_'+builder,['--working-directory',tmp_path,'--input-grid',grid,
        '--ile-args',ile,'--cip-args',cip,'--n-iterations',2,'--condor-nogrid-nonworker'],tmp_path)
    assert (tmp_path/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').stat().st_size>0


def test_real_alternate_deployment(tmp_path,grid):
    ile=tmp_path/'ile.args';ile.write_text('X --time-marginalization --vectorized --gpu --force-xpy --distance-marginalization\n')
    cip=tmp_path/'cip.args';cip.write_text('1 --parameter mc --parameter eta --fit-method rf\n1 --parameter mc --parameter eta --fit-method rf\n')
    conv=tmp_path/'conv.args';conv.write_text('X --fref 20\n')
    puff=tmp_path/'puff.args';puff.write_text('X --parameter mc --parameter eta\n')
    frames=tmp_path/'frames_dir';frames.mkdir()
    run('create_event_parameter_pipeline_AlternateIteration',['--working-directory',tmp_path,'--input-grid',grid,
        '--ile-args',ile,'--cip-args-list',cip,'--n-iterations',2,'--condor-nogrid-nonworker',
        '--ile-request-disk','4G','--cip-request-disk','9G','--general-request-disk','2G',
        '--use-oauth-files','scitokens','--use-osg','--use-osg-cip','--use-singularity',
        '--frames-dir',frames,'--cache-file','local.cache',
        '--last-iteration-extrinsic','--last-iteration-extrinsic-nsamples',20,'--last-iteration-extrinsic-time-resampling','--convert-args',conv,
        '--puff-args',puff,'--puff-cadence',1,'--puff-max-it',2],tmp_path)
    extr=(tmp_path/'ILE_extr.sub').read_text()
    assert 'container.sif' in extr and '4G' in extr and 'scitokens' in extr
    assert '--resample-time-marginalization' in extr and '--fairdraw-extrinsic-output' in extr
    assert '--distance-marginalization' not in extr
    for sub in tmp_path.glob('CIP*.sub'):
        assert '9G' in sub.read_text() and 'scitokens' in sub.read_text()
    dag=(tmp_path/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').read_text()
    import importlib.util
    spec=importlib.util.spec_from_file_location('dag_structure',CODE/'test/dag_contract/dag_structure.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    parsed=module.parse_dag(tmp_path/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag')
    assert not module.validate_dag(parsed)
    puff_setup=parsed.nodes_for_submit('subdag_puff.sub')
    producers=parsed.nodes_for_submit('PUFF.sub')
    assert puff_setup and producers
    for child in puff_setup:
        assert parsed.parents[child] & producers


def test_condor_cap_preserves_waveform_payload(tmp_path,grid):
    import classad
    from RIFT.misc.pipeline_arguments import format_submit_arguments, parse_submit_arguments
    payload = "{'label': '--n-events-to-analyze 9', 'other': 'double\" space'}"
    sub=tmp_path/'ILE.sub'
    sub.write_text('universe = vanilla\nexecutable = /bin/true\narguments = '+format_submit_arguments(['--extra-waveform-kwargs',payload,'--n-events-to-analyze','2'])+'\nqueue\n')
    run('create_ile_sub_dag.py',['--sim-xml',grid,'--submit-script',sub,'--macroiteration',0,'--cap-points',3,'--output-suffix','main'],tmp_path)
    capped=tmp_path/'iteration_0_main_capped.sub'
    dump=tmp_path/'condor-dryrun.ad'
    result=subprocess.run(['condor_submit','-dry-run',str(dump),'macrobatchsize=1',str(capped)],env=ENV,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    ads=list(classad.parseAds(dump.read_text()))
    assert len(ads)==1
    # Native Condor emits the canonical v2 argument representation after parsing.
    encoded=str(ads[0]['Arguments'])
    argv=parse_submit_arguments('"'+encoded.replace('"','""')+'"')
    assert argv==['--extra-waveform-kwargs',payload,'--n-events-to-analyze','1']


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
    assert (result.returncode==0)==(case in ('space_path','condor_fallback')),result.stdout+result.stderr
    if result.returncode:
        assert (tmp_path/'local.cache').read_bytes()==original
        assert not (tmp_path/'frames_dir').exists()
    else:
        frame=next((tmp_path/'frames_dir').glob('*.gwf'))
        np.testing.assert_array_equal(TimeSeries.read(str(frame),'H1:TEST-STRAIN').value,np.arange(102*16,110*16))


@pytest.mark.parametrize('builder',['BasicIteration','AlternateIteration'])
@pytest.mark.parametrize('transfer',[False,True])
def test_real_pseudo_pipe_truncated_schedule_and_frames(tmp_path,grid,builder,transfer,empty_cache=False,helper_cache=False,monkeypatch=None,truncate_check=False):
    import configparser
    import lal
    import lal.series
    from igwn_ligolw import utils
    original=stage_frames(tmp_path)
    if empty_cache:
        (tmp_path/'local.cache').write_bytes(b'')
        original=b''
    if helper_cache:
        assert transfer and not empty_cache and monkeypatch is not None
        stubs=tmp_path/'drop-in';stubs.mkdir()
        datafind=stubs/'gw_data_find'
        datafind.write_text('#!'+sys.executable+'\nimport json, pathlib, sys\n'
            + 'p=pathlib.Path('+repr(str(tmp_path/'datafind-calls.jsonl'))+')\n'
            + 'with p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
            + 'print("\\n".join(line.split()[-1] for line in pathlib.Path('+repr(str(tmp_path/'local.cache'))+').read_text().splitlines()))\n')
        datafind.chmod(0o755)
        truncator=stubs/'util_ForOSG_MakeTruncatedLocalFramesDir.sh'
        truncator.write_text('#!'+sys.executable+'\nimport json, os, pathlib, sys\n'
            + 'p=pathlib.Path('+repr(str(tmp_path/'truncate-calls.jsonl'))+')\n'
            + 'with p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
            + 'os.execv('+repr(str(BIN/'util_ForOSG_MakeTruncatedLocalFramesDir.sh'))+', ['+repr(str(BIN/'util_ForOSG_MakeTruncatedLocalFramesDir.sh'))+']+sys.argv[1:])\n')
        truncator.chmod(0o755)
        monkeypatch.setitem(ENV,'PATH',str(stubs)+os.pathsep+ENV['PATH'])
        if truncate_check:monkeypatch.setitem(ENV,'RIFT_TRUNCATE_CHECK','1')
        else:monkeypatch.delitem(ENV,'RIFT_TRUNCATE_CHECK',raising=False)
    config=configparser.ConfigParser();config.read(INPUTS/'GW150914.ini')
    config.remove_section('rift-pseudo-pipe')
    config['analysis']['ifos']="['H1']"
    config['data']['channels']="{'H1': 'H1:TEST-STRAIN'}"
    config['datafind']['types']="{'H1': 'TEST'}"
    config['engine']['approx']='IMRPhenomD'
    config['lalinference']['flow']="{'H1': 20}"
    config['lalinference']['fhigh']="{'H1': 896}"
    ini=tmp_path/'input.ini'
    with ini.open('w') as handle: config.write(handle)
    psd=lal.CreateREAL8FrequencySeries('H1',lal.LIGOTimeGPS(0),0,1,lal.StrainUnit**2/lal.HertzUnit,2049)
    psd.data.data[:]=1e-46
    psdfile=tmp_path/'psd.xml.gz'
    utils.write_filename(lal.series.make_psd_xmldoc({'H1':psd}),str(psdfile),compress='gz')
    rundir=tmp_path/'run'
    result=run('util_RIFT_pseudo_pipe.py',['--use-ini',ini,'--use-coinc',INPUTS/'coinc.xml',
        '--event-time',106.4,'--use-rundir',rundir,'--manual-initial-grid',grid,
        *([] if helper_cache else ['--fake-data-cache',tmp_path/'local.cache']),'--use-online-psd-file',psdfile,
        '--skip-reproducibility','--condor-nogrid-nonworker','--internal-truncate-cip-arg-list',2,
        '--assume-precessing','--internal-use-aligned-phase-coordinates','--cip-fit-method','rf',
        *(['--use-osg','--use-osg-file-transfer','--internal-truncate-files-for-osg-file-transfer'] if transfer else []),
        '--ile-sampler-method','AV','--pipeline-builder',builder,
        '--manual-extra-ile-args'," --internal-waveform-extra-lalsuite-args \"{'PhenomXPrecVersion':320}\" "],tmp_path,check=not empty_cache)
    if empty_cache:
        assert result.returncode!=0
        assert not (rundir/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').exists()
        assert not (rundir/'frames_dir').exists()
        assert (rundir/'local.cache').read_bytes()==original
        assert (tmp_path/'local.cache').read_bytes()==original
        return
    if helper_cache:
        import json
        calls=[json.loads(x) for x in (tmp_path/'datafind-calls.jsonl').read_text().splitlines()]
        assert calls and all('-u' in x and '--server' in x for x in calls)
        assert (tmp_path/'truncate-calls.jsonl').read_text().splitlines()==['["."]']
        original=(rundir/'H_local.cache').read_bytes()
        assert original and (tmp_path/'local.cache').is_file()
    untrimmed=(rundir/'helper_cip_arg_list.txt').read_text().splitlines()
    assert len(untrimmed)>2
    groups=(rundir/'args_cip_list.txt').read_text().splitlines()
    assert len(groups)==2
    count=sum(1 if g.split()[0]=='Z' else int(g.split()[0].lstrip('G')) for g in groups)
    assert re.search(r'--n-iterations\s+'+str(count)+r'\b',result.stdout)
    assert (rundir/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').is_file()
    if transfer:
        assert list((rundir/'frames_dir').glob('*.gwf'))
        assert (rundir/'local.cache').stat().st_size>0
        assert (rundir/'local_orig.cache').read_bytes()==original

    if builder=='AlternateIteration':
        run('create_ile_sub_dag.py',['--sim-xml',grid,'--submit-script',rundir/'ILE.sub',
            '--macroiteration',0,'--output-suffix','cli_checked'],rundir)
        assert (rundir/'iteration_0_cli_checked.dag').is_file()


@pytest.mark.parametrize('cap',[0,-1])
def test_bad_point_cap(tmp_path,grid,cap):
    sub=tmp_path/'ILE.sub';sub.write_text('executable = /bin/true\nqueue\n')
    result=run('create_ile_sub_dag.py',['--sim-xml',grid,'--submit-script',sub,'--macroiteration',0,'--cap-points',cap],tmp_path,False)
    assert result.returncode!=0 and not list(tmp_path.glob('*.dag'))


def test_duplicate_argument_assignments(tmp_path,grid):
    sub=tmp_path/'ILE.sub';sub.write_text('executable = /bin/true\narguments = "--n-events-to-analyze 2"\narguments = "--n-events-to-analyze=3"\nqueue\n')
    result=run('create_ile_sub_dag.py',['--sim-xml',grid,'--submit-script',sub,'--macroiteration',0],tmp_path,False)
    assert result.returncode!=0 and not list(tmp_path.glob('*.dag'))


@pytest.mark.parametrize('missing_table',[False,True])
def test_empty_intrinsic_grid_fails_closed(tmp_path,grid,missing_table):
    from igwn_ligolw import utils, lsctables
    from RIFT import lalsimutils as lsu
    document=utils.load_filename(str(grid),contenthandler=lsu.cthdler)
    table=lsctables.SimInspiralTable.get_table(document)
    if missing_table:
        table.parentNode.removeChild(table)
    else:
        del table[:]
    utils.write_filename(document,str(grid),compress='gz')
    sub=tmp_path/'ILE.sub';sub.write_text('executable = /bin/true\narguments = "--n-events-to-analyze 2"\nqueue\n')
    result=run('create_ile_sub_dag.py',['--sim-xml',grid,'--submit-script',sub,'--macroiteration',0],tmp_path,False)
    assert result.returncode!=0 and not list(tmp_path.glob('*.dag'))


def test_real_pseudo_pipe_empty_cache_fails_before_dag(tmp_path,grid):
    test_real_pseudo_pipe_truncated_schedule_and_frames(tmp_path,grid,'BasicIteration',True,empty_cache=True)


@pytest.mark.parametrize('builder',['BasicIteration','AlternateIteration'])
@pytest.mark.parametrize('truncate_check',[False,True])
def test_real_pseudo_pipe_helper_produced_cache(tmp_path,grid,monkeypatch,builder,truncate_check):
    test_real_pseudo_pipe_truncated_schedule_and_frames(tmp_path,grid,builder,True,
        helper_cache=True,monkeypatch=monkeypatch,truncate_check=truncate_check)


def test_real_pseudo_pipe_explicit_cache_with_truncate_check(tmp_path,grid,monkeypatch):
    monkeypatch.setitem(ENV,'RIFT_TRUNCATE_CHECK','1')
    test_real_pseudo_pipe_truncated_schedule_and_frames(tmp_path,grid,'BasicIteration',True)
