"""Execute emitted DAG control flow locally with drop-in science workers, no scheduler."""
from pathlib import Path
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import pytest
from test_deployment_cli import CODE, BIN, ENV, grid, run

MOCK = r'''#!/usr/bin/env python3
import json, os, pathlib, shutil, sys
from RIFT import lalsimutils as lsu
role=os.environ['MOCK_ROLE'];
if os.environ.get('MOCK_SKIP_OUTPUT')==role:sys.exit(0)
root=pathlib.Path(os.environ['MOCK_ROOT']); argv=sys.argv[1:]
def arg(flag):
 for i,x in enumerate(argv):
  if x==flag:return argv[i+1]
  if x.startswith(flag+'='):return x.split('=',1)[1]
 raise AssertionError('missing '+flag)
def required(p):
 p=pathlib.Path(p);assert p.is_file() and p.stat().st_size,p;return p
def gridcheck(p):
 p=required(p);pts=lsu.xml_to_ChooseWaveformParams_array(str(p));assert pts;return p
if role.startswith('ILE'):
 p=gridcheck(arg('--sim-xml'));event=int(arg('--event'));batch=int(arg('--n-events-to-analyze')) if any(x.startswith('--n-events-to-analyze') for x in argv) else 1
 pts=lsu.xml_to_ChooseWaveformParams_array(str(p));assert event<len(pts)
 if '--internal-waveform-extra-lalsuite-args' in argv:assert arg('--internal-waveform-extra-lalsuite-args')=='"'+"{'label': '--n-events-to-analyze 9'}"+'"'
 pathlib.Path(arg('--output-file')).write_text(json.dumps({'grid':str(p),'event':event,'batch':min(batch,len(pts)-event),'role':role}))
elif role.startswith('CIP'):
 data=required(arg('--fname'));records=[r for line in data.read_text().splitlines() for r in json.loads(line)]
 target=pathlib.Path(arg('--fname-output-samples'));stage=int(target.name.rsplit('-',1)[1]);expected=stage if role=='CIP_prior' else stage-1
 assert any(pathlib.Path(r['grid']).name=='overlap-grid-'+str(expected)+'.xml.gz' for r in records),(role,expected,records)
 if expected==1:assert any(r['role']=='ILE_puff' for r in records),(role,records)
 shutil.copyfile(gridcheck(root/'grid.xml.gz'),str(target)+'.xml.gz')
 pathlib.Path(arg('--fname-output-integral')+'_withpriorchange+annotation.dat').write_text(json.dumps({'source':str(data),'records':records}))
elif role=='PUFF':
 shutil.copyfile(gridcheck(arg('--inj-file')),arg('--inj-file-out')+'.xml.gz')
elif role=='join':
 files=list(pathlib.Path(argv[0]).glob('CME*xml'));assert files,argv
 records=[json.loads(required(p).read_text()) for p in files];assert all(x['batch']>0 for x in records)
 for kind in {x['role'] for x in records}:
  covered=[i for x in records if x['role']==kind for i in range(x['event'],x['event']+x['batch'])];assert sorted(covered)==list(range(7)),('point coverage',kind,covered)
 pathlib.Path(argv[1]+'.composite').write_text(json.dumps(records))
elif role=='unify':
 files=list(root.glob('consolidated_*.composite'));assert files
 print('\n'.join(required(p).read_text() for p in files))
elif role.startswith('evidence'):
 required(pathlib.Path(arg('--cip-dir'))/(arg('--cip-prefix')+'_withpriorchange+annotation.dat'))
 if '--prior-integral' in argv:required(arg('--prior-integral'))
 pathlib.Path(arg('--output')).write_text('mock evidence')
elif role=='convert_extr':
 files=list((root/('iteration_'+argv[0]+'_ile')).glob('EXTR_out*.xml'));assert len(files)==3,files
 records=[json.loads(required(p).read_text()) for p in files];assert all(r['role']=='ILE_extr' for r in records)
 assert sorted(i for r in records for i in range(r['event'],r['event']+r['batch']))==list(range(6))
 (root/'mock-final.json').write_text(json.dumps([str(p) for p in files]))
else:raise AssertionError('unsupported worker '+role)
with (root/'mock-worker-trace.jsonl').open('a') as f:f.write(json.dumps({'role':role,'argv':argv,'cwd':os.getcwd()})+'\n')
'''

class Executor:
    """Small explicit DAG subset, native submit parsing; unsupported semantics fail."""
    def __init__(self, root, remove_puff_edge=False, skip_output=None):
        self.root=root;self.remove_puff_edge=remove_puff_edge;self.receipt=[];self.skip_output=skip_output
        self.mock=root/'mock-worker';self.mock.write_text(MOCK);self.mock.chmod(0o755)
    def execute(self,path):
        import classad
        from RIFT.misc.pipeline_arguments import parse_submit_arguments
        nodes={};variables={};parents={};posts={}
        for raw in path.read_text().splitlines():
            f=shlex.split(raw,comments=True)
            if not f:continue
            if f[0]=='JOB':
                assert len(f)==3;nodes[f[1]]= ('JOB',f[2]);parents[f[1]]=set()
            elif f[:2]==['SUBDAG','EXTERNAL']:
                assert len(f)==4;nodes[f[2]]=('SUBDAG',f[3]);parents[f[2]]=set()
            elif f[0]=='VARS':variables[f[1]]=dict(x.split('=',1) for x in f[2:])
            elif f[0]=='PARENT':
                i=f.index('CHILD')
                for c in f[i+1:]:parents[c].update(f[1:i])
            elif f[0]=='CATEGORY':assert len(f)==3
            elif f[0]=='DOT':assert len(f)==2  # visualization metadata
            elif f[:2]==['SCRIPT','POST']:posts[f[2]]=f[3:]
            else:raise AssertionError('unsupported DAG directive '+raw)
        assert all(p in nodes for ps in parents.values() for p in ps)
        if self.remove_puff_edge:
            for n,(_,sub) in nodes.items():
                if Path(sub).name=='subdag_puff.sub':parents[n]={p for p in parents[n] if Path(nodes[p][1]).name!='PUFF.sub'}
        done=set()
        while len(done)<len(nodes):
            ready=[n for n in nodes if n not in done and parents[n]<=done];assert ready,'cycle'
            # A newly ready runtime puff builder outranks independent worker stages.
            ready.sort(key=lambda n:(Path(nodes[n][1]).name!='subdag_puff.sub',n))
            n=ready[0];kind,sub=nodes[n];subpath=Path(sub)
            if not subpath.is_absolute():subpath=self.root/subpath
            if kind=='SUBDAG':
                assert subpath.is_file(),subpath;self.execute(subpath)
            else:
                role=subpath.stem
                if role.startswith('iteration_') and role.endswith('_capped'):role='ILE'
                real=role.startswith('subdag')
                if real:assert role in ('subdag','subdag_puff'),role
                else:assert role in ('ILE','ILE_puff','ILE_extr','PUFF','join','unify','convert_extr','evidence','evidence_final') or role.startswith('CIP'),role
                text=subpath.read_text()
                if not real:text=re.sub(r'(?m)^executable\s*=.*$', 'executable = '+str(self.mock),text)
                # Count output rows natively; no homemade queue or argument parser.
                native=self.root/('native-'+n+'.sub');native.write_text(text)
                dump=self.root/('native-'+n+'.ad')
                assignments=[k+'='+v for k,v in variables.get(n,{}).items()]
                native_env={k:v for k,v in ENV.items() if k in ('PATH','PYTHONPATH','RIFT_DAG_BACKEND','LIGO_ACCOUNTING','LIGO_USER_NAME','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}
                result=subprocess.run(['condor_submit','-dry-run',str(dump),*assignments,str(native)],env=native_env,cwd=self.root,capture_output=True,text=True,timeout=30)
                assert result.returncode==0,result.stdout+result.stderr
                ads=list(classad.parseAds(dump.read_text()));assert ads
                inherited={}
                for delta in ads:
                    inherited.update(dict(delta));ad=classad.ClassAd(inherited)
                    assert str(ad.get('JobUniverse')) in ('5','12'),ad.get('JobUniverse')
                    assert not ad.get('TransferOutputRemaps'), 'unsupported remap'
                    assert str(ad.get('In','/dev/null'))=='/dev/null', 'unsupported stdin'
                    assert not ad.get('Env'), 'unsupported legacy worker environment'
                    worker_env=dict(native_env)
                    if ad.get('Environment'):
                        fields=parse_submit_arguments('"'+str(ad['Environment']).replace('"','""')+'"')
                        worker_env.update(x.split('=',1) for x in fields)
                    encoded=str(ad.get('Arguments',''));argv=parse_submit_arguments('"'+encoded.replace('"','""')+'"') if encoded else []
                    assert not any('$(' in x for x in argv),argv
                    cwd=Path(str(ad['Iwd']));assert cwd.is_dir(),cwd
                    # Validate Condor's actual transfer inputs even in this shared-FS fixture.
                    for item in str(ad.get('TransferInput','')).split(','):
                        if item:assert (cwd/item).exists(),(cwd,item)
                    exe=str(ad['Cmd']);env=dict(worker_env,MOCK_ROLE=role,MOCK_ROOT=str(self.root),MOCK_SKIP_OUTPUT=self.skip_output or '')
                    result=subprocess.run([exe,*argv],cwd=cwd,env=env,capture_output=True,text=True,timeout=30)
                    self.receipt.append({'node':n,'role':role,'real':real,'argv':argv,'cwd':str(cwd),'exit':result.returncode})
                    (self.root/'mock-execution.json').write_text(json.dumps(self.receipt,indent=2))
                    assert result.returncode==0,result.stdout+result.stderr
                    if role.startswith('ILE'):
                        flag=next(x for x in argv if x.startswith('--output-file='));assert (cwd/flag.split('=',1)[1]).is_file(),'missing worker output '+role
                    if str(ad.get('Out',''))!='' and role=='unify':Path(str(ad['Out'])).write_text(result.stdout)
            for script in ([posts[n]] if n in posts else []):
                assert Path(script[0]).name=='confirm_exist.sh','unsupported POST'
                result=subprocess.run(script,cwd=self.root,env=ENV,capture_output=True,text=True,timeout=10);assert result.returncode==0,result.stderr
            done.add(n)
        (self.root/'mock-execution.json').write_text(json.dumps(self.receipt,indent=2))

@pytest.mark.parametrize('builder,remove_edge,skip_output',[('BasicIteration',False,None),('AlternateIteration',False,None),('AlternateIteration',True,None),('BasicIteration',False,'ILE')])
def test_emitted_mock_workflow(tmp_path,grid,builder,remove_edge,skip_output):
    from RIFT.misc.pipeline_arguments import format_submit_arguments
    payload="{'label': '--n-events-to-analyze 9'}"
    ile=tmp_path/'ile.args';ile.write_text('X --time-marginalization --vectorized --gpu --force-xpy --n-eff 100 --n-events-to-analyze 2 --internal-waveform-extra-lalsuite-args "'+payload+'"\n')
    cip=tmp_path/'cip.args';cip.write_text('1 --parameter mc --parameter eta --fit-method rf\n1 --parameter mc --parameter eta --fit-method rf\n')
    puff=tmp_path/'puff.args';puff.write_text('X --parameter mc --parameter eta\n')
    conv=tmp_path/'conv.args';conv.write_text('X --fref 20\n')
    run('create_event_parameter_pipeline_'+builder,['--working-directory',tmp_path,'--input-grid',grid,'--ile-args',ile,'--cip-args-list',cip,'--n-iterations',2,'--n-samples-per-job',7,'--ile-n-events-to-analyze',2,'--n-copies',1,'--condor-nogrid-nonworker','--puff-args',puff,'--puff-cadence',1,'--puff-max-it',2,'--last-iteration-extrinsic','--last-iteration-extrinsic-nsamples',6,'--last-iteration-extrinsic-time-resampling','--convert-args',conv],tmp_path)
    ex=Executor(tmp_path,remove_edge,skip_output)
    if skip_output:
        with pytest.raises(AssertionError,match='missing worker output ILE'):ex.execute(tmp_path/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag')
        assert ex.receipt[-1]['role']=='ILE' and ex.receipt[-1]['exit']==0
        assert not (tmp_path/'mock-final.json').exists()
    elif remove_edge and builder=='AlternateIteration':
        with pytest.raises(AssertionError,match='puffball-1.xml.gz'):ex.execute(tmp_path/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag')
        assert ex.receipt[-1]['role']=='subdag_puff' and ex.receipt[-1]['exit']!=0
        assert not (tmp_path/'puffball-1.xml.gz').exists()
        assert not (tmp_path/'mock-final.json').exists()
    else:
        ex.execute(tmp_path/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag');assert (tmp_path/'mock-final.json').is_file()
        from collections import Counter
        roles=Counter(r['role'] for r in ex.receipt)
        assert roles['ILE'] and roles['ILE_puff'] and roles['ILE_extr']==3 and roles['convert_extr']==1,roles
        assert roles['CIP_0']==roles['CIP_1']==1 and roles['PUFF']>=1,roles
        if builder=='AlternateIteration':assert roles['subdag']>=2 and roles['subdag_puff']>=1,roles
        assert all(r['exit']==0 for r in ex.receipt)
        for r in ex.receipt:
            if r['role']=='ILE_extr':
                assert all(x in r['argv'] for x in ('--time-marginalization','--vectorized','--gpu','--force-xpy','--resample-time-marginalization','--fairdraw-extrinsic-output'))
        for i in (1,2):assert (tmp_path/f'overlap-grid-{i}.xml.gz').is_file()


@pytest.mark.parametrize('count',[1,2,7,8])
def test_basic_partial_batch_build(tmp_path,grid,count):
    from RIFT import lalsimutils as lsu
    points=lsu.xml_to_ChooseWaveformParams_array(str(grid))
    points=(points+[points[0]])[:count]
    lsu.ChooseWaveformParams_array_to_xml(points,str(tmp_path/'boundary-grid'))
    ile=tmp_path/'ile.args';ile.write_text('X --time-marginalization --vectorized --gpu --force-xpy --n-events-to-analyze 2\n')
    cip=tmp_path/'cip.args';cip.write_text('X --parameter mc --parameter eta --fit-method rf\n')
    run('create_event_parameter_pipeline_BasicIteration',['--working-directory',tmp_path,'--input-grid',tmp_path/'boundary-grid.xml.gz','--ile-args',ile,'--cip-args',cip,'--n-iterations',1,'--n-samples-per-job',count,'--ile-n-events-to-analyze',2,'--n-copies',1,'--condor-nogrid-nonworker'],tmp_path)
    dag=(tmp_path/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').read_text().splitlines()
    workers={shlex.split(line)[1] for line in dag if line.startswith('JOB ') and shlex.split(line)[2]=='ILE.sub'}
    starts=[]
    for line in dag:
        f=shlex.split(line)
        if f and f[0]=='VARS' and f[1] in workers:
            fields=dict(x.split('=',1) for x in f[2:]);starts.append(int(fields['macroevent']))
    assert sorted(starts)==list(range(0,count,2)),starts
