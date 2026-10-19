#!/usr/bin/env python
"""Build/submit a small real BasicIteration DAG, retaining real PUFF and plumbing.

Only likelihood and posterior construction are toys. Run with --submit on an
HTCondor host; inspect --verify after DAGMan finishes. No production analysis
settings or Condor configuration are changed. The --local-toys option changes
only ILE/CIP toy jobs to local universe to avoid cluster matchmaking latency.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import lal
import numpy as np
from RIFT import lalsimutils

CODE = Path(__file__).resolve().parents[1]
DAG = 'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag'

TOY = '''#!/usr/bin/env python
import argparse, json, os
from pathlib import Path
import lal, numpy as np
from RIFT import lalsimutils
p=argparse.ArgumentParser()
p.add_argument('--sim-xml'); p.add_argument('--event', type=int, default=0)
p.add_argument('--n-events-to-analyze', type=int, default=128)
p.add_argument('--output-file'); p.add_argument('--fname')
p.add_argument('--fname-output-samples'); p.add_argument('--fname-output-integral')
a,unknown=p.parse_known_args()
record=dict(executable=Path(__file__).name, argv=os.sys.argv[1:], cwd=os.getcwd())
if a.sim_xml:
    rows=lalsimutils.xml_to_ChooseWaveformParams_array(a.sim_xml)
    rows=rows[a.event:a.event+a.n_events_to_analyze]
    assert rows, 'toy ILE received an empty/missing grid'
    table=[]
    for i,x in enumerate(rows):
        v=[x.m1/lal.MSUN_SI,x.m2/lal.MSUN_SI,x.s1x,x.s1y,x.s1z,x.s2x,x.s2y,x.s2z]
        assert np.isfinite(v).all() and v[0]>=v[1]>0
        assert np.linalg.norm(v[2:5])<=1.000001 and np.linalg.norm(v[5:8])<=1.000001
        table.append([i]+v+[100.-((v[0]-36.)/4.)**2, .01,10000,100])
    np.savetxt(str(Path(a.output_file).with_suffix('.dat')),table)
    record.update(input=a.sim_xml, rows=len(rows), output=a.output_file)
else:
    assert a.fname and Path(a.fname).stat().st_size>0, 'CIP requires consolidated ILE output'
    rows=lalsimutils.xml_to_ChooseWaveformParams_array(os.environ['PUFF_INTEGRATION_SEEDS'])
    lalsimutils.ChooseWaveformParams_array_to_xml(rows, fname=a.fname_output_samples, fref=20.)
    record.update(input=a.fname, rows=len(rows), output=a.fname_output_samples+'.xml.gz')
trace=Path(os.environ['PUFF_INTEGRATION_TRACE']); trace.mkdir(exist_ok=True)
(trace/(str(os.getpid())+'.json')).write_text(json.dumps(record,indent=2))
'''


def environment(root):
    return dict(os.environ, PATH=os.pathsep.join([str(root/'shim'), str(CODE/'bin'),
                    str(Path(sys.executable).parent), os.environ.get('PATH','')]),
                PYTHONPATH=str(CODE), GW_SURROGATE='', OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', PUFF_INTEGRATION_SEEDS=str(root/'seeds.xml.gz'),
                PUFF_INTEGRATION_TRACE=str(root/'trace'))


def seeds(root):
    rng=np.random.default_rng(233)
    points=[]
    for _ in range(128):
        p=lalsimutils.ChooseWaveformParams()
        p.m1=rng.uniform(34,38)*lal.MSUN_SI; p.m2=rng.uniform(25,31)*lal.MSUN_SI
        for k in (1,2):
            v=rng.normal(size=3); v*=rng.uniform(.1,.9)/np.linalg.norm(v)
            for c,x in zip('xyz',v): setattr(p,'s%d%s'%(k,c),x)
        p.fref=p.fmin=20.; points.append(p)
    lalsimutils.ChooseWaveformParams_array_to_xml(points,fname=str(root/'seeds'),fref=20.)


def build(root, case, local_toys):
    root.mkdir(parents=True)
    shim=root/'shim'; shim.mkdir()
    (root/'trace').mkdir()
    python=shim/'python'
    python.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' \"$@\"\n')
    python.chmod(0o755)
    for name in ('toy_ile','toy_cip','util_ConstructIntrinsicPosterior_GenericCoordinates.py'):
        body=TOY.replace("os.environ['PUFF_INTEGRATION_SEEDS']", repr(str(root/'seeds.xml.gz')))
        body=body.replace("os.environ['PUFF_INTEGRATION_TRACE']", repr(str(root/'trace')))
        p=shim/name; p.write_text(body); p.chmod(0o755)
    seeds(root)
    legacy=('--parameter mc --parameter delta_mc --parameter s1z_bar --parameter s2z_bar '
            '--parameter phi1 --parameter phi2 --parameter chi1_perp_u --parameter chi2_perp_u '
            '--downselect-parameter chi1 --downselect-parameter-range [0,1] '
            '--downselect-parameter chi2 --downselect-parameter-range [0,1] '
            '--puff-factor 2 --force-away 0.05 --random-seed 233 --fail-if-empty')
    if case=='general':
        legacy+=' --reflect-parameter mc --mc-range [20,40] --reflect-parameter s1z_bar --downselect-parameter s1z_bar --downselect-parameter-range [-0.95,0.95]'
    if case=='invalid': legacy+=' --reflect-parameter not_a_puff_coordinate'
    if case=='empty':
        legacy=legacy.replace('--parameter delta_mc', '--parameter eta')
        legacy+=' --reflect-parameter eta --eta-range [1e-7,1e-6]'
    (root/'ile.txt').write_text('X --fmin 20 --event-time 1126259462\n')
    (root/'cip.txt').write_text('2 --parameter mc --parameter delta_mc\n')
    (root/'puff.txt').write_text('X '+legacy+'\n')
    run=root/'run'; run.mkdir()
    command=[sys.executable,str(CODE/'bin/create_event_parameter_pipeline_BasicIteration'),
             '--working-directory',str(run),'--input-grid',str(root/'seeds.xml.gz'),
             '--ile-exe',str(shim/'toy_ile'),'--cip-exe',str(shim/'toy_cip'),
             '--cip-exe-G',str(shim/'toy_cip'), '--puff-exe',str(CODE/'bin/util_ParameterPuffball.py'),
             '--ile-args',str(root/'ile.txt'),'--cip-args',str(root/'cip.txt'),
             '--puff-args',str(root/'puff.txt'),'--puff-cadence','1','--puff-max-it','5',
             '--n-iterations','2','--n-samples-per-job','128','--ile-n-events-to-analyze','128',
             '--request-memory-ILE','512','--request-memory-CIP','512',
             '--general-request-disk','100M','--use-full-submit-paths','--condor-local-nonworker']
    if case=='scheduled':
        from RIFT.misc.mu_puffball_pipeline import stage_arguments
        cip=['1 --parameter mc','G1 --parameter mc','Z --parameter mc']
        (root/'cip_list.txt').write_text('\n'.join(cip)+'\n')
        (root/'puff_list.txt').write_text('\n'.join(stage_arguments(cip,legacy))+'\n')
        for opt in ('--cip-args','--puff-args'):
            i=command.index(opt); del command[i:i+2]
        command[command.index('--n-iterations')+1]='3'
        command+=['--cip-args-list',str(root/'cip_list.txt'), '--puff-args-list',str(root/'puff_list.txt'), '--n-iterations-subdag-max','2']
    env=environment(root)
    result=subprocess.run(command,cwd=run,env=env,capture_output=True,text=True,timeout=180)
    (root/'build.log').write_text(result.stdout+result.stderr)
    (root/'build-command.json').write_text(json.dumps(command,indent=2))
    assert result.returncode==0, (root/'build.log').read_text()[-4000:]
    assert (run/DAG).exists()
    if local_toys:
        # The real PUFF/consolidation/conversion files retain the builder's
        # local-nonworker universe. Only the two toy worker types change.
        for sub in run.rglob('*.sub'):
            text=sub.read_text()
            if 'toy_ile' in text or 'toy_cip' in text or 'util_ConstructIntrinsicPosterior_GenericCoordinates.py' in text:
                sub.write_text(re.sub(r'^universe = vanilla$', 'universe = local',text,flags=re.M))
    return env


def verify(root, case):
    run=root/'run'
    metrics=json.loads((run/(DAG+'.metrics')).read_text())
    traces=[json.loads(p.read_text()) for p in (root/'trace').glob('*.json')]
    puff_logs=list(run.rglob('logs/puff-*.log'))
    errors=list(run.rglob('logs/puff-*.err'))
    if case in ('invalid','empty'):
        assert metrics['exitcode']!=0
        message='must name a --parameter' if case=='invalid' else 'Puff file will be empty'
        assert any(message in p.read_text() for p in errors)
        assert not list(run.glob('puffball-*.xml.gz'))
        assert not any('puffball-' in x['input'] for x in traces)
    else:
        assert metrics['exitcode']==0, metrics
        assert metrics['jobs_failed']==0, metrics
        assert len(puff_logs)>= (5 if case=='scheduled' else 2)
        assert any('puffball-' in x['input'] for x in traces), 'no toy ILE consumed real PUFF'
        for path in run.rglob('puffball-*.xml.gz'):
            points=lalsimutils.xml_to_ChooseWaveformParams_array(str(path))
            assert points, path
            for p in points:
                v=np.array([p.m1,p.m2,p.s1x,p.s1y,p.s1z,p.s2x,p.s2y,p.s2z])
                assert np.isfinite(v).all() and p.m1>=p.m2>0
                assert np.linalg.norm(v[2:5])<=1.000001 and np.linalg.norm(v[5:8])<=1.000001
                if case=='general':
                    assert 20-2e-6<=p.extract_param('mc')/lal.MSUN_SI<=40+2e-6
                    assert -.95-1e-7<=p.s1z<=.95+1e-7
        if case=='scheduled':
            summaries=list(run.glob('puffball-*_mu_puff.json'))
            assert len(summaries)==1
            summary=json.loads(summaries[0].read_text())
            assert summary['exported_points']>0 and set(summary['reflection_bounds'])=={'q','s2z'}
            assert not list(run.glob('iteration_*_cip/puffball-*_mu_puff.json'))
            assert any('puffball-' in x['input'] and '/iteration_2_cip/' in x['cwd'] for x in traces)
    result=dict(case=case, dagman=metrics, toy_executions=len(traces), puff_executions=len(puff_logs),
                puff_grids=len(list(run.rglob('puffball-*.xml.gz'))))
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--case',choices=['baseline','general','scheduled','invalid','empty'],required=True)
    p.add_argument('--submit',action='store_true'); p.add_argument('--verify',action='store_true')
    p.add_argument('--local-toys',action='store_true')
    args=p.parse_args(); root=args.directory.resolve()
    if args.verify: return verify(root,args.case)
    env=build(root,args.case,args.local_toys)
    if args.submit:
        config=root/'dagman.config'
        config.write_text('DAGMAN_USER_LOG_SCAN_INTERVAL = 1\nDAGMAN_SUBMIT_DELAY = 0\nDAGMAN_MAX_SUBMIT_ATTEMPTS = 1\n')
        result=subprocess.run(['condor_submit_dag','-f','-config',str(config),DAG],
                              cwd=root/'run',env=env,capture_output=True,text=True,timeout=60)
        (root/'submit.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
        print(result.stdout)

if __name__=='__main__': main()
