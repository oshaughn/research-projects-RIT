#!/usr/bin/env python
"""Prepare/run/finish a bounded NR extrinsic followup from ONE existing run set.

All writes go to the new output directory. The source run is read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_options(tokens, valued, switches=()):
    result=[]
    i=0
    while i<len(tokens):
        key=tokens[i].split('=',1)[0]
        if key in valued:
            if '=' not in tokens[i]:
                if i+1==len(tokens): raise ValueError('missing argument for '+key)
                i+=1
        elif key not in switches:
            result.append(tokens[i])
        i+=1
    return result


def absolute_inputs(tokens, source):
    """Resolve supported input paths before running in a fresh job directory."""
    result=list(tokens)
    for i,token in enumerate(tokens):
        key=token.split('=',1)[0]
        if key not in ('--cache','--psd-file','--coinc','--injection-file'):
            continue
        inline='=' in token
        value=token.split('=',1)[1] if inline else tokens[i+1]
        prefix=''
        if key=='--psd-file':
            detector,separator,value=value.partition('=')
            if not separator: raise ValueError('--psd-file requires detector=path')
            prefix=detector+'='
        if '://' not in value and not Path(value).is_absolute():
            value=str((source/value).resolve())
        value=prefix+value
        if inline: result[i]=key+'='+value
        else: result[i+1]=value
    return result


def posterior_status(job):
    files=list(job.glob('ile_*_integrator_status.json'))
    if len(files)!=1:
        raise ValueError('expected one modern ILE integrator status in '+str(job))
    status=json.loads(files[0].read_text())
    if status.get('collapsed') is not False:
        raise ValueError('ILE samples are not a valid posterior in {}: {}'.format(
            job,status.get('collapse_reason','missing/nonfalse collapse status')))
    return files[0]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    prepare=commands.add_parser('prepare')
    prepare.add_argument('--source-run',type=Path,required=True)
    prepare.add_argument('--template-xml',type=Path)
    prepare.add_argument('--mass-range',nargs=2,type=float,required=True)
    prepare.add_argument('--samples',type=int,default=32)
    prepare.add_argument('--seed',type=int,default=17)
    prepare.add_argument('--tool',default=shutil.which('util_NRBestEffortPosterior.py'))
    prepare.add_argument('--ile',default=shutil.which('integrate_likelihood_extrinsic_batchmode'))
    prepare.add_argument('--converter',default=shutil.which('convert_output_format_ile2inference'))
    prepare.add_argument('--group',required=True)
    prepare.add_argument('--out',type=Path,required=True)
    run=commands.add_parser('run')
    run.add_argument('--out',type=Path,required=True)
    run.add_argument('--index',type=int,required=True)
    run.add_argument('--smoke-n-max',type=int,help='bounded I/O check only; outputs excluded from finish')
    finish=commands.add_parser('finish')
    finish.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.action=='prepare':
            if args.samples<=0: raise ValueError('samples must be positive')
            if not all([args.tool,args.ile,args.converter]): raise ValueError('tool, ILE, and converter must be installed or explicitly specified')
            source=args.source_run.resolve()
            template=(args.template_xml or source/'overlap-grid-0.xml.gz').resolve()
            composite=source/'all.net'
            ile_args=source/'args_ile.txt'
            tokens=shlex.split(ile_args.read_text())
            if tokens and tokens[0]=='X': tokens=tokens[1:]
            if not tokens or any(t in tokens for t in ['--distance-marginalization','--phase-marginalization','--psi-marginalization']):
                raise ValueError('this demo requires ordinary sampled extrinsic coordinates')
            groups=[tokens[i+1] for i,t in enumerate(tokens[:-1]) if t in ('--nr-lookup-group','--nr-group')]
            if groups != [args.group]:
                raise ValueError('source must specify exactly the requested NR group; found '+repr(groups))
            data=np.loadtxt(composite, max_rows=1, ndmin=2)
            columns=data.shape[1]
            if columns not in (13,14,15): raise ValueError('unsupported composite layout')
            layout=(['--eccentric'] if columns>=14 else [])+(['--mean-per-ano'] if columns==15 else [])
            out=args.out.resolve()
            out.mkdir(parents=True,exist_ok=False)
            command=[sys.executable,str(Path(args.tool).resolve()),'plan','--composite',str(composite),
                '--mass-range',*map(str,args.mass_range),'--allow-partial-mass-range','--coalesce-repeats',
                '--template-xml',str(template),'--posterior-xml-only','--template-settings-only',
                '--n-output-samples',str(args.samples),'--seed',str(args.seed),
                '--output-dir',str(out/'plan'),*layout]
            subprocess.run(command,check=True)
            subprocess.run([sys.executable,str(Path(args.tool).resolve()),'catalog-check',
                '--manifest',str(out/'plan/manifest.json'),'--group',args.group,
                '--output',str(out/'catalog-check.json')],check=True)
            # Preserve data/prior/frequency settings. Replace only job selection/output
            # and fairdraw flags; every draw gets one conditional followup.
            tokens=strip_options(tokens,{'--sim-xml','--event','-E','--n-events-to-analyze',
                '--output-file','-o'}, {'--save-samples','--fairdraw-extrinsic-output','--resample-time-marginalization'})
            tokens=absolute_inputs(tokens,source)
            for flag in ['--time-marginalization','--vectorized','--gpu','--force-xpy']:
                if flag not in tokens: tokens.append(flag)
            tokens+=['--save-samples','--fairdraw-extrinsic-output','--resample-time-marginalization']
            config=dict(source_run=str(source), source_hashes={str(p):sha(p) for p in [composite,ile_args,template]},
                python=sys.executable,tool=str(Path(args.tool).resolve()),ile=str(Path(args.ile).resolve()),
                converter=str(Path(args.converter).resolve()),args=tokens,columns=columns,
                samples=args.samples,seed=args.seed,group=args.group)
            (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
            (out/'source_args_ile.txt').write_text(ile_args.read_text())
            print('Prepared {} posterior-grid followups in {}'.format(args.samples,out))
            print('Run one with: {} {} run --out {} --index 0'.format(sys.executable,Path(__file__).resolve(),out))
        else:
            out=args.out.resolve()
            config=json.loads((out/'config.json').read_text())
            for path,digest in config['source_hashes'].items():
                if sha(path)!=digest: raise ValueError('source changed since preparation: '+path)
            if args.action=='run':
                if not 0<=args.index<config['samples']: raise ValueError('index outside posterior grid')
                if args.smoke_n_max is not None and args.smoke_n_max<=0: raise ValueError('smoke n-max must be positive')
                job=out/('smoke' if args.smoke_n_max else 'events')/str(args.index)
                job.mkdir(parents=True,exist_ok=False)
                tokens=list(config['args'])
                if args.smoke_n_max:
                    tokens=strip_options(tokens,{'--n-max','--n-eff','--n-chunk'})
                    tokens+=['--n-max',str(args.smoke_n_max),'--n-eff','2','--n-chunk',str(min(args.smoke_n_max,10000))]
                command=[config['ile'],*tokens,'--sim-xml',str(out/'plan/intrinsic-posterior.xml.gz'),
                    '--event',str(args.index),'--n-events-to-analyze','1','--output-file',str(job/'ile')]
                (job/'command.json').write_text(json.dumps(command,indent=2)+'\n')
                env=dict(os.environ)
                env['PYTHONUNBUFFERED']='1'
                for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
                    env[key]='2'
                with (job/'ile.log').open('w') as stream:
                    subprocess.run(command,cwd=job,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                posterior_status(job)
                files=list(job.glob('ile_*_.xml.gz'))
                if len(files)!=1: raise ValueError('ILE did not emit exactly one fairdraw XML; inspect '+str(job/'ile.log'))
                converter=[config['converter']]
                if config['columns']>=14: converter+=['--export-eccentricity']
                if config['columns']==15: converter+=['--export-meanPerAno']
                with (job/'posterior.dat').open('w') as stream:
                    subprocess.run([*converter,str(files[0])],env=env,stdout=stream,check=True)
                table=np.atleast_1d(np.genfromtxt(job/'posterior.dat',names=True))
                if not len(table): raise ValueError('no saved fairdraws; increase sampling')
                print('Wrote {} conditional rows in {} ({}).'.format(len(table),job,'smoke only' if args.smoke_n_max else 'full settings'))
            else:
                output=out/'extrinsic_posterior_samples.dat'
                if output.exists(): raise ValueError('refusing to overwrite '+str(output))
                rng=np.random.default_rng(config['seed']+1)
                draws=[]
                names=None
                hashes={}
                intrinsic=np.atleast_1d(np.genfromtxt(out/'plan/intrinsic_posterior_samples.dat',names=True))
                # Require every scheduled intrinsic draw, even if its ILE failed.
                # Equal rows per event prevent fairdraw yield from changing the prior.
                for index in range(config['samples']):
                    job=out/'events'/str(index)
                    status_path=posterior_status(job)
                    hashes[str(status_path)]=sha(status_path)
                    path=job/'posterior.dat'
                    table=np.atleast_1d(np.genfromtxt(path,names=True,dtype=float))
                    if not len(table): raise ValueError('empty conditional posterior '+str(path))
                    if names is None: names=table.dtype.names
                    if names!=table.dtype.names: raise ValueError('conditional schemas differ')
                    if not all(np.all(np.isfinite(table[n])) for n in names): raise ValueError('nonfinite conditional posterior')
                    for name in intrinsic.dtype.names:
                        if name not in names or not np.allclose(table[name],intrinsic[index][name],rtol=1e-6,atol=1e-7):
                            raise ValueError('conditional intrinsic mismatch for event {}: {}'.format(index,name))
                    row=table[rng.integers(len(table))]
                    draws.append([row[n] for n in names])
                    hashes[str(path)]=sha(path)
                np.savetxt(output,draws,header=' '.join(names))
                (out/'extrinsic_provenance.json').write_text(json.dumps(dict(
                    config_sha256=sha(out/'config.json'),files=hashes,rows_per_intrinsic_draw=1,
                    note='prototype; output count is not effective sample count'),indent=2)+'\n')
                print('Wrote {} joint posterior rows to {}'.format(len(draws),output))
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__=='__main__':
    main()
