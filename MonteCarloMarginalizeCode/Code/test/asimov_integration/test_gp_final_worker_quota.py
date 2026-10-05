"""Final extrinsic export quota must not multiply the Kish integration target."""
import ast,shlex
from pathlib import Path
from types import SimpleNamespace
import pytest
SOURCE=Path(__file__).resolve().parents[2]/'bin/create_event_parameter_pipeline_BasicIteration'

@pytest.mark.parametrize('stop_args,target',[('--av-stop-metric kish --n-eff 2500',2500.),
                                             ('--av-stop-metric max-weight --n-eff 2500',12500),
                                             # Kish stage with no explicit --n-eff: CIP's own default target
                                             ('--av-stop-metric kish',3000.)])
def test_actual_final_worker_append(stop_args,target):
    tree=ast.parse(SOURCE.read_text())
    final=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'indx == len(cip_args_lines) - 1' in ast.unparse(n.test) and 'opts.last_iteration_extrinsic' in ast.unparse(n.test))
    # Execute real quota statements before unrelated eccentricity-coordinate options.
    stop=next(i for i,n in enumerate(final.body) if isinstance(n,ast.If) and 'use_eccentricity_squared_sampling' in ast.unparse(n.test))
    program=compile(ast.fix_missing_locations(ast.Module(body=final.body[:stop],type_ignores=[])),str(SOURCE),'exec')
    scope=dict(shlex=shlex,opts=SimpleNamespace(last_iteration_extrinsic_nsamples=100000,cip_explode_jobs_last=8),indx=0,
               cip_args_lines=[stop_args],cip_args_extra='')
    exec(program,scope)
    tokens=shlex.split(scope['cip_args_extra'])
    assert float(tokens[tokens.index('--n-eff')+1])==target
    assert int(tokens[tokens.index('--n-output-samples')+1])==12500
