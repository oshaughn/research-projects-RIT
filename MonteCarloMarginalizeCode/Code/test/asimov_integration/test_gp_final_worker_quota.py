"""Final extrinsic export quota must not multiply the Kish integration target."""
import ast,shlex
from pathlib import Path
from types import SimpleNamespace
import pytest
SOURCE=Path(__file__).resolve().parents[2]/'bin/create_event_parameter_pipeline_BasicIteration'

@pytest.mark.parametrize('mode,target',[('kish',2500.),('max-weight',12500)])
def test_actual_final_worker_append(mode,target):
    tree=ast.parse(SOURCE.read_text())
    final=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and 'indx == len(cip_args_lines) - 1' in ast.unparse(n.test) and 'opts.last_iteration_extrinsic' in ast.unparse(n.test))
    # Execute real quota statements before unrelated eccentricity-coordinate options.
    stop=next(i for i,n in enumerate(final.body) if isinstance(n,ast.If) and 'use_eccentricity_squared_sampling' in ast.unparse(n.test))
    program=compile(ast.fix_missing_locations(ast.Module(body=final.body[:stop],type_ignores=[])),str(SOURCE),'exec')
    scope=dict(shlex=shlex,opts=SimpleNamespace(last_iteration_extrinsic_nsamples=100000,cip_explode_jobs_last=8),indx=0,
               cip_args_lines=['--av-stop-metric '+mode+' --n-eff 2500'],cip_args_extra='')
    exec(program,scope)
    tokens=shlex.split(scope['cip_args_extra'])
    assert float(tokens[tokens.index('--n-eff')+1])==target
    assert int(tokens[tokens.index('--n-output-samples')+1])==12500
