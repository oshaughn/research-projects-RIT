"""An AV worker meeting Kish target must pass CIP despite a low max-weight return."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE = Path(__file__).resolve().parents[2] / 'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py'

@pytest.mark.parametrize('metric,selected,expected_failure', [('kish', 2600, False), ('kish', 1000, True), ('max-weight', 2600, True)])
def test_actual_cip_acceptance_gate(metric, selected, expected_failure):
    tree = ast.parse(SOURCE.read_text())
    start = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cip_acceptance_neff' for t in n.targets))
    statements = tree.body[start:start+3]
    # These are the actual assignment, opt-in selection, and fail-unless gate.
    program = compile(ast.fix_missing_locations(ast.Module(body=statements,type_ignores=[])),str(SOURCE),'exec')
    env = dict(neff=100,dict_return={'av_stopping_statistics':{'selected':selected}},
               opts=SimpleNamespace(av_stop_metric=metric,fail_unless_n_eff=2500,not_worker=False),sys=__import__('sys'))
    if expected_failure:
        with pytest.raises(SystemExit): exec(program,env)
    else:
        exec(program,env)
        assert env['neff']==100
        assert env['cip_acceptance_neff']==2600


def _block_after(path, target, count):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        for i, n in enumerate(body):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in n.targets):
                return compile(ast.Module(body=body[i:i+count], type_ignores=[]), str(path), 'exec')
    raise AssertionError(target)


@pytest.mark.parametrize('line', ['--av-stop-metric kish --n-eff 700', '--av-stop-metric=kish --n-eff=700', '--av-stop-metric=kish --n-eff 700'])
def test_basic_iteration_final_kish_target_parses_both_forms(line):
    import shlex
    path = SOURCE.parents[0] / 'create_event_parameter_pipeline_BasicIteration'
    env = dict(shlex=shlex, cip_args_lines=[line], cip_args_extra='', indx=0, n_samples_per_job=50)
    exec(_block_after(path, 'final_tokens', 4), env)
    assert env['final_target'] == 700.


@pytest.mark.parametrize('method,ok', [('AV', True), ('GMM', False)])
def test_cip_accepts_kish_with_av_without_internal_use_lnl(method, ok):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.If) and 'av_stop_metric' in ast.unparse(n.test))
    class Parser:
        def error(self, message): raise SystemExit(message)
    env = dict(opts=SimpleNamespace(av_stop_metric='kish', sampler_method=method, internal_use_lnL=False), parser=Parser())
    program = compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec')
    if ok:
        exec(program, env)
    else:
        with pytest.raises(SystemExit): exec(program, env)
