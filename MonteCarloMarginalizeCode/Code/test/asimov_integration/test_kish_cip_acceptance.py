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
