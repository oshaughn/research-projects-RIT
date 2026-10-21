"""Exercise eccentric-prior staging and pseudo_pipe's actual arg-file writer.

Load the helper by path and execute the writer's AST so no LAL/HTCondor setup
is needed. The writer check covers ordering relative to final-stage truncation.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

CODE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('cip_pipeline', CODE / 'RIFT/misc/cip_pipeline.py')
module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(module)


def priors(lines):
    import shlex
    result = []
    for line in lines:
        words = shlex.split(line)
        assert words.count('--eccentricity-prior') == 1
        result.append(words[words.index('--eccentricity-prior') + 1])
    return result


@pytest.mark.parametrize('terminal', ['3', 'Z'])
@pytest.mark.parametrize('final_prior', ['uniform', 'log_uniform'])
def test_exploration_groups_and_terminal_prior(terminal, final_prior):
    lines = ['2 --parameter eccentricity_squared --eccentricity-prior log_uniform\n',
             '1 --parameter eccentricity --eccentricity-prior=log_uniform\n',
             terminal + ' --parameter eccentricity --eccentricity-prior uniform\n']
    result = module.configure_eccentricity_prior(lines, final_prior)
    assert priors(result) == ['uniform', 'uniform', final_prior]
    assert result[0].startswith('2 --parameter eccentricity_squared ')
    assert result[-1].startswith(terminal + ' --parameter eccentricity ')
    assert module.configure_eccentricity_prior(result, final_prior) == result


def test_inserted_periodic_stage_and_manual_override_stay_uniform():
    initial = '1 --parameter eccentricity_squared --eccentricity-prior log_uniform'
    periodic = initial.replace('--parameter eccentricity_squared',
                               '--parameter-nofit eccentricity --parameter-implied ecc_cos_meanPerAno')
    result = module.configure_eccentricity_prior([
        initial, periodic + ' --eccentricity-prior log_uniform',
        'Z --parameter eccentricity --eccentricity-prior uniform',
    ], 'log_uniform')
    assert priors(result) == ['uniform', 'uniform', 'log_uniform']


def test_single_final_group_and_empty_schedule():
    assert priors(module.configure_eccentricity_prior(['Z --parameter eccentricity'], 'log_uniform')) == ['log_uniform']
    assert module.configure_eccentricity_prior(['\n', '  '], 'log_uniform') == []
    with pytest.raises(ValueError):
        module.configure_eccentricity_prior([], 'typo')


@pytest.mark.parametrize('eccentric,amr', [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize('truncate', [None, 1, 2])
@pytest.mark.parametrize('deduplicate', [False, True])
def test_actual_pipeline_writer(tmp_path, monkeypatch, eccentric, amr, truncate, deduplicate):
    tree = ast.parse((CODE / 'bin/util_RIFT_pseudo_pipe.py').read_text())
    writer = next(node for node in tree.body if isinstance(node, ast.With)
                  and isinstance(node.items[0].context_expr, ast.Call)
                  and isinstance(node.items[0].context_expr.args[0], ast.Constant)
                  and node.items[0].context_expr.args[0].value == 'args_cip_list.txt')
    # O4c has no truncation option; use only the retained lines there.
    has_truncate = 'internal_truncate_cip_arg_list' in ast.dump(writer)
    lines = ['2 --parameter eccentricity_squared --eccentricity-prior log_uniform',
             '3 --parameter eccentricity --eccentricity-prior log_uniform',
             'Z --parameter eccentricity --eccentricity-prior log_uniform']
    if truncate is not None and not has_truncate:
        lines = lines[-truncate:]
    opts = SimpleNamespace(assume_eccentric=eccentric, internal_use_amr=amr,
                           eccentricity_prior='log_uniform',
                           internal_truncate_cip_arg_list=truncate,
                           internal_ile_deduplicate_grid=deduplicate)
    namespace = dict(opts=opts, lines=lines, n_iterations=6,
                     configure_eccentricity_prior=module.configure_eccentricity_prior,
                     flag_final_group_unique=module.flag_final_group_unique)
    monkeypatch.chdir(tmp_path)
    exec(compile(ast.Module(body=[writer], type_ignores=[]), '<pipeline-writer>', 'exec'), namespace)
    written = (tmp_path / 'args_cip_list.txt').read_text().splitlines()
    expected_count = truncate or 3
    assert len(written) == expected_count
    expected = ['log_uniform'] * expected_count
    if eccentric and not amr:
        expected[:-1] = ['uniform'] * (expected_count - 1)
    assert priors(written) == expected
