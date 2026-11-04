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
                           internal_ile_deduplicate_grid=deduplicate,
                           hierarchical_merger_prior_1g=False, hierarchical_merger_prior_2g=False)
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


def test_convergence_subdag_before_extrinsic_group_uses_target_prior():
    # --add-extrinsic appends a numeric group AFTER the Z convergence stage.
    result = module.configure_eccentricity_prior([
        '2 --parameter eccentricity_squared',
        'Z --parameter eccentricity',
        '1 --parameter eccentricity',
    ], 'log_uniform')
    assert priors(result) == ['uniform', 'log_uniform', 'log_uniform']


@pytest.mark.parametrize('final_prior,extrinsic,worker_subdag', [
    ('log_uniform', True, False), ('log_uniform', True, True),
    ('uniform', True, False), ('log_uniform', False, False),
])
def test_real_build_convergence_children_keep_target_prior(tmp_path, final_prior,
                                                          extrinsic, worker_subdag):
    """Build real parent/child DAGs and inspect actual CIP submit arguments."""
    import os
    import re
    import shlex
    import subprocess
    import sys

    pytest.importorskip('lal')
    pytest.importorskip('igwn_ligolw')
    repo = CODE.parents[1]
    ref = repo / '.travis/ref_ini/GW150914.ini'
    coinc = ref.with_name('coinc.xml')
    if not ref.exists() or not coinc.exists():
        pytest.skip('reference ini/coinc fixtures unavailable')
    text = ref.read_text().replace('[engine]', '[engine]\necc_min=0.001\necc_max=0.4')
    for flag in ('use_osg', 'use_osg_file_transfer', 'use_osg_cip'):
        text = text.replace(flag + '=True', flag + '=False')
    text = text.replace('internal-truncate-files-for-osg-file-transfer=True',
                        'internal-truncate-files-for-osg-file-transfer=False')
    text = text.replace('approx=IMRPhenomXPHM', 'approx=IMRPhenomD')
    text = text.replace('approx="IMRPhenomXPHM"', 'approx="IMRPhenomD"')
    text = text.replace('assume-precessing=True', 'assume-precessing=False')
    # Time-resampling conversion is independent of eccentric-prior propagation.
    text = text.replace('add-extrinsic-time-resampling=True',
                        'add-extrinsic-time-resampling=False')
    text = re.sub(r'force-initial-grid-size=\d+', 'force-initial-grid-size=4', text)
    text = re.sub(r'(?m)^add-extrinsic=True$', 'add-extrinsic=' + str(extrinsic), text)
    ini = tmp_path / 'eccentric.ini'
    ini.write_text(text)
    cache = tmp_path / 'fake.cache'
    cache.write_text('')
    shim = tmp_path / 'shim'
    shim.mkdir()
    (shim / 'python').symlink_to(sys.executable)
    env = dict(os.environ, PYTHONPATH=str(CODE), GW_SURROGATE='',
               RIFT_LOWLATENCY='True', OMP_NUM_THREADS='1',
               MPLCONFIGDIR=str(tmp_path.parent / 'mpl-cache'))
    env['PATH'] = os.pathsep.join([str(shim), str(CODE / 'bin'), env.get('PATH', '')])
    run = tmp_path / 'run'
    command = [sys.executable, str(CODE / 'bin/util_RIFT_pseudo_pipe.py'),
               '--use-ini', str(ini), '--use-coinc', str(coinc),
               '--use-rundir', str(run), '--fake-data-cache', str(cache),
               '--assume-eccentric', '--assume-nonprecessing',
               '--eccentricity-prior', final_prior,
               '--force-ecc-min', '0.001', '--force-ecc-max', '0.4',
               '--internal-n-iterations-subdag-max', '2']
    if worker_subdag:
        command += ['--cip-explode-jobs-dag', '2']
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120)
    (tmp_path / 'build.log').write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    schedule = (run / 'args_cip_list.txt').read_text().splitlines()
    assert schedule[0].split()[0] != 'Z'
    assert priors(schedule[:1]) == ['uniform']
    z_indices = [i for i, line in enumerate(schedule) if line.split()[0] == 'Z']
    assert z_indices, schedule
    if extrinsic:
        assert z_indices[-1] < len(schedule) - 1, schedule
    assert priors([schedule[i] for i in z_indices]) == [final_prior] * len(z_indices)
    children = list(run.glob('iteration_*_cip/args_cip.txt'))
    assert len(children) == len(z_indices), result.stdout[-8000:]
    for child in children:
        assert priors(child.read_text().splitlines()) == [final_prior]
        subs = list(child.parent.glob('CIP*.sub'))
        assert subs, result.stdout[-8000:]
        for sub in subs:
            arguments = next(line.split('=', 1)[1].strip() for line in sub.read_text().splitlines()
                             if line.lower().startswith('arguments'))
            # Condor quotes the entire arguments string; its contents carry the flags.
            arguments = arguments.strip('"')
            assert priors([arguments]) == [final_prior], str(sub)
        assert (child.parent / 'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').exists()
    if worker_subdag:
        assert list(run.glob('**/various_subdags/*')), result.stdout[-8000:]


@pytest.mark.parametrize('coordinate', ['eccentricity', 'eccentricity_squared'])
@pytest.mark.parametrize('ecc_min', [0.0, 0.001])
def test_target_log_prior_installed_in_actual_cip_coordinates(coordinate, ecc_min):
    """A flag in a submit file is insufficient: exercise CIP's actual dispatch."""
    import argparse
    import copy
    import sys
    import numpy as np

    tree = ast.parse((CODE / 'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and 'eccentricity' in node.name and node.name.endswith('_prior')]
    for name in ('prior_map', 'prior_range_map'):
        node = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)))
        pairs = [(key, value) for key, value in zip(node.value.keys, node.value.values)
                 if isinstance(key, ast.Constant) and str(key.value).startswith('eccentricity')]
        node.value.keys = [key for key, _ in pairs]
        node.value.values = [value for _, value in pairs]
        nodes.append(node)
    nodes += [node for node in tree.body if isinstance(node, ast.If)
              and 'opts.eccentricity_prior' in ast.unparse(node.test)]
    namespace = dict(np=np, sys=sys, ECC_MIN=ecc_min, ECC_MAX=0.4,
                     opts=SimpleNamespace(eccentricity_prior='log_uniform'),
                     low_level_coord_names=[coordinate], parser=argparse.ArgumentParser())
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<cip-prior-dispatch>', 'exec'), namespace)
    lower, upper = namespace['prior_range_map'][coordinate]
    assert 0 < lower < upper
    x = np.geomspace(lower, upper, 100)
    density = namespace['prior_map'][coordinate](x)
    np.testing.assert_allclose(density * x, np.full_like(x, 1 / np.log(upper / lower)), rtol=1e-12)
