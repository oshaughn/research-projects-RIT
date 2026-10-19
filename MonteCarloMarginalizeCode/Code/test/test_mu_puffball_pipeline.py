"""Stage correspondence through real pseudo_pipe/CEPP and refinement subdags."""
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys

import pytest

from RIFT.misc import mu_puffball_pipeline as policy
import shutil

CODE = Path(__file__).resolve().parents[1]
BIN = CODE/'bin'
REPO = CODE.parents[1]

def _build(tmp_path, name, extra):
    text = (REPO/'.travis/ref_ini/GW150914.ini').read_text()
    for flag in ('use_osg', 'use_osg_file_transfer', 'use_osg_cip', 'add-extrinsic', 'batch-extrinsic', 'add-extrinsic-time-resampling'):
        text = text.replace(flag+'=True', flag+'=False')
    text = re.sub(r'force-initial-grid-size=\d+', 'force-initial-grid-size=4', text)
    ini = tmp_path/'test.ini'; ini.write_text(text)
    cache = tmp_path/'fake.cache'; cache.write_text('')
    shim = tmp_path/'shim'; shim.mkdir(exist_ok=True)
    if not (shim/'python').exists():
        (shim/'python').symlink_to(sys.executable)
    env = dict(os.environ, PYTHONPATH=str(CODE), GW_SURROGATE='',
               RIFT_LOWLATENCY='True', OMP_NUM_THREADS='1',
               PATH=os.pathsep.join([str(shim),str(BIN),os.environ.get('PATH','')]))
    run = tmp_path/name
    result = subprocess.run([sys.executable,str(BIN/'util_RIFT_pseudo_pipe.py'),
        '--use-ini',str(ini),'--use-coinc',str(REPO/'.travis/ref_ini/coinc.xml'),
        '--use-rundir',str(run),'--fake-data-cache',str(cache),*extra],
        cwd=tmp_path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    (tmp_path/(name+'_build.log')).write_text(result.stdout)
    return result,run



def _execute_built_puff(submit_file, tmp_path, iteration=0):
    """Execute the built submit command with macro expansion and real XML I/O.

    The standalone run_puffball_workflow_integration.py additionally exercises
    these files under actual Condor/DAGMan, including the real plumbing jobs.
    """
    from test_puffball_transverse_reflect import _grid
    fields = dict(line.split(' = ', 1) for line in submit_file.read_text().splitlines()
                  if ' = ' in line)
    arguments = fields['arguments'].strip('"').replace('$(macroiterationnext)', str(iteration+1))
    arguments = arguments.replace('$(macroiteration)', str(iteration))
    args = shlex.split(arguments)
    source = Path(args[args.index('--inj-file')+1])
    _grid(source.with_suffix('').with_suffix(''), n=128)
    output = Path(args[args.index('--inj-file-out')+1] + '.xml.gz')
    command = [sys.executable, fields['executable'].strip(), *args]
    cwd = fields['initialdir'].replace('$(macroiteration)', str(iteration))
    proc = subprocess.run(command, cwd=cwd,
        env=dict(os.environ, PYTHONPATH=str(CODE), GW_SURROGATE='', OMP_NUM_THREADS='1'),
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, str(submit_file) + '\n' + proc.stdout + proc.stderr
    from RIFT import lalsimutils
    import numpy as np
    points = lalsimutils.xml_to_ChooseWaveformParams_array(str(output))
    assert points, (submit_file, proc.stdout)
    for point in points:
        values = [point.m1, point.m2, point.s1x, point.s1y, point.s1z,
                  point.s2x, point.s2y, point.s2z]
        assert np.isfinite(values).all() and point.m1 >= point.m2 > 0
        assert np.linalg.norm(values[2:5]) <= 1+1e-6
        assert np.linalg.norm(values[5:8]) <= 1+1e-6
    return len(points)


LEGACY = ('--parameter mc --parameter delta_mc --parameter s1z_bar --parameter s2z_bar '
          '--parameter phi1 --parameter phi2 --parameter chi1_perp_u --parameter chi2_perp_u '
          '--reflect-parameter s1z_bar --downselect-parameter s1z_bar '
          '--downselect-parameter-range [-1,1] --puff-factor 2 --force-away 0.05 --fref 20')


def test_schedule_pairs_first_finite_batch_then_legacy_including_z_and_g():
    lines = ['3 --parameter mc', 'G2 --parameter mu1', 'Z --parameter mu1']
    actual = policy.stage_arguments(lines, LEGACY)
    assert [x.split()[0] for x in actual] == ['3', 'G2', 'Z']
    first = shlex.split(actual[0])
    assert first[1:3] == ['--puff-coordinate-system', 'mu-random-transverse']
    assert '--parameter' not in first
    assert [first[i+1] for i, t in enumerate(first) if t == '--reflect-parameter'] == ['q', 's2z']
    assert '--force-away' not in first
    assert first[first.index('--puff-factor')+1] == '2'
    assert actual[1] == 'G2 ' + LEGACY
    assert actual[2] == 'Z ' + LEGACY


@pytest.mark.parametrize('args', ['--parameter lambda1', '--parameter eccentricity',
                                 '--random-parameter s1z', '--unknown value'])
def test_discovery_refuses_dropping_nuisance_puff(args):
    with pytest.raises(ValueError):
        policy.discovery_arguments(args)


def test_invalid_initial_stage():
    for lines in (['Z --parameter mc', '2 --parameter q'], ['2 --parameter q'],
                  ['0 --parameter q', 'Z --parameter q']):
        with pytest.raises(ValueError):
            policy.stage_arguments(lines, LEGACY)


def test_real_build_default_uses_legacy_single_puff(tmp_path):
    result, run = _build(tmp_path, 'baseline', ['--internal-n-iterations-subdag-max', '2'])
    assert result.returncode == 0, result.stdout[-4000:]
    assert not (run/'args_puff_list.txt').exists()
    assert '--puff-coordinate-system' not in (run/'PUFF.sub').read_text()
    _execute_built_puff(run/'PUFF.sub', tmp_path)


def test_real_build_schedule_routes_first_batch_and_refinement(tmp_path):
    result, run = _build(tmp_path, 'scheduled', [
        '--internal-puff-mu-random-transverse', '--internal-n-iterations-subdag-max', '2',
        '--rf-transverse-spin-coordinates', 'geometric4',
        '--internal-n-evaluations-per-iteration', '100'])
    assert result.returncode == 0, result.stdout[-6000:]
    cip = (run/'args_cip_list.txt').read_text().splitlines()
    puff = (run/'args_puff_list.txt').read_text().splitlines()
    assert [x.split()[0] for x in cip] == [x.split()[0] for x in puff]
    assert any('--rf-transverse-spin-coordinates geometric4' in x for x in cip)
    assert '--puff-coordinate-system mu-random-transverse' in puff[0]
    assert '--reflect-parameter q --reflect-parameter s2z' in puff[0]
    assert all('--puff-coordinate-system' not in x for x in puff[1:])
    assert all('--parameter chi1_perp_u' in x for x in puff[1:])
    assert '--puff-coordinate-system mu-random-transverse' in (run/'PUFF_0.sub').read_text()
    assert '--puff-coordinate-system' not in (run/'PUFF_1.sub').read_text()
    dag = (run/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').read_text()
    jobs = dict(re.findall(r'^JOB\s+(\S+)\s+(\S+)', dag, re.MULTILINE))
    macro = {name: int(it) for name, it in re.findall(
        r'^VARS\s+(\S+).*?macroiteration="(\d+)"', dag, re.MULTILINE)}
    first_count = int(cip[0].split()[0].lstrip('G'))
    routed = [(macro[name], Path(sub).name) for name, sub in jobs.items()
              if Path(sub).name.startswith('PUFF_')]
    assert routed
    assert all(sub == 'PUFF_0.sub' for it, sub in routed if it < first_count)
    assert any(it >= first_count and sub != 'PUFF_0.sub' for it, sub in routed)
    # Z refinement subdags inherit their stage only, never the discovery chart.
    children = list(run.glob('iteration_*_cip/PUFF.sub'))
    assert children, result.stdout[-4000:]
    for child in children:
        text = child.read_text()
        assert '--puff-coordinate-system' not in text
        assert '--parameter chi1_perp_u' in text
        child_dag = (child.parent/'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag').read_text()
        # Jumpstart has parent composites, no initial ILE/consolidation job.
        child_jobs = dict(re.findall(r'^JOB\s+(\S+)\s+(\S+)', child_dag, re.MULTILINE))
        child_iterations = {name: int(it) for name, it in re.findall(
            r'^VARS\s+(\S+).*?macroiteration="(\d+)"', child_dag, re.MULTILINE)}
        assert not any(Path(sub).name in ('join.sub', 'ILE.sub', 'ILE_puff.sub')
                       and child_iterations[name] == 0 for name, sub in child_jobs.items())
        assert any(Path(sub).name == 'join.sub' and child_iterations[name] == 1
                   for name, sub in child_jobs.items())
    # Execute the generated first-stage arguments against the real initial grid:
    # native cuts and quoting must work, not just appear in a submit file.
    code = Path(__file__).resolve().parents[1]
    args = shlex.split((run/'args_puff_stage_0.txt').read_text())[1:]
    command = [sys.executable, str(code/'bin/util_ParameterPuffball.py'), *args,
               '--inj-file', str(run/'proposed-grid.xml.gz'),
               '--inj-file-out', str(tmp_path/'actual-discovery'),
               '--mu-puff-n-proposals', '100', '--random-seed', '448']
    actual = subprocess.run(command, env=dict(os.environ, PYTHONPATH=str(code)),
                            capture_output=True, text=True, timeout=60)
    assert actual.returncode == 0, actual.stdout + actual.stderr
    assert (tmp_path/'actual-discovery_mu_puff.json').exists()
    import json
    summary = json.loads((tmp_path/'actual-discovery_mu_puff.json').read_text())
    assert set(summary['reflection_bounds']) == {'q', 's2z'}
    for sub in sorted(run.glob('PUFF_*.sub')):
        _execute_built_puff(sub, tmp_path)
    for sub in children:
        _execute_built_puff(sub, tmp_path)


def test_nonprecessing_optin_is_refused(tmp_path):
    result, _ = _build(tmp_path, 'refused', ['--internal-puff-mu-random-transverse',
                                           '--assume-nonprecessing'])
    assert result.returncode != 0
    assert 'requires a precessing analysis' in result.stdout


@pytest.mark.parametrize('failure, expected', [
    ('length', 'same number of lines'),
    ('prefix', 'prefixes must match'),
    ('conflict', 'cannot combine with --puff-args'),
    ('missing-cip', 'needs --cip-args-list'),
    ('missing-cadence', 'needs --cip-args-list and --puff-cadence'),
])
def test_real_builder_rejects_invalid_puff_schedule(tmp_path, failure, expected):
    import json
    from run_puffball_workflow_integration import build, environment
    root = tmp_path/'fixture'
    build(root, 'scheduled', False)
    command = json.loads((root/'build-command.json').read_text())
    puff = root/'puff_list.txt'
    if failure == 'length':
        puff.write_text('\n'.join(puff.read_text().splitlines()[:-1])+'\n')
    elif failure == 'prefix':
        lines = puff.read_text().splitlines(); lines[1] = lines[1].replace('G1 ', 'G2 ', 1)
        puff.write_text('\n'.join(lines)+'\n')
    elif failure == 'conflict':
        command += ['--puff-args', str(root/'puff.txt')]
    else:
        option = '--cip-args-list' if failure == 'missing-cip' else '--puff-cadence'
        i = command.index(option); del command[i:i+2]
        if failure == 'missing-cip':
            command += ['--cip-args', str(root/'cip.txt')]
    proc = subprocess.run(command, cwd=root/'run', env=environment(root),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert expected in proc.stdout + proc.stderr
