"""Stage correspondence through real pseudo_pipe/CEPP and refinement subdags."""
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys

import pytest

from RIFT.misc import mu_puffball_pipeline as policy
from test_q_time_pregrid_dag import _build


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


def test_real_build_schedule_routes_first_batch_and_refinement(tmp_path):
    result, run = _build(tmp_path, 'scheduled', [
        '--internal-puff-mu-random-transverse', '--internal-n-iterations-subdag-max', '2',
        '--rf-transverse-spin-coordinates', 'geometric4',
        '--internal-force-puff-iterations', '20'])
    assert result.returncode == 0, result.stdout[-6000:]
    cip = (run/'args_cip_list.txt').read_text().splitlines()
    puff = (run/'args_puff_list.txt').read_text().splitlines()
    assert [x.split()[0] for x in cip] == [x.split()[0] for x in puff]
    assert any('--parameter-implied rf_total_azimuth' in x for x in cip)
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


def test_nonprecessing_optin_is_refused(tmp_path):
    result, _ = _build(tmp_path, 'refused', ['--internal-puff-mu-random-transverse',
                                           '--assume-nonprecessing'])
    assert result.returncode != 0
    assert 'requires a precessing analysis' in result.stdout
