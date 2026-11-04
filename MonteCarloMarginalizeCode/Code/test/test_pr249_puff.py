"""CLI seam recovery and physical-identity checks for PR 249."""
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]


def plot_functions(**settings):
    # The plotting script runs a full analysis at import time. Execute its real
    # function definitions without the CLI/main block, as existing script tests do.
    import RIFT.lalsimutils as lsu
    namespace = dict(np=np, lsu=lsu, LISA=False, eccentricity=False,
                     precessing=False, non_spinning=False, puffing_file_cache={})
    namespace.update(settings)
    tree = ast.parse((CODE/'bin/plot_RIFT.py').read_text())
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    # Two unrelated functions have defaults referencing these globals.
    namespace.update(run_diagnostics={}, JSD_threshold=.005)
    exec(compile(functions, str(CODE/'bin/plot_RIFT.py'), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def test_matching_preserves_transverse_spin_identity(tmp_path):
    fn = plot_functions()
    data = np.zeros((2, 13))
    data[:, 1:3] = [20., 10.]
    data[:, 3] = [.1, .8]
    data[:, 9] = [100., 0.]
    data[:, 10] = .1
    path = tmp_path/'composite'
    np.savetxt(path, data)
    names = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
    params = {n: data[:, i+1].copy() for i, n in enumerate(names)}
    result = fn.match_to_composite(params, str(path))
    assert result['matched_fraction'] == 1.
    assert result['lnL'].tolist() == [100., 0.]
    del params['s1x']
    assert fn.match_to_composite(params, str(path)) is None


def test_matching_identifies_periodic_anomaly_lifts(tmp_path):
    fn = plot_functions(eccentricity=True)
    data = np.zeros((2, 15))
    data[:, 1:3] = [[20., 10.], [21., 11.]]
    data[:, 9] = .1
    data[:, 10] = [0., .4]
    data[:, 11] = [12., 14.]
    data[:, 12] = .1
    path = tmp_path/'composite'
    np.savetxt(path, data)
    layout = fn.composite_column_layout()
    params = {name: data[:, col].copy() for name, col in layout.items() if name not in ['lnL', 'err']}
    params['meanPerAno'] += 4*np.pi
    result = fn.match_to_composite(params, str(path))
    assert result['lnL'].tolist() == [12., 14.]


def test_grid_reader_keeps_all_spins(monkeypatch):
    fn = plot_functions()
    p = SimpleNamespace(m1=20*fn.lsu.lsu_MSUN, m2=10*fn.lsu.lsu_MSUN,
                        s1x=.1, s1y=.2, s1z=.3, s2x=.4, s2y=.5, s2z=.6)
    monkeypatch.setattr(fn.lsu, 'xml_to_ChooseWaveformParams_array', lambda _: [p])
    params = fn.read_grid_parameters('unused')
    assert params['s1x'].tolist() == [.1]
    assert params['s2y'].tolist() == [.5]


def test_footprint_handles_fixed_spins_and_precession():
    rng = np.random.default_rng(44)
    posterior = {name: rng.normal(size=400) for name in ['m1', 'm2', 's1x', 's1y', 's2x', 's2y']}
    posterior.update(s1z=np.zeros(400), s2z=np.zeros(400))
    puff = {k: v[:3].copy() for k, v in posterior.items()}
    fn = plot_functions(precessing=True)
    puff['s1x'][0] = 100.
    puff['s1z'][1] = .1
    result = fn.puffing_footprint(puff, posterior)
    assert result['outside'].tolist() == [True, True, False]


@pytest.fixture
def seam_cloud(tmp_path):
    import lal
    from RIFT import lalsimutils as lsu
    rng = np.random.default_rng(151)
    angles = rng.uniform(-.08, .08, 160)
    points = []
    for angle in angles:
        p = lsu.ChooseWaveformParams()
        p.m1 = rng.uniform(25, 28)*lal.MSUN_SI
        p.m2 = rng.uniform(12, 14)*lal.MSUN_SI
        p.eccentricity = .15 + .35*angle + rng.normal(0, .001)
        p.meanPerAno = angle % (2*np.pi)
        p.fmin = p.fref = 10.
        points.append(p)
    lsu.ChooseWaveformParams_array_to_xml(points, fname=str(tmp_path/'input'))
    return tmp_path


def run_puff(folder, name, extra=(), coordinates=('eccentricity', 'meanPerAno', 'mc', 'delta_mc')):
    from RIFT import lalsimutils as lsu
    command = [sys.executable, str(CODE/'bin/util_ParameterPuffball.py'),
               '--inj-file', str(folder/'input.xml.gz'), '--inj-file-out', str(folder/name),
               '--random-seed', '17', '--puff-factor', '.5']
    for coordinate in coordinates:
        command += ['--parameter', coordinate]
    command += list(extra)
    done = subprocess.run(command, cwd=folder, env=dict(os.environ, PYTHONPATH=str(CODE), GW_SURROGATE=''),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-5000:]
    return lsu.xml_to_ChooseWaveformParams_array(str(folder/(name+'.xml.gz')))


@pytest.mark.parametrize('center', ['gap', 'mean'])
def test_native_unwrap_recovers_seam_ridge_and_full_period_cut(seam_cloud, center):
    folder = seam_cloud
    flags = ['--unwrap-parameter', 'meanPerAno', '--unwrap-center', center]
    plain = run_puff(folder, 'plain')
    unwrapped = run_puff(folder, 'unwrapped', flags)
    bounded = run_puff(folder, 'bounded', flags + ['--downselect-parameter', 'meanPerAno',
                                                '--downselect-parameter-range', str([0., 2*np.pi])])
    assert len(unwrapped) == len(bounded) == 160
    assert np.allclose([p.meanPerAno for p in bounded], [p.meanPerAno for p in unwrapped])
    def residual(points):
        return np.std([p.eccentricity - .15 - .35*np.angle(np.exp(1j*p.meanPerAno)) for p in points])
    assert residual(unwrapped) < .003
    assert residual(plain) > .03
    assert all(0 <= p.meanPerAno < 2*np.pi for p in bounded)
    from RIFT import lalsimutils as lsu
    original = lsu.xml_to_ChooseWaveformParams_array(str(folder/'input.xml.gz'))
    assert [(p.fref, p.fmin) for p in bounded] == [(p.fref, p.fmin) for p in original]


def test_zero_puff_and_single_angle_have_expected_width(seam_cloud):
    from RIFT import lalsimutils as lsu
    original = lsu.xml_to_ChooseWaveformParams_array(str(seam_cloud/'input.xml.gz'))
    flags = ['--unwrap-parameter', 'meanPerAno', '--puff-factor', '0']
    for coords in [('meanPerAno',), ('meanPerAno', 'mc', 'delta_mc', 'eccentricity')]:
        result = run_puff(seam_cloud, 'zero'+str(len(coords)), flags, coords)
        assert len(result) == len(original)
        assert np.allclose([p.meanPerAno for p in result], [p.meanPerAno for p in original], atol=1e-7)
    result = run_puff(seam_cloud, 'single', ['--unwrap-parameter', 'meanPerAno'], ('meanPerAno',))
    displacement = np.angle(np.exp(1j*(np.array([p.meanPerAno for p in result]) - [p.meanPerAno for p in original])))
    assert np.std(displacement) > .01  # independent draws, not a shared shift


def test_periodic_pipeline_preserves_effective_phase_and_selects_native_angles():
    from RIFT.misc.periodic_puff_pipeline import arguments
    source = '--parameter=phi1 --parameter phi2 --parameter meanPerAno --eccentric-phase-coordinates'
    result = arguments(source, 'gap')
    assert '--unwrap-parameter phi1' in result and '--unwrap-parameter phi2' in result
    assert '--unwrap-parameter meanPerAno' not in result
    assert arguments(source) == source
    with pytest.raises(ValueError):
        arguments(source, 'invalid')


@pytest.mark.parametrize('mode, expected', [(None, None), (False, 'off'), (True, 'gap'), ('mean', 'mean'), ('gap', 'gap')])
def test_asimov_periodic_puff_render(mode, expected):
    liquid = pytest.importorskip('liquid')
    import configparser
    text = (CODE/'RIFT/asimov/rift.ini').read_text()
    start = text.index('# Native periodic puff chart')
    end = text.index('#\n# CIP Settings', start)
    sampler = {'puff': {}}
    if mode is not None:
        sampler['puff']['periodic coordinates'] = mode
    rendered = liquid.Liquid(text[start:end], from_file=False).render(sampler=sampler)
    parser = configparser.RawConfigParser()
    parser.read_string('[rift-pseudo-pipe]\n'+rendered)
    actual = parser.get('rift-pseudo-pipe', 'puff-periodic-mode', fallback=None)
    assert (actual.strip('"') if actual else None) == expected



def _build_periodic(tmp_path, ini, extra=()):
    # O4c does not carry O4d's DAG-test helper modules. Build with the native
    # reference fixtures directly, using a tiny grid and no strain IO/submission.
    import re
    text = ini.read_text()
    for flag in ('use_osg', 'use_osg_file_transfer', 'use_osg_cip'):
        text = text.replace(flag+'=True', flag+'=False')
    text = re.sub(r'force-initial-grid-size=\d+', 'force-initial-grid-size=4', text)
    ini.write_text(text)
    cache = tmp_path/'fake.cache'
    cache.write_text('')
    repo = CODE.parents[1]
    run = tmp_path/'run'
    env = dict(os.environ, PYTHONPATH=str(CODE), GW_SURROGATE='',
               PATH=os.pathsep.join([str(CODE/'bin'), str(Path(sys.executable).parent), os.environ.get('PATH','')]),
               OMP_NUM_THREADS='1', RIFT_LOWLATENCY='True',
               SINGULARITY_RIFT_IMAGE='foo', SINGULARITY_BASE_EXE_DIR='/usr/bin/')
    proc = subprocess.run([sys.executable, str(CODE/'bin/util_RIFT_pseudo_pipe.py'),
        '--use-ini', str(ini), '--use-coinc', str(repo/'.travis/ref_ini/coinc.xml'),
        '--use-rundir', str(run), '--fake-data-cache', str(cache), *extra],
        cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=300)
    (tmp_path/'build.log').write_text(proc.stdout)
    return proc, run

def test_real_pipeline_propagates_native_anomaly_to_puff_jobs(tmp_path):
    import re
    REF_INI = CODE.parents[1]/'.travis/ref_ini/GW150914.ini'
    text = REF_INI.read_text().replace('[engine]', '[engine]\necc_min=0.01\necc_max=0.3')
    text = re.sub(r'(?m)^internal-truncate-files-for-osg-file-transfer=.*$',
                  'internal-truncate-files-for-osg-file-transfer=False', text)
    liquid = pytest.importorskip('liquid')
    template = (CODE/'RIFT/asimov/rift.ini').read_text()
    start = template.index('# Native periodic puff chart')
    end = template.index('#\n# CIP Settings', start)
    rendered = liquid.Liquid(template[start:end], from_file=False).render(sampler={'puff': {'periodic coordinates': 'gap'}})
    text += '\nassume-eccentric=True\nuse-meanPerAno=True\n' + rendered
    text += 'force-ecc-min=0.01\nforce-ecc-max=0.3\nile-no-gpu=True\n'
    text = text.replace('ile-force-gpu=True', 'ile-force-gpu=False')
    ini = tmp_path/'periodic.ini'
    ini.write_text(text)
    proc, rundir = _build_periodic(tmp_path, ini)
    (tmp_path/'build.log').write_text(proc.stdout)
    assert proc.returncode == 0, proc.stdout[-6000:]
    assert '--unwrap-parameter meanPerAno' in (rundir/'args_puff.txt').read_text()
    puffs = [p.read_text() for p in rundir.rglob('*.sub') if 'util_ParameterPuffball.py' in p.read_text()]
    assert puffs
    assert all('--unwrap-parameter meanPerAno' in s and '--unwrap-center gap' in s for s in puffs)


def test_multiple_periodic_exclusion_matches_exhaustive_lifts():
    from itertools import product
    from RIFT.misc.periodic_puff import outside_neighborhood
    rng = np.random.default_rng(81)
    existing = rng.normal(size=(8, 3))
    proposed = rng.normal(size=(10, 3))
    vi = np.array([[2., .7, -.6], [.7, 1., .5], [-.6, .5, 2.]])
    periods = np.array([2*np.pi, np.pi])
    columns = [1, 2]
    radius = .9
    expected = []
    for point in proposed:
        best = np.inf
        for k in product(range(-6, 7), repeat=2):
            delta = existing - point
            delta[:, columns] += periods*np.array(k)
            best = min(best, np.min(np.einsum('...i,ij,...j->...', delta, vi, delta)))
        expected.append(best > radius**2)
    actual = outside_neighborhood(existing, proposed, vi, columns, periods, radius, block_size=3)
    assert np.array_equal(actual, expected)
    shifted = proposed.copy()
    shifted[:, columns] += periods*np.array([7, -4])
    assert np.array_equal(actual, outside_neighborhood(existing, shifted, vi, columns, periods, radius))


def test_real_transverse_build_propagates_both_azimuths(tmp_path):
    ini = tmp_path/'reference.ini'
    ini.write_text((CODE.parents[1]/'.travis/ref_ini/GW150914.ini').read_text())
    result, run = _build_periodic(tmp_path, ini, [
        '--internal-puff-transverse', '--puff-periodic-mode', 'gap',
        '--internal-n-iterations-subdag-max', '2'])
    assert result.returncode == 0, result.stdout[-6000:]
    args = (run/'args_puff.txt').read_text()
    assert '--unwrap-parameter phi1' in args and '--unwrap-parameter phi2' in args
    assert '--unwrap-parameter phi1' in (run/'PUFF.sub').read_text()
    # Refinement subdags consume the same resolved file.
    nested = list(run.glob('iteration_*_cip/PUFF.sub'))
    assert nested
    assert all('--unwrap-parameter phi1' in p.read_text() and '--unwrap-parameter phi2' in p.read_text() for p in nested)


def test_mu_discovery_strips_only_early_unwrap_chart():
    from RIFT.misc.mu_puffball_pipeline import stage_arguments
    args = '--parameter mc --parameter phi1 --parameter phi2 --unwrap-parameter phi1 --unwrap-parameter phi2 --unwrap-center gap --puff-factor 2 --force-away .05'
    first, later = stage_arguments(['2 --parameter mc', 'Z --parameter mc'], args)
    assert '--unwrap-parameter' not in first and '--unwrap-center' not in first
    assert later == 'Z ' + args


def test_html_diagnostics_is_self_contained_and_escapes_text(tmp_path):
    import base64
    import glob
    import html
    import io
    import json
    import re
    import time
    fn = plot_functions(os=os, glob=glob, html=html, json=json, re=re, time=time, base64=base64, io=io)
    ns = fn.write_html_diagnostics.__globals__
    tree = ast.parse((CODE/'bin/plot_RIFT.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ['HTML_TEMPLATE', 'PUFFING_FOLDER'] for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<constants>', 'exec'), ns)
    plots = tmp_path/'plots'
    plots.mkdir()
    (plots/'Diagnostics.txt').write_text('####################\n# Run summary\n####################\n<unsafe>\n\n####################\n# ILE diagnostics\n####################\nILE status: GOOD\n')
    fn.write_html_diagnostics(str(tmp_path))
    page = (plots/'Diagnostics.html').read_text()
    assert '<!DOCTYPE html>' in page
    assert '__MANIFEST__' not in page
    assert '&lt;unsafe&gt;' in page


def test_effective_phase_and_native_spin_azimuths_share_periodic_exclusion(seam_cloud):
    from RIFT import lalsimutils as lsu
    points = lsu.xml_to_ChooseWaveformParams_array(str(seam_cloud/'input.xml.gz'))
    rng = np.random.default_rng(55)
    for p in points:
        a, b = rng.uniform(0, 2*np.pi, 2)
        p.s1x, p.s1y = .2*np.cos(a), .2*np.sin(a)
        p.s2x, p.s2y = .2*np.cos(b), .2*np.sin(b)
    lsu.ChooseWaveformParams_array_to_xml(points, fname=str(seam_cloud/'input'))
    output = run_puff(seam_cloud, 'combined', [
        '--eccentric-phase-coordinates', '--unwrap-parameter', 'phi1',
        '--unwrap-parameter', 'phi2', '--force-away', '.05'],
        ('mc', 'delta_mc', 'eccentricity', 'meanPerAno', 'phi1', 'phi2'))
    assert len(output) > 100
    assert all(np.isfinite(p.meanPerAno) and 0 <= p.meanPerAno < 2*np.pi for p in output)
