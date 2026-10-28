"""Adversarial bootstrap round trips: real Bilby/PESummary -> converter -> XML."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from RIFT.misc.bootstrap import dump_samples, normalize_samples, posterior_label

CODE = Path(__file__).resolve().parents[1]
CONVERTER = CODE / 'bin/convert_output_format_inference2ile'


def posterior(n=3):
    # Standard Bilby spin-angle names, deliberately no chirp_mass/cartesian spins.
    return {name: np.asarray(value[:n], dtype=float) for name, value in {
        'mass_1': [36, 37, 38], 'mass_2': [29, 28, 27],
        'a_1': [0.1, 0.2, 0.3], 'a_2': [0.2, 0.1, 0.2],
        'tilt_1': [0, 0, 0], 'tilt_2': [0, 0, 0],
        'phi_12': [0, 0, 0], 'phi_jl': [0, 0, 0],
        'theta_jn': [0.3, 0.4, 0.5], 'psi': [0.1, 0.2, 0.3],
        'phase': [0.2, 0.4, 0.6], 'geocent_time': [1126259462]*3,
        'luminosity_distance': [100, 200, 300],
        'eccentricity': [0, 1e-6, 0.2], 'mean_anomaly': [0, 1.7, 5.9],
    }.items()}


def source_file(tmp_path, kind, data=None, reference=None):
    data = posterior() if data is None else data
    if kind.startswith('bilby'):
        bilby = pytest.importorskip('bilby')
        pandas = pytest.importorskip('pandas')
        extension = 'json' if kind.startswith('bilby-json') else 'hdf5'
        path = tmp_path / ('eccentric_result.' + extension)
        result = bilby.core.result.Result(label='eccentric', outdir=str(tmp_path),
                                          posterior=pandas.DataFrame(data), search_parameter_keys=[], priors=bilby.core.prior.PriorDict(),
                                          meta_data={'likelihood': {'waveform_arguments': {'reference_frequency': reference}}}
                                          if reference is not None else {})
        result.save_to_file(outdir=str(tmp_path), extension=extension, overwrite=True, gzip=kind.endswith('-gz'))
        if kind.endswith('-gz'):
            path = Path(str(path) + '.gz')
    else:
        h5py = pytest.importorskip('h5py')
        path = tmp_path / 'pesummary.h5'
        with h5py.File(path, 'w') as f:
            group = f.create_group('bilby-eccentric')
            if kind == 'pesummary-columns':
                table = group.create_group('posterior_samples')
                for name, values in data.items():
                    table[name] = values
            else:
                table = np.empty(len(next(iter(data.values()))), [(name, float) for name in data])
                for name in data:
                    table[name] = data[name]
                group['posterior_samples'] = table
    return path


def environment(tmp_path):
    shim = tmp_path / 'bin'
    shim.mkdir(exist_ok=True)
    if not (shim / 'python').exists():
        (shim / 'python').symlink_to(sys.executable)
    return dict(os.environ, PYTHONPATH=str(CODE), GW_SURROGATE='',
                MPLCONFIGDIR=str(tmp_path.parent / 'mpl-cache'),
                PATH=os.pathsep.join([str(shim), str(CODE / 'bin'), os.environ.get('PATH', '')]))


def convert(tmp_path, samples, extra=()):
    pytest.importorskip('lal')
    ascii_file = tmp_path / 'posterior samples.dat'
    np.savetxt(ascii_file, samples, header=' '.join(samples.dtype.names))
    output = tmp_path / 'bootstrap grid.xml.gz'
    result = subprocess.run([sys.executable, str(CONVERTER), '--posterior-samples', str(ascii_file),
                             '--output-xml', str(output), *extra], env=environment(tmp_path),
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    from RIFT import lalsimutils
    return lalsimutils.xml_to_ChooseWaveformParams_array(str(output))


@pytest.mark.parametrize('kind', ['bilby-json', 'bilby-json-gz', 'bilby-hdf5', 'pesummary-table', 'pesummary-columns'])
@pytest.mark.parametrize('add_missing', [False, True])
def test_bilby_and_pesummary_round_trip_preserves_eccentric_coordinates(tmp_path, kind, add_missing):
    path = source_file(tmp_path, kind)
    label = posterior_label(path)
    data = dump_samples(path, label, tmp_path / 'converted.dat')
    params = convert(tmp_path, data, ['--add-eccentricity-params'] if add_missing else [])
    assert len(params) == 3
    np.testing.assert_allclose([p.eccentricity for p in params], posterior()['eccentricity'], rtol=5e-7)
    np.testing.assert_allclose([p.meanPerAno for p in params], posterior()['mean_anomaly'], rtol=5e-7)
    np.testing.assert_allclose([p.s1z for p in params], posterior()['a_1'], rtol=5e-7)


def test_single_row_and_alias_without_chirp_mass(tmp_path):
    params = convert(tmp_path, normalize_samples(posterior(1)))
    assert len(params) == 1
    assert params[0].eccentricity == 0


@pytest.mark.parametrize('missing', ['eccentricity', 'mean_anomaly', 'both'])
def test_random_fallback_only_fills_missing_coordinates(tmp_path, missing):
    data = posterior()
    if missing in ('eccentricity', 'both'):
        data.pop('eccentricity')
    if missing in ('mean_anomaly', 'both'):
        data.pop('mean_anomaly')
    params = convert(tmp_path, normalize_samples(data), ['--add-eccentricity-params',
                                                       '--ecc-min', '0.01', '--ecc-max', '0.1'])
    e = np.array([p.eccentricity for p in params])
    anomaly = np.array([p.meanPerAno for p in params])
    if 'eccentricity' in data:
        np.testing.assert_allclose(e, data['eccentricity'])
    else:
        assert np.all((0.01 <= e) & (e <= 0.1))
    if 'mean_anomaly' in data:
        np.testing.assert_allclose(anomaly, data['mean_anomaly'])
    else:
        assert np.all((0 <= anomaly) & (anomaly <= 2*np.pi))


@pytest.mark.parametrize('bad', [-0.1, 1.0, np.nan, np.inf])
def test_invalid_eccentricity_is_not_replaced_by_random_fallback(bad):
    data = posterior()
    data['eccentricity'][0] = bad
    with pytest.raises(ValueError, match='eccentricity'):
        normalize_samples(data)


def test_conflicting_anomaly_aliases_are_rejected():
    data = posterior()
    data['meanPerAno'] = data['mean_anomaly'] + 0.1
    with pytest.raises(ValueError, match='Conflicting'):
        normalize_samples(data)


@pytest.mark.parametrize('kind', ['bilby-json', 'bilby-hdf5', 'pesummary-table'])
def test_real_asimov_conversion_uses_eccentric_samples_and_paths_with_spaces(tmp_path, monkeypatch, kind):
    pytest.importorskip('asimov')
    from RIFT.asimov.rift import Rift
    from RIFT import lalsimutils
    for key, value in environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    path = source_file(tmp_path, kind)
    stub = SimpleNamespace(production=SimpleNamespace(name='eccentric', meta={
        'dataset': posterior_label(path), 'likelihood': {'assume': ['eccentric', 'nonprecessing']},
        'sampler': {'force-ecc-min': 0.01, 'force-ecc-max': 0.1},
        'scheduler': {'bootstrap size': 3}, 'waveform': {'reference frequency': 20}}))
    output = tmp_path / 'bootstrap with spaces.xml.gz'
    Rift._convert_bootstrap_grid(stub, str(path), str(output), str(tmp_path / 'posterior with spaces.dat'))
    params = lalsimutils.xml_to_ChooseWaveformParams_array(str(output))
    np.testing.assert_allclose([p.eccentricity for p in params], posterior()['eccentricity'])
    np.testing.assert_allclose([p.meanPerAno for p in params], posterior()['mean_anomaly'])


def test_asimov_refuses_converter_failure(tmp_path, monkeypatch):
    pytest.importorskip('asimov')
    from RIFT.asimov.rift import Rift, PipelineException
    path = source_file(tmp_path, 'pesummary-table')
    stub = SimpleNamespace(production=SimpleNamespace(name='eccentric', meta={'dataset': 'bilby-eccentric'}))
    monkeypatch.setattr('shutil.which', lambda _: '/does/not/exist')
    with pytest.raises(PipelineException, match='bootstrap conversion failed'):
        Rift._convert_bootstrap_grid(stub, str(path), str(tmp_path / 'grid.xml.gz'), str(tmp_path / 'p.dat'))
    assert not (tmp_path / 'grid.xml.gz').exists()


@pytest.mark.parametrize('origin', ['column', 'metadata'])
@pytest.mark.parametrize('target', ['waveform', 'likelihood'])
def test_known_eccentric_reference_frequency_mismatch_is_rejected(tmp_path, monkeypatch, origin, target):
    pytest.importorskip('asimov')
    from RIFT.asimov.rift import Rift, PipelineException
    data = posterior()
    if origin == 'column':
        data['reference_frequency'] = np.full(3, 30.)
    path = source_file(tmp_path, 'bilby-json', data, reference=30 if origin == 'metadata' else None)
    settings = {'waveform': {'reference frequency': 20}} if target == 'waveform' else {
        'likelihood': {'start frequency': 20}}
    stub = SimpleNamespace(production=SimpleNamespace(name='eccentric', meta=settings))
    with pytest.raises(PipelineException, match='reference frequency'):
        Rift._convert_bootstrap_grid(stub, str(path), str(tmp_path / 'grid.xml.gz'), str(tmp_path / 'p.dat'))
    assert not (tmp_path / 'grid.xml.gz').exists()


def test_raw_bilby_bootstrap_survives_real_initial_grid_merge(tmp_path, monkeypatch):
    """The final product is a usable eccentric initial grid, not just an XML file."""
    import re
    pytest.importorskip('asimov')
    from RIFT.asimov.rift import Rift
    from RIFT import lalsimutils
    env = environment(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    path = source_file(tmp_path, 'bilby-json')
    seed = tmp_path / 'seed.xml.gz'
    stub = SimpleNamespace(production=SimpleNamespace(name='eccentric', meta={
        'likelihood': {'assume': ['eccentric', 'nonprecessing']},
        'sampler': {'extra eccentric arguments': {'force-ecc-min': 0.001, 'force-ecc-max': 0.4}},
        'scheduler': {'bootstrap size': 3}}))
    Rift._convert_bootstrap_grid(stub, str(path), str(seed), str(tmp_path / 'p.dat'))
    repo = CODE.parents[1]
    ref = repo / '.travis/ref_ini/GW150914.ini'
    text = ref.read_text().replace('[engine]', '[engine]\necc_min=0.001\necc_max=0.4')
    for flag in ('use_osg', 'use_osg_file_transfer', 'use_osg_cip'):
        text = text.replace(flag + '=True', flag + '=False')
    text = text.replace('internal-truncate-files-for-osg-file-transfer=True',
                        'internal-truncate-files-for-osg-file-transfer=False')
    text = text.replace('approx=IMRPhenomXPHM', 'approx=IMRPhenomD')
    text = text.replace('approx="IMRPhenomXPHM"', 'approx="IMRPhenomD"')
    text = text.replace('assume-precessing=True', 'assume-precessing=False')
    text = text.replace('internal-propose-converge-last-stage=True',
                        'internal-propose-converge-last-stage=False')
    text = text.replace('add-extrinsic=True', 'add-extrinsic=False')
    text = re.sub(r'force-initial-grid-size=\d+', 'force-initial-grid-size=4', text)
    ini = tmp_path / 'eccentric.ini'
    ini.write_text(text)
    cache = tmp_path / 'fake.cache'
    cache.write_text('')
    run = tmp_path / 'run'
    env.update(RIFT_LOWLATENCY='True', OMP_NUM_THREADS='1')
    result = subprocess.run([
        sys.executable, str(CODE / 'bin/util_RIFT_pseudo_pipe.py'),
        '--use-ini', str(ini), '--use-coinc', str(ref.with_name('coinc.xml')),
        '--use-rundir', str(run), '--fake-data-cache', str(cache),
        '--manual-initial-grid', str(seed), '--manual-initial-grid-supplements',
        '--assume-eccentric', '--use-meanPerAno', '--assume-nonprecessing',
        '--ile-no-gpu', '--force-ecc-min', '0.001', '--force-ecc-max', '0.4',
    ], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    (tmp_path / 'pipeline.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    params = lalsimutils.xml_to_ChooseWaveformParams_array(str(run / 'proposed-grid.xml.gz'))
    # Helper proposals can be removed by physical cuts; every source row must survive.
    assert len(params) >= len(posterior()['eccentricity'])
    for e, anomaly in zip(posterior()['eccentricity'], posterior()['mean_anomaly']):
        assert any(np.isclose(p.eccentricity, e, rtol=5e-7, atol=0)
                   and np.isclose(p.meanPerAno, anomaly, rtol=5e-7, atol=0) for p in params)
    dag = run / 'marginalize_intrinsic_parameters_BasicIterationWorkflow.dag'
    assert dag.is_file(), result.stdout[-6000:] + result.stderr[-6000:]
    assert '--parameter meanPerAno' in (run / 'args_cip_list.txt').read_text()



def test_asimov_uses_nested_eccentric_fallback_bounds(tmp_path, monkeypatch):
    pytest.importorskip('asimov')
    from RIFT.asimov.rift import Rift
    from RIFT import lalsimutils
    for key, value in environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    data = posterior()
    data.pop('eccentricity')
    path = source_file(tmp_path, 'pesummary-table', data)
    stub = SimpleNamespace(production=SimpleNamespace(name='eccentric', meta={
        'dataset': 'bilby-eccentric', 'likelihood': {'assume': ['eccentric']},
        'sampler': {'extra eccentric arguments': {'force-ecc-min': 0.2, 'force-ecc-max': 0.3}}}))
    output = tmp_path / 'nested/grid.xml.gz'
    Rift._convert_bootstrap_grid(stub, str(path), str(output), str(tmp_path / 'p.dat'))
    params = lalsimutils.xml_to_ChooseWaveformParams_array(str(output))
    assert all(0.2 <= p.eccentricity <= 0.3 for p in params)
    np.testing.assert_allclose([p.meanPerAno for p in params], data['mean_anomaly'])


def test_actual_build_bootstrap_block_converts_bilby_and_attaches_grid(tmp_path, monkeypatch):
    """Execute build_dag's own bootstrap wiring, including its final command args."""
    import ast
    pytest.importorskip('asimov')
    from RIFT.asimov.rift import Rift, PipelineException
    from RIFT import lalsimutils
    for key, value in environment(tmp_path).items():
        monkeypatch.setenv(key, value)
    path = source_file(tmp_path, 'bilby-json')
    meta = {'scheduler': {'bootstrap file': str(path)},
            'likelihood': {'assume': ['eccentric']}}
    stub = SimpleNamespace(production=SimpleNamespace(name='eccentric', meta=meta,
        event=SimpleNamespace(repository=SimpleNamespace(directory=str(tmp_path / 'repo')))),
        logger=SimpleNamespace(info=lambda *_: None), bootstrap=False)
    stub._find_posterior = lambda: str(path)
    stub._bootstrap_source_requested = lambda: Rift._bootstrap_source_requested(stub)
    stub._reuse_existing_bootstrap = lambda *_: False
    stub._convert_bootstrap_grid = lambda *args: Rift._convert_bootstrap_grid(stub, *args)
    tree = ast.parse((CODE / 'RIFT/asimov/rift.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Rift')
    build = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'build_dag')
    convert_block = next(n for n in build.body if isinstance(n, ast.If)
                         and ast.unparse(n.test) == 'self._bootstrap_source_requested()')
    attach_block = next(n for n in build.body if isinstance(n, ast.If)
                        and ast.unparse(n.test) == 'self.bootstrap')
    import tempfile
    namespace = dict(self=stub, os=os, np=np, Path=Path, tempfile=tempfile,
                     dryrun=False, bootstrap_coinc=False, command=[], PipelineException=PipelineException)
    exec(compile(ast.Module(body=[convert_block, attach_block], type_ignores=[]), '<build-bootstrap>', 'exec'), namespace)
    assert stub.bootstrap == 'manual'
    command = namespace['command']
    output = command[command.index('--manual-initial-grid') + 1]
    assert '--manual-initial-grid-supplements' in command
    params = lalsimutils.xml_to_ChooseWaveformParams_array(output)
    np.testing.assert_allclose([p.eccentricity for p in params], posterior()['eccentricity'])
    np.testing.assert_allclose([p.meanPerAno for p in params], posterior()['mean_anomaly'])



@pytest.mark.parametrize('add_missing', [False, True])
def test_bilby_no_spin_eccentric_posterior_needs_no_synthetic_spin_columns(tmp_path, add_missing):
    data = posterior()
    for name in ('a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'mean_anomaly'):
        data.pop(name)
    path = source_file(tmp_path, 'bilby-json', data)
    samples = dump_samples(path, None, tmp_path / 'posterior.dat')
    params = convert(tmp_path, samples, ['--add-eccentricity-params'] if add_missing else [])
    np.testing.assert_allclose([p.eccentricity for p in params], data['eccentricity'])
    assert all(p.s1x == p.s1y == p.s1z == p.s2x == p.s2y == p.s2z == 0 for p in params)
    np.testing.assert_allclose([p.incl for p in params], data['theta_jn'])
