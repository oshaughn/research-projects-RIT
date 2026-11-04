"""run_fixture and main end to end, with FEW and the response replaced by fakes.

The real backends are tested only when their environment variables are set,
so CI would otherwise never execute the fixture -> waveform -> response ->
likelihood -> JSON path.  The fakes return a distance-scaled sinusoid, which
gives a closed-form likelihood for every trial.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import RIFT.LISA.emri_direct_cli as cli
import RIFT.LISA.emri_response as emri_response
import RIFT.LISA.few_adapter as few_adapter

N, DT, TARGET_SNR = 256, 5.0, 20.0


class FakeParameters(SimpleNamespace):
    @classmethod
    def from_dict(cls, values):
        if set(values) != {'dist', 'qS', 'phiS'}:
            raise ValueError('fake parameters require dist, qS, phiS')
        return cls(**values)

    def to_dict(self):
        return dict(vars(self))


class FakeAdapter:
    calls = 0
    epoch_by_dist = {}

    def __init__(self, config):
        self.config = config

    def generate(self, params):
        FakeAdapter.calls += 1
        t = np.arange(N) * DT
        hp = np.cos(2*np.pi*t*7/(N*DT)) / params.dist
        hc = np.sin(2*np.pi*t*11/(N*DT)) / params.dist
        epoch = FakeAdapter.epoch_by_dist.get(params.dist, 30.)
        out = SimpleNamespace(polarizations=(hp, hc), dt_seconds=DT,
                              epoch_start_seconds=epoch, metadata={'fake': True})
        RESPONSE_CALLS.append({'expect': (params.qS, params.phiS, epoch, DT)})
        return out


RESPONSE_CALLS = []


def fake_aet(hp, hc, *, dt, sky_colatitude, sky_longitude, epoch, config):
    RESPONSE_CALLS[-1]['got'] = (sky_colatitude, sky_longitude, epoch, dt)
    # Sky-dependent gain, so a sky change reaches the likelihood.
    gain = 1 + 0.1*np.cos(sky_colatitude) + 0.05*np.sin(sky_longitude)
    return SimpleNamespace(channels=gain*np.asarray([hp, hc, hp + hc]), dt=dt,
                           epoch_seconds=epoch, metadata={'fake': True})


@pytest.fixture
def fake_backends(monkeypatch):
    FakeAdapter.calls = 0
    FakeAdapter.epoch_by_dist = {}
    RESPONSE_CALLS.clear()
    monkeypatch.setattr(few_adapter, 'FEWConfig', SimpleNamespace(
        from_dict=lambda d: SimpleNamespace(epoch_reference=d['epoch_reference'],
                                            n_samples=d['n_samples'])))
    monkeypatch.setattr(few_adapter, 'EMRIParameters', FakeParameters)
    monkeypatch.setattr(few_adapter, 'FewWaveformAdapter', FakeAdapter)
    monkeypatch.setattr(emri_response, 'ResponseConfig',
                        SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(emri_response, 'equal_arm_aet', fake_aet)
    monkeypatch.setattr(cli.importlib.metadata, 'version', lambda name: 'fake')


def fixture(**overrides):
    config = {
        'schema_version': 1,
        'waveform': {'epoch_reference': 'pinned equal-arm orbit t=0', 'n_samples': N},
        'response': {},
        'injection': {'dist': 1.0, 'qS': 1.0, 'phiS': 2.0},
        'trials': [{'id': 'inj', 'changes': {}},
                   {'id': 'far', 'changes': {'dist': 2.0}},
                   {'id': 'near', 'changes': {'dist': 0.8}}],
        'noise': {'kind': 'synthetic_independent_white', 'psd_levels': [1., 2., 3.],
                  'target_snr': TARGET_SNR, 'fmin_hz': 0., 'fmax_hz': 0.1},
    }
    config.update(overrides)
    return config


def test_main_writes_closed_form_distance_likelihoods(fake_backends, tmp_path):
    source = tmp_path / 'fixture.json'
    source.write_text(json.dumps(fixture()))
    output = tmp_path / 'out.json'
    assert cli.main(['--fixture', str(source), '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['status'] == 'COMPLETED'
    assert result['snr_squared'] == pytest.approx(TARGET_SNR**2)
    # A pure distance change gives lnL = -rho^2 (1 - d_inj/d)^2 / 2.
    expected = {'inj': 0., 'far': -.5*TARGET_SNR**2*(1 - 1/2.)**2,
                'near': -.5*TARGET_SNR**2*(1 - 1/.8)**2}
    got = {r['id']: r['log_likelihood'] for r in result['trials']}
    assert got == pytest.approx(expected)
    assert FakeAdapter.calls == 4
    assert all(call['got'] == call['expect'] for call in RESPONSE_CALLS)


def test_each_trial_reaches_the_response_with_its_own_sky(fake_backends):
    trials = [{'id': 'sky', 'changes': {'qS': 1.3, 'phiS': 2.4}}]
    result = cli.run_fixture(fixture(trials=trials))
    assert [call['got'] for call in RESPONSE_CALLS] == [call['expect'] for call in RESPONSE_CALLS]
    assert RESPONSE_CALLS[-1]['got'][:2] == (1.3, 2.4)
    gain = lambda colat, lon: 1 + 0.1*np.cos(colat) + 0.05*np.sin(lon)
    ratio = gain(1.3, 2.4) / gain(1.0, 2.0)
    assert result['trials'][0]['log_likelihood'] == pytest.approx(
        -.5*TARGET_SNR**2*(1 - ratio)**2)


def test_trial_with_different_response_epoch_is_refused(fake_backends):
    FakeAdapter.epoch_by_dist = {2.0: 40.}
    with pytest.raises(ValueError, match='sampling/epoch differs'):
        cli.run_fixture(fixture())


def test_main_refuses_existing_output_or_missing_directory_before_running(
        fake_backends, tmp_path):
    source = tmp_path / 'fixture.json'
    source.write_text(json.dumps(fixture()))
    existing = tmp_path / 'out.json'
    existing.write_text('keep')
    for target in (existing, tmp_path / 'missing' / 'out.json'):
        with pytest.raises(SystemExit) as exc:
            cli.main(['--fixture', str(source), '--output', str(target)])
        assert exc.value.code == 2
    assert FakeAdapter.calls == 0
    assert existing.read_text() == 'keep'


def test_main_leaves_no_file_when_serialization_fails(monkeypatch, tmp_path):
    source = tmp_path / 'fixture.json'
    source.write_text('{}')
    monkeypatch.setattr(cli, 'run_fixture', lambda config, max_points: {'x': np.float32(1)})
    output = tmp_path / 'out.json'
    with pytest.raises(TypeError):
        cli.main(['--fixture', str(source), '--output', str(output)])
    assert not output.exists()


@pytest.mark.parametrize('overrides, message', [
    ({'trials': []}, 'trial count'),
    ({'trials': [{'id': 'a', 'changes': {}}] * 2}, 'unique'),
    ({'trials': [{'id': 'a', 'changes': {'mass': 1.}}]}, 'unknown trial parameters'),
    ({'noise': {'kind': 'synthetic_independent_white', 'psd_levels': [1., 1., 1.],
                'target_snr': 5.}}, 'complete synthetic white noise'),
    ({'waveform': {'epoch_reference': 'other', 'n_samples': N}}, 'epoch_reference'),
    ({'waveform': {'epoch_reference': 'pinned equal-arm orbit t=0',
                   'n_samples': cli.MAX_WAVEFORM_SAMPLES + 1}}, 'limited to'),
    # Literal pin: a cap raised to "no cap" must fail this case.
    ({'waveform': {'epoch_reference': 'pinned equal-arm orbit t=0', 'n_samples': 32769}},
     'limited to 32768 waveform samples'),
])
def test_run_fixture_refuses_before_generating(fake_backends, overrides, message):
    with pytest.raises(ValueError, match=message):
        cli.run_fixture(fixture(**overrides))
    assert FakeAdapter.calls == 0


def test_run_fixture_accepts_the_largest_permitted_sample_count(fake_backends):
    # The dt=5 s Paper 5 fixture needs 26216 samples; the cap must admit it.
    assert cli.MAX_WAVEFORM_SAMPLES >= 26216
    config = fixture(waveform={'epoch_reference': 'pinned equal-arm orbit t=0',
                               'n_samples': cli.MAX_WAVEFORM_SAMPLES})
    assert cli.run_fixture(config)['status'] == 'COMPLETED'
