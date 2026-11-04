"""Unavailable v6 is a skip; actual numerical/backend failures stay failures."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'bin/util_CheckSEOBNRv6EHM.py'
spec = importlib.util.spec_from_file_location('v6_readiness', SCRIPT)
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


def test_provenance_does_not_require_git(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError('git is absent on this execute node')
    monkeypatch.setattr(readiness.subprocess, 'run', missing_git)
    report = readiness.provenance()
    assert report['rift_commit'] is None
    assert report['rift_tree'] is None
    assert report['rift_dirty'] is None


def test_missing_registry_entry_is_unavailable():
    def absent(name):
        assert name == 'SEOBNRv6EHM'
        raise ValueError('Approximant not implemented in GWSignal!')
    with pytest.raises(readiness.WaveformUnavailable):
        readiness.select_generator(absent)


def test_unexpected_factory_error_is_not_a_skip():
    def broken(name):
        raise ValueError('invalid backend configuration')
    with pytest.raises(ValueError, match='invalid backend configuration'):
        readiness.select_generator(broken)


@pytest.mark.parametrize('required', [False, True])
def test_cli_distinguishes_skip_from_required_availability(monkeypatch, tmp_path, required):
    def unavailable(options, report):
        raise readiness.WaveformUnavailable('v6 is not publicly available in this stack')
    monkeypatch.setattr(readiness, 'check', unavailable)
    monkeypatch.setattr(readiness, 'provenance', lambda: {})
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--output', str(output)]
                        + (['--require-available'] if required else []))
    assert readiness.main() == (1 if required else 0)
    assert json.loads(output.read_text())['status'] == ('failed' if required else 'skipped')


def test_numerical_failure_is_never_skipped(monkeypatch, tmp_path):
    def broken(options, report):
        raise RuntimeError('polarization mismatch')
    monkeypatch.setattr(readiness, 'check', broken)
    monkeypatch.setattr(readiness, 'provenance', lambda: {})
    output = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT), '--output', str(output)])
    assert readiness.main() == 1
    assert json.loads(output.read_text())['status'] == 'failed'


def test_public_seobnrv6ehm_waveform():
    # These optional waveform dependencies are not part of every CI environment.
    for dependency in ('lal', 'lalsimulation', 'astropy.units', 'gwpy.timeseries'):
        pytest.importorskip(dependency)
    pytest.importorskip('lalsimulation.gwsignal')
    report = {}
    try:
        readiness.check({'condition': 0}, report)
    except readiness.WaveformUnavailable as exc:
        pytest.skip('Public SEOBNRv6EHM backend unavailable: '+str(exc))
