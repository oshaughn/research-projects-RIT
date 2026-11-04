"""Buffer-initialization mechanics for optional EFPE, with generator stubs.

These tests exercise the real EFPE packing/reconstruction code and real LAL
allocations. The optional pyEFPE generator is replaced explicitly; this is not
physical EFPE waveform validation.
"""
import types

import lal
import numpy as np
import pytest

from RIFT import lalsimutils as lsu
from RIFT.physics import EFPE as efpe


def _params():
    p = lsu.ChooseWaveformParams()
    p.m1 = p.m2 = 50 * lal.MSUN_SI
    p.fmin = p.fref = 16.
    p.deltaT, p.deltaF = 1./128, 1.
    return p


def _allocator(name, value, calls):
    original = getattr(lal, name)
    def create(*args):
        series = original(*args)
        series.data.data = np.full(series.data.length, value, dtype=complex)
        calls.append(series.data.length)
        return series
    proxy = types.SimpleNamespace(**vars(lal))
    setattr(proxy, name, create)
    return proxy


@pytest.mark.parametrize('cutoff', [False, True])
@pytest.mark.parametrize('fill', [complex(np.nan, np.nan), complex(np.inf, -np.inf)],
                         ids=['nan', 'inf'])
def test_efpe_fd_padding_ignores_allocator_contents(monkeypatch, cutoff, fill):
    class StubWaveform:
        def __init__(self, params):
            pass
        def generate_waveform(self, frequencies):
            return 1 + frequencies/128 + 0.2j, 2 - frequencies/256 - 0.3j
    monkeypatch.setattr(efpe, 'pyEFPE', types.SimpleNamespace(pyEFPE=StubWaveform), raising=False)
    monkeypatch.setattr(efpe, 'debug', False)
    outputs = []
    for value in (0.0, fill):
        calls = []
        monkeypatch.setattr(efpe, 'lal', _allocator('CreateCOMPLEX16FrequencySeries', value, calls))
        hp, hc = efpe.get_polarizations(_params(), return_in_FD=True, use_ISCO_cutoff=cutoff)
        assert len(calls) == 2
        outputs.append([np.array(hp.data.data), np.array(hc.data.data)])
    for clean, poisoned in zip(*outputs):
        assert np.isfinite(clean).all() and np.isfinite(poisoned).all()
        np.testing.assert_array_equal(poisoned, clean)
        assert np.all(clean[:16] == 0)
        if cutoff:
            assert np.all(clean[44:] == 0)


@pytest.mark.parametrize('fill', [complex(np.nan, np.nan), complex(np.inf, -np.inf)],
                         ids=['nan', 'inf'])
def test_efpe_mode_reconstruction_ignores_allocator_contents(monkeypatch, fill):
    def stub_complex_hoft(p, **kwargs):
        series = lal.CreateCOMPLEX16TimeSeries('stub', -0.5, 0., p.deltaT, lal.DimensionlessUnit, 16)
        series.data.data = np.arange(16) * (1 + 0.2j) * np.cos(p.incl) + np.sin(p.incl)
        return series
    monkeypatch.setattr(efpe, 'complex_hoft', stub_complex_hoft)
    outputs = []
    for value in (0.0, fill):
        calls = []
        monkeypatch.setattr(efpe, 'lal', _allocator('CreateCOMPLEX16TimeSeries', value, calls))
        modes = efpe.hlmoft(_params())
        assert len(calls) == 5
        outputs.append({key: np.array(series.data.data) for key, series in modes.items()})
    assert set(outputs[0]) == set(outputs[1]) == {(2, m) for m in range(-2, 3)}
    for key in outputs[0]:
        assert np.isfinite(outputs[0][key]).all() and np.isfinite(outputs[1][key]).all()
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
