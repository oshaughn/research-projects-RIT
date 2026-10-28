"""IMRPhenomPv2 modes must ignore prior contents of new LAL buffers (#488).

Both arms control every Python-created complex time-series allocation: the
reference starts at zero and the adversarial arm starts at NaN, infinity or a
large finite value. Real LALSimulation polarization generation is retained.
"""
import types

import lal
import lalsimulation as lalsim
import numpy as np
import pytest

from RIFT import lalsimutils as lsu


def _params(aligned=False):
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 35 * lal.MSUN_SI, 30 * lal.MSUN_SI
    P.s1x, P.s1z, P.s2z = (0.0 if aligned else 0.3), 0.1, -0.2
    P.fmin = P.fref = 30.0
    P.deltaT = 1.0 / 2048
    P.approx = lalsim.IMRPhenomPv2
    return P


def _modes(aligned=False):
    modes = lsu.hlmoft_IMRPv2_dict(_params(aligned))
    return {k: (np.array(v.data.data), (float(v.epoch), v.f0, v.deltaT, v.data.length))
            for k, v in modes.items()}


@pytest.mark.parametrize('aligned', [False, True])
@pytest.mark.parametrize('fill', [complex(np.nan, np.nan), complex(np.inf, -np.inf),
                                 complex(1e200, -1e200)], ids=['nan', 'inf', 'large-finite'])
def test_poisoned_buffers_give_clean_modes(monkeypatch, aligned, fill):
    calls = []

    def install_allocator(value):
        def controlled_create(*args):
            ts = lal.CreateCOMPLEX16TimeSeries(*args)
            ts.data.data = np.full(ts.data.length, value, dtype=complex)
            calls.append(ts.data.length)
            return ts
        proxy = types.SimpleNamespace(**vars(lal))
        proxy.CreateCOMPLEX16TimeSeries = controlled_create
        monkeypatch.setattr(lsu, 'lal', proxy)

    install_allocator(0.0)
    clean = _modes(aligned)
    assert calls and set(clean) == {(2, m) for m in range(-2, 3)}
    assert all(np.isfinite(values).all() for values, _ in clean.values())

    calls.clear()
    install_allocator(fill)
    poisoned = _modes(aligned)
    assert calls and set(poisoned) == set(clean)
    for key in clean:
        values, metadata = poisoned[key]
        assert metadata == clean[key][1]
        np.testing.assert_array_equal(values, clean[key][0], err_msg=str(key))
