"""Mode conditioning must work with native lengths, padding and truncation."""
import numpy as np
import pytest

lal = pytest.importorskip('lal')
lalsim = pytest.importorskip('lalsimulation')
pytest.importorskip('gwpy.timeseries')
from gwpy.timeseries import TimeSeries
from RIFT import lalsimutils as lsu
from RIFT.physics import GWSignal as gws


@pytest.mark.parametrize('deltaF', [None, 1., 8.])
@pytest.mark.parametrize('align', [None, 'peak'])
@pytest.mark.parametrize('fmin', [20., 0.1])
def test_condition_short_modes(monkeypatch, deltaF, align, fmin):
    P = lsu.ChooseWaveformParams()
    P.deltaT, P.deltaF = 1./1024, deltaF
    P.fmin = fmin
    P.taper = lalsim.SIM_INSPIRAL_TAPER_START
    generator = object()
    monkeypatch.setattr(gws.gws.models, 'gwsignal_get_waveform_generator',
                        lambda _: generator)
    def modes(parameters, selected):
        assert selected is generator
        return {(2, m): TimeSeries(np.ones(256, dtype=complex),
                                  dt=P.deltaT, t0=-0.25) for m in (-2, 2)}
    monkeypatch.setattr(gws.wfm, 'GenerateTDModes', modes)
    result = gws.hlmoft(P, approx_string='SEOBNRv6EHM',
                       no_trust_align_method=align, internal_phase_shift=0.)
    expected = 256 if deltaF is None else int(1./deltaF/P.deltaT)
    for h in result.values():
        assert h.data.length == expected
        assert np.isfinite(h.data.data).all()
        assert h.data.data[0] == 0
        if expected > 256:
            assert np.all(h.data.data[256:] == 0)
        if align == 'peak':
            assert float(h.epoch) == pytest.approx(-np.argmax(abs(h.data.data))*P.deltaT)


def test_zero_padding_does_not_change_taper(monkeypatch):
    P = lsu.ChooseWaveformParams()
    P.deltaT, P.deltaF, P.fmin = 1./1024, None, 100.
    P.taper = lalsim.SIM_INSPIRAL_TAPER_START
    monkeypatch.setattr(gws.gws.models, 'gwsignal_get_waveform_generator', lambda _: object())
    monkeypatch.setattr(gws.wfm, 'GenerateTDModes', lambda *args:
                        {(2, 2): TimeSeries(np.ones(256, dtype=complex), dt=P.deltaT, t0=-.25)})
    native = gws.hlmoft(P, approx_string='SEOBNRv6EHM')[(2, 2)]
    P.deltaF = 1./16
    padded = gws.hlmoft(P, approx_string='SEOBNRv6EHM')[(2, 2)]
    np.testing.assert_array_equal(native.data.data, padded.data.data[:256])
