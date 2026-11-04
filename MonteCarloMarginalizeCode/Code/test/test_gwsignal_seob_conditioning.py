"""Shared SEOB mode conditioning must commute with arbitrary projection."""
import numpy as np
import pytest

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")
pytest.importorskip("lalsimulation.gwsignal")
from RIFT import lalsimutils as lsu
from RIFT.physics import GWSignal


def parameters():
    p = lsu.ChooseWaveformParams()
    p.m1, p.m2 = 60 * lal.MSUN_SI, 30 * lal.MSUN_SI
    p.fmin = p.fref = 20.
    p.deltaT = 1. / 2048
    p.s1z, p.s2z = .45, -.3
    return p


def series(values, p):
    h = lal.CreateCOMPLEX16TimeSeries("mode", lal.LIGOTimeGPS(-1.23456789),
                                    0., p.deltaT, lal.DimensionlessUnit, len(values))
    h.data.data[:] = values
    return h


def test_native_operator_is_linear_on_one_grid_including_zero_modes():
    p = parameters()
    t = np.arange(8192) * p.deltaT
    a = np.exp(1j * 2*np.pi*(18*t + 8*t*t))
    b = np.exp(-1j * 2*np.pi*(32*t + 3*t*t))
    a[-10:] = b[-10:] = 0.
    conditioning = GWSignal._seob_native_conditioning_parameters(p)
    common_length = len(t) - 10
    conditioned = [GWSignal._condition_seob_mode(series(v, p), p, conditioning,
                                                common_length)
                   for v in (a, b, np.zeros_like(a))]
    assert all(h.data.length == common_length for h in conditioned)
    assert all(h.epoch == lal.LIGOTimeGPS(-1.23456789) for h in conditioned)
    assert all(h.deltaT == p.deltaT for h in conditioned)
    assert np.count_nonzero(conditioned[2].data.data) == 0
    for alpha, beta in [(1.2+.3j, -.2+.9j), (.1-1.3j, 2.1+.4j)]:
        projected = series(alpha*a + beta*b, p)
        expected = GWSignal._condition_seob_mode(projected, p, conditioning,
                                                common_length)
        actual = alpha*conditioned[0].data.data + beta*conditioned[1].data.data
        np.testing.assert_allclose(actual, expected.data.data, atol=1e-12, rtol=1e-12)


def test_native_operator_matches_lal_stage1_and_stage2_common_trim():
    p = parameters()
    t = np.arange(8192) * p.deltaT
    values = np.exp(1j * 2*np.pi*(18*t + 8*t*t))
    values[-10:] = 0.
    conditioning = GWSignal._seob_native_conditioning_parameters(p)
    expected = [lal.CreateREAL8TimeSeries("component", lal.LIGOTimeGPS(-1.23456789),
                0., p.deltaT, lal.DimensionlessUnit, len(t)) for _ in range(2)]
    expected[0].data.data[:] = values.real
    expected[1].data.data[:] = values.imag
    lalsim.SimInspiralTDConditionStage1(*expected, conditioning[1], p.fmin)
    lalsim.SimInspiralTDConditionStage2(*expected, conditioning[2], conditioning[3])
    actual = GWSignal._condition_seob_mode(series(values, p), p, conditioning, len(t)-10)
    assert actual.data.length == expected[0].data.length
    assert actual.epoch == expected[0].epoch
    np.testing.assert_array_equal(actual.data.data,
                                  expected[0].data.data + 1j*expected[1].data.data)


def test_zero_reference_is_not_reinterpreted_at_lowered_start():
    p = parameters()
    p.fref = 0.
    with pytest.raises(ValueError, match="positive fref"):
        GWSignal._seob_native_conditioning_parameters(p)


def test_mode_selection_cannot_change_conditioning_grid(monkeypatch):
    from gwpy.timeseries import TimeSeries
    p = parameters()
    p.deltaF = None
    p.taper = lalsim.SIM_INSPIRAL_TAPER_START
    t = np.arange(8192) * p.deltaT
    h22 = np.exp(1j * 2*np.pi*(18*t + 8*t*t))
    h33 = np.exp(1j * 2*np.pi*(32*t + 3*t*t))
    h22[-20:] = 0.
    h33[-10:] = 0.
    raw = {(2, 2): h22, (2, -2): h22.conj(),
           (2, 1): np.zeros_like(h22), (3, 3): h33}
    modes = {k: TimeSeries(v, t0=-1.23456789, dt=p.deltaT)
             for k, v in raw.items()}
    monkeypatch.setattr(GWSignal.gws.models, "gwsignal_get_waveform_generator",
                        lambda _: object())
    monkeypatch.setattr(GWSignal.wfm, "GenerateTDModes", lambda *args: modes)
    options = dict(approx_string="SEOBNRv5PHM", gwsignal_mode_conditioning="native")
    full = GWSignal.hlmoft(p, Lmax=3, **options)
    low = GWSignal.hlmoft(p, Lmax=2, **options)
    restricted = GWSignal.hlmoft(p, Lmax=3, force_22_mode=True, **options)
    assert set(restricted) == {(2, -2), (2, 2)}
    for output in (full, low, restricted):
        assert all(h.data.length == len(t)-10 for h in output.values())
        assert all(h.epoch == full[(2, 2)].epoch for h in output.values())
        np.testing.assert_array_equal(output[(2, 2)].data.data,
                                      full[(2, 2)].data.data)
    assert np.count_nonzero(full[(2, 1)].data.data) == 0


@pytest.mark.parametrize("option", ["bad", None, True])
def test_invalid_mode_conditioning_policy_is_refused(option):
    with pytest.raises(ValueError, match="legacy or native"):
        GWSignal.hlmoft(parameters(), approx_string="SEOBNRv5PHM",
                        gwsignal_mode_conditioning=option)


def test_native_no_condition_conflict_is_refused():
    with pytest.raises(ValueError, match="conflicts with no_condition"):
        GWSignal.hlmoft(parameters(), approx_string="SEOBNRv5PHM",
                        gwsignal_mode_conditioning="native", no_condition=True)
