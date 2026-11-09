import numpy as np
import pytest

from RIFT.calmarg import rift_source


def test_hlmoft_expands_extra_waveform_kwargs(monkeypatch):
    captured = {}

    def fake_hlmoft(P, **kwargs):
        captured["P"] = P
        captured["kwargs"] = kwargs
        return "modes"

    monkeypatch.setattr(rift_source.lalsimutils, "hlmoft", fake_hlmoft)
    P = object()
    options = {"fd_L_frame": True, "no_condition": True}

    result = rift_source._hlmoft_with_extra_waveform_kwargs(P, 4, options)

    assert result == "modes"
    assert captured == {
        "P": P,
        "kwargs": {"Lmax": 4, "fd_L_frame": True, "no_condition": True,
                   "fd_standoff_factor": 0.9},
    }
    assert options == {"fd_L_frame": True, "no_condition": True}


class _Stop(Exception):
    pass


def _calmarg_modes(monkeypatch, extra_waveform_kwargs):
    """Run the real calmarg source function up to its hlmoft call; return what hlmoft built.

    Stops right after hlmoft, so bilby (needed later, for utils.nfft) is not required.
    """
    real = rift_source.lalsimutils.hlmoft
    seen = {}

    def spy(P, **kwargs):
        seen["P"] = P.manual_copy()
        seen["kwargs"] = dict(kwargs)
        seen["hlm"] = real(P, **kwargs)
        raise _Stop

    monkeypatch.setattr(rift_source.lalsimutils, "hlmoft", spy)
    freqs = np.arange(0, 1024.0 + 0.125, 0.125)
    try:
        rift_source.RIFT_lal_binary_black_hole(
            freqs, 36.0, 29.0, 400.0, 0.4, 0.0, 0.2, 0.0, -0.3, 0.0, 0.4, 0.3,
            waveform_approximant="IMRPhenomXPHM", reference_frequency=20.0,
            minimum_frequency=20.0, Lmax=2, h_method="hlmoft",
            extra_waveform_kwargs=extra_waveform_kwargs)
    except _Stop:
        pass
    monkeypatch.undo()   # else a second call would wrap this spy
    assert "hlm" in seen, "hlmoft was not called"
    return seen


def test_calmarg_waveform_honors_typed_kwargs_like_ile(monkeypatch):
    """XPHM fd_centering_factor changes the calmarg modes, and they equal ILE's call form."""
    kw = {"fd_L_frame": True, "fd_alignment_postevent_time": None, "fd_centering_factor": 0.75}
    seen = _calmarg_modes(monkeypatch, kw)
    seen_def = _calmarg_modes(monkeypatch, {"fd_L_frame": True})
    a, b = seen["hlm"][(2, 2)], seen_def["hlm"][(2, 2)]
    scale = np.max(np.abs(b.data.data))
    assert (float(a.epoch) != float(b.epoch)
            or np.max(np.abs(a.data.data - b.data.data)) > 1e-3 * scale)
    # ILE builds modes as lalsimutils.hlmoft(P, Lmax, **extra_waveform_kwargs), its lalsuite
    # route having added fd_standoff_factor=0.9
    ile_hlm = rift_source.lalsimutils.hlmoft(seen["P"], Lmax=2, fd_standoff_factor=0.9, **kw)
    for mode, series in seen["hlm"].items():
        np.testing.assert_array_equal(series.data.data, ile_hlm[mode].data.data)
        assert float(series.epoch) == float(ile_hlm[mode].epoch)


def _bilby_tidal_fixture(source_model):
    from bilby.gw.waveform_generator import WaveformGenerator

    generator = WaveformGenerator(
        duration=32, sampling_frequency=4096, frequency_domain_source_model=source_model,
        waveform_arguments=dict(waveform_approximant="IMRPhenomD_NRTidalv2", h_method="hlmoft",
                                minimum_frequency=40., reference_frequency=40., Lmax=2))
    parameters = dict(mass_1=1.4, mass_2=1.3, luminosity_distance=100., iota=.6, phase=.2,
                      spin_1x=0., spin_1y=0., spin_1z=0., spin_2x=0., spin_2y=0., spin_2z=0.)
    return generator, parameters


def test_bilby_circular_source_uses_each_component_tide():
    tidal, parameters = _bilby_tidal_fixture(rift_source.RIFT_lal_tidal_binary)
    bbh, _ = _bilby_tidal_fixture(rift_source.RIFT_lal_binary_black_hole)
    zero = tidal.frequency_domain_strain(dict(parameters, lambda_1=0., lambda_2=0.))["plus"]
    np.testing.assert_array_equal(zero, bbh.frequency_domain_strain(parameters)["plus"])
    zero_norm = np.linalg.norm(zero)
    assert zero_norm > 0.

    for lambda_1, lambda_2 in ((500., 0.), (0., 800.), (500., 800.)):
        plus = tidal.frequency_domain_strain(
            dict(parameters, lambda_1=lambda_1, lambda_2=lambda_2))["plus"]
        assert np.all(np.isfinite(plus))
        assert np.linalg.norm(plus - zero) > 1e-6 * zero_norm


def test_bilby_eccentric_source_passes_tides_and_orbital_parameters(monkeypatch):
    generator, parameters = _bilby_tidal_fixture(rift_source.RIFT_lal_eccentric_tidal_binary)
    captured = []

    def capture(P, **kwargs):
        captured.append((P.lambda1, P.lambda2, P.eccentricity, P.meanPerAno))
        raise _Stop

    monkeypatch.setattr(rift_source.lalsimutils, "hlmoft", capture)
    with pytest.raises(_Stop):
        generator.frequency_domain_strain(
            dict(parameters, lambda_1=0., lambda_2=800., eccentricity=.1, mean_per_ano=.3))
    assert captured == [(0., 800., .1, .3)]
