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
        "kwargs": {"Lmax": 4, "fd_L_frame": True, "no_condition": True},
    }
    assert options == {"fd_L_frame": True, "no_condition": True}


def _calmarg_waveform(monkeypatch, extra_waveform_kwargs):
    """Run the real calmarg source function; return its h+ and the modes hlmoft built."""
    import numpy as np
    real = rift_source.lalsimutils.hlmoft
    seen = {}

    def spy(P, **kwargs):
        seen["P"] = P.manual_copy()
        seen["kwargs"] = dict(kwargs)
        seen["hlm"] = real(P, **kwargs)
        return seen["hlm"]

    monkeypatch.setattr(rift_source.lalsimutils, "hlmoft", spy)
    freqs = np.arange(0, 1024.0 + 0.125, 0.125)
    h = rift_source.RIFT_lal_binary_black_hole(
        freqs, 36.0, 29.0, 400.0, 0.4, 0.0, 0.2, 0.0, -0.3, 0.0, 0.4, 0.3,
        waveform_approximant="IMRPhenomXPHM", reference_frequency=20.0,
        minimum_frequency=20.0, Lmax=2, h_method="hlmoft",
        extra_waveform_kwargs=extra_waveform_kwargs)
    monkeypatch.undo()   # else a second call would wrap this spy and overwrite `seen`
    return h, seen


def test_calmarg_waveform_honors_typed_kwargs_like_ile(monkeypatch):
    """XPHM fd_centering_factor changes the calmarg waveform, and the modes equal ILE's call form."""
    import numpy as np
    kw = {"fd_L_frame": True, "fd_alignment_postevent_time": None, "fd_centering_factor": 0.75}
    h_kw, seen = _calmarg_waveform(monkeypatch, kw)
    h_def, _ = _calmarg_waveform(monkeypatch, {"fd_L_frame": True})
    scale = np.max(np.abs(h_def["plus"]))
    assert np.max(np.abs(h_kw["plus"] - h_def["plus"])) > 1e-3 * scale
    # ILE builds modes as lalsimutils.hlmoft(P, Lmax, **extra_waveform_kwargs)
    ile_hlm = rift_source.lalsimutils.hlmoft(seen["P"], Lmax=2, **kw)
    for mode, series in seen["hlm"].items():
        np.testing.assert_array_equal(series.data.data, ile_hlm[mode].data.data)
        assert float(series.epoch) == float(ile_hlm[mode].epoch)
