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
