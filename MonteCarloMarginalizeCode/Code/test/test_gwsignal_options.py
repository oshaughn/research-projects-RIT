"""Generic backend options at the real GWSignal generator boundary.

Only the backend factory and generation function are replaced. All three RIFT
entry points execute their real parameter construction and safety guards. The
sentinel stops before waveform postprocessing; these are configuration tests,
not numerical waveform validation (which also needs the optional backend).
"""
from copy import deepcopy

import pytest

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")
pytest.importorskip("astropy.units")
pytest.importorskip("gwpy.timeseries")
pytest.importorskip("lalsimulation.gwsignal")

from RIFT import lalsimutils as lsu
from RIFT.physics import GWSignal


class GeneratorReached(Exception):
    pass


def waveform_params():
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 60 * lal.MSUN_SI, 40 * lal.MSUN_SI
    P.s1x, P.s1y, P.s1z = 0.4, 0.2, 0.3
    P.s2x, P.s2y, P.s2z = -0.2, 0.1, -0.1
    P.dist = 500e6 * lal.PC_SI
    P.fmin = P.fref = 30.0
    P.deltaT, P.deltaF = 1.0 / 2048, None
    P.taper = lalsim.SIM_INSPIRAL_TAPER_NONE
    P.approx = lalsim.IMRPhenomXPHM
    return P


def capture_generator(monkeypatch, callback):
    generator = object()
    monkeypatch.setattr(GWSignal.gws.models, "gwsignal_get_waveform_generator",
                        lambda approx: generator)

    def capture(parameters, selected):
        assert selected is generator
        callback(parameters)
        raise GeneratorReached

    monkeypatch.setattr(GWSignal.wfm, "GenerateTDModes", capture)
    monkeypatch.setattr(GWSignal.gws.core.waveform, "GenerateTDWaveform", capture)


@pytest.mark.parametrize("entry", ["hlmoft", "hoft", "complex_hoft"])
def test_generic_options_reach_every_generator(monkeypatch, entry):
    requested = {
        "antisymmetric_modes_hm": True,
        "enable_antisymmetric_modes": True,
        "lmax_nyquist": 0,
        "future_false_option": False,
        "future_zero_option": 0,
        "future_nested_option": {"modes": [(2, 2), (3, 3)], "flags": {"x": False}},
        # Explicit backend dictionary must bypass the RIFT control filter.
        "fd_L_frame": "future backend meaning",
        "force_22_mode": "backend meaning",
        "internal_phase_shift": "backend meaning",
        "condition": 0,
    }
    original = deepcopy(requested)

    def check(parameters):
        for key, value in requested.items():
            assert parameters[key] == value
        assert parameters["legacy_future_option"] == {"x": 1}
        for control in ("e_freq", "fd_alignment_postevent_time", "no_condition",
                        "rom_taper_start", "extra_waveform_args"):
            assert control not in parameters
        assert "gwsignal_extra_kwargs" not in parameters
        assert parameters["mass1"].value == pytest.approx(60)
        assert parameters["deltaT"].value == pytest.approx(1.0 / 2048)
        parameters["future_nested_option"]["flags"]["x"] = True

    capture_generator(monkeypatch, check)
    with pytest.raises(GeneratorReached):
        getattr(GWSignal, entry)(
            waveform_params(), approx_string="SEOBNRv5PHM",
            gwsignal_extra_kwargs=requested, lmax_nyquist=4,
            legacy_future_option={"x": 1}, force_22_mode=True,
            fd_L_frame=True, fd_alignment_postevent_time=2, e_freq=0,
            no_condition=True, rom_taper_start=True, extra_waveform_args={})
    assert requested == original, "backend mutation leaked into later waveform calls"


@pytest.mark.parametrize("entry", ["hlmoft", "hoft", "complex_hoft"])
def test_legacy_options_remain_generic(monkeypatch, entry):
    requested = {"enable_antisymmetric_modes": False, "lmax_nyquist": 0,
                 "future_option": {"anything": [1, 2]}}
    def check(parameters):
        for key, value in requested.items():
            assert parameters[key] == value

    capture_generator(monkeypatch, check)
    with pytest.raises(GeneratorReached):
        getattr(GWSignal, entry)(waveform_params(), approx_string="SEOBNRv5PHM",
                               **requested)


@pytest.mark.parametrize("entry", ["hlmoft", "hoft", "complex_hoft"])
@pytest.mark.parametrize("field", ["mass1", "mass2", "spin1x", "spin1y", "spin1z",
                                    "spin2x", "spin2y", "spin2z", "deltaT",
                                    "f22_start", "f22_ref", "distance", "inclination",
                                    "phi_ref", "eccentricity", "longAscNodes", "meanPerAno"])
@pytest.mark.parametrize("explicit", [True, False])
def test_source_parameters_cannot_silently_disagree(monkeypatch, entry, field, explicit):
    capture_generator(monkeypatch, lambda _: pytest.fail("conflict reached backend"))
    options = {field: 123}
    if explicit:
        options = {"gwsignal_extra_kwargs": options}
    with pytest.raises(ValueError, match="ChooseWaveformParams.*" + field):
        getattr(GWSignal, entry)(waveform_params(), approx_string="SEOBNRv5PHM", **options)


@pytest.mark.parametrize("invalid", [None, [], "{'x': True}", 1])
def test_explicit_options_must_be_mapping(invalid):
    with pytest.raises(TypeError, match="must be a mapping"):
        GWSignal.hlmoft(waveform_params(), gwsignal_extra_kwargs=invalid)


def test_backend_keys_must_be_strings():
    with pytest.raises(TypeError, match="names must be strings"):
        GWSignal.hlmoft(waveform_params(), gwsignal_extra_kwargs={1: True})


@pytest.mark.parametrize("entry", ["hlmoft", "hoft", "complex_hoft"])
def test_backend_rejection_is_not_suppressed(monkeypatch, entry):
    generator = object()
    monkeypatch.setattr(GWSignal.gws.models, "gwsignal_get_waveform_generator",
                        lambda approx: generator)

    def reject(parameters, selected):
        assert parameters["unsupported_new_option"] is True
        raise ValueError("backend rejects unsupported_new_option")

    monkeypatch.setattr(GWSignal.wfm, "GenerateTDModes", reject)
    monkeypatch.setattr(GWSignal.gws.core.waveform, "GenerateTDWaveform", reject)
    with pytest.raises(ValueError, match="backend rejects unsupported_new_option"):
        getattr(GWSignal, entry)(waveform_params(), approx_string="SEOBNRv5PHM",
                               gwsignal_extra_kwargs={"unsupported_new_option": True})


def test_opaque_backend_objects_do_not_need_copy_support(monkeypatch):
    class OpaqueBackendOption:
        def __deepcopy__(self, memo):
            raise TypeError("opaque backend object cannot be copied")

    opaque = OpaqueBackendOption()

    def check(parameters):
        assert parameters["future_opaque_option"] is opaque
        assert parameters["future_nested_option"]["opaque"] is opaque

    capture_generator(monkeypatch, check)
    with pytest.raises(GeneratorReached):
        GWSignal.hlmoft(waveform_params(), approx_string="SEOBNRv5PHM",
                        gwsignal_extra_kwargs={"future_opaque_option": opaque,
                                              "future_nested_option": {"opaque": opaque}})


@pytest.mark.parametrize("entry", ["hlmoft", "hoft", "complex_hoft"])
def test_lal_dictionary_options_cannot_silently_disappear(monkeypatch, entry):
    capture_generator(monkeypatch, lambda _: pytest.fail("LAL options reached backend"))
    with pytest.raises(ValueError, match="LAL-dictionary extra_waveform_args"):
        getattr(GWSignal, entry)(waveform_params(), approx_string="SEOBNRv5PHM",
                               extra_waveform_args={"dQuadMon1": 0.25})


@pytest.mark.parametrize("entry", ["hlmoft", "hoft", "complex_hoft"])
def test_explicit_backend_can_use_name_of_lal_control(monkeypatch, entry):
    requested = {"extra_waveform_args": {"future_backend_option": False}}

    def check(parameters):
        assert parameters["extra_waveform_args"] == requested["extra_waveform_args"]

    capture_generator(monkeypatch, check)
    with pytest.raises(GeneratorReached):
        getattr(GWSignal, entry)(waveform_params(), approx_string="SEOBNRv5PHM",
                               extra_waveform_args={}, gwsignal_extra_kwargs=requested)


@pytest.mark.parametrize("entry", ["hlmoft", "hoft", "complex_hoft"])
@pytest.mark.parametrize("explicit", [True, False])
def test_condition_can_override_backend_without_mutating_source(monkeypatch, entry, explicit):
    p = waveform_params()
    p.taper = lalsim.SIM_INSPIRAL_TAPER_START

    def check(parameters):
        assert parameters["condition"] == 0

    capture_generator(monkeypatch, check)
    options = {"condition": 0}
    if explicit:
        options = {"gwsignal_extra_kwargs": options}
    with pytest.raises(GeneratorReached):
        getattr(GWSignal, entry)(p, approx_string="SEOBNRv5PHM", **options)
    assert p.taper == lalsim.SIM_INSPIRAL_TAPER_START
    assert p.m1 == 60 * lal.MSUN_SI
    assert p.deltaT == 1.0 / 2048
