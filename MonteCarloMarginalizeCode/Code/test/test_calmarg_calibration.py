import pytest

from RIFT.calmarg.calibration import correction_type_for_ifo


@pytest.mark.parametrize(
    "ifo_name, expected",
    [("H1", "data"), ("L1", "data"), ("K1", "data"), ("V1", "template")],
)
def test_bilby_pipe_default_correction_types(ifo_name, expected):
    assert correction_type_for_ifo(None, ifo_name) == expected


@pytest.mark.parametrize("setting", ["data", "template"])
def test_global_correction_type(setting):
    assert correction_type_for_ifo(setting, "V1") == setting


def test_detector_specific_correction_types():
    setting = {"H1": "template", "V1": "data"}
    assert correction_type_for_ifo(setting, "H1") == "template"
    assert correction_type_for_ifo(setting, "V1") == "data"


def test_string_detector_specific_correction_types():
    def parse_dict(value):
        assert value == "{H1: data, V1: template}"
        return {"H1": "data", "V1": "template"}

    assert correction_type_for_ifo(
        "{H1: data, V1: template}", "V1", parse_dict=parse_dict
    ) == "template"


def test_missing_detector_is_rejected():
    with pytest.raises(ValueError, match="No calibration correction type"):
        correction_type_for_ifo({"H1": "data"}, "V1")
