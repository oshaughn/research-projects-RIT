from types import SimpleNamespace

import pytest

from RIFT.physics import teobresums_compat as compat


class _BasicModule:
    __file__ = "/tmp/basic/EOBRun_module.so"

    @staticmethod
    def EOBRunPy(parameters):
        return parameters


class _DaliModule(_BasicModule):
    eob_dyn_j0_py = object()
    eob_ham_s_py = object()
    eob_metric_A5PNlog_py = object()


def test_auto_profile_has_a_stable_default_for_unknown_extensions(monkeypatch):
    monkeypatch.delenv("RIFT_TEOBRESUMS_PROFILE", raising=False)

    assert compat.detect_profile(_BasicModule) == "default"
    assert compat.detect_profile(_DaliModule) == "dali"


def test_explicit_profile_override_and_typo_handling(monkeypatch):
    monkeypatch.setenv("RIFT_TEOBRESUMS_PROFILE", "legacy")
    assert compat.detect_profile(_DaliModule) == "legacy"

    monkeypatch.setenv("RIFT_TEOBRESUMS_PROFILE", "not-a-profile")
    with pytest.raises(compat.TEOBResumSCompatibilityError):
        compat.detect_profile(_DaliModule)


def test_legacy_integer_values_are_normalized_to_semantics():
    normalized = compat.normalize_parameters(
        {
            "arg_out": 1,
            "nqc": 2,
            "nqc_coefs_hlm": 0,
            "nqc_coefs_flx": 0,
            "use_geometric_units": 0,
            "interp_uniform_grid": 1,
            "output_hpc": 0,
            "M": 60,
        }
    )

    assert normalized == {
        "arg_out": "yes",
        "nqc": "no",
        "nqc_coefs_hlm": "none",
        "nqc_coefs_flx": "none",
        "use_geometric_units": "no",
        "interp_uniform_grid": "yes",
        "output_hpc": "no",
        "M": 60,
    }

    with pytest.raises(compat.TEOBResumSCompatibilityError):
        compat.normalize_parameters({"arg_out": 7})


def test_run_probes_before_native_call_and_passes_normalized_values(monkeypatch):
    compat._PROBED_SCHEMAS.clear()
    calls = []

    class RecordingModule(_BasicModule):
        @staticmethod
        def EOBRunPy(parameters):
            calls.append(parameters)
            return (None, None, None, {})

    monkeypatch.setattr(
        compat.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    result = compat.run(RecordingModule, {"arg_out": 1, "nqc": 2})

    assert len(result) == 4
    assert calls == [{"arg_out": "yes", "nqc": "no"}]


def test_failed_probe_prevents_native_call(monkeypatch):
    compat._PROBED_SCHEMAS.clear()
    calls = []

    class RecordingModule(_BasicModule):
        @staticmethod
        def EOBRunPy(parameters):
            calls.append(parameters)

    monkeypatch.setattr(
        compat.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=-11, stdout="", stderr="native extension terminated"
        ),
    )

    with pytest.raises(compat.TEOBResumSCompatibilityError, match="return code -11"):
        compat.run(RecordingModule, {"arg_out": "yes"})
    assert calls == []
