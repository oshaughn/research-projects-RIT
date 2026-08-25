from types import SimpleNamespace
from pathlib import Path

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


def _transverse_parameters(spin1x=0.0, spin1y=0.0, spin2x=0.0, spin2y=0.0):
    return {
        "spin1x": spin1x,
        "spin1y": spin1y,
        "spin2x": spin2x,
        "spin2y": spin2y,
        "untouched": object(),
    }


def test_resums_initial_grid_seed_has_margin_above_native_boundary():
    seed_min, seed_max = compat.initial_transverse_spin_range("TEOBResumSDALI")

    assert (seed_min, seed_max) == (1e-3, 3e-3)
    assert seed_min >= 10 * compat.DALI_TRANSVERSE_SPIN_THRESHOLD
    assert compat.initial_transverse_spin_range("SEOBNRv5PHM") == (1e-5, 3e-5)


@pytest.mark.parametrize("approximant", ["TEOBResumS", "TEOBResumSDALI"])
def test_gwsignal_guard_zeros_only_resums_native_aligned_interval(approximant):
    original = _transverse_parameters(spin1x=6e-5, spin2y=4e-5)

    guarded = compat.guard_gwsignal_transverse_spins(original, approximant)

    assert guarded is not original
    assert [guarded[key] for key in ("spin1x", "spin1y", "spin2x", "spin2y")] == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert original["spin1x"] == 6e-5  # caller-owned parameters are not mutated
    assert guarded["untouched"] is original["untouched"]


def test_gwsignal_guard_preserves_aligned_genuinely_precessing_and_other_models():
    aligned = _transverse_parameters()
    precessing = _transverse_parameters(spin1x=1.000001e-4)
    other_model = _transverse_parameters(spin1x=1e-5)

    assert compat.guard_gwsignal_transverse_spins(aligned, "TEOBResumSDALI") is aligned
    assert compat.guard_gwsignal_transverse_spins(precessing, "TEOBResumSDALI") is precessing
    assert compat.guard_gwsignal_transverse_spins(other_model, "SEOBNRv5PHM") is other_model


def test_pipeline_threads_approximant_into_model_specific_grid_seed():
    code_root = Path(__file__).parents[1]
    helper_source = (code_root / "bin" / "helper_LDG_Events.py").read_text()
    pipe_source = (code_root / "bin" / "util_RIFT_pseudo_pipe.py").read_text()

    assert "initial_transverse_spin_range(" in helper_source
    assert "--internal-initial-grid-approximant {}" in pipe_source


def test_precession_predicate_uses_summed_transverse_magnitude():
    threshold = compat.DALI_TRANSVERSE_SPIN_THRESHOLD

    # a component that is nonzero but summed-small is still aligned for DALI
    assert not compat.is_precessing_for_resums(0.0, 1e-5, 2e-5, 3e-5)
    assert not compat.is_precessing_for_resums(0.0, 0.0, 0.0, 0.0)
    # two components that are individually below threshold can sum above it
    assert compat.is_precessing_for_resums(0.9 * threshold, 0.0, 0.9 * threshold, 0.0)
    assert compat.is_precessing_for_resums(0.0, 0.0, 0.0, 1.000001 * threshold)


def test_direct_teobresums_calls_share_the_backend_precession_predicate():
    source = (Path(__file__).parents[1] / "RIFT" / "lalsimutils.py").read_text()

    # four direct EOBRunPy parameter blocks, plus the complementary decision
    # about conjugating positive-m modes
    assert source.count("is_precessing_for_resums(") == 5
    assert "P.s1y!=0.0" not in source


def test_retained_hyperbolic_peaks_are_kept_in_chronological_order():
    source = (Path(__file__).parents[1] / "RIFT" / "lalsimutils.py").read_text()

    # indices_to_keep is a set: filtered_peaks[-1] is only the last peak if sorted
    assert "peaks[list(indices_to_keep)]" not in source
    assert source.count("peaks[sorted(indices_to_keep)]") == 4


def test_inference2ile_restores_posterior_columns_it_can_read():
    source = (
        Path(__file__).parents[1] / "bin" / "convert_output_format_inference2ile"
    ).read_text()

    # eccentricity must be restored from the input columns, not from the
    # freshly initialized (always zero) parameter value
    assert 'if "eccentricity" in samples_in.dtype.names:' in source
    assert "P.eccentricity < 1e-5" not in source
    assert 'if "a6c" in samples_in.dtype.names:' in source


def test_every_rift_gwsignal_generator_path_uses_transverse_spin_guard():
    code_root = Path(__file__).parents[1]
    source = (code_root / "RIFT" / "physics" / "GWSignal.py").read_text()

    assert source.count("gwsignal_get_waveform_generator(") == 3
    assert source.count("guard_gwsignal_transverse_spins(") == 3
