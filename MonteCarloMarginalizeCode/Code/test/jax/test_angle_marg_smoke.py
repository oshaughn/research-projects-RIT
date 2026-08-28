"""CHEAP smoke / mutation-bearing coverage of the angle-marginalization feature.

Exists because the full validation suite (test_angle_marg_exact.py) is too
expensive for a per-PR gate -- it exceeds 20 minutes on 2 cores and has killed
the CI runner three different ways -- but excluding it wholesale would leave the
ENTIRE production feature ungated: a mutation that returns a wrong marginal,
ignores --angle-marg-scheme, breaks the Laplace root enumeration, or drops the
suspect label would all be green.

So this file covers that surface deliberately at MINIMAL scale.  It is not a
substitute for the validation suite's error-law measurements; it is the
mutation-bearing floor that must stay in CI.  Keep it cheap: every test here
must run in seconds, or it belongs in the excluded file instead.
"""
import ast
import pathlib

import numpy as np
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM


def test_scheme_selector_returns_both_schemes_and_reports():
    """auto must be able to choose EITHER scheme.  A regression here is not
    hypothetical: a previous head floored the amplitude before passing it to the
    selector, so `auto` could never return 'exact' at all."""
    lo, _ = AM.choose_angle_marg_scheme(AM.ANGLE_MARG_CROSSOVER_AMPLITUDE / 10.0)
    hi, _ = AM.choose_angle_marg_scheme(AM.ANGLE_MARG_CROSSOVER_AMPLITUDE * 10.0)
    assert lo == "exact", "sub-crossover amplitude must select the exact scheme"
    assert hi == "laplace", "high amplitude must select the laplace scheme"
    assert lo != hi


def test_dense_sizes_grow_with_amplitude_and_mode_content():
    """Both levers must be live: sqrt(A) AND m_max.  Either being ignored is a
    real bug that shipped once."""
    n_lo = AM._dense_grid_sizes(50.0, m_max=2)
    n_hi = AM._dense_grid_sizes(5000.0, m_max=2)
    n_m = AM._dense_grid_sizes(50.0, m_max=8)
    assert n_hi[0] > n_lo[0] and n_hi[1] > n_lo[1], "must grow with amplitude"
    assert n_m[0] > n_lo[0], "phi sizing must grow with m_max"


def test_amp_sizing_is_required_not_defaulted():
    """A silently-defaulted amplitude is how the SNR-guess bug got in."""
    try:
        AM._require_amp_sizing(None)
    except Exception as exc:
        assert "amp_sizing" in str(exc)
    else:
        raise AssertionError("a missing amp_sizing must raise, not default")


def test_failsafe_record_roundtrips_and_barriers():
    """The host record must reset, report, and barrier -- without it the driver
    cannot label an artifact and the condition dies with the log line."""
    import inspect
    AM.reset_amp_failsafe()
    st = AM.amp_failsafe_state()
    assert st["tripped"] is False
    for fn in (AM.amp_failsafe_state, AM.reset_amp_failsafe):
        assert "effects_barrier" in inspect.getsource(fn)
    src = inspect.getsource(AM._runtime_amp_failsafe)
    assert src.find("lax.cond") < src.find("debug.callback"), (
        "the callback must sit inside lax.cond; an unconditional callback fires "
        "on every likelihood evaluation and destroys throughput")


def _driver_src():
    return (pathlib.Path(__file__).resolve().parents[2] / "bin"
            / "integrate_likelihood_extrinsic_jax").read_text()


def test_driver_actually_passes_the_scheme_through():
    """AST guard on the VALUE node.  Checking only that some `angle_marg=`
    keyword is present is foolable by hardcoding angle_marg="grid" -- flag
    parsed, help present, print present, feature inert.  That is this repo's
    documented silent-no-op pattern."""
    tree = ast.parse(_driver_src())
    seen = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for k in node.keywords or []:
                if k.arg == "angle_marg":
                    seen.append(k.value)
    assert seen, "the driver must pass angle_marg to the wrapper"
    assert any(isinstance(v, ast.Name) and v.id == "angle_marg" for v in seen), (
        "angle_marg must be forwarded as the parsed option, not a constant")


def test_driver_labels_both_artifacts_and_never_implies_verification():
    src = _driver_src()
    assert "def angle_grid_suspect_note(" in src
    wd = src[src.index("def write_dat("):]
    wd = wd[:wd.index("\ndef ")]
    assert "angle_grid_suspect_note(" in wd, (
        "the EVIDENCE row must be labelled independently of sample export -- "
        "write_samples early-returns without --save-samples")
    assert "BEST-EFFORT" in src, (
        "artifacts must state that no-detection is NOT verification: the "
        "detector is a droppable jax callback, so silence cannot be read as "
        "an adequate grid")
    assert "SUSPECT-ANGLE-GRID" in src
