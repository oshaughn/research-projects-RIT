"""A planner fault must not be readable as a conservative policy decline.

``multipeak_local_marginalize`` returns the caller's reserve for two unrelated
reasons.  One is a budget outcome: both tiers ran and a diagnostic failed.  The
other is a fault: something raised and the reserve is standing in for a step
that never ran.  Before the visibility change both looked the same in the log
(silence) and differed in the record only by a substring of ``provenance``, so
a ladder campaign in which every row declined by ``RuntimeError`` was read as a
conservative controller rather than as a defect.

These tests pin the separation itself, not the tier1 defect that exposed it:
the fault is reported and the budget decline is not, ``fail_on_fallback`` is
fatal on the first and inert on the second, the record carries a
machine-readable ``decline_kind``/``fault``, and the default path is
byte-for-byte what it was.

Two properties of the REPORTING are pinned here because a warning cannot supply
them.  The report must not be suppressible or made fatal by a process-global
filter the caller set for unrelated reasons -- ``-W error::RuntimeWarning``
would otherwise turn the default ``fail_on_fallback=False`` path into a raise --
and it must arrive once per CALL, not once per call site, because the campaign
case that motivated the change runs one call site with ``label=None`` and so
emits an identical message every time.
"""

import copy
import logging
import pickle
import warnings

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from RIFT.likelihood.jax_ile import multipeak_planner as planner  # noqa: E402


# The tuple contract MultiPeakResult has, and had before decline_kind/fault
# were added.  Defaulted trailing NamedTuple fields keep a 13-argument
# CONSTRUCTION working, but they do NOT keep 13-name UNPACKING working: a
# NamedTuple is a tuple, so a 15th field makes `a, ..., m = result` raise
# ValueError.  Unpacking is the contract callers hold, so decline_kind and
# fault are attributes and the tuple stays 13 elements.  Frozen here so an
# insertion in the middle -- which would silently reorder a caller's tuple --
# fails instead of passing.
_LEGACY_FIELDS = (
    "value", "accepted", "used_reserve", "provenance", "delta_log_integral",
    "tier0", "tier1", "tier0_portfolio", "tier1_portfolio",
    "total_lattice_evaluations", "total_refinement_steps",
    "total_local_evaluations", "modeled_peak_bytes",
)

# The two provenance strings the pre-change module produced.  Downstream
# analysis that greps them must keep working; the new record fields are an
# addition, not a replacement.
_PROVENANCE_ACCEPTED = "uvq-multipeak-tier1"
_PROVENANCE_BUDGET = "dense-reserve:enrichment-or-local-diagnostic"
_PROVENANCE_FAULT_PREFIX = "dense-reserve:planner-exception:"


def _synthetic_tables(n_time=9):
    """Small reflected-polynomial problem with an interior four-axis peak."""
    time = np.arange(n_time, dtype=float)
    C_A = np.zeros((3, 3, n_time), dtype=np.complex128)
    C_B = np.zeros((5, 5), dtype=np.complex128)
    C_A[0, 1] = 20.0 - 2.0 * np.cos(2.0 * np.pi * time / (n_time - 1))
    C_A[2, 0] = 0.25
    C_A[2, 2] = 0.25
    C_B[0, 2] = 4.0
    C_B[2, 1] = 0.02
    C_B[2, 3] = 0.02
    return C_A, C_B


# Settings that accept the local branch, and settings whose only difference is
# an unreachable agreement budget, so the same tables decline by budget.  Both
# rows run both tiers to completion; neither raises.
_ACCEPT_KWARGS = dict(
    log_integral_tol=0.1, tier0=(2, 2, 24), tier1=(3, 3, 48),
    quadrature_order=7, cell_sigma=4.0, chunk_size=32)
_BUDGET_KWARGS = dict(
    log_integral_tol=1.0e-12, tier0=(2, 2, 24), tier1=(3, 3, 48),
    quadrature_order=5, cell_sigma=4.0, chunk_size=32)


# Captured once, before any test patches it.
_REAL_RUN_STRUCTURAL_TIER = planner._run_structural_tier


def _fault_logs(caplog):
    """The planner's own WARNING records, ignoring anything else logging."""
    return [r for r in caplog.records
            if r.name == planner.__name__ and r.levelno >= logging.WARNING]


class _CountingReserve(object):
    """A finite reserve that records whether the planner actually paid for it."""

    def __init__(self, value=123.456):
        self.value = float(value)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.value


def _fault_at(monkeypatch, which, message="deliberate tier fault"):
    """Make the ``which``-th structural tier raise, and the others run for real.

    ``which=2`` reproduces the campaign's shape exactly: tier0 converges, tier1
    raises, and the row falls back.  The failure is injected at the tier seam
    rather than by feeding bad tables, so tier0's report is real and the stage
    the planner names can be checked against a known answer.
    """
    state = {"n": 0}

    def wrapper(*args, **kwargs):
        state["n"] += 1
        if state["n"] == int(which):
            raise RuntimeError(message)
        # Always the pristine function, so patching twice in one test does not
        # stack wrappers and fire on the wrong call.
        return _REAL_RUN_STRUCTURAL_TIER(*args, **kwargs)

    monkeypatch.setattr(planner, "_run_structural_tier", wrapper)
    return state


def test_budget_decline_and_accepted_row_stay_silent(caplog):
    C_A, C_B = _synthetic_tables()
    caplog.set_level(logging.DEBUG, logger=planner.__name__)

    # A budget decline is a normal outcome.  It must not report, or a campaign
    # that declines legitimately drowns the faults it is supposed to surface.
    reserve = _CountingReserve()
    caplog.clear()
    declined = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, reserve, **_BUDGET_KWARGS)
    assert declined.used_reserve and not declined.accepted
    assert declined.decline_kind == planner.DECLINE_DIAGNOSTIC
    assert _fault_logs(caplog) == []

    # An accepted row must not report either.
    caplog.clear()
    accepted = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, _CountingReserve(), **_ACCEPT_KWARGS)
    assert accepted.accepted and accepted.decline_kind is None
    assert _fault_logs(caplog) == []


def test_fault_log_names_stage_exception_and_label(monkeypatch, caplog):
    C_A, C_B = _synthetic_tables()
    caplog.set_level(logging.DEBUG, logger=planner.__name__)
    _fault_at(monkeypatch, 2, message="tier1 refinement is degenerate")
    reserve = _CountingReserve()
    caplog.clear()
    result = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, reserve, label="ladder-row-7", **_ACCEPT_KWARGS)
    records = _fault_logs(caplog)

    # Exactly one record per CALL, not per Newton step: a 40-row campaign gets
    # 40 lines, which is readable; per-step would not be.
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    text = records[0].getMessage()
    assert "tier1" in text
    assert "RuntimeError" in text
    assert "tier1 refinement is degenerate" in text
    assert "ladder-row-7" in text
    assert "fault" in text.lower()
    assert result.used_reserve and result.value == reserve.value


def test_record_separates_fault_from_budget_without_parsing_provenance(
        monkeypatch):
    C_A, C_B = _synthetic_tables()

    budget = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, _CountingReserve(), **_BUDGET_KWARGS)
    _fault_at(monkeypatch, 2, message="tier1 refinement is degenerate")
    fault = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, _CountingReserve(), **_ACCEPT_KWARGS)

    # Both used the reserve; only the second is a defect.  The distinction is a
    # field comparison, not a substring search on provenance.
    assert budget.used_reserve and fault.used_reserve
    assert budget.decline_kind == planner.DECLINE_DIAGNOSTIC
    assert fault.decline_kind == planner.DECLINE_FAULT
    assert planner.DECLINE_DIAGNOSTIC != planner.DECLINE_FAULT
    assert budget.fault is None
    assert isinstance(fault.fault, planner.FallbackFault)
    assert fault.fault.stage == "tier1"
    assert fault.fault.error_type == "RuntimeError"
    assert fault.fault.message == "tier1 refinement is degenerate"


def test_fault_stage_names_the_step_that_raised(monkeypatch):
    """The stage is measured, not assumed: tier0 and tier1 report differently."""
    C_A, C_B = _synthetic_tables()
    _fault_at(monkeypatch, 1)
    first = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, _CountingReserve(), **_ACCEPT_KWARGS)
    _fault_at(monkeypatch, 2)
    second = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, _CountingReserve(), **_ACCEPT_KWARGS)
    assert first.fault.stage == "tier0"
    assert second.fault.stage == "tier1"

    # A fault before either tier runs is attributed to its own stage, so a bad
    # table is never reported as a tier defect.
    repeated = np.repeat(C_B[..., None], 7, axis=-1)
    repeated[1, 1, 3] += 1.0e-3
    early = planner.multipeak_local_marginalize(
        C_A, repeated, 1.0, 8.0, _CountingReserve(), **_ACCEPT_KWARGS)
    assert early.decline_kind == planner.DECLINE_FAULT
    assert early.fault.stage == "uv-summary"
    assert early.fault.error_type == "ValueError"


def test_fail_on_fallback_raises_on_fault_and_is_inert_on_budget_decline(
        monkeypatch):
    C_A, C_B = _synthetic_tables()

    # Inert on the budget decline: same value, same record, no exception.
    reserve = _CountingReserve()
    declined = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, reserve, fail_on_fallback=True, **_BUDGET_KWARGS)
    assert declined.used_reserve and declined.value == reserve.value
    assert declined.decline_kind == planner.DECLINE_DIAGNOSTIC
    assert reserve.calls == 1

    # Inert on an accepted row.
    accepted = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, _CountingReserve(), fail_on_fallback=True,
        **_ACCEPT_KWARGS)
    assert accepted.accepted and accepted.provenance == _PROVENANCE_ACCEPTED

    # Fatal on the fault, and it does not pay for the reserve first: the point
    # is to stop, not to produce a value nobody should trust.
    _fault_at(monkeypatch, 2, message="tier1 refinement is degenerate")
    fatal_reserve = _CountingReserve()
    with pytest.raises(planner.MultiPeakFallbackError) as excinfo:
        planner.multipeak_local_marginalize(
            C_A, C_B, 1.0, 8.0, fatal_reserve, fail_on_fallback=True,
            label="ladder-row-7", **_ACCEPT_KWARGS)
    assert fatal_reserve.calls == 0
    assert "tier1" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)

    # MultiPeakFallbackError sits outside the set the planner catches, so a
    # nested or repeated call cannot swallow it back into a decline.
    assert not isinstance(
        excinfo.value, (RuntimeError, ValueError, np.linalg.LinAlgError))


def test_default_path_is_unchanged(monkeypatch):
    """Same values, same provenance, same legacy record shape, no exception."""
    C_A, C_B = _synthetic_tables()
    assert planner.MultiPeakResult._fields == _LEGACY_FIELDS

    # A caller built on the pre-change arity still constructs the record, and
    # the record it gets back is still exactly that many elements.
    legacy = planner.MultiPeakResult(*range(len(_LEGACY_FIELDS)))
    assert len(legacy) == len(_LEGACY_FIELDS)
    assert tuple(legacy) == tuple(range(len(_LEGACY_FIELDS)))
    assert legacy.decline_kind is None and legacy.fault is None

    accepted = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, _CountingReserve(), **_ACCEPT_KWARGS)
    assert accepted.accepted and not accepted.used_reserve
    assert accepted.provenance == _PROVENANCE_ACCEPTED
    assert np.isfinite(accepted.value)

    reserve = _CountingReserve()
    budget = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, reserve, **_BUDGET_KWARGS)
    assert not budget.accepted and budget.used_reserve
    assert budget.value == reserve.value
    assert budget.provenance == _PROVENANCE_BUDGET
    assert reserve.calls == 1

    _fault_at(monkeypatch, 2)
    fault_reserve = _CountingReserve(77.25)
    fault = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, fault_reserve, **_ACCEPT_KWARGS)
    # Default is off, so the fault still RETURNS the reserve exactly as before.
    assert not fault.accepted and fault.used_reserve
    assert fault.value == 77.25
    assert fault.provenance == _PROVENANCE_FAULT_PREFIX + "RuntimeError"
    assert not np.isfinite(fault.delta_log_integral)
    assert fault_reserve.calls == 1

    # A failing reserve still surfaces as _DenseReserveError, not as a planner
    # decline and not as the new fallback error.
    def failing_reserve():
        raise ValueError("deliberate reserve failure")

    with pytest.raises(planner._DenseReserveError):
        planner.multipeak_local_marginalize(
            C_A, C_B, 1.0, 8.0, failing_reserve, **_BUDGET_KWARGS)


def test_legacy_thirteen_name_unpacking_still_works():
    """The contract a caller holds is UNPACKING, and it is 13 names wide.

    Defaulted trailing NamedTuple fields keep 13-argument CONSTRUCTION working,
    so a test that only constructs cannot see this break.  A NamedTuple is a
    tuple: two appended fields make ``len()`` 15 and every existing
    ``a, ..., m = result`` raise ``ValueError: too many values to unpack``.
    That is why decline_kind and fault are attributes and not tuple elements.
    """
    record = planner.MultiPeakResult(*range(13))

    # The failure this pins is a ValueError from the unpacking statement
    # itself; there is no way to write it that a construction test also covers.
    (value, accepted, used_reserve, provenance, delta_log_integral,
     tier0, tier1, tier0_portfolio, tier1_portfolio,
     total_lattice_evaluations, total_refinement_steps,
     total_local_evaluations, modeled_peak_bytes) = record
    assert (value, modeled_peak_bytes) == (0, 12)

    # Everything else that reads the tuple as a sequence agrees on 13.
    assert len(record) == 13
    assert len(tuple(record)) == 13
    assert len(list(record)) == 13
    assert len(planner.MultiPeakResult._fields) == 13
    assert len(record._asdict()) == 13
    assert record[-1] == 12
    assert record + () == tuple(range(13))

    # And a real result, not only a hand-built one.
    C_A, C_B = _synthetic_tables()
    live = planner.multipeak_local_marginalize(
        C_A, C_B, 1.0, 8.0, _CountingReserve(), **_ACCEPT_KWARGS)
    assert len(live) == 13
    unpacked_value = list(live)[0]
    assert unpacked_value == live.value


def test_annotations_are_attributes_and_survive_replace_copy_pickle():
    """They are reachable, immutable, and not smuggled into the tuple."""
    fault = planner.FallbackFault("tier1", "RuntimeError", "degenerate")
    record = planner.MultiPeakResult(
        *range(13), decline_kind=planner.DECLINE_FAULT, fault=fault)

    assert record.decline_kind == planner.DECLINE_FAULT
    assert record.fault is fault
    assert tuple(record) == tuple(range(13))
    assert planner.DECLINE_FAULT not in tuple(record)
    assert fault not in tuple(record)
    assert "decline_kind" not in record._fields
    assert "fault" not in record._fields

    # Immutable like the tuple part, so a consumer cannot annotate a record
    # after the fact and have it read as the planner's own finding.
    with pytest.raises(AttributeError):
        record.decline_kind = planner.DECLINE_DIAGNOSTIC
    with pytest.raises(AttributeError):
        del record.fault

    # _replace, _make, copy and pickle all bypass __new__ in some way; each
    # must still carry the annotation, or a round-tripped fault reads as clean.
    replaced = record._replace(value=99.0)
    assert replaced.value == 99.0
    assert replaced.decline_kind == planner.DECLINE_FAULT
    assert replaced.fault == fault
    assert len(replaced) == 13
    assert record._replace(decline_kind=None).decline_kind is None
    with pytest.raises(ValueError):
        record._replace(no_such_field=1)

    remade = planner.MultiPeakResult._make(range(13), fault=fault)
    assert remade.fault == fault and len(remade) == 13

    assert copy.copy(record).fault == fault
    assert copy.deepcopy(record).decline_kind == planner.DECLINE_FAULT
    assert pickle.loads(pickle.dumps(record)).fault == fault

    # A record built the plain way answers None rather than raising, whichever
    # construction route produced it.
    assert planner.MultiPeakResult._make(range(13)).decline_kind is None
    assert repr(record).count("decline_kind") == 1


@pytest.mark.parametrize("filter_action", ["default", "always", "error"])
def test_fault_reporting_does_not_depend_on_warning_filters(
        monkeypatch, caplog, filter_action):
    """The default path stays non-fatal, and the fault stays observable.

    ``-W error::RuntimeWarning`` is a filter a caller sets for unrelated
    reasons.  While the fault was announced with ``warnings.warn`` it raised at
    the warn call, BEFORE the ``fail_on_fallback`` check and before the reserve
    was evaluated, so that filter alone turned the default path fatal.
    """
    C_A, C_B = _synthetic_tables()
    caplog.set_level(logging.DEBUG, logger=planner.__name__)
    _fault_at(monkeypatch, 2, message="tier1 refinement is degenerate")
    reserve = _CountingReserve(77.25)
    caplog.clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter(filter_action, RuntimeWarning)
        # No pytest.raises: the point is that this RETURNS under every filter.
        result = planner.multipeak_local_marginalize(
            C_A, C_B, 1.0, 8.0, reserve, label="ladder-row-7",
            **_ACCEPT_KWARGS)

    assert result.value == 77.25
    assert result.used_reserve and not result.accepted
    assert reserve.calls == 1

    # Observable in the record, which is the primary channel precisely because
    # no filter reaches it...
    assert result.decline_kind == planner.DECLINE_FAULT
    assert result.fault.stage == "tier1"
    assert result.fault.error_type == "RuntimeError"

    # ...and on the log, which is the secondary one.
    records = _fault_logs(caplog)
    assert len(records) == 1
    assert "ladder-row-7" in records[0].getMessage()

    # The module must not route this through the warnings machinery at all:
    # under "always" a warn would show up here, and under "error" it would have
    # raised above instead of returning.
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_fail_on_fallback_is_the_only_thing_that_makes_a_fault_fatal(
        monkeypatch):
    """Under warnings-as-errors, both settings keep their documented meaning."""
    C_A, C_B = _synthetic_tables()

    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("error")

        _fault_at(monkeypatch, 2)
        quiet_reserve = _CountingReserve(5.5)
        result = planner.multipeak_local_marginalize(
            C_A, C_B, 1.0, 8.0, quiet_reserve, **_ACCEPT_KWARGS)
        assert result.value == 5.5 and quiet_reserve.calls == 1

        _fault_at(monkeypatch, 2)
        fatal_reserve = _CountingReserve()
        with pytest.raises(planner.MultiPeakFallbackError):
            planner.multipeak_local_marginalize(
                C_A, C_B, 1.0, 8.0, fatal_reserve, fail_on_fallback=True,
                **_ACCEPT_KWARGS)
        # Still the fallback error, not a RuntimeWarning promoted to an
        # exception, and still without paying for the reserve.
        assert fatal_reserve.calls == 0


def test_identical_faults_from_one_call_site_report_once_each(
        monkeypatch, caplog):
    """Not once in total.

    ``warnings.warn`` de-duplicates on (message, category, module, lineno)
    under the default filters.  The campaign this change exists for calls one
    call site with ``label=None``, so every message is identical and the
    warning would be shown for the first row only -- exactly the silence the
    change is meant to remove.
    """
    C_A, C_B = _synthetic_tables()
    caplog.set_level(logging.DEBUG, logger=planner.__name__)
    n_calls = 3
    caplog.clear()

    with warnings.catch_warnings():
        warnings.resetwarnings()   # the DEFAULT filters, where dedup applies
        for _ in range(n_calls):
            _fault_at(monkeypatch, 2)
            result = planner.multipeak_local_marginalize(
                C_A, C_B, 1.0, 8.0, _CountingReserve(), label=None,
                **_ACCEPT_KWARGS)
            assert result.decline_kind == planner.DECLINE_FAULT

    records = _fault_logs(caplog)
    assert len(records) == n_calls
    assert len({r.getMessage() for r in records}) == 1   # identical text
