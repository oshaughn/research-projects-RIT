#!/usr/bin/env python
"""
Drift gate for the LISA ILE driver.

The two ILE drivers are a DELIBERATE fork:

    bin/integrate_likelihood_extrinsic_batchmode        <- main, moves fast
    bin/integrate_likelihood_extrinsic_batchmode_lisa   <- LISA, lags

RO, 2026-08-13: "It is super annoying we have to have two of them, but the overhead of one
ring to rule them all is too high."  So this gate does NOT try to close the gap, and does
not assert that any particular item was ported.  Closing the gap is not the goal.

What it asserts is that nothing drifts in UNNOTICED: every helper, CLI option, module
constant and sampler provenance marker present in the main driver and absent from the LISA
one carries a recorded decision -- PORT / PORTED / NA / PHYSICS -- with a reason.  "Does not
apply to LISA" is a fine answer; silence is not.

When this fails, the fix is to classify the new item, not to delete the test:

    cd test/expensive_before_merging/integrators
    python3 audit_lisa_driver_drift.py --undecided   # what is unclassified
    $EDITOR make_lisa_drift_ledger.py                # add a rule, with a reason
    python3 make_lisa_drift_ledger.py                # regenerate the ledger

This exists because 2,357 lines of drift accumulated while the LISA driver's nine CI tests
(all import/contract/smoke level) stayed green.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUDIT_DIR = os.path.join(_HERE, 'expensive_before_merging', 'integrators')

if _AUDIT_DIR not in sys.path:
    sys.path.insert(0, _AUDIT_DIR)

audit = pytest.importorskip("audit_lisa_driver_drift",
                            reason="LISA drift auditor not present")


@pytest.fixture(scope="module")
def state():
    gap, extras = audit.compute_gap()
    ledger = audit.load_ledger()
    return audit.annotate(gap, ledger), extras, ledger


def test_the_gap_is_non_empty_so_the_audit_is_actually_looking(state):
    """Guard against a silently broken extractor reporting a clean tree."""
    gap, _extras, _ledger = state
    assert len(gap) > 0, "the audit found no drift at all, which almost certainly means " \
                         "the extractor broke rather than that the drivers converged"


def test_every_gap_item_carries_a_recorded_decision(state):
    gap, _extras, _ledger = state
    undecided = [g for g in gap if g["decision"] is None]
    assert not undecided, (
        "%d item(s) drifted into the main ILE driver with no recorded decision about the "
        "LISA driver:\n%s\n\nClassify each as PORT / PORTED / NA / PHYSICS with a reason "
        "in make_lisa_drift_ledger.py, then regenerate the ledger."
        % (len(undecided), "\n".join("  %s (main:%d)" % (g["key"], g["main_line"])
                                     for g in undecided)))


def test_no_item_claims_to_be_ported_while_still_missing(state):
    """A PORTED verdict is a claim about the tree, so the tree gets to contradict it.

    This is the regression direction: if a ported helper is later deleted from the LISA
    driver, the item reappears in the gap still marked PORTED, and this fails.
    """
    gap, _extras, _ledger = state
    stale = [g for g in gap if g["decision"] == "PORTED"]
    assert not stale, (
        "marked PORTED but absent from the LISA driver: %s"
        % ", ".join(g["key"] for g in stale))


def test_every_decision_is_a_known_verdict(state):
    gap, _extras, _ledger = state
    bad = sorted({g["decision"] for g in gap
                  if g["decision"] is not None and g["decision"] not in audit.DECISIONS})
    assert not bad, "unknown decision value(s) in the ledger: %s" % bad


def test_every_decision_carries_a_reason(state):
    """A verdict without a reason is silence with extra steps."""
    gap, _extras, _ledger = state
    thin = [g["key"] for g in gap
            if g["decision"] is not None and len((g["reason"] or "").strip()) < 20]
    assert not thin, "decision recorded with no usable reason: %s" % ", ".join(thin)


def test_ledger_has_no_entries_for_items_outside_the_gap(state):
    """Spent entries are not a failure, but they should not pile up as fiction.

    An entry naming something no longer in the gap means it was ported or the main driver
    dropped it; regenerating the ledger clears it.
    """
    gap, _extras, ledger = state
    gap_keys = {g["key"] for g in gap}
    spent = sorted(k for k in ledger if k not in gap_keys)
    assert not spent, ("ledger describes %d item(s) that are no longer in the gap: %s\n"
                       "Regenerate with make_lisa_drift_ledger.py."
                       % (len(spent), ", ".join(spent)))


def test_the_committed_ledger_matches_what_its_generator_produces():
    """The ledger is GENERATED.  Nothing enforced that until this test.

    An adversarial audit added an option to the main driver and hand-wrote a
    ``{"decision": "NA", "reason": "..."}`` entry straight into the JSON: the whole gate
    passed while make_lisa_drift_ledger.py still reported the item as matching no rule.
    The stated property -- that a person has to classify new drift AS A RULE, with a reason
    -- was silenceable by a one-line JSON edit.

    So regenerate in memory and compare.  This also catches a ledger left stale after the
    main driver moved.
    """
    gen = pytest.importorskip("make_lisa_drift_ledger",
                              reason="LISA drift ledger generator not present")
    gap, _extras = audit.compute_gap()
    expected, unmatched = {}, []
    for item in gap:
        decision, reason = gen.classify(item["key"])
        if decision is None:
            unmatched.append(item["key"])
        else:
            expected[item["key"]] = {"decision": decision, "reason": reason}

    assert not unmatched, (
        "%d gap item(s) match no rule in make_lisa_drift_ledger.py: %s\n"
        "Add a rule with a reason -- do not hand-edit the JSON."
        % (len(unmatched), ", ".join(unmatched)))

    committed = audit.load_ledger()
    assert committed == expected, (
        "lisa_drift_ledger.json does not match make_lisa_drift_ledger.py.\n"
        "Regenerate it (python3 make_lisa_drift_ledger.py) rather than editing the JSON:\n"
        "  only in committed: %s\n  only in generated: %s\n  differing: %s"
        % (sorted(set(committed) - set(expected)),
           sorted(set(expected) - set(committed)),
           sorted(k for k in set(committed) & set(expected) if committed[k] != expected[k])))


def test_the_fairdraw_helpers_ported_in_this_pass_are_present_in_lisa():
    """Belt and braces: name them, so deleting one fails here as well as via the ledger."""
    lisa = audit.collect(audit.LISA)
    for name in ('ln_weights_from_rvs', 'ln_weights_for_posterior',
                 '_rvs_is_export_resample', '_rvs_is_equal_weight', '_rvs_len',
                 '_rvs_lnL_convention'):
        assert name in lisa["FUNC"], "%s is missing from the LISA driver" % name
    for marker in ('_rvs_is_fairdraw', '_rvs_is_pooled'):
        assert marker in lisa["ATTR"], "%s is no longer read by the LISA driver" % marker
