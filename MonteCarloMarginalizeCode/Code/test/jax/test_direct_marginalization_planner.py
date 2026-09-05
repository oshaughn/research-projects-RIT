"""Focused policy tests for the opt-in direct-marginalization planner.

The amplitude ladder below is a synthetic calibration packet.  The planner is
being tested, not a new accuracy claim for the shipped angle kernels: production
offers must bring their own measured resource and error provenance.
"""

import json
import math

import pytest

from RIFT.likelihood.jax_ile import direct_marginalization_planner as P


def _certified_warrant(scope="synthetic finite spectrum"):
    return P.Warrant(P.WarrantKind.EXACT_TRIG_DEGREE, scope, True,
                     "test fixture: analytic finite-spectrum bound")


def _offer(axis, scheme, error, compute, memory=64, *,
           evidence=P.EvidenceKind.CERTIFIED, warrant=None,
           requires=(), conflicts=(), conditional_requirements=()):
    if warrant is None:
        warrant = _certified_warrant()
    accuracy = P.AccuracyAssessment(
        evidence, error,
        "test fixture: error envelope for %s:%s" % (axis, scheme))
    resources = P.ResourceEstimate(
        compute, memory,
        "test fixture: common-unit cost model for %s:%s" % (axis, scheme))
    return P.SchemeOffer(
        axis, scheme, accuracy, resources, warrant,
        "test fixture offer", requires=frozenset(requires),
        conflicts=frozenset(conflicts),
        conditional_requirements=tuple(conditional_requirements))


def _amplitude_offers(amplitude):
    """Synthetic measured envelopes with distinct accuracy and cost crossings."""
    amplitude = float(amplitude)
    return (
        _offer("angle", "exact", error=1e-8,
               compute=5.0 + amplitude / 50.0),
        _offer("angle", "laplace", error=30.0 / amplitude ** 2,
               compute=40.0 + math.sqrt(amplitude)),
    )


@pytest.mark.parametrize(
    "amplitude, expected",
    [(25.0, "exact"), (400.0, "exact"), (40000.0, "laplace")])
def test_low_moderate_high_amplitude_choose_cheapest_certified(
        amplitude, expected):
    """Accuracy gates low A; measured cost, not one crossover, orders the rest."""
    decision = P.plan_direct_marginalization(
        _amplitude_offers(amplitude), {"angle": 1e-2},
        P.ResourceBudget(2000.0, 1024), required_axes=("angle",))
    assert decision.action == "run"
    assert decision.basis == "cheapest-certified"
    assert decision.certified is True
    assert decision.require_selection()[0].scheme == expected


def test_combination_resource_model_controls_nested_kernel_cost():
    """A measured whole-kernel model can override the additive safe default."""
    offers = (
        _offer("angle", "exact", error=1e-5, compute=1),
        _offer("angle", "laplace", error=1e-5, compute=100),
        _offer("distance", "uniform", error=1e-5, compute=1),
    )

    def nested_cost(combination):
        angle = next(o.scheme for o in combination if o.axis == "angle")
        return P.ResourceEstimate(
            10 if angle == "laplace" else 100, 50,
            "fixture: measured complete nested-kernel cost")

    decision = P.plan_direct_marginalization(
        offers, {"angle": 1e-3, "distance": 1e-3},
        P.ResourceBudget(200, 100),
        required_axes=("angle", "distance"),
        resource_model=nested_cost)
    selected = {offer.axis: offer.scheme
                for offer in decision.require_selection()}
    assert selected == {"angle": "laplace", "distance": "uniform"}
    assert "complete nested-kernel" in decision.resource_use.provenance


@pytest.mark.parametrize(
    "error_budget, resource_budget, reason_code",
    [
        (None, {"max_compute_units": 100, "max_memory_bytes": 100},
         "missing-error-budget"),
        ({}, {"max_compute_units": 100, "max_memory_bytes": 100},
         "missing-error-budget"),
        ({"angle": 0.1}, None, "missing-resource-budget"),
        ({"angle": 0.1}, {"max_compute_units": 100},
         "missing-resource-budget"),
    ])
def test_missing_budget_declines_with_no_selection(
        error_budget, resource_budget, reason_code):
    decision = P.plan_direct_marginalization(
        (_offer("angle", "exact", 1e-3, 10),),
        error_budget, resource_budget, required_axes=("angle",))
    assert decision.action == "decline"
    assert decision.reason_code == reason_code
    assert decision.selected == ()
    with pytest.raises(P.MarginalizationPlanDeclined, match=reason_code):
        decision.require_selection()


def test_repeated_required_axis_declines_instead_of_planning_it_twice():
    """One scheme per axis: a repeated axis is a malformed request, not a plan."""
    decision = P.plan_direct_marginalization(
        (_offer("angle", "exact", 1e-5, 10),), {"angle": 1e-2},
        P.ResourceBudget(1000.0, 1024), required_axes=("angle", "angle"))
    assert decision.action == "decline"
    assert decision.reason_code == "duplicate-axis"
    assert decision.selected == ()
    assert decision.resource_use is None
    assert decision.ledger["details"]["duplicate_axes"] == ["angle"]
    assert decision.ledger["combinations"] == []
    with pytest.raises(P.MarginalizationPlanDeclined, match="duplicate-axis"):
        decision.require_selection()
    json.dumps(decision.as_dict())


def test_shipped_peak_local_plus_gh_is_an_unsupported_combination():
    """The real JAX profile declares this once; the planner refuses the pair."""
    def validated(label):
        return P.AccuracyAssessment(
            P.EvidenceKind.VALIDATED, 1e-4,
            "fixture validation: " + label)

    def resources(label):
        return P.ResourceEstimate(10.0, 10, "fixture cost: " + label)

    offers = (
        P.make_jax_scheme_offer(
            "angle", "peak-local", validated("angle"), resources("angle"),
            provenance="fixture request"),
        P.make_jax_scheme_offer(
            "distance", "gh", validated("distance"), resources("distance"),
            provenance="fixture request"),
    )
    decision = P.plan_jax_direct_marginalization(
        offers, {"angle": 1e-3, "distance": 1e-3},
        P.ResourceBudget(100.0, 100),
        required_axes=("angle", "distance"),
        capabilities=("angle-amplitude-estimate",
                      "angle-peak-local-warranted",
                      "distance-volumetric-prior"),
        allow_best_effort=True)
    assert decision.action == "decline"
    assert decision.reason_code == "no-compatible-plan"
    records = decision.ledger["combinations"]
    assert len(records) == 1
    assert any("angle:peak-local conflicts" in reason
               for reason in records[0]["compatibility_reasons"])


def test_conditional_gh_laplace_warrant_must_be_supplied():
    """GH+Laplace is supported only after the concrete identity predicate passes."""
    validated = P.AccuracyAssessment(
        P.EvidenceKind.VALIDATED, 1e-4, "fixture validation")
    resources = P.ResourceEstimate(10.0, 10, "fixture cost")
    offers = (
        P.make_jax_scheme_offer("angle", "laplace", validated, resources,
                                provenance="fixture request"),
        P.make_jax_scheme_offer("distance", "gh", validated, resources,
                                provenance="fixture request"),
    )
    base_capabilities = ("angle-amplitude-estimate",
                         "distance-volumetric-prior")
    refused = P.plan_jax_direct_marginalization(
        offers, {"angle": 1e-3, "distance": 1e-3},
        P.ResourceBudget(100.0, 100),
        required_axes=("angle", "distance"),
        capabilities=base_capabilities, allow_best_effort=True)
    assert refused.action == "decline"
    assert "gh-laplace-supported" in str(refused.ledger["combinations"])

    allowed = P.plan_jax_direct_marginalization(
        offers, {"angle": 1e-3, "distance": 1e-3},
        P.ResourceBudget(100.0, 100),
        required_axes=("angle", "distance"),
        capabilities=base_capabilities + ("gh-laplace-supported",),
        allow_best_effort=True)
    assert allowed.action == "run"
    assert allowed.basis == "most-accurate-affordable"


def test_jax_direct_path_injects_the_nonlinear_time_incompatibility():
    """Callers cannot omit the wrapper fact that currently excludes bandlimited."""
    validated = P.AccuracyAssessment(
        P.EvidenceKind.VALIDATED, 1e-4, "fixture validation")
    resources = P.ResourceEstimate(10.0, 10, "fixture cost")
    offers = (
        P.make_jax_scheme_offer("angle", "exact", validated, resources,
                                provenance="fixture request"),
        P.make_jax_scheme_offer("distance", "uniform", validated, resources,
                                provenance="fixture request"),
        P.make_jax_scheme_offer("time", "bandlimited", validated,
                                resources, provenance="fixture request"),
    )
    decision = P.plan_jax_direct_marginalization(
        offers, {"angle": 1e-3, "distance": 1e-3, "time": 1e-3},
        P.ResourceBudget(100.0, 100),
        capabilities=("angle-amplitude-estimate", "time-exact-band-limit",
                      "time-independent-rho-sq", "n-cal-one"),
        allow_best_effort=True)
    assert decision.action == "decline"
    assert decision.reason_code == "no-compatible-plan"
    assert "jax-direct-nonlinear-time" in decision.ledger["capabilities"]


def test_no_silent_fallback_and_best_effort_requires_explicit_authority():
    """An affordable estimate is a suggestion, never an implicit replacement."""
    exact = _offer("angle", "exact", error=1e-5, compute=200, memory=20)
    empirical_warrant = P.Warrant(
        P.WarrantKind.EMPIRICAL_CALIBRATION, "measured envelope", False,
        "test fixture: empirical campaign")
    approximate = _offer(
        "angle", "approximate", error=2e-2, compute=10, memory=10,
        evidence=P.EvidenceKind.VALIDATED, warrant=empirical_warrant)
    budget = P.ResourceBudget(100, 100)

    strict = P.plan_direct_marginalization(
        (exact, approximate), {"angle": 1e-2}, budget,
        required_axes=("angle",))
    assert strict.action == "decline"
    assert strict.reason_code == "resource-budget-exceeded"
    assert strict.selected == ()
    assert [offer.scheme for offer in strict.suggested] == ["approximate"]
    assert strict.meets_error_budget is False
    with pytest.raises(P.MarginalizationPlanDeclined):
        strict.require_selection()

    explicit = P.plan_direct_marginalization(
        (exact, approximate), {"angle": 1e-2}, budget,
        required_axes=("angle",), allow_best_effort=True)
    assert explicit.action == "run"
    assert explicit.basis == "most-accurate-affordable"
    assert explicit.certified is False
    assert explicit.meets_error_budget is False
    assert explicit.require_selection()[0].scheme == "approximate"
    record = explicit.as_dict()
    assert record["selected"][0]["accuracy"]["provenance"]
    assert record["selected"][0]["warrant"]["provenance"]
    assert record["selected"][0]["resources"]["provenance"]
    json.dumps(record)


def test_current_angle_profiles_cannot_be_mislabeled_certified():
    """Exact coefficients do not certify the amplitude-sized exp quadrature."""
    accuracy = P.AccuracyAssessment(
        P.EvidenceKind.CERTIFIED, 1e-8, "invalid fixture claim")
    resources = P.ResourceEstimate(1.0, 1, "fixture cost")
    with pytest.raises(ValueError, match="no implemented certificate"):
        P.make_jax_scheme_offer(
            "angle", "exact", accuracy, resources,
            provenance="attempted invalid offer")


def test_bandlimited_time_profile_cannot_be_mislabeled_certified():
    """A derived-and-remeasured refinement factor is not a per-request bound."""
    accuracy = P.AccuracyAssessment(
        P.EvidenceKind.CERTIFIED, 1e-12, "invalid fixture claim")
    resources = P.ResourceEstimate(1.0, 1, "fixture cost")
    with pytest.raises(ValueError, match="no implemented certificate"):
        P.make_jax_scheme_offer(
            "time", "bandlimited", accuracy, resources,
            provenance="attempted invalid offer")


def test_no_shipped_profile_advertises_a_certificate_yet():
    """cheapest-certified stays unreachable until some rule implements a bound."""
    advertised = sorted(key for key, profile in P.JAX_SCHEME_PROFILES.items()
                        if profile.warrant.certificate_available)
    assert advertised == []
