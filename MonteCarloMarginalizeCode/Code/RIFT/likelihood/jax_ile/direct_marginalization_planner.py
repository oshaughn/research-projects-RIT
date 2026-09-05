"""Opt-in planner for error- and resource-budgeted direct marginalization.

This module is deliberately separate from :func:`choose_angle_marg_scheme`.
Importing it changes no default and the existing ``angle_marg='auto'`` path
continues to use the measured amplitude crossover.  A caller must construct
scheme offers, provide every requested per-axis error budget and both resource
budgets, and explicitly call :func:`plan_direct_marginalization`.

The planner does not turn a calibration into a proof.  Each offer carries an
accuracy assessment, a completeness warrant and its provenance.  Only an
assessment marked ``CERTIFIED`` under a warrant with an implemented
certificate participates in the ``cheapest-certified`` choice.  An empirical
or unknown offer can only be run when the caller explicitly sets
``allow_best_effort=True``; otherwise it is returned as a non-executable
suggestion on a structured decline.

By default, resource estimates are conservative additive contributions on a
common unit: compute and peak-memory contributions are summed.  A nested JAX
adapter can instead supply a combination-aware ``resource_model`` whose return
value carries its own provenance.  It may over-count buffers whose lifetimes do
not overlap, but may not under-count them; an optimistic lifetime model would
be another silent OOM fallback.
"""

from dataclasses import dataclass, field
from enum import Enum
from itertools import product
import math
from types import MappingProxyType


__all__ = [
    "AccuracyAssessment",
    "ConditionalRequirement",
    "EvidenceKind",
    "JAX_DIRECT_MARGINALIZATION_AXES",
    "JAX_SCHEME_PROFILES",
    "MarginalizationPlanDeclined",
    "PlanDecision",
    "ResourceBudget",
    "ResourceEstimate",
    "SchemeOffer",
    "SchemeProfile",
    "Warrant",
    "WarrantKind",
    "make_jax_scheme_offer",
    "plan_direct_marginalization",
    "plan_jax_direct_marginalization",
]


class WarrantKind(str, Enum):
    """Finite structures which may warrant a completeness certificate.

    ``EFFECTIVE_BANDWIDTH_WITH_MARGIN`` is intentionally represented even
    though it cannot certify completeness.  Naming it lets the planner refuse
    a proof claim instead of treating every amplitude-sized grid as exact.
    """

    EXACT_BAND_LIMIT = "exact-band-limit"
    EXACT_TRIG_DEGREE = "exact-trig-degree"
    BOUNDED_STATIONARY_SET = "bounded-stationary-set"
    EFFECTIVE_BANDWIDTH_WITH_MARGIN = "effective-bandwidth-with-margin"
    EMPIRICAL_CALIBRATION = "empirical-calibration"
    NONE = "none"


class EvidenceKind(str, Enum):
    """Strength of a quantitative per-axis error assessment."""

    CERTIFIED = "certified"
    VALIDATED = "validated"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


_POTENTIALLY_CERTIFYING_WARRANTS = frozenset((
    WarrantKind.EXACT_BAND_LIMIT,
    WarrantKind.EXACT_TRIG_DEGREE,
    WarrantKind.BOUNDED_STATIONARY_SET,
))


def _enum_value(value, enum_type, field_name):
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except ValueError:
        raise ValueError("unknown %s %r" % (field_name, value))


def _finite_nonnegative(value, field_name):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("%s must be finite and non-negative; got %r"
                         % (field_name, value))
    return value


def _nonnegative_integer(value, field_name):
    if isinstance(value, bool):
        raise ValueError("%s must be a non-negative integer" % field_name)
    try:
        as_float = float(value)
        as_int = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("%s must be a non-negative integer" % field_name)
    if (not math.isfinite(as_float) or as_float < 0.0
            or as_float != float(as_int)):
        raise ValueError("%s must be a non-negative integer; got %r"
                         % (field_name, value))
    return as_int


@dataclass(frozen=True)
class Warrant:
    """Completeness warrant carried by one implementation.

    ``certificate_available`` means the implementation actually discharges a
    quantitative error inequality.  A mathematical structure that could
    support a future certificate is not sufficient.
    """

    kind: WarrantKind
    scope: str
    certificate_available: bool
    provenance: str

    def __post_init__(self):
        object.__setattr__(self, "kind", _enum_value(
            self.kind, WarrantKind, "warrant kind"))
        if not self.scope or not self.provenance:
            raise ValueError("warrant scope and provenance must be non-empty")
        if (self.certificate_available
                and self.kind not in _POTENTIALLY_CERTIFYING_WARRANTS):
            raise ValueError(
                "warrant %s cannot advertise a completeness certificate"
                % self.kind.value)

    def as_dict(self):
        return dict(kind=self.kind.value, scope=self.scope,
                    certificate_available=bool(self.certificate_available),
                    provenance=self.provenance)


@dataclass(frozen=True)
class AccuracyAssessment:
    """Quantitative error information for one axis and scheme.

    The unit is absolute error in the marginalized log likelihood (nats).
    ``UNKNOWN`` must carry ``max_error_nats=None``.  The other evidence kinds
    need a finite non-negative value, but only ``CERTIFIED`` is a hard bound.
    """

    evidence: EvidenceKind
    max_error_nats: object
    provenance: str

    def __post_init__(self):
        object.__setattr__(self, "evidence", _enum_value(
            self.evidence, EvidenceKind, "evidence kind"))
        if not self.provenance:
            raise ValueError("accuracy provenance must be non-empty")
        if self.evidence is EvidenceKind.UNKNOWN:
            if self.max_error_nats is not None:
                raise ValueError(
                    "UNKNOWN accuracy must not carry a numerical error")
        else:
            object.__setattr__(self, "max_error_nats", _finite_nonnegative(
                self.max_error_nats, "max_error_nats"))

    def as_dict(self):
        return dict(evidence=self.evidence.value,
                    max_error_nats=self.max_error_nats,
                    provenance=self.provenance)


@dataclass(frozen=True)
class ResourceEstimate:
    """Conservative contribution to a plan's compute and peak memory."""

    compute_units: float
    memory_bytes: int
    provenance: str

    def __post_init__(self):
        object.__setattr__(self, "compute_units", _finite_nonnegative(
            self.compute_units, "compute_units"))
        object.__setattr__(self, "memory_bytes", _nonnegative_integer(
            self.memory_bytes, "memory_bytes"))
        if not self.provenance:
            raise ValueError("resource provenance must be non-empty")

    def as_dict(self):
        return dict(compute_units=self.compute_units,
                    memory_bytes=self.memory_bytes,
                    provenance=self.provenance)


@dataclass(frozen=True)
class ResourceBudget:
    """Hard request-level ceilings.

    The fields may be ``None`` only so a missing budget can produce a
    structured decline.  A complete request needs both.
    """

    max_compute_units: object
    max_memory_bytes: object

    def __post_init__(self):
        if self.max_compute_units is not None:
            object.__setattr__(self, "max_compute_units", _finite_nonnegative(
                self.max_compute_units, "max_compute_units"))
        if self.max_memory_bytes is not None:
            object.__setattr__(self, "max_memory_bytes", _nonnegative_integer(
                self.max_memory_bytes, "max_memory_bytes"))

    def validation_errors(self):
        errors = []
        if self.max_compute_units is None:
            errors.append("max_compute_units")
        if self.max_memory_bytes is None:
            errors.append("max_memory_bytes")
        return tuple(errors)

    def as_dict(self):
        return dict(max_compute_units=self.max_compute_units,
                    max_memory_bytes=self.max_memory_bytes)


@dataclass(frozen=True)
class ConditionalRequirement:
    """Capability required only when another scheme/token is selected."""

    trigger: str
    capability: str
    reason: str

    def __post_init__(self):
        if not self.trigger or not self.capability or not self.reason:
            raise ValueError("conditional requirement fields must be non-empty")

    def as_dict(self):
        return dict(trigger=self.trigger, capability=self.capability,
                    reason=self.reason)


@dataclass(frozen=True)
class SchemeOffer:
    """One runnable scheme offered for one marginalized axis."""

    axis: str
    scheme: str
    accuracy: AccuracyAssessment
    resources: ResourceEstimate
    warrant: Warrant
    provenance: str
    requires: frozenset = field(default_factory=frozenset)
    provides: frozenset = field(default_factory=frozenset)
    conflicts: frozenset = field(default_factory=frozenset)
    conditional_requirements: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if not self.axis or not self.scheme or not self.provenance:
            raise ValueError("offer axis, scheme and provenance must be non-empty")
        object.__setattr__(self, "requires", frozenset(self.requires))
        object.__setattr__(self, "provides", frozenset(self.provides))
        object.__setattr__(self, "conflicts", frozenset(self.conflicts))
        object.__setattr__(self, "conditional_requirements",
                           tuple(self.conditional_requirements))
        if (self.accuracy.evidence is EvidenceKind.CERTIFIED
                and not self.warrant.certificate_available):
            raise ValueError(
                "%s cannot claim CERTIFIED accuracy: its %s warrant has no "
                "implemented certificate" % (self.key, self.warrant.kind.value))

    @property
    def key(self):
        return "%s:%s" % (self.axis, self.scheme)

    def as_dict(self):
        return dict(
            key=self.key, axis=self.axis, scheme=self.scheme,
            accuracy=self.accuracy.as_dict(),
            resources=self.resources.as_dict(), warrant=self.warrant.as_dict(),
            provenance=self.provenance, requires=sorted(self.requires),
            provides=sorted(self.provides), conflicts=sorted(self.conflicts),
            conditional_requirements=[r.as_dict()
                                      for r in self.conditional_requirements])


class MarginalizationPlanDeclined(RuntimeError):
    """Raised when a caller tries to execute a declined decision."""


@dataclass(frozen=True)
class PlanDecision:
    """Structured planner result.  ``action`` is either ``run`` or ``decline``."""

    action: str
    basis: str
    reason_code: str
    reason: str
    selected: tuple
    suggested: tuple
    resource_use: object
    suggested_resource_use: object
    certified: bool
    meets_error_budget: bool
    ledger: dict

    def require_selection(self):
        """Return the selected offers, refusing a declined recommendation."""
        if self.action != "run":
            raise MarginalizationPlanDeclined(
                "%s: %s" % (self.reason_code, self.reason))
        return self.selected

    def as_dict(self):
        return dict(
            action=self.action, basis=self.basis,
            reason_code=self.reason_code, reason=self.reason,
            selected=[o.as_dict() for o in self.selected],
            suggested=[o.as_dict() for o in self.suggested],
            resource_use=(None if self.resource_use is None
                          else self.resource_use.as_dict()),
            suggested_resource_use=(
                None if self.suggested_resource_use is None
                else self.suggested_resource_use.as_dict()),
            certified=bool(self.certified),
            meets_error_budget=bool(self.meets_error_budget),
            ledger=self.ledger)


def _resource_use(offers, resource_model=None):
    if resource_model is None:
        return ResourceEstimate(
            sum(o.resources.compute_units for o in offers),
            sum(o.resources.memory_bytes for o in offers),
            "conservative additive aggregation of selected offer estimates")
    use = resource_model(tuple(offers))
    if not isinstance(use, ResourceEstimate):
        raise TypeError("resource_model must return ResourceEstimate")
    return use


def _resource_reasons(use, budget):
    reasons = []
    if use.compute_units > float(budget.max_compute_units):
        reasons.append("compute %.9g exceeds budget %.9g"
                       % (use.compute_units,
                          float(budget.max_compute_units)))
    if use.memory_bytes > int(budget.max_memory_bytes):
        reasons.append("memory %d exceeds budget %d"
                       % (use.memory_bytes, int(budget.max_memory_bytes)))
    return reasons


def _compatibility_reasons(offers, capabilities):
    capabilities = frozenset(capabilities)
    tokens = set(capabilities)
    for offer in offers:
        tokens.add(offer.key)
        tokens.update(offer.provides)
    reasons = []
    for offer in offers:
        missing = sorted(offer.requires.difference(tokens))
        if missing:
            reasons.append("%s missing requirements %r" % (offer.key, missing))
        conflicts = sorted(offer.conflicts.intersection(tokens))
        if conflicts:
            reasons.append("%s conflicts with %r" % (offer.key, conflicts))
        for requirement in offer.conditional_requirements:
            if (requirement.trigger in tokens
                    and requirement.capability not in capabilities):
                reasons.append(
                    "%s with %s requires capability %s: %s"
                    % (offer.key, requirement.trigger,
                       requirement.capability, requirement.reason))
    return reasons


def _error_reasons(offers, error_budget, certified_only):
    reasons = []
    for offer in offers:
        assessment = offer.accuracy
        if certified_only and assessment.evidence is not EvidenceKind.CERTIFIED:
            reasons.append("%s accuracy is %s, not certified"
                           % (offer.key, assessment.evidence.value))
            continue
        if assessment.max_error_nats is None:
            reasons.append("%s has no quantitative error assessment" % offer.key)
            continue
        limit = float(error_budget[offer.axis])
        if assessment.max_error_nats > limit:
            reasons.append("%s error %.9g exceeds axis budget %.9g"
                           % (offer.key, assessment.max_error_nats, limit))
    return reasons


def _accuracy_rank(offers, error_budget, resource_model):
    unknown = sum(o.accuracy.max_error_nats is None for o in offers)
    ratios = [o.accuracy.max_error_nats / float(error_budget[o.axis])
              for o in offers if o.accuracy.max_error_nats is not None]
    worst = max(ratios) if ratios else math.inf
    total = sum(ratios) if ratios else math.inf
    evidence_order = {EvidenceKind.CERTIFIED: 0, EvidenceKind.VALIDATED: 1,
                      EvidenceKind.ESTIMATED: 2, EvidenceKind.UNKNOWN: 3}
    evidence = sum(evidence_order[o.accuracy.evidence] for o in offers)
    use = _resource_use(offers, resource_model)
    return (unknown, worst, total, evidence, use.compute_units,
            use.memory_bytes, tuple(o.key for o in offers))


def _cost_rank(offers, error_budget, resource_model):
    use = _resource_use(offers, resource_model)
    ratios = [o.accuracy.max_error_nats / float(error_budget[o.axis])
              for o in offers]
    return (use.compute_units, use.memory_bytes, max(ratios), sum(ratios),
            tuple(o.key for o in offers))


def _preflight_decline(reason_code, reason, axes, error_budget,
                       resource_budget, capabilities, details):
    if resource_budget is None:
        resource_record = None
    elif isinstance(resource_budget, dict):
        resource_record = dict(resource_budget)
    else:
        resource_record = resource_budget.as_dict()
    return PlanDecision(
        action="decline", basis="decline", reason_code=reason_code,
        reason=reason, selected=(), suggested=(), resource_use=None,
        suggested_resource_use=None, certified=False,
        meets_error_budget=False,
        ledger=dict(required_axes=list(axes),
                    error_budget=None if error_budget is None
                    else dict(error_budget),
                    resource_budget=resource_record,
                    capabilities=sorted(capabilities), details=details,
                    combinations=[]))


def plan_direct_marginalization(offers, error_budget, resource_budget, *,
                                required_axes=None, capabilities=(),
                                allow_best_effort=False, resource_model=None):
    """Choose a direct-marginalization plan without changing any RIFT default.

    The primary policy is the least-compute plan whose per-axis errors are
    certified within budget and whose summed resource estimates fit.  If none
    exists, the most accurate affordable compatible plan is recorded as a
    suggestion.  It becomes executable only under the explicit
    ``allow_best_effort=True`` policy.  ``resource_model``, when supplied, is
    called on each complete offer combination and must return a provenance-
    carrying :class:`ResourceEstimate`; exceptions are never converted to a
    decline.
    """
    offers = tuple(offers)
    capabilities = frozenset(capabilities)
    keys = [offer.key for offer in offers]
    if len(keys) != len(set(keys)):
        raise ValueError("offer keys must be unique; got %r" % keys)
    axes = tuple(required_axes) if required_axes is not None else tuple(sorted(
        set(offer.axis for offer in offers)))
    if not axes:
        return _preflight_decline(
            "missing-axis", "no marginalization axes were requested", axes,
            error_budget, resource_budget, capabilities, {})

    by_axis = {axis: tuple(o for o in offers if o.axis == axis) for axis in axes}
    unsupported = [axis for axis in axes if not by_axis[axis]]
    if unsupported:
        return _preflight_decline(
            "unsupported-axis", "no scheme offers for axes %r" % unsupported,
            axes, error_budget, resource_budget, capabilities,
            dict(unsupported_axes=unsupported))

    if error_budget is None:
        return _preflight_decline(
            "missing-error-budget", "a per-axis error budget is required",
            axes, error_budget, resource_budget, capabilities,
            dict(missing_axes=list(axes)))
    missing_axes = [axis for axis in axes if axis not in error_budget]
    if missing_axes:
        return _preflight_decline(
            "missing-error-budget",
            "error budget is missing axes %r" % missing_axes,
            axes, error_budget, resource_budget, capabilities,
            dict(missing_axes=missing_axes))
    clean_error_budget = {}
    for axis in axes:
        value = float(error_budget[axis])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("error budget for %s must be finite and positive"
                             % axis)
        clean_error_budget[axis] = value

    if resource_budget is None:
        return _preflight_decline(
            "missing-resource-budget",
            "both compute and memory budgets are required", axes,
            clean_error_budget, resource_budget, capabilities,
            dict(missing=("max_compute_units", "max_memory_bytes")))
    if isinstance(resource_budget, dict):
        resource_budget = ResourceBudget(
            resource_budget.get("max_compute_units"),
            resource_budget.get("max_memory_bytes"))
    missing_resources = resource_budget.validation_errors()
    if missing_resources:
        return _preflight_decline(
            "missing-resource-budget", "resource budget is missing %r"
            % (missing_resources,), axes, clean_error_budget,
            resource_budget, capabilities,
            dict(missing=missing_resources))

    combinations = []
    compatible = []
    affordable = []
    certified = []
    certified_affordable = []
    for combination in product(*(by_axis[axis] for axis in axes)):
        use = _resource_use(combination, resource_model)
        compat_reasons = _compatibility_reasons(combination, capabilities)
        resource_reasons = _resource_reasons(use, resource_budget)
        certified_error_reasons = _error_reasons(
            combination, clean_error_budget, certified_only=True)
        numeric_error_reasons = _error_reasons(
            combination, clean_error_budget, certified_only=False)
        record = dict(
            schemes=[o.key for o in combination],
            compatibility_reasons=compat_reasons,
            resource_reasons=resource_reasons,
            certified_error_reasons=certified_error_reasons,
            numeric_error_reasons=numeric_error_reasons,
            resource_use=use.as_dict())
        combinations.append(record)
        if compat_reasons:
            continue
        compatible.append(combination)
        if not resource_reasons:
            affordable.append(combination)
        if not certified_error_reasons:
            certified.append(combination)
            if not resource_reasons:
                certified_affordable.append(combination)

    ledger = dict(
        required_axes=list(axes), error_budget=clean_error_budget,
        resource_budget=resource_budget.as_dict(),
        capabilities=sorted(capabilities),
        allow_best_effort=bool(allow_best_effort),
        offers=[offer.as_dict() for offer in offers],
        combinations=combinations)

    if certified_affordable:
        chosen = min(certified_affordable,
                     key=lambda c: _cost_rank(
                         c, clean_error_budget, resource_model))
        use = _resource_use(chosen, resource_model)
        return PlanDecision(
            action="run", basis="cheapest-certified", reason_code="selected",
            reason="least-compute compatible plan certified within every "
                   "axis and resource budget",
            selected=tuple(chosen), suggested=(), resource_use=use,
            suggested_resource_use=None, certified=True,
            meets_error_budget=True, ledger=ledger)

    best = (min(affordable,
                key=lambda c: _accuracy_rank(
                    c, clean_error_budget, resource_model))
            if affordable else None)
    best_use = (_resource_use(best, resource_model)
                if best is not None else None)
    best_numeric_ok = bool(best is not None and not _error_reasons(
        best, clean_error_budget, certified_only=False))

    if best is not None and allow_best_effort:
        return PlanDecision(
            action="run", basis="most-accurate-affordable",
            reason_code="best-effort-authorized",
            reason="no affordable fully certified plan; caller explicitly "
                   "authorized the most accurate affordable compatible plan",
            selected=tuple(best), suggested=(), resource_use=best_use,
            suggested_resource_use=None, certified=False,
            meets_error_budget=best_numeric_ok, ledger=ledger)

    if not compatible:
        code = "no-compatible-plan"
        reason = "all scheme combinations violate declared compatibility"
    elif certified and not certified_affordable:
        code = "resource-budget-exceeded"
        reason = "certified plans exist, but none fits both resource budgets"
    elif not certified:
        code = "no-certified-plan"
        reason = "no compatible plan is certified within every axis budget"
    else:
        code = "no-affordable-plan"
        reason = "no compatible plan fits both resource budgets"
    return PlanDecision(
        action="decline", basis="decline", reason_code=code, reason=reason,
        selected=(), suggested=tuple(best) if best is not None else (),
        resource_use=None, suggested_resource_use=best_use,
        certified=False, meets_error_budget=False, ledger=ledger)


@dataclass(frozen=True)
class SchemeProfile:
    """Static compatibility and warrant facts for a shipped JAX scheme."""

    axis: str
    scheme: str
    warrant: Warrant
    provenance: str
    requires: frozenset = field(default_factory=frozenset)
    conflicts: frozenset = field(default_factory=frozenset)
    conditional_requirements: tuple = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "requires", frozenset(self.requires))
        object.__setattr__(self, "conflicts", frozenset(self.conflicts))
        object.__setattr__(self, "conditional_requirements",
                           tuple(self.conditional_requirements))

    @property
    def key(self):
        return "%s:%s" % (self.axis, self.scheme)


def _warrant(kind, scope, available, provenance):
    return Warrant(kind, scope, available, provenance)


_FRAMEWORK = "RIFT/likelihood/DESIGN_peak_local_framework.md"
_ANGLE = "RIFT/likelihood/jax_ile/anglemarg.py"
_DISTANCE = "RIFT/likelihood/jax_ile/DESIGN_jax_distance_quadrature.md"
_TIME = "RIFT/likelihood/time_marginalization_quadrature.py"


def _profile(axis, scheme, warrant, provenance, requires=(), conflicts=(),
             conditional_requirements=()):
    return SchemeProfile(axis, scheme, warrant, provenance,
                         frozenset(requires), frozenset(conflicts),
                         tuple(conditional_requirements))


# These profiles state structural facts only.  In particular they intentionally
# do not invent error or wall-time envelopes for the current schemes.
_JAX_PROFILE_LIST = (
    _profile("angle", "grid",
             _warrant(WarrantKind.NONE, "fixed legacy product grid", False,
                      _ANGLE), _ANGLE,
             conflicts=("distance:loguniform",)),
    _profile("angle", "exact",
             _warrant(WarrantKind.EFFECTIVE_BANDWIDTH_WITH_MARGIN,
                      "exact angle coefficients, amplitude-sized exp grid",
                      False, _FRAMEWORK), _ANGLE,
             requires=("angle-amplitude-estimate",)),
    _profile("angle", "laplace",
             _warrant(WarrantKind.EFFECTIVE_BANDWIDTH_WITH_MARGIN,
                      "dense phi plus enumerated psi Laplace rule", False,
                      _FRAMEWORK), _ANGLE,
             requires=("angle-amplitude-estimate",),
             conditional_requirements=(ConditionalRequirement(
                 "distance:gh", "gh-laplace-supported",
                 "the A0==0/B1==0 identity must hold on concrete tables"),)),
    _profile("angle", "peak-local",
             _warrant(WarrantKind.EFFECTIVE_BANDWIDTH_WITH_MARGIN,
                      "exact-trig-degree psi cells but amplitude-sized dense phi",
                      False, _FRAMEWORK), _ANGLE,
             requires=("angle-amplitude-estimate",
                       "angle-peak-local-warranted"),
             conflicts=("distance:gh",)),
    _profile("distance", "uniform",
             _warrant(WarrantKind.NONE, "fixed uniform-in-distance grid", False,
                      _DISTANCE), _DISTANCE),
    _profile("distance", "loguniform",
             _warrant(WarrantKind.BOUNDED_STATIONARY_SET,
                      "interior Gaussian peak on finite distance support",
                      False, _DISTANCE), _DISTANCE,
             requires=("angle-amplitude-estimate", "distance-full-prior",
                       "distance-peak-interior",
                       "distance-endpoint-error-ok")),
    _profile("distance", "gh",
             _warrant(WarrantKind.BOUNDED_STATIONARY_SET,
                      "support-aware per-sample distance nodes", False,
                      _FRAMEWORK),
             "RIFT/likelihood/jax_ile/core.py:_distmarg_gh_logL",
             requires=("distance-volumetric-prior",)),
    _profile("time", "simpson",
             _warrant(WarrantKind.NONE, "fixed native time grid", False,
                      _TIME), _TIME),
    _profile("time", "bandlimited",
             _warrant(WarrantKind.EXACT_BAND_LIMIT,
                      "band-limited kappa with time-independent self term",
                      True, _TIME), _TIME,
             requires=("time-exact-band-limit", "time-independent-rho-sq",
                       "n-cal-one"),
             conflicts=("jax-direct-nonlinear-time",)),
)

JAX_SCHEME_PROFILES = MappingProxyType(
    {profile.key: profile for profile in _JAX_PROFILE_LIST})
JAX_DIRECT_MARGINALIZATION_AXES = ("angle", "distance", "time")


def make_jax_scheme_offer(axis, scheme, accuracy, resources, *,
                          provenance, requires=(), provides=(), conflicts=(),
                          conditional_requirements=()):
    """Attach measured request-specific evidence to a shipped scheme profile.

    Static incompatibilities cannot be removed here; callers may only add more
    restrictive request-specific facts.  This prevents an adapter from making
    an unsupported combination look runnable by omission.
    """
    key = "%s:%s" % (axis, scheme)
    try:
        profile = JAX_SCHEME_PROFILES[key]
    except KeyError:
        raise ValueError("unknown JAX direct-marginalization scheme %r" % key)
    return SchemeOffer(
        axis=axis, scheme=scheme, accuracy=accuracy, resources=resources,
        warrant=profile.warrant,
        provenance="%s; request evidence: %s" % (
            profile.provenance, provenance),
        requires=profile.requires.union(requires), provides=provides,
        conflicts=profile.conflicts.union(conflicts),
        conditional_requirements=(profile.conditional_requirements
                                  + tuple(conditional_requirements)))


def plan_jax_direct_marginalization(offers, error_budget, resource_budget, *,
                                    capabilities=(), allow_best_effort=False,
                                    required_axes=None, resource_model=None):
    """RIFT-specific entry point; still entirely opt-in and side-effect free.

    The static profile is rechecked here rather than trusted to the offer
    builder.  A caller may use :func:`plan_direct_marginalization` for an
    experimental catalog, but this entry point cannot be made to forget a
    shipped incompatibility by manually constructing a weaker offer.
    """
    axes = (JAX_DIRECT_MARGINALIZATION_AXES if required_axes is None
            else tuple(required_axes))
    offers = tuple(offers)
    for offer in offers:
        try:
            profile = JAX_SCHEME_PROFILES[offer.key]
        except KeyError:
            raise ValueError("unknown JAX direct-marginalization offer %r"
                             % offer.key)
        if offer.warrant != profile.warrant:
            raise ValueError("%s does not carry the shipped warrant profile"
                             % offer.key)
        if not profile.requires.issubset(offer.requires):
            raise ValueError("%s omits shipped requirements %r"
                             % (offer.key, sorted(
                                 profile.requires.difference(offer.requires))))
        if not profile.conflicts.issubset(offer.conflicts):
            raise ValueError("%s omits shipped conflicts %r"
                             % (offer.key, sorted(
                                 profile.conflicts.difference(offer.conflicts))))
        missing_conditionals = [
            requirement for requirement in profile.conditional_requirements
            if requirement not in offer.conditional_requirements]
        if missing_conditionals:
            raise ValueError("%s omits a shipped conditional requirement"
                             % offer.key)

    active_capabilities = set(capabilities)
    if "time" in axes and ("angle" in axes or "distance" in axes):
        # Every current JAX distance/angle wrapper calls
        # _validate_nonlinear_time_quadrature and refuses bandlimited: its
        # primitive fields would have to be refined before the nonlinear
        # marginalization.  This is an active execution-context fact, not a
        # capability callers should have to remember to declare.
        active_capabilities.add("jax-direct-nonlinear-time")
    return plan_direct_marginalization(
        offers, error_budget, resource_budget, required_axes=axes,
        capabilities=active_capabilities,
        allow_best_effort=allow_best_effort,
        resource_model=resource_model)
