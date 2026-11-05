"""Compatibility helpers for the evolving LALSimulation neutron-star API.

LALSimulation exposes legacy single-branch and released multibranch family
interfaces. The reviewed TOV development additionally provides multipart
equations of state through a separate constructor and accessor interface.
This module keeps capability checks in one place and gives
EOSManager a scalar interface which fails explicitly when a mass has twin-star
solutions.
"""

import math

import numpy as np


class NonFiniteFamilyObservableError(RuntimeError):
    """Raised when a native family solver returns a nonfinite observable."""


class AmbiguousFamilyBranchError(ValueError):
    """Raised when a mass belongs to more than one stable family branch."""


def validate_fixed_eos_branch_request(branch_id, eos_spec, using_eos_for_prior=False):
    """Fail closed when a scalar fixed-EOS branch request cannot be honored.

    EOS hyperprior plugins construct an EOS after the fixed-EOS setup path has
    run.  Until that plugin contract carries branch identity explicitly, a
    ``branch_id`` here would otherwise be accepted and silently ignored.
    """
    if branch_id is None:
        return
    if using_eos_for_prior:
        raise ValueError(
            "--using-eos-branch is not supported with --using-eos-for-prior; "
            "the EOS hyperprior plugin contract does not carry branch identity"
        )
    if eos_spec is None:
        raise ValueError("--using-eos-branch also requires --using-eos")


def evaluate_eos_lambda(eos, masses, use_vector=True, branch_id=None):
    """Evaluate tidal deformabilities, rejecting only failed mass samples.

    A vector evaluator may fail before returning any rows. Retry those rows
    individually so one native EOS exception cannot discard valid neighbors.
    """
    scalar = getattr(eos, "lambda_from_m", None)
    if not callable(scalar):
        raise TypeError("EOS must provide callable lambda_from_m")
    branch_kwargs = {"branch_id": branch_id} if branch_id is not None else {}
    masses = np.asarray(masses, dtype=float)
    flat = masses.reshape(-1)
    result = np.full(len(flat), -np.inf)
    finite_mass = np.isfinite(flat) & (flat > 0)
    vector = getattr(eos, "lambda_from_m_vector", None)
    if use_vector and callable(vector) and len(flat):
        try:
            values = np.asarray(vector(flat, **branch_kwargs), dtype=float)
            if values.shape == flat.shape:
                good = finite_mass & np.isfinite(values) & (values >= 0)
                result[good] = values[good]
                return float(result[0]) if masses.ndim == 0 else result.reshape(masses.shape)
        except Exception:
            pass
    for i in np.flatnonzero(finite_mass):
        try:
            value = float(scalar(float(flat[i]), **branch_kwargs))
            if math.isfinite(value) and value >= 0:
                result[i] = value
        except Exception:
            continue
    return float(result[0]) if masses.ndim == 0 else result.reshape(masses.shape)


class LALSimNeutronStarFamilyAdapter:
    """Version-neutral access to a LALSimulation neutron-star family.

    Parameters
    ----------
    eos:
        A SWIG ``LALSimNeutronStarEOS`` object.
    minimal:
        On multibranch APIs, request the fast family containing only mass,
        radius, and k2. Legacy single-branch APIs ignore this argument.
    reviewed_multibranch:
        True only when ``eos`` came from the reviewed
        ``SimNeutronStarEOSFromFilePhaseTransition`` loader. This selects the
        reviewed PT object ABI explicitly; ordinary EOS families use capability
        detection to distinguish legacy and released APIs.
    log_pressure_min:
        Optional lower natural-log central-pressure bound, ``ln(Pc / Pa)``, for
        the reviewed
        ``CreateSimNeutronStarFamilyPTWithPcmin`` constructor.
    lalsim_module:
        Dependency-injection hook used by the interface contract tests.
    """

    _REVIEWED_REQUIRED = (
        "CreateSimNeutronStarFamilyPT",
        "SimNeutronStarFamNumberOfBranches",
        "SimNeutronStarFamBranchMinMass",
        "SimNeutronStarFamBranchMaxMass",
        "SimNeutronStarFamBranchRadius",
        "SimNeutronStarFamBranchLoveNumberK2",
        "SimNeutronStarFamBranchCentralPressure",
    )

    _RELEASED_BRANCH_REQUIRED = (
        "SimNeutronStarFamMinMassPerBranch",
        "SimNeutronStarFamMaxMassPerBranch",
        "SimNeutronStarFamRadiusOfMassPerBranch",
        "SimNeutronStarFamLoveNumberK2OfMassPerBranch",
        "SimNeutronStarFamCentralPressureOfMassPerBranch",
    )

    @classmethod
    def _family_api(cls, lalsim_module, reviewed_multibranch):
        if reviewed_multibranch:
            cls._require_reviewed_api(lalsim_module)
            return "reviewed"
        if any(hasattr(lalsim_module, name) for name in cls._RELEASED_BRANCH_REQUIRED):
            required = cls._RELEASED_BRANCH_REQUIRED + (
                "CreateSimNeutronStarFamily", "SimNeutronStarFamNumberOfBranches",
            )
            missing = [name for name in required if not hasattr(lalsim_module, name)]
            if missing:
                raise RuntimeError(
                    "released LALSimulation multibranch API is incomplete; missing "
                    "symbols: {}".format(", ".join(missing))
                )
            return "released"
        return "legacy"

    @classmethod
    def _require_reviewed_api(cls, lalsim_module):
        """Validate the public ``stable_dev-TOV-headers`` SWIG surface."""
        present = [hasattr(lalsim_module, name) for name in cls._REVIEWED_REQUIRED]
        if not all(present):
            missing = [
                name for name, available in zip(cls._REVIEWED_REQUIRED, present)
                if not available
            ]
            raise RuntimeError(
                "reviewed LALSimulation phase-transition API is incomplete; "
                "missing symbols: {}"
                .format(", ".join(missing))
            )
        return True

    def __init__(self, eos, minimal=True, lalsim_module=None,
                 reviewed_multibranch=False, log_pressure_min=None):
        if lalsim_module is None:
            import lalsimulation as lalsim_module
        self.lalsim = lalsim_module
        self.eos = eos
        self._api = self._family_api(self.lalsim, reviewed_multibranch)
        self.is_multibranch_api = self._api != "legacy"
        if self._api == "reviewed":
            # The reviewed API requires ``min_fam``: 1 selects the PE-oriented
            # M/R/k2 solver, while 0 also constructs baryonic mass, k3, and k4.
            if log_pressure_min is None:
                self.family = self.lalsim.CreateSimNeutronStarFamilyPT(
                    eos, int(bool(minimal))
                )
            else:
                log_pressure_min = float(log_pressure_min)
                if not math.isfinite(log_pressure_min):
                    raise ValueError(
                        "log_pressure_min must be finite ln(Pc / Pa)"
                    )
                constructor = getattr(
                    self.lalsim, "CreateSimNeutronStarFamilyPTWithPcmin", None
                )
                if constructor is None:
                    raise NotImplementedError(
                        "reviewed LALSimulation build does not expose "
                        "CreateSimNeutronStarFamilyPTWithPcmin"
                    )
                self.family = constructor(
                    eos, int(bool(minimal)), log_pressure_min
                )
        elif self._api == "released":
            if log_pressure_min is None:
                self.family = self.lalsim.CreateSimNeutronStarFamily(eos, int(bool(minimal)))
            else:
                log_pressure_min = float(log_pressure_min)
                if not math.isfinite(log_pressure_min):
                    raise ValueError("log_pressure_min must be finite ln(Pc / Pa)")
                constructor = getattr(self.lalsim, "CreateSimNeutronStarFamilyWithPcmin", None)
                if constructor is None:
                    raise NotImplementedError(
                        "released LALSimulation build does not expose CreateSimNeutronStarFamilyWithPcmin"
                    )
                self.family = constructor(eos, int(bool(minimal)), log_pressure_min)
        else:
            self.family = self.lalsim.CreateSimNeutronStarFamily(eos)

    @classmethod
    def from_family(cls, family, lalsim_module=None, reviewed_multibranch=False):
        """Wrap an already-created family (mainly for tests and transition code)."""
        if lalsim_module is None:
            import lalsimulation as lalsim_module
        obj = cls.__new__(cls)
        obj.lalsim = lalsim_module
        obj.eos = None
        obj.family = family
        obj._api = cls._family_api(obj.lalsim, reviewed_multibranch)
        obj.is_multibranch_api = obj._api != "legacy"
        return obj

    @property
    def number_of_branches(self):
        if not self.is_multibranch_api:
            return 1
        return int(self.lalsim.SimNeutronStarFamNumberOfBranches(self.family))

    def minimum_mass(self, branch_id=None):
        if self.is_multibranch_api:
            if branch_id is None:
                return min(self.minimum_mass(k) for k in range(self.number_of_branches))
            self._validate_branch_id(branch_id)
            if self._api == "released":
                return self.lalsim.SimNeutronStarFamMinMassPerBranch(self.family, int(branch_id))
            return self.lalsim.SimNeutronStarFamBranchMinMass(
                int(branch_id), self.family
            )
        self._validate_legacy_branch_id(branch_id)
        return self.lalsim.SimNeutronStarFamMinimumMass(self.family)

    def maximum_mass(self, branch_id=None):
        if self.is_multibranch_api:
            if branch_id is None:
                return max(self.maximum_mass(k) for k in range(self.number_of_branches))
            self._validate_branch_id(branch_id)
            if self._api == "released":
                return self.lalsim.SimNeutronStarFamMaxMassPerBranch(self.family, int(branch_id))
            return self.lalsim.SimNeutronStarFamBranchMaxMass(
                int(branch_id), self.family
            )
        self._validate_legacy_branch_id(branch_id)
        return self.lalsim.SimNeutronStarMaximumMass(self.family)

    def branches_for_mass(self, mass_si):
        """Return every stable branch whose closed mass interval contains mass_si."""
        if not self.is_multibranch_api:
            return [0] if self.minimum_mass() <= mass_si <= self.maximum_mass() else []
        return [
            branch_id
            for branch_id in range(self.number_of_branches)
            if self.minimum_mass(branch_id) <= mass_si <= self.maximum_mass(branch_id)
        ]

    def resolve_branch(self, mass_si, branch_id=None):
        candidates = self.branches_for_mass(mass_si)
        if branch_id is not None:
            branch_id = int(branch_id)
            self._validate_branch_id(branch_id)
            if branch_id not in candidates:
                raise ValueError(
                    "mass {!r} kg is outside stable branch {} (available branches: {})".format(
                        mass_si, branch_id, candidates
                    )
                )
            return branch_id
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                "mass {!r} kg is outside every stable neutron-star branch".format(
                    mass_si
                )
            )
        raise AmbiguousFamilyBranchError(
            "mass {!r} kg has twin-star solutions on branches {}; pass branch_id explicitly".format(
                mass_si, candidates
            )
        )

    @staticmethod
    def _finite_observable(name, value):
        # Numerical failures must remain distinct from out-of-support ValueError.
        if not math.isfinite(value):
            raise NonFiniteFamilyObservableError(
                "LALSimulation returned nonfinite {} for a neutron-star family".format(name)
            )
        return value

    def radius(self, mass_si, branch_id=None):
        resolved = self.resolve_branch(mass_si, branch_id)
        if self._api == "released":
            value = self.lalsim.SimNeutronStarFamRadiusOfMassPerBranch(mass_si, self.family, resolved)
        elif self.is_multibranch_api:
            value = self.lalsim.SimNeutronStarFamBranchRadius(mass_si, resolved, self.family)
        else:
            value = self.lalsim.SimNeutronStarRadius(mass_si, self.family)
        return self._finite_observable("radius", value)

    def love_number_k2(self, mass_si, branch_id=None):
        resolved = self.resolve_branch(mass_si, branch_id)
        if self._api == "released":
            value = self.lalsim.SimNeutronStarFamLoveNumberK2OfMassPerBranch(mass_si, self.family, resolved)
        elif self.is_multibranch_api:
            value = self.lalsim.SimNeutronStarFamBranchLoveNumberK2(mass_si, resolved, self.family)
        else:
            value = self.lalsim.SimNeutronStarLoveNumberK2(mass_si, self.family)
        return self._finite_observable("Love number k2", value)

    def central_pressure(self, mass_si, branch_id=None):
        resolved = self.resolve_branch(mass_si, branch_id)
        if self._api == "released":
            value = self.lalsim.SimNeutronStarFamCentralPressureOfMassPerBranch(mass_si, self.family, resolved)
        elif self.is_multibranch_api:
            value = self.lalsim.SimNeutronStarFamBranchCentralPressure(mass_si, resolved, self.family)
        else:
            value = self.lalsim.SimNeutronStarCentralPressure(mass_si, self.family)
        return self._finite_observable("central pressure", value)

    def _validate_branch_id(self, branch_id):
        branch_id = int(branch_id)
        if branch_id < 0 or branch_id >= self.number_of_branches:
            raise ValueError(
                "branch_id {} outside [0, {})".format(
                    branch_id, self.number_of_branches
                )
            )

    @staticmethod
    def _validate_legacy_branch_id(branch_id):
        if branch_id not in (None, 0):
            raise ValueError("released LALSimulation family exposes only branch 0")


def create_family(eos, minimal=True, lalsim_module=None,
                  reviewed_multibranch=False, log_pressure_min=None):
    """Return a :class:`LALSimNeutronStarFamilyAdapter` for ``eos``."""
    return LALSimNeutronStarFamilyAdapter(
        eos, minimal=minimal, lalsim_module=lalsim_module,
        reviewed_multibranch=reviewed_multibranch,
        log_pressure_min=log_pressure_min,
    )
