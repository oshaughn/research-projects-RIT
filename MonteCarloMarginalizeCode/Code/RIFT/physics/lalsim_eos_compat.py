"""Compatibility helpers for the evolving LALSimulation neutron-star API.

LALSimulation exposes legacy single-branch and released multibranch family
interfaces.  The reviewed TOV development additionally provides multipart
equations of state through a separate constructor and accessor interface.
This module keeps capability checks in one place and gives EOSManager a scalar interface which fails explicitly when a mass has twin-star
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


def mass_in_eos_support(eos, m1_msun, m2_msun, bh1=False, bh2=False):
    """Per-row test against an EOS's published stellar mass bounds.

    A fixed EOS supports stars only on a bounded mass interval, and a family
    branch selected with ``--using-eos-branch`` is bounded at BOTH ends: a
    twin-star branch typically begins well above the family minimum mass.  A
    draw outside that interval describes no star, so it must carry no posterior
    weight.

    This test is made against MASSES on purpose.  The EOS reports "no star
    here" through lambda, and
    ``lalsimutils.convert_waveform_coordinates_with_eos`` returns only
    ``coord_names``: with a mass-only fit basis such as ``mc,eta`` the flag is
    discarded by the conversion and the row reaches the likelihood fit fully
    finite.  Callers must therefore carry this mask explicitly rather than
    inferring support from the converted coordinates.

    ``bh1``/``bh2`` mark an object as a black hole (``--no-matter1`` /
    ``--no-matter2``, or ``--assume-eos-but-primary-bh`` on export); a black
    hole is under no EOS constraint and is exempt.  A bound the EOS does not
    publish is read as unbounded, which leaves EOS classes exposing neither
    exactly as they were.

    Accepts scalars or arrays; returns a boolean of the broadcast shape.
    """
    m1_msun = np.asarray(m1_msun, dtype=float)
    m2_msun = np.asarray(m2_msun, dtype=float)
    ok = np.logical_and(np.isfinite(m1_msun), np.isfinite(m2_msun))
    m_min = getattr(eos, 'mMinMsun', None)
    m_max = getattr(eos, 'mMaxMsun', None)
    for m_here, is_bh in ((m1_msun, bh1), (m2_msun, bh2)):
        if is_bh:
            continue
        if m_max is not None:
            # Strict, to match the mMaxMsun test inside
            # lalsimutils.convert_waveform_coordinates_with_eos.
            ok = np.logical_and(ok, m_here < m_max)
        if m_min is not None:
            ok = np.logical_and(ok, m_here >= m_min)
    return ok


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
        A released SWIG ``LALSimNeutronStarEOS`` object, or a reviewed
        ``SimEOSMultiParts`` object when ``multipart`` is true.
    minimal:
        On multibranch APIs, request the fast family containing only mass,
        radius, and k2. Legacy single-branch APIs ignore this argument.
    multipart:
        Select the reviewed ``CreateSimNeutronStarFamilyPT`` path. This is
        explicit because reviewed builds still expose the legacy family API
        for ordinary named EOS objects.
    lalsim_module:
        Dependency-injection hook used by the interface contract tests.
    """

    _MULTIPART_REQUIRED = (
        "CreateSimNeutronStarFamilyPT",
        "SimNeutronStarFamNumberOfBranches",
        "SimNeutronStarFamBranchMinMass",
        "SimNeutronStarFamBranchMaxMass",
        "SimNeutronStarFamBranchRadius",
        "SimNeutronStarFamBranchLoveNumberK2",
    )

    _RELEASED_BRANCH_REQUIRED = (
        "SimNeutronStarFamMinMassPerBranch",
        "SimNeutronStarFamMaxMassPerBranch",
        "SimNeutronStarFamRadiusOfMassPerBranch",
        "SimNeutronStarFamLoveNumberK2OfMassPerBranch",
        "SimNeutronStarFamCentralPressureOfMassPerBranch",
    )

    @classmethod
    def _family_api(cls, lalsim_module, multipart):
        if multipart:
            cls._require_multibranch_api(lalsim_module)
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
    def _require_multibranch_api(cls, lalsim_module):
        present = [hasattr(lalsim_module, name) for name in cls._MULTIPART_REQUIRED]
        if not all(present):
            missing = [
                name for name, available in zip(cls._MULTIPART_REQUIRED, present)
                if not available
            ]
            raise RuntimeError(
                "reviewed LALSimulation multipart API is incomplete; missing "
                "symbols: {}"
                .format(", ".join(missing))
            )
        return True

    def __init__(self, eos, minimal=True, multipart=False, lalsim_module=None,
                 log_pressure_min=None):
        if lalsim_module is None:
            import lalsimulation as lalsim_module
        self.lalsim = lalsim_module
        self.eos = eos
        self._api = self._family_api(self.lalsim, multipart)
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
                    self.lalsim,
                    "CreateSimNeutronStarFamilyPTWithPcmin",
                    None,
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
    def from_family(cls, family, multipart=False, lalsim_module=None):
        """Wrap an already-created family (mainly for tests and transition code)."""
        if lalsim_module is None:
            import lalsimulation as lalsim_module
        obj = cls.__new__(cls)
        obj.lalsim = lalsim_module
        obj.eos = None
        obj.family = family
        obj._api = cls._family_api(obj.lalsim, multipart)
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
                fn = getattr(self.lalsim, "SimNeutronStarFamMinMass", None)
                if fn is not None:
                    return fn(self.family)
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
                fn = getattr(self.lalsim, "SimNeutronStarFamMaxMass", None)
                if fn is not None:
                    return fn(self.family)
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
        # This must not be a ValueError: EOSBranchView reserves those for
        # out-of-support samples and converts them to posterior rejection flags.
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
            modern = getattr(self.lalsim, "SimNeutronStarFamBranchCentralPressure", None)
            if modern is None:
                raise NotImplementedError(
                    "reviewed LALSimulation family API lacks branch-specific central pressure"
                )
            value = modern(mass_si, resolved, self.family)
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


def create_family(eos, minimal=True, multipart=False, lalsim_module=None,
                  log_pressure_min=None):
    """Return a :class:`LALSimNeutronStarFamilyAdapter` for ``eos``."""
    return LALSimNeutronStarFamilyAdapter(
        eos, minimal=minimal, multipart=multipart,
        lalsim_module=lalsim_module, log_pressure_min=log_pressure_min
    )
