"""Compatibility helpers for the evolving LALSimulation neutron-star API.

Released LALSimulation versions expose a single-branch
``LALSimNeutronStarFamily`` interface.  The reviewed TOV development adds
multipart equations of state, multiple stable branches, and branch-indexed
interpolators.  This module keeps the version checks in one place and gives
EOSManager a scalar interface which fails explicitly when a mass has twin-star
solutions.
"""


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


class LALSimNeutronStarFamilyAdapter:
    """Version-neutral access to a LALSimulation neutron-star family.

    Parameters
    ----------
    eos:
        A SWIG ``LALSimNeutronStarEOS`` object.
    minimal:
        On the reviewed API, request the fast family containing only mass,
        radius, and k2.  Released APIs do not have this argument and ignore it.
    reviewed_multibranch:
        True only when ``eos`` came from
        ``SimNeutronStarEOSFromFilePhaseTransition``. Reviewed and legacy SWIG
        objects coexist in the same module, so module-level symbol detection
        cannot safely choose the family constructor.
    log_pressure_min:
        Optional lower log-central-pressure bound for the reviewed
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
        self.is_multibranch_api = bool(reviewed_multibranch)
        if self.is_multibranch_api:
            self._require_reviewed_api(self.lalsim)
            # The reviewed API requires ``min_fam``: 1 selects the PE-oriented
            # M/R/k2 solver, while 0 also constructs baryonic mass, k3, and k4.
            if log_pressure_min is None:
                self.family = self.lalsim.CreateSimNeutronStarFamilyPT(
                    eos, int(bool(minimal))
                )
            else:
                constructor = getattr(
                    self.lalsim, "CreateSimNeutronStarFamilyPTWithPcmin", None
                )
                if constructor is None:
                    raise NotImplementedError(
                        "reviewed LALSimulation build does not expose "
                        "CreateSimNeutronStarFamilyPTWithPcmin"
                    )
                self.family = constructor(
                    eos, int(bool(minimal)), float(log_pressure_min)
                )
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
        obj.is_multibranch_api = bool(reviewed_multibranch)
        if obj.is_multibranch_api:
            cls._require_reviewed_api(obj.lalsim)
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

    def radius(self, mass_si, branch_id=None):
        resolved = self.resolve_branch(mass_si, branch_id)
        if self.is_multibranch_api:
            return self.lalsim.SimNeutronStarFamBranchRadius(
                mass_si, resolved, self.family
            )
        return self.lalsim.SimNeutronStarRadius(mass_si, self.family)

    def love_number_k2(self, mass_si, branch_id=None):
        resolved = self.resolve_branch(mass_si, branch_id)
        if self.is_multibranch_api:
            return self.lalsim.SimNeutronStarFamBranchLoveNumberK2(
                mass_si, resolved, self.family
            )
        return self.lalsim.SimNeutronStarLoveNumberK2(mass_si, self.family)

    def central_pressure(self, mass_si, branch_id=None):
        resolved = self.resolve_branch(mass_si, branch_id)
        if self.is_multibranch_api:
            return self.lalsim.SimNeutronStarFamBranchCentralPressure(
                mass_si, resolved, self.family
            )
        return self.lalsim.SimNeutronStarCentralPressure(mass_si, self.family)

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
