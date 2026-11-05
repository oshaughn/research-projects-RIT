import json

import lal
import numpy as np
import pytest

from RIFT.physics.lalsim_eos_compat import (
    AmbiguousFamilyBranchError,
    LALSimNeutronStarFamilyAdapter,
    NonFiniteFamilyObservableError,
    create_family,
    evaluate_eos_lambda,
    validate_fixed_eos_branch_request,
)


class LegacyLALSimulation:
    def __init__(self):
        self.create_calls = []
        self.file_calls = []

    def SimNeutronStarEOSFromFile(self, fname):
        self.file_calls.append((fname, 0))
        return "clean-eos"

    def CreateSimNeutronStarFamily(self, eos):
        self.create_calls.append((eos,))
        return "legacy-family"

    def SimNeutronStarFamMinimumMass(self, family):
        return 1.0

    def SimNeutronStarMaximumMass(self, family):
        return 3.0

    def SimNeutronStarRadius(self, mass, family):
        return 10.0 + mass

    def SimNeutronStarLoveNumberK2(self, mass, family):
        return 0.1 * mass

    def SimNeutronStarCentralPressure(self, mass, family):
        return 100.0 * mass


class MultibranchLALSimulation:
    bounds = ((1.0, 2.0), (1.5, 3.0))

    def __init__(self):
        self.create_calls = []
        self.file_calls = []

    def SimNeutronStarEOSFromFilePhaseTransition(self, fname):
        self.file_calls.append((fname,))
        return "multipart-eos"

    def CreateSimNeutronStarFamilyPT(self, eos, min_fam):
        self.create_calls.append((eos, min_fam))
        return "multibranch-family"

    def CreateSimNeutronStarFamilyPTWithPcmin(self, eos, min_fam, log_pc_min):
        self.create_calls.append((eos, min_fam, log_pc_min))
        return "multibranch-family-pcmin"

    def SimNeutronStarFamNumberOfBranches(self, family):
        return len(self.bounds)

    def SimNeutronStarFamBranchMinMass(self, branch_id, family):
        return self.bounds[branch_id][0]

    def SimNeutronStarFamBranchMaxMass(self, branch_id, family):
        return self.bounds[branch_id][1]

    def SimNeutronStarFamBranchRadius(self, mass, branch_id, family):
        return 10.0 * branch_id + mass

    def SimNeutronStarFamBranchLoveNumberK2(self, mass, branch_id, family):
        return branch_id + 0.1 * mass

    def SimNeutronStarFamBranchCentralPressure(self, mass, branch_id, family):
        return 100.0 * branch_id + mass


class StellarMassMultibranchLALSimulation(MultibranchLALSimulation):
    bounds = tuple(
        (lower * lal.MSUN_SI, upper * lal.MSUN_SI)
        for lower, upper in ((1.0, 2.0), (1.3, 3.0))
    )

    def SimNeutronStarEOSMultiPartsPseudoEnthalpyOfPressure(self, pressure, eos):
        return pressure

    def SimNeutronStarEOSMultiPartsMaxPseudoEnthalpy(self, eos):
        return 1.0

    def SimNeutronStarEOSMultiPartsSpeedOfSoundOfPseudoEnthalpy(
        self, enthalpy, eos
    ):
        return 0.5 * lal.C_SI


class CoexistingLALSimulation(MultibranchLALSimulation):
    """Reviewed builds retain the released EOS/family object family too."""

    def SimNeutronStarEOSFromFile(self, fname):
        self.file_calls.append(("legacy", fname))
        return "legacy-file-eos"

    def CreateSimNeutronStarFamily(self, eos):
        self.create_calls.append(("legacy", eos))
        return "legacy-family"

    def SimNeutronStarFamMinimumMass(self, family):
        return 1.0

    def SimNeutronStarMaximumMass(self, family):
        return 3.0

    def SimNeutronStarRadius(self, mass, family):
        return 10.0 + mass

    def SimNeutronStarLoveNumberK2(self, mass, family):
        return 0.1 * mass

    def SimNeutronStarCentralPressure(self, mass, family):
        return 100.0 * mass


def test_released_lalsimulation_uses_one_argument_family_api():
    lalsim = LegacyLALSimulation()
    family = LALSimNeutronStarFamilyAdapter(
        "eos", minimal=True, lalsim_module=lalsim
    )

    assert lalsim.create_calls == [("eos",)]
    assert family.number_of_branches == 1
    assert family.branches_for_mass(2.0) == [0]
    assert family.radius(2.0) == 12.0
    assert family.love_number_k2(2.0) == pytest.approx(0.2)
    assert family.central_pressure(2.0) == 200.0
    with pytest.raises(ValueError, match="branch_id 1 outside"):
        family.radius(2.0, branch_id=1)


def test_eosmanager_file_loader_keeps_released_api_fallback(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = LegacyLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    eos = EOSManager.EOSLALSimulationFromFile("released-format.dat")

    assert fake_lalsim.file_calls == [("released-format.dat", 0)]
    assert fake_lalsim.create_calls == [("clean-eos",)]
    assert eos._get_lalsim_family_adapter().number_of_branches == 1


def test_reviewed_lalsimulation_uses_minimal_multibranch_api():
    lalsim = MultibranchLALSimulation()
    family = LALSimNeutronStarFamilyAdapter(
        "eos", minimal=True, lalsim_module=lalsim,
        reviewed_multibranch=True,
    )

    assert lalsim.create_calls == [("eos", 1)]
    assert family.number_of_branches == 2
    assert family.minimum_mass() == 1.0
    assert family.maximum_mass() == 3.0
    assert family.branches_for_mass(1.25) == [0]
    assert family.branches_for_mass(1.75) == [0, 1]
    assert family.radius(1.75, branch_id=1) == 11.75
    assert family.love_number_k2(1.75, branch_id=1) == pytest.approx(1.175)
    assert family.central_pressure(1.75, branch_id=1) == 101.75

    with pytest.raises(ValueError, match=r"finite ln\(Pc / Pa\)"):
        LALSimNeutronStarFamilyAdapter(
            "eos",
            lalsim_module=lalsim,
            reviewed_multibranch=True,
            log_pressure_min=np.inf,
        )


def test_reviewed_module_keeps_legacy_family_dispatch_explicit():
    lalsim = CoexistingLALSimulation()
    family = LALSimNeutronStarFamilyAdapter(
        "legacy-eos", lalsim_module=lalsim, reviewed_multibranch=False
    )

    assert lalsim.create_calls == [("legacy", "legacy-eos")]
    assert family.number_of_branches == 1
    assert family.radius(2.0) == 12.0


def test_partial_reviewed_api_fails_diagnostically(monkeypatch):
    lalsim = MultibranchLALSimulation()
    monkeypatch.delattr(
        MultibranchLALSimulation,
        "SimNeutronStarFamBranchLoveNumberK2",
    )
    with pytest.raises(RuntimeError, match="phase-transition API is incomplete"):
        LALSimNeutronStarFamilyAdapter(
            "eos", lalsim_module=lalsim, reviewed_multibranch=True
        )


def test_twin_star_mass_requires_an_explicit_branch():
    family = LALSimNeutronStarFamilyAdapter(
        "eos", lalsim_module=MultibranchLALSimulation(),
        reviewed_multibranch=True,
    )

    with pytest.raises(AmbiguousFamilyBranchError, match=r"branches \[0, 1\]"):
        family.radius(1.75)
    with pytest.raises(ValueError, match="outside stable branch 0"):
        family.radius(2.5, branch_id=0)
    with pytest.raises(ValueError, match="outside every stable"):
        family.radius(4.0)


def test_eosmanager_file_loader_routes_reviewed_phase_transition_api(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = MultibranchLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    eos = EOSManager.EOSLALSimulationFromFile(
        "new-format.dat", dirty_phase_transitions=True
    )

    assert fake_lalsim.file_calls == [("new-format.dat",)]
    assert fake_lalsim.create_calls == [("multipart-eos", 1)]
    assert eos.eos == "multipart-eos"
    assert eos._get_lalsim_family_adapter().number_of_branches == 2

    extended = EOSManager.EOSLALSimulationFromFile(
        "extended-format.dat", minimal_family=False
    )
    assert fake_lalsim.file_calls[-1] == ("extended-format.dat",)
    assert fake_lalsim.create_calls[-1] == ("multipart-eos", 0)

    pressure_floor = EOSManager.EOSLALSimulationFromFile(
        "pressure-floor.dat", family_log_pressure_min=12.5
    )
    assert pressure_floor.eos_fam == "multibranch-family-pcmin"
    assert fake_lalsim.create_calls[-1] == ("multipart-eos", 1, 12.5)

    for invalid in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match=r"finite ln\(Pc / Pa\)"):
            EOSManager.EOSLALSimulationFromFile(
                "pressure-floor.dat", family_log_pressure_min=invalid
            )


def test_eosmanager_file_loader_preserves_legacy_default(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = CoexistingLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    eos = EOSManager.EOSLALSimulationFromFile("ordinary-two-column.dat")

    assert fake_lalsim.file_calls == [
        ("legacy", "ordinary-two-column.dat")
    ]
    assert fake_lalsim.create_calls == [("legacy", "legacy-file-eos")]
    assert eos._get_lalsim_family_adapter().number_of_branches == 1


def test_eosmanager_smoke_with_installed_released_lalsimulation():
    from RIFT.physics import EOSManager

    eos = EOSManager.EOSLALSimulation("SLy")
    assert eos.branches_for_m(1.4) == [0]
    assert eos.radius_from_m(1.4) > 0.0
    assert np.isfinite(eos.lambda_from_m(1.4))


def test_kedia_parametric_eos_scalar_interfaces_remain_compatible():
    from RIFT.physics import EOSManager

    spectral = EOSManager.EOSLindblomSpectral(
        name="contract-spectral",
        spec_params=dict(gamma1=1.0, gamma2=1.0, gamma3=0.0, gamma4=0.0),
        use_lal_spec_eos=True,
    )
    piecewise = EOSManager.EOSPiecewisePolytrope(
        name="contract-piecewise",
        param_dict=dict(
            logP1=34.269, gamma1=2.830, gamma2=3.445, gamma3=3.348
        ),
    )

    assert np.isfinite(spectral.lambda_from_m(1.4))
    native_k2 = EOSManager.lalsim.SimNeutronStarLoveNumberK2(1.4 * lal.MSUN_SI, piecewise.eos_fam)
    if np.isfinite(native_k2):
        assert np.isfinite(piecewise.lambda_from_m(1.4))
    else:
        # A native solver failure must raise, never enter a posterior as NaN.
        assert np.isneginf(piecewise.lambda_from_m_vector(np.array([1.4, 1.4]))).all()


def test_selected_branch_view_preserves_legacy_scalar_consumer_api(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = StellarMassMultibranchLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    eos = EOSManager.EOSLALSimulationFromFile(
        "twin-star.dat", phase_transition_aware=True
    )

    primary = eos.for_branch(0)
    secondary = eos.for_branch(1)
    assert primary.mMaxMsun == pytest.approx(2.0)
    assert secondary.mMaxMsun == pytest.approx(3.0)
    assert secondary.branches_for_m(1.75) == [1]
    assert secondary.radius_from_m(1.75) == pytest.approx(
        10.0 + 1.75 * lal.MSUN_SI
    )
    assert np.isfinite(secondary.lambda_from_m(1.75))
    assert primary.lambda_from_m(2.5) == pytest.approx(1e-8)

    curves = EOSManager.make_mr_lambda_lal_branches(
        eos.eos, n_bins=3, multipart=True
    )
    assert set(curves) == {0, 1}
    assert curves[0].shape == (3, 3)

    explicit = EOSManager.make_mr_lambda_lal(
        eos.eos, n_bins=3, branch_id=1, multipart=True
    )
    assert explicit.shape == (3, 3)

    with pytest.warns(DeprecationWarning, match="use multipart"):
        alias = EOSManager.make_mr_lambda_lal(
            eos.eos,
            n_bins=3,
            branch_id=1,
            reviewed_multibranch=True,
        )
    assert alias.shape == (3, 3)

    with pytest.warns(DeprecationWarning, match="use multipart"):
        alias_branches = EOSManager.make_mr_lambda_lal_branches(
            eos.eos, n_bins=3, reviewed_multibranch=True
        )
    assert set(alias_branches) == {0, 1}

    for helper, kwargs in (
        (EOSManager.make_mr_lambda_lal, {"branch_id": 1}),
        (EOSManager.make_mr_lambda_lal_branches, {}),
    ):
        with pytest.warns(DeprecationWarning, match="use multipart"):
            with pytest.raises(ValueError, match="conflicting multipart"):
                helper(
                    eos.eos,
                    n_bins=3,
                    multipart=True,
                    reviewed_multibranch=False,
                    **kwargs
                )


def test_mr_lambda_branches_helper_keeps_legacy_default(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = CoexistingLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    curves = EOSManager.make_mr_lambda_lal_branches("legacy-eos", n_bins=3)

    assert set(curves) == {0}
    assert curves[0].shape == (3, 3)
    assert fake_lalsim.create_calls == [("legacy", "legacy-eos")]


def test_selected_branch_view_preserves_branch_sensitive_helpers(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = StellarMassMultibranchLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    secondary = EOSManager.EOSLALSimulationFromFile(
        "twin-star.dat", phase_transition_aware=True
    ).for_branch(1)

    expected_radius_km = (10.0 + 1.4 * lal.MSUN_SI) / 1e3
    assert secondary.estimate_baryon_mass_from_mg(1.4) == pytest.approx(
        1.4 + 1.4**2 / expected_radius_km
    )
    assert secondary.test_speed_of_sound_causal()
    assert secondary.test_speed_of_sound_causal(test_only_under_mmax=False)


def test_selected_branch_helpers_fail_closed_when_branch_data_are_missing(monkeypatch):
    from RIFT.physics import EOSManager

    no_reference_star = StellarMassMultibranchLALSimulation()
    no_reference_star.bounds = tuple(
        (lower * lal.MSUN_SI, upper * lal.MSUN_SI)
        for lower, upper in MultibranchLALSimulation.bounds
    )
    monkeypatch.setattr(EOSManager, "lalsim", no_reference_star)
    secondary = EOSManager.EOSLALSimulationFromFile(
        "twin-star.dat", phase_transition_aware=True
    ).for_branch(1)
    with pytest.raises(ValueError, match="does not contain the 1.4-Msun"):
        secondary.estimate_baryon_mass_from_mg(1.6)

    monkeypatch.delattr(
        StellarMassMultibranchLALSimulation,
        "SimNeutronStarEOSMultiPartsSpeedOfSoundOfPseudoEnthalpy",
    )
    assert secondary.test_speed_of_sound_causal() is False


def test_multipart_full_table_causality_requires_max_enthalpy(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = StellarMassMultibranchLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    secondary = EOSManager.EOSLALSimulationFromFile(
        "twin-star.dat", phase_transition_aware=True
    ).for_branch(1)
    monkeypatch.delattr(
        StellarMassMultibranchLALSimulation,
        "SimNeutronStarEOSMultiPartsMaxPseudoEnthalpy",
    )
    assert secondary.test_speed_of_sound_causal(
        test_only_under_mmax=False
    ) is False


def test_eos_hyperprior_rejects_fixed_branch_request():
    with pytest.raises(ValueError, match="not supported with --using-eos-for-prior"):
        validate_fixed_eos_branch_request(1, "file:eos-draws.dat", True)

    assert validate_fixed_eos_branch_request(1, "lalsim_file:eos.dat") is None
    assert validate_fixed_eos_branch_request(None, "file:eos-draws.dat", True) is None


def test_nmb_sequence_dispatch_and_accessors_remain_compatible(tmp_path):
    h5py = pytest.importorskip("h5py")
    from RIFT.physics import EOSManager

    path = tmp_path / "nmb-sequence.h5"
    fields = ["M", "R", "Lambda", "stable"]
    sequence = np.array(
        [[[1.0, 12.0, 500.0, 1.0],
          [1.4, 11.5, 300.0, 1.0],
          [2.0, 10.0, 50.0, 1.0]]]
    )
    with h5py.File(path, "w") as stream:
        stream.attrs["representation"] = "tabular_hc/1"
        stream.attrs["schema_version"] = "nmbackend.nss/1"
        stream.attrs["fields"] = json.dumps(fields)
        stream.create_dataset("sequence", data=sequence)

    eos_sequence = EOSManager.EOSSequenceFromFile(
        fname=str(path), load_ns=True, no_sort=True
    )
    assert isinstance(eos_sequence, EOSManager.EOSSequenceNMB)
    assert eos_sequence.m_max_of_indx(0) == pytest.approx(2.0)
    assert eos_sequence.R_of_m_indx(1.4, 0) == pytest.approx(11.5)
    assert eos_sequence.lambda_of_m_indx(1.4, 0) == pytest.approx(300.0)


def test_nmb_primary_branch_contract_does_not_mix_disconnected_branches(tmp_path):
    h5py = pytest.importorskip("h5py")
    from RIFT.physics import EOSManager

    path = tmp_path / "nmb-twin-sequence.h5"
    fields = ["hc", "M", "R", "Lambda", "stable"]
    sequence = np.array(
        [[[0.1, 1.0, 13.0, 600.0, 1.0],
          [0.2, 2.0, 11.0, 100.0, 1.0],
          [0.3, 1.8, 10.8, 80.0, 0.0],
          [0.4, 1.6, 10.0, 60.0, 1.0],
          [0.5, 2.1, 9.0, 20.0, 1.0]]]
    )
    with h5py.File(path, "w") as stream:
        stream.attrs["representation"] = "tabular_hc/1"
        stream.attrs["schema_version"] = "nmbackend.nss/1"
        stream.attrs["fields"] = json.dumps(fields)
        stream.create_dataset("sequence", data=sequence)

    eos_sequence = EOSManager.EOSSequenceFromFile(
        fname=str(path), load_ns=True, no_sort=True
    )
    assert eos_sequence.stable_branch_counts[0] == 2
    assert eos_sequence.m_max_of_indx(0) == pytest.approx(2.0)
    expected_primary_radius = np.exp(
        np.interp(1.8, [1.0, 2.0], np.log([13.0, 11.0]))
    )
    assert eos_sequence.R_of_m_indx(1.8, 0) == pytest.approx(
        expected_primary_radius
    )


class ReleasedMultibranchLALSimulation(LegacyLALSimulation):
    """Released family ABI uses family-before-branch arguments."""
    bounds = ((1.0, 2.0), (1.5, 3.0))

    def CreateSimNeutronStarFamily(self, eos, min_fam):
        self.create_calls.append((eos, min_fam))
        return "released-family"

    def CreateSimNeutronStarFamilyWithPcmin(self, eos, min_fam, logPcmin):
        self.create_calls.append((eos, min_fam, logPcmin))
        return "released-family"

    def SimNeutronStarFamNumberOfBranches(self, family):
        assert family == "released-family"
        return len(self.bounds)

    def SimNeutronStarFamMinMassPerBranch(self, family, branch_id):
        assert family == "released-family"
        return self.bounds[branch_id][0]

    def SimNeutronStarFamMaxMassPerBranch(self, family, branch_id):
        assert family == "released-family"
        return self.bounds[branch_id][1]

    def SimNeutronStarFamRadiusOfMassPerBranch(self, mass, family, branch_id):
        assert family == "released-family"
        return mass + 10.0 * branch_id

    def SimNeutronStarFamLoveNumberK2OfMassPerBranch(self, mass, family, branch_id):
        assert family == "released-family"
        return .1 * mass + branch_id

    def SimNeutronStarFamCentralPressureOfMassPerBranch(self, mass, family, branch_id):
        assert family == "released-family"
        return 100.0 * mass + branch_id

    def SimNeutronStarRadius(self, mass, family):
        raise AssertionError("released multibranch family used a legacy scalar accessor")

    SimNeutronStarLoveNumberK2 = SimNeutronStarRadius
    SimNeutronStarCentralPressure = SimNeutronStarRadius


@pytest.mark.parametrize("minimal", [False, True])
def test_released_multibranch_family_preserves_branch_semantics(minimal):
    lalsim = ReleasedMultibranchLALSimulation()
    family = LALSimNeutronStarFamilyAdapter("eos", minimal=minimal, lalsim_module=lalsim)
    assert lalsim.create_calls == [("eos", int(minimal))]
    assert family.number_of_branches == 2
    assert family.minimum_mass() == 1.0
    assert family.maximum_mass() == 3.0
    assert family.branches_for_mass(1.75) == [0, 1]
    with pytest.raises(AmbiguousFamilyBranchError):
        family.radius(1.75)
    assert family.radius(1.75, branch_id=1) == 11.75
    assert family.love_number_k2(1.75, branch_id=1) == pytest.approx(1.175)
    assert family.central_pressure(1.75, branch_id=1) == 176.0
    assert family.radius(1.25) == 1.25
    with pytest.raises(ValueError, match="outside stable branch"):
        family.radius(1.25, branch_id=1)


def test_released_multibranch_existing_family_and_pressure_floor():
    lalsim = ReleasedMultibranchLALSimulation()
    family = LALSimNeutronStarFamilyAdapter.from_family("released-family", lalsim_module=lalsim)
    assert lalsim.create_calls == []
    assert family.number_of_branches == 2
    assert family.radius(1.75, branch_id=1) == 11.75
    LALSimNeutronStarFamilyAdapter("eos", minimal=False, log_pressure_min=75.5, lalsim_module=lalsim)
    assert lalsim.create_calls == [("eos", 0, 75.5)]
    with pytest.raises(ValueError, match="finite"):
        LALSimNeutronStarFamilyAdapter("eos", log_pressure_min=float('nan'), lalsim_module=lalsim)


def test_partial_released_multibranch_api_is_not_treated_as_legacy(monkeypatch):
    lalsim = ReleasedMultibranchLALSimulation()
    monkeypatch.delattr(ReleasedMultibranchLALSimulation, "SimNeutronStarFamMaxMassPerBranch")
    with pytest.raises(RuntimeError, match="released.*incomplete.*MaxMassPerBranch"):
        LALSimNeutronStarFamilyAdapter("eos", lalsim_module=lalsim)
    assert lalsim.create_calls == []


def test_released_constructor_domain_error_is_not_retried():
    class InvalidEOS(ReleasedMultibranchLALSimulation):
        def CreateSimNeutronStarFamily(self, *args):
            self.create_calls.append(args)
            raise TypeError("invalid EOS domain")
    lalsim = InvalidEOS()
    with pytest.raises(TypeError, match="invalid EOS domain"):
        LALSimNeutronStarFamilyAdapter("eos", lalsim_module=lalsim)
    assert lalsim.create_calls == [("eos", 1)]


@pytest.mark.parametrize("api", ["legacy", "reviewed", "released"])
def test_nonfinite_native_observables_fail_explicitly(monkeypatch, api):
    implementations = dict(legacy=LegacyLALSimulation, reviewed=MultibranchLALSimulation,
                           released=ReleasedMultibranchLALSimulation)
    lalsim = implementations[api]()
    family = LALSimNeutronStarFamilyAdapter("eos", reviewed_multibranch=api == "reviewed", lalsim_module=lalsim)
    names = dict(
        legacy=("SimNeutronStarRadius", "SimNeutronStarLoveNumberK2", "SimNeutronStarCentralPressure"),
        reviewed=("SimNeutronStarFamBranchRadius", "SimNeutronStarFamBranchLoveNumberK2", "SimNeutronStarFamBranchCentralPressure"),
        released=("SimNeutronStarFamRadiusOfMassPerBranch", "SimNeutronStarFamLoveNumberK2OfMassPerBranch", "SimNeutronStarFamCentralPressureOfMassPerBranch"))
    for accessor, method in zip(names[api], (family.radius, family.love_number_k2, family.central_pressure)):
        for bad in (float('nan'), float('inf')):
            monkeypatch.setattr(lalsim, accessor, lambda *args, result=bad: result)
            with pytest.raises(NonFiniteFamilyObservableError, match="nonfinite"):
                method(1.25, branch_id=0)


@pytest.mark.parametrize("bad_object", [0, 1])
def test_nonfinite_k2_is_rejected_by_branch_batch_and_conversion(monkeypatch, bad_object):
    from RIFT import lalsimutils
    from RIFT.physics import EOSManager
    monkeypatch.setattr(EOSManager, "lalsim", StellarMassMultibranchLALSimulation())
    view = EOSManager.EOSLALSimulationFromFile(
        "twin-star.dat", phase_transition_aware=True).for_branch(1)
    native = view.source._get_lalsim_family_adapter().lalsim
    monkeypatch.setattr(native, "SimNeutronStarFamBranchLoveNumberK2", lambda *args: float('nan'))
    assert np.isneginf(view.lambda_from_m_vector(np.array([1.75, 1.4]))).all()
    rejected = lalsimutils.convert_waveform_coordinates_with_eos(
        np.array([[1.75, 1.4]]), coord_names=['lambda1', 'lambda2'],
        low_level_coord_names=['m1', 'm2'], eos_class=view,
        no_matter1=bad_object == 1, no_matter2=bad_object == 0)
    assert np.isneginf(rejected).all()
    with pytest.raises(ValueError, match="outside stable branch"):
        view.lambda_from_m(1.1)
    rejected = lalsimutils.convert_waveform_coordinates_with_eos(
        np.array([[1.1, 1.1]]), coord_names=['lambda1', 'lambda2'],
        low_level_coord_names=['m1', 'm2'], eos_class=view,
        no_matter1=bad_object == 1, no_matter2=bad_object == 0)
    assert rejected[0, bad_object] == -np.inf
    assert np.isneginf(rejected).all()


def test_o4c_reviewed_public_positional_contracts_are_preserved():
    native = MultibranchLALSimulation()
    direct = LALSimNeutronStarFamilyAdapter("eos", False, native, True, 75.5)
    wrapped = create_family("eos", False, native, True, 75.5)
    existing = LALSimNeutronStarFamilyAdapter.from_family("existing-family", native, True)
    assert native.create_calls == [("eos", 0, 75.5), ("eos", 0, 75.5)]
    for family in (direct, wrapped, existing):
        assert family.number_of_branches == 2
        assert family.radius(1.75, branch_id=1) == 11.75


class MixedEvaluationEOS:
    mMinMsun = 1.
    mMaxMsun = 3.

    def __init__(self):
        self.visited = []
        self.vector_calls = 0

    def lambda_from_m(self, mass):
        mass = mass/lal.MSUN_SI if mass > 1e15 else mass
        self.visited.append(mass)
        if np.isclose(mass, 1.5):
            raise RuntimeError("native EOS evaluation failed")
        if np.isclose(mass, 1.6):
            return np.nan
        if np.isclose(mass, 1.7):
            return np.inf
        return 10*mass

    def lambda_from_m_vector(self, masses):
        self.vector_calls += 1
        return np.array([self.lambda_from_m(mass) for mass in masses])


@pytest.mark.parametrize("bad_object", [0, 1])
@pytest.mark.parametrize("coordinates", [['mc', 'eta'], ['lambda1', 'lambda2']])
@pytest.mark.parametrize("backstop", [None])
def test_mixed_eos_failures_reject_rows_and_preserve_neighbors(bad_object, coordinates, backstop):
    from RIFT import lalsimutils
    masses = np.array([1.8, 1.5, 1.6, 1.7, 1.9])
    points = np.column_stack([masses, np.full(5, 1.2)]) if bad_object == 0 else np.column_stack([np.full(5, 1.95), masses])
    eos = MixedEvaluationEOS()
    kwargs = {} if backstop is None else dict(backstop_novector=backstop)
    result = lalsimutils.convert_waveform_coordinates_with_eos(points,
        coord_names=coordinates, low_level_coord_names=['m1', 'm2'], eos_class=eos,
        no_matter1=bad_object == 1, no_matter2=bad_object == 0, **kwargs)
    assert np.isneginf(result[1:4]).all()
    assert np.isfinite(result[[0, 4]]).all()
    assert any(np.isclose(mass, 1.9) for mass in eos.visited)
    if coordinates == ['lambda1', 'lambda2']:
        np.testing.assert_allclose(result[[0, 4], bad_object], [18., 19.])
        np.testing.assert_array_equal(result[[0, 4], 1-bad_object], 0.)
    else:
        expected = lalsimutils.convert_waveform_coordinates(points[[0, 4]],
            coord_names=coordinates, low_level_coord_names=['m1', 'm2'])
        np.testing.assert_allclose(result[[0, 4]], expected)


def test_vector_eos_nonfinite_results_do_not_remove_finite_rows():
    class VectorEOS(MixedEvaluationEOS):
        def lambda_from_m_vector(self, masses):
            return np.array([18., np.nan, np.inf, 19.])
    np.testing.assert_array_equal(evaluate_eos_lambda(VectorEOS(), [1.8, 1.6, 1.7, 1.9]),
                                  [18., -np.inf, -np.inf, 19.])
    with pytest.raises(TypeError, match="lambda_from_m"):
        evaluate_eos_lambda(object(), [1.4])


@pytest.mark.parametrize("failure", ['raise', 'nan', 'inf'])
def test_native_eos_batch_wrappers_continue_after_bad_mass(monkeypatch, failure):
    from RIFT.physics import EOSManager
    native = StellarMassMultibranchLALSimulation()
    native.bounds = ((1.*lal.MSUN_SI, 3.*lal.MSUN_SI),)
    real_k2 = native.SimNeutronStarFamBranchLoveNumberK2
    def with_failure(mass, branch, family):
        if np.isclose(mass/lal.MSUN_SI, 1.6):
            if failure == 'raise':
                raise RuntimeError('bad native mass')
            return np.nan if failure == 'nan' else np.inf
        return real_k2(mass, branch, family)
    monkeypatch.setattr(native, 'SimNeutronStarFamBranchLoveNumberK2', with_failure)
    monkeypatch.setattr(EOSManager, 'lalsim', native)
    base = EOSManager.EOSLALSimulationFromFile('single-branch.dat', phase_transition_aware=True)
    for eos in (base, base.for_branch(0)):
        expected = [eos.lambda_from_m(1.4), eos.lambda_from_m(1.8)]
        values = eos.lambda_from_m_vector(np.array([1.4, 1.6, 1.8]))
        assert values[1] == -np.inf
        np.testing.assert_array_equal(values[[0, 2]], expected)


def test_concrete_vector_evaluation_preserves_explicit_branch():
    from RIFT.physics import EOSManager
    eos = EOSManager.EOSConcrete.__new__(EOSManager.EOSConcrete)
    visited = []
    def scalar(mass, branch_id=None):
        visited.append(branch_id)
        return mass + branch_id
    eos.lambda_from_m = scalar
    np.testing.assert_array_equal(eos.lambda_from_m_vector(np.array([1., 2.]), branch_id=7), [8., 9.])
    assert visited == [7, 7]
