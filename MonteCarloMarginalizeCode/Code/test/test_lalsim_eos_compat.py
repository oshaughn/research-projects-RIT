import json

import lal
import numpy as np
import pytest

from RIFT.physics.lalsim_eos_compat import (
    AmbiguousFamilyBranchError,
    LALSimNeutronStarFamilyAdapter,
)


class LegacyLALSimulation:
    def __init__(self):
        self.create_calls = []
        self.file_calls = []

    def SimNeutronStarEOSFromFile(self, fname):
        self.file_calls.append((fname, 0))
        return "clean-eos"

    def SimNeutronStarEOSFromFileChoiceDirtyPT(self, fname, dirty):
        self.file_calls.append((fname, dirty))
        return "dirty-eos"

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

    def SimNeutronStarEOSFromFile(self, fname):
        self.file_calls.append((fname, 0))
        return "clean-eos"

    def SimNeutronStarEOSFromFileChoiceDirtyPT(self, fname, dirty):
        self.file_calls.append((fname, dirty))
        return "dirty-eos"

    def CreateSimNeutronStarFamily(self, eos, min_fam):
        self.create_calls.append((eos, min_fam))
        return "multibranch-family"

    def SimNeutronStarFamNumberOfBranches(self, family):
        return len(self.bounds)

    def SimNeutronStarFamMinMassPerBranch(self, family, branch_id):
        return self.bounds[branch_id][0]

    def SimNeutronStarFamMaxMassPerBranch(self, family, branch_id):
        return self.bounds[branch_id][1]

    def SimNeutronStarFamMinMass(self, family):
        return min(x[0] for x in self.bounds)

    def SimNeutronStarFamMaxMass(self, family):
        return max(x[1] for x in self.bounds)

    def SimNeutronStarFamRadiusOfMassPerBranch(self, mass, family, branch_id):
        return 10.0 * branch_id + mass

    def SimNeutronStarFamLoveNumberK2OfMassPerBranch(
        self, mass, family, branch_id
    ):
        return branch_id + 0.1 * mass

    def SimNeutronStarFamCentralPressureOfMassPerBranch(
        self, mass, family, branch_id
    ):
        return 100.0 * branch_id + mass


class StellarMassMultibranchLALSimulation(MultibranchLALSimulation):
    bounds = tuple(
        (lower * lal.MSUN_SI, upper * lal.MSUN_SI)
        for lower, upper in MultibranchLALSimulation.bounds
    )


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


def test_reviewed_lalsimulation_uses_minimal_multibranch_api():
    lalsim = MultibranchLALSimulation()
    family = LALSimNeutronStarFamilyAdapter(
        "eos", minimal=True, lalsim_module=lalsim
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


def test_twin_star_mass_requires_an_explicit_branch():
    family = LALSimNeutronStarFamilyAdapter(
        "eos", lalsim_module=MultibranchLALSimulation()
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

    assert fake_lalsim.file_calls == [("new-format.dat", 1)]
    assert fake_lalsim.create_calls == [("dirty-eos", 1)]
    assert eos.eos == "dirty-eos"
    assert eos._get_lalsim_family_adapter().number_of_branches == 2

    extended = EOSManager.EOSLALSimulationFromFile(
        "extended-format.dat", minimal_family=False
    )
    assert fake_lalsim.file_calls[-1] == ("extended-format.dat", 0)
    assert fake_lalsim.create_calls[-1] == ("clean-eos", 0)


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
    assert np.isfinite(piecewise.lambda_from_m(1.4))


def test_selected_branch_view_preserves_legacy_scalar_consumer_api(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = StellarMassMultibranchLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    eos = EOSManager.EOSLALSimulationFromFile("twin-star.dat")

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
