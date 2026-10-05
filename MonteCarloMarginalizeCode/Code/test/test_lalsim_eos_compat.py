import ast
import json
import os
import types

import lal
import numpy as np
import pytest

from RIFT.physics.lalsim_eos_compat import (
    AmbiguousFamilyBranchError,
    LALSimNeutronStarFamilyAdapter,
    mass_in_eos_support,
    validate_fixed_eos_branch_request,
)

# The fixed-EOS consumer of the branch bounds.  CIP is a script -- importing it
# parses argv and runs the whole fit -- so the tests below read its source with
# ast and exec only the one function they exercise, which keeps them running the
# shipped code rather than a copy that can silently drift away from it.
CIP_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "bin",
    "util_ConstructIntrinsicPosterior_GenericCoordinates.py",
)


def _cip_source_tree():
    with open(CIP_SCRIPT) as stream:
        return ast.parse(stream.read())


def _load_cip_function(name):
    for node in _cip_source_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {"np": np}
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    CIP_SCRIPT,
                    "exec",
                ),
                namespace,
            )
            return namespace[name]
    raise AssertionError("CIP no longer defines {}".format(name))


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

    def SimNeutronStarEOSFromFile(self, fname):
        self.file_calls.append(("legacy", fname))
        return "legacy-file-eos"

    def SimNeutronStarEOSFromFilePhaseTransition(self, fname):
        self.file_calls.append(("multipart", fname))
        return "multipart-eos"

    def CreateSimNeutronStarFamily(self, eos):
        self.create_calls.append(("legacy", eos))
        return "legacy-family"

    def CreateSimNeutronStarFamilyPT(self, eos, min_fam):
        self.create_calls.append(("multipart", eos, min_fam))
        return "multibranch-family"

    def CreateSimNeutronStarFamilyPTWithPcmin(
        self, eos, min_fam, log_pressure_min
    ):
        self.create_calls.append(
            ("multipart-pcmin", eos, min_fam, log_pressure_min)
        )
        return "multibranch-family-pcmin"

    def SimNeutronStarFamNumberOfBranches(self, family):
        return len(self.bounds)

    def SimNeutronStarFamBranchMinMass(self, branch_id, family):
        return self.bounds[branch_id][0]

    def SimNeutronStarFamBranchMaxMass(self, branch_id, family):
        return self.bounds[branch_id][1]

    def SimNeutronStarFamMinMass(self, family):
        return min(x[0] for x in self.bounds)

    def SimNeutronStarFamMaxMass(self, family):
        return max(x[1] for x in self.bounds)

    def SimNeutronStarFamMinimumMass(self, family):
        return min(x[0] for x in self.bounds)

    def SimNeutronStarMaximumMass(self, family):
        return max(x[1] for x in self.bounds)

    def SimNeutronStarRadius(self, mass, family):
        return mass + 10.0

    def SimNeutronStarLoveNumberK2(self, mass, family):
        return 0.1 * mass

    def SimNeutronStarCentralPressure(self, mass, family):
        return 100.0 * mass

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

    def SimNeutronStarEOSMultiPartsPseudoEnthalpyOfPressure(
        self, pressure, eos
    ):
        return pressure

    def SimNeutronStarEOSMultiPartsSpeedOfSoundOfPseudoEnthalpy(
        self, enthalpy, eos
    ):
        return 0.5 * lal.C_SI

    def SimNeutronStarEOSMultiPartsMaxPseudoEnthalpy(self, eos):
        return 10.0

    def SimNeutronStarEOSPseudoEnthalpyOfPressure(self, pressure, eos):
        raise AssertionError("multipart causality used the legacy EOS accessor")

    def SimNeutronStarEOSSpeedOfSoundGeometerized(self, enthalpy, eos):
        raise AssertionError("multipart causality used the legacy EOS accessor")


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
        "eos", minimal=True, multipart=True, lalsim_module=lalsim
    )

    assert lalsim.create_calls == [("multipart", "eos", 1)]
    assert family.number_of_branches == 2
    assert family.minimum_mass() == 1.0
    assert family.maximum_mass() == 3.0
    assert family.branches_for_mass(1.25) == [0]
    assert family.branches_for_mass(1.75) == [0, 1]
    assert family.radius(1.75, branch_id=1) == 11.75
    assert family.love_number_k2(1.75, branch_id=1) == pytest.approx(1.175)
    assert family.central_pressure(1.75, branch_id=1) == 101.75


def test_reviewed_build_preserves_legacy_family_for_nonmultipart_eos():
    lalsim = MultibranchLALSimulation()
    family = LALSimNeutronStarFamilyAdapter(
        "ordinary-eos", multipart=False, lalsim_module=lalsim
    )
    assert lalsim.create_calls == [("legacy", "ordinary-eos")]
    assert family.number_of_branches == 1


def test_partial_reviewed_api_fails_diagnostically(monkeypatch):
    lalsim = MultibranchLALSimulation()
    monkeypatch.delattr(
        MultibranchLALSimulation,
        "SimNeutronStarFamBranchLoveNumberK2",
    )
    with pytest.raises(RuntimeError, match="multipart API is incomplete"):
        LALSimNeutronStarFamilyAdapter(
            "eos", multipart=True, lalsim_module=lalsim
        )


def test_twin_star_mass_requires_an_explicit_branch():
    family = LALSimNeutronStarFamilyAdapter(
        "eos", multipart=True,
        lalsim_module=MultibranchLALSimulation()
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

    assert fake_lalsim.file_calls == [("multipart", "new-format.dat")]
    assert fake_lalsim.create_calls == [
        ("multipart", "multipart-eos", 1)
    ]
    assert eos.eos == "multipart-eos"
    assert eos._get_lalsim_family_adapter().number_of_branches == 2

    extended = EOSManager.EOSLALSimulationFromFile(
        "extended-format.dat", minimal_family=False
    )
    assert fake_lalsim.file_calls[-1] == (
        "multipart", "extended-format.dat"
    )
    assert fake_lalsim.create_calls[-1] == (
        "multipart", "multipart-eos", 0
    )

    pressure_floor = EOSManager.EOSLALSimulationFromFile(
        "pressure-floor.dat", family_log_pressure_min=12.5
    )
    assert pressure_floor.eos_fam == "multibranch-family-pcmin"
    assert fake_lalsim.create_calls[-1] == (
        "multipart-pcmin", "multipart-eos", 1, 12.5
    )

    for invalid in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match=r"finite ln\(Pc / Pa\)"):
            EOSManager.EOSLALSimulationFromFile(
                "pressure-floor.dat", family_log_pressure_min=invalid
            )


def test_eosmanager_file_loader_preserves_legacy_default(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = MultibranchLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    eos = EOSManager.EOSLALSimulationFromFile("ordinary-two-column.dat")

    assert fake_lalsim.file_calls == [
        ("legacy", "ordinary-two-column.dat")
    ]
    assert fake_lalsim.create_calls == [("legacy", "legacy-file-eos")]
    assert eos._get_lalsim_family_adapter().number_of_branches == 1


def test_eosmanager_requested_multipart_requires_reviewed_reader(monkeypatch):
    from RIFT.physics import EOSManager

    legacy_lalsim = LegacyLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", legacy_lalsim)
    for kwargs in (
        {"phase_transition_aware": True},
        {"dirty_phase_transitions": True},
        {"minimal_family": False},
    ):
        with pytest.raises(NotImplementedError, match="multipart EOS family"):
            EOSManager.EOSLALSimulationFromFile("phase.dat", **kwargs)


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


def test_selected_branch_view_flags_masses_below_the_branch_minimum(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = StellarMassMultibranchLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    secondary = EOSManager.EOSLALSimulationFromFile(
        "twin-star.dat", phase_transition_aware=True
    ).for_branch(1)

    # This branch starts above the family minimum, so consumers that mask only
    # against mMaxMsun still hand it lighter masses.  Those must be flagged,
    # not raised through the caller's batch.
    assert secondary.mMinMsun == pytest.approx(1.3)
    assert secondary.branches_for_m(1.1) == []
    assert secondary.lambda_from_m(1.1) == -np.inf
    assert secondary.lambda_from_m(1.1 * lal.MSUN_SI) == -np.inf

    flagged = secondary.lambda_from_m_vector(np.array([1.1, 1.75, 3.5]))
    assert flagged[0] == -np.inf
    assert np.isfinite(flagged[1])
    assert flagged[2] == pytest.approx(1e-8)


def test_cip_fit_guard_drops_out_of_support_rows_and_keeps_the_rest():
    protect = _load_cip_function("protect_fit_against_out_of_support")

    batches = []

    def picky_fit(x):
        # Stands in for the sklearn-backed fits, which raise on nonfinite input.
        assert np.all(np.isfinite(x)), "the fit was handed an out-of-support row"
        batches.append(np.array(x, copy=True))
        return x[:, 0]

    guarded = protect(picky_fit)

    # Row 1 is the flagged draw: the branch view returns -inf for a mass with no
    # stable star, and the coordinate conversion carries that into the fit
    # coordinates.  Row 2 is the nan that an inf-inf coordinate combination
    # (lambda_plus / delta_lambda_tilde, say) produces from the same flag.
    values = guarded(
        np.array([[1.0, 2.0], [-np.inf, 2.0], [np.nan, 2.0], [3.0, 4.0]])
    )
    assert values[1] == -np.inf and values[2] == -np.inf
    assert np.exp(values[1]) == 0.0  # zero probability, not an aborted batch
    assert values[0] == pytest.approx(1.0)
    assert values[3] == pytest.approx(3.0)

    # The valid rows must survive, and in ONE call: a batched fit backend cannot
    # be turned into a per-row loop by the presence of a rejected draw.
    assert len(batches) == 1
    assert batches[0].shape == (2, 2)

    # A batch with no support at all must not reach the fit at all.
    empty = guarded(np.array([[-np.inf, 2.0]]))
    assert empty.shape == (1,) and empty[0] == -np.inf
    assert len(batches) == 1


def test_cip_applies_the_fit_guard_unconditionally_for_a_fixed_eos():
    tree = _cip_source_tree()

    wraps = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "my_fit"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None)
        == "protect_fit_against_out_of_support"
    ]
    assert len(wraps) == 1

    # It must be applied in the fixed-EOS branch of the coordinate-conversion
    # setup ...
    eos_blocks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "using_eos" in ast.dump(node.test)
        and any(
            isinstance(stmt, ast.FunctionDef) and stmt.name == "convert_coords"
            for stmt in node.orelse
        )
    ]
    assert len(eos_blocks) == 1
    assert any(stmt is wraps[0] for stmt in eos_blocks[0].orelse)

    # ... and not underneath the optional --protect-coordinate-conversions flag,
    # because out-of-support draws are ordinary with a fixed EOS.
    optional = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If)
                and "protect_coordinate_conversions" in ast.dump(node.test)):
            optional.update(id(child) for child in ast.walk(node))
    assert id(wraps[0]) not in optional


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


def test_mr_lambda_helpers_accept_explicit_multipart_adapter(monkeypatch):
    from RIFT.physics import EOSManager

    fake_lalsim = StellarMassMultibranchLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", fake_lalsim)
    monkeypatch.setattr(
        EOSManager,
        "create_family",
        lambda eos, minimal=True, multipart=False, **kwargs:
        LALSimNeutronStarFamilyAdapter(
            eos, minimal=minimal, multipart=multipart,
            lalsim_module=fake_lalsim
        ),
    )
    eos = EOSManager.EOSLALSimulationFromFile(
        "twin-star.dat", phase_transition_aware=True
    )
    branches = EOSManager.make_mr_lambda_lal_branches(
        eos.eos, n_bins=4, family_adapter=eos._get_lalsim_family_adapter()
    )
    one_branch = EOSManager.make_mr_lambda_lal(
        eos.eos, n_bins=4, branch_id=1, multipart=True
    )
    assert set(branches) == {0, 1}
    assert branches[1].shape == (4, 3)
    assert one_branch.shape == (4, 3)

    alias_branch = EOSManager.make_mr_lambda_lal(
        eos.eos, n_bins=3, branch_id=1, reviewed_multibranch=True
    )
    assert alias_branch.shape == (3, 3)
    with pytest.raises(ValueError, match="conflicting values"):
        EOSManager.make_mr_lambda_lal(
            eos.eos, branch_id=1, multipart=True,
            reviewed_multibranch=False,
        )


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
        MultibranchLALSimulation,
        "SimNeutronStarFamBranchCentralPressure",
    )
    assert secondary.test_speed_of_sound_causal() is False


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


def _load_cip_nested_function(name, container_predicate=None):
    """exec one CIP function that is nested inside a conditional block.

    ``_load_cip_function`` only reaches module-level defs.  The fixed-EOS
    coordinate-conversion helpers live in the ``else:`` of the ``using_eos``
    test, so they need a full walk.  Loading from the shipped source, rather
    than restating the body here, is what makes these regression tests.
    """
    tree = _cip_source_tree()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == name):
            continue
        if container_predicate is not None and not container_predicate(tree, node):
            continue
        return compile(
            ast.Module(body=[node], type_ignores=[]), CIP_SCRIPT, "exec"
        )
    raise AssertionError("CIP no longer defines a nested {}".format(name))


def _in_fixed_eos_block(tree, target):
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and "using_eos" in ast.dump(node.test)
                and any(child is target for child in node.orelse)):
            return True
    return False


class _MassOnlyCoordinateStub:
    """A converter whose fit basis carries no tidal coordinate.

    This is the configuration behind the review finding: with ``mc,eta`` as
    both the sampled and the fitted coordinates,
    ``convert_waveform_coordinates_with_eos`` computes lambda from the EOS and
    then DISCARDS it, because lambda is not among ``coord_names``.  Every row
    it returns is finite, whatever the EOS said about the masses.
    """

    def __init__(self):
        self.eos_conversions = 0

    @staticmethod
    def _m1_m2_from_mc_eta(x_in):
        mc = x_in[:, 0]
        eta = x_in[:, 1]
        mtot = mc / np.power(eta, 3.0 / 5.0)
        delta = np.sqrt(1.0 - 4.0 * eta)
        return np.c_[mtot * (1 + delta) / 2, mtot * (1 - delta) / 2]

    def convert_waveform_coordinates(self, x_in, coord_names=None,
                                     low_level_coord_names=None, **kwargs):
        assert list(coord_names) == ['m1', 'm2']
        assert list(low_level_coord_names) == ['mc', 'eta']
        return self._m1_m2_from_mc_eta(np.atleast_2d(x_in))

    def convert_waveform_coordinates_with_eos(self, x_in, **kwargs):
        self.eos_conversions += 1
        # Identity: the fit basis IS the sampled basis, so nothing the EOS
        # reported about lambda survives into the returned row.
        return np.array(np.atleast_2d(x_in), dtype=float, copy=True)


def _mc_eta(m1, m2):
    mtot = m1 + m2
    return np.array([np.power(m1 * m2, 3.0 / 5.0) / np.power(mtot, 1.0 / 5.0),
                     m1 * m2 / mtot ** 2])


def _cip_eos_convert_coords(eos, no_matter1=False, no_matter2=False):
    """Build CIP's fixed-EOS convert_coords over a mass-only fit basis."""
    lalsimutils_stub = _MassOnlyCoordinateStub()
    namespace = {
        "np": np,
        "lalsimutils": lalsimutils_stub,
        "mass_in_eos_support": mass_in_eos_support,
        "my_eos": eos,
        "coord_names": ['mc', 'eta'],
        "low_level_coord_names": ['mc', 'eta'],
        "source_redshift": 0,
        "opts": types.SimpleNamespace(
            no_matter1=no_matter1, no_matter2=no_matter2,
            downselect_enforce_kerr=False,
        ),
    }
    exec(_load_cip_nested_function("eos_mass_support_mask"), namespace)
    exec(
        _load_cip_nested_function("convert_coords", _in_fixed_eos_block),
        namespace,
    )
    return namespace["convert_coords"], lalsimutils_stub


def _twin_star_branch(monkeypatch, branch_id):
    from RIFT.physics import EOSManager

    monkeypatch.setattr(
        EOSManager, "lalsim", StellarMassMultibranchLALSimulation()
    )
    return EOSManager.EOSLALSimulationFromFile(
        "twin-star.dat", phase_transition_aware=True
    ).for_branch(branch_id)


def test_mass_support_mask_rejects_masses_outside_both_branch_bounds():
    eos = types.SimpleNamespace(mMinMsun=1.3, mMaxMsun=3.0)

    m1 = np.array([1.5, 1.5, 1.5, 3.5, 1.2])
    m2 = np.array([1.4, 1.1, 3.0, 1.4, 1.25])
    ok = mass_in_eos_support(eos, m1, m2)

    assert list(ok) == [True, False, False, False, False]
    # The upper test is strict, matching the mMaxMsun test the coordinate
    # converter already applies, so a mass exactly at mMaxMsun is out.
    assert not mass_in_eos_support(eos, 1.4, 3.0)
    assert mass_in_eos_support(eos, 1.3, 1.3)  # closed at the lower bound
    assert not mass_in_eos_support(eos, np.nan, 1.4)


def test_mass_support_mask_exempts_objects_declared_to_be_black_holes():
    eos = types.SimpleNamespace(mMinMsun=1.3, mMaxMsun=3.0)

    # A black hole is under no EOS constraint, so its mass must not be tested.
    assert mass_in_eos_support(eos, 30.0, 1.4, bh1=True)
    assert mass_in_eos_support(eos, 1.4, 0.4, bh2=True)
    assert not mass_in_eos_support(eos, 30.0, 0.4, bh1=True)
    assert not mass_in_eos_support(eos, 30.0, 1.4)


def test_mass_support_mask_leaves_an_eos_publishing_no_bounds_unchanged():
    # Several EOS classes set mMaxMsun = None and publish no minimum at all.
    # Reading an absent bound as unbounded is what keeps their behaviour, and
    # that of released LALSimulation callers, exactly as it was.
    unbounded = types.SimpleNamespace(mMaxMsun=None)
    assert mass_in_eos_support(unbounded, 1e-3, 500.0)

    upper_only = types.SimpleNamespace(mMaxMsun=2.2)
    assert mass_in_eos_support(upper_only, 0.2, 0.3)
    assert not mass_in_eos_support(upper_only, 0.2, 2.4)


def test_cip_stamps_out_of_support_rows_when_the_fit_basis_has_no_lambda(
        monkeypatch):
    # The review finding: with mc,eta fitted and mc,eta sampled, the converted
    # row is finite even for a mass with no star on the selected branch, so
    # finiteness of the converted coordinates cannot be the support test.
    secondary = _twin_star_branch(monkeypatch, 1)
    assert secondary.mMinMsun == pytest.approx(1.3)
    assert secondary.mMaxMsun == pytest.approx(3.0)

    convert_coords, stub = _cip_eos_convert_coords(secondary)

    in_support = _mc_eta(1.5, 1.4)
    below_branch_minimum = _mc_eta(1.5, 1.1)
    above_branch_maximum = _mc_eta(3.4, 1.4)

    x_out = convert_coords(
        np.array([in_support, below_branch_minimum, above_branch_maximum])
    )

    # Control: the converter itself reports every row as finite.
    passthrough = stub.convert_waveform_coordinates_with_eos(
        np.array([in_support, below_branch_minimum, above_branch_maximum])
    )
    assert np.all(np.isfinite(passthrough))

    assert np.all(np.isfinite(x_out[0]))
    assert np.all(x_out[0] == pytest.approx(in_support))
    assert np.all(np.isneginf(x_out[1]))
    assert np.all(np.isneginf(x_out[2]))

    # Stamped rows are exactly what the fit guard rejects, so they carry zero
    # probability rather than reaching the fit.
    protect = _load_cip_function("protect_fit_against_out_of_support")
    guarded = protect(lambda x: np.zeros(len(x)))
    assert list(np.exp(guarded(x_out))) == [1.0, 0.0, 0.0]


def test_cip_support_stamp_respects_no_matter_flags(monkeypatch):
    secondary = _twin_star_branch(monkeypatch, 1)

    # m2 = 1.1 has no star on this branch, but --no-matter2 declares it a black
    # hole, so the EOS must not be consulted about it.
    convert_coords, _ = _cip_eos_convert_coords(secondary, no_matter2=True)
    x_out = convert_coords(np.array([_mc_eta(1.5, 1.1)]))
    assert np.all(np.isfinite(x_out[0]))

    convert_coords, _ = _cip_eos_convert_coords(secondary)
    x_out = convert_coords(np.array([_mc_eta(1.5, 1.1)]))
    assert np.all(np.isneginf(x_out[0]))


def test_cip_export_applies_the_same_support_test_before_asking_for_lambda():
    # The fit and the export must not disagree about support: a draw the fit
    # gave zero weight cannot come back as an exported sample with a tidal
    # parameter.  Guard the wiring, since the two sites are 1000 lines apart.
    tree = _cip_source_tree()

    export_blocks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name) and node.test.id == "my_eos"
        and "lambda_from_m" in ast.dump(node)
    ]
    assert len(export_blocks) == 1
    block = export_blocks[0]

    guards = [
        node for node in ast.walk(block)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "mass_in_eos_support"
    ]
    assert len(guards) == 1

    # The support test must gate the EOS call, not merely accompany it.
    gated = [
        node for node in block.body
        if isinstance(node, ast.If)
        and any(child is guards[0] for child in ast.walk(node.test))
    ]
    assert len(gated) == 1
    assert "lambda_from_m" not in ast.dump(ast.Module(
        body=list(gated[0].body), type_ignores=[]))
    assert "lambda_from_m" in ast.dump(ast.Module(
        body=list(gated[0].orelse), type_ignores=[]))


def test_released_lalsimulation_family_publishes_both_mass_bounds(monkeypatch):
    # Backward compatibility: the lower bound is read through the adapter, so a
    # released build answers it with SimNeutronStarFamMinimumMass and never
    # touches a reviewed symbol.
    from RIFT.physics import EOSManager

    legacy = LegacyLALSimulation()
    monkeypatch.setattr(EOSManager, "lalsim", legacy)
    eos = EOSManager.EOSLALSimulationFromFile("released.dat")

    assert eos.mMinMsun == pytest.approx(1.0 / lal.MSUN_SI)
    assert eos.mMaxMsun == pytest.approx(3.0 / lal.MSUN_SI)
    assert legacy.create_calls  # legacy one-argument family API was used
