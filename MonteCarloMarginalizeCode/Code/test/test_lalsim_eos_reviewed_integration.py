"""Opt-in gate against an actual reviewed LALSimulation build.

This is intentionally separate from the fake-backed compatibility tests. It
skips ordinary CI unless RIFT_REVIEWED_LALSIM_MANIFEST names a build-generated
manifest and fails closed once the gate is enabled.
"""

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest


MANIFEST_ENV = "RIFT_REVIEWED_LALSIM_MANIFEST"
REQUIRED_SYMBOLS = (
    "SimNeutronStarEOSFromFileChoiceDirtyPT",
    "SimNeutronStarFamNumberOfBranches",
    "SimNeutronStarFamMinMassPerBranch",
    "SimNeutronStarFamMaxMassPerBranch",
    "SimNeutronStarFamRadiusOfMassPerBranch",
    "SimNeutronStarFamLoveNumberK2OfMassPerBranch",
    "SimNeutronStarFamCentralPressureOfMassPerBranch",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest():
    manifest_name = os.environ.get(MANIFEST_ENV)
    if not manifest_name:
        pytest.skip(
            "real reviewed-LALSimulation gate disabled; set {}".format(
                MANIFEST_ENV
            )
        )
    manifest_path = Path(manifest_name).resolve()
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    ref = manifest.get("lalsuite_ref", "")
    assert re.fullmatch(r"[0-9a-f]{40}", ref), (
        "lalsuite_ref must be the exact 40-character commit built for this job"
    )
    return manifest_path, manifest


def test_actual_reviewed_lalsimulation_tables(record_property):
    import lalsimulation as lalsim
    from RIFT.physics import EOSManager
    from RIFT.physics.lalsim_eos_compat import AmbiguousFamilyBranchError

    manifest_path, manifest = _load_manifest()
    record_property("lalsuite_ref", manifest["lalsuite_ref"])
    record_property(
        "lalsimulation_version",
        getattr(lalsim, "LALSIMULATION_VERSION", "unknown"),
    )
    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(lalsim, name)]
    assert not missing, "reviewed LALSimulation symbols missing: {}".format(missing)

    fixtures = manifest.get("fixtures", {})
    assert set(fixtures) == {"two_column", "nine_column", "twin_star"}
    loaded = {}
    expected_columns = {"two_column": 2, "nine_column": 9, "twin_star": None}
    for name in ("two_column", "nine_column", "twin_star"):
        fixture = fixtures[name]
        path = (manifest_path.parent / fixture["path"]).resolve()
        assert path.is_file(), "missing {} fixture: {}".format(name, path)
        assert _sha256(path) == fixture["sha256"]
        data = np.loadtxt(str(path))
        columns = 1 if data.ndim == 1 else data.shape[1]
        if expected_columns[name] is not None:
            assert columns == expected_columns[name]
        loaded[name] = EOSManager.EOSLALSimulationFromFile(
            str(path),
            dirty_phase_transitions=bool(
                fixture.get("dirty_phase_transitions", False)
            ),
        )
        assert loaded[name]._get_lalsim_family_adapter().number_of_branches >= 1

    family = loaded["twin_star"]._get_lalsim_family_adapter()
    assert family.number_of_branches >= 2
    overlaps = []
    for left in range(family.number_of_branches):
        for right in range(left + 1, family.number_of_branches):
            lower = max(family.minimum_mass(left), family.minimum_mass(right))
            upper = min(family.maximum_mass(left), family.maximum_mass(right))
            if lower < upper:
                overlaps.append((left, right, 0.5 * (lower + upper)))
    assert overlaps, "twin_star fixture has no overlapping stable mass branches"
    left, right, mass = overlaps[0]
    with pytest.raises(AmbiguousFamilyBranchError):
        family.radius(mass)
    assert family.radius(mass, branch_id=left) > 0
    assert family.radius(mass, branch_id=right) > 0
