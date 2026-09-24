"""Opt-in gate against an actual reviewed LALSimulation build.

This is intentionally separate from the fake-backed compatibility tests. It
skips ordinary CI unless RIFT_REVIEWED_LALSIM_MANIFEST names a build-generated
manifest and fails closed once the gate is enabled.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pytest


MANIFEST_ENV = "RIFT_REVIEWED_LALSIM_MANIFEST"
REQUIRED_SYMBOLS = (
    "SimulationVCSInfo",
    "SimNeutronStarEOSMultiPartsByName",
    "SimNeutronStarEOSFromArraysPhaseTransition",
    "SimNeutronStarEOSFromFilePhaseTransition",
    "CreateSimNeutronStarFamilyPT",
    "SimNeutronStarFamNumberOfBranches",
    "SimNeutronStarFamBranchMinMass",
    "SimNeutronStarFamBranchMaxMass",
    "SimNeutronStarFamBranchRadius",
    "SimNeutronStarFamBranchLoveNumberK2",
    "SimNeutronStarFamBranchCentralPressure",
    "SimNeutronStarEOSMultiPartsMaxPseudoEnthalpy",
    "SimNeutronStarEOSMultiPartsPseudoEnthalpyOfPressure",
    "SimNeutronStarEOSMultiPartsSpeedOfSoundOfPseudoEnthalpy",
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


def _load_adapter_module():
    """Load the pure adapter without importing RIFT's heavyweight package."""
    source = (
        Path(__file__).resolve().parents[1]
        / "RIFT" / "physics" / "lalsim_eos_compat.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rift_lalsim_eos_compat_gate", str(source)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_reviewed_build(lalsim, manifest, record_property):
    missing = [name for name in REQUIRED_SYMBOLS if not hasattr(lalsim, name)]
    assert not missing, "reviewed LALSimulation symbols missing: {}".format(missing)
    vcs_info = lalsim.SimulationVCSInfo
    assert vcs_info.vcsId == manifest["lalsuite_ref"]
    assert vcs_info.vcsClean == "CLEAN"
    record_property("lalsuite_ref", vcs_info.vcsId)
    record_property("lalsimulation_vcs_status", vcs_info.vcsStatus)
    record_property("lalsimulation_vcs_tag", vcs_info.vcsTag)


def _run_fixture_subprocess(command):
    """Run native parsing with bounded time, output, and returned status."""
    with tempfile.TemporaryDirectory(prefix="rift-reviewed-eos-") as tmpdir:
        status = Path(tmpdir) / "status.json"
        result = subprocess.run(
            command + ["--status", str(status)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
        if status.exists():
            with status.open("rb") as stream:
                detail = stream.read(65536).decode("utf-8", errors="replace")
        else:
            detail = ""
    assert result.returncode == 0, detail or (
        "fixture subprocess failed with return code {}".format(
            result.returncode
        )
    )


def _run_expected_upstream_crash(command):
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )
    if result.returncode == 0:
        pytest.fail(
            "known two-transition native crash is fixed; promote this fixture "
            "to a required passing gate"
        )
    assert result.returncode in (-11, 139), (
        "expected SIGSEGV/139, got {}".format(result.returncode)
    )
    pytest.xfail("upstream reviewed LALSimulation two-transition SIGSEGV")


def test_fixture_subprocess_status_detail_is_bounded():
    code = (
        "from pathlib import Path; import sys; "
        "Path(sys.argv[-1]).write_text('x' * 1000000); raise SystemExit(3)"
    )
    with pytest.raises(AssertionError) as caught:
        _run_fixture_subprocess([sys.executable, "-c", code])
    assert len(str(caught.value)) <= 66000


def test_actual_reviewed_lalsimulation_builtin(record_property):
    """Safe acceptance using LALSuite's trusted built-in SLY table."""
    import gc
    import lal
    import lalsimulation as lalsim

    _, manifest = _load_manifest()
    _validate_reviewed_build(lalsim, manifest, record_property)
    Adapter = _load_adapter_module().LALSimNeutronStarFamilyAdapter

    multipart_eos = lalsim.SimNeutronStarEOSMultiPartsByName("SLY")
    minimal = Adapter(
        multipart_eos,
        minimal=True,
        reviewed_multibranch=True,
        lalsim_module=lalsim,
    )
    assert minimal.number_of_branches >= 1
    mass = 0.5 * (minimal.minimum_mass(0) + minimal.maximum_mass(0))
    radius = minimal.radius(mass, branch_id=0)
    love = minimal.love_number_k2(mass, branch_id=0)
    pressure = minimal.central_pressure(mass, branch_id=0)
    enthalpy = lalsim.SimNeutronStarEOSMultiPartsPseudoEnthalpyOfPressure(
        pressure, multipart_eos
    )
    sound_si = lalsim.SimNeutronStarEOSMultiPartsSpeedOfSoundOfPseudoEnthalpy(
        enthalpy, multipart_eos
    )
    assert all(
        np.isfinite(value) and value > 0
        for value in (
            radius,
            love,
            pressure,
            enthalpy,
            sound_si,
            lalsim.SimNeutronStarEOSMultiPartsMaxPseudoEnthalpy(multipart_eos),
        )
    )
    assert sound_si > 1.0
    assert sound_si / lal.C_SI < 1.1

    legacy_eos = lalsim.SimNeutronStarEOSByName("SLY")
    legacy = Adapter(
        legacy_eos, reviewed_multibranch=False, lalsim_module=lalsim
    )
    legacy_mass = 0.5 * (legacy.minimum_mass() + legacy.maximum_mass())
    assert legacy.radius(legacy_mass, branch_id=0) > 0

    extended = Adapter(
        multipart_eos,
        minimal=False,
        reviewed_multibranch=True,
        lalsim_module=lalsim,
    )
    for name in (
        "SimNeutronStarFamBranchBaryonicMass",
        "SimNeutronStarFamBranchLoveNumberK3",
        "SimNeutronStarFamBranchLoveNumberK4",
    ):
        fn = getattr(lalsim, name, None)
        if fn is None:
            continue
        assert np.isfinite(fn(mass, 0, extended.family))
        with pytest.raises(Exception):
            fn(mass, 0, minimal.family)

    for _ in range(8):
        eos_here = lalsim.SimNeutronStarEOSMultiPartsByName("SLY")
        family_here = Adapter(
            eos_here, reviewed_multibranch=True, lalsim_module=lalsim
        )
        assert family_here.number_of_branches >= 1
        del family_here, eos_here
        gc.collect()


def test_actual_reviewed_lalsimulation_tables(record_property):
    import lalsimulation as lalsim

    manifest_path, manifest = _load_manifest()
    _validate_reviewed_build(lalsim, manifest, record_property)
    fixtures = manifest.get("fixtures")
    if not fixtures:
        pytest.skip("external reviewed EOS fixtures not supplied in manifest")
    assert {"two_column", "nine_column"}.issubset(fixtures)
    runner = Path(__file__).with_name("run_lalsim_eos_reviewed_fixture.py")
    fixture_specs = [("two_column", 2), ("nine_column", 9)]
    if "twin_star" in fixtures:
        fixture_specs.append(
            ("twin_star", int(fixtures["twin_star"]["columns"]))
        )
    for name, expected_columns in fixture_specs:
        assert expected_columns in (2, 9), (
            "file-loader fixtures must have 2 or 9 columns; four-column wiki "
            "arrays require a separately provenance-recorded transform"
        )
        fixture = fixtures[name]
        path = (manifest_path.parent / fixture["path"]).resolve()
        assert path.is_file(), "missing {} fixture: {}".format(name, path)
        assert _sha256(path) == fixture["sha256"]
        data = np.loadtxt(str(path), ndmin=2)
        columns = data.shape[1]
        assert columns == expected_columns
        command = [
            sys.executable,
            str(runner),
            "--fixture",
            str(path),
            "--columns",
            str(expected_columns),
        ]
        if name == "twin_star":
            command.append("--twin")
        _run_fixture_subprocess(command)

    nine_path = (manifest_path.parent / fixtures["nine_column"]["path"]).resolve()
    for extra in ("--extended", "--eosmanager"):
        _run_fixture_subprocess(
            [
                sys.executable,
                str(runner),
                "--fixture",
                str(nine_path),
                "--columns",
                "9",
                extra,
            ]
        )



def test_known_upstream_two_transition_crash(record_property):
    """Keep the upstream SIGSEGV visible without masking passing fixtures."""
    import lalsimulation as lalsim

    manifest_path, manifest = _load_manifest()
    _validate_reviewed_build(lalsim, manifest, record_property)
    crash = manifest.get("known_upstream_crash")
    if not crash:
        pytest.skip("known upstream two-transition fixture not supplied")
    crash_path = (manifest_path.parent / crash["path"]).resolve()
    assert _sha256(crash_path) == crash["sha256"]
    crash_data = np.loadtxt(str(crash_path), ndmin=2)
    assert crash_data.shape[1] == 4
    expected_codes = crash.get("expected_returncodes", [-11, 139])
    assert expected_codes == [-11, 139]
    runner = Path(__file__).with_name("run_lalsim_eos_reviewed_fixture.py")
    _run_expected_upstream_crash(
        [
            sys.executable,
            str(runner),
            "--fixture",
            str(crash_path),
            "--columns",
            "4",
            "--arrays",
            "--status",
            os.devnull,
        ]
    )
