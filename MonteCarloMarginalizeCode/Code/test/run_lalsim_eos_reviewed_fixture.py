"""Hard-timeout subprocess target for externally supplied EOS tables."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--fixture", required=True)
parser.add_argument("--columns", type=int, choices=(2, 4, 9), required=True)
parser.add_argument("--arrays", action="store_true")
parser.add_argument("--twin", action="store_true")
parser.add_argument("--extended", action="store_true")
parser.add_argument("--eosmanager", action="store_true")
parser.add_argument("--status", required=True)
args = parser.parse_args()


def fail(message):
    Path(args.status).write_text(json.dumps({"error": str(message)}))
    raise SystemExit(2)


def record_uncaught(exc_type, exc, traceback):
    Path(args.status).write_text(
        json.dumps({"error": "{}: {}".format(exc_type.__name__, exc)})
    )


sys.excepthook = record_uncaught

data = np.loadtxt(args.fixture, ndmin=2)
columns = data.shape[1]
if columns != args.columns:
    fail("column mismatch before native loader")

if args.eosmanager:
    try:
        import lalframe  # noqa: F401
    except ImportError:
        lalframe_stub = types.ModuleType("lalframe")
        lalframe_stub.__path__ = []
        lalframe_stub.frread = types.ModuleType("lalframe.frread")
        sys.modules["lalframe"] = lalframe_stub
        sys.modules["lalframe.frread"] = lalframe_stub.frread
    from RIFT.physics import EOSManager
    import lal
    eos = EOSManager.EOSLALSimulationFromFile(
        args.fixture,
        phase_transition_aware=True,
        minimal_family=not args.extended,
    )
    family = eos._get_lalsim_family_adapter()
else:
    import lal
    import lalsimulation as lalsim
    source = (
        Path(__file__).resolve().parents[1]
        / "RIFT" / "physics" / "lalsim_eos_compat.py"
    )
    spec = importlib.util.spec_from_file_location("reviewed_adapter", str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.arrays:
        if args.columns != 4:
            fail("array conversion requires four-column wiki input")
        conversion = 1.602176634e32 * lal.G_SI / lal.C_SI**4
        energy = lal.CreateREAL8Vector(len(data))
        pressure = lal.CreateREAL8Vector(len(data))
        energy.data[:] = data[:, 2] * conversion
        pressure.data[:] = data[:, 3] * conversion
        multipart = lalsim.SimNeutronStarEOSFromArraysPhaseTransition(
            energy, pressure
        )
    else:
        multipart = lalsim.SimNeutronStarEOSFromFilePhaseTransition(
            args.fixture
        )
    family = module.LALSimNeutronStarFamilyAdapter(
        multipart,
        minimal=not args.extended,
        reviewed_multibranch=True,
        lalsim_module=lalsim,
    )

result = {"branches": family.number_of_branches}
if result["branches"] < 1:
    fail("no stable branches")
for branch in range(family.number_of_branches):
    lower = family.minimum_mass(branch)
    upper = family.maximum_mass(branch)
    mass = 0.5 * (lower + upper)
    if family.radius(mass, branch_id=branch) <= 0:
        fail("nonpositive radius")
if args.eosmanager:
    mass_si = 0.5 * (family.minimum_mass(0) + family.maximum_mass(0))
    mass_msun = mass_si / lal.MSUN_SI
    if 0 not in eos.branches_for_m(mass_msun):
        fail("EOSManager branch lookup lost primary branch")
    if not np.isfinite(eos.lambda_from_m(mass_msun, branch_id=0)):
        fail("EOSManager returned nonfinite tidal deformability")
    selected = eos.for_branch(0)
    result["causal_under_branch_max"] = bool(
        selected.test_speed_of_sound_causal()
    )
    result["causal_full_table"] = bool(
        eos.test_speed_of_sound_causal(test_only_under_mmax=False)
    )
if args.twin:
    overlaps = []
    for left in range(family.number_of_branches):
        for right in range(left + 1, family.number_of_branches):
            lower = max(family.minimum_mass(left), family.minimum_mass(right))
            upper = min(family.maximum_mass(left), family.maximum_mass(right))
            if lower < upper:
                mass = 0.5 * (lower + upper)
                radii = [family.radius(mass, branch_id=x) for x in (left, right)]
                love = [
                    family.love_number_k2(mass, branch_id=x)
                    for x in (left, right)
                ]
                if np.isclose(radii[0], radii[1]) or np.isclose(
                    love[0], love[1]
                ):
                    fail("branch_id ignored")
                overlaps.append((left, right))
    if not overlaps:
        fail("no overlapping twin branches")
Path(args.status).write_text(json.dumps(result, sort_keys=True))
