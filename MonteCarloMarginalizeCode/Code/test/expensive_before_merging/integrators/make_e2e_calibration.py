#!/usr/bin/env python
"""Regenerate the CALIBRATION table in test/test_e2e_analytic_pipeline.py.

The tolerances in that file (Z_TOLERANCE, MAX_SIGMA, MIN_NEFF) are derived from a sweep over
seeds, and the table recording it is a comment.  A comment is a claim nobody can check.  This
script re-derives it:

    python make_e2e_calibration.py --seeds 8          # the table as committed
    python make_e2e_calibration.py --seeds 2 --lane distance-marginalized

It imports the gate's OWN _run_ile, build_event and _exact rather than reimplementing them, so
a lane measured here is the lane the gate runs.  Reimplementing them would produce a table that
calibrates a different thing and agrees with itself.

Cost: about 10 s per arm, so the full 8-seed table is roughly 20 minutes on one core.  That is
why this lives here and not in CI.

If you add a lane to the gate, add it to LANES below, or the table silently stops covering it.
"""
import argparse
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEST_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

import test_e2e_analytic_pipeline as gate          # noqa: E402

_AV, _PORTFOLIO, _GMM, _AC = gate._AV, gate._PORTFOLIO, gate._GMM, gate._AC
_ACG = gate._ACG

# (label, sampler argv, A, B, extra kwargs for _run_ile).  A is None for a prior-only lane.
LANES = [
    ("prior-only,    AV",                        _AV,        None, 0.0, {}),
    ("prior-only,    portfolio",                 _PORTFOLIO, None, 0.0, {}),
    ("prior-only,    GMM",                       _GMM,       None, 0.0, {}),
    # The host twin of the device prior-only adaptive_cartesian row, which sets the worst |z|
    # in the committed table.  --n-max from the gate, not repeated here.
    ("prior-only,    adaptive_cartesian",        _AC,        None, 0.0,
     dict(n_max=gate._AC_N_MAX)),
    ("A=0.75 B=0,    AV",                        _AV,        0.75, 0.0, {}),
    ("A=0.75 B=0,    portfolio",                 _PORTFOLIO, 0.75, 0.0, {}),
    ("A=0.75 B=0,    GMM",                       _GMM,       0.75, 0.0, {}),
    ("A=8    B=0,    AV",                        _AV,        8.0,  0.0, {}),
    ("A=8    B=0,    portfolio",                 _PORTFOLIO, 8.0,  0.0, {}),
    ("A=8    B=0,    GMM",                       _GMM,       8.0,  0.0, {}),
    ("A=0.75 B=3,    AV",                        _AV,        0.75, 3.0, {}),
    ("A=0.75 B=3,    portfolio",                 _PORTFOLIO, 0.75, 3.0, {}),
    ("A=0.75 B=3,    GMM",                       _GMM,       0.75, 3.0, {}),
    ("A=8    B=2,    AV (the survives-swap lanes)", _AV,     8.0,  2.0, {}),
    ("raw inclination contract (cosine sampler)", _AV,       8.0,  2.0,
     dict(incl_is_cosine=True, extra=("--inclination-cosine-sampler",))),
    ("time-marginalized portfolio",              _PORTFOLIO, 8.0,  2.0,
     dict(extra=("--time-marginalization",))),
    ("distance-marginalized",                    _AV,        8.0,  2.0, {"_needs_dmarg": True}),
    ("adaptive_cartesian, --n-max 60000",
     _AC, 8.0, 2.0, dict(n_max=gate._AC_N_MAX)),
    # DEVICE lanes.  They need --gpu-slot, and are refused rather than skipped without it, for
    # the same reason as a --lane typo: a table that quietly stops covering eight of its rows
    # under a summary line that reads clean is worse than no table.
    ("GPU prior-only, AV",      _AV,  None, 0.0, {"_needs_gpu": True}),
    ("GPU prior-only, GMM",     _GMM, None, 0.0, {"_needs_gpu": True}),
    ("GPU A=8    B=2, AV",      _AV,  8.0,  2.0, {"_needs_gpu": True}),
    # NOT A=8 B=2: that is the n_eff lottery recorded at the end of the gate's CALIBRATION
    # section.  Mirrors the CPU "A=0.75 B=3, GMM" row instead.
    ("GPU A=0.75 B=3, GMM",     _GMM, 0.75, 3.0, {"_needs_gpu": True}),
    # portfolio and adaptive_cartesian could not run on a device at all until the host/device
    # conversions in mcsamplerGPU.compute_hist and mcsampler.integrate.  Each mirrors its host
    # twin: portfolio the plain-portfolio lanes above, adaptive_cartesian its own --n-max, taken
    # from the gate rather than repeated here.
    ("GPU prior-only, portfolio",  _PORTFOLIO, None, 0.0, {"_needs_gpu": True}),
    ("GPU A=0.75 B=3, portfolio",  _PORTFOLIO, 0.75, 3.0, {"_needs_gpu": True}),
    ("GPU prior-only, adaptive_cartesian", _AC, None, 0.0,
     dict(_needs_gpu=True, n_max=gate._AC_N_MAX)),
    ("GPU A=8    B=2, adaptive_cartesian", _AC, 8.0, 2.0,
     dict(_needs_gpu=True, n_max=gate._AC_N_MAX)),
    # mcsamplerGPU STANDALONE, and the ILE driver's own default --sampler-method.  Device only:
    # it reaches the compute_hist conversion through its own integrate/integrate_log adaptation
    # blocks rather than through a portfolio member, and that route has no other lane.  It stops
    # on the BUDGET at --n-max 20000 with the factor on, like adaptive_cartesian; measured, not
    # assumed -- see _LANE_KW in the gate.
    ("GPU prior-only, adaptive_cartesian_gpu", _ACG, None, 0.0,
     dict(_needs_gpu=True, n_max=gate._AC_N_MAX)),
    ("GPU A=8    B=2, adaptive_cartesian_gpu", _ACG, 8.0, 2.0,
     dict(_needs_gpu=True, n_max=gate._AC_N_MAX)),
]


def _dmarg_extra(event, out):
    import subprocess
    table = out / "marg_lookup.npz"
    proc = subprocess.run(
        [sys.executable, gate.MARG_TABLE_TOOL] + gate.DMARG_TABLE_ARGS + ["--out", str(table)],
        cwd=str(out), env=gate._child_env(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=1800)
    if proc.returncode != 0 or not table.exists():
        raise SystemExit("util_InitMargTable failed:\n%s" % proc.stdout.decode()[-1500:])
    return ("--distance-marginalization",
            "--distance-marginalization-lookup-table", str(table),
            "--d-min", "100", "--d-max", "1000",
            "--time-marginalization", "--vectorized", "--gpu", "--force-xpy")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--lane", default=None, help="substring; measure only matching lanes")
    ap.add_argument("--gpu-slot", default=None,
                    help="CUDA slot for the GPU lanes, as the CHILD should see it.  Probe it "
                         "first: the CIT slot map moves and cupy 12.0.0 cannot build a kernel "
                         "for the Blackwell cards.  Without it the GPU lanes are refused.")
    args = ap.parse_args()

    # Lane selection BEFORE anything expensive or on-disk.  It depends on nothing but argv, and
    # running it after mkdtemp/build_event meant a --lane typo cost a full fixture build and
    # then leaked the directory: the refusal path below neither prints nor removes it, and on
    # the CIT nodes that directory is on /, which is 20-22 GB.
    #
    # The index is carried alongside each lane, and it is the index into LANES rather than into
    # this filtered list, so --lane does not renumber the output tags.
    selected = [(i, L) for i, L in enumerate(LANES) if not args.lane or args.lane in L[0]]
    # WITHOUT --gpu-slot the device rows are dropped, LOUDLY, and the rest are measured.
    # Refusing outright broke the one invocation two comments tell you to run
    # ("RE-DERIVE THIS TABLE, do not trust it" -> `make_e2e_calibration.py --seeds 8`): it
    # exited 1 having measured nothing, and no single --lane selects the 17 host rows and
    # excludes the 4 device ones, so the committed host table could not be regenerated at all.
    # A comment demanding re-derivation, next to a command that cannot re-derive, is the exact
    # unfalsifiable claim this script exists to remove.
    #
    # Naming them ON STDERR matters: the table below is built from stdout, so a dropped row
    # cannot slip into it, and a person still sees which rows were not measured.
    if args.gpu_slot is None:
        dropped = [L[0].strip() for _i, L in selected if L[4].get("_needs_gpu")]
        if dropped:
            print("# NOT MEASURED (no --gpu-slot): %s" % ", ".join(dropped), file=sys.stderr)
            selected = [(i, L) for i, L in selected if not L[4].get("_needs_gpu")]
        if dropped and not selected:
            # Dropping every selected row and then printing a clean summary over nothing is
            # the shape this whole branch exists to remove, so that case is a refusal.
            raise SystemExit("--gpu-slot is required: every lane selected%s needs a device:"
                             "\n  %s\nProbe a slot this cupy can build a kernel for."
                             % ((" by --lane %r" % args.lane) if args.lane else "",
                                "\n  ".join(dropped)))
    if not selected:
        # Exiting 0 having measured nothing, under a summary line that reads like a clean
        # result, is the exact shape this whole branch exists to remove.
        raise SystemExit("--lane %r matched none of:\n  %s"
                         % (args.lane, "\n  ".join(L[0].strip() for L in LANES)))

    import pathlib
    out = pathlib.Path(tempfile.mkdtemp(prefix="e2e_calibration_"))
    event = gate.build_event(out)
    if event is None:
        raise SystemExit(
            "could not build the fixture (lal_path2cache missing or failed); the empty "
            "directory is at %s" % out)

    seeds = [args.first_seed + i for i in range(args.seeds)]
    print("#   lane                                      max |z|   max sigma   min n_eff")
    worst_z = worst_s = 0.0
    least_n = float("inf")
    for idx, (label, sampler, a, b, kw) in selected:
        kw = dict(kw)
        # Both of these EXTEND `extra` rather than assigning it, so a lane that carries its
        # own options does not lose them, and a lane needing both keeps both.
        if kw.pop("_needs_dmarg", False):
            kw["extra"] = tuple(kw.get("extra", ())) + _dmarg_extra(event, out)
        if kw.pop("_needs_gpu", False):
            # expect_device=True, so a lane that fell back to the host FAILS instead of
            # contributing host numbers to a row labelled GPU.
            kw.update(cuda=args.gpu_slot, expect_device=True,
                      extra=tuple(kw.get("extra", ())) + gate._GPU_FLAGS)
        exact = 0.0 if a is None else gate._exact(a, b)
        zs, sigs, neffs = [], [], []
        for seed in seeds:
            # `idx` from enumerate, NOT LANES.index(...): index is a first-match lookup, so two
            # identical lane rows would silently share a tag and overwrite each other's ILE
            # output directory.  LANES.index happened to be correct for every row it ever had;
            # it is wrong the moment one is duplicated, which is the kind of edit this table
            # invites, and nothing would have said so.
            tag = "cal_%d_%d" % (idx, seed)
            lnL, sigma, neff = gate._run_ile(event, tag, sampler, a_coeff=a, b_coeff=b,
                                             seed=seed, **kw)
            zs.append(abs((lnL - exact) / sigma))
            sigs.append(sigma)
            neffs.append(neff)
        worst_z = max(worst_z, max(zs))
        worst_s = max(worst_s, max(sigs))
        least_n = min(least_n, min(neffs))
        print("#   %-44s%5.2f      %6.4f      %6.0f"
              % (label, max(zs), max(sigs), min(neffs)))
    print("#")
    print("# across the %d lane(s) MEASURED above: worst |z| %.2f, worst sigma %.4f, "
          "least n_eff %.0f" % (len(selected), worst_z, worst_s, least_n))
    # Printed, not deleted: a failing lane is worth inspecting.  /tmp is small on the CIT
    # nodes, so clean up when you are done.
    print("# fixture kept at %s   (rm -rf it when done)" % out)


if __name__ == "__main__":
    main()
