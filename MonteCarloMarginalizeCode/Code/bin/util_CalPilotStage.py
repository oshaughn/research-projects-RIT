#! /usr/bin/env python
"""
util_CalPilotStage.py

One-shot orchestrator for a calibration PILOT stage (Option C; see
RIFT/calmarg/DESIGN_adaptive_driver.md).  This is the single executable the DAG runs as
`calpilot_N`, in parallel with iteration N's CIP/puff.  It performs, in sequence:

  1. HARVEST   top-fraction high-lnL intrinsic points from iteration N's *.composite
               -> a small sim-xml grid                       (util_CalHarvestGrid.py)
  2. DUMP      run ILE on that grid with --calibration-dump-responsibilities (cheap: no
               extrinsic sampler), optionally seeded from the previous consolidated
               proposal (refinement)                         (integrate_likelihood_...)
  3. FIT       fit a (auto-tempered) Gaussian cal proposal    (util_CalPilotFit.py)
  4. CONSOLIDATE -> the proposal breadcrumb that seeds wide_{N+1}  (util_CalConsolidate.py)

The ILE command line is taken from the run's args_ile.txt (the same args the wide jobs
use); pilot-specific flags (--sim-xml, --n-events-to-analyze, --output-file,
--calibration-dump-responsibilities, --calibration-proposal-breadcrumb) are stripped and
re-supplied.  Steps 1-4 run as subprocesses so this composes with the existing tools.

  util_CalPilotStage.py --composite consolidated_3.composite --ile-args-file args_ile.txt \
      --iteration 3 --output-breadcrumb cal_consolidated_3.npz \
      [--prev-breadcrumb cal_consolidated_2.npz] [--top-fraction 0.05] [--max-points 32]
"""
from __future__ import division

import sys
import os
import shlex
import argparse
import subprocess

# tokens we re-supply ourselves -> strip them (and their value) from the inherited ILE args
_STRIP_OPTS = {"--sim-xml", "--n-events-to-analyze", "--output-file", "--event",
               "--calibration-dump-responsibilities", "--calibration-proposal-breadcrumb"}


def strip_opts(arg_str, strip=_STRIP_OPTS):
    """Remove `--opt value` pairs (and bare flags) for opts in `strip` from an arg string."""
    toks = shlex.split(arg_str)
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in strip:
            i += 1
            # skip a following value if it is not itself an option
            if i < len(toks) and not toks[i].startswith("--"):
                i += 1
            continue
        out.append(t)
        i += 1
    return " ".join(out)


def _which(name):
    from shutil import which
    return which(name) or name


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--composite", required=True)
    p.add_argument("--ile-args-file", required=True, help="args_ile.txt (the wide ILE command)")
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--output-breadcrumb", required=True, help="consolidated proposal for wide_{N+1}")
    p.add_argument("--prev-breadcrumb", default=None, help="consolidation_{N-1} (refinement seed)")
    p.add_argument("--top-fraction", type=float, default=0.05)
    p.add_argument("--max-points", type=int, default=32)
    p.add_argument("--beta", type=float, default=None, help="passed to util_CalPilotFit (default auto-temper)")
    p.add_argument("--workdir", default=".", help="scratch dir for intermediate products")
    p.add_argument("--ile-exe", default=None)
    opts = p.parse_args(argv)

    wd = opts.workdir
    it = opts.iteration
    grid = os.path.join(wd, "cal_pilot_grid_%d.xml.gz" % it)
    resp = os.path.join(wd, "cal_pilot_resp_%d.npz" % it)
    prop = os.path.join(wd, "cal_proposal_%d.npz" % it)

    # --- 1. harvest -----------------------------------------------------------------
    subprocess.check_call([sys.executable, _which("util_CalHarvestGrid.py"),
                           "--composite", opts.composite, "--output-grid", grid,
                           "--top-fraction", str(opts.top_fraction),
                           "--max-points", str(opts.max_points)])

    # how many points did we harvest? (drives --n-events-to-analyze)
    import RIFT.lalsimutils as lalsimutils
    n_pts = len(lalsimutils.xml_to_ChooseWaveformParams_array(grid))

    # --- 2. ILE dump (cheap: skips the extrinsic sampler) ---------------------------
    with open(opts.ile_args_file) as f:
        ile_args = f.read().strip()
    # args_ile.txt convention: first token is the exe; the rest are args
    toks = shlex.split(ile_args)
    ile_exe = opts.ile_exe or _which("integrate_likelihood_extrinsic_batchmode")
    if toks and (toks[0].endswith("integrate_likelihood_extrinsic_batchmode") or toks[0].startswith("integrate")):
        rest = " ".join(toks[1:])
    else:
        rest = ile_args
    rest = strip_opts(rest)
    cmd = [ile_exe] + shlex.split(rest) + [
        "--sim-xml", grid, "--n-events-to-analyze", str(n_pts),
        "--output-file", os.path.join(wd, "cal_pilot_out_%d" % it),
        "--calibration-dump-responsibilities", resp]
    if opts.prev_breadcrumb and os.path.exists(opts.prev_breadcrumb):
        cmd += ["--calibration-proposal-breadcrumb", opts.prev_breadcrumb]   # refinement
    print(" [calpilot %d] DUMP: %s" % (it, " ".join(cmd)))
    subprocess.check_call(cmd)

    # --- 3. fit ----------------------------------------------------------------------
    fit_cmd = [sys.executable, _which("util_CalPilotFit.py"), "--dump", resp,
               "--output-breadcrumb", prop, "--iteration", str(it)]
    if opts.beta is not None:
        fit_cmd += ["--beta", str(opts.beta)]
    subprocess.check_call(fit_cmd)

    # --- 4. consolidate -------------------------------------------------------------
    subprocess.check_call([sys.executable, _which("util_CalConsolidate.py"),
                           "--breadcrumb", prop, "--output-breadcrumb", opts.output_breadcrumb,
                           "--iteration", str(it)])
    print(" [calpilot %d] wrote consolidated proposal -> %s" % (it, opts.output_breadcrumb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
