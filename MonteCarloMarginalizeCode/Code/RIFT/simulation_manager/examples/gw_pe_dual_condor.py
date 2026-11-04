"""Worked example: 2D GW parameter inference, dual-condor topology, with
iterative refinement.

This script demonstrates the *data flow* of the dual-condor design from
DESIGN.md without needing two real condor pools — it uses
LocalRequestQueue + LocalRunQueue under the hood so it runs in any
Python environment. The structure of the planner loop (scout → submit
to run pool → poll → refine → iterate) is the part that maps directly
to a real DualCondorRequestQueue + DualCondorRunQueue deployment.

Domain:
    Each simulation evaluates a (fake) marginal log-likelihood at one
    point in the (chirp mass, symmetric mass ratio) plane. The
    parameter is `params = {"mc": ..., "eta": ...}`. The output at
    each level is a JSON file with the level's MC samples and an
    aggregate (mean, stderr). Higher level == more samples == tighter
    answer. The summarizer folds all completed levels into a single
    rolling estimate.

Workflow demonstrated:
    1. Build the archive, freeze generator/summarizer/same_q/lookup_key.
    2. PASS 1: register a 3x3 (mc, eta) grid at level=1 (cheap survey).
       Submit, run, collect results.
    3. Identify the top-3 highest lnL points from the index.
    4. PASS 2: REFINE those three points to level=4 (expensive,
       precise). Submit, run, collect.
    5. Print a final report showing the refined sims have lower stderr.
    6. Verify rehydration: re-open the archive and recover everything.

Run:
    python -m RIFT.simulation_manager.examples.gw_pe_dual_condor

In a real dual-condor deployment, the *planner* function below would
itself be a condor DAG node on the request pool. Each call to
`req_q.submit_pending(archive)` would translate to a
`condor_submit_dag -name <run_pool>` call against the run pool's
schedd; `req_q.poll(archive)` would query that schedd via the cached
htcondor bindings. Output files arrive on the shared filesystem; the
planner reads the index and decides what's next.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from RIFT.simulation_manager.database import (
    Archive, Manifest, LocalRequestQueue, LocalRunQueue, StatusRecord,
)


# ---------------------------------------------------------------------------
# Frozen code: generator / summarizer / same_q / lookup_key
# ---------------------------------------------------------------------------
# These are captured by inspect.getsource() and written to <archive>/code/.
# They MUST be self-contained: stdlib imports only, no closure captures.

def my_generator(params, sim_dir, level, prev_levels):
    """Stub ILE evaluation at a single (mc, eta) point.

    Each level adds 500 fresh MC samples on top of any produced at
    prior levels (incremental refinement). The level file stores ONLY
    its new samples, so summing across `prev_levels` recovers the full
    sample history without double-counting. The underlying surface is
    an analytic 2D Gaussian peaked at (mc=27.5, eta=0.24).
    """
    import json, math, os, random
    mc, eta = float(params["mc"]), float(params["eta"])
    true_lnL = -0.5 * (((mc - 27.5) / 1.5) ** 2 + ((eta - 0.24) / 0.02) ** 2)
    # Read NEW samples from each prior level and concatenate to get
    # the cumulative sample set so far.
    prior = []
    for path in prev_levels:
        if os.path.exists(path):
            with open(path) as f:
                prior.extend(json.load(f).get("new_samples", []))
    rng = random.Random((hash((mc, eta, level)) & 0xffffffff))
    new_samples = [true_lnL + rng.gauss(0.0, 1.0) for _ in range(500)]
    cumulative = prior + new_samples
    n = len(cumulative)
    mean = sum(cumulative) / n
    var = sum((s - mean) ** 2 for s in cumulative) / max(n - 1, 1)
    stderr = math.sqrt(var / n)
    out = os.path.join(sim_dir, "level_{}.json".format(level))
    with open(out, "w") as f:
        json.dump({"level": level,
                   "new_samples": new_samples,         # only this level's contribution
                   "cumulative_n": n,
                   "lnL_mean": mean,
                   "lnL_stderr": stderr}, f)


def my_summarizer(sim_dir, params, levels=None):
    """Roll completed levels into a single aggregate lnL estimate.
    Each per-level file stores ONLY that level's new samples, so the
    summarizer must accumulate across all of them. The latest level's
    `lnL_mean`/`lnL_stderr` already reflect the full cumulative set
    (the generator computed them after concatenation), so we read those."""
    import json, os
    if levels:
        latest = levels[-1]
    else:
        cands = sorted(p for p in os.listdir(sim_dir)
                       if p.startswith("level_") and p.endswith(".json"))
        latest = os.path.join(sim_dir, cands[-1])
    with open(latest) as f:
        data = json.load(f)
    return {
        "lnL_mean":   data["lnL_mean"],
        "lnL_stderr": data["lnL_stderr"],
        "n_samples":  data["cumulative_n"],
        "level":      data["level"],
    }


def my_same_q(a, b):
    """Two (mc, eta) points are 'the same simulation' if both
    coordinates agree to ~1e-3 absolute. Tolerance below the
    summarizer's lnL precision so dedup matches the science semantics."""
    return (abs(float(a["mc"])  - float(b["mc"]))  < 1e-3 and
            abs(float(a["eta"]) - float(b["eta"])) < 1e-3)


def my_lookup_key(p):
    """Bucket by (mc, eta) rounded to 3 decimals — coarser than same_q
    so true duplicates always share a bucket, finer than the parameter
    spacing so unrelated points don't collide."""
    return (round(float(p["mc"]), 3), round(float(p["eta"]), 3))


# ---------------------------------------------------------------------------
# Planner — what would be a DAG node on the request pool
# ---------------------------------------------------------------------------

def initial_grid():
    """3x3 grid covering the peak. In a real planner this would come
    from prior bounds + a coarseness heuristic."""
    return [{"mc": mc, "eta": eta}
            for mc in (25.0, 27.5, 30.0)
            for eta in (0.20, 0.24, 0.245)]


def select_for_refinement(archive, k=3):
    """Pick the top-k highest-lnL sims from the index for refinement.
    Real planners would also look at lnL_stderr / convergence metrics."""
    rows = [r for r in archive.index.all() if r.get("summary")]
    rows.sort(key=lambda r: r["summary"]["lnL_mean"], reverse=True)
    return [r["name"] for r in rows[:k]]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    base = Path(args.base or (tempfile.gettempdir() + "/gw_pe_archive"))
    if base.exists():
        shutil.rmtree(base)

    # ----- Build archive -------------------------------------------------
    manifest = Manifest.new(
        name="gw_pe_2d_demo",
        request_queue_kind="condor",   # would be DualCondorRequestQueue in prod
        run_queue_kind="condor",       # would be DualCondorRunQueue in prod
        request_queue_extra={"pool": "request_pool@example",
                             "run_pool":   "run_pool@example",
                             "run_collector": "run_collector.example"},
        run_queue_extra={"pool": "run_pool@example",
                         "request_memory": 4096,
                         "request_disk": "4G"},
        summarizer_entrypoint="summarizer:my_summarizer",
        same_q_entrypoint="same_q:my_same_q",
        lookup_key_entrypoint="lookup_key:my_lookup_key",
        params_schema={"type": "dict",
                       "fields": {"mc": "float", "eta": "float"}},
        summary_schema={"fields": {"lnL_mean": "float",
                                   "lnL_stderr": "float",
                                   "n_samples": "int", "level": "int"}})
    archive = Archive(base_location=base, manifest=manifest,
                      generator_spec=my_generator,
                      summarizer_spec=my_summarizer,
                      same_q_spec=my_same_q,
                      lookup_key_spec=my_lookup_key)

    # In a real deployment we'd attach DualCondorRequestQueue +
    # DualCondorRunQueue here. For the example we use the local pair
    # so we can actually run end-to-end.
    run_q = LocalRunQueue()
    req_q = LocalRequestQueue(run_queue=run_q)
    archive.request_queue = req_q
    archive.run_queue = run_q

    # ----- PASS 1: register the survey grid at level=1 -------------------
    print("=== PASS 1: 3x3 survey grid @ level=1 ===")
    for params in initial_grid():
        archive.register(params, target_level=1)
    pending = req_q.submit_pending(archive)
    print(" submitted (and ran): {} sims".format(len(pending)))

    print("\nindex after PASS 1:")
    for row in archive.index.all():
        s = row["summary"]
        print("  {name}  mc={mc:5.2f} eta={eta:.3f}  "
              "lnL={lnL:7.3f} ± {stderr:5.3f} (lvl {lvl}, n={n})".format(
                  name=row["name"], mc=row["params"]["mc"],
                  eta=row["params"]["eta"],
                  lnL=s["lnL_mean"], stderr=s["lnL_stderr"],
                  lvl=s["level"], n=s["n_samples"]))

    # ----- Refinement strategy -------------------------------------------
    top = select_for_refinement(archive, k=3)
    print("\n=== Top-3 candidates for refinement: {} ===".format(top))
    pre_refine = {n: archive.index.by_name(n)["summary"]["lnL_stderr"]
                  for n in top}

    # ----- PASS 2: refine top-3 to level=4 -------------------------------
    print("\n=== PASS 2: refine to level=4 ===")
    for sim_name in top:
        bumped = archive.refine(sim_name, target_level=4)
        print("  refine({}) -> bumped={}, status={}".format(
            sim_name, bumped, archive.get_status(sim_name)))
    pending = req_q.submit_pending(archive)
    print(" submitted (and ran): {} sims".format(len(pending)))

    print("\nrefined entries:")
    for n in top:
        row = archive.index.by_name(n)
        s = row["summary"]
        before = pre_refine[n]
        print("  {name}  mc={mc:5.2f} eta={eta:.3f}  "
              "lnL={lnL:7.3f} ± {stderr:5.3f} (was ±{was:5.3f}, "
              "lvl {lvl}, n={n_samp})".format(
                  name=n, mc=row["params"]["mc"], eta=row["params"]["eta"],
                  lnL=s["lnL_mean"], stderr=s["lnL_stderr"], was=before,
                  lvl=s["level"], n_samp=s["n_samples"]))
        # Refined sims should have meaningfully smaller stderr.
        # 4 levels × 500 samples each = 2000 cumulative; expect
        # stderr to fall by ~sqrt(4) = 2x.
        assert s["lnL_stderr"] < before * 0.7, (
            "refinement should reduce stderr by ~sqrt(4) but got "
            "{} -> {}".format(before, s["lnL_stderr"]))
        assert row["current_level"] == 4
        assert row["target_level"] == 4

    # Non-refined sims remain at level=1.
    not_refined = [r for r in archive.index.all() if r["name"] not in top]
    for r in not_refined:
        assert r["current_level"] == 1, r

    # ----- Idempotent re-request -----------------------------------------
    print("\n=== Idempotent re-request ===")
    # Re-requesting one of the refined points at level <= current is a no-op.
    n_again = archive.register({"mc": 27.5, "eta": 0.24}, target_level=2)
    assert archive.get_status(n_again) == "complete"
    print("  re-register at level 2 of an already-level-4 sim -> no work,",
          "status={}".format(archive.get_status(n_again)))

    # Re-requesting at HIGHER level bumps target and goes to refine_ready.
    n_bump = archive.register({"mc": 27.5, "eta": 0.24}, target_level=6)
    assert archive.get_status(n_bump) == "refine_ready"
    print("  re-register at level 6 of an already-level-4 sim ->",
          "status={}".format(archive.get_status(n_bump)))
    req_q.submit_pending(archive)
    rec = StatusRecord.read(archive.sim_dir(n_bump))
    assert rec.data["current_level"] == 6, rec.data

    # ----- Rehydration ----------------------------------------------------
    print("\n=== Rehydration ===")
    archive2 = Archive(base_location=base)
    rows = archive2.index.all()
    print(" rehydrated {} sims; statuses:".format(len(rows)))
    for r in rows:
        print("  {} {} (target={}, current={})".format(
            r["name"], r["status"], r.get("target_level"), r.get("current_level")))
    # Frozen generator works with the new (params, sim_dir, level, prev_levels) signature.
    gen2 = archive2.load_generator()
    sd_test = archive2.base / "sims" / rows[0]["name"]
    # Don't actually re-run; just confirm callable + signature.
    import inspect
    sig = inspect.signature(gen2)
    assert {"params", "sim_dir", "level", "prev_levels"}.issubset(sig.parameters), sig
    print(" frozen generator signature OK:", sig)

    print("\nPASS: dual-condor flow + refinement end-to-end.")
    if not args.keep:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
