"""Worked example: embed a 'request a simulation' node in an existing
glue.pipeline DAG (e.g. a hyperpipeline workflow).

The pattern:

    1. Pre-build an archive elsewhere (manifest, frozen code, queue config).
    2. In the consuming workflow's DAG-builder script, instantiate a
       RequestSimulationJob pointing at that archive.
    3. For every (params, target_level) the workflow needs, call
       sim_job.make_node(...) and add it to the DAG. Wire downstream
       nodes as children — they'll block until the sim is ready.

When DAGMan runs the request-sim node, the request_sim CLI:
  * registers the sim in the archive (idempotent),
  * dispatches run-pool work via the configured request_queue if needed,
  * blocks (poll loop) until the sim's current_level >= target_level,
  * exits 0 on success (or non-zero on stuck/timeout).

Run:
    python -m RIFT.simulation_manager.examples.embed_in_existing_dag \\
        --archive /path/to/archive --out /tmp/parent.dag

If glue.pipeline isn't installed, this example prints a representative
submit-file fragment instead so the integration shape is still visible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from RIFT.simulation_manager.database import (
    Archive, Manifest, LocalRequestQueue, LocalRunQueue,
)


# ---- frozen code (matches gw_pe_dual_condor.py shape) ---------------------

def my_generator(params, sim_dir, level, prev_levels):
    import json, math, os, random
    mc, eta = float(params["mc"]), float(params["eta"])
    true_lnL = -0.5 * (((mc - 27.5) / 1.5) ** 2 + ((eta - 0.24) / 0.02) ** 2)
    rng = random.Random(hash((mc, eta, level)) & 0xffffffff)
    samples = [true_lnL + rng.gauss(0.0, 1.0) for _ in range(500 * level)]
    n = len(samples); mean = sum(samples)/n
    var = sum((s-mean)**2 for s in samples)/max(n-1,1)
    out = os.path.join(sim_dir, "level_{}.json".format(level))
    with open(out, "w") as f:
        json.dump({"level": level, "lnL_mean": mean,
                   "lnL_stderr": math.sqrt(var/n), "n_samples": n}, f)


def my_summarizer(sim_dir, params, levels=None):
    import json, os
    if levels:
        latest = levels[-1]
    else:
        cands = sorted(p for p in os.listdir(sim_dir)
                       if p.startswith("level_") and p.endswith(".json"))
        latest = os.path.join(sim_dir, cands[-1])
    with open(latest) as f:
        return json.load(f)


def my_same_q(a, b):
    return (abs(float(a["mc"]) - float(b["mc"])) < 1e-3 and
            abs(float(a["eta"]) - float(b["eta"])) < 1e-3)


def my_lookup_key(p):
    return (round(float(p["mc"]), 3), round(float(p["eta"]), 3))


# ---------------------------------------------------------------------------

def build_archive(base):
    base = Path(base)
    if base.exists():
        shutil.rmtree(base)
    manifest = Manifest.new(
        name="hyperpipe_demo",
        request_queue_kind="condor",
        run_queue_kind="condor",
        summarizer_entrypoint="summarizer:my_summarizer",
        same_q_entrypoint="same_q:my_same_q",
        lookup_key_entrypoint="lookup_key:my_lookup_key",
    )
    archive = Archive(base_location=base, manifest=manifest,
                      generator_spec=my_generator,
                      summarizer_spec=my_summarizer,
                      same_q_spec=my_same_q,
                      lookup_key_spec=my_lookup_key)
    return archive


def build_parent_dag_with_sim_nodes(archive, dag_path, points_of_interest,
                                    target_level=4):
    """Construct a representative parent DAG with one request-sim node
    per (mc, eta) point of interest. Returns (dag, sim_job) on the
    glue.pipeline path; (None, None) when glue.pipeline isn't present
    (and prints an indicative submit-file fragment instead)."""
    try:
        from glue import pipeline
    except ImportError:
        print("glue.pipeline not available — printing indicative",
              "submit fragment instead of building a DAG.")
        _print_indicative_submit(archive, target_level)
        return None, None

    from RIFT.simulation_manager.glue_compat import RequestSimulationJob

    dag = pipeline.CondorDAG(log=str(Path(dag_path).with_suffix('.log')))
    sim_job = RequestSimulationJob(
        archive_path=archive.base,
        log_dir=str(archive.base / "request_queue" / "logs"),
        request_memory=1024,
        accounting_group=os.environ.get("LIGO_ACCOUNTING"),
        accounting_group_user=os.environ.get("LIGO_USER_NAME"),
        timeout=3600.0,
        poll_interval=15.0,
    )
    sim_job.set_sub_file(str(Path(dag_path).parent / "request_sim.sub"))
    sim_job.write_sub_file()

    sim_nodes = []
    for params in points_of_interest:
        node = sim_job.make_node(params=params, target_level=target_level)
        dag.add_node(node)
        sim_nodes.append(node)

    # In a real hyperpipeline DAG, the workflow's downstream nodes
    # would `add_parent(node)` on each of these so they block until
    # the requested simulations are ready.

    dag.set_dag_file(str(Path(dag_path).with_suffix('')))
    dag.write_concrete_dag()
    print("wrote parent DAG to:", dag_path)
    print("  with", len(sim_nodes), "request-sim nodes")
    return dag, sim_job


def _print_indicative_submit(archive, target_level):
    """Show what the .sub file would look like, for environments
    without glue.pipeline."""
    import shlex
    argv_prefix = [sys.executable or "python3", "-m",
                   "RIFT.simulation_manager.cli.request_sim"]
    args = " ".join(shlex.quote(s) for s in (argv_prefix[1:] + [
        "--archive", str(archive.base),
        "--params", "$(macro_params)",
        "--target-level", "$(macro_target_level)",
        "--ensure",
    ]))
    print("\n# request_sim.sub (indicative)")
    print("universe       = vanilla")
    print("executable     = {}".format(argv_prefix[0]))
    print("arguments      = {}".format(args))
    print("getenv         = True")
    print("request_memory = 1024M")
    print("queue\n")
    print("# Per-node macros (one per point of interest):")
    print("# macro_params = '{\"mc\": 27.5, \"eta\": 0.24}'")
    print("# macro_target_level = {}".format(target_level))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive",
                    help="Path to archive (created if it doesn't exist)")
    ap.add_argument("--out", default="/tmp/hyperpipe_parent.dag",
                    help="Where to write the parent DAG")
    args = ap.parse_args()

    if args.archive:
        base = Path(args.archive)
        if not (base / "manifest.json").exists():
            archive = build_archive(base)
        else:
            archive = Archive(base_location=base)
    else:
        base = Path(tempfile.mkdtemp(prefix="hyperpipe_demo_"))
        shutil.rmtree(base)
        archive = build_archive(base)

    print("Using archive at:", archive.base)
    points = [{"mc": 25.0, "eta": 0.20},
              {"mc": 27.5, "eta": 0.24},
              {"mc": 30.0, "eta": 0.24}]

    # Demonstrate the no-DAG path too: drive the same archive directly
    # via the database API + LocalRunQueue, so we can show the request_sim
    # CLI's --check mode against a real archive state.
    archive.run_queue = LocalRunQueue()
    archive.request_queue = LocalRequestQueue(run_queue=archive.run_queue)
    for p in points:
        archive.register(p, target_level=2)
    archive.request_queue.submit_pending(archive)

    dag, job = build_parent_dag_with_sim_nodes(archive, args.out, points,
                                               target_level=2)

    # Demonstrate the --check CLI invocation against the archive we just
    # primed. This is what DAGMan would invoke when the parent DAG runs.
    print("\n# request_sim CLI --check examples (these are what DAGMan invokes):")
    for p in points:
        cmd = "python -m RIFT.simulation_manager.cli.request_sim " \
              "--archive {a} --params {pj!r} --target-level 2 --check".format(
                  a=archive.base, pj=json.dumps(p))
        print(" ", cmd)


if __name__ == "__main__":
    main()
