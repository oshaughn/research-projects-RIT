#!/usr/bin/env python3
"""End-to-end test suite for the v2 simulation archive against a real
SLURM cluster. Mirrors `_simmgr_v2_e2e.py` (the condor suite) using the
same set of correctness checks; configured via `RIFT_TEST_SLURM=1` and
the wrapper at `.travis/test-simulation-manager-v2-slurm.sh`.

Test cases (each independent):
  case_1_single_sim_single_level   smallest round trip
  case_2_chained_levels            level N must see level <N's outputs;
                                    the slurm chain is built via
                                    --dependency=afterok
  case_3_multi_sim_batch           several sims at once
  case_4_refinement                bump 1 -> 3 and refine
  case_5_dedup                     re-request returns same name

Assumes compute nodes share the archive's filesystem (the standard
HPC SLURM model). For sites without a shared FS, configure your
SLURM workers to bind-mount the archive directory (Singularity:
`singularity exec -B <archive>:<archive>`) or use sbcast in a prelude;
see DESIGN.md.

Run from a SLURM submit host:

    RIFT_TEST_SLURM=1 .travis/test-simulation-manager-v2-slurm.sh

Direct invocation:

    python3 .travis/_simmgr_v2_slurm_e2e.py \\
        --base /tmp/simmgr_v2_slurm \\
        --partition compute \\
        --time-limit 00:10:00 \\
        --timeout 600

Optional:
    --account <slurm_account>
    --qos <slurm_qos>
    --prelude 'module load python/3.11'   # shell snippet before the bootstrap
    --python-executable python3
    --only case_2_chained_levels          # restrict to specific cases
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from RIFT.simulation_manager.database import (
    Archive, Manifest, SlurmRequestQueue, SlurmRunQueue,
    StatusRecord, make_queues_from_manifest,
)


# ---------------------------------------------------------------------------
# Frozen generator (stdlib-only; runs on compute nodes via the bootstrap)
# ---------------------------------------------------------------------------

def my_generator(params, sim_dir, level, prev_levels):
    """Same shape as the condor suite. Records prev_count so case 2
    can prove the dependency chain ferried prior level files into
    level N's view."""
    import json, math, os
    val = float(params) * math.sqrt(2.0) * int(level)
    seen = []
    for path in prev_levels:
        if os.path.exists(path):
            seen.append(os.path.basename(path))
    out = os.path.join(sim_dir, "level_{}.json".format(level))
    with open(out, "w") as f:
        json.dump({"level": int(level), "value": val,
                   "prev_count": len(prev_levels),
                   "prev_levels_seen": seen,
                   "params": params}, f)


# ---------------------------------------------------------------------------
# Test framework
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.duration_s = 0.0

    def __repr__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return "[{verdict}] {name} ({d:.1f}s){msg}".format(
            verdict=verdict, name=self.name, d=self.duration_s,
            msg=("  -- " + self.message) if self.message else "")


def run_case(name: str, fn, *args, **kwargs) -> TestResult:
    print("\n=== {} ===".format(name))
    result = TestResult(name)
    t0 = time.time()
    try:
        msg = fn(*args, **kwargs)
        result.passed = True
        result.message = msg or ""
    except AssertionError as exc:
        result.message = "ASSERTION: {}".format(exc)
    except Exception as exc:
        result.message = "{}: {}".format(type(exc).__name__, exc)
        traceback.print_exc()
    result.duration_s = time.time() - t0
    print(result)
    return result


# ---------------------------------------------------------------------------
# Archive factory
# ---------------------------------------------------------------------------

def make_archive(base: Path, opts: argparse.Namespace, name: str) -> Archive:
    base.mkdir(parents=True, exist_ok=True)
    extra: Dict[str, Any] = {
        "partition":         opts.partition,
        "time_limit":        opts.time_limit,
        "request_memory":    int(opts.request_memory),
        "python_executable": opts.python_executable,
    }
    if opts.account:
        extra["account"] = opts.account
    if opts.qos:
        extra["qos"] = opts.qos
    if opts.prelude:
        extra["prelude"] = opts.prelude

    manifest = Manifest.new(
        name=name,
        request_queue_kind="slurm",
        run_queue_kind="slurm",
        request_queue_extra={},
        run_queue_extra=extra,
    )
    archive = Archive(base_location=base / name, manifest=manifest,
                      generator_spec=my_generator)
    req_q, run_q = make_queues_from_manifest(archive)
    archive.request_queue = req_q
    archive.run_queue = run_q
    return archive


# ---------------------------------------------------------------------------
# Polling helper
# ---------------------------------------------------------------------------

def wait_for(archive: Archive, sim_names: List[str], timeout: float,
             *, poll_interval: float = 5.0,
             target_status: str = "complete") -> bool:
    deadline = time.time() + timeout
    last_log = 0.0
    while True:
        archive.refresh_status_from_disk()
        if archive.run_queue is not None:
            try:
                archive.run_queue.poll(archive, sim_names)
            except Exception as exc:
                print("  (poll failed: {} — continuing on disk-presence)".format(exc))
        statuses = {n: archive.get_status(n) for n in sim_names}
        if all(s == target_status for s in statuses.values()):
            return True
        if any(s == "stuck" for s in statuses.values()):
            print("  STUCK:", statuses)
            return False
        if time.time() - last_log > 15:
            print("  ... waiting; statuses:", statuses)
            last_log = time.time()
        if time.time() > deadline:
            print("  TIMEOUT; final statuses:", statuses)
            return False
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def case_1_single_sim_single_level(opts):
    archive = make_archive(opts.base, opts, "case_1")
    name = archive.register(0.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [name], opts.timeout, poll_interval=opts.poll_interval):
        raise AssertionError("sim did not complete")
    out = archive.sim_dir(name) / "level_1.json"
    assert out.exists(), "level_1.json missing at canonical path"
    data = json.loads(out.read_text())
    assert abs(data["value"] - 0.5 * math.sqrt(2.0)) < 1e-12, data
    assert data["prev_count"] == 0, data
    return "value={value:.6f} prev_count={prev_count}".format(**data)


def case_2_chained_levels(opts):
    archive = make_archive(opts.base, opts, "case_2")
    name = archive.register(1.5, target_level=3)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [name], opts.timeout, poll_interval=opts.poll_interval):
        raise AssertionError("chained sim did not complete")
    sd = archive.sim_dir(name)
    for lvl in (1, 2, 3):
        out = sd / "level_{}.json".format(lvl)
        assert out.exists(), "level_{}.json missing".format(lvl)
        data = json.loads(out.read_text())
        # Critical: --dependency=afterok must give level N access to
        # level <N's outputs in the shared FS.
        assert data["prev_count"] == lvl - 1, (
            "level {}: prev_count={} (expected {}). "
            "--dependency chain broken?".format(
                lvl, data["prev_count"], lvl - 1))
        if lvl > 1:
            expected_seen = ["level_{}.json".format(i) for i in range(1, lvl)]
            assert sorted(data["prev_levels_seen"]) == sorted(expected_seen)
    return "3 levels chained, --dependency=afterok works"


def case_3_multi_sim_batch(opts):
    archive = make_archive(opts.base, opts, "case_3")
    names = [archive.register(p, target_level=2)
             for p in (0.1, 0.2, 0.3, 0.4)]
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, names, opts.timeout, poll_interval=opts.poll_interval):
        raise AssertionError("multi-sim batch did not complete")
    for n in names:
        for lvl in (1, 2):
            out = archive.sim_dir(n) / "level_{}.json".format(lvl)
            assert out.exists(), "{}/level_{}.json missing".format(n, lvl)
        data = json.loads((archive.sim_dir(n) / "level_2.json").read_text())
        ref = float(data["params"]) * math.sqrt(2.0) * 2
        assert abs(data["value"] - ref) < 1e-12, (data, ref)
    return "{} sims x 2 levels = {} files all correct".format(
        len(names), len(names) * 2)


def case_4_refinement(opts):
    archive = make_archive(opts.base, opts, "case_4")
    name = archive.register(0.8, target_level=1)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [name], opts.timeout, poll_interval=opts.poll_interval):
        raise AssertionError("level-1 sim did not complete")
    archive.refine(name, target_level=3)
    assert archive.get_status(name) == "refine_ready", archive.get_status(name)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [name], opts.timeout, poll_interval=opts.poll_interval):
        raise AssertionError("refinement did not complete")
    sd = archive.sim_dir(name)
    for lvl in (1, 2, 3):
        out = sd / "level_{}.json".format(lvl)
        assert out.exists(), "post-refine level_{}.json missing".format(lvl)
    data3 = json.loads((sd / "level_3.json").read_text())
    assert data3["prev_count"] == 2, data3
    return "refine 1 -> 3 produced levels 2 & 3 with correct chain"


def case_5_dedup(opts):
    archive = make_archive(opts.base, opts, "case_5")
    n1 = archive.register(2.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [n1], opts.timeout, poll_interval=opts.poll_interval):
        raise AssertionError("first registration did not complete")
    n2 = archive.register(2.5, target_level=1)
    assert n2 == n1, "dedup failed: got {} and {}".format(n1, n2)
    sub_count = len(list((archive.base / "run_queue" / "submit_files").iterdir()))
    assert sub_count == 1, (
        "expected 1 sbatch script, got {}".format(sub_count))
    return "re-request returned existing name {}, no extra sbatch".format(n1)


# ---------------------------------------------------------------------------

CASES = [
    case_1_single_sim_single_level,
    case_2_chained_levels,
    case_3_multi_sim_batch,
    case_4_refinement,
    case_5_dedup,
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--partition", required=True,
                    help="SLURM partition / queue name")
    ap.add_argument("--time-limit", default="00:10:00",
                    help="--time= for each sbatch (default 10 minutes)")
    ap.add_argument("--account", default=os.environ.get("SLURM_ACCOUNT"))
    ap.add_argument("--qos", default=os.environ.get("SLURM_QOS"))
    ap.add_argument("--prelude", default="",
                    help="Shell snippet inserted before the bootstrap "
                         "(`module load python/3.11`, virtualenv activate, etc.)")
    ap.add_argument("--python-executable", default="python3")
    ap.add_argument("--request-memory", default=2048,
                    help="MB; default 2048")
    ap.add_argument("--timeout", type=float, default=600,
                    help="per-case timeout in seconds; default 600")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    args = ap.parse_args(argv)

    if args.base.exists():
        shutil.rmtree(args.base)
    args.base.mkdir(parents=True)

    cases = CASES
    if args.only:
        cases = [c for c in CASES if c.__name__ in args.only]

    print("RIFT simulation_manager v2 archive — slurm end-to-end suite")
    print("============================================================")
    print(" base               :", args.base)
    print(" partition          :", args.partition)
    print(" time_limit         :", args.time_limit)
    print(" account            :", args.account or "(none)")
    print(" qos                :", args.qos or "(none)")
    print(" python_executable  :", args.python_executable)
    print(" prelude            :", args.prelude or "(none)")
    print(" timeout/case       :", args.timeout, "s")
    print(" cases              :", [c.__name__ for c in cases])
    print()

    results = [run_case(c.__name__, c, args) for c in cases]

    print("\n============================================================")
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    for r in results:
        print(" ", r)
    print("\n {} passed, {} failed.".format(n_pass, n_fail))

    if n_fail == 0 and not args.keep:
        shutil.rmtree(args.base, ignore_errors=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
