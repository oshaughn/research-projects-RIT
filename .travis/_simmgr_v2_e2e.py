#!/usr/bin/env python3
"""End-to-end test suite for the v2 simulation archive against a real
HTCondor schedd. Designed to deploy at CIT (or any LIGO/IGWN submit
host) and to exercise the OSG-style execution model where workers have
NO access to the submit-side filesystem — all I/O goes through condor's
transfer_input_files / transfer_output_files / transfer_output_remaps.

Test cases (run each independently; report a per-case verdict):

  case_1_single_sim_single_level
      Smallest possible round trip. Confirms a worker without submit-FS
      access can produce one level file and have it land at the
      canonical archive path.

  case_2_chained_levels
      Multi-level sim where level N must have access to level <N's output.
      The generator records prev_count = len(prev_levels); the test
      verifies prev_count == N-1, which proves transfer_input_files
      correctly chains prior level outputs in the DAG.

  case_3_multi_sim_batch
      Several sims at once. Confirms cluster ids / DAG node naming
      don't cross-contaminate.

  case_4_refinement
      Register at level 1, run, then refine to level 3. Verifies the
      refine transition + a re-submit produces the additional levels.

  case_5_dedup
      Re-request the same params at the same level — no extra work.

Run from a CIT submit host:

    RIFT_TEST_CONDOR=1 .travis/test-simulation-manager-v2-condor.sh

Or directly:

    python3 .travis/_simmgr_v2_e2e.py \\
        --base /tmp/simmgr_v2_e2e \\
        --accounting-group ligo.dev.o4.cbc.pe.lalinferencerapid \\
        --accounting-group-user albert.einstein \\
        --timeout 600

Optional knobs for OSG-targeted runs:
    --run-pool <schedd_name>          # cross-pool: condor_submit_dag -name
    --run-collector <collector_host>  # for poll() Schedd lookups
    --singularity-image <path|osdf>   # use vanilla+singularity
    --desired-sites <s1,s2,...>       # +DESIRED_SITES for OSG selection

The tests pass iff every case completes within --timeout and every
expected per-(sim, level) output file appears at its remapped location
on the submit node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from RIFT.simulation_manager.database import (
    Archive, Manifest, DualCondorRequestQueue, DualCondorRunQueue,
    StatusRecord, make_queues_from_manifest,
)


# ---------------------------------------------------------------------------
# Frozen generator (stdlib-only, runs inside the worker sandbox)
# ---------------------------------------------------------------------------
# Each call must be self-contained: stdlib imports only, no closure captures,
# no submit-side paths. The worker sees only what condor transferred in.

def my_generator(params, sim_dir, level, prev_levels):
    """Deterministic, stdlib-only generator. Output records:
        level         — the level we computed
        value         — params * sqrt(2) * level   (deterministic)
        prev_count    — len(prev_levels)           (proof of file transfer)
        prev_levels_seen — basename list           (also a proof point)
    """
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
# Test framework: tiny pass/fail with per-case timing
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
# Archive factory shared across cases
# ---------------------------------------------------------------------------

def make_archive(base: Path, opts: argparse.Namespace, name: str) -> Archive:
    base.mkdir(parents=True, exist_ok=True)
    extra: Dict[str, Any] = {
        "request_memory": int(opts.request_memory),
        "request_disk":   opts.request_disk,
        "getenv":         opts.getenv,
    }
    if opts.run_pool:
        extra["run_pool"] = opts.run_pool
    if opts.run_collector:
        extra["run_collector"] = opts.run_collector
    if opts.accounting_group:
        extra["accounting_group"] = opts.accounting_group
    if opts.accounting_group_user:
        extra["accounting_group_user"] = opts.accounting_group_user
    if opts.singularity_image:
        extra["use_singularity"] = True
        extra["singularity_image"] = opts.singularity_image
    extra_cmds: Dict[str, str] = {}
    if opts.desired_sites:
        extra_cmds["+DESIRED_SITES"] = '"{}"'.format(opts.desired_sites)
    if opts.undesired_sites:
        extra_cmds["+UNDESIRED_SITES"] = '"{}"'.format(opts.undesired_sites)
    if extra_cmds:
        extra["extra_condor_cmds"] = extra_cmds

    manifest = Manifest.new(
        name=name,
        request_queue_kind="condor",
        run_queue_kind="condor",
        request_queue_extra={"run_pool": opts.run_pool,
                             "run_collector": opts.run_collector},
        run_queue_extra=extra,
    )
    archive = Archive(base_location=base / name, manifest=manifest,
                      generator_spec=my_generator)
    req_q, run_q = make_queues_from_manifest(archive)
    archive.request_queue = req_q
    archive.run_queue = run_q
    return archive


# ---------------------------------------------------------------------------
# Polling helper — wait for sims to reach a target status, with timeout
# ---------------------------------------------------------------------------

def wait_for(archive: Archive, sim_names: List[str], timeout: float,
             *, target_status: str = "complete") -> bool:
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
        time.sleep(opts.poll_interval if isinstance(opts, argparse.Namespace)  # noqa
                  else 5.0)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def case_1_single_sim_single_level(opts):
    archive = make_archive(opts.base, opts, "case_1")
    name = archive.register(0.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [name], opts.timeout):
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
    if not wait_for(archive, [name], opts.timeout):
        raise AssertionError("chained sim did not complete")
    sd = archive.sim_dir(name)
    for lvl in (1, 2, 3):
        out = sd / "level_{}.json".format(lvl)
        assert out.exists(), "level_{}.json missing".format(lvl)
        data = json.loads(out.read_text())
        # The critical correctness check: at level N the worker MUST have
        # seen exactly N-1 prior level files via transfer_input_files.
        assert data["prev_count"] == lvl - 1, (
            "level {}: prev_count={} (expected {}). transfer chain broken?"
            .format(lvl, data["prev_count"], lvl - 1))
        if lvl > 1:
            expected_seen = ["level_{}.json".format(i) for i in range(1, lvl)]
            assert sorted(data["prev_levels_seen"]) == sorted(expected_seen), (
                "level {}: saw {!r}, expected {!r}".format(
                    lvl, data["prev_levels_seen"], expected_seen))
    return "3 levels chained, transfer correct at every level"


def case_3_multi_sim_batch(opts):
    archive = make_archive(opts.base, opts, "case_3")
    names = [archive.register(p, target_level=2)
             for p in (0.1, 0.2, 0.3, 0.4)]
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, names, opts.timeout):
        raise AssertionError("multi-sim batch did not complete")
    for n in names:
        for lvl in (1, 2):
            out = archive.sim_dir(n) / "level_{}.json".format(lvl)
            assert out.exists(), "{}/level_{}.json missing".format(n, lvl)
        # Spot-check the value at the highest level.
        data = json.loads((archive.sim_dir(n) / "level_2.json").read_text())
        ref = float(data["params"]) * math.sqrt(2.0) * 2
        assert abs(data["value"] - ref) < 1e-12, (data, ref)
    return "{} sims x 2 levels = {} files all correct".format(
        len(names), len(names) * 2)


def case_4_refinement(opts):
    archive = make_archive(opts.base, opts, "case_4")
    name = archive.register(0.8, target_level=1)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [name], opts.timeout):
        raise AssertionError("level-1 sim did not complete")
    # Bump.
    archive.refine(name, target_level=3)
    assert archive.get_status(name) == "refine_ready", archive.get_status(name)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [name], opts.timeout):
        raise AssertionError("refinement did not complete")
    sd = archive.sim_dir(name)
    for lvl in (1, 2, 3):
        out = sd / "level_{}.json".format(lvl)
        assert out.exists(), "post-refine level_{}.json missing".format(lvl)
    # Level 3's prev_count must include level 1 + level 2.
    data3 = json.loads((sd / "level_3.json").read_text())
    assert data3["prev_count"] == 2, data3
    return "refine 1 -> 3 produced levels 2 & 3 with correct transfer"


def case_5_dedup(opts):
    archive = make_archive(opts.base, opts, "case_5")
    n1 = archive.register(2.5, target_level=1)
    archive.request_queue.submit_pending(archive)
    if not wait_for(archive, [n1], opts.timeout):
        raise AssertionError("first registration did not complete")
    # Re-request: should return the same name with no work.
    n2 = archive.register(2.5, target_level=1)
    assert n2 == n1, "dedup failed: got {} and {}".format(n1, n2)
    # No new sub files produced (count submit files; should be 1 from the
    # first run).
    sub_count = len(list((archive.base / "run_queue" / "submit_files").iterdir()))
    assert sub_count == 1, (
        "expected 1 submit description, got {}".format(sub_count))
    return "re-request returned existing name {}, no extra submits".format(n1)


# ---------------------------------------------------------------------------
# Main
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
    ap.add_argument("--base", required=True, type=Path,
                    help="Workspace dir (will be created; cleaned on success "
                         "unless --keep)")
    ap.add_argument("--accounting-group",
                    default=os.environ.get("LIGO_ACCOUNTING"))
    ap.add_argument("--accounting-group-user",
                    default=os.environ.get("LIGO_USER_NAME"))
    ap.add_argument("--run-pool", default=None,
                    help="Schedd name passed to condor_submit_dag -name "
                         "(omit for local schedd)")
    ap.add_argument("--run-collector", default=None,
                    help="Collector host for cross-pool poll() lookups")
    ap.add_argument("--singularity-image", default=None,
                    help="If set, submit jobs use Singularity with this image")
    ap.add_argument("--desired-sites", default=None,
                    help="Comma-separated list of OSG sites "
                         "(+DESIRED_SITES classad value)")
    ap.add_argument("--undesired-sites", default=None,
                    help="Comma-separated list of OSG sites to avoid")
    ap.add_argument("--request-memory", default=2048,
                    help="MB; default 2048")
    ap.add_argument("--request-disk", default="2G",
                    help="default 2G")
    ap.add_argument(
        "--getenv",
        default=os.environ.get(
            "RIFT_GETENV",
            "LD_LIBRARY_PATH,PATH,PYTHONPATH,*RIFT*,LIBRARY_PATH"),
        help="condor 'getenv' value. Default is the OSG-blessed allowlist "
             "(many sites including CIT block getenv=True). Honors $RIFT_GETENV "
             "if set.")
    ap.add_argument("--timeout", type=float, default=600,
                    help="per-case timeout in seconds; default 600")
    ap.add_argument("--poll-interval", type=float, default=5.0,
                    help="seconds between schedd polls; default 5")
    ap.add_argument("--keep", action="store_true",
                    help="Don't delete --base on completion")
    ap.add_argument("--only", action="append", default=[],
                    help="Run only these cases (by function name; repeatable)")
    args = ap.parse_args(argv)
    global opts
    opts = args  # for wait_for's poll_interval reference

    if args.base.exists():
        shutil.rmtree(args.base)
    args.base.mkdir(parents=True)

    cases = CASES
    if args.only:
        cases = [c for c in CASES if c.__name__ in args.only]

    print("RIFT simulation_manager v2 archive — condor end-to-end suite")
    print("============================================================")
    print(" base               :", args.base)
    print(" accounting_group   :", args.accounting_group or "(none)")
    print(" accounting_group_user:", args.accounting_group_user or "(none)")
    print(" run_pool           :", args.run_pool or "(local schedd)")
    print(" run_collector      :", args.run_collector or "(local)")
    print(" singularity_image  :", args.singularity_image or "(none)")
    print(" desired_sites      :", args.desired_sites or "(none)")
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
