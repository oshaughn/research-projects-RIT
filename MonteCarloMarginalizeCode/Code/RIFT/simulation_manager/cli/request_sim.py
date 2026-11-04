"""CLI: ensure a simulation exists in an archive at a target level.

This is the executable invoked by a `RequestSimulationJob` node embedded
in a parent DAG (e.g. an existing hyperpipeline workflow). The contract
DAGMan expects: exit 0 iff the requested simulation is present and at
the requested level (or higher) in the archive.

Modes:
    --ensure  (default)  Register the sim, dispatch run-pool work if
                         needed, BLOCK until the sim's current_level
                         meets the target. Exit 0 on success.
    --check              Just check status; exit 0 iff already at
                         target level, else nonzero. Useful for a
                         non-blocking guard.
    --submit-async       Register and dispatch run-pool work, but DO
                         NOT wait. Exit 0 iff the dispatch succeeded.
                         The parent DAG must arrange a separate
                         synchronization mechanism.

This CLI is intended to run on the SUBMIT NODE (where the archive
lives). It uses the configured request_queue + run_queue from the
manifest to dispatch further condor work to the run pool when needed.

Run:
    python -m RIFT.simulation_manager.cli.request_sim \\
        --archive /path/to/archive \\
        --params '{"mc": 27.5, "eta": 0.24}' \\
        --target-level 4

When invoked from a glue.pipeline node, the parent DAG's submit
description supplies these as macros (--archive $(macro_archive),
--params $(macro_params), --target-level $(macro_target_level)).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True,
                    help="Path to an existing archive (must already be "
                         "initialized; this CLI does not create archives)")
    ap.add_argument("--params", required=True,
                    help="JSON-encoded parameters dict (or scalar)")
    ap.add_argument("--target-level", type=int, default=1)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--ensure", action="store_const", dest="mode",
                      const="ensure", help="(default) register, dispatch, "
                      "block until present at target level")
    mode.add_argument("--check", action="store_const", dest="mode",
                      const="check", help="exit 0 iff already complete; "
                      "do nothing else")
    mode.add_argument("--submit-async", action="store_const", dest="mode",
                      const="submit_async", help="register and dispatch, "
                      "but do not wait")
    ap.add_argument("--poll-interval", type=float, default=10.0,
                    help="seconds between status polls in --ensure mode")
    ap.add_argument("--timeout", type=float, default=None,
                    help="give up after this many seconds in --ensure mode")
    ap.set_defaults(mode="ensure")
    args = ap.parse_args(argv)

    # Imported here so a stale --help doesn't trip the (heavy) RIFT import.
    from RIFT.simulation_manager.database import Archive

    archive = Archive(base_location=args.archive)
    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print("ERROR: --params is not valid JSON: {}".format(exc), file=sys.stderr)
        return 2

    name = archive.register(params, target_level=args.target_level)
    row = archive.index.by_name(name)
    print("sim={} target_level={} current_level={} status={}".format(
        name, row.get("target_level"), row.get("current_level"),
        row.get("status")), file=sys.stderr)

    if args.mode == "check":
        return 0 if (row.get("status") == "complete"
                     and row.get("current_level", 0) >= args.target_level) else 1

    # Need work? Dispatch via the configured request_queue.
    needs_work = (row.get("current_level", 0) < args.target_level)
    if needs_work:
        if archive.request_queue is None:
            # Auto-resolve from the manifest's queue config.
            from RIFT.simulation_manager.database import make_queues_from_manifest
            try:
                req_q, run_q = make_queues_from_manifest(archive)
            except Exception as exc:
                print("ERROR: could not instantiate queues from manifest: {}"
                      .format(exc), file=sys.stderr)
                return 3
            archive.request_queue = req_q
            archive.run_queue = run_q
        try:
            archive.request_queue.submit_pending(archive)
        except Exception as exc:
            print("ERROR: submit_pending failed: {}".format(exc), file=sys.stderr)
            return 7

    if args.mode == "submit_async":
        return 0

    # --ensure: block until status is complete at the target level (or stuck).
    deadline = (time.time() + args.timeout) if args.timeout else None
    while True:
        archive2 = Archive(base_location=args.archive)
        row = archive2.index.by_name(name)
        if row is None:
            print("ERROR: sim {} disappeared from archive".format(name),
                  file=sys.stderr)
            return 4
        status = row.get("status")
        cur = row.get("current_level", 0)
        if status == "complete" and cur >= args.target_level:
            print("ready: sim={} level={}".format(name, cur), file=sys.stderr)
            return 0
        if status == "stuck":
            print("ERROR: sim {} is stuck".format(name), file=sys.stderr)
            return 5
        if deadline and time.time() > deadline:
            print("ERROR: timed out waiting for sim {} (status={}, "
                  "level {}/{})".format(name, status, cur,
                                        args.target_level), file=sys.stderr)
            return 6
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
