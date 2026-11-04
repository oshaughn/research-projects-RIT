"""CLI: archive admin operations.

Subcommands:
    verify           run Archive.verify, print the report (exit 0 iff healthy)
    rebuild-index    reconstruct index.jsonl from per-sim status.json files
    resummarize      re-run the manifest's summarizer for every sim
    unstick          clear stuck status; --all for bulk; --bump-memory
                     to also raise the per-sim request_memory

Run:
    python -m RIFT.simulation_manager.cli.admin verify --archive /path
    python -m RIFT.simulation_manager.cli.admin rebuild-index --archive /path
    python -m RIFT.simulation_manager.cli.admin resummarize --archive /path
    python -m RIFT.simulation_manager.cli.admin unstick --archive /path --name 7
    python -m RIFT.simulation_manager.cli.admin unstick --archive /path --all --bump-memory
"""

from __future__ import annotations

import argparse
import json
import sys


def _archive(args):
    from RIFT.simulation_manager.database import Archive
    return Archive(base_location=args.archive)


def cmd_verify(args):
    arch = _archive(args)
    report = arch.verify()
    if args.json:
        print(json.dumps(report, indent=2, default=str, sort_keys=True))
    else:
        print("manifest_ok        :", report["manifest_ok"])
        if report["manifest_issues"]:
            for issue in report["manifest_issues"]:
                print("                     -", issue)
        print("healthy            :", report["healthy"])
        print("complete_sims      :", report["complete_sims"])
        print("incomplete_sims    :", report["incomplete_sims"])
        print("stuck_sims         :", len(report["stuck_sims"]),
              report["stuck_sims"][:5] + (["..."] if len(report["stuck_sims"]) > 5 else []))
        print("missing_in_index   :", len(report["missing_in_index"]))
        print("orphan_in_index    :", len(report["orphan_in_index"]))
        print("status_drift       :", len(report["status_drift"]))
        print("missing_levels     :", len(report["missing_levels"]))
        print("extra_levels       :", len(report["extra_levels"]))
    return 0 if report["healthy"] else 1


def cmd_rebuild_index(args):
    arch = _archive(args)
    n = arch.rebuild_index()
    print("rebuilt index.jsonl with {} rows".format(n))
    return 0


def cmd_resummarize(args):
    arch = _archive(args)
    report = arch.resummarize_all(only_complete=args.only_complete)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        ok = sum(1 for v in report.values() if v == "ok")
        for v in ("no-summarizer", "no-levels"):
            n = sum(1 for x in report.values() if x == v)
            if n:
                print("  {}: {}".format(v, n))
        for sim, msg in report.items():
            if msg.startswith("error"):
                print("  {}: {}".format(sim, msg))
        print("ok: {} / {}".format(ok, len(report)))
    return 0


def cmd_unstick(args):
    arch = _archive(args)
    if args.all:
        names = arch.unstick_all(bump_memory=args.bump_memory,
                                 bump_factor=args.bump_factor)
        print("unstuck", len(names), "sims:", names[:10],
              ("..." if len(names) > 10 else ""))
    else:
        if not args.name:
            print("ERROR: --name required (or --all)", file=sys.stderr)
            return 2
        for name in args.name:
            arch.unstick(name, bump_memory=args.bump_memory,
                         bump_factor=args.bump_factor)
            print("unstuck", name)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_v = sub.add_parser("verify", help="cross-check index, status.json, "
                                          "and on-disk files")
    p_v.add_argument("--archive", required=True)
    p_v.add_argument("--json", action="store_true")
    p_v.set_defaults(fn=cmd_verify)

    p_r = sub.add_parser("rebuild-index",
                         help="reconstruct index.jsonl from per-sim status.json")
    p_r.add_argument("--archive", required=True)
    p_r.set_defaults(fn=cmd_rebuild_index)

    p_s = sub.add_parser("resummarize",
                         help="re-run the manifest's summarizer over every sim")
    p_s.add_argument("--archive", required=True)
    p_s.add_argument("--only-complete", action="store_true",
                     help="skip sims whose status isn't 'complete'")
    p_s.add_argument("--json", action="store_true")
    p_s.set_defaults(fn=cmd_resummarize)

    p_u = sub.add_parser("unstick", help="clear stuck status; "
                                           "--bump-memory to raise request_memory")
    p_u.add_argument("--archive", required=True)
    p_u.add_argument("--name", action="append",
                     help="sim_name to unstick (repeatable)")
    p_u.add_argument("--all", action="store_true",
                     help="unstick every 'stuck' sim")
    p_u.add_argument("--bump-memory", action="store_true",
                     help="multiply per-sim request_memory by --bump-factor")
    p_u.add_argument("--bump-factor", type=float, default=1.5)
    p_u.set_defaults(fn=cmd_unstick)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
