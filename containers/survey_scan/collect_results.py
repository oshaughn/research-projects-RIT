#!/usr/bin/env python3
"""Collect completed survey_scan warmup JSON outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import read_json, write_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", required=True, help="Survey directory.")
    ap.add_argument("--out", default=None, help="Summary JSON path.")
    args = ap.parse_args(argv)

    survey = Path(args.survey)
    jobs = survey / "jobs"
    result_files = sorted(jobs.glob("*.json"))
    results = []
    for path in result_files:
        try:
            item = read_json(path)
            item["_path"] = str(path)
            results.append(item)
        except Exception as exc:  # noqa: BLE001
            results.append({"_path": str(path), "error": str(exc)})

    summary = {
        "survey": str(survey),
        "n_results": len(results),
        "results": results,
    }
    out = Path(args.out) if args.out else survey / "warmup_summary.json"
    write_json(out, summary)

    md = out.with_suffix(".md")
    with md.open("w", encoding="utf-8") as f:
        f.write("# Warmup Summary\n\n")
        f.write("| profile | status | device | elapsed s | cache bytes | path |\n")
        f.write("|---|---|---|---:|---:|---|\n")
        for item in results:
            profile = item.get("profile", "?")
            status = "PASS" if item.get("ok") else "FAIL"
            device = item.get("device", {}).get("name", "?") if isinstance(item.get("device"), dict) else "?"
            elapsed = item.get("elapsed_s", "")
            cache = item.get("cache", {}).get("bytes", "") if isinstance(item.get("cache"), dict) else ""
            f.write(f"| {profile} | {status} | {device} | {elapsed} | {cache} | {item.get('_path')} |\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

