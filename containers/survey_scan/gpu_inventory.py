#!/usr/bin/env python3
"""Survey Condor GPU inventory for RIFT container-family planning."""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path

from common import manifest_entries, write_json


FIELDS = [
    "Name",
    "GPUs_DeviceName",
    "GPUs_Capability",
    "GPUs_GlobalMemoryMb",
    "CUDACapability",
    "CUDADeviceName",
    "CUDADeviceGlobalMemoryMb",
]


def _run_condor_status(constraint: str) -> list[dict[str, str]]:
    if shutil.which("condor_status") is None:
        raise SystemExit("condor_status not found on PATH")
    cmd = ["condor_status", "-constraint", constraint, "-af", *FIELDS]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "condor_status failed")
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < len(FIELDS):
            parts = parts + ["undefined"] * (len(FIELDS) - len(parts))
        rows.append(dict(zip(FIELDS, parts[: len(FIELDS)])))
    return rows


def _norm(row: dict[str, str]) -> tuple[str, str, str]:
    name = row.get("GPUs_DeviceName") or row.get("CUDADeviceName") or "undefined"
    cap = row.get("GPUs_Capability") or row.get("CUDACapability") or "undefined"
    mem = row.get("GPUs_GlobalMemoryMb") or row.get("CUDADeviceGlobalMemoryMb") or "undefined"
    return name, cap, mem


def _recommend_bands(summary: Counter[tuple[str, str, str]]) -> list[dict[str, object]]:
    caps = []
    for (_name, cap, _mem), count in summary.items():
        try:
            caps.append((float(cap), count))
        except ValueError:
            continue
    if not caps:
        return []
    min_cap = min(c for c, _ in caps)
    max_cap = max(c for c, _ in caps)
    bands = []
    if min_cap < 9.0:
        bands.append(
            {
                "label": "cc60-90" if min_cap >= 6.0 else "default",
                "cuda_capability_min": max(3.5, min_cap),
                "cuda_capability_max": 9.0,
                "reason": "Observed pre-Blackwell CUDA 11-compatible GPUs.",
            }
        )
    if max_cap >= 9.0:
        bands.append(
            {
                "label": "cc90-120",
                "cuda_capability_min": 9.0,
                "cuda_capability_max": max(12.0, max_cap),
                "reason": "Observed Hopper/Blackwell-class GPUs; use CUDA 12 devel when NVRTC headers are needed.",
            }
        )
    return bands


def _coverage(summary: Counter[tuple[str, str, str]], manifest: Path | None) -> list[dict[str, object]]:
    if manifest is None:
        return []
    entries = manifest_entries(manifest)
    rows = []
    for (device, cap, mem), count in summary.most_common():
        matches = []
        try:
            cap_f = float(cap)
        except ValueError:
            cap_f = None
        if cap_f is not None:
            for entry in entries:
                lo = entry.cuda_capability_min
                hi = entry.cuda_capability_max
                if lo is not None and cap_f < lo:
                    continue
                if hi is not None and cap_f > hi:
                    continue
                matches.append(entry.label)
        rows.append(
            {
                "device": device,
                "capability": cap,
                "memory_mb": mem,
                "slots": count,
                "manifest_matches": matches,
                "covered": bool(matches),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="Output directory.")
    ap.add_argument(
        "--constraint",
        default="TotalGPUs > 0",
        help="condor_status constraint for GPU inventory.",
    )
    ap.add_argument("--manifest", default=None, help="Optional container-family manifest to check coverage.")
    args = ap.parse_args(argv)

    stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or f"survey/{socket.gethostname()}-{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    rows = _run_condor_status(args.constraint)
    summary = Counter(_norm(row) for row in rows)

    manifest_path = Path(args.manifest) if args.manifest else None
    coverage = _coverage(summary, manifest_path)

    write_json(out / "gpu_inventory.json", {
        "created_utc": stamp,
        "host": socket.gethostname(),
        "constraint": args.constraint,
        "manifest": str(manifest_path) if manifest_path else None,
        "fields": FIELDS,
        "rows": rows,
        "summary": [
            {"device": k[0], "capability": k[1], "memory_mb": k[2], "slots": v}
            for k, v in summary.most_common()
        ],
        "coverage": coverage,
    })
    write_json(out / "recommended_matrix.json", {
        "created_utc": stamp,
        "bands": _recommend_bands(summary),
    })

    with (out / "gpu_inventory.tsv").open("w", encoding="utf-8") as f:
        f.write("slots\tdevice\tcapability\tmemory_mb\n")
        for (device, cap, mem), count in summary.most_common():
            f.write(f"{count}\t{device}\t{cap}\t{mem}\n")

    with (out / "coverage.md").open("w", encoding="utf-8") as f:
        f.write("# GPU Survey\n\n")
        f.write(f"- Created UTC: `{stamp}`\n")
        f.write(f"- Host: `{socket.gethostname()}`\n")
        f.write(f"- Constraint: `{args.constraint}`\n\n")
        f.write("| slots | device | capability | memory MB |\n")
        f.write("|---:|---|---:|---:|\n")
        for (device, cap, mem), count in summary.most_common():
            f.write(f"| {count} | {device} | {cap} | {mem} |\n")
        f.write("\n## Suggested Bands\n\n")
        for band in _recommend_bands(summary):
            f.write(
                f"- `{band['label']}`: cc {band['cuda_capability_min']} - "
                f"{band['cuda_capability_max']} ({band['reason']})\n"
            )
        if coverage:
            f.write("\n## Manifest Coverage\n\n")
            f.write("| slots | device | capability | matching labels |\n")
            f.write("|---:|---|---:|---|\n")
            for row in coverage:
                labels = ", ".join(row["manifest_matches"]) or "UNCOVERED"
                f.write(
                    f"| {row['slots']} | {row['device']} | "
                    f"{row['capability']} | {labels} |\n"
                )

    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
