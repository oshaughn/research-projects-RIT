#!/usr/bin/env python3
"""Emit Condor jobs that run RIFT container warmup profiles."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path

from common import manifest_entries, rel_or_abs, repo_root_from_here, safe_name


PROFILE_MAP = {
    "cupy": "rift_cupy_common.py",
    "jax": "rift_jax_ile_common.py",
}


def _constraint(min_cap: float | None, max_cap: float | None) -> str:
    parts = ["(Capability =!= undefined)"]
    if min_cap is not None:
        parts.append(f"(Capability >= {min_cap})")
    if max_cap is not None:
        parts.append(f"(Capability <= {max_cap})")
    return " && ".join(parts)


def _write_runner(path: Path, image: str, profile: str, result: str) -> None:
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
log() {{ echo "[survey_scan] $*" >&2; }}

image={image!r}
profile={profile!r}
result={result!r}
sif="$image"

if [[ "$image" == osdf://* ]]; then
    base="$(basename "$image")"
    if [ -e "$base" ]; then
        sif="./$base"
    else
        log "fetching $image"
        if command -v stashcp >/dev/null 2>&1; then
            stashcp "$image" "$base"
        elif command -v pelican >/dev/null 2>&1; then
            pelican object get "$image" "$base"
        else
            log "FATAL: no stashcp or pelican available for $image"
            exit 4
        fi
        sif="./$base"
    fi
fi

cache_root="${{RIFT_SURVEY_CACHE_ROOT:-${{_CONDOR_SCRATCH_DIR:-$PWD}}/.rift_cache}}"
mkdir -p "$cache_root"
export CUPY_CACHE_DIR="${{CUPY_CACHE_DIR:-$cache_root/cupy}}"
export CUPY_CACHE_IN_MEMORY="${{CUPY_CACHE_IN_MEMORY:-0}}"
export JAX_COMPILATION_CACHE_DIR="${{JAX_COMPILATION_CACHE_DIR:-$cache_root/jax}}"
export JAX_ENABLE_X64="${{JAX_ENABLE_X64:-1}}"
export XLA_FLAGS="${{XLA_FLAGS:---xla_cpu_multi_thread_eigen=false}}"
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"

log "image=$sif"
log "profile=$profile"
log "result=$result"
apptainer exec --nv "$sif" python3 "$profile" --json-out "$result"
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_submit(
    path: Path,
    runner: Path,
    profile_path: Path,
    result_name: str,
    min_cap: float | None,
    max_cap: float | None,
    request_disk: str,
) -> None:
    path.write_text(
        f"""universe = vanilla
executable = {runner.name}
arguments =
request_GPUs = 1
request_disk = {request_disk}
require_gpus = {_constraint(min_cap, max_cap)}
transfer_input_files = {profile_path}
transfer_output_files = {result_name}
output = $(Cluster).$(Process).out
error = $(Cluster).$(Process).err
log = $(Cluster).log
queue 1
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", required=True, help="Survey directory.")
    ap.add_argument("--manifest", required=True, help="Container-family manifest.")
    ap.add_argument("--out", default=None, help="Output jobs directory.")
    ap.add_argument(
        "--profiles",
        default="cupy",
        help="Comma-separated profiles: cupy,jax. Default: cupy.",
    )
    ap.add_argument("--request-disk", default="16000M")
    args = ap.parse_args(argv)

    survey = Path(args.survey)
    out = Path(args.out) if args.out else survey / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    profile_dir = repo_root_from_here() / "containers" / "survey_scan" / "profiles"
    selected_profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]

    manifest = Path(args.manifest)
    entries = manifest_entries(manifest)
    if not entries:
        raise SystemExit(f"No container entries found in {manifest}")

    generated = []
    for entry in entries:
        for profile_key in selected_profiles:
            profile_name = PROFILE_MAP.get(profile_key, profile_key)
            profile_path = profile_dir / profile_name
            if not profile_path.exists():
                raise SystemExit(f"Profile not found: {profile_path}")
            stem = safe_name(f"{entry.label}_{Path(profile_name).stem}")
            runner = out / f"run_{stem}.sh"
            result = f"{stem}.json"
            submit = out / f"{stem}.sub"
            _write_runner(runner, entry.image, profile_path.name, result)
            _write_submit(
                submit,
                runner,
                Path(profile_path.name),
                result,
                entry.cuda_capability_min,
                entry.cuda_capability_max,
                args.request_disk,
            )
            generated.append(submit)

    # Copy profile scripts next to the jobs so condor_submit can run from out/.
    for profile_key in selected_profiles:
        profile_name = PROFILE_MAP.get(profile_key, profile_key)
        src = profile_dir / profile_name
        dst = out / profile_name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    with (out / "submit_all.sh").open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write("cd \"$(dirname \"$0\")\"\n")
        for sub in generated:
            f.write(f"condor_submit {sub.name}\n")
    (out / "submit_all.sh").chmod(0o755)

    print(rel_or_abs(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
