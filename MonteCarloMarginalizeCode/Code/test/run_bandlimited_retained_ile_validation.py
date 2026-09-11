#!/usr/bin/env python3
"""Run one accepted-reference ILE argv against a frozen RIFT commit."""

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _git(tree, *args):
    return subprocess.check_output(
        ["git", "-C", str(tree), *args], universal_newlines=True).strip()


def _option(argv, name):
    return argv[argv.index(name) + 1]


def _set_option(argv, name, value):
    where = argv.index(name) + 1
    argv[where] = str(value)


def _elapsed_seconds(resource_text):
    match = re.search(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)$", resource_text, re.M)
    if not match:
        return None
    fields = [float(item) for item in match.group(1).split(":")]
    if len(fields) == 2:
        return 60 * fields[0] + fields[1]
    if len(fields) == 3:
        return 3600 * fields[0] + 60 * fields[1] + fields[2]
    return None


def _resource_value(pattern, resource_text, cast=int):
    match = re.search(pattern, resource_text, re.M)
    return cast(match.group(1)) if match else None


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-record", required=True, type=Path)
    parser.add_argument("--rift-tree", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--container", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--cpuset", default="0-7")
    parser.add_argument("--control", choices=("retained", "full"), default="retained")
    args = parser.parse_args()

    baseline = json.loads(args.baseline_record.read_text())
    if not baseline.get("accepted", False):
        raise SystemExit("baseline record is not accepted")
    argv = list(baseline["argv"])
    required = {
        "--sampler-method": "AV",
        "--time-marginalization-quadrature": "bandlimited",
        "--interpolate-time": "sinc",
        "--n-max": "4000000",
        "--n-eff": "100",
        "--n-chunk": "40000",
    }
    mismatch = {}
    for key, expected in required.items():
        observed = _option(argv, key) if key in argv else None
        if observed != expected:
            mismatch[key] = (observed, expected)
    if mismatch:
        raise SystemExit("baseline argv does not meet production contract: %r" % mismatch)

    commit = _git(args.rift_tree, "rev-parse", "HEAD")
    if commit != args.expected_commit:
        raise SystemExit("RIFT commit mismatch: %s != %s" % (commit, args.expected_commit))
    dirty = _git(args.rift_tree, "status", "--porcelain")
    if dirty:
        raise SystemExit("RIFT tree is dirty:\n" + dirty)
    if not args.container.is_file():
        raise SystemExit("missing container: %s" % args.container)
    observed_input_hashes = {}
    input_hash_mismatches = {}
    for path, expected in baseline.get("input_sha256", {}).items():
        observed = _sha256(path)
        observed_input_hashes[path] = observed
        if observed != expected:
            input_hash_mismatches[path] = {"expected": expected, "observed": observed}
    if input_hash_mismatches:
        raise SystemExit("baseline inputs changed: %s" % json.dumps(
            input_hash_mismatches, sort_keys=True))
    container_sha256 = observed_input_hashes.get(str(args.container))
    expected_container_sha256 = baseline.get("input_sha256", {}).get(str(args.container))
    if expected_container_sha256 != container_sha256:
        raise SystemExit("container does not match the accepted baseline record")

    cell = "%s_bandlimited_snr%s_seed%s_%s" % (
        baseline["model"], baseline["snr_label"], baseline["seed"], args.control)
    out = args.output_root / cell
    if out.exists():
        raise SystemExit("refusing to overwrite validation directory: %s" % out)
    out.mkdir(parents=True)
    output_prefix = out / "output"
    _set_option(argv, "--output-file", output_prefix)

    code = args.rift_tree / "MonteCarloMarginalizeCode" / "Code"
    ile = code / "bin" / "integrate_likelihood_extrinsic_batchmode"
    wrapper = Path(__file__).with_name("telemetry_bandlimited_retained_ile.py")
    telemetry = out / "fft_telemetry.json"
    env_opts = {
        "PYTHONPATH": str(code),
        "PATH": str(code / "bin") + ":/usr/local/bin:/usr/bin:/bin",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "RIFT_REAL_ILE": str(ile),
        "RIFT_FFT_TELEMETRY_FILE": str(telemetry),
        "RIFT_VALIDATION_FORCE_FULL_FFT": "1" if args.control == "full" else "0",
    }
    launch = ["apptainer", "exec", "--nv"]
    for key, value in env_opts.items():
        launch.extend(("--env", key + "=" + value))
    launch.extend((str(args.container), "python3", "-u", str(wrapper)))
    timed = ["/usr/bin/time", "-v", "-o", str(out / "resource.txt"),
             "taskset", "-c", args.cpuset] + launch + argv

    start = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    (out / "start_utc.txt").write_text(start + "\n")
    (out / "rift_commit.txt").write_text(commit + "\n")
    (out / "baseline_record.txt").write_text(str(args.baseline_record.resolve()) + "\n")
    (out / "argv.nul").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
    (out / "command.txt").write_text(
        " ".join(shlex.quote(item) for item in timed) + "\n")
    provenance = {
        "schema": 1,
        "baseline_record": str(args.baseline_record.resolve()),
        "baseline_record_sha256": _sha256(args.baseline_record),
        "baseline_status_sha256": baseline.get("status_sha256"),
        "input_sha256": observed_input_hashes,
        "rift_commit": commit,
        "container": str(args.container),
        "container_sha256": container_sha256,
        "control": args.control,
        "physical_gpu": args.gpu,
        "cpuset": args.cpuset,
        "argv_matches_baseline_except_output": True,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    # One persistent nvidia-smi matches the accepted campaign monitor and avoids
    # racing Apptainer's Go runtime with a new helper process every 200 ms on
    # login nodes with a tight per-user thread limit.
    monitor_path = out / "gpu_usage.csv"
    monitor_cmd = [
        "nvidia-smi",
        "-i",
        str(args.gpu),
        "--query-gpu=timestamp,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
        "--loop-ms=200",
        "--filename=" + str(monitor_path),
    ]
    monitor_process = subprocess.Popen(
        monitor_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    launch_env = os.environ.copy()
    launch_env["GOMAXPROCS"] = "4"
    time.sleep(0.3)
    try:
        with (out / "run.log").open("w") as log:
            process = subprocess.Popen(
                timed, stdout=log, stderr=subprocess.STDOUT, env=launch_env)
            rc = process.wait()
    finally:
        monitor_process.terminate()
        try:
            monitor_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor_process.kill()
            monitor_process.wait()

    monitor = []
    if monitor_path.exists():
        for line in monitor_path.read_text(errors="replace").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                monitor.append((fields[0], float(fields[1]), float(fields[2])))
            except ValueError:
                pass
    (out / "exit_code.txt").write_text(str(rc) + "\n")
    final_commit = _git(args.rift_tree, "rev-parse", "HEAD")
    final_dirty = _git(args.rift_tree, "status", "--porcelain")
    (out / "rift_commit_final.txt").write_text(final_commit + "\n")

    status_path = out / "output_0_integrator_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    resource_text = (out / "resource.txt").read_text(errors="replace")
    log_text = (out / "run.log").read_text(errors="replace")
    if "CuPy Platform" in log_text and "NVIDIA CUDA" in log_text:
        backend = "cuda"
    elif "no cupy" in log_text.lower():
        backend = "numpy-cpu"
    else:
        backend = "unknown"
    n_ess = status.get("n_ESS")
    khat = status.get("pareto_khat")
    run_rejection = []
    if rc:
        run_rejection.append("nonzero exit")
    if final_commit != commit:
        run_rejection.append("RIFT source commit changed during run")
    if final_dirty:
        run_rejection.append("RIFT source tree became dirty during run")
    if status.get("collapsed", False):
        run_rejection.append("AV live-volume collapse")
    if n_ess is None or not math.isfinite(float(n_ess)) or float(n_ess) < 100:
        run_rejection.append("n_ESS below 100")
    if khat is None or not math.isfinite(float(khat)) or float(khat) >= 0.7:
        run_rejection.append("Pareto k_hat not below 0.7")
    if backend != "cuda":
        run_rejection.append("requested GPU backend not verified")
    if not monitor:
        run_rejection.append("GPU memory monitor produced no samples")

    optimization_rejection = []
    if not telemetry.exists():
        optimization_rejection.append("FFT telemetry missing")
        fft_telemetry = None
    else:
        fft_telemetry = json.loads(telemetry.read_text())
        if fft_telemetry.get("failed_calls"):
            optimization_rejection.append("band-limited marginalization call failed")
        if fft_telemetry.get("full_fft_fallback_rows"):
            optimization_rejection.append("retained FFT fell back to full padding")
        if args.control == "retained":
            if not fft_telemetry.get("retained_fft_rows"):
                optimization_rejection.append("retained control used no retained FFT rows")
            if fft_telemetry.get("full_fft_selected_rows"):
                optimization_rejection.append("retained GPU control selected full padding")
        else:
            if not fft_telemetry.get("full_fft_selected_rows"):
                optimization_rejection.append("full control used no full-padding rows")
            if fft_telemetry.get("retained_fft_rows"):
                optimization_rejection.append("full control used retained FFT rows")

    memories = [item[1] for item in monitor]
    utilizations = [item[2] for item in monitor]
    result = {
        "schema": 1,
        "accepted": not run_rejection and not optimization_rejection,
        "sampler_accepted": not run_rejection,
        "sampler_rejection_reasons": run_rejection,
        "optimization_validated": not optimization_rejection,
        "optimization_rejection_reasons": optimization_rejection,
        "model": baseline["model"],
        "snr_label": baseline["snr_label"],
        "seed": baseline["seed"],
        "control": args.control,
        "exit_code": rc,
        "backend_actual": backend,
        "rift_commit": commit,
        "rift_commit_final": final_commit,
        "rift_dirty_final": bool(final_dirty),
        "lnZ": status.get("lnL"),
        "sigma_lnZ": status.get("sigma_lnL"),
        "n_ESS": n_ess,
        "pareto_khat": khat,
        "ntotal": status.get("ntotal"),
        "collapsed": status.get("collapsed", False),
        "wall_seconds": _elapsed_seconds(resource_text),
        "max_rss_kib": _resource_value(r"Maximum resident set size \(kbytes\):\s*(\d+)", resource_text),
        "gpu_peak_mib": max(memories) if memories else None,
        "gpu_utilization_median": statistics.median(utilizations) if utilizations else None,
        "gpu_monitor_samples": len(monitor),
        "fft_telemetry": fft_telemetry,
        "baseline": {
            key: baseline.get(key) for key in (
                "rift_commit", "lnZ", "sigma_lnZ", "n_ESS", "pareto_khat",
                "ntotal", "wall_seconds", "max_rss_kib", "gpu_peak_mib")
        },
    }
    if result["lnZ"] is not None and baseline.get("lnZ") is not None:
        result["delta_lnZ_vs_baseline"] = result["lnZ"] - baseline["lnZ"]
        combined = math.hypot(result["sigma_lnZ"], baseline["sigma_lnZ"])
        result["delta_lnZ_over_combined_sigma"] = result["delta_lnZ_vs_baseline"] / combined
    (out / "validation_record.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["accepted"]:
        (out / "DONE").touch()
    else:
        reasons = (["sampler: " + reason for reason in run_rejection] +
                   ["optimization: " + reason for reason in optimization_rejection])
        (out / "REJECTED").write_text("\n".join(reasons) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 20


if __name__ == "__main__":
    sys.exit(main())
