#!/usr/bin/env python3
"""Run a short two-intrinsic ILE-JAX AV smoke test with GPU precompute.

This intentionally uses a short BBH signal.  Timing begins at the Python worker
invocation, so queueing and container transfer are excluded.  The legacy LAL
waveform generator remains the default; GPU precompute uploads both conditioned
mode banks and reuses the detector state for the second intrinsic point.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

import numpy as np


def set_option(args, name, value=None):
    """Replace a long option in an optparse-style argument vector."""
    while name in args:
        i = args.index(name)
        del args[i:i + (1 if value is None else 2)]
    args.append(name)
    if value is not None:
        args.append(str(value))


def drop_option(args, name, takes_value=False):
    while name in args:
        i = args.index(name)
        del args[i:i + (2 if takes_value else 1)]


def load_generator(code_dir):
    path = code_dir / "demo/rift/slowrot_gpu_validate/make_e2e_inputs.py"
    spec = importlib.util.spec_from_file_location("short_jax_gpu_e2e_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The fixture predates the igwn_ligolw namespace used by current containers.
    def write_psd_xml(det, delta_f, fmax, path):
        import lal
        import lal.series
        import lalsimulation as lalsim
        from igwn_ligolw import utils as ligolw_utils
        count = int(fmax / delta_f) + 1
        series = lal.CreateREAL8FrequencySeries(
            det, lal.LIGOTimeGPS(0), 0.0, delta_f, lal.SecondUnit, count)
        frequency = np.arange(count) * delta_f
        values = np.array([
            lalsim.SimNoisePSDaLIGOZeroDetHighPower(max(float(f), 1.0))
            for f in frequency])
        values[~np.isfinite(values)] = 0.0
        series.data.data[:] = values
        document = lal.series.make_psd_xmldoc({det: series})
        ligolw_utils.write_filename(document, path, compress="gz")

    module.write_psd_xml = write_psd_xml
    return module


def result_rows(path):
    lines = [line for line in path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 1:
        raise RuntimeError("expected one result row in %s, found %d" %
                           (path, len(lines)))
    fields = lines[0].split()
    if len(fields) != 13:
        raise RuntimeError("expected 13 JAX result columns in %s, found %d" %
                           (path, len(fields)))
    values = np.asarray([float(value) for value in fields])
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non-finite JAX result in %s" % path)
    return values


def validate_fairdraw_bounds(path, boxes):
    """A finite evidence and ESS do not certify that AV respected its box."""
    header = path.read_text().splitlines()[0].lstrip("# ").split()
    values = np.atleast_2d(np.loadtxt(path))
    if values.shape[1] != len(header) or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid fairdraw columns or non-finite values: %s" % path)
    for column, key in (("right_ascension", "right_ascension"),
                        ("declination", "declination"),
                        ("inclination", "inclination"), ("psi", "psi"),
                        ("distance", "distance_mpc")):
        if column not in header:
            raise RuntimeError("missing fairdraw column %s" % column)
        x = values[:, header.index(column)]
        lo, hi = boxes[key]
        if np.any((x < lo) | (x > hi)):
            raise RuntimeError("fairdraw %s outside requested [%g, %g]: %s" %
                               (column, lo, hi, path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--inputs-only", action="store_true",
                        help="build the fixture and print the exact worker command")
    parser.add_argument("--n-eff", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=800_000)
    parser.add_argument("--n-chunk", type=int, default=8_000)
    parser.add_argument("--jax-av-eval-chunk", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=6907)
    parser.add_argument("--distance-mpc", type=float, default=400.0)
    parser.add_argument("--sky-half-width", type=float, default=0.1)
    parser.add_argument("--orientation-half-width", type=float, default=0.2)
    parser.add_argument("--distance-fraction-half-width", type=float, default=0.25)
    opts = parser.parse_args()

    if opts.n_eff <= 0 or opts.n_max <= 0 or opts.n_chunk <= 0:
        raise ValueError("n-eff, n-max, and n-chunk must be positive")
    if opts.jax_av_eval_chunk <= 0:
        raise ValueError("jax-av-eval-chunk must be positive")
    if opts.distance_mpc <= 0 or not (0 < opts.distance_fraction_half_width < 1):
        raise ValueError("distance and its fractional half-width must define a positive box")
    if opts.sky_half_width <= 0 or opts.orientation_half_width <= 0:
        raise ValueError("angular box half-widths must be positive")

    code_dir = Path(__file__).resolve().parents[2]
    made_temp = opts.work_dir is None
    work = (opts.work_dir.resolve() if opts.work_dir else
            Path(tempfile.mkdtemp(prefix="rift-gpu-jax-av-")))
    work.mkdir(parents=True, exist_ok=True)

    generator = load_generator(code_dir)
    generator.FMIN, generator.FMAX = 30.0, 512.0
    generator.SEGLEN, generator.SRATE = 8.0, 1024.0
    generator.EVENT_TIME = 1_000_000_000.25
    generator.M1, generator.M2, generator.DIST_MPC = 30.0, 25.0, opts.distance_mpc
    generator.QMAX = 1
    generator.main(str(work))

    import RIFT.lalsimutils as lsu
    case = json.loads((work / "case.json").read_text())
    grid = work / case["grid"]
    first = generator.base_params("H1")
    second = first.manual_copy()
    second.m1 *= 1.0002
    lsu.ChooseWaveformParams_array_to_xml([first, second], str(grid))
    if len(lsu.xml_to_ChooseWaveformParams_array(str(grid))) != 2:
        raise RuntimeError("two-point grid rewrite failed")

    args = list(case["ile_common"])
    # These classic accelerator selectors must not leak into the JAX smoke test:
    # the JAX backend and GPU-precompute route are selected explicitly below.
    for flag in ("--gpu", "--force-gpu-only", "--force-xpy",
                 "--force-adapt-all", "--internal-hard-fail-on-error",
                 "--internal-use-lnL"):
        drop_option(args, flag)
    drop_option(args, "--inv-spec-trunc-time", takes_value=True)
    for name, value in (("--mode", "laplace-is"),
                        ("--sampler-method", "AV"),
                        ("--n-eff", opts.n_eff), ("--n-max", opts.n_max),
                        ("--n-chunk", opts.n_chunk),
                        ("--jax-av-eval-chunk", opts.jax_av_eval_chunk),
                        ("--seed", opts.seed), ("--n-events-to-analyze", 2),
                        ("--event", 0), ("--output-file", "short_jax_gpu_av"),
                        ("--interp", "cubic"), ("--rotation-p-max", 1),
                        ("--freqresponse-qmax", 1),
                        ("--freqresponse-arm-length", "H1=40000,L1=40000")):
        set_option(args, name, value)
    for flag in ("--save-samples", "--fairdraw-extrinsic-output",
                 "--vectorized", "--rotation-slow", "--freqresponse"):
        set_option(args, flag)
    set_option(args, "--fairdraw-extrinsic-output-n-max", 200)
    # Match classic explicitly; this driver's default is 30 Hz, not 100 Hz.
    set_option(args, "--reference-freq", 100.0)

    boxes = {
        "right_ascension": [generator.RA - opts.sky_half_width,
                            generator.RA + opts.sky_half_width],
        "declination": [generator.DEC - opts.sky_half_width,
                        generator.DEC + opts.sky_half_width],
        "inclination": [generator.INCL - opts.orientation_half_width,
                        generator.INCL + opts.orientation_half_width],
        "psi": [generator.PSI - opts.orientation_half_width,
                generator.PSI + opts.orientation_half_width],
        "distance_mpc": [opts.distance_mpc * (1 - opts.distance_fraction_half_width),
                         opts.distance_mpc * (1 + opts.distance_fraction_half_width)],
    }
    for flag, key in (("--limit-right-ascension", "right_ascension"),
                      ("--limit-declination", "declination"),
                      ("--limit-inclination", "inclination"),
                      ("--limit-psi", "psi")):
        set_option(args, flag, "%.9g,%.9g" % tuple(boxes[key]))
    set_option(args, "--d-min", boxes["distance_mpc"][0])
    set_option(args, "--d-max", boxes["distance_mpc"][1])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("RIFT_HYPERPIPELINE_FORMAT", None)
    env["RIFT_GPU_PRECOMPUTE"] = "1"
    # This checkpoint deliberately exercises the production-compatible route:
    # existing conditioned LAL modes followed by one host-to-device transfer.
    env["RIFT_GPU_WAVEFORM"] = "lal"
    # Must be present before the worker imports JAX.  Setting this in the
    # handoff itself is too late because the driver's cache probe initializes
    # the JAX backend before waveform/precompute construction.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["PYTHONPATH"] = str(code_dir) + os.pathsep + env.get("PYTHONPATH", "")
    command = [env.get("PYTHON", "python"),
               str(code_dir / "bin/integrate_likelihood_extrinsic_jax")] + args

    if opts.inputs_only:
        print(json.dumps(dict(
            result="PASS", scope="inputs-only", intrinsic_points=2,
            work_dir=str(work), command=shlex.join(command),
            environment={"RIFT_GPU_PRECOMPUTE": "1", "RIFT_GPU_WAVEFORM": "lal",
                         "XLA_PYTHON_CLIENT_PREALLOCATE":
                             env["XLA_PYTHON_CLIENT_PREALLOCATE"]},
            sampler="AV", mode="laplace-is", internal_log_weights=True,
            fairdraw_max=200,
            compatibility_limits=[
                "short boxed BBH execution smoke, not posterior recovery",
                "legacy conditioned LAL waveform modes plus H2D; native JAX waveform disabled",
                "JAX AV keeps importance weights in log space; classic --internal-use-lnL is not used",
                "sample products are equal-weight *_samples.dat files, not classic ILE XML",
            ]), indent=2, sort_keys=True))
        return

    log_path = work / "short_jax_gpu_av.log"
    started = time.perf_counter()
    with log_path.open("w") as log:
        completed = subprocess.run(command, cwd=work, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    host_runtime = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError("ILE-JAX exited %d; inspect %s" %
                           (completed.returncode, log_path))

    rows = []
    sample_rows = {}
    for index in range(2):
        dat = work / ("short_jax_gpu_av_%d_.dat" % index)
        samples = work / ("short_jax_gpu_av_%d_samples.dat" % index)
        if not dat.is_file() or not samples.is_file():
            raise RuntimeError("missing event %d result/fairdraw; inspect %s" %
                               (index, log_path))
        values = result_rows(dat)
        lnl, sigma, ntotal, neff = values[-4:]
        if neff < opts.n_eff:
            raise RuntimeError("INCONCLUSIVE: event %d stopped at neff %.3f below target %d" %
                               (index, neff, opts.n_eff))
        if ntotal > opts.n_max:
            raise RuntimeError("event %d exceeded n-max: %.0f > %d" %
                               (index, ntotal, opts.n_max))
        lines = samples.read_text().splitlines()
        payload = [line for line in lines
                   if line.strip() and not line.lstrip().startswith("#")]
        if not payload or len(payload) > 200:
            raise RuntimeError("event %d fairdraw row count %d is outside [1, 200]" %
                               (index, len(payload)))
        if not any("fairdraw:" in line for line in lines if line.startswith("#")):
            raise RuntimeError("event %d fairdraw lacks provenance header" % index)
        validate_fairdraw_bounds(samples, boxes)
        sample_rows[samples.name] = len(payload)
        rows.append(dict(index=index, path=str(dat), samples=str(samples),
                         lnL=float(lnl), sigma_lnL=float(sigma),
                         ntotal=int(ntotal), neff=float(neff)))

    print(json.dumps(dict(
        result="PASS", host_runtime_s=host_runtime,
        timing_scope="inside worker; no queue/container transfer",
        sampler="AV", mode="laplace-is", internal_log_weights=True,
        n_eff_target=opts.n_eff, n_max=opts.n_max, n_chunk=opts.n_chunk,
        jax_av_eval_chunk=opts.jax_av_eval_chunk, fairdraw_max=200,
        intrinsic_points=2, prior_boxes=boxes, rows=rows,
        sample_rows=sample_rows, work_dir=str(work), log=str(log_path),
        validation_scope="boxed legacy-waveform H2D plus ILE-JAX AV execution; not posterior recovery",
    ), indent=2, sort_keys=True))
    if made_temp and not opts.keep:
        print("Temporary products retained for inspection at %s" % work)


if __name__ == "__main__":
    main()
