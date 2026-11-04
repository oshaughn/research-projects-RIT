#!/usr/bin/env python3
"""Run a short, self-contained combined-response ILE with AV and GPU precompute.

The clock starts inside the worker process invocation.  Queue and container transfer are
outside this runner.  Two nearby intrinsic points are evaluated sequentially so the shared
GPUPrecomputeContext must reuse detector data and PSD uploads.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def set_option(args, name, value=None):
    while name in args:
        i = args.index(name)
        del args[i:i + (1 if value is None else 2)]
    args.append(name)
    if value is not None:
        args.append(str(value))


def load_generator(code_dir):
    path = code_dir / "demo/rift/slowrot_gpu_validate/make_e2e_inputs.py"
    spec = importlib.util.spec_from_file_location("short_gpu_e2e_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The older validation helper imports the retired `ligo.lw` namespace.  Keep the
    # fixture logic but supply the current production igwn_ligolw writer here.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--inputs-only", action="store_true",
                        help="build and validate the two-point fixture without launching ILE")
    parser.add_argument("--n-eff", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=800_000)
    parser.add_argument("--n-chunk", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=6907)
    parser.add_argument("--distance-mpc", type=float, default=400.0,
                        help="injected distance and center of the boxed distance prior")
    parser.add_argument("--sky-half-width", type=float, default=0.1,
                        help="RA and declination half-width in radians")
    parser.add_argument("--orientation-half-width", type=float, default=0.2,
                        help="inclination and polarization half-width in radians")
    parser.add_argument("--distance-fraction-half-width", type=float, default=0.25,
                        help="fractional half-width of the distance prior around truth")
    opts = parser.parse_args()
    code_dir = Path(__file__).resolve().parents[2]
    made_temp = opts.work_dir is None
    work = opts.work_dir.resolve() if opts.work_dir else Path(tempfile.mkdtemp(prefix="rift-gpu-av-"))
    work.mkdir(parents=True, exist_ok=True)

    # Reuse the maintained self-contained fixture builder at a genuinely short BBH grid.
    generator = load_generator(code_dir)
    generator.FMIN, generator.FMAX = 30.0, 512.0
    # The maintained fixture builder reserves two seconds at each frame edge.
    generator.SEGLEN, generator.SRATE = 8.0, 1024.0
    # Avoid landing exactly on the helper's integer, strictly-open data boundary.
    generator.EVENT_TIME = 1_000_000_000.25
    generator.M1, generator.M2, generator.DIST_MPC = 30.0, 25.0, opts.distance_mpc
    generator.QMAX = 1
    generator.main(str(work))

    import RIFT.lalsimutils as lsu
    case = json.loads((work / "case.json").read_text())
    grid = work / case["grid"]
    # The fixture's grid is deliberately arbitrary.  Replace it with the injected
    # intrinsic point and one nearby point so this is an execution/parity integration.
    first = generator.base_params("H1")
    second = first.manual_copy()
    second.m1 *= 1.0002
    lsu.ChooseWaveformParams_array_to_xml([first, second], str(grid))
    if opts.inputs_only:
        check = lsu.xml_to_ChooseWaveformParams_array(str(grid))
        if len(check) != 2:
            raise RuntimeError("two-point grid rewrite failed")
        print(json.dumps(dict(result="PASS", scope="inputs-only", intrinsic_points=2,
                              work_dir=str(work), case=case), indent=2, sort_keys=True))
        return

    args = list(case["ile_common"])
    for name, value in (("--n-eff", opts.n_eff), ("--n-max", opts.n_max),
                        ("--n-chunk", opts.n_chunk), ("--seed", opts.seed),
                        ("--n-events-to-analyze", 2), ("--event", 0),
                        ("--output-file", "short_gpu_av")):
        set_option(args, name, value)
    # AV/log weights and a bounded fairdraw are load-bearing test contracts.
    set_option(args, "--sampler-method", "AV")
    for flag in ("--internal-use-lnL", "--fairdraw-extrinsic-output", "--save-samples",
                 "--vectorized", "--gpu", "--force-gpu-only", "--force-xpy", "--force-adapt-all",
                 "--rotation-slow", "--freqresponse", "--internal-hard-fail-on-error"):
        set_option(args, flag)
    set_option(args, "--fairdraw-extrinsic-output-n-max", 200)
    set_option(args, "--inv-spec-trunc-time", 0)
    set_option(args, "--interpolate-time", "cubic")
    # Driver defaults differ (classic 100 Hz, JAX 30 Hz); pin the convention.
    set_option(args, "--reference-freq", 100.0)
    set_option(args, "--rotation-p-max", 1)
    set_option(args, "--freqresponse-qmax", 1)
    set_option(args, "--freqresponse-arm-length", "H1=40000,L1=40000")

    # This is deliberately a boxed execution/integration smoke test.  A broad
    # all-sky prior at high SNR tests AV convergence rather than GPU precompute
    # integration and can exhaust n-max with a one-point live volume.
    if opts.distance_mpc <= 0 or not (0 < opts.distance_fraction_half_width < 1):
        raise ValueError("distance and fractional distance half-width must define a positive box")
    if opts.sky_half_width <= 0 or opts.orientation_half_width <= 0:
        raise ValueError("angular box half-widths must be positive")
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
    # Keep the result schema deterministic even if the submission environment is
    # also used by a hyperpipeline job.  The status sidecar below is canonical,
    # but checking the legacy row as well catches a partial/truncated export.
    env.pop("RIFT_HYPERPIPELINE_FORMAT", None)
    env["RIFT_GPU_PRECOMPUTE"] = "1"
    env["PYTHONPATH"] = str(code_dir) + os.pathsep + env.get("PYTHONPATH", "")
    command = [env.get("PYTHON", "python"),
               str(code_dir / "bin/integrate_likelihood_extrinsic_batchmode")] + args
    log_path = work / "short_gpu_av.log"
    started = time.perf_counter()
    with log_path.open("w") as log:
        completed = subprocess.run(command, cwd=work, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    host_runtime = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError("ILE exited %d; inspect %s" % (completed.returncode, log_path))

    # Batchmode writes one .dat, status sidecar, and XML per intrinsic index.
    # Name them exactly: a broad glob can silently accept stale products or future
    # sibling exports (for example calibration output) in a reused work directory.
    rows = []
    for index in range(2):
        dat = work / ("short_gpu_av_%d_.dat" % index)
        status_path = work / ("short_gpu_av_%d_integrator_status.json" % index)
        if not dat.is_file() or not status_path.is_file():
            raise RuntimeError(
                "missing event %d result/status sidecar; inspect %s" % (index, log_path))
        data_lines = [line for line in dat.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
        if len(data_lines) != 1:
            raise RuntimeError("expected one result row in %s, found %d" %
                               (dat, len(data_lines)))
        fields = data_lines[0].split()
        if len(fields) < 4:
            raise RuntimeError("truncated ILE result in %s" % dat)
        dat_lnl, dat_sigma, dat_ntotal, dat_neff = map(float, fields[-4:])
        status = json.loads(status_path.read_text())
        try:
            lnl = float(status["lnL"])
            sigma = float(status["sigma_lnL"])
            ntotal = int(status["ntotal"])
            neff = float(status["neff"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("invalid status sidecar %s: %s" % (status_path, exc))
        if status.get("indx_event") != index or status.get("sampler_method") != "AV":
            raise RuntimeError("wrong event/sampler recorded in %s" % status_path)
        if status.get("collapsed") is not False:
            raise RuntimeError("AV integration collapsed according to %s" % status_path)
        if not np.all(np.isfinite([lnl, sigma, ntotal, neff])):
            raise RuntimeError("non-finite ILE result in %s" % status_path)
        if neff < opts.n_eff:
            raise RuntimeError(
                "INCONCLUSIVE: event %d stopped at neff %.3f below target %d" %
                (index, neff, opts.n_eff))
        # With hyperpipeline format explicitly disabled, the final four legacy
        # columns must agree with the machine-readable sidecar.
        if not np.allclose([dat_lnl, dat_sigma, dat_ntotal, dat_neff],
                           [lnl, sigma, ntotal, neff], rtol=2e-6, atol=1e-8):
            raise RuntimeError(".dat/status disagreement for event %d" % index)
        rows.append(dict(index=index, path=str(dat), status_path=str(status_path),
                         lnL=lnl, sigma_lnL=sigma, ntotal=ntotal, neff=neff))
    xmls = [work / ("short_gpu_av_%d_.xml.gz" % index) for index in range(2)]
    missing_xmls = [str(path) for path in xmls if not path.is_file()]
    if missing_xmls:
        raise RuntimeError("save-samples missing XML(s) %s; inspect %s" %
                           (missing_xmls, log_path))
    xml_rows = {}
    for xml in xmls:
        saved = lsu.xml_to_ChooseWaveformParams_array(str(xml))
        xml_rows[xml.name] = len(saved)
        if len(saved) > 200:
            raise RuntimeError("fairdraw cap violated: %s has %d rows" % (xml, len(saved)))
        if xml.stat().st_size > 10 * 1024 * 1024:
            raise RuntimeError("bounded fairdraw XML unexpectedly exceeds 10 MiB: %s" % xml)
    print(json.dumps(dict(
        result="PASS", host_runtime_s=host_runtime, timing_scope="inside worker; no queue/container transfer",
        sampler="AV", internal_log_weights=True, n_eff_target=opts.n_eff,
        n_max=opts.n_max, n_chunk=opts.n_chunk, fairdraw_max=200,
        injected_distance_mpc=opts.distance_mpc, prior_boxes=boxes,
        time_interpolation="cubic",
        intrinsic_points=2,
        validation_scope="boxed GPU-precompute plus AV/ILE execution; not posterior recovery",
        rows=rows, xml_rows=xml_rows, xml_bytes={p.name: p.stat().st_size for p in xmls},
        work_dir=str(work), log=str(log_path),
    ), indent=2, sort_keys=True))
    if made_temp and not opts.keep:
        print("Temporary products retained for inspection at %s" % work)


if __name__ == "__main__":
    main()
