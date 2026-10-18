"""BNS full-rate precompute: timings include waveform generation and transfers.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and the
checkout's Code directory on PYTHONPATH. Explicit CuPy is required by default.
"""
import argparse
import contextlib
import gc
import io
import json
import os
import platform
import statistics
import time
from pathlib import Path

import lal
import lalsimulation as lalsim
import numpy as np
from RIFT import lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl
from RIFT.likelihood.precompute_array import FullRatePrecompute


def synchronize(backend):
    if backend == "cupy":
        import cupy as cp
        cp.cuda.get_current_stream().synchronize()


def compare(ref, new):
    errors = {}
    for name, index in [("U", 1), ("V", 2), ("Q", 3)]:
        err = 0.
        for det in ref[index]:
            a, b = ref[index][det], new[index][det]
            assert set(a) == set(b)
            if name == "Q":
                scale = max(np.max(np.abs(v.data.data)) for v in a.values())
                for key in a:
                    assert a[key].epoch == b[key].epoch
                    assert a[key].deltaT == b[key].deltaT
                    err = max(err, np.max(np.abs(a[key].data.data-b[key].data.data))/scale)
            else:
                scale = max(abs(v) for v in ref[1][det].values())
                err = max(err, max(abs(a[k]-b[k]) for k in a)/scale)
        errors[name] = float(err)
    assert max(errors.values()) < 1e-10, errors
    return errors


def likelihood(out, p, lmax):
    total = 0.
    for det, qs in out[3].items():
        t = fl.ComputeArrivalTimeAtDetector(det, p.phi, p.theta, p.tref)
        q = {k: ts.data.data[int(np.round((float(t)-float(ts.epoch))/ts.deltaT)+.5)]
             for k, ts in qs.items()}
        y = fl.ComputeYlms(lmax, p.incl, -p.phiref, selected_modes=qs.keys())
        f = fl.ComplexAntennaFactor(det, p.phi, p.theta, p.psi, p.tref)
        total += fl.SingleDetectorLogLikelihood(q, out[1][det], out[2][det], y, f, p.dist)
    return total


def run_case(duration, fs, lmax, backends, repeats, batch, worker_points=0):
    n = duration*fs
    epoch = lal.LIGOTimeGPS(1000000000)
    event = epoch + duration - 2.
    fmin = 25. if duration == 128 else 20.
    fmax = .45*fs
    p = lsu.ChooseWaveformParams(m1=1.4*lal.MSUN_SI, m2=1.3*lal.MSUN_SI,
        deltaT=1./fs, deltaF=1./duration, fmin=fmin, fref=fmin,
        approx=lalsim.IMRPhenomXHM, s1z=.02, s2z=-.01,
        dist=fl.distMpcRef*1e6*lal.PC_SI, radec=True, theta=.3, phi=.7)
    kw = dict(verbose=False, quiet=True, ignore_threshold=None, skip_interpolation=True,
              inv_spec_trunc_Q=True, T_spec=8.,
              extra_waveform_kwargs=dict(fd_alignment_postevent_time=2))
    # A fixed zero-noise signal on the same grid; templates are regenerated inside
    # every timed call. Detector inputs differ, testing network weight reuse.
    with contextlib.redirect_stdout(io.StringIO()):
        injection, _ = fl.internal_hlm_generator(p, lmax, verbose=False, quiet=True,
                              extra_waveform_kwargs=dict(fd_alignment_postevent_time=2))
    h = injection[2,2].data.data
    reflected = np.conj(h[(-np.arange(n)) % n])
    strain = .5*(h+reflected)*10.  # 100 Mpc, real strain; source geometry is fixed.
    data, psds = {}, {}
    for i, det in enumerate(["H1", "L1"]):
        d = lal.CreateCOMPLEX16FrequencySeries("data", epoch, -fs/2, 1./duration, lal.HertzUnit, n)
        d.data.data = strain*(1-.3*i)
        psd = lal.CreateREAL8FrequencySeries("psd", epoch, 0., 1./duration, lal.HertzUnit, n//2+1)
        lalsim.SimNoisePSDaLIGOAdVO4T1800545(psd, fmin)
        psd.data.data = np.where(psd.data.data > 0, psd.data.data*(1+.2*i), np.inf)
        data[det], psds[det] = d, psd
    del h, reflected, strain, injection
    gc.collect()
    result = dict(duration=duration, srate=fs, lmax=lmax, n=n, detectors=2, fmin=fmin, fmax=fmax,
                  truncation_seconds=8., batch_size=batch, backends={})
    reference = None
    worker_reference = []
    for backend in backends:
        actual_backend = "lal" if backend == "lal_batched" else backend
        old_batched = os.environ.get("RIFT_PRECOMPUTE_BATCHED_CROSSTERMS")
        os.environ["RIFT_PRECOMPUTE_BATCHED_CROSSTERMS"] = "1" if backend == "lal_batched" else "0"
        batched_calls_before = fl._CROSSTERM_BATCH_CALLS[0]
        ctx = None
        init = time.perf_counter()
        if actual_backend != "lal":
            ctx = FullRatePrecompute(backend, batch)
        synchronize(backend)
        init = time.perf_counter()-init
        timings = []
        stages = []
        original = fl.internal_hlm_generator
        current = {}
        def waveform(*args, **kwargs):
            begin = time.perf_counter()
            out = original(*args, **kwargs)
            current["waveform"] = time.perf_counter()-begin
            return out
        fl.internal_hlm_generator = waveform
        try:
            for repeat in range(repeats+1):
                current = {}
                synchronize(backend)
                begin = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    out = fl.PrecomputeLikelihoodTerms(event, .15, p, data, psds, lmax, fmax,
                        **kw, precompute_backend=actual_backend, precompute_batch_size=batch,
                        precompute_context=ctx)
                synchronize(backend)
                elapsed = time.perf_counter()-begin
                timings.append(elapsed)
                stages.append(current["waveform"])
                if reference is None:
                    reference = out
                errors = compare(reference, out)
                extr = p.manual_copy()
                extr.tref, extr.dist = event, 100e6*lal.PC_SI
                extr.incl, extr.psi, extr.phiref = .5, .2, .4
                # Same extrinsic samples for both products; no sampler noise.
                ld = []
                for dra in [0., .1, -.2]:
                    extr.phi = .7+dra
                    a = likelihood(reference, extr, lmax)
                    b = likelihood(out, extr, lmax)
                    ld.append(float(abs(a-b)))
                assert max(ld) < 1e-7, ld
        finally:
            fl.internal_hlm_generator = original
        row = dict(init_seconds=init, cold_seconds=timings[0], warm_seconds=timings[1:],
                   warm_median=statistics.median(timings[1:]),
                   waveform_median=statistics.median(stages[1:]),
                   arithmetic_and_transfers_median=statistics.median([a-b for a,b in zip(timings[1:], stages[1:])]),
                   max_scaled_errors=errors, max_abs_lnL_error=max(ld),
                   modes=len(out[3]["H1"]))
        if worker_points:
            # An actual many-point sequence with a fresh detector context and
            # initialized runtime; include the cache miss in point zero.
            ctx = None if actual_backend == "lal" else FullRatePrecompute(backend, batch)
            m1, m2 = p.m1, p.m2
            point_seconds = []
            batch_errors = []
            for i in range(worker_points):
                p.m1 = m1*(1.+1e-4*(i-worker_points/2))
                p.m2 = m2*(1.+1e-4*(i-worker_points/2))
                synchronize(backend)
                begin = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    products = fl.PrecomputeLikelihoodTerms(event, .15, p, data, psds, lmax, fmax,
                        **kw, precompute_backend=actual_backend, precompute_batch_size=batch,
                        precompute_context=ctx)
                synchronize(backend)
                point_seconds.append(time.perf_counter()-begin)
                if backend == backends[0]:
                    worker_reference.append(products)
                batch_errors.append(compare(worker_reference[i], products))
            p.m1, p.m2 = m1, m2
            row["worker_batch"] = dict(points=worker_points,
                total_precompute_seconds=sum(point_seconds), point_seconds=point_seconds,
                max_scaled_error=max(max(e.values()) for e in batch_errors),
                runtime="initialized; detector context fresh at point zero")
        if backend == "cupy":
            import cupy as cp
            row["device_pool_reserved_bytes"] = cp.get_default_memory_pool().total_bytes()
            row["device_pool_live_bytes"] = cp.get_default_memory_pool().used_bytes()
            row["cuda_free_bytes"], row["cuda_total_bytes"] = cp.cuda.runtime.memGetInfo()
        row["batched_crossterm_calls"] = fl._CROSSTERM_BATCH_CALLS[0] - batched_calls_before
        assert (row["batched_crossterm_calls"] > 0) == (backend == "lal_batched")
        result["backends"][backend] = row
        if old_batched is None:
            os.environ.pop("RIFT_PRECOMPUTE_BATCHED_CROSSTERMS", None)
        else:
            os.environ["RIFT_PRECOMPUTE_BATCHED_CROSSTERMS"] = old_batched
        del out, ctx
        if backend == "cupy":
            cp.fft.config.get_plan_cache().clear()
            cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--backends", nargs="+", default=["lal", "lal_batched", "numpy", "cupy"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--worker-points", type=int, default=50)
    args = ap.parse_args()
    info = dict(host=platform.node(), numpy=np.__version__, lal=lal.__version__,
                lalsimulation=lalsim.__version__, target_baseline="b0fc1469c",
                threads={k:os.environ.get(k) for k in
                ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}, cases=[])
    if "cupy" in args.backends:
        import cupy as cp
        info["cupy"] = cp.__version__
        props = cp.cuda.runtime.getDeviceProperties(0)
        info["gpu"] = str(props["name"])
        info["compute_capability"] = [props["major"], props["minor"]]
    cases = [(128,4096,2)] if args.quick else [(d,s,l) for d in [128,256]
                                           for s in [4096,8192] for l in [2,4]]
    for d,s,l in cases:
        result = run_case(d,s,l,args.backends,args.repeats,args.batch,
                          args.worker_points if (d,s,l)==(128,4096,2) else 0)
        info["cases"].append(result)
        Path(args.output).write_text(json.dumps(info, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
