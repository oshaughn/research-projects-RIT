#!/usr/bin/env python3
"""Reproduce the full-padding versus retained-grid FFT microbenchmark.

Run each timed arm in a fresh process so ``ru_maxrss`` and the CuPy memory pool
belong to that arm.  ``parity`` evaluates both arms on one deterministic batch
and reports differences after a nonlinear likelihood-like map and time
integration.  This is a transform/kernel benchmark, not an ILE evidence run.

Examples (inside a RIFT environment)::

    python benchmark_bandlimited_retained_fft.py --backend cupy --arm full \
        --npts 614 --factor 64
    python benchmark_bandlimited_retained_fft.py --backend cupy --arm retained \
        --npts 614 --factor 64
    python benchmark_bandlimited_retained_fft.py --backend cupy --arm parity \
        --npts 614 --factor 64
"""
import argparse
import json
import os
import resource
import time

# A benchmark must not inherit a many-thread BLAS default and then measure
# thread creation or exceed a batch system's process limit during imports.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np

from RIFT.likelihood import time_marginalization_quadrature as tmq


def _backend(name):
    if name == "numpy":
        from scipy.special import logsumexp
        return np, logsumexp
    import cupy
    from cupyx.scipy.special import logsumexp
    if cupy.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("--backend cupy requested but no CUDA device is visible")
    return cupy, logsumexp


def _synchronize(xpy):
    if xpy is not np:
        xpy.cuda.Stream.null.synchronize()


def _memory_start(xpy):
    if xpy is np:
        return None
    try:
        xpy.fft.config.get_plan_cache().clear()
    except Exception:
        pass
    xpy.get_default_memory_pool().free_all_blocks()
    xpy.get_default_pinned_memory_pool().free_all_blocks()
    _synchronize(xpy)
    free, total = xpy.cuda.runtime.memGetInfo()
    return free, total


def _memory_finish(xpy, start):
    out = {
        "host_maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "cupy_pool_total_mib": None,
        "device_resident_delta_mib": None,
        "device_total_mib": None,
    }
    if xpy is not np:
        free, _ = xpy.cuda.runtime.memGetInfo()
        out.update(
            cupy_pool_total_mib=xpy.get_default_memory_pool().total_bytes() / 2**20,
            device_resident_delta_mib=(start[0] - free) / 2**20,
            device_total_mib=start[1] / 2**20,
        )
    return out


def _inputs(nrows, npts, xpy):
    rng = np.random.default_rng(20260905 + npts)
    host = rng.normal(size=(nrows, npts)) + 1j * rng.normal(
        size=(nrows, npts))
    host *= np.exp(0.013j * np.arange(nrows)[:, None])
    return xpy.asarray(host, dtype=np.complex128)


def _transform(arm, rows, factor, cache, xpy):
    if arm == "full":
        return tmq.reflected_bandlimited_upsample(rows, factor, xpy=xpy)
    return tmq._reflected_bandlimited_upsample_retained(
        rows, factor, plan_cache=cache, xpy=xpy)


def _timed(args, xpy):
    batch = args.batch or max(1, int(
        tmq._DENSE_CHUNK_BYTES // (args.npts * args.factor * 16 * 8)))
    rows = _inputs(batch, args.npts, xpy)
    if xpy is not np:
        xpy.fft.fft(xpy.ones((1, 32), dtype=np.complex128)).sum().get()
    start_memory = _memory_start(xpy)
    cache = {}
    checksum = xpy.zeros((), dtype=np.float64)
    _synchronize(xpy)
    start = time.perf_counter()
    done = 0
    while done < args.rows:
        take = min(batch, args.rows - done)
        dense = _transform(args.arm, rows[:take], args.factor, cache, xpy)
        checksum += xpy.sum(dense[..., ::args.factor].real)
        del dense
        done += take
    _synchronize(xpy)
    wall = time.perf_counter() - start
    checksum = float(checksum if xpy is np else checksum.get())
    record = {
        "arm": args.arm,
        "backend": args.backend,
        "npts": args.npts,
        "factor": args.factor,
        "rows": args.rows,
        "batch": batch,
        "dense_points_evaluated": args.rows * ((args.npts - 1) * args.factor + 1),
        "wall_s": wall,
        "rows_per_s": args.rows / wall,
        "checksum": checksum,
        "full_fft_length": 2 * args.npts * args.factor,
        "retained_grid_length": (args.npts - 1) * args.factor + 1,
        "retained_plan_fft_length": max(
            (p["n_fft"] for p in cache.values()), default=None),
    }
    record.update(_memory_finish(xpy, start_memory))
    return record


def _parity(args, xpy, logsumexp):
    batch = args.batch or 32
    rows = _inputs(batch, args.npts, xpy)
    full = tmq.reflected_bandlimited_upsample(rows, args.factor, xpy=xpy)
    retained = tmq._reflected_bandlimited_upsample_retained(
        rows, args.factor, xpy=xpy)
    # Smooth and nonlinear, as distance/phase marginalization is.  The factor
    # 100 makes transform-level roundoff visible instead of rounding to zero.
    lnlt_full = 100.0 * xpy.logaddexp(0.0, full.real)
    lnlt_retained = 100.0 * xpy.logaddexp(0.0, retained.real)

    def integrate(lnlt):
        offset = xpy.max(lnlt, axis=-1)
        density = xpy.exp(lnlt - offset[:, None])
        density[:, 0] *= 0.5
        density[:, -1] *= 0.5
        return offset + xpy.log(xpy.sum(density, axis=-1) / args.factor)

    il_full = integrate(lnlt_full)
    il_retained = integrate(lnlt_retained)
    lnz_full = logsumexp(il_full) - np.log(batch)
    lnz_retained = logsumexp(il_retained) - np.log(batch)
    _synchronize(xpy)

    def scalar(value):
        return float(value if xpy is np else value.get())

    return {
        "arm": "parity",
        "backend": args.backend,
        "npts": args.npts,
        "factor": args.factor,
        "rows": batch,
        "max_abs_delta_kappa": scalar(xpy.max(xpy.abs(retained - full))),
        "max_abs_delta_lnL": scalar(xpy.max(xpy.abs(il_retained - il_full))),
        "delta_lnZ": scalar(lnz_retained - lnz_full),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--arm", choices=("full", "retained", "parity"), required=True)
    parser.add_argument("--npts", type=int, required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    xpy, logsumexp = _backend(args.backend)
    record = (_parity(args, xpy, logsumexp) if args.arm == "parity"
              else _timed(args, xpy))
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
