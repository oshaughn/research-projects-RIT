#!/usr/bin/env python3
"""Synchronized host-runtime benchmark; excludes process/container transfer time."""

import argparse
import json
import time

import numpy as np

from RIFT.likelihood.gpu_precompute import compound_precompute_arrays


def sync(xp):
    if xp.__name__.startswith("cupy"):
        xp.cuda.Stream.null.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("numpy", "cupy"), default="cupy")
    p.add_argument("--bins", type=int, default=1 << 20)
    p.add_argument("--basis", type=int, default=40)
    p.add_argument("--modes", type=int, default=1)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--intrinsics", type=int, default=3)
    p.add_argument("--frequency-chunk", type=int, default=1 << 16)
    p.add_argument("--seed", type=int, default=771)
    args = p.parse_args()
    if args.backend == "cupy":
        import cupy as xp
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device")
    else:
        xp = np

    rng = np.random.default_rng(args.seed)
    n = args.bins
    # Allocate once: this models a persistent ILE worker with data/PSD resident.
    data = xp.asarray((rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128))
    weights = xp.asarray(rng.uniform(0.2, 1.5, size=n).astype(np.float64))
    context = {}
    records = []
    for intrinsic in range(args.intrinsics):
        basis = xp.asarray((rng.normal(size=(args.basis, args.modes, n))
                            + 1j * rng.normal(size=(args.basis, args.modes, n))).astype(np.complex128))
        basis_conj = xp.asarray((rng.normal(size=(args.basis, args.modes, n))
                                 + 1j * rng.normal(size=(args.basis, args.modes, n))).astype(np.complex128))
        sync(xp)
        t0 = time.perf_counter()
        out = compound_precompute_arrays(
            basis, basis_conj, data, weights, 1.0 / 8192.0, 1.0 / (n / 8192.0),
            0, min(args.window, n), backend=xp, return_device=True, context=context,
            frequency_chunk=args.frequency_chunk)
        sync(xp)
        elapsed = time.perf_counter() - t0
        records.append({"intrinsic": intrinsic, "host_runtime_s": elapsed})
        del basis, basis_conj, out
    print(json.dumps({
        "backend": args.backend, "bins": n, "basis": args.basis,
        "modes": args.modes, "intrinsics": args.intrinsics,
        "timing_scope": "inside worker; synchronized; excludes container transfer",
        "records": records,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
