#!/usr/bin/env python3
"""Warm common RIFT CuPy kernels inside a GPU-enabled container."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path


def _cache_stats(path: str | None) -> dict[str, object]:
    if not path:
        return {"path": None, "files": 0, "bytes": 0}
    root = Path(path)
    files = 0
    total = 0
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                files += 1
                total += p.stat().st_size
    return {"path": str(root), "files": files, "bytes": total}


def _write(path: str | None, obj: dict[str, object]) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    if path:
        Path(path).write_text(text, encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    steps: list[dict[str, object]] = []
    ok = True
    err = None
    device: dict[str, object] = {}
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        device = {
            "name": name,
            "compute_capability": f"{props['major']}.{props['minor']}",
            "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "cupy_version": cp.__version__,
        }

        # 1. Standard NoLoop Q inner-product RawKernel.
        from RIFT.likelihood.Q_inner_product import Q_inner_product_cupy

        Q = cp.ascontiguousarray(
            (cp.random.random((64, 4)) + 1j * cp.random.random((64, 4))).astype(cp.complex128)
        )
        A = cp.ascontiguousarray(
            (cp.random.random((8, 4)) + 1j * cp.random.random((8, 4))).astype(cp.complex128)
        )
        starts = cp.asarray([0, 2, 4, 6, 8, 10, 12, 14], dtype=cp.int32)
        s0 = time.perf_counter()
        out = Q_inner_product_cupy(Q, A, starts, 16)
        cp.cuda.Stream.null.synchronize()
        steps.append({"name": "Q_inner_product_cupy", "elapsed_s": time.perf_counter() - s0, "shape": list(out.shape)})

        # 2. Fused calmarg kernels.
        from RIFT.likelihood.Q_fused_calmarg import (
            Q_fused_calmarg_cupy,
            Q_fused_calmarg_distmarg_cupy,
        )

        n_det, n_cal, n_window, n_lms, n_ext, npts = 2, 3, 32, 4, 8, 16
        Qf = cp.ascontiguousarray(
            (cp.random.random((n_det, n_cal * n_window, n_lms))
             + 1j * cp.random.random((n_det, n_cal * n_window, n_lms))).astype(cp.complex128)
        )
        Af = cp.ascontiguousarray(
            (cp.random.random((n_det, n_ext, n_lms))
             + 1j * cp.random.random((n_det, n_ext, n_lms))).astype(cp.complex128)
        )
        ifirst = cp.ascontiguousarray(cp.tile(cp.arange(n_ext, dtype=cp.int32), (n_det, 1)))
        inv_dist = cp.ones(n_ext, dtype=cp.float64)
        rho_sq = cp.ones((n_ext, npts), dtype=cp.float64)
        w_t = cp.ones(npts, dtype=cp.float64) / npts
        s0 = time.perf_counter()
        y = Q_fused_calmarg_cupy(Qf, Af, ifirst, inv_dist, rho_sq, w_t, n_cal, n_window)
        cp.cuda.Stream.null.synchronize()
        steps.append({"name": "Q_fused_calmarg_cupy", "elapsed_s": time.perf_counter() - s0, "shape": list(y.shape)})

        lnI = cp.zeros((8, 8), dtype=cp.float64)
        distmarg = {
            "lnI_array": lnI,
            "s0": -4.0,
            "ds": 1.0,
            "smin": -4.0,
            "smax": 3.0,
            "t0": -4.0,
            "dt": 1.0,
            "tmax": 3.0,
            "xmin": 0.001,
            "xmax": 10.0,
            "sqrt_bmax": 1.0,
            "bref": 1.0,
        }
        s0 = time.perf_counter()
        yd = Q_fused_calmarg_distmarg_cupy(Qf, Af, ifirst, inv_dist, rho_sq, w_t, n_cal, n_window, distmarg)
        cp.cuda.Stream.null.synchronize()
        steps.append({"name": "Q_fused_calmarg_distmarg_cupy", "elapsed_s": time.perf_counter() - s0, "shape": list(yd.shape)})

        # 3. RIFT's temporary cupy interp ElementwiseKernel.
        from RIFT.interpolators.interp_gpu import interp

        xp = cp.linspace(0.0, 1.0, 32, dtype=cp.float64)
        fp = cp.sin(xp)
        x = cp.linspace(-0.1, 1.1, 128, dtype=cp.float64)
        s0 = time.perf_counter()
        zi = interp(x, xp, fp)
        cp.cuda.Stream.null.synchronize()
        steps.append({"name": "interp_gpu.interp", "elapsed_s": time.perf_counter() - s0, "shape": list(zi.shape)})

    except Exception as exc:  # noqa: BLE001
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t0
    result = {
        "profile": "rift_cupy_common",
        "ok": ok,
        "error": err,
        "host": socket.gethostname(),
        "elapsed_s": elapsed,
        "device": device,
        "cache": _cache_stats(os.environ.get("CUPY_CACHE_DIR")),
        "steps": steps,
    }
    _write(args.json_out, result)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

