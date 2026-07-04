#!/usr/bin/env python3
"""Warm common synthetic RIFT JAX ILE wrappers inside a JAX-enabled container."""

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


def _synthetic_data(npts: int, n_full: int, l_max: int):
    import numpy as np
    import jax.numpy as jnp
    from RIFT.likelihood.jax_ile.core import JAXLikelihoodData

    lms = [(2, -2), (2, 2)]
    if l_max >= 3:
        lms += [(3, -3), (3, 3)]
    if l_max >= 4:
        lms += [(4, -4), (4, 4)]
    k = len(lms)
    rng = np.random.default_rng(1234)
    detectors = {}
    for i, det in enumerate(("H1", "L1")):
        q = rng.normal(size=(n_full, k)) + 1j * rng.normal(size=(n_full, k))
        u = np.eye(k, dtype=np.complex128)
        v = 0.05 * np.eye(k, dtype=np.complex128)
        detectors[det] = {
            "lms": lms,
            "Q": jnp.asarray(q, dtype=jnp.complex128),
            "U": jnp.asarray(u, dtype=jnp.complex128),
            "V": jnp.asarray(v, dtype=jnp.complex128),
            "epoch": 1000000000.0 + i * 0.002,
            "location": jnp.asarray([3000.0 + i, 4000.0 - i, 5000.0 + 2 * i], dtype=jnp.float64),
            "response": jnp.asarray(np.eye(3), dtype=jnp.float64),
            "npts_full": n_full,
            "l_max": l_max,
        }
    tvals = np.linspace(-0.05, 0.05, npts)
    return JAXLikelihoodData(detectors, deltaT=1.0 / 4096.0, gmst=1.0, tvals=tvals, tref=1000000000.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--npts", type=int, default=64)
    ap.add_argument("--n-full", type=int, default=512)
    ap.add_argument("--l-max", type=int, default=4)
    ap.add_argument("--distance-grid", type=int, default=64)
    ap.add_argument("--phi-grid", type=int, default=16)
    ap.add_argument("--psi-grid", type=int, default=8)
    args = ap.parse_args(argv)

    # Must happen before first substantial JAX use.
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

    t0 = time.perf_counter()
    steps: list[dict[str, object]] = []
    ok = True
    err = None
    device: dict[str, object] = {}
    try:
        import numpy as np
        import jax
        import jax.numpy as jnp
        from RIFT.likelihood.jax_ile.wrapper import (
            JAXDistanceMarginalizedLikelihood,
            JAXDistPhiMargLikelihood,
            JAXDistPhiPsiMargLikelihood,
            JAXExtrinsicLikelihood,
        )

        device = {
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
        }
        data = _synthetic_data(args.npts, args.n_full, args.l_max)
        batch = {
            "ra": jnp.asarray([0.1, 1.0]),
            "dec": jnp.asarray([0.2, -0.1]),
            "psi": jnp.asarray([0.3, 0.7]),
            "incl": jnp.asarray([0.8, 1.1]),
            "phiref": jnp.asarray([0.4, 1.2]),
            "dist": jnp.asarray([500.0, 800.0]),
        }

        s0 = time.perf_counter()
        like6 = JAXExtrinsicLikelihood(data)
        y = like6.log_likelihood(batch["ra"], batch["dec"], batch["psi"], batch["incl"], batch["phiref"], batch["dist"])
        np.asarray(y).tolist()
        v, g = like6.value_and_grad(np.array([0.1, 0.2, 0.3, 0.8, 0.4, 500.0]))
        steps.append({"name": "JAXExtrinsicLikelihood", "elapsed_s": time.perf_counter() - s0, "value": float(v), "grad_norm": float(np.linalg.norm(g))})

        s0 = time.perf_counter()
        like5 = JAXDistanceMarginalizedLikelihood(data, 100.0, 2000.0, n_grid=args.distance_grid)
        y = like5.log_likelihood(batch["ra"], batch["dec"], batch["psi"], batch["incl"], batch["phiref"])
        np.asarray(y).tolist()
        v, g = like5.value_and_grad(np.array([0.1, 0.2, 0.3, 0.8, 0.4]))
        steps.append({"name": "JAXDistanceMarginalizedLikelihood", "elapsed_s": time.perf_counter() - s0, "value": float(v), "grad_norm": float(np.linalg.norm(g))})

        s0 = time.perf_counter()
        like4 = JAXDistPhiMargLikelihood(data, 100.0, 2000.0, nphi=args.phi_grid, n_grid=args.distance_grid)
        y = like4.log_likelihood(batch["ra"], batch["dec"], batch["psi"], batch["incl"])
        np.asarray(y).tolist()
        v, g = like4.value_and_grad(np.array([0.1, 0.2, 0.3, 0.8]))
        steps.append({"name": "JAXDistPhiMargLikelihood", "elapsed_s": time.perf_counter() - s0, "value": float(v), "grad_norm": float(np.linalg.norm(g))})

        s0 = time.perf_counter()
        like3 = JAXDistPhiPsiMargLikelihood(
            data,
            100.0,
            2000.0,
            nphi=args.phi_grid,
            npsi=args.psi_grid,
            n_grid=args.distance_grid,
        )
        y = like3.log_likelihood(batch["ra"], batch["dec"], batch["incl"])
        np.asarray(y).tolist()
        v, g = like3.value_and_grad(np.array([0.1, 0.2, 0.8]))
        steps.append({"name": "JAXDistPhiPsiMargLikelihood", "elapsed_s": time.perf_counter() - s0, "value": float(v), "grad_norm": float(np.linalg.norm(g))})

    except Exception as exc:  # noqa: BLE001
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - t0
    result = {
        "profile": "rift_jax_ile_common",
        "ok": ok,
        "error": err,
        "host": socket.gethostname(),
        "elapsed_s": elapsed,
        "device": device,
        "cache": _cache_stats(os.environ.get("JAX_COMPILATION_CACHE_DIR")),
        "steps": steps,
    }
    _write(args.json_out, result)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

