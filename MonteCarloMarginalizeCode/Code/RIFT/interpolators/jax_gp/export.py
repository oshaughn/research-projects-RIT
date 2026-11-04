"""
Self-contained, differentiable export for jax_gp interpolators.

This replaces "dump the lnL grid and hope": the artifact written here is a small,
portable bundle that reconstructs a *pure-JAX, differentiable* ``lnL(theta)`` --
``jax.grad`` / ``jax.value_and_grad`` work out of the box, which is what
downstream users need.

Layout (two files sharing a base path)::

    <base>.npz        whitening vectors + method parameters (NumPy arrays)
    <base>.meta.json  schema/method/dimension/target-scaling + coordinate names

``coord_names`` records *which* fit coordinates the axes of ``theta`` are (e.g.
``["mc", "eta", "chi_eff", ...]``).  The exported lnL is differentiable in those
*fit* coordinates -- the same space the GP was trained on.  (Pushing the
derivative all the way back to raw physical parameters would require a JAX
reimplementation of CIP's coordinate transforms; that is deliberately out of
scope here and noted as future work.)
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import get_interpolator


def _split_path(path):
    """Accept a base path or either concrete file; return the base path."""
    for suffix in (".meta.json", ".npz"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def save(model, path, coord_names=None, extra_meta=None):
    """Write ``model`` to ``<path>.npz`` + ``<path>.meta.json``; return the base path."""
    base = _split_path(path)
    meta, arrays = model.export_state()
    if coord_names is not None:
        meta["coord_names"] = list(coord_names)
    if extra_meta:
        meta["extra"] = extra_meta
    np.savez(base + ".npz", **arrays)
    with open(base + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return base


def load(path):
    """Reconstruct a (predict + differentiate) interpolator from an exported bundle.

    Returns an interpolator instance exposing the usual contract:
    ``predict(X)``, ``predict_callable()``, ``lnL_physical(theta)`` (pure JAX),
    ``lnL_and_grad(theta)``, ``grad_fn()``.
    """
    base = _split_path(path)
    with open(base + ".meta.json") as f:
        meta = json.load(f)
    with np.load(base + ".npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    cls = get_interpolator(meta["method"])
    model = cls.from_state(meta, arrays)
    model.coord_names = meta.get("coord_names")
    return model


def exists(path):
    base = _split_path(path)
    return os.path.exists(base + ".npz") and os.path.exists(base + ".meta.json")
