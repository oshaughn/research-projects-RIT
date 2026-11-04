"""Disk cache of exported (traced and lowered) JAX programs.

JAX's persistent compilation cache is keyed on the lowered HLO, so it skips only
the XLA compile: every run still traces and lowers.  For ripple's XPHM that
costs ~6 min per run (S250830bp, seed climb).  ``exported_call`` serializes the
lowered program with ``jax.export`` and reloads it on later runs; with the
compilation cache also set, a warm run neither traces nor compiles.

The key covers the JAX version, backend, the caller's source files, the argument
shapes and dtypes, and ``key_arrays`` (every array the function closes over).
Any change produces a new entry; nothing is invalidated in place.
"""
import hashlib
import os

import numpy as np
import jax
from jax import export


def _digest(parts):
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:32]


def _specs(args):
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(np.shape(a), np.asarray(a).dtype), args)


def exported_call(fn, example_args, cache_dir, key_parts=(), key_arrays=(), source_files=()):
    """Return a callable equivalent to jax.jit(fn), loaded from cache_dir if present."""
    specs = _specs(example_args)
    parts = [jax.__version__, jax.default_backend(), repr(jax.tree.structure(specs)),
             repr(jax.tree.leaves(specs))] + list(key_parts)
    for a in key_arrays:
        a = np.asarray(a)
        parts += [a.dtype.str, a.shape, a.tobytes()]
    for f in source_files:
        with open(f, "rb") as fh:
            parts.append(fh.read())
    path = os.path.join(cache_dir, "jaxexport-%s.bin" % _digest(parts))
    if os.path.exists(path):
        with open(path, "rb") as fh:
            exp = export.deserialize(bytearray(fh.read()))
    else:
        exp = export.export(jax.jit(fn))(*specs)
        try:
            blob = exp.serialize()
        except Exception as exc:          # e.g. flatbuffers missing: run without the cache
            print(" [export_cache] cannot serialize (%r); not caching %s" % (exc, os.path.basename(path)))
            return jax.jit(exp.call)
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp%d" % os.getpid()
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, path)             # atomic: concurrent jobs may race
    return jax.jit(exp.call)
