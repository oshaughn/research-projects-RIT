# -*- coding: utf-8 -*-
"""Rotation of (x0, x1) as an ``inplace`` coordinate plugin.

Adapted from M. Ebiri's ``demo/hyperpipe/new_demo/demo_conversion.py`` (PR #434),
the reference example of the ``inplace`` plugin API in
``RIFT.misc.coordinate_plugin``: the forward map returns the columns it is given
with x0, x1 replaced by their rotated values, under the SAME names.

Fixes relative to the PR #434 version:
  * ``float()`` instead of ``np.float`` (removed in NumPy 1.24);
  * the forward and inverse maps return a new array and never modify their input;
  * ``get_bounds`` honours ``theta`` (it hard-coded 45 degrees) and returns the
    axis-aligned bounding box of the rotated rectangle (it rotated each
    coordinate's [lo, hi] pair as if it were a 2-vector);
  * a missing x0/x1 range raises a clear error instead of a bare KeyError, and
    nothing is printed per call.

Use:  --supplementary-coordinate-code <this file> --supplementary-coordinate-function demo_rotation
      [--get-range-from-external] [--external-range-args theta=0.3 --external-range-args buffer=0.1]
"""

import numpy as np

PLUGIN_API = "inplace"
ROTATED = ("x0", "x1")


def rotate(th):
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def _theta(kwargs):
    return float(kwargs.get("theta", np.pi / 4.0))


def _apply(X, coord_names, R):
    X_out = np.array(X, dtype=float, copy=True)
    cols = [list(coord_names).index(name) for name in ROTATED]
    X_out[:, cols] = X_out[:, cols] @ R.T
    return X_out


def demo_rotation(X, coord_names, **kwargs):
    """Rotate (x0, x1) by theta (default 45 degrees); other columns pass through."""
    return _apply(X, coord_names, rotate(_theta(kwargs)))


def inverse_demo_rotation(X, coord_names, **kwargs):
    """Undo demo_rotation."""
    return _apply(X, coord_names, rotate(_theta(kwargs)).T)


def get_bounds(param_list, bounds_dict, **kwargs):
    """File-basis ranges in, rotated-basis ranges out (same names).

    x0, x1: the axis-aligned bounding box of the rotated file-basis rectangle,
    widened on each side by ``buffer`` times its width (default 0).  Other names
    keep their range.
    """
    buffer = float(kwargs.get("buffer", 0.0))
    missing = [name for name in ROTATED if name not in bounds_dict]
    if missing:
        raise ValueError("demo_conversion.get_bounds needs file-basis ranges for {!r}".format(missing))
    (a0, b0), (a1, b1) = (np.asarray(bounds_dict[name], dtype=float) for name in ROTATED)
    corners = np.array([[a0, a1], [a0, b1], [b0, a1], [b0, b1]]) @ rotate(_theta(kwargs)).T
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    pad = buffer * (hi - lo)
    out = {}
    for name in param_list:
        if name in ROTATED:
            k = ROTATED.index(name)
            out[name] = [float(lo[k] - pad[k]), float(hi[k] + pad[k])]
        elif name in bounds_dict:
            out[name] = [float(v) for v in bounds_dict[name]]
    return out
