# -*- coding: utf-8 -*-
"""
RIFT.precision
==============

Centralized high-precision floating-point dtype for RIFT.

RIFT historically used ``numpy.float128`` directly throughout the integrators
to suppress overflow when accumulating very large or very small probability
weights.  ``numpy.float128`` is, however, a *platform-dependent* alias:

* On x86_64 Linux it is ``numpy.longdouble`` with 80-bit extended precision
  stored in 16 bytes, and exposes the alias ``numpy.float128``.
* On macOS arm64 (Apple Silicon), Windows MSVC, and many embedded / non-x86
  Linux builds, ``numpy.longdouble`` has the same width as ``numpy.float64``
  (8 bytes), and ``numpy.float128`` does not exist.
* In NumPy 2.x the ``numpy.float128`` alias was further narrowed and is now
  only present where the platform actually has a wider long double type.

Hardcoding ``numpy.float128`` therefore breaks imports on any platform that
lacks an extended-precision long double.  This module provides:

* ``RiftFloat`` -- the dtype to use for high-precision accumulators.  Equals
  ``numpy.longdouble`` whenever the platform really gives extra precision
  (``itemsize > 8``), otherwise falls back to ``numpy.float64``.
* ``RIFT_FLOAT_HIGH_PRECISION`` -- ``True`` iff ``RiftFloat`` is wider than
  ``numpy.float64``.  Use this if a code path needs to know whether the
  extra precision is actually available (e.g. to skip a downcast that is
  only meaningful when float128 is real).

Recommended usage
-----------------

Replace ::

    import numpy
    neff = numpy.float128("inf")
    arr = numpy.array(values, dtype=numpy.float128)
    if weights.dtype == numpy.float128:
        weights = weights.astype(numpy.float64)

with ::

    from RIFT.precision import RiftFloat
    neff = RiftFloat("inf")
    arr = numpy.array(values, dtype=RiftFloat)
    if weights.dtype == RiftFloat:
        weights = weights.astype(numpy.float64)

When ``RiftFloat`` is ``numpy.float64`` the type-equality guards become
self-consistent (the ``astype(float64)`` is a no-op) and no platform-specific
branching is required at the call site.
"""

import numpy as _np

__all__ = [
    "RiftFloat",
    "RIFT_FLOAT_HIGH_PRECISION",
    "RIFT_FLOAT_NAME",
]


def _select_high_precision_dtype():
    """Pick the widest available real-floating dtype, falling back to float64.

    Prefers ``numpy.longdouble`` (always defined) whenever the platform
    actually gives more than 8-byte storage; otherwise returns
    ``numpy.float64`` so that consumers can use the result as a drop-in
    replacement for ``numpy.float128`` without raising on platforms that
    do not expose it.
    """
    longdouble_itemsize = _np.dtype(_np.longdouble).itemsize
    if longdouble_itemsize > 8:
        return _np.longdouble, True
    # Some NumPy 1.x builds expose float128 even when longdouble is 8 bytes
    # (older alias kept for ABI stability).  Treat that as high-precision.
    if hasattr(_np, "float128"):
        try:
            if _np.dtype(_np.float128).itemsize > 8:
                return _np.float128, True
        except TypeError:
            pass
    return _np.float64, False


RiftFloat, RIFT_FLOAT_HIGH_PRECISION = _select_high_precision_dtype()
RIFT_FLOAT_NAME = _np.dtype(RiftFloat).name
