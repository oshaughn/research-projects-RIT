"""Centralised NR-waveform-catalog import shim.

Every RIFT call site that used to do ::

    try:
        import NRWaveformCatalogManager3 as nrwf
        hasNR = True
    except ImportError:
        nrwf = None
        hasNR = False

now does ::

    from RIFT.physics._nrwf_loader import get_nrwf
    nrwf, hasNR = get_nrwf()

Resolution order
----------------
1. ``nrcatalog.compat_nrwf`` (cleanup_2026 branch of
   NRWaveformCatalogManager_repo). Re-exports every legacy name and
   overlays new providers on top.
2. ``NRWaveformCatalogManager3`` (the legacy module by itself).
3. ``None`` (NR unavailable).

Override with the environment variable ``RIFT_NRWF_BACKEND``:
``auto`` (default), ``new``, ``legacy``, or ``none``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

log = logging.getLogger(__name__)

_cached: Optional[Tuple[object, bool]] = None


def get_nrwf() -> Tuple[Optional[object], bool]:
    """Return ``(module, has_nrwf_bool)``. Cached after the first call."""
    global _cached
    if _cached is not None:
        return _cached

    backend = os.environ.get("RIFT_NRWF_BACKEND", "auto").lower()

    if backend == "none":
        _cached = (None, False)
        return _cached

    if backend in ("auto", "new"):
        try:
            import nrcatalog.compat_nrwf as nrwf  # type: ignore
            log.info("RIFT NR backend: nrcatalog.compat_nrwf (legacy %s)",
                     "wrapped" if nrwf.is_using_legacy() else "absent")
            _cached = (nrwf, True)
            return _cached
        except ImportError as exc:
            if backend == "new":
                log.warning("RIFT_NRWF_BACKEND=new but nrcatalog not importable: %s", exc)
                _cached = (None, False)
                return _cached
            # fall through to legacy

    if backend in ("auto", "legacy"):
        try:
            import NRWaveformCatalogManager3 as nrwf  # type: ignore
            log.info("RIFT NR backend: NRWaveformCatalogManager3 (legacy)")
            _cached = (nrwf, True)
            return _cached
        except ImportError as exc:
            log.info("RIFT NR backend: no NR module available (%s)", exc)

    _cached = (None, False)
    return _cached


def reset_cache() -> None:
    """Forget the cached choice — used in tests."""
    global _cached
    _cached = None
