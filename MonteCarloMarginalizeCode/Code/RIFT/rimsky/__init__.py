"""Rimsky-to-RIFT orchestration helpers.

The public API is intentionally independent of Rimsky's Python internals.  Rimsky
configuration and event documents are plain mappings, which keeps this bridge
usable across Rimsky release candidates without importing its large online-PE
runtime stack.
"""

from .integration import (
    RimskyIntegrationError,
    build_analysis,
    load_rimsky_config,
    normalize_event_metadata,
    write_analysis,
)

__all__ = [
    "RimskyIntegrationError",
    "build_analysis",
    "load_rimsky_config",
    "normalize_event_metadata",
    "write_analysis",
]
