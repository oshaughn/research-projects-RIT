"""Convenience builder for a GW PE synthetic-targeted archive.

Wires the backend's frozen code (generator/summarizer/same_q/
lookup_key) and the chosen subdag_factory into a fresh `Archive`
plus a `DualCondorRunQueue` ready to dispatch (or embed, if the
caller will include the wrapper DAG in a parent workflow).

Usage:
    from RIFT.simulation_manager.backends.gw_pe_synthetic import (
        make_archive, generator, summarizer, same_q, lookup_key,
    )
    from RIFT.simulation_manager.backends.gw_pe_synthetic.factory_stub \\
        import subdag_factory as stub_factory

    archive = make_archive(
        base_location="/data/archives/synthetic_gw_pe_demo",
        subdag_factory=stub_factory,    # swap for factory_pseudo_pipe
        run_queue_extra={"singularity_image": "/cvmfs/.../rift:prod"},
        submit_mode="embed",            # for hyperpipeline-style use
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from RIFT.simulation_manager.database import (
    Archive, Manifest, DualCondorRunQueue, DualCondorRequestQueue,
)

from . import generator as _generator_mod
from . import summarizer as _summarizer_mod
from . import same_q as _same_q_mod
from . import lookup_key as _lookup_key_mod


# We pass spec dicts to freeze_code (rather than bare callables) so the
# WHOLE module file is captured into <archive>/code/. This preserves
# module-level helpers (e.g. summarizer._find_posterior) and constants
# (e.g. same_q._TOL) that the entrypoint depends on. inspect.getsource
# of just the function would drop them.
_THIS_DIR = Path(__file__).parent

_GENERATOR_SPEC   = {"module_path": str(_THIS_DIR / "generator.py"),
                     "entrypoint": "generator:run"}
_SUMMARIZER_SPEC  = {"module_path": str(_THIS_DIR / "summarizer.py"),
                     "entrypoint": "summarizer:summarize"}
_SAME_Q_SPEC      = {"module_path": str(_THIS_DIR / "same_q.py"),
                     "entrypoint": "same_q:same_q"}
_LOOKUP_KEY_SPEC  = {"module_path": str(_THIS_DIR / "lookup_key.py"),
                     "entrypoint": "lookup_key:lookup_key"}


def make_archive(base_location: Union[str, Path],
                 *,
                 subdag_factory: Callable[[Any, str, int], str],
                 submit_mode: str = "submit",
                 run_queue_extra: Optional[Dict[str, Any]] = None,
                 request_queue_extra: Optional[Dict[str, Any]] = None,
                 ini_localizer: Optional[Callable[..., str]] = None,
                 ) -> Archive:
    """Create a fresh GW PE synthetic-targeted archive at
    `base_location`. Returns the Archive with both queues attached;
    caller can immediately `archive.register(params, target_level=N)`
    and `archive.request_queue.submit_pending(archive)`."""
    manifest = Manifest.new(
        name="gw_pe_synthetic",
        request_queue_kind="condor",
        run_queue_kind="condor",
        summarizer_entrypoint="summarizer:summarize",
        same_q_entrypoint="same_q:same_q",
        lookup_key_entrypoint="lookup_key:lookup_key",
        request_queue_extra=request_queue_extra or {},
        run_queue_extra=run_queue_extra or {},
        params_schema={"type": "dict",
                       "fields": {"mc": "float", "eta": "float",
                                  "distance": "float",
                                  "ra": "float", "dec": "float",
                                  "geocent_time": "float"}},
    )
    archive = Archive(
        base_location=base_location, manifest=manifest,
        generator_spec=_GENERATOR_SPEC,
        summarizer_spec=_SUMMARIZER_SPEC,
        same_q_spec=_SAME_Q_SPEC,
        lookup_key_spec=_LOOKUP_KEY_SPEC,
    )

    # If the caller supplied an ini_localizer AND the chosen factory is
    # the pseudo_pipe one, wrap it via make_factory so the localizer
    # rides along with each invocation. For the stub factory (or any
    # other) the localizer is irrelevant and we ignore it.
    final_factory = subdag_factory
    if ini_localizer is not None:
        from . import factory_pseudo_pipe as _fpp
        if subdag_factory is _fpp.subdag_factory:
            final_factory = _fpp.make_factory(ini_localizer=ini_localizer)
    run_queue = DualCondorRunQueue(
        subdag_factory=final_factory,
        submit_mode=submit_mode,
        **(run_queue_extra or {}),
    )
    request_queue = DualCondorRequestQueue(
        run_queue=run_queue,
        **(request_queue_extra or {}),
    )
    archive.run_queue = run_queue
    archive.request_queue = request_queue
    return archive
