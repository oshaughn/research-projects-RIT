"""GW PE synthetic-targeted backend for the v2 simulation archive.

Each archive sim corresponds to ONE synthetic GW event (params: source
properties + data-generation knobs). The backend's per-(sim, level)
work unit is a sub-DAG produced by `util_RIFT_pseudo_pipe.py`; the
archive's run-queue includes that DAG via `SUBDAG EXTERNAL`. This
backend OWNS frame-writing (gwpy noise OR user-provided noise +
injection) and renders the synthetic coinc.xml, PSD, and ini before
invoking pseudo_pipe.

Two factory implementations are provided:

  * `factory_stub.subdag_factory` — writes a no-op sub-DAG that just
    creates `level_<N>.json` so the full archive flow can be exercised
    without lalsuite. Useful for development and smoke tests.
  * `factory_pseudo_pipe.subdag_factory` — the real implementation.
    Renders synthetic data (gwpy or user-supplied noise + injection),
    PSD, and an ini, then invokes util_RIFT_pseudo_pipe.py. Imports
    are deferred so this module is importable without lalsuite/gwpy.

See examples/gw_pe_synthetic_hello.py for a runnable end-to-end demo
using the stub factory.

Per BACKENDS.md: "no duplicate code paths to maintain" — the real
factory shells out to `util_RIFT_pseudo_pipe.py`, never re-implements
the PE pipeline.
"""

from . import generator     # noqa: F401
from . import summarizer    # noqa: F401
from . import same_q        # noqa: F401
from . import lookup_key    # noqa: F401
from .build import make_archive   # noqa: F401
