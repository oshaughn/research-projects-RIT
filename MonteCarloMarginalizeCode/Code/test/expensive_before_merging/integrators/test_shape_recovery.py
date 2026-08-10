#!/usr/bin/env python
"""pytest wrapper for the shape-recovery merge gate.

Guarded by an env var so ordinary `pytest` sweeps stay fast:

    RIFT_RUN_EXPENSIVE=1 pytest -v test_shape_recovery.py            # quick matrix
    RIFT_RUN_EXPENSIVE=1 RIFT_SHAPE_PRESET=standard pytest -v ...    # full gate

The canonical merge-gate invocation is run_shape_recovery.sh (JSON output +
base-vs-candidate comparison); this wrapper exists so the suite also shows up
in standard pytest tooling.
"""
import os

import pytest

from shape_recovery import MixtureTarget, PRESETS, evaluate, run_one, cell_budget

pytestmark = pytest.mark.skipif(
    not os.environ.get("RIFT_RUN_EXPENSIVE"),
    reason="expensive merge-gate suite; set RIFT_RUN_EXPENSIVE=1")

_PRESET = PRESETS[os.environ.get("RIFT_SHAPE_PRESET", "quick")]
_STRICT = os.environ.get("RIFT_SHAPE_STRICT", "AV,GMM").split(",")

_MATRIX = [(kind, d, nc, ts)
           for kind in _STRICT
           for d in _PRESET["dims"]
           for nc in _PRESET["ncomps"]
           for ts in _PRESET["seeds"]]


@pytest.mark.parametrize("kind,ndim,ncomp,tseed", _MATRIX)
def test_shape_recovery(kind, ndim, ncomp, tseed):
    target = MixtureTarget(ndim, ncomp, tseed)
    # via cell_budget(), not nmax_per_dim*ndim inline: otherwise a per-cell override applies
    # under run_shape_recovery.sh but not under pytest, and the two disagree on the same cell.
    r = run_one(kind, target,
                cell_budget(kind, ndim, ncomp, tseed, _PRESET["nmax_per_dim"]),
                _PRESET["neff"])
    # evaluate() returns a STATUS STRING, not a bool.  "FAIL", "STARVED" and "ERROR" are all
    # truthy, so the `assert ok` this line used to carry passed on every outcome it existed to
    # catch -- vacuous since 6467ac91 changed evaluate()'s contract from bool to status.
    status, reasons = evaluate(r)
    why = "{} on {}: {} -- {}".format(kind, target.name, status, "; ".join(reasons))
    # STARVED is "shape untestable at this budget", and the gate defines it as NON-blocking in
    # absolute terms, gating only differentially (6467ac91: whole d=8 rows legitimately starve at
    # production budgets).  Skipping honours that and keeps it visible in the pytest summary; it
    # is NOT a pass.  Absolute-vs-base gating is compare_shape_results.py's job, not this one's.
    if status == "STARVED":
        pytest.skip(why + " [not a pass: use run_shape_recovery.sh + compare_shape_results.py "
                          "to gate starvation against a base run]")
    assert status == "PASS", why
