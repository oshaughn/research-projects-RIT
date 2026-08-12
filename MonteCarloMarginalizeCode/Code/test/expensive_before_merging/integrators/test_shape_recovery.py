#!/usr/bin/env python
"""pytest wrapper for the shape-recovery merge gate.

Guarded by an env var so ordinary `pytest` sweeps stay fast.  PYTHONPATH is NOT optional: it is
what decides whether the branch or the environment's installed RIFT gets measured (see below), so
the documented invocation carries it, and the module refuses to run without it.

    export PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code:$PYTHONPATH
    RIFT_RUN_EXPENSIVE=1 pytest -v test_shape_recovery.py            # quick matrix
    RIFT_RUN_EXPENSIVE=1 RIFT_SHAPE_PRESET=standard pytest -v ...    # full gate

The canonical merge-gate invocation is run_shape_recovery.sh (JSON output +
base-vs-candidate comparison); this wrapper exists so the suite also shows up
in standard pytest tooling.
"""
import os

import pytest

from shape_recovery import (MixtureTarget, PRESETS, assert_rift_under_test, cell_budget,
                            evaluate, run_one)

_EXPENSIVE = bool(os.environ.get("RIFT_RUN_EXPENSIVE"))

pytestmark = pytest.mark.skipif(
    not _EXPENSIVE, reason="expensive merge-gate suite; set RIFT_RUN_EXPENSIVE=1")

# The gate has two entry points and only one of them carried the piece of setup that decides WHICH
# RIFT is measured: run_shape_recovery.sh exports PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/
# Code, pytest exports nothing.  So in any environment with RIFT installed -- every IGWN conda env
# -- this file gated the INSTALLED RIFT and reported pass/fail as if it had gated the branch.  On
# `GMM mix_d4_n2_s101` at the quick budget (run seed 987654) that is n_eff 42.3 (branch) vs 4.6
# (CVMFS-installed); whole-sweep medians differ by ~2x and the width_ratio signature differs
# qualitatively.  Same family as FOLLOWUPS items 3 and 5: one thing, two ways in, one of them
# missing required setup, both plausible in isolation, failure silent.
#
# This REFUSES rather than repairing sys.path.  Prepending the checkout here would make these
# numbers right and leave the operator's invocation wrong, so the next thing they run by hand --
# the probe, a bisect, an interactive reproduction -- would measure the installed RIFT again.
# Checked only under RIFT_RUN_EXPENSIVE so a plain `pytest` sweep still just skips.
_WRONG_RIFT = None
if _EXPENSIVE:
    try:
        assert_rift_under_test(os.environ.get("RIFT_SHAPE_CHECKOUT"),
                               who="the pytest entry point of the shape-recovery gate")
    except RuntimeError as exc:
        _WRONG_RIFT = str(exc)
# raised OUTSIDE the handler: inside it, pytest chains the RuntimeError and prints its traceback,
# burying the operator-facing message this exists to deliver.  Collection ERROR, message only.
if _WRONG_RIFT:
    pytest.fail(_WRONG_RIFT, pytrace=False)

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
