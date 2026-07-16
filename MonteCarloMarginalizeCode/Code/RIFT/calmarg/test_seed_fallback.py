#!/usr/bin/env python3
"""
Regression tests for the calmarg PILOT "graceful degradation" contract.

Background
----------
In the in-loop-calmarg PILOT pipeline the wide ILE jobs (and the last-iteration
EXTRINSIC ILE jobs) are SEEDED from the previous iteration's consolidated cal
proposal, referenced as  cal_consolidated_$(macroiterationprev).npz  and (on OSG
file transfer) listed in condor transfer_input_files.  A calpilot only produces
that seed for iterations it<=--calmarg-pilot-max-it on-cadence, so a wide/extrinsic
iteration whose seed was never produced used to HARD-HOLD condor on a missing
transfer source (HoldReasonCode 13) and deadlock the whole DAG.

The fix pre-seeds, for EVERY referenced iteration index, a placeholder copy of the
always-present  cal_consolidated_-1.npz  iteration-0 seed -- which is a VALID prior
breadcrumb (proposal == prior).  A real calpilot overwrites its placeholder at run
time, so behavior is unchanged whenever the learned seed IS produced; a missing seed
now degrades to the PRIOR instead of dead-holding.  This module locks the two
properties that make that safe:

  1. A prior breadcrumb (the placeholder content) round-trips and, when seeded from,
     yields UNWEIGHTED prior cal draws (importance log-weights == 0).  I.e. "fall back
     to the placeholder" is exactly equivalent to "draw from the broad prior".
  2. The absent/empty-seed guard the ILE binary uses --
     `os.path.exists(p) and os.path.getsize(p) > 0` -- correctly classifies a missing
     path and a 0-byte placeholder as "not present yet" (-> prior fallback), and
     breadcrumbs.load() raises on an empty file (so the ILE except-fallback fires).

Run:  python3 -m RIFT.calmarg.test_seed_fallback
"""
import os
import tempfile

import numpy as np

import RIFT.calmarg.breadcrumbs as breadcrumbs
import RIFT.calmarg.generate_realizations as genr


def _prior_breadcrumb_cal(n_amp=5, dets=("H1", "L1"), fmin=20.0, fmax=500.0,
                          prior_sigma_val=0.1):
    """A 'placeholder' cal breadcrumb: proposal == prior (mean, diagonal cov).

    This is the semantic content of cal_consolidated_-1.npz and of the copies the
    pipeline pre-seeds into iterations a calpilot never produced.
    """
    dim = 2 * n_amp * len(dets)
    prior_mean = np.zeros(dim)
    prior_sigma = np.full(dim, prior_sigma_val)
    return dict(
        proposal_mean=prior_mean.copy(),
        proposal_cov=np.diag(prior_sigma ** 2),   # proposal == prior
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
        node_log_f=np.linspace(np.log10(fmin), np.log10(fmax), n_amp),
        n_nodes_amp=n_amp,
        dets=list(dets),
    )


def test_prior_placeholder_seeds_as_prior():
    """Seeding from a prior placeholder == unweighted prior draws (log_w == 0)."""
    n_amp = 5
    dets = ("H1", "L1")
    fmin, fmax = 20.0, 500.0
    cal = _prior_breadcrumb_cal(n_amp=n_amp, dets=dets, fmin=fmin, fmax=fmax)

    p = os.path.join(tempfile.mkdtemp(), "cal_consolidated_-1.npz")
    breadcrumbs.save(p, cal=cal, meta=dict(placeholder=True, iteration=-1))
    bc = breadcrumbs.load(p)

    _dat, cal_log_weights, nodes = genr.seed_realizations_from_breadcrumb(
        bc, T_segment=4.0, dT=1.0 / 1024, fmin=fmin, fmax=fmax,
        n_spline_points=n_amp, n_realizations=256,
        rng=np.random.default_rng(1234))

    assert nodes.shape == (256, 2 * n_amp * len(dets)), nodes.shape
    # proposal == prior  =>  log(prior/proposal) == 0 for every realization.
    assert np.allclose(cal_log_weights, 0.0, atol=1e-8), \
        "prior-placeholder importance weights should be exactly 0; max|w|=%g" \
        % np.max(np.abs(cal_log_weights))
    print("test_prior_placeholder_seeds_as_prior: OK "
          "(max|log_w|=%.2e over %d realizations)"
          % (np.max(np.abs(cal_log_weights)), len(cal_log_weights)))


def _seed_present(path):
    """Exact guard used by the ILE binary to decide 'seed present vs fall back'."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def test_absent_and_empty_seed_guard():
    """A missing path and a 0-byte placeholder both classify as 'not present'."""
    d = tempfile.mkdtemp()

    missing = os.path.join(d, "cal_consolidated_3.npz")   # never produced
    assert not _seed_present(missing), "missing seed must classify as absent"

    empty = os.path.join(d, "cal_consolidated_.npz")      # 0-byte placeholder
    open(empty, "a").close()
    assert not _seed_present(empty), "0-byte seed must classify as absent"
    # ILE loads inside try/except; an empty .npz must raise so the except-fallback fires.
    raised = False
    try:
        breadcrumbs.load(empty)
    except Exception:
        raised = True
    assert raised, "breadcrumbs.load() must raise on an empty (0-byte) placeholder"

    real = os.path.join(d, "cal_consolidated_0.npz")       # a genuine seed
    breadcrumbs.save(real, cal=_prior_breadcrumb_cal(), meta=dict(iteration=0))
    assert _seed_present(real), "a genuine breadcrumb must classify as present"
    breadcrumbs.load(real)   # must load without raising
    print("test_absent_and_empty_seed_guard: OK "
          "(missing -> absent, 0-byte -> absent+raises, real -> present)")


if __name__ == "__main__":
    test_prior_placeholder_seeds_as_prior()
    test_absent_and_empty_seed_guard()
    print("ALL OK: calmarg pilot seed absent/placeholder -> graceful prior fallback "
          "(no transfer hard-hold).")
