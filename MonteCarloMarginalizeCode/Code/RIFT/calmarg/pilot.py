"""
Calibration pilot / brute-force reference (Options A + C in DESIGN_adaptive_driver.md).

  A (brute-force reference, the ONLY validation): marginalize cal with a large PRIOR set,
    converged.  Slow, ground truth.  `brute_force_logZcal`.
  C (production pilot): harvest the top-fraction high-lnL points from the previous
    iteration's *.composite, do full cal there, fit a Gaussian proposal, and seed the
    next iteration's cal realizations with importance weights.  `harvest_high_L`,
    `fit_pilot_proposal`, `consolidate`, `seed_cal`.

The actual DAG wiring (pilot_N || wide_N, consolidation barrier, pilot_N -> wide_{N+1},
the iteration cap) lives in the pipeline builder; it is STUBBED here with TODOs so the
plan is remembered.  The numeric core (fit, seed, brute-vs-seeded agreement) is real and
tested below.
"""
from __future__ import division

import numpy as np
from scipy.special import logsumexp

from RIFT.calmarg import adaptive, breadcrumbs
from RIFT.calmarg import generate_realizations as _gr


# ---------------------------------------------------------------------------
# C: harvest pilot points from a previous iteration's composite
# ---------------------------------------------------------------------------
def harvest_high_L(composite_path, top_fraction=0.05, lnL_col="lnL", max_points=512):
    """Return the indices+rows of the top `top_fraction` of evaluated points by lnL from
    a RIFT *.composite file (whitespace, named header).  These are the pilot points where
    we will do full calibration (the cal posterior is ~the same across the high-L region,
    so a handful suffice)."""
    arr = np.atleast_2d(np.genfromtxt(composite_path, names=True))
    names = arr.dtype.names
    col = lnL_col if lnL_col in names else _guess_lnL_col(names)
    lnL = arr[col]
    n_keep = max(1, int(np.ceil(len(lnL) * top_fraction)))
    if max_points:
        n_keep = min(n_keep, max_points)
    order = np.argsort(lnL)[::-1][:n_keep]
    return order, arr[order]


def _guess_lnL_col(names):
    for cand in ("lnL", "lnL_raw", "loglikelihood", "log_likelihood"):
        if cand in names:
            return cand
    raise KeyError("no lnL-like column in composite; columns=%s" % (names,))


# ---------------------------------------------------------------------------
# A: brute-force reference (prior-only, large n_cal)
# ---------------------------------------------------------------------------
def brute_force_logZcal(log_L):
    """Self-normalized cal-marginalized log-likelihood from a LARGE PRIOR cal set:
    log Z_cal = logmeanexp(log_L)  (importance weights are uniform for prior draws).
    Returns (logZ, neff)."""
    log_L = np.asarray(log_L)
    logZ = logsumexp(log_L) - np.log(len(log_L))
    neff = adaptive.neff_from_logweights(log_L)
    return float(logZ), neff


# ---------------------------------------------------------------------------
# C: fit a pilot proposal, seed the next run, consolidate breadcrumbs
# ---------------------------------------------------------------------------
def fit_pilot_proposal(nodes, log_resp, prior_mean, prior_sigma, node_log_f,
                       n_nodes_amp, dets, beta=1.0, meta=None):
    """Fit a Gaussian cal proposal from pilot evaluations and package it as a breadcrumb
    `cal` dict.  `log_resp` = posterior responsibility (log_w + log integral L) per
    realization, averaged/accumulated over the harvested pilot points by the caller."""
    mean, cov = adaptive.fit_proposal(nodes, log_resp, beta)
    return dict(proposal_mean=mean, proposal_cov=cov,
                prior_mean=np.asarray(prior_mean), prior_sigma=np.asarray(prior_sigma),
                node_log_f=np.asarray(node_log_f), n_nodes_amp=int(n_nodes_amp),
                dets=list(dets))


def seed_cal(cal_proposal, n_cal, rng=None):
    """Draw `n_cal` cal node vectors from the learned proposal and return
    (nodes, log_weights) where log_weights = log prior - log proposal (Phase 0 importance
    weights for the marginalization).  Feed nodes through
    adaptive.nodes_to_cal_factors(...) per detector to get the actual cal factors."""
    rng = rng or _gr._default_cal_rng('calmarg.seed_cal')
    mean = np.asarray(cal_proposal["proposal_mean"])
    cov = np.asarray(cal_proposal["proposal_cov"])
    nodes = rng.multivariate_normal(mean, cov, size=n_cal)
    log_q = adaptive._mvn_logpdf(nodes, mean, cov)
    log_p = adaptive.log_prior(nodes, np.asarray(cal_proposal["prior_mean"]),
                               np.asarray(cal_proposal["prior_sigma"]))
    return nodes, (log_p - log_q)


def consolidate(breadcrumb_paths, out_path=None):
    """Combine cal proposals from several pilot breadcrumbs into one (the consolidation
    job between iteration N and N+1).  Gaussian case: precision-weighted combination
    (a moment-matched product/average of the per-pilot Gaussians)."""
    cals = [breadcrumbs.load(p)["cal"] for p in breadcrumb_paths]
    cals = [c for c in cals if c is not None]
    if not cals:
        raise ValueError("no cal proposals to consolidate")
    # precision-weighted mean, average covariance (robust, simple)
    Ps = [np.linalg.inv(c["proposal_cov"]) for c in cals]
    P = np.sum(Ps, axis=0)
    cov = np.linalg.inv(P)
    mean = cov @ np.sum([Pi @ c["proposal_mean"] for Pi, c in zip(Ps, cals)], axis=0)
    out = dict(cals[0]); out["proposal_mean"] = mean; out["proposal_cov"] = cov
    if out_path:
        breadcrumbs.save(out_path, cal=out, meta=dict(consolidated_from=len(cals)))
    return out


# ---------------------------------------------------------------------------
# DAG job stubs (Option C pipeline wiring -- TODO; see DESIGN_adaptive_driver.md)
# ---------------------------------------------------------------------------
def pilot_job(prev_composite, data_args, out_breadcrumb, top_fraction=0.05, n_cal_full=1000):
    """STUB.  pilot_N: harvest top-fraction points from prev_composite, run FULL cal at
    each (large prior n_cal, parallel), fit the proposal, write a breadcrumb.
    TODO: wire to the ILE precompute/likelihood to get per-point per-realization lnL."""
    raise NotImplementedError("pilot_job: pipeline wiring TODO (see DESIGN_adaptive_driver.md)")


def consolidation_job(pilot_breadcrumbs, out_breadcrumb):
    """consolidation_N: collect pilot breadcrumbs -> one consolidated proposal that seeds
    wide_{N+1}.  (The numeric core is `consolidate` above.)"""
    return consolidate(pilot_breadcrumbs, out_path=out_breadcrumb)


# ---------------------------------------------------------------------------
# A-vs-C validation: brute force == pilot-seeded on Z_cal, at far higher efficiency
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    dim = 8
    prior_mean = np.zeros(dim); prior_sigma = np.ones(dim)
    true_node = prior_mean + 2.0 * prior_sigma * rng.standard_normal(dim) / np.sqrt(dim)
    like_sigma = 0.4

    def log_like(nodes):                       # extrinsic-marg log-like proxy (no prior)
        z = (nodes - true_node) / like_sigma
        return -0.5 * np.sum(z * z, axis=1)

    # A: brute force, large PRIOR set (ground truth)
    big = rng.multivariate_normal(prior_mean, np.diag(prior_sigma ** 2), size=20000)
    logZ_brute, neff_brute = brute_force_logZcal(log_like(big))
    print("A brute force : logZcal=%.4f  neff=%.1f / 20000" % (logZ_brute, neff_brute))

    # C: learn the proposal (pilot), then seed a SMALL set and importance-weight
    res = adaptive.adaptive_cal(log_like, prior_mean, prior_sigma, n_nodes_amp=dim // 2,
                                n_real=300, n_iter=6, rng=rng)
    cal = dict(proposal_mean=res["proposal_mean"], proposal_cov=res["proposal_cov"],
               prior_mean=prior_mean, prior_sigma=prior_sigma,
               node_log_f=np.linspace(1, 3, dim // 2), n_nodes_amp=dim // 2,
               dets=["H1", "L1", "V1"])
    nodes, log_w = seed_cal(cal, n_cal=300, rng=rng)
    log_resp = log_w + log_like(nodes)
    # UNBIASED importance estimate Z_cal = (1/M) sum_c w_c L_c, w_c = prior/proposal,
    # E[w]=1 -> normalize by log(M), NOT logsumexp(log_w) (the biased self-normalized form)
    logZ_seeded = logsumexp(log_resp) - np.log(len(nodes))
    neff_seeded = adaptive.neff_from_logweights(log_resp)
    print("C pilot-seeded: logZcal=%.4f  neff=%.1f / 300" % (logZ_seeded, neff_seeded))
    print("agreement |dlogZ| = %.4f ;  efficiency gain x%.0f"
          % (abs(logZ_seeded - logZ_brute), (neff_seeded / 300) / (neff_brute / 20000)))

    assert abs(logZ_seeded - logZ_brute) < 0.1, "pilot-seeded Z disagrees with brute force"
    assert (neff_seeded / 300) > 20 * (neff_brute / 20000), "pilot did not improve efficiency"
    print("\nPASS: pilot-seeded (C) reproduces the brute-force reference (A) at far higher "
          "effective sampling.")
