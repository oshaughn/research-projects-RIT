"""
Adaptive calibration sampling (Phase 1).

Motivation
----------
In-loop calibration marginalization draws cal realizations from the PRIOR.  As SNR
grows the calibration parameters become measurable, so the cal posterior pulls away
from the prior and almost all prior draws land in low-likelihood cal regions -- the
effective number of cal samples collapses.  This module learns a unimodal Gaussian
PROPOSAL in cal spline-node space and uses importance weighting (w_c = prior/proposal)
so the marginalized result stays unbiased while the sampling efficiency recovers.

Tempering
---------
The per-realization responsibilities (log integral contributions) have a very large
dynamic range at high SNR -- a naive Gaussian fit would be dominated by a single
sample.  We fit with TEMPERED weights softmax(beta * log_resp), starting at small beta
(broad, many samples contribute) and ramping beta -> 1 as the proposal narrows onto the
cal posterior.  The importance weights used for the *marginalization itself* are always
the full (untempered) w_c = prior/proposal; tempering only shapes the proposal fit.

This module is backend-agnostic numpy and has no GPU/lal dependency for the learning
machinery itself (it consumes an `evaluate` callback that runs the actual likelihood).
The cal-factor construction reuses the spline convention in generate_realizations.
"""
from __future__ import division

import numpy as np
import scipy.interpolate
from scipy.special import logsumexp

from RIFT.calmarg import generate_realizations as _gr


# ---------------------------------------------------------------------------
# Prior / node bookkeeping
# ---------------------------------------------------------------------------
def envelope_node_prior(fname, fmin, fmax, n_nodes):
    """Per-node Gaussian prior (mean, sigma) for amplitude then phase nodes, and the
    log10 spline node frequencies.  Node vector layout: [amp_0..amp_{N-1}, ph_0..ph_{N-1}]."""
    log_f = np.linspace(np.log10(fmin), np.log10(fmax), n_nodes)
    dat_amp, dat_phase = _gr.retrieve_envelope_from_file(fname, frequency_array=10 ** log_f)
    mean = np.concatenate([dat_amp[:, 1], dat_phase[:, 1]])
    sigma = np.concatenate([dat_amp[:, 2], dat_phase[:, 2]])
    sigma = np.where(sigma > 0, sigma, 1.0)   # guard degenerate (delta) priors
    return mean, sigma, log_f


def log_prior(nodes, prior_mean, prior_sigma):
    """Independent-Gaussian prior log-pdf for each row of `nodes` (n_real, dim)."""
    z = (nodes - prior_mean) / prior_sigma
    return np.sum(-0.5 * z * z - np.log(prior_sigma * np.sqrt(2 * np.pi)), axis=1)


def _mvn_logpdf(nodes, mean, cov):
    dim = mean.shape[0]
    d = nodes - mean
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, d.T).T          # (n_real, dim)
    quad = np.sum(sol * sol, axis=1)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + logdet + dim * np.log(2 * np.pi))


# ---------------------------------------------------------------------------
# Cal-factor construction (spline; matches generate_realizations convention)
# ---------------------------------------------------------------------------
def nodes_to_cal_factors(amp_nodes, phase_nodes, log_f_nodes, T_segment, dT, fmin, fmax):
    """Build two-sided complex calibration factors (npts_seg, n_real) from per-node
    amplitude/phase values, on the lalsimutils FFT frequency packing.

    amp_nodes, phase_nodes : (n_real, n_nodes)
    """
    n_real = amp_nodes.shape[0]
    deltaF_seg = 1. / T_segment
    npts_seg = int(T_segment / dT)
    freq = deltaF_seg * np.array([npts_seg / 2 - k if k <= npts_seg / 2 else -k + npts_seg / 2
                                  for k in np.arange(npts_seg)])
    mask_in = (np.abs(freq) >= fmin) & (np.abs(freq) <= fmax)
    mask_plus = mask_in & (freq > 0)
    mask_minus = mask_in & (freq < 0)
    lf_pos = np.log10(freq[mask_plus])
    lf_neg = np.log10(-freq[mask_minus])

    out = np.ones((npts_seg, n_real), dtype=complex)
    for i in range(n_real):
        cs_a = scipy.interpolate.CubicSpline(log_f_nodes, amp_nodes[i])
        cs_p = scipy.interpolate.CubicSpline(log_f_nodes, phase_nodes[i])
        out[mask_plus, i] = cs_a(lf_pos) * np.exp(1j * cs_p(lf_pos))
        out[mask_minus, i] = cs_a(lf_neg) * np.exp(-1j * cs_p(lf_neg))
    return out


# ---------------------------------------------------------------------------
# Tempered proposal fit + diagnostics
# ---------------------------------------------------------------------------
def fit_proposal(nodes, log_resp, beta, cov_floor=1e-8, cov_inflate=1.0,
                 prior_sigma=None, shrink=None):
    """Tempered weighted-Gaussian fit.  Weights = softmax(beta * log_resp).

    beta in (0,1]: small -> broad (many samples), 1 -> full responsibility weighting.

    prior_sigma : if given (length-dim 1-sigma of the diagonal prior), SHRINK the fitted
        covariance toward diag(prior_sigma**2).  This is essential when the fit is starved
        -- a weighted sample covariance from ~neff effective points cannot constrain the
        dim*(dim+1)/2 entries of a dim-dimensional covariance (cal node space is ~60-D),
        so the UNINFORMED directions otherwise collapse to ~0 variance.  A near-zero
        proposal variance is a near-delta: seeded draws are pinned and the importance
        weights log(prior/proposal) blow up, producing the pathological seeded likelihoods
        we saw.  Shrinking keeps uninformed directions at ~prior width (log_w ~ 0 there).
    shrink : explicit shrinkage weight rho in [0,1] toward the prior; default auto =
        (dim+1)/(dim+1+neff), i.e. ~1 (all prior) when starved, ->0 (all data) when
        neff >> dim.

    Returns (mean, cov)."""
    lw = beta * log_resp
    lw = lw - logsumexp(lw)
    w = np.exp(lw)
    mean = w @ nodes
    d = nodes - mean
    cov = (w[:, None] * d).T @ d
    dim = mean.shape[0]
    if prior_sigma is not None:
        prior_sigma = np.asarray(prior_sigma, dtype=float)
        neff = neff_from_logweights(beta * log_resp)
        rho = shrink if shrink is not None else (dim + 1.0) / (dim + 1.0 + neff)
        rho = float(min(max(rho, 0.0), 1.0))
        cov = (1.0 - rho) * cov_inflate * cov + rho * np.diag(prior_sigma ** 2)
    else:
        cov = cov_inflate * cov
    cov = cov + cov_floor * np.eye(dim)
    return mean, cov


def neff_from_logweights(log_w):
    """Kish effective sample size from log-weights: (sum w)^2 / sum w^2."""
    return float(np.exp(2 * logsumexp(log_w) - logsumexp(2 * log_w)))


def cal_mc_error_from_components(comp, cal_log_weights=None, sample_log_weights=None):
    """Calibration Monte-Carlo error budget for the cal-marginalized evidence.

    The in-loop marginalization estimates Z = E_c[ w_c Z_c ] over n_cal iid cal
    draws, where Z_c = int dtheta p(theta) L(theta, c).  The extrinsic sampler's
    reported variance CANNOT see the spread over c (the draw set is held fixed for
    the whole job), so this term must be estimated separately and added in
    quadrature to the extrinsic sampling error.

    comp : (n_samples, n_cal) RAW per-realization time-integrated lnL at a batch of
        extrinsic samples (``return_cal_components=True`` output).
    cal_log_weights : (n_cal,) importance log-weights log(prior/proposal);
        None = prior draws (uniform).
    sample_log_weights : (n_samples,) posterior log-weights of the extrinsic batch.
        For a batch drawn from the extrinsic PRIOR pass None: the marginal lnL of
        each sample (logsumexp_c of comp+cal_log_weights) is then the correct
        importance weight.

    Returns (sigma_lnZ_cal, neff_cal, a_c):
      a_c       : (n_cal,) normalized posterior contribution of realization c,
                  a_c = w_c Z_c / (n_cal Z); sums to 1.
      sigma_lnZ_cal : delta-method standard error of lnZ from the cal MC average,
                  Var(lnZ) ~= n_cal * Var_c(a_c).   (Lognormal cross-check: this
                  reproduces (exp(sigma_lnL^2)-1)/n_cal.)
      neff_cal  : Kish size 1 / sum_c a_c^2.  When neff_cal is O(1) the
                  marginalization is dominated by a single draw and the error
                  estimate itself is a LOWER BOUND -- treat the point as unreliable.
    """
    comp = np.atleast_2d(np.asarray(comp, dtype=float))
    n_samples, n_cal = comp.shape
    logw = np.zeros(n_cal) if cal_log_weights is None else np.asarray(cal_log_weights, dtype=float)
    lc = comp + logw[None, :]                       # log( w_c L_jc )
    lnL_marg = logsumexp(lc, axis=1)                # per-sample log sum_c w_c L_jc (norm cancels)
    log_r = lc - lnL_marg[:, None]                  # responsibilities r_jc, sum_c r_jc = 1
    if sample_log_weights is None:
        slw = lnL_marg                              # prior-drawn batch -> weight by marginal L
    else:
        slw = np.asarray(sample_log_weights, dtype=float)
    slw = slw - logsumexp(slw)                      # sum_j W_j = 1
    log_a = logsumexp(slw[:, None] + log_r, axis=0) # a_c = sum_j W_j r_jc
    a_c = np.exp(log_a - logsumexp(log_a))          # exact renormalization
    var_lnZ = n_cal * np.var(a_c, ddof=1) if n_cal > 1 else 0.0
    neff_cal = 1.0 / np.sum(a_c ** 2)
    return float(np.sqrt(max(var_lnZ, 0.0))), float(neff_cal), a_c


# ---------------------------------------------------------------------------
# Adaptive loop
# ---------------------------------------------------------------------------
def adaptive_cal(evaluate, prior_mean, prior_sigma, n_nodes_amp,
                 n_real=200, n_iter=4, betas=None, rng=None, return_history=False):
    """Run the adaptive cal-sampling loop.

    evaluate(nodes) -> log_L : callback returning, for each realization (row of
        `nodes`), the extrinsic-marginalized log-likelihood  log integral_theta
        L(theta, cal(nodes_c))  -- NO prior, NO importance weight (the loop folds those
        in).  In practice `evaluate` builds the cal factors (nodes_to_cal_factors) and
        runs the ILE integral per realization.

    The per-realization posterior responsibility (used to fit the next proposal and to
    measure efficiency) is  log_w + log_L = log( prior(c) * integral L / proposal(c) ),
    i.e. posterior/proposal; neff of these -> n_real exactly when the proposal matches
    the cal posterior.  The final `log_w` are the importance weights for the
    marginalization itself ( Z_cal = sum_c exp(log_w_c) integral L_c ).

    Returns dict with the final realizations' `nodes`, `log_w` (prior/proposal, for the
    marginalization), `proposal` (mean,cov), and per-iteration `neff` history.
    """
    rng = rng or np.random.default_rng()
    dim = prior_mean.shape[0]
    if betas is None:
        # ramp tempering 0.3 -> 1.0
        betas = np.linspace(0.3, 1.0, n_iter)
    mean = prior_mean.copy()
    cov = np.diag(prior_sigma ** 2)

    history = []
    nodes = log_w = None
    for it in range(n_iter):
        nodes = rng.multivariate_normal(mean, cov, size=n_real)        # (n_real, dim)
        log_q = _mvn_logpdf(nodes, mean, cov)
        log_p = log_prior(nodes, prior_mean, prior_sigma)
        log_w = log_p - log_q                                           # importance weights
        log_L = np.asarray(evaluate(nodes))                            # extrinsic-marg log-like
        log_resp = log_w + log_L                                       # posterior/proposal
        # next proposal from tempered posterior responsibilities; inflate the covariance
        # early (while tempering is on) to keep exploring, relax as beta -> 1.
        beta = float(betas[min(it, len(betas) - 1)])
        mean, cov = fit_proposal(nodes, log_resp, beta, cov_inflate=1.0 + (1.0 - beta),
                                 prior_sigma=prior_sigma)
        neff_resp = neff_from_logweights(log_resp)
        neff_w = neff_from_logweights(log_w)
        history.append(dict(iter=it, beta=beta, neff_resp=neff_resp, neff_w=neff_w))

    out = dict(nodes=nodes, log_w=log_w, proposal_mean=mean, proposal_cov=cov,
               history=history)
    if return_history:
        out['history'] = history
    return out


# ---------------------------------------------------------------------------
# Self-contained convergence demo (mock likelihood): no GPU/lal needed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A "true" calibration sits ~3 sigma off the prior mean in node space, with a
    # narrow likelihood (high SNR -> measurable cal).  Prior-only sampling would have
    # tiny neff; the adaptive loop should lock onto it and neff should climb.
    rng = np.random.default_rng(1234)
    dim = 8
    prior_mean = np.zeros(dim)
    prior_sigma = np.ones(dim)
    # measurable cal ~2 sigma off the prior, narrow likelihood (high SNR)
    true_node = prior_mean + 2.0 * prior_sigma * rng.standard_normal(dim) / np.sqrt(dim)
    like_sigma = 0.4

    def evaluate(nodes):
        # extrinsic-marginalized log-like proxy (no prior, no weights -- the loop adds them)
        z = (nodes - true_node) / like_sigma
        return -0.5 * np.sum(z * z, axis=1)

    # analytic cal posterior (Gaussian prior x Gaussian like): mean pulled from `true`
    # toward the prior mean; this is the target the proposal should converge to.
    w_like = 1.0 / like_sigma ** 2
    w_prior = 1.0 / prior_sigma ** 2
    post_mean = (true_node * w_like + prior_mean * w_prior) / (w_like + w_prior)

    # prior-only baseline: neff of the posterior responsibilities prior*L/prior = L
    base = rng.multivariate_normal(prior_mean, np.diag(prior_sigma ** 2), size=300)
    base_neff = neff_from_logweights(evaluate(base))
    err0 = float(np.max(np.abs(post_mean - prior_mean)))
    print("prior-only  neff_resp = %.1f / 300   (posterior is %.2f sigma off the prior mean)"
          % (base_neff, err0))

    res = adaptive_cal(evaluate, prior_mean, prior_sigma, n_nodes_amp=dim // 2,
                       n_real=300, n_iter=6, rng=rng)
    for h in res['history']:
        print("iter %d  beta=%.2f  neff_resp=%6.1f  neff_w=%6.1f" % (
            h['iter'], h['beta'], h['neff_resp'], h['neff_w']))
    err = float(np.max(np.abs(res['proposal_mean'] - post_mean)))
    print("proposal mean vs cal posterior: max|delta| = %.3f sigma" % err)
    assert res['history'][-1]['neff_resp'] > 10 * base_neff, \
        "adaptive did not improve effective cal sample size"
    assert err < 0.3, "proposal did not converge onto the cal posterior"
    print("\nPASS: tempered adaptive cal sampling converges onto the cal posterior "
          "and recovers effective samples.")
