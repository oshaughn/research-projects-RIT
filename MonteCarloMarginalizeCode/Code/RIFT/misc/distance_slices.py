"""Plan-B distance-slice export for ILE.

After ILE finishes its normal extrinsic integration at one intrinsic point,
this module produces K independent fixed-distance integrals: for each slice
d_k, an estimate of

    L_pure(d_k) = integral L(d_k, Omega) pi_Omega(Omega) dOmega

i.e. the extrinsic-marginalized likelihood at distance d_k, with the
distance prior divided out.  These can be re-marginalized against any prior
downstream and (because they are independent integrals, not bin
re-weightings of one shared sampler state) the *shape* L_pure(d) is honest
at the n_eff RIFT routinely uses.

Two estimators are provided:

* ``importance_reweight_slices`` -- reuses the Omega samples from the main
  integration and re-evaluates ``like_to_integrate`` at each slice distance.
  Cost: K * N likelihood evaluations using the already-precomputed
  rholms_intp / cross_terms (cheap).  Best when the Omega posterior is
  close to d-independent (typical: sky/inclination weakly couple to d
  except via the overall amplitude).

* ``fresh_sample_slices`` -- builds a fresh, low-dim sampler over Omega
  only at fixed d_k.  More expensive but doesn't assume the main run's
  Omega proposal is good for d_k.  Intended primarily as a cross-check.

The output schema is one row per (intrinsic, d_slice) pair so the file is
the natural Plan-B analogue of ``.composite``.  Target size: <~ 10x the
original ``.composite`` (K=10 by default).
"""
import numpy as np


DISTANCE_SLICE_FIELDS = (
    "lnL",        # extrinsic-marginalized lnL at d=dist (pure likelihood,
                  # i.e. distance sampling prior divided out)
    "sigmaL",     # log-space uncertainty on the slice integral
    "neff",       # effective sample count contributing to the slice
    "ntotal",     # total samples consumed by the slice estimator
    "method",     # 0 = importance_reweight, 1 = fresh_sample
    "m1",
    "m2",
    "s1x",
    "s1y",
    "s1z",
    "s2x",
    "s2y",
    "s2z",
    "lambda1",
    "lambda2",
    "eccentricity",
    "meanPerAno",
    "eos_index",
    "dist",
    "ln_prior_d_sampling",  # log pi_d(d_k) under the ILE sampling prior,
                            # so default reconstruction reproduces log_res
)


METHOD_REWEIGHT = 0
METHOD_FRESH = 1


def _logsumexp(x):
    x = np.asarray(x, dtype=float)
    m = np.max(x)
    if not np.isfinite(m):
        return m
    return m + np.log(np.sum(np.exp(x - m)))


def quantile_slice_centers(distance_samples, ln_weights, n_slices):
    """Choose K slice centers as equi-probable quantiles of the posterior in d.

    Falls back to uniform-in-log-d if the posterior is degenerate (n_eff < 2).
    """
    distance_samples = np.asarray(distance_samples, float)
    ln_weights = np.asarray(ln_weights, float)
    finite = np.isfinite(ln_weights) & np.isfinite(distance_samples)
    d = distance_samples[finite]
    lw = ln_weights[finite]
    if len(d) == 0:
        raise ValueError("no finite samples to choose slice centers from")
    p = np.exp(lw - _logsumexp(lw))
    n_eff = 1.0 / np.sum(p**2)
    if n_eff < 2.0:
        # degenerate posterior; cover the sample range uniformly in log d
        d_lo, d_hi = float(d.min()), float(d.max())
        d_lo = max(d_lo, 1e-3)
        return np.exp(np.linspace(np.log(d_lo), np.log(d_hi), n_slices))
    order = np.argsort(d)
    d_sorted = d[order]
    cdf = np.cumsum(p[order])
    cdf /= cdf[-1]
    quant = (np.arange(n_slices) + 0.5) / n_slices
    return np.interp(quant, cdf, d_sorted)


def _ln_omega_iw_factor(rvs, ln_prior_d_at_samples, ln_proposal_d_at_samples):
    """log( pi_Omega(Omega_i) / q_Omega(Omega_i) ) per sample.

    Decomposes the stored joint weight ln(pi_joint/q_joint) into a distance
    piece and an Omega piece, returning the Omega piece.
    """
    # Pull joint prior / proposal ratio
    if "joint_prior" in rvs and "joint_s_prior" in rvs:
        jp = np.asarray(rvs["joint_prior"], float)
        jsp = np.asarray(rvs["joint_s_prior"], float)
        with np.errstate(divide="ignore"):
            ln_pi_over_q_joint = np.log(np.maximum(jp, np.finfo(float).tiny)) \
                                 - np.log(np.maximum(jsp, np.finfo(float).tiny))
    elif "log_joint_prior" in rvs and "log_joint_s_prior" in rvs:
        ln_pi_over_q_joint = np.asarray(rvs["log_joint_prior"], float) \
                             - np.asarray(rvs["log_joint_s_prior"], float)
    else:
        raise KeyError("sampler._rvs missing joint prior/proposal columns")
    return ln_pi_over_q_joint - (np.asarray(ln_prior_d_at_samples, float)
                                  - np.asarray(ln_proposal_d_at_samples, float))


def importance_reweight_slices(
    sampler, like_to_integrate, d_slices,
    ln_prior_d_at_samples, ln_proposal_d_at_samples,
    manual_overflow=0.0, return_lnL=True,
):
    """Importance-reweight existing Omega samples at K slice distances.

    Returns
    -------
    lnL_slices : (K,) array
        Extrinsic-marginalized lnL at each d_k (pure likelihood, with
        ``manual_overflow`` restored so the value is directly comparable
        to ILE's reported ``log_res``).
    sigmaL_slices : (K,) array
        Per-slice 1-sigma uncertainty in lnL (Monte Carlo standard error).
    neff_slices : (K,) array
        Effective sample count at each slice.
    ntotal : int
        Total samples consumed (same for all slices: it is N).
    """
    rvs = sampler._rvs
    if "distance" not in rvs:
        raise KeyError("sampler._rvs has no 'distance' samples")
    N = len(rvs["distance"])
    ln_omega_iw = _ln_omega_iw_factor(rvs, ln_prior_d_at_samples,
                                       ln_proposal_d_at_samples)

    # Identify the param signature of like_to_integrate
    arg_names = like_to_integrate.__code__.co_varnames[
        :like_to_integrate.__code__.co_argcount]

    # Build per-arg arrays from the sampler's stored samples; distance gets
    # broadcast per slice below.
    fixed_inputs = {}
    for a in arg_names:
        if a == "distance":
            continue
        if a not in rvs:
            raise KeyError("sampler._rvs missing required column {!r} for "
                            "slice reweighting".format(a))
        fixed_inputs[a] = np.asarray(rvs[a])

    K = len(d_slices)
    lnL_out = np.empty(K)
    sigmaL_out = np.empty(K)
    neff_out = np.empty(K)
    for k, d_k in enumerate(d_slices):
        like_inputs = []
        for a in arg_names:
            if a == "distance":
                like_inputs.append(np.full(N, float(d_k)))
            else:
                like_inputs.append(fixed_inputs[a])
        lnL_at = like_to_integrate(*like_inputs)
        lnL_at = np.asarray(lnL_at, dtype=np.float64)
        if not return_lnL:
            # function returned exp(lnL - overflow); take log
            with np.errstate(divide="ignore"):
                lnL_at = np.log(np.maximum(lnL_at, np.finfo(float).tiny))
        # ln L_k(Omega_i) was returned with manual_overflow subtracted; add
        # it back so the slice marginal matches log_res's overflow scaling.
        ln_terms = lnL_at + manual_overflow + ln_omega_iw
        lnL_marg = _logsumexp(ln_terms) - np.log(N)
        # Slice n_eff in the importance sample
        m = np.max(ln_terms)
        if not np.isfinite(m):
            neff_out[k] = 0.0
        else:
            w = np.exp(ln_terms - m)
            neff_out[k] = (w.sum())**2 / np.sum(w**2)
        # MC std error of the log of the mean: approx by w-std / w-mean
        # std(lnI) ~ sqrt(var(w)/mean(w)^2 / N)
        if np.isfinite(m) and neff_out[k] > 1:
            mean_w = np.mean(np.exp(ln_terms - m))
            var_w = np.var(np.exp(ln_terms - m))
            with np.errstate(invalid="ignore"):
                sigmaL_out[k] = np.sqrt(var_w / (N * max(mean_w, np.finfo(float).tiny)**2))
        else:
            sigmaL_out[k] = np.inf
        lnL_out[k] = lnL_marg

    return lnL_out, sigmaL_out, neff_out, N


def is_uninformative(lnL_core, threshold=1.0):
    """Detect a flat / non-informative distance profile from the core slices.

    If the spread in core slice lnL is below ``threshold`` nats, the run has
    essentially no distance information and there is nothing for wing
    integrations to learn -- skip them.
    """
    finite = np.isfinite(lnL_core)
    if not np.any(finite):
        return True
    if np.sum(finite) < 2:
        return True
    return (np.max(lnL_core[finite]) - np.min(lnL_core[finite])) < threshold


def pick_wing_centers(d_min, d_max, d_core, n_wing,
                       min_log_gap=0.05):
    """Place K_wing slice centers log-uniformly outside the core span.

    Half go below the core, half above (rounded so the lower half gets the
    extra when K_wing is odd).  Returns a sorted array of distances.
    """
    n_wing = int(n_wing)
    if n_wing <= 0:
        return np.array([])
    d_core = np.asarray(d_core, float)
    finite = np.isfinite(d_core) & (d_core > 0)
    d_core_lo = float(np.min(d_core[finite])) if np.any(finite) else d_min
    d_core_hi = float(np.max(d_core[finite])) if np.any(finite) else d_max

    n_low = (n_wing + 1) // 2
    n_high = n_wing - n_low
    wings = []
    if d_core_lo > d_min * np.exp(min_log_gap) and n_low > 0:
        wings.append(np.exp(np.linspace(np.log(d_min),
                                         np.log(d_core_lo),
                                         n_low + 2)[1:-1]))
    if d_max > d_core_hi * np.exp(min_log_gap) and n_high > 0:
        wings.append(np.exp(np.linspace(np.log(d_core_hi),
                                         np.log(d_max),
                                         n_high + 2)[1:-1]))
    if not wings:
        return np.array([])
    return np.sort(np.concatenate(wings))


def fresh_sample_slices(reference_sampler, like_to_integrate, d_slices,
                         n_max=20000, n_eff_target=30, n_chunk=2000,
                         return_lnL=True, verbose=False):
    """Independent Omega-only integration at each pinned distance d_k.

    Build a fresh AdaptiveVolume sampler for the Omega parameters by
    cloning the reference sampler's per-parameter (pdf, prior, bounds)
    config, then integrate the cached likelihood with distance pinned to
    each slice.  Cost per slice: up to ``n_max`` cached-likelihood
    evaluations (no waveform/PSD regeneration).

    Returns the same (lnL, sigmaL, neff, ntotal_array) tuple shape as
    ``importance_reweight_slices``.
    """
    from RIFT.integrators import mcsamplerAdaptiveVolume

    arg_names = like_to_integrate.__code__.co_varnames[
        :like_to_integrate.__code__.co_argcount]
    if "distance" not in arg_names:
        raise ValueError("like_to_integrate has no 'distance' arg; fresh "
                         "slice integration not applicable")
    omega_params = [a for a in arg_names if a != "distance"]
    missing = [p for p in omega_params
               if p not in reference_sampler.params_ordered]
    if missing:
        raise KeyError("reference_sampler missing Omega params {!r} needed "
                        "for fresh slice integration".format(missing))

    K = len(d_slices)
    lnL_out = np.full(K, -np.inf)
    sigmaL_out = np.full(K, np.inf)
    neff_out = np.zeros(K)
    ntotal_out = np.zeros(K, dtype=int)

    for k, d_k in enumerate(d_slices):
        sampler = mcsamplerAdaptiveVolume.MCSampler()
        for p in omega_params:
            sampler.add_parameter(
                p,
                pdf=reference_sampler.pdf[p],
                prior_pdf=reference_sampler.prior_pdf[p],
                left_limit=float(reference_sampler.llim[p]),
                right_limit=float(reference_sampler.rlim[p]),
                adaptive_sampling=True,
            )

        d_fixed = float(d_k)
        # Per-Omega-param bounds, used to clip values defensively against
        # boundary noise (e.g. np.random.uniform can return rlim - 1ULP which
        # makes downstream arccos(...) NaN).
        omega_bounds = {p: (float(reference_sampler.llim[p]),
                             float(reference_sampler.rlim[p]))
                         for p in omega_params}

        def like_at_pinned_d(**kw):
            # AV's integrate_log passes Omega params as kwargs by name.
            sample = next(iter(kw.values()))
            N_eval = len(sample)
            d_arr = np.full(N_eval, d_fixed)
            full = {}
            for p, arr in kw.items():
                lo, hi = omega_bounds.get(p, (-np.inf, np.inf))
                # nudge inward by a tiny epsilon relative to range, so arccos
                # and friends never see the exact boundary
                eps = 1e-12 * max(abs(hi - lo), 1.0)
                full[p] = np.clip(np.asarray(arr, float), lo + eps, hi - eps)
            full["distance"] = d_arr
            return like_to_integrate(*(full[a] for a in arg_names))

        try:
            res = sampler.integrate_log(
                like_at_pinned_d,
                *omega_params,
                nmax=int(n_max), neff=int(n_eff_target), n=int(n_chunk),
                tempering_exp=0.1, n_adapt=10,
                verbose=verbose,
            )
        except Exception as e:
            print("  fresh slice d={:.2f} failed: {!r}".format(d_k, e))
            continue
        # AV's integrate_log returns (log_int, log(rel_var) + 2*log_int,
        # eff_samp, dict). Convert to sigma_lnL ~ sqrt(rel_var).
        if isinstance(res, tuple):
            lnI = float(res[0])
            if len(res) > 1:
                log_abs_var = float(res[1])
                ln_rel_var = log_abs_var - 2.0 * lnI
                sigma = float(np.exp(0.5 * ln_rel_var)) if np.isfinite(ln_rel_var) else np.inf
            else:
                sigma = np.nan
            neff_val = float(res[2]) if len(res) > 2 else np.nan
        else:
            lnI = float(res)
            sigma = np.nan
            neff_val = np.nan
        # When return_lnL=True the cached like_to_integrate returned
        # lnL - manual_overflow, so integrate_log's lnI is log of the integral
        # of exp(lnL - overflow). We restore the overflow OUTSIDE this helper
        # (caller knows manual_avoid_overflow_logarithm).
        lnL_out[k] = lnI
        if not(np.isnan(sigma)):
            sigmaL_out[k] = sigma
        if not(np.isnan(neff_val)):
            neff_out[k] = neff_val
        ntotal_out[k] = int(getattr(sampler, "ntotal", 0))

    return lnL_out, sigmaL_out, neff_out, ntotal_out


def build_distance_slice_table(d_slices, lnL_slices, sigmaL_slices,
                                neff_slices, ntotal, method_code,
                                params, ln_prior_d_at_slices):
    """Assemble the K-row slice table for one intrinsic point."""
    d_slices = np.asarray(d_slices, float)
    K = len(d_slices)
    dtype = [(name, float) for name in DISTANCE_SLICE_FIELDS]
    table = np.zeros(K, dtype=dtype)
    table["lnL"] = lnL_slices
    table["sigmaL"] = sigmaL_slices
    table["neff"] = neff_slices
    table["ntotal"] = float(ntotal)
    table["method"] = float(method_code)
    table["dist"] = d_slices
    table["ln_prior_d_sampling"] = ln_prior_d_at_slices
    for name in DISTANCE_SLICE_FIELDS:
        if name in {"lnL", "sigmaL", "neff", "ntotal", "method", "dist",
                    "ln_prior_d_sampling"}:
            continue
        table[name] = float(params.get(name, 0.0))
    return table


def save_distance_slice_table(fname, table):
    header = " ".join(table.dtype.names)
    np.savetxt(fname, np.column_stack([table[n] for n in table.dtype.names]),
               header=header)


def load_distance_slice_table(fname):
    return np.genfromtxt(fname, names=True)


def reconstruct_marginal_lnL(table, ln_prior_d=None):
    """Re-marginalize the slice table over distance with the given prior.

    Default (``ln_prior_d=None``): use ``ln_prior_d_sampling`` stored in the
    table -- reproduces ILE's reported ``log_res`` up to MC noise.

    Pass a callable ``ln_prior_d(d)`` to integrate against a custom prior.

    Uses the trapezoid rule on the K slice points -- the slices were placed
    at equi-probable quantiles, so this gives a moderate-K decent integral.
    """
    order = np.argsort(table["dist"])
    d = table["dist"][order]
    lnL = table["lnL"][order]
    if ln_prior_d is None:
        ln_pi = table["ln_prior_d_sampling"][order]
    else:
        ln_pi = np.asarray(ln_prior_d(d), float)
    log_integrand = lnL + ln_pi
    # logsumexp trapezoid
    m = np.max(log_integrand)
    if not np.isfinite(m):
        return m
    integrand = np.exp(log_integrand - m)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return m + np.log(trap(integrand, d))
