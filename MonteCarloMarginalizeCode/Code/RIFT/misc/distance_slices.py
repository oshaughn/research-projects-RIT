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


def _to_cpu(x):
    """Host (numpy) view of x, converting cupy arrays if needed.

    On GPU ILE runs the sampler stores CUPY arrays in ``_rvs`` and the cached
    likelihood returns cupy; numpy refuses implicit conversion of those
    ("Implicit conversion to a NumPy array is not allowed. Please use .get()").
    This collapses cupy -> numpy at the boundary. Plain numpy arrays / python
    scalars pass straight through (cupy is not even imported here).
    """
    get = getattr(x, "get", None)
    if get is not None and type(x).__module__.split(".")[0] == "cupy":
        return get()
    return x


def _array_module(x):
    """Array module that owns ``x``: cupy for device arrays, numpy otherwise.

    Same duck typing as ``_to_cpu``, so cupy stays an optional import -- only a
    genuine cupy array triggers it.  Lets a helper do arithmetic on whatever
    backend it was handed instead of forcing a host round trip.
    """
    if type(x).__module__.split(".")[0] == "cupy":
        import cupy
        return cupy
    return np


#: AV block size at which ``nsel = min(1000, int(0.1*n_chunk))`` saturates at its
#: intended 1000.  This is a saturation point, not a tuned value -- see
#: :func:`resolve_slice_chunk` for why going below it is not merely noisy.  It also
#: happens to be the ILE driver's ``--n-chunk`` default, so on a default production
#: run the slice path and the main extrinsic loop run at the same block size.
AV_NSEL_SATURATION_CHUNK = 10000


def resolve_slice_chunk(explicit_chunk, ile_n_chunk, verbose=True):
    """AV block size to use for the fresh per-slice integrations.

    The slice path has no business inventing its own block size: by default it
    inherits the ILE driver's ``--n-chunk``, i.e. whatever the main extrinsic loop
    is already running at (default 10000).  ``explicit_chunk`` is the optional
    ``--distance-slice-chunk`` override and is honored as given.

    The one guard: an inherited value below :data:`AV_NSEL_SATURATION_CHUNK` is
    raised to it, with a warning.  AV's live volume only ever CONTRACTS, and each
    cycle's likelihood threshold is estimated from ``nsel = min(1000,
    int(0.1*n_chunk))`` samples; below n_chunk=10000 that cap binds, so every
    threshold becomes a permanent support cut decided by fewer samples than
    intended.  The resulting error is irreversible, not merely noisy.  This matters
    concretely because ``helper_LDG_Events.py`` drops ``--n-chunk`` to 500 on the
    input-skymap path -- correct for the main loop, catastrophic for an Omega-only
    slice integral, which *is* the hard sky dimension.  An explicit
    ``--distance-slice-chunk`` is a deliberate choice and is never clamped.

    Measured on S240615dg (within-run duplicate pairs, 4 intrinsic points x 2 seeds)
    and reproduced on analytic targets with known lnZ:

        n_chunk   rms(dlnL)   rms/mad   sigma understatement   cost
          2000      3.575       5.20          38.8x           1.00x
         10000      0.350       1.14           4.0x           1.09x
         15000      0.285       1.02           3.2x           1.24x

    rms/mad 5.20 -> 1.14 is the load-bearing number: at 2000 the per-slice error is
    a MIXTURE (a heavy tail on top of a core) and at 10000 it is not.  The same
    change eliminates "cap-burner" slices -- those that exhaust ``n_max`` without
    reaching the n_eff target -- which at 2000 were 3.1% of slices consuming 40.9%
    of all likelihood evaluations.  Reclaiming that waste is why the net cost is
    only +9%.

    Larger is NOT uniformly better, in two independent ways, so both are warned
    about rather than silently accepted:

    * accuracy: on hard targets 40000 was measurably worse than 10000-15000
      (coverage 0.17 vs 0.95 at KL 6.94).  10000-15000 is an optimum, not a floor.
    * host RAM on a CPU export: with no device to hold them, every block's arrays
      are in *system* memory, so per-block RSS scales with n_chunk.  Going
      2000 -> 10000 took a 760-job CPU export pilot from ~1 GB typical to spikes
      past 8 GB, holding ~12% of jobs on cgroup limits.  Budget >=8 GB
      request_memory for CPU slice exports at 10000, and more above it.  Note that
      MemoryUsage under-reports on cgroup OOM kills, so a job that dies this way
      will not obviously look memory-bound.

      GPU jobs are much less exposed.  ``like_at_pinned_d`` clips and pins on the
      sampler's own backend, so on a GPU run the Omega block never leaves the
      device and only the lnL vector comes back through ``_to_cpu``; the per-block
      host arrays that used to scale with n_chunk are gone.  4 GB sufficed in the
      campaign that measured this.  What still grows with n_chunk on a GPU run is
      device memory, not host.
    """
    if explicit_chunk is not None:
        n_chunk = int(explicit_chunk)
        if n_chunk < 1:
            raise ValueError(
                "--distance-slice-chunk must be >= 1, got {}. A nonpositive block"
                " size makes AV draw an empty or negative-sized batch, which raises"
                " inside the per-slice try/except and silently turns EVERY slice"
                " into a -inf row instead of failing the run.".format(n_chunk))
        if n_chunk < AV_NSEL_SATURATION_CHUNK and verbose:
            print("    : WARNING --distance-slice-chunk {} is below the AV nsel"
                  " saturation point {}; per-slice thresholds will be permanent"
                  " support cuts decided by only {} samples. Honoring it as"
                  " explicitly requested.".format(
                      n_chunk, AV_NSEL_SATURATION_CHUNK, int(0.1 * n_chunk)))
    else:
        n_chunk = int(ile_n_chunk)
        if n_chunk < AV_NSEL_SATURATION_CHUNK:
            if verbose:
                print("    : NOTE inherited --n-chunk {} is below the AV nsel"
                      " saturation point; raising the slice block size to {}."
                      " Pass --distance-slice-chunk to override.".format(
                          n_chunk, AV_NSEL_SATURATION_CHUNK))
            n_chunk = AV_NSEL_SATURATION_CHUNK
    if n_chunk > 15000 and verbose:
        print("    : WARNING slice block size {} exceeds the measured 10000-15000"
              " optimum (40000 was WORSE: coverage 0.17 vs 0.95 at KL 6.94). CPU"
              " exports run these integrations entirely host-side, so raise"
              " request_memory there (>8 GB at 10000); GPU runs keep the Omega"
              " block on the device.".format(n_chunk))
    return n_chunk


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


def quantile_slice_centers(distance_samples, ln_weights, n_slices,
                           randomize=False, rng=None):
    """Choose K slice centers as quantiles of the posterior in d.

    By default the K centers are the equi-probable quantiles (k+0.5)/K, which
    are identical for every intrinsic point -- so a single slice per intrinsic
    (n_slices=1) always lands on the median d. With ``randomize=True`` the K
    quantiles are instead drawn at random (uniform in CDF) on each call, so one
    slice per intrinsic becomes a fair-draw of d from THAT intrinsic's
    posterior; over the intrinsic grid this samples (intrinsic, d) jointly --
    cheap dense coverage for a continuous AD surrogate -- instead of pinning
    every point to the same quantile. ``rng`` (a numpy Generator/RandomState)
    makes the draw reproducible; without it np.random is used (fresh per
    process, so per-intrinsic draws differ across the batch).

    Falls back to uniform-in-log-d if the posterior is degenerate (n_eff < 2).
    """
    draw = rng.uniform if rng is not None else np.random.uniform
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
        # degenerate posterior; cover the sample range in log d
        d_lo, d_hi = float(d.min()), float(d.max())
        d_lo = max(d_lo, 1e-3)
        if randomize:
            u = np.sort(draw(0.0, 1.0, n_slices))
            return np.exp(np.log(d_lo) + u * (np.log(d_hi) - np.log(d_lo)))
        return np.exp(np.linspace(np.log(d_lo), np.log(d_hi), n_slices))
    order = np.argsort(d)
    d_sorted = d[order]
    cdf = np.cumsum(p[order])
    cdf /= cdf[-1]
    if randomize:
        quant = np.sort(draw(0.0, 1.0, n_slices))
    else:
        quant = (np.arange(n_slices) + 0.5) / n_slices
    return np.interp(quant, cdf, d_sorted)


def _ln_omega_iw_factor(rvs, ln_prior_d_at_samples, ln_proposal_d_at_samples):
    """log( pi_Omega(Omega_i) / q_Omega(Omega_i) ) per sample.

    Decomposes the stored joint weight ln(pi_joint/q_joint) into a distance
    piece and an Omega piece, returning the Omega piece.
    """
    # Pull joint prior / proposal ratio
    if "joint_prior" in rvs and "joint_s_prior" in rvs:
        jp = np.asarray(_to_cpu(rvs["joint_prior"]), float)
        jsp = np.asarray(_to_cpu(rvs["joint_s_prior"]), float)
        with np.errstate(divide="ignore"):
            ln_pi_over_q_joint = np.log(np.maximum(jp, np.finfo(float).tiny)) \
                                 - np.log(np.maximum(jsp, np.finfo(float).tiny))
    elif "log_joint_prior" in rvs and "log_joint_s_prior" in rvs:
        ln_pi_over_q_joint = np.asarray(_to_cpu(rvs["log_joint_prior"]), float) \
                             - np.asarray(_to_cpu(rvs["log_joint_s_prior"]), float)
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
        fixed_inputs[a] = np.asarray(_to_cpu(rvs[a]))

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
        lnL_at = np.asarray(_to_cpu(lnL_at), dtype=np.float64)
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
    """Detect a non-detectable event from the core slices via an absolute lnL.

    In RIFT's framing lnL is a likelihood ratio relative to the noise
    hypothesis, so it carries an absolute scale.  If the *peak* lnL across the
    core slices does not exceed ``threshold`` nats, the event is effectively
    undetected -- its distance posterior carries no information worth probing,
    so wing integrations are wasted compute and we skip them.

    This intentionally does NOT key off the spread ``max - min``: a high-SNR
    event with a flat distance profile (e.g. well-constrained inclination but
    unconstrained distance) has a small spread yet a large peak lnL, and *does*
    deserve wings.  A relative-spread test would wrongly skip it.
    """
    finite = np.isfinite(lnL_core)
    if not np.any(finite):
        return True
    return np.max(lnL_core[finite]) < threshold


def _log_uniform_wings(d_min, d_max, d_core_lo, d_core_hi, n_wing, min_log_gap):
    """Log-uniform wing placement across the full spans outside the core.

    Half below the core, half above (lower half gets the extra when n_wing is
    odd).  This is the likelihood-shape-agnostic fallback used whenever the
    parabolic fit is degenerate.
    """
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


def fit_lnL_parabola_in_inv_d(d_core, lnL_core):
    """Fit lnL_core to a quadratic in u = 1/dist.

    Near the peak the extrinsic-marginalized lnL is well modeled by

        lnL(d) ~= lnL_peak - 0.5 * A^2 * (1/d - 1/d_peak)^2

    which is a downward parabola in u = 1/d.  Returns ``(a, b, c)`` from
    ``lnL ~= a u^2 + b u + c`` (so ``A^2 = -2 a`` and the vertex sits at
    ``u_peak = -b/2a``), or ``None`` if the fit is degenerate (fewer than 3
    distinct finite core points, no lnL variation, or a non-downward fit).
    """
    d_core = np.asarray(d_core, float)
    lnL_core = np.asarray(lnL_core, float)
    finite = np.isfinite(d_core) & (d_core > 0) & np.isfinite(lnL_core)
    if np.sum(finite) < 3:
        return None
    u = 1.0 / d_core[finite]
    y = lnL_core[finite]
    if np.ptp(u) <= 0 or np.ptp(y) <= 0:
        return None
    try:
        a, b, c = np.polyfit(u, y, 2)
    except Exception:
        return None
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)) or a >= 0:
        return None
    return float(a), float(b), float(c)


def _parabolic_wing_bounds(d_core, lnL_core, lnL_peak, delta_lnL_target,
                           d_min, d_max):
    """Boundary distances where the lnL parabola drops ``delta_lnL_target``.

    Solves the fitted ``lnL(u) = a u^2 + b u + c`` (u = 1/dist) for the two
    u where lnL equals ``(lnL_peak or fitted vertex) - delta_lnL_target``,
    then maps back to distance and clamps to ``[d_min, d_max]``.

    Returns ``(d_small_bound, d_large_bound)`` or ``None`` if the fit is
    degenerate (caller falls back to log-uniform).
    """
    fit = fit_lnL_parabola_in_inv_d(d_core, lnL_core)
    if fit is None:
        return None
    a, b, c = fit
    vertex_u = -b / (2.0 * a)
    vertex_val = c - b * b / (4.0 * a)
    target = (vertex_val if lnL_peak is None else float(lnL_peak)) \
        - float(delta_lnL_target)
    disc = b * b - 4.0 * a * (c - target)
    if disc > 0:
        sq = np.sqrt(disc)
        r1 = (-b - sq) / (2.0 * a)
        r2 = (-b + sq) / (2.0 * a)
        u_lo, u_hi = min(r1, r2), max(r1, r2)
    else:
        # target above the fitted vertex (observed peak exceeds the fit, or
        # delta too small): fall back to the vertex-symmetric half-width,
        # which always yields real roots for a downward parabola.
        half_width = np.sqrt(-float(delta_lnL_target) / a)
        u_lo, u_hi = vertex_u - half_width, vertex_u + half_width
    # u_lo  -> larger distance boundary; u_hi -> smaller distance boundary.
    d_large = 1.0 / u_lo if u_lo > 0 else d_max
    d_small = 1.0 / u_hi if u_hi > 0 else d_min
    d_small = float(np.clip(d_small, d_min, d_max))
    d_large = float(np.clip(d_large, d_min, d_max))
    if not (d_large > d_small):
        return None
    return d_small, d_large


def pick_wing_centers(d_min, d_max, d_core, n_wing,
                       lnL_core=None, lnL_peak=None, delta_lnL_target=7.0,
                       min_log_gap=0.05):
    """Place K_wing slice centers outside the core span.

    When ``lnL_core`` is supplied and a quadratic fit of lnL vs 1/dist is
    non-degenerate, the wing span on each side is bounded by the parabolic
    model: wings extend from the core edge out to where lnL drops
    ``delta_lnL_target`` nats below the peak (default 7, i.e. prior weight
    < ~10^-3 outside).  This concentrates wing compute where the likelihood
    actually has support instead of spreading it across the whole prior range.

    Falls back to ``_log_uniform_wings`` (likelihood-agnostic, full-range)
    whenever the fit is degenerate or leaves no room outside the core.  Half
    the wings go below the core, half above (lower half gets the extra when
    n_wing is odd).  Returns a sorted array of distances.
    """
    n_wing = int(n_wing)
    if n_wing <= 0:
        return np.array([])
    d_core = np.asarray(d_core, float)
    finite = np.isfinite(d_core) & (d_core > 0)
    d_core_lo = float(np.min(d_core[finite])) if np.any(finite) else d_min
    d_core_hi = float(np.max(d_core[finite])) if np.any(finite) else d_max

    bounds = None
    if lnL_core is not None:
        bounds = _parabolic_wing_bounds(d_core, lnL_core, lnL_peak,
                                        delta_lnL_target, d_min, d_max)
    if bounds is None:
        return _log_uniform_wings(d_min, d_max, d_core_lo, d_core_hi,
                                  n_wing, min_log_gap)

    d_small_bound, d_large_bound = bounds
    n_low = (n_wing + 1) // 2
    n_high = n_wing - n_low
    wings = []
    if d_core_lo > d_small_bound * np.exp(min_log_gap) and n_low > 0:
        wings.append(np.exp(np.linspace(np.log(d_small_bound),
                                         np.log(d_core_lo),
                                         n_low + 2)[1:-1]))
    if d_large_bound > d_core_hi * np.exp(min_log_gap) and n_high > 0:
        wings.append(np.exp(np.linspace(np.log(d_core_hi),
                                         np.log(d_large_bound),
                                         n_high + 2)[1:-1]))
    if not wings:
        return _log_uniform_wings(d_min, d_max, d_core_lo, d_core_hi,
                                  n_wing, min_log_gap)
    return np.sort(np.concatenate(wings))


def walk_out_slices(d_slices, lnL_slices, ln_prior_d, run_fresh, d_min, d_max,
                    delta_lnL=7.0, max_steps=8, min_log_step=0.02, reasons=None):
    """Extend a slice set outward until the conditional falls ``delta_lnL`` below its peak.

    Quantile placement from the main loop's samples (``quantile_slice_centers``) inherits
    their spread; at the main loop's usual n_eff ~ 10 the outermost slices end where
    lp(d) = lnL(d) + ln pi(d) is typically only 1-3 nats below its peak, so the conditional
    is cut off where it still has mass.  This walks each edge outward with fresh slices, in
    steps that double in ln d (starting at twice the edge spacing), until the edge slice is
    ``delta_lnL`` below the running peak, the next slice would leave [d_min, d_max], a slice
    comes back non-finite, or ``max_steps`` slices were added on that side.  It measures
    rather than extrapolates: no shape is assumed beyond the last slice.

    ``ln_prior_d(d)`` returns ln pi(d) for an array of distances.  ``run_fresh(d_array)``
    returns (lnL, sigmaL, neff, ntotal) arrays for fresh slices at those distances, on the
    same lnL axis as ``lnL_slices``.  Returns the added slices as
    (d, lnL, sigmaL, neff, ntotal) arrays, unsorted (near-side slices first).  If
    ``reasons`` is a list, one stop reason per side is appended to it (near side first):
    'target', 'range', 'nonfinite' or 'max_steps'; 'off' for both sides when the walk-out
    does not run (delta_lnL or max_steps <= 0, or fewer than two finite slices).
    """
    d = np.asarray(d_slices, float)
    y = np.asarray(lnL_slices, float)
    ok = np.isfinite(d) & (d > 0) & np.isfinite(y)
    out = ([], [], [], [], [])
    if np.sum(ok) < 2 or delta_lnL <= 0 or max_steps <= 0:
        if reasons is not None:
            reasons.extend(["off", "off"])
        return tuple(np.array(v) for v in out)
    d, y = d[ok], y[ok]
    o = np.argsort(d)
    d, y = d[o], y[o]
    lp = y + np.asarray(ln_prior_d(d), float)
    peak = float(np.max(lp))
    for side in (-1, +1):
        why = "max_steps"
        i_edge, i_next = (0, 1) if side < 0 else (-1, -2)
        d_edge, lp_edge = float(d[i_edge]), float(lp[i_edge])
        step = 2.0 * max(abs(np.log(d[i_edge] / d[i_next])), min_log_step)
        for _ in range(int(max_steps)):
            if peak - lp_edge >= delta_lnL:
                why = "target"
                break
            d_new = d_edge * np.exp(side * step)
            if not (d_min <= d_new <= d_max):
                d_new = d_min if side < 0 else d_max
                if side * np.log(d_new / d_edge) < min_log_step:   # clamped step must still move outward
                    why = "range"
                    break
            lnL_k, sig_k, neff_k, nt_k = (np.asarray(v).ravel()[0] for v in run_fresh(np.array([d_new])))
            for lst, v in zip(out, (d_new, lnL_k, sig_k, neff_k, nt_k)):
                lst.append(v)
            if not np.isfinite(lnL_k):
                why = "nonfinite"
                break
            d_edge = d_new
            lp_edge = float(lnL_k + np.asarray(ln_prior_d(np.array([d_new])), float)[0])
            peak = max(peak, lp_edge)
            if d_new in (d_min, d_max):
                why = "target" if peak - lp_edge >= delta_lnL else "range"
                break
            step *= 2.0
        if reasons is not None:
            reasons.append(why)
    return (np.array(out[0], float), np.array(out[1], float), np.array(out[2], float),
            np.array(out[3], float), np.array(out[4], int))



def walk_out_fresh_slices(reference_sampler, like_to_integrate, d_slices, lnL_slices, ln_prior_d,
                          d_min, d_max, lnL_offset, delta_lnL, max_steps,
                          n_max, n_eff_target, n_chunk, return_lnL=True, reasons=None, runner=None):
    """``walk_out_slices`` with fresh fixed-d integrations as the step.

    ``runner(d_array)``, if given, replaces ``fresh_sample_slices`` (e.g. ``WarmSliceSampler.run``);
    it returns the same tuple, on the same offset-subtracted axis.

    ``lnL_slices`` are on ILE's lnL axis (the overflow offset restored); fresh_sample_slices
    returns the integral of the offset-subtracted likelihood, so ``lnL_offset``
    (ILE's ``manual_avoid_overflow_logarithm``) is added back to every walk-out slice before
    it is compared with the existing slices or returned.
    """
    def run_fresh(d_new):
        if runner is not None:
            l_raw, s_k, n_k, t_k = runner(d_new)
        else:
            l_raw, s_k, n_k, t_k = fresh_sample_slices(
                reference_sampler, like_to_integrate, d_new,
                n_max=n_max, n_eff_target=n_eff_target, n_chunk=n_chunk, return_lnL=return_lnL)
        return np.asarray(l_raw, float) + float(lnL_offset), s_k, n_k, t_k
    return walk_out_slices(d_slices, lnL_slices, ln_prior_d, run_fresh, d_min, d_max,
                           delta_lnL=delta_lnL, max_steps=max_steps, reasons=reasons)


def refine_slice_gaps(d_slices, lnL_slices, ln_prior_d, run_fresh, dlnL_max, delta_lnL, max_add=12,
                      min_log_step=0.005, reasons=None):
    """Bisect slice gaps the conditional crosses too fast to interpolate.

    The walk-out's steps double in ln d, so its last step can jump straight over an edge: on
    S240615dg MR-1 grid point 6 the conditional lp(d) = lnL + ln pi is flat to 1.7 Gpc and the next
    walk-out slice sits at 2.36 Gpc, 15 nats down, while the deep draws have q95 1.86 Gpc.  Linear
    interpolation of lnL in 1/d across that gap moves the cliff inward and drops real mass
    (checkpoints/2026-10-25-truth in RIFT_roboto_paper).

    Repeatedly takes the neighbouring pair (in d) with the largest |lp_i - lp_{i+1}| among pairs
    where that jump exceeds ``dlnL_max`` and at least one slice is within ``delta_lnL`` of the
    running peak, and adds a slice at their geometric mean via ``run_fresh`` (same contract as in
    ``walk_out_slices``).  A pair whose ln d gap is below 2 * ``min_log_step`` is not bisected and is
    skipped; a non-finite new slice is kept in the output and its pair is not tried again.  Stops when
    no untried live pair remains or after ``max_add`` slices.  Returns the added rows (d, lnL,
    sigmaL, neff, ntotal); ``reasons`` (a list) gets one of 'max_add' (cap reached), 'min_step' (no
    pair left, but some live gap was too narrow to bisect), 'resolved', or 'off'.
    """
    d = np.asarray(d_slices, float)
    y = np.asarray(lnL_slices, float)
    out = ([], [], [], [], [])
    ok = np.isfinite(d) & (d > 0) & np.isfinite(y)
    if dlnL_max <= 0 or max_add <= 0 or np.sum(ok) < 2:
        if reasons is not None:
            reasons.append("off")
        return tuple(np.array(v) for v in out)
    D, Y = list(d[ok]), list(y[ok])
    tried = set()
    why = "resolved"
    narrow = False
    while True:
        o = np.argsort(D)
        dd, yy = np.asarray(D)[o], np.asarray(Y)[o]
        lp = yy + np.asarray(ln_prior_d(dd), float)
        peak = np.max(lp)
        jump = np.abs(np.diff(lp))
        live = (np.maximum(lp[1:], lp[:-1]) > peak - delta_lnL) & (jump > dlnL_max)
        cand = [i for i in np.flatnonzero(live) if (round(dd[i], 9), round(dd[i + 1], 9)) not in tried]
        if not cand:
            break
        if len(out[0]) >= max_add:
            why = "max_add"
            break
        i = max(cand, key=lambda k: jump[k])
        tried.add((round(dd[i], 9), round(dd[i + 1], 9)))
        if np.log(dd[i + 1] / dd[i]) < 2 * min_log_step:
            narrow = True
            continue
        d_new = float(np.sqrt(dd[i] * dd[i + 1]))
        lnL_k, sig_k, neff_k, nt_k = (np.asarray(v).ravel()[0] for v in run_fresh(np.array([d_new])))
        for lst, v in zip(out, (d_new, lnL_k, sig_k, neff_k, nt_k)):
            lst.append(v)
        if np.isfinite(lnL_k):
            D.append(d_new)
            Y.append(float(lnL_k))
    if why == "resolved" and narrow:
        why = "min_step"
    if reasons is not None:
        reasons.append(why)
    return (np.array(out[0], float), np.array(out[1], float), np.array(out[2], float),
            np.array(out[3], float), np.array(out[4], int))


def refine_fresh_slices(reference_sampler, like_to_integrate, d_slices, lnL_slices, ln_prior_d, lnL_offset,
                        dlnL_max, delta_lnL, max_add, n_max, n_eff_target, n_chunk, return_lnL=True,
                        reasons=None, runner=None):
    """``refine_slice_gaps`` with fixed-d integrations as the step (offset handled as in
    ``walk_out_fresh_slices``: ``lnL_slices`` are on ILE's axis, the runner's output is shifted by
    ``lnL_offset`` before comparison)."""
    def run_fresh(d_new):
        if runner is not None:
            l_raw, s_k, n_k, t_k = runner(d_new)
        else:
            l_raw, s_k, n_k, t_k = fresh_sample_slices(
                reference_sampler, like_to_integrate, d_new,
                n_max=n_max, n_eff_target=n_eff_target, n_chunk=n_chunk, return_lnL=return_lnL)
        return np.asarray(l_raw, float) + float(lnL_offset), s_k, n_k, t_k
    return refine_slice_gaps(d_slices, lnL_slices, ln_prior_d, run_fresh, dlnL_max, delta_lnL,
                             max_add=max_add, reasons=reasons)


def merge_walkout(d, lnL, sigmaL, neff, ntotal, added):
    """Append walk-out rows ``added = (d, lnL, sigmaL, neff, ntotal)`` to a slice set (ntotal int)."""
    base = (np.asarray(d, float), np.asarray(lnL, float), np.asarray(sigmaL, float),
            np.asarray(neff, float), np.asarray(ntotal, dtype=int))
    if len(added[0]) == 0:
        return base
    return tuple(np.concatenate([b, np.asarray(a, dtype=b.dtype)]) for b, a in zip(base, added))

def fresh_sample_slices(reference_sampler, like_to_integrate, d_slices,
                         n_max=20000, n_eff_target=30,
                         n_chunk=AV_NSEL_SATURATION_CHUNK,
                         return_lnL=True, verbose=False):
    """Independent Omega-only integration at each pinned distance d_k.

    Build a fresh AdaptiveVolume sampler for the Omega parameters by
    cloning the reference sampler's per-parameter (pdf, prior, bounds)
    config, then integrate the cached likelihood with distance pinned to
    each slice.  Cost per slice: up to ``n_max`` cached-likelihood
    evaluations (no waveform/PSD regeneration).

    Returns the same (lnL, sigmaL, neff, ntotal_array) tuple shape as
    ``importance_reweight_slices``.

    ``n_chunk`` is the AV block size.  ILE callers should pass
    ``resolve_slice_chunk(opts.distance_slice_chunk, opts.n_chunk)`` so the slice
    path runs at the driver's ordinary chunk size rather than a private number;
    the default here is the same value the driver defaults to.  It is not a free
    parameter -- read :func:`resolve_slice_chunk` before changing it, in either
    direction.

    ``n_max`` is a BLOCK-GRANULAR budget, not a hard cap.  AV tests
    ``ntotal < nmax`` *before* drawing a whole block, so the real ceiling is
    ``ceil(n_max/n_chunk)`` blocks -- up to nearly a full block over n_max, plus a
    small bin-rounding remainder.  Measured at n_max=20000: n_chunk=10000 -> 20045
    (1.00x), 15000 -> 30050 (1.50x), 40000 -> 40000 (2.00x).  This is pre-existing
    AV behavior, but it only becomes reachable once n_chunk is configurable, so a
    non-multiple pairing is warned about below.  Keep n_max a whole multiple of
    n_chunk if the per-slice cost matters to you.
    """
    from RIFT.integrators import mcsamplerAdaptiveVolume

    n_chunk = int(n_chunk)
    n_max = int(n_max)
    if n_chunk > n_max or (n_max % n_chunk):
        n_actual = int(np.ceil(n_max / float(n_chunk)) * n_chunk)
        print("    : NOTE n_max={} is not a whole multiple of the block size {};"
              " AV checks its budget BEFORE drawing a block, so each slice will"
              " actually draw up to {} samples ({:.2f}x the requested max)."
              .format(n_max, n_chunk, n_actual, n_actual / float(max(n_max, 1))))

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
            # AV's integrate_log passes Omega params as kwargs by name, in its
            # own native backend: cupy on a GPU run, numpy otherwise.  Stay on
            # that backend.  Clipping with numpy here would raise TypeError on a
            # cupy array; AV catches that and retries the whole block with a host
            # copy, so every block would cross PCIe twice (D2H for this helper,
            # then H2D inside the device-native likelihood) for nothing.
            sample = next(iter(kw.values()))
            xp = _array_module(sample)
            full = {}
            inside = None
            for p, arr in kw.items():
                lo, hi = omega_bounds.get(p, (-np.inf, np.inf))
                # nudge inward by a tiny epsilon relative to range, so arccos
                # and friends never see the exact boundary
                eps = 1e-12 * max(abs(hi - lo), 1.0)
                arr = xp.asarray(arr, dtype=float)
                ok = (arr >= lo - eps) & (arr <= hi + eps)
                inside = ok if inside is None else (inside & ok)
                full[p] = xp.clip(arr, lo + eps, hi - eps)
            full["distance"] = xp.full_like(sample, d_fixed, dtype=float)
            # like_to_integrate is the cached ILE likelihood -> returns CUPY on a
            # GPU run; the AV bookkeeping downstream is host-side, so bring the
            # lnL vector (and only that) back.  A draw outside the prior box is
            # outside the integral: lnL = -inf there, never the clipped edge value
            # (test_dslice_bounds.py).
            lnL = like_to_integrate(*(full[a] for a in arg_names))
            return _to_cpu(xp.where(inside, lnL, -xp.inf))

        try:
            res = sampler.integrate_log(
                like_at_pinned_d,
                *omega_params,
                nmax=int(n_max), neff=int(n_eff_target), n=int(n_chunk),
                # No n_adapt: AV accepts it but ignores it as an adaptation
                # schedule (see mcsamplerAdaptiveVolume.integrate_log). Its only
                # residual effect is gating save_intg, which tempering_exp>0
                # already turns on, so passing it only implied a knob that
                # does not exist.
                tempering_exp=0.1,
                # Integer bin counts, as ILE's main loop and CIP pass: otherwise edge
                # bins overhang the box and lnZ includes the overhang
                # (test_dslice_bounds.py).
                enforce_bounds=True,
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


# ---------------------------------------------------------------------------------------------
# Warm-started slices.  Every slice integrates the same extrinsic space at a nearby distance, and
# the main loop already sampled that space; a fresh AV per slice re-learns it ~K times.  Here each
# slice is an adaptive importance sampler (population Monte Carlo) whose first proposal is a kernel
# density fitted to weighted samples from the main loop or from the nearest finished slice, mixed
# with a uniform component over the whole box.  The uniform component keeps q > 0 wherever the
# integrand is, so a displaced seed costs samples, not bias.
# ---------------------------------------------------------------------------------------------

class _RunningNeff:
    """Kish n_eff of log weights added block by block: running sums, rescaled when the maximum moves."""

    def __init__(self):
        self.m, self.s1, self.s2 = -np.inf, 0.0, 0.0

    def add(self, lw):
        lw = np.asarray(lw, float)
        lw = lw[np.isfinite(lw)]
        if not len(lw):
            return
        m = max(self.m, float(lw.max()))
        if self.m > -np.inf and m > self.m:
            self.s1 *= np.exp(self.m - m)
            self.s2 *= np.exp(2 * (self.m - m))
        e = np.exp(lw - m)
        self.m, self.s1, self.s2 = m, self.s1 + e.sum(), self.s2 + (e * e).sum()

    @property
    def neff(self):
        return self.s1 * self.s1 / self.s2 if self.s2 > 0 else 0.0


def _neff(ln_w):
    ln_w = np.asarray(ln_w, float)
    m = np.max(ln_w) if len(ln_w) else -np.inf
    if not np.isfinite(m):
        return 0.0
    w = np.exp(ln_w - m)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def fit_kde_proposal(x, ln_w, lo, hi, n_kernel=1000, rng=None, min_frac=1e-3, inflate=1.0, local_k=None,
                     local_shrink=0.1, n_pool=5000):
    """Gaussian KDE fitted to weighted points ``x`` (N, p): kernels at ``n_kernel`` points drawn by
    weight, one shared covariance = inflate x Scott factor^2 x weighted covariance, floored at
    (min_frac x box width)^2 per axis.  Returns a dict for ``kde_draw``/``kde_logpdf``, or None if
    no point has finite weight."""
    rng = np.random.default_rng() if rng is None else rng
    x = np.atleast_2d(np.asarray(x, float))
    ln_w = np.asarray(ln_w, float)
    ok = np.isfinite(ln_w) & np.all(np.isfinite(x), axis=1)
    if not np.any(ok):
        return None
    x, ln_w = x[ok], ln_w[ok]
    w = np.exp(ln_w - ln_w.max())
    w /= w.sum()
    p = x.shape[1]
    neff = 1.0 / np.sum(w ** 2)
    mu = w @ x
    cov = (x - mu).T @ ((x - mu) * w[:, None]) / max(1.0 - np.sum(w ** 2), 1e-12)
    floor = (min_frac * (np.asarray(hi, float) - np.asarray(lo, float))) ** 2
    h2 = neff ** (-2.0 / (p + 4))
    H = inflate * h2 * cov + np.diag(floor)
    L = np.linalg.cholesky(H)
    # Collapsed: the weight sits on fewer than p+2 distinct points, so the covariance has rank < p and
    # the kernels are a thin sheet in the other directions.  Draws stay correctly weighted, but on a 5-d
    # test ~5% of slices from such a seed passed every trust check while 1-3 nats low (#511 review).
    o = np.argsort(w)[::-1]
    top = o[: int(np.searchsorted(np.cumsum(w[o]), 1.0 - 1e-6)) + 1]
    collapsed = bool(len(np.unique(x[top], axis=0)) < p + 2)
    centers = x[rng.choice(len(x), size=int(n_kernel), p=w)]
    if local_k:
        out = _local_kernels(x, w, centers, cov, H, floor, inflate, int(local_k), local_shrink, n_pool, rng, neff)
    else:
        out = dict(centers=centers, L=L, Linv=np.linalg.inv(L),
                   ln_norm=-0.5 * p * np.log(2 * np.pi) - np.sum(np.log(np.diag(L))), neff=neff)
    out["collapsed"] = collapsed
    return out


def _local_kernels(x, w, centers, cov, H_global, floor, inflate, k, shrink, n_pool, rng, neff):
    """Per-kernel covariances: each kernel takes the covariance of its k nearest neighbours in a
    weight-proportional pool of the points (unique rows), so it lies along the ridge it sits on.  At
    fixed distance the O4 conditionals lock phi -/+ 2 psi on each inclination branch (S240514x,
    2026-10-22): two straight ridges wrapping the (phi, psi) torus, which one shared covariance
    spreads over the whole torus.  Shrunk toward the global kernel by ``shrink`` and floored."""
    p = x.shape[1]
    pool = np.unique(x[rng.choice(len(x), size=int(n_pool), p=w)], axis=0)   # distinct rows, by value
    if len(pool) < p + 2:
        # Too few distinct points for a p-dimensional local covariance (a seed collapsed onto a handful
        # of rows; S240615dg export, 2026-10-25: "kth(=6) out of bounds (5)").  Use the shared kernel.
        L = np.linalg.cholesky(H_global)
        return dict(centers=centers, L=L, Linv=np.linalg.inv(L),
                    ln_norm=-0.5 * p * np.log(2 * np.pi) - np.sum(np.log(np.diag(L))), neff=neff)
    k = min(max(k, p + 1), len(pool) - 1)
    Lg = np.linalg.cholesky(cov + np.diag(floor))
    zp = np.linalg.solve(Lg, pool.T).T
    zc = np.linalg.solve(Lg, centers.T).T
    idx = np.empty((len(centers), k), dtype=int)
    for i in range(0, len(centers), 256):           # (256 x n_pool) distance blocks
        d2 = (np.sum(zc[i:i + 256] ** 2, 1)[:, None] + np.sum(zp ** 2, 1)[None, :]
              - 2 * zc[i:i + 256] @ zp.T)
        idx[i:i + 256] = np.argpartition(d2, k, axis=1)[:, :k]
    nb = pool[idx]                                   # (M, k, p)
    dnb = nb - nb.mean(axis=1, keepdims=True)
    C = np.einsum("mki,mkj->mij", dnb, dnb) / (k - 1)
    H = (1 - shrink) * inflate * C + shrink * H_global + np.diag(floor)
    Ls = np.linalg.cholesky(H)
    ln_norm = -0.5 * p * np.log(2 * np.pi) - np.sum(np.log(np.diagonal(Ls, axis1=1, axis2=2)), axis=1)
    return dict(centers=centers, Ls=Ls, Linvs=np.linalg.inv(Ls), ln_norms=ln_norm, neff=neff)


def kde_draw(kde, n, rng):
    i = rng.integers(len(kde["centers"]), size=n)
    c = kde["centers"][i]
    z = rng.standard_normal((n, c.shape[1]))
    if "Ls" in kde:
        return c + np.einsum("nij,nj->ni", kde["Ls"][i], z)
    return c + z @ kde["L"].T


def kde_logpdf(kde, x, block=2048, xp=None):
    """log density of the KDE at x (host array in, host array out).  ``xp`` (e.g. cupy) does the
    N x n_kernel arithmetic on that backend; on a GPU run it was 75% of the slice wall time on the
    host (S240426s profile, 2026-10-22)."""
    xp = np if xp is None else xp
    if "Ls" in kde:
        return _local_kde_logpdf(kde, x, xp, block=max(64, block // 8))
    Linv = xp.asarray(kde["Linv"])
    zc = xp.asarray(kde["centers"]) @ Linv.T
    nc = xp.sum(zc ** 2, 1)[None, :]
    out = np.empty(len(x))
    for i in range(0, len(x), block):
        z = xp.asarray(x[i:i + block]) @ Linv.T
        a = -0.5 * (xp.sum(z ** 2, 1)[:, None] + nc - 2 * z @ zc.T)
        m = xp.max(a, axis=1)
        r = m + xp.log(xp.sum(xp.exp(a - m[:, None]), axis=1))
        out[i:i + block] = _to_cpu(r)
    return out - np.log(len(kde["centers"])) + kde["ln_norm"]


def _local_kde_logpdf(kde, x, xp, block=256):
    Linvs = xp.asarray(kde["Linvs"])
    c = xp.asarray(kde["centers"])
    lnn = xp.asarray(kde["ln_norms"])
    out = np.empty(len(x))
    for i in range(0, len(x), block):
        d = xp.asarray(x[i:i + block])[:, None, :] - c[None, :, :]          # (b, M, p)
        z = xp.einsum("mij,bmj->bmi", Linvs, d)
        a = lnn[None, :] - 0.5 * xp.sum(z * z, axis=2)
        m = xp.max(a, axis=1)
        out[i:i + block] = _to_cpu(m + xp.log(xp.sum(xp.exp(a - m[:, None]), axis=1)))
    return out - np.log(len(kde["centers"]))


def _block_estimate(ln_w, n):
    m = np.max(ln_w) if len(ln_w) else -np.inf
    if not np.isfinite(m):
        return -np.inf, np.inf
    w = np.exp(ln_w - m)
    s = w.sum() / n
    var = max(np.sum(w ** 2) / n - s ** 2, 0.0) / n
    return m + np.log(s), np.sqrt(var) / s


def _blocks_agree(lws, n_block, nsigma):
    """The last block alone and all earlier blocks pooled give the same lnZ within nsigma.

    A proposal still creeping toward the target's bulk gives a rising sequence of block estimates
    while the pooled n_eff already looks healthy; this catches it."""
    if len(lws) < 2:
        return False
    a, sa = _block_estimate(lws[-1], n_block)
    b, sb = _block_estimate(np.concatenate(lws[:-1]), n_block * (len(lws) - 1))
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= nsigma * np.hypot(sa, sb)


def _gpd_k(x):
    """Shape k of a generalized Pareto fitted to exceedances ``x`` > 0 (Zhang & Stephens 2009, with the
    weak prior toward k = 0.5 used by PSIS, Vehtari et al. 2024)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    m = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    bs = bs / (3.0 * x[max(int(n / 4.0 + 0.5) - 1, 0)]) + 1.0 / x[-1]
    ks = np.mean(np.log1p(-bs[:, None] * x[None, :]), axis=1)
    L = n * (np.log(-bs / ks) - ks - 1.0)
    w = 1.0 / np.sum(np.exp(L[None, :] - L[:, None]), axis=1)
    b = np.sum(bs * w) / np.sum(w)
    k = np.mean(np.log1p(-b * x))
    return float((n * k + 5.0) / (n + 10.0))


def pareto_khat(ln_w):
    """PSIS tail diagnostic of importance log weights: k-hat of a generalized Pareto fitted to the
    largest min(n/5, 3 sqrt n) weights.  k-hat > 0.7 means the proposal's tails are too light for
    the target -- the estimate is unreliable however its sigma looks -- which is how a proposal that
    sits in the target's tail shows up (Vehtari et al. 2024).  inf if too few finite weights."""
    lw = np.asarray(ln_w, float)
    lw = lw[np.isfinite(lw)]
    n = len(lw)
    M = int(min(np.ceil(0.2 * n), np.ceil(3 * np.sqrt(n))))
    if n < 10 or M < 5:
        return np.inf
    lw = np.sort(lw)
    w = np.exp(lw - lw[-1])
    tail = w[-M:] - w[-M - 1]
    if not np.all(tail[-1:] > 0) or np.sum(tail > 0) < 5:
        return np.inf
    return _gpd_k(tail[tail > 0])


def _weighted_moments(x, ln_w):
    ok = np.isfinite(ln_w)
    if np.sum(ok) < 2:
        return None, None, 0.0
    w = np.exp(ln_w[ok] - ln_w[ok].max())
    w /= w.sum()
    mu = w @ x[ok]
    cov = (x[ok] - mu).T @ ((x[ok] - mu) * w[:, None])
    return mu, cov, 1.0 / np.sum(w ** 2)


def _means_agree(xs, lws, nsigma, lo, hi):
    """The last block's weighted mean sits where the earlier blocks put the target's mean.

    Shift between the two weighted means, whitened by the pooled weighted covariance, against
    what n_eff alone would give: |dz|^2 <= nsigma^2 * p * (1/neff_last + 1/neff_prev).  A proposal
    creeping up the target's tail moves its weighted mean every block, even when consecutive lnZ
    estimates happen to agree."""
    if len(lws) < 2:
        return False
    xa, la = xs[-1], lws[-1]
    xb, lb = np.concatenate(xs[:-1]), np.concatenate(lws[:-1])
    ma, _, na = _weighted_moments(xa, la)
    mb, _, nb = _weighted_moments(xb, lb)
    _, cov, _ = _weighted_moments(np.concatenate(xs), np.concatenate(lws))
    if ma is None or mb is None or cov is None or na < 2 or nb < 2:
        return False
    cov = cov + np.diag((1e-6 * (np.asarray(hi) - np.asarray(lo))) ** 2)
    d = ma - mb
    chi2 = float(d @ np.linalg.solve(cov, d))
    return chi2 <= nsigma ** 2 * len(d) * (1.0 / na + 1.0 / nb)


def warm_integrate_slice(ln_f, lo, hi, seed_x, seed_ln_w, n_max, n_eff_target, n_block=AV_NSEL_SATURATION_CHUNK,
                         defensive=0.1, n_kernel=1000, rng=None, inflate=4.0, stable_nsigma=2.0,
                         include_decider=True, mean_nsigma=None, khat_max=np.inf, max_adapt_blocks=5,
                         broad=None, broad_frac=0.1, refit_temper=0.5, xp=None, max_blocks=50,
                         local_k=None):
    """ln of the integral over the box [lo, hi] of exp(ln_f(x)) by adaptive importance sampling.

    ``ln_f(x)`` takes an (n, p) array of in-box points and returns ln integrand (lnL + ln prior).
    The proposal is defensive * Uniform(box) + (1 - defensive) * KDE (``fit_kde_proposal``, kernels
    widened by ``inflate``); the KDE starts from the seed (``seed_x``, ``seed_ln_w``).

    Two phases.  ADAPT: draw a block, refit the KDE to every weighted draw so far, repeat until a
    block's estimate agrees with the earlier blocks' pooled estimate within ``stable_nsigma``
    (``_blocks_agree``).  A proposal still moving toward the target gives rising block estimates and
    keeps adapting.  ESTIMATE: freeze the proposal that drew the agreeing block and draw from it until
    those draws alone reach ``n_eff_target``; only they enter lnZ (plain importance sampling, so
    sigma is the usual one).  Adaptation blocks are discarded from the estimate: under a displaced
    seed their weights are heavy-tailed, and pooling them leaves a bias their sigma does not show.
    ``include_decider`` counts the agreeing block toward the estimate.  Draws outside the box count
    as zeros.  ``n_max`` caps all draws (block-granular); if it runs out while adapting, the last
    block's proposal is used as is.
    The proposal also freezes only once the agreeing block's Pareto k-hat (``pareto_khat``) is at most
    ``khat_max``: two blocks from a proposal stuck in the target's tail can agree with each other.
    Refits use the draws' weights raised to ``refit_temper``: at n_eff ~ 30 the raw weights put the
    kernels on ~30 points, and a curved conditional (sky ring, phase-polarization ridge) falls
    between them; tempering spreads the kernels along it.  The estimate uses the raw weights.
    ``broad`` = (x, ln_w), if given, adds a fixed KDE of those points with weight ``broad_frac`` to every
    block's mixture (ILE passes the main loop's tempered posterior): mass anywhere the main loop saw
    stays reachable when the slice seed is displaced, where the uniform box would almost never land.
    The whole slice stops after ``max_blocks`` blocks (or ``n_max`` draws) whether or not n_eff was
    reached; a warm proposal that slow is not worth continuing, and the caller redoes it cold.
    Adaptation stops after ``max_adapt_blocks`` blocks whatever the checks say; a seed that needs more is
    far from the target, and on synthetic tests those are the runs that later pass k-hat while biased
    low by nats, so the result is returned with ``converged`` False for the caller to redo cold.
    Returns a dict: lnZ, sigma, neff, ntotal, x, ln_w (all draws, the seed for the next slice), khat
    (estimate-phase weights; > 0.7: do not trust lnZ), n_adapt (adaptation blocks), converged.
    """
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    p = len(lo)
    ln_vol = float(np.sum(np.log(hi - lo)))
    kde = fit_kde_proposal(seed_x, seed_ln_w, lo, hi, n_kernel, rng, inflate=inflate,
                           local_k=local_k) if seed_x is not None else None

    broad_kde = None
    if broad is not None and broad_frac > 0:
        broad_kde = fit_kde_proposal(broad[0], broad[1], lo, hi, n_kernel, rng, inflate=inflate, local_k=local_k)
    # a slice whose seed or broad component collapsed is never trusted: the caller redoes it cold
    collapsed = bool((kde is not None and kde.get("collapsed")) or (broad_kde is not None and broad_kde.get("collapsed")))

    def block(kde):
        # mixture: uniform box (defensive), fixed broad KDE (broad_frac), adaptive KDE (the rest)
        comps = [(float(defensive), None)]
        if broad_kde is not None:
            comps.append((float(broad_frac), broad_kde))
        rest = 1.0 - sum(c[0] for c in comps)
        if kde is not None and rest > 0:
            comps.append((rest, kde))
        else:
            comps[0] = (comps[0][0] + rest, None)
        fr = np.array([c[0] for c in comps])
        counts = rng.multinomial(n_block, fr / fr.sum())
        x = np.empty((n_block, p))
        i0 = 0
        for (fc, kc), nc in zip(comps, counts):
            x[i0:i0 + nc] = lo + (hi - lo) * rng.random((nc, p)) if kc is None else kde_draw(kc, nc, rng)
            i0 += nc
        inside = np.all((x >= lo) & (x <= hi), axis=1)
        ln_q = np.full(n_block, -np.inf)
        for fc, kc in comps:
            if fc > 0:
                ln_q = np.logaddexp(ln_q, np.log(fc) + (-ln_vol if kc is None else kde_logpdf(kc, x, xp=xp)))
        ln_w = np.full(n_block, -np.inf)
        if np.any(inside):
            ln_w[inside] = np.asarray(ln_f(x[inside]), float) - ln_q[inside]
        ln_w[~np.isfinite(ln_w)] = -np.inf
        return x[inside], ln_w[inside]

    xs, lws = [], []          # every draw
    est = []                  # ln_w of the estimate phase, per block
    n_total = 0
    converged = False
    n_adapt = 0
    while n_total < n_max and n_adapt < max_adapt_blocks:    # adapt
        n_adapt += 1
        x, lw = block(kde)
        xs.append(x)
        lws.append(lw)
        n_total += n_block
        if _blocks_agree(lws, n_block, stable_nsigma) and pareto_khat(lw) <= khat_max and (
                mean_nsigma is None or _means_agree(xs, lws, mean_nsigma, lo, hi)):
            if include_decider:
                est.append(lw)
            converged = True
            break
        all_lw = np.concatenate(lws)
        if np.any(np.isfinite(all_lw)):
            kde = fit_kde_proposal(np.concatenate(xs), refit_temper * all_lw, lo, hi, n_kernel, rng, inflate=inflate,
                                   local_k=local_k)
    run = _RunningNeff()
    for lw in est:
        run.add(lw)
    n_cap = min(n_max, int(max_blocks) * n_block)
    while n_total < n_cap and run.neff < n_eff_target:   # estimate
        x, lw = block(kde)
        xs.append(x)
        lws.append(lw)
        est.append(lw)
        run.add(lw)
        n_total += n_block
    x_all, lw_all = np.concatenate(xs), np.concatenate(lws)
    res = dict(lnZ=-np.inf, sigma=np.inf, neff=0.0, ntotal=int(n_total), x=x_all, ln_w=lw_all, khat=np.inf,
               n_adapt=n_adapt, converged=converged and not collapsed, collapsed=collapsed)
    if est:
        lw_est = np.concatenate(est)
        lnZ, sigma = _block_estimate(lw_est, n_block * len(est))
        res.update(lnZ=float(lnZ), sigma=float(sigma), neff=_neff(lw_est), khat=pareto_khat(lw_est))
    return res


#: Extrinsic axes on which ILE's likelihood is periodic over its box (RA and orbital phase over 2 pi;
#: polarization over pi, so also over a 2 pi box; under --internal-rotate-phase the phi_orb and psi
#: slots hold phi +/- psi on (0, 4 pi), periodic over that box too).
PERIODIC_EXTRINSIC = ("right_ascension", "phi_orb", "psi")


class WarmSliceSampler:
    """Warm-started fixed-d integrals over the extrinsic parameters of ``reference_sampler``.

    Seeded with the main loop's samples ``seed`` ({param: array}, all extrinsic axes) and their
    posterior log weights ``seed_ln_w``; ``seed_distance`` (optional) orders the first slices
    outward from the seed's weighted median distance.  Each later slice is seeded by the finished
    slice nearest in ln d, with the main loop's seed kept in every slice's mixture as a fixed broad
    component (``warm_integrate_slice``'s ``broad``).  ``run(d_slices)`` returns (lnL, sigmaL, neff, ntotal) like
    ``fresh_sample_slices`` (lnL is the integral of the offset-subtracted likelihood).

    A slice is trusted when it reaches ``n_eff_target`` with Pareto k-hat <= ``khat_max``; only trusted
    slices seed later ones.  An untrusted slice is redone by ``fallback(d_array)`` (same return as
    ``fresh_sample_slices``; ILE passes the cold AV) when given, its draws added to ntotal, and counted
    in ``n_fallback``; without a fallback it is returned as is.
    """

    def __init__(self, reference_sampler, like_to_integrate, seed, seed_ln_w, seed_distance=None,
                 n_block=AV_NSEL_SATURATION_CHUNK, n_kernel=1000, defensive=0.1, rng=None, return_lnL=True,
                 verbose=True, fallback=None, khat_max=np.inf, periodic=PERIODIC_EXTRINSIC, xp=None, **integrator_kw):
        from RIFT.integrators.mcsamplerAdaptiveVolume import _prior_pdf_accepts_xpy
        self._accepts_xpy = _prior_pdf_accepts_xpy
        self.like = like_to_integrate
        self.arg_names = like_to_integrate.__code__.co_varnames[:like_to_integrate.__code__.co_argcount]
        if "distance" not in self.arg_names:
            raise ValueError("like_to_integrate has no 'distance' arg")
        self.params = [a for a in self.arg_names if a != "distance"]
        self.lo = np.array([float(reference_sampler.llim[q]) for q in self.params])
        self.hi = np.array([float(reference_sampler.rlim[q]) for q in self.params])
        self.prior = [reference_sampler.prior_pdf[q] for q in self.params]
        self.n_block, self.n_kernel, self.defensive = int(n_block), int(n_kernel), float(defensive)
        self.rng = np.random.default_rng() if rng is None else rng
        self.return_lnL, self.verbose, self.integrator_kw = return_lnL, verbose, integrator_kw
        self.fallback, self.khat_max, self.n_fallback = fallback, float(khat_max), 0
        self.integrator_kw["khat_max"] = self.khat_max
        self.integrator_kw["xp"] = xp
        seed_x = np.column_stack([np.asarray(_to_cpu(seed[q]), float) for q in self.params])
        self.seed_ln_w = np.asarray(_to_cpu(seed_ln_w), float)
        # Periodic axes are sampled in a rotated copy whose box centre is the seed's circular mean,
        # so a mode at the wrap point (sky at RA ~ 0, measured on S240426s) is not split across the
        # two ends of the box, where one shared kernel covariance would smear it over the whole
        # circle.  A rotation mod the period preserves measure: the integral is unchanged.
        self.rot = np.zeros(len(self.params))
        self.period = np.zeros(len(self.params))
        ok = np.isfinite(self.seed_ln_w)
        w = np.exp(self.seed_ln_w[ok] - self.seed_ln_w[ok].max()) if np.any(ok) else None
        for j, q in enumerate(self.params):
            width = self.hi[j] - self.lo[j]
            if q in periodic and w is not None and any(np.isclose(width, c * np.pi) for c in (1, 2, 4)):
                a = 2 * np.pi * (seed_x[ok, j] - self.lo[j]) / width
                m = np.arctan2(w @ np.sin(a), w @ np.cos(a)) % (2 * np.pi)
                self.period[j], self.rot[j] = width, m * width / (2 * np.pi) - 0.5 * width
        self.seed_x = self._to_internal(seed_x)
        self.d_center = None
        if seed_distance is not None:
            dd = np.asarray(_to_cpu(seed_distance), float)
            ok = np.isfinite(self.seed_ln_w) & (dd > 0)
            if np.any(ok):
                w = np.exp(self.seed_ln_w[ok] - self.seed_ln_w[ok].max())
                o = np.argsort(dd[ok])
                c = np.cumsum(w[o])
                self.d_center = float(dd[ok][o][np.searchsorted(c, 0.5 * c[-1])])
        self.done = []          # (ln d, x, ln_w) of finished slices

    def _to_internal(self, x):
        y = np.array(x, float, copy=True)
        for j in np.flatnonzero(self.period):
            y[:, j] = self.lo[j] + np.mod(y[:, j] - self.lo[j] - self.rot[j], self.period[j])
        return y

    def _to_external(self, y):
        x = np.array(y, float, copy=True)
        for j in np.flatnonzero(self.period):
            x[:, j] = self.lo[j] + np.mod(x[:, j] - self.lo[j] + self.rot[j], self.period[j])
        return x

    def _ln_prior(self, x):
        out = np.zeros(len(x))
        for j, fn in enumerate(self.prior):
            xc = x[:, j]
            v = np.asarray(_to_cpu(fn(xc, xpy=np) if self._accepts_xpy(fn) else fn(xc)), float)
            with np.errstate(divide="ignore"):
                out += np.log(v)
        return out

    def _ln_f(self, d):
        def f(y):
            x = self._to_external(y)
            cols = {q: x[:, j] for j, q in enumerate(self.params)}
            cols["distance"] = np.full(len(x), float(d))
            lnL = np.asarray(_to_cpu(self.like(*(cols[a] for a in self.arg_names))), float)
            if not self.return_lnL:
                with np.errstate(divide="ignore"):
                    lnL = np.log(lnL)
            return lnL + self._ln_prior(x)
        return f

    def run(self, d_slices, n_max, n_eff_target):
        d = np.asarray(d_slices, float)
        K = len(d)
        out = (np.full(K, -np.inf), np.full(K, np.inf), np.zeros(K), np.zeros(K, dtype=int))
        ref = np.log(self.d_center) if self.d_center else float(np.median(np.log(d))) if K else 0.0
        for k in np.argsort(np.abs(np.log(d) - ref), kind="stable"):
            lnd = np.log(d[k])
            if self.done:
                j = int(np.argmin([abs(lnd - r[0]) for r in self.done]))
                sx, sw = self.done[j][1], self.integrator_kw.get("refit_temper", 0.5) * self.done[j][2]
            else:
                sx, sw = self.seed_x, self.seed_ln_w
            try:
                res = warm_integrate_slice(
                    self._ln_f(d[k]), self.lo, self.hi, sx, sw, n_max, n_eff_target,
                    n_block=self.n_block, defensive=self.defensive, n_kernel=self.n_kernel, rng=self.rng,
                    broad=(self.seed_x, self.seed_ln_w), **self.integrator_kw)
                lnZ, sig, ne, nt, x, lw, kh = (res[q] for q in ("lnZ", "sigma", "neff", "ntotal", "x", "ln_w", "khat"))
                conv = res["converged"]
            except Exception as e:
                print("  warm slice d={:.2f} failed: {!r}".format(d[k], e))
                lnZ, sig, ne, nt, kh, conv = -np.inf, np.inf, 0.0, 0, np.inf, False
            trusted = conv and np.isfinite(lnZ) and ne >= n_eff_target and kh <= self.khat_max
            if self.verbose:
                print("    : warm slice d={:.1f} lnZ={:.3f} sigma={:.3f} neff={:.1f} khat={:.2f} ntotal={} seed={}{}".format(
                    d[k], lnZ, sig, ne, kh, nt, "slice" if self.done else "main",
                    "" if trusted else " -> cold fallback" if self.fallback else " UNTRUSTED"))
            if trusted:
                self.done.append((lnd, x, lw))
            elif self.fallback is not None:
                self.n_fallback += 1
                fl, fs, fn, ft = (np.asarray(v).ravel()[0] for v in self.fallback(np.array([d[k]])))
                lnZ, sig, ne, nt = fl, fs, fn, nt + int(ft)
            for arr, v in zip(out, (lnZ, sig, ne, nt)):
                arr[k] = v
        return out


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
