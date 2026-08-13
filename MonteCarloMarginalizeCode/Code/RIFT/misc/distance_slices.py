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
            for p, arr in kw.items():
                lo, hi = omega_bounds.get(p, (-np.inf, np.inf))
                # nudge inward by a tiny epsilon relative to range, so arccos
                # and friends never see the exact boundary
                eps = 1e-12 * max(abs(hi - lo), 1.0)
                full[p] = xp.clip(xp.asarray(arr, dtype=float), lo + eps, hi - eps)
            full["distance"] = xp.full_like(sample, d_fixed, dtype=float)
            # like_to_integrate is the cached ILE likelihood -> returns CUPY on a
            # GPU run; the AV bookkeeping downstream is host-side, so bring the
            # lnL vector (and only that) back.
            return _to_cpu(like_to_integrate(*(full[a] for a in arg_names)))

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
