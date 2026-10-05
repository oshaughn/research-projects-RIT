"""Per-intrinsic likelihood-vs-distance export for ILE.

The exported ``lnL`` is the *pure* extrinsic-marginalized likelihood as a
function of luminosity distance::

    L_pure(d) = integral L(d, Omega) pi_Omega(Omega) dOmega

i.e. the distance sampling prior has been divided out.  Downstream consumers
can re-marginalize over distance with any prior pi'(d)::

    L_marg' = sum_k exp(lnL[k]) * pi'(dist[k]) * dist_weight[k]

For convenience the grid also carries ``ln_prior_d_sampling``, the per-bin
log of the distance prior that ILE used while integrating, so the original
marginal likelihood can be reproduced exactly::

    lnL_marg = logsumexp(lnL + ln_prior_d_sampling + log(dist_weight))
"""
import numpy as np


DISTANCE_GRID_FIELDS = (
    "lnL",
    "sigmaL",
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
    "dist_weight",
    "ln_prior_d_sampling",
)


def _logsumexp(vals):
    vals = np.asarray(vals, dtype=float)
    vmax = np.max(vals)
    if not np.isfinite(vmax):
        return vmax
    return vmax + np.log(np.sum(np.exp(vals - vmax)))


# A bin centre is the weighted mean distance of its block, so two blocks built
# from the same distance value share a centre and leave a zero-width bin between
# them.  One distinct distance therefore cannot be a grid at all.
MIN_UNIQUE_DISTANCES = 2


def _as_positive_integer(value, default):
    if value is None:
        return default
    value = int(value)
    if value < 1:
        raise ValueError("distance grid size must be positive")
    return value


def _weighted_blocks(distance, ln_prior_d, probability, n_grid):
    """Sort samples by distance, split into at most n_grid blocks that never cut a
    run of identical distances, and return per-block (center, mass, width, mean
    ln-prior).  n_grid is reduced to the number of DISTINCT distances when there are
    fewer of those than bins asked for."""
    order = np.argsort(distance)
    distance = np.asarray(distance, dtype=float)[order]
    probability = np.asarray(probability, dtype=float)[order]
    ln_prior_d = np.asarray(ln_prior_d, dtype=float)[order]

    finite = np.isfinite(distance) & np.isfinite(probability) & (probability > 0) & np.isfinite(ln_prior_d)
    distance = distance[finite]
    probability = probability[finite]
    ln_prior_d = ln_prior_d[finite]
    if len(distance) == 0:
        raise ValueError("no finite positive-weight distance samples to export")

    # Block boundaries must fall BETWEEN distinct distances, never inside a run
    # of identical ones.  ILE's fair draw resamples with replacement, so a
    # starved extrinsic pass hands us a few distinct distances each repeated
    # several times; an equal-count split of the raw samples then cuts such a
    # run, gives both halves the same weighted-mean centre, and leaves a
    # zero-width bin between them.  Splitting the DISTINCT distances instead is
    # the identical equal-count split whenever the samples are all distinct.
    starts = np.concatenate(([0], np.flatnonzero(np.diff(distance)) + 1))
    n_unique = len(starts)
    if n_unique < MIN_UNIQUE_DISTANCES:
        raise ValueError(
            "cannot resolve a distance grid: {} finite positive-weight sample(s) "
            "span only {} distinct distance(s)".format(len(distance), n_unique))

    n_grid = min(_as_positive_integer(n_grid, n_unique), n_unique)
    first_unique = [group[0] for group in np.array_split(np.arange(n_unique), n_grid)]
    bounds = np.append(starts[first_unique], len(distance))
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_grid)]
    grid_dist = np.empty(len(blocks))
    grid_mass = np.empty(len(blocks))
    grid_ln_prior = np.empty(len(blocks))
    for i, block in enumerate(blocks):
        w = probability[block]
        grid_mass[i] = np.sum(w)
        grid_dist[i] = np.sum(distance[block] * w) / grid_mass[i]
        # weighted average of ln_prior_d (in log space, by importance weights):
        # log E_w[pi_d] = logsumexp(ln_prior_d + log w) - log sum_w
        grid_ln_prior[i] = (
            _logsumexp(ln_prior_d[block] + np.log(w)) - np.log(grid_mass[i])
        )

    if len(grid_dist) == 1:
        width = np.array([np.ptp(distance)])
    else:
        edges = np.empty(len(grid_dist) + 1)
        edges[1:-1] = 0.5 * (grid_dist[1:] + grid_dist[:-1])
        edges[0] = min(distance[0], grid_dist[0] - (edges[1] - grid_dist[0]))
        edges[-1] = max(distance[-1], grid_dist[-1] + (grid_dist[-1] - edges[-2]))
        width = np.diff(edges)

    # lnL carries -log(width), so a floored zero width is not a small error: the
    # old np.maximum(width, eps) floor reported a bin as ~36 nats brighter than
    # its neighbours.  Distinct-distance blocking makes the centres strictly
    # increasing, so duplicate rows can no longer reach here -- but distances
    # separated by an ULP still can (5000.0 + arange(5)*1e-12 raises), because
    # their midpoints collapse onto the centres in float64.  Such a grid cannot be
    # resolved at all; refuse it rather than floor it into a bright bin.
    if not np.all(width > 0):
        raise ValueError(
            "cannot resolve a distance grid: {} of {} bin(s) have non-positive "
            "width over distances [{:g}, {:g}]".format(
                int(np.sum(width <= 0)), len(width), distance[0], distance[-1]))

    return grid_dist, grid_mass, width, grid_ln_prior


def reserve_distance_and_ln_weights(reserve, param="distance"):
    """(distance, ln importance weight) from a sampler's retained-set reserve, or None.

    The reserve (RIFT.integrators.mcsamplerAdaptiveVolume.make_warm_seed_reserve) is a
    bounded copy of the rows a pass actually RETAINED, taken before the fair draw
    rebinds ``_rvs`` to a few rows resampled with replacement.  It carries both prior
    components precisely so a consumer can rebuild the importance weight from it, which
    is what a distance grid needs: the retained rows are NOT equal-weight, so they must
    be weighted by w, unlike a fair-drawn record.

    Returns None whenever the reserve cannot answer -- absent, empty, missing a prior
    component, or built by a sampler whose parameter list has no such coordinate -- so
    the caller can fall back to ``_rvs`` without inspecting the dict itself.
    """
    if not reserve:
        return None
    names = list(reserve.get("params_ordered") or [])
    if param not in names:
        return None
    X = np.asarray(reserve.get("X"))
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(names):
        return None
    ln_prior = reserve.get("log_joint_prior")
    ln_s_prior = reserve.get("log_joint_s_prior")
    if ln_prior is None or ln_s_prior is None:
        return None
    lnL = np.asarray(reserve.get("lnL"), dtype=float).ravel()
    ln_prior = np.asarray(ln_prior, dtype=float).ravel()
    ln_s_prior = np.asarray(ln_s_prior, dtype=float).ravel()
    if not (len(lnL) == len(ln_prior) == len(ln_s_prior) == X.shape[0]):
        return None
    ln_w = lnL + ln_prior - ln_s_prior

    # HORVITZ-THOMPSON: divide each row by the probability it was included.  A bounded
    # reserve draws n_subsample of n_finite rows uniformly, so that probability is identical
    # for every ordinary row and cancels in a normalized histogram -- EXCEPT for a peak row
    # appended unconditionally, whose probability is 1.  Correcting it HERE, rather than
    # asking every builder for force_peak=False, is what makes this right for the reserves
    # the .dgrid actually meets: AV and the portfolio build their own for the L0 rescue,
    # which wants that row, and never go through make_reserve_from_rvs.
    if reserve.get("capped") and reserve.get("force_peak"):
        n_sub, n_fin = reserve.get("n_subsample"), reserve.get("n_finite")
        if n_sub and n_fin and 0 < n_sub < n_fin and len(ln_w):
            ln_w = ln_w.copy()
            ln_w[int(np.argmax(lnL))] += np.log(float(n_sub) / float(n_fin))
        else:
            # A capped, peak-forced reserve with no recorded n_subsample cannot be
            # corrected -- it predates the field.  Using it anyway means exporting the
            # forced row at probability 1, which is the bias this whole path exists to
            # remove, so decline and let the caller fall back to _rvs.
            return None
    return X[:, names.index(param)].astype(float), ln_w


def _ess(ln_weights):
    """Kish effective sample size of log weights; 0.0 if none are usable.

    EXACTLY EQUAL weights must give exactly n.  Summing n copies of 1/n in float64 lands a
    few ULP off, so 1/sum(p^2) came back as n*(1-eps) -- 2.9999999999999996 for n=3 -- and
    a strict `n_eff < n_rows` then reported "fewer than one effective sample per bin" for
    a perfectly flat weight set, at n = 3, 5, 6, 8, 9, 10 but not 2, 4, 7, 11.  Rounding to
    the representable neighbour removes the artifact without touching a genuine shortfall.
    """
    lw = np.asarray(ln_weights, dtype=float).ravel()
    lw = lw[np.isfinite(lw)]
    if lw.size == 0:
        return 0.0
    p = np.exp(lw - _logsumexp(lw))
    denom = float(np.sum(p ** 2))
    if denom <= 0:
        return 0.0
    ess = 1.0 / denom
    return float(np.round(ess)) if abs(ess - np.round(ess)) < 1e-9 * max(1.0, ess) else ess


def distance_grid_resolution_warning(distance, n_grid=None, ln_weights=None,
                                     n_eff_population=None, n_eff_sampler=None):
    """Describe a distance grid the samples cannot resolve, or return None.

    Pass ``n_eff_population`` when the rows handed in are a bounded subsample of a larger
    retained set (RvsRecord's reserve records it as ``ess_finite``): the subsample's own
    effective sample size describes the subsample, not the pass.  Pass ``n_eff_sampler``
    when the sampler reported one -- it is the only input here that a saturated integrand
    column cannot fake.

    Two separate ways the samples can fail to support the table that gets written, and
    which one bites depends on where the rows came from:

    * FROM THE FAIR DRAW.  ``_rvs`` after integrate_log holds a few rows resampled WITH
      REPLACEMENT, so the grid cannot carry more rows than there are DISTINCT distances
      in it, however many were asked for.
    * FROM THE RETAINED SET.  Thousands of distinct distances, so the row count looks
      healthy -- but on a starved pass nearly all the weight sits on a handful of them.
      Pass ``ln_weights`` and the effective sample size is checked against the bin
      count: below one effective sample per bin the per-bin lnL is not estimated at all.

    The second check is why the first is not enough on its own.  Reading the retained
    set instead of the export resample removes the duplicate-distance failure and would
    otherwise turn a visibly broken table into a smooth-looking one at the same n_eff.
    """
    d = np.asarray(distance, dtype=float).ravel()
    keep = np.isfinite(d)
    lw = None
    if ln_weights is not None:
        lw = np.asarray(ln_weights, dtype=float).ravel()
        if len(lw) != len(d):
            lw = None
    if lw is not None:
        # THE SAME ROWS THE BUILDER WILL BIN.  _weighted_blocks drops every row whose
        # normalized probability is not strictly positive, so counting distinct distances
        # over all FINITE rows describes a different sample set than the one that gets
        # binned -- and the divergence is not symmetric: a pass where one weight survives
        # and the rest underflow is the genuinely starved one, and it was the silent one.
        finite_w = np.isfinite(lw)
        if not np.any(finite_w):
            return "no finite importance weight survived: there is nothing to bin"
        # `p > 0` subsumes isfinite(lw): a -inf weight gives p == 0 and a NaN gives NaN,
        # and both fail this test.  Masking on isfinite(lw) as well was dead belt-and-braces
        # -- it could be deleted with every test still green, which is how it was found.
        p = np.exp(lw - _logsumexp(lw[finite_w]))
        keep = keep & np.isfinite(p) & (p > 0)
    d = d[keep]
    if len(d) == 0:
        return "no finite positive-weight distance samples to export"
    n_unique = len(np.unique(d))
    n_requested = min(_as_positive_integer(n_grid, len(d)), len(d))
    n_rows = min(n_requested, n_unique)
    problems = []
    if n_unique < n_requested:
        # A FACT ABOUT THE TABLE, not a verdict on it.  Four usable samples over three
        # distinct distances gives a perfectly sound three-row grid; whether three rows
        # is a curve is the effective-sample-size question below, and saying so here too
        # made one duplicate in the default five-row draw print the strongest wording
        # available.
        problems.append(
            "{} usable sample(s) span only {} distinct distance(s), so the grid carries "
            "{} row(s), not the {} requested.".format(
                len(d), n_unique, n_rows, n_requested))
    if n_eff_sampler is not None and not (float(n_eff_sampler) >= n_rows):
        # `not (x >= n)` rather than `x < n`, so a NaN reports rather than passes: a NaN
        # n_eff is a pass that could not measure its own convergence, which is the case
        # this clause exists for.
        # THE SAMPLER'S OWN n_eff, which no weight column can contradict.  On a collapsed
        # pass the linear `integrand` column bottoms out at its underflow floor, so the
        # derived weights come back UNIFORM and every check above reports a healthy grid:
        # measured on a collapsed Ensemble pass, ESS 4000 of 4000 rows while the exported
        # curve was flat to 0.13 nats across a true 1130-nat span.  The pass itself knew.
        problems.append(
            "the sampler reported n_eff={:.1f} for this pass, not the {} bin(s) exported: "
            "the integration did not converge, whatever the weight column says.".format(
                float(n_eff_sampler), n_rows))
    if lw is not None or n_eff_population is not None:
        # THE POPULATION'S ESS WHEN THE CALLER HAS IT.  When the rows are a bounded uniform
        # subsample of the retained set, the subsample's own ESS is not an estimate of the
        # population's: drop the one row that carries the weight and what is left looks
        # healthy (measured 505 against a population 2.1).  The reserve records the exact
        # pre-cap value for exactly this reason; use it when it is there.
        n_eff = float(n_eff_population) if n_eff_population is not None else _ess(lw[keep])
        if n_eff < n_rows:
            problems.append(
                "the weights carry n_eff={:.1f} across {} bin(s), fewer than one "
                "effective sample per bin: the exported curve is resampled points, not a "
                "resolved likelihood-vs-distance curve.".format(n_eff, n_rows))
    return " ".join(problems) if problems else None


def _rvs_row_count(rvs):
    """Row count of a sample-column dict.  Deferred import: rvs_record owns the one
    implementation (it handles the tuple keys --skymap-file creates), and importing it
    lazily keeps this module free of an integrator dependency at import time."""
    try:
        from RIFT.integrators.rvs_record import n_rows
        return n_rows(rvs)
    except Exception:
        for v in rvs.values():
            arr = np.asarray(v)
            return arr.shape[-1] if arr.ndim > 1 else len(arr)
        return 0


def distance_grid_inputs(record, rvs, posterior_ln_weights, convert=None,
                         param="distance", n_grid=None, n_eff_sampler=None,
                         prior_pdf=None):
    """Everything the .dgrid export needs to decide, as one testable function.

    -> (distance, ln_weights, ln_prior_at_rows, notes, warning)

    ``prior_pdf`` is the sampler's own distance prior, evaluated HERE so that it cannot be
    separated from the choice of rows.  Keeping it in the caller left a two-line gap in
    which the prior could be taken at the fair draw's distances while the curve was built
    from the retained set's: measured, that moves the exported lnL by 2.87 nats and turns
    ln_prior_d_sampling from a real d^2 ramp into a flat column, with every test green.
    That is the same shape as the params_out deletion, so the fix is structural rather
    than another assertion.

    THIS LIVES HERE BECAUSE IT HAS TO BE RUNNABLE.  It was six statements inside
    ``analyze_event`` in the ILE script, which nothing can import, so the only available
    tests asserted that the source text mentioned the right names.  Those tests pass while
    the branch is inverted (``if not _resampled``), while the returned rows are thrown away
    and re-read from ``_rvs``, while ``param='psi'`` is exported as distance, and while the
    warning is computed and never printed -- all measured.  A named function with a return
    value can simply be called.

    ``record`` is the sampler's RvsRecord for THESE rows (or None), ``rvs`` the column dict
    the fair draw has by now replaced, and ``posterior_ln_weights`` a zero-argument callable
    giving the fall-back weighting for the rvs rows.
    """
    conv = convert if convert is not None else (lambda x: x)
    notes = []

    # is_equal_weight(), not rows_are_resampled(): a POOLED record is resampled per block
    # but is not one sampler's draws, so one replica's reserve would give the curve of one
    # replica under the evidence of the pool.  And when the record is absent it is absent
    # BECAUSE it does not describe these rows, which is when the sampler's reserve
    # attribute is least entitled to speak for them -- so no reserve then either.
    reserve = getattr(record, "reserve", None) if record is not None else None
    retained = None
    if record is not None and record.is_equal_weight():
        retained = reserve_distance_and_ln_weights(reserve, param=param)

    if retained is not None:
        distance, ln_weights = retained
        n_eff_population = (reserve or {}).get("ess_finite")
        notes.append("built from the RETAINED set ({} rows) rather than the {}-row fair "
                     "draw".format(len(distance), _rvs_row_count(rvs)))
        if (reserve or {}).get("capped"):
            notes.append("that is a uniform subsample of {} finite retained rows, so the "
                         "exported curve carries subsample noise".format(
                             (reserve or {}).get("n_finite")))
    else:
        distance = np.asarray(conv(rvs[param]), dtype=float).ravel()
        ln_weights = np.asarray(posterior_ln_weights(), dtype=float)
        n_eff_population = None

    if prior_pdf is not None:
        pi_d = np.asarray(conv(prior_pdf(distance)), dtype=float).ravel()
        # The prior must be strictly positive at the rows being exported.
        ln_prior_at_rows = np.log(np.where(pi_d > 0, pi_d, np.finfo(float).tiny))
    else:
        ln_prior_at_rows = None

    warning = distance_grid_resolution_warning(
        distance, n_grid=n_grid, ln_weights=ln_weights,
        n_eff_population=n_eff_population, n_eff_sampler=n_eff_sampler)
    return distance, ln_weights, ln_prior_at_rows, notes, warning


def build_distance_grid(distance, ln_weights, lnL_marginal, sigmaL, params,
                        ln_prior_d_at_samples, n_grid=None):
    """Build a likelihood-vs-distance grid from weighted ILE samples.

    Parameters
    ----------
    distance : array
        Per-sample luminosity distances drawn by the ILE sampler.
    ln_weights : array
        Per-sample log importance weights, ``log L_i + log pi(theta_i) - log q(theta_i)``,
        with ``pi`` and ``q`` being the joint prior and proposal used by ILE.
        These weights include the distance prior.
    lnL_marginal : float
        The marginalized lnL the ILE batchmode would report (``log_res +
        manual_avoid_overflow_logarithm``).  Used as the absolute calibration.
    sigmaL : float
        ILE's reported lnL uncertainty.  Carried verbatim into the grid.
    params : dict
        Intrinsic parameters to broadcast across the grid rows (mass, spins,
        tides, ...).  Missing keys default to 0.
    ln_prior_d_at_samples : array
        Per-sample log of the *distance* prior pi_d(d_i) used by ILE.  This
        is divided out so the exported ``lnL`` is a pure likelihood, not a
        density-times-prior.
    n_grid : int, optional
        Number of grid bins.  Defaults to, and is capped at, the number of
        DISTINCT finite positive-weight distances -- two bins built from the same
        distance would share a centre and leave no width between them.  (Bin
        widths themselves are the midpoints between adjacent centres, with the two
        end bins extrapolated, so an individual width is not the local sample
        spacing.)
    """
    ln_weights = np.asarray(ln_weights, dtype=float)
    ln_norm = _logsumexp(ln_weights)
    probability = np.exp(ln_weights - ln_norm)
    grid_dist, grid_mass, grid_width, grid_ln_prior = _weighted_blocks(
        distance, ln_prior_d_at_samples, probability, n_grid)

    dtype = [(name, float) for name in DISTANCE_GRID_FIELDS]
    grid = np.zeros(len(grid_dist), dtype=dtype)
    # Pure likelihood density in d: subtract log mean prior_d in bin so
    # exp(lnL) = L_marg * p_post(d) / pi_d(d) = L(d) [extrinsic-marginalized].
    grid["lnL"] = (
        lnL_marginal + np.log(grid_mass) - np.log(grid_width) - grid_ln_prior
    )
    grid["sigmaL"] = sigmaL
    grid["dist"] = grid_dist
    grid["dist_weight"] = grid_width
    grid["ln_prior_d_sampling"] = grid_ln_prior

    for name in DISTANCE_GRID_FIELDS:
        if name in {"lnL", "sigmaL", "dist", "dist_weight", "ln_prior_d_sampling"}:
            continue
        grid[name] = float(params.get(name, 0.0))
    return grid


def save_distance_grid(fname, grid):
    header = " ".join(grid.dtype.names)
    np.savetxt(fname, np.column_stack([grid[name] for name in grid.dtype.names]), header=header)


def load_distance_grid(fname):
    return np.genfromtxt(fname, names=True)


def reconstruct_marginal_lnL(grid, ln_prior_d=None):
    """Reconstruct the marginal lnL by integrating exp(lnL)*prior(d) over the
    grid.  If ``ln_prior_d`` is None and the grid has the ``ln_prior_d_sampling``
    column, that column (the sampling prior) is used.  Otherwise integrates
    against a flat prior (treats lnL as already-pure).  Pass a callable
    ``ln_prior_d(d)`` to integrate against a custom distance prior.
    """
    names = grid.dtype.names
    if "dist_weight" not in names:
        # legacy grids without dist_weight: trapezoidal
        order = np.argsort(grid["dist"])
        trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return np.log(trap(np.exp(grid["lnL"][order]), grid["dist"][order]))

    log_dw = np.log(grid["dist_weight"])
    if ln_prior_d is not None:
        ln_pi = np.asarray(ln_prior_d(grid["dist"]), dtype=float)
        return _logsumexp(grid["lnL"] + ln_pi + log_dw)
    if "ln_prior_d_sampling" in names:
        return _logsumexp(grid["lnL"] + grid["ln_prior_d_sampling"] + log_dw)
    # legacy grids with dist_weight but no separate prior column: treat lnL
    # as a pre-multiplied density (old format)
    return _logsumexp(grid["lnL"] + log_dw)
