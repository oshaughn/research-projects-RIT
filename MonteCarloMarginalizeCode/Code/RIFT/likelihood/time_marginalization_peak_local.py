"""Peak-local time marginalization: enumerate the peaks, integrate only near them.

READ ``time_marginalization_quadrature.py`` FIRST.  That module states the defect
(Simpson at the fixed spacing ``deltaT = 1/srate`` against an integrand of width
``sigma_t = 1/(2 pi rho sigma_f)``), the reason the existing samples already
determine the continuous integrand (``kappa(t)`` is band-limited below Nyquist and
``rho_sq`` is time-independent on this path), and the derived-not-configured
discipline for resolution.  All of that is inherited here unchanged.  This module
changes ONE thing: WHERE the refined grid is placed.

WHAT IS WRONG WITH REFINING THE WHOLE WINDOW
--------------------------------------------
The dense strategy refines the entire window to the peak's width, so its cost
grows as ``rho`` while the peak it is resolving gets NARROWER as ``1/rho``: the
work grows exactly where it is least needed.  It conflates two resolution
requirements that are not the same requirement:

* Resolving ``kappa(t)`` enough to **enumerate its extrema**.  ``kappa`` is
  band-limited at Nyquist by construction, so the narrowest feature it can have is
  a half-cycle of width ``deltaT``.  Enumerating its extrema therefore needs a
  small FIXED factor and is **SNR-INDEPENDENT**.
* Resolving ``exp(lnL(t))`` well enough to integrate it.  That is the rho-dependent
  part, and it is only needed over a few ``sigma_t`` around each enumerated peak.

So: upsample ``kappa`` by ``PEAK_ENUM_FACTOR``, enumerate every local maximum of
the band-limited interpolant, put an interval of half-width ``W_SIGMA * sigma_i``
around each, **merge overlapping intervals into disjoint ones**, integrate each
merged interval at its own derived spacing, and sum in log space.

TWO PROPERTIES THAT MAKE THIS AN ALGORITHM RATHER THAN A HACK
-------------------------------------------------------------
**1.  Merging makes it one algorithm, not a regime switch.**  Isolated peaks give a
tiny union of intervals; overlapping peaks grow the union until it is the whole
window, and the method degenerates *continuously* into the dense grid.  There is no
threshold to tune and no regime to detect.  The un-merged version is not a slightly
worse variant of this -- it double-counts the shared region and is measurably wrong:
``~/tmarg_harness/peaklocal.py`` errs **+1.6 nats at rho ~ 6** where the merged
version is exact, and ``test_unmerged_intervals_double_count`` reproduces that here
so the merge cannot be quietly removed.

**2.  Enumeration plus a computed tail bound makes the truncation rigorous.**  The
danger in any "integrate near the peak" scheme is the mass you did not look at.
RIFT PR #201 was caught by exactly that: a Newton solve seeded at two guessed points
missed genuine maxima and returned ``-inf`` for a finite integral.  There is
deliberately no seeded root-finder here.  Instead:

* every local maximum of the band-limited interpolant is enumerated on a grid that
  resolves ``kappa`` itself, and
* the mass outside the merged intervals is **bounded and checked, per row**, not
  assumed.  ``log(T_outside) + max_{t outside} lnL(t)`` is an upper bound on the
  omitted integral; it is compared against the value actually computed, and a row
  whose bound is not below ``TAIL_LOG_TOL`` relative is handed to the dense
  band-limited path instead of being reported.

That second bullet is the load-bearing one, and it is worth being precise about why:
**the bound does not depend on the enumeration being complete.**  If a peak were
missed entirely, its neighbourhood would be outside the intervals, the sampled
maximum outside would sit on it, and the bound would fail -- sending the row to the
dense path.  Completeness of the enumeration buys SPEED; the bound buys CORRECTNESS.
Those are deliberately not the same mechanism, because a missed-peak argument that
rests on "the grid is fine enough" is exactly the kind of claim that is true until
it is not.

Evaluating the maximum outside is free, and that is not an accident.  Every shipped
``loglikelihood`` callback in scope here -- plain and distance-marginalized -- is
monotone increasing in ``Re kappa`` at fixed ``rho_sq``, and ``rho_sq`` is
time-independent on this path, so a monotone map cannot move a maximum:

    argmax_t lnL(t)  ==  argmax_t Re kappa(t)

whatever the distance, the distance prior, or the callback.  (Verified over an 800x
range in ``1/D`` and across three callback shapes;
``test_peak_positions_do_not_depend_on_distance_or_callback`` pins it.)  So both the
enumeration AND the outside-maximum are computed on ``Re kappa`` alone, with no
callback evaluation on the full time axis -- which matters, because with the
distance-marginalized callback that evaluation is a table interpolation over
``n_extrinsic * npts * factor`` points and is the dominant cost of the dense path.
The callback is evaluated only at the enumerated peaks, at their curvature stencils,
at one point per row for the bound, and on the local grids.

THE COST CLAIM, AND WHAT THE PROTOTYPE'S NUMBER DOES NOT MEAN
-------------------------------------------------------------
The prototype (``~/tmarg_harness/peaklocal2.py``, tabulated in
``DESIGN_time_marginalization_quadrature.md``) reports a flat ~97 evaluation points
against the dense grid's 19,648 -> 628,736, i.e. "200x at rho ~ 15, 6,482x at
rho ~ 692".  **Those numbers were measured with the analytic ``kappa`` in hand**, so
evaluating ``kappa(t)`` at an arbitrary time cost one closed-form call.  In the
shipped code it does not: only the coarse samples exist, and the band-limited
interpolant has to be evaluated at the local grid points.  A point count is
therefore not a cost, and the prototype's speedups are NOT this module's speedups.
See ``DESIGN_time_marginalization_peak_local.md`` for what was measured here, through
this code path, including the interpolation.

The evaluator used is a direct spectral sum, ``kappa(t) = sum_k Xw_k exp(2 pi i f_k
t / T)``, advanced along a uniform local grid by a phase recurrence so that no
transcendental is evaluated per output point.  Its cost is ``O(npts)`` per output
point and, crucially, **independent of rho**: the number of local points is set by
``W_SIGMA`` and ``UPSAMPLE_SAFETY``, not by the peak's width.  The dense path costs
``O(npts * factor * log(npts * factor))`` with ``factor`` proportional to rho, plus a
callback and an ``exp`` over every one of those points.  So the two cross over, this
one wins by more as rho grows, and it LOSES at low rho -- which is why a row whose
estimated local cost exceeds its estimated dense cost is given the dense path.  That
switch is a cost decision only: both branches satisfy the same derived resolution
criterion, so it cannot trade accuracy for speed.

A chirp-z (Bluestein) evaluation would reduce the local evaluation from
``O(npts * M)`` to ``O((npts + M) log(npts + M))`` and is the obvious next step; it
is deliberately not in this draft.

SCOPE
-----
The band-limited path's scope, MINUS phase marginalization: baseline (non-rotating)
likelihood, ``n_cal == 1``, time-independent ``rho_sq`` (checked, not assumed).  The
rotating-response and finite-size-response likelihoods refuse it for the same reason
they refuse ``bandlimited``.

``phase_marginalization=True`` is REFUSED here, and that is a deliberate scope cut
rather than an oversight.  Production marginalizes over distance, not phase, so it
would be a corner nobody runs -- and it is the one corner that genuinely complicates
this design.  Under phase marginalization the Laplace width of the time peak picks up
a factor ``(I1/I0)(|kappa|/D)``, which depends on distance in a way that is not a
power law, so the width does NOT reduce the way it does for the plain and
distance-marginalized callbacks and the derived local spacing would no longer be
derivable from ``rho_sq`` and the curvature alone.  ``bandlimited`` still supports it
and is unchanged; a caller who needs it should use that.  Refusing rather than
silently falling back is the same discipline the rest of this option follows: an
accuracy option that quietly does something else is worse than one that is
unavailable.
"""

import numpy as np

from .time_marginalization_quadrature import (
    UPSAMPLE_SAFETY,
    EDGE_GUARD_FRACTION,
    CURVATURE_STENCIL_HALFWIDTHS,
    bandlimited_upsample,
    peak_width_from_lnL,
    required_upsample_factors,
    time_marginalize_bandlimited,
    _log_simps_rows,
    _safe_offset,
    _require_time_independent_rho_sq,
    _default_simps,
)

__all__ = [
    "PEAK_ENUM_FACTOR",
    "W_SIGMA",
    "PEAK_KEEP_NATS",
    "TAIL_LOG_TOL",
    "MAX_INTERVALS",
    "MIN_LOCAL_POINTS",
    "bandlimited_spectrum",
    "eval_bandlimited_uniform",
    "enumerate_peak_indices",
    "merge_intervals_by_row",
    "time_marginalize_peak_local",
    "last_report",
]


#: Upsampling factor used ONLY to enumerate the extrema of ``kappa`` and to sample
#: the maximum outside the intervals.  SNR-INDEPENDENT, and that independence is the
#: whole point of the split: ``kappa`` is band-limited below Nyquist, so its fastest
#: possible oscillation is a half-cycle of width ``deltaT``, and this places 8 grid
#: points across that narrowest possible lobe regardless of how sharp ``exp(lnL)``
#: has become.  It is NOT an accuracy knob for the integral: the integral's accuracy
#: is set by the local spacing (``UPSAMPLE_SAFETY``) and by the tail bound, and a
#: value too small here shows up as a FAILED tail bound and a dense-path fallback,
#: not as a wrong number.  Measured on the synthetic band-limited fixture, the
#: enumerated peak set at this factor is identical to the set found at factor 64 on
#: every row of the accuracy sweep (see DESIGN_time_marginalization_peak_local.md).
PEAK_ENUM_FACTOR = 8

#: Local interval half-width, in units of the peak's own ``sigma_t``.  A Gaussian
#: peak truncated at ``W_SIGMA`` sigma omits ``erfc(W/sqrt2) ~ exp(-W^2/2)`` of its
#: mass: ``exp(-72) = 5.4e-32`` here, which is below double precision against the
#: largest window-to-sigma dynamic range this path can see (``UPSAMPLE_FACTOR_MAX``
#: bounds it at ~1e4).  Not a knob that can be set too small: the omitted mass is
#: BOUNDED per row by the tail check below, so shrinking this widens the intervals
#: the check demands or sends the row to the dense path -- it cannot silently buy
#: speed with accuracy.
W_SIGMA = 12.0

#: Enumerated peaks more than this far below a row's highest peak are dropped before
#: intervals are built.  ``exp(-60) = 8.8e-27`` relative, so such a peak cannot carry
#: representable mass.  Dropping is safe rather than hopeful for the same reason a
#: missed peak is: a dropped peak's neighbourhood is then OUTSIDE the intervals, so
#: it enters the tail bound, and a row where the drop mattered fails the bound and
#: goes dense.
PEAK_KEEP_NATS = 60.0

#: A row is accepted only if ``log(T_outside) + max_{outside} lnL - result`` is below
#: this, i.e. the bounded omitted mass is under ``e^-23 = 1e-10`` of the value
#: reported.  A relative error ``eps`` in the integral is an ABSOLUTE error ``eps``
#: in the returned log, so this is 1e-10 nats -- seven orders below the ~1e-3 nats
#: scale at which a difference in this quantity means anything.  Rows that fail are
#: handed to the dense band-limited path, not reported with a caveat.
TAIL_LOG_TOL = -23.0

#: Ceiling on the number of DISJOINT intervals a row may have after merging.  This
#: is a fail-closed cost guard, not an accuracy one: a row with more structure than
#: this is not approximated, it is sent to the dense path.  (In the degenerate
#: many-overlapping-peaks case merging collapses the count, so this bites only on a
#: genuinely comb-like integrand.)
MAX_INTERVALS = 32

#: Re-anchor the phase recurrence in ``eval_bandlimited_uniform`` every this many
#: steps.  Purely numerical: ``exp(i x)`` is not exactly unit modulus in floating
#: point, so ``z**m`` by repeated multiplication drifts as ``m * eps``.  Re-anchoring
#: bounds that at ``64 * eps ~ 1e-14`` regardless of how long the local grid is.  It
#: cannot change the answer beyond rounding and is not a tunable.
_RECURRENCE_REANCHOR = 64

#: Working-set budget for one dense temporary, in bytes.  Internal memory chunking over
#: the extrinsic axis.  Rows are independent and the per-row plan -- interval count and
#: point count, which is bucketed to a power of two -- depends only on that row, so this
#: cannot change WHICH rule a row gets or how finely it is integrated.
#:
#: It does move the last bits, and saying otherwise would be an overclaim: chunking
#: changes the leading dimension of the FFT and of the reduction inside
#: :func:`eval_bandlimited_uniform`, and both numpy's FFT and its pairwise summation
#: reassociate with batch shape.  MEASURED at one row per chunk versus all rows at once,
#: on a three-row block spanning rho ~ 100 to 700: 0, 0 and 2 ULPs.
#: ``test_the_memory_chunking_path_assembles_its_result`` pins that, at a bar sharp
#: enough that a dropped or mis-ordered chunk -- which moves a row by nats -- cannot hide
#: behind it.
_CHUNK_BYTES = 128 * 1024 * 1024

_LAST_REPORT = {}


def last_report():
    """Diagnostics from the most recent :func:`time_marginalize_peak_local` call.

    Shares the row-classification keys of
    :func:`time_marginalization_quadrature.last_report` -- ``n_rows``,
    ``n_wrap_exposed_rows``, ``n_unmeasurable_rows``, ``n_flat_rows``,
    ``n_refined_rows`` -- which mean exactly what they mean there, and adds:

    ``n_peak_local_rows``  rows actually integrated by this module's rule.
    ``n_dense_fallback_rows``  refined rows handed to the dense band-limited path.
    ``n_dense_fallback_cost``  of those, how many went for the COST estimate (the
        local grid would have been more work than the dense one -- the low-rho end,
        where this method is expected to lose).
    ``n_dense_fallback_tail``  of those, how many went because the omitted-mass
        bound was not small enough.  **This is the count to watch**: it is the
        method admitting it could not justify its own truncation, and a run where it
        is not ~0 is a run where the enumeration is not doing its job.
    ``n_dense_fallback_structure``  rows exceeding ``MAX_INTERVALS``.
    ``n_intervals_total`` / ``n_local_points_total``  the work actually done.
    ``n_peaks_total``  enumerated maxima kept, over the peak-local rows.
    ``tail_bound_worst``  the worst (largest) ``bound - result`` among ACCEPTED
        rows, in nats.  A number near ``TAIL_LOG_TOL`` means the truncation is only
        just being justified.
    """
    return dict(_LAST_REPORT)


# --------------------------------------------------------------- the evaluator

def bandlimited_spectrum(x, xpy=np):
    """Spectral representation of the band-limited interpolant through rows of ``x``.

    Returns ``(Xw, fk)`` with

        x(t) = sum_j Xw[..., j] * exp(2 pi i * fk[j] * t / (n * deltaT))

    exact at ``t = i * deltaT`` for every integer ``i``, and equal to the unique
    band-limited interpolant that :func:`bandlimited_upsample` evaluates on a
    uniform refinement -- so the two agree wherever both are defined, which
    ``test_local_evaluator_matches_the_dense_upsample`` asserts on every production
    ``npts``.

    ODD ``n`` IS THE COMMON CASE and is the trap this function shares with
    :func:`bandlimited_upsample`: ``marginalization_time_grid(0.075, 1/srate)``
    returns 153 / 307 / 614 / 1228 / 2457 at srate 1024 / 2048 / 4096 / 8192 / 16384,
    odd at three of five.  The positive-frequency block is ``0 .. (n-1)//2``; putting
    the top positive bin at a negative frequency (which a split at ``n//2`` does for
    odd ``n``) leaves the reconstruction EXACT AT THE SAMPLES and wrong everywhere
    between them, so a "reproduces its input" test cannot see it.

    For even ``n`` the Nyquist bin is genuinely ambiguous -- ``+fNyq`` and ``-fNyq``
    are the same sequence on the samples -- so it is split evenly between the two,
    matching :func:`bandlimited_upsample`.  No interpolant can recover which it was;
    for rholm data the bin is empty anyway.
    """
    n = x.shape[-1]
    X = xpy.fft.fft(x, axis=-1) / float(n)
    n_pos = (n - 1) // 2
    pos = np.arange(0, n_pos + 1)
    if n % 2 == 0:
        neg = np.arange(n_pos + 2, n) - n
        fk = np.concatenate([pos, [n // 2], [-(n // 2)], neg]).astype(np.float64)
        half = 0.5 * X[..., n_pos + 1:n_pos + 2]
        Xw = xpy.concatenate([X[..., :n_pos + 1], half, half, X[..., n_pos + 2:]],
                             axis=-1)
    else:
        neg = np.arange(n_pos + 1, n) - n
        fk = np.concatenate([pos, neg]).astype(np.float64)
        Xw = X
    return Xw, xpy.asarray(fk)


def eval_bandlimited_uniform(Xw, fk, t0, dt_local, n_local, period, xpy=np):
    """Evaluate the interpolant on a per-row uniform grid ``t0[r] + m * dt_local[r]``.

    ``Xw, fk`` come from :func:`bandlimited_spectrum`; ``t0`` and ``dt_local`` are
    per-row (shape ``(n_rows,)``); ``period`` is ``npts * deltaT``.  Returns
    ``(n_rows, n_local)`` complex.

    THIS IS THE COST THE PROTOTYPE DID NOT PAY.  With the analytic ``kappa`` in hand
    a local time costs one closed-form call; here it costs a sum over the spectrum.
    The loop below is ``n_local`` steps of one complex multiply and one reduction
    over ``(n_rows, n_freq)``, i.e. ``O(n_rows * npts * n_local)`` and -- the property
    that makes the method worth having -- INDEPENDENT OF RHO, because ``n_local`` is
    fixed by ``W_SIGMA`` and ``UPSAMPLE_SAFETY`` while the dense path's point count
    grows linearly with it.

    No transcendental is evaluated per output point: the grid is uniform, so the
    phase advances by a constant factor.  It is re-anchored every
    ``_RECURRENCE_REANCHOR`` steps because ``exp(i x)`` is not exactly unit modulus.
    """
    n_rows = Xw.shape[0]
    two_pi_i = 2j * np.pi
    scale = fk[None, :] / float(period)
    z = xpy.exp(two_pi_i * scale * dt_local[:, None])
    out = xpy.empty((n_rows, n_local), dtype=Xw.dtype)
    acc = None
    for m in range(n_local):
        if acc is None or (m % _RECURRENCE_REANCHOR) == 0:
            acc = Xw * xpy.exp(two_pi_i * scale * (t0 + m * dt_local)[:, None])
        out[:, m] = xpy.sum(acc, axis=-1)
        acc = acc * z
    return out


# -------------------------------------------------------------- enumeration

def enumerate_peak_indices(q, xpy=np):
    """Boolean mask of INTERIOR local maxima of each row of ``q``.

    ``q`` is ``Re kappa`` on the enumeration grid, NOT ``lnL``.  Every callback in
    this path's scope is monotone increasing in it at fixed ``rho_sq``, and
    ``rho_sq`` is time-independent here, so a monotone map cannot move a maximum and
    the extrema of ``lnL`` are exactly the extrema of ``Re kappa``.  Enumerating here
    rather than on ``lnL`` keeps the callback -- a table interpolation, for the
    distance-marginalized case -- off the full time axis entirely.  (Phase
    marginalization would make the relevant quantity ``|kappa|``, which peaks
    elsewhere; it is refused by this path, see the module docstring.)

    The two comparisons are deliberately asymmetric (``>=`` left, ``>`` right): a
    plateau then yields exactly one index, its last, instead of none or all of them.

    Endpoints are excluded because a maximum AT the window edge is a statement that
    the window is mis-centred, which the inherited edge guard has already routed to
    the historical rule before this is ever called.
    """
    return (q[..., 1:-1] >= q[..., :-2]) & (q[..., 1:-1] > q[..., 2:])


def merge_intervals_by_row(rows, lo, hi, span):
    """Merge per-row interval lists into disjoint intervals, ALL ROWS AT ONCE.

    ``rows`` must be ascending; ``lo``/``hi`` are the interval ends and ``span`` an
    upper bound on ``hi``.  Returns ``(order, gid, g_row, g_lo, g_hi)``: the sort
    permutation, the merged-interval index each sorted input fell into, and the row,
    start and stop of each merged interval.  ``gid`` is what lets per-peak quantities
    -- the widths that set the local spacing -- be reduced over the merge.

    MERGING IS NOT AN OPTIMISATION.  Two overlapping windows integrated separately
    both contain the shared region, so the log-sum-exp of the parts double-counts it.
    On a broad integrand at rho ~ 6 that is +1.6 nats -- measured, and reproduced by
    ``test_unmerged_intervals_double_count``.  It is also what makes this ONE
    algorithm rather than a regime switch: as peaks crowd together the union grows
    continuously to the whole window and the method becomes the dense grid, with no
    threshold anywhere.

    Vectorised across rows deliberately, not for elegance: a broad integrand has MANY
    enumerated peaks per row, and a Python loop over them made this rule slower than
    the dense path it delegates to (measured 0.69x at sigma_t/deltaT = 0.17, on a
    block where every row fell back anyway).  The running maximum is restarted at
    each row boundary by offsetting row ``r`` by ``r * big`` -- every value in row
    ``r-1`` is then below every value in row ``r``, so one global
    ``maximum.accumulate`` gives the per-row running maximum with no segmentation.
    """
    order = np.lexsort((lo, rows))
    r_s, lo_s, hi_s = rows[order], lo[order], hi[order]
    big = 2.0 * (float(span) + 1.0)
    cummax = np.maximum.accumulate(hi_s + r_s * big) - r_s * big
    new = np.ones(r_s.size, dtype=bool)
    new[1:] = (r_s[1:] != r_s[:-1]) | (lo_s[1:] > cummax[:-1])
    gid = np.cumsum(new) - 1
    heads = np.nonzero(new)[0]
    tails = np.append(heads[1:] - 1, r_s.size - 1)
    return order, gid, r_s[heads], lo_s[heads], cummax[tails]


# ------------------------------------------------------------------ the rule

def _peak_curvature_sigma(lnL_stencil, h, xpy=np):
    """``sigma`` per peak from a widening centred second difference of ``lnL``.

    ``lnL_stencil`` is ``(n_peaks, 2*maxd+1)``, the callback evaluated on the
    enumeration grid at offsets ``-maxd .. +maxd`` about each peak.  Same estimator
    and same widening rationale as
    :func:`time_marginalization_quadrature.peak_width_from_lnL`: the second
    difference of a parabola is its second derivative at ANY spacing, so an
    under-resolved peak still reports its own width honestly, and stepping the
    stencil out over a ``-inf`` hole (the distance-marginalized callback returns
    ``-inf`` outside its table) costs nothing.  A peak with no finite negative
    curvature at any half-width gets ``inf`` and is not given an interval -- which
    routes it into the tail bound rather than into a guess.
    """
    maxd = (lnL_stencil.shape[-1] - 1) // 2
    centre = lnL_stencil[:, maxd]
    sigma = xpy.full(lnL_stencil.shape[0], np.inf, dtype=np.float64)
    done = xpy.zeros(lnL_stencil.shape[0], dtype=bool)
    for d in CURVATURE_STENCIL_HALFWIDTHS:
        if d > maxd:
            break
        with np.errstate(invalid='ignore'):
            d2 = (lnL_stencil[:, maxd - d] - 2.0 * centre
                  + lnL_stencil[:, maxd + d]) / float(d * h) ** 2
        fresh = xpy.isfinite(d2) & (~done)
        neg = fresh & (d2 < 0)
        sigma = xpy.where(neg, 1.0 / xpy.sqrt(xpy.where(neg, -d2, 1.0)), sigma)
        done = done | fresh
        if bool(xpy.all(done)):
            break
    return sigma


#: Fewest local points any row can ever need: one interval of half-width
#: ``W_SIGMA * sigma`` at spacing ``sigma / UPSAMPLE_SAFETY`` is
#: ``2 * W_SIGMA * UPSAMPLE_SAFETY`` sub-intervals however large ``sigma`` is, and
#: merging or a coarser-capped spacing can only ADD points.  So this is a genuine
#: lower bound, which is what lets it be used to reject a row BEFORE any work is
#: done for it -- not an estimate.
MIN_LOCAL_POINTS = int(2 * W_SIGMA * UPSAMPLE_SAFETY) + 1


def _estimated_costs(n_local_total, npts, factor):
    """(local, dense) cost estimates for one row, in complex-multiply units.

    The local evaluator is a spectral sum advanced by a recurrence: one complex
    multiply and one add per (frequency, output point), hence ``npts *
    n_local_total``.  The dense path is a zero-padded FFT of length ``npts*factor``,
    hence ``npts*factor*log2(npts*factor)``, and it additionally pays the likelihood
    callback and an ``exp`` on every one of those points while the local path pays
    them on ``n_local_total`` -- so this estimate is CONSERVATIVE against the local
    path (it counts the arithmetic the two share and ignores the per-point cost the
    local path avoids).

    It decides only WHICH of two paths runs, and both satisfy the same derived
    resolution criterion, so it cannot trade accuracy for cost.
    """
    dense_n = npts * np.asarray(factor, dtype=np.float64)
    return (npts * np.asarray(n_local_total, dtype=np.float64),
            dense_n * np.maximum(np.log2(dense_n), 1.0))


def _log_trapz_local(lnL_loc, h, xpy=np):
    """``log \\int exp(lnL) dt`` by trapezoid on one uniform local grid, per row.

    Trapezoid, not Simpson, for the reason the dense path gives: on a peak that has
    decayed to nothing inside the interval every Euler-Maclaurin boundary term
    vanishes and the trapezoidal rule is spectrally accurate, while Simpson's
    ``(4 T_h - T_2h)/3`` reintroduces the ``2h`` alias that is the original defect.
    The offset is per row and taken on this grid; the result is offset-invariant.
    """
    w = xpy.full(lnL_loc.shape[-1], 1.0, dtype=np.float64)
    w[0] = 0.5
    w[-1] = 0.5
    off = _safe_offset(xpy.max(lnL_loc, axis=-1, keepdims=True), xpy=xpy)
    return off[..., 0] + xpy.log(xpy.sum(xpy.exp(lnL_loc - off) * w, axis=-1)) \
        + xpy.log(h)


def _logaddexp_reduce(parts, xpy=np):
    """``log sum_j exp(parts[:, j])``, NaN-safe for an all ``-inf`` row.

    Rows with no intervals at all are all ``-inf`` here by construction -- they are
    the ones being handed to the dense path -- so ``log(0)`` is the CORRECT answer for
    them and its warning is noise, not a signal.  The offset guard is what keeps that
    case at ``-inf`` instead of ``NaN``.
    """
    off = _safe_offset(xpy.max(parts, axis=-1, keepdims=True), xpy=xpy)
    with np.errstate(divide='ignore'):
        return off[..., 0] + xpy.log(xpy.sum(xpy.exp(parts - off), axis=-1))


def time_marginalize_peak_local(kappa, rho_sq, deltaT, loglikelihood,
                                phase_marginalization=False, simps=None,
                                lnL_coarse=None, xpy=np, return_peaks=False):
    """``log \\int dt exp(lnL(t))``, refining only around the enumerated peaks.

    Signature, preconditions, row classification and fallback semantics are those of
    :func:`time_marginalization_quadrature.time_marginalize_bandlimited`, which see;
    the parameters mean the same things and the same rows fall back to the caller's
    Simpson rule for the same reasons.  What changes is the rule applied to the rows
    that ARE refined.

    ``return_peaks=True`` additionally returns a list, one entry per input row, of
    ``(t_star, sigma)`` arrays for the peaks that were enumerated and kept --
    ``None`` for a row this rule did not handle.  These are the same ``t_star`` a
    time-first reordering of the marginalizations would need, and they are
    distance- and callback-independent (see the module docstring), so they are
    exposed as an output rather than kept as a temporary.
    """
    if phase_marginalization:
        # Deliberate scope cut, not an oversight -- see the module docstring.  Refuse
        # rather than silently running something else.
        raise NotImplementedError(
            "time_marginalize_peak_local does not support phase marginalization: the "
            "Laplace width of the time peak then carries an (I1/I0)(|kappa|/D) factor "
            "that does not reduce, so the local spacing is no longer derivable from "
            "rho_sq and the curvature alone.  Production marginalizes over distance, "
            "not phase.  Use time_quadrature='bandlimited', which does support it.")

    simps = _default_simps(simps, xpy)

    kappa = xpy.asarray(kappa)
    rho_sq = xpy.asarray(rho_sq)
    npts = kappa.shape[-1]
    n_rows = kappa.shape[0]
    deltaT = float(deltaT)
    period = npts * deltaT
    t_last = (npts - 1) * deltaT

    _require_time_independent_rho_sq(rho_sq, xpy=xpy, rule='peak-local')
    rho_col = rho_sq[..., :1]

    _term = lambda k: k.real            # phase marginalization is refused above
    if lnL_coarse is None:
        lnL_coarse = loglikelihood(_term(kappa), rho_sq)

    sigma, jmax, measurable = peak_width_from_lnL(lnL_coarse, deltaT, xpy=xpy)
    guard = max(1, int(npts * EDGE_GUARD_FRACTION))
    has_peak = measurable & xpy.isfinite(sigma)
    flat = measurable & (~xpy.isfinite(sigma))
    exposed = has_peak & ((jmax < guard) | (jmax > npts - 1 - guard))
    unmeasurable = ~measurable
    factors = xpy.maximum(required_upsample_factors(sigma, deltaT, xpy=xpy), 1)
    refined = (~(exposed | unmeasurable)) & (factors > 1)

    out = _log_simps_rows(lnL_coarse, deltaT, simps, xpy=xpy)
    peaks_out = [None] * n_rows if return_peaks else None

    stats = dict(n_peak_local_rows=0, n_dense_fallback_cost=0,
                 n_dense_fallback_tail=0, n_dense_fallback_structure=0,
                 n_intervals_total=0, n_local_points_total=0, n_peaks_total=0,
                 tail_bound_worst=-np.inf)
    idx_all = np.asarray(xpy.where(refined)[0] if xpy is np
                         else xpy.where(refined)[0].get())
    n_dense = 0
    if idx_all.size:
        per_row = npts * PEAK_ENUM_FACTOR * 16 * 4
        chunk = max(1, min(int(idx_all.size), int(_CHUNK_BYTES // max(per_row, 1))))
        dense_rows = []
        for start in range(0, int(idx_all.size), chunk):
            sel = idx_all[start:start + chunk]
            sel_x = xpy.asarray(sel)
            vals, ok, peaks = _peak_local_chunk(
                kappa[sel_x], rho_col[sel_x], factors[sel_x], npts, deltaT, period,
                t_last, loglikelihood, _term, stats, xpy=xpy, want_peaks=return_peaks)
            if ok.any():
                got = xpy.asarray(np.where(ok)[0])
                out[xpy.asarray(sel[ok])] = vals[got]
            if return_peaks:
                for j in np.where(ok)[0]:
                    peaks_out[int(sel[j])] = peaks[int(j)]
            if (~ok).any():
                dense_rows.append(sel[~ok])
        if dense_rows:
            di = np.concatenate(dense_rows)
            di_x = xpy.asarray(di)
            # The dense band-limited path is the BACKSTOP, deliberately: a row this
            # rule declines is given a value from the reviewed reference
            # implementation, not an approximation with a caveat attached.
            out[di_x] = time_marginalize_bandlimited(
                kappa[di_x], rho_sq[di_x], deltaT, loglikelihood,
                phase_marginalization=phase_marginalization, simps=simps,
                lnL_coarse=lnL_coarse[di_x], xpy=xpy)
            n_dense = int(di.size)

    _LAST_REPORT.clear()
    _LAST_REPORT.update(
        n_rows=n_rows,
        n_wrap_exposed_rows=int(xpy.sum(exposed)),
        n_unmeasurable_rows=int(xpy.sum(unmeasurable)),
        n_flat_rows=int(xpy.sum(flat)),
        n_refined_rows=int(xpy.sum(refined)),
        n_dense_fallback_rows=n_dense,
        **stats)
    if return_peaks:
        return out, peaks_out
    return out


def _host(a, xpy=np):
    """Copy a device array to host numpy.  The RAGGED bookkeeping -- which peaks
    belong to which row, how their intervals merge -- is scalar work on a handful of
    values per row, and doing it on the host keeps one implementation instead of two.
    Everything whose size scales (the enumeration grid, the spectrum, the local
    evaluation and the callback) stays on the device."""
    return np.asarray(a) if xpy is np else np.asarray(a.get())


def _peak_local_chunk(kappa_rows, rho_col_rows, factors, npts, deltaT, period,
                      t_last, loglikelihood, _term, stats, xpy=np, want_peaks=False):
    """Peak-local integration of one chunk of refined rows.

    Returns ``(values, ok, peaks)``; ``ok[r]`` is False for a row that must be given
    to the dense path -- because it had no usable enumerated peak, because it had
    more disjoint structure than ``MAX_INTERVALS``, because its estimated cost was
    worse here than there, or because the omitted-mass bound could not be met.
    """
    n_rows = kappa_rows.shape[0]
    h_enum = deltaT / PEAK_ENUM_FACTOR
    last = (npts - 1) * PEAK_ENUM_FACTOR

    values = xpy.full(n_rows, -np.inf, dtype=np.float64)
    ok = np.zeros(n_rows, dtype=bool)
    peaks = [None] * n_rows
    factors_np = _host(factors, xpy)

    # ---- gate 1, before any work is done for the row.  MIN_LOCAL_POINTS is a lower
    # bound on this rule's point count, so a row that already loses on it can never
    # win, and enumerating it would be pure waste -- measured: without this gate a
    # block whose rows all fall back still paid 0.66 s of enumeration on top of the
    # dense path it ended up using anyway.  A gate applied only AFTER enumeration
    # makes the method slower than the path it delegates to, which is the opposite
    # of the point.
    c_lo, c_dn = _estimated_costs(MIN_LOCAL_POINTS, npts, factors_np)
    viable = c_lo < c_dn
    stats['n_dense_fallback_cost'] += int(np.sum(~viable))
    if not viable.any():
        return values, ok, peaks

    # ---- enumeration.  One FFT upsample of kappa at a FIXED, SNR-independent
    # factor.  The callback is NOT evaluated on this grid: only term(kappa) is
    # needed, because a monotone callback cannot move an extremum.
    k_up = bandlimited_upsample(kappa_rows, PEAK_ENUM_FACTOR, xpy=xpy)[..., :last + 1]
    q_up = _term(k_up)
    del k_up
    n_enum = q_up.shape[-1]

    mask = enumerate_peak_indices(q_up, xpy=xpy)
    mask = mask & xpy.asarray(viable)[:, None]
    rows_p, cols_p = xpy.where(mask)
    cols_p = cols_p + 1                       # the mask covers interior points only
    if int(rows_p.shape[0]) == 0:
        return values, ok, peaks

    # ---- widths.  These need lnL, so the callback runs HERE -- on a short stencil
    # about each peak, never on the full time axis.  The stencil CENTRE is clipped
    # inward rather than the offsets, so it stays centred; the second difference of a
    # parabola is its second derivative at any spacing and about any centre, which is
    # the same property peak_width_from_lnL relies on.
    maxd = max(CURVATURE_STENCIL_HALFWIDTHS)
    if 2 * maxd >= n_enum:
        return values, ok, peaks
    centre = xpy.clip(cols_p, maxd, n_enum - 1 - maxd)
    take = centre[:, None] + xpy.arange(-maxd, maxd + 1)[None, :]
    q_st = q_up[rows_p[:, None], take]
    lnL_st = loglikelihood(q_st, xpy.broadcast_to(rho_col_rows[rows_p], q_st.shape))
    sigma_pk = _peak_curvature_sigma(lnL_st, h_enum, xpy=xpy)
    lnL_pk = loglikelihood(q_up[rows_p, cols_p], rho_col_rows[rows_p, 0])

    rows_np = _host(rows_p, xpy)
    cols_np = _host(cols_p, xpy)
    sig_np = _host(sigma_pk, xpy)
    val_np = _host(lnL_pk, xpy)

    # ---- drop peaks that cannot carry representable mass, and peaks with no
    # resolvable curvature.  Both drops are SAFE rather than hopeful: what is dropped
    # then lies outside the intervals and so enters the tail bound below.
    row_best = np.full(n_rows, -np.inf)
    np.maximum.at(row_best, rows_np, val_np)
    keep = np.isfinite(sig_np) & (val_np > row_best[rows_np] - PEAK_KEEP_NATS)
    rows_np, cols_np, sig_np = rows_np[keep], cols_np[keep], sig_np[keep]
    if rows_np.size == 0:
        return values, ok, peaks

    t_np = cols_np * h_enum
    lo_np = np.maximum(t_np - W_SIGMA * sig_np, 0.0)
    hi_np = np.minimum(t_np + W_SIGMA * sig_np, t_last)
    bounds = np.searchsorted(rows_np, np.arange(n_rows + 1))

    # ---- merge, and derive each merged interval's own spacing.  All rows at once:
    # the per-peak Python loop this replaces was the reason the rule could come out
    # slower than the dense path it delegates to.
    order, gid, g_row, g_lo, g_hi = merge_intervals_by_row(rows_np, lo_np, hi_np,
                                                           t_last)
    n_groups = g_row.size
    # Spacing per merged interval: set by the SHARPEST peak inside it, and never
    # coarser than the grid on which the structure was established.
    s_min = np.full(n_groups, np.inf)
    np.minimum.at(s_min, gid, sig_np[order])
    h_want = np.minimum(s_min / UPSAMPLE_SAFETY, h_enum)
    n_loc = np.maximum(3, np.ceil((g_hi - g_lo) / h_want).astype(np.int64) + 1)

    n_iv_row = np.bincount(g_row, minlength=n_rows)
    n_loc_row = np.bincount(g_row, weights=n_loc, minlength=n_rows)
    c_local, c_dense = _estimated_costs(n_loc_row, npts, factors_np)

    too_much = n_iv_row > MAX_INTERVALS
    too_slow = (~too_much) & (n_iv_row > 0) & (c_local >= c_dense)
    stats['n_dense_fallback_structure'] += int(np.sum(too_much))
    stats['n_dense_fallback_cost'] += int(np.sum(too_slow))
    keep_row = (n_iv_row > 0) & (~too_much) & (~too_slow)

    gbounds = np.searchsorted(g_row, np.arange(n_rows + 1))
    plan = []
    for r in np.nonzero(keep_row)[0]:
        ga, gb = int(gbounds[r]), int(gbounds[r + 1])
        plan.append((int(r), g_lo[ga:gb], g_hi[ga:gb], int(n_loc[ga:gb].max()),
                     int(bounds[r]), int(bounds[r + 1])))

    if not plan:
        return values, ok, peaks

    Xw, fk = bandlimited_spectrum(kappa_rows, xpy=xpy)

    # ---- batched evaluation.  Rows are grouped by (interval count, point-count
    # bucket) so padding to a common shape can waste at most a factor of two, and
    # every interval slot of a group is one batched call.
    covered = np.zeros((n_rows, n_enum), dtype=bool)
    parts = xpy.full((n_rows, MAX_INTERVALS), -np.inf, dtype=np.float64)
    buckets = {}
    for entry in plan:
        key = (entry[1].size, int(2 ** np.ceil(np.log2(max(entry[3], 2)))))
        buckets.setdefault(key, []).append(entry)

    for (n_iv, m_pad), members in buckets.items():
        rr = np.array([m[0] for m in members])
        rr_x = xpy.asarray(rr)
        for j in range(n_iv):
            a_h = np.array([m[1][j] for m in members], dtype=np.float64)
            b_h = np.array([m[2][j] for m in members], dtype=np.float64)
            h_h = (b_h - a_h) / float(m_pad - 1)
            # A zero-length merged interval (a peak pinned against a window end)
            # would give h=0 and a degenerate grid; give it the enumeration spacing
            # so the trapezoid has a domain.  Whatever it then misses is bounded by
            # the tail check like everything else.
            h_h = np.where(h_h > 0, h_h, h_enum)
            k_loc = eval_bandlimited_uniform(Xw[rr_x], fk, xpy.asarray(a_h),
                                             xpy.asarray(h_h), m_pad, period, xpy=xpy)
            lnL_loc = loglikelihood(
                _term(k_loc), xpy.broadcast_to(rho_col_rows[rr_x], k_loc.shape))
            parts[rr_x, j] = _log_trapz_local(lnL_loc, xpy.asarray(h_h), xpy=xpy)
            stats['n_local_points_total'] += int(rr.size) * m_pad
            for i_m, m in enumerate(members):
                lo_i = int(np.ceil(a_h[i_m] / h_enum))
                hi_i = int(np.floor(b_h[i_m] / h_enum))
                if hi_i >= lo_i:
                    covered[m[0], max(lo_i, 0):hi_i + 1] = True
        stats['n_intervals_total'] += int(rr.size) * n_iv

    result = _logaddexp_reduce(parts, xpy=xpy)

    # ---- the tail bound.  log(T_outside) + max_{outside} lnL is an upper bound on
    # the omitted integral.  It is evaluated on term(kappa) -- one callback value per
    # row -- because the callback is monotone in it, so no evaluation on the full
    # time axis is needed.  A row whose bound is not small enough is NOT reported
    # with a caveat: it goes to the dense path.
    cov_x = xpy.asarray(covered)
    q_out_max = xpy.max(xpy.where(cov_x, -np.inf, q_up), axis=-1)
    n_out = _host(xpy.sum(~cov_x, axis=-1), xpy).astype(np.float64)
    lnL_out = loglikelihood(q_out_max, rho_col_rows[:, 0])
    with np.errstate(divide='ignore', invalid='ignore'):
        bound = np.where(n_out > 0, np.log(np.maximum(n_out * h_enum, 1e-300))
                         + _host(lnL_out, xpy), -np.inf)
        margin = bound - _host(result, xpy)

    planned = np.array([m[0] for m in plan])
    accepted = planned[margin[planned] < TAIL_LOG_TOL]
    rejected = planned[~(margin[planned] < TAIL_LOG_TOL)]
    stats['n_dense_fallback_tail'] += int(rejected.size)
    if accepted.size:
        acc_x = xpy.asarray(accepted)
        values[acc_x] = result[acc_x]
        ok[accepted] = True
        stats['tail_bound_worst'] = max(stats['tail_bound_worst'],
                                        float(margin[accepted].max()))
        stats['n_peak_local_rows'] += int(accepted.size)
        stats['n_peaks_total'] += int(np.isin(rows_np, accepted).sum())

    if want_peaks:
        for r, starts, stops, m_max, a, b in plan:
            if ok[r]:
                peaks[r] = (t_np[a:b].copy(), sig_np[a:b].copy())

    return values, ok, peaks
