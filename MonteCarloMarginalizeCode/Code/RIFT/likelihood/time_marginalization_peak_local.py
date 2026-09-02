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
requirements that are not the same requirement.  There are in fact THREE, and the
first version of this module named only two -- which is exactly where its one
correctness bug lived:

* Resolving ``kappa(t)`` enough to **enumerate its extrema**.  ``kappa`` is
  band-limited at Nyquist by construction, so the narrowest feature it can have is
  a half-cycle of width ``deltaT``.  Enumerating its extrema therefore needs a
  small FIXED factor and is **SNR-INDEPENDENT**.
* **Localising** each enumerated extremum to a fraction of ``sigma_t``.  Enumeration
  returns a grid INDEX, and an index is not a location: the crest can lie
  ``h_enum/2`` from the sample that reports it.  This requirement is
  **SNR-DEPENDENT**, and it belongs to neither of the other two -- which is why it
  went missing.  Building the interval around the sample instead drops the peak
  entirely once ``W_SIGMA * sigma_t < h_enum/2`` and cost up to **165 nats**; see
  :data:`LOCALISE_SAFETY`.  It is met by Newton on the spectral interpolant, which
  converges quadratically, so its cost is logarithmic in rho rather than linear.
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
switch is a cost decision only -- it never substitutes an approximation for a value, it
chooses which of two paths computes it.

Be precise about the sense in which the two branches agree, because the looser statement
this module used to make was false.  The dense path ENFORCES its criterion: it remeasures
the width on the grid it actually integrated and doubles until the criterion holds there.
This path has no such loop.  It DERIVES the local spacing from the coarse width and then
verifies the outcome two other ways -- the localisation must converge to
``LOCALISE_SAFETY * sigma_t``, and the local grids must ATTAIN the localised crest's
``lnL`` to within ``CONTAINMENT_SLACK_NATS``.  Those are a-posteriori checks rather than
an enforcement loop, and a row failing either goes to the dense path.  So both branches
are checked and neither returns a number it cannot defend, but they are NOT the same
criterion and this module should not say they are.

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

from . import time_marginalization_quadrature as _tmq
from .time_marginalization_quadrature import (
    UPSAMPLE_SAFETY,
    CURVATURE_STENCIL_HALFWIDTHS,
    bandlimited_upsample,
    reflected_bandlimited_upsample,
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
    "LOCALISE_SAFETY",
    "CONTAINMENT_SLACK_NATS",
    "localise_peaks",
    "bandlimited_spectrum",
    "spectral_curvature_bound",
    "spectral_derivative_bound",
    "crest_upper_bound",
    "enum_grid_derivatives",
    "eval_bandlimited_uniform",
    "eval_bandlimited_points",
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

#: Localise each enumerated crest to this fraction of its own ``sigma_t``, and widen
#: its interval by the same amount.
#:
#: THIS IS THE THIRD RESOLUTION REQUIREMENT, and the first version of this module did
#: not have it.  The design names two -- resolve ``kappa`` to ENUMERATE, resolve
#: ``exp(lnL)`` to INTEGRATE -- but enumeration returns a grid INDEX, and an index is
#: not a location.  The true crest can be up to ``h_enum/2`` from the sample that
#: reports it, while the interval half-width is ``W_SIGMA * sigma_t``, so whenever
#:
#:     W_SIGMA * sigma_t  <  h_enum / 2
#:
#: the peak falls entirely OUTSIDE its own interval.  At ``PEAK_ENUM_FACTOR = 8`` and
#: ``W_SIGMA = 12`` that is ``sigma_t/deltaT < 0.0052``, i.e. any row whose derived
#: factor reaches 512.  MEASURED on the synthetic fixture at ``sigma_t/deltaT =
#: 0.0024``, error against the dense path as the crest is walked off the enumeration
#: grid: **0.00 nats at offset 0, -6.52 at h_enum/4, -164.93 at h_enum/2**.  Always
#: negative -- it silently deletes mass -- and the old tail bound reported -275 nats
#: while dropping 165.
#:
#: Unlike the other two requirements this one is SNR-DEPENDENT: the crest must be found
#: to a fraction of ``sigma_t``, and ``sigma_t`` shrinks as ``1/rho``.  It is met by
#: Newton on the spectral interpolant (see :func:`localise_peaks`), which converges
#: quadratically, so its cost grows only logarithmically.
#:
#: Value: 0.25 puts the residual at ``sigma/4``, so widening by it costs 2% of the
#: interval, while the trapezoid error it can induce is ``exp(-(12)^2/2)``-scale --
#: nothing.  It is not an accuracy knob that can be set too small: convergence to this
#: tolerance is ASSERTED, and a row that misses it goes to the dense path.
LOCALISE_SAFETY = 0.25

#: Newton iterations allowed per peak.  From a parabolic seed the initial error is a few
#: percent of ``h_enum`` and convergence is quadratic, so 2-4 is typical; the cap exists
#: so a pathological row FAILS (and is handed to the dense path) rather than spinning.
LOCALISE_MAX_ITER = 16

#: A row is accepted only if its local integration grids actually ATTAIN the localised
#: crest value, to within this many nats.  The a-posteriori half of the F1 fix, and it
#: costs nothing -- the maximum over each local grid is already computed for the
#: log-sum-exp offset.
#:
#: Why the slack is safe and why it is small: the local spacing is at most
#: ``sigma/UPSAMPLE_SAFETY = sigma/2``, so the grid's nearest point to the crest is
#: within ``sigma/4``, i.e. its ``lnL`` is within ``1/32`` of the crest's.  0.5 nats is
#: a 16x margin on that, and is still 13x smaller than the smallest miss F1 produced
#: (-6.52 nats).
#:
#: NOTE this compares against ``lnL`` at the LOCALISED crest, not at the enumeration
#: sample.  Comparing against the sample cannot work and it is worth saying why: when
#: the crest sits off-grid the sample is already tens of nats below it, so a check
#: against the sample passes precisely in the case it is meant to catch.
#:
#: SCOPE, stated because it is narrower than it looks: ``row_star`` is a per-row MAXIMUM
#: over the peaks that survived the keep filter.  So this verifies that the DOMINANT
#: crest was reached; it does not independently verify each secondary crest, and a peak
#: dropped by the keep filter can never enter it at all.  That is tolerable only because
#: the keep filter now compares crests (G1) and its own magnitude argument stands on its
#: own -- not because this check covers it.
CONTAINMENT_SLACK_NATS = 0.5

#: Local interval half-width, in units of the peak's own ``sigma_t``.  A Gaussian
#: peak truncated at ``W_SIGMA`` sigma omits ``erfc(W/sqrt2) ~ exp(-W^2/2)`` of its
#: mass: ``exp(-72) = 5.4e-32`` here, which is below double precision against the
#: largest window-to-sigma dynamic range this path can see (``UPSAMPLE_FACTOR_MAX``
#: bounds it at ~1e4).
#:
#: ``erfc(W/sqrt2)`` is a statement about a Gaussian truncated symmetrically ABOUT ITS
#: CREST, so it is only a truncation bound if the interval is actually centred there.
#: The first version of this module centred on the enumeration SAMPLE, which left an
#: unstated precondition ``W_SIGMA * sigma_t >= h_enum/2`` -- coupling ``W_SIGMA`` to
#: ``PEAK_ENUM_FACTOR``, violated by every sharp row, and nowhere asserted.  It is
#: discharged, not asserted: :func:`localise_peaks` finds the crest to
#: ``LOCALISE_SAFETY * sigma_t`` and the interval is widened by that residual, so the
#: two constants are decoupled and raising ``PEAK_ENUM_FACTOR`` is no longer the
#: dangerous move it used to be.
W_SIGMA = 12.0

#: Enumerated peaks more than this far below a row's highest CREST -- not its highest
#: sample; see the G1 note in ``_peak_local_chunk`` -- are dropped before intervals are
#: built.
#:
#: The justification is DIRECT and does not go through the tail bound.  An earlier
#: version of this docstring said a dropped peak was safe "because it enters the tail
#: bound", and that claim is vacuous: since ``q_out_max`` is at least the dropped peak's
#: own sample, the bound accepts the row unless ``log(T_out / 2.5 sigma) >= 37``, i.e.
#: ``sigma_t < 2.6e-18 s``, while ``UPSAMPLE_FACTOR_MAX`` bounds the sharpest legal row
#: at ``sigma_t = 3.0e-08 s`` -- ten orders of magnitude away.  For EVERY row this
#: module can legally handle, a keep-filter drop is automatically accepted by the bound.
#: The bound cannot backstop this filter and must not be cited as if it could.
#:
#: What justifies it instead is magnitude, crest to crest: a peak ``PEAK_KEEP_NATS``
#: below the highest crest contributes ``exp(-60) = 8.8e-27`` of its mass per unit
#: width, and the widest window-to-sigma ratio this module can reach is ``T/sigma_min
#: = 2.5e6`` (again from ``UPSAMPLE_FACTOR_MAX``), so the omitted relative mass is below
#: ``2e-20`` -- under double precision.  That inequality is asserted in the suite,
#: because it ties this constant to ``UPSAMPLE_FACTOR_MAX`` and neither is free.
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
    ``n_dense_fallback_cost_pregate`` / ``n_dense_fallback_cost``  cost declines, split
        by WHICH gate fired: before enumeration (from a point-count floor) and after it
        (from the merged intervals).  Kept apart because only the first can prevent work
        being done, and a single shared counter made the pre-gate removable without any
        test noticing.
    ``n_dense_fallback_nopeak``  rows that passed the gates but enumerated no usable
        peak.  Exists so the sub-counts RECONCILE with ``n_dense_fallback_rows``.
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


def eval_bandlimited_points(Xw, fk, rows, t, period, xpy=np, point_chunk=1024):
    """``q``, ``q'`` and ``q''`` at arbitrary ``(row, time)`` pairs.

    The uniform-grid evaluator above cannot be used for the omitted-mass bound: the
    points that matter there are the ENDS OF THE MERGED INTERVALS, which are wherever
    localisation put them and are not on any grid.  Cost is one exponential array per
    point and ``O(npts)`` per point, and the caller uses at most ``2 * MAX_INTERVALS``
    of them per row, so this is negligible against the local grids.

    Same three sums as :func:`localise_peaks` -- the derivatives are the spectral sum
    with ``w_j`` and ``w_j**2`` folded in -- and chunked over points for the same
    reason: the temporary is ``(n_points, n_freq)``.
    """
    w = (2j * np.pi / float(period)) * fk
    n_pt = int(t.shape[0])
    q0 = xpy.zeros(n_pt, dtype=np.float64)
    q1 = xpy.zeros(n_pt, dtype=np.float64)
    q2 = xpy.zeros(n_pt, dtype=np.float64)
    for a in range(0, n_pt, point_chunk):
        b = min(a + point_chunk, n_pt)
        E = Xw[rows[a:b]] * xpy.exp(w[None, :] * t[a:b][:, None])
        q0[a:b] = xpy.sum(E, axis=-1).real
        q1[a:b] = xpy.sum(E * w[None, :], axis=-1).real
        q2[a:b] = xpy.sum(E * (w * w)[None, :], axis=-1).real
    return q0, q1, q2


def enum_grid_derivatives(Xw, fk, factor, n_keep, period, xpy=np):
    """``q'`` and ``q''`` on the ENUMERATION grid, by FFT, from the same spectrum.

    The enumeration grid is ``m * period / (n * factor)`` for ``m = 0 .. n_keep-1``,
    which is exactly the grid :func:`bandlimited_upsample` produces, so placing
    ``Xw_j * w_j**k`` at bin ``fk_j mod (n*factor)`` and inverse-transforming gives the
    ``k``-th derivative of the SAME interpolant on the SAME points.  Two transforms of
    the length the enumeration upsample already uses; the alternative -- evaluating the
    spectral sum pointwise -- is ``O(npts)`` per point and would cost more than the
    integration it is protecting.

    Differencing ``q`` on the grid would NOT do: a difference of an under-resolved
    sample sequence is an estimate, and everything downstream of these arrays is an
    inequality.
    """
    n = int(Xw.shape[-1])
    n_pad = n * int(factor)
    w = (2j * np.pi / float(period)) * fk
    idx = xpy.asarray(_host(fk, xpy).astype(np.int64) % n_pad)
    coef = Xw * w[None, :]
    out = []
    for _ in range(2):
        pad = xpy.zeros(Xw.shape[:-1] + (n_pad,), dtype=coef.dtype)
        pad[..., idx] = coef
        out.append((xpy.fft.ifft(pad, axis=-1)[..., :n_keep] * float(n_pad)).real)
        del pad
        coef = coef * w[None, :]
    return out[0], out[1]


# -------------------------------------------------------------- enumeration

def spectral_derivative_bound(Xw, fk, period, order, xpy=np):
    """``max_t |q^(order)(t)|`` for the interpolant, bounded rather than estimated.

    ``q(t) = Re sum_j Xw_j exp(w_j t)`` with ``w_j = 2 pi i fk_j / period``, so
    ``|q^(k)| <= sum_j |Xw_j| |w_j|^k`` everywhere, by the triangle inequality.  One
    reduction over the spectrum per row; nothing is fitted and no shape is assumed.
    """
    w = xpy.abs(2.0 * np.pi * xpy.asarray(fk) / float(period)) ** int(order)
    return xpy.sum(xpy.abs(Xw) * w[None, :], axis=-1)


def spectral_curvature_bound(Xw, fk, period, xpy=np):
    """``max_t |q''(t)|`` for the interpolant, bounded rather than estimated.

    The ``order = 2`` case of :func:`spectral_derivative_bound`, kept under its own
    name because the crest pre-filter is the one caller that must not be read as
    depending on anything else.
    """
    return spectral_derivative_bound(Xw, fk, period, 2, xpy=xpy)


def crest_upper_bound(q_at_peak, q_ddot_max, h_enum):
    """Upper bound on a crest, from its enumeration SAMPLE and a bound on ``|q''|``.

    Expanding about the crest ``t*``, where ``q'`` vanishes by definition,

        q(t_s) = q(t*) + q''(xi) (t_s - t*)^2 / 2,   |t_s - t*| <= h_enum

    so ``q(t*) <= q(t_s) + q_ddot_max * h_enum^2 / 2``.  A Taylor remainder with a TRUE
    bound on the second derivative -- not a parabolic fit through three samples, which is
    what the previous version of this was and which is why it was not a bound: ``lnL`` is
    not quadratic across an enumeration cell, and the anharmonic part of the deficit
    carries the same ``1/sigma^2`` amplification as the quadratic part.  Measured at
    derived factor 1024, the pure quantisation excess is 4.4 nats and the true shortfall
    was 122.3.

    ``h_enum``, not ``h_enum/2``: the localiser's bracket is ``+/- h_enum`` and
    displacements up to 0.959 of it have been observed.
    """
    return q_at_peak + 0.5 * q_ddot_max * h_enum ** 2


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

    ENDPOINTS ARE INCLUDED, and the reason is worth recording because an earlier version
    excluded them on a justification that was simply false.  That version said a maximum
    at the window edge means the window is mis-centred and the inherited edge guard has
    already routed such rows away.  It has not: ``exposed`` keys on the row's GLOBAL
    coarse argmax, so a row whose dominant peak sits comfortably mid-window is refined
    even when it also carries a SECONDARY maximum hard against an edge.  That secondary
    peak was then never enumerated, never covered, and -- because the outside maximum is
    sampled on this same grid -- under-represented in the tail bound too.
    """
    interior = (q[..., 1:-1] >= q[..., :-2]) & (q[..., 1:-1] > q[..., 2:])
    left = (q[..., :1] > q[..., 1:2])
    right = (q[..., -1:] > q[..., -2:-1])
    return _cat_last(left, interior, right, xpy=xpy)


def _cat_last(*parts, **kw):
    xpy = kw.get('xpy', np)
    return xpy.concatenate(parts, axis=-1)


def localise_peaks(Xw, fk, rows, t_grid, h_enum, tol, period, xpy=np,
                   peak_chunk=4096, t_last=None):
    """Newton on the band-limited interpolant: turn a grid INDEX into a LOCATION.

    ``t_grid`` are the enumeration-grid times of the enumerated maxima and ``rows`` says
    which row each belongs to.  Returns ``(t_star, q_star, converged)``.

    An enumerated extremum is a grid index, and the crest it stands for can be anywhere
    within ``+/- h_enum/2`` of it.  Building the integration interval around the index
    instead of the crest is what made this module drop up to 165 nats -- see
    :data:`LOCALISE_SAFETY` for the measured table.  This is the missing step.

    It is NOT a seeded root-finder in the sense that sank RIFT PR #201.  That failure
    was Newton seeded at GUESSED points, used to FIND extrema, so the ones it did not
    guess were never found.  Here every extremum has already been enumerated, and Newton
    only refines a location inside the bracket ``[t_grid - h_enum, t_grid + h_enum]``
    that the enumeration already established.  It cannot discover or lose a peak; it can
    only place one.  An iterate that leaves the bracket, or a peak that does not reach
    ``tol``, is reported as NOT converged and its row goes to the dense path.

    ``q``, ``q'`` and ``q''`` come from the spectral representation directly --
    ``q(t) = Re sum_j Xw_j exp(w_j t)`` with ``w_j = 2 pi i f_j / T``, so the
    derivatives are the same sum with ``w_j`` and ``w_j^2`` folded in and cost one
    exponential array between them.  Convergence is quadratic, so the SNR-dependence of
    this step is logarithmic: 2-4 iterations over the whole production range.
    """
    n_pk = int(t_grid.shape[0])
    w = (2j * np.pi / float(period)) * fk
    t_out = xpy.zeros(n_pk, dtype=np.float64)
    q_out = xpy.zeros(n_pk, dtype=np.float64)
    ok_out = xpy.zeros(n_pk, dtype=bool)

    for a in range(0, n_pk, peak_chunk):
        b = min(a + peak_chunk, n_pk)
        Xr = Xw[rows[a:b]]                      # (P, nf)
        tg = t_grid[a:b]
        tol_c = tol[a:b]
        t = tg.copy()
        step = xpy.zeros(t.shape, dtype=np.float64)
        for _ in range(LOCALISE_MAX_ITER):
            E = Xr * xpy.exp(w[None, :] * t[:, None])
            q1 = xpy.sum(E * w[None, :], axis=-1).real
            q2 = xpy.sum(E * (w * w)[None, :], axis=-1).real
            # Only a strictly concave point is a maximum to walk towards.  A
            # non-concave iterate is left where it is and reported unconverged.
            safe = q2 < 0
            step = xpy.where(safe, -q1 / xpy.where(safe, q2, -1.0), 0.0)
            step = xpy.clip(step, -h_enum, h_enum)
            t_new = xpy.clip(t + step, tg - h_enum, tg + h_enum)
            if t_last is not None:
                # The crest of a peak enumerated at an endpoint can lie OUTSIDE the
                # window; within it the maximum is then the boundary itself.  Clamping
                # here keeps the reported crest inside the integration domain, which is
                # what the interval is built around.
                t_new = xpy.clip(t_new, 0.0, t_last)
            step = t_new - t
            t = t_new
            if bool(xpy.all(xpy.abs(step) <= tol_c)):
                break
        E = Xr * xpy.exp(w[None, :] * t[:, None])
        q0 = xpy.sum(E, axis=-1).real
        q2 = xpy.sum(E * (w * w)[None, :], axis=-1).real
        inside = xpy.abs(t - tg) <= h_enum
        if t_last is not None:
            pinned = (t <= 0.0) | (t >= t_last)
        else:
            pinned = xpy.zeros(t.shape, dtype=bool)
        t_out[a:b] = t
        q_out[a:b] = q0
        # A PINNED PEAK IS NOT CONVERGED, and calling it converged cost up to +4.38 nats.
        #
        # The crest of a peak enumerated at an endpoint can lie outside the window; the
        # maximum over the domain is then the boundary itself, and it was tempting to
        # call that "found".  It is found -- but it is not a PEAK, and everything
        # downstream assumes it is.  At an interior crest `q'` vanishes, so `exp(lnL)`
        # is locally Gaussian and a spacing derived from the curvature resolves it.  At a
        # boundary the maximum is a CORNER: `q'` is non-zero there, so the local integrand
        # is an exponential decay of rate `|q'|`, and the spacing derived from `sigma`
        # does not resolve that scale at all.  The trapezoid's half-weight endpoint then
        # over-counts by
        #
        #     log( lam * (0.5 + 1/(exp(lam) - 1)) )  ->  log(lam/2),   lam = |q'| * h_loc
        #
        # which GROWS WITHOUT BOUND in amplitude -- +log 2 per factor 4.  MEASURED on a
        # single band-limited bump centred at t = 0, against the same interpolant
        # integrated on the very interval the module chose: **+1.27 / +1.96 / +2.66 /
        # +3.35 / +4.03 nats** at rho-scale 4e4 / 1.6e5 / 6.4e5 / 2.56e6 / 1e7, and +4.38
        # at the ceiling.  All ACCEPTED, and both a-posteriori defences are silent BY
        # CONSTRUCTION: the containment check compares the grid's attained maximum against
        # `row_star`, and the grid's FIRST POINT is the pinned crest, so `attained ==
        # row_star` identically and it can never fire; the tail bound is a statement about
        # mass OUTSIDE the intervals and the error is entirely inside.
        #
        # So the row is declined and handed to the dense path.  That is the fail-closed
        # direction and it restores the contract this rule actually makes -- never a worse
        # value than the backstop.  (The dense path carries a milder form of the same
        # defect, +0.77 to +2.74 nats on these rows, because its grid also begins at the
        # boundary with a half weight; that is a defect of the shared reconstruction and
        # is recorded in the design note, not papered over here.)
        ok_out[a:b] = inside & (xpy.abs(step) <= tol_c) & (q2 < 0) & (~pinned)
    return t_out, q_out, ok_out


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
#: ``2 * W_SIGMA * UPSAMPLE_SAFETY`` sub-intervals however large ``sigma`` is.  Used to
#: reject a row BEFORE any work is done for it.
#:
#: It is NOT a strict lower bound, and calling it one was wrong: an interval clipped by a
#: window end carries fewer points than this.  Such a row is wrap-exposed or nearly so and
#: has almost always been routed away by the edge guard already -- but "almost always" is
#: not "always", so this is a COST heuristic and nothing more.  It cannot affect a
#: returned value: a row it wrongly declines is computed by the dense path instead.
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

    # ROW CLASSIFICATION IS THE DENSE PATH'S -- literally, by calling it.  This module's
    # contract is that it changes WHERE the refined grid is placed and nothing about
    # WHICH rows get one, so any row `time_marginalize_bandlimited` refines must be a row
    # peak-local refines.  That was previously kept true by copying, and the copy went
    # stale across the rebase onto rift_O4d in three separate clauses; see _classify_rows.
    (sigma, jmax, measurable, has_peak, flat, exposed, unmeasurable,
     factors) = _tmq._classify_rows(lnL_coarse, deltaT, npts, xpy=xpy)
    refined = has_peak & (factors > 1)

    out = _log_simps_rows(lnL_coarse, deltaT, simps, xpy=xpy)
    peaks_out = [None] * n_rows if return_peaks else None

    stats = dict(n_peak_local_rows=0, n_dense_fallback_cost=0,
                 n_dense_fallback_cost_pregate=0, n_dense_fallback_nopeak=0,
                 n_dense_fallback_tail=0, n_dense_fallback_structure=0,
                 n_dense_fallback_ceiling=0, n_dense_fallback_localise=0,
                 n_dense_fallback_containment=0,
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
    # THE CEILING FIRST.  `required_upsample_factors` saturates a row it cannot justify
    # at 2*UPSAMPLE_FACTOR_MAX precisely so the dense path will RAISE on it -- that is
    # the fail-closed behaviour of the whole option.  The cost gate below compares
    # against the dense cost, so the sharper the row the more certainly this rule keeps
    # it, and a row past the ceiling is the sharpest kind there is: it would be handled
    # here and silently returned instead of raising.  Measured before this check: at a
    # derived factor of 8192 the dense path raised (as designed) while peak-local
    # returned -24451 nats and reported tail_bound_worst = -2721.
    #
    # WHAT THIS CHECK IS, AND DELIBERATELY IS NOT.  It catches the SATURATION SENTINEL --
    # `required_upsample_factors` returning 2*UPSAMPLE_FACTOR_MAX because it could not
    # justify a factor at all.  It does NOT mirror the dense path's ceiling test, and that
    # is a decision rather than an oversight.
    #
    # `time_marginalize_bandlimited` re-measures the width on the grid it just refined and
    # doubles until the criterion holds, raising if the chain leaves the legal range, so it
    # can refuse a row whose COARSE factor was legal.  This module tests the coarse factor
    # once.  The two therefore disagree on exactly one band, `factor_coarse ==
    # UPSAMPLE_FACTOR_MAX`: measured over 220 random multi-bump rows, 27 landed there and 8
    # of them are rows the dense path refuses and this one returns.  (Below 4096 the
    # disagreement never arose -- it would need the coarse estimate to be optimistic by
    # three doublings -- and at the 8192 sentinel both refuse, 44 of 44.)
    #
    # Those 8 rows are NOT approximated: their values were scored exact against a converged
    # spectral reference.  UPSAMPLE_FACTOR_MAX bounds the DENSE GRID -- the point past which
    # a zero-padded FFT over the whole window stops being affordable or believable -- and
    # this rule never builds that grid.  Its resolution comes from a per-peak width measured
    # on the enumeration grid, which is already 8x finer than the coarse grid the ceiling
    # factor is derived from, and the containment check and tail bound then verify the
    # outcome.  So a row the dense path declines on grid size is one this rule may legitimately
    # still resolve, and RO's ruling is that it may: the ceiling is a limit on the dense
    # grid, not on the physics.  The cost of forcing agreement would be `>=` here, which
    # sends every row at the legal ceiling to the dense path -- 19 of those 27 rows, and the
    # sharpest legal rows are precisely the regime this rule exists to serve.
    over_ceiling = factors_np > _tmq.UPSAMPLE_FACTOR_MAX
    viable = (c_lo < c_dn) & (~over_ceiling)
    stats['n_dense_fallback_ceiling'] += int(np.sum(over_ceiling))
    # Counted SEPARATELY from the post-enumeration gate.  Both are cost decisions, but
    # only this one runs before any work is done for the row, and it is the one credited
    # with removing the regression where this rule came out slower than the path it
    # delegates to.  A single shared counter made it removable with the suite green:
    # nothing could tell which gate had fired.
    stats['n_dense_fallback_cost_pregate'] += int(np.sum((~viable) & (~over_ceiling)))
    if not viable.any():
        return values, ok, peaks

    # ---- enumeration.  One FFT upsample of kappa at a FIXED, SNR-independent
    # factor.  The callback is NOT evaluated on this grid: only term(kappa) is
    # needed, because a monotone callback cannot move an extremum.
    k_up = reflected_bandlimited_upsample(
        kappa_rows, PEAK_ENUM_FACTOR, xpy=xpy)[..., :last + 1]
    q_up = _term(k_up)
    del k_up
    n_enum = q_up.shape[-1]

    # The reflected spectrum is needed for the rigorous crest bound below and again for
    # localisation; it is one FFT, built once here.
    kappa_reflected = xpy.concatenate(
        (kappa_rows, xpy.flip(kappa_rows, axis=-1)), axis=-1)
    Xw, fk = bandlimited_spectrum(kappa_reflected, xpy=xpy)
    period_ref = 2.0 * period
    # RIGOROUS bound on |q''| for this row, from the spectrum rather than from a model:
    # q(t) = Re sum_j Xw_j exp(w_j t), so |q''| <= sum_j |Xw_j| |w_j|^2 everywhere, by the
    # triangle inequality.  No parabola is assumed and nothing is fitted, which is the
    # whole point -- see the keep note below.
    q_ddot_max = spectral_curvature_bound(Xw, fk, period_ref, xpy=xpy)

    mask = enumerate_peak_indices(q_up, xpy=xpy)
    mask = mask & xpy.asarray(viable)[:, None]
    rows_p, cols_p = xpy.where(mask)          # full-width mask: index is the sample
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
    # The stencil centre is shifted inward by the MINIMUM needed for a three-point
    # stencil to exist -- one sample -- and out-of-range half-widths are masked off.
    #
    # Two wrong versions preceded this.  The first clipped the centre by `maxd = 8`,
    # which measured a point up to 8 enumeration samples away.  The second stopped
    # clipping entirely and masked instead, which looks more principled and is worse at
    # the two indices that matter: at `cols_p == 0` the whole left half of the stencil
    # is out of range at EVERY half-width, so `d2` is NaN throughout, `sigma = inf`, and
    # the peak is dropped for "no resolvable curvature" before anything else runs.  The
    # vertex height degrades the same way, falling back to the raw sample.  Measured:
    # indices 0 and n_enum-1 dropped, 1 / 7 / 8 / n-2 kept -- so revision 2's endpoint
    # enumeration was DEAD CODE (22 endpoint maxima enumerated, zero able to obtain a
    # finite width), and the near-edge defect it was meant to fix re-ran byte-identical.
    #
    # Shifting by one gives a genuine ONE-SIDED fit at an endpoint -- the parabola
    # through (0,1,2) -- which is the correct estimator there, not a compromise.
    offs = xpy.arange(-maxd, maxd + 1)
    c_st = xpy.clip(cols_p, 1, n_enum - 2)
    take_raw = c_st[:, None] + offs[None, :]
    st_valid = (take_raw >= 0) & (take_raw < n_enum)
    q_st = q_up[rows_p[:, None], xpy.clip(take_raw, 0, n_enum - 1)]
    lnL_st = loglikelihood(q_st, xpy.broadcast_to(rho_col_rows[rows_p], q_st.shape))
    lnL_st = xpy.where(st_valid, lnL_st, np.nan)
    sigma_pk = _peak_curvature_sigma(lnL_st, h_enum, xpy=xpy)

    # G1/Door 1: the KEEP filter must compare CRESTS, not samples -- and not estimates
    # of crests either.  See the two-stage note below.
    #
    # This is the same defect as the interval-centring bug, at a different site, and it
    # is the reason to grep for every consumer of the enumeration index rather than fix
    # the one that was reported.  `q_up[rows_p, cols_p]` is the SAMPLE value; a crest a
    # distance d from its sample reads (d/sigma)^2/2 nats low, which for a sharp row is
    # tens of nats.  Comparing a between-samples peak against an on-sample peak then
    # drops the former.  MEASURED on the two-peak fixture at rho ~ 700: the secondary
    # crest is 1.003 nats below the dominant crest, but its SAMPLE is 70.99 nats below,
    # past PEAK_KEEP_NATS -- so one of two equal peaks was deleted and the answer came
    # back exactly -log(2) = -0.693147 low.
    #
    # THE PARABOLIC VERTEX IS NOT THE CREST, and believing it was is how this class of
    # defect survived a second round.  The vertex is exact only if `lnL` is quadratic
    # across the whole stencil; it is band-limited `q(t)`, so the quartic term survives
    # and the vertex obeys the SAME `(delta/sigma)^2` law as the sample, just smaller.
    # MEASURED at half-cell phase -- under-read of the crest, in nats:
    #
    #   sigma_t/deltaT   factor    sample     vertex
    #     0.004934         512     -108.75     -1.17
    #     0.002467        1024     -434.98     -4.66
    #     0.001233        2048    -1739.93    -18.66
    #     0.000617        4096    -6959.72    -74.63   <-- past PEAK_KEEP_NATS
    #
    # `UPSAMPLE_FACTOR_MAX = 4096` permits sigma_t/deltaT down to 0.000488, so that last
    # row is a LEGAL configuration, and end-to-end it reproduced -log(2) exactly with
    # both defences silent.  Every approximation substituted for the crest fails the
    # same way one octave further out.
    #
    # ROUND 6: A TARGETING MODEL WAS PROMOTED TO A BOUND, AND THAT IS THE FOURTH DOOR.
    #
    # THE LAPLACE MODEL IS NOT THE DEFECT.  `lnL` is of course not a parabola; the whole
    # design uses the Gaussian/curvature picture to TARGET -- where the mass is, how wide
    # an interval must be, how fine a spacing it needs -- and then either the Laplace
    # quadrature is reliable enough or the row falls back to the grid.  Used that way the
    # model owes nothing, because the answer comes from the quadrature actually performed
    # and from the checks that verify it.
    #
    # The defect is using the same model where an INEQUALITY is required.  The pre-filter
    # compared `crest_upper = lnL_sample + (h_enum/2)^2 / (2 sigma^2)` against the largest
    # sample in the row, and was described as an upper bound that "can only ever keep too
    # many".  It is a targeting estimate wearing the word "bound", and an independent
    # re-attack broke it:
    #
    #   * the displacement is bounded by `h_enum`, not `h_enum/2` -- the localiser's own
    #     bracket says so and 0.959*h_enum has been observed -- so the correction is
    #     taken at less than half the distance it must cover; and, much worse,
    #   * the ANHARMONIC part of the deficit carries the same 1/sigma^2 amplification as
    #     the quadratic part, so the error does not shrink where it matters.  MEASURED at
    #     derived factor 1024 on a skewed peak: the pure quantisation excess is 4.4 nats
    #     while the actual shortfall is 122.30.
    #
    # A targeting model that is off by 122 nats costs a slightly wrong interval, which the
    # verification catches.  The same model asked to justify DELETING a peak costs the peak.
    #
    # So `crest_upper` fell short of the true crest by 122 / 489 / 1957 nats at derived
    # factor 1024 / 2048 / 4096 -- and being short, it DROPPED co-dominant peaks.
    # End-to-end, shipped code, both peaks well inside PEAK_KEEP_NATS: **-0.358 nats** at
    # factor 1024 and, when the deleted peak is raised above the survivor, **-1849 nats**,
    # ACCEPTED, with the tail bound and the containment check both silent.  A peak may sit
    # ~1900 nats ABOVE the one that survives and still be deleted.  The tail bound cannot
    # backstop it because `q_out_max` reads the dropped peak at its SAMPLE -- the very
    # quantity the defect corrupts; recomputed as an honest supremum, those rows' margins
    # are +11.5 / +12.1 / +7.2 and every one would be REJECTED.
    #
    # This is the fourth time this class has reopened, and the fourth time the estimate of
    # the crest was one octave too optimistic.  Widening the constant would be the fifth.
    # NO QUANTITY DERIVED FROM THE ENUMERATION INDEX MAY DROP A PEAK.  The index survives
    # only as a Newton seed and bracket centre, which is the one thing it is entitled to
    # be.  The keep decision is now taken ONCE, after localisation, on `lnL_star`, which
    # is the crest rather than an estimate of it.
    #
    # What used to justify the pre-filter was COST -- bounding how many peaks reach
    # localisation.  That job is already done, and done safely, by the gate below: it runs
    # BEFORE localisation, it is built from the enumeration samples alone, and it declines
    # the whole ROW to the dense path rather than selecting which peaks to believe.  A
    # gate that declines a row is safe in a way that a filter which deletes a peak is not.
    # THE PRE-FILTER, REBUILT ON AN INEQUALITY INSTEAD OF A FIT.
    #
    # Cost still has to be bounded -- without any pre-filter every row enumerates its
    # whole oscillation (295 maxima on one fixture here), the gate below sees more
    # intervals than `MAX_INTERVALS` and declines EVERY row, and the option becomes inert.
    # Measured: `n_peak_local_rows = 0` on all six fixture families.
    #
    # So a peak may still be dropped, but only against a bound that holds unconditionally.
    # Let `t*` be the crest and `t_s` its enumeration sample.  Expanding about the CREST,
    # where `q'` vanishes by definition,
    #
    #     q(t_s) = q(t*) + q''(xi) (t_s - t*)^2 / 2,     |t_s - t*| <= h_enum
    #
    # so  q(t*) <= q(t_s) + q_ddot_max * h_enum^2 / 2  with `q_ddot_max` the spectral
    # bound computed above.  This is a Taylor remainder with a TRUE bound on the second
    # derivative, not a parabolic fit, so it is immune to the anharmonicity that broke the
    # previous version: there is no assumption that `lnL` is quadratic across a cell, and
    # no `1/sigma^2` amplification of a modelling error.  It uses `h_enum`, the localiser's
    # actual bracket, not `h_enum/2`.
    #
    # `loglikelihood` is monotone in its first argument over its DOMAIN, so bounding `q`
    # bounds `lnL`.  Stated precisely because the obvious stronger claim is false: the
    # shipped distance-marginalized callback returns `-inf` ABOVE its table as well as
    # below, so it is not monotone increasing everywhere.  That direction is safe here --
    # an upper bound that evaluates into the hole becomes `-inf`, the peak is dropped, and
    # the row loses peaks until it falls back to the dense path -- so it costs coverage,
    # never accuracy.  With the shipped table the boundary sits at `D_eff < d_min/10` and
    # is not reachable at default settings.  The
    # comparison is then a genuine upper bound against a genuine lower bound (the largest
    # SAMPLE in the row, which cannot exceed the crest above it), and a peak is discarded
    # only when it cannot be within `PEAK_KEEP_NATS` however the quantisation falls.
    #
    # NOTE the sample is read AT THE ENUMERATED INDEX.  The previous version read
    # `lnL_st[:, maxd]`, which sits at the stencil centre -- clipped inward by one at the
    # array ends -- so at `cols_p` 0 or `n_enum-1` it was a full enumeration cell away from
    # the peak, measured 132 nats low at rho ~ 40 and 8449 nats low at rho ~ 700, growing
    # with SNR.  That is the same defect at a third site.
    q_at_peak = q_up[rows_p, cols_p]
    q_crest_upper = crest_upper_bound(q_at_peak, q_ddot_max[rows_p], h_enum)
    rho_at_peak = rho_col_rows[rows_p, 0]
    lnL_upper = loglikelihood(q_crest_upper, rho_at_peak)
    lnL_lower = loglikelihood(q_at_peak, rho_at_peak)

    rows_np = _host(rows_p, xpy)
    cols_np = _host(cols_p, xpy)
    sig_np = _host(sigma_pk, xpy)
    up_np = _host(lnL_upper, xpy)            # rigorous UPPER bound on this crest
    low_np = _host(lnL_lower, xpy)           # LOWER bound (the sample cannot exceed it)

    row_best = np.full(n_rows, -np.inf)
    np.maximum.at(row_best, rows_np, low_np)
    # `isfinite(sig_np)` is not a magnitude decision: a peak with no finite negative
    # curvature at any stencil half-width has no width, so no interval can be built for
    # it.  What is dropped here lies outside the intervals and enters the tail bound.
    keep = np.isfinite(sig_np) & (up_np > row_best[rows_np] - PEAK_KEEP_NATS)
    rows_np, cols_np, sig_np = rows_np[keep], cols_np[keep], sig_np[keep]
    if rows_np.size == 0:
        return values, ok, peaks

    # ---- rows that survived every gate but enumerated no usable peak.  Counted, so the
    # sub-counts of last_report() RECONCILE against n_dense_fallback_rows: an operator
    # reading the columns should not find an unexplained residual.
    _has_peak = np.zeros(n_rows, dtype=bool)
    _has_peak[rows_np] = True
    stats['n_dense_fallback_nopeak'] += int(np.sum(viable & (~_has_peak)))

    # ---- gate 2, on a CONSERVATIVE SUPERSET of the intervals, BEFORE localising.
    # Localisation is the expensive step (Newton over the spectrum, per peak), and a
    # broad row has many peaks, so running it before the gate that discards the row is
    # pure waste -- the same mistake that once made this rule slower than the path it
    # delegates to, and it cost 7x here before this gate was added (0.14x vs the dense
    # path at sigma_t/deltaT = 0.17, on a block where every row fell back anyway).
    #
    # The crest is somewhere within h_enum/2 of its sample, so an interval built about
    # the sample and widened by h_enum/2 CONTAINS the interval that localisation will
    # produce, whatever the answer turns out to be.  Its point count is therefore an
    # over-estimate, and a gate on an over-estimate can only decline rows -- never keep
    # one it should have declined.  Where this rule actually wins the conservatism is
    # irrelevant: at sigma_t/deltaT ~ 0.002 the superset costs 66k against a dense cost
    # of 12M, so the row is kept with four orders of magnitude to spare.
    tol_np = LOCALISE_SAFETY * sig_np
    t_grid_np = cols_np * h_enum
    # h_enum, NOT h_enum/2.  The bracket in `localise_peaks` is +/- h_enum and accepts
    # anything strictly inside it, so |t_star - t_grid| is bounded by h_enum and not by
    # half of it -- and that bound is reached: over 14,182 localised peaks the largest
    # observed displacement was 0.959 * h_enum, and 2.62% of intervals were NOT
    # contained by a half-cell margin.  Narrowing the bracket instead would be wrong:
    # an asymmetric peak's crest genuinely can sit more than half a cell from its
    # sample, which is what that 0.959 measures.
    prov_half = W_SIGMA * sig_np + tol_np + h_enum
    p_order, p_gid, pg_row, pg_lo, pg_hi = merge_intervals_by_row(
        rows_np, np.maximum(t_grid_np - prov_half, 0.0),
        np.minimum(t_grid_np + prov_half, t_last), t_last)
    p_smin = np.full(pg_row.size, np.inf)
    np.minimum.at(p_smin, p_gid, sig_np[p_order])
    p_nloc = np.maximum(3, np.ceil(
        (pg_hi - pg_lo) / np.minimum(p_smin / UPSAMPLE_SAFETY, h_enum)
    ).astype(np.int64) + 1)
    p_niv_row = np.bincount(pg_row, minlength=n_rows)
    p_cl, p_cd = _estimated_costs(np.bincount(pg_row, weights=p_nloc,
                                              minlength=n_rows), npts, factors_np)
    p_much = p_niv_row > MAX_INTERVALS
    p_slow = (~p_much) & (p_niv_row > 0) & (p_cl >= p_cd)
    stats['n_dense_fallback_structure'] += int(np.sum(p_much))
    stats['n_dense_fallback_cost'] += int(np.sum(p_slow))
    prov_keep = (p_niv_row > 0) & (~p_much) & (~p_slow)
    sel_pk = prov_keep[rows_np]
    if not sel_pk.any():
        return values, ok, peaks
    rows_np, cols_np = rows_np[sel_pk], cols_np[sel_pk]
    sig_np, tol_np = sig_np[sel_pk], tol_np[sel_pk]
    t_grid_np = t_grid_np[sel_pk]

    # ---- LOCALISE.  An enumerated extremum is a grid INDEX; the interval has to be
    # built around the CREST.  Centring on the index instead cost up to 165 nats, always
    # negative -- see LOCALISE_SAFETY.  Newton is confined to the bracket the
    # enumeration already established, so it places peaks, it cannot find or lose them.
    # THE RECONSTRUCTION MUST BE THE ONE THE DENSE PATH USES, and it is no longer the
    # raw periodic interpolant.  `time_marginalize_bandlimited` periodizes the EVEN
    # REFLECTION `[kappa forward, kappa backward]` (rift_O4d e4ed25c7, "Avoid Gibbs
    # ringing"), because a zero-padded FFT of the gathered slice alone identifies its
    # unlike endpoints and rings globally -- measured +140.9 nats on an adversarial row.
    #
    # This module was written against the older periodic contract and the rebase onto
    # rift_O4d changed it underneath.  Leaving it periodic makes peak-local integrate a
    # DIFFERENT continuous function from the one its own fallback rows get, inside a
    # single call: measured -3.79 nats on a row with peaks near both window ends
    # (`test_intervals_are_clipped_to_the_integration_domain`) and a residual 9.0e-6 nat
    # median bias on the uniform-arrival block.  So enumeration, localisation and local
    # evaluation all run on the reflected row.
    #
    # The reflected row has length `2*npts` -- always EVEN -- so `bandlimited_spectrum`
    # takes its even branch and splits the Nyquist bin exactly as `bandlimited_upsample`
    # does inside `reflected_bandlimited_upsample`.  The two therefore agree on the
    # forward interval, which is the only part this module ever evaluates.
    t_star, q_star, loc_ok = localise_peaks(
        Xw, fk, xpy.asarray(rows_np), xpy.asarray(t_grid_np), h_enum,
        xpy.asarray(tol_np), period_ref, xpy=xpy, t_last=t_last)
    t_np = _host(t_star, xpy)
    lnL_star = _host(loglikelihood(q_star, rho_col_rows[xpy.asarray(rows_np), 0]), xpy)
    loc_ok_np = _host(loc_ok, xpy).astype(bool)

    # A row with ANY peak the localiser could not place is not approximated -- it goes
    # to the dense path.  Fail closed: an unplaced crest is exactly the condition that
    # produced the bias.
    bad_loc = np.zeros(n_rows, dtype=bool)

    # ---- KEEP, stage 2: EXACT, on the localised crest values.
    #
    # This is the decision that matters, and it is taken here rather than before
    # localisation for one reason: `lnL_star` is the crest, not an estimate of it.  The
    # pre-filter above only bounded how many peaks reached this point, and it errs
    # toward keeping, so nothing that matters has been discarded on an approximation.
    row_top = np.full(n_rows, -np.inf)
    np.maximum.at(row_top, rows_np, lnL_star)
    exact_keep = lnL_star > row_top[rows_np] - PEAK_KEEP_NATS
    # Every localised crest, INCLUDING the ones about to be dropped.  They cost nothing --
    # Newton has already run on them -- and they are exact, so the tail bound below can use
    # a crest rather than the sample under it.  See the outside-supremum note there.
    q_all_np = _host(q_star, xpy)
    t_all_np = t_np.copy()
    rows_all_np = rows_np.copy()
    if not exact_keep.all():
        rows_np, cols_np = rows_np[exact_keep], cols_np[exact_keep]
        sig_np, tol_np = sig_np[exact_keep], tol_np[exact_keep]
        t_np, lnL_star = t_np[exact_keep], lnL_star[exact_keep]
        loc_ok_np = loc_ok_np[exact_keep]
        if rows_np.size == 0:
            return values, ok, peaks

    # The interval is centred on the crest and widened by the localisation residual, so
    # containment does not depend on the crest happening to sit near a grid sample.
    half_np = W_SIGMA * sig_np + tol_np
    lo_np = np.maximum(t_np - half_np, 0.0)
    hi_np = np.minimum(t_np + half_np, t_last)

    # Localisation failure is judged on the peaks that SURVIVED the exact filter: a peak
    # about to be discarded as immaterial should not condemn its row.
    bad_loc[rows_np[~loc_ok_np]] = True
    stats['n_dense_fallback_localise'] += int(np.sum(bad_loc))

    # Per-row crest value, for the a-posteriori containment check after integration.
    row_star = np.full(n_rows, -np.inf)
    np.maximum.at(row_star, rows_np, lnL_star)
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

    # These can only catch what the provisional pass could not.  The localised intervals
    # are contained in the provisional ones, so their point count -- and hence the cost
    # test -- can only have improved; the INTERVAL COUNT can go up, though, because
    # narrower intervals merge less readily, so the structure test is not redundant.
    too_much = n_iv_row > MAX_INTERVALS
    too_slow = (~too_much) & (n_iv_row > 0) & (c_local >= c_dense)
    stats['n_dense_fallback_structure'] += int(np.sum(too_much))
    stats['n_dense_fallback_cost'] += int(np.sum(too_slow))
    keep_row = (n_iv_row > 0) & (~too_much) & (~too_slow) & (~bad_loc)

    gbounds = np.searchsorted(g_row, np.arange(n_rows + 1))
    plan = []
    for r in np.nonzero(keep_row)[0]:
        ga, gb = int(gbounds[r]), int(gbounds[r + 1])
        plan.append((int(r), g_lo[ga:gb], g_hi[ga:gb], int(n_loc[ga:gb].max()),
                     int(bounds[r]), int(bounds[r + 1])))

    if not plan:
        return values, ok, peaks

    # ---- batched evaluation.  Rows are grouped by (interval count, point-count
    # bucket) so padding to a common shape can waste at most a factor of two, and
    # every interval slot of a group is one batched call.
    covered = np.zeros((n_rows, n_enum), dtype=bool)
    covered_len = np.zeros(n_rows)
    attained = np.full(n_rows, -np.inf)
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
            # would give h=0 and a degenerate grid; give it the enumeration spacing so
            # the trapezoid has a domain.  Note which way the risk runs: this makes the
            # integration domain EXCEED the merged interval, so the hazard is
            # double-counting against a neighbour, not missing mass.  Unreachable as
            # written -- a zero-length interval needs a peak exactly at a window end,
            # which the edge guard has already routed away -- but the thing to check is
            # over-coverage.
            h_h = np.where(h_h > 0, h_h, h_enum)
            k_loc = eval_bandlimited_uniform(Xw[rr_x], fk, xpy.asarray(a_h),
                                             xpy.asarray(h_h), m_pad, period_ref,
                                             xpy=xpy)
            lnL_loc = loglikelihood(
                _term(k_loc), xpy.broadcast_to(rho_col_rows[rr_x], k_loc.shape))
            parts[rr_x, j] = _log_trapz_local(lnL_loc, xpy.asarray(h_h), xpy=xpy)
            # Highest lnL the integration grid actually REACHED, for the containment
            # check below.  Free: the same maximum is already taken for the offset.
            attained[rr] = np.maximum(attained[rr],
                                      _host(xpy.max(lnL_loc, axis=-1), xpy))
            stats['n_local_points_total'] += int(rr.size) * m_pad
            for i_m, m in enumerate(members):
                lo_i = int(np.ceil(a_h[i_m] / h_enum))
                hi_i = int(np.floor(b_h[i_m] / h_enum))
                if hi_i >= lo_i:
                    covered[m[0], max(lo_i, 0):hi_i + 1] = True
                # EXACT covered length, not a count of grid indices.  A merged interval
                # narrower than h_enum, or a gap between two of them that happens to
                # contain no integer index, contributes nothing to the index count -- so
                # a T_outside built from that count omits real, uncovered time.  Measured
                # on a comb-like row before this change: 4.8% of the true mass sat in
                # such gaps, worth -0.050 nats, while the bound reported -43.
                covered_len[m[0]] += float(b_h[i_m] - a_h[i_m])
        stats['n_intervals_total'] += int(rr.size) * n_iv

    result = _logaddexp_reduce(parts, xpy=xpy)

    # ---- the tail bound.  log(T_outside) + max_{outside} lnL is an upper bound on
    # the omitted integral.  It is evaluated on term(kappa) -- one callback value per
    # row -- because the callback is monotone in it, so no evaluation on the full
    # time axis is needed.  A row whose bound is not small enough is NOT reported
    # with a caveat: it goes to the dense path.
    # ---- the outside supremum, evaluated OFF-GRID rather than sampled.
    #
    # `max(q_up over uncovered samples)` is a LOWER bound on the continuous supremum, and the
    # gap GROWS WITH AMPLITUDE.  Measured, honest supremum against the sampled one on rows this
    # rule accepted: the sampled margin read -79.6 / -289.5 / -2998.6 at amplitude 2e4 / 2e5 /
    # 2e6 while the honest margin was -65.5 / -71.1 / -75.8 -- an under-read of 14, then 218,
    # then 2923 nats.  The reported `tail_bound_worst` was therefore a diagnostic that got more
    # flattering the sharper the row, which is the wrong direction for a safety margin.
    #
    # WHERE THE GAP COMES FROM, and it is not a peak the enumeration missed.  The supremum over
    # a union of closed intervals is attained at an interior stationary point or at an end, so
    # the candidates are exactly: the crests of uncovered enumerated maxima, and the ENDS of the
    # covered intervals.  The ends dominate.  An interval end sits `W_SIGMA * sigma` from its
    # crest, i.e. `W_SIGMA**2/2 = 72` nats below it whatever the amplitude -- but the nearest
    # SAMPLE outside that end is a further `W_SIGMA * h_enum / sigma` nats down, which diverges
    # as the peak sharpens.  That single term is the whole measured under-read.
    #
    # So evaluate the candidates instead of sampling near them.  `q` is band-limited and the
    # double-copy (forward+backward) Fourier model reconstructs it exactly between samples, so
    # the interval ends are evaluated directly on that model, and the localised crests are
    # already in hand from Newton -- including the peaks the exact filter dropped, which is
    # precisely the set the sampled version read at their samples.
    #
    # This does not certify anything, and does not claim to: a peak the enumeration never found
    # would still be missed.  For a Nyquist-band-limited `q` on an 8x grid that is not a
    # realistic failure -- across npts 153/307/614 and spectral widths 30/120/300, zero material
    # continuous maxima (within 100 nats of the row top) were missed, and enumeration returns
    # one or two MORE than the interior continuous count.  What this does remove is the
    # amplitude-divergent term, which was real.
    cov_x = xpy.asarray(covered)
    q_out_max = xpy.max(xpy.where(cov_x, -np.inf, q_up), axis=-1)
    if g_row.size:
        # the ends of every merged interval, on the reflected interpolant
        edge_rows = np.concatenate([g_row, g_row])
        edge_t = np.concatenate([g_lo, g_hi])
        q_edge = _host(eval_bandlimited_points(Xw, fk, xpy.asarray(edge_rows),
                                               xpy.asarray(edge_t), period_ref,
                                               xpy=xpy)[0], xpy)
        q_out_np = _host(q_out_max, xpy)
        np.maximum.at(q_out_np, edge_rows, q_edge)
        # ... and the crests of enumerated peaks left outside, exact from localisation
        if rows_all_np.size:
            # Is each localised crest inside one of ITS OWN row's merged intervals?  Vectorised
            # by the same row-offset trick merge_intervals_by_row uses: the intervals are
            # ascending in (row, lo), so offsetting by `row * big` makes one global searchsorted
            # answer it for every peak at once.  A Python loop over intervals here is O(groups x
            # peaks) and is exactly the shape that once made this rule slower than the path it
            # delegates to.
            #
            # The exclusion is NOT optional and cannot be skipped for conservatism: a crest that
            # IS covered is already integrated, and feeding it to the outside maximum would make
            # the bound `log(T_out) + crest - result`, which rejects every row.
            big = 2.0 * (float(t_last) + 1.0)
            j = np.searchsorted(g_lo + g_row * big, t_all_np + rows_all_np * big,
                                side='right') - 1
            jc = np.maximum(j, 0)
            in_cov = (j >= 0) & (g_row[jc] == rows_all_np) & (t_all_np <= g_hi[jc])
            out_pk = ~in_cov
            if out_pk.any():
                np.maximum.at(q_out_np, rows_all_np[out_pk], q_all_np[out_pk])
        q_out_max = xpy.asarray(q_out_np)
    T_out = np.maximum(t_last - covered_len, 0.0)
    lnL_out = loglikelihood(q_out_max, rho_col_rows[:, 0])
    with np.errstate(divide='ignore', invalid='ignore'):
        bound = np.where(T_out > 0, np.log(np.maximum(T_out, 1e-300))
                         + _host(lnL_out, xpy), -np.inf)
        margin = bound - _host(result, xpy)

    planned = np.array([m[0] for m in plan])
    # TWO conditions, and they fail differently on purpose.  The tail bound is a
    # statement about mass OUTSIDE the intervals, computed from a sampled maximum, so it
    # is only as good as the grid it samples -- which is the grid that produced F1.  The
    # containment check is a statement about the crest being INSIDE, verified from the
    # integration grid's own values, and it is what actually catches a mis-placed
    # interval.  Neither subsumes the other and a row must satisfy both.
    contained = attained >= row_star - CONTAINMENT_SLACK_NATS
    good_mask = (margin[planned] < TAIL_LOG_TOL) & contained[planned]
    accepted = planned[good_mask]
    rejected = planned[~good_mask]
    stats['n_dense_fallback_tail'] += int(np.sum(
        ~(margin[planned] < TAIL_LOG_TOL)))
    stats['n_dense_fallback_containment'] += int(np.sum(
        (margin[planned] < TAIL_LOG_TOL) & (~contained[planned])))
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
                # t_np is the LOCALISED crest, not the enumeration sample -- which is the
                # whole point of exposing it for a time-first reordering.
                peaks[r] = (t_np[a:b].copy(), sig_np[a:b].copy())

    return values, ok, peaks
