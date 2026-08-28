"""Band-limited quadrature for the ILE time marginalization.

WHAT IS WRONG WITH THE HISTORICAL PATH
--------------------------------------
``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`` integrates
``exp(lnL(t))`` over the marginalization window with Simpson's rule at a FIXED
spacing ``dx = deltaT = 1/srate``.  The integrand's width is not fixed: after
the angle/phase reductions ``exp(lnL(t))`` is a near-Gaussian peak of width

    sigma_t = 1 / (2 pi rho sigma_f)

with ``sigma_f`` the noise-weighted template frequency spread.  ``sigma_t``
shrinks like 1/rho, so a grid tied to the data sample rate under-resolves its
own integrand, and does so WORSE the louder the event.

Simpson is the wrong rule in that regime.  ``simpson = (4 T_h - T_2h)/3``
carries the coarser trapezoid ``T_2h`` with a minus sign, so it inherits an
alias with period ``2h`` -- which is why the historical value depends on where
the peak happens to fall between samples.  Trapezoid, by contrast, is
spectrally accurate on a Gaussian: by Poisson summation its relative error is

    ~ 2 exp(-2 pi^2 (sigma_t/h)^2)

so it needs only ``h <~ sigma_t`` and then converges at a rate no polynomial
rule can match.  That bound is what ``UPSAMPLE_SAFETY`` is derived from; it is
not a tuning constant.

THE FIX
-------
``kappa(t)`` (the data term) is band-limited -- it is built from the rholm
cross-correlation timeseries -- and on this path ``rho_sq`` (the template
self-term) is time-independent.  By the sampling theorem the samples the code
ALREADY computed determine the continuous ``kappa(t)`` exactly, so one
zero-padded FFT per row recovers it at any spacing: no extra likelihood
evaluations, no extra precompute, no new physics.  We then apply the SAME
``loglikelihood`` callback on the dense grid and integrate with the trapezoid
rule.

Because the callback is applied after upsampling, this is correct for every
caller -- the plain helper, phase marginalization (``|kappa|``), and the
distance-marginalization table lookup alike.  For the same reason the
resolution requirement is derived from ``lnL(t)`` itself and never from
``kappa``: ``lnL`` is affine in ``kappa`` only for the default helper.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
* It does not change the integration DOMAIN.  The dense grid spans exactly
  ``[t_0, t_{npts-1}]``, the closed interval Simpson used; the only dense
  samples discarded are the ``factor-1`` that lie past the last coarse sample,
  which are precisely the ones sitting across the FFT's periodic wrap.  An
  edge taper would shrink the domain instead, which is invisible for a sharp
  interior peak but is a real error for a flat, low-amplitude integrand -- the
  regime where the historical path was already fine.
* It does not re-centre the window.  Rows whose peak sits at the very first or
  last sample are COUNTED and reported (``last_report()['n_edge_peak_rows']``),
  not repaired: a mis-centred window is a separate defect and is not smuggled
  in here.  (This is distinct from ``EDGE_GUARD_FRACTION`` below, which does
  act -- by handing those rows back the historical value untouched.)
* It exposes no accuracy-versus-cost knob.  The upsampling factor is derived
  from the measured peak width and then re-verified on the dense grid; it
  cannot be set too small by a caller.

On the DENSE grid the choice of trapezoid over Simpson buys COST, not accuracy:
at ``h <= sigma_t/2`` Simpson would also be far below any error that matters.
Trapezoid reaches the same bound at half the resolution, which is a factor of
two off the dominant cost.  It is on the COARSE grid that Simpson is actively
the wrong rule.

Backend notes (``xpy`` is ``numpy`` or ``cupy``; this module imports nothing
from RIFT, so the same file serves the rift_O4c and rift_O4d lines):

* **Untested on GPU.**  Every array call used here exists in cupy and all of
  them already appear in this repo's cupy paths -- but it has not been run on a
  GPU and should not be described as though it had.
* **It synchronises.**  The refinement decision needs host-side scalars, so
  there are a handful of device-to-host transfers per block and two per memory
  chunk.  Free on CPU; on GPU a large batch could spend real time there.  If
  this is ever enabled on GPU, profile it rather than assuming the measured CPU
  cost ratios carry over.
* ``rho_sq`` reaches the callback as a **read-only broadcast view** on the dense
  path, where the historical path passed a writable array.  Every shipped
  callback allocates rather than writing in place (the distance-marginalization
  one was checked end-to-end), but a callback that wrote to ``rho_sq`` would
  work historically and raise here.
"""

import numpy as np

__all__ = [
    "TIME_QUADRATURE_CHOICES", "UPSAMPLE_SAFETY", "UPSAMPLE_FACTOR_MAX",
    "EDGE_GUARD_FRACTION", "SUPPORT_NATS", "validate_time_quadrature", "bandlimited_upsample", "peak_width_from_lnL",
    "required_upsample_factor", "time_marginalize_bandlimited", "last_report",
]

#: Accepted values of the ``time_quadrature`` argument.  ``'simpson'`` is the
#: historical behavior and remains the default everywhere.
TIME_QUADRATURE_CHOICES = ("simpson", "bandlimited")

#: Samples demanded per Gaussian sigma: ``h <= sigma_t / UPSAMPLE_SAFETY``.
#: With the Poisson-summation bound ``2 exp(-2 pi^2 (sigma/h)^2)`` above, 2.0
#: leaves a residual quadrature error of ~2e-34 -- far below every other error
#: in the calculation, which is the point: this constant should never be the
#: reason a result is wrong, and it is not exposed to callers.
UPSAMPLE_SAFETY = 2.0

#: Rows whose peak falls within this fraction of the window at either end are
#: handed back the HISTORICAL Simpson value, bit for bit, instead of the
#: band-limited one.  The FFT interpolant is periodic, so a peak near the window
#: edge sits next to the wrap and the reconstruction is not merely inaccurate
#: there but wrong in the dangerous direction: a spuriously HIGH lnL biases the
#: evidence up and importance-weights that sample into dominance.  Measured with
#: the guard disabled, against an analytic non-periodic integrand: +0.0006 nats
#: at 32 bins from the edge (the guard boundary here), +0.017 at 16, +0.16 at 8,
#: +2.2 at 0 -- and the companion rift_O4d measurement, with a sharper peak,
#: reached +88.8.  The magnitude depends on the peak; the sign does not.
#: The deviation cannot be estimated from the window's own samples (the periodic
#: band-limited interpolant is uniquely determined by them, so bounding the
#: departure needs information from outside), so it has to be bounded a priori.
#: The guard makes the wrap failure mode UNREACHABLE: no row can be handed a
#: value contaminated by the periodic wrap, because a row whose peak is near the
#: wrap is returned untouched.  Combined with the refinement predicate below,
#: the property is: a row's value differs from the historical one IF AND ONLY IF
#: its integrand was under-resolved.  That is NOT the same as promising every
#: row is at least as accurate as Simpson -- the refined path carries its own
#: ~1e-4 nat residual at the sharpest amplitudes measured, and a row where
#: Simpson's error happens to cross zero could be marginally better under the
#: old rule.  State the failure mode, not an accuracy ordering.
#:
#: The band is a FRACTION of the window, so it interacts with
#: --data-integration-window-half.  Measured, 2000 random-sky rows at rho=10:
#:
#:     iwh (s)   0.075   0.050   0.030   0.020   0.010
#:     exposed    0.0%    0.0%   33.5%   23.4%   41.4%
#:
#: At the 0.075 s default no row is ever claimed.  Narrow the window and a large
#: share of rows falls back to Simpson -- still correct, but the option stops
#: doing anything for them.  last_report()['n_wrap_exposed_rows'] and
#: ['n_refined_rows'] are how you see that; check them before concluding an
#: enabled run actually used the new path.
#:
#: The real fix is a wider GATHER, so the wrap falls outside the integration
#: domain; that touches the GPU kernel and its buffer-margin assumptions and is
#: deliberately not done here.
EDGE_GUARD_FRACTION = 0.125

#: Only bins within this many nats of a row's maximum can affect the integral
#: (exp(-100) ~ 4e-44, so even summed over every dense sample they are far below
#: float64 resolution).  The curvature scan is restricted to them.  Without
#: that restriction the scan is unbounded below: numerical noise in a far tail,
#: a distance-marginalization table edge, or a row where kappa is essentially
#: zero can present an arbitrarily narrow "feature", which would inflate the
#: derived factor for every row in its group and can reach the raising ceiling
#: on a row that contributes nothing.
SUPPORT_NATS = 100.0

#: Refusal threshold.  Reaching it means the integrand is orders of magnitude
#: sharper than the grid can describe, which is a modelling problem (or a bug),
#: not something to silently truncate to a "best effort" answer.
UPSAMPLE_FACTOR_MAX = 4096

#: Working-set budget for the dense arrays, in bytes.  Internal only: it splits
#: the extrinsic axis into chunks and cannot change the answer.
_DENSE_CHUNK_BYTES = 128 * 1024 * 1024

_LAST_REPORT = {}


def last_report():
    """Diagnostics from the most recent :func:`time_marginalize_bandlimited`.

    Keys:

    ``factor``
        Largest upsampling actually used in the batch.
    ``factor_initial``
        Largest factor derived from the COARSE grid, before the dense
        re-verification.  ``factor > factor_initial`` means the re-verification
        fired, i.e. the coarse-grid width estimate was optimistic.
    ``factor_histogram``
        ``{factor: n_rows}``.  The factor is derived PER ROW, so a handful of
        sharp rows do not impose their resolution on the whole batch; this is
        where the cost went.
    ``sigma_t_min``, ``sigma_t_min_dense``
        Sharpest integrand width found (s), on the coarse and dense grids.
    ``npts``, ``npts_dense``, ``n_rows``, ``n_chunks``
        Shapes and the internal memory chunking.
    ``n_wrap_exposed_rows``
        Rows handed back the historical Simpson value because their peak fell
        within ``EDGE_GUARD_FRACTION`` of a window edge.
    ``n_edge_peak_rows``
        Rows whose peak is at the very first or last sample -- a mis-centred
        window, which is a SEPARATE defect this change does not address.
        Reported so it is visible, not repaired.
    ``n_nonfinite_rows``
        Rows whose lnL is not finite anywhere (e.g. entirely outside the
        distance-marginalization table).  Their curvature is UNMEASURABLE, not
        zero, so they derive factor 1 -- and would otherwise appear in a
        histogram as cheap rows and in a timing table as a speedup, when in fact
        nothing was refined.  Counted unconditionally.
    ``n_flat_rows``
        Rows with finite lnL and no resolvable curvature -- no signal to refine
        (e.g. an extrinsic sample in an antenna null, where kappa is numerically
        zero and lnL is constant).
    ``n_resolved_rows``
        Rows with a peak the grid ALREADY resolves, so factor 1 and nothing to
        do.  This is the cheap-and-correct population.

    These five -- refined, resolved, wrap-exposed, flat, nonfinite -- PARTITION
    the batch: every row is in exactly one, and only ``n_refined_rows`` changed
    value relative to the historical path.

    Returns a copy.
    """
    return dict(_LAST_REPORT)


def validate_time_quadrature(time_quadrature):
    """Return `time_quadrature`, or raise ValueError naming the valid choices."""
    if time_quadrature not in TIME_QUADRATURE_CHOICES:
        raise ValueError(
            "time_quadrature must be one of {}, got {!r}".format(
                TIME_QUADRATURE_CHOICES, time_quadrature))
    return time_quadrature


def bandlimited_upsample(x, factor, xpy=np):
    """Zero-padded-FFT (band-limited) upsample of the LAST axis by `factor`.

    ``x`` is treated as `factor` times oversampled samples of a periodic
    band-limited function; the returned array has ``n*factor`` samples of that
    same function, and reproduces the input exactly at indices ``j*factor``.
    Handles even ``n`` (the Nyquist bin is split half to each of +/-f_Nyq, the
    only choice that keeps a real input real) and odd ``n`` (no Nyquist bin).
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be >= 1, got {}".format(factor))
    if factor == 1:
        return x
    n = x.shape[-1]
    n_dense = n * factor
    X = xpy.fft.fft(x, axis=-1)
    Y = xpy.zeros(x.shape[:-1] + (n_dense,), dtype=complex)
    n_pos = (n + 1) // 2          # bins 0 .. n_pos-1 are the non-negative freqs
    n_neg = n - n_pos             # bins n_pos .. n-1 are the negative freqs
    Y[..., :n_pos] = X[..., :n_pos]
    if n_neg:
        Y[..., n_dense - n_neg:] = X[..., n_pos:]
    if n % 2 == 0:
        # X[n//2] is the Nyquist bin: it is a cosine of ambiguous sign, and the
        # band-limited interpolant is the one that splits it symmetrically.
        half = 0.5 * X[..., n // 2]
        Y[..., n // 2] = half
        Y[..., n_dense - n // 2] = half
    return xpy.fft.ifft(Y, axis=-1) * factor


def peak_width_from_lnL(lnL_t, dx, xpy=np):
    """Width of the SHARPEST feature in each row of ``exp(lnL_t)``, and the argmax.

    Uses the three-point second difference of ``lnL_t`` -- NOT of ``kappa`` and
    NOT of ``exp(lnL_t)`` -- taken at EVERY interior bin, and reports the
    largest curvature found::

        d2[j]   = (lnL[j-1] - 2 lnL[j] + lnL[j+1]) / dx**2
        sigma_t = 1 / sqrt(max_j(-d2[j]))

    Three properties, and all three are load-bearing.

    1. For a Gaussian ``exp(lnL)``, ``lnL`` is exactly quadratic, and the
       second difference of a quadratic is its second derivative at ANY spacing
       and ANY offset between the peak and the grid.  So a peak the grid cannot
       resolve still reports its own width honestly, which is what lets the
       upsampling factor be derived rather than guessed.

    2. It reads ``lnL_t``, so it is agnostic to which ``loglikelihood``
       callback produced it.  A rule built on ``kappa``'s curvature would be
       correct only for the affine default helper and would silently
       under-resolve the phase- and distance-marginalized runs, which are the
       production configurations.

    3. It scans the WHOLE row rather than only the neighbourhood of the argmax.
       The quantity the quadrature needs is the width of the SHARPEST feature
       present, not the width of the tallest one, and when the peak is much
       narrower than the grid no sample need land near it, so in principle the
       coarse argmax can sit on an unrelated, shallower feature.  Being honest
       about what this buys: an A/B of the two rules against the analytic
       reference (band-limited pulses AND adversarial combs, three amplitudes,
       25 seeds, five sub-sample offsets) found NO case where they gave
       different answers -- the worst argmax-rule error was 1.3e-7 nats, the
       same as the sharpest-feature rule's, and on physical kappa(t) the
       sampled argmax does track the true peak (measured widths were stable to
       <3% across a full grid-phase scan on a real injection).  This is
       therefore a conservative variant, not a bug fix: it can only ever return
       a smaller sigma, hence a finer grid, at the cost of one array pass.

    Rows with no concave interior bin (flat or convex: nothing to resolve)
    return ``inf``, i.e. "no upsampling required".

    Returns ``(sigma_t, argmax)``, both shape ``lnL_t.shape[:-1]``.  The argmax
    is returned for the edge-row diagnostic, not for the width.
    """
    n = lnL_t.shape[-1]
    if n < 3:
        raise ValueError("need at least 3 time samples, got {}".format(n))
    d2 = (lnL_t[..., :-2] - 2.0 * lnL_t[..., 1:-1] + lnL_t[..., 2:]) / (dx * dx)

    # Restrict to bins that can actually affect the integral, and drop
    # non-finite curvature.  Both matter in production: the
    # distance-marginalization callback returns -inf outside its table, and
    # (-inf) - 2(-inf) + (-inf) is NaN, which would poison a plain max() into
    # NaN -> "no peak" -> factor 1 -> a silently UNDER-resolved row.
    rowmax = xpy.max(lnL_t, axis=-1, keepdims=True)
    contributes = lnL_t[..., 1:-1] > (rowmax - SUPPORT_NATS)
    usable = contributes & xpy.isfinite(d2)
    curv = xpy.max(xpy.where(usable, -d2, -xpy.inf), axis=-1)

    peaked = curv > 0
    sigma = xpy.where(peaked, 1.0 / xpy.sqrt(xpy.where(peaked, curv, 1.0)),
                      xpy.inf)
    return sigma, xpy.argmax(lnL_t, axis=-1)


def _factor_per_row(sigma, dx, xpy=np):
    """Power-of-two upsampling factor for EACH row, from that row's own width.

    The criterion is per row, so the factor is derived per row rather than
    taking the sharpest row's factor for the whole batch.

    Honest about what this buys, because it is less than it sounds: it helps
    when a batch genuinely mixes broad and sharp integrands, and at rho=10 with
    phase marginalization it does (1772 of 10000 rows derive factor 1 and cost
    nothing).  It does NOT rescue the loud case: at rho=40, 9404 of 10000 rows
    derive the same factor 32, because an extrinsic point far from the source
    still has a band-limited kappa of comparable curvature -- its peak is
    lower, not broader.  The cost there is intrinsic to resolving the
    integrand, and is reported rather than optimised away.
    """
    # sigma = inf means "nothing to resolve" -> need 0 -> factor 1.
    # sigma = 0 means "not resolvable at ANY finite factor" -- the opposite --
    # and must not be folded into the same branch.  It arises if a curvature of
    # -inf ever reaches here (a -inf bin from the distance-marginalization table
    # sitting next to a contributing one); peak_width_from_lnL masks those out,
    # so this is defence in depth, but silently returning a small factor for an
    # unresolvable row is exactly the failure this module exists to prevent.
    if bool(xpy.any(sigma <= 0)):
        raise ValueError(
            "time-marginalization integrand has a non-positive measured width "
            "({:.3e} s): the peak is not resolvable at any finite upsampling "
            "factor.  Refusing to guess.".format(float(xpy.min(sigma))))
    need = UPSAMPLE_SAFETY * dx / sigma          # sigma=inf -> 0
    need = xpy.where(xpy.isfinite(need), need, 0.0)
    factor = 2.0 ** xpy.ceil(xpy.log2(xpy.maximum(need, 1.0)))
    # log2/ceil can land one step short when `need` sits just under a power of
    # two.  Erring low here would silently under-resolve, so check the actual
    # criterion and bump.  (Rows with sigma=inf compare False and stay at 1.)
    factor = xpy.where(dx / factor > sigma / UPSAMPLE_SAFETY, factor * 2, factor)
    # Test the FACTOR against the ceiling, not `need`: need=4097 rounds to 8192,
    # so a check on `need` would let a factor through that exceeds the ceiling.
    worst = float(xpy.max(factor))
    if worst > UPSAMPLE_FACTOR_MAX:
        raise ValueError(
            "time-marginalization integrand needs an upsampling factor of {:.0f} "
            "(sharpest sigma_t={:.3e} s vs grid spacing {:.3e} s), above "
            "UPSAMPLE_FACTOR_MAX={}.  Refusing to return a knowingly "
            "under-resolved integral.".format(
                worst, float(xpy.min(sigma)), dx, UPSAMPLE_FACTOR_MAX))
    return factor.astype(np.int64)


def required_upsample_factor(lnL_t, dx, xpy=np):
    """Upsampling factor needed so that ``dx/factor <= sigma_min/UPSAMPLE_SAFETY``.

    Rounded UP to a power of two: FFT-friendly, and it makes every coarse
    sample land on dense index ``j*factor``, which the tests check exactly.

    Returns ``(factor, sigma_min)``.  ``sigma_min`` is ``inf`` when no row has
    a resolvable peak, in which case the factor is 1 and the caller does no
    interpolation at all.
    """
    sigma, _ = peak_width_from_lnL(lnL_t, dx, xpy=xpy)
    finite = xpy.isfinite(sigma)
    if not bool(xpy.any(finite)):
        return 1, float("inf")
    sigma_min = float(xpy.min(xpy.where(finite, sigma, xpy.inf)))
    if not np.isfinite(sigma_min) or sigma_min <= 0:
        return 1, sigma_min
    # Delegates to the SAME rule the integrator uses.  There were briefly two
    # implementations (a doubling loop here, log2+ceil there); two copies of a
    # numeric rule drift, and one of them had an off-by-one that erred LOW.
    factor = int(xpy.max(_factor_per_row(sigma, dx, xpy=xpy)))
    return factor, sigma_min


def _eval_lnL(kappa, rho_sq, loglikelihood, phase_marginalization, xpy):
    if phase_marginalization:
        return loglikelihood(xpy.abs(kappa), rho_sq)
    return loglikelihood(kappa.real, rho_sq)


def _historical_simpson_log(lnL_rows, deltaT, lnLmax, simps, xpy):
    """The historical expression, verbatim: lnLmax + log(simps(exp(lnL-lnLmax))).

    Same rule, same dx, same global offset, so wrap-exposed rows come back
    bit-for-bit as they would have without this option.
    """
    if simps is None:
        from scipy import integrate
        simps = getattr(integrate, "simpson", None) or integrate.simps
    return lnLmax + xpy.log(simps(xpy.exp(lnL_rows - lnLmax), dx=deltaT, axis=-1))


def _trapezoid_log(lnL_dense, dx, lnLmax, xpy):
    """log( trapz(exp(lnL_dense), dx) ) along the last axis, offset-stabilised.

    `lnLmax` is per row.  A row that is -inf at EVERY bin has a -inf maximum,
    and `-inf - (-inf)` is NaN -- which would propagate into the sampler's
    weights, where -inf merely meant zero weight.  Substituting a finite offset
    for those rows makes the arithmetic give exp(-inf)=0, sum=0, log(0)=-inf,
    i.e. exactly what the historical global-offset path returned for them.
    """
    w = xpy.full(lnL_dense.shape[-1], dx, dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    safe = xpy.where(xpy.isfinite(lnLmax), lnLmax, 0.0)
    return xpy.log(xpy.sum(w * xpy.exp(lnL_dense - safe[..., None]), axis=-1))


def time_marginalize_bandlimited(kappa, rho_sq, deltaT, loglikelihood,
                                 phase_marginalization=False, lnL_t=None,
                                 simps=None, lnLmax=None, xpy=np):
    """``log \\int dt exp(lnL(t))`` with the integrand resolved, not assumed.

    Parameters
    ----------
    kappa : (n_rows, npts) complex
        The accumulated data term on the coarse deltaT grid.
    rho_sq : (n_rows, npts) or (n_rows, 1) float
        The template self-term.  MUST be constant along the time axis -- that
        is what makes ``lnL(t)`` a pointwise function of the band-limited
        ``kappa(t)``.  Checked, not assumed: a future banded / slow-rotation
        backport has a genuinely time-dependent ``rho_sq`` and must not reach
        this code silently.
    deltaT : float
        The spacing the likelihood actually steps by -- one SAMPLE.  Pass it
        explicitly; do not infer it from a ``tvals`` array.  On rift_O4c the
        window grid is a closed-interval ``linspace``, so its spacing is
        ``deltaT*N/(N-1)``, ~0.2% coarser than deltaT, and inferring ``dx``
        from it would be quietly wrong.
    loglikelihood : callable
        ``f(kappa_real_or_abs, rho_sq) -> lnL``.  Applied on the DENSE grid, so
        nonlinear callbacks (phase marginalization, the distance-marginalization
        table) are handled correctly rather than approximated.
    lnL_t : optional
        The coarse-grid ``lnL(t)`` if the caller already computed it; saves one
        evaluation.  Must correspond to `kappa`/`rho_sq`.
    simps, lnLmax : optional
        The caller's Simpson implementation and its global log-offset, used
        verbatim for wrap-exposed rows so their values are bit-for-bit the
        historical ones.  Supplied by the caller rather than imported here so
        the GPU path keeps its own ``optimized_gpu_tools.simps`` and this module
        stays backend-generic.  Falls back to ``scipy`` if omitted.

    Returns
    -------
    lnL : (n_rows,) float
    """
    kappa = xpy.asarray(kappa)
    rho_sq = xpy.asarray(rho_sq)
    npts = kappa.shape[-1]

    if rho_sq.shape[-1] != 1:
        # Cheap, and it is the guard that stops a time-dependent self-term from
        # being interpolated as if it were constant.
        if not bool(xpy.all(rho_sq == rho_sq[..., :1])):
            raise NotImplementedError(
                "time_quadrature='bandlimited' requires a time-independent "
                "rho_sq; this rho_sq varies along the time axis (banded / "
                "slow-rotation response?).  The band-limited argument applies "
                "to kappa(t) only, so that path must be handled separately.")
    rho_col = rho_sq[..., :1]

    if lnL_t is None:
        lnL_t = _eval_lnL(kappa, rho_sq, loglikelihood, phase_marginalization, xpy)

    n_rows = int(np.prod(kappa.shape[:-1])) if kappa.ndim > 1 else 1
    k2 = kappa.reshape(n_rows, npts)
    r2 = xpy.broadcast_to(rho_col.reshape(-1, 1), (n_rows, 1))
    lnL2 = lnL_t.reshape(n_rows, npts)
    if lnLmax is None:
        lnLmax = xpy.max(lnL2)

    sigma_rows, jmax = peak_width_from_lnL(lnL2, deltaT, xpy=xpy)
    has_peak = xpy.isfinite(sigma_rows)

    # --- which rows are REFINED, and which are handed back untouched ---
    #
    # "not refined" and "gets the historical value" are ONE predicate, not two.
    # Keeping them separate is how a row ends up neither refined nor Simpson --
    # e.g. a signal-free row falling through to a coarse-grid TRAPEZOID, which
    # is a different rule from Simpson even though it interpolates nothing.
    # With one predicate the property a reviewer can check by reading is:
    #
    #     a row's value differs from the historical one IF AND ONLY IF its
    #     integrand was under-resolved.
    #
    # Wrap-exposed: the peak sits too near a window edge for the periodic
    # interpolant to be trusted.  The guard only claims rows that HAVE a
    # resolvable peak -- a signal-free row is constant, so its argmax is 0, and
    # claiming it would report "the window is mis-centred" when it means "these
    # samples have no signal".
    guard = max(1, int(EDGE_GUARD_FRACTION * npts))
    exposed = has_peak & ((jmax < guard) | (jmax >= npts - guard))
    n_exposed = int(xpy.sum(exposed))
    n_edge = int(xpy.sum((jmax <= 0) | (jmax >= npts - 1)))
    n_nonfinite = int(xpy.sum(~xpy.isfinite(xpy.max(lnL2, axis=-1))))
    n_flat = int(xpy.sum((~has_peak) & xpy.isfinite(xpy.max(lnL2, axis=-1))))

    # Exposed rows are excluded from the factor derivation as well: otherwise a
    # single mis-centred row inflates the refinement every healthy row pays for
    # -- and, since the ceiling RAISES rather than truncating, one such row
    # could abort a batch it is not even going to be integrated from.
    sigma_eff = xpy.where(exposed, xpy.inf, sigma_rows)
    sigma_min = (float(xpy.min(xpy.where(has_peak & (~exposed), sigma_eff, xpy.inf)))
                 if bool(xpy.any(has_peak & (~exposed))) else float("inf"))
    factor_rows = _factor_per_row(sigma_eff, deltaT, xpy=xpy)
    factor_initial = int(xpy.max(factor_rows)) if n_rows else 1

    refined = has_peak & (~exposed) & (factor_rows > 1)
    n_refined = int(xpy.sum(refined))
    # Rows with a peak the grid ALREADY resolves.  Counted so the five buckets
    # partition the batch exactly -- without it a reader cannot tell "resolved,
    # so nothing to do" from a row that fell through some gap.
    n_resolved = int(xpy.sum(has_peak & (~exposed) & (factor_rows == 1)))

    out = xpy.empty(n_rows, dtype=np.float64)
    keep = ~refined
    if bool(xpy.any(keep)):
        idx = xpy.nonzero(keep)[0]
        out[idx] = _historical_simpson_log(lnL2[idx], deltaT, lnLmax, simps, xpy)

    hist, n_chunks, n_dense_max = {}, 0, 0
    sigma_dense_min = np.inf

    for f0 in sorted(set(int(x) for x in np.asarray(
            factor_rows if xpy is np else factor_rows.get()))):
        rows = xpy.nonzero((factor_rows == f0) & refined)[0]
        n_g = int(rows.size)
        if n_g == 0:
            continue
        kg, rg = k2[rows], r2[rows]
        factor = f0
        while True:
            # Domain: exactly [t_0, t_{npts-1}], the closed interval Simpson
            # used.  The only dense samples dropped are the factor-1 past the
            # last coarse sample -- precisely the ones across the FFT's
            # periodic wrap.  No interior sample is dropped.
            n_keep = (npts - 1) * factor + 1
            dx = deltaT / factor
            per_row = n_keep * (16 + 8 * 6)   # dense complex row + callback temporaries
            chunk = max(1, min(n_g, int(_DENSE_CHUNK_BYTES // max(per_row, 1))))
            vals = xpy.empty(n_g, dtype=np.float64)
            sd_min = np.inf
            n_ch = 0
            for lo in range(0, n_g, chunk):
                hi = min(lo + chunk, n_g)
                n_ch += 1
                kd = bandlimited_upsample(kg[lo:hi], factor, xpy=xpy)[..., :n_keep]
                rd = xpy.broadcast_to(rg[lo:hi], (hi - lo, n_keep))
                lnLd = _eval_lnL(kd, rd, loglikelihood, phase_marginalization, xpy)
                sd, _ = peak_width_from_lnL(lnLd, dx, xpy=xpy)
                fin = xpy.isfinite(sd)
                if bool(xpy.any(fin)):
                    sd_min = min(sd_min,
                                 float(xpy.min(xpy.where(fin, sd, xpy.inf))))
                # PER-ROW offset, taken on the DENSE grid.  Both halves matter.
                # Reusing the coarse grid's maximum overflows: under-resolution
                # means the dense maximum can exceed it by thousands of nats, so
                # exp(v - coarse_max) returns +inf -- and it fails precisely in
                # the regime this option exists to fix.  Reusing the historical
                # GLOBAL offset underflows the other way: every row more than
                # ~745 nats below the block maximum becomes exp()=0 and then
                # log(0) = -inf.  The expression is offset-invariant, so per-row
                # is not a change of estimator.
                m = xpy.max(lnLd, axis=-1)
                # m is -inf only when the row is -inf everywhere; _trapezoid_log
                # then returns -inf, and -inf + -inf is -inf, which is right.
                vals[lo:hi] = m + _trapezoid_log(lnLd, dx, m, xpy)

            # Re-verify on the grid actually integrated.  A coarse-grid width
            # can be optimistic for a strongly non-Gaussian peak; this is what
            # turns the derivation into an assertion rather than a guess.
            if not np.isfinite(sd_min) or dx <= sd_min / UPSAMPLE_SAFETY:
                break
            if factor * 2 > UPSAMPLE_FACTOR_MAX:
                raise ValueError(
                    "time-marginalization integrand still under-resolved at "
                    "factor={} (dense sigma_t={:.3e} s vs spacing {:.3e} s) and "
                    "doubling would exceed UPSAMPLE_FACTOR_MAX={}.".format(
                        factor, sd_min, dx, UPSAMPLE_FACTOR_MAX))
            factor *= 2

        out[rows] = vals
        hist[factor] = hist.get(factor, 0) + n_g
        n_chunks += n_ch
        n_dense_max = max(n_dense_max, n_keep)
        sigma_dense_min = min(sigma_dense_min, sd_min)

    _LAST_REPORT.clear()
    _LAST_REPORT.update(
        factor=max(hist) if hist else 1,
        factor_initial=factor_initial,
        factor_histogram=dict(sorted(hist.items())),
        sigma_t_min=sigma_min, sigma_t_min_dense=sigma_dense_min,
        npts=npts, npts_dense=n_dense_max, n_rows=n_rows, n_chunks=n_chunks,
        n_edge_peak_rows=n_edge, n_wrap_exposed_rows=n_exposed,
        n_nonfinite_rows=n_nonfinite, n_flat_rows=n_flat,
        n_refined_rows=n_refined, n_resolved_rows=n_resolved)
    return out.reshape(kappa.shape[:-1])
