"""Band-limited time marginalization for the factored likelihood.

WHAT IS WRONG WITH THE HISTORICAL PATH
--------------------------------------
``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`` forms ``lnL(t)`` on the
grid built by ``factored_likelihood.marginalization_time_grid`` -- spacing
``deltaT = 1/srate``, fixed -- and integrates ``exp(lnL(t))`` over it with
Simpson's rule at that fixed spacing.  The grid spacing is a property of the
DATA; the integrand's width is a property of the SIGNAL, and the two are
unrelated.  After marginalizing the angles, ``exp(lnL(t))`` is a near-Gaussian
peak of width

    sigma_t = 1 / (2 pi rho sigma_f)

with ``sigma_f`` the noise-weighted template frequency spread.  Resolving it
needs ``deltaT <~ sigma_t``, i.e. ``srate >~ 2 pi sigma_f rho`` -- a requirement
that grows LINEARLY WITH SNR and that production, which runs at the data sample
rate, does not meet.  Simpson makes the under-resolved case worse rather than
better: ``simpson = (4 T_h - T_2h)/3`` carries the coarser trapezoid ``T_2h``,
so it inherits an alias with period ``2h``.

Measured on a 35+30 Msun SEOBNRv4 H1L1V1 injection at rho=40 (sigma_t = 61.2 us,
a property of the signal and so srate-independent), rigidly scanning the grid
phase over 2*deltaT moves the reported lnL by

    srate 4096: 1.649 nats     8192: 0.385 nats     16384: 0.0095 nats

i.e. the answer depends on where the sample grid happens to fall relative to the
peak, by over a nat at the production sample rate.  (Those three numbers were
taken on the JAX mirror of this quadrature, which integrates the SAME grid with
the SAME fixed Simpson weights; they are quoted as the physical scale of the
defect.  The numbers for THIS path are the synthetic ones below, which are
against an analytic truth rather than against another estimate.)

WHY THE EXISTING SAMPLES ALREADY CONTAIN THE ANSWER
---------------------------------------------------
The data term ``kappa(t) = sum_det <h_det | d_det>(t)`` is built from the
precomputed rholm cross-correlation timeseries, which are inverse FFTs of a
frequency-domain product band-limited to ``[fmin, fmax]`` with
``fmax <= fNyq = 1/(2 deltaT)``.  So ``kappa(t)`` is band-limited below Nyquist,
and by the sampling theorem the samples the code ALREADY COMPUTES determine the
continuous function exactly.  The template self-term ``rho_sq`` is
time-independent on this path.  Therefore ``lnL(t) = f(kappa(t), rho_sq)`` is
recoverable on an arbitrarily fine grid from the samples in hand -- one
zero-padded FFT per row, no extra likelihood evaluations, no extra precompute
and no extra accumulator passes.

Two independent checks of that claim.  On the real injection above, against a
converged dense reference built by re-gathering the rholms at shifted window
offsets -- the expensive, genuinely independent construction -- at srate 4096,
rho=40:

    band-limited upsampling:  -0.007 nats
    Simpson at deltaT      :  +0.745 nats

And for this path specifically, against an ANALYTIC truth rather than another
numerical estimate (see test_time_marginalization_quadrature.py: kappa is built
as a sum of complex exponentials below Nyquist, so the continuous function is
known in closed form).  srate 4096, npts 614, error in nats at three grid phases:

    sigma_t/deltaT   Simpson                         band-limited   factor
      2.27-2.61      +5e-6  ..  +0e0                 0              1
      0.72-0.83      +2.5e-3 .. -1.1e-4              0              4
      0.25-0.28      +0.844 / +0.242 / -1.101        0              16
      0.10-0.12      +1.742 / -1.833 / -11.68        0              32
      0.046-0.052    +2.548 / -15.33 / -66.18        0              64
      0.016-0.019    +3.589 / -139.4 / -549.0        0              256
      0.007-0.008    +4.393 / -710.6 / -2760.4       0              512

Two things to read off that table.  The first row is the reassuring one: where
the integrand is already resolved the historical rule is fine, the derivation
returns a factor of 1 and nothing is paid.  The row at sigma_t/deltaT = 0.25
spans 1.95 nats across grid phase, which is the same scale as the 1.649 nats
measured on the real injection at the same ratio -- the synthetic reproduces the
defect's magnitude rather than a caricature of it.  On a window cut from a longer
band-limited signal (so the periodic interpolant genuinely rings at the wrap, the
realistic case) the band-limited error stays at or below 5e-5 nats where Simpson
is off by up to 420.

RESOLUTION IS DERIVED, NOT CONFIGURED
-------------------------------------
There is deliberately no "upsampling factor" option.  This defect class exists
because a resolution was once a settable number whose docstring claimed it was
ample.  The factor here is derived from the integrand actually in hand: the
three-point second difference of ``lnL`` about its peak is an EXACT estimator of
``-1/sigma_t^2`` for a Gaussian peak at any grid spacing and any peak-vs-grid
phase, so the width can be measured on the coarse grid even when the peak is
badly under-resolved by it.  The factor is then raised to a power of two
satisfying ``h_dense <= sigma_t / SAFETY``, and -- this is the part that makes it
an assertion rather than a guess -- the width is REMEASURED on the dense grid and
the factor doubled until the criterion holds there too.  A flat integrand
measures an infinite width, derives a factor of 1 and costs nothing.

``SAFETY = 2`` is not tunable and is not a compromise.  The trapezoidal rule on
a Gaussian of width ``sigma`` at spacing ``h`` has relative error
``2 exp(-2 pi^2 sigma^2 / h^2)`` (Poisson summation); at ``h = sigma/2`` that is
``1.0e-34``.  Even ``h = sigma`` would give ``5.3e-9``.

That bound is the DESIGN CRITERION FOR THE REFINED GRID, where ``sigma/h >= 2``
puts us deep in the exponential regime.  It is NOT a model of the defect's
magnitude, and checking it against the measured Simpson errors above will not
work: at ``sigma/h ~ 0.1-0.4`` the peak is barely sampled at all, the asymptotic
alias picture has not engaged, and the observed spans scale roughly as ``h^2``.
Both statements are correct about different regimes.  The quadrature on the dense
grid is TRAPEZOID, not Simpson: for a peak that decays to nothing inside the
window every Euler-Maclaurin boundary term vanishes, so the trapezoidal rule is
spectrally accurate there, while Simpson would reintroduce the ``2h`` alias that
is the original defect.

WHAT "EXACTLY" IS EXACT ABOUT (read before quoting the accuracy numbers)
-----------------------------------------------------------------------
The reconstruction is exact for the integrand THE CODE ACTUALLY FORMS, which is
the true ``kappa(t)`` only when ``time_interp='nearest'`` -- the default -- where
the gathered values are exact samples of ``Q`` (on a grid offset by up to
deltaT/2, which is a pre-existing property of that stencil).

With ``time_interp='cubic'`` or ``'sinc'`` the gathered values are a fixed FIR
filter applied to ``Q``.  That is still a band-limited sequence, so the sampling
argument survives and the refinement is still exact -- but exact for the FILTERED
function.  It then converges precisely to the integral of a stencil-biased
integrand, and the stencil bias, not the quadrature, is the larger term.
Measured at srate 4096, peak lnL ~5300, peak centred: with 'nearest' this path is
+0.0002 nats against an analytic truth where Simpson is -521; with 'sinc' it is
-2.29 where Simpson is +1.28, and over a scan of seeds and grid phases Simpson
wins about half the cases.  Neither number says the quadrature is wrong -- they
say that once a stencil is in use its own error dominates, and fixing the
quadrature exposes it rather than adding to it.  The advantages quoted above are
for the default stencil.

SCOPE
-----
Applies to the baseline (non-rotating) likelihood with ``n_cal == 1``.  The
banded / slow-rotation path has a TIME-DEPENDENT ``rho_sq`` and sidereal
post-phases, so ``lnL(t)`` there is not a pointwise function of a band-limited
``kappa`` and a constant, and the argument above does not carry; that path
refuses this quadrature rather than silently mis-applying it.  Calibration
marginalization (``n_cal > 1``) is excluded for now for the same
refuse-rather-than-guess reason: the reduction sums ``exp`` over realizations, so
each realization's kappa row would have to be upsampled and the derived factor
reconciled across realizations, which is untested here.
"""

import numpy as np

__all__ = [
    "TIME_QUADRATURE_CHOICES",
    "UPSAMPLE_SAFETY",
    "UPSAMPLE_FACTOR_MAX",
    "EDGE_GUARD_FRACTION",
    "bandlimited_upsample",
    "peak_width_from_lnL",
    "required_upsample_factors",
    "validate_time_quadrature",
    "time_marginalize_bandlimited",
    "last_report",
]

TIME_QUADRATURE_CHOICES = ("simpson", "bandlimited")

#: ``h_dense <= sigma_t / UPSAMPLE_SAFETY``.  See the module docstring: at this
#: value the trapezoidal rule's Poisson-summation error on a Gaussian peak is
#: 2e-34, so this is a hard-coded constant and not an accuracy/cost trade.
UPSAMPLE_SAFETY = 2.0

#: Fail-closed ceiling.  The band limit bounds the useful factor: with
#: ``sigma_t >= deltaT / (pi rho)`` the derivation cannot legitimately ask for
#: more than ``~2 rho``.  Exceeding this raises rather than silently truncating
#: the resolution.
UPSAMPLE_FACTOR_MAX = 4096

#: Fraction of the window at EACH end within which a row's peak is treated as
#: wrap-exposed.  The zero-padded FFT reconstructs the unique PERIODIC
#: band-limited interpolant through the window's samples; the true kappa is a
#: segment of a longer function and is not periodic on the window, so the
#: endpoint mismatch rings, and the ringing contaminates the reconstruction most
#: where the peak sits closest to the wrap.  Crucially this deviation is NOT
#: measurable from the window's own samples -- the periodic interpolant is
#: uniquely determined by them, so any estimate of the departure needs
#: information from outside the window.  It therefore has to be bounded a priori,
#: and rows that fall outside the bound fall back rather than guess.
#:
#: Measured on a window cut from a longer band-limited signal (peaked kernel plus
#: a 12%-amplitude coloured background, so the two ends genuinely disagree),
#: sigma_t/deltaT = 0.042, error against the analytic continuous truth, in nats:
#:
#:   peak distance from edge   307     100      30      8       2       0
#:   band-limited              5e-6   4.6e-3  5.2e-2  5.6e-2  -3.3    +88.8
#:   Simpson (for scale)      -29.2   -29.9   -29.3   -29.4   -29.7   -29.9
#:
#: The +88.8 is the reason this is a guard and not just a report: it is wrong in
#: the DANGEROUS direction, and a spuriously high lnL importance-weights that
#: sample into dominance.  1/8 of the window is 77 samples at the production
#: npts=614.  TWO HONEST CAVEATS on that choice, both measured:
#:
#: * The table above is at ONE amplitude.  ``lnL`` is LINEAR in ``kappa``, so the
#:   wrap error in nats scales with it: for a row just outside the guard, at peak
#:   ``lnL`` of 5.3e2 / 5.3e3 / 5.3e4 / 5.3e5 (rho ~ 33 / 103 / 326 / 1031), the
#:   measured error was -8.0e-4 / -8.1e-3 / -8.1e-2 / -0.846 nats.  So the fixed
#:   fraction is a bound on WHERE, not on HOW MUCH: adequate through O4
#:   amplitudes, weaker in the 3G regime.
#: * Do NOT justify the guard by saying such rows are truncated anyway.  Often
#:   they are not -- at 20-60 samples from the edge the peak sits entirely inside
#:   the window, yet those rows get a Simpson value measured 2.87 nats wrong where
#:   the reconstruction would have been 0.007-0.02.  The guard is deliberately
#:   conservative: the crossover where the reconstruction actually loses is nearer
#:   5-10 samples, and 1/8 buys margin against the amplitude scaling above.
#:
#: In a well-posed run nothing comes close: the grid is centred on the trigger's
#: geocentre time, so the peak sits within the trigger timing uncertainty (a few
#: ms, tens of samples) of the CENTRE, not of an edge.  Rows that do violate it
#: are given the historical Simpson value and counted in ``last_report()``.
#: (The route to supporting such rows properly is to widen the GATHER so the wrap
#: sits outside the integration domain -- deliberately not done here, since it
#: touches the GPU kernel and the buffer-margin assumptions.)
EDGE_GUARD_FRACTION = 0.125

#: Half-widths, in coarse samples, tried in turn for the curvature stencil.  A
#: centred three-point stencil at the peak is the natural choice, but ``lnL_t``
#: genuinely contains ``-inf`` in production -- the distance-marginalization
#: callback returns ``-inf`` outside its interpolation table -- and
#: ``(-inf) - 2*(-inf) + (-inf)`` is NaN, while ``NaN < 0`` is False.  A row whose
#: stencil straddles the table edge would therefore report "no resolvable peak",
#: derive a factor of 1, and be SILENTLY UNDER-RESOLVED: no raise, no warning,
#: and the exact failure this whole change exists to remove.  Widening the
#: stencil steps over the hole, and costs nothing in accuracy because the second
#: difference of a parabola is its second derivative at ANY spacing.  A row where
#: no half-width yields a finite curvature is not guessed at: it is counted and
#: given the historical value.
CURVATURE_STENCIL_HALFWIDTHS = (1, 2, 4, 8)

#: Working-set budget for one dense temporary, in bytes.  Purely an internal
#: memory-chunking parameter: it changes how many extrinsic rows are processed at
#: a time and cannot change the answer.
_DENSE_CHUNK_BYTES = 128 * 1024 * 1024

_LAST_REPORT = {}


def last_report():
    """Diagnostics from the most recent :func:`time_marginalize_bandlimited` call.

    Keys: ``upsample_factor`` (the largest used), ``factor_histogram``
    (factor -> row count, over the rows that were refined), ``n_refinements``,
    ``sigma_t_min``, ``n_rows``, ``n_wrap_exposed_rows``, ``n_unmeasurable_rows``,
    ``n_flat_rows``, ``n_refined_rows``.

    The three row counts are deliberately kept apart, because they mean different
    things and only two of them are ever worth acting on:

    ``n_wrap_exposed_rows`` -- a resolvable peak sitting inside
    ``EDGE_GUARD_FRACTION`` of a window edge.  This is a statement that the
    WINDOW is mis-centred for those samples, which truncates their integral under
    either rule.  Given the historical Simpson value.

    ``n_unmeasurable_rows`` -- ``lnL(t)`` non-finite around its maximum at every
    stencil half-width, so no width can be justified.  Given the historical value.

    ``n_flat_rows`` -- finite ``lnL(t)`` with no resolvable curvature: an
    extrinsic sample with no signal in it.  Nothing is wrong and nothing is paid.

    ``n_refined_rows`` is the count that matters for auditing a change: the
    QUADRATURE RULE changes for these rows and for no others.  Every other row --
    exposed, unmeasurable, or already resolved -- is integrated by the caller's
    own Simpson rule over the same domain.

    Read that precisely: it is a statement about the RULE, not about the returned
    VALUE.  The log-sum-exp offset also changes, from the historical single
    global maximum over the whole block to a per-row maximum, and that applies to
    every row including the unrefined ones.  It is unavoidable on the refined
    path (see :func:`_log_trapz_over_window`) and it has a visible consequence:
    a row far enough below the block maximum that ``exp(lnL - global_max)``
    underflowed -- which happens once a batch spans more than ~745 nats, routine
    at rho >~ 40 -- returned ``-inf`` historically and now returns a finite
    value, refined or not.  On rows that did not underflow the two agree to
    floating-point rounding, not bit-for-bit.
    """
    return dict(_LAST_REPORT)


def validate_time_quadrature(time_quadrature):
    if time_quadrature not in TIME_QUADRATURE_CHOICES:
        raise ValueError(
            "time_quadrature must be one of {}, got {!r}".format(
                TIME_QUADRATURE_CHOICES, time_quadrature))
    return time_quadrature


def bandlimited_upsample(x, factor, xpy=np):
    """Zero-padded-FFT upsample of complex rows ``x`` (..., n) by ``factor``.

    Exact for a sequence of samples of a function band-limited below Nyquist and
    periodic on the window, which is what the rholm timeseries are by
    construction (they are inverse FFTs of a band-limited product).  The output
    has ``n*factor`` columns and reproduces the input exactly at every
    ``factor``-th column.

    ODD ``n`` is not a special case to be waved through: an earlier version split
    the spectrum at ``h = n//2`` unconditionally, which for odd ``n`` puts the
    HIGHEST POSITIVE frequency at a negative frequency in the padded array.  The
    reconstruction then stays exact at the original samples -- so "it reproduces
    the input" still passes -- while being wrong everywhere in between.  Measured
    against the analytic band-limited truth at ``factor=4``: max error 1.4e-12 at
    ``n=614`` but 4.1e-1 at ``n=613``, 5.4e-1 at ``n=307`` and 6.0e-2 at
    ``n=2457``.  That matters because ``marginalization_time_grid`` produces ODD
    ``npts`` at three of the five production sample rates -- 153 at srate 1024,
    307 at 2048 and 2457 at 16384 -- so the broken case was the common one.

    The Nyquist bin exists only for even ``n``, and is split evenly between
    ``+fNyq`` and ``-fNyq``.  For the rholm data it is empty anyway (the
    two-sided weight construction never populates it), so that half is a
    formality; the positive/negative boundary is not.
    """
    factor = int(factor)
    if factor == 1:
        return x
    n = x.shape[-1]
    lead = x.shape[:-1]
    X = xpy.fft.fft(x, axis=-1)
    Xup = xpy.zeros(lead + (n * factor,), dtype=xpy.asarray(X).dtype)
    n_pos = (n - 1) // 2                 # DC plus n_pos strictly-positive bins
    Xup[..., :n_pos + 1] = X[..., :n_pos + 1]
    if n % 2 == 0:
        nyq = X[..., n // 2]
        Xup[..., n // 2] = 0.5 * nyq
        Xup[..., -(n // 2)] = 0.5 * nyq
        Xup[..., -n_pos:] = X[..., n // 2 + 1:]
    else:
        Xup[..., -n_pos:] = X[..., n_pos + 1:]
    return xpy.fft.ifft(Xup, axis=-1) * factor


def peak_width_from_lnL(lnL_t, dx, xpy=np):
    """Per-row Gaussian width ``sigma_t`` of ``exp(lnL_t)``, from its peak curvature.

    Uses a centred second difference of ``lnL`` (not of ``exp lnL``) about the
    peak sample.  For a Gaussian ``lnL`` this returns ``sigma`` EXACTLY at any
    spacing, any peak-vs-grid phase, and any stencil half-width, because the
    second difference of a parabola is its second derivative; that is what lets an
    under-resolved peak still report its own width honestly, and it is why
    stepping the stencil out over a ``-inf`` hole costs nothing.

    Returns ``(sigma_t, jmax, measurable)``.  ``sigma_t`` is ``inf`` for a row
    with non-negative curvature -- flat or monotone, where no refinement is
    warranted.  ``measurable`` distinguishes that legitimate case from a row whose
    curvature could not be evaluated at all; the caller must not treat the two
    alike, since "flat" means no refinement is NEEDED while "unmeasurable" means
    none can be JUSTIFIED.
    """
    n = lnL_t.shape[-1]
    if n < 3:
        raise ValueError("need at least 3 time samples to measure a peak width")
    jmax = xpy.argmax(xpy.where(xpy.isfinite(lnL_t), lnL_t, -np.inf), axis=-1)
    take = lambda j: xpy.take_along_axis(lnL_t, j[..., None], axis=-1)[..., 0]

    sigma = xpy.full(jmax.shape, np.inf, dtype=np.float64)
    measurable = xpy.zeros(jmax.shape, dtype=bool)
    for d in CURVATURE_STENCIL_HALFWIDTHS:
        if 2 * d >= n:
            break
        jc = xpy.clip(jmax, d, n - 1 - d)
        with np.errstate(invalid='ignore'):
            # inf - inf is exactly the case being handled; the NaN it produces is
            # the signal that this half-width straddles a hole, not an anomaly.
            d2 = (take(jc - d) - 2.0 * take(jc) + take(jc + d)) / float(d * dx) ** 2
        fresh = xpy.isfinite(d2) & (~measurable)
        if not bool(xpy.any(fresh)):
            continue
        neg = fresh & (d2 < 0)
        sigma = xpy.where(neg, 1.0 / xpy.sqrt(xpy.where(neg, -d2, 1.0)), sigma)
        measurable = measurable | fresh
        if bool(xpy.all(measurable)):
            break
    return sigma, jmax, measurable


def required_upsample_factors(sigma, dx, xpy=np):
    """Per-row power-of-two factor with ``dx/factor <= sigma/UPSAMPLE_SAFETY``.

    PER ROW, deliberately.  A single block-wide factor is correct but ruinous:
    the handful of rows near the source impose their resolution on every other
    row in the batch, and the cost is dominated by the likelihood callback and
    ``exp`` over ``n_rows * npts * factor`` points.  Measured on the companion
    O4c line at ``--n-chunk 10000``, srate 4096, one block-wide factor cost 18x
    the Simpson likelihood call at rho=40 and 39x at rho=80 -- in the ILE inner
    loop.  Grouping by the derived factor leaves every row meeting its own
    criterion while the broad majority stop paying for the sharpest few.
    """
    need = UPSAMPLE_SAFETY * float(dx) / xpy.where(xpy.isfinite(sigma) & (sigma > 0),
                                                   sigma, np.inf)
    need = xpy.where(need > 1.0, need, 1.0)
    factor = xpy.exp2(xpy.ceil(xpy.log2(need)))
    # 2**ceil(log2(need)) can land one power of two SHORT when log2 rounds down
    # for a `need` a hair above a power of two -- erring LOW, i.e. silently
    # under-resolving, which is the failure mode this whole module exists to
    # remove.  The criterion is `factor >= need`, so test exactly that and bump.
    # (The margin can only ever be one ulp, so a single bump closes it.)
    factor = xpy.where(factor < need, 2.0 * factor, factor)
    # Clamp BEFORE the cast.  A float factor above 2**63 wraps to a large
    # NEGATIVE int64, which downstream `maximum(factors, 1)` turns into 1 -- so an
    # unresolvably sharp row would be classified "nothing to refine" and silently
    # given the coarse value, which is the failure this module exists to remove.
    # Saturating at the ceiling instead sends it into the loop, which raises.
    factor = xpy.where(factor > UPSAMPLE_FACTOR_MAX,
                       float(2 * UPSAMPLE_FACTOR_MAX), factor)
    return factor.astype(np.int64)


def _safe_offset(off, xpy=np):
    """Log-sum-exp offset, guarded for a row that is ``-inf`` everywhere.

    Such a row has zero likelihood over the whole window, and the historical path
    returns ``-inf`` for it (its GLOBAL offset is finite, so every term underflows
    to zero and ``log(0)`` follows).  A per-row offset would instead compute
    ``-inf - (-inf) = NaN`` and hand back a NaN that propagates into the sampler
    weights.  Substituting a finite offset reproduces ``-inf`` exactly.
    """
    return xpy.where(xpy.isfinite(off), off, 0.0)


def _log_simps_rows(lnL_t, dx, simps, xpy=np):
    """``log \\int exp(lnL) dt`` by the caller's Simpson rule, per row.

    The caller's rule, not a private copy: on GPU the likelihood integrates with
    ``optimized_gpu_tools.simps``, so a scipy copy here would agree on CPU and
    quietly disagree on the device -- and the whole point of this path is to
    reproduce what the historical code would have returned for these rows.
    """
    off = _safe_offset(xpy.max(lnL_t, axis=-1, keepdims=True), xpy=xpy)
    return off[..., 0] + xpy.log(simps(xpy.exp(lnL_t - off), dx=float(dx), axis=-1))


def _log_trapz_over_window(lnL_dense, dx_dense, npts_coarse, factor, xpy=np):
    """``log \\int exp(lnL) dt`` by trapezoid over the ORIGINAL window span.

    The dense grid returned by the FFT upsample is periodic on
    ``[t_0, t_0 + npts*deltaT)``, i.e. it carries ``factor-1`` samples PAST the
    last coarse sample.  Those lie across the periodic wrap and are dropped, so
    the integration domain is exactly ``[t_0, t_{npts-1}]`` -- identical to the
    domain Simpson used.  Changing the domain would have been a second,
    confounded change.

    The log-sum-exp offset is PER ROW and taken on the DENSE grid, not the single
    global coarse maximum the Simpson path uses.  It has to be: the whole point of
    refining the grid is that the true peak sits between coarse samples, so the
    dense maximum can exceed the coarse one -- by thousands of nats for a sharp
    peak -- and offsetting by the coarse maximum overflows ``exp()`` precisely in
    the regime this quadrature exists to serve.  (A per-row offset also avoids the
    underflow-to-``log(0) = -inf`` that a shared global offset gives rows far below
    the block maximum.)  The result is offset-invariant, so this is a numerical
    choice and not a change of estimator.
    """
    last = (npts_coarse - 1) * factor
    v = lnL_dense[..., :last + 1]
    w = xpy.full(v.shape[-1], dx_dense, dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    off = _safe_offset(xpy.max(v, axis=-1, keepdims=True), xpy=xpy)
    return off[..., 0] + xpy.log(xpy.sum(xpy.exp(v - off) * w, axis=-1))


def time_marginalize_bandlimited(kappa, rho_sq, deltaT, loglikelihood,
                                 phase_marginalization=False, simps=None,
                                 lnL_coarse=None, xpy=np):
    """``log \\int dt exp(lnL(t))`` with the time grid refined to the integrand.

    Parameters
    ----------
    kappa : (n_extrinsic, npts) complex
        The accumulated data term on the coarse grid, exactly as the caller
        already builds it.  Band-limited below Nyquist by construction.
    rho_sq : (n_extrinsic, npts) float
        The template self-term.  MUST be constant along the time axis on this
        path; that is what makes ``lnL(t)`` a pointwise function of a
        band-limited quantity.  Checked, not assumed.
    loglikelihood : callable
        ``f(kappa_term, rho_sq) -> lnL``, the same callback the caller passes to
        the coarse path (default helper, phase- or distance-marginalized).
    simps : callable, optional
        The caller's Simpson rule, ``simps(y, dx=..., axis=-1)``, used for rows
        that fall back to the historical path.  Defaults to scipy's -- which
        RAISES on a cupy array, so the GPU caller must supply its own.
    lnL_coarse : array, optional
        ``loglikelihood`` already evaluated on the coarse grid.  The caller
        normally has it; passing it avoids re-evaluating the callback over
        ``n_extrinsic * npts`` points, which for the distance-marginalized
        callback is a table interpolation over millions of points and is the
        difference between "no extra likelihood evaluations" being true and
        being nearly true.

    Returns
    -------
    lnL : (n_extrinsic,) float
    """
    if simps is None:
        # Default ONLY for the numpy backend.  scipy's simpson raises
        # `TypeError: Implicit conversion to a NumPy array is not allowed` on a
        # cupy array, and that default is exactly how every --vectorized --gpu run
        # of this option crashed.  Refuse rather than leave the trap armed.
        if xpy is not np:
            raise ValueError(
                "time_marginalize_bandlimited: `simps` must be supplied for a "
                "non-numpy backend -- scipy's Simpson rule cannot consume a device "
                "array, and the fallback rows must use the rule the caller's own "
                "likelihood uses (on GPU, optimized_gpu_tools.simps).")
        from scipy import integrate
        simps = getattr(integrate, 'simpson', None) or integrate.simps

    kappa = xpy.asarray(kappa)
    rho_sq = xpy.asarray(rho_sq)
    npts = kappa.shape[-1]
    n_rows = kappa.shape[0]
    deltaT = float(deltaT)

    # rho_sq time-independence is the load-bearing precondition, so verify it
    # rather than trusting the caller: a time-dependent self-term (the banded /
    # slow-rotation response) would make the upsampled lnL wrong in a way no
    # downstream check would catch.
    rho_col = rho_sq[..., :1]
    # Compare only where both sides are finite.  A NaN self-term is NORMAL: the
    # defensive proposal component deliberately draws physically-extreme points
    # where the likelihood is NaN, and the historical path just returns NaN for
    # that row and lets the sampler move on.  A bare `==` makes `nan != nan` trip
    # this tripwire and abort the whole ILE process, blaming a rotating-response
    # path that is not even in use.
    _cmp = xpy.isfinite(rho_sq) & xpy.isfinite(xpy.broadcast_to(rho_col, rho_sq.shape))
    if not bool(xpy.all(xpy.where(_cmp, rho_sq == rho_col, True))):
        raise NotImplementedError(
            "band-limited time marginalization requires a time-independent rho_sq; "
            "the supplied self-term varies with time (banded / rotating-response path)")

    _term = (lambda k: xpy.abs(k)) if phase_marginalization else (lambda k: k.real)
    if lnL_coarse is None:
        lnL_coarse = loglikelihood(_term(kappa), rho_sq)

    sigma, jmax, measurable = peak_width_from_lnL(lnL_coarse, deltaT, xpy=xpy)

    # Classify the rows.  The edge guard must apply only to rows that HAVE a
    # resolvable peak: a row whose lnL(t) is constant -- an extrinsic sample in an
    # antenna null, where kappa is numerically zero -- has an argmax of 0 by
    # convention and would otherwise be reported as wrap-exposed.  That is
    # harmless numerically (Simpson is exact on a constant) but it makes the
    # diagnostic lie: measured on a random-sky batch of 4000, it reported 810
    # "wrap-exposed" rows, which in a production log reads as a mis-centred
    # window rather than as 810 rows with no signal in them.
    guard = max(1, int(npts * EDGE_GUARD_FRACTION))
    has_peak = measurable & xpy.isfinite(sigma)
    flat = measurable & (~xpy.isfinite(sigma))
    exposed = has_peak & ((jmax < guard) | (jmax > npts - 1 - guard))
    # Counted unconditionally, NOT `& ~exposed`: an all -inf row also has an
    # argmax of 0, so a conditional counter would hide it behind the edge guard.
    unmeasurable = ~measurable

    factors = xpy.maximum(required_upsample_factors(sigma, deltaT, xpy=xpy), 1)
    # A row is REFINED only if it has a trustworthy peak AND the derivation
    # actually asks for a finer grid.  Everything else -- wrap-exposed,
    # unmeasurable, or simply already resolved -- gets the historical Simpson
    # value.  That is the whole rule: the QUADRATURE changes for under-resolved
    # rows and for no others.  (The log-sum-exp offset changes for every row --
    # see last_report() -- so this is a statement about the rule, not a promise
    # that unrefined rows come back bit-identical.)
    # The alternative, letting an unrefined row fall through to
    # a coarse TRAPEZOID, is numerically a non-event but silently changes the
    # rule for rows this option was never meant to touch, and costs the property
    # a reviewer can actually check.  (Trapezoid is in fact slightly the better
    # rule on a resolved integrand -- 5e-6 nats against an analytic truth, versus
    # Simpson's 5e-6 the other way -- so this trades nothing measurable for an
    # auditable claim.)
    refined = (~(exposed | unmeasurable)) & (factors > 1)

    out = _log_simps_rows(lnL_coarse, deltaT, simps, xpy=xpy)

    hist = {}
    n_refine_total = 0
    sigma_seen = np.inf
    for f in xpy.unique(xpy.where(refined, factors, 1)):
        f = int(f)
        if f == 1:
            continue
        sel = refined & (factors == f)
        n_sel = int(xpy.sum(sel))
        if not n_sel:
            continue
        idx = xpy.where(sel)[0]
        vals, f_used, n_ref, s_min = _integrate_group(
            kappa[idx], rho_col[idx], npts, deltaT, f, loglikelihood, _term, xpy=xpy)
        out[idx] = vals
        hist[int(f_used)] = hist.get(int(f_used), 0) + n_sel
        n_refine_total += n_ref
        sigma_seen = min(sigma_seen, s_min)

    _LAST_REPORT.clear()
    _LAST_REPORT.update(
        upsample_factor=max(hist) if hist else 1,
        factor_histogram=dict(hist),
        n_refinements=n_refine_total,
        sigma_t_min=sigma_seen,
        n_rows=n_rows,
        n_wrap_exposed_rows=int(xpy.sum(exposed)),
        n_unmeasurable_rows=int(xpy.sum(unmeasurable)),
        n_flat_rows=int(xpy.sum(flat)),
        n_refined_rows=int(xpy.sum(refined)),
    )
    return out


def _integrate_group(kappa_rows, rho_col_rows, npts, deltaT, factor,
                     loglikelihood, _term, xpy=np):
    """Refine and integrate one group of rows that share a derived factor.

    Returns ``(values, factor_used, n_refinements, sigma_dense_min)``.
    """
    n_rows = kappa_rows.shape[0]
    n_refine = 0
    while True:
        if factor > UPSAMPLE_FACTOR_MAX:
            raise RuntimeError(
                "band-limited time marginalization needs an upsampling factor above "
                "the ceiling UPSAMPLE_FACTOR_MAX=%d (deltaT=%.3e s).  This is far "
                "beyond what the band limit can justify: suspect a pathological "
                "lnL(t), not an under-resolved one." % (UPSAMPLE_FACTOR_MAX, deltaT))

        dx_dense = deltaT / factor
        # Chunk the extrinsic axis so one dense temporary stays inside the
        # working-set budget.  Rows are independent; this cannot change results.
        per_row = npts * factor * 16 * 3
        chunk = max(1, min(n_rows, int(_DENSE_CHUNK_BYTES // max(per_row, 1))))

        pieces = []
        sigma_dense_min = np.inf
        for start in range(0, n_rows, chunk):
            k_up = bandlimited_upsample(kappa_rows[start:start + chunk], factor, xpy=xpy)
            rho_up = xpy.broadcast_to(rho_col_rows[start:start + chunk], k_up.shape)
            lnL_up = loglikelihood(_term(k_up), rho_up)
            s_d, _, meas = peak_width_from_lnL(lnL_up, dx_dense, xpy=xpy)
            s_d = xpy.where(meas, s_d, np.inf)
            sigma_dense_min = min(sigma_dense_min, float(xpy.min(s_d)))
            pieces.append(_log_trapz_over_window(lnL_up, dx_dense, npts, factor, xpy=xpy))

        # The assertion that turns the derivation into a guarantee: the width
        # remeasured on the grid we actually integrated on must still satisfy the
        # criterion.  A coarse-grid estimate can be optimistic when the peak is
        # strongly non-Gaussian; this catches that and pays for another doubling
        # instead of reporting a number it cannot defend.
        if (not np.isfinite(sigma_dense_min)) or dx_dense <= sigma_dense_min / UPSAMPLE_SAFETY:
            return (xpy.concatenate(pieces) if len(pieces) > 1 else pieces[0],
                    factor, n_refine, sigma_dense_min)

        factor *= 2
        n_refine += 1
