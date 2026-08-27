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

Measured on the reference 35+30 Msun SEOBNRv4 H1L1V1 injection at rho=40
(sigma_t = 61.2 us, a property of the signal and so srate-independent), rigidly
scanning the grid phase over 2*deltaT moves the reported lnL by

    srate 4096: 1.649 nats     8192: 0.385 nats     16384: 0.0095 nats

i.e. the answer depends on where the sample grid happens to fall relative to the
peak, by over a nat at the production sample rate.

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

Measured against a converged dense reference built by re-gathering the rholms at
shifted window offsets (the expensive, independent construction), at srate 4096,
rho=40, on the same injection:

    band-limited upsampling:  -0.007 nats
    Simpson at deltaT      :  +0.745 nats

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
``2e-34``.  Even ``h = sigma`` would give ``5e-9``.  The quadrature on the dense
grid is TRAPEZOID, not Simpson: for a peak that decays to nothing inside the
window every Euler-Maclaurin boundary term vanishes, so the trapezoidal rule is
spectrally accurate there, while Simpson would reintroduce the ``2h`` alias that
is the original defect.

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
    "bandlimited_upsample",
    "peak_width_from_lnL",
    "required_upsample_factor",
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
#: more than ``~2 rho`` here.  Exceeding this raises rather than silently
#: truncating the resolution.
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
#: 1/8 of the window is 77 samples at the production npts=614, i.e. the region
#: where the deviation stays at or below ~5e-3 nats.  In a well-posed run nothing
#: comes close: the grid is centred on the trigger's geocentre time, so the peak
#: sits within the trigger timing uncertainty (a few ms, tens of samples) of the
#: CENTRE, not of an edge.  A row that does violate this has a mis-centred
#: window, which truncates its integral under EITHER rule; it is handed back the
#: historical Simpson value and counted in ``last_report()``, so the option can
#: never make a row worse than the status quo it replaces.  (The route to
#: supporting such rows properly is to widen the GATHER so the wrap sits outside
#: the integration domain -- deliberately not done here, since it touches the
#: GPU kernel and the buffer-margin assumptions.)
EDGE_GUARD_FRACTION = 0.125

#: Working-set budget for one dense temporary, in bytes.  Purely an internal
#: memory-chunking parameter: it changes how many extrinsic rows are processed at
#: a time and cannot change the answer.
_DENSE_CHUNK_BYTES = 128 * 1024 * 1024

_LAST_REPORT = {}
_SIMPSON_WEIGHT_CACHE = {}


def last_report():
    """Diagnostics from the most recent :func:`time_marginalize_bandlimited` call.

    Keys: ``upsample_factor``, ``n_refinements``, ``sigma_t_min``,
    ``dense_npts``, ``n_rows``, ``n_wrap_exposed_rows``.  The last counts rows
    whose integrand peaks inside ``EDGE_GUARD_FRACTION`` of a window edge; those
    rows were handed the historical Simpson value instead of a refined one.  A
    nonzero count is a statement about the WINDOW being mis-centred for those
    samples, not about this quadrature.
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

    A single Nyquist bin, when ``n`` is even, is split evenly between ``+fNyq``
    and ``-fNyq``; the alternative (dumping it entirely into one) is the standard
    way to make a real-input upsample come out complex.  For the rholm data this
    bin is empty anyway -- ``fmax <= fNyq`` -- so the choice is a formality kept
    for correctness on synthetic inputs.
    """
    factor = int(factor)
    if factor == 1:
        return x
    n = x.shape[-1]
    lead = x.shape[:-1]
    X = xpy.fft.fft(x, axis=-1)
    Xup = xpy.zeros(lead + (n * factor,), dtype=xpy.asarray(X).dtype)
    h = n // 2
    Xup[..., :h] = X[..., :h]
    Xup[..., -(n - h):] = X[..., h:]
    if n % 2 == 0:
        Xup[..., h] = 0.5 * X[..., h]
        Xup[..., -h] = 0.5 * X[..., h]
    return xpy.fft.ifft(Xup, axis=-1) * factor


def peak_width_from_lnL(lnL_t, dx, xpy=np):
    """Per-row Gaussian width ``sigma_t`` of ``exp(lnL_t)``, from its peak curvature.

    Uses the three-point second difference of ``lnL`` (not of ``exp lnL``) about
    the peak sample.  For a Gaussian ``lnL`` this returns ``sigma`` EXACTLY at any
    spacing and any peak-vs-grid phase, because the second difference of a
    parabola is its second derivative; that is what lets an under-resolved peak
    still report its own width honestly.  Rows with non-negative curvature
    (flat, monotone, or peaked at the window edge) return ``inf``: no upsampling
    is warranted or possible for them.

    Returns ``(sigma_t, argmax_index)``, both shape ``lnL_t.shape[:-1]``.
    """
    n = lnL_t.shape[-1]
    if n < 3:
        raise ValueError("need at least 3 time samples to measure a peak width")
    jmax = xpy.argmax(lnL_t, axis=-1)
    jc = xpy.clip(jmax, 1, n - 2)
    take = lambda j: xpy.take_along_axis(lnL_t, j[..., None], axis=-1)[..., 0]
    d2 = (take(jc - 1) - 2.0 * take(jc) + take(jc + 1)) / (dx * dx)
    # Guard: -inf entries (e.g. a distance-marginalization table edge) make d2
    # nan; treat those rows as unresolvable rather than letting nan propagate
    # into the factor derivation.
    bad = ~xpy.isfinite(d2)
    d2 = xpy.where(bad, 0.0, d2)
    sigma = xpy.where(d2 < 0, 1.0 / xpy.sqrt(xpy.where(d2 < 0, -d2, 1.0)), np.inf)
    return sigma, jmax


def required_upsample_factor(lnL_t, dx, xpy=np, sigma=None):
    """Smallest power-of-two factor with ``dx/factor <= sigma_t/UPSAMPLE_SAFETY``.

    Derived from the narrowest peak present, so one factor serves the whole
    block.  ``sigma`` may be supplied to reuse an already-measured width array
    (or to mask rows out of the derivation by setting theirs to ``inf``).
    Returns ``(factor, sigma_t_min)``.
    """
    if sigma is None:
        sigma, _ = peak_width_from_lnL(lnL_t, dx, xpy=xpy)
    sigma_min = float(xpy.min(sigma))
    if not np.isfinite(sigma_min) or sigma_min <= 0:
        return 1, sigma_min
    need = UPSAMPLE_SAFETY * dx / sigma_min
    if need <= 1.0:
        return 1, sigma_min
    factor = int(2 ** int(np.ceil(np.log2(need))))
    return factor, sigma_min


def _simpson_weights(n, dx, xpy=np):
    """Weight vector w with ``sum(w*f) == scipy.integrate.simpson(f, dx=dx)``.

    Simpson's rule is linear in the samples, so its weights are exactly its
    action on the identity.  Building them explicitly lets the wrap-exposed
    fallback below reproduce the historical value with a PER-ROW log-sum-exp
    offset instead of the shared global one, which is what keeps a row far below
    the block maximum from underflowing to ``log(0)``.
    """
    key = (int(n), float(dx))
    w = _SIMPSON_WEIGHT_CACHE.get(key)
    if w is None:
        from scipy import integrate
        simpson = getattr(integrate, 'simpson', None) or integrate.simps
        w = simpson(np.eye(int(n), dtype=np.float64), dx=float(dx), axis=-1)
        _SIMPSON_WEIGHT_CACHE.clear()   # one shape per run; do not grow unbounded
        _SIMPSON_WEIGHT_CACHE[key] = w
    return xpy.asarray(w)


def _log_simps_rows(lnL_t, dx, xpy=np):
    """``log \\int exp(lnL) dt`` by the HISTORICAL Simpson rule, per row."""
    w = _simpson_weights(lnL_t.shape[-1], dx, xpy=xpy)
    off = xpy.max(lnL_t, axis=-1, keepdims=True)
    return off[..., 0] + xpy.log(xpy.sum(xpy.exp(lnL_t - off) * w, axis=-1))


def _apply_exposed_fallback(out, exposed, n_exposed, lnL_coarse, deltaT, xpy=np):
    """Overwrite wrap-exposed rows with the historical Simpson value.

    Applied on BOTH return paths, including the one where the derived factor is 1
    and no interpolation happened.  It has to be: with every row exposed the
    factor derivation sees no usable width and returns 1, and silently handing
    those rows a coarse TRAPEZOID instead would still be a change of rule for
    them -- measurably worse than Simpson on some under-resolved peaks -- which
    would break the property that enabling this option can never make a row worse
    than the status quo.
    """
    if not n_exposed:
        return out
    return xpy.where(exposed, _log_simps_rows(lnL_coarse, deltaT, xpy=xpy), out)


def _log_trapz_over_window(lnL_dense, dx_dense, npts_coarse, factor, xpy=np):
    """``log \\int exp(lnL) dt`` by trapezoid over the ORIGINAL window span.

    The dense grid returned by the FFT upsample is periodic on
    ``[t_0, t_0 + npts*deltaT)``, i.e. it carries ``factor-1`` samples PAST the
    last coarse sample.  Those lie across the periodic wrap and are dropped, so
    the integration domain is exactly ``[t_0, t_{npts-1}]`` -- byte-identical to
    the domain Simpson used.  Changing the domain would have been a second,
    confounded change.

    The log-sum-exp offset is PER ROW and taken on the DENSE grid, not the single
    global coarse maximum the Simpson path uses.  It has to be: the whole point of
    refining the grid is that the true peak sits between coarse samples, so the
    dense maximum can exceed the coarse one -- by thousands of nats for a sharp
    peak -- and offsetting by the coarse maximum overflows exp() precisely in the
    regime this quadrature exists to serve.  (A per-row offset also avoids the
    underflow-to-``log(0) = -inf`` the shared global offset gives rows far below
    the block maximum.)  The result is offset-invariant, so this is a numerical
    choice and not a change of estimator.
    """
    last = (npts_coarse - 1) * factor
    v = lnL_dense[..., :last + 1]
    w = xpy.full(v.shape[-1], dx_dense, dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    off = xpy.max(v, axis=-1, keepdims=True)
    return off[..., 0] + xpy.log(xpy.sum(xpy.exp(v - off) * w, axis=-1))


def time_marginalize_bandlimited(kappa, rho_sq, deltaT, loglikelihood,
                                 phase_marginalization=False, xpy=np):
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

    Returns
    -------
    lnL : (n_extrinsic,) float
    """
    kappa = xpy.asarray(kappa)
    rho_sq = xpy.asarray(rho_sq)
    npts = kappa.shape[-1]
    n_rows = kappa.shape[0]

    # rho_sq time-independence is the load-bearing precondition, so verify it
    # rather than trusting the caller: a time-dependent self-term (the banded /
    # slow-rotation response) would make the upsampled lnL wrong in a way no
    # downstream check would catch.
    rho_col = rho_sq[..., :1]
    if not bool(xpy.all(rho_sq == rho_col)):
        raise NotImplementedError(
            "band-limited time marginalization requires a time-independent rho_sq; "
            "the supplied self-term varies with time (banded / rotating-response path)")

    _term = (lambda k: xpy.abs(k)) if phase_marginalization else (lambda k: k.real)
    lnL_coarse = loglikelihood(_term(kappa), rho_sq)

    sigma, jmax = peak_width_from_lnL(lnL_coarse, float(deltaT), xpy=xpy)

    # Wrap-exposed rows: peak too close to the window edge for the periodic
    # interpolant to be trusted there (see EDGE_GUARD_FRACTION).  They are
    # excluded from the factor derivation too -- otherwise a single mis-centred
    # row could inflate the refinement everyone else pays for -- and are handed
    # the historical Simpson value at the end.
    guard = max(1, int(npts * EDGE_GUARD_FRACTION))
    exposed = (jmax < guard) | (jmax > npts - 1 - guard)
    n_exposed = int(xpy.sum(exposed))

    factor, sigma_min = required_upsample_factor(
        lnL_coarse, float(deltaT), xpy=xpy,
        sigma=xpy.where(exposed, np.inf, sigma))

    if factor == 1:
        # Nothing to resolve: the peak (if any) is already wide compared with
        # deltaT, so the coarse samples already meet the criterion.  Integrate on
        # the coarse grid with the same trapezoid rule, so the two branches of
        # this function agree with each other rather than one of them silently
        # reverting to Simpson.
        out = _log_trapz_over_window(lnL_coarse, float(deltaT), npts, 1, xpy=xpy)
        out = _apply_exposed_fallback(out, exposed, n_exposed, lnL_coarse,
                                      float(deltaT), xpy=xpy)
        _LAST_REPORT.update(upsample_factor=1, n_refinements=0,
                            sigma_t_min=sigma_min, dense_npts=npts,
                            n_rows=n_rows, n_wrap_exposed_rows=n_exposed)
        return out

    n_refine = 0
    while True:
        if factor > UPSAMPLE_FACTOR_MAX:
            raise RuntimeError(
                "band-limited time marginalization needs an upsampling factor above "
                "the ceiling UPSAMPLE_FACTOR_MAX=%d (narrowest measured sigma_t=%.3e s, "
                "deltaT=%.3e s).  This is far beyond what the band limit can justify: "
                "suspect a pathological lnL(t), not an under-resolved one."
                % (UPSAMPLE_FACTOR_MAX, sigma_min, float(deltaT)))

        dx_dense = float(deltaT) / factor
        # Chunk the extrinsic axis so one dense temporary stays inside the
        # working-set budget.  Rows are independent; this cannot change results.
        per_row = npts * factor * 16 * 3
        chunk = max(1, min(n_rows, int(_DENSE_CHUNK_BYTES // max(per_row, 1))))

        pieces = []
        sigma_dense_min = np.inf
        for start in range(0, n_rows, chunk):
            k_up = bandlimited_upsample(kappa[start:start + chunk], factor, xpy=xpy)
            rho_up = xpy.broadcast_to(rho_col[start:start + chunk], k_up.shape)
            lnL_up = loglikelihood(_term(k_up), rho_up)
            s_d, _ = peak_width_from_lnL(lnL_up, dx_dense, xpy=xpy)
            sigma_dense_min = min(sigma_dense_min, float(xpy.min(s_d)))
            pieces.append(_log_trapz_over_window(lnL_up, dx_dense, npts, factor,
                                                 xpy=xpy))

        # The assertion that turns the derivation into a guarantee: the width
        # remeasured on the grid we actually integrated on must still satisfy the
        # criterion.  A coarse-grid width estimate can be optimistic when the
        # peak is strongly non-Gaussian; this catches that and pays for another
        # doubling instead of reporting a number it cannot defend.
        if (not np.isfinite(sigma_dense_min)) or dx_dense <= sigma_dense_min / UPSAMPLE_SAFETY:
            out = _apply_exposed_fallback(xpy.concatenate(pieces), exposed,
                                          n_exposed, lnL_coarse, float(deltaT),
                                          xpy=xpy)
            _LAST_REPORT.update(upsample_factor=factor, n_refinements=n_refine,
                                sigma_t_min=sigma_dense_min,
                                dense_npts=npts * factor, n_rows=n_rows,
                                n_wrap_exposed_rows=n_exposed)
            return out

        factor *= 2
        n_refine += 1
