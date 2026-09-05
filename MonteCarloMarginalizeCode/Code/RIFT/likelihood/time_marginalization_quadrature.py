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
``fmax <= fNyq = 1/(2 deltaT)``.  So ``kappa(t)`` is band-limited below Nyquist.
The template self-term ``rho_sq`` is time-independent on this path.  Therefore
``lnL(t) = f(kappa(t), rho_sq)`` is recoverable on an arbitrarily fine grid from
samples covering a full underlying period.

The samples in hand are only a gathered INTEGRATION-WINDOW SLICE, however.  A
zero-padded FFT of that slice alone identifies its generally unlike endpoints.
Decay of ``exp(lnL)`` does NOT make that safe: the FFT acts on ``kappa``, whose
endpoint mismatch can remain large where the likelihood is negligible, and the
resulting Gibbs overshoot is global.  A centred adversarial row with both outer
eighths more than 49 nats below its peak still moved by +140.9 nats when the raw
slice was periodized.

The implementation instead doubles the finite row by literal even reflection,
``[kappa forward, kappa backward]``, before FFT interpolation, then discards the
backward half.  Both joins are value-continuous, the supplied samples are
unchanged, and no unavailable samples outside the deliberately narrow
integration domain are invented.  The same adversarial row is accurate to
2.4e-4 nats.  Boundary proximity and coarse tail height are reported or tested,
but neither selects a lower-accuracy Simpson fallback: such a discontinuous
switch silently changes likelihood quality, while an exception can be mapped by
the calling pipeline to waveform failure and silently excise configurations.

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
defect's magnitude rather than a caricature of it.  On windows cut from a longer
band-limited signal, the reflected reconstruction measured 2e-8 to 2.5e-6 nats
error over the tested sharpness range; the raw periodic-slice reconstruction
could ring at the artificial wrap.

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
the true ``kappa(t)`` only when ``time_interp='nearest'`` -- which since 2026-09-02
is no longer the ILE driver's default (issue #233; the driver now defaults to
``time_interp_choice.TIME_INTERP_DEFAULT``), so the paragraph below is now the
ORDINARY case rather than the exceptional one -- where
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
for ``time_interp='nearest'``, which is NOT the driver default any more: pass
``--interpolate-time nearest`` alongside ``--time-marginalization-quadrature
bandlimited`` to reproduce them.  Re-measuring this pairing under the new default
is an OPEN item, not a settled result.

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

import os
import warnings

import numpy as np

__all__ = [
    "TIME_QUADRATURE_CHOICES",
    "UPSAMPLE_SAFETY",
    "UPSAMPLE_FACTOR_MAX",
    "EDGE_GUARD_FRACTION",
    "bandlimited_upsample",
    "reflected_bandlimited_upsample",
    "peak_width_from_lnL",
    "required_upsample_factors",
    "validate_time_quadrature",
    "time_quadrature_pipeline_prereqs",
    "ILE_TIME_QUADRATURE_FLAG",
    "refuse_unless_time_quadrature_emitted",
    "refuse_unhonourable_time_quadrature",
    "find_time_quadrature_in_ile_args",
    "draw_piecewise_linear_log_posterior",
    "time_marginalize_bandlimited",
    "last_report",
]

TIME_QUADRATURE_CHOICES = ("simpson", "bandlimited", "peak-local")

#: 'peak-local' lives in RIFT.likelihood.time_marginalization_peak_local and
#: reuses this module's helpers wholesale (width estimator, derived factor, row
#: classification, edge guard, Simpson hand-over).  It is named here rather than
#: there so that validate_time_quadrature stays the single place a quadrature name
#: is checked; the import runs the other way, so there is no cycle.

#: ``h_dense <= sigma_t / UPSAMPLE_SAFETY``.  See the module docstring: at this
#: value the trapezoidal rule's Poisson-summation error on a Gaussian peak is
#: 2e-34, so this is a hard-coded constant and not an accuracy/cost trade.
UPSAMPLE_SAFETY = 2.0

#: Fail-closed ceiling.  The band limit bounds the useful factor: with
#: ``sigma_t >= deltaT / (pi rho)`` the derivation cannot legitimately ask for
#: more than ``~2 rho``.  Exceeding this raises rather than silently truncating
#: the resolution.
UPSAMPLE_FACTOR_MAX = 4096

#: Fraction of the window at EACH end used only to REPORT a peak close to a
#: truncated integration boundary.  It must not select a different quadrature:
#: crossing an arbitrary threshold cannot silently move an under-resolved row
#: back to Simpson, and raising on such a row can be interpreted upstream as a
#: waveform failure and silently excise that configuration.  The reflected
#: reconstruction below has no value discontinuity at either boundary, so the
#: old periodic-wrap exclusion is no longer needed for numerical safety.
#:
#: Measured on a window cut from a longer band-limited signal (peaked kernel plus
#: a 12%-amplitude coloured background, so the two ends genuinely disagree),
#: sigma_t/deltaT = 0.042, error against the analytic continuous truth, in nats:
#:
#:   peak distance from edge   307     100      30      8       2       0
#:   band-limited              5e-6   4.6e-3  5.2e-2  5.6e-2  -3.3    +88.8
#:   Simpson (for scale)      -29.2   -29.9   -29.3   -29.4   -29.7   -29.9
#:
#: That table is the REJECTED raw-slice periodic reconstruction and records why
#: boundary proximity must remain visible: +88.8 is in the dangerous direction,
#: where a spurious row can dominate importance weights.  It does not justify an
#: algorithm switch.  With reflection, peaks immediately on either side of the
#: old one-eighth line (samples 75.3, 76.3, 77.3 at npts=614) agree with analytic
#: truth at order 1e-6 nats.  Switching them to under-resolved Simpson because
#: they crossed that line would itself create the quality regression.
#:
#: In a well-posed run nothing comes close: the grid is centred on the trigger's
#: geocentre time, so the peak sits within the trigger timing uncertainty (a few
#: ms, tens of samples) of the CENTRE, not of an edge.  Rows that do violate it
#: are still reconstructed and integrated over the caller's unchanged, possibly
#: truncated domain; the count in ``last_report()`` makes that separate physical
#: window-containment issue auditable without changing the likelihood rule.
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

#: Working-set budget for one dense temporary, in bytes.  An internal
#: memory-chunking parameter: it changes how many extrinsic rows are processed at
#: a time.  Rows are independent, so it cannot change the answer BEYOND
#: FLOATING-POINT REASSOCIATION -- the batch shape reaches numpy's FFT and its
#: pairwise summation, so a differently-chunked run can differ in the last
#: bit or two.  Measured on the companion peak-local implementation, which
#: inherited this same wording and then failed a bit-identity test that was
#: written to it: 0, 0 and 2 ULPs.  "Cannot change the answer" was too strong;
#: the honest statement is that it cannot change the answer at any scale that
#: is not floating-point noise.
_DENSE_CHUNK_BYTES = 128 * 1024 * 1024


def _cpu_fft_workers():
    """Bounded CPU FFT parallelism, respecting scheduler CPU affinity.

    The reflected transforms have awkward production lengths (for example
    ``2*307``), and dominate the AV band-limited path.  SciPy's pocketfft can
    parallelize the independent row transforms, while NumPy's public FFT API
    cannot.  Never request more CPUs than the process affinity mask exposes;
    ``RIFT_TIME_FFT_WORKERS`` can lower the cap or raise the default cap of four.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    requested = int(os.environ.get("RIFT_TIME_FFT_WORKERS", "4"))
    return max(1, min(requested, available))


def _fft_rows(x, inverse=False, n=None, xpy=np):
    if xpy is np:
        from scipy import fft as scipy_fft
        fn = scipy_fft.ifft if inverse else scipy_fft.fft
        return fn(x, n=n, axis=-1, workers=_cpu_fft_workers())
    fn = xpy.fft.ifft if inverse else xpy.fft.fft
    return fn(x, n=n, axis=-1)


class _RetainedFFTUnsupported(RuntimeError):
    """The optional retained-grid transform cannot honour this input."""


def _retained_fft_backend(xpy):
    """Return the supported backend name without moving an array to the host."""
    if xpy is np:
        return "numpy"
    if getattr(xpy, "__name__", None) == "cupy":
        return "cupy"
    raise _RetainedFFTUnsupported(
        "retained-grid FFT supports only the numpy and cupy backends")


def _retained_fft_plan(period, factor, dtype, xpy=np):
    """Build stable Bluestein chirps for the forward half of a reflected row.

    The plan evaluates the same Fourier polynomial as zero padding to
    ``period * factor``, but only at the ``(period/2 - 1)*factor + 1`` samples
    consumed by the finite-window integral.  Integer modular phases avoid the
    unit-circle drift of forming a high power of one approximate complex root.
    Plans live only for one marginalization call, so large GPU chirps cannot
    become an unbounded process-wide cache.
    """
    _retained_fft_backend(xpy)
    period = int(period)
    factor = int(factor)
    dtype = np.dtype(dtype)
    if period < 4 or period % 2:
        raise _RetainedFFTUnsupported(
            "reflected FFT period must be even and at least four")
    if factor <= 1 or factor & (factor - 1):
        raise _RetainedFFTUnsupported(
            "retained-grid FFT requires a power-of-two factor above one")
    if dtype != np.dtype(np.complex128):
        raise _RetainedFFTUnsupported(
            "retained-grid FFT is certified only for complex128 spectra, got %s"
            % dtype)

    # The Nyquist coefficient is represented at both signed endpoints, hence
    # period+1 input coefficients.  Linear Bluestein convolution needs the sum
    # of input and output lengths minus one.  next_fast_len is a host-side
    # integer calculation only; all arrays and FFTs stay on xpy's device.
    n_coeff = period + 1
    n_out = (period // 2 - 1) * factor + 1
    n_chirp = max(n_coeff, n_out)
    if n_chirp > 3037000499 or period * factor > np.iinfo(np.int64).max // 2:
        raise _RetainedFFTUnsupported(
            "retained-grid dimensions exceed the exact int64 chirp-phase range")
    from scipy.fft import next_fast_len
    n_fft = int(next_fast_len(n_coeff + n_out - 1))

    k = xpy.arange(n_chirp, dtype=np.int64)
    denominator = period * factor
    # exp(+i*pi*k**2/denominator), reduced exactly modulo 2*denominator
    # before conversion to float.  The largest supported production grid is
    # safely within int64 (roughly 1e14 at npts=2457, factor=4096).
    phase_index = (k * k) % (2 * denominator)
    wk2 = xpy.exp((1j * np.pi / denominator) * phase_index)
    wk2 = xpy.asarray(wk2, dtype=np.complex128)
    kernel = 1.0 / xpy.concatenate(
        (wk2[n_coeff - 1:0:-1], wk2[:n_out]))
    kernel_fft = _fft_rows(kernel, n=n_fft, xpy=xpy)

    j = xpy.arange(n_out, dtype=np.int64)
    shift_index = j % (2 * factor)
    signed_frequency_shift = xpy.exp(
        (-1j * np.pi / factor) * shift_index)
    post = (wk2[:n_out] * signed_frequency_shift) / float(period)
    return {
        "input_chirp": wk2[:n_coeff],
        "kernel_fft": kernel_fft,
        "post_chirp": post,
        "n_fft": n_fft,
        "n_out": n_out,
        "period": period,
        "factor": factor,
    }


def _reflected_bandlimited_upsample_retained(x, factor, plan_cache=None,
                                               xpy=np):
    """Evaluate exactly the retained forward grid of the reflected interpolant.

    This is a pruned *evaluation* of :func:`reflected_bandlimited_upsample`, not
    a different interpolant.  It preserves the literal ``[x, flip(x)]``
    boundary condition and the half-weight split of the even-period Nyquist bin.
    """
    x = xpy.asarray(x)
    factor = int(factor)
    if factor == 1:
        return x
    n = int(x.shape[-1])
    period = 2 * n
    reflected = xpy.concatenate((x, xpy.flip(x, axis=-1)), axis=-1)
    spectrum = _fft_rows(reflected, xpy=xpy)
    dtype = np.dtype(spectrum.dtype)
    cache_key = (period, factor, dtype.str)
    if plan_cache is None:
        plan_cache = {}
    plan = plan_cache.get(cache_key)
    if plan is None:
        plan = _retained_fft_plan(period, factor, dtype, xpy=xpy)
        plan_cache[cache_key] = plan

    half = period // 2
    # Consecutive signed-frequency coefficients k=-half,...,+half.  Splitting
    # the Nyquist bin across the two endpoints is exactly what the full padded
    # inverse FFT does in bandlimited_upsample for an even-length row.
    coeff = xpy.empty(spectrum.shape[:-1] + (period + 1,), dtype=spectrum.dtype)
    coeff[..., 0] = 0.5 * spectrum[..., half]
    coeff[..., 1:half] = spectrum[..., half + 1:]
    coeff[..., half] = spectrum[..., 0]
    coeff[..., half + 1:period] = spectrum[..., 1:half]
    coeff[..., period] = 0.5 * spectrum[..., half]

    transformed = _fft_rows(
        coeff * plan["input_chirp"], n=plan["n_fft"], xpy=xpy)
    transformed *= plan["kernel_fft"]
    convolved = _fft_rows(transformed, inverse=True, xpy=xpy)
    retained = convolved[..., period:period + plan["n_out"]]
    retained *= plan["post_chirp"]
    if retained.shape[-1] != (n - 1) * factor + 1:
        raise RuntimeError("retained-grid FFT returned an inconsistent shape")
    return retained


def _record_transform(report, key, n_rows, period, factor, plan=None):
    report[key + "_batches"] += 1
    report[key + "_rows"] += int(n_rows)
    report["max_reflected_period"] = max(report["max_reflected_period"],
                                         int(period))
    report["max_dense_factor"] = max(report["max_dense_factor"], int(factor))
    report["max_reference_full_fft_length"] = max(
        report["max_reference_full_fft_length"], int(period) * int(factor))
    if plan is not None:
        report["max_retained_fft_length"] = max(
            report["max_retained_fft_length"], int(plan["n_fft"]))
        report["max_retained_grid_length"] = max(
            report["max_retained_grid_length"], int(plan["n_out"]))


def _new_transform_report():
    return dict(
        retained_fft_batches=0,
        retained_fft_rows=0,
        full_fft_selected_batches=0,
        full_fft_selected_rows=0,
        full_fft_selected_reasons={},
        full_fft_fallback_batches=0,
        full_fft_fallback_rows=0,
        full_fft_fallback_reasons={},
        warned_fallback_reasons=set(),
        max_reflected_period=0,
        max_dense_factor=1,
        max_reference_full_fft_length=0,
        max_retained_fft_length=0,
        max_retained_grid_length=0,
    )


def _reflected_upsample_for_integration(x, factor, plan_cache,
                                         transform_report, xpy=np):
    """Use the retained-grid transform, visibly falling back to the reference.

    An optimization failure is not a waveform or likelihood failure.  Any
    unsupported input or transform exception therefore retries the established
    full-padding implementation and records why.  The likelihood callback is
    deliberately outside this function, so its failures are never mislabeled or
    swallowed as FFT fallbacks.
    """
    period = 2 * int(x.shape[-1])
    # Pocketfft measurements across all production npts found the retained
    # convolution neutral-to-slower at factors 2 and 4; that small dense grid is
    # not the bottleneck.  Preserve the cheaper reference algorithm there.  On
    # CuPy the retained path won at every tested factor 2--64.
    if xpy is np and int(factor) in (2, 4):
        reason = "numpy factor %d is below the measured retained-FFT crossover" % factor
        reasons = transform_report["full_fft_selected_reasons"]
        reasons[reason] = reasons.get(reason, 0) + int(x.shape[0])
        _record_transform(transform_report, "full_fft_selected", x.shape[0],
                          period, factor)
        return reflected_bandlimited_upsample(x, factor, xpy=xpy)
    try:
        out = _reflected_bandlimited_upsample_retained(
            x, factor, plan_cache=plan_cache, xpy=xpy)
        plan = next((value for (plan_period, plan_factor, _), value
                     in plan_cache.items()
                     if plan_period == period and plan_factor == int(factor)), None)
        _record_transform(transform_report, "retained_fft", x.shape[0],
                          period, factor, plan)
        return out
    except Exception as exc:
        reason = "%s: %s" % (type(exc).__name__, str(exc))
        reasons = transform_report["full_fft_fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + int(x.shape[0])
        _record_transform(transform_report, "full_fft_fallback", x.shape[0],
                          period, factor)
        if reason not in transform_report["warned_fallback_reasons"]:
            # Warning filters are allowed to promote RuntimeWarning to an
            # exception.  Diagnostics must not turn a successful reference-path
            # retry into a dropped likelihood point, so contain that policy here.
            try:
                warnings.warn(
                    "retained-grid band-limited FFT unavailable ({}); using the "
                    "established full-padding sinc reconstruction for these rows"
                    .format(reason), RuntimeWarning, stacklevel=2)
            except Exception:
                pass
            transform_report["warned_fallback_reasons"].add(reason)
        return reflected_bandlimited_upsample(x, factor, xpy=xpy)

_LAST_REPORT = {}


def last_report():
    """Diagnostics from the most recent :func:`time_marginalize_bandlimited` call.

    Keys: ``upsample_factor`` (the largest used), ``factor_histogram``
    (factor -> row count, over the rows that were refined), ``n_refinements``,
    ``sigma_t_min``, ``n_rows``, ``n_wrap_exposed_rows``, ``n_unmeasurable_rows``,
    ``n_flat_rows``, ``n_refined_rows``, and retained-transform provenance.

    ``bandlimited_fft_strategy`` says whether the production-only optimization
    used the retained-grid ZoomFFT, intentionally selected the established full
    transform below a measured CPU crossover, fell back to it after a transform
    decline, used a mixture, or needed no dense transform.  The corresponding
    ``*_batches`` and ``*_rows`` fields distinguish these cases; the reason maps
    make a cost selection or declined optimization auditable without converting
    either into a failed waveform point.  The reported reference, retained-grid,
    and convolution lengths expose the padding mismatch for performance records.

    The diagnostic row counts are deliberately kept apart because they mean
    different things:

    ``n_wrap_exposed_rows`` -- compatibility name for a resolvable peak sitting
    inside ``EDGE_GUARD_FRACTION`` of a window edge.  It is diagnostic only: the
    row is reconstructed by the same reflected rule as every other measurable
    under-resolved row.  The name records the old implementation; there is no
    periodic wrap in the current reconstruction.

    ``n_unmeasurable_rows`` -- ``lnL(t)`` non-finite around its maximum at every
    stencil half-width, so no width can be justified.  Given the historical value.

    ``n_flat_rows`` -- finite ``lnL(t)`` with no resolvable curvature: an
    extrinsic sample with no signal in it.  Nothing is wrong and nothing is paid.

    ``n_refined_rows`` is the count that matters for auditing a change: the
    QUADRATURE RULE changes for these rows and for no others.  Every other row --
    unmeasurable, flat, or already resolved -- is integrated by the caller's own
    Simpson rule over the same domain.

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


#: Flags in an assembled ILE argument string that the band-limited path REQUIRES,
#: and flags whose presence EXCLUDES it.  This is the pipeline-side mirror of the
#: ``_tq_prereqs`` block in ``bin/integrate_likelihood_extrinsic_batchmode``: the
#: driver refuses at first-job time, which costs a queue-slot cycle, so the
#: workflow builder refuses at DAG-BUILD time on the same conditions.  Kept here
#: rather than re-typed in the pipeline scripts for the reason the whole option is
#: validated through this module -- one list, one place to change it.
_PIPELINE_REQUIRED_ILE_FLAGS = (
    ('--time-marginalization',
     'without it ILE takes the non-time-marginalized branch, which has no time quadrature at all'),
    ('--vectorized',
     'without it ILE calls the SCALAR time-marginalized likelihood, which has no quadrature argument'),
    ('--gpu',
     'the band-limited path lives in the maintained NoLoop likelihood; --force-xpy runs it on numpy'),
)
_PIPELINE_EXCLUDING_ILE_FLAGS = (
    ('--rotation-slow',
     'time-DEPENDENT rho_sq: the band-limited argument does not hold'),
    ('--freqresponse',
     'separate likelihood, not audited for this'),
    ('--calibration-envelope-directory',
     'calibration marginalization: the reduction sums exp over realizations, untested here'),
)


ILE_TIME_QUADRATURE_FLAG = '--time-marginalization-quadrature'

def _ile_tokens(ile_args):
    """Tokenise an ILE argument string the way optparse will see it.

    Splits ``--flag=value`` (optparse accepts it, and a naive split does not) and
    strips the quotes an ini file leaves behind.  Returns ``(flags, pairs)`` where
    ``pairs`` is the token list with values still attached in order.
    """
    raw = str(ile_args).split()
    toks = []
    for t in raw:
        t = t.strip().strip('"').strip("'")
        if not t:
            continue
        if t.startswith('--') and '=' in t:
            k, v = t.split('=', 1)
            toks.append(k)
            toks.append(v)
        else:
            toks.append(t)
    return toks


def _matches(flag, token):
    """True if ``token`` is ``flag`` or a possible optparse abbreviation.

    optparse has no fixed minimum abbreviation length: in the current ILE parser
    ``--g`` uniquely selects ``--gpu`` and ``--vec`` selects ``--vectorized``.
    Ambiguous prefixes are rejected by ILE itself; treating them as a match here
    can only move that refusal to DAG-build time, while imposing an invented
    length floor falsely rejects legal configurations.
    """
    if token == flag:
        return True
    return (flag.startswith(token) and token.startswith('--')
            and len(token) > 2)


def find_time_quadrature_in_ile_args(ile_args):
    """Every value given to ``--time-marginalization-quadrature`` in ``ile_args``.

    Returns a list, in order, so the caller can tell "absent" from "present once"
    from "given twice with different values" -- optparse takes the LAST
    occurrence, so a duplicate silently decides the quadrature.
    """
    toks = _ile_tokens(ile_args)
    out = []
    for n, t in enumerate(toks):
        # optparse accepts unique long-option prefixes.  The exact
        # ``--time-marginalization`` flag wins as an exact match, but anything
        # through the following '-' is a unique prefix of the quadrature flag.
        # Treat those spellings exactly as ILE does or a hand-passed abbreviated
        # bandlimited request can be invisible to the prerequisite guard.
        is_quadrature = (
            t == ILE_TIME_QUADRATURE_FLAG
            or (t.startswith('--time-marginalization-')
                and ILE_TIME_QUADRATURE_FLAG.startswith(t))
        )
        if is_quadrature:
            out.append(toks[n + 1] if n + 1 < len(toks) else None)
    return out


def time_quadrature_pipeline_prereqs(time_quadrature, ile_args):
    """Missing/violated prerequisites for ``time_quadrature`` in an ILE argument string.

    ``ile_args`` is the assembled ILE command line the workflow is about to write
    (``args_ile.txt``).  Returns a list of human-readable reasons; empty means the
    configuration can honour the request.  ``'simpson'`` -- the default -- always
    returns an empty list, since it is what ILE does anyway.

    Refusing rather than ignoring is the point: a silently-inert accuracy option is
    worse than an unavailable one, because a comparison campaign can be run against
    it and believed.
    """
    validate_time_quadrature(time_quadrature)
    if time_quadrature == 'simpson':
        return []
    toks = _ile_tokens(ile_args)
    missing = []
    for flag, why in _PIPELINE_REQUIRED_ILE_FLAGS:
        # Direction matters: a token satisfies a required flag when the FLAG starts
        # with the TOKEN (the token is an abbreviation).  The reverse test would let
        # '--time-marginalization-quadrature' satisfy '--time-marginalization'.
        if not any(_matches(flag, t) for t in toks):
            missing.append("missing {} ({})".format(flag, why))
    for flag, why in _PIPELINE_EXCLUDING_ILE_FLAGS:
        if any(_matches(flag, t) for t in toks):
            missing.append("incompatible {} ({})".format(flag, why))
    return missing


def refuse_unhonourable_time_quadrature(time_quadrature, ile_args, where):
    """Raise unless ``ile_args`` can honour ``time_quadrature``.

    The raise lives HERE, not at the call sites, so that it is executable in a unit
    test: both pipeline scripts are top-level scripts that need real data before
    they reach their guard, and a guard whose only coverage is "an ast walk found a
    call by this name" survives being turned into a print.
    """
    missing = time_quadrature_pipeline_prereqs(time_quadrature, ile_args)
    if missing:
        raise ValueError(
            "time-marginalization quadrature {!r} was requested, but {} cannot honour it: "
            "{}.  Refusing rather than running the historical Simpson quadrature while "
            "reporting that you asked for something else.".format(
                time_quadrature, where, "; ".join(missing)))


def refuse_unless_time_quadrature_emitted(time_quadrature, ile_args, where):
    """Raise unless the REQUESTED quadrature is the one the bytes actually carry.

    The prerequisite check above reads the prerequisites in ``ile_args`` but takes
    the INTENT from the caller's parsed options, so it approves an argument string
    that never received the flag at all.  Three ways that happens in practice, all
    ending in a silent fall back to Simpson while the pipeline logs the opposite:

    * a helper that predates the option argparse-errors while a re-run directory
      still holds a STALE ``helper_ile_args.txt`` (the caller now also removes the
      generated file first and checks the helper's exit status);
    * ``--manual-extra-ile-args`` appends a second ``--time-marginalization-quadrature``
      after the helper's, and optparse takes the LAST occurrence;
    * any future refactor that drops the emission.

    So this checks the bytes for the flag itself.  ``time_quadrature`` of ``None``
    means nothing was requested, in which case the flag must be ABSENT unless the
    user put it there by hand -- and if they did, it is validated and prerequisite
    checked like any other request.
    """
    found = find_time_quadrature_in_ile_args(ile_args)
    if len(found) > 1:
        raise ValueError(
            "{} carries {} occurrences of {} ({!r}).  optparse takes the LAST, so the "
            "quadrature actually used would not be the one this workflow reports -- and "
            "the .sub file would read as though it were.  Refusing.".format(
                where, len(found), ILE_TIME_QUADRATURE_FLAG, found))
    if time_quadrature is None:
        if found:
            # Set by hand (--manual-extra-ile-args or an ini).  Not our flag, but it is
            # about to run, so hold it to the same standard rather than none at all.
            validate_time_quadrature(found[0])
            refuse_unhonourable_time_quadrature(found[0], ile_args, where)
        return
    if not found:
        raise ValueError(
            "time-marginalization quadrature {!r} was requested, but {} contains no {} at "
            "all.  The request was lost between the pipeline and the ILE arguments -- a "
            "stale or version-skewed helper path can do exactly this.  Refusing rather "
            "than submitting a "
            "campaign that would silently run Simpson.".format(
                time_quadrature, where, ILE_TIME_QUADRATURE_FLAG))
    if found[0] != time_quadrature:
        raise ValueError(
            "time-marginalization quadrature {!r} was requested but {} carries {!r}.  "
            "Refusing.".format(time_quadrature, where, found[0]))
    refuse_unhonourable_time_quadrature(time_quadrature, ile_args, where)


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
    X = _fft_rows(x, xpy=xpy)
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
    return _fft_rows(Xup, inverse=True, xpy=xpy) * factor


def reflected_bandlimited_upsample(x, factor, xpy=np):
    """Upsample a finite row after a literal ``2*n`` even reflection.

    Upsampling the gathered row directly identifies its unlike endpoints and
    can create global Gibbs ringing.  Instead periodize
    ``[x[0], ..., x[-1], x[-1], ..., x[0]]`` and return only the original
    forward interval.  Both periodic joins are value-continuous, every input
    sample is reproduced exactly, and no unavailable samples outside the
    integration domain are invented.  The duplicated turning samples are
    deliberate: the ``2*(n-1)`` reflection was less accurate in measured
    realistic and adversarial cases because it locates the turn differently.

    This is a numerical boundary condition, not a claim that the physical
    correlation reverses outside the caller's finite integration window.
    """
    x = xpy.asarray(x)
    factor = int(factor)
    if factor == 1:
        return x
    n = x.shape[-1]
    reflected = xpy.concatenate((x, xpy.flip(x, axis=-1)), axis=-1)
    dense = bandlimited_upsample(reflected, factor, xpy=xpy)
    return dense[..., :(n - 1) * factor + 1]


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
    # numpy.take_along_axis was introduced in 1.15, while RIFT still declares a
    # NumPy >=1.14 floor.  Flatten the leading axes and use ordinary advanced
    # indexing, which has the same semantics on NumPy and CuPy at that floor.
    lead_shape = jmax.shape
    flat_lnL = lnL_t.reshape((-1, n))
    row_index = xpy.arange(flat_lnL.shape[0])
    take = lambda j: flat_lnL[row_index, j.reshape(-1)].reshape(lead_shape)

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


def _default_simps(simps, xpy):
    """The caller's Simpson rule, defaulting to scipy's ONLY on the numpy backend.

    scipy's ``simpson`` raises ``TypeError: Implicit conversion to a NumPy array is
    not allowed`` on a cupy array, and that default is exactly how every
    ``--vectorized --gpu`` run of this option crashed.  Refuse rather than leave the
    trap armed -- and note the two rules are not interchangeable even where both run
    (the vendored GPU copy is an old scipy with ``even='avg'``), so a fallback row
    must be integrated by the rule the caller's own likelihood uses.
    """
    if simps is not None:
        return simps
    if xpy is not np:
        raise ValueError(
            "time marginalization: `simps` must be supplied for a non-numpy backend "
            "-- scipy's Simpson rule cannot consume a device array, and the fallback "
            "rows must use the rule the caller's own likelihood uses (on GPU, "
            "optimized_gpu_tools.simps).")
    from scipy import integrate
    return getattr(integrate, 'simpson', None) or integrate.simps


def _require_time_independent_rho_sq(rho_sq, xpy=np, rule='band-limited'):
    """Verify the load-bearing precondition rather than trusting the caller.

    A time-dependent self-term (the banded / slow-rotation response) would make the
    refined ``lnL`` wrong in a way no downstream check would catch.

    Compare only where both sides are finite.  A NaN self-term is NORMAL: the
    defensive proposal component deliberately draws physically-extreme points where
    the likelihood is NaN, and the historical path just returns NaN for that row and
    lets the sampler move on.  A bare ``==`` makes ``nan != nan`` trip this tripwire
    and abort the whole ILE process, blaming a rotating-response path that is not
    even in use.
    """
    rho_col = rho_sq[..., :1]
    _cmp = xpy.isfinite(rho_sq) & xpy.isfinite(xpy.broadcast_to(rho_col, rho_sq.shape))
    if not bool(xpy.all(xpy.where(_cmp, rho_sq == rho_col, True))):
        raise NotImplementedError(
            "%s time marginalization requires a time-independent rho_sq; the supplied "
            "self-term varies with time (banded / rotating-response path)" % rule)
    return rho_col


def _classify_rows(lnL_coarse, deltaT, npts, xpy=np):
    """Which rule each row gets.  THE SINGLE DEFINITION, shared with 'peak-local'.

    Returns ``(sigma, jmax, measurable, has_peak, flat, exposed, unmeasurable,
    factors)``.  A row is REFINED by the caller iff ``has_peak & (factors > 1)``.

    It lives here, and is called rather than copied, because
    :mod:`RIFT.likelihood.time_marginalization_peak_local` promises to change WHERE the
    refined grid is placed and nothing about WHICH rows get one.  That promise was made
    good by duplication and it did not survive: three clauses below -- reflection's
    demotion of the edge guard, ``boundary_unresolved``, and the guard's Simpson routing
    -- reached this module and not that one, and each showed up as peak-local silently
    returning a lower-accuracy value than this function for the same row.  Duplicated
    policy that MUST agree is policy that will eventually not.

    The boundary diagnostic applies only to rows that HAVE a resolvable peak: a row whose
    lnL(t) is constant -- an extrinsic sample in an antenna null, where kappa is
    numerically zero -- has an argmax of 0 by convention and would otherwise be reported
    as boundary-exposed.  That is harmless numerically (Simpson is exact on a constant)
    but it makes the diagnostic lie: measured on a random-sky batch of 4000, it reported
    810 "wrap-exposed" rows, which in a production log reads as a mis-centred window
    rather than as 810 rows with no signal in them.

    ``boundary_unresolved``: at the first/last sample the centred stencil is clipped
    inward, so for a severely under-resolved endpoint peak it can see positive curvature
    away from the maximum and label a strongly varying row "flat".  That would silently
    retain Simpson for exactly the truncated-boundary case we intend to report and
    reconstruct.  Such rows get a small seed factor; dense-grid remeasurement takes over
    as soon as the reflected peak is measurable.

    ``exposed`` REPORTS possible physical truncation and selects nothing.  Reflection
    removes the endpoint value jump, so neither boundary proximity nor a tail threshold
    may select a Simpson fallback: crossing an arbitrary threshold cannot silently change
    likelihood quality, and raising on such a row can be read upstream as a waveform
    failure and silently excise that configuration.
    """
    sigma, jmax, measurable = peak_width_from_lnL(lnL_coarse, deltaT, xpy=xpy)
    guard = max(1, int(npts * EDGE_GUARD_FRACTION))
    finite_lnL = xpy.isfinite(lnL_coarse)
    row_max = xpy.max(xpy.where(finite_lnL, lnL_coarse, -np.inf), axis=-1)
    row_min = xpy.min(xpy.where(finite_lnL, lnL_coarse, np.inf), axis=-1)
    varies = xpy.isfinite(row_max) & xpy.isfinite(row_min) & (row_max > row_min)
    boundary_unresolved = (measurable & (~xpy.isfinite(sigma)) & varies
                           & ((jmax == 0) | (jmax == npts - 1)))
    has_peak = measurable & (xpy.isfinite(sigma) | boundary_unresolved)
    flat = measurable & (~xpy.isfinite(sigma)) & (~boundary_unresolved)
    exposed = has_peak & ((jmax < guard) | (jmax > npts - 1 - guard))
    # Counted unconditionally, NOT `& ~exposed`: an all -inf row also has an argmax of 0,
    # so a conditional counter would hide it behind the edge guard.
    unmeasurable = ~measurable
    factors = xpy.maximum(required_upsample_factors(sigma, deltaT, xpy=xpy), 1)
    factors = xpy.where(boundary_unresolved, xpy.maximum(factors, 4), factors)
    return (sigma, jmax, measurable, has_peak, flat, exposed, unmeasurable, factors)


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


def draw_piecewise_linear_log_posterior(lnL_t, dx, t0=0.0,
                                        uniforms=None, xpy=np):
    """Draw one continuous time per row from a nodal log density.

    Between adjacent nodes the *density* ``exp(lnL)`` is linear.  Its interval
    mass is therefore exactly the trapezoid used by the refined quadrature.  We
    first choose an interval by those masses, then invert the linear-density CDF
    analytically inside it.  The result has no output lattice: ``dx`` describes
    the representation's knots, not the support of the returned variate.

    ``uniforms`` may be supplied as shape ``(n_rows, 2)``.  The first variate
    selects the interval and the second selects the position inside it.  This is
    both the reproducibility seam and the way CPU/GPU tests ask the two backends
    exactly the same question.  If omitted, numpy's global generator is used,
    preserving the driver's existing ``--seed`` contract.

    Returns ``(times, lnL_at_times)`` in the input backend.  ``-inf`` nodes are
    supported and carry zero density.  NaN/+inf nodes and rows with no positive
    finite mass are rejected rather than assigned an invented timestamp.
    """
    values = xpy.asarray(lnL_t)
    if values.ndim == 1:
        values = values[xpy.newaxis, :]
    if values.ndim != 2 or values.shape[-1] < 2:
        raise ValueError("lnL_t must have shape (n_rows, n_time>=2)")
    if not bool(xpy.all((~xpy.isnan(values)) & (~xpy.isposinf(values)))):
        raise ValueError("continuous time posterior contains NaN or +inf")

    n_rows, n_time = values.shape
    if uniforms is None:
        uniforms = np.random.random((n_rows, 2))
    uniforms = xpy.asarray(uniforms, dtype=np.float64)
    if uniforms.shape != (n_rows, 2):
        raise ValueError("uniforms must have shape (n_rows, 2)")
    if not bool(xpy.all((uniforms >= 0.0) & (uniforms < 1.0))):
        raise ValueError("uniforms must lie in [0, 1)")

    finite = xpy.isfinite(values)
    off = xpy.max(xpy.where(finite, values, -np.inf), axis=-1)
    if not bool(xpy.all(xpy.isfinite(off))):
        raise ValueError("time posterior has no finite positive mass")
    density = xpy.where(finite, xpy.exp(values - off[:, xpy.newaxis]), 0.0)
    interval_mass = 0.5 * float(dx) * (density[:, :-1] + density[:, 1:])
    total = xpy.sum(interval_mass, axis=-1)
    if not bool(xpy.all(xpy.isfinite(total) & (total > 0.0))):
        raise ValueError("time posterior has no finite positive mass")

    cdf = xpy.cumsum(interval_mass, axis=-1) / total[:, xpy.newaxis]
    # `uniforms < 1` guarantees an interval, but clip defensively against a
    # backend whose final cumsum rounds a hair below one.
    # `<=` skips a leading/embedded zero-mass plateau even when the supplied
    # variate is exactly zero or exactly on a cumulative boundary.  `<` would
    # select a zero-density interval at u=0 and return lnL=-inf for a posterior
    # that has positive mass later in the window.
    interval = xpy.sum(cdf <= uniforms[:, :1], axis=-1).astype(np.int64)
    interval = xpy.minimum(interval, n_time - 2)
    row = xpy.arange(n_rows)
    a = density[row, interval]
    b = density[row, interval + 1]
    delta = b - a
    # A pseudo-random float can (very rarely) be exactly zero.  On an interval
    # whose left endpoint has zero density, the literal inverse-CDF endpoint
    # would then return lnL=-inf and could be silently excised downstream.  Use
    # the centre of the lowest float64 RNG bin for that one endpoint, matching
    # the open-interval variate required by a continuous posterior draw.
    r = xpy.maximum(uniforms[:, 1], 0.5 * np.finfo(float).eps)

    # For density p(u)=a+(b-a)u on u in [0,1], inverse-CDF sampling gives
    # p(u)^2 = a^2 + r*(b^2-a^2).  Use the uniform limit when the interval is
    # numerically flat to avoid cancellation in (p-a)/(b-a).
    scale = xpy.maximum(xpy.maximum(xpy.abs(a), xpy.abs(b)), 1.0)
    flat = xpy.abs(delta) <= 16.0 * np.finfo(float).eps * scale
    # Scale locally before squaring.  A selected far-tail interval can have
    # representable endpoint densities whose squares underflow; the CDF inverse
    # must not turn that positive interval into zero density.
    local_scale = xpy.maximum(a, b)
    a_scaled = a / local_scale
    b_scaled = b / local_scale
    endpoint_density = local_scale * xpy.sqrt(xpy.maximum(
        0.0, a_scaled * a_scaled
        + r * (b_scaled * b_scaled - a_scaled * a_scaled)))
    frac = xpy.where(flat, r, (endpoint_density - a) /
                     xpy.where(flat, 1.0, delta))
    frac = xpy.clip(frac, 0.0, 1.0)
    drawn_density = a + delta * frac
    times = float(t0) + (interval.astype(np.float64) + frac) * float(dx)
    lnL_draw = off + xpy.log(drawn_density)
    return times, lnL_draw


def time_marginalize_bandlimited(kappa, rho_sq, deltaT, loglikelihood,
                                 phase_marginalization=False, simps=None,
                                 lnL_coarse=None, return_time_draw=False,
                                 draw_uniforms=None, t0=0.0, xpy=np):
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
    return_time_draw : bool, optional
        Also return one continuous conditional-posterior draw per row and its
        instantaneous log likelihood.  Refined rows use the exact same validated
        dense representation as the trapezoid integral.  Unrefined rows are
        already resolved and are drawn continuously between their coarse knots.
    draw_uniforms : array, optional
        Shape ``(n_extrinsic, 2)`` uniforms for deterministic draws.  Omit to use
        numpy's global RNG, matching the batch driver's ``--seed`` behavior.
    t0 : float, optional
        Time of the first coarse knot; returned draws are in this coordinate.

    Returns
    -------
    lnL : (n_extrinsic,) float
        With ``return_time_draw=True``, returns
        ``(lnL, time_draw, lnL_at_draw)``.
    """
    simps = _default_simps(simps, xpy)

    kappa = xpy.asarray(kappa)
    rho_sq = xpy.asarray(rho_sq)
    npts = kappa.shape[-1]
    n_rows = kappa.shape[0]
    deltaT = float(deltaT)

    _require_time_independent_rho_sq(rho_sq, xpy=xpy, rule='band-limited')
    rho_col = rho_sq[..., :1]

    _term = (lambda k: xpy.abs(k)) if phase_marginalization else (lambda k: k.real)
    if lnL_coarse is None:
        lnL_coarse = loglikelihood(_term(kappa), rho_sq)

    # THE SINGLE DEFINITION, shared with 'peak-local' -- see _classify_rows.
    (sigma, jmax, measurable, has_peak, flat, exposed, unmeasurable,
     factors) = _classify_rows(lnL_coarse, deltaT, npts, xpy=xpy)

    # A row is REFINED only if it has a trustworthy peak AND the derivation
    # actually asks for a finer grid.  Reflection removes the endpoint value
    # jump, so neither boundary proximity nor a tail threshold selects a Simpson
    # fallback.  ``exposed`` reports possible physical truncation only.
    # Everything else -- unmeasurable, flat, or already resolved -- gets the
    # historical Simpson value.  (The log-sum-exp offset changes for every row --
    # see last_report() -- so this is a statement about the rule, not a promise
    # that unrefined rows come back bit-identical.)
    # The alternative, letting an unrefined row fall through to
    # a coarse TRAPEZOID, is numerically a non-event but silently changes the
    # rule for rows this option was never meant to touch, and costs the property
    # a reviewer can actually check.  (Trapezoid is in fact slightly the better
    # rule on a resolved integrand -- 5e-6 nats against an analytic truth, versus
    # Simpson's 5e-6 the other way -- so this trades nothing measurable for an
    # auditable claim.)
    refined = has_peak & (factors > 1)

    # Do not pay for the historical coarse-grid integral on rows that we already
    # know will be overwritten by the dense reconstruction below.  In ordinary
    # AV ILE the coarse likelihood has already been evaluated for classification;
    # the old unconditional call added another exp/reduction over every
    # extrinsic×time point even when every row required refinement.  Allocate the
    # result once and run Simpson only on the rows for which it is the answer.
    out = xpy.empty((n_rows,), dtype=xpy.asarray(lnL_coarse).dtype)
    unrefined = ~refined
    if bool(xpy.any(unrefined)):
        idx_unrefined = xpy.where(unrefined)[0]
        out[idx_unrefined] = _log_simps_rows(
            lnL_coarse[idx_unrefined], deltaT, simps, xpy=xpy)
    time_draw = None
    lnL_at_draw = None
    if return_time_draw:
        if draw_uniforms is None:
            draw_uniforms = np.random.random((n_rows, 2))
        draw_uniforms = xpy.asarray(draw_uniforms, dtype=np.float64)
        if draw_uniforms.shape != (n_rows, 2):
            raise ValueError("draw_uniforms must have shape (n_rows, 2)")
        # Seed every row from the already-resolved coarse representation.  Rows
        # refined below are overwritten with draws from their final validated
        # dense representation; flat/already-resolved rows remain continuous
        # rather than being snapped back to a coarse knot.
        time_draw, lnL_at_draw = draw_piecewise_linear_log_posterior(
            lnL_coarse, deltaT, t0=t0, uniforms=draw_uniforms, xpy=xpy)

    hist = {}
    n_refine_total = 0
    sigma_seen = np.inf
    # Reuse chirps across every batch at a given factor, but only for this
    # marginalization call.  In particular, do not pin successively larger GPU
    # plans in a process-global cache after a high-SNR cell has finished.
    retained_plan_cache = {}
    transform_report = _new_transform_report()
    for f in xpy.unique(xpy.where(refined, factors, 1)):
        f = int(f)
        if f == 1:
            continue
        sel = refined & (factors == f)
        n_sel = int(xpy.sum(sel))
        if not n_sel:
            continue
        idx = xpy.where(sel)[0]
        vals, group_hist, n_ref, s_min, drawn_t, drawn_lnL = _integrate_group(
            kappa[idx], rho_col[idx], npts, deltaT, f, loglikelihood, _term,
            draw_uniforms_rows=(draw_uniforms[idx] if return_time_draw else None),
            t0=t0, retained_plan_cache=retained_plan_cache,
            transform_report=transform_report, xpy=xpy)
        out[idx] = vals
        if return_time_draw:
            time_draw[idx] = drawn_t
            lnL_at_draw[idx] = drawn_lnL
        for f_used, n_used in group_hist.items():
            hist[int(f_used)] = hist.get(int(f_used), 0) + int(n_used)
        n_refine_total += n_ref
        sigma_seen = min(sigma_seen, s_min)

    strategies = []
    if transform_report["retained_fft_batches"]:
        strategies.append("retained-grid-zoomfft")
    if transform_report["full_fft_selected_batches"]:
        strategies.append("full-padding-selected")
    if transform_report["full_fft_fallback_batches"]:
        strategies.append("full-padding-fallback")
    transform_strategy = (strategies[0] if len(strategies) == 1 else
                          ("mixed:" + ",".join(strategies) if strategies
                           else "not-used"))
    transform_report.pop("warned_fallback_reasons")
    transform_report.update(
        bandlimited_fft_strategy=transform_strategy,
        n_retained_fft_plans=len(retained_plan_cache),
    )

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
        cpu_fft_workers=(_cpu_fft_workers() if xpy is np else None),
        **transform_report
    )
    if return_time_draw:
        return out, time_draw, lnL_at_draw
    return out


def _integrate_group(kappa_rows, rho_col_rows, npts, deltaT, factor,
                     loglikelihood, _term, draw_uniforms_rows=None, t0=0.0,
                     retained_plan_cache=None, transform_report=None, xpy=np):
    """Refine and integrate one group of rows that share a derived factor.

    Returns ``(values, factor_histogram, n_refinements, sigma_dense_min,
    time_draws, lnL_at_draws)``.  The final two entries are ``None`` unless
    ``draw_uniforms_rows`` is supplied.
    """
    n_rows = kappa_rows.shape[0]
    if retained_plan_cache is None:
        retained_plan_cache = {}
    if transform_report is None:
        transform_report = _new_transform_report()
    n_refine = 0
    remaining = xpy.arange(n_rows)
    values = xpy.empty((n_rows,), dtype=np.float64)
    time_values = (xpy.empty((n_rows,), dtype=np.float64)
                   if draw_uniforms_rows is not None else None)
    draw_lnL_values = (xpy.empty((n_rows,), dtype=np.float64)
                       if draw_uniforms_rows is not None else None)
    factor_hist = {}
    sigma_seen = np.inf
    while int(remaining.size):
        if factor > UPSAMPLE_FACTOR_MAX:
            raise RuntimeError(
                "band-limited time marginalization needs an upsampling factor above "
                "the ceiling UPSAMPLE_FACTOR_MAX=%d (deltaT=%.3e s).  This is far "
                "beyond what the band limit can justify: suspect a pathological "
                "lnL(t), not an under-resolved one." % (UPSAMPLE_FACTOR_MAX, deltaT))

        dx_dense = deltaT / factor
        # Chunk the extrinsic axis so one dense temporary stays inside the
        # working-set budget.  Rows are independent; this cannot change results.
        # The FFT period is 2*n after reflection; budget for it and the forward
        # kappa/rho/lnL temporaries.
        per_row = npts * factor * 16 * 8
        n_remaining = int(remaining.size)
        chunk = max(1, min(n_remaining, int(_DENSE_CHUNK_BYTES // max(per_row, 1))))

        pieces = []
        draw_time_pieces = []
        draw_lnL_pieces = []
        sigma_pieces = []
        for start in range(0, n_remaining, chunk):
            take = remaining[start:start + chunk]
            k_up = _reflected_upsample_for_integration(
                kappa_rows[take], factor, retained_plan_cache,
                transform_report, xpy=xpy)
            rho_up = xpy.broadcast_to(rho_col_rows[take], k_up.shape)
            lnL_up = loglikelihood(_term(k_up), rho_up)
            s_d, _, meas = peak_width_from_lnL(lnL_up, dx_dense, xpy=xpy)
            s_d = xpy.where(meas, s_d, np.inf)
            sigma_pieces.append(s_d)
            pieces.append(_log_trapz_over_window(lnL_up, dx_dense, npts, factor, xpy=xpy))
            if draw_uniforms_rows is not None:
                drawn_t, drawn_lnL = draw_piecewise_linear_log_posterior(
                    lnL_up, dx_dense, t0=t0,
                    uniforms=draw_uniforms_rows[take], xpy=xpy)
                draw_time_pieces.append(drawn_t)
                draw_lnL_pieces.append(drawn_lnL)

        # The assertion that turns the derivation into a guarantee: the width
        # remeasured on the grid we actually integrated on must still satisfy the
        # criterion.  A coarse-grid estimate can be optimistic when the peak is
        # strongly non-Gaussian; this catches that and pays for another doubling
        # instead of reporting a number it cannot defend.
        current_values = xpy.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        current_sigma = (xpy.concatenate(sigma_pieces)
                         if len(sigma_pieces) > 1 else sigma_pieces[0])
        finite_sigma = xpy.isfinite(current_sigma)
        if bool(xpy.any(finite_sigma)):
            sigma_seen = min(sigma_seen, float(xpy.min(current_sigma[finite_sigma])))
        resolved = (~finite_sigma) | (dx_dense <= current_sigma / UPSAMPLE_SAFETY)
        accepted = remaining[resolved]
        values[accepted] = current_values[resolved]
        n_accepted = int(xpy.sum(resolved))
        if n_accepted:
            factor_hist[int(factor)] = factor_hist.get(int(factor), 0) + n_accepted
        if draw_uniforms_rows is not None:
            current_t = (xpy.concatenate(draw_time_pieces) if len(draw_time_pieces) > 1
                         else draw_time_pieces[0])
            current_draw_lnL = (xpy.concatenate(draw_lnL_pieces)
                                if len(draw_lnL_pieces) > 1 else draw_lnL_pieces[0])
            time_values[accepted] = current_t[resolved]
            draw_lnL_values[accepted] = current_draw_lnL[resolved]
        remaining = remaining[~resolved]
        if not int(remaining.size):
            return (values, factor_hist, n_refine, sigma_seen,
                    time_values, draw_lnL_values)
        factor *= 2
        n_refine += 1
