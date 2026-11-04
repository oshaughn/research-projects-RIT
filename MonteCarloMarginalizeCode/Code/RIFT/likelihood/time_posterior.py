"""Continuous draws from an interpolated time posterior.

The ILE time-marginalization likelihood is evaluated on a regular FFT grid.
When sub-sample interpolation is enabled, exporting a time by choosing one of
those grid points throws that resolution away.  This module draws directly
from ``exp(CubicSpline(t, lnL))`` instead.

The sampler uses a piecewise-constant rejection envelope.  On every spline
interval the envelope is the exact maximum of the cubic (endpoints plus all
stationary points), so accepted draws are continuous and have the requested
interpolated density without introducing another export lattice.
"""

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator


TIME_POSTERIOR_EXPORT_MODES = ("auto", "continuous", "grid")
TIME_POSTERIOR_WORKING_SET_MAX = 512 * 1024 * 1024
TIME_POSTERIOR_BYTES_PER_CELL = 128


def validate_time_posterior_working_set(n_rows, n_times,
                                        limit=TIME_POSTERIOR_WORKING_SET_MAX):
    """Refuse a dense export before its dominant full matrices can exhaust GPU RAM."""
    if not isinstance(n_rows, (int, np.integer)) or n_rows < 0:
        raise ValueError("n_rows must be a non-negative integer")
    if not isinstance(n_times, (int, np.integer)) or n_times < 2:
        raise ValueError("n_times must be an integer >= 2")
    estimated = int(n_rows) * int(n_times) * TIME_POSTERIOR_BYTES_PER_CELL
    if estimated > int(limit):
        raise MemoryError(
            "continuous time export would require an unsafe dense working set "
            "(estimated at least {:.3g} MiB for {} rows x {} times; limit {:.3g} "
            "MiB). Reduce --fairdraw-extrinsic-output-n-max or use "
            "--time-posterior-export grid.".format(
                estimated / 2.0**20, n_rows, n_times, int(limit) / 2.0**20))
    return estimated


def legacy_time_interpolation_enabled(value):
    """Parse the LISA driver's historical boolean/string interpolation flag."""
    normalized = str(value).strip().lower()
    if normalized in ("true", "t", "yes", "y", "1", "on", "cubic", "sinc"):
        return True
    if normalized in ("false", "f", "no", "n", "0", "off", "none", "nearest"):
        return False
    raise ValueError(
        "--interpolate-time: unrecognised LISA value {!r}; use a boolean, "
        "nearest, cubic, or sinc".format(value))


def resolve_time_posterior_export_mode(requested, time_interpolation,
                                       continuous_available=True):
    """Resolve ``auto`` against interpolation and driver capabilities.

    ``continuous_available=False`` affects only ``auto``: it preserves the
    driver's historical grid export.  An explicit ``continuous`` request is
    returned unchanged so the caller can reject it with a capability-specific
    error instead of silently downgrading it.
    """
    if requested not in TIME_POSTERIOR_EXPORT_MODES:
        raise ValueError("unknown time-posterior export mode %r" % (requested,))
    if requested == "auto":
        return ("continuous" if continuous_available and
                time_interpolation != "nearest" else "grid")
    return requested


def _interval_log_envelopes(spline, knots, log_values):
    """Return the exact maximum of a cubic spline on every knot interval."""
    maxima = np.maximum(log_values[:-1], log_values[1:]).astype(float, copy=True)
    # SciPy 1.0's PPoly.roots() has no ``extrapolate`` keyword.  Calling it
    # without the keyword is cross-version-safe because the in-domain test
    # below already rejects every extrapolated root.
    roots = np.asarray(spline.derivative().roots(), dtype=float)
    roots = roots[np.isfinite(roots)]
    if roots.size:
        interval = np.searchsorted(knots, roots, side="right") - 1
        interval = np.clip(interval, 0, len(knots) - 2)
        inside = (roots >= knots[interval]) & (roots <= knots[interval + 1])
        for i, value in zip(interval[inside], np.asarray(spline(roots[inside]))):
            maxima[i] = max(maxima[i], float(value))
    return maxima


def draw_continuous_time_posterior(tvals, lnlt, rng=None, max_attempts=100000):
    """Draw one continuous time per row from an interpolated ``lnL(t)``.

    Parameters
    ----------
    tvals : array-like, shape (n_time,)
        Strictly increasing time knots.
    lnlt : array-like, shape (n_rows, n_time) or (n_time,)
        Log likelihood at the knots.
    rng : numpy RNG-like object, optional
        Must provide ``choice`` and ``uniform``.  The default is
        ``numpy.random`` so the driver's existing ``--seed`` contract remains
        unchanged.
    max_attempts : int, optional
        Maximum rejection proposals per row.  Exhaustion raises rather than
        allowing a pathological interpolant or RNG to hang an ILE job.

    Returns
    -------
    times, log_likelihoods : ndarray
        One accepted continuous draw and its interpolated log likelihood for
        each input row.
    """
    tvals = np.asarray(tvals, dtype=float)
    values = np.asarray(lnlt, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError("lnlt must be a 1-D or 2-D array")
    one_row = values.ndim == 1
    values = np.atleast_2d(values)
    if tvals.ndim != 1 or tvals.size < 2 or not np.all(np.diff(tvals) > 0):
        raise ValueError("tvals must be a strictly increasing 1-D grid")
    if values.shape[1] != tvals.size:
        raise ValueError("lnlt's final axis must match tvals")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("continuous time export does not accept NaN or +inf lnL(t)")
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    if rng is None:
        rng = np.random

    widths = np.diff(tvals)
    times = np.empty(values.shape[0], dtype=float)
    log_likelihoods = np.empty(values.shape[0], dtype=float)
    for row, log_values in enumerate(values):
        finite = np.isfinite(log_values)
        if not np.any(finite):
            raise ValueError("time posterior has no finite positive mass")

        if np.all(finite):
            # Preserve the requested cubic interpolation of lnL for the normal
            # path.  Work relative to the envelope maximum to avoid overflow.
            spline = CubicSpline(tvals, log_values)
            maxima = _interval_log_envelopes(spline, tvals, log_values)
            shift = float(np.max(maxima))
            envelope_mass = widths * np.exp(maxima - shift)

            def evaluate(candidate):
                log_candidate = float(spline(candidate))
                interval_maximum = maxima[interval]
                ratio = np.exp(log_candidate - interval_maximum)
                return log_candidate, min(1.0, float(ratio))
        else:
            # -inf is a valid zero-posterior-mass value.  A log spline cannot
            # represent it.  Interpolate the shifted density with PCHIP, which
            # preserves non-negativity and the zero knots without manufacturing
            # the huge ringing that replacing -inf by an arbitrary log floor can
            # cause.  This branch is only used for rows containing -inf.
            shift = float(np.max(log_values[finite]))
            density_values = np.zeros_like(log_values)
            density_values[finite] = np.exp(log_values[finite] - shift)
            spline = PchipInterpolator(tvals, density_values)
            maxima = _interval_log_envelopes(
                spline, tvals, density_values)
            maxima = np.maximum(maxima, 0.0)
            envelope_mass = widths * maxima

            def evaluate(candidate):
                density = max(0.0, float(spline(candidate)))
                interval_maximum = maxima[interval]
                ratio = density / interval_maximum if interval_maximum > 0 else 0.0
                log_candidate = shift + np.log(density) if density > 0 else -np.inf
                return log_candidate, min(1.0, ratio)

        total = float(np.sum(envelope_mass))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("time posterior has no finite positive mass")
        probabilities = envelope_mass / total

        for _attempt in range(max_attempts):
            interval = int(rng.choice(len(widths), p=probabilities))
            candidate = float(rng.uniform(tvals[interval], tvals[interval + 1]))
            log_candidate, accept_probability = evaluate(candidate)
            # A float RNG can return exactly zero.  With ``<=``, a proposal at
            # a zero-density endpoint then satisfies 0 <= 0 and escapes with
            # lnL=-inf, which downstream fair-draw filtering can silently
            # excise.  Zero posterior density must never be accepted.
            if (accept_probability > 0.0 and
                    float(rng.uniform()) < accept_probability):
                times[row] = candidate
                log_likelihoods[row] = log_candidate
                break
        else:
            raise RuntimeError(
                "continuous time-posterior rejection sampler exhausted "
                "{} proposals for row {}".format(max_attempts, row))

    if one_row:
        return times[0], log_likelihoods[0]
    return times, log_likelihoods
