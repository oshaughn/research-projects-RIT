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
from scipy.interpolate import CubicSpline


TIME_POSTERIOR_EXPORT_MODES = ("auto", "continuous", "grid")


def resolve_time_posterior_export_mode(requested, time_interpolation):
    """Resolve ``auto`` against the likelihood's time-interpolation mode."""
    if requested not in TIME_POSTERIOR_EXPORT_MODES:
        raise ValueError("unknown time-posterior export mode %r" % (requested,))
    if requested == "auto":
        return "continuous" if time_interpolation != "nearest" else "grid"
    return requested


def _interval_log_envelopes(spline, knots, log_values):
    """Return the exact maximum of a cubic spline on every knot interval."""
    maxima = np.maximum(log_values[:-1], log_values[1:]).astype(float, copy=True)
    roots = np.asarray(spline.derivative().roots(extrapolate=False), dtype=float)
    roots = roots[np.isfinite(roots)]
    if roots.size:
        interval = np.searchsorted(knots, roots, side="right") - 1
        interval = np.clip(interval, 0, len(knots) - 2)
        inside = (roots >= knots[interval]) & (roots <= knots[interval + 1])
        for i, value in zip(interval[inside], np.asarray(spline(roots[inside]))):
            maxima[i] = max(maxima[i], float(value))
    return maxima


def draw_continuous_time_posterior(tvals, lnlt, rng=None):
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

    Returns
    -------
    times, log_likelihoods : ndarray
        One accepted continuous draw and its interpolated log likelihood for
        each input row.
    """
    tvals = np.asarray(tvals, dtype=float)
    values = np.asarray(lnlt, dtype=float)
    one_row = values.ndim == 1
    values = np.atleast_2d(values)
    if tvals.ndim != 1 or tvals.size < 2 or not np.all(np.diff(tvals) > 0):
        raise ValueError("tvals must be a strictly increasing 1-D grid")
    if values.shape[1] != tvals.size:
        raise ValueError("lnlt's final axis must match tvals")
    if not np.all(np.isfinite(values)):
        raise ValueError("continuous time export requires finite lnL(t)")
    if rng is None:
        rng = np.random

    widths = np.diff(tvals)
    times = np.empty(values.shape[0], dtype=float)
    log_likelihoods = np.empty(values.shape[0], dtype=float)
    for row, log_values in enumerate(values):
        spline = CubicSpline(tvals, log_values)
        maxima = _interval_log_envelopes(spline, tvals, log_values)
        shift = float(np.max(maxima))
        envelope_mass = widths * np.exp(maxima - shift)
        total = float(np.sum(envelope_mass))
        if not np.isfinite(total) or total <= 0:
            raise ValueError("time posterior has no finite positive mass")
        probabilities = envelope_mass / total

        while True:
            interval = int(rng.choice(len(widths), p=probabilities))
            candidate = float(rng.uniform(tvals[interval], tvals[interval + 1]))
            log_candidate = float(spline(candidate))
            if float(rng.uniform()) <= np.exp(log_candidate - maxima[interval]):
                times[row] = candidate
                log_likelihoods[row] = log_candidate
                break

    if one_row:
        return times[0], log_likelihoods[0]
    return times, log_likelihoods
