# -*- coding: utf-8 -*-
"""Plottability checks for the ``range`` argument passed to ``corner.corner``.

``corner`` raises ``ValueError`` -- it does not warn, and it does not draw an empty
panel -- when a 2-D panel's histogram comes out empty.  In a driver that plots
several coordinate pairs at the end of a run, that turns a cosmetic problem into a
nonzero exit status for the whole job.

:func:`unplottable_reason` answers the narrow question "would corner reject this?"
without calling it, so a caller can widen a degenerate interval, or decline one
panel and keep going, while every OTHER corner failure still propagates.
:func:`overlay_or_warn` is the one-line form for an overlay added to a figure that
already exists.  Neither catches exceptions: a caller that wraps corner in
``try/except`` loses the ability to tell an unplottable sample from broken
plotting, which is the distinction these exist to preserve.

The predicates mirror ``corner.core.hist2d`` as of corner 2.3.0:

* bin edges come from ``np.linspace(min(range[j]), max(range[j]), bins+1)``, so an
  INVERTED interval is sorted by corner and is not an error.  Neither is a ZERO-WIDTH
  one, measured rather than assumed: every edge collapses to the same value,
  ``np.histogram2d`` accepts non-DECREASING edges, and samples sitting on that value
  are still counted.  It gives a singular axis, so :func:`pad_degenerate_intervals`
  widens it, but it is not a reason to drop a panel;
* the histogram is built with the caller's ``weights`` and is rejected when
  ``H.sum() == 0``, i.e. when no sample carrying positive weight lands in the
  panel's box.  This is a per-PAIR condition: a row outside the box in a third
  coordinate still counts towards the panel it is inside.
"""

from __future__ import print_function, absolute_import

import numpy as np

__all__ = ["pad_degenerate_intervals", "unplottable_reason", "overlay_or_warn"]


def pad_degenerate_intervals(ranges, rel_pad=1e-3, abs_pad=1e-6):
    """Widen any zero-width interval in ``ranges`` so it does not give a singular axis.

    A coordinate held at one value across the whole input grid (zero spins on a BBH
    grid, a fixed mass) gives ``[v, v]``.  corner does not reject that -- see the
    module docstring -- but it collapses every bin edge onto one value and matplotlib
    then expands the axis itself with a warning, so the panel is drawn on an axis
    nobody chose.  The pad is ``max(rel_pad*|v|, abs_pad)`` on each side.

    Returns ``(new_ranges, padded)``: a fresh list of ``[lo, hi]`` lists, and the
    sorted indices that were widened.  Non-finite bounds are left alone -- they
    are reported by :func:`unplottable_reason`, not repaired here.
    """
    out = []
    padded = []
    for j, interval in enumerate(ranges):
        lo, hi = float(interval[0]), float(interval[1])
        if np.isfinite(lo) and np.isfinite(hi) and lo == hi:
            pad = max(abs(lo) * rel_pad, abs_pad)
            lo, hi = lo - pad, hi + pad
            padded.append(j)
        out.append([lo, hi])
    return out, padded


def unplottable_reason(sample, ranges, weights=None, labels=None):
    """Return None if ``corner.corner(sample, range=ranges, weights=weights)``
    can build every 2-D panel, otherwise a one-line reason it cannot.

    ``sample`` is ``(nsamples, ndim)`` and ``ranges`` is one ``(lo, hi)`` per
    column.  ``labels`` names the columns in the message; column indices are used
    when it is omitted.

    A shape disagreement is a caller bug rather than an unplottable sample, so it
    raises ``ValueError`` here instead of being reported as a reason to skip.

    Being too STRICT here is a failure too: a panel declined for a condition corner
    would have drawn is a plot silently lost.  The cases in
    test/test_corner_range_guard.py are run through the real ``corner.corner`` in both
    directions for that reason.

    With ``ndim == 1`` there are no 2-D panels and corner does not raise, so the
    emptiness check is still reported: the panel would be blank, and no caller in
    this package plots a single column.
    """
    arr = np.asarray(sample, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            "unplottable_reason: expected a (nsamples, ndim) sample, got shape {}".format(arr.shape))
    ndim = arr.shape[1]
    if len(ranges) != ndim:
        raise ValueError(
            "unplottable_reason: {} range intervals for {} columns".format(len(ranges), ndim))

    def name(j):
        if labels is None or j >= len(labels):
            return "column {}".format(j)
        return str(labels[j])

    if arr.shape[0] == 0:
        return "sample is empty (0 rows)"

    bounds = []
    for j in range(ndim):
        lo, hi = float(ranges[j][0]), float(ranges[j][1])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return "range for {} is not finite: [{}, {}]".format(name(j), lo, hi)
        # corner sorts each interval, so an inverted one is not an error there either,
        # and neither is a zero-width one -- see the module docstring.  Both are left to
        # the emptiness check below, which is what corner actually rejects on.
        bounds.append((min(lo, hi), max(lo, hi)))

    if weights is None:
        wt = np.ones(arr.shape[0], dtype=float)
    else:
        wt = np.asarray(weights, dtype=float)
        if wt.shape != (arr.shape[0],):
            raise ValueError(
                "unplottable_reason: {} weights for {} rows".format(wt.shape, arr.shape[0]))

    # NaN compares False, so a non-finite entry counts as outside -- which is what
    # np.histogram2d does with it too.
    inside = [(arr[:, j] >= bounds[j][0]) & (arr[:, j] <= bounds[j][1]) for j in range(ndim)]

    for j in range(ndim):
        if not np.any(inside[j]):
            col = arr[:, j][np.isfinite(arr[:, j])]
            span = "all non-finite" if col.size == 0 else "[{}, {}]".format(col.min(), col.max())
            return "no sample falls inside the plotted range of {}: {} vs data {}".format(
                name(j), list(bounds[j]), span)

    if ndim == 1:
        if not (wt[inside[0]].sum() > 0):
            return "no positively weighted sample inside the plotted range of {}".format(name(0))
        return None

    for j in range(ndim):
        for k in range(j + 1, ndim):
            if not (wt[inside[j] & inside[k]].sum() > 0):
                return (
                    "no sample falls inside the plotted range in the ({}, {}) panel: "
                    "{} x {}".format(name(j), name(k), list(bounds[j]), list(bounds[k])))
    return None


def overlay_or_warn(sample, ranges, coord_labels, what, **kwargs):
    """Add one overlay to an existing corner figure, unless corner would reject it.

    A corner plot here draws several data sets against ONE range: a posterior, the
    input grid, its significant subset, sometimes lalinference.  Only the first of
    those had any say in the range.  ``kwargs`` goes straight to ``corner.corner``
    along with ``range=ranges``; the figure to keep drawing on is returned, which on
    a declined overlay is ``kwargs['fig']`` unchanged.

    The positional names are deliberately not corner's.  ``labels`` IS a corner
    keyword, and an earlier version of this signature took the coordinate names under
    that name: the one call site that passes ``labels=`` to corner -- the lalinference
    overlay, which no CI job runs -- then raised TypeError before corner was reached.
    test_corner_range_guard.py walks the driver's call sites for that collision.
    """
    reason = unplottable_reason(sample, ranges, labels=coord_labels)
    if reason:
        print(" WARNING: skipping the ", what, " overlay -- ", reason)
        return kwargs.get('fig')
    import corner
    return corner.corner(sample, range=ranges, **kwargs)
