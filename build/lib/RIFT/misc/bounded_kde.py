#
# kde.py: KDE utilities for density estimation in unusual topologies.
#
# Copyright 2012 Will M. Farr <will.farr@ligo.org>
# Modified 2017 Leo P. Singer <leo.singer@ligo.org> to handle 1D KDEs
# gracefully.
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

from __future__ import division
import numpy as np
from scipy.stats import gaussian_kde


class BoundedKDE(gaussian_kde):
    """Density estimation using a KDE on bounded domains.
    Bounds can be any combination of low or high (if no bound, set to
    ``float('inf')`` or ``float('-inf')``), and can be periodic or
    non-periodic.  Cannot handle topologies that have
    multi-dimensional periodicities; will only handle topologies that
    are direct products of (arbitrary numbers of) R, [0,1], and S1.
    :param pts:
        ``(Ndim, Npts)`` shaped array of points (as in :class:`gaussian_kde`).
    :param low: 
        Lower bounds; if ``None``, assume no lower bounds.
    :param high:
        Upper bounds; if ``None``, assume no upper bounds.
    :param periodic:
        Boolean array giving periodicity in each dimension; if
        ``None`` assume no dimension is periodic.
    :param bw_method: (optional)
        Bandwidth estimation method (see :class:`gaussian_kde`)."""
    def __init__(self, pts, low=-np.inf, high=np.inf, periodic=False,
                 bw_method=None):

        super(BoundedKDE, self).__init__(pts, bw_method=bw_method)
        self._low = np.broadcast_to(
            low, self.d).astype(self.dataset.dtype)
        self._high = np.broadcast_to(
            high, self.d).astype(self.dataset.dtype)
        self._periodic = np.broadcast_to(
            periodic, self.d).astype(bool)

    def evaluate(self, pts):
        """Evaluate the KDE at the given points."""

        pts = np.atleast_2d(pts)
        d, m = pts.shape
        if d != self.d and d == 1 and m == self.d:
            pts = pts.T

        pts_orig = pts
        pts = np.copy(pts_orig)

        den = super(BoundedKDE, self).evaluate(pts)

        for i, (low, high, period) in enumerate(zip(self._low, self._high,
                                                    self._periodic)):
            if period:
                P = high - low
                
                pts[i, :] += P
                den += super(BoundedKDE, self).evaluate(pts)

                pts[i,:] -= 2.0*P
                den += super(BoundedKDE, self).evaluate(pts)

                pts[i,:] = pts_orig[i,:]

            else:
                if not np.isneginf(low):
                    pts[i,:] = 2.0*low - pts[i,:]
                    den += super(BoundedKDE, self).evaluate(pts)
                    pts[i,:] = pts_orig[i,:]

                if not np.isposinf(high):
                    pts[i,:] = 2.0*high - pts[i,:]
                    den += super(BoundedKDE, self).evaluate(pts)
                    pts[i,:] = pts_orig[i,:]

        return den

    __call__ = evaluate

    def quantile(self, pt):
        """Quantile of ``pt``, evaluated by a greedy algorithm.
        :param pt:
            The point at which the quantile value is to be computed.
        The quantile of ``pt`` is the fraction of points used to
        construct the KDE that have a lower KDE density than ``pt``."""

        return np.count_nonzero(self(self.dataset) < self(pt)) / self.n



# Special case
class BoundedKDE_2d(gaussian_kde):
    r"""Represents a two-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, xlow=None, xhigh=None, ylow=None, yhigh=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param xlow: The lower x domain boundary.

        :param xhigh: The upper x domain boundary.

        :param ylow: The lower y domain boundary.

        :param yhigh: The upper y domain boundary.
        """
        pts = np.atleast_2d(pts)

        assert pts.ndim == 2, 'Bounded_kde can only be two-dimensional'

        super(BoundedKDE_2d, self).__init__(pts.T, *args, **kwargs)

        self._xlow = xlow
        self._xhigh = xhigh
        self._ylow = ylow
        self._yhigh = yhigh

    @property
    def xlow(self):
        """The lower bound of the x domain."""
        return self._xlow

    @property
    def xhigh(self):
        """The upper bound of the x domain."""
        return self._xhigh

    @property
    def ylow(self):
        """The lower bound of the y domain."""
        return self._ylow

    @property
    def yhigh(self):
        """The upper bound of the y domain."""
        return self._yhigh

    def evaluate(self, pts):
        """Return an estimate of the density evaluated at the given
        points."""
        pts = np.atleast_2d(pts)
        assert pts.ndim == 2, 'points must be two-dimensional'

        x, y = pts.T
        pdf = super(BoundedKDE_2d, self).evaluate(pts.T)
        if self.xlow is not None:
            pdf += super(BoundedKDE_2d, self).evaluate([2*self.xlow - x, y])

        if self.xhigh is not None:
            pdf += super(BoundedKDE_2d, self).evaluate([2*self.xhigh - x, y])

        if self.ylow is not None:
            pdf += super(BoundedKDE_2d, self).evaluate([x, 2*self.ylow - y])

        if self.yhigh is not None:
            pdf += super(BoundedKDE_2d, self).evaluate([x, 2*self.yhigh - y])

        if self.xlow is not None:
            if self.ylow is not None:
                pdf += super(BoundedKDE_2d, self).evaluate([2*self.xlow - x, 2*self.ylow - y])

            if self.yhigh is not None:
                pdf += super(BoundedKDE_2d, self).evaluate([2*self.xlow - x, 2*self.yhigh - y])

        if self.xhigh is not None:
            if self.ylow is not None:
                pdf += super(BoundedKDE_2d, self).evaluate([2*self.xhigh - x, 2*self.ylow - y])
            if self.yhigh is not None:
                pdf += super(BoundedKDE_2d, self).evaluate([2*self.xhigh - x, 2*self.yhigh - y])

        return pdf

    def __call__(self, pts):
        pts = np.atleast_2d(pts)
        out_of_bounds = np.zeros(pts.shape[0], dtype='bool')

        if self.xlow is not None:
            out_of_bounds[pts[:, 0] < self.xlow] = True
        if self.xhigh is not None:
            out_of_bounds[pts[:, 0] > self.xhigh] = True
        if self.ylow is not None:
            out_of_bounds[pts[:, 1] < self.ylow] = True
        if self.yhigh is not None:
            out_of_bounds[pts[:, 1] > self.yhigh] = True

        results = self.evaluate(pts)
        results[out_of_bounds] = 0.
        return results
