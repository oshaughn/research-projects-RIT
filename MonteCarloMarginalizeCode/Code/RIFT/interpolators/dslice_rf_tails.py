"""Physical distance continuation for a fitted lnL(x, d) beyond each grid point's exported slices.

A tree fit is piecewise constant outside its training envelope, so beyond each intrinsic point's
nearest and farthest slice it holds lnL flat, and the volumetric distance prior turns that into
spurious mass at small or large d. This wrapper keeps the base fit inside the slice range of the
nearest grid point and, outside it, continues from the base fit's edge value with the shape of the
averaged-amplitude model (dslice_amplitude_model.log_model, C = 0) fitted to that point's slices:

    lnL(x, d) = base(x, d_edge) + min(0, A(1/d; theta_j) - A(1/d_edge; theta_j))      for d outside [d_min_j, d_max_j]

with j the nearest grid point in standardized intrinsic coordinates. f_min may be shared by the whole
event (the median of free per-point fits): the inclination degeneracy it encodes is set by the network.

Cost: fit() does one scipy least_squares per grid point (twice with a shared f_min); on a 24k-point,
1.2M-row grid that is ~220 s on one core, ~30 s with n_jobs=8. Each call adds a host-side nearest-point
lookup, ~1 s per 1e5 query rows.
"""
import numpy as np
from scipy.spatial import cKDTree

from RIFT.interpolators.dslice_amplitude_model import fit_all_points, fit_all_points_batched, log_model


class RFDistanceTails:
    def __init__(self, base_fit, dist_index, sides="both", shared_fmin=True, xp=np, point_fit="scipy",
                 n_jobs=1, coord_names=None):
        """coord_names: names of the fit coordinates, needed only to spot 'inv_dist'. It is a
        reparametrisation of the same axis as 'dist', so it varies across one point's slices: it must
        stay out of the intrinsic key (or every slice would look like its own grid point, leaving no
        point with enough slices to fit) and must move with 'dist' when the base fit is read at the
        slice edge."""
        if sides not in ("near", "far", "both"):
            raise ValueError("sides must be near, far or both")
        if point_fit not in ("scipy", "batched"):
            raise ValueError("point_fit must be scipy or batched")
        self.base_fit, self.dist_index, self.sides = base_fit, int(dist_index), sides
        self.shared_fmin, self.xp = bool(shared_fmin), xp
        self.point_fit, self.n_jobs = point_fit, int(n_jobs)
        self.coord_names = None if coord_names is None else list(coord_names)
        self.inv_dist_index = (self.coord_names.index("inv_dist")
                               if self.coord_names is not None and "inv_dist" in self.coord_names else None)
        self.drop = [self.dist_index] if self.inv_dist_index is None else [self.dist_index, self.inv_dist_index]

    def _fit_points(self, args, fmin_fixed=None):
        # scipy (default): one least_squares per point. batched: all points at once on numpy/cupy; it
        # reaches a higher cost than scipy on a fraction of points, so it is an option, not the default.
        if self.point_fit == "batched":
            return fit_all_points_batched(*args, fix_C=True, xp=self.xp, fmin_fixed=fmin_fixed)
        return fit_all_points(*args, fix_C=True, fmin_fixed=fmin_fixed, n_jobs=self.n_jobs)

    def fit(self, x, y_unshifted, y_errors):
        """x: training rows in fit coordinates; y_unshifted: lnL with no shift (the model's d -> inf
        limit is 0); y_errors: per-row sigma."""
        x = np.asarray(x, dtype=float)
        if self.coord_names is not None and len(self.coord_names) != x.shape[1]:
            # the distance columns are found by name, so a mismatched list would drop the wrong ones
            raise ValueError("coord_names has %d entries but the training rows have %d columns"
                             % (len(self.coord_names), x.shape[1]))
        y_unshifted = np.asarray(y_unshifted, dtype=float)
        y_errors = np.asarray(y_errors, dtype=float)
        ok = np.all(np.isfinite(x), axis=1) & np.isfinite(y_unshifted) & np.isfinite(y_errors) \
            & (x[:, self.dist_index] > 0)          # the base fit may accept rows the tail model cannot
        x, y_unshifted, y_errors = x[ok], y_unshifted[ok], y_errors[ok]
        d = x[:, self.dist_index]
        key = np.round(np.delete(x, self.drop, axis=1), 10)
        # a fit in distance alone leaves no intrinsic coordinate: the whole grid is one fixed point, so
        # group it by a single constant column rather than by zero-width rows
        self.n_int = key.shape[1]
        if not self.n_int:
            key = np.zeros((len(key), 1))
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        self.dmin = np.full(len(uk), np.inf)
        self.dmax = np.zeros(len(uk))
        np.minimum.at(self.dmin, inv, d)
        np.maximum.at(self.dmax, inv, d)
        args = (key, 1.0 / d, y_unshifted, y_errors)
        uk2, P, _, _ = self._fit_points(args)
        fmin = None
        if self.shared_fmin:
            fmin = float(np.nanmedian(P[:, 2]))
            uk2, P, _, _ = self._fit_points(args, fmin_fixed=fmin)
        assert np.array_equal(uk2, uk)
        self.P = P
        self.good = np.all(np.isfinite(P), axis=1)
        self.mu, self.sd = uk.mean(0), uk.std(0)
        self.sd[self.sd == 0] = 1.0
        self.tree = cKDTree((uk - self.mu) / self.sd) if self.n_int else None
        self.report = dict(points=int(len(uk)), points_fit=int(self.good.sum()), rows_dropped=int((~ok).sum()), sides=self.sides,
                           fmin_shared=fmin, point_fit=self.point_fit, intrinsic_dim=int(self.n_int))
        return self

    def __call__(self, x_in):
        # keep the caller's array type (the GPU driver passes and expects cupy); neighbour lookup and
        # the continuation run on the host
        dev = hasattr(x_in, "get")
        xmod = __import__("cupy") if dev else np
        x = x_in.get() if dev else np.asarray(x_in, dtype=float)
        to_host = (lambda a: a.get()) if dev else (lambda a: np.asarray(a, dtype=float))
        fin = np.all(np.isfinite(x), axis=1)
        xf = x[fin]
        d = xf[:, self.dist_index]
        if self.tree is None:
            j = np.zeros(len(xf), dtype=int)                        # distance-only fit: the one point
        else:
            _, j = self.tree.query((np.delete(xf, self.drop, axis=1) - self.mu) / self.sd)
        g = self.good[j]
        lo = g & (d < self.dmin[j]) if self.sides in ("near", "both") else np.zeros(len(d), bool)
        hi = g & (d > self.dmax[j]) if self.sides in ("far", "both") else np.zeros(len(d), bool)
        d_edge = np.where(lo, self.dmin[j], np.where(hi, self.dmax[j], d))
        # one base call on the whole batch: non-finite rows go through the base fit's own guard (its fill
        # value), never as a batch of their own, which an sklearn forest rejects once its guard empties it
        xe = x.copy()
        xe[fin, self.dist_index] = d_edge
        if self.inv_dist_index is not None:
            # inv_dist is the same axis: reading the base fit at d_edge means reading it at 1/d_edge.
            # d_edge <= 0 has no 1/d, so those rows keep the caller's value and the base fit's guard.
            iv = x[fin, self.inv_dist_index].copy()
            iv[d_edge > 0] = 1.0 / d_edge[d_edge > 0]
            xe[fin, self.inv_dist_index] = iv
        out = to_host(self.base_fit(xmod.asarray(xe))).astype(float, copy=True)
        val = out[fin]
        off = lo | hi
        if np.any(off):
            Pj = self.P[j[off]]
            # d <= 0 has no 1/d; the model's u -> inf limit is lnL -> -inf, so give it zero likelihood
            pos = d[off] > 0
            # Never rise above the edge value: where a point's fitted peak lies beyond its slices the
            # shape would climb away from the data, and the sampler piles onto that unmeasured spike.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                step = log_model(1.0 / np.where(pos, d[off], 1.0), Pj[:, 0], Pj[:, 1], Pj[:, 2], 0.0) \
                    - log_model(1.0 / d_edge[off], Pj[:, 0], Pj[:, 1], Pj[:, 2], 0.0)
            step = np.where(pos & ~np.isnan(step), step, -np.inf)
            val[off] = val[off] + np.minimum(step, 0.0)
        out[fin] = val
        return xmod.asarray(out)
