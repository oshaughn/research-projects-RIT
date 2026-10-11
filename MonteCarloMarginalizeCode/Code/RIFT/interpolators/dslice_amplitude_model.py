"""Physical per-point distance model for distance-slice grids.

At fixed intrinsic parameters and fixed angles the log likelihood ratio is exactly quadratic in
x = u/u*, u = 1/d, through the origin:  lnL = R (2 f x - f^2 x^2) = R (1 - (1 - f x)^2), with f the
angle-dependent amplitude factor. Averaging exp(lnL) over f uniform on [f_min, 1] gives

    lnL(u) = C + log[ (1/(1-f_min)) * integral_{f_min}^{1} exp(R (1 - (1 - f x)^2)) df ]

in closed form (an erf difference). Properties: lnL -> C as d -> infinity (C = 0 for an exact
likelihood ratio), flat across the distance-inclination degenerate range x in [1, 1/f_min], Gaussian
fall-off outside it. Four parameters per intrinsic point: R (> 0), u* (> 0), f_min in (0, 1), C.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_ndtr

LOG_SQRT_PI_2 = 0.5 * np.log(np.pi) - np.log(2.0)


def log_model(u, R, ustar, fmin, C):
    """Vectorized lnL(u) of the averaged-amplitude model."""
    u = np.asarray(u, dtype=float)
    x = np.maximum(u / ustar, 1e-300)
    sR = np.sqrt(R)
    b = sR * (1 - x)            # f = 1 end
    a = sR * (1 - fmin * x)     # f = fmin end; a >= b
    # erf(a) - erf(b) = 2 [Phi(sqrt2 a) - Phi(sqrt2 b)] = 2 [Phi(-sqrt2 b) - Phi(-sqrt2 a)].
    # Take whichever form keeps both arguments in the lower tail, where log_ndtr is accurate.
    upper = a > 0
    hi_ = np.where(upper, log_ndtr(-np.sqrt(2) * b), log_ndtr(np.sqrt(2) * a))
    lo_ = np.where(upper, log_ndtr(-np.sqrt(2) * a), log_ndtr(np.sqrt(2) * b))
    log_diff = np.log(2.0) + hi_ + np.log1p(-np.exp(np.minimum(lo_ - hi_, -1e-300)))
    logI = R - np.log(x) + LOG_SQRT_PI_2 - np.log(sR) + log_diff - np.log(1 - fmin)
    return C + logI


def log_model_rounding_scale(u, R, ustar, fmin):
    """Size of float64 rounding in log_model(u, ...), in nats, up to an O(1) factor: eps times the
    magnitude of the terms that cancel. log1p(-exp(lo - hi)) amplifies the rounding of lo and hi by
    1/|lo - hi|, which is large where the two erf arguments nearly coincide (u/u* -> 0, the far tail
    of degenerate fits), and log(1 - f_min) amplifies the rounding of f_min by 1/(1 - f_min). Used to
    set the CPU/GPU comparison tolerance, not in the model."""
    u = np.asarray(u, dtype=float)
    x = np.maximum(u / ustar, 1e-300)
    sR = np.sqrt(R)
    b = sR * (1 - x)
    a = sR * (1 - fmin * x)
    upper = a > 0
    hi_ = np.where(upper, log_ndtr(-np.sqrt(2) * b), log_ndtr(np.sqrt(2) * a))
    lo_ = np.where(upper, log_ndtr(-np.sqrt(2) * a), log_ndtr(np.sqrt(2) * b))
    gap = np.maximum(np.abs(lo_ - hi_), 1e-300)
    amp = np.where(gap < 1, 1 / gap, 1.0)
    eps = np.finfo(float).eps
    return eps * (R + np.abs(np.log(x)) + (np.abs(hi_) + np.abs(lo_) + np.abs(a) + np.abs(b)) * (1 + amp)
                  + 1 / np.maximum(1 - fmin, 1e-300))


def _xp_of(a):
    try:
        import cupy
        return cupy.get_array_module(a)
    except ImportError:
        return np


def _log_ndtr(xp):
    if xp is np:
        return log_ndtr
    import cupyx.scipy.special
    return cupyx.scipy.special.log_ndtr


def log_model_xp(u, R, ustar, fmin, C, xp=None):
    """log_model on numpy or cupy (the array module of u unless xp is given); same formula."""
    xp = xp if xp is not None else _xp_of(u)
    lnd = _log_ndtr(xp)
    x = xp.maximum(u / ustar, 1e-300)
    sR = xp.sqrt(R)
    b = sR * (1 - x)
    a = sR * (1 - fmin * x)
    upper = a > 0
    s2 = np.sqrt(2)
    hi_ = xp.where(upper, lnd(-s2 * b), lnd(s2 * a))
    lo_ = xp.where(upper, lnd(-s2 * a), lnd(s2 * b))
    log_diff = np.log(2.0) + hi_ + xp.log1p(-xp.exp(xp.minimum(lo_ - hi_, -1e-300)))
    logI = R - xp.log(x) + LOG_SQRT_PI_2 - xp.log(sR) + log_diff - xp.log(1 - fmin)
    return C + logI


def _robust_rho(z, loss, xp):
    """scipy least_squares robust losses: rho(z) and rho'(z), z = (r/f_scale)^2."""
    if loss == "soft_l1":
        t = 1 + z
        return 2 * (xp.sqrt(t) - 1), 1 / xp.sqrt(t)
    if loss == "cauchy":
        return xp.log1p(z), 1 / (1 + z)
    if loss == "huber":
        return xp.where(z <= 1, z, 2 * xp.sqrt(z) - 1), 1 / xp.sqrt(xp.maximum(z, 1.0))
    raise ValueError("unknown loss %r" % loss)


def fit_points_batched(U, Yv, W, fscale, fix_C=False, loss="linear", max_iter=200, xp=np, fmin_fixed=None):
    """Fit every point at once. U, Yv, W: (G, N) padded 1/d, lnL and weights (W = 0 marks padding);
    fscale: (G,) robust-loss scale in residual units (unused for loss='linear').

    Same objective and parameterization as fit_point, from its two starting points and two more. The
    optimizer is a batched Levenberg-Marquardt with forward-difference Jacobians (scipy trf with jac='2-point' is the CPU
    reference); robust losses use iteratively reweighted least squares, whose fixed points are the
    stationary points of scipy's robust cost. fmin_fixed holds f_min at one value for every point (an
    event-level inclination degeneracy) and fits the remaining parameters. Returns (params (G,4), NaN rows
    for failures; rms (G,))."""
    G, N = U.shape
    mask = W > 0
    nslc = mask.sum(axis=1)
    Uf = xp.where(mask, U, 1.0)
    srt = xp.sort(xp.where(mask, U, xp.inf), axis=1)

    def pct(q):     # np.percentile(u, q), linear interpolation, over each point's own slices
        pos = q / 100.0 * (nslc - 1)
        i0 = xp.floor(pos).astype(xp.int64)
        i1 = xp.minimum(i0 + 1, nslc - 1)
        v0 = xp.take_along_axis(srt, i0[:, None], axis=1)[:, 0]
        v1 = xp.take_along_axis(srt, i1[:, None], axis=1)[:, 0]
        return v0 + (pos - i0) * (v1 - v0)
    u10, u90 = pct(10.0), pct(90.0)
    ymax = xp.max(xp.where(mask, Yv, -xp.inf), axis=1)
    sw = xp.sqrt(W)
    free_f = fmin_fixed is None
    npar = 2 + int(free_f) + int(not fix_C)
    jC = 2 + int(free_f)          # column of C when it is fitted
    eye = xp.eye(npar)[None]

    def unpack(p):
        R, us = xp.exp(p[:, 0:1]), xp.exp(p[:, 1:2])
        fm = 1 / (1 + xp.exp(-p[:, 2:3])) if free_f else fmin_fixed
        C = 0.0 if fix_C else p[:, jC:jC + 1]
        return R, us, fm, C

    def resid(p, ix):
        R, us, fm, C = unpack(p)
        return xp.where(mask[ix], sw[ix] * (log_model_xp(Uf[ix], R, us, fm, C, xp=xp) - Yv[ix]), 0.0)

    def cost_and_w(r, ix):
        if loss == "linear":
            return 0.5 * xp.sum(r * r, axis=1), None
        f = fscale[ix]
        rho, drho = _robust_rho((r / f[:, None]) ** 2, loss, xp)
        return 0.5 * f ** 2 * xp.sum(xp.where(mask[ix], rho, 0.0), axis=1), xp.where(mask[ix], drho, 0.0)

    # fit_point's two starts, then the same two with f_min0 halved: cheap here, and a batched
    # Levenberg-Marquardt is less robust to a poor start than scipy's trf
    starts = [(1.0, 1.0, 1.0), (0.8, 0.9, 1.0), (1.0, 1.0, 0.5), (0.8, 0.9, 0.5)]
    best_p = best_c = None
    with np.errstate(all="ignore"):
        for f_fac, u_fac, f_half in starts:
            fmin0 = xp.minimum(0.9, xp.maximum(0.05, u10 / u90 * f_fac)) * f_half
            cols = [xp.log(xp.maximum(ymax, 1.0)), xp.log(u10 * u_fac)] + ([xp.log(fmin0 / (1 - fmin0))] if free_f else [])
            P = xp.stack(cols + ([] if fix_C else [xp.zeros(G)]), axis=1)
            allix = xp.arange(G)
            R_ = resid(P, allix)
            Cst, RW = cost_and_w(R_, allix)
            ix = allix[xp.isfinite(Cst)]
            p, r, c = P[ix], R_[ix], Cst[ix]
            rw = None if RW is None else RW[ix]
            lam = xp.full(len(ix), 1e-3)
            for it in range(max_iter):
                h = 1.49e-8 * xp.maximum(1.0, xp.abs(p))
                J = xp.stack([(resid(p + h[:, j:j + 1] * eye[0, j], ix) - r) / h[:, j:j + 1] for j in range(npar)], axis=2)
                Jw = J if rw is None else J * rw[:, :, None]
                A = xp.einsum('gni,gnj->gij', Jw, J)
                g = xp.einsum('gni,gn->gi', Jw, r)
                dA = xp.maximum(xp.einsum('gii->gi', A), 1e-300)
                Ad = A + lam[:, None, None] * dA[:, :, None] * eye
                good = xp.all(xp.isfinite(Ad), axis=(1, 2)) & xp.all(xp.isfinite(g), axis=1)
                step = -xp.linalg.solve(xp.where(good[:, None, None], Ad, eye), xp.where(good[:, None], g, 0.0)[:, :, None])[:, :, 0]
                p_new = p + step
                r_new = resid(p_new, ix)
                c_new, rw_new = cost_and_w(r_new, ix)
                better = good & xp.isfinite(c_new) & (c_new <= c)
                # scipy's tests: ftol on an accepted step whose reduction matches the model's
                # prediction (ratio > 0.25), xtol on an accepted step, gtol on the gradient. A
                # nonfinite system is not convergence: it raises the damping and retries.
                pred = -xp.einsum('gi,gi->g', g, step) - 0.5 * xp.einsum('gi,gij,gj->g', step, A, step)
                ratio = (c - c_new) / xp.where(pred > 0, pred, xp.inf)
                conv = (better & ((c - c_new) < 1e-8 * c) & (ratio > 0.25)) \
                    | (better & (xp.linalg.norm(step, axis=1) < 1e-8 * (1e-8 + xp.linalg.norm(p, axis=1)))) \
                    | (good & (xp.max(xp.abs(g), axis=1) < 1e-8))
                p = xp.where(better[:, None], p_new, p)
                r = xp.where(better[:, None], r_new, r)
                if rw is not None:
                    rw = xp.where(better[:, None], rw_new, rw)
                c = xp.where(better, c_new, c)
                lam = xp.where(better, xp.maximum(lam / 3.0, 1e-12), lam * 4.0)
                done = conv | (lam >= 1e12)
                if bool(xp.any(done)):
                    P[ix[done]], Cst[ix[done]] = p[done], c[done]
                    keep = ~done
                    ix, p, r, c, lam = ix[keep], p[keep], r[keep], c[keep], lam[keep]
                    if rw is not None:
                        rw = rw[keep]
                    if len(ix) == 0:
                        break
            if len(ix):
                P[ix], Cst[ix] = p, c
            if best_p is None:
                best_p, best_c = P, Cst
            else:
                take = xp.isfinite(Cst) & ~(Cst >= best_c)
                best_p, best_c = xp.where(take[:, None], P, best_p), xp.where(take, Cst, best_c)
        p = best_p
        R_, us_, fm_, C_ = unpack(p)
        params = xp.stack([R_[:, 0], us_[:, 0], (fm_[:, 0] if free_f else xp.full(G, float(fmin_fixed))),
                           (xp.zeros(G) if fix_C else C_[:, 0])], axis=1)
        fail = ~xp.all(xp.isfinite(params), axis=1) | ~xp.isfinite(best_c)
        params = xp.where(fail[:, None], xp.nan, params)
        res_ = xp.where(mask, log_model_xp(Uf, params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4], xp=xp) - Yv, 0.0)
        if loss == "linear":
            rms = xp.sqrt(xp.sum(W * res_ ** 2, axis=1) / xp.sum(W, axis=1))
        else:   # 1.4826 * median |residual| over the point's slices
            a = xp.sort(xp.where(mask, xp.abs(res_), xp.inf), axis=1)
            lo = xp.take_along_axis(a, ((nslc - 1) // 2)[:, None], axis=1)[:, 0]
            hi = xp.take_along_axis(a, (nslc // 2)[:, None], axis=1)[:, 0]
            rms = 1.4826 * 0.5 * (lo + hi)
    return params, xp.where(fail, xp.inf, rms)


def fit_all_points_batched(key, u, y, sig, min_slices=5, fix_C=False, loss="linear", f_scale=1.0, xp=np,
                           block=None, fmin_fixed=None):
    """fit_all_points with the points fit in batches by fit_points_batched on numpy or cupy."""
    block = block or (8192 if xp is np else 65536)
    uk, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    w = 1.0 / np.maximum(sig, 1e-3) ** 2
    params = np.full((len(uk), 4), np.nan)
    rms = np.full(len(uk), np.nan)
    elig = np.flatnonzero(counts >= min_slices)
    for b0 in range(0, len(elig), block):
        gs = elig[b0:b0 + block]
        cnt = counts[gs]
        pos = np.arange(int(cnt.max()))[None, :]
        live = pos < cnt[:, None]
        idx = order[starts[gs][:, None] + np.minimum(pos, cnt[:, None] - 1)]
        Ub, Yb, Wb = (np.where(live, a[idx], 0.0) for a in (u, y, w))
        fsc = f_scale * np.sqrt(np.nanmedian(np.where(live, Wb, np.nan), axis=1))
        P, E = fit_points_batched(*(xp.asarray(a) for a in (Ub, Yb, Wb, fsc)), fix_C=fix_C, loss=loss, xp=xp,
                                  fmin_fixed=fmin_fixed)
        params[gs], rms[gs] = (a.get() if hasattr(a, 'get') else a for a in (P, E))
    return uk, params, rms, counts


def fit_point(u, y, w, n_restart=2, fix_C=False, loss="linear", f_scale=1.0, fmin_fixed=None):
    """Weighted least-squares fit of (R, u*, f_min, C) to one point's slices. Returns (params, rms).

    fix_C pins C = 0, the exact d -> infinity limit; otherwise C absorbs model mismatch, and over a
    narrow slice range it trades off against f_min, which changes the extrapolation.
    loss/f_scale are passed to scipy least_squares on residuals in nats (weights still apply); a robust
    loss such as 'cauchy' with f_scale ~ 1 discounts slices where the integrator missed the peak."""
    order = np.argsort(u)
    u, y, w = u[order], y[order], w[order]
    ymax = y.max()
    u10, u90 = np.percentile(u, [10, 90])
    best = None
    for k in range(n_restart):
        fmin0 = min(0.9, max(0.05, u10 / u90 * (0.8 if k else 1.0)))
        p0 = np.array([np.log(max(ymax, 1.0)), np.log(u10 * (0.9 if k else 1.0)), np.log(fmin0 / (1 - fmin0)), 0.0])

        if fix_C:
            p0 = p0[:3]
        if fmin_fixed is not None:
            p0 = np.delete(p0, 2)

        def unpack(p):
            if fmin_fixed is not None:
                p = np.insert(p, 2, np.log(fmin_fixed / (1 - fmin_fixed)))
            return np.exp(p[0]), np.exp(p[1]), 1 / (1 + np.exp(-p[2])), (0.0 if fix_C else p[3])

        def resid(p):
            R, us, fm, C = unpack(p)
            return np.sqrt(w) * (log_model(u, R, us, fm, C) - y)

        try:
            r = least_squares(resid, p0, method="trf", max_nfev=200, loss=loss,
                              f_scale=f_scale * float(np.sqrt(np.median(w))))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None or not np.all(np.isfinite(best.x)):
        return None, np.inf
    params = np.array(unpack(best.x))
    res_ = log_model(u, *params) - y
    rms = float(np.sqrt(np.sum(w * res_ ** 2) / np.sum(w))) if loss == "linear" else float(1.4826 * np.median(np.abs(res_)))
    return params, rms


def _fit_point_chunk(slices, kw):
    out = []
    for u_, y_, w_ in slices:
        out.append(fit_point(u_, y_, w_, **kw))
    return out


def fit_all_points(key, u, y, sig, min_slices=5, fix_C=False, loss="linear", f_scale=1.0, fmin_fixed=None,
                   n_jobs=1):
    """Fit every unique intrinsic row of `key`. Returns (unique keys, params (n,4), rms, nslices).

    n_jobs > 1 spreads the per-point fits over worker processes. fit_point is deterministic, so the
    result equals the serial one."""
    uk, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(counts)]
    w = 1.0 / np.maximum(sig, 1e-3) ** 2
    params = np.full((len(uk), 4), np.nan)
    rms = np.full(len(uk), np.nan)
    kw = dict(fix_C=fix_C, loss=loss, f_scale=f_scale, fmin_fixed=fmin_fixed)
    n_jobs = 1 if n_jobs is None else int(n_jobs)
    if n_jobs < 1:
        raise ValueError("fit_all_points: n_jobs must be a positive number of processes")
    elig = [g for g in range(len(uk)) if counts[g] >= min_slices]
    n_jobs = min(n_jobs, max(1, len(elig)))
    if n_jobs > 1:
        from joblib import Parallel, delayed
        chunks = np.array_split(np.asarray(elig, dtype=int), 8 * n_jobs)
        jobs = [[(u[order[starts[g]:starts[g + 1]]], y[order[starts[g]:starts[g + 1]]], w[order[starts[g]:starts[g + 1]]])
                 for g in c] for c in chunks]
        res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_point_chunk)(j, kw) for j in jobs)
        for c, rr in zip(chunks, res):
            for g, (p, e) in zip(c, rr):
                if p is not None:
                    params[g], rms[g] = p, e
        return uk, params, rms, counts
    for g in range(len(uk)):
        r = order[starts[g]:starts[g + 1]]
        if len(r) < min_slices:
            continue
        p, e = fit_point(u[r], y[r], w[r], **kw)
        if p is not None:
            params[g], rms[g] = p, e
    return uk, params, rms, counts


class DistanceAmplitudeModel:
    """lnL(x, d) = log_model(1/d; R(x), u*(x), f_min(x), C=0) with the three fields interpolated over
    the intrinsic fit coordinates x by an ExtraTrees regressor fit to the per-point fits.

    Fields are interpolated as log R, log u*, logit f_min. Points whose fit failed are left out."""

    def __init__(self, dist_index, n_estimators=100, n_jobs=-1, min_slices=5, max_overshoot=1.0, mass_index=None,
                 prior=None, d_range=None, n_dgrid=256, loss="linear", f_scale=1.0, point_fit="scipy", xp=None,
                 point_fit_jobs=1, random_state=None):
        """mass_index: column (in the full fit coordinates) holding a mass M. If given, the distance
        scale is interpolated as log(u* M): the horizon distance scales with mass, so u* M varies far
        less across the grid than u* itself.

        point_fit: 'scipy' (one least_squares call per point; point_fit_jobs worker processes) or
        'batched' (fit_all_points_batched on array module xp). batched is not for production: it ends
        in a worse local minimum than scipy on ~0.5% of points, which degraded the distance posterior
        on S240501an and S240527en (RIFT_roboto_paper analyses/distance_gp_interp)."""
        self.dist_index = int(dist_index)
        self.mass_index = None if mass_index is None else int(mass_index)
        # prior + d_range switch on the marginal/conditional decomposition:
        #   lnL(x,d) = M(x) + [log_model(1/d; theta(x)) - N(theta(x))],
        # M = log int exp(log_model) prior dd over d_range per point, interpolated directly, N the same
        # integral evaluated at the interpolated shape. The intrinsic marginal is then M(x) alone, so
        # interpolation error in the distance-shape fields cannot move intrinsic weights.
        self.prior, self.d_range, self.n_dgrid = prior, d_range, int(n_dgrid)
        self.loss, self.f_scale = loss, float(f_scale)
        self.max_overshoot = float(max_overshoot)
        self.n_estimators, self.n_jobs, self.min_slices = n_estimators, n_jobs, min_slices
        if point_fit not in ("scipy", "batched"):
            raise ValueError("point_fit must be 'scipy' or 'batched'")
        self.point_fit, self._fit_xp, self.point_fit_jobs = point_fit, xp, int(point_fit_jobs)
        self.random_state = random_state     # field forest (ExtraTrees); None = unseeded

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_fit_xp", None)
        state.pop("_device", None)
        return state

    @staticmethod
    def _to_fields(P):
        return np.column_stack([np.log(P[:, 0]), np.log(P[:, 1]), np.log(P[:, 2] / (1 - P[:, 2]))])

    @staticmethod
    def _from_fields(F):
        return np.exp(F[:, 0]), np.exp(F[:, 1]), 1 / (1 + np.exp(-F[:, 2]))

    def fit(self, x, y, y_errors):
        from sklearn.ensemble import ExtraTreesRegressor
        self._device = None     # a cached device copy belongs to the previous fit
        x = np.asarray(x, dtype=float)
        xi = np.delete(x, self.dist_index, axis=1)
        key = np.round(xi, 10)
        args = (key, 1.0 / x[:, self.dist_index], np.asarray(y, dtype=float), np.asarray(y_errors, dtype=float),
                self.min_slices)
        if getattr(self, "point_fit", "scipy") == "batched":
            uk, P, rms, counts = fit_all_points_batched(*args, fix_C=True, loss=self.loss, f_scale=self.f_scale,
                                                        xp=self._fit_xp if self._fit_xp is not None else np)
        else:
            uk, P, rms, counts = fit_all_points(*args, fix_C=True, loss=self.loss, f_scale=self.f_scale,
                                                n_jobs=getattr(self, "point_fit_jobs", 1))
        # a fit with f_min rounded to 0 or 1 has a nonfinite field (logit); count it as failed
        with np.errstate(all="ignore"):
            good = np.all(np.isfinite(P), axis=1) & np.all(np.isfinite(self._to_fields(P)), axis=1)
        # Guard: a point whose slices do not resolve its flat top (e.g. all slices at nearly one d)
        # can fit a peak far above anything it measured; the sampler then piles onto that spike.
        # Drop points whose fitted maximum over d exceeds their highest slice by > max_overshoot.
        _, inv = np.unique(key, axis=0, return_inverse=True)
        ymax = np.full(len(uk), -np.inf)
        np.maximum.at(ymax, inv.reshape(-1), np.asarray(y, dtype=float))
        # the fitted peak sits in x = u/u* in [1, 1/f_min]: scan x from 1e-3 to 10/f_min for each point, so
        # the guard works at any distance scale (a fixed u grid misses points nearer than its range)
        t = np.linspace(0.0, 1.0, 2000)[None, :]
        over = np.full(len(uk), np.inf)
        kg = np.flatnonzero(good)
        for s0 in range(0, len(kg), 2048):
            k = kg[s0:s0 + 2048]
            lx = np.log(1e-3) + t * (np.log(10.0 / P[k, 2, None]) - np.log(1e-3))
            over[k] = log_model(P[k, 1, None] * np.exp(lx), *(P[k, j, None] for j in range(4))).max(axis=1) - ymax[k]
        n_over = int(np.sum(good & (over > self.max_overshoot)))
        good &= over <= self.max_overshoot
        self.report = dict(points=int(len(uk)), points_fit=int(good.sum()), dropped_overshoot=n_over,
                           per_point_rms_median=float(np.nanmedian(rms)), slices_median=float(np.median(counts)))
        self.rf = ExtraTreesRegressor(n_estimators=self.n_estimators, n_jobs=self.n_jobs,
                                      random_state=getattr(self, "random_state", None))
        F = self._to_fields(P[good])
        if self.mass_index is not None:
            F[:, 1] += np.log(self._mass(uk[good]))
        if self.prior is not None:
            # normalization grid: start at n_dgrid nodes and refine (x8) until M agrees with the next
            # refinement to 0.01 nats on every fitted point; narrow (loud or nearby) distance peaks need it
            prior_fn, n = self.prior, self.n_dgrid
            def grid(m):
                g = np.linspace(self.d_range[0], self.d_range[1], m)
                return g, np.log(np.maximum(np.asarray(prior_fn(g), dtype=float), 1e-300)) + np.log(np.gradient(g))
            M = None
            while True:
                g1, w1 = grid(n); g2, w2 = grid(8 * n)
                M1 = self._lognorm(P[good, 0], P[good, 1], P[good, 2], dgrid=g1, lw=w1)
                M2 = self._lognorm(P[good, 0], P[good, 1], P[good, 2], dgrid=g2, lw=w2)
                err = float(np.max(np.abs(M1 - M2))) if len(M1) else 0.0
                if err <= 0.01 or n >= 32768:
                    break
                n *= 8
            self._dgrid, self._lw, self.n_dgrid = g1, w1, n
            M = M1
            self.report.update(dgrid_nodes=int(n), dgrid_check_nats=err)
            F = np.column_stack([F, M])
            self.prior = "tabulated"     # keep only the tabulated weights: closures do not pickle
        self.rf.fit(uk[good], F)
        return self

    def _lognorm(self, R, us, fm, xp=np, dgrid=None, lw=None):
        """log int exp(log_model(1/d)) prior(d) dd over the d grid, vectorized over points."""
        dgrid = self._dgrid if dgrid is None else dgrid
        lw = self._lw if lw is None else lw
        lm = log_model if xp is np else (lambda *a: log_model_xp(*a, xp=xp))
        out = xp.empty(len(R))
        step = 4096 if xp is np else 65536
        for s0 in range(0, len(R), step):
            sl = slice(s0, s0 + step)
            L = lm(1.0 / dgrid[None, :], R[sl, None], us[sl, None], fm[sl, None], 0.0) + lw[None, :]
            m = L.max(axis=1)
            out[sl] = m + xp.log(xp.exp(L - m[:, None]).sum(axis=1))
        return out

    def _mass(self, xi):
        # mass column index refers to the full fit coordinates; xi has the distance column removed
        j = self.mass_index - (1 if self.mass_index > self.dist_index else 0)
        return xi[:, j]

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        xi = np.delete(x, self.dist_index, axis=1)
        F = self.rf.predict(xi)
        if self.mass_index is not None:
            F[:, 1] -= np.log(self._mass(xi))
        R, us, fm = self._from_fields(F[:, :3])
        ll = log_model(1.0 / x[:, self.dist_index], R, us, fm, 0.0)
        if self.prior is not None:
            ll = F[:, 3] + ll - self._lognorm(R, us, fm)
        return ll

    def check_device(self, x, field_rtol=1e-10, atol=1e-8, k_round=100.0):
        """Compare predict_device with predict on rows x. Returns (ok, report).

        The field forest must agree to roundoff (field_rtol). lnL rows must agree within
        atol + k_round * log_model_rounding_scale: on degenerate fits of weak points (R up to 1e10,
        f_min -> 0, u/u* ~ 1e-13) log_model cancels large terms and either path carries ~1e-3 nats of
        rounding while the value is near 0, so a flat tolerance flags rows where nothing is wrong."""
        import cupy as cp
        x = np.asarray(x, dtype=float)
        ll_dev = cp.asnumpy(self.predict_device(x))
        ll = self.predict(x)
        xi = np.delete(x, self.dist_index, axis=1)
        F = self.rf.predict(xi)
        F_dev = cp.asnumpy(self._device["forest"].predict(xi))
        field_err = float(np.max(np.abs(F_dev - F) / (1 + np.abs(F))))
        Fm = F.copy()
        if self.mass_index is not None:
            Fm[:, 1] -= np.log(self._mass(xi))
        R, us, fm = self._from_fields(Fm[:, :3])
        scale = log_model_rounding_scale(1.0 / x[:, self.dist_index], R, us, fm)
        if self.prior is not None:      # the normalization integral carries the same rounding, at its worst node
            scale = scale + np.max(log_model_rounding_scale(1.0 / self._dgrid[None, :], R[:, None], us[:, None],
                                                            fm[:, None]), axis=1)
        ratio = np.abs(ll_dev - ll) / (atol + k_round * scale)
        w = int(np.argmax(ratio))
        report = dict(rows=len(x), field_max_rel=field_err, lnl_max_abs=float(np.max(np.abs(ll_dev - ll))),
                      lnl_max_scaled=float(ratio[w]), worst_row=dict(R=float(R[w]), x=float(1.0 / x[w, self.dist_index] / us[w]),
                                                                    fmin=float(fm[w]), rounding_scale=float(scale[w])))
        return bool(field_err < field_rtol and np.max(ratio) <= 1.0), report

    def predict_device(self, x):
        """predict on the GPU: x a cupy (or numpy) array, returns cupy. Same formulas as predict; the
        field forest is evaluated by RIFT.interpolators.cupy_forest.CupyForest."""
        import cupy as cp
        if getattr(self, "_device", None) is None:
            from RIFT.interpolators.cupy_forest import CupyForest
            dev = dict(forest=CupyForest(self.rf))
            if self.prior is not None:
                dev.update(dgrid=cp.asarray(self._dgrid), lw=cp.asarray(self._lw))
            self._device = dev
        dev = self._device
        x = cp.asarray(x, dtype=cp.float64)
        keep = [j for j in range(x.shape[1]) if j != self.dist_index]
        xi = x[:, keep]
        F = dev["forest"].predict(xi)
        if self.mass_index is not None:
            F[:, 1] -= cp.log(self._mass(xi))
        R, us, fm = cp.exp(F[:, 0]), cp.exp(F[:, 1]), 1 / (1 + cp.exp(-F[:, 2]))
        ll = log_model_xp(1.0 / x[:, self.dist_index], R, us, fm, 0.0, xp=cp)
        if self.prior is not None:
            ll = F[:, 3] + ll - self._lognorm(R, us, fm, xp=cp, dgrid=dev["dgrid"], lw=dev["lw"])
        return ll
