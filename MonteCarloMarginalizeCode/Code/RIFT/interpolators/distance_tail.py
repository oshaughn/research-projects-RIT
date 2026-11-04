#!/usr/bin/env python
"""distance_tail_fit.py -- make the CIP fit fall off in distance beyond the exported support.

THE DEFECT.  An ExtraTrees/RF ensemble is piecewise constant outside its training envelope.  The
.dslice export trains it on ~50 discrete distances per intrinsic point, so beyond that point's own
outermost slice the ensemble returns the EDGE value of lnL forever instead of letting it decay.
Measured directly (tools/test_distance_extrapolation.py): the held-out lnL bias grows with how far
past the training edge you ask, +0.00 / +0.14 / +0.84 nats at reach <5% / 5-15% / >15%.

WHY IT MATTERS SO MUCH.  The recovered distance posterior is exp(lnL) * pi(d), and pi(d) is
volumetric -- it GROWS like d^2.  Hold lnL flat and the integrand grows without bound out to
whatever --d-max the CIP range allows, so probability that should have died survives, and the
posterior comes out ~18% too wide in every quantile span including the core, median untouched.
Confirmed at Spearman rho=+0.952 (p=4e-5) between an event's off-support sample fraction and its
width excess.  Confining d to the exported support removes 18 of those 18 points.

THIS IS NOT A COORDINATE PROBLEM.  Fitting in 1/d instead of d was tried and made things WORSE
(off-support mass 0.098 -> 0.194).  A tree extrapolates flat in whatever coordinate you give it;
changing coordinates only moves where "flat" lands.  What is missing is a BOUNDARY CONDITION.

--------------------------------------------------------------------------------------------------
WHAT WAS TRIED FIRST AND REJECTED, because it looks right on paper and fails on the data.

For fixed intrinsic parameters AND fixed extrinsic angles, RIFT's log likelihood ratio is exactly
quadratic in x = 1/d through the origin, lnL = a x - b x^2/2, since the signal enters as h/d.  That
suggests fitting each slice with a global through-the-origin polynomial: exact at d -> infinity by
construction, which is precisely where the tree fails.

It does not work, and the reason is instructive.  The export marginalises over the extrinsic angles,
and the distance-inclination degeneracy makes the MARGINALISED lnL nearly FLAT across the whole
exported support -- on S240615ea, 14.3 down to 13.1 while d runs 1085 -> 4909 Mpc.  A low-order
polynomial through the origin cannot be both flat over the support and zero at x=0: the per-slice
residual is 1.7 nats at order 2 and 0.55 at order 3, against a per-slice noise of ~0.13.  Forcing it
anyway drives the conditional width 58% LOW.  `polyfit_slice` below is retained as the diagnostic
that measures this, not as the fix.

The flatness is also the reason the defect is severe rather than subtle: the fit reaches the edge of
its support with lnL still near its peak, so "hold the last value" is not a small error.

--------------------------------------------------------------------------------------------------
THE FIX.  Keep the base fit -- it is right ON the support, which the `nnexp` rung established
independently -- and give it a tail.  Write x = 1/d and u = x/x_edge, and continue each slice past
its outermost exported distance with the exact single-angle form THROUGH THE ORIGIN, matched to the
value and the slope the data actually has at that edge:

    lnL(u) / lnL_edge  =  (2 - s) u  +  (s - 1) u^2,        s = x_edge * dlnL/dx |_edge / lnL_edge

s is the slice's dimensionless log-slope at its edge, and it is the whole content of the model:

    s = 0  (slice still FLAT at its edge, the common case)  ->  2u - u^2, a gentle roll-off
    s = 1  (slice already decaying linearly in 1/d)         ->  u, the pure asymptotic form

Clamped to s in [0, 2), which is exactly the range over which the continuation is monotone in u,
so walking out in distance always walks lnL down, never up.

WHY VALUE-AND-SLOPE AND NOT A FITTED POLYNOMIAL.  Fitting a through-the-origin quadratic to the
outer part of a slice directly was tried first and over-corrects: over a narrow span of x the
origin constraint sits far outside the data, so the fit has an enormous extrapolation lever and
comes back much steeper than the slice really is.  On the catalog it recovered only ~50% of the
available width error and made the two events that had NO defect measurably worse.  Matching the
value and a locally regressed slope has no lever: a flat slice provably gets the gentle 2u - u^2
roll-off, because that is the unique through-the-origin quadratic that is flat at the edge.

Three properties, and all three matter:

  * CONTINUOUS.  The ratio is 1 at u=1, so this changes nothing on the support and introduces no
    step that the sampler would read as structure.
  * CORRECT LIMIT.  The form is through the origin, so lnL -> 0 as d -> infinity.  RIFT's lnL is a
    likelihood RATIO, so an infinitely distant source is exactly the noise hypothesis: this limit
    is an identity, not a modelling choice.
  * LOCALLY MATCHED.  A flat-edged slice rolls off gently rather than being guillotined, and a
    slice that is already decaying is continued at its own rate.

The decay still comfortably beats the d^2 prior, which is all that is needed to remove the spurious
mass: at twice the edge distance (u=1/2) a flat-edged slice retains 0.75 of its edge lnL, so for a
typical lnL_edge ~ 13 that is a 3.3-nat drop against a 1.4-nat prior gain.  The integrand falls
instead of growing.

WHAT THIS DOES NOT CLAIM.  Beyond the support the data does not constrain the turnover, so the tail
is a physically-motivated continuation, not a measurement.  Its job is to be DECAYING and correct in
the d -> infinity limit, not to be quantitatively exact where nothing was exported.  The honest test
is whether the recovered posterior matches the reference; that is what the validation check scores.
The near-d end is left to the base fit: it is bounded by the CIP distance range and suppressed by
the d^2 prior, and it carries no measured defect.
"""
import numpy as np

__all__ = ["wrap_distance_tail", "polyfit_slice", "slice_fit_report"]

MIN_DISTINCT_D = 6      # below this a slice cannot support the tail fit
OUTER_FRAC = 0.5        # fraction of each slice, from the FAR end, used to fit the tail


def polyfit_slice(d, y, y_err=None, order=2, sel=None):
    """WLS fit of y(x) = sum_{k=1..order} c_k x^k, x = 1/d, THROUGH THE ORIGIN.

    Returns (c, c_err, rms_resid, n) or None.  The intercept is not free: lnL(x=0)=0 is a physical
    identity for a likelihood ratio, and letting it float discards the only information the tree is
    missing.  `sel` restricts the fit to a subset of the points (used to fit only the outer, small-x
    end, where the local expansion is good and where extrapolation will actually happen).
    """
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(d) & np.isfinite(y) & (d > 0)
    if sel is not None:
        ok &= sel
    if ok.sum() < order + 2 or len(np.unique(d[ok])) < MIN_DISTINCT_D:
        return None
    d, y = d[ok], y[ok]
    x = 1.0 / d
    M = np.stack([x ** k for k in range(1, order + 1)], axis=1)
    if y_err is None:
        w = np.ones(len(y))
    else:
        w = 1.0 / np.maximum(np.asarray(y_err, dtype=float)[ok], 1e-3) ** 2
    Mw = M * w[:, None]
    try:
        cov = np.linalg.inv(M.T @ Mw)
    except np.linalg.LinAlgError:
        return None
    c = cov @ (Mw.T @ y)
    r = y - M @ c
    dof = max(len(y) - order, 1)
    chi2_red = float(np.sum(w * r ** 2) / dof)
    c_err = np.sqrt(np.maximum(np.diag(cov), 0.0) * max(chi2_red, 1e-12))
    return c, c_err, float(np.sqrt(np.mean(r ** 2))), len(y)


def slice_fit_report(d, y, y_err=None, orders=(1, 2, 3)):
    """{order: rms residual} for a through-the-origin polynomial on one slice.  Compare against the
    slice's own lnL noise: this is what showed the GLOBAL parametric route to be untenable."""
    return {o: (None if (f := polyfit_slice(d, y, y_err, order=o)) is None else f[2])
            for o in orders}


def _slice_index(X_int):
    """Group rows by identical intrinsic coordinates.  The export holds the intrinsic point fixed
    and varies only distance, so rows within a slice match exactly and np.unique on the raw rows is
    correct and cheap.  Rounding would be wrong: two genuinely distinct grid points can be
    arbitrarily close in a dense grid."""
    _, inv = np.unique(X_int, axis=0, return_inverse=True)
    return inv


def wrap_distance_tail(base_fit, X, Y, coord_names, y_errors=None, dist_name="dist",
                       outer_frac=OUTER_FRAC, lnL_offset=0.0, law="chord", power=None, report=None):
    """Wrap a fitted CIP callable so it decays beyond each intrinsic point's exported distance
    support instead of holding flat.

    base_fit     the already-fitted callable from fit_rf / fit_gp / ... : f(X) -> lnL
    X, Y         the SAME training arrays that base_fit was fitted on, in the CIP fit basis
    coord_names  names of X's columns; must contain `dist_name`
    y_errors     per-row lnL errors, used to weight the per-slice tail fit
    lnL_offset   CIP fits Y = lnL_physical - lnL_shift, so pass lnL_shift here.  The lnL(d->inf)=0
                 identity holds for the PHYSICAL likelihood ratio, so the tail must be built and
                 applied on lnL + lnL_offset and the offset removed again on return.  Leaving this
                 at 0 when CIP has applied a shift anchors the decay to the wrong asymptote --
                 silently, and in the direction that reintroduces the bug.
    report       optional dict, filled with diagnostics

    Returns a callable with the same signature as base_fit.  On the support it returns base_fit
    unchanged, so this is a strict addition: nothing that currently works is altered.
    """
    from scipy.spatial import cKDTree

    names = list(coord_names)
    if dist_name not in names:
        raise ValueError("distance tail fix needs a %r column in the fit basis; got %r"
                         % (dist_name, names))
    idist = names.index(dist_name)
    # 'inv_dist' is a redundant reparametrisation of the same axis; it must not enter the intrinsic
    # key or two rows of one slice would look like two different intrinsic points.
    keep = [i for i, n in enumerate(names) if n not in (dist_name, "inv_dist")]

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    d = X[:, idist]
    sig = None if y_errors is None else np.asarray(y_errors, dtype=float)
    Xint = X[:, keep]
    inv = _slice_index(Xint)
    nsl = int(inv.max()) + 1 if len(inv) else 0

    cent, dmax, slope = [], [], []
    n_flat = 0
    for si in range(nsl):
        m = np.where(inv == si)[0]
        ds = d[m]
        if len(m) < MIN_DISTINCT_D or len(np.unique(ds)) < MIN_DISTINCT_D:
            continue
        # Outer end in DISTANCE == small x, which is the side we extrapolate to.  The slope is fit
        # with a FREE intercept: we want the local gradient, and forcing this local regression
        # through the origin is exactly the lever that made the earlier version over-correct.
        thr = np.quantile(ds, 1.0 - outer_frac)
        sel = ds >= thr
        if sel.sum() < 3 or len(np.unique(ds[sel])) < 3:
            sel = np.ones(len(ds), bool)
        xs = 1.0 / ds[sel]
        ys = Y[m][sel] + lnL_offset
        w = (np.ones(sel.sum()) if sig is None
             else 1.0 / np.maximum(sig[m][sel], 1e-3) ** 2)
        # weighted linear regression ys ~ p + q xs
        sw = w.sum()
        mx = float((w * xs).sum() / sw)
        my = float((w * ys).sum() / sw)
        vxx = float((w * (xs - mx) ** 2).sum())
        if vxx <= 0:
            continue
        q = float((w * (xs - mx) * (ys - my)).sum() / vxx)
        if q <= 0:
            n_flat += 1
        cent.append(Xint[m][0]); dmax.append(ds.max()); slope.append(q)

    if len(cent) < 8:
        raise ValueError("distance tail fix: only %d usable slices (need >=8). Is this a dslice "
                         "export, and did the intrinsic grouping work?" % len(cent))

    cent = np.asarray(cent); dmax = np.asarray(dmax); slope = np.asarray(slope)
    # Nearest slice in intrinsic space, standardised so no coordinate dominates the metric by units
    # alone (mc is O(30), spins O(1)).
    scale = cent.std(axis=0)
    scale[scale <= 0] = 1.0
    tree = cKDTree(cent / scale)

    if report is not None:
        report.update(n_slices_total=int(nsl), n_slices_used=int(len(cent)), law=law,
                      n_flat_or_rising_edge=int(n_flat), outer_frac=float(outer_frac),
                      dmax_median=float(np.median(dmax)))

    def fn_return(x_in):
        x_in = np.asarray(x_in, dtype=float)
        out = np.asarray(base_fit(x_in), dtype=float).copy()
        dq = x_in[:, idist]
        ok = np.isfinite(dq) & (dq > 0) & np.all(np.isfinite(x_in), axis=-1)
        if not ok.any():
            return out
        idx = np.where(ok)[0]
        _, j = tree.query(x_in[idx][:, keep] / scale, k=1)
        de = dmax[j]
        beyond = dq[idx] > de
        if not beyond.any():
            return out
        sel = idx[beyond]
        j = j[beyond]; de = de[beyond]
        # value at the edge, from the base fit itself (denoised), same intrinsic coordinates
        Xe = x_in[sel].copy()
        Xe[:, idist] = de
        if "inv_dist" in names:
            Xe[:, names.index("inv_dist")] = 1.0 / de
        Le = np.asarray(base_fit(Xe), dtype=float) + lnL_offset
        xq = 1.0 / dq[sel]
        xe = 1.0 / de
        u = xq / xe
        # THE DECAY LAW.  Measured, not assumed (see the module docstring):
        #   "chord"  ratio = u.  Parameter-free.  lnL is convex in x=1/d wherever Var(a) > <b> for
        #            the marginalised likelihood, and a convex function through the origin lies
        #            BELOW its chord, so u is an upper bound on lnL beyond the edge as well as the
        #            exact asymptotic form.  This is the default and the only law used in production.
        #   "slope"  the through-the-origin quadratic matched to the slice's local value AND slope.
        #            Faithful to the data at the edge and provably gentle -- and that is exactly why
        #            it fails: the marginalised lnL is a plateau across the whole exported support,
        #            so the local slope cannot see the turnover, and this law barely decays at all.
        #            Kept because measuring that is what ruled it out.
        #   power=p  DIAGNOSTIC ONLY, not a production setting.  Scanning p says whether the best
        #            decay sits at the principled p=1 or only at a tuned value; a fix that needs
        #            tuning to the reference is not a fix.  Never set this in a production run.
        if power is not None:
            ratio = u ** float(power)
        elif law == "chord":
            ratio = u
        else:
            sdl = np.clip(slope[j] * xe / np.where(np.abs(Le) < 1e-12, 1e-12, Le), 0.0, 1.999)
            ratio = (2.0 - sdl) * u + (sdl - 1.0) * u ** 2
        # Guard the continuation rather than trusting it: it may only shrink the edge value, never
        # grow it or flip its sign.  Without this a pathological slice could turn the tail fix into
        # a second, worse extrapolation bug.
        out[sel] = Le * np.clip(ratio, 0.0, 1.0) - lnL_offset
        return out

    return fn_return
