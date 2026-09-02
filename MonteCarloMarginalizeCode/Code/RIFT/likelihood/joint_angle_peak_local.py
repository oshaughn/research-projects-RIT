"""Joint (phi, psi) peak-local marginalization: numpy reference kernel.

WHAT THIS COMPUTES

    L = (1/(2 pi)) int dphi (1/pi) int dpsi exp(g(phi, psi))

with the same normalization as `jax_ile.anglemarg` (uniform priors dphi/2pi,
dpsi/pi).  In the chart u = 2 psi this is a plain double average over the torus,
``(2 pi)^-2 int int dphi du exp(g)``, which is the form everything below uses.

WHY NOT A DENSE GRID.  ``exp(g)`` is a needle whose width falls as ``A^-1/2``, so a
product grid costs ``~A``: with the shipped ``_dense_grid_sizes`` constants that is
4.3e5 points at amplitude 3250 and 4.2e7 at 3.25e5, per row.  The exponent ``g``,
however, is an exact 2-D trig polynomial of bidegree ``(2 m_max, 2)`` -- its
structure does not change with amplitude at all.  Enumerating its modes and
integrating only near them converts an amplitude-scaling cost into a
physics-scaling one.

WHY NOT A SINGLE-CENTRE LAPLACE.  Measured on the shipped coefficient tables, the
(phi, psi) surface carries 8-16 maxima whose values agree to machine precision, and
a one-centre Laplace is 1.4 nats low with an error that does NOT decay with
amplitude, because the deficit is combinatorial rather than curvature.  Localisation
here must be multi-mode; that is the whole point.

HOW THE MODES ARE FOUND, and why this is not a 2-D root solve.  The u-degree of the
exponent is pinned at 2 for ANY mode set (spin-2), so at fixed ``phi`` the
u-stationary points are the unit-circle roots of a degree-4 polynomial -- the same
object ``anglemarg._laplace_psi_lnI`` already solves.  The curve ``{d_u g = 0}`` is
therefore available EXACTLY, with no grid in u, and the 2-D stationary points lie on
it.  What remains is a one-dimensional search in ``phi`` along that curve.  No
resultant, no hidden-variable pencil, no BKK machinery -- and no exposure to the
conditioning of a 2-D solve at the machine-degenerate configurations that are the
normal operating point here.

NO ON-CIRCLE TOLERANCE, deliberately.  The obvious filter ``| |z| - 1 | < tol`` is an
estimate promoted to a bound: at exact multiplicity ``m`` the computed roots smear off
the unit circle by ``eps_machine^(1/m)`` (measured 4.6e-6 for a triple root), so a
1e-6 filter returns ONE mode where there are four, in precisely the degenerate regime
that is production.  Every root is therefore kept and used only as a SEED; the region
machinery below is what decides what is real.  Over-covering is free because regions
merge; under-covering is the only failure that matters.

WHAT IS CERTIFIED, AND WHAT IS NOT.  Read this before quoting the accuracy.

  * The u axis is certified at enumeration time (all roots of an exact quartic).
  * The phi axis is GRID-SEEDED and is therefore NOT certified at enumeration time.
    It carries exactly the caveat the time module carries: a grid is a resolution,
    not a certificate.
  * Correctness is restored the way the time module restores it -- by a bound on the
    part of the domain the regions do not cover.  ``outside_bound`` below is a TRUE
    upper bound on ``g`` outside the covered set: a grid maximum plus the Lipschitz
    remainder ``M_1 * h / 2``, with ``M_1 = sum |C_kq| |k| (or |q|)`` by the triangle
    inequality over the exact coefficient table.  Nothing there is fitted.

  A row whose omitted-mass bound is not small enough is NOT returned with a caveat:
  it is declined, and the caller falls back to the dense rule.
"""

import numpy as np

__all__ = [
    "W_SIGMA",
    "MERGE_MAX_PASSES",
    "OUTSIDE_TOL_NATS",
    "joint_table",
    "eval_g",
    "u_stationary_at_phi",
    "enumerate_modes",
    "derivative_bound",
    "outside_bound",
    "joint_marginalize_peak_local",
]

#: Local integration half-width, in units of the mode's MARGINAL Gaussian sigma, per
#: axis.  It is a CERTIFICATE-COVERAGE constant, not an accuracy knob, and the
#: distinction is measured: over W = 8, 14, 20, 30 the returned value does not move at
#: all (2.27e-13 nats from a converged reference at every W), while the omitted-mass
#: margin goes -18.4, -70.8, -160.2, -308.9 nats.  Widening buys provability, not
#: accuracy.
#:
#: WHY 8 IS NOT ENOUGH, although exp(-8^2/2) = 1.3e-14 suggests it should be.  That
#: estimate assumes the mode is locally Gaussian out to 8 sigma.  These modes sit on
#: RIDGES -- the Hessian condition number at the dominant mode measures 11-23, and the
#: co-dominant modes come in pairs only ~0.009 rad apart -- so an axis-aligned box of
#: marginal sigmas under-covers the ridge, and the outside supremum then lands on a
#: shoulder only 18 nats below the peak rather than on a genuine subdominant maximum
#: 432 nats down.  At W = 8 that fails a -23 nat tolerance; at 14 it passes with -71.
#: 16 is chosen with margin over the measured 14, and is re-derived rather than
#: inherited if the mode structure changes.
W_SIGMA = 16.0

#: Region merging is iterated to a fixed point; this only bounds pathological input.
MERGE_MAX_PASSES = 12

#: Accept a row when log(omitted area) + sup_outside - log(integral) is below this.
#: exp(-23) ~ 1e-10 of the mass.
OUTSIDE_TOL_NATS = -23.0


def joint_table(C_A, C_B, x=1.0):
    """Coefficient table of ``g = x*A - x**2/2 * B`` from the anglemarg tables.

    ``C_A`` and ``C_B`` have DIFFERENT bidegrees -- ``A`` is linear in the waveform
    (phi <= m_max, u <= 1), ``B`` is quadratic (phi <= 2 m_max, u <= 2) -- which is
    why the ``e^{2iu}`` coefficient carries no ``A`` contribution at all, exactly as
    ``anglemarg._laplace_psi_lnI`` documents.  ``A`` is zero-padded into ``B``'s shape.
    """
    a = np.asarray(C_A)
    b = np.asarray(C_B)
    out = (-0.5 * x * x) * b.astype(complex)
    kp = a.shape[0]
    ksa = (a.shape[1] - 1) // 2
    ksb = (b.shape[1] - 1) // 2
    out[:kp, ksb - ksa:ksb + ksa + 1] += x * a
    return out


def _kq(C):
    KP = C.shape[0]
    KS = (C.shape[1] - 1) // 2
    k = np.arange(KP)[:, None]
    q = np.arange(-KS, KS + 1)[None, :]
    w = np.ones((KP, 1))
    w[1:] = 2.0                       # k > 0 stored once, counted twice (real field)
    return k, q, w, KS


#: Points per chunk in :func:`eval_g`.  The temporary is ``(chunk, KP, 2KS+1)``
#: complex, so an unchunked call on a fine reference grid allocates tens of GB -- a
#: 8192^2 torus grid would ask for ~27 GB.  Internal memory parameter only: it cannot
#: change the answer beyond floating-point reassociation, and does not even do that
#: here because each point is summed independently.
_POINT_CHUNK = 200_000


def eval_g(C, phi, u, order=(0, 0)):
    """``d^a_phi d^b_u g`` at points ``(phi, u)``; ``order=(a, b)``.

    Chunked over points: see :data:`_POINT_CHUNK`.
    """
    k, q, w, _ = _kq(C)
    a, b = order
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    u = np.atleast_1d(np.asarray(u, dtype=float))
    fac = ((1j * k) ** a * (1j * q) ** b)[None]
    wC = (w * C)[None]
    n = phi.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, _POINT_CHUNK):
        j = min(i + _POINT_CHUNK, n)
        E = np.exp(1j * (phi[i:j, None, None] * k[None]
                         + u[i:j, None, None] * q[None]))
        out[i:j] = (E * fac * wC).sum(axis=(1, 2)).real
    return out


def derivative_bound(C, order=(0, 0)):
    """TRUE bound on ``|d^a_phi d^b_u g|`` by the triangle inequality on the table.

    Not overridable and not fitted -- the one construction that cannot be a fit.  This
    is the 2-D multi-index form of the time module's ``spectral_derivative_bound``.
    """
    k, q, w, _ = _kq(C)
    a, b = order
    return float((w * np.abs(C) * (np.abs(k) ** a) * (np.abs(q) ** b)).sum())


def u_stationary_at_phi(C, phi):
    """EXACT u-stationary points at fixed ``phi``, as angles in [0, 2 pi).

    At fixed ``phi`` the exponent is ``a + Re(c1 e^{iu}) + Re(c2 e^{2iu})``; with
    ``z = e^{iu}`` its u-derivative vanishes on the roots of a quartic.  ALL roots are
    returned as seeds -- see the module docstring on why there is no ``|z| = 1`` filter.
    """
    k, q, w, KS = _kq(C)
    ph = (np.exp(1j * phi * k) * w).ravel()
    c1 = complex((ph * C[:, KS + 1]).sum())
    c2 = complex((ph * C[:, KS + 2]).sum()) if KS >= 2 else 0.0 + 0.0j
    P = np.array([c2, c1 / 2.0, 0.0, -np.conj(c1) / 2.0, -np.conj(c2)])
    nz = np.nonzero(np.abs(P) > 0.0)[0]
    if nz.size < 2:
        return np.zeros(0)
    return np.mod(np.angle(np.roots(P[nz[0]:])), 2.0 * np.pi)


def _wrap(d):
    """Signed periodic difference in (-pi, pi]."""
    return (np.asarray(d) + np.pi) % (2.0 * np.pi) - np.pi


def enumerate_modes(C, n_phi=64, newton_iters=12):
    """Local maxima of ``g`` on the torus, as ``(points, hessians)``.

    Seeds are ``phi`` grid x EXACT u-roots (see :func:`u_stationary_at_phi`), refined
    by 2-D Newton.  Seeds are targeting only: a seed that converges nowhere useful is
    dropped, and a mode found twice is deduplicated.  Neither costs correctness --
    what the regions miss is carried by :func:`outside_bound`.
    """
    phis = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    seeds = [(p, u) for p in phis for u in u_stationary_at_phi(C, p)]
    if not seeds:
        return np.zeros((0, 2)), np.zeros((0, 2, 2))
    P = np.array(seeds, dtype=float)

    for _ in range(int(newton_iters)):
        gp = eval_g(C, P[:, 0], P[:, 1], (1, 0))
        gu = eval_g(C, P[:, 0], P[:, 1], (0, 1))
        gpp = eval_g(C, P[:, 0], P[:, 1], (2, 0))
        guu = eval_g(C, P[:, 0], P[:, 1], (0, 2))
        gpu = eval_g(C, P[:, 0], P[:, 1], (1, 1))
        det = gpp * guu - gpu * gpu
        ok = np.abs(det) > 1e-300
        dp = np.where(ok, -(guu * gp - gpu * gu) / np.where(ok, det, 1.0), 0.0)
        du = np.where(ok, -(-gpu * gp + gpp * gu) / np.where(ok, det, 1.0), 0.0)
        step = np.hypot(dp, du)
        # Trust region: an unbounded Newton step means the seed is on a saddle ridge,
        # not that the mode is far away.
        scale = np.where(step > 0.5, 0.5 / np.maximum(step, 1e-300), 1.0)
        P[:, 0] = np.mod(P[:, 0] + dp * scale, 2.0 * np.pi)
        P[:, 1] = np.mod(P[:, 1] + du * scale, 2.0 * np.pi)

    gpp = eval_g(C, P[:, 0], P[:, 1], (2, 0))
    guu = eval_g(C, P[:, 0], P[:, 1], (0, 2))
    gpu = eval_g(C, P[:, 0], P[:, 1], (1, 1))
    res = np.hypot(eval_g(C, P[:, 0], P[:, 1], (1, 0)),
                   eval_g(C, P[:, 0], P[:, 1], (0, 1)))
    m1 = derivative_bound(C, (1, 0)) + derivative_bound(C, (0, 1))
    is_max = (gpp < 0) & (gpp * guu - gpu * gpu > 0) & (res <= 1e-6 * max(m1, 1e-300))
    P = P[is_max]
    H = np.stack([np.stack([gpp[is_max], gpu[is_max]], -1),
                  np.stack([gpu[is_max], guu[is_max]], -1)], -2)
    if P.shape[0] == 0:
        return P, H

    # deduplicate: modes closer than 1e-6 rad are the same mode found twice
    keep = []
    for i in range(P.shape[0]):
        d = np.hypot(_wrap(P[i, 0] - P[keep, 0]), _wrap(P[i, 1] - P[keep, 1])) \
            if keep else np.array([np.inf])
        if d.min() > 1e-6:
            keep.append(i)
    return P[keep], H[keep]


def _merge_boxes(cen, half):
    """Merge overlapping axis-aligned boxes on the torus, to a fixed point.

    Merging is not tidiness: overlapping regions would double-count the mass between
    them.  It is also what makes the rule degrade CONTINUOUSLY into the dense grid --
    as amplitude falls the regions widen, merge, and the union grows to the whole
    torus with no threshold anywhere.
    """
    cen = cen.copy()
    half = half.copy()
    for _ in range(MERGE_MAX_PASSES):
        n = cen.shape[0]
        if n < 2:
            break
        merged = False
        out_c, out_h, used = [], [], np.zeros(n, dtype=bool)
        for i in range(n):
            if used[i]:
                continue
            c, h = cen[i].copy(), half[i].copy()
            for j in range(i + 1, n):
                if used[j]:
                    continue
                d = np.abs(_wrap(cen[j] - c))
                if np.all(d < h + half[j]):
                    lo = np.minimum(-h, _wrap(cen[j] - c) - half[j])
                    hi = np.maximum(h, _wrap(cen[j] - c) + half[j])
                    c = np.mod(c + 0.5 * (lo + hi), 2.0 * np.pi)
                    h = np.minimum(0.5 * (hi - lo), np.pi)
                    used[j] = True
                    merged = True
            used[i] = True
            out_c.append(c)
            out_h.append(h)
        cen = np.array(out_c)
        half = np.array(out_h)
        if not merged:
            break
    return cen, half


def outside_bound(C, cen, half, n_grid=256):
    """TRUE upper bound on ``g`` outside the covered boxes, and the uncovered area.

    A grid maximum alone is a LOWER bound on a supremum and the gap grows with
    amplitude, so it is corrected by the Lipschitz remainder ``(M_phi + M_u) * h / 2``
    with the ``M`` from :func:`derivative_bound` -- a true bound from the exact
    coefficient table, nothing fitted.
    """
    t = np.linspace(0.0, 2.0 * np.pi, int(n_grid), endpoint=False)
    PHI, U = np.meshgrid(t, t, indexing='ij')
    inside = np.zeros(PHI.shape, dtype=bool)
    for c, h in zip(cen, half):
        inside |= ((np.abs(_wrap(PHI - c[0])) <= h[0])
                   & (np.abs(_wrap(U - c[1])) <= h[1]))
    area_out = float((~inside).sum()) * (2.0 * np.pi / n_grid) ** 2
    if not np.any(~inside):
        return -np.inf, 0.0

    # THE SLOPES ARE WHAT MAKE THIS AFFORDABLE, exactly as in the time module's
    # certificate.  A zeroth-order bound `max_grid + M1 * r` carries the GLOBAL first
    # derivative bound, and M1 grows linearly with amplitude, so the remainder swamps
    # the tolerance on any grid a peak-local rule can afford (measured: +1225 nats of
    # remainder at amplitude 3.2e4 on a 256^2 grid, a "bound" above the integral
    # itself).  Using each cell's own gradient and paying M2 only on the quadratic term
    # makes the remainder local: it is small wherever the surface is flat, which is
    # precisely where the outside supremum lives.
    r = 0.5 * np.sqrt(2.0) * (2.0 * np.pi / n_grid)     # half-diagonal of a cell
    m = ~inside
    ph = PHI[m].ravel()
    uu = U[m].ravel()
    g0 = eval_g(C, ph, uu)
    gp = eval_g(C, ph, uu, (1, 0))
    gu = eval_g(C, ph, uu, (0, 1))
    m2 = (derivative_bound(C, (2, 0)) + 2.0 * derivative_bound(C, (1, 1))
          + derivative_bound(C, (0, 2)))
    local = g0 + np.hypot(gp, gu) * r + 0.5 * m2 * r * r
    return float(local.max()), area_out


#: Trapezoid points per local sigma.  Derived, not tuned: the trapezoidal rule on a
#: Gaussian of width sigma at spacing h has relative error 2 exp(-2 pi^2 sigma^2/h^2)
#: by Poisson summation, so sigma/h = 3 gives 2e-77 -- the same argument, and the same
#: kind of margin, as UPSAMPLE_SAFETY in the band-limited time quadrature.
#:
#: Measured when dropping 6 -> 3: the value is UNCHANGED (0.0 nats) at amplitudes 325,
#: 3249 and 3.2e4, and the local point count falls from 147k to 82k.  At amplitude 3.4
#: it moves by 1.1e-8 nats -- because there the single region has merged to the whole
#: torus and the integrand is not a Gaussian bump at all, so the Poisson argument above
#: does not apply to it.  That is the LOW-AMPLITUDE end of this rule's exclusion region,
#: where the selector should be routing to the dense rule anyway; the residual is
#: recorded rather than hidden because "bit-identical" would have been the wrong claim.
_PTS_PER_SIGMA = 3


def _log_box_integral(C, c, h, pts_per_sigma=_PTS_PER_SIGMA, max_pts=256):
    """``log int_box exp(g)`` by a tensor trapezoid sized from the LOCAL curvature."""
    n = []
    for ax in (0, 1):
        order = (2, 0) if ax == 0 else (0, 2)
        curv = abs(float(eval_g(C, c[0], c[1], order)[0]))
        sig = 1.0 / np.sqrt(curv) if curv > 0 else h[ax]
        want = int(np.ceil(2.0 * h[ax] / max(sig, 1e-12) * pts_per_sigma)) + 1
        n.append(int(np.clip(want, 9, max_pts)))
    a = c[0] + np.linspace(-h[0], h[0], n[0])
    b = c[1] + np.linspace(-h[1], h[1], n[1])
    A, B = np.meshgrid(a, b, indexing='ij')
    g = eval_g(C, A.ravel(), B.ravel()).reshape(A.shape)
    wa = np.full(n[0], 2.0 * h[0] / (n[0] - 1)); wa[0] *= 0.5; wa[-1] *= 0.5
    wb = np.full(n[1], 2.0 * h[1] / (n[1] - 1)); wb[0] *= 0.5; wb[-1] *= 0.5
    W = np.log(wa)[:, None] + np.log(wb)[None, :]
    m = g.max()
    return m + np.log(np.sum(np.exp(g - m + W))), n[0] * n[1]


def joint_marginalize_peak_local(C, n_phi=64, n_bound_grid=256,
                                 tol_nats=OUTSIDE_TOL_NATS):
    """``log[(2 pi)^-2 int int dphi du exp(g)]``, refining only near the modes.

    Returns ``(value, ok, report)``.  ``ok`` is False when the omitted-mass bound could
    not be made small enough; the caller must then use the dense rule.  The value is
    returned either way for diagnosis, but a value with ``ok=False`` is NOT to be used.
    """
    C = np.asarray(C)
    rep = {'n_modes': 0, 'n_regions': 0, 'n_local_points': 0,
           'margin': np.inf, 'area_outside': np.nan, 'sup_outside': np.nan,
           'decline': None}

    P, H = enumerate_modes(C, n_phi=n_phi)
    rep['n_modes'] = int(P.shape[0])
    if P.shape[0] == 0:
        rep['decline'] = 'no modes enumerated'
        return -np.inf, False, rep

    # marginal sigmas of the local Gaussian: sqrt of the diagonal of (-H)^-1
    half = np.empty_like(P)
    for i in range(P.shape[0]):
        Ci = np.linalg.inv(-H[i])
        half[i, 0] = W_SIGMA * np.sqrt(max(Ci[0, 0], 1e-300))
        half[i, 1] = W_SIGMA * np.sqrt(max(Ci[1, 1], 1e-300))
    half = np.minimum(half, np.pi)

    cen, half = _merge_boxes(P, half)
    rep['n_regions'] = int(cen.shape[0])

    parts, npts = [], 0
    for c, h in zip(cen, half):
        v, k = _log_box_integral(C, c, h)
        parts.append(v)
        npts += k
    rep['n_local_points'] = int(npts)
    parts = np.array(parts)
    m = parts.max()
    log_inside = m + np.log(np.exp(parts - m).sum())

    sup_out, area_out = outside_bound(C, cen, half, n_grid=n_bound_grid)
    rep['sup_outside'] = sup_out
    rep['area_outside'] = area_out
    if area_out <= 0.0:
        rep['margin'] = -np.inf
    else:
        rep['margin'] = float(np.log(area_out) + sup_out - log_inside)

    ok = rep['margin'] < tol_nats
    if not ok:
        rep['decline'] = 'omitted-mass bound too large'
    return float(log_inside - 2.0 * np.log(2.0 * np.pi)), bool(ok), rep
