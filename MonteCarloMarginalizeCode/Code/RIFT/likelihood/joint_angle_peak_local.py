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
    "joint_marginalize_over_distance",
    "u_profile",
    "phi_local_marginalize",
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

#: Per-axis ceiling on a box's trapezoid.  NOT a free tuning knob: it is the point at
#: which the local integration stops honouring the curvature it derived, and the
#: certificate cannot report that -- the omitted-mass bound covers what is outside the
#: boxes, so a capped box can carry ``margin = -inf`` and still be wrong.  Measured on
#: the rho=163.08 production tables (amplitude ~2.7e4, ``area_outside == 0``) against a
#: torus reference self-converged to 2e-12: at 256 the value was off by up to 0.36 nats,
#: at 512 by 3e-4, at 1024 exact to 1e-4.  Cost went 0.07 s -> 0.21 s -> 0.83 s.  512
#: buys three orders of magnitude for 3x, and 1024 buys almost nothing more for 12x.
#: Raising this does not widen the certificate's REACH -- declines are omitted-mass
#: declines and this is internal accuracy; the two are independent, and both are needed.
_BOX_MAX_PTS = 512


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
    # both signs of q are stored, and Re[D_{-q} e^{-iqu}] = Re[conj(D_{-q}) e^{+iqu}],
    # so the effective coefficient is D_{+q} + conj(D_{-q}).  Only the +q column was
    # used here originally; the error was invisible because these roots are SEEDS that
    # 2-D Newton then corrects -- the jax kernel, where the roots define the integration
    # partition, is where it showed up.
    _D = lambda qq: complex((ph * C[:, KS + qq]).sum())
    c1 = _D(1) + np.conj(_D(-1))
    c2 = (_D(2) + np.conj(_D(-2))) if KS >= 2 else 0.0 + 0.0j
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
    # VERIFY, do not assume.  Exiting on the pass limit with boxes still overlapping
    # would double-count the mass between them -- silently, and in the direction that
    # inflates the answer.  The caller declines on `converged=False`.
    converged = True
    for i in range(cen.shape[0]):
        for j in range(i + 1, cen.shape[0]):
            if np.all(np.abs(_wrap(cen[j] - cen[i])) < half[i] + half[j]):
                converged = False
                break
        if not converged:
            break
    return cen, half, converged


def outside_bound(C, cen, half, n_grid=256):
    """TRUE upper bound on ``g`` outside the covered boxes, and the uncovered area.

    A grid maximum alone is a LOWER bound on a supremum and the gap grows with
    amplitude, so it is corrected by the Lipschitz remainder ``(M_phi + M_u) * h / 2``
    with the ``M`` from :func:`derivative_bound` -- a true bound from the exact
    coefficient table, nothing fitted.
    """
    step = 2.0 * np.pi / int(n_grid)
    t = np.linspace(0.0, 2.0 * np.pi, int(n_grid), endpoint=False)
    PHI, U = np.meshgrid(t, t, indexing='ij')

    # A CELL IS COVERED ONLY IF THE WHOLE CELL IS INSIDE A BOX, not merely its centre.
    # Classifying centres leaves cells that straddle a box edge unexamined: their
    # uncovered part contributes to neither the supremum search nor the area.  The
    # failure is not hypothetical or small -- a box positioned half a step off-axis can
    # cover every grid CENTRE while leaving several rad^2 genuinely uncovered, and this
    # function would then return (-inf, 0.0), i.e. UNCONDITIONAL ACCEPTANCE.  Shrinking
    # each box by half a cell before testing makes a straddling cell count as outside;
    # the uncovered area is then an OVER-estimate, which is the safe direction.
    # A box that already spans the full circle on an axis covers it whatever the
    # shrink does; without this the low-amplitude case -- where the regions have merged
    # to the whole torus, which is the rule degenerating into the dense grid exactly as
    # intended -- would report an uncovered band and decline every such row.
    shrink = 0.5 * step
    inside = np.zeros(PHI.shape, dtype=bool)
    for c, h in zip(cen, half):
        eff = np.where(h >= np.pi, np.pi + 1.0, h - shrink)
        inside |= ((np.abs(_wrap(PHI - c[0])) <= eff[0])
                   & (np.abs(_wrap(U - c[1])) <= eff[1]))
    area_out = float((~inside).sum()) * step * step
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
    r = 0.5 * np.sqrt(2.0) * step                       # half-diagonal of a cell
    m = ~inside
    ph = PHI[m].ravel()
    uu = U[m].ravel()
    g0 = eval_g(C, ph, uu)
    gp = eval_g(C, ph, uu, (1, 0))
    gu = eval_g(C, ph, uu, (0, 1))
    # Why this M2 is a valid remainder, written out because it is not obvious and was
    # doubted on review.  For |d| <= r, |0.5 d^T H d| <= 0.5 r^2 [max(M20,M02) + M11]
    # (maximise M20 cos^2 + M11|sin 2t| + M02 sin^2).  The value used here,
    # M20 + 2 M11 + M02, dominates that for any non-negative M, so the bound holds --
    # conservatively.  Checked as well as argued: 0 violations over 1800 (point, radius)
    # samples on 300 random tables.
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


def _log_box_integral(C, c, h, pts_per_sigma=_PTS_PER_SIGMA, max_pts=_BOX_MAX_PTS):
    """``log int_box exp(g)`` by a tensor trapezoid sized from the LOCAL curvature.

    Returns ``(value, n_points, capped)``.  ``capped`` is True when ``max_pts`` bound the
    curvature-derived count on either axis -- i.e. the sizing rule ASKED FOR MORE NODES
    THAN IT GOT.  That is a truncated request, NOT a verdict that the value is wrong:
    measured on the ladder, rung 1 (rho=40.77, amplitude ~2.5e3) caps on every
    mass-carrying point and is still exact to 0.00000 nats against a converged reference,
    while rung 3 (rho=163.08, amplitude ~2.8e4) caps and is 0.36 nats out.  The trapezoid
    on a periodic integrand converges fast enough that the derived count is conservative
    at low amplitude and binding at high.  So treat the flag as "look here", not "this is
    broken" -- it is the only signal available, because the certificate cannot see inside
    a box at all.  It has to be reported,
    because the certificate cannot see it: the omitted-mass bound covers what is OUTSIDE
    the boxes and says nothing about the quadrature inside one, so a capped box is exactly
    the case where ``margin`` can read ``-inf`` (nothing omitted at all) while the value is
    still wrong.  Measured on the rho=163 production tables: at the shipped cap of 256 the
    value sat 0.36 nats from a converged torus reference with ``area_outside == 0``.
    """
    n = []
    capped = False
    for ax in (0, 1):
        order = (2, 0) if ax == 0 else (0, 2)
        curv = abs(float(eval_g(C, c[0], c[1], order)[0]))
        sig = 1.0 / np.sqrt(curv) if curv > 0 else h[ax]
        want = int(np.ceil(2.0 * h[ax] / max(sig, 1e-12) * pts_per_sigma)) + 1
        if want > max_pts:
            capped = True
        n.append(int(np.clip(want, 9, max_pts)))
    a = c[0] + np.linspace(-h[0], h[0], n[0])
    b = c[1] + np.linspace(-h[1], h[1], n[1])
    A, B = np.meshgrid(a, b, indexing='ij')
    g = eval_g(C, A.ravel(), B.ravel()).reshape(A.shape)
    wa = np.full(n[0], 2.0 * h[0] / (n[0] - 1)); wa[0] *= 0.5; wa[-1] *= 0.5
    wb = np.full(n[1], 2.0 * h[1] / (n[1] - 1)); wb[0] *= 0.5; wb[-1] *= 0.5
    W = np.log(wa)[:, None] + np.log(wb)[None, :]
    m = g.max()
    return m + np.log(np.sum(np.exp(g - m + W))), n[0] * n[1], capped


def joint_marginalize_peak_local(C, n_phi=64, n_bound_grid=256,
                                 tol_nats=OUTSIDE_TOL_NATS):
    """``log[(2 pi)^-2 int int dphi du exp(g)]``, refining only near the modes.

    Returns ``(value, ok, report)``.  ``ok`` is False when the omitted-mass bound could
    not be made small enough; the caller must then use the dense rule.  The value is
    returned either way for diagnosis, but a value with ``ok=False`` is NOT to be used.
    """
    C = np.asarray(C)
    rep = {'n_modes': 0, 'n_regions': 0, 'n_local_points': 0,
           'n_boxes_pts_capped': 0,
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

    cen, half, merged_ok = _merge_boxes(P, half)
    rep['n_regions'] = int(cen.shape[0])
    if not merged_ok:
        rep['decline'] = 'regions still overlap after MERGE_MAX_PASSES'
        return -np.inf, False, rep

    parts, npts, n_capped = [], 0, 0
    for c, h in zip(cen, half):
        v, k, capped = _log_box_integral(C, c, h)
        parts.append(v)
        npts += k
        n_capped += int(capped)
    rep['n_local_points'] = int(npts)
    # a capped box had its node request truncated and the certificate CANNOT see inside a
    # box at all, so surface it: it is the only available signal that 'nothing omitted'
    # might be sitting on a quadrature error.  Capped does NOT mean wrong -- rung 1 caps
    # everywhere and is exact -- it means this is where to look if a value is doubted.
    rep['n_boxes_pts_capped'] = int(n_capped)
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


def joint_marginalize_over_distance(C_A_st, C_B_st, x_grid, log_w_grid,
                                    n_phi=64, n_bound_grid=256,
                                    tol_nats=OUTSIDE_TOL_NATS, keep_nats=None,
                                    _retry=False):
    """Distance-, phi- and psi-marginalized value at ONE ``(sample, time)`` point.

    ``log sum_x exp(log_w_x) * (2 pi)^-2 int int exp(x A - x^2/2 B)``, i.e. the same
    quantity ``anglemarg.fused_log_likelihood_distphipsimarg_exact`` produces before
    time marginalization, with the same normalization.

    THE DISTANCE AXIS IS NOT FREE, and this is where the joint rule's cost actually
    lives.  The mode locations depend on ``x``, so the dense scheme's trick -- form one
    ``(phi, u)`` grid and reuse it for every distance node -- is not available: the
    enumeration is per node.  What rescues it is that the number of nodes CARRYING MASS
    falls as the signal sharpens (measured on the synthetic fixture: 16 of 64 nodes
    within 20 nats at exponent amplitude 32, and 1 of 64 at amplitude 3249), so a cheap
    pre-pass on the nodes bounds the work before any enumeration happens.

    READ THIS BEFORE QUOTING THE HIGH-SNR NUMBERS.  That same collapse means the FIXED
    distance grid is itself under-resolving the distance peak there -- the peak is
    ``~1/SNR`` narrow, which is precisely the defect ``core._distmarg_gh_logL`` exists
    to fix with adaptive nodes.  So "1 node carries the mass" is simultaneously a cost
    win for this rule and a warning about the grid it was handed.  This function
    inherits the caller's distance quadrature and does not repair it.

    ``keep_nats`` selects the nodes to work on, using a CHEAP upper bound on each node's
    contribution rather than the node's actual value: ``log_w_x + max_(phi,u) g_x``,
    where the maximum is taken over the coarse bound grid and lifted by the same local
    slope/curvature remainder the outside bound uses.  Dropping a node therefore drops
    something provably below the kept mass, not something estimated to be -- and the
    drop is CHECKED against ``tol_nats``, not merely reported.

    ``keep_nats`` is DERIVED from the tolerance by default, not an independent constant.
    It was one (25.0), and the two never had to agree: the dropped set's certified
    contribution came out only 15.7 nats below the kept value against a 23 nat
    tolerance, so the filter was quietly discarding more than the rule was allowed to
    lose.  The requirement is ``log(n_dropped) + max(ub_dropped) - value < tol_nats``;
    keeping everything within ``|tol_nats| + log(n_nodes)`` of the best bound satisfies
    it with room to spare, and adapts automatically if either is changed.
    """
    if keep_nats is None:
        keep_nats = abs(float(tol_nats)) + np.log(max(len(x_grid), 1)) + 5.0
    keep_nats = float(keep_nats)
    x_grid = np.asarray(x_grid, dtype=float).ravel()
    log_w_grid = np.asarray(log_w_grid, dtype=float).ravel()

    # --- cheap pre-pass: a true upper bound on each node's contribution
    t = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    PHI, U = np.meshgrid(t, t, indexing='ij')
    r = 0.5 * np.sqrt(2.0) * (2.0 * np.pi / 96)
    ub = np.empty(x_grid.size)
    for i, x in enumerate(x_grid):
        C = joint_table(C_A_st, C_B_st, x=float(x))
        g0 = eval_g(C, PHI.ravel(), U.ravel())
        gp = eval_g(C, PHI.ravel(), U.ravel(), (1, 0))
        gu = eval_g(C, PHI.ravel(), U.ravel(), (0, 1))
        m2 = (derivative_bound(C, (2, 0)) + 2.0 * derivative_bound(C, (1, 1))
              + derivative_bound(C, (0, 2)))
        ub[i] = log_w_grid[i] + float((g0 + np.hypot(gp, gu) * r
                                       + 0.5 * m2 * r * r).max())
    # The filter is a COST optimization and must never change the answer.  A threshold
    # alone cannot guarantee that: `ub` is certified but LOOSE -- measured about 13 nats
    # above the value it bounds -- so a cut that looks safe against `ub.max()` can still
    # leave the dropped set above tolerance relative to the ACTUAL value.  That is why
    # the check below is made against the computed value, and why failing it retries
    # with every node rather than widening by a guess: adding nodes only raises the
    # value and empties the dropped set, so the retry always succeeds.
    live = np.nonzero(ub > ub.max() - keep_nats)[0]

    parts, ok_all, rep = [], True, {'n_nodes': int(x_grid.size),
                                    'n_nodes_live': int(live.size),
                                    'worst_margin': -np.inf, 'declines': []}
    for i in live:
        C = joint_table(C_A_st, C_B_st, x=float(x_grid[i]))
        v, ok, r_i = joint_marginalize_peak_local(
            C, n_phi=n_phi, n_bound_grid=n_bound_grid, tol_nats=tol_nats)
        if not ok:
            ok_all = False
            rep['declines'].append((int(i), r_i['decline']))
        rep['worst_margin'] = max(rep['worst_margin'], r_i['margin'])
        parts.append(log_w_grid[i] + v)

    if not parts:
        return -np.inf, False, rep
    parts = np.array(parts)
    m = parts.max()
    value = m + np.log(np.exp(parts - m).sum())

    # THE DROPPED NODES MUST GATE THE RESULT, not merely be recorded.  An earlier
    # revision computed `ub` for them, stored the maximum in the report, and then
    # returned `ok` without ever comparing it to anything -- the docstring promised a
    # provably-negligible drop while the code performed an unchecked one.  Each dropped
    # node contributes at most `exp(ub_i)`, so the whole dropped set contributes at most
    # `log(n_dropped) + max(ub)`, and that must sit below the tolerance relative to the
    # kept value on the same scale.
    dropped = np.setdiff1d(np.arange(x_grid.size), live)
    if dropped.size:
        # logsumexp over the dropped set, not log(n) + max: both are valid upper
        # bounds on the omitted contribution, but the aggregate one is tighter and so
        # declines fewer rows for the same guarantee.
        _dm = ub[dropped].max()
        drop_bound = float(_dm + np.log(np.exp(ub[dropped] - _dm).sum()))
        rep['dropped_bound'] = drop_bound
        rep['dropped_margin'] = drop_bound - value
        if rep['dropped_margin'] >= tol_nats and not _retry:
            # the cut was not justified against the real value: redo with every node.
            rep2 = dict(rep)
            v2, ok2, r2 = joint_marginalize_over_distance(
                C_A_st, C_B_st, x_grid, log_w_grid, n_phi=n_phi,
                n_bound_grid=n_bound_grid, tol_nats=tol_nats,
                keep_nats=np.inf, _retry=True)
            r2['prefilter_retried'] = True
            r2['prefilter_first_margin'] = rep['dropped_margin']
            return v2, ok2, r2
        if rep['dropped_margin'] >= tol_nats:
            ok_all = False
            rep['declines'].append(('dropped-nodes', 'pre-filter bound too large'))
    return float(value), bool(ok_all), rep


# ------------------------------------------------------------------ phi-local

def _g_uu_at(C, phi, u):
    """``d^2 g / du^2`` at one ``phi`` and several ``u``."""
    return eval_g(C, np.full(np.size(u), float(phi)), np.asarray(u, dtype=float), (0, 2))


def u_profile(C, phi, n_nodes=64, window_sigma=12.0):
    """``F(phi) = log int du exp(g)``, and its first two EXACT derivatives.

    The u integral is done on the cell partition (the sorted u-stationary points tile the
    circle), so ``F`` is exact rather than a Laplace model.  Its derivatives come from
    the same nodes at no extra evaluation cost, by differentiating under the integral:

        F'  = E[d_phi g]
        F'' = E[d^2_phi g] + Var(d_phi g)

    with the expectation under the normalized ``exp(g) du`` on the same axis.  That
    variance term is why the phi axis cannot inherit the u axis's economy: it grows with
    amplitude, so ``F`` sharpens as the signal does even though ``g`` itself does not.

    Returns ``(F, dF, ddF)``, each shaped like ``phi``.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    k, q, w, KS = _kq(C)
    out = np.empty((3, phi.size))
    for i, p in enumerate(phi):
        ph = (np.exp(1j * p * k) * w).ravel()
        _D = lambda qq: complex((ph * C[:, KS + qq]).sum())
        c1 = _D(1) + np.conj(_D(-1))
        c2 = (_D(2) + np.conj(_D(-2))) if KS >= 2 else 0.0 + 0.0j
        P = np.array([c2, c1 / 2.0, 0.0, -np.conj(c1) / 2.0, -np.conj(c2)])
        nz = np.nonzero(np.abs(P) > 0.0)[0]
        roots = (np.mod(np.angle(np.roots(P[nz[0]:])), 2 * np.pi)
                 if nz.size >= 2 else np.linspace(0, 2 * np.pi, 4, endpoint=False))
        u = np.sort(np.concatenate([roots, np.zeros(max(0, 4 - roots.size))]))[:4]
        mid = 0.5 * (u + np.roll(u, -1) + np.where(np.arange(4) == 3, 2 * np.pi, 0.0))
        lo_c = np.roll(mid, 1) - np.where(np.arange(4) == 0, 2 * np.pi, 0.0)

        # WINDOW EACH CELL BY ITS OWN sigma, do not spread a fixed node count over the
        # whole cell.  The cells do NOT shrink with amplitude -- the stationary points of
        # g are invariant under g -> lambda g -- while the peak inside them does, so a
        # fixed uniform rule over the full cell silently under-resolves as the signal
        # sharpens.  Measured before this change, against a converged reference: error
        # 8.8e-07 at exponent amplitude 1265 and 5.4e-04 at 4217, both of which fall to
        # EXACTLY 0.0 when the node count is raised -- i.e. the entire residual was this
        # rule, not the method.  A resolution that is a fixed number whose default is
        # assumed ample is the defect this whole line of work exists to remove.
        # REFINE THE CENTRE INSIDE EACH CELL FIRST.  The quartic roots are SEEDS, not
        # located maxima: this module says elsewhere that they may leave the unit circle
        # (a conjugate-reciprocal pair does exactly that), and a spurious root's angle is
        # not a stationary point at all.  Windowing +-W sigma around the raw root then
        # centres the window on the wrong place and takes sigma from the wrong curvature,
        # which under-resolves the peak that IS in the cell -- an inside-cell quadrature
        # error the phi omitted-mass certificate cannot see.  Measured on a constructed
        # table before this change: -7.2e-04 nats returned with ok=True, converging to
        # the reference only as n_nodes was raised.  The jax kernel already refines;
        # this path did not, and the two must not differ on something load-bearing.
        ustar = u.copy()
        pv = np.full(ustar.size, float(p))
        for _ in range(8):
            g1 = eval_g(C, pv, ustar, (0, 1))
            g2 = eval_g(C, pv, ustar, (0, 2))
            step = np.where(np.abs(g2) > 0.0,
                            -g1 / np.where(np.abs(g2) > 0.0, g2, 1.0), 0.0)
            ustar = np.clip(ustar + np.clip(step, -0.5, 0.5), lo_c, mid)

        # A CLIPPED NEWTON POINT IS NOT A PEAK, however negative the curvature there.
        # The iteration is clamped to [lo_c, mid], so it can come to rest ON a cell
        # boundary with a large stationary residual; classifying that as a maximum
        # centres a +-W sigma window on a non-stationary point and sizes sigma from the
        # wrong curvature.  Require, as well as g'' < 0, that the residual is small
        # relative to the axis's own derivative bound AND that the point is interior.
        # A cell failing either is integrated WHOLE rather than windowed, which is the
        # conservative branch: it can only add nodes, never move the centre.
        g1c = eval_g(C, pv, ustar, (0, 1))
        g2c = _g_uu_at(C, p, ustar)
        _m1u = max(derivative_bound(C, (0, 1)), 1e-300)
        _edge = 1e-9 * max(float(np.max(mid - lo_c)), 1e-300)
        peaked = ((g2c < 0.0)
                  & (np.abs(g1c) <= 1e-8 * _m1u)
                  & (ustar > lo_c + _edge) & (ustar < mid - _edge))
        sig_c = np.where(peaked, 1.0 / np.sqrt(np.where(peaked, -g2c, 1.0)), np.inf)
        lo = np.where(peaked, np.maximum(ustar - window_sigma * sig_c, lo_c), lo_c)
        hi = np.where(peaked, np.minimum(ustar + window_sigma * sig_c, mid), mid)
        # DERIVE THE NODE COUNT; the fallback cell is where a fixed one fails.  A
        # windowed cell spans +-W sigma so a fixed count resolves it, but a cell that
        # FELL BACK spans the whole cell with the same nodes -- and an earlier comment
        # here claimed that branch "can only add nodes", which was simply false: it adds
        # none and spreads them wider, so rejecting a peak made the resolution WORSE.
        # Measured on a searched counterexample: 1.7e-03 nats at 64 nodes, converging
        # only by n = 1024.
        #
        # The requirement is derived, not tuned: |d^2 g/du^2| <= M2u = |c1| + 4|c2|
        # EXACTLY, so no feature of exp(g) on this axis is narrower than
        # sigma_min = 1/sqrt(M2u), and a spacing of sigma_min/_PTS_PER_SIGMA resolves the
        # sharpest thing the coefficients admit.
        # SCALE, AND WHY THIS ONE.  |d2g/du2| <= M2u = |c1| + 4|c2| exactly, so nothing
        # on this axis is narrower than sigma_min = 1/sqrt(M2u) and a spacing of
        # sigma_min/_PTS_PER_SIGMA resolves the sharpest feature the coefficients admit.
        # Measured on a searched counterexample, this takes the 64-vs-1024-node gap from
        # 1.7e-03 to 2.2e-04 nats at essentially no cost.
        #
        # A BOUNDARY-PEAKED FALLBACK CELL IS NOT GAUSSIAN THERE -- exp(g) falls off like
        # exp(-|g'| du), so full convergence would need a spacing set by 1/M1u, and that
        # was measured to cost a 25x slowdown for the remaining 2.2e-04 nats.  Not taken:
        # 2e-04 nats is orders below anything this rule is asked to decide, and the
        # residual is reported (`n_fallback_cells`) rather than hidden.  If a future
        # caller needs it, raise _PTS_PER_SIGMA or pass n_nodes -- both are honest knobs
        # and both cost what they cost.
        m2u = abs(c1) + 4.0 * abs(c2)
        width_c = np.maximum(hi - lo, 0.0)
        need = int(np.ceil(float(width_c.max()) * np.sqrt(max(m2u, 1e-300))
                           * _PTS_PER_SIGMA)) + 1
        n_use = int(np.clip(max(n_nodes, need), n_nodes, 8192))
        s = np.linspace(0.0, 1.0, n_use)
        uu = lo[:, None] + width_c[:, None] * s[None, :]
        pp = np.full(uu.size, p)
        g = eval_g(C, pp, uu.ravel())
        gp = eval_g(C, pp, uu.ravel(), (1, 0))
        gpp = eval_g(C, pp, uu.ravel(), (2, 0))
        wq = np.full(n_use, 1.0 / (n_use - 1)); wq[0] *= 0.5; wq[-1] *= 0.5
        lw = (np.log(np.maximum(hi - lo, 1e-300))[:, None] + np.log(wq)[None, :]).ravel()
        m = g.max()
        wgt = np.exp(g - m + lw)
        Z = wgt.sum()
        e1 = float((wgt * gp).sum() / Z)
        out[0, i] = m + np.log(Z)
        out[1, i] = e1
        out[2, i] = float((wgt * (gpp + gp * gp)).sum() / Z) - e1 * e1
    return out[0], out[1], out[2]


def phi_local_marginalize(C, n_seed=64, w_sigma=12.0, n_nodes=64,
                          n_bound_grid=512, tol_nats=OUTSIDE_TOL_NATS):
    """``log[(2 pi)^-2 int int dphi du exp(g)]`` with BOTH axes localized.

    u is exact on the cell partition; phi is localized around the maxima of the profile
    ``F`` using its exact derivatives.  The phi axis has no algebraic completeness
    warrant -- ``F`` is a log-integral, not a trig polynomial -- so it is the framework's
    grid-seeded class and its correctness rests on the cover bound, exactly as the time
    axis does.

    Returns ``(value, ok, report)``; ``ok=False`` means the omitted-mass bound on phi
    could not be made small enough and the caller must fall back.
    """
    rep = {'n_phi_modes': 0, 'n_phi_regions': 0, 'margin': np.inf, 'decline': None}
    seeds = np.linspace(0.0, 2.0 * np.pi, int(n_seed), endpoint=False)
    p = seeds.copy()
    for _ in range(24):
        _, d1, d2 = u_profile(C, p, n_nodes=n_nodes)
        step = np.where(np.abs(d2) > 0, -d1 / np.where(np.abs(d2) > 0, d2, 1.0), 0.0)
        p = np.mod(p + np.clip(step, -0.3, 0.3), 2.0 * np.pi)
    F, d1, d2 = u_profile(C, p, n_nodes=n_nodes)
    keep = (d2 < 0) & (np.abs(d1) < 1e-6 * max(derivative_bound(C, (1, 0)), 1e-300))
    p, F, d2 = p[keep], F[keep], d2[keep]
    if p.size == 0:
        rep['decline'] = 'no phi modes'
        return -np.inf, False, rep
    order = np.argsort(p); p, F, d2 = p[order], F[order], d2[order]
    uniq = np.concatenate([[True], np.diff(p) > 1e-6])
    p, F, d2 = p[uniq], F[uniq], d2[uniq]
    rep['n_phi_modes'] = int(p.size)

    sig = 1.0 / np.sqrt(-d2)
    lo = p - w_sigma * sig
    hi = p + w_sigma * sig
    # 1-D merge: sort by lo and absorb overlaps.  Same argument as the time module --
    # merging is what stops the mass between two windows being counted twice.
    # MERGE ON THE CIRCLE, NOT ON THE LINE.  A linear sweep over raw
    # [p - W sigma, p + W sigma] never joins a window near 0 to one near 2 pi -- but
    # each region is afterwards integrated at mod(., 2 pi), so BOTH regions cover BOTH
    # peaks and that mass is counted twice.  Measured: +log 2 = +0.693 nats returned
    # with ok=True and a margin of -437, because the error is INSIDE the regions and an
    # omitted-mass certificate cannot see it.  Same family as the wrapped-circuit bug,
    # one step over.
    #
    # Reduce every interval to the circle, SPLIT any that crosses the seam, merge the
    # pieces linearly, then close the circle by joining a piece touching 2 pi to one
    # touching 0.
    pieces = []
    for a, b in zip(lo, hi):
        wdt = min(float(b - a), 2.0 * np.pi)
        a = float(np.mod(a, 2.0 * np.pi))
        if a + wdt <= 2.0 * np.pi:
            pieces.append((a, a + wdt))
        else:
            pieces.append((a, 2.0 * np.pi))
            pieces.append((0.0, a + wdt - 2.0 * np.pi))
    pieces.sort()
    ml, mh = [pieces[0][0]], [pieces[0][1]]
    for a, b in pieces[1:]:
        if a <= mh[-1]:
            mh[-1] = max(mh[-1], b)
        else:
            ml.append(a)
            mh.append(b)
    if len(ml) > 1 and ml[0] <= 1e-12 and mh[-1] >= 2.0 * np.pi - 1e-12:
        ml[0] = ml[-1] - 2.0 * np.pi      # the two seam pieces are one region
        ml.pop()
        mh.pop()
    ml, mh = np.array(ml, dtype=float), np.array(mh, dtype=float)

    # CLAMP TO ONE CIRCUIT.  At low amplitude F is nearly flat, so sigma is huge and
    # [p - W sigma, p + W sigma] spans far MORE than 2 pi; integrating that range
    # literally wraps the circle several times and counts the same mass repeatedly.
    # Found on real coefficient tables, not synthetic ones: exponent amplitude 1.09,
    # one merged region, +1.84 nats too high -- a factor of e^1.84 = 6.3, i.e. six
    # circuits -- and ACCEPTED, because a region covering everything leaves nothing
    # outside for the omitted-mass certificate to object to.  The certificate bounds
    # what is OUTSIDE the regions; it cannot see an error made INSIDE one.
    if float((mh - ml).sum()) >= 2.0 * np.pi:
        ml, mh = np.array([0.0]), np.array([2.0 * np.pi])
    else:
        span = np.minimum(mh - ml, 2.0 * np.pi)
        mh = ml + span
    covered = float(np.minimum(mh - ml, 2 * np.pi).sum())
    rep['n_phi_regions'] = int(ml.size)
    # exposed so the disjointness invariant can be ASSERTED rather than inferred from a
    # value comparison: overlapping regions double-count, and that error lives inside
    # the regions where the omitted-mass certificate is blind to it.
    rep['phi_regions'] = list(zip(ml.tolist(), mh.tolist()))

    parts = []
    for a, b in zip(ml, mh):
        # spacing <= sigma/4 for the SHARPEST mode inside this region, and never fewer
        # than 64 points across a full circuit -- a nearly-flat F has a huge sigma, so a
        # sigma-derived count alone would leave a wrapped region with a handful of nodes.
        inside_r = (p >= a - 1e-12) & (p <= b + 1e-12)
        sloc = sig[inside_r].min() if np.any(inside_r) else sig.min()
        n = int(np.ceil((b - a) / max(sloc, 1e-12) * 4)) + 1
        n = max(n, int(np.ceil(64 * (b - a) / (2 * np.pi))) + 1)
        n = max(16, min(2048, n))
        gp = np.linspace(a, b, n)
        Fv, _, _ = u_profile(C, np.mod(gp, 2 * np.pi), n_nodes=n_nodes)
        wq = np.full(n, (b - a) / (n - 1)); wq[0] *= 0.5; wq[-1] *= 0.5
        m = Fv.max()
        parts.append(m + np.log(np.sum(wq * np.exp(Fv - m))))
    parts = np.array(parts); m = parts.max()
    value = m + np.log(np.exp(parts - m).sum()) - 2.0 * np.log(2.0 * np.pi)

    # cover bound on phi: same slope-plus-curvature form as the 2-D outside bound, with
    # F'' bounded by M_(2,0) + M_(1,0)^2 (the variance term cannot exceed the square of
    # the first-derivative bound).
    t = np.linspace(0.0, 2.0 * np.pi, int(n_bound_grid), endpoint=False)
    inside = np.zeros(t.size, dtype=bool)
    step_t = 2.0 * np.pi / n_bound_grid
    for a, b in zip(ml, mh):
        d = _wrap(t - 0.5 * (a + b))
        inside |= np.abs(d) <= 0.5 * (b - a) - 0.5 * step_t
    T_out = float((~inside).sum()) * step_t
    if T_out <= 0.0 or covered >= 2 * np.pi:
        rep['margin'] = -np.inf
        return float(value), True, rep
    # BOUND F THROUGH g, NOT THROUGH F''.  The obvious route -- Taylor on F with
    # F'' <= M_(2,0) + M_(1,0)^2 -- is useless: that variance bound grows as the SQUARE
    # of the amplitude, so the remainder swamped everything (measured margins of +51 and
    # +1196 nats at amplitude 1265 and 4217, i.e. no bound at all).  Instead use
    #     F(phi) = log int du exp(g) <= log(2 pi) + sup_u g(phi, u),
    # and bound that supremum with the SAME slope-plus-M2 form already validated for the
    # 2-D outside bound, whose remainder grows only linearly in amplitude.
    phi_out = t[~inside]
    # The u axis SETS the covering radius here: at n_bound_grid = 512 the phi half-step
    # is 0.0061 while a 128-point u grid gives pi/128 = 0.0245, so u dominates r and the
    # remainder is four times larger than it need be.  Tie the u resolution to the same
    # knob so refining the bound refines BOTH axes; the cost is paid only on the
    # UNCOVERED phi, which is the small set by construction.
    n_ug = int(n_bound_grid)
    ug = np.linspace(0.0, 2.0 * np.pi, n_ug, endpoint=False)
    PH2, UU2 = np.meshgrid(phi_out, ug, indexing='ij')
    # HALF-DIAGONAL OF THE CELL, with no extra factor.  The grid spacings are step_t in
    # phi and 2*pi/128 in u, so the farthest a point can sit from its grid point is
    # hypot(step_t/2, pi/128).  An earlier revision multiplied that by a further 0.5, so
    # the Taylor remainder covered only half the cell and the "bound" was not one -- it
    # happened not to be violated in the trials run, which is exactly why an unjustified
    # constant is dangerous rather than harmless.
    r = np.sqrt((0.5 * step_t) ** 2 + (np.pi / n_ug) ** 2)
    g0 = eval_g(C, PH2.ravel(), UU2.ravel())
    gpv = eval_g(C, PH2.ravel(), UU2.ravel(), (1, 0))
    guv = eval_g(C, PH2.ravel(), UU2.ravel(), (0, 1))
    m2 = (derivative_bound(C, (2, 0)) + 2.0 * derivative_bound(C, (1, 1))
          + derivative_bound(C, (0, 2)))
    sup_g = float((g0 + np.hypot(gpv, guv) * r + 0.5 * m2 * r * r).max())
    sup_out = np.log(2.0 * np.pi) + sup_g
    rep['margin'] = float(np.log(T_out) + sup_out - 2.0 * np.log(2 * np.pi) - value)
    ok = rep['margin'] < tol_nats
    if not ok:
        rep['decline'] = 'phi omitted-mass bound too large'
    return float(value), bool(ok), rep
