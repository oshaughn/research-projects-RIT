"""EXACT 2-D stationary enumeration for the joint (phi, u) angle exponent.

SUPERSEDED ON REBASE, AND THIS MODULE SHOULD NOT SURVIVE THE MERGE.  PR 252 adds
``bivariate_trig_stationary.py``, which solves the same problem and solves it better: a
generic affine projection into a generalized eigenproblem rather than a resultant on the
roots of unity, plus four checks this module does not have -- the BKK mixed-volume root
count, nonsingular complex Jacobians, an unambiguous unit-torus classification, and
agreement between two independent projections -- and an ``ok`` flag that fails closed on
any of them.  This module has no ``ok`` at all and returns silently empty on a degenerate
input.

It is here because 247 has to stand on its own branch off ``rift_O4d`` while 252 is open
against a different base.  Carrying BOTH after 252 lands is the outcome review called out,
together with the contradiction it creates -- 252's header in ``joint_anglemarg_peaklocal``
says both-axis algebraic localization is not attempted, which 247 then contradicts in the
same file.  On rebase: delete this module, point ``phi_seeds_algebraic`` at
``bivariate_trig_stationary``, keep this file's tests as tests of that one, and re-collect
the CI gate counts rather than taking either branch's number.

WHY THIS EXISTS.  ``enumerate_modes`` is exact in u and GRIDDED in phi -- it seeds from
``linspace(0, 2pi, n_phi)`` -- and the JAX twin's ``phi_local_lnI`` is worse: it iterates on
the maxima of ``F(phi) = log int du exp(g)``, a log-integral with no completeness warrant,
from ``PHI_SEEDS`` arbitrary seeds.  Neither can say it found everything.

But g ITSELF carries the warrant, and it is the same one psi has.  The orbital phase enters
the modes as ``e^{-i m phi}``, so ``A`` reaches phi-harmonic ``m_max`` and ``B``, quadratic
in the waveform, reaches ``2 m_max``.  The combined table's ``k_max = KP-1 = 2 m_max`` is
therefore EXACT, fixed by the mode content -- knowing m tells you the phi content, exactly
as knowing the polarization tells you psi is degree 2.

VALIDATED (10 draws x 5 mode orders KP = 3,5,7,9,13, amplitudes 1e2 and 1e4): every maximum
found by a dense n_phi=256 grid is recovered, worst separation 3.0e-06, and no spurious
extra maxima.  Stationary-point counts stay inside the mixed-volume degree bound.

EXACT 2-D stationary enumeration for g on the torus, via the Sylvester resultant.

g = sum_{k,q} D[k,q] z^k w^q  with D[-k,-q] = conj(D[k,q]) (g real), z=e^{i phi}, w=e^{i u}.
The stationary system dg/dphi = dg/du = 0 becomes, after clearing negative powers,

    P1(z,w) = sum (i k) D[k,q] z^{k+K} w^{q+Q}
    P2(z,w) = sum (i q) D[k,q] z^{k+K} w^{q+Q}

both of bidegree (2K, 2Q).  Eliminating w by the Sylvester resultant gives a univariate
polynomial in z of degree <= (2K)(2Q)(2) = 16 k_max -- exactly the mixed-volume bound, and
fixed by the MODE CONTENT since K = k_max = 2 m_max.

det S(z) is recovered WITHOUT symbolic algebra: it is a polynomial of known degree, so
evaluating it on N > deg roots of unity and inverse-FFTing gives its coefficients exactly.
Every shape is static given the table -- the property JAX needs.
"""
import numpy as np


def laurent_D(C):
    """Hermitian Laurent coefficients D[k+K, q+Q] of g from the (KP, 2KS+1) table."""
    KP = C.shape[0]; KS = (C.shape[1] - 1) // 2
    K = KP - 1; Q = KS
    D = np.zeros((2 * K + 1, 2 * Q + 1), dtype=complex)
    for k in range(KP):
        wk = 1.0 if k == 0 else 2.0
        for qi in range(2 * KS + 1):
            q = qi - KS
            D[k + K, q + Q] += 0.5 * wk * C[k, qi]
            D[-k + K, -q + Q] += 0.5 * wk * np.conj(C[k, qi])
    return D, K, Q


def _sylvester_det_on_circle(D, K, Q, N):
    """det of the w-Sylvester matrix of (P1,P2), evaluated at N roots of unity in z."""
    kk = np.arange(-K, K + 1)[:, None]
    qq = np.arange(-Q, Q + 1)[None, :]
    A1 = (1j * kk) * D            # dg/dphi coefficients
    A2 = (1j * qq) * D            # dg/du
    zs = np.exp(2j * np.pi * np.arange(N) / N)
    # coefficients in w (degree 2Q) after substituting each z
    zpow = zs[:, None] ** np.arange(-K, K + 1)[None, :]      # (N, 2K+1)
    c1 = zpow @ A1                                            # (N, 2Q+1)
    c2 = zpow @ A2
    n1 = n2 = 2 * Q
    S = np.zeros((N, n1 + n2, n1 + n2), dtype=complex)
    for r in range(n2):
        S[:, r, r:r + n1 + 1] = c1[:, ::-1]
    for r in range(n1):
        S[:, n2 + r, r:r + n2 + 1] = c2[:, ::-1]
    return np.linalg.det(S)


def stationary_points(C, newton_iters=24, res_tol=1e-8):
    """All (phi, u) with dg/dphi = dg/du = 0.  Algebraic COVER, Newton PRECISION.

    Two things the first version got wrong, both about conditioning rather than algebra:

    1. SCALE FIRST.  The resultant's coefficients are products of eight Sylvester entries,
       so they grow as amplitude^8 -- 1e32 at amplitude 1e4, measured -- and degree-64
       root-finding at that dynamic range returns roots ~1e-2 off the true ones.  The
       stationary set is INVARIANT under g -> g/s, so normalising the table first costs
       nothing and fixes the conditioning.
    2. THE RESULTANT LOCATES, IT DOES NOT POLISH.  Its job is a COMPLETE seed set -- that
       is the property 32 arbitrary seeds cannot claim -- and 2-D Newton then refines each
       to machine precision.  The earlier version paired every z-root with every |w|=1 root
       of dg/du without requiring the root be SHARED with dg/dphi, so most candidates were
       not stationary at all; the residual filter below is what selects the shared ones.
    """
    C = np.asarray(C, dtype=complex)
    scale = float(np.max(np.abs(C)))
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros((0, 2))
    C = C / scale
    D, K, Q = laurent_D(C)
    deg = (2 * K) * (2 * Q) * 2
    N = 1
    while N <= deg + 2:
        N *= 2
    vals = _sylvester_det_on_circle(D, K, Q, N)
    # det S(z) is a LAURENT polynomial in z spanning z^-h .. z^+h with h = deg/2: the
    # Sylvester entries themselves carry z^-K..z^+K because the negative powers were never
    # cleared on the z side.  ifft returns a_j at index j mod N, so the negative half lives
    # at the TOP of the array.  Truncating to coeffs[:deg+1] silently discarded it and left
    # a polynomial with no roots on the circle -- the whole enumeration returned nothing.
    # Multiply through by z^h (a shift, which cannot move a root) to clear the negatives.
    h = deg // 2
    # COEFFICIENTS COME FROM fft/N, NOT ifft.  The determinant is sampled at
    # z_k = exp(+2 pi i k / N), so for f(z) = sum_j a_j z^j,
    #     fft(vals)[m] = sum_k sum_j a_j e^{2pi i jk/N} e^{-2pi i mk/N} = N a_m,
    # while ifft(vals)[n] = a_{-n} -- the REVERSED polynomial, whose roots are the
    # reciprocals 1/z.  On the unit circle 1/z = conj(z), so the seeds came out at -phi.
    # This shipped, and the completeness validation PASSED anyway: 256 Newton starts
    # scattered over the torus recover the maxima wherever they begin, so the construction
    # was working as a multi-start SEARCH while claiming to be an enumeration.  Measured
    # after external review: reconstructing the sampled determinant from the ifft
    # coefficients gives relative error 0.98-1.00; from fft/N it gives 1.3e-15.
    raw = np.fft.fft(vals) / N
    coeffs = np.concatenate([raw[N - h:], raw[:h + 1]])      # ascending, j = -h .. +h
    nz = np.nonzero(np.abs(coeffs) > 1e-9 * max(np.abs(coeffs).max(), 1e-300))[0]
    if nz.size < 2:
        return np.zeros((0, 2))
    c = coeffs[nz[0]:nz[-1] + 1][::-1]            # numpy.roots wants descending
    zr = np.roots(c)
    # NO |z| = 1 FILTER.  This module's own rule for the u axis is that all roots are
    # returned as SEEDS -- an on-circle tolerance on an ill-conditioned root-find drops
    # real solutions, and at degree 128 a genuine stationary point was measured 2.9e-2 off
    # the circle and discarded by a 1e-3 test.  Take every root, read phi off its argument,
    # and let the post-Newton residual decide what was real.  Same reason, same rule.
    on = zr[np.isfinite(zr)]
    if on.size == 0:
        return np.zeros((0, 2))
    out = []
    kk = np.arange(-K, K + 1)[:, None]; qq = np.arange(-Q, Q + 1)[None, :]
    for z in on:
        phi = np.angle(z)
        zp = z ** np.arange(-K, K + 1)
        cu = zp @ ((1j * qq) * D)                 # dg/du coefficients in w
        idx = np.nonzero(np.abs(cu) > 1e-12 * max(np.abs(cu).max(), 1e-300))[0]
        if idx.size < 2:
            continue
        wr = np.roots(cu[idx[0]:idx[-1] + 1][::-1])
        for w in wr[np.isfinite(wr)]:                 # likewise: no |w| = 1 filter
            out.append((np.mod(phi, 2 * np.pi), np.mod(np.angle(w), 2 * np.pi)))
    if not out:
        return np.zeros((0, 2))
    P = np.array(out, dtype=float)

    # POLISH: 2-D Newton on the normalised table, same trust region as the reference.
    def d(a, b):
        kkk = np.arange(-K, K + 1)[None, :, None]; qqq = np.arange(-Q, Q + 1)[None, None, :]
        E = np.exp(1j * (P[:, 0][:, None, None] * kkk + P[:, 1][:, None, None] * qqq))
        return np.real((E * ((1j * kkk) ** a) * ((1j * qqq) ** b) * D[None]).sum((1, 2)))
    for _ in range(int(newton_iters)):
        gp, gu = d(1, 0), d(0, 1)
        gpp, guu, gpu = d(2, 0), d(0, 2), d(1, 1)
        det = gpp * guu - gpu * gpu
        okd = np.abs(det) > 1e-300
        dp = np.where(okd, -(guu * gp - gpu * gu) / np.where(okd, det, 1.0), 0.0)
        du = np.where(okd, -(-gpu * gp + gpp * gu) / np.where(okd, det, 1.0), 0.0)
        st = np.hypot(dp, du)
        sc = np.where(st > 0.5, 0.5 / np.maximum(st, 1e-300), 1.0)
        P[:, 0] = np.mod(P[:, 0] + dp * sc, 2 * np.pi)
        P[:, 1] = np.mod(P[:, 1] + du * sc, 2 * np.pi)

    # keep only points that are ACTUALLY stationary (the shared root of both equations)
    m1 = float(np.abs((1j * kk) * D).sum() + np.abs((1j * qq) * D).sum())
    keep = np.hypot(d(1, 0), d(0, 1)) <= res_tol * max(m1, 1e-300)
    P = P[keep]
    if P.shape[0] == 0:
        return P
    sel = [0]
    for i in range(1, P.shape[0]):
        dd = np.hypot(np.abs(((P[i, 0] - P[sel, 0] + np.pi) % (2 * np.pi)) - np.pi),
                      np.abs(((P[i, 1] - P[sel, 1] + np.pi) % (2 * np.pi)) - np.pi))
        if dd.min() > 1e-6:
            sel.append(i)
    return P[sel]
