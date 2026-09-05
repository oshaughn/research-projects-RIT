"""Algebraic stationary-point enumeration for real bivariate trig polynomials.

This is the NumPy/SciPy reference enumerator for the finite ``(phi, 2 psi)``
polynomials used by the higher-mode angle likelihood.  It does not sample an
angle grid.  Both derivatives are Laurent polynomials; after clearing the
Laurent powers and making a generic affine projection

    t = z + alpha w,  z = exp(i phi),  w = exp(i u),

their common zeros are the zeros of a Sylvester resultant in ``t``.  The
resultant is solved as a generalized polynomial eigenproblem.  Every isolated
stationary point is therefore in a finite algebraic candidate set whose size is
fixed by the harmonic support, not by the likelihood amplitude.

The input uses the RIFT coefficient-table convention: ``C[k, q + Q]`` stores
non-negative phi harmonics only and the real field is

    Re sum_{k=0..K,q=-Q..Q} (2 if k else 1) C[k,q] exp(i(k phi+q u)).

Floating-point algebra cannot make an unconditional exact-root promise.  The
enumeration certificate is consequently fail closed.  A certified result requires:

* the mixed-volume (BKK) number of isolated roots in ``(C*)^2``;
* nonsingular complex stationary Jacobians;
* an unambiguous unit-torus classification for every algebraic root; and
* identical torus stationary sets from two independent affine projections.

An exact or near degeneracy, a projection collision, a root close enough to the
unit torus that its membership is numerically ambiguous, or a root-count deficit
returns ``ok=False``.  Definite candidates remain available for a downstream
outside-cover bound; if that cannot certify their omitted impact the caller must
take its dense fallback.  No tolerance silently discards a possible real mode or
likelihood sample.  A certified call returns all isolated local maxima, including
exactly co-dominant symmetry-related maxima.

This module is deliberately host-side.  Generalized QZ, variable finite-root
counts, cross-projection matching, and fail-closed conditioning diagnostics do
not have an honest static-shape JAX transcription yet.  A JAX adapter should
consume a host-built fixed-capacity plan; it must not replace this solve with a
sampled phi grid and call that enumeration.
"""

from dataclasses import dataclass
from math import factorial

import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment


__all__ = [
    "StationaryPointEnumeration",
    "canonical_laurent_table",
    "stationary_mixed_volume",
    "enumerate_torus_maxima",
]


def _binomial(n, k):
    """Small exact binomial coefficient (keeps the reference Python-3.6 safe)."""
    return factorial(n) // (factorial(k) * factorial(n - k))


@dataclass(frozen=True)
class StationaryPointEnumeration:
    """Result of :func:`enumerate_torus_maxima`.

    ``stationary_points`` contains every verified isolated stationary candidate
    on the torus.  ``points`` is its negative-definite-Hessian subset.  When
    ``ok`` is false these arrays may be partial and are targeting data only: a
    caller may use them if an independent outside-cover bound passes, otherwise
    it must take a dense fallback.  ``report`` contains no fallback policy.
    """

    points: np.ndarray
    hessians: np.ndarray
    values: np.ndarray
    stationary_points: np.ndarray
    ok: bool
    report: dict


def canonical_laurent_table(C):
    """Return the full Hermitian Laurent table represented by a RIFT table.

    The returned array has shape ``(2*K+1, 2*Q+1)`` and indices ``(k+K,q+Q)``.
    It obeys ``A[-k,-q] = conj(A[k,q])`` by construction, including the
    potentially overlapping ``k=0`` contributions.
    """
    C = np.asarray(C, dtype=np.complex128)
    if C.ndim != 2 or C.shape[0] < 2 or C.shape[1] < 3 or C.shape[1] % 2 != 1:
        raise ValueError("C must have shape (K+1, 2*Q+1) with K,Q >= 1")
    if not np.all(np.isfinite(C.real) & np.isfinite(C.imag)):
        raise ValueError("C must be finite")
    K = C.shape[0] - 1
    Q = (C.shape[1] - 1) // 2
    A = np.zeros((2 * K + 1, 2 * Q + 1), dtype=np.complex128)
    for k in range(K + 1):
        weight = 1.0 if k == 0 else 2.0
        for iq, q in enumerate(range(-Q, Q + 1)):
            a = 0.5 * weight * C[k, iq]
            A[k + K, q + Q] += a
            A[-k + K, -q + Q] += np.conj(a)
    return A


def _convex_hull(points):
    """Integer monotone-chain hull, without a numerical geometry tolerance."""
    pts = sorted(set(tuple(map(int, p)) for p in points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _twice_polygon_area(points):
    hull = _convex_hull(points)
    if len(hull) < 3:
        return 0
    return abs(sum(
        hull[i][0] * hull[(i + 1) % len(hull)][1]
        - hull[(i + 1) % len(hull)][0] * hull[i][1]
        for i in range(len(hull))))


def _derivative_tables(A):
    K = (A.shape[0] - 1) // 2
    Q = (A.shape[1] - 1) // 2
    k = np.arange(-K, K + 1)[:, None]
    q = np.arange(-Q, Q + 1)[None, :]
    return 1j * k * A, 1j * q * A


def stationary_mixed_volume(C):
    """BKK count for the two stationary Laurent equations.

    This is the exact integer mixed volume of their Newton polygons.  It is the
    number of isolated roots in ``(C*)^2`` for a non-degenerate system, counted
    with multiplicity, and an upper bound otherwise.
    """
    A = canonical_laurent_table(C)
    F, G = _derivative_tables(A)
    K = (A.shape[0] - 1) // 2
    Q = (A.shape[1] - 1) // 2
    exponents = [(k, q) for k in range(-K, K + 1)
                 for q in range(-Q, Q + 1)]
    sf = [p for p, c in zip(exponents, F.ravel()) if c != 0.0]
    sg = [p for p, c in zip(exponents, G.ravel()) if c != 0.0]
    # A derivative may have a one-dimensional Newton polytope without making
    # the JOINT system one-dimensional: the separable field
    # cos(m phi)+cos(n u) has two transverse segments and 4mn isolated roots.
    if len(sf) < 2 or len(sg) < 2:
        return 0
    hf = _convex_hull(sf)
    hg = _convex_hull(sg)
    summed = [(a[0] + b[0], a[1] + b[1]) for a in hf for b in hg]
    twice = (_twice_polygon_area(summed)
             - _twice_polygon_area(hf) - _twice_polygon_area(hg))
    if twice < 0 or twice % 2:
        raise RuntimeError("stationary mixed volume was not a non-negative integer")
    return twice // 2


def _projected_polynomial(D, alpha):
    """Coefficients in ``w,t`` after ``z=t-alpha*w`` and Laurent clearing."""
    K = (D.shape[0] - 1) // 2
    Q = (D.shape[1] - 1) // 2
    # After multiplying by z^K w^Q, z-degree is <=2K and w-degree <=2Q.
    # Substitution can transfer all z degree to w.
    out = np.zeros((2 * (K + Q) + 1, 2 * K + 1), dtype=np.complex128)
    for iz in range(2 * K + 1):
        for iw in range(2 * Q + 1):
            c = D[iz, iw]
            if c == 0.0:
                continue
            for it in range(iz + 1):
                out[iw + iz - it, it] += (
                    c * _binomial(iz, it) * ((-alpha) ** (iz - it)))
    nz = np.nonzero(np.any(out != 0.0, axis=1))[0]
    if nz.size == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    out = out[nz[0]:nz[-1] + 1]
    scale = np.max(np.abs(out))
    return out / scale if scale > 0.0 else out


def _sylvester_matrix_polynomial(F, G):
    """Return ``S[j]`` for the Sylvester matrix polynomial ``sum t^j S[j]``."""
    if F.size == 0 or G.size == 0:
        raise ValueError("an identically-zero stationary equation is degenerate")
    m = F.shape[0] - 1
    n = G.shape[0] - 1
    if m < 1 or n < 1:
        raise ValueError("projection produced an equation independent of the eliminated variable")
    degree = max(F.shape[1], G.shape[1]) - 1
    size = m + n
    S = np.zeros((degree + 1, size, size), dtype=np.complex128)
    for shift in range(n):
        for j in range(m + 1):
            S[:F.shape[1], shift, shift + j] = F[j]
    for shift in range(m):
        for j in range(n + 1):
            S[:G.shape[1], n + shift, shift + j] = G[j]
    nz = np.nonzero(np.any(S != 0.0, axis=(1, 2)))[0]
    if nz.size < 2:
        raise ValueError("constant or zero resultant pencil")
    return S[:nz[-1] + 1]


def _linearize_matrix_polynomial(S):
    """First companion linearization ``L0 - t L1`` of ``sum S[j] t^j``."""
    degree = S.shape[0] - 1
    size = S.shape[1]
    L0 = np.zeros((degree * size, degree * size), dtype=np.complex128)
    L1 = np.zeros_like(L0)
    eye = np.eye(size, dtype=np.complex128)
    for i in range(degree - 1):
        L0[i * size:(i + 1) * size, (i + 1) * size:(i + 2) * size] = eye
        L1[i * size:(i + 1) * size, i * size:(i + 1) * size] = eye
    last = slice((degree - 1) * size, degree * size)
    for j in range(degree):
        L0[last, j * size:(j + 1) * size] = -S[j]
    L1[last, (degree - 1) * size:degree * size] = S[degree]
    return L0, L1


def _eval_laurent(D, z, w):
    K = (D.shape[0] - 1) // 2
    Q = (D.shape[1] - 1) // 2
    zp = z ** np.arange(-K, K + 1)
    wp = w ** np.arange(-Q, Q + 1)
    return np.einsum("ij,i,j->", D, zp, wp)


def _laurent_scale(D, z, w):
    K = (D.shape[0] - 1) // 2
    Q = (D.shape[1] - 1) // 2
    zp = np.abs(z) ** np.arange(-K, K + 1)
    wp = np.abs(w) ** np.arange(-Q, Q + 1)
    return float(np.einsum("ij,i,j->", np.abs(D), zp, wp))


def _laurent_order(A, a, b):
    K = (A.shape[0] - 1) // 2
    Q = (A.shape[1] - 1) // 2
    k = np.arange(-K, K + 1)[:, None]
    q = np.arange(-Q, Q + 1)[None, :]
    return ((1j * k) ** int(a)) * ((1j * q) ** int(b)) * A


def _laurent_newton(A, z, w, iterations=30):
    """Newton in complex angle coordinates, avoiding cleared-power scaling."""
    Dp = _laurent_order(A, 1, 0)
    Du = _laurent_order(A, 0, 1)
    Dpp = _laurent_order(A, 2, 0)
    Dpu = _laurent_order(A, 1, 1)
    Duu = _laurent_order(A, 0, 2)
    for _ in range(int(iterations)):
        gradient = np.array([_eval_laurent(Dp, z, w),
                             _eval_laurent(Du, z, w)])
        H = np.array([[_eval_laurent(Dpp, z, w),
                       _eval_laurent(Dpu, z, w)],
                      [_eval_laurent(Dpu, z, w),
                       _eval_laurent(Duu, z, w)]])
        if not np.all(np.isfinite(H)) or np.linalg.cond(H) > 1e16:
            return z, w, np.inf, 0.0, False
        try:
            step = np.linalg.solve(H, -gradient)
        except np.linalg.LinAlgError:
            return z, w, np.inf, 0.0, False
        if not np.all(np.isfinite(step)) or np.max(np.abs(step)) > 4.0:
            return z, w, np.inf, 0.0, False
        z *= np.exp(1j * step[0])
        w *= np.exp(1j * step[1])
        if (not np.isfinite(z) or not np.isfinite(w)
                or abs(z) < 1e-12 or abs(w) < 1e-12
                or max(abs(z), abs(w)) > 1e12):
            return z, w, np.inf, 0.0, False
        if np.max(np.abs(step)) < 5e-14:
            break
    rp = abs(_eval_laurent(Dp, z, w)) / max(_laurent_scale(Dp, z, w), 1e-300)
    ru = abs(_eval_laurent(Du, z, w)) / max(_laurent_scale(Du, z, w), 1e-300)
    H = np.array([[_eval_laurent(Dpp, z, w), _eval_laurent(Dpu, z, w)],
                  [_eval_laurent(Dpu, z, w), _eval_laurent(Duu, z, w)]])
    cond = float(np.linalg.cond(H)) if np.all(np.isfinite(H)) else np.inf
    return z, w, max(float(rp), float(ru)), 1.0 / cond, True


def _solution_distance(a, b):
    return max(abs(a[0] - b[0]) / max(1.0, abs(a[0]), abs(b[0])),
               abs(a[1] - b[1]) / max(1.0, abs(a[1]), abs(b[1])))


def _one_projection(A, alpha, expected, root_tol, jacobian_rcond_min):
    Dp, Du = _derivative_tables(A)
    F = _projected_polynomial(Dp, alpha)
    G = _projected_polynomial(Du, alpha)
    report = {"alpha": alpha, "expected_roots": int(expected),
              "pencil_size": 0, "finite_eigenvalues": 0,
              "verified_complex_roots": 0, "min_jacobian_rcond": 0.0,
              "decline": None}
    try:
        S = _sylvester_matrix_polynomial(F, G)
        L0, L1 = _linearize_matrix_polynomial(S)
        report["pencil_size"] = int(L0.shape[0])
        eig, left, right = linalg.eig(
            L0, L1, left=True, right=True, homogeneous_eigvals=True,
            check_finite=False)
    except (ValueError, linalg.LinAlgError) as exc:
        report["decline"] = "singular resultant construction: %s" % exc
        return [], report

    aa, bb = eig
    pair_scale = np.hypot(np.abs(aa), np.abs(bb))
    finite = ((np.abs(bb) > 100.0 * np.finfo(float).eps * pair_scale)
              & np.isfinite(aa) & np.isfinite(bb))
    report["finite_eigenvalues"] = int(np.count_nonzero(finite))
    bnorm = max(float(np.linalg.norm(L1, ord="fro")), 1e-300)
    anorm = max(float(np.linalg.norm(L0, ord="fro")), 1e-300)
    solutions = []
    jac_rconds = []
    eig_rconds = []
    eig_backward = []
    for idx in np.nonzero(finite)[0]:
        t0 = aa[idx] / bb[idx]
        if not np.isfinite(t0):
            continue
        St = sum(S[j] * (t0 ** j) for j in range(S.shape[0]))
        y, x = left[:, idx], right[:, idx]
        eig_rc = abs(np.vdot(y, L1 @ x)) / max(
            np.linalg.norm(y) * np.linalg.norm(x) * bnorm, 1e-300)
        eig_be = np.linalg.norm(L0 @ x - t0 * (L1 @ x)) / max(
            (anorm + abs(t0) * bnorm) * np.linalg.norm(x),
            1e-300)
        # The right null vector is a geometric sequence in the eliminated
        # variable for a simple fibre.  A projection collision makes its null
        # space multidimensional; that is not guessed through with extra seeds
        # but exposed by the BKK count / second-projection checks below.
        # In this companion linearization the first block of the generalized
        # eigenvector is already a null vector of S(t).  It is jointly computed
        # with t by QZ and is materially more accurate than recomputing the
        # smallest singular vector at a rounded eigenvalue.  Retain SVD only as
        # a fallback for a zero first block.
        v = right[:S.shape[1], idx]
        if np.linalg.norm(v) == 0.0:
            try:
                _, _, vh = np.linalg.svd(St)
                v = vh[-1].conj()
            except np.linalg.LinAlgError:
                continue
        denom = np.vdot(v[:-1], v[:-1])
        if abs(denom) == 0.0:
            continue
        w0 = np.vdot(v[:-1], v[1:]) / denom
        z0 = t0 - alpha * w0
        z, w, residual, jac_rcond, converged = _laurent_newton(A, z0, w0)
        if not converged or residual > root_tol:
            continue
        if (not np.isfinite(z) or not np.isfinite(w)
                or abs(z) < 1e-10 or abs(w) < 1e-10):
            continue
        rp = abs(_eval_laurent(Dp, z, w)) / max(
            _laurent_scale(Dp, z, w), 1e-300)
        ru = abs(_eval_laurent(Du, z, w)) / max(
            _laurent_scale(Du, z, w), 1e-300)
        if max(rp, ru) > 10.0 * root_tol:
            continue
        candidate = (z, w, residual, jac_rcond, float(eig_rc))
        close = [_solution_distance(candidate, old) for old in solutions]
        if not close or min(close) > 5e-8:
            solutions.append(candidate)
            jac_rconds.append(jac_rcond)
            eig_rconds.append(float(eig_rc))
            eig_backward.append(float(eig_be))

    report["verified_complex_roots"] = len(solutions)
    report["min_jacobian_rcond"] = float(min(jac_rconds, default=0.0))
    report["min_pencil_eigen_rcond"] = float(min(eig_rconds, default=0.0))
    report["max_pencil_backward_error"] = float(max(eig_backward, default=np.inf))
    if min(jac_rconds, default=0.0) < jacobian_rcond_min:
        report["decline"] = "singular or ill-conditioned stationary Jacobian"
        return solutions, report
    if len(solutions) != expected:
        report["decline"] = "BKK root-count mismatch (%d != %d)" % (
            len(solutions), expected)
        return solutions, report
    if max(eig_backward, default=np.inf) > root_tol:
        report["decline"] = "resultant eigenproblem failed its backward-error check"
        return solutions, report
    return solutions, report


def _angle_eval(A, points, order=(0, 0)):
    K = (A.shape[0] - 1) // 2
    Q = (A.shape[1] - 1) // 2
    k = np.arange(-K, K + 1)[:, None]
    q = np.arange(-Q, Q + 1)[None, :]
    a, b = order
    factor = (1j * k) ** a * (1j * q) ** b
    phi = points[:, 0, None, None]
    u = points[:, 1, None, None]
    phase = np.exp(1j * (phi * k[None] + u * q[None]))
    return np.real(np.sum(phase * factor[None] * A[None], axis=(1, 2)))


def _torus_points(solutions, torus_on_tol, torus_off_tol):
    """Classify roots using the real-field reciprocal-conjugate involution.

    A torus root is a fixed point of ``(z,w)->(1/conj(z),1/conj(w))``.
    A genuinely complex root has a distinct partner.  This is stronger than an
    ``abs(abs(z)-1) < tol`` filter: a close off-torus pair is declared ambiguous
    and declines the whole solve instead of being rounded onto or away from the
    torus.
    """
    points = []
    ambiguous = 0
    roots = [(s[0], s[1]) for s in solutions]
    for i, (z, w) in enumerate(roots):
        involution = (1.0 / np.conj(z), 1.0 / np.conj(w))
        distance = np.asarray([_solution_distance(involution, other)
                               for other in roots])
        order = np.argsort(distance)
        nearest = int(order[0])
        match_error = float(distance[nearest])
        self_error = float(distance[i])
        if nearest == i and match_error <= torus_on_tol:
            points.append((np.mod(np.angle(z), 2.0 * np.pi),
                           np.mod(np.angle(w), 2.0 * np.pi)))
        elif (nearest != i and match_error <= torus_on_tol
              and self_error >= torus_off_tol):
            # A resolved non-real reciprocal-conjugate pair: safely off torus.
            continue
        else:
            ambiguous += 1
    return np.asarray(points, dtype=float).reshape((-1, 2)), ambiguous


def _periodic_assignment_distance(a, b):
    if len(a) != len(b):
        return np.inf
    if len(a) == 0:
        return 0.0
    delta = (a[:, None, :] - b[None, :, :] + np.pi) % (2.0 * np.pi) - np.pi
    cost = np.linalg.norm(delta, axis=-1)
    row, col = linear_sum_assignment(cost)
    return float(np.max(cost[row, col]))


def _dedupe_periodic(points, tolerance=1e-7):
    keep = []
    for point in np.asarray(points, dtype=float).reshape((-1, 2)):
        if not keep:
            keep.append(point)
            continue
        delta = (np.asarray(keep) - point + np.pi) % (2.0 * np.pi) - np.pi
        if np.min(np.linalg.norm(delta, axis=1)) > tolerance:
            keep.append(point)
    return np.asarray(keep, dtype=float).reshape((-1, 2))


def enumerate_torus_maxima(
        C, *, projections=(0.371 + 0.193j, -0.227 + 0.419j),
        root_tol=2e-9, jacobian_rcond_min=2e-10,
        torus_on_tol=2e-7, torus_off_tol=2e-5,
        projection_match_tol=2e-6):
    """Enumerate every isolated local maximum of ``g(phi,u)`` algebraically.

    Certification is conditional on a regular zero-dimensional stationary
    system.  ``ok=False`` is the promised behavior for exact/near stationary
    degeneracy, ill-conditioned resultants, ambiguous torus membership, or
    disagreement between the independent projections.  Such a result may carry
    definite best-effort targets, but never claims them as complete.
    """
    C = np.asarray(C, dtype=np.complex128)
    empty_p = np.zeros((0, 2), dtype=float)
    empty_h = np.zeros((0, 2, 2), dtype=float)
    empty_v = np.zeros(0, dtype=float)
    report = {"ok": False, "mixed_volume": 0, "n_stationary": 0,
              "n_maxima": 0, "projections": [], "decline": None}
    try:
        A = canonical_laurent_table(C)
        expected = stationary_mixed_volume(C)
    except (ValueError, RuntimeError) as exc:
        report["decline"] = str(exc)
        return StationaryPointEnumeration(
            empty_p, empty_h, empty_v, empty_p, False, report)
    report["mixed_volume"] = int(expected)
    if expected <= 0:
        report["decline"] = "stationary system is not two-dimensional"
        return StationaryPointEnumeration(
            empty_p, empty_h, empty_v, empty_p, False, report)
    scale = float(np.max(np.abs(A)))
    if not scale > 0.0:
        report["decline"] = "constant field has a positive-dimensional stationary set"
        return StationaryPointEnumeration(
            empty_p, empty_h, empty_v, empty_p, False, report)
    A = A / scale

    torus_sets = []
    complete_sets = []
    for alpha in projections:
        solutions, one = _one_projection(
            A, complex(alpha), expected, float(root_tol),
            float(jacobian_rcond_min))
        report["projections"].append(one)
        points, ambiguous = _torus_points(
            solutions, float(torus_on_tol), float(torus_off_tol))
        one["torus_roots"] = int(len(points))
        one["ambiguous_torus_roots"] = int(ambiguous)
        if ambiguous and one["decline"] is None:
            one["decline"] = "ambiguous unit-torus root"
        if len(points):
            torus_sets.append(points)
        if one["decline"] is None:
            complete_sets.append(points)

    certified = False
    if len(complete_sets) >= 2:
        mismatch = _periodic_assignment_distance(complete_sets[0], complete_sets[1])
        report["projection_match_error"] = mismatch
        certified = bool(np.isfinite(mismatch) and mismatch <= projection_match_tol)
        if not certified:
            report["decline"] = "independent projections disagree on torus roots"
    else:
        report["decline"] = "fewer than two algebraically complete projections"
    if not torus_sets:
        return StationaryPointEnumeration(
            empty_p, empty_h, empty_v, empty_p, False, report)

    # On an uncertified solve keep the UNION of every definitely-on-torus root.
    # A downstream cover bound can safely validate this best-effort targeting
    # set; returning no candidates would force a dense fallback unnecessarily.
    stationary = _dedupe_periodic(np.concatenate(torus_sets, axis=0))
    # Refine in real angles.  Algebra supplies all seeds; Newton only restores
    # unit-modulus/roundoff accuracy and never supplies completeness.
    real_ok = np.ones(len(stationary), dtype=bool)
    for _ in range(8):
        gp = _angle_eval(A, stationary, (1, 0))
        gu = _angle_eval(A, stationary, (0, 1))
        gpp = _angle_eval(A, stationary, (2, 0))
        gpu = _angle_eval(A, stationary, (1, 1))
        guu = _angle_eval(A, stationary, (0, 2))
        for i in range(len(stationary)):
            H = np.array([[gpp[i], gpu[i]], [gpu[i], guu[i]]])
            try:
                step = np.linalg.solve(H, -np.array([gp[i], gu[i]]))
            except np.linalg.LinAlgError:
                real_ok[i] = False
                continue
            if not np.all(np.isfinite(step)) or np.linalg.norm(step) > 1.0:
                real_ok[i] = False
                continue
            stationary[i] = np.mod(stationary[i] + step, 2.0 * np.pi)

    stationary = _dedupe_periodic(stationary[real_ok])
    if len(stationary) == 0:
        if report["decline"] is None:
            report["decline"] = "no usable real stationary candidates"
        return StationaryPointEnumeration(
            empty_p, empty_h, empty_v, empty_p, False, report)

    gp = _angle_eval(A, stationary, (1, 0))
    gu = _angle_eval(A, stationary, (0, 1))
    gpp = _angle_eval(A, stationary, (2, 0))
    gpu = _angle_eval(A, stationary, (1, 1))
    guu = _angle_eval(A, stationary, (0, 2))
    hessian = np.stack((np.stack((gpp, gpu), axis=-1),
                        np.stack((gpu, guu), axis=-1)), axis=-2)
    eig_h = np.linalg.eigvalsh(hessian)
    hscale = max(float(np.max(np.abs(eig_h))), 1e-300)
    grad_resid = np.hypot(gp, gu)
    report["max_stationary_residual"] = float(np.max(grad_resid, initial=0.0))
    usable = ((np.min(np.abs(eig_h), axis=1) > jacobian_rcond_min * hscale)
              & (grad_resid <= 5e-8))
    if not np.all(usable):
        certified = False
        report["decline"] = "degenerate or unconverged real stationary candidate"
    stationary = stationary[usable]
    hessian = hessian[usable]
    eig_h = eig_h[usable]

    is_max = np.all(eig_h < 0.0, axis=1)
    maxima = stationary[is_max]
    max_h = hessian[is_max] * scale
    values = _angle_eval(A, maxima, (0, 0)) * scale
    report["n_stationary"] = int(len(stationary))
    report["n_maxima"] = int(len(maxima))
    report["ok"] = bool(certified)
    return StationaryPointEnumeration(
        maxima, max_h, values, stationary, bool(certified), report)
