"""The phi axis has an algebraic warrant after all -- via g, not via F."""
import numpy as np
import pytest

from RIFT.likelihood import joint_angle_algebraic as ALG
from RIFT.likelihood import joint_angle_peak_local as JN


def _maxima(C, P):
    if not P.shape[0]:
        return P
    gpp = JN.eval_g(C, P[:, 0], P[:, 1], (2, 0))
    guu = JN.eval_g(C, P[:, 0], P[:, 1], (0, 2))
    gpu = JN.eval_g(C, P[:, 0], P[:, 1], (1, 1))
    return P[(gpp < 0) & (gpp * guu - gpu * gpu > 0)]


def _table(rng, KP, amp, KS=2):
    C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
    return C * (amp / np.sum(np.abs(C)))


@pytest.mark.parametrize("KP", [3, 5, 7, 9, 13])
def test_algebraic_cover_recovers_every_maximum_a_dense_grid_finds(KP):
    """COMPLETENESS, which is the entire point.  A grid can only claim what its density
    happens to catch; the resultant enumerates the stationary system of ``g``, whose degree
    is fixed by the mode content (``k_max = 2 m_max``).  Measured against a dense n_phi=256
    grid: nothing unmatched at any mode order, worst separation 3.0e-06."""
    rng = np.random.default_rng(101)
    worst = 0.0
    for amp in (1e2, 1e4):
        for _ in range(3):
            C = _table(rng, KP, amp)
            M = _maxima(C, ALG.stationary_points(C))
            G, _ = JN.enumerate_modes(C, n_phi=256)
            if G.shape[0] == 0:
                continue
            assert M.shape[0] > 0, "algebraic cover returned nothing where the grid found maxima"
            d = np.hypot(
                np.abs(((G[:, None, 0] - M[None, :, 0] + np.pi) % (2 * np.pi)) - np.pi),
                np.abs(((G[:, None, 1] - M[None, :, 1] + np.pi) % (2 * np.pi)) - np.pi),
            ).min(axis=1)
            assert (d <= 1e-4).all(), (KP, amp, float(d.max()))
            worst = max(worst, float(d.max()))
    assert worst < 1e-4, worst


def test_no_on_circle_tolerance_is_applied_to_the_roots():
    """This module's rule for the u axis -- all roots are seeds, no |z|=1 filter -- applies
    here too, and was violated in the first version.  At degree 128 a genuinely stationary
    maximum was measured 2.9e-02 off the unit circle and discarded by a 1e-3 test; the
    residual after Newton is what decides, never the modulus.  Non-vacuous: a table whose
    roots are ill-conditioned must still yield every maximum."""
    import inspect
    assert "tol_circle" not in inspect.signature(ALG.stationary_points).parameters
    rng = np.random.default_rng(101)
    C = _table(rng, 9, 1e4)
    M = _maxima(C, ALG.stationary_points(C))
    G, _ = JN.enumerate_modes(C, n_phi=256)
    d = np.hypot(
        np.abs(((G[:, None, 0] - M[None, :, 0] + np.pi) % (2 * np.pi)) - np.pi),
        np.abs(((G[:, None, 1] - M[None, :, 1] + np.pi) % (2 * np.pi)) - np.pi),
    ).min(axis=1)
    assert (d <= 1e-4).all(), float(d.max())


def test_stationary_count_stays_inside_the_mode_order_bound():
    """The count is bounded by the mixed volume of the (2 k_max, 2 Q) system -- a property
    of the MODE CONTENT, which is what makes this an enumeration rather than a search."""
    rng = np.random.default_rng(5)
    for KP in (3, 5, 7, 9):
        KS = 2
        bound = (2 * (KP - 1)) * (2 * KS) * 2
        for amp in (1e2, 1e4):
            P = ALG.stationary_points(_table(rng, KP, amp, KS))
            assert P.shape[0] <= bound, (KP, amp, P.shape[0], bound)


def test_the_resultant_coefficients_reproduce_the_sampled_determinant():
    """DIRECT test of the elimination, not downstream agreement with a grid.

    External review P1.  The determinant is sampled at ``z_k = exp(+2 pi i k / N)``, so for
    ``f(z) = sum_j a_j z^j`` the forward transform gives ``fft(vals)[m] = N a_m``, while
    ``ifft(vals)[n] = a_{-n}`` -- the REVERSED polynomial, whose roots are the reciprocals
    ``1/z``.  On the unit circle ``1/z = conj(z)``, so the shipped code seeded at ``-phi``.

    IT PASSED ITS COMPLETENESS TEST ANYWAY, which is why this test exists: 256 Newton starts
    scattered over the torus recover the maxima wherever they begin, so the construction
    worked as a multi-start SEARCH while claiming to be an enumeration.  Agreement with a
    grid could not see the difference.  Reconstruction can: 0.98-1.00 relative error before
    the fix, ~1e-15 after.
    """
    for KP in (3, 5, 9):
        KS = 2
        rng = np.random.default_rng(3)
        C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
        C = C / np.max(np.abs(C))
        D, K, Q = ALG.laurent_D(C)
        deg = (2 * K) * (2 * Q) * 2
        N = 1
        while N <= deg + 2:
            N *= 2
        vals = ALG._sylvester_det_on_circle(D, K, Q, N)
        zs = np.exp(2j * np.pi * np.arange(N) / N)
        h = deg // 2
        raw = np.fft.fft(vals) / N
        co = np.concatenate([raw[N - h:], raw[:h + 1]])
        rec = np.array([sum(co[i] * z ** (i - h) for i in range(len(co))) for z in zs])
        rel = np.abs(rec - vals).max() / np.abs(vals).max()
        assert rel < 1e-10, (KP, rel)


def test_raw_roots_locate_the_maxima_before_newton_touches_them():
    """The ENUMERATION property, which agreement-after-Newton cannot demonstrate.

    If the algebraic step is really an enumeration, the resultant's on-circle roots already
    sit at the stationary ``phi`` -- Newton only polishes.  If it is a multi-start dressed
    up, the roots sit somewhere else and Newton does the finding.  That is exactly what the
    FFT-sign defect produced (roots at ``-phi``), and only this test distinguishes them.
    """
    for KP in (3, 5):
        KS = 2
        rng = np.random.default_rng(3)
        C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
        C = C * (1e3 / np.sum(np.abs(C)))
        G, _ = JN.enumerate_modes(C, n_phi=256)
        if G.shape[0] == 0:
            continue
        Cs = C / np.max(np.abs(C))
        D, K, Q = ALG.laurent_D(Cs)
        deg = (2 * K) * (2 * Q) * 2
        N = 1
        while N <= deg + 2:
            N *= 2
        raw = np.fft.fft(ALG._sylvester_det_on_circle(D, K, Q, N)) / N
        h = deg // 2
        co = np.concatenate([raw[N - h:], raw[:h + 1]])
        nz = np.nonzero(np.abs(co) > 1e-9 * np.abs(co).max())[0]
        zr = np.roots(co[nz[0]:nz[-1] + 1][::-1])
        phis = np.mod(np.angle(zr[np.abs(np.abs(zr) - 1) < 1e-3]), 2 * np.pi)
        assert phis.size > 0
        worst = max(float(np.abs(((G[i, 0] - phis + np.pi) % (2 * np.pi)) - np.pi).min())
                    for i in range(G.shape[0]))
        assert worst < 1e-6, (KP, worst)


def test_the_root_finder_returns_every_root_without_reference_to_a_grid():
    """The completeness link that grid comparison cannot supply.

    "Every stationary phi is a root of the resultant" is a theorem.  The step that can still
    lose one is the companion eigensolve, and checking it against a grid only shows the two
    agree.  This checks the eigensolve on its own terms: a degree-n polynomial has n roots,
    so the finder must return n DISTINCT values that each SATISFY it.

    Measured across KS in {1,2,3} x KP in {3,5,9,13}, degrees 16 to 288: every root
    satisfies the polynomial to a median 1e-16 (worst 4.5e-14) and all are distinct.

    NOT tested by reconstructing the polynomial from its roots -- that is Wilkinson-ill-
    conditioned at these degrees and reports relative errors up to 1e+24 even for perfect
    roots.  It looked like a completeness failure and was a property of numpy.poly; the
    residual direction is the well-conditioned one.
    """
    for KS in (1, 2, 3):
        for KP in (3, 5, 9):
            rng = np.random.default_rng(7)
            C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
            C = C / np.max(np.abs(C))
            D, K, Q = ALG.laurent_D(C)
            deg = (2 * K) * (2 * Q) * 2
            N = 1
            while N <= deg + 2:
                N *= 2
            raw = np.fft.fft(ALG._sylvester_det_on_circle(D, K, Q, N)) / N
            h = deg // 2
            co = np.concatenate([raw[N - h:], raw[:h + 1]])
            nz = np.nonzero(np.abs(co) > 1e-9 * np.abs(co).max())[0]
            c = co[nz[0]:nz[-1] + 1][::-1]
            r = np.roots(c)
            n = len(c) - 1
            assert r.size == n, (KS, KP, r.size, n)
            worst = 0.0
            for z in r:
                scale = np.sum(np.abs(c) * np.abs(z) ** np.arange(n, -1, -1))
                worst = max(worst, abs(np.polyval(c, z)) / max(scale, 1e-300))
            assert worst < 1e-10, (KS, KP, deg, worst)
            sep = np.abs(r[:, None] - r[None, :])
            np.fill_diagonal(sep, np.inf)
            assert (sep.min(axis=1) > 1e-8).all(), (KS, KP, "coincident roots")
