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
    src = inspect.getsource(ALG.stationary_points)
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
