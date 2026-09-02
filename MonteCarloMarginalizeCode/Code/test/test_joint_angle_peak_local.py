"""Tests for the joint (phi, psi) peak-local kernel.

Reference is a converged periodic trapezoid on the torus -- the mathematical content
of the shipped `anglemarg` exact scheme -- so accuracy is measured against a
quadrature, never against another peak-local run.
"""
import numpy as np
import pytest

from RIFT.likelihood import joint_angle_peak_local as J


def synth_table(seed=0, scale=1.0, bidegree=(4, 2)):
    """A random exact 2-D trig table of the shipped bidegree, amplitude `scale`."""
    KP, KS = bidegree[0] + 1, bidegree[1]
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
    return scale * C


def _ref(C, n=2048):
    """log[(2pi)^-2 int int exp(g)] by the periodic trapezoid (== the plain mean)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    PHI, U = np.meshgrid(t, t, indexing='ij')
    g = J.eval_g(C, PHI.ravel(), U.ravel())
    m = g.max()
    return m + np.log(np.exp(g - m).mean())


def test_every_exported_name_is_defined():
    """A star-import must not raise: __all__ gaining an undefined name has bitten the
    companion time modules before, and only a star-import can see it."""
    missing = [n for n in J.__all__ if not hasattr(J, n)]
    assert not missing, missing
    ns = {}
    exec("from RIFT.likelihood.joint_angle_peak_local import *", ns)


@pytest.mark.parametrize("scale", [1.0, 4.0, 12.0])
def test_matches_a_converged_dense_reference(scale):
    """The whole point: the same answer as the dense rule, from local work only."""
    C = synth_table(seed=3, scale=scale)
    val, ok, rep = J.joint_marginalize_peak_local(C, n_phi=96)
    assert ok, rep
    assert abs(val - _ref(C)) < 1e-6, (scale, val, _ref(C), rep)


def test_derivative_bound_is_actually_a_bound():
    """M_(a,b) is the multi-index triangle inequality on the exact table.  It must
    never be exceeded; a fitted bound is the defect this whole design refuses."""
    C = synth_table(seed=11, scale=6.0)
    t = np.linspace(0.0, 2.0 * np.pi, 401, endpoint=False)
    PHI, U = np.meshgrid(t, t, indexing='ij')
    for order in ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1)):
        d = J.eval_g(C, PHI.ravel(), U.ravel(), order)
        assert np.max(np.abs(d)) <= J.derivative_bound(C, order) * (1 + 1e-12), order


def test_u_solve_returns_every_root_and_filters_nothing():
    """No |z| = 1 tolerance, deliberately: at exact multiplicity the computed roots
    smear off the circle by eps^(1/m), so a filter drops real modes in precisely the
    degenerate regime that is production.  Roots are seeds; regions decide."""
    C = synth_table(seed=5, scale=3.0)
    for phi in np.linspace(0, 2 * np.pi, 17):
        assert J.u_stationary_at_phi(C, phi).size == 4


def test_eval_g_chunking_cannot_change_the_answer():
    """_POINT_CHUNK is a memory parameter; each point is summed independently, so it
    cannot move the result even in the last bit."""
    C = synth_table(seed=7, scale=2.0)
    phi = np.linspace(0, 6, 257)
    u = np.linspace(1, 5, 257)
    big = J._POINT_CHUNK
    try:
        J._POINT_CHUNK = 200000
        a = J.eval_g(C, phi, u)
        J._POINT_CHUNK = 13
        b = J.eval_g(C, phi, u)
    finally:
        J._POINT_CHUNK = big
    assert a.tobytes() == b.tobytes()


def test_an_undersized_region_is_DECLINED_not_returned():
    """The load-bearing behaviour.  W_SIGMA too small leaves mass outside the cover;
    the value may still be right, but the rule cannot PROVE it and must decline.
    Measured on the shipped tables: at W = 8 the margin is -18 nats against a -23
    tolerance, and at 14 it is -71 -- with the returned value identical at both."""
    C = synth_table(seed=3, scale=12.0)
    keep = J.W_SIGMA
    try:
        J.W_SIGMA = 0.6
        val_small, ok_small, rep_small = J.joint_marginalize_peak_local(C, n_phi=96)
        J.W_SIGMA = keep
        val_big, ok_big, _ = J.joint_marginalize_peak_local(C, n_phi=96)
    finally:
        J.W_SIGMA = keep
    assert not ok_small, rep_small
    assert rep_small['decline'] == 'omitted-mass bound too large'
    assert ok_big


def test_regions_merge_rather_than_double_counting():
    """Overlapping regions would count the mass between them twice.  Merging is what
    makes the rule degrade CONTINUOUSLY into the dense grid as amplitude falls."""
    C = synth_table(seed=3, scale=0.4)
    _, _, rep = J.joint_marginalize_peak_local(C, n_phi=96)
    assert rep['n_regions'] <= rep['n_modes']
