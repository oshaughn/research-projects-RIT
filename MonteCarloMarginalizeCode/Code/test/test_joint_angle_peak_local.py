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


def _ab_tables(seed=0, scale=1.0):
    """Separate A and B tables with the SHIPPED bidegrees: A is linear in the
    waveform (phi <= m_max, u <= 1), B quadratic (phi <= 2 m_max, u <= 2)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    B = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    # B must give a genuine -x^2/2 B penalty, i.e. positive mean curvature
    B[0, 2] = abs(B[0, 2].real) + 3.0
    return scale * A, scale * B


def test_joint_table_pads_A_into_B_and_leaves_e2iu_to_B_alone():
    """The bidegrees differ, which is why the e^{2iu} coefficient carries no A
    contribution -- exactly as anglemarg._laplace_psi_lnI documents.  Getting this
    wrong is a silent broadcast error, so it is pinned."""
    A, B = _ab_tables(seed=1)
    x = 0.7
    C = J.joint_table(A, B, x=x)
    assert C.shape == B.shape
    # the q = +2 column (index 4) must be EXACTLY -x^2/2 * B there: no A contribution
    assert np.array_equal(C[:, 4], (-0.5 * x * x) * B[:, 4].astype(complex))
    # and A must have landed in the central 3 columns of the first 3 rows
    assert np.allclose(C[:3, 1:4], x * A + (-0.5 * x * x) * B[:3, 1:4])
    # rows beyond A's phi-degree keep B alone
    assert np.array_equal(C[3:, :], (-0.5 * x * x) * B[3:, :].astype(complex))


def test_distance_composition_matches_a_direct_sum_over_nodes():
    """The composed value must equal an explicit logsumexp over ALL nodes; the
    node pre-filter may only drop what it can bound as negligible."""
    A, B = _ab_tables(seed=2, scale=1.5)
    x = np.linspace(0.35, 3.0, 24)
    logw = -0.5 * (x - 1.3) ** 2 * 4.0
    val, ok, rep = J.joint_marginalize_over_distance(A, B, x, logw, n_phi=64,
                                                     n_bound_grid=128)
    assert ok, rep
    direct = []
    for xi, wi in zip(x, logw):
        v, o, _ = J.joint_marginalize_peak_local(J.joint_table(A, B, x=float(xi)),
                                                 n_phi=64, n_bound_grid=128)
        direct.append(wi + v)
    m = max(direct)
    ref = m + np.log(np.exp(np.array(direct) - m).sum())
    assert abs(val - ref) < 1e-9, (val, ref, rep)


def test_the_node_prefilter_cannot_change_the_answer():
    """The filter is a COST optimization only.  A cut it cannot JUSTIFY against the
    computed value is undone by retrying with every node, so however aggressively it is
    asked to cut, the answer must not move.  (Tightening `keep_nats` therefore does NOT
    imply fewer live nodes -- an unjustified cut is reverted, which is the point.)"""
    A, B = _ab_tables(seed=4, scale=2.0)
    x = np.linspace(0.3, 4.0, 40)
    logw = -0.5 * (x - 1.0) ** 2 * 3.0
    wide, _, rw = J.joint_marginalize_over_distance(A, B, x, logw, n_phi=64,
                                                    n_bound_grid=128, keep_nats=np.inf)
    tight, _, rt = J.joint_marginalize_over_distance(A, B, x, logw, n_phi=64,
                                                     n_bound_grid=128, keep_nats=25.0)
    assert abs(wide - tight) < 1e-8, (wide, tight, rw, rt)
    # an aggressive cut here is not justifiable, so it must have been retried
    assert rt.get('prefilter_retried') or rt['n_nodes_live'] == x.size, rt


def test_an_unjustified_node_cut_is_retried_not_reported():
    """Regression: `dropped_bound` was once computed, stored in the report, and never
    compared to anything -- the docstring promised a provably-negligible drop while the
    code performed an unchecked one."""
    A, B = _ab_tables(seed=6, scale=2.5)
    x = np.linspace(0.3, 4.0, 32)
    logw = -0.5 * (x - 1.0) ** 2 * 3.0
    _, ok, rep = J.joint_marginalize_over_distance(A, B, x, logw, n_phi=64,
                                                   n_bound_grid=128, keep_nats=5.0)
    # either the cut was justified, or it was undone -- never silently kept
    assert ok
    assert rep.get('prefilter_retried') or rep.get('dropped_margin', -np.inf) < J.OUTSIDE_TOL_NATS
