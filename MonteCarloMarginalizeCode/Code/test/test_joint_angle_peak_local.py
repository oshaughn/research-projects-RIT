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


def test_a_cell_straddling_a_box_edge_counts_as_OUTSIDE():
    """P1 from review.  Classifying grid CENTRES lets a box cover every centre while
    leaving real area uncovered; outside_bound then returned (-inf, 0.0) -- unconditional
    acceptance.  The reviewer's counterexample: n_grid = 8, a box offset half a step with
    half-width pi - h/4 covers all 64 centres and still leaves ~4.8 rad^2 outside."""
    A, B = _ab_tables(seed=2, scale=1.0)
    C = J.joint_table(A, B, x=1.0)
    n = 8
    h = 2 * np.pi / n
    cen = np.array([[h / 2.0, h / 2.0]])
    half = np.array([[np.pi - h / 4.0, np.pi - h / 4.0]])
    sup, area = J.outside_bound(C, cen, half, n_grid=n)
    assert area > 0.0, "a straddling cell must not be reported as fully covered"
    assert np.isfinite(sup), "an uncovered region must yield a finite supremum bound"
    # and the reported area must OVER-estimate, never under-estimate, the true uncovered
    true_uncovered = (2 * np.pi) ** 2 - (2 * (np.pi - h / 4.0)) ** 2
    assert area >= true_uncovered - 1e-9, (area, true_uncovered)


def test_a_full_circle_box_still_counts_as_covering():
    """The conservative shrink must not fire on a box that already spans the circle --
    that is the low-amplitude case where regions merge to the whole torus, which is the
    rule degenerating into the dense grid on purpose."""
    A, B = _ab_tables(seed=2, scale=1.0)
    C = J.joint_table(A, B, x=1.0)
    cen = np.array([[0.0, 0.0]])
    half = np.array([[np.pi, np.pi]])
    sup, area = J.outside_bound(C, cen, half, n_grid=32)
    assert area == 0.0 and sup == -np.inf, (sup, area)


# ------------------------------------------------- both axes localized (phi-local)

def test_u_profile_derivatives_match_finite_differences():
    """F' and F'' come from differentiating UNDER the integral -- F' = E[d_phi g],
    F'' = E[d^2_phi g] + Var(d_phi g) -- so they are exact and cost no extra evaluation.
    That identity is what makes localizing phi possible at all, so it is pinned against
    finite differences of F itself."""
    A, B = _ab_tables(seed=3, scale=3.0)
    C = J.joint_table(A, B, x=1.0)
    h = 1e-5
    for phi in np.linspace(0.3, 5.9, 6):
        F0, d1, d2 = J.u_profile(C, np.array([phi]))
        Fp, _, _ = J.u_profile(C, np.array([phi + h]))
        Fm, _, _ = J.u_profile(C, np.array([phi - h]))
        fd1 = (Fp[0] - Fm[0]) / (2 * h)
        fd2 = (Fp[0] - 2 * F0[0] + Fm[0]) / h ** 2
        assert abs(d1[0] - fd1) < 1e-4 * max(1.0, abs(fd1)), (phi, d1[0], fd1)
        assert abs(d2[0] - fd2) < 1e-2 * max(1.0, abs(fd2)), (phi, d2[0], fd2)


@pytest.mark.parametrize("scale", [1.0, 3.0, 10.0])
def test_phi_local_matches_a_converged_dense_reference(scale):
    """Both axes localized, against a dense torus quadrature."""
    A, B = _ab_tables(seed=3, scale=1.0)
    C = J.joint_table(A * scale, B * scale, x=1.0)
    val, ok, rep = J.phi_local_marginalize(C)
    assert ok, rep
    assert abs(val - _ref(C, n=2048)) < 1e-4, (scale, val, rep)


def test_phi_local_cost_does_not_grow_with_amplitude():
    """The whole point of localizing BOTH axes: the dense rule spends ~A points on the
    (phi,u) product, this spends a number set by the mode structure, which does not move
    when the data amplitude does."""
    A, B = _ab_tables(seed=3, scale=1.0)
    counts = []
    for scale in (1.0, 10.0, 100.0):
        _, ok, rep = J.phi_local_marginalize(J.joint_table(A * scale, B * scale, x=1.0))
        assert ok
        counts.append(rep['n_phi_regions'])
    assert max(counts) <= 8, counts          # bounded, not growing with sqrt(A)


def test_phi_cover_bound_is_routed_through_g_not_through_F_curvature():
    """Regression.  Bounding F by Taylor with F'' <= M_(2,0) + M_(1,0)^2 is useless: that
    variance bound grows as the SQUARE of the amplitude and produced margins of +51 and
    +1196 nats (no bound at all).  Routing through F <= log(2pi) + sup_u g keeps the
    remainder linear in amplitude, so high-amplitude rows are ACCEPTED rather than
    declined for a defect in the bound."""
    A, B = _ab_tables(seed=3, scale=1.0)
    for scale in (30.0, 100.0):
        _, ok, rep = J.phi_local_marginalize(J.joint_table(A * scale, B * scale, x=1.0))
        assert ok, (scale, rep)
        assert rep['margin'] < J.OUTSIDE_TOL_NATS, (scale, rep)
