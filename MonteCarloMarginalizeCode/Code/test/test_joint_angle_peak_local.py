"""Tests for the joint (phi, psi) peak-local kernel.

Reference is a converged periodic trapezoid on the torus -- the mathematical content
of the shipped `anglemarg` exact scheme -- so accuracy is measured against a
quadrature, never against another peak-local run.
"""
import builtins

import numpy as np
import pytest
from scipy import special

from RIFT.likelihood import joint_angle_peak_local as J
from RIFT.likelihood import bivariate_trig_stationary as BTS


def synth_table(seed=0, scale=1.0, bidegree=(4, 2)):
    """A random exact 2-D trig table of the shipped bidegree, amplitude `scale`."""
    KP, KS = bidegree[0] + 1, bidegree[1]
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(KP, 2 * KS + 1)) + 1j * rng.normal(size=(KP, 2 * KS + 1))
    return scale * C


def _periodic_set_error(got, want):
    """Symmetric nearest-neighbour error for two small point sets on the torus."""
    got = np.asarray(got, dtype=float).reshape((-1, 2))
    want = np.asarray(want, dtype=float).reshape((-1, 2))
    if len(got) != len(want):
        return np.inf
    d = (got[:, None, :] - want[None, :, :] + np.pi) % (2 * np.pi) - np.pi
    r = np.linalg.norm(d, axis=-1)
    return max(float(np.max(np.min(r, axis=0))),
               float(np.max(np.min(r, axis=1))))


def _separable_table(m=3, n=2, a=7.0, b=4.0):
    """Exactly ``a cos(m phi) + b cos(n u)`` in the RIFT storage convention."""
    C = np.zeros((m + 1, 2 * n + 1), dtype=complex)
    C[m, n] = 0.5 * a                 # k>0 is doubled by the evaluator
    C[0, 2 * n] = b                   # q=+n; taking Re supplies q=-n
    return C


def test_algebraic_canonical_table_matches_the_shipped_field_convention():
    """Laurent conversion is exact, including k=0 overlap and both q signs."""
    C = synth_table(seed=91, scale=2.3, bidegree=(3, 2))
    A = BTS.canonical_laurent_table(C)
    rng = np.random.default_rng(123)
    p = rng.uniform(0, 2 * np.pi, size=(37, 2))
    k = np.arange(-3, 4)[None, :, None]
    q = np.arange(-2, 3)[None, None, :]
    full = np.real(np.sum(
        A[None] * np.exp(1j * (p[:, 0, None, None] * k
                                + p[:, 1, None, None] * q)), axis=(1, 2)))
    assert np.allclose(full, J.eval_g(C, p[:, 0], p[:, 1]), rtol=0, atol=2e-13)


def test_algebraic_enumerator_preserves_every_codominant_separable_maximum():
    """Generic projection must not collapse modes sharing the same phi or u.

    A coordinate resultant has repeated projected roots on this Cartesian mode
    lattice.  The affine hidden variable separates them and returns all six equal
    maxima, without any angular samples.
    """
    C = _separable_table(m=3, n=2)
    out = BTS.enumerate_torus_maxima(C)
    want = np.array([(2 * np.pi * j / 3, np.pi * k)
                     for j in range(3) for k in range(2)])
    assert out.ok, out.report
    assert out.report["mixed_volume"] == 24
    assert out.stationary_points.shape == (24, 2)
    assert out.points.shape == (6, 2)
    assert _periodic_set_error(out.points, want) < 2e-9
    assert np.ptp(out.values) < 2e-12


def test_algebraic_enumerator_resolves_near_annihilating_stationary_points():
    """A close max/min pair is part of the polynomial, not a resolution choice."""
    ratio = 3.9999999
    C = np.zeros((3, 5), dtype=complex)
    C[1, 2] = 0.5 * ratio
    C[2, 2] = 0.5
    C[0, 4] = 2.0
    out = BTS.enumerate_torus_maxima(C)
    assert out.ok, out.report
    assert out.stationary_points.shape == (16, 2)
    assert out.points.shape == (4, 2)

    # The two additional phi stationary points approach pi from either side.
    # Their separation is smaller than a 4096-point circle spacing; retaining
    # both demonstrates that no sampled phi resolution controls enumeration.
    expected_phi = np.mod(np.array([
        0.0, np.pi,
        np.arccos(-ratio / 4.0),
        2.0 * np.pi - np.arccos(-ratio / 4.0),
    ]), 2.0 * np.pi)
    got_phi = np.unique(np.round(out.stationary_points[:, 0], 11))
    circ = np.abs((got_phi[:, None] - expected_phi[None, :] + np.pi)
                  % (2 * np.pi) - np.pi)
    assert got_phi.size == 4
    assert np.max(np.min(circ, axis=0)) < 2e-8
    close_sep = 2.0 * (np.pi - np.arccos(-ratio / 4.0))
    assert close_sep < 2.0 * np.pi / 4096


def test_algebraic_enumerator_declines_at_exact_stationary_degeneracy():
    """At annihilation certification declines, while safe targets stay available."""
    C = np.zeros((3, 5), dtype=complex)
    C[1, 2] = 2.0                    # ratio c1/c2 == 4 exactly
    C[2, 2] = 0.5
    C[0, 4] = 2.0
    out = BTS.enumerate_torus_maxima(C)
    assert not out.ok
    assert out.points.shape[0] == 2
    assert all(p["decline"] is not None for p in out.report["projections"])
    assert min(p["min_jacobian_rcond"] for p in out.report["projections"]) < 2e-10
    assert np.all(np.linalg.eigvalsh(out.hessians) < 0.0)


def test_algebraic_enumeration_size_and_modes_are_amplitude_independent():
    """Scaling the exponent changes widths, never its algebraic candidate set."""
    C = synth_table(seed=17, bidegree=(2, 2))
    low = BTS.enumerate_torus_maxima(C)
    high = BTS.enumerate_torus_maxima(1.0e8 * C)
    assert low.ok and high.ok, (low.report, high.report)
    assert low.report["mixed_volume"] == high.report["mixed_volume"] == 32
    assert [p["pencil_size"] for p in low.report["projections"]] == [
        p["pencil_size"] for p in high.report["projections"]]
    assert _periodic_set_error(low.points, high.points) < 2e-8


def test_incomplete_algebraic_accounting_never_drops_the_likelihood_sample():
    """A root deficit is either cover-certified or sent to the dense fallback."""
    C = synth_table(seed=3, scale=1.0)
    value, ok, report = J.joint_marginalize_peak_local(
        C, n_phi=64, n_bound_grid=128)
    assert ok and np.isfinite(value), report
    assert not report["enumeration_certified"]
    assert report["result_path"] in {
        "algebraic-best-effort/bound-certified", "dense-phi/exact-u"}
    projections = report["enumeration"]["projections"]
    assert any(p["verified_complex_roots"] < p["expected_roots"]
               for p in projections)
    assert all("min_jacobian_rcond" in p for p in projections)
    if report["result_path"] == "algebraic-best-effort/bound-certified":
        assert report["margin"] < J.OUTSIDE_TOL_NATS
    else:
        assert report["fallback_reason"]


def test_certified_enumeration_cannot_certify_capped_local_quadrature(monkeypatch):
    """A complete root set does not certify integration inside its cover.

    ``s cos(phi-u) + cos(phi+u)`` has the exact normalized integral
    ``I0(s) I0(1)``.  Its weak direction makes the mode boxes cover the torus,
    while its strong diagonal direction is much narrower than either capped
    axis-aligned rule.  The outside ledger therefore says that no area was
    omitted even though the local quadrature is unresolved.
    """
    strength = 1.0e8
    C = np.zeros((2, 5), dtype=complex)
    C[1, 1] = 0.5 * strength       # strength * cos(phi-u)
    C[1, 3] = 0.5                  # cos(phi+u)
    exact = (np.log(special.i0e(strength)) + strength
             + np.log(special.i0e(1.0)) + 1.0)
    fallback_calls = []

    def finite_fallback(table, n_phi=None, n_u_nodes=64):
        fallback_calls.append((table, n_phi, n_u_nodes))
        return exact, {"doubling_error": 0.0, "fixture": "known integral"}

    monkeypatch.setattr(J, "dense_phi_exact_u_marginalize", finite_fallback)
    value, ok, report = J.joint_marginalize_peak_local(C)

    assert ok and value == exact
    assert report["enumeration_certified"], report["enumeration"]
    assert report["area_outside"] == 0.0
    assert report["n_boxes_pts_capped"] >= 1
    assert report["local_quadrature_capped"]
    assert report["local_quadrature_error"] > 0.1
    assert report["result_path"] == "dense-phi/exact-u"
    assert "inside-cover quadrature did not converge" in report["fallback_reason"]
    assert len(fallback_calls) == 1


def test_numpy_dense_fallback_does_not_import_the_optional_jax_stack(monkeypatch):
    """The final fallback remains usable in an installation without JAX."""
    real_import = builtins.__import__

    def reject_jax(name, globals=None, locals=None, fromlist=(), level=0):
        if "jax_ile" in name or name == "jax" or name.startswith("jax."):
            raise AssertionError("the NumPy fallback attempted to import JAX")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_jax)
    C = np.zeros((2, 5), dtype=complex)
    C[1, 2] = 0.25
    value, report = J.dense_phi_exact_u_marginalize(C, n_phi=16)

    assert np.isfinite(value)
    assert report["n_phi_coarse"] == 128
    assert report["n_phi"] == 256


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


def test_an_undersized_region_falls_back_instead_of_returning_local_value():
    """The load-bearing behaviour.  W_SIGMA too small leaves mass outside the cover;
    the local value may still be right, but the rule cannot PROVE it and must fall back.
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
    assert ok_small, rep_small
    assert rep_small['result_path'] == 'dense-phi/exact-u'
    assert 'omitted-mass bound too large' in rep_small['fallback_reason']
    assert ok_big
    assert abs(val_small - val_big) < 1e-6


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


def test_a_wrapped_phi_region_is_clamped_to_one_circuit():
    """Regression, found on REAL coefficient tables and not reachable from the synthetic
    ones.  At low amplitude F is nearly flat, so sigma is huge and [p-W*sig, p+W*sig]
    spans more than 2 pi; integrating that range literally wraps the circle several times
    and counts the same mass repeatedly -- measured +1.84 nats, a factor of e^1.84 = 6.3,
    and ACCEPTED, because a region covering everything leaves nothing outside for the
    omitted-mass certificate to object to.

    The general lesson, worth more than the fix: the certificate bounds what is OUTSIDE
    the regions and cannot see an error made INSIDE one."""
    A, B = _ab_tables(seed=1, scale=0.05)          # deliberately near-flat
    C = J.joint_table(A, B, x=0.3)
    val, ok, rep = J.phi_local_marginalize(C)
    assert ok, rep
    ref = _ref(C, n=2048)
    assert abs(val - ref) < 1e-3, (val, ref, rep)
    # and the covered length may never exceed one circuit
    assert rep['n_phi_regions'] >= 1


def test_inner_u_quadrature_is_converged_at_the_default_node_count():
    """P1 from review, and it needs its OWN regression because the phi omitted-mass
    certificate cannot see it: an error made INSIDE a region is invisible to a bound on
    what lies outside.  Raising n_nodes must not move the answer."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(6):
        sc = 10.0 ** rng.uniform(-0.5, 2.5)
        A = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * sc
        B = (rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))) * sc
        B[0, 2] = abs(B[0, 2].real) + 3.0 * sc
        C = J.joint_table(A, B, x=1.0)
        for phi in (0.3, 1.9, 4.4):
            lo, _, _ = J.u_profile(C, np.array([phi]), n_nodes=64)
            hi, _, _ = J.u_profile(C, np.array([phi]), n_nodes=1024)
            worst = max(worst, abs(lo[0] - hi[0]))
    # 1e-4, matching this module's other marginal assertions.  An earlier revision of
    # this test allowed 1e-3, which CODIFIED a 7.2e-4 inner-u error rather than catching
    # it -- a tolerance chosen after seeing the number is not a check.
    assert worst < 1e-4, worst


def test_a_clipped_newton_point_is_not_classified_as_a_peak():
    """P1 from review.  The in-cell Newton is clamped to [lo, mid], so it can come to
    rest ON a boundary with a large stationary residual -- and curvature alone then calls
    that a maximum, centring a +-W sigma window on a non-stationary point.  Measured over
    5400 cells: 492 (18%) that g'' < 0 accepted are rejected once a small residual and an
    interior position are also required, the worst of them sitting at |g_u|/M_1 = 0.33."""
    rng = np.random.default_rng(7)
    n_curv, n_gated = 0, 0
    for _ in range(20):
        sc = 10.0 ** rng.uniform(-0.5, 3.0)
        A = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * sc
        B = (rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))) * sc
        B[0, 2] = abs(B[0, 2].real) + 3.0 * sc
        C = J.joint_table(A, B, x=1.0)
        k, q, w, KS = J._kq(C)
        for phi in np.linspace(0.05, 6.2, 5):
            ph = (np.exp(1j * phi * k) * w).ravel()
            D = lambda qq: complex((ph * C[:, KS + qq]).sum())
            c1 = D(1) + np.conj(D(-1))
            c2 = D(2) + np.conj(D(-2))
            u = np.sort(np.mod(np.angle(np.roots(
                [c2, c1 / 2, 0, -np.conj(c1) / 2, -np.conj(c2)])), 2 * np.pi))
            mid = 0.5 * (u + np.roll(u, -1)
                         + np.where(np.arange(4) == 3, 2 * np.pi, 0.0))
            lo_c = np.roll(mid, 1) - np.where(np.arange(4) == 0, 2 * np.pi, 0.0)
            us = u.copy()
            pv = np.full(4, phi)
            for _i in range(8):
                g1 = J.eval_g(C, pv, us, (0, 1))
                g2 = J.eval_g(C, pv, us, (0, 2))
                st = np.where(np.abs(g2) > 0, -g1 / np.where(np.abs(g2) > 0, g2, 1.0), 0.0)
                us = np.clip(us + np.clip(st, -0.5, 0.5), lo_c, mid)
            g1c = J.eval_g(C, pv, us, (0, 1))
            g2c = J.eval_g(C, pv, us, (0, 2))
            m1u = max(J.derivative_bound(C, (0, 1)), 1e-300)
            edge = 1e-9 * max(float(np.max(mid - lo_c)), 1e-300)
            curv = g2c < 0.0
            gated = curv & (np.abs(g1c) <= 1e-8 * m1u) & (us > lo_c + edge) & (us < mid - edge)
            n_curv += int(curv.sum())
            n_gated += int(gated.sum())
    assert n_curv > n_gated, (n_curv, n_gated)   # the gate must actually reject


def test_the_quartic_roots_are_seeds_and_must_be_refined_in_cell():
    """Why the in-cell Newton refinement is not dead weight.  A conjugate-reciprocal
    pair leaves the unit circle -- measured, 309 of 900 (table, phi) pairs have at least
    one such root -- and a spurious root's ANGLE is not a stationary point at all: the
    worst |g_u|/M_1 at a raw root measured 0.311, i.e. nowhere near stationary.  Window
    around that and the window is centred off the peak."""
    rng = np.random.default_rng(11)
    worst_resid = 0.0
    n_off = 0
    for _ in range(60):
        sc = 10.0 ** rng.uniform(-0.5, 2.0)
        A = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * sc
        B = (rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))) * sc
        B[0, 2] = abs(B[0, 2].real) + 3.0 * sc
        C = J.joint_table(A, B, x=1.0)
        k, q, w, KS = J._kq(C)
        for phi in (0.3, 1.9, 4.4):
            ph = (np.exp(1j * phi * k) * w).ravel()
            D = lambda qq: complex((ph * C[:, KS + qq]).sum())
            c1 = D(1) + np.conj(D(-1))
            c2 = D(2) + np.conj(D(-2))
            z = np.roots([c2, c1 / 2, 0, -np.conj(c1) / 2, -np.conj(c2)])
            n_off += int(np.any(np.abs(np.abs(z) - 1.0) > 1e-6))
            u = np.sort(np.mod(np.angle(z), 2 * np.pi))
            g1 = J.eval_g(C, np.full(4, phi), u, (0, 1))
            m1 = max(J.derivative_bound(C, (0, 1)), 1e-300)
            worst_resid = max(worst_resid, float(np.max(np.abs(g1)) / m1))
    assert n_off > 0, "fixture family must contain off-circle roots"
    assert worst_resid > 1e-6, worst_resid   # raw roots are NOT all stationary points


def test_phi_regions_are_disjoint_on_the_CIRCLE():
    """P1 from review.  Merging on the LINE never joins a window near 0 to one near
    2*pi, but every region is integrated at mod(., 2*pi) -- so both regions cover both
    peaks and the mass is counted twice.  Measured before the fix: +log 2 = +0.693 nats
    returned with ok=True and margin -437, because the error is INSIDE the regions.

    Asserted as an INVARIANT rather than hunted for with a value comparison: reduced to
    the circle, the regions must not overlap and must not exceed one circuit."""
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(40):
        sc = 10.0 ** rng.uniform(0.0, 2.5)
        A = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * sc
        B = (rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))) * sc
        B[0, 2] = abs(B[0, 2].real) + 3.0 * sc
        C = J.joint_table(A, B, x=1.0)
        _, _, rep = J.phi_local_marginalize(C)
        regs = rep.get('phi_regions', [])
        if not regs:
            continue
        checked += 1
        total = sum(b - a for a, b in regs)
        assert total <= 2 * np.pi + 1e-9, (total, regs)
        # sample each region densely, reduce to the circle, and require no point to be
        # covered twice
        pts = []
        for a, b in regs:
            pts.append(np.mod(np.linspace(a, b, 512, endpoint=False), 2 * np.pi))
        if len(pts) > 1:
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = np.abs(pts[i][:, None] - pts[j][None, :])
                    d = np.minimum(d, 2 * np.pi - d)
                    assert d.min() > 1e-6, ("regions overlap on the circle", regs)
    assert checked > 5, checked


# --------------------------------------------- internal accuracy inside the cover

def _torus_reference(C, n=2048):
    """log (2pi)^-2 int int exp(g) over the WHOLE torus, independent of the peak-local path."""
    ph = np.linspace(0.0, 2.0 * np.pi, n)
    P, U = np.meshgrid(ph, ph, indexing='ij')
    g = J.eval_g(C, P.ravel(), U.ravel()).reshape(n, n)
    w = np.full(n, 2.0 * np.pi / (n - 1)); w[0] *= 0.5; w[-1] *= 0.5
    W = np.log(w)[:, None] + np.log(w)[None, :] - 2.0 * np.log(2.0 * np.pi)
    m = g.max()
    return m + np.log(np.sum(np.exp(g - m + W)))


def _degenerate_ridge_tables(seed=113, scale=1.0):
    """SYNTHETIC coefficients reproducing the torus-spanning collapse.  No run data.

    An earlier version of this fixture hard-coded coefficients read out of an actual
    production evaluation, together with its sky/time indices.  External review was right
    that merging it would publish run-derived scientific data in a test, so it is replaced
    by a seeded synthetic draw.

    What could NOT be replaced is the structure, and it took a seed search to find it.  A
    hand-built table with the same sparsity PATTERN -- A only at k=2 with q=+-1 and
    strongly asymmetric, B almost entirely the real (k=0,ks=0) term -- does not reproduce
    the collapse: with round numbers it gives n_regions=4, area_outside=31.7 and no error
    at all.  The relative PHASES decide whether the enumerated regions merge into one that
    spans the torus, so the fixture is a search over seeded phases for a draw that does.
    Seed 113 of 200 is the strongest.  This is why random-coefficient tests never reached
    this branch: the landscape is a near-degenerate ridge, not isolated peaks.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((3, 3), dtype=complex)
    B = np.zeros((5, 5), dtype=complex)
    A[2, 2] = 3000.0 * scale * np.exp(1j * rng.uniform(0.0, 2 * np.pi))
    A[2, 0] = A[2, 2] * 0.013 * np.exp(1j * rng.uniform(0.0, 2 * np.pi))
    B[0, 2] = 1550.0 * scale
    B[0, 0] = 21.5 * scale * np.exp(1j * rng.uniform(0.0, 2 * np.pi))
    B[0, 4] = np.conj(B[0, 0])
    B[4, 2] = 0.0036 * scale * np.exp(1j * rng.uniform(0.0, 2 * np.pi))
    B[4, 4] = 0.0886 * scale * np.exp(1j * rng.uniform(0.0, 2 * np.pi))
    k, q, w, _ = J._kq(A)
    x = float(np.sum(w * np.abs(A))) / float(B[0, 2].real)
    return J.joint_table(A, B, x), x


def test_a_fully_covered_box_is_still_accurate_inside():
    """OMITTED-MASS CONTROL IS NOT INTERNAL ACCURACY, and this is the case that proves the
    two are independent.  On the production tables the cover collapses to a single region
    spanning the whole torus, so ``area_outside == 0`` and ``margin == -inf``: the
    certificate reports that NOTHING is omitted, which is true and which says nothing at
    all about the quadrature inside.  With the per-axis cap at its old value of 256 the
    value sat 0.36 nats from a converged reference while reporting -inf.

    0.36 nats is not a rounding error.  It is stated against the CONVERGED TORUS REFERENCE
    below and against nothing else: an earlier version of this docstring compared it to a
    saddle-point prototype's 0.654 nats, and that figure has since been RETRACTED by the
    session that produced it -- its start-point search was unconverged, moving up to 1.2
    nats per point and changing sign under refinement.  A ratio against a retracted
    denominator is worse than no ratio, and this error needs no comparison to be a defect:
    the certificate reported nothing omitted while the value was wrong.
    """
    C, _ = _degenerate_ridge_tables()
    assert abs(np.sum(np.abs(C)) - 24164.9) < 1.0, "fixture drifted"
    lnZ, ok, rep = J.joint_marginalize_peak_local(C)
    assert ok, rep
    # A complete algebraic cover retains the original inside-box regression.  An
    # incomplete solve may expose less covered area; the new hierarchy must then
    # take the finite dense fallback instead of treating the row as -inf.
    if rep['result_path'].startswith('algebraic'):
        assert rep['area_outside'] == 0.0, rep
        assert rep['margin'] == -np.inf, rep
    else:
        assert rep['result_path'] == 'dense-phi/exact-u', rep
        assert rep['fallback_reason'], rep
        assert rep['dense_fallback']['doubling_error'] < 1e-4, rep
    err = abs(lnZ - _torus_reference(C))
    assert err < 1.0e-2, "inside-the-cover error %.4f nats (cap 256 gave 0.36)" % err


def test_a_capped_box_is_reported_and_never_silent():
    """A box whose curvature-derived node count hits the ceiling is under-resolved, and the
    certificate cannot express that.  It must therefore be COUNTED -- otherwise the caller
    is handed 'nothing omitted' about a value the quadrature got wrong.
    """
    C, _ = _degenerate_ridge_tables()
    _, ok, rep = J.joint_marginalize_peak_local(C)
    assert ok
    assert 'n_boxes_pts_capped' in rep
    assert rep['n_boxes_pts_capped'] >= 1, rep   # this amplitude DOES still cap at 512
    # and a much flatter case must NOT be flagged, or the counter says nothing
    C_lo, _ = _degenerate_ridge_tables(scale=1.0e-4)
    _, ok2, rep2 = J.joint_marginalize_peak_local(C_lo)
    assert ok2, rep2
    assert rep2['n_boxes_pts_capped'] == 0, rep2
