"""Relative binning reproduces the exact Q/U/V reduction (ripple_modes.reduce_qvu)."""
import numpy as np
import jax.numpy as jnp
import pytest

from RIFT.likelihood.jax_ile.relative_binning import RelativeBinning, phase_bins, UnsupportedFiducial
from RIFT.likelihood.jax_ile.ripple_modes import reduce_qvu

N, DF = 2 ** 14, 0.25
F = -0.5 * N * DF + DF * np.arange(N)
BAND = (np.abs(F) >= 20) & (np.abs(F) <= 1000)


def _chirp(k, amp=1.0, dphi=0.0, dt0=0.0):
    a = np.where(BAND, np.abs(F + 1e-9) ** (-7 / 6), 0) * amp
    ph = (k + 1) * 3e4 * np.abs(F + 1e-9) ** (-5 / 3) + np.sign(F) * 0.3 * k + 2 * np.pi * F * dt0 + dphi
    return a * np.exp(1j * ph * np.sign(F + 1e-12))


def _setup():
    rng = np.random.default_rng(1)
    h0 = np.stack([_chirp(k) for k in range(3)])
    d = np.where(BAND, rng.normal(size=N) + 1j * rng.normal(size=N), 0) + 0.3 * h0.sum(0)
    w = np.where(BAND, 1.0, 0.0)
    dets = {"H1": dict(d=jnp.asarray(d), w=jnp.asarray(w), n_shift=1000)}
    return h0, dets


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


def test_rb_exact_at_fiducial_and_converges():
    h0, dets = _setup()
    h1 = np.stack([_chirp(k, amp=1.05, dphi=0.2, dt0=2e-4) for k in range(3)])
    errs = []
    for eps in (1.0, 0.1):
        rb = RelativeBinning(F, dets, 200, DF, 1.0 / (N * DF), phase_bins(20, 1000, eps)).set_fiducial(h0)
        for h, store in ((h0, None), (h1, errs)):
            q, u, v = rb.qvu(jnp.asarray(h[:, rb.k_pos]), jnp.asarray(h[:, rb.k_neg]))["H1"]
            q0, u0, v0 = reduce_qvu(jnp.asarray(h), dets["H1"]["d"], dets["H1"]["w"], DF, 1000, 200)
            e = max(_rel(q, q0), _rel(u, u0), _rel(v, v0))
            if store is None:
                assert e < 1e-4, e          # exact up to the ratio regularization
            else:
                store.append(e)
    assert errs[1] < errs[0] and errs[1] < 1e-4, errs


def test_zero_mode_fiducial_declines_instead_of_dropping_emerging_mode():
    h0, dets = _setup()
    h0[1] = 0
    h1 = h0.copy()
    h1[1] = _chirp(1, amp=1.05)
    # This mode genuinely contributes to a candidate's exact inner products.
    _, u, _ = reduce_qvu(jnp.asarray(h1), dets["H1"]["d"], dets["H1"]["w"], DF, 1000, 200)
    assert float(np.asarray(u)[1, 1].real) > 0
    rb = RelativeBinning(F, dets, 200, DF, 1.0 / (N * DF), phase_bins(20, 1000, .3))
    with pytest.raises(ValueError, match="zero fiducial modes"):
        rb.set_fiducial(h0)


@pytest.mark.parametrize("support", ["outside_band", "interior_only"])
def test_unrepresented_weighted_support_declines(support):
    h0, dets = _setup()
    rb = RelativeBinning(F, dets, 200, DF, 1./(N*DF), [20.,100.,200.,1000.])
    if support == "outside_band":
        h0[1] = 0
        h0[1, np.flatnonzero(F > 1100)[0]] = 1.
    else:
        # Nonzero mode inside the weighted bin, but absent at its two edges.
        h0[1, rb.k_pos[1:3]] = 0
        h0[1, rb.k_neg[1:3]] = 0
    with pytest.raises(ValueError, match="weighted"):
        rb.set_fiducial(h0)


def test_zero_edges_in_unweighted_bins_are_harmless():
    h0, dets = _setup()
    # Add two bins outside the existing detector band. Zero modes there must
    # not turn a usable fiducial into a full-grid evaluation.
    rb = RelativeBinning(F, dets, 200, DF, 1./(N*DF), [20.,1000.,1100.,1200.])
    rb.set_fiducial(h0)
    assert np.isfinite(np.asarray(rb.state()["scale"])).all()


def _one_zero_edge(weight_freq, zero_freqs):
    f = np.arange(-4., 4.)
    h = np.ones((1, len(f)), complex)
    h[:, np.isin(f, zero_freqs)] = 0
    dets = {"H1": dict(d=jnp.zeros(len(f), dtype=complex),
                       w=jnp.asarray(np.isin(f, weight_freq).astype(float)), n_shift=0)}
    return f, h, dets


def test_single_zero_edge_with_weighted_interior_declines():
    # With the old guard this accepted its own fiducial but returned U=1,
    # versus exact U=4: the undefined left ratio was replaced by zero.
    f, h, dets = _one_zero_edge([-2.,2.], [-1.,1.])
    _, exact_u, _ = reduce_qvu(jnp.asarray(h), dets["H1"]["d"], dets["H1"]["w"], 1., 0, 1)
    np.testing.assert_allclose(exact_u, [[4.]])
    with pytest.raises(ValueError, match="weighted"):
        RelativeBinning(f, dets, 1, 1., 1/8, [1.,3.]).set_fiducial(h)


def test_zero_edge_unused_by_weighted_basis_is_harmless():
    f, h, dets = _one_zero_edge([-3.,3.], [-1.,1.])
    rb = RelativeBinning(f, dets, 1, 1., 1/8, [1.,3.]).set_fiducial(h)
    actual = rb.qvu(h[:,rb.k_pos], h[:,rb.k_neg])["H1"]
    expected = reduce_qvu(jnp.asarray(h), dets["H1"]["d"], dets["H1"]["w"], 1., 0, 1)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-6, atol=0)


def test_mirrored_support_is_needed_by_v_with_asymmetric_weights():
    f, h, dets = _one_zero_edge([2.], [-1.])
    with pytest.raises(ValueError, match="weighted negative"):
        RelativeBinning(f, dets, 1, 1., 1/8, [1.,3.]).set_fiducial(h)


@pytest.mark.parametrize("edge_amplitude", [1e-3,1e-4,1e-8,1e-150,1e-300])
def test_nonzero_edge_preserves_fiducial_quv_without_soft_shrink(edge_amplitude):
    f, h, dets = _one_zero_edge([-2.,2.], [])
    h[:, np.abs(f)==1] = edge_amplitude * (1+1j)
    dets["H1"]["d"] = jnp.ones(len(f), dtype=complex)
    assert np.all(h[:,np.abs(f)==1] != 0)
    rb = RelativeBinning(f, dets, 1, 1., 1/8, [1.,3.]).set_fiducial(h)
    actual = rb.qvu(h[:,rb.k_pos], h[:,rb.k_neg])["H1"]
    expected = reduce_qvu(jnp.asarray(h), dets["H1"]["d"], dets["H1"]["w"], 1., 0, 1)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=0)


def test_tiny_nonzero_mode_keeps_finite_linear_terms():
    f, h, dets = _one_zero_edge([-2.,2.], [])
    h *= 1e-300 * (1+1j)
    dets["H1"]["d"] = jnp.ones(len(f), dtype=complex) * 1e300
    rb = RelativeBinning(f, dets, 1, 1., 1/8, [1.,3.]).set_fiducial(h)
    actual = rb.qvu(h[:,rb.k_pos], h[:,rb.k_neg])["H1"]
    expected = reduce_qvu(jnp.asarray(h), dets["H1"]["d"], dets["H1"]["w"], 1., 0, 1)
    assert np.max(np.abs(np.asarray(expected[0]))) > 1.
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=0)


@pytest.mark.parametrize("mirrored_only", [False, True])
def test_weighted_interior_zero_declines_candidate_power_reappearance(mirrored_only):
    freqs = np.arange(-4., 4.)
    # Both reference edges are finite; an edge-only validity check misses this
    # interior support hole. With asymmetric weights V also needs the other side.
    h0 = np.ones((1,8), complex)
    h0[:, freqs == 2] = 0.
    candidate = np.ones_like(h0)
    weighted = freqs == (-2 if mirrored_only else 2)
    dets = {"H1": dict(d=np.zeros(8, complex), w=weighted.astype(float), n_shift=0)}
    if mirrored_only:
        mirror = (len(freqs)-np.arange(len(freqs))) % len(freqs)
        reference = 2*np.sum(h0[:,mirror]*h0*weighted)
        exact = 2*np.sum(candidate[:,mirror]*candidate*weighted)
    else:
        reference = 2*np.sum(abs(h0[:,weighted])**2)
        exact = 2*np.sum(abs(candidate[:,weighted])**2)
    assert reference == 0
    assert exact == 2
    rb = RelativeBinning(freqs, dets, 1, 1., 1/8., [1.,3.])
    with pytest.raises(UnsupportedFiducial, match="zero fiducial support"):
        rb.set_fiducial(h0)


def test_nyquist_clipping_collapses_automatic_mesh_cleanly():
    from RIFT import lalsimutils
    from RIFT.likelihood.jax_ile.relative_binning import phase_bins
    f = np.arange(-64., 64.)
    h = np.ones((1, len(f)), complex)
    overlap = lalsimutils.ComplexOverlap(63., 64., 64., 1., lambda _: 1., True)
    dets = {"H1": dict(d=np.zeros(len(f), complex), w=overlap.weights2side, n_shift=0)}
    np.testing.assert_array_equal(f[overlap.weights2side != 0], [-63., 63.])
    rb = RelativeBinning(f, dets, 1, 1., 1/128., phase_bins(63., 64., .3))
    assert len(rb.f_edges) == 1
    exact = reduce_qvu(jnp.asarray(h), jnp.zeros(len(f)), jnp.asarray(overlap.weights2side), 1., 0, 1)
    np.testing.assert_allclose(exact[1], [[4.]])
    with pytest.raises(UnsupportedFiducial, match="two distinct edges"):
        rb.set_fiducial(h)


@pytest.mark.parametrize("weighted_frequency", [-22., -19., 19., 22.])
def test_weighted_support_outside_snapped_span_declines(weighted_frequency):
    f = np.arange(-64., 64.)
    h0 = np.ones((1, len(f)), complex)
    h0[:, f == weighted_frequency] = 0.
    candidate = np.ones_like(h0)
    weight = (f == weighted_frequency).astype(float)
    assert np.sum(abs(h0)**2 * weight) == 0.
    assert np.sum(abs(candidate)**2 * weight) == 1.
    dets = {"H1": dict(d=np.zeros(len(f), complex), w=weight, n_shift=0)}
    rb = RelativeBinning(f, dets, 1, 1., 1/128., [20.4, 21.4])
    np.testing.assert_array_equal(rb.f_edges, [20., 21.])
    with pytest.raises(UnsupportedFiducial, match="weighted frequencies outside"):
        rb.set_fiducial(h0)


def test_actual_rounded_mask_is_covered_and_unweighted_zeros_are_harmless():
    from RIFT import lalsimutils
    from RIFT.likelihood.jax_ile.relative_binning import phase_bins
    f = np.arange(-64., 64.)
    overlap = lalsimutils.ComplexOverlap(20.4, 21.4, 64., 1., lambda _: 1., True)
    np.testing.assert_array_equal(f[overlap.weights2side != 0], [-20., 20.])
    assert not np.any(overlap.weights2side[np.abs(f) == 22.])
    h = ((np.abs(f) >= 20.) & (np.abs(f) <= 21.)).astype(complex)[None]
    dets = {"H1": dict(d=np.zeros(len(f), complex), w=overlap.weights2side, n_shift=0)}
    rb = RelativeBinning(f, dets, 1, 1., 1/128., phase_bins(20.4, 21.4, .3)).set_fiducial(h)
    np.testing.assert_array_equal(rb.f_edges, [20., 21.])
    actual = rb.qvu(h[:,rb.k_pos], h[:,rb.k_neg])["H1"]
    expected = reduce_qvu(jnp.asarray(h), jnp.zeros(len(f)), jnp.asarray(overlap.weights2side), 1., 0, 1)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-12)
