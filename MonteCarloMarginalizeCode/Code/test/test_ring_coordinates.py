"""phi12 and the ring coordinates (chi1_perp, chi2_perp, SOverM2_perp, DeltaOverM2_perp, chi_p_vec):
the vectorized spherical path of convert_waveform_coordinates must agree with extract_param."""
import numpy as np
import lal
import RIFT.lalsimutils as lsu

RING = ['chi1_perp', 'chi2_perp', 'phi12', 'SOverM2_perp', 'DeltaOverM2_perp', 'chi_p_vec']
LOW = ['mc', 'delta_mc', 'chi1', 'cos_theta1', 'phi1', 'chi2', 'cos_theta2', 'phi2']


def _draws(n=300, seed=4):
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.uniform(5, 30, n), rng.uniform(0.01, 0.8, n), rng.uniform(0, 0.99, n),
                         rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 0.99, n),
                         rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n)])
    return x


def test_vectorized_matches_extract_param():
    x = _draws()
    y = lsu.convert_waveform_coordinates(x, coord_names=RING, low_level_coord_names=LOW)
    for i, row in enumerate(x):
        mc, dmc, c1, ct1, p1, c2, ct2, p2 = row
        eta = 0.25 * (1 - dmc ** 2)
        m1, m2 = lsu.m1m2(mc, eta)
        P = lsu.ChooseWaveformParams()
        P.m1, P.m2 = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
        s1 = c1 * np.sqrt(1 - ct1 ** 2); s2 = c2 * np.sqrt(1 - ct2 ** 2)
        P.s1x, P.s1y, P.s1z = s1 * np.cos(p1), s1 * np.sin(p1), c1 * ct1
        P.s2x, P.s2y, P.s2z = s2 * np.cos(p2), s2 * np.sin(p2), c2 * ct2
        for j, name in enumerate(RING):
            ref = P.extract_param(name)
            d = abs(y[i, j] - ref)
            if name == 'phi12':
                d = min(d, 2 * np.pi - d)
            assert d < 1e-9, (name, y[i, j], ref)


def test_chi_p_vec_limits():
    # one spin: chi_p_vec equals chi1_perp; opposite in-plane spins of the weighted size cancel
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 10 * lal.MSUN_SI, 5 * lal.MSUN_SI
    P.s1x, P.s1y, P.s2x, P.s2y = 0.4, 0.0, 0.0, 0.0
    assert abs(P.extract_param('chi_p_vec') - 0.4) < 1e-12
    q = 0.5; A1 = 2 + 1.5 * q; A2 = 2 + 1.5 / q
    P.s2x = -0.4 / ((A2 / A1) * q ** 2)
    if abs(P.s2x) < 1:
        assert P.extract_param('chi_p_vec') < 1e-12
    assert abs(P.extract_param('phi12') - np.pi) < 1e-12


def _one(mc, dmc, c1, ct1, p1, c2, ct2, p2):
    return np.array([[mc, dmc, c1, ct1, p1, c2, ct2, p2]])


def test_phi12_sign_both_paths():
    # spin 2 60 degrees ahead of spin 1: phi12 = pi/3, not 5 pi/3 (a sign flip in both paths fails here)
    x = _one(10., 0.3, 0.5, 0., 0.2, 0.5, 0., 0.2 + np.pi / 3)
    y = lsu.convert_waveform_coordinates(x, coord_names=['phi12'], low_level_coord_names=LOW)
    assert abs(y[0, 0] - np.pi / 3) < 1e-12
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 10 * lal.MSUN_SI, 5 * lal.MSUN_SI
    P.s1x, P.s1y = 0.5 * np.cos(0.2), 0.5 * np.sin(0.2)
    P.s2x, P.s2y = 0.5 * np.cos(0.2 + np.pi / 3), 0.5 * np.sin(0.2 + np.pi / 3)
    assert abs(P.extract_param('phi12') - np.pi / 3) < 1e-12


def test_object_array_input():
    # CIP's default sampler (adaptive_cartesian) passes an object array of python floats
    x = _draws(20).astype(object)
    y = lsu.convert_waveform_coordinates(x, coord_names=RING, low_level_coord_names=LOW)
    y_ref = lsu.convert_waveform_coordinates(x.astype(float), coord_names=RING, low_level_coord_names=LOW)
    assert np.allclose(np.asarray(y, dtype=float), y_ref, atol=1e-12)


def test_phi12_zero_inplane_spin():
    # phi12 is undefined without an in-plane component; both paths return 0
    for x in (_one(10., 0.3, 0.0, 0.5, 1.5, 0.5, 0.2, 2.5), _one(10., 0.3, 0.5, 0.2, 1.5, 0.5, -1.0, 2.5)):
        y = lsu.convert_waveform_coordinates(x, coord_names=['phi12'], low_level_coord_names=LOW)
        assert y[0, 0] == 0.
    P = lsu.ChooseWaveformParams()
    P.m1, P.m2 = 10 * lal.MSUN_SI, 5 * lal.MSUN_SI
    P.s1x = P.s1y = 0.
    P.s2x, P.s2y = 0.3, 0.1
    assert P.extract_param('phi12') == 0.


def test_enforce_kerr_in_ring_block():
    # the vectorized block can end the conversion early; it applies the fallthrough's Kerr rule itself
    x = np.vstack([_one(10., 0.3, 1.2, 0.2, 1.5, 0.5, 0.2, 2.5), _one(10., 0.3, 0.5, 0.2, 1.5, 0.5, 0.2, 2.5)])
    y = lsu.convert_waveform_coordinates(x, coord_names=['mc', 'delta_mc', 'chi1_perp', 'chi_p_vec'],
                                         low_level_coord_names=LOW, enforce_kerr=True)
    assert np.all(y[0] == -np.inf) and np.all(np.isfinite(y[1]))
    y = lsu.convert_waveform_coordinates(x, coord_names=['chi1_perp'], low_level_coord_names=LOW, enforce_kerr=False)
    assert np.all(np.isfinite(y))


def test_chi_p_vec_not_assignable():
    # grid readers call assign_param on every column named in valid_params; chi_p_vec is derived only
    assert 'chi_p_vec' not in lsu.valid_params
