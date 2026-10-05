"""assign_param must give the same spin for any order of the coordinates in one spin system.

Truth is a random precessing P.  Its coordinates are extracted, assigned to a fresh P in every
order, and the Cartesian spins compared.  The systems are (chi, theta or cos_theta, phi) and
(s_z_bar, chi_perp_bar or chi_perp_u, phi).  Mixing Cartesian s1z with chi1_perp_bar is not
covered: s1z holds s1x, s1y fixed, so s1z after chi1_perp_bar changes chi1_perp_bar.
"""
import itertools

import numpy as np
import pytest

from RIFT import lalsimutils as lsu

N = 40
CART = ['s1x', 's1y', 's1z', 's2x', 's2y', 's2z']
SYSTEMS = {
    'spherical': ['chi{k}', 'theta{k}', 'phi{k}'],
    'cos': ['chi{k}', 'cos_theta{k}', 'phi{k}'],
    'bar': ['s{k}z_bar', 'chi{k}_perp_bar', 'phi{k}'],
    'u': ['s{k}z_bar', 'chi{k}_perp_u', 'phi{k}'],
}


def _truth(seed):
    rng = np.random.default_rng(seed)
    Ps = []
    for _ in range(N):
        P = lsu.ChooseWaveformParams()
        P.m1, P.m2 = np.sort(rng.uniform(5, 50, 2))[::-1]*lsu.lsu_MSUN
        for k in (1, 2):
            v = rng.normal(size=3)
            v *= rng.uniform(0.05, 0.99)/np.linalg.norm(v)
            for c, vc in zip('xyz', v):
                setattr(P, 's%d%s' % (k, c), vc)
        Ps.append(P)
    return Ps


def _cart(P):
    return np.array([getattr(P, n) for n in CART])


@pytest.mark.parametrize("system", list(SYSTEMS))
@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_any_order_recovers_spins(system, perm):
    names = ['mc', 'delta_mc'] + [SYSTEMS[system][i].format(k=k) for k in (1, 2) for i in perm]
    for P in _truth(7):
        Q = lsu.ChooseWaveformParams()
        for n in names:
            Q.assign_param(n, P.extract_param(n))
        assert np.allclose(_cart(Q), _cart(P), rtol=0, atol=1e-12), (names, _cart(Q), _cart(P))


def test_phi_before_theta_through_fallback(capsys):
    # the order reported against convert_waveform_coordinates: azimuth requested before theta1
    Ps = _truth(11)
    for P in Ps:
        P.s2x = P.s2y = 0.
    low = ['mc', 'delta_mc', 's2z', 'chi1', 'phi1', 'theta1']
    x_in = np.array([[P.extract_param(n) for n in low] for P in Ps])
    got = lsu.convert_waveform_coordinates(x_in, coord_names=CART, low_level_coord_names=low)
    assert "Fallthrough to non-vector-coords" in capsys.readouterr().out  # premise: the fallback ran
    want = np.array([_cart(P) for P in Ps])
    assert np.all(np.abs(want[:, 1]) > 1e-3)  # premise: s1y = 0 would be wrong on every row
    assert np.allclose(got, want, rtol=0, atol=1e-12)


def test_no_azimuth_request_keeps_azimuth_zero():
    P = lsu.ChooseWaveformParams()
    P.assign_param('chi1', 0.5)
    P.assign_param('theta1', 1.0)
    assert np.allclose(_cart(P)[:3], [0.5*np.sin(1.0), 0., 0.5*np.cos(1.0)], rtol=0, atol=1e-15)


def test_theta_on_zero_spin_is_finite():
    P = lsu.ChooseWaveformParams()
    P.assign_param('theta1', 1.0)
    P.assign_param('cos_theta2', 0.3)
    assert np.all(_cart(P) == 0)


def test_s_z_bar_holds_chi_perp_bar_and_phi():
    P = _truth(3)[0]
    before = [P.extract_param(n) for n in ['chi1_perp_bar', 'phi1', 'chi2_perp_bar', 'phi2']]
    P.assign_param('s1z_bar', -0.4)
    P.assign_param('s2z_bar', 0.7)
    after = [P.extract_param(n) for n in ['chi1_perp_bar', 'phi1', 'chi2_perp_bar', 'phi2']]
    assert P.s1z == -0.4 and P.s2z == 0.7
    assert np.allclose(after, before, rtol=0, atol=1e-12)


def test_azimuth_from_spin_once_it_has_one():
    # a recorded phi1 applies only while the in-plane spin is zero
    P = lsu.ChooseWaveformParams()
    P.assign_param('phi1', 2.0)
    P.s1x, P.s1z = 0.3, 0.1
    P.assign_param('chi1_perp_bar', 0.5)
    assert np.allclose(_cart(P)[:3], [0.5*np.sqrt(1-0.01), 0., 0.1], rtol=0, atol=1e-15)


def test_theta_hint_unused_on_nonzero_spin():
    P = lsu.ChooseWaveformParams()
    P.assign_param('theta1', 1.0)
    P.s1z = 0.5
    P.assign_param('chi1', 0.8)
    assert np.allclose(_cart(P)[:3], [0., 0., 0.8], rtol=0, atol=1e-15)


def test_s_z_bar_at_the_poles():
    P = lsu.ChooseWaveformParams()
    P.s1z = 1.
    P.assign_param('s1z_bar', 0.5)   # old s1z = 1: no chi1_perp_bar to hold
    assert np.all(np.isfinite(_cart(P))) and P.s1z == 0.5
    P = lsu.ChooseWaveformParams()
    P.s1x, P.s1y, P.s1z = 0.3, 0.4, 0.2
    P.assign_param('s1z_bar', 1.)    # in-plane spin goes to zero; its azimuth is kept
    P.assign_param('s1z_bar', 0.5)
    P.assign_param('chi1_perp_bar', 0.4)
    assert np.isclose(P.extract_param('phi1'), np.arctan2(0.4, 0.3), rtol=0, atol=1e-14)


def test_swap_components_swaps_requested_angles():
    P = lsu.ChooseWaveformParams()
    P.assign_param('theta1', 0.7)
    P.assign_param('phi1', 2.0)
    P.swap_components()
    P.assign_param('chi2', 0.5)
    P.assign_param('chi1', 0.5)
    want2 = 0.5*np.array([np.sin(0.7)*np.cos(2.0), np.sin(0.7)*np.sin(2.0), np.cos(0.7)])
    assert np.allclose(_cart(P)[3:], want2, rtol=0, atol=1e-15)
    assert np.allclose(_cart(P)[:3], [0., 0., 0.5], rtol=0, atol=1e-15)
