"""Independent check that FEW polarizations enter the response as hp + i hc.

The response is fed ``hp + 1j*hc`` in the FLR basis
u = (sin lam, -cos lam, 0), v = (-sin b cos lam, -sin b sin lam, cos b),
with propagation k = u x v pointing from the source to the SSB.  Two facts fix
the sign without reference to either code's own convention:

1. FLR: in the long-wavelength limit each link responds to n.h.n, so for a
   pure-plus and a pure-cross input with the same time series the link
   outputs are in the ratio (n.e+.n)/(n.ex.n), e+ = uu - vv, ex = uv + vu.
2. Physics: for an orbit whose angular momentum points along k (toward the
   observer) the mass quadrupole rotates counterclockwise in the (u, v) plane,
   so hp*dhc/dt - hc*dhp/dt > 0.  The opposite orientation gives < 0.  This is
   independent of the polarization-angle convention.

Opt-in: requires the pinned FEW model data and orbit file (see the response
and adapter tests).
"""

import os
from dataclasses import replace

import numpy as np
import pytest

needs_runtime = pytest.mark.skipif(
    os.environ.get('RIFT_RUN_FEW_REAL') != '1' or not os.environ.get('RIFT_EMRI_ORBIT_FILE'),
    reason='requires prefetched FEW 2.2.0 data and the pinned orbit file; explicit opt-in')


def flr_basis(lam, beta):
    u = np.array([np.sin(lam), -np.cos(lam), 0.])
    v = np.array([-np.sin(beta)*np.cos(lam), -np.sin(beta)*np.sin(lam), np.cos(beta)])
    return u, v


def test_flr_basis_is_right_handed_about_propagation():
    lam, beta = 2.0, 0.4
    u, v = flr_basis(lam, beta)
    n_source = np.array([np.cos(beta)*np.cos(lam), np.cos(beta)*np.sin(lam), np.sin(beta)])
    np.testing.assert_allclose(np.cross(u, v), -n_source, atol=1e-15)


@needs_runtime
def test_flr_link_response_uses_plus_and_cross_tensors():
    from fastlisaresponse import pyResponseTDI
    from lisatools.detector import Orbits

    count, dt, trim = 4096, 10., 1000
    lam, beta = 2.0, np.pi/2 - 1.0
    t = np.arange(count) * dt
    signal = 1e-20 * np.sin(2*np.pi*1e-4*t)
    orbits = Orbits(filename=os.environ['RIFT_EMRI_ORBIT_FILE'], force_backend='cpu')
    outputs = []
    for h in (signal + 0j, 1j*signal):
        model = pyResponseTDI(1/dt, count, orbits=orbits, force_backend='cpu',
                              tdi_chan='XYZ', order=25, tdi='1st generation')
        model.get_projections(h, lam, beta, t0=trim*dt)
        outputs.append(np.asarray(model.y_gw))
        links, response_orbits = model.response_orbits.LINKS, model.response_orbits
    y_plus, y_cross = outputs
    u, v = flr_basis(lam, beta)
    e_plus, e_cross = np.outer(u, u) - np.outer(v, v), np.outer(u, v) + np.outer(v, u)
    t_mid = float(count*dt/2)
    checked = 0
    for i, link in enumerate(links):
        n = np.asarray(response_orbits.get_normal_unit_vec(t_mid, int(link))).ravel()
        p_plus, p_cross = n @ e_plus @ n, n @ e_cross @ n
        if abs(p_cross) < 0.1 * abs(p_plus) or abs(p_plus) < 0.1 * abs(p_cross):
            continue
        measured = np.dot(y_plus[i], y_cross[i]) / np.dot(y_cross[i], y_cross[i])
        assert measured == pytest.approx(p_plus/p_cross, rel=0.05), link
        checked += 1
    assert checked >= 2


@needs_runtime
def test_few_orbit_toward_observer_rotates_counterclockwise_in_flr_basis():
    from RIFT.LISA import few_adapter as fa

    qS, phiS = 1.0, 2.0
    base = fa.EMRIParameters(1e6, 10., 0., 12., 0.2, 1., 1.,
                             qS, phiS, 0., 0., 0.2, 0., 0.4)
    adapter = fa.FewWaveformAdapter(fa.FEWConfig(4096, 10., 0., 'test mission epoch, seconds'))

    def circulation(qK, phiK):
        hp, hc = adapter(replace(base, qK=qK, phiK=phiK)).polarizations
        dhp, dhc = np.gradient(hp), np.gradient(hc)
        return np.mean(hp*dhc - hc*dhp) / np.sqrt(
            np.mean(hp**2 + hc**2) * np.mean(dhp**2 + dhc**2))

    # Spin (and, for xI0=1, orbital angular momentum) along k = -n_source.
    toward = circulation(np.pi - qS, (phiS + np.pi) % (2*np.pi))
    away = circulation(qS, phiS)
    assert toward > 0.5
    assert away < -0.5
