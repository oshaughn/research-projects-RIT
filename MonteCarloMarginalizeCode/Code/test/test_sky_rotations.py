"""Regression tests for the detector-network sky coordinate frame."""

import numpy as np
import lal

from RIFT import lalsimutils
from RIFT.misc import sky_rotations


def _angular_separation(theta_a, phi_a, theta_b, phi_b):
    """Return the angle between two directions without RA wrap ambiguity."""
    a = lalsimutils.nhat(theta_a, phi_a)
    b = lalsimutils.nhat(theta_b, phi_b)
    return np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))


def test_physical_to_network_polar_angle_is_baseline_projection():
    """Network-frame cos(theta) is n dot the detector baseline at the epoch."""
    epoch = lal.LIGOTimeGPS(1389783918.0)
    sky_rotations.assign_sky_frame("H1", "L1", epoch)

    theta = np.deg2rad(90.0 - 42.0195)
    phi = np.deg2rad(167.5254)
    theta_network, _ = sky_rotations.physical_to_network(theta, phi, xpy=np)

    expected = np.dot(lalsimutils.nhat(theta, phi), sky_rotations.vecZnew)
    assert np.isclose(np.cos(theta_network), expected, rtol=0.0, atol=1e-12)


def test_network_to_physical_round_trip_uses_inverse_frame():
    """The inverse frame maps sampled network angles back to equatorial sky."""
    epoch = lal.LIGOTimeGPS(1389783918.0)
    sky_rotations.assign_sky_frame("H1", "L1", epoch)

    for dec_deg, ra_deg in ((42.0195, 167.5254), (14.7999, 135.8866)):
        theta = np.deg2rad(90.0 - dec_deg)
        phi = np.deg2rad(ra_deg)
        theta_network, phi_network = sky_rotations.physical_to_network(
            theta, phi, xpy=np
        )
        theta_out, phi_out = sky_rotations.network_to_physical(
            theta_network, phi_network, xpy=np
        )

        assert _angular_separation(theta, phi, theta_out, phi_out) < 1e-7
