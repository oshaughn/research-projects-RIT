"""Regression tests for the detector-network sky coordinate frame."""

import warnings

import numpy as np
import lal
import lalsimulation as lalsim

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


def test_inverse_frame_matches_linalg_inv_without_deprecation():
    """frmInverse must invert frm to machine precision, with no deprecation."""
    epoch = lal.LIGOTimeGPS(1389783918.0)
    with warnings.catch_warnings():
        # np.matrix (the old way of building frmInverse) raises
        # PendingDeprecationWarning; nothing here may reintroduce it.
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", PendingDeprecationWarning)
        sky_rotations.assign_sky_frame("H1", "L1", epoch)

    frm = sky_rotations.frm
    frm_inverse = sky_rotations.frmInverse
    assert frm_inverse.dtype == np.float64
    # an independent array, not a view that aliases frm
    assert not np.shares_memory(frm, frm_inverse)
    # frm is orthonormal only to floating-point precision, so the residual is
    # set by frm itself: require the transpose to be as good as np.linalg.inv.
    reference = np.abs(np.linalg.inv(frm) @ frm - np.eye(3)).max()
    assert np.abs(frm_inverse @ frm - np.eye(3)).max() <= max(reference, 1e-15)
    assert np.abs(frm @ frm_inverse - np.eye(3)).max() <= max(reference, 1e-15)
    assert np.allclose(frm_inverse, np.linalg.inv(frm), rtol=0.0, atol=1e-15)


def test_network_polar_angle_is_the_time_delay_coordinate():
    """The decisive property, checked against lal's own arrival-time model.

    --internal-sky-network-coordinates exists so that the sampler's polar angle
    IS the detector time-delay coordinate.  Sweeping the other network angle at
    fixed theta must therefore leave the H1-L1 arrival-time difference
    unchanged.  Rotating with the forward frame instead of its inverse breaks
    this by more than the 10 ms HL light travel time.
    """
    epoch = lal.LIGOTimeGPS(1389783918.0)
    sky_rotations.assign_sky_frame("H1", "L1", epoch)
    loc_h = lalsim.DetectorPrefixToLALDetector("H1").location
    loc_l = lalsim.DetectorPrefixToLALDetector("L1").location

    for cos_theta_network in (-0.8, -0.3, 0.0, 0.25, 0.7):
        theta_network = np.full(64, np.arccos(cos_theta_network))
        phi_network = np.linspace(0.0, 2 * np.pi, 64, endpoint=False)
        theta, phi = sky_rotations.network_to_physical(
            theta_network, phi_network, xpy=np
        )
        delays = np.array([
            lal.ArrivalTimeDiff(loc_h, loc_l, float(p), float(np.pi / 2 - t), epoch)
            for t, p in zip(theta, phi)
        ])
        assert np.ptp(delays) < 1e-11
