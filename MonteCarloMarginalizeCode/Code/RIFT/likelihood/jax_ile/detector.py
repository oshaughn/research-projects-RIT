"""
JAX ports of the detector-geometry helpers used by the ILE likelihood.

These are faithful re-expressions of
``RIFT.likelihood.vectorized_lal_tools.{ComputeDetAMResponse,
TimeDelayFromEarthCenter}`` in ``jax.numpy``.  The only structural change is
that JAX arrays are immutable, so the ``X[...,0] = ...`` style assignments of
the numpy original are replaced by ``jnp.stack`` of the three components.

The Greenwich Mean Sidereal Time (GMST) is *not* recomputed here: in the
production NoLoop path it is a single scalar obtained from the (fixed) fiducial
geocenter epoch via ``lal.GreenwichMeanSiderealTime(tref)``.  Time is
marginalized out in the likelihood, so ``tref`` is not a differentiable leaf;
we therefore take GMST as a host-side float constant, exactly as the reference
code does.
"""

import jax.numpy as jnp

# Speed of light, identical constant to vectorized_lal_tools (LAL value).
_NEG_C_SI = -299792458.0


def time_delay_from_earth_center(detector_location, ra, dec, gmst):
    """Geometric time delay (s) of arrival at a detector relative to geocenter.

    Mirrors ``vectorized_lal_tools.TimeDelayFromEarthCenter``.

    Parameters
    ----------
    detector_location : array_like, shape (3,)
        Detector position relative to Earth centre, metres (LAL ``det.location``).
    ra, dec : array_like, shape (S,)
        Source right ascension and declination, radians.
    gmst : float
        Greenwich mean sidereal time, radians (host scalar constant).

    Returns
    -------
    array_like, shape (S,)
        ``-(detector . ehat_src) / c`` , the LAL sign convention.
    """
    detector_location = jnp.asarray(detector_location)
    ra = jnp.asarray(ra)
    dec = jnp.asarray(dec)

    cos_dec = jnp.cos(dec)
    greenwich_hour_angle = gmst - ra

    ehat0 = cos_dec * jnp.cos(greenwich_hour_angle)
    ehat1 = -cos_dec * jnp.sin(greenwich_hour_angle)
    ehat2 = jnp.sin(dec)
    ehat_src = jnp.stack([ehat0, ehat1, ehat2], axis=-1)  # (S, 3)

    neg_separation = jnp.tensordot(ehat_src, detector_location, axes=([-1], [0]))
    return neg_separation / _NEG_C_SI


def compute_detamresponse(detector_response, ra, dec, psi, gmst):
    """Complex antenna response ``F = F_plus + i F_cross``.

    Mirrors ``vectorized_lal_tools.ComputeDetAMResponse`` exactly (same trig
    matrices and contraction order), returning a complex array so the rest of
    the likelihood can treat ``F`` uniformly.

    Parameters
    ----------
    detector_response : array_like, shape (3, 3)
        LAL detector response matrix (``det.response``).
    ra, dec, psi : array_like, shape (S,)
        Right ascension, declination, polarization angle, radians.
    gmst : float
        Greenwich mean sidereal time, radians (host scalar constant).

    Returns
    -------
    array_like (complex), shape (S,)
    """
    D = jnp.asarray(detector_response)
    ra = jnp.asarray(ra)
    dec = jnp.asarray(dec)
    psi = jnp.asarray(psi)

    source_greenwich = gmst - ra

    cos_gha = jnp.cos(source_greenwich)
    sin_gha = jnp.sin(source_greenwich)
    cos_dec = jnp.cos(dec)
    sin_dec = jnp.sin(dec)
    cos_psi = jnp.cos(psi)
    sin_psi = jnp.sin(psi)

    X0 = -cos_psi * sin_gha - sin_psi * cos_gha * sin_dec
    X1 = -cos_psi * cos_gha + sin_psi * sin_gha * sin_dec
    X2 = sin_psi * cos_dec
    X = jnp.stack([X0, X1, X2], axis=-1)  # (S, 3)

    Y0 = sin_psi * sin_gha - cos_psi * cos_gha * sin_dec
    Y1 = sin_psi * cos_gha + cos_psi * sin_gha * sin_dec
    Y2 = cos_psi * cos_dec
    Y = jnp.stack([Y0, Y1, Y2], axis=-1)  # (S, 3)

    # inner(X, D) -> (S, 3): contract last axis of X with last axis of D.
    XD = jnp.tensordot(X, D, axes=([-1], [-1]))
    YD = jnp.tensordot(Y, D, axes=([-1], [-1]))

    F_plus = (X * XD - Y * YD).sum(axis=-1)
    F_cross = (X * YD + Y * XD).sum(axis=-1)

    return F_plus + 1.0j * F_cross
