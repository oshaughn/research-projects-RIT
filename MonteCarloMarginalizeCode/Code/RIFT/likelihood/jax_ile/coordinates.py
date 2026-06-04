"""
Network-frame sky coordinates for the JAX ILE extrinsic likelihood.

WHY THIS EXISTS
---------------
For a gravitational-wave network dominated by two detectors, the extrinsic
likelihood over sky position ``(ra, dec)`` is strongly *multimodal* along the
inter-detector "time-delay ring".  The arrival-time difference between two
detectors,

    dt = tau_2(ra,dec) - tau_1(ra,dec)
       = -( (loc_2 - loc_1) . ehat_src ) / c ,

depends on the source direction ``ehat_src`` *only* through its projection onto
the baseline ``b = loc_2 - loc_1``.  Equivalently, ``dt`` depends only on the
angle between the source and the baseline, so it is constant around the whole
cone (ring on the sky) of directions making that fixed angle with ``b``.  The
likelihood is therefore nearly degenerate around that ring, producing several
discrete sky blobs that confound samplers.

THE FIX (cf. RIFT --internal-sky-network-coordinates, Princeton "cogwheel")
---------------------------------------------------------------------------
Rotate to a frame whose polar (z) axis is the unit baseline direction.  In that
"network frame" the angle to the baseline IS the polar angle ``theta_n``, so

    cos(theta_n) = ehat_src . bhat  ==>  dt = -|b| cos(theta_n) / c

is a *monotonic, one-to-one* function of ``cos(theta_n)`` and is completely
*independent* of the network azimuth ``phi_n``.  The time-delay ring is folded
onto a single line of constant ``theta_n``; the hard 2D-multimodal sky problem
becomes well structured (sample in ``(cos theta_n, phi_n)``, rotate back to
``(ra, dec)``).

COORDINATE CONVENTION (must match detector.py)
----------------------------------------------
``detector.time_delay_from_earth_center`` represents the source unit vector as

    ehat_src = [ cos(dec) cos(gha), -cos(dec) sin(gha), sin(dec) ],
    gha = gmst - ra,

and contracts it with ``detector_location`` (LAL ``det.location``, Earth-fixed
ECEF metres).  Note that with ``gha = gmst - ra`` we have

    cos(gha) =  cos(ra - gmst),   -sin(gha) = sin(ra - gmst),

so ``ehat_src`` is exactly the **Earth-Centred Earth-Fixed (ECEF)** unit vector
of a source whose ECEF longitude is ``(ra - gmst)`` and latitude ``dec``:

    ehat_src = [ cos(dec) cos(ra-gmst), cos(dec) sin(ra-gmst), sin(dec) ].

Because ``detector_location`` is *also* ECEF, the dot product in
``time_delay_from_earth_center`` is a genuine same-frame contraction.  Therefore
the consistent frame in which to build the network basis is simply the ECEF
frame: the baseline ``loc_2 - loc_1`` is already an ECEF vector and is used
*as-is* (no rotation by ``gmst`` is needed, because we represent the source the
same ECEF way and rotate it by ``R`` before reading off angles).  ``gmst`` only
enters when converting between ``(ra, dec)`` and the ECEF source vector.

All functions are pure ``jax.numpy`` and automatic-differentiation friendly.
"""

import jax.numpy as jnp

# Speed of light (LAL value), matching detector.py.
_C_SI = 299792458.0


# ---------------------------------------------------------------------------
# Source-direction <-> (ra, dec) in the ECEF frame used by detector.py
# ---------------------------------------------------------------------------
def radec_to_ehat(ra, dec, gmst):
    """ECEF source unit vector, identical to detector.py's ``ehat_src``.

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination, radians.
    gmst : float
        Greenwich mean sidereal time, radians.

    Returns
    -------
    array_like, shape (..., 3)
        ``[cos(dec)cos(gha), -cos(dec)sin(gha), sin(dec)]`` with ``gha=gmst-ra``.
    """
    ra = jnp.asarray(ra)
    dec = jnp.asarray(dec)
    cos_dec = jnp.cos(dec)
    gha = gmst - ra
    ehat0 = cos_dec * jnp.cos(gha)
    ehat1 = -cos_dec * jnp.sin(gha)
    ehat2 = jnp.sin(dec)
    return jnp.stack([ehat0, ehat1, ehat2], axis=-1)


def ehat_to_radec(ehat, gmst):
    """Inverse of :func:`radec_to_ehat`: ECEF unit vector -> ``(ra, dec)``.

    Returns ``ra`` wrapped to ``[0, 2*pi)`` and ``dec`` in ``[-pi/2, pi/2]``.
    """
    ehat = jnp.asarray(ehat)
    ex, ey, ez = ehat[..., 0], ehat[..., 1], ehat[..., 2]
    dec = jnp.arcsin(jnp.clip(ez, -1.0, 1.0))
    # gha = gmst - ra ; ex = cos(dec)cos(gha), ey = -cos(dec)sin(gha)
    #   => cos(gha) ~ ex, sin(gha) ~ -ey  => gha = atan2(-ey, ex)
    gha = jnp.arctan2(-ey, ex)
    ra = jnp.mod(gmst - gha, 2.0 * jnp.pi)
    return ra, dec


# ---------------------------------------------------------------------------
# Build the network frame
# ---------------------------------------------------------------------------
def build_network_frame(loc1, loc2, gmst):
    """Build an orthonormal network frame from two detector locations.

    The polar (z) axis is the unit baseline direction ``bhat = (loc2-loc1)/|.|``
    expressed in the **same ECEF frame** as ``detector.time_delay_from_earth_center``'s
    ``ehat_src`` (see module docstring: that frame *is* ECEF, so the baseline is
    used directly with no ``gmst`` rotation).

    The two transverse axes (network x, y) are chosen by a stable, smooth
    construction: pick a reference vector not parallel to ``bhat`` and
    Gram-Schmidt.  Their absolute orientation is arbitrary (it only fixes the
    zero of the network azimuth ``phi_n``); what matters physically is that
    ``z`` is the baseline, so that ``cos(theta_n) = ehat_src . bhat`` controls
    the inter-detector delay.

    Parameters
    ----------
    loc1, loc2 : array_like, shape (3,)
        Detector positions, ECEF metres (LAL ``det.location``).
    gmst : float
        Greenwich mean sidereal time, radians.  (Accepted for interface
        symmetry / documentation; the ECEF baseline does not depend on it.)

    Returns
    -------
    R : array_like, shape (3, 3)
        Orthonormal rotation matrix whose ROWS are the network basis vectors
        ``(xhat_n, yhat_n, zhat_n=bhat)`` in ECEF components.  For an ECEF
        vector ``v``, ``R @ v`` gives its components in the network frame, and
        ``R.T @ v_network`` maps back.  ``R @ R.T = I`` and ``det(R) = +1``.
    """
    del gmst  # not needed: baseline and ehat_src share the ECEF frame
    loc1 = jnp.asarray(loc1, dtype=jnp.result_type(float))
    loc2 = jnp.asarray(loc2, dtype=jnp.result_type(float))

    baseline = loc2 - loc1
    zhat = baseline / jnp.linalg.norm(baseline)

    # Reference vector for Gram-Schmidt: pick the global axis least aligned with
    # zhat (smooth + numerically stable; avoids the singular near-parallel case).
    # Use ECEF e_z = [0,0,1] unless zhat is nearly polar, else e_x.
    ref_z = jnp.array([0.0, 0.0, 1.0])
    ref_x = jnp.array([1.0, 0.0, 0.0])
    use_x = jnp.abs(zhat[2]) > 0.9
    ref = jnp.where(use_x, ref_x, ref_z)

    xhat = ref - zhat * jnp.dot(ref, zhat)
    xhat = xhat / jnp.linalg.norm(xhat)
    yhat = jnp.cross(zhat, xhat)  # right-handed: x cross y = z

    R = jnp.stack([xhat, yhat, zhat], axis=0)  # rows = basis vectors
    return R


# ---------------------------------------------------------------------------
# (ra, dec) <-> network angles
# ---------------------------------------------------------------------------
def equatorial_to_network(ra, dec, R, gmst):
    """Map ``(ra, dec)`` to network polar/azimuth angles ``(theta_n, phi_n)``.

    Parameters
    ----------
    ra, dec : array_like
        Right ascension and declination, radians.
    R : array_like, shape (3, 3)
        Network frame matrix from :func:`build_network_frame`.
    gmst : float
        Greenwich mean sidereal time, radians.

    Returns
    -------
    theta_n : array_like
        Network polar angle in ``[0, pi]`` (angle from the baseline ``zhat``).
    phi_n : array_like
        Network azimuth in ``[0, 2*pi)``.
    """
    ehat = radec_to_ehat(ra, dec, gmst)            # (..., 3) ECEF
    v = jnp.tensordot(ehat, R, axes=([-1], [-1]))  # components in network frame
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    theta_n = jnp.arccos(jnp.clip(vz, -1.0, 1.0))
    phi_n = jnp.mod(jnp.arctan2(vy, vx), 2.0 * jnp.pi)
    return theta_n, phi_n


def network_to_equatorial(theta_n, phi_n, R, gmst):
    """Inverse of :func:`equatorial_to_network`: network angles -> ``(ra, dec)``.

    Parameters
    ----------
    theta_n, phi_n : array_like
        Network polar and azimuth angles, radians.
    R : array_like, shape (3, 3)
        Network frame matrix from :func:`build_network_frame`.
    gmst : float
        Greenwich mean sidereal time, radians.

    Returns
    -------
    ra : array_like in ``[0, 2*pi)``
    dec : array_like in ``[-pi/2, pi/2]``
    """
    theta_n = jnp.asarray(theta_n)
    phi_n = jnp.asarray(phi_n)
    sin_t = jnp.sin(theta_n)
    v = jnp.stack([sin_t * jnp.cos(phi_n),
                   sin_t * jnp.sin(phi_n),
                   jnp.cos(theta_n)], axis=-1)      # network-frame components
    # Back to ECEF: ehat = R.T @ v  (rows of R are basis vectors).
    ehat = jnp.tensordot(v, R, axes=([-1], [0]))
    return ehat_to_radec(ehat, gmst)


def network_costheta_to_delay(cos_theta_n, baseline_length):
    """Inter-detector arrival-time difference (s) from ``cos(theta_n)``.

    ``dt = tau_2 - tau_1 = -|b| cos(theta_n) / c``.  Monotonic in
    ``cos(theta_n)`` and independent of ``phi_n`` -- this is the whole point of
    the network frame.
    """
    return -baseline_length * jnp.asarray(cos_theta_n) / _C_SI


# ---------------------------------------------------------------------------
# Polarization (psi) + reference-phase (phiref) degeneracy folding
# ---------------------------------------------------------------------------
def polarization_phase_fold(psi, phiref):
    """Fold ``(psi, phiref)`` into the canonical fundamental domain.

    DEGENERACY (dominant quadrupole)
    --------------------------------
    For a signal dominated by the ``(l,m) = (2, +/-2)`` modes, the detector
    strain enters through the two combinations

        h(t) ~ F+ h_+ + Fx h_x,
        h_+ ~ (1+cos^2 i)/2 cos(2 phiref + ...),  h_x ~ cos i sin(2 phiref + ...),

    while the antenna factors ``F+(psi), Fx(psi)`` depend on ``psi`` only through
    ``cos(2 psi)`` and ``sin(2 psi)``.  Two exact discrete symmetries of the
    *dominant* (2,2)/(2,-2) likelihood follow:

    1. ``psi  -> psi  + pi``   leaves ``(cos 2psi, sin 2psi)`` invariant
       => the likelihood is **pi-periodic in psi** (``2*psi`` periodicity).
    2. ``phiref -> phiref + pi`` flips the sign of both ``cos(2 phiref)`` and
       ``sin(2 phiref)``; this is undone by ``psi -> psi + pi/2`` (which sends
       ``F+ -> -F+``, ``Fx -> -Fx``).  Hence the *joint* map

           (psi, phiref) -> (psi + pi/2, phiref + pi/2)            [see note]

       is (to the dominant-mode approximation) a likelihood symmetry, so psi may
       be folded from ``[0, pi)`` into the **fundamental domain ``[0, pi/2)``**
       by ``psi -> psi - pi/2`` accompanied by ``phiref -> phiref + pi/2``.

    Note: the precise companion shift in ``phiref`` depends on the phase
    convention; here we apply the conventional ``+pi/2`` partner shift.  The
    folding is exact for the dominant quadrupole and approximate once higher
    harmonics (which break the ``2*psi`` symmetry) are included.

    FUNDAMENTAL DOMAIN
    ------------------
    Returns ``psi_f in [0, pi/2)`` and ``phiref_f in [0, 2*pi)``.

    PROPERTIES (verified in ``__main__``)
    -------------------------------------
    * Idempotent: ``fold(fold(x)) == fold(x)``.
    * Range: output ``psi_f`` always lies in ``[0, pi/2)``.

    Parameters
    ----------
    psi, phiref : array_like
        Polarization angle and reference phase, radians.

    Returns
    -------
    psi_f, phiref_f : array_like
        Folded values.
    """
    psi = jnp.asarray(psi)
    phiref = jnp.asarray(phiref)

    # Step 1: psi is pi-periodic -> bring into [0, pi).
    psi_mod = jnp.mod(psi, jnp.pi)

    # Step 2: if psi in [pi/2, pi), subtract pi/2 and shift phiref by +pi/2.
    need_shift = psi_mod >= (jnp.pi / 2.0)
    psi_f = jnp.where(need_shift, psi_mod - jnp.pi / 2.0, psi_mod)
    phiref_f = jnp.where(need_shift, phiref + jnp.pi / 2.0, phiref)

    # phiref is 2*phiref-periodic in amplitude, but the observable phase is
    # 2*pi-periodic; wrap to [0, 2*pi).
    phiref_f = jnp.mod(phiref_f, 2.0 * jnp.pi)
    return psi_f, phiref_f


# ===========================================================================
# Demonstration / validation
# ===========================================================================
if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import lal
    import lalsimulation

    from RIFT.likelihood.jax_ile import detector

    print("=" * 70)
    print("jax_ile.coordinates  --  network-frame sky coordinates validation")
    print("=" * 70)

    # Fixed sidereal time.
    tref = 1000000000.0
    gmst = float(lal.GreenwichMeanSiderealTime(tref))

    loc_H1 = jnp.asarray(
        lalsimulation.DetectorPrefixToLALDetector("H1").location)
    loc_L1 = jnp.asarray(
        lalsimulation.DetectorPrefixToLALDetector("L1").location)

    R = build_network_frame(loc_H1, loc_L1, gmst)
    baseline_len = float(jnp.linalg.norm(loc_L1 - loc_H1))

    # --- orthonormality of R -------------------------------------------------
    ortho_err = float(jnp.max(jnp.abs(R @ R.T - jnp.eye(3))))
    det_R = float(jnp.linalg.det(R))
    print(f"\n[frame] |R R^T - I|_max = {ortho_err:.2e}, det(R) = {det_R:+.6f}")
    print(f"[frame] H1-L1 baseline length = {baseline_len:.1f} m "
          f"({baseline_len / _C_SI * 1e3:.3f} ms light-travel)")

    # --- round-trip ----------------------------------------------------------
    rng = np.random.default_rng(0)
    n = 5000
    ra = jnp.asarray(rng.uniform(0.0, 2 * np.pi, n))
    dec = jnp.asarray(np.arcsin(rng.uniform(-1.0, 1.0, n)))

    theta_n, phi_n = equatorial_to_network(ra, dec, R, gmst)
    ra2, dec2 = network_to_equatorial(theta_n, phi_n, R, gmst)

    # Compare via the unit vectors (robust to ra wrap at the poles).
    e1 = radec_to_ehat(ra, dec, gmst)
    e2 = radec_to_ehat(ra2, dec2, gmst)
    rt_err = float(jnp.max(jnp.linalg.norm(e1 - e2, axis=-1)))
    print(f"\n[round-trip] max |ehat(ra,dec) - ehat(roundtrip)| = {rt_err:.2e}")

    # --- KEY PHYSICS CHECK: delay depends on theta_n only -------------------
    def hl_delay(ra_, dec_):
        tau_H = detector.time_delay_from_earth_center(loc_H1, ra_, dec_, gmst)
        tau_L = detector.time_delay_from_earth_center(loc_L1, ra_, dec_, gmst)
        return tau_L - tau_H  # H1-L1 difference

    print("\n[physics] H1-L1 delay should be CONSTANT vs phi_n at fixed theta_n,")
    print("          and VARY with theta_n:")
    phi_scan = jnp.asarray(np.linspace(0.0, 2 * np.pi, 360, endpoint=False))
    max_spread = 0.0
    print(f"   {'theta_n [rad]':>14} {'delay [s]':>16} "
          f"{'spread over phi_n [s]':>22}  {'predicted [s]':>16}")
    for th in [0.3, 0.9, 1.5708, 2.2, 2.9]:
        th_arr = jnp.full_like(phi_scan, th)
        ra_s, dec_s = network_to_equatorial(th_arr, phi_scan, R, gmst)
        d = hl_delay(ra_s, dec_s)
        spread = float(jnp.max(d) - jnp.min(d))
        max_spread = max(max_spread, spread)
        predicted = network_costheta_to_delay(np.cos(th), baseline_len)
        print(f"   {th:>14.4f} {float(jnp.mean(d)):>16.9e} "
              f"{spread:>22.2e}  {float(predicted):>16.9e}")
    print(f"\n[physics] worst-case delay spread over phi_n (any theta_n) = "
          f"{max_spread:.2e} s  (target < 1e-9 s)")

    # --- polarization/phase folding -----------------------------------------
    psi = jnp.asarray(rng.uniform(-2 * np.pi, 2 * np.pi, 1000))
    phiref = jnp.asarray(rng.uniform(-2 * np.pi, 2 * np.pi, 1000))
    psi_f, phiref_f = polarization_phase_fold(psi, phiref)
    psi_ff, phiref_ff = polarization_phase_fold(psi_f, phiref_f)
    idem_err = float(jnp.max(jnp.abs(psi_f - psi_ff))
                     + jnp.max(jnp.abs(jnp.mod(phiref_f - phiref_ff + np.pi,
                                               2 * np.pi) - np.pi)))
    in_domain = bool(jnp.all((psi_f >= 0.0) & (psi_f < np.pi / 2.0)))
    print("\n[pol-fold] psi folded into [0, pi/2):  in_domain =", in_domain)
    print(f"[pol-fold] idempotency error = {idem_err:.2e}")

    # --- summary -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  frame orthonormality error : {ortho_err:.2e}")
    print(f"  round-trip (ra,dec) error  : {rt_err:.2e}   (target ~1e-12)")
    print(f"  delay spread over phi_n    : {max_spread:.2e} s (target <1e-9 s)")
    print(f"  pol-fold idempotent / in-domain : "
          f"{idem_err:.2e} / {in_domain}")
    ok = (ortho_err < 1e-12 and rt_err < 1e-11 and max_spread < 1e-9
          and in_domain and idem_err < 1e-12)
    print(f"  ALL CHECKS PASS : {ok}")
    print("=" * 70)
