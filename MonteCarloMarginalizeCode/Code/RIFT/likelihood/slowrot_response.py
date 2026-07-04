"""
slowrot_response : closed-form sidereal-harmonic decomposition of the ground-based
detector response, for the "slow-rotation" generalization of the RIFT likelihood.

Over a long signal the Earth rotates, so the antenna response and the geometric
propagation delay drift.  Both are exactly band-limited in harmonics of the Greenwich
hour angle g(t) = GMST(t) - RA:

    F(t)   = F_+(t) + i F_x(t) = sum_{n=-2}^{+2} A_n  exp(i n g(t))          (5 harmonics)
    tau(t) = -(1/c) r_det . nhat(t)  = sum_{n=-1}^{+1} B_n exp(i n g(t))     (3 harmonics)

The key structural facts (see notes/ sec_amplitude, sec_delay, and the response appendix):

  * The complex antenna harmonics A_n depend ONLY on the detector response tensor, the
    source declination, and the polarization angle -- NOT on right ascension or time.
    RA and time enter only through the phase exp(i n g(t)) = exp(i n (GMST(t)-RA)).
    Since GMST(t) ~ GMST(t_ref) + Omega (t - t_ref), this is exactly the sidereal-harmonic
    modulation exp(i n Omega t) of the design notes, with sky/time in analytic scalars.

  * The delay harmonics B_n depend only on the detector location and the declination.

This module derives A_n, B_n from the SAME algebra used by
RIFT.likelihood.vectorized_lal_tools.ComputeDetAMResponse / TimeDelayFromEarthCenter, so
the reconstruction

    F(t)   = sum_n A_n exp(i n g)      matches lal.ComputeDetAMResponse
    tau(t) = sum_n B_n exp(i n g)      matches lal.TimeDelayFromEarthCenter

to machine precision for any g.  This is an exact algebraic identity in g, not an
approximation; the ONLY approximation in the slow-rotation program is the near-linearity
of GMST(t) (handled elsewhere) and the truncation of the delay-derivative expansion
(Path B), neither of which enters here.

Conventions mirror vectorized_lal_tools.py exactly:
    g = GMST - RA  (Greenwich hour angle)
    nhat = (cos_dec cos g, -cos_dec sin g, sin_dec)
    X, Y polarization triads as in ComputeDetAMResponse
    F_+ = X.D.X - Y.D.Y,   F_x = X.D.Y + Y.D.X    (D = detector response tensor, symmetric)
"""
from __future__ import print_function, division

import numpy as np

C_SI = 299792458.0  # m/s, matches vectorized_lal_tools


# ---------------------------------------------------------------------------
# Constant polarization-triad basis vectors (independent of hour angle g).
#
# Writing X = Xc cos g + Xs sin g + X0 (and likewise Y), read off directly from
# vectorized_lal_tools.ComputeDetAMResponse:
#   X[0] = -cos_psi sin_g - sin_psi cos_g sin_dec
#   X[1] = -cos_psi cos_g + sin_psi sin_g sin_dec
#   X[2] =  sin_psi cos_dec
#   Y[0] =  sin_psi sin_g - cos_psi cos_g sin_dec
#   Y[1] =  sin_psi cos_g + cos_psi sin_g sin_dec
#   Y[2] =  cos_psi cos_dec
# ---------------------------------------------------------------------------
def _triad_components(dec, psi):
    """Return the six constant 3-vectors (Xc, Xs, X0, Yc, Ys, Y0)."""
    cd, sd = np.cos(dec), np.sin(dec)
    cp, sp = np.cos(psi), np.sin(psi)

    Xc = np.array([-sp * sd, -cp,       0.0])   # coeff of cos g in X
    Xs = np.array([-cp,       sp * sd,  0.0])   # coeff of sin g in X
    X0 = np.array([ 0.0,      0.0,      sp * cd])

    Yc = np.array([-cp * sd,  sp,       0.0])   # coeff of cos g in Y
    Ys = np.array([ sp,       cp * sd,  0.0])   # coeff of sin g in Y
    Y0 = np.array([ 0.0,      0.0,      cp * cd])
    return Xc, Xs, X0, Yc, Ys, Y0


def antenna_harmonics(response_matrix, dec, psi):
    """Complex antenna-pattern harmonics A_n, n = -2..2.

    F(t) = F_+(t) + i F_x(t) = sum_{n=-2}^{2} A_n exp(i n g),  g = GMST(t) - RA.

    Parameters
    ----------
    response_matrix : (3,3) array   detector response tensor (Earth-fixed), symmetric
    dec, psi : float                declination and polarization angle [rad]

    Returns
    -------
    A : dict {n: complex}  for n in (-2,-1,0,1,2).  Independent of RA and time.
    """
    D = np.asarray(response_matrix, dtype=float)
    Xc, Xs, X0, Yc, Ys, Y0 = _triad_components(dec, psi)

    def B(u, v):
        return float(u @ D @ v)  # bilinear form; D symmetric => B(u,v)=B(v,u)

    # --- F_+ = X.D.X - Y.D.Y, expanded in cos/sin of (n g) ---
    # Using cos^2 = 1/2 + 1/2 cos2g, sin^2 = 1/2 - 1/2 cos2g, cos sin = 1/2 sin2g.
    Pp0 = (0.5 * (B(Xc, Xc) + B(Xs, Xs)) + B(X0, X0)
           - (0.5 * (B(Yc, Yc) + B(Ys, Ys)) + B(Y0, Y0)))
    Pp1 = 2.0 * (B(Xc, X0) - B(Yc, Y0))          # cos g
    Qp1 = 2.0 * (B(Xs, X0) - B(Ys, Y0))          # sin g
    Pp2 = 0.5 * ((B(Xc, Xc) - B(Xs, Xs)) - (B(Yc, Yc) - B(Ys, Ys)))  # cos 2g
    Qp2 = B(Xc, Xs) - B(Yc, Ys)                  # sin 2g

    # --- F_x = X.D.Y + Y.D.X = 2 X.D.Y (D symmetric) ---
    Pc0 = B(Xc, Yc) + B(Xs, Ys) + 2.0 * B(X0, Y0)
    Pc1 = 2.0 * (B(Xc, Y0) + B(X0, Yc))          # cos g
    Qc1 = 2.0 * (B(Xs, Y0) + B(X0, Ys))          # sin g
    Pc2 = B(Xc, Yc) - B(Xs, Ys)                  # cos 2g
    Qc2 = B(Xc, Ys) + B(Xs, Yc)                  # sin 2g

    # Complex combine: F = F_+ + i F_x.  For harmonic n>=1, the real signal piece
    #   P_n cos n g + Q_n sin n g  ->  A_n exp(i n g) + A_{-n} exp(-i n g)
    # with A_{+n} = (P_n - i Q_n)/2, A_{-n} = (P_n + i Q_n)/2, and here P_n, Q_n are
    # THEMSELVES complex (P_n = Pp_n + i Pc_n, Q_n = Qp_n + i Qc_n).
    P0 = Pp0 + 1j * Pc0
    P1 = Pp1 + 1j * Pc1
    Q1 = Qp1 + 1j * Qc1
    P2 = Pp2 + 1j * Pc2
    Q2 = Qp2 + 1j * Qc2

    A = {
        0:  P0,
        1:  0.5 * (P1 - 1j * Q1),
        -1: 0.5 * (P1 + 1j * Q1),
        2:  0.5 * (P2 - 1j * Q2),
        -2: 0.5 * (P2 + 1j * Q2),
    }
    return A


def delay_harmonics(location_xyz, dec):
    """Geometric-delay harmonics B_n, n = -1..1  [seconds].

    tau(t) = -(1/c) r_det . nhat(t) = sum_{n=-1}^{1} B_n exp(i n g),  g = GMST(t) - RA,
    with nhat = (cos_dec cos g, -cos_dec sin g, sin_dec) as in
    vectorized_lal_tools.TimeDelayFromEarthCenter.

    Parameters
    ----------
    location_xyz : (3,) array  detector position relative to Earth center [m], Earth-fixed
    dec : float                declination [rad]

    Returns
    -------
    B : dict {n: complex} for n in (-1,0,1).  B_0 real; B_{-1} = conj(B_1).
        Independent of RA and time.
    """
    r = np.asarray(location_xyz, dtype=float)
    cd, sd = np.cos(dec), np.sin(dec)

    # tau = -(1/c)[ r_x cos_dec cos g - r_y cos_dec sin g + r_z sin_dec ]
    #     = T0 + T1c cos g + T1s sin g
    T0 = -(r[2] * sd) / C_SI
    T1c = -(cd * r[0]) / C_SI
    T1s = +(cd * r[1]) / C_SI

    B = {
        0:  complex(T0),
        1:  0.5 * (T1c - 1j * T1s),
        -1: 0.5 * (T1c + 1j * T1s),
    }
    return B


# ---------------------------------------------------------------------------
# Reconstruction helpers (evaluate the harmonic series at given hour angle g).
# ---------------------------------------------------------------------------
def greenwich_hour_angle(gmst, ra):
    """g = GMST - RA."""
    return gmst - ra


def antenna_from_harmonics(A, g):
    """F_+ + i F_x = sum_n A_n exp(i n g).  g may be scalar or array."""
    g = np.asarray(g, dtype=float)
    F = np.zeros(g.shape, dtype=complex) if g.shape else 0j
    for n, An in A.items():
        F = F + An * np.exp(1j * n * g)
    return F


def delay_from_harmonics(B, g):
    """tau(t) = sum_n B_n exp(i n g) (imaginary part cancels to ~0). g scalar or array."""
    g = np.asarray(g, dtype=float)
    tau = np.zeros(g.shape, dtype=complex) if g.shape else 0j
    for n, Bn in B.items():
        tau = tau + Bn * np.exp(1j * n * g)
    return np.real(tau)


def antenna_response(A, gmst, ra):
    """Convenience: (F_plus, F_cross) from harmonics at (gmst, ra)."""
    F = antenna_from_harmonics(A, greenwich_hour_angle(gmst, ra))
    return np.real(F), np.imag(F)
