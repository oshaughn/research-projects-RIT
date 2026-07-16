"""
slowrot_freqresponse : closed-form FREQUENCY-DEPENDENT (finite-size) detector antenna
response of a ground-based Michelson interferometer, beyond the long-wavelength
approximation.  "Path D" of the slow-rotation generalization of the RIFT likelihood.

Motivation
----------
The usual antenna response F_+(RA,DEC,psi), F_x(RA,DEC,psi) returned by
lal.ComputeDetAMResponse is the LONG-WAVELENGTH LIMIT: it assumes the GW strain is
uniform across the detector while a photon traverses an arm.  Beyond that limit the
finite light-travel time T = L/c across an arm imprints genuine in-band frequency
structure on the response, controlled by the free spectral range

    f_FSR = c / (2 L)      ( 37.5 kHz for 4-km LIGO ;  3.75 kHz for a 40-km CE arm ).

The response becomes F_A -> F_A(f; RA, DEC, psi).  This matters for 3G detectors
(Cosmic Explorer, 40 km) whose long signals reach a non-negligible fraction of f_FSR
in band; it is completely negligible for the 4-km LIGO instruments.

Formula and convention  (arXiv:2412.01693, "Beyond the Long Wavelength Approximation",
Eqs. 4-6; equivalent to Rakhmanov-Romano-Whelan 2008, arXiv:0808.3805, and
Essick-Vitale-Evans 2017, PRD 96 084004 / arXiv:1708.06843)
-------------------------------------------------------------------------------------
Single-arm round-trip transfer function, for arm unit vector a-hat and source direction
n-hat (direction TO the source, Sathyaprakash-Schutz convention -- the SAME n-hat used
by lal.ComputeDetAMResponse / TimeDelayFromEarthCenter), with T = L/c the one-way
photon time of flight and  T_pm^a = T (1 +/- a-hat . n-hat):

    D~(a-hat, n-hat, f) = (e^{-i 2 pi f T} / 2)
                          * [ e^{+i pi f T_-^a} sinc(pi f T_+^a)
                            + e^{-i pi f T_+^a} sinc(pi f T_-^a) ]        (Eq. 6)

with sinc(z) = sin(z)/z, so D~ -> 1 as f -> 0.  The frequency-dependent antenna
patterns are then (Eqs. 4-5)

    F_A(f, n-hat) = (1/2) [ D~(x-hat, n-hat, f) (eps^A : x-hat x-hat)
                          - D~(y-hat, n-hat, f) (eps^A : y-hat y-hat) ] ,  A in {+, x}

where eps^+ , eps^x are the GW polarization tensors set by (RA, DEC, psi) and
x-hat, y-hat are the two arm unit vectors.  At f -> 0 both D~ -> 1 and this collapses to
    F_A(0) = eps^A : (1/2)(x-hat x-hat - y-hat y-hat) = eps^A : D_response
          = lal.ComputeDetAMResponse ,
since the detector response tensor is exactly D_response = (1/2)(x-hat x-hat - y-hat y-hat).

Implementation note (exactness of the f -> 0 limit)
---------------------------------------------------
LAL stores the detector response tensor `d.response` in single precision (REAL4).
Reconstructing (1/2)(x x - y y) from the geodetic arm geometry therefore agrees with
`d.response` only to ~1e-7.  To reproduce lal.ComputeDetAMResponse to MACHINE PRECISION
at f = 0 we write the response as an EXACT long-wavelength baseline plus a finite-size
correction that vanishes identically at f = 0:

    F_A(f) = F_A^LWL   +   (1/2)[ (D~_x - 1)(eps^A : x x)  -  (D~_y - 1)(eps^A : y y) ]

The baseline F_A^LWL is computed from the SAME `d.response` and triad algebra as
lal.ComputeDetAMResponse (see slowrot_response.py), so at f = 0 (where D~_x = D~_y = 1
EXACTLY, sinc(0)=1) the correction is exactly zero and F_A(0) == ComputeDetAMResponse.
The tiny (~1e-7) geodetic imprecision only enters the O((f/f_FSR)^2) correction, i.e.
utterly negligibly.

Pure numpy + lal; importable without the heavy RIFT stack (mirrors slowrot_response).

Conventions (identical to slowrot_response.py / vectorized_lal_tools.py):
    g    = GMST - RA                                    (Greenwich hour angle)
    nhat = (cos_dec cos g, -cos_dec sin g, sin_dec)     (direction to source)
    X = (-cp sg - sp cg sd, -cp cg + sp sg sd,  sp cd)  (polarization triad)
    Y = ( sp sg - cp cg sd,  sp cg + cp sg sd,  cp cd)
    F_+ = X.D.X - Y.D.Y ,   F_x = X.D.Y + Y.D.X
    (cp,sp)=(cos,sin)psi ; (cd,sd)=(cos,sin)dec ; (cg,sg)=(cos,sin)g
"""
from __future__ import print_function, division

import numpy as np

try:
    import lalsimulation as _lalsim
    _HAVE_LAL = True
except Exception:                                    # pragma: no cover
    _HAVE_LAL = False

C_SI = 299792458.0            # m/s, matches vectorized_lal_tools / slowrot_response
DEFAULT_L_LIGO = 3994.5       # m, nominal LIGO arm length (override for CE etc.)


# ---------------------------------------------------------------------------
# Detector geometry: arm unit vectors and arm length from a LAL detector.
# ---------------------------------------------------------------------------
def _cartesian_arm(cosAlt, sinAlt, cosAz, sinAz, cosLat, sinLat, cosLon, sinLon):
    """Earth-fixed Cartesian unit vector of a detector arm.

    Exact port of LAL's getCartesianComponents (LALDetectors.c): azimuth measured
    from local North toward East, altitude above the local horizontal.
    """
    uNorth = cosAlt * cosAz
    uEast = cosAlt * sinAz
    uRho = -sinLat * uNorth + cosLat * sinAlt
    return np.array([
        cosLon * uRho - sinLon * uEast,
        sinLon * uRho + cosLon * uEast,
        cosLat * uNorth + sinLat * sinAlt,
    ])


def detector_geometry(det, L_arm=None):
    """Return (response, x_arm, y_arm, L) for a detector prefix ('H1','L1','V1',...).

    Parameters
    ----------
    det : str                 detector prefix understood by lalsimulation
    L_arm : float or None     arm length [m] override (e.g. 40000. for a 40-km CE arm);
                              default = 2 * xArmMidpoint from the LAL detector.

    Returns
    -------
    response : (3,3) float ndarray   Earth-fixed detector response tensor D (from LAL)
    x_arm, y_arm : (3,) float ndarray  Earth-fixed arm unit vectors
    L : float                        arm length [m] used for the finite-size transfer
    """
    if not _HAVE_LAL:                                 # pragma: no cover
        raise RuntimeError("lalsimulation is required for detector_geometry")
    d = _lalsim.DetectorPrefixToLALDetector(det)
    fr = d.frDetector
    lat, lon = fr.vertexLatitudeRadians, fr.vertexLongitudeRadians
    cL, sL, cO, sO = np.cos(lat), np.sin(lat), np.cos(lon), np.sin(lon)
    x_arm = _cartesian_arm(np.cos(fr.xArmAltitudeRadians), np.sin(fr.xArmAltitudeRadians),
                           np.cos(fr.xArmAzimuthRadians), np.sin(fr.xArmAzimuthRadians),
                           cL, sL, cO, sO)
    y_arm = _cartesian_arm(np.cos(fr.yArmAltitudeRadians), np.sin(fr.yArmAltitudeRadians),
                           np.cos(fr.yArmAzimuthRadians), np.sin(fr.yArmAzimuthRadians),
                           cL, sL, cO, sO)
    L = float(L_arm) if L_arm is not None else 2.0 * float(fr.xArmMidpoint)
    return np.asarray(d.response, dtype=float), x_arm, y_arm, L


# ---------------------------------------------------------------------------
# Kinematic pieces (triad + source direction), matching ComputeDetAMResponse.
# ---------------------------------------------------------------------------
def _triad(dec, psi, g):
    """Polarization triad X, Y and source direction nhat at hour angle g = GMST - RA.

    dec, psi, g may be scalars or broadcastable arrays; returned vectors carry the
    3-component along the last axis.
    """
    cd, sd = np.cos(dec), np.sin(dec)
    cp, sp = np.cos(psi), np.sin(psi)
    cg, sg = np.cos(g), np.sin(g)

    X = np.stack([-cp * sg - sp * cg * sd,
                  -cp * cg + sp * sg * sd,
                   sp * cd * np.ones_like(sg)], axis=-1)
    Y = np.stack([ sp * sg - cp * cg * sd,
                   sp * cg + cp * sg * sd,
                   cp * cd * np.ones_like(sg)], axis=-1)
    nhat = np.stack([cd * cg, -cd * sg, sd * np.ones_like(sg)], axis=-1)
    return X, Y, nhat


# ---------------------------------------------------------------------------
# Single-arm finite-size transfer function (Eq. 6 of arXiv:2412.01693).
# ---------------------------------------------------------------------------
def single_arm_transfer(a_dot_n, f, L):
    """Round-trip single-arm transfer function D~(a-hat, n-hat, f).

    Parameters
    ----------
    a_dot_n : float or ndarray   a-hat . n-hat  (arm unit vector dotted with source dir)
    f       : float or ndarray   frequency [Hz]
    L       : float              arm length [m]

    Returns
    -------
    complex ndarray broadcast over (a_dot_n, f).  D~ -> 1 as f -> 0.

    D~ = (e^{-i 2 pi f T}/2)[ e^{+i pi f T_-} sinc(pi f T_+) + e^{-i pi f T_+} sinc(pi f T_-) ]
    with T = L/c, T_pm = T(1 +/- a_dot_n), sinc(z)=sin(z)/z.
    (np.sinc(u) = sin(pi u)/(pi u), so sinc(pi f T_pm) = np.sinc(f T_pm).)
    """
    a = np.asarray(a_dot_n, dtype=float)
    f = np.asarray(f, dtype=float)
    T = L / C_SI
    Tp = T * (1.0 + a)          # T_+
    Tm = T * (1.0 - a)          # T_-
    pref = np.exp(-1j * 2.0 * np.pi * f * T) / 2.0
    term_p = np.exp(1j * np.pi * f * Tm) * np.sinc(f * Tp)
    term_m = np.exp(-1j * np.pi * f * Tp) * np.sinc(f * Tm)
    return pref * (term_p + term_m)


# ---------------------------------------------------------------------------
# Frequency-dependent antenna response.
# ---------------------------------------------------------------------------
def _lwl_response(response, X, Y):
    """Long-wavelength F_+, F_x from triad and response tensor (== ComputeDetAMResponse).

    F_+ = X.D.X - Y.D.Y ,  F_x = X.D.Y + Y.D.X .   Vectorized over leading axes of X,Y.
    """
    D = response
    XDX = np.einsum('...i,ij,...j->...', X, D, X)
    YDY = np.einsum('...i,ij,...j->...', Y, D, Y)
    XDY = np.einsum('...i,ij,...j->...', X, D, Y)
    YDX = np.einsum('...i,ij,...j->...', Y, D, X)
    return XDX - YDY, XDY + YDX


def antenna_response_fd(det, ra, dec, psi, f, gmst=0.0, L_arm=None):
    """Frequency-dependent antenna response F_+(f), F_x(f) for a ground-based detector.

    Beyond the long-wavelength limit: includes the finite light-travel-time transfer
    across the arms.  Reduces EXACTLY to lal.ComputeDetAMResponse as f -> 0.

    Parameters
    ----------
    det : str                  detector prefix ('H1','L1','V1','K1',...)
    ra, dec, psi : float       right ascension, declination, polarization angle [rad]
    f : float or (Nf,) ndarray frequency [Hz] (scalar or 1-D array)
    gmst : float               Greenwich mean sidereal time [rad] (default 0).  Enters
                              only through g = gmst - ra, exactly as ComputeDetAMResponse.
    L_arm : float or None      arm-length override [m] (e.g. 40000. for 40-km CE);
                              default = LAL detector arm length (~4 km for LIGO).

    Returns
    -------
    (Fp, Fc) : complex ndarray, same shape as f.  At f=0, imag part -> 0 and the real
               parts equal lal.ComputeDetAMResponse(response, ra, dec, psi, gmst).
    """
    response, x_arm, y_arm, L = detector_geometry(det, L_arm=L_arm)
    return antenna_response_fd_geom(response, x_arm, y_arm, L, ra, dec, psi, f, gmst=gmst)


def antenna_response_fd_geom(response, x_arm, y_arm, L, ra, dec, psi, f, gmst=0.0):
    """Same as antenna_response_fd but with detector geometry supplied explicitly.

    Useful for exotic geometries (custom L, custom arms).  See antenna_response_fd.
    """
    g = gmst - ra
    X, Y, nhat = _triad(dec, psi, g)                 # shape (3,)

    # Exact long-wavelength baseline (matches ComputeDetAMResponse to machine precision).
    Fp_lwl, Fc_lwl = _lwl_response(response, X, Y)

    # Arm-basis polarization projections eps^A : (a a).
    Xx, Yx = float(X @ x_arm), float(Y @ x_arm)
    Xy, Yy = float(X @ y_arm), float(Y @ y_arm)
    Ppx = Xx * Xx - Yx * Yx      # eps^+ : x x
    Ppy = Xy * Xy - Yy * Yy      # eps^+ : y y
    Pcx = 2.0 * Xx * Yx          # eps^x : x x
    Pcy = 2.0 * Xy * Yy          # eps^x : y y

    # Finite-size transfer per arm (vanishes-minus-one -> 0 at f=0).
    ax = float(x_arm @ nhat)
    ay = float(y_arm @ nhat)
    Dx = single_arm_transfer(ax, f, L)
    Dy = single_arm_transfer(ay, f, L)

    Fp = Fp_lwl + 0.5 * ((Dx - 1.0) * Ppx - (Dy - 1.0) * Ppy)
    Fc = Fc_lwl + 0.5 * ((Dx - 1.0) * Pcx - (Dy - 1.0) * Pcy)
    return Fp, Fc


def free_spectral_range(L):
    """Free spectral range f_FSR = c / (2 L) [Hz]."""
    return C_SI / (2.0 * L)


# ===========================================================================
# Step 1 : sky-harmonic / power-series expansion of the finite-size response.
# ===========================================================================
# The finite-size antenna response is (pure Eqs. 4-6, no LWL split)
#
#     F_A(f;sky) = (1/2)[ D~(x,n,f) (eps^A:xx) - D~(y,n,f) (eps^A:yy) ] .
#
# Factor out the common one-way light-crossing delay T = L/c that both arms
# share (DIRECTION-INDEPENDENT, degenerate with coalescence time):
#
#     D~(a,n,f) = e^{-i 2 pi f T} g(f; a) ,   a = a-hat . n-hat ,
#     g(f; a)   = (1/2)[ e^{+i u(1-a)} sinc(u(1+a)) + e^{-i u(1+a)} sinc(u(1-a)) ] ,
#     u = pi f T ,   sinc(x) = sin(x)/x ,   g(f;0) = sinc(2u) , g -> 1 as f->0.
#
# Taylor-expand the residual g in the arm projection a  (the small in-band
# parameter is eps*a with eps = f L/c = f T):
#
#     g(f; a) = sum_{q>=0} c_q(f) a^q .                        (c_q sky-INDEPENDENT)
#
# The arm projections a_x = x-hat.n, a_y = y-hat.n and the arm-basis polarization
# contractions combine into a single COMPLEX analytic sky/pol scalar per power q,
#
#     beta_q(sky) = (1/2)[ zx^2 a_x^q - zy^2 a_y^q ] ,
#     zx = X.x-hat + i Y.x-hat ,  zy = X.y-hat + i Y.y-hat     (X,Y polarization triad)
#
# so that  G_A(f;sky) := F_A e^{+i2 pi f T}  gives the complex response
#
#     G(f) = G_+ + i G_x = sum_q c_q(f) beta_q(sky) ,          (the analogue of A_n)
#     beta_0 = (1/2)(zx^2 - zy^2) = F_+^LWL + i F_x^LWL   (== ComputeDetAMResponse, up
#                                                          to the ~1e-7 REAL4 geodety).
#
# The full response, EXACTLY reproducing antenna_response_fd, is then
#
#     F(f)    = F0_lal  +  e^{-i2 pi f T} G(f)  -  beta_0            (Fp+iFc form)
#     Fbar(f) = conj(F0_lal) + e^{-i2 pi f T} Gbar(f) - conj(beta_0),  Gbar=sum_q c_q conj(beta_q)
#     F_+ = (F+Fbar)/2 ,   F_x = (F-Fbar)/(2i)
#
# with F0_lal the EXACT lal.ComputeDetAMResponse baseline (so f->0 is machine
# precise even though beta_0 from the arm triads carries the ~1e-7 geodetic error;
# the arm-based pieces enter only the finite-size correction, which vanishes at f=0).
# ---------------------------------------------------------------------------
from math import comb as _comb, factorial as _factorial


def _sinc_shift_apoly(u, s, Qmax, jmax=64):
    """a-power coefficients d_m(u), m=0..Qmax, of  sinc(u(1 + s a))  (s = +/-1).

    Entire-series form (stable at u=0):  sinc(x) = sum_j (-1)^j x^{2j}/(2j+1)!,
    x = u(1+s a) => coeff of a^m is  s^m sum_{j>=ceil(m/2)} (-1)^j u^{2j}/(2j+1)! C(2j,m).
    u may be a numpy array (frequencies); returns shape (Qmax+1,) + u.shape.
    """
    u = np.asarray(u, dtype=float)
    out = np.zeros((Qmax + 1,) + u.shape, dtype=float)
    u2 = u * u
    # precompute u^{2j}
    upow = np.ones_like(u)          # u^{0}
    for j in range(0, jmax + 1):
        coef = ((-1) ** j) / float(_factorial(2 * j + 1))
        for m in range(0, min(Qmax, 2 * j) + 1):
            out[m] += coef * _comb(2 * j, m) * upow
        upow = upow * u2
    for m in range(Qmax + 1):
        out[m] *= float(s) ** m
    return out.astype(complex)


def _poly_mul(A, B, Qmax):
    """Truncated product (to order Qmax) of two a-power stacks A[k],B[k] over freq axis."""
    out = np.zeros_like(A[:Qmax + 1])
    for i in range(min(len(A), Qmax + 1)):
        for j in range(min(len(B), Qmax + 1 - i)):
            out[i + j] += A[i] * B[j]
    return out


def finite_size_c_coeffs(f, L, Qmax):
    """Sky-independent frequency basis c_q(f), q = 0..Qmax, of g(f;a)=sum_q c_q(f) a^q.

    Parameters
    ----------
    f : ndarray        frequency [Hz] (signed values allowed; c_q(-f)=conj(c_q(f)))
    L : float          arm length [m]
    Qmax : int         highest power of the arm projection a retained

    Returns
    -------
    c : complex ndarray, shape (Qmax+1,) + f.shape.  c_0(f)=sinc(2 pi f T),
        c_{q>=1} carry the finite-size shape distortion; all -> delta_{q0} as f->0.
    """
    f = np.asarray(f, dtype=float)
    T = L / C_SI
    u = np.pi * f * T
    # exp(-i u a): coeff of a^k is (-i u)^k / k!
    exp_neg = np.zeros((Qmax + 1,) + u.shape, dtype=complex)
    term = np.ones_like(u, dtype=complex)
    for k in range(Qmax + 1):
        exp_neg[k] = term / float(_factorial(k))
        term = term * (-1j * u)
    sinc_p = _sinc_shift_apoly(u, +1.0, Qmax)      # sinc(u(1+a))
    sinc_m = _sinc_shift_apoly(u, -1.0, Qmax)      # sinc(u(1-a))
    eiu = np.exp(1j * u)
    term1 = eiu * _poly_mul(exp_neg, sinc_p, Qmax)         # e^{iu} e^{-iua} sinc(u(1+a))
    term2 = np.conj(eiu) * _poly_mul(exp_neg, sinc_m, Qmax)  # e^{-iu} e^{-iua} sinc(u(1-a))
    return 0.5 * (term1 + term2)


def finite_size_geometry(det, ra, dec, psi, gmst=0.0, L_arm=None):
    """Geometric scalars for the finite-size expansion at (det, ra, dec, psi, gmst).

    Returns dict with:
        T      = L/c common one-way delay [s]
        L      arm length [m]
        ax, ay arm projections x-hat.n, y-hat.n
        zx, zy = X.a-hat + i Y.a-hat (complex pol/arm scalars)
        F0     = Fp_lwl + i Fc_lwl  (EXACT lal baseline response, machine precise f=0)
    """
    response, x_arm, y_arm, L = detector_geometry(det, L_arm=L_arm)
    g = gmst - ra
    X, Y, nhat = _triad(dec, psi, g)
    Fp_lwl, Fc_lwl = _lwl_response(response, X, Y)
    Xx, Yx = float(X @ x_arm), float(Y @ x_arm)
    Xy, Yy = float(X @ y_arm), float(Y @ y_arm)
    zx = Xx + 1j * Yx
    zy = Xy + 1j * Yy
    ax = float(x_arm @ nhat)
    ay = float(y_arm @ nhat)
    return dict(T=L / C_SI, L=L, ax=ax, ay=ay, zx=zx, zy=zy,
                F0=complex(Fp_lwl) + 1j * complex(Fc_lwl))


def finite_size_beta(geom, Qmax):
    """Analytic sky/pol coefficients beta_q = (1/2)[zx^2 a_x^q - zy^2 a_y^q], q=0..Qmax."""
    zx2, zy2, ax, ay = geom['zx'] ** 2, geom['zy'] ** 2, geom['ax'], geom['ay']
    return np.array([0.5 * (zx2 * ax ** q - zy2 * ay ** q) for q in range(Qmax + 1)],
                    dtype=complex)


def F_fd_expanded(det, ra, dec, psi, f, Qmax, gmst=0.0, L_arm=None):
    """Order-Qmax reconstruction of the finite-size response F_+(f), F_x(f).

    Converges to antenna_response_fd(det,ra,dec,psi,f,...) as Qmax -> infinity.
    Uses the EXACT lal baseline F0 for the constant part and the power-series
    finite-size correction e^{-i2 pi f T}(sum_q c_q beta_q) - beta_0.
    """
    f = np.asarray(f, dtype=float)
    geom = finite_size_geometry(det, ra, dec, psi, gmst=gmst, L_arm=L_arm)
    beta = finite_size_beta(geom, Qmax)
    c = finite_size_c_coeffs(f, geom['L'], Qmax)             # (Qmax+1,)+f.shape
    G = np.tensordot(beta, c, axes=(0, 0))                   # sum_q beta_q c_q(f)
    Gbar = np.tensordot(np.conj(beta), c, axes=(0, 0))
    phase = np.exp(-1j * 2.0 * np.pi * f * geom['T'])
    F = geom['F0'] + phase * G - beta[0]
    Fbar = np.conj(geom['F0']) + phase * Gbar - np.conj(beta[0])
    Fp = 0.5 * (F + Fbar)
    Fc = (F - Fbar) / (2.0j)
    return Fp, Fc


def finite_size_response_weights(fvals, geom, Qmax):
    """Per-basis frequency weights W_p(f) folded into the FD modes for the likelihood.

    Response basis (p = 0..Qmax+1) reproducing F(f) = sum_p b_p W_p(f):
        p=0   ("baseline") : W_0(f) = 1                       b_0 = F0 (exact lal)
        p=1+q              : W_{1+q}(f) = e^{-i2pi f T} c_q(f) - [q==0]
                                                              b_{1+q} = beta_q  (arm)
    Each W_p is Hermitian (W_p(-f)=conj(W_p(f))) so the V cross term needs NO
    harmonic reflection.  The common delay e^{-i2 pi f T} (= a T=L/c arrival-time
    shift of the finite-size correction relative to the LWL baseline) is carried
    inside the correction weights.  Returns (weights (Npbasis, Nf) complex, coeff-builder).
    """
    fvals = np.asarray(fvals, dtype=float)
    c = finite_size_c_coeffs(fvals, geom['L'], Qmax)
    phase = np.exp(-1j * 2.0 * np.pi * fvals * geom['T'])
    W = np.empty((Qmax + 2, fvals.shape[0]), dtype=complex)
    W[0] = 1.0
    for q in range(Qmax + 1):
        W[1 + q] = phase * c[q] - (1.0 if q == 0 else 0.0)
    return W
