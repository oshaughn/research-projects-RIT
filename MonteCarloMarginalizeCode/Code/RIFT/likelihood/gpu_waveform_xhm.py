"""Batched IMRPhenomXHM aligned-spin higher modes for CuPy or NumPy.

Stage 2 of the dependency-free GPU port of IMRPhenomXPHM (stage 1:
``gpu_waveform_xphm``).  Adds the IMRPhenomXHM modes (2,1), (3,3), (3,2) with
its spheroidal-to-spherical mode mixing, and (4,4), alongside the IMRPhenomXAS
(2,2) mode of stage 1.  The API follows stage 1: every function takes an array
module ``xp`` (``cupy`` or ``numpy``) and many aligned-spin points at once;
parameters are arrays of shape ``(n,)`` and the output has shape
``(n, nmodes, N)``.  The per-point setup (fits, collocation solves, finite
differences) runs on the host in NumPy; only the frequency-grid evaluation runs
on ``xp``.

Implemented with LALSuite 6.2.0 defaults: PhenomXHMReleaseVersion 122022
(InsAmp 123, IntAmp 211112 or 110102 for the (2,1), RDAmp 2 or 1 for the (3,2);
equispaced intermediate amplitude points), phase versions 122019, and no
multibanding (``ThresholdMband = 0``).  No tides, no PNR, no non-GR terms.

The host setup uses the platform libm (through ``math``) for every
transcendental, not NumPy's SIMD ufuncs, because LAL takes 4th-order finite
differences with steps of 1e-9 (amplitude) and 1e-7 (phase of the (3,2)); a
one-ulp change in an ansatz value there moves the waveform by about 1e-7.

Provenance and licence
----------------------
This file is a line-by-line re-implementation of the LALSimulation sources
``LALSimIMRPhenomXHM.c``, ``LALSimIMRPhenomXHM_internals.c``,
``LALSimIMRPhenomXHM_inspiral.c``, ``LALSimIMRPhenomXHM_intermediate.c``,
``LALSimIMRPhenomXHM_ringdown.c`` and ``LALSimIMRPhenomXHM_qnm.c`` at tag
``lalsimulation-v6.2.0`` (Copyright (C) 2019-2022 Cecilio Garcia-Quiros, Marta
Colleoni, Sascha Husa, Hector Estelles, Geraint Pratten and the LALSuite
authors).  The fit coefficients live in the generated ``gpu_waveform_xhm_fits``.
LALSuite is distributed under the GNU General Public License, version 2 or
later; this derived file is therefore distributed under the same terms
(GPL-2.0-or-later), not under the MIT licence of the rest of RIFT.

Reference: Garcia-Quiros et al., PRD 102, 064002 (2020), arXiv:2001.10914.
"""

import math
from types import SimpleNamespace

import numpy as np

from RIFT.likelihood import gpu_waveform_xphm as X
from RIFT.likelihood import gpu_waveform_xhm_fits as F

__all__ = ["XHM_MODES", "imrphenomxhm_modes", "imrphenomxhm_setup"]

XHM_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
LAL_PI = X.LAL_PI
LAL_PI_4 = X.LAL_PI_4
FALSE_ZERO = 1.0e-15
ETA_EMR = 0.05
_MODE_INT = {21: 0, 33: 1, 32: 2, 44: 3}

# ---------------------------------------------------------------------------
# Host arithmetic: NumPy arrays, libm transcendentals
# ---------------------------------------------------------------------------


def _libm(fun, nin=1):
    uf = np.frompyfunc(fun, nin, 1)

    def wrapped(*args):
        out = uf(*args)
        if isinstance(out, np.ndarray):
            return out.astype(np.float64)
        return float(out)
    return wrapped


def _libm_c(name, nargs):
    """A double-precision function straight from the C libm.

    Used for hypot (CPython's math.hypot is its own algorithm) and cbrt
    (math.cbrt needs Python >= 3.11)."""
    import ctypes
    import ctypes.util
    fun = getattr(ctypes.CDLL(ctypes.util.find_library("m")), name)
    fun.restype = ctypes.c_double
    fun.argtypes = (ctypes.c_double,) * nargs
    return fun


class _LibmNumPy:
    """NumPy namespace whose transcendentals call the platform libm."""
    power = staticmethod(_libm(math.pow, 2))
    exp = staticmethod(_libm(math.exp))
    log = staticmethod(_libm(math.log))
    cbrt = staticmethod(_libm(_libm_c("cbrt", 1)))
    arctan = staticmethod(_libm(math.atan))
    atan = arctan
    sqrt = staticmethod(np.sqrt)
    arctan2 = staticmethod(_libm(math.atan2, 2))
    sin = staticmethod(_libm(math.sin))
    cos = staticmethod(_libm(math.cos))
    tanh = staticmethod(_libm(math.tanh))
    hypot = staticmethod(_libm(_libm_c("hypot", 2), 2))

    class linalg:
        @staticmethod
        def solve(A, b):
            """GSL-order LU solve for the stage-1 collocation systems."""
            return _lu_solve(A, b[..., 0])[..., None]

    def __getattr__(self, name):
        return getattr(np, name)


H = _LibmNumPy()
F.np = H        # the generated fits call np.<fun>; route them through libm
F._pow = H.power


def _powers(x):
    """IMRPhenomX_Initialize_Powers on host arrays (all fields)."""
    p = SimpleNamespace()
    sixth = H.power(x, 1.0 / 6.0)
    p.one_sixth = sixth
    p.m_one_sixth = 1.0 / sixth
    p.m_one = 1.0 / x
    p.itself = x
    p.one_third = sixth * sixth
    p.m_one_third = 1.0 / p.one_third
    p.two_thirds = p.one_third * p.one_third
    p.m_two_thirds = 1.0 / p.two_thirds
    p.four_thirds = p.two_thirds * p.two_thirds
    p.m_four_thirds = 1.0 / p.four_thirds
    p.five_thirds = p.four_thirds * p.one_third
    p.m_five_thirds = 1.0 / p.five_thirds
    p.seven_thirds = p.four_thirds * x
    p.m_seven_thirds = 1.0 / p.seven_thirds
    p.eight_thirds = p.seven_thirds * p.one_third
    p.m_eight_thirds = 1.0 / p.eight_thirds
    p.two = x * x
    p.three = p.two * x
    p.four = p.three * x
    p.five = p.four * x
    p.m_two = 1.0 / p.two
    p.m_three = 1.0 / p.three
    p.m_four = 1.0 / p.four
    p.m_five = 1.0 / p.five
    p.seven_sixths = p.one_sixth * p.itself
    p.m_seven_sixths = p.m_one_sixth * p.m_one
    p.log = H.log(x)
    p.sqrt = p.one_sixth * p.one_sixth * p.one_sixth
    return p


PI = _powers(np.float64(LAL_PI))  # powers_of_lalpiHM


def _cexp_i(phi):
    """cexp(I*phi) for real phi, as glibc evaluates it."""
    return H.cos(phi) + 1j * H.sin(phi)


def _cabs(z):
    return H.hypot(np.real(z), np.imag(z))


def _carg(z):
    return H.arctan2(np.imag(z), np.real(z))


def _lu_solve(A, b):
    """Batched gsl_linalg_LU_decomp + LU_solve (unblocked, partial pivoting).

    ``A`` is (n, N, N) and ``b`` is (n, N); follows the GSL 2.x unblocked
    algorithm (first maximal pivot, scaling by the reciprocal of the pivot,
    rank-1 updates, forward then backward substitution)."""
    A = np.array(A, dtype=np.float64, copy=True)
    x = np.array(b, dtype=np.float64, copy=True)
    n, N, _ = A.shape
    rows = np.arange(n)
    for j in range(N):
        piv = j + np.argmax(np.abs(A[:, j:, j]), axis=1)
        swap = piv != j
        if np.any(swap):
            r = rows[swap]
            pj = piv[swap]
            tmp = A[r, j, :].copy()
            A[r, j, :] = A[r, pj, :]
            A[r, pj, :] = tmp
            tb = x[r, j].copy()
            x[r, j] = x[r, pj]
            x[r, pj] = tb
        if j < N - 1:
            inv = 1.0 / A[:, j, j]
            A[:, j + 1:, j] = A[:, j + 1:, j] * inv[:, None]
            A[:, j + 1:, j + 1:] = A[:, j + 1:, j + 1:] + (-A[:, j + 1:, j][:, :, None]) * A[:, j, j + 1:][:, None, :]
    for i in range(N):
        tmp = x[:, i]
        for k in range(i):
            tmp = tmp - A[:, i, k] * x[:, k]
        x[:, i] = tmp
    for i in range(N - 1, -1, -1):
        tmp = x[:, i]
        for k in range(i + 1, N):
            tmp = tmp - A[:, i, k] * x[:, k]
        x[:, i] = tmp / A[:, i, i]
    return x


# ---------------------------------------------------------------------------
# (2,2) quantities used by the higher modes (stage-1 code on the libm host)
# ---------------------------------------------------------------------------

def _setup_22(m1, m2, chi1z, chi2z, distance, phi_ref, f_min, f_ref, f_max):
    # LAL receives SI masses and divides by MSUN_SI again.
    m1 = (m1 * X.LAL_MSUN_SI) / X.LAL_MSUN_SI
    m2 = (m2 * X.LAL_MSUN_SI) / X.LAL_MSUN_SI
    w = X._waveform_variables(m1, m2, chi1z, chi2z, f_ref, phi_ref, f_min, f_max, distance, H)
    big = np.where(m1 < m2, m2, m1)
    small = np.where(m1 < m2, m1, m2)
    w.q = np.where(big > small, big / small, small / big)
    w.q = np.where(w.eta == 0.25, 1.0, w.q)
    w.dchi_half = w.dchi * 0.5
    a = X._amp_coefficients(w, H)
    c = X._phase_coefficients(w, H)
    c = X._connection_and_shift(w, c, H)
    # IMRPhenomX_TimeShift_22 is c.linb; phiref22 as written in XHM.
    pref = _powers(w.MfRef)
    w.timeshift = c.linb
    w.phiref22 = (-1.0 / w.eta * X._phase_22(w.MfRef, w, c, H) - w.timeshift * w.MfRef - 0.0
                  + 2.0 * w.phi0 + LAL_PI_4)
    del pref
    return w, a, c


def _phase22(ff, w, c):
    return X._phase_22(ff, w, c, H)


def _dphase22(ff, w, c):
    return X._dphase_22(ff, w, c, H)


# ---------------------------------------------------------------------------
# Per-mode setup (host)
# ---------------------------------------------------------------------------

def _hm_variables(ell, emm, w):
    """IMRPhenomXHM_SetHMWaveformVariables for release 122022."""
    h = SimpleNamespace(ell=ell, emm=emm, tag=10 * ell + emm)
    h.modeInt = _MODE_INT[h.tag]
    h.mixing = h.tag == 32
    h.ampNorm = w.ampNorm
    h.fMECOlm = w.fMECO * emm * 0.5
    zero = (w.q == 1.0) & (w.chi1L == w.chi2L)
    h.Ampzero = zero if h.tag in (21, 33) else np.zeros_like(zero)
    h.inter_version = 110102 if h.tag == 21 else 211112
    h.rd_amp_version = 1 if h.mixing else 2
    h.digits = [int(d) for d in str(h.inter_version)]
    h.nCoefficientsInter = sum(h.digits)
    h.nCollocPtsInterPhase = 6 if h.mixing else 5
    h.chi_s = (w.chi1L + w.chi2L) * 0.5
    h.chi_a = (w.chi1L - w.chi2L) * 0.5
    fring = getattr(F, "evaluate_QNMfit_fring%d" % h.tag)
    fdamp = getattr(F, "evaluate_QNMfit_fdamp%d" % h.tag)
    h.fRING = fring(w.afinal) / w.Mfinal
    h.fDAMP = fdamp(w.afinal) / w.Mfinal
    if h.mixing:
        af = w.afinal
        mix = [F.evaluate_QNMfit_re_l2m2lp2(af) + 1j * F.evaluate_QNMfit_im_l2m2lp2(af),
               F.evaluate_QNMfit_re_l2m2lp3(af) + 1j * F.evaluate_QNMfit_im_l2m2lp3(af),
               F.evaluate_QNMfit_re_l3m2lp2(af) + 1j * F.evaluate_QNMfit_im_l3m2lp2(af),
               F.evaluate_QNMfit_re_l3m2lp3(af) + 1j * F.evaluate_QNMfit_im_l3m2lp3(af)]
        # LAL stores the mixing coefficients as single-precision complex numbers.
        mix = [np.real(z).astype(np.float32).astype(np.float64)
               + 1j * np.imag(z).astype(np.float32).astype(np.float64) for z in mix]
        mix[2] = -1.0 * mix[2]
        mix[3] = -1.0 * mix[3]
        h.mix2c = np.conj(mix[2])
        h.mix3c = np.conj(mix[3])
    h.timeshift = 0.0
    h.phaseshift = 0.0
    psi4 = X._psi4_to_strain(w.eta, w.STotR, w.dchi, H)
    h.DeltaT = -2. * LAL_PI * (500 + psi4)
    h.fPhaseRDflat = 0.0
    h.fAmpRDfalloff = 0.0
    return h


# ---- amplitude ------------------------------------------------------------

def _pn_amp_coefficients(h, w, A):
    chiA, chiS = h.chi_a, h.chi_s
    eta, delta = w.eta, w.delta
    Pi = PI.itself
    log = H.log
    prefactors = (math.sqrt(2) / 3., 0.75 * math.sqrt(5 / 7.), math.sqrt(5 / 7.) / 3.,
                  4 * math.sqrt(2) / 9 * math.sqrt(5 / 7.))
    A.PNglobalfactor = math.pow(2. / h.emm, -7 / 6.) * prefactors[h.modeInt]
    z = np.zeros_like(eta)
    I = 1j
    if h.tag == 21:
        p2 = _powers(np.float64(2.))
        A.pn = [z, delta * PI.one_third * p2.one_third,
                _cdiv((-3 * (chiA + chiS * delta)), 2.) * PI.two_thirds * p2.two_thirds,
                _cdiv((335 * delta + 1404 * delta * eta), 672.) * PI.itself * p2.itself,
                _cdiv((3427 * chiA - (I * 672) * delta + 3427 * chiS * delta - 8404 * chiA * eta - 3860 * chiS * delta * eta - 1344 * delta * Pi - (I * 672) * delta * log(16)), 1344.) * PI.four_thirds * p2.four_thirds,
                _cdiv((-155965824 * chiA * chiS - 964357 * delta + 432843264 * chiA * chiS * eta - 23670792 * delta * eta + 24385536 * chiA * Pi + 24385536 * chiS * delta * Pi - 77982912 * delta * chiA * chiA + 81285120 * delta * eta * chiA * chiA - 77982912 * delta * chiS * chiS + 39626496 * delta * eta * chiS * chiS + 21535920 * delta * eta * eta), 8.128512e6) * PI.five_thirds * p2.five_thirds,
                _cdiv((143063173 * chiA - (I * 1350720) * delta + 143063173 * chiS * delta - 546199608 * chiA * eta - (I * 72043776) * delta * eta - 169191096 * chiS * delta * eta - 9898560 * delta * Pi + 20176128 * delta * eta * Pi - (I * 5402880) * delta * log(2) - (I * 17224704) * delta * eta * log(2) + 61725888 * chiS * delta * chiA * chiA - 81285120 * chiS * delta * eta * chiA * chiA + 20575296 * H.power(chiA, 3) - 81285120 * eta * H.power(chiA, 3) + 61725888 * chiA * chiS * chiS - 165618432 * chiA * eta * chiS * chiS + 20575296 * delta * H.power(chiS, 3) - 1016064 * delta * eta * chiS * chiS * chiS + 128873808 * chiA * eta * eta - 3859632 * chiS * delta * eta * eta), 5.419008e6) * PI.two * p2.two]
    elif h.tag == 33:
        p = _powers(np.float64(2. / 3.))
        A.pn = [z, delta * PI.one_third * p.one_third, z,
                _cdiv((-1945 * delta + 2268 * delta * eta), 672.) * PI.itself * p.itself,
                _cdiv((325 * chiA - (I * 504) * delta + 325 * chiS * delta - 1120 * chiA * eta - 80 * chiS * delta * eta + 120 * delta * Pi + (I * 720) * delta * log(1.5)), 120.) * PI.four_thirds * p.four_thirds,
                _cdiv((-2263282560 * chiA * chiS - 1077664867 * delta + 9053130240 * chiA * chiS * eta - 5926068792 * delta * eta - 1131641280 * delta * chiA * chiA + 4470681600 * delta * eta * chiA * chiA - 1131641280 * delta * chiS * chiS + 55883520 * delta * eta * chiS * chiS + 2966264784 * delta * eta * eta), 4.4706816e8) * PI.five_thirds * p.five_thirds,
                _cdiv((22007835 * chiA + (I * 26467560) * delta + 22007835 * chiS * delta - 80190540 * chiA * eta - (I * 98774368) * delta * eta - 31722300 * chiS * delta * eta - 9193500 * delta * Pi + 17826480 * delta * eta * Pi - (I * 37810800) * delta * log(1.5) + (I * 37558080) * delta * eta * log(1.5) - 12428640 * chiA * eta * eta - 6078240 * chiS * delta * eta * eta), 2.17728e6) * PI.two * p.two]
    elif h.tag == 32:
        A.pn = [z, z, (-1 + 3 * eta) * PI.two_thirds,
                -4 * chiS * eta * PI.itself,
                _cdiv((10471 - 61625 * eta + 82460 * eta * eta), 10080.) * PI.four_thirds,
                _cdiv(((I * 2520) - 3955 * chiS - 3955 * chiA * delta - (I * 11088) * eta + 10810 * chiS * eta + 11865 * chiA * delta * eta - 12600 * chiS * eta * eta), 840.) * PI.five_thirds,
                _cdiv((824173699 + 2263282560 * chiA * chiS * delta - 26069649 * eta - 15209631360 * chiA * chiS * delta * eta + 3576545280 * chiS * eta * Pi + 1131641280 * chiA * chiA - 7865605440 * eta * chiA * chiA + 1131641280 * chiS * chiS - 11870591040 * eta * chiS * chiS - 13202119896 * eta * eta + 13412044800 * chiA * chiA * eta * eta + 5830513920 * chiS * chiS * eta * eta + 5907445488 * H.power(eta, 3)), 4.4706816e8) * PI.two]
    else:
        p = _powers(np.float64(0.5))
        A.pn = [z, z, (1 - 3 * eta) * PI.two_thirds * p.two_thirds, z,
                _cdiv((-158383 + 641105 * eta - 446460 * eta * eta), 36960.) * PI.four_thirds * p.four_thirds,
                _cdiv(((I * -1008) + 565 * chiS + 565 * chiA * delta + (I * 3579) * eta - 2075 * chiS * eta - 1695 * chiA * delta * eta + 240 * Pi - 720 * eta * Pi + (I * 960) * log(2) - (I * 2880) * eta * log(2) + 1140 * chiS * eta * eta), 120.) * PI.five_thirds * p.five_thirds,
                _cdiv((7888301437 - 147113366400 * chiA * chiS * delta - 745140957231 * eta + 441340099200 * chiA * chiS * delta * eta - 73556683200 * chiA * chiA + 511264353600 * eta * chiA * chiA - 73556683200 * chiS * chiS + 224302478400 * eta * chiS * chiS + 2271682065240 * eta * eta - 871782912000 * chiA * chiA * eta * eta - 10897286400 * chiS * chiS * eta * eta - 805075876080 * H.power(eta, 3)), 2.90594304e10) * PI.two * p.two]
    A.pn = [np.asarray(v, dtype=np.complex128) * np.ones_like(eta) for v in A.pn]


def _cdiv(z, d):
    """C99 ``double complex / double``: divide the two parts (NumPy's complex
    division multiplies by a reciprocal)."""
    z = np.asarray(z)
    if np.iscomplexobj(z):
        return z.real / d + 1j * (z.imag / d)
    return z / d


def _cmul_real(r, z):
    """C99 ``double * double complex``: scale the two parts."""
    return r * z.real + 1j * (r * z.imag)


def _pn_amp_ansatz(p, h, A, xp=None):
    """IMRPhenomXHM_Inspiral_PNAmp_Ansatz with InspRescaleFactor 0.

    Host (``xp`` None) evaluation keeps C's real*complex products."""
    pn = A.pn
    if xp is None:
        s = pn[0]
        for pw, cf in ((p.one_third, pn[1]), (p.two_thirds, pn[2]), (p.itself, pn[3]),
                       (p.four_thirds, pn[4]), (p.five_thirds, pn[5]), (p.two, pn[6])):
            s = s + _cmul_real(pw, cf)
        amp = _cabs(s)
    else:
        re = pn[0].real + p.one_third * pn[1].real + p.two_thirds * pn[2].real + p.itself * pn[3].real \
            + p.four_thirds * pn[4].real + p.five_thirds * pn[5].real + p.two * pn[6].real
        im = pn[0].imag + p.one_third * pn[1].imag + p.two_thirds * pn[2].imag + p.itself * pn[3].imag \
            + p.four_thirds * pn[4].imag + p.five_thirds * pn[5].imag + p.two * pn[6].imag
        amp = xp.hypot(re, im)
    return amp * (A.PNglobalfactor * p.m_seven_sixths * h.ampNorm)


def _insp_amp_ansatz(p, h, A, xp=None):
    amp = _pn_amp_ansatz(p, h, A, xp)
    pseudo = p.seven_thirds / A.fcutInsp_seven_thirds * A.ic0 \
        + p.eight_thirds / A.fcutInsp_eight_thirds * A.ic1 \
        + p.three / A.fcutInsp_three * A.ic2
    pseudo = pseudo * (p.m_seven_sixths * A.PNdominant)
    return amp + pseudo


def _insp_amp_nd(f0, h, A):
    df = 10e-10
    vals = [_insp_amp_ansatz(_powers(f0 + k * df), h, A) for k in (2, 1, -1, -2)]
    return (-vals[0] + 8 * vals[1] - 8 * vals[2] + vals[3]) / (12 * df)


def _rd_amp_ansatz(p, h, A, xp=None, falloff=True):
    """IMRPhenomXHM_RD_Amp_Ansatz, versions 1 and 2 (RDRescaleFactor 0)."""
    xpm = H if xp is None else xp
    ff = p.itself
    frd, fda = h.fRING, h.fDAMP
    dfr = ff - frd
    dfd = fda * A.rd2
    lor = A.rd0 * fda / (xpm.exp(A.rd1 / dfd * dfr) * (dfr * dfr + dfd * dfd))
    out = lor
    if falloff:
        tail = A.rd3 * xpm.exp(-A.rd4 * (ff - h.fAmpRDfalloff))
        out = xpm.where(ff >= h.fAmpRDfalloff, tail, out)
    if h.mixing and A.has_aux:
        poly = 0.0
        fpower = 1.0
        for i in range(4):
            poly = poly + fpower * A.aux[i]
            fpower = fpower * ff
        out = xpm.where(ff >= A.fRDAux, out, poly)
    return out


def _rd_amp_dansatz(p, h, A):
    ff = p.itself
    frd, fda = h.fRING, h.fDAMP
    dfr = ff - frd
    numerator = A.rd0 * (dfr * dfr * A.rd1 + 2 * fda * dfr * A.rd2 + fda * fda * A.rd1 * A.rd2 * A.rd2)
    denom = (dfr * dfr + fda * fda * A.rd2 * A.rd2)
    denom = A.rd2 * denom * denom * H.exp(dfr * A.rd1 / (fda * A.rd2))
    return -numerator / denom


def _fit(name, w, version):
    return F.FITS[name](w, version)


def _amp_setup(h, w, P):
    """IMRPhenomXHM_GetAmplitudeCoefficients for release 122022."""
    A = SimpleNamespace()
    eta = w.eta
    A.PNdominant = w.ampNorm * math.pow(2. / h.emm, -7 / 6.)
    ld = h.tag
    if ld == 21:
        A.PNdominantlm = np.where(w.q == 1, math.sqrt(2.) / 3. * 1.5 * w.dchi * 0.5 * math.pow(2 * LAL_PI / h.emm, 2 / 3.),
                                  math.sqrt(2.) / 3. * w.delta * math.pow(2 * LAL_PI / h.emm, 1 / 3.))
    elif ld == 33:
        A.PNdominantlm = np.where(w.q == 1, 0.75 * math.sqrt(5. / 7) * w.dchi * 0.5 * (65 / 24. - 28 / 3. * eta) * math.pow(2 * LAL_PI / h.emm, 4 / 3.),
                                  0.75 * math.sqrt(5. / 7) * w.delta * math.pow(2 * LAL_PI / h.emm, 1 / 3.))
    h.fAmpRDfalloff = h.fRING + 2 * h.fDAMP
    A.has_aux = h.mixing
    if h.mixing:
        A.fRDAux = h.fRING - h.fDAMP
    # cutting frequencies
    emmf = 1. * h.emm
    chi1 = w.chi1L
    fcutEMR = 1.25 * emmf * ((0.011671068725758493 - 0.0000858396080377194 * chi1 + 0.000316707064291237 * H.power(chi1, 2)) * (0.8447212540381764 + 6.2873167352395125 * eta)) / (1.2857082764038923 - 0.9977728883419751 * chi1)
    funcs = 0.5 + 0.5 * H.tanh((eta - 0.0192234) / 0.004)
    A.fAmpMatchIN = np.where(w.q < 20., h.fMECOlm, funcs * h.fMECOlm + (1 - funcs) * fcutEMR)
    if h.mixing:
        A.fAmpMatchIM = w.fRING - 0.5 * w.fDAMP
    else:
        A.fAmpMatchIM = h.fRING - h.fDAMP
    _pn_amp_coefficients(h, w, A)

    # Inspiral: 3 collocation points, version 123
    fi = [0.5 * A.fAmpMatchIN, 0.75 * A.fAmpMatchIN, A.fAmpMatchIN]
    names = ["IMRPhenomXHM_Insp_Amp_%d_iv%d" % (h.tag, i) for i in (1, 2, 3)]
    vals = [np.abs(_fit(nm, w, 122022)) for nm in names]
    pw = [_powers(f) for f in fi]
    pfin = _powers(A.fAmpMatchIN)
    A.fcutInsp_seven_thirds = pfin.seven_thirds
    A.fcutInsp_eight_thirds = pfin.eight_thirds
    A.fcutInsp_three = pfin.three
    pn = [_pn_amp_ansatz(p, h, A) for p in pw]
    A.cvinsp, A.pninsp = vals, pn
    v1, v2, v3 = [(vals[i] - pn[i]) / A.PNdominant / pw[i].m_seven_sixths for i in range(3)]
    f1, f2, f3, fs = pw[0], pw[1], pw[2], pfin
    den = (f1.seven_thirds * (f1.one_third - f2.one_third) * f2.seven_thirds * (f1.one_third - f3.one_third) * (f2.one_third - f3.one_third) * f3.seven_thirds)
    A.ic0 = (fs.seven_thirds * (-(f1.three * f3.eight_thirds * v2) + f1.eight_thirds * f3.three * v2 + f2.three * (f3.eight_thirds * v1 - f1.eight_thirds * v3) + f2.eight_thirds * (-(f3.three * v1) + f1.three * v3))) / den
    A.ic1 = (fs.eight_thirds * (f1.three * f3.seven_thirds * v2 - f1.seven_thirds * f3.three * v2 + f2.three * (-(f3.seven_thirds * v1) + f1.seven_thirds * v3) + f2.seven_thirds * (f3.three * v1 - f1.three * v3))) / den
    A.ic2 = (fs.three * (f1.seven_thirds * (-f1.one_third + f3.one_third) * f3.seven_thirds * v2 + f2.seven_thirds * (-(f3.eight_thirds * v1) + f1.eight_thirds * v3) + f2.eight_thirds * (f3.seven_thirds * v1 - f1.seven_thirds * v3))) / den

    # Ringdown
    if h.rd_amp_version == 1:
        A.rd0 = np.abs(_fit("IMRPhenomXHM_RD_Amp_32_alambda", w, 122022))
        A.rd1 = _fit("IMRPhenomXHM_RD_Amp_32_lambda", w, 122022)
        A.rd2 = _fit("IMRPhenomXHM_RD_Amp_32_sigma", w, 122022) * np.ones_like(eta)
    else:
        r1, r2, r3 = [np.abs(_fit("IMRPhenomXHM_RD_Amp_%d_rdcp%d" % (h.tag, i), w, 122022)) for i in (1, 2, 3)]
        r3 = np.where(r3 >= r2 * r2 / r1, 0.5 * r2 * r2 / r1, r3)
        r3 = np.where(r3 > r2, 0.5 * r2, r3)
        r3 = np.where((r1 < r2) & (r3 > r1), r1, r3)
        deno = (np.sqrt(r1 / r3) - (r1 / r2))
        deno = np.where(deno <= 0, 1e-16, deno)
        A.rd0 = r1 * h.fDAMP / deno
        A.rd2 = np.sqrt(A.rd0 / (r2 * h.fDAMP))
        A.rd1 = 0.5 * A.rd2 * H.log(r1 / r3)
    pfo = _powers(h.fAmpRDfalloff)
    A.has_aux = False
    A.rd3 = _rd_amp_ansatz(pfo, h, A, falloff=False)
    A.rd4 = -1. * _rd_amp_dansatz(pfo, h, A) / A.rd3
    if h.mixing:
        vals = [np.abs(_fit("IMRPhenomXHM_RD_Amp_32_rdaux%d" % i, w, 122022)) for i in (1, 2)]
        paux = _powers(A.fRDAux)
        vals.append(_rd_amp_ansatz(paux, h, A))
        vals.append(_rd_amp_dansatz(paux, h, A))
        fr = [A.fAmpMatchIM, 0.5 * (A.fAmpMatchIM + A.fRDAux), A.fRDAux, A.fRDAux]
        n = eta.shape[0]
        M = np.zeros((n, 4, 4))
        for i in range(4):
            fpower = np.ones(n)
            if i < 3:
                for j in range(4):
                    M[:, i, j] = fpower
                    fpower = fpower * fr[i]
            else:
                M[:, i, 0] = 0.
                for j in range(1, 4):
                    M[:, i, j] = j * fpower
                    fpower = fpower * fr[i]
        A.aux = list(_lu_solve(M, np.stack(vals, axis=-1)).T)
        A.has_aux = True

    # Intermediate
    nint = len(h.digits)
    deltaf = (A.fAmpMatchIM - A.fAmpMatchIN) / (nint - 1)
    freqs = []
    for i, d in enumerate(h.digits):
        for _ in range(d):
            freqs.append(A.fAmpMatchIN + deltaf * i)
    values = []
    pins = _powers(A.fAmpMatchIN)
    values.append(_insp_amp_ansatz(pins, h, A))
    if h.digits[0] == 2:
        values.append(_insp_amp_nd(A.fAmpMatchIN, h, A))
    for i in range(1, nint - 1):
        if h.digits[i] == 1:
            nm = "IMRPhenomXHM_Inter_Amp_%d_int%d" % (h.tag, i)
            values.append(np.abs(_fit(nm, w, 122022)))
    prd = _powers(A.fAmpMatchIM)
    if h.mixing:
        values.append(_cabs(_spheroidal_to_spherical(prd, h, A, P, w)))
        values.append(_rd_amp_nd_mixing(A.fAmpMatchIM, h, A, P, w))
    else:
        values.append(_rd_amp_ansatz(prd, h, A))
        values.append(_rd_amp_dansatz(prd, h, A))
    ncoef = h.nCoefficientsInter
    n = eta.shape[0]
    M = np.zeros((n, ncoef, ncoef))
    row = 0
    seven_sixths = 7 / 6.
    for i, d in enumerate(h.digits):
        if d == 0:
            continue
        fc = freqs[row]
        fcm = H.power(fc, -seven_sixths)
        fpower = np.ones(n)
        for j in range(ncoef):
            M[:, row, j] = fpower * fcm
            fpower = fpower * fc
        row += 1
        if d == 2:
            fpower = 1. / fc
            for j in range(ncoef):
                M[:, row, j] = (j - seven_sixths) * fpower * fcm
                fpower = fpower * fc
            row += 1
    A.cvinter = values
    A.inter = list(_lu_solve(M, np.stack(values, axis=-1)).T)
    return A


def _rd_amp_nd_mixing(f0, h, A, P, w):
    df = 10e-10
    vals = []
    for ff in (f0 + 2 * df, f0 + df, f0 - df, f0 - 2 * df):
        vals.append(_cabs(_spheroidal_to_spherical(_powers(ff), h, A, P, w)))
    return (-vals[0] + 8 * vals[1] - 8 * vals[2] + vals[3]) / (12 * df)


# ---- phase ----------------------------------------------------------------

def _insp_phase_int(Mf, p, P, xp=None):
    xpm = H if xp is None else xp
    freqs = [p.m_five_thirds, p.m_four_thirds, p.m_one, p.m_two_thirds, p.m_one_third, 1.,
             p.one_third, p.two_thirds, Mf, p.four_thirds, p.five_thirds, p.two, p.seven_thirds]
    logMf = p.log
    philm = 0.
    for i in range(13):
        if xp is not None and not P.nzL[i]:
            if P.nz[i]:
                philm = philm + P.phi[i] * freqs[i]
            continue
        philm = philm + (P.phi[i] + P.phiL[i] * logMf) * freqs[i]
    return philm


def _insp_phase(Mf, p, P):
    coeffs = [-5. / 3, -4. / 3, -1., -2. / 3, -1. / 3, 0., 1. / 3, 2. / 3, 1., 4. / 3, 5. / 3, 2., 7. / 3]
    freqs = [p.m_eight_thirds, p.m_seven_thirds, p.m_two, p.m_five_thirds, p.m_four_thirds, p.m_one,
             p.m_two_thirds, p.m_one_third, 1., p.one_third, p.two_thirds, Mf, p.four_thirds]
    logMf = p.log
    d = 0.
    for i in range(13):
        d = d + ((P.phi[i] + P.phiL[i] * logMf) * coeffs[i] + P.phiL[i]) * freqs[i]
    return d


def _inter_phase_int(ff, p, h, P, xp=None):
    xpm = H if xp is None else xp
    fda, frd = h.fDAMP, h.fRING
    out = P.c0 * ff + (P.c1) * p.log - (P.c2) * p.m_one - 1. / 3. * (P.c4) * p.m_three
    if h.mixing:
        out = out - 0.5 * P.c3 * p.m_two
    return out + (P.cL) * xpm.arctan((ff - frd) / fda)


def _inter_phase(ff, p, h, P):
    fda, frd = h.fDAMP, h.fRING
    out = P.c0 + (P.c1) * p.m_one + (P.c2) * p.m_two + (P.c4) * p.m_four
    if h.mixing:
        out = out + (P.c3) * p.m_three
    return out + ((P.cL) * fda / (fda * fda + (ff - frd) * (ff - frd)))


def _rd_phase(ff, p, h, P):
    frd, fda = h.fRING, h.fDAMP
    if not h.mixing:
        return (P.alpha0 + frd * frd * (P.alpha2) * p.m_two + ((P.alphaL) * fda / (fda * fda + (ff - frd) * (ff - frd))))
    return (P.alpha0_S + (P.alpha2_S) * p.m_two + (P.alpha4_S) * p.m_four
            + ((P.alphaL_S) * fda / (fda * fda + (ff - frd) * (ff - frd))))


def _rd_phase_int(ff, p, h, P, xp=None):
    xpm = H if xp is None else xp
    frd, fda = h.fRING, h.fDAMP
    if not h.mixing:
        return P.alpha0 * ff - frd * frd * (P.alpha2) * p.m_one + (P.alphaL) * xpm.arctan((ff - frd) / fda)
    return (P.phi0_S + P.alpha0_S * ff - (P.alpha2_S) * p.m_one - 1. / 3. * (P.alpha4_S) * p.m_three
            + (P.alphaL_S) * xpm.arctan((ff - frd) / fda))


def _spheroidal_setup(h, w, a, c, P):
    """GetSpheroidalCoefficients, ringdown phase version 122019 (4 points)."""
    frlm, fdlm = h.fRING, h.fDAMP
    fr = [w.fRING, frlm - 1.5 * fdlm, frlm - 0.5 * fdlm, frlm + 0.5 * fdlm]
    vals = [_fit("IMRPhenomXHM_RD_Phase_32_p%d" % (i + 1), w, 122019) for i in range(4)]
    n = w.eta.shape[0]
    M = np.zeros((n, 4, 4))
    for i in range(4):
        ff = fr[i]
        ffm1 = 1. / ff
        ffm2 = ffm1 * ffm1
        M[:, i, 0] = 1.
        M[:, i, 1] = (h.fDAMP) / (H.power(h.fDAMP, 2) + H.power(ff - (h.fRING), 2))
        M[:, i, 2] = ffm2
        M[:, i, 3] = ffm2 * ffm2
    x = _lu_solve(M, np.stack(vals, axis=-1))
    P.alpha0_S, P.alphaL_S, P.alpha2_S, P.alpha4_S = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    frefRD = w.fRING + w.fDAMP
    pref = _powers(frefRD)
    tshift = _fit("IMRPhenomXHM_RD_Phase_32_SpheroidalTimeShift", w, 122019)
    h.timeshift = w.timeshift
    P.phi0_S = np.zeros(n)
    dphi22ref = 1. / w.eta * _dphase22(frefRD, w, c) + h.timeshift
    P.alpha0_S = P.alpha0_S + dphi22ref + tshift - _rd_phase(frefRD, pref, h, P)
    frefRD = w.fRING
    pref = _powers(frefRD)
    h.phiref22 = w.phiref22
    phi22ref = 1. / w.eta * _phase22(frefRD, w, c) + h.timeshift * frefRD + h.phaseshift + h.phiref22
    phishift = _fit("IMRPhenomXHM_RD_Phase_32_SpheroidalPhaseShift", w, 122019)
    P.phi0_S = phi22ref - _rd_phase_int(frefRD, pref, h, P) + phishift


def _spheroidal_to_spherical(p, h, A, P, w, xp=None, c=None, a=None):
    """SpheroidalToSpherical on the host (``xp`` None) or the grid."""
    xpm = H if xp is None else xp
    S = h.s22
    Mf = p.itself
    dfr = Mf - S.fRING
    amp22 = xpm.exp(-dfr * S.gammaR) * (S.gammaD13) / (dfr * dfr + S.gammaD2)
    if xp is None:
        phi22 = 1. / S.eta * _phase22(Mf, S.w, S.c) + h.timeshift * Mf + h.phaseshift + h.phiref22
        wf22R = _cmul_real(amp22, _cexp_i(phi22))
        wf22R = _cmul_real(h.ampNorm * p.m_seven_sixths, wf22R)
        amplm = _rd_amp_ansatz(p, h, A)
        philm = _rd_phase_int(Mf, p, h, P)
        return h.mix2c * wf22R + h.mix3c * _cmul_real(amplm, _cexp_i(philm))
    phi22 = 1. / S.eta * X._phase_22(Mf, S.w, S.c, xp) + h.timeshift * Mf + h.phaseshift + h.phiref22
    wf22R = (amp22 * h.ampNorm * p.m_seven_sixths) * xp.exp(1j * phi22)
    amplm = _rd_amp_ansatz(p, h, A, xp)
    philm = _rd_phase_int(Mf, p, h, P, xp)
    return h.mix2c * wf22R + h.mix3c * (amplm * xp.exp(1j * philm))


def _phase_setup(h, w, a, c, A, P):
    """IMRPhenomXHM_GetPhaseCoefficients (versions 122019)."""
    ell, emm = h.ell, h.emm
    n = w.eta.shape[0]
    eta_m1 = 1. / w.eta
    # Intermediate collocation frequencies
    fcut = (1.0 + 0.001 * (0.25 / w.eta - 1.)) * emm * 0.5 * w.fMECO
    fring, fdamp = h.fRING, h.fDAMP
    if h.mixing:
        fEnd = w.fRING - 0.5 * w.fDAMP
        cf = [fcut, (math.sqrt(3) * (fcut - fEnd) + 2 * (fcut + fEnd)) / 4., (3 * fcut + fEnd) / 4.,
              (fcut + fEnd) / 2., fEnd, fEnd]
        P.fPhaseMatchIM = fEnd
        # LAL corrects the cutting frequency for EMR with negative spins
        # (IMRPhenomXHM_Intermediate_CollocPtsFreqs, phase freqs version 122019).
        emr_neg = (w.eta < 0.01) & (w.chi1L < 0)
        P.fPhaseMatchIM = np.where(emr_neg, P.fPhaseMatchIM * (1.2 - 0.25 * w.chi1L), P.fPhaseMatchIM)
    else:
        cf = [fcut, (math.sqrt(3) * (fcut - fring) + 2 * (fcut + fring)) / 4., (3 * fcut + fring) / 4.,
              (fcut + fring) / 2., (fcut + 3 * fring) / 4., (fcut + 7 * fring) / 8.]
        P.fPhaseMatchIM = fring - fdamp
    P.fPhaseMatchIN = h.fMECOlm
    cv = []
    for i in range(6):
        nm = "IMRPhenomXHM_Inter_Phase_%d_p%d" % (h.tag, i + 1)
        cv.append(_fit(nm, w, 122019) + h.DeltaT)
    fcutRD = P.fPhaseMatchIM
    fcutInsp = P.fPhaseMatchIN
    P.cvinter, P.cfinter = cv, cf

    # Inspiral: rescaled 22 coefficients
    z = np.zeros(n)
    nonlog = [c.phi0 * np.ones(n), z, c.phi2, c.phi3, c.phi4, z, c.phi6, c.phi7, c.phi8, z, z, z, z]
    logs = [z, z, z, z, z, c.phi5L, c.phi6L * np.ones(n), z, c.phi8L, z, z, z, z]
    pseudo = [z, z, z, z, z, z, z, z, c.sigma1, c.sigma2, c.sigma3, c.sigma4, c.sigma5]
    m_over_2 = emm * 0.5
    two_over_m = 1. / m_over_2
    fact = c.phiNorm / w.eta
    P.phi, P.phiL = [], []
    for i in range(13):
        nl = nonlog[i] * fact
        lg = logs[i] * fact
        ps = pseudo[i] * fact
        P.phi.append((nl + ps - lg * math.log(m_over_2)) * math.pow(m_over_2, (8 - i) / 3.))
        P.phiL.append(lg * math.pow(m_over_2, (8 - i) / 3.))
    PIv = PI.itself
    tag = h.tag
    if tag == 21:
        lam = (2. * PIv * (-0.5 - 2. * math.log(2)))
    elif tag == 33:
        lam = 2. / 3. * PIv * (-21. / 5. + 6. * math.log(1.5))
    elif tag == 32:
        lam = -((2376. * PIv * (-5. + 22. * w.eta)) / (-3960. + 11880 * w.eta))
    else:
        lam = 45045. * PIv * (336. - 1193. * w.eta + 320. * (-1. + 3. * w.eta) * math.log(2)) / (2. * (-1801800. + 5405400. * w.eta))
    lam = -1. * lam * np.ones(n)
    lam_fit = _fit("IMRPhenomXHM_Insp_Phase_%d_lambda" % tag, w, 122019)
    P.LambdaPN = np.where(w.eta > 0.01, lam, lam_fit)
    P.phi[8] = P.phi[8] + P.LambdaPN
    # Grid evaluation skips terms whose coefficients vanish for every point;
    # x + 0*y == x, so the result is unchanged.
    P.nz = [bool(np.any(v != 0)) for v in P.phi]
    P.nzL = [bool(np.any(v != 0)) for v in P.phiL]

    if tag == 21:
        hs = w.STotR >= 0.8
        insp = []
        for i in range(3):
            FF = two_over_m * cf[i]
            insp.append(1. / w.eta * _dphase22(FF, w, c))
        d12 = insp[0] - insp[1]
        d23 = insp[1] - insp[2]
        new1 = cv[2] + d23
        new0 = new1 + d12
        cv[1] = np.where(hs, new1, cv[1])
        cv[0] = np.where(hs, new0, cv[0])

    def lorentz(ff):
        return (h.fDAMP) / (H.power(h.fDAMP, 2) + H.power(ff - (h.fRING), 2))

    if not h.mixing:
        wlm = 2 if ell == emm else emm / 3.
        P.alpha2 = 1. / H.power(h.fRING, 2) * wlm * _fit("IMRPhenomXHM_RD_Phase_22_alpha2", w, 122019)
        P.alphaL = eta_m1 * _fit("IMRPhenomXHM_RD_Phase_22_alphaL", w, 122019)
        P.alpha0 = np.zeros(n)
        prd = _powers(fcutRD)
        P.phi0RD = _rd_phase_int(fcutRD, prd, h, P)
        P.dphi0RD = _rd_phase(fcutRD, prd, h, P)
        c1 = (w.eta < ETA_EMR) | ((emm == ell) & (w.STotR >= 0.8)) | ((tag == 33) & (w.STotR < 0))
        c2 = (~c1) & (w.STotR >= 0.8) & (tag == 21)
        idx2 = np.where(c1, 3, 2)
        idx3 = np.where(c1, 4, np.where(c2, 4, 3))
        cvs = np.stack(cv, axis=-1)
        cfs = np.stack([np.broadcast_to(f, (n,)) for f in cf], axis=-1)
        rows = np.arange(n)
        sel = [np.zeros(n, int), np.ones(n, int), idx2, idx3, np.full(n, 5)]
        M = np.zeros((n, 5, 5))
        b = np.zeros((n, 5))
        for i, ind in enumerate(sel):
            b[:, i] = cvs[rows, ind]
            ff = cfs[rows, ind]
            ffm1 = 1. / ff
            ffm2 = ffm1 * ffm1
            fp = [np.ones(n), lorentz(ff), ffm1, ffm2, ffm2 * ffm2]
            for j in range(5):
                M[:, i, j] = fp[j]
        x = _lu_solve(M, b)
        P.c0, P.cL, P.c1, P.c2, P.c4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        P.c3 = np.zeros(n)
    else:
        fstep = 0.0000001
        sph = []
        for i in range(3):
            FF = fcutRD + (i - 1) * fstep
            sph.append(_spheroidal_to_spherical(_powers(FF), h, A, P, w))
        twopi = np.longdouble(2. * LAL_PI)
        args = []
        for s in sph:
            v = np.fmod(np.asarray(_carg(s), dtype=np.longdouble), twopi)
            v = np.where(v > 0, v - twopi, v)
            args.append(v)
        P.phi0RD = args[1].astype(np.float64)
        fstep_m1 = np.longdouble(1. / fstep)
        P.dphi0RD = (np.longdouble(0.5) * fstep_m1 * (args[2] - args[0])).astype(np.float64)
        d2phi0RD = (fstep_m1 * fstep_m1 * (args[2] - np.longdouble(2.) * args[1] + args[0])).astype(np.float64)
        cv[4] = np.where(w.eta > ETA_EMR, P.dphi0RD, cv[4])
        cf[4] = np.where(w.eta > ETA_EMR, fcutRD, cf[4])
        cf[5] = fcutRD
        cv[5] = d2phi0RD
        M = np.zeros((n, 6, 6))
        b = np.zeros((n, 6))
        for i in range(6):
            b[:, i] = cv[i]
            ff = cf[i]
            ffm1 = 1. / ff
            ffm2 = ffm1 * ffm1
            fp = [np.ones(n), lorentz(ff), ffm1, ffm2, ffm2 * ffm2, ffm1 * ffm2]
            for j in range(6):
                M[:, i, j] = fp[j]
        ff = cf[5]
        ffm1 = 1. / ff
        ffm2 = ffm1 * ffm1
        ffm3 = ffm2 * ffm1
        ffm4 = ffm2 * ffm2
        ffm5 = ffm3 * ffm2
        fp = [np.zeros(n), -2 * (h.fDAMP) * (ff - (h.fRING)) / H.power((H.power(h.fDAMP, 2) + H.power(ff - (h.fRING), 2)), 2),
              -ffm2, -2. * ffm3, -4. * ffm5, -3 * ffm4]
        for j in range(6):
            M[:, 5, j] = fp[j]
        x = _lu_solve(M, b)
        P.c0, P.cL, P.c1, P.c2, P.c4, P.c3 = [x[:, k] for k in range(6)]
        prd = _powers(fcutRD)
        glue = P.c0 + P.dphi0RD - _inter_phase(fcutRD, prd, h, P)
        P.c0 = np.where(w.eta < ETA_EMR, glue, P.c0)

    pin = _powers(fcutInsp)
    P.C1INSP = _inter_phase(fcutInsp, pin, h, P) - _insp_phase(fcutInsp, pin, P)
    P.CINSP = -(P.C1INSP * fcutInsp) + _inter_phase_int(fcutInsp, pin, h, P) - _insp_phase_int(fcutInsp, pin, P)
    prd = _powers(fcutRD)
    P.C1RD = _inter_phase(fcutRD, prd, h, P) - P.dphi0RD
    P.CRD = -P.C1RD * fcutRD + _inter_phase_int(fcutRD, prd, h, P) - P.phi0RD

    falign = np.where(w.eta > ETA_EMR, 0.6 * m_over_2 * w.fMECO, m_over_2 * w.fMECO)
    pal = _powers(falign)
    if not h.mixing:
        h.timeshift = w.timeshift
        h.phiref22 = w.phiref22
    ph22 = _phase22(two_over_m * falign, w, c)
    dphi = m_over_2 * (1. / w.eta * ph22 + h.phaseshift + h.phiref22) + h.timeshift * falign \
        - 3. * LAL_PI_4 * (1 - m_over_2) - (_insp_phase_int(falign, pal, P) + P.C1INSP * falign + P.CINSP)
    P.deltaphiLM = np.fmod(dphi, 2. * LAL_PI)
    if tag == 21:
        eta, chi1, chi2 = w.eta, w.chi1L, w.chi2L
        delta = np.sqrt(1 - 4 * eta)
        ff = 0.008
        out = (-16 * delta * eta * ff * math.pow(LAL_PI, 1.5)) / (3. * math.sqrt(5)) \
            + (4 * math.pow(2, 0.3333333333333333) * (chi1 - chi2 + delta * (chi1 + chi2)) * eta * math.pow(ff, 1.3333333333333333) * math.pow(LAL_PI, 1.8333333333333333)) / math.sqrt(5) \
            + (2 * math.pow(2, 0.6666666666666666) * eta * (306 * delta - 360 * delta * eta) * math.pow(ff, 1.6666666666666667) * math.pow(LAL_PI, 2.1666666666666665)) / (189. * math.sqrt(5))
        P.deltaphiLM = np.where(out >= 0, P.deltaphiLM + LAL_PI, P.deltaphiLM)
    return P


def imrphenomxhm_setup(m1, m2, chi1z, chi2z, distance, phi_ref, f_min, f_ref, f_max,
                       modes=XHM_MODES):
    """Host setup for all requested higher modes; returns (w, a, c, per-mode list)."""
    m1 = np.atleast_1d(X._host(m1))
    n = m1.shape[0]
    m2 = X._as_points(X._host(m2), n, np)
    chi1z = X._as_points(X._host(chi1z), n, np)
    chi2z = X._as_points(X._host(chi2z), n, np)
    distance = X._as_points(X._host(distance), n, np)
    phi_ref = X._as_points(X._host(phi_ref), n, np)
    fref = float(f_ref) if f_ref else float(f_min)
    with np.errstate(divide="ignore", invalid="ignore"):   # vanishing odd modes at q = 1
        return _setup_all(m1, m2, chi1z, chi2z, distance, phi_ref, float(f_min), fref, float(f_max), modes)


def _setup_all(m1, m2, chi1z, chi2z, distance, phi_ref, f_min, fref, f_max, modes):
    w, a, c = _setup_22(m1, m2, chi1z, chi2z, distance, phi_ref, f_min, fref, f_max)
    s22 = SimpleNamespace(w=w, c=c, eta=w.eta, fRING=w.fRING, gammaR=a.gamma2 / (a.gamma3 * w.fDAMP),
                          gammaD13=w.fDAMP * a.gamma1 * a.gamma3,
                          gammaD2=(w.fDAMP * a.gamma3) * (w.fDAMP * a.gamma3))
    out = []
    for ell, emm in modes:
        emm = abs(emm)
        if (ell, emm) == (2, 2):
            out.append(None)
            continue
        h = _hm_variables(ell, emm, w)
        h.s22 = s22
        P = SimpleNamespace()
        if h.mixing:
            _spheroidal_setup(h, w, a, c, P)
        A = _amp_setup(h, w, P)
        P = _phase_setup(h, w, a, c, A, P)
        out.append((h, A, P))
    return w, a, c, out


# ---------------------------------------------------------------------------
# Grid evaluation (xp)
# ---------------------------------------------------------------------------

def _dev(v, xp):
    if isinstance(v, np.ndarray) and v.ndim == 1:
        v = v[:, None]
        return v if xp is np else xp.asarray(v)
    return v


def _ns_dev(ns, xp, skip=()):
    out = SimpleNamespace()
    for k, v in vars(ns).items():
        if k in skip:
            continue
        if isinstance(v, list):
            setattr(out, k, [_dev(x, xp) for x in v])
        else:
            setattr(out, k, _dev(v, xp))
    return out


_GRID_CACHE = {}


def _grid_factors(safe, xp):
    """Host f, f^(1/6), log f on ``xp``; the last grid is cached."""
    fs = X._host(safe)
    key = (id(xp), fs.shape[0], float(fs[0]), float(fs[-1]))
    hit = _GRID_CACHE.get("last")
    if hit is not None and hit[0] == key:
        return fs, hit[1], hit[2]
    g6 = xp.asarray(np.power(fs, 1.0 / 6.0))[None, :]
    lg = xp.asarray(np.log(fs))[None, :]
    _GRID_CACHE["last"] = (key, g6, lg)
    return fs, g6, lg


def _grid_powers(M_sec, safe, xp):
    """Powers of Mf = M_sec * f on the grid, identical on NumPy and CuPy.

    Mf^(1/6) = M_sec^(1/6) * f^(1/6) and log(Mf) = log(M_sec) + log(f), with the
    two transcendentals of f taken once on the host.  Every remaining operation
    is an IEEE multiply or divide, so both backends give the same bits.  A
    device pow differs from the host's by an ulp, which at the inspiral phases of
    a long BNS signal (1e6 rad) is a 1e-10 error."""
    fs, g6, lg = _grid_factors(safe, xp)
    Ms = X._host(M_sec)
    m6 = xp.asarray(H.power(Ms, 1.0 / 6.0))[:, None]
    lm = xp.asarray(H.log(Ms))[:, None]
    x = xp.asarray(Ms)[:, None] * xp.asarray(fs)[None, :]
    p = SimpleNamespace()
    sixth = m6 * g6
    p.one_sixth = sixth
    p.m_one_sixth = 1.0 / sixth
    p.itself = x
    p.m_one = 1.0 / x
    p.one_third = sixth * sixth
    p.m_one_third = 1.0 / p.one_third
    p.two_thirds = p.one_third * p.one_third
    p.m_two_thirds = 1.0 / p.two_thirds
    p.four_thirds = p.two_thirds * p.two_thirds
    p.m_four_thirds = 1.0 / p.four_thirds
    p.five_thirds = p.four_thirds * p.one_third
    p.m_five_thirds = 1.0 / p.five_thirds
    p.seven_thirds = p.four_thirds * x
    p.m_seven_thirds = 1.0 / p.seven_thirds
    p.eight_thirds = p.seven_thirds * p.one_third
    p.m_eight_thirds = 1.0 / p.eight_thirds
    p.two = x * x
    p.three = p.two * x
    p.four = p.three * x
    p.five = p.four * x
    p.m_two = 1.0 / p.two
    p.m_three = 1.0 / p.three
    p.m_four = 1.0 / p.four
    p.m_five = 1.0 / p.five
    p.seven_sixths = sixth * x
    p.m_seven_sixths = p.m_one_sixth * p.m_one
    p.log = lm + lg
    return p


def _hm_on_grid(p, h, A, P, xp):
    """h_{l,-m}(f) / amp0 for f > 0 (without the (-1)^l sign)."""
    hd = _ns_dev(h, xp, skip=("s22",))
    Ad = _ns_dev(A, xp)
    Pd = _ns_dev(P, xp)
    Mf = p.itself
    # amplitude
    a_ins = _insp_amp_ansatz(p, hd, Ad, xp)
    a_int = 0.0
    fpower = 1.0
    for cf in Ad.inter:
        a_int = a_int + cf * fpower
        fpower = fpower * Mf
    a_int = a_int * p.m_seven_sixths
    if h.mixing:
        s22 = h.s22
        hd.s22 = SimpleNamespace(eta=_dev(s22.eta, xp), fRING=_dev(s22.fRING, xp), gammaR=_dev(s22.gammaR, xp),
                                 gammaD13=_dev(s22.gammaD13, xp), gammaD2=_dev(s22.gammaD2, xp),
                                 w=X._expand(X._to_device(s22.w, xp), list(vars(s22.w))),
                                 c=X._expand(X._to_device(s22.c, xp), list(vars(s22.c))))
        sph = _spheroidal_to_spherical(p, hd, Ad, Pd, None, xp=xp)
        a_rd = xp.abs(sph)
        phi_rd = xp.angle(sph) + Pd.C1RD * Mf + Pd.CRD
    else:
        a_rd = _rd_amp_ansatz(p, hd, Ad, xp)
        phi_rd = _rd_phase_int(Mf, p, hd, Pd, xp) + Pd.C1RD * Mf + Pd.CRD
    amp = xp.where(Mf < Ad.fAmpMatchIN, a_ins, xp.where(Mf >= Ad.fAmpMatchIM, a_rd, a_int))
    amp = xp.where(amp < 0, FALSE_ZERO, amp)
    # phase
    phi_ins = _insp_phase_int(Mf, p, Pd, xp) + Pd.C1INSP * Mf + Pd.CINSP
    phi_int = _inter_phase_int(Mf, p, hd, Pd, xp)
    phi = xp.where(Mf < Pd.fPhaseMatchIN, phi_ins, xp.where(Mf >= Pd.fPhaseMatchIM, phi_rd, phi_int))
    phi = phi + Pd.deltaphiLM
    amp = xp.where(_dev(h.Ampzero, xp), 0.0, amp)
    return amp * xp.exp(1j * phi)


def imrphenomxhm_modes(m1, m2, chi1z, chi2z, distance, phi_ref, f_min, f_ref,
                       delta_f, n_freq, f_max=0.0, modes=XHM_MODES, xp=None,
                       two_sided=True, chunk=None):
    """IMRPhenomXHM modes h_lm(f) for n aligned-spin points.

    Equals ``SimInspiralChooseFDModes(..., IMRPhenomXHM)`` with
    ``PhenomXHMThresholdMband=0``.  Masses in Msun, distance in metres,
    ``f_ref=0`` means ``f_min``.  ``modes`` lists (l, m) with m of either sign.

    With ``two_sided`` (the default) returns (n, nmodes, 2*n_freq-1) on LAL's
    ascending grid ``f = (k - n_freq + 1) * delta_f``: h_{l,-|m|} lives at
    f > 0 and h_{l,|m|}(f) = (-1)^l conj(h_{l,-|m|}(-f)) at f < 0.  Otherwise
    returns (n, nmodes, n_freq) holding h_{l,-|m|}(k delta_f) for every entry.
    Points are evaluated in chunks of ``chunk`` (default: about 4e6 grid values
    per chunk) to bound the temporary memory.
    """
    xp = X._module(xp)
    m1 = np.atleast_1d(X._host(m1))
    n = m1.shape[0]
    args = [m1] + [X._as_points(X._host(v), n, np) for v in (m2, chi1z, chi2z, distance, phi_ref)]
    if chunk is None:
        chunk = max(1, int(4e6 // n_freq))
    if n <= chunk:
        return _modes_chunk(*args, f_min, f_ref, delta_f, n_freq, f_max, modes, xp, two_sided)
    width = 2 * n_freq - 1 if two_sided else n_freq
    out = xp.empty((n, len(modes), width), dtype=xp.complex128)
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        out[sl] = _modes_chunk(*[v[sl] for v in args], f_min, f_ref, delta_f, n_freq, f_max,
                               modes, xp, two_sided)
    return out


def _modes_chunk(m1, m2, chi1z, chi2z, distance, phi_ref, f_min, f_ref, delta_f, n_freq,
                 f_max, modes, xp, two_sided):
    w, a, c, per_mode = imrphenomxhm_setup(m1, m2, chi1z, chi2z, distance, phi_ref,
                                           f_min, f_ref, f_max, modes)
    wd = X._to_device(w, xp)
    freqs, safe = X._grid(delta_f, n_freq, f_min, xp)
    p = _grid_powers(w.M_sec, safe, xp)
    mask = X._frequency_mask(freqs, wd, float(f_min), float(delta_f), xp)
    amp0 = wd.amp0[:, None]
    out = []
    h22 = None
    for (ell, emm), entry in zip(modes, per_mode):
        if entry is None:
            if h22 is None:
                ad, cd = X._to_device(a, xp), X._to_device(c, xp)
                h22 = xp.where(mask, X._xas_22_on_grid(p.itself, wd, ad, cd, xp, p=p), 0.0)
            hneg = h22
        else:
            hh, A, P = entry
            sign = -1.0 if ell % 2 else 1.0
            hneg = xp.where(mask, (sign * amp0) * _hm_on_grid(p, hh, A, P, xp), 0.0)
        if two_sided:
            sgn = -1.0 if ell % 2 else 1.0
            if emm < 0:
                full = xp.concatenate([xp.zeros_like(hneg[:, :0:-1]), hneg], axis=-1)
            else:
                full = xp.concatenate([sgn * xp.conj(hneg[:, :0:-1]), xp.zeros_like(hneg)], axis=-1)
            out.append(full)
        else:
            out.append(hneg)
    return xp.stack(out, axis=1)
