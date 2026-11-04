"""Batched IMRPhenomXAS (2,2) and IMRPhenomXP NNLO twisting-up for CuPy or NumPy.

Stage 1 of a dependency-free GPU port of IMRPhenomXPHM.  Every function takes
an array module ``xp`` (``cupy`` or ``numpy``) and evaluates many intrinsic
points at once: parameters are arrays of shape ``(n,)`` and frequency-domain
outputs have shape ``(n, ..., n_freq)``.  The per-point setup (fits, 5x5
collocation solves, precession angles at f_ref) always runs on the host in
NumPy; only the frequency-grid evaluation runs on ``xp``.  There is no JAX or
Ripple import.

Implemented here, with LALSuite defaults:

* IMRPhenomXAS (2,2): InsPhaseVersion 104, IntPhaseVersion 105,
  RDPhaseVersion 105, InsAmpVersion 103, IntAmpVersion 104, RDAmpVersion 103,
  QNM fits; no tides, no PNR, no non-GR deformations.
* IMRPhenomXP twisting-up with PrecVersion 102 (NNLO single-spin PN Euler
  angles, 3PN spinning L), Convention 1, FinalSpinMod 4 (which LALSuite maps to
  version 0, the chi_p final spin, for PrecVersion 102).

Not implemented: MSA angles (PrecVersion 223, the XPHM default), the XHM higher
modes, multibanding.  See ``DESIGN_gpu_waveform_xphm.md`` beside this file.

Provenance and licence
----------------------
This file is a line-by-line re-implementation of the LALSimulation sources
``LALSimIMRPhenomX.c``, ``LALSimIMRPhenomX_internals.c``,
``LALSimIMRPhenomX_inspiral.c``, ``LALSimIMRPhenomX_intermediate.c``,
``LALSimIMRPhenomX_ringdown.c``, ``LALSimIMRPhenomX_qnm.c``,
``LALSimIMRPhenomXUtilities.c``, ``LALSimIMRPhenomX_precession.c`` and
``LALSimIMRPhenomXPHM.c`` at tag ``lalsimulation-v6.2.0``
(Copyright (C) 2018-2021 Geraint Pratten, Cecilio Garcia-Quiros, Marta Colleoni,
Hector Estelles, Sascha Husa and the LALSuite authors).  The numerical fit
coefficients are copied from those files.  LALSuite is distributed under the
GNU General Public License, version 2 or later; this derived file is therefore
distributed under the same terms (GPL-2.0-or-later), not under the MIT licence
of the rest of RIFT.

References: Pratten et al., PRD 102, 064001 (2020), arXiv:2001.11412
(IMRPhenomXAS); Pratten et al., PRD 103, 104056 (2021), arXiv:2004.06503
(IMRPhenomXPHM).
"""

import math
from types import SimpleNamespace

import numpy as np

__all__ = [
    "imrphenomxas_h22",
    "imrphenomxp_nnlo_modes",
    "imrphenomxp_nnlo_polarizations",
    "pack_two_sided",
    "wigner_D_l2",
    "jframe_to_rift_lframe",
    "rift_jl_angles",
]

# LAL constants (lal 7.x, LALConstants.h)
LAL_PI = 3.141592653589793238462643383279502884
LAL_PI_2 = 1.570796326794896619231321691639751442
LAL_PI_4 = 0.785398163397448309615660845819875721
LAL_GAMMA = 0.577215664901532860606512090082402431
LAL_MTSUN_SI = 4.925490947641266978e-06
LAL_MRSUN_SI = 1.476625038050124729e+03
LAL_MSUN_SI = 1.988409870698050731e+30
MAX_TOL_ATAN = 1.0e-15


def _module(xp):
    return np if xp is None else xp


def _host(value):
    """Host NumPy copy of a NumPy/CuPy array or scalar."""
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float64)


def _to_device(ns, xp):
    """Copy every NumPy array attribute of a namespace to ``xp``."""
    if xp is np:
        return ns
    out = SimpleNamespace()
    for name, value in vars(ns).items():
        setattr(out, name, xp.asarray(value) if isinstance(value, np.ndarray) else value)
    return out


def _powers(x, xp):
    """IMRPhenomX_Initialize_Powers, elementwise."""
    p = SimpleNamespace()
    sixth = xp.power(x, 1.0 / 6.0)
    p.one_sixth = sixth
    p.m_one_sixth = 1.0 / sixth
    p.itself = x
    p.m_one = 1.0 / x
    p.one_third = sixth * sixth
    p.m_one_third = 1.0 / p.one_third
    p.two_thirds = p.one_third * p.one_third
    p.m_two_thirds = 1.0 / p.two_thirds
    p.four_thirds = p.two_thirds * p.two_thirds
    p.five_thirds = p.four_thirds * p.one_third
    p.m_five_thirds = 1.0 / p.five_thirds
    p.seven_thirds = p.four_thirds * x
    p.eight_thirds = p.seven_thirds * p.one_third
    p.m_eight_thirds = 1.0 / p.eight_thirds
    p.two = x * x
    p.three = p.two * x
    p.four = p.three * x
    p.m_two = 1.0 / p.two
    p.m_three = 1.0 / p.three
    p.m_four = 1.0 / p.four
    p.seven_sixths = sixth * x
    p.m_seven_sixths = p.m_one_sixth * p.m_one
    p.log = xp.log(x)
    return p


PI = _powers(LAL_PI, np)  # powers_of_lalpi (host scalars)


# ---------------------------------------------------------------------------
# Spin parameterizations and final state (LALSimIMRPhenomXUtilities.c)
# ---------------------------------------------------------------------------

def _chi_eff(eta, c1, c2, xp):
    delta = xp.sqrt(1.0 - 4.0 * eta)
    return 0.5 * (1 + delta) * c1 + 0.5 * (1 - delta) * c2


def _chi_pn_hat(eta, c1, c2, xp):
    delta = xp.sqrt(1.0 - 4.0 * eta)
    chi_eff = 0.5 * (1.0 + delta) * c1 + 0.5 * (1.0 - delta) * c2
    return (chi_eff - (38.0 / 113.0) * eta * (c1 + c2)) / (1.0 - (76.0 * eta / 113.0))


def _stot_r(eta, c1, c2, xp):
    delta = xp.sqrt(1.0 - 4.0 * eta)
    m1s = (0.5 * (1 + delta)) ** 2
    m2s = (0.5 * (1 - delta)) ** 2
    return (m1s * c1 + m2s * c2) / (m1s + m2s)


def _fmeco(eta, c1, c2, xp):
    eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta
    delta = xp.sqrt(1.0 - 4.0 * eta)
    S = _chi_pn_hat(eta, c1, c2, xp)
    S2 = S * S; S3 = S2 * S
    dchi = c1 - c2
    dchi2 = dchi * dchi
    noSpin = (0.018744340279608845 + 0.0077903147004616865*eta + 0.003940354686136861*eta2 - 0.00006693930988501673*eta3)/(1. - 0.10423384680638834*eta)
    eqSpin = (S*(0.00027180386951683135 - 0.00002585252361022052*S + eta4*(-0.0006807631931297156 + 0.022386313074011715*S - 0.0230825153005985*S2) + eta2*(0.00036556167661117023 - 0.000010021140796150737*S - 0.00038216081981505285*S2) + eta*(0.00024422562796266645 - 0.00001049013062611254*S - 0.00035182990586857726*S2) + eta3*(-0.0005418851224505745 + 0.000030679548774047616*S + 4.038390455349854e-6*S2) - 0.00007547517256664526*S2))/(0.026666543809890402 + (-0.014590539285641243 - 0.012429476486138982*eta + 1.4861197211952053*eta4 + 0.025066696514373803*eta2 + 0.005146809717492324*eta3)*S + (-0.0058684526275074025 - 0.02876774751921441*eta - 2.551566872093786*eta4 - 0.019641378027236502*eta2 - 0.001956646166089053*eta3)*S2 + (0.003507640638496499 + 0.014176504653145768*eta + 1.*eta4 + 0.012622225233586283*eta2 - 0.00767768214056772*eta3)*S3)
    uneqSpin = dchi2*(0.00034375176678815234 + 0.000016343732281057392*eta)*eta2 + dchi*delta*eta*(0.08064665214195679*eta2 + eta*(-0.028476219509487793 - 0.005746537021035632*S) - 0.0011713735642446144*S)
    return noSpin + eqSpin + uneqSpin


def _fisco(chif, xp):
    Z1 = 1.0 + xp.cbrt(1.0 - chif * chif) * (xp.cbrt(1 + chif) + xp.cbrt(1 - chif))
    Z1 = xp.minimum(Z1, 3.0)
    Z2 = xp.sqrt(3.0 * chif * chif + Z1 * Z1)
    rISCO = 3.0 + Z2 - xp.sign(chif) * xp.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    rISCOsq = xp.sqrt(rISCO)
    OmegaISCO = 1.0 / (rISCOsq * rISCOsq * rISCOsq + chif)
    return OmegaISCO / LAL_PI


def _final_mass_2017(eta, c1, c2, xp):
    delta = xp.sqrt(1.0 - 4.0 * eta)
    eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta
    S = _stot_r(eta, c1, c2, xp)
    S2 = S * S; S3 = S2 * S
    dchi = c1 - c2
    dchi2 = dchi * dchi
    noSpin = 0.057190958417936644*eta + 0.5609904135313374*eta2 - 0.84667563764404*eta3 + 3.145145224278187*eta4
    eqSpin = ((0.057190958417936644*eta + 0.5609904135313374*eta2 - 0.84667563764404*eta3 + 3.145145224278187*eta4)*
              (1
               + (-0.13084389181783257 - 1.1387311580238488*eta + 5.49074464410971*eta2)*S
               + (-0.17762802148331427 + 2.176667900182948*eta2)*S2
               + (-0.6320191645391563 + 4.952698546796005*eta - 10.023747993978121*eta2)*S3)) \
        / (1 + (-0.9919475346968611 + 0.367620218664352*eta + 4.274567337924067*eta2)*S)
    eqSpin = eqSpin - noSpin
    uneqSpin = (- 0.09803730445895877*dchi*delta*(1 - 3.2283713377939134*eta)*eta2
                + 0.01118530335431078*dchi2*eta3
                - 0.01978238971523653*dchi*delta*(1 - 4.91667749015812*eta)*eta*S)
    return 1.0 - (noSpin + eqSpin + uneqSpin)


def _final_spin_2017(eta, c1, c2, xp):
    delta = xp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)
    m1Sq = m1 * m1; m2Sq = m2 * m2
    eta2 = eta * eta; eta3 = eta2 * eta
    S = _stot_r(eta, c1, c2, xp)
    S2 = S * S; S3 = S2 * S
    dchi = c1 - c2
    dchi2 = dchi * dchi
    noSpin = (3.4641016151377544*eta + 20.0830030082033*eta2 - 12.333573402277912*eta3)/(1 + 7.2388440419467335*eta)
    eqSpin = (m1Sq + m2Sq)*S \
        + ((-0.8561951310209386*eta - 0.09939065676370885*eta2 + 1.668810429851045*eta3)*S
           + (0.5881660363307388*eta - 2.149269067519131*eta2 + 3.4768263932898678*eta3)*S2
           + (0.142443244743048*eta - 0.9598353840147513*eta2 + 1.9595643107593743*eta3)*S3) \
        / (1 + (-0.9142232693081653 + 2.3191363426522633*eta - 9.710576749140989*eta3)*S)
    uneqSpin = (0.3223660562764661*dchi*delta*(1 + 9.332575956437443*eta)*eta2
                - 0.059808322561702126*dchi2*eta3
                + 2.3170397514509933*dchi*delta*(1 - 3.2624649875884852*eta)*eta3*S)
    return noSpin + eqSpin + uneqSpin


def _precessing_final_spin_2017(eta, c1, c2, chi_inplane, xp):
    # m1 >= m2 always holds here (m1 = (1+delta)/2), so q_factor = m1.
    m1 = 0.5 * (1.0 + xp.sqrt(1 - 4.0 * eta))
    af_parallel = _final_spin_2017(eta, c1, c2, xp)
    Sperp = chi_inplane * m1 * m1
    return xp.copysign(1.0, af_parallel) * xp.sqrt(Sperp * Sperp + af_parallel * af_parallel)


def _qnm_fring22(a):
    x2 = a*a; x3 = x2*a; x4 = x2*x2; x5 = x3*x2; x6 = x3*x3; x7 = x4*x3
    return (0.05947169566573468 - 0.14989771215394762*a + 0.09535606290986028*x2
            + 0.02260924869042963*x3 - 0.02501704155363241*x4 - 0.005852438240997211*x5
            + 0.0027489038393367993*x6 + 0.0005821983163192694*x7) / (
        1 - 2.8570126619966296*a + 2.373335413978394*x2 - 0.6036964688511505*x4
        + 0.0873798215084077*x6)


def _qnm_fdamp22(a):
    x2 = a*a; x3 = x2*a; x4 = x2*x2; x5 = x3*x2; x6 = x3*x3
    return (0.014158792290965177 - 0.036989395871554566*a + 0.026822526296575368*x2
            + 0.0008490933750566702*x3 - 0.004843996907020524*x4
            - 0.00014745235759327472*x5 + 0.0001504546201236794*x6) / (
        1 - 2.5900842798681376*a + 1.8952576220623967*x2 - 0.31416610693042507*x4
        + 0.009002719412204133*x6)


def _linb(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta; eta5 = eta3*eta2; eta6 = eta4*eta2
    S2 = S*S; S3 = S2*S; S4 = S3*S
    noSpin = 3155.1635543201924 + 1257.9949740608242*eta - 32243.28428870599*eta2 + 347213.65466875216*eta3 - 1.9223851649491738e6*eta4 + 5.3035911346921865e6*eta5 - 5.789128656876938e6*eta6
    eqSpin = (-24.181508118588667 + 115.49264174560281*eta - 380.19778216022763*eta2)*S + (24.72585609641552 - 328.3762360751952*eta + 725.6024119989094*eta2)*S2 + (23.404604124552 - 646.3410199799737*eta + 1941.8836639529036*eta2)*S3 + (-12.814828278938885 - 325.92980012408367*eta + 1320.102640190539*eta2)*S4
    uneqSpin = -148.17317525117338*dchi*delta*eta2
    return noSpin + eqSpin + uneqSpin


def _psi4_to_strain(eta, S, dchi, xp):
    eta2 = eta**2; eta3 = eta**3; eta4 = eta**4
    S2 = S**2; S3 = S**3; S4 = S**4
    noSpin = 13.39320482758057 - 175.42481512989315*eta + 2097.425116152503*eta2 - 9862.84178637907*eta3 + 16026.897939722587*eta4
    eqSpin = (4.7895602776763 - 163.04871764530466*eta + 609.5575850476959*eta2)*S + (1.3934428041390161 - 97.51812681228478*eta + 376.9200932531847*eta2)*S2 + (15.649521097877374 + 137.33317057388916*eta - 755.9566456906406*eta2)*S3 + (13.097315867845788 + 149.30405703643288*eta - 764.5242164872267*eta2)*S4
    uneqSpin = 105.37711654943146*dchi*xp.sqrt(1. - 4.*eta)*eta2
    return noSpin + eqSpin + uneqSpin


# ---------------------------------------------------------------------------
# Collocation-point fits (default versions only)
# ---------------------------------------------------------------------------

def _ins_amp_v2(eta, S, dchi, delta):  # InsAmpVersion 103, S = chiPNHat
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta; S2 = S*S; S3 = S2*S
    noSpin = (-0.015178276424448592 - 0.06098548699809163*eta + 0.4845148547154606*eta2)/(1. + 0.09799277215675059*eta)
    eqSpin = ((0.02300153747158323 + 0.10495263104245876*eta2)*S + (0.04834642258922544 - 0.14189350657140673*eta)*eta*S3
              + (0.01761591799745109 - 0.14404522791467844*eta2)*S2)/(1. - 0.7340448493183307*S)
    uneqSpin = dchi*delta*eta4*(0.0018724905795891192 + 34.90874132485147*eta)
    return noSpin + eqSpin + uneqSpin


def _ins_amp_v3(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta
    noSpin = (-0.058572000924124644 - 1.1970535595488723*eta + 8.4630293045015*eta2)/(1. + 15.430818840453686*eta)
    eqSpin = ((-0.08746408292050666 + eta*(-0.20646621646484237 - 0.21291764491897636*S)
               + eta2*(0.788717372588848 + 0.8282888482429105*S) - 0.018924013869130434*S)*S)/(-1.332123330797879 + 1.*S)
    uneqSpin = dchi*delta*eta4*(0.004389995099201855 + 105.84553997647659*eta)
    return noSpin + eqSpin + uneqSpin


def _ins_amp_v4(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta; S2 = S*S
    noSpin = (-0.16212854591357853 + 1.617404703616985*eta - 3.186012733446088*eta2 + 5.629598195000046*eta3)/(1. + 0.04507019231274476*eta)
    eqSpin = (S*(1.0055835408962206 + eta2*(18.353433894421833 - 18.80590889704093*S) - 0.31443470118113853*S
                 + eta*(-4.127597118865669 + 5.215501942120774*S) + eta3*(-41.0378120175805
                                                                          + 19.099315016873643*S)))/(5.852706459485663 - 5.717874483424523*S + 1.*S2)
    uneqSpin = dchi*delta*eta4*(0.05575955418803233 + 208.92352600701068*eta)
    return noSpin + eqSpin + uneqSpin


def _ins_amp_rho(v1, v2, v3, F1, F2, F3, xp):
    f1p1o3 = xp.cbrt(F1); f2p1o3 = xp.cbrt(F2); f3p1o3 = xp.cbrt(F3)
    f1p7o3 = f1p1o3**7; f1p8o3 = f1p7o3 * f1p1o3
    f2p7o3 = f2p1o3**7; f2p8o3 = f2p7o3 * f2p1o3
    f3p7o3 = f3p1o3**7; f3p8o3 = f3p7o3 * f3p1o3
    den = (f1p7o3*(f1p1o3 - f2p1o3)*f2p7o3*(f1p1o3 - f3p1o3)*(f2p1o3 - f3p1o3)*f3p7o3)
    rho1 = (-(f2p8o3*(F3*F3*F3)*v1) + F2*F2*F2*f3p8o3*v1 + f1p8o3*(F3*F3*F3)*v2 - F1*F1*F1*f3p8o3*v2 - f1p8o3*(F2*F2*F2)*v3 + F1*F1*F1*f2p8o3*v3) / den
    rho2 = (f2p7o3*(F3*F3*F3)*v1 - F2*F2*F2*f3p7o3*v1 - f1p7o3*(F3*F3*F3)*v2 + F1*F1*F1*f3p7o3*v2 + f1p7o3*(F2*F2*F2)*v3 - F1*F1*F1*f2p7o3*v3) / den
    rho3 = (f2p8o3*f3p7o3*v1 - f2p7o3*f3p8o3*v1 - f1p8o3*f3p7o3*v2 + f1p7o3*f3p8o3*v2 + f1p8o3*f2p7o3*v3 - f1p7o3*f2p8o3*v3) / den
    return rho1, rho2, rho3


def _int_amp_vA(eta, S, dchi, delta):  # IntAmpVersion 104, S = STotR
    eta2 = eta*eta; eta3 = eta2*eta; S2 = S*S
    noSpin = (1.4873184918202145 + 1974.6112656679577*eta + 27563.641024162127*eta2 - 19837.908020966777*eta3)/(1. + 143.29004876335128*eta + 458.4097306093354*eta2)
    eqSpin = (S*(27.952730865904343 + eta*(-365.55631765202895 - 260.3494489873286*S) + 3.2646808851249016*S + 3011.446602208493*eta2*S - 19.38970173389662*S2 + eta3*(1612.2681322644232 - 6962.675551371755*S + 1486.4658089990298*S2)))/(12.647425554323242 - 10.540154508599963*S + 1.*S2)
    uneqSpin = dchi*delta*(-0.016404056649860943 - 296.473359655246*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def _int_amp_deltas_104(d1, d4, v1, v2, v4, f1, f2, f4):
    f12 = f1*f1; f13 = f12*f1; f14 = f13*f1; f15 = f14*f1
    f22 = f2*f2; f23 = f22*f2; f24 = f23*f2
    f42 = f4*f4; f43 = f42*f4; f44 = f43*f4; f45 = f44*f4
    f1mf2 = f1-f2; f1mf4 = f1-f4; f2mf4 = f2-f4
    f1mf22 = f1mf2*f1mf2; f2mf42 = f2mf4*f2mf4
    f1mf43 = f1mf4*f1mf4*f1mf4
    den = (f1mf22*f1mf43*f2mf42)
    delta0 = ((-(d4*f12*f1mf22*f1mf4*f2*f2mf4*f4) + d1*f1*f1mf2*f1mf4*f2*f2mf42*f42 + f42*(f2*f2mf42*(-4*f12 + 3*f1*f2 + 2*f1*f4 - f2*f4)*v1 + f12*f1mf43*v2) +
               f12*f1mf22*f2*(f1*f2 - 2*f1*f4 - 3*f2*f4 + 4*f42)*v4)/den)
    delta1 = ((d4*f1*f1mf22*f1mf4*f2mf4*(2*f2*f4 + f1*(f2 + f4)) + f4*(-(d1*f1mf2*f1mf4*f2mf42*(2*f1*f2 + (f1 + f2)*f4)) -
              2*f1*(f44*(v1 - v2) + 3*f24*(v1 - v4) + f14*(v2 - v4) + 4*f23*f4*(-v1 + v4)
                    + 2*f13*f4*(-v2 + v4) + f1*(2*f43*(-v1 + v2) + 6*f22*f4*(v1 - v4) + 4*f23*(-v1 + v4)))))/den)
    delta2 = ((-(d4*f1mf22*f1mf4*f2mf4*(f12 + f2*f4 + 2*f1*(f2 + f4))) + d1*f1mf2*f1mf4*f2mf42*(f1*f2 + 2*(f1 + f2)*f4 + f42)
               - 4*f12*f23*v1 + 3*f1*f24*v1 - 4*f1*f23*f4*v1 + 3*f24*f4*v1 + 12*f12*f2*f42*v1 -
               4*f23*f42*v1 - 8*f12*f43*v1 + f1*f44*v1 + f45*v1 + f15*v2 + f14*f4*v2 - 8*f13*f42*v2 + 8*f12*f43*v2 - f1*f44*v2 - f45*v2 -
               f1mf22*(f13 + f2*(3*f2 - 4*f4)*f4 + f12*(2*f2 + f4) + f1*(3*f2 - 4*f4)*(f2 + 2*f4))*v4)/den)
    delta3 = ((d4*f1mf22*f1mf4*f2mf4*(2*f1 + f2 + f4) - d1*f1mf2*f1mf4*f2mf42*(f1 + f2 + 2*f4)
               + 2*(f44*(-v1 + v2) + 2*f12*f2mf42*(v1 - v4) + 2*f22*f42*(v1 - v4)
                    + 2*f13*f4*(v2 - v4) + f24*(-v1 + v4) + f14*(-v2 + v4) + 2*f1*f4*(f42*(v1 - v2) + f22*(v1 - v4) + 2*f2*f4*(-v1 + v4)))) / den)
    delta4 = ((-(d4*f1mf22*f1mf4*f2mf4) + d1*f1mf2*f1mf4*f2mf42 - 3*f1*f22*v1 + 2*f23*v1 + 6*f1*f2*f4*v1 - 3*f22*f4*v1
               - 3*f1*f42*v1 + f43*v1 + f13*v2 - 3*f12*f4*v2 + 3*f1*f42*v2 - f43*v2 - f1mf22*(f1 + 2*f2 - 3*f4)*v4)/den)
    return delta0, delta1, delta2, delta3, delta4


def _rd_amp_gamma2(eta, S, dchi, delta):  # RDAmpVersion 103, S = STotR
    S2 = S*S; eta2 = eta*eta
    noSpin = (0.8312293675316895 + 7.480371544268765*eta - 18.256121237800397*eta2)/(1. + 10.915453595496611*eta - 30.578409433912874*eta2)
    eqSpin = (S*(0.5869408584532747 + eta*(-0.1467158405070222 - 2.8489481072076472*S) + 0.031852563636196894*S + eta2*(0.25295441250444334 + 4.6849496672664594*S)))/(3.8775263105069953 - 3.41755361841226*S + 1.*S2)
    uneqSpin = -0.00548054788508203*dchi*delta*eta
    return noSpin + eqSpin + uneqSpin


def _rd_amp_gamma3(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta
    noSpin = (1.3666000000000007 - 4.091333144596439*eta + 2.109081209912545*eta2 - 4.222259944408823*eta3)/(1. - 2.7440263888207594*eta)
    eqSpin = (0.07179105336478316 + eta2*(2.331724812782498 - 0.6330998412809531*S) + eta*(-0.8752427297525086 + 0.4168560229353532*S) - 0.05633734476062242*S)*S
    return noSpin + eqSpin


def _rd_amp_v1(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta
    S2 = S*S; S3 = S*S2; S4 = S*S3
    noSpin = (0.03689164742964719 + 25.417967754401182*eta + 162.52904393600332*eta2)/(1. + 61.19874463331437*eta - 29.628854485544874*eta2)
    eqSpin = (S*(-0.14352506969368556 + 0.026356911108320547*S + 0.19967405175523437*S2 - 0.05292913111731128*S3 + eta3*(-48.31945248941757 - 3.751501972663298*S + 81.9290740950083*S2 + 30.491948143930266*S3 - 132.77982622925845*S4) + eta*(-4.805034453745424 + 1.11147906765112*S + 6.176053843938542*S2 - 0.2874540719094058*S3 - 8.990840289951514*S4) - 0.18147275151697131*S4 + eta2*(27.675454081988036 - 2.398327419614959*S - 47.99096500250743*S2 - 5.104257870393138*S3 + 72.08174136362386*S4)))/(-1.4160870461211452 + 1.*S)
    uneqSpin = -0.04426571511345366*dchi*delta*eta2
    return noSpin + eqSpin + uneqSpin


def _ins_phase_v3(eta, S, dchi, delta):  # InsPhaseVersion 104, S = chiPNHat
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta
    S2 = S*S; S3 = S2*S; dchi2 = dchi*dchi
    noSpin = (15415.000000000007 + 873401.6255736464*eta + 376665.64637025696*eta2 - 3.9719980569125614e6*eta3 + 8.913612508054944e6*eta4)/(1. + 46.83697749859996*eta)
    eqSpin = (S*(397951.95299014193 - 207180.42746987*S + eta3*(4.662143741417853e6 - 584728.050612325*S - 1.6894189124921719e6*S2) + eta*(-1.0053073129700898e6 + 1.235279439281927e6*S - 174952.69161683554*S2) - 130668.37221912303*S2 + eta2*(-1.9826323844247842e6 + 208349.45742548333*S + 895372.155565861*S2)))/(-9.675704197652225 + 3.5804521763363075*S + 2.5298346636273306*S2 + 1.*S3)
    uneqSpin = -1296.9289110696955*dchi2*eta + dchi*delta*eta*(-24708.109411857182 + 24703.28267342699*eta + 47752.17032707405*S)
    return noSpin + eqSpin + uneqSpin


def _ins_phase_d13(eta, S, dchi, delta):
    eta2 = eta*eta
    S2 = S*S; S3 = S2*S; S4 = S3*S
    noSpin = (-17294.000000000007 - 19943.076428555978*eta + 483033.0998073767*eta2)/(1. + 4.460294035404433*eta)
    eqSpin = (S*(68384.62786426462 + 67663.42759836042*S - 2179.3505885609297*S2 + eta*(-58475.33302037833 + 62190.404951852535*S + 18298.307770807573*S2 - 303141.1945565486*S3) + 19703.894135534803*S3 + eta2*(-148368.4954044637 - 758386.5685734496*S - 137991.37032619823*S2 + 1.0765877367729193e6*S3) + 32614.091002011017*S4))/(2.0412979553629143 + 1.*S)
    uneqSpin = 12017.062595934838*dchi*delta*eta
    return noSpin + eqSpin + uneqSpin


def _ins_phase_d23(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta
    S2 = S*S; S3 = S2*S
    noSpin = (-7579.300000000004 - 120297.86185566607*eta + 1.1694356931282217e6*eta2 - 557253.0066989232*eta3)/(1. + 18.53018618227582*eta)
    eqSpin = (S*(-27089.36915061857 - 66228.9369155027*S + eta2*(150022.21343386435 - 50166.382087278434*S - 399712.22891153296*S2) - 44331.41741405198*S2 + eta*(50644.13475990821 + 157036.45676788126*S + 126736.43159783827*S2) + eta3*(-593633.5370110178 - 325423.99477314285*S + 847483.2999508682*S2)))/(-1.5232497464826662 - 3.062957826830017*S - 1.130185486082531*S2 + 1.*S3)
    uneqSpin = 3843.083992827935*dchi*delta*eta
    return noSpin + eqSpin + uneqSpin


def _ins_phase_d43(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta
    S2 = S*S
    noSpin = (2439.000000000001 - 31133.52170083207*eta + 28867.73328134167*eta2)/(1. + 0.41143032589262585*eta)
    eqSpin = (S*(16116.057657391262 + eta3*(-375818.0132734753 - 386247.80765802023*S) + eta*(-82355.86732027541 - 25843.06175439942*S) + 9861.635308837876*S + eta2*(229284.04542668918 + 117410.37432997991*S)))/(-3.7385208695213668 + 0.25294420589064653*S + 1.*S2)
    uneqSpin = 194.5554531509207*dchi*delta*eta
    return noSpin + eqSpin + uneqSpin


def _int_phase_v2(eta, S, dchi, delta):  # IntPhaseVersion 105, S = STotR
    eta2 = eta*eta; eta3 = eta2*eta
    S2 = S*S; S3 = S2*S; S4 = S3*S
    noSpin = (-82.54500000000004 - 5.58197349185435e6*eta - 3.5225742421184325e8*eta2 + 1.4667258334378073e9*eta3)/(1. + 66757.12830903867*eta + 5.385164380400193e6*eta2 + 2.5176585751772933e6*eta3)
    eqSpin = (S*(19.416719811164853 - 36.066611959079935*S - 0.8612656616290079*S2 + eta2*(170.97203068800542 - 107.41099349364234*S - 647.8103976942541*S3) + 5.95010003393006*S3 + eta3*(-1365.1499998427248 + 1152.425940764218*S + 415.7134909564443*S2 + 1897.5444343138167*S3 - 866.283566780576*S4) + 4.984750041013893*S4 + eta*(207.69898051583655 - 132.88417400679026*S - 17.671713040498304*S2 + 29.071788188638315*S3 + 37.462217031512786*S4)))/(-1.1492259468169692 + 1.*S)
    uneqSpin = dchi*delta*eta3*(7343.130973149263 - 20486.813161100774*eta + 515.9898508588834*S)
    return noSpin + eqSpin + uneqSpin


def _int_phase_v2mRDv4(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta
    S2 = S*S
    noSpin = (eta*(0.9951733419499662 + 101.21991715215253*eta + 632.4731389009143*eta2))/(0.00016803066316882238 + 0.11412314719189287*eta + 1.8413983770369362*eta2 + 1.*eta3)
    eqSpin = (S*(18.694178521101332 + 16.89845522539974*S + 4941.31613710257*eta2*S + eta*(-697.6773920613674 - 147.53381808989846*S2) + 0.3612417066833153*S2 + eta3*(3531.552143264721 - 14302.70838220423*S + 178.85850322465944*S2)))/(2.965640445745779 - 2.7706595614504725*S + 1.*S2)
    uneqSpin = dchi*delta*eta2*(356.74395864902294 + 1693.326644293169*eta2*S)
    return noSpin + eqSpin + uneqSpin


def _int_phase_v3mRDv4(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta; eta6 = eta3*eta3
    S2 = S*S
    noSpin = (eta*(-5.126358906504587 - 227.46830225846668*eta + 688.3609087244353*eta2 - 751.4184178636324*eta3))/(-0.004551938711031158 - 0.7811680872741462*eta + 1.*eta2)
    eqSpin = (S*(0.1549280856660919 - 0.9539250460041732*S - 539.4071941841604*eta2*S + eta*(73.79645135116367 - 8.13494176717772*S2) - 2.84311102369862*S2 + eta3*(-936.3740515136005 + 1862.9097047992134*S + 224.77581754671272*S2)))/(-1.5308507364054487 + 1.*S)
    uneqSpin = 2993.3598520496153*dchi*delta*eta6
    return noSpin + eqSpin + uneqSpin


def _int_phase_d43(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta
    S2 = S*S
    noSpin = (0.4248820426833804 - 906.746595921514*eta - 282820.39946006844*eta2 - 967049.2793750163*eta3 + 670077.5414916876*eta4)/(1. + 1670.9440812294847*eta + 19783.077247023448*eta2)
    eqSpin = (S*(0.22814271667259703 + 1.1366593671801855*S + eta3*(3499.432393555856 - 877.8811492839261*S - 4974.189172654984*S2) + eta*(12.840649528989287 - 61.17248283184154*S2) + 0.4818323187946999*S2 + eta2*(-711.8532052499075 + 269.9234918621958*S + 941.6974723887743*S2) + eta4*(-4939.642457025497 - 227.7672020783411*S + 8745.201037897836*S2)))/(-1.2442293719740283 + 1.*S)
    uneqSpin = dchi*delta*(-514.8494071830514 + 1493.3851099678195*eta)*eta3
    return noSpin + eqSpin + uneqSpin


def _rd_phase_v4(eta, S, dchi, delta):  # RDPhaseVersion 105, S = STotR
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta; eta5 = eta4*eta
    S2 = S*S; S3 = S2*S; S4 = S3*S
    noSpin = (-85.86062966719405 - 4616.740713893726*eta - 4925.756920247186*eta2 + 7732.064464348168*eta3 + 12828.269960300782*eta4 - 39783.51698102803*eta5)/(1. + 50.206318806624004*eta)
    eqSpin = (S*(33.335857451144356 - 36.49019206094966*S + eta3*(1497.3545918387515 - 101.72731770500685*S)*S - 3.835967351280833*S2 + 2.302712009652155*S3 + eta2*(93.64156367505917 - 18.184492163348665*S + 423.48863373726243*S2 - 104.36120236420928*S3 - 719.8775484010988*S4) + 1.6533417657003922*S4 + eta*(-69.19412903018717 + 26.580344399838758*S - 15.399770764623746*S2 + 31.231253209893488*S3 + 97.69027029734173*S4) + eta4*(1075.8686153198323 - 3443.0233614187396*S - 4253.974688619423*S2 - 608.2901586790335*S3 + 5064.173605639933*S4)))/(-1.3705601055555852 + 1.*S)
    uneqSpin = dchi*delta*eta*(22.363215261437862 + 156.08206945239374*eta)
    return noSpin + eqSpin + uneqSpin


def _rd_phase_d12(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta
    S2 = S*S; S3 = S2*S; S4 = S3*S; S5 = S4*S
    noSpin = (eta*(0.7207992174994245 - 1.237332073800276*eta + 6.086871214811216*eta2))/(0.006851189888541745 + 0.06099184229137391*eta - 0.15500218299268662*eta2 + 1.*eta3)
    eqSpin = ((0.06519048552628343 - 25.25397971063995*eta - 308.62513664956975*eta4 + 58.59408241189781*eta2 + 160.14971486043524*eta3)*S + eta*(-5.215945111216946 + 153.95945758807616*eta - 693.0504179144295*eta2 + 835.1725103648205*eta3)*S2 + (0.20035146870472367 - 0.28745205203100666*eta - 47.56042058800358*eta4)*S3 + eta*(5.7756520242745735 - 43.97332874253772*eta + 338.7263666984089*eta3)*S4 + (-0.2697933899920511 + 4.917070939324979*eta - 22.384949087140086*eta4 - 11.61488280763592*eta2)*S5)/(1. - 0.6628745847248266*S)
    uneqSpin = -23.504907495268824*dchi*delta*eta2
    return noSpin + eqSpin + uneqSpin


def _rd_phase_d24(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta; eta4 = eta3*eta
    S2 = S*S; S3 = S2*S
    noSpin = (eta*(-9.460253118496386 + 9.429314399633007*eta + 64.69109972468395*eta2))/(-0.0670554310666559 - 0.09987544893382533*eta + 1.*eta2)
    eqSpin = (17.36495157980372*eta*S + eta3*S*(930.3458437154668 + 808.457330742532*S) + eta4*S*(-774.3633787391745 - 2177.554979351284*S - 1031.846477275069*S2) + eta2*S*(-191.00932194869588 - 62.997389062600035*S + 64.42947340363101*S2) + 0.04497628581617564*S3)/(1. - 0.7267610313751913*S)
    uneqSpin = dchi*delta*(-36.66374091965371 + 91.60477826830407*eta)*eta2
    return noSpin + eqSpin + uneqSpin


def _rd_phase_d34(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta; eta5 = eta3*eta2
    S2 = S*S; S3 = S2*S; S4 = S3*S
    noSpin = (eta*(-8.506898502692536 + 13.936621412517798*eta))/(-0.40919671232073945 + 1.*eta)
    eqSpin = (eta*(1.7280582989361533*S + 18.41570325463385*S3 - 13.743271480938104*S4) + eta2*(73.8367329022058*S - 95.57802408341716*S3 + 215.78111099820157*S4) + 0.046849371468156265*S2 + eta3*S*(-27.976989112929353 + 6.404060932334562*S - 633.1966645925428*S3 + 109.04824706217418*S2))/(1. - 0.6862449113932192*S)
    uneqSpin = 641.8965762829259*dchi*delta*eta5
    return noSpin + eqSpin + uneqSpin


def _rd_phase_d54(eta, S, dchi, delta):
    eta2 = eta*eta; eta3 = eta2*eta
    noSpin = (eta*(7.05731400277692 + 22.455288821807095*eta + 119.43820622871043*eta2))/(0.26026709603623255 + 1.*eta)
    eqSpin = (eta2*(134.88158268621922 - 56.05992404859163*S)*S + eta*S*(-7.9407123129681425 + 9.486783128047414*S) + eta3*S*(-316.26970506215554 + 90.31815139272628*S))/(1. - 0.7162058321905909*S)
    uneqSpin = 43.82713604567481*dchi*delta*eta3
    return noSpin + eqSpin + uneqSpin


# ---------------------------------------------------------------------------
# Waveform struct, coefficients and ansatze
# ---------------------------------------------------------------------------

def _as_points(value, n, xp):
    arr = xp.asarray(value, dtype=xp.float64)
    if arr.ndim == 0:
        arr = xp.full((n,), float(arr))
    if arr.shape != (n,):
        raise ValueError("per-point parameters must be scalars or shape (n,)")
    return arr


def _waveform_variables(m1_in, m2_in, chi1L_in, chi2L_in, f_ref, phi0, f_min, f_max,
                        distance, xp):
    """IMRPhenomXSetWaveformVariables for default versions (masses in Msun)."""
    w = SimpleNamespace()
    swap = m1_in < m2_in
    m1 = xp.where(swap, m2_in, m1_in)
    m2 = xp.where(swap, m1_in, m2_in)
    chi1L = xp.where(swap, chi2L_in, chi1L_in)
    chi2L = xp.where(swap, chi1L_in, chi2L_in)
    delta = xp.abs((m1 - m2) / (m1 + m2))
    eta = xp.abs(0.25 * (1.0 - delta * delta))
    eta = xp.minimum(eta, 0.25)
    w.m1_SI = m1 * LAL_MSUN_SI
    w.m2_SI = m2 * LAL_MSUN_SI
    w.eta = eta
    w.Mtot_SI = w.m1_SI + w.m2_SI
    w.Mtot = m1 + m2
    w.m1 = m1 / w.Mtot
    w.m2 = m2 / w.Mtot
    w.M_sec = w.Mtot * LAL_MTSUN_SI
    w.delta = delta
    w.chi1L = chi1L
    w.chi2L = chi2L
    w.chiEff = _chi_eff(eta, chi1L, chi2L, xp)
    w.chiPNHat = _chi_pn_hat(eta, chi1L, chi2L, xp)
    w.STotR = _stot_r(eta, chi1L, chi2L, xp)
    w.dchi = chi1L - chi2L
    w.fRef = f_ref
    w.phiRef_In = phi0
    w.phi0 = phi0
    w.MfRef = w.fRef * (LAL_MTSUN_SI * w.Mtot)
    w.piM = LAL_PI * w.M_sec
    w.v_ref = xp.cbrt(w.piM * w.fRef)
    w.fCutDef = xp.where(w.chiEff > 0.99, 0.33, 0.3)
    w.fMin = f_min
    w.fMax = f_max
    w.fCut = w.fCutDef / w.M_sec
    if f_max:
        w.f_max_prime = xp.minimum(xp.full_like(w.fCut, float(f_max)), w.fCut)
    else:
        w.f_max_prime = w.fCut
    w.Mfinal = _final_mass_2017(eta, chi1L, chi2L, xp)
    w.afinal = _final_spin_2017(eta, chi1L, chi2L, xp)
    w.fRING = _qnm_fring22(w.afinal) / w.Mfinal
    w.fDAMP = _qnm_fdamp22(w.afinal) / w.Mfinal
    w.fMECO = _fmeco(eta, chi1L, chi2L, xp)
    w.fISCO = _fisco(w.afinal, xp)
    w.distance = distance
    w.amp0 = w.Mtot * LAL_MRSUN_SI * w.Mtot * LAL_MTSUN_SI / distance
    w.ampNorm = math.sqrt(2.0 / 3.0) * xp.sqrt(eta) * PI.m_one_sixth
    w.dphase0 = 5.0 / (128.0 * math.pow(LAL_PI, 5.0 / 3.0))
    return w


def _amp_coefficients(w, xp):
    a = SimpleNamespace()
    eta = w.eta; eta2 = eta * eta; eta3 = eta2 * eta
    delta = w.delta
    chi1L = w.chi1L; chi2L = w.chi2L
    chi1L2 = chi1L * chi1L; chi1L3 = chi1L2 * chi1L; chi2L2 = chi2L * chi2L
    a.gamma2 = _rd_amp_gamma2(eta, w.STotR, w.dchi, delta)
    a.gamma3 = _rd_amp_gamma3(eta, w.STotR, w.dchi, delta)
    g2, g3, frd, fda = a.gamma2, a.gamma3, w.fRING, w.fDAMP
    peak_low = xp.abs(frd + fda * g3 * (xp.sqrt(xp.maximum(1.0 - g2 * g2, 0.0)) - 1.0) / g2)
    peak_high = xp.abs(frd + fda * (-1.0) * g3 / g2)
    a.fAmpRDMin = xp.where(g2 <= 1.0, peak_low, peak_high)
    a.v1RD = _rd_amp_v1(eta, w.STotR, w.dchi, delta)
    F1 = a.fAmpRDMin
    a.gamma1 = (a.v1RD / (fda * g3)) * (F1*F1 - 2.0*F1*frd + frd*frd + fda*fda*g3*g3) \
        * xp.exp(((F1 - frd) * g2) / (fda * g3))
    a.gammaR = g2 / (fda * g3)
    a.gammaD2 = (g3 * fda) * (g3 * fda)
    a.gammaD13 = fda * a.gamma1 * g3

    a.fAmpMatchIN = w.fMECO + 0.25 * (w.fISCO - w.fMECO)
    a.pnTwoThirds = ((-969 + 1804*eta)/672.) * PI.two_thirds
    a.pnThreeThirds = ((81*(chi1L + chi2L) + 81*chi1L*delta - 81*chi2L*delta - 44*(chi1L + chi2L)*eta)/48.) * PI.itself
    a.pnFourThirds = ((-27312085 - 10287648*chi1L2*(1 + delta)
                       + 24*(428652*chi2L2*(-1 + delta) + (-1975055 + 10584*(81*chi1L2 - 94*chi1L*chi2L + 81*chi2L2))*eta
                             + 1473794*eta2))/8.128512e6) * PI.one_third * PI.itself
    a.pnFiveThirds = ((-6048*chi1L3*(-1 - delta + (3 + delta)*eta) + chi2L*(-((287213 + 6048*chi2L2)*(-1 + delta))
                                                                         + 4*(-93414 + 1512*chi2L2*(-3 + delta) + 2083*delta)*eta - 35632*eta2)
                       + chi1L*(287213*(1 + delta) - 4*eta*(93414 + 2083*delta + 8908*eta))
                       + 42840*(-1 + 4*eta)*LAL_PI)/32256.) * PI.five_thirds
    a.pnSixThirds = ((-1242641879927 + 12.0*(28.0*(-3248849057.0
                                                  + 11088*(163199*chi1L2 - 266498*chi1L*chi2L + 163199*chi2L2))*eta2
                                             + 27026893936*eta3 - 116424*(147117*(-(chi2L2*(-1.0 + delta)) + chi1L2*(1.0 + delta))
                                                                          + 60928*(chi1L + chi2L + chi1L*delta - chi2L*delta)*LAL_PI)
                                             + eta*(545384828789.0 - 77616*(638642*chi1L*chi2L + chi1L2*(-158633 + 282718*delta)
                                                                            - chi2L2*(158633.0 + 282718.0*delta) - 107520.0*(chi1L + chi2L)*LAL_PI
                                                                            + 275520*PI.two))))/6.0085960704e10) * PI.two

    V1 = _ins_amp_v2(eta, w.chiPNHat, w.dchi, delta)
    V2 = _ins_amp_v3(eta, w.chiPNHat, w.dchi, delta)
    V3 = _ins_amp_v4(eta, w.chiPNHat, w.dchi, delta)
    F1 = 0.50 * a.fAmpMatchIN
    F2 = 0.75 * a.fAmpMatchIN
    F3 = 1.00 * a.fAmpMatchIN
    a.rho1, a.rho2, a.rho3 = _ins_amp_rho(V1, V2, V3, F1, F2, F3, xp)

    F1 = a.fAmpMatchIN
    F4 = a.fAmpRDMin
    pF1 = _powers(F1, xp)
    pF4 = _powers(F4, xp)
    d1 = _ins_amp_dansatz(F1, w, a, xp)
    d4 = _rd_amp_dansatz(F4, w, a, xp)
    inspF1 = _ins_amp_ansatz(F1, pF1, a)
    rdF4 = _rd_amp_ansatz(F4, w, a, xp)
    d1 = ((7.0/6.0) * pF1.one_sixth / inspF1) - (pF1.seven_sixths * d1 / (inspF1*inspF1))
    d4 = ((7.0/6.0) * pF4.one_sixth / rdF4) - (pF4.seven_sixths * d4 / (rdF4*rdF4))
    # IntAmpVersion 104
    F2 = F1 + (1.0/2.0) * (F4 - F1)
    V1 = 1.0 / (pF1.m_seven_sixths * inspF1)
    V2 = 1.0 / _int_amp_vA(eta, w.STotR, w.dchi, delta)
    V4 = 1.0 / (pF4.m_seven_sixths * rdF4)
    a.delta0, a.delta1, a.delta2, a.delta3, a.delta4 = _int_amp_deltas_104(d1, d4, V1, V2, V4, F1, F2, F4)
    return a


def _ins_amp_ansatz(Mf, p, a):
    return (1.0
            + p.two_thirds * a.pnTwoThirds
            + Mf * a.pnThreeThirds
            + Mf * (+ p.one_third * a.pnFourThirds
                    + p.two_thirds * a.pnFiveThirds
                    + Mf * a.pnSixThirds
                    + Mf * (+ p.one_third * a.rho1
                            + p.two_thirds * a.rho2
                            + Mf * a.rho3)))


def _ins_amp_dansatz(Mf, w, a, xp):
    eta = w.eta; chi1L = w.chi1L; chi2L = w.chi2L
    rho1, rho2, rho3 = a.rho1, a.rho2, a.rho3
    chi1L2 = chi1L*chi1L; chi1L3 = chi1L2*chi1L; chi2L2 = chi2L*chi2L
    eta2 = eta*eta; eta3 = eta*eta2
    Mf2 = Mf*Mf
    LALPi = LAL_PI
    delta = w.delta
    pw = xp.power
    return (((chi2L*(81 - 81*delta - 44*eta) + chi1L*(81*(1 + delta) - 44*eta))*LALPi)/48.
            + ((-969 + 1804*eta)*math.pow(LALPi, 2./3.))/(1008.*pw(Mf, 1/3.))
            + ((-27312085 - 10287648*chi2L2 + 10287648*chi2L2*delta - 10287648*chi1L2*(1 + delta)
                + 24*(-1975055 + 857304*chi1L2 - 994896*chi1L*chi2L + 857304*chi2L2)*eta + 35371056*eta2)*math.pow(LALPi, 4./3.)*pw(Mf, 1./3.))/6.096384e6
            + (5*math.pow(LALPi, 5./3.)*(-6048*chi1L3*(-1 - delta + (3 + delta)*eta) + chi1L*(287213*(1 + delta) - 4*(93414 + 2083*delta)*eta - 35632*eta2)
                                         + chi2L*(-((287213 + 6048*chi2L2)*(-1 + delta)) + 4*(-93414 + 1512*chi2L2*(-3 + delta) + 2083*delta)*eta - 35632*eta2)
                                         + 42840*(-1 + 4*eta)*LALPi)*pw(Mf, 2./3.))/96768.
            - (math.pow(LALPi, 2.0)*(-336*(-3248849057 + 1809550512*chi1L2 - 2954929824*chi1L*chi2L + 1809550512*chi2L2)*eta2 - 324322727232*eta3
                                     + 7*(177520268561 + 29362199328*chi2L2 - 29362199328*chi2L2*delta + 29362199328*chi1L2*(1 + delta)
                                          + 12160253952*(chi1L + chi2L + chi1L*delta - chi2L*delta)*LALPi)
                                     + 12*eta*(-545384828789 + 49568837472*chi1L*chi2L - 12312458928*chi2L2 - 21943440288*chi2L2*delta
                                               + 77616*chi1L2*(-158633 + 282718*delta) - 8345272320*(chi1L + chi2L)*LALPi
                                               + 21384760320*math.pow(LALPi, 2.0)))*Mf)/3.0042980352e10
            + (7.0/3.0)*pw(Mf, 4.0/3.0)*rho1 + (8.0/3.0)*pw(Mf, 5.0/3.0)*rho2 + 3*Mf2*rho3)


def _rd_amp_ansatz(ff, w, a, xp):
    dfr = ff - w.fRING
    return xp.exp(-dfr * a.gammaR) * a.gammaD13 / (dfr*dfr + a.gammaD2)


def _rd_amp_dansatz(ff, w, a, xp):
    g1, g2, g3 = a.gamma1, a.gamma2, a.gamma3
    frd, fda = w.fRING, w.fDAMP
    dfr = ff - frd
    dfd = fda * g3
    prefactor = -xp.exp(-g2 * dfr / dfd) * g1
    numerator = (dfr*dfr*g2 + 2.0*fda*dfr*g3 + fda*fda*g2*g3*g3)
    denominator = (dfr*dfr + dfd*dfd) * (dfr*dfr + dfd*dfd)
    return prefactor * numerator / denominator


def _int_amp_ansatz(ff, p, a):
    polynomial = a.delta0 + ff*(a.delta1 + ff*(a.delta2 + ff*(a.delta3 + ff*a.delta4)))
    return p.seven_sixths / polynomial


def _solve(A, b, xp):
    return xp.linalg.solve(A, b[..., None])[..., 0]


def _phase_coefficients(w, xp):
    """IMRPhenomXGetPhaseCoefficients for (104, 105, 105) and GR."""
    c = SimpleNamespace()
    gpoints4 = (0.0, 1.0/4.0, 3.0/4.0, 1.0)
    gpoints5 = (0.0, 1.0/2 - 1.0/(2*math.sqrt(2.0)), 1.0/2.0, 1.0/2 + 1.0/(2.0*math.sqrt(2.0)), 1.0)
    frd, fda = w.fRING, w.fDAMP
    fIMmatch = 0.6 * (0.5 * frd + w.fISCO)
    fINmatch = w.fMECO
    deltaf = (fIMmatch - fINmatch) * 0.03
    c.fPhaseMatchIN = fINmatch - 1.0 * deltaf
    c.fPhaseMatchIM = fIMmatch + 0.5 * deltaf
    fPhaseInsMin = 0.0026
    fPhaseInsMax = 1.020 * w.fMECO
    fPhaseRDMin = fIMmatch
    fPhaseRDMax = frd + 1.25 * fda
    c.phiNorm = -(3.0 * PI.m_five_thirds) / (128.0)
    eta = w.eta; eta2 = eta*eta; eta3 = eta*eta2
    delta = w.delta
    chi1L = w.chi1L; chi2L = w.chi2L
    chi1L2L = chi1L*chi2L
    chi1L2 = chi1L*chi1L; chi1L3 = chi1L*chi1L2
    chi2L2 = chi2L*chi2L; chi2L3 = chi2L*chi2L2

    # Ringdown (version 105, 5 collocation points)
    deltax = fPhaseRDMax - fPhaseRDMin
    pts = [g * deltax + fPhaseRDMin for g in gpoints5]
    pts[3] = frd
    RDv4 = _rd_phase_v4(eta, w.STotR, w.dchi, delta)
    v0 = _rd_phase_d12(eta, w.STotR, w.dchi, delta)
    v1 = _rd_phase_d24(eta, w.STotR, w.dchi, delta)
    v2 = _rd_phase_d34(eta, w.STotR, w.dchi, delta)
    v3 = RDv4
    v4 = _rd_phase_d54(eta, w.STotR, w.dchi, delta)
    v4 = v4 + v3
    v2 = v2 + v3
    v1 = v1 + v3
    v0 = v0 + v1
    phaseRD = v0
    rows = []
    for ff in pts:
        invff = 1.0 / ff
        ff1 = xp.cbrt(invff)
        ff2 = invff * invff
        ff3 = ff2 * ff2
        ff4 = -(w.dphase0) / (fda*fda + (ff - frd)*(ff - frd))
        rows.append(xp.stack([xp.ones_like(ff), ff1, ff2, ff3, ff4], axis=-1))
    A = xp.stack(rows, axis=-2)
    x = _solve(A, xp.stack([v0, v1, v2, v3, v4], axis=-1), xp)
    c.c0 = x[:, 0]; c.c1 = x[:, 1]; c.c2 = x[:, 2]; c.c4 = x[:, 3]; c.cRD = x[:, 4]
    c.cL = -(w.dphase0 * c.cRD)

    # Inspiral (version 104, 4 pseudo-PN terms)
    deltax = fPhaseInsMax - fPhaseInsMin
    pts = [g * deltax + fPhaseInsMin for g in gpoints4]
    S = w.chiPNHat
    i0 = _ins_phase_d13(eta, S, w.dchi, delta)
    i1 = _ins_phase_d23(eta, S, w.dchi, delta)
    i2 = _ins_phase_v3(eta, S, w.dchi, delta)
    i3 = _ins_phase_d43(eta, S, w.dchi, delta)
    i0 = i0 + i2; i1 = i1 + i2; i3 = i3 + i2
    rows = []
    for ff in pts:
        ff1 = xp.cbrt(ff)
        rows.append(xp.stack([xp.ones_like(ff), ff1, ff1 * ff1, ff], axis=-1))
    A = xp.stack(rows, axis=-2)
    x = _solve(A, xp.stack([i0, i1, i2, i3], axis=-1), xp)
    c.a0 = x[:, 0]; c.a1 = x[:, 1]; c.a2 = x[:, 2]; c.a3 = x[:, 3]
    c.a4 = xp.zeros_like(c.a0)
    c.sigma1 = (-5.0/3.0) * c.a0
    c.sigma2 = (-5.0/4.0) * c.a1
    c.sigma3 = (-5.0/5.0) * c.a2
    c.sigma4 = (-5.0/6.0) * c.a3
    c.sigma5 = (-5.0/7.0) * c.a4

    # TaylorF2 coefficients (normalised as in LAL)
    phi0NS = 1.0
    phi2NS = (3715/756. + (55*eta)/9.) * PI.two_thirds
    phi3NS = -16.0 * PI.two
    phi3S = ((113*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 76*(chi1L + chi2L)*eta)/6.) * PI.itself
    phi4NS = (15293365/508032. + (27145*eta)/504. + (3085*eta2)/72.) * PI.four_thirds
    phi4S = ((-5*(81*chi1L2*(1 + delta - 2*eta) + 316*chi1L2L*eta - 81*chi2L2*(-1 + delta + 2*eta)))/16.) * PI.four_thirds
    phi5LNS = ((5*(46374 - 6552*eta)*LAL_PI)/4536.) * PI.five_thirds
    phi5LS = ((-732985*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 560*(-1213*(chi1L + chi2L)
                                                                          + 63*(chi1L - chi2L)*delta)*eta + 85680*(chi1L + chi2L)*eta2)/4536.) * PI.five_thirds
    phi6NS = (11583231236531/4.69421568e9 - (5*eta*(3147553127 + 588*eta*(-45633 + 102260*eta)))/3.048192e6 - (6848*LAL_GAMMA)/21.
              - (640*PI.two)/3. + (2255*eta*PI.two)/12. - (13696*math.log(2))/21. - (6848*PI.log)/63.) * PI.two
    phi6S = ((5*(227*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 156*(chi1L + chi2L)*eta)*LAL_PI)/3.) * PI.two
    phi6S = phi6S + ((5*(20*chi1L2L*eta*(11763 + 12488*eta) + 7*chi2L2*(-15103*(-1 + delta) + 2*(-21683 + 6580*delta)*eta - 9808*eta2) -
                         7*chi1L2*(-15103*(1 + delta) + 2*(21683 + 6580*delta)*eta + 9808*eta2)))/4032.) * PI.two
    phi6LNS = (-6848/63.) * PI.two
    phi7NS = ((5*(15419335 + 168*(75703 - 29618*eta)*eta)*LAL_PI)/254016.) * PI.seven_thirds
    phi7S = ((5*(-5030016755*(chi1L + chi2L + chi1L*delta - chi2L*delta) + 4*(2113331119*(chi1L + chi2L) + 675484362*(chi1L - chi2L)*delta)*eta - 1008*(208433*(chi1L + chi2L) + 25011*(chi1L - chi2L)*delta)*eta2 + 90514368*(chi1L + chi2L)*eta3))/6.096384e6) * PI.seven_thirds
    phi7S = phi7S + (-5*(57*chi1L2*(1 + delta - 2*eta) + 220*chi1L2L*eta - 57*chi2L2*(-1 + delta + 2*eta))*LAL_PI) * PI.seven_thirds
    phi7S = phi7S + ((14585*(-(chi2L3*(-1 + delta)) + chi1L3*(1 + delta)) - 5*(chi2L3*(8819 - 2985*delta) + 8439*chi1L*chi2L2*(-1 + delta) - 8439*chi1L2*chi2L*(1 + delta) + chi1L3*(8819 + 2985*delta))*eta + 40*(chi1L + chi2L)*(17*chi1L2 - 14*chi1L2L + 17*chi2L2)*eta2)/48.) * PI.seven_thirds
    phi8S = ((-5*(1263141*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 2*(794075*(chi1L + chi2L) + 178533*(chi1L - chi2L)*delta)*eta + 94344*(chi1L + chi2L)*eta2)*LAL_PI*(-1 + PI.log))/9072.) * PI.eight_thirds
    phi8LS = ((-5*(1263141*(chi1L + chi2L + chi1L*delta - chi2L*delta) - 2*(794075*(chi1L + chi2L) + 178533*(chi1L - chi2L)*delta)*eta
               + 94344*(chi1L + chi2L)*eta2)*LAL_PI)/9072.) * PI.eight_thirds
    c.phi0 = phi0NS
    c.phi2 = phi2NS
    c.phi3 = phi3NS + phi3S
    c.phi4 = phi4NS + phi4S
    c.phi5L = phi5LNS + phi5LS
    c.phi6 = phi6NS + phi6S
    c.phi6L = phi6LNS
    c.phi7 = phi7NS + phi7S
    c.phi8 = phi8S
    c.phi8L = phi8LS
    c.dphi0 = c.phi0
    c.dphi2 = 3.0 / 5.0 * c.phi2
    c.dphi3 = 2.0 / 5.0 * c.phi3
    c.dphi4 = 1.0 / 5.0 * c.phi4
    c.dphi5 = -3.0 / 5.0 * c.phi5L
    c.dphi6 = -1.0 / 5.0 * c.phi6 - 3.0 / 5.0 * c.phi6L
    c.dphi6L = -1.0 / 5.0 * c.phi6L
    c.dphi7 = -2.0 / 5.0 * c.phi7
    c.dphi8 = -3.0 / 5.0 * c.phi8 - 3.0 / 5.0 * c.phi8L
    c.dphi8L = -3.0 / 5.0 * c.phi8L

    # Phase derivative at fmatchIN: intermediate collocation value
    p = _powers(c.fPhaseMatchIN, xp)
    phaseIN = c.dphi0 + c.dphi2 * p.two_thirds + c.dphi3 * p.itself + c.dphi4 * p.four_thirds \
        + c.dphi5 * p.five_thirds + c.dphi6 * p.two + c.dphi6L * p.two * p.log \
        + c.dphi7 * p.seven_thirds + c.dphi8 * p.eight_thirds + c.dphi8L * p.eight_thirds * p.log
    phaseIN = phaseIN + (c.a0 * p.eight_thirds + c.a1 * p.three
                         + c.a2 * p.eight_thirds * p.two_thirds
                         + c.a3 * p.eight_thirds * p.itself
                         + c.a4 * p.eight_thirds * p.four_thirds)
    phaseIN = phaseIN * p.m_eight_thirds * w.dphase0

    # Intermediate (version 105, 5 collocation points)
    deltax = c.fPhaseMatchIM - c.fPhaseMatchIN
    pts = [g * deltax + c.fPhaseMatchIN for g in gpoints5]
    St = w.STotR
    v2IMmRDv4 = _int_phase_v2mRDv4(eta, St, w.dchi, delta)
    v3IMmRDv4 = _int_phase_v3mRDv4(eta, St, w.dchi, delta)
    v2IM = _int_phase_v2(eta, St, w.dchi, delta)
    b0v = phaseIN
    b1v = 0.75*(v2IMmRDv4 + RDv4) + 0.25*v2IM
    b2v = v3IMmRDv4 + RDv4
    b3v = _int_phase_d43(eta, St, w.dchi, delta) + b2v
    b4v = phaseRD
    rows = []
    rhs = []
    for ff, val in zip(pts, (b0v, b1v, b2v, b3v, b4v)):
        ff1 = frd / ff
        ff2 = ff1 * ff1
        ff3 = ff1 * ff2
        ff4 = ff2 * ff2
        ff0 = (4.0 * c.cL) / ((2.0*fda)*(2.0*fda) + (ff - frd)*(ff - frd))
        rows.append(xp.stack([xp.ones_like(ff), ff1, ff2, ff3, ff4], axis=-1))
        rhs.append(val - ff0)
    A = xp.stack(rows, axis=-2)
    x = _solve(A, xp.stack(rhs, axis=-1), xp)
    c.b0 = x[:, 0]
    c.b1 = x[:, 1] * frd
    c.b2 = x[:, 2] * frd * frd
    c.b3 = x[:, 3] * frd * frd * frd
    c.b4 = x[:, 4] * frd * frd * frd * frd
    c.cLGR = c.cL
    c.c4ov3 = c.c4 / 3.0
    c.cLovfda = c.cL / fda
    return c


def _ins_phase_ansatz(Mf, p, c):
    phase = c.dphi0 + c.dphi2 * p.two_thirds + c.dphi3 * Mf + c.dphi4 * p.four_thirds \
        + c.dphi5 * p.five_thirds + c.dphi6 * p.two + c.dphi6L * p.two * p.log \
        + c.dphi7 * p.seven_thirds + c.dphi8 * p.eight_thirds + c.dphi8L * p.eight_thirds * p.log
    phase = phase + (c.a0 * p.eight_thirds + c.a1 * p.three
                     + c.a2 * p.eight_thirds * p.two_thirds
                     + c.a3 * p.eight_thirds * p.itself
                     + c.a4 * p.eight_thirds * p.four_thirds)
    return phase * p.m_eight_thirds * (5.0 / (128.0 * PI.five_thirds))


def _ins_phase_ansatz_int(Mf, p, c):
    phasing = c.phi0 + c.phi2 * p.two_thirds + c.phi3 * Mf + c.phi4 * p.four_thirds \
        + c.phi5L * p.five_thirds * p.log + c.phi6 * p.two + c.phi6L * p.two * p.log \
        + c.phi7 * p.seven_thirds + c.phi8 * p.eight_thirds + c.phi8L * p.eight_thirds * p.log
    phasing = phasing + (c.sigma1 * p.eight_thirds + c.sigma2 * p.three
                         + c.sigma3 * p.one_third * p.three
                         + c.sigma4 * p.two_thirds * p.three
                         + c.sigma5 * p.itself * p.three)
    return phasing * c.phiNorm * p.m_five_thirds


def _int_phase_ansatz(ff, p, w, c):
    lorentz = (4.0 * c.cLGR) / ((4.0*w.fDAMP*w.fDAMP) + (ff - w.fRING)*(ff - w.fRING))
    return c.b0 + c.b1*p.m_one + c.b2*p.m_two + c.b3*p.m_three + c.b4*p.m_four + lorentz


def _int_phase_ansatz_int(f, p, w, c, xp):
    return (c.b0*f + c.b1*p.log - c.b2*p.m_one - c.b3*p.m_two/2.0 - (c.b4*p.m_three/3.0)
            + (2.0 * c.cLGR * xp.arctan((f - w.fRING) / (2.0 * w.fDAMP))) / w.fDAMP)


def _rd_phase_ansatz(ff, p, w, c):
    return (c.c0 + c.c1*p.m_one_third + c.c2*p.m_two + c.c4*p.m_four
            + (c.cL / (w.fDAMP*w.fDAMP + (ff - w.fRING)*(ff - w.fRING))))


def _rd_phase_ansatz_int(ff, p, w, c, xp):
    return (c.c0*ff + 1.5*c.c1*p.two_thirds - c.c2*p.m_one - c.c4ov3*p.m_three
            + (c.cLovfda * xp.arctan((ff - w.fRING)/w.fDAMP)))


def _connection_and_shift(w, c, xp):
    """Connection coefficients, linb time shift and phifRef (uses w.phi0)."""
    fIns = c.fPhaseMatchIN
    fInt = c.fPhaseMatchIM
    p = _powers(fIns, xp)
    c.C2Int = _ins_phase_ansatz(fIns, p, c) - _int_phase_ansatz(fIns, p, w, c)
    c.C1Int = _ins_phase_ansatz_int(fIns, p, c) - _int_phase_ansatz_int(fIns, p, w, c, xp) - c.C2Int * fIns
    p = _powers(fInt, xp)
    phiIMC = _int_phase_ansatz_int(fInt, p, w, c, xp) + c.C1Int + c.C2Int * fInt
    phiRD = _rd_phase_ansatz_int(fInt, p, w, c, xp)
    DPhiIntC = _int_phase_ansatz(fInt, p, w, c) + c.C2Int
    DPhiRD = _rd_phase_ansatz(fInt, p, w, c)
    c.C2MRD = DPhiIntC - DPhiRD
    c.C1MRD = phiIMC - phiRD - c.C2MRD * fInt

    linb = _linb(w.eta, w.STotR, w.dchi, w.delta)
    frefFit = w.fRING - w.fDAMP
    dphi22Ref = 1.0 / w.eta * _dphase_22(frefFit, w, c, xp)
    linb = linb - dphi22Ref - 2.*LAL_PI*(500 + _psi4_to_strain(w.eta, w.STotR, w.dchi, xp))
    c.linb = linb
    phi_ref_model = _phase_22(w.MfRef, w, c, xp)
    c.phifRef = -(1.0 / w.eta * phi_ref_model + linb * w.MfRef) + 2.0 * w.phi0 + LAL_PI_4
    return c


def _phase_22(ff, w, c, xp):
    """IMRPhenomX_Phase_22 (region selection with >= at the boundaries)."""
    p = _powers(ff, xp)
    ins = _ins_phase_ansatz_int(ff, p, c)
    mrd = _rd_phase_ansatz_int(ff, p, w, c, xp) + c.C1MRD + c.C2MRD * ff
    inter = _int_phase_ansatz_int(ff, p, w, c, xp) + c.C1Int + c.C2Int * ff
    return xp.where(ff < c.fPhaseMatchIN, ins, xp.where(ff >= c.fPhaseMatchIM, mrd, inter))


def _dphase_22(ff, w, c, xp):
    p = _powers(ff, xp)
    ins = _ins_phase_ansatz(ff, p, c)
    mrd = _rd_phase_ansatz(ff, p, w, c) + c.C2MRD
    inter = _int_phase_ansatz(ff, p, w, c) + c.C2Int
    return xp.where(ff < c.fPhaseMatchIN, ins, xp.where(ff >= c.fPhaseMatchIM, mrd, inter))


def _expand(c, attr_names):
    """View per-point coefficient arrays as (n,1) for broadcasting on frequency."""
    out = SimpleNamespace(**vars(c))
    for name in attr_names:
        v = getattr(c, name)
        if hasattr(v, "ndim") and v.ndim == 1:
            setattr(out, name, v[:, None])
    return out


def _xas_22_on_grid(Mf, w, a, c, xp, p=None):
    """XAS (2,2) h(f) at geometric frequencies Mf (shape (n, nf)), unmasked.

    ``p`` optionally supplies precomputed powers of Mf (see gpu_waveform_xhm)."""
    w2 = _expand(w, list(vars(w)))
    a2 = _expand(a, list(vars(a)))
    c2 = _expand(c, list(vars(c)))
    if p is None:
        p = _powers(Mf, xp)
    phi_ins = _ins_phase_ansatz_int(Mf, p, c2)
    phi_rd = _rd_phase_ansatz_int(Mf, p, w2, c2, xp) + c2.C1MRD + c2.C2MRD * Mf
    phi_int = _int_phase_ansatz_int(Mf, p, w2, c2, xp) + c2.C1Int + c2.C2Int * Mf
    phi = xp.where(Mf < c2.fPhaseMatchIN, phi_ins, xp.where(Mf > c2.fPhaseMatchIM, phi_rd, phi_int))
    phi = phi / w2.eta + c2.linb * Mf + c2.phifRef
    amp_ins = _ins_amp_ansatz(Mf, p, a2)
    amp_rd = _rd_amp_ansatz(Mf, w2, a2, xp)
    amp_int = _int_amp_ansatz(Mf, p, a2)
    amp = xp.where(Mf < a2.fAmpMatchIN, amp_ins, xp.where(Mf > a2.fAmpRDMin, amp_rd, amp_int))
    amp0 = (w2.amp0 * w2.ampNorm)
    return amp0 * p.m_seven_sixths * amp * xp.exp(1j * phi)


def _frequency_mask(freqs, w, f_min, delta_f, xp):
    k = xp.arange(freqs.shape[-1])[None, :]
    i_start = int(f_min / delta_f)
    i_stop = xp.floor(w.f_max_prime / delta_f)[:, None]
    Mf = w.M_sec[:, None] * freqs[None, :]
    return (k >= i_start) & (k <= i_stop) & (Mf <= (w.f_max_prime * w.M_sec)[:, None])


def _grid(delta_f, n_freq, f_min, xp):
    freqs = xp.arange(n_freq, dtype=xp.float64) * float(delta_f)
    # Avoid evaluating the ansatze at f=0 (masked anyway).
    safe = xp.where(freqs > 0, freqs, float(delta_f))
    return freqs, safe


def imrphenomxas_h22(m1, m2, chi1z, chi2z, distance, f_min, f_ref, phi_ref,
                     delta_f, n_freq, f_max=0.0, xp=None):
    """IMRPhenomXAS h22(f) for n aligned-spin points, shape (n, n_freq).

    Matches ``lalsim.SimIMRPhenomXASGenerateFD`` on the grid ``k*delta_f``.
    Masses in Msun, distance in metres.  ``f_ref=0`` means ``f_min``.
    """
    xp = _module(xp)
    hp = np
    m1 = np.atleast_1d(_host(m1))
    n = m1.shape[0]
    m2 = _as_points(_host(m2), n, hp)
    chi1z = _as_points(_host(chi1z), n, hp)
    chi2z = _as_points(_host(chi2z), n, hp)
    distance = _as_points(_host(distance), n, hp)
    phi_ref = _as_points(_host(phi_ref), n, hp)
    fref = float(f_ref) if f_ref else float(f_min)
    w = _waveform_variables(m1, m2, chi1z, chi2z, fref, phi_ref, float(f_min), float(f_max), distance, hp)
    a = _amp_coefficients(w, hp)
    c = _phase_coefficients(w, hp)
    c = _connection_and_shift(w, c, hp)
    w, a, c = _to_device(w, xp), _to_device(a, xp), _to_device(c, xp)
    freqs, safe = _grid(delta_f, n_freq, f_min, xp)
    Mf = w.M_sec[:, None] * safe[None, :]
    h = _xas_22_on_grid(Mf, w, a, c, xp)
    return xp.where(_frequency_mask(freqs, w, float(f_min), float(delta_f), xp), h, 0.0)


# ---------------------------------------------------------------------------
# Precession: NNLO angles (PrecVersion 102, Convention 1)
# ---------------------------------------------------------------------------

def _rotate_z(angle, x, y, z, xp):
    ca, sa = xp.cos(angle), xp.sin(angle)
    return x*ca - y*sa, x*sa + y*ca, z


def _rotate_y(angle, x, y, z, xp):
    ca, sa = xp.cos(angle), xp.sin(angle)
    return x*ca + z*sa, y, -x*sa + z*ca


def _atan2tol(a, b, tol, xp):
    return xp.where((xp.abs(a) < tol) & (xp.abs(b) < tol), 0.0, xp.arctan2(a, b))


def _lpn(v, eta, P):
    x = v*v
    sqx = v  # LAL evaluates sqrt(v*v); equal for v > 0
    x2 = x*x; x3 = x*x2
    LNorm = eta / v
    return LNorm * (P.L0 + P.L2*x + P.L3*(x*sqx) + P.L4*x2 + P.L5*(x2*sqx) + P.L6*x3)


def _precession_setup(w, m1_msun, m2_msun, chi1, chi2, inclination, phi_ref, xp):
    """IMRPhenomXGetAndSetPrecessionVariables for PrecVersion 102, Convention 1.

    ``chi1``/``chi2`` have shape (n, 3) and already follow LAL's strict
    ``m1 > m2`` ordering.  Updates ``w`` (final spin, QNM frequencies, phi0).
    """
    P = SimpleNamespace()
    Mtot = m1_msun + m2_msun
    m1 = m1_msun / Mtot
    m2 = m2_msun / Mtot
    m1_2 = m1*m1; m1_3 = m1*m1_2; m1_4 = m1*m1_3; m1_5 = m1*m1_4; m1_6 = m1*m1_5; m1_8 = m1_6*m1_2
    m2_2 = m2*m2
    q = m1 / m2
    eta = w.eta; eta2 = eta*eta; eta3 = eta*eta2; eta4 = eta*eta3; eta6 = eta3*eta3
    delta = w.delta; delta2 = delta*delta; delta3 = delta*delta2
    chi_eff = w.chiEff
    c1x, c1y, c1z = chi1[:, 0], chi1[:, 1], chi1[:, 2]
    c2x, c2y, c2z = chi2[:, 0], chi2[:, 1], chi2[:, 2]
    S1_perp = m1_2 * xp.sqrt(c1x*c1x + c1y*c1y)
    S2_perp = m2_2 * xp.sqrt(c2x*c2x + c2y*c2y)
    A1 = 2.0 + (3.0 * m2) / (2.0 * m1)
    A2 = 2.0 + (3.0 * m1) / (2.0 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp
    num = xp.where(ASp2 > ASp1, ASp2, ASp1)
    den = xp.where(m2 > m1, A2*m2_2, A1*m1_2)
    chip = num / den
    P.chi_p = chip
    chi1L, chi2L = c1z, c2z
    P.SL = chi1L*m1_2 + chi2L*m2_2
    P.Sperp = chip * m1_2

    # Final spin: FinalSpinMod 4 -> 3 -> (not MSA) version 0
    afinal = _precessing_final_spin_2017(eta, chi1L, chi2L, chip, xp)
    afinal = xp.where(xp.abs(afinal) > 1.0, xp.copysign(1.0, afinal), afinal)
    w.afinal = afinal
    w.fRING = _qnm_fring22(afinal) / w.Mfinal
    w.fDAMP = _qnm_fdamp22(afinal) / w.Mfinal

    # 3PN orbital angular momentum (PrecVersion 102)
    P.L0 = 1.0
    P.L2 = 3.0/2. + eta/6.0
    P.L3 = (5*(chi1L*(-2 - 2*delta + eta) + chi2L*(-2 + 2*delta + eta)))/6.
    P.L4 = (81 + (-57 + eta)*eta)/24.
    P.L5 = (-7*(chi1L*(72 + delta*(72 - 31*eta) + eta*(-121 + 2*eta)) + chi2L*(72 + eta*(-121 + 2*eta) + delta*(-72 + 31*eta))))/144.
    P.L6 = (10935 + eta*(-62001 + eta*(1674 + 7*eta) + 2214*PI.two))/1296.
    LRef = _lpn(w.v_ref, eta, P)  # M = 1

    J0x = m1_2*c1x + m2_2*c2x
    J0y = m1_2*c1y + m2_2*c2y
    J0z = m1_2*c1z + m2_2*c2z + LRef
    J0 = xp.sqrt(J0x*J0x + J0y*J0y + J0z*J0z)
    thetaJ = xp.where(J0 < 1e-10, 0.0, xp.arccos(J0z / J0))
    aligned = (xp.abs(J0x) < MAX_TOL_ATAN) & (xp.abs(J0y) < MAX_TOL_ATAN)
    phiJ = xp.where(aligned, 0.0, xp.arctan2(J0y, J0x))
    w.phi0 = xp.zeros_like(w.phi0)  # Convention 1

    incl = inclination
    Nx = xp.sin(incl)*xp.cos(LAL_PI_2 - phi_ref)
    Ny = xp.sin(incl)*xp.sin(LAL_PI_2 - phi_ref)
    Nz = xp.cos(incl)
    tx, ty, tz = _rotate_z(-phiJ, Nx, Ny, Nz, xp)
    tx, ty, tz = _rotate_y(-thetaJ, tx, ty, tz, xp)
    kappa = _atan2tol(ty, tx, MAX_TOL_ATAN, xp)
    P.alpha0 = LAL_PI - kappa
    J0dotN = J0x*Nx + J0y*Ny + J0z*Nz
    P.thetaJN = xp.arccos(J0dotN / J0)
    Nz_Jf = xp.cos(P.thetaJN)
    Nx_Jf = xp.sin(P.thetaJN)
    Xx = -xp.cos(incl)*xp.sin(phi_ref)
    Xy = -xp.cos(incl)*xp.cos(phi_ref)
    Xz = xp.sin(incl)
    tx, ty, tz = _rotate_z(-phiJ, Xx, Xy, Xz, xp)
    tx, ty, tz = _rotate_y(-thetaJ, tx, ty, tz, xp)
    tx, ty, tz = _rotate_z(-kappa, tx, ty, tz, xp)
    XdotP = tx*Nz_Jf + tz*(-Nx_Jf)
    XdotQ = ty
    P.zeta_polarization = xp.arctan2(XdotQ, XdotP)

    # NNLO PN Euler angle coefficients
    chiL = (1.0 + q) * (chi_eff / q)
    chiL2 = chiL * chiL
    chip2 = chip * chip
    P.alpha1 = -35/192. + (5*delta)/(64.*m1)
    P.alpha2 = ((15*chiL*delta*m1)/128. - (35*chiL*m1_2)/128.)/eta
    P.alpha3 = (-5515/3072. + eta*(-515/384. - (15*delta2)/(256.*m1_2)
                                   + (175*delta)/(256.*m1)) + (4555*delta)/(7168.*m1)
                + ((15*chip2*delta*m1_3)/128. - (35*chip2*m1_4)/128.)/eta2)
    P.alpha4L = ((5*chiL*delta2)/16. - (5*chiL*delta*m1)/3. + (2545*chiL*m1_2)/1152.
                 + ((-2035*chiL*delta*m1)/21504.
                    + (2995*chiL*m1_2)/9216.)/eta + ((5*chiL*chip2*delta*m1_5)/128.
                                                     - (35*chiL*chip2*m1_6)/384.)/eta3
                 - (35*LAL_PI)/48. + (5*delta*LAL_PI)/(16.*m1))
    P.alpha5 = (5*(-190512*delta3*eta6 + 2268*delta2*eta3*m1*(eta2*(323 + 784*eta)
                                                              + 336*(25*chiL2 + chip2)*m1_4) + 7*m1_3*(8024297*eta4 + 857412*eta**5
                                                                                                       + 3080448*eta6 + 143640*chip2*eta2*m1_4
                                                                                                       - 127008*chip2*(-4*chiL2 + chip2)*m1_8
                                                                                                       + 6048*eta3*((2632*chiL2 + 115*chip2)*m1_4 - 672*chiL*m1_2*LAL_PI))
                + 3*delta*m1_2*(-5579177*eta4 + 80136*eta**5 - 3845520*eta6
                                + 146664*chip2*eta2*m1_4 + 127008*chip2*(-4*chiL2 + chip2)*m1_8
                                - 42336*eta3*((726*chiL2 + 29*chip2)*m1_4
                                              - 96*chiL*m1_2*LAL_PI))))/(6.5028096e7*eta4*m1_3)
    P.epsilon1 = -35/192. + (5*delta)/(64.*m1)
    P.epsilon2 = ((15*chiL*delta*m1)/128. - (35*chiL*m1_2)/128.)/eta
    P.epsilon3 = -5515/3072. + eta*(-515/384. - (15*delta2)/(256.*m1_2)
                                    + (175*delta)/(256.*m1)) + (4555*delta)/(7168.*m1)
    P.epsilon4L = ((5*chiL*delta2)/16. - (5*chiL*delta*m1)/3. + (2545*chiL*m1_2)/1152.
                   + ((-2035*chiL*delta*m1)/21504. + (2995*chiL*m1_2)/9216.)/eta - (35*LAL_PI)/48.
                   + (5*delta*LAL_PI)/(16.*m1))
    P.epsilon5 = (5*(-190512*delta3*eta3 + 2268*delta2*m1*(eta2*(323 + 784*eta)
                                                           + 8400*chiL2*m1_4)
                     - 3*delta*m1_2*(eta*(5579177 + 504*eta*(-159 + 7630*eta))
                                     + 254016*chiL*m1_2*(121*chiL*m1_2 - 16*LAL_PI))
                     + 7*m1_3*(eta*(8024297 + 36*eta*(23817 + 85568*eta))
                               + 338688*chiL*m1_2*(47*chiL*m1_2 - 12*LAL_PI))))/(6.5028096e7*eta*m1_3)
    P.epsilon0 = phiJ - LAL_PI  # Convention 1
    omega_ref = w.piM * w.fRef
    a_ref, e_ref = _nnlo_alpha_epsilon(omega_ref, P, xp)
    P.alpha_offset = a_ref - P.alpha0
    P.epsilon_offset = e_ref - P.epsilon0
    return P


def _nnlo_alpha_epsilon(omega, P, xp):
    logomega = xp.log(omega)
    omega_cbrt = xp.cbrt(omega)
    omega_cbrt2 = omega_cbrt * omega_cbrt
    alpha = (P.alpha1 / omega + P.alpha2 / omega_cbrt2 + P.alpha3 / omega_cbrt
             + P.alpha4L * logomega + P.alpha5 * omega_cbrt)
    eps = (P.epsilon1 / omega + P.epsilon2 / omega_cbrt2 + P.epsilon3 / omega_cbrt
           + P.epsilon4L * logomega + P.epsilon5 * omega_cbrt)
    return alpha, eps


def _y2m_real(theta, xp):
    """-2Y_2m(theta, 0) for m=-2..2 (real), stacked on the last axis."""
    c = xp.cos(theta); s = xp.sin(theta)
    f1 = math.sqrt(5.0 / (64.0 * LAL_PI))
    f2 = math.sqrt(5.0 / (16.0 * LAL_PI))
    f3 = math.sqrt(15.0 / (32.0 * LAL_PI))
    return xp.stack([f1*(1.0 - c)**2, f2*s*(1.0 - c), f3*s*s, f2*s*(1.0 + c), f1*(1.0 + c)**2], axis=-1)


def _xp_setup(m1, m2, chi1, chi2, distance, inclination, phi_ref, f_min, f_ref, f_max, xp_out):
    """Per-point setup.  Runs on the host in NumPy (O(n) scalar work and
    5x5 solves, which cost more in kernel launches than in arithmetic on a
    GPU) and returns the coefficient namespaces on ``xp_out``."""
    xp = np
    m1 = np.atleast_1d(_host(m1))
    n = m1.shape[0]
    m2 = _as_points(_host(m2), n, xp)
    chi1 = _host(chi1).reshape(n, 3)
    chi2 = _host(chi2).reshape(n, 3)
    distance = _as_points(_host(distance), n, xp)
    inclination = _as_points(_host(inclination), n, xp)
    phi_ref = _as_points(_host(phi_ref), n, xp)
    fref = float(f_ref) if f_ref else float(f_min)
    # XLALIMRPhenomXPCheckMassesAndSpins: swap unless m1 > m2 (strict)
    swap = ~(m1 > m2)
    m1p = xp.where(swap, m2, m1)
    m2p = xp.where(swap, m1, m2)
    chi1p = xp.where(swap[:, None], chi2, chi1)
    chi2p = xp.where(swap[:, None], chi1, chi2)
    # SetWaveformVariables sees the swapped masses (as in XLALSimIMRPhenomXPHMOneMode)
    w = _waveform_variables(m1p, m2p, chi1p[:, 2], chi2p[:, 2], fref, phi_ref,
                            float(f_min), float(f_max), distance, xp)
    P = _precession_setup(w, m1p, m2p, chi1p, chi2p, inclination, phi_ref, xp)
    a = _amp_coefficients(w, xp)
    c = _phase_coefficients(w, xp)
    c = _connection_and_shift(w, c, xp)
    return tuple(_to_device(ns, xp_out) for ns in (w, P, a, c))


def _twist_quantities(w, P, Mf, xp):
    """NNLO alpha, epsilon and cos/sin(beta/2) on the grid (mprime = 2)."""
    P2 = _expand(P, list(vars(P)))
    omega = LAL_PI * Mf
    alpha, eps = _nnlo_alpha_epsilon(omega, P2, xp)
    alpha = alpha - P2.alpha_offset
    eps = eps - P2.epsilon_offset
    v = xp.cbrt(omega)
    L = _lpn(v, w.eta[:, None], P2)
    s = P2.Sperp / (L + P2.SL)
    cos_beta = xp.copysign(1.0, L + P2.SL) / xp.sqrt(1.0 + s*s)
    cbh = xp.sqrt(xp.abs(1.0 + cos_beta) / 2.0)
    sbh = xp.sqrt(xp.abs(1.0 - cos_beta) / 2.0)
    return alpha, eps, cbh, sbh


def _d22_rows(cbh, sbh):
    """[d^2_{m,2}](m=-2..2) and [d^2_{m,-2}] as lists of arrays (LAL ordering)."""
    c2 = cbh*cbh; c3 = c2*cbh; c4 = c3*cbh
    s2 = sbh*sbh; s3 = s2*sbh; s4 = s3*sbh
    sqrt6 = 2.44948974278317788
    d22 = [s4, 2.0*cbh*s3, sqrt6*s2*c2, 2.0*c3*sbh, c4]
    d2m2 = [d22[4], -d22[3], d22[2], -d22[1], d22[0]]
    return d22, d2m2


def imrphenomxp_nnlo_modes(m1, m2, chi1, chi2, distance, phi_ref, f_min, f_ref,
                           delta_f, n_freq, f_max=0.0, inclination=0.0, xp=None,
                           two_sided=True, return_metadata=False):
    """Inertial J-frame (2,m) modes of IMRPhenomXP-NNLO for n points.

    Equals LAL ``SimInspiralChooseFDModes`` for ``IMRPhenomXPHM`` with
    ``PrecVersion=102``, a co-precessing ModeArray of (2,+-2) only and
    ``ThresholdMband=0`` (no multibanding).  ``chi1``/``chi2`` have shape
    (n, 3) in the L0 (source) frame; masses in Msun; distance in metres.
    Inclination only enters the polarization angle and thetaJN, not the modes.

    Returns modes of shape (n, 5, 2*n_freq-1) on LAL's ascending two-sided
    grid ``f = (k - n_freq + 1) * delta_f`` when ``two_sided`` (the default),
    else the pair (h_pos, h_neg), each (n, 5, n_freq), where ``h_neg[..., k]``
    is the mode at ``f = -k delta_f``.  Mode axis order is m = -2..2.
    """
    xp = _module(xp)
    w, P, a, c = _xp_setup(m1, m2, chi1, chi2, distance, inclination, phi_ref,
                           f_min, f_ref, f_max, xp)
    freqs, safe = _grid(delta_f, n_freq, f_min, xp)
    Mf = w.M_sec[:, None] * safe[None, :]
    mask = _frequency_mask(freqs, w, float(f_min), float(delta_f), xp)
    h22 = xp.where(mask, _xas_22_on_grid(Mf, w, a, c, xp), 0.0)
    alpha, eps, cbh, sbh = _twist_quantities(w, P, Mf, xp)
    d22, d2m2 = _d22_rows(cbh, sbh)
    e_eps = xp.exp(-2j * eps)
    pos_pref = e_eps * h22
    neg_pref = xp.conj(e_eps) * xp.conj(h22)
    hpos = []
    hneg = []
    for i, m in enumerate(range(-2, 3)):
        ema = xp.exp(-1j * m * alpha)
        hpos.append(pos_pref * ema * d2m2[i])
        hneg.append(neg_pref * ema * d22[i])
    hpos = xp.stack(hpos, axis=1)
    hneg = xp.stack(hneg, axis=1)
    out = pack_two_sided(hpos, hneg, xp) if two_sided else (hpos, hneg)
    if return_metadata:
        meta = {"thetaJN": P.thetaJN, "zeta_polarization": P.zeta_polarization,
                "alpha0": P.alpha0, "chi_p": P.chi_p, "afinal": w.afinal,
                "f_max_prime": w.f_max_prime}
        return out, meta
    return out


def pack_two_sided(hpos, hneg, xp=None):
    """Assemble LAL's ascending two-sided grid from positive/negative halves."""
    xp = _module(xp)
    return xp.concatenate([hneg[..., :0:-1], hpos], axis=-1)


def imrphenomxp_nnlo_polarizations(m1, m2, chi1, chi2, distance, inclination, phi_ref,
                                   f_min, f_ref, delta_f, n_freq, f_max=0.0, xp=None):
    """h+(f), hx(f) of IMRPhenomXP (PrecVersion 102), each (n, n_freq).

    Matches ``SimInspiralChooseFDWaveform(..., IMRPhenomXP)`` with
    ``PrecVersion=102`` on the grid ``k*delta_f`` (one-sided).
    """
    xp = _module(xp)
    w, P, a, c = _xp_setup(m1, m2, chi1, chi2, distance, inclination, phi_ref,
                           f_min, f_ref, f_max, xp)
    freqs, safe = _grid(delta_f, n_freq, f_min, xp)
    Mf = w.M_sec[:, None] * safe[None, :]
    mask = _frequency_mask(freqs, w, float(f_min), float(delta_f), xp)
    hAS = xp.where(mask, _xas_22_on_grid(Mf, w, a, c, xp), 0.0)
    alpha, eps, cbh, sbh = _twist_quantities(w, P, Mf, xp)
    d22, d2m2 = _d22_rows(cbh, sbh)
    Y = _y2m_real(P.thetaJN, xp)  # (n, 5)
    hp_sum = 0.0
    hc_sum = 0.0
    for i, m in enumerate(range(-2, 3)):
        y = Y[:, i][:, None]
        # LAL indexes cexp_im_alpha_l2[-m+2] = exp(-i m alpha) here; the
        # inline comments in IMRPhenomXPTwistUp22 state the opposite sign.
        A2m2emm = xp.exp(-1j * m * alpha) * d2m2[i] * y
        A22emmstar = xp.exp(1j * m * alpha) * d22[i] * y
        hp_sum = hp_sum + A2m2emm + A22emmstar
        hc_sum = hc_sum + 1j * (A2m2emm - A22emmstar)
    pref = xp.exp(-2.0j * eps) * hAS / 2.0
    hp = pref * hp_sum
    hc = pref * hc_sum
    zeta = P.zeta_polarization[:, None]
    cz = xp.cos(2.0 * zeta)
    sz = xp.sin(2.0 * zeta)
    hp_out = cz * hp + sz * hc
    hc_out = cz * hc - sz * hp
    return hp_out, hc_out


# ---------------------------------------------------------------------------
# RIFT frame convention (fd_L_frame=True)
# ---------------------------------------------------------------------------

def wigner_D_l2(alpha, beta, gamma, xp=None):
    """LAL ``XLALWignerDMatrix(2, mp, m, alpha, beta, gamma)`` as (n, 5, 5).

    Index order ``[..., mp+2, m+2]``; ``D = exp(-i mp alpha) d(beta) exp(-i m gamma)``.
    """
    xp = _module(xp)
    alpha = xp.atleast_1d(xp.asarray(alpha, dtype=xp.float64))
    beta = xp.atleast_1d(xp.asarray(beta, dtype=xp.float64))
    gamma = xp.atleast_1d(xp.asarray(gamma, dtype=xp.float64))
    cb = xp.cos(beta / 2.0)
    sb = xp.sin(beta / 2.0)
    fact = math.factorial
    j = 2
    rows = []
    for mp in range(-2, 3):
        row = []
        for m in range(-2, 3):
            total = 0.0
            for s in range(0, 2 * j + 1):
                if j + m - s < 0 or mp - m + s < 0 or j - mp - s < 0:
                    continue
                coef = ((-1) ** (mp - m + s)) * math.sqrt(fact(j + mp) * fact(j - mp) * fact(j + m) * fact(j - m)) \
                    / (fact(j + m - s) * fact(s) * fact(mp - m + s) * fact(j - mp - s))
                total = total + coef * cb ** (2 * j + m - mp - 2 * s) * sb ** (mp - m + 2 * s)
            row.append(xp.exp(-1j * mp * alpha) * total * xp.exp(-1j * m * gamma))
        rows.append(xp.stack(row, axis=-1))
    return xp.stack(rows, axis=-2)


def jframe_to_rift_lframe(modes, jl_angles, xp=None):
    """Apply RIFT's ``fd_L_frame`` rotation to J-frame (2,m) modes.

    ``modes`` has shape (n, 5, N) (m = -2..2); ``jl_angles`` is (n, 3) holding
    ``lalsimutils.extract_JL_angles(P, return_inverse=True)``.  Reproduces
    ``rotate_hlm_static(h, -gamma - pi/2, -beta, -alpha, extra_polarization=pi/2)``
    as used in ``lalsimutils.hlmoft``.  The rotation is linear and the same for
    every frequency, so it commutes with RIFT's FD/TD conditioning, which this
    module does NOT apply (see the design note).
    """
    xp = _module(xp)
    ang = xp.asarray(jl_angles, dtype=xp.float64).reshape(-1, 3)
    a, b, g = ang[:, 0], ang[:, 1], ang[:, 2]
    D = wigner_D_l2(-g - LAL_PI_2, -b, -a, xp)
    phase = complex(np.exp(-2j * LAL_PI_2))
    return xp.einsum("nij,njk->nik", D, modes) * phase


def rift_jl_angles(P_list):
    """Host helper: ``extract_JL_angles`` for a list of ChooseWaveformParams."""
    from RIFT import lalsimutils
    return np.array([lalsimutils.extract_JL_angles(P, return_inverse=True) for P in P_list])
