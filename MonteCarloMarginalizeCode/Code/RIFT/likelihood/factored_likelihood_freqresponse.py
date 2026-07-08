"""
factored_likelihood_freqresponse : finite-size (frequency-dependent detector response)
generalization of the RIFT precompute + scalar likelihood.  "Thrust 2 / route (b)"
(sky-harmonic expansion) of the slow-rotation program -- KEEPS the sky extrinsic.

Physics (see slowrot_freqresponse.py for the response derivation)
----------------------------------------------------------------
Beyond the long-wavelength limit the detector strain in the frequency domain is a
per-frequency antenna pattern acting on the two polarizations,

    h_k(f) = F_+(f;sky) h_+(f) + F_x(f;sky) h_x(f)
           = (1/2)[ F(f) Sigma(f) + Fbar(f) Sigma*(f) ] ,
    F = F_+ + i F_x,  Fbar = F_+ - i F_x = conj(F(-f)),  Sigma = sum_lm Y_lm h_lm.

The finite-size response factors into a SKY-INDEPENDENT frequency basis W_p(f) times
an analytic SKY/pol scalar b_p (slowrot_freqresponse.finite_size_response_weights /
finite_size_beta):

    F(f;sky) = sum_p b_p(sky) W_p(f) ,     Fbar(f;sky) = sum_p conj(b_p) W_p(f) ,

    p=0        : W_0 = 1,                       b_0 = F0  (exact lal ComputeDetAMResponse)
    p=1+q      : W_{1+q} = e^{-i2pi f T} c_q(f) - [q==0],  b_{1+q} = beta_q (arm expansion)

with T = L/c the common (direction-independent) light-crossing delay folded into W_p as
a linear FD phase, and c_q(f) the sky-independent power-series basis of the single-arm
transfer.  Every W_p is Hermitian, so -- UNLIKE the sidereal-rotation case -- the V cross
term needs NO harmonic reflection.

This module is the DIRECT analogue of factored_likelihood_with_rotation: it folds each
W_p(f) into the FD modes ONCE (h_lm(f) -> W_p(f) h_lm(f)) and reuses the EXISTING
ComputeModeIPTimeSeries / ComputeModeCrossTermIP to build

    Q^p_lm(t) = < W_p h_lm | d >,   U^{p,p'} = < W_p h_lm | W_p' h_l'm' >,
    V^{p,p'} = < conj(W_p h_lm) | W_p' h_l'm' > .

MAINTAINED PATH: the production entry point is the vectorized (NoLoop)
DiscreteFactoredLogLikelihoodFreqResponseNoLoop -- the direct finite-size analogue of
factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop (the maintained
likelihood used by integrate_likelihood_extrinsic_batchmode).  It is VALIDATED against that
NoLoop baseline: at L->0 it reduces to it (to ~1e-9), NOT to the older per-detector scalar
SingleDetectorLogLikelihood.  FactoredLogLikelihoodFreqResponse (scalar) is a per-sample
validation companion only (mirrors factored_likelihood_with_rotation.FactoredLogLikelihoodWithRotation).

The term1/term2 use the same factored form as the NoLoop path (all RIFT factored likelihoods
share this algebra; W_0=1 and b_0 = ComputeDetAMResponse at L->0):
    term1 = Re[ (distRef/dist) sum_lm conj(Ylm) sum_p conj(b_p) Q^p_lm(t_det) ]
    term2 = -(1/4)(distRef/dist)^2 Re[ sum_{p,p'} ( conj(b_p) b_p' U^{p,p'} conj(Y1)Y2
                                                   +      b_p  b_p' V^{p,p'} Y1 Y2 ) ]

Heavy RIFT imports are lazy so the light FD helpers stay importable with numpy alone.
"""
from __future__ import print_function, division

import numpy as np

from . import slowrot_freqresponse as sfr


# ---------------------------------------------------------------------------
# Low-level FD primitive: fold a per-bin weight into a COMPLEX16FrequencySeries.
# ---------------------------------------------------------------------------
def evaluate_fvals_from_length(npts, deltaF):
    """Signed frequency array for a RIFT two-sided COMPLEX16FrequencySeries.

    Identical packing to factored_likelihood_with_rotation.evaluate_fvals_from_length /
    RIFT.lalsimutils.evaluate_fvals: f[k] = deltaF*(npts/2 - k), descending from +fNyq.
    """
    k = np.arange(npts)
    return deltaF * (npts / 2.0 - k)


def _copy_freqseries(hf):
    import lal
    out = lal.CreateCOMPLEX16FrequencySeries(
        hf.name, hf.epoch, hf.f0, hf.deltaF, hf.sampleUnits, hf.data.length)
    out.data.data[:] = hf.data.data[:]
    return out


def fd_apply_weight(hf, weight):
    """COMPLEX16FrequencySeries -> new series with spectrum multiplied by weight[k]."""
    out = _copy_freqseries(hf)
    out.data.data[:] = hf.data.data * weight
    return out


# ---------------------------------------------------------------------------
# The precompute (sky-independent; per detector uses only its arm length L).
# ---------------------------------------------------------------------------
def PrecomputeLikelihoodTermsFreqResponse(
        event_time_geo, t_window, P, data_dict, psd_dict, Lmax, fMax,
        Qmax=4, L_arm=None,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.,
        verbose=True, quiet=False, skip_interpolation=False, **hlm_kwargs):
    """Finite-size analogue of PrecomputeLikelihoodTerms(WithRotation).

    Builds each FD mode once and forms the generalized overlaps for every response
    basis element p = 0..Qmax+1:

        rholms_intp_fr[det][p][(l,m)] : interpolant of Q^p_lm(t) = <W_p h_lm(.-t)|d>
        crossTerms_fr[det][(p,p')]    : { ((l,m),(l',m')) : <W_p h_lm | W_p' h_l'm'> }
        crossTermsV_fr[det][(p,p')]   : { ((l,m),(l',m')) : <conj(W_p h_lm) | W_p' h_l'm'> }

    Parameters mirror PrecomputeLikelihoodTerms; finite-size specific:
        Qmax  : highest power of the arm projection retained (basis size Qmax+2).
        L_arm : arm-length override [m] (float -> all detectors, or dict det->L).
                None -> each detector's native LAL arm length.

    The response coefficients b_p(det,RA,DEC,psi,tref) that contract these overlaps are
    NOT computed here -- see FactoredLogLikelihoodFreqResponse.  Returns the intrinsic-
    only, sky-independent overlap bank plus meta (carries per-detector L, T, Qmax).
    """
    import lal
    from . import factored_likelihood as FL
    from .. import lalsimutils as lsu

    assert data_dict.keys() == psd_dict.keys()
    detectors = list(data_dict.keys())

    def _L_of(det):
        if isinstance(L_arm, dict):
            return L_arm.get(det, None)
        return L_arm

    P.dist = FL.distMpcRef * 1e6 * lsu.lsu_PC
    P.deltaF = data_dict[detectors[0]].deltaF

    # --- build base FD modes ONCE ---
    hlms, hlms_conj = FL.internal_hlm_generator(P, Lmax, verbose=verbose, quiet=quiet,
                                                **hlm_kwargs)
    modes = list(hlms.keys())
    npts = hlms[modes[0]].data.length
    deltaF = hlms[modes[0]].deltaF
    fvals = evaluate_fvals_from_length(npts, deltaF)
    p_list = list(range(Qmax + 2))

    rholms_fr = {}
    rholms_intp_fr = {}
    crossTerms_fr = {}
    crossTermsV_fr = {}
    Ldict = {}
    Tdict = {}

    for det in detectors:
        psd = psd_dict[det]
        data = data_dict[det]

        # per-detector geometry (arm length only; sky enters later via b_p)
        _, _, _, L = sfr.detector_geometry(det, L_arm=_L_of(det))
        Ldict[det] = float(L)
        Tdict[det] = float(L / sfr.C_SI)
        # sky-independent response basis weights W_p(f) on the signed fvals
        geom_dummy = dict(L=float(L), T=float(L / sfr.C_SI))
        W = sfr.finite_size_response_weights(fvals, geom_dummy, Qmax)     # (Qmax+2, npts)

        # weighted mode families eta^p = W_p h_lm , etac^p = W_p conj(h_lm)
        eta = {p: {lm: fd_apply_weight(hlms[lm], W[p]) for lm in modes} for p in p_list}
        etac = {p: {lm: fd_apply_weight(hlms_conj[lm], W[p]) for lm in modes} for p in p_list}

        t_det = FL.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, event_time_geo)
        rho_epoch = data.epoch - hlms[modes[0]].epoch
        t_shift = float(float(t_det) - float(t_window) - float(rho_epoch))
        N_shift = int(t_shift / P.deltaT + 0.5)
        N_window = int(2 * t_window / P.deltaT)
        t = np.arange(N_window) * P.deltaT + float(rho_epoch + N_shift * P.deltaT)

        rholms_fr[det] = {}
        rholms_intp_fr[det] = {}
        for p in p_list:
            rho = FL.ComputeModeIPTimeSeries(
                eta[p], data, psd, P.fmin, fMax, 1. / 2. / P.deltaT,
                N_shift, N_window, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
            rholms_fr[det][p] = rho
            if not skip_interpolation:
                rholms_intp_fr[det][p] = FL.InterpolateRholms(rho, t, verbose=verbose)
            else:
                rholms_intp_fr[det][p] = None

        crossTerms_fr[det] = {}
        crossTermsV_fr[det] = {}
        for p in p_list:
            for pp in p_list:
                crossTerms_fr[det][(p, pp)] = FL.ComputeModeCrossTermIP(
                    eta[p], eta[pp], psd, P.fmin, fMax, 1. / 2. / P.deltaT, P.deltaF,
                    analyticPSD_Q, inv_spec_trunc_Q, T_spec, verbose=False,
                    same_waveform_Q=False)
                crossTermsV_fr[det][(p, pp)] = FL.ComputeModeCrossTermIP(
                    etac[p], eta[pp], psd, P.fmin, fMax, 1. / 2. / P.deltaT, P.deltaF,
                    analyticPSD_Q, inv_spec_trunc_Q, T_spec, prefix="V", verbose=False,
                    same_waveform_Q=False)

    meta = dict(Qmax=Qmax, p_list=p_list, modes=modes,
                event_time_geo=float(event_time_geo),
                L=Ldict, T=Tdict, L_arm=L_arm)
    return rholms_intp_fr, crossTerms_fr, crossTermsV_fr, rholms_fr, meta


# ---------------------------------------------------------------------------
# Response coefficients b_p(det, RA, DEC, psi, tref) : b_0 = F0, b_{1+q} = beta_q.
# ---------------------------------------------------------------------------
def response_coefficients(det, RA, DEC, psi, tref, Qmax, L_arm=None):
    """{p: b_p} response coefficients for the finite-size basis at (RA,DEC,psi,tref).

    b_0 = F0 = lal.ComputeDetAMResponse (exact LWL baseline);
    b_{1+q} = beta_q(sky) = (1/2)[zx^2 a_x^q - zy^2 a_y^q].
    Uses gmst = GreenwichMeanSiderealTime(tref) exactly like ComplexAntennaFactor.
    """
    import lal
    gmst = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(tref))))
    geom = sfr.finite_size_geometry(det, RA, DEC, psi, gmst=gmst, L_arm=L_arm)
    beta = sfr.finite_size_beta(geom, Qmax)
    b = {0: geom['F0']}
    for q in range(Qmax + 1):
        b[1 + q] = complex(beta[q])
    return b


def FactoredLogLikelihoodFreqResponse(extr_params, rholms_intp_fr, crossTerms_fr,
                                      crossTermsV_fr, meta, Lmax):
    """Finite-size analogue of factored_likelihood.FactoredLogLikelihood.

    Contracts the response-basis precompute bank with b_p(det,RA,DEC,psi,tref) and the
    Ylm.  Reduces EXACTLY to the baseline FactoredLogLikelihood as L -> 0.
    """
    from . import factored_likelihood as FL
    from .. import lalsimutils as lsu

    Qmax = meta['Qmax']
    p_list = list(meta['p_list'])
    L_arm = meta.get('L_arm', None)

    RA = extr_params.phi
    DEC = extr_params.theta
    tref = extr_params.tref
    phiref = extr_params.phiref
    incl = extr_params.incl
    psi = extr_params.psi
    dist = extr_params.dist

    detectors = list(rholms_intp_fr.keys())
    p0 = p_list[0]
    modes = list(rholms_intp_fr[detectors[0]][p0].keys())
    Ylms = FL.ComputeYlms(Lmax, incl, -phiref, selected_modes=modes)

    distMpc = dist / (lsu.lsu_PC * 1e6)
    invDistMpc = FL.distMpcRef / distMpc

    def _L_of(det):
        if isinstance(L_arm, dict):
            return L_arm.get(det, None)
        return L_arm

    lnL = 0.
    for det in detectors:
        b = response_coefficients(det, RA, DEC, psi, tref, Qmax, L_arm=_L_of(det))
        t_det = FL.ComputeArrivalTimeAtDetector(det, RA, DEC, tref)
        CT = crossTerms_fr[det]
        CTV = crossTermsV_fr[det]

        Q = {p: {m: rholms_intp_fr[det][p][m](float(t_det))
                 for m in modes} for p in p_list}

        term1 = 0.
        for m in modes:
            s = 0.
            for p in p_list:
                s += np.conj(b[p]) * Q[p][m]
            term1 += np.conj(Ylms[m]) * s
        term1 = np.real(term1) * invDistMpc

        term2 = 0.
        for m1 in modes:
            for m2 in modes:
                u = 0.
                v = 0.
                for p in p_list:
                    for pp in p_list:
                        u += np.conj(b[p]) * b[pp] * CT[(p, pp)][(m1, m2)]
                        v += b[p] * b[pp] * CTV[(p, pp)][(m1, m2)]
                term2 += u * np.conj(Ylms[m1]) * Ylms[m2] + v * Ylms[m1] * Ylms[m2]
        term2 = -np.real(term2) / 4. / (distMpc / FL.distMpcRef) ** 2

        lnL += term1 + term2

    return lnL


# ---------------------------------------------------------------------------
# Dense-array packing + vectorized time-marginalized lnL (for peak / validation).
# ---------------------------------------------------------------------------
def pack_freqresponse_arrays(meta, rholms_fr, crossTerms_fr, crossTermsV_fr):
    """Pack the basis precompute bank into dense arrays keyed per detector by p.

    Returns (lookupNKDict, rho_by_p, U_by_pp, V_by_pp, epochDict).
    """
    p_list = list(meta['p_list'])
    lookupNKDict = {}; rho_by_p = {}; U_by_pp = {}; V_by_pp = {}; epochDict = {}
    for det in rholms_fr:
        p0 = p_list[0]
        modes = list(rholms_fr[det][p0].keys())
        n_lms = len(modes)
        idx = {m: i for i, m in enumerate(modes)}
        lookupNKDict[det] = np.array([[m[0], m[1]] for m in modes], dtype=int)
        npts = rholms_fr[det][p0][modes[0]].data.length
        epochDict[det] = float(rholms_fr[det][p0][modes[0]].epoch)
        rho_by_p[det] = {}
        for p in p_list:
            arr = np.zeros((n_lms, npts), dtype=np.complex128)
            for m in modes:
                arr[idx[m]] = rholms_fr[det][p][m].data.data
            rho_by_p[det][p] = arr
        U_by_pp[det] = {}; V_by_pp[det] = {}
        for p in p_list:
            for pp in p_list:
                Um = np.zeros((n_lms, n_lms), dtype=np.complex128)
                Vm = np.zeros((n_lms, n_lms), dtype=np.complex128)
                cU = crossTerms_fr[det][(p, pp)]
                cV = crossTermsV_fr[det][(p, pp)]
                for m1 in modes:
                    for m2 in modes:
                        Um[idx[m1], idx[m2]] = cU[(m1, m2)]
                        Vm[idx[m1], idx[m2]] = cV[(m1, m2)]
                U_by_pp[det][(p, pp)] = Um
                V_by_pp[det][(p, pp)] = Vm
    return lookupNKDict, rho_by_p, U_by_pp, V_by_pp, epochDict


def DiscreteFactoredLogLikelihoodFreqResponseNoLoop(
        tvals, P_vec, meta, lookupNKDict, rho_by_p, U_by_pp, V_by_pp, epochDict,
        Lmax=2, array_output=False):
    """Vectorized finite-size lnL over a time window (single extrinsic point per call
    is supported; P_vec fields RA,DEC,incl,phiref,psi,dist may be length-1 arrays).

    array_output=True returns lnL_t of shape (npts_ex, npts); else time-marginalized.
    """
    import lal
    from . import factored_likelihood as FL

    p_list = list(meta['p_list'])
    Qmax = meta['Qmax']
    L_arm = meta.get('L_arm', None)
    npts = len(tvals); npts_ex = len(np.atleast_1d(P_vec.phi))
    RA = np.atleast_1d(P_vec.phi); DEC = np.atleast_1d(P_vec.theta)
    incl = np.atleast_1d(P_vec.incl); phiref = np.atleast_1d(P_vec.phiref)
    psi = np.atleast_1d(P_vec.psi)
    distMpc = np.atleast_1d(P_vec.dist) / (lal.PC_SI * 1e6)
    lnL_t = np.zeros((npts_ex, npts), dtype=np.float64)

    def _L_of(det):
        if isinstance(L_arm, dict):
            return L_arm.get(det, None)
        return L_arm

    for det in rho_by_p:
        n_lms = len(lookupNKDict[det])
        Ylms = FL.ComputeYlmsArrayVector(lookupNKDict[det], incl, -phiref).T  # (npts_ex,n_lms)
        # per-sample response coefficients b_p (npts_ex,)
        bvec = {}
        for i in range(npts_ex):
            bi = response_coefficients(det, float(RA[i]), float(DEC[i]), float(psi[i]),
                                       P_vec.tref, Qmax, L_arm=_L_of(det))
            for p in p_list:
                bvec.setdefault(p, np.zeros(npts_ex, dtype=complex))
                bvec[p][i] = bi[p]

        t_ref = epochDict[det]
        t_det = FL.lalT(det, RA, DEC, P_vec.tref)
        ifirst = (np.round((t_det + tvals[0] - t_ref) / P_vec.deltaT) + 0.5).astype(int)
        ilast = ifirst + npts

        term1 = np.zeros((npts_ex, npts), dtype=np.complex128)
        for p in p_list:
            det_rho = rho_by_p[det][p]
            Qa = np.empty((npts_ex, npts, n_lms), dtype=np.complex128)
            for i in range(npts_ex):
                Qa[i] = det_rho[..., ifirst[i]:ilast[i]].T
            term1 += np.conj(bvec[p])[:, None] * np.einsum('xi,xti->xt', np.conj(Ylms), Qa)
        term1 = term1.real * (FL.distMpcRef / distMpc)[:, None]

        term2 = np.zeros(npts_ex, dtype=np.complex128)
        for p in p_list:
            for pp in p_list:
                term2 += np.conj(bvec[p]) * bvec[pp] * np.einsum(
                    'xi,xj,ij->x', np.conj(Ylms), Ylms, U_by_pp[det][(p, pp)])
                term2 += bvec[p] * bvec[pp] * np.einsum(
                    'xi,xj,ij->x', Ylms, Ylms, V_by_pp[det][(p, pp)])
        term2 = (-0.25 * term2.real) * (FL.distMpcRef / distMpc) ** 2

        lnL_t += term1 + term2[:, None]

    if array_output:
        return lnL_t
    lnLmax = np.max(lnL_t, axis=-1, keepdims=True)
    L = FL.my_simps(np.exp(lnL_t - lnLmax), dx=P_vec.deltaT, axis=-1)
    return lnLmax[..., 0] + np.log(L)
