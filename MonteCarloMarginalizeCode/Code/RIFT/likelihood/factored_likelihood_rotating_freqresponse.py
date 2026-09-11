"""Combined Earth-rotation and finite-arm detector response for conventional ILE.

The finite-arm response is first factored as ``sum_b beta_b(sky,t) W_b(f)``.  Each
``beta_b`` is a finite sidereal Fourier series: its half-width is 2 for the exact
long-wavelength term and ``q+2`` for the arm-projection term of order ``q``.  The
slow-delay Taylor expansion then composes with each frequency basis element, giving

    chi_(b,p,n) = M_n d_t^p [W_b h_lm]

with all sky dependence in a short coefficient vector.  This is the direct composition
described in the generalized-response section of the RIFT scaling paper.  It acts on the
full inertial-frame modes, so precession and higher modes require no special treatment.

The compound bank is deliberately opt-in.  Its U/V precompute scales as the square of
the number of ``(b,p,n)`` elements; this implementation targets long, loud BNS-like
signals, not short eccentric mergers for which finite-arm response alone is sufficient.
"""
from __future__ import division, print_function

import math
import numpy as np

from . import factored_likelihood_with_rotation as flwr
from . import slowrot_freqresponse as sfr


def response_harmonic_width(basis_index):
    """Exact sidereal half-width of finite-response coefficient ``b``."""
    return 2 if basis_index == 0 else (basis_index - 1) + 2


def compound_index_set(Qmax, p_max):
    """Nonzero elementary indices ``a=(b,p,n)`` for the compound response."""
    out = []
    for b in range(Qmax + 2):
        for p in range(p_max + 1):
            width = response_harmonic_width(b) + p
            out.extend((b, p, n) for n in range(-width, width + 1))
    return out


def _basis_harmonics_geom(response, x_arm, y_arm, DEC, psi, Qmax):
    """Fourier coefficients ``A[b][n]`` of each finite-response sky coefficient.

    A small exact DFT is used instead of maintaining separate symbolic expressions for
    every arm-projection power.  The sampled functions are polynomials in sin(g),cos(g)
    with known half-width ``q+2``, so ``2*(Qmax+2)+1`` samples recover them to roundoff.
    ``DEC`` and ``psi`` are vectors of extrinsic samples.
    """
    DEC = np.atleast_1d(np.asarray(DEC, dtype=float))
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    width = Qmax + 2
    ngrid = 2 * width + 1
    g = 2.0 * np.pi * np.arange(ngrid, dtype=float) / float(ngrid)
    X, Y, nhat = sfr._triad(DEC[:, None], psi[:, None], g[None, :])

    Fp, Fc = sfr._lwl_response(response, X, Y)
    zx = np.einsum('...i,i->...', X, x_arm) + 1j * np.einsum('...i,i->...', Y, x_arm)
    zy = np.einsum('...i,i->...', X, y_arm) + 1j * np.einsum('...i,i->...', Y, y_arm)
    ax = np.einsum('...i,i->...', nhat, x_arm)
    ay = np.einsum('...i,i->...', nhat, y_arm)

    values = {0: Fp + 1j * Fc}
    for q in range(Qmax + 1):
        values[1 + q] = 0.5 * (zx ** 2 * ax ** q - zy ** 2 * ay ** q)

    harmonics = {}
    for b, vals in values.items():
        bw = response_harmonic_width(b)
        harmonics[b] = {
            n: np.mean(vals * np.exp(-1j * n * g)[None, :], axis=1)
            for n in range(-bw, bw + 1)
        }
    return harmonics


def combined_response_coefficients_vector(det, RA, DEC, psi, tref, p_max,
                                          Qmax=4, L_arm=None):
    """Compound coefficients ``{(b,p,n): C}`` for a vector of extrinsic samples."""
    import lal
    import lalsimulation as lalsim
    from . import slowrot_response as srr

    RA = np.atleast_1d(np.asarray(RA, dtype=float))
    DEC = np.atleast_1d(np.asarray(DEC, dtype=float))
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    response, x_arm, y_arm, _ = sfr.detector_geometry(det, L_arm=L_arm)
    A = _basis_harmonics_geom(response, x_arm, y_arm, DEC, psi, Qmax)
    gmst = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(tref))))
    g_ev = gmst - RA
    Atil = {b: {n: val * np.exp(1j * n * g_ev) for n, val in Ab.items()}
            for b, Ab in A.items()}

    lald = lalsim.DetectorPrefixToLALDetector(det)
    Bd = srr.delay_harmonics_vector(lald.location, DEC)
    Btil = {m: val * np.exp(1j * m * g_ev) for m, val in Bd.items()}
    tau0 = np.real(sum(Btil.values()))
    drift = dict(Btil)
    drift[0] = drift[0] - tau0
    neg_drift = {m: -val for m, val in drift.items()}

    C = {}
    E = {0: np.ones_like(g_ev, dtype=complex)}
    for p in range(p_max + 1):
        if p:
            E = flwr._convolve_harmonics(E, neg_drift)
        inv_fact = 1.0 / math.factorial(p)
        for b, Ab in Atil.items():
            for n, an in Ab.items():
                for m, em in E.items():
                    key = (b, p, n + m)
                    C[key] = C.get(key, 0j) + inv_fact * an * em
    return C


def _arm_for_detector(L_arm, det):
    return L_arm.get(det, None) if isinstance(L_arm, dict) else L_arm


def PrecomputeLikelihoodTermsRotatingFreqResponse(
        event_time_geo, t_window, P, data_dict, psd_dict, Lmax, fMax,
        Qmax=4, L_arm=None, p_max=0, f_sidereal=flwr.F_SIDEREAL,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.,
        verbose=True, quiet=False, skip_interpolation=False, **hlm_kwargs):
    """Build the intrinsic compound bank indexed by ``(b,p,n)``."""
    from . import factored_likelihood as FL
    from .. import lalsimutils as lsu

    if data_dict.keys() != psd_dict.keys():
        raise ValueError("data and PSD detector sets differ")
    detectors = list(data_dict)
    P.dist = FL.distMpcRef * 1e6 * lsu.lsu_PC
    P.deltaF = data_dict[detectors[0]].deltaF
    hlms, hlms_conj = FL.internal_hlm_generator(P, Lmax, verbose=verbose, quiet=quiet,
                                                **hlm_kwargs)
    modes = list(hlms)
    nfreq = hlms[modes[0]].data.length
    fvals = flwr.evaluate_fvals_from_length(nfreq, hlms[modes[0]].deltaF)
    a_list = compound_index_set(Qmax, p_max)

    rholms = {}; rholms_intp = {}; cross = {}; cross_v = {}; lengths = {}
    for det in detectors:
        _, _, _, length = sfr.detector_geometry(det, L_arm=_arm_for_detector(L_arm, det))
        lengths[det] = float(length)
        weights = sfr.finite_size_response_weights(
            fvals, {'L': float(length), 'T': float(length) / sfr.C_SI}, Qmax)

        weighted = {b: {lm: sfr_weighted_mode(hlms[lm], weights[b]) for lm in modes}
                    for b in range(Qmax + 2)}
        weighted_conj = {
            b: {lm: sfr_weighted_mode(hlms_conj[lm], weights[b]) for lm in modes}
            for b in range(Qmax + 2)}
        deriv = {(b, p): {lm: flwr.fd_apply_time_derivative(weighted[b][lm], p)
                          for lm in modes}
                 for b in range(Qmax + 2) for p in range(p_max + 1)}
        deriv_conj = {
            (b, p): {lm: flwr.fd_apply_time_derivative(weighted_conj[b][lm], p)
                     for lm in modes}
            for b in range(Qmax + 2) for p in range(p_max + 1)}
        chi = {a: {lm: flwr.fd_apply_sidereal_modulation(
                    deriv[(a[0], a[1])][lm], a[2], f_sidereal, 0.0) for lm in modes}
               for a in a_list}
        chi_conj = {a: {lm: flwr.fd_apply_sidereal_modulation(
                         deriv_conj[(a[0], a[1])][lm], a[2], f_sidereal, 0.0)
                         for lm in modes}
                    for a in a_list}

        data = data_dict[det]; psd = psd_dict[det]
        t_det = FL.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, event_time_geo)
        rho_epoch = data.epoch - hlms[modes[0]].epoch
        t_shift = float(float(t_det) - float(t_window) - float(rho_epoch))
        n_shift = int(t_shift / P.deltaT + 0.5)
        n_window = int(2 * t_window / P.deltaT)
        tgrid = np.arange(n_window) * P.deltaT + float(rho_epoch + n_shift * P.deltaT)

        rholms[det] = {}; rholms_intp[det] = {}
        for a in a_list:
            rho = FL.ComputeModeIPTimeSeries(
                chi[a], data, psd, P.fmin, fMax, 1. / 2. / P.deltaT,
                n_shift, n_window, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
            rholms[det][a] = rho
            rholms_intp[det][a] = (None if skip_interpolation else
                                    FL.InterpolateRholms(rho, tgrid, verbose=verbose))

        cross[det] = {}; cross_v[det] = {}
        for a in a_list:
            for ap in a_list:
                cross[det][(a, ap)] = FL.ComputeModeCrossTermIP(
                    chi[a], chi[ap], psd, P.fmin, fMax, 1. / 2. / P.deltaT,
                    P.deltaF, analyticPSD_Q, inv_spec_trunc_Q, T_spec,
                    verbose=False, same_waveform_Q=False)
                cross_v[det][(a, ap)] = FL.ComputeModeCrossTermIP(
                    chi_conj[a], chi[ap], psd, P.fmin, fMax, 1. / 2. / P.deltaT,
                    P.deltaF, analyticPSD_Q, inv_spec_trunc_Q, T_spec, prefix="V",
                    verbose=False, same_waveform_Q=False)

    meta = dict(feature='rotation_freqresponse', Qmax=Qmax, p_max=p_max,
                f_sidereal=f_sidereal, a_list=a_list, modes=modes,
                event_time_geo=float(event_time_geo), L=lengths, L_arm=L_arm,
                post_phase_required=True)
    return rholms_intp, cross, cross_v, rholms, meta


def sfr_weighted_mode(hf, weight):
    """Apply a finite-response weight without importing the scalar likelihood module."""
    out = flwr._copy_freqseries(hf)
    out.data.data[:] = hf.data.data * weight
    return out


def pack_rotating_freqresponse_arrays(meta, rholms, cross, cross_v):
    """Pack the compound bank, with dense U/V arrays to avoid O(A^2) kernel launches."""
    lookup, rho, u_dict, v_dict, epoch = flwr.pack_rotation_arrays(
        meta, rholms, cross, cross_v)
    a_list = list(meta['a_list'])
    u_dense = {}; v_dense = {}
    for det in u_dict:
        u_dense[det] = np.stack([
            np.stack([u_dict[det][(a, ap)] for ap in a_list], axis=0)
            for a in a_list], axis=0)
        v_dense[det] = np.stack([
            np.stack([v_dict[det][(a, ap)] for ap in a_list], axis=0)
            for a in a_list], axis=0)
    return lookup, rho, u_dense, v_dense, epoch


def DiscreteFactoredLogLikelihoodRotatingFreqResponseNoLoop(
        tvals, P_vec, meta, lookupNKDict, rho_by_a, U_by_aa, V_by_aa, epochDict,
        Lmax=2, array_output=False, time_interp='nearest', xpy=np):
    """Vectorized CPU/GPU likelihood for simultaneous rotation and finite-arm response."""
    Qmax = int(meta['Qmax'])
    L_arm = meta.get('L_arm')

    def coefficients(det, RA, DEC, psi, tref, p_max):
        return combined_response_coefficients_vector(
            det, RA, DEC, psi, tref, p_max, Qmax=Qmax,
            L_arm=_arm_for_detector(L_arm, det))

    return flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        tvals, P_vec, meta, lookupNKDict, rho_by_a, U_by_aa, V_by_aa, epochDict,
        Lmax=Lmax, array_output=array_output, time_interp=time_interp, xpy=xpy,
        coefficient_function=coefficients,
        reflection_function=lambda a: (a[0], a[1], -a[2]))
