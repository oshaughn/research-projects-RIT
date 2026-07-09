"""
factored_likelihood_with_rotation : slow-rotation generalization of the RIFT precompute.

This is a *separate realization* of factored_likelihood.PrecomputeLikelihoodTerms that
accounts for the time dependence of the ground-based detector response over long signals
(Earth rotation).  It does NOT regenerate a family of time-domain templates.  Instead it
builds each frequency-domain mode ONCE and realizes the two response modulations as cheap
frequency-domain operations on that single template:

    * delay-derivative order p  (Path B): time derivative  h^(p)(t)  <->  (s 2 pi i f)^p H(f)
      -- an exact per-bin weight in the FD.  (s = FT_SIGN, fixed to match RIFT's FFT
      convention; see evaluate_fvals, which itself warns the omega-sign is reversed.)

    * sidereal harmonic n        (Paths A,B): modulation  exp(i n Omega t)  <->  frequency
      shift  H(f - n f_sid).  Since f_sid = Omega/2pi ~ 1.16e-5 Hz is sub-bin, the shift is
      realized exactly as a time-domain linear phase (one FFT round trip per template), not
      an integer bin roll.

The response harmonics themselves (the analytic scalars that multiply these precomputed
overlaps in the extrinsic layer) come from RIFT.likelihood.slowrot_response:

      F_k(t)   = sum_{n=-2}^{2} A_n e^{i n g},   tau_k(t) = sum_{n=-1}^{1} B_n e^{i n g},
      g = GMST(t) - RA,   A_n, B_n depend only on (detector, dec, psi/dec).

See the design notes (rift-slow-rotation: sec_formalism 'master expansion', sec_amplitude
Path A, sec_delay Path B, app_response) for the derivation and the meaning of the indices.

Index conventions in the returned structures
--------------------------------------------
An "elementary modulated template" is labelled a = (p, n):
      chi_a(t) = exp(i n Omega t) * d^p/dt^p h_lm(t - tau_0).
The physical data-term time series carries a post-phase (derived in the notes):
      Q^a_lm(t) = exp(i n Omega t) * < chi_a(.-t) | d >          [applied here]
while the cross terms are arrival-time independent:
      U^{a,a'} = < chi_a | chi_a' >,   V^{a,a'} = < chi_a^* | chi_a' >.

Path A (default) uses only p = 0 (amplitude drift; exact 5-harmonic).  Path B adds p >= 1.

Heavy RIFT imports (factored_likelihood, lalsimutils) are done lazily inside the precompute
so the light FD primitives below are importable with numpy alone (used by the unit tests).
"""
from __future__ import print_function, division

import numpy as np

# Sidereal angular rate [rad/s] and frequency [Hz]
OMEGA_EARTH = 7.292115e-5
F_SIDEREAL = OMEGA_EARTH / (2.0 * np.pi)

# Sign in the time-derivative weight (s 2 pi i f)^p, matched to RIFT's evaluate_fvals FFT
# convention.  VALIDATED empirically against a LAL FFT round trip in
# test_slowrot_fd_ops.py (which will fail loudly if this is wrong).
FT_SIGN = -1.0


# ---------------------------------------------------------------------------
# Low-level FD primitives (numpy only; operate on a complex spectrum + its fvals).
# ---------------------------------------------------------------------------
def evaluate_fvals_from_length(npts, deltaF):
    """Signed frequency array for a RIFT two-sided COMPLEX16FrequencySeries.

    Replicates RIFT.lalsimutils.evaluate_fvals (kept local so these primitives need no
    heavyweight import).  Note RIFT's expression  ``npts/2 - k if k<=npts/2 else -k+npts/2``
    is the same in both branches, i.e. simply f[k] = deltaF*(npts/2 - k), descending from
    +fNyq at k=0.
    """
    k = np.arange(npts)
    return deltaF * (npts / 2.0 - k)


def time_derivative_weight(fvals, p):
    """(FT_SIGN * 2 pi i f)^p : exact FD weight for the p-th time derivative."""
    if p == 0:
        return np.ones_like(fvals, dtype=complex)
    return (FT_SIGN * 2.0j * np.pi * fvals) ** p


def apply_time_derivative_array(spectrum, fvals, p):
    """Return the spectrum of d^p/dt^p h given the spectrum of h (both two-sided)."""
    return spectrum * time_derivative_weight(fvals, p)


def apply_sidereal_modulation_array(spectrum, coef, deltaF, f_sidereal=F_SIDEREAL):
    """Return the spectrum of exp(i coef Omega t) h(t) given the spectrum of h(t).

    exp(i coef Omega t) multiplication in time <-> frequency shift by coef*f_sidereal.
    This is sub-bin (f_sidereal << deltaF for realistic segments), so it is applied exactly
    as a time-domain linear phase via a DFT round trip that respects RIFT's fvals packing.

    WARNING: this reference implementation forms an explicit (npts x npts) DFT matrix -- it
    is O(npts^2) and intended ONLY for unit tests at small npts.  The production path uses
    the LAL-FFT round trip in _lal_freq_modulate() below (O(npts log npts)).
    """
    if coef == 0:
        return spectrum.copy()
    npts = len(spectrum)
    dt = 1.0 / (npts * deltaF)
    t = np.arange(npts) * dt
    fvals = evaluate_fvals_from_length(npts, deltaF)
    # Round-trip-consistent DFT pair for the packing f[k]=deltaF*(npts/2-k):
    #   h(t_j) = (1/npts) sum_k H_k exp(+2 pi i f_k t_j),  H_k = sum_j h(t_j) exp(-2 pi i f_k t_j)
    # LAL's convention (encoded by evaluate_fvals) reverses the omega sign: a tone
    # exp(+2 pi i f0 t) lands at fvals = -f0.  The consistent DFT pair is therefore
    #   h(t_j) = (1/npts) sum_k H_k exp(-2 pi i f_k t_j),  H_k = sum_j h(t_j) exp(+2 pi i f_k t_j).
    phase_inv = np.exp(-2.0j * np.pi * np.outer(t, fvals))      # (npts, npts): H -> h(t)
    h_td = phase_inv.dot(spectrum) / npts
    h_td *= np.exp(1.0j * coef * 2.0 * np.pi * f_sidereal * t)  # exp(i coef Omega t)
    phase_fwd = np.exp(2.0j * np.pi * np.outer(fvals, t))       # h(t) -> H
    return phase_fwd.dot(h_td)


# ---------------------------------------------------------------------------
# LAL-series wrappers (used inside the precompute; import lal lazily).
# ---------------------------------------------------------------------------
def _copy_freqseries(hf):
    import lal
    out = lal.CreateCOMPLEX16FrequencySeries(
        hf.name, hf.epoch, hf.f0, hf.deltaF, hf.sampleUnits, hf.data.length)
    out.data.data[:] = hf.data.data[:]
    return out


def fd_apply_time_derivative(hf, p):
    """COMPLEX16FrequencySeries -> new series for the p-th time derivative."""
    if p == 0:
        return _copy_freqseries(hf)
    out = _copy_freqseries(hf)
    fvals = evaluate_fvals_from_length(hf.data.length, hf.deltaF)
    out.data.data[:] = apply_time_derivative_array(hf.data.data, fvals, p)
    return out


def _lal_freq_modulate(hf, coef, f_sidereal=F_SIDEREAL, t_ref=0.0):
    """Modulate by exp(i coef Omega (t_abs - t_ref)) via a LAL-FFT round trip (O(N log N)).

    Reverse-FFT the spectrum to the time domain, multiply by the sidereal linear phase
    referenced to ABSOLUTE sample time t_abs = float(hf.epoch) + j*deltaT (minus t_ref),
    forward-FFT back.  Uses the same COMPLEX16 transforms RIFT uses for its overlaps.

    The reference t_ref is physical, not cosmetic: the true antenna phase is
    exp(i n (GMST(t')-RA)) = exp(i n (GMST(t_ev)-RA)) * exp(i n Omega (t'-t_ev)), so the
    precompute must carry exactly exp(i n Omega (t' - t_ev)) at absolute data time t',
    with the constant GMST(t_ev) piece carried analytically by A_n (slowrot_response).
    Hence callers pass t_ref = event_time_geo.
    """
    import lal
    if coef == 0:
        return _copy_freqseries(hf)
    npts = hf.data.length
    deltaT = 1.0 / (npts * hf.deltaF)
    ts = lal.CreateCOMPLEX16TimeSeries("m", hf.epoch, 0., deltaT,
                                       lal.DimensionlessUnit, npts)
    revplan = lal.CreateReverseCOMPLEX16FFTPlan(npts, 0)
    fwdplan = lal.CreateForwardCOMPLEX16FFTPlan(npts, 0)
    lal.COMPLEX16FreqTimeFFT(ts, hf, revplan)
    t_abs = float(hf.epoch) + np.arange(npts) * deltaT
    ts.data.data[:] = ts.data.data * np.exp(
        1.0j * coef * 2.0 * np.pi * f_sidereal * (t_abs - t_ref))
    out = _copy_freqseries(hf)
    lal.COMPLEX16TimeFreqFFT(out, ts, fwdplan)
    return out


def fd_apply_sidereal_modulation(hf, n, f_sidereal=F_SIDEREAL, t_ref=0.0):
    """COMPLEX16FrequencySeries -> new series for exp(i n Omega (t_abs - t_ref)) h(t)."""
    return _lal_freq_modulate(hf, n, f_sidereal, t_ref)


def build_elementary_template(hf, p, n, f_sidereal=F_SIDEREAL):
    """chi_{(p,n)} spectrum from the base mode spectrum hf: derivative then modulation."""
    return fd_apply_sidereal_modulation(fd_apply_time_derivative(hf, p), n, f_sidereal)


def _elementary_index_set(harmonics, p_max):
    """List of a=(p,n).  Superset grid; unused (p,n) get zero response coefficients."""
    return [(p, n) for p in range(p_max + 1) for n in harmonics]


# ---------------------------------------------------------------------------
# The precompute.
# ---------------------------------------------------------------------------
def PrecomputeLikelihoodTermsWithRotation(
        event_time_geo, t_window, P, data_dict, psd_dict, Lmax, fMax,
        harmonics=(-2, -1, 0, 1, 2), p_max=0, f_sidereal=F_SIDEREAL,
        analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.,
        verbose=True, quiet=False, internal_fast_precompute=True,
        skip_interpolation=False, **hlm_kwargs):
    """Slow-rotation analogue of factored_likelihood.PrecomputeLikelihoodTerms.

    Builds each FD mode once (via factored_likelihood.internal_hlm_generator) and forms the
    generalized overlaps for every elementary modulated template a=(p,n):

        rholms_intp_rot[det][a][(l,m)] : interpolant of Q^a_lm(t) = e^{inOmega t}<chi_a(.-t)|d>
        crossTerms_rot[det][(a,a')]    : { ((l,m),(l',m')) : <chi_a|chi_a'> }
        crossTermsV_rot[det][(a,a')]   : { ((l,m),(l',m')) : <chi_a^*|chi_a'> }

    Parameters mirror PrecomputeLikelihoodTerms; rotation-specific:
        harmonics : sidereal harmonic indices n to carry (antenna needs |n|<=2).
        p_max     : max delay-derivative order (0 = Path A amplitude-only; >=1 = Path B).
        f_sidereal: sidereal frequency [Hz].

    Response coefficients A_n(det,dec,psi), B_n(det,dec) that contract these overlaps are
    NOT computed here -- see slowrot_response and the (forthcoming) rotation-aware
    FactoredLogLikelihood assembly.  This routine produces only the intrinsic-only,
    sky-independent overlap bank.

    NOTE: full-stack numerical validation (V1 of the notes' validation matrix: agreement
    with a brute-force time-varying-response likelihood) requires the data/PSD/waveform
    environment and is done separately; the FD primitives used here are unit-tested in
    test_slowrot_fd_ops.py.
    """
    # Lazy heavy imports (need the full RIFT stack / lal).
    import lal
    from . import factored_likelihood as FL
    from .. import lalsimutils as lsu

    assert data_dict.keys() == psd_dict.keys()
    detectors = list(data_dict.keys())
    t_ev = float(event_time_geo)
    # The exp(i n Omega t) modulation for the data term Q is applied to the DATA (shift by
    # -n f_sidereal, referenced to t_ev), which is mode-independent: one shift per (det,n),
    # and -- since the modulation lives on the fixed absolute data-time axis -- needs NO
    # arrival-time-dependent post-phase.  U,V use modulated templates (same t_ev reference).

    # Reference distance handling identical to the base precompute.
    P.dist = FL.distMpcRef * 1e6 * lsu.lsu_PC
    P.deltaF = data_dict[detectors[0]].deltaF

    # --- build base FD modes ONCE ---
    hlms, hlms_conj = FL.internal_hlm_generator(P, Lmax, verbose=verbose, quiet=quiet,
                                                **hlm_kwargs)
    a_list = _elementary_index_set(harmonics, p_max)
    p_values = sorted(set(p for (p, n) in a_list))

    # --- derivative-weighted template families (per p), built once, reused for all n,det ---
    #     (the derivative is a trivial FD weight; no waveform regeneration)
    hlms_p = {p: {lm: fd_apply_time_derivative(hlms[lm], p) for lm in hlms} for p in p_values}
    hlms_conj_p = {p: {lm: fd_apply_time_derivative(hlms_conj[lm], p) for lm in hlms_conj}
                   for p in p_values}

    # For U,V we need the modulated templates chi_a (modulation cannot be pushed onto data
    # in a <template|template> overlap).  Build them once per a (cheap FD op).
    #
    # CRITICAL reference-time note: the template time series carry the INTRINSIC epoch
    # (hlms.epoch ~ -T_dur, i.e. near 0), NOT the absolute event time t_ev ~ 1e9.  When the
    # template is placed at the event, sample j sits at absolute time t' = t_ev + (hlms.epoch
    # + j*dt), so the physical modulation exp(i n Omega (t' - t_ev)) = exp(i n Omega
    # (hlms.epoch + j*dt)) -- i.e. reference the template modulation to 0, NOT to t_ev.
    # (Referencing to t_ev with the tiny template epoch would apply exp(i n Omega * ~-1e9),
    # a ~1e4 rad spurious phase that randomizes U,V and inflates lnL beyond 0.5<d|d>.)
    # The data-term route above keeps t_ev because the DATA epoch really is ~ t_ev.
    chi = {a: {lm: fd_apply_sidereal_modulation(hlms_p[a[0]][lm], a[1], f_sidereal, 0.0)
               for lm in hlms} for a in a_list}
    chi_conj = {a: {lm: fd_apply_sidereal_modulation(hlms_conj_p[a[0]][lm], a[1], f_sidereal, 0.0)
                    for lm in hlms_conj} for a in a_list}

    rholms_rot = {}
    rholms_intp_rot = {}
    crossTerms_rot = {}
    crossTermsV_rot = {}

    for det in detectors:
        psd = psd_dict[det]
        data = data_dict[det]
        t_det = FL.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, event_time_geo)
        rho_epoch = data.epoch - hlms[list(hlms.keys())[0]].epoch
        t_shift = float(float(t_det) - float(t_window) - float(rho_epoch))
        N_shift = int(t_shift / P.deltaT + 0.5)
        N_window = int(2 * t_window / P.deltaT)
        t = np.arange(N_window) * P.deltaT + float(rho_epoch + N_shift * P.deltaT)

        # ---- data-term overlaps Q^a_lm(t) ----
        # exp(i n Omega t) on the template is equivalent to shifting the data spectrum by
        # -n f_sidereal (mode-independent).  Realize it by modulating the DATA time series
        # by exp(-i n Omega (t_abs - t_ev)) (round trip).  Because the modulation lives on
        # the absolute data-time axis, the resulting overlap is directly
        #   Q^a_lm(t) = int e^{-i n Omega (t'-t_ev)} [d^p h_lm]^*(t'-t) d(t') dt'
        # with NO arrival-time-dependent post-phase.
        rholms_rot[det] = {}
        rholms_intp_rot[det] = {}
        data_by_n = {}
        for n in set(nn for (_, nn) in a_list):
            data_by_n[n] = data if n == 0 else _lal_freq_modulate(data, -n, f_sidereal, t_ev)

        for a in a_list:
            p, n = a
            rho = FL.ComputeModeIPTimeSeries(
                hlms_p[p], data_by_n[n], psd, P.fmin, fMax, 1. / 2. / P.deltaT,
                N_shift, N_window, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
            rholms_rot[det][a] = rho
            if not skip_interpolation:
                rholms_intp_rot[det][a] = FL.InterpolateRholms(rho, t, verbose=verbose)
            else:
                rholms_intp_rot[det][a] = None

        # ---- cross terms U,V for each ordered pair (a,a') ----
        crossTerms_rot[det] = {}
        crossTermsV_rot[det] = {}
        for a in a_list:
            for ap in a_list:
                crossTerms_rot[det][(a, ap)] = FL.ComputeModeCrossTermIP(
                    chi[a], chi[ap], psd, P.fmin, fMax, 1. / 2. / P.deltaT, P.deltaF,
                    analyticPSD_Q, inv_spec_trunc_Q, T_spec, verbose=False,
                    same_waveform_Q=False)
                crossTermsV_rot[det][(a, ap)] = FL.ComputeModeCrossTermIP(
                    chi_conj[a], chi[ap], psd, P.fmin, fMax, 1. / 2. / P.deltaT, P.deltaF,
                    analyticPSD_Q, inv_spec_trunc_Q, T_spec, prefix="V", verbose=False,
                    same_waveform_Q=False)

    meta = dict(harmonics=tuple(harmonics), p_max=p_max, f_sidereal=f_sidereal,
                a_list=a_list, event_time_geo=float(event_time_geo),
                omega_earth=OMEGA_EARTH, modes=list(hlms.keys()))
    return rholms_intp_rot, crossTerms_rot, crossTermsV_rot, rholms_rot, meta


# ---------------------------------------------------------------------------
# Rotation-aware log-likelihood assembly (Path A: amplitude drift, p_max=0).
# ---------------------------------------------------------------------------
def antenna_harmonics_tilde(det, RA, DEC, psi, tref):
    """Return {n: A_tilde_n} where F_k(t) = sum_n A_tilde_n exp(i n Omega (t - tref)).

    A_tilde_n = A_n(response, dec, psi) * exp(i n (GMST(tref) - RA)), with A_n the
    RA/time-independent antenna harmonics from slowrot_response.  So A_tilde_n is the
    coefficient of the precompute's modulation exp(i n Omega (t - tref)); sum_n A_tilde_n
    = F_k(tref) reproduces lal.ComputeDetAMResponse exactly.
    """
    import lal
    import lalsimulation as lalsim
    from . import slowrot_response as srr
    lald = lalsim.DetectorPrefixToLALDetector(det)
    A = srr.antenna_harmonics(lald.response, DEC, psi)
    g_ev = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(tref)))) - RA
    return {n: A[n] * np.exp(1.0j * n * g_ev) for n in A}


def _convolve_harmonics(a, b):
    """Convolve two harmonic sequences (dicts {m: coef}) -> dict {m: coef}."""
    out = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            out[m1 + m2] = out.get(m1 + m2, 0j) + c1 * c2
    return out


def rotation_coefficients(det, RA, DEC, psi, tref, p_max):
    """Coefficients {(p, ntilde): C} of the elementary modulated templates
    chi_{(p,n)} = exp(i n Omega (t-tref)) h^{(p)}(t-tau0) in the folded response+delay
    template (Path B).  For p_max=0 this is {(0,n): A_tilde_n} (Path A).

        C_{(p, ntilde)} = (1/p!) sum_{n+m=ntilde} A_tilde_n [(-D)^{*p}]_m
    with A_tilde_n the antenna harmonics and D_m the delay-DRIFT harmonics:
    delta_tau(t) = tau(t) - tau(tref) = sum_m D_m exp(i m Omega (t-tref)),
    D_0 = -(B_tilde_1 + B_tilde_-1), D_{+-1} = B_tilde_{+-1}  (B_tilde = delay harmonics).
    At Omega->0 delta_tau=0 -> D=0 -> C_{(p,n)}=0 for p>=1, so Path B -> Path A.
    """
    import math
    import lal
    import lalsimulation as lalsim
    from . import slowrot_response as srr
    lald = lalsim.DetectorPrefixToLALDetector(det)
    g_ev = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(tref)))) - RA
    A = srr.antenna_harmonics(lald.response, DEC, psi)
    Atil = {n: A[n] * np.exp(1.0j * n * g_ev) for n in A}
    if p_max == 0:
        return {(0, n): Atil[n] for n in Atil}
    Bd = srr.delay_harmonics(lald.location, DEC)          # {m: B_m}, m in -1,0,1
    Btil = {m: Bd[m] * np.exp(1.0j * m * g_ev) for m in Bd}
    tau0 = np.real(sum(Btil.values()))                   # tau(tref)
    D = {m: Btil[m] for m in Btil}
    D[0] = D[0] - tau0                                    # = -(Btil_1 + Btil_-1)
    negD = {m: -D[m] for m in D}
    C = {}
    E = {0: 1.0 + 0j}                                     # (-D)^{*0}
    for p in range(p_max + 1):
        if p > 0:
            E = _convolve_harmonics(E, negD)             # (-D)^{*p}
        inv = 1.0 / math.factorial(p)
        for n, an in Atil.items():
            for m, em in E.items():
                key = (p, n + m)
                C[key] = C.get(key, 0j) + inv * an * em
    return C


def FactoredLogLikelihoodWithRotation(extr_params, rholms_intp_rot, crossTerms_rot,
                                      crossTermsV_rot, meta, Lmax):
    """Slow-rotation analogue of factored_likelihood.FactoredLogLikelihood (Path A).

    Contracts the harmonic-resolved precompute bank with the antenna harmonics
    A_tilde_n(det, RA, DEC, psi, tref).  Reduces EXACTLY to the baseline
    FactoredLogLikelihood when f_sidereal -> 0 (all modulations become identity and
    sum_n A_tilde_n -> F_k(tref)).

    Currently implements p_max=0 (amplitude drift only); the delay-derivative (Path B)
    contraction with B_n is a TODO.
    """
    import lal
    from . import factored_likelihood as FL
    from .. import lalsimutils as lsu

    p_max = meta['p_max']
    a_list = list(meta['a_list'])
    harmonics = list(meta['harmonics'])
    hset = set(harmonics)
    for n in harmonics:
        assert -n in hset, "harmonic set must be symmetric (need -n for the V term): %s" % harmonics

    RA = extr_params.phi
    DEC = extr_params.theta
    tref = extr_params.tref
    phiref = extr_params.phiref
    incl = extr_params.incl
    psi = extr_params.psi
    dist = extr_params.dist

    detectors = list(rholms_intp_rot.keys())
    a0 = a_list[0]
    modes = list(rholms_intp_rot[detectors[0]][a0].keys())
    Ylms = FL.ComputeYlms(Lmax, incl, -phiref, selected_modes=modes)

    distMpc = dist / (lsu.lsu_PC * 1e6)
    invDistMpc = FL.distMpcRef / distMpc

    lnL = 0.
    for det in detectors:
        C = rotation_coefficients(det, RA, DEC, psi, tref, p_max)  # {(p,n): C_a}
        t_det = FL.ComputeArrivalTimeAtDetector(det, RA, DEC, tref)
        CT = crossTerms_rot[det]
        CTV = crossTermsV_rot[det]

        # Q^a_lm(t_det) for each elementary template a=(p,n)
        Q = {a: {m: rholms_intp_rot[det][a][m](float(t_det))
                 for m in modes} for a in a_list}

        # term1 = Re[ sum_lm conj(Ylm) sum_a conj(C_a) Q^a_lm(t_det) ]
        term1 = 0.
        for m in modes:
            s = 0.
            for a in a_list:
                s += np.conj(C.get(a, 0j)) * Q[a][m]
            term1 += np.conj(Ylms[m]) * s
        term1 = np.real(term1) * invDistMpc

        # term2 = -1/4 Re[ sum_{p1,p2} ( U-part conj(Y1)Y2 + V-part Y1 Y2 ) ]
        #   U-part = sum_{a,a'} conj(C_a) C_a' U^{(a,a')}[p1,p2]
        #   V-part = sum_{a=(p,nu),a'} C_{(p,-nu)} C_a' V^{(a,a')}[p1,p2]
        term2 = 0.
        for p1 in modes:
            for p2 in modes:
                u = 0.
                v = 0.
                for a in a_list:
                    aR = (a[0], -a[1])
                    for ap in a_list:
                        u += np.conj(C.get(a, 0j)) * C.get(ap, 0j) * CT[(a, ap)][(p1, p2)]
                        v += C.get(aR, 0j) * C.get(ap, 0j) * CTV[(a, ap)][(p1, p2)]
                term2 += u * np.conj(Ylms[p1]) * Ylms[p2] + v * Ylms[p1] * Ylms[p2]
        term2 = -np.real(term2) / 4. / (distMpc / FL.distMpcRef) ** 2

        lnL += term1 + term2

    return lnL


# ---------------------------------------------------------------------------
# Vectorized (NoLoop) rotation likelihood -- the maintained batchmode path.
# ---------------------------------------------------------------------------
def rotation_coefficients_vector(det, RA, DEC, psi, tref, p_max):
    """Vectorized rotation_coefficients: RA, DEC, psi are arrays (npts_ex,); returns
    {(p, n): complex ndarray (npts_ex,)}.  Same algebra as rotation_coefficients."""
    import math
    import lal
    import lalsimulation as lalsim
    from . import slowrot_response as srr
    lald = lalsim.DetectorPrefixToLALDetector(det)
    RA = np.asarray(RA); DEC = np.asarray(DEC); psi = np.asarray(psi)
    g_ev = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(float(tref)))) - RA
    A = srr.antenna_harmonics_vector(lald.response, DEC, psi)
    Atil = {n: A[n] * np.exp(1.0j * n * g_ev) for n in A}
    if p_max == 0:
        return {(0, n): Atil[n] for n in Atil}
    Bd = srr.delay_harmonics_vector(lald.location, DEC)
    Btil = {m: Bd[m] * np.exp(1.0j * m * g_ev) for m in Bd}
    tau0 = np.real(sum(Btil.values()))
    D = {m: Btil[m] for m in Btil}
    D[0] = D[0] - tau0
    negD = {m: -D[m] for m in D}
    C = {}
    E = {0: np.ones_like(g_ev, dtype=complex)}
    for p in range(p_max + 1):
        if p > 0:
            E = _convolve_harmonics(E, negD)
        inv = 1.0 / math.factorial(p)
        for n, an in Atil.items():
            for m, em in E.items():
                key = (p, n + m)
                C[key] = C.get(key, 0j) + inv * an * em
    return C


def pack_rotation_arrays(meta, rholms_rot, crossTerms_rot, crossTermsV_rot):
    """Pack the elementary-template precompute bank into dense arrays for the NoLoop path.

    Returns (lookupNKDict, rho_by_a, U_by_aa, V_by_aa, epochDict), keyed per detector by
    elementary template a=(p,n) (Path A: a=(0,n); Path B: also p>=1).
    """
    a_list = list(meta['a_list'])
    lookupNKDict = {}; rho_by_a = {}; U_by_aa = {}; V_by_aa = {}; epochDict = {}
    for det in rholms_rot:
        a0 = a_list[0]
        modes = list(rholms_rot[det][a0].keys())
        n_lms = len(modes)
        idx = {m: i for i, m in enumerate(modes)}
        lookupNKDict[det] = np.array([[m[0], m[1]] for m in modes], dtype=int)
        npts = rholms_rot[det][a0][modes[0]].data.length
        epochDict[det] = float(rholms_rot[det][a0][modes[0]].epoch)
        rho_by_a[det] = {}
        for a in a_list:
            arr = np.zeros((n_lms, npts), dtype=np.complex128)
            for m in modes:
                arr[idx[m]] = rholms_rot[det][a][m].data.data
            rho_by_a[det][a] = arr
        U_by_aa[det] = {}; V_by_aa[det] = {}
        for a in a_list:
            for ap in a_list:
                Um = np.zeros((n_lms, n_lms), dtype=np.complex128)
                Vm = np.zeros((n_lms, n_lms), dtype=np.complex128)
                cU = crossTerms_rot[det][(a, ap)]
                cV = crossTermsV_rot[det][(a, ap)]
                for m1 in modes:
                    for m2 in modes:
                        Um[idx[m1], idx[m2]] = cU[(m1, m2)]
                        Vm[idx[m1], idx[m2]] = cV[(m1, m2)]
                U_by_aa[det][(a, ap)] = Um
                V_by_aa[det][(a, ap)] = Vm
    return lookupNKDict, rho_by_a, U_by_aa, V_by_aa, epochDict


def DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        tvals, P_vec, meta, lookupNKDict, rho_by_a, U_by_aa, V_by_aa, epochDict,
        Lmax=2, array_output=False, time_interp='nearest', xpy=np):
    """Vectorized rotation-aware lnL (Path A).

    GPU: pass xpy=cupy with rho_by_a/U_by_aa/V_by_aa already on device (the ILE converts them
    when --gpu).  term1 reuses the baseline fused Q_inner_product kernel PER elementary
    template a (no (n_ext,npts,n_lms) temporary -> same memory footprint as the baseline,
    looped over |a_list|); term2 is small |a_list|^2 einsums over n_lms^2.  Requires n_cal=1
    (no glitch/calibration marginalization).

    Mirrors factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopOrig with
    sums over sidereal harmonics and the per-sample antenna harmonics
    A_tilde_n = A_n(det,dec,psi) exp(i n (GMST(tref)-RA)).  Extrinsic params in P_vec are
    ARRAYS of length npts_ex (RA, DEC, incl, phiref, psi, dist); tref, deltaT scalar.

    array_output=True returns lnL_t of shape (npts_ex, npts) (before time marginalization);
    array_output=False returns the time-marginalized lnL of shape (npts_ex,).
    """
    import lal
    from . import factored_likelihood as FL
    on_gpu = not (xpy is np)
    if on_gpu:
        from . import Q_inner_product

    def _h(v):   # host (numpy) copy -- the antenna-harmonic / Ylm / delay helpers are numpy/LAL
        return v.get() if hasattr(v, 'get') else np.asarray(v)

    a_list = list(meta['a_list'])
    p_max = meta['p_max']
    RA = _h(P_vec.phi); DEC = _h(P_vec.theta)
    incl = _h(P_vec.incl); phiref = _h(P_vec.phiref); psi = _h(P_vec.psi)
    npts = len(tvals); npts_ex = len(RA)
    distMpc = _h(P_vec.dist) / (lal.PC_SI * 1e6)
    inv_dist = xpy.asarray(FL.distMpcRef / distMpc)   # (npts_ex,) on device
    lnL_t = xpy.zeros((npts_ex, npts), dtype=np.float64)

    for det in rho_by_a:
        n_lms = len(lookupNKDict[det])
        Ylms = FL.ComputeYlmsArrayVector(lookupNKDict[det], incl, -phiref).T  # (npts_ex, n_lms)
        C = rotation_coefficients_vector(det, RA, DEC, psi, P_vec.tref, p_max)  # {(p,n): (npts_ex,)}
        zeroC = np.zeros(npts_ex, dtype=complex)

        def Cg(a):
            return C[a] if a in C else zeroC
        t_ref = epochDict[det]
        # Match the maintained baseline NoLoop's precision-preserving time reference: keep the
        # tiny (tref - epoch) offset and the small geometric delay separate, rather than
        # subtracting two ~1e9 s absolute arrival times.  The absolute-difference form loses
        # ~1e-3 of a sample bin, which is harmless for nearest-neighbour snapping but shows up
        # directly in the sub-bin fraction used by cubic interpolation.
        detector_location = np.asarray(FL.lalsim.DetectorPrefixToLALDetector(det).location)
        gmst_tref = float(lal.GreenwichMeanSiderealTime(P_vec.tref))
        # RA/DEC are host (numpy) copies via _h(); force the delay onto the host too.
        # TimeDelayFromEarthCenter defaults xpy=cupy whenever cupy is importable, which would
        # feed host arrays to cupy.cos and raise -- invisible in a no-cupy sandbox, fatal on a GPU.
        t_det = float(P_vec.tref - float(t_ref)) + FL.TimeDelayFromEarthCenter(
            detector_location, RA, DEC, gmst_tref, xpy=np)
        sample_first = (t_det + float(tvals[0])) / P_vec.deltaT   # float(): tvals may be a cupy array on GPU
        if time_interp == 'nearest':
            ifirst = (np.round(sample_first) + 0.5).astype(int)
        else:
            ifirst = np.floor(sample_first).astype(int)
            frac_first = (sample_first - np.floor(sample_first)).astype(np.float64)
        ilast = ifirst + npts

        # Device-side arrays for the heavy contraction (identity on CPU; host->device on GPU).
        Ylms_d = xpy.asarray(Ylms); conjY_d = xpy.conj(Ylms_d)
        zero_d = xpy.zeros(npts_ex, dtype=complex)
        C_d = {k: xpy.asarray(v) for k, v in C.items()}
        Cg_d = lambda a: C_d[a] if a in C_d else zero_d

        term1 = xpy.zeros((npts_ex, npts), dtype=np.complex128)
        if on_gpu:
            # term1 = Re[ sum_a conj(C_a) sum_lm conj(Ylm) Q^a_lm(t) ]: reuse the baseline fused
            # kernel per elementary template a (A = conj(Ylm)), no (n_ex,npts,n_lms) temporary.
            ifirst_i32 = xpy.asarray(ifirst).astype(np.int32)
            frac_d = None if time_interp == 'nearest' else xpy.asarray(frac_first)
            for a in a_list:
                Q = xpy.ascontiguousarray(rho_by_a[det][a].T)   # (n_time, n_lms), device
                if time_interp == 'nearest':
                    res = Q_inner_product.Q_inner_product_cupy(Q, conjY_d, ifirst_i32, npts)
                else:
                    res = Q_inner_product.Q_inner_product_cubic_cupy(Q, conjY_d, ifirst_i32, frac_d, npts)
                term1 += xpy.conj(Cg_d(a))[:, None] * res
        else:
            for a in a_list:
                det_rho = rho_by_a[det][a]
                if time_interp == 'nearest':
                    Qa = np.empty((npts_ex, npts, n_lms), dtype=np.complex128)
                    for i in range(npts_ex):
                        Qa[i] = det_rho[..., ifirst[i]:ilast[i]].T
                else:
                    # cubic sub-sample interpolation (calmarg time_interp='cubic'):
                    # _cubic_Q_window_numpy expects Q_block shape (n_time, n_lm).
                    Qa = FL._cubic_Q_window_numpy(det_rho.T, ifirst, frac_first, npts)
                term1 += np.conj(Cg(a))[:, None] * np.einsum('xi,xti->xt', np.conj(Ylms), Qa)
        term1 = term1.real * inv_dist[:, None]

        term2 = xpy.zeros(npts_ex, dtype=np.complex128)
        for a in a_list:
            aR = (a[0], -a[1])
            for ap in a_list:
                term2 += xpy.conj(Cg_d(a)) * Cg_d(ap) * xpy.einsum(
                    'xi,xj,ij->x', conjY_d, Ylms_d, xpy.asarray(U_by_aa[det][(a, ap)]))
                term2 += Cg_d(aR) * Cg_d(ap) * xpy.einsum(
                    'xi,xj,ij->x', Ylms_d, Ylms_d, xpy.asarray(V_by_aa[det][(a, ap)]))
        term2 = (-0.25 * term2.real) * inv_dist ** 2

        lnL_t += term1 + term2[:, None]

    if array_output:
        return lnL_t
    lnLmax = xpy.max(lnL_t, axis=-1, keepdims=True)
    simps = FL.optimized_gpu_tools.simps if on_gpu else FL.my_simps
    L = simps(xpy.exp(lnL_t - lnLmax), dx=P_vec.deltaT, axis=-1)
    return lnLmax[..., 0] + xpy.log(L)
