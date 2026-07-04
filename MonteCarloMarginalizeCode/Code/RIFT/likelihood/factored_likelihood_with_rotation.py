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


def _lal_freq_modulate(hf, coef, f_sidereal=F_SIDEREAL):
    """Production exp(i coef Omega t) modulation via a LAL-FFT round trip (O(N log N)).

    Reverse-FFT the spectrum to the time domain, multiply by the sidereal linear phase,
    forward-FFT back.  Uses the same COMPLEX16 transforms RIFT uses for its overlaps, so
    the convention is identical.  The absolute time origin only sets a constant phase
    (carried analytically by A_n / GMST_ref downstream), so t_j = j*deltaT is fine.
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
    t = np.arange(npts) * deltaT
    ts.data.data[:] = ts.data.data * np.exp(1.0j * coef * 2.0 * np.pi * f_sidereal * t)
    out = _copy_freqseries(hf)
    lal.COMPLEX16TimeFreqFFT(out, ts, fwdplan)
    return out


def fd_apply_sidereal_modulation(hf, n, f_sidereal=F_SIDEREAL):
    """COMPLEX16FrequencySeries -> new series for exp(i n Omega t) h(t)."""
    return _lal_freq_modulate(hf, n, f_sidereal)


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
    modulate_data = True  # apply the exp(i n Omega t) frequency shift on the DATA for Q
                          # (mode-independent: one shift per (det,n) instead of per template)

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
    chi = {a: {lm: fd_apply_sidereal_modulation(hlms_p[a[0]][lm], a[1], f_sidereal)
               for lm in hlms} for a in a_list}
    chi_conj = {a: {lm: fd_apply_sidereal_modulation(hlms_conj_p[a[0]][lm], a[1], f_sidereal)
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
        rholms_rot[det] = {}
        rholms_intp_rot[det] = {}
        # Pre-shift the (whitened-by-the-IP) data once per harmonic n: exp(i n Omega t) on
        # the template is equivalent to shifting the data spectrum by -n f_sidereal, which
        # is mode-independent.  We realize it by modulating the DATA time series by
        # exp(-i n Omega t) (round trip), then post-multiply Q by exp(+i n Omega t).
        data_by_n = {}
        for n in set(nn for (_, nn) in a_list):
            if modulate_data and n != 0:
                data_by_n[n] = _lal_freq_modulate(data, -n, f_sidereal)
            else:
                data_by_n[n] = data

        for a in a_list:
            p, n = a
            templates = hlms_p[p] if modulate_data else chi[a]
            rho = FL.ComputeModeIPTimeSeries(
                templates, data_by_n[n], psd, P.fmin, fMax, 1. / 2. / P.deltaT,
                N_shift, N_window, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
            # post-phase exp(+i n Omega (t - event_time_geo)); the constant piece
            # exp(i n Omega event_time_geo) is carried analytically by A_n (GMST_ref).
            if n != 0:
                phase = np.exp(1.0j * n * OMEGA_EARTH * (t - float(event_time_geo)))
                for lm in rho:
                    rho[lm].data.data[:] = rho[lm].data.data[:len(t)] * phase[:rho[lm].data.length]
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
