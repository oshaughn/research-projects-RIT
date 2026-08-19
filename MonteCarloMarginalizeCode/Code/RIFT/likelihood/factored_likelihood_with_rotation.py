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
      chi_a(u) = exp(i n Omega u) * d^p/du^p h_lm(u - tau_0),
with u the template's INTRINSIC time (its own epoch, ~0), not absolute GPS.  Everything the
precompute returns is a plain overlap against that intrinsic-time object:
      Q^a_lm(t) = < chi_a(.-t) | d >,
      U^{a,a'}  = < chi_a | chi_a' >,   V^{a,a'} = < chi_a^* | chi_a' >.

THE ARRIVAL-TIME POST-PHASE IS THE EXTRINSIC LAYER'S JOB, AND IT APPLIES TO BOTH TERMS.
The physical modulation runs on absolute time, exp(i n Omega (t' - tref)); placing the
template at arrival time t splits it as exp(i n Omega u) * exp(i n Omega (t - tref)).  So the
coefficient that multiplies chi_a in the model is not C_a but

      C~_a(t) = C_a * exp(i n_a Omega (t - tref)),                     [rotation_post_phase]

and the SAME C~ must be used in the data term AND in the model norm.  Using C~ in only one of
them evaluates <d|h> and <h|h> for two different h, which breaks the Cauchy-Schwarz bound
lnL <= (1/2)<d|d> by O(n Omega (t-tref)) -- see test_slowrot_cauchy_schwarz.py.

Path A (default) uses only p = 0 (amplitude drift; exact 5-harmonic).  Path B adds p >= 1.

Heavy RIFT imports (factored_likelihood, lalsimutils) are done lazily inside the precompute
so the light FD primitives below are importable with numpy alone (used by the unit tests).
"""
from __future__ import print_function, division

import warnings

import numpy as np

# Sidereal angular rate [rad/s] and frequency [Hz]
OMEGA_EARTH = 7.292115e-5
F_SIDEREAL = OMEGA_EARTH / (2.0 * np.pi)

# Sign in the time-derivative weight (s 2 pi i f)^p, matched to RIFT's evaluate_fvals FFT
# convention.  VALIDATED empirically against a LAL FFT round trip in
# test_slowrot_fd_ops.py (which will fail loudly if this is wrong).
FT_SIGN = -1.0

# Half-width of the ANTENNA harmonic set: F_k(t) = sum_{|n|<=2} A_n e^{i n g} is exact
# (the antenna pattern is quadratic in the rotating detector basis vectors).  The DELAY
# harmonic set B_n has half-width 1.  rotation_coefficients convolves the antenna
# harmonics with the delay-drift harmonics once per derivative order, so the harmonic
# index of the response coefficients C_{(p,ntilde)} widens by exactly one per order --
# see required_harmonic_width, and test_slowrot_harmonic_width.py, which MEASURES both
# half-widths rather than trusting this comment.
N_ANTENNA_HARMONICS = 2
N_DELAY_HARMONICS = 1


def required_harmonic_width(p_max):
    """Half-width |ntilde|_max actually populated by rotation_coefficients at this p_max.

    C_{(p,ntilde)} = (1/p!) sum_{n+m=ntilde} A_tilde_n [(-D)^{*p}]_m, with |n| <= 2 and
    |m| <= 1, so the p-th derivative order reaches |ntilde| <= 2 + p and the full bank
    needs |ntilde| <= 2 + p_max.  Any C outside the precomputed harmonic set has no
    elementary-template band, and BOTH maintained evaluators drop it without complaint
    (the NoLoop's Cg/Cg_d return zero for a missing a; the JAX packer in jax_ile.banded
    packs only a_list) -- i.e. a narrow harmonic set silently truncates the model.  See
    issue #142.
    """
    return N_ANTENNA_HARMONICS + N_DELAY_HARMONICS * int(p_max)


def widen_harmonics_for_p_max(harmonics, p_max):
    """Union of a requested harmonic set with the symmetric range required at p_max.

    Returns ``(harmonics_out, widened_Q)``.  The requested set is returned UNCHANGED
    (same order) when it is already wide enough, so callers that rely on the ordering of
    ``meta['a_list']`` are unaffected in the common case.
    """
    w = required_harmonic_width(p_max)
    required = set(range(-w, w + 1))
    have = set(int(n) for n in harmonics)
    if required.issubset(have):
        return tuple(harmonics), False
    return tuple(sorted(have | required)), True


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
    """(FT_SIGN * 2 pi i f)^p : FD weight for the p-th time derivative.

    THE NYQUIST BIN IS ZEROED FOR ODD p, and only for odd p.  This packing carries +fNyq
    (k=0) but NOT -fNyq: the bin holding -f[k] is k' = N-k, which for k=0 is bin 0 itself.
    So that one bin has to serve for both signs, and the weight can only do that when it is
    EVEN in f -- i.e. when p is even.  For odd p it is odd in f, and two analytically
    identical expressions then disagree there by a SIGN:

        conj(h^(p))   ->  -(FT_SIGN 2 pi i fNyq)^p conj(H[0])      (differentiate, then conj)
        (conj h)^(p)  ->  +(FT_SIGN 2 pi i fNyq)^p conj(H[0])      (conj, then differentiate)

    The precompute takes the second route for the conjugate template family (hlms_conj_p),
    and the first is what any explicitly assembled model gives, so U -- which takes both
    factors from the same family -- never notices, while V = <chi_a^*|chi_a'> pairs the two
    orders against each other and picks up the sign flip.  The sidereal modulation is a
    sub-bin frequency shift applied as a time-domain phase, so it SPREADS that one bin
    across the whole band rather than leaving it at the top.

    That was not a rounding-level effect: an FD mode from internal_hlm_generator carries
    |H(+fNyq)| ~ 0.02-0.14 of |H(100 Hz)|, and the resulting p_max=1 model norm was wrong by
    1.5e-07 relative (0.015 nats out of 1.0e+05) -- enough to push the Cauchy-Schwarz check
    4e-03 nats OVER (1/2)<d|d>.  See issue #159.

    Zero is the RIGHT value at odd p, not a compromise.  On this grid the Nyquist component
    is the alternating sequence (-1)^j; as a real signal cos(2 pi fNyq t) its derivative
    -2 pi fNyq sin(2 pi fNyq t) vanishes at every sample, and as a complex tone
    exp(+2 pi i fNyq t) it is indistinguishable from exp(-2 pi i fNyq t), whose odd
    derivatives differ by a sign.  Zero is both the sampled answer and the only consistent
    one, and it is what keeps d^p/dt^p of a REAL series real.

    EVEN p IS LEFT ALONE, and zeroing it would be a regression rather than extra safety:
    (2 pi i fNyq)^p is real for even p, so there is no ambiguity to resolve, and the
    derivative IS representable -- d^2/dt^2 (-1)^j = -(2 pi fNyq)^2 (-1)^j exactly.  An
    earlier revision of this fix zeroed every p >= 1; measured against the analytic
    derivative of a Nyquist-carrying multitone that cost 90% relative error at p = 2 and
    99% at p = 4 (the untouched weight is exact there to 3e-14), and moved a real p_max=2
    bank by 0.207 nats.  test_slowrot_fd_ops pins both halves, at p = 1..6.

    Do NOT reason that the Nyquist bin sits above fMax and therefore cannot matter -- it
    does sit above fMax, and it still mattered, because the modulation round trip does not
    leave it there.

    THE SAME RULE APPLIES ELSEWHERE, and if you are editing this you probably need to edit
    that too: slowrot_freqresponse.finite_size_response_weights (Path D) has the same
    unpaired-bin problem and resolves it the same way -- the Hermitian average, which there
    is Re W_p(+fNyq).  Its predicate lives in slowrot_freqresponse.unpaired_extreme_bin.
    Neither module imports the other, so the two are a deliberate duplicate; see #164.  They
    are not byte-identical (that one also declines on an all-negative axis), so do not
    assume they are interchangeable.
    """
    if p == 0:
        return np.ones_like(fvals, dtype=complex)
    w = (FT_SIGN * 2.0j * np.pi * fvals) ** p
    if p % 2 == 0:
        return w
    f = np.asarray(fvals)
    if f.ndim < 1 or f.size < 2 or not np.any(f < 0):
        # Nothing to repair: a one-sided (or degenerate) frequency axis has no unpaired
        # Nyquist bin.  Leave it rather than eat the top of its band.
        return w
    fn = np.max(np.abs(f))
    if np.any(f >= fn) and np.any(f <= -fn):
        # Both +fn and -fn are present, so the extreme bin IS paired and the weight is well
        # defined there.  Test UNPAIREDNESS, not magnitude: keying on |f| == max alone would
        # blank both ends of a symmetric axis, where nothing is wrong.
        return w
    w = np.array(w, dtype=complex)
    w[np.abs(f) >= fn] = 0.      # abs(): the unpaired bin is at -fNyq in fftfreq ordering
    return w


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
    exp(i n (GMST(t')-RA)) = exp(i n (GMST(tref)-RA)) * exp(i n Omega (t'-tref)), with the
    constant GMST(tref) piece carried analytically by A_n (slowrot_response).

    ALL CALLERS NOW PASS t_ref = 0.0, i.e. they modulate on the template's own INTRINSIC time
    axis (hf.epoch ~ -T_dur, near zero).  An earlier revision also modulated the DATA with
    t_ref = event_time_geo, to push exp(i n Omega t) off the template and onto the data; that
    identity is false for a noise-weighted overlap and is gone.  The remaining absolute-time
    piece, exp(i n Omega (t_arrival - tref)), is applied once in the extrinsic layer by
    rotation_post_phase() -- to BOTH the data term and the model norm.
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
        skip_interpolation=False, widen_harmonics=True, **hlm_kwargs):
    """Slow-rotation analogue of factored_likelihood.PrecomputeLikelihoodTerms.

    Builds each FD mode once (via factored_likelihood.internal_hlm_generator) and forms the
    generalized overlaps for every elementary modulated template a=(p,n):

        rholms_intp_rot[det][a][(l,m)] : interpolant of Q^a_lm(t) = e^{inOmega t}<chi_a(.-t)|d>
        crossTerms_rot[det][(a,a')]    : { ((l,m),(l',m')) : <chi_a|chi_a'> }
        crossTermsV_rot[det][(a,a')]   : { ((l,m),(l',m')) : <chi_a^*|chi_a'> }

    Parameters mirror PrecomputeLikelihoodTerms; rotation-specific:
        harmonics : sidereal harmonic indices ntilde to carry.  The bank must cover EVERY
            index the response coefficients populate, which is NOT just the antenna's
            |n| <= 2: rotation_coefficients convolves the antenna harmonics (|n| <= 2)
            with the delay-drift harmonics (|m| <= 1) once per derivative order, so the
            required half-width is

                required_harmonic_width(p_max) = 2 + p_max

            i.e. |ntilde| <= 2 at p_max=0, <= 3 at p_max=1, <= 4 at p_max=2.  The default
            (-2..2) is the p_max=0 answer ONLY.  A coefficient with no band is dropped
            without complaint by both maintained evaluators (the NoLoop's Cg/Cg_d return
            zero for a missing a; the JAX packer in jax_ile.banded packs only a_list), so
            a too-narrow set yields a quietly truncated model -- consistent, but not the
            model that was asked for.  See issue #142.
        widen_harmonics : if True (default) a too-narrow `harmonics` is widened to the
            union with (-(2+p_max) .. 2+p_max) and a RuntimeWarning names the new width;
            the extra bands cost |a_list|^2 cross-term overlaps, so the warning is worth
            reading.  Set False ONLY to build a deliberately truncated bank for band-level
            inspection that will never be turned into a likelihood -- the truncation is
            then recorded as meta['harmonics_truncated'].
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
    # --- harmonic-width contract (issue #142) -------------------------------------
    # rotation_coefficients populates |ntilde| <= 2 + p_max; anything outside the bank is
    # dropped silently downstream.  Widen (or, if the caller opted out, record the fact).
    # tuple() FIRST and use only the tuple below: `harmonics` may be any iterable, and a
    # generator consumed here and re-iterated later would silently yield an empty a_list.
    harmonics_requested = tuple(harmonics)
    n_required = required_harmonic_width(p_max)
    if widen_harmonics:
        harmonics, _widened = widen_harmonics_for_p_max(harmonics_requested, p_max)
        harmonics_truncated = False
        if _widened:
            warnings.warn(
                "PrecomputeLikelihoodTermsWithRotation: harmonics=%s cannot carry every "
                "response coefficient at p_max=%d (rotation_coefficients populates "
                "|ntilde| <= 2 + p_max = %d); widened to %s.  Pass a harmonic set at "
                "least this wide to silence this, or widen_harmonics=False to accept a "
                "truncated model." % (harmonics_requested, p_max, n_required, harmonics),
                RuntimeWarning, stacklevel=2)
    else:
        harmonics = harmonics_requested
        harmonics_truncated = not set(range(-n_required, n_required + 1)).issubset(
            set(int(n) for n in harmonics))

    # Lazy heavy imports (need the full RIFT stack / lal).
    import lal
    from . import factored_likelihood as FL
    from .. import lalsimutils as lsu

    assert data_dict.keys() == psd_dict.keys()
    detectors = list(data_dict.keys())
    # NOTE: event_time_geo now only sets the retained-window placement (t_shift/N_shift) and
    # is recorded in meta.  The bank itself is referenced entirely to the template's intrinsic
    # epoch; the absolute-time reference enters once, in the extrinsic layer, as the
    # rotation_post_phase() applied to BOTH the data term and the model norm.

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

        # ---- data-term overlaps Q^a_lm(t) = <chi_a(.-t)|d> ----
        # The MODULATED template goes into the overlap, against the untouched data, so Q and
        # the U,V cross terms below are overlaps of the same chi_a and the extrinsic layer's
        # post-phase C~_a = C_a exp(i n Omega (t-tref)) makes term1 and term2 consistent.
        #
        # An earlier revision instead pushed the modulation onto the DATA (shift its spectrum
        # by -n f_sidereal) and dropped the post-phase, on the grounds that
        # <e^{inOmega.}h | d> == <h | e^{-inOmega.}d>.  That identity holds for the UNWEIGHTED
        # overlap and FAILS for the noise-weighted one used here: a frequency shift does not
        # commute with the 1/S(f) band weight.  Measured, the two routes differ by ~1e-4 of
        # <d|d> at the physical rate -- enough to violate Cauchy-Schwarz, and it is the U,V
        # terms (which have no data-side route available) that are then left inconsistent.
        rholms_rot[det] = {}
        rholms_intp_rot[det] = {}

        for a in a_list:
            p, n = a
            rho = FL.ComputeModeIPTimeSeries(
                chi[a], data, psd, P.fmin, fMax, 1. / 2. / P.deltaT,
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

    # post_phase_required marks the BANK CONVENTION, which changed when the arrival-time
    # post-phase moved to the extrinsic layer: Q is now <chi_a(.-t)|d> against untouched
    # data, and any evaluator MUST apply rotation_post_phase() to both terms.  A consumer
    # written against the old convention is silently wrong rather than broken, so it is
    # recorded here and every evaluator that post-phases REJECTS a bank without it (see
    # require_post_phase_bank): FactoredLogLikelihoodWithRotation and
    # DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation below, and
    # jax_ile.banded.build_rotation_data / jax_ile.core._accumulate_unit_banded.
    meta = dict(harmonics=tuple(harmonics), p_max=p_max, f_sidereal=f_sidereal,
                a_list=a_list, event_time_geo=float(event_time_geo),
                omega_earth=OMEGA_EARTH, modes=list(hlms.keys()),
                post_phase_required=True,
                # issue #142: what was asked for, what the coefficients need, and whether
                # this bank is a truncated model (only possible via widen_harmonics=False).
                harmonics_requested=harmonics_requested,
                harmonics_required=n_required,
                harmonics_truncated=bool(harmonics_truncated))
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


def rotation_post_phase(C, omega, delta):
    """Arrival-time post-phase on the elementary-template coefficients: C~_a = C_a e^{i n_a omega delta}.

    ``delta`` = (arrival time) - (the tref the coefficients were referenced to), in seconds.
    It may be a scalar or an ndarray broadcastable against the entries of ``C``.

    Why this exists: the bank is built from chi_a(u) = e^{i n Omega u} h^{(p)}(u) on the
    template's INTRINSIC time u, while the physical response modulation is e^{i n Omega
    (t'-tref)} on absolute time.  Placing the template at arrival time t gives t' = u + t, so
    the modulation factorizes as e^{i n Omega u} * e^{i n Omega (t-tref)}; the second factor
    belongs to the coefficient.  Apply it to BOTH the data term and the model norm, or
    lnL = <d|h> - (1/2)<h|h> is evaluated for two different h and can exceed (1/2)<d|d>.
    """
    return {a: c * np.exp(1.0j * a[1] * omega * delta) for a, c in C.items()}


def require_post_phase_bank(meta, where):
    """Refuse a bank that does not declare the post-phase convention (see rotation_post_phase).

    ``meta['post_phase_required']`` marks a bank whose Q is <chi_a(.-t)|d> against UNTOUCHED
    data, so the evaluator owes the arrival-time post-phase on BOTH the data term and the
    model norm.  A bank from the previous revision instead pushed the modulation onto the
    DATA and carries no such debt: post-phasing it produces finite, silently WRONG lnL rather
    than an error, so check the marker rather than assume it.  Same guard as
    jax_ile.banded.build_rotation_data / jax_ile.core._accumulate_unit_banded.
    """
    if not bool(meta.get('post_phase_required', False)):
        raise ValueError(
            "%s requires meta['post_phase_required'] == True: this evaluator applies the "
            "arrival-time post-phase (rotation_post_phase) to both the data term and the "
            "model norm, which is only correct for a bank built in that convention.  Got "
            "meta['post_phase_required']=%r.\n"
            "That key is set by PrecomputeLikelihoodTermsWithRotation as of PR #117.  A bank "
            "from the earlier revision folded the modulation into the data instead and must "
            "NOT be evaluated here -- regenerate it with the current "
            "PrecomputeLikelihoodTermsWithRotation rather than hand-assembling meta."
            % (where, meta.get('post_phase_required')))


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
    require_post_phase_bank(meta, 'FactoredLogLikelihoodWithRotation')

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
        # Arrival-time post-phase (see rotation_post_phase): delta = t_arrival - tref is just
        # the geometric delay here, taken directly rather than as a difference of two ~1e9 s.
        delta_arr = float(lal.TimeDelayFromEarthCenter(
            FL.lalsim.DetectorPrefixToLALDetector(det).location, RA, DEC, tref))
        C = rotation_post_phase(C, 2.0 * np.pi * meta['f_sidereal'], delta_arr)
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

    Issue #142: this is the gateway to the NoLoop, whose Cg/Cg_d return zero for a response
    coefficient with no band.  A bank built with widen_harmonics=False can be missing bands,
    so say so HERE -- at the point the bank becomes a likelihood -- rather than let the
    evaluator drop them quietly.  (The precompute's default widens, so this never fires for
    a caller who did not opt out.)
    """
    if meta.get('harmonics_truncated'):
        warnings.warn(
            "pack_rotation_arrays: this bank was built with widen_harmonics=False and "
            "carries harmonics=%s, which is narrower than the |ntilde| <= 2 + p_max = %s "
            "the response coefficients populate at p_max=%s.  The NoLoop will evaluate a "
            "TRUNCATED model (missing coefficients contribute zero), silently.  Rebuild "
            "the bank with widen_harmonics=True unless the truncation is deliberate."
            % (meta.get('harmonics'), meta.get('harmonics_required'), meta.get('p_max')),
            RuntimeWarning, stacklevel=2)
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
    require_post_phase_bank(
        meta, 'DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation')

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
        # NOTE: this file previously had NO validation, so an unknown time_interp
        # silently executed the cubic branch below.  Gate it.  (An earlier revision of this
        # comment also said 'sinc' was rejected on GPU; that stopped being true when
        # Q_inner_sinc landed -- all three stencils now have both backends.)
        FL.validate_time_interp(time_interp, on_gpu=not (xpy is np))
        sample_first = (t_det + float(tvals[0])) / P_vec.deltaT   # float(): tvals may be a cupy array on GPU
        if time_interp == 'nearest':
            ifirst = (np.round(sample_first) + 0.5).astype(int)
        else:
            ifirst = np.floor(sample_first).astype(int)
            frac_first = (sample_first - np.floor(sample_first)).astype(np.float64)
        ilast = ifirst + npts

        # ---- arrival-time post-phase (see rotation_post_phase) ----
        # Output sample j of extrinsic sample i is the template placed at arrival time
        # t_ref + (samp0_i + j)*deltaT, so delta_ij = (samp0_i + j)*deltaT - off with
        # off = tref - t_ref.  That SEPARATES, so no (npts_ex, npts) phase array is ever
        # materialized: exp(i m omega delta_ij) = pe_m[i] * pt_m[j].
        off = float(P_vec.tref - float(t_ref))
        samp0 = ifirst.astype(np.float64) if time_interp == 'nearest' else sample_first
        delta0 = samp0 * P_vec.deltaT - off                       # (npts_ex,)
        jgrid = np.arange(npts) * P_vec.deltaT                    # (npts,)
        omega_sid = 2.0 * np.pi * meta['f_sidereal']
        _ph_cache = {}

        def _ph(m):
            """exp(i m omega_sid delta_ij) as rank-1 factors (pe (npts_ex,), pt (npts,))."""
            if m not in _ph_cache:
                if m == 0:
                    _ph_cache[m] = (None, None)      # identity; callers skip the multiply
                else:
                    _ph_cache[m] = (xpy.asarray(np.exp(1.0j * m * omega_sid * delta0)),
                                    xpy.asarray(np.exp(1.0j * m * omega_sid * jgrid)))
            return _ph_cache[m]

        # Device-side arrays for the heavy contraction (identity on CPU; host->device on GPU).
        Ylms_d = xpy.asarray(Ylms); conjY_d = xpy.conj(Ylms_d)
        zero_d = xpy.zeros(npts_ex, dtype=complex)
        C_d = {k: xpy.asarray(v) for k, v in C.items()}
        Cg_d = lambda a: C_d[a] if a in C_d else zero_d

        def _apply_post_phase(a, coef_ex, res):
            """conj(C~_a) Q^a  =  conj(C_a) e^{-i n_a omega delta_ij} Q^a_ij."""
            pe, pt = _ph(-a[1])
            if pe is None:
                return coef_ex[:, None] * res
            return (coef_ex * pe)[:, None] * (pt[None, :] * res)

        term1 = xpy.zeros((npts_ex, npts), dtype=np.complex128)
        if on_gpu:
            # term1 = Re[ sum_a conj(C~_a) sum_lm conj(Ylm) Q^a_lm(t) ]: reuse the baseline fused
            # kernel per elementary template a (A = conj(Ylm)), no (n_ex,npts,n_lms) temporary.
            ifirst_i32 = xpy.asarray(ifirst).astype(np.int32)
            frac_d = None if time_interp == 'nearest' else xpy.asarray(frac_first)
            for a in a_list:
                Q = xpy.ascontiguousarray(rho_by_a[det][a].T)   # (n_time, n_lms), device
                res = FL._q_inner_product_gpu(Q, conjY_d, ifirst_i32, frac_d, npts, time_interp)
                term1 += _apply_post_phase(a, xpy.conj(Cg_d(a)), res)
        else:
            for a in a_list:
                det_rho = rho_by_a[det][a]
                if time_interp == 'nearest':
                    Qa = np.empty((npts_ex, npts, n_lms), dtype=np.complex128)
                    for i in range(npts_ex):
                        Qa[i] = det_rho[..., ifirst[i]:ilast[i]].T
                else:
                    # sub-sample interpolation; the helpers expect Q_block shape (n_time, n_lm).
                    Qa = FL._q_window_numpy_interp(det_rho.T, ifirst, frac_first, npts,
                                                  time_interp)
                term1 += _apply_post_phase(a, np.conj(Cg(a)),
                                           np.einsum('xi,xti->xt', np.conj(Ylms), Qa))
        term1 = term1.real * inv_dist[:, None]

        # term2 also carries the post-phase, and it enters ONLY through m = n_a' - n_a for both
        # the U contraction (conj(C~_a) C~_a') and the V one (C~_{(p,-n_a)} C~_a').  So bucket
        # the |a_list|^2 einsums -- unchanged in cost -- by m, and pay one rank-1 phase per
        # distinct m (4*n_harmonics+1 of them, so 4*(2+p_max)+1 at the default width)
        # instead of one per pair.
        term2_by_m = {}
        for a in a_list:
            aR = (a[0], -a[1])
            for ap in a_list:
                val = xpy.conj(Cg_d(a)) * Cg_d(ap) * xpy.einsum(
                    'xi,xj,ij->x', conjY_d, Ylms_d, xpy.asarray(U_by_aa[det][(a, ap)]))
                val = val + Cg_d(aR) * Cg_d(ap) * xpy.einsum(
                    'xi,xj,ij->x', Ylms_d, Ylms_d, xpy.asarray(V_by_aa[det][(a, ap)]))
                m = ap[1] - a[1]
                term2_by_m[m] = term2_by_m[m] + val if m in term2_by_m else val
        # Re[] is linear, so accumulate the real part per m and keep the persistent array real.
        term2 = xpy.zeros((npts_ex, npts), dtype=np.float64)
        for m, val in term2_by_m.items():
            pe, pt = _ph(m)
            if pe is None:
                term2 += val.real[:, None]
            else:
                term2 += ((val * pe)[:, None] * pt[None, :]).real
        term2 = (-0.25 * term2) * (inv_dist ** 2)[:, None]

        lnL_t += term1 + term2

    if array_output:
        return lnL_t
    lnLmax = xpy.max(lnL_t, axis=-1, keepdims=True)
    simps = FL.optimized_gpu_tools.simps if on_gpu else FL.my_simps
    L = simps(xpy.exp(lnL_t - lnLmax), dx=P_vec.deltaT, axis=-1)
    return lnLmax[..., 0] + xpy.log(L)
