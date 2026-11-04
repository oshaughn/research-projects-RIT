"""Early-time multirate precompute by data-side weighting (DSWc). Opt-in; nothing calls it by default.

Design, evidence and acceptance tests: DESIGN_early_time_multirate.md (this directory).

Conventions follow ComputeModeIPTimeSeries (factored_likelihood.py): lal FFTs carry dt and df, and
two-sided frequency arrays are in centred order.  With dbar = IFFT(d~ * weights2side), the overlap
time series that function returns is

    Q(n dt) = 2 dt sum_s dbar[s] conj(h[(s - n) mod N]),

so the data-side weighting can be applied once per job and the template enters raw.
"""
import sys

import numpy as np
import lal

import RIFT.lalsimutils as lsu

__all__ = ["data_side_weighted", "overlap_series_dsw", "ComputeModeIPTimeSeriesDSW", "ComputeModeCrossTermIPDSW"]


def data_side_weighted(data, psd, fmin, fMax, fNyq, analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.):
    """dbar(t) = IFFT(d~ * weights2side): the data weighted by RIFT's own inner-product weights.

    `data` is the two-sided COMPLEX16FrequencySeries ILE loads; the weights are those of the
    ComplexOverlap that ComputeModeIPTimeSeries builds, so truncation settings carry over.
    Returns a complex numpy array of length N on the data's time grid (starts at data.epoch).
    """
    IP = lsu.ComplexOverlap(fmin, fMax, fNyq, data.deltaF, psd, analyticPSD_Q, inv_spec_trunc_Q, T_spec,
                            full_output=True)
    assert data.data.length == IP.len2side
    w = lal.CreateCOMPLEX16FrequencySeries("dbar(f)", data.epoch, data.f0, data.deltaF,
                                           lsu.lsu_HertzUnit, data.data.length)
    w.data.data = data.data.data * IP.weights2side
    return np.array(lsu.DataInverseFourier(w).data.data)


def overlap_series_dsw(dbar, h, dt, lags):
    """2 dt sum_s dbar[s] conj(h[(s - n) mod N]) at integer lags n (any sign), circular as in RIFT."""
    N = len(dbar)
    assert len(h) == N
    c = np.fft.ifft(np.fft.fft(dbar) * np.conj(np.fft.fft(h)))
    return 2. * dt * c[np.asarray(lags) % N]


def ComputeModeIPTimeSeriesDSW(hlms, dbar, data_epoch, deltaT, N_shift, N_window):
    """One-segment, full-rate counterpart of ComputeModeIPTimeSeries from precomputed dbar.

    Returns {mode: COMPLEX16TimeSeries} with the same samples and epoch as ComputeModeIPTimeSeries
    (acceptance test 1). `hlms` are the two-sided FD modes PrecomputeLikelihoodTerms passes.
    """
    rholms = {}
    lags = N_shift + np.arange(N_window)
    for pair, hf in hlms.items():
        h = np.array(lsu.DataInverseFourier(hf).data.data)
        q = overlap_series_dsw(dbar, h, deltaT, lags)
        ts = lal.CreateCOMPLEX16TimeSeries("rho", data_epoch - hf.epoch, 0., deltaT,
                                           lsu.lsu_DimensionlessUnit, N_window)
        ts.epoch += N_shift * deltaT
        ts.data.data = q
        rholms[pair] = ts
    return rholms


# ---------------------------------------------------------------------------------------------
# Two-rate Q (DSWc).  Windows, filter and low-rate correlation follow session A's prototype
# (RIFT_roboto_paper analyses/early_time_compression/prototype/twrate.py, experiment.py).
# Index alignment uses ILE's template convention (fd_alignment_postevent_time=2, power-of-2
# buffers): every template puts its peak 2 s before its buffer end, so coarse sample j of the
# early template is full-rate index j*M, and sample j of a late template of length N_l is
# full-rate index j + N - N_l.
# ---------------------------------------------------------------------------------------------
from scipy import signal as _signal, special as _special, fft as _sp_fft

ERFC_HALF = 8.3            # erfc(x/sqrt2)/2 < 1e-16 beyond 8.3 sigma
GUARD_PER_SIGMA = 2 * ERFC_HALF
G_MSUN_C3 = 4.925490947641267e-06


def early_window(t, t_tr, sigma):
    return 0.5 * _special.erfc((np.asarray(t) - t_tr) / (np.sqrt(2.) * sigma))


def f22_newtonian(tau, mc_msun):
    return (1. / np.pi) * (5. / 256.)**0.375 * (mc_msun * G_MSUN_C3)**(-0.625) * np.asarray(tau, float)**(-0.375)


def kaiser_lowpass(fs, f_pass, f_stop, atten_db):
    ntap = int(np.ceil((atten_db - 7.95) * fs / (14.36 * (f_stop - f_pass)))) + 1
    ntap += 1 - ntap % 2                       # odd length: integer group delay
    return _signal.firwin(ntap, 0.5 * (f_pass + f_stop), window=('kaiser', _signal.kaiser_beta(atten_db)), fs=fs)


def decimate(x, M, h):
    """Zero-phase FIR, then every M-th sample (index 0 kept)."""
    if M == 1:
        return x.copy()
    if np.iscomplexobj(x):
        return _signal.resample_poly(x.real, 1, M, window=h) + 1j * _signal.resample_poly(x.imag, 1, M, window=h)
    return _signal.resample_poly(x, 1, M, window=h)


def taper_top_quarter(hf, fs_e):
    """DSWr: cos^2 roll-off over the top quarter below fs_e/2, on a two-sided centred FD series (in place)."""
    n = hf.data.length
    f = (np.arange(n) - n // 2) * hf.deltaF
    x = np.clip((np.abs(f) - 0.75 * fs_e / 2) / (0.25 * fs_e / 2), 0, 1)
    hf.data.data = hf.data.data * np.cos(0.5 * np.pi * x)**2
    return hf


def _corr_offsets(x, x0, y, y0, lags):
    """sum_s x[s-x0] conj(y[s-n-y0]) for integer lags n, linear (zero-padded)."""
    L = _sp_fft.next_fast_len(len(x) + len(y) + 8)
    z = np.fft.ifft(np.fft.fft(x, L) * np.conj(np.fft.fft(y, L)))
    return z[(np.asarray(lags) - x0 + y0) % L]


def _corr_lowrate_fulllags(De, He, M, dt, lags):
    """M sum_j De[j] conj(He_tau[j]) at full-rate lags tau = n dt, via M sub-phase inverse FFTs.
    He_tau is the band-limited shift of He; exact for inputs band-limited below fs_e/2."""
    lags = np.asarray(lags)
    L = _sp_fft.next_fast_len(max(len(De), len(He)) + int(np.max(np.abs(lags))) // M + 8)
    C = M * np.fft.fft(De, L) * np.conj(np.fft.fft(He, L))
    fk = np.fft.fftfreq(L, d=M * dt)
    q, r = np.divmod(lags, M)
    out = np.empty(len(lags), dtype=np.complex128)
    for rr in np.unique(r):
        sel = (r == rr)
        out[sel] = np.fft.ifft(C * np.exp(2j * np.pi * fk * rr * dt))[q[sel] % L]
    return out


def _corr_lowrate_interp(De, He, M, dt, lags, h_interp):
    """As _corr_lowrate_fulllags, from one inverse FFT at coarse lags and band-limited
    interpolation to the full-rate lags (zero-stuff by M, filter h_interp, odd length).
    Exact to the filter's stop-band attenuation when the correlation is band-limited below
    the filter's pass band."""
    lags = np.asarray(lags)
    L = _sp_fft.next_fast_len(max(len(De), len(He)) + int(np.max(np.abs(lags))) // M + 8)
    c = np.fft.ifft(M * np.fft.fft(De, L) * np.conj(np.fft.fft(He, L)))
    D = (len(h_interp) - 1) // 2
    P = D // M + 1
    q = np.arange(int(np.min(lags)) // M - P, int(np.max(lags)) // M + P + 2)
    k = lags[:, None] - M * q[None, :] + D          # filter tap for (lag, coarse sample)
    ok = (k >= 0) & (k < len(h_interp))
    W = np.where(ok, h_interp[np.clip(k, 0, len(h_interp) - 1)], 0.)
    return W @ c[q % L]


XHM_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))     # IMRPhenomXHM/XPHM; lalsim rejects others


def mode_array_for_m(lmax, am, model_modes=XHM_MODES):
    """lalsimulation ModeArray holding the model's (l, +-am) with l <= lmax: one per-|m| call.
    Activating a mode the model lacks makes ChooseFDModes fail, so only model_modes are used."""
    import lalsimulation as lalsim
    ma = lalsim.SimInspiralCreateModeArray()
    for (l, m) in model_modes:
        if l <= lmax and m == am:
            lalsim.SimInspiralModeArrayActivateMode(ma, l, am)
            lalsim.SimInspiralModeArrayActivateMode(ma, l, -am)
    return ma


class TwoRateSchedule(object):
    """Frozen two-rate schedule: full rate fs, early rate fs_e, transition tau_tr before the
    peak, guard (erfc width sigma = guard/16.6), template window sigma_h, data-side filter."""

    def __init__(self, fs, fs_e, tau_tr, guard, atten_db, mc_min, m_max, lag_halfwidth, sigma_h=1.0,
                 late_buffer=None):
        self.fs, self.fs_e = float(fs), float(fs_e)
        self.M = int(round(fs / fs_e))
        assert self.M * fs_e == fs
        self.tau_tr, self.guard, self.atten_db = float(tau_tr), float(guard), float(atten_db)
        self.sigma, self.sigma_h = guard / GUARD_PER_SIGMA, float(sigma_h)
        self.mc_min, self.m_max = float(mc_min), int(m_max)
        self.nlag = int(np.ceil(lag_halfwidth * fs))
        self.ng = int(np.ceil(ERFC_HALF * self.sigma * fs))
        # template window w_h: 1 on the data's early support plus filter, lags and one coarse sample
        nL0 = 0
        for _ in range(3):                     # the filter length depends on f_pass_h; iterate
            self.tau_hc = tau_tr - (self.ng + nL0 + self.nlag + self.M) / fs - ERFC_HALF * self.sigma_h
            tau_min_h = self.tau_hc - ERFC_HALF * self.sigma_h
            self.f_pass_h = 1.1 * (m_max / 2.) * f22_newtonian(tau_min_h, mc_min) + 1.4 / self.sigma_h
            # the early template is tapered (DSWr) over the top quarter below fs_e/2, so the
            # template window's pass band must stay below 0.75 fs_e/2
            if not np.isfinite(self.f_pass_h) or self.f_pass_h >= 0.75 * fs_e / 2:
                raise ValueError("template window needs %.1f Hz, above the DSWr taper start %.1f Hz; "
                                 "raise fs_e" % (self.f_pass_h, 0.75 * fs_e / 2))
            self.h = kaiser_lowpass(fs, self.f_pass_h, fs_e / 2, atten_db)
            nL0 = len(self.h)
        # lag interpolation: the early correlation is band-limited below f_pass_h, so its images
        # at coarse sampling start at fs_e - f_pass_h; upsample by M with this filter
        self.h_interp = (np.ones(1) if self.M == 1 else
                         self.M * kaiser_lowpass(fs, self.f_pass_h, fs_e - self.f_pass_h, atten_db))
        self.late_buffer = late_buffer


def prepare_data_two_rate(dbar, deltaT, s_peak, sch):
    """Once per job and detector: split dbar at the transition and decimate the early part.

    s_peak: data index the template peak sits at for lag 0 (template and data indices aligned).
    Returns dict with dE (early, rate fs_e, index j <-> data index j*M), dL (late, full rate,
    starting at data index i_l0), and the indices used.
    """
    N, M = len(dbar), sch.M
    s_tr = s_peak - int(round(sch.tau_tr * sch.fs))
    s = np.arange(N)
    i_de = min(N, s_tr + sch.ng + len(sch.h))
    we = early_window(s[:i_de] * deltaT, s_tr * deltaT, sch.sigma)
    i_l0 = max(0, s_tr - sch.ng - sch.nlag - 8)
    wl = 1. - early_window(s[i_l0:] * deltaT, s_tr * deltaT, sch.sigma)
    dE = decimate(we * dbar[:i_de], M, sch.h)
    dL = wl * dbar[i_l0:]
    return dict(dE=dE, dL=dL, i_l0=i_l0, i_de=i_de, s_tr=s_tr, s_peak=s_peak, N=N)


def early_template_window(n_coarse, j_peak_full, sch, deltaT):
    """w_h on the early template's coarse grid (index j <-> full index j*M)."""
    t = np.arange(n_coarse) * sch.M * deltaT
    t_hc = (j_peak_full - int(round(sch.tau_hc * sch.fs))) * deltaT
    return early_window(t, t_hc, sch.sigma_h)


def Q_two_rate(prep, hE, hL, j_peak_full, n_late_offset, sch, deltaT, lags, early_method="interp"):
    """Q(n dt) = 2 dt [ M sum_j dE[j] conj(w_h hE)_tau[j] + sum_s dL[s] conj(hL[s - n - off]) ].

    hE: early template on the coarse grid (length N/M), DSWr-tapered.  hL: late template at full
    rate (length N_l), sample j at full index j + n_late_offset.  Template index = data index - n.
    early_method: "interp" (one coarse-lag FFT, band-limited lag interpolation) or "subphase"
    (M inverse FFTs, the reference).
    """
    wh = early_template_window(len(hE), j_peak_full, sch, deltaT)
    if early_method == "subphase":
        qE = _corr_lowrate_fulllags(prep["dE"], wh * hE, sch.M, deltaT, lags)
    else:
        qE = _corr_lowrate_interp(prep["dE"], wh * hE, sch.M, deltaT, lags, sch.h_interp)
    qL = _corr_offsets(prep["dL"], prep["i_l0"], hL, n_late_offset, lags)
    return 2. * deltaT * (qE + qL)


def ComputeModeCrossTermIPDSW(hlmsA, hlmsB, psd, fmin, fMax, fNyq, deltaF, analyticPSD_Q=False,
                              inv_spec_trunc_Q=False, T_spec=0., prefix="U"):
    """One-segment counterpart of ComputeModeCrossTermIP in the time domain:
    <a|b> = 2 dt sum_t conj(a(t)) bbar(t), bbar = IFFT(b~ weights2side), for every ordered pair.
    The reference the multibanded U, V are judged against (acceptance test 1b)."""
    IP = lsu.ComplexIP(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
    tA = {k: np.array(lsu.DataInverseFourier(v).data.data) for k, v in hlmsA.items()}
    tB = {}
    for k, v in hlmsB.items():
        w = lal.CreateCOMPLEX16FrequencySeries("bbar", v.epoch, v.f0, v.deltaF, lsu.lsu_HertzUnit, v.data.length)
        w.data.data = v.data.data * IP.weights2side
        tB[k] = np.array(lsu.DataInverseFourier(w).data.data)
    dt = 1. / (2. * fNyq)
    return {(a, b): 2. * dt * np.vdot(tA[a], tB[b]) for a in tA for b in tB}


# ---------------------------------------------------------------------------------------------
# Multibanded U, V (Vinciguerra et al. 2017), band schedule after session A's multiband_UV.
# Band j keeps every K_j-th bin of the full grid; K_j is the largest power of two with
# T/K_j >= T_j, T_j = (1 + kappa) tau_N(2 f_eff/m_max; Mc_min) + post + extra_s.
# Smooth erfc partition of unity in |f|; full-resolution bands of width edge_hz at hard edges.
# ---------------------------------------------------------------------------------------------
def multiband_schedule(flow, fhigh, T, mc_min, m_max, kappa=0.2, extra_s=32.0, sigma_f=1.0, edge_hz=4.0,
                       post=1.0, top_edge_K=1):
    """top_edge_K: bin stride in the hard-edge band below fhigh.  1 keeps every bin; the pieces
    U, V need the late buffer's stride there (measured 2026-10-09: stride 32 changes U by at most
    1.1e-7 of max |U|, O4 256 s and CE 2048 s, XHM l <= 4)."""
    tauN = lambda f22: (5. / 256.) * (np.pi * f22)**(-8. / 3) * (mc_min * G_MSUN_C3)**(-5. / 3)
    top = fhigh - edge_hz
    edges = [flow, flow + edge_hz]
    while edges[-1] < top:
        edges.append(edges[-1] * 2**0.375)
    edges[-1] = top
    bands = [(flow, flow + edge_hz, 1)]
    for j in range(1, len(edges) - 1):
        f_eff = max(flow, edges[j] - 8.3 * sigma_f)
        Tj = (1 + kappa) * tauN(2 * f_eff / m_max) + post + extra_s
        K = 1
        while T / (2 * K) >= Tj:
            K *= 2
        bands.append((edges[j], edges[j + 1], K))
    bands.append((top, fhigh + 1.0, int(top_edge_K)))
    return dict(bands=bands, sigma_f=sigma_f)


def _band_window(sched, j, af):
    bands, sf = sched["bands"], sched["sigma_f"]

    def step(i):
        if i == 0:
            return (af >= bands[0][0]).astype(float)
        if i >= len(bands):
            return np.zeros(len(af))
        return 0.5 * _special.erfc(-(af - bands[i][0]) / (np.sqrt(2) * sf))
    return step(j) - step(j + 1)


def ComputeModeCrossTermIPMultiband(hlmsA, hlmsB, psd, fmin, fMax, fNyq, deltaF, sched, analyticPSD_Q=False,
                                    inv_spec_trunc_Q=False, T_spec=0.):
    """<a|b> = 2 df sum_f conj(a~) b~ w on the multibanded grid, from two-sided centred FD series.

    Emulation step: the FD values are subsampled from the full grid, which tests the band schedule;
    evaluating the model only at the kept frequencies is a separate step.  Pass the conjugate modes
    as hlmsA for V, as PrecomputeLikelihoodTerms does.  Returns (dict, number of kept bins)."""
    IP = lsu.ComplexIP(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
    w = IP.weights2side
    n = len(w)
    kf = np.arange(n) - n // 2                 # centred bin index: f = kf * deltaF
    af = np.abs(kf) * deltaF
    out = {(a, b): 0j for a in hlmsA for b in hlmsB}
    nkept = 0
    for j, (lo, hi, K) in enumerate(sched["bands"]):
        sel = np.nonzero(kf % K == 0)[0]
        g = _band_window(sched, j, af[sel]) * w[sel] * K
        keep = np.abs(g) > 0
        sel, g = sel[keep], g[keep]
        nkept += int(np.sum(kf[sel] >= 0))
        A = {k: v.data.data[sel] for k, v in hlmsA.items()}
        B = {k: v.data.data[sel] for k, v in hlmsB.items()}
        for a in A:
            for b in B:
                out[(a, b)] += 2. * deltaF * np.sum(np.conj(A[a]) * B[b] * g)
    return out, nkept


def pieces_uv_plan(psd, fmin, fMax, fNyq, deltaF, sched, fs_e, f22_late, n_early, n_late, mode_keys,
                   analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.):
    """Once per job (depends only on the PSD, the schedule and the template grid sizes): the kept
    multibanded bins and their weights, and for each |m| the gather indices that read each bin from
    the early FD series (rate fs_e, full duration, no DSWr taper) below the switch or from the late
    FD series (short buffer T/R) above it.  A late bin kf (full-grid units) needs kf % R == 0; there
    the buffer phase factor is exp(2 pi i kf / R) = 1, so the two sources combine without
    correction.  Mode |m| switches to the late series at 1.05 (|m|/2) f22_late, above its
    start-frequency conditioning; the early series is used up to 0.9 fs_e/2.  Raises if a kept bin
    has no source.  Pass the result to ComputeModeCrossTermsPieces for every intrinsic point."""
    IP = lsu.ComplexIP(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
    w = IP.weights2side
    n = len(w)
    R = int(round(n / n_late))
    assert R * n_late == n
    nb = sched["bands"]
    kfs, gs = [], []
    for j, (lo, hi, K) in enumerate(nb):
        # only the bins inside this band's support (erfc edges cut at 12 sigma, < 1e-32):
        # no pass over the full grid per band
        f_lo = max(0., lo - 12 * sched["sigma_f"]) if j > 0 else lo
        f_hi = (nb[j + 1][0] + 12 * sched["sigma_f"]) if j + 1 < len(nb) else fNyq
        k_lo, k_hi = int(np.ceil(f_lo / deltaF / K)) * K, int(np.floor(min(f_hi, fNyq) / deltaF / K)) * K
        kpos = np.arange(k_lo, k_hi + 1, K)
        kpos = kpos[kpos < n // 2]
        kf = np.concatenate([kpos, -kpos[kpos > 0]])
        g = _band_window(sched, j, np.abs(kf) * deltaF) * w[kf + n // 2] * K
        keep = np.abs(g) > 0
        kfs.append(kf[keep]); gs.append(g[keep])
    kf, g = np.concatenate(kfs), np.concatenate(gs)
    af = np.abs(kf) * deltaF
    f_e_max = 0.9 * fs_e / 2.
    gather = {}
    for am in sorted({abs(k[1]) for k in mode_keys}):
        use_late = (af >= 1.05 * (am / 2.) * f22_late) & (kf % R == 0)
        use_early = ~use_late & (af <= f_e_max)
        if not np.all(use_late | use_early):
            bad = af[~(use_late | use_early)]
            raise ValueError("|m| = %d: %d kept bins (%.1f-%.1f Hz) have neither an early nor a late value; "
                             "raise fs_e or the late buffer" % (am, len(bad), bad.min(), bad.max()))
        idx = {}
        for sgn, kk in (("pos", kf), ("neg", -kf)):
            idx[sgn] = (use_late, np.clip(kk // R + n_late // 2, 0, n_late - 1), np.clip(kk + n_early // 2, 0, n_early - 1))
        gather[am] = idx
    return dict(g=g, gather=gather, nkept=int(np.sum(kf >= 0)), n_early=n_early, n_late=n_late, deltaF=deltaF)


def ComputeModeCrossTermsPieces(early, late, plan):
    """Per point: U_ab = <h_a|h_b> and V_ab = <conj(h_a)|h_b> on the multibanded grid, from the early
    and late FD series and a plan from pieces_uv_plan.  Gathers each mode's values, then forms all
    pairs as two matrix products.  Returns (U, V, number of kept bins with f >= 0)."""
    keys = sorted(early)
    assert early[keys[0]].data.length == plan["n_early"] and late[keys[0]].data.length == plan["n_late"]

    def gather(k, sgn):
        use_late, il, ie = plan["gather"][abs(k[1])][sgn]
        return np.where(use_late, late[k].data.data[il], early[k].data.data[ie])
    A = np.array([gather(k, "pos") for k in keys])
    An = np.array([gather(k, "neg") for k in keys])
    g = plan["g"]
    Um = (np.conj(A) * g) @ A.T
    Vm = (An * g) @ A.T
    df2 = 2. * plan["deltaF"]
    U = {(a, b): df2 * Um[i, j] for i, a in enumerate(keys) for j, b in enumerate(keys)}
    V = {(a, b): df2 * Vm[i, j] for i, a in enumerate(keys) for j, b in enumerate(keys)}
    return U, V, plan["nkept"]


# ---------------------------------------------------------------------------------------------
# Per-point preparation on an array backend (cupy on a GPU, numpy otherwise).  Same arithmetic as
# Q_two_rate / ComputeModeCrossTermsPieces, which stay as the reference; the data-side FFTs, the
# lag-interpolation matrix and the U, V gather indices are built once per job and detector.
# ---------------------------------------------------------------------------------------------
def _select_xpy(choice=None):
    """numpy or cupy, chosen as factored_likelihood chooses: cupy when it imports and can allocate
    on a device.  choice 'numpy' or 'cupy' forces one."""
    if choice not in (None, "numpy", "cupy"):
        raise ValueError("multirate backend must be numpy or cupy, got %r" % choice)
    if choice == "numpy":
        return np
    try:
        import cupy
        cupy.zeros(1)
        return cupy
    except Exception:
        if choice == "cupy":
            raise
        return np


def _to_host(xp, a):
    return a.get() if xp is not np else a


def ifft_centred(xp, X, deltaF):
    """lsu.DataInverseFourier for a two-sided centred FD array: x[k] = df sum_j X[j] exp(2 pi i f_j t_k)."""
    return xp.fft.ifft(xp.fft.ifftshift(X)) * (len(X) * deltaF)


def taper_top_quarter_factor(n, deltaF, fs_e):
    """The DSWr factor taper_top_quarter applies, as an array over a centred two-sided grid."""
    f = (np.arange(n) - n // 2) * deltaF
    x = np.clip((np.abs(f) - 0.75 * fs_e / 2) / (0.25 * fs_e / 2), 0, 1)
    return np.cos(0.5 * np.pi * x)**2


class TwoRateDevice(object):
    """Once per job and detector: the data-side pieces of Q_two_rate on the backend xp."""

    def __init__(self, xp, prep, sch, deltaT, n_coarse, n_late, N, j_peak, lags, n_late_off):
        self.xp, M = xp, sch.M
        lags = np.asarray(lags)
        self.LE = _sp_fft.next_fast_len(max(len(prep["dE"]), n_coarse) + int(np.max(np.abs(lags))) // M + 8)
        self.LL = _sp_fft.next_fast_len(len(prep["dL"]) + n_late + 8)
        self.FE = M * xp.fft.fft(xp.asarray(prep["dE"]), self.LE)
        self.FL = xp.fft.fft(xp.asarray(prep["dL"]), self.LL)
        hI = sch.h_interp
        D = (len(hI) - 1) // 2
        P = D // M + 1
        q = np.arange(int(np.min(lags)) // M - P, int(np.max(lags)) // M + P + 2)
        k = lags[:, None] - M * q[None, :] + D
        ok = (k >= 0) & (k < len(hI))
        self.W = xp.asarray(np.where(ok, hI[np.clip(k, 0, len(hI) - 1)], 0.))
        self.qidx = xp.asarray(q % self.LE)
        self.lidx = xp.asarray((lags - prep["i_l0"] + n_late_off) % self.LL)
        self.wh = xp.asarray(early_template_window(n_coarse, j_peak, sch, deltaT))
        self.scale = 2. * deltaT

    def Q(self, hE_td, hL_td):
        """Q_two_rate (interp) for one mode, from device TD templates (hE tapered, coarse)."""
        xp = self.xp
        c = xp.fft.ifft(self.FE * xp.conj(xp.fft.fft(self.wh * hE_td, self.LE)))
        z = xp.fft.ifft(self.FL * xp.conj(xp.fft.fft(hL_td, self.LL)))
        return self.scale * (self.W @ c[self.qidx] + z[self.lidx])


def pieces_uv_plan_device(xp, plan):
    """pieces_uv_plan's arrays on the backend xp."""
    g = {am: {sgn: tuple(xp.asarray(a) for a in v) for sgn, v in d.items()} for am, d in plan["gather"].items()}
    return dict(plan, g_dev=xp.asarray(plan["g"]), gather_dev=g)


def ComputeModeCrossTermsPiecesDevice(xp, early_fd, late_fd, plan_dev, keys):
    """ComputeModeCrossTermsPieces from device FD arrays (dicts keyed by mode)."""
    def gather(k, sgn):
        use_late, il, ie = plan_dev["gather_dev"][abs(k[1])][sgn]
        return xp.where(use_late, late_fd[k][il], early_fd[k][ie])
    A = xp.stack([gather(k, "pos") for k in keys])
    An = xp.stack([gather(k, "neg") for k in keys])
    g = plan_dev["g_dev"]
    Um = _to_host(xp, (xp.conj(A) * g) @ A.T)
    Vm = _to_host(xp, (An * g) @ A.T)
    df2 = 2. * plan_dev["deltaF"]
    U = {(a, b): df2 * Um[i, j] for i, a in enumerate(keys) for j, b in enumerate(keys)}
    V = {(a, b): df2 * Vm[i, j] for i, a in enumerate(keys) for j, b in enumerate(keys)}
    return U, V


# ---------------------------------------------------------------------------------------------
# Opt-in precompute (DESIGN_early_time_multirate.md, "Opt-in interface").  Same arguments and
# return tuple as factored_likelihood.PrecomputeLikelihoodTerms, plus a schedule:
#   kind        "one_segment" (full rate, per-|m| calls: the matched control) or "two_rate"
#   fs_e, tau_tr, guard, atten_db, late_start, late_buffer, post, uv_margin, mc_range
# Data products (weighted data, decimated early part) are built once per job and detector.
# ---------------------------------------------------------------------------------------------
_JOB_CACHE = {}
SCHEDULE_KEYS = ("kind", "fs_e", "tau_tr", "guard", "atten_db", "late_start", "late_buffer", "post",
                 "uv_margin", "mc_range")


def _cache_context(data, psds, schedule, params, xp):
    """Bound storage to one input context, including in-place data/PSD changes.

    Content hashing costs a linear host read per point, but avoids stale likelihoods and
    recycled Python ids. Intrinsic masses/spins are deliberately absent: plans are shared
    across those points. Include the CUDA device so allocations never cross devices.
    """
    import hashlib, json

    def signature(series):
        a = np.ascontiguousarray(series.data.data)
        return (str(series.epoch), series.f0, series.deltaF, a.shape, a.dtype.str,
                hashlib.blake2b(a.view(np.uint8), digest_size=16).digest())

    device = None if xp is np else xp.cuda.runtime.getDevice()
    context = (json.dumps({k: schedule[k] for k in SCHEDULE_KEYS}, sort_keys=True),
               params, xp.__name__, device,
               tuple((det, signature(data[det]), signature(psds[det])) for det in sorted(data)))
    if _JOB_CACHE.get("context") != context:
        _JOB_CACHE.clear()
        _JOB_CACHE["context"] = context
        return True
    return False


def _validate_schedule(sch):
    """Reject malformed schedules before allocating or generating templates."""
    missing = [k for k in SCHEDULE_KEYS if k not in sch]
    if missing:
        raise ValueError("early-time multirate schedule lacks %s" % missing)
    if sch["kind"] not in ("one_segment", "two_rate"):
        raise ValueError("schedule kind must be one_segment or two_rate, not %r" % sch["kind"])
    for key in SCHEDULE_KEYS[1:-1]:
        value = sch[key]
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError("schedule %s must be finite and positive" % key)
    mc = sch["mc_range"]
    if not isinstance(mc, (list, tuple)) or len(mc) != 2 or not all(
            isinstance(v, (int, float)) and np.isfinite(v) and v > 0 for v in mc) or mc[0] > mc[1]:
        raise ValueError("schedule mc_range must be two ordered positive finite masses")
    if sch["atten_db"] <= 8:
        raise ValueError("schedule atten_db must exceed 8 dB")
    if sch["late_buffer"] < sch["late_start"] + sch["post"] + 8:
        raise ValueError("late_buffer cannot hold late_start plus post and 8 s")
    if sch["late_start"] * (mc[0] / mc[1])**(5. / 3) <= sch["tau_tr"] + sch["guard"]:
        raise ValueError("late_start is too close to tau_tr for the heaviest template")


def load_schedule(path):
    import json, hashlib
    raw = open(path, "rb").read()
    sch = json.loads(raw)
    _validate_schedule(sch)
    sch["_hash"] = hashlib.sha256(raw).hexdigest()[:12]
    sch["_path"] = path
    return sch


def _per_m_modes(P, Lmax, srate, seglen, f22_start, post):
    import lalsimulation as lalsim
    out = {}
    for am in range(1, Lmax + 1):
        if not any(l <= Lmax and m == am for (l, m) in XHM_MODES):
            continue
        Pm = P.manual_copy()
        Pm.fmin = (am / 2.) * f22_start
        Pm.deltaT, Pm.deltaF = 1. / srate, 1. / seglen
        extra = dict(PhenomXHMThresholdMband=0, PhenomXPHMThresholdMband=0, ModeArray=mode_array_for_m(Lmax, am))
        if P.approx == lalsim.IMRPhenomXPHM:
            # Restrict the output only. Restricting XPHM's co-precessing inputs
            # changes even aligned-spin modes (e.g. (2,1) by 3% at zero spin).
            # Its unrestricted J-frame output also requests absent l values and
            # fails if only the per-|m| co-precessing inputs were supplied.
            del extra["ModeArray"]
            extra["ModeArrayJframe"] = mode_array_for_m(Lmax, am)
        hF, _ = lsu.std_and_conj_hlmoff(Pm, Lmax=Lmax, fd_alignment_postevent_time=post, extra_waveform_args=extra)
        out.update({k: v for k, v in hF.items() if abs(k[1]) == am})
    return out


class _DeviceModeSeries(object):
    """The fields of a lal FD series the device path reads, plus the device array."""

    def __init__(self, dev, epoch, deltaF):
        self.dev, self.epoch, self.deltaF = dev, epoch, deltaF
        self.data = type("data", (), {"length": int(dev.shape[-1])})()


def _per_m_modes_xp(xp, P, Lmax, srate, seglen, f22_start, post, fd_standoff_factor=0.964,
                    fd_end_taper_max_time=1.0):
    """_per_m_modes for IMRPhenomXHM on xp: the modes come from RIFT.likelihood.gpu_waveform_xhm
    (equal to ChooseFDModes with PhenomXHMThresholdMband=0) and are conditioned on xp exactly as
    lsu.hlmoft conditions ChooseFDModes output (high-pass, centring roll, start and capped end
    tapers), then Fourier transformed as lsu.DataFourier does.  Returns {(l, m): _DeviceModeSeries}."""
    import lalsimulation as lalsim
    from RIFT.likelihood import gpu_waveform_xhm as XHM
    dt, df = 1. / srate, 1. / seglen
    n = int(round(srate * seglen))
    centring = 1 - post / seglen
    m1, m2 = P.m1 / lal.MSUN_SI, P.m2 / lal.MSUN_SI
    s = lalsim.SimInspiralFinalBlackHoleSpinBound(P.s1z, P.s2z)
    t_merge = lalsim.SimInspiralMergeTimeBound(P.m1, P.m2) + lalsim.SimInspiralRingdownTimeBound(P.m1 + P.m2, s)
    epoch = -(n * dt) * centring
    out = {}
    for am in range(1, Lmax + 1):
        modes = [(l, sg * am) for (l, m) in XHM_MODES if m == am and l <= Lmax for sg in (1, -1)]
        if not modes:
            continue
        fmin = (am / 2.) * f22_start
        H = XHM.imrphenomxhm_modes(np.array([m1]), np.array([m2]), np.array([P.s1z]), np.array([P.s2z]), P.dist,
                                   np.array([P.phiref]), fmin * fd_standoff_factor, P.fref, df, n // 2 + 1,
                                   f_max=srate / 2., modes=modes, xp=xp)[0]
        # hlmoft: high-pass on |f| (LAL's ascending grid), then resize to n, zeroing the -fNyq bin
        f = np.abs(df * (np.arange(n + 1) - (n + 1) // 2))
        hp = np.where(f < fmin, 0.5 + 0.5 * np.cos(np.pi * (f / fmin - 1) / (1 - fd_standoff_factor)), 1.)
        hp[f < fmin * fd_standoff_factor] = 0.
        hp[0] = 0.
        H = H[:, :n] * xp.asarray(hp[:n])
        h = -xp.fft.ifft(xp.fft.ifftshift(H, axes=-1), axis=-1) * (n * df)
        h = xp.roll(h, -int(n * (1 - centring)), axis=-1)
        ntaper = max(int(0.01 * n), int(1. / (fmin * dt)))
        ntaper_end = int(min(ntaper, max(int(fd_end_taper_max_time / dt), 1)))
        w = np.ones(n)
        w[:ntaper] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ntaper) / (1. * ntaper))
        vend = (0.5 - 0.5 * np.cos(np.pi * np.arange(ntaper_end) / (1. * ntaper_end)))[::-1]
        crit = int(n * centring) + int((t_merge + 3. / fmin) / dt)
        if crit + ntaper_end > n:
            crit = n - ntaper_end
        if n - ntaper_end < crit:
            w[n - ntaper_end:] *= vend
        else:
            w[crit:crit + ntaper_end] *= vend
            w[crit + ntaper_end:] = 0.
        X = xp.fft.fftshift(xp.fft.fft(h * xp.asarray(w), axis=-1), axes=-1) * dt
        for i, k in enumerate(modes):
            out[k] = _DeviceModeSeries(X[i], epoch, df)
    return out


def _refuse(P, schedule, kwargs):
    bad = [k for k in ("calibration_realizations", "NR_group", "ROM_group") if kwargs.get(k) is not None]
    bad += [k for k in ("ROM_use_basis", "use_gwsignal", "use_external_EOB", "nr_lookup", "hybrid_use",
                        "use_provided_strain", "analyticPSD_Q", "force_22_mode",
                        "perturbative_extraction", "perturbative_extraction_full", "extra_waveform_kwargs") if kwargs.get(k)]
    if bad:
        raise ValueError("early-time multirate refuses %s" % bad)
    _refuse_point(P, schedule)


def _refuse_point(P, schedule):
    import lalsimulation as lalsim
    if P.approx not in (lalsim.IMRPhenomXHM, lalsim.IMRPhenomXPHM):
        raise ValueError("early-time multirate supports IMRPhenomXHM/XPHM only, not %s"
                         % lalsim.GetStringFromApproximant(P.approx))
    if max(abs(P.s1x), abs(P.s1y), abs(P.s2x), abs(P.s2y)) > 0:
        raise ValueError("early-time multirate refuses in-plane spins: per-|m| start frequencies "
                         "assume aligned spins")
    mc = (P.m1 * P.m2)**0.6 / (P.m1 + P.m2)**0.2 / lal.MSUN_SI
    lo, hi = schedule["mc_range"]
    if not (lo <= mc <= hi):
        raise ValueError("intrinsic point Mc = %.5f outside the schedule's frozen range [%g, %g]" % (mc, lo, hi))


PROTOTYPE_WARNING = ("early-time multirate is a PROTOTYPE, NOT READY FOR HIGH-SNR USE: its template "
                     "conventions (per-|m| f_min, XHM multibanding off) bias lnL by up to ~1e-3 of "
                     "rho^2/2 against the LAL-closest convention; see DESIGN_early_time_multirate.md")


def validate_job(schedule_path, P_list, Lmax, backend=None, waveform=None):
    """Job-level checks, run by ILE before its event loop.  A refusal inside the precompute is
    caught by ILE's per-point handler, which logs it and exits 0 with no output; this runs first
    so a refused configuration exits nonzero instead.  Returns the loaded schedule."""
    print("WARNING: " + PROTOTYPE_WARNING, file=sys.stderr)
    schedule = load_schedule(schedule_path)
    if Lmax not in (2, 3, 4):
        raise ValueError("early-time multirate requires Lmax = 2, 3 or 4, not %s" % Lmax)
    if backend not in (None, "numpy", "cupy", "reference"):
        raise ValueError("multirate backend must be numpy, cupy or reference, got %r" % backend)
    if waveform not in (None, "lal", "xp"):
        raise ValueError("multirate waveform must be lal or xp, got %r" % waveform)
    if backend == "cupy":
        _select_xpy("cupy")
    if waveform == "xp":
        import lalsimulation as lalsim
        if backend == "reference":
            raise ValueError("multirate waveform xp needs a non-reference backend")
        if schedule["kind"] == "one_segment":
            raise ValueError("multirate waveform xp applies to two_rate schedules; one_segment "
                             "always uses lalsimulation templates")
        bad = sorted({lalsim.GetStringFromApproximant(P.approx) for P in P_list
                      if P.approx != lalsim.IMRPhenomXHM})
        if bad:
            raise ValueError("multirate waveform xp supports IMRPhenomXHM only, not %s" % bad)
        try:
            from RIFT.likelihood import gpu_waveform_xhm  # noqa: F401
        except ImportError as exc:
            raise ImportError("multirate waveform xp needs RIFT.likelihood.gpu_waveform_xhm "
                              "(research-projects-RIT #465): %s" % exc)
    for i, P in enumerate(P_list):
        try:
            _refuse_point(P, schedule)
        except ValueError as exc:
            raise ValueError("point %d: %s" % (i, exc))
    return schedule


def PrecomputeLikelihoodTermsMultirate(event_time_geo, t_window, P, data_dict, psd_dict, Lmax, fMax,
                                       analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0., verbose=True,
                                       quiet=False, schedule=None, return_calibration_crossterms=False,
                                       skip_interpolation=False, **kwargs):
    import lalsimulation as lalsim
    from RIFT.likelihood import factored_likelihood as fl
    if schedule is None:
        raise ValueError("early-time multirate requires a schedule")
    _validate_schedule(schedule)
    _refuse(P, schedule, dict(kwargs, analyticPSD_Q=analyticPSD_Q))
    if Lmax not in (2, 3, 4):
        raise ValueError("early-time multirate requires Lmax = 2, 3 or 4")
    if not data_dict or set(data_dict) != set(psd_dict):
        raise ValueError("multirate requires matching nonempty data and PSD detectors")
    detectors = list(data_dict.keys())
    first = data_dict[detectors[0]]
    P.dist = fl.distMpcRef * 1e6 * lsu.lsu_PC
    P.deltaF = first.deltaF
    dt = P.deltaT; fs = 1. / dt; N = first.data.length; seglen = N * dt
    fNyq = fs / 2.
    post = schedule["post"]
    if not (np.isfinite(t_window) and t_window > 0 and int(2 * t_window / dt) >= 4):
        raise ValueError("multirate t_window must contain at least four samples")
    if not (0 < P.fmin < fMax <= fNyq):
        raise ValueError("multirate requires 0 < fmin < fMax <= Nyquist")
    if any(d.data.length != N or d.deltaF != first.deltaF for d in data_dict.values()):
        raise ValueError("multirate detector data must share a frequency grid")
    if not np.isclose(seglen * first.deltaF, 1., rtol=0, atol=1e-12):
        raise ValueError("multirate data grid does not match deltaT")
    for label, n in (("full", N), ("early", seglen * schedule["fs_e"]),
                     ("late", schedule["late_buffer"] * fs)):
        if n != int(n) or n < 2 or int(n) & (int(n) - 1):
            raise ValueError("multirate %s buffer must have a power-of-two sample count" % label)
    if schedule["late_buffer"] > seglen or fs < schedule["fs_e"] or fs % schedule["fs_e"]:
        raise ValueError("multirate requires an integer rate ratio and late_buffer <= data duration")
    if post >= schedule["late_buffer"] / 2 or not np.isclose(post * schedule["fs_e"], round(post * schedule["fs_e"]), atol=1e-10, rtol=0):
        raise ValueError("multirate post must be below half the late buffer and align to the early grid")
    mc_lo = schedule["mc_range"][0]
    f22_late = float(f22_newtonian(schedule["late_start"], mc_lo))
    if schedule["kind"] == "one_segment":
        hlms = _per_m_modes(P, Lmax, fs, seglen, P.fmin, post)
        keys = sorted(hlms)
    else:
        sch = TwoRateSchedule(fs, schedule["fs_e"], schedule["tau_tr"], schedule["guard"], schedule["atten_db"],
                              mc_min=mc_lo, m_max=Lmax, lag_halfwidth=t_window)
        # backend for the per-point preparation; multirate_reference=True keeps the CPU reference path
        use_ref = bool(kwargs.get("multirate_reference", False))
        xp = _select_xpy("numpy" if use_ref else kwargs.get("multirate_xpy"))
        # template source: lalsimulation (default), or gpu_waveform_xhm on xp ("xp", XHM only)
        wf = kwargs.get("multirate_waveform") or "lal"
        if wf not in ("lal", "xp"):
            raise ValueError("multirate waveform must be lal or xp, got %r" % wf)
        if wf == "xp" and (use_ref or P.approx != lalsim.IMRPhenomXHM):
            raise ValueError("multirate waveform xp requires IMRPhenomXHM and a non-reference backend")
        gen = (lambda *a: _per_m_modes_xp(xp, P, Lmax, *a)) if wf == "xp" else (lambda *a: _per_m_modes(P, Lmax, *a))
        early = gen(schedule["fs_e"], seglen, P.fmin, post)
        late = gen(fs, schedule["late_buffer"], f22_late, post)
        keys = sorted(early)
        if _cache_context(data_dict, psd_dict, schedule,
                          (dt, P.fmin, fMax, Lmax, t_window, str(event_time_geo), P.phi, P.theta,
                           inv_spec_trunc_Q, T_spec, P.approx), xp):
            print(" early-time multirate: backend %s%s, templates %s" % (
                "reference" if use_ref else xp.__name__,
                "" if xp is np else " device=%d" % xp.cuda.runtime.getDevice(), wf))
        if not use_ref:
            n_e, n_l = early[keys[0]].data.length, late[keys[0]].data.length
            tk = ("taper", n_e, early[keys[0]].deltaF, schedule["fs_e"], xp.__name__)
            if tk not in _JOB_CACHE:
                _JOB_CACHE[tk] = xp.asarray(taper_top_quarter_factor(n_e, early[keys[0]].deltaF, schedule["fs_e"]))
            e_fd = {k: early[k].dev if wf == "xp" else xp.asarray(early[k].data.data) for k in keys}
            l_fd = {k: late[k].dev if wf == "xp" else xp.asarray(late[k].data.data) for k in keys}
            hE_td = {k: ifft_centred(xp, e_fd[k] * _JOB_CACHE[tk], early[k].deltaF) for k in keys}
            hL_td = {k: ifft_centred(xp, l_fd[k], late[k].deltaF) for k in keys}
        bands = multiband_schedule(P.fmin, fMax, seglen, mc_lo, Lmax, extra_s=schedule["uv_margin"],
                                   top_edge_K=int(round(seglen / schedule["late_buffer"])))
    rholms, rholms_intp, crossTerms, crossTermsV = {}, {}, {}, {}
    # all pieces put the peak `post` s before their buffer end, so a full-length template
    # starts at -(seglen - post): that is the epoch the rholm time axis is built from
    full_epoch = -(seglen - post)
    if schedule["kind"] == "one_segment":
        pieces = [(hlms, seglen)]
    else:
        pieces = [(early, seglen), (late, schedule["late_buffer"])]
    for d, T in pieces:
        e = float(d[keys[0]].epoch)
        assert abs(e + (T - post)) < 0.5 * dt, "template epoch %r, expected %r" % (e, -(T - post))
    for det in detectors:
        t_det = fl.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, event_time_geo)
        rho_epoch = data_dict[det].epoch - full_epoch          # LIGOTimeGPS
        t_shift = float(float(t_det) - float(t_window) - float(rho_epoch))
        N_shift = int(t_shift / dt + 0.5)
        N_window = int(2 * t_window / dt)
        if schedule["kind"] == "one_segment":
            crossTerms[det] = fl.ComputeModeCrossTermIP(hlms, hlms, psd_dict[det], P.fmin, fMax, fNyq, P.deltaF,
                                                        analyticPSD_Q, inv_spec_trunc_Q, T_spec, verbose=False)
            hc = {}
            for k, v in hlms.items():
                t = lsu.DataInverseFourier(v); t.data.data = np.conj(t.data.data); hc[k] = lsu.DataFourier(t)
            crossTermsV[det] = fl.ComputeModeCrossTermIP(hc, hlms, psd_dict[det], P.fmin, fMax, fNyq, P.deltaF,
                                                         analyticPSD_Q, inv_spec_trunc_Q, T_spec, prefix="V",
                                                         verbose=False)
            rholms[det] = fl.ComputeModeIPTimeSeries(hlms, data_dict[det], psd_dict[det], P.fmin, fMax, fNyq,
                                                     N_shift, N_window, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
        else:
            ck = ("data", det)
            if ck not in _JOB_CACHE:
                dbar = data_side_weighted(data_dict[det], psd_dict[det], P.fmin, fMax, fNyq, analyticPSD_Q,
                                          inv_spec_trunc_Q, T_spec)
                s_peak = int(round((float(t_det) - float(data_dict[det].epoch)) / dt))
                _JOB_CACHE[ck] = prepare_data_two_rate(dbar, dt, s_peak, sch)
                print(" early-time multirate: schedule %s (hash %s), %s: early %g Hz before t_det - %g s, "
                      "late %g s at %g Hz" % (schedule["_path"], schedule["_hash"], det, schedule["fs_e"],
                                              schedule["tau_tr"], schedule["late_buffer"], fs))
            prep = _JOB_CACHE[ck]
            pk = ("uv_plan", det)
            if pk not in _JOB_CACHE:      # once per job: bins, weights and gather indices
                _JOB_CACHE[pk] = pieces_uv_plan(psd_dict[det], P.fmin, fMax, fNyq, P.deltaF, bands, schedule["fs_e"],
                                                f22_late, early[keys[0]].data.length, late[keys[0]].data.length,
                                                keys, analyticPSD_Q, inv_spec_trunc_Q, T_spec)
            j_peak = N - int(round(post * fs))
            lags = np.arange(N_shift, N_shift + N_window)
            n_late_off = N - late[keys[0]].data.length
            rholms[det] = {}
            if not use_ref:
                dk = ("device", det, N_shift)
                if dk not in _JOB_CACHE:      # once per job and detector: data FFTs, interpolation, indices
                    _JOB_CACHE[dk] = TwoRateDevice(xp, prep, sch, dt, n_e, n_l, N, j_peak, lags, n_late_off)
                    _JOB_CACHE[dk + ("uv",)] = pieces_uv_plan_device(xp, _JOB_CACHE[pk])
                dev = _JOB_CACHE[dk]
                crossTerms[det], crossTermsV[det] = ComputeModeCrossTermsPiecesDevice(xp, e_fd, l_fd, _JOB_CACHE[dk + ("uv",)], keys)
                for k in keys:
                    ts = lal.CreateCOMPLEX16TimeSeries("rho", rho_epoch, 0., dt, lsu.lsu_DimensionlessUnit, N_window)
                    ts.epoch += N_shift * dt
                    ts.data.data = _to_host(xp, dev.Q(hE_td[k], hL_td[k]))
                    rholms[det][k] = ts
                t = np.arange(N_window) * dt + float(rho_epoch + N_shift * dt)
                rholms_intp[det] = None if skip_interpolation else fl.InterpolateRholms(rholms[det], t, verbose=verbose)
                continue
            U, V, _ = ComputeModeCrossTermsPieces(early, late, _JOB_CACHE[pk])
            crossTerms[det], crossTermsV[det] = U, V
            for k in keys:
                hE = lal.CreateCOMPLEX16FrequencySeries("e", early[k].epoch, early[k].f0, early[k].deltaF,
                                                        early[k].sampleUnits, early[k].data.length)
                hE.data.data = early[k].data.data.copy()
                hE = np.array(lsu.DataInverseFourier(taper_top_quarter(hE, schedule["fs_e"])).data.data)
                hL = np.array(lsu.DataInverseFourier(late[k]).data.data)
                ts = lal.CreateCOMPLEX16TimeSeries("rho", rho_epoch, 0., dt,
                                                   lsu.lsu_DimensionlessUnit, N_window)
                ts.epoch += N_shift * dt
                ts.data.data = Q_two_rate(prep, hE, hL, j_peak, n_late_off, sch, dt, lags)
                rholms[det][k] = ts
        t = np.arange(N_window) * dt + float(rho_epoch + N_shift * dt)
        rholms_intp[det] = None if skip_interpolation else fl.InterpolateRholms(rholms[det], t, verbose=verbose)
    rho_max = 0.
    for det in rholms:
        for k in rholms[det]:
            u = np.real(crossTerms[det][(k, k)])
            if u > 0:
                rho_max += np.max(np.abs(rholms[det][k].data.data))**2 / u
    guess_snr = np.sqrt(rho_max) / 2.3
    if return_calibration_crossterms:
        return rholms_intp, crossTerms, crossTermsV, rholms, guess_snr, None, None, None
    return rholms_intp, crossTerms, crossTermsV, rholms, guess_snr, None
