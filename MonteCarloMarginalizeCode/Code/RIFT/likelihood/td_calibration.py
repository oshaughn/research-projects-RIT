r"""
td_calibration.py: calibration uncertainty on the time-domain (TD) likelihood code path.

Study module.  The production code path is RIFT.likelihood.td_gs_precompute (TDCalibrationFilters
and TDGSDetectorContext.precompute(calibration=...)), which uses this module's packing and
convention helpers (realizations_to_positive, is_hermitian_rift, template_factor) and the
N-tap inverse-DFT filter truncated at the template's reach (exact on the pieces; tdanalysis'
fir_taps_from_response and fir_truncate_at_max_lag); the FIR designs with a tolerance, the
SVD basis of realizations and the sliding Gram methods below are the study's.  The Toeplitz
solves, generators and the sliding data term here are tdanalysis' (tdanalysis >= 1.1).  The TD code path analyses the
data on fixed sample selections ("pieces", index ranges [i0, i1) of one sampled stream)
with noise covariance C_W = Toeplitz(ACF) on each piece, and never applies an FFT, window
or taper to data or templates.  Calibration uncertainty is represented in RIFT by
per-bin realizations

    C_c(f) = A_c(f) exp(i phi_c(f)),     c = 0 .. n_cal-1,

(RIFT.calmarg.generate_realizations.create_realizations: cubic splines in log10 f drawn
from an envelope file, unity outside [fmin, fmax], on RIFT's two-sided grid).  Here the
calibration is applied to the TEMPLATE as a real FIR filter in the TD:

    (K_c h)[n] = sum_{j=-L}^{L} c_c[j] h[n-j],        c_c real, 2L+1 taps,

"calibrate then cut": the filtered template on a piece [i0, i1) needs h on [i0-L, i1+L),
and is obtained by direct linear convolution, then cut to the piece.  The data are never
touched; w = C_W^{-1} d_W is formed once.  Per realization

    Q_c = (K_c h)_W^H w,          U_c = (K_c h)_W^H C_W^{-1} (K_c h)_W.

Low-rank basis.  Writing C_c(f) = sum_k a_ck phi_k(f) with REAL coefficients a_ck (SVD of
the realizations with real and imaginary parts stacked, so that every phi_k is itself a
Hermitian-completable response and every K_c = sum_k a_ck K_k is a real filter), the
per-template cost is r FIR convolutions and r solves instead of n_cal:

    g_k = K_k h,   z_k = C_W^{-1} (g_k)_W,   q_k = (g_k)_W^H w,   G_kl = (g_k)_W^H z_l,
    Q_c = sum_k a_ck q_k,                     U_c = a_c^T G a_c  (= a_c^T Re(G) a_c).

Frequency conventions (see td_calibration_study.md for the derivation and a numerical
check).  RIFT packs two-sided series REVERSED, f_R[k] = deltaF (N/2 - k), and its bin
labelled +f holds int d(t) e^{+2 pi i f t} dt, i.e. the standard-convention value at -f.
RIFT's default calibration branch multiplies the DATA in that packing by the realization
C_R.  Relabelling f -> -f and using C(-f) = conj C(f) gives, in the standard convention
(e^{-2 pi i f t}, numpy fft),

    kappa_RIFT = <h | C_R d>_R  =  <C h | d>_S  =  conj( <d | C h>_S ),

with C the realization at FACE VALUE: amplitude A(f), phase +phi(f) at f > 0.  So the TD
code path reproduces RIFT's default when the TEMPLATE is filtered with the FIR whose
frequency response is C(f) itself (not its conjugate), and RIFT's
--calibration-conjugate-phase option corresponds to filtering with conj C(f).  Since the
likelihood uses Re(kappa) (or |kappa| under phase marginalization), RIFT's default
already agrees with the template-side <d|C h> of bilby for a correction of
"template" type (C maps theoretical strain to calibrated data, d = C h + n).  For an
envelope of bilby's "data" type (H1, L1, K1 by bilby_pipe default: the envelope factor
eta maps calibrated data to theoretical strain, h = eta d) the template factor is
C = 1/eta, which neither RIFT option applies; template_factor() offers both.

The FIR taps are designed from the realization with a direct discrete Fourier sum (no
FFT in this module's own code; tdanalysis' Toeplitz solvers use FFT products internally);
FFTs appear otherwise only in the tests, as an independent check.
"""

import numpy as np

__author__ = "RIFT TD code path (calibration study)"


# --------------------------------------------------------------------------- grids / packing

def rift_packing_frequencies(npts, deltaF):
    r"""Frequency labels of RIFT's two-sided packing: f[k] = deltaF (npts/2 - k).

    Matches lalsimutils.evaluate_fvals (default) and the grid used by
    generate_realizations.build_realizations_from_nodes.  npts must be even.
    """
    npts = int(npts)
    if npts % 2:
        raise ValueError("RIFT two-sided packing needs an even length, got %d" % npts)
    k = np.arange(npts)
    return deltaF * (npts // 2 - k).astype(float)


def realizations_to_positive(cal2side):
    r"""Positive-frequency half (f = 0, deltaF, ..., fNyq) of RIFT-packed realizations.

    cal2side : (npts, n_cal) complex, RIFT packing (output of create_realizations).
    Returns  : (npts/2 + 1, n_cal) complex, bin m at f = m deltaF.  The negative half of
               the RIFT array is conj of this (checked by is_hermitian_rift).
    """
    cal = np.asarray(cal2side)
    npts = cal.shape[0]
    if npts % 2:
        raise ValueError("expected even RIFT two-sided length, got %d" % npts)
    half = npts // 2
    # f_R[k] = (half - k) deltaF  ->  bin m = half - k, k = half - m, m = 0..half
    return cal[half - np.arange(half + 1)]


def is_hermitian_rift(cal2side, rtol=1e-12):
    r"""True if the negative-frequency half of a RIFT-packed array is the conjugate of
    the positive half, C(-f) = conj C(f) (the condition for real FIR taps)."""
    cal = np.asarray(cal2side)
    npts = cal.shape[0]
    half = npts // 2
    pos = cal[half - np.arange(1, half)]           # f = deltaF .. (half-1) deltaF
    neg = cal[half + np.arange(1, half)]           # f = -deltaF .. -(half-1) deltaF
    return np.allclose(neg, np.conj(pos), rtol=rtol, atol=rtol * np.abs(pos).max())


def positive_to_standard(C_pos):
    r"""Hermitian completion of a positive-frequency response onto the standard
    two-sided numpy packing [0, deltaF, ..., fNyq, -(fNyq-deltaF), ..., -deltaF].

    The DC and Nyquist bins are forced real (their imaginary parts cannot be carried by a
    real filter of even length).  Input (half+1,) or (half+1, n); output (2 half,) or
    (2 half, n).
    """
    C_pos = np.asarray(C_pos, dtype=complex)
    half = C_pos.shape[0] - 1
    out = np.zeros((2 * half,) + C_pos.shape[1:], dtype=complex)
    out[:half + 1] = C_pos
    out[0] = C_pos[0].real
    out[half] = C_pos[half].real
    out[half + 1:] = np.conj(C_pos[1:half][::-1])
    return out


def template_factor(cal2side, correction_type="template", conjugate=False):
    r"""Per-bin factor that multiplies the TEMPLATE, from RIFT realizations eta_c(f).

    correction_type : "template" -> C = eta (eta maps theoretical strain to calibrated
                      data, d = eta h + n).  This reproduces RIFT's default data-side
                      branch (see module docstring) and bilby's convention for V1.
                      "data"     -> C = 1/eta (bilby's convention for H1, L1, K1: eta maps
                      calibrated data to theoretical strain, h = eta d, so d = h/eta + n).
    conjugate       : also conjugate (RIFT's --calibration-conjugate-phase, in TD terms).
    Returns an array of the same shape and packing as the input.
    """
    cal = np.asarray(cal2side, dtype=complex)
    if correction_type == "template":
        C = cal.copy()
    elif correction_type == "data":
        C = 1.0 / cal
    else:
        raise ValueError("correction_type must be 'template' or 'data', got %r" % (correction_type,))
    if conjugate:
        C = np.conj(C)
    return C


def smooth_out_of_band(C_pos, f_pos, fmin, fmax, mode="hold"):
    r"""Replace the out-of-band values of a positive-frequency response by a continuous
    continuation, so the FIR impulse response decays faster.  In-band values are kept.

    RIFT realizations are unity outside [fmin, fmax], which puts a jump at each band edge
    (and, through Hermitian symmetry, a phase jump at DC); the impulse response of a jump
    decays only like 1/j.  mode="hold" keeps the edge amplitude on each side and runs the
    phase linearly to zero at DC (below fmin) and at fNyq (above fmax), so the response
    is continuous and real at DC and Nyquist.  mode="unity" returns the input unchanged.

    C_pos : (half+1,) or (half+1, n) complex;  f_pos : (half+1,) frequencies.
    """
    if mode == "unity":
        return np.array(C_pos, dtype=complex, copy=True)
    if mode != "hold":
        raise ValueError("mode must be 'hold' or 'unity'")
    C = np.array(C_pos, dtype=complex, copy=True)
    f = np.asarray(f_pos, dtype=float)
    squeeze = (C.ndim == 1)
    if squeeze:
        C = C[:, None]
    inband = np.where((f >= fmin) & (f <= fmax))[0]
    if inband.size == 0:
        raise ValueError("no bins in [fmin, fmax]")
    ilo, ihi = inband[0], inband[-1]
    f_nyq = f[-1]
    Alo = np.abs(C[ilo]); plo = np.angle(C[ilo])
    Ahi = np.abs(C[ihi]); phi_hi = np.angle(C[ihi])
    if ilo > 0:
        ramp = (f[:ilo] / f[ilo])[:, None]
        C[:ilo] = Alo[None, :] * np.exp(1j * plo[None, :] * ramp)
    if ihi < f.size - 1:
        ramp = ((f_nyq - f[ihi + 1:]) / (f_nyq - f[ihi]))[:, None]
        C[ihi + 1:] = Ahi[None, :] * np.exp(1j * phi_hi[None, :] * ramp)
    return C[:, 0] if squeeze else C


# --------------------------------------------------------------------------- FIR design

def _chunked_rows(n, chunk):
    for start in range(0, n, chunk):
        yield start, min(n, start + chunk)


def fir_taps_dft(C_pos, L, chunk=256):
    r"""Real FIR taps c[j], j = -L..L, from a positive-frequency response by the direct
    inverse discrete Fourier sum on the N = 2 half grid (window method, rectangular):

        c[j] = (1/N) [ C_0 + (-1)^j Re C_{N/2} + 2 sum_{m=1}^{N/2-1} Re( C_m e^{+2 pi i j m / N} ) ].

    The DC and Nyquist bins contribute their real parts only.  Returned array has length
    2L+1 with taps[L + j] = c[j].  Cost ~ (2L+1) (N/2) real multiply-adds; done in
    row chunks to bound memory.  C_pos : (half+1,) complex.
    """
    C = np.asarray(C_pos, dtype=complex)
    half = C.shape[0] - 1
    N = 2 * half
    if L >= half:
        raise ValueError("L=%d must be < N/2=%d" % (L, half))
    j = np.arange(-L, L + 1)
    m = np.arange(1, half)
    taps = np.empty(2 * L + 1)
    cre = C[1:half].real
    cim = C[1:half].imag
    for a, b in _chunked_rows(j.size, chunk):
        theta = (2.0 * np.pi / N) * np.outer(j[a:b], m)
        taps[a:b] = 2.0 * (np.cos(theta) @ cre - np.sin(theta) @ cim)
    taps += C[0].real + ((-1.0) ** j) * C[half].real
    taps /= N
    return taps


def fir_response(taps, f, fs, chunk=256):
    r"""Frequency response H(f) = sum_j c[j] e^{-2 pi i f j / fs} of real taps (length
    2L+1, taps[L+j] = c[j]) at frequencies f, by direct summation."""
    taps = np.asarray(taps, dtype=float)
    L = (taps.size - 1) // 2
    j = np.arange(-L, L + 1)
    f = np.atleast_1d(np.asarray(f, dtype=float))
    H = np.empty(f.size, dtype=complex)
    for a, b in _chunked_rows(f.size, chunk):
        theta = (-2.0 * np.pi / fs) * np.outer(f[a:b], j)
        H[a:b] = np.cos(theta) @ taps + 1j * (np.sin(theta) @ taps)
    return H


def fir_taps_ls(C_pos, f_pos, fs, L, fmin, fmax, weight=None, n_grid=None, ridge=1e-3):
    r"""Real FIR taps by weighted least squares on the in-band grid only:

        min_c  sum_i w_i | sum_j c[j] e^{-2 pi i f_i j / fs} - C(f_i) |^2   + ridge |c|^2,

    f_i the positive grid frequencies in [fmin, fmax] (the full grid unless n_grid
    sub-samples it; fir_band_error checks the result on the full grid).  The response
    outside the band is unconstrained by the data term, which is what lets the in-band
    error reach a tolerance with fewer taps than the window method; without the ridge
    that freedom is abused (out-of-band gains of 1e4-1e5 were seen at L ~ 400), so the
    default ridge = 1e-3 (tiny next to the ~n_grid scale of the normal matrix) keeps the
    out-of-band gain near 1 at a small in-band cost.  Returns taps of length 2L+1.
    """
    C = np.asarray(C_pos, dtype=complex)
    f = np.asarray(f_pos, dtype=float)
    sel = np.where((f >= fmin) & (f <= fmax))[0]
    if n_grid is not None and sel.size > n_grid:
        sel = sel[np.linspace(0, sel.size - 1, int(n_grid)).astype(int)]
    fi = f[sel]; Ci = C[sel]
    w = np.ones(fi.size) if weight is None else np.asarray(weight, dtype=float)[sel]
    theta = (2.0 * np.pi / fs) * fi
    # Normal equations.  For real taps the objective is sum_i w_i |sum_j c_j e^{-i theta_i j}
    # - C_i|^2, whose normal matrix M_{jj'} = sum_i w_i cos(theta_i (j - j')) is Toeplitz
    # (lags 0..2L) and whose right side is b_j = sum_i w_i Re(C_i e^{+i theta_i j}).  Both
    # cost O(n_band L) to form, and (M + ridge I) c = b is a Levinson solve: far cheaper than
    # the explicit (2 n_band) x (2L+1) least-squares problem, with the same minimiser.
    import scipy.linalg
    lags = np.arange(0, 2 * L + 1)
    m = np.empty(lags.size)
    for a_, b_ in _chunked_rows(lags.size, 512):
        m[a_:b_] = np.cos(np.outer(lags[a_:b_], theta)) @ w
    m[0] += ridge
    j = np.arange(-L, L + 1)
    b = np.empty(j.size)
    wre = w * Ci.real
    wim = w * Ci.imag
    for a_, b_ in _chunked_rows(j.size, 512):
        th = np.outer(j[a_:b_], theta)
        b[a_:b_] = np.cos(th) @ wre - np.sin(th) @ wim
    return scipy.linalg.solve_toeplitz(m, b)


def taper_taps(taps, kind="kaiser", beta=4.0):
    r"""Window the filter COEFFICIENTS (a filter-design step, never applied to data or
    templates).  kind: 'kaiser' (beta), 'hann', or None."""
    if kind is None:
        return np.asarray(taps, dtype=float)
    n = len(taps)
    if kind == "kaiser":
        win = np.kaiser(n, beta)
    elif kind == "hann":
        win = np.hanning(n + 2)[1:-1]
    else:
        raise ValueError("unknown taper %r" % (kind,))
    return np.asarray(taps, dtype=float) * win


def fir_band_error(taps, C_pos, f_pos, fs, fmin, fmax, return_gain=False, weight=None):
    r"""Max and rms relative error |H(f) - C(f)| / |C(f)| of the FIR over the positive
    in-band grid [fmin, fmax]; with return_gain also the max |H(f)| outside the band
    (the design is free there, and a template's small out-of-band leakage must not be
    amplified).

    weight : optional non-negative array on f_pos (e.g. 1/S(f), or |h(f)|^2/S(f)); the rms
             is then the weighted rms  sqrt(sum w |H-C|^2 / sum w |C|^2) over the band.
             The max error is set by the band edges (the first spline interval at fmin is
             the sharpest feature) where a template has little weight, so for Q and U the
             weighted rms is the relevant measure; see td_calibration_study.md.
    """
    f = np.asarray(f_pos, dtype=float)
    sel = (f >= fmin) & (f <= fmax)
    H = fir_response(taps, f, fs)
    C = np.asarray(C_pos)
    rel = np.abs(H[sel] - C[sel]) / np.abs(C[sel])
    if weight is None:
        rms = float(np.sqrt(np.mean(rel ** 2)))
    else:
        w = np.asarray(weight, dtype=float)[sel]
        rms = float(np.sqrt(np.sum(w * np.abs(H[sel] - C[sel]) ** 2) / np.sum(w * np.abs(C[sel]) ** 2)))
    if return_gain:
        gain = float(np.abs(H[~sel]).max()) if (~sel).any() else 0.0
        return float(rel.max()), rms, gain
    return float(rel.max()), rms


def fir_taps(C_pos, f_pos, fs, L, fmin, fmax, method="dft", taper=None, taper_beta=4.0,
             extension="hold", **ls_kwargs):
    r"""Design 2L+1 real taps for one positive-frequency response.

    method    : "dft" (window method on the smoothed two-sided response) or "ls" (in-band
                weighted least squares, fir_taps_ls).
    taper     : optional coefficient window ('kaiser'/'hann'), applied to the taps.
    extension : out-of-band continuation used by "dft" (see smooth_out_of_band).
    """
    if method == "dft":
        Cs = smooth_out_of_band(C_pos, f_pos, fmin, fmax, mode=extension)
        taps = fir_taps_dft(Cs, L)
    elif method == "ls":
        taps = fir_taps_ls(C_pos, f_pos, fs, L, fmin, fmax, **ls_kwargs)
    else:
        raise ValueError("method must be 'dft' or 'ls'")
    if taper is not None:
        taps = taper_taps(taps, taper, taper_beta)
    return taps


def choose_half_length(C_pos, f_pos, fs, tol, fmin, fmax, method="dft", L_min=4, L_max=None,
                       taper=None, taper_beta=4.0, extension="hold", gain_max=2.0,
                       criterion="max", weight=None, **ls_kwargs):
    r"""Smallest half-length L (doubling then bisection, assuming the in-band error
    decreases with L) whose FIR reproduces C(f) on [fmin, fmax] to error <= tol with
    out-of-band gain max |H| <= gain_max.

    criterion : "max"  -- max relative error over the band (set by the band edges);
                "wrms" -- weighted rms error with `weight` on f_pos (e.g. 1/S(f)), the
                          measure that tracks the error in Q and U.
    For "dft" the L_max taps are computed once and truncated (truncation of the window
    method IS the shorter design).  For "ls" each probe is a new least-squares fit.
    Returns (L, err, taps); if L_max does not meet the criteria, L = L_max and err > tol
    (or the gain is above gain_max).
    """
    if criterion not in ("max", "wrms"):
        raise ValueError("criterion must be 'max' or 'wrms'")
    half = len(C_pos) - 1
    if L_max is None:
        L_max = half - 1
    L_max = int(min(L_max, half - 1))

    if method == "dft":
        Cs = smooth_out_of_band(C_pos, f_pos, fmin, fmax, mode=extension)
        full = fir_taps_dft(Cs, L_max)

        def taps_at(L):
            t = full[L_max - L:L_max + L + 1]
            return taper_taps(t, taper, taper_beta) if taper is not None else t
    else:
        def taps_at(L):
            return fir_taps(C_pos, f_pos, fs, L, fmin, fmax, method=method, taper=taper,
                            taper_beta=taper_beta, **ls_kwargs)

    def err_at(L):
        e_max, e_rms, gain = fir_band_error(taps_at(L), C_pos, f_pos, fs, fmin, fmax,
                                            return_gain=True, weight=weight)
        e = e_max if criterion == "max" else e_rms
        # a design whose out-of-band gain exceeds gain_max is treated as not meeting tol
        return e if gain <= gain_max else max(e, 2.0 * tol)

    lo = L_min
    e_lo = err_at(lo)
    if e_lo <= tol:
        return lo, e_lo, taps_at(lo)
    # Doubling search upward from L_min (cheap small designs first; matters for "ls",
    # where each probe is a fresh fit whose cost grows with L), then bisection.
    hi = lo
    while True:
        hi = min(2 * hi, L_max)
        e_hi = err_at(hi)
        if e_hi <= tol:
            break
        lo = hi
        if hi >= L_max:
            return hi, e_hi, taps_at(hi)
    # invariant: err(lo) > tol >= err(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if err_at(mid) <= tol:
            hi = mid
        else:
            lo = mid
    return hi, err_at(hi), taps_at(hi)


# --------------------------------------------------------------------------- filtering

def apply_fir_piece(h_full, taps, i0, i1):
    r"""(K h)[i0:i1] for the real taps (length 2L+1), by direct linear convolution of h
    on [i0-L, i1+L) (zero outside the array), then cut to the piece.  Calibrate-then-cut:
    the result is exactly the piece of the fully filtered stream.  h_full may be complex.
    """
    taps = np.asarray(taps, dtype=float)
    L = (taps.size - 1) // 2
    h = np.asarray(h_full)
    n = h.shape[0]
    a, b = i0 - L, i1 + L
    seg = np.zeros(b - a, dtype=h.dtype)
    lo, hi = max(a, 0), min(b, n)
    if hi > lo:
        seg[lo - a:hi - a] = h[lo:hi]
    # np.convolve is a direct (not FFT) linear convolution; 'valid' keeps the fully
    # overlapped outputs, which are exactly indices i0..i1-1 of the filtered stream.
    return np.convolve(seg, taps, mode="valid")


def apply_fir_piece_reference(h_full, taps, i0, i1):
    r"""Same as apply_fir_piece by an explicit loop over taps (clarity/check only)."""
    taps = np.asarray(taps, dtype=float)
    L = (taps.size - 1) // 2
    h = np.asarray(h_full)
    n = h.shape[0]
    out = np.zeros(i1 - i0, dtype=np.result_type(h.dtype, float))
    idx = np.arange(i0, i1)
    for j in range(-L, L + 1):
        src = idx - j
        ok = (src >= 0) & (src < n)
        out[ok] += taps[L + j] * h[src[ok]]
    return out


def fir_cost_flops(pieces, L):
    r"""Real multiply-adds of one complex-template FIR pass over the pieces (2 real
    convolutions): 2 (2L+1) sum_pieces (i1 - i0)."""
    n = sum(int(i1 - i0) for (i0, i1) in pieces)
    return 2 * (2 * L + 1) * n


# --------------------------------------------------------------------------- low-rank basis

def realization_basis(C_pos_all, tol, f_pos=None, fmin=None, fmax=None, r_max=None):
    r"""Real-coefficient low-rank basis of calibration realizations.

    C_pos_all : (half+1, n_cal) complex positive-frequency realizations (the factor that
                multiplies the template, e.g. template_factor(...) then
                realizations_to_positive).
    tol       : rank r is the smallest for which max_c max_f |C_c - sum_k a_ck phi_k| <= tol
                over the band (all bins if fmin/fmax are None); |C| ~ 1 so this is also a
                relative error.

    The SVD is taken of the REAL matrix [Re C; Im C] (2 (half+1) x n_cal), so the basis
    responses phi_k(f) = U_re,k + i U_im,k are complex on f > 0 with REAL coefficients
    a_ck: each phi_k Hermitian-completes to a real FIR, and K_c = sum_k a_ck K_k is a real
    combination of real filters.  Returns dict(phi (half+1, r) complex, a (n_cal, r) real,
    s singular values, r, err_max).
    """
    C = np.asarray(C_pos_all, dtype=complex)
    nf, n_cal = C.shape
    M = np.vstack([C.real, C.imag])                       # (2 nf, n_cal) real
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    if f_pos is not None and fmin is not None and fmax is not None:
        band = (np.asarray(f_pos) >= fmin) & (np.asarray(f_pos) <= fmax)
    else:
        band = np.ones(nf, dtype=bool)
    r_cap = s.size if r_max is None else min(int(r_max), s.size)

    # Scan r upward with rank-one residual updates.  (The SVD minimises the Frobenius
    # error; the max error over bins is not monotone in r, so a bisection can return a
    # non-minimal r.  The scan takes the first r that meets tol.)
    res = M.copy()
    r = r_cap
    err = None
    for k in range(r_cap):
        res -= s[k] * np.outer(U[:, k], Vt[k])
        res_c = res[:nf] + 1j * res[nf:]
        e = float(np.abs(res_c[band]).max())
        if e <= tol:
            r = k + 1; err = e
            break
    if err is None:
        res_c = res[:nf] + 1j * res[nf:]
        err = float(np.abs(res_c[band]).max())
    phi = U[:nf, :r] + 1j * U[nf:, :r]
    a = (Vt[:r].T * s[:r])                                # (n_cal, r)
    return dict(phi=phi, a=a, s=s, r=int(r), err_max=err)


def basis_taps(phi, f_pos, fs, L, fmin, fmax, **kwargs):
    r"""Real taps (r, 2L+1) for the r basis responses phi (half+1, r)."""
    phi = np.asarray(phi, dtype=complex)
    return np.array([fir_taps(phi[:, k], f_pos, fs, L, fmin, fmax, **kwargs)
                     for k in range(phi.shape[1])])


# --------------------------------------------------------------------------- Q_c, U_c

def data_weights(d_full, pieces, solver):
    r"""w_W = C_W^{-1} d_W per piece, formed once from the (untouched) data."""
    d = np.asarray(d_full)
    return [solver(d[i0:i1], ip) for ip, (i0, i1) in enumerate(pieces)]


def cal_terms_exact(h_full, pieces, taps_all, solver, w_pieces):
    r"""Per-realization reference: for each c, filter the template with taps_all[c]
    (n_cal, 2L+1), cut to the pieces, and form
        Q_c = sum_pieces (K_c h)_W^H w_W,
        U_c = sum_pieces (K_c h)_W^H C_W^{-1} (K_c h)_W   (real).
    solver(v, ipiece) returns C_W^{-1} v for the piece (complex v allowed).
    Cost per intrinsic point: n_cal FIR passes and n_cal solves per piece.
    """
    taps_all = np.asarray(taps_all)
    n_cal = taps_all.shape[0]
    Q = np.zeros(n_cal, dtype=complex)
    U = np.zeros(n_cal)
    for c in range(n_cal):
        for ip, (i0, i1) in enumerate(pieces):
            g = apply_fir_piece(h_full, taps_all[c], i0, i1)
            Q[c] += np.vdot(g, w_pieces[ip])
            U[c] += np.vdot(g, solver(g, ip)).real
    return Q, U


def cal_terms_basis(h_full, pieces, taps_basis, a, solver, w_pieces, return_blocks=False):
    r"""Q_c, U_c for all realizations from the rank-r basis:
        g_k = K_k h (r FIR passes), z_k = C_W^{-1} (g_k)_W (r solves per piece),
        q_k = (g_k)_W^H w_W,  G_kl = (g_k)_W^H z_l  (summed over pieces),
        Q_c = sum_k a_ck q_k,  U_c = a_c^T Re(G) a_c.
    taps_basis : (r, 2L+1) real;  a : (n_cal, r) real.
    """
    taps_basis = np.asarray(taps_basis)
    a = np.asarray(a, dtype=float)
    r = taps_basis.shape[0]
    q = np.zeros(r, dtype=complex)
    G = np.zeros((r, r), dtype=complex)
    for ip, (i0, i1) in enumerate(pieces):
        g = np.array([apply_fir_piece(h_full, taps_basis[k], i0, i1) for k in range(r)])  # (r, n)
        z = np.array([solver(g[k], ip) for k in range(r)])                                # (r, n)
        q += g.conj() @ w_pieces[ip]
        G += g.conj() @ z.T
    Q = a @ q
    U = np.einsum("ck,kl,cl->c", a, G.real, a)
    if return_blocks:
        return Q, U, q, G
    return Q, U


# --------------------------------------------------------------------------- time-indexed terms

def toeplitz_generator(acf_n, solver=None):
    r"""Gohberg-Semencul generator x = C^{-1} e_0 of the symmetric Toeplitz C = Toeplitz(acf_n)
    (length n).  With x~ = (0, x_{n-1}, ..., x_1) (reversed, shifted down one) and L(.) the
    lower-triangular Toeplitz matrix with the given first column,

        C^{-1} = (1/x_0) [ L(x) L(x)^T - L(x~) L(x~)^T ],

    so u^H C^{-1} v = (1/x_0) [ <L(x)^T u, L(x)^T v> - <L(x~)^T u, L(x~)^T v> ] and L(.)^T is
    an ANTI-causal correlation, (L(k)^T u)[m] = sum_{p >= m} k[p-m] u[p].  This is the same
    generator tdanalysis' GSQuadraticForm uses (first column of C^{-1}).  solver(v) -> C^{-1} v
    (default: tdanalysis.forms.gs.gs_generators_from_acorr, Levinson with iterative refinement).
    """
    acf_n = np.asarray(acf_n, dtype=float)
    n = acf_n.size
    e0 = np.zeros(n); e0[0] = 1.0
    if solver is None:
        from tdanalysis.forms.gs import gs_generators_from_acorr
        x = gs_generators_from_acorr(acf_n)[0]
    else:
        x = np.asarray(solver(e0), dtype=float)
    xt = np.concatenate([[0.0], x[:0:-1]])
    return x, xt


def gs_dense_inverse(acf_n):
    r"""Dense C^{-1} from the generator formula (check/reference only, O(n^2) memory)."""
    import scipy.linalg
    x, xt = toeplitz_generator(acf_n)
    n = x.size
    Lx = scipy.linalg.toeplitz(x, np.zeros(n))
    Lxt = scipy.linalg.toeplitz(xt, np.zeros(n))
    return (Lx @ Lx.T - Lxt @ Lxt.T) / x[0]


def anticausal_correlation(u, kernel, p0, p1, correlate=None):
    r"""Y[p] = sum_{q >= p} kernel[q-p] u[q] for p in [p0, p1), u a full stream (zero outside
    its array), kernel lags 0..n-1.  Direct (np.correlate) unless a `correlate(u_seg,
    kernel)` callable is supplied (it must return the same 'valid' correlation)."""
    u = np.asarray(u)
    k = np.asarray(kernel, dtype=float)
    n = k.size
    a, b = p0, p1 + n - 1
    lo, hi = max(a, 0), min(b, u.shape[0])
    seg = np.zeros(b - a, dtype=u.dtype)
    if hi > lo:
        seg[lo - a:hi - a] = u[lo:hi]
    if correlate is None:
        # np.correlate(seg, k, 'valid')[i] = sum_j seg[i+j] k[j]  (direct, exact)
        return np.correlate(seg, k, mode="valid")
    return correlate(seg, k)


def causal_convolution(u, kernel, p0, p1, convolve=None):
    r"""Y[p] = sum_{q <= p} kernel[p-q] u[q] for p in [p0, p1) (the mirror of
    anticausal_correlation; direct np.convolve unless `convolve(u_seg, kernel)` is given)."""
    u = np.asarray(u)
    k = np.asarray(kernel, dtype=float)
    n = k.size
    a, b = p0 - (n - 1), p1
    lo, hi = max(a, 0), min(b, u.shape[0])
    seg = np.zeros(b - a, dtype=u.dtype)
    if hi > lo:
        seg[lo - a:hi - a] = u[lo:hi]
    if convolve is None:
        return np.convolve(seg, k, mode="valid")
    return convolve(seg, k)


def gs_kernel_streams(g_full, x, xt, p0, p1, correlate=None, side="end"):
    r"""The two kernel streams of a full-stream template for output indices [p0, p1).

    side="end"   : anti-causal correlations (L(x)^T g, L(x~)^T g); the resulting sliding
                   quadratic form is exact when the shifted template ENDS inside the piece.
    side="start" : causal convolutions (L(x) g, L(x~) g).  By persymmetry J C J = C the
                   generator identity can also be written C^{-1} = (1/x0)[L(x)^T L(x) - L(x~)^T L(x~)],
                   whose kernels only look backward, so this form is exact when the shifted
                   template STARTS inside the piece (and may run past its end).
    A template crossing both piece boundaries is exact in neither form; use the per-shift
    reference (cal_terms_exact_shifted) or a piece layout that avoids it.
    """
    if side == "end":
        return (anticausal_correlation(g_full, x, p0, p1, correlate),
                anticausal_correlation(g_full, xt, p0, p1, correlate))
    if side == "start":
        return (causal_convolution(g_full, x, p0, p1, correlate),
                causal_convolution(g_full, xt, p0, p1, correlate))
    raise ValueError("side must be 'end' or 'start'")


def sliding_gram(streams, x0, i0, i1, shifts, p0):
    r"""G_kl(t) = (S_t g_k)_W^H C_W^{-1} (S_t g_l)_W for all shifts t, from the kernel streams.

    streams : list over k of (Y_k, Yt_k), each evaluated on indices [p0, p0+len).
    (S_t g)[n] = g[n - t]; the piece is [i0, i1).  Using the generator identity and the
    shift-invariance of the full-line correlations,

        G_kl(t) = (1/x0) sum_{m=i0}^{i1-1} [ conj(Y_k) Y_l - conj(Yt_k) Yt_l ][m - t]
                = (1/x0) [ P_kl(i1 - t) - P_kl(i0 - t) ],   P the prefix sum of the products,

    which is EXACT when every shifted template ends inside the piece (its samples at or
    beyond i1 vanish), because the anti-causal kernels only look forward from each sample
    and so never see the piece start or anything past the template end.  Cost O(r^2 n) after
    the streams.  Returns (n_shift, r, r) complex, Hermitian in (k, l).
    """
    r = len(streams)
    shifts = np.asarray(shifts, dtype=int)
    G = np.zeros((shifts.size, r, r), dtype=complex)
    for k in range(r):
        Yk, Ytk = streams[k]
        for l in range(k, r):
            Yl, Ytl = streams[l]
            prod = np.conj(Yk) * Yl - np.conj(Ytk) * Ytl
            P = np.concatenate([[0.0], np.cumsum(prod)])          # P[j] = sum prod[:j]
            hi = (i1 - shifts) - p0
            lo = (i0 - shifts) - p0
            if lo.min() < 0 or hi.max() > prod.size:
                raise ValueError("kernel streams do not cover the shifted piece")
            val = (P[hi] - P[lo]) / x0
            G[:, k, l] = val
            if l != k:
                G[:, l, k] = np.conj(val)
    return G


def sliding_data_term(g_full, w_piece, i0, i1, shifts):
    r"""q(t) = (S_t g)_W^H w_W = sum_{m=i0}^{i1-1} conj(g[m - t]) w[m] for all shifts (direct
    correlation, O(n n_shift)): tdanalysis.forms.sliding.sliding_data_term on the piece
    [i0, i1) (offset i0, template sample j at stream sample j + t)."""
    from tdanalysis.forms.sliding import sliding_data_term as _sliding_data_term
    w = np.asarray(w_piece)
    if w.size != int(i1) - int(i0):
        raise ValueError("w_piece has %d samples; the piece [%d, %d) has %d" % (w.size, i0, i1, i1 - i0))
    return _sliding_data_term(np.asarray(g_full), w, np.asarray(shifts, dtype=int), offset=int(i0))[0]


def cal_terms_basis_shifted(h_full, pieces, taps_basis, a, shifts, w_pieces, acf=None,
                            gram_fn=None, q_fn=None, correlate=None, return_blocks=False,
                            piece_sides=None, solver=None):
    r"""Time-indexed Q_c(t), U_c(t) for all realizations and shifts from the rank-r basis:

        Q_c(t) = sum_k a_ck q_k(t),   U_c(t) = a_c^T Re G(t) a_c,
        q_k(t) = (S_t g_k)_W^H w_W,   G_kl(t) = (S_t g_k)_W^H C_W^{-1} (S_t g_l)_W,

    summed over pieces, with g_k = K_k h computed once on the extended range the shifts
    need (the FIR commutes with the shift, so "calibrate, shift, cut" is a slice of the
    filtered full stream).  By default G(t) comes from the Gohberg-Semencul generator of
    Toeplitz(acf) on each piece (gs_kernel_streams + sliding_gram: two anti-causal kernel
    correlations per basis template, then prefix sums; exact when every shifted template
    ends inside the piece -- or starts inside it with piece_sides[ipiece] = "start", see
    gs_kernel_streams) and q_k(t) from a direct correlation.  piece_sides[ipiece] =
    "exact" instead solves per shift with `solver` (r solves per shift and piece), the
    method for a piece the template crosses on both sides.  gram_fn(g_list, i0, i1,
    shifts, ipiece) -> (n_shift, r, r) and q_fn(g, i0, i1, shifts, ipiece) -> (n_shift,)
    can replace them (e.g. the Precompute code path for U(t)).  Returns Q (n_cal, n_shift)
    complex and U (n_cal, n_shift) real.
    """
    if piece_sides is None:
        piece_sides = ["end"] * len(pieces)
    taps_basis = np.asarray(taps_basis)
    a = np.asarray(a, dtype=float)
    shifts = np.asarray(shifts, dtype=int)
    r = taps_basis.shape[0]
    t_lo, t_hi = int(shifts.min()), int(shifts.max())
    q = np.zeros((r, shifts.size), dtype=complex)
    G = np.zeros((shifts.size, r, r), dtype=complex)
    for ip, (i0, i1) in enumerate(pieces):
        n = i1 - i0
        p0, p1 = i0 - t_hi, i1 - t_lo                  # indices the shifted piece can touch
        g_ext = [apply_fir_piece(h_full, taps_basis[k], p0, p1) for k in range(r)]
        # embed on the full index line so shift bookkeeping stays in stream indices
        g_list = []
        for g in g_ext:
            full = np.zeros(max(p1, h_full.shape[0]), dtype=complex)
            full[p0:p1] = g
            g_list.append(full)
        if q_fn is None:
            for k in range(r):
                q[k] += sliding_data_term(g_list[k], w_pieces[ip], i0, i1, shifts)
        else:
            for k in range(r):
                q[k] += q_fn(g_list[k], i0, i1, shifts, ip)
        if gram_fn is not None:
            G += gram_fn(g_list, i0, i1, shifts, ip)
        elif piece_sides[ip] == "exact":
            # per-shift solves (r per shift): the fallback for a piece the template
            # crosses on both sides; cheap when the piece is short (e.g. a ringdown piece)
            if solver is None:
                raise ValueError("piece_sides 'exact' needs the solver")
            for it, t in enumerate(shifts):
                gs = np.array([g_list[k][i0 - t:i1 - t] for k in range(r)])
                zs = np.array([solver(gs[k], ip) for k in range(r)])
                G[it] += gs.conj() @ zs.T
        else:
            if acf is None:
                raise ValueError("acf is needed for the default (generator) G(t) method")
            x, xt = toeplitz_generator(np.asarray(acf)[:n])
            streams = [gs_kernel_streams(g_list[k], x, xt, p0, p1, correlate, side=piece_sides[ip])
                       for k in range(r)]
            G += sliding_gram(streams, x[0], i0, i1, shifts, p0)
    Q = a @ q                                                   # (n_cal, n_shift)
    U = np.einsum("ck,tkl,cl->ct", a, G.real, a)
    if return_blocks:
        return Q, U, q, G
    return Q, U


def cal_terms_exact_shifted(h_full, pieces, taps_all, shifts, solver, w_pieces):
    r"""Per-realization, per-shift reference (one solve per realization, piece and shift):
    slice the filtered full stream at the shifted piece and form Q, U directly."""
    taps_all = np.asarray(taps_all)
    shifts = np.asarray(shifts, dtype=int)
    n_cal = taps_all.shape[0]
    Q = np.zeros((n_cal, shifts.size), dtype=complex)
    U = np.zeros((n_cal, shifts.size))
    for c in range(n_cal):
        for ip, (i0, i1) in enumerate(pieces):
            p0, p1 = i0 - shifts.max(), i1 - shifts.min()
            g = apply_fir_piece(h_full, taps_all[c], p0, p1)
            for it, t in enumerate(shifts):
                gs = g[(i0 - t) - p0:(i1 - t) - p0]
                Q[c, it] += np.vdot(gs, w_pieces[ip])
                U[c, it] += np.vdot(gs, solver(gs, ip)).real
    return Q, U


# --------------------------------------------------------------------------- solvers (study aid)

def toeplitz_piece_solver(acf, pieces, backend="dense"):
    r"""solver(v, ipiece) -> Toeplitz(acf[:n])^{-1} v for the piece lengths n = i1 - i0.

    backend "dense": scipy.linalg.solve_toeplitz (Levinson; O(n^2) per solve, no FFT of
    the data), fine for pieces up to ~1e4 samples.  backend "gs": tdanalysis'
    ToeplitzSolver on its Gohberg-Semencul method (tdanalysis >= 1.1, as in
    td_inner_product._gs_apply); complex v is solved as real and imaginary parts.  One
    solver per piece length.
    """
    acf = np.asarray(acf, dtype=float)
    lengths = [int(i1 - i0) for (i0, i1) in pieces]
    for n in lengths:
        if acf.size < n:
            raise ValueError("ACF has %d lags; piece needs %d" % (acf.size, n))
    if backend == "dense":
        import scipy.linalg

        def solver(v, ip):
            n = lengths[ip]
            v = np.asarray(v)
            if np.iscomplexobj(v):
                return (scipy.linalg.solve_toeplitz(acf[:n], v.real)
                        + 1j * scipy.linalg.solve_toeplitz(acf[:n], v.imag))
            return scipy.linalg.solve_toeplitz(acf[:n], v)
        return solver
    if backend == "gs":
        from tdanalysis.forms.toeplitz_solver import ToeplitzSolver
        forms = {}
        for n in lengths:
            if n not in forms:
                forms[n] = ToeplitzSolver(acf[:n], dense_max_n=0)

        def solver(v, ip):
            return forms[lengths[ip]].apply_Cinv(np.asarray(v))
        return solver
    raise ValueError("backend must be 'dense' or 'gs'")
