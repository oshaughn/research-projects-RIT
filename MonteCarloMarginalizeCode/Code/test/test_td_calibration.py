"""
test_td_calibration.py: tests and study runner for RIFT/likelihood/td_calibration.py
(calibration uncertainty as a real FIR filter on the template, TD likelihood code path).

Runnable two ways:
    pytest test_td_calibration.py
    python test_td_calibration.py --study      (prints the tables in td_calibration_study.md)

Tests:
  test_rift_packing_is_hermitian      : RIFT draws satisfy C(-f) = conj C(f); packing helpers.
  test_template_factor_conventions    : 'template' -> eta, 'data' -> 1/eta, conjugate flag.
  test_apply_fir_matches_reference    : np.convolve path == explicit tap loop, incl. edges.
  test_fir_equals_fd_multiplication   : direct TD convolution == FFT multiplication by the
                                        FIR's own response (exact, away from edges) and by
                                        C(f) itself (to the design tolerance, in band).
  test_choose_half_length_meets_tol   : the chosen L meets the tolerance.
  test_basis_vs_exact                 : Q_c, U_c from the rank-r basis vs per-realization
                                        reference on a ringdown and an inspiral piece.
  test_convention_matches_rift_default: on a long white-noise segment the TD Q_c with
                                        taps from C (face value) equals RIFT's default
                                        data-side kappa (lsu.ComplexIP with C_R on the data),
                                        and taps from conj C equal RIFT's conjugate option.
  test_gs_solver_matches_dense        : tdanalysis GS backend == dense Toeplitz (this and the
                                        generator tests skip without tdanalysis >= 1.1).
  test_gs_generator_formula           : Gohberg-Semencul generator identity vs dense inverse.
  test_sliding_gram_vs_exact          : G_kl(t) for all shifts (kernel streams + prefix sums)
                                        == per-shift quadratic forms, 'end' and 'start' sides.
  test_basis_shifted_vs_exact         : time-indexed Q_c(t), U_c(t) basis vs per-shift reference.

FFTs appear here only as independent checks; the module's own code uses none.
"""

import os
import sys
import time

import numpy as np
import pytest

import RIFT.likelihood.td_calibration as tc
from RIFT.calmarg import generate_realizations as genr

FS = 4096.0
DT = 1.0 / FS
PSD_FILE = os.path.expanduser(
    "~/Projects/Codes/tdanalysis/dev/nbs/acorr_from_psd/H1_GW250114_psd_4s_4096.txt")


# --------------------------------------------------------------------------- helpers

def write_synthetic_envelope(path, sigma_amp=(0.10, 0.05), sigma_phase=(0.15, 0.08)):
    """RIFT/bilby envelope format: f, median_mag, median_phase, 16_mag, 16_phase, 84_mag,
    84_phase.  Amplitude centred near 1; sigmas interpolate (log f) from the first value
    at 10 Hz to the second at 4096 Hz: ~5-10% in amplitude, ~5-9 deg in phase."""
    f = np.logspace(1, np.log10(4096.0), 120)
    x = (np.log(f) - np.log(10.0)) / (np.log(4096.0) - np.log(10.0))
    sa = sigma_amp[0] + (sigma_amp[1] - sigma_amp[0]) * x
    sp = sigma_phase[0] + (sigma_phase[1] - sigma_phase[0]) * x
    med_a = 1.0 + 0.02 * np.sin(np.log(f))
    med_p = 0.03 * np.cos(np.log(f))
    env = np.column_stack([f, med_a, med_p, med_a - sa, med_p - sp, med_a + sa, med_p + sp])
    np.savetxt(path, env)
    return path


def rift_realizations(env_path, T, fmin, fmax, n_spline, n_cal, seed=0):
    """Draw realizations exactly as ILE does (create_realizations), RIFT packing."""
    np.random.seed(seed)
    return genr.create_realizations(env_path, T, DT, fmin, fmax, n_spline, n_cal)


def smooth_test_waveform(N, t_merge, f_start=25.0, M_sun=60.0, f_rd=250.0, chirp_time=None):
    """Smooth complex TD test template h(t) = A(t) exp(-i Phi(t)): a Newtonian-like chirp
    from f_start up to a merger frequency, joined continuously to a damped sinusoid
    (ringdown) with damping time tau = M/0.09 (~3.3 ms at 60 Msun)."""
    t = np.arange(N) * DT
    M_sec = M_sun * 4.925491e-6
    tau_rd = M_sec / 0.09
    if chirp_time is None:
        chirp_time = t_merge
    t0 = t_merge - chirp_time
    f_merge = 0.6 * f_rd
    # f(t) = f_start (1 - (t - t0)/tc_eff)^(-3/8), with tc_eff so f(t_merge) = f_merge
    tc_eff = chirp_time / (1.0 - (f_start / f_merge) ** (8.0 / 3.0))
    insp = (t >= t0) & (t < t_merge)
    f_of_t = np.zeros(N)
    f_of_t[insp] = f_start * np.clip(1.0 - (t[insp] - t0) / tc_eff, 1e-6, None) ** (-3.0 / 8.0)
    f_of_t[t >= t_merge] = f_rd
    phase = 2.0 * np.pi * np.cumsum(f_of_t) * DT
    amp = np.zeros(N)
    amp[insp] = (f_of_t[insp] / f_start) ** (2.0 / 3.0)
    # smooth turn-on at the chirp start (part of the waveform model, not a data taper)
    ramp = np.clip((t - t0) / 0.25, 0.0, 1.0)
    amp *= ramp ** 2 * (3 - 2 * ramp)
    a_m = (f_merge / f_start) ** (2.0 / 3.0)
    rd = t >= t_merge
    amp[rd] = a_m * np.exp(-(t[rd] - t_merge) / tau_rd)
    return amp * np.exp(-1j * phase), f_of_t


_ACF_CACHE = {}


def acf_from_psd_file(path, n, fmin, fmax):
    """ACF lags 0..n-1 from a one-sided PSD file (f, S), with S walled outside [fmin, fmax]
    (held at the edge values, as the TD code path's gs_acf does), by the direct cosine sum
    R(tau) = int S(f) cos(2 pi f tau) df, scaled to R[0] = 1.  The raw study ACF file
    (no wall) gives a numerically singular Toeplitz at these piece lengths (eigenvalues
    ~1e-13 of the largest at n=1024), on which Levinson, Cholesky and GS all fail; with
    the wall the condition number is ~1e10 and the three agree to ~1e-7."""
    psd = np.loadtxt(path)
    f, S = psd[:, 0], psd[:, 1].copy()
    S[f < fmin] = S[np.searchsorted(f, fmin)]
    S[f > fmax] = S[np.searchsorted(f, fmax)]
    lag = np.arange(n) * DT
    R = np.zeros(n)
    for a in range(0, n, 512):
        R[a:a + 512] = np.cos(2 * np.pi * np.outer(lag[a:a + 512], f)) @ S
    return R / R[0]


def load_acf(n_needed, fmin=20.0, fmax=2000.0):
    """GW-like ACF: from the GW250114 H1 PSD of the tdanalysis study (walled outside the
    band), else a synthetic coloured one."""
    key = (n_needed, fmin, fmax)
    if key in _ACF_CACHE:
        return _ACF_CACHE[key]
    if os.path.exists(PSD_FILE):
        out = (acf_from_psd_file(PSD_FILE, n_needed, fmin, fmax),
               "H1_GW250114_psd_4s_4096.txt walled outside [%g, %g] Hz" % (fmin, fmax))
    else:
        lag = np.arange(n_needed) * DT
        R = np.exp(-lag / 0.01) * np.cos(2 * np.pi * 60.0 * lag) + 0.3 * np.exp(-lag / 0.002)
        out = (R / R[0], "synthetic")
    _ACF_CACHE[key] = out
    return out


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def env_path(tmp_path_factory):
    return write_synthetic_envelope(str(tmp_path_factory.mktemp("cal") / "H1.txt"))


@pytest.fixture(scope="module")
def grid4s():
    T = 4.0
    N = int(T * FS)
    return dict(T=T, N=N, f_pos=np.arange(N // 2 + 1) / T, fmin=20.0, fmax=2000.0)


# --------------------------------------------------------------------------- tests

def test_rift_packing_is_hermitian(env_path, grid4s):
    cal = rift_realizations(env_path, grid4s["T"], 20.0, 2048.0, 10, 4)
    assert cal.shape == (grid4s["N"], 4)
    assert tc.is_hermitian_rift(cal)
    fR = tc.rift_packing_frequencies(grid4s["N"], 1.0 / grid4s["T"])
    assert fR[0] == FS / 2 and fR[grid4s["N"] // 2] == 0.0 and fR[-1] == -(FS / 2 - 1.0 / grid4s["T"])
    Cp = tc.realizations_to_positive(cal)
    assert Cp.shape == (grid4s["N"] // 2 + 1, 4)
    # bin m of the positive half is RIFT bin N/2 - m
    assert np.allclose(Cp[40], cal[grid4s["N"] // 2 - 40])
    # Hermitian completion is real at DC / Nyquist and conj-symmetric
    Cs = tc.positive_to_standard(Cp[:, 0])
    assert Cs[0].imag == 0 and Cs[grid4s["N"] // 2].imag == 0
    assert np.allclose(Cs[1:grid4s["N"] // 2], np.conj(Cs[grid4s["N"] // 2 + 1:][::-1]))
    assert np.allclose(Cp[1:-1, 0], Cs[1:grid4s["N"] // 2])


def test_template_factor_conventions(env_path, grid4s):
    cal = rift_realizations(env_path, grid4s["T"], 20.0, 2048.0, 10, 2)
    assert np.allclose(tc.template_factor(cal, "template"), cal)
    assert np.allclose(tc.template_factor(cal, "data") * cal, 1.0)
    assert np.allclose(tc.template_factor(cal, "template", conjugate=True), np.conj(cal))
    with pytest.raises(ValueError):
        tc.template_factor(cal, "bogus")


def test_apply_fir_matches_reference():
    rng = np.random.default_rng(1)
    h = rng.standard_normal(3000) + 1j * rng.standard_normal(3000)
    taps = rng.standard_normal(2 * 37 + 1)
    for (i0, i1) in [(500, 900), (0, 100), (2950, 3000), (10, 3000)]:
        g = tc.apply_fir_piece(h, taps, i0, i1)
        g_ref = tc.apply_fir_piece_reference(h, taps, i0, i1)
        assert g.shape == (i1 - i0,)
        assert np.allclose(g, g_ref, rtol=1e-12, atol=1e-12)
    # pieces are the pieces of the full filtered stream (calibrate-then-cut)
    full = np.convolve(np.concatenate([np.zeros(37), h, np.zeros(37)]), taps, mode="valid")
    assert np.allclose(tc.apply_fir_piece(h, taps, 500, 900), full[500:900])


def test_fir_equals_fd_multiplication(env_path, grid4s):
    N, T, f_pos, fmin, fmax = (grid4s[k] for k in ("N", "T", "f_pos", "fmin", "fmax"))
    cal = rift_realizations(env_path, T, 20.0, 2048.0, 10, 1, seed=3)
    Cp = tc.realizations_to_positive(cal)[:, 0]
    tol = 1e-3
    L, err, taps = tc.choose_half_length(Cp, f_pos, FS, tol, fmin, fmax, method="ls", L_max=1024)
    assert err <= tol
    assert tc.fir_band_error(taps, Cp, f_pos, FS, fmin, fmax, return_gain=True)[2] <= 2.0
    # band-limited smooth template on a long zero-padded stream (support well inside)
    h, _ = smooth_test_waveform(N, t_merge=2.5, f_start=30.0, chirp_time=1.5)
    assert np.abs(h[:int(0.5 * FS)]).max() == 0 and np.abs(h[-int(0.5 * FS):]).max() < 1e-6 * np.abs(h).max()
    i0, i1 = int(1.2 * FS), int(2.8 * FS)
    g_td = tc.apply_fir_piece(h, taps, i0, i1)
    # (i) exact: FFT multiplication by the FIR's own response (linear == circular away
    #     from the edges of a zero-padded stream)
    f_std = np.fft.fftfreq(N, DT)
    H_taps = tc.fir_response(taps, f_std, FS)
    g_fd = np.fft.ifft(H_taps * np.fft.fft(h))[i0:i1]
    assert np.abs(g_td - g_fd).max() <= 1e-10 * np.abs(g_td).max()
    # (ii) to the design tolerance: FFT multiplication by C(f) itself
    Cs = tc.positive_to_standard(Cp)
    g_C = np.fft.ifft(Cs * np.fft.fft(h))[i0:i1]
    #      In band the error is bounded by the design tolerance; out of band the design
    #      is free (gain <= 2) and C is unity, so the stream-level rms error is bounded
    #      by tol plus (1 + gain) times the template's out-of-band amplitude fraction.
    hf = np.fft.fft(h)
    inb = (np.abs(f_std) >= fmin) & (np.abs(f_std) <= fmax)
    err_f = (H_taps - Cs) * hf
    assert np.sqrt(np.sum(np.abs(err_f[inb]) ** 2) / np.sum(np.abs(hf) ** 2)) <= tol
    eps_out = np.sqrt(np.sum(np.abs(hf[~inb]) ** 2) / np.sum(np.abs(hf) ** 2))
    rel_rms = np.linalg.norm(g_td - g_C) / np.linalg.norm(g_C)
    assert rel_rms <= tol + 3.0 * eps_out, (rel_rms, eps_out)


def test_choose_half_length_meets_tol(env_path, grid4s):
    T, f_pos, fmin, fmax = (grid4s[k] for k in ("T", "f_pos", "fmin", "fmax"))
    cal = rift_realizations(env_path, T, 20.0, 2048.0, 10, 1, seed=5)
    Cp = tc.realizations_to_positive(cal)[:, 0]
    for method, L_max in (("ls", 1024), ("dft", 2048)):
        L, err, taps = tc.choose_half_length(Cp, f_pos, FS, 1e-2, fmin, fmax, method=method, L_max=L_max)
        assert taps.size == 2 * L + 1
        assert err <= 1e-2
        assert tc.fir_band_error(taps, Cp, f_pos, FS, fmin, fmax)[0] == pytest.approx(err)
        if L > 4:
            e_short = tc.fir_band_error(tc.fir_taps(Cp, f_pos, FS, L // 2, fmin, fmax, method=method),
                                        Cp, f_pos, FS, fmin, fmax)[0]
            assert e_short > 1e-2 or method == "ls"


def _pieces_and_solver(N, t_merge, backend="dense", inspiral_len=2.0):
    """Inspiral piece (inspiral_len s ending 50 ms before merger) and ringdown piece
    (~150 M at 60 Msun, from 2 ms after merger), with Toeplitz(ACF) solves on each."""
    M_sec = 60.0 * 4.925491e-6
    i_m = int(round(t_merge * FS))
    rd = (i_m + int(0.002 * FS), i_m + int(0.002 * FS) + int(round(150 * M_sec * FS)))
    insp = (i_m - int(0.05 * FS) - int(inspiral_len * FS), i_m - int(0.05 * FS))
    pieces = [insp, rd]
    acf, label = load_acf(max(i1 - i0 for i0, i1 in pieces))
    return pieces, tc.toeplitz_piece_solver(acf, pieces, backend=backend), label


def test_basis_vs_exact(env_path, grid4s):
    N, T, f_pos, fmin, fmax = (grid4s[k] for k in ("N", "T", "f_pos", "fmin", "fmax"))
    n_cal = 40
    cal = rift_realizations(env_path, T, 20.0, 2048.0, 10, n_cal, seed=7)
    Cp = tc.realizations_to_positive(cal)
    tol = 1e-3
    basis = tc.realization_basis(Cp, tol, f_pos, fmin, fmax)
    assert basis["err_max"] <= tol and basis["r"] < n_cal
    L = 256
    taps_all = np.array([tc.fir_taps(Cp[:, c], f_pos, FS, L, fmin, fmax, method="ls") for c in range(n_cal)])
    taps_b = tc.basis_taps(basis["phi"], f_pos, FS, L, fmin, fmax, method="ls")
    h, _ = smooth_test_waveform(N, t_merge=3.0, f_start=25.0, chirp_time=2.8)
    pieces, solver, _ = _pieces_and_solver(N, 3.0)
    rng = np.random.default_rng(11)
    d = h.real + 0.3 * rng.standard_normal(N)       # data: untouched by calibration
    w = tc.data_weights(d, pieces, solver)
    Q_ex, U_ex = tc.cal_terms_exact(h, pieces, taps_all, solver, w)
    Q_b, U_b = tc.cal_terms_basis(h, pieces, taps_b, basis["a"], solver, w)
    assert np.abs(Q_b - Q_ex).max() <= 5 * tol * np.abs(Q_ex).max()
    assert np.abs(U_b - U_ex).max() <= 5 * tol * np.abs(U_ex).max()
    # U is a norm: positive; and the spread across realizations is of order sigma_amp
    assert U_ex.min() > 0
    assert 0.02 < U_ex.std() / U_ex.mean() < 0.3


def test_convention_matches_rift_default(env_path, grid4s):
    lal = pytest.importorskip("lal")
    import RIFT.lalsimutils as lsu
    N, T, f_pos, fmin, fmax = (grid4s[k] for k in ("N", "T", "f_pos", "fmin", "fmax"))
    cal = rift_realizations(env_path, T, fmin, fmax, 10, 1, seed=9)      # unity outside band
    C_R = cal[:, 0]
    Cp = tc.realizations_to_positive(cal)[:, 0]
    # data = C h in the STANDARD convention (d_S(f) = C(f) h_S(f), numpy fft), real
    h, _ = smooth_test_waveform(N, t_merge=2.5, f_start=35.0, chirp_time=1.4)
    hr = h.real
    d = np.fft.ifft(tc.positive_to_standard(Cp) * np.fft.fft(hr)).real
    # RIFT side: white PSD, lsu.ComplexIP, calibration on the DATA in RIFT packing
    def to_rift_fd(x):
        s = lal.CreateCOMPLEX16TimeSeries("x", lal.LIGOTimeGPS(0), 0, DT, lal.DimensionlessUnit, N)
        s.data.data = x.astype(complex)
        return lsu.DataFourier(s)
    S0 = 2.0
    IP = lsu.ComplexIP(fLow=fmin, fMax=fmax, fNyq=FS / 2, deltaF=1.0 / T, psd=lambda f: S0, analyticPSD_Q=True)
    hF, dF = to_rift_fd(hr), to_rift_fd(d)
    dC = lal.CreateCOMPLEX16FrequencySeries("dC", dF.epoch, dF.f0, dF.deltaF, lsu.lsu_HertzUnit, N)
    dC.data.data = C_R * dF.data.data
    kappa_default = IP.ip(hF, dC)
    dC.data.data = np.conj(C_R) * dF.data.data
    kappa_conj = IP.ip(hF, dC)
    # TD side: white noise -> C_W = sigma^2 I with sigma^2 = S0 fs / 2 (one-sided PSD)
    sigma2 = S0 * FS / 2.0
    pieces = [(0, N)]
    solver = lambda v, ip: v / sigma2
    w = tc.data_weights(d, pieces, solver)
    L, err, taps_C = tc.choose_half_length(Cp, f_pos, FS, 1e-3, fmin, fmax, method="ls", L_max=1024)
    taps_cC = tc.fir_taps(np.conj(Cp), f_pos, FS, L, fmin, fmax, method="ls")
    Q_C, U_C = tc.cal_terms_exact(hr, pieces, taps_C[None, :], solver, w)
    Q_cC, _ = tc.cal_terms_exact(hr, pieces, taps_cC[None, :], solver, w)
    # face-value C on the template == RIFT default; conj C == RIFT's conjugate option
    assert abs(Q_C[0] - kappa_default) <= 2e-3 * abs(kappa_default), (Q_C[0], kappa_default)
    assert abs(Q_cC[0] - kappa_conj) <= 2e-3 * abs(kappa_conj), (Q_cC[0], kappa_conj)
    # and the two conventions differ by far more than that (phase sigma ~ 0.1 rad)
    assert abs(kappa_default - kappa_conj) > 1e-2 * abs(kappa_default)
    # with d = C h, the face-value filter gives Q = U (the template matches the data)
    assert abs(Q_C[0] - U_C[0]) <= 2e-3 * abs(U_C[0])


def test_gs_solver_matches_dense():
    pytest.importorskip("tdanalysis.forms.toeplitz_solver")    # tdanalysis >= 1.1
    N = int(4 * FS)
    pieces, dense, _ = _pieces_and_solver(N, 3.0, backend="dense")
    _, gs, _ = _pieces_and_solver(N, 3.0, backend="gs")
    rng = np.random.default_rng(3)
    for ip, (i0, i1) in enumerate(pieces):
        v = rng.standard_normal(i1 - i0) + 1j * rng.standard_normal(i1 - i0)
        a, b = dense(v, ip), gs(v, ip)
        assert np.abs(a - b).max() <= 1e-5 * np.abs(a).max()


def test_gs_generator_formula():
    """C^{-1} = (1/x0)[L(x)L(x)^T - L(x~)L(x~)^T] against the dense inverse, and the
    persymmetric (causal) form."""
    pytest.importorskip("tdanalysis.forms.gs")    # generator: tdanalysis' gs_generators_from_acorr
    import scipy.linalg
    acf, _ = load_acf(4096)
    n = 64
    C = scipy.linalg.toeplitz(acf[:n])
    Cinv = np.linalg.inv(C)
    G = tc.gs_dense_inverse(acf[:n])
    assert np.abs(G - Cinv).max() <= 1e-8 * np.abs(Cinv).max()
    x, xt = tc.toeplitz_generator(acf[:n])
    Lx = scipy.linalg.toeplitz(x, np.zeros(n)); Lxt = scipy.linalg.toeplitz(xt, np.zeros(n))
    G2 = (Lx.T @ Lx - Lxt.T @ Lxt) / x[0]
    assert np.abs(G2 - Cinv).max() <= 1e-8 * np.abs(Cinv).max()


def test_sliding_gram_vs_exact():
    """G_kl(t) by kernel streams + prefix sums == explicit per-shift quadratic forms, for
    a template ending inside the piece (side='end') and one starting inside ('start')."""
    pytest.importorskip("tdanalysis.forms.gs")    # generator: tdanalysis' gs_generators_from_acorr
    import scipy.linalg
    acf, _ = load_acf(4096)
    n = 600
    i0, i1 = 1000, 1600
    shifts = np.arange(-25, 26, 5)
    rng = np.random.default_rng(2)
    t = np.arange(3000) * DT
    r = 3
    # templates: complex, compact support
    def make(start, stop):
        g = np.zeros(3000, dtype=complex)
        sl = slice(start, stop)
        g[sl] = (rng.standard_normal(stop - start) + 1j * rng.standard_normal(stop - start)) * np.hanning(stop - start)
        return g
    x, xt = tc.toeplitz_generator(acf[:n])
    Cn = scipy.linalg.toeplitz(acf[:n])
    for side, (s0, s1) in (("end", (700, 1500)), ("start", (1100, 2000))):
        g_list = [make(s0, s1) for _ in range(r)]
        p0, p1 = i0 - shifts.max(), i1 - shifts.min()
        streams = [tc.gs_kernel_streams(g, x, xt, p0, p1, side=side) for g in g_list]
        G = tc.sliding_gram(streams, x[0], i0, i1, shifts, p0)
        for it, sh in enumerate(shifts):
            for k in range(r):
                for l in range(r):
                    gk = g_list[k][i0 - sh:i1 - sh]; gl = g_list[l][i0 - sh:i1 - sh]
                    ref = np.vdot(gk, scipy.linalg.solve_toeplitz(acf[:n], gl))
                    assert abs(G[it, k, l] - ref) <= 1e-7 * abs(ref), (side, sh, k, l)
        # the other side is NOT exact for this layout (template crosses that boundary)
        other = "start" if side == "end" else "end"
        streams_o = [tc.gs_kernel_streams(g, x, xt, p0, p1, side=other) for g in g_list]
        G_o = tc.sliding_gram(streams_o, x[0], i0, i1, shifts, p0)
        assert np.abs(G_o - G).max() > 1e-3 * np.abs(G).max()


def test_basis_shifted_vs_exact(env_path, grid4s):
    """Time-indexed Q_c(t), U_c(t) from the basis (generator method) vs the per-shift,
    per-realization reference; ringdown piece (template ends inside) and inspiral piece
    (template starts inside)."""
    pytest.importorskip("tdanalysis.forms.sliding")    # generator and data term from tdanalysis >= 1.1
    N, T, f_pos, fmin, fmax = (grid4s[k] for k in ("N", "T", "f_pos", "fmin", "fmax"))
    n_cal = 12
    cal = rift_realizations(env_path, T, 20.0, 2048.0, 10, n_cal, seed=21)
    Cp = tc.realizations_to_positive(cal)
    basis = tc.realization_basis(Cp, 1e-3, f_pos, fmin, fmax)
    L = 256
    taps_all = np.array([tc.fir_taps(Cp[:, c], f_pos, FS, L, fmin, fmax, method="ls") for c in range(n_cal)])
    taps_b = tc.basis_taps(basis["phi"], f_pos, FS, L, fmin, fmax, method="ls")
    t_merge = 3.0
    # chirp starts 0.9 s before merger: inside the 1 s inspiral piece (which ends 50 ms
    # before merger), with margin for the FIR pre-ringing (L samples) and the shifts
    h, _ = smooth_test_waveform(N, t_merge=t_merge, f_start=25.0, chirp_time=0.9)
    pieces, solver, _ = _pieces_and_solver(N, t_merge, backend="dense", inspiral_len=1.0)
    acf, _ = load_acf(max(i1 - i0 for i0, i1 in pieces))
    rng = np.random.default_rng(4)
    d = h.real + 0.3 * rng.standard_normal(N)
    w = tc.data_weights(d, pieces, solver)
    shifts = np.array([-40, -7, 0, 13, 40])
    Q_b, U_b = tc.cal_terms_basis_shifted(h, pieces, taps_b, basis["a"], shifts, w, acf=acf,
                                          piece_sides=["start", "end"])
    Q_ex, U_ex = tc.cal_terms_exact_shifted(h, pieces, taps_all, shifts, solver, w)
    assert Q_b.shape == (n_cal, shifts.size) and U_b.shape == (n_cal, shifts.size)
    assert np.abs(Q_b - Q_ex).max() <= 5e-3 * np.abs(Q_ex).max()
    assert np.abs(U_b - U_ex).max() <= 5e-3 * np.abs(U_ex).max()
    # per-shift 'exact' mode on the (short) ringdown piece agrees with the 'end' method
    Q_x, U_x = tc.cal_terms_basis_shifted(h, pieces, taps_b, basis["a"], shifts, w, acf=acf,
                                          piece_sides=["start", "exact"], solver=solver)
    assert np.allclose(Q_x, Q_b)
    assert np.abs(U_x - U_b).max() <= 1e-3 * np.abs(U_b).max()
    # the wrong side for the inspiral piece (template runs past its end) is not exact
    _, U_wrong = tc.cal_terms_basis_shifted(h, pieces, taps_b, basis["a"], shifts, w, acf=acf,
                                            piece_sides=["end", "end"])
    assert np.abs(U_wrong - U_ex).max() > 1e-2 * np.abs(U_ex).max()
    # shift 0 agrees with the unshifted functions
    Q0, U0 = tc.cal_terms_basis(h, pieces, taps_b, basis["a"], solver, w)
    i_zero = int(np.where(shifts == 0)[0][0])
    assert np.allclose(Q_b[:, i_zero], Q0, rtol=1e-6, atol=1e-9 * np.abs(Q0).max())
    # (the 'end' form on the ringdown piece carries the FIR-smeared tail past the piece
    # end, ~1e-3 of the peak, so U agrees with the direct solve to ~1e-5, not roundoff)
    assert np.allclose(U_b[:, i_zero], U0, rtol=1e-4)


# --------------------------------------------------------------------------- study runner

def run_study(out=sys.stdout, n_cal_list=(100, 1000), tols=(1e-2, 1e-3, 1e-4), backend="gs"):
    def p(*a):
        print(*a, file=out); out.flush()
    T = 4.0; N = int(T * FS); f_pos = np.arange(N // 2 + 1) / T
    fmin, fmax = 20.0, 2000.0
    env = write_synthetic_envelope("/tmp/td_calibration_study_H1.txt")
    cal = {n: rift_realizations(env, T, 20.0, 2048.0, 10, n, seed=100 + n) for n in n_cal_list}
    Cp = {n: tc.realizations_to_positive(cal[n]) for n in n_cal_list}
    p("# envelope: synthetic, 10 log-spaced nodes 20-2048 Hz, sigma_amp 10%->5%, sigma_phase 0.15->0.08 rad")
    p("# grid: T=%g s, fs=%g Hz, N=%d; band for the FIR error metric [%g, %g] Hz" % (T, FS, N, fmin, fmax))

    p("\n## FIR half-length L (max in-band relative error <= tol), 5 realizations, median [min-max]")
    p("| method | tol | L | err |")
    p("|---|---|---|---|")
    L_at = {}
    for method, L_max in (("ls", 4096), ("dft", 8000)):
        for tol in tols:
            Ls, errs = [], []
            for c in range(5):
                t0 = time.time()
                L, err, _ = tc.choose_half_length(Cp[n_cal_list[0]][:, c], f_pos, FS, tol, fmin, fmax,
                                                  method=method, L_max=L_max)
                Ls.append(L); errs.append(err)
            L_at[(method, tol)] = int(np.median(Ls))
            p("| %s | %g | %d [%d-%d] | %.1e | " % (method, tol, np.median(Ls), min(Ls), max(Ls), max(errs)))
    for taper in ("kaiser",):
        for tol in tols:
            L, err, _ = tc.choose_half_length(Cp[n_cal_list[0]][:, 0], f_pos, FS, tol, fmin, fmax,
                                              method="dft", L_max=8000, taper=taper, taper_beta=4.0)
            p("| dft+%s taper | %g | %d | %.1e |" % (taper, tol, L, err))
    L, err, _ = tc.choose_half_length(Cp[n_cal_list[0]][:, 0], f_pos, FS, 1e-2, fmin, fmax,
                                      method="dft", extension="unity", L_max=8000)
    p("| dft, unity extension (RIFT's out-of-band values) | 1e-2 | %d | %.1e |" % (L, err))

    p("\n## SVD rank r (max |C_c - sum a phi| <= tol over the band)")
    p("| n_cal | tol | r | err |")
    p("|---|---|---|---|")
    bases = {}
    for n in n_cal_list:
        for tol in tols:
            b = tc.realization_basis(Cp[n], tol, f_pos, fmin, fmax)
            bases[(n, tol)] = b
            p("| %d | %g | %d | %.1e |" % (n, tol, b["r"], b["err_max"]))

    p("\n## Accuracy and timing of basis vs exact per-realization (ringdown + inspiral pieces)")
    t_merge = 3.0
    h, _ = smooth_test_waveform(N, t_merge=t_merge, f_start=25.0, chirp_time=2.8)
    pieces, solver, acf_label = _pieces_and_solver(N, t_merge, backend=backend)
    p("# pieces: inspiral %s (%d samples), ringdown %s (%d samples); ACF: %s; solver backend %s" % (
        pieces[0], pieces[0][1] - pieces[0][0], pieces[1], pieces[1][1] - pieces[1][0], acf_label, backend))
    rng = np.random.default_rng(5)
    d = h.real + 0.3 * rng.standard_normal(N)
    w = tc.data_weights(d, pieces, solver)
    # reference filter: LS at a fixed long half-length
    L_ref = 4096
    err_ref = tc.fir_band_error(tc.fir_taps(Cp[n_cal_list[0]][:, 0], f_pos, FS, L_ref, fmin, fmax, method="ls"),
                                Cp[n_cal_list[0]][:, 0], f_pos, FS, fmin, fmax)[0]
    p("# reference FIR: ls, L=%d (err %.1e)" % (L_ref, err_ref))
    p("| n_cal | tol | L | r | max|dQ|/|Q| basis vs exact | max|dU|/U basis vs exact | max|dQ|/|Q| exact(L) vs exact(L_ref) | t_exact [s] | t_basis [s] | Q0, U0 |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for n in n_cal_list:
        n_ex = min(n, 100)      # exact reference on the first 100 realizations
        taps_ref = np.array([tc.fir_taps(Cp[n][:, c], f_pos, FS, L_ref, fmin, fmax, method="ls") for c in range(n_ex)])
        Q_ref, U_ref = tc.cal_terms_exact(h, pieces, taps_ref, solver, w)
        for tol in tols:
            L = L_at[("ls", tol)]
            b = bases[(n, tol)]
            t0 = time.time()
            taps_all = np.array([tc.fir_taps(Cp[n][:, c], f_pos, FS, L, fmin, fmax, method="ls") for c in range(n_ex)])
            t_design_ex = time.time() - t0
            t0 = time.time()
            Q_ex, U_ex = tc.cal_terms_exact(h, pieces, taps_all, solver, w)
            t_ex = (time.time() - t0) * n / n_ex
            t0 = time.time()
            taps_b = tc.basis_taps(b["phi"], f_pos, FS, L, fmin, fmax, method="ls")
            t_design_b = time.time() - t0
            t0 = time.time()
            Q_b, U_b = tc.cal_terms_basis(h, pieces, taps_b, b["a"], solver, w)
            t_b = time.time() - t0
            dq = np.abs(Q_b[:n_ex] - Q_ex).max() / np.abs(Q_ex).max()
            du = np.abs(U_b[:n_ex] - U_ex).max() / np.abs(U_ex).max()
            dq_ref = np.abs(Q_ex - Q_ref).max() / np.abs(Q_ref).max()
            p("| %d | %g | %d | %d | %.1e | %.1e | %.1e | %.2f (scaled from %d; +%.1f design) | %.2f (+%.1f design, once) | %.3g, %.3g |" % (
                n, tol, L, b["r"], dq, du, dq_ref, t_ex, n_ex, t_design_ex * n / n_ex, t_b, t_design_b,
                abs(Q_ex[0]), U_ex[0]))
    p("# FIR cost per pass over the pieces at L=%d: %.2e real multiply-adds" % (
        L_at[("ls", 1e-3)], tc.fir_cost_flops(pieces, L_at[("ls", 1e-3)])))



def _study_setup(n_cal=100, tol=1e-3, L=298, seed=200):
    T = 4.0; N = int(T * FS); f_pos = np.arange(N // 2 + 1) / T
    fmin, fmax = 20.0, 2000.0
    env = write_synthetic_envelope("/tmp/td_calibration_study_H1.txt")
    cal = rift_realizations(env, T, 20.0, 2048.0, 10, n_cal, seed=seed)
    Cp = tc.realizations_to_positive(cal)
    b = tc.realization_basis(Cp, tol, f_pos, fmin, fmax)
    return dict(T=T, N=N, f_pos=f_pos, fmin=fmin, fmax=fmax, Cp=Cp, basis=b, L=L, tol=tol, n_cal=n_cal)


def run_study_shifted(out=sys.stdout):
    """Time-indexed Q_c(t), U_c(t) = a_c^T G(t) a_c: accuracy vs per-shift reference and cost."""
    def p(*a):
        print(*a, file=out); out.flush()
    S = _study_setup()
    N, f_pos, fmin, fmax, Cp, b, L, n = (S[k] for k in ("N", "f_pos", "fmin", "fmax", "Cp", "basis", "L", "n_cal"))
    p("\n## Time-indexed terms Q_c(t), U_c(t) = a_c^T G(t) a_c, n_cal=%d, tol %g (r=%d), ls L=%d" % (n, S["tol"], b["r"], L))
    taps_b = tc.basis_taps(b["phi"], f_pos, FS, L, fmin, fmax, method="ls")
    t_merge = 3.0
    # inspiral piece 1 s with the chirp starting inside it (generator method, 'start');
    # ringdown piece 44 ms: with +-50 ms shifts the merger can move past its end, so the
    # template is not one-sided there -> per-shift 'exact' (cheap at 182 samples)
    h2, _ = smooth_test_waveform(N, t_merge=t_merge, f_start=25.0, chirp_time=0.9)
    pieces2, solver2, _ = _pieces_and_solver(N, t_merge, backend="dense", inspiral_len=1.0)
    acf2, _ = load_acf(max(i1 - i0 for i0, i1 in pieces2))
    rng = np.random.default_rng(5)
    d2 = h2.real + 0.3 * rng.standard_normal(N)
    w2 = tc.data_weights(d2, pieces2, solver2)
    n_half = int(0.05 * FS)
    shifts = np.arange(-n_half, n_half + 1)
    p("# pieces: inspiral %s ('start'), ringdown %s ('exact'); shifts +-%d samples (%d values)" % (
        pieces2[0], pieces2[1], n_half, shifts.size))
    for sides in (["start", "exact"], ["start", "end"]):
        t0 = time.time()
        Q_t, U_t, q_t, G_t = tc.cal_terms_basis_shifted(h2, pieces2, taps_b, b["a"], shifts, w2, acf=acf2,
                                                        piece_sides=sides, solver=solver2, return_blocks=True)
        t_basis_t = time.time() - t0
        # breakdown: the two kernel streams of one basis template on the inspiral piece
        i0, i1 = pieces2[0]
        x, xt = tc.toeplitz_generator(acf2[:i1 - i0])
        p0, p1 = i0 - shifts.max(), i1 - shifts.min()
        gf = np.zeros(N, dtype=complex); gf[p0:p1] = tc.apply_fir_piece(h2, taps_b[0], p0, p1)
        t0 = time.time(); tc.gs_kernel_streams(gf, x, xt, p0, p1, side="start"); t_streams_one = time.time() - t0
        sub = np.array([0, shifts.size // 4, shifts.size // 2, 3 * shifts.size // 4, shifts.size - 1])
        n_ex = 10
        taps_ex = np.array([tc.fir_taps(Cp[:, c], f_pos, FS, L, fmin, fmax, method="ls") for c in range(n_ex)])
        t0 = time.time()
        Q_e, U_e = tc.cal_terms_exact_shifted(h2, pieces2, taps_ex, shifts[sub], solver2, w2)
        t_exact_sub = time.time() - t0
        dq = np.abs(Q_t[:n_ex][:, sub] - Q_e).max() / np.abs(Q_e).max()
        du = np.abs(U_t[:n_ex][:, sub] - U_e).max() / np.abs(U_e).max()
        p("| sides | r | max|dQ(t)|/|Q| | max|dU(t)|/U | t_basis(all %d shifts) [s] | kernel streams, one template, inspiral piece [s] | t_exact per realization per shift [s] | G(t) memory |" % shifts.size)
        p("|---|---|---|---|---|---|---|---|")
        p("| %s | %d | %.1e | %.1e | %.1f | %.2f | %.3f | %.1f MB |" % (
            "+".join(sides), b["r"], dq, du, t_basis_t, t_streams_one, t_exact_sub / (n_ex * sub.size),
            G_t.nbytes / 1e6))
    p("# exact per-shift cost for n_cal=%d and %d shifts: %.0f s (one Levinson solve each, n=4096); basis method above" % (
        n, shifts.size, t_exact_sub / (n_ex * sub.size) * n * shifts.size))


def run_study_weighted(out=sys.stdout):
    """FIR design judged by Q and U against the FFT-multiplication truth (C with RIFT's unity
    outside the band), and L chosen by the 1/S-weighted rms criterion."""
    def p(*a):
        print(*a, file=out); out.flush()
    S = _study_setup()
    N, T, f_pos, fmin, fmax, Cp = (S[k] for k in ("N", "T", "f_pos", "fmin", "fmax", "Cp"))
    t_merge = 3.0
    h, _ = smooth_test_waveform(N, t_merge=t_merge, f_start=25.0, chirp_time=2.8)
    pieces, solver, _ = _pieces_and_solver(N, t_merge, backend="dense")
    rng = np.random.default_rng(5)
    d = h.real + 0.3 * rng.standard_normal(N)
    w = tc.data_weights(d, pieces, solver)
    n_ex = 10
    hf = np.fft.fft(h)

    def truth(c):
        g = np.fft.ifft(tc.positive_to_standard(Cp[:, c]) * hf)
        Q = sum(np.vdot(g[i0:i1], w[ip]) for ip, (i0, i1) in enumerate(pieces))
        U = sum(np.vdot(g[i0:i1], solver(g[i0:i1], ip)).real for ip, (i0, i1) in enumerate(pieces))
        return Q, U
    QT = np.array([truth(c)[0] for c in range(n_ex)]); UT = np.array([truth(c)[1] for c in range(n_ex)])
    # 1/S weight on the positive grid (PSD file, held outside the band as for the ACF)
    psd = np.loadtxt(PSD_FILE)
    Sf = np.interp(f_pos, psd[:, 0], psd[:, 1])
    inb = (f_pos >= fmin) & (f_pos <= fmax)
    wS = np.zeros_like(f_pos); wS[inb] = 1.0 / Sf[inb]
    p("\n## FIR designs against the FFT-multiplication truth (unity out of band), %d realizations" % n_ex)
    p("| design | L | max in-band err | 1/S-weighted rms err | max|dQ|/|Q| vs truth | max|dU|/U vs truth | t design per realization [s] |")
    p("|---|---|---|---|---|---|---|")
    rows = [("ls ridge 1e-3", "ls", dict()), ("dft unity (RIFT values)", "dft", dict(extension="unity")),
            ("dft hold", "dft", dict(extension="hold"))]
    for L in (298, 1211, 2471):
        for name, method, kw in rows:
            t0 = time.time()
            taps = np.array([tc.fir_taps(Cp[:, c], f_pos, FS, L, fmin, fmax, method=method, **kw) for c in range(n_ex)])
            t_des = (time.time() - t0) / n_ex
            Q, U = tc.cal_terms_exact(h, pieces, taps, solver, w)
            e_max, e_w = tc.fir_band_error(taps[0], Cp[:, 0], f_pos, FS, fmin, fmax, weight=wS)
            p("| %s | %d | %.1e | %.1e | %.1e | %.1e | %.2f |" % (
                name, L, e_max, e_w, np.abs(Q - QT).max() / np.abs(QT).max(), np.abs(U - UT).max() / np.abs(UT).max(), t_des))
    p("\n## L by the 1/S-weighted rms criterion (dft unity), 5 realizations, median [min-max]")
    p("| tol | L | wrms err | max in-band err |")
    p("|---|---|---|---|")
    for tol in (1e-2, 1e-3, 1e-4):
        Ls, ews, ems = [], [], []
        for c in range(5):
            L, e, taps = tc.choose_half_length(Cp[:, c], f_pos, FS, tol, fmin, fmax, method="dft", extension="unity",
                                               L_max=8000, criterion="wrms", weight=wS)
            Ls.append(L); ews.append(e); ems.append(tc.fir_band_error(taps, Cp[:, c], f_pos, FS, fmin, fmax)[0])
        p("| %g | %d [%d-%d] | %.1e | %.1e |" % (tol, np.median(Ls), min(Ls), max(Ls), max(ews), max(ems)))


if __name__ == "__main__":
    if "--study" in sys.argv:
        run_study()
    elif "--study-shifted" in sys.argv:
        run_study_shifted()
    elif "--study-weighted" in sys.argv:
        run_study_weighted()
    else:
        sys.exit(pytest.main([__file__, "-v"]))
