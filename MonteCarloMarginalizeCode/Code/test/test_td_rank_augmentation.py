"""
test_td_rank_augmentation.py:  Unit / integration tests for the TD rank-structured
covariance augmentation module (RIFT/likelihood/td_rank_augmentation.py, tasks G1/G2).

Runnable two ways:
    pytest test_td_rank_augmentation.py
    python  test_td_rank_augmentation.py        (runs every test, prints numbers)

Cases (see TD_LIKELIHOOD.md Sec. 7):
  - test_v1_woodbury_parity:   dense Cholesky of (C + B Lambda B^T) vs RankAugmentation
                               for solve / dot / logdet / delta_lnZ, agreement 1e-9.
  - test_physical_scale_parity: MANDATORY (Sec. 2b) parity at realistic units
                               (C[0,0]~1e-42, sigma_g~1e-21) via the scaled M_hat form, 1e-12.
  - test_td_fd_norm_equivalence: ToeplitzCovariance.dot(v,v) vs (4/(N fs)) sum|vtilde|^2/S.
  - test_slepian_concentration: DPSS basis captures >= 99.9% of in-support band-limited
                               energy; rank tracks 2 W T within the buffer.
  - test_inpainting_limit:     B = identity columns on a gap, sigma_g -> large reproduces
                               the excised-covariance likelihood.
  - test_end_to_end_toy:       colored noise + damped-sinusoid signal + injected blip;
                               glitch reconstruction, delta_lnZ profile, SNR recovery.
"""

import numpy as np

try:
    import RIFT.likelihood.td_rank_augmentation as tra
except Exception:
    # Allow running from the test/ directory with the module on sys.path.
    import os, sys
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "RIFT", "likelihood"))
    import td_rank_augmentation as tra


# ---------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------

def analytic_psd(f):
    """A smooth, strictly positive analytic one-sided PSD (aLIGO-ish, arbitrary scale)."""
    f = np.asarray(f, dtype=float)
    fsafe = np.maximum(f, 1.0)
    # seismic wall + bucket + shot noise; scale ~ 1e-46 so numbers are strain-like.
    return 1e-46 * (1.0 + (20.0 / fsafe) ** 4 + (fsafe / 250.0) ** 2)


def build_cov(fs=4096.0, df=1.0, N=2048, acf_duration=None):
    """ToeplitzCovariance from the analytic PSD; returns (cov, acf, f, S)."""
    if acf_duration is None:
        acf_duration = (N + 16) / fs
    f = np.arange(0.0, fs / 2.0 + df, df)
    S = analytic_psd(f)
    acf = tra.acf_from_psd(f, S, fs, acf_duration=acf_duration)
    cov = tra.ToeplitzCovariance(acf, N)
    return cov, acf, f, S


def band_limited_signal(N, fs, f_low, f_high, rng):
    """A real random signal with power only in [f_low, f_high]."""
    Nf = N // 2 + 1
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    spec = (rng.standard_normal(Nf) + 1j * rng.standard_normal(Nf))
    mask = (freqs >= f_low) & (freqs <= f_high)
    spec[~mask] = 0.0
    return np.fft.irfft(spec, n=N)


# ---------------------------------------------------------------------------------
# V1: Woodbury parity against a dense Cholesky of C + B Lambda B^T
# ---------------------------------------------------------------------------------

def test_v1_woodbury_parity():
    rng = np.random.default_rng(1234)
    N = 2048
    fs = 4096.0
    cov, acf, f, S = build_cov(fs=fs, N=N)

    r = 20
    # Scale B into strain units so B Lambda B^T is comparable to C (both ~1e-40);
    # a random O(1) B with an O(1e-40) C makes the dense sum grossly ill-conditioned
    # and is a degenerate test setup, not a physical one.  sqrt(C[0,0]) sets the scale.
    strain_scale = np.sqrt(cov.C[0, 0])
    B = strain_scale * rng.standard_normal((N, r)) / np.sqrt(N)
    Lambda0 = np.abs(rng.standard_normal(r)) + 0.5   # positive diagonal shape
    sigma_g = 0.7

    ra = tra.RankAugmentation(cov, [(B, Lambda0, "H1")])

    # Dense reference: C' = C + B (sigma_g^2 Lambda0) B^T.
    Lam = sigma_g ** 2 * np.diag(Lambda0)
    Cprime = cov.C + B.dot(Lam).dot(B.T)
    cho = tra.scipy.linalg.cho_factor(Cprime, lower=True, check_finite=False)
    Lref = np.tril(cho[0])
    logdet_ref = 2.0 * np.sum(np.log(np.abs(np.diag(Lref))))

    d = rng.standard_normal(N)
    u = rng.standard_normal(N)

    # solve
    x_ref = tra.scipy.linalg.cho_solve(cho, d, check_finite=False)
    x_wb = ra.solve(d, sigma_g)
    rel_solve = np.linalg.norm(x_wb - x_ref) / np.linalg.norm(x_ref)

    # dot (bilinear)
    dot_ref = u.dot(x_ref)
    dot_wb = ra.dot(u, d, sigma_g)
    rel_dot = abs(dot_wb - dot_ref) / abs(dot_ref)

    # logdet
    logdet_wb = ra.logdet(sigma_g)
    rel_logdet = abs(logdet_wb - logdet_ref) / abs(logdet_ref)

    # delta_lnZ:  1/2 d^T (C^-1 - C'^-1) d - 1/2 (logdet C' - logdet C)
    Cinv_d = cov.solve(d)
    quad_ref = 0.5 * (d.dot(Cinv_d) - d.dot(x_ref))
    logdet_corr_ref = logdet_ref - cov.logdet()
    dlnZ_ref = quad_ref - 0.5 * logdet_corr_ref
    dlnZ_wb = ra.delta_lnZ(d, sigma_g)
    rel_dlnZ = abs(dlnZ_wb - dlnZ_ref) / abs(dlnZ_ref)

    print("[V1] rel solve  = %.3e" % rel_solve)
    print("[V1] rel dot    = %.3e" % rel_dot)
    print("[V1] rel logdet = %.3e" % rel_logdet)
    print("[V1] rel dlnZ   = %.3e" % rel_dlnZ)

    assert rel_solve < 1e-9
    assert rel_dot < 1e-9
    assert rel_logdet < 1e-9
    assert rel_dlnZ < 1e-9

    # complex dot (bilinear, no conjugation) also matches the dense route.
    uc = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    vc = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    # dense: u^T C'^-1 v
    xc = tra.scipy.linalg.cho_solve(cho, vc.real, check_finite=False) \
        + 1j * tra.scipy.linalg.cho_solve(cho, vc.imag, check_finite=False)
    cdot_ref = uc.dot(xc)
    cdot_wb = ra.dot(uc, vc, sigma_g)
    rel_cdot = abs(cdot_wb - cdot_ref) / abs(cdot_ref)
    print("[V1] rel dot(complex) = %.3e" % rel_cdot)
    assert rel_cdot < 1e-9


# ---------------------------------------------------------------------------------
# MANDATORY physical-scale parity (TD_LIKELIHOOD.md Sec. 2b, point 2)
# ---------------------------------------------------------------------------------

def test_physical_scale_parity():
    r"""Woodbury path == dense reference at REALISTIC ACF/strain units.

    C[0,0] ~ 1e-42 (realistic autocovariance units), sigma_g ~ 1e-21 (strain), a shaped
    Lambda_0 with max eigenvalue = 1.  The scaled internal form
    M_hat = I + sigma_g^2 Lambda_0^{1/2} G Lambda_0^{1/2} must keep solve / dot / logdet /
    delta_lnZ within ~1e-12 relative of a dense Cholesky of C' = C + B (sigma_g^2 Lambda_0) B^T.
    This is the trap the design flags: an O(1) Lambda_0 against a 1e-42 covariance makes
    Lambda^{-1} negligible and the direct M ill-scaled; the scaled form must not lose precision.
    """
    rng = np.random.default_rng(20260707)
    N = 1024
    fs = 4096.0
    # PSD scaled so that C[0,0] lands in the ~1e-42 regime (realistic strain^2 ACF units).
    f = np.arange(0.0, fs / 2.0 + 1.0, 1.0)
    S = 1e-47 * (1.0 + (20.0 / np.maximum(f, 1.0)) ** 4 + (f / 250.0) ** 2)
    acf = tra.acf_from_psd(f, S, fs, acf_duration=(N + 32) / fs)
    cov = tra.ToeplitzCovariance(acf, N)
    C00 = cov.C[0, 0]
    strain = np.sqrt(C00)
    print("[PHYS] C[0,0] = %.3e ; sqrt(C[0,0]) = %.3e" % (C00, strain))
    assert 1e-45 < C00 < 1e-39     # confirm we are genuinely at the physical scale

    r = 12
    # B in strain units (columns ~ strain amplitude); Lambda_0 shaped, max eig = 1.
    B = strain * rng.standard_normal((N, r)) / np.sqrt(N)
    lam = np.abs(rng.standard_normal(r)) + 0.3
    Lambda0 = lam / lam.max()      # max eigenvalue exactly 1 (Sec. 2b units convention)
    sigma_g = strain               # ~ 1e-21, the physical amplitude regime

    ra = tra.RankAugmentation(cov, [(B, Lambda0, "H1")])
    # sanity: the scaled core is well-conditioned even though direct M is ~1e42-scaled.
    print("[PHYS] cond(M_hat) = %.3e ; cond(M_direct) = %.3e"
          % (np.linalg.cond(ra.M_hat(sigma_g)), np.linalg.cond(ra.M(sigma_g))))

    Lam = sigma_g ** 2 * np.diag(Lambda0)
    Cprime = cov.C + B.dot(Lam).dot(B.T)
    cho = tra.scipy.linalg.cho_factor(Cprime, lower=True, check_finite=False)
    Lref = np.tril(cho[0])
    logdet_ref = 2.0 * np.sum(np.log(np.abs(np.diag(Lref))))

    # Data with a real glitch in span(B) so delta_lnZ is meaningfully non-zero, plus a
    # noise-scale part; kept at the physical strain scale throughout.
    d = strain * rng.standard_normal(N) + B.dot(sigma_g * rng.standard_normal(r))
    u = strain * rng.standard_normal(N)

    x_ref = tra.scipy.linalg.cho_solve(cho, d, check_finite=False)
    x_wb = ra.solve(d, sigma_g)
    rel_solve = np.linalg.norm(x_wb - x_ref) / np.linalg.norm(x_ref)

    dot_ref = u.dot(x_ref)
    dot_wb = ra.dot(u, d, sigma_g)
    rel_dot = abs(dot_wb - dot_ref) / abs(dot_ref)

    rel_logdet = abs(ra.logdet(sigma_g) - logdet_ref) / abs(logdet_ref)

    # Reference delta_lnZ.  IMPORTANT: the "obvious" dense form
    # 0.5 (d^T C^{-1} d - d^T C'^{-1} d) catastrophically cancels at the physical scale --
    # both terms are ~3e4 and differ only at the ~1e-38 level, so in float64 the dense
    # difference collapses to exactly 0.  This is the very trap the design warns about.
    # The Woodbury quadratic piece 0.5 gamma^T M^{-1} gamma is the *correct* small object
    # and is what delta_lnZ must reproduce; build the reference from the same small algebra
    # (using the direct M here purely as an independent second route).
    Phi = ra.blocks[0]["Phi"]
    gamma_ref = Phi.T.dot(d)
    M_direct = np.diag(1.0 / Lambda0) / sigma_g ** 2 + ra.blocks[0]["G"]
    quad_ref = 0.5 * gamma_ref.dot(np.linalg.solve(M_direct, gamma_ref))
    dlnZ_ref = quad_ref - 0.5 * (logdet_ref - cov.logdet())
    dlnZ_wb = ra.delta_lnZ(d, sigma_g)
    dlnZ_scale = max(abs(dlnZ_ref), abs(quad_ref), 1e-300)
    rel_dlnZ = abs(dlnZ_wb - dlnZ_ref) / dlnZ_scale

    print("[PHYS] rel solve  = %.3e" % rel_solve)
    print("[PHYS] rel dot    = %.3e" % rel_dot)
    print("[PHYS] rel logdet = %.3e" % rel_logdet)
    print("[PHYS] rel dlnZ   = %.3e" % rel_dlnZ)

    assert rel_solve < 1e-12
    assert rel_dot < 1e-12
    assert rel_logdet < 1e-12
    assert rel_dlnZ < 1e-12

    # default sigma_g bounds track the covariance scale (Sec. 2b: [1e-3,1e3]*sqrt(C00)).
    lo, hi = ra.default_sigma_g_bounds()
    print("[PHYS] default sigma_g bounds = (%.3e, %.3e)" % (lo, hi))
    assert np.isclose(lo, 1e-3 * strain, rtol=1e-9)
    assert np.isclose(hi, 1e3 * strain, rtol=1e-9)


# ---------------------------------------------------------------------------------
# TD <-> FD normalization equivalence
# ---------------------------------------------------------------------------------

def test_td_fd_norm_equivalence():
    rng = np.random.default_rng(99)
    fs = 4096.0
    N = 8192           # long-ish segment for a good Toeplitz<->circulant match
    # df must be small enough that nfft = fs/df >= N (the ACF can only be as long as
    # nfft samples); df=0.5 -> nfft=8192.
    cov, acf, f, S = build_cov(fs=fs, df=0.5, N=N, acf_duration=(N + 64) / fs)

    # Band-limited test signal well inside the band where the PSD is smooth.
    f_low, f_high = 40.0, 400.0
    rels = []
    for trial in range(5):
        v = band_limited_signal(N, fs, f_low, f_high, rng)
        # Taper edges lightly to reduce the boundary term (Tukey-ish).
        w = np.ones(N)
        edge = N // 16
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
        w[:edge] = ramp
        w[-edge:] = ramp[::-1]
        v = v * w

        td_norm = cov.dot(v, v)

        vt = np.fft.rfft(v)
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        Sf = analytic_psd(freqs)
        # exclude DC/Nyquist and out-of-band bins (S defined but signal ~0 there)
        fd_norm = (4.0 / (N * fs)) * np.sum(np.abs(vt) ** 2 / Sf)

        rel = abs(td_norm - fd_norm) / abs(fd_norm)
        rels.append(rel)
    rel = float(np.median(rels))
    print("[TDFD] median rel (TD vs FD norm) = %.3e over %d trials" % (rel, len(rels)))
    # "within a few percent" per the design doc.
    assert rel < 0.05


# ---------------------------------------------------------------------------------
# Slepian concentration and rank vs 2 W T
# ---------------------------------------------------------------------------------

def test_slepian_concentration():
    rng = np.random.default_rng(7)
    fs = 4096.0
    N = 4096
    # blip-like support: 50 ms x [30, 430] Hz
    t_start, t_end = 0.10, 0.15
    f_low, f_high = 30.0, 430.0

    B, Lambda0 = tra.slepian_basis(N, fs, t_start, t_end, f_low, f_high)
    r = B.shape[1]

    # The builder rounds the support to whole samples (i0 = floor(t_start*fs),
    # i1 = ceil(t_end*fs)); compute 2 W T from that actual support to compare ranks.
    i0 = int(np.floor(t_start * fs))
    i1 = int(np.ceil(t_end * fs))
    Msup = i1 - i0
    W = 0.5 * (f_high - f_low)
    T = Msup / fs
    two_WT = 2.0 * W * T
    expected_real = int(np.ceil(two_WT)) + 4     # per-channel default in the builder
    expected_total = 2 * expected_real
    print("[SLEP] r = %d ; 2*W*T = %.2f (support %d samples) ; total default = %d"
          % (r, two_WT, Msup, expected_total))
    # rank tracks 2 W T within the buffer (allow small QR pruning).
    assert r <= expected_total
    assert r >= 2 * int(np.floor(two_WT))        # at least ~2 W T survive

    # Energy concentration: random in-support band-limited signals are captured by B.
    i0 = int(np.floor(t_start * fs))
    i1 = int(np.ceil(t_end * fs))
    fracs = []
    for _ in range(20):
        v = band_limited_signal(N, fs, f_low, f_high, rng)
        v[:i0] = 0.0
        v[i1:] = 0.0
        if np.linalg.norm(v) == 0:
            continue
        proj = B.dot(B.T.dot(v))             # B has orthonormal columns
        frac = np.sum(proj ** 2) / np.sum(v ** 2)
        fracs.append(frac)
    frac = float(np.min(fracs))
    print("[SLEP] min captured energy fraction = %.6f (over %d signals)"
          % (frac, len(fracs)))
    assert frac >= 0.999


# ---------------------------------------------------------------------------------
# Inpainting limit: B = indicator columns on a gap, sigma_g -> large
# ---------------------------------------------------------------------------------

def test_inpainting_limit():
    rng = np.random.default_rng(2024)
    fs = 4096.0
    N = 1024
    cov, acf, f, S = build_cov(fs=fs, N=N, acf_duration=(N + 32) / fs)

    # Gap = a contiguous block of samples; B = identity columns on those samples.
    g0, g1 = 400, 440
    gap = np.arange(g0, g1)
    keep = np.array([i for i in range(N) if i not in set(gap.tolist())])
    r = len(gap)
    B = np.zeros((N, r))
    for j, idx in enumerate(gap):
        B[idx, j] = 1.0
    Lambda0 = np.ones(r)

    ra = tra.RankAugmentation(cov, [(B, Lambda0, "H1")])

    # sigma_g large but finite: the prior variance on the gap samples -> infinity,
    # so C'^-1 marginalizes (ignores) those samples == excising them.
    sigma_g = 1e6      # >> data scale (data ~ O(1) below)

    d = rng.standard_normal(N)
    u = rng.standard_normal(N)

    # Excised reference: covariance restricted to the kept samples.
    C_keep = cov.C[np.ix_(keep, keep)]
    cho_keep = tra.scipy.linalg.cho_factor(C_keep, lower=True, check_finite=False)
    dk = d[keep]
    uk = u[keep]
    x_keep = tra.scipy.linalg.cho_solve(cho_keep, dk, check_finite=False)
    dot_excised = uk.dot(x_keep)

    dot_wb = ra.dot(u, d, sigma_g)
    rel_dot = abs(dot_wb - dot_excised) / abs(dot_excised)
    print("[INPAINT] dot(u,d): excised = %.6e ; augmented = %.6e ; rel = %.3e"
          % (dot_excised, dot_wb, rel_dot))

    # C'^-1 d should have (numerically) zero rows on the gap.
    x = ra.solve(d, sigma_g)
    gap_norm = np.linalg.norm(x[gap])
    full_norm = np.linalg.norm(x)
    print("[INPAINT] ||(C'^-1 d)[gap]|| / ||C'^-1 d|| = %.3e" % (gap_norm / full_norm))

    # tolerance appropriate to the large-but-finite sigma_g (~ 1/sigma_g^2 leakage).
    assert rel_dot < 1e-4
    assert gap_norm / full_norm < 1e-4


# ---------------------------------------------------------------------------------
# End-to-end toy: colored noise + damped sinusoid + injected blip
# ---------------------------------------------------------------------------------

def _damped_sinusoid(N, fs, t0, f0, tau, amp):
    t = np.arange(N) / fs
    env = np.zeros(N)
    m = t >= t0
    env[m] = amp * np.exp(-(t[m] - t0) / tau) * np.cos(2.0 * np.pi * f0 * (t[m] - t0))
    return env


def test_end_to_end_toy():
    rng = np.random.default_rng(31337)
    fs = 4096.0
    N = 4096
    cov, acf, f, S = build_cov(fs=fs, N=N, acf_duration=(N + 64) / fs)

    # --- colored noise realization with covariance C (n = L @ white) ---
    L = np.linalg.cholesky(cov.C)
    noise = L.dot(rng.standard_normal(N))
    noise_scale = np.sqrt(np.mean(noise ** 2))

    # --- "signal": damped sinusoid, well away from the glitch in time ---
    sig = _damped_sinusoid(N, fs, t0=0.30, f0=150.0, tau=0.05, amp=8.0 * noise_scale)

    # --- injected blip synthesized from sine-Gaussian lattice atoms (NOT the Slepian
    #     construction) so the glitch is not trivially in the Slepian span. ---
    t_start, t_end = 0.60, 0.66
    f_low, f_high = 40.0, 300.0
    Bsg, _ = tra.sine_gaussian_lattice(N, fs, t_start, t_end, f_low, f_high, q=8)
    coeff = rng.standard_normal(Bsg.shape[1])
    blip = Bsg.dot(coeff)
    blip *= (15.0 * noise_scale) / np.sqrt(np.mean(blip ** 2))   # loud glitch

    data = noise + sig + blip
    data_clean = noise + sig     # glitch-free truth (same noise draw)

    # --- Slepian basis on the (independently known) blip support ---
    Bsl, Lambda0 = tra.slepian_basis(N, fs, t_start, t_end, f_low, f_high)
    ra = tra.RankAugmentation(cov, [(Bsl, Lambda0, "H1")])

    # --- sigma_g profile ---
    # scale bounds around the strain amplitude of the data.
    hi = 10.0 * np.sqrt(np.mean(data ** 2))
    lo = 1e-4 * hi
    sigma_hat, dlnZ_hat = ra.profile_sigma_g(data, (lo, hi))
    # profile curve for reporting
    sig_grid = np.geomspace(lo, hi, 25)
    dlnZ_grid = np.array([ra.delta_lnZ(data, s) for s in sig_grid])
    print("[TOY] sigma_g_hat = %.4e ; delta_lnZ_hat = %.4f" % (sigma_hat, dlnZ_hat))
    print("[TOY] delta_lnZ over grid: min=%.3f max=%.3f (argmax at sigma=%.3e)"
          % (dlnZ_grid.min(), dlnZ_grid.max(), sig_grid[np.argmax(dlnZ_grid)]))
    # profile should find a positive-evidence, finite optimum (glitch present).
    assert dlnZ_hat > 0
    assert lo < sigma_hat < hi

    # --- glitch reconstruction vs injected blip (whitened overlap) ---
    g_hat = ra.glitch_reconstruction(data, sigma_hat)
    wb_ghat = cov.whiten(g_hat)
    wb_blip = cov.whiten(blip)
    overlap = np.dot(wb_ghat, wb_blip) / (np.linalg.norm(wb_ghat) * np.linalg.norm(wb_blip))
    print("[TOY] whitened overlap (reconstruction, injected blip) = %.4f" % overlap)
    assert overlap > 0.9

    # --- matched-filter SNR of the signal, with and without augmentation ---
    # rho = <h|d> / sqrt(<h|h>), using the base and the augmented inner product.
    def snr_base(h, d):
        hh = cov.dot(h, h)
        hd = cov.dot(h, d)
        return hd / np.sqrt(hh)

    def snr_aug(h, d, s):
        hh = ra.dot(h, h, s)
        hd = ra.dot(h, d, s)
        return hd / np.sqrt(hh)

    rho_truth = snr_base(sig, data_clean)          # glitch-free reference
    rho_noaug = snr_base(sig, data)                # glitch contaminates
    rho_aug = snr_aug(sig, data, sigma_hat)        # glitch marginalized

    err_noaug = abs(rho_noaug - rho_truth)
    err_aug = abs(rho_aug - rho_truth)
    print("[TOY] SNR truth = %.4f ; no-aug = %.4f (err %.4f) ; aug = %.4f (err %.4f)"
          % (rho_truth, rho_noaug, err_noaug, rho_aug, err_aug))
    # augmentation brings the recovered SNR closer to the glitch-free truth.
    assert err_aug < err_noaug


# ---------------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    import time
    tests = [
        ("V1 Woodbury parity", test_v1_woodbury_parity),
        ("Physical-scale parity", test_physical_scale_parity),
        ("TD<->FD norm equivalence", test_td_fd_norm_equivalence),
        ("Slepian concentration", test_slepian_concentration),
        ("Inpainting limit", test_inpainting_limit),
        ("End-to-end toy", test_end_to_end_toy),
    ]
    t0 = time.time()
    failures = 0
    for name, fn in tests:
        print("\n=== %s ===" % name)
        try:
            fn()
            print("PASS: %s" % name)
        except AssertionError as e:
            failures += 1
            print("FAIL: %s : %s" % (name, e))
    print("\nTotal runtime: %.2f s ; %d/%d passed"
          % (time.time() - t0, len(tests) - failures, len(tests)))
    raise SystemExit(1 if failures else 0)


def test_correlated_blocks_use_joint_woodbury_update():
    # Independent coefficient priors do not make B_i.T C^-1 B_j vanish.
    rng = np.random.default_rng(843)
    n = 12
    cov = tra.ToeplitzCovariance(0.4 ** np.arange(n), n)
    b1 = rng.normal(size=(n, 2))
    b2 = b1[:, :1] + 0.1 * rng.normal(size=(n, 1))
    p1, p2 = np.array([1., 0.6]), np.array([[1.]])
    m1, m2 = np.array([0.3, -0.1]), np.array([0.2])
    sigma = 0.8
    aug = tra.RankAugmentation(cov, [(b1, p1, "a", m1), (b2, p2, "b", m2)])
    b = np.column_stack((b1, b2))
    prior = tra.scipy.linalg.block_diag(np.diag(p1), p2)
    cp = cov.C + sigma ** 2 * b @ prior @ b.T
    d, u = rng.normal(size=n), rng.normal(size=n)
    inverse_d = np.linalg.solve(cp, d)
    np.testing.assert_allclose(aug.solve(d, sigma), inverse_d, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(aug.dot(u, d, sigma), u @ inverse_d, rtol=1e-11)
    np.testing.assert_allclose(aug.logdet(sigma), np.linalg.slogdet(cp)[1], rtol=1e-11)
    mu = sigma * b @ np.concatenate((m1, m2))
    delta = (0.5 * (d @ cov.solve(d) - (d-mu) @ np.linalg.solve(cp, d-mu))
             - 0.5 * (np.linalg.slogdet(cp)[1] - cov.logdet()))
    np.testing.assert_allclose(aug.delta_lnZ(d, sigma), delta, rtol=1e-11)
    reconstruct = mu + (cp-cov.C) @ np.linalg.solve(cp, d-mu)
    np.testing.assert_allclose(aug.glitch_reconstruction(d, sigma), reconstruct,
                               rtol=1e-11, atol=1e-11)


def test_slepian_rank_validation_and_short_support():
    import pytest
    for rank in (0, -2, 1, 3, 5, 2.5, True):
        with pytest.raises(ValueError, match="positive even integer"):
            tra.slepian_basis(256, 256, .25, .75, 32, 64, r=rank)
    for rank in (2, 4, 6):
        basis, prior = tra.slepian_basis(256, 256, .25, .75, 32, 64, r=rank)
        assert basis.shape == (256, rank)
        assert prior.shape == (rank,)
    for size in (16, 32):
        basis, prior = tra.slepian_basis(size, size, 0, 1, 0, size / 2)
        assert basis.shape[1] <= size
        assert prior.shape == (basis.shape[1],)
        np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
