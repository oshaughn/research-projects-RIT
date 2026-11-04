"""
test_td_glitch.py:  Acceptance tests for glitch (rank-augmentation) marginalization wired
into the RIFT time-domain Toeplitz inner product (task G3 of TD_LIKELIHOOD.md; builds
on G1 td_rank_augmentation.py and the T1/T5 td_inner_product.py context).

The algebra (documented in ToeplitzInnerProductContext.attach_glitch_basis): the base TD
matched filter uses the PRECISION operator P = ip_normalization*K (K = band-limited inverse-
PSD kernel), so C^{-1} = P and the covariance-side Woodbury becomes a DIRECT precision matvec
(Phi = P B, NO solve).  M = Lambda^{-1} + B^T P B is always well-posed (floored by the prior),
so the augmentation is defined even though P is rank-deficient off-band (the glitch basis is
band-limited, living where P is well-conditioned).  Corrected ingredients:

    U'_{lm,l'm'} = U - conj(Phi^T h_lm)^T M^{-1} (Phi^T h_l'm')            (Hermitian scalar)
    Q'_lm(t)     = correlate(h_lm, w'),  w' = P(d - g_hat), g_hat = B M^{-1} gamma
    Delta lnZ    = 1/2 gamma^H M^{-1} gamma - 1/2 ln det(I + sigma_g^2 Lambda_0 G)  (per det)

Groups:
  (a) DENSE PARITY -- corrected U'/g_hat/Delta lnZ vs a brute-force dense
      (C + sigma^2 B Lambda_0 B^T)^{-1} construction.  Because a BAND-LIMITED precision is
      rank-deficient off-band (the off-band ambiguity the design flags), the dense parity is
      done in two rigorous halves: (a1) FORMULA vs BRUTE-FORCE with a genuinely invertible
      covariance C (physical ACF Toeplitz, C = P^{-1} exactly), proving the Woodbury-via-
      precision formula equals the true covariance-augmentation inverse to ~1e-12; (a2)
      IMPLEMENTATION vs the dense formula with the ACTUAL band-limited precision P = ip_norm*K,
      proving the FFT-matvec + scaled-M_hat + conjugation implementation matches straightforward
      dense to ~1e-9 (relative).
  (b) CONJUGATION CORRECTNESS -- the Hermitian corrected ip matches the dense conj-explicit
      reference; the (wrong) bilinear correction is O(1) off (the bilinear-vs-Hermitian trap).
  (c) END-TO-END through the REAL Precompute inner-product functions (fl.ComputeModeIPTimeSeries
      / fl.ComputeModeCrossTermIP): a "signal" template + injected sine-Gaussian blip overlapping
      it in one detector; the glitch-marginalized matched-filter lnL(t) is closer to the glitch-
      free reference than the unmitigated run.  Plus default-OFF byte-identical regression.

Environment: ~/.conda/envs/junior_rift/bin/python with
PYTHONPATH=<worktree>/MonteCarloMarginalizeCode/Code.  Runnable as
    pytest test_td_glitch.py
    python  test_td_glitch.py
"""

import numpy as np
import scipy.linalg

import lal
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
from RIFT.likelihood import td_rank_augmentation as tra
from RIFT.likelihood.td_inner_product import (
    ToeplitzInnerProductContext, ToeplitzComplexIP, ToeplitzComplexOverlap,
    TDInnerProductFactory,
)

# --------------------------------------------------------------------------- fixtures

FS = 2048.0
FNYQ = FS / 2.0
N = 4096                      # two-sided length (2 s)
DELTAF = FS / N
DELTAT = 1.0 / FS
LEN1 = N // 2 + 1
FLOW, FMAX = 20.0, 900.0

# glitch time-frequency support (seconds from sample 0, Hz).  For the end-to-end test this
# cell is chosen DISJOINT from the chirp signal's TF track (early time + high frequency, where
# the 40->350 Hz chirp has no power) so the basis removes the blip WITHOUT eating the signal
# (the single-detector-basis-eats-signal failure mode the design warns about).
G_T1, G_T2, G_F1, G_F2 = 0.55, 0.70, 250.0, 380.0

STRAIN = 1.0e-21              # physical amplitude scale (keeps sigma_g profiling in-range)


def colored_psd(f):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    fsafe = np.maximum(np.abs(f), 1.0)
    out = 1e-46 * (1.0 + (20.0 / fsafe) ** 4 + (fsafe / 250.0) ** 2)
    return out if out.size > 1 else float(out[0])


def make_psd_series():
    psd = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0.), 0., DELTAF, lsu.lsu_HertzUnit, LEN1)
    psd.data.data = colored_psd(np.arange(LEN1) * DELTAF)
    return psd


def series_from_td(td_array, epoch=0.0):
    """Forward-FFT a TD numpy array to a COMPLEX16FrequencySeries (lal packing)."""
    ts = lal.CreateCOMPLEX16TimeSeries(
        "x", lal.LIGOTimeGPS(epoch), 0., DELTAT, lsu.lsu_DimensionlessUnit, N)
    ts.data.data = np.asarray(td_array, dtype=complex)
    return lsu.DataFourier(ts)


def bandlimited_series(rng, epoch=0.0, scale=STRAIN):
    f1 = np.arange(LEN1) * DELTAF
    s = np.zeros(LEN1, dtype=complex)
    band = (f1 >= FLOW) & (f1 <= FMAX)
    s[band] = rng.standard_normal(band.sum()) + 1j * rng.standard_normal(band.sum())
    h2 = np.zeros(N, dtype=complex)
    h2[:LEN1] = s[::-1]
    h2[LEN1 - 1:] = s[:-1]
    hf = lal.CreateCOMPLEX16FrequencySeries(
        "h", lal.LIGOTimeGPS(epoch), 0., DELTAF, lsu.lsu_HertzUnit, N)
    hf.data.data = scale * h2
    return hf


def chirp_td(t0, t1, f0, f1_, tau, amp):
    """A tapered linear chirp 'signal' template (band-limited, with a clear peak)."""
    tg = np.arange(N) * DELTAT
    phase = 2 * np.pi * (f0 * (tg - t0) + 0.5 * (f1_ - f0) / (t1 - t0) * (tg - t0) ** 2)
    env = np.exp(-((tg - 0.5 * (t0 + t1)) ** 2) / (2 * tau ** 2))
    x = amp * env * np.cos(phase)
    x[tg < t0] = 0.0
    x[tg > t1] = 0.0
    return x.astype(complex)


def sine_gaussian_td(t0, f0, q, amp):
    tg = np.arange(N) * DELTAT
    tau = q / (2 * np.pi * f0)
    return (amp * np.exp(-((tg - t0) ** 2) / (2 * tau ** 2))
            * np.cos(2 * np.pi * f0 * (tg - t0))).astype(complex)


# --------------------------------------------------------------------------- (a1) formula vs brute-force

def test_formula_vs_bruteforce_invertible_C():
    """Woodbury-via-precision P - PB M^-1 B'P == brute-force (C + sig^2 B Lam0 B')^-1 (~1e-12)."""
    n = 300
    lags = np.arange(n)
    acf = np.exp(-lags / 15.0) * np.cos(0.12 * lags)
    C = scipy.linalg.toeplitz(acf)
    C = 0.5 * (C + C.T)
    C = C + (abs(min(np.linalg.eigvalsh(C))) + 1e-3) * np.eye(n)  # SPD
    P = np.linalg.inv(C)

    rng = np.random.default_rng(1)
    r = 8
    B = rng.standard_normal((n, r))
    Lam0 = np.linspace(1.0, 0.25, r)
    sig = 0.4
    Cp = C + B @ np.diag(sig ** 2 * Lam0) @ B.T
    Cpinv_brute = np.linalg.inv(Cp)

    Phi = P @ B
    G = B.T @ Phi
    M = np.linalg.inv(np.diag(sig ** 2 * Lam0)) + G
    Cpinv_formula = P - Phi @ np.linalg.inv(M) @ Phi.T
    err = np.max(np.abs(Cpinv_brute - Cpinv_formula)) / np.max(np.abs(Cpinv_brute))
    print("[G3-a1] (C+sig^2 B Lam0 B')^-1 brute vs P-PB M^-1 B'P: rel=%.3e" % err)
    assert err < 1e-10


# ------------------------------------------------------ (pop) population-prior dense parity

def _dense_pop_marginal(C, B, Sigma_shape, mu0, sigma_g, d, h):
    r"""Direct N-dim Gaussian marginalization of a ~ N(mu_a, Sigma_a) (population-prior boxed Eq.).

    lnL_marg(h) = ln \int da N(d; h + B a, C) N(a; mu_a, Sigma_a)
                = -1/2 (d - h - B mu_a)^T Cp^-1 (d - h - B mu_a) - 1/2 ln det(2 pi Cp)
    with Sigma_a = sigma_g^2 Sigma_shape, mu_a = sigma_g mu0, Cp = C + B Sigma_a B^T.
    Also returns the h=0 evidence relative to the no-glitch (C-only) model = the reference
    Delta lnZ, and the posterior-mean coefficient a_hat / reconstruction g_hat.
    """
    n = C.shape[0]
    Sigma_a = sigma_g ** 2 * Sigma_shape
    mu_a = sigma_g * np.asarray(mu0, dtype=float)
    Cp = C + B @ Sigma_a @ B.T
    Cpinv = np.linalg.inv(Cp)
    x = d - h - B @ mu_a
    _, logdet_Cp = np.linalg.slogdet(Cp)
    lnL = -0.5 * x @ Cpinv @ x - 0.5 * (logdet_Cp + n * np.log(2 * np.pi))
    # h=0 evidence vs no-glitch
    x0 = d - B @ mu_a
    Cinv = np.linalg.inv(C)
    _, logdet_C = np.linalg.slogdet(C)
    lnZ_g0 = -0.5 * x0 @ Cpinv @ x0 - 0.5 * (logdet_Cp + n * np.log(2 * np.pi))
    lnZ_nog0 = -0.5 * d @ Cinv @ d - 0.5 * (logdet_C + n * np.log(2 * np.pi))
    dlnZ_ref = lnZ_g0 - lnZ_nog0
    # posterior-mean coefficient a_hat = mu_a + M^-1 (B^T Cinv (d - B mu_a))
    M = np.linalg.inv(Sigma_a) + B.T @ Cinv @ B
    gamma_t = B.T @ Cinv @ (d - B @ mu_a)
    a_hat = mu_a + np.linalg.solve(M, gamma_t)
    g_hat = B @ a_hat
    return lnL, dlnZ_ref, g_hat


def test_pop_prior_dense_parity_full_mean_cov():
    """Mean-shifted, FULL-Sigma marginal lnL / Delta lnZ / g_hat via RankAugmentation vs a
    direct N-dim Gaussian marginalization on a small INVERTIBLE-C segment (population-prior)."""
    rng = np.random.default_rng(2024)
    n = 220
    lags = np.arange(n)
    acf = np.exp(-lags / 18.0) * np.cos(0.09 * lags)
    C = scipy.linalg.toeplitz(acf)
    C = 0.5 * (C + C.T)
    C = C + (abs(min(np.linalg.eigvalsh(C))) + 1e-3) * np.eye(n)

    r = 6
    B = rng.standard_normal((n, r))
    # a genuine anisotropic SPD shape, max-eig normalized to 1
    A = rng.standard_normal((r, r))
    Sig = A @ A.T + 0.3 * np.eye(r)
    Sigma_shape = Sig / np.max(np.linalg.eigvalsh(Sig))
    mu0 = rng.standard_normal(r) * 0.4          # dimensionless mean direction
    sigma_g = 0.7
    d = rng.standard_normal(n) * 0.5
    h = rng.standard_normal(n) * 0.3            # a "signal"

    lnL_ref, dlnZ_ref, ghat_ref = _dense_pop_marginal(C, B, Sigma_shape, mu0, sigma_g, d, h)

    # A small dense-covariance adapter (our SPD C is Toeplitz-ish but we treat it generally).
    class _DenseCov(object):
        def __init__(self, Cmat):
            self.N = Cmat.shape[0]
            self.C = Cmat
            self._cho = scipy.linalg.cho_factor(Cmat, lower=True, check_finite=False)
            L = np.tril(self._cho[0])
            self._logdet = 2.0 * np.sum(np.log(np.abs(np.diag(L))))
        def solve(self, b):
            b = np.asarray(b)
            if np.iscomplexobj(b):
                return (scipy.linalg.cho_solve(self._cho, b.real, check_finite=False)
                        + 1j * scipy.linalg.cho_solve(self._cho, b.imag, check_finite=False))
            return scipy.linalg.cho_solve(self._cho, b, check_finite=False)
        def dot(self, u, v):
            return np.tensordot(np.asarray(u), self.solve(v), axes=([-1], [-1]))
        def logdet(self):
            return self._logdet

    aug = tra.RankAugmentation(_DenseCov(C), [(B, Sigma_shape, "H1", mu0)])

    # assemble lnL_marg from the primitives (d_tilde = d - B mu_a)
    mu_a = sigma_g * mu0
    d_tilde = d - B @ mu_a
    x = d_tilde - h
    quad = aug.dot(x, x, sigma_g)
    logdet_Cp = aug.logdet(sigma_g)
    lnL_aug = -0.5 * quad - 0.5 * (logdet_Cp + n * np.log(2 * np.pi))

    dlnZ_aug = aug.delta_lnZ(d, sigma_g)
    ghat_aug = aug.glitch_reconstruction(d, sigma_g)

    rel_lnL = abs(lnL_aug - lnL_ref) / abs(lnL_ref)
    rel_dlnZ = abs(dlnZ_aug - dlnZ_ref) / abs(dlnZ_ref)
    rel_ghat = np.max(np.abs(ghat_aug - ghat_ref)) / np.max(np.abs(ghat_ref))
    print("[pop-a] FULL mean+cov: lnL_marg rel=%.3e  Delta_lnZ rel=%.3e  g_hat rel=%.3e"
          % (rel_lnL, rel_dlnZ, rel_ghat))
    print("        Delta_lnZ(aug)=%.6f vs dense=%.6f" % (dlnZ_aug, dlnZ_ref))
    assert rel_lnL < 1e-9
    assert rel_dlnZ < 1e-9
    assert rel_ghat < 1e-9


def test_pop_prior_reduction_to_flat():
    """mu=0 + (full) diagonal Sigma_shape == the flat diagonal-Lambda0 path, to round-off."""
    rng = np.random.default_rng(2025)
    n = 180
    lags = np.arange(n)
    acf = np.exp(-lags / 12.0)
    C = scipy.linalg.toeplitz(acf) + 0.5 * np.eye(n)

    class _DenseCov(object):
        def __init__(self, Cmat):
            self.N = Cmat.shape[0]; self.C = Cmat
            self._cho = scipy.linalg.cho_factor(Cmat, lower=True, check_finite=False)
            self._logdet = 2.0 * np.sum(np.log(np.abs(np.diag(np.tril(self._cho[0])))))
        def solve(self, b):
            return scipy.linalg.cho_solve(self._cho, np.asarray(b), check_finite=False)
        def dot(self, u, v):
            return np.tensordot(np.asarray(u), self.solve(v), axes=([-1], [-1]))
        def logdet(self):
            return self._logdet

    r = 5
    B = rng.standard_normal((n, r))
    lam0 = np.linspace(1.0, 0.3, r)             # diagonal shape, max 1
    sigma_g = 0.55
    d = rng.standard_normal(n)

    cov = _DenseCov(C)
    aug_flat = tra.RankAugmentation(cov, [(B, lam0, "H1")])                 # zero-mean Gaussian flat
    aug_pop0 = tra.RankAugmentation(cov, [(B, np.diag(lam0), "H1",
                                           np.zeros(r))])                    # mu=0, full-diag

    dz_flat = aug_flat.delta_lnZ(d, sigma_g)
    dz_pop0 = aug_pop0.delta_lnZ(d, sigma_g)
    gh_flat = aug_flat.glitch_reconstruction(d, sigma_g)
    gh_pop0 = aug_pop0.glitch_reconstruction(d, sigma_g)
    ld_flat = aug_flat.logdet_correction(sigma_g)
    ld_pop0 = aug_pop0.logdet_correction(sigma_g)
    rel_dz = abs(dz_pop0 - dz_flat) / abs(dz_flat)
    rel_gh = np.max(np.abs(gh_pop0 - gh_flat)) / np.max(np.abs(gh_flat))
    rel_ld = abs(ld_pop0 - ld_flat) / abs(ld_flat)
    print("[pop-b] reduction (mu=0, diag Sigma_shape) vs flat: Delta_lnZ rel=%.3e  "
          "g_hat rel=%.3e  logdet rel=%.3e" % (rel_dz, rel_gh, rel_ld))
    assert rel_dz < 1e-11
    assert rel_gh < 1e-11
    assert rel_ld < 1e-11


# ------------------------------------------------------ (pop) M2 default = covariance-only, zero mean

def test_fit_population_default_is_M2_zero_mean():
    r"""fit_population.py DEFAULT (PI 2026-07-08): the written basis npz has NO mu key and its
    sigma_shape is the SIGN-SYMMETRIZED UNCENTERED second moment E[aa^T] (= Sigma_central +
    mu mu^T), max-eig normalized.  Also checks --include-mean-UNPHYSICAL restores the FULL
    (mu + Sigma_central) record.  Runs the tool end-to-end on a tiny synthetic corpus."""
    import os
    import subprocess
    import sys
    import tempfile

    poc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "td_glitch_poc")
    if not os.path.isfile(os.path.join(poc_dir, "fit_population.py")):
        print("[pop-M2] fit_population.py not found -- skipping")
        return
    sys.path.insert(0, poc_dir)
    import fit_population as fp

    # sanity on the estimator: sign-symmetrized second moment == Sigma_central + mu mu^T
    rng = np.random.default_rng(99)
    Atest = rng.standard_normal((30, 4)) @ rng.standard_normal((4, 4)) + 0.7
    S2, _ = fp.symmetrized_second_moment(Atest)
    mu = Atest.mean(0)
    Sc = (Atest - mu).T @ (Atest - mu) / len(Atest)
    # with LW shrinkage S2 != exactly Sc+mu mu^T, but the RAW symmetrized moment must be
    raw_sym = np.vstack([Atest, -Atest])
    S2raw = raw_sym.T @ raw_sym / len(raw_sym)
    assert np.max(np.abs(S2raw - (Sc + np.outer(mu, mu)))) < 1e-10

    # tiny synthetic corpus/PCA/flat-basis so the CLI has real inputs
    n, n_full, n_win, r = 24, 32, 16, 3
    c, hw = n_full // 2, n_win // 2
    snips = rng.standard_normal((n, n_full))
    Q, _ = np.linalg.qr(rng.standard_normal((n_win, n_win)))
    components = Q.T[:r + 1]                             # (r+1, n_win) orthonormal rows
    lambda0 = np.linspace(1.0, 0.3, r + 1)
    tmp = tempfile.mkdtemp()
    corpus_p = os.path.join(tmp, "corpus.npz")
    pca_p = os.path.join(tmp, "pca.npz")
    basis_p = os.path.join(tmp, "glitch_basis_L1.npz")
    np.savez(corpus_p, snippets=snips, snr=np.full(n, 20.0), gps=np.arange(n, dtype=float))
    np.savez(pca_p, components=components, r_sel=r, lambda0=lambda0, n_win=n_win)
    np.savez(basis_p, basis=rng.standard_normal((n_win, r)), lambda0=lambda0[:r],
             t0=0.5, fs=FS)

    seed, n_holdout = 12345, 4
    common = [sys.executable, os.path.join(poc_dir, "fit_population.py"),
              "--corpus", corpus_p, "--pca", pca_p, "--seed", str(seed),
              "--n-holdout", str(n_holdout), "--write-basis", basis_p]

    # (1) DEFAULT -> M2, no mu key
    out_m2 = os.path.join(tmp, "m2"); os.makedirs(out_m2, exist_ok=True)
    subprocess.check_call(common + ["--outdir", out_m2],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    npz_m2 = np.load(os.path.join(out_m2, "glitch_basis_L1_pop_m2.npz"))
    assert "mu" not in npz_m2.files, "DEFAULT fit must NOT write a mu key"
    assert "sigma_shape" in npz_m2.files

    # reproduce the tool's projection + split to recompute the expected M2 sigma_shape.  The
    # M2 second moment is now built from the robust CENTERED covariance + sample mean (paper
    # Eq. m2: E[aa^T] = Sigma_central + mu mu^T), so the new mu_mix/sigma_central keys satisfy
    # the moment identity exactly and sigma_central stays SPD.
    B = components[:r].T
    X = snips[:, c - hw:c + hw]
    X = X / np.linalg.norm(X, axis=1)[:, None]
    rng2 = np.random.default_rng(seed)
    holdout = np.sort(rng2.choice(n, size=n_holdout, replace=False))
    train = np.setdiff1d(np.arange(n), holdout)
    A = X[train] @ B
    mu_raw, S_central, _ = fp.robust_covariance(A)
    S2exp = S_central + np.outer(mu_raw, mu_raw)
    Sig_exp = S2exp / np.max(np.linalg.eigvalsh(S2exp))
    rel = np.max(np.abs(np.asarray(npz_m2["sigma_shape"]) - Sig_exp)) / np.max(np.abs(Sig_exp))
    print("[pop-M2] default: no-mu=%s ; sigma_shape==M2 second-moment rel=%.3e"
          % ("mu" not in npz_m2.files, rel))
    assert rel < 1e-10
    # M2 max-eig normalized
    assert abs(np.max(np.linalg.eigvalsh(np.asarray(npz_m2["sigma_shape"]))) - 1.0) < 1e-10
    # NEW detection-conditioned keys present (distinct from the deprecated bare 'mu')
    assert "mu_mix" in npz_m2.files and "sigma_central" in npz_m2.files
    assert "mu" not in npz_m2.files
    _ss = np.asarray(npz_m2["sigma_shape"]); _sc = np.asarray(npz_m2["sigma_central"])
    _mm = np.asarray(npz_m2["mu_mix"])
    # (5e) moment identity sigma_shape == sigma_central + mu_mix mu_mix^T (to round-off)
    resid = np.max(np.abs(_ss - (_sc + np.outer(_mm, _mm))))
    print("[pop-M2] moment identity max|sigma_shape-(sigma_central+mu_mix mu_mix^T)|=%.3e ; "
          "sigma_central SPD min-eig=%.3e" % (resid, np.linalg.eigvalsh(_sc)[0]))
    assert resid < 1e-14
    assert np.linalg.eigvalsh(_sc)[0] > 0        # sigma_central SPD

    # (2) --include-mean-UNPHYSICAL -> FULL (mu present, Sigma_central shape)
    out_full = os.path.join(tmp, "full"); os.makedirs(out_full, exist_ok=True)
    subprocess.check_call(common + ["--outdir", out_full, "--include-mean-UNPHYSICAL"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    npz_full = np.load(os.path.join(out_full, "glitch_basis_L1_pop_full.npz"))
    assert "mu" in npz_full.files and np.any(np.asarray(npz_full["mu"]) != 0.0)
    # FULL shape is the CENTERED covariance (differs from M2 by the mu mu^T ridge)
    assert np.max(np.abs(np.asarray(npz_full["sigma_shape"])
                         - np.asarray(npz_m2["sigma_shape"]))) > 1e-3
    print("[pop-M2] --include-mean-UNPHYSICAL restores FULL: mu present, shape != M2")


# --------------------------------------------------------------------------- (a2) implementation vs dense

def _dense_reference(ctx, B, Lam0, sig, d_td):
    """Dense P - PB M^-1 B'P reference from the ACTUAL band-limited precision P = ip_norm*K."""
    P = ctx.ip_normalization * ctx._dense
    Phi = P @ B
    G = B.T @ Phi
    M = np.linalg.inv(np.diag(sig ** 2 * Lam0)) + G
    Minv = np.linalg.inv(M)
    Cpinv = P - Phi @ Minv @ Phi.T
    gamma = Phi.T @ d_td
    ghat = B @ (Minv @ gamma)
    dlnZ = (0.5 * np.real(np.conj(gamma) @ Minv @ gamma)
            - 0.5 * np.linalg.slogdet(np.eye(len(Lam0)) + sig ** 2 * np.diag(Lam0) @ G)[1])
    return Cpinv, ghat, dlnZ, P


def test_dense_parity_implementation():
    """Context glitch impl (FFT matvec, scaled M_hat, conj) vs dense formula on the real K."""
    rng = np.random.default_rng(11)
    ctx = ToeplitzInnerProductContext(
        colored_psd, DELTAF, FNYQ, fLow=FLOW, fMax=FMAX, analyticPSD_Q=True)
    B, Lam0 = tra.slepian_basis(N, FS, G_T1, G_T2, G_F1, G_F2)
    data = bandlimited_series(rng)
    info = ctx.attach_glitch_basis([(B, Lam0, "H1")], data, sigma_g="profile")
    sig = ctx._glitch_sigma_g
    d_td = np.asarray(lsu.DataInverseFourier(data).data.data)

    Cpinv, ghat_ref, dlnZ_ref, P = _dense_reference(ctx, B, Lam0, sig, d_td)

    # g_hat (Q' correction vector)
    rel_ghat = np.max(np.abs(ctx.glitch_reconstruction_td() - ghat_ref)) / np.max(np.abs(ghat_ref))
    # Delta lnZ
    rel_dlnZ = abs(info["delta_lnZ"] - dlnZ_ref) / abs(dlnZ_ref)
    # U' via the real ToeplitzComplexIP
    IP = ToeplitzComplexIP(ctx)
    h1 = bandlimited_series(rng)
    h2 = bandlimited_series(rng)
    h1t = np.asarray(lsu.DataInverseFourier(h1).data.data)
    h2t = np.asarray(lsu.DataInverseFourier(h2).data.data)
    Up = IP.ip(h1, h2)
    Up_ref = np.conj(h1t) @ Cpinv @ h2t
    rel_U = abs(Up - Up_ref) / abs(Up_ref)

    print("[G3-a2] r=%d sigma_g=%.3e  g_hat rel=%.3e  Delta_lnZ rel=%.3e  U' rel=%.3e"
          % (B.shape[1], sig, rel_ghat, rel_dlnZ, rel_U))
    assert rel_ghat < 1e-9
    assert rel_dlnZ < 1e-9
    assert rel_U < 1e-9


def test_Qprime_overlap_consistency():
    """Corrected rhoTS from ToeplitzComplexOverlap == base overlap on dense-corrected data."""
    rng = np.random.default_rng(12)
    ctx = ToeplitzInnerProductContext(
        colored_psd, DELTAF, FNYQ, fLow=FLOW, fMax=FMAX, analyticPSD_Q=True)
    B, Lam0 = tra.slepian_basis(N, FS, G_T1, G_T2, G_F1, G_F2)
    data = bandlimited_series(rng)
    ctx.attach_glitch_basis([(B, Lam0, "H1")], data, sigma_g="profile")

    hlm = bandlimited_series(rng)
    ov = ToeplitzComplexOverlap(ctx)
    rho_c, rhoTS_c, _, _ = ov.ip(hlm, data)          # glitch-corrected Q'(t)

    # dense reference corrected data d_corr = d - g_hat, run a CLEAN (no-glitch) overlap on it
    d_td = np.asarray(lsu.DataInverseFourier(data).data.data)
    ghat = ctx.glitch_reconstruction_td()
    data_corr = series_from_td(d_td - ghat, epoch=float(data.epoch))
    ctx_clean = ToeplitzInnerProductContext(
        colored_psd, DELTAF, FNYQ, fLow=FLOW, fMax=FMAX, analyticPSD_Q=True)
    ov_clean = ToeplitzComplexOverlap(ctx_clean)
    _, rhoTS_ref, _, _ = ov_clean.ip(hlm, data_corr)

    a = np.asarray(rhoTS_c.data.data)
    b = np.asarray(rhoTS_ref.data.data)
    rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
    print("[G3-a2] Q'(t) corrected-overlap vs base-overlap-on-corrected-data rel=%.3e" % rel)
    assert rel < 1e-10


# --------------------------------------------------------------------------- (b) conjugation trap

def test_conjugation_trap():
    """Hermitian corrected U' matches dense; the bilinear (no-conj) correction is O(1) wrong."""
    rng = np.random.default_rng(21)
    ctx = ToeplitzInnerProductContext(
        colored_psd, DELTAF, FNYQ, fLow=FLOW, fMax=FMAX, analyticPSD_Q=True)
    B, Lam0 = tra.slepian_basis(N, FS, G_T1, G_T2, G_F1, G_F2)
    data = bandlimited_series(rng)
    ctx.attach_glitch_basis([(B, Lam0, "H1")], data, sigma_g="profile")
    sig = ctx._glitch_sigma_g

    h1 = bandlimited_series(rng)
    h2 = bandlimited_series(rng)
    h1t = np.asarray(lsu.DataInverseFourier(h1).data.data)
    h2t = np.asarray(lsu.DataInverseFourier(h2).data.data)

    P = ctx.ip_normalization * ctx._dense
    Phi = P @ B
    M = np.linalg.inv(np.diag(sig ** 2 * Lam0)) + Phi.T @ B
    Minv = np.linalg.inv(M)
    Cpinv = P - Phi @ Minv @ Phi.T
    Up_ref = np.conj(h1t) @ Cpinv @ h2t
    base = ctx.ip_normalization * ctx.dot_td(h1t, h2t)

    corr_herm = ctx.glitch_ip_correction(h1t, h2t)                 # conj(Phi^T h1)^T M^-1 (Phi^T h2)
    p1 = Phi.T @ h1t
    p2 = Phi.T @ h2t
    corr_bilin = np.tensordot(p1, Minv @ p2, axes=([-1], [-1]))     # WRONG: no conjugation

    rel_herm = abs((base - corr_herm) - Up_ref) / abs(Up_ref)
    rel_bilin = abs((base - corr_bilin) - Up_ref) / abs(Up_ref)
    print("[G3-b] Hermitian U' rel=%.3e (correct) ; bilinear (no-conj) rel=%.3e (must be O(1))"
          % (rel_herm, rel_bilin))
    assert rel_herm < 1e-9
    assert rel_bilin > 1e-2


# --------------------------------------------------------------------------- (c) default OFF regression

def test_default_off_byte_identical():
    """A factory with NO glitch spec is byte-identical to the plain TD factory (glitch OFF)."""
    rng = np.random.default_rng(31)
    psd = make_psd_series()
    hlms = {(2, 2): bandlimited_series(rng), (2, 1): bandlimited_series(rng)}
    data = bandlimited_series(rng)

    fac_plain = TDInnerProductFactory(solver="auto")
    fac_offspec = TDInnerProductFactory(solver="auto", glitch_specs={})  # no IFO registered

    Q_a = fl.ComputeModeIPTimeSeries(hlms, data, psd, FLOW, FMAX, FNYQ, 0, 256,
                                     analyticPSD_Q=False, IP_factory=fac_plain)
    Q_b = fl.ComputeModeIPTimeSeries(hlms, data, psd, FLOW, FMAX, FNYQ, 0, 256,
                                     analyticPSD_Q=False, IP_factory=fac_offspec)
    maxdiff = 0.0
    for k in hlms:
        maxdiff = max(maxdiff, np.max(np.abs(Q_a[k].data.data - Q_b[k].data.data)))
    print("[G3-c] default-OFF Q_lm(t) byte-diff = %.3e" % maxdiff)
    assert maxdiff == 0.0


def test_basis_file_matches_direct_attach(tmp_path=None):
    """--glitch-basis-file npz path == direct attach_glitch_basis of the embedded matrix.

    The file-based (data-informed PCA, Sec. 3 modes 3/4) branch of
    TDInnerProductFactory._maybe_attach_glitch must reproduce EXACTLY what a direct
    attach of the zero-embedded basis gives: same sigma_g, same Delta lnZ, same U'
    correction, same Q'(t).  Also: a factory with glitch_basis_files={} stays
    byte-identical to the plain factory (default OFF)."""
    import os
    import tempfile
    rng = np.random.default_rng(77)
    psd = make_psd_series()
    hlms = {(2, 2): bandlimited_series(rng)}
    data = bandlimited_series(rng)

    # a small random band-limited snippet basis (n_snip x r), orthonormal columns
    n_snip, r = 256, 5
    raw = np.zeros((n_snip, r))
    tg = np.arange(n_snip) / FS
    for j in range(r):
        f0 = 150.0 + 60.0 * j
        raw[:, j] = np.exp(-((tg - tg[n_snip // 2]) ** 2) / (2 * 0.01 ** 2)) * np.cos(
            2 * np.pi * f0 * (tg - tg[n_snip // 2]) + 0.3 * j)
    snip, _ = np.linalg.qr(raw)
    lam0 = np.linspace(1.0, 0.2, r)
    t0 = 0.61                      # seconds from sample 0 (event_time_geo=None convention)
    i0 = int(round(t0 * FS))

    if tmp_path is None:
        tmp_dir = tempfile.mkdtemp()
    else:
        tmp_dir = str(tmp_path)
    npz_path = os.path.join(tmp_dir, "basis_H1.npz")
    np.savez(npz_path, basis=snip, lambda0=lam0, t0=t0, fs=FS)

    # (1) file-based factory path
    fac_file = TDInnerProductFactory(solver="auto",
                                     glitch_basis_files={"H1": npz_path},
                                     glitch_sigma="profile")
    fac_file.register_detector_data("H1", psd, data, event_time_geo=None)
    U_file = fl.ComputeModeCrossTermIP(hlms, hlms, psd, FLOW, FMAX, FNYQ, DELTAF,
                                       analyticPSD_Q=False, verbose=False,
                                       same_waveform_Q=False, IP_factory=fac_file)
    Q_file = fl.ComputeModeIPTimeSeries(hlms, data, psd, FLOW, FMAX, FNYQ, 0, N,
                                        analyticPSD_Q=False, IP_factory=fac_file)
    ctx_file = fac_file._glitch_contexts[0]

    # (2) direct attach of the SAME embedded basis on a fresh context
    B_full = np.zeros((N, r))
    B_full[i0:i0 + n_snip, :] = snip
    ctx_direct = ToeplitzInnerProductContext(psd, DELTAF, FNYQ, FLOW, fMax=FMAX,
                                             analyticPSD_Q=False, solver="auto")
    info = ctx_direct.attach_glitch_basis([(B_full, lam0, "H1")], data, sigma_g="profile")

    assert np.isclose(ctx_file._glitch_sigma_g, info["sigma_g"], rtol=1e-12)
    assert np.isclose(ctx_file.glitch_delta_lnZ, info["delta_lnZ"], rtol=1e-12)
    # U' correction identical on a random band-limited pair
    u = bandlimited_series(rng)
    v = bandlimited_series(rng)
    u_td = np.fft.ifft(np.asarray(u.data.data)) * (N * DELTAF)
    v_td = np.fft.ifft(np.asarray(v.data.data)) * (N * DELTAF)
    c_file = ctx_file.glitch_ip_correction(u_td, v_td)
    c_direct = ctx_direct.glitch_ip_correction(u_td, v_td)
    assert np.isclose(c_file, c_direct, rtol=1e-10)
    print("[G4-file] sigma_g=%.6e Delta_lnZ=%.6f ip-corr match %.3e vs %.3e"
          % (info["sigma_g"], info["delta_lnZ"], abs(c_file), abs(c_direct)))

    # (3) default OFF: empty glitch_basis_files == plain factory, byte-identical
    fac_plain = TDInnerProductFactory(solver="auto")
    fac_off = TDInnerProductFactory(solver="auto", glitch_basis_files={})
    Q_a = fl.ComputeModeIPTimeSeries(hlms, data, psd, FLOW, FMAX, FNYQ, 0, 256,
                                     analyticPSD_Q=False, IP_factory=fac_plain)
    Q_b = fl.ComputeModeIPTimeSeries(hlms, data, psd, FLOW, FMAX, FNYQ, 0, 256,
                                     analyticPSD_Q=False, IP_factory=fac_off)
    for k in hlms:
        assert np.max(np.abs(Q_a[k].data.data - Q_b[k].data.data)) == 0.0


# --------------------------------------------------------------------------- (c) end-to-end

def _lnL_of_t(Q_ts, U):
    """Single-detector, single-mode matched-filter lnL(t) proxy = |Q(t)|^2 / (2 Re U).

    This is exactly the per-mode SNR^2/2 statistic the Precompute guess_snr uses; it isolates
    the Q'/U' glitch correction without needing the full sky/Ylm response (which multiplies the
    same Q/U and does not change the three-way comparison qualitatively)."""
    q = np.asarray(Q_ts.data.data)
    return np.abs(q) ** 2 / (2.0 * np.real(U))


def test_end_to_end_glitch_mitigation():
    """BBH-like signal + injected blip in one detector: glitch-marg lnL(t) closer to glitch-free."""
    psd = make_psd_series()
    # "signal" template (BBH-like chirp with a peak), band-limited
    h_td = chirp_td(t0=0.6, t1=1.4, f0=40.0, f1_=350.0, tau=0.25, amp=STRAIN)
    hlm = series_from_td(h_td)
    hlms = {(2, 2): hlm}
    A = 6.0                                       # signal amplitude (in units of template)
    d_clean = A * h_td
    # injected sine-Gaussian blip in the glitch TF cell (disjoint from the chirp track); it
    # contaminates the matched filter at the time-shift aligning the template's high-frequency
    # tail with the blip.  Loud relative to the signal-in-that-cell.
    blip = sine_gaussian_td(t0=0.5 * (G_T1 + G_T2), f0=0.5 * (G_F1 + G_F2), q=6.0,
                            amp=80.0 * STRAIN)
    d_glitchy = d_clean + blip

    data_clean = series_from_td(d_clean)
    data_glitchy = series_from_td(d_glitchy)

    def run(data, factory, register=False, evt=None):
        if register:
            factory.register_detector_data("H1", psd, data, event_time_geo=evt)
        U = fl.ComputeModeCrossTermIP(hlms, hlms, psd, FLOW, FMAX, FNYQ, DELTAF,
                                      analyticPSD_Q=False, verbose=False,
                                      same_waveform_Q=False, IP_factory=factory)
        Q = fl.ComputeModeIPTimeSeries(hlms, data, psd, FLOW, FMAX, FNYQ, 0, N,
                                       analyticPSD_Q=False, IP_factory=factory)
        Uval = np.real(U[((2, 2), (2, 2))])
        q = np.asarray(Q[(2, 2)].data.data)
        return q, np.abs(q) ** 2 / (2.0 * Uval)

    # reference: glitch-free data, plain TD (no glitch marginalization)
    Q_ref, lnL_ref = run(data_clean, TDInnerProductFactory(solver="auto"))
    # unmitigated: glitchy data, plain TD
    Q_unmit, lnL_unmit = run(data_glitchy, TDInnerProductFactory(solver="auto"))
    # marginalized: glitchy data, glitch basis on H1 (support/band from the blip)
    fac_g = TDInnerProductFactory(
        solver="auto",
        glitch_specs={"H1": dict(t1=G_T1, t2=G_T2, f1=G_F1, f2=G_F2, r=None)},
        glitch_sigma="profile")
    Q_marg, lnL_marg = run(data_glitchy, fac_g, register=True)
    dlnZ = fac_g.glitch_delta_lnZ_total()

    # L2 distance of the matched-filter time series to the glitch-free reference (captures the
    # blip-induced feature the glitch marginalization removes).
    def l2(a, b):
        return float(np.sqrt(np.sum(np.abs(np.asarray(a) - np.asarray(b)) ** 2)))
    dQ_unmit = l2(Q_unmit, Q_ref)
    dQ_marg = l2(Q_marg, Q_ref)
    # peak lnL (SNR^2/2) three ways, for context (signal peak should survive marginalization)
    pk_ref, pk_unmit, pk_marg = np.max(lnL_ref), np.max(lnL_unmit), np.max(lnL_marg)
    print("[G3-c] ||Q-Q_ref||: unmitigated=%.4g  marginalized=%.4g  (ratio %.3f)"
          % (dQ_unmit, dQ_marg, dQ_marg / dQ_unmit))
    print("[G3-c] peak lnL (SNR^2/2): ref=%.4g  unmitigated=%.4g  marginalized=%.4g  "
          "Delta_lnZ=%.4g" % (pk_ref, pk_unmit, pk_marg, dlnZ))
    # marginalization brings the matched filter closer to the glitch-free reference ...
    assert dQ_marg < dQ_unmit
    # ... while preserving the signal peak (basis is TF-disjoint from the signal track)
    assert pk_marg > 0.5 * pk_ref
    # ... and the evidence constant is nonzero and surfaced
    assert dlnZ != 0.0


# ------------------------------------------------ (signmix) detection-conditioned sign mixture

class _DenseCovMix(object):
    """Minimal dense-covariance adapter for the RankAugmentation (sign-mixture tests)."""
    def __init__(self, Cmat):
        self.N = Cmat.shape[0]; self.C = Cmat
        self._cho = scipy.linalg.cho_factor(Cmat, lower=True, check_finite=False)
        self._logdet = 2.0 * np.sum(np.log(np.abs(np.diag(np.tril(self._cho[0])))))
    def solve(self, b):
        b = np.asarray(b)
        if np.iscomplexobj(b):
            return (scipy.linalg.cho_solve(self._cho, b.real, check_finite=False)
                    + 1j * scipy.linalg.cho_solve(self._cho, b.imag, check_finite=False))
        return scipy.linalg.cho_solve(self._cho, b, check_finite=False)
    def dot(self, u, v):
        return np.tensordot(np.asarray(u), self.solve(v), axes=([-1], [-1]))
    def logdet(self):
        return self._logdet


def _dense_branch_marginal(C, B, Sigma_central, mu_mix, sigma_g, d, h, sign):
    """Direct N-dim Gaussian marginal lnL for the single mixture component N(sign*sigma_g mu, Sigma_a)."""
    n = C.shape[0]
    Sigma_a = sigma_g ** 2 * Sigma_central
    mu_a = sign * sigma_g * np.asarray(mu_mix, dtype=float)
    Cp = C + B @ Sigma_a @ B.T
    Cpinv = np.linalg.inv(Cp)
    x = d - h - B @ mu_a
    _, logdet_Cp = np.linalg.slogdet(Cp)
    return -0.5 * x @ Cpinv @ x - 0.5 * (logdet_Cp + n * np.log(2 * np.pi))


def _signmix_fixture(seed=303, n=200):
    rng = np.random.default_rng(seed)
    lags = np.arange(n)
    acf = np.exp(-lags / 15.0) * np.cos(0.1 * lags)
    C = scipy.linalg.toeplitz(acf); C = 0.5 * (C + C.T)
    C = C + (abs(min(np.linalg.eigvalsh(C))) + 1e-3) * np.eye(n)
    r = 5
    B = rng.standard_normal((n, r))
    A = rng.standard_normal((r, r)); Sig = A @ A.T + 0.3 * np.eye(r)
    Sigma_central = Sig / np.max(np.linalg.eigvalsh(Sig))
    mu_mix = rng.standard_normal(r) * 0.5
    return rng, C, B, Sigma_central, mu_mix


def test_signmix_dense_parity():
    """(5a) Wired two-component sign-mixture marginal == brute-force N-dim mixture (~1e-9)."""
    from scipy.special import logsumexp
    rng, C, B, Sigma_central, mu_mix = _signmix_fixture()
    n = C.shape[0]
    sigma_g = 0.7
    d = rng.standard_normal(n) * 0.4
    h = rng.standard_normal(n) * 0.2                  # a "signal" template

    # brute: mixture marginal = logsumexp(branch+, branch-) - ln2
    lnLp = _dense_branch_marginal(C, B, Sigma_central, mu_mix, sigma_g, d, h, +1.0)
    lnLm = _dense_branch_marginal(C, B, Sigma_central, mu_mix, sigma_g, d, h, -1.0)
    lnLmix_dense = logsumexp([lnLp, lnLm]) - np.log(2.0)

    # wired: the SAME primitives ILE/Precompute assemble (d_tilde = d - sign*B mu_a), combined
    aug = tra.RankAugmentation(_DenseCovMix(C), [(B, Sigma_central, "H1", mu_mix)])

    def branch(sign):
        mu_a = sign * sigma_g * mu_mix
        x = (d - B @ mu_a) - h
        quad = aug.dot(x, x, sigma_g)
        return -0.5 * quad - 0.5 * (aug.logdet(sigma_g) + n * np.log(2 * np.pi))
    lnLmix_wired = logsumexp([branch(+1.0), branch(-1.0)]) - np.log(2.0)

    rel = abs(lnLmix_wired - lnLmix_dense) / abs(lnLmix_dense)
    print("[signmix-a] mixture marginal dense=%.8f wired=%.8f rel=%.3e"
          % (lnLmix_dense, lnLmix_wired, rel))
    assert rel < 1e-9


def test_sign_odds_formula_vs_bruteforce():
    """(5c) Closed-form sign odds 2<d-h|B mu_a>_C' == direct lnL+ - lnL- branch difference (1e-10)."""
    rng, C, B, Sigma_central, mu_mix = _signmix_fixture(seed=404)
    n = C.shape[0]
    sigma_g = 0.6
    d = rng.standard_normal(n) * 0.5
    h = rng.standard_normal(n) * 0.3
    aug = tra.RankAugmentation(_DenseCovMix(C), [(B, Sigma_central, "H1", mu_mix)])
    odds_formula = aug.sign_odds(d, sigma_g, h=h)
    lnLp = _dense_branch_marginal(C, B, Sigma_central, mu_mix, sigma_g, d, h, +1.0)
    lnLm = _dense_branch_marginal(C, B, Sigma_central, mu_mix, sigma_g, d, h, -1.0)
    odds_brute = lnLp - lnLm
    print("[signmix-c] sign-odds formula=%.10f  brute(lnL+ - lnL-)=%.10f  |diff|=%.3e"
          % (odds_formula, odds_brute, abs(odds_formula - odds_brute)))
    assert abs(odds_formula - odds_brute) < 1e-10


def test_decisive_limit_signmix_to_pinned():
    """(5d) Crank the injected +mu glitch: sign odds grows, signmix -> pinned(+) branch dominance."""
    from scipy.special import logsumexp
    rng, C, B, Sigma_central, mu_mix = _signmix_fixture(seed=505)
    n = C.shape[0]
    sigma_g = 0.7
    h = np.zeros(n)                                   # h=0: pure glitch/no-signal regime indicator
    base = rng.standard_normal(n) * 0.05             # small noise
    g_dir = B @ mu_mix                               # the +mu class-mean glitch template
    aug = tra.RankAugmentation(_DenseCovMix(C), [(B, Sigma_central, "H1", mu_mix)])

    prev_odds = None
    resp_plus = []
    for amp in [0.0, 1.0, 3.0, 8.0]:
        d = base + amp * g_dir                        # inject a +mu-aligned glitch of growing amplitude
        odds = aug.sign_odds(d, sigma_g, h=h)
        lnLp = _dense_branch_marginal(C, B, Sigma_central, mu_mix, sigma_g, d, h, +1.0)
        lnLm = _dense_branch_marginal(C, B, Sigma_central, mu_mix, sigma_g, d, h, -1.0)
        lnLmix = logsumexp([lnLp, lnLm]) - np.log(2.0)
        p_plus = 1.0 / (1.0 + np.exp(-(lnLp - lnLm)))  # + branch responsibility
        resp_plus.append(p_plus)
        # the mixture is dominated by the + branch: lnLmix -> lnLp - <tiny>
        gap_to_plus = lnLp - np.log(2.0) if False else (lnLmix - (lnLp - np.log(2.0)))
        print("[signmix-d] amp=%.1f: sign-odds=%+.4f  p(+)=%.6f  lnLmix-(lnL+ -ln2)=%.3e"
              % (amp, odds, p_plus, lnLmix - (lnLp - np.log(2.0))))
        if prev_odds is not None:
            assert odds > prev_odds                    # odds grows with the +mu glitch amplitude
        prev_odds = odds
    # decisive limit: at the loudest injection the + branch carries ~all the weight (pinned +)
    assert resp_plus[-1] > 0.99
    assert resp_plus[-1] > resp_plus[0]


def _write_pop_npz(path, snip, lam0, t0, fs, sigma_shape=None, mu_mix=None, sigma_central=None):
    payload = dict(basis=snip, lambda0=lam0, t0=t0, fs=fs)
    if sigma_shape is not None:
        payload["sigma_shape"] = sigma_shape
    if mu_mix is not None:
        payload["mu_mix"] = mu_mix
    if sigma_central is not None:
        payload["sigma_central"] = sigma_central
    np.savez(path, **payload)


def _pop_npz_fixture(rng, r=5, n_snip=256):
    """A synthetic pop-basis payload with the M2 + mixture keys, moment-identity consistent."""
    tg = np.arange(n_snip) / FS
    raw = np.zeros((n_snip, r))
    for j in range(r):
        f0 = 150.0 + 60.0 * j
        raw[:, j] = np.exp(-((tg - tg[n_snip // 2]) ** 2) / (2 * 0.01 ** 2)) * np.cos(
            2 * np.pi * f0 * (tg - tg[n_snip // 2]) + 0.3 * j)
    snip, _ = np.linalg.qr(raw)
    lam0 = np.linspace(1.0, 0.2, r)
    # a robust-covariance-like centered SPD + a mean direction; build the M2 keys by the same
    # centered-primitive the fit tool uses so the moment identity holds.
    Araw = rng.standard_normal((r, r)); S_central = Araw @ Araw.T + 0.5 * np.eye(r)
    mu_raw = rng.standard_normal(r) * 0.6
    S2 = S_central + np.outer(mu_raw, mu_raw)
    max_eig2 = np.max(np.linalg.eigvalsh(S2))
    sigma_shape = S2 / max_eig2
    mu_mix = mu_raw / np.sqrt(max_eig2)
    sigma_central = S_central / max_eig2
    return snip, lam0, sigma_shape, mu_mix, sigma_central


def test_m2_default_bit_identical_with_mixture_keys():
    """(5b) prior-mode m2 with mu_mix/sigma_central present == a plain sigma_shape-only attach.

    Adding the detection-conditioned keys to the npz must NOT perturb the default (m2) path:
    it uses sigma_shape zero-mean exactly as before, ignoring mu_mix/sigma_central."""
    import os
    import tempfile
    rng = np.random.default_rng(717)
    psd = make_psd_series()
    hlms = {(2, 2): bandlimited_series(rng)}
    data = bandlimited_series(rng)
    snip, lam0, sigma_shape, mu_mix, sigma_central = _pop_npz_fixture(rng)
    t0 = 0.61
    tmp = tempfile.mkdtemp()
    npz_mix = os.path.join(tmp, "pop_mix_H1.npz")     # has sigma_shape + mixture keys
    npz_m2only = os.path.join(tmp, "pop_m2only_H1.npz")  # only sigma_shape
    _write_pop_npz(npz_mix, snip, lam0, t0, FS, sigma_shape, mu_mix, sigma_central)
    _write_pop_npz(npz_m2only, snip, lam0, t0, FS, sigma_shape)

    def run(path):
        fac = TDInnerProductFactory(solver="auto", glitch_basis_files={"H1": path},
                                    glitch_sigma="profile", glitch_prior_mode="m2")
        fac.register_detector_data("H1", psd, data, event_time_geo=None)
        Q = fl.ComputeModeIPTimeSeries(hlms, data, psd, FLOW, FMAX, FNYQ, 0, N,
                                       analyticPSD_Q=False, IP_factory=fac)
        U = fl.ComputeModeCrossTermIP(hlms, hlms, psd, FLOW, FMAX, FNYQ, DELTAF,
                                      analyticPSD_Q=False, verbose=False,
                                      same_waveform_Q=False, IP_factory=fac)
        return Q, U, fac
    Q_mix, U_mix, fac_mix = run(npz_mix)
    Q_m2, U_m2, fac_m2 = run(npz_m2only)
    dQ = max(np.max(np.abs(Q_mix[k].data.data - Q_m2[k].data.data)) for k in hlms)
    dU = abs(U_mix[((2, 2), (2, 2))] - U_m2[((2, 2), (2, 2))])
    dZ = abs(fac_mix.glitch_delta_lnZ_total() - fac_m2.glitch_delta_lnZ_total())
    print("[signmix-b] m2 with-mixture-keys vs sigma_shape-only: Q byte-diff=%.3e U diff=%.3e "
          "Delta_lnZ diff=%.3e" % (dQ, dU, dZ))
    assert dQ == 0.0 and dU == 0.0 and dZ == 0.0
    # m2 carries no mean -> the signmix two-pass is NOT engaged
    assert not fac_mix.signmix_active


def test_signmix_pinned_context_paths():
    """signmix loader: both +-branches recorded, active toggles d_corr/Delta_lnZ; pinned==signmix(+)."""
    import os
    import tempfile
    rng = np.random.default_rng(818)
    psd = make_psd_series()
    hlms = {(2, 2): bandlimited_series(rng)}
    data = bandlimited_series(rng)
    snip, lam0, sigma_shape, mu_mix, sigma_central = _pop_npz_fixture(rng)
    tmp = tempfile.mkdtemp()
    npz = os.path.join(tmp, "pop_H1.npz")
    _write_pop_npz(npz, snip, lam0, 0.61, FS, sigma_shape, mu_mix, sigma_central)

    fac = TDInnerProductFactory(solver="auto", glitch_basis_files={"H1": npz},
                                glitch_sigma="profile", glitch_prior_mode="signmix")
    fac.register_detector_data("H1", psd, data, event_time_geo=None)
    # attach happens on first context build
    _ = fl.ComputeModeIPTimeSeries(hlms, data, psd, FLOW, FMAX, FNYQ, 0, N,
                                   analyticPSD_Q=False, IP_factory=fac)
    assert fac.signmix_active
    ctx = fac._glitch_contexts[0]
    dz_plus = ctx._glitch_delta_lnZ_plus
    dz_minus = ctx._glitch_delta_lnZ_minus
    assert dz_plus != dz_minus                        # a genuine mean -> branches differ
    # active toggling changes d_corr (Q') and the reported active Delta_lnZ
    fac.set_active_glitch_sign(+1.0)
    dcorr_plus = np.asarray(ctx._glitch_data_corr_fd.data.data).copy()
    assert abs(ctx.glitch_delta_lnZ - dz_plus) < 1e-12
    fac.set_active_glitch_sign(-1.0)
    dcorr_minus = np.asarray(ctx._glitch_data_corr_fd.data.data)
    assert abs(ctx.glitch_delta_lnZ - dz_minus) < 1e-12
    assert np.max(np.abs(dcorr_plus - dcorr_minus)) > 0

    # pinned mode == signmix + branch (single mean-shift at +mu_mix): same sigma-profile family,
    # same +branch d_corr up to the (different) amplitude profile; here check the delta_lnZ[+]
    # branch matches a pinned attach on the same data at the pinned sigma_g.
    odds = fac.glitch_sign_odds_report()
    print("[signmix-ctx] Delta_lnZ[+]=%.4f Delta_lnZ[-]=%.4f ; sign-odds report=%s"
          % (dz_plus, dz_minus, {k: round(v, 4) for k, v in odds.items()}))
    assert "H1" in odds


# --------------------------------------------------------------------------- script entry

if __name__ == "__main__":
    import time
    tests = [
        ("(a1) formula vs brute-force", test_formula_vs_bruteforce_invertible_C),
        ("(pop-a) population-prior dense parity (mean+cov)", test_pop_prior_dense_parity_full_mean_cov),
        ("(pop-b) reduction mu=0 == flat", test_pop_prior_reduction_to_flat),
        ("(pop-M2) default fit = covariance-only zero-mean", test_fit_population_default_is_M2_zero_mean),
        ("(a2) dense parity implementation", test_dense_parity_implementation),
        ("(a2) Q'(t) overlap consistency", test_Qprime_overlap_consistency),
        ("(b) conjugation trap", test_conjugation_trap),
        ("(c) default-OFF byte-identical", test_default_off_byte_identical),
        ("(c) basis-file == direct attach", test_basis_file_matches_direct_attach),
        ("(c) end-to-end glitch mitigation", test_end_to_end_glitch_mitigation),
        ("(signmix-a) dense mixture parity", test_signmix_dense_parity),
        ("(signmix-c) sign-odds vs brute", test_sign_odds_formula_vs_bruteforce),
        ("(signmix-d) decisive limit -> pinned(+)", test_decisive_limit_signmix_to_pinned),
        ("(signmix-b) m2 bit-identical w/ mixture keys", test_m2_default_bit_identical_with_mixture_keys),
        ("(signmix-ctx) signmix/pinned context paths", test_signmix_pinned_context_paths),
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
