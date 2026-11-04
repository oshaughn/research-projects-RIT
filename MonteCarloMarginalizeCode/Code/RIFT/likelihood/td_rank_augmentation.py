"""
td_rank_augmentation.py:  Rank-structured covariance augmentation for the RIFT
time-domain (TD) segment likelihood.

This module implements the glitch/line/waveform-uncertainty marginalization layer
described in TD_LIKELIHOOD.md (tasks G1, G2). The physical model, on a single
analysis segment of length N in one detector, is

    d = h(theta) + g + n,   n ~ N(0, C),   g = B a,   a ~ N(0, Lambda),

with

- C:  the base Toeplitz noise covariance (from the ACF; here dense, for SHORT segments).
- B:  a FIXED, low-rank glitch basis (N x r, r << N), independent of intrinsic and
      extrinsic parameters.  This fixedness is what keeps the marginalization cheap.
- Lambda = sigma_g^2 * Lambda_0:  a shaped Gaussian coefficient prior; sigma_g is the
      single amplitude hyperparameter, Lambda_0 the (fixed) shape.

Pinned conventions (TD_LIKELIHOOD.md Sec. 2b, binding on this module):
  * Units split.  Lambda_0 is a DIMENSIONLESS shape with max eigenvalue = 1; sigma_g
    carries physical STRAIN units, so Lambda = sigma_g^2 Lambda_0 reaches this layer in
    covariance units (~C[0,0] ~ 1e-42 for real ACFs) automatically.  Default sigma_g
    search bounds are RELATIVE to the covariance scale, [1e-3, 1e3] * sqrt(C[0,0]),
    log-spaced (see profile_sigma_g).
  * Inner products are BILINEAR.  dot(u, v) = u^T C'^{-1} v with NO conjugation (C' is
    real).  Hermitian forms are the caller's job (pass conj(u)).
  * The Woodbury r x r matrix is formed in the SCALED (dimensionless) representation
    M_hat = I + sigma_g^2 Lambda_0^{1/2} G Lambda_0^{1/2} rather than the direct
    M = Lambda_0^{-1}/sigma_g^2 + G, so the algebra stays well-conditioned at the physical
    covariance scale (C[0,0] ~ 1e-42, sigma_g ~ 1e-21) where Lambda_0^{-1}/sigma_g^2 ~ 1e42
    would otherwise swamp G.  The public API is unchanged; only the internals scale.
    logdet_correction = ln det M_hat exactly.

Marginalizing the coefficients a analytically gives  d - h ~ N(0, C')  with

    C'      = C + B Lambda B^T
    C'^{-1} = C^{-1} - C^{-1} B M^{-1} B^T C^{-1},   M = Lambda^{-1} + B^T C^{-1} B   (r x r)
    ln det C' = ln det C + ln det(I + Lambda B^T C^{-1} B).

Everything sigma_g-independent (Phi = C^{-1} B, G = B^T Phi, gamma = Phi^T d) is
precomputed once; varying sigma_g only touches the r x r matrix M.

Disturbance blocks in the same detector are concatenated, retaining their Gram
cross-correlations; independent detectors have separate updates:  C' = C + sum_k B_k Lambda_k B_k^T.

Basis builders (Sec. 3 of the design):
- slepian_basis:        the default, DPSS time-frequency-limited basis (minimal r for a
                        given time-bandwidth support, per the Slepian/prolate theorem).
- sine_gaussian_lattice: a secondary Morlet-Gabor lattice for irregular support.

Scope: numpy/scipy only, self-contained.  This module targets SHORT segments
(N <= ~2^14): C is formed densely and Cholesky-factored.  The iterative-solver path
(PCG / Gohberg-Semencul) for full-segment N ~ 10^5 is the subject of task T1/T2 and is
deliberately NOT implemented here (see ToeplitzCovariance for the guard).

Normalization convention (pinned by a test): for the Toeplitz C built from an ACF that
is the Wiener-Khinchin transform of a one-sided PSD S(f), the whitened inner product
reproduces the familiar FD matched-filter norm on long segments with smooth PSDs,

    v^T C^{-1} v  ~=  (4 / (N * fs)) * sum_k |vtilde_k|^2 / S(f_k) .

References: van Haasteren & Vallisneri (1407.1838) for the B-a-Woodbury formalism;
Slepian (1978) / Percival & Walden for DPSS; TD_LIKELIHOOD.md Sec. 1-4, 7.
"""

__author__ = "R. O'Shaughnessy and the RIFT TD team"

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.signal


# Segments larger than this are refused by ToeplitzCovariance: the dense O(N^2) storage
# and O(N^3) Cholesky are only appropriate for short segments (this module's target).
_MAX_DENSE_N = 1 << 14


def acf_from_psd(freqs_onesided, psd_onesided, fs, acf_duration):
    """Autocovariance function (ACF) of stationary noise from its one-sided PSD.

    Standard Wiener-Khinchin construction: the one-sided PSD S(f) is band-limited /
    interpolated onto the rfft frequency grid implied by (fs, df), converted to the
    two-sided spectrum, and inverse-real-FFT'd with the nfft*df scaling that makes the
    result an autocovariance in strain^2 units.  The ACF is then truncated to
    acf_duration seconds (n_keep = round(acf_duration * fs) + 1 samples).

    Normalization convention.  For the symmetric Toeplitz covariance C built from this
    ACF (C[i,j] = acf[|i-j|]), the whitened inner product reproduces the standard
    frequency-domain matched-filter norm on long segments with smooth PSDs:

        v^T C^{-1} v  ~=  (4 / (N * fs)) * sum_k |vtilde_k|^2 / S(f_k),

    where vtilde = rfft(v).  (This equivalence is pinned by a unit test.)

    Parameters
    ----------
    freqs_onesided : array_like
        One-sided frequency grid [Hz], non-negative, strictly increasing.
    psd_onesided : array_like
        One-sided PSD S(f) [1/Hz], same length as freqs_onesided.
    fs : float
        Sample rate [Hz].
    acf_duration : float
        Duration [s] of the returned ACF (number of positive lags kept).

    Returns
    -------
    acf : ndarray
        The autocovariance R(tau) sampled at tau = 0, 1/fs, 2/fs, ... for
        n_keep = round(acf_duration * fs) + 1 lags.
    """
    f = np.asarray(freqs_onesided, dtype=np.float64)
    s1 = np.asarray(psd_onesided, dtype=np.float64)
    if f.ndim != 1 or s1.ndim != 1 or f.size != s1.size:
        raise ValueError("freqs_onesided and psd_onesided must be 1D arrays of equal length.")

    # Frequency resolution of the target rfft grid.  We choose df from the input grid
    # spacing (assumed uniform; enforce it by resampling if not).
    df = np.median(np.diff(f))
    if not np.allclose(np.diff(f), df, rtol=1e-6, atol=1e-12):
        f_uniform = np.arange(f[0], f[-1] + 0.5 * df, df)
        s1 = np.interp(f_uniform, f, s1)
        f = f_uniform
        df = f[1] - f[0]

    # nfft so that fs = nfft*df; force even so rfft has a clean Nyquist bin.
    nfft = int(round(fs / df))
    if nfft < 2:
        raise ValueError("Computed nfft < 2; check the PSD spacing versus the sample rate.")
    if nfft % 2 == 1:
        nfft += 1

    # Interpolate/band-limit S(f) onto the exact rfft grid [0, fs/2] of length nfft//2+1.
    target_len = nfft // 2 + 1
    f_target = np.linspace(0.0, fs / 2.0, target_len)
    s1_grid = np.interp(f_target, f, s1, left=s1[0], right=s1[-1])

    # One-sided -> two-sided nonnegative part.  The two-sided spectrum is S/2 in the
    # interior; the DC and Nyquist bins carry the full one-sided value (they are their
    # own conjugates, no negative-frequency partner to share with).
    s2 = 0.5 * s1_grid.copy()
    s2[0] = s1_grid[0]
    s2[-1] = s1_grid[-1]

    # Wiener-Khinchin: R(tau) = irfft(S2) * (nfft * df).  irfft divides by nfft, and the
    # continuous->discrete measure contributes df, so R has strain^2 units.
    r = np.fft.irfft(s2, n=nfft) * (nfft * df)

    n_keep = int(round(acf_duration * fs)) + 1
    n_keep = min(n_keep, r.size)
    return r[:n_keep].astype(np.float64)


class ToeplitzCovariance(object):
    """Dense symmetric-Toeplitz noise covariance for a SHORT analysis segment.

    Built from an autocovariance function (ACF) of length >= N and a segment length N.
    The covariance is C[i,j] = acf[|i-j|]; it is formed densely with
    scipy.linalg.toeplitz and Cholesky-factored (cho_factor) once at construction.
    All subsequent operations (solve, quadratic form, log-determinant, whitening) reuse
    that factorization.

    This class deliberately targets short segments only.  For N above _MAX_DENSE_N it
    raises, with a pointer to the future iterative-solver (PCG / Gohberg-Semencul) path
    that task T1/T2 will provide -- dense O(N^2) storage and O(N^3) Cholesky do not
    scale to full-segment N ~ 10^5.
    """

    def __init__(self, acf, N):
        acf = np.asarray(acf, dtype=np.float64)
        if acf.ndim != 1:
            raise ValueError("acf must be a 1D array.")
        if N < 1:
            raise ValueError("N must be >= 1.")
        if acf.size < N:
            raise ValueError(
                "acf length ({0}) must be >= segment length N ({1}); increase acf_duration.".format(
                    acf.size, N))
        if N > _MAX_DENSE_N:
            raise ValueError(
                "ToeplitzCovariance is a dense short-segment solver (N <= {0}); got N = {1}. "
                "The full-segment path (PCG / Gohberg-Semencul iterative solver) is task "
                "T1/T2 and is not implemented in this module.".format(_MAX_DENSE_N, N))

        self.N = int(N)
        self.acf = acf
        col = acf[:N]
        self.C = scipy.linalg.toeplitz(col)
        # cho_factor: lower/upper triangular Cholesky, reused by cho_solve.
        self._cho = scipy.linalg.cho_factor(self.C, lower=True, check_finite=False)
        # ln det C = 2 * sum ln(diag(L)).
        L = np.tril(self._cho[0]) if self._cho[1] else np.triu(self._cho[0])
        self._logdet = 2.0 * np.sum(np.log(np.abs(np.diag(L))))

    def solve(self, b):
        """Return C^{-1} b (real or complex b; b may be a matrix, columns solved)."""
        b = np.asarray(b)
        if np.iscomplexobj(b):
            xr = scipy.linalg.cho_solve(self._cho, b.real, check_finite=False)
            xi = scipy.linalg.cho_solve(self._cho, b.imag, check_finite=False)
            return xr + 1j * xi
        return scipy.linalg.cho_solve(self._cho, b, check_finite=False)

    def dot(self, u, v):
        """Return the BILINEAR quadratic form u^T C^{-1} v (NO conjugation; Sec. 2b).

        C is real, so C^{-1} v is computed on the real and imaginary parts separately.
        For complex u, v this returns the bilinear form sum_i u_i (C^{-1} v)_i WITHOUT
        conjugation, matching the convention used throughout the augmentation algebra
        (Woodbury identities are bilinear).  Callers wanting a Hermitian form pass conj(u).
        """
        Cinv_v = self.solve(v)
        u = np.asarray(u)
        return np.tensordot(u, Cinv_v, axes=([-1], [-1]))

    def logdet(self):
        """Return ln det C."""
        return self._logdet

    def whiten(self, v):
        """Return L^{-1} v, the whitened data (so that ||whiten(v)||^2 = v^T C^{-1} v).

        Uses the Cholesky factor L (C = L L^T): solving L x = v gives x with
        x^T x = v^T (L L^T)^{-1} v = v^T C^{-1} v.
        """
        v = np.asarray(v)
        L = self._cho[0]
        lower = self._cho[1]

        def _solve_L(rhs):
            return scipy.linalg.solve_triangular(
                L, rhs, lower=lower, trans=0, check_finite=False)

        if np.iscomplexobj(v):
            return _solve_L(v.real) + 1j * _solve_L(v.imag)
        return _solve_L(v)


class RankAugmentation(object):
    """Woodbury-corrected covariance augmentation C' = C + sum_k B_k Lambda_k B_k^T.

    The supplied blocks live in ONE detector and have independent coefficient
    priors, not independent noise-weighted projections. Multiple blocks are
    concatenated into B with a block-diagonal Lambda0; the joint Gram matrix
    B.T C^{-1} B retains cross-correlations between their columns. Independent
    detectors require separate RankAugmentation objects.

    The resulting joint update is stored as one entry in blocks. All primitives
    use its full Woodbury matrix M = Lambda^{-1} + B.T C^{-1} B.
    """

    def __init__(self, toeplitz_cov, blocks):
        self.cov = toeplitz_cov
        self.N = toeplitz_cov.N
        blocks = list(blocks)
        if len(blocks) > 1:
            bases, priors, means, labels = [], [], [], []
            for entry in blocks:
                if len(entry) not in (2, 3, 4):
                    raise ValueError("Each block must be (B, Lambda0[, ifo_label[, mu0]]).")
                B = np.asarray(entry[0], dtype=np.float64)
                if B.ndim != 2 or B.shape[0] != self.N:
                    raise ValueError("Every block basis must have shape (N, r).")
                r = B.shape[1]
                prior = np.asarray(entry[1], dtype=np.float64)
                if prior.ndim == 1 and prior.shape == (r,):
                    prior = np.diag(prior)
                if prior.shape != (r, r):
                    raise ValueError("Every block prior must have shape (r,) or (r, r).")
                mean = (np.asarray(entry[3], dtype=np.float64).reshape(-1)
                        if len(entry) == 4 and entry[3] is not None else np.zeros(r))
                if mean.shape != (r,):
                    raise ValueError("Every block mean must have length r.")
                bases.append(B)
                priors.append(prior)
                means.append(mean)
                labels.append(entry[2] if len(entry) >= 3 else None)
            blocks = [(np.concatenate(bases, axis=1), scipy.linalg.block_diag(*priors),
                       tuple(labels), np.concatenate(means))]
        self.blocks = []
        for entry in blocks:
            # A block is (B, Lambda0[, ifo_label[, mu0]]).  mu0 is the OPTIONAL dimensionless
            # population-mean DIRECTION mu_hat (length r); absent => zero mean (zero-mean Gaussian flat
            # behavior, bit-identical).  In the recommended scale-mixture convention the
            # physical prior mean is mu_a(sigma_g) = sigma_g * mu0 and the physical prior
            # covariance is Sigma_a(sigma_g) = sigma_g^2 * Lambda0 (Lambda0 == Sigma_shape,
            # max-eig 1); a SINGLE sigma_g carries the strain scale for both (population-prior).
            mu0 = None
            if len(entry) == 4:
                B, Lambda0, label, mu0 = entry
            elif len(entry) == 3:
                B, Lambda0, label = entry
            elif len(entry) == 2:
                B, Lambda0 = entry
                label = None
            else:
                raise ValueError("Each block must be (B, Lambda0[, ifo_label[, mu0]]).")
            B = np.asarray(B, dtype=np.float64)
            if B.ndim != 2 or B.shape[0] != self.N:
                raise ValueError(
                    "Block basis B must have shape (N, r) with N = {0}; got {1}.".format(
                        self.N, B.shape))
            r = B.shape[1]
            Lambda0 = np.asarray(Lambda0, dtype=np.float64)
            # Accept either a length-r vector (diagonal) or a full (r,r) matrix.
            # Lambda0 is a DIMENSIONLESS shape prior with max eigenvalue = 1 (Sec. 2b);
            # sigma_g carries the physical strain scale.  We also cache the symmetric
            # matrix square root Lambda0^{1/2}, used to form the scaled Woodbury matrix
            # M_hat = I + sigma_g^2 Lambda0^{1/2} G Lambda0^{1/2} (well-conditioned at the
            # physical covariance scale, unlike the direct M = Lambda0^{-1}/sigma_g^2 + G).
            if Lambda0.ndim == 1:
                if Lambda0.size != r:
                    raise ValueError("Diagonal Lambda0 must have length r = {0}.".format(r))
                if np.any(Lambda0 <= 0):
                    raise ValueError("Diagonal Lambda0 entries must be strictly positive.")
                Lambda0_mat = np.diag(Lambda0)
                Lambda0_inv = np.diag(1.0 / Lambda0)
                Lambda0_half = np.diag(np.sqrt(Lambda0))
            elif Lambda0.ndim == 2:
                if Lambda0.shape != (r, r):
                    raise ValueError("Matrix Lambda0 must be (r, r) = ({0},{0}).".format(r))
                Lambda0_mat = 0.5 * (Lambda0 + Lambda0.T)
                Lambda0_inv = np.linalg.inv(Lambda0_mat)
                evals, evecs = np.linalg.eigh(Lambda0_mat)
                if np.any(evals <= 0):
                    raise ValueError("Matrix Lambda0 must be positive definite.")
                Lambda0_half = (evecs * np.sqrt(evals)).dot(evecs.T)
            else:
                raise ValueError("Lambda0 must be 1D (diagonal) or 2D (r x r).")

            max_eig = np.max(np.linalg.eigvalsh(Lambda0_mat))
            if not np.isclose(max_eig, 1.0, rtol=1e-6, atol=1e-9):
                # Sec. 2b: Lambda0 is a shape prior with max eigenvalue 1; the amplitude
                # lives in sigma_g.  Warn but do not mutate (basis builders normalize).
                import warnings
                warnings.warn(
                    "Lambda0 max eigenvalue = {0:.3e} != 1; the units convention (Sec. 2b) "
                    "expects a shape prior with max eig 1 and sigma_g carrying the strain "
                    "scale.".format(max_eig), stacklevel=2)

            Phi = self.cov.solve(B)          # C^{-1} B, (N x r)
            G = B.T.dot(Phi)                 # B^T C^{-1} B, (r x r); symmetric
            G = 0.5 * (G + G.T)
            # G_hat = Lambda0^{1/2} G Lambda0^{1/2}: the symmetric core of M_hat.
            G_hat = Lambda0_half.dot(G).dot(Lambda0_half)
            G_hat = 0.5 * (G_hat + G_hat.T)
            # Population-mean direction mu_hat (dimensionless).  None / all-zero => flat.
            if mu0 is None:
                mu0_vec = np.zeros(r, dtype=np.float64)
            else:
                mu0_vec = np.asarray(mu0, dtype=np.float64).reshape(-1)
                if mu0_vec.size != r:
                    raise ValueError(
                        "Population mean mu0 must have length r = {0}; got {1}.".format(
                            r, mu0_vec.size))
            self.blocks.append(dict(
                B=B, r=r, label=label,
                Lambda0=Lambda0_mat, Lambda0_inv=Lambda0_inv, Lambda0_half=Lambda0_half,
                Phi=Phi, G=G, G_hat=G_hat, mu0=mu0_vec,
                has_mean=bool(np.any(mu0_vec != 0.0))))

    # ---- per-block sigma_g-dependent pieces -------------------------------------

    def M(self, sigma_g, k=0):
        """Return the (direct) Woodbury matrix M_k = Lambda_k^{-1} + G_k for block k.

        Lambda_k = sigma_g^2 Lambda0_k, so Lambda_k^{-1} = Lambda0_inv / sigma_g^2.

        NOTE: this direct form is provided for reference/inspection only.  All internal
        Woodbury solves use the SCALED, well-conditioned representation
        M_hat = I + sigma_g^2 Lambda0^{1/2} G Lambda0^{1/2} (see _M_hat_cho and Sec. 2b),
        which relates to M by  M^{-1} = sigma_g^2 Lambda0^{1/2} M_hat^{-1} Lambda0^{1/2}.
        """
        blk = self.blocks[k]
        return blk["Lambda0_inv"] / (sigma_g ** 2) + blk["G"]

    def M_hat(self, sigma_g, k=0):
        """Return the scaled Woodbury matrix M_hat_k = I + sigma_g^2 G_hat_k.

        G_hat_k = Lambda0^{1/2} G_k Lambda0^{1/2}.  Dimensionless and well-conditioned at
        the physical covariance scale; ln det M_hat_k == ln det(I + sigma_g^2 Lambda0 G).
        """
        blk = self.blocks[k]
        return np.eye(blk["r"]) + (sigma_g ** 2) * blk["G_hat"]

    def _M_hat_cho(self, sigma_g, k):
        return scipy.linalg.cho_factor(self.M_hat(sigma_g, k), lower=True, check_finite=False)

    def _Minv_apply(self, y, sigma_g, k):
        """Apply M_k^{-1} to a vector/matrix y (last axis = r), via the scaled form.

        M^{-1} y = sigma_g^2 Lambda0^{1/2} M_hat^{-1} (Lambda0^{1/2} y).  Real or complex y
        (solves act on real/imag parts; the algebra is bilinear, no conjugation).
        """
        blk = self.blocks[k]
        L0h = blk["Lambda0_half"]
        Mhat_cho = self._M_hat_cho(sigma_g, k)

        def _solve_real(z):  # z: (..., r) real -> M_hat^{-1} applied on last axis
            return scipy.linalg.cho_solve(Mhat_cho, z.T, check_finite=False).T

        y = np.asarray(y)
        # Lambda0^{1/2} y  (contract last axis with L0h)
        z = np.tensordot(y, L0h, axes=([-1], [0]))
        if np.iscomplexobj(z):
            w = _solve_real(z.real) + 1j * _solve_real(z.imag)
        else:
            w = _solve_real(z)
        # sigma_g^2 Lambda0^{1/2} w
        return (sigma_g ** 2) * np.tensordot(w, L0h, axes=([-1], [0]))

    def _gamma(self, d, k):
        """gamma_k = Phi_k^T d for the supplied data vector d."""
        return self.blocks[k]["Phi"].T.dot(np.asarray(d))

    def mu_a(self, sigma_g, k=0, sign=1.0):
        """Physical prior mean mu_a = sign * sigma_g * mu0 for block k (scale-mixture convention).

        Zero vector for a flat (no-mean) block.  In the scale mixture the prior-predicted
        glitch is g_hat_0 = B mu_a = sigma_g B mu0, scaling with the amplitude hyperparameter.

        ``sign`` selects the sign-mixture COMPONENT (population-prior, Eq. signmix): the
        detection-conditioned prior p(a|det) = 1/2 N(+sigma_g mu_hat, Sigma) + 1/2
        N(-sigma_g mu_hat, Sigma) is a two-component mixture over the discrete sign latent;
        ``sign=+1`` / ``sign=-1`` pick the two components, ``sign=+1`` (default) reproduces the
        single mean-shifted (pinned / FULL) path bit-identically.
        """
        return float(sign) * sigma_g * self.blocks[k]["mu0"]

    def _gamma_tilde(self, d, sigma_g, k, sign=1.0):
        r"""Mean-shifted gamma_tilde_k = Phi_k^T (d - B mu_a) = gamma_k - G_k mu_a (population-prior).

        Uses Phi^T B = B^T C^{-1} B = G, so no extra basis apply is needed.  For a flat block
        (mu0 = 0) this is exactly gamma_k, so the reduction to paper 1 is bit-identical.
        ``sign`` selects the sign-mixture component (see ``mu_a``).
        """
        blk = self.blocks[k]
        gamma = blk["Phi"].T.dot(np.asarray(d))
        if not blk["has_mean"]:
            return gamma
        return gamma - blk["G"].dot(self.mu_a(sigma_g, k, sign=sign))

    # ---- Woodbury-corrected primitives ------------------------------------------

    def solve(self, d, sigma_g):
        """Return C'^{-1} d (Woodbury-corrected)."""
        d = np.asarray(d)
        out = self.cov.solve(d)
        for k, blk in enumerate(self.blocks):
            Phi = blk["Phi"]
            gamma = Phi.T.dot(d)
            corr = self._Minv_apply(gamma, sigma_g, k)
            out = out - Phi.dot(corr)
        return out

    def dot(self, u, v, sigma_g):
        """Return the bilinear form u^T C'^{-1} v (Woodbury-corrected).

        BILINEAR: no conjugation is applied (C' is real); pass conj(u) for a Hermitian
        form.  Supports complex u, v (C real; solves act on real/imag parts).
        """
        u = np.asarray(u)
        v = np.asarray(v)
        val = self.cov.dot(u, v)
        for k, blk in enumerate(self.blocks):
            Phi = blk["Phi"]
            pu = np.tensordot(u, Phi, axes=([-1], [0]))   # Phi^T u (last axis contracted)
            pv = np.tensordot(v, Phi, axes=([-1], [0]))   # Phi^T v
            Minv_pv = self._Minv_apply(pv, sigma_g, k)
            val = val - np.tensordot(pu, Minv_pv, axes=([-1], [-1]))
        return val

    def logdet_correction(self, sigma_g):
        """Return ln det(C') - ln det(C) = sum_k ln det M_hat_k.

        ln det(I + sigma_g^2 Lambda0_k G_k) = ln det(I + sigma_g^2 Lambda0^{1/2} G
        Lambda0^{1/2}) = ln det M_hat_k (similarity transform preserves the determinant).
        Computed from the scaled, symmetric-positive-definite M_hat for stability.
        """
        total = 0.0
        for k, blk in enumerate(self.blocks):
            sign, logabs = np.linalg.slogdet(self.M_hat(sigma_g, k))
            total += logabs
        return total

    def logdet(self, sigma_g):
        """Return the full ln det C' = ln det C + logdet_correction."""
        return self.cov.logdet() + self.logdet_correction(sigma_g)

    def delta_lnZ(self, d, sigma_g, sign=1.0):
        r"""Return the glitch evidence term Delta lnZ(sigma_g) for data d.

        This is the glitch/no-glitch data-level log Bayes factor (h = 0): the marginalized
        data self-normalization minus the zero-mean Gaussian no-glitch one.  For a general population
        prior a ~ N(mu_a, Sigma_a), Sigma_a = sigma_g^2 Lambda0, mu_a = sigma_g mu0, completing
        the square with ALL constants retained gives (population-prior, derived + tested):

            Delta lnZ = sum_k [ mu_a^T gamma_k - 1/2 mu_a^T G_k mu_a
                                + 1/2 gamma_tilde_k^T M_k^{-1} gamma_tilde_k ]
                        - 1/2 ln det(I + sigma_g^2 Lambda0 G)

        with gamma_k = Phi_k^T d and gamma_tilde_k = gamma_k - G_k mu_a.  The first two
        (mean) terms are the paper's carried "-1/2 mu^T Sigma_a^-1 mu + data self-term"
        constants that change with mu; they VANISH when mu0 = 0, reducing bit-identically to
        the zero-mean Gaussian form  1/2 gamma^T M^-1 gamma - 1/2 ln det(I + sigma_g^2 Lambda0 G).

        The quadratic term is the power the glitch model absorbs; the log-det is the Occam
        penalty.  Their balance disciplines the glitch amplitude (Sec. 4).
        """
        d = np.asarray(d)
        total = 0.0
        for k, blk in enumerate(self.blocks):
            gamma = blk["Phi"].T.dot(d)
            if blk["has_mean"]:
                mu_a = self.mu_a(sigma_g, k, sign=sign)
                gamma_t = gamma - blk["G"].dot(mu_a)
                # mean terms: mu_a^T gamma - 1/2 mu_a^T G mu_a
                total += float(np.real(np.dot(mu_a, gamma)))
                total -= 0.5 * float(np.real(mu_a.dot(blk["G"]).dot(mu_a)))
            else:
                gamma_t = gamma
            Minv_gt = self._Minv_apply(gamma_t, sigma_g, k)
            total += 0.5 * np.real(np.vdot(gamma_t, Minv_gt))
        total -= 0.5 * self.logdet_correction(sigma_g)
        return total

    def delta_lnZ_signmix(self, d, sigma_g):
        r"""Two-component sign-mixture glitch evidence (population-prior, Eq. signmix).

        The detection-conditioned prior p(a|det) = 1/2 N(+sigma_g mu_hat, Sigma_a) + 1/2
        N(-sigma_g mu_hat, Sigma_a) is a mixture over a discrete sign latent that is
        INDEPENDENT of every other parameter, so its marginal combines exactly at the lnL
        level:

            Delta lnZ_mix = logsumexp(Delta lnZ(+mu), Delta lnZ(-mu)) - ln 2.

        Each branch's ``delta_lnZ`` already carries its own (sign-dependent) mean constants,
        so the combination is exact.  For a flat block (mu0 = 0) both branches coincide and
        this reduces to ``delta_lnZ`` exactly (logsumexp(x, x) - ln2 == x)."""
        from scipy.special import logsumexp
        dzp = self.delta_lnZ(d, sigma_g, sign=+1.0)
        dzm = self.delta_lnZ(d, sigma_g, sign=-1.0)
        return float(logsumexp([dzp, dzm]) - np.log(2.0)), float(dzp), float(dzm)

    def sign_odds(self, d, sigma_g, h=None, k=0):
        r"""Closed-form sign log-odds ln(L+/L-) = 2 <d - h | B mu_a>_{C'} (population-prior).

        The posterior weight between the +mu and -mu mixture components is computable in
        closed form from the already-precomputed Woodbury pieces (no new solves): twice the
        C'-weighted matched-filter overlap of the residual ``d - h`` with the mean-glitch
        template ``B mu_a = sigma_g B mu0``.  ``h`` (the reference/fiducial template, in the
        same TD packing as ``d``) defaults to 0 -- the data-only regime indicator
        2 <d | B mu_a>_{C'}, i.e. the matched-filter SNR of the glitch's projection onto the
        class mean.  Decisive |odds| >> 1 is where a pinned mean is legitimate (loud,
        well-localized glitch); moderate odds is the sign-undecided regime the mixture
        refines.  Returns a real scalar (C' is real).

        Derivation: lnL(+) - lnL(-) with C' shared between the two mixture components leaves
        only the cross term, = 2 (d-h)^T C'^{-1} (B mu_a) (the sign-independent logdet and
        |B mu_a|^2 terms cancel).  Uses ``dot`` (the bilinear C'^{-1} form)."""
        blk = self.blocks[k]
        mu_a = self.mu_a(sigma_g, k, sign=+1.0)      # +mu component's mean-glitch coefficients
        Bmu = blk["B"].dot(mu_a)                      # B mu_a, length N
        r = np.asarray(d) if h is None else (np.asarray(d) - np.asarray(h))
        # 2 <d-h | B mu_a>_{C'} = 2 (d-h)^T C'^{-1} (B mu_a), bilinear (C' real)
        return 2.0 * float(np.real(self.dot(r, Bmu, sigma_g)))

    def default_sigma_g_bounds(self, sigma_g_min=None):
        """Return the default profile/prior bounds (lo, hi) for sigma_g (Sec. 2b).

        Relative to the covariance scale: [1e-3, 1e3] * sqrt(C[0,0]).  This keeps both
        O(1) unit-test covariances and physical ~1e-42 covariances in range with the same
        code, and avoids the underflow trap (an O(1) prior against a 1e-42 covariance makes
        Lambda^{-1} negligible in M and the prior silently improper).

        ``sigma_g_min`` (strain units, population-prior presence conditioning: a DETECTED
        glitch means sigma_g is not small) raises the lower bound to
        ``max(1e-3*sqrt(C[0,0]), sigma_g_min)`` -- a per-event run-config knob (default None
        = the code default), never a code default.
        """
        scale = np.sqrt(self.cov.C[0, 0])
        lo = 1e-3 * scale
        if sigma_g_min is not None:
            lo = max(lo, float(sigma_g_min))
        hi = 1e3 * scale
        if lo >= hi:
            hi = lo * 1e3
        return (lo, hi)

    def profile_sigma_g(self, d, bounds=None, sigma_g_min=None):
        """Maximize Delta lnZ(sigma_g) over sigma_g in `bounds` = (lo, hi).

        Returns (sigma_g_hat, delta_lnZ_hat).  Uses a bounded scalar maximization
        (scipy.optimize.minimize_scalar on -Delta lnZ) in log(sigma_g) for good
        conditioning over the typically wide amplitude range.  If `bounds` is None, the
        default [1e-3, 1e3] * sqrt(C[0,0]) log-spaced range (Sec. 2b) is used, floored by
        ``sigma_g_min`` (presence conditioning) when supplied.
        """
        if bounds is None:
            bounds = self.default_sigma_g_bounds(sigma_g_min=sigma_g_min)
        lo, hi = bounds
        if lo <= 0:
            raise ValueError("profile_sigma_g bounds must be strictly positive.")
        loglo, loghi = np.log(lo), np.log(hi)

        def neg(log_sig):
            return -self.delta_lnZ(d, np.exp(log_sig))

        res = scipy.optimize.minimize_scalar(
            neg, bounds=(loglo, loghi), method="bounded")
        sigma_hat = float(np.exp(res.x))
        return sigma_hat, -float(res.fun)

    def profile_sigma_g_signmix(self, d, bounds=None, sigma_g_min=None):
        """Maximize the sign-MIXTURE evidence Delta lnZ_mix(sigma_g) over sigma_g.

        The empirical-Bayes amplitude for the detection-conditioned two-component sign
        mixture (population-prior): a single sigma_g shared by both +-mu components.  Returns
        (sigma_g_hat, delta_lnZ_mix_hat).  For a flat block this coincides with
        ``profile_sigma_g`` (both branches equal)."""
        if bounds is None:
            bounds = self.default_sigma_g_bounds(sigma_g_min=sigma_g_min)
        lo, hi = bounds
        if lo <= 0:
            raise ValueError("profile_sigma_g_signmix bounds must be strictly positive.")
        loglo, loghi = np.log(lo), np.log(hi)

        def neg(log_sig):
            return -self.delta_lnZ_signmix(d, np.exp(log_sig))[0]

        res = scipy.optimize.minimize_scalar(
            neg, bounds=(loglo, loghi), method="bounded")
        return float(np.exp(res.x)), -float(res.fun)

    def glitch_reconstruction(self, d, sigma_g, k=None, sign=1.0):
        r"""Return the MAP glitch reconstruction g_hat_total = B (mu_a + M^{-1} gamma_tilde).

        This is the posterior-mean glitch under the Gaussian coefficient prior a ~ N(mu_a,
        Sigma_a).  The posterior mean coefficient is

            a_hat = mu_a + M^{-1} (Phi^T d - G mu_a) = mu_a + M^{-1} gamma_tilde,

        i.e. the prior-predicted piece mu_a = sigma_g mu0 plus the data-driven update on the
        mean-shifted residual gamma_tilde = gamma - G mu_a.  With mu0 = 0 this collapses to
        the zero-mean Gaussian  a_hat = M^{-1} gamma  and g_hat = B M^{-1} gamma (bit-identical).  This
        g_hat_total is exactly the vector subtracted from the data to form the corrected
        filter (w' = C'^{-1} d_tilde = P(d - g_hat_total); see td_inner_product), so
        Q'_lm(t) = overlap(h, d - g_hat_total) holds unchanged.  With k=None the block
        reconstructions are summed; pass an integer k for a single block.
        """
        d = np.asarray(d)
        ks = range(len(self.blocks)) if k is None else [k]
        g_hat = np.zeros(self.N, dtype=float)
        for kk in ks:
            blk = self.blocks[kk]
            gamma_t = self._gamma_tilde(d, sigma_g, kk, sign=sign)
            a_hat = self._Minv_apply(gamma_t, sigma_g, kk)   # M^{-1} gamma_tilde, scaled form
            if blk["has_mean"]:
                a_hat = a_hat + self.mu_a(sigma_g, kk, sign=sign)  # + prior-predicted piece
            g_hat = g_hat + blk["B"].dot(a_hat)
        return g_hat


# ---------------------------------------------------------------------------------
# Basis builders (Sec. 3 of the design)
# ---------------------------------------------------------------------------------

def slepian_basis(N, fs, t_start, t_end, f_low, f_high, r=None):
    """DPSS (Slepian) time-frequency-limited disturbance basis.

    Builds a low-rank basis B (N x r) spanning signals concentrated in time to
    [t_start, t_end] and in frequency to [f_low, f_high].  The prolate spheroidal
    (DPSS) sequences on the support samples are the optimal (maximally band-concentrated)
    orthonormal set for a given time-bandwidth product; modulating them to the band
    center with cos/sin quadrature pairs shifts that concentration from the baseband
    [0, W] to the band [f_low, f_high].  The columns are embedded (zero-padded) into
    length-N vectors supported only on [t_start, t_end].

    The number of well-concentrated modes is governed by the time-bandwidth theorem:
    approximately 2 * W * T real DPSS tapers (W = half the bandwidth here, since the
    baseband taper covers +-W around the carrier), doubled by the cos/sin pairing.

    Parameters
    ----------
    N : int
        Segment length (samples).
    fs : float
        Sample rate [Hz].
    t_start, t_end : float
        Support interval [s], measured from the start of the segment (sample 0 at t=0).
    f_low, f_high : float
        Band edges [Hz].
    r : int or None
        Positive even number of columns (paired cosine/sine quadratures). If None, r = 2 * (ceil(2 * W * T) + 4) where W is the band
        half-width and T the support duration (time-bandwidth theorem + buffer, doubled
        for the cos/sin pair).

    Returns
    -------
    B : ndarray, (N, r_out)
        Orthonormal (columns) basis, supported on [t_start, t_end].  r_out may be < r if
        the support is too short to host that many tapers.
    Lambda0 : ndarray, (r_out,)
        Diagonal shape prior from the DPSS eigenvalue concentrations, normalized so that
        its largest entry is 1 (sigma_g carries the overall strain-amplitude scale).
    """
    i0 = int(np.floor(t_start * fs))
    i1 = int(np.ceil(t_end * fs))
    i0 = max(i0, 0)
    i1 = min(i1, N)
    Msup = i1 - i0
    if Msup < 2:
        raise ValueError("Support [t_start, t_end] spans < 2 samples on the segment.")

    T = Msup / fs                       # support duration [s]
    W = 0.5 * (f_high - f_low)          # baseband half-bandwidth [Hz]
    fc = 0.5 * (f_low + f_high)         # carrier / band center [Hz]
    if W <= 0:
        raise ValueError("f_high must exceed f_low.")

    # NW = normalized half-bandwidth in DPSS convention: (baseband bandwidth W) * (Msup
    # samples) / fs = W * T.  This is the time-bandwidth product per taper family.
    NW = W * T

    # Number of real tapers requested (per cos/sin channel).
    if r is None:
        n_real = int(np.ceil(2.0 * W * T)) + 4
        # r counts total columns after cos/sin doubling.
    else:
        if (isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer))
                or r <= 0 or r % 2):
            raise ValueError("Slepian rank r must be a positive even integer.")
        n_real = int(r) // 2

    # scipy.signal.windows.dpss needs 1 <= Kmax <= Msup, and NW must be > 0.5 for a
    # meaningful concentration; clamp gently.
    # There cannot be more independent supported columns than support samples.
    # Keep complete quadrature pairs and keep reduced QR square in column space.
    n_real = min(n_real, Msup // 2)
    NW_use = max(NW, 0.5 + 1e-6)

    tapers, eigvals = scipy.signal.windows.dpss(
        Msup, NW_use, Kmax=n_real, sym=False, return_ratios=True)
    # tapers: (n_real, Msup); eigvals: (n_real,) concentration ratios in (0, 1].

    tt = (np.arange(Msup) + i0) / fs    # absolute time of each support sample
    carrier_c = np.cos(2.0 * np.pi * fc * tt)
    carrier_s = np.sin(2.0 * np.pi * fc * tt)

    cols = []
    lams = []
    for j in range(n_real):
        taper = tapers[j]
        for carrier in (carrier_c, carrier_s):
            v = taper * carrier
            col = np.zeros(N, dtype=float)
            col[i0:i1] = v
            cols.append(col)
            lams.append(eigvals[j])

    B = np.array(cols).T                # (N, 2*n_real)
    # Orthonormalize the columns (cos/sin modulation of the same taper is nearly
    # orthogonal but QR pins it exactly and drops any rank-deficient columns).
    Q, R = np.linalg.qr(B)
    keep = np.abs(np.diag(R)) > 1e-10 * np.max(np.abs(np.diag(R)))
    B = Q[:, keep]
    lams = np.array(lams)[keep]

    # Shape prior from the concentration ratios, normalized so max = 1.
    Lambda0 = lams / np.max(lams)
    return B, Lambda0


def sine_gaussian_lattice(N, fs, t_start, t_end, f_low, f_high, q=8):
    """Sine-Gaussian (Morlet-Gabor) atom lattice tiling a TF support (secondary basis).

    Places Morlet-Gabor atoms

        psi(t) = exp(-(t - t0)^2 / (2 tau^2)) * cos/sin(2 pi f0 (t - t0)),
        tau = q / (2 pi f0),

    on a lattice covering [t_start, t_end] x [f_low, f_high] at the standard critical
    spacing for quality factor q (time step ~ tau, frequency step ~ f0 / q).  cos and
    sin quadratures are both included.  The lattice is intentionally allowed to be
    redundant / non-orthogonal -- the Gaussian coefficient prior acts as ridge
    regularization (Sec. 3, basis mode 2).

    Parameters
    ----------
    N, fs, t_start, t_end, f_low, f_high : as in slepian_basis.
    q : float
        Quality factor of the atoms (controls the time-frequency aspect ratio).

    Returns
    -------
    B : ndarray, (N, r)
        Column-normalized atom basis (columns need not be orthogonal).
    Lambda0 : ndarray, (r,)
        Flat unit shape prior (ones); sigma_g carries the scale.
    """
    i0 = max(int(np.floor(t_start * fs)), 0)
    i1 = min(int(np.ceil(t_end * fs)), N)
    if i1 - i0 < 2:
        raise ValueError("Support [t_start, t_end] spans < 2 samples on the segment.")
    if f_high <= f_low or f_low <= 0:
        raise ValueError("Require 0 < f_low < f_high.")

    tgrid = np.arange(N) / fs

    # Frequency lattice: spacing at df ~ f0/q per step (critical for quality factor q),
    # floored so a very narrow band still gets a few rows.
    freqs = []
    f0 = f_low
    while f0 <= f_high + 1e-12:
        freqs.append(f0)
        f0 = f0 + max(f0 / q, (f_high - f_low) / 32.0)

    cols = []
    for fc in freqs:
        tau = q / (2.0 * np.pi * fc)
        dt_step = max(tau, 1.0 / fs)          # critical time spacing ~ tau
        t0 = t_start
        while t0 <= t_end + 1e-12:
            env = np.exp(-((tgrid - t0) ** 2) / (2.0 * tau ** 2))
            # Restrict to the nominal support window (env is already localized).
            for phase in (0.0, np.pi / 2.0):
                atom = env * np.cos(2.0 * np.pi * fc * (tgrid - t0) + phase)
                atom[:i0] = 0.0
                atom[i1:] = 0.0
                nrm = np.linalg.norm(atom)
                if nrm > 0:
                    cols.append(atom / nrm)
            t0 = t0 + dt_step

    if not cols:
        raise ValueError("Lattice produced no atoms; check support / band / q.")
    B = np.array(cols).T                     # (N, r), columns normalized
    Lambda0 = np.ones(B.shape[1])
    return B, Lambda0
