"""
td_inner_product.py:  Time-domain (Toeplitz) inner-product adapter for RIFT.

This module is the T1 deliverable of TD_LIKELIHOOD.md: a drop-in,
call-compatible replacement for RIFT's frequency-domain-diagonal inner products
(``lsu.ComplexIP`` and ``lsu.ComplexOverlap``) that evaluates the same PSD-weighted
matched-filter algebra as a *time-domain quadratic form* against a Toeplitz operator.
It is the seam behind which the rank-structured covariance augmentation
(``td_rank_augmentation.py``: glitches, lines, waveform uncertainty) and the truncated /
gapped segment likelihoods (plan Stage 3) plug in without touching the Precompute call
sites at ``factored_likelihood.py`` FL:920 (ComplexOverlap -> Q_lm(t)) and FL:1005
(ComplexIP -> scalar U/V).

What "the same algebra" means, exactly (derived and pinned by tests here)
------------------------------------------------------------------------
RIFT's ``lsu.ComplexIP.ip`` computes, for two two-sided COMPLEX16FrequencySeries,

    ip_FD(h1, h2) = 2 deltaF * sum_k conj(h1~_k) h2~_k * weights2side_k          (I)

where ``weights2side`` is the band-limited two-sided inverse PSD (1/S, zeroed outside
[round(fLow/deltaF), round(fMax/deltaF)) exactly as ``lsu.InnerProduct`` builds it).
Transforming h1~, h2~ to the time domain with lal's conventions
(``h_td = lsu.DataInverseFourier(h_f)``, a reverse FFT that multiplies by deltaF), the
*same number* is the time-domain bilinear quadratic form

    ip_TD(h1, h2) = 2 deltaT * conj(h1_td)^T K h2_td                             (II)

against the symmetric-Toeplitz PRECISION operator K whose frequency-domain symbol is the
band-limited weight ``weights`` (i.e. K is the "inverse-ACF" Toeplitz built from
kappa[tau] = iFFT(weights2side)).  The load-bearing normalization constant between (I)
and (II) is

    ip_FD = (2 deltaT) * ip_TD_unnormalized                                      (III)

i.e. **2 deltaT = 2 / fs** (verified bit-exact in test (i)).  Equivalently, in the
real one-sided packing this is the familiar (4 / (N * fs)) matched-filter family factor.

Band-limiting convention (documented per plan T1)
-------------------------------------------------
The weight is 1/S, floored to exactly 0 on bins outside [round(fLow/deltaF),
round(fMax/deltaF)) BEFORE forming the operator -- identical to ``lsu.InnerProduct``.
Consequence: K is a *precision* (inverse-covariance) operator, NOT the inverse of an
invertible covariance.  A band-limited weight corresponds to a covariance with INFINITE
variance out of band (a singular precision), so one cannot recover this inner
product by building an invertible ACF covariance and calling C^{-1}; the operator must be
built directly from the weight.  This is why ``ToeplitzInnerProductContext`` forms K from
``acf_from_psd`` applied to the *weight symbol*, not from the physical PSD.  (The physical
ACF covariance, needed by the augmentation layer's glitch marginalization, is a separate
object built from S itself in ``td_rank_augmentation``.)

Why the rhoTS construction is done in the frequency domain
----------------------------------------------------------
``ToeplitzComplexOverlap.ip`` must return an rhoTS that is a bit-for-bit drop-in for
``lsu.ComplexOverlap.ip`` (same epoch, same argmax bin, same overall scale) at the
FL:920-929 call site.  Because K's FD symbol is exactly ``weights2side``, the impulse
response rhoTS = iFFT(conj(h~) * FFT(K data)) is identical to iFFT(conj(h~) * weights *
data~), which is precisely ``lsu.ComplexOverlap``'s integrand.  We therefore form that
integrand and reverse-FFT it with the *same lal plan* lsu uses -- guaranteeing exact
agreement of the deltaF factor, FFT direction, packing, and epoch handling rather than
re-deriving them.  (Test (ii) confirms 0.0 relative difference and exact argmax.)

Solvers and boundaries
----------------------
Full-segment precision applies use dense matrices for small arrays and exact FFT
Toeplitz matvecs for larger arrays. They do not require tdanalysis. Only explicit
physical-covariance solves use the optional public PCGQuadraticForm backend from
tdanalysis 1.x (Python >=3.11). Install with ``pip install 'RIFT[td-pcg]'``.
The PCG adapter currently constructs a dense covariance during preparation, so its
initial memory requirement remains O(N^2). No tdanalysis implementation is vendored.

Shared-operator FD/TD equality does not establish equivalence between a cropped
precision and the inverse of a marginal finite-segment covariance. See
TD_LIKELIHOOD.md for the supported boundary conventions and experimental limits.
"""

__author__ = "R. O'Shaughnessy and the RIFT TD team"

import os

import numpy as np
import scipy.linalg

import lal

from RIFT.likelihood import td_rank_augmentation as tra

# The dense Toeplitz operator is only appropriate for short segments; larger N routes to
# the optional iterative (tdanalysis PCG) path.  Mirrors td_rank_augmentation._MAX_DENSE_N.
_MAX_DENSE_N = 1 << 14

# Relative eigenvalue cutoff for the pseudo-inverse of the gap block K_GG.  Exactly-null
# modes contribute nothing, but with fMax < fNyq K_GG also has near-null modes whose
# contribution is O(1), so the gap marginal depends on this cutoff there.  ILE --td-gap is
# refused until that regularization is defined.
_GAP_RCOND = 1e-12

# Frequency tolerance for deltaF comparisons, matching lalsimutils.TOL_DF spirit.
_TOL_DF = 1e-6


class ToeplitzInnerProductContext(object):
    r"""Shared context for the TD Toeplitz inner products (built once per detector).

    Holds the band-limited inverse-PSD weight (identical to ``lsu.InnerProduct``), the
    symmetric-Toeplitz PRECISION operator K it implies, and (for N <= 2^14) its dense
    form for the pure-TD quadratic form.  ``ToeplitzComplexIP`` and
    ``ToeplitzComplexOverlap`` are thin consumers of this object.

    Parameters
    ----------
    psd : callable or ndarray or lal.REAL8FrequencySeries
        The one-sided PSD S(f).  A callable S(f) [Hz^-1] (analytic mode), OR a numpy array
        sampled on the one-sided rfft grid at spacing ``deltaF`` (what ILE holds after
        ``resample_psd_series``), OR a lal.REAL8FrequencySeries.
    deltaF : float
        Frequency resolution [Hz] of the (two-sided) series this context operates on.
    fNyq : float
        Nyquist frequency [Hz]; the two-sided series length is len2side = 2*(fNyq/deltaF).
    fLow : float
        Low-frequency band edge [Hz] (weights zero below round(fLow/deltaF)).
    fMax : float or None
        High-frequency band edge [Hz] (weights zero at/above round(fMax/deltaF)).  If None,
        defaults to fNyq (exactly as lsu.InnerProduct).
    analyticPSD_Q : bool
        True if ``psd`` is a callable to be evaluated at bin frequencies; False if it is an
        array / REAL8FrequencySeries to be indexed.  Mirrors lsu.InnerProduct.
    N : int or None
        Two-sided segment length.  If None it is derived as len2side = 2*(round(fNyq/deltaF)).
    """

    def __init__(self, psd, deltaF, fNyq, fLow, fMax=None,
                 analyticPSD_Q=True, N=None, solver="auto",
                 segment_duration=None, segment_start_index=0,
                 gaps=None, gap_time_origin=None, gap_sample_ranges=None,
                 template_start_offset=None, template_end_offset=None,
                 template_ref_time=None, spd_tol=1e-8, acf=None):
        self.deltaF = float(deltaF)
        self.fNyq = float(fNyq)
        self.fLow = float(fLow)
        self.fMax = float(fMax) if fMax is not None else float(fNyq)
        if self.fMax > self.fNyq + _TOL_DF:
            raise ValueError("fMax ({0}) must be <= fNyq ({1}).".format(self.fMax, self.fNyq))

        self.len1side = int(self.fNyq / self.deltaF) + 1
        self.len2side = 2 * (self.len1side - 1)
        if N is None:
            N = self.len2side
        self.N = int(N)
        self.deltaT = 1.0 / (2.0 * self.fNyq)
        self.fs = 1.0 / self.deltaT

        self.minIdx = int(round(self.fLow / self.deltaF))
        self.maxIdx = int(round(self.fMax / self.deltaF))

        # ---- band-limited one-sided inverse-PSD weight (exactly as lsu.InnerProduct) ----
        self.weights = self._build_weights(psd, analyticPSD_Q)

        # ---- two-sided packed weight (exactly as lsu.InnerProduct.weights2side) ----
        # W(-N/2 df), ..., W(-df), W(0), W(df), ..., W((N/2-1) df)
        self.weights2side = np.zeros(self.len2side)
        self.weights2side[:len(self.weights)] = self.weights[::-1]
        self.weights2side[len(self.weights) - 1:] = self.weights[:-1]

        # ---- precision operator K (symbol = one-sided weight, mirrored to a real,
        #      even two-sided symbol) => "inverse ACF" kappa[tau] = iFFT(symbol) ----
        # kappa is the first column of the symmetric-Toeplitz precision operator.
        # Reuse acf_from_psd's Wiener-Khinchin machinery by feeding it the weight as if it
        # were a PSD: kappa[tau] = irfft(weight_2side) * (nfft * dF) with nfft = len2side.
        self.kappa = self._weight_to_kappa()

        self._solver = solver
        self._template_data_time_origin = None
        self._dense = None
        self._pcg = None
        # Whether the dense N x N precision Toeplitz is the chosen apply path.
        #   * solver="dense": always dense (any N), but BUILT LAZILY (see _ensure_dense) so
        #     __init__ never pays the O(N^2) memory / build cost -- important because at
        #     N = 2^14 the dense matrix is ~2 GB and building one per context OOMs when many
        #     are constructed in parallel.
        #   * solver="auto": dense only for SMALL segments (N < _MAX_DENSE_N); at
        #     N >= _MAX_DENSE_N "auto" routes to the FFT (circulant-embedding) Toeplitz matvec,
        #     which is solver-independent for the full-segment inner products and needs no dense
        #     matrix at all.
        #   * solver="pcg": FFT matvec for the precision apply; the optional tdanalysis PCG is
        #     used only for the physical-covariance solve (lazy, see _ensure_pcg).
        #   * solver="gs": COVARIANCE kernel.  Every inner product uses C^{-1}, C =
        #     Toeplitz(ACF of the physical PSD) cut to the segment, applied with tdanalysis'
        #     ToeplitzSolver on its GS method (see _gs_apply).  The precision kernel K is
        #     not used.
        #     The ACF is ``acf`` if given (lags 0, 1/fs, ...; e.g. from tdanalysis), else
        #     tdanalysis' PSD->ACF on this context's PSD (see gs_acf).
        self._use_gs = (solver == "gs")
        self._gs_forms = {}     # segment length n -> tdanalysis ToeplitzSolver (GS)
        self._acf_input = None if acf is None else np.asarray(acf, dtype=np.float64)
        self._acf_from_psd = None
        self._use_dense = (solver == "dense") or (solver == "auto" and self.N < _MAX_DENSE_N)
        if self._use_dense and self.N < _MAX_DENSE_N:
            # Cheap for small N; build eagerly so callers that read ``ctx._dense`` directly
            # (and the byte-identical numerics the T1/glitch tests pin) see the same operator.
            # The large-N explicit-dense case stays lazy (built on first apply_K).
            self._build_dense()
        # cache of w = K^{-1} d (or K d, precision-operator apply) per data vector id
        self._w_cache = {}

        # ---- Stage-3 (T5) truncated / gapped segment support ----
        # Store the template-frame truncation window (D1: user offsets in seconds relative
        # to the template's peak-amplitude reference; applied to the template streams by the
        # IP wrappers via truncate_timeseries_template_frame).
        self.template_start_offset = (None if template_start_offset is None
                                      else float(template_start_offset))
        self.template_end_offset = (None if template_end_offset is None
                                    else float(template_end_offset))
        self.template_ref_time = (None if template_ref_time is None
                                  else float(template_ref_time))
        self.spd_tol = float(spd_tol)
        self._configure_segment(segment_duration, segment_start_index,
                                gaps, gap_time_origin, gap_sample_ranges)

        # ---- G3 glitch marginalization state (attached lazily; OFF by default) ----
        # When no glitch basis is attached the object is byte-identical to the T1/T5
        # context (all glitch branches are gated on self._glitch_active).
        self._glitch_active = False
        self._glitch_aug = None            # RankAugmentation over the precision operator
        self._glitch_sigma_g = None        # profiled amplitude (per detector)
        self._glitch_delta_lnZ = 0.0       # Occam-disciplined evidence term (Sec. 4)
        self._glitch_data_id = None        # id() of the data series used to build gamma/ghat
        self._glitch_data_corr_fd = None   # cached glitch-subtracted data (FD, for Q')
        self._glitch_ghat_td = None        # MAP glitch reconstruction B M^-1 gamma (TD)
        self._glitch_support = None        # (i0, i1) sample support of the basis (diagnostics)
        # population-prior detection-conditioned prior modes.  prior_mode in {m2,signmix,pinned}:
        #   m2      zero-mean moment-matched Gaussian (Sigma_shape); the operating default,
        #           bit-identical to the zero-mean Gaussian path.
        #   pinned  single mean-shifted branch at +mu_mix (decisive-sign limit).
        #   signmix the exact two-component sign mixture -- BOTH +-mu_mix branches evaluated,
        #           combined logsumexp - ln2 at the marginal level (the ILE two-pass toggles
        #           the active sign via set_active_glitch_sign; each branch carries its own
        #           per-branch Delta lnZ).
        self._glitch_prior_mode = "m2"
        self._glitch_active_sign = 1.0     # signmix: which +-mu component is currently active
        self._glitch_data_td = None        # cached TD data (to re-sign d_corr per branch)
        self._glitch_data_fd_ref = None    # the data COMPLEX16FrequencySeries the basis used
        self._glitch_delta_lnZ_plus = 0.0  # per-branch evidence (signmix bookkeeping)
        self._glitch_delta_lnZ_minus = 0.0
        self._glitch_has_mean = False      # True iff mu_mix is carried (pinned / signmix)
        self._glitch_sign_odds = None      # closed-form ln(L+/L-) at attach (h=0)

    # ------------------------------------------------------------------ segment (T5)

    def _configure_segment(self, segment_duration, segment_start_index,
                           gaps, gap_time_origin, gap_sample_ranges):
        r"""Configure the truncated / gapped analysis segment (plan T5, decision D5).

        The truncated-segment precision operator is ``K_seg = Toeplitz(kappa[:N_seg])``
        with ``kappa`` the FULLY-RESOLVED (len2side) inverse transform of the band-limited
        weight, cut to ``N_seg`` (STUDY.md recommendation: build kappa on the long grid,
        then truncate -- never sample 1/S at the segment's own coarse resolution).  Per the
        ratified D5 resolution this is the DEFAULT truncation convention: ``K_seg`` is a
        PRECISION operator, applied directly as a quadratic-form kernel and NEVER factored
        (it is commonly non-SPD after truncation for sharp lines -- see ``is_spd``).

        Gaps.  The gap samples G are marginalized: the retained-sample precision is
        ``(C_RR)^-1`` with ``C = K_seg^-1``, i.e. the Schur complement
        ``K_RR - K_RG K_GG^+ K_GR``.  Cropping ``K_seg`` to ``K_seg[R][:,R]`` instead gives
        ``(C^-1)_RR``, which mis-weights the data near the gap by several noise sigma when
        the PSD has lines.  ``_apply_seg`` applies the Schur complement as inpainting: fill
        the gap with ``x = -K_GG^+ (K v0)_G`` so that ``K (v0 + E_G x)`` vanishes on G.
        ``dense_retained_operator`` returns the same operator densely.

        Parameters
        ----------
        segment_duration : float or None
            Analysis-segment duration [s]; ``N_seg = round(segment_duration * fs)``.  None
            leaves the full segment (``N_seg = N``); the segment path is then active only if
            gaps or a non-zero start index are supplied.
        segment_start_index : int
            Start sample of the analysis window within the full (len2side) stream.
        gaps : list[(t0, t1)] or None
            Gap intervals in seconds relative to ``gap_time_origin`` (the GPS/geocenter time
            of segment sample 0).  Converted to sample ranges within the segment.
        gap_time_origin : float or None
            GPS/geocenter time [s] of segment sample 0, used to place ``gaps``.  If None, the
            gap times are interpreted as seconds relative to segment sample 0 (origin 0).
        gap_sample_ranges : list[(i0, i1)] or None
            Gap intervals given directly as integer sample ranges within the segment
            (half-open [i0, i1)); a convenience for tests, additive with ``gaps``.
        """
        self.segment_start_index = int(segment_start_index)
        if segment_duration is None:
            self.N_seg = self.N
        else:
            self.N_seg = int(round(float(segment_duration) * self.fs))
        if self.N_seg <= 0 or self.segment_start_index < 0 \
                or self.segment_start_index + self.N_seg > self.len2side:
            raise ValueError(
                "segment window [{0}, {0}+{1}) does not fit inside the full stream of "
                "length {2}".format(self.segment_start_index, self.N_seg, self.len2side))

        # retain mask (True = kept, False = gapped/deleted)
        self.retain_mask = np.ones(self.N_seg, dtype=bool)
        gap_ranges = []
        if gap_sample_ranges:
            gap_ranges.extend((int(i0), int(i1)) for (i0, i1) in gap_sample_ranges)
        if gaps:
            origin = 0.0 if gap_time_origin is None else float(gap_time_origin)
            for (t0, t1) in gaps:
                # seconds (geocenter) -> sample index within the segment
                i0 = int(np.floor((float(t0) - origin) / self.deltaT))
                i1 = int(np.ceil((float(t1) - origin) / self.deltaT))
                gap_ranges.append((i0, i1))
        for (i0, i1) in gap_ranges:
            i0c = max(0, min(self.N_seg, i0))
            i1c = max(0, min(self.N_seg, i1))
            if i1c > i0c:
                self.retain_mask[i0c:i1c] = False
        self._gap_ranges = gap_ranges
        self._gap_idx = None if self.retain_mask.all() else np.nonzero(~self.retain_mask)[0]
        self._gap_Q = None

        # Is a non-trivial (truncated or gapped) segment active?
        self.segment_active = (
            self.N_seg != self.len2side
            or self.segment_start_index != 0
            or (not self.retain_mask.all())
            or self.template_start_offset is not None
            or self.template_end_offset is not None)

        # Build the (dense) truncated precision operator K_seg and run the D5 SPD check.
        # ONLY for a genuinely sub-length segment. With full-length data, template
        # cuts and gaps retain N_seg == len2side: ``segment_dot_td`` applies
        # the FULL precision operator (``_apply_seg`` -> ``apply_K``, the FFT matvec), so
        # materialising the len2side x len2side dense K_seg (~2 GB at 2^14) and running its
        # O(N^3) ``eigvalsh`` SPD check would be wasted work.  Skip it there; ``is_spd`` stays
        # None (an informational diagnostic only -- the full-segment kernel is SPD by
        # construction up to the single dropped Nyquist bin).
        self._Kseg = None
        self.is_spd = None
        self._Kseg_min_eig = None
        # Full-length gaps use the full FFT operator inside _apply_seg; a dense
        # len2side x len2side K_seg and its O(N^3) eigenspectrum are not needed.
        needs_dense_Kseg = (self.N_seg != self.len2side)
        if self._use_gs:
            # Covariance kernel on spliced segments: each contiguous retained run [i0, i1) is
            # one piece with its own Toeplitz C = Toeplitz(acf[:i1-i0]), and pieces are
            # treated as independent (block-diagonal C, as in tdanalysis' segmented
            # likelihood).  Noise correlation across a gap is neglected.
            edges = np.diff(np.concatenate(([0], self.retain_mask.astype(np.int8), [0])))
            self._gs_runs = list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))
            return
        if self.segment_active and needs_dense_Kseg and self.N_seg <= _MAX_DENSE_N:
            self._Kseg = scipy.linalg.toeplitz(self.kappa[:self.N_seg])
            self._spd_check_Kseg()

    def _spd_check_Kseg(self):
        r"""Log (do NOT fail) whether the truncated precision kernel K_seg is SPD (D5).

        The truncated precision kernel is commonly non-SPD (STUDY.md: SPD in only 16/51
        corners for sharp lines) -- this does NOT affect the matched filter / lnL, which
        apply K_seg directly and never factor it.  We record ``is_spd`` and the minimum
        eigenvalue and emit a one-time informational log when non-SPD, so a run leaves a
        breadcrumb.  Any code path that tries to Cholesky-factor K_seg is hard-guarded (see
        ``cholesky``).
        """
        try:
            w = np.linalg.eigvalsh(self._Kseg)
        except np.linalg.LinAlgError:
            self.is_spd = None
            return
        self._Kseg_min_eig = float(w[0])
        scale = float(np.max(np.abs(w))) if w.size else 1.0
        self.is_spd = bool(w[0] > -self.spd_tol * max(scale, 1.0))
        if not self.is_spd:
            import logging
            logging.getLogger(__name__).info(
                "ToeplitzInnerProductContext: truncated precision kernel K_seg (N_seg=%d) is "
                "NOT positive-definite (min eig=%.3e, scale=%.3e).  This is expected under "
                "the D5 precision-kernel convention; K_seg is applied directly as a "
                "quadratic-form kernel and is never factored.", self.N_seg,
                self._Kseg_min_eig, scale)

    def cholesky(self):
        """Anti-factoring guard (plan D5): refuse to Cholesky-factor the precision kernel.

        ``K_seg`` is a PRECISION (inverse-covariance) operator and is commonly non-SPD after
        truncation; factoring it is mathematically wrong (it is not an invertible
        covariance).  Any code path reaching here is a bug -- raise loudly rather than return
        a silently-wrong factor.  If a whitening transform is genuinely needed for a
        diagnostic, use ``|Q diag(|lambda|) Q^T|`` (symmetric eigendecomposition), not this.
        """
        raise RuntimeError(
            "Refusing to Cholesky-factor the truncated precision kernel K_seg: it is a "
            "PRECISION operator (D5), commonly non-SPD after truncation, and must be applied "
            "directly as a quadratic-form kernel -- never factored.  (is_spd=%r, "
            "min_eig=%r)" % (self.is_spd, self._Kseg_min_eig))

    def apply_Kseg(self, v):
        """Return K_seg v for a length-N_seg (real or complex) vector v (dense matvec)."""
        if self._Kseg is None:
            raise NotImplementedError(
                "Truncated-segment operator K_seg is only built densely for N_seg <= {0}; "
                "N_seg={1} would need the iterative path (not wired for the segment kernel)."
                .format(_MAX_DENSE_N, self.N_seg))
        v = np.asarray(v)
        if np.iscomplexobj(v):
            return self._Kseg.dot(v.real) + 1j * self._Kseg.dot(v.imag)
        return self._Kseg.dot(v)

    def _ensure_gap_solve(self):
        """Eigendecompose the gap block K_GG once, for the inpainting solve in _apply_seg."""
        if self._gap_Q is not None:
            return
        G = self._gap_idx
        # K_seg[i, j] = kappa[|i - j|] for both the full (circulant) and truncated segment
        lam, Q = np.linalg.eigh(self.kappa[np.abs(G[:, None] - G[None, :])])
        keep = np.abs(lam) > _GAP_RCOND * np.abs(lam).max()
        self._gap_Q = Q[:, keep]
        self._gap_lam_inv = 1.0 / lam[keep]

    def _apply_seg(self, v):
        """Apply the segment precision operator, with the gap samples marginalized.

        The output is defined on the retained samples only and is zero on the gap; the gap
        entries of ``v`` are ignored.  solver="gs" uses its own block-diagonal operator.
        """
        if self._use_gs or self._gap_idx is None:
            return self._apply_seg_cropped(v)
        self._ensure_gap_solve()
        G = self._gap_idx
        v = np.array(v, copy=True)
        v[G] = 0.0
        Kv = self._apply_seg_cropped(v)
        fill = np.zeros_like(Kv)
        fill[G] = -self._gap_Q.dot(self._gap_lam_inv * self._gap_Q.T.dot(Kv[G]))
        Kv = Kv + self._apply_seg_cropped(fill)
        Kv[G] = 0.0
        return Kv

    def _apply_seg_cropped(self, v):
        """Apply K_seg itself, falling back to the full K when untruncated.

        When ``K_seg`` is built (truncated and/or gapped segment) use it.  For a full-length
        segment (N_seg == len2side, e.g. the T6 full-segment diagnostic) fall back to the
        full precision operator ``apply_K`` (dense or FFT matvec) rather than materialising a
        second len2side dense matrix.  With solver="gs" this is the block-diagonal C^{-1} v
        over the retained pieces (zero on gap samples).
        """
        if self._use_gs:
            v = np.asarray(v)
            out = np.zeros(v.shape, dtype=np.result_type(v, np.float64))
            for (i0, i1) in self._gs_runs:
                out[i0:i1] = self._gs_apply(v[i0:i1], int(i1 - i0))
            return out
        if self._Kseg is not None:
            return self.apply_Kseg(v)
        if self.N_seg == self.len2side:
            return self.apply_K(v)
        raise NotImplementedError(
            "truncated K_seg (N_seg={0}) needs a dense build (N_seg <= {1})".format(
                self.N_seg, _MAX_DENSE_N))

    def _segment_window(self, x_full_td):
        """Extract, and gap-mask, the analysis-segment samples from a full TD stream."""
        seg = np.asarray(x_full_td)[
            self.segment_start_index:self.segment_start_index + self.N_seg].copy()
        if not self.retain_mask.all():
            seg[~self.retain_mask] = 0.0
        return seg

    def segment_dot_td(self, u_full_td, v_full_td):
        r"""UNNORMALIZED conj(u_seg)^T K_seg v_seg on the truncated/gapped segment.

        Segment analogue of ``dot_td``: window both full-length TD streams to the analysis
        segment, zero the gap samples, and apply the truncated precision kernel with the
        gap marginalized (see ``_configure_segment``).  Multiply by ``ip_normalization`` for the
        physical inner product (identical (2 deltaT) factor as the full path).
        """
        u_seg = self._segment_window(u_full_td)
        v_seg = self._segment_window(v_full_td)
        return np.tensordot(np.conj(u_seg), self._apply_seg(v_seg), axes=([-1], [-1]))

    def u_at_shift(self, h_td, n_shift):
        r"""Scalar segment U = <shift_{n} h | shift_{n} h> for an integer-sample shift (T6).

        The template TD stream is rolled by ``n_shift`` samples (positive = later) and the
        truncated/gapped-segment quadratic form is evaluated on the FIXED analysis window --
        so sliding the template partly out of the retained window (a GPS-frame cut) shows up
        as a change in U.  Returns the physically-normalized ``ip_normalization * quadform``.
        """
        hs = np.roll(np.asarray(h_td), int(n_shift))
        return self.ip_normalization * self.segment_dot_td(hs, hs)

    def uv_edge_sensitivity(self, h_td, window_half):
        r"""U edge-sensitivity diagnostic over the time-marginalization window (plan T6).

        Evaluate the dominant-mode U at the two extremes of the marginalization window
        (t_ref +- ``window_half``) and at the centre, and return the max relative variation.
        Under the template-frame truncation convention (D1) U(t) is ~shift-invariant, so this
        is ~0 (full segment) or tail-level (template-frame ringdown cut); a GPS-frame-cut
        misconfiguration (segment truncated across the sliding template support) makes it
        O(1).  Cheap: two extra scalar IPs (plus the centre).

        Returns dict with U0, U_minus, U_plus, rel_var, n_shift.
        """
        n = int(round(float(window_half) * self.fs))
        U0 = self.u_at_shift(h_td, 0)
        Um = self.u_at_shift(h_td, -n)
        Up = self.u_at_shift(h_td, +n)
        denom = abs(U0) if abs(U0) > 0 else 1.0
        rel = max(abs(Up - U0), abs(Um - U0)) / denom
        return {"U0": U0, "U_minus": Um, "U_plus": Up, "rel_var": float(rel), "n_shift": n}

    def dense_retained_operator(self):
        """Dense marginal retained-sample precision K_RR - K_RG K_GG^+ K_GR (gap validation).

        Returns ``(Kret, retained_indices)``.  ``conj(u[R])^T Kret v[R]`` equals
        ``segment_dot_td`` to rounding error.
        """
        if self._Kseg is None:
            raise NotImplementedError("K_seg not built densely for N_seg={0}".format(self.N_seg))
        R = np.nonzero(self.retain_mask)[0]
        Kret = self._Kseg[np.ix_(R, R)]
        if self._gap_idx is not None:
            self._ensure_gap_solve()
            KRG = self._Kseg[np.ix_(R, self._gap_idx)].dot(self._gap_Q)
            Kret = Kret - (KRG * self._gap_lam_inv).dot(KRG.T)
        return Kret, R

    # ------------------------------------------------------------------ builders

    def _build_weights(self, psd, analyticPSD_Q):
        """Band-limited one-sided 1/S weight, byte-compatible with lsu.InnerProduct.

        Also caches ``self.psd_onesided`` (the one-sided S(f) sampled on the rfft grid) for
        the optional physical-covariance iterative path; off-band bins are set to +inf
        (weight 0) so they do not enter the band-limited operator.
        """
        weights = np.zeros(self.len1side)
        S = np.full(self.len1side, np.inf)
        if analyticPSD_Q:
            for i in range(self.minIdx, self.maxIdx):
                sval = psd(i * self.deltaF)
                S[i] = sval
                weights[i] = 1.0 / sval
        else:
            if isinstance(psd, lal.REAL8FrequencySeries):
                data = psd.data.data
            else:
                data = np.asarray(psd)
            for i in range(self.minIdx, self.maxIdx):
                if i < len(data) and data[i] != 0.0:
                    S[i] = data[i]
                    weights[i] = 1.0 / data[i]
        self.psd_onesided = S
        return weights

    def _weight_to_kappa(self):
        r"""First column of the precision Toeplitz operator, kappa[tau], length len2side.

        The precision operator K has frequency-domain symbol equal to the one-sided
        band-limited ``weights`` (= 1/S on band), viewed as the DFT symbol of a real, even
        two-sided sequence.  Its inverse discrete transform IS that sequence:

            kappa = iFFT(symbol)  =  np.fft.irfft(weights, n = len2side)

        (numpy's irfft treats ``weights`` as the non-negative-frequency half of a real,
        Hermitian-symmetric spectrum and divides by nfft, giving exactly Re[iFFT(mirrored
        symbol)]).  K[i,j] = kappa[|i-j|].  The (2 deltaT) physical normalization that maps
        conj(u)^T K v onto lsu.ComplexIP is applied separately (``ip_normalization``); it is
        NOT folded into kappa, so kappa is purely the operator symbol's inverse transform.
        """
        nfft = self.len2side
        kappa = np.fft.irfft(self.weights, n=nfft)
        return kappa.astype(np.float64)

    def _build_dense(self):
        """Dense symmetric-Toeplitz precision operator K (N x N) for short segments."""
        col = self.kappa[:self.N]
        self._dense = scipy.linalg.toeplitz(col)

    def _ensure_dense(self):
        """Return the dense precision operator (building it lazily), or None for the FFT path.

        Returns ``None`` when the dense path is not selected (solver="auto" at
        N >= _MAX_DENSE_N, or solver="pcg"), in which case ``apply_K`` uses the FFT
        (circulant-embedding) Toeplitz matvec.  When dense IS selected (explicit
        solver="dense", or "auto" at small N) the matrix is materialised on first use so a
        large explicit-dense operator is never built eagerly in ``__init__``.
        """
        if not self._use_dense:
            return None
        if self._dense is None:
            self._build_dense()
        return self._dense

    # ------------------------------------------------------------------ operator apply

    def apply_K(self, v):
        """Return K v, the precision operator applied to a (real or complex) vector v.

        For the dense path this is a matrix-vector product; for the iterative path the
        exact Toeplitz matvec via circulant embedding is used.  Bilinear (no conjugation).
        With solver="gs" the operator is instead C^{-1} on the full segment (``_gs_apply``).
        """
        v = np.asarray(v)
        if self._use_gs:
            return self._gs_apply(v, self.N)
        dense = self._ensure_dense()
        if dense is not None:
            if np.iscomplexobj(v):
                return dense.dot(v.real) + 1j * dense.dot(v.imag)
            return dense.dot(v)
        # circulant-embedding exact Toeplitz matvec (matches the dense result), used for
        # large N where forming the dense matrix is too costly.
        return self._toeplitz_matvec(v)

    def _toeplitz_matvec(self, v):
        """Exact symmetric-Toeplitz matvec K v via zero-padded circular convolution.

        Accepts a vector (N,) or a MATRIX (N, r) of stacked column vectors (the
        rank-augmentation layer solves Phi = C^{-1} B column-block-wise; the FFTs
        vectorize over the trailing axis at identical numerics to the per-column loop)."""
        N = self.N
        col = self.kappa[:N]
        # embed the symmetric Toeplitz into a circulant of length >= 2N-1
        m = 2 * N - 1
        c = np.zeros(m)
        c[:N] = col
        c[N:] = col[1:][::-1]        # symmetric first row = first column for symmetric T
        fc = np.fft.fft(c)

        def _mv(x):
            # x: (N,) or (N, r); pad + FFT along axis 0
            shape = (m,) + x.shape[1:]
            xp = np.zeros(shape, dtype=x.dtype)
            xp[:N] = x
            fx = np.fft.fft(xp, axis=0)
            y = np.fft.ifft(fc.reshape((m,) + (1,) * (x.ndim - 1)) * fx, axis=0)[:N]
            return y

        v = np.asarray(v)
        if np.iscomplexobj(v):
            return (_mv(v.real) + 1j * _mv(v.imag))
        return np.real(_mv(v.astype(float)))

    def dot_td(self, u_td, v_td):
        r"""Return the UNNORMALIZED bilinear TD quadratic form conj(u)^T K v.

        NOTE bilinear + Hermitian usage: this applies conj to the FIRST argument (matching
        lsu.ComplexIP's ``conj(h1) h2`` convention).  Callers who want a pure bilinear form
        can pass already-conjugated inputs.  The physical inner product is obtained by
        multiplying by the (2 deltaT) normalization -- see ``ip_normalization``.
        """
        Kv = self.apply_K(v_td)
        return np.tensordot(np.conj(np.asarray(u_td)), Kv, axes=([-1], [-1]))

    @property
    def ip_normalization(self):
        """The scalar (2 deltaT) mapping the TD quadratic form onto lsu.ComplexIP (eq. III).

        With solver="gs" it is 1: the ACF convention (tdanalysis PSDToAutocorr) already makes
        conj(u)^T C^{-1} v the matched-filter inner product, and C^{-1} ~= (2 deltaT) K.
        """
        if self._use_gs:
            return 1.0
        return 2.0 * self.deltaT

    def gs_acf(self, n):
        r"""ACF lags 0..n-1 defining the GS covariance C_n = Toeplitz(acf[:n]).

        The supplied ``acf`` if one was given.  Otherwise tdanalysis' ``PSDToAutocorr`` on
        this context's in-band PSD, walled outside [fLow, fMax): it re-grids onto an FFT
        long enough that the lags kept here do not wrap around, so C_n is a genuine
        Toeplitz covariance rather than the circulant an N-point FFT of S would give.
        """
        if self._acf_input is not None:
            if self._acf_input.size < n:
                raise ValueError("supplied ACF has {0} lags; segment needs {1}".format(
                    self._acf_input.size, n))
            return self._acf_input[:n]
        if self._acf_from_psd is None:
            from tdanalysis.covariance.psd import PSDToAutocorr, PSDToAutocorrConfig
            fs = int(round(self.fs))
            duration = (self.N - 1) / self.fs
            band = np.isfinite(self.psd_onesided)
            f = np.arange(self.len1side) * self.deltaF
            conv = PSDToAutocorr(PSDToAutocorrConfig(
                sampling_rate=fs, acorr_duration=duration, f_min=self.fLow, f_max=self.fMax,
                patch_band=True, run_diagnostics=False))
            _, self._acf_from_psd = conv._psd_to_autocorr(
                f[band], self.psd_onesided[band], fs, duration)
        return self._acf_from_psd[:n]

    def _gs_apply(self, v, n):
        r"""Return C_n^{-1} v, C_n = Toeplitz(gs_acf(n)), with tdanalysis' ToeplitzSolver.

        The solver always takes its Gohberg-Semencul method here (``dense_max_n=0``): the
        generators come from the ACF by refined Levinson solves
        (tdanalysis.forms.gs.gs_generators_from_acorr) and nothing n x n is formed.  It also
        checks that C_n is positive definite.  ``v`` is real or complex, shape (n,) or
        (n, r).  One solver is kept per length n.
        """
        gs = self._gs_forms.get(n)
        if gs is None:
            try:
                from tdanalysis.forms.toeplitz_solver import ToeplitzSolver
            except Exception as exc:
                raise NotImplementedError(
                    "solver='gs' needs tdanalysis >= 1.1 (tdanalysis.forms.toeplitz_solver): "
                    "{0!r}".format(exc))
            if not hasattr(ToeplitzSolver, "apply_Cinv"):
                # earlier 1.1 development trees carry the same version string with apply()
                raise NotImplementedError(
                    "solver='gs' needs tdanalysis 1.1 (the final release): the tdanalysis "
                    "on this path has no ToeplitzSolver.apply_Cinv")
            gs = ToeplitzSolver(self.gs_acf(n), dense_max_n=0)
            self._gs_forms[n] = gs
        return gs.apply_Cinv(np.asarray(v))

    def apply_template_truncation(self, x_td, epoch):
        """Apply the stored template-frame truncation window to a template TD array (T5).

        No-op if no template window is configured.  Uses ``self.template_ref_time`` as the
        common peak reference if the factory has set one (so every mode shares one t_ref);
        otherwise the per-array argmax is used (correct for a single dominant mode).
        """
        if self.template_start_offset is None and self.template_end_offset is None:
            return x_td
        return truncate_timeseries_template_frame(
            x_td, self.deltaT, epoch,
            self.template_start_offset, self.template_end_offset,
            t_ref=self.template_ref_time)

    def template_in_data_frame(self, x_td, epoch):
        """Place a template at the nominal geocenter event in data coordinates.

        Q carries its time translation through the overlap epoch. Scalar U/V
        must explicitly use the same nominal placement for a nonstationary
        operator. The origin is event_time - data_sample_zero_time.
        """
        if self._template_data_time_origin is None:
            return x_td
        shift = (self._template_data_time_origin + float(epoch)) * self.fs
        nearest = int(round(shift))
        if abs(shift - nearest) < 1e-8:
            return np.roll(x_td, nearest)
        frequencies = np.fft.fftfreq(self.len2side)
        return np.fft.ifft(np.fft.fft(x_td) * np.exp(-2j * np.pi * frequencies * shift))

    # ------------------------------------------------------------------ physical covariance

    def covariance_acf(self, acf_duration=None):
        """Physical noise autocovariance C from the band-limited PSD (for whitening / T3).

        This is the physical covariance object (symbol S, NOT the precision symbol 1/S) that
        the augmentation layer (td_rank_augmentation) and the whitening seam consume.  Built
        via ``tra.acf_from_psd`` on the cached one-sided S; off-band bins (S = inf) are
        floored so C stays well-defined.
        """
        if acf_duration is None:
            acf_duration = self.N / self.fs
        f = np.arange(self.len1side) * self.deltaF
        S = np.array(self.psd_onesided, dtype=np.float64)
        finite = np.isfinite(S) & (S > 0)
        if not finite.any():
            raise ValueError("PSD has no finite in-band samples.")
        # Off-band: use a large-but-finite value so C is invertible (its in-band structure
        # is what matters; off-band variance is arbitrary here).
        S_fill = np.where(finite, S, np.max(S[finite]) * 1e6)
        return tra.acf_from_psd(f, S_fill, self.fs, acf_duration=acf_duration)

    # ------------------------------------------------------------------ glitch marginalization (G3)

    @property
    def glitch_active(self):
        """True iff a rank-augmentation (glitch) basis has been attached to this context."""
        return self._glitch_active

    def attach_glitch_basis(self, blocks, data_fd, sigma_g="profile",
                            sigma_g_min=None, prior_mode="m2"):
        r"""Attach a per-detector rank-augmentation (glitch) basis and profile sigma_g (G3).

        ``prior_mode`` (population-prior): ``m2`` (default, zero-mean moment-matched, the
        block's ``mu0`` should be absent/zero), ``pinned`` (single +mu_mix mean-shift), or
        ``signmix`` (two-component sign mixture; sigma_g profiled on the MIXTURE evidence, both
        +-mu branches recorded, the active branch toggled by ``set_active_glitch_sign``).
        ``sigma_g_min`` (strain) floors the amplitude profile bounds (presence conditioning).


        THE ALGEBRA (K-vs-C^{-1} Woodbury placement -- the crux of G3).
        ---------------------------------------------------------------
        The base TD matched filter uses the PRECISION operator ``P = ip_normalization * K``
        (``K`` = band-limited inverse-PSD Toeplitz kernel).  ``P`` IS the noise precision
        ``C^{-1}`` in matched-filter normalization: every inner product is
        ``<u|v> = conj(u)^T P v``.  So the standard covariance-side Woodbury of the design
        (Phi = C^{-1} B via r solves) becomes here a DIRECT MATVEC, no linear solve:

            Phi   = C^{-1} B = P B            (apply the precision to each basis column)
            G     = B^T Phi = B^T P B
            M     = Lambda^{-1} + G           (Lambda = sigma_g^2 Lambda_0)
            gamma = Phi^T d = B^T P d
            C'^{-1} = P - Phi M^{-1} Phi^T    (Woodbury; NEVER inverts P)

        This is exactly ``RankAugmentation`` fed an object whose ``.solve(x)`` returns
        ``P x`` (the precision apply) rather than a covariance solve -- see
        ``_PrecisionAsCovariance``.  ``M`` inverts ``Lambda^{-1} + B^T P B`` (a well-posed
        r x r system floored by the prior), so the augmentation is well-defined even though
        ``P`` itself is rank-deficient off-band (the always-dropped Nyquist bin + the
        band-limit): the glitch basis is band-limited into [f1,f2], living where ``P`` is
        well-conditioned.  ``C`` = "inverse of P on the band" is never formed.

        THE TWO CORRECTED INGREDIENTS.
          * U'_{lm,l'm'} = conj(h_lm)^T C'^{-1} h_l'm'
                         = U - conj(Phi^T h_lm)^T M^{-1} (Phi^T h_l'm')      (scalar, per pair)
            -- the base Hermitian ip minus a rank-r correction (``glitch_ip_correction``).
          * Q'_lm(t) = correlate(h_lm, w'),  w' = C'^{-1} d = P (d - B M^{-1} gamma)
                     = the base overlap machinery run on the GLITCH-SUBTRACTED data
                       d_corr = d - g_hat,  g_hat = B M^{-1} gamma  (the MAP glitch).
            Because the overlap is linear in the data, Q'(t) = overlap(h, d_corr); we
            precompute d_corr once per detector (``glitch_corrected_data_fd``).

        THE EVIDENCE CONSTANT (does NOT cancel; surfaced for ILE).
            Delta lnZ(sigma_g) = 1/2 gamma^H M^{-1} gamma - 1/2 ln det(I + sigma_g^2 Lambda_0 G)
        A per-detector data constant (independent of intrinsic/extrinsic params); ILE adds
        sum_det Delta lnZ to lnL_marg (Sec. 4).

        CONJUGATION.  ``RankAugmentation`` is BILINEAR; RIFT ips are HERMITIAN.  The Hermitian
        corrected ip is obtained by passing ``conj`` on the first slot -- validated against a
        dense conj-explicit reference in ``test_td_glitch.py`` (the bilinear-vs-Hermitian trap).

        Parameters
        ----------
        blocks : list of (B, Lambda0[, label])
            Rank-augmentation blocks (N x r basis + dimensionless shape prior), as consumed by
            ``td_rank_augmentation.RankAugmentation``.  B must have N = ``self.N`` rows.
        data_fd : lal.COMPLEX16FrequencySeries
            The detector data (same object the overlap step will receive); used to profile
            sigma_g, form gamma / g_hat / Delta lnZ.
        sigma_g : "profile" or float
            "profile" (default, the only wired mode) maximizes Delta lnZ over sigma_g on d
            alone (Sec. 4, empirical Bayes).  A float pins sigma_g.  Grid marginalization is
            OUT OF SCOPE for G3 (follow-on: precompute {d_corr, M} per grid value and
            logsumexp inside ILE).

        NOTE (scope): glitch marginalization is implemented for the FULL segment only.  On a
        truncated/gapped segment (``segment_active``) the base operator is the truncated
        precision kernel K_seg and Phi = K_seg B must use ``apply_Kseg`` on the retained
        subspace; that extension is documented but deferred (raises here).
        """
        if self.segment_active:
            raise NotImplementedError(
                "Glitch marginalization is wired for the FULL segment only.  On a truncated/"
                "gapped segment the Woodbury base operator is the truncated precision kernel "
                "K_seg (Phi = K_seg B on the retained subspace); that extension is deferred "
                "(see attach_glitch_basis docstring / REPORT.md).")
        from RIFT.likelihood import td_rank_augmentation as tra
        prec = _PrecisionAsCovariance(self)
        self._glitch_aug = tra.RankAugmentation(prec, blocks)
        # record the basis support (union over blocks) for the overlap diagnostic
        i0s, i1s = [], []
        for blk in self._glitch_aug.blocks:
            nz = np.nonzero(np.any(np.abs(blk["B"]) > 0, axis=1))[0]
            if nz.size:
                i0s.append(int(nz[0])); i1s.append(int(nz[-1]) + 1)
        self._glitch_support = (min(i0s), max(i1s)) if i0s else None

        self._glitch_prior_mode = prior_mode
        self._glitch_has_mean = bool(any(blk["has_mean"] for blk in self._glitch_aug.blocks))
        d_td = _ifft_to_td(data_fd)
        self._glitch_data_td = d_td
        self._glitch_data_fd_ref = data_fd

        # --- amplitude profile / pin (population-prior presence conditioning via sigma_g_min) ---
        if sigma_g == "profile":
            if prior_mode == "signmix":
                # empirical-Bayes on the MIXTURE evidence (a single sigma_g for both branches)
                sig, _ = self._glitch_aug.profile_sigma_g_signmix(d_td, sigma_g_min=sigma_g_min)
            else:
                sig, _ = self._glitch_aug.profile_sigma_g(d_td, sigma_g_min=sigma_g_min)
        else:
            sig = float(sigma_g)
        self._glitch_sigma_g = sig

        # --- per-branch evidence (population-prior bookkeeping) ---
        # MAP glitch reconstruction g_hat_total(sign) = B(sign*mu_a + M^{-1} gamma_tilde) and
        # the corrected data d_corr = d - g_hat_total feed Q'_lm(t); each branch's Delta lnZ
        # carries its own sign-dependent mean constants (mu0 = 0 => both branches identical =>
        # zero-mean Gaussian bit-identical).  signmix combines the two at the marginal level
        # (logsumexp - ln2), but the ILE adds each branch's Delta lnZ to that branch's lnL_marg
        # BEFORE the logsumexp, so we store both and expose the ACTIVE one.
        if self._glitch_has_mean:
            self._glitch_delta_lnZ_plus = float(self._glitch_aug.delta_lnZ(d_td, sig, sign=+1.0))
            self._glitch_delta_lnZ_minus = float(self._glitch_aug.delta_lnZ(d_td, sig, sign=-1.0))
        else:
            _dz = float(self._glitch_aug.delta_lnZ(d_td, sig))
            self._glitch_delta_lnZ_plus = _dz
            self._glitch_delta_lnZ_minus = _dz

        self._glitch_active = True
        self.set_active_glitch_sign(+1.0)   # default branch (+mu; m2: sign irrelevant)

        # closed-form sign log-odds ln(L+/L-) = 2 <d | B mu_a>_C' (data-only regime indicator,
        # h=0; population-prior).  Available whenever a mean direction is carried.
        sodds = float(self._glitch_aug.sign_odds(d_td, sig, h=None)) if self._glitch_has_mean else None
        self._glitch_sign_odds = sodds

        return dict(sigma_g=sig, delta_lnZ=self._glitch_delta_lnZ,
                    delta_lnZ_plus=self._glitch_delta_lnZ_plus,
                    delta_lnZ_minus=self._glitch_delta_lnZ_minus,
                    prior_mode=prior_mode, sign_odds=sodds,
                    support=self._glitch_support)

    def set_active_glitch_sign(self, sign):
        r"""Select the active +-mu sign-mixture branch (population-prior, signmix two-pass).

        Recomputes the glitch-subtracted data ``d_corr = d - g_hat(sign)`` (so Q'_lm(t) picks
        up the branch's mean-predicted glitch) and sets the active per-branch Delta lnZ.  A
        no-op difference for a zero-mean (m2) block.  The ILE runs this for sign=+1 then
        sign=-1 in signmix mode and combines the two lnL_marg via logsumexp - ln2."""
        if not self._glitch_active:
            return
        sign = float(sign)
        self._glitch_active_sign = sign
        d_td = self._glitch_data_td
        ghat = self._glitch_aug.glitch_reconstruction(
            d_td, self._glitch_sigma_g, sign=sign).astype(complex)
        self._glitch_ghat_td = ghat
        self._glitch_data_corr_fd = self._fd_from_td(d_td - ghat, self._glitch_data_fd_ref)
        self._glitch_data_id = id(self._glitch_data_fd_ref)
        self._glitch_delta_lnZ = (self._glitch_delta_lnZ_plus if sign > 0
                                  else self._glitch_delta_lnZ_minus)

    def glitch_sign_odds(self, h_td=None):
        r"""Closed-form sign log-odds ln(L+/L-) = 2 <d - h | B mu_a>_C' (population-prior).

        ``h_td`` is the reference template (TD packing; default 0 = the data-only indicator).
        Returns None if no mean direction is carried."""
        if not self._glitch_active or not self._glitch_has_mean:
            return None
        return float(self._glitch_aug.sign_odds(
            self._glitch_data_td, self._glitch_sigma_g, h=h_td))

    def _fd_from_td(self, td_array, template_fd):
        """Forward-FFT a TD numpy array into a COMPLEX16FrequencySeries matching template_fd.

        Uses ``lalsimutils.DataFourier`` (the exact inverse of the ``DataInverseFourier`` used
        by ``_ifft_to_td``), so ``_fd_from_td(_ifft_to_td(x), x)`` round-trips to x's packing
        (verified in test_td_glitch).
        """
        from RIFT import lalsimutils as lsu
        ts = lal.CreateCOMPLEX16TimeSeries(
            "gc", template_fd.epoch, 0., self.deltaT, _dimensionless_unit(), self.len2side)
        ts.data.data = np.asarray(td_array, dtype=complex)
        return lsu.DataFourier(ts)

    def glitch_ip_correction(self, u_td, v_td):
        r"""Hermitian rank-r correction subtracted from U/V to form U'/V' (G3).

            correction = sum_k conj(Phi_k^T u)^T M_k^{-1} (Phi_k^T v)

        so that ``U'(u,v) = ip_normalization*dot_td(u,v) - glitch_ip_correction(u,v)`` equals
        ``conj(u)^T C'^{-1} v``.  ``Phi_k = P B_k`` already carries the ``ip_normalization``
        (2 deltaT) factor (it is ``cov.solve`` on the precision-as-covariance adapter), so the
        correction is in the same physical units as the base ip -- no extra normalization.
        The explicit ``conj`` on ``Phi^T u`` is the Hermitian-vs-bilinear reconciliation.
        """
        if not self._glitch_active:
            return 0.0
        aug = self._glitch_aug
        sig = self._glitch_sigma_g
        u_td = np.asarray(u_td); v_td = np.asarray(v_td)
        total = 0.0 + 0.0j
        for k, blk in enumerate(aug.blocks):
            Phi = blk["Phi"]
            pu = Phi.T.dot(u_td)                       # Phi^T u  (bilinear r-vector)
            pv = Phi.T.dot(v_td)                       # Phi^T v
            Minv_pv = aug._Minv_apply(pv, sig, k)
            total += np.tensordot(np.conj(pu), Minv_pv, axes=([-1], [-1]))
        return total

    def glitch_corrected_data_fd(self, data_fd):
        """Return the cached glitch-subtracted data FD series (for Q'); see attach_glitch_basis.

        Asserts the passed data is the same object the basis was attached with (the profiled
        sigma_g, gamma and g_hat are specific to that data).  If a different data object is
        passed the correction is recomputed on it (keeps the method correct if the caller
        rebuilds the series), which is rare on the ILE path.
        """
        if not self._glitch_active:
            return data_fd
        if id(data_fd) == self._glitch_data_id and self._glitch_data_corr_fd is not None:
            return self._glitch_data_corr_fd
        # Different data object: recompute (data-conditional gamma / g_hat, mean-inclusive).
        d_td = _ifft_to_td(data_fd)
        ghat = self._glitch_aug.glitch_reconstruction(d_td, self._glitch_sigma_g).astype(complex)
        return self._fd_from_td(d_td - ghat, data_fd)

    @property
    def glitch_delta_lnZ(self):
        """Per-detector Occam-disciplined glitch evidence term (0 if no basis attached)."""
        return self._glitch_delta_lnZ

    def glitch_reconstruction_td(self):
        """MAP glitch reconstruction g_hat = B M^-1 gamma (TD, diagnostic); None if inactive."""
        return self._glitch_ghat_td

    # ------------------------------------------------------------------ optional iterative

    def _ensure_pcg(self, covariance_acf=None):
        """Lazily build the optional tdanalysis PCG solver for the physical COVARIANCE.

        The iterative path solves against the physical ACF covariance C (not the precision
        operator K), for the large-N whitening / augmentation seam.  If ``covariance_acf``
        is None it is built from the cached PSD via ``covariance_acf()``.  Raises
        NotImplementedError with an actionable message if tdanalysis is unavailable.
        """
        if self._pcg is not None:
            return self._pcg
        try:
            from tdanalysis.forms.pcg import PCGQuadraticForm  # optional public dependency
        except Exception as exc:
            raise NotImplementedError(
                "Iterative Toeplitz solve (N = {0}) requires the optional "
                "dependency 'tdanalysis' (tdanalysis.forms.pcg.PCGQuadraticForm), which is "
                "not importable in this environment ({2!r}).  Install RIFT[td-pcg] "
                "with Python >=3.11, or use a segment with N <= {1} for the dense solver."
                .format(self.N, _MAX_DENSE_N, exc))
        from types import SimpleNamespace
        if covariance_acf is None:
            covariance_acf = self.covariance_acf()
        n = self.N
        Cij = scipy.linalg.toeplitz(np.asarray(covariance_acf)[:n]).astype(np.float64)
        S = np.array(self.psd_onesided, dtype=np.float64)
        S = np.where(np.isfinite(S) & (S > 0), S, np.max(S[np.isfinite(S)]) * 1e6)
        # duck-typed parent per tdanalysis' own tests (test_whitening_backends.py):
        parent = SimpleNamespace(
            Cij=Cij,
            n_samples_l=n,
            data_dtype=np.float64,
            sampling_rate=self.fs,
            cfft_use_fast_len=True,
            cg_use_pc=True,
            cg_warm_start=True,
            cg_eps=1e-10,
            cg_maxiter=5000,
            psd_frequencies=np.arange(self.len1side) * self.deltaF,
            psd=S,
        )
        self._pcg = PCGQuadraticForm(parent=parent, preconditioner_kind="psd",
                                     tol=1e-10, maxiter=5000)
        self._pcg.prepare()
        return self._pcg

    def solve_covariance(self, d, method="auto"):
        """Return C^{-1} d for the physical covariance C (dense or tdanalysis PCG).

        method="dense" uses a Cholesky factor (N <= 2^14); method="pcg" uses the optional
        tdanalysis iterative solver; "auto" picks dense for small N, PCG otherwise.
        """
        d = np.asarray(d, dtype=np.float64)
        if method == "auto":
            method = "dense" if self.N <= _MAX_DENSE_N else "pcg"
        if method == "dense":
            acf = self.covariance_acf()
            cov = tra.ToeplitzCovariance(acf, self.N)
            return cov.solve(d)
        elif method == "pcg":
            pcg = self._ensure_pcg()
            return np.array(pcg.apply_Cinv(d), copy=True)
        raise ValueError("method must be 'auto', 'dense', or 'pcg'.")

    # ------------------------------------------------------------------ checks

    def _check_series(self, hf):
        assert hf.data.length == self.len2side, (
            "series length {0} != context len2side {1}".format(hf.data.length, self.len2side))
        assert abs(hf.deltaF - self.deltaF) <= _TOL_DF, (
            "series deltaF {0} != context deltaF {1}".format(hf.deltaF, self.deltaF))


class _PrecisionAsCovariance(object):
    r"""Adapter presenting a ToeplitzInnerProductContext's PRECISION as a covariance-solve.

    ``RankAugmentation`` (td_rank_augmentation) is written against a covariance object whose
    ``.solve(x)`` returns ``C^{-1} x``.  In the RIFT TD path the base operator we hold IS the
    precision ``P = ip_normalization * K`` (``K`` = band-limited inverse-PSD kernel), i.e.
    ``C^{-1} = P``.  So this adapter's ``.solve(x)`` is a DIRECT precision matvec ``P x`` (no
    linear solve), which makes ``RankAugmentation`` compute Phi = C^{-1} B = P B, G = B^T P B,
    gamma = B^T P d, and the Woodbury correction exactly as the K-side algebra of
    ``attach_glitch_basis`` requires -- reusing the validated G1 code unchanged.

    ``.dot(u, v)`` is the BILINEAR form ``u^T P v`` (no conjugation; ``RankAugmentation`` is
    bilinear -- Hermitian forms pass conj explicitly).  ``.C`` exposes only ``C[0,0]`` (the
    physical noise variance from the ACF) for the default sigma_g bounds; it is a 1x1 array,
    never the full covariance (which is not formed).
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.N = ctx.N
        # C[0,0] = physical noise variance (ACF at lag 0) -> default sigma_g scale (Sec. 2b).
        try:
            c00 = float(ctx.covariance_acf(acf_duration=1.0 / ctx.fs)[0])
        except Exception:
            c00 = 1.0
        self.C = np.full((1, 1), c00, dtype=float)

    def solve(self, b):
        """Return C^{-1} b = P b (precision matvec; no solve).  Real or complex, matrix ok."""
        return self.ctx.ip_normalization * self.ctx.apply_K(b)

    def dot(self, u, v):
        """Bilinear u^T C^{-1} v = u^T P v (NO conjugation; Sec. 2b)."""
        Pv = self.solve(v)
        return np.tensordot(np.asarray(u), Pv, axes=([-1], [-1]))

    def logdet(self):
        # Not used by the glitch path (Delta lnZ uses only the logdet CORRECTION); the
        # precision determinant is off-band-singular and deliberately never formed.
        raise NotImplementedError("precision logdet is off-band-singular and not formed")


class ToeplitzComplexIP(object):
    r"""Scalar TD-Toeplitz analogue of ``lsu.ComplexIP`` (no time/phase maximization).

    ``ip(h1, h2)`` accepts the same two-sided ``lal.COMPLEX16FrequencySeries`` objects
    ``lsu.ComplexIP.ip`` accepts (length ``len2side``, spacing ``deltaF``), transforms them
    to the time domain internally, and returns the bilinear PSD-weighted inner product
    ``2 deltaF sum conj(h1~) h2~ / S`` -- computed as ``2 deltaT conj(h1_td)^T K h2_td``.
    The two agree to machine precision on smooth PSDs / long segments (test (i)).

    Epoch handling mirrors ``lsu.ComplexIP.ip`` exactly: NO epoch phase is applied by
    default; passing ``include_epoch_differences=True`` applies the same
    exp(-i (epoch1 - epoch2) f 2 pi) factor lsu applies (kept in the FD integrand so it is
    byte-identical).
    """

    def __init__(self, context):
        self.ctx = context

    def ip(self, h1, h2, include_epoch_differences=False):
        ctx = self.ctx
        ctx._check_series(h1)
        ctx._check_series(h2)
        assert abs(h1.deltaF - h2.deltaF) <= _TOL_DF

        if include_epoch_differences:
            if ctx.segment_active or ctx.glitch_active:
                raise ValueError("include_epoch_differences is unsupported for TD cuts, "
                                 "gaps or glitch projection; the FD phase shortcut "
                                 "cannot apply the nonstationary operator")
            if ctx._use_gs:
                raise NotImplementedError(
                    "include_epoch_differences uses the FD weights; not available with solver='gs'.")
            # Mirror lsu.ComplexIP: multiply h2's spectrum by the epoch phase in FD, then
            # transform.  Uses lalsimutils.evaluate_fvals for the (reversed) lal f grid.
            from RIFT import lalsimutils as lsu
            fvals = lsu.evaluate_fvals(h1)
            phase = np.exp(-1j * (float(h1.epoch) - float(h2.epoch)) * fvals * 2 * np.pi)
            h1_data = np.asarray(h1.data.data)
            h2_data = np.asarray(h2.data.data) * phase
            # Do the weighted sum directly in FD (bit-identical to lsu in this branch).
            val = np.sum(np.conj(h1_data) * h2_data * ctx.weights2side)
            return val * 2.0 * ctx.deltaF

        # Default (no epoch phase): TD quadratic form.
        h1_td = _ifft_to_td(h1)
        h2_td = _ifft_to_td(h2)
        if ctx.segment_active:
            # Stage-3 (T5): both arguments are templates for U/V -> apply the template-frame
            # truncation to each, then the truncated/gapped precision quadratic form.
            h1_td = ctx.apply_template_truncation(h1_td, float(h1.epoch))
            h2_td = ctx.apply_template_truncation(h2_td, float(h2.epoch))
            h1_td = ctx.template_in_data_frame(h1_td, h1.epoch)
            h2_td = ctx.template_in_data_frame(h2_td, h2.epoch)
            return ctx.ip_normalization * ctx.segment_dot_td(h1_td, h2_td)
        if ctx.glitch_active:
            h1_td = ctx.template_in_data_frame(h1_td, h1.epoch)
            h2_td = ctx.template_in_data_frame(h2_td, h2.epoch)
        val = ctx.ip_normalization * ctx.dot_td(h1_td, h2_td)
        if ctx.glitch_active:
            # G3: U'/V' = U/V - conj(Phi^T h1)^T M^{-1} (Phi^T h2)  (Hermitian correction).
            val = val - ctx.glitch_ip_correction(h1_td, h2_td)
        return val

    def norm(self, h):
        """sqrt(|ip(h, h)|), matching lsu.ComplexIP.norm (2 deltaF sum |h~|^2 / S)."""
        val = self.ip(h, h)
        return np.sqrt(np.abs(val))


class ToeplitzComplexOverlap(object):
    r"""TD-Toeplitz analogue of ``lsu.ComplexOverlap`` (time/phase-maximized matched filter).

    ``ip(hlm, data)`` returns ``(rho, rhoTS, rhoIdx, rhoPhase)`` exactly like
    ``lsu.ComplexOverlap.ip`` with ``full_output=True``:

      * rhoTS : a ``lal.COMPLEX16TimeSeries`` of the FULL two-sided segment length at
                deltaT, the impulse response rhoTS(t) = iFFT(conj(hlm~) * FFT(K data))
                == iFFT(conj(hlm~) * weights * data~), built with the SAME lal reverse-FFT
                plan (the 2 deltaF factor, FFT direction, epoch conventions) as
                ``lsu.ComplexOverlap``.  Bit-identical drop-in for FL:920-929.
      * rho     = max |rhoTS|
      * rhoIdx  = argmax |rhoTS|
      * rhoPhase = angle(rhoTS[rhoIdx])
    """

    def __init__(self, context):
        self.ctx = context
        self._revplan = lal.CreateReverseCOMPLEX16FFTPlan(self.ctx.len2side, 0)
        self._fwdplan = None    # built lazily for the segment path only
        self._intgd = lal.CreateCOMPLEX16FrequencySeries(
            "SNR integrand", lal.LIGOTimeGPS(0.), 0., self.ctx.deltaF,
            _hertz_unit(), self.ctx.len2side)
        self._ovlp = lal.CreateCOMPLEX16TimeSeries(
            "Complex overlap", lal.LIGOTimeGPS(0.), 0., self.ctx.deltaT,
            _dimensionless_unit(), self.ctx.len2side)

    def ip(self, hlm, data, **kwargs):
        ctx = self.ctx
        ctx._check_series(hlm)
        ctx._check_series(data)
        assert abs(hlm.deltaF - data.deltaF) <= _TOL_DF

        if ctx.segment_active:
            return self._ip_segment(hlm, data)

        # G3: Q'_lm(t) = correlate(h_lm, w'), w' = C'^{-1} d = P (d - g_hat).  Because the
        # overlap is linear in the data, this is the base machinery run on the glitch-
        # subtracted data d_corr = d - g_hat (g_hat = B M^{-1} gamma).  Default (no glitch)
        # returns `data` unchanged -> byte-identical.
        data_use = ctx.glitch_corrected_data_fd(data) if ctx.glitch_active else data

        if ctx._use_gs:
            # Covariance kernel: no FD weight exists, so correlate h with w = C^{-1} d in TD.
            return self._ip_segment(hlm, data_use)

        # Integrand exactly as lsu.ComplexOverlap: 2 conj(hlm~) data~ weights2side.
        # Because K's FD symbol == weights2side, this is 2 conj(hlm~) (K data)~, i.e. the
        # impulse-response construction rhoTS = iFFT(conj(hlm~) FFT(K data)) up to the
        # identical lal reverse-FFT scaling.
        self._intgd.data.data = (2.0 * np.conj(np.asarray(hlm.data.data))
                                 * np.asarray(data_use.data.data) * ctx.weights2side)
        lal.COMPLEX16FreqTimeFFT(self._ovlp, self._intgd, self._revplan)

        rhoSeries = np.abs(self._ovlp.data.data)
        rho = rhoSeries.max()
        rhoIdx = int(rhoSeries.argmax())
        rhoPhase = float(np.angle(self._ovlp.data.data[rhoIdx]))
        # Copy to a fresh TimeSeries (do not hand back the reused internal buffer),
        # matching lsu.ComplexOverlap.
        rhoTS = lal.CreateCOMPLEX16TimeSeries(
            "Complex overlap", lal.LIGOTimeGPS(0.), 0., ctx.deltaT,
            _dimensionless_unit(), ctx.len2side)
        rhoTS.data.data[:] = self._ovlp.data.data[:]
        return rho, rhoTS, rhoIdx, rhoPhase

    def _ip_segment(self, hlm, data):
        r"""Matched-filter Q(t) on the truncated/gapped segment (plan T5, D1).

        Q(t) = 2 deltaT * conj(shift_t h_trunc)^T K_seg (mask . d_seg), for all time shifts
        at once.  Construction: the data is windowed to the analysis segment and gap-masked,
        the truncated precision kernel is applied ONCE (``Kd_seg = K_seg d_seg``), and the
        result is embedded back into a full-length TD stream; then the impulse response
        rhoTS = reverseFFT( conj(h~) * forwardFFT(Kd_full) ) reproduces the same lal scaling
        / epoch contract as the full path (the (2 deltaT) factor lives in the forward FFT
        that produced ``data~`` originally, so it is applied here via the forward FFT of
        ``Kd_full``).  h_trunc is the template after the template-frame cut; it has compact
        support that (under the D1 convention + T6 diagnostic) stays inside the analysis
        window across the time-marginalization shifts, so windowing h at each shift is a
        no-op and this FFT correlation is exact.
        """
        ctx = self.ctx
        if self._fwdplan is None:
            self._fwdplan = lal.CreateForwardCOMPLEX16FFTPlan(ctx.len2side, 0)

        hlm_td = ctx.apply_template_truncation(_ifft_to_td(hlm), float(hlm.epoch))

        # w = K_seg d (or C^{-1} d) depends only on the data: computed once per data series
        # and reused for every mode and intrinsic point.  The series is held in the cache
        # so its id() cannot be recycled.
        cached = ctx._w_cache.get(id(data))
        if cached is not None and cached[0] is data:
            kd_f = cached[1]
        else:
            data_seg = ctx._segment_window(_ifft_to_td(data))   # windowed + gap-masked
            Kd_seg = ctx._apply_seg(data_seg)                   # length N_seg
            # Zero on the gap: the shifted template need not vanish there.
            if not ctx.retain_mask.all():
                Kd_seg[~ctx.retain_mask] = 0.0
            if ctx._use_gs:
                # The FFT correlation below supplies (2 deltaT); the C^{-1} form needs none.
                Kd_seg = Kd_seg / (2.0 * ctx.deltaT)
            Kd_full = np.zeros(ctx.len2side, dtype=complex)
            Kd_full[ctx.segment_start_index:ctx.segment_start_index + ctx.N_seg] = Kd_seg

            # forward-FFT Kd_full into lal FD packing (multiplies by deltaT, as data~ was)
            kd_ts = lal.CreateCOMPLEX16TimeSeries(
                "Kd", data.epoch, 0., ctx.deltaT, _dimensionless_unit(), ctx.len2side)
            kd_ts.data.data = Kd_full
            kd_f = lal.CreateCOMPLEX16FrequencySeries(
                "Kd_f", data.epoch, 0., ctx.deltaF, _hertz_unit(), ctx.len2side)
            lal.COMPLEX16TimeFreqFFT(kd_f, kd_ts, self._fwdplan)
            if not ctx.glitch_active:
                # glitch-corrected data is rebuilt per sign branch; do not hold every copy
                ctx._w_cache[id(data)] = (data, kd_f)

        # hlm_td back to FD (same forward plan) to form conj(h~) * kd_f
        h_ts = lal.CreateCOMPLEX16TimeSeries(
            "h", hlm.epoch, 0., ctx.deltaT, _dimensionless_unit(), ctx.len2side)
        h_ts.data.data = hlm_td
        h_f = lal.CreateCOMPLEX16FrequencySeries(
            "h_f", hlm.epoch, 0., ctx.deltaF, _hertz_unit(), ctx.len2side)
        lal.COMPLEX16TimeFreqFFT(h_f, h_ts, self._fwdplan)

        self._intgd.data.data = 2.0 * np.conj(np.asarray(h_f.data.data)) \
            * np.asarray(kd_f.data.data)
        lal.COMPLEX16FreqTimeFFT(self._ovlp, self._intgd, self._revplan)

        rhoSeries = np.abs(self._ovlp.data.data)
        rho = rhoSeries.max()
        rhoIdx = int(rhoSeries.argmax())
        rhoPhase = float(np.angle(self._ovlp.data.data[rhoIdx]))
        rhoTS = lal.CreateCOMPLEX16TimeSeries(
            "Complex overlap", lal.LIGOTimeGPS(0.), 0., ctx.deltaT,
            _dimensionless_unit(), ctx.len2side)
        rhoTS.data.data[:] = self._ovlp.data.data[:]
        return rho, rhoTS, rhoIdx, rhoPhase

    def norm(self, h):
        """sqrt(|2 deltaF sum |h~|^2 / S|), matching lsu.ComplexOverlap.norm."""
        if self.ctx._use_gs:
            return ToeplitzComplexIP(self.ctx).norm(h)
        self.ctx._check_series(h)
        val = np.sum(np.conj(np.asarray(h.data.data)) * np.asarray(h.data.data)
                     * self.ctx.weights2side)
        return np.sqrt(2.0 * self.ctx.deltaF * np.abs(val))


class TDInnerProductFactory(object):
    r"""Factory of TD-Toeplitz inner products for the Precompute seam (task T3).

    This is the object threaded through ``factored_likelihood.PrecomputeLikelihoodTerms``
    as the ``IP_factory`` kwarg (default None => the classic frequency-domain-diagonal
    lsu inner products, byte-identical).  Passing a ``TDInnerProductFactory`` instead swaps
    in ``ToeplitzComplexOverlap`` (for Q_lm(t)) and ``ToeplitzComplexIP`` (for scalar U/V)
    at the two FL call sites (FL:453/456 via ``ip_scalar``, FL:463 via ``overlap``) WITHOUT
    any other change to the Precompute path.

    It exposes exactly the two constructor-shaped methods FL needs:

      * ``ip_scalar(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q, inv_spec_trunc_Q,
        T_spec)`` -> a ``ToeplitzComplexIP`` (drop-in for ``lsu.ComplexIP``).
      * ``overlap(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q, inv_spec_trunc_Q, T_spec,
        full_output=...)`` -> a ``ToeplitzComplexOverlap`` (drop-in for
        ``lsu.ComplexOverlap``).

    The signatures mirror the lsu constructors the FL call sites use, so FL can call the
    factory with the identical positional/keyword arguments it would have handed to lsu.

    One ``ToeplitzInnerProductContext`` is held per DISTINCT detector configuration.  The
    intended usage is "one factory per detector" (build it next to the ILE PSD handling),
    but the factory also memoizes contexts keyed on
    ``(fmin, fMax, fNyq, deltaF, analyticPSD_Q, id(psd), solver)`` so repeated FL calls
    with the same detector configuration reuse the same operator (the cross-term and
    Q calls for a given detector share one context, built once).

    Parameters
    ----------
    solver : {"auto", "dense", "pcg", "gs"}
        Solver mode forwarded to ``ToeplitzInnerProductContext``.  For auto/dense/pcg this
        affects only the physical-covariance / augmentation path (``solve_covariance``); the
        precision-kernel inner products (this seam) are FFT-based and solver-independent.
        "gs" switches every inner product (Q, U, V) to the covariance kernel C^{-1}, with C
        = Toeplitz(ACF) cut to the segment, solved by tdanalysis' ToeplitzSolver (GS method;
        tdanalysis >= 1.1).
        Corresponds to the ILE ``--td-solver`` flag.
    acf_duration : float or None
        Optional ACF truncation duration [s] carried for the Stage-3 truncated-segment
        path; unused by the full-segment precision-kernel inner products (D5 DEFAULT: the
        full-segment path uses K = irfft(band-limited weights), not the ACF-covariance
        route).  Corresponds to the ILE ``--td-acf-duration`` flag.
    warn_inv_spec_trunc : bool
        If True (default), emit a one-time warning when a call site requests inverse-spectrum
        truncation (``inv_spec_trunc_Q=True``): the TD precision kernel K = irfft(1/S on band)
        IS the exact band-limited inverse-spectrum operator, so the approximate FD
        inverse-spectrum truncation is redundant here and is ignored (see D5 and the note in
        ``_resolve_inv_spec_trunc``).
    """

    def __init__(self, solver="auto", acf_duration=None, warn_inv_spec_trunc=True,
                 template_start_offset=None, template_end_offset=None,
                 gaps=None, segment_start_index=0, gap_time_origin=None,
                 glitch_specs=None, glitch_sigma="profile", glitch_basis_files=None,
                 glitch_prior_mode="m2", glitch_sigma_min=None,
                 window_shape=None):
        self.solver = solver
        # population-prior detection-conditioned prior mode {m2,signmix,pinned} (default m2 =
        # zero-mean Gaussian behavior, bit-identical) and the presence-conditioning amplitude floor
        # sigma_g_min (strain; default None = code default bounds).  Both apply only to a
        # data-informed --glitch-basis-file that carries mu_mix/sigma_central (Slepian bases
        # have no population mean -> m2 only).
        self.glitch_prior_mode = glitch_prior_mode
        self.glitch_sigma_min = glitch_sigma_min
        # Data Tukey window shape the surrounding pipeline applied to the data + PSD BEFORE
        # Precompute (ILE --window-shape, set in production by helper_LDG_Events from
        # --data-tukey-window-time; T1900249 <w^2> PSD factor auto-applied).  The TD path is
        # DOWNSTREAM of windowing: it receives the already-windowed data stream and the
        # already-rescaled PSD, exactly as the fd-diagonal path does, so no windowing lives in
        # this kernel.  Stored only for the one-time provenance banner in context_for (below):
        # window_shape==0/None means the data is un-windowed (rectangular) -- correct for
        # zero-noise equivalence tests, but a foot-gun on real data (see plan T7c: the TD path
        # MUST inherit the production window, not bypass it).
        self.window_shape = window_shape
        self._logged_window = False
        self.acf_duration = acf_duration
        self.warn_inv_spec_trunc = warn_inv_spec_trunc
        # G3 glitch marginalization: per-IFO basis support specs and the sigma_g mode.
        #   glitch_specs: {ifo: dict(t1,t2,f1,f2,r)} parsed from --glitch-basis (Sec. 3 Slepian).
        #   glitch_basis_files: {ifo: path.npz} parsed from --glitch-basis-file (Sec. 3 modes
        #                 3/4, data-informed PCA/SVD basis).  REQUIRED npz keys: 'basis'
        #                 (n_snip x r time-domain snippet columns at the analysis sample rate),
        #                 'lambda0' (length r, dimensionless diagonal shape prior, max entry 1
        #                 per Sec. 2b), 't0' (scalar: geocenter seconds relative to the EVENT of
        #                 snippet sample 0), 'fs' (snippet sample rate [Hz], validated vs ctx).
        #                 OPTIONAL population-informed keys (population-prior; absent => zero-mean Gaussian
        #                 flat behavior EXACTLY): 'sigma_shape' (r x r dimensionless SPD shape
        #                 covariance, max-eig 1 -- replaces the diagonal lambda0 as Lambda0 ==
        #                 Sigma_shape; COVARIANCE-ONLY or FULL mode), 'mu' (length r
        #                 dimensionless population-mean direction mu_hat, scale mixture
        #                 mu_a = sigma_g mu_hat; FULL mode).  SPD is validated on load.
        #   glitch_sigma: "profile" (only wired mode); grid marginalization is a documented
        #                 follow-on (out of scope for G3).
        # The data + geocenter epoch needed to place the basis and profile sigma_g are supplied
        # per detector via register_detector_data (called by ILE before Precompute), keyed on
        # id(psd) so context_for can attach the basis when it first builds a detector's context.
        self.glitch_specs = glitch_specs or {}
        self.glitch_basis_files = glitch_basis_files or {}
        self.glitch_sigma = glitch_sigma
        self._detector_registry = {}   # id(psd) -> dict(ifo, data_fd, event_time_geo)
        self._glitch_contexts = []     # contexts that had a basis attached (for Delta lnZ sum)
        # Stage-3 (T5) segment specification, forwarded to every context this factory builds.
        #   * acf_duration          -> segment_duration -> N_seg (kernel truncation, D5)
        #   * template_start/end     -> template-frame truncation offsets (ILE
        #                               --td-analysis-start / --td-analysis-end), seconds,
        #                               geocenter, relative to the template peak reference
        #   * gaps                   -> list of (t0, t1) seconds (library only; ILE --td-gap is refused)
        # NOTE for the full end-to-end ILE segment run (plan T7): the per-detector geocenter
        # placement (segment_start_index derived from the event time, gap_time_origin from
        # the detector epoch, and the segment margin = data-integration-window-half + Earth
        # light-crossing) is applied by the caller; the context here honors whatever indices
        # / origin it is handed.  The AC-level validation drives the context directly.
        self.template_start_offset = template_start_offset
        self.template_end_offset = template_end_offset
        self.gaps = gaps
        self.segment_start_index = segment_start_index
        self.gap_time_origin = gap_time_origin
        self._contexts = {}
        self._template_ref_time = None
        self._warned_inv_spec_trunc = False

    # -------------------------------------------------------------- inv-spec-trunc policy

    def _resolve_inv_spec_trunc(self, inv_spec_trunc_Q, T_spec):
        r"""Decide how inverse-spectrum truncation interacts with the TD kernel (D5).

        The TD factory uses the untruncated PSD-weight operator. Inverse-spectrum
        truncation changes that operator and is not applied a second time here.
        This convention does not remove the need to taper or pad the data or to
        choose a finite-segment covariance model consistently.
        """
        if inv_spec_trunc_Q and self.warn_inv_spec_trunc and not self._warned_inv_spec_trunc:
            import warnings
            warnings.warn(
                "TDInnerProductFactory ignores inv_spec_trunc_Q (T_spec={0}); "
                "it uses the untruncated PSD-weight operator. Data conditioning "
                "and finite-segment boundary choices remain required.".format(T_spec),
                stacklevel=3)
            self._warned_inv_spec_trunc = True
        return False, 0.0

    # -------------------------------------------------------------- context management

    def context_for(self, fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q=False):
        """Return (memoized) ToeplitzInnerProductContext for this detector configuration."""
        key = (round(float(fmin), 9),
               None if fMax is None else round(float(fMax), 9),
               round(float(fNyq), 9), round(float(deltaF), 9),
               bool(analyticPSD_Q), id(psd), self.solver,
               self.acf_duration, self.template_start_offset, self.template_end_offset,
               tuple(self.gaps) if self.gaps else None,
               self.segment_start_index, self.gap_time_origin)
        ctx = self._contexts.get(key)
        if ctx is None:
            reg = self._detector_registry.get(id(psd))
            has_glitch = reg is not None and (
                reg["ifo"] in self.glitch_specs or reg["ifo"] in self.glitch_basis_files)
            full_n = 2 * int(float(fNyq) / float(deltaF))
            shortened = (self.acf_duration is not None and
                         int(round(float(self.acf_duration) * 2.0 * float(fNyq))) != full_n)
            if has_glitch and (self.gaps or self.template_start_offset is not None
                    or self.template_end_offset is not None or self.segment_start_index != 0
                    or shortened):
                raise ValueError("Glitch marginalization requires the full segment; "
                                 "cannot combine glitch bases with TD cuts, gaps or "
                                 "a shortened --td-acf-duration")
            if not self._logged_window:
                # One-time provenance banner: assert the TD kernel inherited the SAME data
                # window the surrounding pipeline applied (windowing + PSD <w^2> rescale happen
                # upstream at data load, so the TD path sees identically-windowed data as fd).
                _ws = self.window_shape
                print("  [TD] inner-product factory: inherited data window_shape=%s "
                      "(windowing + T1900249 <w^2> PSD rescale applied UPSTREAM at data load; "
                      "the TD kernel sees the same windowed data/PSD as the fd-diagonal path)."
                      % (_ws,))
                if _ws is None or float(_ws) == 0.0:
                    import warnings
                    warnings.warn(
                        "TDInnerProductFactory: window_shape=%s (rectangular / un-windowed "
                        "data).  This is correct ONLY for zero-noise equivalence tests; a real "
                        "production run must inherit the pipeline window (helper_LDG_Events "
                        "--data-tukey-window-time -> ILE --window-shape).  See plan T7c." % (_ws,),
                        stacklevel=2)
                self._logged_window = True
            # The registered stream epoch is measured relative to the event.
            # Context gap conversion subtracts the time of segment sample zero.
            gap_origin = self.gap_time_origin
            reg = self._detector_registry.get(id(psd))
            if self.gaps and gap_origin is None and reg is not None:
                evt = reg["event_time_geo"]
                if evt is not None:
                    gap_origin = (float(reg["data_fd"].epoch) - float(evt)
                                  + self.segment_start_index / (2.0 * float(fNyq)))
            ctx = ToeplitzInnerProductContext(
                psd, deltaF, fNyq, fmin, fMax=fMax,
                analyticPSD_Q=analyticPSD_Q, solver=self.solver,
                segment_duration=self.acf_duration,
                segment_start_index=self.segment_start_index,
                gaps=self.gaps, gap_time_origin=gap_origin,
                template_start_offset=self.template_start_offset,
                template_end_offset=self.template_end_offset,
                template_ref_time=self._template_ref_time)
            if reg is not None and reg["event_time_geo"] is not None:
                ctx._template_data_time_origin = (float(reg["event_time_geo"])
                                                 - float(reg["data_fd"].epoch))
            self._maybe_attach_glitch(ctx, psd)
            self._contexts[key] = ctx
        return ctx

    def assert_data_calibration_supported(self, fmin, fMax, fNyq, deltaF, psd,
                                          analyticPSD_Q=False):
        """Data-side frequency calibration must commute with the precision."""
        ctx = self.context_for(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q)
        if ctx.segment_active or ctx.glitch_active:
            raise ValueError("Data-side calibration does not commute with TD cuts, gaps "
                             "or glitch projection; this combination is unsupported, "
                             "including --calibration-global-norm")

    def register_template_modes(self, hlms):
        """Use one dominant-mode peak for every mode and detector's temporal cut.

        Called once per intrinsic waveform before any U, V or Q is constructed.
        Cached detector contexts must also update when processing another waveform.
        """
        if self.template_start_offset is None and self.template_end_offset is None:
            return
        if not hlms:
            raise ValueError("A template-frame cut requires waveform modes")
        modes = {mode: _ifft_to_td(h) for mode, h in hlms.items()}
        dominant = max(modes, key=lambda m: float(np.max(np.abs(modes[m]))))
        h = hlms[dominant]
        dt = 1.0 / (h.data.length * h.deltaF)
        self._template_ref_time = template_peak_time(modes[dominant], dt, float(h.epoch))
        for ctx in self._contexts.values():
            ctx.template_ref_time = self._template_ref_time

    def register_detector_data(self, ifo, psd, data_fd, event_time_geo=None):
        """Register a detector's (psd, data, event time) so its glitch basis can be attached.

        Called by ILE once per detector BEFORE Precompute.  Keyed on ``id(psd)`` (the same
        object Precompute forwards to ``context_for``), so the memoized context for that
        detector is the one that receives the basis.  ``event_time_geo`` is the geocenter
        event GPS time; the basis support offsets (geocenter seconds, per --glitch-basis) are
        placed at sample ``round((event_time_geo + t - float(data.epoch)) / deltaT)``.
        """
        self._detector_registry[id(psd)] = dict(
            ifo=ifo, data_fd=data_fd, event_time_geo=event_time_geo)

    def _maybe_attach_glitch(self, ctx, psd):
        r"""Attach the per-IFO Slepian glitch basis to a freshly built context (G3).

        EPOCH ARITHMETIC (documented).  --glitch-basis gives (t1,t2) in GEOCENTER seconds
        relative to the event; the analysis stream's sample 0 sits at GPS ``float(data.epoch)``.
        The basis support in stream samples is therefore
            i0 = round((event_time_geo + t1 - float(data.epoch)) / deltaT)
            i1 = round((event_time_geo + t2 - float(data.epoch)) / deltaT)
        which we hand to ``slepian_basis`` as seconds-from-sample-0 (t_start = i0*deltaT).  If
        ``event_time_geo`` is None (unit tests), t1/t2 are taken as seconds from sample 0.
        """
        reg = self._detector_registry.get(id(psd))
        if reg is None:
            return
        ifo = reg["ifo"]
        spec = self.glitch_specs.get(ifo)
        basis_file = self.glitch_basis_files.get(ifo)
        if spec is None and basis_file is None:
            return
        if ctx.segment_active:
            raise ValueError(
                "Glitch marginalization requires the full segment; cannot combine "
                "--glitch-basis/--glitch-basis-file with TD cuts, gaps or a "
                "shortened --td-acf-duration (%s)." % ifo)
        from RIFT.likelihood import td_rank_augmentation as tra
        data_fd = reg["data_fd"]
        evt = reg["event_time_geo"]
        origin = 0.0 if evt is None else (float(evt) - float(data_fd.epoch))
        if basis_file is not None:
            # Data-informed basis from file (--glitch-basis-file, Sec. 3 modes 3/4).  The
            # snippet columns are zero-embedded at t0 (same epoch arithmetic as the Slepian
            # branch: geocenter-relative-to-event -> seconds from segment sample 0 -> sample).
            payload = np.load(basis_file)
            for _key in ("basis", "lambda0", "t0", "fs"):
                if _key not in payload:
                    raise ValueError(
                        "--glitch-basis-file %s: npz %r missing required key %r "
                        "(need basis, lambda0, t0, fs)." % (ifo, basis_file, _key))
            snip = np.asarray(payload["basis"], dtype=np.float64)
            Lambda0 = np.asarray(payload["lambda0"], dtype=np.float64)
            fs_file = float(payload["fs"])
            if not np.isclose(fs_file, ctx.fs, rtol=1e-9):
                raise ValueError(
                    "--glitch-basis-file %s: snippet sample rate %g Hz != analysis rate %g Hz."
                    % (ifo, fs_file, ctx.fs))
            if snip.ndim != 2 or snip.shape[1] != Lambda0.shape[0]:
                raise ValueError(
                    "--glitch-basis-file %s: basis shape %s inconsistent with lambda0 shape %s."
                    % (ifo, snip.shape, Lambda0.shape))
            r_file = snip.shape[1]
            # OPTIONAL population-informed keys (population-prior; backward compatible: absent
            # keys => the zero-mean Gaussian flat behavior exactly).  The prior MODE (--glitch-prior-mode,
            # self.glitch_prior_mode) selects how they are used:
            #   sigma_shape   (r,r): dimensionless SPD moment-matched (M2) shape covariance
            #                 (max-eig 1); the OPERATING DEFAULT (mode m2), ZERO MEAN.
            #   mu_mix        (r,) : dimensionless population MEAN DIRECTION mu_hat (sqrt-max-eig
            #                 normalized).  With sigma_central it enables the detection-conditioned
            #                 modes: mu_a(sigma_g) = sigma_g mu_mix, the sign-mixture component
            #                 mean.  Named distinctly from the deprecated bare 'mu'.
            #   sigma_central (r,r): dimensionless SPD CENTERED covariance (same normalization);
            #                 the per-component covariance of the two-component sign mixture.
            #   mu            (r,) : DEPRECATED bare strain-domain mean (legacy FULL record).
            #                 Keeps its loud UNPHYSICAL warning; honored only in the m2/flat path.
            def _read_spd(key):
                M = np.asarray(payload[key], dtype=np.float64)
                if M.shape != (r_file, r_file):
                    raise ValueError("--glitch-basis-file %s: %s shape %s != (r,r)=(%d,%d)."
                                     % (ifo, key, M.shape, r_file, r_file))
                M = 0.5 * (M + M.T)
                if np.linalg.eigvalsh(M)[0] <= 0:
                    raise ValueError("--glitch-basis-file %s: %s is not SPD (min eig <= 0)."
                                     % (ifo, key))
                return M

            files = getattr(payload, "files", list(payload.keys()))
            mode = self.glitch_prior_mode
            mu0 = None
            Sigma_shape = _read_spd("sigma_shape") if "sigma_shape" in files else None
            mu_mix = (np.asarray(payload["mu_mix"], dtype=np.float64).reshape(-1)
                      if "mu_mix" in files else None)
            sigma_central = _read_spd("sigma_central") if "sigma_central" in files else None
            if mu_mix is not None and mu_mix.size != r_file:
                raise ValueError("--glitch-basis-file %s: mu_mix length %d != r=%d."
                                 % (ifo, mu_mix.size, r_file))
            has_mix = (mu_mix is not None and sigma_central is not None)

            if mode in ("signmix", "pinned"):
                # Detection-conditioned two-component sign mixture (signmix) or its decisive-sign
                # limit (pinned).  Both use the CENTERED covariance per component and +-mu_mix.
                if not has_mix:
                    raise ValueError(
                        "--glitch-prior-mode %s for %s requires the npz keys 'mu_mix' + "
                        "'sigma_central' (regenerate the basis with fit_population.py; a Slepian "
                        "band basis has no population mean -- use mode m2)." % (mode, ifo))
                Lambda0 = sigma_central
                mu0 = mu_mix
                band_note = ("file:%s [%s pop-prior: +-mu_mix, Sigma_central]"
                             % (os.path.basename(basis_file), mode))
                if mode == "pinned":
                    import warnings
                    warnings.warn(
                        "--glitch-prior-mode pinned for %s pins the glitch to the +mu_mix branch "
                        "(the decisive-sign limit of the sign mixture).  This is legitimate ONLY "
                        "when the closed-form sign odds ln(L+/L-) = 2<d-h|B mu_a>_C' is DECISIVE "
                        "(loud, well-localized glitch); check the [G3] sign-odds line.  For weak "
                        "glitches use m2 (default) or signmix." % ifo,
                        RuntimeWarning, stacklevel=2)
                    print("  [G3] %s prior-mode=pinned: pinning +mu_mix (decisive-sign limit; "
                          "assumes the sign odds below are decisive)." % ifo)
            else:                                        # m2 (default) or flat -- ZERO population mean
                if Sigma_shape is not None:
                    Lambda0 = Sigma_shape                # M2 moment-matched shape (zero mean)
                    band_note = ("file:%s [M2 pop-prior: Sigma_shape (zero mean)]"
                                 % os.path.basename(basis_file))
                else:
                    band_note = "file:%s" % os.path.basename(basis_file)
                # DEPRECATED bare 'mu' legacy path (loud UNPHYSICAL warning; unchanged behavior).
                if "mu" in files:
                    _mu = np.asarray(payload["mu"], dtype=np.float64).reshape(-1)
                    if _mu.size != r_file:
                        raise ValueError("--glitch-basis-file %s: mu length %d != r=%d."
                                         % (ifo, _mu.size, r_file))
                    if np.any(_mu != 0.0):
                        import warnings
                        warnings.warn(
                            "--glitch-basis-file %s carries a NONZERO deprecated bare population "
                            "MEAN (mu). This is UNPHYSICAL as a strain-domain prior mean (no "
                            "phase-coherent glitches; aligned-corpus mean is a sign-alignment-frame "
                            "artifact) and is superseded by covariance-only (M2) / the sign mixture. "
                            "Honoring it only to reproduce the FULL-mode record." % ifo,
                            RuntimeWarning, stacklevel=2)
                        print("  [G3] WARNING %s: NONZERO deprecated bare MEAN (mu) present -- "
                              "UNPHYSICAL strain-domain prior mean (PI 2026-07-08); prefer m2/signmix."
                              % ifo)
                        mu0 = _mu
                        band_note = ("file:%s [FULL(legacy) pop-prior: mu+Sigma_shape]"
                                     % os.path.basename(basis_file))
            t_start = origin + float(payload["t0"])
            i0 = int(round(t_start * ctx.fs))
            n_snip = snip.shape[0]
            if i0 < 0 or i0 + n_snip > ctx.N:
                raise ValueError(
                    "--glitch-basis-file %s: snippet [%d,%d) falls outside the segment [0,%d)."
                    % (ifo, i0, i0 + n_snip, ctx.N))
            B = np.zeros((ctx.N, snip.shape[1]))
            B[i0:i0 + n_snip, :] = snip
            t_end = t_start + n_snip / ctx.fs
            mode_for_attach = mode
        else:
            t_start = origin + float(spec["t1"])
            t_end = origin + float(spec["t2"])
            B, Lambda0 = tra.slepian_basis(
                ctx.N, ctx.fs, t_start, t_end, float(spec["f1"]), float(spec["f2"]),
                r=spec.get("r"))
            mu0 = None
            band_note = "band=[%g,%g]Hz" % (spec["f1"], spec["f2"])
            # A Slepian band basis carries no population mean -> the mixture/pinned modes are
            # meaningless here; fall back to m2 (zero-mean) regardless of the requested mode.
            mode_for_attach = "m2"
        block = (B, Lambda0, ifo) if mu0 is None else (B, Lambda0, ifo, mu0)
        info = ctx.attach_glitch_basis([block], data_fd, sigma_g=self.glitch_sigma,
                                       sigma_g_min=self.glitch_sigma_min,
                                       prior_mode=mode_for_attach)
        ctx._glitch_ifo_label = ifo
        self._glitch_contexts.append(ctx)
        _odds = info.get("sign_odds")
        print("  [G3] %s glitch basis attached: r=%d, support=[%.4f,%.4f]s, %s, "
              "mode=%s, sigma_g=%.3e, Delta_lnZ=%.4f"
              % (ifo, B.shape[1], t_start, t_end, band_note, mode_for_attach,
                 info["sigma_g"], info["delta_lnZ"]))
        if _odds is not None:
            # per-event regime indicator the paper promises, logged next to guess_snr/Delta_lnZ
            print("  [G3] %s sign-odds ln(L+/L-)=2<d|B mu_a>_C' = %+.4f (data-only, h=0); "
                  "Delta_lnZ[+]=%.4f Delta_lnZ[-]=%.4f -- %s regime"
                  % (ifo, _odds, info["delta_lnZ_plus"], info["delta_lnZ_minus"],
                     "DECISIVE" if abs(_odds) > 4.0 else "sign-undecided"))

    def glitch_delta_lnZ_total(self):
        """Sum of per-detector glitch Delta lnZ over all contexts with a basis attached (G3).

        This is the additive evidence constant ILE adds to lnL_marg (it does NOT cancel;
        Sec. 4).  For signmix this is the sum of the ACTIVE branch's per-detector Delta lnZ
        (the ILE combines branch lnL_marg via logsumexp - ln2; each branch adds its own active
        Delta lnZ before the combine).  Zero if no glitch basis was attached (default OFF)."""
        return float(sum(c.glitch_delta_lnZ for c in self._glitch_contexts))

    @property
    def signmix_active(self):
        """True iff a two-component sign-mixture glitch prior is CONFIGURED (population-prior).

        When True the ILE must evaluate the per-intrinsic-point likelihood for BOTH +-mu
        branches (via ``set_active_glitch_sign``) and combine the two lnL_marg by
        logsumexp - ln2.  m2 / pinned / off => False (single pass).

        CONFIG-based (not context-based): the glitch context is attached lazily during the
        first Precompute, so this must be decidable BEFORE any context exists (the ILE reads
        it at the top of each intrinsic point, ahead of Precompute).  signmix requires a
        data-informed --glitch-basis-file carrying mu_mix/sigma_central; a Slepian
        --glitch-basis has no population mean and is forced to m2, so it never triggers the
        two-pass even under --glitch-prior-mode signmix."""
        return (self.glitch_prior_mode == "signmix" and bool(self.glitch_basis_files))

    def set_active_glitch_sign(self, sign):
        """Set the active +-mu sign-mixture branch on EVERY attached glitch context (signmix).

        The ILE two-pass calls this with +1 then -1; each context recomputes its
        glitch-subtracted data (Q') and active per-detector Delta lnZ for that branch."""
        for c in self._glitch_contexts:
            c.set_active_glitch_sign(sign)

    def glitch_sign_odds_report(self):
        """Return {ifo: closed-form sign log-odds} for every glitch context that carries a mean.

        The per-event regime indicator (population-prior), computed at attach time with
        h=0 (data-only).  Empty when no mean direction is carried (m2 without mu_mix, Slepian
        bases)."""
        out = {}
        for c in self._glitch_contexts:
            odds = getattr(c, "_glitch_sign_odds", None)
            if odds is not None:
                out[getattr(c, "_glitch_ifo_label", "?")] = odds
        return out

    # -------------------------------------------------------------- FL-shaped constructors

    def ip_scalar(self, fmin, fMax, fNyq, deltaF, psd,
                  analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.):
        """Drop-in for ``lsu.ComplexIP(...)`` -> a ``ToeplitzComplexIP`` (scalar U/V)."""
        self._resolve_inv_spec_trunc(inv_spec_trunc_Q, T_spec)
        ctx = self.context_for(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q=analyticPSD_Q)
        return ToeplitzComplexIP(ctx)

    def overlap(self, fmin, fMax, fNyq, deltaF, psd,
                analyticPSD_Q=False, inv_spec_trunc_Q=False, T_spec=0.,
                full_output=True):
        """Drop-in for ``lsu.ComplexOverlap(..., full_output=True)`` -> Q_lm(t) overlap."""
        self._resolve_inv_spec_trunc(inv_spec_trunc_Q, T_spec)
        ctx = self.context_for(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q=analyticPSD_Q)
        return ToeplitzComplexOverlap(ctx)

    # -------------------------------------------------------------- T6 diagnostic

    def report_uv_edge_sensitivity(self, hlms, psd, fmin, fMax, fNyq, deltaF,
                                   analyticPSD_Q=False, det=None, window_half=0.075,
                                   warn_tol=1e-2):
        r"""Log the dominant-mode U edge-sensitivity next to guess_snr (plan T6).

        Called from Precompute when the TD inner product is active.  Picks the dominant mode
        (largest |h| peak), applies the template-frame truncation, and reports the U
        variation across +-window_half of the marginalization window (see
        ``ToeplitzInnerProductContext.uv_edge_sensitivity``).  Warns above ``warn_tol``
        (default 1e-2 on the dominant pair, per the plan).  Returns the diagnostic dict.
        """
        if not hlms:
            return None
        ctx = self.context_for(fmin, fMax, fNyq, deltaF, psd, analyticPSD_Q=analyticPSD_Q)
        # dominant mode = largest time-domain peak amplitude
        dom = max(hlms.keys(),
                  key=lambda m: float(np.max(np.abs(_ifft_to_td(hlms[m])))))
        h_td = ctx.apply_template_truncation(_ifft_to_td(hlms[dom]), float(hlms[dom].epoch))
        h_td = ctx.template_in_data_frame(h_td, hlms[dom].epoch)
        rep = ctx.uv_edge_sensitivity(h_td, window_half)
        tag = "" if det is None else (" [%s]" % det)
        msg = ("  [T6]%s U edge-sensitivity (mode %s, +-%.3f s window): rel_var=%.3e "
               "(U0=%.6g); segment_active=%s, N_seg=%d, is_spd=%s"
               % (tag, dom, window_half, rep["rel_var"], abs(rep["U0"]),
                  ctx.segment_active, ctx.N_seg, ctx.is_spd))
        print(msg)
        if rep["rel_var"] > warn_tol:
            import warnings
            warnings.warn(
                "TDInnerProductFactory: U edge-sensitivity rel_var=%.3e exceeds tol %.1e on "
                "the dominant pair%s -- the scalar-U (shift-invariant) assumption at "
                "factored_likelihood FL:1936/2019 may be violated for this segment "
                "configuration (a GPS-frame cut or too-wide time window; see plan D1/T6)."
                % (rep["rel_var"], warn_tol, tag), stacklevel=2)

        # G3 extension: when a glitch basis is attached, check whether it overlaps the SLIDING
        # dominant-template support across the marginalization window.  The scalar U' correction
        # is evaluated at the reference time (template-frame convention); if the glitch support
        # overlaps the template track the correction becomes time-dependent (U'(t)) and the
        # time-indexed U'(t) lookup -- deferred here -- would be needed.  We warn but proceed.
        if ctx.glitch_active and ctx._glitch_support is not None:
            gi0, gi1 = ctx._glitch_support
            hpk = int(np.argmax(np.abs(h_td)))
            nz = np.nonzero(np.abs(h_td) > 1e-6 * np.max(np.abs(h_td)))[0]
            hi0, hi1 = (int(nz[0]), int(nz[-1]) + 1) if nz.size else (hpk, hpk + 1)
            n = rep["n_shift"]
            # sliding support of the template across +-n samples
            overlap = (gi0 < hi1 + n) and (hi0 - n < gi1)
            rep["glitch_overlaps_template"] = bool(overlap)
            print("  [G3]%s glitch basis support=[%d,%d) vs template support=[%d,%d) (+-%d "
                  "slide): overlap=%s; U' evaluated at reference time (time-indexed U'(t) "
                  "deferred)." % (tag, gi0, gi1, hi0, hi1, n, overlap))
            if overlap:
                import warnings
                warnings.warn(
                    "TDInnerProductFactory: glitch basis%s overlaps the sliding template "
                    "support -- the U' correction is time-dependent across the window; G3 uses "
                    "the reference-time scalar U' (time-indexed U'(t) lookup deferred, see "
                    "TD_LIKELIHOOD.md Sec. 2/2b)." % tag, stacklevel=2)
        return rep


# --------------------------------------------------------------------------- template truncation (T5)

def template_peak_time(x_td, deltaT, epoch):
    r"""Peak-amplitude reference time t_ref of a template TD array (plan D1).

    Epoch bookkeeping (documented precisely).  Sample ``i`` of ``x_td`` sits at time
    ``t_i = float(epoch) + i * deltaT`` in the template's OWN frame -- ``epoch`` is the GPS
    time of sample 0 as carried by the lal series the mode was inverse-FFT'd from (for a
    RIFT ``hlmoff`` FD mode this is the ``hlm.epoch`` set at lalsimutils.py:3338,
    ``epoch = -(N deltaT) * fd_centering_factor``, i.e. the peak sits near ``t ~ 0`` in this
    frame).  ``t_ref`` is defined as the argmax of ``|x_td|``:

        t_ref = float(epoch) + argmax(|x_td|) * deltaT.

    This is the reference the user-facing ``--td-analysis-start/--td-analysis-end`` offsets
    are measured from.
    """
    x = np.asarray(x_td)
    ipk = int(np.argmax(np.abs(x)))
    return float(epoch) + ipk * float(deltaT)


def truncate_timeseries_template_frame(x_td, deltaT, epoch,
                                       start_offset, end_offset, t_ref=None):
    r"""Hard-cut a template TD array to [t_ref+start_offset, t_ref+end_offset] (plan D1, T5).

    Zero every sample OUTSIDE the template-frame window; NO taper at the cut -- the whole
    point of the exact finite-segment treatment is a sharp boundary (the correlation
    structure of K_seg handles the template sliding off the edge exactly).  ``t_ref`` is the
    peak-amplitude reference (see ``template_peak_time``); if None it is computed from
    ``x_td``.  ``start_offset`` / ``end_offset`` are seconds relative to ``t_ref`` (either
    may be None to leave that side open).

    Interaction with the always-on start taper in ``lalsimutils.hlmoft`` (LSU:~3341,
    ``hlmsT[mode].data.data[:ntaper] *= vectaper``): that taper multiplies the FIRST
    ``ntaper ~ 1% of TDlen`` samples -- the low-frequency turn-on at the *start* of the
    waveform.  For a ringdown / post-peak cut (``start_offset >= 0``) the retained window
    begins at the peak, tens of thousands of samples AFTER that start-taper region, so the
    taper lies entirely in the discarded pre-peak portion and does NOT touch the analysis
    segment.  (Verified: for the 8 s / 2048 Hz template used here ntaper ~ 160 samples at the
    segment start while the peak sits near the segment centre.)  No taper disabling is needed
    for post-peak cuts; a cut that RETAINS the waveform start (start_offset very negative)
    would include the taper, which is documented here rather than silently altered.

    Returns a NEW array (copy) with the out-of-window samples set to 0.
    """
    x = np.asarray(x_td)
    if start_offset is None and end_offset is None:
        return x.copy()
    if t_ref is None:
        t_ref = template_peak_time(x, deltaT, epoch)
    N = x.shape[-1]
    i_lo = 0
    i_hi = N
    if start_offset is not None:
        # first retained sample: t_i >= t_ref+start_offset
        i_lo = int(np.ceil((t_ref + float(start_offset) - float(epoch)) / float(deltaT)))
    if end_offset is not None:
        # last retained sample: t_i <= t_ref+end_offset  =>  half-open upper index
        i_hi = int(np.floor((t_ref + float(end_offset) - float(epoch)) / float(deltaT))) + 1
    i_lo = max(0, min(N, i_lo))
    i_hi = max(0, min(N, i_hi))
    out = np.zeros_like(x)
    if i_hi > i_lo:
        out[..., i_lo:i_hi] = x[..., i_lo:i_hi]
    return out


# --------------------------------------------------------------------------- helpers

def _ifft_to_td(hf):
    """iFFT a two-sided COMPLEX16FrequencySeries to a TD complex array (lal conventions).

    Uses lal's reverse COMPLEX16 FFT (multiplies by deltaF), matching every convention the
    FD inner products assume.  Returns a plain numpy array (a copy).
    """
    from RIFT import lalsimutils as lsu
    return np.asarray(lsu.DataInverseFourier(hf).data.data).copy()


def _hertz_unit():
    try:
        from RIFT.lalsimutils import lsu_HertzUnit
        return lsu_HertzUnit
    except Exception:  # pragma: no cover
        return lal.HertzUnit


def _dimensionless_unit():
    try:
        from RIFT.lalsimutils import lsu_DimensionlessUnit
        return lsu_DimensionlessUnit
    except Exception:  # pragma: no cover
        return lal.DimensionlessUnit
