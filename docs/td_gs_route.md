# Time-domain likelihood in RIFT: the `td-gs` code path

`integrate_likelihood_extrinsic_batchmode` (ILE) selects the inner product at run time:

| `--internal-ile-inner-product` | What it does |
|---|---|
| `fd-diagonal` (default) | Standard RIFT: frequency-domain inner products weighted by 1/S(f). Unchanged. |
| `td-toeplitz` | Time-domain precision kernel K = irfft(1/S) on the full or truncated segment. |
| `td-gs` | Time-domain code path described here: the likelihood of tdanalysis on arbitrary spliced segments. |

## What `td-gs` computes

The data are cut to one or more **pieces** (square windows: samples are kept or dropped, nothing
is tapered). Each piece has its own noise covariance C = Toeplitz(ACF[:n]); pieces are independent
(block-diagonal C), as in tdanalysis' segmented likelihood. For every intrinsic point, Precompute
returns, on RIFT's usual storage time grid:

- Q_lm(t) = conj(h_lm placed at t)ᵀ w, with w = C⁻¹ d computed once per detector per job;
- U_lm,l'm'(t) and V_lm,l'm'(t), the template cross terms on the pieces. Because the pieces are
  fixed in time and the template slides through them, these depend on the arrival time t;
- with calibration marginalization, the same three quantities per realization c for the template
  filtered with that realization's FIR (section "Calibration marginalization" below).

The vectorized likelihood (`DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`, CPU numpy or GPU
cupy) looks up U(t), V(t) at each sample's arrival-time index, exactly as it looks up Q(t).

- No FFT, window or taper is applied to the data or the templates. The data are read as a time
  series with no Tukey window (`--window-shape` is skipped and the data-side <w²> PSD factor is
  not applied). Templates are the TD modes from `hlmoft(..., no_condition=True)`.
- C⁻¹ is applied by tdanalysis' `ToeplitzSolver`: a Cholesky factor for pieces up to
  `--td-dense-max-n` samples (default 4096), and the Gohberg–Semencul (GS) form for longer pieces.
  FFTs appear only inside this Toeplitz algebra, as exact zero-padded linear correlations.
- U(t), V(t): on Cholesky pieces, the cut shifted templates are stacked (in column blocks under a
  memory bound) and C⁻¹ is applied directly. On GS pieces, when the shifted template ends inside
  the piece for every shift of the storage grid (the full segment; a ringdown piece that extends
  past the template's last sample at the latest shift) — or, mirrored, starts inside it for every
  shift — each U(t) is a windowed sum of one correlation of the GS kernels with the unshifted
  template (`prefix`: four FFT correlations per mode, then O(n + N_window) per mode pair); on GS
  pieces the template overruns at both ends, the kernel correlations are updated sample by sample
  (`recursion`, no per-shift solve). Every method is exact; they agree to rounding (~1e-15 on the
  same kernels). `--td-u-method {auto,prefix,recursion,batched,chunked}` (default `auto` = this
  split; `prefix` refuses pieces where its condition fails). ILE prints the method used for each piece
  (`U/V methods`) in the verbose Precompute line.

## What is tdanalysis' and what RIFT adds

The Toeplitz algebra is tdanalysis' public API (tdanalysis 1.1, documented in its
`docs/toeplitz_api.md`; the names are those of the final 1.1 release); RIFT imports it and
keeps no copy:

| tdanalysis | used by RIFT for |
|---|---|
| `tdanalysis.forms.toeplitz_solver.ToeplitzSolver` | C⁻¹ on each piece (`TDGSDetectorContext.solvers`, `apply_Cinv`; `whiten` for `--td-whiteness-check`): checks that Toeplitz(ACF) is positive definite, Cholesky up to `--td-dense-max-n`, GS above, with generators from refined Levinson solves of the ACF (no dense inverse); also `td_inner_product`'s `solver="gs"` |
| `tdanalysis.forms.sliding.sliding_data_term` | Q(t) on each piece, against w = C⁻¹d |
| `tdanalysis.forms.sliding.sliding_self_terms` | U(t), V(t) on each piece, with the `--td-u-method` methods (`auto`, `prefix`, `recursion`, `batched`, `chunked`); the method used is reported per piece |
| `tdanalysis.forms.sliding.is_turn_on_inside_window` | the check that the template is already on at each piece start |
| `tdanalysis.utils.filters.highpass_zero_phase` | `--td-highpass` |
| `tdanalysis.utils.filters.fir_taps_from_response`, `fir_truncate`, `fir_max_lag`, `fir_truncate_at_max_lag` | calibration realizations as exact FIR filters on the template |
| `tdanalysis.utils.subsample_align.fractional_shift`, `SHIFT_STENCIL` | `--td-subsample` |

RIFT keeps what belongs to RIFT: lal series, epochs and the output contract of
`ComputeModeIPTimeSeries`; the (l,m) mode bookkeeping (modes as rows of one template array, pair
dictionaries for the likelihood); pieces in units of M_ref from the command line, converted to
sample ranges; RIFT's storage grid of shifts (each piece is passed to tdanalysis as the window
`[i0, i1)` of the data with `offset = i0`, and the per-piece results are summed, C being
block-diagonal over pieces); the turn-on refusal and its message; the calibration conventions
(RIFT's packing, `rift`/`bilby` correction types, conjugate phase), the data weights W_c = K_cᵀw and
the support of the filtered template; the interleaving of the sub-sample phases onto the fine
grid; the ILE options; and the SVD compression in `factored_likelihood`.

`prefix` and `recursion` use GS generators of the ACF on every piece, also on Cholesky pieces. On a
badly conditioned ACF (cond 1e8) they agree with the Cholesky solve to ~cond(C)·ε (~1e-9 relative);
`auto` therefore uses `chunked` on Cholesky pieces. Earlier versions of this code path took the
generators of a dense piece from the columns of its dense inverse; the refined Levinson generators
are at least as accurate.

## Requirements

```bash
pip install 'RIFT[td]'                           # RIFT with tdanalysis >= 1.1 (brings waveformtools, spectools)
```

In an environment that already has lalsuite, bilby and pycbc, the dependencies can instead be
added without letting pip touch those:

```bash
pip install --no-deps 'tdanalysis>=1.1' waveformtools spectools
pip install termcolor                            # imported by waveformtools; matplotlib is assumed present
```

- tdanalysis 1.1 or later: `tdanalysis.forms.toeplitz_solver`, `tdanalysis.forms.sliding`,
  `tdanalysis.utils.filters` and `tdanalysis.utils.subsample_align` (tdanalysis 1.0.0 has none of
  them; importing `RIFT.likelihood.td_gs_precompute` then fails with an `ImportError` that names
  tdanalysis 1.1). RIFT uses the names of the final 1.1 release (`ToeplitzSolver.apply_Cinv` and
  `whiten`, `is_turn_on_inside_window`, `SLIDING_METHODS`, `PrefixNotApplicable`, the `support=`
  and `compute_V` arguments of `sliding_self_terms`, `fir_taps_from_response`, `fir_max_lag`,
  `fir_truncate_at_max_lag`, `gs_generators_from_acorr`, `PSDToAutocorrConfig(sampling_rate=)`).
  Earlier 1.1 development trees carry the same version string with other names; the import
  then fails with an `ImportError` that lists what is missing and asks for tdanalysis 1.1 (the
  final release).
- Sample rate: the ACF lag spacing must equal the analysis deltaT. The tdanalysis event ACFs are at
  4096 Hz, so run ILE with `--srate 4096` (ILE's default is 16384).
- ACF conditioning: use well-conditioned ACFs. The raw real-event ACFs in
  `tdanalysis/dev/nbs/acorr_from_psd` give cond(Toeplitz) ≈ 1e14–1e16 and a float64 error of
  ~0.1 nat on long pieces; the conditioned versions (`*_cond1e8_cap.npy`, cond ≈ 1e7–4e7) give
  ~1e-7 nat.
- Waveforms: only models that produce time-domain modes directly through lalsimulation
  `ChooseTDModes` (NRSur7dq4, NRHybSur3dq8, SEOBNRv4PHM, IMRPhenomTPHM, Taylor/SpinTaylor).
  NRSur7dq4 needs `--fmin-template 0` (or a low value) to return its full length. Models native to
  the frequency domain, ROM/gwsurrogate, gwsignal, NR lookup and external EOB are refused, because
  they taper or transform the template.
- `--vectorized` is required. Without `--gpu`, the NoLoop likelihood runs on the CPU with numpy.

## Flags

| ILE flag | pseudo_pipe pass-through | Meaning |
|---|---|---|
| `--internal-ile-inner-product td-gs` | `--internal-ile-inner-product` | select the code path |
| `--td-mref MSUN` | `--internal-ile-td-mref` | reference total mass (detector frame) that sets the unit of `--td-piece` |
| `--td-piece START,END` | `--internal-ile-td-piece` | one piece, in units of M_ref relative to the event time; repeatable; none = the full segment |
| `--td-ifo-time-offset IFO:SEC` | `--internal-ile-td-ifo-time-offset` | shift the piece reference time for one detector (e.g. to its detector-frame peak) |
| `--td-acf-file IFO:PATH.npy` | `--internal-ile-td-acf-file` | ACF in the tdanalysis format, float64 (2, N) [tau, R(tau)], tau[k] = k/fs |
| `--td-acf-cond K` | `--internal-ile-td-acf-cond` | without an ACF file: build the ACF from the PSD with tdanalysis PSDToAutocorr at this condition number (default 1e8); the number of floored in-band bins is printed |
| `--td-dense-max-n N` | `--internal-ile-td-dense-max-n` | Cholesky factor of C up to N samples per piece, GS above (default 4096; both exact) |
| `--td-u-method` | `--internal-ile-td-u-method` | `auto` (default), `prefix`, `recursion`, `batched`, `chunked` |
| `--td-u-stride K` | `--internal-ile-td-u-stride` | compute U(t), V(t) every K samples and interpolate (default 1 = exact) |
| `--td-subsample K` | `--internal-ile-td-subsample` | place templates to 1/K of a sample with tdanalysis' Lanczos resampler (default 1 = whole samples); Precompute cost x K (see below) |
| `--td-highpass F` | `--internal-ile-td-highpass` | zero-phase 8th-order Butterworth high-pass of the TD data at F Hz before C⁻¹d (tdanalysis' `highpass_zero_phase`; default off; e.g. 10 for real strain with walled ACFs) |
| `--td-whiteness-check` | `--internal-ile-td-whiteness-check` | print whitening diagnostics per piece (dᵀC⁻¹d/n ≈ 1 expected on noise-only data; variance and lag-1 correlation of L⁻¹d, by the Cholesky factor on Cholesky pieces and Levinson-Durbin on GS pieces) |
| `--td-u-svd-tol TOL` | `--internal-ile-td-u-svd-tol` | compress U(t), V(t) by an SVD over time, keeping components with s_k > TOL·s_0 (recommended 1e-13; default off; see below; not with calibration) |
| `--td-calibration-correction-type {rift,bilby}` | `--internal-ile-td-calibration-correction-type` | with `--calibration-envelope-directory`: how a realization becomes the template factor (default `rift` = RIFT's FD convention; see below) |
| `--td-calibration-fir-half-length L` | `--internal-ile-td-calibration-fir-half-length` | force the calibration FIR half-length (default: exact, from the template's reach; a forced shorter L is an approximation) |

RIFT's in-loop calibration flags (`--calibration-envelope-directory`, `--calibration-n-realizations`,
`--calibration-spline-count`, `--calibration-conjugate-phase`, `--calibration-global-norm`, the
adaptive draw count `--calibration-neff-cal-target` / `--calibration-n-realizations-max`, the
burn-in and pilot options) work unchanged on `td-gs`.

Refused with a clear error on `td-gs`: glitch marginalization, `--rotation-slow` / `--freqresponse`,
the `td-toeplitz` flags, `--calibration-fused-kernel` (the per-realization self term is time-indexed;
the loop reduction is used), `--td-u-svd-tol` together with calibration marginalization, and any
piece that would contain the template's turn-on for some shift on the storage grid (lower
`--fmin-template` or shrink `--internal-data-storage-window-half`).

## Sub-sample template placement: `--td-subsample K`

By default the template is placed by whole samples: Q(t), U(t), V(t) live on RIFT's storage grid at
spacing deltaT, the fractional part of the template epoch is carried in the rholms epoch, and the
likelihood looks each extrinsic sample's arrival time up at the nearest grid sample (or, with
`--interpolate-time`, by a cubic stencil on Q and on U, V). With nearest lookup lnL is a staircase in
t_c and in the sky (through each detector's delay): flat for a whole sample (0.24 ms at 4096 Hz) and
jumping at its edges, the same staircase tdanalysis removed in its `subsample_align` commits
('Place the IMR template peak at t_coal to a fraction of a sample'). Interpolating Q(t) is exact in
principle (Q is linear in the template, so the Lanczos interpolation of Q equals Q of the Lanczos-
shifted template), but U and V are quadratic in the template and interpolating them is not
equivalent to shifting the template.

`--td-subsample K` (K > 1) makes Precompute evaluate Q, U, V for K template sets: phase j is every
mode moved by s_j of a sample with tdanalysis' own kernel, `tdanalysis.utils.subsample_align.
fractional_shift` (a local Lanczos-windowed sinc of half-width `SHIFT_STENCIL` = 16, imported, not
copied; tdanalysis >= 1.1), with
s_j = j/K for j/K <= 1/2 and s_j = j/K - 1 placed one sample later otherwise — the residual-in-
[-1/2, 1/2] decomposition `GWTransientTD._get_peak_and_target_indices` uses, so where a
sample's fractional
offset falls on the 1/K grid the two codes evaluate the identical resampled template. Every phase
goes through the same exact methods as phase 0 (`auto` selection per piece and phase unchanged; phase
0 is the K = 1 result bit for bit), and the series are interleaved on a grid of spacing deltaT/K:
entry r <-> shift N_shift + r/K; the rholms carry deltaT/K and K N_window entries, same epoch.

The likelihood (`DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(..., td_subsample=K)`) finds
each sample's first time bin on the fine grid, `rint(t_first K / deltaT)`, and gathers its npts time
bins with stride K (one gather per bin as before, the same Q_inner kernels with a `stride`
argument), so the time-marginalization integral keeps its coarse bins and Simpson spacing
dx = deltaT and the likelihood cost does not grow with K. With the default nearest lookup the residual
placement step is deltaT/K; `--interpolate-time` applies its four-point cubic stencil on the fine
grid. The result is exact to the 1/K grid: for every sample and bin, lnL equals the dense residual
likelihood with the template moved by tdanalysis' kernel to the nearest 1/K point
(`test/test_td_subsample.py`: 1e-12 relative for Q, U, V on a well-conditioned ACF, ~1e-9 nat end to
end on the test setups). Measured on the test setup, lnL(t_c) over 0.1-sample steps has 4 plateaus
in 3 samples at K = 1 and 25 at K = 8.

Cost: Precompute x K per intrinsic point (K sets of shifted modes through the same methods; the
shifts themselves are negligible), packed series x K in memory (and in the GPU transfer); the
likelihood stage is unchanged. The piece/template-support checks apply to every phase: the kernel
spreads the template's first sample over the 15 samples before it, so an explicit piece must also
clear that margin (refused with the margin named). Calibration marginalization combines with
`--td-subsample` (section "Calibration marginalization", paragraph "With `--td-subsample K`"): each
phase's shifted template is filtered with each realization's FIR, and the realization blocks carry
the fine grid. The `--td-u-svd-tol` compression and `--td-u-stride` act per phase on the fine grid.

## Compressing U(t), V(t): `--td-u-svd-tol`

With 21 modes (l ≤ 4) the self term ⟨h|h⟩(t) is the expensive part of the likelihood stage: the
vectorized likelihood gathers n_lm(n_lm+1) = 462 pair time series per sample and time bin, against 21
for the data term Q(t). `--td-u-svd-tol TOL` replaces them by their singular components over time.

For each detector the pair time series of U(t) and V(t) (the same unordered-pair form the
uncompressed path uses: U_ii, U_ij + conj(U_ji) for i<j, and likewise V) are stacked as one
complex matrix M (N_window × n_lm(n_lm+1)) and its thin SVD M = W S Zᴴ is truncated to the r
components with s_k > TOL·s_0. Writing c_k(t) = W_tk s_k,

    ρ²(e, t) = Re Σ_col w_e[col] M[t_e, col] = Re Σ_k g_ek c_k(t_e),   g_ek = Σ_col w_e[col] Zᴴ_k,col ,

with w_e the extrinsic weights |F|² conj(Y_i) Y_j (U columns) and F² Y_i Y_j (V columns). The
weights are the same complex numbers as before, so the only change is the dropped components,
each bounded by TOL·s_0; g is formed once per sample (r contractions), and the time gather runs
over r series with the same nearest/cubic Q kernels as the data term (cubic interpolation is
linear in the series, so interpolating c_k equals interpolating M). The compression is done once
per intrinsic point on the CPU, after `PackLikelihoodDataStructuresAsArrays` and before the GPU
transfer (which then moves r series instead of two (N_window, n_lm, n_lm) arrays); ILE prints r,
s_1/s_0, s_r/s_0 and the reconstruction error per detector.

Measured on the test cases of `test/test_td_u_svd.py` (IMRPhenomTPHM l ≤ 2 and NRSur7dq4 l ≤ 4, PSD-derived
ACF):

| case | columns | r at 1e-10 / 1e-12 / 1e-13 / 1e-14 | lnL agreement at 1e-13 |
|---|---|---|---|
| l ≤ 2, full segment, 1228 points | 30 | 10 / 10 / 10 / 10 | 1e-15 |
| l ≤ 2, ringdown piece, 1228 points | 30 | 12 / 16 / 19 / 21 | 1e-13 |
| l ≤ 2, ringdown piece, 122 points | 30 | 10 / 12 / 15 / 17 | 1e-13 |
| l ≤ 4, full segment, 1228 points | 462 | 20 / 30 / 40 / 49 | 1e-14 |
| l ≤ 4, ringdown piece, 122 points | 462 | 84 / 111 / 116 / 121 | 1e-13 |

A full-segment U(t) is nearly constant (s_1/s_0 ~ 1e-5) and compresses to ~10 components; a
ringdown piece, where U(t) changes by factors of 100 across the grid, needs more, but never more
than min(N_window, columns). Recommended: `--td-u-svd-tol 1e-13`, which reproduces the
uncompressed likelihood to ~1e-13 relative in every case above; 1e-10 still gives ~1e-10. The
flag is refused outside `td-gs` (scalar cross terms have nothing to compress) and is subject to the
same restrictions as the time-indexed arrays (no calibration marginalization).

## Calibration marginalization

RIFT's in-loop calibration marginalization draws n_cal realizations η_c(f) per detector
(`RIFT/calmarg/generate_realizations.py`: cubic splines in log₁₀ f of amplitude and phase on
RIFT's reversed two-sided packing, unity outside [f_min, f_max]), multiplies the DATA by η_c in the
frequency domain, and averages the likelihood over c with the per-realization self term
<η_c h|η_c h>. On `td-gs` the data are never transformed, so each realization becomes a real FIR
filter K_c on the TEMPLATE ("calibrate the template, then cut"):

    (K_c h)[j] = Σ_i c_c[i] h[j−i],   c_c = inverse DFT of η_c on its own frequency grid (N real taps,
                                      i = −N/2+1 … N/2; real because η_c(−f) = conj η_c(f)),

so that K_c's response on the realization grid f_m = m/(N Δt) is η_c exactly (tdanalysis'
`fir_taps_from_response`). The FD code path multiplies
by η_c on that grid, i.e. convolves CIRCULARLY with the same taps; the TD code path convolves linearly
(zero-extended template), and the two agree wherever the FD code path does not wrap around. Then, per
realization,

    Q_c(t) = (K_c h_t)|_piecesᴴ w = h_tᴴ (K_cᵀ w),      U_c(t) = (K_c h_t)|ᴴ C⁻¹ (K_c h_t)|,   V_c likewise.

**Exactness.** Nothing is approximated beyond rounding:

- `W_c = K_cᵀ w` is one exact linear correlation (w is zero outside the pieces) with the full N taps,
  formed once per realization set (`TDCalibrationFilters`); Q_c(t) is then the exact correlation of
  the template's support with the matching window of W_c, for every template.
- U_c(t), V_c(t) run the existing exact U/V methods on the filtered template g_c = K_c h. Because the
  template has compact support, a tap c_c[i] meets a non-zero template sample on a piece only if |i| ≤
  R, the reach = max over pieces and grid shifts of the distance from a piece sample to the template's
  support; the filter truncated at L ≥ R gives the SAME values on the pieces as the N-tap filter.
  Per intrinsic point L = min(R, N/2) is used (all N taps when R ≥ N/2, e.g. on a full segment;
  tdanalysis' `fir_max_lag` and `fir_truncate_at_max_lag`). On the
  test ringdown piece (R = 1209 at N = 8192) U_c with L = R equals U_c with L = N/2−1 to 1e-15, while
  the error of a forced L = R/4 is 5e-4 (and 1e-2 at L = 64, 7e-4 at 256, 1e-5 at 1024 on a 4 s grid;
  the lnL_t error there is 160, 9, 0.15 nat). There is no tolerance to choose;
  `--td-calibration-fir-half-length` only exists to force a shorter (approximate) L for cost studies.
- The method selection sees the filtered template's support (extended by the taps' reach on each
  side, and passed to tdanalysis as `support=`, which ignores the rounding residue the
  convolution leaves outside it), so the prefix method is used only where its condition still holds; otherwise
  the recursion (GS pieces) or the chunked method (Cholesky pieces). The turn-on check applies to the unfiltered template as
  before; the FIR smears the turn-on step into the piece with the taps' magnitude at the turn-on
  distance, which Precompute prints as the "turn-on leakage" (the FD code path's circulant multiplication
  smears it the same way).

**Convention** (derivation in `test/td_calibration_study.md`, section 1). RIFT's FD default — η_c
multiplying the data in RIFT's reversed packing — equals, in the standard e^{−2πift} convention, the
template filtered with η_c at face value. That is the default here, `--td-calibration-correction-type
rift`: C_c = η_c for every detector, and `--calibration-conjugate-phase` gives C_c = conj η_c, as on the
FD code path. `bilby` uses bilby_pipe's per-detector semantics (`RIFT/calmarg/calibration.py`): C_c = η_c
for V1 (envelope of "template" type) and C_c = 1/η_c for H1, L1, K1 ("data" type: η maps calibrated
data to theoretical strain), conjugated as well with `--calibration-conjugate-phase`. The checks in
`test/test_td_gs_calibration.py` inject d = K_true Re h with the code path's own taps of η_true: the
matched convention recovers Q_c = (U_c + conj V_c)/2 at the injection to 1e-15 (`rift` with η_true,
`bilby`/H1 with 1/η_true, `bilby`/V1 with η_true, `rift --calibration-conjugate-phase` with conj
η_true), the mismatched ones differ by 3–6e-2.

**Likelihood.** Precompute returns rholms with n_cal contiguous blocks of Q_c(t) (block 0's epoch, as
the FD code path), the calibration-independent U(t), V(t) (one block long; used by
`--calibration-global-norm`), and, as the two trailing return values, the per-realization time-indexed
U_c(t), V_c(t), which ILE packs to `ctUArrayDict_cal[det]` of shape (n_cal, N_window, n_modes,
n_modes). `DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop` (loop method) gathers realization c's
self term at the same block-local window as its Q_c block, exactly as the uncalibrated time-indexed
self term; the streaming log-sum-exp over realizations, the importance weights, `return_lnLt`,
`return_cal_components` (so the adaptive draw count and the pilot work) and the burn-in layout
(n_cal = 1 on the full arrays) are unchanged. The fused kernel refuses these inputs.

**With `--td-subsample K`.** Each sub-sample phase j runs the calibrated Precompute on its own
template: the Lanczos shift first (the shifted template exactly as the uncalibrated `--td-subsample`
and `GWTransientTD` place it, same length as the input), then the realization's FIR as a full linear
convolution of that template, then the cut and the exact U/V methods. The shift and the FIR are both
linear time-invariant filters, so their order can matter only through the shift's cut at the
template's ends; this order is the calibration applied to the physically shifted signal, and it keeps
identity realizations equal to the uncalibrated fine grid. The reach R and the exact tap truncation
are those of each shifted template (the per-phase reaches differ by a sample at most), so every phase
is exact by the argument above. The realization blocks of Q_c and the per-realization U_c(t), V_c(t)
carry the fine grid, n_cal blocks of K·N_window entries with r = K·i + j within a block
(`ctUArrayDict_cal[det]` is (n_cal, K·N_window, n_modes, n_modes)), and the NoLoop calibration loop
gathers each block with the same stride K as the uncalibrated fine grid. Cost: K times the calibrated
Precompute; the likelihood stage is unchanged. `--calibration-fused-kernel` stays refused. Measured
(`test/test_td_gs_calibration.py`, section (g), K = 4): identity realizations reproduce the
uncalibrated K = 4 grid to 4e-15 (Q) and 1e-15 (U, V) and phase 0 is the K = 1 calibrated result bit
for bit (ringdown piece dense and GS, full segment); per-realization Q_c, U_c, V_c at every fractional
phase vs the dense brute force on the template moved by tdanalysis' kernel 1.2e-9 to 5.8e-9 (the
dense reference's precision) on the ringdown piece (dense and GS) and on two spliced pieces, while
the phases differ from one another by up to 3e-3; the calmarg loop vs the per-realization fine-grid
runs 2e-16 (nearest and cubic, weighted, lnL(t), components); each realization's lnL(t) vs the dense
residual likelihood of the calibrated shifted template 1.8e-8; the coarse-grid (phase 0) calmarg lnL
differs from the fine-grid one by 0.1 relative with nearest lookup.

**Cost** (this machine, 4 threads, relative; `test/td_gs_calibration` study scripts; IMRPhenomTPHM at
4096 Hz, 4 s, N_window = 256). The calibrated Precompute costs the uncalibrated one plus about one
uncalibrated Precompute per realization:

| case | modes | uncalibrated Precompute | per realization (self term + Q_c) | ratio |
|---|---|---|---|---|
| ringdown piece 120 M (dense, chunked) | 5 | 0.01 s | 0.01 s | 1.3 |
| inspiral 0.5 s + ringdown (dense, chunked) | 5 | 0.17 s | 0.15 s | 0.9 |
| full 4 s segment (GS, prefix) | 5 | 0.04 s | 0.02 s | 0.6 |
| ringdown piece 120 M | 21 | 0.05 s | 0.06 s | 1.3 |
| inspiral 0.5 s + ringdown | 21 | 1.04 s | 1.16 s | 1.1 |
| full 4 s segment | 21 | 0.22 s | 0.17 s | 0.8 |

So n_cal = 100 costs ~2 s (l ≤ 2) to ~20 s (l ≤ 4, full segment) or ~120 s (l ≤ 4, dense inspiral
piece) per intrinsic point on this CPU. The likelihood stage gathers the n_modes(n_modes+1) pair
series of U_c(t), V_c(t) once per realization (n_cal × the uncalibrated self-term cost), and the
4-D arrays take n_cal × N_window × n_modes² × 32 bytes per detector (180 MB per detector at n_cal =
100, 21 modes, N_window = 256). A per-realization SVD compression of U_c(t), V_c(t) is the natural
next step and is not implemented; the SVD basis of the realizations themselves (`td_calibration.py`)
is not used because the exact per-realization method is cheaper than a basis whose rank² exceeds
n_cal, and any truncated basis would be an approximation.

**Measured agreement** (`test/test_td_gs_calibration.py`): identity realizations reproduce the
uncalibrated Q, U, V to 2e-15 and the uncalibrated lnL to 3e-15; per-realization Q_c, U_c, V_c vs a
dense brute force (template filtered by exact frequency-domain multiplication with the full N-tap
response on a long zero-padded stream, then cut, dense C⁻¹) 1.2e-9 to 5.7e-9 (the dense reference's
own precision) on a ringdown piece (dense and GS solvers) and on two spliced pieces; the NoLoop
calmarg lnL vs a brute force over realizations 1e-16 (nearest and cubic, with explicit orbital phase, weighted, lnL(t), components); on a full 2 s segment with a compact signal, TD vs
RIFT's FD calmarg (same realizations) 7e-6 relative in lnL for both conventions, the level at which
the code paths agree without calibration (7e-6), while the realizations move lnL by 0.13–0.16.

## Example

```bash
export GW_SURROGATE='' PYTHONPATH=<tdanalysis wheel dir>:<repo>/MonteCarloMarginalizeCode/Code
integrate_likelihood_extrinsic_batchmode \
  -c frames.cache -C H1=FAKE-STRAIN -C L1=FAKE-STRAIN -p H1=H1-psd.xml.gz -p L1=L1-psd.xml.gz \
  -I grid.xml.gz --event 0 --event-time <GPS> --srate 4096 --fmin-template 0 --fmax 1792 \
  --l-max 4 --approximant NRSur7dq4 --time-marginalization --vectorized --gpu \
  --internal-ile-inner-product td-gs --td-mref 69 --td-piece 10,150 \
  --td-acf-file H1:H1_GW250114_acorr_4s_4096_cond1e8_cap.npy \
  --td-acf-file L1:L1_GW250114_acorr_4s_4096_cond1e8_cap.npy -o out
```

## Code

| File | Contents |
|---|---|
| `RIFT/likelihood/td_gs_precompute.py` | the code path: TD data and modes, pieces, the per-piece tdanalysis solvers and sliding forms summed into Q(t), U(t), V(t); `TDCalibrationFilters` and the calibrated `precompute` (Q_c, U_c, V_c per realization) |
| `RIFT/likelihood/factored_likelihood.py` | `PrecomputeLikelihoodTerms(..., td_gs_contexts=...)`; time-indexed U/V in the NoLoop likelihood, their SVD compression (`compress_time_indexed_cross_terms`), and the time-indexed per-realization self terms of the calmarg loop; shared Simpson rule for CPU and GPU |
| `RIFT/likelihood/td_inner_product.py` | `td-toeplitz` adapter, with `solver="gs"` (covariance kernel through tdanalysis' `ToeplitzSolver`) |
| `RIFT/likelihood/td_calibration.py` | study module: packing/convention helpers (`template_factor`, `realizations_to_positive` are used by the code path), FIR designs, SVD basis of realizations, sliding Gram methods (solves, generators and the sliding data term from tdanalysis) |
| `bin/integrate_likelihood_extrinsic_batchmode`, `bin/util_RIFT_pseudo_pipe.py` | flags |

## Tests

```bash
cd MonteCarloMarginalizeCode/Code
python -m pytest -q test/test_td_gs_route.py test/test_td_time_indexed_U.py test/test_td_u_svd.py \
    test/test_td_subsample.py test/test_simpson_weights.py test/test_td_inner_product.py \
    test/test_td_calibration.py test/test_td_gs_calibration.py
cd test/td_gs_validation && TD_GS_ACF=cond1e8_cap python -m pytest -q test_vs_reference.py
```

`test/td_gs_validation/` compares the code path with an independent dense implementation of the
residual likelihood and with tdanalysis' `GWTransientTD` (`rift_vs_tdanalysis.py --subsample K`
compares RIFT's 1/K placement with the fixed tdanalysis at each sample's own sky and fractional
placement; `--placement legacy` keeps the pre-fix whole-sample emulation for the record);
`test/td_gs_injections/` holds the injection-recovery and timing study.
`test/test_td_subsample.py` covers `--td-subsample`; `test/test_td_gs_calibration.py` covers
calibration marginalization, alone and together with `--td-subsample` (section (g)). The tests skip
without tdanalysis >= 1.1. Tests of the Toeplitz algebra itself (generators, solver, sliding forms
and filters against dense numpy/scipy) are tdanalysis' own (`tests/test_toeplitz_generators.py`,
`test_toeplitz_solver.py`, `test_sliding_forms.py`, `test_filters.py`); RIFT's tests compare its
outputs (Q, U, V, lnL) with dense brute force.

## Scope of the likelihood

- tdanalysis is an optional external package with its own licence (noncommercial use, with
  additional terms; see its `LICENSE` and <https://pypi.org/project/tdanalysis/>). Base RIFT does
  not import it; it is needed for `td-gs` and for the optional tdanalysis solvers of `td-toeplitz`.
- Disjoint pieces are independent covariance blocks: the noise correlation between samples of
  different pieces is left out. This is a modelling choice, not the covariance of all the retained
  samples taken together.
- `--td-highpass` filters the data only. It does not transform the templates or the covariance, so
  it is a conditioning approximation whose effect has to be checked for each analysis.
- `td-gs` (the finite covariance of each piece, inverted exactly) and `td-toeplitz` (the
  band-limited precision kernel on the segment) are different noise models; their posteriors need
  not agree.
- The tests compare the code path with dense reference algebra. They do not establish coverage on
  real data or agreement with frequency-domain posteriors.
- lnL is a likelihood ratio for a fixed noise model and a fixed data selection (the pieces).
  Absolute evidences from different piece selections can be compared only once the noise
  normalization of each selection is included.

## Known limitations

- Calibration marginalization runs on the loop reduction only (no fused kernel, also with
  `--td-subsample`) and without `--td-u-svd-tol`; the likelihood stage pays n_cal time-indexed
  self-term gathers per sample.
- Options of the NoLoop likelihood that the time-indexed self term does not implement are refused
  with `td-gs`: the `sinc` Q stencil (an inherited `--interpolate-time` default falls back to
  `nearest`; pass `--interpolate-time cubic` for the cubic stencil), time quadratures other than
  Simpson, `--q-time-pregrid-factor 8`, and continuous time-posterior export (the `auto` export is
  `grid`).
- Precompute runs on the CPU (numpy) even with `--gpu`; its per-shift loops dominate the cost per
  intrinsic point.
- The data are not high-passed by default. Real strain needs a high-pass before C⁻¹d when
  the ACF walls the band below f_min (the conditioned `*_cond1e8_cap` ACFs do): on GW250114
  off-source data dᵀC⁻¹d/N is 18–52 without it. A zero-phase 8th-order Butterworth at 10 Hz on
  padded data (as tdanalysis' opt-in `highpass` option) gives 0.83–0.86 and is flat to 2e-5 at
  20 Hz. Use `--td-highpass 10` (tdanalysis' zero-phase IIR Butterworth, order 8, run forward and
  backward with `sosfiltfilt`; off by default) and check whiteness (`--td-whiteness-check`) on
  off-source data.

## Orbital phase and calibration burn-in

Finite analysis pieces make the template self norm depend on orbital phase, including
for the quadrupole pair. Analytic orbital-phase marginalization is refused at startup
and by NoLoop. Sample phase explicitly; a distance lookup table must marginalize only
distance. CPU/GPU phase cases test this refusal, rather than comparing the same invalid
absolute-overlap shortcut on both backends.

Calibration burn-in evaluates the first stored realization while retaining the full
calibration buffers. With `--calibration-global-norm`, the one-block U/V length determines
the stored Q block length independently of the active `n_cal`. Gathers stay inside that
block, including at its zero-extension boundary.


### Envelope-prior handoff after the FD calibration convention update

TDGS retains its raw envelope draws: the shared generator is asked for template
envelopes, and the TD FIR applies `--td-calibration-correction-type` exactly once.
The FD routes instead convert their envelope priors before drawing. Their
`--calibration-correction-type` option is refused on TDGS; use the TD option there.
`--calibration-conjugate-phase` retains its explicit TD response conjugation and
is deprecated/ignored only on FD routes. TD pilot priors and draw growth use the
same raw envelope convention as the initial draws.
