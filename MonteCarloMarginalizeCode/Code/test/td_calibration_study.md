# Calibration uncertainty on the TD likelihood code path: study notes

Module: `RIFT/likelihood/td_calibration.py` (study module).  The production implementation is
in `RIFT/likelihood/td_gs_precompute.py` (`TDCalibrationFilters`), documented in
`docs/td_gs_route.md`, section "Calibration marginalization"; it settles the open questions
below as follows: (1) the convention is a flag (`--td-calibration-correction-type rift|bilby`,
default `rift` = RIFT's FD code path); (2) no tolerance: the N-tap inverse DFT of the realization
(unity outside the band, as RIFT defines it) truncated at the template's reach is exact on the
pieces, because the template has compact support; (3) the filtered template's extended support
is passed to the method selection of `td_gs_precompute` (prefix where valid, recursion or chunked
otherwise); (4) the per-realization method is used, no basis; (5) the taps' reach, not a fixed L,
sets the support the filter needs.
Tests and study runner: `test/test_td_calibration.py` (`pytest`; `python test_td_calibration.py
--study | --study-shifted | --study-weighted`).  Environment: anaconda python 3.12, numpy,
scipy, lal, tdanalysis 1.0.0 (`GSQuadraticForm`), run with `OMP_NUM_THREADS=2-4`.  The
timings below were measured with that version; the module now takes its GS solves, generators
and sliding data term from tdanalysis >= 1.1 (`ToeplitzSolver`, `gs_generators_from_acorr`,
`sliding_data_term`), whose GS setup needs no dense inverse.

The TD code path analyses fixed sample selections ("pieces", `[i0, i1)` of one stream) with
`C_W = Toeplitz(ACF)` per piece, and uses no FFT, window or taper on data or templates.
Calibration is applied to the TEMPLATE as a real FIR filter, calibrate-then-cut:
`(K_c h)[n] = sum_{j=-L}^{L} c_c[j] h[n-j]` on `[i0-L, i1+L)` by direct convolution, then
cut.  The data are untouched; `w = C_W^{-1} d_W` once.  `Q_c = (K_c h)_W^H w`,
`U_c = (K_c h)_W^H C_W^{-1} (K_c h)_W`.

## 1. How RIFT builds calibration realizations, and the convention

`RIFT/calmarg/generate_realizations.py::create_realizations(fname, T, dT, fmin, fmax,
n_spline, n_real)` (called from `integrate_likelihood_extrinsic_batchmode` with
`--calibration-envelope-directory/<ifo>.txt`, `--calibration-n-realizations`,
`--calibration-spline-count`):

* Envelope file columns: `f, median_mag, median_phase, 16_mag, 16_phase, 84_mag, 84_phase`
  (bilby format); sigma = (84 - 16)/2; nodes at `n_spline` log-spaced frequencies in
  `[fmin, fmax]`; amplitude and phase node values drawn independently from
  `N(median, sigma)`; cubic splines in `log10 f` (`scipy CubicSpline`).
* Output `dat_out`, shape `(npts_seg, n_cal)` complex with `npts_seg = T/dT`, in RIFT's
  two-sided packing `f_R[k] = deltaF (npts/2 - k)` (`lalsimutils.evaluate_fvals` default:
  index 0 is `+fNyq`, index `npts/2` is DC, then negative frequencies).  Unity outside
  `[fmin, fmax]`.
* Negative half: `C(-f) = A(f) exp(-i phi(f)) = conj C(f)` -- Hermitian, so a real FIR
  exists (`is_hermitian_rift` checks it; true for all draws).
* `factored_likelihood.ComputeModeIPTimeSeries` applies realization `c` to the DATA:
  `data_now = C_R * data` (default) or `conj(C_R) * data` (`--calibration-conjugate-phase`),
  then `kappa = IP.ip(h, data_now)` with `ComplexIP.ip = 2 deltaF sum conj(h_R) d_R / S(|f|)`.

Derivation of the template-side equivalent.  Let `x_S(f) = int x(t) e^{-2 pi i f t} dt`
(standard, numpy fft) and `x_R(f) = int x(t) e^{+2 pi i f t} dt`.  Checked numerically:
`lalsimutils.DataFourier` of `e^{+2 pi i f0 t}` peaks at `evaluate_fvals = -f0`, so RIFT's
bin labelled `+f` holds `x_S(-f)`: `x_R(f) = x_S(-f)` for data and (consistently, or the
matched filter would not work) templates.  With the realization placed at face value on
RIFT's labels, `C_R(f) = A(|f|) e^{i sgn(f) phi(|f|)} =: C(f)`,

    kappa_RIFT = sum_f conj(h_R(f)) C(f) d_R(f) / S
               = sum_f conj(h_S(-f)) C(f) d_S(-f) / S          (x_R(f) = x_S(-f))
               = sum_f conj(h_S(f)) C(-f) d_S(f) / S           (relabel f -> -f)
               = sum_f conj(h_S(f)) conj(C(f)) d_S(f) / S      (Hermitian)
               = sum_f conj(C(f) h_S(f)) d_S(f) / S
               = <C h | d>_S  =  conj( <d | C h>_S ).

So RIFT's default, in the standard convention, is the TEMPLATE multiplied by `C` at face
value in the conjugated slot; its real part and modulus equal bilby's template-side
`<d | C h>`.  Numerical check (`test_convention_matches_rift_default`, white noise, long
segment, `d = C h` built by FFT in the standard convention): `lsu.ComplexIP` with `C_R` on
the data = `1.97889495`, TD code path with taps from `C` = same to `< 2e-3` (FIR tolerance);
`conj(C_R)` on the data = `1.93913721` = TD taps from `conj C`.  With `d = C h` the
face-value filter gives `Q = U` (template matches the data); the conjugate does not.

Conclusion:

    TD template filter with response C_c(f) (face value, standard e^{-2 pi i f t} packing)
        ==  RIFT default (C_R on the data);
    TD filter with response conj C_c(f)
        ==  RIFT --calibration-conjugate-phase.

The physically correct model is `d = C h + n` with `C` the factor mapping theoretical
strain to calibrated data.  Bilby's envelope semantics (`bilby.gw.detector.calibration`,
`RIFT/calmarg/calibration.py::correction_type_for_ifo`): correction type `template` (V1)
means the envelope factor `eta` IS that map, `C = eta`; type `data` (H1, L1, K1 by
bilby_pipe default) means `eta` maps calibrated data to theoretical strain, `h = eta d`,
so `C = 1/eta` (bilby inverts the draws).  RIFT's generator uses `eta` at face value, so
RIFT's default equals bilby for V1-type envelopes; for H1/L1 (`data` type) the template
factor should be `1/eta` ~ `(1 - dA) e^{-i dphi}`: both the amplitude deviation and the
phase have the opposite sign to RIFT's default at first order, and the conjugate option
flips only the phase.  `template_factor(cal, correction_type, conjugate)` provides all of
these; which to use for a given envelope file is an input decision (open question 1).

## 2. FIR design and half-length L (fs = 4096 Hz, T = 4 s grid, band [20, 2000] Hz)

Synthetic envelope (no real envelope files on this machine): 10 log-spaced nodes
20-2048 Hz, sigma_amp 10% -> 5%, sigma_phase 0.15 -> 0.08 rad (8.6 -> 4.6 deg), medians
`1 + 0.02 sin(ln f)`, `0.03 cos(ln f)`; realizations drawn by RIFT's own
`create_realizations`.

Designs (all real taps, no FFT in the module; taps from the direct inverse DFT sum or
from normal equations):

* `dft`: window method = truncated inverse DFT of the two-sided response, with the
  out-of-band values either RIFT's (`extension="unity"`, jumps at the band edges) or a
  continuous continuation (`"hold"`: edge amplitude held, phase run linearly to 0 at DC
  and Nyquist).  Optional coefficient taper (Kaiser).
* `ls`: weighted least squares on the in-band grid only (Toeplitz normal equations +
  Levinson, 0.1-2 s per design); ridge `1e-3` because the unconstrained fit reached
  out-of-band gains of `6e4` at L ~ 400 (a template's 0.4% out-of-band leakage would be
  amplified beyond the signal).  With the ridge the out-of-band gain stays ~1.

Half-length by the MAX relative in-band error (5 realizations, median [min-max]):

| method | tol | L | err |
|---|---|---|---|
| ls | 1e-2 | 206 [161-313] | 9.9e-03 |
| ls | 1e-3 | 286 [237-418] | 1.0e-03 |
| ls | 1e-4 | 2471 [2468-3315] | 1.0e-04 |
| dft hold | 1e-2 | 1211 [1037-1981] | 1.0e-02 |
| dft hold | 1e-3 | 6122 [5826-6909] | 1.0e-03 |
| dft hold | 1e-4 | 7962 [7944-8000] (cap 8000) | 1.5e-04 |
| dft hold + Kaiser taper (beta 4) | 1e-2 | 2359 | 1.0e-02 |
| dft hold + Kaiser taper | 1e-3, 1e-4 | 8000 (cap) | 2.5e-03 |
| dft unity (RIFT's out-of-band values) | 1e-2 | 7170 | 1.0e-02 |

The max error is set by the 20-30 Hz band edge: the first spline interval (20-33.6 Hz in
log spacing) is the sharpest feature, and the jump/kink at `fmin` dominates the window
designs (error by sub-band for an L=387 fit: 1.4e-2 at 20-30 Hz, 2.9e-3 at 30-60 Hz,
3e-4 at 200-1000 Hz, 7e-5 at 1-2 kHz).  Coefficient tapering does not help the in-band
max error.

What matters for the likelihood is the error in Q and U, which are noise-weighted.
Against the FFT-multiplication truth (C with RIFT's unity outside the band, applied on the
full 4 s stream, then cut; 10 realizations; GW250114-like PSD; inspiral piece 2 s + ringdown
piece 182 samples):

| design | L | max in-band err | 1/S-weighted rms err | max dQ/Q vs truth | max dU/U vs truth | design time / realization |
|---|---|---|---|---|---|---|
| ls ridge 1e-3 | 298 | 5.7e-04 | 4.9e-05 | 1.1e-03 | 4.1e-05 | 0.12 s |
| dft unity | 298 | 1.1e-01 | 2.1e-04 | 3.0e-04 | 2.6e-04 | 0.08 s |
| dft hold | 298 | 3.5e-02 | 6.3e-05 | 2.3e-04 | 2.9e-04 | 0.08 s |
| ls ridge 1e-3 | 1211 | 1.3e-04 | 2.3e-05 | 8.4e-04 | 2.2e-05 | 0.49 s |
| dft unity | 1211 | 7.4e-02 | 5.1e-05 | 2.6e-05 | 2.2e-05 | 0.32 s |
| dft hold | 1211 | 8.7e-03 | 7.2e-06 | 3.1e-04 | 1.1e-05 | 0.28 s |
| ls ridge 1e-3 | 2471 | 9.9e-05 | 1.7e-05 | 8.0e-04 | 1.8e-05 | 0.88 s |
| dft unity | 2471 | 5.8e-02 | 2.1e-05 | 1.1e-05 | 6.0e-06 | 0.58 s |
| dft hold | 2471 | 4.0e-03 | 2.4e-06 | 3.1e-04 | 2.8e-06 | 0.57 s |

Findings:

* U converges with L for every design (to ~1e-5 at L >= 1211); the 1/S-weighted rms error
  tracks the U error.
* Q has a floor (~1e-3 for `ls`, 3e-4 for `dft hold`) independent of L: it comes from the
  out-of-band response, where these designs differ from the truth's unity and the whitened
  data still carry some noise power at f < 20 Hz.  The plain truncated inverse DFT of
  RIFT's realization (`dft unity`) -- whose max in-band error is poor because of Gibbs
  ringing within a few bins of 20 Hz, where neither template nor whitened data has power --
  converges in Q as well: 3e-4, 2.6e-5, 1.1e-5 at L = 298, 1211, 2471.
* Hence the recommendation: taps = truncated inverse DFT of the realization as RIFT
  defines it (`fir_taps(..., method="dft", extension="unity")`), L chosen by the
  1/S-weighted rms criterion (`choose_half_length(..., criterion="wrms", weight=1/S)`):

| wrms tol | L (median [min-max], 5 realizations) | max in-band err at that L |
|---|---|---|
| 1e-2 | 14 [5-47] | 2.4e-01 |
| 1e-3 | 205 [52-419] | 1.5e-01 |
| 1e-4 | 1872 [547-3533] | 8.8e-02 |

  For a 1e-3 target on Q and U at 4096 Hz, L ~ 200-400 (0.05-0.1 s of margin on each side
  of a piece); for 1e-4, L ~ 2000-3500.  The spread across realizations is wide, so L
  should be chosen per run from the drawn realizations (max over them), or fixed at a
  conservative value.

FIR cost: one pass of a complex template over the pieces costs `2 (2L+1) sum_pieces n`
real multiply-adds: `1.0e7` at L = 298 for the 8192 + 182 sample pieces (~10 ms in numpy).

## 3. Low-rank basis of realizations

`realization_basis`: SVD of the real matrix `[Re C; Im C]` over the positive band so the
coefficients `a_ck` are REAL (a complex combination of Hermitian responses would not be
Hermitian); rank = first r with `max_c max_f |C_c - sum_k a_ck phi_k| <= tol` (scanned
upward with rank-one residual updates; the max-norm error is not monotone in r).

| n_cal | tol | r | err |
|---|---|---|---|
| 100 | 1e-2 | 39 | 1.3e-03 |
| 100 | 1e-3 | 45 | 8.4e-04 |
| 100 | 1e-4 | 58 | 5.1e-05 |
| 1000 | 1e-2 | 39 | 4.7e-03 |
| 1000 | 1e-3 | 57 | 2.7e-04 |
| 1000 | 1e-4 | 70 | 8.2e-05 |

The rank is well above the 20 node parameters because `C = A e^{i phi}` is nonlinear in
the node values (amplitude-phase products and `phi^2/2 ~ 0.01` at sigma_phase 0.15 rad
are at the 1e-2 level); it saturates near 40 at 1e-2 for both n_cal.  Relative to n_cal
the saving is 14-25x at n_cal = 1000 and ~2x at n_cal = 100.

## 4. Accuracy and timing of the basis Q_c, U_c (unshifted)

Pieces: inspiral `[3892, 12084)` (8192 samples = 2 s, ending 50 ms before merger),
ringdown `[12296, 12478)` (182 samples = 150 M at 60 Msun, from 2 ms after merger);
template: smooth analytic chirp (25 Hz to 150 Hz over 2.8 s, amplitude ~ f^{2/3}) joined
to a damped sinusoid at 250 Hz, tau = M/0.09 = 3.3 ms; ACF from the H1 GW250114 PSD of the
tdanalysis study (walled outside [20, 2000] Hz, cosine sum, no FFT); solver tdanalysis GS
(prepare once per piece length: 8 s at n = 8192, dense inverse inside tdanalysis);
`ls` taps at the max-criterion L; basis vs exact on 100 realizations.

| n_cal | tol | L | r | max dQ/Q basis vs exact | max dU/U basis vs exact | t_exact [s] (+ design) | t_basis [s] (+ design, once) |
|---|---|---|---|---|---|---|---|
| 100 | 1e-2 | 214 | 39 | 3.1e-05 | 6.0e-05 | 0.23 (+8.1) | 0.09 (+3.1) |
| 100 | 1e-3 | 298 | 45 | 1.3e-05 | 1.8e-05 | 0.23 (+10.7) | 0.12 (+4.8) |
| 100 | 1e-4 | 2471 | 58 | 5.0e-07 | 7.5e-07 | 1.41 (+99.9) | 0.93 (+61.3) |
| 1000 | 1e-2 | 214 | 39 | 5.2e-05 | 6.7e-05 | 3.53 (+124) | 0.18 (+4.2) |
| 1000 | 1e-3 | 298 | 57 | 1.8e-06 | 3.2e-06 | 7.04 (+209) | 0.21 (+12.2) |
| 1000 | 1e-4 | 2471 | 70 | 2.1e-07 | 2.3e-07 | 16.9 (+1229) | 1.44 (+101) |

(`t_exact` for n_cal = 1000 scaled from 100 realizations.)  The basis error in Q and U
is 1-2 orders of magnitude below the realization tolerance, since the error is
noise-weighted.  Per intrinsic point the basis method costs r FIR passes + r GS solves +
an r x r Gram: 0.1-0.2 s at r ~ 40-60 for these pieces against 0.2-17 s for the exact
method; the design of the r basis filters is done once per run.  Memory: `G` is r x r
complex per mode pair.

Note on the raw ACF file `H1_GW250114_acorr_4s_4096.npy`: unwalled, its Toeplitz is
numerically singular at these lengths (eigenvalues ~1e-13 of the largest at n = 1024);
Levinson, Cholesky and GS all fail the residual check (GS by 1e6).  With the wall the
condition number is ~1e10 and the three agree to ~1e-7; the TD code path's `gs_acf` applies
the same wall.

## 5. Time-indexed terms: Q_c(t) and U_c(t) = a_c^T G(t) a_c

With the time-marginalization shift `(S_t g)[n] = g[n - t]`, the FIR commutes with the
shift, so "calibrate, shift, cut" is a slice of the filtered full stream: `g_k = K_k h`
is computed once on `[i0 - t_max, i1 - t_min)`.  Then

    q_k(t)    = (S_t g_k)_W^H w_W                      (direct correlation, O(n n_shift) per k)
    G_kl(t)   = (S_t g_k)_W^H C_W^{-1} (S_t g_l)_W
    Q_c(t)    = sum_k a_ck q_k(t),     U_c(t) = a_c^T Re G(t) a_c.

`G(t)` through the Gohberg-Semencul generator method (`toeplitz_generator`,
`gs_kernel_streams`, `sliding_gram`): with `x = C^{-1} e_0`, `x~ = (0, x_{n-1}, ..., x_1)`,

    C^{-1} = (1/x0) [ L(x) L(x)^T - L(x~) L(x~)^T ],
    u^H C^{-1} v = (1/x0) [ <L(x)^T u, L(x)^T v> - <L(x~)^T u, L(x~)^T v> ],

`L(.)^T` being an anti-causal correlation.  The kernel streams `Y_k = L(x)^T g_k`,
`Y~_k = L(x~)^T g_k` are computed once per basis template on the extended range (two
direct correlations with length-n kernels: 0.01 s per template on the 1 s piece), and
`G_kl(t) = (1/x0)[P_kl(i1 - t) - P_kl(i0 - t)]` with `P` the prefix sum of
`conj(Y_k) Y_l - conj(Y~_k) Y~_l` -- O(r^2 n) for all shifts.  This is exact when every
shifted template ENDS inside the piece (the anti-causal kernels never see the piece start
nor anything past the template end).  By persymmetry `J C J = C` the same identity with
causal kernels (`side="start"`) is exact when the shifted template STARTS inside the piece.
A template crossing both boundaries is exact in neither; `piece_sides[ip] = "exact"` then
solves per shift (r solves per shift), cheap for short pieces.  (`gram_fn` / `q_fn`
callables let the Precompute code path for U(t) replace these.)  The generator of the GW-like
ACF does NOT decay (|x| > 1e-2 of its peak out to 2 s: narrow spectral lines), so the
kernel correlations cannot be shortened.

Measured (n_cal = 100, tol 1e-3, r = 45, L = 298; inspiral piece 1 s with the chirp
starting inside it, ringdown piece 182 samples; shifts +-204 samples = +-50 ms, 409
values; reference = per-shift, per-realization solves on 5 shifts x 10 realizations):

| sides | max dQ(t)/Q | max dU(t)/U | t_basis, all 409 shifts and 100 realizations | per-shift exact, per realization and shift | G(t) memory |
|---|---|---|---|---|---|
| start + exact | 1.9e-04 | 3.1e-05 | 2.6 s | 0.039 s | 13 MB |
| start + end | 1.9e-04 | 2.0e-02 | 0.7 s | 0.038 s | 13 MB |

With `end` on the ringdown piece the +50 ms shifts move the merger past the 44 ms piece,
the template no longer ends inside it, and U(t) is off by 2%: the one-sided condition has
to hold for every shift.  The per-shift exact method for all 100 realizations and 409 shifts
would take ~1600 s (one Levinson solve each at n = 4096); the basis method takes 2.6 s.

Cost per intrinsic point, time-indexed: r FIR passes on the extended range, 2r kernel
correlations per generator-method piece (O(n^2) direct, or the FFT-based GS apply that the
TD code path already accepts), r direct correlations for q_k(t) (O(n n_shift)), r^2 n prefix
sums, and for an `exact` piece r solves per shift.  Memory: `n_shift x r x r` complex per
mode pair (13 MB at 409 x 45 x 45).

## 6. Module API (`RIFT.likelihood.td_calibration`)

* Packing / conventions: `rift_packing_frequencies(npts, deltaF)`,
  `realizations_to_positive(cal2side)`, `is_hermitian_rift(cal2side)`,
  `positive_to_standard(C_pos)`, `template_factor(cal2side, correction_type, conjugate)`,
  `smooth_out_of_band(C_pos, f_pos, fmin, fmax, mode)`.
* FIR design: `fir_taps_dft(C_pos, L)` (direct inverse DFT sum), `fir_taps_ls(C_pos, f_pos,
  fs, L, fmin, fmax, weight, n_grid, ridge)` (Toeplitz normal equations), `taper_taps`,
  `fir_taps(..., method, taper, extension)`, `fir_response(taps, f, fs)`,
  `fir_band_error(..., return_gain, weight)`, `choose_half_length(..., tol, method,
  criterion="max"|"wrms", weight, gain_max)`, `fir_cost_flops(pieces, L)`.
* Filtering: `apply_fir_piece(h_full, taps, i0, i1)` (np.convolve 'valid' on the margined
  segment: direct linear convolution, cut to the piece), `apply_fir_piece_reference`.
* Basis: `realization_basis(C_pos_all, tol, f_pos, fmin, fmax)` -> `phi (nf, r)`, `a
  (n_cal, r)` real, `s`, `r`, `err_max`; `basis_taps(phi, f_pos, fs, L, fmin, fmax)`.
* Terms: `data_weights(d, pieces, solver)`, `cal_terms_exact(h, pieces, taps_all, solver,
  w)`, `cal_terms_basis(h, pieces, taps_basis, a, solver, w)`; time-indexed:
  `toeplitz_generator`, `gs_dense_inverse`, `anticausal_correlation`, `causal_convolution`,
  `gs_kernel_streams(..., side)`, `sliding_gram`, `sliding_data_term`,
  `cal_terms_basis_shifted(h, pieces, taps_basis, a, shifts, w, acf, gram_fn, q_fn,
  piece_sides, solver)`, `cal_terms_exact_shifted`.
* Solvers (study aid): `toeplitz_piece_solver(acf, pieces, backend="dense"|"gs")` ->
  `solver(v, ipiece)` (Levinson, or tdanalysis GS).

## 7. Tests (`pytest test/test_td_calibration.py`, 11 passed, ~60-130 s)

packing/Hermitian symmetry; template-factor conventions; FIR piece filtering == explicit
tap loop and == slice of the full filtered stream; FIR == FFT multiplication by its own
response (1e-10) and by C(f) to the tolerance in band; chosen L meets tol; basis vs exact
Q, U (ringdown + inspiral pieces, GW-like ACF); the convention test of section 1; GS
backend == dense; generator identity vs dense inverse; sliding Gram == per-shift quadratic
forms ('end' and 'start', and the wrong side is not exact); time-indexed basis vs
per-shift reference, 'exact' mode, and shift-0 consistency.

## 8. Open questions

1. Envelope semantics per detector: RIFT's in-loop calmarg uses the envelope factor at
   face value for every IFO.  Under bilby_pipe's defaults H1/L1/K1 envelopes are of
   "data" type, for which the template factor is `1/eta` -- opposite first-order sign in
   both amplitude and phase to RIFT's default (the conjugate option flips only the phase).
   Which `correction_type` should the TD code path (and RIFT's FD calmarg) use per detector?
2. Out-of-band behaviour of the filter: RIFT's realizations are unity outside
   `[fmin, fmax]`, and the TD code path has no band mask, so the filter's out-of-band response
   enters Q at the ~1e-3 level through template leakage times whitened-noise power below
   20 Hz.  Is "unity outside the band" (RIFT's convention, reproduced by the truncated
   inverse DFT) the intended model, and is a 1e-3 to 1e-4 target on Q/U per realization
   the right accuracy goal?  If so, L ~ 200-400 (1e-3) or ~2000-3500 (1e-4) at 4096 Hz.
3. Pieces crossed on both sides by the template (long inspirals cut by a piece of a few
   seconds): the generator prefix-sum method for U(t) is one-sided; the fallback is per-shift
   solves (r per shift) or a piece layout that keeps one boundary clear.  How does the
   Precompute code path for the uncalibrated U(t) handle this case?
4. Rank: 40-70 basis filters at 1e-2 to 1e-4 (nonlinearity of `A e^{i phi}` in the node
   values).  A smaller `--calibration-spline-count` or a tighter envelope lowers it; is the
   per-intrinsic-point budget (r FIR passes + r solves, ~0.1-1 s here) acceptable, or
   should the self term use a coarser basis than the data term?
5. The FIR design and the basis are per detector and per run (realizations are drawn
   once); the L_max margin must be reflected in the template-generation range
   (`[i0 - L - t_max, i1 + L - t_min)`).
