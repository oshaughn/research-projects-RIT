# Slow-rotation RIFT likelihood — handoff / breadcrumb

Branch: `rift_slowrot` (off `rift_O4d_fix_rvs_clear_fairdraw_batch`). Design notes + LaTeX
paper live in a **separate local repo** `~/rift-slow-rotation` (no git remote — local only;
21-page PDF via `latexmk -pdf notes/main.tex`).

## PATH B DELAY PHYSICS — VALIDATED (2026-07 update)
Path B's propagation-delay drift is validated against an INDEPENDENT folded-template ground
truth (`test_slowrot_pathB_bruteforce.py`): data built as Re[F(t) Sigma(t-tau(t))] from the
SAME harmonic F(t),tau(t) model + the SAME modes the likelihood uses (no SimDetectorStrain
convention floor), at an inflated sidereal rate so the delay drift is large.  At the REAL
90-min-BNS rate (= x340 inflation on a 16s test): p_max=0 deficit 3.43 -> p_max=1 0.23 ->
p_max=2 0.207 -> p_max=3 0.207: CONVERGES, bound-respected, NO blow-up.  So Path B recovers
the delay drift and is production-ready for the target signals with p_max<=2.
KNOWN LIMIT: the p>=3 catastrophic cancellation (huge high-f U terms x tiny delta_tau^p
coefficients) only bites at x1000+ inflation (>2.6x faster than any physical signal): x1000
gives p=2 deficit 5.9 but p=3 blows to 1e5.  So the band-limit fix (low-pass the p>=1
derivative templates) is a robustness nicety, NOT a blocker for real signals.
Separately validated vs LAL's SimDetectorStrainREAL8TimeSeries (`test_slowrot_pathB_groundtruth.py`):
baseline/PathA/PathB all agree with Jolien's full delay map to ~0.07 at fmax=256 (the ~26
deficit at fmax=1024 was SimDetectorStrain's high-f TD delay-INTERPOLATION, not a bug --
confirmed: it vanishes when fmax is lowered).  NoLoop uses nearest-neighbor time sampling
(factored_likelihood.py ~L1691), so absolute peak comparisons at high SNR have a resolution
floor (~0.1-0.2); the time-interpolated NoLoop is in oshaughn/rift_O4d and
origin/rift_O4d_junior_calmarg_in_loop if a cleaner absolute comparison is wanted.

## TWO PARALLEL THRUSTS (2026-07)
Path A + Path B are implemented, validated, and wired into the ILE.  Next work is two tracks:
1. **Rotation PE value demo (near-term).**  VERIFY-ANYWHERE quick-look DONE
   (`~/RIFT_roboto_paper/analyses/slowrot_demo/`, `make demo` / `demo_local.py`): rotation-vs-
   static lnL gain grows with Omega*T (null 0.004 -> 64s 0.19 at fixed SNR~30); figure
   `outputs/gain_vs_duration.png`.  REMAINING: the full DAG injection-recovery PE (posterior
   bias + single-network sky map) on the cluster -> `paper/` figure cited from `sec:slowrot`
   (structured Makefile targets inject/baseline/rotationA/rotationB/compare).
2. **Frequency-dependent (finite-size) response = Path D (3G regime where 'long' lives).**
   RESPONSE FUNCTION IMPLEMENTED + VALIDATED: `slowrot_freqresponse.py` +
   `test_slowrot_freqresponse.py`.  F_k(f;RA,DEC,psi) from arXiv:2412.01693 single-arm sinc
   transfer; f->0 == ComputeDetAMResponse to 6.7e-16; exact FSR null at c/2L=3747 Hz (40 km);
   in-band SHAPE distortion 0.24%@1kHz/0.62%@2kHz (LIGO) vs 11.8%/42% (CE) -> negligible for
   LIGO, first-order only for 3G.  Key: the complex ratio ~1 for CE is dominated by a benign
   common e^{-i2pi fL/c} delay (degenerate with tc), NOT sky shape.  LAL has NO closed-form FD
   response.  ROUTE DECISION (notes/sec_freqdep.tex): CE -> route (b) sky-harmonic expansion
   (keeps sky extrinsic; fold the common delay into geocenter time; few angular orders);
   precessing+HM -> route (a) pinned-sky TD fold.  REMAINING: precompute integration (fold
   T_p(f) into the mode inner products) + validation vs a direct finite-size injection.

## What this is
Generalizes RIFT's marginalized likelihood to account for the **time dependence of the
ground-based detector response over the signal** (Earth rotation), while reusing the
precompute-and-marginalize architecture. Two effects, both implemented (Path A + Path B):
- **Path A** — antenna amplitude drift `F(t)`: exact 5 sidereal harmonics, sky extrinsic.
- **Path B** — propagation-delay drift `tau(t)`: time-domain derivative expansion; adds
  delay-derivative order `p` on top of Path A. Matters only for LONG signals.
- **Deferred (Path D)** — frequency-dependent (finite-size) response. Out of scope.

## Files (all under RIFT/likelihood/ unless noted)
- `slowrot_response.py` — closed-form antenna harmonics `A_n` and delay harmonics `B_n`
  (scalar + `*_vector`), derived from `vectorized_lal_tools` conventions. Validated vs LAL
  to machine precision (`test_slowrot_response.py`).
- `factored_likelihood_with_rotation.py` — the core:
  - `PrecomputeLikelihoodTermsWithRotation(...)` — FD-native precompute: builds each mode
    once, applies derivative `(FT_SIGN*2pi i f)^p` (FT_SIGN=-1) and sidereal modulation
    `exp(i n Omega t)` (sub-bin freq shift via LAL-FFT phase). Returns bank keyed by
    elementary template `a=(p,n)`: `Q^a(t)`, `U^{(a,a')}`, `V^{(a,a')}`.
  - `rotation_coefficients` / `rotation_coefficients_vector` — the analytic scalars
    `C_{(p,ntilde)} = (1/p!) sum_{n+m=ntilde} A_tilde_n [(-D)^{*p}]_m` (Path A: `{(0,n): A_tilde_n}`).
  - `FactoredLogLikelihoodWithRotation(...)` — scalar lnL (per-sample); term1 = `Re[sum_lm
    conj(Ylm) sum_a conj(C_a) Q^a(t_det)]`, term2 with `U^{(a,a')}` (coef `conj(C_a)C_a'`)
    and `V^{(a,a')}` (coef `C_{(p,-nu)} C_a'`).
  - `pack_rotation_arrays` + `DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation`
    — the MAINTAINED vectorized (NoLoop) path, mirroring
    `factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopOrig`.
- `factored_likelihood.py` — one small FIX: non-numba `lalylm` fallback -> `np.vectorize`
  (it broke `ComputeYlmsArrayVector` for the whole vectorized path in non-numba/non-GPU envs).
- `bin/integrate_likelihood_extrinsic_batchmode` — wiring: `--rotation-slow`,
  `--rotation-n-harmonics`, `--rotation-p-max`. Guarded (requires `--vectorized`, CPU, no
  dist/phase marg). Builds+packs the rotation bank after the baseline pack; the CPU
  vectorized `likelihood_function` calls the rotation NoLoop.

## How to run (RIFT venv; worktree first on PYTHONPATH)
    source ~/RIFT_develUWM/bin/activate
    export PYTHONPATH=~/RIFT_slowrot/MonteCarloMarginalizeCode/Code
    python RIFT/likelihood/test_slowrot_response.py            # A_n,B_n vs LAL
    python RIFT/likelihood/test_slowrot_fd_ops.py              # FD derivative/modulation
    python RIFT/likelihood/test_slowrot_likelihood_v1.py       # scalar Path A vs baseline + brute force
    python RIFT/likelihood/test_slowrot_noloop.py              # vectorized Path A vs baseline NoLoop
    python RIFT/likelihood/test_slowrot_noloop_bruteforce.py   # vectorized Path A vs brute force
    python RIFT/likelihood/test_slowrot_pathB.py               # Path B reduction + bound

End-to-end ILE head-to-head (ILE-GPU-Paper demo data), baseline vs rotation:
    D=~/RIFT_develUWM/src/research-projects-RIT/.travis/ILE-GPU-Paper/demos
    integrate_likelihood_extrinsic_batchmode --vectorized [std opts] ...            # baseline
    integrate_likelihood_extrinsic_batchmode --vectorized --rotation-slow ...        # Path A
    integrate_likelihood_extrinsic_batchmode --vectorized --rotation-slow --rotation-p-max 1 ...  # Path B

## Validation status (all PASSING)
- Response harmonics vs LAL: ~1e-16.  FD ops vs LAL round trips: ~1e-13.
- Path A scalar: V1a (Omega=0 vs baseline) 2.7e-12; V1b (real vs brute force) 2.6e-9.
- Path A vectorized: vs baseline NoLoop 3.6e-12; vs brute force 3.2e-10; V0 (precompute
  recovery on real data) exact.
- Path B: scalar reduce-to-baseline 9e-13; respects 0.5<d|d>; vectorized reduce 6.4e-12.
- ILE runs clean (no inf/nan) for baseline / Path A / Path B; rotation ~ baseline for the
  short demo signal (rotation effect ~0.008 in lnL, negligible as physics requires).

## CRITICAL lesson (a bug that hid for a while)
The U/V template modulation MUST be referenced to the template's INTRINSIC time (epoch ~0),
not the absolute event time t_ev~1e9 -- else a ~1e4 rad spurious phase randomizes U/V and
inflates lnL ABOVE 0.5<d|d>. This bug once made a short-BBH rotation shift look like ~660 in
lnL; corrected it is ~0.008. My independent "brute force" reference shared the same
convention, so `vec == brute-force` PASSED while both were wrong. **Always cross-check
against the Cauchy-Schwarz bound 0.5<d|d>, not only against a reference that can share
conventions.**

## PATH B STATUS (findings 2026-07-04, the systematic pass in progress)
- Matched-seed head-to-head DONE (test_slowrot_headtohead.py): rot(f_sid=0)==baseline 9e-13;
  evidence shift ln Z_rot - ln Z_base = -1.1e-3 (MC-noise-free) for the short signal.
- Path B DELAY PHYSICS still only partially validated. In an INFLATED-Omega regime (f_sidereal
  x3000, single-det 30+25 BBH) where the delay drift matters, scalar Path B gives
  lnL(p_max=0..3) = [1794.39, 1781.31, 1781.63, 928.30]: p=0->1->2 captures a real ~13-in-lnL
  delay effect and appears to converge (~1781.6) and respects 0.5<d|d>=1938 -- BUT p_max=3
  BLOWS UP (increment 853).
- ROOT CAUSE of the p>=3 blow-up (likely): the FD derivative weight (2 pi i f)^p amplifies high
  frequencies; in the model norm the integrand ~ (2 pi f)^{2p} |h(f)|^2 / S grows like f^{11/3}
  for a chirp (|h|^2 ~ f^{-7/3}), so high-order terms are dominated by the f_max edge, not the
  physical low-frequency delay drift.  FIX for the systematic pass: BAND-LIMIT the delay-
  derivative (p>=1) terms to low frequency (the delay drift physically matters in the long early
  inspiral, i.e. low f, not at merger).  Options: apply a low-pass / taper before the (2 pi i f)^p
  weight, or use a reduced f_max for p>=1 templates, or work with dimensionless (f/f_ref) weights.
- Consequence: current Path B is trustworthy only through p_max<=2 and only after the band-limit
  fix is validated.  Do NOT ship Path B (p>=1) until this is fixed AND checked against an
  independent time-varying-delay brute force (below).

## OPEN / NEXT (the one remaining systematic pass — do it all together)
1. **Path B rigorous validation** (TWO parts, do together):
   (a) FIX the p>=3 high-frequency derivative blow-up by band-limiting the delay-derivative
       terms to low frequency (see PATH B STATUS above); re-check convergence is monotone.
   (b) Validate vs an INDEPENDENT ground truth that uses LAL's OWN full delay-time map --
       lalsim.SimDetectorStrainREAL8TimeSeries (Jolien's code).  KEY FACTS FOUND 2026-07-04:
       - RIFT's data ALREADY uses it: non_herm_hoff -> hoft -> SimInspiralTD +
         SimDetectorStrainREAL8TimeSeries (lalsimutils.py ~line 3020).  So a LONG injection
         already carries the real Earth-rotation response; baseline (static) will under-recover
         it and Path A/B should recover it.
       - GROUND TRUTH for the rotation likelihood must apply SimDetectorStrain to the SAME
         RIFT modes the likelihood uses:  hk = lsu.hoft_from_hlm(hlmsT, P_extr)  (radec path,
         applies SimDetectorStrain).  Then convert to 2-sided FD exactly like non_herm_hoff
         (copy REAL8 -> COMPLEX16, COMPLEX16TimeFreqFFT) and lnL = ComplexIP(d,hk).real -
         0.5*ComplexIP(hk,hk).real.
       - DO NOT use non_herm_hoff(P_extr) as the template ground truth: it regenerates via
         SimInspiralTD, whose modes differ from RIFT's hlmoff modes -- with IMRPhenomD the
         mode-based baseline recovers only ~SNR 35 of the data's ~SNR 82 at the injection (a
         waveform-convention mismatch, NOT rotation).  Using hoft_from_hlm keeps the modes
         identical so ONLY the rotation differs.
       - Two gotchas to fix: (i) hoft_from_hlm output length = mode length (e.g. 16451) !=
         padded data length (16384) -> ResizeREAL8TimeSeries + align epoch (replicate
         non_herm_hoff padding); (ii) the mode-based likelihood at a FIXED tref is OFF-PEAK
         (epoch bookkeeping) -> compare at the time-MAXIMIZED / marginalized value, not at
         extr.tref.
       - Use a LONG signal (real sidereal rate) so delay drift matters, OR the inflated-Omega
         path (already shows p=1,2 capture the delay, p>=3 needs the band-limit fix).
       A convergence test alone is NOT enough: Path B could converge to a WRONG value if the C
       coefficients are off (the delay analogue of the reference-time bug).  Cross-check 0.5<d|d>.
2. **Matched-seed quantitative ILE head-to-head** (fixed RNG seed) -> exact, regression-grade
   baseline-vs-rotation comparison (currently only MC-noise-level agreement).
3. Cauchy-Schwarz bound checks everywhere; sweep for other shared-convention bugs.
4. Possible follow-ups: GPU (xpy) path for the rotation NoLoop; distance/phase/cal marg
   support; Path B `t_star` reference-epoch choice; `rotation_coefficients` closed-form
   `A_n`/`B_n` appendix (notes sec app_response has the A_n,B_n).

## Also note
- A separate background session was fixing the py2-ism in
  `factored_likelihood.PackLikelihoodDataStructuresAsArrays` (`rholm_intpArray = range(nKeys)`
  fails in py3 when the interpolant arg is truthy). Our tests pass `None` for that arg to
  dodge it.
- Subagents were unreliable for this work (deferred, reverted fixes, over-claimed). The
  validated results above were done/checked directly.
