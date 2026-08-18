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
   precessing+HM -> route (a) pinned-sky TD fold.
   LIKELIHOOD INTEGRATION DONE + VALIDATED (route b): `factored_likelihood_freqresponse.py`
   (+ `test_slowrot_freqresponse_likelihood.py`).  Fold the common e^{-i2pi fL/c} delay into the
   arrival time; power-series the residual transfer -> sky-independent W_p(f) folded into the FD
   modes (reuse ComputeModeIP*) x analytic b_p(sky) -> SKY EXTRINSIC.  MAINTAINED-STYLE NoLoop
   entry point DiscreteFactoredLogLikelihoodFreqResponseNoLoop (NOT SingleDetectorLogLikelihood).
   Validated (CE 40km, 16s): V1 finite-size(L->0) == maintained NoLoop baseline to 3.3e-9; V2 on
   a finite-size injection the long-wavelength NoLoop deficit 2.71 -> finite-size 0.558 (converged
   Qmax=2), residual ~= peak-resolution floor; V3 Cauchy-Schwarz respected.  Scalar companion
   agrees with the NoLoop to 0.156 (interp-vs-nearest floor).
   REMAINING: ILE wiring (a --freqresponse flag mirroring --rotation-slow); full precessing+HM
   (route a pinned-sky); a value demo (the finite-size effect only bites for CE/3G).
   AUDIT NOTE (2026-07): all slowrot likelihoods route through the maintained NoLoop; there are
   NO SingleDetectorLogLikelihood calls; the ILE wires only NoLoop paths.  The scalar
   FactoredLogLikelihood*/SingleDetectorLogLikelihood are used only as secondary references and
   carry a peak-resolution floor at high SNR -- prefer NoLoop for any absolute comparison.

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
    python RIFT/likelihood/test_slowrot_cauchy_schwarz.py      # lnL <= 0.5<d|d> + explicit-model value
    python RIFT/likelihood/test_slowrot_pathB.py               # Path B reduction + bound
    python RIFT/likelihood/test_slowrot_headtohead.py          # matched-sample rotation vs baseline (cubic)
    python RIFT/likelihood/test_slowrot_freqresponse.py        # [Path D] finite-size response vs LAL
    python RIFT/likelihood/test_slowrot_freqresponse_likelihood.py  # [Path D] likelihood: V1/V3 + V4 positive control

VALUE DEMOS (verify-anywhere, no condor/GPU) -- consolidated in the RIFT tree:
    cd demo/rift/slowrot && make demo        # rotation (Path A/B) + finite-size (Path D)
    #   demo_rotation.py    : gain lnL_rot-lnL_static grows with Omega*T (null control at short T)
    #   demo_finite_size.py : gain lnL_finite-lnL_LWL grows with arm length fL/c (null at LIGO 4km,
    #                         +39.6 nats at CE 40km); writes outputs/*.{txt,png}.  See its README.md.

End-to-end ILE head-to-head (ILE-GPU-Paper demo data), baseline vs rotation vs finite-size:
    D=~/RIFT_develUWM/src/research-projects-RIT/.travis/ILE-GPU-Paper/demos
    integrate_likelihood_extrinsic_batchmode --vectorized [std opts] ...            # baseline
    integrate_likelihood_extrinsic_batchmode --vectorized --rotation-slow ...        # Path A
    integrate_likelihood_extrinsic_batchmode --vectorized --rotation-slow --rotation-p-max 1 ...  # Path B
    integrate_likelihood_extrinsic_batchmode --vectorized --freqresponse \
        --freqresponse-arm-length 40000 --freqresponse-qmax 6 ...                     # Path D (finite-size)
    # --interpolate-time selects cubic sub-bin time interpolation for all of the above (default nearest).

## Validation status (all PASSING)
- Response harmonics vs LAL: ~1e-16.  FD ops vs LAL round trips: ~1e-13.
- Path A scalar: V1a (Omega=0 vs baseline) 2.7e-12; V1b (real vs brute force) 2.6e-9.
- Path A vectorized: vs baseline NoLoop 3.6e-12; vs brute force 3.9e-10 (against the REWRITTEN,
  convention-free brute force -- see below; the old figure 3.2e-10 was against a reference that
  shared the implementation's conventions); V0 (precompute recovery on real data) exact.
- jax_ile (issue #131, ported 2026-08-18): the JAX rotation contraction now carries the
  arrival-time post-phase in BOTH terms, so its rho_sq is arrival-time dependent (rank-1 in
  (sample, time bin), bucketed by m = n_a' - n_a, as the NoLoop does).  test_jax_slowrot.py
  rotation gate (a) vs the NoLoop: max|rel| 1.33e-05 -> 2.14e-15 (max|abs| 5.37e-02 -> 5.46e-12)
  at p_max=0, and 7.75e-14 (2.62e-10 nats) at p_max=1, which the file now also runs -- Path B is
  a distinct branch here because several p share a harmonic, so the m buckets mix p and the V
  reflection must resolve within p.  Gate restored to 1e-10.  Path D (freqresponse) has no
  post-phase and is unchanged at 1.6e-14.  Value pinned independently by
  test/jax/test_jax_slowrot_cauchy_schwarz.py (see below).
- Cauchy-Schwarz (test_slowrot_cauchy_schwarz.py, 2026-08-17): lnL sits ON 0.5<d|d> to 0 nats
  with the data equal to the exact Path-A model, and matches an explicit time-domain
  <d|h>-(1/2)<h|h> to 5e-11.  Before the rotation_post_phase fix the same test overshot the
  bound by 83.6 nats.
- Cauchy-Schwarz, JAX (test/jax/test_jax_slowrot_cauchy_schwarz.py, 2026-08-18): the same ladder
  against jax_ile, at p_max=0 AND p_max=1, with the data equal to the exact model at each p_max
  so lnL sits ON the bound.  p_max=0: (A) 4.99 nats, (B) deficit 0.0, (C) 6.5e-11 vs an explicit
  time-domain <d|h>-(1/2)<h|h>, (D) 5.8e-11 vs the numpy NoLoop.  p_max=1: (A) 36.4 nats,
  (B) deficit 5.1e-04 of 3.2e+05, (C) 1.36e-01 = 4.2e-07 of 0.5<h|h> -- and the numpy NoLoop
  disagrees with the SAME explicit reference by the identical 1.36e-01, so that residual is the
  reference's conditioning (a divergent delay Taylor series at INFL=1350), not the port --
  (D) 1.3e-09.  Mutation-tested at both p_max: dropping the post-phase from both terms is
  self-consistent (bound NOT violated) and (C) catches it at 95.3 nats (p_max=0) / 965.7 nats
  (p_max=1); dropping it from the model norm only overshoots the bound by 10.6 / 1122.5 nats and
  (B) catches it.
- Path B: scalar reduce-to-baseline 9e-13; respects 0.5<d|d>; vectorized reduce 6.4e-12.
- Path D (finite-size, --freqresponse): response Sum_p b_p W_p == antenna_response_fd to 6e-11
  on both +/-f; likelihood L->0 reduces to baseline NoLoop 3e-9; Cauchy-Schwarz respected;
  V4 positive control asserts finite-size beats LWL by +38.9 nats (15+13 Msun, fmax=2000, CE 40km).
- Cubic time-interp (from calmarg_in_loop, --interpolate-time): both slow-response NoLoops now
  support time_interp='nearest'|'cubic'.  Cubic exposed+fixed a sub-bin GPS-cancellation bug in
  the time reference; head-to-head regression floor 1.6e-3 -> 4.5e-13, test_slowrot_noloop 3.6e-12.
- ILE runs clean (no inf/nan) for baseline / Path A / Path B / Path D; rotation ~ baseline for the
  short demo signal (rotation effect ~0.008 in lnL, negligible as physics requires).

## CRITICAL lesson (a bug that hid for a while)
The U/V template modulation MUST be referenced to the template's INTRINSIC time (epoch ~0),
not the absolute event time t_ev~1e9 -- else a ~1e4 rad spurious phase randomizes U/V and
inflates lnL ABOVE 0.5<d|d>. This bug once made a short-BBH rotation shift look like ~660 in
lnL; corrected it is ~0.008. My independent "brute force" reference shared the same
convention, so `vec == brute-force` PASSED while both were wrong. **Always cross-check
against the Cauchy-Schwarz bound 0.5<d|d>, not only against a reference that can share
conventions.**

### The same lesson fired again, and this time the bound caught it (2026-08-17)
Referencing the modulation to the intrinsic epoch is necessary but NOT sufficient. It leaves a
residual `exp(i n Omega (t_arrival - tref))` -- the post-phase -- which the implementation
dropped from the model norm, and it hid behind a second shortcut: term1 pushed the modulation
onto the DATA (`<e^{inOmega.}h|d> == <h|e^{-inOmega.}d>`), an identity that is **false for a
noise-weighted overlap**, because a frequency shift does not commute with the 1/S(f) band
weight. term1 and term2 were therefore evaluating different templates, and lnL exceeded
0.5<d|d> by ~1e-4 of <d|d> -- growing linearly with `Omega * (t_arrival - tref)`.

Both are fixed: `chi_a` now goes into the data-term overlap directly (data untouched), and
`rotation_post_phase()` applies `C~_a = C_a exp(i n_a Omega (t - tref))` to BOTH terms. Patching
term2 alone does NOT work -- measured, it still violates by 73 nats where the full fix sits on
the bound exactly.

**And, exactly as the lesson above predicted, `test_slowrot_noloop_bruteforce` certified the bug
at 3e-10 because its reference took the same two shortcuts.** That reference has been rewritten
to build the real strain `Re[F(t') hY(t'-t_arr)]` in the time domain at every arrival sample and
take both inner products of that one series -- sharing no convention with the implementation. It
now fails against the old code (2.5e-3) and passes against the new one (3.9e-10).
`test_slowrot_cauchy_schwarz.py` guards the bound itself.

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
