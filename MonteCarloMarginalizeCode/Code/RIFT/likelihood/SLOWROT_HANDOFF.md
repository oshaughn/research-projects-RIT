# Slow-rotation RIFT likelihood — handoff / breadcrumb

Branch: `rift_slowrot` (off `rift_O4d_fix_rvs_clear_fairdraw_batch`). Design notes + LaTeX
paper live in a **separate local repo** `~/rift-slow-rotation` (no git remote — local only;
21-page PDF via `latexmk -pdf notes/main.tex`).

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

## OPEN / NEXT (the one remaining systematic pass — do it all together)
1. **Path B rigorous validation**: vs a brute force with TIME-VARYING DELAY tau_k(t) on a
   genuinely LONG (BNS) signal, where p>=1 delay drift is non-negligible (short-BBH delay
   drift is ~1e-8, so current Path B tests exercise machinery/bound but not the delay
   PHYSICS). This is the analogue of V1b that caught the reference-time bug.
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
