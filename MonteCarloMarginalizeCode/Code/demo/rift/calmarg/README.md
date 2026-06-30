# In-loop calibration marginalization demo (H1 · L1 · V1)

This demo exercises RIFT's **in-loop calibration marginalization** — marginalizing
over detector calibration uncertainty *inside* the ILE GPU likelihood loop, instead
of as a postprocessing reweighting step (`calibration_reweighting.py`).

It uses the zero-spin synthetic data shipped with the CI
(`.travis/ILE-GPU-Paper/demos/`): a zero-noise injection (`m1=35, m2=30` at 200 Mpc,
network SNR ≈ 17.5) observed in **three detectors (H1, L1, V1)**, with the bundled
`HLV-ILIGO_PSD.xml.gz` PSDs. No real data or proprietary frames are needed.

The template analyzed is the **injection itself** (`mdc.xml.gz`, a single matched
point), so the signal is actually present and `lnL ≈ ρ²/2 ~ 150` — large enough for the
calibration-marginalization effect to be visible. (Analyzing a far-off intrinsic grid
point instead would give `lnL ~ 0` and hide the effect.)

## Why in-loop, and what it does

The calibration model multiplies each detector's data by an uncertain,
frequency-dependent complex factor `C(f)`. Marginalizing over it in postprocessing
reweights extrinsic samples that were drawn *without* calibration knowledge — which
is very inefficient for high-SNR sources or broad calibration priors.

In RIFT's factored likelihood, applying `C(f)` to the **data** changes only the
data–template term `Q_lm(t) = <h_lm|d>(t)`; the template–template terms `U, V`
(`rho_sq`) are calibration-independent. So we draw `N` calibration realizations once,
build the per-realization `Q_lm` blocks, and Monte-Carlo marginalize over them on the
GPU while the extrinsic likelihood is evaluated:

```
Z_cal(theta) = (1/N) * sum_c  integral dt  exp( lnL_t(theta, c) )
```

Two implementations are demonstrated (both validated to agree to ~1e-14 on identical
inputs — see `make verify-exact`):

| path        | flag                                | where it runs            |
|-------------|-------------------------------------|--------------------------|
| **baseline**| *(none)*                            | no calibration marginalization |
| **loop**    | `--calibration-envelope-directory`  | Option B: Python loop over realizations reusing the existing kernel (CPU or GPU) |
| **fused**   | `+ --calibration-fused-kernel`      | Option C: a single fused CUDA kernel does Q + distmarg loglikelihood + cal/time log-sum-exp on-board (GPU only) |

The demo runs with **distance marginalization** on, so the fused path uses the
dedicated fused distmarg kernel.

## Running

```bash
make inputs        # build the distance-marginalization table + per-IFO cal envelopes
make verify-exact  # DETERMINISTIC: loop == fused == reference to ~1e-14 (the rigorous check)
make run-baseline  # ILE, no calibration marginalization
make run-loop      # ILE + in-loop calmarg, Option B
make run-fused     # ILE + in-loop calmarg, Option C (fused kernel)
make compare       # print the marginalized lnL from the three runs
# or simply:
make all
```

## Make targets reference

The demo has grown from the single-ILE correctness check into a ladder that goes all the way
to a runnable condor pipeline.  Targets, grouped by what they exercise:

**A. Numerical correctness + single-ILE (no condor)** — the original demo:
| target | what |
|---|---|
| `inputs` | build the distmarg table + per-IFO cal envelopes |
| `verify-exact` | DETERMINISTIC loop == fused == reference to ~1e-14 (the rigorous proof) |
| `run-baseline` / `run-loop` / `run-fused` | one ILE: no cal / loop calmarg / fused calmarg |
| `compare` | print the marginalized lnL from the three runs |
| `all` | inputs + verify-exact + the three runs + compare |
| `lowsnr-inputs` / `low-snr` | fainter copy of the source (~SNR 9) for a robust full-sampler check |

**B. Direct-ILE DAG + n-max tuning (condor on one GPU, e.g. cardassia)** — hand-rolled DAG, no
pseudo_pipe.  `DMARG_DAG`/`NCAL_DAG`/`NMAX_DAG`/`NEFF_DAG`/`NCHUNK_DAG` tunables; `FUSED=1`/`PILOT=1`:
| target | what |
|---|---|
| `dag-build` / `dag-validate` / `dag-run` / `dag` | build / check / submit / build+submit a vanilla fused-calmarg DAG |
| `tune-single` / `tune-condor` | one big ILE (python -u / a single condor submit) to push n_eff up at large `NMAX_DAG` |

**C. Top-level pipeline via `util_RIFT_pseudo_pipe.py` — OFFLINE build-validate** (no GPU/condor;
confirms everything threads through incl. TIME SAMPLES):
| target | what |
|---|---|
| `pp-build` / `pp-validate` / `pp` | build + validate a full pipeline (AV sampler, calmarg+fused, time-resampling) |
| `extr-build` / `extr-validate` / `extr` | same, with the **extrinsic handoff** (GMM seed) — checks the EXTRCONSOLIDATE node + seed wiring |

**D. RUNNABLE pipeline on the CI fake data (condor; one GPU on cardassia, or OSG/CIT)**:
| target | what |
|---|---|
| `pp-coinc` | build `ci_coinc.xml` from the injection (`util_SimInspiralToCoinc.py`) |
| `pp-run-build` / `pp-run` | build / build+submit the real pipeline (real cache + PSD + FAKE-STRAIN, calmarg, time-resampling) |
| `pp-run-pilot-build` / `pp-run-pilot` | same with the adaptive cal **pilots** enabled (separate `rundir_pp_pilot`) |
| `extr-run-build` / `extr-run` | tiny GMM **extrinsic-handoff** GPU+condor run (separate `rundir_pp_extr_run`) |

Runnable-target toggles (override on the make line):
| toggle | default | meaning |
|---|---|---|
| `OSG` | `0` (auto `1` if `SINGULARITY_RIFT_IMAGE` set) | layer on `--use-osg*` + container + frame transfer for CIT (container-only) |
| `PP_PILOT` | `0` | enable the cal pilots on `pp-run` |
| `PP_DMARG` | `0` | OPTIONAL distance marginalization with the fused kernel (`--internal-marginalize-distance`); recommended with `--extrinsic-handoff` |
| `PP_CALPOST` | `1` | write the recovered **calibration posterior** (`<out>_<event>_cal.dat`) at the final fairdraw |
| `PP_NIT` | `2` | forced iteration count |

> The runnable targets each start with their own `rm -rf <rundir>` and use **separate** run
> directories, so launching one never clobbers another that is still running.

**Helper utilities** (also runnable standalone): `util_ExtrinsicConsolidate.py` (pick the best
per-event extrinsic proposal), `util_CalMakePriorBreadcrumb.py` (write/patch a valid iteration-0
`cal_consolidated_-1.npz` prior placeholder — see "Pilot / OSG notes" below).

### Backends and review matrix

The fused path has two interchangeable backends and works with or without distance
marginalization, selected by `BACKEND` and `DMARG`:

| toggle | values | meaning |
|---|---|---|
| `BACKEND` | `gpu` (default) / `cpu` | CUDA kernels / pure-numpy (laptop, no CUDA) |
| `DMARG`   | `1` (default) / `0`      | distance-marginalization (fused distmarg kernel) / off (default-helper kernel) |

The deterministic check covers the whole matrix; e.g. on a laptop with no CUDA:

```bash
make verify-exact BACKEND=cpu DMARG=1      # CPU, distance marginalization
make verify-exact BACKEND=cpu DMARG=0      # CPU, no distance marginalization
make all DMARG=0                           # full non-distmarg end-to-end run
```

`BACKEND=cpu` runs on numpy where cupy is absent (a Mac); the full ILE runs use
`--gpu --force-xpy`, which is the numpy code path only on a machine without cupy.
`verify-exact BACKEND=cpu` always uses numpy (the backtest picks the backend directly),
so it is the portable cross-check.

Generated inputs:
* `distance_marg.npz` — distance-marginalization lookup table (`util_InitMargTable`).
* `cal_env/{H1,L1,V1}.txt` — synthetic calibration envelopes at **GWTC-4/O4a scale**
  (amplitude 1-sigma 1–1.6%, phase 1-sigma 1–1.6°, slightly different per IFO; the
  LIGO O4a strain calibration error is bounded at ≲2% / ≲2° below 2 kHz,
  arXiv:2508.18079).  `tools/make_cal_envelopes.py --wide` restores the original
  deliberately-broad 5–8% / 3–4.8° envelopes — a STRESS TEST that collapses the cal
  n_eff to O(1) at any practical `NCAL` (raw prior-draw marginalization cannot
  resolve it; use it only to exercise the error diagnostics or the adaptive pilot).

### Calibration MC error in the reported sigma

The sigma column of `out_*.dat` now **includes the calibration MC error in
quadrature** with the extrinsic sampling error (`--calibration-mc-error-extrinsic`,
default on; the probe batch is adaptive and draws distance from the run's own
distance prior).  The extrinsic integrator's variance is structurally blind to the
spread over the fixed cal draw set, so the old sigma badly understated the truth
whenever the per-point cal n_eff was small — at `NCAL_DAG=20` with wide envelopes
this produced a 2d lnL surface with ~1.0 point-to-point noise quoted at sigma~0.18.
Each ILE log now prints
`[calmarg error] sigma_lnZ: extrinsic ... (+) cal ... -> total ... ; cal n_eff X / N`
and warns when `cal n_eff < 10` (where the quoted sigma is only a lower bound).

### Adaptive cal draw count

The number of cal realizations is no longer trusted as given: after the cal-block
precompute, ILE probes the effective draw count at each intrinsic point
(`[calmarg adapt]` log lines) and **doubles the draw set** (fresh independent
draws, incremental precompute of only the new blocks) until
`--calibration-neff-cal-target` (default 10) is met or
`--calibration-n-realizations-max` (default 8× the initial count) is reached.
`NCAL` therefore sets the *starting* size; 100 is a sensible start with
GWTC-4-scale envelopes, with headroom to 800 by default.

Each `out_*_0_.dat` row is one intrinsic point. The columns end with
`... lnL  sqrt_var  ntotal  neff`, so the **marginalized lnL is column `[-4]`** and
the **last column is `neff`** (effective sample count, a sampler diagnostic — *not*
the result). `make compare` reads the right column and also prints `neff` and the
sampling error; don't compare the last column.

## Convergence caveat (read before trusting a single full-sampler lnL)

This demo analyzes **one** intrinsic point with the matched template, so the extrinsic
likelihood is a single narrow peak. RIFT's adaptive extrinsic sampler can struggle to
lock such a peak robustly from a single point (the full pipeline normally seeds from a
grid of points), and the convergence is sensitive to `NCHUNK`, SNR, and even the GPU
environment. Symptoms of a *non-converged* run: `neff` of order 1 (one draw dominates),
or `neff` large but lnL near 0 (the sampler spread into an off-peak region and missed
the signal). Always sanity-check `sqrt(2*lnLmax)` in the run log — it should be ~the
injected network SNR (≈17.5 here, ≈9 for the low-SNR variant); if it is, the signal was
found and any oddness is marginalization/convergence, not a missing signal.

Practical guidance: for a robust full-sampler run use the **low-SNR variant** (`make
low-snr`, broad peak) and a modest `NCHUNK` (~1000). For the *rigorous* loop-vs-fused
correctness check that does **not** depend on sampler convergence, use **`make
verify-exact`** (deterministic, ~1e-14). The calibration-marginalization correctness
claim rests on `verify-exact`; the full-sampler runs are an end-to-end illustration.

## How to read the results

* **`make verify-exact` is the exact numerical proof.** It bypasses the stochastic
  sampler and evaluates loop, fused, and a brute-force reference on identical inputs
  using this demo's real lookup table: they agree to ~1e-14.

* **The full ILE runs are stochastic.** RIFT's GPU Monte-Carlo integrator is **not
  bit-reproducible even with `--seed`** (cupy reductions and the sampler are not fully
  deterministic). At the small sample counts used here the run-to-run scatter in the
  marginalized lnL is a few tenths; loop and fused agree *within that Monte-Carlo
  noise*, as does a repeat of either one. Raise `NEFF`/`NMAX` (on a GPU with enough
  memory) to shrink the scatter and to resolve the calibration penalty (in-loop
  calmarg lowers and broadens the marginalized lnL relative to baseline).

## Quiet-source variant (low SNR)

The bundled CI injection is **network SNR ≈ 17.5** (`m1=35, m2=30` at 200 Mpc). For a
full-sampler sanity where loop-vs-fused agreement sits clearly above Monte-Carlo noise,
generate a fainter copy of the same source at larger distance (~SNR 9) and run the
comparison on it:

```bash
make lowsnr-inputs                  # writes mdc_lowsnr.xml.gz + lowsnr.cache (3 IFOs), prints the injected SNR
make low-snr                        # = make all  CACHE=$(CURDIR)/lowsnr.cache
# tune the loudness with INJ_DIST (Mpc): larger = quieter
make lowsnr-inputs INJ_DIST=600     # ~SNR 6
```

Nothing binary is committed — the frames/cache are regenerated locally, exactly as the
CI data itself is built (`util_WriteInjectionFile.py` → `util_WriteFrameAndCacheFromXML.sh`).

## Tunables

```bash
make all NCAL=100 NCHUNK=4000 NMAX=20000 NEFF=1000   # production-ish (needs a larger GPU)
```

`NCAL` (calibration realizations), `NCHUNK`/`NMAX` (extrinsic samples per block / max),
`NEFF` (target effective samples), `SEED`, `DMAX`, `SAMPLER`.

The demo uses the **adaptive-volume sampler** (`SAMPLER=AV`), the mature/stable GPU
code path. The GMM sampler (`mcsamplerEnsemble`) is newer and heavier on the GPU; if you
hit `CUDA_ERROR_ILLEGAL_ADDRESS` or other GPU instability, stay on AV (override with
`SAMPLER=GMM` only if you specifically want to test it). The fused kernels themselves
also guard against out-of-range window offsets, so a pathological draw can't trigger an
illegal memory access.

> **Memory note.** The fused/loop precompute holds `N` calibration realizations, so GPU
> memory scales with `NCAL` and `NCHUNK`. On a small card (≈2 GB) keep `NCHUNK` ≲ 1000;
> if you see `Out of memory ...` from cupy, lower `NCHUNK`/`NCAL`.

## Using this from the full pipeline

`util_RIFT_pseudo_pipe.py` threads the same options down to ILE:

```
--calmarg-envelope-directory DIR   # enables in-loop calmarg (per-IFO <IFO>.txt files)
--calmarg-n-realizations N         # default 100
--calmarg-spline-count M           # default 10
--calmarg-fused-kernel             # use the fused GPU kernel (else the loop method)
--calmarg-export-posterior         # write the recovered cal posterior at the final fairdraw (see below)
--internal-marginalize-distance    # OPTIONAL distance marginalization (composes with the fused kernel)
--calmarg-pilot                    # adaptive cal pilots: learn a cal proposal and SEED wide_{N+1}
--extrinsic-handoff                # GMM-seed handoff: carry the extrinsic posterior between iterations
```

These append the corresponding `--calibration-*` / `--extrinsic-*` flags to the ILE arguments.
To live-test on real data, copy your coinc, frames, PSD, and ini, then launch the pipe with
these flags.  Design notes for the advanced paths live next to the code:
`RIFT/calmarg/DESIGN_adaptive_driver.md` (cal pilots) and
`RIFT/calmarg/DESIGN_extrinsic_handoff.md` (extrinsic handoff).

### Recovered calibration posterior

With `--calmarg-export-posterior` (pipeline) / `--calibration-export-posterior` (ILE), the final
fairdraw stage draws, per output sample, one calibration realization in proportion to its
posterior weight and writes a **self-contained sibling `<output>_<event>_cal.dat`** with the
FULL draw — intrinsic + extrinsic + the drawn realization's spline nodes as labeled columns
`cal_<IFO>_amp_<k>` / `cal_<IFO>_phase_<k>`.  The recovered cal posterior is just those columns,
plottable with the standard tooling (it should sit inside, and no wider than, the input envelope
band).  In the demo this is `PP_CALPOST=1` (default on).

### Pilot / OSG notes

- **Iteration-0 placeholder.**  On OSG the cal pilots seed wide_{N+1} from a transferred
  breadcrumb; iteration 0 has none yet, so pseudo_pipe writes a `cal_consolidated_-1.npz`
  *placeholder*.  It is a **valid "prior" breadcrumb** (proposal == prior → seeding from it ==
  cold prior draws, zero weights) so it loads cleanly on any ILE binary.  To (re)generate one for
  an already-built run dir (e.g. to patch a run launched before this fix, without rebuilding the
  container):
  ```bash
  util_CalMakePriorBreadcrumb.py --calibration-envelope-directory rundir/cal_env \
      --ifo H1 --ifo L1 --ifo V1 --fmin 10 --fmax 2047 --calibration-spline-count 10 \
      --output rundir/cal_consolidated_-1.npz
  ```
  (`--ifo` order / `--fmin` / `--fmax` / `--calibration-spline-count` must match the wide-ILE cal
  settings so the node dimension `2·spline·n_ifo` lines up.)

- **Execute-point vs pipeline-writer.**  Changes to the ILE binary / likelihood / `RIFT/calmarg`
  need a **container rebuild** to take effect on OSG/CIT; changes to `util_RIFT_pseudo_pipe.py` /
  `create_event_*` / `dag_utils*` / the Makefile are pipeline-writer only (no rebuild).
