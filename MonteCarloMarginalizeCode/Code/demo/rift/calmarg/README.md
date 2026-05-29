# In-loop calibration marginalization demo (H1 · L1 · V1)

This demo exercises RIFT's **in-loop calibration marginalization** — marginalizing
over detector calibration uncertainty *inside* the ILE GPU likelihood loop, instead
of as a postprocessing reweighting step (`calibration_reweighting.py`).

It uses the zero-spin synthetic data shipped with the CI
(`.travis/ILE-GPU-Paper/demos/`): a zero-noise injection observed in **three
detectors (H1, L1, V1)**, with the bundled `HLV-ILIGO_PSD.xml.gz` PSDs and
`overlap-grid.xml.gz` intrinsic grid. No real data or proprietary frames are needed.

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

Generated inputs:
* `distance_marg.npz` — distance-marginalization lookup table (`util_InitMargTable`).
* `cal_env/{H1,L1,V1}.txt` — synthetic calibration envelopes (amplitude 1-sigma 5–8%,
  phase 1-sigma 3–4.8°, deliberately different per IFO; see `tools/make_cal_envelopes.py`).

Each `out_*_0_.dat` row is one intrinsic point; the **last column is the
extrinsic-marginalized lnL**.

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
```

These append `--calibration-envelope-directory/--calibration-n-realizations/`
`--calibration-spline-count/--calibration-fused-kernel` to the ILE arguments. To
live-test on real data, copy your coinc, frames, PSD, and ini, then launch the pipe
with these flags.
