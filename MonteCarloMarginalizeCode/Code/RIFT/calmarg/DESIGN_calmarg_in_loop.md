# In-loop calibration marginalization in RIFT ILE

Branch: `rift_O4d_junior_calmarg_in_loop` (off `rift_O4d_junior_distance`)

## Motivation

RIFT currently marginalizes over calibration uncertainty in **postprocessing**
(`bin/calibration_reweighting.py`, bilby-based): after `extrinsic_posterior_samples.dat`
is produced, each sample is reweighted against a set of random calibration draws.
The extrinsic samples entering this step are *not* informed by calibration, so for
high-SNR sources and/or broad calibration priors the reweighting is very inefficient
(most proposed samples get tiny weights).

Modern GPUs are heavily under-utilized by RIFT's inner loop, so we move the
calibration marginalization **inside ILE**, marginalizing over calibration draws
on-board while the extrinsic likelihood is being evaluated.

## Key idea: apply calibration to the *data*

The factored likelihood evaluated by
`factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`
combines two quantities:

* `kappa_sq` — the **data-template** term, built from the GPU Q-product over the
  precomputed rholm timeseries `rholmsArrayDict[det] = <h_lm|d>(t)`.  This is the
  **only** data-dependent quantity.
* `rho_sq` — the **template-template** cross terms `U,V` (`ctUArrayDict`,
  `ctVArrayDict`), `<h_lm|h_l'm'>`.  These depend on the template and PSD but **not**
  on the data.

If calibration `C(f)` is applied to the **data** (`d -> C(f)·d`), then `rho_sq` is
**calibration-independent** and is computed once; only `kappa_sq` changes per
realization.  This is what makes in-loop marginalization cheap.

> **Convention note for review.** Bilby's `GravitationalWaveTransient` applies the
> calibration factor to the *template/response*, which also rescales the `<h|h>`
> norm.  Applying to the data (our choice) and applying to the template agree to
> first order in the calibration amplitude but differ at second order.  The
> apply-to-data choice is what preserves the efficiency win (shared `U,V`).  The
> backtest below quantifies the difference against `calibration_reweighting.py`.

## Data layout

`RIFT/calmarg/generate_realizations.py::create_realizations` draws `n_cal` complex,
two-sided calibration factors on the full FFT frequency grid (matching
`lalsimutils` packing) from a bilby envelope `.txt` file, shape `(npts_seg, n_cal)`.
Column `c` across detectors is one **joint** draw.

`ComputeModeIPTimeSeries` (cal branch) applies realization `c` to the data and
concatenates the resulting windowed rholm into one timeseries:

```
rholm[det]  =  [ block_0 | block_1 | ... | block_{n_cal-1} ]      length = N_window * n_cal
```

`PackLikelihoodDataStructuresAsArrays` carries this long array through unchanged,
so `rholmsArrayDict[det]` has shape `(n_lms, N_window * n_cal)`.  Realization `c` is
selected simply by shifting the per-sample window offset:

```
ifirst_c = ifirst + c * N_window
```

## Marginalization (implemented: Option B)

We Monte-Carlo marginalize over the `n_cal` draws:

```
Z_cal(theta) = (1/n_cal) * sum_c   integral dt  exp( lnL_t(theta, c) )
```

`DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop` gains an `n_cal` argument.
With `n_cal == 1` the code path is byte-for-byte the original.  With `n_cal > 1`:

1. `rho_sq` is accumulated once in the detector loop (calibration-independent).
2. The per-detector Q-product inputs (`Q`, `FY_conj`, `ifirst`, `N_window`) are cached.
3. For each realization `c`, `kappa` is recomputed via the **existing** GPU kernel
   (`Q_inner_product.Q_inner_product_cupy`) with `ifirst + c*N_window`, combined with
   the shared `rho_sq` through the same `loglikelihood` callback (distance/phase marg),
   and accumulated with a **streaming log-sum-exp** for numerical stability.
4. Finish with `simps` over time and `- log(n_cal)`.

**Why Option B (cal loop) over the alternatives:**

| Option | Idea | Memory | Kernel | Review cost |
|---|---|---|---|---|
| A | replicate extrinsic batch ×n_cal, one kernel call | ×n_cal (forces smaller batch) | reused verbatim | lowest LOC |
| **B (chosen)** | loop realizations, reuse kernel, stream log-sum-exp | **unchanged** | reused verbatim, n_cal launches | low |
| C | fused CUDA kernel: Q + loglikelihood + cal-LSE on-board | minimal | new kernel | highest |

Option B is memory-neutral and reuses the validated kernel — the right
minimum-violence first step given GPUs have spare throughput.

### Option C (implemented for the default helper)

`cal_method='fused'` runs a single fused CUDA kernel
(`RIFT/likelihood/cuda_Q_fused_calmarg.cu`, wrapped by
`RIFT/likelihood/Q_fused_calmarg.py`).  One thread per extrinsic sample loops over
realizations × time × detectors × modes, forms the data term `kappa`, applies the
default factored-likelihood helper `lnL_t = invDist*Re(kappa) - 0.5*rho_sq`, and
accumulates a streaming, Simpson-weighted log-sum-exp over `(c,t)` — returning
`lnL[j]` directly.  No `(batch, n_cal, npts)` intermediate, no per-realization
Python launches.

Time integration matches Option B exactly by passing the composite-Simpson weight
vector `w_t = simps(I, dx=deltaT)` (simps is linear, so its action is a fixed weight
vector) into the kernel.  `rho_sq` is calibration-independent and passed in
pre-summed over detectors.

**Validated** in the harness vs the brute-force reference and Option B to ~1e-15 on
GPU, single- and multi-detector (H1,L1,V1 — exercises the kernel's detector loop and
the per-detector ifirst stacking).  Throughput (NVS 510, sm_30; single synthetic
detector):

| case | reference | Option B | Option C |
|---|---|---|---|
| n_cal=100, 1024 samples | 695 ms | 170 ms | **22 ms** |
| n_cal=200, 8192 samples | 7080 ms | 2422 ms | **279 ms** |

i.e. ~8–9× over Option B and ~25–32× over brute force, with bit-level agreement.

### Option C, stage 2 — distance marginalization (implemented, separate kernel)

The dominant production path uses the distance-marginalization `loglikelihood`
(sites 1828/1871).  This is implemented as a **separate** kernel
(`RIFT/likelihood/cuda_Q_fused_calmarg_distmarg.cu`, wrapper
`Q_fused_calmarg_distmarg_cupy`), kept apart from the default-helper kernel on
purpose: it keeps each kernel's review surface small, leaves the simpler kernel as a
baseline, and leaves `cal_method='loop'` (Option B) as a full fallback for distmarg
on both CPU and GPU.

It reproduces `distmarg_loglikelihood` exactly on-board:
`x0 = kappa/rho_sq`; `s = asinh(√bmax·(x0−xmin)) − asinh(√bmax·(xmax−x0))`;
`t = asinh(rho_sq/bref)`; bilinear interpolation of `lnI_array` at `(s,t)` (matching
`EvenBivariateLinearInterpolator`, with the same in-bounds mask, contributing 0
otherwise); plus `exponent_max`.  Selected via `cal_method='fused'` **and** passing a
`cal_distmarg` table dict (`lnI_array`, `s0/ds/smin/smax`, `t0/dt/tmax`,
`xmin/xmax/sqrt_bmax/bref`); with `cal_distmarg=None` the default-helper kernel is
used.

**Validated** in the harness (`--loglikelihood distmarg`, which builds a
self-consistent table and the mirror Python closure for reference/Option B) to
~1e-14 vs the brute-force reference, single- and multi-detector (the asinh/bilinear
differ from numpy only at ULP level).  Throughput (sm_30):

| case (distmarg) | reference | Option B | Option C |
|---|---|---|---|
| n_cal=100, 1024 samp, 2 det | 1364 ms | 495 ms | **77 ms** |
| n_cal=200, 2048 samp, 3 det | 6136 ms | 2358 ms | **333 ms** |

i.e. ~6–7× over Option B.

**Scope / limitations of both fused kernels** (raise `NotImplementedError`
otherwise): GPU only; `phase_marginalization=False`; all detectors share
modes/length (true after global mode pruning).

### Driver wiring (opt-in)

`integrate_likelihood_extrinsic_batchmode` exposes the fused path behind
`--calibration-fused-kernel` (off by default).  When set (and on GPU, with
calibration marginalization active), the driver packages the distance-marginalization
`lookup_table` (`s_array`, `t_array`, `lnI_array`, `bmax`, `bref`) plus `xmin/xmax`
into a `cal_distmarg` dict and passes `cal_method='fused'` at the **non-phase-marg**
distmarg call site.  The phase-marg distmarg site and everything else stay on
`cal_method='loop'` (Option B), which remains the default and the fallback for all
cases.  On CPU the flag is ignored with a warning (the kernel is GPU-only).

**End-to-end status.** Run through `integrate_likelihood_extrinsic_batchmode` on the
CI fake data with `--distance-marginalization` + a real `util_InitMargTable` table +
`--calibration-envelope-directory --calibration-fused-kernel`.  This caught a real
wiring bug — in the distmarg path `P.dist` is fixed at the fiducial, so `invDistMpc`
is a scalar, but the fused kernel wants one value per extrinsic sample; the fused
branch now broadcasts it to `(npts_extrinsic,)`.  After the fix the fused path runs to
completion.  Numerics were validated deterministically with
`backtest_calmarg.py --loglikelihood distmarg --real-table <npz>`: fused == reference
== loop to ~2e-14 on the production table.  (A full *sampler* end-to-end numerical
comparison needs a larger GPU than the local 2 GB card, which OOMs / returns nan under
load.)

Remaining: a full numerical end-to-end on a larger GPU; phase-marginalization support
in the fused kernels (then the phase-marg distmarg site can opt in too).

## Driver wiring

`bin/integrate_likelihood_extrinsic_batchmode` already had the scaffolding:

* options `--calibration-envelope-directory`, `--calibration-n-realizations`,
  `--calibration-spline-count`;
* builds `calibration_realization_dict` and passes it into `PrecomputeLikelihoodTerms`.

This branch adds `n_cal_for_likelihood` (= `--calibration-n-realizations` when
calibration marginalization is active, else 1) and threads it into the three
production `DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop` call sites
(plain / distance-marg / distance+phase-marg).

## Bug fixed

In `ComputeModeIPTimeSeries`'s calibration branch the inner product was being taken
against the *original* `data` instead of the calibration-modified `data_now`
(`IP.ip(hlms[pair], data)` → `IP.ip(hlms[pair], data_now)`), so the calibration
factor was previously never applied.

## Validation

`RIFT/calmarg/test_calmarg_reduction.py` builds a synthetic 2-mode, single-detector
case and checks the `n_cal>1` result against a brute-force reference — running the
unchanged `n_cal==1` path on each realization block separately and combining by hand
(`logsumexp_c(lnL_c) - log n_cal`).  Agreement is machine precision (~1e-15) on both
the CPU (`xpy=np`) and the production GPU (`xpy=cupy`, real `Q_inner_product` kernel)
paths.  It also confirms the `n_cal==1` path is a regression-identical block-0 eval.

## Backtest harness

`RIFT/calmarg/backtest_calmarg.py` is the rig **Option C is developed against**.  It
holds a `METHODS` registry — `reference` (brute-force per-block + logsumexp),
`in_loop_B` (the `n_cal>1` call), and `in_loop_C` (a stub raising `NotImplementedError`
until the fused kernel exists) — and evaluates each over synthetic inputs that
exercise the cal-block structure, reporting `max|lnL - reference|` and best-of-N
timing on CPU (`--backend cpu`) or GPU (`--backend gpu`).  Wire the fused kernel into
`method_in_loop_C` and the harness validates it automatically.

```
python -m RIFT.calmarg.backtest_calmarg --backend gpu --n-cal 100 --npts-extrinsic 4096 --repeat 5
```

Current status: `in_loop_B` reproduces `reference` to ~1e-15 on CPU and GPU, with and
without phase marginalization; on GPU it is ~3–4× faster than the brute-force
reference (which redundantly recomputes `rho_sq` per realization — exactly the
redundancy Option C removes).

`run_physics_backtest()` in the same module is the **scaffold** (docstring + TODOs) for
the heavier real-data comparison vs bilby `calibration_reweighting.py`: load a real
ILE precompute + cal envelopes + the bilby data_dump, evaluate the in-loop calmarg
likelihood on the same extrinsic samples, and compare per-sample lnL and the
log-evidence shift.  It needs frames/PSDs, so it runs on the stable host (not in CI).

## Calibration MC error budget (implemented)

The sampler's reported variance is the *extrinsic* sampling variance with the cal
draw set held fixed — it is structurally blind to the Monte-Carlo error of the
`(1/n_cal) sum_c` average.  Empirically (demo `pp-run`, wide envelopes, `NCAL_DAG=20`)
this produced a 2d lnL surface with ~1.0 point-to-point noise quoted at sigma~0.18
(chi^2/dof ~ 34 against a smooth surface fit).

`adaptive.cal_mc_error_from_components(comp, cal_log_weights)` computes, from the
per-realization components on a modest extrinsic-prior batch (`return_cal_components`,
responsibilities are ~extrinsic-independent — same trick as the pilot):

* `a_c = w_c Z_c / (n_cal Z)` — normalized per-draw contributions (sum to 1);
* `Var(lnZ) ~= n_cal * Var_c(a_c)` (delta method; reproduces the lognormal
  `(e^{sigma^2}-1)/n_cal`, validated in `test_cal_mc_error.py`);
* `neff_cal = 1/sum a_c^2` — when `< 10` the estimate is a LOWER BOUND and the
  point is flagged in the log.

The driver folds this in quadrature into the reported sigma column and prints
`[calmarg error] sigma_lnZ: extrinsic X (+) cal Y -> total Z ; cal n_eff ...`.
The probe (`_cal_error_probe`) uses an ADAPTIVE extrinsic batch (doubling until the
estimate stabilizes, capped by `--calibration-mc-error-extrinsic`, default 8192,
0 disables) and draws distance from the RUN'S distance prior: the sampler's own
`prior_pdf['distance']` when distance is sampled (uniform proposal + importance
weight, so the cosmo/redshift variants are handled by construction), the `--d-prior`
pdf when distance marginalization is active, or the PINNED value (warned: at fixed
distance the distance/amplitude degeneracy cannot absorb amplitude-like cal
perturbations, so the estimate is conservative).

**Adaptive draw count** (`--calibration-neff-cal-target`, default 10;
`--calibration-n-realizations-max`, default 8x initial): after the cal-block
precompute, the same probe measures `neff_cal` at this intrinsic point; while below
target the draw set is DOUBLED — fresh independent draws appended via
`_draw_more_calibration_draws` (extends the realization dict, importance weights,
and node bookkeeping in place), with an incremental `PrecomputeLikelihoodTerms` of
only the new blocks concatenated onto the packed rholm arrays.  So
`--calibration-n-realizations` is a *starting* size, not a trusted constant.
`[calmarg adapt]` log lines record the escalation.

**Sizing guidance** (toy-model scaling, see the paper repo
`demos/calmarg/cal_envelope_scaling.py`): per-draw spread `sigma_lnL ~ rho^2 eps_A`
(amplitude-envelope dominated, ~1.0 per 1% amplitude at network SNR 20), and
`n_cal ~ (e^{sigma_lnL^2}-1)/sigma_target^2`.  GWTC-4-scale envelopes (<~2% / <~2 deg)
need `n_cal ~ 100-1000` at SNR 20: **start at 100 and let the adaptive escalation
work; 300 is a comfortable fixed choice**.  Beyond ~3% amplitude (or proportionally
higher SNR) prior draws are hopeless; the learned-proposal machinery (pilot /
breadcrumbs) targets that regime but is EXPERIMENTAL — it must be validated against
the brute-force path before being relied on, and is deliberately kept out of the
active/default paths.  Memory: realization blocks add ~0.3 MB/draw GPU-resident in
the demo config (88 MB at n_cal=300); per-eval cost is linear in n_cal (fused
kernel: ~0.25 s per 1000-sample chunk at n_cal=300 extrapolating the sm_30 timings
above).

## Open items / future work

* **Option C** fused kernel for maximum throughput; backtest vs Option B and vs the
  bilby postprocessor on a high-SNR / broad-prior event.
* **Reproducibility:** `create_realizations` uses unseeded `np.random`; add a
  `--calibration-seed` so a run's draw set is reproducible.  DECISION (2026-06):
  workers must KEEP drawing independent sets — common random numbers across
  intrinsic points were considered and rejected (a shared draw set makes the lnL
  surface artificially smooth and bakes its O(1/sqrt(n_eff_cal)) bias into the
  posterior); the variance is instead disclosed via the cal MC error budget above
  and beaten down with larger n_cal (now grown adaptively per point).
* **Calibration-parameter export:** Option B does not record which realization was
  selected (acceptable per scope — parameter draws can be regenerated at the end as
  the current `--dump_cal_realization` path does).
* **Grid sanity asserts:** verify `len(realizations) == data.length` and
  `N_window*n_cal == rholm length` explicitly at setup time.
* **CPU + phase-marg + calmarg** uses an explicit einsum mirroring the kernel; covered
  by the test for the non-phase-marg case.
