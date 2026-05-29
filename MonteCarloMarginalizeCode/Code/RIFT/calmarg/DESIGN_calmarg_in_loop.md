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
minimum-violence first step given GPUs have spare throughput.  Option C remains the
optimized drop-in to **backtest** against B once the physics is confirmed.

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

## Open items / future work

* **Option C** fused kernel for maximum throughput; backtest vs Option B and vs the
  bilby postprocessor on a high-SNR / broad-prior event.
* **Reproducibility:** `create_realizations` uses unseeded `np.random`; add a
  `--calibration-seed` so a run's draw set is reproducible. Different ILE workers
  currently draw independent sets (valid for the integral; worth pinning).
* **Calibration-parameter export:** Option B does not record which realization was
  selected (acceptable per scope — parameter draws can be regenerated at the end as
  the current `--dump_cal_realization` path does).
* **Grid sanity asserts:** verify `len(realizations) == data.length` and
  `N_window*n_cal == rholm length` explicitly at setup time.
* **CPU + phase-marg + calmarg** uses an explicit einsum mirroring the kernel; covered
  by the test for the non-phase-marg case.
