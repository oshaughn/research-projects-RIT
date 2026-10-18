# Batching the ILE mode cross terms

## The problem

`PrecomputeLikelihoodTerms` and its two response variants build the mode cross-term
banks `U` and `V` before any Monte Carlo sampling starts. For the slow-rotation path at
`--l-max 4` the precompute is 94% of ILE wall clock.

Profiled on `ldas-pcdev2`, one intrinsic point, CE+ET+K (5 detectors), IMRPhenomXPHM,
seglen 128 s, srate 8192, RTX PRO 4000 Blackwell, container
`rift_o4d_cc90-120_cuda128_20260717.sif`, RIFT `43918b22a`. Records:
`/scratch/richard.oshaughnessy/response_cost/campaign_main/`.

| response | modes | precompute (s) | integration (s) | wall (s) |
|---|---|---|---|---|
| long-wavelength | l<=2 | 26.5 | 28 | 54.6 |
| long-wavelength | l<=4 | 61.4 | 32 | 93.0 |
| slow rotation | l<=2 | 335 | 38 | 373 |
| slow rotation | l<=4 | 1318 | 86 | 1404 |

`cProfile` on the l<=4 rotation run splits the 1318 s as follows. All three response
paths reach the same function, `factored_likelihood.ComputeModeCrossTermIP`.

| site | calls | cumulative (s) | share of precompute |
|---|---|---|---|
| `ComputeModeCrossTermIP` | 260 | 1151.6 | 87% |
| ... of which `ComplexIP.ip` | 112560 | 929.8 | 71% |
| ... of which `InnerProduct.__init__` | 260 | 221.0 | 17% |
| `ComputeModeIPTimeSeries` (data-term overlaps, FFTs) | 25 | ~130 | 10% |
| waveform generation | 2 | ~21 | 2% |

Waveform generation is 2% of the precompute. The FFTs are 10%. The cost is the cross
terms, and inside them it is 112560 separate reductions over a 1048576-point array.

## Structure

For one detector and one pair of response-basis elements,

    U[a][b] = 2 df sum_f conj(hA_a[f]) hB_b[f] W(f)

with `W` the two-sided `1/S(f)` band weight. The shipped code evaluates this with a
Python loop over `(a,b)`, calling `ComplexIP.ip` once per pair. Each call allocates four
full-length temporaries and reduces them.

This is one matrix product. Stack the modes into `A[a,f]` and `B[b,f]` and the whole
block is `2 df . conj(A) . (B W)^T`: one pass over the data plus a GEMM, instead of
`Na*Nb` passes.

The counts multiply. At `--l-max 4` there are 21 modes, so 441 pairs per block. The
slow-rotation path has 5 basis elements, giving 25 blocks per detector for `U` and 25
for `V`, over 5 detectors: 250 blocks, 110250 inner products. `--freqresponse`
(`Qmax+2 = 6` elements) and the combined path added by PR #307
(`factored_likelihood_rotating_freqresponse.py`, `6 x 5 = 30` elements, 900 blocks per
detector per bank) scale as the square of the basis size. All four call
`ComputeModeCrossTermIP`, so all four take the batched path when it is enabled.

## What is not the answer

**More cores.** The inner product is memory-bandwidth bound, not compute bound. Raising
`OMP_NUM_THREADS` from 1 to 8 or 16 made every variant slower, including the GEMM.
Per-block times on `ldas-pcdev2` in the container, 21 modes, 1048576 bins, 4 replicates:

| variant | OMP=1 (s) | OMP=8 (s) |
|---|---|---|
| shipped loop | 6.94 | 7.17 |
| batched GEMM | 0.478 | 0.502 |

**The GPU.** The device sits at 3-9% mean utilization through these runs, but it is not
what is missing. A cupy version of the same GEMM takes 0.037 s against 0.118 s on the
CPU, and it still needs the host-side stacking that dominates the batched path. The
card's FP64 throughput is a small fraction of its FP32 throughput, so the arithmetic
never becomes the constraint. The 26x from batching on the CPU is available without a
device transfer, and the remaining time is host memory traffic that a GPU does not
remove.

## Changes

Four, in `lalsimutils.py` and `factored_likelihood.py`. No response-path module is
modified.

1. `ComplexIP.ip` no longer builds an all-ones `factor_shift` array when
   `include_epoch_differences` is false. Bit-identical: `x * 1.0 == x` in IEEE 754. It
   removes a `len2side` allocation and one full-length complex multiply per call, 55.5 s
   of allocation alone in the profiled run.

2. `InnerProduct.__init__` zeroes the inverse-spectrum-truncation window by slice
   assignment instead of a per-element loop over ~1e6 SWIG elements. Bit-identical.
   Construction drops from 0.287 s to 0.0455 s at campaign shapes (`ldas-grid`, IGWN
   CVMFS python). This is the whole 221 s above. It applies only when
   `--inv-spec-trunc-time` is nonzero, which is the driver default but is not the
   production setting; see "Provenance of the motivating numbers".

3. The array-PSD weight fill is vectorized, preserving the loop's `!= 0` mask: 22.2 ms
   to 1.9 ms at 128512 in-band bins. The driver passes a `REAL8FrequencySeries`, so this
   branch is off the measured path.

4. `ComplexIP.ip_matrix(listA, listB)` computes a whole block as a chunked GEMM over the
   nonzero support of the weights. `ComputeModeCrossTermIP` uses it when
   `RIFT_PRECOMPUTE_BATCHED_CROSSTERMS=1` or `batched=True`. Default is off.

Changes 1-3 are on by default because they alter no output bit. Change 4 is opt-in
because it does.

## Numerics of the batched path

The reduction order changes: pairwise `np.sum` over the full array becomes blocked GEMM
accumulation over the band. Measured against the shipped loop, worst element relative to
`max|U|` over the block, 21 modes, 1048576 bins:

| weight support | deviation |
|---|---|
| band only, 41% support | 1.4e-15 |
| full support (inverse spectrum truncation on) | 2.7e-15 |

`lnL` is a contraction of these matrices, so a fractional error of 3e-15 on entries whose
scale sets `lnL ~ rho^2/2` gives `|d lnL| ~ 1e-12` nats at `rho = 40`. End-to-end
confirmation is in the results section below.

The batched path re-imposes the `same_waveform_Q` mirror explicitly, so the exact
Hermitian and transpose relations the shipped path guarantees still hold element for
element rather than to rounding.

## Known behaviour difference

Frequency bins where the weight is exactly zero are skipped. Their contribution is
`0.0` in the shipped path, so the value is unaffected, but a non-finite template sample
outside the band would propagate to `NaN` in the shipped path and be dropped here.
Inverse spectrum truncation smears the weights to full support, in which case nothing is
skipped and `band_lo2side, band_hi2side` span the whole array; the support is read off
the weights rather than assumed from `(fmin, fMax)`.

## Provenance of the motivating numbers

The `response_cost` campaign ran without `--inv-spec-trunc-time`, taking the driver
default of 8 s. Production runs set it to 0 (RO, 2026-09-09). Two consequences for the
table at the top of this file: about 190 s of the 1318 s precompute is a term production
does not pay, and the band weights in production are zero outside `[fmin, fMax]`, which
production runs.

## Results

Interleaved arms, `ldas-pcdev2`, container `rift_o4d_cc90-120_cuda128_20260717.sif`,
one intrinsic point, CE+ET+K, IMRPhenomXPHM, seglen 128 s, srate 8192, `--rotation-slow
--l-max 4`, `--inv-spec-trunc-time 0` (production), seed 1002, 2 replicates. Run dirs:
`/scratch/richard.oshaughnessy/precompute_speed/ab/`.

| stage | baseline (s) | batched (s) | speedup |
|---|---|---|---|
| `ComputeModeCrossTermIP`, 260 calls | 679.9, 659.6 | 25.9, 25.7 | 26.0 |
| slow-rotation precompute | 777.8, 758.4 | 137.0, 137.2 | 5.6 |
| long-wavelength precompute | 43.4, 43.0 | 29.4, 29.8 | 1.5 |
| `ComputeModeIPTimeSeries`, 30 calls | 94.4, 95.0 | 93.6, 94.4 | 1.0 |
| `InnerProduct.__init__`, 290 calls | 2.1, 1.0 | 1.1, 1.0 | 1.5 |
| ILE wall | 917.3, 889.3 | 253.8, 253.0 | 3.57 |

Replicate spread is 3.1% on the baseline arm and 0.3% on the batched arm, so the ratio
is not a single-run artefact.

`lnL` was 769.5527137530971 in both baseline arms and 769.5527137530969 in both batched
arms: a difference of 2.3e-13 nats against a reported `sigma_lnL` of 0.1297. The sampler
took an identical path in all four runs, with the same `ntotal` (523199) and the same
`n_ESS` (73.189198442), so the difference is the precompute and nothing downstream of it.

At production settings `InnerProduct.__init__` costs 2 s, not the 221 s in the campaign
profile. That 221 s was the inverse-spectrum-truncation window, which the campaign paid
because it took the driver default.

The data-term overlaps in `ComputeModeIPTimeSeries` are now the largest remaining term,
at 94 s. They are FFT-bound and untouched here.

## Enabling it

    export RIFT_PRECOMPUTE_BATCHED_CROSSTERMS=1

Submit files inherit it: RIFT's condor jobs default to `getenv = *`, the same route
`RIFT_ILE_GPU_FANOUT` uses. There is no CLI flag yet; adding one touches the ILE driver, which this change stays
out of.

## Tried and rejected

- Caching the stacked mode matrix across blocks. Stacking is 0.46 s of the 0.577 s
  batched block, so a cache would give a further 4x, but one entry is 146 MB at l<=4 and
  the combined path in PR #307 would hold 60 of them.
- One GEMM per detector over all basis elements at once, rather than per block. Same
  arithmetic, fewer calls, but it requires changing every response module's caller
  instead of the one function they share.
- A bit-identical batched form. Floating-point multiplication does not reassociate, so
  no reordering of the reduction reproduces `np.sum` exactly.

## Tests

`test_precompute_crossterm_batching.py`, wired into `.travis/test-core-units.sh`. Three
of the four changes claim bit identity, so each is checked against a replay of the exact
code it replaced rather than a tolerance. The default-off behaviour is pinned by a
counter the batched path increments.
