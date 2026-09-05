# Retained-grid FFT for ordinary-ILE band-limited time marginalization

## Scope and status

This note covers only the dense `bandlimited` time-marginalization implementation
in `time_marginalization_quadrature.py`.  It does not change the Q_lm time
stencil (`sinc` remains the ordinary-ILE default in the benchmark), the
peak-local implementation, the method selector, or the frozen paper benchmark.

The code is a production-safe optimization candidate: supported CuPy complex128
inputs use a retained-grid chirp-z evaluation, as do NumPy inputs at factor 8 and
above.  NumPy factors 2 and 4 intentionally retain the established full FFT
below a conservative measured CPU crossover.  Every declined or failed optimized
transform also retries the full-padding reconstruction.  Cost selection and
failure retry are separately recorded by `last_report()`.  An optimization
decline is therefore not reported as a waveform/likelihood failure and does not
by itself remove an AV sample.

The numerical-identity and focused-kernel claims below are verified.  Matched
end-to-end AV evidence/posterior runs remain a promotion gate; this note does not
turn the microbenchmark into an evidence claim.

## Exact mismatch at production window sizes

Let the gathered integration window have `n` coarse samples and let `f` be the
derived power-of-two refinement factor.  The boundary construction forms the
literal reflected period

```
[x[0], ..., x[n-1], x[n-1], ..., x[0]]
```

of length `N = 2n`.  The reference implementation zero-pads its spectrum to
`N f`, takes the entire inverse FFT, and retains only
`m = (n - 1) f + 1` forward-window samples.

The two representative NCHUNK=40,000 shapes are:

| cell | n | f | reflected N | reference IFFT Nf | consumed m | factorization |
|---|---:|---:|---:|---:|---:|---|
| 22, srate 4096 | 614 | 64 | 1228 | 78,592 | 39,233 | 78,592 = 256 x 307 |
| Lmax=4, srate 8192 | 1228 | 32 | 2456 | 78,592 | 39,265 | 78,592 = 256 x 307 |

Thus roughly half of the explicitly generated inverse-FFT outputs are discarded.
More importantly, reflection leaves the prime factor 307 in every power-of-two
refinement length.  The exact vendor-library implementation of that nonsmooth
FFT is not assumed here; measured cost, rather than a claim about proprietary
cuFFT internals, is the performance evidence below.

## Retained-grid identity

After the length-N FFT, arrange `N+1` coefficients at consecutive signed
frequencies `k=-N/2,...,+N/2`.  As in the reference implementation, split the
even-period Nyquist coefficient equally between the two endpoints.  The desired
sample `j` is then

```
y[j] = exp(-i pi j/f) / N
       sum(q=0..N) C[q] exp(2 pi i q j/(N f)),   j=0,...,m-1.
```

The sum is a uniform unit-circle chirp-z transform.  Bluestein convolution
evaluates only the requested `m` points.  Its compatible FFT lengths are 40,500
for the 22 cell and 42,000 for the Lmax=4 cell, versus 78,592 in the reference
path.  Chirp phases are reduced exactly modulo `2 N f` in int64 before conversion
to complex128; this avoids the accumulated unit-circle drift of repeatedly
raising one rounded complex root to high powers.

All arrays, coefficient rearrangement, chirps, and FFTs use the caller's `xpy`
backend.  `scipy.fft.next_fast_len` computes one host integer; it does not move
data off a GPU.  Independent rows remain batched.  Chirp plans are reused across
chunks and factors within one marginalization call, then released rather than
held in a process-global GPU cache.

## Why cost still grows with SNR

For the near-Gaussian time peak,
`sigma_t = 1/(2 pi rho sigma_f)`.  The certified resolution requires
`deltaT/f <= sigma_t/2`, so the derived `f` grows approximately linearly with
SNR (in power-of-two steps).  Both the reconstructed grid and the nonlinear
distance/phase likelihood callback contain `m ~ n f` points per refined row.
Consequently the irreducible callback/reduction work grows approximately as
rho, while the reference transform grows as roughly `rho log rho` and also pays
for the discarded reflected half and the nonsmooth FFT length.

NCHUNK=40,000 is not itself an accuracy parameter.  It supplies many rows to the
dense stage, which is divided into about 128-MiB working chunks.  Commit
`70599f1f` already prevents a rare unresolved row from doubling the factor for
the whole group; each row now retires at its own certified factor.  A larger AV
chunk still means proportionally more row transforms/callback evaluations and
can contain more high-factor rows.  This optimization reduces the transform
constant, but intentionally does not alter the SNR-dependent resolution rule or
the number of `sinc`/likelihood evaluations.

## GPU benchmark

Hardware and software: NVIDIA RTX PRO 4000 Blackwell SFF (24,026.7 MiB), CUDA
12.8 runtime, cuFFT 11.3.3, CuPy 14.1.1, SciPy 1.15.3.  Source base was the
immutable ordinary-ILE benchmark commit `476145cb`; candidate source was an
isolated clone based on that commit.  The time-quadrature source in the HM
snapshot `50f470f8` was byte-identical to `476145cb`.  Each arm processed 40,000
row transforms in production-sized batches of 26.  The reported wall interval
excludes Python and RIFT import but includes optimized-plan construction.  Each
arm ran in a fresh process; RSS therefore includes the same RIFT/container
import baseline.

The committed reproducer is `Code/test/benchmark_bandlimited_retained_fft.py`.
Two independent executions gave the wall ranges below; memory columns are from
the committed-reproducer execution.

| cell | retained outputs | full wall (s) | retained wall (s) | paired speedup | host max RSS full/new (MiB) | CuPy pool full/new (MiB) | device delta full/new (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 22, n=614, f=64 | 1,569,320,000 | 6.03--6.79 | 1.77--3.13 | 2.17--3.40x | 483.2 / 483.1 | 277.4 / 55.2 | 296 / 60 |
| Lmax=4, n=1228, f=32 | 1,570,600,000 | 6.12--6.69 | 2.28--2.89 | 2.32--2.69x | 480.7 / 484.1 | 285.1 / 66.8 | 306 / 72 |

The host RSS difference is noise at an import-dominated baseline.  The device
figures demonstrate that the explicit retained-grid transform does not hide a
larger chirp/workspace or CPU transfer: its CuPy-pool footprint is 20--23% of
the full-padding arm in these cells.

A pre-commit sweep over every factor 2, 4, 8, 16, 32, and 64 processed 40,000
rows at each of `n=614` and `n=1228`, using the same 128-MiB-derived batches.
The retained path was faster in all 12 cells; the smallest measured speedup was
1.53x (n=1228, factor 2).  Thus applying it to every supported refinement factor
does not hide a measured low-SNR crossover on this device.

The corresponding four-worker CPU sweep did have a small-grid crossover.  In
balanced repeats the retained factor-2 transform cost 1.03--1.9 times the full
FFT for `n=614,1228,2457`, and factor 4 was 1.14 times slower at `n=2457`
(although faster for the prime-307 lengths).  Since these grids are cheap and
not the high-SNR bottleneck, NumPy conservatively selects the full transform at
both factors 2 and 4.  This selection is telemetry, not a failed optimization;
CuPy continues to use retained evaluation because its measured crossover is
below factor 2.

Fixed-input parity used 32 full-band complex rows and a smooth nonlinear map
`100 logaddexp(0, Re(kappa))` before trapezoidal time integration and a
log-sum-exp evidence-like reduction:

| cell | max abs delta kappa | max abs delta row lnL | delta aggregate lnZ |
|---|---:|---:|---:|
| 22, n=614, f=64 | 7.71e-15 | 2.27e-13 nat | -1.14e-13 nat |
| Lmax=4, n=1228, f=32 | 8.04e-15 | 3.98e-13 nat | +5.68e-14 nat |

CPU tests also compare random Nyquist-populated rows at `n=614,1228,2457`
against the full-padding reference.  The largest observed complex discrepancy
in the wider diagnostic sweep (`n=3` through 2457, factors 2 through 64) was
`5.7e-15`.

A matched bounded 22 ordinary-ILE integration smoke used `bandlimited+sinc`, SNR
label 160, seed 99002, and `NMAX=NCHUNK=4000`.  Both arms completed 4000 AV
evaluations.  Full/new wall was 19.88/19.67 s, host max RSS was
1491.6/1493.5 MiB, and the reported log integral differed by `7.3e-12` nat
(13224.475448007970 versus 13224.475448007977).  The deliberately tiny run had
ESS 1.73 and Pareto k-hat 11 in both arms, so it is an integration smoke, not
acceptable evidence or a throughput benchmark.  The Lmax=4 claim remains the
fixed-shape kernel/parity result above; a matched converged HM AV run is still in
the promotion gate.

## Failure and telemetry contract

The retained path is certified only for NumPy/CuPy, complex128 spectra, even
reflected periods, and power-of-two factors above one whose modular chirp indices
fit exactly in int64.  Other combinations, plan-construction failures, and
transform exceptions enter the full-padding reference path.  A RuntimeWarning is
emitted once per reason per call when warning policy permits it; warnings promoted
to exceptions are contained so diagnostics cannot drop the point.

`last_report()` records:

- `bandlimited_fft_strategy`: retained, full selected, full fallback, mixed, or
  unused;
- retained/selected/fallback batch and row-transform counts;
- a reason map for an intentional full-FFT cost selection;
- the fallback exception/reason map;
- reference full length, retained-grid length, compatible convolution length,
  largest factor, and number of per-call plans.

The likelihood callback is invoked outside the guarded transform helper.  Its
exception is therefore not swallowed or relabeled as an FFT decline.  Tests pin
both directions: forced optimized failure returns the finite full-sinc result
with provenance, while a forced callback failure retains its original identity.

## Validation and promotion gate

The focused suite passes on the actual CuPy backend, including GPU/CPU parity,
unsupported-factor fallback, warnings-as-errors, and callback-failure identity.
The complete `test_time_marginalization_quadrature.py` gate passed 90 tests.

Before claiming an end-to-end AV speedup or unchanged scientific evidence,
run matched old/new 22 and Lmax=4 ILE cells with identical seeds, data, sinc
stencil, NCHUNK, and stopping rules.  Require zero unplanned fallback rows,
record the factor histogram and transform provenance, compare pointwise replayed
lnL where available, and require delta-lnZ to be negligible relative to the
combined Monte Carlo uncertainty.  That stochastic validation is deliberately
not inferred from the transform-level `delta lnZ` above.
