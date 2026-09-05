# NoLoop: what the detector loop was recomputing, and why the split is bitwise exact

Scope: `DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`, the maintained GPU
likelihood (`--vectorized --gpu`). This note records *why* the source geometry was
lifted out of the detector loop, and the constraint that decided the implementation.

## The measurement that motivated it

Stage attribution inside NoLoop, RTX PRO 4000 Blackwell (sm_120), cupy 14.1.1 /
CUDA 12.9, ILE-GPU-Paper demo, `--interpolate-time nearest`, `--n-chunk 10000`,
1000 calls, each stage device-synced (which inflates the total by 2.5%):

| stage | share of NoLoop |
|---|---|
| `simps` | 32.6% |
| `SphericalHarmonicsVectorized` | 22.2% |
| `ComputeDetAMResponse` | 17.1% |
| residual (`kappa_sq`, `rho_sq` einsums, `exp`/`log`, allocation) | 18.8% |
| `TimeDelayFromEarthCenter` | 3.6% |
| `Q_inner_product_cupy` (the CUDA kernel) | **5.7%** |

The hand-written kernel is a twentieth of the cost; the rest is cupy glue. The three
geometry stages total ~43% and act on `(n_extrinsic,)` arrays — a few hundred KB. Time
spent there is therefore kernel-launch and op-count bound, not bandwidth bound: long
chains of small elementwise operations. Each was being rebuilt once **per detector**
although none of them depends on the detector.

Independent confirmation that the per-call cost is launch-bound: sweeping `--n-chunk`
on an RTX 3080 fits `cost = 5.7 ms + 0.61 us x n_chunk`, i.e. at `--n-chunk 10000`
roughly half of every call does no more work for a larger batch.

## What is actually per-detector

Only the contraction with the interferometer's own constants:

- `ComputeDetAMResponse` — six trig evaluations and twelve elementwise combinations
  build the `(X, Y)` polarization basis from RA/DEC/psi/GMST. Only the two `inner`
  contractions against `detector_response_matrix` are per-detector.
- `TimeDelayFromEarthCenter` — `ehat_src`, the unit vector towards the source, is
  source-only. Only the `inner` against `detector_earthfixed_xyz_metres` is not.
- `SphericalHarmonicsVectorized` — depends on `(modes, incl, phiref)`. Detectors share
  a mode list in practice, since the modes come from one waveform.
- `DetectorPrefixToLALDetector` plus two host-to-device transfers were also being
  redone every call, for values fixed for the lifetime of the process.

## The constraint: bitwise, not approximately

This is a likelihood behind published results, so the split had to leave lnL
*bit-identical*, which rules out the obvious vectorization. `ComputeDetAMResponse`'s
docstring advertises a leading detector axis, but that path does not actually work —
`X * xpy.inner(X, R)` fails to broadcast for `(n_ex, 3)` against `(n_det, 3, 3)`. The
natural fix, one batched `einsum` over stacked detectors, reassociates the contraction
and agrees only to ~4e-16. Fewer launches, but not the same number.

So the per-detector halves keep the identical `inner` calls in the identical order and
only the source-only prologue is shared. `test/test_vectorized_lal_tools_split.py`
pins that with `array_equal`, not a tolerance, on three real interferometer geometries.

## Sharing hazards, and how they are handled

- **The phase-marginalization branch mutates `Ylms_vec` in place** (`[:, 1] = conj(...)`),
  and `rho_sq_det` above it must see the un-conjugated array. A shared array would leak
  one detector's conjugation into the next detector's self-term. Each detector gets a
  copy when `phase_marginalization` is on; a copy of `(n_extrinsic, n_lms)` is still far
  cheaper than rebuilding the harmonics.
- **`lookupNKDict[det]` may be a device array**, so comparing mode lists per call would
  force a synchronization. `_mode_list_key` memoizes a hashable host key on the array
  *object*, keeping a reference so `id()` cannot be recycled. Detectors with genuinely
  different mode lists therefore get a correct, merely unshared, result.
- `TimeDelayFromEarthCenterPrecomputed` divides in place into the result of `inner`,
  which is a fresh array — not into the shared `ehat_src`. The test pins that too.

## Measured effect

Same captured NoLoop arguments replayed through both trees (Blackwell, `nearest`,
`--n-chunk 10000`, 100 calls per timing, 3 repetitions), output bitwise identical:

| configuration | before | after | |
|---|---|---|---|
| H1 L1 (2 detectors) | 12.41 ms/call | 10.26 ms/call | **-17.4%** |
| H1 L1 V1 (3 detectors) | 16.81 ms/call | 13.06 ms/call | **-22.3%** |

The saving scales with detector count, as it should: the shared prologue is paid once
instead of `n_det` times. The CPU (`xpy=numpy`) path is unchanged within noise — it is
dominated by the `(n_extrinsic, npts, n_lms)` window build, not by this glue.

## What this deliberately does NOT do

- `simps`, the single largest stage, is untouched. It is a fixed linear functional, so
  it could be one `gemv` against precomputed weights — which is what the fused calmarg
  path already does via `w_t = simps(eye(npts))`. That changes summation order and so
  is not bitwise; it belongs in its own change with its own accuracy argument.
- The post-kernel reduction is untouched. Routing `n_cal == 1` through the existing
  `Q_fused_calmarg` kernel measured a further ~24%, agreeing within Monte Carlo error
  but not bitwise. Also a separate change.

---

# Round 2: the accumulators, and a per-operation cost table

After the hoist above, stage attribution became misleading: it device-syncs after every
wrapped call, so a function called once per *detector* is charged three times the sync
penalty of one called once per *likelihood call*, and the mode inflated the total by 26%.
The numbers below come instead from timing each operation in a tight loop with a single
sync (`bench/micro_ops.py` in the profiling archive), at production shapes
`n_extrinsic = 10000`, `npts = 614`, three detectors, on an RTX PRO 4000 Blackwell. They
sum to 11.87 ms against a measured 12.10 ms/call, i.e. they account for 98% of the
function.

| operation | ms/op | x per call | ms per NoLoop call |
|---|---|---|---|
| `kappa_sq += Q_prod * invDist` | 1.519 | 3 | **4.556** |
| `ComputeDetAMResponsePrecomputed` | 0.628 | 3 | 1.885 |
| `simps` over `(10000, 614)` | 1.765 | 1 | 1.765 |
| `Q_inner_product_cupy` | 0.391 | 3 | 1.173 |
| `kappa.real - 0.5*rho` (stride-0 view) | 0.619 | 1 | 0.619 |
| `exp` in place | 0.499 | 1 | 0.499 |
| `SphericalHarmonicsVectorized` | 0.401 | 1 | 0.401 |
| `SourcePolarizationBasis` | 0.367 | 1 | 0.367 |
| `max(axis=-1, keepdims)` | 0.264 | 1 | 0.264 |
| `TimeDelayFromEarthCenterPrecomputed` | 0.062 | 3 | 0.187 |
| `SourcePropagationDirection` | 0.127 | 1 | 0.127 |
| `rho_sq` vector accumulate | 0.008 | 3 | 0.024 |

The data term dominates, and it dominates because of how it is *stored*, not what is
computed into it.

## rho_sq was 49 MB of duplicated scalars

`rho_sq` is the `<h|h>` term. Every detector contributes `rho_sq_det` of shape
`(npts_extrinsic,)` — it has no time dependence at all — and that was being broadcast
into a dense `(npts_extrinsic, npts)` accumulator: a 49 MB zero-fill, then one 49 MB
read-modify-write per detector, to store `npts` identical copies of each value.

It is now summed as a vector and exposed as a stride-0 `broadcast_to` view. Measured:
dense accumulate 0.121 ms/detector against 0.008 for the vector, and the downstream
`kappa.real - 0.5*rho` drops from 0.758 ms to 0.619 ms because the subtrahend now fits
in cache. The calibration path already did exactly this for `rho_sq_cal`; this brings
the ordinary path in line.

Consumers that need real backing memory go through `_dense_rho_sq()` and pay what they
paid before. There are two classes: the fused calmarg CUDA kernels, which index raw
device pointers and would read garbage from a stride-0 view, and the non-Simpson
quadrature helpers, which are free to write into what they are handed.

## kappa_sq did not need to start at zero

`kappa_sq` is 98 MB of complex128. It was zero-filled, then for each detector the
distance scaling allocated another full-size temporary and the result was accumulated in
— so a three-detector network paid one 98 MB fill, three 98 MB temporaries, and three
98 MB read-modify-writes. It now scales the Q kernel's own freshly allocated output
buffer in place and takes the first detector's buffer as the accumulator.

The one arithmetic caveat: `0.0 + x` is exactly `x` for every finite `x`, and for Inf and
NaN, but `0.0 + (-0.0)` is `+0.0` while starting from the buffer preserves `-0.0`. A
signed zero in `kappa_sq` is unobservable downstream — it survives `.real`, and
`exp(-0.0) == exp(+0.0) == 1.0` — so this is noted for completeness rather than as a
behavioural difference.

## Measured, cumulative, all bitwise

Same captured NoLoop arguments replayed through each tree, H1 L1 V1, `nearest`,
`n_chunk 10000`, 100 calls per timing:

| tree | ms/call | vs base |
|---|---|---|
| `rift_O4d` | 17.06 | — |
| \+ hoist source-only geometry | 13.08 | −23.3% |
| \+ `rho_sq` as a vector | 12.10 | −29.1% |
| \+ `kappa_sq` in-place | 11.12 | **−34.8%** |

`test/test_noloop_accumulator_shapes.py` pins both accumulators against a reference
implementation written the original way, with `array_equal` rather than a tolerance, at
one, two and three detectors. The reference passes against the unpatched tree as well,
which is what makes it a check on the change rather than a transcription of it.

## Round 3: the time integral, and the one change that is not bitwise

`simps` was 1.765 ms/call. It is a fixed linear functional at fixed `dx`, so it equals a
matrix-vector product against precomputed weights — measured at **0.049 ms**, a 36x
saving. The fused calmarg path already built exactly those weights by hand with
`w_t = simps(eye(npts))`; that is now a single cached helper, `_simps_weights`, so the
tree carries one definition of the equivalence instead of two.

A `gemv` reassociates the summation, so unlike everything above this is **not** bitwise.
It is the same RULE: the weights come from the very `simps` implementation the call site
would otherwise have used, so the `even='avg'`-versus-Cartwright distinction that
separates the vendored GPU copy from scipy's is preserved exactly. Only the order of the
additions changes.

**Measured discrepancy**, over 10 000 real extrinsic samples spanning lnL from
-2.2e6 to +116:

| | |
|---|---|
| max abs difference | **2.8e-14 nats** |
| median abs difference | exactly 0 |
| max relative difference | 7.1e-14 |
| float64 rounding scale of the values themselves (`eps x max abs lnL`) | 4.9e-10 |

The difference is below the rounding scale of the quantities being compared, and both
paths are deterministic run to run. For physical scale, the errors already present in
this integral are between eleven and sixteen orders of magnitude larger: the two `simps`
variants in this tree disagree by **0.405 nats** on an under-resolved peak, and the
`nearest` time stencil costs **200-443 nats at SNR 100** (`--interpolate-time` help text,
issue #233). Simpson's real accuracy limit here is sub-sample resolution of a peak whose
width is set by the signal rather than by the sample rate — which is what the
`time_quadrature` and stencil work addresses — not the order of its additions.

`test/test_noloop_accumulator_shapes.py` splits the two guarantees rather than blurring
them: the accumulators are checked with `array_equal` at `return_lnLt=True`, before the
integral, and the quadrature is checked separately against `simps` at a tolerance far
tighter than anything physical. A failure of the second means the rule changed, not that
rounding drifted.

## Where the remaining time goes

After all three rounds, at three detectors and `n_chunk 10000`, no single item dominates:
the Q kernel (~1.2 ms), the detector-response contraction (~1.9 ms), and the
`exp`/`max`/subtract reduction (~1.4 ms) are the three largest, and none has an obvious
order-preserving win left. The response contraction is the best remaining candidate —
four `inner` calls per detector against a 3x3 matrix — but batching it over stacked
detectors reassociates, for a much smaller payoff than this round bought.
