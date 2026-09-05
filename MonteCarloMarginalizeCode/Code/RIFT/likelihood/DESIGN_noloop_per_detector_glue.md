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
