# Finite-size response coefficients: one block, not one sample at a time

`DiscreteFactoredLogLikelihoodFreqResponseNoLoop` is the production finite-size
(`--freqresponse`) likelihood. Everything in it was vectorized over the Monte Carlo block
except the response coefficients `b_p(det, RA, DEC, psi)`, which were built by a Python loop
calling the scalar `response_coefficients` once per sample per detector. That loop cost more
than the rest of the likelihood.

## What the loop was doing

Per sample, per detector, `response_coefficients` called:

| call | depends on the sample? |
|---|---|
| `lal.GreenwichMeanSiderealTime(tref)` | no, `tref` is one scalar for the whole block |
| `slowrot_freqresponse.detector_geometry(det, L_arm)` | no, detector and arm length only |
| `_triad(dec, psi, g)` | yes |
| `_lwl_response(response, X, Y)` | yes |
| arm projections `X.x_arm`, `nhat.x_arm`, ... | yes |
| `finite_size_beta(geom, Qmax)` | yes |

The two sample-independent rows were recomputed for every sample. In a profiled five-detector
CE+ET+K run (282,000 samples) that was 1,409,085 LAL detector lookups for five distinct
answers.

The loop body also read

```python
bvec.setdefault(p, np.zeros(npts_ex, dtype=complex))
```

Python evaluates `setdefault`'s second argument whether or not the key is present, so this
allocated an `npts_ex`-long complex array on every `(sample, p)` iteration: 8,455,844
allocations in that run, 83 s of a 293 s integration.

## What it does now

`slowrot_freqresponse` gained two functions:

- `detector_geometry_cached(det, L_arm)` memoizes `detector_geometry` on `(det, L_arm)` and
  returns read-only arrays, so a caller that writes to one fails instead of corrupting the
  next caller. `detector_geometry` itself is unchanged and uncached.
- `finite_size_geometry_vector(det, ra, dec, psi, gmst, L_arm)` evaluates the triad, the
  long-wavelength contraction and the arm projections for a whole block. `finite_size_beta`
  already accepted arrays and takes its result unchanged.

`factored_likelihood_freqresponse` gained `response_coefficients_vector`, which returns
`{p: (npts_ex,) complex}`. The NoLoop calls it once per detector. The scalar
`response_coefficients` is still shipped and is still the definition of `b_p`; the block
form is checked against it.

This is the same structure `factored_likelihood_with_rotation.rotation_coefficients_vector`
already had for the sidereal-harmonic coefficients, which is why the `--rotation-slow` row
cost 5.4x the long-wavelength baseline while `--freqresponse` cost 137x for six basis
elements.

## Does lnL move?

`finite_size_beta` now squares `zx` through `_csquare`, which writes out CPython's own
complex multiply. numpy's `complex128 ** 2` rounds differently from Python's `complex ** 2`
in about 29% of random samples, at one ulp; `_csquare` reproduces the scalar value bit for
bit on both paths.

After that, `b_0` through `b_3` are bit-identical to the scalar loop for every detector
tested. `b_4` and `b_5` carry `a_x**3` and `a_x**4`, where CPython's `float.__pow__` calls
libm `pow` and numpy's power loop does not; those agree to one ulp on about 18% of samples,
worst relative difference 5e-16. No reassociation of the physics is involved and no
alternative array expression matches libm `pow` without a per-element Python call.

The consequence for the likelihood was measured, not argued. `analyses/slowrot_finite-size`
(RIFT_roboto_paper) builds a self-consistent finite-size injection where truth is the exact
global maximum. Evaluating the NoLoop on a fixed block of extrinsic samples under the two
trees:

| configuration | samples x times | `lnL_t` differing | max abs difference |
|---|---|---|---|
| CE+ET+K, Qmax=4, SNR 30.1, seed 4242 | 2000 x 64 | 0 of 128,000 | 0 |
| CE-ET, Qmax=6, SNR 27.1, seed 909 | 1500 x 64 | 0 of 96,000 | 0 |

lnL is bit-identical on both. The ulp differences in `b_4`, `b_5` are below the rounding of
the sums those coefficients enter.

## Cost

`analyses/response_cost_scaling/run_one.py --response finite --lmax 2 --device gpu --stencil
sinc`, CE+ET+K, IMRPhenomXPHM, seglen 128 s, srate 8192, Qmax 4, on an RTX PRO 4000 Blackwell
(ldas-pcdev11). Stage attribution by that directory's `parse_profile.py`. Arms alternate per
seed.

| seed | baseline `tau_it,like` | block `tau_it,like` | speedup | `tau_it,rest` | `N_it` both arms |
|---|---|---|---|---|---|
| 1002 | 1706.83 us | 37.03 us | 46.1x | 7.2 us | 281,817 |
| 2002 | 1778.03 us | 71.09 us | 25.0x | 16.3 us | 321,687 |
| 3002 | 1755.94 us | 73.08 us | 24.0x | 16.6 us | 403,083 |

The spread is node contention. `tau_it,rest` is the sampler outside the likelihood and
matches within each pair; the block arm tracks it at 4.4 to 5.2x, while the baseline arm has
sd 2.1% across the three because it is bound by Python call overhead.

Provenance, correctness table and the one row that was killed by the per-UID memory cgroup:
RIFT_roboto_paper
`analyses/finite_response_vectorization/RESULTS_2026-09-09_block_response_coefficients.md`.

## Not done

The GPU path still brings `RA`, `DEC`, `psi` to the host (`_h()`) and computes `b_p` in
numpy, then copies `b_p` back. That is one transfer of three float64 blocks per likelihood
call, not one per sample, and it was already there before this change. Moving the geometry
onto the device would need `lal.ComputeDetAMResponse` reimplemented in cupy, and the
long-wavelength baseline `F0` is the one part of the response the module keeps exact against
LAL. The remaining host cost is a fixed number of array operations per block.

PR #307 (`codex/combined-slowrot-freqresponse`) adds a combined rotation-and-finite-size
module with its own block-form coefficients. It does not touch either file changed here.
