# Simultaneous slow rotation and finite-arm response

## Scope

`integrate_likelihood_extrinsic_batchmode --rotation-slow --freqresponse` now selects
one compound response model. It is intended for long, loud BNS-like signals, including
strongly precessing systems and higher modes. The response operators act on each full
inertial-frame mode, so the implementation makes no per-mode stationary-phase or
time-frequency-track approximation.

This is not the economical path for short eccentric BBH mergers. Use `--freqresponse`
alone there unless Earth rotation is independently relevant.

## Factorization

The finite-arm response already has the form

```text
F(f,t) = sum_b beta_b(t) W_b(f).
```

`beta_0` is the exact LAL long-wavelength response and `beta_(1+q)` contains an
arm-projection polynomial of order `q`. Under Earth rotation these coefficients have
finite sidereal half-widths 2 and `q+2`, respectively. A small exact DFT recovers their
Fourier coefficients. Composing the slow-delay expansion gives elementary templates

```text
chi_(b,p,n) = M_n d_t^p [W_b h_lm].
```

The coefficient half-width is `width(beta_b)+p`. Conjugation reflects only the sidereal
index, `(b,p,n) -> (b,p,-n)`, because each `W_b` is Hermitian.

## Cost and controls

The number of compound elements is

```text
sum_b sum_(p=0)^pmax [2 (width(beta_b)+p) + 1].
```

At the current defaults (`Qmax=4`, `pmax=0`) this is 50 elements and 2500 ordered U/V
pairs per detector. At `pmax=1` it is 112 elements and 12544 pairs. The driver prints
both counts before integration. Start with `--rotation-p-max 0` and the smallest
`--freqresponse-qmax` justified by a response-convergence check.

## JAX implementation

The JAX library now supports `feature="rotation_freqresponse"`. It consumes the same
production precompute and dense U/V bank as conventional ILE, evaluates the compound
coefficients in `jax.numpy`, reflects `(b,p,n) -> (b,p,-n)`, and applies the arrival-time
post-phase to both likelihood terms. The production JAX driver selects the individual or
compound banded builder from `--rotation-slow` and `--freqresponse`; these flags are no
longer compatibility no-ops.

For the compound norm, JAX contracts one sidereal-difference bucket at a time. This avoids
materializing several full `(A,A,S)` arrays: peak pair scratch scales with the largest
harmonic bucket rather than all `A^2` pairs. The driver reports `A`, `A^2`, and persistent
device-bank storage before sampling.

## Initial usability profile

`test/jax/profile_rotating_freqresponse.py` profiles both conventional and JAX contractions
using a one-detector IMRPhenomD `(2,+/-2)` model. On one constrained CPU core, 32 extrinsic
samples, `pmax=0`, the measured progression was:

| Qmax | A | pairs | precompute (s) | conventional eval (s) | JAX compile (s) | warm samples/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 100 | 0.30 | 0.006 | 4.26 | 77,100 |
| 1 | 17 | 289 | 0.82 | 0.006 | 4.90 | 44,800 |
| 2 | 26 | 676 | 1.93 | 0.010 | 6.65 | 28,000 |
| 3 | 37 | 1369 | 3.50 | 0.015 | 8.60 | 18,300 |
| 4 | 50 | 2500 | 6.51 | 0.021 | 13.0 | 9,640 |

The `Qmax=4` warm rate is after memory bucketing; it trades about 15% CPU throughput for a
substantially smaller production-chunk temporary footprint. With delay drift, `pmax=1`
gave `A=24`/576 pairs/24,300 samples/s at `Qmax=0`, and `A=40`/1600 pairs/11,500 samples/s
at `Qmax=1`. These are microbenchmark numbers, not a detector-count or BNS-duration cost
model. The first science runs should use `pmax=0` and establish Qmax convergence; enable
`pmax=1` only after showing that delay drift changes the target high-SNR likelihood.

## Validation

`test_slowrot_rotating_freqresponse.py` checks the exact compound basis roster, sidereal
reconstruction of every finite-response coefficient, delay-order band support, and the
full precompute/NoLoop reduction to the existing finite-response likelihood at zero
sidereal rate, plus the Cauchy--Schwarz bound at zero and physical sidereal rates. The
existing `test_slowrot_noloop.py` remains the regression for the shared rotation
contraction.
