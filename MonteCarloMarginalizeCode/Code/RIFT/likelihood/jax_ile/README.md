# `jax_ile` — an AD-compatible JAX reimplementation of the ILE extrinsic likelihood

A `jax.numpy`, automatic-differentiation-compatible reimplementation of RIFT's
ILE extrinsic likelihood, mirroring the production
`factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`
(the "...NoLoop", array-vector, **fused** code branch, `n_cal == 1`).

The goal: a likelihood that is **differentiable**, `jit`/`vmap`-able, and
exact-to-the-reference, so the narrow extrinsic peak can be climbed and sampled
with gradient-based methods (NUTS, flowMC) instead of brute-force Monte Carlo,
and so downstream AD applications (Fisher forecasts, etc.) come for free.

## What is reused vs. new

**Reused unchanged from production RIFT** (deliberately *not* reinvented — frame
reading and inner products are fiddly and already correct):

| step | production function |
|------|---------------------|
| frame reading | `lalsimutils.frame_data_to_non_herm_hoff` |
| PSD handling | `lalsimutils.get_psd_series_from_xmldoc` / `resample_psd_series` |
| waveform + `<h_lm(t)\|d>`, `<h_lm\|h_l'm'>` | `factored_likelihood.PrecomputeLikelihoodTerms` |
| array packing + epoch | `factored_likelihood.PackLikelihoodDataStructuresAsArrays` |

**New (pure JAX):** the cheap extrinsic → lnL combination — detector antenna
response, geometric time delay, spin-(-2) spherical harmonics, the
`kappa`/`rho^2` assembly, continuous time-shift interpolation, time
marginalization, and **analytic distance marginalization**.

## Modules

- `detector.py` — `compute_detamresponse`, `time_delay_from_earth_center`
  (JAX ports of `vectorized_lal_tools`, validated to ~1e-16).
- `spherical.py` — spin-(-2) spherical harmonics for `l = 2 .. 8` (coefficients
  imported from the production table; `python spherical.py` validates every
  `(l,m)` against `lal.SpinWeightedSphericalHarmonic` to ~2e-16).
- `core.py` — the fused likelihood:
  - `fused_log_likelihood(...)` — time-marginalized lnL at fixed distance over
    `(ra, dec, psi, incl, phiref, distMpc)`.
  - `fused_log_likelihood_distmarg(...)` — **distance- and time-marginalized**
    lnL over the 5 angular parameters (regulates the amplitude degeneracy; see
    below).
  - `make_distance_grid(...)`, `JAXLikelihoodData`, `build_likelihood_data`.
- `wrapper.py` — `build_data_from_precompute` (runs the production precompute +
  packing and returns a device-resident `JAXLikelihoodData`), and the
  convenience classes `JAXExtrinsicLikelihood` (6-D, value/grad/Fisher) and
  `JAXDistanceMarginalizedLikelihood` (5-D angular, value/grad/Fisher).

## Validation

`test/jax/test_jax_likelihood.py` (synthetic arrays) and
`test/jax/test_jax_endtoend.py` (synthetic injection through the *real*
`PrecomputeLikelihoodTerms`) check:

- JAX `interp="nearest"` reproduces `DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`
  (`xpy=np`) to **~1e-13** (machine precision);
- `interp="linear"` gradients match finite differences to ~1e-8;
- `jit` / `vmap` / `grad` / `hessian` all execute and stay finite.

```
PYTHONPATH=<...>/MonteCarloMarginalizeCode/Code \
  python test/jax/test_jax_likelihood.py
  python test/jax/test_jax_endtoend.py
```

## "Epoch" and the two time windows (important)

The rholm timeseries sample `k` is GPS time `epoch_det + k*deltaT`.  The window
time-bin maps to the *continuous* fractional sample position
`((tref - epoch_det) + tau_det(RA,DEC) + tvals[0]) / deltaT + t`, matching the
reference `ifirst = round(pos)` at `t=0`.  Keeping `pos` continuous (linear
interp) is what makes the sky-location dependence differentiable.

There are **two** windows (as in the production driver):
- `--internal-data-storage-window-half` (default 0.15 s) — the rholm **buffer**.
- `--data-integration-window-half` (default 0.075 s) — the **marginalization**
  window (`tvals`).

The buffer must exceed the integration window by the maximum per-detector
time-delay excursion, or the sliding window runs off the buffer.  Positions
outside the buffer contribute **zero** (not a flat edge-clamp), matching the
production "over-running window zeros" semantics and avoiding a spurious peak.

## The distance / amplitude degeneracy (and the fix)

The bare factored likelihood, optimized freely over distance, is
`lnL_opt = |Re kappa|^2 / (2 rho^2)` and **diverges** on thin slivers where the
template power `rho^2 -> 0` (e.g. inclination → π, antenna nulls).  Production
ILE never *maximizes* — it *integrates* against the volumetric prior, where
those slivers carry negligible volume.

`fused_log_likelihood_distmarg` marginalizes distance analytically (numerical
quadrature over a distance grid with the `p(d) ∝ d^2` prior) **before** the time
integral — exactly the ordering of the production `distmarg_loglikelihood`.  The
result is smooth, bounded, and peaks at the correct sky location, and is the
right object for gradient-based exploration.

## Driver

`bin/integrate_likelihood_extrinsic_jax` mirrors the ILE CLI/output conventions
and uses the JAX likelihood.

**Drop-in argument compatibility.**  Every
`integrate_likelihood_extrinsic_batchmode` option is accepted, so the driver can
be substituted directly into an existing production command line.  Implemented
options are used; unimplemented non-critical options are silently accepted and
reported (`Note: ... accepted but IGNORED ...`); options that would silently
change the *science* if ignored — calibration marginalization (`--calibration-*`),
ROM-basis waveforms (`--rom-*`), NR templates (`--nr-*`),
supplementary-likelihood factors, `--zero-likelihood`, `--maximize-only` — cause
a hard failure instead of a misleading result.

**Intrinsic input + batch.**  `--sim-xml` / `--sim-grid` load intrinsic
templates (with `--event` / `--n-events-to-analyze` selecting a slice), exactly
as ILE does; tidal `--eff-lambda`/`--deff-lambda` are converted to
`lambda1,lambda2`.  Multiple events are processed in a batch loop, each writing
`<output>_<index>_.dat` (and `_samples.dat`).  `--inj-mode` synthesizes
zero-noise data for self-tests (single event).

Modes (`--mode`):

- `prior-mc` — brute-force importance sampling from the physical prior (robust, slow).
- `laplace-is` — prior-seeded adaptive Gaussian importance sampling (default).
- `nuts` — single-chain gradient NUTS (numpyro) over the distance-marginalized
  angular posterior, seeded at the best prior draw.
- `multistart-nuts` — **mode-covering** NUTS: a pilot prior scan picks several
  well-separated high-lnL seeds (one per resolvable sky mode), runs a NUTS chain
  from each, pools them, and forms a Gaussian-**mixture** importance estimate of
  the evidence (one component per mode → usable `neff` on a multimodal posterior).
- `flowmc` — **normalizing-flow sampler** (flowMC RQSpline+MALA): trains a flow on
  the multimodal target using the exact JAX gradient, captures all modes at once,
  with a flow-seeded importance evidence.  Fast and the recommended sampler.
- `map` — gradient-ascend the angular peak + report Fisher.

`multistart-nuts`, `flowmc` and `nuts` require `--distance-marginalization` (they
sample the 5-D angular posterior).  Implemented in `samplers.py`.

Validation (standard injection, truth sky RA,DEC=(1.20,-0.40)): both samplers
recover the truth sky and **agree on the evidence** — `multistart-nuts`
`logZ≈524.09` (`neff≈200`) and `flowmc` `logZ≈524.16` (`neff≈20`, ~4 min on CPU)
— versus a single Gaussian / single chain that gave `neff≈2-4`.  (The highest-lnL
*orientation* differs from the injected one: the ψ/φ_ref polarization-phase
degeneracy admits an equal-or-higher-likelihood orientation; the sky is what is
recovered.)

Self-test (no frames needed):

```
PYTHONPATH=<...>/Code python bin/integrate_likelihood_extrinsic_jax \
   --inj-mode --mass1 35 --mass2 30 --spin1z 0.1 --spin2z -0.2 \
   --mode nuts --d-max 5000 --save-samples --output-file out
```

Output: `out_0_.dat` (`event_id m1 m2 s1x..s2z lnL sigma_lnL ntotal neff`) and,
with `--save-samples`, `out_0_samples.dat`.

## Status and next steps

**Done & validated:** the AD likelihood core (1e-13 vs reference), gradients,
distance marginalization (vectorized, fast reverse-mode), the wrapper, the CLI
driver + ILE-format I/O + full ILE argument compatibility + batch processing,
spherical harmonics l=2..8 (vs lal), network-frame sky coordinates, and
multimodal sampling via **multi-start NUTS** and **flowMC** (both recover the
truth sky and agree on the evidence; see the Driver section).

**Multimodality** — the detector time-delay ring (discrete sky modes) plus the
phase/polarization degeneracy — is the central difficulty and is now handled by
the mode-covering samplers above.  Further hardening available to compound:
- the network-frame sky coordinates (`coordinates.py`) fold the time-delay ring
  onto a constant-polar-angle line; sampling the sky in `(cos theta_n, phi_n)`
  (where the sky prior is simply uniform, since rotation preserves the sphere
  measure) should sharpen mode separation — wiring this into the samplers is the
  natural next step;
- `polarization_phase_fold` folds the ψ/φ_ref quadrupole degeneracy into a
  fundamental domain.
- flowMC is the fast, recommended sampler; multi-start NUTS is the slower but
  gradient-exact cross-check (NUTS on CPU is still costly — GPU would help).

Not yet ported (structured for): in-loop calibration marginalization
(`n_cal>1`) and the lookup-table distance marginalization (we use direct grid
quadrature instead, which is AD-friendly and needs no precomputed table).
