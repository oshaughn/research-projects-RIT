# `jax_ile` — an AD-compatible JAX reimplementation of the ILE extrinsic likelihood

> **`--save-samples` output is a FAIR DRAW**, not the raw sampler cloud:
> equal-weight rows, no weight column, and the same columns as before *for a
> given `--mode`* (different modes export different column sets — see the Driver
> section). A second header line records the mode and the export ESS, e.g.
> `# mode=laplace-is fairdraw: ESS=5.5 n_in=300000 n_out=9`. **Check that ESS
> before trusting a file**: a low-ESS export is not a usable posterior sample
> however it is drawn, and the driver warns on stderr when it is below 200.
> When the weights admit no fair draw at all (degenerate/unnormalizable), the
> event fails and **no samples file is written** (any stale one at that path is
> removed) — there is no mode in which this product holds unreweighted rows.
> That refusal is checked *before* the `<output>_<index>_.dat` result row is
> written, so a failed event leaves **no result row either** (a stale one is
> removed too): with `--soft-fail-event-range` the batch goes on, and a row left
> behind would be collected as a successful integration.

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

### Time quadrature

All JAX likelihood wrappers accept the conventional ILE keyword
`time_quadrature={"simpson","bandlimited"}`.  Simpson remains the default.
The opt-in `bandlimited` path is currently supported by
`JAXExtrinsicLikelihood`, including analytic phase marginalization.  It forms
the endpoint-nonduplicating even extension
`[kappa[0], ..., kappa[-1], kappa[-2], ..., kappa[1]]`, FFT-interpolates it,
applies the phase reduction on the
fine grid, and integrates the original closed interval with a stable trapezoid
rule.  The per-row power-of-two factor is derived from fine-grid peak curvature,
remeasured after interpolation, and doubled until the integral agrees within
1e-3 nat.  Row-local `lax.map` execution bounds scratch memory independently of
the sampler batch.  There is deliberately no public factor knob; a row that
cannot meet the criterion fails closed.

The supported signal regime assumes spectral headroom below the sampled
Nyquist frequency and negligible likelihood mass at both ends of the short
integration window.  The latter is checked on the refined grid: either endpoint
must be at least 15 natural-log units below the peak, otherwise `bandlimited`
fails closed rather than trusting a boundary extension that can affect the
answer.  Increase the physical time window or use Simpson when this diagnostic
fires.

The primitive gather includes support outside that window.  Its initial guard
is the established half-window default rounded up to a power of two; one guard
doubling is gathered at the same time.  A raised-cosine pad acts only across
the support samples, reaching exactly one at the integration crop and zero
with zero slope at the remote even-reflection turns.  The value is accepted
only when both guard widths agree within 1e-3 nat, independently of the fine
quadrature-factor doubling check.  Thus short-window truncation and fine-grid
resolution have separate certificates.

The JAX driver derives this support requirement before waveform precompute and
widens `--internal-data-storage-window-half` when necessary.  It includes the
full certified guard, a conservative 50 ms detector-delay allowance (larger
than the Earth-diameter light time), and the
largest shipped interpolation stencil.  The accumulator also validates every
guarded gather index per row; missing support produces a fail-closed likelihood
instead of inheriting the ordinary gatherer's out-of-buffer zero fill.  The
baseline and banded finite-size/frequency-response accumulators enforce the
same check; rotation remains refused because its norm depends on arrival time.
The
curvature-derived starting fine factor is capped at 1024 and certified once at
2048; a sharper row is refused with guidance to increase the input/rholm sample
rate rather than allocating multi-gigabyte FFT branches.

Distance, phi, psi, exact-angle, and Laplace-marginalized wrappers currently
refuse `bandlimited`.  Those nonlinear reductions generate time harmonics, so
interpolating their already-reduced `lnL(t)` can converge to the wrong function;
they require endpoint-specific primitive refinement before they can safely opt
in.  They continue to use the unchanged Simpson default.
The driver exposes the same public spelling as conventional ILE:
`--time-marginalization-quadrature`.  `--interpolate-time` is an alias for the
JAX-native `--interp` with conflict detection.  Conditional nuisance recovery
is outside this implementation.  For drop-in CLI compatibility,
`--resample-time-marginalization`, `--srate-resample-time-marginalization`, and
`--time-posterior-export` are accepted and reported as ignored: JAX ILE's
sample export keeps time terminally marginalized rather than reconstructing one
conditional time per exported row.  This intentionally differs from
conventional ILE's XML export semantics, but a high-level DAG can swap
executables without dying during option parsing.

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

**Efficiency / robustness options:**
- **Flow re-use across a batch** (`--mode flowmc` + `--n-events-to-analyze`) —
  **OFF by default; opt in with `--flow-reuse`.**  When enabled, the trained
  normalizing flow is bootstrapped from one intrinsic template to the next (its
  NF weights warm-start the next event and its posterior draws initialize the
  chains).  `--no-flow-reuse` is accepted and now restates the default.

  **Do not enable it for any run whose extrinsic SAMPLES are used.**  Re-use
  contracts the posterior in later slots: across an 8-event batch at two seeds,
  psi fell to ~40% of its no-re-use width by slot 7 on both seeds, with slot 0
  (no re-use yet) at ~1.0 as a control, and inclination to 0.49/0.61.  Confirmed
  against independent per-slot references, not just arm-vs-arm.  It also
  reproduces an earlier, independent measurement (mean incl 0.5795 → 0.3465,
  sd(psi) 0.9122 → 0.3738) that caused an amortization claim to be retracted from
  the companion paper.

  *What the earlier validation actually showed.* `test/jax/test_flow_reuse.py`
  checks that a re-used run **recovers the truth sky with neff ≥ the fresh run**.
  That remains true and is not contradicted here — it tests sky location and an
  evidence-side neff, neither of which is posterior *width*.  The contraction is
  in the width of the orientation parameters, an observable that check does not
  look at.
- **Network sky coordinates** (`--sky-coordinates network`, `multistart-nuts`
  only): sample the sky in the two-detector baseline frame `(cosθ_n, φ_n)` to
  fold the time-delay ring (the prior stays uniform there).  Falls back to
  equatorial if fewer than two detectors.
- **Variable / single-detector networks**: the likelihood and samplers handle
  any number of detectors (including one — rare but supported); only the network
  sky frame needs ≥2 detectors and degrades gracefully when it can't be built.

**High-SNR benchmark:** `test/jax/benchmark_snr_sequence.py` builds injections at
network SNR 40,80,160,320,640 (by scaling distance), runs one flowMC evaluation
per source (threading the re-used flow), and records sky recovery / evidence /
neff / wall time — the data for the skymap-vs-SNR figure and the high-SNR
efficiency comparison vs the adaptive (AV) integrator.  Preliminary small-budget
run (H1/L1/V1): the flow **recovers the truth sky at every SNR through 640**, the
90% sky credible area shrinks with SNR (≈0.05 deg² at 40 → 3.6e-5 deg² at 80 →
sub-sample-resolution above), and in **that small-budget configuration** flow
re-use cut the per-event wall time ~2× (first event ≈114 s, warm-started events
≈60 s).  **That saving does not carry to production settings:** on an 8-event BNS
batch at full settings it measured 1589 s with re-use against 1567 s without
(1549/1629 vs 1644/1489 across two seeds) — a difference smaller than the
seed-to-seed spread, with its sign flipping.  Treat the ~2× as specific to the
small-budget benchmark, not as a general amortization argument, and see the
accuracy warning above before enabling re-use at all.  The simple moment-matched
Gaussian importance evidence is reliable only at moderate SNR (it is flagged
`nan` once `neff` collapses, since `logZ ≤ lnL_max` is violated by an
ill-conditioned proposal at sub-resolution peaks) — a robust narrow-peak evidence
estimator is future work; sky recovery is the high-SNR deliverable.

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
   --mode nuts --distance-marginalization --d-max 5000 \
   --save-samples --output-file out
```

(`--mode nuts` requires `--distance-marginalization`: `run_nuts` raises
`SystemExit` without it, because the bare 5-D angular+distance likelihood is
degenerate.  The command above previously omitted the flag and could not run.)

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
