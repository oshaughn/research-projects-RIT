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

### Phase marginalization and the packed mode set

`phase_marginalization=True` is implemented for the `(2,2)`/`(2,-2)` pair only:
the reduction conjugates the `m = -2` component of the harmonic, the antenna
response and the `rholm` timeseries, which is specific to a single
`m = +2`/`m = -2` pair.  Any other mode set raises `NotImplementedError` rather
than silently dropping a mode.

**Either packed ORDER is accepted.**  The column order of `lms`, `Q`, `U` and `V`
follows the iteration order of the precompute's mode dictionary, not anything the
caller chooses, so both `[(2,2), (2,-2)]` and `[(2,-2), (2,2)]` arrive in practice;
the accumulator canonicalizes internally.  Note that `U` and `V` carry the mode
index on BOTH axes -- any code reordering a packed bank by hand must permute both,
or it returns a wrong likelihood with no error.

### Time quadrature

All JAX likelihood wrappers accept the conventional ILE keyword
`time_quadrature={"simpson","bandlimited"}`.  Simpson remains the default.
The opt-in `bandlimited` path is currently supported by
`JAXExtrinsicLikelihood` (6-D, including analytic phase marginalization) and by
`JAXDistanceMarginalizedLikelihood` (5-D, the wrapper `--mode nuts`,
`multistart-nuts` and `flowmc` use under `--distance-marginalization`).  It forms
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
Nyquist frequency.  Likelihood mass at the ends of the short integration window
is covered by the guard-agreement certificate below, not by an endpoint gap.
The 15-nat endpoint gap of 2026-08-29 was switched off on 2026-09-08.  A row's
peak-to-endpoint contrast is bounded by its own amplitude, so the gap rejected
every blind or far draw whatever the quadrature did.  The rows it rejected
alone agree with an independent reference to 1e-4 nat (DESIGN record, "The
endpoint certificate").

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

The distance reduction runs on the refined nodes, not on an interpolated
`lnL(t)`.  `fused_log_likelihood_distmarg` gathers the guarded primitive, hands
the refiner the same blocked distance quadrature the Simpson path uses, and that
quadrature is evaluated inside the row-local `lax.map` at every fine node before
the trapezoid.  The block count for the fine grid is derived from the refined row
length rather than inherited from `grid_block`, so a 2048x row does not scale the
working set with it.  Three of the fixed-distance certificates -- factor
doubling, two-guard agreement, and remeasured resolution -- apply to the
distance-marginalized field, and the endpoint gap applies to neither.  With a
full-sky prior the integration half-window must contain the detector arrival
shifts of a wrong-sky draw (up to 2 R_earth / c, 42.6 ms).  A row whose arrival
peak sits at the window edge fails the doubling or guard certificate and stops
the driver.  The driver therefore refuses the modes that push full-sky draws
through that stop (`prior-mc`, `laplace-is`, `map`, `nuts`) at parse time when
the half-window is below that bound.
`return_lnLt` is refused under `bandlimited`: there is no reduced field on the
data grid to return.

The phi, psi, exact-angle, and Laplace-marginalized wrappers still refuse
`bandlimited`, and the reason is now specific rather than generic.  The phi_ref
grid sum streams one primitive per grid point through a `lax.scan` that carries
only `(S, npts)`; refining first means carrying `(nphi, n_fine)` per row, which
at the shipped `nphi` and the certified factor is two orders of magnitude more
scratch than the row-local budget allows.  The psi and exact-angle wrappers
reach the coefficient-table kernels in `anglemarg.py`, which return an
already-reduced `lnL(t)`: there is no primitive at that seam to refine without
an adapter.  Both continue to use the unchanged Simpson default.
`time_first_peaklocal.py` contains an unwired, fixed-shape prototype of that
primitive-first composition: it reconstructs one raw complex correlation per
downstream distance/angle quadrature state, builds a certified time-cell cover,
and only then performs the nonlinear reduction on local nodes.  It returns an
explicit validity ledger and changes no wrapper or CLI default.  Production
wiring still needs a tighter Hermite certificate, two-guard convergence, and an
adapter from the coefficient-table angle kernels.
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

For an Asimov/pseudo-pipe final fair-draw stage, the pipeline detects the JAX
executable and runs `util_ConvertJAXILEFairdraws.py`.  The converter strictly
pairs each tabular sidecar with its intrinsic likelihood record and writes the
usual joint posterior coordinates that are actually available.  It omits
`time`, which remains marginalized and is not exported by this driver, rather
than fabricating a coordinate.  Redshift and source-frame masses are likewise
not inferred; custom `--convert-args` are refused instead of silently ignored.
The compatibility columns `p` and `ps` are both the neutral value one because
the exported rows are already equal-weight fair draws.  Missing pairs,
malformed records, nonfinite rows, noncontiguous grid IDs, or unexpected
intrinsic/draw counts fail the terminal job and remove any stale terminal
output.  Conventional ILE retains its existing XML conversion path.  The
converter also writes a JSON provenance ledger beside the posterior, recording
input and output hashes, row counts, shuffle seed, neutral columns, and omitted
coordinates.

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
    below).  Honours `time_quadrature="bandlimited"` by refining the primitive
    first.
  - `make_distance_grid(...)`, `JAXLikelihoodData`, `build_likelihood_data`.
- `time_first_peaklocal.py` — experimental primitive-first time-cover planner
  and distance adapter; not selected by any production endpoint.
- `direct_marginalization_policy.py` — opt-in cross-axis policy
  (`--direct-marginalization-policy auto`): per evaluation, the four-axis
  peak-local controller of `all_axis_peaklocal.py` under its acceptance
  ledger, with a reserve on decline.  The reserve is a pair named by
  `--direct-marginalization-reserve-scheme`: exact angles on the whole-window
  refined rule (default), psi-Laplace angles on that rule, or `peaklocal`,
  psi-Laplace angles on a fixed-count time rule sized from the predicted
  peak width `1 / (2 pi rho sigma_f)` around maxima located on the primitive
  (`peaklocal_time_reserve.py`), with the coarse scan limited to the
  support of the located maxima, so the node count grows with neither rho
  nor the window.
  Value-only; see `DESIGN_direct_marginalization_policy.md`.
- `peaklocal_time_reserve.py` — the peak-local time rule: width prediction
  from the row's table and the stored Q's bandwidth, primitive-based maxima
  locator, one commensurate lattice, and the pair selector's node-count
  prediction.
- `../bivariate_trig_stationary.py` — host reference for complete finite-order
  `(phi_ref, 2 psi)` stationary enumeration by a Sylvester resultant and
  generalized eigenproblem.  It records BKK expected/found counts,
  conditioning, cross-projection agreement, and supplies best-effort targets
  only behind an outside-cover bound; no sampled phi grid is called
  enumeration.  A fixed-capacity JAX plan adapter remains future work.
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

The fixed distance grid under-resolves the per-sample integrand's peak (width
`~d0/SNR`) at high SNR.  The driver's `--distance-gh-nodes N` places `N`
Gauss-Hermite-style nodes centred on that peak, PER SAMPLE, resolving it to
machine precision at any SNR with a few dozen nodes; `N=0` (default) keeps the
legacy fixed grid.  Equivalent to the environment variable
`JAX_ILE_DISTMARG_GH`, still honoured for compatibility (`core.
set_distmarg_gh_nodes`); the two are refused, not silently reconciled, if set
to different nonzero values.  See `core.make_distance_gh` /
`core._distmarg_gh_logL` and `DESIGN_jax_distance_quadrature.md`.

## Driver

### Value-only adaptive volume and portfolio

The opt-in ``--sampler-method AV`` and ``--sampler-method portfolio`` paths use
the same JAX likelihood selected by ``--mode`` but do not differentiate it
during integration.  Likelihood rows are evaluated in one fixed JAX shape;
``--jax-av-eval-chunk`` therefore controls accelerator memory independently of
the larger ``--n-chunk`` used to cover and contract the adaptive volume.
AV/portfolio honor the production ``--d-prior pseudo_cosmo`` distance density,
including its normalization over ``[--d-min, --d-max]``; Euclidean/volumetric
remains the default.  Other cosmological distance-prior variants are refused
for this backend rather than silently changed.  A sampling-only
``--limit-distance`` does not renormalize either physical prior.

Portfolio defaults to AV plus a defensive GMM member.  An optional Fisher-sky
initializer pays an explicit, one-time AD cost for hill climbing and local
curvature; every integration evaluation remains value-only.  A finite seed
cloud does not itself guarantee prior support.  For blind/full-prior inference,
use the defensive portfolio rather than interpreting seeded standalone AV as a
global calculation.

For deliberately local tests, AV/portfolio honor
``--limit-right-ascension``, ``--limit-declination``, ``--limit-psi``, and
``--limit-inclination`` as comma-separated sampling limits.  These restrict the
domain sampled while the integrand retains the normalized full physical prior.
Consequently the evidence is the full-prior contribution from that domain; it
is not conditional on the box and must not receive an inverse-volume correction.
A widened-box repeat and a posterior edge-contact check are required before the
boxed contribution can be identified with the all-sky evidence.  RA windows
that cross 0/2pi are refused because one AV hyperrectangle cannot represent the
wrapped union.

``JAXFixedDistanceLikelihood`` provides a five-angular-coordinate view of the
six-dimensional likelihood for controlled validation problems.  It can also
shift the periodic phase coordinate so a narrow mode at physical phase zero is
not split across the sampler's box boundary; exported points must be mapped
back with ``to_physical_coordinates``.
``JAXRotatedPhaseLikelihood`` supplies the conventional
``--internal-rotate-phase`` sum/difference coordinates on a redundant
``[0,4 pi)`` cover, making the leading phase--polarization ridge axis-aligned
for AV as well as for gradient samplers.

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

### Persistent compilation cache

The shipped driver enables JAX's cross-process compilation cache by default.
RIFT disables JAX's auxiliary per-fusion autotune cache while doing so. In JAX
0.9.2 that auxiliary cache places its absolute directory in the executable
cache key, so leaving it enabled makes an otherwise compatible exported bundle
miss after import at a different filesystem path. The persistent compiled-
executable cache remains enabled and is the transferable cache described here.
It selects a stable directory under `$RIFT_JAX_CACHE_ROOT` (or
`$XDG_CACHE_HOME/rift/jax`, normally `~/.cache/rift/jax`) and adds a
compatibility namespace derived from Python, JAX/JAXLIB, the CUDA plugin,
backend/platform version, GPU kind, and compute capability. JAX's own keys then
separate static argument shapes and compiler options inside that namespace.

Use `--jax-cache-dir /shared/rift-jax-cache` to choose a shared root, or
`--no-jax-persistent-cache` for a diagnostic cold run. The standard
`JAX_COMPILATION_CACHE_DIR` variable remains an exact-directory expert
override. The selected directory contains its provenance manifest.
Runtime identity and durable imported-bundle profile/static-shape provenance
are stored separately. Each distinct contributing bundle gets an atomic record
keyed by its manifest digest, so neither a later ordinary startup nor a second
compatible bundle import can erase the earlier provenance.
On Condor, an unset root falls back to
`$_CONDOR_SCRATCH_DIR/.rift_cache/jax`; transfer that directory or set a shared
root to reuse it across jobs. An unwritable cache disables itself with a warning
rather than failing the ILE calculation.

Condor scratch is job-local, so default enablement there avoids duplicate
compilation only within that job; it does not provide automatic cross-job
persistence. To reuse a survey/full-run cache, transfer the bundle as an input
and append `--jax-cache-bundle rift-o4-laplace.zip --jax-cache-profile
o4-laplace` to the ordinary ILE arguments. The driver validates and imports it
before importing modules that construct ILE JITs. Sites with a genuinely shared
writable filesystem can instead set `RIFT_JAX_CACHE_ROOT` in the submit
environment.

Warm with the real production command, then package that active namespace and
record the important static shapes:

```sh
integrate_likelihood_extrinsic_jax --jax-cache-dir /scratch/rift-cache \
  <the production ILE arguments>
rift_jax_cache --cache-root /scratch/rift-cache export rift-o4-laplace.zip \
  --profile o4-laplace --shape detectors=3 --shape l_max=2 \
  --shape n_chunk=8000 --shape distance_grid=256 --shape n_phi=8
```

On a compatible target host/container, import and reuse it:

```sh
rift_jax_cache --cache-root /shared/rift-cache import rift-o4-laplace.zip \
  --expect-profile o4-laplace
integrate_likelihood_extrinsic_jax --jax-cache-dir /shared/rift-cache \
  <the same production ILE arguments>
```

Import rejects a different JAX/JAXLIB/CUDA backend, GPU kind/capability,
Python, requested profile, unexpected archive members, or checksum failure.
Different static shapes safely miss JAX's inner cache and compile normally;
the bundle's shape metadata makes those misses explainable.
Import also bounds member count, individual/total uncompressed size, and
compression ratio and streams entries through their checksum, so a corrupt or
hostile archive cannot expand without limit. Cache bundles contain compiler
artifacts and should still be accepted only from a trusted build workflow.

The compatibility namespace does not cover `XLA_FLAGS`, and does not need to:
JAX covers it. `jax/_src/cache_key.py::_hash_xla_flags` reads the `XLA_FLAGS`
and `LIBTPU_INIT_ARGS` environment variables and every `--xla*` token in
`sys.argv`, and hashes each into the key, skipping only the dump/debug flags in
`xla_flags_to_exclude_from_cache_key`. Measured on jax 0.9.2: adding
`--xla_cpu_enable_fast_math=true` to `XLA_FLAGS` produced a second, distinct
`jit_work-*` entry rather than reusing the first, and rerunning with unchanged
flags reused it. So a numerics-affecting flag varies the key rather than
silently reusing a kernel compiled under a different one, and flags need not be
held fixed per cache root.

Cache entries are written by JAX, not by RIFT, and `LRUCache.put` writes them
with a plain `write_bytes` rather than through a temporary file, so a reader can
observe a partial entry. Measured on jax 0.9.2 against a populated cache: an
entry truncated to 60%, an entry with one byte flipped mid-blob, and an entry
with 4 KiB of random bytes spliced in each produced the same lnL as the
uncorrupted run, via a `UserWarning: Error reading persistent compilation cache
entry` and a recompile. Corruption therefore fails CLOSED: a shared root
degrades to recompilation under contention and does not hand back a wrong
kernel.

It does not self-heal, and that is the operational cost. `LRUCache.put` returns
early when the path already exists, so a process killed mid-write -- a node
reboot, an OOM, a thread-budget abort -- leaves a truncated entry that no later
process rewrites. In the measurement above the file stayed at its truncated
2943 bytes across reruns. That key then recompiles forever while only warning.
If a warm cache stops saving time, delete the compatibility-keyed directory
(`rift_jax_cache fingerprint` names it) and re-warm; there is no partial repair.

The amplitude-adequacy diagnostic of the amp-sized schemes (exact, Laplace,
peak-local, phi-local) is deliberately data returned
by a pure JIT, not a `jax.debug.callback`: JAX does not persist graphs with host
callbacks. The driver synchronously accumulates the maximum over every pilot,
reweight, and final production/output-cloud batch and records that deterministic
scope in result provenance. Transient flow-training-only proposals are not
claimed; they do not enter the reported evidence or exported cloud. The
`direct-marginalization-policy` composite exposes no such metric and is
therefore unlabelled. A tripped
check still leaves likelihood values finite and labels the artifacts
`SUSPECT-ANGLE-GRID` rather than silently excising the affected region.

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
Waveform precomputation uses the same two-second post-event FD alignment as
production numpy ILE.  The compatibility options
``--internal-waveform-fd-L-frame`` and
``--internal-waveform-fd-no-condition`` are forwarded to the production
precompute call; they are not JAX-only transformations.
Input frames are first loaded at ``--srate`` and, when requested, upsampled to
``--srate-internal`` before the mode time series are constructed, matching the
two-cadence production ILE path.
The production defaults also retain all modes (no implicit precompute
threshold), retain memory modes unless ``--no-memory`` is given, and apply the
same ``--fmin-ifo`` and PSD-window normalization to each detector.
