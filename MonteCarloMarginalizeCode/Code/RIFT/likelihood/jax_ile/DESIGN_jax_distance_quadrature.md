# Distance quadrature for the dense angle-marginalization schemes

**This is an accuracy change, not a speedup.**  At the shipped tolerance, ON THE
REFERENCE CONFIGURATION (injected SNR 40), it uses ~46% MORE distance nodes than
the current default and is ~2000x more accurate.  (It was ~3% until the spacing
was sized from the fail-safe's TRIP threshold rather than from `amp_sizing`
itself -- see section 1's `rho_max` note.  The cost went up; the accuracy claim
did not change.)

**That ~46% does not travel.**  The count is derived from the run's own
amplitude, so it GROWS with it -- which is the scheme working as designed, not a
defect.  At `rho_sampled = 163.08` the same contract derives n = 1558, i.e. ~6x
the default rather than 1.46x.  Carrying "+46%" to a loud event is wrong by more
than a factor of four; see "quote a node-count error only with its amplitude
attached" in section 2a.
Matched to the current default's own accuracy it is 2.0-2.6x cheaper on the
distance axis, which is one axis of one kernel and does not rescue any campaign
(section 3).  Read the cost section before citing a factor from this document.

Scope: the `(x_grid, log_w_grid)` that
`fused_log_likelihood_distphipsimarg_{exact,laplace}` integrate distance over,
reached from `JAXDistPhiPsiMargLikelihood` / `--mode flowmc-phipsimarg`.

Everything below was measured on `origin/rift_O4d` @ `52433198` plus this
branch, on the ladder-2 reference configuration: SEOBNRv4, 35+30 Msun,
H1L1V1, `--srate 4096` (npts = 614), `--l-max 2`, injected SNR 40 at
d = 633.92 Mpc, prior range `[--d-min 1, --d-max 10000]` Mpc, `interp="sinc"`.
Environment: `~/.conda/envs/rift_jax` (python 3.13, jax 0.9.2), `JAX_ENABLE_X64=1`,
`OMP_NUM_THREADS=1`, `taskset -c 0-15`, `JAX_PLATFORMS=cpu`, `RIFT.__file__`
asserted into the tree under test by every script.

---

## 1. The contract, before and after

**Before (still the default).** `make_distance_grid` lays `--distance-grid-points`
nodes **uniformly in d** across the whole prior range and gives each node the
same interval `dd = (d_max - d_min)/(n-1)`.  What that guarantees is a *fixed*
discretization, independent of the data: the same grid for every event, and a
quadrature error that nobody has to think about because it never changes.  What
it does **not** guarantee is that the error is small.  The distance integrand's
width is set by the data (see below), and the shipped 256-node grid's error at
this operating point is 0.17-0.22 nats, not a design target anyone chose.

**After (opt-in, `--distance-grid-scheme loguniform`).** Nodes are laid
**uniformly in ln d** with a count derived from the run's own data:

    Delta(ln d)  <=  c(tol) / rho_max,     c(tol) = pi * sqrt(2 / ln(2/tol))
    n            =   ceil(rho_max * ln(d_max/d_min) / c) + 1

The guarantee changes from "a fixed grid" to "**a stated fractional error on
the distance integral, uniformly over the prior range and over every angle
sample -- PROVIDED the maximizing distance is interior to that range**".  The
error is no longer data-independent; it is *bounded* by a number the caller
states, conditional on a sky-sweep estimator (section 1's `rho_max`
discussion) and on that interiority precondition, which is checked at build
time and REFUSED when violated (section 1a).  Both conditions are stated here
because neither is an inequality anyone has proved.

### Why relative spacing is the right invariant

Per angle sample and time bin the distance integrand is

    exp(K x - 0.5 R x^2),    x = distMpcRef / d,
    K = Re<h|d>,  R = <h|h>  at the reference distance

a Gaussian in `x` peaked at `x* = K/R` with standard deviation `1/sqrt(R)`.
Its **relative** width is

    sigma / x* = sqrt(R)/K = 1 / rho,      rho = K / sqrt(R) = that sample's matched SNR

which does not depend on where the peak sits.  So one relative spacing resolves
every peak everywhere, with no peak location entering anywhere.  Uniform-in-d
spacing has the opposite property: it over-resolves large d (where the
likelihood is flat) and starves small d (where the peaks are narrowest in
absolute terms).

### Why `c(tol)` is derived rather than tuned

The trapezoid rule on a Gaussian converges super-algebraically.  By Poisson
summation, for `f = exp(-(u-mu)^2/2s^2)` sampled at spacing `h`,

    (h * sum_k f(u_k)) / integral(f)  =  1 + 2 exp(-2 pi^2 s^2 / h^2) cos(2 pi mu / h) + ...

so the worst-case fractional error over the peak phase `mu` is
`2 exp(-2 pi^2 s^2 / h^2)`.  Setting that to `tol` and writing `h = c * s` gives
`c = pi * sqrt(2/ln(2/tol))`.  `test_tolerance_constant_matches_the_gaussian_trapezoid_error_law`
pins this **two-sided**: a one-sided "error <= tol" check is satisfied by any
smaller `c`, including `c -> 0`, which is perfectly accurate and arbitrarily
expensive.

### Where `rho_max` comes from -- and why it needs no new estimator

`anglemarg.estimate_angle_amplitude` returns, by its own construction,
`ANGLE_AMP_MARGIN` times the maximum over a SAMPLED set of angles, and over the
distance support, of `x A - 0.5 x^2 B`, whose closed-form maximum in `x` is
`A^2/(2B) = rho^2/2`.  So

    rho_max = sqrt(2 * A) = sqrt(ANGLE_AMP_MARGIN) * rho_sampled_max
            = sqrt(2) * rho_sampled_max      (measured: exactly 1.4142 at rho 25/40/80)

**This is not an identity and `rho_max` is not a proven bound.**  It is
`sqrt(2)` times the largest matched SNR a finite sky sweep found, and that
estimator's own docstring says so: *"THIS IS AN ESTIMATOR, NOT A PROVEN
BOUND"*.  Two consequences worth stating plainly:

* the shipped node count is **41% above what the stated tolerance requires** --
  the reference configuration's 373 nodes meet the `tol = 1e-2` contract at
  ~187.  That margin is deliberate, and it is earning its keep: on this
  configuration the sweep's own empirical maximum was 758.17 against the
  injection's 800, i.e. it DID under-read the true maximum, and the margin
  absorbed it;
* the contract is therefore conditional on the sweep, not on an inequality.  It
  is stated that way here and should be read that way.

Three consequences, all of them the point of this design:

* no peak is located, so there is no peak estimate to be wrong;
* the number that sizes the distance grid is the **same** number that sizes the
  dense angle lattice, so the two cannot disagree;
* the fused kernels' existing runtime fail-safe (`_runtime_amp_failsafe`)
  recomputes `A` from the coefficient tables on **every** likelihood call and
  warns if the shipped sizing was exceeded.  That fail-safe therefore now
  covers the distance grid too, at no extra cost.  The residual failure mode is
  named in §5.

`A` here is `amp_sizing` -- the amplitude **floored at
`ANGLE_MARG_CROSSOVER_AMPLITUDE = 450`** -- and not the unfloored `amp_data`.
That is deliberate and it is what makes the fail-safe coverage in section 5 an
identity rather than an approximation: `_runtime_amp_failsafe` compares the
per-call amplitude against `amp_sizing`, so sizing the distance spacing from
anything else leaves a gap.  Concretely, on a quiet target (`amp_data < 450`) a
runtime amplitude anywhere between `amp_data` and 450 would under-resolve the
distance peak and trip nothing.  The floor costs a minimum of ~144 nodes on an
event whose run is cheap anyway.  Scheme *selection* still uses the unfloored
`amp_data`, unchanged, so quiet targets still take the `exact` branch.

### 1a. Where the contract does NOT hold, and what happens instead

The derivation above assumes the integrand is a Gaussian **peak inside the
support**.  When the maximizing distance `x* = A/B` is EXTERIOR to
`[x_min, x_max]` -- the distance posterior rails against a prior edge -- the
integrand is monotone on the support instead: a boundary layer at `d_max` (or
`d_min`).  The log-uniform grid is the wrong instrument for that, twice over:

* its ABSOLUTE spacing is coarsest exactly at `d_max`, where the layer sits;
* refining it adds nodes proportionally *everywhere*, so the layer never
  resolves.  Measured on the reference configuration with the peak pushed to
  `d* = 20000` Mpc against a `[1, 10000]` Mpc prior and a 2^21-node reference,
  tightening `tol` from 0.5 to 1e-9 moves the error only **5.23 -> 3.92 nats**,
  while uniform 256 -> 4096 moves **2.52 -> 0.36**.

And the node count moves the WRONG WAY, because `estimate_angle_amplitude`
maximizes at `clip(A/B, x_min, x_max)`: an exterior `A/B` makes the clipped
value UNDER-read `rho^2/2`, and in the extreme it returns exactly 0, at which
point the crossover floor pins `rho_max = 30` and the grid collapses to 145
nodes over `[1, 10^4]` Mpc.  Measured:

| rho | d* (Mpc) | support | n at tol=1e-2 | log-uniform err | uniform-256 err |
|---|---|---|---|---|---|
| 40 | 12000 | [1, 10000] | 271 | **+1.89 nats** | +0.67 |
| 40 | 20000 | [1, 10000] | **145** (amp -> 0) | **+4.60** | +2.53 |
| 40 | 3000 | [100, 2000] | 89 | **+3.01** | +1.51 |
| 40 | 634 (interior control) | [1, 10000] | 271 | +5.6e-05 | +0.049 |

So at the UPPER edge the scheme is **1.3-3x WORSE in nats than the default it
replaces**.  It is REFUSED at build time, not silently mis-sized and not
silently fallen back to uniform.  The detector is a single scalar from the same
sky sweep: the amplitude recomputed WITHOUT the clip (`A^2/(2B)`, the true
stationary value) against the clipped one.  `clip_excess > 1 + 1e-3` means the
maximizer is exterior.  Verified to fire on truncated supports and to stay at
exactly 1.0 on interior ones.

**The refusal is symmetric in the two prior edges; the harm is NOT, and the
table above measures only one of them.**  Every row of it puts the maximizer
ABOVE `d_max`, which is where the mechanism argument applies -- the log grid's
absolute spacing is coarsest at `d_max`, so a layer there is the worst case for
it.  At the LOWER edge the same grid is *finest* exactly where the layer sits,
and the sign reverses.  Measured (external re-review, 2026-09-02) on a
`rho_max = 30` target with the maximizer at 86 Mpc, sweeping `d_min` past it
with `d_max = 10^4` Mpc.  **The reference is two references**: a uniform
65536-node grid and a log-uniform grid at `tol = 1e-10`, and their mutual
disagreement is reported as a column rather than assumed away -- an earlier
draft of this table used a single uniform-8192 reference and argued its
convergence from the fact that the log grid agreed with it, which is a
common-mode argument and was ~25% low.

| `d_min` | `clip_excess` | verdict | n (log) | refs disagree | log-uniform err | uniform-256 err |
|---|---|---|---|---|---|---|
| 86 | 1.0 | accepted | 75 | 3.68e-5 | 1.85e-4 | 1.41e-3 |
| 92 | 1.0044 | REFUSED | 74 | 3.67e-5 | 1.85e-4 | 1.41e-3 |
| 120 | 1.0879 | REFUSED | 70 | 3.69e-5 | 1.84e-4 | 1.41e-3 |
| 400 | 2.5579 | REFUSED | 52 | 3.79e-5 | 1.81e-4 | 1.36e-3 |

**Read this to two significant figures and no further.**  The two references
disagree by 3.7e-5, which is ~20% of the log-uniform error being measured, so
that column is good to about that and no better.  What survives the uncertainty
is the ratio and its stability: log-uniform is **~7.6x more accurate than the
uniform-256 default** here, and both errors are flat to a few percent while the
support shrinks 4.6x and the node count falls 75 -> 52.  An independent
reconstruction of this measurement by a second reviewer reproduced the
DIRECTION (log-uniform better at the lower edge, by 2.3x-540x on their
`(A, B)`) but not the flatness; their fixture is not this one, and the
disagreement is unresolved.  Treat the direction as established and the
magnitude as fixture-specific.

We refuse both edges anyway, and that is a deliberate choice rather than an
oversight: the CONTRACT is what fails once the maximizer leaves the support --
`c(tol)/rho_max` is derived from a Gaussian peak's relative width, and there is
no peak on the support to have a width -- so the stated fractional error is not
being delivered even where the realised error happens to be small.  Refusing on
the condition we can actually detect (`clip_excess`) rather than on a realised
error we cannot compute at build time keeps the option's promise honest in both
directions.  The cost is stated here rather than hidden: at the lower edge the
refusal sends the caller back to a grid that is measurably worse.  Recourse is
the same one the message names -- narrow `--d-min` so the posterior is interior.

**What `tol` actually delivers near an edge, measured.**  The endpoint guard
budgets the whole of `tol` to the endpoint term while the alias law has already
spent it: `c(tol)` is derived from `2 exp(-2 pi^2 / c^2) = tol`, so at the
sizing peak the alias error IS `tol` by construction.  The two add.  Measured on
the scheme's own quadrature in pure numpy (uniform in `ln d`, half-width end
intervals, against a 2e6-node reference):

| clearance (peak widths) | measured error | guard |
|---|---|---|
| 1.0 | 0.113 | refuses |
| 2.93 (the reported minimum) | 0.0132 | accepts |
| 30 (deep interior) | 0.0093 | accepts |

So deep in the interior the contract is met (0.0093 against 0.01), and **at the
acceptance boundary for a peak at the sizing SNR the delivered error is ~1.3x
`tol`, not `tol`**.  That is stated rather than fixed: making the comparison
`alias + endpoint <= tol` would refuse EVERY loud peak, because the alias term
alone already equals `tol` at `rho = rho_max`.  The honest bound is that the
option delivers `tol` deep in the interior and up to ~2x `tol` (the guard's own
`ENDPOINT_ERROR_MARGIN`) in the last few widths before a refusal.  Quieter peaks
sit on a proportionally finer grid and are covered with room to spare -- measured
0.0053 against a 0.0081 estimate at `rho_pk = 10.3` on a grid sized for 42.4.

**The non-positive-clearance window, and why it is a separate refusal.**
`_endpoint_bell` clamps `k <= 0` to zero, on the correct observation that an
endpoint sitting ON the peak is a stationary point -- measured error 1.2e-9 at
exactly `k = 0`.  But the clamp cannot distinguish that from `k < 0`, where the
peak has left the support and the error climbs steeply, so the term scored its
worst case as its safest.  Between that and the `clip_excess` trip lay a window
neither guard saw: measured, clearance -0.656 built with a true error of
**0.0668, 6.7x `tol`**, while `clip_excess` still read 1.0.  A non-positive
clearance is now refused outright.  Measured transition, monotone with no
building window:

| clearance | true error | verdict |
|---|---|---|
| 6.97 | -- | builds |
| 2.07 | 0.0074 | refused (endpoint) |
| -0.66 | 0.0668 | refused (clearance) |
| -2.40 | 0.485 | refused (exterior) |

**Why refuse rather than fall back to uniform.**  Three reasons.  A fallback
would make `--distance-grid-scheme loguniform` silently produce the *other*
scheme's grid -- the silently-inert-flag class this module keeps being bitten
by, and the same defect as the `JAX_ILE_DISTMARG_GH` combination refused
alongside it.  Both grids are bad here anyway: uniform 256 is itself 2.5 nats
out, so a fallback substitutes one wrong answer for another rather than fixing
anything.  And the regime is a *physics* signal -- the posterior is railing
against `--d-max` -- which the caller should see rather than have papered over.
The refusal names the recourse: widen `--d-max` (or narrow `--d-min`) so the
posterior is interior, or stay on the uniform default and raise
`--distance-grid-points`.

Residual limitation, stated: the detector reads the same finite sky sweep as
the amplitude.  If that sweep misses an exterior-peak angle configuration
entirely, the build proceeds.  Nothing here bounds that.

### The decoupling property (the most important safety property here)

`estimate_angle_amplitude` reads only `x_grid.min()` and `x_grid.max()`: the
per-angle distance maximum is closed form at `clip(A/B, x_min, x_max)`, and
`A/B` is interior to any window that contains it.

**CORRECTION, and it weakens this section: no in-tree scheme can currently
trigger the coupling.**  `make_distance_grid_adaptive` always concatenates a
full-range `linspace` backbone (`coarse = np.linspace(d_min, d_max, n_coarse)`)
before dedup, so it spans the whole support and returns an identical amplitude;
and both grids this PR ships span the full support by construction.  The
`[0.8 d, 1.25 d]` window in the table below is a hand-built diagnostic, not a
grid any code path produces.  So the decoupling change is **prospective
insurance**, not an active protection, and the mutation that reverses it
survives every value-level assertion for exactly that reason (see the PR's
mutation matrix, M12).  It is still worth having -- it makes the invariant
structural rather than incidental -- but it should not be sold as fixing a live
bug.  Measured on this configuration (`amp_vs_distgrid.py`, same directory):

| distance grid | n | amp | dense (n_phi, n_u) |
|---|---|---|---|
| shipped `[1, 10000]` | 256 | 1516.33 | 624, 320 |
| `[1, 10000]` | 64 | 1516.33 | 624, 320 |
| `[1, 10000]` | 24 | 1516.33 | 624, 320 |
| `[0.5d, 2d]` | 24 | 1516.33 | 624, 320 |
| `[0.8d, 1.25d]` | 24 | **1325.78** | **592, 304** |

Identical to every printed digit until the window stops containing `A/B`, and
then the **angle lattice silently shrinks**.  This branch removes that coupling
by construction rather than bounding it: the amplitude is computed on a
full-support uniform grid built for that purpose, never on the grid the
likelihood integrates over.  For `--distance-grid-scheme uniform` the two are
the same object, so the default path is unchanged node for node.

---

## 2. Accuracy, in nats

### 2a. The distance quadrature in isolation

`dist_quad_error2.py` (measurement scripts: RIFT_roboto_paper `analyses/jax_anglemarg_exec_cost/`).  The dense kernels integrate
distance as a plain log-sum-exp over `(x_grid, log_w_grid)` of
`exp(x A - 0.5 x^2 B)`; with `A, B` replaced by the `(K, R)` that
`core._accumulate_unit` returns, the identical quadrature can be evaluated on
the real precompute at negligible cost, for 256 angle samples x 614 time bins,
against a **65536-node uniform reference** (self-converged: the 32768 -> 65536
step moves the result by 2.3e-5 nats).  The reported quantity is the per-sample
value after the time reduction, `L_s = logsumexp_t lnZ_d(s,t)` -- the number the
sampler consumes -- and `dlnZ`, the cloud evidence proxy.

Prior-draw cloud (S = 256):

| grid | n | max abs dL_s | dlnZ (nats) |
|---|---|---|---|
| uniform 24 ("the 10.7x") | 24 | **27.4** | **+1.57** |
| uniform 64 | 64 | 4.50 | +0.238 |
| uniform 128 | 128 | 0.640 | -0.104 |
| **uniform 256 (shipped default)** | 256 | **0.170** | **-0.170** |
| uniform 512 | 512 | 0.0029 | -0.0029 |
| log-uniform 96 | 96 | 0.208 | +0.184 |
| log-uniform 128 | 128 | 0.0415 | -0.026 |
| log-uniform 160 | 160 | 0.0061 | -0.0061 |
| log-uniform 192 | 192 | 0.0030 | +8.9e-6 |
| log-uniform 256 | 256 | 0.0017 | +2.4e-5 |
| `make_distance_grid_adaptive` (in tree) | 144 | **9.44** | -0.312 |

**Quote a node-count error only with its AMPLITUDE attached.**  Every row in
these two tables is at the header's operating point (injected SNR 40).  The
distance peak narrows as `1/rho` against a grid fixed by the PRIOR range, so a
256-node uniform grid is a different instrument at a different amplitude: the
same family measures 0.170-0.216 nats here and 43.16 nats at `rho = 163.08`
(paper-1 ladder, `snr160_laplace_gh0_dg256`).  Same rule, opposite sides of the
threshold -- not a discrepancy.

It is NOT a conversion, and two sessions have now been tempted to use it as one.
Matching the two series on points-per-peak-width does not collapse them: on
`N/rho` the louder series is 9.2-17.0x worse and roughly flat; on `N/rho^2` (the
better-motivated scaling, since a grid uniform in `d` over a fixed range
resolves a peak of width `d*/rho`, and `d* ~ 1/rho` when amplitude is set by
injected distance) it is 0.67x, 0.52x, 0.06x -- overshooting and not flat.  The
amplitude threshold is the dominant effect and explains the sign and most of the
size; the residual is real, not constant, and carries the rest of the
configuration (seglen, deltaF, probe points against a cloud max).  So compare
grids WITHIN one operating point and re-measure across them.

For scale at the loud end: `rho_sampled = 163.08` derives
`rho_max = sqrt(ANGLE_AMP_MARGIN * AMP_FAILSAFE_TRIP_FACTOR) * rho = 326.16` and
**n = 1558** at `tol = 1e-2` over `[1, 10^4]` Mpc -- against 4096 linear nodes
for that rung's converged reference, so under a factor of three, not the ninth a
transfer of the SNR-40 count (373) would suggest.

Cloud concentrated near the injection (S = 256):

| grid | n | max abs dL_s | dlnZ (nats) |
|---|---|---|---|
| uniform 24 | 24 | **73.0** | **-34.0** |
| uniform 64 | 64 | 9.04 | -2.70 |
| uniform 128 | 128 | 2.30 | -0.594 |
| **uniform 256 (shipped default)** | 256 | **0.216** | -0.022 |
| uniform 512 | 512 | 0.0032 | -0.0029 |
| log-uniform 128 | 128 | 0.184 | +0.158 |
| log-uniform 160 | 160 | 0.040 | -0.033 |
| log-uniform 192 | 192 | 0.0066 | -6.3e-4 |
| log-uniform 256 | 256 | 1.0e-4 | -1.0e-4 |
| `make_distance_grid_adaptive` (in tree) | 144 | **22.6** | -1.73 |

Independently confirmed by the coordinator with a different reference (uniform
`n_grid = 8192`), a different cloud (the exported 4800-row ladder-2 cloud) and
the **`grid`** angle scheme, so the distance path is isolated from the dense
machinery entirely.  That sweep gives, against its own reference: `n_grid = 24`
mean +34.64 / max 56.05 / lnZ +50.86 nats; `n_grid = 128` mean +0.32 / max 3.05;
`n_grid = 256` (shipped) mean +0.010 / max 0.326 / lnZ +0.013; `n_grid = 512`
mean +0.0028.  Two references, two clouds and two angle schemes sharing no
estimator agree on the shipped default's max error (0.326 against 0.216 here)
and on the catastrophe at 24 nodes.

Read off:

* **The shipped default's error is 0.17-0.22 nats**, not a chosen tolerance.
* **Equal accuracy to the shipped default is reached at n ~ 100-128 log-uniform
  nodes: a 2.0-2.6x reduction on the distance axis.**
* At the shipped tolerance default `tol = 1e-2`, at the reference
  configuration's amplitude, the derived count is **n = 373** -- ~46% *more*
  nodes than the default -- for **1.0e-4 nats instead of 0.216**, i.e. ~2000x
  more accurate at ~1.46x the distance-axis cost.  Both the count and the ratio
  are amplitude-dependent: n = 1558 (~6x the default) at `rho_sampled = 163.08`.
* **There is no 10.7x at fixed accuracy.**  `n_grid = 24` costs 27-73 nats of
  lnL and 1.6-34 nats of evidence.  See §4.

### 2a'. What these numbers do and do not establish

* **One operating point, one event, zero noise.**  `load_injection` builds
  `data_dict[det] = non_herm_hoff(P)` -- pure signal, no noise realization --
  with `SimNoisePSDaLIGOZeroDetHighPower` for all three detectors including V1.
  So these are quadrature-error measurements on a clean signal, not a
  population statement.  Nothing here is averaged over noise realizations,
  masses, or sky positions.
* **The "equal accuracy at n ~ 100-128" figure is empirical, not the
  contract.**  It is where the measured error happens to match the shipped
  grid's on THIS configuration.  The shipped default `tol = 1e-2` deliberately
  sizes above it (n = 373), because the option's selling point is a bound that
  holds without this measurement, not the smallest number that passed it.
* **The error metric is the per-sample lnL after the time reduction, plus a
  cloud evidence proxy.**  It is not a posterior-level statement: a bias that is
  constant across the cloud cancels in the posterior and not in the evidence,
  and this metric reports both.
* The reference is a uniform grid, i.e. the SAME quadrature family refined --
  a common-mode assumption.  It is defensible here because the integrand is a
  smooth 1-D Gaussian-times-polynomial with no singular structure, and because
  the two grid families (uniform and log-uniform) agree with each other to
  1e-4 nats at their converged ends, which a shared systematic would not
  produce.

### 2b. The same thing through the real `laplace` kernel

`e2e_laplace.py` drives
`fused_log_likelihood_distphipsimarg_laplace` itself with `amp_sizing` held
fixed across arms, so the dense angle lattice is bit-identical and the distance
grid is the only variable.  Numbers in §2c of the PR body.

---

## 3. Cost

Cost on the distance axis is linear in the node count: the laplace kernel scans
`ceil(n/dist_block)` blocks of a fixed kernel (#209), and the exact kernel calls
`_logsumexp_grid_blocked` over the same nodes.  Timing method: interleaved arms
inside each replicate, order flipped between replicates, explicit
`block_until_ready()` on every arm, minimum and full spread reported -- a
sequential A/B on a quiet host has overstated a speedup by ~50% on this code
before.  Numbers in the PR body.

**There is no speedup to quote at the recommended operating point**, and the
matched-accuracy one is small.  At `tol = 1e-2` and the reference amplitude the
derived count is 373 against the shipped 256: ~46% more distance work -- and ~6x
more at `rho_sampled = 163.08`, since the count follows the amplitude.  Matched to the shipped grid's own
accuracy the count is ~100-128, i.e. 2.0-2.6x less work **on the distance axis
only** — the dense `(phi, u)` lattice is unchanged by construction.  Combined
with the other measured lever (per-distance-block phi sizing: 1.45x standalone,
1.11x after this one) the total reaches ~2.9-4x against a campaign requirement
of order 140x, so this does not rescue that campaign and must not be cited as
if it might.

### Compile and execute, separated, at production `npts`

`compile_vs_execute.py` times the two phases apart using jax's explicit
lowering API (`jax.jit(f).lower(...)` then `.compile()`), so the compile is not
hidden inside a first call.  `npts = 614`, S = 64 sky samples, `amp_sizing`
pinned so the dense angle lattice is identical across arms.  RTX PRO 4000
Blackwell.

| arm | n | trace | **compile** | **first execute** |
|---|---|---|---|---|
| uniform 256 (shipped) | 256 | 1.76 s | **29.16 s** | **421.4 s** |
| log-uniform `tol = 0.5` | 136 | 2.31 s | **28.00 s** | **243.4 s** |

Interleaved execute timings, order flipped between replicates, explicit
`block_until_ready()` on every arm, 5 replicates:

| arm | n | compile | exec min | exec med | exec max |
|---|---|---|---|---|---|
| uniform 256 (shipped) | 256 | 29.16 s | 419.993 s | 420.100 s | 420.165 s |
| log-uniform `tol = 0.5` | 136 | 28.00 s | 243.132 s | 243.138 s | 243.245 s |

**Matched-accuracy execute speedup: 1.727x** for a 1.88x node reduction.  The
run-to-run spread is 0.17 s on a 420 s arm (0.04%), and the sequential
first-execute pair measured before the replicate loop gave 1.732x -- so the
sequential-vs-interleaved concern that has bitten this code before did NOT bite
here, and that is a measured statement rather than an assumption.  Fitting the
two arms plus the second GPU's n = 1024 point gives
`t_exec = 41.5 s + 1.484 s/node`: linear in the node count with a small fixed
term, which is why the wall-clock ratio (1.73x) sits just under the node ratio
(1.88x).  The fit predicts 1561 s at n = 1024 against 1685 s measured on the
other card (6%).

The two arms differ by **0.597 nats** at their largest over the 64-sample prior
cloud.  That is an arm-to-arm DIFFERENCE, not either arm's error: section 2a
puts the shipped uniform 256 at 0.17-0.33 nats from a converged reference and a
136-node log-uniform grid at a comparable distance from it, so a difference of
this size is what two independently-wrong approximations should show.  It is
also why `tol = 0.5` is the loose end of the option and not what this document
recommends -- at the shipped `tol = 1e-2` the count is 373 and the error is
~1e-4 nats, and there is no speedup at all.

The practical number: **~6.6 s per sky sample at `npts = 614`, n = 256**.  That
predicts ~94 minutes for an 853-sample chunk, which is consistent with a
separate `laplace` job observed spending over 78 minutes of GPU at 99% without
completing one such chunk — and it is ~2.9x slower than the rate #210 published.
**#209 and #210 were both benchmarked at `npts = 64`**, about a tenth of the
production value on the axis that drives this cost.  That discrepancy is a
finding about the published cost tables, not about this change, and it deserves
its own issue.

Two scaling consequences that follow from the node count
`n = ceil(rho_max * ln(d_max/d_min) / c) + 1`, and that a reader should have in
front of them before choosing this scheme:

* **Cost is linear in `rho_max`, i.e. in SNR.**  At the reference SNR 40,
  `rho_max = 77.88` (= sqrt(TRIP) x 55.07) and `n = 373` at `tol = 1e-2`.  At
  the top of the cost bake-off ladder (SNR 640) `rho_max ~ 905` and `n ~ 4300`,
  ~17x the shipped grid.  That is not a regression introduced here: a uniform
  256-node grid at that SNR is wrong by far more than the numbers in section 2
  (its spacing is ~80 sigma).  It does mean **the log-uniform grid is a good
  deal at moderate SNR and an expensive one at extreme SNR**, where the
  per-sample quadrature of section 4(d) is the right answer instead.  The
  derived node count is printed by the driver, so the cost is visible before the
  run rather than after it.
* **Cost is linear in `ln(d_max/d_min)`.**  The driver defaults are
  `[1, 10000]` Mpc, `ln = 9.21`, and roughly half of that -- everything below
  100 Mpc -- carries essentially no posterior for a 35+30 source.  Narrowing an
  unphysically wide distance prior now buys proportional speed; under the
  uniform grid it buys nothing, because the spacing is set by `d_max` alone.
  `--d-min 50` on this configuration takes `n` from 373 to 215.

---

## 4. Rejected alternatives, with the measurement that rejected them

**First, a distinction this document must not blur.**  "The adaptive distance
machinery" names TWO different mechanisms and only one of them is being
rejected here.

* `estimate_distance_peak` + `make_distance_grid_adaptive` build ONE static
  window from an EXTERNAL peak estimate.  That estimate is a 300-step gradient
  ascent on `0.5*K^2/R`, measured 15.5-19.8 sigma from truth and varying
  224.8-1231.5 Mpc with the random seed alone.  This is what is rejected
  below, and the rejection is about the ESTIMATOR, not about adaptivity.
* `core._distmarg_gh_logL` is a per-sample adaptive quadrature: centre
  `stop_gradient(clip(K/R, x_min, x_max))`, width `1/sqrt(R)`, both derived
  from the data AT THAT (sample, time-bin), with no external estimator
  anywhere.  Nothing here rejects it, and the published `JAX_ILE_DISTMARG_GH`
  rows are not tarred by the paragraph above.  It is REFUSED in combination
  with this option only because it consumes just the SUPPORT of `x_grid`, so
  the option would be bit-identically inert beside it -- a flag-inertness
  refusal, not a quality judgement.  Its centre-clipping is designed for
  exactly the boundary-layer regime section 1a refuses, so the two may well be
  complementary rather than competing; that is unmeasured here and is an open
  lead, not a claim.

**(a) `make_distance_grid_adaptive` + `estimate_distance_peak` (the in-tree
machinery behind `JAX_ILE_DISTGRID_ADAPTIVE`).**  Not shipped.  Three defects,
all measured:

1. `estimate_distance_peak` returned `d_peak = 1037.58 Mpc, sigma_d = 25.96 Mpc`
   for a `d_inj = 633.92 Mpc` injection.  Its `+-12 sigma` fine window is
   `[726.1, 1349.1]` Mpc, which **does not contain the injected distance**.
2. Its 300-step fixed-schedule gradient ascent is not converged: the `rho` it
   implies (`d_peak/sigma_d = 39.97`) is far below the amplitude bound's 55.07.
3. The returned peak does not respond to `guess_snr` at all: it is
   byte-identical (`d_peak = 1037.5768181433305`,
   `sigma_d = 25.95911760361334`) for `guess_snr` of `None`, 1.0, 17.38, 25.36,
   40.0 and 400.0 — a 400x span, one distinct pair.  Recorded as an
   OBSERVATION.  The only thing added here is a citation, not a mechanism: the
   function's own docstring says `guess_snr` "is accepted only as a fallback if
   the sweep finds no K>0 sample", so on any path where the sky sweep succeeds
   the argument is inert by design.  `d_peak / d_inj = 1.636763` is **not**
   explained in this document and should not be guessed at.
4. Its trapezoid gives the first and last nodes a **full** rather than half
   interval.  On a coarse backbone that misplaces percent-level volumetric
   prior mass onto `d_max`; measured as a **0.018-0.026 nat error floor that no
   refinement removes** (see the "FULL-endpoint" rows: 0.052 nats at n = 256,
   still 0.026 at n = 512).

Net: **9.4 nats (prior cloud) / 22.6 nats (near-injection cloud)** of lnL error
at 144 nodes.  The helper's own docstring already carries a `LIMITATION`
paragraph saying a single static window is insufficient here; this measures it.
The env-var branch is left reachable so nothing that sets it changes behaviour,
but it now prints a deprecation warning and is mutually exclusive with the new
option.  **Recommend removing it outright in a follow-up.**

One behaviour change does reach that branch, and it is a fix rather than a
regression: the dense angle lattice used to be sized from the *adaptive* grid
and is now sized from the full prior support.  Because
`estimate_angle_amplitude` clips to the grid's own `[x_min, x_max]`, the old
ordering could only ever make the lattice SMALLER than the full-support answer
(measured: 12.6% smaller amplitude, `(624, 320) -> (592, 304)`, on a
`[0.8d, 1.25d]` window).  So the new ordering can only make it the same or
larger.  `--distance-grid-scheme uniform` without the environment variable --
i.e. every existing command line -- is unaffected: there the two grids are the
same object.

**(b) A data-derived *window* (fine zone + coarse backbone) instead of a
full-range log-uniform grid.**  Rejected, and this is the more interesting
result.  `support_window.py` reads the amplitude
estimator's own sky sweep (reproduced bit-exactly: the amplitude it recomputes
matches `estimate_angle_amplitude` to 0.00e+00 relative) and asks how wide the
set of `x* = clip(A/B)` is over angle points within `T` nats of the maximum:

| T (nats) | window (Mpc, padded) | ln-width | fine nodes at spacing 1/rho_max |
|---|---|---|---|
| 30 | [932, 1260] | 0.30 | 17 |
| 50 | [895, 1331] | 0.40 | 22 |
| 100 | [830, 1489] | 0.58 | 33 |
| 200 | [669, 1798] | 0.99 | 55 |
| 400 | [193, 2609] | 2.60 | 144 |

That looks like a large win -- 22 fine nodes instead of 373 -- and it is a trap.
The window is centred on **1073 Mpc**, and `d_inj = 633.92 Mpc` is **outside it
until T = 400**.  The reason is structural: the sweep is 64 random
sky/inclination draws plus deterministic extremes, and its empirical maximum
(758.17) is *below* the injection's own value (800).  `ANGLE_AMP_MARGIN = 2.0`
bounds the sweep's error in the amplitude **value**, which is all the angle
lattice needs -- but nothing bounds its error in the **location** `x*`, which is
what a window needs.  A window built from that sweep can therefore exclude the
true peak while the amplitude bound it shares is perfectly sound.  So: no
window.  The full-range log-uniform grid is location-free by construction, and
that is precisely why it is the one shipped.

**(c) Per-distance-block dense phi sizing.**  Measured separately and rejected:
1.45x standalone (not the 2-3x claimed), and 1.11x once an adaptive distance
grid has run, because the adaptive grid deletes exactly the low-amplitude
far-distance nodes that per-block sizing feeds on.  Sizing goes as `sqrt(A)`, so
halving `n_phi` needs a 4x smaller amplitude.  Documented dead end; do not
re-propose.

**(d) The per-sample adaptive quadrature (`core._distmarg_gh_logL`, env
`JAX_ILE_DISTMARG_GH`).**  This is the numerically superior answer -- nodes at
`x* +- 7 sigma` per sample, ~32-64 of them, gradient-stable via
`stop_gradient` -- and it is already implemented and already wired into the
`exact` scheme.  It is **not** touched here because `laplace`
(1.9x faster than `exact` since #210, and the scheme the SNR-40 selector picks)
explicitly refuses it: its node placement is defined per fixed-`psi` exponent,
and the Laplace path has already integrated `psi` out analytically at each node.
Extending it would need a psi-marginal node-placement rule and its own
validation.  The log-uniform grid works on **both** dense schemes today.
Promoting `JAX_ILE_DISTMARG_GH` from an environment variable to a first-class
option for `exact` is a good, separate PR.

---

## 5. What can still go wrong, and what detects it

Three distinct failure modes.  They have different detectors and one of them is
NOT covered by the runtime fail-safe, which an earlier draft of this document
wrongly claimed it was.

**(a) `rho_max` underestimates the interior peak's sharpness.**  Then the
spacing is too coarse and the distance marginal is biased.  `rho_max` derives
from `amp_sizing`, the same number the dense lattice is sized from, and
`anglemarg._runtime_amp_failsafe` recomputes the amplitude from the coefficient
tables inside every jitted likelihood call and warns when it exceeds
`amp_sizing`.  That coverage is inherited, and it only holds because the
spacing is sized from `amp_sizing` (floored) rather than the unfloored
`amp_data` -- see section 1.

**Limits, and they are more severe than an earlier draft of this section
said.**  (i) The fail-safe is a `jax.debug.callback`, which XLA may drop (the
driver already labels artifacts `BEST-EFFORT` for this reason, and silence is
not verification).  (ii) It compares the amplitude, not the spacing, so the
claim lapses if a future change sources `rho_max` from anywhere else.  (iii)
**Under `vmap` the label is uninformative in BOTH directions**, so on the
`--mode flowmc-*` paths -- which is what production runs -- inheriting its
coverage buys nothing.  `_runtime_amp_failsafe` guards its callback with
`jax.lax.cond`; a BATCHED predicate lowers to a select, so both branches
execute, and the callback branch passes a literal `True` rather than the
predicate, so the recorded state cannot distinguish tripped from not-tripped.
Measured on a sound run with the predicate false by four orders of magnitude
(`amp_sizing` forced to 1e12 against a worst reported amplitude of 10.49):

| transformation | `tripped` | `n_calls` |
|---|---|---|
| plain call | False | 0 |
| under `jit` | False | 0 |
| under `vmap` | **True** | 2 |

That is a defect in `_runtime_amp_failsafe`, NOT in this PR, and it is
deliberately not fixed here -- it is pre-existing, it needs a host-side
predicate evaluation and its own validation, and folding it in would widen a
PR under review.  It is recorded here because it bounds what the sentence
above may be used for: for the interior-undersizing mode (a), **treat the
runtime fail-safe as absent on any vmapped path** and rely on the build-time
sizing.  (Independently reported by the chip-05 session; confirmed here by the
measurement above.)

**(b) The maximizing distance is EXTERIOR to the prior support** (section 1a).
**The runtime fail-safe is BLIND to this one.**
`_runtime_amp_failsafe` applies the identical `jnp.clip(M_A/B0, x_min, x_max)`,
so it under-reads by exactly the same mechanism the build-time amplitude does:
`amp_call <= amp_sizing`, and its `amp_call > 2 * amp_sizing` trigger never
fires.  Nothing at runtime detects this regime.  It is handled by REFUSING at
build time instead, using the unclipped-amplitude diagnostic, which is why that
refusal is not optional and must not be softened into a fallback.

**(c) The detector's own accumulation, on the sky re-draw path.**
`estimate_angle_amplitude` re-draws the sky when its split-half check says the
maximum is still growing.  Both maxima -- clipped and unclipped -- must be
updated there; updating only the clipped one leaves `clip_excess` reading the
first batch against a denominator that grew, so it falls BELOW 1 and the
refusal in (b) stops firing on exactly the events whose sky sampling was too
coarse to trust.  External re-review deleted that update and all 30 tests of
the gate stayed green, so the two are now accumulated in the same idiom on
adjacent lines and guarded at the source
(`test_sky_doubling_updates_the_unclipped_maximum_too`).  A behavioural test is
not available, and the reason is worth stating precisely because the obvious
one is wrong: the re-draw branch IS reachable (19 of 120 searched
`(data seed, n_sky, seed)` combinations enter it, one of them on an exterior
support).  What is not reachable is a DIFFERENCE.  The deterministic
face-on/face-off extremes are appended to the FIRST batch and are what attains
the unclipped maximum, so the second batch's unclipped contribution was a no-op
in every configuration measured -- deleting the update leaves `clip_excess`
bit-identical at 2.42571586419 on the one exterior doubling case available.
The corruption is therefore real but silent, and nothing bounds a dataset whose
unclipped maximum comes from a second-batch draw.  A test that cannot be made
to fail is the kind this file has already deleted once, so the guard is on the
source instead.

The contract in section 1 is therefore stated as holding uniformly over the
prior range and over every angle sample **given that the maximizing distance is
interior** -- a precondition that is checked, and refused when violated, rather
than assumed.

`--distance-grid-scheme uniform` (the default) is untouched by all of this.

### Refused combinations

Two of these are refused in the CONSTRUCTOR only; the rest also fail at
option-validation time, before any precompute.

* the exterior maximizing distance, and it cannot be otherwise: it is a
  property of the DATA, not of the option set, so nothing before the precompute
  can see it;
* `JAX_ILE_DISTGRID_ADAPTIVE=1` together with a non-uniform scheme.  This one
  COULD be caught at parse time -- it is visible from the option set plus the
  environment, exactly like the `JAX_ILE_DISTMARG_GH` row -- and is not, which
  is an inconsistency rather than a necessity.  It is left alone here because
  that variable is deprecated and its branch additionally requires
  `guess_snr`, so the parse-time check could not reproduce the constructor's
  condition without duplicating it.  Recorded so the asymmetry is deliberate
  rather than overlooked.  (Found by external review of the fix round.)

| combination | why |
|---|---|
| exterior maximizing distance | section 1a: 1.9-4.6 nats, worse than the default |
| `JAX_ILE_DISTMARG_GH` set | `core._distmarg_gh_logL` places its own per-sample nodes and reads only the SUPPORT of `x_grid`, so the option would be bit-identically inert while still reported as active.  Reachable without the user naming a dense scheme: under GH `choose_angle_marg_scheme` resolves to one regardless, and this refusal does not depend on which |
| `--angle-marg-scheme grid` | the sizing amplitude is not computed on that path |
| a mode other than `flowmc-phipsimarg` | not validated there |
| `--distance-grid-points` also given | two options setting the same node count |
| `--distance-grid-tol` with the uniform scheme | would be inert |
| an unrecognised scheme value | optparse `choices` |

## 5a. Verified through the driver, not only through the library

Run on the reference configuration with `--srate 4096`, `--mode
flowmc-phipsimarg --angle-marg-scheme exact --distance-grid-scheme loguniform`,
the driver reports:

    Distance + phi_ref + psi marginalization: ON (grid=256, nphi=8, npsi=8, d in [1,10000] Mpc)
      angle-marg scheme: exact (requested exact): amp_sizing=1109.1705986675768; amplitude=1109.1705986675768; ...
      distance grid: dlnd=0.04093484609767193; mode=loguniform; n=226; n_uniform_requested=256; rho_max=47.09926960511334; tol=0.01

(`amp` differs from section 2's 1516.33 because the driver computes its own
fiducial epoch rather than the fixed one the study scripts pass; the derived
node count tracks it, which is the behaviour under test.)  Note that the
`grid=256` on the first line is `--distance-grid-points`, which the log-uniform
scheme does not use; the authoritative line is the one below it, and it is
printed unconditionally.

All four fail-closed refusals were exercised the same way and each raised:

| what was passed | result |
|---|---|
| `--distance-grid-scheme loguniform --mode flowmc-phimarg` | `... apply only to --mode flowmc-phipsimarg ...` |
| `--distance-grid-scheme loguniform --distance-grid-points 256` | `... both set the distance node count ...` |
| `--distance-grid-tol 0.1` with the uniform scheme | `... it would be silently inert here.` |
| `--distance-grid-scheme adaptive` | optparse: `invalid choice: 'adaptive'` |

---

## 6. Reproducing

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
    export ADAPT_TREE=$PWD PYTHONPATH=$PWD/MonteCarloMarginalizeCode/Code
    taskset -c 0-15 python -u dist_quad_error2.py 40 256    # section 2a
    taskset -c 0-15 python -u support_window.py 40          # section 4b
    taskset -c 0-15 python -u e2e_laplace.py   40 64 4      # sections 2b, 3
    taskset -c 0-15 python -u compile_vs_execute.py 64 5    # section 3, compile vs execute
    taskset -c 0-7  python -u verify_peak_snr.py            # section 4a, guess_snr

Gate: `MonteCarloMarginalizeCode/Code/test/jax/test_distance_grid_loguniform.py`,
run by `.travis/test-jax.sh`.  Every test in it was verified to fail under a
named mutation; the matrix is in the PR body.
