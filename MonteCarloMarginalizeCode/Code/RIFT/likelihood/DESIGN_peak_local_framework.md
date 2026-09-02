# Peak-local marginalization as a framework: the axis contract

Companion to no single module yet — this is the planning note for consolidating a
method that now exists three times.  Read
`DESIGN_time_marginalization_quadrature.md` and then
`DESIGN_time_marginalization_peak_local.md` first: this note assumes the time
instance's vocabulary (enumeration, localisation, merged intervals, the tail bound,
fail-closed delegation) and does not re-derive it.

Everything measured here was measured on `citlogin6` (CIT), CVMFS IGWN python 3.11,
`OMP_NUM_THREADS=1`, on `rift_O4d` at `6e5bd4b1` (PR #205 merged).  Where a claim is
NOT measured it says so.  This note is a design record, not a shipped-behaviour
record: nothing in it has been implemented.

## The method has been written three times, independently

| | **time**<br>`time_marginalization_peak_local` | **angle**<br>`jax_ile/anglemarg.py` | **distance**<br>`jax_ile/core.py` |
|---|---|---|---|
| sharpness measure | `peak_width_from_lnL` → σ_t | `estimate_angle_amplitude` → A | `R`, analytic |
| resolution derived from it | `required_upsample_factors`, h ≤ σ/2 | `_dense_grid_sizes`, N = K√A | node scale 1/√R |
| rule selection | `_classify_rows` | `choose_angle_marg_scheme` | — |
| local placement | interval around each enumerated crest | dense (φ,u) grid, or Laplace | nodes centred on x\*=K/R |
| runtime check | dense remeasure + tail-bound certificate | `_runtime_amp_failsafe` | — |
| policy on failure | **fail-closed** → dense rule | **fail-open** → warn + label | — |

`_distmarg_gh_logL` (`jax_ile/core.py:1405`) states the shared motivation in its own
words: *"the (1/SNR)-narrow high-SNR distance peak that a fixed uniform-in-d grid
under-resolves (biasing the average ~1% low) is resolved at EVERY SNR."*  That is the
time module's opening argument with one noun changed.  Three authors, three axes, one
skeleton — measure the sharpness, derive the resolution, place work where the mass is,
check afterwards that you were entitled to.

## The organizing principle: the EXPONENT is band-limited, never the integrand

The single most costly misreading available here — I made it once already this
session — is to ask whether `exp(lnL)` is band-limited.  It never is.  On every axis
`exp(lnL)` is a needle, and resolving the needle is precisely the expensive thing
peak-local exists to avoid.

What the method actually requires is that the **smooth exponent** be band-limited, or
otherwise structurally constrained.  For time that object is `kappa(t)`; for
polarization it is `g(ψ)`; for distance it is `A u − B u²` plus the prior term.  Every
question about whether the method transfers to a new axis is a question about that
object, and about nothing else.

Read that way, the band limit plays **four** distinct roles for time, and they
generalize separately:

1. **Enumeration completeness.**  `kappa` is band-limited below Nyquist, so its
   narrowest possible lobe is `deltaT` and a fixed, SNR-independent `PEAK_ENUM_FACTOR`
   provably brackets every extremum.  This is what makes "enumerate, do not search"
   honest.
2. **The derivative bound `M_k`.**  `sum_j |X_j| |w_j|^k` is a TRUE bound because the
   interpolant is a finite trig sum.  Three consumers: the crest pre-filter (`M2`), the
   certificate remainder (`M4`), and — the round-6 lesson — it is the only place an
   inequality rather than a targeting estimate is available at all.
3. **The reconstruction / evaluator.**  `bandlimited_spectrum`, `eval_bandlimited_*`,
   `enum_grid_derivatives`, and the even-reflection contract: ~400 lines, and the source
   of a whole defect family (Nyquist-bin splitting, odd/even `n`, periodic-vs-reflected
   drift).
4. **It DEFINES the ground truth.**  For time the continuous integrand does not exist
   independently of the samples; it *is* the band-limited interpolant.  "Exact" is
   therefore a claim about reconstruction.

Roles 1 and 2 are what the method needs.  Role 3 is a tax time pays because it only has
samples.  Role 4 is the deep one: on every other axis the integrand exists on its own
and every evaluation of it is truth, so role 3 disappears and role 4 inverts — but
roles 1 and 2 must then be re-sourced, because "the samples determine the function" was
doing the certifying.

## The completeness warrant

Whether an axis can be certified at all, and therefore what its fail policy may be, is
decided by ONE property: what guarantees that no mass was missed.  Three kinds occur in
this codebase, and they are not interchangeable.

**`exact-band-limit` — time.**  `kappa`'s spectrum is identically zero above `fmax`.
Certificate available; fail-closed correct.

**`exact-trig-degree` — polarization ψ, and the strongest case of the three.**  The
exponent at `factored_likelihood.py:943` is

    g(ψ) = term2a + Re(term1 · e^{+2iψ}) + Re(term2b · e^{−4iψ})

Harmonics 2 and 4 only, so period π and an exact degree-4 trig polynomial.  Measured
over 4000 random coefficient draws spanning six decades of amplitude:

| | measured | bound |
|---|---|---|
| extrema of `g` on [0,π) | **4**, amplitude-independent | 4, from degree 4 |
| violations of `M_k = 2^k\|term1\| + 4^k\|term2b\|` | **0** | — |

So ψ gets all of roles 1–3 exactly and in closed form, and *more cheaply than time*: a
fixed ~16-point grid brackets every extremum forever, `M_k` is two terms rather than a
spectral sum, evaluation at arbitrary ψ is O(1), and the domain is genuinely periodic
so none of the reflection machinery applies.  ψ is not the hard axis.  It is the
easiest one, and it is the right second instance.

**`provable-unimodality` — distance.**  In `u = Dref/D` the plain-callback exponent is
`A u − B u²`, plus `−4 ln u` for the volumetric prior `p(d) ∝ d²`.  Measured: the full
exponent is **not** concave, but on (0,∞) it has exactly one interior maximum — the
smaller root of `B u² − A u + 4` is a minimum, the larger a maximum.  Completeness is
free; no certificate is needed because there is nothing to miss.  Note the shipped node
centre `K/R` is the peak of the Gaussian factor only, so it is offset from the true
maximum: measured **−0.08σ at K=50,R=1** but **−1.6σ at K=4.1,R=1**.  Benign at the SNR
this exists for, but a heuristic, not a bound, and undocumented at the call site.

**`effective-bandwidth` — the existing angle grid, and the one that CANNOT be
certified.**  `_dense_grid_sizes` sizes N = K√A.  That constant is not arbitrary:
`exp(A cos φ)` has a Gaussian-in-k coefficient envelope `exp(−k²/2A)`, and the measured
cutoff index over `A = 25 … 10000` is

| A | 25 | 100 | 450 | 2000 | 10000 |
|---|---|---|---|---|---|
| k(10⁻⁸)/√A | 6.40 | 6.20 | 6.13 | 6.08 | 6.08 |

Stable to 5% over three decades — the rule is well-founded.  But the coefficients never
reach zero.  The cutoff is a TOLERANCE, not a band limit, so no true `M_k` exists and no
certificate is possible.  Hence the 2× margin and the runtime failsafe, which is the
correct design for that warrant.

**This is why the fail policies differ, and why unifying them would be a bug.**  Time
is fail-closed because it can prove what it dropped.  The angle grid is deliberately
fail-OPEN — warn and label, do not poison — because NaN is silently filtered by flowMC,
the SMC path, and `write_samples`, so failing closed there would EXCISE the hot sky
region and publish a clean-looking posterior over what remains.  Both are right.  A core
that imposes one on the other is a regression in whichever axis it is imposed on.

## What the existing angle instance is, and is not

`anglemarg.py` sizes a dense grid for `exp(lnL)` and adds a Laplace branch above a
measured crossover.  In the time module's vocabulary that is the **dense rule** — the
analogue of `time_marginalize_bandlimited` — plus a peak-approximation shortcut.  It is
*not* a peak-local rule: nothing enumerates, nothing localises, nothing bounds omitted
mass, and there is no fallback ladder.  So the first concrete payoff of this framework
is not deduplication.  It is giving the angle axis a peak-local branch it does not have,
with a certificate that ψ's structure makes exact.

## Anti-goals

* **Do not unify the fail policy.**  It follows from the warrant.  See above.
* **Do not let the core own per-axis constants.**  `W_SIGMA`, `PEAK_KEEP_NATS`,
  `TAIL_LOG_TOL`, `MAX_INTERVALS` are inequalities over *time's* dynamic range.  Each
  axis re-derives its own and asserts its own; the core demands that the assertions
  exist, and supplies no values.
* **Do not accept a fitted `M_k`.**  The recurring defect in this module's history is a
  targeting model promoted to a bound (Door 4: "off by 122 nats"; the crest estimate was
  one octave too optimistic four times running).  An adapter interface that accepts an
  estimated bound institutionalizes that defect.  Either the adapter supplies a proof-
  carrying bound or a completeness certificate, or the axis is REFUSED — in the style of
  the existing `phase_marginalization` refusal, not approximated.
* **Do not let sub-cell certificate geometry leak into adapters.**  Whole-cell bounding
  does not work: on sharp rows the merged interval is narrower than one enumeration cell
  (half-width 0.05 of a cell at derived factor 4096), a cell-granular covered mask marks
  nothing, the crest's own cell counts as outside, the bound then bounds the CREST, and
  every row is rejected — the option goes inert.  Measured.  If this geometry lives in
  the adapter, every new axis rediscovers it.
* **Do not extract before the second instance exists in draft.**  `_classify_rows` is
  the house precedent and it was extracted only AFTER the duplicated policy had drifted
  three times.  Of ~1600 lines here, perhaps 300 are the generic skeleton; the rest is
  the time evaluator plus eight rounds of measured adversarial fixes that a speculative
  core would inherit without having earned.

## Evidence that the certificate arithmetic belongs in ONE place

`parabolic_sup` shipped, then needed a second numerical-robustness pass after merge
(`87a7a98b`): normalized discriminant to stop `inf − inf = NaN` erasing genuine
stationary points from a purported upper bound, cancellation-safe quadratic roots
(`q = −½(B + copysign(√disc, B))`, other root from `C/q`) because the direct form loses
the in-range root of a nearly-quadratic Hermite cell — which happens naturally for a
symmetric band-limited crest, where `a` should vanish and the endpoint arithmetic leaves
a few ulps.  Two rounds of hardening on 30 lines of pure arithmetic.  Copied per axis,
that is a defect per axis.

## Open, and not established here

* The composition/nesting contract.  "Peak-local in several places" means NESTED
  marginalizations, and the time rule's refusal of phase marginalization is the existence
  proof that nesting couples widths: under phase marginalization the time peak's Laplace
  width picks up an `(I1/I0)(|kappa|/D)` factor that does not reduce, so the local spacing
  stops being derivable.  Three individually-correct axis tools whose composition is
  unsound is the failure mode to design against.
* Whether ψ-marginalization is actually moving into the vectorized likelihood.
  `NetworkLogLikelihoodPolarizationMarginalized` is on the old non-vectorized API and
  production samples ψ by Monte Carlo.  NOT verified.
* The harmonic degree of the φ_orb exponent (asserted ≤ 2·Lmax via the Ylm crossterms).
  NOT verified — check the crossterm conventions before relying on it.
* Whether cosmological distance priors preserve the unimodality measured above for the
  volumetric prior.  NOT checked.
