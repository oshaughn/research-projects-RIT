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

Harnesses (host-local, `ldas-*` NFS home): `~/pl_framework_harness/`.
`psi_warrant.py` the psi extrema count, the `M_k` bound, and the shipped-function
check; `effective_bandwidth.py` the `exp(A cos phi)` coefficient envelope;
`distance_unimodality.py` the distance stationary-point structure and the node-centre
offset; `psi_bracket_adversarial.py` the near-annihilation extremum separation that
falsified this note's first bracketing claim.  Each runs standalone with `PYTHONPATH` set to `MonteCarloMarginalizeCode/Code`,
so a reviewer can re-run rather than take the tables on assertion.

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

1. **Enumeration RESOLUTION — and note what it does and does not buy.**  `kappa` is
   band-limited below Nyquist, so its narrowest possible LOBE is `deltaT`, and a fixed,
   SNR-independent `PEAK_ENUM_FACTOR` places 8 points across it whatever the SNR.  That
   is what makes "enumerate, do not search" affordable.  It is NOT a guarantee that every
   extremum is separately bracketed: a band-limited function can carry a max and a saddle
   approaching annihilation, arbitrarily close together (measured for the ψ case below).
   The shipped module is already careful about exactly this — *"Completeness of the
   enumeration buys SPEED; the bound buys CORRECTNESS"* — and a generalized core must
   inherit that division rather than the looser reading.
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

So ψ gets all of roles 1–3 exactly and in closed form, and more cheaply than time:
`M_k` is two terms rather than a spectral sum, evaluation at arbitrary ψ is O(1), and the
domain is genuinely periodic so none of the reflection machinery applies.  ψ is not the
hard axis.  It is the easiest one, and it is the right second instance.

**But an earlier draft of this note claimed a fixed grid "brackets every extremum,
forever", and that is FALSE.**  It is recorded here rather than quietly fixed, because it
is the project's characteristic error — a bound on the extremum COUNT silently promoted
to a guarantee about RESOLVING them.  A degree-2 trig polynomial can carry a maximum and
a saddle that approach annihilation, so adjacent extrema come arbitrarily close.  Measured
on a plain 3000-draw random family: **minimum adjacent-extremum separation 0.068 rad,
against a 0.196 rad cell on a 32-point grid** — two extrema in one cell, with no tuning
required.  Driven deliberately toward the bifurcation:

| `c1/c2` | 3.9 | 3.99 | 3.999 | 3.99999 |
|---|---|---|---|---|
| min separation (rad) | 0.224 | 0.0707 | 0.0224 | 0.0022 |
| value spread of the merging pair (nats) | 1.3e-3 | 1.3e-5 | 1.3e-7 | **1.2e-11** |

Two things follow, and both matter for the contract.

*What is actually true* is the weaker pair of statements the method needs: the extremum
COUNT is bounded by 4 and is amplitude-independent (which is what sizes `MAX_INTERVALS`),
and the pair that can hide inside one cell is a max/saddle whose value difference vanishes
as roughly the fourth power of their separation — 1.2e-11 nats at 0.0022 rad — so missing
it cannot lose mass.

*And the certificate is not optional for ψ.*  `segment_sup_bound` bounds `max q` over a
cell from its endpoint values, slopes and `M4` remainder, **regardless of how many extrema
are inside it**.  So an unenumerated maximum in a covered cell is bounded anyway, and one
in an uncovered cell is carried by the tail bound.  An adapter with an `exact-trig-degree`
warrant might look like it could skip the certificate on the strength of its extremum
count; it cannot, and the contract must not let it.

**`bounded-stationary-set` — distance.**  In `u = Dref/D` the plain-callback exponent is
`A u − B u²`, plus `−4 ln u` for the volumetric prior `p(d) ∝ d²` — which is the exponent
the shipped `_distmarg_gh_logL` sums over its nodes (`jax_ile/core.py:1454`:
`K u − ½R u² − 4 ln u`, so `A = K` and `B = R/2`).  It is **not** concave, and — an
earlier draft of this note said otherwise — it is **not** unimodal either, nor does it
always have an interior maximum.  What is provable is weaker and support-dependent:

    u·e′(u) = −(2B u² − A u + 4) = −(R u² − K u + 4)

so the stationary points are the roots of a quadratic, which are **real only when
`A² ≥ 32B`, i.e. `K² ≥ 16R`**; when they exist they are
`u_± = (K ± √(K² − 16R)) / (2R)`, the smaller a minimum and the larger a maximum.  Two
regimes follow, and the second is the one an implementation would get wrong:

* **`K² < 16R`.**  The quadratic has no real root and is positive throughout, so
  `e′ < 0` on all of (0,∞): the exponent is strictly DECREASING and there is no interior
  maximum at all.  On the physical support the maximum sits at the lower endpoint
  `u_min = Dref/D_max`.
* **`K² ≥ 16R`.**  `u_+` is an interior maximum only if it lies inside the support, and
  even then it must be compared against the endpoints — on (0,∞) the exponent runs to
  `+∞` as `u → 0⁺` (the `−4 ln u` prior term), so there is no global maximum on (0,∞) and
  the integral is finite only because `u_min > 0`.

So the completeness that is actually available is over the **support-aware** candidate set
`{u_min, u_max} ∪ ({u_+} ∩ [u_min, u_max])` — at most three points, at most two of them
local maxima, all in closed form.  That is still cheap and still certificate-free, but it
is *not* "nothing to miss": an adapter that enumerated interior stationary points alone
would silently drop the boundary-dominated mass.  The shipped code already anticipates
that regime (how often it fires is NOT measured here) — it clips the node centre into
`[x_min, x_max]` exactly so
that bins whose Gaussian peak falls outside the support "still get their
boundary-dominated integral resolved" (`jax_ile/core.py:1437-1442`).  A peak-local
distance adapter inherits that obligation.

Note also the shipped node centre `K/R` is the peak of the Gaussian factor only — it
ignores both the `−4 ln u` prior term and the endpoints — so it is offset from the true
maximum: measured **−0.08σ at K=50,R=1** but **−1.6σ at K=4.1,R=1**, and that second case
sits barely above the `K² ≥ 16R` threshold (16.8 against 16), which is precisely where the
interior maximum is weakest and the endpoints take over.  Benign at the SNR this exists
for, but a heuristic, not a bound, and undocumented at the call site.

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

## The seam is "supply a spectrum", not "supply an evaluator"

Both smooth exponents in scope are **finite exponential sums**.  Time:
`q(t) = Re sum_j Xw_j exp(w_j t)`, an `npts`-term spectrum.  Polarization, in the
`u = 2ψ` chart, is `g(u) = a + Re(c1 e^{iu}) + Re(c2 e^{2iu})` — a **two-term spectrum
with frequencies {1, 2}** (`jax_ile/anglemarg.py:961` states exactly this).

That is not an analogy.  MEASURED: calling the shipped time function
`spectral_derivative_bound(Xw=(c1,c2), fk=(1,2), period=2π, order=k)` — unmodified,
no new code — returns ψ's bound `|c1| + 2^k|c2|` over 300 draws at orders 1, 2, 4,
with **zero violations and tightness exactly 1.000**, i.e. the bound is *attained*
(with two terms the triangle inequality is achievable).

So the primitives the time rule built are already axis-free; they are merely misfiled
as time code.  `spectral_derivative_bound`, the Newton in `localise_peaks`,
`eval_bandlimited_points`, and the whole certified-supremum stack (`parabolic_sup`,
`segment_sup_bound`, the sub-cell cuts) consume only *(coefficients, frequencies,
period)* and cell geometry.  What an axis must supply beyond a spectrum is the short
list a spectrum cannot carry: domain topology, the lnL map, the width source, the
backstop, and the fail policy.

The reflection machinery is not on that list.  It exists because time's spectrum is
*implicit in coarse samples of a truncated window* — reflection is how the time adapter
**manufactures** its spectrum.  ψ's spectrum is explicit input.  Reflection is adapter
internals, invisible to the core.

## The axis contract

**The warrant is a closed, typed union owned by the core**, with four kinds — the three
above plus `effective-bandwidth-with-margin`.  The fourth is in the type *so the core can
refuse it by name*:

    if axis.warrant.kind is EFFECTIVE_BANDWIDTH_WITH_MARGIN:
        raise NotImplementedError(
            "peak-local requires a warrant that certifies enumeration completeness and "
            "true derivative bounds; effective-bandwidth-with-margin certifies neither "
            "(it is the sizing rule for a DENSE grid with a runtime failsafe).  Use the "
            "axis's dense rule; peak-local cannot be built on it.")

Same texture as the existing `phase_marginalization` refusal: name the reason, name the
alternative, refuse rather than degrade.

**A fitted `M_k` is made unrepresentable, not merely discouraged.**  `derivative_bound`
is NOT an adapter-overridable method.  The adapter supplies `spectrum(rows)` and the
**core** computes `M_k` by the triangle inequality — the one construction that cannot be
a fit.  An axis that cannot express its exponent as an exponential sum but holds a
`bounded-stationary-set` warrant gets no `M_k` and no Hermite certificate; its tail control
is the bracket around the closed-form candidate set — interior stationary maximum AND both
support endpoints, per the distance case above.  There is no third path.  This is the structural answer to the
defect that reopened four times.

**The core never applies a fail policy.**  It returns `(values, ok, decline_ledger)`.
The axis-owning wrapper decides: time hands `~ok` rows to `time_marginalize_bandlimited`;
a jitted consumer labels and continues.  That is how "the fail policy differs by design"
survives consolidation — it becomes a property of the call site, expressed on the ledger,
rather than a branch inside the core.

Protocol members, each named for the site that consumes it: `domain` (span, periodic),
`warrant`, `spectrum`, `enum_grid`, `exponent_on_enum_grid`, `exponent_deriv_on_enum_grid`,
`eval_points`, `eval_uniform`, `lnL` (must be monotone), `peak_scale`, `viable_rows`,
`backstop`, `backstop_cost`, and the per-axis constants **together with the discharge
inequality each must register** for the suite to assert.

## ψ worked through, and where it falsifies the contract

| member | ψ supplies |
|---|---|
| domain | span 2π in `u`, **periodic** — no reflection, no pinning, no edge guard |
| warrant | `exact-trig-degree(harmonics=(1,2), period=2π)` |
| spectrum | `(c1, c2)`, `(1, 2)` → core `M_k = \|c1\| + 2^k\|c2\|`, exact and attained |
| enum_grid | 32 points on [0,2π): 8 per period of the top harmonic, the same discipline as `PEAK_ENUM_FACTOR`. Sizes the enumeration, and does NOT by itself guarantee separation — see the bracketing measurement above |
| eval | closed form, O(1)/point; `q'`, `q''` analytic — every evaluation is truth |
| lnL | identity |
| backstop | fixed-N trapezoid on the full circle — super-exponentially convergent for a periodic band-limited exponent, so the backstop itself carries an aliasing *bound*, unlike time's |

Two places ψ pushes back on the contract, and both should be treated as findings rather
than smoothed over:

* **`peak_scale` must admit truth-grade curvature.**  ψ has exact `σ = 1/√(−g''(u*))`.
  If the protocol forces the shared stencil estimator, ψ is pushed through an
  approximation it does not need.  (`σ` may be fitted in general — it *targets*, it never
  *bounds* — but the contract must not forbid exactness.)
* **`max_intervals` means different things on different axes.**  For time it is a cost
  guard.  For ψ the warrant *proves* ≤2 maxima, so exceeding it is a broken-adapter
  assertion.  The core may only count and decline; the meaning belongs to the adapter.

**Fail policy for ψ: fail-closed — and the reason is cost, not ideology.**  A cheap
certified backstop is available per row in numpy, so declining is affordable.  A future
jitted port inherits `anglemarg`'s constraint (static shapes, no per-row backstop) and
must go fail-open-with-label.  Same axis, different policy, decided by the call site —
which is exactly why the policy is a seam.

**Reconciliation with the existing `anglemarg`:** its *exact/dense* scheme is the ψ
analogue of `time_marginalize_bandlimited`; peak-local does not compete with it and the
core refuses its warrant.  It stays.  Its *Laplace branch* is what peak-local-ψ is a
certified replacement for — that branch already enumerates all maxima, then applies an
O(1/A) width model with documented worst-phase error and a blend band that absorbed three
review rounds.  But not in one step: the core does host-side ragged bookkeeping while
`anglemarg`'s kernel runs under `lax.scan` with static shapes.  The numpy ψ instance
exists to prove the *contract*; porting it into the Laplace slot is a separate PR with its
own gates.

## Migration, and the gate at each step

The house rule is `_classify_rows`: no extraction before the second instance exists in
draft.  Gate numbers below are measured on `rift_O4d` at `6e5bd4b1`.

**Step 0 — pin the baseline.**  Hash `time_marginalize_peak_local` outputs over a mixed
block.  The bar for every later step is **exact identity, not ULP-close**: pure code
motion cannot reassociate floats.  Plus, with the test files UNTOUCHED:

| gate | files | count |
|---|---|---|
| peak-local | `test_time_marginalization_peak_local.py` | **121** collected |
| band-limited | `..._quadrature.py` (81) + `..._quadrature_pipeline.py` (57) + `test_continuous_time_posterior_export.py` (23) | **161** collected, 160 passed, 1 skipped (cupy absent) |

**Step 1 — write ψ as a draft that imports from the time module and duplicates
deliberately.**  Importing across modules is the safe direction; it is *copying policy*
that rotted.  Gate: agreement with a converged `quad` reference over an amplitude ladder
and adversarial phases; enumeration-completeness and `M_k` property tests; time hash
unchanged (trivially — nothing has moved).

**Step 2 — diff the two spines, and extract only what the diff proves shared.**  Move the
axis-free helpers to the core verbatim, leaving re-export shims so `__all__` keeps
resolving.  Gate: time hash identical, both suites green *unmodified*, plus a new test
asserting a helper is the **same object** in both consumers — the `is`-identity anti-drift
analogue of the existing classification-itself test, since a parity test cannot see a
change both sides read.

**Step 3 — formalize the contract types**, deliberately *after* step 2: the protocol is
transcribed from the seam the diff exposed, not designed ahead of it.  Gate: refusal tests
executable, time hash still identical, public signature unchanged.

**Falsifiable prediction, recorded now so step 2 can settle it:** the plan/bucket skeleton
of `_peak_local_chunk` will **not** extract.  It exists because time's interval and point
counts are ragged; ψ has ≤2 maxima and a fixed per-interval point count, so its natural
shape is fixed-slot arrays with no plan, no buckets, no host round trip.  What should
extract from that spine is the two-stage keep discipline, the accounting-reconciliation
ledger, and the accept predicate.

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
* **Do not own row classification or viability.**  `_classify_rows` stays where it is
  for time; another axis brings its own dispatcher.  The core never sees `factors`,
  ceilings, or edge guards.
* **Do not call the likelihood on a full axis grid, and do not call the backstop.**  All
  exponent evaluation routes through the adapter, and the backstop is invoked only by the
  policy-owning wrapper — otherwise the core smuggles a fail policy in through the back
  door.
* **Do not carry cross-call state.**  Batch-local only; any persistent scale makes results
  batch-order-dependent.
* **Do not silently widen.**  Every decline goes on the ledger under a named reason, with
  the reconcile invariant that the sub-counts sum to the declined rows.  A change that adds
  an unledgered decline path must fail a reconcile test.
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

* **The composition/nesting contract**, and it is the sharpest open question.  Nesting
  means the outer axis's exponent is the inner MARGINAL, `G(t) = log int exp(g(t,psi)) dpsi`,
  and `G` inherits neither the warrant nor the derivative bounds of `g`.  Concretely, under
  the tilted inner measure `mu`,

      d2G = E_mu[d2 g] + Var_mu(d_t g)

  and the variance term scales with inner amplitude — so **the enumeration factor for `G`
  is not SNR-independent**, which is the pillar the whole time construction stands on.
  The existing refusal of phase marginalization is the special case: the outer exponent
  becomes `log I0(|kappa|/D)`-shaped, its width carries `(I1/I0)(|kappa|/D)`, and `|kappa|`
  peaks elsewhere than `Re kappa`, so both the width derivation and the monotone-argmax
  separation die at once.

  Two classes, and only one is open:

  - **Monotone-reduction inner — shipped, works, and should be NAMED.**  If the inner
    marginalization is presented as a callback monotone in the outer exponent at fixed row,
    argmaxes are preserved, the outer warrant is untouched, and the width is measured
    through the callback.  Distance-inside-time is this class and is the production
    configuration today.  It is already the `lnL` + monotone member of the contract.
  - **Genuine inner peak-local — DEFERRED, with named preconditions.**  The core composes
    nothing.  Before the contract can be written, two things must exist: (i) a cumulant
    bound giving `sup|d_t^k G|` from `M_k(g)` and inner-measure moment bounds, with the
    `I1/I0` case as its acceptance test — SPECULATION: such bounds plausibly exist but grow
    with inner amplitude, and whether they stay tight enough to beat the dense alternative
    is unmeasured; and (ii) a measured completeness study for enumerating extrema of `G`,
    since a fixed factor is provably insufficient.  Until both exist, every nested pairing
    refuses per pairing, naming the missing lemma.
  - **The practical bridge that needs neither:** `return_peaks=True` already exports
    `(t_star, sigma)` per row, callback-independent.  A time-first reordering composes over
    PEAK SETS rather than over nested exponents — which is why the peaks are an output and
    not a temporary.
* Whether ψ-marginalization is actually moving into the vectorized likelihood.
  `NetworkLogLikelihoodPolarizationMarginalized` is on the old non-vectorized API and
  production samples ψ by Monte Carlo.  NOT verified.
* The harmonic degree of the φ_orb exponent (asserted ≤ 2·Lmax via the Ylm crossterms).
  NOT verified — check the crossterm conventions before relying on it.
* Whether cosmological distance priors preserve the stationary-point structure derived
  above for the volumetric prior — the quadratic, its discriminant, and hence the size of
  the candidate set all come from the `−4 ln u` term specifically.  NOT checked.
