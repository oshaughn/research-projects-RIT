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
falsified this note's first bracketing claim; `psi_backstop.py` the trapezoid aliasing
table and the calibrated-vs-certified sizing cost.  Each runs standalone with `PYTHONPATH` set to `MonteCarloMarginalizeCode/Code`,
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
| backstop | trapezoid on the full circle at **N derived from amplitude**, `N ≈ 6.2·√A` for a 1e-8 relative tolerance. NOT certified at that N — see below |

Two places ψ pushes back on the contract, and both should be treated as findings rather
than smoothed over:

* **`peak_scale` must admit truth-grade curvature.**  ψ has exact `σ = 1/√(−g''(u*))`.
  If the protocol forces the shared stencil estimator, ψ is pushed through an
  approximation it does not need.  (`σ` may be fitted in general — it *targets*, it never
  *bounds* — but the contract must not forbid exactness.)
* **`max_intervals` means different things on different axes.**  For time it is a cost
  guard.  For ψ the warrant *proves* ≤2 maxima, so exceeding it is a broken-adapter
  assertion.  The core may only count and decline; the meaning belongs to the adapter.

**Fail policy for ψ: fail-closed, but to an UNCERTIFIED backstop — and that distinction
has to be carried, not glossed.**  Declining is affordable per row in numpy, so
fail-closed is the right mechanism.  What an earlier draft got wrong is the *status* of
the thing it declines to: at the calibrated `N ≈ 6.2·√A` the backstop's accuracy rests on
a sizing rule with a margin, exactly like `anglemarg`'s dense scheme, and it is therefore
an `effective-bandwidth-with-margin` object.  A certified backstop is available (below)
but costs ~24× the nodes at `A = 10⁴`.  So the honest statement is: peak-local-ψ's own
answer is certified by the sub-cell Hermite bound on `g`; the rule it falls back to is
not, unless the certified sizing is paid for.  A future jitted port inherits
`anglemarg`'s constraint (static shapes, no per-row backstop) and must go
fail-open-with-label.

### RETRACTED: "the backstop carries an aliasing bound"

An earlier draft called the ψ backstop a *fixed*-N trapezoid, "super-exponentially
convergent for a periodic band-limited exponent, so the backstop itself carries an
aliasing bound".  **That is false**, and it is the same error this note diagnoses
elsewhere: super-exponential convergence is in `N` at FIXED coefficients, not uniform
over amplitude.  `g(u) = A cos u` is degree one, but `exp(g)` has width `∝ A^{-1/2}`, so
any fixed grid eventually under-resolves it.  The note already contained the disproof —
the `exp(A cos φ)` coefficient envelope measured two sections above is the same object.

For the `N`-point trapezoid on a `2π`-periodic `f` the error is *exactly* the aliasing
sum `T_N − I = 2π Σ_{m≠0} ĉ_{mN}`, and for `f = exp(A cos u)` the coefficients are
`I_k(A)`, so the relative error is `2 Σ_{m≥1} I_{mN}(A) / I_0(A)` — computable, not
estimated.  Measured:

| relative error | N=16 | N=32 | N=64 | N=128 | N=256 |
|---|---|---|---|---|---|
| A = 25 | 1.3e-2 | 1.4e-8 | 4.6e-28 | 7.6e-85 | 4.8e-236 |
| A = 450 | 2.3e0 | 6.6e-1 | 2.1e-2 | 2.7e-8 | 2.7e-31 |
| A = 10000 | 1.2e1 | 6.8e0 | 2.9e0 | 9.6e-1 | 7.6e-2 |

Read the `N=32` column: the error grows without limit in `A`.  The required `N` follows
the same `√A` law as everything else here — `N(10⁻⁸)/√A` = 6.80, 6.40, 6.22, 6.22, 6.20,
6.19 over `A = 25 … 5×10⁴`.

**Can it be certified?**  Yes, and the price is measured.  For the two-term exponent the
coefficients of `exp(g)` are a Bessel convolution, so
`|ĉ_k| ≤ Σ_j I_j(|c2|) I_{k−2j}(|c1|)` is a true bound.  Converting it to a *relative*
bound needs a rigorous lower bound on `ĉ_0`; the only cheap one is Jensen — the harmonics
have zero mean, so `ĉ_0 ≥ e^{⟨g⟩} = 1` — which is valid but conservative by roughly
`e^A`:

| A | 25 | 450 | 2000 | 10000 | 50000 |
|---|---|---|---|---|---|
| `N` calibrated (relative, 1e-8) | 34 | 132 | 278 | 620 | 1384 |
| `N` certified (Jensen) | 50 | 692 | 3030 | **15100** | **75456** |
| ratio | 1.5× | 5.2× | 10.9× | 24.4× | 54.5× |

So the certified sizing grows like `1.5·A` against the calibrated `6.2·√A`, and the
penalty grows as roughly `√A/4.4`.  **This is an open design choice with a measured
price, not a settled fact**, and the contract must record which one an adapter takes —
because only one of them may be described as certified.

(Computed with a uniform-asymptotic `log I_k(A)`; a first attempt used `scipy.special.ive`
directly and silently underflowed to zero, which read as "tolerance met" and produced
certified-`N` values too small by up to 9×.  Validated against `scipy` wherever `scipy`
is still finite: max disagreement 5e-3 nats.)

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

## The primitive is PER-AXIS {localize, dense}, not "1-D vs 2-D marginalizer"

The natural-looking decomposition — a 1-D marginalizer primitive and a 2-D one — is the
wrong cut.  We do not always dual-localize, and the reason is physical, not a limitation
to be engineered away.  The right primitive is a per-axis CHOICE, and the existing schemes
are already members of one family rather than a ladder:

| | φ dense | φ localized |
|---|---|---|
| **ψ dense** | `..._exact`, cost ~A | (no use case found) |
| **ψ localized** | `..._laplace`, cost ~√A — SHIPPED | the joint high-SNR target |

Today's `laplace` is not a stepping stone to be replaced.  It is the correct member
whenever ψ is localizable and φ is not, and a core that only offers "all axes localized"
would delete a working configuration.

**The joint exponent is a 2-D trig polynomial, so the warrant extends verbatim.**  This is
already in the shipped code, not an assumption: `angle_coefficient_tables` builds *"Exact
2-D Fourier coefficient tables"*, `_reconstruct_field` evaluates *"the real trig polynomial
... at (phi, u)"*, and `angle_sample_grid_sizes` states the bidegree as derived —
φ-harmonics up to `2*m_max`, and u-harmonics **at most 2 for ANY mode set** (spin-2), so
the ψ degree never grows.  Measured on random tables of bidegree (4,2), 150 draws × 7
derivative orders: `M_(a,b) = sum_kq |C_kq| k^a |q|^b` had **0 violations**, tightness 0.89.
`spectral_derivative_bound` generalizes to a multi-index unchanged in construction.

## "Localized" ALWAYS means multi-mode.  Single-centre is not in the family

Stated first because it is the easy thing to get wrong when extending to a second axis.
The shipped ψ branch is already multi-mode — `_psi_lnI_lap_branch` is *"the
enumerated-maxima Laplace branch"*, resolving up to four roots — so "Laplace" in this
module has never meant "expand about one point".  The joint (φ,ψ) kernel must inherit that
structure on BOTH axes.  The measurement below is the quantitative reason, not a criticism
of anything shipped: it sizes what a naive single-centre joint extension would cost.

Measured on the shipped coefficient tables (`make_synth` fixture, real (φ,ψ) structure,
not random coefficients), at fixed `x`:

| kappa boost | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| exponent amplitude | 3.4 | 32.5 | 325 | 3250 |
| co-dominant maxima | 8 | 12 | 8 | 8 |
| **single-centre Laplace error (nats)** | **−1.66** | **−1.48** | **−1.40** | **−1.40** |

Two readings, and the second is the load-bearing one.

*The obstruction is MULTIPLICITY, not conditioning.*  `cond(H)` at the dominant mode
measured 11–23 — unremarkable.  What breaks a single-centre Laplace is that with dominant
(2,±2) content the (φ,ψ) surface carries ~8–12 maxima whose values are equal to machine
precision (measured second-best gap 0 to 4.6e-13 nats).  Adding (3,±3) leaves 11 maxima
within 0.031 nats.  These are exact structural degeneracies, and the count is
amplitude-independent — consistent with the fixed bidegree.

*And the single-centre error DOES NOT DECAY WITH AMPLITUDE.*  It sits at −1.4 nats from
amplitude 3 to 3250.  That is the opposite of the usual "Laplace improves at high SNR"
intuition, and the reason is that the deficit is combinatorial rather than curvature: one
centre represents one of k equal modes however sharp each becomes.  High SNR does not
rescue a single centre — it is precisely where enumerating the modes matters.  Which is
why the joint kernel is peak-local (enumerate, then integrate near each) rather than a
Laplace refinement, and why the enumeration budget, not the curvature model, is the thing
the φ-axis selector has to size.

HONEST LIMIT OF THIS HARNESS: the multi-mode sum measured +0.42 to +1.01 nats and did not
show clean `O(1/A)` convergence.  The per-mode Hessians here are grid finite differences on
a 512² torus, so that residual is plausibly the harness rather than the method — it is NOT
evidence that a multi-mode estimator converges, and a real localizer plus certificate is
needed before any such claim.  Recorded so the number is not quoted as a validation.

## Selection must not become an option matrix

The tree already shows the failure mode: `..._laplace` REFUSES `JAX_ILE_DISTMARG_GH`
(`jax_ile/anglemarg.py:1119`) because its node placement is defined per fixed-ψ exponent.
That is a pairwise option corner case, and a family of per-axis choices multiplies them if
each pair is hand-checked.  Two rules keep the surface from growing:

* **Selection stays keyed on measured quantities, per axis.**  `choose_angle_marg_scheme`
  already picks from the data-derived amplitude rather than a user flag; a new member of
  the family is a new regime on that same measured axis, not a new option.  The degeneracy
  measurement above is what the φ-axis selector must key on — multiplicity and mode gap,
  not amplitude alone.
* **An incompatibility is a property of a scheme's warrant, declared once**, never an `if`
  per pair.  The GH refusal is really "this scheme's node-placement warrant is defined per
  fixed-ψ exponent"; stated that way it is checked generically.  NOTE: distance
  marginalization is expected to arrive on the other path, so this particular refusal must
  be expressed as a warrant property and not frozen as a permanent pairwise rule.

## ENUMERATION AS A PRIMITIVE

The reusable building block, worked out abstractly because it is where the long-term
compute is saved: every axis needs it, and re-deriving it per axis is how the certificate
gets weakened by accident.

### The output is a certified COVER, not a list of peaks

Three tempting contracts all fail at the operating point:

* *"all critical points"* — the count is DISCONTINUOUS in the coefficients at a max/saddle
  annihilation, so it cannot be honoured continuously; and annihilation is normal here, not
  a corner (measured, above).
* *"all local maxima"* — with 8–12 maxima equal to machine precision, which of them are
  "the" maxima is a rounding accident.
* *"all modes above a mass threshold"* — the right question, wrong contract: mode mass is
  not knowable before integrating, so a threshold on estimated mass is exactly an estimate
  promoted to a bound.

What survives — and what the time module already implements, distributed across
`enumerate_peak_indices`, `merge_intervals_by_row` and `segment_sup_bound` — is:

> **ModeCover**: a finite set of disjoint regions `{R_i}`, each with a representative point
> and targeting data, plus a CERTIFIED bound `B_out >= sup{ g : outside the union }`, plus a
> ledger.

Three promises, in decreasing strength.  **(1) The bound** is the only correctness-bearing
one: it converts "did I miss a mode?" into an inequality the caller discharges, omitted mass
`<= |domain \ union| * exp(B_out)`.  **(2) Targeting** — each representative is a stationary
point to a stated residual — buys speed only, and is allowed to fail; failure surfaces as
`B_out` too large and the row declines.  **(3) There is NO count promise.**  The number of
regions is an output.  A merged max/saddle pair is one region; twelve machine-equal maxima
may be twelve regions or fewer if their covers overlap.

Mass therefore SELECTS (drop a region whose certified sup is below tolerance — a bound-based
rejection, safe) but never CERTIFIES PRESENCE.

### Warrants, by what finite object exhausts the stationary set

| class | axis | certificate | cost | amplitude? |
|---|---|---|---|---|
| **exact trig poly, 1-D** | ψ | fundamental theorem of algebra: `z = e^{iu}` gives degree `2n`; those `2n` roots are all of them | one `2n × 2n` companion eigenproblem | **independent** |
| **exact trig poly, multi-D** | joint (φ,ψ) | BKK / mixed volume | one algebraic solve of that size | **independent** |
| **band-limited, large spectrum** | time | algebraically possible (`2·npts` companion) but `O(npts³)` unaffordable | grid seeds + `segment_sup_bound` | independent |
| **closed-form stationary set** | distance | the algebra of that functional form; ≤3 candidates incl. support endpoints | `O(1)` | independent |
| **effective bandwidth** | the dense angle grid | NONE | — | refused by name |

The economic argument, stated plainly: under the first two, everything that sizes the
enumeration — degree, mixed volume, `M_k` — is invariant under `C -> λC`.  Amplitude enters
only downstream, in how narrow each mode's LOCAL grid must be.  Dense pays `√A` per axis
(so `~A` for the 2-D product); enumeration converts an amplitude-scaling cost into a
physics-scaling one.  That conversion is the whole justification.

**Time is the instructive case**: it is class-1 in principle but not in practice, so it
enumerates on a grid — which is NOT a certificate — and restores correctness at
CERTIFICATION time via the cover bound.  Classes 1 and 2 certify at ENUMERATION time.  Both
are certified; only where the cost is paid differs.

### The algebraic core, and the tolerance that must not exist

1-D: `P(z) = c2 z⁴ + (c1/2) z³ − (c̄1/2) z − c̄2` for ψ.  Roots on `|z|=1` are the critical
points.  Measured: residual `|g'|` at the roots is **3.4e-15** relative to `M_1` over 3000
draws spanning six decades; never fewer than a 2e5-point grid; **~20 µs/call, flat from
amplitude 1 to 1e6** — the amplitude-independence, demonstrated.

Grid seeding fails where it matters.  Driving `c1/c2 -> 4`:

| `c1/c2` | separation | grid-32 | grid-256 | grid-4096 | algebraic |
|---|---|---|---|---|---|
| 3.99 | 0.0707 | 1 | 3 | 3 | **4** |
| 3.99999 | 0.0022 | 1 | 1 | 3 | **4** |

(Verified this is genuine resolution loss, not a seam-wrap artefact: a circular counter gives
identical numbers.)

**AND A TRAP THAT WAS WALKED PAST ONCE, recorded so it is not walked past twice.**  The
obvious implementation filters roots by `||z| − 1| < tol`.  That tolerance is itself an
estimate promoted to a bound.  At EXACT multiplicity `m` the computed roots smear off the
circle by `ε_machine^(1/m)` — measured **4.6e-6 for a triple root** — so:

| on-circle tol | 1e-9 | 1e-7 | 1e-6 | 1e-3 |
|---|---|---|---|---|
| roots found at exact degeneracy (true: 4) | 1 | 1 | **1** | 4 |

A `1e-6` filter — a perfectly reasonable-looking choice, and the one first written here —
silently returns ONE mode where there are four, in precisely the machine-degenerate
configuration that is the production regime.  A conjugate-reciprocal pairing test does not
escape it either: measured, it returns counts identical to the naive filter at every
tolerance, because the partner is off-circle by the same amount.

**The fix is to have no tolerance at all**, and it falls out of promise 3.  Seed regions
from `arg(z)` of ALL `2n` roots and filter nothing.  Over-covering is free — redundant
regions merge — while under-covering is the only real danger.  Measured over 4000 draws
spanning six decades, every true extremum lies within **3.1e-4 rad** of a seed, which is the
reference grid's own resolution, i.e. exact.  And at exact degeneracy, where sign-change
enumeration finds **zero** extrema (`g'` touches zero without crossing), the algebraic seeds
still cover the point exactly.

### Degeneracy

Cluster, and treat a cluster as one region.  The codebase already contains this as
`merge_intervals_by_row`, whose docstring makes the deeper point: merging is not an
optimization, it is what prevents double-counting and what makes the method degrade
CONTINUOUSLY into the dense grid with no threshold anywhere.  Under ModeCover, clustering is
not even a special case — regions are built around every seed, and overlapping ones merge.

The invariant that expresses "do not split an annihilating pair" is upper-semicontinuity IN
THE MASS SENSE: as coefficients vary, the integral over the union plus the certified outside
bound must vary continuously, while the region COUNT is free to jump.  A primitive promising
only the union and `B_out` cannot be broken by the bifurcation, because nothing it promises
changes discontinuously there.  That is a testable property: perturb through a measured
bifurcation and assert the accepted mass moves continuously.

SPECULATION, not verified: at exact degeneracy the co-dominant modes look like orbit-mates
of a discrete symmetry, and quotienting would divide the algebraic degree by `|G|` and
replace `k` numerically-rediscovered copies by an exact `+ln k`.  Attractive, but the
measured gaps are 0 *to 4.6e-13* nats and the tables are built by sampled accumulation, so
the symmetry can be broken at roundoff.  A symmetry assumed exact when it is 1e-13-broken is
the same defect in a new costume.  Correct layering: numerical clustering stays load-bearing;
a declared symmetry may SEED clustering and tighten the budget, and the certificate verifies.

### Where enumeration loses — the exclusion region is part of the design

* **Low amplitude**: modes are wide, regions merge toward the whole domain, and the method
  IS the dense grid plus overhead.  Selection should not enter peak-local below crossover.
* **High degree**: cost is the algebraic solve regardless of how much mass matters.
* **Dimension ≥ 3**: mixed volume and solve cost grow multiplicatively; do not extend
  without a new measurement campaign.  Composing over peak SETS is the designed alternative.
* **No warrant**: refuse by name.

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
