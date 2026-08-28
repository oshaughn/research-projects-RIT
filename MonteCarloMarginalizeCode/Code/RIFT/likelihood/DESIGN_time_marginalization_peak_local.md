# Peak-local time marginalization: measured record

Companion to `time_marginalization_peak_local.py`, and a follow-up to
`DESIGN_time_marginalization_quadrature.md` — read that one first.  The module
docstring carries the argument; this file carries the numbers behind it and the
harnesses that produced them.

Everything here was measured on `ldas-pcdev` class CPU (CIT), CVMFS IGWN python 3.11,
`OMP_NUM_THREADS=1`, on branch `rift_O4d_tmarg_peaklocal` (based on
`rift_O4d_tmarg_bandlimited`, PR #203).  Harnesses: `~/tmarg_harness/` for the
prototypes, `~/pl_work/` for the measurements below.

## THE BUG THIS SHIPPED WITH, and the requirement that was missing

An adversarial review of PR #205 found a critical correctness bug.  It is recorded
first because the design statement below was *wrong*, not merely incomplete, and the
shape of the error is the reusable lesson.

The design named TWO resolution requirements.  **There are three.**  Enumeration
returns a grid INDEX, and an index is not a location: the true crest can lie
`h_enum/2` from the sample that reports it.  Localising it to a fraction of `sigma_t`
is a third requirement, it is SNR-DEPENDENT, and it belonged to neither of the two
named mechanisms — so it was assigned to neither and simply did not happen.  The
interval was built around `cols_np * h_enum`, the grid sample.  Whenever

    W_SIGMA * sigma_t  <  h_enum / 2      i.e.  sigma_t/deltaT < 0.0052

— any row with a derived factor of 512 or more — **the peak lay entirely outside its
own interval**.  Measured on the synthetic fixture at `sigma_t/deltaT = 0.0024`, error
against the dense path as the crest is walked off the enumeration grid:

| crest offset from the sample | 0 | h_enum/4 | h_enum/2 |
|---|---|---|---|
| peak-local − reference | +0.000000 | **−6.52** | **−164.93** |

Always negative: it deletes mass, never adds it.  At production scale (64 rows,
arrival times uniform w.r.t. the sample grid, `sigma_t/deltaT = 0.0023`): **median
−1.15 nats, worst −131 nats, 56% of rows wrong by >0.01 nats, and every one of them
ACCEPTED.**  A bias, not noise, silently deleting extrinsic samples.

**Why the suite was blind, which is the part worth internalising.**  Every sharp
fixture used `peak_sample = NPTS//2 + 0.25`, and the phase sweep in
`test_exact_on_a_periodic_window` was `[0.0, 0.25, 0.5]`.  All are exact multiples of
`1/PEAK_ENUM_FACTOR = 1/8`, so the crest landed EXACTLY on an enumeration sample in
every single test.  A bug that only exists *between* samples was invisible to all of
them.  That test's own docstring already said the phase is swept because "a fixture
pinned to one phase can be exactly wrong and look exactly right" — it was pinned, in
the only sense this module cares about.  **One added phase of 0.3125 catches all of
it.**  Sweeping a parameter is not the same as sweeping it over the quantisation that
matters.

Two further holes the same review found, both fixed:

* the tail bound could not catch this, because `q_out_max` is a maximum over
  *enumeration-grid samples* — the same grid that failed.  In the worst case it
  reported `tail_bound_worst = -275` (claiming `e^-275` of omitted mass) while dropping
  164.9 nats.  `T_outside` was also counted in grid indices, so a gap narrower than
  `h_enum` contributed to neither the length nor the maximum;
* `enumerate_peak_indices` excluded endpoints on the justification that the edge guard
  had already routed such rows away.  That is false: `exposed` keys on the row's
  GLOBAL coarse argmax, so a row with a mid-window dominant peak and a secondary peak
  hard against an edge was refined with the secondary peak never enumerated.

### The fix

1. **`localise_peaks`** — Newton on the spectral interpolant, seeded at the enumerated
   sample and confined to the bracket `[t_i - h_enum, t_i + h_enum]` that the
   enumeration already established.  It places peaks; it cannot find or lose one, which
   is what distinguishes it from the seed-and-hope that sank RIFT PR #201.  Quadratic
   convergence, so the SNR-dependence of this step is logarithmic: 2–4 iterations
   across the whole production range.  Convergence to `LOCALISE_SAFETY * sigma_t` is
   ASSERTED, the interval is widened by that residual, and a peak that misses it sends
   its row to the dense path.
2. **An a-posteriori containment check** — the local integration grids must ATTAIN the
   localised crest's `lnL` to within `CONTAINMENT_SLACK_NATS`.  Free (that maximum is
   already computed for the log-sum-exp offset) and it is what actually catches a
   mis-placed interval.  It compares against the LOCALISED crest, not the enumeration
   sample: against the sample it would pass precisely in the case it must catch, since
   an off-grid crest leaves the sample tens of nats low.
3. **Exact `T_outside`** from interval geometry rather than a grid-index count.
4. **Endpoints enumerated**, and the false justification deleted.
5. **The ceiling checked before the cost gate** (see below).

After the fix, the same sweeps: **0.000000 at every crest offset**, and at production
scale median +0.000000, worst +0.000000, 0/64 rows outside 1e-3.

`W_SIGMA`'s `erfc(W/sqrt2)` argument is a statement about a Gaussian truncated about
its CREST, so centring on the sample carried an unstated precondition
`W_SIGMA * sigma_t >= h_enum/2`, coupling `W_SIGMA` to `PEAK_ENUM_FACTOR` and nowhere
asserted — note that raising `PEAK_ENUM_FACTOR`, the intuitively safer move, would have
made it *worse*.  Localisation discharges the precondition rather than asserting it.

### F3: the ceiling was bypassed for exactly the sharpest rows

`viable = c_lo < c_dn` compares against the dense cost, so the sharper the row the more
certainly peak-local kept it — and a row past `UPSAMPLE_FACTOR_MAX` is the sharpest kind
there is.  At a derived factor of 8192 the dense path RAISED, as designed, while
peak-local returned −24451 nats and reported `tail_bound_worst = -2721`.  The ceiling is
now checked before the cost gate.  The old test only passed because it set
`UPSAMPLE_FACTOR_MAX = 2`, broad enough that the *cost* gate declined the row first; the
ceiling was never what routed it.

## G-class: the defect was a CLASS, and fixing one site was not enough

A third review, after the localisation fix, found the same quantisation error at three
more consumers of the enumeration index.  The audit that finds them is one grep:
`cols_p` / `q_up[`.  Every quantity computed from `q_up[rows_p, cols_p]` inherits it;
the index is legitimate only as a Newton SEED and bracket centre, never as a value.

**G1 (critical).** `PEAK_KEEP_NATS` was applied to the SAMPLE value, before
localisation.  A crest `d` from its sample reads `(d/sigma)^2/2` nats low, so a peak
between samples was compared against a peak on a sample and dropped.  On the two-peak
fixture at rho ~ 700, shipped code, no monkeypatching:

| B offset (h_enum) | peak-local − truth | n_peaks_total |
|---|---|---|
| 0.00 | +0.000000 | 2 |
| 0.25 | **−0.693147** | 1 |
| 0.50 | **−0.693147** | 1 |

`−log 2` exactly: one of two equal peaks deleted.  The secondary crest was 1.003 nats
below the dominant crest while its SAMPLE was 70.99 nats below.  Fixed by comparing
crests: the vertex height of the parabola through the three stencil points is the crest
value exactly for a Gaussian peak, at any spacing — the same property the width
estimator uses, applied to the zeroth moment instead of the second.  It costs nothing.
After: 2 peaks kept and +0.000000 at every offset.

**G2.** The claim that a dropped peak is "safe because it enters the tail bound" was
vacuous.  Since `q_out_max` is at least the dropped peak's own sample, the bound accepts
the row unless `sigma_t < 2.6e-18 s`, while `UPSAMPLE_FACTOR_MAX` bounds the sharpest
legal row at `3.0e-08 s` — ten orders of magnitude away.  For EVERY legal row a
keep-filter drop is automatically accepted.  The claim is removed; the filter now rests
on a direct magnitude argument tying `PEAK_KEEP_NATS` to `UPSAMPLE_FACTOR_MAX`, and that
inequality is asserted.

**G3.** The containment check's scope is narrower than it looked: `row_star` is a per-row
maximum over KEPT peaks, so it verifies the dominant crest only and a dropped peak can
never enter it.  Documented, not widened.

**G4.** The "conservative superset" was not conservative: `localise_peaks` brackets at
`+/- h_enum` and accepts anything strictly inside, so `|t* - t_grid|` is bounded by
`h_enum`, not half of it — and the bound is approached (0.959 `h_enum` observed over
14,182 peaks).  Widened to `+ h_enum`.  Narrowing the bracket instead would be wrong: an
asymmetric peak's crest genuinely can sit more than half a cell from its sample.

**G5.** The curvature stencil CENTRE was clipped inward by `maxd = 8`, so a peak within 8
enumeration samples of an end had its width measured elsewhere, returning `sigma = inf`
at index 0 and dropping the peak before localisation could help.  Now the centre stays
put and out-of-range half-widths are masked.

**G6 — why the suite missed G1.** Every multi-peak fixture ran at `amp = 200`
(`sigma_t/deltaT = 0.0072`, just above the 0.0057 threshold); every sharp fixture was
single-peak; and the updated F1 tests moved BOTH peaks off-grid by similar amounts, so
the deficits cancelled.  **The defect needs asymmetry** — one peak near a sample, one
between — and the suite never crossed "more than one peak" with "sharp".  That cell now
exists.

## What this changes

The dense band-limited rule refines the WHOLE window to a peak whose width shrinks as
`1/rho`, so its cost grows exactly where the peak occupies least of the domain.  This
splits the two resolution requirements the dense rule conflates:

* enumerating the extrema of `kappa` — a small, **SNR-independent** factor, because
  `kappa` is band-limited at Nyquist by construction;
* integrating `exp(lnL)` — the rho-dependent part, needed only within a few `sigma_t`
  of each enumerated peak.

Intervals of half-width `W_SIGMA * sigma_i` are built around every enumerated maximum,
**merged into disjoint intervals**, and each is integrated at its own derived spacing.

## THE COST CLAIM: what the prototype's number was, and what this one is

`DESIGN_time_marginalization_quadrature.md` tabulates the prototype
(`~/tmarg_harness/peaklocal2.py`) at a flat ~97 evaluation points against the dense
grid's 19,648 → 628,736, i.e. **"200x at rho ~ 15, 6,482x at rho ~ 692"**.

**That measurement had the analytic `kappa` in hand**, so evaluating `kappa(t)` at an
arbitrary time was one closed-form call.  In the shipped code only the coarse samples
exist and the band-limited interpolant must be evaluated at the local grid points,
which costs `O(npts)` per point.  A point count is therefore not a cost, and **the
prototype's speedups are not this module's speedups.**  Nothing below is inherited
from it.

## Cost, measured END-TO-END through the shipped likelihood

`DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`, `n_extrinsic = 4000`, 3 IFOs,
srate 4096, `npts = 614`, CPU `process_time`, same call and same inputs for all three
rules (`~/pl_work/cost_pl.py`, an extension of `~/tmarg_harness/cost_e2e.py`):

| `sigma_t/deltaT*` | simpson | bandlimited | **peak-local** | bandlimited / peak-local | rows peak-local handled |
|---|---|---|---|---|---|
| 1.735 | 0.519 s | 0.559 s | 0.549 s | 1.02x | 0 / 20 |
| 0.549 | 0.536 s | 1.456 s | 1.535 s | 0.95x | 0 / 2345 |
| 0.174 | 0.569 s | 5.583 s | 7.465 s | 0.75x | 0 / 3486 |
| 0.055 | 0.550 s | 19.493 s | 9.328 s | **2.09x** | 1866 / 3834 |
| 0.017 | 0.569 s | 46.011 s | 6.005 s | **7.66x** | 3335 / 3950 |

All five rows are on `ldas-pcdev13`, re-measured AFTER the localisation fix.  An earlier
table in this file was taken on a different, faster host (Simpson baseline 0.18–0.30 s
rather than 0.55 s) and has been replaced rather than merged: seconds are not comparable
across hosts and mixing them would invent a trend.  Read the RATIOS.

What the fix cost: at the sharp end essentially nothing (7.78x → 7.66x, 2.03x → 2.09x,
measured before/after on their respective hosts), and about 19% at `sigma_t/deltaT =
0.17` (0.93x → 0.75x), which is a regime where this rule has nothing to offer anyway and
delegates every row.

Read against Simpson instead, the same table says peak-local costs 0.9x / 2.5x / 9.6x /
12.8x / 8.8x the historical rule, where the dense rule costs 1.0x / 2.4x / 8.8x /
26.0x / 68.5x — i.e. **peak-local's cost stops growing with rho and the dense rule's
does not**, which is the structural claim, and is visible in the last two rows.

A first attempt at the fix ran the localiser BEFORE the gate that discards the row,
which is the same error as the one recorded below and cost far more: **0.14x** at
`sigma_t/deltaT = 0.17`, i.e. 7x slower than the path it delegates to, on a block where
every single row fell back.  The gate now runs first, on intervals built about the
sample and widened by `h_enum/2` — a CONSERVATIVE SUPERSET of whatever localisation will
produce, so a gate on it can only decline rows, never wrongly keep one.

`sigma_t/deltaT*` is the sharpest row in the block.  The Simpson baseline is
rho-independent by construction; a run where it moves with rho is contaminated.
Host-sensitive: the O4c effort measured the same quantity moving up to 2x between
hosts, so read the RATIOS, not the seconds.

Two honest readings of that table:

* The win grows with rho and is real at the sharp end, but it is **an order of
  magnitude, not three**.  The gap between this and the prototype's figure is
  entirely the cost of evaluating the interpolant, which the prototype did not pay.
* At low rho this method LOSES to the dense rule, which is why the per-row cost gates
  exist.  They are cost decisions only — both branches satisfy the same derived
  resolution criterion, so no accuracy is traded.

### The gates, and a mistake worth recording

A first version applied the cost comparison only AFTER enumerating.  Rows that then
fell back had paid for an enumeration FFT they did not use, and the method came out
**slower than the dense path it delegated to** (0.43x and 0.74x at
`sigma_t/deltaT` = 0.55 and 0.17) even though every single row fell back.  Two changes
fixed it, and the worst case is now 0.93x:

1. before any work: `MIN_LOCAL_POINTS = 2*W_SIGMA*UPSAMPLE_SAFETY + 1 = 49` is the
   fewest local points any row can need, since one interval at spacing
   `sigma/UPSAMPLE_SAFETY` is `2*W_SIGMA*UPSAMPLE_SAFETY` sub-intervals whatever
   `sigma` is, and merging only adds points;
2. the merge itself was vectorised across rows.  A broad integrand has many enumerated
   peaks, and a Python loop over them cost more than the enumeration did.  The running
   maximum is restarted at each row boundary by offsetting row `r` by `r * big`, so one
   global `maximum.accumulate` gives the per-row running maximum with no segmentation.

The residual 0.93x at `sigma_t/deltaT = 0.17` is the enumeration FFT for rows that then
fall back — the price of a gate that cannot be decided without looking.  It is a 7%
tax on a regime where this rule has nothing to offer, and it is reported here rather
than tuned away.

## Accuracy

Against the **analytic** truth (`kappa` a sum of exponentials below Nyquist, so the
continuous integral is closed-form), srate 4096, npts 614, peak at sample 307.25:

| `sigma_t/deltaT` | rho ~ | bandlimited − truth | **peak-local − truth** | rule used | local points | intervals |
|---|---|---|---|---|---|---|
| 0.743 | 2 | +2.3e-09 | +2.3e-09 | dense (cost) | — | — |
| 0.470 | 3 | +7.5e-10 | +7.5e-10 | dense (cost) | — | — |
| 0.255 | 6 | 0 | 0 | dense (cost) | — | — |
| 0.149 | 11 | +5.7e-14 | +5.7e-14 | dense (cost) | — | — |
| 0.105 | 15 | +5.7e-14 | +5.7e-14 | peak-local | 64 | 1 |
| 0.047 | 35 | +6.0e-13 | +8.1e-13 | peak-local | 64 | 1 |
| 0.017 | 98 | +6.4e-12 | +7.3e-12 | peak-local | 64 | 1 |
| 0.0074 | 219 | 0 | −3.6e-12 | peak-local | 64 | 1 |
| 0.0023 | 692 | +1.5e-10 | +1.5e-10 | peak-local | 64 | 1 |

Errors at the 1e-14…1e-10 level are the REFERENCE's own resolution (a 2048x uniform
refinement of the closed-form `kappa`), not the method's; the two rules agree with each
other far more closely than either is being measured to here.  The flat **64 local
points** across four decades of `rho` is the structural point: the local point count is
set by `W_SIGMA` and `UPSAMPLE_SAFETY`, not by the peak width, so it does not grow with
SNR while the dense grid's does (`npts * factor` = 19,648 → 628,736 over the same rows).
What that flat point count is NOT is a flat cost — see above.

And against the dense band-limited path row by row over the 4000-row blocks above:
max |peak-local − band-limited| = **1.9e-11 nats**, median 6.8e-13.

The local evaluator reconstructs the same interpolant the dense zero-padded FFT does,
to 2e-14 … 4e-13 relative, at every production `npts` — 153, 307, 613, 614, 1228, 2457
— and at 8, 9, 3.  Odd `npts` is the common case (three of the five production sample
rates), and the failure it invites is exact AT the samples and wrong between them, so
it is parametrised rather than spot-checked.

### Memory chunking moves the last bits, and the docstring that said otherwise was wrong

The extrinsic axis is chunked so one dense temporary stays inside a working-set budget.
The per-row plan — which rule the row gets, how many intervals, and the point count,
which is bucketed to a power of two — depends only on that row, so chunking cannot
change any of it.  It DOES change the leading dimension of the FFT and of the reduction
inside `eval_bandlimited_uniform`, and both numpy's FFT and its pairwise summation
reassociate with batch shape.

Measured, one row per chunk versus all rows at once, on a three-row block spanning
rho ~ 100 to 700: **0, 0 and 2 ULPs** (2.4e-16 relative).  The first version of this
module carried the inherited claim that chunking "cannot change the answer"; the test
that asserted bit-identity failed, which is how the overclaim was found.  The test now
pins a few ULPs — still sharp enough that a dropped or mis-ordered chunk, which moves a
row by nats, cannot hide behind it.

## Why the truncation is rigorous and not hopeful

RIFT PR #201 was caught seeding a Newton solve at guessed points, missing genuine
maxima and returning `-inf` for a finite integral.  There is no seeded root-finder
here.  Two separate mechanisms:

* **Enumeration.**  Every local maximum of the band-limited interpolant is found on a
  grid that resolves `kappa` itself.  Checked against a factor-64 enumeration: every
  peak carrying representable mass at factor 64 is also found at the shipped factor 8,
  to within one coarse sample.
* **A computed bound.**  `log(T_outside) + max_{outside} lnL` upper-bounds the omitted
  integral and is compared per row against the value computed; a row that cannot meet
  `TAIL_LOG_TOL` goes to the dense path.  **This does not depend on the enumeration
  being complete** — a missed peak lands outside the intervals, so the sampled maximum
  outside sits on it and the bound fails.  Enumeration buys speed; the bound buys
  correctness.

That is asserted, not asserted-about: `test_a_sabotaged_enumeration_is_caught_by_the_tail_bound`
monkeypatches the enumeration down to a single maximum on a two-peak integrand — which
discards half the mass, `log 2 = 0.69` nats — and requires the module to detect the
shortfall and return the dense value.

Evaluating the outside maximum is free because every shipped `loglikelihood` callback
is monotone increasing in `Re kappa` (plain, distance-marginalized) or `|kappa|`
(phase-marginalized), and `rho_sq` is time-independent on this path, so
`argmax_t lnL = argmax_t term(kappa)` whatever the distance, the distance prior or the
callback.  Verified over 0.05 … 40 in `1/D` and across three callback shapes: identical
peak sets.  This is what keeps the callback — a table interpolation in production —
off the full time axis entirely.

## Merging is correctness, not tidiness

Two overlapping windows integrated separately both contain the shared region, so the
log-sum-exp of the parts counts it twice.  Prototype (`~/tmarg_harness/peaklocal.py`):
**+1.6 nats at rho ~ 6**.  `test_unmerged_intervals_double_count` rebuilds the
un-merged variant against this module's own evaluator and requires it to be wrong,
because on a sharply peaked row the intervals do not overlap at all and every other
accuracy test stays green when the merge is deleted.

Merging is also what makes this ONE algorithm rather than a regime switch: isolated
peaks give a tiny union, crowded peaks grow the union to the whole window and the
method degenerates continuously into the dense grid.  No threshold anywhere.

## Mutation sweep

25 mutations against the post-G-fix code (`244e7cca`), baseline **90 passed / 4
deselected** (the 4 driver subprocess tests, which no numerical mutation can reach;
full suite 94).  Restores from `git show HEAD:` — a pristine source, never a
reverse-edit, never a snapshot taken while a mutation was live — with every anchor
required to match exactly once so a stale anchor reports a HARNESS FAILURE rather than a
false survivor.  Run on `ldas-pcdev13` with the intended branch verified live.

**17 killed, 8 survived.**  Killed, with the test that did it:

| mutation | killed by |
|---|---|
| L1 skip localisation | uniform-arrival block, return_peaks, ceiling |
| L2 one Newton step | 12 tests |
| L3 drop the convergence assertion | localiser-reports-non-convergence |
| L5 drop the tol widening | tail-bound recomputation |
| G1 keep filter on the SAMPLE value | secondary-crest-between-samples |
| C1 containment always passes | containment-catches-mis-placed-interval |
| **C2 containment vs the enumeration SAMPLE** | containment-catches-mis-placed-interval |
| C3 `T_outside` from grid indices | tail-bound recomputation |
| C4 tail bound drops `log(T_outside)` | tail-bound recomputation |
| C6 ceiling after the cost gate | ceiling-fails-closed-for-sharpest |
| C7 disable the pre-enumeration gate | pre-enumeration-gate-actually-fires |
| C9 `LOCALISE_SAFETY` breaks the relation | tuned-constants |
| E2/E3 plateau asymmetry, both ways | plateau-yields-exactly-one-maximum |
| E4 merge running maximum | merge-keeps-contained-interval |
| E7 drop trapezoid half-weights | local-trapezoid-half-weights |
| **W1 shipped branch delegates to bandlimited** | option-reaches-the-shipped-likelihood |

C2 and W1 are the two that had to die.  W1 makes the option INERT — same signature, same
numbers, entire cost benefit gone — and it was invisible until `last_report()` was
asserted, because peak-local is *designed* to agree with bandlimited so no value
comparison can distinguish them.  C2 is the check comparing against the sample instead of
the localised crest, which passes precisely in the case it must catch.

### The 8 survivors, and which of them are evidence

A mutation that does not change behaviour is a harness artifact, not a coverage gap.
Each survivor was re-applied and run over a battery of six fixture families chosen to hit
the shape it targets, comparing values AND report counters against pristine:

| survivor | changes behaviour? | verdict |
|---|---|---|
| L4 widen the Newton bracket | **no** — identical on all 6 | no-op: Newton converges well inside `+/-h_enum`, so widening is unobservable |
| G4 gate interval not a superset | **no** — identical on all 6 | no-op here; cost-only by construction (a row it wrongly keeps is still computed correctly) |
| E1 re-exclude endpoints | **no** — identical on all 6 | no-op: no fixture puts a discrete maximum exactly at index 0 or last |
| E5 drop interval clipping | **no** — identical on all 6 | no-op: no fixture produces `lo < 0` or `hi > t_last` |
| E6 curvature ladder → d=1 | **no** — identical on all 6 | no-op: no fixture has a `-inf` hole at d=1 on the enumeration grid |
| G5 re-clip the stencil centre | values, at **1e-12** | effectively a no-op; see below |
| C5 one extra covered sample per end | counters only | genuine gap, diagnostics only |
| C8 disable the final `MAX_INTERVALS` check | counters only | genuine gap, precisely diagnosed below |

So **five of the eight are not evidence at all**, and reporting them as coverage gaps
would be the sweep lying to itself.  What remains:

* **G5** changes the answer only at 1e-12 on the fixtures available, because on every
  one of them the near-edge peak is more than `PEAK_KEEP_NATS` below the dominant crest
  and is dropped before the stencil ever runs (`n_peaks_total == 1`).  Making a peak both
  near-edge and within 60 nats puts the row in a regime where it is declined on cost
  instead, so **the shape is not reachable through the public entry point with these
  fixtures.**  The fix is applied and is strictly better; the test that claimed to cover
  it did not, and has been renamed to say what it actually checks.  The reviewer measured
  the defect directly (`e2_edgepeak`, −0.3124 nats).  **Open gap.**
* **C8** is killed by the PROVISIONAL structure gate, not the final one — disabling only
  the final `too_much` check leaves the suite green because the provisional gate declines
  the row first.  The final check is kept (the final interval count *can* exceed the
  provisional one, since narrower intervals merge less readily) but no fixture reaches
  it.  **Open gap, precisely located.**
* **C5** perturbs `covered` near an interval edge; with `T_outside` now exact geometry
  this moves only `q_out_max`, and only when the outside maximum sits adjacent to an
  edge, which no fixture arranges.  **Open gap, diagnostics only.**

## Not done in this draft

* **The evaluator is a direct spectral sum**, `O(npts)` per output point.  A chirp-z
  (Bluestein) evaluation would make it `O((npts + M) log(npts + M))` and is the single
  largest remaining cost lever.  Not attempted here.
* **No GPU measurement.**  The path is `xpy`-generic and there is a cupy parity test,
  but the cost table above is CPU only.
* **No real-data run.**  Accuracy is against analytic truth and against the dense path;
  the dense path's own real-injection comparison has not been repeated for this rule.
* **`MAX_INTERVALS`, `PEAK_KEEP_NATS`** are fail-closed guards with an argument behind
  them but no sweep behind the specific values.
* **The tail bound is still a sampled maximum, not a supremum.**  `q_out_max` is the
  largest `Re kappa` over enumeration-grid points outside the intervals; between those
  points it is not bounded rigorously.  `T_outside` is now exact and endpoints are now
  enumerated, and the containment check covers the failure mode that mattered, but a
  Bernstein-type bound on the interpolant between samples would make this a proof rather
  than a strong check.  Not attempted.
* **No re-measure-and-double loop.**  The dense path ENFORCES its resolution criterion;
  this path derives the spacing and then verifies the outcome two other ways.  Both are
  checked; they are not the same criterion, and this file no longer claims they are.
* **Phase marginalization is REFUSED**, at the library and at driver startup — a
  deliberate scope cut, not an omission.  Production marginalizes over distance, not
  phase, and under phase marginalization the time peak's Laplace width picks up an
  `(I1/I0)(|kappa|/D)` factor that does not reduce, so the local spacing would stop
  being derivable from `rho_sq` and the curvature alone.  `bandlimited` still supports
  it and is unchanged.
* **`resample_samples()` is not served.**  The extrinsic time-export path needs a full
  `lnL(t)` array on the original grid, which this rule by construction does not
  produce.  `return_lnLt=True` therefore still runs the coarse path, exactly as it does
  under `bandlimited`.  `t_star` and the local widths ARE exposed
  (`return_peaks=True`), which is the piece a future export or a time-first
  reordering of the marginalizations would build on.
