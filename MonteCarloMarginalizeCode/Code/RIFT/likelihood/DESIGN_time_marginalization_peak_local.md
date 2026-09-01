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

## Round 4: the class was still open, at two more doors

### Door 1 — the parabolic vertex obeys the SAME `(delta/sigma)^2` law

`_parabolic_vertex_height` was introduced to fix G1 and its docstring claimed the vertex
is the crest "EXACTLY for a Gaussian peak, at any spacing".  That is true only if `lnL`
is quadratic across the whole `+/-h_enum` stencil.  It is band-limited `q(t)`; the
quartic term survives.  Under-read of the crest at half-cell phase, in nats:

| `sigma_t/deltaT` | factor | sample | **vertex** |
|---|---|---|---|
| 0.004934 | 512 | −108.75 | −1.17 |
| 0.002467 | 1024 | −434.98 | −4.66 |
| 0.001233 | 2048 | −1739.93 | −18.66 |
| 0.000617 | 4096 | −6959.72 | **−74.63** |

`UPSAMPLE_FACTOR_MAX = 4096` permits down to `sigma_t/deltaT = 0.000488`, so the last row
is a LEGAL configuration, and end-to-end against the exact interpolant at 16384x it
reproduced **−0.693147** — `−log 2`, one of two equal peaks deleted — with both defences
silent.  Roughly an 8x improvement in reach over comparing samples, not a fix.

**The lesson, and the reason this was a third round on one class: every approximation
substituted for the crest fails the same way one octave further out.**  So the keep
decision is now taken in two stages, and neither treats an estimate as the answer:

1. a CONSERVATIVE PRE-FILTER, whose only job is to bound how many peaks reach
   localisation.  It compares an UPPER bound on each crest — the sample plus the
   worst-case correction `(h_enum/2)^2/(2 sigma^2)` — against a LOWER bound on the
   highest crest — the largest sample, which cannot exceed its own crest.  It can only
   ever keep too many;
2. after localisation, the EXACT filter on `lnL_star`, which is the crest by
   construction rather than to second order.

Verified on the reviewer's own probe at every legal factor: **+0.000000 with
`n_peaks_total = 2` at factor 1024, 2048 and 4096, at offsets 0, 0.25 and 0.5.**

### Door 2 — both estimators degraded to the raw sample at the array ends

At `cols_p == 0` the whole left half of the curvature stencil is out of range at EVERY
half-width, so `d2` was NaN throughout and `sigma = inf`; the vertex height fell back to
`y0`, i.e. pre-G1 behaviour.  Revision 2's "mask instead of clip" change looked more
principled than the `maxd`-clipping it replaced and was worse exactly at the two indices
its own justification named.

Fixed by shifting the stencil centre inward by the MINIMUM needed for a three-point
stencil to exist — one sample — which is a genuine one-sided fit at an endpoint, not a
compromise.  The localiser's bracket is also clamped to `[0, t_last]` and a peak pinned
at a window boundary counts as converged.

**Partially closed, and measured as such.**  Against the reviewer's `f2_edge_sigma`,
comparing `761cafb3` with the fix, same fixtures:

| case | before | after |
|---|---|---|
| `sigma@edge = inf`, dH 2.0 / 1.0 / 0.5 / 0.2 | −0.126 / −0.312 / −0.473 / −0.598 | **+0.000 / −0.000 / +0.000 / +0.000** |
| `sigma@edge` finite, `sig2/sig1 = 0.58` | −6.001 | −6.001 (unchanged) |
| `sigma@edge` finite, `sig2/sig1 = 0.41` | −0.335 | −0.335 (unchanged) |

So the family Door 2 describes — no finite width obtainable at an endpoint — is closed
with no regression anywhere.  **A residual family remains and is NOT that mechanism**:
those rows already had a finite edge sigma at `761cafb3`, up to **−6.0 nats**, accepted.

**Re-measured after Round 5's rebase, this family changed character but did NOT close.**
Against a converged reflected reference (32768x; successive refinements differ by
−1.8e-4), on the sharpest case (`sig2/sig1 = 0.58`, secondary at `0.4 h_enum` from t=0):

| | value | vs reference |
|---|---|---|
| converged reference | 39844.73839 | — |
| `bandlimited`, derived factor 256 | 39845.34836 | **+0.610** |
| `peak-local` | 39845.87353 | **+1.135** |

So the disagreement with the rule it delegates to is **+0.53 nats, ACCEPTED** —
`tail_bound_worst = −250.71` and the containment check passes, so neither defence fires.
It was −6.0 nats and is now +0.53, a 10x improvement and a sign flip, but it is still an
accepted error three orders above the 1e-3 nat bar the rest of this rule meets.

**Located, not yet fixed.** The even reflection is value-continuous but its DERIVATIVE
flips sign at the join, so a peak sitting at t=0 is a two-half-peak cusp rather than a
locally Gaussian peak.  Both rules derive a width — and hence a factor and an interval
half-width — from a Gaussian curvature model, so both under-resolve it; `bandlimited` is
wrong here too (+0.610 at its own derived factor), which places the root cause in the
shared reconstruction rather than in the local placement.  peak-local roughly doubles the
error because it also sizes its interval as `W_SIGMA * sigma` from that same model.
Diagnostics: the row enumerates 295 maxima, the keep filter retains 1, and that one is at
enumeration index 0 with `q` decreasing monotonically away from it.

**The quantisation class should not be called closed on my say-so, and this is not the
quantisation class**: the enumeration-index defect is closed and re-verified above, while
this is a curvature-model defect at the reflection join, inherited from the dense path.
It is the top open item.

### The `W_SIGMA` coupling is now asserted

The sampled `q_out_max` survived a determined attempt to break it (24 accepted rows,
honest supremum on a 4096x grid, worst honest margin −63.42 against `TAIL_LOG_TOL =
-23`).  But the reason is structural slack, not adequate sampling: the outside supremum
sits at an interval edge, already `W_SIGMA**2/2 = 72` nats below the crest.  Dropping
`W_SIGMA` below ~8–9 would silently invalidate the bound.  The inequality

    W_SIGMA**2 / 2  >  |TAIL_LOG_TOL| + log(T_out / (sqrt(2 pi) sigma_min))

is now asserted, tying `W_SIGMA` to `TAIL_LOG_TOL` and `UPSAMPLE_FACTOR_MAX` so none can
move alone.

## Round 5: the rebase onto `rift_O4d` changed the reconstruction underneath this module

Retargeting this PR from the merged `rift_O4d_tmarg_bandlimited` onto `rift_O4d` moves it
across `e4ed25c7` ("Avoid Gibbs ringing in time marginalization"), which

* replaced the raw periodic zero-padded FFT with an EVEN REFLECTION — periodize
  `[kappa forward, kappa backward]`, keep the forward interval — because a zero-padded FFT
  of the gathered slice alone identifies its unlike endpoints and rings globally
  (+140.9 nats measured on an adversarial row); and
* demoted `EDGE_GUARD_FRACTION` to a diagnostic, on the grounds that crossing an arbitrary
  threshold must not silently move an under-resolved row back to Simpson.

**The diff did not change; its meaning did.** This module was written against the older
contract, and nothing in the seven-file delta says so. Three clauses had drifted, and each
one made peak-local return a WORSE value than `time_marginalize_bandlimited` for the same
row — the one thing this rule promises never to do.

| drift | measured |
|---|---|
| enumeration, localisation and local evaluation still on the PERIODIC interpolant, while fallback rows got the reflected one — two different continuous functions inside one call | **−3.79 nats** on a row with peaks at both window ends; **9.0e-6 nat** median bias on the uniform-arrival block |
| `EDGE_GUARD_FRACTION` still routing a near-edge row to SIMPSON | row never entered the rule; every fallback counter read zero |
| `boundary_unresolved` missing: an endpoint maximum whose inward-clipped stencil reads positive curvature is mislabelled "flat" and silently keeps Simpson | **+4.60 nats** against the reflected reference, where the dense path was +0.81 |

Fixed by taking the spectrum of `concatenate((kappa, flip(kappa)))` at period
`2*npts*deltaT`. The local evaluator then reproduces `reflected_bandlimited_upsample` to
**2.3e-13 relative** at every production `npts` (153 / 307 / 614 / 1228 / 2457, odd and
even).

### The lesson is structural, and the fix is too

Fixing the copy leaves the copy. Row classification is now `_classify_rows` in
`time_marginalization_quadrature.py`, the SINGLE definition, called by both rules;
peak-local keeps only `refined = has_peak & (factors > 1)`. Verified behaviour-preserving
rather than asserted: `time_marginalize_bandlimited` is **bit-identical** over a 32-row
battery spanning flat, edge, near-edge, broad, sharp and both-ends-peaked rows, with every
`last_report()` counter identical.

`test_row_classification_matches_the_dense_path_exactly` compares the CLASSIFICATION, not
the values, because a value check passes whenever the two rules happen to agree — which on
most rows they are designed to, which is exactly why this drifted unnoticed three times.

### Door 1 re-verified at the sharpest legal row, against the right reference

The round-3 reopen (`g1e_reopen2`) reproduced at derived factor 4096, off = half a cell:
**−0.693147** = −log 2, one of two equal crests deleted. Re-run against a REFLECTED 16384x
reference — the object the module now integrates — all nine rows are exact and both peaks
survive:

| sigma_t/deltaT | factor | offset | peak_local − truth | n_peaks |
|---|---|---|---|---|
| 0.000615 | 4096 | 0.00 | +0.000000 | 2 |
| 0.000615 | 4096 | 0.25 | +0.000000 | 2 |
| 0.000615 | 4096 | 0.50 | **+0.000000** | 2 |

### Door 2's mechanism is closed, and measured at the source

The estimator is now genuinely one-sided at the array ends rather than degrading to the raw
sample. On a parabola of known width `3 h_enum` sampled on the enumeration grid, the
recovered sigma at peak index 0, 1, 7, 8, 100, n−9, n−2 and n−1 is **finite and exact
(3.0000 h) at every one**, including both endpoints. The dead-code condition — 22 endpoint
maxima enumerated, zero able to obtain a finite width — is gone.

End to end, a row whose dominant maximum sits at the very edge is either handled correctly
or delegated, never silently approximated: at tau = 0.5 and 612.5 samples peak-local is
exact; at 0.2 and 612.8 it is **better** than the dense path (+1.44 against +1.71); at 0.0
and 613.0 it declines and returns the dense value **bit-identically**.

### The contract, checked reference-free

`|peak_local − bandlimited|` over 46 rows spanning sharp phase-scan, broad, near-edge,
two-peak, both-ends and flat/null families: **worst 2.6e-10 nats**, no family above 1e-3.
All ten near-edge rows are now handled by the rule; before this round they were routed to
Simpson.

## Round 6 — DOOR 4: the pre-filter's "upper bound" was not one

The fourth independent re-attack reopened the class at the one site the previous three had
left alone, and it is the site whose own comment said it could not fail: the
**conservative pre-filter**, described as comparing "an UPPER bound on each crest against a
LOWER bound on the highest crest, so it can only ever keep too many."

    crest_upper = lnL_sample + (h_enum/2)**2 / (2 sigma**2)

is not an upper bound, for two independent reasons, and the second is the one that matters:

* the localiser's bracket is `+/- h_enum` and displacements of `0.959*h_enum` have been
  observed, so the correction covers less than half the distance it must; and
* **`lnL` is not a parabola across a half enumeration cell.** The ANHARMONIC part of the
  crest deficit carries the same `1/sigma**2` amplification as the quadratic part. At
  derived factor 1024 the pure quantisation excess is **4.4 nats** and the true shortfall
  is **122.30**.

Being short, it DELETED peaks — before localisation, so the exact filter never saw them.

| derived factor | shortfall of `crest_upper` | `pl - bl` | accepted? |
|---|---|---|---|
| 1024 | +122.30 nats | **−0.358385** | yes, both defences silent |
| 2048 | +489.19 | **−0.358427** | yes |
| 4096 | +1956.74 | **−0.358383** | yes |

and the magnitude is not bounded by `log 2`. Raising the deleted peak above the survivor
by `Delta`:

| Delta | `pl - ref` | accepted |
|---|---|---|
| 0 | −0.358 | yes |
| 800 | −799.159 | yes |
| **1850** | **−1849.159** | **yes** |
| 1950 | +0.000 | yes (the peak survives) |

The cutoff is `shortfall - PEAK_KEEP_NATS`, so **a peak may sit ~1900 nats ABOVE the one
that survives and still be deleted.** The tail bound cannot backstop it: `q_out_max` reads
the deleted peak at its SAMPLE, the very quantity the defect corrupts. Recomputed as an
honest supremum on a 4096x refinement, those rows score **+11.5 / +12.1 / +7.2** against
`TAIL_LOG_TOL = -23` — every one would be REJECTED.

### The fix is an inequality, not a better fit

Widening the constant would have been the fifth version of the same mistake. Expanding
about the CREST, where `q'` vanishes by definition,

    q(t_s) = q(t*) + q''(xi) (t_s - t*)^2 / 2,    |t_s - t*| <= h_enum

so `q(t*) <= q(t_s) + max|q''| * h_enum**2 / 2`, and `max|q''| <= sum_j |Xw_j| |w_j|**2`
by the triangle inequality on the spectral sum. **Nothing is fitted**, so there is no model
error left to amplify, and it uses `h_enum` rather than half of it. `loglikelihood` is
monotone in `q`, so a bound on `q` is a bound on `lnL`.

Verified as a PROPERTY rather than on the fixture that motivated it: over **9311 enumerated
peaks** across npts 153/307/614/1228/2457 and amplitudes 2e4–2.5e7, **zero violations**,
where the old bound failed on **0.3–3 %** of peaks by up to **19456 nats**.

`spectral_curvature_bound` and `crest_upper_bound` are module functions precisely so the
suite tests the SHIPPED formula and not a copy of it.

### Deleting the pre-filter outright is not the answer, and was tried

Without it every row enumerates its whole oscillation — 295 maxima on one fixture — the
structure gate sees more than `MAX_INTERVALS` and declines every row, and the option goes
**inert**: `n_peak_local_rows = 0` on all six fixture families, which is the W1 hazard the
suite already guards against. With the rigorous bound, coverage is exactly what it was
(24/24 sharp, 10/10 near-edge, 3/3 two-peak) and `|peak_local - bandlimited|` is
**2.6e-10 nats** over 46 rows.

### The same defect at a third site, fixed with it

The pre-filter's LOWER bound read `lnL_st[:, maxd]` — the callback at the stencil CENTRE,
which is clipped inward by one at the array ends. At enumeration index 0 or `n-1` that is a
full cell from the peak it describes: **132 nats** low at rho ~ 40, **8449** at rho ~ 700,
growing with SNR. A stencil centre is clipped so the CURVATURE can be measured; the peak's
own value must be read where the peak is.

### What makes the regression test worth anything

`test_a_codominant_crest_is_not_deleted_by_the_prefilter` **FAILS on the parent commit** and
passes here. Its fixture solves for the second amplitude by bisection rather than hard-coding
a ratio, because the crest gap is linear in amplitude — a ratio tuned at rho ~ 40 leaves the
second peak thousands of nats down at rho ~ 700 and the fixture silently stops testing
anything, which is exactly how round 2's suite missed round 3's defect.

The helper that finds the two crests LOCALISES them. The first version of it did not, and
was wrong in the same way as everything else in this file: peak A sits on an enumeration
sample so its sample is its crest, while peak B is deliberately off-grid and its sample
understates its crest by ~3125 nats in the sharp regime. Equalising the SAMPLES leaves B
thousands of nats above A, A is correctly dropped, and the test passes while measuring
nothing.

### Still open after round 6

* **The ceiling contract.** `over_ceiling` is taken on the COARSE derived factor, while
  `time_marginalize_bandlimited` raises on a factor remeasured on the REFINED grid. At
  npts=307 and H = 6.5e4 / 7e4 / 7.5e4 peak-local returns an ACCEPTED value while the dense
  path RAISES. The values are exact to 1e-6 against a 32768x reference, so this is a broken
  fail-closed contract rather than an observed wrong number — but it is the hole the
  module's own ceiling comment claims to have closed. **Not fixed.**
* The tail bound is still a SAMPLED maximum, and its safety still comes from the
  `W_SIGMA**2/2 = 72` nat structural slack rather than from the sampling being adequate.

## Round 7 — DOOR 5: a crest pinned at a window end is not a peak

The round-6 bound held. An independent re-attack could not violate it — **0 violations**,
measured slack **15x to 84.5x** — so the class did not reopen where it had been patched. It
reopened one site further along, at the **resolution scale** rather than the crest location.

`localise_peaks` counted a peak pinned against a window end as CONVERGED, reasoning that
the maximum over the integration domain really is the boundary. It is. **But it is not a
STATIONARY point, and everything downstream assumes one.** At an interior crest `q'`
vanishes, so `exp(lnL)` is locally Gaussian and a spacing derived from the curvature
resolves it. At a boundary the maximum is a CORNER: `q'` is non-zero there, so the local
integrand is an **exponential decay of rate `|q'|`**, and a spacing derived from `sigma`
does not resolve that scale at all. The trapezoid's half-weight endpoint then over-counts by

    log( lam * (0.5 + 1/(exp(lam) - 1)) )  ->  log(lam/2),     lam = |q'| * h_loc

MEASURED on a single band-limited bump centred at `t = 0`, against the same interpolant
integrated on **the very interval the module chose** — so this is the rule's own quadrature
error on its own domain, not a reference artifact:

| amplitude | coarse factor | `lam` | over-count | accepted |
|---|---|---|---|---|
| 4e4 | 512 | 7.12 | **+1.27410** | yes |
| 1.6e5 | 1024 | 14.24 | **+1.96381** | yes |
| 6.4e5 | 2048 | 28.49 | **+2.65648** | yes |
| 2.56e6 | 4096 | 56.98 | **+3.34949** | yes |
| 1e7 | 4096 | 112.6 | **+4.03071** | yes |
| 2e7 | 4096 (ceiling) | — | **+4.37722** | yes |

`log(112.6/2) = 4.031` against a measured `+4.03071` — the closed form is exact. **The error
grows by +log 2 per factor 4 in amplitude and is unbounded inside the legal range.**

### Neither defence could ever have caught it

This is structural, not bad luck, and it is the reason the fix is a refusal rather than
another check:

* **Containment** compares the local grid's attained maximum against `row_star`. The grid's
  FIRST POINT is the pinned crest, so `attained == row_star` identically. It cannot fire on
  this family, at any parameters.
* **The tail bound** is a statement about mass OUTSIDE the intervals. This error is entirely
  inside. `tail_bound_worst` reads −250 to −32990 on exactly these rows.
* `localise_peaks` declared the peak converged, so `n_dense_fallback_localise` was 0.
* `_classify_rows` flags these rows `exposed` and, by design since `e4ed25c7`, `exposed`
  selects nothing.

### The fix, and what it does not fix

A pinned peak is reported UNCONVERGED and its row goes to the dense path. Fail-closed, and
it restores the contract this rule actually makes — **never a worse value than the
backstop**. Coverage is unchanged (24/24 sharp, 10/10 near-edge, 3/3 two-peak; worst
`|peak_local - bandlimited|` **2.6e-10** over 46 rows), because only rows whose crest is
literally at the boundary now decline.

It also closed a **regression round 6 had introduced and the author had not found**: the
looser pre-filter kept a second peak on a near-edge row that round 5 correctly REJECTED, so
the row became accepted carrying this defect — `pl - bl` went `0.000000` (rejected) to
`-0.778178` (accepted). It is back to `+0.000000`, declined. The edge/cusp family that this
note carried at −6.0 nats, then +0.53 and growing with SNR, is now `pl - bl = 0.000000`
throughout: every row either matches the dense path exactly or declines.

**DISCLOSED, NOT FIXED.** `time_marginalize_bandlimited` carries the same defect in milder
form — its refined grid also begins at the boundary with a half weight. Measured against a
converged spectral reference: **+0.768 / +1.429 / +2.121 / +2.813 / +3.479 nats** at
amplitude 4e4 / 1.6e5 / 6.4e5 / 2.56e6 / 1e7, i.e. derived factor 256 → 4096. It obeys the
same `+log 2 per factor 4` law, and finer references move the top row DOWN, so **+3.48 nats
at the legal ceiling is a lower bound.** (An earlier measurement here stopped one octave
short and quoted a maximum of +2.74; that was wrong and is corrected.)

peak-local now matches it **exactly** — `pl - bl = +0.000000` at every amplitude with
`n_peak_local_rows = 0` — so it inherits the defect rather than adding to it, which is what
makes the residual cleanly attributable: it belongs to the shared reconstruction and to
#203's rule, not to this one. **It should be raised upstream, not patched here.**

### A correction to a stated precondition

The module claimed every shipped `loglikelihood` callback is monotone increasing in
`Re kappa`. **That is false**: the distance-marginalized callback returns `-inf` ABOVE its
table as well as below. The direction is safe — an upper bound landing in the hole evaluates
to `-inf`, the peak is dropped, and the row loses peaks until it falls back — so it costs
coverage and never accuracy, and with the shipped table the boundary sits at
`D_eff < d_min/10`, unreachable at default settings. But the claim as written was wrong, and
the pre-filter's `q`-bound-to-`lnL`-bound step rests on it, so it is now stated precisely.

## Round 8 — the first independent YES, and what is left

A sixth reviewer, handed the round-6 and round-7 fixes and told to break them, returned
**YES: the quantisation/resolution class is closed.** That is the first affirmative verdict
in six adversarial passes, and what it rests on is worth recording rather than just the word.

### Why `~pinned` is the right predicate, and not a patch over one fixture

The obvious next door is a crest that is NOT pinned but sits close enough to a boundary that
its interval is CLIPPED, leaving a non-stationary edge. That band was scanned directly —
auto-bracketing the pinned-to-interior transition, then bisecting the bump offset to land
`t*/sigma` on 13 targets in (0.25, 11.9), with the module's own local grid captured and
re-integrated at 512x/1024x on the same interval (convergence ≤ 9.3e-9):

| derived factor | worst `pl − exact(own interval)` | worst `pl − bl` | worst `lam` |
|---|---|---|---|
| 256 | −0.00238 | −0.00193 | 0.172 |
| 1024 | −0.00272 | −0.00224 | 0.80 |
| 4096 | −0.00268 | −0.00222 | 1.24 |

**Flat in amplitude** — 0.0024 to 0.0027 across 250x in amplitude and 16x in resolution —
and negative. The opposite signature to the pinned defect, which grew +1.27 → +4.03 nats at
+log 2 per factor 4.

The mechanism is why, and it is the argument that actually closes the class. At an INTERIOR
crest the edge deficit and the decay rate are TIED: for a peak of width `sigma` integrated at
`h <= sigma/2`, an edge at distance `d` has `lam = d*h/sigma^2 <= d/(2 sigma)` and depth
`Delta = d^2/(2 sigma^2) ~ 2 lam^2`. So `lam >> 1` forces `Delta >> 1` and the over-counted
mass is exponentially suppressed, while `Delta ~ 0` forces `lam ~ 0` and the half weight is
correct. **The pinned case escaped only because a boundary CORNER has `q' != 0` at
`Delta = 0`** — there is no such relation there. Round 7 removes exactly the configurations
that break the tie.

Confirmed at the extreme: an accepted row whose clipped edge is **0.005 nats** below its
interval's own maximum — a 72-nat violation of what `W_SIGMA` assumes — still has
`lam_lo = 0.031` and an error of −6.1e-4 nats. The other end of that same interval has
`lam_hi = 5.67` at depth 72.85. The two conditions never co-occur.

**Merged edges** cannot be the door either, and structurally so: the union's endpoints are
`min_i(t_i − W sigma_i)` and `max_i(t_i + W sigma_i)`, so merging can only push an edge
FURTHER from every crest. Over 250 random rows (1–3 bumps, log-uniform amplitude
10^3.5–10^7.5, 70% of positions biased hard against the window ends) the worst
`pl − bandlimited` was **+0.000000 nats**.

### What is still open, in severity order

* **The ceiling contract, and it is far more reachable than it looked.** `over_ceiling` is
  taken on the COARSE derived factor while `time_marginalize_bandlimited` re-measures on the
  refined grid and raises, so peak-local ACCEPTS where the dense path REFUSES. The crafted
  fixture was not the point: a random hunt hit this on **9 of 250 uncrafted rows** at
  npts=614. The values are exact — scored against a converged spectral reference on three of
  them, `pl − ref = 0.000000` — so it is a **broken fail-closed contract, not a wrong
  number.** Fixing it is a design decision, not a one-line change: `>=` instead of `>` would
  route every row at the legal ceiling to the dense path, which is precisely the regime this
  rule exists to serve. The honest options are to re-measure the width on the local grid the
  way the dense path does, or to state that peak-local is deliberately NOT bound by a ceiling
  that exists for the dense grid. **Not fixed; the top open item.**
* **The dense path's own boundary defect** — merged code, up to **+3.48 nats** at the legal
  ceiling, inherited rather than caused, and peak-local now matches it exactly.
* **A cost regression from round 7.** A row carrying an interior dominant crest AND a
  boundary crest within `PEAK_KEEP_NATS` is now declined whole: `exact_keep` keeps the pinned
  peak and `bad_loc` then condemns the row. Fail-closed and correct, but a real coverage loss
  on edge-peaked rows. Measured: a uniform-arrival block is 60/60 accepted and a
  near-far-end block 6/7, so it bites on edge-peaked rows specifically, not broadly.
* **A latent accounting fragility.** `n_dense_fallback_localise` is incremented
  unconditionally while `keep_row` later ANDs `~bad_loc` with `~too_much`/`~too_slow`, which
  increment their own counters, so a row that is both would be double counted and the
  sub-counts would exceed `n_dense_fallback_rows`. Three early returns also skip the
  `n_dense_fallback_nopeak` accounting. **Not reachable on any fixture tried** — a 47-row
  mixed block reconciles exactly — because the `bad_loc & too_much` path needs peaks
  separated by between `24.5 sigma` and `24.5 sigma + 2 h_enum`, a window the band limit
  closes at high SNR. Recorded as fragility, not as a demonstrated bug. Relatedly `too_slow`
  looks like dead code: the provisional point count is an over-estimate, so `p_slow` fires
  first.

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
| E1 re-exclude endpoints | **no** — identical on all 6 | no-op **over dead code** — see the correction below |
| E5 drop interval clipping | **no** — identical on all 6 | no-op, but the stated reason was wrong — see below |
| E6 curvature ladder → d=1 | **no** — identical on all 6 | no-op: no fixture has a `-inf` hole at d=1 on the enumeration grid |
| G5 re-clip the stencil centre | values, at **1e-12** | right observation, WRONG mechanism — see below |

### Four of those reasons were wrong, and a right verdict on a false premise is how the next bug hides

An independent check reproduced all five no-op VERDICTS on a wider battery. Four of the
reasons I gave for them did not survive:

* **E1.** I wrote "no fixture puts a maximum exactly at index 0 or last". False — their
  battery has 22. The true reason is stronger and much worse: at revision `761cafb3` an
  endpoint maximum could never obtain a finite `sigma` (both estimators degrade at the
  array ends, see Door 2 below), so it was dropped before anything else ran. **Revision
  2's endpoint enumeration was dead code**: 22 endpoint maxima enumerated, zero usable.
  The mutation was a no-op over a feature that did nothing.
* **E5.** I wrote "no fixture produces `lo < 0` or `hi > t_last`". False — the clip fires
  for 138 of 2682 peaks. It remains a no-op only because the clipped region carries
  `e^-72`. But the clip is what makes the integration domain exactly `[0, t_last]`,
  identical to the dense path's, and that invariant was unasserted.
* **G5.** Right that the fixtures do not exercise it, wrong about why: the near-edge peak
  is dropped because `sigma = inf` at index 0, not because it is below
  `PEAK_KEEP_NATS`. Their secondary crest is 1.003 nats down and still dropped.
  "Unexercised" and "the fix is inoperative there" are different bugs, and it was the
  second.
* **C8.** Diagnosis right, and now sharper: 4 of 21 rows have final interval count >
  provisional, so the final check is genuinely non-redundant — it is **untested, not
  unreachable**, and a fixture in the band `prov <= 32 < final` is constructible.
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
