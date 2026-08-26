# Q_lm sub-sample time-interpolation stencil: measurements and decisions

**Status as of 2026-08-16.** Investigation record for `RIFT.likelihood.time_interp_choice` and the
`--interpolate-time` / `--internal-ile-interpolate-time` flags.

This file is a **record of measurements**, not a specification. It is expected to be superseded.
The **live decision** is the single constant `CROSSOVER_GUIDANCE` in
`RIFT/likelihood/time_interp_choice.py`, which every user-facing help string interpolates and
which `test_interpolate_time_cli.py` pins across all entry points. **If this document and that
constant ever disagree, the constant is authoritative and this document is stale.**

Numbers here were measured against PR #97 (merged as `c1a2e2df`) and PR #109.

---

## 1. The decision, in one line

The crossover in total mass **rises with fmin**:

| fmin | crossover | below it | above it |
|---|---|---|---|
| ≤ 50 Hz | 20–35 M☉ | `sinc` | `cubic` |
| 100 Hz | 35–55 M☉ | `sinc` | `cubic` |
| 150 Hz | above 55 M☉ | `sinc` at every mass measured (9–55) | *unmeasured* |

**Measured range is 9–55 M☉ at srate 4096.** There is no high-fmin evidence at 80 or 120 M☉. The
fmin-30 ladder puts those in cubic's regime, but **do not extrapolate that to high fmin**: the
whole finding of §4 is that the crossover rises with fmin, and it moved M=35 and M=55 across it.
Extrapolating a fmin-30 result is the exact error that made #97 wrong. 80 and 120 M☉ at fmin ≥ 100
are simply **unmeasured**. Likewise do not read the fmin-150 row as "sinc at any mass" — it is
"sinc everywhere we looked, and we stopped at 55".

`nearest` is never competitive: 200–443 nats throughout, crossing 1 nat of error by SNR 2–6, i.e.
already unusable at O4 SNRs.

---

## 2. Why there is no automatic selection

Three successive candidate rules were built and **all three were disproved by measurement**.

**Rule 1 — key on `fNyq/fmax`.** Wrong quantity: that number is identical for every system at
fixed settings, but the right stencil is not. `Q^a_lm(t) = <h_lm(t)|d>` is band-limited by
whichever is lower, `fmax` or the *template's* own highest frequency.

**Rule 2 — key on `fNyq / min(fmax, f_ISCO(M))`.** Mis-selected at 2 of 8 masses. Fatally, the
correct stencil depends on **fmin** as strongly as on mass: at M = 5 M☉, srate 4096 / fmax 1700,
the winner flips from cubic (fmin 30) to sinc (fmin 150) with mass, srate and fmax all identical.
Those two cases require disjoint threshold ranges — (1.21, 2.33) and (2.33, 4.66) — so **no
threshold can make a `(srate, fmax, mass)` signature correct**. The signature is wrong, not the
constant.

**Rule 3 — key on `fNyq /` a PSD-integrated bandwidth (`RIFT.misc.psd_bandwidth`).** Looked clean
at quantile 0.99 on the fmin-30 points: sinc ≤ 2.99, cubic ≥ 4.33, a 45% gap. But all 9 of those
points were at **one fmin**. Across the fmin sweep the classes **overlap** over [4.21, 6.01] with 5
points inside, one sinc winner ranking above four cubic winners. A quantile sweep from 0.50 to
0.99999 finds **no** separating value (best 0.95, still 1.18× overlap). The estimator moves the
M=55 score only −7% over fmin 20→150 while the physics flips the winner.

A wrong automatic choice here is **silent** — it does not raise, it just makes the likelihood less
accurate. That is exactly the kind of error that should not be guessed at, so the flag requires an
explicit stencil name and the retired "choose for me" spelling raises.

---

## 3. Mass ladder (fmin 30)

SEOBNRv4, an IMR model. Against an exact FFT-zero-padded reference; paired, K=2000, 3 seeds; each
mass normalised to SNR_lik = 100. srate 4096, fmax 1700, fmin 30, Lmax 2. max|ΔlnL| in nats:

| M/M☉ | nearest | cubic | sinc | winner |
|---|---|---|---|---|
| 9 | 369 | 8.70 | **3.90** | sinc, 2.2× |
| 10 | 286 | 12.2 | **4.11** | sinc, 3.0× |
| 20 | 284 | 7.85 | **3.65** | sinc, 2.2× |
| 35 | 200 | **1.67** | 3.51 | cubic, 2.1× |
| 55 | 443 | **1.31** | 3.88 | cubic, 3.0× |
| 80 | 437 | **0.346** | 3.15 | cubic, 9.1× |
| 120 | 433 | **0.143** | 7.89 | cubic, 55× |

At srate 16384 (SEOBNRv4 cannot be generated at 4096 below M ≈ 8): M = 5 → cubic 21×, M = 2.6 →
cubic 34×.

**Those two rows are at a HIGHER srate, and that is why they read the other way.** The same binary
is far more oversampled at srate 16384, and oversampling — not mass alone — is what sets the
answer. Do not read them as "cubic wins at low mass"; read them as "srate moves the crossover
as surely as fmin does". Every crossover quoted in this document is **at srate 4096**.

**Do not reintroduce inspiral-only numbers here.** An earlier version of this table used TaylorT4,
which terminates at ISCO and carries no merger-ringdown. It named the **wrong stencil** at M = 9,
10 and 20, and overstated cubic's high-mass margins by up to 99×.

---

## 4. fmin sweep

Same method, 20 points, 3 seeds each; marginal winners replicated with 3 fresh seeds (all 12
identical). srate 4096, fmax 1700 throughout. Winner and margin; **capitals** mark where the
fmin-blind rule shipped in #97 named the worse stencil:

| M \ fmin | 20 | 30 | 50 | 100 | 150 |
|---|---|---|---|---|---|
| 9 | sinc 2.1× | sinc 2.2× | sinc 2.5× | sinc 6.1× | sinc 12.4× |
| 20 | sinc 1.8× | sinc 2.2× | sinc 2.9× | sinc 8.6× | sinc 15.9× |
| 35 | cubic 2.3× | cubic 2.1× | cubic 1.7× | **SINC 2.5×** | **SINC 5.6×** |
| 55 | cubic 2.4× | cubic 3.0× | cubic 4.4× | cubic 1.1× | **SINC 1.2×** |

The M=35 / fmin=150 mis-call costs 5.6×, and at 15.8 nats is a *larger absolute error than
anything cubic does at fmin 30 anywhere over 9–120 M☉* — not a bookkeeping difference.

**Conservative rule inside the measured range:** over fmin ≥ 100 **and** M ≤ 55, always choosing
sinc costs at most 1.12× (at M=55, fmin=100, the single point where cubic still wins), against **15.9×** for always choosing cubic (M=20, fmin=150 — the largest sinc-win margin in
that region; the 5.58× quoted in an earlier draft was a different quantity, the worst harm of the
old fmin-blind rule). That asymmetry is why a flat "prefer sinc" is defensible there —
bounded by the measurement, not universal.

---

## 5. Mechanism

`sinc`'s error is **flat** — 3.1–7.9 nats across the fmin-30 mass ladder and both approximants,
2.3–5.6 nats across the 20-point fmin sweep. Flat in *both* sweeps is exactly what a
window-limited, oversampling-independent error must do.

All the variation is `cubic`'s: it degrades **~6.5–9.6×** as fmin goes 20 → 150 at fixed mass (M=9:
10.7 → 69.3 nats, 6.5×; M=20: 4.7 → 45.2, 9.6×).  Note this is an ENDPOINT ratio, not a
monotone trend — cubic at M=9 is 10.7 at fmin 20 but 8.70 at fmin 30. Raising fmin cuts the long low-frequency inspiral out of
band, broadening Q relative to Nyquist — exactly sinc's regime. That is why the crossover rises.

**Margins are scoped, and the two scopes are not interchangeable.** At fmin 30, every margin either
way over M = 9–55 is 2.1–3.0× and the worst below 120 is 9.1×. Across the fmin sweep the range is
1.1× to 15.9×. Quote whichever matches the configuration you are describing.

The "330× penalty for picking sinc wrongly" quoted in pre-IMR revisions was a TaylorT4 artifact
and is gone either way — there is no longer a strong safety reason to break ties toward cubic.

**Error grows as SNR²** (measured exponent 1.999–2.006 over two decades), so the choice matters
more at 3G sensitivities.

---

## 6. Why bandwidth is hard to estimate at build time

What actually sets the answer is `fNyq` divided by the true Q bandwidth. Estimating that bandwidth
is the open problem:

- **f_ISCO is not a usable proxy.** measured/f_ISCO drifts 15.8× across 2.6–120 M☉ with IMR (worse
  than the 7.4× seen with TaylorT4) and *reverses sign* near M ≈ 10.
- **A 99.99%-power quantile of the measured spectrum is not either.** With IMR points it is
  non-monotone — sinc still wins at fNyq/f_Q = 4.63 while cubic already wins at 4.23 — because an
  IMR spectrum has a ringdown bump rather than a smooth roll-off.
- **`RIFT.misc.psd_bandwidth` does not separate the winners** once fmin varies. See Rule 3 above.

---

## 7. Cost

Measured. `sinc` relative to `cubic` in the Q product: **~4.2–4.5× on CPU** (16 taps against 4;
tap-count bound), **~1.6–3.0× on GPU** (bandwidth bound). End-to-end on CPU at fixed n_max:
nearest 9.3 s, cubic 25.1 s, sinc 85.3 s. On GPU the difference is not resolvable in wall time.

---

## 8. Limitations, and which axes have been swept

Zero noise, analytic ZDHP PSD, Lmax 2, non-spinning, equal mass except 2.6, one sky location, one
srate/fmax/PSD combination, 3 seeds. SEOBNRv4 is unreachable at srate 4096 below M ≈ 8, so the
low-fmin crossover is bracketed 20 < M < 35 but not resolved further, and the high-fmin crossover
only as "> 55".

**Swept: mass and fmin. Both moved the answer — and the second moved it *after* the first had been
published as settled.** `srate`, `fmax` and `Lmax` have **not** been swept and should be presumed
load-bearing until they are; on this heuristic that presumption has been correct twice out of two.

`srate` deserves special suspicion: it is the numerator of the fNyq/bandwidth ratio this whole
document says sets the answer, and the two srate-16384 rows in §3 already show it flipping the
winner. The entire fmin sweep is at srate 4096.

---

## 9. Backend coverage, and the one place the stencil is deliberately absent

**Added 2026-08-26.** The stencil now has **four** implementations across **three** likelihood
variants, but that is fewer moving parts than it sounds, because the variants all funnel through
two primitives.

| backend | entry point | weights from |
|---|---|---|
| numpy | `_sinc_Q_window_numpy` | `_sinc_lanczos_weight_matrix` |
| cupy | `Q_inner_product_sinc_cupy` | `_sinc_lanczos_weight_matrix` (built on device) |
| CUDA | `Q_inner_sinc` kernel | passed in from the above -- **not** re-derived in C |
| JAX | `jax_ile.core._make_gather_sinc` | `_sinc_lanczos_weights_jax` (see below) |

`factored_likelihood_freqresponse` (finite-size) and `factored_likelihood_with_rotation`
(slow-rotation) do **not** carry their own stencils: both call
`FL._q_window_numpy_interp` / `FL._q_inner_product_gpu`, so they inherit whatever the two
primitives do. Three variants, three implementations of the weights, not nine.

**The fused calibration-marginalization kernels (`cuda_Q_fused_calmarg.cu`,
`cuda_Q_fused_calmarg_distmarg.cu`) implement `nearest` only, and that is deliberate.** They do
an integer gather at `ifirst + c*N_window + i_time` inside an already register-heavy fused
reduction. `time_interp != 'nearest'` therefore raises `NotImplementedError` at the library level
and falls back to the `loop` path at the driver level; `test_calmarg_stencil_gating.py` pins both,
and pins them against `== 'nearest'` rather than a hard-coded list of the stencils that existed
when the guard was written.

### 9.1 JAX is the one backend that cannot share the weight array

The other three consume one `(n_extrinsic, 2a)` array from `_sinc_lanczos_weight_matrix`, so they
cannot drift. JAX cannot: the weights depend on the sub-sample offset, which is a *traced*
function of sky location, and `jax.grad` has to see through them. `_sinc_lanczos_weights_jax` is
therefore a second, independent expression of the same formula -- the one real divergence risk in
this feature.

What keeps it honest is `test/jax/test_jax_stencil_parity.py`, which compares the two generators
directly rather than trusting review. Measured agreement, ldas-grid + ldas-pcdev13, igwn python
3.11 / jax 0.7.1 / cupy 12.0.0, 2026-08-26:

| comparison | measured | gate |
|---|---|---|
| JAX vs numpy weights | 4.4e-16 | `1e-14` |
| JAX vs numpy assembled window (interior / left edge / right edge) | 1.1e-13 / 7.1e-15 / 1.8e-13 | `1e-12` |
| JAX vs CUDA `Q_inner_sinc` (RTX 2080 Ti, sm_75) | passes | `1e-12` |

Two conventions are easy to get right in one backend and wrong in another, so both are pinned
explicitly: the `|x| >= a` hard zero, and the fact that the unit-sum renormalisation is applied
over the **full** stencil and is **not** redone after out-of-buffer taps are dropped. A backend
that renormalised after masking agrees on weights and disagrees at the buffer edge.

**The parity suite was mutation-tested, not merely written.** Four source-level mutants of the
JAX gatherer: pointing `_GATHERERS['sinc']` at the cubic stencil (kills 6 of 15 tests), dropping
the normalisation (5), shifting the tap offsets by one (7), and replacing zero-extension with an
edge clamp (5). A fifth -- deleting the `|x| >= a` clause -- initially **survived**, and is an
*equivalent* mutation on the wired path: for `u` in `[0,1)` the only guarded tap sits at `x = -a`
where `sinc(-1) = 3.9e-17`, so the clause is worth `1.5e-33`. Off the wired path it is worth
`2.2e-3`, so it is pinned by `test_weight_parity_outside_the_unit_interval` rather than deleted.

### 9.2 Half-width is a library parameter, NOT reachable from any driver

`SINC_HALFWIDTH_DEFAULT = 8` (16 taps). `_sinc_Q_window_numpy(..., a=)` and
`Q_inner_product_sinc_cupy(..., halfwidth=)` accept a value, but **no CLI, pipeline flag or
`time_interp` spelling exposes it**, so every production path is locked to `a = 8`. That is worth
stating because the sky-offset diagnosis measured `a = 16` as visibly better on the 3G
finite-size demo (§9.3), and reaching it requires a code change, not a flag. Adding a name for it
would have to flow through `TIME_INTERP_CHOICES` in two modules, the CLI help pinning in
`test_interpolate_time_cli.py`, and the calmarg gating truth table -- and §2 is explicit that
stencil naming is deliberately locked down -- so it is left as a decision to be taken, not a
silent addition.

### 9.3 The 3G finite-size demo, where the JAX gap was found

Archived configuration of the 3-site 3G demo (CE+ET+K, m1 1.6 / m2 1.4, fmin 50, fmax 1024,
srate 2048, self-consistent Qmax 4, SNR 600, zero noise). Deterministic hierarchical-zoom peak
scan -- no sampler, no seed -- so the stencil is the only variable. Offset of the lnL peak from
the injected sky position, and `dlnL` above lnL at the truth (which for a zero-noise
self-consistent injection should be ~0):

| stencil | peak offset | `dlnL` above truth | lnL at truth | shortfall vs rho^2/2 = 180000 |
|---|---|---|---|---|
| `cubic` | 9.22' | 64.53 | 178214.14 | 1785.9 |
| **`sinc` (shipped, a = 8)** | **0.74'** | **0.515** | **179824.16** | **175.8** |
| `lanczos8` (prototype, a = 8, unnormalised) | 0.33' | 0.117 | 179915.91 | 84.1 |
| `lanczos16` (prototype, a = 16, unnormalised) | 0.13' | 0.018 | 179944.54 | 55.5 |

Against the marginalisation overhead of 14-23 nats (measured separately; roughly constant in
nats, NOT a constant fraction, so an absolute threshold is right at every SNR), shipping `sinc`
takes the shortfall from 1786 nats to 176 -- a 10x reduction, and the peak from 9.2 arcmin to
0.74. That is the improvement this stencil buys on this configuration.

**But note rows 2 and 3.** The shipped `sinc` is 2.3x worse in offset, and 92 nats worse in lnL
at the injection, than the `lanczos8` prototype **at the same half-width**. The cause is
identified exactly, not inferred: deleting the unit-sum renormalisation from the shipped stencil
reproduces the prototype **bit-for-bit** (max|diff| = 0.000e+00 over 5000 positions including
out-of-buffer). The renormalisation -- which moves the weights by at most 3.1e-4, and which the
docstring justifies as making interpolation exact for constants -- is the entire difference.

**The mechanism is NOT established, and two plausible ones were tested and died.** (i) "the
normalisation fixes DC gain and thereby tilts the passband": a bandwidth sweep from f/fNyq 0.05
to 0.98 on a synthetic band-limited signal shows no trend, ratios scattering over 0.79-1.70 with
1.00 at the top of the band. (ii) "it is a systematic gain deficit worth rho^2*(1-g)": the
best-fit complex gain differs between the two by ~1.5e-4, implying ~0.2 nats, three orders of
magnitude short of 92. Whatever produces the 3G effect is not reproduced by a single-detector
synthetic and has not been isolated -- it may involve the 3-site network, the finite-size
response, or the distance marginalisation. **Do not quote a mechanism for this row.**

**A second configuration does not replicate the ordering, so do not generalise the row above.**
Same demo, same everything, at fmax 2048 / srate 4096:

| stencil | peak offset | lnL at truth | shortfall |
|---|---|---|---|
| `cubic` | 0.571' | 179799.15 | 200.9 |
| `sinc` (shipped, a = 8) | **0.030'** | 179960.42 | 39.6 |
| `lanczos8` (a = 8, unnormalised) | 0.145' | **179999.42** | **0.6** |
| `lanczos16` (a = 16, unnormalised) | 0.031' | 179982.00 | 18.0 |

Two things break here that hold at fmax 1024. **(i) The two metrics disagree.** In lnL at truth
the renormalisation still costs (39 nats, same sign as the 92 at fmax 1024), but in sky offset it
has *flipped*: shipped `sinc` at 0.030' now beats the unnormalised `lanczos8` at 0.145'. Judge
this knob on both moments or it will tell you whatever you asked. **(ii) Half-width is
non-monotone**: `lanczos16` claims *less* lnL at truth than `lanczos8` (179982 against 179999),
which a purely stencil-limited error cannot do. Something other than the stencil is binding at
this sampling -- consistent with the `slowrot_fs_lib` at-Nyquist defect being tracked separately.

That is also an independent cross-check on that separate defect, arrived at without using any of
the sampling evidence from the sky-offset diagnosis: at fmax 1024 even `lanczos16` leaves 55.5
nats against a 14-23 nat marginalisation floor, so **33-41 nats there are not stencil error**.

So "should the shipped stencil renormalise?" is an open question with a measured cost in one
metric at two configurations and a measured *reversal* in the other, **not** a settled defect. It
is deliberately NOT changed here: the renormalisation is shared by all four backends, so flipping
it changes results for every existing `sinc` user, and the §3-§4 crossover tables were all
measured with it on. Anyone revisiting it needs more than two configurations and should report
offset and lnL shortfall side by side.


`cubic` is not merely less accurate here: it moves the likelihood peak 9.2 arcmin off a
zero-noise self-consistent injection and buys 64.5 nats for doing so. This configuration is far
below the crossover of §1 (M = 3 Msun, fmin 50), which is exactly the regime §1 assigns to
`sinc` -- the demo is a confirmation of that table, not a counter-example to it.

### 9.4 The JAX default moved from `linear` to `sinc` (2026-08-26)

`RIFT.likelihood.jax_ile.core.JAX_INTERP_DEFAULT` is the single definition, consumed by every
entry point in the package and by `--interp` in `bin/integrate_likelihood_extrinsic_jax`.
**This changes results for any caller that did not pass `interp=`/`--interp`.** Pass
`interp="linear"` to reproduce a pre-2026-08-26 run.

**Why it moved at all.** `linear` was never chosen on merit -- it was the first differentiable
stencil the JAX path had, and the default was inherited. It is the *worst* option in the registry
at high SNR, worse than `nearest`, and this driver is used **exclusively** at high SNR, so the
one regime the default was tuned for is the one regime it is wrong in. On the 3G demo `linear`
sits between `nearest` and `cubic`; §3's ladder puts `nearest` at 200-443 nats, crossing 1 nat of
error by SNR 2-6.

**Why `sinc` rather than `cubic`,** given §1 says neither is universally better: a default is
chosen for its *worst* case, not its average, because it is what people get without thinking.
§5 measured `sinc`'s error as FLAT -- 3.1-7.9 nats across the fmin-30 mass ladder, 2.3-5.6 across
the 20-point fmin sweep -- while all the variation belongs to `cubic`, whose error ranges 0.143
to 69.3 nats over the same points. Error also grows as SNR^2 (measured exponent 1.999-2.006), so
at 3G sensitivities the tail is what matters. `sinc` costs ~1.6-3.0x `cubic` in the Q product on
GPU (§7), which is the price of that bound.

**THE CONCERN, recorded because it is not resolved by the above.** `sinc` is the bounded choice,
not the better one, and there is a real population for which this default is *worse* than the old
behaviour would have been had anyone set `cubic`: §4 measures `cubic` winning by 2.1-4.4x over
9-55 Msun at fmin <= 50, and by 9.1x at 80 Msun and 55x at 120 Msun on the fmin-30 ladder. A
high-mass, low-fmin BBH -- much of the O4 catalogue -- is squarely in `cubic`'s regime, and will
now silently get `sinc` unless the caller says otherwise. The mitigation is that `sinc`'s loss in
that regime is bounded (its error never leaves 2.3-7.9 nats) whereas `cubic`'s loss in the other
regime is not, so the asymmetry favours `sinc` for a default -- but "bounded loss" is still loss,
and anyone running a high-mass campaign should pass `--interp cubic` explicitly.

Two further limits on that reasoning, both from §8: every crossover in this document is at
**srate 4096**, which has *not* been swept and is the numerator of the ratio §6 says sets the
answer; and the flatness claim for `sinc` rests on the same measurements. If srate turns out to
move the crossover as strongly as fmin did, this default should be revisited -- it would be the
third time a rule here was overturned by an axis that had not been swept.

### 9.5 The tap-axis memory trap, and why `u` is passed separately

Found by adversarial review of §9.4. The first version of this section was **wrong**, and how it
was wrong is the point: it quoted 2.7x, measured on the **CPU** backend. On CPU nothing fuses
well, so the sinc overhead hid inside an already-large baseline. On GPU -- the backend this
driver actually runs on -- XLA fuses `nearest`, `linear` and `cubic` *completely*, and only sinc
materialised its `(S, npts, 2a)` weight array. The real figure was **65x**, not 2.7x.

Whole-likelihood XLA temp, `compile().memory_analysis()`, 3 detectors, npts 614:

| S | backend | `nearest` | `linear` | `cubic` | `sinc` before | `sinc` after |
|---|---|---|---|---|---|---|
| 20000 | CPU | 1285 MB | 2464 MB | 4821 MB | 6586 MB | -- |
| 20000 | **GPU** | 101 MB | 101 MB | 101 MB | **6583 MB** | **1279 MB** |
| 2000 | GPU | 10.1 MB | 10.1 MB | 10.3 MB | 658 MB | 128 MB |

**The fix: pass the fractional offset separately.** Both accumulators build
`pos = p0[:, None] + arange(npts)` with INTEGER offsets, so `frac(pos)` does not vary along the
time axis -- only `S` distinct values exist, not `S * npts`. `_separable_u(p0)` returns that
`(S, 1)` array and every gatherer takes it as an optional third argument, so the weight array
becomes `(S, 1, 2a)` and is small enough that the surrounding product and reduction fuse, exactly
as cubic's inline weights already do. Isolated gather at S=20000/npts=614:
**1719.2 MB -> 2.7 MB, a 637x reduction.**

**It is a strict win, not a trade** -- measured on GPU (RTX 3080, container jaxlib):

| axis | before | after |
|---|---|---|
| whole-likelihood temp, S=20000 | 6583 MB | **1279 MB** |
| runtime, S=2000 | 0.00540 s/call | **0.00269 s/call** |
| accuracy | -- | unchanged, see below |

Runtime *halves* because the old form recomputed 16 sinc pairs per (sample, time-bin) when only
`S` distinct weight rows exist -- 614-fold redundant arithmetic. Post-fix sinc costs 1.33x cubic
(0.00269 against 0.00203) and 1.44x linear, which is the honest price of the stencil.

**And it is slightly MORE accurate.** `p0` is a sample index of order 1e5-1e6, so `p0 + t` can
cross a binade and drop a low mantissa bit; `frac(p0 + t)` then differs from `frac(p0)` by up to
an ulp of the position (~1.5e-11 at 65536, measured). The numpy, cupy and CUDA backends all
compute one fractional offset per sample *from the sample position* -- i.e. the separable form --
so this brings JAX into line with them rather than away.

Residual: sinc is still 12.7x cubic at the whole-likelihood level (1279 vs 101 MB), because the
per-(detector, mode) `(S, npts, 2a)` *gathered-value* array is only partly fused. At the
production default `--n-chunk 8000` that is ~512 MB against ~40 MB -- comfortable on any card
this runs on. Squeezing the last factor would mean fusing the lm contraction into the gather, as
the CUDA kernels do; not attempted.

**`u` is passed only to the weight-building stencils.** Feeding it to `nearest`, which ignores
it, is not free: an unused extra input in the banded slow-rotation trace -- which is compile-bound
rather than arithmetic-bound -- cost **>60% wall** (`test_rotation_path_a`, 69.8 s -> >113 s,
same tree, same load, one variable). So the call sites read
`u_sep = None if interp == "nearest" else _separable_u(p0)`. The memory win is unaffected: it
belongs to `cubic`/`sinc`, which still receive it. This is the same lesson as the unrolled-tap
compile blow-up in `_make_gather_sinc` -- in a large trace, graph shape can cost more than
arithmetic.

Getting `u` wrong is **silent** -- the gather returns the right shape evaluated at the wrong
offsets -- so it is pinned two ways: `test_separable_u_matches_the_general_path` compares the two
paths for all four stencils at production magnitudes, and `test_accumulators_pass_separable_u`
parses `core.py` to assert both call sites actually pass it. Mutation-tested: perturbing `u` by
0.05 fails the first for `linear`/`cubic`/`sinc` (and correctly not for `nearest`, which ignores
it); dropping the argument at either call site fails the second.

A `lax.scan` variant was also measured and rejected -- it cut the temp only to 1427 MB and cost
2.9x runtime, against the separable form's 1279 MB at 0.5x runtime.

## 10. Provenance

The fmin sweep was measured against a pinned `git archive` of the #97 merge commit `c1a2e2df`,
not a shared checkout, so a branch switch could not move code mid-run. Its fmin-30 column
reproduces #97's shipped numbers bit-for-bit, and the analysis code was validated by re-deriving
#97's published bracket from the original 9 points alone. No row is reference-limited (per-stencil
reference floors ≥ 400× below the smallest measured error; M→2M reference checks ≤ 5.7e-5 nats).
