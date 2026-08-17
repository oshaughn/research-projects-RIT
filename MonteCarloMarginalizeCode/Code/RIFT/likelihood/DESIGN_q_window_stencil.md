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

`nearest` is never competitive: 200–440 nats throughout, crossing 1 nat of error by SNR 2–6, i.e.
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

## 9. Provenance

The fmin sweep was measured against a pinned `git archive` of the #97 merge commit `c1a2e2df`,
not a shared checkout, so a branch switch could not move code mid-run. Its fmin-30 column
reproduces #97's shipped numbers bit-for-bit, and the analysis code was validated by re-deriving
#97's published bracket from the original 9 points alone. No row is reference-limited (per-stencil
reference floors ≥ 400× below the smallest measured error; M→2M reference checks ≤ 5.7e-5 nats).
