# Time-marginalization quadrature: measured record

Companion to `time_marginalization_quadrature.py`.  The module docstring carries the
argument; this file carries the numbers behind it and the harnesses that produced them,
so a reviewer can re-run rather than take them on assertion.

Harnesses (host-local, `ldas-*` NFS home): `~/tmarg_harness/`.
`probe.py` periodic-window accuracy, `wrap.py` non-periodic window, `adv.py` edge sweep and
mixed blocks, `detrend.py` the rejected endpoint-detrend, `cost.py` quadrature-only cost,
`cost_e2e.py` end-to-end through the shipped likelihood, `peaklocal2.py` the peak-local
prototype below, `real_path.py` / `simps_iso.py` the GPU runs.

## The defect

Simpson at the fixed spacing `deltaT = 1/srate` against an integrand of width
`sigma_t = 1/(2 pi rho sigma_f)`.  Grid-phase span of the reported lnL over `2*deltaT`,
35+30 Msun SEOBNRv4 H1L1V1, rho=40 — **measured on the JAX mirror**, quoted as the physical
scale of the defect:

| srate | 4096 | 8192 | 16384 |
|---|---|---|---|
| span | 1.649 nats | 0.385 | 0.0095 |

## Accuracy, against an analytic truth

Synthetic band-limited kappa, srate 4096, npts 614, error in nats at three grid phases:

| sigma_t/deltaT | Simpson | band-limited | factor |
|---|---|---|---|
| 2.27–2.61 | +5e-6 … +0 | 0 | 1 |
| 0.72–0.83 | +2.5e-3 … -1.1e-4 | 0 | 4 |
| 0.25–0.28 | +0.844 / +0.242 / -1.101 | 0 | 16 |
| 0.10–0.12 | +1.742 / -1.833 / -11.68 | 0 | 32 |
| 0.046–0.052 | +2.548 / -15.33 / -66.18 | 0 | 64 |
| 0.016–0.019 | +3.589 / -139.4 / -549.0 | 0 | 256 |
| 0.007–0.008 | +4.393 / -710.6 / -2760.4 | 0 | 512 |

Non-periodic window (segment of a longer band-limited signal, peak centred): band-limited
error <= 5e-5 nats where Simpson is off by up to 420.

## Edge guard: a bound on WHERE, not on HOW MUCH

Peak swept toward the window edge, `sigma_t/deltaT = 0.042`:

| distance from edge (samples) | 307 | 100 | 30 | 8 | 2 | 0 |
|---|---|---|---|---|---|---|
| band-limited | 5e-6 | 4.6e-3 | 5.2e-2 | 5.6e-2 | -3.3 | **+88.8** |
| Simpson | -29.2 | -29.9 | -29.3 | -29.4 | -29.7 | -29.9 |

`lnL` is linear in `kappa`, so the error at a fixed distance scales with amplitude.  Just
outside the guard: -8.0e-4 / -8.1e-3 / -8.1e-2 / **-0.846** nats at peak lnL 5.3e2 / 5.3e3 /
5.3e4 / 5.3e5 (rho ~ 33 / 103 / 326 / 1031).  The O4c effort measured the same linear scaling
on a different fixture and implementation and reached the same magnitude at the same
amplitude — corroboration across lines, not a single-fixture artefact.

Rejected: an endpoint-ramp detrend.  It halves the interior error but is WORSE at 8 and 2
samples from the edge (`detrend.py`).

## Odd npts

`marginalization_time_grid(0.075, 1/srate)` gives npts = 153 / 307 / 614 / 1228 / 2457 at
srate 1024 / 2048 / 4096 / 8192 / 16384 — **odd at three of five**.  A spectrum split at
`h = n//2` files the highest positive frequency under a negative frequency for odd `n`:
max error 1.4e-12 at n=614 but 4.1e-1 at n=613, 5.4e-1 at n=307, 6.0e-2 at n=2457.  Exact at
the samples, wrong between them — so a "reproduces its input" test cannot see it, and a
fixture with an empty top bin cannot either.

## CPU/GPU: a PRE-EXISTING divergence, not introduced here

`factored_likelihood` integrates with scipy's `simpson` on CPU and the vendored
`optimized_gpu_tools.simps` on GPU.  The latter is an old scipy with `even='avg'`; modern
scipy uses the Cartwright correction.  **Odd N agree exactly; even N do not** — and
production npts is even at srate 4096/8192:

    n=613 random  scipy=318.0505029932  gpu=318.0505029932   reldiff 0
    n=614 random  scipy=309.1334604435  gpu=305.7046546187   reldiff 1.1e-2
    n=615 random  scipy=298.5401667745  gpu=298.5401667745   reldiff 0

Through the shipped likelihood on `ldas-pcdev13`, numpy vs cupy differ by up to **0.405 nats**
under the historical `simpson` quadrature.  Worth its own investigation; not repaired here.

Measured *after* this change, same inputs: band-limited numpy-vs-cupy max 1.1e-6, median
4.9e-14, versus Simpson's max 0.405, median 5.4e-3 — refined rows integrate with trapezoid on
the dense grid, which has no even/odd ambiguity, so this path removes the divergence where it
applies.

## Cost, and why the strategy should change

End-to-end through the shipped likelihood, n_extrinsic 4000, 3 IFOs, CPU time:

| sigma_t/deltaT | Simpson | band-limited | ratio |
|---|---|---|---|
| 1.74 | 0.212 s | 0.241 s | 1.1x |
| 0.55 | 0.302 s | 0.610 s | 2.0x |
| 0.17 | 0.180 s | 1.741 s | 9.7x |
| 0.055 | 0.250 s | 6.652 s | 26.6x |

Host-sensitive: O4c measured the same quantity moving up to 2x between hosts.  The Simpson
baseline is rho-independent by construction, so a run where it moves with rho is contaminated.

### The follow-up: enumerate peaks, integrate locally  (RO'S, 2026-08-27)

The dense strategy refines the WHOLE window to a peak whose width shrinks as 1/rho, so it
works hardest exactly where the peak occupies least of the domain.  It conflates two
requirements that should be separated:

* resolving `kappa(t)` enough to **enumerate its extrema** — a small factor, and
  **SNR-independent**, because kappa is band-limited at Nyquist by construction;
* resolving `exp(lnL)` to integrate it — the rho-dependent part, needed only over a few
  `sigma_t` around each enumerated peak.

Enumeration is also what makes the truncation *rigorous* rather than hopeful, which is the
brief's warning about PR #201's seed-and-hope: every maximum of the band-limited interpolant
is found, so the mass outside the local windows is bounded rather than assumed.  The O4c
effort sharpened the argument usefully — every shipped callback is monotone in `Re kappa` /
`|kappa|`, so the maxima of `lnL` ARE the maxima of `kappa`, and enumeration on kappa alone
suffices whatever the callback.

Prototype (`peaklocal2.py`), same analytic truth, windows merged into disjoint intervals:

| sigma_t/deltaT | rho~ | dense err | peak-local err | dense pts | local pts | speedup |
|---|---|---|---|---|---|---|
| 0.74 | 2 | 0.000000 | 0.000000 | 2,456 | 3,710 | 0.7x |
| 0.15 | 11 | 0.000000 | 0.000000 | 9,824 | 430 | 23x |
| 0.105 | 15 | 0.000000 | -0.000000 | 19,648 | 98 | 200x |
| 0.017 | 98 | 0.000000 | 0.000000 | 78,592 | 98 | 802x |
| 0.0023 | 692 | 0.000000 | 0.000000 | 628,736 | 97 | **6,482x** |

Exact at every SNR, flat at ~97 points above rho ~ 15.  **Merging the windows is what makes
this one algorithm rather than a regime switch**: isolated peaks give a tiny union;
overlapping peaks grow the union to the whole window and the method degenerates continuously
into the dense grid.  Without merging it double-counts and gives +1.6 nats at rho ~ 6.

Sequencing (RO'S): land the dense implementation first as the reviewed reference, then this
as a separate PR that can be A/B'd against it.  The default stays `simpson` regardless of how
cheap it turns out.
