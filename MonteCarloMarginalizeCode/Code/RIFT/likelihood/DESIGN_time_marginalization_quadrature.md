# Time-marginalization quadrature: measured record

Companion to `time_marginalization_quadrature.py`.  The module docstring carries the
argument; this file carries the numbers behind it and the harnesses that produced them,
so a reviewer can re-run rather than take them on assertion.

Harnesses (host-local, `ldas-*` NFS home): `~/tmarg_harness/`.
`probe.py` periodic-window accuracy, `wrap.py` non-periodic window, `adv.py` edge sweep and
mixed blocks, `detrend.py` the rejected endpoint-detrend, `cost.py` quadrature-only cost,
`cost_e2e.py` end-to-end through the shipped likelihood, `peaklocal2.py` the peak-local
prototype below, `real_path.py` / `simps_iso.py` the GPU correctness runs, `cost_gpu.py` the
GPU cost table.

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

Non-periodic window (segment of a longer band-limited signal, peak centred): the literal
2N forward/backward reflection measured errors from 2e-8 to 2.5e-6 nats over amplitudes
0.05–5, where Simpson can be wrong by hundreds of nats.

## Finite-window reconstruction: why decay is not an eligibility test

`kappa_rows` contains only the gathered integration-window slice of a longer inverse-FFT
series.  Zero-padding its FFT directly treats that slice as one period; centring the coarse
argmax does not bound the artificial wrap or its ringing.  More subtly, decay of the
integrand does not bound it either: `exp(lnL)` may be negligible at the edges while the
quantity being FFT-interpolated, complex `kappa`, has a large endpoint mismatch.

The adversarial review pinned that distinction with a centred analytic row: outer-eighth
log-likelihood drops 51.4 and 49.6 nats, endpoint mismatch 1720, derived refinement factor
64.  The raw periodic-slice FFT was **+140.88 nats** wrong.  Thus a 30-nat tail guard would
have certified precisely the catastrophic case it was intended to exclude.

The shipped mitigation forms the literal length-2N sequence
`[kappa[0], ..., kappa[-1], kappa[-1], ..., kappa[0]]`, FFT-interpolates that periodic,
value-continuous sequence, and retains only the forward interval.  On the counterexample the
error is **+2.4e-4 nats**.  The superficially standard 2(N-1) reflection was tested too and
was worse (+0.078 nats) because it places the turn on the endpoint sample rather than between
the duplicated endpoints.  On the ordinary longer-period fixtures the literal 2N form was
also better: 2e-8–2.5e-6 nats versus 2e-6–2.5e-4 for 2(N-1).

An internal randomized review added 400 coherent sinusoids (random amplitude, mode and
phase) to the longer-period analytic fixture.  Of 387 rows that passed the former centring
and 30-nat tail guards, raw periodization still reached 130 nats error.  Reflection reduced
the worst error to 0.0137 nats; 364/387 were below 1e-3 and only one exceeded 1e-2.  The
worst residual used a deliberately coherent near-Nyquist mode of amplitude 1894, comparable
to the signal peak.  Production remeasurement doubled its quadrature factor from 128 to 256
and left the value unchanged, establishing that 0.0137 is reconstruction, not integration,
error.  Local Lanczos reconstruction was also tested on that worst row and was inferior:
errors 7.24, -0.779, 0.164, 0.0646 and 0.0273 nats for half-widths 8, 16, 32, 64 and 128.
This records the measured limitation instead of implying that reflection recovers the
unavailable full-period series exactly.

This is a numerical boundary condition, not a claim that the physical correlation reverses
outside the window.  The integration domain remains exactly `[t0, t_{N-1}]`; the backward
half contributes no probability mass.

## Boundary peaks: diagnostic, never a rule switch

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

The table records the rejected raw-slice periodic reconstruction.  An endpoint-ramp detrend
was also rejected: it halves the interior error but is worse at 8 and 2 samples from the edge
(`detrend.py`).

The former outer-eighth guard is now diagnostic-only.  A sweep immediately across its
boundary (peak samples 75.3, 76.3, 77.3 for N=614) gave reflected errors of order 1e-6 nats;
switching to Simpson there would create a discontinuous, arbitrary loss of accuracy.  Peaks
at the actual integration endpoint remain a distinct physical truncation problem.  The
reflected result is best-effort on the unchanged domain and the boundary count remains in
`last_report()` so truncation is visible, but it neither returns a lower-resolution rule nor
raises: in the production pipeline either behavior can silently bias selection, because an
exception may be interpreted as waveform failure and excise that configuration.

This distinction is load-bearing:

* a **numerical reconstruction boundary** is mitigated by forward/backward reflection;
* a **physical integration boundary** is reported, not silently reclassified;
* Simpson fallback is retained only where no peak width can be measured or no refinement is
  needed, not because a row crossed a location or tail-height threshold.

One endpoint corner needs explicit classification.  At an argmax on the first or last sample,
the nominal centred curvature stencil is clipped inward; on a severely under-resolved row it
can measure positive curvature away from the peak and call the row flat.  A nonconstant row in
that state now receives a seed factor of 4, after which dense-grid remeasurement derives the
needed resolution (factor 16 in the pinned sharp-endpoint fixture).  A truly constant antenna
null remains flat and unrefined.  This prevents an exact-boundary row from silently reaching
Simpson through a different classification path.

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

The table below predates the finite-window fix and measures the rejected raw-slice FFT.
Forward/backward reconstruction doubles the FFT period and the chunking budget accounts for
that larger temporary.  Treat these numbers as the lower-bound historical record, not as a
current performance claim; correctness is the gate for this opt-in quadrature and the
peak-local follow-up remains the intended cost reduction.

End-to-end through the shipped likelihood, n_extrinsic 4000, 3 IFOs, CPU time:

| sigma_t/deltaT | Simpson | band-limited | ratio |
|---|---|---|---|
| 1.74 | 0.212 s | 0.241 s | 1.1x |
| 0.55 | 0.302 s | 0.610 s | 2.0x |
| 0.17 | 0.180 s | 1.741 s | 9.7x |
| 0.055 | 0.250 s | 6.652 s | 26.6x |

Host-sensitive: O4c measured the same quantity moving up to 2x between hosts.  The Simpson
baseline is rho-independent by construction, so a run where it moves with rho is contaminated.

### On GPU, which is what production actually runs

The table above is CPU.  Production ILE runs `--vectorized --gpu`, and the ratio there is
DIFFERENT and mostly WORSE, so the CPU table must not be quoted as the cost of the option.

Method (`~/tmarg_harness/cost_gpu.py`, the GPU sibling of `cost_e2e.py`; same shipped
likelihood, same synthetic band-limited kappa, n_extrinsic 4000, 3 IFOs).  `ldas-pcdev13`,
`CUDA_VISIBLE_DEVICES=3` (GeForce RTX 2080 Ti, cc75 -- the cc120 Blackwell devices on that
host cannot be compiled for by cupy 12), cupy 12.0.0 from the IGWN CVMFS python,
`OMP_NUM_THREADS=1`, the GPU otherwise idle.  The two arms are INTERLEAVED within a replicate
and the order is swapped between replicates, so a host that gets busier mid-run cannot
masquerade as a ratio; 5 replicates on GPU, 3 on CPU; the quoted spread is min-max of the
per-replicate ratios, not a self-reported error.  **Timing is wall clock with an explicit
device synchronize**, NOT `process_time` as on the CPU arm: `process_time` on a cupy arm is
launch plus synchronize-spin, which is not the quantity of interest.  The Simpson baseline is
rho-independent by construction and is checked to move by <1.25x across each ladder; it moved
by 1.03-1.12x, so none of these runs is contaminated.  `sigma_t/deltaT` is measured per rung,
not assumed, and the srate-16384 amplitude ladder is scaled by 16 rather than 4 because on
this fixture `rho_sq = 0`, so `lnL` is LINEAR in the amplitude and `sigma_t ~ 1/sqrt(amp)`.

GPU, srate 4096, npts 614 (even -- median wall seconds per call):

| sigma_t/deltaT | Simpson | band-limited | ratio | spread over 5 reps |
|---|---|---|---|---|
| 1.735 | 0.0130 s | 0.0250 s | 1.9x | 1.9-1.9 |
| 0.549 | 0.0127 s | 0.0549 s | 4.3x | 4.2-4.4 |
| 0.174 | 0.0128 s | 0.1429 s | 11.1x | 11.0-11.3 |
| 0.055 | 0.0127 s | 0.4324 s | 34.1x | 33.2-35.5 |

GPU, srate 16384, npts 2457 (**odd** -- the common production case, three of five rates):

| sigma_t/deltaT | Simpson | band-limited | ratio | spread over 5 reps |
|---|---|---|---|---|
| 1.647 | 0.0172 s | 0.0282 s | 1.7x | 1.4-1.7 |
| 0.521 | 0.0174 s | 0.0840 s | 4.8x | 4.4-5.0 |
| 0.165 | 0.0171 s | 0.2690 s | 15.8x | 14.0-16.9 |
| 0.052 | 0.0178 s | 0.9422 s | 56.4x | 51.1-64.6 |

Same host, same script, numpy backend, as the control that ties this to the CPU table above
(the published CPU row was taken elsewhere; this rules out a host difference masquerading as a
backend difference):

| sigma_t/deltaT | srate 4096 CPU ratio | srate 16384 CPU ratio |
|---|---|---|
| ~1.7 | 1.1x | 1.1x |
| ~0.53 | 2.4x | 2.1x |
| ~0.17 | 8.9x | 6.2x |
| ~0.055 | 31.3x | 19.7x |

The srate-4096 CPU column reproduces the published CPU table (1.1 / 2.0 / 9.7 / 26.6) within
the documented host-to-host scatter, so the GPU/CPU difference below is a backend effect.

**Read the ratio and the seconds separately, because they say opposite things.**  The GPU
ratio is worse -- 1.9x rather than 1.1x where the integrand is already resolved, and 56x
rather than 20x at srate 16384 -- and it gets worse with npts on GPU while it gets BETTER with
npts on CPU.  The mechanism is the denominator, not the numerator: the GPU Simpson baseline is
overhead-dominated and barely scales with npts (0.0127 -> 0.0172 s, 1.35x, for 4x the points),
while on CPU it scales with the work (0.53 -> 1.83 s, 3.5x).  The refinement itself scales
about the same on both (GPU 0.43 -> 0.94 s, CPU 17.0 -> 36.3 s).  So on GPU the option is
measured against a baseline that is nearly free, and any added work reads as a large multiple.
In absolute terms the GPU band-limited call at the worst rung costs 0.43 s (srate 4096) or
0.94 s (srate 16384), against 17.0 s and 36.3 s for the CPU SIMPSON baseline on the same host
-- i.e. GPU band-limited is still ~20-40x cheaper in seconds than CPU Simpson.

That is decision-relevant in two directions.  A campaign already on GPU pays a much larger
FACTOR than the CPU table suggests, and at 3G rates and SNRs the factor keeps growing, which
strengthens rather than weakens the case for the peak-local follow-up below: the dense
strategy's cost is what the ratio measures, and peak-local removes it.  But the factor is
being taken against a baseline of ~13-18 ms, so for an ILE job whose per-call budget is
dominated by anything else, the absolute cost may still be acceptable where the accuracy is
needed.  Neither reading is available from the CPU table alone.

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
