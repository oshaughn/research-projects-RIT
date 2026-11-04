# DESIGN: early-time multirate precompute (data-side weighting, DSWc)

> **PROTOTYPE, NOT READY FOR HIGH-SNR USE** (RO, 2026-10-20). The multirate template conventions
> (per-|m| f_min, XHM multibanding off, standoff 0.964) differ from the LAL-closest convention
> (common f_min, multibanding off) by up to ~1.2e-3 of rho^2/2: about 55 nats at rho 300 and 610
> at rho 1000 for 30+6. Per-|m| f_min drops in-band m >= 3 content. High- and low-SNR use will
> need different conventions (future work). Offsets: "Template conventions against production"
> below. ILE prints this warning whenever `--early-time-multirate` is set.

**Status:** implemented as an opt-in prototype (research-projects-RIT #436,
`factored_likelihood_multirate.py`, ILE `--early-time-multirate`); default off. Tests:
`test/test_multirate_*.py` (core-unit-check; cupy legs in `test_multirate_cupy.py`, GPU only).
Validation records from the PR's own review moved to RIFT_roboto_paper
`analyses/early_time_compression/rift_interface/validation_pr436/`.
**Evidence:** RIFT_roboto_paper `analyses/early_time_compression/`: session A's
`RESULTS_2026-10-07b_review.md` (PR #216, `64573f9a`) and session B's `rift_interface/`
notes (PR #215). Base: `rift_O4d` at `615dc1bfa`. No default changes.

## Variant

| Item | Choice | Evidence |
|---|---|---|
| Q, data side | `dbar = C^-1 d` once per job; early segment low-passed (200 dB FIR) and decimated to fs_e; final 100 s (400 s for CE at rho ~ 1000) at full rate | A: Verdict; Table E |
| C^-1 | RIFT's own inner-product weights, `InnerProduct.weights2side` (`lalsimutils.py:2354`). With `--inv-spec-trunc-time 0` these are 1/S; otherwise they are the truncated weights. Either way Q matches today's | |
| Data cut at f_max | RIFT's hard cut. Combined DSWc + multibanded U, V at 160 dB with the hard cut, rho 1000: bns_ce D1 1.55e-3, D3 4.4e-3; bns_o4 D1 1.73e-4, D3 7.1e-3 (32 s) / 3.4e-3 (64 s); hmedge_o4 D1 1.7e-5, D3 2.4e-4; peak-time shift <= 3e-9 sigma_t. Equal to the roll-off rows (A, 2026-10-08). At 200 dB the hard cut again equals the roll-off: bns_ce D1 7.4e-6, bns_o4 1.1e-4 | A: combined runs 569269890 (`451e2115`) |
| Early template | RIFT's own call at `deltaT = 1/fs_e`, then cos^2 taper over the top quarter below fs_e/2 | A: Table C (emulated with IMRPhenomTHM); B: XHM pair agrees after a 30-38 Hz low-pass |
| Late template | full rate, short buffer, per-mode f_min (below) | B: `wf_pair_hm.py` |
| U, V | FD modes on a multibanded grid, 32 s margin | A: Table D |
| Template window | sigma_h >= 1 s | A: Verdict |

The full-rate lnL is unchanged. It is not converged in f_max either way; that is a separate
open item (A's `development/OPEN_fmax_convergence_full_rate.md`).

## Opt-in interface

- ILE flag `--early-time-multirate SCHEDULE.json`, default off. The schedule carries, per
  detector and segment: epoch, duration, rate, filter, valid interior, guard, owner rule,
  noise-weight convention, plus the frozen chirp-mass and t_c range and a hash.
- Module `RIFT/likelihood/factored_likelihood_multirate.py`, function
  `PrecomputeLikelihoodTermsMultirate(...)`. It takes the arguments of
  `PrecomputeLikelihoodTerms` (`factored_likelihood.py:540`) plus the schedule and the
  per-job data products, and returns the same tuple. Everything downstream, NoLoop
  included, is untouched.
- Dispatch at the precompute call (`integrate_likelihood_extrinsic_batchmode:3821`). Build
  the data products once, after the data and PSD load (`:1503`, `:1525`).
- Refuse before data loading, with a nonzero exit (`validate_job`): calibration marginalization, `--freqresponse`, ROM basis, a
  TD-only approximant, an intrinsic point outside the frozen range, a waveform whose
  generator rejects the low rate, in-plane spins (precession; see below),
  `--internal-precompute-ignore-threshold` (not ported), or `--early-time-multirate-waveform xp`
  without `RIFT.likelihood.gpu_waveform_xhm` (#465).
- Log one line per job naming the schedule, its hash and the segments.

## Time convention

RIFT's FD branch (`lalsimutils.py:3796-3870`) rolls each mode by an integer number of samples
and sets the epoch from the exact centering fraction. Without `fd_alignment_postevent_time`
the fraction is 0.9, so sample k sits at `epoch + (k - frac(0.1 N)) dt`. Direct calls of
different N then disagree by up to one sample (0.42 rad in (2,2) between 128 Hz and 4096 Hz;
B's `wf_pair.py`). ILE passes `fd_alignment_postevent_time=2`
(`integrate_likelihood_extrinsic_batchmode:3737`). The roll is then 2 s x srate, a whole
number of samples, and the epoch is exact for segments of at least 8 s. The t_c
investigation (RIFT_roboto_paper `5eabd7022`) measured 0.00 us at T = 8, 12 and 16 s.

The prototype passes the same kwarg for every piece. Every buffer is a power of two of at
least 8 s (early 8192 s at 128 Hz, late 512 or 1024 s at 4096 Hz), so all pieces have exact
epochs. They agree with each other and with today's ILE templates, and acceptance 1 holds
with no correction.

## Per-mode start frequency for late templates

**Aligned spins only.** For a non-precessing source, mode (l, m) at time tau before the
peak has frequency about (m/2) f22(tau). RIFT's call starts every mode at the same f_min in
the mode's own frequency. For |m| > 2 that is earlier than a short buffer holds, so it wraps.
For |m| = 1 it starts too late.

For a precessing source this rule does not hold (R. O'Shaughnessy, 2026-10-08). Each
inertial-frame (l, m) mode mixes co-precessing components m' = -l..l, so it carries content
near (m'/2) f22 for several m' at once. No single f_min per inertial mode both covers the
m' = 1 part and keeps the m' = l part inside the buffer. One call covering both needs a
buffer about l^(8/3) times longer than tau_start (about 40x for l = 4). Candidates, none
tested:
- generate co-precessing-frame modes per m' with f_min = (m'/2) f22(tau_start), then rotate
  to the inertial frame with the precession angles over the late segment;
- generate the late template from a low f_min on a long buffer, then window it in time
  (no saving on the late piece).
A's precessing partition (hmprec_o4: IMRPhenomTPHM, in-plane chi1 = 0.5, l <= 4) passes the
accuracy gates with A's own templates: D1 1.8e-5 at 160 dB and 2.9e-7 at 200 dB, D3 <= 1.3e-4,
spectral D2 4.2e-7 (PR #216, `2950de6f`). That is one system (8 + 1.4, chi1 = 0.5) whose early
window holds 5e-4 of rho^2. It tests the rate bound and mode mixing, not lnL accuracy at
large early SNR. The RIFT late-template route above is also still missing. Until one is designed and measured,
the prototype refuses in-plane spins. The early-rate
bound uses m_max = l_max, which covers mode mixing. The precession-frequency spread needs
its own margin there.

- One call per |m| group: `f_min,lm = (m/2) f22(tau_start)`, inside RIFT's call with
  `fd_standoff_factor = 0.964` (the `hlmoft` default). The production driver sets 0.9
  (`factored_likelihood.py:488`); see "Template conventions against production" below. tau_start
  is the late-segment start plus guard plus 100 s.
- Buffer: a power of two at least tau_start plus 2 s.
- Measured (B, 3G BNS, zero spin, XHM l <= 4, tau_start 600 s, 1024 s buffer): all modes
  agree with the early template across the guard to 1.1e-4 rad and 7e-5 in amplitude. With
  one f_min, |m| = 1, 3, 4 fail at order unity. Precessing sources were not tested.
- Cost: four late calls instead of one, all on the short buffer. The cost row's 0.064-0.107
  late fraction assumed one call; per-|m| calls must be measured.

## Slow rotation (path to G9)

The rotation bank (`factored_likelihood_with_rotation.py:380-463`) builds chi_a by applying
a time-derivative weight (order p) and a sidereal modulation exp(i n Omega t) to each mode.
Both are linear, so each segment builds its own chi_a on its own grid, with the same time
reference. Q_a is computed per segment as for a = 0. Two conditions (C, 2026-10-08):
- The derivative weight commutes with the partition only where the template window is flat.
  Require w_h = 1 on the support of w_e, erfc tails included. Otherwise apply the derivative
  before windowing.
- The sidereal shift n f_sid (2.3e-5 Hz for n = 2) is far below a coarse bin, but it is not
  negligible. Over the earliest band (T ~ 6000 s) it builds about 0.9 rad of phase. The
  2|a|^2 U, V terms per detector need the modes evaluated at f - n f_sid on the coarse grid.
  XHM's `FrequencySequence` path supports that; reusing the nearest bin would not work.
Memory and time then scale with |a| on the low-rate arrays, not on N. Not in the first
prototype: it refuses `--rotation-slow` until the static path passes.

## Waveform-side controls (measured 2026-10-08)

Two template effects are not multirate errors, and the tests control both
(RIFT_roboto_paper PRs #220, #221):
- RIFT's FD-branch end taper covers the last 1% of the buffer. With a 2 s post-event time it
  reaches the merger for buffers longer than about 180 s, differently for each buffer length.
  The tests use a 24 s post-event time on every piece; production keeps 2 s, and the fix there
  is Richard's call.
- XHM's internal multibanding grid depends on f_min, so calls at different f_min differ by up to
  4e-5 in amplitude and 3e-4 rad at a fixed frequency (XPHM: order unity). The tests set
  `PhenomXHMThresholdMband=0` and `PhenomXPHMThresholdMband=0` on every piece. This costs time.
- On the DSWr path the low-rate template is already band-limited, so w_h has no measurable
  effect; it is kept to match the validated recipe.

## Stages measured

Per stage: elapsed time, peak RSS (`VmHWM`) and read bytes, using the existing profiling
wrapper extended to the new functions.

| Stage | When |
|---|---|
| dbar, low-pass and decimate, late segment cut | once per job |
| early modes at fs_e plus taper | per point |
| late modes, per-|m| f_min, short buffer | per point |
| U, V on the multibanded grid | per point |
| Q early (low-rate sub-phase inverse FFTs, lag window only) and Q late | per point |
| integration | per point, unchanged |

## Acceptance

1. Status 2026-10-20: met only with production set to the multirate template conventions
   (multibanding off, per-|m| f_min, standoff 0.964): then one_segment equals production
   `rholms`, `crossTerms` and `crossTermsV` to 1e-16. Against the production default it differs
   by the template offset in "Template conventions against production".
2. Status 2026-10-09: passed (RIFT_roboto_paper PR #221: CE BNS from 10 Hz, XHM l <= 4, 256 Hz).
   2a: D3_Q 2.5e-5 at rho = 1000 with the interpolated early Q. 2b: D1 2.3e-4 at rho = 1000 over
   four templates, draws and lags, with U, V from the early and late templates; peak shift
   2e-9 sigma_t; D3_UV 8e-8.
   The two-rate schedule meets A's pre-registered gates (`TOLERANCES.json`): D1 <= 0.1 nats,
   D3 <= 0.01, at rho = 20, 100, 300 and 1000, three noise seeds. D2 is the DSW form
   (`amendment_2026-10-08`, PR #216 `5b8e374a3`): D2_DSW = || W [ w_e (1 - LP^2)(w_h h) ] ||,
   with h the raw template, gate min(0.03, 0.1/rho_e). The prototype computes it both per mode
   and for the injected strain. The injected-strain form has never been computed. For
   precessing templates the per-inertial-mode D2_DSW is the spectral rate-bound check.
   G4 also holds: the peak-time shift is at most 0.1 sigma_t. A's end-to-end D1 values at
   rho = 1000 are in PR #216 (`2950de6f`); they use A's IMRPhenomTHM/TPHM modes, not RIFT's XHM
   early-template path.
3. Measured per-point time and peak RSS are compared with B's prediction (5.0-7.3x for the
   3G CE BNS). A gap of more than 2x is a finding.
4. Status 2026-10-09: passed (PR #221, `RESULTS_2026-10-09_accept4.md`): zero-noise O4 BNS,
   NoLoop, two_rate against the one_segment control, per-sample lnL equal to 1.1e-5 within 50
   nats of the maximum. One end-to-end ILE run on a known injection, with the multirate path in force (log line
   present). A run that falls back to the full-rate path does not test it. Both paths use
   per-|m| f_min and the controls above, so a production higher-mode wrap or end-taper
   difference is not read as a multirate failure.

## Template conventions against production (measured 2026-10-20)

Multirate templates use three conventions that production does not: XHM multibanding off,
per-|m| f_min = (m/2) f_min, and the `hlmoft` standoff 0.964 (production 0.9). With all three
set, production arithmetic reproduces the one_segment control (D) to 1e-16.

lnL at 33 fixed extrinsic samples, zero noise, H1+L1, O4 PSD, 20-1700 Hz, XHM l <= 4; data from
`lsu.hoft` at 8 Hz, multibanding off. Entries are max |Delta lnL| / (rho^2/2). Delta lnL scales
as rho^2 in zero noise.

| Difference | 36+29 | 30+6 | 10+4 |
|---|---|---|---|
| multibanding on vs off (A-B) | 1.1e-4 | 1.4e-4 | 1.6e-4 |
| per-\|m\| f_min vs common (B-C) | 8.2e-5 | 9.4e-4 | 1.7e-4 |
| standoff 0.964 vs 0.9 (C-C964) | 9.9e-5 | 2.9e-4 | 5.5e-5 |
| C964 vs one_segment (C964-D) | 1.4e-16 | 1.7e-16 | 1.1e-16 |
| production vs one_segment (A-D) | 1.7e-4 | 1.2e-3 | 2.9e-4 |
| A-D in nats, rho 300 / 1000 | 7 / 83 | 55 / 610 | 13 / 145 |

Production with multibanding off and a common f_min (B) equals a reference with every mode
started at f22 = 8 Hz to 1e-5 nats, so B is closest to LAL with multibanding off. Per-|m| f_min
drops the m >= 3 modes between f_min and (m/2) f_min, which is in band, and moves the standoff
taper into band. Multirate-vs-production comparisons therefore carry about 10^2 nats at rho 1000
before any multirate error; two_rate against one_segment does not. Script and outputs:
RIFT_roboto_paper `analyses/early_time_compression/rift_interface/validation_pr436/template_offset/`.
