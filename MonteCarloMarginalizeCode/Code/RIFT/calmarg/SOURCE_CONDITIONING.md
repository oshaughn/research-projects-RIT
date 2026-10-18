# RIFT source adapter conditioning and time convention

The bilby source adapters in `rift_source.py` must preserve the waveform selected
by RIFT, not condition it again after assembling polarizations. The old adapter
added `SimInspiralREAL8WaveTaper` to plus and cross and rotated their joint peak
to sample zero. The latter discarded the returned waveform epoch and could move
the waveform differently for different inclinations or orbital phases.

## Which path owns the taper?

| `h_method` | Generator | Conditioning responsibility |
| --- | --- | --- |
| `gws_hlmoft` | `GWSignal.hlmoft` | Requests backend conditioning and applies RIFT's mode start taper when `P.taper` requests it. Backend options can change backend conditioning; the RIFT wrapper retains its own policy. |
| `hlmoft` | `lalsimutils.hlmoft` | Chooses conditioning by model and options. ChooseFDModes and the general TD-mode path support `no_condition`; legacy SEOB and reconstructed-polarization routes have their own tapering. |
| `internal_hlmoft` | `internal_hlm_generator` | Uses the selected ILE generator and its supplied options. ROM, NR and diagnostic routes need not share the usual mode-taper policy. |
| `hoft` and the existing fallback | `lalsimutils.complex_hoft` | Applies the requested LAL taper to both polarizations before padding and assembly. The adapter still requests `SIM_INSPIRAL_TAPER_START`. |

The legacy `RIFT_lal_binary_black_hole_orig` has no internal-mode branch:
its existing fallback remains `complex_hoft`.

This is an ownership contract, not a claim that every low-level output has a
taper. In particular, an explicitly unconditioned mode waveform must remain
unconditioned in reweighting if that is the waveform used by ILE. The adapter
must not undo that choice. An additional automatic polarization taper would
also be unsafe for internal routes whose conditioning differs by model.
Required default tapering remains in the selected generator. This patch does
not change those generators, their defaults, or their interpretation of options.

## Preserve the physical epoch

For a complex strain with sample interval `deltaT` and returned epoch `t0`, the
DFT of each polarization must be multiplied by `exp(-2 pi i f t0)`. The source
adapter now does this on the native FFT grid, with no polarization-peak shift.
Frequency selection/interpolation then returns the requested grid, including a
zero-start grid with a cutoff below Nyquist. Geocentric arrival time supplied to
bilby therefore has the same physical meaning as the epoch in RIFT's modes.
No fitted timing correction or caller-side time offset is required.

Both changes alter the fixed-time likelihood from this adapter. Calibration
reweighting should use the corrected waveform in both baseline and calibrated
likelihoods; do not combine old baseline values with new calibrated values.
Time marginalization can absorb a small epoch shift when adequately resolved
and supported by the same prior, but does not justify changing the waveform's
conditioning or epoch. Existing saved reweighting results are not rewritten.

## Regression checks

`test/test_calmarg_source_conditioning_epoch.py` compares fractional-epoch
transforms against an independent LAL FFT, checks invariance under a change of
sample origin, and tests all three adapters and their routes with conditioned
and intentionally unconditioned generator output. Real-generator cases cover
SEOBNRv5PHM GWSignal, legacy SEOBNRv4 modes, IMRPhenomTPHM TD modes, IMRPhenomXPHM FD modes,
internal TD modes, and a TaylorT4 polarization fallback. The fallback's required
polarization taper is counted and retained; the adapter adds none. Optional
pySEOBNR absence skips only that real-generator case.


A separate development-branch H1/L1 replay uses the aligned-spin SEOBNRv5PHM
fixture retained in paper-repository PR234. Corrected bilby-source versus
unchanged refined NoLoop differences at its fiducial, first and former worst
sky points are 3.5e-6, 3.1e-6 and 1.2e-5 nat, with no empirical time correction.
`SOURCE_CONDITIONING_VALIDATION.json` records the exact configuration, inputs,
source hash, development base commit, dependency versions and replay script.
The replay needs the recorded Mac paper checkout and external PSDs; it is not
an automatic CI test or a calibration-integration/production-posterior validation.


## Optional native SEOB mode conditioning

Use `--gwsignal-mode-conditioning native` with `--use-gwsignal` to select
SEOBNRv5HM/PHM native conditioning on the shared modes. Legacy remains the
default. The Asimov key is `waveform.gwsignal mode conditioning`; see
`RIFT/physics/GWSignal-options.rst` for an example. The selector travels through
ordinary/extrinsic ILE and calibration reweighting without entering the native
backend options dictionary. Native conditioning rejects disabled tapering,
`no_condition`, invalid/zero reference frequency, unsupported models and epoch
realignment. Its common linear operation retains physical mode asymmetry and
NoLoop factorization.

The O4c adaptation targets the maintained release commit
`28b1f71d9d6b7aa863c7808a55b5f2069b2d7a5a`, identical on GitLab
`rapidpe-rift/rift` and GitHub `oshaughn/research-projects-RIT` when checked on
2026-10-18. The two conditioning helpers match the O4d implementation exactly;
O4c's existing tidal parameter constructor and Lmax selection are retained.
O4c has no JAX ILE executable; the public selector/default match O4d's batch and
JAX drivers, without adding a JAX implementation to this port.

At CIT with pySEOBNR 0.3.7, LALSuite 7.26.16, Bilby 2.5.0, NumPy 2.4.6 and
SciPy 1.17.1, the existing cross-code runner passes all four standard cases
(generic spins first) with asymmetric modes retained and cubic NoLoop.
Maximum polarization residual is `1.6502004e-7`; maximum Bilby/NoLoop and
NoLoop/direct differences are `0.04967248` and `0.04973451` nat respectively,
under the unchanged `0.1` nat criterion. Source imports were explicitly checked;
installed distribution metadata is not used to identify the tested source.
