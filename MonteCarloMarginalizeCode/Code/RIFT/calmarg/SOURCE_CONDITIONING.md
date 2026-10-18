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

## Native SEOBNRv5 mode conditioning

For independent comparisons to pySEOBNR's conditioned FD polarizations, pass
`gwsignal_mode_conditioning="native"` in the shared
`extra_waveform_kwargs` supplied to the adapter and likelihood precomputation.
The option supports SEOBNRv5HM and SEOBNRv5PHM with taper enabled and backend
`condition=1`, `conditioning=2`. It extends generation below the requested
template minimum using pySEOBNR's LAL chirp-time bounds, then applies identical
Stage1/Stage2 tapering and filtering to every mode's real and imaginary
components on their common native grid. Stage1 filtering runs on the full
native grid before a single trailing-zero trim determined from all returned
modes, before Lmax or 22-only selection. This avoids mode-dependent lengths
and the native Stage1 routine's unsafe trailing-zero scan on an all-zero mode.
Padding follows conditioning; the
usual cosine mode taper is omitted on this policy. Original source parameters,
the reference frequency, and the physical mode epoch are retained.

Because this operator is linear and projection independent, the same modes
serve the adapter and production NoLoop precomputation. No inclination-specific
polarization taper, fitted timing shift or phase correction is added.
`gwsignal_mode_conditioning="legacy"` remains the default. Native conditioning
refuses unsupported models, a conflicting backend conditioning choice or
requested peak realignment instead of silently selecting a different waveform.
The reference frequency must be explicitly positive, as required by pySEOBNR
0.3.7; zero is not reinterpreted at the lowered generation frequency.

### Selecting the policy in production

Both `integrate_likelihood_extrinsic_batchmode` and
`integrate_likelihood_extrinsic_jax` accept the same
`--gwsignal-mode-conditioning legacy|native` option, defaulting to `legacy`.
This is a wrapper policy, separate from `--internal-waveform-extra-kwargs`,
which supplies backend options when GWSignal is selected. Native requires
`--use-gwsignal`; unsupported models and conditioning combinations are refused.

Asimov can select it without a backend dictionary:

```yaml
waveform:
  approximant: SEOBNRv5PHM
  gwsignal mode conditioning: native
```

Select the existing GWSignal generator route as usual. Omitting the new field
preserves the legacy default and existing generated arguments. The template
carries this one selector through `manual-extra-ile-args`; no additional
pseudo-pipeline configuration surface is introduced. Existing manual ILE args
retain their normal last-value override precedence. Calibration reweighting
receives the same resolved policy as a top-level wrapper control, while backend
options remain in `gwsignal_extra_kwargs`. The option acts in shared waveform
precomputation before NumPy NoLoop or JAX likelihood packing, so neither driver
implements a separate conditioning algorithm.

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
