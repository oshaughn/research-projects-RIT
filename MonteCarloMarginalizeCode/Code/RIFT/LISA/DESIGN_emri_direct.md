# Restricted EMRI direct evaluation

This record defines the initial direct evaluation interface. Tested code wins
on disagreement. Measured evidence belongs in the Paper 5 analysis record:
`RIFT_roboto_paper/analyses/paper5_emri_few_20261022/`.

`util_EMRI_direct_likelihood.py` performs a fixed-nuisance self-injection.
It generates FEW SSB polarizations, applies the pinned time-domain LISA response,
and evaluates direct A/E/T residual inner products. No parameters are marginalized.
The existing LISA ILE path is unchanged. Its BBH factorization is never selected.

## Input contract

The JSON fixture has exactly six fields:

| Field | Meaning |
|-------|---------|
| `schema_version` | Integer 1 |
| `waveform` | `FEWConfig` fields, including sampling, tolerance and epoch |
| `response` | `ResponseConfig` fields, including the local pinned orbit path |
| `injection` | Complete explicit `EMRIParameters` fields |
| `trials` | Objects with unique string `id` and parameter `changes` |
| `noise` | Explicit synthetic white noise settings |

Waveform, response, and parameter classes reject unsupported settings.
The waveform epoch reference must be `pinned equal-arm orbit t=0`.
The response requires its zero input epoch and discards declared padding.
Every trial must produce the injection's retained sampling, epoch and shape.
The interface caps waveform generation at `MAX_WAVEFORM_SAMPLES` (32768) and defaults
to eight trial points. The command permits at most 64 points.

Noise fields are `kind`, `psd_levels`, `target_snr`, `fmin_hz`, and `fmax_hz`.
The only supported kind is `synthetic_independent_white`.
Three positive PSD levels define their relative channel weights.
A positive `target_snr` sets their common scale once from the injection.
Set `target_snr` to JSON `null` to use the supplied absolute PSD levels.
For convergence runs, copy the canonical output PSD and use this fixed mode.
That PSD remains fixed for every trial. The covariance is diagonal by this
synthetic experiment's definition; no physical LISA noise model is asserted.

Continuous Fourier amplitudes use `dt*rfft`. Frequency cuts select interior
positive bins, excluding DC and the even-length Nyquist endpoint.
The rectangular window and identical retained segment apply to every trial.
The likelihood omits a fixed noise normalization constant.

## Execution and evidence

```sh
util_EMRI_direct_likelihood.py --fixture fixture.json --output result.json
```

Run inside a bounded allocation with prefetched, checksum-verified model data
and orbit files. The output path must be new. Failed commands retain stderr;
an external bounded runner records failure or timeout status.

The result records the input fixture/hash, module hashes, dependency versions,
waveform/response metadata, channel hashes, fixed PSD, selected band, and timings.
It also records injection, null, and trial log likelihoods.
`COMPLETED` means the requested evaluation finished. Scientific acceptance
requires independent convention, response, and numerical convergence checks.

For a distance-only trial, linear response predicts
`lnL = -0.5 * rho^2 * (1 - d_injection/d_trial)^2`.
This provides an analytic check independent of the likelihood implementation.
The Paper 5 runtime fixture registers that identity before execution.
Mass perturbations require direct numerical evaluation; no BBH mass or phase
approximation is substituted.

## Separately declared periodic Hann observable (schema 2)

Schema 1 retains its rectangular analysis and field set. Schema 2 adds exactly
`window: {"kind": "periodic_hann"}` and requires
`noise.kind="synthetic_independent_white_after_window"`, fixed absolute positive
PSDs, and `target_snr=null`. On each identical retained half-open segment of N
samples, the analysis multiplies every A/E/T injection and trial by
`w[j]=0.5*(1-cos(2*pi*j/N))` before the existing `dt*rfft` calculation. This is
the periodic Hann, with no fitted window parameters or normalization change.

`apply_analysis_window` records the weights digest, raw and windowed channel
digests, retained start, duration, and count. Likelihood channel hashes describe
the windowed observable. Noise is an artificial independent Fourier noise model
**after** the declared window; this does not model the covariance induced by
windowing a physical stationary time series. The CLI sample cap and
waveform/response guards remain in force. This extension supplies an explicitly
separate experiment; it supplies no numerical convergence evidence by itself.
