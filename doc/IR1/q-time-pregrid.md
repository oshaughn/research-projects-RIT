# Opt-in Q-only refinement on rc11

Base: 900181faa36b5e7a134bd71bc99e8dbcd2de00ba (0.0.17.14rc11).
This source patch is a development overlay, not the unmodified rc11 release.

Use --q-time-pregrid-factor 2 with the ordinary vectorized GPU-only NoLoop
likelihood and cubic detector-time interpolation. The default factor is 1.
Supported CLI factors are 1, 2, 4 and 8. Data and waveform generation remain
4096 Hz; geocentric integration nodes and quadrature stay at their previous
spacing. Only the packed Q array is refined. The NoLoop API receives the
separate spacing through q_deltaT; callers must not change P.deltaT to request it.

## Reuse and deliberate differences

This adapts rift_O4d PR261, initially 964cb4e76236dfafaffa115253a6985558e6f3ec,
with transactional-memory fixes through 7a42bf08f7e2f1760777435e08fd84f1ffe5f094.
Pipeline guards and Asimov transport follow 6d409f7f5.
The finite packed Q window is reflected before Fourier interpolation and the
forward interval retained. Its epoch is unchanged, and decimation must reproduce
the original samples. This is not exact reconstruction of the unavailable full
correlation period. Edge behavior depends on the packed window.

The original O4d CLI accepted factors 1 and 8; this port also admits 2 and 4.
A requested refinement failure aborts the driver rather than silently reverting
to the coarse grid. The transactional helper retains its upstream fallback API,
but ILE treats a fallback report as a fatal QTimePregridError.

rc11 interpolation defaults remain nearest with Simpson and cubic with
bandlimited quadrature. No O4d sinc default, explicit-time kernel, quadrature,
calibration or waveform changes are imported. Refinement selects cubic if no
stencil was specified, and rejects an explicit conflicting stencil. O4d-only
rotation/frequency-response/calibration CLI flags do not exist in this rc11
driver; they remain rejected by its parser and by workflow preparation guards.

## Workflow transport

Asimov 0.8 settings:
    sampler:
      ile:
        q time pregrid factor: 2

Pseudo-pipeline option: --internal-ile-q-time-pregrid-factor 2.
The resulting ILE arguments must contain exactly one matching
--q-time-pregrid-factor 2. Workflow construction refuses lost, duplicated or
conflicting settings before writing the executable arguments.

## Validation and cost interpretation

Run the transport and native Q-pregrid regression suites, and the existing
NoLoop interpolation, accumulator and row-chunk suites. Scientific acceptance
requires the frozen three-model GPU-only likelihood matrix and same-worker
1x/2x profiling; the associated review repository holds the receipts.

Retained Q storage is approximately factor times its old size, with endpoint
count ((N-1)*factor+1). Host FFT preparation uses additional temporary memory.
The helper reports input/retained bytes, round-trip error and a conservative
allocation estimate, not a measured process peak. Runtime ILE telemetry also
reports preparation wall time. Waveform generation is not repeated at a higher
sample rate. GPU contraction still evaluates the same number of integration
nodes using a four-tap cubic stencil. Actual evaluation and total costs must
be measured rather than inferred from operation counts.

Ordinary 4096 Hz is the cross-code baseline. RIFT Q-only interpolation is an
intentional code-specific numerical approximation; Bilby waveform generation
is unchanged. This cannot repair waveform/normalization differences and does
not by itself establish posterior JSD agreement. Historical marginalized Q2
diagnostics also refined integration nodes, so they are not clean Q-only
marginalization controls. Native tests must hold those nodes fixed.
