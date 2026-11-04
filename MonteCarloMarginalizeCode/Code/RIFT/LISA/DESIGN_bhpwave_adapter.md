# Restricted BHPWave CPU adapter

Dated interface record, 2026-10-23. Tested code wins on disagreement.
The paper evidence and runtime-lock producer live in
`RIFT_roboto_paper/analyses/paper5_other_waveforms_20261023/`.
This adapter changes no existing FEW, response, likelihood or ILE interface.

`BHPWaveAdapter` returns explicit hp,hc arrays plus cadence, initial epoch,
epoch convention and provenance. It selects only the circular (2,+/-2)
harmonic pair through BHPWave's short KerrCircularWaveform API on one CPU thread.
It does not return orientation-independent RIFT h_lm or LISA channels.

## Parameters and support

Primary and secondary masses are observed solar masses. BHPWave receives those
component inputs directly; secondary mass is not replaced by reduced mass.
Radius r0 is a Boyer-Lindquist coordinate in primary-mass units, distance is
Gpc, and qS/phiS give the SSB sky position. qK/phiK give the direction of the
orbital angular momentum, not the spin axis: BHPWave encodes a retrograde orbit as
a<0 with the same (qK, phiK). For a<0 the spin axis points opposite to (qK, phiK).
Phi_phi0 is the circular orbital phase. Eccentricity, orbital inclination,
secondary spin, and independent radial/polar phases are not accepted fields.

The adapter keeps |a|<=0.995, the documented amplitude range, and checks r0
against the loaded trajectory table. Ordered positive masses are an input
contract, not an adiabatic-accuracy bound. The initial tested configuration is
recorded in the paper evidence; it does not certify an extended physical domain.

Sampling is explicit, limited to 2–262144 points, with named epoch convention.
The returned float64 hp,hc pair occupies at most 4 MiB; backend workspace is
additional. The cap permits the bounded dt5 observation window and its finer
reference grid. Longer-duration coverage and convergence require native checks;
the allocation limit does not establish them.
An optional generation_dt_seconds selects a finer native grid. Its binary64
value must divide the requested output cadence by an exact integer. The adapter
checks the complete native count, then copies the genuine samples at that stride;
it does no interpolation, padding or phase adjustment. Native allocations are
separately limited to 524288 samples (8 MiB for a native hp/hc pair, excluding
backend workspace). The default None keeps direct native sampling unchanged.
Generation cadence, native count and stride are recorded in metadata. Output
carrier Nyquist and trajectory/coverage checks remain required. Choosing a finer
grid does not itself prove convergence or physical waveform accuracy.

The backend uses sidereal years and an inclusive endpoint. The adapter requests
a small floating-point duration margin within the same output bin and refuses
any actual sample-count mismatch. Physical time to merger must exceed the
requested span. Padding is disabled; no trimming or extrapolation repairs output.

The loaded table's maximum orbital frequency gives a conservative m=2 carrier
Nyquist check. It is not a strict signal bandlimit. Amplitude modulation,
response sidebands, omitted harmonics and longer observations require separate
convergence checks. The selected-pair pilot's cadence comparison is not a
universal waveform-accuracy result.

## Explicit runtime

The optional backend is loaded only when constructing an adapter. Importing
this module performs no model import, installation, or download. The caller
places the pinned CPU runtime on PYTHONPATH and supplies a JSON runtime lock.

Schema 1 fixes the source revision, Python ABI, Linux architecture, canonical
published data index, selected native-extension SHA256 and shared-library
provenance. Scientific Python source hashes are pinned by the adapter.
The loader checks all model data and the actual imported extension. It verifies
shared libraries against `/proc/self/maps`, including libraries loaded before
the adapter. A different architecture, ABI, binary, data file, source module or
mapped library requires an explicitly rebuilt and tested runtime lock.
The lock is a provenance selection, not proof of binary-to-source equivalence.
Its producer's source/build record remains part of the acceptance evidence.

The current paper runtime uses a recorded setup.py build-flag change from
`-march=native` to `-march=x86-64`; waveform source is unchanged. Compiled runtimes
remain outside the code and paper PRs. No default installation requirement is added.

## Validation and use

Invariant tests belong in `test_lisa_bhpwave_adapter.py`, included in the LISA
CI gate. Native CPU tests require `RIFT_RUN_BHPWAVE_REAL=1` and
`RIFT_BHPWAVE_RUNTIME_LOCK=/absolute/path/runtime_lock.json`, with the selected
runtime on PYTHONPATH. A skip is not numerical evidence.

A caller may pass returned hp,hc and the explicit sampling/sky fields into a
separately validated response. The adapter returns real arrays (hp, hc).
fastlisaresponse takes `hp+1j*hc` without `flip_hx`; BHPWave's internal complex
form `hp-1j*hc` must not be passed to it. Cross-model SSB basis agreement and response qualification remain
separate gates. This module does not enable BBH factored-phase ILE for EMRIs.

The runtime lock is a provenance check after import: BHPWave's Python modules
run before any hash is compared. It pins the host's glibc and loaded libraries,
so it is valid only on the runtime that wrote it.
