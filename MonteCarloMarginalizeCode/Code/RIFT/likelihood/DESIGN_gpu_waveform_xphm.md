# DESIGN: batched CuPy IMRPhenomXPHM, stages 1-2

Stage 1 module: `gpu_waveform_xphm.py`. Test: `test/waveforms/test_gpu_waveform_xphm.py`.
Benchmark: `test/waveforms/benchmark_gpu_waveform_xphm.py`. Stage 2 (aligned-spin
higher modes) is described in its own section below.

## Scope

Stage 1 ports IMRPhenomXAS (2,2) and the IMRPhenomXP NNLO twisting-up
(PrecVersion 102) from LALSimulation 6.2.0. It returns the five inertial (2,m)
modes for many intrinsic points at once, shape `(n, 5, N)`. The array module is
`cupy` or `numpy`. There is no JAX or Ripple dependency, so the module can be
backported to `rift_O4c`.

Only LAL defaults are implemented:

| Setting | Value |
|---|---|
| InsPhase / IntPhase / RDPhase | 104 / 105 / 105 |
| InsAmp / IntAmp / RDAmp | 103 / 104 / 103 |
| PrecVersion | 102 (NNLO angles, 3PN spinning L) |
| Convention | 1 |
| FinalSpinMod | 4, which LAL maps to 0 (chi_p final spin) for PrecVersion 102 |
| Tides, PNR, non-GR terms | off |

## Structure

The per-point setup runs on the host in NumPy. It covers the fits, the three
collocation solves (batched 4x4 and 5x5), the connection coefficients, the time
shift, and the Euler-angle offsets at f_ref. On a GPU this work was 74 ms of
kernel launches for any n. On the host it takes 4-6 ms. The coefficient arrays
are then copied to the device. The frequency-grid evaluation runs on `xp` with
shapes `(n, n_freq)`.

All three phase and amplitude regions are evaluated on the whole grid and
selected with `where`. This triples the arithmetic but avoids ragged arrays.

## Conventions

- `imrphenomxp_nnlo_modes` equals `SimInspiralChooseFDModes(IMRPhenomXPHM)` with
  PrecVersion 102, co-precessing ModeArray (2,+-2), ModeArrayJframe l=2 and
  `ThresholdMband=0`. These are J-frame modes on LAL's ascending two-sided grid
  `f = (k - n_freq + 1) df`, the ordering `compound_precompute_arrays` expects.
- `imrphenomxp_nnlo_polarizations` equals `SimInspiralChooseFDWaveform(IMRPhenomXP)`
  with PrecVersion 102, including the 2 zeta polarization rotation.
- Masses are in Msun and distance in metres. Spins are in the L0 source frame.
  Points with m1 <= m2 are relabelled as LAL does.
- `jframe_to_rift_lframe` applies the static rotation that `lalsimutils.hlmoft`
  applies with `fd_L_frame=True`. The Euler angles come from
  `lalsimutils.extract_JL_angles`, computed on the host by `rift_jl_angles`.

## Convention gaps

1. Conditioning is not applied. RIFT's ChooseFDModes path applies an FD
   high-pass taper, an inverse FFT, a sign flip, a centring roll, and TD tapers.
   These operations are linear and identical for every mode, so they commute
   with the L-frame rotation. A device version would reuse the helpers in
   `gpu_waveform.py`.
2. RIFT has no ChooseFDModes path for the `IMRPhenomXP` approximant. It builds XP
   modes from polarizations evaluated at five inclinations
   (`hlmoft_IMRPv2_dict`). The production equivalent of these modes is XPHM with
   PrecVersion 102 and only the (2,+-2) co-precessing modes.
3. Default XPHM multibands the (2,2) mode (`ThresholdMband=1e-3`). The reference
   here has multibanding off.
4. `extract_JL_angles` uses RIFT's own PN L. It is not ported to the device.
5. `lalsimutils.rotate_hlm_static` fails for FrequencySeries and ndarray input,
   because it references `lal.Complex16FrequencySeries`. The test wraps arrays
   in TimeSeries.

## Validation (2026-10-18)

50 random points: Mtot 10-150 Msun, q 0.1-1, |chi| <= 0.95 isotropic,
fmin = fref = 20 Hz, df = 1/16 Hz, fmax = 2048 Hz. Mismatch is maximized over
time and phase with the aLIGO zero-detuned high-power PSD.

| Comparison | max mismatch | max relative error |
|---|---|---|
| J-frame (2,m) modes vs ChooseFDModes | 1.2e-15 | 6.4e-12 |
| h+, hx vs ChooseFDWaveform | 7.8e-16 | 6.4e-12 |
| aligned-spin h22 vs SimIMRPhenomXASGenerateFD (20 points) | 8.9e-16 | 2.1e-12 |
| CuPy vs NumPy modes (pcdev11) | | 6.6e-12 of max mode |

A mismatch of 1e-15 is the round-off floor of 1 - match. CuPy and NumPy differ
at 1e-11 because their exp and sin differ at phases of order 1e4 rad.

Throughput, modes only, f_min = 20 Hz, ldas-pcdev11 (RTX PRO 4000 Blackwell;
Xeon E5-2630 v4, one thread):

| Grid | n | CuPy pts/s | NumPy pts/s | LAL pts/s |
|---|---|---|---|---|
| 8 s, 2048 Hz | 1 | 67 | 76 | 66 |
| 8 s, 2048 Hz | 10 | 642 | 98 | 66 |
| 8 s, 2048 Hz | 100 | 5160 | 113 | 66 |
| 32 s, 4096 Hz | 1 | 63 | 13 | 14 |
| 32 s, 4096 Hz | 10 | 607 | 13 | 13 |
| 32 s, 4096 Hz | 100 | 805 | 11 | 12 |

The CuPy floor of about 15 ms per call is the host setup plus kernel launches.

## Stage 2: IMRPhenomXHM aligned higher modes

Module: `gpu_waveform_xhm.py`, with fits generated into `gpu_waveform_xhm_fits.py`
by `test/waveforms/gen_gpu_waveform_xhm_fits.py` from the LAL 6.2.0 C files.
Test: `test/waveforms/test_gpu_waveform_xhm.py`. Benchmark:
`test/waveforms/benchmark_gpu_waveform_xhm.py`.

`imrphenomxhm_modes` returns the modes (2,2), (2,1), (3,3), (3,2) and (4,4) for n
aligned-spin points. It equals `SimInspiralChooseFDModes(IMRPhenomXHM)` with
`PhenomXHMThresholdMband=0`. With `two_sided=True` the output is (n, nmodes, 2N-1) on
LAL's two-sided grid, and m may have either sign. Otherwise it is (n, nmodes, N) and
holds h_{l,-|m|}(f >= 0). The (2,2) mode is the stage-1 XAS evaluation.

Only LAL defaults are implemented:

| Setting | Value |
|---|---|
| PhenomXHMReleaseVersion | 122022 |
| Amplitude: inspiral / intermediate / ringdown | 123 / 211112 (110102 for the 21) / 2 (1 for the 32) |
| Intermediate amplitude points | equispaced |
| Phase: inspiral / intermediate / ringdown | 122019 |
| (3,2) spheroidal ringdown phase | 122019, four collocation points |
| Multibanding, PNR, tides | off |

### Host setup and exactness

LAL builds several collocation values from finite differences. The amplitude uses
fourth-order differences with step 1e-9 at the inspiral boundary. The (3,2) phase
uses a second difference with step 1e-7, in long double. One ulp in an ansatz value
there changes the waveform by about 1e-7. The host setup therefore reproduces LAL's
arithmetic:

- Transcendentals call the C libm (`math`, or `ctypes` for `hypot` and `cbrt`).
  NumPy's SIMD `pow`, `exp` and `log` differ from libm by an ulp on AVX-512 hosts
  such as ldas-grid. CPython's `math.hypot` is not libm's `hypot`.
- C divides a complex number by a real one part by part. NumPy multiplies by the
  reciprocal, so the PN amplitude coefficients use `_cdiv`.
- LAL stores the (3,2) mixing coefficients as `COMPLEX8`. The port rounds them to
  float32.
- Linear systems use the GSL unblocked LU algorithm (`_lu_solve`). The stage-1
  (2,2) setup is called with the same solver.

The grid powers are backend-independent. `Mf^(1/6)` is `M^(1/6) f^(1/6)` and
`log Mf` is `log M + log f`, with the transcendentals of f taken once on the host.
The rest is IEEE multiplication and division, so CuPy and NumPy agree to 1e-13.
A device `pow` alone gave 6e-10 on 256 s BNS grids, where the inspiral phase
reaches 1e6 rad.

### Validation (2026-10-20, ldas-grid, CVMFS IGWN python, numpy backend)

50 random aligned-spin points: 20 BBH (Mtot 10-150, q 0.05-1, |chi| <= 0.95), 10
NSBH and 20 BNS. f_min = f_ref is drawn from 10-20 Hz, the duration from 8-256 s,
and the sample rate is 4096 Hz. Each mode is compared on its own norm, both m signs.
Mismatch is maximized over time and phase with the aLIGO zero-detuned high-power PSD.

| Mode | max mismatch | max relative error |
|---|---|---|
| (2,2) | 6.7e-16 | 1.1e-10 |
| (2,1) | 1.0e-15 | 1.6e-11 |
| (3,3) | 8.9e-16 | 3.3e-10 |
| (4,4) | 1.0e-15 | 7.0e-10 |
| (3,2) | 8.8e-7 | 5.9e-3 |
| CuPy vs NumPy, all modes (pcdev13) | | 1.6e-13 of max mode |

Two more samples run in the same test:

| Sample | strict modes: max mismatch / max rel. error | (3,2): mismatch / rel. error |
|---|---|---|
| 12 points, q 50-200, chi1 of both signs, 16-64 s | 7.8e-16 / 9.3e-10 | 1.3e-5 / 3.0e-2 |
| 6 points, 512-2048 s, f_min 5-12 Hz | 3.1e-15 / 1.6e-9 | 4.2e-15 / 5.4e-7 |

On the long grids the inspiral phase reaches 1e5 rad. The remaining 1e-15 relative
differences in the (2,2) phase coefficients (linear solves, the time shift) then give
phase errors of about 1e-9 rad. The test bounds the relative error there at 1e-8; the
#465 review measured 2.8e-9.

The (3,2) error sits in the intermediate phase. Its coefficients depend on the
second difference above. A 1e-15 change in the (2,2) setup moves them by 1e-4. LAL
shows the same sensitivity: at q = 186 a 1e-13 change of m1 moves LAL's own (3,2) by
mismatch 1.3e-5, equal to the port's error there. The test bounds the (3,2) at
mismatch 5e-5 and relative error 0.1.

At eta < 0.01 and chi1 < 0 LAL multiplies the (3,2) phase cut by (1.2 - 0.25 chi1)
(`IMRPhenomXHM_Intermediate_CollocPtsFreqs`). Stage 2 at bef7889cd missed this; at
q = 199, chi = (-0.68, 0.37) the (3,-2) mismatch was 0.24-0.62. With the branch it
is at most 2e-11, and `test_32_emr_negative_spin_cut` guards it. The eta <= 0.01 inspiral-phase
fits (`Insp_Phase_*_lambda`) add a term linear in f, which the inspiral-intermediate
connection removes again. A change to them does not change the waveform, so no test
can detect it.

Mutations that fail the tests: one (4,4) inspiral-amplitude coefficient changed by
4e-11 (relative error 1.2e-9); the (3,2) cut factor 1.2 changed to 1.19.

Throughput of the five modes h_{l,-m}(f >= 0), f_min = 20 Hz, ldas-pcdev13 (RTX
PRO 4000 Blackwell, CuPy 14.1 in `rift_o4d_cc90-120_cuda128_20260717.sif`; Xeon
E5-2650 v4, one thread). LAL is ChooseFDModes with the five negative-m modes.

| Grid | n | CuPy pts/s | GPU pool | NumPy pts/s | LAL pts/s |
|---|---|---|---|---|---|
| 8 s, 2048 Hz | 1 | 11 | 0.01 GB | 22 | 192 |
| 8 s, 2048 Hz | 10 | 112 | 0.05 GB | 51 | 123 |
| 8 s, 2048 Hz | 100 | 945 | 0.5 GB | 54 | 126 |
| 256 s, 4096 Hz | 1 | 11 | 0.4 GB | 0.9 | 5.2 |
| 256 s, 4096 Hz | 10 | 42 | 2.9 GB | 0.7 | 2.7 |
| 256 s, 4096 Hz | 100 | 46 | 6.6 GB | 0.7 | 2.8 |
| 2048 s, 4096 Hz | 1 | 4.4 | 2.9 GB | 0.1 | 0.6 |
| 2048 s, 4096 Hz | 10 | 5.0 | 6.3 GB | | |
| 2048 s, 4096 Hz | 100 | out of memory | | | |

The CuPy floor is about 85 ms per call: 20-30 ms of host setup and 10-25 ms of
kernel launches per mode. On long grids the evaluation is limited by memory
traffic, about 130 passes over (n, N) arrays per mode. At 2048 s and n = 100 the
output alone is 33 GB. Points are evaluated in chunks of about 4e6 grid values.

### Remaining gaps

1. Precessing XPHM higher modes: twisting-up of the (2,1), (3,3), (3,2) and (4,4)
   with the l = 3, 4 Wigner-d matrices and the mprime-dependent angle offsets.
2. MSA angles (PrecVersion 223) and the SpinTaylor angles (PrecVersion 300, the LAL
   6.2.0 default).
3. RIFT conditioning and the L-frame rotation for l > 2.
4. Kernel fusion. A fused kernel must disable FMA contraction (`--fmad=false`) to
   keep CuPy and NumPy equal.
5. The (3,2) mode matches LAL only to 1e-3 in phase, as stated above.

## Next stages

See "Remaining gaps" in the stage-2 section. Later: device-side
`extract_JL_angles`.

## Licence

The module is a translation of GPL-2.0-or-later LALSuite code, so it carries
that licence. The rest of RIFT is MIT. The maintainer should confirm that this
is acceptable before merge.
