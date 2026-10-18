Full-rate array precompute on rift_O4c
====================================

Benchmark/GPU-validation baseline: oshaughn/rift_O4c at b0fc1469c (PRs 223/225).
Final branch rebased onto eaa19a7f5 (through PR 228); the local combined suite was
rerun there (118 passed, one GPU-only skip). The later SEOB and aligned
XPHM/XO4a/XPNR convention changes do not alter the benchmark's XHM path or the
precessing XPHM regression path. Subsequent eccentric-coordinate and CIP changes
also leave those paths unchanged. The array kernel is identical on both bases.
Inspired by the array operations in junior PR 436; no multirate code is imported.

Use and scope
-------------

Add to the ordinary batch ILE command::

  --precompute-backend cupy --precompute-batch-size 4

The default remains ``lal``. ``numpy`` selects the same full-rate array
arithmetic on the CPU. This choice is independent of the integration backend:
retain the usual ``--vectorized --gpu`` for GPU NoLoop integration. Explicit
``cupy`` requires usable CUDA/CuPy and never silently falls back. Smaller batch
sizes reduce FFT/matrix-row workspaces; the full ordinary-mode matrix remains
on the device, and the later integrator has its own memory requirements.

Waveform generation, frequency grids, mode selection, lag rounding, interpolation
and public return structures are preserved relative to this target. Q uses
full-grid batched inverse FFTs and the existing circular lag window. U and V use
complex128 matrix products and production ``weights2side``; V uses the actual
supplied conjugated waveform modes. Ordinary modes are uploaded once per point
and shared across detectors. Only small Q windows and U/V matrices return to the
host for the existing packing/integration code. Optional effective-mode selection
still uses the existing CPU calculation. There are no JAX or calibration changes.

Direct Python callers can pass ``precompute_backend`` and
``precompute_batch_size`` to ``PrecomputeLikelihoodTerms``. Reuse a
``FullRatePrecompute(backend, batch_size)`` from ``RIFT.likelihood.precompute_array``
as ``precompute_context`` to retain detector products across intrinsic points.
The context retains one configuration; data/PSD contents, metadata, frequency
limits, truncation settings and CUDA device participate in invalidation. Analytic
PSD callables are rebuilt every call because hidden callable state cannot be
fingerprinted. There is no global data/device cache.

Adversarial review
------------------

Four new negative tests failed on the original implementation and passed after
repair:

* A replacement data object with identical contents reused a cache entry holding
  the old object's mutable epoch. Mutating the old object then shifted Q's time
  axis even though the active inputs were unchanged. Cache entries now own an
  epoch snapshot and no longer retain the old data object.
* Changing a callable analytic PSD's internal scale left stale weights under an
  identity-only signature. Analytic PSDs now bypass detector-product cache hits.
* Calling compute after a CUDA device change could use arrays prepared on another
  device. It now requires prepare on the current device; a mocked switch tests
  both rejection and successful re-prepare. Actual multi-GPU switching is not
  claimed from a one-visible-device test run.
* The first ordinary mode's deltaF could differ from data by less than TOL_DF and
  be accepted, while production Q explicitly requires equality. The array path
  now preserves that exact first-mode grid check.

A non-power-of-two FFT/grid test also passed. The default LAL route and target's
new CPU batching, waveform-taper and NoLoop fixes remain intact. The implementation
was rebased before validation; old measurements against 28b1f71d9 are superseded.

Validation and reproduction
---------------------------

The local combined suite passed 118 tests, with one GPU-only skip. It includes
``test_precompute_array.py``, ``test_o4c_critical_ports.py``,
``test_fd_end_taper_cap.py``, ``test_noloop_time_interp.py`` and
``test_noloop_bandlimited_row_chunks.py``. Set ``RIFT_CI_REQUIRE_GPU=1`` to require
both NumPy and CuPy in the new suite, rather than allowing a GPU-only skip.

The independent LAL oracles cover arbitrary complex ordinary/conjugated modes,
distinct H1/L1 data and PSDs, truncation on/off, signed/wrapping lags, batch sizes
1/4/20, cache mutations, generated XHM and precessing XPHM waveforms, interpolants,
SNR, scalar lnL and packed cubic NoLoop likelihoods.

``benchmark.py`` is self-contained given RIFT/LAL/CuPy. It regenerates a
1.4+1.3 solar-mass aligned-spin XHM waveform inside every timed call. H1/L1 use
distinct O4 PSDs and eight-second inverse-spectrum truncation. Durations are
128/256 s, sampling rates 4096/8192 Hz, Lmax 2/4 (four/ten modes), fmin 25/20 Hz,
and fmax 0.45 fs. Each case has one cold and three warm calls. All point timings
include waveform generation, cache validation, transfers and host outputs;
process startup, input construction and subsequent integration are excluded.
CPU BLAS/OpenMP use one thread, with all comparisons on the same shared node.

Four controls are measured: current default LAL, current LAL with
``RIFT_PRECOMPUTE_BATCHED_CROSSTERMS=1``, NumPy and CuPy. The harness overrides
ambient batching settings and checks the batching counter to establish that
only the labelled batched control used it. The first case additionally runs a
real 50-point BNS mass sequence per backend. Its runtime is already initialized,
but each backend starts with a fresh detector context and includes the first
cache miss. Point-by-point Q/U/V parity is checked against the LAL sequence.

``cli_check.py`` runs two intrinsic points per worker through GPU NoLoop/AV,
using supplied frame/PSD/grid fixtures. Sky, orientation and phase are fixed to
the injection; distance is sampled and time marginalized, requiring n_eff >= 50.
This checks a conditional integral, not full posterior convergence.
``seeded_cli.py`` explicitly seeds CuPy because O4c's existing --seed only seeds
NumPy. Saved samples and conditional integrals are compared on identical draws.

The final conditional CLI gate passed for both array backends: 338 and 909
matched saved samples per point had identical saved lnL values at the output's
precision. CuPy's maximum absolute conditional log-integral difference from LAL
was 7.11e-15 (NumPy: 2.14e-14); n_eff was about 125 and 300. Whole-process times
for these short two-point workers were LAL 117.40 s, NumPy 87.76 s and CuPy
91.77 s. Initialization and integration matter for this small workload; use
the actual 50-point sequence below to assess the intended many-point use case.

The GPU-required combined suite passed 153 tests without skips on RTX PRO 4000
Blackwell SFF Edition, using the CUDA 12.8 container and CuPy 14.1.1.

Measured many-point BNS result
-----------------------------

The 50-point, 128 s / 4096 Hz / four-mode sequence regenerated the waveform for
every intrinsic point, varied both component masses, and checked Q/U/V against
the LAL sequence. Total precompute times were:

====================== ============
Backend                Seconds / 50
====================== ============
Current default LAL    142.81
Current batched LAL    140.79
NumPy                   86.63
CuPy                    62.91
====================== ============

CuPy was 2.27x faster than current default LAL, 2.24x faster than its batched
control, and 1.38x faster than the new NumPy path. This is a measured sequence,
not an extrapolation of one warm point. The runtime was already initialized,
with a fresh detector context including its first cache miss; it is not an
end-to-end worker timing. The first GPU precompute earlier in that process cost
16.63 s, so cold library/kernel/FFT initialization should be allowed for when
choosing a backend for very short workers.

Long signals and high sampling rates create large arrays for the inner products
and FFTs. Workers handling many tens of points reuse the same detector products
and amortize initialization, which is the intended use case. Waveform generation
remains on the CPU, and integration/input loading still limit whole-job gains.

Warm single-point medians (seconds, three repeats) on the same GPU node:

======== ======== ===== ======= =========== ======= =======
Duration Rate     Modes LAL     Batched LAL NumPy   CuPy
======== ======== ===== ======= =========== ======= =======
128 s    4096 Hz  4       2.857       2.912   1.744   1.579
128 s    4096 Hz  10      6.594       6.186   3.642   1.983
128 s    8192 Hz  4       5.344       5.169   3.306   2.189
128 s    8192 Hz  10     12.387      12.421   7.461   5.196
256 s    4096 Hz  4       5.282       5.817   3.423   2.057
256 s    4096 Hz  10     13.212      11.809   8.487   5.958
256 s    8192 Hz  4      12.607      12.181   7.474   6.249
256 s    8192 Hz  10     32.540      30.628  19.917  13.047
======== ======== ===== ======= =========== ======= =======

These include CPU waveform generation and show shared-node variability; the
raw repeat and per-point timings are retained in the numeric dataset. At the
largest case, the median CuPy arithmetic/transfer stage was 0.827 s, while CPU
waveform generation dominated the remaining time. CuPy's allocator reservation
there was 1.158 GB; this is not a measurement of total peak device memory.

Across the eight cases and all controls, the maximum scaled Q/U/V discrepancy
against default LAL was 1.25e-14 and the largest absolute scalar lnL discrepancy
was 8.53e-14. The 50-point sequence also checks each point's products against LAL.

``run_gpu.sh``/``gpu.sub`` and ``run_cli.sh``/``cli.sub`` record a CIT execution
recipe; users should select their own workspace/container and fixture paths.
Raw host-specific logs are kept outside Git, following the target branch's
recent evidence cleanup. Numeric results, hardware/software versions and the
kernel fingerprint are retained in ``results-rebased.json`` so the summary
does not depend on a rotating scratch directory. The 24 GB Blackwell test does
not qualify smaller GPUs or Ampere/Pascal. Allocator reservations in the JSON
are not total process peak GPU memory.
