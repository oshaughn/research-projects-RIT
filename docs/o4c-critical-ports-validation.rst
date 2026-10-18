Critical O4d ports to O4c: validation
==================================

Target baseline: oshaughn/rift_O4c, 28b1f71d9 (0.0.17.14rc7).
Source PRs in oshaughnessy-junior/research-projects-RIT:
#385, #439, #433, #311, #255 and #256.

Changes
-------

* #385: assign RF nonfinite/oversized queries the -500 floor in CIP and
  EOSPosterior. Also handle all-invalid and empty batches without calling
  sklearn.predict on an empty selection.
* #439: cap the ChooseFDModes end taper at one second. Start tapering is
  unchanged; fd_end_taper_max_time=None restores the uncapped end taper.
* #433: key native SphHarmTimeSeries mode dictionaries by (l,m).
* #311: opt-in chunked Gram products for cross-term precompute, plus the
  bit-preserving setup/inner-product optimizations. Batching remains OFF
  unless RIFT_PRECOMPUTE_BATCHED_CROSSTERMS=1 or batched=True is passed.
* #255/#256: share source geometry/harmonics across detectors, cache detector
  constants, accumulate time-independent norms as vectors, reuse data-term
  buffers, and integrate Simpson weights with a cached blocked construction.

O4c's existing bandlimited quadrature, continuous export, row chunking,
nearest/cubic interpolation, and external calibration interfaces are retained.
The O4d fused in-loop calibration/JAX/response machinery is not imported.

Adversarial findings fixed during the port
----------------------------------------

* Upstream's identity-only mode-key cache returned stale harmonics after a
  caller mutated an existing lookup array. O4c reads mode values per call,
  before launching device geometry work, rather than retaining that cache.
* Detector/Simpson array caches now distinguish CUDA devices, and quadrature
  keys retain the callable rather than a recyclable function id.
* A custom likelihood callback could write to the former dense norm array.
  Passing a stride-zero broadcast would break that API and race on CUDA.
  Custom callbacks receive writable dense storage; the built-in helper uses
  the compact view. Bandlimited rejection of a time-dependent norm is preserved.
* Batched same_waveform_Q now aligns B by A's keys, preserving the legacy
  behavior when dictionary insertion orders differ.
* ip_matrix validates frequency spacing and positive integer chunk sizes,
  and handles empty mode lists without stacking an empty sequence.

Test results
------------

1. Dedicated port gate on local NumPy/SciPy/LALSuite:
   83 passed, one CUDA-only test skipped.

2. Same gate on pcdev12-alt, NVIDIA A100-SXM4-80GB, IGWN Python 3.11,
   CuPy 13.6.0, CUDA_VISIBLE_DEVICES=0, RIFT_CI_REQUIRE_GPU=1:
   84 passed, zero skipped, 102.99 seconds.

   The GPU test compares the actual frozen pre-port O4c NoLoop function
   against the port. Nearest/cubic and phase/non-phase time series must be
   bit-identical. Simpson and bandlimited integrals agree within 3e-10 nats
   on the synthetic cases. GPU custom writable callbacks and odd/even
   vendored Simpson rules, including 614/2457 time points, are exercised.

3. Existing O4c time quadrature/wiring/interpolation/chunking/per-row-offset
   regressions: 135 passed. This includes seeded continuous-time export.

4. Calibration conditioning/epoch and waveform alignment tests in the full
   worktree with Bilby and writable Numba caches: 45 passed, five failed.
   All five failures are DAG-builder cases where Calib_reweight.sub is not
   generated. The same no_L_frame case also fails on untouched baseline
   28b1f71d9, before the ports. The numerical waveform/conditioning tests pass.
   This pre-existing pipeline-test failure is not hidden or fixed in this bundle.

   Initial runs with the default Python lacked Bilby; another run lacked a
   writable Numba cache. Those environment failures are not counted as port
   regressions. An earlier exported source-only tree omitted calibration
   fixture files, so the full-worktree results above supersede its skips.

5. Deliberate corruption in disposable copies, with the tests required to fail:

   * RF floor sign reversal: detected (two failed regression cases).
   * Removed per-detector phase-array copy: detected (ten failed cases).
   * Removed Gram-product conjugation: detected (two failed cases).
   * Removed the end-taper cap: detected (two failed cases).

6. Additional physical-mode integration check: generated IMRPhenomXHM FD modes,
   built U/V banks with both scalar and batched precompute, and compared the
   downstream cubic NoLoop likelihood within 1e-10 nats. This is in the gate.

CI and reproduction
-------------------

Run from the repository root, with the intended environment's python on PATH::

    export GW_SURROGATE=''
    bash .travis/test-o4c-critical-ports.sh

The gate is registered in existing GitLab and Travis unit-test entry points.
It checks a collection/pass floor and allows only the explicit optional GPU
skip. Set RIFT_CI_REQUIRE_GPU=1 to make unavailable CUDA a failure.

The frozen reference test helper records target 28b1f71d9 and preserves the
pre-port function body. Geometry regressions additionally compare against
frozen pre-split formulas, rather than comparing wrappers to themselves.

Evidence summaries and logs accompany this document under
docs/o4c-critical-ports-evidence/. GPU scratch directory:
/scratch/richard.oshaughnessy/rift-o4c-ports-20261018.y3XQrL on pcdev12-alt.

Performance interpretation
--------------------------

GPU numerical correctness is verified here. Historical O4d timing records are
included in the two DESIGN documents; they are not new O4c worker benchmarks.
Whole-event speedup depends on waveform cost, precompute/integration balance,
detector/mode counts, interpolation, and GPU. Batched cross terms change
floating-point reduction order and skip exactly zero-weight frequencies;
nonfinite waveform values outside that support have different propagation.
No production analysis or release deployment is performed by this port.
