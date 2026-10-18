O4c AV evidence and prior-box corrections
=======================================

Scope
-----

Target: ``oshaughn/rift_O4c`` at ``28b1f71d9``. This is independent
of the critical-performance ports in PR #223.

* O4d #412 (``084a2f138``): retained-point proposal density is the
  draw-count-weighted mixture of the actual proposal grids. The stochastic
  scalar V guides grid sizing; it no longer normalizes the evidence.
* O4d #430 (``82b9ce026``): projected bins retain their multiplicities when
  coordinates are pinned; rejected draws enter the Monte Carlo variance.
  O4c has no bootstrap-interval or warm-start API, so those O4d-specific
  additions are not imported.
* O4d #455 (``a61407ec7``): carry the prior-box support contract into O4c's
  existing sampler and EOS caller. O4c has no distance_slices module to patch.
  Integer bin counts are the default, EOS requests them explicitly, and
  out-of-box draws retain zero weight even with enforce_bounds=False.
  Neither priors nor likelihoods are evaluated at those illegal coordinates.

The existing portfolio draw_simplified density convention, fair-draw cache
aliases, threshold clamp and collapse diagnostics are preserved. The
log_joint_s_prior cache now contains a per-point retained-mixture density;
it need not integrate to one because its divisor is the retained count.
The subsequent evidence mean uses that same retained count. Together these
recover the all-draw balance-heuristic denominator.

Validation (2026-10-18)
-----------------------

Run these six files together with pytest and GW_SURROGATE set to the empty
string, using this checkout's MonteCarloMarginalizeCode/Code on PYTHONPATH:

* test/test_av_retained_density.py
* test/test_av_backport_adversarial.py
* test/test_av_empty_live_volume.py
* test/test_portfolio_compatibility.py
* test/test_portfolio_fairdraw_backend.py
* test/test_portfolio_driver_selection.py

All paths above are relative to MonteCarloMarginalizeCode/Code.

CPU: Python 3.12, NumPy 2.3.5, SciPy 1.15.2: **82 passed, 6 skipped**.
The skips require CuPy/GPU. A100 host pcdev12-alt, Python 3.11, CuPy 13.6.0:
**88 passed, no skips**. The retained-density analytic suite explicitly uses
NumPy on both hosts; the adversarial integration and collapse suites use the
active backend, and the existing portfolio GPU legs run on the A100.

The gate includes narrow analytic Gaussians, zero-prior support, one- and
multiple-cycle rejected-draw variance, overlapping grids, duplicate projected
bins, int64-key overflow fallback, partially and fully pinned coordinates,
non-unit domains, upper-bound pins, nonadaptive sampling, a nonuniform prior,
both likelihood argument conventions, the EOS log wrapper, fair-draw exports,
and existing collapse and portfolio contracts.

Mutation checks: all four deliberately broken implementations failed their
targeted regressions (pytest exit 1, not collection errors):

* restore scalar-V evidence density -> narrow-Gaussian normalization fails;
* replace projected multiplicities by membership -> direct bin-sum oracle fails;
* normalize variance by retained count -> rejected-draw uncertainty fails;
* allow out-of-box prior/likelihood evaluation -> domain assertions fail.

An additional five-dimensional EOS-shaped probe used uniform priors on [-1,1],
two Gaussian axes of width 0.01 at (0.2,-0.3), three flat axes, chunk size
10000, nmax 400000, neff 30 and seeds 11, 12, 13. With fractional grids, the
#412/#430-only O4d reference had mean lnZ error +0.63096 nats and retained
out-of-box fractions 0.467--0.490. This bounded backport gives mean error
-0.00998 nats and zero outside exports. With integer grids the backport
matches the #430 reference errors (-0.03953, +0.04119, -0.06976), also with
zero outside exports. This is an analytic correctness probe, not a production
EOS analysis or a throughput benchmark.

Final adversarial review
------------------------

No unresolved blocking finding after the tests and mutation checks. Reviewed
all-draw counting, proposal snapshots and origin labels through pruning,
pinned-coordinate volume and multiplicity, backend transfers, bounded-domain
zero weights, cached posterior weights, variance normalization, and separation
from the portfolio density interface. The first GPU snapshot omitted the
unchanged ILE driver required by existing portfolio tests; the complete-snapshot
rerun above passed all 88 tests.

Existing diagnostic RuntimeWarnings for negative log-likelihood square roots
and log-zero running moments remain; they do not change the returned results.
AV still has finite-sampling, threshold-truncation and missed-mode limitations.
This correction does not establish multimodal coverage or a universally
calibrated uncertainty interval. Full production PE/EOS runs are not claimed.
