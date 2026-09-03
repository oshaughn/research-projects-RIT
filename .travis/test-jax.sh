#!/usr/bin/env bash
# CPU regression gate for the JAX extrinsic likelihood (RIFT/likelihood/jax_ile),
# driven from test/jax/.
#
# WHY THIS SCRIPT EXISTS AT ALL, AND WHY IT COUNTS TESTS
# -----------------------------------------------------
# Until this gate landed, NOTHING in .github/workflows/ci.yml ran test/jax/ -- the
# workflow had zero matches for "jax".  Two real defects survived a month each behind
# that gap (see the PR that adds this file).
#
# The obvious repair -- point pytest at test/jax/ -- would have manufactured MORE
# confidence than it earned.  Several files in that directory are scripts with an
# `if __name__ == "__main__":` block and NO `test_*` function.  Pointing pytest at such
# a file collects ZERO items and exits 5, "no tests ran", which reads as a pass in a
# skim of the log.  So this script does two things a bare pytest invocation does not:
#
#   1. It asserts a FLOOR on the number of collected tests before running anything.
#      If a future refactor drops a `test_*` entry point, renames a file, or moves it,
#      collection silently shrinks and this job goes RED instead of green-on-nothing.
#      The floor is pinned to the exact count as of this commit; raise it when you add
#      tests, and never lower it without saying why in the commit message.
#   2. It fails on ANY nonzero pytest exit, which includes exit 5.
#
# JAX_PLATFORMS=cpu is set: no GPU is required, and jax must not go hunting for one.
set -uo pipefail
# NOTE: deliberately no -e.  Every command below has its rc handled explicitly so the
# failure messages stay specific; if you add a command, guard it yourself.

# JAXDIR below is repo-relative, so anchor cwd rather than trusting the caller.
cd "$(dirname "$0")/.." || { echo "test-jax.sh: cannot cd to repo root" >&2; exit 1; }

PYTHON_BIN="${RIFT_JAX_PYTHON:-${PYTHON:-python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
fi

# Guard the tool checks: a missing interpreter plus a redirected stderr is
# indistinguishable from a clean result.
"${PYTHON_BIN}" -c 'import pytest' || { echo "test-jax.sh: pytest unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import jax, jaxlib; print("jax", jax.__version__)' \
  || { echo "test-jax.sh: jax unavailable" >&2; exit 1; }
"${PYTHON_BIN}" -c 'import numpyro; print("numpyro", numpyro.__version__)' \
  || { echo "test-jax.sh: numpyro unavailable (needed by test_nuts_phimarg)" >&2; exit 1; }

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

JAXDIR="MonteCarloMarginalizeCode/Code/test/jax"

# Included files, with the count each contributes as of this commit:
#   test_jax_likelihood.py             3  synthetic packed data: nearest-vs-NoLoop, AD
#                                         vs finite differences, jit/vmap
#   test_jax_endtoend.py               1  full precompute -> pack -> JAX vs the numpy
#                                         NoLoop on a real injection (fixed by #144)
#   test_jax_slowrot_coeffs.py         2  rotation + freqresponse response coefficients
#                                         against their numpy references
#   test_jax_slowrot_wrapper.py        1  the one-call build_*_data_from_precompute path
#   test_jax_slowrot.py                3  rotation Path A (p_max=0), Path B (p_max=1)
#                                         and freqresponse: NoLoop parity + AD/jit/
#                                         vmap/hessian
#   test_jax_slowrot_cauchy_schwarz.py 2  the rotation lnL VALUE (bound + explicit
#                                         time-domain model), Path A and Path B.
#                                         Agreement with the NoLoop is necessary but
#                                         not sufficient -- see that file's docstring
#   test_network_coords.py             1  network-frame sky fold on a real injection
#   test_nuts_phimarg.py               1  fisher_nuts_sample_phimarg vs an analytic 4-D
#                                         target (needs numpyro; no lal)
#   test_jax_fairdraw_export.py       34  the --save-samples export contract of
#                                         bin/integrate_likelihood_extrinsic_jax:
#                                         that it is a FAIR DRAW (reweighted against
#                                         the sampler's own importance weights, then
#                                         multinomial-resampled as ILE does), that
#                                         ILE's 1.5*ESS cap binds, that the count
#                                         options act exactly where the driver reports
#                                         them implemented and nowhere else, that the
#                                         export RNG is never the science generator,
#                                         that a fair draw which CANNOT be performed
#                                         exports nothing at all (and clears a stale
#                                         file at that path) instead of shipping the
#                                         raw cloud, that a refused export leaves no
#                                         `_.dat` result row for the event either, that
#                                         an SMC ladder which stops short of inv_T=1
#                                         publishes neither artifact (and that the
#                                         sampler reports the exponent it reached), and
#                                         that the provenance header
#                                         describes the file it sits on.  Needs no lal or GPU: the
#                                         driver is imported by path and driven on an
#                                         analytic 4-D target with known moments.
#                                         Several of these are AST guards on the
#                                         DRIVER SOURCE (the F1 post_weight gate, the
#                                         write_samples call site, the export-before-
#                                         result write order) because the defects
#                                         they pin live at call sites, where a
#                                         helper-level assertion cannot see them.
#   test_jax_tempering_chooser.py     45  the --adapt-weight-exponent chooser and the
#                                         tempering-cost law
#                                         ESS/N = [beta(2-beta)]^(dim/2) it rests on.
#                                         Pins the law against the EXACT sweep measured
#                                         on the real BNS likelihood (both directions:
#                                         a law that under-predicts the cost would
#                                         silently under-budget a run), that it takes
#                                         no SNR argument -- the non-JAX helper's rule
#                                         keys on SNR and does not transfer -- that the
#                                         chooser RETURNS the exponent rather than
#                                         writing it back to opts (writing it made
#                                         event 1 of a batch read event 0's choice),
#                                         and that the degenerate-export branch WARNS
#                                         (it used to raise; the calibrated estimate
#                                         cannot support a hard floor -- see
#                                         DESIGN_jax_tempering.md 4a).  Includes a
#                                         RETIRED-claim
#                                         guard asserting the SNR rule has not crept
#                                         back into the driver.  Needs no lal, no GPU
#                                         and no flowMC.
#   test_jax_stencil_parity.py        23  #193: the JAX 'sinc' gatherer is the SAME
#                                         stencil as the numpy/cupy/CUDA paths.  Those
#                                         three share one weight array and cannot drift;
#                                         JAX re-expresses the formula independently
#                                         (its weights depend on the traced sub-sample
#                                         offset), so this is what converts that
#                                         duplication from "trust the reviewer" into
#                                         "CI fails".  Landed WITHOUT a manifest entry,
#                                         which made rift_O4d fail its own manifest
#                                         check; added here.  24 tests, of which the
#                                         cupy leg is deselected on this CPU runner --
#                                         see DESELECTED_TESTS -- so 23 are gated.
#   test_flow_reuse_default.py         7  flow re-use is OFF by default, and --flow-reuse
#                                         still reaches the old behaviour.  A store_true
#                                         flag cannot express its own negation, so simply
#                                         flipping default=True would have made
#                                         --no-flow-reuse inert AND deleted the capability;
#                                         both directions and last-one-wins are pinned, as
#                                         is the batch loop still reading the flag.
#   test_interp_choices.py             3  #190: --interp cubic is reachable from the
#                                         CLI and selects _gather_cubic.  Merged in
#                                         from rift_O4d, which added it to FILES
#                                         without a count in this ledger; recorded
#                                         here so the block stays a complete
#                                         accounting of EXPECTED_TESTS.
#   test_tvals_grid_convention.py     13  issue #146: the time-marginalization window
#                                         grid the JAX wrapper and
#                                         bin/integrate_likelihood_extrinsic_batchmode
#                                         build, extracted BY AST FROM THE DRIVER
#                                         SOURCES and compared by value at srate
#                                         1024/2048/4096/8192/16384.  Needs no jax; it
#                                         lives here because it pins the jax_ile
#                                         wrapper against the production driver, and
#                                         because 16384 is the rate test_jax_endtoend
#                                         (4096) structurally cannot cover.
#   test_angle_marg_smoke.py          8  CHEAP mutation-bearing floor for the whole
#                                         angle-marg feature: scheme selection (a
#                                         previous head could never return 'exact'),
#                                         both dense-sizing levers, required
#                                         amp_sizing, the host failsafe record and
#                                         its cond-guard, the driver AST guard on the
#                                         VALUE node (hardcoding angle_marg="grid"
#                                         passes a weaker guard), and that BOTH
#                                         artifacts are labelled and never imply
#                                         verification.  Seconds, not minutes.
#   test_angle_marg_compile_cost.py   6  the laplace path's COMPILE- and RUN-cost
#                                         structure (2026-08-28: an unrolled kernel
#                                         x 64 distance blocks put a production
#                                         SNR-40 run >88 min / 22 GiB into XLA
#                                         compilation; the fix then exposed a
#                                         36.41 GiB RESOURCE_EXHAUSTED at the
#                                         default eval chunk).  Trace-only where
#                                         possible: the traced graph must not grow
#                                         with the distance grid, the kernel must
#                                         stay rolled (equation-count ceiling), the
#                                         distance tail padding must be exactly-
#                                         zero-weight, and the anglemarg eval-chunk
#                                         cap must stay WIRED in samplers and the
#                                         driver.  Each fails under a verified
#                                         mutation (see the PR).  Seconds.
#   test_angle_marg_block_dispatch.py 4  the laplace path's EXECUTION-cost
#                                         structure (2026-08-28: with compilation
#                                         fixed, the kernel executed ~2,950x the
#                                         grid scheme because BOTH blend branches
#                                         ran at every lattice point, the 320-pt
#                                         quadrature everywhere included the 99.5%
#                                         of points needing N ~ 32-96).  Pins the
#                                         lax.switch block dispatch: the N ladder
#                                         keeps the shipped aliasing exponent, the
#                                         dispatcher matches the undispatched
#                                         kernel in every branch, and the fused
#                                         driver actually CALLS the dispatcher
#                                         (wiring).  Each fails under a verified
#                                         mutation (see the PR).  Seconds.
#   test_distance_grid_loguniform.py 33  the OPT-IN log-uniform ("peak-resolving")
#                                         distance quadrature for the dense
#                                         angle-marg schemes.  Pins the spacing
#                                         contract (Delta ln d <= c/rho_max), the
#                                         TWO-SIDED calibration of c against the
#                                         Gaussian trapezoid error law it is
#                                         derived from (a one-sided check is
#                                         satisfied by c -> 0, which is accurate
#                                         and arbitrarily expensive), that
#                                         --distance-grid-scheme still DEFAULTS to
#                                         the historical uniform grid node for
#                                         node, and -- the safety property -- that
#                                         the dense angle lattice is sized from the
#                                         amplitude on the FULL prior support, so
#                                         no distance grid can shrink it.  Includes
#                                         driver AST guards on the option VALUE
#                                         node, on the forwarded (not hardcoded)
#                                         keyword, and on the fail-closed refusal
#                                         when the flag is set on a mode that does
#                                         not implement it.  One numerical
#                                         execution test against a 1024-node
#                                         uniform reference; the rest are numpy or
#                                         AST.  ~19 s.  Each fails under a verified
#                                         mutation (matrix in the PR).
#   test_angle_marg_sizing_rule.py    1  the m_max-aware dense phi sizing rule.
#
#   test_angle_marg_gh_laplace.py    15  the psi-marginal distance-node placement
#                                        that lets 'laplace' honour
#                                        JAX_ILE_DISTMARG_GH.  GATED despite
#                                        costing ~3.5 min: it is a NEW numerical
#                                        path, and every constant in it is
#                                        pinned by MUTATION (collapse the
#                                        half-span, drop the node floor, force
#                                        the sigma cap) rather than by a
#                                        pass-through assertion.  The two
#                                        agreement legs (converged uniform grid,
#                                        exact scheme under the same quadrature)
#                                        are the expensive ones; they are also
#                                        the only ones that would catch a wiring
#                                        error, so they stay.
#                                         Pure numpy, milliseconds, closed-form I0
#                                         reference.  FAILS under the old m_max-blind
#                                         rule (0.498 nats vs 1.17e-10), which every
#                                         low-scale brute-force test passes -- so this
#                                         is the only gated check that distinguishes
#                                         the corrected sizing.  The rest of the
#                                         angle-marg suite is EXCLUDED; see below.
#   test_limit_distance_jax.py        21  --limit-distance on this arm: the distance
#                                         QUADRATURE narrows while the prior keeps its
#                                         [d_min,d_max] normalization.  Includes the
#                                         bitwise no-op of the default call (both the
#                                         uniform and the adaptive grid), the ACCEPTANCE
#                                         comparison (narrowed vs full-range lnZ at equal
#                                         n_grid: 0.0 nats, measured 2.8e-14), and its
#                                         power check -- the pre-change call signature on
#                                         the same box moves lnZ by +4.16 nats.  ~110 s,
#                                         CPU, one synthetic precompute.

#   test_nuts_phimarg_injection.py  Not a pytest file at all: it runs the whole study at
#                                 module scope and calls sys.exit() there.  WITHOUT numpyro
#                                 that surfaces as a fast COLLECTION ERROR; WITH numpyro --
#                                 which THIS JOB INSTALLS -- `--collect-only` actually
#                                 EXECUTES the study and hangs (reproduced: no output after
#                                 ~6 min).  So re-adding it would burn to timeout-minutes,
#                                 not fail fast.  It
#                                 is also long -- a full NUTS run on a real injection
#                                 that has exceeded a 1800 s cap in hand testing.  Too
#                                 expensive for every PR; run it by hand.
#
#   test_flow_reuse.py            Collects 0 (pytest exit 5); passes as a script.
#                                 Excluded on DEPENDENCY risk, not runtime: three flowMC
#                                 runs, and flowMC is an extra heavy dependency with a
#                                 fast-moving sampler API that this test tracks closely,
#                                 so an unpinned flowMC release would redden the gate
#                                 for reasons unrelated to RIFT.  Reasonable to add
#                                 later behind a PINNED flowMC.  Run it by hand when
#                                 touching samplers.flowmc_sample.
#
#   demo_*.py, debug_*.py,        Demos, debugging scripts and a figure generator, not
#   benchmark_snr_sequence.py,    assertions.  None defines a test_* function and none
#   make_3g_figdata.py            is intended as a gate.
#   test_jax_time_quadrature.py      12  band-limited time marginalization.  The
#                                         stock path integrates exp(lnL_t) with fixed
#                                         Simpson weights at the DATA spacing while the
#                                         integrand width sigma_t = 1/(2 pi rho sigma_f)
#                                         SHRINKS with SNR -- 61.2 us against grid
#                                         spacings of 244/122/61 us at srate
#                                         4096/8192/16384 on a 35+30 HLV injection at
#                                         rho=40.  Simpson is not a safeguard: it is
#                                         (4 T_h - T_2h)/3, so it carries the coarser
#                                         T_2h alias and is WORSE than trapezoid when
#                                         under-resolved.  Pins that upsampling is EXACT
#                                         (sampling theorem, not an approximation), that
#                                         the Nyquist bin is split rather than dumped,
#                                         that the band-limited result is grid-phase
#                                         INDEPENDENT where stock Simpson swings 4.26
#                                         nats, convergence in the free upsample factor,
#                                         and that an unknown time_quad RAISES instead of
#                                         silently giving the old behaviour.  Also pins
#                                         the two ways the reconstruction can silently
#                                         change the ANSWER rather than the resolution:
#                                         that it integrates the ORIGINAL (n-1)*deltaT
#                                         window and not the periodic FFT continuation
#                                         past the last sample (a constant integrand
#                                         makes that a pure normalization shift), and
#                                         that it REFUSES data whose <h|h> depends on
#                                         arrival time (the slow-rotation post-phase),
#                                         where holding the norm at one bin would be a
#                                         different likelihood.  And the GUARD
#                                         SAMPLES: the window is a CROP, so its
#                                         ends do not join, and the FFT's
#                                         periodic seam rings into the inserted
#                                         samples while every retained sample
#                                         stays exact -- invisible to an
#                                         exactness test built from whole-period
#                                         modes.  Pins the defect on a
#                                         non-periodic cropped tone, its removal
#                                         by guard samples, that the guard is
#                                         support and never enters the integral,
#                                         that it has no default, and that the
#                                         band-limited path actually widens the
#                                         accumulation window (while Simpson
#                                         does not).  Pure numpy
#                                         and jax, no lal, no GPU.
#   test_jax_terminal_time_marginalization.py
#                                      18  adaptive primitive-field integration:
#                                         odd/even reflection, exact normalization,
#                                         Event-B high-SNR convergence, AD, bounded
#                                         batch-independent dispatch, a near-Nyquist
#                                         phase-marginalization counterexample, explicit
#                                         nonlinear-endpoint refusal, driver wiring, and
#                                         honest phase-marginalized sky/psi export,
#                                         K=14/K=88 independent guarded references,
#                                         and executable baseline/banded support refusal.

FILES=(
  "${JAXDIR}/test_jax_time_quadrature.py"
  "${JAXDIR}/test_jax_terminal_time_marginalization.py"
  "${JAXDIR}/test_jax_likelihood.py"
  "${JAXDIR}/test_jax_endtoend.py"
  "${JAXDIR}/test_jax_slowrot_coeffs.py"
  "${JAXDIR}/test_jax_slowrot_wrapper.py"
  "${JAXDIR}/test_jax_slowrot.py"
  "${JAXDIR}/test_jax_slowrot_cauchy_schwarz.py"
  "${JAXDIR}/test_network_coords.py"
  "${JAXDIR}/test_nuts_phimarg.py"
  "${JAXDIR}/test_jax_fairdraw_export.py"
  "${JAXDIR}/test_jax_tempering_chooser.py"
  "${JAXDIR}/test_tvals_grid_convention.py"
  "${JAXDIR}/test_interp_choices.py"
  "${JAXDIR}/test_jax_stencil_parity.py"
  "${JAXDIR}/test_flow_reuse_default.py"
  "${JAXDIR}/test_angle_marg_sizing_rule.py"
  "${JAXDIR}/test_angle_marg_smoke.py"
  "${JAXDIR}/test_angle_marg_compile_cost.py"
  "${JAXDIR}/test_angle_marg_block_dispatch.py"
  "${JAXDIR}/test_distance_grid_loguniform.py"
  "${JAXDIR}/test_angle_marg_gh_laplace.py"
  "${JAXDIR}/test_angle_marg_default.py"
  "${JAXDIR}/test_angle_marg_gh_selection.py"
  "${JAXDIR}/test_joint_anglemarg_peaklocal.py"
  "${JAXDIR}/test_limit_distance_jax.py"
)

# EXCLUDED: files in JAXDIR matching test_*.py that are deliberately NOT gated.  The
# manifest check below fails if a file is in neither FILES nor EXCLUDED, so adding a new
# test_*.py to test/jax/ forces a decision instead of being silently unrun -- which is
# this gate's own failure mode, one level up.
DESELECTED_TESTS=(
  "${JAXDIR}/test_jax_stencil_parity.py::test_gpu_gather_parity_against_numpy_window"
)
EXCLUDED=(
  # test_angle_marg_exact.py -- the angle-marginalization VALIDATION suite.
  #
  # NOT gated per-PR, and this is a deliberate, measured decision rather than a
  # convenience.  It is a development check in the same sense that full RIFT
  # analysis runs are: it establishes the schemes' ERROR LAW at production
  # amplitude, a property of the mathematics that does not change commit to
  # commit.  Three separate CI failures forced the split, each a different
  # symptom of the same cost: the 169-test gate was CANCELLED at the job's
  # 60-minute cap; a later head OOM-killed the runner at 19 min; and the run
  # after that reached 83% and then died with "the runner has received a
  # shutdown signal" (exit 143).  The 139-test baseline ran in 13m53s.
  #
  # What remains GATED is the coverage that actually bites:
  # test_angle_marg_sizing_rule.py pins the m_max-aware dense sizing with a
  # pure-numpy, millisecond test against a closed-form I0 reference, and FAILS
  # under the old m_max-blind rule (0.498 nats vs 1.17e-10).  The low-scale
  # brute-force comparisons in the excluded file prove exactness but do NOT
  # distinguish the sizing rule -- the broken rule passes them all -- which is
  # why extracting that one test was necessary before excluding the rest.
  #
  # RUN IT BY HAND when touching anglemarg.py, on a quiet host with >=16 cores:
  #   PYTHONPATH=<tree>/MonteCarloMarginalizeCode/Code JAX_PLATFORMS=cpu \
  #   JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 JAX_COMPILATION_CACHE_DIR="" \
  #   taskset -c 0-15 python -m pytest -q \
  #     <tree>/MonteCarloMarginalizeCode/Code/test/jax/test_angle_marg_exact.py
  # and record the numbers in the PR, per records-protocol.
  "${JAXDIR}/test_angle_marg_exact.py"
  "${JAXDIR}/test_nuts_phimarg_injection.py"
  "${JAXDIR}/test_flow_reuse.py"
)

# DESELECT: individual tests inside a GATED file that cannot run on this CPU runner.
# File-level EXCLUDED is too blunt for these -- dropping test_jax_stencil_parity.py to
# silence its one GPU leg would also drop the 23 CPU tests that are the whole point of
# #193.  Deselecting is not the same as tolerating a skip: a skip leaves the gate green
# while asserting nothing, whereas a deselected test is accounted for HERE, in writing.
#
#   test_jax_stencil_parity.py::test_gpu_gather_parity_against_numpy_window
#       The cupy leg of the sinc-stencil parity check.  It needs a real CUDA device;
#       this job has none, so it self-skips.  It is a genuine gate on a GPU host --
#       run it by hand there when touching Q_inner_product_sinc_cupy.
DESELECT=()
for t in "${DESELECTED_TESTS[@]}"; do DESELECT+=( --deselect "$t" ); done

echo "== manifest check (every test_*.py is gated or explicitly excluded) =="
manifest_rc=0
for f in "${JAXDIR}"/test_*.py; do
  known=0
  for g in "${FILES[@]}" "${EXCLUDED[@]}"; do
    [ "${f}" = "${g}" ] && { known=1; break; }
  done
  if [ "${known}" -eq 0 ]; then
    echo "test-jax.sh: ${f} is neither gated nor explicitly excluded." >&2
    manifest_rc=1
  fi
done
if [ "${manifest_rc}" -ne 0 ]; then
  echo "  Add it to FILES (and raise EXPECTED_TESTS), or to EXCLUDED with a reason." >&2
  exit 1
fi

# Sum of the per-file counts above.
# Pinned deliberately: a bare `pytest test/jax/`
# that collected 0 would exit 5, and a partial loss (say 14 -> 3) would still exit 0.
# NOTE: this environment collects ONE MORE test than the CI runner does (local
# 147 vs CI 146; the delta was 3 earlier in this branch's life).  So "recount by
# collection" must mean collection IN THE GATE'S ENVIRONMENT -- a local count has
# tripped this floor twice.  When in doubt, take the number from a CI log line
# ("collected N tests from M files") rather than from your shell.
# Raised 153 -> 155 by the two new test_jax_time_quadrature.py pins (original
# integration window; refusal of arrival-time-dependent norms), then 155 -> 160 by
# the five guard-sample pins in the same file (periodic-seam defect on a
# non-periodic crop; its removal by guard samples; guard is support, not window;
# no default guard; the band-limited path widens the accumulation window).
# PR #209 then adds six test_angle_marg_compile_cost.py pins, raising 160 -> 166,
# and PR #210 adds five test_angle_marg_block_dispatch.py pins, raising 166 -> 171.
# PR #216 adds eighteen adaptive primitive-time pins, raising 171 -> 189.
# The psi-marginal GH placement (#225) adds 36, raising 189 -> 225: 15 in
# test_angle_marg_gh_laplace.py, 5 in test_angle_marg_default.py, 8 in
# test_angle_marg_gh_selection.py, plus 4 answering external review on the
# identity gate (imaginary-A0 coefficient, B1 in the conjugate slice, the gate
# applying to an explicit laplace, the kernel guard staying trace-safe).
# The log-uniform distance quadrature adds 37 on top of the base, raising
# it by that amount wherever the base then sat: 30 for the scheme itself, 3 from external re-review (the
# zero-clipped-amplitude extreme of the F1 detector, the DRIVER half of the F2
# refusal, and a guard on the sky-doubling path -- each because a mutation
# SURVIVED the 33-mutation matrix without it), and 3 covering the truncated-
# endpoint precondition added by the automated review pass, which shipped with
# none (the estimator against an independent numpy measurement, the
# interior-but-too-close refusal, and the non-positive-clearance window that
# built at 6.7x tol).
# Raising the floor
# by exactly the number of tests ADDED is safe whatever the environment delta above,
# since it preserves the margin the previous floor already had.
# Raised 189 -> 217 by the 28 tests added in the angle-marg GH branch (#225); then
# 217 -> 225 by the tests answering external review on its identity gate; then
# 225 -> 234 while this branch was open, by the peak-local framework (#224), the
# joint (phi,psi) peak-local kernel (#230) and the AV batch-max change (#234);
# then 234 -> 272 by the adaptive distance quadrature branch's own 37 (#221).
# Raised 272 -> 293 on merging --limit-distance, by that branch's 21
# test_limit_distance_jax.py pins: 15 shipped; 4 added after an adversarial
# mutation sweep found two INERT guards (the adaptive grid's d_prior_range branch
# could be deleted with the suite green, worth +5.5 nats silently, and
# sample_prior could draw over the full range while the box correction stayed);
# and 2 more for this merge's own interactions with the #221 distance quadrature
# -- the box must not shrink the angle lattice, and the log-uniform grid must
# refuse a box rather than renormalize the prior onto it.
# THREE branches have now raised this constant, so it is the single place this
# merge is most likely to go quietly wrong; the FILES array above is the other.
# Taken from a collection RUN, never by adding the three accountings.
EXPECTED_TESTS=293

echo "== collection floor check (expect >= ${EXPECTED_TESTS} tests) =="
collect_out="$("${PYTHON_BIN}" -m pytest --collect-only -q -p no:cacheprovider "${DESELECT[@]}" "${FILES[@]}" 2>&1)"
collect_rc=$?
if [ "${collect_rc}" -ne 0 ]; then
  printf '%s\n' "${collect_out}"
  echo "test-jax.sh: pytest collection failed (exit ${collect_rc})" >&2
  exit 1
fi
# Anchor to '<path>.py::' at line start.  An unanchored grep -c '::' also counts merged
# stderr (jax/XLA log lines, C++ symbols, '::1'), and because the floor is a >= test,
# OVER-counting is the dangerous direction: one stray line masks exactly one lost test.
n_collected="$(printf '%s\n' "${collect_out}" | grep -cE '^[^[:space:]]+\.py::')"
echo "collected ${n_collected} tests from ${#FILES[@]} files"
if [ "${n_collected}" -lt "${EXPECTED_TESTS}" ]; then
  printf '%s\n' "${collect_out}"
  echo "test-jax.sh: collected ${n_collected} tests, expected at least ${EXPECTED_TESTS}." >&2
  echo "  A file was renamed/moved, or a test_* entry point was dropped and pytest is" >&2
  echo "  now passing on fewer tests than this gate promises.  Fix the file, or update" >&2
  echo "  EXPECTED_TESTS in this script and say why." >&2
  exit 1
fi

# A --deselect whose nodeid does not resolve is SILENTLY IGNORED by pytest: rename the
# test, or fat-finger the path, and the deselect quietly stops applying while this script
# still claims the test is accounted for.  The skip would then come back and the count
# would be off by one, which is exactly the confusion the deselect was added to end.  So
# verify both halves: the test still EXISTS under that name, and it is actually GONE from
# the collection.
for t in "${DESELECTED_TESTS[@]}"; do
  f="${t%%::*}"; nm="${t##*::}"
  if [ ! -f "${f}" ]; then
    echo "test-jax.sh: DESELECTED_TESTS names ${f}, which does not exist." >&2; exit 1
  fi
  if ! grep -qE "^[[:space:]]*def ${nm}\\(" "${f}"; then
    echo "test-jax.sh: DESELECTED_TESTS names ${nm}, which ${f} no longer defines." >&2
    echo "  It was probably renamed.  Update the nodeid, or drop it from DESELECTED_TESTS." >&2
    exit 1
  fi
  if printf '%s\n' "${collect_out}" | grep -qE "^${f}::${nm}(\\[|$)"; then
    echo "test-jax.sh: --deselect did not take effect for ${t}." >&2; exit 1
  fi
done

junit="$(mktemp -t jaxci-junit-XXXXXX.xml)"
trap 'rm -f "${junit}"' EXIT

echo "== running =="
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider --durations=0 --junit-xml="${junit}" "${DESELECT[@]}" "${FILES[@]}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  # rc 5 == "no tests ran"; it is a FAILURE here, not a pass.
  echo "test-jax.sh: pytest exited ${rc}" >&2
  exit "${rc}"
fi

# OUTCOME check.  The floor above counts COLLECTION, which cannot see a test that
# collects, runs, and asserts nothing: one pytest.skip() or importorskip() disables a
# gate while both the collected count and the pytest exit status stay green.  That is
# the very shape this script exists to prevent, so assert what the RUN did.
"${PYTHON_BIN}" - "${junit}" "${EXPECTED_TESTS}" <<'PYCHECK'
import sys, xml.etree.ElementTree as ET
path, expected = sys.argv[1], int(sys.argv[2])
root = ET.parse(path).getroot()
ts = root if root.tag == "testsuite" else root.find("testsuite")
if ts is None:
    sys.stderr.write("test-jax.sh: no <testsuite> in the junit report\n"); sys.exit(1)
g = lambda k: int(ts.get(k, 0) or 0)
tests, skipped, failures, errors = g("tests"), g("skipped"), g("failures"), g("errors")
print("junit: tests=%d skipped=%d failures=%d errors=%d" % (tests, skipped, failures, errors))
bad = []
if tests < expected:
    bad.append("ran %d tests, expected at least %d" % (tests, expected))
if skipped:
    bad.append("%d SKIPPED -- a skip silently disables a gate here; if a skip is "
               "legitimate, exclude the file in FILES -- or, for a single test, add it to "
               "DESELECTED_TESTS -- and say why" % skipped)
if failures or errors:
    bad.append("%d failures, %d errors" % (failures, errors))
if bad:
    sys.stderr.write("test-jax.sh: " + "; ".join(bad) + "\n"); sys.exit(1)
PYCHECK
if [ $? -ne 0 ]; then exit 1; fi

echo "jax_ile CPU regression gate: PASS (${n_collected} tests)"
