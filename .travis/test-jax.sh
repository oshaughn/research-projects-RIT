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
#                                         against their numpy references; the compound
#                                         algebra is folded into the freqresponse test
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
#   test_angle_marg_smoke.py         12  CHEAP mutation-bearing floor for the whole
#                                         angle-marg feature: scheme selection (a
#                                         previous head could never return 'exact'),
#                                         both dense-sizing levers, required
#                                         amp_sizing, synchronous output-cloud
#                                         recording with training-call exclusion,
#                                         that an amp-sized scheme which recorded
#                                         NOTHING is labelled NOT-PERFORMED and
#                                         never OUTPUT-CLOUD-PASS (the composite
#                                         policy leaves the recorder unwired),
#                                         the driver AST guard on the
#                                         VALUE node (hardcoding angle_marg="grid"
#                                         passes a weaker guard), and that BOTH
#                                         artifacts carry the deterministic checked
#                                         scope.  Seconds, not minutes.
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
#   test_jax_cache.py                30  the shipped ILE selects a stable
#                                         compatibility namespace, Condor uses
#                                         scratch by default, unwritable caches
#                                         fail open, and transferred bundles
#                                         round-trip while rejecting profile,
#                                         runtime, checksum, and archive-member
#                                         mismatches; concurrent manifest
#                                         writers and imported-entry readers
#                                         cannot race; import provenance survives
#                                         later startup; accelerator
#                                         plugin identity is recorded; a device
#                                         probe that raises disables the cache
#                                         instead of escaping driver import;
#                                         and two fresh real JAX processes
#                                         prove an actual persistent-cache
#                                         reuse.
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
#   test_is_proposal_jitter.py        29  issue #227: a Gaussian IS proposal must be
#                                         SCORED under the matrix it was DRAWN from.
#                                         Seven sites drew from cov + 1e-12*I and
#                                         scored under bare cov; the DEFAULT --mode
#                                         laplace-is returned lnZ = 5.85e9 on real O4
#                                         data and exited 0.  Pure numpy synthetic
#                                         likelihood -- no frames, no PSDs, ~8 s.
#                                         14 of the 24 FAIL on the parent commit
#                                         (4c4f6492).  The 10 that pass there are the
#                                         AST detector's own self-test and the nine
#                                         healthy-regime reference comparisons, which
#                                         must pass on BOTH sides by construction --
#                                         they are the gate on the fix (and on the
#                                         collapse guard) not disturbing the regime
#                                         where this estimator actually works.
#   test_jax_phase_marg_mode_order.py 14  phase marginalization must accept EITHER
#                                         packed order of the (2,+-2) pair.
#                                         _accumulate_unit hardcoded column 0 = (2,2)
#                                         and raised NotImplementedError otherwise --
#                                         but the column order comes from a dict's
#                                         iteration order in the precompute, not from
#                                         the caller, so a correctly configured
#                                         --phase-marginalization run died on valid,
#                                         complete data.  U and V carry the mode index
#                                         on BOTH axes, so a half-permutation is a
#                                         silent wrong answer; one test asserts the
#                                         fixture can SEE each single-axis mistake, or
#                                         the equality tests would not gate it.  Two
#                                         tests defend the ordering that already works
#                                         by making _permute_modes fatal: the canonical
#                                         order must take the untouched path, not an
#                                         identity permutation.  Synthetic packed data,
#                                         no frames, no PSDs, ~60 s.
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
#   test_all_axis_peaklocal.py          30  fail-closed four-axis peak-local prototype:
#                                         U,V-guided time ranking and algebraic angular
#                                         starts, JAX refinement, fixed-shape multimode
#                                         quadrature, exact selected-time reconstruction,
#                                         explicit omitted-mass/time-reconstruction
#                                         warrants, geometry/capacity refusal, and outer
#                                         jit/grad/hessian/vmap transform compatibility,
#                                         two-guard primitive support/convergence,
#                                         harmonic-order U,V/Q starts, and the
#                                         empirical enrichment/exact-reserve
#                                         disposition gate.
#   test_multipeak_planner.py          15  opt-in U,V,Q-guided four-axis multi-peak
#                                         planner: exact symmetry expansion, strict
#                                         stationary refinement, two-tier empirical
#                                         convergence, overlap ownership and finite
#                                         reserve, plus FOUR refinement-stall guards:
#                                         the bounded step's ascent contract, the
#                                         step-bound sweep, the max_step check the
#                                         rescale requires, and the symmetry-orbit
#                                         invariant a campaign write-up misread as
#                                         degeneracy.  Three of the four use a
#                                         narrow-time-peak fixture (the ascent contract
#                                         needs no table): the older _synthetic_tables
#                                         puts its maximum ON the targeting lattice, so
#                                         the Newton loop was never exercised and a
#                                         fixed point in it passed this gate for a
#                                         month while declining every row of the
#                                         2026-09-07 ladder campaign.  CPU-only; no
#                                         lal, cupy, or GPU required.  The file defines
#                                         19 tests; the four real-table oracle
#                                         regressions need external validation packets
#                                         that no fixture in this repository provides,
#                                         so they are DESELECTED here -- see
#                                         DESELECTED_TESTS -- and 15 are gated.
#   test_direct_marginalization_policy.py
#                                      18  opt-in cross-axis policy WIRING: choices and
#                                         refusals, measure conversion on both distance
#                                         paths against the exact scheme, decline to a
#                                         warranted band-limited reserve that keeps the
#                                         sample, ledger completeness, wrapper end to
#                                         end on real synthetic tables with a finite
#                                         gradient, and the driver CLI (subprocess).
#   test_jax_q_time_pregrid.py         21  opt-in reflected Q time pregrid on the JAX
#                                         arm: factor-1 bit identity (same array object,
#                                         positions bit-identical to the pre-pregrid
#                                         expressions), the 2n-vs-2(n-1) reflection
#                                         choice measured against an exact-period
#                                         oracle, refined-grid position scaling, the
#                                         fail-closed length/factor checks, 'nearest'
#                                         refusal, wrapper/driver forwarding, and the
#                                         merge interaction with #272's phase-marginalized
#                                         mode permutation.
#                                         Synthetic fixtures; no lal frames, no GPU.
#   test_multipeak_fallback_visibility.py
#                                      13  the multi-peak planner's fallback must not
#                                         read as a policy decline: an exception-driven
#                                         fallback is reported once per call and carries
#                                         decline_kind/fault in the record, a
#                                         budget-driven decline does neither,
#                                         fail_on_fallback is fatal on the first and
#                                         inert on the second, and the default path is
#                                         value- and provenance-identical to the
#                                         pre-change module.  Synthetic tables; the
#                                         tier faults are injected at the
#                                         _run_structural_tier seam.  CPU-only.
#                                         Two of these pin properties that the first
#                                         version of the change got wrong.  The record
#                                         is a 13-element tuple, tested by UNPACKING it
#                                         (13 defaulted-field CONSTRUCTION kept working
#                                         while `a, ..., m = result` had started to
#                                         raise, so a construction test could not see
#                                         it).  And the fault report goes to a logger,
#                                         tested under `-W error::RuntimeWarning`, where
#                                         warnings.warn had made the DEFAULT
#                                         fail_on_fallback=False path raise.
#   test_jax_bandlimited_distmarg.py   20  time_quadrature="bandlimited" on the
#                                         DISTANCE-marginalized wrapper: agreement at
#                                         two amplitudes with an independently
#                                         reconstructed fine-grid reference (plain
#                                         periodic FFT + numpy reduction + numpy
#                                         trapezoid, converged in its own guard and
#                                         factor), the sample-rate ladder closing on
#                                         that value, the reduce-then-refine order
#                                         being a different number, the refusal set
#                                         still refusing, the two fail-closed doors
#                                         (return_lnLt, rotation norms), the single
#                                         definitions of the guard pair and the
#                                         distance reduction, and one subprocess run
#                                         of the driver through --mode flowmc
#                                         --distance-marginalization.  Real
#                                         precompute; needs lal, no GPU.
#   test_distance_gh_nodes_cli.py      27  --distance-gh-nodes: makes the per-sample
#                                         Gauss-Hermite distance quadrature (previously
#                                         reachable only via JAX_ILE_DISTMARG_GH) an ILE
#                                         argument, and the warn-not-silently-ignore
#                                         compatibility notes for --phase-marginalization
#                                         on the four phi_ref-analytic modes,
#                                         --sky-coordinates on the modes that do not
#                                         implement it, and --d-prior (always volumetric).
#                                         Parse-time CLI/env resolution and refusal run the
#                                         real check_critical_and_report in-process (no
#                                         subprocess), including that a CLI/env conflict on
#                                         DIFFERENT nonzero values is REFUSED rather than
#                                         reconciled, and that a refused command line never
#                                         mutates core._DISTMARG_GH_N.  BLOCKER fix (external
#                                         review, same day): the option now defaults to None,
#                                         not 0, so an explicit --distance-gh-nodes 0 is
#                                         distinguishable from not-passed; four tests pin
#                                         this -- explicit 0 against a nonzero env refuses,
#                                         explicit 16 against agreeing env 16 is accepted,
#                                         env 16 alone resolves and is named in the banner,
#                                         and env 16 against CLI 32 refuses (the mutation
#                                         target: a reversed CLI/env priority passes every
#                                         other test in this file, since most cases here
#                                         exercise only one of the two knobs).  A numeric liveness
#                                         check on cheap synthetic packed data (no lal, no
#                                         frames) pins that the resolved count actually
#                                         changes the constructed likelihood's VALUE, not
#                                         just an echoed CLI flag, with a fresh jax.jit
#                                         closure per setting so no stale trace can mask the
#                                         difference; and that 0 reproduces the untouched
#                                         legacy grid bit-for-bit.  Two subprocess checks
#                                         against the real driver entry point (--inj-mode,
#                                         tiny budget, stopped at the same known
#                                         post-construction validation error
#                                         test_distance_grid_loguniform.py's own subprocess
#                                         test relies on) confirm the resolved count reaches
#                                         the run log end to end.  Needs no lal beyond what
#                                         build_likelihood_data already requires; no GPU.

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
  "${JAXDIR}/test_jax_av.py"
  "${JAXDIR}/test_jax_fairdraw_export.py"
  "${JAXDIR}/test_jax_tempering_chooser.py"
  "${JAXDIR}/test_tvals_grid_convention.py"
  "${JAXDIR}/test_interp_choices.py"
  "${JAXDIR}/test_jax_stencil_parity.py"
  "${JAXDIR}/test_flow_reuse_default.py"
  "${JAXDIR}/test_angle_marg_sizing_rule.py"
  "${JAXDIR}/test_anglemarg_buffer_cap.py"
  "${JAXDIR}/test_angle_marg_smoke.py"
  "${JAXDIR}/test_angle_marg_compile_cost.py"
  "${JAXDIR}/test_angle_marg_block_dispatch.py"
  "${JAXDIR}/test_distance_grid_loguniform.py"
  "${JAXDIR}/test_angle_marg_gh_laplace.py"
  "${JAXDIR}/test_angle_marg_default.py"
  "${JAXDIR}/test_angle_marg_gh_selection.py"
  "${JAXDIR}/test_joint_anglemarg_peaklocal.py"
  "${JAXDIR}/test_angle_marg_peaklocal_wiring.py"
  "${JAXDIR}/test_angle_marg_multipeak_wiring.py"
  "${JAXDIR}/test_limit_distance_jax.py"
  "${JAXDIR}/test_direct_marginalization_planner.py"
  "${JAXDIR}/test_time_first_peaklocal.py"
  "${JAXDIR}/test_all_axis_peaklocal.py"
  "${JAXDIR}/test_is_proposal_jitter.py"
  "${JAXDIR}/test_multipeak_planner.py"
  "${JAXDIR}/test_multipeak_fallback_visibility.py"
  "${JAXDIR}/test_jax_phase_marg_mode_order.py"
  "${JAXDIR}/test_jax_q_time_pregrid.py"
  "${JAXDIR}/test_direct_marginalization_policy.py"
  "${JAXDIR}/test_reserve_pair_selection.py"
  "${JAXDIR}/test_angle_marg_laplace_table.py"
  "${JAXDIR}/test_distance_gh_nodes_cli.py"
  "${JAXDIR}/test_jax_cache.py"
  "${JAXDIR}/test_direct_marginalization_policy_cli.py"
  "${JAXDIR}/test_jax_bandlimited_distmarg.py"
  "${JAXDIR}/test_jax_bandlimited_6d_blind.py"
  "${JAXDIR}/test_policy_peaklocal_reserve.py"
)

# EXCLUDED: files in JAXDIR matching test_*.py that are deliberately NOT gated.  The
# manifest check below fails if a file is in neither FILES nor EXCLUDED, so adding a new
# test_*.py to test/jax/ forces a decision instead of being silently unrun -- which is
# this gate's own failure mode, one level up.
DESELECTED_TESTS=(
  # importorskip("flowMC"): flowMC is deliberately not installed in jax-ile-check
  # (ci.yml), and the OUTCOME check below rejects a skip.  The prior-mc and
  # laplace-is driver tests in the same file are the executable coverage that
  # runs here; run the flowMC one by hand where flowMC is installed.
  "${JAXDIR}/test_jax_bandlimited_distmarg.py::test_driver_runs_flowmc_distance_marginalized_bandlimited"
  "${JAXDIR}/test_jax_stencil_parity.py::test_gpu_gather_parity_against_numpy_window"
  "${JAXDIR}/test_multipeak_planner.py::test_hm_second_mode_survives_unsafe_proxy_gap"
  "${JAXDIR}/test_multipeak_planner.py::test_hm_two_tier_integral_matches_overcomplete_oracle"
  "${JAXDIR}/test_multipeak_planner.py::test_real_low_snr_declines_to_finite_reserve"
  "${JAXDIR}/test_multipeak_planner.py::test_real_high_snr_two_tier_path_matches_overcomplete_oracle"
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
#
#   test_multipeak_planner.py::test_hm_second_mode_survives_unsafe_proxy_gap
#   test_multipeak_planner.py::test_hm_two_tier_integral_matches_overcomplete_oracle
#   test_multipeak_planner.py::test_real_low_snr_declines_to_finite_reserve
#   test_multipeak_planner.py::test_real_high_snr_two_tier_path_matches_overcomplete_oracle
#       The four real-table oracle regressions of the multi-peak planner.  Each is
#       skipif-guarded on an external validation packet -- a saved (C_A, C_B) coefficient
#       table from a real analysis -- and NOTHING in this repository or in the CI setup
#       supplies one, so on this runner all four skip.  A skip is precisely what the
#       post-run junit check below refuses, so leaving them selected would redden the
#       gate on every PR while asserting nothing.  They cannot be made to run from a
#       synthetic fixture either: they pin numbers measured on those tables (mode
#       spacings, oracle log-integrals) to ~1e-8, which is a property of the real
#       tables and not of any stand-in this repo could ship.
#       The 15 remaining tests in that file are self-contained and stay gated; they
#       carry the planner's structural coverage (symmetry expansion, strict stationary
#       refinement, two-tier convergence, overlap ownership, reserve fallback).
#       RUN THE FOUR BY HAND, with the packets present, when touching
#       multipeak_planner.py, and record the numbers in the PR per records-protocol.
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
#
# The peak-local ILE WIRING branch adds 13: 11 in
# test_angle_marg_peaklocal_wiring.py (the scheme reaches the likelihood,
# matches exact, is absent from 'auto', joins the amp failsafe / batch-memory
# cap / artifact label, and the CLI rejects a misspelling) and 2 in
# test_joint_anglemarg_peaklocal.py (twice differentiable, and the gradient stays
# finite as the quartic leading coefficient vanishes).  293 + 13 = 306, re-derived
# by RUNNING the gate's own collection after rebasing over #221/#238/#223.
#
# The u-FALLBACK branch adds 2 in test_joint_anglemarg_peaklocal.py (required_u_nodes
# is derived and follows the sqrt-A law under a cap, and a whole-cell integration sized
# by it agrees with a 4x finer one).  The floor is 310, READ FROM THIS JOB'S OWN LOG.
# Two wrong numbers preceded it, failing in opposite directions:
#   308 -- by adding 2 to the previous 306, which is exactly what the paragraph above
#          says not to do.  The base is 309 after #239 merged, so 308 would still have
#          PASSED while silently under-promising three tests.
#   311 -- by running the collection on a dev host.  Wrong by exactly one, because the
#          harness sliced this script by line number to reuse FILES and stopped before
#          the loop that populates DESELECT from DESELECTED_TESTS -- so it counted
#          test_gpu_gather_parity_against_numpy_window, which THIS job deselects.
# Arithmetic lands below the truth and passes; a mis-set-up local collection lands above
# it and fails.  Read the floor off this job's "collected N tests from 27 files" line --
# the only source that is not a guess.
# The production-policy follow-up adds one mutation-bearing streaming test; this job's
# own collection reports 312.
# PR #250 adds test_anglemarg_buffer_cap.py, test_direct_marginalization_planner.py and
# test_time_first_peaklocal.py; 247 adds four tests to test_joint_anglemarg_peaklocal.py
# and REMOVES test_joint_angle_algebraic.py with the duplicate enumerator it covered.
# Neither branch guessed well: 250 derived a provisional 408 from arithmetic and said to
# replace it with a real collection, and 247 measured 329 against a different file set.
# This number is the MERGED collection, run over this job own FILES/DESELECT with the
# DESELECT loop actually applied (the run reports "424/425 tests collected (1 deselected)").
#
# 250 also inferred a standing "this environment collects one more than CI" offset and
# subtracted it.  There is no such offset: the 311 case documented above was wrong by one
# because a harness sliced this script by line number and never ran the DESELECT loop, so
# it counted the one test this job deselects.  That was a one-off setup bug, not a property
# of the environment, and subtracting for it would under-promise by one -- which is the
# failure direction this whole comment exists to warn about, because a low floor PASSES.
#
# The #227 IS-proposal branch then adds the 29 pins in test_is_proposal_jitter.py (27,
# plus the two external review's P1 required: the Markov floor and the inflated-pilot
# regression case).  The FILES array above takes the UNION of every side that has
# touched it.
#
# The phase-marginalization mode-order branch then adds the 14 pins in
# test_jax_phase_marg_mode_order.py.  Its number was NOT derived by adding 14 to the
# constant above -- that shortcut is what the paragraphs below warn about.  It was read
# off this job's own line after rebasing on rift_O4d: "collected 475 tests from 32
# files", with the DESELECT loop applied (the script itself prints it, so there is no
# way to run this and get the half-configured count).
#
# THIS BRANCH HAS NOW HIT THIS CONFLICT THREE TIMES, on three consecutive days, and
# every one of its own numbers was read off a real collection run when written:
#
#     2026-09-05  339   stale within a day        (main had moved 293 -> 312)
#     2026-09-06  451   stale within a day        (main had moved 312 -> 424)
#     2026-09-06  453   stale by the next merge   (main had moved 424 -> 432)
#
# So the constant does not go stale because someone was careless.  It goes stale
# because rift_O4d moves faster than any single branch can hold a global count, and
# that is a property of the counter, not of the people using it.  Note that the
# arithmetic would have been RIGHT all three times (312+27 = 339 after the two review
# tests landed later, 424+27 = 451, 432+29 = 461).  That is exactly what makes it an
# unreliable shortcut rather than a safe one: it is nearly always right, so the once it
# is wrong there is no habit of checking left to catch it.  The number below is READ
# OFF this job's own collection line after the #227 merge: 461/462 collected,
# 1 deselected, 31 files. PR #270 adds 15 multipeak tests but deselects the four
# real-table regressions whose external packets CI does not provide.  The merged
# gate therefore adds 11 self-contained tests.  Confirmed from the merged
# collection: 472/477 collected, 5 deselected, 32 files.
#
# The phase-marginalization mode-order merge added the 14 pins in
# test_jax_phase_marg_mode_order.py on top of #270, and hit the same conflict a
# fourth time: each side of it carried a number the other side had already
# invalidated (475 vs 472).  Read off the job's own line then: 486 from 33 files.
#
# FIFTH time, on the Q time-pregrid branch (this merge).  Both sides were stale
# again -- 480 on the branch, 486 on rift_O4d -- for the same reason, and the
# resolution is again a MEASUREMENT, not the sum.  Read off this job's own line
# after resolving, DESELECT loop applied, on the merged tree:
# "collected 505 tests from 34 files".  The arithmetic (486 + the 19 in
# test_jax_q_time_pregrid.py) agrees, and is again not where the number came
# from.  Independently recollected on citlogin6 with ~/.cache/jaxci_venv
# (jax 0.9.2, numpyro 0.21.0) during the landing review: the same 505 from the
# same 34 files.
#
# 507, not 505, and the gap is two tests added after that measurement: the P2
# review's no-metadata refusal, and one for a path the merge creates that
# neither side covers (#272's phase-marginalized mode permutation acting on a
# REFINED Q grid).  The P2 commit left this constant at 505, which the >= floor
# accepts silently -- exactly the drift this comment exists to stop.  Recollected
# after both: "507/512 tests collected (5 deselected)",
# "collected 507 tests from 34 files".
#
# SIXTH time, on the full-circuit phi-region branch (this merge).  Both sides were
# stale in the usual way -- 487 on the branch, 507 on rift_O4d -- and the resolution
# is again a MEASUREMENT.  This branch adds ONE test,
# test_a_full_circuit_phi_window_is_one_region_at_every_peak_location.  Recollected on
# the merged tree, citlogin6, ~/.cache/jaxci_venv, DESELECT loop applied:
# "collected 508 tests from 34 files".
# SEVENTH time, on the four-axis peak-local branch (#268, this merge).  Both
# sides stale again: 453 on the branch, 507 and then 508 on rift_O4d.
# Resolution is again a measurement on the merged tree with the DESELECT loop
# applied, read off this
# job's own collection line on citlogin6 (~/.cache/jaxci_venv, jax 0.9.2):
# "542/547 tests collected (5 deselected)", gate-style count 542 from 35
# files.  Def-count arithmetic (508 + 30 in test_all_axis_peaklocal.py + 1 in
# test_angle_marg_exact.py + 2 in test_time_first_peaklocal.py = 541) does NOT
# reproduce it, which is one more reason the constant is measured.
#
# EIGHTH: the cross-axis policy wiring adds test_direct_marginalization_policy.py
# (15 tests).  Measured on this tree with the DESELECT loop applied, citlogin6,
# ~/.cache/jaxci_venv (jax 0.9.2): "557/562 tests collected (5 deselected)",
# gate-style count 557 from 36 files.  Independently recollected with the CVMFS
# igwn python on ldas-pcdev11 during the same landing: same 557 from 36 files.
#
# NINTH, on the multi-peak refinement-stall branch (this change).  It adds FOUR
# tests to test_multipeak_planner.py and touches no other test file.  The
# branch measured 546 against a base of 542; #278 has since taken the base to
# 557, so 546 is stale and 557+4 would be the arithmetic this comment forbids.
# Re-measured on the merged tree, DESELECT loop applied, read off the gate's
# own collection line:
# "561/566 tests collected (5 deselected)", gate-style count 561 from 36 files.
#
# TENTH, on the multi-peak fallback-visibility branch (#277, this merge).  It
# adds ONE file, test_multipeak_fallback_visibility.py (13 tests), and touches
# no existing test file.  The branch measured 555 against a base of 542; #278
# and #279 have since taken the base to 561, so 555 is stale and neither side
# nor their sum is usable.  Re-measured on the merged tree, ldas-grid,
# ~/.cache/jaxci_venv (jax 0.9.2, numpyro 0.21.0), DESELECT loop applied, read
# off this job own collection line:
# "574/579 tests collected (5 deselected)", gate-style count 574 from 37
# files.
#
# The policy follow-up PR (this branch, numbered NINTH on its own side before the
# merge) adds three wiring tests (guard preflight,
# operating-point defaults, and a parametrized fail-closed case).  Measured on
# the follow-up tree with the DESELECT loop applied, ldas-grid, CVMFS igwn
# python: "560/565 tests collected (5 deselected)", gate-style count 560 from
# 36 files.
# Both sides stale again after PR #279 (multipeak planner Newton step) landed
# under this branch: re-measured on the merged tree, ldas-grid, CVMFS igwn
# python, DESELECT applied: "564/569 tests collected (5 deselected)", gate-style
# count 564 from 36 files.
#
# ELEVENTH, reconciling the policy follow-ups with #277 (this merge).  Both sides
# were stale in the usual way -- 564 on the branch, 574 on rift_O4d -- and neither
# number nor their difference describes the merged tree, because each side counted
# a file set the other had already changed.  The FILES array is again the UNION,
# now 37 files.  Re-measured on the merged tree with the DESELECT loop applied,
# ldas-grid, CVMFS igwn python
# (/cvmfs/software.igwn.org/conda/envs/igwn/bin/python), read off this job's own
# collection line: "577/582 tests collected (5 deselected)", gate-style count 577
# from 37 files.
#
# TWELFTH, adding test_distance_gh_nodes_cli.py (--distance-gh-nodes, this branch).
# 20 test_* entry points, one parametrized x4, so 23 collected; none deselected.
# FILES is now 38 files, EXPECTED_TESTS raised by exactly that: 577 + 23 = 600.
#
# THIRTEENTH, same branch, same day: adversarial review found a BLOCKER (the
# 0-default made an explicit --distance-gh-nodes 0 indistinguishable from
# not-passed, so a nonzero JAX_ILE_DISTMARG_GH silently won).  Fixed with a
# None default and four new tests pinning CLI-given-including-0 wins, plus a
# mutation-target regression test for the reversed-priority case.  24 test_*
# entry points now, one parametrized x4, so 27 collected; none deselected.
# EXPECTED_TESTS raised by exactly that: 600 + 4 = 604.
#
# FOURTEENTH, on rift_O4d (#285, not this branch): the jax 0.9.2 empty-pool cap fix
# touches only test_anglemarg_buffer_cap.py: removes 1 test
# (test_a_zero_largest_free_block_is_a_known_full_device, whose "0 means full" premise
# was the bug) and adds 6, net +5, none parametrized.  577 + 5 = 582.
#
# FIFTEENTH, same file, the review-MAJOR follow-up (forced probe allocation before
# reading memory_stats(), plus the on-demand-allocator bound): adds 5 tests, none
# parametrized, no removals.  582 + 5 = 587.
#
# SIXTEENTH, reconciling TWELFTH/THIRTEENTH (this branch, test_distance_gh_nodes_cli.py,
# +27 off the 577 base) with FOURTEENTH/FIFTEENTH (rift_O4d #285,
# test_anglemarg_buffer_cap.py, +10 off the same 577 base) at this merge.  The two
# deltas land in disjoint files, so unlike the earlier reconciliations in this history
# the sum is exact, not a guess: 577 + 27 + 10 = 614.  FILES is 38 (37 + this branch's
# one new file; #285 added no file).
# THIRTEENTH, the review-MAJOR follow-up (forced probe allocation before reading
# memory_stats(), plus the on-demand-allocator bound) again touches only
# test_anglemarg_buffer_cap.py: adds 5 tests, none parametrized, no removals.
# 582 + 5 = 587.
# FOURTEENTH, on the JAX persistent/transferable compilation cache (#214, this
# merge).  It adds ONE file, test_jax_cache.py, and changes no test count in an
# existing file: the amplitude failsafe changes HOW it reports -- a returned
# value instead of a host callback, which is what makes the angle-marg graph
# eligible for JAX's persistent cache at all -- but not how many pins cover it.
# The branch opened carrying 189 against a base of 171, both months stale, was
# then measured at 600 against a base of 574, and pushed 603 against a base of
# 577.  rift_O4d has since moved to 587 (#285 and its review follow-up) and
# again with #284, so every number either side carries is stale, for the
# fourteenth time running.  Re-measured on the merged tree, DESELECT loop
# applied, read off this job's own collection line:
#   "621/626 tests collected (5 deselected)", gate-style count 621 from 38
#   files.  Independently recollected on citlogin6 and on ldas-grid, same
#   interpreter, same 621/626.
#
# The +8 over the branch's own 613 are ALL from this landing, not from the
# branch.  Three pin defects found reviewing it (the NOT-PERFORMED label, the
# policy-composite wiring, and a device probe that raised out of driver
# import), and five close mutation survivors: both --jax-cache-dir spellings,
# both cache opt-outs separately, a member declaring zero compressed size, and
# the two refusals that keep an unsized scheme from inventing a metric.
# test_jax_cache.py collects 30, not the 17 the branch's per-file line claimed.
#
# READ THIS BEFORE TREATING A LOCAL RED AS A BRANCH DEFECT.  jax and numpyro
# are installed UNPINNED here (see ci.yml for why), so CI and your shell can be
# on different jax versions at the same time, and which one is newer changes
# over time -- do not infer it from this comment.  Landing #214, two failures
# reproduced in a local venv on the branch AND on its pristine base while CI
# was green on the whole gate: a trend assertion at the noise floor, and a
# full-suite abort (134/139) in a file the branch never touched.  Both were the
# environment, and finding that out cost two runs.
#
# So: check the jax version each side actually ran (the gate prints it as its
# second line), and reproduce any local failure on the PRISTINE BASE in the SAME
# environment before believing it.  The collected COUNT has been stable across
# versions; pass/fail has not.  Issue #292 tracks the environment spread and
# what to do about it.
#
# One practical note for whoever hits this next, because it cost a wasted run:
# PYTHONPATH must be pinned to the tree under test before collecting.  The
# conda environment on the CIT interactive hosts resolves RIFT to a DIFFERENT
# checkout (~/RIFT_ralph), and collection then fails on imports that have
# nothing to do with the branch.
#
#
# FIFTEENTH, the four-axis policy row-batching branch (this merge).  It adds
# SEVEN tests to test_direct_marginalization_policy.py (the batched/sequential
# equivalence test, five parametrized validate_batch_rows cases, and the driver
# knob test) and adds no file.  Its own side measured 581 against a base of 574;
# rift_O4d has since reached 587, so neither number nor their sum describes the
# merged tree.  Re-measured by running THIS script on the merged tree,
# ldas-grid, ~/.cache/jaxci_venv, DESELECT loop applied, read off its own
# collection line: "collected 594 tests from 37 files".
# Both sides were stale in the usual way: 594 on this branch against a base of
# 587, and 621 on rift_O4d, and neither number nor their difference describes
# the merged tree because each counted a file set the other had changed.
# Re-measured on the MERGED tree by running this script and reading its own
# line: "collected 628 tests from 38 files" (ldas-grid, ~/.cache/jaxci_venv,
# DESELECT loop applied).
#
# NEXT, the policy observability branch (this change).  It adds ONE file,
# test_direct_marginalization_policy_cli.py, with twenty tests: seventeen
# driver-seam refusals and three that pin the return arity of
# direct_marginalization_policy_note.  File count 38 -> 39.  Measured on the
# REBASED tree with /scratch/richard.oshaughnessy/envs/jaxci-py311 (python
# 3.11.13, jax 0.10.2) by running the collection and reading its own line, not
# by adding 20 to 628: "648/653 tests collected (5 deselected)".
# FIFTEENTH, the peak-local phi-scan reduction (joint_lnL_phi_dense reduces into its
# lax.scan carry instead of stacking the phi axis; RIFT PR #295).  Touches only
# test_angle_marg_peaklocal_wiring.py: replaces
# test_peak_local_model_includes_streamed_body_and_scan_output (2 params) with
# test_peak_local_model_is_flat_in_n_phi_because_the_scan_reduces (the same 2 params)
# and adds test_peak_local_model_does_not_grow_with_the_phi_axis.  The file-local delta
# is exact at +1.
#
# NOT 628 + 1, for the reason this block has now recorded four times.  Re-measured by
# running THIS script on the merged tree and reading its own collection line:
# "collected 629 tests from 38 files".
# SIXTEENTH, the bandlimited distance-marginalization branch (this merge).  It
# adds ONE file, test_jax_bandlimited_distmarg.py, which collects 20 and has one
# test DESELECTED here (its flowMC driver run would importorskip, and a skip
# fails the OUTCOME check), so +19 over the merged base.  Its own side carried
# 596 against a base of 577; rift_O4d reached 628 meanwhile.  Re-measured by
# running this script on the MERGED tree, ldas-grid, ~/.cache/jaxci_venv,
# DESELECT loop applied, read off its own collection line:
#   "647/653 tests collected (6 deselected)", gate-style count 647 from 39 files
#   (the new file alone: "19/20 tests collected (1 deselected)").
#
# SEVENTEENTH, the fixed-distance blind-draw follow-up (endpoint gap off on the
# 6-D field; parse-time window refusal).  ONE new file,
# test_jax_bandlimited_6d_blind.py, nothing deselected.  Re-measured by running
# this script on this tree, ldas-grid, ~/.cache/jaxci_venv, DESELECT loop
# applied, read off its own collection line:
#   "collected 657 tests from 40 files" (the gate's own line; 647 + 10)
# SEVENTEENTH, merging rift_O4d (#214's test_jax_cache.py, gate count 621 from 38
# files) into this branch (test_distance_gh_nodes_cli.py, +27): the two deltas
# land in disjoint files, so 621 + 27 = 648 from 39 files.  Re-measured on the
# merged tree with the CI-equivalent /scratch jaxci-py311 interpreter before
# this commit.

# EIGHTEENTH, the YOLO integration merge of 2026-09-08 (RIFT PRs #286, #295,
# #297, #298, #299 -- #299 carries #288 -- merged onto rift_O4d after #294).
# Every block above was measured on its own tree, so none of their numbers nor
# their sum describes this one.  Re-measured by running THIS script on the merged
# tree (ldas-grid, ~/.cache/jaxci_venv, jax 0.9.2, DESELECT loop applied) and
# reading its own collection line: "collected 705 tests from 42 files".  This
# assignment is the one that binds; the earlier ones are kept as provenance.
# ONE ASSIGNMENT ONLY.  This file carried FIVE consecutive unconditional
# EXPECTED_TESTS= assignments (648, 629, 657, 648, 705) accumulated by parallel
# merges, separated only by their comment blocks.  Bash keeps the LAST, so the
# four above it were dead while reading as authoritative -- the same shadowing
# that the driver's duplicate add_option produced, one file over.  The comment
# history is kept; the dead assignments are not.  Re-derive by running this
# script and reading its own collection line, never by adding a delta.
# Measured on this branch: 731 collected, 44 files, 6 deselected -- superseded by
# the twentieth block below, which is the one that binds.

# NINETEENTH, RIFT PR #301 (preset local-plan capacities) merged with rift_O4d
# at 43918b22.  #301 adds four tests to
# test/jax/test_direct_marginalization_policy.py, a file already in FILES.
# The eighteenth block measured 705 on the integration tree, so the tempting
# number here is 705 + 4 = 709.  That is wrong: the gate's own line on THIS
# tree reads "collected 712 tests from 42 files" (ldas-grid,
# /scratch/$USER/envs/jaxci-py311, jax 0.10.2, DESELECT loop applied).  The two
# trees are not the same tree, which is the whole reason this file says to
# measure and never to add.
#
# TWENTIETH, this branch (PR #305) rebased onto #301 -- first onto its head
# f30b6a06 while it was open, then onto rift_O4d c0654025 once #301 merged.
# The second rebase was CLEAN and the count did not move; it was re-measured
# anyway, because a clean rebase is when this file's one-assignment property is
# most likely to have been quietly undone.  The nineteenth block and
# the eighteenth-plus-mine block were BOTH left in the file by that rebase, in
# that order, and git merged them without a conflict because they touch
# different lines.  Bash keeps the last, so #301's 712 silently replaced the 731
# this branch had measured -- the same shadowing this file was already collapsed
# once to remove, reintroduced by a clean rebase rather than by an edit.  A
# textual merge cannot see that two assignments to one name are in conflict, so
# ONE ASSIGNMENT ONLY is a property this file has to be re-checked for after
# every merge, not one it keeps on its own.
# Re-measured by running this script on the rebased tree (ldas-grid,
# /scratch/$USER/envs/jaxci-py311, jax 0.10.2, PYTHONPATH pinned to THIS
# checkout, DESELECT loop applied): "collected 733 tests from 45 files", and
# 737 after the four reserve-roster/refusal tests added later in the branch, and
# 738 once the wrapper's source-text gate test became two behaviour tests, and
# 740 with the two reserve-resolution tests.
# This assignment is the one that binds.
# Plus the peak-local time reserve branch (PR #304): test_policy_peaklocal_reserve.py,
# one file.  Re-measured by running this script on the rebased tree, ldas-grid,
# ~/.cache/jaxci_venv, DESELECT loop applied, read off its own collection line:
# "754/760 tests collected (6 deselected)", gate-style count 754 from 46 files
# (2026-09-09).  This assignment is the one that binds.
#
# SIXTEENTH, --angle-marg-scheme multipeak (the four-axis controller wired into the
# driver).  Adds ONE file, test_angle_marg_multipeak_wiring.py, 7 tests, none
# parametrized, no removals, and touches no existing test count.  754 + 7 = 761,
# measured by running this script.
# (superseded assignment removed 2026-09-09; see the single EXPECTED_TESTS= below)
# 2026-09-09 (laplace reserve kernel keyword fix): +1 test in test_policy_peaklocal_reserve.py
# (the resolved kernel through anglemarg's REAL Laplace function).  Read off this script's
# own collection line on ldas-pcdev12 (~/.cache/jaxci_venv, CPU): "collected 755 tests from 45 files".
# (superseded assignment removed 2026-09-09; see the single EXPECTED_TESTS= below)

# Simultaneous rotation + finite response adds cheap analytic coefficient parity
# to an existing collected test.  Real waveform precompute, JIT/grad, the one-call
# wrapper, and scaling profiles remain explicit manual checks in the same files:
# they are too expensive for the already runner-limited per-PR JAX gate.

# 2026-09-09, #313 rebased over #312 (laplace keyword fix, 755) and the base's
# test_limit_distance_jax tightening: neither 761 + 1 nor 755 + 7 is the number.
# Re-measured by running THIS script on the merged tree (ldas-grid, ~/.cache/jaxci_venv,
# jax 0.9.2, CPU, DESELECT applied): "collected 762 tests from 46 files".  Binding.
# 2026-09-09 (locator search sizing, stacked on #312 + #313): +1 test in
# test_policy_peaklocal_reserve.py (the sized locator on a rho-632 carrier).  Read off this
# script's own collection line on ldas-pcdev12 (~/.cache/jaxci_venv, CPU) after rebasing on
# rift_O4d 336f86133: "collected 763 tests from 46 files".  The #312/#313 merges had left
# three EXPECTED_TESTS= assignments (761, 755, 762; last wins); this is the single one.
# 2026-09-10: +22 value-only AV/portfolio, prior-window, wrapper, and driver
# contract tests in test_jax_av.py.
EXPECTED_TESTS=785

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

# SHARDING.  Only the EXECUTE step is split.  Everything above -- the FILES
# manifest, the EXCLUDED accounting, the collection floor against
# EXPECTED_TESTS, and the deselect-resolution check -- runs in full in every
# shard, so no shard can pass on a partial view of the suite and the counts
# stay one number rather than N.
#
# Why: the suite outgrew the 60-minute job cap.  Measured 2026-09-08, the base
# suite ran 3277 s of pytest against a ~3518 s budget, and one new test in #290
# added ~850 s, so every run was ~609 s over and jax-ile-check was cancelled at
# 1h00m17s with the suite at 91% and ZERO failures.  Trimming was costed at
# ~529 s across five changes and does not clear the overrun on its own.  Three
# shards leave each well inside the cap with room for the next test.
#
# Round-robin by index, not a hand-tuned split: a cost table in this file would
# go stale exactly the way the EXPECTED_TESTS comments above record every other
# hardcoded number going stale.
JAX_GATE_SHARDS="${JAX_GATE_SHARDS:-1}"
JAX_GATE_SHARD="${JAX_GATE_SHARD:-1}"
if ! [ "${JAX_GATE_SHARDS}" -ge 1 ] 2>/dev/null || ! [ "${JAX_GATE_SHARD}" -ge 1 ] 2>/dev/null \
   || [ "${JAX_GATE_SHARD}" -gt "${JAX_GATE_SHARDS}" ]; then
  echo "test-jax.sh: bad shard ${JAX_GATE_SHARD}/${JAX_GATE_SHARDS}" >&2; exit 1
fi
if [ "${JAX_GATE_SHARDS}" -gt 1 ]; then
  SHARD_FILES=()
  for i in "${!FILES[@]}"; do
    if [ $(( i % JAX_GATE_SHARDS )) -eq $(( JAX_GATE_SHARD - 1 )) ]; then
      SHARD_FILES+=( "${FILES[$i]}" )
    fi
  done
  if [ "${#SHARD_FILES[@]}" -eq 0 ]; then
    echo "test-jax.sh: shard ${JAX_GATE_SHARD}/${JAX_GATE_SHARDS} got 0 files" >&2
    exit 1
  fi
  echo "== running shard ${JAX_GATE_SHARD}/${JAX_GATE_SHARDS}: ${#SHARD_FILES[@]} of ${#FILES[@]} files =="
else
  SHARD_FILES=( "${FILES[@]}" )
  echo "== running =="
fi
# The OUTCOME floor below must be the count THIS invocation was asked to run.  With
# JAX_GATE_SHARDS>1 that is the shard's own collection, not EXPECTED_TESTS: the
# whole-suite floor has already been asserted above on the full FILES list, and
# holding one shard to it fails every shard that passes ("ran 257 tests, expected
# at least 705", 2026-09-08, first run of the three-way split).  Collect the shard's
# files the same way, so a shard still cannot go green on a partial view of ITS files.
if [ "${JAX_GATE_SHARDS}" -gt 1 ]; then
  shard_collect="$("${PYTHON_BIN}" -m pytest --collect-only -q -p no:cacheprovider "${DESELECT[@]}" "${SHARD_FILES[@]}" 2>&1)" || {
    printf '%s\n' "${shard_collect}"; echo "test-jax.sh: shard collection failed" >&2; exit 1; }
  RUN_EXPECTED="$(printf '%s\n' "${shard_collect}" | grep -cE '^[^[:space:]]+\.py::')"
  echo "shard ${JAX_GATE_SHARD}/${JAX_GATE_SHARDS} collects ${RUN_EXPECTED} tests from ${#SHARD_FILES[@]} files"
  if [ "${RUN_EXPECTED}" -lt 1 ]; then echo "test-jax.sh: shard collected 0 tests" >&2; exit 1; fi
else
  RUN_EXPECTED="${EXPECTED_TESTS}"
fi
"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider --durations=0 --junit-xml="${junit}" "${DESELECT[@]}" "${SHARD_FILES[@]}"
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
"${PYTHON_BIN}" - "${junit}" "${RUN_EXPECTED}" <<'PYCHECK'
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

echo "jax_ile CPU regression gate: PASS (${n_collected} tests collected; this invocation ran ${RUN_EXPECTED})"
