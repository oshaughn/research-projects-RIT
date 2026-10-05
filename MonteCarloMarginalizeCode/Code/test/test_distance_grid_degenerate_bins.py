#!/usr/bin/env python
"""
Regression tests for the .dgrid exporter's degenerate bins
(RIFT/misc/distance_grid.py, bin/integrate_likelihood_extrinsic_batchmode).

Background (the defect these tests lock down).  With --sampler-method GMM on the
ILE-GPU-Paper demo, `--export-marginal-distance-grid` wrote a table that is not a
function of distance -- two rows per bin centre, ~40 nats apart:

    dist [74.84 74.84 94.26 94.26]
    lnL  [103.99 65.68 65.21 103.53]

The same command line on AV wrote 5 distinct, monotonically increasing distances with
smooth lnL.  Reproduced byte-identically on an unmodified junior/rift_O4d, so it predates
the gate widening in PR #167 -- that widening only makes more configurations reach it.

  1. THE BLOCK SPLIT CUT A RUN OF IDENTICAL DISTANCES.  integrate_log's fair draw
     resamples _rvs WITH REPLACEMENT at n_extr = min(n_extr, 1.5*eff_samp, 1.5*neff), so a
     starved pass (GMM n_eff 2.9 -> 4 rows) exports a few distinct distances each repeated
     several times.  _weighted_blocks then ran np.array_split over the RAW sample indices,
     which puts the two copies of one distance in different blocks.  Both blocks take the
     weighted mean of the same value, so both report the same centre.

  2. THE WIDTH FLOOR TURNED THAT INTO A 36-NAT LIE.  Bin edges are midpoints of the
     centres, so a repeated centre gives a zero-width bin, and `np.maximum(width, eps)`
     floored it at 2.2e-16.  lnL carries -log(width), so those bins were reported
     -log(eps) = 36.04 nats brighter than their neighbours -- which is the 38-39 nat
     first-and-last-row spread above, not a physical feature.

  3. NOTHING SAID THE GRID WAS UNRESOLVABLE.  n_grid rows were emitted whatever the
     samples could support; a file with two distinct distances in it is indistinguishable
     from a converged curve once it leaves the job.

The requirement is not "runs without crashing": a table sold as likelihood-vs-distance
must be single-valued in d, its widths must come from the sample spacing rather than from
a floor, and a sample set that cannot resolve a curve must be refused or flagged -- while
a healthy all-distinct sample set must be binned exactly as before.
"""

import ast
import os

import numpy as np
import pytest

from RIFT.misc.distance_grid import (
    MIN_UNIQUE_DISTANCES,
    build_distance_grid,
    distance_grid_resolution_warning,
    reconstruct_marginal_lnL,
)

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')

EPS = np.finfo(float).eps


def _volumetric_log_prior(d):
    return 2.0 * np.log(np.asarray(d, dtype=float))


def _fair_draw(uniq, counts, rng=None):
    """An _rvs column as the fair draw leaves it: each surviving distance repeated.

    Returns (distance, ln_prior_d, ln_weights) with the weights flat in d, so ANY
    structure in the exported lnL is an artifact of the binning and nothing else.
    """
    uniq = np.asarray(uniq, dtype=float)
    counts = np.asarray(counts, dtype=int)
    distance = np.repeat(uniq, counts)
    if rng is not None:
        distance = rng.permutation(distance)          # the draw is not sorted
    ln_pi = _volumetric_log_prior(distance)
    return distance, ln_pi, np.zeros(len(distance))


###
### 1. the exported table must be a function of distance
###

def test_repeated_fair_draw_distances_do_not_share_a_bin_centre():
    """The reported symptom, minimised: 4 rows, 2 distinct distances."""
    distance, ln_pi, ln_w = _fair_draw([74.84, 94.26], [2, 2])

    grid = build_distance_grid(distance, ln_w, 64.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=4)

    assert len(np.unique(grid["dist"])) == len(grid["dist"]), \
        "two rows share a bin centre: the grid is not a function of d"
    assert np.all(np.diff(grid["dist"]) > 0), \
        "bin centres are not strictly increasing"


def test_the_split_is_pinned_where_the_row_cap_cannot_rescue_it():
    """The four-sample symptom above does NOT test the blocking.

    With 4 samples over 2 distinct distances the row cap alone forces n_grid = 2, and
    raw-index splitting happens to land on the run boundary -- so restoring the exact
    defective line leaves the two tests named after the symptom green.  Found by mutating
    `blocks = np.array_split(np.arange(len(distance)), n_grid)` back in.

    Here one distance is repeated 20 times among 23 samples and 4 bins are asked for.
    Raw-index blocks are 6/6/6/5 rows, so THREE of the four boundaries fall inside that
    one run and three bins report the same centre.  Only blocking on distinct values can
    give four distinct centres.
    """
    uniq = [110.0, 240.0, 380.0, 505.0]
    distance, ln_pi, ln_w = _fair_draw(uniq, [1, 20, 1, 1])

    grid = build_distance_grid(distance, ln_w, 7.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=4)

    assert len(grid) == 4
    assert np.allclose(grid["dist"], uniq), \
        "block boundaries fell inside the repeated distance: centres {}".format(grid["dist"])
    assert np.all(np.diff(grid["dist"]) > 0)
    assert np.all(grid["dist_weight"] > 1e6 * EPS)


def test_a_bin_centre_is_the_weighted_mean_of_its_block():
    """Pinned against the closed form.  Nothing else here would notice a centre computed
    as the block midpoint instead (found by mutating it to 0.5*(min+max))."""
    distance = np.array([100.0, 400.0])
    ln_w = np.array([np.log(3.0), np.log(1.0)])
    ln_pi = np.zeros(2)

    grid = build_distance_grid(distance, ln_w, 0.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=1)

    assert len(grid) == 1
    assert np.isclose(grid["dist"][0], (3.0 * 100.0 + 1.0 * 400.0) / 4.0), \
        "centre {} is not the weighted mean (midpoint would be 250)".format(grid["dist"][0])


def test_the_bin_prior_is_the_log_of_the_weighted_MEAN_prior():
    """4a: pinned against the closed form.  The two tests that read this column are blind to
    it -- one recomputes exp(lnL + ln_prior_d_sampling + log dist_weight), in which it
    cancels exactly, and the other is an invariance the wrong rule also satisfies.  An
    arithmetic mean in log space survives both, and differs by the Jensen gap."""
    distance = np.array([100.0, 400.0])
    ln_pi = np.array([np.log(2.0), np.log(50.0)])        # DIFFERENT per row, or no gap
    ln_w = np.array([np.log(3.0), np.log(1.0)])

    grid = build_distance_grid(distance, ln_w, 0.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=1)

    w = np.array([3.0, 1.0]) / 4.0
    want = np.log(np.sum(np.exp(ln_pi) * w) / np.sum(w))
    assert np.isclose(grid["ln_prior_d_sampling"][0], want), \
        "bin prior {} is not log E_w[pi_d] = {} (arithmetic mean in log space gives {})".format(
            grid["ln_prior_d_sampling"][0], want, float(np.sum(ln_pi * w) / np.sum(w)))


def test_a_non_finite_sampling_prior_drops_its_row():
    """4d: the builder filters on ln_prior_d as well as on distance and weight, and nothing
    supplied a row where that mattered."""
    distance = np.array([100.0, 200.0, 300.0])
    ln_pi = np.array([0.0, -np.inf, 0.0])
    grid = build_distance_grid(distance, np.zeros(3), 0.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=3)
    assert len(grid) == 2, 'the -inf-prior row was binned'
    assert 200.0 not in set(grid["dist"].tolist())


def test_the_end_bins_still_reach_the_samples_that_defined_them():
    """4c: both end-edge clamps were dead in every existing setup, because the extrapolated
    edge already lay outside the data there.  Here the first centre sits close to the first
    sample, so dropping min()/max() would leave the outermost samples outside the grid."""
    distance = np.array([100.0, 101.0, 102.0, 400.0, 900.0])
    grid = build_distance_grid(distance, np.zeros(5), 0.0, 0.0, {},
                               ln_prior_d_at_samples=np.zeros(5), n_grid=3)

    lo = grid["dist"][0] - 0.5 * grid["dist_weight"][0]
    hi = grid["dist"][-1] + 0.5 * grid["dist_weight"][-1]
    assert np.sum(grid["dist_weight"]) >= np.ptp(distance), \
        'the bins no longer span the sampled range'
    assert lo <= distance.min() + 1e-9 or hi >= distance.max() - 1e-9


def test_a_grid_too_fine_for_float64_is_refused_not_floored():
    """The width raise, which nothing else reaches.  These distances ARE distinct, so
    distinct-value blocking gives them distinct centres -- but the midpoints between those
    centres collapse onto the centres in float64, so the bins have no width.  The old
    np.maximum(width, eps) floor turned exactly this into a row 36 nats bright."""
    distance = 5000.0 + np.arange(5) * 1e-12
    assert len(np.unique(distance)) == 5, "the inputs must be distinct or this tests nothing"

    with pytest.raises(ValueError) as excinfo:
        build_distance_grid(distance, np.zeros(5), 1.0, 0.0, {},
                            ln_prior_d_at_samples=np.zeros(5), n_grid=5)
    assert "non-positive" in str(excinfo.value)


def test_no_bin_width_is_floored_at_machine_epsilon():
    """A width of eps is not a narrow bin, it is a missing one; -log(eps) = 36 nats
    then lands in lnL."""
    distance, ln_pi, ln_w = _fair_draw([74.84, 94.26], [2, 2])

    grid = build_distance_grid(distance, ln_w, 64.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=4)

    assert np.all(grid["dist_weight"] > 1e6 * EPS), \
        "a bin width came from the floor, not from the sample spacing"
    # the sampled range is ~19.4 Mpc; no bin may be a rounding error of it
    assert np.min(grid["dist_weight"]) > 1e-6 * np.ptp(distance)


def test_a_likelihood_that_is_flat_in_distance_is_exported_flat():
    """Evenly spaced distinct distances, each repeated the same number of times, each
    copy weighted by the sampling prior.  Every bin then has the same width and the
    same mass, so the exported lnL is analytically constant -- L(d) is flat.  The
    defect split each triple across bins and wrote a -log(eps) = 36-nat sawtooth."""
    uniq = np.linspace(100.0, 900.0, 9)
    distance, ln_pi, _ = _fair_draw(uniq, np.full(len(uniq), 3))
    ln_w = ln_pi.copy()          # p(d) proportional to pi_d(d) <=> L(d) constant

    grid = build_distance_grid(distance, ln_w, 0.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=len(distance))

    spread = float(np.ptp(grid["lnL"]))
    assert spread < 1e-9, \
        "flat likelihood exported with a {:.2f}-nat spread across bins".format(spread)
    assert np.allclose(grid["dist_weight"], 100.0)


def test_duplicating_a_sample_only_reweights_it():
    """k copies of one distance must give exactly the grid one copy of k times the
    weight gives.  This is what makes the export independent of how many times the
    fair draw happened to land on a point."""
    rng = np.random.default_rng(3)
    for _ in range(25):
        uniq = np.sort(rng.lognormal(np.log(400.0), 0.35, size=int(rng.integers(3, 20))))
        counts = rng.integers(1, 6, size=len(uniq))
        ln_pi_u = _volumetric_log_prior(uniq)
        ln_w_u = -0.5 * ((uniq - 430.0) / 70.0) ** 2
        n_grid = int(rng.integers(1, len(uniq) + 1))

        repeated = build_distance_grid(
            np.repeat(uniq, counts), np.repeat(ln_w_u, counts), 5.0, 0.0, {},
            ln_prior_d_at_samples=np.repeat(ln_pi_u, counts), n_grid=n_grid)
        collapsed = build_distance_grid(
            uniq, ln_w_u + np.log(counts), 5.0, 0.0, {},
            ln_prior_d_at_samples=ln_pi_u, n_grid=n_grid)

        for field in ("dist", "dist_weight", "lnL", "ln_prior_d_sampling"):
            assert np.allclose(repeated[field], collapsed[field], rtol=0, atol=1e-12), field


###
### 2. a grid it cannot resolve must be refused or flagged, never written quietly
###

def test_rows_are_capped_at_the_number_of_distinct_distances():
    distance, ln_pi, ln_w = _fair_draw([120.0, 305.0, 900.0], [40, 7, 13])

    grid = build_distance_grid(distance, ln_w, 3.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=500)

    assert len(grid) == 3, "asked for 500 bins over 3 distinct distances, got {}".format(len(grid))


def test_a_single_distinct_distance_is_refused():
    """One distance has no width to report; the old code shipped it with width eps."""
    distance = np.full(6, 300.0)
    with pytest.raises(ValueError) as excinfo:
        build_distance_grid(distance, np.zeros(6), 1.0, 0.0, {},
                            ln_prior_d_at_samples=_volumetric_log_prior(distance), n_grid=6)
    assert "distinct" in str(excinfo.value)
    # and the boundary is where the refusal claims it is: TWO distinct distances export.
    # (Asserting MIN_UNIQUE_DISTANCES == 2 instead only restates the module's own literal.)
    two = build_distance_grid(np.array([300.0, 300.0, 460.0]), np.zeros(3), 1.0, 0.0, {},
                              ln_prior_d_at_samples=np.zeros(3), n_grid=6)
    assert len(two) == MIN_UNIQUE_DISTANCES


def test_resolution_warning_fires_exactly_when_the_draw_is_starved():
    starved, _, _ = _fair_draw([74.84, 94.26], [2, 2])
    assert distance_grid_resolution_warning(starved, n_grid=4) is not None
    assert distance_grid_resolution_warning(starved, n_grid=500) is not None

    healthy = np.linspace(130.1, 216.8, 5)          # the AV arm of the same command line
    assert distance_grid_resolution_warning(healthy, n_grid=5) is None
    assert distance_grid_resolution_warning(healthy) is None
    assert distance_grid_resolution_warning(np.array([])) is not None


###
### 3. healthy, all-distinct samples must be binned exactly as before
###

def test_all_distinct_samples_keep_the_equal_count_split():
    """No duplicates means unique-value blocking IS the old sample-index blocking, so
    equal weights must still give every bin the same mass."""
    n, n_grid = 120, 12
    # NON-UNIFORMLY SPACED on purpose.  On np.linspace an equal-WIDTH split is also an
    # equal-count split, so this assertion passed with the bins rebuilt by width (found by
    # mutating it).  A lognormal draw separates the two.
    rng = np.random.default_rng(4)
    distance = np.sort(rng.lognormal(np.log(400.0), 0.5, size=n))
    assert len(np.unique(distance)) == n
    ln_pi = _volumetric_log_prior(distance)

    grid = build_distance_grid(distance, np.zeros(n), -4.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=n_grid)

    assert len(grid) == n_grid
    mass = np.exp(grid["lnL"] + grid["ln_prior_d_sampling"] + np.log(grid["dist_weight"]))
    mass /= np.sum(mass)
    assert np.allclose(mass, 1.0 / n_grid, atol=1e-12), \
        "bins no longer carry equal probability mass: the split changed"


def test_starved_draw_still_reconstructs_the_marginal_it_was_given():
    """Capping rows must not move lnZ: the grid is still a partition of the same mass.

    NOT evidence that the binning is right.  lnL carries -log(width) and the
    reconstruction adds +log(dist_weight) back with the SAME width, so the identity holds
    for any partition -- the eps-floored one included.  It is here to catch a change to the
    normalization itself, and the tests above are what cover the blocking."""
    distance, ln_pi, _ = _fair_draw([74.84, 94.26], [2, 2])
    ln_w = np.array([0.0, 0.0, -0.3, -0.3])

    grid = build_distance_grid(distance, ln_w, 64.0, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=4)

    assert np.isclose(reconstruct_marginal_lnL(grid), 64.0)


###
### 4. the ILE must actually say so
###

def test_a_starved_grid_is_flagged_before_it_is_written():
    """Run the exporter's decision rather than read the driver's source.  This used to
    assert that the ILE source mentioned `distance_grid_resolution_warning`, which is
    satisfied by a comment and was silently satisfied for a while by an import line."""
    from RIFT.misc.distance_grid import distance_grid_inputs
    from RIFT.integrators.rvs_record import RvsRecord

    distance, ln_pi, ln_w = _fair_draw([74.84, 94.26], [2, 2])
    rvs = {'distance': distance, 'integrand': np.exp(ln_w),
           'joint_prior': np.exp(ln_pi), 'joint_s_prior': np.ones(len(distance))}
    rec = RvsRecord.fair_draw(rvs, reserve=None, integrand_is_log=False)

    _, _, _, notes, warning = distance_grid_inputs(
        rec, rvs, lambda: ln_w, n_grid=4)

    assert notes == [], 'claimed a retained set it does not have'
    assert warning is not None, 'four rows over two distinct distances was called fine'
    assert 'distinct distance' in warning
