#!/usr/bin/env python3
"""The ILE extrinsic (psi, phi_orb) proposal starts UNIFORM, and a seeded one must fit.

mcsamplerEnsemble once carried create_wide_single_component_prior(), a one-component gmm
meant as a wide seed for this group.  It had no `return`, so both ILE drivers installed None
-- which MonteCarloEnsemble._sample() reads as exact uniform sampling over the group's
bounds.  The helper is gone and the drivers say None outright.

These pin what made "just add the return" the wrong repair.  The load-bearing one is
test_seeded_model_dimension_must_match_group: when phase is marginalized the drivers' group
is the 1-element (psi,), the helper always built a 2-D model, and a 2-D model in a 1-D slot
biases lnZ by ln(width of phi_orb) with every recovered marginal still the right shape.

Reproduce the measurements behind this file with
    test/integrators/shape_extrinsic_phase_seed.py
"""
import ast
import os

import numpy as np
import pytest

from RIFT.integrators import mcsamplerEnsemble as ME
from RIFT.integrators import gaussian_mixture_model as GMM
from RIFT.integrators import MonteCarloEnsemble as monte_carlo

TWOPI = 2 * np.pi
PHASE_BOUNDS = [(0.0, TWOPI), (0.0, TWOPI)]
BINDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "bin"))
DRIVERS = ("integrate_likelihood_extrinsic_batchmode",
           "integrate_likelihood_extrinsic_batchmode_lisa")


def _physical_unit_seed(bounds, as_array=False):
    """Verbatim body of the removed helper, plus the `return` it never had."""
    b = np.array(bounds, dtype=float) if as_array else list(bounds)
    model = GMM.gmm(1, b)
    widths = np.array([bounds[k][1] - bounds[k][0] for k in np.arange(len(bounds))])
    model.means = [np.array([np.mean(bounds[k]) for k in np.arange(len(bounds))])]
    model.covariances = [np.diag(widths ** 2)]
    model.weights = [1]
    model.adapt = [False]
    model.d = len(bounds)
    return model


def _normalized_seed(bounds, d=None):
    """A wide seed expressed correctly: normalized [-1,1] frame, array bounds."""
    b = np.array(bounds, dtype=float)
    d = len(bounds) if d is None else d
    m = GMM.gmm(1, b)
    m.means = [np.zeros(len(bounds))]
    m.covariances = [np.diag(np.full(len(bounds), 4.0))]   # sigma = 2 = full normalized width
    m.weights = [1.0]
    m.adapt = [False]
    m.d = d
    m.N = 0
    return m


def _driver_tree(name):
    return ast.parse(open(os.path.join(BINDIR, name)).read())


###
### 1.  The removal itself.
###

def test_helper_is_gone():
    """Re-adding it, with or without the return, re-opens every defect this file documents."""
    assert not hasattr(ME, "create_wide_single_component_prior")


def test_no_driver_calls_the_removed_helper():
    """AST, not text: survives reformatting and renamed locals, and still names the call."""
    for name in DRIVERS:
        hits = [n for n in ast.walk(_driver_tree(name))
                if isinstance(n, ast.Attribute) and n.attr == "create_wide_single_component_prior"]
        assert not hits, "%s still calls the removed helper" % name


def test_driver_gmm_dict_literals_are_all_none():
    """Every `gmm_dict = {...}` literal in the drivers maps its groups to None.

    Checked on the VALUES, so it does not care what the key locals are called
    (pair_phi_psi, pair_ra_dec, ...) nor how the literal is wrapped across lines.
    """
    for name in DRIVERS:
        literals = [n.value for n in ast.walk(_driver_tree(name))
                    if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                    and any(isinstance(t, ast.Name) and t.id == "gmm_dict" for t in n.targets)]
        assert literals, "%s has no gmm_dict literal" % name
        for lit in literals:
            for k, v in zip(lit.keys, lit.values):
                assert isinstance(v, ast.Constant) and v.value is None, (
                    "%s: gmm_dict literal seeds %s with a model; a seed must clear "
                    "test_seeded_model_dimension_must_match_group first"
                    % (name, ast.dump(k) if k else "?"))


# A census of `gmm_dict[...] = <model>` writes used to live here.  Removed: it caught the one
# spelling it was written against and missed `.update({...})`, a dict comprehension, an alias
# (`_gd = gmm_dict`), and a seed written straight into the `extra_args` literal -- while firing
# on a legitimate second breadcrumb write.  validate_gmm_dict() is spelling-independent and
# catches what actually matters, so the illusory coverage is worse than none.


###
### 2.  What None actually does, and what a seed would have to beat.
###

def test_none_entry_is_exact_uniform():
    """A None gmm_dict entry draws uniformly and reports sampling prior 1/volume exactly."""
    d = 2
    bounds = np.array(PHASE_BOUNDS, dtype=float)
    # the integrator takes bounds as a dict keyed by dim_group, as mcsamplerEnsemble builds it
    integ = monte_carlo.integrator(d, {(0, 1): bounds}, {(0, 1): None}, 1, n=20000,
                                   user_func=None, L_cutoff=None)
    np.random.seed(17)
    integ._sample()
    x = integ.sample_array
    assert x.shape == (20000, d)
    vol = np.prod(bounds[:, 1] - bounds[:, 0])
    # exactly 1/vol -- not merely close: the None branch multiplies by a constant
    assert np.allclose(integ.sampling_prior_array, 1.0 / vol, rtol=0, atol=0)
    assert x.min() >= bounds[:, 0].min() and x.max() <= bounds[:, 1].max()
    # flat marginals (10 bins, 20000 draws: a 6-sigma band is ~+/-19%)
    for k in range(d):
        h, _ = np.histogram(x[:, k], bins=10, range=tuple(bounds[k]))
        assert np.all(np.abs(h / 2000.0 - 1.0) < 0.19), h


def test_normalized_seed_round_trips():
    """A correctly framed wide seed IS usable -- the bar any future seed has to clear."""
    model = _normalized_seed(PHASE_BOUNDS)
    np.random.seed(23)
    s = model.sample(50000)
    sc = model.score(s)
    assert np.all(sc > 1e-299), "proposal fell to the score() floor"
    assert abs(np.mean(1.0 / sc) / (TWOPI ** 2) - 1.0) < 0.02


###
### 3.  The dimension mismatch, which is the only silent failure of the three.
###

def test_seeded_model_dimension_must_match_group():
    """A model whose d differs from its dim-group is refused, in BOTH directions.

    _sample() divides the weights by model.score(), the model's full joint density.  An extra
    axis inflates lnZ by ln(its width) -- +1.84 nats for the 2-D (psi, phi_orb) model the
    drivers built against the 1-element (psi,) group they use when phase is marginalized --
    and nothing downstream sees it, because every marginal keeps its shape.  A missing axis
    raises IndexError on the first _sample() instead, which is loud but still worth refusing
    up front.  Only the d > len(group) direction is silent, so the message must say so.
    """
    bounds2 = np.array(PHASE_BOUNDS, dtype=float)
    two_d = _normalized_seed(PHASE_BOUNDS)
    one_d = _normalized_seed(PHASE_BOUNDS[:1])
    assert (two_d.d, one_d.d) == (2, 1)
    # matching: accepted
    monte_carlo.integrator(2, {(0, 1): bounds2}, {(0, 1): two_d}, 1, n=10)
    # too many axes -- the silent one
    with pytest.raises(ValueError) as e:
        monte_carlo.integrator(2, {(0,): bounds2[:1]}, {(0,): two_d}, 1, n=10)
    msg = str(e.value)
    assert "d=2" in msg and "1-dimensional" in msg
    assert "HIGH" in msg, "the message must name the direction of the bias: %r" % msg
    # too few axes -- the loud one, refused just the same
    with pytest.raises(ValueError) as e2:
        monte_carlo.integrator(2, {(0, 1): bounds2}, {(0, 1): one_d}, 1, n=10)
    assert "d=1" in str(e2.value) and "2-dimensional" in str(e2.value)
    # None is always allowed
    monte_carlo.integrator(2, {(0, 1): bounds2}, {(0, 1): None}, 1, n=10)


@pytest.mark.parametrize("bad_d", [2.4, "2"])
def test_dimension_is_compared_by_value_not_by_int(bad_d):
    """int(d) would accept 2.4 and '2' against a 2-element group.  Compare the value."""
    bounds2 = np.array(PHASE_BOUNDS, dtype=float)
    m = _normalized_seed(PHASE_BOUNDS)
    m.d = bad_d
    with pytest.raises(ValueError):
        monte_carlo.integrator(2, {(0, 1): bounds2}, {(0, 1): m}, 1, n=10)
    m.d = np.int64(2)      # a legitimate numpy integer must still pass
    monte_carlo.integrator(2, {(0, 1): bounds2}, {(0, 1): m}, 1, n=10)


def test_seeded_model_bounds_must_match_the_group_box():
    """The dimension can match while the BOX does not, and that is the reachable failure.

    A seed is drawn and scored over its own bounds, so one normalized on a smaller box never
    draws outside it: the run explores the overlap only and lnZ is low by ln(area ratio),
    with every marginal still the right shape.  --internal-rotate-phase doubles psi and
    phi_orb to (0, 4pi), so an --extrinsic-proposal-breadcrumb recorded without the flag and
    replayed with it is exactly this case: a quarter of the box reachable, -1.39 nats.
    """
    narrow = _normalized_seed(PHASE_BOUNDS)                       # (0, 2pi)^2
    wide = np.array([(0.0, 4 * np.pi), (0.0, 4 * np.pi)])         # the group under the flag
    assert narrow.d == 2 == len(wide), "dimension alone cannot see this"
    with pytest.raises(ValueError) as e:
        monte_carlo.integrator(2, {(0, 1): wide}, {(0, 1): narrow}, 1, n=10)
    assert "normalized over" in str(e.value)
    # the same seed against its own box is fine
    monte_carlo.integrator(2, {(0, 1): np.array(PHASE_BOUNDS, dtype=float)},
                           {(0, 1): _normalized_seed(PHASE_BOUNDS)}, 1, n=10)


def test_guard_reads_flat_array_bounds_too():
    """`integrator.bounds` is a dict only when the caller passed an explicit grouping.

    A bare setup() leaves a flat (d,2) array, which is what the portfolio's GMM member runs
    on.  Reading only the dict form made the box half of the check a silent no-op there.
    """
    # DISTINCT per-row boxes: with identical rows the test cannot tell `bounds[list(grp)]`
    # from `bounds[[0]*len(grp)]`, so it would not see the guard reading the wrong rows.
    flat = np.array([(0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0), (0.0, TWOPI)])
    seed_on_row4 = _normalized_seed([(0.0, TWOPI)])
    # row 4 matches -> accepted
    monte_carlo.validate_gmm_dict(flat, {(4,): seed_on_row4}, where="flat bounds")
    # the same seed against row 0 (box (0,1)) must be refused, which only holds if the
    # guard indexed row 0 rather than falling back to some other row
    with pytest.raises(ValueError) as e:
        monte_carlo.validate_gmm_dict(flat, {(0,): seed_on_row4}, where="flat bounds")
    assert "normalized over" in str(e.value)
    # multi-element, non-contiguous group reads exactly those rows
    seed_rows_1_3 = _normalized_seed([(0.0, 2.0), (0.0, 4.0)])
    monte_carlo.validate_gmm_dict(flat, {(1, 3): seed_rows_1_3}, where="flat bounds")
    with pytest.raises(ValueError):
        monte_carlo.validate_gmm_dict(flat, {(0, 3): seed_rows_1_3}, where="flat bounds")


def test_warm_start_transfer_is_validated():
    """The warm-start transfer writes into gmm_dict AFTER construction, so it needs its own
    call.  Deleting that call leaves every other test in this file green."""
    import RIFT.integrators.mcsamplerEnsemble as _me
    src = open(_me.__file__).read()
    # anchor on the transfer's own line; "[GMM warm-start]" alone also matches the seed-fit
    anchor = src.index("transferred {} fitted proposal group(s)")
    block = src[anchor - 1400:anchor + 120]
    assert "validate_gmm_dict" in block, (
        "the warm-start transfer installs models post-construction and must re-validate")
    # and it works: a wrong-box model planted the way the transfer plants one is refused
    flat = np.array([(0.0, TWOPI), (0.0, TWOPI)])
    narrow = _normalized_seed([(0.0, np.pi / 2), (0.0, np.pi / 2)])
    with pytest.raises(ValueError) as e:
        monte_carlo.validate_gmm_dict(flat, {(0, 1): narrow}, where="warm-start gmm_dict")
    assert "warm-start" in str(e.value)


def test_message_names_the_parameters_not_just_dim_indices():
    """An operator reading a held job cannot map (4,) to psi without reading the driver."""
    flat = np.array([(0.0, TWOPI), (0.0, TWOPI), (0.0, TWOPI), (0.0, TWOPI), (0.0, TWOPI)])
    two_d = _normalized_seed(PHASE_BOUNDS)
    names = ["right_ascension", "declination", "distance", "inclination", "psi"]
    with pytest.raises(ValueError) as e:
        monte_carlo.validate_gmm_dict(flat, {(4,): two_d}, param_names=names)
    assert "psi" in str(e.value), "the message must name the parameter: %r" % str(e.value)


###
### 4.  Why repairing the helper in place was never one line.
###

def test_mismatched_seed_would_have_biased_lnZ_by_the_extra_log_volume():
    """Price the bias the dimension check prevents, at the gmm level the guard does not touch.

    E[1/score] over the model's own draws is the volume its density normalizes over.  For a
    2-D model that is the 2-D box, so using it for a 1-D group inflates every weight by the
    width of the extra axis -- the whole bias, and a pure constant, which is why it never
    shows up in a recovered marginal.  This is the only in-tree measurement behind the
    "+1.84 nats" that the guard's docstring, its ValueError text and the mcsamplerEnsemble
    removal comment all assert; deleting it leaves that number unmeasured.
    """
    np.random.seed(29)
    m = _normalized_seed(PHASE_BOUNDS)
    s = m.sample(200000)
    inflation = float(np.mean(1.0 / m.score(s)) / TWOPI)   # 2-D volume / 1-D volume
    assert abs(np.log(inflation) - np.log(TWOPI)) < 0.02, (
        "expected ln(2*pi)=%.4f of lnZ bias, measured %.4f" % (np.log(TWOPI), np.log(inflation)))


def test_flat_bounds_are_converted_off_the_device():
    """The flat bounds array is built with self.xpy, which is cupy on a GPU host.

    Without a _to_host_bounds() hop, `np.asarray()` of a device array raises, the except
    yields gb=None and the box check is skipped -- inert on exactly the hosts whose portfolio
    GMM member runs the flat path.  Uses the vetted _DeviceLike stand-in from
    test_gmm_backend_dispatch, which refuses implicit numpy conversion the way cupy does, so
    this fails on a cupy-free runner if the hop is removed.
    """
    import os
    import sys
    _testdir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if _testdir not in sys.path:
        sys.path.insert(0, _testdir)
    try:
        from test_gmm_backend_dispatch import _device
    except Exception as e:                                    # pragma: no cover
        pytest.skip("device stand-in unavailable: %s" % e)
    flat = np.array([(0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0), (0.0, 4 * np.pi)])
    wrong = _normalized_seed([(0.0, TWOPI)])                  # row 4 is (0, 4pi)
    right = _normalized_seed([(0.0, 4 * np.pi)])
    # host control first: if this does not refuse, the test is not testing the device at all
    with pytest.raises(ValueError):
        monte_carlo.validate_gmm_dict(flat, {(4,): wrong}, where="host flat")
    # and the same on the device
    with pytest.raises(ValueError) as e:
        monte_carlo.validate_gmm_dict(_device(flat), {(4,): wrong}, where="device flat")
    assert "normalized over" in str(e.value)
    # the matching seed is still accepted through the same conversion
    monte_carlo.validate_gmm_dict(_device(flat), {(4,): right}, where="device flat")


def test_list_bounds_seed_is_not_silently_usable():
    """The drivers' list-of-tuples bounds must never yield an UNNORMALIZED density.

    gmm.score() indexes self.bounds as an array (self.bounds.T).  Before PR #347 that raised
    on a list, which is the only reason restoring the missing `return` did not install the
    seed.  score() now converts its bounds, so this takes the assertion path and the helper's
    output would have been installed and used.  Written to hold either way: score() may
    refuse, but if it answers, the density must integrate to one -- an unnormalized one would
    bias lnZ with nothing downstream to catch it.
    """
    model = _physical_unit_seed(PHASE_BOUNDS, as_array=False)
    np.random.seed(41)
    try:
        s = model.sample(50000)
        sc = model.score(s)
    except (AttributeError, TypeError, IndexError):
        return  # refused -- the documented behaviour today
    assert np.all(sc > 1e-299), "proposal fell to the score() floor"
    assert abs(np.mean(1.0 / sc) / (TWOPI ** 2) - 1.0) < 0.02, (
        "score() disagrees with sample(): the proposal density is not normalized")


def test_physical_unit_seed_puts_its_mean_outside_the_model_frame():
    """A gmm stores mean/cov in the normalized [-1,1] image of its bounds, not in radians.

    The consequence is a wrong FRAME, not a wrong answer: with cov=diag(width**2) the
    truncated result is still near-uniform, so score() and sample() stay consistent and lnZ
    stays unbiased.  That is why this defect alone never showed up as anything.
    """
    model = _physical_unit_seed(PHASE_BOUNDS, as_array=True)
    assert np.all(np.abs(model.means[0]) > 1.0)     # (pi, pi) in a frame spanning [-1, 1]
    np.random.seed(5)
    s = model.sample(50000)
    assert abs(np.mean(1.0 / model.score(s)) / (TWOPI ** 2) - 1.0) < 0.02


def test_seed_with_no_sample_count_is_discarded_by_the_first_update():
    """_merge caps the old model's weight at self.N, so an N=0 seed contributes nothing.

    This is a property of _merge, not a judgement about seeding: any consumer that wants a
    seed to survive its first refit must set model.N to the sample count the seed represents.
    (RIFT/calmarg/extrinsic_handoff.reconstruct_gmm does not, which makes the adapting mode
    of the extrinsic breadcrumb inert -- separate defect, not fixed here.)
    """
    bounds = np.array(PHASE_BOUNDS, dtype=float)
    np.random.seed(31)
    cloud = np.clip(np.random.normal([1.0, 5.0], 0.2, (3000, 2)), 1e-3, TWOPI - 1e-3)
    seeded = _normalized_seed(PHASE_BOUNDS)
    assert seeded.N == 0
    seeded.update(cloud)
    fresh = GMM.gmm(1, bounds)
    fresh.fit(cloud)
    assert seeded.N == 3000
    assert np.allclose(seeded.means[0], fresh.means[0], atol=1e-9), (
        "seed retained influence; _merge's N cap is expected to discard an N=0 seed")
    # and a seeded entry stays at its own k -- it never grows to comp_dict's n_phase
    assert seeded.k == 1
