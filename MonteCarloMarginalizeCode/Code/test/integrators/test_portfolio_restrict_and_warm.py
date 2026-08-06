#!/usr/bin/env python
"""Cheap CPU-only regression tests for two portfolio invariants that fail SILENTLY.

Both bugs these cover produce a WRONG ANSWER with no exception and a healthy-looking n_eff, so
there is no runtime signal to catch them -- they can only be caught here.

  1. Range restriction must register coverage bookkeeping.  `restrict_member_range()` narrows a
     member so it can spend its fixed bin budget where the posterior is.  That is safe ONLY while
     some member keeps full support: proposals need not cover the prior, but the MIXTURE must cover
     the support of L*p.  The per-member draw floor, the restricted-only active-member guard and
     the q_mix fallback guard are all keyed off `_has_restricted_member`/`_full_support_members`.
     If the public API narrows a member without setting them, those guards go dark, member 0 can be
     allocated zero draws, and the mixture loses full support -- biasing the integral LOW.

  2. Warm state must be cleared on the MEMBERS.  `_warm` and the contracted AV grid live on each
     member; portfolio.integrate_log() does not rerun member setup().  A driver that clears only
     `portfolio._warm` leaves the previous point's CONTRACTED live volume installed, so the next
     point draws from a box that may exclude its own support.

Run:  python test_portfolio_restrict_and_warm.py
"""
import numpy as np

import RIFT.integrators.mcsamplerPortfolio as mcsP
import RIFT.integrators.mcsamplerAdaptiveVolume as mcsAV
import RIFT.integrators.mcsamplerEnsemble as mcsGMM


def _mk(n=3):
    return mcsP.MCSampler(portfolio=[mcsAV] * n)


def _mk_av_gmm():
    """AV + GMM, the production portfolio.  An AV-only portfolio cannot detect configuration loss
    across a reset -- AV.setup() ignores kwargs, so a bare setup() looks identical to a replayed
    one.  Only a member that CONSUMES its setup arguments can show the difference."""
    return mcsP.MCSampler(portfolio=[mcsAV, mcsGMM])


def _flat(x):
    return np.vectorize(lambda z: 0.1)


def test_restrict_rejects_invalid_member():
    s = _mk()
    for bad in (-1, 0, 99):
        try:
            s.restrict_member_range(bad, 'x', 0., 1.)
        except ValueError:
            continue
        raise AssertionError(
            "restrict_member_range accepted member_index={}; it would never match the positive "
            "enumerate() in add_parameter and would be a silent no-op".format(bad))


def test_restrict_sets_coverage_bookkeeping():
    s = _mk()
    s.restrict_member_range(1, 'x', -1., 1.)
    assert s._has_restricted_member is True
    assert s._full_support_members == [0, 2], s._full_support_members


def test_restrict_refuses_to_restrict_every_member():
    s = _mk(3)
    s.restrict_member_range(1, 'x', -1., 1.)
    s.restrict_member_range(2, 'x', -1., 1.)
    # members 1 and 2 restricted, member 0 is the backstop -> still fine
    assert s._full_support_members == [0]
    # and member 0 can never be restricted, so full coverage cannot be lost through this API
    try:
        s.restrict_member_range(0, 'x', -1., 1.)
    except ValueError:
        return
    raise AssertionError("restrict_member_range narrowed the full-support backstop")


def test_unconsumed_restriction_raises_at_setup():
    """A restriction naming a parameter that never arrives must FAIL, not silently do nothing."""
    s = _mk()
    s.restrict_member_range(1, 'typo_param', -1., 1.)
    s.add_parameter('x', _flat('x'), left_limit=-5., right_limit=5.)
    try:
        s.setup()
    except Exception as e:
        assert 'never applied' in str(e), str(e)
        return
    raise AssertionError("an unapplied range restriction survived setup() as a silent no-op")


def test_restriction_narrows_only_that_member():
    s = _mk()
    s.restrict_member_range(1, 'x', -1., 1.)
    s.add_parameter('x', _flat('x'), left_limit=-5., right_limit=5.)
    s.setup()
    lims = [(m.llim['x'], m.rlim['x']) for m in s.portfolio_realizations]
    assert lims == [(-5., 5.), (-1., 1.), (-5., 5.)], lims
    # the PORTFOLIO's own reference limits must stay the full physical range: they are taken from
    # member 0 before narrowing, and downstream code uses them as the prior's extent.
    assert (s.llim['x'], s.rlim['x']) == (-5., 5.)


def test_clear_warm_state_reaches_members():
    s = _mk()
    s.add_parameter('x', _flat('x'), left_limit=-5., right_limit=5.)
    s.setup()
    m = s.portfolio_realizations[1]
    m._warm = {'binunique': np.array([[0]]), 'dx': np.array([1.0]),
               'nbins': np.array([1]), 'ninbin': [10], 'V': 0.001}
    m._warm_applied = True
    m.V = 1e-6                      # pretend a heavily contracted live volume
    m.dx = np.array([1e-3])
    s.clear_warm_state()
    assert m._warm is None
    assert m._warm_applied is False
    assert m.V == 1, "clear_warm_state left the contracted grid installed (V={})".format(m.V)
    assert np.allclose(m.dx, [10.]), m.dx


def test_restrict_refuses_to_widen():
    """`restrict` must not silently WIDEN.  The prior callables are absolute densities normalized
    over the ORIGINAL range, so a member sampling outside it reports a wrong prior and biases the
    integral -- exactly the failure the API is supposed to prevent."""
    for lo, hi, what in [(-9., 9., "two-sided"), (-1., 7., "upper-only"), (-7., 1., "lower-only")]:
        s = _mk()
        s.restrict_member_range(1, 'x', lo, hi)   # not contained in the [-5,5] added below
        try:
            s.add_parameter('x', _flat('x'), left_limit=-5., right_limit=5.)
        except ValueError:
            continue
        lims = (s.portfolio_realizations[1].llim['x'], s.portfolio_realizations[1].rlim['x'])
        raise AssertionError("{} widening to [{}, {}] was accepted: member range is now {}".format(
            what, lo, hi, lims))


def test_clear_warm_state_preserves_member_configuration():
    """The reset must REPLAY each member's setup arguments.

    A bare setup() restores the cold grid but discards configuration: mcsamplerEnsemble.setup()
    rebuilds its dimension grouping and re-reads n_comp/gmm_adapt from kwargs, so a configured
    (0,1) GMM would come back as separate (0,), (1,) groups with n_comp defaulted -- a quietly
    different sampler for every point after the first.  An AV-only portfolio cannot see this."""
    s = _mk_av_gmm()
    for p in ('x', 'y'):
        s.add_parameter(p, _flat(p), left_limit=-5., right_limit=5.)
    # supply gmm_dict, as production does -- the no-gmm_dict path takes a different branch in
    # mcsamplerEnsemble.setup() and would not exercise what production actually runs
    cfg = dict(n_comp={(0, 1): 3}, gmm_adapt={(0, 1): False}, correlate_all_dims=True,
               gmm_dict={(0, 1): None})
    s.setup(**cfg)
    gmm = s.portfolio_realizations[1]

    def _snapshot():
        # repr(), not dict(): a bare setup() collapses n_comp from {(0,1): 3} to the scalar
        # default, and dict() on an int raises TypeError instead of reporting the defect.
        i = gmm.integrator
        return (sorted(i.gmm_dict), repr(i.n_comp), repr(i.gmm_adapt))

    before = _snapshot()
    s.clear_warm_state()
    after = _snapshot()
    assert before[0] == after[0], \
        "reset changed the GMM dimension grouping: {} -> {}".format(before[0], after[0])
    assert before[1] == after[1], "reset lost n_comp: {} -> {}".format(before[1], after[1])
    assert before[2] == after[2], "reset lost gmm_adapt: {} -> {}".format(before[2], after[2])


class _FakeModel(object):
    """Stand-in for a trained GMM component; identity is all this test needs."""
    def __repr__(self):
        return "<trained model>"


def test_clear_warm_state_clears_trained_proposal_not_just_config():
    """The reset must clear the TRAINED PROPOSAL, not merely restore the grouping.

    `gmm_dict` is not an inert spec.  mcsamplerEnsemble hands the caller's dict straight to
    monte_carlo.integrator, which stores it WITHOUT copying (MonteCarloEnsemble.py:110) and then
    writes trained models into it (`self.gmm_dict[dim_group] = model`, :403).  So a reset that
    replays a *reference* to those setup arguments hands the next point the previous point's
    trained proposal -- reintroducing, through the reset itself, the leak the reset exists to
    remove.  Checking grouping / n_comp / gmm_adapt alone cannot see this: all three survive."""
    s = _mk_av_gmm()
    for p in ('x', 'y'):
        s.add_parameter(p, _flat(p), left_limit=-5., right_limit=5.)
    caller_spec = {(0, 1): None}
    s.setup(n_comp={(0, 1): 3}, gmm_adapt={(0, 1): False}, correlate_all_dims=True,
            gmm_dict=caller_spec)
    gmm = s.portfolio_realizations[1]

    # the stored args must not alias the caller's dict, or training pollutes them
    assert s._member_setup_args[1]['gmm_dict'] is not caller_spec, \
        "stored setup args alias the caller's gmm_dict"

    # point 1 trains: the integrator writes a model into its gmm_dict
    gmm.integrator.gmm_dict[(0, 1)] = _FakeModel()
    s.clear_warm_state()
    assert gmm.integrator.gmm_dict.get((0, 1)) is None, \
        "reset left point 1's trained proposal installed: {}".format(gmm.integrator.gmm_dict)
    assert sorted(gmm.integrator.gmm_dict) == [(0, 1)], "reset lost the grouping"

    # point 2 trains, and resets again: the stored snapshot must not have been polluted by the
    # first replay (passing the stored dict itself would let the rebuilt integrator train into it,
    # so the leak would simply return one point later)
    gmm2 = s.portfolio_realizations[1]
    gmm2.integrator.gmm_dict[(0, 1)] = _FakeModel()
    s.clear_warm_state()
    assert gmm2.integrator.gmm_dict.get((0, 1)) is None, \
        "second reset leaked: the stored snapshot was polluted by the first replay"


class _AdaptiveSeed(object):
    """A seeded GMM whose update() mutates in place, as gaussian_mixture_model.gmm.update does."""
    def __init__(self):
        self.tempering_coeff = 1.0
        self.n_updates = 0
        self.means = np.zeros(2)

    def update(self, *a, **kw):
        self.tempering_coeff /= 2.0
        self.n_updates += 1
        self.means += 1.0


def test_snapshot_clones_a_real_gmm_model():
    """The clone path must work on the ACTUAL model class, not just a stand-in.

    A real `gmm` holds a module reference (`xpy`) and bound functions, so copy.deepcopy raises
    "cannot pickle 'module' object".  If the snapshot fell back to sharing on that failure, seeded
    models would not be isolated at all -- the fix would be inert on exactly the configuration it
    exists for.  Hence the shallow-copy-then-clone-attributes path."""
    from RIFT.integrators.gaussian_mixture_model import gmm
    rng = np.random.RandomState(0)
    m = gmm(2, np.array([[-5., 5.], [-5., 5.]]))
    m.fit(rng.normal(size=(400, 2)), log_sample_weights=np.zeros(400))
    clone = mcsP.MCSampler._snapshot_setup_args({'gmm_dict': {(0, 1): m}})['gmm_dict'][(0, 1)]
    assert clone is not m, "the real gmm model was SHARED, not cloned"
    t0 = m.tempering_coeff
    mu0 = None if m.means is None else np.array(m.means)
    clone.update(rng.normal(size=(200, 2)), log_sample_weights=np.zeros(200))
    assert m.tempering_coeff == t0 and (mu0 is None or np.array_equal(m.means, mu0)), \
        "mutating the clone changed the original: attribute state is still shared"


def test_adaptive_seeded_model_does_not_drift_across_reset():
    """A seeded-and-ADAPTING GMM must not carry point 1's adaptation into point 2.

    Production seeds per-group GMMs from a breadcrumb; with --extrinsic-proposal-adapt those
    seeded groups keep re-fitting, and update() mutates the model object in place.  Snapshotting
    only the containing dict would leave the stored baseline pointing at the live model, so the
    baseline drifts during point 1 and is replayed into point 2 -- the same leak one level down.
    (With adapt OFF, the default, _train skips seeded groups, so nothing mutates.)"""
    s = _mk_av_gmm()
    for p in ('x', 'y'):
        s.add_parameter(p, _flat(p), left_limit=-5., right_limit=5.)
    seed = _AdaptiveSeed()
    s.setup(n_comp={(0, 1): 3}, gmm_adapt={(0, 1): True}, correlate_all_dims=True,
            gmm_dict={(0, 1): seed})
    gmm_member = s.portfolio_realizations[1]
    stored = s._member_setup_args[1]['gmm_dict'][(0, 1)]
    assert stored is not seed, "stored baseline aliases the live seeded model"

    # point 1 adapts the live model in place
    gmm_member.integrator.gmm_dict[(0, 1)].update(None)
    gmm_member.integrator.gmm_dict[(0, 1)].update(None)
    assert s._member_setup_args[1]['gmm_dict'][(0, 1)].n_updates == 0, \
        "the stored baseline drifted while point 1 adapted"

    s.clear_warm_state()
    restored = s.portfolio_realizations[1].integrator.gmm_dict[(0, 1)]
    assert restored is not None, "reset cleared the seeded model entirely"
    assert restored.n_updates == 0, \
        "point 2 inherited point 1's adaptation (n_updates={})".format(restored.n_updates)
    assert restored.tempering_coeff == 1.0, "point 2 inherited point 1's tempering state"


def test_warm_start_keeps_a_backstop_when_every_member_is_compact():
    """With no full-support member, a warm start must not narrow ALL of them.

    `cover_frac` is not a coverage guarantee: a FINITE set of uniform points occupies only the
    bins it lands in, so a seeded grid is not a superset of a cold start (measured at d=6, even
    cover_frac=0.9 covers 2.9% of the box).  In an ALL-AV portfolio every component is a
    hard-edged box, so seeding all of them removes the mixture's coverage of the prior box.
    Member 0 -- the backstop restrict_member_range also refuses to narrow -- stays cold."""
    d = 4
    s = mcsP.MCSampler(portfolio=[mcsAV, mcsAV])
    for i in range(d):
        p = "x%d" % i
        s.add_parameter(p, _flat(p), prior_pdf=_flat(p), left_limit=-5., right_limit=5.,
                        adaptive_sampling=True)
    s.setup()
    rng = np.random.RandomState(1)
    cloud = rng.normal(0, 0.2, size=(1500, d))     # a tight seed, as at high SNR
    s.bootstrap_from_samples(cloud, cover_frac=0.5)
    for i in (0, 1):
        s.portfolio_realizations[i].draw_simplified(500)   # forces the seed onto the draw path
    v0 = float(s.portfolio_realizations[0].V)
    v1 = float(s.portfolio_realizations[1].V)
    assert v0 >= 1.0, "the full-support backstop was narrowed by the warm start (V={})".format(v0)
    assert v1 < 0.5, "member 1 was not actually warm-started (V={}); the test proves nothing".format(v1)


def test_warm_start_backstop_opt_out_still_works():
    """The old seed-everything behaviour must remain reachable, for A/B and for callers who
    know their seed is right."""
    d = 4
    s = mcsP.MCSampler(portfolio=[mcsAV, mcsAV])
    for i in range(d):
        p = "x%d" % i
        s.add_parameter(p, _flat(p), prior_pdf=_flat(p), left_limit=-5., right_limit=5.,
                        adaptive_sampling=True)
    s.setup()
    rng = np.random.RandomState(1)
    s.bootstrap_from_samples(rng.normal(0, 0.2, size=(1500, d)), cover_frac=0.5,
                             keep_backstop_cold=False)
    for i in (0, 1):
        s.portfolio_realizations[i].draw_simplified(500)
    assert float(s.portfolio_realizations[0].V) < 0.5, "opt-out did not seed member 0"


def test_warm_start_does_not_sacrifice_av_when_a_gmm_is_present():
    """The rule is "SOME member has support everywhere", not "member 0 is cold".

    A GMM member carries an explicit uniform defensive component (gmm_defensive_frac, default
    0.05) plus Gaussian tails, so q_mix never vanishes however it is seeded -- measured, a
    displaced seed in [AV, GMM] left |lnZ bias| <= 0.05 either way.  Holding member 0 cold there
    would disable the AV warm start entirely (member 0 IS the AV member) to buy a guarantee that
    already exists.  The merge gate caught exactly that regression, so it is pinned here."""
    d = 4
    s = mcsP.MCSampler(portfolio=[mcsAV, mcsGMM])
    for i in range(d):
        p = "x%d" % i
        s.add_parameter(p, _flat(p), prior_pdf=_flat(p), left_limit=-5., right_limit=5.,
                        adaptive_sampling=True)
    s.setup()
    rng = np.random.RandomState(1)
    s.bootstrap_from_samples(rng.normal(0, 0.2, size=(1500, d)), cover_frac=0.5)
    s.portfolio_realizations[0].draw_simplified(500)
    v0 = float(s.portfolio_realizations[0].V)
    assert v0 < 0.5, ("the AV member was left cold even though a full-support GMM member is "
                      "present: the warm start is disabled for no benefit (V={})".format(v0))

def test_clear_warm_state_propagates_failures():
    """A reset that quietly did not happen leaves the next point on the previous point's grid.
    That must abort, not become a log line."""
    s = _mk()
    s.add_parameter('x', _flat('x'), left_limit=-5., right_limit=5.)
    s.setup()

    def _boom(**kwargs):
        raise RuntimeError("member reset failed")
    s.portfolio_realizations[1].setup = _boom
    try:
        s.clear_warm_state()
    except RuntimeError:
        return
    raise AssertionError("clear_warm_state swallowed a failed member reset")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print("PASS", name)
    print("all portfolio restrict/warm invariants hold")
