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
    cfg = dict(n_comp={(0, 1): 3}, gmm_adapt={(0, 1): False}, correlate_all_dims=True)
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
