"""mcsamplerEnsemble (--sampler-method GMM): dim-group keys, and the evidence they decide.

A gmm_dict key is a tuple of INTEGRATION DIMENSION indices, and those index the POSITIONAL
ARGUMENT order of integrate(), not the order add_parameter() was called in.  ILE built its keys
from sampler.params_ordered instead, and the two orders are different, so:

  * every named group adapted the wrong parameters, and
  * a dimension could end up in no group at all (t_ref, without --time-marginalization).

MonteCarloEnsemble._sample() allocates sample_array with xpy.empty and writes only the columns a
dim-group names, so an uncovered column is uninitialized memory -- fed to the integrand and to
the prior, with no matching factor in the sampling density.  The estimator is then unnormalized:
on the zero-likelihood ILE fixture it returned ln Z = -28.7 (and, on a different allocation, NaN)
where the exact answer is 0, reporting sigma = 0.178 against it.

The two defects are NOT the same size, and the analytic assertions below only catch one of
them.  An uncovered dimension moves ln Z by tens of nats and is caught by test_analytic_evidence.
The wrong FRAME, where the cover happens to be complete, leaves ln Z unbiased -- measured at
0.006-0.02 nats on these targets, an order of magnitude BELOW the tolerance here -- and costs
n_eff instead.  It is pinned by the frame tests, not the evidence tests.

Tolerances come from the measured across-seed spread of this integrator on these two targets
(seeds 0-7, worst |deviation| 0.0349 nats, so the 0.15 used below is 4.3x that; on seeds
1000-1007 it is 0.0251, i.e. 6.0x -- state the seeds, the two families differ); the cover
defect being pinned is 11-29 nats.
"""
import contextlib
import io
import warnings

import numpy as np
import pytest

from scipy.special import i0

import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
from RIFT.integrators import MonteCarloEnsemble as monte_carlo


# ILE's extrinsic parameters, in the two orders that disagree.
PARAM_LIMITS = {
    'psi': (0.0, 2 * np.pi),
    'phi_orb': (0.0, 2 * np.pi),
    'inclination': (0.0, np.pi),
    'distance': (1.0, 10000.0),
    'right_ascension': (0.0, 2 * np.pi),
    'declination': (-np.pi / 2, np.pi / 2),
    't_ref': (-0.075, 0.075),
}
# the order ILE calls add_parameter() in
PARAMS_ORDERED = ['psi', 'phi_orb', 'inclination', 'distance',
                  'right_ascension', 'declination', 't_ref']
# the order ILE calls sampler.integrate() with (opts.time_marginalization off)
INTEGRATE_ARGS = ['right_ascension', 'declination', 't_ref', 'phi_orb',
                  'inclination', 'psi', 'distance']

LN_Z_TOL = 0.15   # 4.3x the measured worst across-seed deviation, 70x below the smallest defect


def _prior_pdf(name):
    """Priors normalized over that parameter's own box, so a flat target integrates to exactly 1."""
    lo, hi = PARAM_LIMITS[name]
    if name == 'declination':
        return lambda v: 0.5 * np.cos(v)
    if name == 'inclination':
        return lambda v: 0.5 * np.sin(v)
    if name == 'distance':
        return lambda v: v ** 2 / (hi ** 3 / 3.0 - lo ** 3 / 3.0)
    return lambda v: np.ones(np.shape(v)) / (hi - lo)


def _make_sampler():
    sampler = mcsamplerEnsemble.MCSampler()
    for p in PARAMS_ORDERED:
        sampler.add_parameter(p, left_limit=PARAM_LIMITS[p][0], right_limit=PARAM_LIMITS[p][1],
                              prior_pdf=_prior_pdf(p))
    return sampler


def _args_frame(names):
    """A dim-group key in the frame integrate() actually uses."""
    return tuple(INTEGRATE_ARGS.index(x) for x in names)


def _params_ordered_frame(names):
    """A dim-group key in the frame ILE used to build (the bug)."""
    return tuple(PARAMS_ORDERED.index(x) for x in names)


def _ile_groups(frame):
    """ILE's three named groups, plus the leftover dimension as its own group."""
    sky = frame(['right_ascension', 'declination'])
    d_incl = frame(['distance', 'inclination'])
    phi_psi = frame(['psi', 'phi_orb'])
    n_comp = {sky: 4, d_incl: 2, phi_psi: 4}
    adapt = {sky: True, d_incl: True, phi_psi: True}
    covered = set(sky) | set(d_incl) | set(phi_psi)
    for i in range(len(INTEGRATE_ARGS)):
        if i not in covered:
            n_comp[(i,)] = 1
            adapt[(i,)] = False
    return {k: None for k in n_comp}, n_comp, adapt


def _integrate(sampler, gmm_dict, n_comp, adapt, a_coeff, seed, nmax=200000, neff=1000):
    i_phi = INTEGRATE_ARGS.index('phi_orb')

    def integrand(*columns):
        # supplementary factor A*cos(phi_orb) on a zero signal likelihood: since phi_orb's prior
        # is uniform on [0, 2pi) and the factor depends on nothing else, the exact marginal is
        # ln I0(A), to machine precision, with no fit error and no MC scatter in the target.
        return a_coeff * np.cos(columns[i_phi])

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return sampler.integrate(integrand, *INTEGRATE_ARGS, nmax=nmax, neff=neff, n=10000,
                                 use_lnL=True, return_lnI=True, n_comp=n_comp,
                                 gmm_dict=dict(gmm_dict), gmm_adapt=adapt, max_iter=20,
                                 verbose=False)
    finally:
        np.random.set_state(state)


def test_the_two_frames_really_disagree():
    """Anchor: without this, every frame assertion below could pass vacuously."""
    for names in (['right_ascension', 'declination'], ['distance', 'inclination'], ['psi', 'phi_orb']):
        assert _args_frame(names) != _params_ordered_frame(names), names


def test_dim_group_keys_index_positional_arguments():
    """The bounds attached to a dim-group must be the bounds of the parameters it names.

    This is what makes the key frame observable: ILE's params_ordered-frame key (4, 5) was
    'right_ascension, declination', but integrate() read it as 'inclination, psi'.
    """
    sampler = _make_sampler()
    gmm_dict, n_comp, adapt = _ile_groups(_args_frame)
    _integrate(sampler, gmm_dict, n_comp, adapt, a_coeff=0.0, seed=1000, nmax=20000, neff=10)

    integrator = sampler.integrator
    for names in (['right_ascension', 'declination'], ['distance', 'inclination'], ['psi', 'phi_orb']):
        key = _args_frame(names)
        expected = np.array([PARAM_LIMITS[n] for n in names], dtype=float)
        got = np.asarray(sampler.identity_convert(integrator.bounds[key]), dtype=float)
        assert np.allclose(got, expected), \
            "dim-group {} carries bounds {}, expected {} for {}".format(key, got, expected, names)


@pytest.mark.parametrize('bad_gmm_dict,reason', [
    ({(0, 1): None, (6, 4): None, (5, 3): None}, 'uncovered'),        # t_ref (dim 2) in no group
    ({(0, 1): None, (2, 3): None, (4, 5): None, (6, 0): None}, 'duplicate'),
    ({(0, 1): None, (2, 3): None, (4, 5): None, (6, 7): None}, 'out of range'),
])
def test_incomplete_dim_group_cover_is_refused(bad_gmm_dict, reason):
    """Refuse the integral rather than return a confident wrong number.

    An uncovered dimension is never written, so sample_array carries whatever xpy.empty
    returned; a repeated one is double counted.  Both make ln Z meaningless while the reported
    sigma stays small, and neither is recoverable downstream.
    """
    sampler = _make_sampler()
    n_comp = {k: 1 for k in bad_gmm_dict}
    adapt = {k: False for k in bad_gmm_dict}
    with pytest.raises(ValueError):
        _integrate(sampler, bad_gmm_dict, n_comp, adapt, a_coeff=0.0, seed=1000,
                   nmax=20000, neff=10)


def test_complete_cover_of_the_same_run_is_accepted():
    """Control for the test above: the only difference is that every dimension has a group."""
    sampler = _make_sampler()
    gmm_dict, n_comp, adapt = _ile_groups(_args_frame)
    assert sorted(i for k in gmm_dict for i in k) == list(range(len(INTEGRATE_ARGS)))
    _integrate(sampler, gmm_dict, n_comp, adapt, a_coeff=0.0, seed=1000, nmax=20000, neff=10)


@pytest.mark.parametrize('a_coeff', [0.0, 8.0])
def test_analytic_evidence(a_coeff):
    """ln Z against a closed form: E_prior[exp(A cos phi_orb)] = I0(A)."""
    sampler = _make_sampler()
    gmm_dict, n_comp, adapt = _ile_groups(_args_frame)
    ln_z, _, n_eff, _ = _integrate(sampler, gmm_dict, n_comp, adapt, a_coeff, seed=1000)

    exact = float(np.log(i0(a_coeff)))
    assert np.isfinite(ln_z), "ln Z = {} (A={})".format(ln_z, a_coeff)
    assert abs(float(ln_z) - exact) < LN_Z_TOL, \
        "ln Z = {:.4f}, exact {:.4f}, off by {:.4f} nats (A={})".format(
            float(ln_z), exact, float(ln_z) - exact, a_coeff)
    # A collapsed proposal is the other way this integral goes wrong; the measured floor over 8
    # seeds on these targets was n_eff ~1000, so 100 only catches a collapse.
    assert float(n_eff) > 100, "n_eff = {:.1f} (A={})".format(float(n_eff), a_coeff)


def test_dim_group_frame_differs_by_how_the_sampler_is_driven():
    """mcsamplerEnsemble numbers its dimensions differently depending on its driver.

    integrate(func, *args) builds raw_bounds from `args`, so dim i is args[i].  A portfolio
    never calls member.integrate(): it calls member.setup() -- where _setup_impl uses
    dim = len(self.params_ordered) -- and then member.draw_simplified(n, *params_ordered), so
    dim i is params_ordered[i].  One key set cannot serve both, and building keys for the wrong
    one groups the wrong parameters with no effect on ln Z that a flat target can see.
    """
    sampler = _make_sampler()
    assert list(sampler.params_ordered) == PARAMS_ORDERED
    assert mcsamplerEnsemble.dim_group_frame(sampler, INTEGRATE_ARGS, driven_by_integrate=True) \
        == list(INTEGRATE_ARGS)
    assert mcsamplerEnsemble.dim_group_frame(sampler, INTEGRATE_ARGS, driven_by_integrate=False) \
        == list(PARAMS_ORDERED)
    # anchor: the distinction is only meaningful because the two orders disagree
    assert list(INTEGRATE_ARGS) != list(PARAMS_ORDERED)


def test_setup_path_groups_the_parameters_the_key_names():
    """The setup()/draw_simplified() path, which the portfolio uses and integrate() does not.

    Keys built in the params_ordered frame must land on the parameters they name.  Feeding this
    path args-frame keys silently regroups everything -- that is the defect this pins, and
    test_analytic_evidence cannot see it because the estimator stays unbiased.
    """
    sampler = _make_sampler()
    frame = mcsamplerEnsemble.dim_group_frame(sampler, INTEGRATE_ARGS, driven_by_integrate=False)
    groups = [['right_ascension', 'declination'], ['distance', 'inclination'], ['psi', 'phi_orb']]
    gmm_dict = {tuple(frame.index(n) for n in g): None for g in groups}
    mcsamplerEnsemble.complete_dim_group_cover(gmm_dict, len(frame))
    sampler.setup(n_comp={k: 1 for k in gmm_dict}, gmm_dict=dict(gmm_dict))

    for g in groups:
        key = tuple(frame.index(n) for n in g)
        assert key in sampler.integrator.gmm_dict
        # compare against the integrator's OWN bounds, which is what _sample() will draw from --
        # reading the names back out of `frame` would be true by construction.
        got = np.asarray(sampler.identity_convert(sampler.integrator.bounds[key]), dtype=float)
        expected = np.array([PARAM_LIMITS[n] for n in g], dtype=float)
        assert np.allclose(got, expected), \
            "key {} carries bounds {}, expected {} for {}".format(key, got, expected, g)
    # and the setup() path is covered by the guard too (it was not, before)
    assert sorted(i for k in sampler.integrator.gmm_dict for i in k) == list(range(len(frame)))


def test_setup_path_is_guarded_too():
    """_setup_impl builds its own integrator; the guard has to run on that path as well."""
    sampler = _make_sampler()
    partial = {(0, 1): None, (2, 3): None, (4, 5): None}      # last dim uncovered
    with pytest.raises(ValueError):
        sampler.setup(n_comp={k: 1 for k in partial}, gmm_dict=dict(partial), setup_forget=True)


def test_setup_path_names_the_frame_for_an_out_of_range_key():
    """The guard must run BEFORE _setup_impl indexes raw_bounds with the key.

    Dropping the _setup_impl call site and relying on integrator.__init__ is invisible for an
    uncovered dimension -- __init__ catches that either way -- but an out-of-range index dies
    first, building bounds, as a bare IndexError that says nothing about dim-group frames.  That
    is exactly the message an operator needs, because an out-of-range key IS the signature of
    keys built in the wrong frame against a shorter parameter list.
    """
    sampler = _make_sampler()
    n = len(PARAMS_ORDERED)
    bad = {(i,): None for i in range(n - 1)}
    bad[(n - 1, n + 4)] = None                                 # n+4 is not a dimension
    with pytest.raises(ValueError) as e:
        sampler.setup(n_comp={k: 1 for k in bad}, gmm_dict=dict(bad), setup_forget=True)
    msg = str(e.value)
    assert "outside the" in msg and "POSITIONAL ARGUMENT" in msg, \
        "must be the OUT-OF-RANGE branch and must name the frame: %r" % msg


def test_complete_dim_group_cover_fills_and_reports():
    gmm_dict = {(0, 1): None, (4, 5): None}
    comp, adapt = {(0, 1): 4, (4, 5): 2}, {(0, 1): True, (4, 5): True}
    added = mcsamplerEnsemble.complete_dim_group_cover(gmm_dict, 7, comp_dict=comp, gmm_adapt=adapt)
    assert added == [(2,), (3,), (6,)]
    assert sorted(i for k in gmm_dict for i in k) == list(range(7))
    assert set(comp) == set(gmm_dict) == set(adapt), "n_comp/gmm_adapt must stay keyed like gmm_dict"
    # idempotent: a second pass adds nothing
    assert mcsamplerEnsemble.complete_dim_group_cover(gmm_dict, 7, comp_dict=comp, gmm_adapt=adapt) == []


def test_empty_dim_group_key_is_dropped_and_harmless():
    """An empty dim-group key must be DROPPED, not merely tolerated.

    It is not inert.  `_sample()` survives it -- an (n, 0) draw, `prod([]) == 1` -- but `_train()`
    also iterates gmm_dict and tries to FIT a mixture to that zero-width block: LAPACK raises, the
    whole proposal is `_reset()`, and enough consecutive failures end the run with "GMM proposal
    refit failed 5 consecutive times".  Measured before this was fixed: 4 proposal resets.

    This uses CIP's ACTUAL call shape, which is what makes the defect reachable and is the corner
    an earlier version of this test missed:
      * `n_comp` is an INT (`opts.internal_n_comp`, default 1), not a dict, so `_train` cannot be
        talked out of fitting the group by a per-group count of 0;
      * NO `gmm_adapt` is passed, so the `gmm_adapt.get(group)` skip never fires;
      * the target is SHARP, so the proposal genuinely has to adapt -- on a flat target the
        uniform proposal converges in one chunk and swallows the training failures.
    """
    lo, hi = 0.0, 1.0
    def make():
        s = mcsamplerEnsemble.MCSampler()
        for nm in ('a', 'b', 'c'):
            s.add_parameter(nm, left_limit=lo, right_limit=hi,
                            prior_pdf=lambda v: np.ones(np.shape(v)))
        return s

    def go(gmm_dict):
        sampler = make()
        state = np.random.get_state()
        buf = io.StringIO()
        try:
            np.random.seed(1000)
            with warnings.catch_warnings(record=True) as caught, \
                 contextlib.redirect_stdout(buf):
                warnings.simplefilter("always")
                res = sampler.integrate(
                    lambda *cols: -0.5 * (((cols[0] - 0.5) / 0.02) ** 2
                                          + ((cols[1] - 0.5) / 0.02) ** 2),
                    'a', 'b', 'c', nmax=400000, neff=40, n=5000, use_lnL=True, return_lnI=True,
                    n_comp=1, gmm_dict=dict(gmm_dict), max_iter=80, verbose=False)
                warned = any("empty dim-group" in str(w.message) for w in caught)
        finally:
            np.random.set_state(state)
        return float(res[0]), float(res[2]), buf.getvalue().count("Error training, resetting"), warned

    plain = {(0,): None, (1,): None, (2,): None}
    with_empty = dict(plain); with_empty[()] = None

    ln_z_ref, n_eff_ref, resets_ref, _ = go(plain)
    assert resets_ref == 0, "control should not reset the proposal"
    ln_z, n_eff, resets, warned = go(with_empty)

    assert resets == 0, "the empty key still made the proposal refit fail %d time(s)" % resets
    assert warned, "dropping the key must be reported -- it means a parameter name went unmatched"
    assert ln_z == ln_z_ref and n_eff == n_eff_ref, \
        "dropping the empty key must change nothing: %r/%r vs %r/%r" % (ln_z, n_eff, ln_z_ref, n_eff_ref)


def test_public_validate_wrapper_is_live():
    """The driver calls mcsamplerEnsemble.validate_dim_group_cover() at startup so a bad mapping
    fails before analyze_event's broad `except` can turn it into an empty .dat and exit 0.  Making
    that wrapper a no-op has to fail something."""
    with pytest.raises(ValueError):
        mcsamplerEnsemble.validate_dim_group_cover({(0, 1): None}, 3)      # dim 2 uncovered
    with pytest.raises(ValueError):
        mcsamplerEnsemble.validate_dim_group_cover({(0,): None, (0, 1): None}, 2)   # dim 0 twice
    mcsamplerEnsemble.validate_dim_group_cover({(0, 1): None}, 2)          # complete: accepted


def test_integrator_init_guard_is_live():
    """The guard in monte_carlo.integrator.__init__ protects every caller that is NOT
    mcsamplerEnsemble.setup()/integrate() -- CIP, EOS, and the warm-start write-back, which
    deliberately steps over it.  Without a test here, deleting that call site is invisible."""
    bounds = {(0, 1): np.array([[0.0, 1.0], [0.0, 1.0]])}
    with pytest.raises(ValueError):
        monte_carlo.integrator(3, bounds, {(0, 1): None}, 1, n=10)      # dim 2 in no group
    with pytest.raises(ValueError):
        monte_carlo.integrator(2, bounds, {(0, 1): None, (1,): None}, 1, n=10)   # dim 1 twice
    # the matching complete cover is accepted
    monte_carlo.integrator(2, bounds, {(0, 1): None}, 1, n=10)


def test_complete_dim_group_cover_default_values():
    """The VALUES the fill writes, not just the key sets.

    `adapt_default` in particular is load-bearing: the driver passes bool(--force-adapt-all)
    because the BIC (`gmm_adaptive`) block sizes exactly the groups gmm_adapt marks adapting,
    and it runs BEFORE the force-adapt-all sweep.
    """
    gd, comp, adapt = {(0, 1): None}, {(0, 1): 4}, {(0, 1): True}
    mcsamplerEnsemble.complete_dim_group_cover(gd, 3, comp_dict=comp, gmm_adapt=adapt)
    assert comp[(2,)] == 1 and adapt[(2,)] is False, (comp, adapt)
    gd2, comp2, adapt2 = {(0, 1): None}, {(0, 1): 4}, {(0, 1): True}
    mcsamplerEnsemble.complete_dim_group_cover(gd2, 3, comp_dict=comp2, gmm_adapt=adapt2,
                                               n_comp_default=5, adapt_default=True)
    assert comp2[(2,)] == 5 and adapt2[(2,)] is True, (comp2, adapt2)
