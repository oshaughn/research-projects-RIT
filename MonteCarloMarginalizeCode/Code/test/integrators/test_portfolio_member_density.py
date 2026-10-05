"""A portfolio member must report its sampling density as a NORMALIZED density.

THE DEFECT.  mcsamplerPortfolio builds the balance-heuristic mixture denominator
q_mix = sum_m frac_m q_m from each member's sampling_density(X).  mcsamplerGPU did not
implement that method, so every portfolio containing one fell back to the legacy STRATIFIED
denominator, which uses each sample's own member joint_p_s from draw_simplified().  That is
unbiased only if every member's joint_p_s is a normalized density -- and mcsamplerAdaptiveVolume
deliberately reports V_s/V, which is V_s**2 times the density it draws from, compensating inside
its own integrate_log.  Mixing the two pooled weights on scales differing by V_s**2.

    constant likelihood L = 1 on [-1,1]^2, prior_pdf == 1, exact ln Z = ln V = 1.386294
        [AV]                       1.386294      (exact, then and now)
        [adaptive_cartesian_gpu]   1.386294      (exact, after PR #354)
        [AV, adaptive_cartesian_gpu]
                                   0.753772      <- before this fix
                                   1.386294      <- after

0.753772 is not a mysterious number: half the 2000 draws came from AV with weight
prior/p_s = 1/(V_s/V) = 0.25 and half from GPU with weight V = 4, and log((0.25+4)/2) =
0.7537718.  That arithmetic is what identifies this as a member-contract problem rather than
either sampler's bug -- each is exact on its own scale.

SECOND DEFECT, same contract.  mcsamplerGPU swaps self.pdf[p] for the adaptive histogram
proposal, which compute_hist already normalizes to integrate to 1 over the box, but left
_pdf_norm[p] at the ANALYTIC pdf's mass.  pdf/_pdf_norm -- what draw(), draw_simplified() and
now sampling_density() all report -- was then too small by that mass once adaptation started
driving the draws, and ln Z came out HIGH by log(prod(_pdf_norm)).  Invisible to a run that
converges in one chunk, which is why the constant-integrand test above does not see it: it needs
an integrand demanding enough to adapt.  A caller passing an already-normalized pdf (production
CIP passes pdf = 1/range) has _pdf_norm == 1 and is unaffected either way.

WHAT IS NOT CHANGED.  mcsamplerAdaptiveVolume's draw_simplified still reports V_s/V.  Its
sampling_density() -- the contract -- already returned the true density, and its own evidence
must not move.
"""
import contextlib
import io

import numpy as np
import pytest

import RIFT.integrators.mcsamplerGPU as mcsamplerGPU
import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV
import RIFT.integrators.mcsamplerEnsemble as mcsamplerGMM
import RIFT.integrators.mcsamplerPortfolio as mcsamplerPortfolio

# a genuinely non-square box, so nothing here can be satisfied by tuning to V = 4
BOX = ((-1.0, 3.0), (-2.0, 0.5))
V_BOX = (BOX[0][1] - BOX[0][0]) * (BOX[1][1] - BOX[1][0])    # 10.0

CONSTANT_LNL = lambda *x: np.zeros(np.asarray(x[0]).shape)


class _UndeclaredNoDensity(mcsamplerGPU.MCSampler):
    """A member that has NOT declared what scale its joint_p_s is on -- a plugin pipeline from
    mcsamplerPortfolio.known_pipelines, or a new in-tree sampler whose author did not know the
    contract exists.

    Modelled by setting the attribute to None rather than by removing it, because a subclass
    cannot unset an inherited class attribute.  That is faithful to the production probe, which
    is `getattr(m, 'joint_p_s_is_normalized_density', None) is not True` -- absent and None take
    the same branch.  test_absent_and_undeclared_take_the_same_branch pins that equivalence."""
    sampling_density = None

    @property
    def joint_p_s_is_normalized_density(self):
        # Model a GENUINELY ABSENT attribute, not one set to a falsy value.  A subclass cannot
        # unset an inherited class attribute, and setting it to None is NOT faithful: a probe
        # written `getattr(m, name, True) is not True` -- the permissive default this test
        # exists to forbid -- also rejects None, so the test would pass with the defect present.
        # A property that raises AttributeError makes getattr(instance, name, default) return
        # the default, which is exactly what absence does.
        raise AttributeError("joint_p_s_is_normalized_density")


class _GPUNoDensity(mcsamplerGPU.MCSampler):
    """A member that never had sampling_density, which is the case the contract is about.

    Subclassing and setting the attribute to None is closer to a real density-less member than
    assigning a lambda over the instance: getattr(m, 'sampling_density', None) is what both the
    setup check and integrate_log actually read, and a bound method that returns None would also
    satisfy a hasattr() test that a genuinely absent method would not."""
    sampling_density = None


def _add_params(sampler, box=BOX):
    for p, (lo, hi) in zip(("xx", "yy"), box):
        sampler.add_parameter(p,
                              pdf=np.vectorize(lambda x: 1.0),
                              prior_pdf=np.vectorize(lambda x: 1.0),
                              left_limit=lo, right_limit=hi, adaptive_sampling=True)
    return sampler


def _portfolio(mods, box=BOX):
    s = mcsamplerPortfolio.MCSampler(portfolio=[m.MCSampler() for m in mods])
    _add_params(s, box=box)
    s.setup()
    return s


def _lnZ(mods, box=BOX, seed=4321):
    np.random.seed(seed)
    res = _portfolio(mods, box=box).integrate(
        CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
        use_lnL=True, return_lnI=True, save_intg=True,
        no_protect_names=True, verbose=False)
    return float(res[0])


# ---------------------------------------------------------------------------------------
# the mixture itself

@pytest.mark.parametrize("box", [BOX, ((-1.0, 1.0), (-1.0, 1.0))])
def test_mixed_backend_portfolio_is_exact_on_a_constant_integrand(box):
    """The headline.  A constant integrand has no fit error and no MC scatter, so a
    normalization error cannot hide behind noise: the answer is ln(prior mass), exactly.

    Two boxes, one of them non-square, because the pre-fix error was a function of the box
    (-0.632523 at V = 4, -0.683197 at V = 10) and a single volume cannot tell a correction from
    a coincidence."""
    V = (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])
    got = _lnZ([mcsamplerAV, mcsamplerGPU], box=box)
    assert np.isclose(got, np.log(V), atol=1e-6), \
        "mixed AV+GPU portfolio ln Z = %r on a constant integrand over a box of volume %r; " \
        "exact answer is ln V = %r" % (got, V, np.log(V))


def test_single_backend_portfolios_do_not_move():
    """The other direction, and the reason this is a fix rather than a regression: every
    single-backend portfolio was ALREADY exact, so it must be exactly where it was."""
    for mods, label in (([mcsamplerAV], "AV"), ([mcsamplerGPU], "GPU")):
        got = _lnZ(mods)
        assert np.isclose(got, np.log(V_BOX), atol=1e-6), \
            "%s-only portfolio ln Z = %r, exact ln V = %r" % (label, got, np.log(V_BOX))


def test_member_order_does_not_matter():
    """q_mix is symmetric in its members; the stratified fallback happened to be too, but only
    because the two members drew equal counts.  Pin the symmetry."""
    a = _lnZ([mcsamplerAV, mcsamplerGPU])
    b = _lnZ([mcsamplerGPU, mcsamplerAV])
    assert np.isclose(a, b, atol=1e-6), \
        "portfolio ln Z depends on member ORDER: %r vs %r" % (a, b)


def test_gpu_gmm_mixture_is_exact_too():
    """Not an AV-specific patch: mcsamplerEnsemble already implemented sampling_density, and a
    GPU+GMM portfolio could not form q_mix either, for the same missing method."""
    got = _lnZ([mcsamplerGMM, mcsamplerGPU])
    assert np.isclose(got, np.log(V_BOX), atol=1e-6), \
        "mixed GMM+GPU portfolio ln Z = %r, exact ln V = %r" % (got, np.log(V_BOX))


# ---------------------------------------------------------------------------------------
# the contract, stated as a property of each member

def test_gpu_sampling_density_integrates_to_one():
    """The contract: sampling_density must be a normalized density over the box.  Checked by
    quadrature on a grid, not by sampling, so this cannot be satisfied by an offsetting error."""
    s = _add_params(mcsamplerGPU.MCSampler())
    nx, ny = 401, 397
    xg = np.linspace(BOX[0][0], BOX[0][1], nx)
    yg = np.linspace(BOX[1][0], BOX[1][1], ny)
    XX, YY = np.meshgrid(xg, yg, indexing="ij")
    q = s.sampling_density(np.column_stack([XX.ravel(), YY.ravel()])).reshape(nx, ny)
    total = np.trapz(np.trapz(q, yg, axis=1), xg)
    assert np.isclose(total, 1.0, rtol=1e-5), \
        "mcsamplerGPU.sampling_density integrates to %r over its own box, not 1" % total


def test_gpu_sampling_density_matches_its_own_draws():
    """sampling_density(X) and draw_simplified()'s joint_p_s must be the SAME function.  A
    member whose two answers disagree makes q_mix inconsistent with the pool it denominates."""
    s = _add_params(mcsamplerGPU.MCSampler())
    p_s, _, rv = s.draw_simplified(5000, "xx", "yy")
    X = np.asarray(rv, dtype=float).T
    q = s.sampling_density(X)
    assert np.allclose(np.asarray(p_s, dtype=float), q, rtol=1e-8), \
        "sampling_density disagrees with draw_simplified's joint_p_s on the sampler's own draws"


def test_gpu_sampling_density_matches_its_own_draws_after_adaptation():
    """The same agreement, but with the HISTOGRAM proposal supplying the draws.

    The check above runs on a fresh sampler, where both paths read the analytic pdf.  Nothing
    there would notice if the two diverged once self.pdf[p] became pdf_from_hist -- which is
    exactly the state the second defect lived in, and exactly the state a long portfolio run
    spends almost all of its chunks in."""
    s = _add_params(mcsamplerGPU.MCSampler())
    peak = np.array([0.4, -0.8])
    lnL = lambda *x: -0.5*sum(((np.asarray(xi) - c)/0.25)**2 for xi, c in zip(x, peak))
    np.random.seed(7)
    s.integrate(lnL, "xx", "yy", n=1000, nmax=8000, neff=2000, use_lnL=True, return_lnI=True,
                save_intg=True, no_protect_names=True, verbose=False)
    # GUARD: if adaptation never fired, both paths still read the analytic pdf and this test
    # silently degenerates into the one above.  pdf_from_hist is wrapped in a local closure,
    # so identity against pdf_initial is the predicate that is actually available.
    assert all(s.pdf[p] is not s.pdf_initial[p] for p in s.params_ordered), \
        "the histogram proposal never replaced the analytic pdf; this test is not testing " \
        "what it claims"
    p_s, _, rv = s.draw_simplified(4000, "xx", "yy", save_no_samples=True)
    q = s.sampling_density(np.asarray(rv, dtype=float).T)
    assert np.allclose(np.asarray(p_s, dtype=float), q, rtol=1e-8), \
        "after adaptation, sampling_density and draw_simplified's joint_p_s disagree"


def test_gpu_sampling_density_is_zero_outside_the_box():
    """pdf_from_hist clamps its bin index, so a point outside the box would otherwise be given
    the edge bin's density.  cdf_inv cannot produce such a point, so the density there is 0."""
    s = _add_params(mcsamplerGPU.MCSampler())
    outside = np.array([[BOX[0][0] - 1.0, 0.0], [0.0, BOX[1][1] + 1.0], [99.0, 99.0]])
    q = s.sampling_density(outside)
    assert np.all(q == 0.0), "sampling_density is nonzero outside the box: %r" % (q,)


def test_AV_reported_p_s_is_left_alone():
    """AV's draw_simplified p_s is NOT a density and is deliberately not changed: its own
    integrate_log is written against that scale.  This test exists so a later 'cleanup' that
    normalizes it has to argue with a named decision instead of a silent convention."""
    s = _add_params(mcsamplerAV.MCSampler())
    s.setup()
    p_s, _, _ = s.draw_simplified(2000)
    reported = float(np.mean(np.asarray(p_s, dtype=float)))
    assert np.isclose(reported, V_BOX / s.V, rtol=1e-8), \
        "mcsamplerAdaptiveVolume.draw_simplified now reports %r, not V_s/V = %r; if that is " \
        "intended, its integrate_log and this decision both have to change" % (
            reported, V_BOX / s.V)
    # ... and its sampling_density, the actual contract, IS the density
    X = np.random.uniform([b[0] for b in BOX], [b[1] for b in BOX], size=(500, 2))
    q = s.sampling_density(X)
    assert np.allclose(q, 1.0 / (V_BOX * s.V), rtol=1e-8), \
        "mcsamplerAdaptiveVolume.sampling_density is not 1/(V_s*V)"


# ---------------------------------------------------------------------------------------
# the second defect: the adaptive histogram's normalization

def test_gpu_evidence_is_exact_once_adaptation_drives_the_draws():
    """Forces many chunks, so the histogram proposal -- not the analytic pdf -- supplies the
    draws.  compute_hist already normalizes, so _pdf_norm must become 1 when the swap happens;
    leaving it at the analytic pdf's mass made ln Z HIGH by log(prod(_pdf_norm)) (+1.386 on this
    box, measured).  A one-chunk run cannot see this."""
    lo, hi = -1.0, 3.0
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1.0),       # UNNORMALIZED -> _pdf_norm = 4
                    prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=lo, right_limit=hi, adaptive_sampling=True)
    lnL = lambda x: -0.5 * ((np.asarray(x) - 0.2) / 0.4) ** 2
    np.random.seed(11)
    res = s.integrate(lnL, "xx", n=1000, nmax=20000, neff=4000, use_lnL=True,
                      return_lnI=True, save_intg=True, no_protect_names=True, verbose=False)
    from scipy.special import erf
    exact = np.log(0.4 * np.sqrt(2 * np.pi) * 0.5 *
                   (erf((hi - 0.2) / 0.4 / np.sqrt(2)) - erf((lo - 0.2) / 0.4 / np.sqrt(2))))
    # TOLERANCE: this run is a real MC estimate, not a constant integrand, so it carries
    # sampling scatter.  Measured over seeds 0..9 on ldas-grid (numpy backend): std 0.0095 nats,
    # max |error| 0.0206.  0.10 sits ~5x above the worst of those and ~14x below the defect it
    # guards, log(4) = 1.386, so the two cannot be confused.
    assert abs(res[0] - exact) < 0.10, \
        "adaptive mcsamplerGPU ln Z = %r, exact %r (defect size log(_pdf_norm) = %r)" % (
            res[0], exact, np.log(4.0))
    for p in s.params_ordered:
        assert np.isclose(float(s._pdf_norm[p]), 1.0), \
            "_pdf_norm[%s] = %r after the histogram swap; pdf_from_hist is already a density" % (
                p, float(s._pdf_norm[p]))


def test_reset_sampling_restores_the_analytic_normalization():
    """_pdf_norm travels with the pdf.  reset_sampling puts the unnormalized analytic pdf back,
    so it must put its mass back too -- otherwise the density would be reported 4x too large
    here, in the other direction from the defect above."""
    lo, hi = -1.0, 3.0
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1.0), prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=lo, right_limit=hi, adaptive_sampling=True)
    norm_initial = float(s._pdf_norm["xx"])
    assert np.isclose(norm_initial, hi - lo, rtol=1e-6)
    s._pdf_norm["xx"] = 1.0          # as the histogram swap leaves it
    s.reset_sampling("xx")
    assert np.isclose(float(s._pdf_norm["xx"]), norm_initial, rtol=1e-12), \
        "reset_sampling restored the analytic pdf but left _pdf_norm at the histogram's 1.0"
    p_s, p_prior, _ = s.draw_simplified(20000, "xx")
    ratio = np.asarray(p_prior, dtype=float) / np.asarray(p_s, dtype=float)
    assert np.allclose(ratio, hi - lo, rtol=1e-6), \
        "after reset_sampling, E[prior/p_s] = %r, expected the prior integral %r" % (
            float(np.mean(ratio)), hi - lo)


# ---------------------------------------------------------------------------------------
# the refusal

def test_portfolio_refuses_AV_pooled_with_a_density_less_member():
    """The refusal exists for ONE configuration: a member with no sampling_density forces the
    stratified fallback, and AV's joint_p_s is not a density, so the pool mixes scales.

    Checked at setup(), which is before any likelihood evaluation -- the run should die cheaply,
    not after a chunk.  The match pins the MEMBER NAMES, not just the word 'sampling_density':
    that substring also appears in the pre-existing _has_restricted_member message, so a loose
    match cannot tell the two raises apart."""
    s = mcsamplerPortfolio.MCSampler(
        portfolio=[mcsamplerAV.MCSampler(), _GPUNoDensity()])
    _add_params(s)
    with pytest.raises(Exception, match=r"_GPUNoDensity.*mcsamplerAdaptiveVolume\.MCSampler"):
        s.setup()


def test_portfolio_allows_a_single_density_less_member():
    """A lone member that DECLARES its joint_p_s to be a density can take the stratified path
    safely, and refusing it aborted a configuration that returned the exact answer.

    Note the reason is the declaration, NOT the member count: the stratified estimator is not
    scale-invariant even with one member (a single AV on that path is wrong by ln(V_s**2)).  An
    earlier revision of this docstring claimed one member is its own mixture and therefore safe,
    which is false and is what produced the permissive default that review 2 measured at 0.68
    nats of silent bias."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity()])
    _add_params(s)
    s.setup()
    np.random.seed(4321)
    res = s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                      use_lnL=True, return_lnI=True, save_intg=True,
                      no_protect_names=True, verbose=False)
    assert np.isclose(float(res[0]), np.log(V_BOX), atol=1e-6), \
        "single density-less member: ln Z = %r, exact ln V = %r" % (float(res[0]), np.log(V_BOX))


def test_portfolio_allows_density_less_members_that_report_a_density():
    """Two members, neither with sampling_density, both reporting a normalized joint_p_s.  The
    stratified fallback is sound here and #354 measured it as such; refusing it would be a
    false alarm."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity(), _GPUNoDensity()])
    _add_params(s)
    s.setup()
    np.random.seed(4321)
    res = s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                      use_lnL=True, return_lnI=True, save_intg=True,
                      no_protect_names=True, verbose=False)
    assert np.isclose(float(res[0]), np.log(V_BOX), atol=1e-6), \
        "two density-less members: ln Z = %r, exact ln V = %r" % (float(res[0]), np.log(V_BOX))


def test_the_refusal_can_be_opted_out_of():
    """...and the opt-out still gives the OLD, biased answer, which is what makes it an opt-out
    rather than a second code path nobody checked."""
    s = mcsamplerPortfolio.MCSampler(
        portfolio=[mcsamplerAV.MCSampler(), _GPUNoDensity()])
    _add_params(s)
    # the opt-out has to reach BOTH gates: setup() refuses first, so an integrate()-only kwarg
    # would name an escape hatch that cannot be used.
    s.setup(portfolio_allow_stratified_density=True)
    np.random.seed(4321)
    res = s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                      use_lnL=True, return_lnI=True, save_intg=True,
                      no_protect_names=True, verbose=False,
                      portfolio_allow_stratified_density=True)
    # the stratified pool: half the draws at prior/p_s = V/V_s = 1/V, half at V
    expected_bad = np.log(0.5 * (1.0 / V_BOX) + 0.5 * V_BOX)
    assert np.isclose(float(res[0]), expected_bad, atol=1e-6), \
        "the opt-out path returned %r, not the legacy stratified value %r" % (
            float(res[0]), expected_bad)


def test_the_stratified_fallback_always_announces_itself():
    """The fallback is the one route a future non-density member could ride in on unnoticed, and
    the driver-level warning that used to cover it was removed.  It must print EVERY time it is
    taken, including on the explicit portfolio_use_mixture_density=False opt-out, which is
    otherwise the only fully silent path to the biased estimator."""
    for kw in ({}, {"portfolio_use_mixture_density": False}):
        s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity()])
        _add_params(s)
        s.setup()
        np.random.seed(4321)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                        use_lnL=True, return_lnI=True, save_intg=True,
                        no_protect_names=True, verbose=False, **kw)
        out = buf.getvalue()
        assert "STRATIFIED" in out and "_GPUNoDensity" in out, \
            "fallback with kwargs %r printed no stratified-density notice naming the member; got:"\
            "\n%s" % (kw, out[-800:])


def test_AV_declares_its_scale_and_the_others_do_not():
    """The refusal is gated on a DECLARATION, so the declaration has to be right.  If AV ever
    starts reporting a density, this flag must move with it or the guard silently stops firing."""
    assert mcsamplerAV.MCSampler.joint_p_s_is_normalized_density is False
    for mod in (mcsamplerGPU, mcsamplerGMM):
        assert getattr(mod.MCSampler, "joint_p_s_is_normalized_density", True) is True, \
            "%s now declares a non-density joint_p_s" % mod.__name__


def test_sampling_density_works_in_one_dimension():
    """ndim == 1 had no coverage at all, and under the contract an unhandled shape is a
    run-killing abort rather than a fallback."""
    lo, hi = -1.0, 3.0
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1.0), prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=lo, right_limit=hi, adaptive_sampling=True)
    xg = np.linspace(lo, hi, 20001)
    q = s.sampling_density(xg.reshape(-1, 1))
    assert q is not None, "sampling_density returned None for a 1-D sampler"
    assert np.isclose(np.trapz(q, xg), 1.0, rtol=1e-5), \
        "1-D sampling_density integrates to %r, not 1" % float(np.trapz(q, xg))
    p_s, _, rv = s.draw_simplified(4000, "xx")
    assert np.allclose(np.asarray(p_s, dtype=float),
                       s.sampling_density(np.asarray(rv, dtype=float).reshape(-1, 1)), rtol=1e-8)


def test_sampling_density_accepts_the_transposed_shape():
    """sampling_density tolerates (ndim, N) as well as (N, ndim).  Nothing in tree calls it that
    way, so without this the branch is dead code that could rot unnoticed."""
    s = _add_params(mcsamplerGPU.MCSampler())
    X = np.column_stack([np.linspace(BOX[0][0], BOX[0][1], 50),
                         np.linspace(BOX[1][0], BOX[1][1], 50)])       # (50, 2)
    assert np.allclose(s.sampling_density(X), s.sampling_density(X.T), rtol=1e-12), \
        "sampling_density disagrees between (N,ndim) and (ndim,N) input"


def test_supplied_cdf_inv_with_unnormalized_pdf_is_the_known_hole():
    """THE CONTRACT IS NOT SELF-ENFORCING, and this pins how far it reaches.

    _pdf_norm is only populated when add_parameter builds the CDF itself.  When the caller
    supplies cdf_inv -- what ILE does -- it stays 1, and sampling_density is normalized only if
    the caller's pdf already was.  Every in-tree cdf_inv caller passes a normalized pdf, so this
    is latent.  The test asserts the CURRENT behaviour so that closing the hole is a deliberate
    change with a failing test, rather than something a later reader assumes is already true."""
    lo, hi = -1.0, 3.0
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1.0),           # UNNORMALIZED, mass 4
                    cdf_inv=np.vectorize(lambda p: lo + (hi - lo) * p),   # caller-supplied
                    prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=lo, right_limit=hi, adaptive_sampling=True)
    assert np.isclose(float(s._pdf_norm["xx"]), 1.0), \
        "_pdf_norm was populated after all; this hole may be closed and the test should change"
    xg = np.linspace(lo, hi, 20001)
    total = float(np.trapz(s.sampling_density(xg.reshape(-1, 1)), xg))
    assert np.isclose(total, hi - lo, rtol=1e-5), \
        "supplied-cdf_inv sampling_density integrates to %r; expected the caller's pdf mass %r" % (
            total, hi - lo)



def _attach(s, member, breakpoint_=0):
    """Add a member to a portfolio that has ALREADY been set up, so setup()'s roster check
    cannot see it.  This is the state the run-time backstop exists for."""
    for p, (lo, hi) in zip(("xx", "yy"), BOX):
        member.add_parameter(p, pdf=np.vectorize(lambda x: 1.0),
                             prior_pdf=np.vectorize(lambda x: 1.0),
                             left_limit=lo, right_limit=hi, adaptive_sampling=True)
    member.setup()
    s.portfolio_realizations.append(member)
    s.portfolio_weights = np.ones(len(s.portfolio_realizations))
    s.portfolio_breakpoints = np.append(np.zeros(len(s.portfolio_realizations) - 1), breakpoint_)
    return s

def test_absent_and_undeclared_take_the_same_branch():
    """The stand-in above sets the attribute to None; a real undeclared member simply lacks it.
    The production probe must not distinguish them, or the tests below prove nothing about the
    plugin case they exist for."""
    class _Bare(object):
        pass
    sentinel = object()
    assert getattr(_Bare(), "joint_p_s_is_normalized_density", sentinel) is sentinel
    assert getattr(_UndeclaredNoDensity(), "joint_p_s_is_normalized_density", sentinel) is sentinel
    # ... and the permissive default must not rescue it either: this is the shape that made a
    # None-valued stand-in unable to see the defect.
    assert getattr(_UndeclaredNoDensity(), "joint_p_s_is_normalized_density", True) is True


def test_setup_refuses_a_member_that_never_declared_its_scale():
    """DEFAULT TO REFUSING.  An undeclared member is unknown-scale, not known-good.  Treating
    absence as 'normalized' made this pool return 1.619388 where the exact answer is 2.302585 --
    silently, because the fallback notice is a caveat rather than a detection."""
    s = mcsamplerPortfolio.MCSampler(
        portfolio=[_UndeclaredNoDensity(), _UndeclaredNoDensity()])
    _add_params(s)
    with pytest.raises(Exception, match=r"_UndeclaredNoDensity"):
        s.setup()


def test_setup_calls_the_optout_biased_and_not_sound():
    """On the opt-out the setup notice is the ONLY thing a caller sees, and it used to certify
    the biased configuration as 'Sound here because every member reports a normalized
    joint_p_s' -- which is false exactly when AV is the other member."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[mcsamplerAV.MCSampler(), _GPUNoDensity()])
    _add_params(s)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.setup(portfolio_allow_stratified_density=True)
    out = buf.getvalue()
    assert "BIASED" in out, "opt-out setup printed no bias warning; got:\n%s" % out[-600:]
    assert "Sound here" not in out, "opt-out setup still certifies the biased pool as sound"


def test_setup_announces_a_sound_stratified_fallback():
    """The other half: when every member DOES declare a normalized joint_p_s the fallback is
    unbiased, and setup should say so rather than staying silent about running the weaker
    estimator."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity(), _GPUNoDensity()])
    _add_params(s)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s.setup()
    out = buf.getvalue()
    assert "_GPUNoDensity" in out and "weaker estimator" in out, \
        "setup did not announce the sound stratified fallback; got:\n%s" % out[-600:]


def test_runtime_backstop_fires_when_the_member_list_changes_after_setup():
    """setup() cannot see a member added afterwards, so the run-time check is the only thing
    between that and a biased evidence.  Nothing pinned it before review 2: deleting the raise
    outright, and swapping its _chunk_members read for portfolio_realizations, both passed."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity()])
    _add_params(s)
    s.setup()                                   # legal: declared, so the fallback is sound
    _attach(s, mcsamplerAV.MCSampler())         # ... and now it is not
    np.random.seed(4321)
    with pytest.raises(Exception, match=r"mcsamplerAdaptiveVolume"):
        s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=4000, neff=30,
                    use_lnL=True, return_lnI=True, save_intg=True,
                    no_protect_names=True, verbose=False)


def test_runtime_backstop_also_defaults_an_undeclared_member_to_refusing():
    """The setup gate and the backstop must use the SAME default.  An UNDECLARED member reaching
    the backstop must stop the run; a permissive default there returns a number instead, which is
    the silent-bias shape review 2 measured at 0.68 nats."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity()])
    _add_params(s)
    s.setup()
    _attach(s, _UndeclaredNoDensity())
    np.random.seed(4321)
    with pytest.raises(Exception, match=r"_UndeclaredNoDensity"):
        s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=4000, neff=30,
                    use_lnL=True, return_lnI=True, save_intg=True,
                    no_protect_names=True, verbose=False)


def test_runtime_backstop_ignores_a_member_that_is_not_drawing():
    """The backstop reads _chunk_members -- the members that actually drew this chunk -- not the
    whole roster.  A member held behind an activation breakpoint contributes nothing to the pool,
    so it cannot skew it, and raising on its mere presence would abort a sound run.  Reading
    portfolio_realizations instead passed every other test in this file.

    The portfolio is built with both members and set up WITH the opt-out, because the roster gate
    would otherwise refuse it; the opt-out is then cleared to re-arm the backstop in isolation.
    Building it this way rather than appending a member after setup() keeps every per-member
    array (weights, quality, frozen flags) consistently sized."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity(), mcsamplerAV.MCSampler()])
    _add_params(s)
    s.setup(portfolio_allow_stratified_density=True)
    s._allow_stratified_density = False               # re-arm the run-time gate
    s.portfolio_breakpoints = np.array([0, 10 ** 9])  # AV never activates
    np.random.seed(4321)
    res = s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                      use_lnL=True, return_lnI=True, save_intg=True,
                      no_protect_names=True, verbose=False)
    assert np.isclose(float(res[0]), np.log(V_BOX), atol=1e-6), \
        "only the declared member drew, so the answer should be exact; got %r vs %r" % (
            float(res[0]), np.log(V_BOX))
    assert not any("AdaptiveVolume" in type(m).__module__
                   for m in getattr(s, "_chunk_members", [])), \
        "the un-activated AV member drew after all; this test is not testing what it claims"

def test_no_mixture_density_is_not_itself_consent_to_a_biased_pool():
    """portfolio_use_mixture_density=False selects the stratified MECHANISM; it is not a
    statement that the caller accepts a biased evidence.  It used to short-circuit the guard, so
    [AV,GMM] -- where BOTH members implement sampling_density, so setup sees nothing wrong --
    returned 1.619388 against an exact 2.302585 with no refusal and no opt-out given."""
    s = _portfolio([mcsamplerAV, mcsamplerGMM])
    np.random.seed(4321)
    with pytest.raises(Exception, match=r"portfolio_use_mixture_density=False"):
        s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=4000, neff=30,
                    use_lnL=True, return_lnI=True, save_intg=True,
                    no_protect_names=True, verbose=False,
                    portfolio_use_mixture_density=False)


def test_no_mixture_density_still_works_once_consent_is_given():
    """...and with the opt-out it runs and returns the legacy stratified value, so the escape
    hatch is a real path rather than a branch nobody exercised."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[mcsamplerAV.MCSampler(), mcsamplerGMM.MCSampler()])
    _add_params(s)
    s.setup(portfolio_allow_stratified_density=True)
    np.random.seed(4321)
    res = s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                      use_lnL=True, return_lnI=True, save_intg=True,
                      no_protect_names=True, verbose=False,
                      portfolio_use_mixture_density=False)
    assert np.isfinite(float(res[0]))


def test_setup_optout_alone_is_enough():
    """The opt-out is remembered by setup(), so a driver does not have to thread it through every
    integrate() call site -- integrate_likelihood_extrinsic_batchmode has five and its LISA twin
    four, and missing one turns the hatch back into an abort the caller cannot clear."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[mcsamplerAV.MCSampler(), _GPUNoDensity()])
    _add_params(s)
    s.setup(portfolio_allow_stratified_density=True)
    np.random.seed(4321)
    res = s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                      use_lnL=True, return_lnI=True, save_intg=True,
                      no_protect_names=True, verbose=False)      # NOT repeated here
    expected_bad = np.log(0.5 * (1.0 / V_BOX) + 0.5 * V_BOX)
    assert np.isclose(float(res[0]), expected_bad, atol=1e-6), \
        "setup-only opt-out returned %r, not the legacy stratified value %r" % (
            float(res[0]), expected_bad)


def test_the_fallback_notice_fires_on_every_pass_not_once_per_object():
    """The notice is latched so it prints once per pass rather than once per chunk.  The latch
    used to survive the pass, so a driver that integrates once per iteration announced the
    fallback for iteration 1 only, and an MC-error replica reusing the sampler announced
    nothing while running the same estimator."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity()])
    _add_params(s)
    s.setup()
    seen = []
    for _ in range(2):
        np.random.seed(4321)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                        use_lnL=True, return_lnI=True, save_intg=True,
                        no_protect_names=True, verbose=False)
        seen.append("STRATIFIED" in buf.getvalue())
    assert seen == [True, True], \
        "fallback notice printed on passes %r; it must fire on every pass" % (seen,)




def test_no_test_in_this_file_is_shadowed_by_a_duplicate():
    """Python silently rebinds a duplicated `def test_x`, so the earlier one vanishes and the
    collected count does not change.  Two rewrites of this file left stale copies that shadowed
    the new ones, and the suite reported the OLD behaviour as passing."""
    import collections
    import re as _re
    src = open(__file__.replace(".pyc", ".py")).read()
    names = _re.findall(r"(?m)^def (test_\w+)\(", src)
    dupes = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dupes, "duplicate test definitions shadow earlier ones: %r" % (dupes,)


def test_the_refusal_names_only_the_members_that_triggered_it():
    """A message that names innocent members is how rounds 2 and 3 of review were both failed,
    and substring matching cannot see it: `pytest.raises(match=...)` passes just as happily when
    every member is listed.  Assert the complement -- the declared-good member must NOT appear."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity()])
    _add_params(s)
    s.setup()
    _attach(s, mcsamplerAV.MCSampler())
    np.random.seed(4321)
    with pytest.raises(Exception) as exc:
        s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=4000, neff=30,
                    use_lnL=True, return_lnI=True, save_intg=True,
                    no_protect_names=True, verbose=False)
    msg = str(exc.value)
    assert "mcsamplerAdaptiveVolume" in msg, "the triggering member is not named: %s" % msg
    # _GPUNoDensity declares True; it is why q_mix could not be formed, which the message says
    # separately, but it must not be listed among the members whose SCALE is the problem.
    scale_clause = msg.split("normalized joint_p_s.")[-1]
    assert "_GPUNoDensity" not in scale_clause, \
        "a member that declared True is listed as a scale problem: %s" % scale_clause


def test_AV_is_called_declared_false_and_not_undeclared():
    """AV declares False.  Calling that 'not declared' and then advising the reader to declare
    it True would tell them to restore the exact bias this guard removes."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[_GPUNoDensity()])
    _add_params(s)
    s.setup()
    _attach(s, mcsamplerAV.MCSampler())
    np.random.seed(4321)
    with pytest.raises(Exception) as exc:
        s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=4000, neff=30,
                    use_lnL=True, return_lnI=True, save_intg=True,
                    no_protect_names=True, verbose=False)
    msg = str(exc.value)
    assert "DECLARE joint_p_s_is_normalized_density = False" in msg
    av_clause = [c for c in msg.split(".  ") if "mcsamplerAdaptiveVolume" in c]
    assert av_clause and "set joint_p_s_is_normalized_density = True" not in av_clause[0], \
        "the message advises declaring a False member True: %s" % (av_clause,)


def test_an_integrate_only_optout_is_honoured():
    """setup() remembering the opt-out must not stop integrate() from granting it: a library
    caller that opts out per pass would otherwise be ignored."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[mcsamplerAV.MCSampler(), _GPUNoDensity()])
    _add_params(s)
    s.setup(portfolio_allow_stratified_density=True)   # get past the roster gate
    s._allow_stratified_density = False                 # re-arm, as if setup never consented
    np.random.seed(4321)
    res = s.integrate(CONSTANT_LNL, "xx", "yy", n=2000, nmax=20000, neff=30,
                      use_lnL=True, return_lnI=True, save_intg=True,
                      no_protect_names=True, verbose=False,
                      portfolio_allow_stratified_density=True)
    expected_bad = np.log(0.5 * (1.0 / V_BOX) + 0.5 * V_BOX)
    assert np.isclose(float(res[0]), expected_bad, atol=1e-6)


def test_a_later_setup_can_take_the_consent_back():
    """Consent is stated per setup().  If it were inherited, a reconfigured portfolio could
    never re-arm the guard, and an opt-out given once would follow the object forever."""
    s = mcsamplerPortfolio.MCSampler(portfolio=[mcsamplerAV.MCSampler(), _GPUNoDensity()])
    _add_params(s)
    s.setup(portfolio_allow_stratified_density=True)
    assert s._allow_stratified_density is True
    with pytest.raises(Exception, match=r"mcsamplerAdaptiveVolume"):
        s.setup()            # no flag this time -> consent revoked -> roster gate refuses
