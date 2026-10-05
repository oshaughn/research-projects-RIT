"""mcsamplerGPU.draw_simplified() must report the density its draws actually come from.

THE DEFECT.  `draw_simplified()` built `joint_p_s` from the RAW `self.pdf[param]`, while the
samples themselves come from `cdf_inv[param]`, which is built from the NORMALIZED cdf.  So the
reported sampling density was too large by `prod(self._pdf_norm)` -- `_pdf_norm[param]` being
`cdf[-1]`, i.e. the integral of the supplied pdf over its range.  `draw()` in the same class has
always divided by `_pdf_norm` (see its `res.append` lines); only `draw_simplified()` did not, and
`integrate()` uses `draw_simplified()`.

CONSEQUENCE.  The estimator weight is `prior/p_s`, so a caller passing an UNNORMALIZED sampling
pdf got ln Z low by `log(prod(_pdf_norm))`.  With `util_ConstructEOSPosterior.py`, which passes
`pdf=lambda x: 1` and `prior_pdf=uniform_prior` (returns ones), that is exactly ln V:

    constant likelihood L = 1 on [-1,1]^2, V = 4, exact ln Z = ln 4 = 1.3862944
        AV / GMM / adaptive_cartesian   1.3862943      (all three, exactly)
        adaptive_cartesian_gpu          0.0000000      <- before this fix
        adaptive_cartesian_gpu          1.3862943      <- after

NOT A CONVENTION CHOICE.  `_pdf_norm` is 1 whenever the supplied pdf already integrates to 1, so
this is a no-op for every normalized-pdf caller; it only ever corrected a sampler that was
inconsistent with its own draws.  The tests below pin both halves -- the correction AND the
no-op -- because a "fix" that also moved normalized-pdf callers would change real evidences.
"""
import numpy as np
import pytest

import RIFT.integrators.mcsamplerGPU as mcsamplerGPU
import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV


def _build(sampler, pdf_value, prior_value, lo=-1.0, hi=1.0, params=("xx", "yy")):
    for p in params:
        sampler.add_parameter(p,
                              pdf=np.vectorize(lambda x, _v=pdf_value: _v),
                              prior_pdf=np.vectorize(lambda x, _v=prior_value: _v),
                              left_limit=lo, right_limit=hi, adaptive_sampling=True)
    return sampler


def test_reported_p_s_matches_the_density_drawn_from():
    """E[prior/p_s] over fresh draws must equal the integral of the prior, here V.

    This is the direct normalization identity, and it is what separates "the sampler applies a
    different convention" from "the sampler contradicts its own draws".  With pdf == 1 on a box
    of volume V the draws are uniform with density 1/V, so a p_s reported as the raw 1 makes
    this expectation come out V times too small.
    """
    V = 4.0
    s = _build(mcsamplerGPU.MCSampler(), 1.0, 1.0)
    p_s, p_prior, _ = s.draw_simplified(40000)
    ratio = np.asarray(p_prior, dtype=float) / np.asarray(p_s, dtype=float)
    # TOLERANCE.  _pdf_norm is cdf[-1] from a NUMERICALLY integrated cdf (odeint/lsoda, default
    # atol=rtol=1.49e-8), so the error on it is an ABSOLUTE ~atol and the RELATIVE error scales
    # as atol/_pdf_norm -- it is NOT a constant floor.  Measured at V = 4: E[prior/p_s] =
    # 3.99999998505, i.e. 1.49e-8 absolute and 3.74e-9 RELATIVE.  (An earlier revision of this
    # comment quoted the absolute figure as relative; that was wrong.)  rtol=1e-6 therefore sits
    # ~270x above the floor HERE, but it is configuration-specific: a box with _pdf_norm <~ 0.015
    # would make a CORRECT implementation fail this, so re-measure before reusing these bounds at
    # very small parameter scales.  The defect guarded is a factor of _pdf_norm (300% at V = 4),
    # so nothing plausible lands in between.
    assert np.allclose(ratio, V, rtol=1e-6), \
        "E[prior/p_s] = %r, expected the prior integral V = %r; draw_simplified is reporting a " \
        "density inconsistent with cdf_inv" % (float(np.mean(ratio)), V)


def test_draw_and_draw_simplified_agree_on_the_p_s_scale():
    """The two draw paths in ONE class must not disagree about what p_s means.

    draw() has always divided by _pdf_norm; draw_simplified() did not.  integrate() uses
    draw_simplified(), so the class's own two entry points disagreed by prod(_pdf_norm).
    """
    s = _build(mcsamplerGPU.MCSampler(), 1.0, 1.0)
    p_s_simpl, _, _ = s.draw_simplified(20000)
    mean_simpl = float(np.mean(np.asarray(p_s_simpl, dtype=float)))
    # pdf == 1 on [-1,1] per dimension -> _pdf_norm == 2 per dimension -> joint density 1/4
    expected = 1.0 / 4.0
    assert np.isclose(mean_simpl, expected, rtol=1e-6), \
        "draw_simplified reports p_s ~ %r; the normalized joint density is %r" % (mean_simpl,
                                                                                 expected)


def test_normalized_pdf_caller_is_unaffected():
    """The no-op half: _pdf_norm == 1 for a pdf that already integrates to 1.

    Every production caller that passes a proper density must see NO change -- otherwise this
    fix would silently move real evidences instead of correcting an inconsistent one.
    """
    s = _build(mcsamplerGPU.MCSampler(), 0.5, 0.5)   # pdf = prior = 1/2 per dim, both normalized
    for p in ("xx", "yy"):
        assert np.isclose(float(s._pdf_norm[p]), 1.0, rtol=1e-6), \
            "_pdf_norm[%s] = %r for an already-normalized pdf; the correction would not be a " \
            "no-op for normalized callers" % (p, float(s._pdf_norm[p]))
    p_s, p_prior, _ = s.draw_simplified(20000)
    # prior 1/2 per dim -> joint 1/4 ; p_s likewise 1/4 ; ratio 1 (the prior integrates to 1)
    ratio = np.asarray(p_prior, dtype=float) / np.asarray(p_s, dtype=float)
    assert np.allclose(ratio, 1.0, rtol=1e-6), \
        "normalized-pdf caller sees E[prior/p_s] = %r, expected 1" % float(np.mean(ratio))


@pytest.mark.parametrize("box,V", [(((-1.0, 1.0), (-3.0, 3.0)), 12.0),
                                   (((-2.0, 3.0), (0.0, 1.0)), 5.0)])
def test_constant_likelihood_evidence_is_exact_and_matches_AV(box, V):
    """End to end on a case with an exact answer, at two different volumes.

    A constant integrand removes fit error and MC scatter entirely: every sampler must return
    ln(integral of prior) = ln V, to machine precision.  Two volumes, because a single one
    cannot distinguish a genuine correction from a coincidence.

    BOTH BOXES ARE NON-SQUARE, deliberately.  With [-R,R] in every dimension all the per-
    parameter _pdf_norm values are equal, so a fix that reuses ONE parameter's norm for all of
    them, or takes max() over them, is indistinguishable from the correct per-parameter product.
    On [-1,1] x [-3,3] those two give ln 4 and ln 36 against the correct ln 12.
    """
    lnL = lambda *x: np.zeros(np.asarray(x[0]).shape)
    out = {}
    for name, mod in (("GPU", mcsamplerGPU), ("AV", mcsamplerAV)):
        s = mod.MCSampler()
        for pname, (lo, hi) in zip(("xx", "yy"), box):
            s.add_parameter(pname, pdf=np.vectorize(lambda x: 1),
                            prior_pdf=np.vectorize(lambda x: 1.0),
                            left_limit=lo, right_limit=hi, adaptive_sampling=True)
        res = s.integrate(lnL, "xx", "yy", n=2000, nmax=20000, neff=30,
                          use_lnL=True, return_lnI=True, save_intg=True,
                          no_protect_names=True, verbose=False)
        out[name] = float(res[0])
    assert np.isclose(out["GPU"], np.log(V), atol=1e-6), \
        "mcsamplerGPU ln Z = %r on a constant integrand, exact answer ln V = %r" % (out["GPU"],
                                                                                   np.log(V))
    assert np.isclose(out["GPU"], out["AV"], atol=1e-6), \
        "mcsamplerGPU %r and mcsamplerAdaptiveVolume %r disagree on a constant integrand" % (
            out["GPU"], out["AV"])


@pytest.mark.parametrize("nmax,chunks,atol", [(10000, 5, 0.03), (40000, 20, 0.02)])
def test_evidence_is_exact_AFTER_ADAPTATION(nmax, chunks, atol):
    """The evidence must stay exact once the adapted proposal is in use -- MULTIPLE CHUNKS.

    THIS IS THE AXIS THE REST OF THIS FILE MISSES, and missing it hid a real defect.  Every
    other test here either calls draw_simplified() directly or integrates a constant with
    neff=30, which clears neff on the FIRST chunk -- so `self.pdf[p]` is still the caller's
    original function throughout, and only the un-adapted regime is ever exercised.

    From the second chunk on, `self.pdf[p]` has been REPLACED by pdf_from_hist (at the three
    install sites in update_sampling_prior/integrate_log/integrate), which is already a density:
    compute_hist normalizes to sum 1 and divides by the bin width, and cdf_inverse_from_hist
    draws from that same normalized cdf.  So `_pdf_norm[p]` -- the integral of the pdf the
    CALLER supplied -- is stale from that point, and it must be reset to 1 alongside each
    install.  Without that reset, a `/_pdf_norm` in draw_simplified() is an error rather than a
    correction, and the integral converges to ln V + log(prod(_pdf_norm)) instead of ln V.

    Measured on a non-square box [-2,3] x [0,1] (V = 5, exact ln Z = 1.6094379):

        chunks   base      dividing-without-reset   with the reset
             1   0.000000            1.609438            1.609438
             5   1.432952            3.046184            1.609690
            20   1.567732            3.176679            1.610915
           100   1.600740            3.210876            1.609291

    TOLERANCES.  Twice re-measured, because both earlier figures were optimistic -- first from 6
    seeds (which cannot estimate a maximum at all), then from 300/200 seeds whose maxima were
    still under-reported.  Independently re-derived, same seed ranges:

        5 chunks, 300 seeds : sd 0.00262, max|err| 0.00709, 0/300 exceed atol 0.03 -> 4.23x
       20 chunks, 200 seeds : sd 0.00117, max|err| 0.00358, 0/200 exceed atol 0.02 -> 5.59x

    (Earlier revisions claimed sd 0.00236/0.00113, maxima 0.00660/0.00305 and margins 4.5x/6.6x.
    A 120-seed run already exceeded both of those quoted 300/200-seed maxima.)  max/sd is 2.71
    and 3.07 -- clean gaussians, no tail.

    HEADROOM OVER THE DEFECTS, measured on the true pre-PR base: 0.174404 at 5 chunks, so atol
    0.03 sits 5.81x below it; but only 0.040784 at 20 chunks, so atol 0.02 sits **2.04x** below
    -- NOT the "~20x" an earlier revision claimed and then re-asserted inside the sentence that
    was correcting it.  The 20-chunk case is the tight one; treat it as ~2x, not a wide margin.
    Against the unreset division the margins are 47.8x and 78.4x.

    Known blind spot, measured: a post-adaptation density misreport of up to ~x1.0075 per
    dimension (0.015 nats in 2-D) passes these bounds.  That is the honest floor of an MC test
    here; the reset VALUE is pinned by the structural test instead.
    """
    lnL = lambda *x: np.zeros(np.asarray(x[0]).shape)
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1), prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=-2.0, right_limit=3.0, adaptive_sampling=True)
    s.add_parameter("yy", pdf=np.vectorize(lambda x: 1), prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=0.0, right_limit=1.0, adaptive_sampling=True)
    res = s.integrate(lnL, "xx", "yy", n=2000, nmax=nmax, neff=1e9,
                      use_lnL=True, return_lnI=True, save_intg=True,
                      no_protect_names=True, verbose=False, n_adapt=100, tempering_adapt=True)
    lnV = np.log(5.0)
    assert np.isclose(float(res[0]), lnV, atol=atol), \
        "over %d chunks ln Z = %r, exact ln V = %r; a stale _pdf_norm after adaptation drives " \
        "this toward ln V + log(prod(_pdf_norm))" % (chunks, float(res[0]), lnV)


def test_pdf_norm_is_reset_when_the_adapted_proposal_is_installed():
    """Direct structural check of the invariant behind the test above.

    Kept separate because the integral tests see only the CONSEQUENCE, and the two catch
    genuinely different things: a reset with a WRONG VALUE (0.999 rather than 1.0) is invisible
    to the integral tests and caught only here, while a reset that is transiently absent and
    restored before the function returns is caught only by them.

    Its reach is limited and was overstated in an earlier revision: this test drives
    integrate(use_lnL=True) -> integrate_log() and therefore only ONE of the three install
    sites.  A fourth install site added to update_sampling_prior would NOT be caught here.  The
    other two sites have their own tests below; anyone adding a fourth should add one too.
    """
    lnL = lambda *x: np.zeros(np.asarray(x[0]).shape)
    s = mcsamplerGPU.MCSampler()
    for p, (lo, hi) in (("xx", (-2.0, 3.0)), ("yy", (0.0, 1.0))):
        s.add_parameter(p, pdf=np.vectorize(lambda x: 1), prior_pdf=np.vectorize(lambda x: 1.0),
                        left_limit=lo, right_limit=hi, adaptive_sampling=True)
    # before adaptation _pdf_norm is the supplied pdf's integral, i.e. the range width
    assert np.isclose(float(s._pdf_norm["xx"]), 5.0, rtol=1e-6)
    s.integrate(lnL, "xx", "yy", n=2000, nmax=10000, neff=1e9, use_lnL=True, return_lnI=True,
                save_intg=True, no_protect_names=True, verbose=False, n_adapt=100,
                tempering_adapt=True)
    for p in ("xx", "yy"):
        assert np.isclose(float(s._pdf_norm[p]), 1.0, rtol=1e-6), \
            "_pdf_norm[%s] = %r after adaptation; pdf_from_hist is already normalized, so it " \
            "must be 1 or every consumer of _pdf_norm is working from a stale value" % (
                p, float(s._pdf_norm[p]))


def test_non_constant_pdf_pins_the_draws_not_just_the_algebra():
    """With a NON-CONSTANT sampling pdf, E[prior/p_s] still equals the prior integral.

    THIS IS THE TEST THAT PINS THE FILE'S HEADLINE CLAIM.  Every other check here uses a
    constant pdf, and with a constant pdf `p_s` takes the same value at every sample, so
    `prior/p_s` is an algebraic identity that holds no matter WHERE the samples landed.  A
    sampler that drew from an entirely wrong distribution while still reporting `pdf/_pdf_norm`
    would pass all of them.

    This test is not a complete guard on its own and the docstring should not pretend otherwise:
    an earlier revision quoted "53.96" for a middle-10% truncation, which was wrong -- measured,
    that mutation gives 1.56 read in quantile space and 2.00 in x space, and the x-space reading
    PASSES here (the adaptation tests are what catch it).

    Its detection floor, measured against the distortion family q ~ p_s^(1+e): the statistic
    reaches this test's 0.05 threshold at e ~ 0.10, which is a ~16% peak relative density error
    and ~1.7% total variation.  (An earlier revision called that "a ~2.5% bias in the draw
    distribution"; 2.5% was just atol/target, a property of the statistic and not of the draws.)

    With pdf(x) = 1 + 0.8x the reported density genuinely varies with position, so the identity
    E_{p_s}[prior/p_s] = int prior dx = (hi - lo) holds only if the draws really are distributed
    as the reported p_s.

    Tolerance is MC, not numerical: the standard error over 40000 draws is 0.0061, measured
    across three seeds (means 1.99507, 2.00169, 2.00102), so 0.05 is ~8 sigma.
    """
    lo, hi = -1.0, 1.0
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1.0 + 0.8 * x),
                    prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=lo, right_limit=hi, adaptive_sampling=True)
    p_s, p_prior, _ = s.draw_simplified(40000)
    ratio = np.asarray(p_prior, dtype=float) / np.asarray(p_s, dtype=float)
    assert np.isclose(float(np.mean(ratio)), hi - lo, atol=0.05), \
        "E[prior/p_s] = %r for a non-constant pdf, expected the prior integral %r; the draws do " \
        "not follow the density draw_simplified reports" % (float(np.mean(ratio)), hi - lo)


def test_a_non_adaptive_parameter_is_corrected_too():
    """The correction must not be conditional on `adaptive_sampling`.

    Restricting it to `param in self.adaptive` passes every other test here, because they make
    every parameter adaptive.  A non-adaptive parameter keeps the caller's pdf for the whole run,
    so it is precisely the one that always needs the normalization.
    """
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1), prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=-2.0, right_limit=3.0, adaptive_sampling=False)
    p_s, p_prior, _ = s.draw_simplified(20000)
    ratio = np.asarray(p_prior, dtype=float) / np.asarray(p_s, dtype=float)
    assert np.allclose(ratio, 5.0, rtol=1e-6), \
        "E[prior/p_s] = %r on a NON-adaptive parameter, expected the prior integral 5.0" % (
            float(np.mean(ratio)))


def test_draw_agrees_with_draw_simplified():
    """draw() and draw_simplified() must report p_s on the same scale.

    draw() is currently dead code -- nothing in the tree calls it (mcsamplerPortfolio and
    mcsampler have their own).  It is pinned anyway for two reasons: this file's own rationale
    is that the two paths must agree, and a double-correction applied to draw() would otherwise
    be invisible if anything ever revived it.
    """
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1), prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=-2.0, right_limit=3.0, adaptive_sampling=True)
    p_s_simpl = float(np.mean(np.asarray(s.draw_simplified(20000)[0], dtype=float)))
    out = s.draw(20000, "xx")
    p_s_draw = float(np.mean(np.asarray(mcsamplerGPU.identity_convert(out[0]), dtype=float)))
    assert np.isclose(p_s_simpl, p_s_draw, rtol=1e-6), \
        "draw_simplified reports p_s ~ %r and draw() reports ~ %r; the two paths disagree " \
        "about the normalization" % (p_s_simpl, p_s_draw)


def _ratio(sampler, n=20000):
    p_s, p_prior, _ = sampler.draw_simplified(n)
    return float(np.mean(np.asarray(p_prior, dtype=float) / np.asarray(p_s, dtype=float)))


def test_reset_also_happens_on_the_update_sampling_prior_path():
    """`update_sampling_prior` installs the adapted proposal too, and MUST reset _pdf_norm.

    THIS PATH IS NOT REACHED BY ANY OTHER TEST HERE.  Every other test that INTEGRATES calls
    integrate(use_lnL=True), which returns integrate_log() immediately, so only that ONE of the
    three install sites runs; the rest call draw_simplified() directly and reach none of them.  Verified by deleting each reset in turn: dropping the
    update_sampling_prior one leaves the whole file green while the reported density is wrong by
    prod(_pdf_norm) -- about +1.6 nats of evidence error.

    It is also the path mcsamplerPortfolio drives: the portfolio adapts a member by calling
    member.update_sampling_prior(...), so a portfolio containing this sampler runs exclusively
    through here.
    """
    lo, hi = -2.0, 3.0
    s = mcsamplerGPU.MCSampler()
    s.add_parameter("xx", pdf=np.vectorize(lambda x: 1), prior_pdf=np.vectorize(lambda x: 1.0),
                    left_limit=lo, right_limit=hi, adaptive_sampling=True)
    s.setup()                                    # builds the histogram state the update needs,
    s.draw_simplified(4000)                      # and populate _rvs so it has history
    assert np.isclose(_ratio(s), hi - lo, rtol=1e-6)
    n = len(np.asarray(mcsamplerGPU.identity_convert(s._rvs["xx"])).reshape(-1))
    # n_history MUST be realistic.  mcsamplerPortfolio passes int(history_mult*n) (~2000); with
    # n_history=1 compute_hist fits the histogram to a SINGLE point (peak/mean bin ratio ~99 on
    # a 100-bin grid) and E[prior/p_s] then carries 4.7-8.9% scatter -- which is an artifact of
    # the invocation, not a property of the sampler.  At a realistic n_history the scatter is
    # 0.13% and the behavioural assertion below is perfectly sharp: it fails on a density
    # misreported x1.3, on a proposal whose cdf_inv was never swapped, and on a 0.999 reset.
    # TILTED weights, not flat.  With np.zeros() the adapted histogram comes out essentially
    # uniform, and then a proposal installed WITHOUT swapping cdf_inv (draws still uniform,
    # density reported from the histogram) is very nearly self-consistent and slips through.
    # A tilt makes the adapted proposal genuinely non-uniform, so the draws and the reported
    # density have to agree for the identity below to hold.  Measured at this tilt: control
    # E[prior/p_s] = 5.02, an unswapped cdf_inv gives 13.18 (exact 5.0).
    xs = np.asarray(mcsamplerGPU.identity_convert(s._rvs["xx"])).reshape(-1)[:n]
    s.update_sampling_prior(20.0 * xs, n, tempering_exp=1.0)
    assert np.isclose(float(s._pdf_norm["xx"]), 1.0, rtol=1e-6), \
        "_pdf_norm not reset on the update_sampling_prior path: %r" % float(s._pdf_norm["xx"])
    # ...and the sampler must actually BE coherent afterwards, not merely have had the
    # assignment executed: _pdf_norm alone cannot tell "the reset ran" from "the adapted
    # proposal and the draws agree".
    # 200k draws and rtol=0.03 (tolerance 0.15), both set from the DISTRIBUTION, not from one
    # observation: at this tilt the statistic has sd 0.0128 and max|err| 0.0313 over 12 runs at
    # 200k draws (it was sd 0.0341 / max 0.0722 at the 20k default, which made this test fail
    # 2 runs in 10 -- a tolerance fitted to a single measurement).  0.15 is ~12 sigma and ~4.8x
    # the observed maximum, while the mutants it must catch sit at 13.30 (unswapped cdf_inv)
    # and 3.85 (density x1.3), i.e. 55x and 7.7x outside it.
    ratio = _ratio(s, n=200000)
    assert np.isclose(ratio, hi - lo, rtol=0.03), \
        "after update_sampling_prior E[prior/p_s] = %r, expected the prior integral %r" % (
            ratio, hi - lo)


def test_reset_also_happens_on_the_linear_integrate_path():
    """integrate() WITHOUT use_lnL runs its own body, which installs the proposal separately.

    That is not an exotic branch: `--internal-use-lnL` is a store_true with no default while
    `--sampler-method` defaults to adaptive_cartesian_gpu, so an ordinary run reaches this code.
    Dropping the reset here leaves all the other tests green while the integral goes badly wrong.
    """
    lo, hi = -2.0, 3.0
    fn = lambda *x: np.ones(np.asarray(x[0]).shape)       # LINEAR likelihood, L = 1
    s = mcsamplerGPU.MCSampler()
    for p in ("xx", "yy"):
        s.add_parameter(p, pdf=np.vectorize(lambda x: 1), prior_pdf=np.vectorize(lambda x: 1.0),
                        left_limit=lo, right_limit=hi, adaptive_sampling=True)
    res = s.integrate(fn, "xx", "yy", n=2000, nmax=20000, neff=1e9,
                      save_intg=True, no_protect_names=True, verbose=False,
                      n_adapt=100, tempering_adapt=True)
    # This path returns the LINEAR integral, which must be the prior volume 5*5 = 25.  Measured
    # over 6 seeds: ln I - ln 25 = -0.0022 with the reset, +3.12 without it -- a 1400x margin,
    # free, on the path an ordinary run actually takes.
    assert np.isclose(np.log(float(res[0])), np.log(25.0), atol=0.05), \
        "linear integrate() gave ln I = %r, exact ln 25 = %r" % (np.log(float(res[0])),
                                                                 np.log(25.0))
    for p in ("xx", "yy"):
        assert np.isclose(float(s._pdf_norm[p]), 1.0, rtol=1e-6), \
            "_pdf_norm[%s] = %r after the linear integrate() path" % (p, float(s._pdf_norm[p]))


def test_three_parameters_are_all_corrected():
    """Every parameter, not just the first two.

    Correcting only the first two (or three) parameters passed every test in this file, because
    none of them used more than two.  util_ConstructEOSPosterior.py adds parameters in a loop
    over low_level_coord_names and routinely has more, so that mutation would have shipped.

    The three widths are 5, 1 and 3 -- all DIFFERENT, which matters and was got wrong once: an
    earlier revision used widths 5, 1, 5, and with two equal widths a permutation of the norms
    across parameters (an ordinary indexing bug) cancels and passes.
    """
    boxes = (("xx", -2.0, 3.0), ("yy", 0.0, 1.0), ("zz", -1.0, 2.0))
    V = float(np.prod([hi - lo for _, lo, hi in boxes]))
    s = mcsamplerGPU.MCSampler()
    for p, lo, hi in boxes:
        s.add_parameter(p, pdf=np.vectorize(lambda x: 1), prior_pdf=np.vectorize(lambda x: 1.0),
                        left_limit=lo, right_limit=hi, adaptive_sampling=True)
    assert np.isclose(_ratio(s), V, rtol=1e-6), \
        "E[prior/p_s] = %r over 3 parameters, expected the prior integral %r" % (_ratio(s), V)
