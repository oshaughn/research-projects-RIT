"""Log-space cubic-Hermite time quadrature.

The terminal time integral is the binding error term on the differentiable arm,
and the band-limited rule cannot fix it there: exp(lnL_t) is not band-limited
after a nonlinear (phi_ref, psi) marginalization, so FFT reconstruction is the
wrong operation and `_time_marginalize_terminal` refuses it.

This rule claims no band-limitedness.  It interpolates lnL in LOG space, where
the function is smooth and nearly quadratic near its peak, and integrates exp of
that interpolant with per-interval Gauss-Legendre.  The tests below pin that it
is far more accurate than Simpson in the regime that actually bites: a peak
narrower than the sample spacing.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from RIFT.likelihood.jax_ile.core import (  # noqa: E402
    _time_marginalize, _time_marginalize_log_hermite, _simpson_weights,
    _log_hermite_eval, _TIME_QUAD_CHOICES)


def _gaussian_case(h_over_sigma, npts=1025, srate=4096.0, offset_frac=0.3):
    """lnL(t) = -(t-t0)^2 / 2 sigma^2 with an OFF-GRID peak.

    Off-grid on purpose: a peak sitting exactly on a sample is the easy case and
    would flatter any rule.
    """
    deltaT = 1.0 / srate
    sigma = deltaT / h_over_sigma
    t = (np.arange(npts) - npts // 2) * deltaT
    t0 = offset_frac * deltaT
    lnL = -((t - t0) ** 2) / (2.0 * sigma ** 2)
    truth = 0.5 * np.log(2.0 * np.pi) + np.log(sigma)   # log integral, analytic
    return jnp.asarray(lnL[None, :]), deltaT, npts, truth


def test_registered_as_a_choice():
    assert "log-hermite" in _TIME_QUAD_CHOICES


# NOTE h/sigma <= 0.5 only.  h/sigma = 1.0 is NOT the resolved regime: Simpson is
# already 2.8e-3 nats off there, which this test caught when it was written with
# 1.0 included.  Asserting agreement with a rule that is itself wrong would have
# pinned the wrong tolerance and made the comparison meaningless.
@pytest.mark.parametrize("h_over_sigma", [0.25, 0.5])
def test_agrees_with_simpson_when_resolved(h_over_sigma):
    """Where Simpson is trustworthy the two must agree: same quantity, same norm."""
    y, deltaT, npts, truth = _gaussian_case(h_over_sigma)
    w = jnp.asarray(_simpson_weights(npts, deltaT))
    a = float(_time_marginalize(y, w)[0])
    b = float(_time_marginalize_log_hermite(y, deltaT)[0])
    assert abs(a - b) < 1e-6, (a, b)
    assert abs(b - truth) < 1e-6


@pytest.mark.parametrize("h_over_sigma,simpson_floor",
                         [(1.0, 1e-3), (2.0, 1e-2), (4.0, 1e-1)])
def test_beats_simpson_when_under_resolved(h_over_sigma, simpson_floor):
    """The regime that bites: peak narrower than the spacing.

    Asserts BOTH that Simpson is genuinely bad here (so the comparison is not
    against a straw man) and that log-hermite is orders better.
    """
    y, deltaT, npts, truth = _gaussian_case(h_over_sigma)
    w = jnp.asarray(_simpson_weights(npts, deltaT))
    e_s = abs(float(_time_marginalize(y, w)[0]) - truth)
    e_h = abs(float(_time_marginalize_log_hermite(y, deltaT)[0]) - truth)
    assert e_s > simpson_floor, "Simpson is not actually failing here; test is vacuous"
    assert e_h < e_s / 100.0, (e_s, e_h)


def test_jit_vmap_grad():
    """Must survive the three transforms the JAX likelihood composes."""
    y, deltaT, npts, truth = _gaussian_case(2.0)
    f = lambda z: _time_marginalize_log_hermite(z, deltaT)
    plain = float(f(y)[0])
    assert abs(float(jax.jit(f)(y)[0]) - plain) < 1e-12
    stacked = jnp.concatenate([y, y + 1.0, y - 2.0], axis=0)
    v = jax.jit(f)(stacked)
    assert v.shape == (3,)
    # the additive shifts must pass straight through a log integral
    assert abs(float(v[1] - v[0]) - 1.0) < 1e-10
    assert abs(float(v[2] - v[0]) + 2.0) < 1e-10
    g = jax.grad(lambda z: _time_marginalize_log_hermite(z, deltaT)[0])(y)
    assert np.all(np.isfinite(np.asarray(g)))
    # d/d(lnL) of a log integral is a normalized weight: it must sum to 1
    assert abs(float(jnp.sum(g)) - 1.0) < 1e-8


def test_shift_and_scale_invariances():
    """Two identities a log integral must satisfy exactly."""
    y, deltaT, npts, truth = _gaussian_case(1.5)
    base = float(_time_marginalize_log_hermite(y, deltaT)[0])
    shifted = float(_time_marginalize_log_hermite(y + 3.5, deltaT)[0])
    assert abs((shifted - base) - 3.5) < 1e-10
    doubled = float(_time_marginalize_log_hermite(y, 2.0 * deltaT)[0])
    assert abs((doubled - base) - np.log(2.0)) < 1e-10


# --------------------------------------------------------------------------
# Tests added by adversarial review of this file's own first version.  The
# original suite tested ONE Gaussian, which is the single easiest integrand for
# a cubic interpolant -- a fixture chosen, unintentionally, to flatter the
# method.  These pin the cases where the advantage is small or absent.
# --------------------------------------------------------------------------

# numpy 2 renamed trapz -> trapezoid and REMOVED the old name; CI installs numpy
# unpinned, and the two interpreters here disagree (IGWN 1.26.4 has only trapz,
# the dev env 2.4.6 has only trapezoid).  Written against one, this helper failed
# on the other -- caught only by running the suite under both.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _log_integral_reference(fn, t, n=400001):
    """Reference by dense trapezoid.  Converged: identical to 12 digits from
    n=1e5 to n=8e6 on the skewed case, checked rather than assumed."""
    tt = np.linspace(t[0], t[-1], n)
    g = fn(tt)
    m = g.max()
    return m + np.log(_TRAPZ(np.exp(g - m), tt))


def test_advantage_is_smaller_on_a_skewed_peak():
    """A skewed lnL is physical and much harder than the suite's Gaussian.

    Pins the ADVANTAGE, not just the direction: on the Gaussian log-hermite is
    ~4 orders better; skewed it is ~5x.  Quoting the Gaussian figure as if it
    were general would overstate the method.
    """
    npts, deltaT = 513, 1.0 / 4096
    t = (np.arange(npts) - npts // 2) * deltaT
    fn = lambda tt: -(tt ** 2) / (2 * (deltaT / 2) ** 2) - 8.0 * np.tanh(tt / deltaT)
    truth = _log_integral_reference(fn, t)
    y = jnp.asarray(fn(t)[None, :])
    w = jnp.asarray(_simpson_weights(npts, deltaT))
    e_s = abs(float(_time_marginalize(y, w)[0]) - truth)
    e_h = abs(float(_time_marginalize_log_hermite(y, deltaT)[0]) - truth)
    assert e_h < e_s, (e_s, e_h)
    assert e_h > e_s / 100.0, (
        "skewed advantage is now >100x; if that is real, update the claim in the "
        "docstring, which says the large factors are the smooth-Gaussian case")


def test_error_changes_sign_so_convergence_must_not_be_read_at_one_anchor():
    """Undershoots while resolved, overshoots once not -- zero-crossing ~h/sigma 3.

    A convergence test anchored near the crossing would report near-perfect
    accuracy for the wrong reason.
    """
    npts, deltaT = 513, 1.0 / 4096
    t = (np.arange(npts) - npts // 2) * deltaT
    errs = {}
    for hs in (2.0, 8.0):
        sigma = deltaT / hs
        y = jnp.asarray((-((t - 0.3 * deltaT) ** 2) / (2 * sigma ** 2))[None, :])
        truth = 0.5 * np.log(2 * np.pi) + np.log(sigma)
        errs[hs] = float(_time_marginalize_log_hermite(y, deltaT)[0]) - truth
    assert errs[2.0] < 0.0 < errs[8.0], errs


def test_non_finite_rows_are_handled_but_never_silently_wrong():
    """Four cases; the mixed -inf one FALLS BACK to Simpson rather than refusing.

    It used to return NaN.  That was wrong: a mixed -inf row is a well-defined
    integrand (the -inf samples contribute exactly zero), Simpson gets it right,
    and NaN turns a valid likelihood row into a fatal sampler value.  See
    test_mixed_neg_inf_wings_are_integrated_not_refused for the physical case.

    The fallback must be a real Simpson evaluation and NOT a clamp: flooring -inf
    to row_max-700 makes Catmull-Rom overshoot to +51.85 and return +43.10 where
    the truth is just below -6.24, and no floor depth is safe (the overshoot
    scales with it; a shallow floor contributes spuriously).  The equality below
    is what rules a clamp out.
    """
    deltaT = 1.0 / 4096
    finite = float(_time_marginalize_log_hermite(jnp.zeros((1, 9)), deltaT)[0])
    assert np.isfinite(finite)

    allneg = np.full((1, 9), -np.inf)
    assert float(_time_marginalize_log_hermite(jnp.asarray(allneg), deltaT)[0]) == -np.inf

    mixed = np.zeros((1, 9)); mixed[0, 3] = -np.inf
    w = jnp.asarray(_simpson_weights(9, deltaT))
    got = float(_time_marginalize_log_hermite(jnp.asarray(mixed), deltaT)[0])
    want = float(_time_marginalize(jnp.asarray(mixed), w)[0])
    assert not np.isnan(got), "a mixed -inf row is valid and must not be refused"
    assert abs(got - want) < 1e-12, (got, want,
        "the mixed -inf fallback must BE Simpson, not a floored Hermite: a "
        "row_max-700 floor returns +43.10 here where the truth is below -6.24")

    nan = np.zeros((1, 9)); nan[0, 3] = np.nan
    assert np.isnan(float(_time_marginalize_log_hermite(jnp.asarray(nan), deltaT)[0])), \
        "NaN must propagate; hiding a numerical failure is worse than the failure"

    pos = np.zeros((1, 9)); pos[0, 3] = np.inf
    assert float(_time_marginalize_log_hermite(jnp.asarray(pos), deltaT)[0]) == np.inf

    # a row mixing -inf with +inf is still a failure, and +inf must win
    both = np.zeros((1, 9)); both[0, 2] = -np.inf; both[0, 6] = np.inf
    assert float(_time_marginalize_log_hermite(jnp.asarray(both), deltaT)[0]) == np.inf


def test_mixed_neg_inf_wings_are_integrated_not_refused():
    """The physical case, and the reason the refusal had to go.

    Whenever the inner angle/distance marginalization underflows in the wings --
    the ORDINARY situation at high amplitude, which is the regime this rule exists
    to serve -- lnL_t arrives with -inf wings and a finite core.  Refusing that
    row converts a perfectly good likelihood into a fatal sampler value.

    The two rows below differ only in wings carrying 5.4e-32 of the peak, so the
    correct answer is the SAME analytic number for both, and Simpson gets it to
    1.1e-13 nats.  Asserted against the analytic truth, not merely against
    Simpson, so the test cannot pass by agreeing with a broken fallback.
    """
    npts, deltaT = 129, 1.0 / 4096
    sigma = 3 * deltaT
    t = (np.arange(npts) - npts // 2) * deltaT
    lnL = 800.0 - (t ** 2) / (2 * sigma ** 2)
    truth = 800.0 + 0.5 * np.log(2 * np.pi) + np.log(sigma)

    mixed = lnL.copy()
    mixed[np.abs(t) > 12 * sigma] = -np.inf
    assert np.isneginf(mixed).any() and np.isfinite(mixed).any(), "fixture is not mixed"

    all_finite = float(_time_marginalize_log_hermite(jnp.asarray(lnL[None, :]), deltaT)[0])
    got = float(_time_marginalize_log_hermite(jnp.asarray(mixed[None, :]), deltaT)[0])
    assert abs(all_finite - truth) < 1e-6, all_finite
    assert not np.isnan(got), "mixed -inf row was refused; it is valid and Simpson-exact"
    assert abs(got - truth) < 1e-6, (got, truth)

    # ...and it must survive the transforms the likelihood composes.
    f = lambda z: _time_marginalize_log_hermite(z, deltaT)
    stacked = jnp.asarray(np.stack([lnL, mixed]))
    v = np.asarray(jax.jit(f)(stacked))
    assert np.all(np.abs(v - truth) < 1e-6), v
    vv = np.asarray(jax.jit(jax.vmap(lambda z: f(z[None, :])[0]))(stacked))
    assert np.all(np.abs(vv - truth) < 1e-6), vv

    # A NEGATIVE peak, which is the case that discriminates.  Internally the
    # kernel substitutes 0.0 for -inf so the Hermite arithmetic cannot produce a
    # spurious value; with a peak at +800 that substitution is 800 nats DOWN and
    # therefore invisible, so a mutant that returned the substituted Hermite value
    # instead of Simpson would pass the fixture above.  At a peak of -50 the same
    # substitution is 50 nats UP -- a spike where the wings were -- and only a
    # genuine Simpson evaluation still lands on the truth.  (Caught by mutation
    # testing this file, not by writing it.)
    lo = -50.0 - (t ** 2) / (2 * sigma ** 2)
    lo_truth = -50.0 + 0.5 * np.log(2.0 * np.pi) + np.log(sigma)
    lo_mixed = lo.copy()
    lo_mixed[np.abs(t) > 12 * sigma] = -np.inf
    got_lo = float(_time_marginalize_log_hermite(jnp.asarray(lo_mixed[None, :]), deltaT)[0])
    assert abs(got_lo - lo_truth) < 1e-6, (got_lo, lo_truth)

    # The gradient of a log integral is a normalized weight: finite, zero on the
    # -inf samples, summing to 1.  The forward value cannot see the -inf -> 0.0
    # substitution; this can (removing it makes every entry NaN).
    for row in (mixed, lo_mixed):
        g = np.asarray(jax.grad(lambda z: f(z[None, :])[0])(jnp.asarray(row)))
        assert np.all(np.isfinite(g)), g
        assert np.all(g[np.isneginf(row)] == 0.0)
        assert abs(g.sum() - 1.0) < 1e-10, g.sum()


def test_all_neg_inf_row_has_a_finite_gradient():
    """-inf forward, and no NaN leaking back from the unselected Simpson branch."""
    deltaT = 1.0 / 4096
    row = jnp.full((33,), -jnp.inf)
    f = lambda z: _time_marginalize_log_hermite(z[None, :], deltaT)[0]
    assert np.isneginf(float(f(row)))
    g = np.asarray(jax.grad(f)(row))
    assert np.all(np.isfinite(g)), g


def test_n_sub_reaches_the_quadrature():
    """At h/sigma = 2 the per-interval rule is visibly unconverged at n_sub = 2
    (-8.093045 vs -8.091975 converged) and converged by 6.  A kernel that ignored
    n_sub would return the default-node value for every n_sub."""
    y, deltaT, npts, truth = _gaussian_case(2.0)
    v = {n: float(_time_marginalize_log_hermite(y, deltaT, n_sub=n)[0])
         for n in (2, 4, 6, 10)}
    assert abs(v[2] - v[4]) > 1e-4, v
    assert abs(v[6] - v[10]) < 1e-8, v
    assert abs(v[10] - truth) < 1e-4, (v, truth)


def test_quadratic_lnL_is_reproduced_far_better_than_a_cubic_one():
    """The corrected claim, pinned on a physical observable -- with a BOUNDED fixture.

    Finite differences are exact for a quadratic and not for a cubic, so a cubic
    Hermite segment fed them reproduces quadratics and NOT cubics.  A purely
    quadratic lnL is a Gaussian, which the rule integrates essentially exactly at
    h/sigma = 1; adding a cubic term at the same resolution must degrade it by
    orders of magnitude.

    THE FIXTURE IS THE POINT.  The first version of this test used npts = 513 with
    sigma = deltaT, so t/sigma spanned [-256, 256] and the +0.35 (t/sigma)^3 term
    reached lnL = 5.8e6 at the right edge.  ALL of the reference integral's mass
    then sat in the last few samples -- measured, the final five samples carried
    a fraction 1.000000 of it -- so `e_cubic` measured an exploding boundary layer
    rather than cubic reproduction near the peak.

    That was demonstrated, not assumed.  Re-running the old fixture with the same
    quadrature but CUBIC-EXACT endpoint slopes -- an interpolant that reproduces
    the integrand's cubic exactly, so the assertion should have nothing left to
    detect -- gives e_cubic = 4753 either way, identical to four figures, and the
    assertion passes in both worlds.  It could not tell them apart.  On the
    fixture below the same substitution moves e_cubic from 2.4e-07 to 5.0e-12 and
    the assertion correctly fails, so it is measuring what it claims to.

    Here the span is +-16 sigma and the cubic coefficient is 0.01, so the tail
    mass is ~1e-35 and the integral is genuinely peak-dominated.  That condition
    is asserted below rather than assumed, so the test cannot quietly go vacuous
    again.

    Deliberately NOT a convergence-order test: the error crosses zero near
    h/sigma ~ 3 (see the docstring), so a slope fitted across that region would be
    meaningless.  The order O(h^3.01) quoted there was measured on the interpolant
    directly, not end-to-end.
    """
    npts, deltaT = 33, 1.0 / 4096
    t = (np.arange(npts) - npts // 2) * deltaT
    sigma = deltaT                      # h/sigma = 1, the resolved regime

    quad = lambda tt: -(tt ** 2) / (2 * sigma ** 2)
    cubic = lambda tt: -(tt ** 2) / (2 * sigma ** 2) + 0.01 * (tt / sigma) ** 3

    # FIXTURE GUARD: both integrands must be peak-dominated, not boundary-dominated.
    for fn in (quad, cubic):
        wgt = np.exp(fn(t) - fn(t).max())
        tail = (wgt[:2].sum() + wgt[-2:].sum()) / wgt.sum()
        assert tail < 1e-10, (tail, "fixture is boundary-dominated; the "
                              "comparison below would be vacuous")

    e_quad = abs(float(_time_marginalize_log_hermite(jnp.asarray(quad(t)[None, :]), deltaT)[0])
                 - _log_integral_reference(quad, t))
    ref = _log_integral_reference(cubic, t)
    e_cubic = abs(float(_time_marginalize_log_hermite(jnp.asarray(cubic(t)[None, :]), deltaT)[0]) - ref)

    assert e_quad < 1e-9, e_quad
    assert e_cubic > 1e3 * e_quad, (e_quad, e_cubic,
                                    "cubic is now as accurate as quadratic; the "
                                    "docstring's reproduction claim needs revisiting")


def test_quadratic_is_reproduced_on_the_BOUNDARY_intervals_too():
    """The interpolant itself, at the ends, where the claim used to be false.

    The docstring says the interpolant reproduces quadratics exactly.  The
    endpoint slopes were two-point differences, which are only FIRST order: on
    f(x) = x^2 with h = 1 the left slope came out 1 where f'(0) = 0, and the
    worst error over all intervals was 1.5e-01 rather than 1.4e-14.  Interior
    intervals were fine, so a test that averaged over the grid, or one written
    with npts large enough that the ends carry no mass, would never have seen it.

    Measured through `_log_hermite_eval`, the function the kernel itself calls, so
    this cannot pass against a re-implementation that has drifted.
    """
    npts, h = 9, 1.0
    x = np.arange(npts) * h
    s = np.linspace(0.0, 1.0, 201)

    def worst(fn, intervals):
        p = np.asarray(_log_hermite_eval(jnp.asarray(fn(x)[None, :]), s))[0]
        exact = fn(x[:-1, None] + s[None, :] * h)
        e = np.abs(p - exact)
        return float(max(e[i].max() for i in intervals))

    boundary, interior = [0, npts - 2], list(range(1, npts - 2))
    for label, fn in (("linear", lambda z: 2 * z - 1), ("quadratic", lambda z: z ** 2)):
        assert worst(fn, boundary) < 1e-12, (label, "boundary", worst(fn, boundary))
        assert worst(fn, interior) < 1e-12, (label, "interior", worst(fn, interior))

    # ...and a cubic is still NOT reproduced, so the test is not passing because
    # the interpolant became exact for everything.
    cube = lambda z: z ** 3
    assert worst(cube, boundary) > 1e-2, worst(cube, boundary)
    assert worst(cube, interior) > 1e-2, worst(cube, interior)


def test_too_few_samples_is_refused_not_silently_degraded():
    """The three-point endpoint slopes need npts >= 3; below that, ValueError."""
    deltaT = 1.0 / 4096
    for npts in (1, 2):
        with pytest.raises(ValueError, match="at least 3 time samples"):
            _time_marginalize_log_hermite(jnp.zeros((1, npts)), deltaT)
    # 3 is the smallest accepted grid, and it must still be a real integral
    y = jnp.zeros((1, 3))
    got = float(_time_marginalize_log_hermite(y, deltaT)[0])
    want = float(_time_marginalize(y, jnp.asarray(_simpson_weights(3, deltaT)))[0])
    assert abs(got - want) < 1e-12, (got, want)


def test_real_jax_vmap_not_just_a_stacked_batch():
    """test_jit_vmap_grad stacks rows; that is not jax.vmap.  This is."""
    npts, deltaT = 257, 1.0 / 4096
    t = (np.arange(npts) - npts // 2) * deltaT
    y = jnp.asarray((-((t - 0.3 * deltaT) ** 2) / (2 * (deltaT / 2) ** 2))[None, :])
    stacked = jnp.concatenate([y, y + 1.0], axis=0)
    v = jax.vmap(lambda z: _time_marginalize_log_hermite(z[None, :], deltaT)[0])(stacked)
    assert v.shape == (2,)
    assert abs(float(v[1] - v[0]) - 1.0) < 1e-10
