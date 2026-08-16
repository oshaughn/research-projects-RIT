#!/usr/bin/env python
"""
Regression tests for the DISTANCE TAIL fix
(RIFT/interpolators/distance_tail.py, wired into bin/util_ConstructEOSPosterior.py
behind --fit-distance-tail).

Background (the bug these tests lock down).  In a distance-export (.dslice) run the
CIP fits lnL over the intrinsic coordinates PLUS distance, and the training set is
~50 discrete distances per intrinsic point.  The default fit is an ExtraTrees
ensemble, which is piecewise constant outside its training envelope: past a given
intrinsic point's outermost exported slice it returns that slice's lnL forever
instead of letting the likelihood decay.

That is not a small error, because the recovered distance posterior is
exp(lnL) * pi(d) and the distance prior is VOLUMETRIC -- it grows like d^2.  Holding
lnL flat therefore makes the integrand GROW out to whatever --d-max the CIP range
allows, so probability that should have died survives.  Measured across a 10-event
distance-export catalog, two seeds each: the recovered 90-10 distance width came out
+16.8% (median) too wide, in every quantile span including the core, with the median
distance itself untouched.  The fraction of recovered samples falling outside their
nearest grid point's exported support orders those events by width excess at Spearman
rho = +0.952, p = 4e-5.

The fix continues each slice past its edge with the exact single-angle form through
the origin, matched to the value AND slope the data has at that edge.  With
x = 1/d, u = x/x_edge, and s the dimensionless log-slope at the edge:

    lnL(u)/lnL_edge = (2-s) u + (s-1) u^2

so a slice that is still flat at its edge (the common case, because the
distance-inclination degeneracy keeps the marginalised lnL flat across the exported
support) rolls off as 2u - u^2 rather than being guillotined.

The properties below are the ones that make this safe to turn on, and each is a way
the fix could regress into a second extrapolation bug if someone edits the module:
  * it is a NO-OP on the support, so nothing that currently works changes;
  * it is CONTINUOUS at the edge, so the sampler sees no step to mistake for structure;
  * it is MONOTONE beyond the edge, so walking out in distance always walks lnL down;
  * it goes to ZERO as d -> infinity, which for a likelihood RATIO is an identity, not
    a modelling choice -- an infinitely distant source is exactly the noise hypothesis.
"""
import numpy as np
import pytest

from RIFT.interpolators.distance_tail import wrap_distance_tail

COORDS = ["mc", "dist"]
D_LO, D_HI, LNL = 1000.0, 5000.0, 13.0


def _flat_grid(n_slices=30, n_d=40, lnL=LNL, slope_per_x=0.0):
    """A synthetic .dslice-shaped grid: n_slices intrinsic points, each exported at n_d
    distances.  `slope_per_x` puts a known linear-in-1/d trend on lnL so the slope
    branch can be exercised as well as the flat one."""
    rows, y = [], []
    for i in range(n_slices):
        mc = 20.0 + 0.1 * i
        for d in np.linspace(D_LO, D_HI, n_d):
            rows.append([mc, d])
            y.append(lnL + slope_per_x * (1.0 / d - 1.0 / D_HI))
    return np.asarray(rows), np.asarray(y)


def _wrapped(X, Y, base_value=LNL, **kw):
    base = lambda Xf: np.full(len(Xf), base_value)   # noqa: E731  a perfectly flat "fit"
    return wrap_distance_tail(base, X, Y, COORDS, y_errors=np.full(len(Y), 0.1), **kw)


def _probe(f, mc=20.0, d=None):
    d = np.atleast_1d(d)
    return f(np.stack([np.full(len(d), mc), d], axis=1))


def test_no_op_on_the_support():
    """Inside the exported distance range the wrapper must return the base fit untouched.
    If this fails the fix is no longer a strict addition and every existing result moves."""
    X, Y = _flat_grid()
    f = _wrapped(X, Y)
    d = np.linspace(D_LO, D_HI, 500)
    assert np.allclose(_probe(f, d=d), LNL)


def test_continuous_at_the_edge():
    """A step at the support edge would be read by the sampler as real structure."""
    X, Y = _flat_grid()
    f = _wrapped(X, Y)
    inside = _probe(f, d=D_HI)[0]
    just_outside = _probe(f, d=D_HI * (1 + 1e-9))[0]
    assert abs(inside - just_outside) < 1e-6


def test_default_law_is_the_chord():
    """The production law.  lnL is convex in x = 1/d wherever Var(a) > <b> for the
    marginalised likelihood, and a convex function through the origin lies BELOW its
    chord, so ratio = u is both the exact asymptotic form and an upper bound on lnL
    beyond the edge.  It is parameter-free, which is the point: a decay rate tuned to
    match the reference would not be a fix."""
    X, Y = _flat_grid()
    f = _wrapped(X, Y)
    for d in (5500.0, 6000.0, 10000.0, 20000.0):
        u = (1.0 / d) / (1.0 / D_HI)
        assert _probe(f, d=d)[0] == pytest.approx(LNL * u, rel=1e-6)


def test_slope_law_rolls_off_as_2u_minus_u_squared():
    """The REJECTED alternative, kept because measuring it is what ruled it out.  A slice
    still flat at its edge has a unique through-the-origin quadratic that is also flat
    there, and it is 2u - u^2.  It is faithful to the data at the edge -- and far too
    gentle: on the catalog it moved the recovered width by under a point, because the
    marginalised lnL is a plateau across the whole exported support and the local slope
    therefore cannot see the turnover just past it."""
    X, Y = _flat_grid()
    f = _wrapped(X, Y, law="slope")
    for d in (5500.0, 6000.0, 10000.0, 20000.0):
        u = (1.0 / d) / (1.0 / D_HI)
        assert _probe(f, d=d)[0] == pytest.approx(LNL * (2 * u - u * u), rel=1e-6)


def test_decays_to_zero_at_large_distance():
    """lnL is a likelihood RATIO, so lnL(d -> infinity) = 0 is an identity.  This is the
    boundary condition the tree ensemble was missing."""
    X, Y = _flat_grid()
    f = _wrapped(X, Y)
    assert _probe(f, d=1e9)[0] == pytest.approx(0.0, abs=1e-3)


def test_monotone_decreasing_beyond_the_edge():
    """Walking out in distance must always walk lnL down.  A continuation that turns
    back up would put weight at large d all over again."""
    X, Y = _flat_grid()
    f = _wrapped(X, Y)
    v = _probe(f, d=np.linspace(D_HI, 60 * D_HI, 4000))
    assert np.all(np.diff(v) <= 1e-9)


def test_beats_the_volumetric_prior():
    """The point of the fix: beyond the edge the integrand exp(lnL) * d^2 must FALL.
    Flat extrapolation makes it rise, which is the whole bug."""
    X, Y = _flat_grid()
    f = _wrapped(X, Y)
    d = np.linspace(D_HI, 8 * D_HI, 400)
    integrand = _probe(f, d=d) + 2.0 * np.log(d)          # log of exp(lnL) * d^2
    assert integrand[-1] < integrand[0]
    flat = LNL + 2.0 * np.log(d)                          # what the bare tree gives
    assert flat[-1] > flat[0]


@pytest.mark.parametrize("law", ["chord", "slope"])
@pytest.mark.parametrize("slope_per_x", [0.0, 5000.0, 20000.0])
def test_continuation_stays_monotone_and_bounded(slope_per_x, law):
    """A slice already decaying at its edge is continued at its own rate.  Whatever the
    slope, the continuation may only shrink the edge value -- never grow it, never flip
    its sign."""
    X, Y = _flat_grid(slope_per_x=slope_per_x)
    f = _wrapped(X, Y, law=law)
    v = _probe(f, d=np.linspace(D_HI, 40 * D_HI, 2000))
    assert np.all(v <= LNL + 1e-6)
    assert np.all(v >= -1e-6)
    assert np.all(np.diff(v) <= 1e-9)


def test_rising_edge_is_clamped_not_trusted():
    """Monte-Carlo noise can leave a slice whose outer end RISES with distance.  Taken at
    face value that would extrapolate upward forever; it must be clamped to the flat
    roll-off instead."""
    X, Y = _flat_grid(slope_per_x=-20000.0)               # lnL increasing with distance
    f = _wrapped(X, Y, law="slope")
    v = _probe(f, d=np.linspace(D_HI, 40 * D_HI, 2000))
    assert np.all(np.diff(v) <= 1e-9)
    # clamped to the flat roll-off, i.e. exactly 2u - u^2 rather than anything rising
    u = (1.0 / (40 * D_HI)) / (1.0 / D_HI)
    assert v[-1] == pytest.approx(LNL * (2 * u - u * u), rel=1e-6)
    assert _probe(f, d=1e9)[0] == pytest.approx(0.0, abs=1e-3)


def test_lnL_offset_is_applied_to_the_physical_likelihood():
    """CIP fits Y = lnL_physical - lnL_shift.  The d -> infinity identity holds for the
    PHYSICAL likelihood ratio, so the tail must decay to -lnL_shift in fit units.
    Ignoring the offset anchors the decay to the wrong asymptote -- silently, and in the
    direction that reintroduces the bug."""
    shift = 40.0
    X, Y = _flat_grid(lnL=LNL - shift)
    f = _wrapped(X, Y, base_value=LNL - shift, lnL_offset=shift)
    assert _probe(f, d=1e9)[0] == pytest.approx(-shift, abs=1e-2)
    assert np.allclose(_probe(f, d=np.linspace(D_LO, D_HI, 200)), LNL - shift)


def test_missing_distance_coordinate_is_an_error():
    """A run that asked for the fix and silently did not get it would carry the very bias
    the option exists to remove, with nothing in the log to say so."""
    X, Y = _flat_grid()
    with pytest.raises(ValueError, match="dist"):
        wrap_distance_tail(lambda Xf: np.zeros(len(Xf)), X, Y, ["mc", "not_distance"])


def test_too_few_slices_is_an_error():
    """Guard against being pointed at something that is not a dslice export at all."""
    X, Y = _flat_grid(n_slices=3)
    with pytest.raises(ValueError, match="slices"):
        _wrapped(X, Y)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
