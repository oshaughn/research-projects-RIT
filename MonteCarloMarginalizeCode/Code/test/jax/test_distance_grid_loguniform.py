"""Gate for the log-uniform ("peak-resolving") distance quadrature.

WHY THESE TESTS AND NOT OTHERS.  The lever is one number -- the node count --
and one placement rule.  A test that merely builds a grid and checks it has
nodes would pass under every mutation that matters.  Each test below was
written against a specific mutation and VERIFIED to fail under it; the matrix
is in the PR body.  Three properties carry the contract:

  * the SPACING contract (relative spacing <= c/rho_max everywhere), and the
    calibration of c against the Gaussian trapezoid error law it is derived
    from -- pinned two-sided, because a c that is merely small satisfies a
    one-sided bound while making the grid uselessly expensive;
  * the DECOUPLING property: the dense angle lattice is sized from the
    amplitude on the FULL prior support, so no distance grid can shrink it.
    Asserted on the built objects, not on the helper, because a helper-level
    assertion cannot see a call site that stops calling the helper;
  * the DEFAULT: dist_grid="uniform" must reproduce today's grid node for node.

Everything here runs in seconds.  The convergence measurements that justify
the shipped tolerance live in DESIGN_jax_distance_quadrature.md beside the
module; they need a real precompute and are not gated.
"""
import ast
import os
import pathlib
import sys

import numpy as np
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import build_likelihood_data
from RIFT.likelihood.jax_ile.core import (
    make_distance_grid, make_distance_grid_loguniform, loguniform_grid_size,
    loguniform_spacing_for_tolerance, DIST_GRID_TOL_DEFAULT, DIST_GRID_SCHEMES)
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood

D_MIN, D_MAX = 1.0, 10000.0
DREF = 1000.0


def _nodes(x_grid, distMpcRef=DREF):
    return distMpcRef / np.asarray(x_grid)


# ---------------------------------------------------------------------------
# 1. The spacing contract, and the calibration of the constant it rests on.
# ---------------------------------------------------------------------------

def test_relative_spacing_meets_the_stated_contract():
    """Delta(ln d) <= c(tol)/rho_max at EVERY interval, for every case.

    This is the whole claim: one spacing resolves a peak of relative width
    1/rho wherever it sits.  An off-by-one in the node count, a floor for the
    ceil, or a linspace for the geomspace all break it.
    """
    for rho in (5.0, 38.9, 55.07, 300.0):
        for tol in (5e-1, 1e-1, DIST_GRID_TOL_DEFAULT, 1e-3):
            for lo, hi in ((1.0, 10000.0), (50.0, 2000.0)):
                x, _ = make_distance_grid_loguniform(
                    lo, hi, rho, distMpcRef=DREF, tol=tol)
                d = _nodes(x)
                h = np.max(np.diff(np.log(d)))
                c = loguniform_spacing_for_tolerance(tol)
                assert h <= c / rho * (1 + 1e-12), (
                    "spacing contract violated: rho=%g tol=%g range=[%g,%g] "
                    "h=%.6g > c/rho=%.6g" % (rho, tol, lo, hi, h, c / rho))
                # ...and the grid must actually span the requested support
                assert np.isclose(d[0], lo) and np.isclose(d[-1], hi)


def test_tolerance_constant_matches_the_gaussian_trapezoid_error_law():
    """c = pi*sqrt(2/ln(2/tol)) must reproduce the error it is derived from.

    Two-sided on purpose.  A one-sided "error <= tol" check passes for any c
    smaller than the right one -- including c -> 0, which satisfies every
    accuracy claim while making the grid arbitrarily expensive.  The lower
    bound is what pins the constant to the LAW rather than to caution.
    """
    for tol in (5e-1, 1e-1, 1e-2, 1e-3, 1e-4):
        c = loguniform_spacing_for_tolerance(tol)
        h = c * 1.0                       # sigma == 1 without loss of generality
        worst = 0.0
        for mu in np.linspace(0.0, h, 25):        # the error oscillates with phase
            k = np.arange(-int(np.ceil(40.0 / h)) - 1, int(np.ceil(40.0 / h)) + 2)
            u = k * h
            approx = h * np.sum(np.exp(-0.5 * (u - mu) ** 2))
            worst = max(worst, abs(approx / np.sqrt(2 * np.pi) - 1.0))
        assert worst <= tol * 1.05, (
            "tol=%g: measured trapezoid error %.4g exceeds the target" % (tol, worst))
        assert worst >= tol / 20.0, (
            "tol=%g: measured error %.4g is far below the target -- the "
            "constant no longer tracks the error law it is derived from, so "
            "the grid is paying for accuracy nobody asked for" % (tol, worst))


def test_node_count_responds_to_every_lever():
    """rho_max, the prior range, and tol must all be live.  Each has been an
    inert argument in some draft of this helper."""
    base = loguniform_grid_size(D_MIN, D_MAX, 50.0, 1e-2)
    assert loguniform_grid_size(D_MIN, D_MAX, 100.0, 1e-2) > base, "rho lever dead"
    assert loguniform_grid_size(10.0, D_MAX, 50.0, 1e-2) < base, "range lever dead"
    assert loguniform_grid_size(D_MIN, D_MAX, 50.0, 1e-4) > base, "tol lever dead"
    # and the count is the closed form, not an approximation of it
    c = loguniform_spacing_for_tolerance(1e-2)
    assert base == int(np.ceil(50.0 * np.log(D_MAX / D_MIN) / c)) + 1


def test_rho_max_must_be_a_bound_and_never_falls_back():
    """A missing or degenerate bound must RAISE.  Falling back to a default
    node count is the silent-no-op pattern that has bitten this module: the
    run would look normal and the marginal would be wrong."""
    for bad in (0.0, -1.0, float("nan"), float("inf"), None):
        try:
            loguniform_grid_size(D_MIN, D_MAX, bad, 1e-2)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("rho_max=%r must raise, not fall back" % (bad,))
    for lo, hi in ((0.0, 10.0), (10.0, 10.0), (100.0, 10.0)):
        try:
            loguniform_grid_size(lo, hi, 50.0, 1e-2)
        except ValueError:
            pass
        else:
            raise AssertionError("range (%r,%r) must raise" % (lo, hi))
    for bad_tol in (0.0, -1.0, 2.0, 5.0):
        try:
            loguniform_spacing_for_tolerance(bad_tol)
        except ValueError:
            pass
        else:
            raise AssertionError("tol=%r must raise" % (bad_tol,))


def test_n_max_raises_instead_of_clamping():
    """Clamping would silently violate the spacing contract -- the grid would
    still be built, still be log-uniform, and no longer resolve the peak."""
    try:
        make_distance_grid_loguniform(D_MIN, D_MAX, 5000.0, distMpcRef=DREF,
                                      tol=1e-3, n_max=64)
    except ValueError as exc:
        assert "n_max" in str(exc)
    else:
        raise AssertionError("an over-cap node count must raise, not clamp")


def test_weights_are_a_normalized_proper_distance_average():
    x, lw = make_distance_grid_loguniform(D_MIN, D_MAX, 55.07, distMpcRef=DREF)
    w = np.exp(np.asarray(lw))
    assert np.isclose(w.sum(), 1.0, rtol=0, atol=1e-12)
    assert np.all(np.isfinite(np.asarray(lw)))
    d = _nodes(x)
    assert np.all(np.diff(d) > 0), "nodes must be strictly increasing"
    # the volumetric prior must be the one being averaged over
    x2, lw2 = make_distance_grid_loguniform(D_MIN, D_MAX, 55.07,
                                            d_prior="uniform", distMpcRef=DREF)
    assert not np.allclose(np.asarray(lw), np.asarray(lw2))


def test_end_intervals_are_half_width():
    """On a log grid the last interval is ~1% of d_max, so giving the end nodes
    a FULL interval (the convention make_distance_grid_adaptive uses) misplaces
    percent-level volumetric prior mass onto d_max -- measured as a ~0.018 nat
    error floor that no refinement removes.  Pinned by comparing the raw
    trapezoid prior mass against the exact integral, with the wrong convention
    computed here so the test discriminates rather than merely passing."""
    n = loguniform_grid_size(D_MIN, D_MAX, 55.07, DIST_GRID_TOL_DEFAULT)
    d = np.geomspace(D_MIN, D_MAX, n)
    exact = (D_MAX ** 3 - D_MIN ** 3) / 3.0
    dd = np.empty_like(d)
    dd[1:-1] = 0.5 * (d[2:] - d[:-2])
    dd[0], dd[-1] = 0.5 * (d[1] - d[0]), 0.5 * (d[-1] - d[-2])
    half = np.sum(d ** 2 * dd) / exact
    dd[0], dd[-1] = d[1] - d[0], d[-1] - d[-2]
    full = np.sum(d ** 2 * dd) / exact
    assert abs(half - 1.0) < 1e-3, "half-width ends: prior mass off by %.4g" % (half - 1)
    assert abs(full - 1.0) > 1e-2, (
        "the two conventions no longer differ measurably; this test can no "
        "longer detect the wrong one")
    # and the shipped helper must use the half-width convention
    _, lw = make_distance_grid_loguniform(D_MIN, D_MAX, 55.07, distMpcRef=DREF)
    w = np.exp(np.asarray(lw))
    assert np.isclose(w[-1] / w.sum(), (d[-1] ** 2 * 0.5 * (d[-1] - d[-2]))
                      / (np.sum(d ** 2 * np.concatenate(
                          [[0.5 * (d[1] - d[0])], 0.5 * (d[2:] - d[:-2]),
                           [0.5 * (d[-1] - d[-2])]]))), rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. Wiring, on a real (tiny, synthetic) likelihood object.
# ---------------------------------------------------------------------------

def _synth(scale=1.0, seed=3, modes=((2, 2), (2, -2)), npts=32, deltaT=1.0 / 1024,
           kappa_boost=1.0):
    """Structurally-faithful packed data (U Hermitian PD, V complex symmetric),
    the same construction test_angle_marg_smoke uses, duplicated here so this
    file does not depend on another test module's import order."""
    rng = np.random.default_rng(seed)
    tw = npts * deltaT / 2.0
    tvals = np.linspace(-tw, tw, npts)
    tref = 1126259462.413
    K = len(modes)
    packed = {}
    for det in ("H1", "L1"):
        white = (rng.standard_normal((K, 4096)) + 1j * rng.standard_normal((K, 4096)))
        kx = np.arange(-40, 41)
        kern = np.exp(-0.5 * (kx / 12.0) ** 2)
        kern /= kern.sum()
        rho = np.stack([np.convolve(white[k].real, kern, "same")
                        + 1j * np.convolve(white[k].imag, kern, "same")
                        for k in range(K)]).astype(np.complex128)
        rho *= np.sqrt(len(kx)) * scale * kappa_boost
        M = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        U = (M @ M.conj().T + 3 * np.eye(K)) * scale ** 2
        B = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        V = (B @ B.T) * scale ** 2 * 0.3
        packed[det] = dict(lms=np.array(modes, dtype=int), rholmArray=rho,
                           U=U, V=V, epoch=tref - 0.5)
    return build_likelihood_data(packed, deltaT, tref, tvals)


def _like(data, dist_grid="uniform", angle_marg="laplace", n_grid=64,
          d_min=D_MIN, d_max=D_MAX, **kw):
    return JAXDistPhiPsiMargLikelihood(
        data, d_min, d_max, nphi=8, npsi=8, n_grid=n_grid, interp="sinc",
        angle_marg=angle_marg, dist_grid=dist_grid, **kw)


def test_default_distance_grid_is_bit_identical_to_the_shipped_uniform_grid():
    """NEVER change a RIFT default.  An existing command line must reproduce
    an existing run node for node -- asserted with exact equality, not
    allclose, because a reproduction claim tolerates no drift."""
    data = _synth()
    # Constructed WITHOUT dist_grid=, deliberately: passing it explicitly --
    # even as "uniform" -- means the constructor DEFAULT is never exercised,
    # and a flipped default sails through.  (It did: this test survived that
    # mutation until the explicit keyword was removed.)
    like = JAXDistPhiPsiMargLikelihood(
        data, D_MIN, D_MAX, nphi=8, npsi=8, n_grid=64, interp="sinc",
        angle_marg="laplace")
    x_ref, lw_ref = make_distance_grid(D_MIN, D_MAX, 64, "euclidean",
                                       distMpcRef=data.distMpcRef)
    assert np.array_equal(np.asarray(like.x_grid), np.asarray(x_ref))
    assert np.array_equal(np.asarray(like.log_w_grid), np.asarray(lw_ref))
    assert like.dist_grid_info["mode"] == "uniform"
    assert like.dist_grid_info["n"] == 64


def test_narrowing_the_distance_grid_can_move_the_sizing_amplitude():
    """The PREMISE of the next test.  estimate_angle_amplitude reads only
    min/max of the grid it is handed and clips the per-angle distance maximum
    to it, so a grid that stops containing A/B reports a smaller amplitude --
    which would silently shrink the dense angle lattice.  If this ever stops
    being true, the guard below is guarding nothing and must be revisited."""
    from RIFT.likelihood.jax_ile import anglemarg as AM
    data = _synth(scale=3.0, kappa_boost=4.0)
    full, _ = make_distance_grid(D_MIN, D_MAX, 64, "euclidean",
                                 distMpcRef=data.distMpcRef)
    narrow, _ = make_distance_grid(2000.0, 4000.0, 64, "euclidean",
                                   distMpcRef=data.distMpcRef)
    a_full = AM.estimate_angle_amplitude(data, full, interp="sinc")
    a_narrow = AM.estimate_angle_amplitude(data, narrow, interp="sinc")
    assert a_narrow < a_full, (
        "a narrowed distance grid no longer lowers the sizing amplitude "
        "(%.6g vs %.6g); the decoupling guard has nothing left to guard"
        % (a_narrow, a_full))


def test_angle_lattice_is_sized_from_the_full_support_grid_by_name():
    """THE safety property, guarded at the CALL SITE.

    A behavioural test cannot discriminate this one, and saying so is the
    point: BOTH shipped schemes span the full prior range, so handing
    estimate_angle_amplitude `self.x_grid` instead of `x_grid_full` produces
    the identical amplitude today.  The mutation survives every value-level
    assertion.  What the change actually buys is that the invariant is
    structural rather than incidental -- it stays true the moment any
    narrowing scheme is added.  NO in-tree scheme is currently such a scheme,
    and that correction matters: an earlier draft of this docstring named the
    deprecated JAX_ILE_DISTGRID_ADAPTIVE branch and quoted 12.6%, which is
    wrong.  make_distance_grid_adaptive concatenates a full-range `linspace`
    backbone before dedup, so its x_min/x_max ARE the full support's and it
    returns a byte-identical amplitude -- 105.737261 either way on THIS file's
    _synth(scale=3.0, kappa_boost=4.0), and 10573.7261 either way on the louder
    _synth(scale=30.0, kappa_boost=40.0) the figure was first taken from.  (An
    earlier draft quoted only the second, unlabelled, which reproduces 100x off
    if you use the fixture this file actually ships.)  The 12.6-14.8% figure
    belongs to the hand-built [0.8 d, 1.25 d] window in wrapper.py, which no
    code path produces.  So the guard is on the argument
    the wrapper passes, which is where the property lives, and it is
    prospective -- see test_narrowing_the_distance_grid_can_move_the_sizing_
    amplitude, which pins the premise the guard rests on."""
    import inspect
    import textwrap
    import RIFT.likelihood.jax_ile.wrapper as W
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(W.JAXDistPhiPsiMargLikelihood.__init__)))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "estimate_angle_amplitude"]
    assert len(calls) == 1, (
        "expected exactly one estimate_angle_amplitude call site in "
        "JAXDistPhiPsiMargLikelihood.__init__, found %d" % len(calls))
    arg = calls[0].args[1]
    assert isinstance(arg, ast.Name) and arg.id == "x_grid_full", (
        "the sizing amplitude must be computed on the FULL-prior-support grid "
        "(x_grid_full), not on the grid the likelihood integrates over; got %s"
        % ast.dump(arg))
    # and the shipped schemes must agree, which is the invariant users see
    data = _synth(scale=3.0, kappa_boost=4.0)
    a = _like(data, "uniform", n_grid=64)
    b = _like(data, "loguniform", n_grid=64)
    assert a.angle_marg_info["amp_sizing"] == b.angle_marg_info["amp_sizing"]
    assert a.angle_marg_info["sample_grid"] == b.angle_marg_info["sample_grid"]
    assert a.x_grid.shape[0] != b.x_grid.shape[0], (
        "the two arms must actually differ in the distance grid, or the "
        "invariant above is vacuous")


def test_loguniform_records_what_it_built():
    data = _synth()
    like = _like(data, "loguniform", n_grid=64)
    gi = like.dist_grid_info
    assert gi["mode"] == "loguniform"
    assert gi["tol"] == DIST_GRID_TOL_DEFAULT
    assert gi["rho_max"] > 0.0
    assert gi["n"] == loguniform_grid_size(D_MIN, D_MAX, gi["rho_max"],
                                           DIST_GRID_TOL_DEFAULT)
    assert gi["dlnd"] <= loguniform_spacing_for_tolerance(
        DIST_GRID_TOL_DEFAULT) / gi["rho_max"] * (1 + 1e-12)


def test_rho_max_is_the_amplitude_the_runtime_failsafe_actually_compares():
    """The distance grid's coverage by the existing fail-safe is an identity,
    not a correlation -- and it only holds if the spacing is sized from
    amp_SIZING (floored at the crossover), which is what
    anglemarg._runtime_amp_failsafe compares the per-call amplitude against.

    Sizing from the UNfloored amp_data instead leaves a silent gap for quiet
    targets: a runtime amplitude between amp_data and amp_sizing under-resolves
    the distance peak and trips nothing.  This test uses a quiet synthetic
    target, where the two differ, so it can see the difference."""
    from RIFT.likelihood.jax_ile import anglemarg as AM
    data = _synth(scale=0.05, kappa_boost=0.05)
    like = _like(data, "loguniform", n_grid=64)
    amp_data = like.angle_marg_info["amplitude"]
    amp_sizing = like.angle_marg_info["amp_sizing"]
    assert amp_data < AM.ANGLE_MARG_CROSSOVER_AMPLITUDE, (
        "this synthetic target is no longer quiet enough for the floor to "
        "bind, so the test can no longer distinguish the two amplitudes")
    assert amp_sizing == AM.ANGLE_MARG_CROSSOVER_AMPLITUDE
    assert np.isclose(like.dist_grid_info["rho_max"], np.sqrt(2 * amp_sizing),
                      rtol=1e-12), (
        "rho_max must come from amp_sizing (%.6g), not amp_data (%.6g)"
        % (amp_sizing, amp_data))


def test_unrecognised_scheme_raises_and_never_falls_through():
    data = _synth()
    for bad in ("adaptive", "log-uniform", "geometric", "", None, 1):
        try:
            _like(data, bad)
        except ValueError as exc:
            assert "dist_grid" in str(exc)
        else:
            raise AssertionError("dist_grid=%r must raise, not default" % (bad,))
    assert DIST_GRID_SCHEMES == ("uniform", "loguniform")


def test_loguniform_is_refused_on_the_grid_angle_scheme_not_ignored():
    """The sizing amplitude does not exist on the 'grid' path.  Silently using
    the uniform grid there would be a flag that parses, prints and does
    nothing."""
    data = _synth()
    try:
        _like(data, "loguniform", angle_marg="grid")
    except ValueError as exc:
        assert "angle_marg" in str(exc)
    else:
        raise AssertionError("dist_grid='loguniform' with angle_marg='grid' "
                             "must raise")


def test_loguniform_marginal_agrees_with_a_fine_uniform_grid():
    """NUMERICAL execution.  Everything above checks structure; a mutation that
    returns a wrong marginal (a mis-signed weight, a reversed node order, a
    dropped prior factor) passes all of it.  This does not."""
    data = _synth(scale=2.0, kappa_boost=3.0)
    ra = np.array([0.9, 2.4]); dec = np.array([0.4, -0.7]); incl = np.array([1.1, 2.0])
    fine = JAXDistPhiPsiMargLikelihood(
        data, D_MIN, D_MAX, nphi=8, npsi=8, n_grid=1024, interp="sinc",
        angle_marg="laplace", dist_grid="uniform")
    lg = _like(data, "loguniform", n_grid=64, dist_grid_tol=1e-3)
    v_fine = np.asarray(fine.log_likelihood(ra, dec, incl))
    v_log = np.asarray(lg.log_likelihood(ra, dec, incl))
    assert np.all(np.isfinite(v_log))
    assert np.max(np.abs(v_log - v_fine)) < 0.02, (
        "loguniform marginal differs from a 1024-node uniform reference by "
        "%.4g nats" % np.max(np.abs(v_log - v_fine)))


# A gradient test lived here and was REMOVED, deliberately.  It asserted that
# value_and_grad stays finite under the log-uniform grid.  It passed -- and it
# passed under every mutation that could be constructed against it, including a
# degenerate node set with zero-width intervals (which four other tests here
# caught).  The reason is structural: the distance grid is a compile-time
# CONSTANT in the traced graph, so no change to it can poison the backward pass
# while leaving the forward value intact.  A test that cannot be made to fail is
# coverage-shaped, not coverage, and it cost 15 s of gate time.  If the node
# positions are ever made traced (the per-sample quadrature of
# DESIGN_jax_distance_quadrature.md section 4d does exactly that), reinstate it
# -- there it would have real work to do.


# ---------------------------------------------------------------------------
# 3. The driver seam.  A library test cannot see a driver that stops calling.
# ---------------------------------------------------------------------------

_CODE = pathlib.Path(__file__).resolve().parents[2]


def _driver_src():
    return (_CODE / "bin" / "integrate_likelihood_extrinsic_jax").read_text()


def _option_call(tree, flag):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and a0.value == flag:
                return node
    return None


def test_driver_exposes_the_option_and_defaults_it_to_uniform():
    """AST on the VALUE node.  'the string appears in the file' is satisfied by
    a help text alone."""
    tree = ast.parse(_driver_src())
    call = _option_call(tree, "--distance-grid-scheme")
    assert call is not None, "the driver must define --distance-grid-scheme"
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw["default"], ast.Constant) and kw["default"].value == "uniform", (
        "the default must remain 'uniform': an existing command line must "
        "reproduce an existing run")
    choices = kw["choices"]
    assert isinstance(choices, ast.Tuple)
    assert [c.value for c in choices.elts] == ["uniform", "loguniform"], (
        "an unrecognised value must be refused by the parser, not defaulted")


def test_driver_forwards_the_parsed_option_not_a_constant():
    """Hardcoding dist_grid="uniform" at the call site passes every weaker
    guard: flag parsed, help printed, feature inert."""
    tree = ast.parse(_driver_src())
    seen = [k.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            for k in (node.keywords or []) if k.arg == "dist_grid"]
    assert seen, "the driver must pass dist_grid to JAXDistPhiPsiMargLikelihood"
    assert any(isinstance(v, ast.Name) for v in seen), (
        "dist_grid must be forwarded as the parsed option, not a constant")
    seen_tol = [k.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                for k in (node.keywords or []) if k.arg == "dist_grid_tol"]
    assert any(isinstance(v, ast.Name) for v in seen_tol), (
        "dist_grid_tol must be forwarded as the parsed option")


def test_driver_refuses_the_option_on_modes_that_do_not_implement_it():
    src = _driver_src()
    assert "--distance-grid-scheme/--distance-grid-tol apply only to" in src, (
        "the driver must fail closed when the option is set on a mode that "
        "ignores it; a silently inert flag is this pipeline's documented "
        "failure mode")


def test_driver_refuses_distance_grid_points_together_with_loguniform():
    """Two options that both set the node count.  Whichever loses silently is a
    silently-inert flag; refusing is the only reading that cannot mislead.  The
    option therefore defaults to None (a sentinel), resolved to
    DISTANCE_GRID_POINTS_DEFAULT -- optparse cannot otherwise distinguish
    "explicitly 256" from "not passed"."""
    src = _driver_src()
    tree = ast.parse(src)
    call = _option_call(tree, "--distance-grid-points")
    assert call is not None
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw["default"], ast.Constant) and kw["default"].value is None, (
        "--distance-grid-points must default to the None sentinel, or the "
        "driver cannot tell an explicit 256 from an unset option")
    assert "--distance-grid-points and --distance-grid-scheme loguniform both" in src
    assert any(isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "DISTANCE_GRID_POINTS_DEFAULT"
                       for t in n.targets)
               for n in ast.walk(tree)), "the resolved default must be a named constant"


def test_driver_never_writes_the_resolved_option_back_onto_opts():
    """Writing a resolved option back onto `opts` made event 1 of a batch read
    event 0's choice once already in this driver (see DESIGN_jax_tempering.md
    and test_jax_tempering_chooser).  The resolution must land in a local."""
    tree = ast.parse(_driver_src())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Attribute)
                        and t.attr in ("distance_grid_points", "distance_grid_scheme",
                                       "distance_grid_tol")
                        and isinstance(t.value, ast.Name) and t.value.id == "opts"):
                    raise AssertionError(
                        "opts.%s is assigned; resolve into a local instead" % t.attr)


def test_driver_always_reports_the_distance_grid_that_ran():
    src = _driver_src()
    assert 'print("  distance grid: %s"' in src, (
        "the resolved distance grid must be printed unconditionally; a report "
        "guarded on mode == 'adaptive' cannot distinguish uniform from a "
        "silently-failed request")


# ---------------------------------------------------------------------------
# 4. Regressions added after external review (F1, F2, F3, F8).  Each is here
#    because a mutation SURVIVED the first matrix without it.
# ---------------------------------------------------------------------------

def test_truncated_distance_support_is_refused_not_silently_mis_sized():
    """F1.  The spacing contract assumes the integrand is a Gaussian PEAK
    inside the support.  When the maximizing distance x* = A/B is exterior the
    integrand is a boundary LAYER at a prior edge instead, and a log-uniform
    grid is the wrong instrument twice over: its absolute spacing is coarsest
    exactly at d_max, and refining it adds nodes proportionally everywhere so
    the layer never resolves (measured: tol 0.5 -> 1e-9 moves the error only
    5.23 -> 3.92 nats, while uniform 256 -> 4096 moves 2.52 -> 0.36).  The
    clip also makes the amplitude UNDER-read, so the node count moves the
    WRONG WAY.  Refused at build time; NOT silently fallen back to uniform,
    which would make this flag produce the other scheme's grid."""
    data = _synth(scale=3.0, kappa_boost=4.0)
    try:
        _like(data, "loguniform", n_grid=64, d_min=2000.0, d_max=10000.0)
    except ValueError as exc:
        assert "OUTSIDE" in str(exc) and "d_min" in str(exc)
        assert "--d-max" in str(exc), "the refusal must name the recourse"
    else:
        raise AssertionError("a truncated distance support must be refused")
    # ...and the interior control must still build
    like = _like(data, "loguniform", n_grid=64)
    assert like.dist_grid_info["mode"] == "loguniform"


def test_clip_excess_diagnostic_detects_exteriority_and_is_quiet_when_interior():
    """The premise of the refusal above, at the estimator.  A guard whose
    detector cannot distinguish the two regimes is not a guard."""
    from RIFT.likelihood.jax_ile import anglemarg as AM
    data = _synth(scale=3.0, kappa_boost=4.0)
    xg_in, _ = make_distance_grid(1.0, 10000.0, 64, "euclidean",
                                  distMpcRef=data.distMpcRef)
    xg_out, _ = make_distance_grid(2000.0, 10000.0, 64, "euclidean",
                                   distMpcRef=data.distMpcRef)
    _, d_in = AM.estimate_angle_amplitude(data, xg_in, interp="sinc",
                                          return_diagnostics=True)
    _, d_out = AM.estimate_angle_amplitude(data, xg_out, interp="sinc",
                                           return_diagnostics=True)
    assert d_in["clip_excess"] <= 1.0 + 1e-9, "interior support must not trip"
    assert d_out["clip_excess"] > 1.0 + 1e-3, "exterior support must trip"
    assert d_out["amp_clipped"] < d_out["amp_unclipped"]
    # the returned amplitude itself is unchanged by the diagnostic
    a_plain = AM.estimate_angle_amplitude(data, xg_in, interp="sinc")
    a_diag, _ = AM.estimate_angle_amplitude(data, xg_in, interp="sinc",
                                            return_diagnostics=True)
    assert a_plain == a_diag


def test_loguniform_is_refused_under_the_per_sample_gh_quadrature():
    """F2.  core._distmarg_gh_logL places its own nodes and reads ONLY
    min/max of x_grid, so both schemes are bit-identical under it while
    dist_grid_info still reports mode='loguniform'.  Reachable without typing
    'exact': choose_angle_marg_scheme FORCES exact whenever GH is set."""
    from RIFT.likelihood.jax_ile import core as C
    data = _synth()
    saved = C._DISTMARG_GH_N
    C._DISTMARG_GH_N = 32
    try:
        _like(data, "loguniform", n_grid=64)
    except ValueError as exc:
        assert "JAX_ILE_DISTMARG_GH" in str(exc)
        assert "inert" in str(exc)
    else:
        raise AssertionError("loguniform under GH must raise, not run inert")
    finally:
        C._DISTMARG_GH_N = saved


def test_zero_clipped_amplitude_is_the_most_exterior_case_and_is_refused():
    """F1, the EXTREME, which the test above cannot reach.

    Section 1a's worst row is the one where the clipped amplitude reaches
    exactly 0: the whole prior support lies beyond the maximizer, so
    ``x*A - x^2 B/2 <= 0`` at every sampled angle and the max over ``x >= 0``
    is the floor.  ``clip_excess`` then has no ratio to form, and a separate
    arm of the expression -- ``inf if amp_unclipped > 0 else 1.0`` -- is what
    decides the refusal.  Replacing that arm with a bare ``1.0`` leaves every
    other test in this file GREEN while the design note's worst case (amp -> 0,
    crossover floor, grid collapses) BUILDS.  Verified: 30/30 still passed
    under exactly that mutation, and the constructor returned a 37-node grid.

    ``test_clip_excess_diagnostic_...`` uses a [2000, 10000] support, where the
    clipped amplitude is positive, so it never enters this arm.
    """
    from RIFT.likelihood.jax_ile import anglemarg as AM
    data = _synth(scale=3.0, kappa_boost=4.0)
    xg, _ = make_distance_grid(1.0, 10.0, 64, "euclidean",
                               distMpcRef=data.distMpcRef)
    _, diag = AM.estimate_angle_amplitude(data, xg, interp="sinc",
                                          return_diagnostics=True)
    assert diag["amp_clipped"] == 0.0, (
        "this support no longer drives the CLIPPED amplitude to exactly 0 "
        "(%.6g), so it can no longer exercise the amp_emp == 0 arm and this "
        "test is guarding nothing" % diag["amp_clipped"])
    assert diag["amp_unclipped"] > 0.0, (
        "the UNCLIPPED amplitude must stay positive here, or there is no "
        "exteriority left to detect")
    assert diag["clip_excess"] == float("inf"), (
        "a zero clipped amplitude against a positive unclipped one is the "
        "MOST exterior case there is; reporting a finite ratio -- above all "
        "an interior-looking 1.0 -- disarms the refusal in exactly the regime "
        "section 1a measures at +4.60 nats")
    try:
        _like(data, "loguniform", n_grid=64, d_min=1.0, d_max=10.0)
    except ValueError as exc:
        assert "OUTSIDE" in str(exc)
    else:
        raise AssertionError(
            "the extreme exterior case (clipped amplitude 0) must be refused, "
            "not built")


def test_driver_refuses_the_gh_combination_at_PARSE_time():
    """F2, the DRIVER half -- which no other test in this file covers.

    The constructor refuses this combination as well, so nothing can silently
    run; but the parse-time half is the one that spares the user a full
    precompute (F8), and DESIGN section 5 asserts in writing that it fires
    there.  Deleting that arm of check_critical_and_report left all 30 tests
    here green.

    ``--angle-marg-scheme auto``, not ``exact``: choose_angle_marg_scheme
    FORCES the exact scheme whenever GH is enabled, so this is reachable
    without the user ever typing it.  Executable -- the real
    check_critical_and_report runs, reading the same environment variable the
    shipping code reads.
    """
    import contextlib, io
    mod = _driver_module()
    args = ["--mode", "flowmc-phipsimarg",
            "--distance-grid-scheme", "loguniform",
            "--angle-marg-scheme", "auto"]
    saved = os.environ.get("JAX_ILE_DISTMARG_GH")
    os.environ["JAX_ILE_DISTMARG_GH"] = "32"
    try:
        optp = mod.build_parser()
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                opts, _ = optp.parse_args(list(args))
                mod.check_critical_and_report(opts, optp)
        except SystemExit:
            msg = err.getvalue()
            assert "JAX_ILE_DISTMARG_GH" in msg, msg[-400:]
            assert "inert" in msg, msg[-400:]
        else:
            raise AssertionError(
                "--distance-grid-scheme loguniform under JAX_ILE_DISTMARG_GH "
                "must be refused at PARSE time, not deferred to the "
                "constructor after a full precompute")
        # ...and the identical command line must be ACCEPTED both with the
        # variable unset AND with it explicitly OFF.  "0" is the case that
        # separates the shipped `int(...) > 0` from a truthiness test on the
        # raw string: under `if os.environ.get(...)` the driver refuses while
        # core._DISTMARG_GH_N is 0 and the constructor would accept, so the two
        # seams disagree and the user is refused a combination that works.
        # External review found exactly that mutation surviving.
        for off in ("0", "00", None):
            if off is None:
                os.environ.pop("JAX_ILE_DISTMARG_GH", None)
            else:
                os.environ["JAX_ILE_DISTMARG_GH"] = off
            from RIFT.likelihood.jax_ile import core as _C
            assert _C._DISTMARG_GH_N == 0, (
                "precondition: the kernels must see GH as OFF for %r" % (off,))
            optp = mod.build_parser()
            opts, _ = optp.parse_args(list(args))
            mod.check_critical_and_report(opts, optp)   # must not raise
    finally:
        if saved is None:
            os.environ.pop("JAX_ILE_DISTMARG_GH", None)
        else:
            os.environ["JAX_ILE_DISTMARG_GH"] = saved


def test_sky_doubling_updates_the_unclipped_maximum_too():
    """F1's detector on the SKY-DOUBLING path, guarded at the source.

    estimate_angle_amplitude re-draws the sky when its split-half check says
    the maximum is still growing.  The CLIPPED maximum is updated there; if the
    UNCLIPPED companion is not, the exteriority detector reads only the first
    batch while its denominator keeps growing, so clip_excess falls BELOW 1 and
    the F1 refusal disarms itself on exactly the events whose sky sampling was
    too coarse to trust.

    BOTH a value pin and a source guard, because neither alone is enough --
    and an earlier draft of this test shipped only the source guard on the
    strength of a claim that was too strong.

    The re-draw branch IS reachable: sweeping (data seed in 0..5) x (support
    [1, 10000] Mpc) x (n_sky in 4, 8, 16, 32, 64) x (estimator seed in 0..3) --
    120 combinations -- 19 enter it, across five of the six data seeds.  The
    fixture below is one that enters it AND sits on an exterior support.

    What a value pin CAN catch: any change that scales or replaces the
    accumulated unclipped maximum (halving it, wrapping it, resetting it after
    the loop, or reverting the CONSUMER to read the first batch's array).
    Those all move clip_excess on this fixture, and external review found three
    such mutations that the source guard alone missed -- including one that
    restores the pre-fix defect bit-for-bit by touching a line the loop guard
    never looks at.

    What a value pin CANNOT catch, which is why the source guard stays: simply
    DELETING the loop update leaves clip_excess bit-identical here, because the
    deterministic face-on/face-off extremes are appended to the FIRST batch and
    are what attains the unclipped maximum, so the second batch contributes
    nothing on every fixture available.  That corruption is real but silent --
    a dataset whose unclipped maximum came from a second-batch draw would take
    clip_excess BELOW 1 and disarm the F1 refusal, and nothing here bounds
    that.
    """
    import inspect
    import textwrap
    from RIFT.likelihood.jax_ile import anglemarg as AM

    # ---- 1. VALUE PIN, on a fixture that actually enters the re-draw ----
    data = _synth(scale=3.0, kappa_boost=4.0, seed=3)
    xg, _ = make_distance_grid(500.0, 10000.0, 64, "euclidean",
                               distMpcRef=data.distMpcRef)
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _, diag = AM.estimate_angle_amplitude(
            data, xg, interp="sinc", n_sky=64, seed=1, return_diagnostics=True)
    assert "doubling" in buf.getvalue(), (
        "this fixture no longer enters the sky re-draw branch, so the pin "
        "below no longer exercises it; find another (see the docstring sweep)")
    # RELATIVE, not exact.  These come through BLAS-heavy reconstruction, and
    # float64 is not bit-portable across CPUs: CI measured amp_clipped
    # 21.79506318041593 against 21.795063180415923 here, a 3.2e-16 relative
    # difference that failed an == pin.  The tolerance is chosen from BOTH
    # sides and must stay there: ~1e-16 of platform drift below it, and the
    # mutations it exists to catch far above it -- halving the accumulator
    # (5e-1), a stray rescale (1e-3), and a wrapper that multiplies by
    # 1.0000001 (1e-7).  1e-11 sits five orders above the drift and four
    # below the tightest mutation.  Do NOT loosen it past 1e-8.
    import math
    _RTOL = 1e-11
    for key, want in (("amp_clipped", 21.795063180415923),
                      ("amp_unclipped", 52.868630517667135),
                      ("clip_excess", 2.4257158641858192)):
        assert math.isclose(diag[key], want, rel_tol=_RTOL), (
            "%s on the re-draw fixture is %.17g, expected %.17g (rel %.3g > "
            "%.0e).  Any rescaling, wrapping, post-loop reset, or reversion "
            "of the CONSUMER to the first batch's array lands here; a drift "
            "at the 1e-16 level instead means a new platform, and the "
            "tolerance -- not the expectation -- is what to revisit."
            % (key, diag[key], want, abs(diag[key] - want) / want, _RTOL))
    assert diag["clip_excess"] > 1.0 + 1e-3, "and it must still refuse"

    # ---- 2. SOURCE GUARD, for the one mutation a value pin cannot see ----
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(AM.estimate_angle_amplitude)))
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    assert len(loops) == 1, (
        "expected exactly one re-draw loop in estimate_angle_amplitude, found "
        "%d; this guard names the loop by being the only one" % len(loops))
    assigned = {t.id for n in ast.walk(loops[0])
                if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert "amp_emp" in assigned, (
        "the re-draw loop no longer updates the clipped maximum; this guard "
        "is anchored to that update and must be revisited")
    assert "amp_u_emp" in assigned, (
        "the re-draw loop updates the CLIPPED maximum but not the UNCLIPPED "
        "one.  clip_excess = amp_unclipped / amp_clipped then reads the first "
        "sky batch against a denominator that grew, falls below 1, and the "
        "F1 exterior-peak refusal stops firing.  Update both, adjacently.")
    # and the two must be accumulated the same way, so that dropping one is
    # visible on sight rather than only to this test
    src = inspect.getsource(AM.estimate_angle_amplitude)
    assert "amp_emp = max(amp_emp, float(amps2.max()))" in src
    assert "amp_u_emp = max(amp_u_emp, float(amps_u2.max()))" in src
    # ...and the CONSUMER must read the accumulator, not re-derive from the
    # array.  With the concatenate gone, `amps_u` holds batch 1 ONLY, so
    # `amp_unclipped = np.max(amps_u)` is the pre-fix defect bit-for-bit while
    # the loop guard above still passes.  External review found exactly that.
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)][0]
    reads = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "amp_unclipped"
                     for t in n.targets)]
    assert len(reads) == 1 and isinstance(reads[0].value, ast.Name) \
        and reads[0].value.id == "amp_u_emp", (
        "amp_unclipped must be the accumulator amp_u_emp itself; re-deriving "
        "it from amps_u reads the FIRST sky batch only and silently restores "
        "the defect this test exists for")
    # ...and nothing may reassign the accumulator AFTER the loop, which would
    # discard the second batch just as effectively.
    loop_line = loops[0].lineno
    late = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
            and n.lineno > loop_line + len(loops[0].body)
            and any(isinstance(t, ast.Name) and t.id == "amp_u_emp"
                    for t in n.targets)]
    assert not late, (
        "amp_u_emp is reassigned after the re-draw loop (line %s); that "
        "discards the second batch exactly as dropping the in-loop update "
        "would" % [n.lineno for n in late])


def test_dist_grid_tol_is_forwarded_and_not_hardcoded():
    """F3/N1.  Hardcoding the module default at the call site leaves
    --distance-grid-tol silently inert while dist_grid_info keeps echoing the
    user's value -- indistinguishable from working."""
    data = _synth()
    seen = {}
    for tol in (5e-1, 1e-3):
        like = _like(data, "loguniform", n_grid=64, dist_grid_tol=tol)
        seen[tol] = int(like.x_grid.shape[0])
        assert like.dist_grid_info["tol"] == tol
        assert seen[tol] == loguniform_grid_size(
            D_MIN, D_MAX, like.dist_grid_info["rho_max"], tol), (
            "the grid was not built with the tol the caller asked for")
    assert seen[5e-1] < seen[1e-3], (
        "a looser tolerance must produce FEWER nodes; the argument is inert")


def test_shipped_tolerance_constant_is_pinned_by_value():
    """F3/N8.  Every other test derives its expectation from
    DIST_GRID_TOL_DEFAULT, so changing the constant moved the whole suite with
    it and pinned nothing.  This asserts the shipped value and one node count
    computed from it, both as literals."""
    assert DIST_GRID_TOL_DEFAULT == 1e-2
    assert abs(loguniform_spacing_for_tolerance(1e-2) - 1.930171443998096) < 1e-12
    # rho_max = 55.06965 is the reference configuration's measured value
    assert loguniform_grid_size(1.0, 10000.0, 55.06965048323435, 1e-2) == 264


def test_driver_distance_grid_tol_defaults_to_the_none_sentinel():
    """F3/N2.  A concrete default makes `is not None` true on every run, which
    fires the mode guard for every mode that does not implement the option."""
    tree = ast.parse(_driver_src())
    call = _option_call(tree, "--distance-grid-tol")
    assert call is not None
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw["default"], ast.Constant) and kw["default"].value is None, (
        "--distance-grid-tol must default to the None sentinel; a concrete "
        "default is indistinguishable from the user having passed one")


def test_driver_never_passes_the_raw_option_as_the_node_count():
    """F3/N9.  Reverting the sentinel refactor makes every default
    flowmc-phipsimarg run die with `TypeError: 'NoneType' object cannot be
    interpreted as an integer` -- a change to a LIVE default path.  The
    executable half of this is the subprocess test below; this is the precise
    half, because an AST guard can name the call site."""
    tree = ast.parse(_driver_src())
    bad = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
           for k in (n.keywords or [])
           if k.arg == "n_grid" and isinstance(k.value, ast.Attribute)
           and k.value.attr == "distance_grid_points"]
    assert not bad, ("n_grid= must receive the RESOLVED local, not the raw "
                     "optparse value, which is None unless the user passed it")
    names = [k.value.id for n in ast.walk(tree) if isinstance(n, ast.Call)
             for k in (n.keywords or [])
             if k.arg == "n_grid" and isinstance(k.value, ast.Name)]
    assert names and set(names) == {"n_dist_grid"}, (
        "every n_grid= call site must use the one resolved local; got %r" % (names,))


def _run_driver(args, timeout=240):
    import subprocess, tempfile
    env = dict(os.environ, PYTHONPATH=str(_CODE), OMP_NUM_THREADS="1",
               JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    return subprocess.run([sys.executable, str(_CODE / "bin"
                           / "integrate_likelihood_extrinsic_jax")] + args,
                          capture_output=True, text=True, env=env,
                          cwd=tempfile.mkdtemp(), timeout=timeout)


def _driver_module():
    """Import the driver BY PATH (it has no .py suffix and is not importable
    normally).  Gives in-process access to the real build_parser and
    check_critical_and_report, so the parse-time refusals below are executable
    coverage of the shipping functions rather than five 8-second subprocesses."""
    import importlib.util, importlib.machinery
    path = str(_CODE / "bin" / "integrate_likelihood_extrinsic_jax")
    spec = importlib.util.spec_from_loader(
        "_ilejax_under_test", importlib.machinery.SourceFileLoader(
            "_ilejax_under_test", path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_refuses_the_bad_combinations_at_PARSE_time():
    """F8.  These used to surface as a raw ValueError from the wrapper AFTER a
    full precompute; they now fail during option validation.  Executable: the
    real check_critical_and_report is called, not grepped.  (The only other
    gated test that runs this driver runs --help, which exits before validation
    happens at all.)"""
    cases = [
        (["--mode", "flowmc-phipsimarg", "--distance-grid-scheme", "loguniform"],
         "requires --angle-marg-scheme"),
        (["--mode", "flowmc-phimarg", "--distance-grid-scheme", "loguniform"],
         "applies only to --mode flowmc-phipsimarg"),
        (["--mode", "flowmc-phipsimarg", "--distance-grid-scheme", "loguniform",
          "--angle-marg-scheme", "exact", "--distance-grid-points", "256"],
         "both set the distance node count"),
        (["--distance-grid-tol", "0.1"], "applies only to"),
        (["--distance-grid-scheme", "adaptive"], "invalid choice"),
        # the option's VALUE, not just its combinations: the valid range is a
        # closed-form constant, so there is no reason to make the user sit
        # through a precompute to be told 7.0 is not a fractional error.
        (["--mode", "flowmc-phipsimarg", "--distance-grid-scheme", "loguniform",
          "--angle-marg-scheme", "exact", "--distance-grid-tol", "7.0"],
         "--distance-grid-tol must be in (0, 2)"),
        (["--mode", "flowmc-phipsimarg", "--distance-grid-scheme", "loguniform",
          "--angle-marg-scheme", "exact", "--distance-grid-tol", "-1"],
         "--distance-grid-tol must be in (0, 2)"),
    ]
    import contextlib, io
    mod = _driver_module()
    for args, expect in cases:
        optp = mod.build_parser()
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                opts, _ = optp.parse_args(list(args))
                mod.check_critical_and_report(opts, optp)
        except SystemExit:
            assert expect in err.getvalue(), (
                "%r: expected %r, got %r" % (args, expect, err.getvalue()[-400:]))
        else:
            raise AssertionError("%r must be refused, it was accepted" % (args,))


def test_driver_reaches_and_uses_the_resolved_node_count_on_a_real_input():
    """F3/N9, EXECUTABLE.  Runs the driver past parsing on a real (tiny)
    injection, far enough to print the resolved distance grid, then stops on a
    known validation error.  ~12 s.  Under the reverted sentinel this dies with
    TypeError at that very print instead -- which is exactly the live-default
    breakage no other gated test could see."""
    p = _run_driver([
        "--inj-mode", "--mass1", "35", "--mass2", "30", "--inj-deltaF", "0.25",
        "--inj-ra", "1.2", "--inj-dec", "0.3", "--inj-psi", "0.5",
        "--inj-incl", "1.05", "--inj-phiref", "0.0", "--inj-distance", "633.92",
        "--inj-detectors", "H1,L1", "--fmin-template", "40", "--fmax", "400.0",
        "--l-max", "2", "--approximant", "SEOBNRv4", "--reference-freq", "100.0",
        "--srate", "1024", "--d-min", "1", "--d-max", "10000",
        "--distance-marginalization", "--mode", "flowmc-phipsimarg",
        "--angle-marg-scheme", "grid", "--n-phi", "4", "--n-psi", "4",
        "--time-marginalization-quadrature", "bandlimited",
        "--n-max", "1", "--n-chunk", "1"])
    out = p.stdout + p.stderr
    assert "TypeError" not in out, (
        "the driver did not resolve --distance-grid-points: %s" % out[-500:])
    assert "(grid=256," in out, (
        "the resolved default node count did not reach the run log: %s" % out[-500:])
    assert "time_quadrature='bandlimited' is not valid" in out, (
        "expected the run to stop on the known validation error; got %s" % out[-500:])
