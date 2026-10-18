"""'log-hermite' as something a RUN can ask for, in the mode the rule exists for.

``test_time_log_hermite.py`` tests the RULE.  This file tests the two seams between a
command line and that rule.  Both were silently broken when the rule landed, and both
failure modes are invisible to a test of the rule itself:

  SELECTABILITY.  ``bin/integrate_likelihood_extrinsic_jax`` re-typed its optparse
  ``choices=`` tuple as ``("simpson", "bandlimited")`` instead of deriving it from
  ``_TIME_QUAD_CHOICES``.  So no command line, ini or ``.sub`` file could request
  'log-hermite' -- only a direct Python caller could -- while every unit test of the
  rule passed.  The NON-jax driver has been guarded against exactly this by
  ``test_time_marginalization_quadrature_pipeline.py::
  test_argparse_choices_are_pinned_to_the_library_tuple``; the jax driver had no
  equivalent, which is why the same gap opened here.

  DISPATCH.  ``_time_marginalize_terminal``'s ``if time_quadrature == "log-hermite"``
  branch was untested.  Two mutations were run against the whole jax time-quadrature
  suite and BOTH left it green:

      ``if False:``                 -> a 'log-hermite' request silently returns the
                                       BAND-LIMITED FFT value, i.e. precisely the rule
                                       this arm exists because it may not use.
      ``2.0 * data.deltaT``         -> a +0.693-nat units error.

  ``test_registered_as_a_choice`` asserted tuple MEMBERSHIP; nothing bound the string
  to the function.  The tests below do, bitwise.

THE MODE.  ``bandlimited`` asserts that ``exp(lnL_t)`` is band-limited, which it is not
after a nonlinear (distance/phase/polarization) marginalization, so
``_validate_nonlinear_time_quadrature`` REFUSES it on those endpoints and Simpson was
the only shipped alternative there.  That -- phi_ref and psi marginalized inside the
likelihood -- is the regime tested in part C, not a fixed-angle configuration where
``bandlimited`` is available anyway.

TIER.  Everything here is in the per-PR gate (``.travis/test-jax.sh``).  There is no
full ``integrate_likelihood_extrinsic_jax`` run against frame data in any tier: nothing
in ``test/jax/`` ships frames, and the only data-driven driver script,
``demo_real_data.py``, needs real strain.  The driver is therefore covered here at its
CLI seam by subprocess and at its likelihood seam through the same wrapper class
``analyze_one`` constructs for ``--mode flowmc-phipsimarg``.
"""
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from RIFT.likelihood.jax_ile import core  # noqa: E402
from RIFT.likelihood.jax_ile.core import (  # noqa: E402
    TIME_QUAD_DEFAULT, _TIME_QUAD_CHOICES, _time_marginalize_log_hermite,
    fused_log_likelihood_distphipsimarg, phi_ref_grid, psi_grid,
    make_distance_grid)
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood  # noqa: E402

from test_angle_marg_exact import make_synth, RA, DEC, INCL, INTERP  # noqa: E402

_CODE = pathlib.Path(__file__).resolve().parents[2]
_DRIVER = _CODE / "bin" / "integrate_likelihood_extrinsic_jax"


# --------------------------------------------------------------------------- A
# Selectability: the library tuple must reach the command line.
# ---------------------------------------------------------------------------
def _option_call_source():
    """Source of the single add_option() call that registers the flag.

    Via AST, not a line grep: the call spans several lines, and ``choices=`` sits on
    a different one from the flag name.
    """
    import ast
    src = _DRIVER.read_text()
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_option"):
            continue
        if any(isinstance(a, ast.Constant)
               and a.value == "--time-marginalization-quadrature"
               for a in node.args):
            hits.append(ast.get_source_segment(src, node))
    assert len(hits) == 1, hits
    return hits[0]


def test_the_driver_offers_exactly_the_library_choices():
    """THE regression guard for finding 1, stated behaviourally so it does not depend
    on how the derivation is spelled: a choice added to the library tuple and not
    reaching the CLI fails HERE, at the commit that adds it, rather than being
    discovered by a user whose command line is rejected."""
    parser = _load_driver().build_parser()
    opt = parser.get_option("--time-marginalization-quadrature")
    assert tuple(opt.choices) == tuple(_TIME_QUAD_CHOICES), opt.choices


def test_the_driver_derives_its_choices_from_the_library_tuple():
    """The source form of the same rule, mirroring the non-jax driver's guard.  The
    behavioural test above passes today for a re-typed literal that happens to match;
    this one refuses the literal itself, which is the thing that rots."""
    call = _option_call_source()
    assert "choices=list(_TIME_QUAD_CHOICES)" in call, call
    body = call.split("help=")[0]
    assert "simpson" not in body, (
        "a re-typed literal choice list is how 'log-hermite' became unreachable", body)


def test_the_driver_derives_its_default_from_the_library_default():
    call = _option_call_source()
    assert "default=TIME_QUAD_DEFAULT" in call, call


def test_making_the_rule_selectable_did_not_make_it_the_default():
    """The whole point of an opt-in.  Deriving the CLI default from the library
    default means this one assertion covers both."""
    assert TIME_QUAD_DEFAULT == "simpson"
    parser = _load_driver().build_parser()
    opts, _ = parser.parse_args([])
    assert opts.time_marginalization_quadrature == "simpson"


def _load_driver():
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader("_jaxdrv_sel", str(_DRIVER))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def _run_driver(*args):
    """Run the real driver in a SUBPROCESS.

    Deliberately not ``--help``: optparse prints help and exits BEFORE it validates
    any option value, so a --help test proves nothing about whether a choice is
    accepted.  These invocations carry no data, so a value that survives option
    parsing lands on the driver's own '--event-time is required' check -- which is
    therefore the positive evidence that parsing accepted it.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_CODE) + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = "cpu"
    p = subprocess.run([sys.executable, str(_DRIVER)] + list(args), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    return p.stdout.decode("utf-8", "replace")


@pytest.mark.parametrize("choice", list(_TIME_QUAD_CHOICES))
def test_every_library_choice_is_accepted_by_the_real_parser(choice):
    out = _run_driver("--time-marginalization-quadrature", choice)
    assert "invalid choice" not in out, out[-2000:]
    assert "--event-time is required" in out, (
        "did not reach the driver's own validation; the run failed earlier than "
        "the option value", out[-2000:])


def test_a_misspelling_is_refused_and_names_log_hermite():
    """A choice absorbed as 'not recognised' would silently run a different rule."""
    out = _run_driver("--time-marginalization-quadrature", "loghermite")
    assert "invalid choice" in out, out[-2000:]
    assert "log-hermite" in out, out[-2000:]


@pytest.mark.parametrize("extra", [
    ["--phase-marginalization"],
    ["--mode", "flowmc-phipsimarg", "--distance-marginalization"],
])
def test_log_hermite_is_selectable_in_the_phase_marginalized_configurations(extra):
    """Selectability is not a property of the flag alone: these are the command lines
    where 'bandlimited' is refused and this rule is the only alternative to Simpson."""
    out = _run_driver("--time-marginalization-quadrature", "log-hermite", *extra)
    assert "invalid choice" not in out, out[-2000:]
    assert "--event-time is required" in out, out[-2000:]


# --------------------------------------------------------------------------- B
# Dispatch: the string must be bound to the function, not merely to the tuple.
# ---------------------------------------------------------------------------
def _seam_case(npts=257, srate=4096.0, h_over_sigma=2.0, offset_frac=0.3):
    """A packed data object plus an OFF-GRID, under-resolved lnL(t) row.

    Under-resolved on purpose: the three rules must be far enough apart that a
    substitution cannot hide inside a tolerance.
    """
    data = make_synth(scale=2.0, npts=npts, deltaT=1.0 / srate)
    h = float(data.deltaT)
    sigma = h / h_over_sigma
    t = (np.arange(npts) - npts // 2) * h
    lnL = -((t - offset_frac * h) ** 2) / (2.0 * sigma ** 2)
    return data, jnp.asarray(lnL[None, :])


@pytest.mark.parametrize("bandlimited_safe", [False, True])
def test_the_seam_returns_exactly_the_log_hermite_rule(bandlimited_safe):
    """Bitwise, both with and without the band-limited permission.

    ``bandlimited_safe=True`` is the leg that catches a deleted branch: with the
    permission granted, a fallen-through 'log-hermite' returns the FFT value instead
    of raising, which is the silent substitution.
    """
    data, y = _seam_case()
    got = core._time_marginalize_terminal(y, data, "log-hermite",
                                          bandlimited_safe=bandlimited_safe)
    want = _time_marginalize_log_hermite(y, data.deltaT)
    assert np.array_equal(np.asarray(got), np.asarray(want)), (got, want)


def test_the_seam_gives_a_distinct_answer_for_every_choice():
    """If two selections agreed, the bitwise check above could be satisfied by the
    wrong branch.  These three must be measurably apart on this row."""
    data, y = _seam_case()
    vals = {c: float(core._time_marginalize_terminal(
                y, data, c, bandlimited_safe=True)[0])
            for c in _TIME_QUAD_CHOICES}
    assert all(np.isfinite(v) for v in vals.values()), vals
    assert abs(vals["log-hermite"] - vals["simpson"]) > 1e-3, vals
    assert vals["log-hermite"] != vals["bandlimited"], vals


def test_the_seam_hands_the_rule_the_sample_spacing_itself():
    """The second surviving mutation was ``2.0 * data.deltaT``: a pure units error,
    worth ln 2 = 0.693 nats on a resolved peak and invisible to every test that
    compares log-hermite only against itself."""
    data, y = _seam_case()
    got = float(core._time_marginalize_terminal(y, data, "log-hermite")[0])
    doubled = float(_time_marginalize_log_hermite(y, 2.0 * data.deltaT)[0])
    assert abs(doubled - got) > 0.5, (got, doubled)
    assert got != doubled


def test_an_unknown_rule_is_refused_at_the_seam():
    data, y = _seam_case()
    with pytest.raises(ValueError, match="time_quadrature"):
        core._time_marginalize_terminal(y, data, "log_hermite")


# --------------------------------------------------------------------------- C
# The mode that matters: phi_ref AND psi marginalized inside the likelihood.
# ---------------------------------------------------------------------------
_D_MIN, _D_MAX = 30.0, 3000.0
_NPHI, _NPSI, _NGRID = 32, 8, 64


def _phipsi_data():
    return make_synth(scale=2.0)


def test_bandlimited_is_refused_exactly_where_log_hermite_is_offered():
    """The premise of the whole PR, asserted rather than assumed: on this endpoint
    'bandlimited' is not available, so before this rule Simpson had no alternative."""
    data = _phipsi_data()
    with pytest.raises(ValueError, match="bandlimited"):
        JAXDistPhiPsiMargLikelihood(data, _D_MIN, _D_MAX, nphi=_NPHI, npsi=_NPSI,
                                    n_grid=_NGRID, interp=INTERP, angle_marg="grid",
                                    time_quadrature="bandlimited")
    like = JAXDistPhiPsiMargLikelihood(data, _D_MIN, _D_MAX, nphi=_NPHI, npsi=_NPSI,
                                       n_grid=_NGRID, interp=INTERP, angle_marg="grid",
                                       time_quadrature="log-hermite")
    assert like.time_quadrature == "log-hermite"


def test_the_phi_psi_marginalized_likelihood_actually_applies_the_log_hermite_rule():
    """(b) of 'tested in that mode': not a fallback.

    ``return_lnLt=True`` hands back the very lnL(t) rows the terminal integral is
    about to consume, so the expected value can be formed from the run's OWN
    intermediate rather than from a reimplementation of the kernel.
    """
    data = _phipsi_data()
    xg, lwg = make_distance_grid(_D_MIN, _D_MAX, _NGRID, "euclidean",
                                 distMpcRef=data.distMpcRef)
    pg, sg = phi_ref_grid(_NPHI), psi_grid(_NPSI)
    args = (data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL), xg, lwg, pg, sg)
    lnL_t = fused_log_likelihood_distphipsimarg(*args, interp=INTERP, return_lnLt=True)
    got = fused_log_likelihood_distphipsimarg(*args, interp=INTERP,
                                              time_quadrature="log-hermite")
    want = _time_marginalize_log_hermite(lnL_t, data.deltaT)
    assert np.array_equal(np.asarray(got), np.asarray(want)), (got, want)

    simpson = fused_log_likelihood_distphipsimarg(*args, interp=INTERP,
                                                  time_quadrature="simpson")
    assert not np.array_equal(np.asarray(got), np.asarray(simpson)), (got, simpson)


@pytest.mark.parametrize("angle_marg", ["grid", "exact"])
def test_the_phi_psi_marginalized_lnL_is_finite_and_agrees_with_simpson(angle_marg):
    """(a) and (c): the request is honoured through the wrapper ``analyze_one``
    builds for --mode flowmc-phipsimarg, and the answer is a sane lnL.

    The two rules must AGREE loosely here -- this synthetic target's time peak is
    resolved by the grid, so a several-nat gap would mean one of them is wrong, not
    that log-hermite is better.  No accuracy claim is made or tested here; the
    accuracy question on real phase-marginalized lnL(t) rows is measured separately.
    """
    data = _phipsi_data()
    kw = dict(nphi=_NPHI, npsi=_NPSI, n_grid=_NGRID, interp=INTERP,
              angle_marg=angle_marg)
    ra, dec, incl = jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL)
    ref = JAXDistPhiPsiMargLikelihood(data, _D_MIN, _D_MAX,
                                      time_quadrature="simpson", **kw)
    lh = JAXDistPhiPsiMargLikelihood(data, _D_MIN, _D_MAX,
                                     time_quadrature="log-hermite", **kw)
    a = np.asarray(ref.log_likelihood(ra, dec, incl))
    b = np.asarray(lh.log_likelihood(ra, dec, incl))
    assert np.all(np.isfinite(b)), b
    assert np.abs(a - b).max() < 0.5, (angle_marg, a, b)


def test_log_hermite_survives_grad_through_the_phi_psi_marginalized_endpoint():
    """The differentiable arm is the reason this rule exists; a terminal integral that
    is finite but not differentiable would be useless there."""
    data = _phipsi_data()
    like = JAXDistPhiPsiMargLikelihood(data, _D_MIN, _D_MAX, nphi=_NPHI, npsi=_NPSI,
                                       n_grid=_NGRID, interp=INTERP,
                                       angle_marg="grid",
                                       time_quadrature="log-hermite")
    v, g = like.value_and_grad(np.array([float(RA[0]), float(DEC[0]), float(INCL[0])]))
    assert np.isfinite(v)
    assert np.all(np.isfinite(g)) and np.any(g != 0.0), g


@pytest.mark.parametrize("scheme", ["multipeak", "multipeak-jax"])
def test_the_multipeak_schemes_refuse_log_hermite_instead_of_ignoring_it(scheme):
    """Both multipeak controllers integrate time themselves and never read
    time_quadrature.  Accepting 'log-hermite' there returned the Simpson-path
    value bitwise while like.time_quadrature (and the driver's log) said
    'log-hermite'."""
    data = _phipsi_data()
    with pytest.raises(ValueError, match="integrates time inside its controller"):
        JAXDistPhiPsiMargLikelihood(data, _D_MIN, _D_MAX, nphi=_NPHI, npsi=_NPSI,
                                    n_grid=_NGRID, interp=INTERP, angle_marg=scheme,
                                    time_quadrature="log-hermite")
