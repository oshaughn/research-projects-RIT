"""``--distance-gh-nodes``: make the per-sample Gauss-Hermite distance
quadrature reachable by an ILE argument, and the driver's warn-not-ignore
compatibility notes for three previously-silent knobs.

RO'S dispositions (2026-09-08): (a) unreachable code (the per-sample distance
quadrature) gets an ILE-style CLI argument; (b) missing knobs stay no-op for
compatibility but must WARN.  This file gates both, plus the CLI/env
compatibility contract the new option must honour without import-order
fragility (``core.set_distmarg_gh_nodes`` / ``get_distmarg_gh_nodes``, read at
CALL time by every consumer -- see core.py and the module docstring there).

Layout:
  * CLI/env resolution and refusal (parse-time, no subprocess: the real
    ``check_critical_and_report`` is called in-process, exactly as
    test_distance_grid_loguniform.py's parse-time tests do).
  * The three "accepted but IGNORED, and SAYS SO" notes: --phase-marginalization
    on the phi_ref-analytic modes, --sky-coordinates network on the modes that
    do not implement it, --d-prior on any non-volumetric value.
  * A numeric liveness check: the resolved node count actually changes the
    constructed likelihood's VALUE (not just an echoed CLI flag) on a cheap
    synthetic packed-data fixture -- no lal, no frames.
  * One subprocess check against the real driver with --inj-mode (tiny
    injection, stopped at a known post-construction validation error, exactly
    as test_distance_grid_loguniform.py's own subprocess test does) that the
    resolved count reaches the run log end to end.

Run:
  PYTHONPATH=<...>/Code  python -m pytest -q test/jax/test_distance_gh_nodes_cli.py
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import core as core_mod
from RIFT.likelihood.jax_ile import build_likelihood_data
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiMargLikelihood

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
_JAXDRIVER = os.path.join(_CODE, "bin", "integrate_likelihood_extrinsic_jax")


def _driver_module():
    """Import the driver BY PATH (no .py suffix, so plain import cannot see
    it).  Gives in-process access to the real build_parser/
    check_critical_and_report, so the checks below are executable coverage of
    the shipping functions, not a grep of the source."""
    assert os.path.exists(_JAXDRIVER), "driver missing: %s" % _JAXDRIVER
    loader = importlib.machinery.SourceFileLoader("_ghnodes_driver", _JAXDRIVER)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture
def saved_gh_env():
    """Every test that touches --distance-gh-nodes mutates process-global
    state (core._DISTMARG_GH_N and/or os.environ), exactly like the existing
    JAX_ILE_DISTMARG_GH tests in test_distance_grid_loguniform.py -- restore
    both, unconditionally, so one test's mutation cannot leak into the next."""
    saved_env = os.environ.get("JAX_ILE_DISTMARG_GH")
    saved_core = core_mod._DISTMARG_GH_N
    try:
        yield
    finally:
        if saved_env is None:
            os.environ.pop("JAX_ILE_DISTMARG_GH", None)
        else:
            os.environ["JAX_ILE_DISTMARG_GH"] = saved_env
        core_mod._DISTMARG_GH_N = saved_core


def _parse(mod, args):
    optp = mod.build_parser()
    opts, _ = optp.parse_args(list(args))
    return optp, opts


def _run_checked(mod, args):
    """check_critical_and_report(), capturing stdout+stderr as one string."""
    optp, opts = _parse(mod, args)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        mod.check_critical_and_report(opts, optp)
    return opts, out.getvalue() + err.getvalue()


def _run_refused(mod, args):
    """Like _run_checked, but the call is expected to SystemExit; returns the
    captured text.  Raises AssertionError if it does not exit."""
    optp, opts = _parse(mod, args)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            mod.check_critical_and_report(opts, optp)
    except SystemExit:
        return out.getvalue() + err.getvalue()
    raise AssertionError("expected a refusal (SystemExit); accepted %r" % (args,))


# ---------------------------------------------------------------------------
# CLI/env resolution and refusal
# ---------------------------------------------------------------------------

def test_option_is_registered_with_default_none():
    """Default must be None, not 0: 0 is a legal explicit value (BLOCKER,
    external review of this PR) and a 0-default makes an explicit
    ``--distance-gh-nodes 0`` indistinguishable from "not passed", so a
    nonzero JAX_ILE_DISTMARG_GH would silently win instead of losing to the
    explicit CLI 0."""
    mod = _driver_module()
    optp = mod.build_parser()
    opt = next(o for g in ([optp] + optp.option_groups)
              for o in getattr(g, "option_list", [])
              if "--distance-gh-nodes" in (o._long_opts or []))
    assert opt.default is None, (
        "unreachable feature (a) must default to None, not 0, so an "
        "explicit 0 is distinguishable from not-passed: got %r" % (opt.default,))
    assert opt.type == "int"


def test_cli_alone_resolves_and_mutates_core_state(saved_gh_env):
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    opts, text = _run_checked(mod, ["--distance-gh-nodes", "48"])
    assert opts._distance_gh_nodes_resolved == 48
    assert core_mod.get_distmarg_gh_nodes() == 48, (
        "set_distmarg_gh_nodes() must reach core's live module state")
    assert "nodes=48" in text and "--distance-gh-nodes" in text


def test_env_alone_is_still_honoured(saved_gh_env):
    mod = _driver_module()
    os.environ["JAX_ILE_DISTMARG_GH"] = "17"
    opts, text = _run_checked(mod, [])
    assert opts._distance_gh_nodes_resolved == 17
    assert core_mod.get_distmarg_gh_nodes() == 17
    assert "nodes=17" in text and "JAX_ILE_DISTMARG_GH" in text


def test_default_is_zero_and_legacy_grid_is_named(saved_gh_env):
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    opts, text = _run_checked(mod, [])
    assert opts._distance_gh_nodes_resolved == 0
    assert core_mod.get_distmarg_gh_nodes() == 0
    assert "nodes=0" in text and "legacy uniform grid" in text


def test_cli_and_env_agreeing_is_accepted(saved_gh_env):
    mod = _driver_module()
    os.environ["JAX_ILE_DISTMARG_GH"] = "9"
    opts, text = _run_checked(mod, ["--distance-gh-nodes", "9"])
    assert opts._distance_gh_nodes_resolved == 9
    assert core_mod.get_distmarg_gh_nodes() == 9


def test_cli_and_env_conflict_is_REFUSED_not_reconciled(saved_gh_env):
    """(a): the CLI must WIN, but silently picking one of two disagreeing
    values is exactly the drift a compatibility path exists to prevent -- so
    two different nonzero values must refuse, not resolve."""
    mod = _driver_module()
    os.environ["JAX_ILE_DISTMARG_GH"] = "12"
    text = _run_refused(mod, ["--distance-gh-nodes", "40"])
    assert "--distance-gh-nodes" in text
    assert "JAX_ILE_DISTMARG_GH" in text
    assert "conflicts" in text


def test_refusal_does_not_mutate_core_state(saved_gh_env):
    """A refused command line must never apply its (disputed) resolution --
    mirrors test_distance_grid_loguniform.py's own precondition discipline for
    this exact global."""
    mod = _driver_module()
    core_mod._DISTMARG_GH_N = 0
    os.environ["JAX_ILE_DISTMARG_GH"] = "12"
    try:
        _run_refused(mod, ["--distance-gh-nodes", "40"])
    finally:
        pass
    assert core_mod._DISTMARG_GH_N == 0, (
        "a refused --distance-gh-nodes/JAX_ILE_DISTMARG_GH conflict must not "
        "mutate core._DISTMARG_GH_N; got %r" % (core_mod._DISTMARG_GH_N,))


def test_cli_route_still_trips_the_loguniform_incompatibility(saved_gh_env):
    """F2's driver-side refusal (originally env-only) must also fire when the
    node count arrives via --distance-gh-nodes, not just JAX_ILE_DISTMARG_GH --
    the parse-time check must read the RESOLVED value, not re-read the
    environment directly (that would silently accept the CLI route)."""
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    text = _run_refused(mod, [
        "--mode", "flowmc-phipsimarg", "--distance-grid-scheme", "loguniform",
        "--angle-marg-scheme", "auto", "--distance-gh-nodes", "32"])
    assert "distance-gh-nodes" in text or "JAX_ILE_DISTMARG_GH" in text
    assert "inert" in text


def test_explicit_cli_zero_with_nonzero_env_is_REFUSED(saved_gh_env):
    """The BLOCKER this file exists to close: with the old ``0``-default,
    ``getattr(opts, "distance_gh_nodes", 0) or 0`` could not tell an explicit
    ``--distance-gh-nodes 0`` apart from "not passed", so JAX_ILE_DISTMARG_GH
    silently won.  An explicit 0 against a nonzero env is a conflict like any
    other -- it must refuse, not resolve to 0 and not resolve to 64."""
    mod = _driver_module()
    os.environ["JAX_ILE_DISTMARG_GH"] = "64"
    text = _run_refused(mod, ["--distance-gh-nodes", "0"])
    assert "--distance-gh-nodes" in text
    assert "JAX_ILE_DISTMARG_GH" in text
    assert "conflicts" in text


def test_explicit_cli_16_with_agreeing_env_16_is_accepted(saved_gh_env):
    mod = _driver_module()
    os.environ["JAX_ILE_DISTMARG_GH"] = "16"
    opts, text = _run_checked(mod, ["--distance-gh-nodes", "16"])
    assert opts._distance_gh_nodes_resolved == 16
    assert core_mod.get_distmarg_gh_nodes() == 16
    assert "nodes=16" in text


def test_env_16_with_no_cli_resolves_to_16_and_banner_says_so(saved_gh_env):
    mod = _driver_module()
    os.environ["JAX_ILE_DISTMARG_GH"] = "16"
    opts, text = _run_checked(mod, [])
    assert opts._distance_gh_nodes_resolved == 16
    assert core_mod.get_distmarg_gh_nodes() == 16
    assert "nodes=16" in text and "JAX_ILE_DISTMARG_GH" in text


def test_cli_wins_regression_env_16_cli_32_must_refuse(saved_gh_env):
    """Mutation-test target: a priority reversal (env wins over CLI) passes
    every OTHER test in this file just as readily as "CLI wins" does, because
    most cases here use only one of the two knobs.  This is the one case that
    tells them apart: CLI 32 disagrees with env 16, so the correct policy
    (CLI wins when given; conflicts are refused, never silently reconciled)
    must refuse.  A reversed-priority implementation that silently resolves
    to the env value (16) instead of refusing passes _run_refused's
    SystemExit check trivially only if it also refuses -- if it does not,
    _run_refused raises AssertionError itself, so this test fails loudly
    rather than passing on an unresolved value."""
    mod = _driver_module()
    os.environ["JAX_ILE_DISTMARG_GH"] = "16"
    text = _run_refused(mod, ["--distance-gh-nodes", "32"])
    assert "--distance-gh-nodes" in text
    assert "32" in text and "16" in text
    assert "JAX_ILE_DISTMARG_GH" in text
    assert "conflicts" in text


# ---------------------------------------------------------------------------
# (b) missing/scoped knobs: no-op for compatibility, but must WARN
# ---------------------------------------------------------------------------

def test_phase_marginalization_ignored_note_on_phase_analytic_modes(saved_gh_env):
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    for mode in sorted(mod._PHASE_ANALYTIC_MODES):
        _, text = _run_checked(mod, ["--mode", mode, "--phase-marginalization"])
        assert "--phase-marginalization" in text and "IGNORED" in text, (
            "mode %s: expected an IGNORED note, got %r" % (mode, text))
        assert mode in text


def test_phase_marginalization_note_absent_where_the_flag_is_honoured(saved_gh_env):
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    for mode in ("laplace-is", "prior-mc", "nuts"):
        _, text = _run_checked(mod, [
            "--mode", mode, "--distance-marginalization",
            "--phase-marginalization"])
        assert "--phase-marginalization" not in text, (
            "mode %s honours --phase-marginalization; it must not be reported "
            "IGNORED: %r" % (mode, text))


def test_sky_coordinates_ignored_note_off_multistart(saved_gh_env):
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    for mode in ("nuts", "laplace-is", "flowmc-phimarg"):
        _, text = _run_checked(mod, ["--mode", mode,
                                    "--sky-coordinates", "network"])
        assert "--sky-coordinates" in text and "IGNORED" in text, (
            "mode %s: expected an IGNORED note, got %r" % (mode, text))


def test_sky_coordinates_note_absent_on_multistart_nuts(saved_gh_env):
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    _, text = _run_checked(mod, ["--mode", "multistart-nuts",
                                "--sky-coordinates", "network"])
    assert "--sky-coordinates" not in text, (
        "multistart-nuts implements --sky-coordinates network; got %r" % (text,))


def test_sky_coordinates_default_equatorial_never_notes(saved_gh_env):
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    for mode in ("nuts", "multistart-nuts", "laplace-is"):
        _, text = _run_checked(mod, ["--mode", mode])
        assert "--sky-coordinates" not in text


def test_d_prior_ignored_note_for_a_real_alternative_prior(saved_gh_env):
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    _, text = _run_checked(mod, ["--d-prior", "cosmo"])
    assert "--d-prior" in text and "IGNORED" in text and "volumetric" in text


@pytest.mark.parametrize("value", ["Euclidean", "euclidean", "Volumetric", "volumetric"])
def test_d_prior_no_note_for_the_volumetric_prior_itself(saved_gh_env, value):
    """Euclidean/volumetric (any case) IS the JAX driver's prior, so stating
    it explicitly is not a deviation and must not print a note."""
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    _, text = _run_checked(mod, ["--d-prior", value])
    assert "--d-prior" not in text, "no note expected for --d-prior %s: %r" % (value, text)


def test_d_prior_excluded_from_the_generic_ignored_bag(saved_gh_env):
    """--d-prior must not ALSO appear in the blanket 'accepted but IGNORED'
    list (it would be reported twice, once generically and once with the
    substantive message) -- moved out of `implemented` per the task, and
    explicitly excluded from the generic bag alongside that."""
    mod = _driver_module()
    os.environ.pop("JAX_ILE_DISTMARG_GH", None)
    _, text = _run_checked(mod, ["--d-prior", "cosmo"])
    assert text.count("--d-prior") == 1, (
        "expected exactly one --d-prior mention (the substantive note); got %r" % (text,))


# ---------------------------------------------------------------------------
# Numeric liveness: the resolved node count must change the VALUE, not just
# an echoed CLI flag.  Cheap synthetic packed data -- no lal, no frames.
# ---------------------------------------------------------------------------

def _synth(scale=1.0, seed=3, modes=((2, 2), (2, -2)), npts=32,
          deltaT=1.0 / 1024, kappa_boost=6.0):
    """Structurally-faithful packed data (Hermitian PD U, complex-symmetric V);
    the same construction test_distance_grid_loguniform.py uses, duplicated
    here so this file has no cross-file import-order dependency.  kappa_boost
    is large: the GH quadrature only differs measurably from the fixed grid
    once the distance peak is narrow (high effective SNR)."""
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


def test_gh_nodes_changes_the_constructed_likelihood_value(saved_gh_env):
    """The core claim of (a): setting the resolved node count through
    core.set_distmarg_gh_nodes -- exactly what the driver's CLI path now does
    -- must change what JAXDistPhiMargLikelihood.value() returns, at a FIXED
    theta and FIXED data.  A fresh likelihood object per setting (each gets
    its own jax.jit closure), so no stale JIT trace masks the difference."""
    data = _synth()
    theta4 = np.array([1.1, -0.3, 0.6, 1.0])

    core_mod.set_distmarg_gh_nodes(0)
    like_grid = JAXDistPhiMargLikelihood(data, 1.0, 10000.0, nphi=8, n_grid=64,
                                         interp="sinc")
    v_grid = like_grid.value(theta4)

    core_mod.set_distmarg_gh_nodes(48)
    like_gh = JAXDistPhiMargLikelihood(data, 1.0, 10000.0, nphi=8, n_grid=64,
                                       interp="sinc")
    v_gh = like_gh.value(theta4)

    assert np.isfinite(v_grid) and np.isfinite(v_gh)
    assert abs(v_gh - v_grid) > 1e-6, (
        "distance-gh-nodes=48 vs 0 must give a DIFFERENT lnL on this fixture "
        "(grid=%.6f, gh=%.6f) -- if these agree, the resolved count is not "
        "reaching the kernel" % (v_grid, v_gh))


def test_gh_nodes_zero_reproduces_the_legacy_grid_exactly(saved_gh_env):
    """The companion safety property: --distance-gh-nodes 0 (the default)
    must be BIT IDENTICAL to never having called the setter at all."""
    data = _synth()
    theta4 = np.array([1.1, -0.3, 0.6, 1.0])

    core_mod._DISTMARG_GH_N = 0
    like_untouched = JAXDistPhiMargLikelihood(data, 1.0, 10000.0, nphi=8,
                                              n_grid=64, interp="sinc")
    v_untouched = like_untouched.value(theta4)

    core_mod.set_distmarg_gh_nodes(48)
    core_mod.set_distmarg_gh_nodes(0)   # explicit round trip back to off
    like_reset = JAXDistPhiMargLikelihood(data, 1.0, 10000.0, nphi=8, n_grid=64,
                                          interp="sinc")
    v_reset = like_reset.value(theta4)

    assert v_untouched == v_reset, (v_untouched, v_reset)


# ---------------------------------------------------------------------------
# One subprocess check against the real driver (--inj-mode, tiny budget)
# ---------------------------------------------------------------------------

def _run_driver(args, timeout=240):
    env = dict(os.environ, PYTHONPATH=_CODE, OMP_NUM_THREADS="1",
              JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    env.pop("JAX_ILE_DISTMARG_GH", None)
    return subprocess.run([sys.executable, _JAXDRIVER] + args,
                          capture_output=True, text=True, env=env,
                          cwd=tempfile.mkdtemp(), timeout=timeout)


_INJ_ARGS = [
    "--inj-mode", "--mass1", "35", "--mass2", "30", "--inj-deltaF", "0.25",
    "--inj-ra", "1.2", "--inj-dec", "0.3", "--inj-psi", "0.5",
    "--inj-incl", "1.05", "--inj-phiref", "0.0", "--inj-distance", "633.92",
    "--inj-detectors", "H1,L1", "--fmin-template", "40", "--fmax", "400.0",
    "--l-max", "2", "--approximant", "SEOBNRv4", "--reference-freq", "100.0",
    "--srate", "1024", "--d-min", "1", "--d-max", "10000",
    "--distance-marginalization", "--mode", "flowmc-phipsimarg",
    "--angle-marg-scheme", "grid", "--n-phi", "4", "--n-psi", "4",
    # Deliberately invalid so the run stops right after construction --
    # cheap, and reaches the same point test_distance_grid_loguniform.py's
    # own "reaches and uses" subprocess test relies on.
    "--time-marginalization-quadrature", "bandlimited",
    "--n-max", "1", "--n-chunk", "1",
]


def test_driver_reaches_and_reports_the_resolved_gh_nodes_end_to_end():
    """--distance-gh-nodes must reach the run log through the real subprocess
    entry point, exactly as the equivalent test does for
    --distance-grid-points in test_distance_grid_loguniform.py."""
    p = _run_driver(_INJ_ARGS + ["--distance-gh-nodes", "48"])
    out = p.stdout + p.stderr
    assert "TypeError" not in out, out[-500:]
    assert "nodes=48" in out and "--distance-gh-nodes" in out, out[-800:]
    assert "time_quadrature='bandlimited' is not valid" in out, out[-500:]


def test_driver_reports_zero_nodes_without_the_flag():
    p = _run_driver(_INJ_ARGS)
    out = p.stdout + p.stderr
    assert "nodes=0" in out and "legacy uniform grid" in out, out[-500:]
