"""The 'peak-local' angle-marg scheme, as wired into the ILE likelihood.

These test the WIRING -- that the option reaches the likelihood, produces the same
answer as the scheme it is meant to replace, records its provenance, refuses what it
cannot honour, and does NOT change any default.  The kernel's own numerics are tested in
test_joint_anglemarg_peaklocal.py.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood
from test_angle_marg_exact import make_synth, RA, DEC, INCL, INTERP


def test_peak_local_is_an_offered_choice_and_reaches_the_cli():
    """optparse builds --angle-marg-scheme's choices from ANGLE_MARG_CHOICES, so being
    in that tuple IS the CLI wiring; a scheme absent from it is unreachable."""
    assert "peak-local" in AM.ANGLE_MARG_CHOICES
    assert hasattr(AM, "fused_log_likelihood_distphipsimarg_peaklocal")


def test_peak_local_is_NOT_reachable_from_auto():
    """A scheme that changes the likelihood must not become reachable by default on the
    strength of unit tests.  'auto' must keep choosing among the schemes that have
    campaign evidence until a head-to-head pilot says otherwise."""
    for amp in (1.0, 50.0, 500.0, 5.0e4, 5.0e6):
        scheme, _ = AM.choose_angle_marg_scheme(amp)
        assert scheme != "peak-local", (amp, scheme)


@pytest.mark.parametrize("boost", [1.0, 30.0])
def test_wrapper_peak_local_matches_exact(boost):
    """The wiring's whole claim: asking for it by name gives the same likelihood as the
    scheme it parallels."""
    data = make_synth(scale=2.0, kappa_boost=boost)
    kw = dict(nphi=32, npsi=8, interp=INTERP)
    ex = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, angle_marg="exact", **kw)
    pl = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, angle_marg="peak-local", **kw)
    assert pl.angle_marg_scheme == "peak-local"
    a = np.asarray(ex._batched(jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL)))
    b = np.asarray(pl._batched(jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL)))
    assert np.abs(a - b).max() < 1e-4, (boost, a, b)


def test_peak_local_records_its_provenance():
    """This pipeline has a documented history of silently-inert flags, so the scheme
    actually used must be visible in the run record, not inferred from the request."""
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                       interp=INTERP, angle_marg="peak-local")
    info = like.angle_marg_info
    assert info["scheme"] == "peak-local"
    assert info["requested"] == "peak-local"
    assert "amp_sizing" in info, info


def test_peak_local_refuses_the_adaptive_distance_quadrature():
    """Refuse rather than silently ignore: this kernel sums the caller's distance grid
    and implements no psi-marginal node placement, exactly as the laplace branch does
    not.  The incompatibility is a property of the scheme, declared once."""
    from RIFT.likelihood.jax_ile import core as _core
    data = make_synth(scale=2.0)
    old = _core._DISTMARG_GH_N
    try:
        _core._DISTMARG_GH_N = 8
        with pytest.raises(ValueError, match="peak-local"):
            AM.fused_log_likelihood_distphipsimarg_peaklocal(
                data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
                jnp.linspace(0.4, 2.0, 8), jnp.zeros(8), interp=INTERP,
                amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)
    finally:
        _core._DISTMARG_GH_N = old


def test_peak_local_requires_amp_sizing_rather_than_guessing_it():
    """There is deliberately no default: the phi axis is still dense here and must be
    SIZED from the data, and a silently-undersized grid is the defect this module family
    exists to remove."""
    data = make_synth(scale=2.0)
    with pytest.raises(Exception):
        AM.fused_log_likelihood_distphipsimarg_peaklocal(
            data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
            jnp.linspace(0.4, 2.0, 8), jnp.zeros(8), interp=INTERP)


def test_the_driver_CLI_accepts_it_and_rejects_a_typo():
    """Deliberately a SUBPROCESS: the flag's job is to travel from a command line into
    the likelihood, and optparse builds its choices from ANGLE_MARG_CHOICES at import
    time -- a test that imports the module cannot see the CLI wiring.  A misspelling
    must be loud, because a scheme name absorbed as 'not recognised' would silently run
    a different likelihood."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    driver = os.path.join(root, "bin", "integrate_likelihood_extrinsic_jax")
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = "cpu"

    def run(value):
        p = subprocess.run([sys.executable, driver, "--angle-marg-scheme", value],
                           env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=600)
        return p.stdout.decode("utf-8", "replace")

    good = run("peak-local")
    assert "invalid choice" not in good, good[-1500:]

    bad = run("peaklocal")            # the plausible misspelling
    assert "invalid choice" in bad, bad[-1500:]
    assert "peak-local" in bad, bad[-1500:]


# --------------------------------------------------------- review findings

def test_peak_local_runs_the_runtime_amplitude_failsafe():
    """P1 from review.  The u axis is localized and needs no sizing, but THE PHI AXIS IS
    STILL DENSE and is sized from amp_sizing, which estimate_angle_amplitude is explicit
    about being an estimator and NOT a proven bound.  A hotter sampled sky location can
    therefore under-resolve phi exactly as it can for exact/laplace, and skipping the
    check would publish that silently."""
    data = make_synth(scale=2.0, kappa_boost=50.0)
    x = jnp.linspace(0.4, 2.0, 8)
    lw = jnp.zeros(8)
    AM.reset_amp_failsafe()
    # size for a much quieter target than the data actually is: the check must notice
    AM.fused_log_likelihood_distphipsimarg_peaklocal(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL), x, lw,
        interp=INTERP, amp_sizing=1.0)
    st = AM.amp_failsafe_state(barrier=True)
    assert st.get("tripped"), st
    assert st.get("scheme") == "peak-local", st
    AM.reset_amp_failsafe()


def test_peak_local_is_capped_by_the_batch_memory_rule():
    """P1 from review.  peak-local still nests sample/time vmaps over the distance grid,
    phi chunks, four cells and 48 u nodes, so the batch multiplies the same way the
    dense schemes do.  Leaving it out of the cap kept an uncapped 8000-sample batch and
    reopened a documented 36.4 GiB failure."""
    from RIFT.likelihood.jax_ile import samplers as S

    class _Data(object):
        npts = 614

    class _Like(object):
        data = _Data()
        angle_marg_scheme = "peak-local"
        x_grid = np.zeros(256)

    class _Exact(_Like):
        angle_marg_scheme = "exact"

    class _NoScheme(object):
        pass

    capped = S.angle_marg_eval_chunk(_Like(), 8000)
    assert capped < 8000
    # NOT "same cap as exact" -- that was the earlier assertion and review rightly
    # objected that it pins the wrong invariant.  peak-local carries the WHOLE distance
    # grid inside every phi chunk, so its live slab is ~770x the dense model's
    # 8192 bytes/sample/time-point; a cap equal to exact's would look protective and
    # would not be.  The scheme-specific model must therefore be STRICTLY tighter.
    assert capped < S.angle_marg_eval_chunk(_Exact(), 8000), capped
    # and it must scale with the distance grid, which is what makes it a model rather
    # than a constant
    class _Wide(_Like):
        x_grid = np.zeros(1024)
    assert S.angle_marg_eval_chunk(_Wide(), 8000) <= capped
    # the "grid" sentinel means "runs no dense angle scheme" and must stay uncapped
    assert S.angle_marg_eval_chunk(_NoScheme(), 8000) == 8000


def test_peak_local_artifacts_carry_the_standing_best_effort_label():
    """P1 from review.  A scheme missing from the label's list publishes output with NO
    standing statement at all -- and silence is precisely what a reader six months later
    would misread as verification.  peak-local's phi grid is amp-sized, so its artifacts
    are entitled to no more confidence than exact's."""
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "bin", "integrate_likelihood_extrinsic_jax")
    spec = importlib.util.spec_from_loader("_ile_jax_driver",
                                           importlib.machinery.SourceFileLoader(
                                               "_ile_jax_driver", path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AM.reset_amp_failsafe()
    note = mod.angle_grid_suspect_note("peak-local")
    assert note.startswith("ANGLE-GRID-CHECK=BEST-EFFORT"), note
    assert mod.angle_grid_suspect_note("grid") == ""
