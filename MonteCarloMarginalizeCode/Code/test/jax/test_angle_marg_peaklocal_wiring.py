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
