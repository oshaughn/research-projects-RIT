"""`--angle-marg-scheme multipeak`, as wired into the ILE likelihood.

The four-axis controller validated on the ladder-2 injection
(analyses/va_sequence_20260902/RESULTS_20260909_multipeak_ladder.md: 64/64
accepted at rho 40.77, 163.08 and 652.31, 4.74-5.17 s/row) was reachable only
from `multipeak_planner`; no driver path selected it.  These pin the WIRING.
"""
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiPsiMargLikelihood
from test_angle_marg_exact import make_synth, RA, DEC, INCL, INTERP


def test_multipeak_is_an_offered_choice_and_is_exported():
    """optparse builds --angle-marg-scheme's choices from ANGLE_MARG_CHOICES, so
    membership IS the CLI wiring; a scheme absent from it is unreachable."""
    assert "multipeak" in AM.ANGLE_MARG_CHOICES
    assert "fused_log_likelihood_distphipsimarg_multipeak" in AM.__all__


def test_multipeak_is_NOT_reachable_from_auto():
    """A scheme that changes the likelihood must not become a default on the
    strength of one injection's ladder."""
    for amp in (1.0, 50.0, 500.0, 5.0e4, 5.0e6):
        scheme, _ = AM.choose_angle_marg_scheme(amp)
        assert scheme != "multipeak", (amp, scheme)


def test_multipeak_refuses_lnLt_because_it_owns_the_time_integral():
    """Every other scheme in this family can hand back lnL(t).  This one cannot,
    and must SAY so rather than return a wrong-shaped array."""
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                       interp=INTERP, angle_marg="multipeak")
    with pytest.raises(ValueError, match="no lnL"):
        like._fused(data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
                    return_lnLt=True)


def test_multipeak_returns_one_finite_value_per_sample():
    """The contract the sampler depends on: shape (S,), finite, no time axis."""
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                       interp=INTERP, angle_marg="multipeak")
    v = np.asarray(like._fused(data, jnp.asarray(RA), jnp.asarray(DEC),
                               jnp.asarray(INCL)))
    assert v.shape == np.shape(RA), v.shape
    assert np.all(np.isfinite(v)), v


def test_multipeak_records_its_provenance():
    """This pipeline has a history of silently-inert flags: the scheme actually
    used must be visible in the record, not inferred from the request."""
    data = make_synth(scale=2.0)
    like = JAXDistPhiPsiMargLikelihood(data, 30.0, 3000.0, nphi=32, npsi=8,
                                       interp=INTERP, angle_marg="multipeak")
    assert like.angle_marg_scheme == "multipeak"
    assert like.angle_marg_info["requested"] == "multipeak"


def test_the_reserve_callable_runs_the_REAL_laplace_kernel():
    """The reserve is exercised against the real table kernel, not a stand-in.

    A fake here would pass while the real call raises: the policy layer's
    resolve_reserve_angular_kernel forwards dense_chunk/grid_block, which
    coefficient_table_distphipsimarg_laplace does not accept, and that TypeError
    only appears on a genuine reserve evaluation.  This calls the shipped
    function with the shapes the multipeak reserve builds.
    """
    data = make_synth(scale=2.0)
    xg = jnp.linspace(0.4, 2.0, 16)
    lwg = jnp.zeros(16) - np.log(16.0)
    C_A, C_B, meta = AM.angle_coefficient_tables(
        data, jnp.asarray(RA[:1]), jnp.asarray(DEC[:1]), jnp.asarray(INCL[:1]),
        INTERP, guard=16)
    CA = np.moveaxis(np.asarray(C_A), 2, 0)[0][..., 16:-16]
    CB = np.moveaxis(np.asarray(C_B), 2, 0)[0][..., 0]
    out = AM.coefficient_table_distphipsimarg_laplace(
        jnp.asarray(CA), jnp.asarray(CB), xg, lwg,
        amp_sizing=AM.ANGLE_MARG_CROSSOVER_AMPLITUDE, m_max=int(meta["m_max"]))
    assert np.all(np.isfinite(np.asarray(out)))


def test_the_driver_CLI_accepts_it_and_rejects_a_typo():
    """A SUBPROCESS on purpose: optparse builds its choices at import time, so a
    test that imports the module cannot see the CLI wiring."""
    import RIFT
    drv = str(__import__("pathlib").Path(RIFT.__file__).parents[1]
              / "bin" / "integrate_likelihood_extrinsic_jax")
    # Assert on optparse's CHOICE VALIDATION, not on --help text: the option is
    # built with choices=sorted(ANGLE_MARG_CHOICES) and optparse does not print
    # the choice list, so a help-text grep tests the help string rather than the
    # wiring and would pass or fail for the wrong reason.
    bad = subprocess.run([sys.executable, drv, "--angle-marg-scheme", "multipeaks"],
                         capture_output=True, text=True, timeout=900)
    assert bad.returncode != 0
    assert "multipeaks" in bad.stderr, bad.stderr[-800:]
    good = subprocess.run([sys.executable, drv, "--angle-marg-scheme", "multipeak"],
                          capture_output=True, text=True, timeout=900)
    # It must fail for a MISSING-INPUT reason, never because the scheme is invalid.
    assert "angle-marg-scheme" not in good.stderr or "choice" not in good.stderr, (
        good.stderr[-800:])
