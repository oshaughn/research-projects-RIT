"""The table-level psi-Laplace reserve: same seam, same normalization.

`coefficient_table_distphipsimarg_laplace` exists so the four-axis policy can
select a reserve METHOD.  Its reserve consumes coefficient tables at refined
time nodes, and until now the only table-level kernel was the exact one, so the
composite could only ever fall back to exact angles whatever the amplitude --
even where the selector's own calibration says laplace is the more accurate and
far cheaper choice.

Two things are pinned here, and they are different claims:

1. EXTRACTION FIDELITY.  The fused laplace kernel is now a thin wrapper over
   this function, so the two cannot drift.  The body moved verbatim; if it did
   not, this fails.
2. CROSS-SCHEME AGREEMENT.  The table laplace and table exact kernels compute
   the same quantity by different routes.  The tolerance is MEASURED on these
   fixtures and pinned, not copied from the selector's calibration table, which
   compares the FUSED paths on real data at stated SNRs and is a different
   comparison.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile import core as core_mod
from RIFT.likelihood.jax_ile.core import make_distance_grid

from test_angle_marg_exact import make_synth, RA, DEC, INCL, INTERP, _dist_grid


def _tables(data):
    return AM.angle_coefficient_tables(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL), INTERP)


def test_the_table_kernel_reproduces_the_fused_laplace_kernel():
    """Extraction fidelity: the fused kernel delegates, so they must agree to
    round-off.  A larger gap means the body changed in the move, which is the
    one thing a refactor may not do."""
    data = make_synth(scale=0.1, npts=9)
    x_grid, log_w = _dist_grid(data, n=16)
    C_A, C_B, meta = _tables(data)

    from_tables = AM.coefficient_table_distphipsimarg_laplace(
        C_A, C_B, x_grid, log_w, amp_sizing=30.0, m_max=meta["m_max"])
    wrapped = AM.fused_log_likelihood_distphipsimarg_laplace(
        data, jnp.asarray(RA), jnp.asarray(DEC), jnp.asarray(INCL),
        x_grid, log_w, interp=INTERP, amp_sizing=30.0, return_lnLt=True)

    np.testing.assert_allclose(np.asarray(from_tables), np.asarray(wrapped),
                               rtol=0.0, atol=2.0e-12)


def test_the_table_kernel_takes_the_collapsed_table_form():
    """The four-axis path holds an unbatched C_A and a time-independent C_B.
    The reserve seam must accept that form, or the policy cannot call it."""
    data = make_synth(scale=0.1, npts=9)
    x_grid, log_w = _dist_grid(data, n=16)
    C_A, C_B, meta = _tables(data)

    batched = AM.coefficient_table_distphipsimarg_laplace(
        C_A, C_B, x_grid, log_w, amp_sizing=30.0, m_max=meta["m_max"])
    collapsed = AM.coefficient_table_distphipsimarg_laplace(
        C_A[:, :, 0, :], C_B[:, :, 0, 0], x_grid, log_w,
        amp_sizing=30.0, m_max=meta["m_max"])

    np.testing.assert_allclose(np.asarray(collapsed), np.asarray(batched),
                               rtol=0.0, atol=2.0e-12)


@pytest.mark.parametrize("scale,kappa", [(1.0, 1.0), (3.0, 6.0), (6.0, 12.0)])
def test_the_two_table_reserves_agree_at_machine_level(scale, kappa):
    """Cross-scheme agreement on IDENTICAL tables, with the dense grid sized to
    the data.

    MEASURED, not copied: 1.2e-14, 1.4e-14 and 2.8e-14 on these three fixtures.
    The tolerance below is not the selector's calibrated laplace-vs-exact figure
    (-1.6e-05 at rho 40), which compares the FUSED paths on real data at a
    stated SNR and is a different comparison; at these synthetic amplitudes both
    kernels are essentially exact, so the Laplace error regime is NOT exercised
    here and this test must not be read as evidence about it.
    """
    data = make_synth(scale=scale, kappa_boost=kappa, npts=9)
    x_grid, log_w = _dist_grid(data, n=32)
    C_A, C_B, meta = _tables(data)

    kw = dict(amp_sizing=3000.0, m_max=meta["m_max"])
    lap = np.asarray(AM.coefficient_table_distphipsimarg_laplace(
        C_A, C_B, x_grid, log_w, **kw))
    exa = np.asarray(AM.coefficient_table_distphipsimarg_exact(
        C_A, C_B, x_grid, log_w, dense_chunk=8, grid_block=8, **kw))

    assert np.all(np.isfinite(lap)), lap
    np.testing.assert_allclose(lap, exa, rtol=0.0, atol=1.0e-12)


def test_an_undersized_dense_grid_degrades_the_exact_side_not_laplace():
    """Which kernel is fragile when the dense grid is too small for the data.

    This pins a correction to my own first version of this file, which held
    amp_sizing fixed while raising the amplitude and read the growing
    disagreement as the Laplace error growing.  It is the opposite: laplace
    removes the psi axis analytically and has no dense u grid to undersize, so
    the degradation is entirely on the exact side.  Measured on the loud
    fixture: 5.7e-07 at amp_sizing 30 against 2.8e-14 at 3000, same tables.

    That is also the practical argument for the reserve hierarchy -- above the
    crossover the exact reserve needs a grid that grows with amplitude, and the
    laplace reserve does not.
    """
    data = make_synth(scale=6.0, kappa_boost=12.0, npts=9)
    x_grid, log_w = _dist_grid(data, n=32)
    C_A, C_B, meta = _tables(data)

    def gap(amp):
        lap = np.asarray(AM.coefficient_table_distphipsimarg_laplace(
            C_A, C_B, x_grid, log_w, amp_sizing=amp, m_max=meta["m_max"]))
        exa = np.asarray(AM.coefficient_table_distphipsimarg_exact(
            C_A, C_B, x_grid, log_w, amp_sizing=amp, m_max=meta["m_max"],
            dense_chunk=8, grid_block=8))
        return float(np.max(np.abs(lap - exa)))

    starved, sized = gap(30.0), gap(3000.0)
    assert sized < 1.0e-12, sized
    assert starved > 100.0 * sized, (starved, sized)
