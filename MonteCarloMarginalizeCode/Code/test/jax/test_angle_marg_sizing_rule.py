"""The BITING regression for the m_max-aware dense phi sizing rule.

Split out of test_angle_marg_exact.py so it can stay in the per-PR gate while
that (expensive, validation-oriented) file does not.  Pure numpy, milliseconds,
no JAX and no likelihood -- it costs the gate nothing.

It exists because "correctness does not depend on amplitude" is FALSE for the
dense quadrature: resolving exp(lnL) in phi needs a grid set by the highest
harmonic 2*m_max as well as by sqrt(A), and the old m_max-blind rule passes
every low-scale brute-force test in the excluded file.  Without this, reverting
_dense_grid_sizes to the broken rule is green.
"""
import numpy as np


def test_dense_phi_sizing_must_scale_with_m_max():
    """BITING regression for the m_max-aware dense phi sizing.

    Pure numpy, no likelihood, no JAX, ~ms -- so it stays in the per-PR gate,
    unlike the amplitude ladder that was moved out of it.

    Why it must exist: "correctness does not depend on amplitude" is FALSE for
    the dense quadrature.  Resolving exp(lnL) in phi needs a grid set by the
    highest harmonic 2*m_max as WELL as by sqrt(A), and the old m_max-blind
    rule passes every low-scale brute-force test in this file.  Without this
    test, reverting _dense_grid_sizes to the broken rule is green.

    Construction: a pure order-(2*m_max) harmonic of amplitude b, whose
    circular mean has the closed form I0(b) -- so the reference is exact and
    needs no dense grid.  MEASURED at amp=450, b=150, order=16: the blind rule
    (n=352) errs by 4.98e-01 nats at its worst phase; the m_max-aware rule
    (n=1360) errs by 1.17e-10.  The phase sweep matters -- the error is
    phase-dependent and vanishes at favourable alignments.
    """
    from scipy.special import ive
    import numpy as _np
    from RIFT.likelihood.jax_ile.anglemarg import _dense_grid_sizes

    amp, m_max, b = 450.0, 8, 150.0
    order = 2 * m_max
    exact = float(_np.log(ive(0, b)) + b)

    def worst(n):
        e = 0.0
        for ph in _np.linspace(0.0, 2 * _np.pi / order, 9):
            phi = _np.linspace(0.0, 2 * _np.pi, n, endpoint=False)
            v = b * _np.cos(order * phi + ph)
            m = v.max()
            e = max(e, abs(m + _np.log(_np.mean(_np.exp(v - m))) - exact))
        return e

    n_old = _dense_grid_sizes(amp)[0]                  # m_max-blind (the bug)
    n_new = _dense_grid_sizes(amp, m_max=m_max)[0]
    err_old, err_new = worst(n_old), worst(n_new)

    assert n_new > n_old, (
        "m_max-aware sizing must request MORE phi points (n_new=%d <= n_old=%d)"
        % (n_new, n_old))
    assert err_new < 1e-6, (
        "m_max-aware sizing inaccurate: err=%.3e at n=%d" % (err_new, n_new))
    assert err_old > 1e-2, (
        "this test no longer BITES: the m_max-blind rule errs only %.3e at "
        "n=%d, so a revert would pass.  Re-tune (b, m_max) until it does."
        % (err_old, n_old))

