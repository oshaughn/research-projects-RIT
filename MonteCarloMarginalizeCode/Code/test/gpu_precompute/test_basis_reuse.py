"""Verify sidereal-index reuse against the independent per-index recipe."""
import numpy as np
import pytest

from RIFT.likelihood import gpu_precompute as gp
from conftest import to_host


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_reverse_fft_reused_preserving_order_and_duplicates(backend, monkeypatch, batch):
    rng = np.random.default_rng(9421)
    base = backend.asarray(rng.normal(size=(3, 32)) + 1j*rng.normal(size=(3, 32)))
    response = backend.asarray(rng.normal(size=(2, 32)) + 1j*rng.normal(size=(2, 32)))
    indices = [(1, 0, 2), (0, 1, -1), (1, 0, -2), (1, 0, 2), (0, 1, 1)]
    df, dt, epoch, sidereal = .125, .25, -.37, .012
    reverse = gp._lal_reverse
    frequency = gp.lal_frequency_axis(32, df, xp=backend)
    times = epoch + backend.arange(32)*dt
    expected = backend.stack([
        gp._lal_forward(reverse(base*response[b][None, :]*
                                gp._derivative_weight(frequency, p, backend)[None, :], backend)*
                        backend.exp(2j*np.pi*n*sidereal*times)[None, :], backend)
        for b, p, n in indices])
    calls = []

    def counted_reverse(values, xp):
        calls.append(values.shape)
        return reverse(values, xp)

    monkeypatch.setattr(gp, "_lal_reverse", counted_reverse)
    got = gp.build_compound_basis(base, response, indices, df, dt, epoch,
                                  f_sidereal=sidereal, backend=backend, fft_batch=batch)
    np.testing.assert_allclose(to_host(got), to_host(expected), rtol=3e-12, atol=3e-12)
    assert calls == [(3, 32), (3, 32)]  # Two unique (b,p), not five sidereal rows.
