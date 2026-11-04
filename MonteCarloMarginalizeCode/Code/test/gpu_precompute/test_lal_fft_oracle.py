import numpy as np
import pytest

from oracle import q_oracle


def test_numpy_oracle_exactly_matches_lal_reverse_fft_roll_cut():
    lal = pytest.importorskip("lal")
    rng = np.random.default_rng(6103)
    n, df, n_shift, n_window = 32, 0.125, -5, 17
    h = rng.normal(size=n) + 1j * rng.normal(size=n)
    d = rng.normal(size=n) + 1j * rng.normal(size=n)
    w = rng.uniform(size=n)
    integrand = 2 * np.conj(h) * d * w
    hf = lal.CreateCOMPLEX16FrequencySeries(
        "integrand", lal.LIGOTimeGPS(100.25), 0, df, lal.DimensionlessUnit, n)
    hf.data.data[:] = integrand
    ht = lal.CreateCOMPLEX16TimeSeries(
        "q", lal.LIGOTimeGPS(0), 0, 1.0 / (n * df), lal.DimensionlessUnit, n)
    lal.COMPLEX16FreqTimeFFT(ht, hf, lal.CreateReverseCOMPLEX16FFTPlan(n, 0))
    lal_q = np.roll(np.asarray(ht.data.data).copy(), -n_shift)[:n_window]
    got = q_oracle(h[None, None, :], d, w, df, n_shift, n_window)[0, 0]
    np.testing.assert_allclose(got, lal_q, rtol=3e-15, atol=3e-15)
