"""RIFT.misc.waveform_coordinates_xpy against lalsimutils.convert_waveform_coordinates (numpy).

Both input bases a distance-slice reconstruction uses: the data file's (m1, m2, Cartesian spins), which
convert_waveform_coordinates handles in its per-row fallthrough, and the sampler's (mc, delta_mc,
spherical spins), including rows with |cos_theta| > 1 whose NaN outputs must match. A sign error in
chiMinus is the deliberately broken case: check_against must reject it.
"""
import numpy as np
import pytest

from RIFT import lalsimutils
import RIFT.misc.waveform_coordinates_xpy as W

FILE = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'dist']
MC = ['delta_mc', 'dist', 'mc', 'chi1', 'chi2', 'cos_theta1', 'cos_theta2', 'phi1', 'phi2']
FIT = ['delta_mc', 'dist', 'mu1', 'mu2', 'chiMinus', 's1x', 's1y', 's2x', 's2y', 'mtot', 'xi', 'mc', 'eta', 'q']


def _host(x, coord_names, low_level_coord_names):
    # dist is not a waveform coordinate; pass it through as the distance plugins do
    tgt = [n for n in coord_names if n != 'dist']
    out = lalsimutils.convert_waveform_coordinates(x, coord_names=tgt, low_level_coord_names=low_level_coord_names)
    cols = {n: out[:, i] for i, n in enumerate(tgt)}
    cols['dist'] = x[:, low_level_coord_names.index('dist')]
    return np.column_stack([cols[n] for n in coord_names])


def _file_rows(n=300, seed=1):
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(5, 80, n); m2 = m1 * rng.uniform(0.1, 1.0, n)
    v = rng.normal(size=(n, 2, 3)); v *= (rng.uniform(0, 0.99, (n, 2, 1)) / np.linalg.norm(v, axis=2, keepdims=True))
    return np.column_stack([m1, m2, v[:, 0], v[:, 1], rng.uniform(100, 5000, n)])


def _mc_rows(n=2000, seed=2):
    rng = np.random.default_rng(seed)
    lo = np.array([0.0, 100, 5, 0, 0, -1, -1, 0, 0]); hi = np.array([0.9, 5000, 60, 0.99, 0.99, 1, 1, 2 * np.pi, 2 * np.pi])
    x = rng.uniform(lo, hi, (n, 9))
    x[:20, 5] = 1.01                      # out of range: sqrt(1 - cos^2) is NaN in both
    return x


@pytest.mark.parametrize("basis,rows", [(FILE, _file_rows), (MC, _mc_rows)])
def test_matches_convert_waveform_coordinates(basis, rows):
    ok, worst, detail = W.check_against(_host, FIT, basis, rows())
    assert ok, (worst, detail)


def test_inv_dist_matches_plugin_definition():
    x = _mc_rows()
    got = W.convert_waveform_coordinates_xpy(x, ['inv_dist'], MC)[:, 0]
    assert np.array_equal(got, 1.0 / np.clip(x[:, 1], 1e-6, None))


def test_unsupported_name_raises():
    with pytest.raises(NotImplementedError):
        W.convert_waveform_coordinates_xpy(_mc_rows(10), ['lambda1'], MC)


def test_sign_error_is_rejected(monkeypatch):
    real = W.convert_waveform_coordinates_xpy

    def broken(x, coord_names, low_level_coord_names, xpy=None):
        out = real(x, coord_names, low_level_coord_names, xpy=xpy)
        out[:, list(coord_names).index('chiMinus')] *= -1
        return out
    monkeypatch.setattr(W, "convert_waveform_coordinates_xpy", broken)
    ok, worst, detail = W.check_against(_host, FIT, MC, _mc_rows())
    assert not ok and 'chiMinus' in detail


def test_cupy_matches_numpy():
    try:
        import cupy as cp
        cp.zeros(1)
    except Exception as err:
        pytest.skip("needs cupy and a CUDA device: %s" % type(err).__name__)
    for basis, rows in ((FILE, _file_rows()), (MC, _mc_rows())):
        a = W.convert_waveform_coordinates_xpy(rows, FIT, basis)
        b = cp.asnumpy(W.convert_waveform_coordinates_xpy(cp.asarray(rows), FIT, basis))
        nan = np.isnan(a)
        assert np.array_equal(nan, np.isnan(b))
        assert np.all(np.abs(a[~nan] - b[~nan]) <= 1e-12 + 1e-11 * np.abs(a[~nan]))
