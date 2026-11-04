"""convert_waveform_coordinates' vectorized mass/aligned-spin branch against the per-row loop it replaces.

The reference is the fallthrough loop itself (ChooseWaveformParams.assign_param / extract_param per
row). Inputs: a data file's basis (m1, m2, Cartesian spins) and the sampler's (mc, delta_mc, spherical
spins). enforce_kerr rows must come back as -inf rows, and source_redshift must still take the loop.
A wrong-sign chiMinus is the deliberately broken case.

The reference builds a fresh ChooseWaveformParams per row. The library loop reuses one, so an invalid
row (NaN, delta_mc > 1) corrupts later rows there; the vectorized branch does not inherit that.
"""
import numpy as np
import pytest

from RIFT import lalsimutils

FILE = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
MC = ['delta_mc', 'mc', 'chi1', 'chi2', 'cos_theta1', 'cos_theta2', 'phi1', 'phi2']


def _per_row(x, coord_names, low, enforce_kerr=False, source_redshift=0):
    out = np.zeros((len(x), len(coord_names)))
    for i in range(len(x)):
        P = lalsimutils.ChooseWaveformParams()
        for j, n in enumerate(low):
            P.assign_param(n, x[i, j])
        P.m1 *= (1 + source_redshift); P.m2 *= (1 + source_redshift)
        out[i] = [P.extract_param(p) for p in coord_names]
        if enforce_kerr and (P.extract_param('chi1') > 1 or P.extract_param('chi2') > 1):
            out[i] = -np.inf
    return out


def _file_rows(n=200, seed=1, chimax=0.99):
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(5, 80, n); m2 = m1 * rng.uniform(0.1, 1.0, n)
    v = rng.normal(size=(n, 2, 3)); v *= rng.uniform(0, chimax, (n, 2, 1)) / np.linalg.norm(v, axis=2, keepdims=True)
    return np.column_stack([m1, m2, v[:, 0], v[:, 1]])


def _mc_rows(n=200, seed=2):
    rng = np.random.default_rng(seed)
    lo = np.array([0.01, 5, 0, 0, -1, -1, 0, 0]); hi = np.array([0.9, 60, 0.99, 0.99, 1, 1, 2 * np.pi, 2 * np.pi])
    return rng.uniform(lo, hi, (n, 8))


def _close(a, b):
    both = np.isneginf(a) & np.isneginf(b)
    return np.all(both | (np.abs(a - b) <= 1e-12 + 1e-10 * np.abs(b)))


@pytest.mark.parametrize("names", [['delta_mc', 'mu1', 'mu2', 'chiMinus', 'mtot', 'q', 'eta', 'mc', 'xi'],
                                   ['mtot', 's1x', 's2y']])
def test_file_basis_matches_per_row(names):
    x = _file_rows()
    assert _close(lalsimutils.convert_waveform_coordinates(x, coord_names=list(names), low_level_coord_names=FILE),
                  _per_row(x, names, FILE))


def test_sampler_basis_mtot_matches_per_row():
    names = ['mu1', 'mu2', 'chiMinus', 's1x', 's1y', 's2x', 's2y', 'mtot', 'q']
    x = _mc_rows()
    assert _close(lalsimutils.convert_waveform_coordinates(x, coord_names=list(names), low_level_coord_names=MC),
                  _per_row(x, names, MC))


def test_kerr_rows_and_redshift_follow_the_loop():
    x = _file_rows(chimax=1.3)
    names = ['delta_mc', 'mtot', 'chiMinus']
    got = lalsimutils.convert_waveform_coordinates(x, coord_names=list(names), low_level_coord_names=FILE, enforce_kerr=True)
    ref = _per_row(x, names, FILE, enforce_kerr=True)
    assert np.isneginf(ref).any() and _close(got, ref)
    got = lalsimutils.convert_waveform_coordinates(x[:20], coord_names=list(names), low_level_coord_names=FILE, source_redshift=0.3)
    assert _close(got, _per_row(x[:20], names, FILE, source_redshift=0.3))


def test_sign_error_would_be_caught():
    x = _file_rows()
    got = lalsimutils.convert_waveform_coordinates(x, coord_names=['chiMinus'], low_level_coord_names=FILE)
    assert not _close(-got, _per_row(x, ['chiMinus'], FILE))


def test_aligned_spin_kerr_rule_and_bad_rows_do_not_leak():
    rng = np.random.default_rng(4)
    low = ['mc', 'delta_mc', 's1z', 's2z']
    x = np.column_stack([rng.uniform(5, 60, 300), rng.uniform(0, 0.9, 300), rng.uniform(-1.2, 1.2, 300), rng.uniform(-1.2, 1.2, 300)])
    names = ['q', 'mtot', 'xi', 'mu1', 'mu2']
    got = lalsimutils.convert_waveform_coordinates(x, coord_names=list(names), low_level_coord_names=low, enforce_kerr=True)
    ref = _per_row(x, names, low, enforce_kerr=True)
    assert np.isneginf(ref).any() and _close(got, ref)
    x[10, 1] = 1.5                                    # an invalid row
    got = lalsimutils.convert_waveform_coordinates(x, coord_names=list(names), low_level_coord_names=low)
    assert np.all(np.isfinite(got[11:]))
