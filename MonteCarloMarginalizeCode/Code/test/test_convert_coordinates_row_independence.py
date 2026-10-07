"""The per-row loops of convert_waveform_coordinates and convert_waveform_coordinates_with_eos must convert each row on its own.

assign_param reads the current ChooseWaveformParams state for some coordinates (mu1, mu2 and q
read m1, m2, s1z, s2z; chi1_perp_bar reads s1z), so a P reused across rows makes a row's output
depend on the row before it.  Each case below is state-sensitive in that way; the checks are
permutation invariance and agreement with a fresh P per row.
"""
import numpy as np
import pytest

from RIFT import lalsimutils as lsu

N = 40


def _per_row(x_in, coord_names, low_level_coord_names, z=0.0):
    out = np.zeros((len(x_in), len(coord_names)))
    for i, row in enumerate(x_in):
        P = lsu.ChooseWaveformParams()
        for p, val in zip(low_level_coord_names, row):
            P.assign_param(p, val)
        P.m1 *= 1 + z
        P.m2 *= 1 + z
        out[i] = [P.extract_param(p) for p in coord_names]
    return out


def _basic(rng, mc_range):
    return np.column_stack([rng.uniform(*mc_range, N), rng.uniform(0.05, 0.85, N),
                            rng.uniform(-0.8, 0.8, N), rng.uniform(-0.8, 0.8, N),
                            rng.uniform(0., 0.9, N), rng.uniform(0., 0.9, N),
                            rng.uniform(0., 2000., N), rng.uniform(0., 2000., N)])


BASIC = ['mc', 'delta_mc', 's1z_bar', 's2z_bar', 'chi1_perp_bar', 'chi2_perp_bar', 'lambda1', 'lambda2']

CASES = [
    # chi*_perp_bar before s*z_bar: the perpendicular magnitude is scaled by the current s*z
    (['mc', 'delta_mc', 'chi1_perp_bar', 'chi2_perp_bar', 's1z_bar', 's2z_bar'], ['chi1', 'chi2', 'chi_p']),
    # mu1, mu2 hold the current q and s2z; q holds the current mtot.  Some rows give nan.
    (['mu1', 'mu2', 'q', 's2z'], ['mc', 'delta_mc', 's1z']),
    # LambdaTilde without DeltaLambdaTilde holds the current lambda1, lambda2 difference
    (['mc', 'eta', 's1z', 's2z', 'LambdaTilde'], ['lambda1', 'lambda2']),
    # chieff_aligned rescales the current s1z, s2z (the puffball's spin coordinate)
    (['mc', 'delta_mc', 'chieff_aligned'], ['s1z', 's2z']),
]


def _inputs(low, rng, mc_range=(5., 40.)):
    return _per_row(_basic(rng, mc_range), low, BASIC)


@pytest.mark.parametrize("z", [0.0, 0.3])
@pytest.mark.parametrize("low,coord_names", CASES)
def test_fallback_rows_are_independent(low, coord_names, z, capsys):
    rng = np.random.default_rng(20261005)
    x_in = _inputs(low, rng)
    got = lsu.convert_waveform_coordinates(x_in, coord_names=coord_names,
                                           low_level_coord_names=low, source_redshift=z)
    assert "Fallthrough to non-vector-coords" in capsys.readouterr().out  # premise: the fallback ran
    perm = rng.permutation(N)
    got_perm = lsu.convert_waveform_coordinates(x_in[perm], coord_names=coord_names,
                                                low_level_coord_names=low, source_redshift=z)
    np.testing.assert_array_equal(got_perm, got[perm])
    # determinism, not correctness: the fresh-P answer is itself wrong for some cases (mu1, mu2
    # without a mass coordinate return kg masses)
    np.testing.assert_allclose(got, _per_row(x_in, coord_names, low, z), rtol=1e-12, atol=0)


class _StubEOS:
    mMaxMsun = 10.

    def lambda_from_m(self, m_kg):
        return 400. * (1.4 * lsu.lal.MSUN_SI / m_kg) ** 6


def test_eos_rows_are_independent():
    rng = np.random.default_rng(7)
    low, coord_names = CASES[0][0], ['chi1', 'chi2', 'LambdaTilde']
    x_in = _inputs(low, rng, mc_range=(1.0, 1.4))
    kw = dict(coord_names=coord_names, low_level_coord_names=low, eos_class=_StubEOS())
    got = lsu.convert_waveform_coordinates_with_eos(x_in, **kw)
    assert np.isfinite(got).all()
    perm = rng.permutation(N)
    np.testing.assert_array_equal(lsu.convert_waveform_coordinates_with_eos(x_in[perm], **kw), got[perm])
