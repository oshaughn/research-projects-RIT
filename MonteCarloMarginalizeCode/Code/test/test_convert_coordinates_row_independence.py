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


@pytest.mark.parametrize('mode', ['geometric4', 'geometric4-phase-excess'])
@pytest.mark.parametrize('z', [0., .3])
@pytest.mark.parametrize('frequency', [17., 43.])
@pytest.mark.parametrize('entry', ['converter', 'cip'])
def test_eos_rf_training_matches_prediction(mode, z, frequency, entry):
    """EOS prediction must use the same detector-frame chart and fref as training."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from RIFT.misc import rf_transverse_spin

    low = ['m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z']
    names = ['m1', 'm2', 'lambda1', 'lambda2'] + list(rf_transverse_spin.geometric_names(mode))
    x = np.array([[1.4, 1.2, .1, -.2, .3, -.2, .1, -.1],
                  [1.7, 1.1, -.3, .1, -.2, .2, -.1, .4]])
    eos = _StubEOS()
    if entry == 'converter':
        got = lsu.convert_waveform_coordinates_with_eos(
            x, coord_names=names, low_level_coord_names=low, eos_class=eos,
            source_redshift=z, fref=frequency)
    else:
        # Execute CIP's actual EOS branch so a converter-only fix cannot pass.
        source = Path(__file__).resolve().parents[1] / 'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py'
        blocks = [n for n in ast.parse(source.read_text()).body
                  if isinstance(n, ast.If) and ast.unparse(n.test) == 'not opts.using_eos'
                  and any(isinstance(b, ast.FunctionDef) and b.name == 'convert_coords' for b in n.orelse)]
        assert len(blocks) == 1
        ns = dict(np=np, my_fit=lambda rows: np.zeros(len(rows)),
                  opts=SimpleNamespace(using_eos=True, fref=frequency, no_matter1=False,
                                      no_matter2=False, downselect_enforce_kerr=False),
                  lalsimutils=lsu, coord_names=names, low_level_coord_names=low,
                  my_eos=eos, source_redshift=z)
        exec(compile(ast.Module(body=blocks, type_ignores=[]), str(source), 'exec'), ns)
        got = ns['convert_coords'](x)

    expected = []
    for row in x:
        p = lsu.ChooseWaveformParams(fref=frequency)
        p.m1, p.m2 = row[:2] * lsu.lal.MSUN_SI * (1 + z)
        p.s1x, p.s1y, p.s1z, p.s2x, p.s2y, p.s2z = row[2:]
        # The EOS is evaluated on source masses, before detector-frame redshift.
        p.lambda1, p.lambda2 = [eos.lambda_from_m(m * lsu.lal.MSUN_SI) for m in row[:2]]
        expected.append([p.m1 / lsu.lal.MSUN_SI, p.m2 / lsu.lal.MSUN_SI,
                         p.lambda1, p.lambda2] + [p.extract_param(n) for n in names[4:]])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
