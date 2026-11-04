"""Atomic mu inverse and RANDOM transverse discovery, including aligned seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from RIFT.misc import mu_puffball as puff
from RIFT.misc import tools

CODE = Path(__file__).resolve().parents[1]


def physical_rows(n=1000):
    rng = np.random.default_rng(448)
    mc = np.exp(rng.uniform(np.log(1), np.log(80), n))
    q = rng.uniform(.001, 1, n)
    m1, m2 = tools.McqTom1m2(mc, q)
    spins = rng.normal(size=(n, 2, 3))
    spins /= np.linalg.norm(spins, axis=2)[:, :, None]
    spins *= rng.uniform(0, .999, (n, 2, 1))
    return np.column_stack((m1, m2, spins.reshape(n, 6)))


def test_atomic_inverse_roundtrips_negative_mu1_and_equal_mass():
    rows = physical_rows(10000)
    rows[:2] = [[40, 40, 0, 0, .999999, 0, 0, -.999999],
                [40, .04, 0, 0, .999999, 0, 0, -.999999]]
    z = puff.aligned_coordinates(rows)
    assert np.any(z[:, 0] < 0)
    with np.errstate(all='raise'):
        mc, z1 = puff.inverse_mu(*z.T)
    np.testing.assert_allclose(mc, tools.m1m2ToMc(*rows[:, :2].T), rtol=1e-11)
    np.testing.assert_allclose(z1, rows[:, 4], atol=1e-11)
    # Independent existing inverse is valid in its positive-mu1 domain.
    z = z[z[:, 0] > 0][:30]
    expected = tools.mu1mu2qchi2ToMcqchi1chi2(*z.T)
    actual = puff.inverse_mu(*z.T)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6)
    np.testing.assert_allclose(actual[1], expected[2], atol=2e-5)


@pytest.mark.parametrize('row', [
    [1, 0, 0, 0], [1, 0, -1, 0], [1, 0, 1.01, 0],
    [np.inf, 0, 1, 0], [1, np.nan, 1, 0], [1, 0, np.inf, 0],
    [1, 0, 1, np.nan], [-1, 0, 1, 0], [0, 0, 1, 0]])
def test_invalid_inverse_returns_nan_without_warnings(row):
    with np.errstate(all='raise'):
        mc, z1 = puff.inverse_mu(*row)
    assert np.isnan(mc) and np.isnan(z1)


def test_broadcast_inverse_and_near_positive_boundary():
    z = puff.aligned_coordinates(physical_rows(4))
    mc, z1 = puff.inverse_mu(z[:, 0], z[:, 1], z[:, 2], z[:, 3])
    assert mc.shape == z1.shape == (4,)
    mc, _ = puff.inverse_mu(1e-12, 0., 1., 0.)
    assert np.isfinite(mc) and mc > 0


def test_random_transverse_works_for_identical_aligned_seeds_and_zero_puff():
    # Zero covariance in ALL four axes must not inject arbitrary mass/spin noise.
    rows = np.tile([40, 20, 0, 0, .6, 0, 0, -.8], (20, 1))
    out, parents, stats = puff.propose(rows, count=10000, scale=0, rng=17)
    assert stats['attempted'] == stats['inverse_accepted'] == len(out) == 10000
    np.testing.assert_allclose(out[:, :2], np.tile([40, 20], (len(out), 1)), atol=1e-10)
    np.testing.assert_allclose(out[:, [4, 7]], np.tile([.6, -.8], (len(out), 1)), atol=1e-12)
    radius1, radius2 = np.hypot(out[:, 2], out[:, 3]), np.hypot(out[:, 5], out[:, 6])
    assert abs(np.mean(radius1)-.4) < .01
    assert abs(np.mean(radius2)-.3) < .01
    assert np.mean(radius1 > .5) > .35
    assert np.max(np.linalg.norm(out[:, 2:5], axis=1)) <= 1+1e-12
    assert np.max(np.linalg.norm(out[:, 5:8], axis=1)) <= 1+1e-12
    assert parents.min() >= 0 and parents.max() < len(rows)
    assert abs(np.mean(out[:, 2])) < .02  # no preferred azimuth
    same, same_parents, _ = puff.propose(rows, count=10000, scale=0, rng=17)
    np.testing.assert_array_equal(out, same)
    np.testing.assert_array_equal(parents, same_parents)


def test_transverse_redraw_is_independent_of_old_transverse_covariance():
    rows = physical_rows(200)
    aligned = rows.copy()
    aligned[:, [2, 3, 5, 6]] = 0
    a = puff.propose(rows, count=2000, rng=9)[0]
    b = puff.propose(aligned, count=2000, rng=9)[0]
    np.testing.assert_array_equal(a, b)
    single = puff.propose(aligned[:1], count=10, transverse_max=.2, rng=9)[0]
    assert np.all(np.hypot(single[:, 2], single[:, 3]) <= .2)


def test_default_budget_does_not_resample_away_seed_branches():
    rows = physical_rows(20)
    out, parents, stats = puff.propose(rows, scale=0, rng=99)
    np.testing.assert_array_equal(parents, np.arange(len(rows)))
    np.testing.assert_allclose(out[:, :2], rows[:, :2], rtol=1e-11)
    assert stats['attempted'] == len(rows)


def test_reflection_handles_multiple_crossings_and_preserves_mu_inverse():
    rows = np.tile([40, 20, 0, 0, .2, 0, 0, -.3], (5, 1))
    z = puff.aligned_coordinates(rows)
    z[:, 2] = [-2.3, -.3, .3, 1.7, 4.3]
    z[:, 3] = [-8.3, -2.3, -.3, 1.7, 6.3]
    reflected = puff.reflect_coordinates(z, {'q': (0, 1), 's2z': (-1, 1)})
    np.testing.assert_allclose(reflected[:, 2], [.3, .3, .3, .3, .3], atol=1e-14)
    np.testing.assert_allclose(reflected[:, 3], [-.3, .3, -.3, .3, -.3], atol=1e-14)
    np.testing.assert_array_equal(reflected[:, :2], z[:, :2])
    mc, z1 = puff.inverse_mu(*reflected.T)
    mu1, mu2, _ = tools.Mcqchi1chi2Tomu1mu2mu3(mc, reflected[:, 2], z1, reflected[:, 3])
    np.testing.assert_allclose(np.column_stack((mu1, mu2)), z[:, :2], atol=1e-11)
    # The input chart remains untouched, including out-of-bounds values.
    assert z[0, 2] == -2.3


def test_reflection_narrows_to_native_eta_and_spin_cuts_and_keeps_nan_invalid():
    bounds = puff.reflection_ranges(['q', 's2z'], {'eta': [.1, .24], 'q': [.2, .9], 's2z': [-.5, .5]})
    np.testing.assert_allclose(bounds['q'], [max(.2, tools.etaToq(.1)), tools.etaToq(.24)])
    assert bounds['s2z'] == (-.5, .5)
    with np.errstate(all='raise'):
        out = puff.reflect_coordinates([[1, 2, np.inf, np.nan]], bounds)
        mc, z1 = puff.inverse_mu(*out.T)
    assert np.isnan(mc[0]) and np.isnan(z1[0])
    for bounds in ({'s1z': [-1, 1]}, {'q': [1, 0]}, {'q': [-1, 1]}, {'s2z': [-2, 2]}):
        with pytest.raises(ValueError):
            puff.reflect_coordinates([[1, 2, .5, 0]], bounds)


def test_reflection_rescues_proposals_without_exceeding_spin_bounds():
    rows = physical_rows(500)
    plain, _, _ = puff.propose(rows, count=4000, scale=2, rng=448)
    reflected, _, stats = puff.propose(rows, count=4000, scale=2, rng=448,
                                     reflection_bounds={'q': (.1, .9), 's2z': (-.8, .8)})
    assert len(reflected) > len(plain)
    q = reflected[:, 1]/reflected[:, 0]
    assert np.all((q >= .1) & (q <= .9))
    assert np.all(np.abs(reflected[:, 7]) <= .8)
    assert np.all(np.linalg.norm(reflected[:, 2:5], axis=1) <= 1+1e-12)
    assert np.all(np.linalg.norm(reflected[:, 5:8], axis=1) <= 1+1e-12)
    assert set(stats['reflection_bounds']) == {'q', 's2z'}


@pytest.mark.parametrize('kwargs', [{'scale': -1}, {'scale': np.nan},
                                  {'transverse_max': 1.01}, {'count': 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        puff.propose(physical_rows(5), **kwargs)


def test_waveform_metadata_and_native_downselection():
    import lal
    from RIFT import lalsimutils
    p = lalsimutils.ChooseWaveformParams()
    p.m1, p.m2 = 40*lal.MSUN_SI, 20*lal.MSUN_SI
    p.s1z, p.s2z = .6, -.8
    p.lambda1, p.lambda2, p.eccentricity, p.meanPerAno = 123., 456., .1, 1.2
    p.fref, p.fmin = 20., 15.
    out, stats = puff.waveform_proposals([p], count=100, scale=0, rng=10,
                                        downselect={'chi1_perp': [.2, .7]}, fref=30.)
    assert 0 < len(out) < 100
    assert stats['exported_points'] == len(out)
    for child in out:
        assert child is not p
        assert child.lambda1 == 123 and child.lambda2 == 456
        assert child.eccentricity == .1 and child.meanPerAno == 1.2
        assert child.fref == 30 and child.fmin == 15
        assert .2 <= child.extract_param('chi1_perp') <= .7
    assert p.fref == 20 and p.s1x == p.s1y == 0


def test_cli_xml_and_ascii_preserve_metadata_and_attempt_budget(tmp_path):
    import lal
    from RIFT import lalsimutils
    from RIFT.misc import hyperpipeline_io as hpio
    p = lalsimutils.ChooseWaveformParams()
    p.m1, p.m2, p.s1z, p.s2z = 40*lal.MSUN_SI, 20*lal.MSUN_SI, .6, -.8
    p.fref = p.fmin = 20.
    p.lambda1, p.lambda2 = 123., 456.
    lalsimutils.ChooseWaveformParams_array_to_xml([p], fname=str(tmp_path/'in'), fref=20.)
    columns = hpio.build_column_list(use_tides=True)
    hpio.write_grid_from_P_list(str(tmp_path/'in.dat'), [p], columns,
                              lal_module=lal, lalsimutils_module=lalsimutils)
    env = dict(os.environ, PYTHONPATH=str(CODE), GW_SURROGATE='')
    for kind in ('xml.gz', 'dat'):
        output = str(tmp_path/('out_' + kind))
        command = [sys.executable, str(CODE/'bin/util_ParameterPuffball.py'),
                   '--inj-file', str(tmp_path/('in.'+kind)), '--inj-file-out', output,
                   '--puff-coordinate-system', 'mu-random-transverse',
                   '--puff-factor', '0', '--mu-puff-n-proposals', '200',
                   '--random-seed', '448', '--fail-if-empty', '--mc-range', '[20,30]']
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads(Path(output+'_mu_puff.json').read_text())
        assert summary['attempted'] == summary['exported_points'] == 200
        if kind == 'xml.gz':
            out = lalsimutils.xml_to_ChooseWaveformParams_array(output+'.xml.gz')
        else:
            assert hpio.read_header(output+'.dat') == columns
            out, _ = hpio.read_grid_to_P_list(output+'.dat', P_factory=lalsimutils.ChooseWaveformParams,
                                            lal_module=lal, valid_params=lalsimutils.valid_params)
        assert len(out) == 200
        assert all(p.lambda1 == 123 and p.lambda2 == 456 for p in out)
        assert max(p.extract_param('chi1_perp') for p in out) > .7
        bad = subprocess.run(command + ['--parameter', 'mc'], env=env,
                             capture_output=True, text=True, timeout=60)
        assert bad.returncode != 0 and 'does not support --parameter' in bad.stderr
