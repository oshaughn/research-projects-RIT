"""util_ParameterPuffball.py with the --internal-puff-transverse coordinates.

Puffing s1z_bar, s2z_bar, chi1_perp_u, chi2_perp_u, phi1, phi2 must reflect excursions back
into the spin box (not discard them), wrap only the azimuths, and never write non-finite spins.
"""
import os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

lal = pytest.importorskip('lal')
CODE = Path(__file__).resolve().parents[1]
from RIFT import lalsimutils  # noqa: E402

TRANSVERSE = ['--parameter', 'mc', '--parameter', 'delta_mc', '--parameter', 's1z_bar', '--parameter', 's2z_bar',
              '--parameter', 'phi1', '--parameter', 'phi2', '--parameter', 'chi1_perp_u', '--parameter', 'chi2_perp_u']


def _grid(path, n=400, seed=3):
    """Points crowded against |chi|=1 and the poles, so a large puff crosses every box edge."""
    rng = np.random.default_rng(seed)
    P_list = []
    for _ in range(n):
        P = lalsimutils.ChooseWaveformParams()
        P.m1, P.m2 = rng.uniform(34, 38) * lal.MSUN_SI, rng.uniform(28, 32) * lal.MSUN_SI
        for k in (1, 2):
            a, cz, ph = rng.uniform(0.85, 0.99), rng.uniform(-0.98, 0.98), rng.uniform(0, 2 * np.pi)
            sp = a * np.sqrt(1 - cz ** 2)
            setattr(P, 's%dx' % k, sp * np.cos(ph)); setattr(P, 's%dy' % k, sp * np.sin(ph)); setattr(P, 's%dz' % k, a * cz)
        P.fmin = P.fref = 20.
        P_list.append(P)
    lalsimutils.ChooseWaveformParams_array_to_xml(P_list, fname=str(path))
    return P_list


def _puff(tmp_path, params=TRANSVERSE):
    _grid(tmp_path / 'in')
    env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), OMP_NUM_THREADS='1')
    cmd = [sys.executable, str(CODE / 'bin/util_ParameterPuffball.py'), '--inj-file', str(tmp_path / 'in.xml.gz'),
           '--inj-file-out', str(tmp_path / 'out'), '--puff-factor', '3', '--fref', '20', '--fmin', '20',
           '--downselect-parameter', 'chi1', '--downselect-parameter-range', '[0,1]',
           '--downselect-parameter', 'chi2', '--downselect-parameter-range', '[0,1]'] + list(params)
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:]
    return proc.stdout, lalsimutils.xml_to_ChooseWaveformParams_array(str(tmp_path / 'out.xml.gz'))


def test_transverse_puff_reflects_and_keeps_points(tmp_path):
    log, out = _puff(tmp_path)
    assert 'Range downselect :  400 400' in log, [l for l in log.splitlines() if 'downselect' in l]
    assert len(out) == 400
    s = np.array([[P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z] for P in out])
    assert np.isfinite(s).all()
    assert (np.linalg.norm(s[:, :3], axis=1) <= 1 + 1e-12).all() and (np.linalg.norm(s[:, 3:], axis=1) <= 1 + 1e-12).all()


def test_only_azimuths_are_wrapped(tmp_path):
    # The periodic wrap once applied np.mod(2 pi) to the whole row, folding every coordinate assigned
    # after an azimuth. Put mc (~29) last so that bug would fold it to ~4.
    _, out = _puff(tmp_path, TRANSVERSE[4:] + TRANSVERSE[:4])
    mc = np.array([P.extract_param('mc') / lal.MSUN_SI for P in out])
    assert np.median(mc) > 20, np.median(mc)
