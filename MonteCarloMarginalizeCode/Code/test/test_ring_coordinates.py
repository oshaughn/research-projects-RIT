"""phi12 and the ring coordinates (chi1_perp, chi2_perp, SOverM2_perp, DeltaOverM2_perp, chi_p_vec).

The vectorized spherical-spin path of convert_waveform_coordinates must agree with
ChooseWaveformParams.extract_param, and must not fall through to the per-row loop.
"""

import numpy as np
import lal

from RIFT import lalsimutils

RING = ['chi1_perp', 'chi2_perp', 'phi12', 'SOverM2_perp', 'DeltaOverM2_perp', 'chi_p_vec']
LOW = ['mc', 'delta_mc', 'chi1', 'cos_theta1', 'phi1', 'chi2', 'cos_theta2', 'phi2']


def _draws(n=300, seed=4):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(5, 30, n), rng.uniform(0.01, 0.8, n), rng.uniform(0, 0.99, n),
                            rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 0.99, n),
                            rng.uniform(-1, 1, n), rng.uniform(0, 2 * np.pi, n)])


def _params(row):
    mc, dmc, c1, ct1, p1, c2, ct2, p2 = row
    m1, m2 = lalsimutils.m1m2(mc, 0.25 * (1 - dmc ** 2))
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
    s1 = c1 * np.sqrt(1 - ct1 ** 2)
    s2 = c2 * np.sqrt(1 - ct2 ** 2)
    P.s1x, P.s1y, P.s1z = s1 * np.cos(p1), s1 * np.sin(p1), c1 * ct1
    P.s2x, P.s2y, P.s2z = s2 * np.cos(p2), s2 * np.sin(p2), c2 * ct2
    return P


def test_vectorized_matches_extract_param(capsys):
    x = _draws()
    y = lalsimutils.convert_waveform_coordinates(x, coord_names=RING, low_level_coord_names=LOW)
    assert "Fallthrough" not in capsys.readouterr().out
    for i, row in enumerate(x):
        P = _params(row)
        for j, name in enumerate(RING):
            ref = P.extract_param(name)
            d = abs(y[i, j] - ref)
            if name == 'phi12':
                d = min(d, 2 * np.pi - d)
            assert d < 1e-9, (name, y[i, j], ref)


def test_chi_p_vec_limits():
    # one in-plane spin: chi_p_vec equals chi1_perp.  Antiparallel in-plane spins of the weighted size cancel.
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = 10 * lal.MSUN_SI, 5 * lal.MSUN_SI
    P.s1x, P.s1y, P.s2x, P.s2y = 0.4, 0.0, 0.0, 0.0
    assert abs(P.extract_param('chi_p_vec') - 0.4) < 1e-12
    q = 0.5
    A1 = 2 + 1.5 * q
    A2 = 2 + 1.5 / q
    P.s2x = -0.4 / ((A2 / A1) * q ** 2)
    assert P.extract_param('chi_p_vec') < 1e-12
    assert abs(P.extract_param('phi12') - np.pi) < 1e-12
