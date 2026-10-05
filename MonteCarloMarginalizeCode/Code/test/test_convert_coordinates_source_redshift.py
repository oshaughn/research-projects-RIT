"""convert_waveform_coordinates(..., source_redshift=z) must return detector-frame values.

Reference: the per-row ChooseWaveformParams path the function itself falls back to
(assign source-frame low-level coordinates, scale P.m1, P.m2 by 1+z, extract_param).
Masses are in Msun, as CIP passes them.
"""
import numpy as np
import pytest

from RIFT import lalsimutils as lsu

N = 50


def _draw(low_level_coord_names, rng):
    ranges = {
        'mc': (5., 40.), 'delta_mc': (0.05, 0.85), 'eta': (0.1, 0.249),
        's1z': (-0.9, 0.9), 's2z': (-0.9, 0.9),
        'chi1': (0., 0.99), 'chi2': (0., 0.99),
        'cos_theta1': (-1., 1.), 'cos_theta2': (-1., 1.),
        'phi1': (0., 2*np.pi), 'phi2': (0., 2*np.pi),
        's1z_bar': (-0.9, 0.9), 's2z_bar': (-0.9, 0.9),
        'chi1_perp_bar': (0., 0.9), 'chi2_perp_bar': (0., 0.9),
        'lambda1': (0., 1000.), 'lambda2': (0., 1000.),
    }
    return np.column_stack([rng.uniform(*ranges[p], N) for p in low_level_coord_names])


def _per_row(x_in, coord_names, low_level_coord_names, z):
    out = np.zeros((len(x_in), len(coord_names)))
    for i, row in enumerate(x_in):
        P = lsu.ChooseWaveformParams()
        for p, val in zip(low_level_coord_names, row):
            P.assign_param(p, val)
        P.m1 *= 1 + z
        P.m2 *= 1 + z
        out[i] = [P.extract_param(p) for p in coord_names]
    return out


CASES = [
    # spherical spins: vectorized mu1, mu2 (the case PR #204's RF features use)
    (['mc', 'delta_mc', 'mu1', 'mu2', 'xi', 'chiMinus', 's1x', 's1y', 's2x', 's2y'],
     ['mc', 'delta_mc', 'chi1', 'cos_theta1', 'phi1', 'chi2', 'cos_theta2', 'phi2']),
    # spherical spins: vectorized in-plane ring coordinates (#387) next to mu1, mu2
    (['mc', 'mu1', 'mu2', 'chi1_perp', 'chi2_perp', 'phi12', 'SOverM2_perp', 'DeltaOverM2_perp', 'chi_p_vec'],
     ['mc', 'delta_mc', 'chi1', 'cos_theta1', 'phi1', 'chi2', 'cos_theta2', 'phi2']),
    # aligned cartesian: vectorized mu1, mu2
    (['mu1', 'mu2', 'delta_mc', 'chiMinus', 'mc'], ['mc', 'delta_mc', 's1z', 's2z']),
    # vectorized m1, m2 from mc, eta
    (['mc', 'm1', 'm2', 'eta', 'xi'], ['mc', 'eta', 's1z', 's2z']),
    # pseudo-cylindrical spins: vectorized mu1, mu2.  s*z_bar precedes chi*_perp_bar because
    # assign_param('chi1_perp_bar') reads the current s1z; the reverse order builds a different spin.
    (['mu1', 'mu2', 'delta_mc', 'xi', 'chiMinus', 'chi_p'],
     ['mc', 'delta_mc', 's1z_bar', 'chi1_perp_bar', 'phi1', 's2z_bar', 'chi2_perp_bar', 'phi2']),
    # tidal: vectorized mu1, mu2 next to LambdaTilde
    (['mu1', 'mu2', 'delta_mc', 'LambdaTilde', 'DeltaLambdaTilde'],
     ['mc', 'delta_mc', 's1z', 's2z', 'lambda1', 'lambda2']),
    # mtot and q go through the per-row fallback: the redshift must be applied once, not twice
    (['mc', 'mtot', 'q', 'm1'], ['mc', 'delta_mc', 's1z', 's2z']),
]


@pytest.mark.parametrize("z", [0.0, 0.3])
@pytest.mark.parametrize("coord_names,low_level_coord_names", CASES)
def test_vectorized_matches_per_row(coord_names, low_level_coord_names, z):
    rng = np.random.default_rng(20261004)
    x_in = _draw(low_level_coord_names, rng)
    x_in_before = x_in.copy()
    got = lsu.convert_waveform_coordinates(x_in, coord_names=coord_names,
                                           low_level_coord_names=low_level_coord_names,
                                           source_redshift=z)
    ref = _per_row(x_in, coord_names, low_level_coord_names, z)
    np.testing.assert_array_equal(x_in, x_in_before)  # caller's array is not rescaled in place
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12,
                               err_msg=f"z={z} coords={coord_names}")


def test_mass_columns_scale_on_every_row():
    rng = np.random.default_rng(1)
    low = ['mc', 'eta', 's1z', 's2z']
    x_in = _draw(low, rng)
    names = ['mc', 'm1', 'm2']
    z0 = lsu.convert_waveform_coordinates(x_in, coord_names=names, low_level_coord_names=low)
    z1 = lsu.convert_waveform_coordinates(x_in, coord_names=names, low_level_coord_names=low,
                                          source_redshift=0.3)
    np.testing.assert_allclose(z1, 1.3*z0, rtol=1e-12)
