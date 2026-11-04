"""Vectorized, array-module-agnostic subset of lalsimutils.convert_waveform_coordinates.

Runs on numpy or cupy (``xpy``) for the coordinate sets distance-slice reconstructions use. Inputs:
masses as (m1, m2), (mc, delta_mc) or (mc, eta); spins as Cartesian s1x..s2z or spherical
chi/cos_theta/phi; any other input name is passed through when the output asks for it. The formulas
are those of convert_waveform_coordinates and ChooseWaveformParams.extract_param (L frame, solar-mass
units), so outputs agree with them to float64 roundoff. A name outside the supported set raises
NotImplementedError: callers fall back to the host converter.
"""
import numpy as np

# RIFT.misc.tools: mu-coordinate rotation and constants (Lee, Morisaki, Tagoshi 2022)
from RIFT.misc.tools import U as _U, fref as _FREF, MsunToTime as _MSUN_T

SUPPORTED_OUTPUTS = ('m1', 'm2', 'mc', 'eta', 'delta_mc', 'q', 'mtot', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
                     'chi1', 'chi2', 'xi', 'chiMinus', 'mu1', 'mu2', 'inv_dist')


def _get_xpy(x):
    try:
        import cupy
        return cupy.get_array_module(x)
    except ImportError:
        return np


def _m1m2(xp, mc, eta):
    ev = 1 - 4 * eta
    ev_sqrt = xp.where(ev >= 0, xp.sqrt(xp.maximum(ev, 0)), 0.0)
    m1 = 0.5 * mc * eta ** (-3. / 5.) * (1. + ev_sqrt)
    m2 = 0.5 * mc * eta ** (-3. / 5.) * (1. - ev_sqrt)
    return m1, m2


def _mu12(xp, mc, q, a1z, a2z):
    """tools.Mcqchi1chi2Tomu1mu2mu3 restricted to (mu1, mu2)."""
    eta = q / ((1 + q) ** 2.0)
    psi0 = (3. / 4.) * (8. * np.pi * mc * _MSUN_T * _FREF) ** (-5. / 3.)
    psi2 = psi0 * (20. / 9.) * (743. / 336. + 11. * eta / 4.) * eta ** (-2. / 5.) * (np.pi * mc * _MSUN_T * _FREF) ** (2. / 3.)
    beta = ((113. / 12. + 25. * q / 4.) * a1z + q ** 2. * (113. / 12. + 25. / (4. * q)) * a2z) / ((1. + q) ** 2.)
    psi3 = psi0 * (4. * beta - 16. * np.pi) * eta ** (-3. / 5.) * np.pi * mc * _MSUN_T * _FREF
    mu1 = _U[0, 0] * psi0 + _U[0, 1] * psi2 + _U[0, 2] * psi3
    mu2 = _U[1, 0] * psi0 + _U[1, 1] * psi2 + _U[1, 2] * psi3
    return mu1, mu2


def convert_waveform_coordinates_xpy(x_in, coord_names, low_level_coord_names, xpy=None):
    """Map rows of x_in from low_level_coord_names to coord_names on x_in's array module."""
    xp = xpy if xpy is not None else _get_xpy(x_in)
    x_in = xp.asarray(x_in, dtype=xp.float64)
    low = list(low_level_coord_names)
    col = {n: x_in[:, i] for i, n in enumerate(low)}
    cache = {}

    def masses():
        if 'm' not in cache:
            if 'm1' in col and 'm2' in col:
                m1, m2 = col['m1'], col['m2']
            elif 'mc' in col and ('delta_mc' in col or 'eta' in col):
                eta = 0.25 * (1 - col['delta_mc'] ** 2) if 'delta_mc' in col else col['eta']
                m1, m2 = _m1m2(xp, col['mc'], eta)
            else:
                raise NotImplementedError("no mass coordinates in %s" % low)
            cache['m'] = (m1, m2)
        return cache['m']

    def spins():
        if 's' not in cache:
            if all(n in col for n in ('s1x', 's1y', 's1z', 's2x', 's2y', 's2z')):
                cache['s'] = tuple(col[n] for n in ('s1x', 's1y', 's1z', 's2x', 's2y', 's2z'))
            elif all(n in col for n in ('chi1', 'cos_theta1', 'phi1', 'chi2', 'cos_theta2', 'phi2')):
                out = []
                for b in '12':
                    chi, ct, ph = col['chi' + b], col['cos_theta' + b], col['phi' + b]
                    st = xp.sqrt(1 - ct ** 2)
                    out.append((chi * st * xp.cos(ph), chi * st * xp.sin(ph), chi * ct))
                cache['s'] = out[0] + out[1]
            else:
                raise NotImplementedError("no spin coordinates in %s" % low)
        return cache['s']

    def mc_of():
        if 'mc' in col:
            return col['mc']
        m1, m2 = masses()
        return (m1 * m2) ** (3. / 5.) * (m1 + m2) ** (-1. / 5.)

    def value(p):
        if p in col:
            return col[p]
        if p in ('m1', 'm2'):
            return masses()[0 if p == 'm1' else 1]
        if p == 'mc':
            return mc_of()
        if p == 'eta':
            if 'delta_mc' in col:
                return 0.25 * (1 - col['delta_mc'] ** 2)
            m1, m2 = masses()
            return m1 * m2 / (m1 + m2) / (m1 + m2)
        if p == 'delta_mc':
            m1, m2 = masses()
            return (m1 - m2) / (m1 + m2)
        if p == 'q':
            m1, m2 = masses()
            return m2 / m1
        if p == 'mtot':
            m1, m2 = masses()
            return m1 + m2
        if p in ('s1x', 's1y', 's1z', 's2x', 's2y', 's2z'):
            return spins()[('s1x', 's1y', 's1z', 's2x', 's2y', 's2z').index(p)]
        if p in ('chi1', 'chi2'):
            s = spins()[0:3] if p == 'chi1' else spins()[3:6]
            return xp.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
        if p in ('xi', 'chiMinus'):
            m1, m2 = masses()
            s = spins()
            sgn = 1 if p == 'xi' else -1
            return (m1 * s[2] + sgn * m2 * s[5]) / (m1 + m2)
        if p in ('mu1', 'mu2'):
            m1, m2 = masses()
            s = spins()
            if 'mu' not in cache:
                cache['mu'] = _mu12(xp, mc_of(), m2 / m1, s[2], s[5])
            return cache['mu'][0 if p == 'mu1' else 1]
        if p == 'inv_dist' and 'dist' in col:
            # as dslice_distance_coord_plugin: the distance fit coordinate 1/d
            return 1.0 / xp.clip(col['dist'], 1e-6, None)
        raise NotImplementedError("coordinate %r from %s" % (p, low))

    return xp.stack([xp.broadcast_to(value(p), (x_in.shape[0],)) for p in coord_names], axis=1)


def check_against(convert_host, coord_names, low_level_coord_names, x_test, rtol=1e-10, atol=1e-12, xpy=None):
    """Compare convert_waveform_coordinates_xpy with a host converter on x_test (numpy rows).

    Returns (ok, max_scaled_error, detail). NaN must match NaN (out-of-range rows are part of the contract)."""
    ref = np.asarray(convert_host(np.asarray(x_test, dtype=float), coord_names=list(coord_names),
                                  low_level_coord_names=list(low_level_coord_names)), dtype=float)
    got = convert_waveform_coordinates_xpy(x_test if xpy is None else xpy.asarray(x_test), coord_names,
                                           low_level_coord_names, xpy=xpy)
    got = np.asarray(got.get() if hasattr(got, 'get') else got, dtype=float)
    if ref.shape != got.shape:
        return False, np.inf, "shape %s vs %s" % (ref.shape, got.shape)
    both_nan = np.isnan(ref) & np.isnan(got)
    err = np.where(both_nan, 0.0, np.abs(got - ref) / (atol + rtol * np.abs(ref)))
    err = np.where(np.isnan(err), np.inf, err)
    worst = float(err.max()) if err.size else 0.0
    j = int(np.unravel_index(np.argmax(err), err.shape)[1]) if err.size else 0
    return worst <= 1.0, worst, "worst column %s" % list(coord_names)[j]
