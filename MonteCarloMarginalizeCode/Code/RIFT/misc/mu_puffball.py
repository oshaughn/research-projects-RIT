"""Mass/aligned-spin covariance puff with independent random transverse spins.

Masses are in solar masses and q=m2/m1 <= 1. These are proposal coordinates,
not a waveform approximation or a prior. No transverse covariance is fitted.
"""
import numpy as np

from RIFT.misc import tools


def inverse_mu(mu1, mu2, q, s2z):
    """Broadcast an atomic (mu1,mu2,q,s2z) inverse; invalid rows return NaNs.

    Eliminating psi3 yields A*x**5+B(eta)*x**3=y for x=Mc**(-1/3).
    A,B are positive for tools.U, so y>0 has exactly one positive root.
    Bounded vectorized bisection also handles physical negative-mu1 points.
    tools.fref (200 Hz) is a coordinate convention, not waveform fref.
    """
    mu1, mu2, q, s2z = np.broadcast_arrays(
        *[np.asarray(v, dtype=float) for v in (mu1, mu2, q, s2z)])
    valid = (np.isfinite(mu1) & np.isfinite(mu2) & np.isfinite(q)
             & np.isfinite(s2z) & (q > 0) & (q <= 1))
    safe_q = np.where(valid, q, 1.)
    eta = tools.qToeta(safe_q)
    ratio = tools.U[0, 2] / tools.U[1, 2]
    c = tools.U[0] - ratio * tools.U[1]
    # Wild finite proposals can overflow just like nonfinite proposals; reject
    # them rather than letting a single row contaminate the rest of the batch.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        y = np.where(valid, mu1, 1.) - ratio * np.where(valid, mu2, 0.)
        valid &= np.isfinite(y) & (y > 0)
        safe_y = np.where(valid, y, 1.)
        a = c[0] * tools.psi0(1.)
        b = c[1] * tools.psi2(1., eta)
        lo = np.zeros_like(safe_y)
        hi = np.minimum((safe_y/a)**.2, (safe_y/b)**(1./3))
        for _ in range(48):
            x = (lo + hi) * .5
            above = a*x**5 + b*x**3 > safe_y
            hi = np.where(above, x, hi)
            lo = np.where(above, lo, x)
        mc = ((hi + lo)*.5)**-3
        p0 = tools.psi0(mc)
        beta = 4*np.pi + (
            np.where(valid, mu2, 0.) - tools.U[1, 0]*p0
            - tools.U[1, 1]*tools.psi2(mc, eta)) / (
                tools.U[1, 2]*4*np.pi*mc*tools.MsunToTime*tools.fref
                * eta**(-.6)*p0)
        # Use q directly: etaToq loses precision near equal masses.
        z1 = ((1+safe_q)**2*beta
              - safe_q**2*(113/12+25/(4*safe_q))*np.where(valid, s2z, 0.)) / (
                  113/12+25*safe_q/4)
    valid &= np.isfinite(mc) & (mc > 0) & np.isfinite(z1)
    return np.where(valid, mc, np.nan), np.where(valid, z1, np.nan)


def aligned_coordinates(rows):
    """Encode [m1,m2,s1x,s1y,s1z,s2x,s2y,s2z] to mu1,mu2,q,s2z."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 8 or not len(rows):
        raise ValueError('mu puff requires a nonempty (n,8) mass/spin array')
    if (not np.isfinite(rows).all() or np.any(rows[:, :2] <= 0)
            or np.any(rows[:, 1] > rows[:, 0])):
        raise ValueError('mu puff seeds require finite fields and m1 >= m2 > 0')
    if (np.any(np.linalg.norm(rows[:, 2:5], axis=1) > 1+1e-12)
            or np.any(np.linalg.norm(rows[:, 5:8], axis=1) > 1+1e-12)):
        raise ValueError('mu puff seeds must satisfy both unit-spin bounds')
    m1, m2 = rows[:, :2].T
    q = m2/m1
    mu1, mu2, _ = tools.Mcqchi1chi2Tomu1mu2mu3(
        tools.m1m2ToMc(m1, m2), q, rows[:, 4], rows[:, 7])
    return np.column_stack((mu1, mu2, q, rows[:, 7]))


def reflection_ranges(parameters, downselect):
    """Physical chart bounds, narrowed by native q/eta/s2z cuts if present."""
    bounds = {}
    for name in parameters or ():
        if name not in ('q', 's2z'):
            raise ValueError('mu puff reflection supports only q and s2z')
        lo, hi = (0., 1.) if name == 'q' else (-1., 1.)
        if name in downselect:
            limits = downselect[name]
            if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError('invalid native reflection bounds for ' + name)
            lo, hi = max(lo, limits[0]), min(hi, limits[1])
        if name == 'q' and 'eta' in downselect:
            emin, emax = downselect['eta']
            if not np.isfinite([emin, emax]).all() or not 0 <= emin < emax <= .25:
                raise ValueError('q reflection requires 0 <= eta_min < eta_max <= 0.25')
            # Stable also at eta=0 and near equal mass; etaToq subtracts
            # large nearly equal quantities at small eta and divides by zero.
            lo = max(lo, 4*emin/(1+np.sqrt(1-4*emin))**2)
            hi = min(hi, 4*emax/(1+np.sqrt(1-4*emax))**2)
        bounds[name] = (lo, hi)
    return bounds


def reflect_coordinates(coordinates, bounds):
    """Fold q/s2z repeatedly before inversion, keeping both mu values fixed."""
    out = np.array(coordinates, dtype=float, copy=True)
    for name, limits in (bounds or {}).items():
        if name not in ('q', 's2z'):
            raise ValueError('mu puff reflection supports only q and s2z')
        lo, hi = map(float, limits)
        physical = (0., 1.) if name == 'q' else (-1., 1.)
        if not np.isfinite([lo, hi]).all() or not physical[0] <= lo < hi <= physical[1]:
            raise ValueError('invalid reflection bounds for ' + name)
        index = 2 if name == 'q' else 3
        with np.errstate(invalid='ignore', over='ignore'):
            folded = np.remainder(out[:, index] - lo, 2*(hi-lo))
            out[:, index] = lo + np.minimum(folded, 2*(hi-lo)-folded)
    return out


def propose(rows, *, count=None, scale=1., transverse_max=1., rng=None,
            reflection_bounds=None):
    """Puff four correlated coordinates, independently REDRAW transverse spins.

    By default each input row gets one proposal. An explicit count instead
    samples parents uniformly, preserving parent indices for metadata.
    Radius is uniform on [0,min(transverse_max,sqrt(1-z**2))],
    azimuth uniform on [0,2pi), independently for both bodies. This is proposal
    exploration, not an isotropic-spin prior. It works with aligned seeds and
    with contracted later grids using the same settings. Rejected inverse rows
    are discarded without clipping or retries; the attempted budget is count.
    Optional q/s2z reflections precede the coupled inverse, preserving mu1/mu2.
    """
    if not np.isfinite(scale) or scale < 0:
        raise ValueError('mu puff scale must be finite and nonnegative')
    if not np.isfinite(transverse_max) or not 0 <= transverse_max <= 1:
        raise ValueError('transverse_max must lie in [0,1]')
    z = aligned_coordinates(rows)
    one_per_seed = count is None
    if one_per_seed:
        count = len(z)
    if isinstance(count, bool) or int(count) != count or count < 1:
        raise ValueError('mu puff count must be a positive integer')
    count = int(count)
    rng = np.random.default_rng(rng)
    covariance = np.cov(z, rowvar=False) if len(z) > 1 else np.zeros((4, 4))
    # Normalize before factoring: mu and q have very different scales. Genuine
    # zero directions stay zero, rather than receiving the legacy huge fallback.
    sd = np.sqrt(np.maximum(np.diag(covariance), 0))
    units = np.where(sd > 0, sd, 1.)
    correlation = covariance / units[:, None] / units[None, :]
    ev, vec = np.linalg.eigh(correlation)
    factor = units[:, None] * vec * np.sqrt(np.maximum(ev, 0))[None, :]
    parents = np.arange(len(z)) if one_per_seed else rng.integers(len(z), size=count)
    out = z[parents] + scale * (rng.normal(size=(count, 4)) @ factor.T)
    out = reflect_coordinates(out, reflection_bounds)
    mc, z1 = inverse_mu(*out.T)
    q, z2 = out[:, 2], out[:, 3]
    valid = (np.isfinite(mc) & np.isfinite(z1) & (np.abs(z1) <= 1)
             & np.isfinite(z2) & (np.abs(z2) <= 1))
    mc, q, z1, z2 = mc[valid], q[valid], z1[valid], z2[valid]
    m1, m2 = tools.McqTom1m2(mc, q)
    radii = rng.uniform(size=(len(mc), 2)) * np.minimum(
        transverse_max, np.sqrt(np.maximum(0, 1-np.column_stack((z1, z2))**2)))
    phi = rng.uniform(0, 2*np.pi, size=(len(mc), 2))
    x, y = radii*np.cos(phi), radii*np.sin(phi)
    proposed = np.column_stack((m1, m2, x[:, 0], y[:, 0], z1,
                                x[:, 1], y[:, 1], z2))
    return proposed, parents[valid], dict(
        attempted=count, inverse_accepted=len(proposed),
        coordinate_system='mu-random-transverse', puff_factor=float(scale),
        transverse_max=float(transverse_max),
        reflection_bounds={k: list(v) for k, v in (reflection_bounds or {}).items()},
        transverse_distribution='uniform radius conditional on aligned spin; uniform azimuth')


def waveform_proposals(parameters, *, downselect, reflect_parameters=None, fref=None, fmin=None,
                       duration_bound=None, **kwargs):
    """Atomically update mass/spin fields and apply native physical cuts.

    All other fields, including tides, eccentricity, frequencies and source
    metadata, come from the selected parent object, never sequential setters.
    """
    import lal
    from RIFT import lalsimutils

    if not parameters:
        raise ValueError('mu puff requires at least one seed')
    fields = ('m1', 'm2', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z')
    rows = np.array([[getattr(p, name) / (lal.MSUN_SI if i < 2 else 1.)
                      for i, name in enumerate(fields)] for p in parameters])
    rows, parents, stats = propose(
        rows, reflection_bounds=reflection_ranges(reflect_parameters, downselect), **kwargs)
    result = []
    for row, parent in zip(rows, parents):
        p = parameters[parent].copy()
        for i, name in enumerate(fields):
            setattr(p, name, row[i] * (lal.MSUN_SI if i < 2 else 1.))
        if fref is not None:
            p.fref = fref
        if fmin is not None:
            p.fmin = fmin
        keep = True
        for name, bounds in downselect.items():
            value = p.extract_param(name)
            if name in ('m1', 'm2', 'mc', 'mtot'):
                value /= lal.MSUN_SI
            if not np.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                keep = False
                break
        if keep and duration_bound is not None:
            duration = lalsimutils.estimateWaveformDuration(p)
            keep = np.isfinite(duration) and duration <= duration_bound
        if keep:
            result.append(p)
    stats['exported_points'] = len(result)
    return result, stats
