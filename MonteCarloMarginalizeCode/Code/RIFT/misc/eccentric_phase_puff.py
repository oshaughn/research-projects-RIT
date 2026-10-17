"""Circular-chart helpers for the optional effective eccentric-phase puff."""
import numpy as np


def centered_angles(angles):
    """Place a circle in a local covariance chart, without unwrapping a ridge."""
    angles = np.asarray(angles, dtype=float)
    center = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return center + (angles-center+np.pi) % (2*np.pi)-np.pi, float(center)


def minimum_periodic_distances(existing, proposed, inverse_covariance, column,
                               block_size=64):
    """Minimum Mahalanobis distance on one periodic angle and linear axes.

    Minimize the quadratic form over every integer 2pi lift analytically.
    Cross-covariances mean simply wrapping the angle difference is insufficient.
    """
    existing = np.asarray(existing, dtype=float)
    proposed = np.asarray(proposed, dtype=float)
    vi = np.asarray(inverse_covariance, dtype=float)
    result = np.empty(len(proposed))
    for start in range(0, len(proposed), block_size):
        d = existing[:, None, :] - proposed[None, start:start+block_size, :]
        square = np.einsum('...i,ij,...j->...', d, vi, d)
        if vi[column, column] > 0:
            cross = np.einsum('...i,i->...', d, vi[:, column])
            k = np.rint(-cross/(2*np.pi*vi[column, column]))
            lift = 2*np.pi*k
            square += 2*lift*cross + lift*lift*vi[column, column]
        result[start:start+block_size] = np.sqrt(np.maximum(0, square.min(axis=0)))
    return result


def native_proposals(parameters, values, names, reference_frequency,
                     target_frequency):
    """Assign all non-angle proposals before inverting the effective angle.

    Retain unchanged fields from each original waveform object. This also works
    when masses are not puffed, or the phase column precedes mass/e columns.
    Invalid physical proposals get a NaN anomaly and are rejected downstream.
    """
    import lal
    from RIFT.misc import eccentric_phase
    angle_column = names.index('meanPerAno')
    physical = np.array(values, dtype=float, copy=True)
    candidates = []
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        for old, row in zip(parameters, values):
            p = old.copy()
            for index, name in enumerate(names):
                if index == angle_column:
                    continue
                scale = lal.MSUN_SI if name in ('m1','m2','mc','mtot') else 1.
                p.assign_param(name, row[index]*scale)
            candidates.append(p)
    m1 = np.array([p.m1/lal.MSUN_SI for p in candidates])
    m2 = np.array([p.m2/lal.MSUN_SI for p in candidates])
    e = np.array([p.eccentricity for p in candidates])
    valid = np.isfinite(m1+m2+e+values[:,angle_column]) & (m1>0) & (m2>0) & (e>=0) & (e<1)
    physical[:,angle_column] = np.nan
    physical[valid,angle_column] = eccentric_phase.inverse(
        m1[valid],m2[valid],e[valid],values[valid,angle_column],
        reference_frequency=reference_frequency,target_frequency=target_frequency)
    for p, angle in zip(candidates, physical[:,angle_column]):
        p.meanPerAno = angle
    return candidates, physical
