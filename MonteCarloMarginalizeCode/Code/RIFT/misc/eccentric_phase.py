"""Optional waveform-free effective eccentric phase coordinates.

Masses are detector-frame solar masses. This leading-PN O(e**2) shear is a
regression/proposal coordinate, not waveform evolution or a change to fref.
The input angle is generic: a backend's ``meanPerAno`` need not be a dynamical
mean anomaly. Native parameters and priors retain their original meanings.
At fixed masses/e, angle -> theta is a circle translation with unit Jacobian.
The redundant sine/cosine embedding is for fitting, never a sampling measure.
"""
import numpy as np

MTSUN = 4.9254909476412675e-6
FEATURE_NAMES = ('eccentric_phase_sin', 'eccentric_phase_cos')
ECCENTRICITY_NAMES = ('eccentricity', 'eccentricity_ln', 'eccentricity_squared')


def validate_frequencies(reference_frequency=10., target_frequency=20.):
    """Configuration frequencies are fixed positive scalars, independent of P.fref."""
    for frequency in (reference_frequency, target_frequency):
        if np.ndim(frequency) or not np.isfinite(frequency) or frequency <= 0:
            raise ValueError('Eccentric phase frequencies must be finite positive scalars')


def phase_advance(m1, m2, e, reference_frequency=10., target_frequency=20.):
    """Leading circular plus O(e**2) Peters phase advance (radians).

    A low-eccentricity effective coordinate; accuracy at large e is not claimed.
    Keep the circular term when masses vary. No phase calibration is applied.
    """
    validate_frequencies(reference_frequency, target_frequency)
    m1, m2, e = np.broadcast_arrays(np.asarray(m1, float), np.asarray(m2, float), np.asarray(e, float))
    if (not np.isfinite(m1).all() or not np.isfinite(m2).all() or
            not np.isfinite(e).all() or np.any(m1 <= 0) or np.any(m2 <= 0) or
            np.any(e < 0) or np.any(e >= 1)):
        raise ValueError('Require finite positive masses and 0 <= eccentricity < 1')
    total = m1 + m2
    eta = (m1 / total) * (m2 / total)
    x0 = (np.pi * total * MTSUN * reference_frequency)**(2./3.)
    ratio = (target_frequency / reference_frequency)**(2./3.)
    # expm1 avoids cancellation for nearby coordinate frequencies.
    log_ratio = np.log(ratio)
    circular = -np.expm1(-2.5 * log_ratio) / (32. * eta)
    eccentric = 785. * e**2 * (-np.expm1(-17./3. * log_ratio)) / (8704. * eta)
    return x0**(-2.5) * (circular - eccentric)


def forward(m1, m2, e, angle, reference_frequency=10., target_frequency=20.):
    """Native input angle -> wrapped effective theta in [0, 2*pi)."""
    angle = np.asarray(angle, float)
    if not np.isfinite(angle).all():
        raise ValueError('Require finite angles')
    return (angle + phase_advance(m1, m2, e, reference_frequency, target_frequency)) % (2*np.pi)


def inverse(m1, m2, e, theta, reference_frequency=10., target_frequency=20.):
    """Invert using the *proposed* masses and eccentricity, not the seed's."""
    theta = np.asarray(theta, float)
    if not np.isfinite(theta).all():
        raise ValueError('Require finite angles')
    return (theta - phase_advance(m1, m2, e, reference_frequency, target_frequency)) % (2*np.pi)


def features(m1, m2, e, angle, reference_frequency=10., target_frequency=20.):
    theta = forward(m1, m2, e, angle, reference_frequency, target_frequency)
    return np.stack((np.sin(theta), np.cos(theta)), axis=-1)


def fitting_names(fit_names, sampled_names):
    """Replace only the fitted angle; never mutate sampling or CLI lists."""
    fit_names, sampled_names = list(fit_names), list(sampled_names)
    if set(FEATURE_NAMES).intersection(fit_names + sampled_names):
        raise ValueError('Eccentric phase features must be enabled through the opt-in flag')
    for names in (fit_names, sampled_names):
        if len(set(names)) != len(names) or names.count('meanPerAno') != 1 or sum(p in names for p in ECCENTRICITY_NAMES) != 1:
            raise ValueError('Eccentric phase coordinates require unique coordinates with meanPerAno and one eccentricity coordinate both fitted and sampled')
    return [q for p in fit_names for q in (FEATURE_NAMES if p == 'meanPerAno' else (p,))]


def extract(P, name, reference_frequency=10., target_frequency=20.):
    """Scalar training/plot extraction; waveform-object masses are SI."""
    if name not in FEATURE_NAMES:
        return P.extract_param(name)
    import lal
    return features(P.m1/lal.MSUN_SI, P.m2/lal.MSUN_SI, P.eccentricity,
                    P.meanPerAno, reference_frequency, target_frequency)[FEATURE_NAMES.index(name)]


def convert(x, coord_names, low_level_coord_names, converter,
            reference_frequency=10., target_frequency=20., **kwargs):
    """Vector fit projection using native masses/redshift/Kerr conversion.

    Invalid native rows remain rejected (-inf) without poisoning valid rows.
    ``source_redshift`` and all converter options are passed through unchanged.
    """
    validate_frequencies(reference_frequency, target_frequency)
    base = [p for p in coord_names if p not in FEATURE_NAMES]
    native = converter(x, coord_names=base, low_level_coord_names=low_level_coord_names, **kwargs)
    physical_names = ['m1', 'm2', 'eccentricity', 'meanPerAno']
    if kwargs.get('enforce_kerr', False):
        physical_names += ['chi1', 'chi2']
    # Avoid native scalar fallback merely to undo the sampled e representation.
    # This copy is used only for physical projection: fitting/prior inputs stay
    # native, and the converter still owns all mass/redshift/spin semantics.
    physical_input, physical_low_names = x, list(low_level_coord_names)
    for alias in ('eccentricity_ln', 'eccentricity_squared'):
        if alias in physical_low_names:
            physical_input = np.array(x, dtype=float, copy=True)
            column = physical_low_names.index(alias)
            with np.errstate(over='ignore', invalid='ignore'):
                physical_input[:, column] = (np.exp(physical_input[:, column]) if alias == 'eccentricity_ln'
                                              else np.sqrt(physical_input[:, column]))
            physical_low_names[column] = 'eccentricity'
            break
    physical = converter(physical_input, coord_names=physical_names,
                         low_level_coord_names=physical_low_names, **kwargs)
    valid = (np.isfinite(physical).all(axis=1) & (physical[:, 0] > 0) &
             (physical[:, 1] > 0) & (physical[:, 2] >= 0) & (physical[:, 2] < 1))
    valid &= np.isfinite(native).all(axis=1)
    if kwargs.get('enforce_kerr', False):
        valid &= (physical[:, 4] <= 1) & (physical[:, 5] <= 1)
    values = np.full((len(x), 2), -np.inf)
    values[valid] = features(*physical[valid, :4].T, reference_frequency, target_frequency)
    out = np.empty((len(x), len(coord_names)))
    for i, name in enumerate(coord_names):
        out[:, i] = values[:, FEATURE_NAMES.index(name)] if name in FEATURE_NAMES else native[:, base.index(name)]
    out[~valid] = -np.inf
    return out
