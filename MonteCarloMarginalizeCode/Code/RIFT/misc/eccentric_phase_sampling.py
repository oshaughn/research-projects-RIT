"""Internal periodic-angle sampling with native physical priors and exports.

The sampler integrates L(native) * K against its separable reference prior,
where K = 2*pi*p_native(ell)*|dell/dtheta|. Export converts BOTH prior and
proposal densities to native measure and removes K from the reported lnL.
No integrator implementation or waveform parameter convention is changed.
"""
import numpy as np
from RIFT.misc import eccentric_phase as ep
from RIFT.misc import eccentric_phase_models as models

ANGLE = 'eccentric_phase_angle'


def validate_options(options):
    """Reject native-only auxiliary paths before fitting or integration."""
    if not getattr(options, 'eccentric_phase_sampling', False):
        return
    unsupported = ('sampler_oracle', 'oracle_reference_sample_file',
                   'supplementary_prior_code', 'import_prior_dictionary_file',
                   'output_prior_dictionary_file')
    if any(getattr(options, name, None) for name in unsupported):
        raise ValueError('Sheared-angle sampling does not support native oracle/prior dictionaries')
    supported = {'AV', 'GMM', 'adaptive_cartesian'}
    method = options.sampler_method
    if method == 'portfolio':
        if not set(options.sampler_portfolio or []).issubset({'AV', 'GMM'}):
            raise ValueError('Sheared-angle portfolio sampling supports AV/GMM members only')
    elif method not in supported:
        raise ValueError('Sheared-angle sampling supports AV, GMM, adaptive_cartesian and AV/GMM portfolios')
    if not (0 <= options.meanPerAno_min < options.meanPerAno_max <= 2*np.pi):
        raise ValueError('Sheared-angle sampling requires native anomaly support inside [0, 2pi]')


class SamplingChart:
    def __init__(self, native_names, converter, angle_prior, angle_bounds,
                 reference_frequency=10., target_frequency=20.,
                 procedure='peters-full-e', map_data=None, **converter_options):
        self.native_names = list(native_names)
        if (len(set(native_names)) != len(native_names) or native_names.count('meanPerAno') != 1
                or sum(p in native_names for p in ep.ECCENTRICITY_NAMES) != 1
                or ANGLE in native_names):
            raise ValueError('Sheared sampling requires one native angle and eccentricity coordinate')
        self.names = [ANGLE if p == 'meanPerAno' else p for p in native_names]
        self.index = native_names.index('meanPerAno')
        self.converter = converter
        self.converter_options = converter_options
        self.angle_prior = angle_prior
        self.angle_bounds = tuple(angle_bounds)
        self.reference = reference_frequency
        self.target = target_frequency
        self.model = dict(procedure=procedure, map_data=map_data)
        ep.validate_frequencies(reference_frequency, target_frequency)

    def to_native(self, x):
        """Return native rows, log inverse angular Jacobians, and valid mask.

        Physical extraction uses the existing source-redshift/Kerr converter;
        the map always sees detector-frame masses. Invalid rows are isolated.
        """
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[1] != len(self.native_names):
            raise ValueError('Wrong sheared sampling dimension')
        native = x.copy()
        physical_input = x.copy()
        physical_names = self.native_names.copy()
        for alias in ('eccentricity_ln', 'eccentricity_squared'):
            if alias in physical_names:
                column = physical_names.index(alias)
                with np.errstate(over='ignore', invalid='ignore'):
                    physical_input[:, column] = (np.exp(x[:, column]) if alias.endswith('_ln')
                                                  else np.sqrt(x[:, column]))
                physical_names[column] = 'eccentricity'
        valid = np.isfinite(physical_input).all(axis=1)
        e = physical_input[:, physical_names.index('eccentricity')]
        valid &= (e >= 0) & (e < 1)
        needed = ['m1', 'm2', 'eccentricity']
        if self.model['procedure'] == 'eob-tabulated':
            needed += ['s1z', 's2z']
        physical = np.full((len(x), len(needed)), np.nan)
        if valid.any():
            physical[valid] = self.converter(physical_input[valid], coord_names=needed,
                low_level_coord_names=physical_names, **self.converter_options)
        valid &= np.isfinite(physical).all(axis=1) & (physical[:, 0] > 0) & (physical[:, 1] > 0)
        log_jacobian = np.full(len(x), -np.inf)
        if valid.any():
            m1, m2, e = physical[valid, :3].T
            spin = dict(s1z=physical[valid, 3], s2z=physical[valid, 4]) if len(needed) == 5 else {}
            ell = ep.inverse(m1, m2, e, x[valid, self.index], self.reference, self.target,
                             **self.model, **spin)
            native[valid, self.index] = ell
            derivative = np.ones(len(ell))
            if self.model['procedure'] == 'eob-tabulated' and self.reference != self.target:
                _, harmonics = models.map_profile(m1, m2, e, spin['s1z'], spin['s2z'],
                    self.reference, self.target, self.model['map_data'])
                k = np.arange(1, harmonics.shape[-1]+1)
                derivative += np.sum(harmonics*k*np.cos(ell[:, None]*k), axis=-1)
            log_jacobian[valid] = -np.log(derivative)
        native[~valid] = -np.inf
        return native, log_jacobian, valid

    def log_correction(self, native, log_jacobian, valid):
        """Replace uniform theta prior by the requested native angular prior."""
        result = np.full(len(native), -np.inf)
        ell = native[:, self.index]
        supported = valid & (ell >= self.angle_bounds[0]) & (ell < self.angle_bounds[1])
        if supported.any():
            with np.errstate(divide='ignore', invalid='ignore'):
                result[supported] = (np.log(2*np.pi*np.asarray(self.angle_prior(ell[supported])))
                                     + log_jacobian[supported])
        return result

    def wrap(self, native_function, log_space):
        """Preserve the native callback contract, including supplementary factors."""
        def function(*args):
            x = np.column_stack([np.atleast_1d(v) for v in args])
            native, log_jacobian, valid = self.to_native(x)
            correction = self.log_correction(native, log_jacobian, valid)
            good = valid & np.isfinite(correction)
            result = np.full(len(x), -np.inf if log_space else 0.)
            if good.any():
                value = np.asarray(native_function(*native[good].T)).reshape(-1)
                result[good] = value+correction[good] if log_space else value*np.exp(correction[good])
            return result
        return function

    def restore_samples(self, samples, log_space):
        """Restore physical keys, likelihoods, and native-measure densities.

        p_native = p_reference*K/J; q_native = q_theta/J, J=|dell/dtheta|.
        Thus L_native*p_native/q_native exactly preserves the integration weights.
        """
        x = np.column_stack([samples[p] for p in self.names])
        native, log_jacobian, valid = self.to_native(x)
        correction = self.log_correction(native, log_jacobian, valid)
        good = valid & np.isfinite(correction)
        for key in ('log_integrand', 'integrand'):
            if key not in samples:
                continue
            values = np.asarray(samples[key], float).copy()
            if key == 'log_integrand' or log_space:
                values[good] -= correction[good]
                values[~good] = -np.inf
            else:
                values[good] *= np.exp(-correction[good])
                values[~good] = 0.
            samples[key] = values
        # Rejected nonphysical draws have zero weight. Keep their proposal
        # denominator positive: -inf - (-inf) would poison downstream weights.
        prior_shift = np.full(len(x), -np.inf)
        prior_shift[valid] = correction[valid]-log_jacobian[valid]
        for prior, shift in [('joint_prior', prior_shift),
                             ('joint_s_prior', -log_jacobian)]:
            for key, logarithmic in [(prior, False), ('log_'+prior, True)]:
                if key not in samples:
                    continue
                values = np.asarray(samples[key], float).copy()
                if logarithmic:
                    values[valid] += shift[valid]
                    values[~valid] = -np.inf if prior == 'joint_prior' else 0.
                else:
                    values[valid] *= np.exp(shift[valid])
                    values[~valid] = 0. if prior == 'joint_prior' else 1.
                samples[key] = values
        samples['meanPerAno'] = native[:, self.index]
        del samples[ANGLE]
