"""Optional FEW 2.2.0 CPU adapter for restricted Schwarzschild EMRIs.

Outputs are SSB polarizations hp,hc. They contain no LISA response or TDI.
Component masses are detector-frame solar masses, distance is Gpc, and p0 is
passed unchanged in FEW's dimensionless orbital coordinate. Orbital phases
are independent; qK is a spin-axis angle, not orbital inclination xI0.
All-mode selection includes the finite ROMAN catalog, not infinite harmonics.
"""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import hashlib
import math
from typing import Optional

import numpy as np


def _finite_fields(obj, names):
    for name in names:
        value = getattr(obj, name)
        if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isrealobj(value) or not np.isfinite(value):
            raise ValueError(f'{name} must be a finite real scalar')


@dataclass(frozen=True)
class EMRIParameters:
    m1_msun: float
    m2_msun: float
    a: float
    p0: float
    e0: float
    xI0: float
    distance_gpc: float
    qS: float
    phiS: float
    qK: float
    phiK: float
    Phi_phi0: float
    Phi_theta0: float
    Phi_r0: float

    def __post_init__(self):
        _finite_fields(self, self.__dataclass_fields__)
        if not self.m1_msun >= self.m2_msun > 0 or self.distance_gpc <= 0:
            raise ValueError('require m1_msun >= m2_msun > 0 and distance_gpc > 0')
        if self.a != 0 or self.xI0 != 1:
            raise ValueError('restricted model requires a=0 and xI0=1')
        if not (0 < self.e0 < 0.7 and 10 <= self.p0 < 16 + 2*self.e0):
            raise ValueError('restricted interior requires 0<e0<0.7 and 10<=p0<16+2*e0')
        if not (0 <= self.qS <= np.pi and 0 <= self.qK <= np.pi):
            raise ValueError('qS and qK are polar angles in [0,pi]')
        if not (0 <= self.phiS < 2*np.pi and 0 <= self.phiK < 2*np.pi):
            raise ValueError('phiS and phiK are azimuths in [0,2*pi)')
        if self.Phi_theta0 != 0:
            raise ValueError('restricted equatorial model requires unused Phi_theta0=0')

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FEWConfig:
    n_samples: int
    dt_seconds: float
    epoch_start_seconds: float
    epoch_reference: str
    eps: Optional[float] = 1e-5  # FEW threshold; explicitly None for all modes.
    mode_selection: str = 'threshold'
    # Optional trajectory horizon, independent of the sampled output span.
    integration_span_seconds: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, (int, np.integer)) or self.n_samples < 2:
            raise ValueError('n_samples must be an integer >=2')
        _finite_fields(self, ('dt_seconds', 'epoch_start_seconds'))
        if self.dt_seconds <= 0:
            raise ValueError('require dt_seconds>0')
        if self.integration_span_seconds is not None:
            _finite_fields(self, ('integration_span_seconds',))
            if not (self.n_samples-1)*self.dt_seconds <= self.integration_span_seconds < self.n_samples*self.dt_seconds:
                raise ValueError('integration_span_seconds must lie in [(N-1)*dt,N*dt)')
        if self.mode_selection == 'threshold':
            _finite_fields(self, ('eps',))
            if not 0 < self.eps < 1:
                raise ValueError('threshold selection requires 0<eps<1')
        elif self.mode_selection == 'all':
            if self.eps is not None:
                raise ValueError('all-mode selection requires eps=None; threshold is unused')
        else:
            raise ValueError('mode_selection must be threshold or all')
        if not isinstance(self.epoch_reference, str) or not self.epoch_reference.strip():
            raise ValueError('epoch_reference must name the origin and time convention')

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class FEWPolarizations:
    polarizations: np.ndarray
    dt_seconds: float
    epoch_start_seconds: float
    epoch_reference: str
    metadata: dict

    @property
    def n_samples(self):
        return self.polarizations.shape[1]


MODEL_DATA_SHA256 = {
    'FluxNewMinusPNScaled_fixed_y_order.dat': 'c551ace42fbc79febbda132a78868e20900a0a97a06fa4f4d4bf23b12cc1517c',
    'SchwarzschildEccentricInput.hdf5': 'edb90365f3a5ee39b92ea1ebeb96c916d059f0835234b768087d9b7e80d1bede',
    'AmplitudeVectorNorm.dat': 'df04fd629353703d140fad80b6877e0b51e0944f1ba60c50412d66df422752e1',
}


def _verify_prefetched_files(manager):
    """Check pinned model files locally without FEW's downloading get_file API."""
    for name, expected in MODEL_DATA_SHA256.items():
        path = manager.try_get_local_file(name)
        if path is None:
            raise FileNotFoundError(f'Prefetch required FEW model data before running: {name}')
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'FEW model data SHA256 mismatch: {name}')


def _load_few():
    try:
        installed = version('fastemriwaveforms')
    except ModuleNotFoundError as exc:
        raise ImportError('FEW adapter requires fastemriwaveforms==2.2.0 with Python>=3.12') from exc
    if installed != '2.2.0':
        raise ImportError(f'FEW adapter requires fastemriwaveforms==2.2.0; found {installed}')
    from few import get_file_manager
    _verify_prefetched_files(get_file_manager())
    from few.waveform import GenerateEMRIWaveform
    from few.utils.constants import YRSID_SI
    return GenerateEMRIWaveform, YRSID_SI, installed


def _carrier_frequency_bound(times, phase_coefficients, retained_lmkn, span_seconds):
    """Bound retained carrier frequencies via FEW's DOPR853 phase polynomials.

    Public phase coefficients already include FEW's mass-ratio scaling. The
    Bernstein convex-hull bound covers every overlapping dense-output interval.
    This checks carriers only; amplitude/response sidebands still need cadence
    convergence and no strict signal bandlimit is asserted.
    """
    times = np.asarray(times, dtype=float)
    phases = np.asarray(phase_coefficients, dtype=float)
    modes = np.asarray(retained_lmkn)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError('FEW dense-output times must be finite and strictly increasing')
    if phases.shape != (len(times)-1, 3, 8) or not np.all(np.isfinite(phases)):
        raise ValueError('FEW phase coefficients must be finite with shape (n_segments,3,8)')
    if (modes.ndim != 2 or modes.shape[1] != 4 or len(modes) == 0
            or not np.issubdtype(modes.dtype, np.integer)):
        raise ValueError('retained harmonics must be a nonempty integer (l,m,k,n) array')
    if not np.isfinite(span_seconds) or span_seconds <= 0:
        raise ValueError('sample span must be finite and positive')
    slack = 16*np.finfo(float).eps*max(1., span_seconds)
    if times[0] > slack or times[-1]+slack < span_seconds:
        raise ValueError('FEW dense-output intervals do not cover the requested sample span')
    # FEW dopr853.py order=1: coefficients in ascending powers of normalized s.
    derivative_basis = np.array([
        [0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0],
        [1, -2, 0, 0, 0, 0, 0], [0, 2, -3, 0, 0, 0, 0],
        [0, 2, -6, 4, 0, 0, 0], [0, 0, 3, -8, 5, 0, 0],
        [0, 0, 3, -12, 15, -6, 0], [0, 0, 0, 4, -15, 18, -7],
    ], dtype=float)
    power_to_bernstein = np.array([
        [math.comb(k, j)/math.comb(6, j) if j <= k else 0 for j in range(7)]
        for k in range(7)])
    active = (times[:-1] < span_seconds) & (times[1:] > 0)
    widths = np.diff(times)[active]
    derivative = np.einsum('spi,ij->spj', phases[active], derivative_basis)
    carrier_power = np.einsum('mp,spj->smj', modes[:, 1:], derivative)
    carrier_power /= widths[:, None, None] * (2*np.pi)
    bernstein = np.einsum('smj,kj->smk', carrier_power, power_to_bernstein)
    bound = np.max(np.abs(bernstein), axis=-1)
    # Cover floating-point coefficient conversion with a conservative margin.
    bound += 64*np.finfo(float).eps*(np.sum(np.abs(carrier_power), axis=-1)+bound)
    result = float(np.max(bound))
    if not np.isfinite(result):
        raise ValueError('retained carrier frequency bound is not finite')
    return result


def _termination_metadata(generator, config):
    """Refuse FEW's early-termination zero padding for this fixed-span contract."""
    padding = int(generator.create_waveform.num_pts_pad)
    end_time = float(generator.end_time)
    requested_span = (config.n_samples-1)*config.dt_seconds
    slack = 16*np.finfo(float).eps*max(1., requested_span)
    if padding != 0 or not np.isfinite(end_time) or end_time+slack < requested_span:
        raise ValueError('FEW physical trajectory ended before the requested span; zero padding is refused')
    return {'trajectory_end_time_seconds': end_time, 'zero_padding_samples': padding,
            'termination_policy': 'require physical evolution across all requested samples'}


def _few_duration_years(n_samples, dt_seconds, year_seconds, integration_span_seconds=None):
    """Request N-1 steps with roundoff margin inside the same FEW output bin.

    FEW passes unquantized T to trajectory integration before summation floors
    it. The forward margin avoids an endpoint rounded below the last sample.
    An explicit horizon may extend beyond the last sample within the same output
    bin, allowing two cadences to share a trajectory integration request. The
    unchanged physical-support/no-padding guard verifies the actual result.
    """
    steps = n_samples - 1
    span = steps*dt_seconds
    if int(span/dt_seconds) != steps:
        raise ValueError('FEW sample-count arithmetic cannot represent this cadence and count')
    horizon = span if integration_span_seconds is None else integration_span_seconds
    if not np.isfinite(horizon) or not span <= horizon < n_samples*dt_seconds:
        raise ValueError('integration horizon must lie in [(N-1)*dt,N*dt)')
    duration = (horizon*(1+64*np.finfo(float).eps))/year_seconds
    for _ in range(4):
        recovered = int(duration*year_seconds/dt_seconds)
        if recovered == steps:
            return duration
        duration = np.nextafter(duration, np.inf if recovered < steps else 0.)
    raise ValueError('cannot represent the requested FEW endpoint duration')


class FewWaveformAdapter:
    """Generate SSB hp,hc with a fixed CPU model and explicit sampling contract.

    Required model data must be prefetched by the caller. Local SHA256 checks
    precede model construction and refuse missing data. This adapter does not certify
    adiabatic accuracy at any mass ratio; its bounds restrict the model domain.
    """
    model = 'FastSchwarzschildEccentricFlux'

    def __init__(self, config):
        if not isinstance(config, FEWConfig):
            raise TypeError('config must be FEWConfig')
        self.config = config
        factory, self.year_seconds, self.few_version = _load_few()
        self.waveform = factory(self.model, force_backend='cpu', frame='detector',
                                return_list=True, flip_output=False,
                                sum_kwargs={'output_type': 'td', 'pad_output': True})

    def generate(self, params):
        if not isinstance(params, EMRIParameters):
            raise TypeError('params must be EMRIParameters')
        c = self.config
        duration_years = _few_duration_years(c.n_samples, c.dt_seconds, self.year_seconds,
                                             c.integration_span_seconds)
        horizon = ((c.n_samples-1)*c.dt_seconds if c.integration_span_seconds is None
                   else c.integration_span_seconds)
        selection = {'mode_selection': c.mode_selection, 'include_minus_mkn': True, 'batch_size': -1}
        if c.mode_selection == 'threshold':
            selection['mode_selection_threshold'] = c.eps
        hp, hc = self.waveform(
            params.m1_msun, params.m2_msun, params.a, params.p0, params.e0, params.xI0,
            params.distance_gpc, params.qS, params.phiS, params.qK, params.phiK,
            params.Phi_phi0, params.Phi_theta0, params.Phi_r0,
            T=duration_years, dt=c.dt_seconds, **selection)
        polarizations = np.asarray([hp, hc])
        if polarizations.ndim != 2 or polarizations.shape[0] != 2 or not np.isrealobj(polarizations):
            raise ValueError('FEW must return real hp,hc arrays with matching shape')
        if polarizations.shape[1] != c.n_samples or not np.all(np.isfinite(polarizations)):
            raise ValueError('FEW output must be finite with the requested sample count')
        generator = self.waveform.waveform_generator
        termination = _termination_metadata(generator, c)
        retained_modes = np.column_stack([generator.ls, generator.ms,
                                          generator.ks, generator.ns]).astype(int).tolist()
        trajectory = generator.inspiral_generator
        carrier_bound = _carrier_frequency_bound(trajectory.integrator_spline_t,
            trajectory.integrator_spline_phase_coeff, retained_modes,
            (c.n_samples-1)*c.dt_seconds)
        carrier_ratio = 2*c.dt_seconds*carrier_bound
        if carrier_ratio >= 1:
            raise ValueError('retained EMRI carrier frequency bound reaches Nyquist; reduce dt_seconds')
        metadata = {**termination, 'retained_lmkn': retained_modes,
                    'max_retained_carrier_frequency_bound_hz': carrier_bound,
                    'carrier_nyquist_ratio': carrier_ratio,
                    'mode_selection': c.mode_selection, 'retained_mode_count': len(retained_modes),
                    'include_minus_mkn': True,
                    'few_version': self.few_version, 'model': self.model, 'backend': 'cpu',
                    'frame': 'SSB polarizations (FEW detector frame)',
                    'YRSID_SI': self.year_seconds, 'backend_duration_years': duration_years,
                    'sample_span_seconds': (c.n_samples-1)*c.dt_seconds,
                    'requested_integration_span_seconds': horizon,
                    'integration_margin_seconds': duration_years*self.year_seconds-horizon,
                    'model_data_sha256': dict(MODEL_DATA_SHA256),
                    'parameters': params.to_dict(),
                    'config': c.to_dict()}
        return FEWPolarizations(polarizations, c.dt_seconds, c.epoch_start_seconds,
                                c.epoch_reference, metadata)

    __call__ = generate
