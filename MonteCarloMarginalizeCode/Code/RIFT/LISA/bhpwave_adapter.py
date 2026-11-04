"""Optional, explicitly locked BHPWave circular CPU polarizations.

Masses are observed primary/secondary solar masses, r0 is in primary-mass
Boyer-Lindquist units, distance is Gpc, and angles use BHPWave's SSB basis.
The selected (2,+/-2) pair is not a complete waveform or a LISA response.
"""
from fractions import Fraction
from typing import Optional
from dataclasses import asdict, dataclass
import hashlib
import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import re
import sys

import numpy as np


SOURCE_REVISION = '50649254f4cd2e8d2ac6896aa6d4e4c7b73ed604'
DATA_INDEX_SHA256 = 'da5634dffee613afab8f2a97a71496a142c64e16b0b0bdc86d0626b4412cf2b1'
SOURCE_FILES = {
    'bhpwave/waveform.py': 'c896ee1f6cb74527d3e96fa31243aafac902b672c7dbaf86b79c05f29443c661',
    'bhpwave/constants.py': '223736924c0b0dd0b8961af3cbb713fd5087aeb4302ace96b7db629db47cb435',
    'bhpwave/trajectory/geodesic.py': '3b1d14b7bbf7b75aa18707d77e0b8badc43ba86fd2e179e41b8c2461f8745f73',
    'bhpwave/trajectory/inspiral.py': '0bffe347b90e67d03f6fa8890960f5bc1c1413bc08248b08235888b8e5872394',
}
MAX_SAMPLES = 262144
MAX_NATIVE_SAMPLES = 524288


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('{} must be a finite real scalar'.format(name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('{} must be a finite real scalar'.format(name))
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BHPWaveParameters:
    primary_msun: float
    secondary_msun: float
    a: float
    r0: float
    distance_gpc: float
    qS: float
    phiS: float
    qK: float
    phiK: float
    Phi_phi0: float

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if not self.primary_msun >= self.secondary_msun > 0:
            raise ValueError('require primary_msun >= secondary_msun > 0')
        if self.distance_gpc <= 0 or self.r0 <= 0 or abs(self.a) > 0.995:
            raise ValueError('require distance_gpc>0, r0>0 and |a|<=0.995')
        if not (0 <= self.qS <= np.pi and 0 <= self.qK <= np.pi):
            raise ValueError('qS and qK are SSB polar angles in [0,pi]')
        if not (0 <= self.phiS < 2*np.pi and 0 <= self.phiK < 2*np.pi):
            raise ValueError('phiS and phiK are SSB azimuths in [0,2*pi)')

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        return asdict(self)


def _native_sampling(n_samples, dt_seconds, generation_dt_seconds):
    native_dt = dt_seconds if generation_dt_seconds is None else _real(generation_dt_seconds, 'generation_dt_seconds')
    if native_dt <= 0 or native_dt > dt_seconds:
        raise ValueError('generation cadence must be positive and no coarser than output')
    ratio = Fraction.from_float(dt_seconds) / Fraction.from_float(native_dt)
    if ratio.denominator != 1:
        raise ValueError('output cadence must be an exact integer stride of generation cadence')
    stride = ratio.numerator
    native_count = (n_samples-1)*stride+1
    if native_count > MAX_NATIVE_SAMPLES:
        raise ValueError('internal native sample count exceeds {}'.format(MAX_NATIVE_SAMPLES))
    return native_count, native_dt, stride


@dataclass(frozen=True)
class BHPWaveConfig:
    n_samples: int
    dt_seconds: float
    epoch_start_seconds: float
    epoch_reference: str
    generation_dt_seconds: Optional[float] = None

    def __post_init__(self):
        if (isinstance(self.n_samples, (bool, np.bool_))
                or not isinstance(self.n_samples, Integral)
                or not 2 <= self.n_samples <= MAX_SAMPLES):
            raise ValueError('n_samples must be an integer in [2,{}]'.format(MAX_SAMPLES))
        object.__setattr__(self, 'n_samples', int(self.n_samples))
        for name in ('dt_seconds', 'epoch_start_seconds'):
            object.__setattr__(self, name, _real(getattr(self, name), name))
        if self.dt_seconds <= 0:
            raise ValueError('dt_seconds must be positive')
        _real((self.n_samples-1)*self.dt_seconds, 'sample span')
        if self.generation_dt_seconds is not None:
            object.__setattr__(self, 'generation_dt_seconds', _real(self.generation_dt_seconds, 'generation_dt_seconds'))
        _native_sampling(self.n_samples, self.dt_seconds, self.generation_dt_seconds)
        if not isinstance(self.epoch_reference, str) or not self.epoch_reference.strip():
            raise ValueError('epoch_reference must name the time origin and scale')

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BHPWavePolarizations:
    polarizations: np.ndarray
    dt_seconds: float
    epoch_start_seconds: float
    epoch_reference: str
    metadata: dict

    @property
    def n_samples(self):
        return self.polarizations.shape[1]


def _read_lock(path):
    path = Path(path)
    lock = json.loads(path.read_text())
    required = {'schema_version', 'source_revision', 'python_abi', 'architecture',
                'data_files', 'native_module_sha256', 'native_libraries',
                'build_preparation', 'producer_result_sha256'}
    if not isinstance(lock, dict) or set(lock) != required:
        raise ValueError('runtime lock has unsupported or missing fields')
    if type(lock['schema_version']) is not int or lock['schema_version'] != 1:
        raise ValueError('runtime lock requires integer schema_version=1')
    if lock['source_revision'] != SOURCE_REVISION:
        raise ValueError('unsupported BHPWave source revision')
    if platform.system() != 'Linux' or lock['architecture'] != platform.machine():
        raise ValueError('runtime lock requires its recorded Linux architecture')
    if lock['python_abi'] != sys.implementation.cache_tag:
        raise ValueError('runtime lock Python ABI mismatch')
    files = lock['data_files']
    if not isinstance(files, dict) or any(not isinstance(key, str) for key in files):
        raise ValueError('runtime lock needs a data-file mapping')
    canonical = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    if hashlib.sha256(canonical).hexdigest() != DATA_INDEX_SHA256:
        raise ValueError('runtime lock data index does not match pinned BHPWave data')
    if not isinstance(lock['native_module_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', lock['native_module_sha256']):
        raise ValueError('runtime lock needs a native-module SHA256')
    if not isinstance(lock['native_libraries'], dict) or not lock['native_libraries']:
        raise ValueError('runtime lock needs native-library provenance')
    for entry in lock['native_libraries'].values():
        if (not isinstance(entry, dict) or not {'resolved', 'sha256'} <= set(entry)
                or not isinstance(entry['resolved'], str)
                or not re.fullmatch('[0-9a-f]{64}', str(entry['sha256']))):
            raise ValueError('runtime lock native-library entries need resolved and sha256')
    return lock, _sha256(path)


def _verify_native_libraries(entries):
    """Check actual process mappings, including libraries loaded before BHPWave."""
    mapped = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            mapped.add(str(Path(fields[5]).resolve()))
    verified = {}
    for name, entry in entries.items():
        path = Path(entry['resolved']).resolve(strict=True)
        if str(path) not in mapped or _sha256(path) != entry['sha256']:
            raise ValueError('locked native library is not the mapped artifact: '+name)
        verified[name] = {'resolved': str(path), 'sha256': entry['sha256']}
    return verified


def _load_backend(runtime_lock):
    lock, lock_sha256 = _read_lock(runtime_lock)
    try:
        waveform = importlib.import_module('bhpwave.waveform')
        trajectory = importlib.import_module('bhpwave.trajectory.inspiral')
        constants = importlib.import_module('bhpwave.constants')
        geodesic = importlib.import_module('bhpwave.trajectory.geodesic')
        native = importlib.import_module('bhpwaveformcy')
    except ModuleNotFoundError as exc:
        raise ImportError('missing BHPWave runtime or dependency '+repr(exc.name)+
                          '; provide the pinned optional CPU runtime on PYTHONPATH') from exc
    except ImportError as exc:
        raise ImportError('BHPWave runtime could not load; check native ABI and shared-library '
                          'compatibility against the runtime lock: '+str(exc)) from exc
    root = Path(waveform.__file__).resolve().parent.parent
    for name, expected in SOURCE_FILES.items():
        if _sha256(root/name) != expected:
            raise ValueError('BHPWave source mismatch: '+name)
    for module, relative in [(trajectory,'bhpwave/trajectory/inspiral.py'),
                             (constants,'bhpwave/constants.py'),
                             (geodesic,'bhpwave/trajectory/geodesic.py')]:
        if Path(module.__file__).resolve() != root/relative:
            raise ValueError('loaded BHPWave modules do not share the verified root')
    if not any(str(native.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise ValueError('BHPWave CPU backend is not a native extension')
    if _sha256(native.__file__) != lock['native_module_sha256']:
        raise ValueError('loaded BHPWave native module differs from runtime lock')
    for relative, expected in lock['data_files'].items():
        if _sha256(root/relative) != expected:
            raise ValueError('BHPWave data mismatch: '+relative)
    libraries = _verify_native_libraries(lock['native_libraries'])
    receipt = {'runtime_lock_sha256': lock_sha256, 'source_revision': SOURCE_REVISION,
               'source_files': dict(SOURCE_FILES), 'data_index_sha256': DATA_INDEX_SHA256,
               'native_module_sha256': lock['native_module_sha256'],
               'native_libraries': libraries, 'model_root': str(root)}
    return waveform, trajectory, constants, native, receipt


class BHPWaveAdapter:
    """Generate the explicitly selected circular harmonic pair on one CPU thread."""

    def __init__(self, config, runtime_lock):
        if not isinstance(config, BHPWaveConfig):
            raise TypeError('config must be a BHPWaveConfig')
        self.config = config
        self._waveform, self._trajectory, self._constants, self._native, self._receipt = _load_backend(runtime_lock)
        self._data = self._trajectory.TrajectoryData()
        self._evolution = self._trajectory.InspiralGenerator(trajectory_data=self._data)
        self._generator = self._waveform.KerrCircularWaveform(trajectory_data=self._data, num_threads=1)
        if type(self._generator.waveform_generator) is not self._native.WaveformGeneratorPy:
            raise RuntimeError('selected generator is not the locked native CPU class')

    def generate(self, parameters):
        if not isinstance(parameters, BHPWaveParameters):
            raise TypeError('parameters must be BHPWaveParameters')
        p, c = parameters, self.config
        self._evolution.check_radius(p.a, p.r0)
        span = (c.n_samples-1)*c.dt_seconds
        native_count, native_dt, native_stride = _native_sampling(c.n_samples, c.dt_seconds, c.generation_dt_seconds)
        year_seconds = _real(self._constants.yr_MKS, 'BHPWave sidereal year')
        if year_seconds <= 0:
            raise ValueError('BHPWave sidereal year must be positive')
        duration = (span/year_seconds)*(1+64*np.finfo(float).eps)
        merger = _real(self._data.time_to_merger(p.primary_msun,p.secondary_msun,p.a,p.r0), 'time to merger')
        if merger <= duration*year_seconds:
            raise ValueError('physical evolution ends before the requested span')
        count = self._generator.waveform_generator.time_step_number(p.primary_msun,p.secondary_msun,p.a,p.r0,native_dt,duration)
        if count != native_count:
            raise ValueError('physical backend sample count differs from request')
        omega_max = _real(self._data.base_class.max_orbital_frequency(p.a), 'table orbital frequency limit')
        primary_seconds = p.primary_msun*self._constants.Modot_GC1_to_S
        bound = _real(omega_max/(np.pi*primary_seconds), 'carrier frequency bound')
        if bound <= 0 or bound >= 0.5/c.dt_seconds:
            raise ValueError('selected m=2 carrier bound reaches Nyquist')
        hp,hc = self._generator(p.primary_msun,p.secondary_msun,p.a,p.r0,p.distance_gpc,
                                p.qS,p.phiS,p.qK,p.phiK,p.Phi_phi0,
                                dt=native_dt,T=duration,pad_output=False,return_list=True,
                                select_modes=[(2,2)],include_negative_m=True)
        arrays = [np.asarray(value) for value in (hp,hc)]
        if any(value.shape != (native_count,) or value.dtype.kind not in 'fi'
               or not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError('backend must return finite real hp,hc at every requested sample')
        values = np.array(np.asarray(arrays)[:,::native_stride],dtype=float,copy=True)
        if values.shape != (2,c.n_samples):
            raise ValueError('native restriction does not cover the exact output grid')
        values.setflags(write=False)
        metadata = dict(self._receipt)
        metadata.update(model='KerrCircularWaveform', frame='SSB, BHPWave polarization basis',
                        complex_convention='real (hp, hc); response input hp + 1j*hc', parameters=p.to_dict(), config=c.to_dict(),
                        retained_positive_modes=[[2,2]],include_negative_m=True,
                        num_threads=1,pad_output=False,time_to_merger_seconds=merger,
                        native_n_samples=native_count,native_dt_seconds=native_dt,native_stride=native_stride,
                        sampling_policy='direct native sampling' if native_stride == 1 else 'explicit genuine native samples at an integer stride; no interpolation',
                        carrier_frequency_upper_bound_hz=bound,carrier_nyquist_ratio=2*c.dt_seconds*bound,
                        sample_span_seconds=span,requested_duration_years=duration,
                        scope='selected-pair CPU polarizations; no waveform-accuracy or response qualification')
        return BHPWavePolarizations(values,c.dt_seconds,c.epoch_start_seconds,c.epoch_reference,metadata)
