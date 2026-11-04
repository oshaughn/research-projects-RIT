"""Bounded CPU response for SSB EMRI polarizations and a pinned equal-arm orbit.

The output is TDI fractional-frequency A/E/T. Noise covariance is supplied by
the likelihood caller; the response does not imply independent channel noise.
"""

from dataclasses import dataclass
import hashlib
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
import sys
from numbers import Integral, Real
from pathlib import Path

import numpy as np


RESPONSE_VERSION = "1.1.17"
ORBIT_PACKAGE_VERSION = "1.2.8"
ORBIT_GIT_BLOB = "c1f7f0579cfd6d82d2037fae85ed825110c9855c"


def _finite_scalar(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value)):
        raise ValueError("{} must be a finite real scalar".format(name))
    return float(value)


@dataclass(frozen=True)
class ResponseConfig:
    """Restricted response settings; orbit_file must already exist locally."""

    orbit_file: str
    trim_seconds: float = 10000.0
    interpolation_order: int = 25
    tdi_generation: str = "1st generation"

    def __post_init__(self):
        if not isinstance(self.orbit_file, str) or not Path(self.orbit_file).is_absolute():
            raise ValueError("orbit_file must be an absolute local file path")
        if _finite_scalar(self.trim_seconds, "trim_seconds") <= 0:
            raise ValueError("trim_seconds must be finite and positive")
        if (isinstance(self.interpolation_order, (bool, np.bool_))
                or not isinstance(self.interpolation_order, Integral)
                or self.interpolation_order != 25):
            raise ValueError("the initial response supports interpolation_order=25 only")
        if self.tdi_generation != "1st generation":
            raise ValueError("the initial response supports 1st generation TDI only")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("response configuration must be a dictionary")
        return cls(**values)


@dataclass(frozen=True)
class AETResponse:
    channels: np.ndarray
    dt: float
    epoch_seconds: float
    metadata: dict


def _verified_orbit(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("pre-stage the pinned equal-arm orbit file: {}".format(path))
    payload = path.read_bytes()
    header = ("blob %d\0" % len(payload)).encode("ascii")
    if hashlib.sha1(header + payload).hexdigest() != ORBIT_GIT_BLOB:
        raise ValueError("orbit_file does not match the pinned synthetic equal-arm asset")
    return hashlib.sha256(payload).hexdigest()


def _load_backend():
    try:
        versions = {
            "fastlisaresponse": metadata.version("fastlisaresponse"),
            "lisaanalysistools": metadata.version("lisaanalysistools"),
        }
    except metadata.PackageNotFoundError as exc:
        raise ImportError("install the pinned optional EMRI response dependencies") from exc
    if versions != {"fastlisaresponse": RESPONSE_VERSION,
                    "lisaanalysistools": ORBIT_PACKAGE_VERSION}:
        raise RuntimeError("unsupported EMRI response package versions: {}".format(versions))
    try:
        from fastlisaresponse import pyResponseTDI
        from lisatools.detector import Orbits
    except ImportError as exc:
        raise ImportError("the pinned CPU response dependencies could not import") from exc
    return pyResponseTDI, Orbits, versions



def _module_artifact(module):
    """Describe the loaded file, including an explicitly selected overlay."""
    origin = getattr(module, "__file__", None)
    if not origin:
        raise RuntimeError("loaded response provenance requires a module file")
    path = Path(origin).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"module": module.__name__, "path": str(path), "sha256": digest.hexdigest()}


def _loaded_backend_provenance(response, response_class, orbit_class):
    """Bind provenance to the native object actually constructed by FLR."""
    native = sys.modules.get("fastlisaresponse_backend_cpu.responselisa")
    if (native is None or not hasattr(native, "pyLISAResponseWrap")
            or type(response.cpp_response) is not native.pyLISAResponseWrap):
        raise RuntimeError("selected response object is not the loaded native CPU backend")
    artifact = _module_artifact(native)
    if not any(artifact["path"].endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise RuntimeError("selected CPU response module is not a native extension file")
    modules = {"python_wrapper": response_class.__module__,
               "lat_detector": orbit_class.__module__}
    result = {"native_cpu": artifact}
    for role, name in modules.items():
        module = sys.modules.get(name)
        if module is None:
            raise RuntimeError("defining response module is not loaded: " + name)
        result[role] = _module_artifact(module)
    return result


def equal_arm_aet(hp, hc, *, dt, sky_colatitude, sky_longitude, epoch=0.0,
                  config):
    """Return trimmed A/E/T from hp,hc sampled at the orbit's zero epoch.

    The caller generates real waveform padding on both sides of its retained
    segment. Padding is discarded here; it is never filled by extrapolation.
    Epoch is seconds relative to the pinned orbit's t=0, not a GPS timestamp.
    """
    if not isinstance(config, ResponseConfig):
        raise TypeError("config must be a ResponseConfig")
    dt = _finite_scalar(dt, "dt")
    epoch = _finite_scalar(epoch, "epoch")
    sky_colatitude = _finite_scalar(sky_colatitude, "sky_colatitude")
    sky_longitude = _finite_scalar(sky_longitude, "sky_longitude")
    if dt <= 0:
        raise ValueError("dt must be finite and positive")
    if epoch != 0:
        raise ValueError("only epoch=0 relative to the pinned orbit is supported")
    if not 0 <= sky_colatitude <= np.pi:
        raise ValueError("sky_colatitude must be in [0, pi]")
    if not 0 <= sky_longitude < 2 * np.pi:
        raise ValueError("sky_longitude must be in [0, 2*pi)")
    hp, hc = np.asarray(hp), np.asarray(hc)
    if (hp.ndim != 1 or hp.shape != hc.shape or np.iscomplexobj(hp)
            or np.iscomplexobj(hc) or not np.all(np.isfinite(hp))
            or not np.all(np.isfinite(hc))):
        raise ValueError("hp and hc must be equal-length finite real 1D arrays")
    ratio = config.trim_seconds / dt
    trim = int(round(ratio))
    if trim < 1 or not np.isclose(ratio, trim, rtol=0, atol=1e-9):
        raise ValueError("trim_seconds must be a positive integer multiple of dt")
    count = len(hp)
    if count <= 2 * trim:
        raise ValueError("waveform must exceed both response padding intervals")
    orbit_sha256 = _verified_orbit(config.orbit_file)
    response_class, orbit_class, versions = _load_backend()
    # Orbits accepts an existing absolute path; its missing-file fallback downloads.
    orbits = orbit_class(filename=str(Path(config.orbit_file)), force_backend="cpu")
    if count * dt > float(orbits.t_base.max()):
        raise ValueError("waveform extends beyond the pinned orbit time coverage")
    response = response_class(
        sampling_frequency=1.0 / dt, num_pts=count,
        order=config.interpolation_order, tdi=config.tdi_generation,
        orbits=orbits, tdi_chan="AET", force_backend="cpu",
    )
    backend_artifacts = _loaded_backend_provenance(response, response_class, orbit_class)
    if response.num_pts != count:
        raise RuntimeError("the response changed the requested sample count")
    # pyResponseTDI consumes hp + i*hc; FEW's complex convention is hp - i*hc.
    response.get_projections(
        np.asarray(hp, dtype=np.float64) + 1j * np.asarray(hc, dtype=np.float64),
        float(sky_longitude), float(np.pi / 2 - sky_colatitude),
        t0=trim * dt,
    )
    if response.tdi_start_ind != trim:
        raise RuntimeError("the response changed the requested trim index")
    full = np.asarray(response.get_tdi_delays(), dtype=np.float64)
    if full.shape != (3, count) or not np.all(np.isfinite(full)):
        raise RuntimeError("the response returned invalid A/E/T samples")
    channels = full[:, trim:-trim].copy()
    channels.setflags(write=False)
    info = {
        **versions, "backend": "cpu", "backend_artifacts": backend_artifacts, "channels": ["A", "E", "T"],
        "observable": "TDI fractional-frequency", "orbit_git_blob": ORBIT_GIT_BLOB,
        "orbit_sha256": orbit_sha256, "input_epoch_seconds": float(epoch),
        "epoch_reference": "pinned equal-arm orbit t=0",
        "input_samples": count, "trim_samples_each_end": trim,
        "trim_seconds_each_end": trim * dt, "output_samples": channels.shape[1],
        "tdi_generation": config.tdi_generation,
        "interpolation_order": config.interpolation_order,
        "input_polarization_convention": "hp,hc; backend receives hp+i*hc",
    }
    return AETResponse(channels, float(dt), float(epoch + trim * dt), info)
