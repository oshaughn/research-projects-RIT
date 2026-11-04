"""Response wiring guards and an opt-in pinned CPU backend comparison."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest


_path = Path(__file__).parents[1] / "RIFT/LISA/emri_response.py"
_spec = importlib.util.spec_from_file_location("emri_response_under_test", _path)
response = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = response
_spec.loader.exec_module(response)


def test_polarization_sign_angles_trim_epoch_and_backend(monkeypatch, tmp_path):
    seen = {}

    class Orbit:
        def __init__(self, **kwargs):
            seen["orbit"] = kwargs
            self.t_base = np.array([0.0, 1000000.0])

    class Backend:
        def __init__(self, **kwargs):
            seen["response"] = kwargs
            self.num_pts = kwargs["num_pts"]

        def get_projections(self, data, longitude, latitude, t0):
            seen["projection"] = (data, longitude, latitude, t0)
            self.tdi_start_ind = int(t0 / 10)

        def get_tdi_delays(self):
            return np.arange(3 * self.num_pts).reshape(3, self.num_pts)

    artifacts = {"native_cpu": {"path": "/test/native.so", "sha256": "recorded"}}
    monkeypatch.setattr(response, "_loaded_backend_provenance", lambda *args: artifacts)
    monkeypatch.setattr(response, "_verified_orbit", lambda path: "sha256")
    monkeypatch.setattr(response, "_load_backend", lambda: (Backend, Orbit, {}))
    config = response.ResponseConfig(str(tmp_path / "orbit.h5"), trim_seconds=20)
    hp, hc = np.arange(12.), np.arange(12.) + 100
    output = response.equal_arm_aet(hp, hc, dt=10, sky_colatitude=1,
                                   sky_longitude=2, config=config)
    np.testing.assert_array_equal(seen["projection"][0], hp + 1j * hc)
    assert seen["projection"][1:] == (2, np.pi / 2 - 1, 20)
    assert seen["response"]["force_backend"] == "cpu"
    assert seen["response"]["tdi_chan"] == "AET"
    np.testing.assert_array_equal(output.channels, np.arange(36).reshape(3, 12)[:, 2:-2])
    assert output.epoch_seconds == 20
    assert output.dt == 10
    assert output.metadata["backend_artifacts"] == artifacts
    assert output.metadata["input_samples"] == 12
    assert output.metadata["output_samples"] == 8
    assert not output.channels.flags.writeable


@pytest.mark.parametrize("changes", [
    {"epoch": 1}, {"epoch": np.nan}, {"dt": 0}, {"dt": np.inf},
    {"sky_colatitude": -1}, {"sky_colatitude": np.nan},
    {"sky_longitude": 2*np.pi}, {"sky_longitude": np.nan},
    {"dt": True}, {"dt": "10"}, {"dt": [10]}, {"dt": 1j},
    {"epoch": False}, {"sky_colatitude": [1]}, {"sky_longitude": "2"},
])
def test_invalid_coordinates_rejected_before_dependency_load(changes, monkeypatch, tmp_path):
    monkeypatch.setattr(response, "_load_backend", lambda: pytest.fail("backend loaded"))
    args = dict(dt=10, sky_colatitude=1, sky_longitude=2, epoch=0,
                config=response.ResponseConfig(str(tmp_path / "orbit.h5")))
    args.update(changes)
    with pytest.raises(ValueError):
        response.equal_arm_aet(np.ones(4096), np.ones(4096), **args)


@pytest.mark.parametrize("hp,hc", [
    (np.ones(10), np.ones(11)), (np.ones((1, 10)), np.ones((1, 10))),
    (np.ones(10, complex), np.ones(10)), (np.full(10, np.nan), np.ones(10)),
    (np.ones(10), np.full(10, np.inf)),
])
def test_invalid_polarizations(hp, hc, tmp_path):
    with pytest.raises(ValueError):
        response.equal_arm_aet(hp, hc, dt=10, sky_colatitude=1,
                              sky_longitude=2,
                              config=response.ResponseConfig(str(tmp_path / "orbit.h5")))


def test_orbit_missing_and_wrong_hash(tmp_path):
    path = tmp_path / "orbit.h5"
    with pytest.raises(FileNotFoundError):
        response._verified_orbit(path)
    path.write_bytes(b"not the pinned orbit")
    with pytest.raises(ValueError, match="pinned"):
        response._verified_orbit(path)


def test_no_epoch_or_padding_fallback(tmp_path):
    path = str(tmp_path / "orbit.h5")
    with pytest.raises(ValueError, match="integer multiple"):
        response.equal_arm_aet(np.ones(4096), np.ones(4096), dt=30,
                              sky_colatitude=1, sky_longitude=2,
                              config=response.ResponseConfig(path))
    with pytest.raises(ValueError, match="padding"):
        response.equal_arm_aet(np.ones(2000), np.ones(2000), dt=10,
                              sky_colatitude=1, sky_longitude=2,
                              config=response.ResponseConfig(path))


def test_optional_dependency_version_is_checked(monkeypatch):
    monkeypatch.setattr(response.metadata, "version", lambda name: "wrong")
    with pytest.raises(RuntimeError, match="unsupported"):
        response._load_backend()


@pytest.mark.parametrize("setting", [
    {"trim_seconds": True}, {"trim_seconds": "10000"},
    {"trim_seconds": [10000]}, {"interpolation_order": 25.0},
    {"interpolation_order": True}, {"tdi_generation": "2nd generation"},
])
def test_invalid_configuration(setting, tmp_path):
    with pytest.raises(ValueError):
        response.ResponseConfig(str(tmp_path / "orbit.h5"), **setting)


@pytest.mark.skipif(not os.environ.get("RIFT_EMRI_ORBIT_FILE"),
                    reason="requires explicitly staged pinned CPU runtime/orbit")
def test_pinned_cpu_response_matches_wrapper_and_xyz_transform():
    from fastlisaresponse import ResponseWrapper, pyResponseTDI
    from fastlisaresponse.response import YRSID_SI
    from lisatools.detector import Orbits

    path = os.environ["RIFT_EMRI_ORBIT_FILE"]
    count, dt, trim = 4096, 10.0, 1000
    t = np.arange(count) * dt
    hp = 1e-20 * np.cos(2*np.pi*0.003*t)
    hc = 0.7e-20 * np.sin(2*np.pi*0.003*t + 0.4)
    config = response.ResponseConfig(path)
    actual = response.equal_arm_aet(hp, hc, dt=dt, sky_colatitude=1,
                                   sky_longitude=2, config=config)
    orbits = Orbits(filename=path, force_backend="cpu")
    wrapper = ResponseWrapper(
        lambda *args, **kwargs: hp - 1j*hc,
        count*dt/YRSID_SI, dt, index_lambda=1, index_beta=0,
        t0=trim*dt, flip_hx=True, is_ecliptic_latitude=False,
        remove_garbage=True, n_overide=count, orbits=orbits,
        force_backend="cpu", tdi_chan="AET", order=25,
        tdi="1st generation",
    )
    expected = np.asarray(wrapper(1, 2))
    np.testing.assert_allclose(actual.channels, expected, rtol=2e-12, atol=1e-35)
    assert np.linalg.norm(actual.channels) > 0
    xyz_response = pyResponseTDI(1/dt, count, orbits=orbits, force_backend="cpu",
                                 tdi_chan="XYZ", order=25, tdi="1st generation")
    xyz_response.get_projections(hp + 1j*hc, 2, np.pi/2-1, t0=trim*dt)
    x, y, z = np.asarray(xyz_response.get_tdi_delays())[:, trim:-trim]
    manual_aet = np.array([(z-x)/np.sqrt(2), (x-2*y+z)/np.sqrt(6),
                           (x+y+z)/np.sqrt(3)])
    np.testing.assert_allclose(actual.channels, manual_aet, rtol=2e-12, atol=1e-35)
    # A cross-polarization sign defect must change the actual channels.
    flipped = response.equal_arm_aet(hp, -hc, dt=dt, sky_colatitude=1,
                                     sky_longitude=2, config=config)
    assert np.linalg.norm(flipped.channels-actual.channels) > 0.01*np.linalg.norm(actual.channels)


def test_actual_loaded_module_paths_and_hashes_not_distribution_guess(monkeypatch,tmp_path):
    import hashlib
    from types import ModuleType,SimpleNamespace
    cpu=ModuleType('fastlisaresponse_backend_cpu.responselisa')
    wrapper=ModuleType('test_loaded_flr_wrapper')
    detector=ModuleType('test_loaded_lat_detector')
    class Native:pass
    class Wrapper:pass
    class Orbit:pass
    cpu.pyLISAResponseWrap=Native
    Wrapper.__module__=wrapper.__name__;Orbit.__module__=detector.__name__
    for module,filename,payload in [(cpu,'overlay'+response.EXTENSION_SUFFIXES[0],b'loaded native overlay'),
            (wrapper,'wrapper.py',b'python wrapper bytes'),(detector,'detector.py',b'LAT detector bytes')]:
        path=tmp_path/filename;path.write_bytes(payload);module.__file__=str(path)
        monkeypatch.setitem(sys.modules,module.__name__,module)
    obj=SimpleNamespace(cpp_response=Native())
    artifacts=response._loaded_backend_provenance(obj,Wrapper,Orbit)
    for role,module in [('native_cpu',cpu),('python_wrapper',wrapper),('lat_detector',detector)]:
        assert artifacts[role]=={'module':module.__name__,'path':str(Path(module.__file__).resolve()),
            'sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    # Same distribution/version cannot conceal a different selected extension.
    Path(cpu.__file__).write_bytes(b'different loaded overlay')
    changed=response._loaded_backend_provenance(obj,Wrapper,Orbit)
    assert changed['native_cpu']['sha256']!=artifacts['native_cpu']['sha256']
    assert changed['python_wrapper']==artifacts['python_wrapper']
    with pytest.raises(RuntimeError,match='native CPU backend'):
        response._loaded_backend_provenance(SimpleNamespace(cpp_response=object()),Wrapper,Orbit)
    monkeypatch.delitem(sys.modules,cpu.__name__)
    with pytest.raises(RuntimeError,match='native CPU backend'):
        response._loaded_backend_provenance(obj,Wrapper,Orbit)


def test_provenance_refuses_unavailable_or_non_native_origins(monkeypatch,tmp_path):
    from types import ModuleType,SimpleNamespace
    cpu=ModuleType('fastlisaresponse_backend_cpu.responselisa')
    class Native:pass
    cpu.pyLISAResponseWrap=Native
    cpu.__file__=str(tmp_path/'pretend.py');Path(cpu.__file__).write_bytes(b'not native')
    monkeypatch.setitem(sys.modules,cpu.__name__,cpu)
    with pytest.raises(RuntimeError,match='native extension'):
        response._loaded_backend_provenance(SimpleNamespace(cpp_response=Native()),Native,Native)
    with pytest.raises(RuntimeError,match='module file'):
        response._module_artifact(ModuleType('no_origin'))
    cpu.__file__=str(tmp_path/('missing'+response.EXTENSION_SUFFIXES[0]))
    with pytest.raises(FileNotFoundError):response._module_artifact(cpu)
