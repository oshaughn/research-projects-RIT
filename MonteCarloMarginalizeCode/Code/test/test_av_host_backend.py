import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import numpy as np
spec=importlib.util.spec_from_file_location('backend',Path(__file__).parents[1]/'RIFT/misc/av_backend.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_late_discovery_host_prior_cycles_and_captured_defaults():
    fake=SimpleNamespace(__name__='cupy')
    def forbidden(x):raise AssertionError('host sampling used device converter')
    av=ModuleType('native_av')
    namespace={'__name__':av.__name__,'xpy_default':fake,'identity_convert_togpu':forbidden,'cupy_ok':True}
    exec('def draw(values, xpy=xpy_default):\n return identity_convert_togpu(xpy.asarray(values))\nclass MCSampler:\n def prior(self, values, *, xpy=xpy_default):\n  return xpy.asarray(values)*2\n',namespace)
    av.draw=namespace['draw'];av.MCSampler=namespace['MCSampler'];av.xpy_default=fake
    # Simulates plugin functions retained from a distinct globals dictionary.
    mod.configure_host_av(av)
    for _ in range(2):
        x=av.draw([1.,2.]);assert isinstance(x,np.ndarray)
        np.testing.assert_equal(av.MCSampler().prior(x),[2.,4.])
    assert not namespace['cupy_ok']
    assert av.identity_convert_togpu(x) is x
    mod.configure_host_av(av)
    np.testing.assert_equal(av.draw([1.,2.]),[1.,2.])
