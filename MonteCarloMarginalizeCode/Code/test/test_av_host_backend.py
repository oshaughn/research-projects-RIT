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


def test_default_av_keeps_native_backend():
    for method in ('rf', 'gp', 'gp-matern'):
        assert not mod.needs_host_av(method, 'sklearn', 'auto')
    assert not mod.needs_host_av('gp-torch', 'sklearn', 'cpu')
    assert mod.needs_host_av('gp-matern', 'cupy', 'auto')
    assert mod.needs_host_av('gp', 'cupy', 'auto')
    for method in ('rf', 'quadratic', 'gp_lazy'):
        assert not mod.needs_host_av(method, 'cupy', 'auto')
    assert mod.needs_host_av('gp-torch', 'sklearn', 'cuda:0')


def test_cip_calls_configure_host_av_only_under_gate():
    src = (Path(__file__).parents[1]/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py').read_text().splitlines()
    calls = [i for i, line in enumerate(src) if 'configure_host_av(' in line and 'import' not in line]
    assert calls
    for i in calls:
        assert src[i-1].strip().startswith('if needs_host_av('), src[i-1]


def test_cip_rejects_cupy_backend_for_other_fit_methods():
    import ast
    from types import SimpleNamespace
    import pytest
    path = Path(__file__).parents[1]/'bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.If) and 'gp_predict_backend' in ast.unparse(n.test))
    program = compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec')
    class Parser:
        def error(self, message): raise SystemExit(message)
    def run(method, load=None, backend='cupy'):
        exec(program, dict(opts=SimpleNamespace(gp_predict_backend=backend, fit_method=method, fit_load_gp=load), parser=Parser()))
    for method in ('rf', 'quadratic', 'gp-torch', 'gp_lazy'):
        with pytest.raises(SystemExit):
            run(method)
    with pytest.raises(SystemExit):
        run('gp')
    run('gp', load='fit.pkl'); run('gp-matern'); run('rf', backend='sklearn')
