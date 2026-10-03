"""Prediction parity and rejection tests; no optional CUDA dependency at import."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, RBF

source = Path(__file__).resolve().parents[2] / "RIFT/interpolators/cached_matern_gp.py"
spec = importlib.util.spec_from_file_location("cached_matern_gp_test_subject", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def model():
    rng = np.random.default_rng(260103)
    x = rng.normal(size=(180, 4)) * np.array([0.2, 3, 1.4, 0.6]) + [100, -7, 0.3, 8]
    y = 43 + np.sin(x[:, 1]) + 0.3 * x[:, 2] ** 2
    kernel = ConstantKernel(1.8) * Matern([0.5, 1.1, 2.2, 0.7], nu=2.5) + WhiteKernel(0.02)
    fit = make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, alpha=0.01))
    fit.fit(x, y)
    return fit, x


def test_parity_and_partition_invariance(model):
    fit, x = model
    queries = np.concatenate([x[:30], x[:30] + [0.1, 0.4, -0.1, 0.3], x[:4] + [3, 30, 8, 4]])
    adapter = module.from_sklearn(fit, backend="numpy", batch_size=7)
    np.testing.assert_allclose(adapter.predict(queries), fit.predict(queries), atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(np.concatenate([adapter.predict(queries[:2]), adapter.predict(queries[2:])]), adapter.predict(queries), atol=1e-11, rtol=1e-12)
    assert adapter.predict(queries[:0]).shape == (0,)


def test_unscaled_gp_and_disabled_scaler_flags(model):
    fit, x = model
    gp = fit[-1]
    adapter = module.from_sklearn(gp, backend="numpy")
    transformed = fit[0].transform(x[:10])
    np.testing.assert_allclose(adapter.predict(transformed), gp.predict(transformed), atol=1e-10)
    for mean, std in [(False, True), (True, False), (False, False)]:
        variant = make_pipeline(StandardScaler(with_mean=mean, with_std=std), GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None, normalize_y=False))
        variant.fit(x, np.sin(x[:, 1]))
        np.testing.assert_allclose(module.from_sklearn(variant, backend="numpy").predict(x[:10]), variant.predict(x[:10]), atol=1e-8)


def test_reject_unsupported_options_and_shapes(model):
    fit, x = model
    adapter = module.from_sklearn(fit, backend="numpy")
    for q in [x[0], x[:2, :2], np.full((1, 4), np.nan)]:
        with pytest.raises(ValueError):
            adapter.predict(q)
    with pytest.raises(NotImplementedError):
        adapter.predict(x[:1], return_std=True)
    with pytest.raises(NotImplementedError):
        adapter.predict(x[:1], return_cov=True)
    with pytest.raises(ValueError):
        module.from_sklearn(fit, batch_size=0)
    unsupported = GaussianProcessRegressor(kernel=RBF(), optimizer=None).fit(x, np.ones(len(x)))
    with pytest.raises(ValueError):
        module.from_sklearn(unsupported)
    multi = GaussianProcessRegressor(kernel=fit[-1].kernel_, optimizer=None).fit(x, np.ones((len(x), 2)))
    with pytest.raises(ValueError, match="single-target"):
        module.from_sklearn(multi)


def test_lazy_cuda_backend(model):
    fit, x = model
    before = "cupy" in sys.modules
    adapter = module.from_sklearn(fit)
    assert adapter._device_state is None
    adapter.predict(x[:0])
    assert adapter._device_state is None
    assert ("cupy" in sys.modules) == before


def test_cuda_mean_parity_if_available(model):
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except (ImportError, RuntimeError):
        pytest.skip("CuPy/CUDA unavailable")
    fit, x = model
    adapter = module.from_sklearn(fit, batch_size=17)
    np.testing.assert_allclose(adapter.predict(x[:40]), fit.predict(x[:40]), atol=1e-7, rtol=1e-10)


def test_native_loaded_gp_hook(monkeypatch, model):
    """Exercise the actual CLI hook without requiring LAL or a CUDA allocation."""
    import ast
    from types import ModuleType, SimpleNamespace
    cli = source.parents[2] / "bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py"
    tree = ast.parse(cli.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_gp")
    called = []
    shim = ModuleType("RIFT.interpolators.cached_matern_gp")
    def selected_adapter(fit, *, backend, batch_size):
        called.append((backend, batch_size))
        return module.from_sklearn(fit, backend="numpy", batch_size=batch_size)
    shim.from_sklearn = selected_adapter
    package = ModuleType("RIFT"); package.__path__ = []
    interpolators = ModuleType("RIFT.interpolators"); interpolators.__path__ = []
    monkeypatch.setitem(sys.modules, "RIFT", package)
    monkeypatch.setitem(sys.modules, "RIFT.interpolators", interpolators)
    monkeypatch.setitem(sys.modules, "RIFT.interpolators.cached_matern_gp", shim)
    opts = SimpleNamespace(gp_predict_backend="cupy", fit_method="gp", fit_load_gp="model.pkl",
                           fit_uncertainty_added=False, protect_coordinate_conversions=False,
                           gp_predict_batch_size=7)
    fit, x = model
    namespace = {"opts": opts, "joblib": SimpleNamespace(load=lambda path: fit), "np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(cli), "exec"), namespace)
    predict = namespace["fit_gp"](x, np.zeros(len(x)))
    assert called == [("cupy", 7)]
    np.testing.assert_allclose(predict(x[:10]), fit.predict(x[:10]), atol=1e-10)
    opts.fit_load_gp = None
    with pytest.raises(ValueError, match="without refitting"):
        namespace["fit_gp"](x, np.zeros(len(x)))
    opts.fit_load_gp = "model.pkl"; opts.fit_uncertainty_added = True
    with pytest.raises(ValueError, match="means only"):
        namespace["fit_gp"](x, np.zeros(len(x)))


def test_large_offset_near_coincident_distance_parity():
    # The norm/dot identity loses close-point distances on large offsets.
    rng = np.random.default_rng(103)
    x = 1e5 + rng.normal(size=(40, 3))
    y = 1e3 * np.sin(x[:, 0] - 1e5)
    fit = GaussianProcessRegressor(
        kernel=ConstantKernel(2.0) * Matern([1.1, 2.2, 0.7], nu=2.5),
        alpha=1e-7, optimizer=None, normalize_y=True).fit(x, y)
    q = np.concatenate([x, x + 1e-6])
    adapter = module.from_sklearn(fit, backend="numpy", batch_size=9)
    np.testing.assert_allclose(adapter.predict(q), fit.predict(q), atol=1e-7, rtol=1e-10)
