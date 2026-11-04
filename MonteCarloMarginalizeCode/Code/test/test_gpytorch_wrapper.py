"""Scientific regression tests for the optional native gp-torch backend.

Load this isolated module without importing the LAL-dependent RIFT package. Run
with pytest; the module skips when the optional torch/gpytorch stack is absent.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")
CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rift_gpytorch_under_test", CODE / "RIFT/interpolators/gpytorch_wrapper.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
Interpolator = backend.Interpolator


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def data():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(28, 2))*[2.0, .2]+[10.0, -3.0]
    y = 7.0+np.sin(x[:, 0])+2*x[:, 1]
    e = np.linspace(.02, .15, len(y))
    return x, y, e


def test_float64_stored_scaler_deterministic_batch_invariant_without_mutation(data):
    x, y, e = data
    originals = [a.copy() for a in data]
    default_dtype = torch.get_default_dtype()
    x.setflags(write=False)
    model = Interpolator(x, y[:, None], y_errors=e[:, None], epochs=4, device="cpu")
    model.train()
    query = np.array([[10.5, -2.9], [11.2, -3.1], [8.1, -2.8]])
    q_before = query.copy()
    q = model.evaluate(query)
    assert q.dtype == np.float64
    assert torch.get_default_dtype() == default_dtype
    assert q.shape == (3,)
    assert model.input_train.dtype == model.target_train.dtype == torch.float64
    assert all(p.dtype == torch.float64 for p in model.model.parameters())
    np.testing.assert_array_equal(q, model.evaluate(query))
    np.testing.assert_allclose(q, model.evaluate(query, batch_size=1), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(q[[2, 0, 1]], model.evaluate(query[[2, 0, 1]]), rtol=1e-11, atol=1e-11)
    with_far_points = np.vstack([query[0], [1e5, -1e5]])
    np.testing.assert_allclose(model.evaluate(query[:1])[0], model.evaluate(with_far_points)[0], rtol=1e-11, atol=1e-11)
    for current, before in zip(data, originals):
        np.testing.assert_array_equal(current, before)
    np.testing.assert_array_equal(query, q_before)
    assert model.evaluate(np.empty((0, 2))).shape == (0,)


def test_exact_mean_matches_independent_dense_matern_formula(data):
    x, y, e = data
    model = Interpolator(x, y, y_errors=e, epochs=3, device="cpu").train()
    q = np.vstack([x[:3], x.mean(axis=0), x[:3]+[.12, -.01]])
    tx = (x-model.input_mu)/model.input_sigma
    tq = (q-model.input_mu)/model.input_sigma
    ls = model.model.covar_module.base_kernel.lengthscale.detach().numpy().reshape(-1)
    amplitude = float(model.model.covar_module.outputscale.detach())
    learned_noise = float(model.likelihood.second_noise.detach())

    def kernel(a, b):
        r = np.linalg.norm(a[:, None, :]/ls-b[None, :, :]/ls, axis=2)
        return amplitude*(1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)

    fixed = np.maximum((e/model.target_sigma)**2, backend.FIXED_NOISE_FLOOR)
    observed = kernel(tx, tx)+np.diag(fixed+learned_noise)
    alpha = np.linalg.solve(observed, (y-model.target_mu)/model.target_sigma)
    expected = kernel(tq, tx)@alpha*model.target_sigma+model.target_mu
    np.testing.assert_allclose(model.evaluate(q), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(model.fixed_noise.numpy(), fixed, rtol=1e-13)


def test_reported_errors_are_variances_in_normalized_units_and_downweight_outlier():
    x = np.array([[-1.0], [0.0], [1.0]])
    y = np.array([0.0, 10.0, 0.0])
    low = Interpolator(x, y, epochs=0, y_errors=[.01, .01, .01], device="cpu")
    high = Interpolator(x, y, epochs=0, y_errors=[.01, 100., .01], device="cpu")
    np.testing.assert_allclose(high.fixed_noise.numpy(), np.array([.01, 100., .01])**2/y.std()**2)
    assert high.evaluate(x[1:2])[0] < low.evaluate(x[1:2])[0]-5
    assert high.likelihood.second_noise.requires_grad
    assert all(p.dtype == torch.float64 for p in high.likelihood.parameters())


def test_predict_only_cross_kernel_blocks_and_training_invalidates_cache(data, monkeypatch):
    x, y, e = data
    model = Interpolator(x, y, y_errors=e, epochs=1, device="cpu")
    model.evaluate(x[:1])
    assert model._mean_weights is not None
    model.train()
    assert model._mean_weights is None
    model.evaluate(x[:1])
    observed = []
    forward = torch.cdist

    def record(a, b, **kwargs):
        observed.append((len(a), len(b)))
        return forward(a, b, **kwargs)

    monkeypatch.setattr(torch, "cdist", record)
    model.evaluate(np.tile(x, (3, 1)), batch_size=7)
    assert observed
    assert all(a <= 7 and b == len(x) for a, b in observed)


def test_save_load_preserves_scalers_noise_mean_and_coordinate_order(data, tmp_path):
    x, y, e = data
    model = Interpolator(x, y, y_errors=e, epochs=2, device="cpu",
                         feature_names=["mu1", "s1x"], provenance={"lnL_shift": 40.0}).train()
    path = tmp_path / "fit.pt"
    model.save(path)
    restored = Interpolator.load(path, device="cpu", expected_feature_names=["mu1", "s1x"])
    np.testing.assert_allclose(restored.evaluate(x), model.evaluate(x), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(restored.fixed_noise.numpy(), model.fixed_noise.numpy())
    np.testing.assert_array_equal(restored.input_mu, model.input_mu)
    assert restored.provenance["lnL_shift"] == 40
    assert restored.loss_history == model.loss_history
    with pytest.raises(ValueError, match="feature names/order"):
        Interpolator.load(path, expected_feature_names=["s1x", "mu1"])
    with pytest.raises(ValueError, match="exceeding the explicit limit"):
        Interpolator.load(path, max_train_points=10)


def test_constant_coordinates_targets_and_invalid_inputs():
    x = np.column_stack([np.arange(6.), np.ones(6)])
    model = Interpolator(x, np.full(6, 3.), y_errors=np.zeros(6), epochs=1, device="cpu").train()
    np.testing.assert_allclose(model.evaluate(x), 3.)
    assert np.all(model.input_sigma > 0)
    assert model.target_sigma == 1.
    with pytest.raises(ValueError, match="exceeding the explicit limit"):
        Interpolator(x, np.ones(6), max_train_points=5)
    with pytest.raises(ValueError, match="nonnegative"):
        Interpolator(x, np.ones(6), y_errors=-np.ones(6))
    with pytest.raises(ValueError, match="finite"):
        Interpolator(x, np.array([np.nan]*6))
    with pytest.raises(ValueError, match="shape"):
        Interpolator(x, np.ones((6, 2)))
    with pytest.raises(ValueError, match="feature count"):
        model.evaluate(np.ones((1, 3)))
    with pytest.raises(ValueError, match="batch_size"):
        model.evaluate(x, batch_size=0)
    with pytest.raises(ValueError, match="Only CPU and CUDA"):
        Interpolator(x, np.ones(6), device="mps")


def load_native_hook(opts, shift):
    """Execute the actual CLI fit function without initializing LAL/integration."""
    tree = ast.parse((CODE / "bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit_gpytorch")
    scope = {"np": np, "gpytorch_ok": True, "gpytorch_wrapper": backend,
             "opts": opts, "coord_names": ["mu1", "s1x"], "lnL_shift": shift}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "native-gp-torch-hook", "exec"), scope)
    return scope["fit_gpytorch"]


def test_native_cli_hook_save_load_avoids_refit_and_preserves_lnl_shift(data, tmp_path, monkeypatch):
    x, raw_y, e = data
    opts = SimpleNamespace(fit_load_gp=None, fit_save_gp=str(tmp_path/"native"),
        gp_torch_device="cpu", gp_torch_epochs=2, gp_torch_batch_size=5,
        gp_torch_max_train_points=50, fit_uncertainty_added=False, protect_coordinate_conversions=False)
    first = load_native_hook(opts, 10.0)(x, raw_y-10.0, y_errors=e)
    saved_prediction = first(x)
    assert (tmp_path/"native.pt").exists()
    opts.fit_load_gp = str(tmp_path/"native.pt")
    opts.fit_save_gp = None
    monkeypatch.setattr(Interpolator, "train", lambda self: pytest.fail("Loading must not refit"))
    loaded = load_native_hook(opts, 23.0)(x, raw_y-23.0, y_errors=e)
    np.testing.assert_allclose(loaded(x), saved_prediction-13.0, rtol=1e-12, atol=1e-12)
    opts.fit_uncertainty_added = True
    with pytest.raises(ValueError, match="predictive mean"):
        load_native_hook(opts, 23.0)(x, raw_y-23.0, y_errors=e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this host")
def test_cuda_float64_mean_and_portable_cpu_reload(data, tmp_path):
    x, y, e = data
    cpu = Interpolator(x, y, y_errors=e, epochs=2, device="cpu").train()
    gpu = Interpolator(x, y, y_errors=e, epochs=2, device="cuda").train()
    assert gpu.input_train.device.type == "cuda"
    assert gpu.input_train.dtype == torch.float64
    np.testing.assert_allclose(gpu.evaluate(x, batch_size=3), cpu.evaluate(x), rtol=1e-9, atol=1e-9)
    gpu.save(tmp_path/"cuda.pt")
    loaded = Interpolator.load(tmp_path/"cuda.pt", device="cpu")
    np.testing.assert_allclose(loaded.evaluate(x), gpu.evaluate(x), rtol=1e-9, atol=1e-9)
