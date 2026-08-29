"""Regression tests for adaptive terminal time marginalization and t_ref export."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from RIFT.likelihood.jax_ile import anglemarg, core, wrapper


@pytest.mark.parametrize("n", [31, 32])
def test_literal_reflection_reproduces_odd_and_even_samples(n):
    rng = np.random.default_rng(10 + n)
    x = rng.normal(size=(2, n))
    for factor in (2, 4, 8):
        dense = np.asarray(core._reflected_fft_upsample(jnp.asarray(x), factor))
        assert dense.shape[-1] == (n - 1) * factor + 1
        np.testing.assert_allclose(dense[..., ::factor], x, atol=2e-12, rtol=0)


@pytest.mark.parametrize("n", [31, 32])
def test_constant_integrand_has_exact_interval_normalization(n):
    dt, c = 1.0 / 8192, 17.0
    row = jnp.full((2, n), c)
    exact = c + np.log((n - 1) * dt)
    for factor in (1, 2, 8, 32):
        got, resolved = core._terminal_reflected_fft_at_factor(row, dt, factor)
        np.testing.assert_allclose(np.asarray(got), exact, atol=2e-12, rtol=0)
        assert np.all(np.asarray(resolved))


def _event_b_like_row(phase=0.37):
    n, dt, amp = 491, 1.0 / 8192, 600.0 ** 2 / 2.0
    x = np.arange(n) - n // 2 - phase
    # Broad, reconstructible terminal lnL; exp(lnL) is narrower than one input
    # sample by sqrt(amp), which is the Event-B high-SNR failure geometry.
    return amp * np.exp(-0.5 * (x / 3.0) ** 2), dt, amp


def test_event_b_scale_adaptive_integral_matches_local_dense_truth():
    row, dt, amp = _event_b_like_row()
    w = jnp.asarray(core._simpson_weights(row.size, dt))
    got = float(core._time_marginalize_reflected_fft(
        jnp.asarray(row[None, :]), dt, w)[0])
    sigma_t = 3.0 * dt / np.sqrt(amp)
    want = amp + np.log(np.sqrt(2.0 * np.pi) * sigma_t)
    assert abs(got - want) < 2e-3


def test_nonfinite_row_falls_back_to_historical_simpson():
    n, dt = 31, 1.0 / 4096
    row = np.linspace(-4.0, 0.0, n)[None, :]
    row[0, 3] = -np.inf
    w = jnp.asarray(core._simpson_weights(n, dt))
    got = core._time_marginalize_reflected_fft(jnp.asarray(row), dt, w)
    want = core._time_marginalize(jnp.asarray(row), w)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=0, atol=0)


def test_fixed_factor_value_gradient_and_hessian_are_finite():
    n, dt = 25, 1.0 / 4096
    x = jnp.arange(n, dtype=jnp.float64) - 12.2

    def f(scale):
        row = (scale * jnp.exp(-0.5 * (x / 3.0) ** 2))[None, :]
        return core._terminal_reflected_fft_at_factor(row, dt, 16)[0][0]

    value = f(600.0)
    grad = jax.grad(f)(600.0)
    hess = jax.hessian(f)(600.0)
    assert np.all(np.isfinite(np.asarray([value, grad, hess])))


class _TimeData:
    def __init__(self, n, dt):
        self.deltaT = dt
        self.tvals = jnp.asarray((np.arange(n) - n // 2) * dt)


def test_t_ref_draws_use_converged_fine_grid_and_are_deterministic():
    row, dt, _ = _event_b_like_row(phase=0.41)
    rows = np.repeat(row[None, :], 16, axis=0)
    data = _TimeData(row.size, dt)
    a, fa = wrapper.sample_time_offsets(
        data, rows, "bandlimited", rng=np.random.default_rng(1234))
    b, fb = wrapper.sample_time_offsets(
        data, rows, "bandlimited", rng=np.random.default_rng(1234))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(fa, fb)
    assert np.min(fa) > 1
    # The posterior is substantially narrower than one input sample and the
    # draws are not quantized to the input grid.
    assert np.std(a) < 0.2 * dt
    coarse_phase = np.mod((a - float(data.tvals[0])) / dt, 1.0)
    assert np.any(np.minimum(coarse_phase, 1.0 - coarse_phase) > 1e-6)


def test_all_terminal_kernels_expose_one_canonical_selector():
    kernels = [
        core.fused_log_likelihood,
        core.fused_log_likelihood_distmarg,
        core.fused_log_likelihood_phimarg,
        core.fused_log_likelihood_distphimarg,
        core.fused_log_likelihood_distphipsimarg,
        core.fused_log_likelihood_distpsimarg,
        anglemarg.fused_log_likelihood_distphipsimarg_exact,
        anglemarg.fused_log_likelihood_distphipsimarg_laplace,
        core.phi_ref_conditional_lnL,
    ]
    for fn in kernels:
        assert "time_quadrature" in inspect.signature(fn).parameters, fn.__name__


def test_all_wrappers_expose_quadrature_and_conditional_time():
    classes = [wrapper.JAXExtrinsicLikelihood,
               wrapper.JAXDistanceMarginalizedLikelihood,
               wrapper.JAXDistPhiMargLikelihood,
               wrapper.JAXDistPhiPsiMargLikelihood,
               wrapper.JAXDistPsiMargLikelihood]
    for cls in classes:
        assert "time_quadrature" in inspect.signature(cls.__init__).parameters
        assert hasattr(cls, "conditional_time_lnL")


def test_jax_driver_uses_conventional_ile_flag_names():
    import pathlib
    driver = pathlib.Path(__file__).parents[2] / "bin" / "integrate_likelihood_extrinsic_jax"
    src = driver.read_text()
    for flag in ("--time-marginalization-quadrature",
                 "--resample-time-marginalization",
                 "--srate-resample-time-marginalization",
                 "--interpolate-time"):
        assert flag in src


def _load_driver():
    import importlib.machinery
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parents[2] / "bin" / "integrate_likelihood_extrinsic_jax"
    loader = importlib.machinery.SourceFileLoader("_jax_tmarg_driver", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_driver_parses_readback_and_conflict_checks_ile_aliases():
    drv = _load_driver()
    parser = drv.build_parser()
    argv = ["--time-marginalization-quadrature", "bandlimited",
            "--interpolate-time", "sinc", "--resample-time-marginalization",
            "--srate-resample-time-marginalization", "65536", "--save-samples"]
    opts, _ = parser.parse_args(argv)
    drv.record_supplied_options(opts, argv, parser)
    drv.resolve_ile_interface_aliases(opts, parser)
    drv.check_critical_and_report(opts, parser)
    assert opts.time_marginalization_quadrature == "bandlimited"
    assert opts.interp == "sinc"
    assert opts.srate_resample_time_marginalization == 65536

    argv = ["--interp", "linear", "--interpolate-time", "sinc"]
    opts, _ = parser.parse_args(argv)
    drv.record_supplied_options(opts, argv, parser)
    with pytest.raises(SystemExit):
        drv.resolve_ile_interface_aliases(opts, parser)


def test_driver_exports_gps_t_ref_on_refined_grid(tmp_path):
    import types
    drv = _load_driver()
    n, dt = 31, 1.0 / 8192
    x = np.arange(n) - n // 2 - 0.37
    row = 5000.0 * np.exp(-0.5 * (x / 3.0) ** 2)
    data = _TimeData(n, dt)

    class Like:
        time_quadrature = "bandlimited"
        def __init__(self):
            self.data = data
        def conditional_time_lnL(self, *args):
            return np.repeat(row[None, :], len(args[0]), axis=0)

    opts = types.SimpleNamespace(
        output_file=str(tmp_path / "ile"), save_samples=True, seed=91,
        mode="nuts", resample_time_marginalization=True,
        srate_resample_time_marginalization=None)
    theta = np.zeros((8, 6))
    drv.write_samples(opts, 0, theta, np.zeros(8), True, like=Like(),
                      fiducial_epoch=1000000000.25)
    path = tmp_path / "ile_0_samples.dat"
    header = path.read_text().splitlines()
    assert "t_ref" in header[0]
    assert "time_grid_factor=" in header[1]
    values = np.loadtxt(path)
    t_ref = values[:, -2]
    assert np.all(np.abs(t_ref - 1000000000.25) < n * dt)
    fine_phase = np.mod((t_ref - 1000000000.25 - float(data.tvals[0])) / dt, 1.0)
    assert np.any(np.minimum(fine_phase, 1.0 - fine_phase) > 1e-5)
