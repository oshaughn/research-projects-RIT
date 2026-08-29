"""Regression tests for adaptive, primitive-field time marginalization."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from RIFT.likelihood.jax_ile import anglemarg, core, wrapper


@pytest.mark.parametrize("n", [31, 32])
def test_even_extension_reproduces_odd_and_even_samples(n):
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
    got = float(core._time_marginalize_reflected_primitive(
        jnp.asarray(row[None, :], dtype=jnp.complex128),
        jnp.zeros((1, row.size)), dt)[0])
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


def test_adaptive_primitive_value_gradient_and_hessian_are_finite_and_rematerialized():
    n, dt = 25, 1.0 / 4096
    x = jnp.arange(n, dtype=jnp.float64) - 12.2

    def f(scale):
        kappa = (scale * jnp.exp(-0.5 * (x / 3.0) ** 2))[None, :].astype(
            jnp.complex128)
        return core._time_marginalize_reflected_primitive(
            kappa, jnp.zeros((1, n)), dt)[0]

    value = f(600.0)
    grad = jax.grad(f)(600.0)
    hess = jax.hessian(f)(600.0)
    assert np.all(np.isfinite(np.asarray([value, grad, hess])))
    assert "jax.checkpoint(refine_one)" in inspect.getsource(
        core._time_marginalize_reflected_primitive)


def test_phase_marginalization_refines_kappa_before_abs_near_nyquist():
    n, dt, amp = 17, 1.0, 5.0
    kappa = amp * (-1.0) ** np.arange(n)
    rho = np.zeros((1, n))
    factor = 64
    dense = np.asarray(core._reflected_fft_upsample(
        jnp.asarray(kappa[None, :]), factor))[0].real
    x = np.arange((n - 1) * factor + 1) / factor
    np.testing.assert_allclose(dense, amp * np.cos(np.pi * x),
                               rtol=0, atol=2e-12)
    got = float(core._time_marginalize_reflected_primitive(
        jnp.asarray(kappa[None, :]), jnp.asarray(rho), dt,
        phase_marginalization=True)[0])
    # On the input grid abs(kappa) is identically amp, so interpolating the
    # already-marginalized field returns this demonstrably wrong constant-row
    # result.  The band-limited primitive is amp*cos(pi*t), with zeros at every
    # half sample; integrate that independent continuous model densely.
    wrong = amp + np.log((n - 1) * dt)
    x_truth = np.linspace(0.0, n - 1.0, (n - 1) * 8192 + 1)
    y = np.exp(amp * np.abs(np.cos(np.pi * x_truth)) - amp)
    want = amp + np.log(np.trapz(y, x=x_truth))
    assert abs(want - wrong) > 0.5
    # This adversary has likelihood maxima at both window endpoints and lies
    # outside the documented spectral-headroom regime.  The reconstruction is
    # mathematically correct, but the production adapter must refuse to trust
    # a boundary condition carrying material posterior mass.
    assert np.isnan(got)


def test_refinement_is_batch_composition_independent():
    sharp, dt, _ = _event_b_like_row(phase=0.41)
    broad = 20.0 * np.exp(-0.5 * ((np.arange(sharp.size) - 245.1) / 20.0) ** 2)
    w = jnp.asarray(core._simpson_weights(sharp.size, dt))
    alone = core._time_marginalize_reflected_fft(jnp.asarray(sharp[None, :]), dt, w)
    paired = core._time_marginalize_reflected_fft(
        jnp.asarray(np.stack((broad, sharp))), dt, w)
    np.testing.assert_allclose(np.asarray(alone[0]), np.asarray(paired[1]),
                               rtol=0, atol=1e-10)


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


def test_all_wrappers_expose_quadrature_but_nonlinear_path_refuses_bandlimited():
    classes = [wrapper.JAXExtrinsicLikelihood,
               wrapper.JAXDistanceMarginalizedLikelihood,
               wrapper.JAXDistPhiMargLikelihood,
               wrapper.JAXDistPhiPsiMargLikelihood,
               wrapper.JAXDistPsiMargLikelihood]
    for cls in classes:
        assert "time_quadrature" in inspect.signature(cls.__init__).parameters
    with pytest.raises(ValueError, match="primitive time fields"):
        wrapper._validate_nonlinear_time_quadrature(
            "bandlimited", "distance/phase marginalization")


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
            "--interpolate-time", "sinc"]
    opts, _ = parser.parse_args(argv)
    drv.record_supplied_options(opts, argv, parser)
    drv.resolve_ile_interface_aliases(opts, parser)
    drv.check_critical_and_report(opts, parser)
    assert opts.time_marginalization_quadrature == "bandlimited"
    assert opts.interp == "sinc"

    argv = ["--resample-time-marginalization"]
    opts, _ = parser.parse_args(argv)
    drv.record_supplied_options(opts, argv, parser)
    with pytest.raises(SystemExit):
        drv.check_critical_and_report(opts, parser)

    argv = ["--interp", "linear", "--interpolate-time", "sinc"]
    opts, _ = parser.parse_args(argv)
    drv.record_supplied_options(opts, argv, parser)
    with pytest.raises(SystemExit):
        drv.resolve_ile_interface_aliases(opts, parser)


def test_headline_phase_marginalized_export_keeps_sky_and_psi_not_phi_or_time(tmp_path):
    import types
    drv = _load_driver()
    opts = types.SimpleNamespace(
        output_file=str(tmp_path / "ile"), save_samples=True, seed=19,
        mode="nuts", phase_marginalization=True)
    theta = np.arange(24.0).reshape(4, 6)
    drv.write_samples(opts, 0, theta, np.arange(4.0), with_distance=True)
    path = tmp_path / "ile_0_samples.dat"
    lines = path.read_text().splitlines()
    assert lines[0].lstrip("# ") == (
        "right_ascension declination distance inclination psi loglikelihood")
    assert "phi_orb" not in lines[0]
    assert "t_ref" not in lines[0]
    assert "phase=analytically-marginalized" in lines[1]
    values = np.loadtxt(path)
    np.testing.assert_array_equal(values[:, 0], theta[:, 0])
    np.testing.assert_array_equal(values[:, 1], theta[:, 1])
    np.testing.assert_array_equal(values[:, 4], theta[:, 2])
