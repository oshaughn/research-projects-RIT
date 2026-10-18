"""Network AV: delay rings, isotropic measure, likelihood and export invariants."""
from types import SimpleNamespace
import importlib.machinery
import importlib.util
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pytest
from RIFT.likelihood.jax_ile import samplers


class RingLikelihood:
    ANGULAR_PARAM_ORDER = ("ra", "dec", "incl")

    def __init__(self, width=0.04):
        self.data = SimpleNamespace(
            detector_names=["A", "B"], gmst=1.7,
            detectors={"A": {"location": np.array([6370000., 0., 0.])},
                       "B": {"location": np.array([0., 6370000., 0.])}})
        self.width = width
        from RIFT.likelihood.jax_ile.coordinates import build_network_frame
        self.rotation = build_network_frame(
            self.data.detectors["A"]["location"],
            self.data.detectors["B"]["location"], self.data.gmst)

    def log_likelihood(self, ra, dec, incl):
        from RIFT.likelihood.jax_ile.coordinates import equatorial_to_network
        polar, _ = equatorial_to_network(ra, dec, self.rotation, self.data.gmst)
        return -0.5 * ((jnp.cos(polar) - 0.31) / self.width) ** 2


def test_network_roundtrip_and_constant_delay_ring():
    adapter = samplers._AVNetworkSky(RingLikelihood())
    theta = np.column_stack((np.full(100, 0.31),
                             np.linspace(0., 2*np.pi, 100, endpoint=False),
                             np.full(100, 1.1)))
    physical = np.asarray(adapter.to_physical(theta))
    back = np.asarray(adapter.from_physical(physical))
    np.testing.assert_allclose(back[:, [0, 2]], theta[:, [0, 2]], atol=2e-12)
    angle_error = np.angle(np.exp(1j * (back[:, 1] - theta[:, 1])))
    np.testing.assert_allclose(angle_error, 0., atol=2e-12)
    # Ring crosses a wide RA/DEC support but is one line in AV's first axis.
    assert np.ptp(physical[:, 0]) > 2.
    assert np.ptp(physical[:, 1]) > 1.
    np.testing.assert_allclose(adapter.log_likelihood(*theta.T), 0., atol=1e-20)


def test_rotated_prior_is_isotropic_and_normalized():
    adapter = samplers._AVNetworkSky(RingLikelihood())
    order = adapter.ANGULAR_PARAM_ORDER
    theta = samplers._av_prior_draw(order, 40000, np.random.default_rng(74), 1., 100.)
    physical = np.asarray(adapter.to_physical(theta))
    assert abs(np.mean(np.sin(physical[:, 1]))) < .01
    assert abs(np.mean(np.sin(physical[:, 1])**2) - 1./3) < .01
    for name in ("cos_theta_n", "phi_n"):
        lo, hi, density = samplers._av_prior_spec(name, 1., 100.)
        assert float(density(np.array([.1]))[0]) * (hi-lo) == pytest.approx(1.)


def test_fixed_batch_callback_evaluates_physical_likelihood():
    like = RingLikelihood()
    adapter = samplers._AVNetworkSky(like)
    theta = samplers._av_prior_draw(adapter.ANGULAR_PARAM_ORDER, 19,
                                   np.random.default_rng(7), 1., 100.)
    got = samplers._fixed_shape_value_callback(adapter, 3, 8)(*theta.T)
    physical = np.asarray(adapter.to_physical(theta))
    np.testing.assert_allclose(got, like.log_likelihood(*physical.T), atol=1e-12)


def test_actual_av_ring_evidence_and_export_density():
    from scipy.special import erf
    like = RingLikelihood(width=.08)
    result = samplers.adaptive_volume_sample(
        like, 1., 100., sky_coords="network", nmax=40000,
        n_chunk=1024, eval_chunk=256, neff=300., seed=42)
    # Analytic integral of the ring likelihood under isotropic sky.
    width = like.width
    z = width * np.sqrt(np.pi/2.) * .5 * (
        erf((1.-.31)/(np.sqrt(2.)*width)) - erf((-1.-.31)/(np.sqrt(2.)*width)))
    assert abs(np.exp(result["logZ"])/z - 1.) < .15
    theta = result["theta"]
    np.testing.assert_allclose(result["lnL"], like.log_likelihood(*theta.T), atol=1e-12)
    expected_logp = np.log(np.cos(theta[:, 1]) * np.sin(theta[:, 2]) / (8*np.pi))
    np.testing.assert_allclose(result["log_joint_prior"], expected_logp, atol=1e-12)
    np.testing.assert_allclose(result["log_weight"], result["lnL"] +
                               result["log_joint_prior"] - result["log_joint_s_prior"], atol=1e-12)
    assert result["diagnostics"]["sky_coordinates"] == "network"
    assert np.all((theta[:, 0] >= 0.) & (theta[:, 0] < 2*np.pi))
    assert "cos_theta_n" in result["sampler"].params
    assert "ra" not in result["sampler"].params


@pytest.mark.parametrize("problem", ["one", "duplicate", "restricted"])
def test_network_request_fails_instead_of_silently_ignoring(problem):
    like = RingLikelihood()
    kwargs = {}
    if problem == "one":
        like.data.detector_names = ["A"]
    elif problem == "duplicate":
        like.data.detectors["B"]["location"] = like.data.detectors["A"]["location"]
    else:
        kwargs["sample_bounds"] = {"ra": (1., 2.)}
    with pytest.raises(ValueError):
        samplers.adaptive_volume_sample(like, 1., 100., sky_coords="network", **kwargs)


def test_driver_compatibility_flag_is_implemented_for_av(capsys):
    path = Path(__file__).parents[2] / "bin/integrate_likelihood_extrinsic_jax"
    loader = importlib.machinery.SourceFileLoader("network_av_driver", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    driver = importlib.util.module_from_spec(spec)
    loader.exec_module(driver)
    parser = driver.build_parser()
    opts, _ = parser.parse_args(["--sampler-method", "AV", "--internal-sky-network-coordinates"])
    driver.check_critical_and_report(opts, parser)
    assert opts.internal_sky_network_coordinates
    assert driver._av_sky_sampling_kwargs(opts, ("V1", "H1", "L1")) == {
        "sky_coords": "network", "network_exclude_detectors": ("V1", "K1")}
    assert "--internal-sky-network-coordinates" not in capsys.readouterr().out
    # Classic ILE falls back to equatorial when V1/K1 exclusion leaves <2 IFOs.
    for names in (("L1", "V1"), ("H1", "K1"), ("H1",)):
        assert driver._av_sky_sampling_kwargs(opts, names) == {"sky_coords": "equatorial"}
        assert "equatorial" in capsys.readouterr().out
    # The explicit request is not an alias: it keeps every detector, fails closed in AV.
    opts, _ = parser.parse_args(["--sampler-method", "AV", "--sky-coordinates", "network"])
    assert driver._av_sky_sampling_kwargs(opts, ("L1", "V1")) == {"sky_coords": "network"}
    # -raw keeps V1/K1 and the given order, as in classic ILE.
    opts, _ = parser.parse_args(["--sampler-method", "AV", "--internal-sky-network-coordinates",
                                 "--internal-sky-network-coordinates-raw"])
    driver.check_critical_and_report(opts, parser)
    assert "--internal-sky-network-coordinates-raw" not in capsys.readouterr().out
    assert driver._av_sky_sampling_kwargs(opts, ("H1", "V1")) == {
        "sky_coords": "network", "network_exclude_detectors": ()}


def test_driver_passes_sky_kwargs_to_av():
    """The helper is only useful if analyze_one forwards it to the AV call."""
    import ast
    path = Path(__file__).parents[2] / "bin/integrate_likelihood_extrinsic_jax"
    tree = ast.parse(path.read_text())
    analyze = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "analyze_one")
    calls = [node for node in ast.walk(analyze) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "adaptive_volume_sample"]
    assert len(calls) == 1
    forwarded = [kw.value for kw in calls[0].keywords if kw.arg is None]
    assert any(isinstance(v, ast.Call) and getattr(v.func, "id", None) == "_av_sky_sampling_kwargs"
               and ast.unparse(v.args[1]) == "like.data.detector_names" for v in forwarded)


def test_full_range_sky_window_is_accepted():
    like = RingLikelihood(width=.3)
    for bounds in ({"ra": (0., 2*np.pi)}, {"ra": (0., 6.283185)},
                   {"dec": (-np.pi/2, np.pi/2)}):
        result = samplers.adaptive_volume_sample(
            like, 1., 100., sky_coords="network", sample_bounds=bounds,
            nmax=4000, n_chunk=1000, eval_chunk=256, neff=5., seed=3)
        assert result["diagnostics"]["sky_coordinates"] == "network"


def test_network_portfolio_bootstraps_physical_caller_cloud():
    like = RingLikelihood(width=.15)
    adapter = samplers._AVNetworkSky(like)
    network_cloud = samplers._av_prior_draw(
        adapter.ANGULAR_PARAM_ORDER, 500, np.random.default_rng(31), 1., 100.)
    physical_cloud = np.asarray(adapter.to_physical(network_cloud))
    result = samplers.adaptive_volume_sample(
        like, 1., 100., sky_coords="network", sampler_method="portfolio",
        initial_samples=physical_cloud, nmax=8000, n_chunk=1000,
        eval_chunk=256, neff=25., seed=31)
    recovered = np.asarray(adapter.to_physical(result["seed_cloud"]))
    np.testing.assert_allclose(recovered, physical_cloud, atol=1e-12)
    np.testing.assert_allclose(result["lnL"], like.log_likelihood(*result["theta"].T), atol=1e-12)
    assert result["neff"] >= 25.


def test_fisher_seed_is_built_in_physical_frame_then_rotated(monkeypatch):
    like = RingLikelihood(width=.15)
    adapter = samplers._AVNetworkSky(like)
    cloud = samplers._av_prior_draw(("ra", "dec", "incl"), 500,
                                   np.random.default_rng(8), 1., 100.)
    def seed(physical_like, order, callback, *args, **kwargs):
        assert physical_like is like
        assert order == like.ANGULAR_PARAM_ORDER
        np.testing.assert_allclose(callback(*cloud.T), like.log_likelihood(*cloud.T), atol=1e-12)
        return cloud, cloud[:2], np.asarray(like.log_likelihood(*cloud[:2].T))
    monkeypatch.setattr(samplers, "_fisher_sky_seed", seed)
    result = samplers.adaptive_volume_sample(
        like, 1., 100., sky_coords="network", sampler_method="portfolio",
        seed_method="fisher-sky", nmax=8000, n_chunk=1000,
        eval_chunk=256, neff=25., seed=31)
    np.testing.assert_allclose(np.asarray(adapter.to_physical(result["seed_cloud"])), cloud, atol=1e-12)
    np.testing.assert_array_equal(result["seed_modes"], cloud[:2])


def test_classic_alias_baseline_keeps_hl_with_virgo_first():
    like = RingLikelihood()
    like.data.detectors = {
        "H1": like.data.detectors["A"], "L1": like.data.detectors["B"],
        "V1": {"location": np.array([0., 0., 6370000.])}}
    like.data.detector_names = ["V1", "H1", "L1"]
    assert samplers._AVNetworkSky(like).baseline == ("V1", "H1")
    adapter = samplers._AVNetworkSky(like, exclude_detectors=("V1", "K1"))
    assert adapter.baseline == ("H1", "L1")
    assert adapter.physical.data.detector_names == ["V1", "H1", "L1"]
