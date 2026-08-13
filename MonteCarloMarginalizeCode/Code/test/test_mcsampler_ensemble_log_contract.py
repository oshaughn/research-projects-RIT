import numpy as np

from RIFT.integrators import mcsamplerEnsemble


def test_integrate_log_dispatches_with_log_flags():
    sampler = mcsamplerEnsemble.MCSampler()
    seen = {}

    def fake_integrate(func, *args, **kwargs):
        seen.update(func=func, args=args, kwargs=kwargs)
        return "sentinel"

    sampler.integrate = fake_integrate
    func = object()

    assert sampler.integrate_log(func, "x", nmax=17) == "sentinel"
    assert seen == {
        "func": func,
        "args": ("x",),
        "kwargs": {"nmax": 17, "use_lnL": True, "return_lnI": True},
    }


def test_log_weight_convergence_avoids_linear_underflow():
    rvs = {"log_weights": np.array([-10000.0, -10001.0, -10002.0])}

    expected_fraction = 1.0 / (1.0 + np.exp(-1.0) + np.exp(-2.0))
    assert mcsamplerEnsemble.convergence_test_MostSignificantPoint(
        expected_fraction + 1e-12, rvs, None
    )
    assert not mcsamplerEnsemble.convergence_test_MostSignificantPoint(
        expected_fraction - 1e-12, rvs, None
    )


def test_integrate_log_exports_consistent_log_sample_fields():
    sampler = mcsamplerEnsemble.MCSampler()
    sampler.add_parameter(
        "x",
        pdf=lambda x: np.ones_like(x) / 2.0,
        prior_pdf=lambda x: np.ones_like(x) / 2.0,
        left_limit=-1.0,
        right_limit=1.0,
        adaptive_sampling=True,
    )

    sampler.integrate_log(
        lambda x: -0.5 * np.asarray(x) ** 2,
        "x",
        n=100,
        nmax=100,
        neff=1,
        min_iter=1,
        max_iter=1,
        correlate_all_dims=True,
        n_comp=1,
    )

    np.testing.assert_allclose(
        sampler._rvs["log_weights"],
        sampler._rvs["log_integrand"]
        + sampler._rvs["log_joint_prior"]
        - sampler._rvs["log_joint_s_prior"],
    )
    np.testing.assert_array_equal(
        sampler._rvs["integrand"], sampler._rvs["log_integrand"]
    )


def test_linear_integration_clears_log_fields_when_sampler_is_reused():
    sampler = mcsamplerEnsemble.MCSampler()
    sampler.add_parameter(
        "x",
        pdf=lambda x: np.ones_like(x) / 2.0,
        prior_pdf=lambda x: np.ones_like(x) / 2.0,
        left_limit=-1.0,
        right_limit=1.0,
        adaptive_sampling=True,
    )
    integration_options = {
        "n": 100,
        "nmax": 100,
        "neff": 1,
        "min_iter": 1,
        "max_iter": 1,
        "correlate_all_dims": True,
        "n_comp": 1,
    }

    sampler.integrate_log(
        lambda x: -0.5 * np.asarray(x) ** 2,
        "x",
        **integration_options,
    )
    assert "log_weights" in sampler._rvs

    sampler.integrate(
        lambda x: np.exp(-0.5 * np.asarray(x) ** 2),
        "x",
        **integration_options,
    )

    log_fields = {
        "log_integrand",
        "log_joint_prior",
        "log_joint_s_prior",
        "log_weights",
    }
    assert log_fields.isdisjoint(sampler._rvs)

    weights = (
        sampler._rvs["integrand"]
        * sampler._rvs["joint_prior"]
        / sampler._rvs["joint_s_prior"]
    )
    expected_fraction = np.max(weights) / np.sum(weights)
    assert mcsamplerEnsemble.convergence_test_MostSignificantPoint(
        expected_fraction + 1e-12, sampler._rvs, None
    )
    assert not mcsamplerEnsemble.convergence_test_MostSignificantPoint(
        expected_fraction - 1e-12, sampler._rvs, None
    )


def test_normal_subintegrals_accept_log_weights(monkeypatch):
    # Eight equal-mass chunks at a scale where conversion to linear weights
    # would underflow.  Stub only the normality statistic; the test exercises
    # the logsumexp chunk reduction and its relative-error gate.
    monkeypatch.setattr(
        mcsamplerEnsemble.stats,
        "normaltest",
        lambda values: (0.0, 0.5),
    )
    rvs = {"log_weights": np.full(80, -10000.0)}

    assert mcsamplerEnsemble.convergence_test_NormalSubIntegrals(
        8, 0.01, 1e-12, rvs, None
    )
