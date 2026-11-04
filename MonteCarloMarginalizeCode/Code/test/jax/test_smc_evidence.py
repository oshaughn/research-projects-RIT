"""The SMC ladder normalizes over every prior walker, including zero likelihood."""

import numpy as np


def test_smc_evidence_counts_nonfinite_walkers_in_prior_mean(monkeypatch):
    from RIFT.likelihood.jax_ile import samplers

    class ThreeAngleLike:
        ANGULAR_PARAM_ORDER = ("ra", "dec", "incl")

    cloud = np.array([[0.1, 0.0, 0.5], [0.2, 0.0, 0.5],
                      [0.3, 0.0, 0.5], [0.4, 0.0, 0.5]])
    ln_likelihood = np.array([0.0, -1.0, -np.inf, -np.inf])
    monkeypatch.setattr(samplers, "sample_prior_3", lambda n, rng: cloud.copy())
    monkeypatch.setattr(samplers, "eval_lnL_3",
                        lambda like, theta, desc: ln_likelihood.copy())
    result = samplers.smc_puffball_sample(
        ThreeAngleLike(), 1.0, 1000.0, n_walkers=4, n_move=0,
        max_stages=1, max_dbeta=1.0, ess_frac=0.4, is_evidence=False, seed=1)
    assert result["inv_T"] == 1.0
    assert np.isclose(result["logZ_laplace"],
                      np.log((1.0 + np.exp(-1.0)) / 4.0))


def test_smc_all_finite_reference_is_unchanged(monkeypatch):
    from RIFT.likelihood.jax_ile import samplers

    class ThreeAngleLike:
        ANGULAR_PARAM_ORDER = ("ra", "dec", "incl")

    cloud = np.array([[0.1, 0.0, 0.5], [0.2, 0.0, 0.5],
                      [0.3, 0.0, 0.5], [0.4, 0.0, 0.5]])
    ln_likelihood = np.array([0.0, -1.0, -2.0, -3.0])
    monkeypatch.setattr(samplers, "sample_prior_3", lambda n, rng: cloud.copy())
    monkeypatch.setattr(samplers, "eval_lnL_3",
                        lambda like, theta, desc: ln_likelihood.copy())
    result = samplers.smc_puffball_sample(
        ThreeAngleLike(), 1.0, 1000.0, n_walkers=4, n_move=0,
        max_stages=1, max_dbeta=1.0, ess_frac=0.2, is_evidence=False, seed=1)
    assert result["inv_T"] == 1.0
    assert np.isclose(result["logZ_laplace"],
                      np.log(np.mean(np.exp(ln_likelihood))))
