"""GMM updates and the fair-draw export must not write into arrays they do not own.

Regression tests for two measured defects (#418):

1. gmm._merge() wrote merged weights/means/covariances element-wise into the model's arrays,
   which np.asarray handed back unchanged when a caller had assigned its own array, so the
   caller's array drifted.
2. mcsamplerEnsemble's fair-draw export did `ln_wt = integrator.cumulative_values; ln_wt += ...`,
   rewriting the stored lnL (and _rvs['log_integrand'], which aliases it) in place.
"""
import numpy as np

from RIFT.integrators import gaussian_mixture_model as GMM
from RIFT.integrators import mcsamplerEnsemble


def test_update_does_not_write_into_caller_arrays():
    model = GMM.gmm(2, np.array([[-5.0, 5.0]]))
    weights = np.array([0.6, 0.4])
    means = np.array([[-0.5], [0.5]])
    model.means = means
    covs = [np.array([[0.01]]), np.array([[0.04]])]
    model.covariances = covs
    model.weights = weights
    model.d = 1
    model.N = 10000
    w0, m0, c0 = weights.copy(), means.copy(), list(covs)
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.normal(-2.0, 0.6, 5000), rng.normal(2.5, 1.0, 5000)])[:, None]
    np.random.seed(0)
    model.update(x, np.zeros(len(x)))
    assert not np.allclose(np.asarray(model.weights), w0)   # the update did move the model
    np.testing.assert_array_equal(weights, w0)
    np.testing.assert_array_equal(means, m0)
    assert all(c is c_ref for c, c_ref in zip(covs, c0))   # caller's list not rewritten


def test_fairdraw_export_keeps_stored_lnL():
    sampler = mcsamplerEnsemble.MCSampler()
    sampler.add_parameter("x", pdf=lambda x: np.ones_like(x) / 2.0,
                          prior_pdf=lambda x: np.ones_like(x) / 2.0,
                          left_limit=-1.0, right_limit=1.0, adaptive_sampling=True)
    lnL = lambda x: 50.0 - 0.5 * (np.asarray(x) / 0.2) ** 2
    np.random.seed(3)
    sampler.integrate_log(lnL, "x", n=2000, nmax=2000, neff=1, min_iter=1, max_iter=1,
                          correlate_all_dims=True, n_comp=1,
                          igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=10**6)
    assert getattr(sampler, "_rvs_is_fairdraw", False)
    np.testing.assert_allclose(sampler._rvs["log_integrand"], lnL(sampler._rvs["x"]), atol=1e-10)
    integ = sampler.integrator
    np.testing.assert_allclose(integ.cumulative_values, lnL(integ.cumulative_samples[:, 0]), atol=1e-10)
