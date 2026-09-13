"""The GMM score must describe its component-conditioned bounded draws."""

import numpy as np
from scipy.stats import multivariate_normal, norm

from RIFT.integrators import gaussian_mixture_model as GMM


def test_component_truncated_score_matches_draws():
    model = GMM.gmm(2, np.array([[0.0, 1.0]]))
    model.d = 1
    model.means = [np.array([0.0]), np.array([0.9])]
    model.covariances = [np.array([[0.2**2]]), np.array([[0.5**2]])]
    model.weights = np.array([0.5, 0.5])

    component_mass = np.array([
        norm.cdf((1.0 - mu) / sd) - norm.cdf((-1.0 - mu) / sd)
        for mu, sd in ((0.0, 0.2), (0.9, 0.5))
    ])
    assert component_mass[0] > 0.99 and component_mass[1] < 0.6

    x = np.array([0.25, 0.5, 0.9, 0.97])
    y = 2.0 * x - 1.0
    expected = 2.0 * sum(
        0.5 * norm.pdf(y, loc=mu, scale=sd) / mass
        for (mu, sd), mass in zip(((0.0, 0.2), (0.9, 0.5)), component_mass)
    )
    np.testing.assert_allclose(model.score(x[:, None]), expected, rtol=1e-12)

    # This bin is supplied mainly by the second, more strongly truncated
    # component.  Verify that sample() actually follows the component-wise
    # conditional law that score() now reports.
    rng_state = np.random.get_state()
    try:
        np.random.seed(19)
        draws = np.asarray(model.sample(10000)).reshape(-1)
    finally:
        np.random.set_state(rng_state)
    empirical = np.mean((draws >= 0.85) & (draws <= 0.95))
    expected_bin = 0.5 * sum(
        (norm.cdf((2.0 * 0.95 - 1.0 - mu) / sd)
         - norm.cdf((2.0 * 0.85 - 1.0 - mu) / sd)) / mass
        for (mu, sd), mass in zip(((0.0, 0.2), (0.9, 0.5)), component_mass)
    )
    assert abs(empirical - expected_bin) < 0.015


def test_multivariate_component_truncation_score():
    model = GMM.gmm(2, np.array([[0.0, 1.0], [0.0, 1.0]]))
    model.d = 2
    model.means = [np.array([0.0, 0.0]), np.array([0.8, 0.8])]
    model.covariances = [np.eye(2) * 0.25**2, np.eye(2) * 0.6**2]
    model.weights = np.array([0.5, 0.5])

    mass = [
        (norm.cdf((1.0 - mu) / sd)
         - norm.cdf((-1.0 - mu) / sd))**2
        for mu, sd in ((0.0, 0.25), (0.8, 0.6))
    ]
    assert mass[0] > 0.99 and mass[1] < 0.5
    x = np.array([[0.5, 0.5], [0.9, 0.9]])
    y = 2.0 * x - 1.0
    expected = 4.0 * sum(
        0.5 * multivariate_normal.pdf(y, mean=[mu, mu], cov=np.eye(2) * sd**2) / c
        for (mu, sd), c in zip(((0.0, 0.25), (0.8, 0.6)), mass)
    )
    np.testing.assert_allclose(model.score(x), expected, rtol=1e-7)
