#!/usr/bin/env python3
"""Draws of separate dim groups must be independent row by row.

MonteCarloEnsemble._sample() draws each dim group with its own model.sample(n), writes the
draws into the same rows, and reports the product of the group densities as the joint
sampling density.  gmm.sample() used to return rows grouped by component (first ~n*w0 rows
from component 0, ...).  Two groups then shared component labels row by row, so the true
joint density was not the claimed product and E_q[p/q] was far from 1.
"""
import numpy as np
import pytest

from RIFT.integrators import gaussian_mixture_model as GMM
from RIFT.integrators import MonteCarloEnsemble as monte_carlo

BOUNDS_1D = np.array([[0.0, 1.0]])
VOL = 1.0   # each group spans [0,1]
# normalized-frame components: two narrow modes of different widths plus a broad one
MEANS = [-0.5, 0.5, 0.0]
SIGMAS = [0.02, 0.08, 2.0]
WEIGHTS = [0.45, 0.45, 0.10]


@pytest.fixture(autouse=True)
def _restore_numpy_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _model(means=MEANS, sigmas=SIGMAS, weights=WEIGHTS, xpy=np):
    """Host model by default.  Pass the integrator's backend when the model goes into one."""
    m = GMM.gmm(len(weights), xpy.asarray(BOUNDS_1D))
    m.means = [xpy.asarray([mu]) for mu in means]
    m.covariances = [xpy.asarray([[s ** 2]]) for s in sigmas]
    m.weights = xpy.asarray(weights, dtype=float)
    m.adapt = [False] * len(weights)
    m.d = 1
    m.N = 0
    return m


def _draw(n, seed):
    np.random.seed(seed)
    xpy = monte_carlo.xpy_default   # cupy wherever a device is visible
    gmm_dict = {(0,): _model(xpy=xpy), (1,): _model(xpy=xpy)}
    bounds = {(0,): xpy.asarray(BOUNDS_1D[0]), (1,): xpy.asarray(BOUNDS_1D[0])}
    integ = monte_carlo.integrator(2, bounds, gmm_dict, 3, n=n, user_func=None, L_cutoff=None)
    integ._sample()
    host = monte_carlo.identity_convert
    return host(integ.sample_array), host(integ.sampling_prior_array)


def _mean_and_err(v):
    return np.mean(v), np.std(v) / np.sqrt(len(v))


def _norm_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


# Measured 2026-10-06 on the host backend, seeds 0-39: with the fix max |mean-1|/err is 2.6
# (inverse density) and 3.5 (evidence), err ~0.02; without it E[1/(vol q)] = 5.3-5.4 and the
# evidence 0.38-0.45.
N = 100000
SEEDS = (3, 5, 7, 11)


@pytest.mark.parametrize("seed", SEEDS)
def test_joint_inverse_density_has_unit_mean(seed):
    """E_q[1/(vol q)] = 1 jointly; every importance weight relies on it."""
    _, q = _draw(N, seed)
    mean, err = _mean_and_err(1.0 / (VOL * q))
    assert abs(mean - 1.0) < 5 * err, (mean, err)


@pytest.mark.parametrize("seed", SEEDS)
def test_separable_target_evidence(seed):
    """Z = int g(x) h(y) dx dy = 1 for a target on component 0 in x and component 1 in y."""
    x, q = _draw(N, seed)
    f = _norm_pdf(x[:, 0], 0.25, 0.05) * _norm_pdf(x[:, 1], 0.75, 0.05)
    mean, err = _mean_and_err(f / q)
    assert abs(mean - 1.0) < 5 * err, (mean, err)


def test_row_order_carries_no_component_label():
    """Component 0 (x < 0.3) must land in the first and second halves of a draw equally."""
    np.random.seed(19)
    m = _model()
    n = 50
    frac = [np.mean(np.asarray(m.sample(n))[: n // 2, 0] < 0.3) for _ in range(400)]
    other = [np.mean(np.asarray(m.sample(n))[n // 2:, 0] < 0.3) for _ in range(400)]
    assert abs(np.mean(frac) - np.mean(other)) < 0.03, (np.mean(frac), np.mean(other))


def test_small_weight_component_is_drawn():
    """A weight below 1/n is drawn at rate n*w on average, not truncated to zero draws."""
    np.random.seed(23)
    w0, n = 0.004, 50
    m = _model(means=[-0.5, 0.5], sigmas=[0.02, 0.02], weights=[w0, 1 - w0])
    hits = np.array([np.sum(np.asarray(m.sample(n))[:, 0] < 0.5) for _ in range(4000)])
    mean, err = _mean_and_err(hits)
    assert abs(mean - n * w0) < 5 * err, (mean, err, n * w0)
