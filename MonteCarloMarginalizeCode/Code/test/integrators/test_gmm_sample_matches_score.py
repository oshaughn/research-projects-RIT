#!/usr/bin/env python3
"""gmm.sample() must draw the density gmm.score() reports, on the backend of the model.

score() normalizes each component to the box individually, so sample() must draw component k
with probability w_k and then from N(mu_k, Sigma_k) truncated to the box.  The hard cases are
narrow and strongly correlated components at a bound or corner, where most of the Gaussian
mass lies outside the box.

Each check runs on the host backend and, where a CUDA device is usable, on cupy.
"""
import functools
import os

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import kstest, ks_2samp

from RIFT.integrators import gaussian_mixture_model as GMM


@functools.lru_cache(maxsize=None)
def _device_usable():
    if not getattr(GMM, 'cupy_ok', False):
        return False
    try:
        import cupy
        return float(cupy.asarray([1.0]).sum().get()) == 1.0
    except Exception:
        return False


# The device legs are collected only where a device is usable, so the cupy-less CI runner
# reports no skips.  .travis/precommit-recommended.sh sets RIFT_REQUIRE_GPU=1 so that a GPU
# host without a usable device fails here instead of quietly dropping them.
if os.environ.get('RIFT_REQUIRE_GPU') and not _device_usable():
    raise RuntimeError('RIFT_REQUIRE_GPU is set but no usable CUDA device is visible')
BACKENDS = ['host'] + (['device'] if _device_usable() else [])


def _xpy(backend):
    if backend == 'host':
        return np
    import cupy
    return cupy


def _host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


@pytest.fixture(autouse=True)
def _restore_rng():
    state = np.random.get_state()
    cp_state = None
    if _device_usable():
        import cupy
        cp_state = cupy.random.get_random_state()
        cupy.random.set_random_state(cupy.random.RandomState(0))
    yield
    np.random.set_state(state)
    if cp_state is not None:
        import cupy
        cupy.random.set_random_state(cp_state)


def _seed(backend, seed):
    np.random.seed(seed)
    if backend == 'device':
        import cupy
        cupy.random.seed(seed)


def _model(bounds, means, covs, weights, xpy):
    """Parameters are in the normalized [-1, 1] frame, as gmm stores them."""
    bounds = np.asarray(bounds, dtype=float)
    m = GMM.gmm(len(weights), xpy.asarray(bounds))
    m.d = bounds.shape[0]
    m.means = [xpy.asarray(np.atleast_1d(mu).astype(float)) for mu in means]
    m.covariances = [xpy.asarray(np.atleast_2d(c).astype(float)) for c in covs]
    m.weights = xpy.asarray(np.asarray(weights, dtype=float))
    m.adapt = [False] * len(weights)
    m.N = 0
    return m


def _cov(sigma, rho, d):
    c = np.full((d, d), rho)
    np.fill_diagonal(c, 1.0)
    return sigma ** 2 * c


# Normalized-frame mixtures.  Each has a narrow component whose mass is mostly outside the box,
# and a broad one so that 1/q has finite variance over the whole box.
CASES = {
    'd1_edge': ([[0.0, 2.0]],
                [[0.97], [-0.3], [0.0]],
                [[[0.03 ** 2]], [[0.1 ** 2]], [[2.0 ** 2]]],
                [0.5, 0.3, 0.2]),
    'd2_corner_anticorr': ([[0.0, 1.0], [-2.0, 3.0]],
                           [[0.98, 0.98], [-0.5, 0.2], [0.0, 0.0]],
                           [_cov(0.05, -0.95, 2), _cov(0.2, 0.3, 2), _cov(1.0, 0.0, 2)],
                           [0.5, 0.35, 0.15]),
    'd3_corner_corr': ([[0.0, 1.0], [-1.0, 1.0], [10.0, 20.0]],
                       [[0.99, -0.99, 0.99], [0.0, 0.0, 0.0]],
                       [_cov(0.1, 0.9, 3), _cov(1.0, -0.2, 3)],
                       [0.7, 0.3]),
}

N = 200000


def _draw_and_score(case, backend, seed=11):
    xpy = _xpy(backend)
    bounds, means, covs, weights = CASES[case]
    m = _model(bounds, means, covs, weights, xpy)
    _seed(backend, seed)
    x = m.sample(N)
    assert isinstance(x, xpy.ndarray), 'draws must come back on the model backend'
    q = m.score(x)
    return m, _host(x), _host(q), np.asarray(bounds, dtype=float)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('case', sorted(CASES))
def test_draws_lie_inside_the_box(case, backend):
    _, x, _, bounds = _draw_and_score(case, backend)
    assert np.all(np.isfinite(x))
    assert np.all((x > bounds[:, 0]) & (x < bounds[:, 1]))


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('case', sorted(CASES))
def test_inverse_score_integrates_sub_boxes(case, backend):
    """E_q[1{x in B} / q] = vol(B) for every sub-box B, if the draws follow score().

    Sub-boxes are the 2^d orthants about the box centre and the 2^d about the narrow
    component's corner, so a sampler that mis-weights the truncated tail moves mass between
    them.  Checked within 5 standard errors."""
    _, x, q, bounds = _draw_and_score(case, backend)
    d = bounds.shape[0]
    vol = np.prod(bounds[:, 1] - bounds[:, 0])
    mean0 = np.asarray(CASES[case][1][0], dtype=float)
    split_points = [0.5 * (bounds[:, 0] + bounds[:, 1]),
                    bounds[:, 0] + 0.5 * (1.0 + 0.9 * mean0) * (bounds[:, 1] - bounds[:, 0])]
    checked = 0
    for c in split_points:
        for corner in range(2 ** d):
            hi_side = np.array([(corner >> j) & 1 for j in range(d)], dtype=bool)
            lo = np.where(hi_side, c, bounds[:, 0])
            hi = np.where(hi_side, bounds[:, 1], c)
            inside = np.all((x >= lo) & (x < hi), axis=1)
            v = inside / q
            mean, err = np.mean(v), np.std(v) / np.sqrt(N)
            expect = np.prod(hi - lo)
            assert abs(mean - expect) < 5 * err + 1e-3 * vol, (c, corner, mean, expect, err)
            checked += 1
    assert checked == 2 ** (d + 1)


@pytest.mark.parametrize('backend', BACKENDS)
def test_one_dim_draws_follow_the_truncated_mixture_cdf(backend):
    """KS against the closed-form CDF of the individually truncated 1-D mixture."""
    m, x, _, bounds = _draw_and_score('d1_edge', backend)
    _, means, covs, weights = CASES['d1_edge']
    w = np.asarray(weights) / np.sum(weights)

    def cdf(xx):
        u = 2.0 * (xx - bounds[0, 0]) / (bounds[0, 1] - bounds[0, 0]) - 1.0
        out = 0.0
        for wk, mu, c in zip(w, means, covs):
            s = np.sqrt(c[0][0])
            a, b = ndtr((-1 - mu[0]) / s), ndtr((1 - mu[0]) / s)
            out = out + wk * (ndtr((u - mu[0]) / s) - a) / (b - a)
        return out
    assert kstest(x[:, 0], cdf).pvalue > 1e-3


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('case', ['d2_corner_anticorr', 'd3_corner_corr'])
def test_marginals_match_brute_force_rejection(case, backend):
    """Two-sample KS of every 1-D marginal, per component, against draws of the untruncated
    Gaussian kept only inside the box.

    d3_corner_corr has a repeated covariance eigenvalue.  The scipy-truncnorm sampler this
    replaced (multivariate_truncnorm.sample) fails this check there, p ~ 1e-35, because
    np.linalg.eig returns non-orthogonal eigenvectors for it."""
    bounds, means, covs, weights = CASES[case]
    d = len(bounds)
    xpy = _xpy(backend)
    rng = np.random.default_rng(4)
    for mu, c in zip(means, covs):
        m = _model(bounds, [mu], [c], [1.0], xpy)
        _seed(backend, 5)
        new = _host(m._normalize(m.sample(20000)))
        truth = rng.multivariate_normal(np.asarray(mu, float), np.asarray(c), 2000000)
        truth = truth[np.all((truth > -1) & (truth < 1), axis=1)]
        assert len(truth) > 20000
        for j in range(d):
            assert ks_2samp(new[:, j], truth[:, j]).pvalue > 1e-3, (mu, j)


@pytest.mark.parametrize('backend', BACKENDS)
def test_seeded_draws_repeat_and_use_only_the_backend_generator(backend):
    """The same backend seed gives the same draw, and the OTHER generator has no influence,
    so --seed (seed_everything seeds both) covers the whole draw on either backend."""
    xpy = _xpy(backend)
    bounds, means, covs, weights = CASES['d2_corner_anticorr']
    m = _model(bounds, means, covs, [0.37, 0.43, 0.2], xpy)

    def draw(np_seed, dev_seed):
        np.random.seed(np_seed)
        if backend == 'device':
            import cupy
            cupy.random.seed(dev_seed)
        return _host(m.sample(1001))
    a = draw(1, 2)
    assert np.array_equal(a, draw(1, 2))
    if backend == 'device':
        assert np.array_equal(a, draw(99, 2)), 'device draw consumed numpy.random'
        assert not np.array_equal(a, draw(1, 3))
    else:
        assert not np.array_equal(a, draw(2, 2))


@pytest.mark.parametrize('backend', BACKENDS)
def test_low_acceptance_component_is_filled(backend):
    """A component whose whitened-box acceptance is below 1% still yields exactly n rows,
    all from the truncated density."""
    xpy = _xpy(backend)
    m = _model([[-1.0, 1.0]] * 3, [[1.0, -1.0, 1.0]], [_cov(0.03, 0.999, 3)], [1.0], xpy)
    _seed(backend, 3)
    x = _host(m.sample(5000))
    assert x.shape == (5000, 3)
    assert np.all((x > -1) & (x < 1))


def test_non_positive_definite_covariance_raises():
    m = _model([[0.0, 1.0]] * 2, [[0.0, 0.0]], [[[1.0, 2.0], [2.0, 1.0]]], [1.0], np)
    with pytest.raises(Exception, match='positive'):
        m.sample(10)


def _brute_force(mu, c, n, seed):
    rng = np.random.default_rng(seed)
    truth = rng.multivariate_normal(np.asarray(mu, float), np.asarray(c), n)
    return truth[np.all((truth > -1) & (truth < 1), axis=1)]


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('side', [-1.0, 1.0])
def test_mean_far_outside_the_box_uses_the_reflected_interval(backend, side):
    """A mean 9 sigma outside the box puts a whitened interval at [9, 29] or [-29, -9],
    depending on the eigenvector sign; one of the two sides needs the reflection, without
    which ndtr(9) rounds to 1.  Fitted means stay inside the box, so only this reaches it.
    Diagonal covariance, so each marginal is a 1-D truncated normal."""
    from scipy.stats import truncnorm
    xpy = _xpy(backend)
    sig = np.array([0.1, 0.3])
    mu = [side * 1.9, 0.2]
    m = _model([[-1.0, 1.0]] * 2, [mu], [np.diag(sig ** 2)], [1.0], xpy)
    _seed(backend, 8)
    x = _host(m.sample(20000))
    for j in range(2):
        a, b = (-1 - mu[j]) / sig[j], (1 - mu[j]) / sig[j]
        cdf = truncnorm(a, b, loc=mu[j], scale=sig[j]).cdf
        assert kstest(x[:, j], cdf).pvalue > 1e-3, j


@pytest.mark.parametrize('backend', BACKENDS)
def test_batched_rounds_keep_the_distribution(backend, monkeypatch):
    """With the per-round candidate cap far below n, rows are filled over many batches."""
    monkeypatch.setattr(GMM, '_MAX_CANDIDATES_PER_ROUND', 997)
    xpy = _xpy(backend)
    _, means, covs, _ = CASES['d2_corner_anticorr']
    m = _model([[-1.0, 1.0]] * 2, [means[0]], [covs[0]], [1.0], xpy)
    _seed(backend, 12)
    new = _host(m.sample(30000))
    truth = _brute_force(means[0], covs[0], 2000000, 13)
    for j in range(2):
        assert ks_2samp(new[:, j], truth[:, j]).pvalue > 1e-3, j


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_component_with_no_rows_is_not_set_up(backend):
    """A zero-weight component is never drawn, so it cannot make sample() fail, with or
    without bounds."""
    xpy = _xpy(backend)
    bad = [[1.0, 2.0], [2.0, 1.0]]                  # not positive definite
    m = _model([[-1.0, 1.0]] * 2, [[0.0, 0.0], [0.0, 0.0]],
               [_cov(0.3, 0.0, 2), bad], [1.0, 0.0], xpy)
    for use_bounds in (True, False):
        x = _host(m.sample(500, use_bounds=use_bounds))
        assert x.shape == (500, 2) and np.all(np.isfinite(x))


@pytest.mark.parametrize('backend', BACKENDS)
def test_numpy_integer_count(backend):
    """mcsamplerPortfolio passes n as a numpy integer; cupy.random.permutation takes that
    for an array and fails on len()."""
    xpy = _xpy(backend)
    m = _model([[-1.0, 1.0]] * 2, [[0.0, 0.0]], [_cov(0.3, 0.2, 2)], [1.0], xpy)
    assert _host(m.sample(np.int64(300))).shape == (300, 2)
