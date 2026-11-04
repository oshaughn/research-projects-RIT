#!/usr/bin/env python3
"""gmm.score() must be a deterministic function of the model and the points, and normalized.

Each component is normalized by its Gaussian mass inside the box.  For d>=3 that mass used to
come from scipy's mvnun, whose lattice shifts come from a Fortran RNG that no seed reaches, so
the same model scored the same points differently depending on earlier calls in the process.
Normalization is checked against brute force, not against the old code.
"""
import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import multivariate_normal

from RIFT.integrators import gaussian_mixture_model as GMM


def _host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def _cov(d, rng, scale):
    A = rng.normal(size=(d, d))
    return scale * (A @ A.T) / d + 0.1 * scale * np.eye(d)


def _model(d, seed):
    '''A broad component plus a narrow, correlated one at a corner, mostly outside the box.'''
    rng = np.random.default_rng(seed)
    bounds = np.column_stack([-1.0 - rng.uniform(0, 2, d), 1.0 + rng.uniform(0, 3, d)])
    m = GMM.gmm(2, bounds)
    m.d = d
    m.means = [rng.uniform(-0.3, 0.3, d), np.full(d, 0.95)]
    # The broad component is well conditioned, so 1/q has a finite, modest variance.
    m.covariances = [0.4 * np.eye(d) + 0.1, _cov(d, rng, 0.02)]
    m.weights = np.array([0.6, 0.4])
    m.adapt = [False, False]
    return m


@pytest.mark.parametrize('d', [3, 4, 6])
def test_score_repeats_bitwise_after_other_calls(d):
    m = _model(d, seed=d)
    other = _model(d, seed=100 + d)
    x = _host(m.sample(50))
    first = _host(m.score(x))
    for _ in range(3):
        other.score(x)          # advances any hidden RNG state the mass computation uses
    second = _host(m.score(x))
    np.testing.assert_array_equal(first, second)


CASES = [  # mean offset, covariance scale, lower, upper
    (0.0, 0.3, -1.0, 1.0),
    (0.9, 0.05, -1.0, 1.0),
    (1.5, 0.4, -1.0, 1.0),
    (-0.4, 0.2, -2.0, 0.3),
]


@pytest.mark.parametrize('d', [2, 3, 4])
@pytest.mark.parametrize('case', range(len(CASES)))
def test_box_mass_matches_brute_force(d, case):
    off, scale, lo, hi = CASES[case]
    rng = np.random.default_rng(10 * d + case)
    mean = off + rng.uniform(-0.2, 0.2, d)
    cov = _cov(d, rng, scale)
    lower = np.full(d, lo) + rng.uniform(0, 0.3, d)   # unequal widths
    upper = np.full(d, hi) - rng.uniform(0, 0.3, d)
    n = 4_000_000
    x = rng.multivariate_normal(mean, cov, size=n)
    p = np.mean(np.all((x > lower) & (x < upper), axis=1))
    sigma = np.sqrt(max(p, 1.0 / n) * (1 - p) / n)
    got = GMM._box_mass(lower, upper, mean, cov)
    assert abs(got - p) < 5 * sigma + 1e-5 * p, (got, p, sigma)


@pytest.mark.parametrize('d', [2, 3, 4])
def test_score_normalized_over_box(d):
    '''E_q[1/q] over the box is the box volume when q is the normalized sampling density.'''
    m = _model(d, seed=d)
    np.random.seed(7)
    x = _host(m.sample(400_000))
    inv = 1.0 / _host(m.score(x))
    vol = float(np.prod(m.bounds[:, 1] - m.bounds[:, 0]))
    est, err = inv.mean(), inv.std() / np.sqrt(len(inv))
    assert abs(est / vol - 1) < 5 * err / vol, (est / vol, err / vol)


def test_box_mass_terminates_on_bad_covariance():
    '''A non-finite or degenerate covariance must return, not spin in the jitter retry.'''
    lo, hi = -np.ones(3), np.ones(3)
    bad = np.eye(3)
    bad[0, 0] = np.nan
    assert np.isnan(GMM._box_mass(lo, hi, np.zeros(3), bad))
    assert np.isfinite(GMM._box_mass(lo, hi, np.zeros(3), np.zeros((3, 3))))
    assert np.isfinite(GMM._box_mass(lo, hi, np.zeros(3), np.diag([1.0, 1.0, -1e-3])))


# Mean 10 sigma below a unit box: the first whitened interval is [9, 11], where both ndtr
# values round to exactly 1.  cov = I factorizes the box mass, so this is exact.
TAIL_MEAN = np.array([-10.0, 0.0, 0.0])
TAIL_MASS = (ndtr(-9.0) - ndtr(-11.0)) * (ndtr(1.0) - ndtr(-1.0)) ** 2   # ~5.26e-20


def _tail_model():
    '''One component whose mean is far outside the box, which sample() draws from happily.

    bounds = [-1,1]^3, so the normalized frame the parameters live in is the physical one.
    '''
    m = GMM.gmm(1, np.column_stack([-np.ones(3), np.ones(3)]))
    m.d = 3
    m.means = [TAIL_MEAN]
    m.covariances = [np.eye(3)]
    m.weights = np.array([1.0])
    m.adapt = [False]
    return m


def test_box_mass_keeps_positive_tail_mass():
    '''Differencing the two saturated CDFs returns zero mass for a box deep in the tail,
    and score() then normalizes by its 1e-300 floor instead.'''
    got = GMM._box_mass(-np.ones(3), np.ones(3), TAIL_MEAN, np.eye(3))
    assert got == pytest.approx(TAIL_MASS, rel=1e-8), (got, TAIL_MASS)


def test_score_normalizes_tail_component():
    '''score() of a tail component is its pdf over its true box mass, not over the floor.'''
    m = _tail_model()
    x = np.array([[0.0, 0.0, 0.0], [-0.9, 0.5, -0.3], [0.95, -0.95, 0.4]])
    expect = multivariate_normal.pdf(x, mean=TAIL_MEAN, cov=np.eye(3)) / TAIL_MASS
    np.testing.assert_allclose(_host(m.score(x)), expect, rtol=1e-8)


def test_tail_component_score_integrates_to_one():
    '''The density score() reports must integrate to 1 over the box -- by ~5e280 the
    wrong factor if the tail mass underflows to the floor.  Deterministic quadrature:
    32-node Gauss-Legendre per axis is far more than this smooth integrand needs.'''
    m = _tail_model()
    nodes, qw = np.polynomial.legendre.leggauss(32)
    grid = np.stack(np.meshgrid(*([nodes] * 3), indexing='ij'), axis=-1).reshape(-1, 3)
    wt = np.prod(np.stack(np.meshgrid(*([qw] * 3), indexing='ij'), axis=-1).reshape(-1, 3),
                 axis=1)
    total = float(wt @ _host(m.score(grid)))
    assert total == pytest.approx(1.0, rel=1e-6), total


def test_box_mass_accuracy_matches_mvnun():
    '''Pin accuracy to mvnun's own at its defaults, so a smaller lattice cannot slip in.

    18 fixed cases, d=3,4,6, against mvnun run to a tight tolerance (within 2e-6 of
    maxpts=1e7).  Measured mean relative error: 2^11 points 1.2e-4, 2^12 9.2e-5.
    '''
    errs = []
    for d in (3, 4, 6):
        for s in range(6):
            rng = np.random.default_rng(1000 * d + s)
            A = rng.normal(size=(d, d))
            cov = 0.1 * A @ A.T / d + 0.02 * np.eye(d)
            mean = rng.uniform(-1.2, 1.2, d)
            lo, hi = -np.ones(d), np.ones(d)
            ref = GMM.mvnun(lo, hi, mean, cov, maxpts=2 * 10**6, abseps=1e-10, releps=1e-8)[0]
            errs.append(abs(GMM._box_mass(lo, hi, mean, cov) / ref - 1))
    assert max(errs) <= 5e-4, max(errs)
    assert np.mean(errs) <= 1.1e-4, np.mean(errs)
