"""LadderClimbingOracle: finds a narrow correlated peak and follows a ridge along the ladder."""
import numpy as np
import pytest

from RIFT.integrators.unreliable_oracle import LadderClimbingOracle


def _toy_lnL(a, b, c, phi):
    # narrow, strongly correlated (a, b); the (a, b) optimum shifts with c (the
    # ladder coordinate); phi couples to c, so the right phi matters only at large c.
    a0 = 0.3 + 0.2 * c
    u = (a - a0) + (b - 0.1)
    v = (a - a0) - (b - 0.1)
    return (-0.5 * (u / 0.003) ** 2 - 0.5 * (v / 0.03) ** 2
            - 0.5 * ((c - 0.2) / 0.3) ** 2 + 20 * c * np.cos(phi - 1.0))


def _oracle(seed):
    o = LadderClimbingOracle()
    o.add_parameter("a", left_limit=-1.0, right_limit=1.0)
    o.add_parameter("b", left_limit=-1.0, right_limit=1.0)
    o.add_parameter("c", left_limit=0.0, right_limit=1.0)
    o.add_parameter("phi", left_limit=0.0, right_limit=2 * np.pi)
    o.setup(lnL=_toy_lnL, periodic=["phi"], ladder_param="c",
            ladder_values=[0.2, 0.5, 0.8], popsize=12, n_generations=60, seed=seed)
    return o


def test_ladder_follows_ridge():
    for seed in (1, 2, 3):
        o = _oracle(seed)
        o.run(np.array([0.0, 0.0, 0.1, 4.0]), np.array([0.1, 0.1, 0.25, 1.5]))
        for v in (0.5, 0.8):
            rec = o.climbs["c=%.3g" % v]
            x = rec["best_x"]
            assert x[2] == v                                   # pinned exactly
            assert abs(x[0] - (0.3 + 0.2 * v)) < 0.02, (seed, v, x)
            assert abs(np.angle(np.exp(1j * (x[3] - 1.0)))) < 0.3, (seed, v, x)
            lnl_opt = _toy_lnL(0.3 + 0.2 * v, 0.1, v, 1.0)
            assert rec["best_lnL"] > lnl_opt - 0.5, (seed, v, rec["best_lnL"], lnl_opt)


def test_bounds_reflect_and_wrap():
    o = _oracle(0)
    X = o._fold(np.array([[1.3, -1.2, 1.4, 7.0], [-3.1, 0.0, -0.2, -1.0]]))  # -3.1 -> 1.1 -> 0.9
    assert np.allclose(X[0], [0.7, -0.8, 0.6, 7.0 - 2 * np.pi])
    assert np.allclose(X[1], [0.9, 0.0, 0.2, 2 * np.pi - 1.0])


def test_draws_stay_in_box():
    o = _oracle(4)
    o.run(np.array([0.0, 0.0, 0.1, 4.0]), np.array([0.1, 0.1, 0.25, 1.5]))
    _, _, rv = o.draw_simplified(2000)
    assert rv.shape == (2000, 4)
    assert np.all((rv[:, 2] >= 0) & (rv[:, 2] <= 1))
    assert np.all((rv[:, 3] >= 0) & (rv[:, 3] < 2 * np.pi))


def test_covariance_placement_shares_and_spreads():
    o = _oracle(5)
    o.run(np.array([0.0, 0.0, 0.1, 4.0]), np.array([0.1, 0.1, 0.25, 1.5]))
    X = o.place(400, delta_lnL=1e9, pinned_sd=0.1)
    assert X.shape == (400, 4)
    assert np.all((X[:, 2] >= 0) & (X[:, 2] <= 1)) and np.all((X[:, 3] >= 0) & (X[:, 3] < 2 * np.pi))
    # rungs at c = 0.5 and 0.8 are spread (pinned_sd), not collapsed onto the rung value
    near = lambda v: np.sum(np.abs(X[:, 2] - v) < 0.25)
    assert near(0.5) > 40 and near(0.8) > 40
    assert len(np.unique(np.round(X[:, 2], 6))) > 300


def test_free_climb_uses_corrected_callback_start():
    """Constraint projection must affect the optimizer as well as its fiducial."""
    o = LadderClimbingOracle()
    o.add_parameter("x", left_limit=-1., right_limit=1.)
    o.setup(lnL=lambda x: -(x-.7)**2, n_generations=1, popsize=4, seed=1)
    o.run(np.array([-.7]), np.array([.001]),
          between_climbs=lambda x: np.array([.7]))
    assert np.all(np.abs(o.reference_samples[:, 0] - .7) < .002)


@pytest.mark.parametrize("step_scale", [1.5, 50.])
def test_periodic_climb_is_invariant_under_rotation_across_seam(step_scale):
    """Wrapping an equivalent chart must not change steps or placement width."""
    results = []
    histories = []
    placements = []
    for target in (.03, 3.):
        o = LadderClimbingOracle()
        o.add_parameter("phi", left_limit=0., right_limit=2*np.pi)
        o.setup(lnL=lambda x: -.5*(np.angle(np.exp(1j*(x-target)))/.01)**2,
                n_generations=30, popsize=10, seed=4, periodic=["phi"])
        start = np.mod(target - (.03 + 2*np.pi - 6.2), 2*np.pi)
        results.append(o.climb(np.array([start]), np.array([step_scale])))
        histories.append(np.array([np.angle(np.exp(1j*(h[1][0]-target))) for h in o.history]))
        placed = o.place(100)
        assert np.all((placed >= 0) & (placed < 2*np.pi))
        placements.append(np.angle(np.exp(1j*(placed[:, 0]-target))))
    np.testing.assert_allclose(histories[0], histories[1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(placements[0], placements[1], atol=1e-8, rtol=0)
    np.testing.assert_allclose(results[0]["cov"], results[1]["cov"], atol=1e-20, rtol=1e-6)
    if step_scale == 1.5:
        assert results[0]["cov"][0, 0] < 1e-10
