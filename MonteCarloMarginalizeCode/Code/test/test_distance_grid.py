import numpy as np

from RIFT.misc.distance_grid import (
    DISTANCE_GRID_FIELDS,
    build_distance_grid,
    load_distance_grid,
    reconstruct_marginal_lnL,
    _logsumexp,
)


def _volumetric_log_prior(d, d_min=1.0, d_max=4000.0):
    norm = (d_max**3 - d_min**3) / 3.0
    return 2.0*np.log(d) - np.log(norm)


def test_distance_grid_reconstructs_marginal_lnL_with_sampling_prior():
    rng = np.random.default_rng(1234)
    distance = rng.lognormal(mean=np.log(450.0), sigma=0.22, size=200)
    ln_weights = -0.5 * ((distance - 430.0) / 35.0) ** 2
    lnL_marginal = 37.25

    grid = build_distance_grid(
        distance,
        ln_weights,
        lnL_marginal,
        sigmaL=0.012,
        params={"m1": 35.0, "m2": 28.0, "s1z": 0.1, "s2z": -0.2},
        ln_prior_d_at_samples=_volumetric_log_prior(distance),
        n_grid=40,
    )

    assert grid.dtype.names == DISTANCE_GRID_FIELDS
    assert np.all(np.diff(grid["dist"]) >= 0)
    assert np.all(grid["dist_weight"] > 0)
    assert np.isclose(reconstruct_marginal_lnL(grid), lnL_marginal)
    assert np.all(grid["m1"] == 35.0)
    assert np.all(grid["s1z"] == 0.1)


def test_distance_grid_roundtrip_preserves_reconstruction(tmp_path):
    distance = np.linspace(100.0, 900.0, 21)
    ln_weights = -0.5 * ((distance - 500.0) / 120.0) ** 2
    lnL_marginal = -12.5
    grid = build_distance_grid(
        distance, ln_weights, lnL_marginal, 0.2, {},
        ln_prior_d_at_samples=_volumetric_log_prior(distance),
        n_grid=10,
    )

    from pathlib import Path
    fname = Path(str(tmp_path)) / "event_0_.dgrid"
    from RIFT.misc.distance_grid import save_distance_grid
    save_distance_grid(fname, grid)
    loaded = load_distance_grid(fname)

    assert loaded.dtype.names == DISTANCE_GRID_FIELDS
    assert np.isclose(reconstruct_marginal_lnL(loaded), lnL_marginal)


def test_exported_lnL_is_pure_likelihood():
    """exp(lnL) is the pure extrinsic-marginalized likelihood density in d;
    integrating it against a different distance prior gives a different
    marginal."""
    rng = np.random.default_rng(7)
    n = 4000
    d_min, d_max = 100.0, 1500.0
    distance = rng.uniform(d_min, d_max, size=n)
    ln_L_pure = -0.5 * ((distance - 600.0)/80.0)**2 + 5.0
    ln_pi = _volumetric_log_prior(distance, d_min, d_max)
    ln_q = -np.log(d_max - d_min)
    ln_w = ln_L_pure + ln_pi - ln_q
    lnL_marg_mc = _logsumexp(ln_w) - np.log(n)
    grid = build_distance_grid(distance, ln_w, lnL_marg_mc, 0.0, {},
                               ln_prior_d_at_samples=ln_pi, n_grid=40)
    # default reconstruction matches the original marginal
    assert np.isclose(reconstruct_marginal_lnL(grid), lnL_marg_mc)
    # reconstruction with a flat-in-d prior gives the pure-likelihood integral
    flat_log_prior = lambda d: np.full_like(np.asarray(d, float), -np.log(d_max - d_min))
    lnL_flat = reconstruct_marginal_lnL(grid, ln_prior_d=flat_log_prior)
    expected = np.log(np.sqrt(2*np.pi)*80.0*np.exp(5.0)/(d_max - d_min))
    assert abs(lnL_flat - expected) < 0.1, (lnL_flat, expected)
    # and is meaningfully different from the volumetric answer
    # closed-form ratio at d~600 over [100,1500]: log(d^2*(d_max-d_min)*3/(d_max^3-d_min^3))
    expected_ratio = np.log(600.0**2 * (d_max-d_min) * 3.0 / (d_max**3 - d_min**3))
    assert abs((lnL_flat - lnL_marg_mc) - (-expected_ratio)) < 0.1


def test_legacy_distance_grid_without_weights_uses_trapezoid():
    dtype = [("lnL", float), ("dist", float)]
    grid = np.zeros(5, dtype=dtype)
    grid["dist"] = np.linspace(0.0, 1.0, 5)
    grid["lnL"] = 0.0

    assert np.isclose(reconstruct_marginal_lnL(grid), 0.0)
