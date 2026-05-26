import numpy as np

from RIFT.misc.distance_grid import (
    DISTANCE_GRID_FIELDS,
    build_distance_grid,
    load_distance_grid,
    reconstruct_marginal_lnL,
    save_distance_grid,
)


def test_distance_grid_reconstructs_marginal_lnL():
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
        n_grid=40,
    )

    assert grid.dtype.names == DISTANCE_GRID_FIELDS
    assert np.all(np.diff(grid["dist"]) >= 0)
    assert np.all(grid["dist_weight"] > 0)
    assert np.isclose(reconstruct_marginal_lnL(grid), lnL_marginal)
    assert np.all(grid["m1"] == 35.0)
    assert np.all(grid["m2"] == 28.0)
    assert np.all(grid["s1z"] == 0.1)
    assert np.all(grid["s2z"] == -0.2)


def test_distance_grid_roundtrip_preserves_reconstruction(tmp_path):
    distance = np.linspace(100.0, 900.0, 21)
    ln_weights = -0.5 * ((distance - 500.0) / 120.0) ** 2
    lnL_marginal = -12.5
    grid = build_distance_grid(distance, ln_weights, lnL_marginal, 0.2, {}, n_grid=10)

    fname = tmp_path / "event_0_.dgrid"
    save_distance_grid(fname, grid)
    loaded = load_distance_grid(fname)

    assert loaded.dtype.names == DISTANCE_GRID_FIELDS
    assert np.isclose(reconstruct_marginal_lnL(loaded), lnL_marginal)


def test_legacy_distance_grid_without_weights_uses_trapezoid():
    dtype = [("lnL", float), ("dist", float)]
    grid = np.zeros(5, dtype=dtype)
    grid["dist"] = np.linspace(0.0, 1.0, 5)
    grid["lnL"] = 0.0

    assert np.isclose(reconstruct_marginal_lnL(grid), 0.0)
