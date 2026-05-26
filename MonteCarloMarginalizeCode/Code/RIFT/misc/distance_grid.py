import numpy as np


DISTANCE_GRID_FIELDS = (
    "lnL",
    "sigmaL",
    "m1",
    "m2",
    "s1x",
    "s1y",
    "s1z",
    "s2x",
    "s2y",
    "s2z",
    "lambda1",
    "lambda2",
    "eccentricity",
    "meanPerAno",
    "eos_index",
    "dist",
    "dist_weight",
)


def _logsumexp(vals):
    vals = np.asarray(vals, dtype=float)
    vmax = np.max(vals)
    if not np.isfinite(vmax):
        return vmax
    return vmax + np.log(np.sum(np.exp(vals - vmax)))


def _as_positive_integer(value, default):
    if value is None:
        return default
    value = int(value)
    if value < 1:
        raise ValueError("distance grid size must be positive")
    return value


def _weighted_blocks(distance, probability, n_grid):
    order = np.argsort(distance)
    distance = np.asarray(distance, dtype=float)[order]
    probability = np.asarray(probability, dtype=float)[order]

    finite = np.isfinite(distance) & np.isfinite(probability) & (probability > 0)
    distance = distance[finite]
    probability = probability[finite]
    if len(distance) == 0:
        raise ValueError("no finite positive-weight distance samples to export")

    n_grid = min(_as_positive_integer(n_grid, len(distance)), len(distance))
    blocks = np.array_split(np.arange(len(distance)), n_grid)
    grid_dist = np.empty(len(blocks))
    grid_mass = np.empty(len(blocks))
    for i, block in enumerate(blocks):
        weights = probability[block]
        grid_mass[i] = np.sum(weights)
        grid_dist[i] = np.sum(distance[block] * weights) / grid_mass[i]

    if len(grid_dist) == 1:
        width = np.array([max(np.ptp(distance), np.finfo(float).eps)])
    else:
        edges = np.empty(len(grid_dist) + 1)
        edges[1:-1] = 0.5 * (grid_dist[1:] + grid_dist[:-1])
        edges[0] = min(distance[0], grid_dist[0] - (edges[1] - grid_dist[0]))
        edges[-1] = max(distance[-1], grid_dist[-1] + (grid_dist[-1] - edges[-2]))
        width = np.diff(edges)
        width = np.maximum(width, np.finfo(float).eps)

    return grid_dist, grid_mass, width


def build_distance_grid(distance, ln_weights, lnL_marginal, sigmaL, params, n_grid=None):
    """Build a distance-extended likelihood grid from weighted ILE samples.

    The exported ``lnL`` is a density in luminosity distance. Therefore
    ``sum(exp(lnL) * dist_weight)`` reconstructs the original marginalized
    likelihood for this intrinsic point.
    """
    ln_weights = np.asarray(ln_weights, dtype=float)
    ln_norm = _logsumexp(ln_weights)
    probability = np.exp(ln_weights - ln_norm)
    grid_dist, grid_mass, grid_width = _weighted_blocks(distance, probability, n_grid)

    dtype = [(name, float) for name in DISTANCE_GRID_FIELDS]
    grid = np.zeros(len(grid_dist), dtype=dtype)
    grid["lnL"] = lnL_marginal + np.log(grid_mass) - np.log(grid_width)
    grid["sigmaL"] = sigmaL
    grid["dist"] = grid_dist
    grid["dist_weight"] = grid_width

    for name in DISTANCE_GRID_FIELDS:
        if name in {"lnL", "sigmaL", "dist", "dist_weight"}:
            continue
        grid[name] = float(params.get(name, 0.0))
    return grid


def save_distance_grid(fname, grid):
    header = " ".join(grid.dtype.names)
    np.savetxt(fname, np.column_stack([grid[name] for name in grid.dtype.names]), header=header)


def load_distance_grid(fname):
    return np.genfromtxt(fname, names=True)


def reconstruct_marginal_lnL(grid):
    if "dist_weight" in grid.dtype.names:
        return _logsumexp(grid["lnL"] + np.log(grid["dist_weight"]))
    order = np.argsort(grid["dist"])
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(np.exp(grid["lnL"][order]), grid["dist"][order])
    else:
        integral = np.trapz(np.exp(grid["lnL"][order]), grid["dist"][order])
    return np.log(integral)
