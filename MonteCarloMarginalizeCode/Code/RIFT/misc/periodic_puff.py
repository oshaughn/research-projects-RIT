"""Periodic exclusion for a covariance puff with several angular coordinates."""
import itertools
import numpy as np
from scipy.spatial.distance import cdist


def outside_neighborhood(existing, proposed, inverse_covariance, columns, periods, radius, block_size=64):
    """Reject exactly the points within radius of any integer-period lift.

    Complete the square in the angular lattice. Its continuous minimum filters
    pairs before enumerating the bounded integer ellipsoid. Cross-covariances
    mean independently wrapping coordinate differences is not sufficient.
    """
    existing, proposed = np.asarray(existing), np.asarray(proposed)
    vi = np.asarray(inverse_covariance)
    basis = np.zeros((vi.shape[0], len(columns)))
    basis[columns, np.arange(len(columns))] = periods
    angular = basis.T @ vi @ basis
    if np.min(np.linalg.eigvalsh(angular)) <= 0:
        raise ValueError('Periodic force-away requires a positive definite angular metric')
    angular_inverse = np.linalg.inv(angular)
    # The continuous minimum depends only on non-angular coordinates.
    # Search in that projected space first, so the integer work is restricted
    # to close pairs rather than materializing full coordinate differences.
    linear = [i for i in range(vi.shape[0]) if i not in columns]
    vb = vi @ basis
    schur = vi - vb @ angular_inverse @ vb.T
    reduced = schur[np.ix_(linear, linear)]
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    transform = eigenvectors * np.sqrt(np.maximum(0., eigenvalues))
    existing_linear = existing[:, linear] @ transform
    proposed_linear = proposed[:, linear] @ transform
    outside = np.ones(len(proposed), dtype=bool)
    r2 = radius**2
    for start in range(0, len(proposed), block_size):
        minimum = cdist(existing_linear, proposed_linear[start:start+block_size], metric='sqeuclidean')
        for i, j in np.argwhere(minimum <= r2 + 1e-10):
            if not outside[start+j]:
                continue
            difference = existing[i] - proposed[start+j]
            optimum = -(difference @ vb) @ angular_inverse
            # Every point inside the ellipsoid lies within these axis bounds.
            width = np.sqrt(np.maximum(0., r2-minimum[i,j]) * np.diag(angular_inverse))
            low = np.ceil(optimum-width-1e-10).astype(int)
            high = np.floor(optimum+width+1e-10).astype(int)
            for lift in itertools.product(*(range(a,b+1) for a,b in zip(low,high))):
                # Evaluate the original form to avoid cancellation at the boundary.
                delta = difference + basis @ lift
                if delta @ vi @ delta <= r2:
                    outside[start+j] = False
                    break
    return outside
