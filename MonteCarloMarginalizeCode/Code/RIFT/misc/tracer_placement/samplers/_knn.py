"""Tiny numpy-only kNN utilities (no scipy)."""
import numpy as np

def pairwise_sqdist(A, B):
    # (na, d) x (nb, d) -> (na, nb)
    AA = np.sum(A*A, axis=1, keepdims=True)
    BB = np.sum(B*B, axis=1, keepdims=True).T
    return np.maximum(AA + BB - 2.0 * A @ B.T, 0.0)

def knn_dist(X, Y, k):
    """For each row of X, distance to its k-th nearest neighbor in Y."""
    d2 = pairwise_sqdist(X, Y)
    # k-th smallest including self if X is Y at the same row (caller handles +1)
    part = np.partition(d2, k-1, axis=1)[:, :k]
    return np.sqrt(part.max(axis=1))
