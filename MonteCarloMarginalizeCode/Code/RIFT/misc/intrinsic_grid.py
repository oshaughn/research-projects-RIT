"""Exact intrinsic-grid deduplication, separate from posterior resampling."""
import math

# Extrinsic parameters are marginalized by ordinary ILE and are intentionally
# absent. No rounded comparisons, mass relabeling, or spin rotations are used.
INTRINSIC_FIELDS = ('m1','m2','s1x','s1y','s1z','s2x','s2y','s2z',
                    'lambda1','lambda2','eccentricity','meanPerAno','fref',
                    'E0','p_phi0','a6c')  # every non-extrinsic column the grid XML carries

def unique_intrinsic_indices(points):
    seen, indices = set(), []
    for index, point in enumerate(points):
        key = tuple(float(getattr(point,name)) for name in INTRINSIC_FIELDS)
        if not all(math.isfinite(value) for value in key):
            raise ValueError('Nonfinite intrinsic grid row {}'.format(index))
        if key not in seen:
            seen.add(key)
            indices.append(index)
    return indices

def pad_with_duplicates(indices, n_rows, n_required):
    """Unique rows first, then the input's own duplicate rows (in order) until n_required."""
    unique = set(indices)
    filler = [k for k in range(n_rows) if k not in unique] or list(indices)
    out = list(indices)
    while len(out) < n_required:
        out.append(filler[(len(out) - len(indices)) % len(filler)])
    return out
