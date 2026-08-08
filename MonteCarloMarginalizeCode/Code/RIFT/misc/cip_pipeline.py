"""Helpers for the CIP posterior export draw and iteration-specific CIP arguments.

The CIP posterior export draws indices from the weighted sample cache.  A
weighted numpy draw without replacement is *successive sampling* (indices
returned in draw order, head enriched in high-weight points), which biased the
export at any output size once the head was truncated.  The replacement here is
systematic (stratified) resampling on the weight CDF: expected counts are
exactly N*p_i at any N, duplication is the minimum possible, and the draw is
duplicate-free whenever N <= sum(w)/max(w).  That sum/max bound (not the Kish
ESS) is the exact frontier of the fair-AND-unique region: no fair draw of size
N > sum(w)/max(w) can avoid duplicates.
"""

import numpy as np

POSTERIOR_UNIQUE_FLAG = "--posterior-unique-draw"


def unique_draw_bound(weights):
    """Largest fair draw size that can be duplicate-free: floor(sum(w)/max(w))."""
    w = np.asarray(weights, dtype=float)
    return int(np.floor(np.sum(w) / np.max(w)))


def systematic_resample(weights, n_out, rng=None):
    """Systematic (stratified) resample: n_out indices drawn ~ weights.

    Expected counts are exactly n_out*w_i/sum(w) for every i, at any n_out
    (unlike weighted choice(replace=False)), and each count is at most
    ceil(n_out*w_i/sum(w)) so the draw has no duplicates when
    n_out <= sum(w)/max(w).  The returned order is shuffled, so any
    contiguous truncation of the result is itself a fair draw.

    rng defaults to the legacy global numpy generator, matching the rest of
    CIP (and old numpy on clusters without default_rng).
    """
    if rng is None:
        rng = np.random
    w = np.asarray(weights, dtype=float)
    cdf = np.cumsum(w / np.sum(w))
    cdf[-1] = 1.0  # guard against roundoff excluding the final bin
    positions = (rng.uniform() + np.arange(n_out)) / n_out
    indx = np.searchsorted(cdf, positions, side='left')
    rng.shuffle(indx)
    return indx


def flag_final_group_unique(lines):
    """Add the unique-draw flag to the final CIP argument-group line.

    CIP argument files group repeated iterations by prefixing each line with a
    count, ``G<count>`` (Gaussian-resampling executable), or ``Z`` (terminal
    run-to-convergence subdag).  Internal iterations keep CIP's default draw
    (fair, duplicates possible); only the final group -- the product consumed
    downstream -- gets the unique-draw cap.  The flag goes on the whole final
    group, not just its last iteration, so a convergence-test abort partway
    through the group still publishes a unique fair draw.

    A final ``G`` line is left untouched: the Gaussian-resampling executable
    does not accept the flag (strict argparse), so flagging it would kill the
    job.  Callers should avoid ending the schedule with a G group if they need
    the uniqueness guarantee.
    """
    lines = [line.rstrip() for line in lines if line.strip()]
    if not lines:
        return []
    final_line = lines[-1]
    prefix = final_line.split()[0]
    if prefix.startswith("G") or POSTERIOR_UNIQUE_FLAG in final_line.split():
        return lines
    lines[-1] = "{} {}".format(final_line, POSTERIOR_UNIQUE_FLAG)
    return lines
