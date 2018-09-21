import numpy as np
import numpy

def histogram(samples, n_bins, xpy=numpy):
    n_samples = samples.size

    # Compute the histogram counts.
    indices = xpy.trunc(samples * n_bins).astype(np.int32)
    histogram_counts = xpy.bincount(
        indices, minlength=n_bins,
        weights=xpy.broadcast_to(
            xpy.asarray([float(n_bins)/n_samples]),
            (n_samples,),
        ),
    )
    return histogram_counts
