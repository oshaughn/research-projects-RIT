"""Which sub-sample Q_lm stencil should a given run use?

This is a leaf module on purpose: numpy only, no lal, no numba, no cupy.  The pipeline scripts
(bin/helper_LDG_Events.py) need the answer while building a workflow, and importing
factored_likelihood there would cost ~4 s of numba compilation for a ten-line decision.  Keeping
it here also means the thresholds are under a real unit test (test_time_interp_choice.py) rather
than buried in a script that cannot be imported.

THE DECISION.  There is no uniformly better stencil, so the choice is made from the run's own
oversampling factor fNyq/fmax = (srate/2)/fmax.  The two interpolating stencils fail differently:

  'cubic'  4-point Lagrange polynomial.  Error O(h^4): improves FAST with oversampling, poor
           near Nyquist, because a cubic cannot follow the signal there.
  'sinc'   Lanczos windowed sinc, 2a taps (a=8).  Error set by the WINDOW, not by h, so it is
           flat in oversampling: far better than cubic near Nyquist, worse once heavily
           oversampled.

MEASURED accuracy crossover (test_q_window_interp.py's harness, max relative error on a
synthetic band-limited signal; 24 seeds x 8 targets per point; ratio = cubic error / sinc error,
so >1 means sinc wins; "frac" is the fraction of seeds in which sinc wins):

    fNyq/fmax     4.0    4.5    5.0   5.25    5.5   5.75    6.0    6.5    7.0
    ratio (med)  3.52   2.08   1.43   1.23   0.95   0.85   0.75   0.62   0.44
    frac         1.00   1.00   0.92   0.88   0.38   0.08   0.04   0.04   0.00

So the median crossover is fNyq/fmax ~= 5.4, sinc wins in EVERY realization up to 4.5, and
essentially never above 5.75.

WHY THERE ARE TWO THRESHOLDS.  Accuracy is only half the decision; the other half is what the
extra taps cost, and that differs by backend.  Measured cost of sinc relative to cubic in the Q
product: ~4.2-4.5x on CPU, where the window builder is tap-count bound (16 taps against 4), but
only ~1.6-3.0x on GPU, where Q_inner_sinc is bandwidth/latency bound and the extra taps are
largely hidden.  Cost cannot outrank accuracy -- a wrong likelihood is worse than a slow one --
but it is the right tie-breaker through the band where the two stencils are within a few tens of
percent of each other.  Hence:

  GPU  threshold 5.5: sinc is only ~2x the cost, so let ACCURACY decide and put the threshold at
       the measured median crossover.
  CPU  threshold 5.0: sinc is ~4.5x the cost, so only pay it while its advantage is robust
       rather than marginal -- at 5.0 the median gain is still 1.43x and 92% of realizations
       favour sinc; past that the gain is a coin flip and the 4.5x is not worth it.

The gap is deliberately small, and that is itself the finding: the accuracy curves are steep
through the crossover, so a 2x difference in cost moves the optimum by only ~0.5 in fNyq/fmax.
Do not widen it without re-measuring -- these are measured numbers, not taste.

Typical production -- srate 4096 with fmax 1700 -- is fNyq/fmax ~ 1.2, deep in sinc's regime,
where sinc is 35-50x more accurate.  Both thresholds select sinc there, so the backend
distinction does not change the production answer; it matters only for the oversampled
configurations near the crossover.  A heavily oversampled configuration (the slow-rotation
brute-force tests run fmax 512 at srate 16384, i.e. 16) gets cubic on either backend.
"""
from __future__ import division

import numpy as np

# See the module docstring for the measurement behind each of these.
INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU = 5.0
INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU = 5.5

# The stencils this tree knows about.  Kept here rather than imported from
# factored_likelihood so the pipeline can validate a user's spelling without paying ~4 s of
# numba compilation; factored_likelihood.TIME_INTERP_CHOICES must agree, and
# test_time_interp_choice asserts that it does.
TIME_INTERP_CHOICES = ('nearest', 'cubic', 'sinc')

# integrate_likelihood_extrinsic_batchmode's own --srate default.  DUPLICATED ON PURPOSE and
# therefore a drift risk: the pipeline has to know what sampling rate the ILE will use when the
# helper does NOT emit --srate, and the driver is a script that cannot be imported.
# test_time_interp_choice reads the value back out of the driver source and fails if the two
# disagree, so the duplication cannot rot silently.
ILE_DEFAULT_SRATE = 16384

# Back-compatible alias: the CPU value is the conservative one.
INTERP_TIME_OVERSAMPLING_THRESHOLD = INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU

# Values of --internal-ile-interpolate-time that mean "choose for me" rather than naming a
# stencil.  'True' is the legacy spelling: before automatic selection existed, the helper
# appended a literal '--interpolate-time True', which the ILE driver read as 'cubic'.
AUTO_REQUEST_TOKENS = ('true', '1', 'yes', 'auto')

# ...and the values that mean "don't interpolate at all".  These matter because the flag now
# takes a VALUE: '--internal-ile-interpolate-time False' passes the STRING 'False', which is
# truthy in Python, so without this it would sail past an `if opts...:` guard and then be
# rejected as an unknown stencil name.  The flag reads like a boolean, so the boolean spellings
# have to work.
OFF_REQUEST_TOKENS = ('false', '0', 'no', 'off', 'none')


def interp_time_threshold(on_gpu=False):
    """The oversampling threshold that applies on this backend."""
    return (INTERP_TIME_OVERSAMPLING_THRESHOLD_GPU if on_gpu
            else INTERP_TIME_OVERSAMPLING_THRESHOLD_CPU)


def choose_time_interp_stencil(srate, fmax, on_gpu=False):
    """Return (stencil, oversampling, threshold) for a run at this srate, fmax and backend.

    stencil is 'sinc' below the backend's threshold and 'cubic' at or above it.  oversampling is
    fNyq/fmax, or None if the inputs were unusable -- in which case the stencil falls back to
    'cubic', the long-standing default, so a missing or malformed srate/fmax can never silently
    select the more expensive stencil.

    on_gpu should reflect whether the ILE job will actually run with --gpu, because the cost of
    the extra taps -- and therefore where cost should break the tie -- differs by roughly 2x
    between the backends.  See the module docstring.
    """
    threshold = interp_time_threshold(on_gpu)
    try:
        oversampling = (float(srate) / 2.0) / float(fmax)
    except (TypeError, ValueError, ZeroDivisionError):
        return 'cubic', None, threshold
    if not np.isfinite(oversampling) or oversampling <= 0:
        return 'cubic', None, threshold
    return ('sinc' if oversampling < threshold else 'cubic'), oversampling, threshold


def is_auto_request(value):
    """True if this --internal-ile-interpolate-time value asks for automatic selection."""
    return str(value).strip().lower() in AUTO_REQUEST_TOKENS


def is_off_request(value):
    """True if this --internal-ile-interpolate-time value means "disabled"."""
    return str(value).strip().lower() in OFF_REQUEST_TOKENS


def validate_stencil_name(value):
    """Return the canonical stencil name, or raise ValueError.

    The pipeline calls this so a misspelled stencil fails while the workflow is being BUILT.
    Without it the bad name rides onto every generated ILE command line and each job dies
    separately at run time, after submission -- the cheapest possible error made expensive.
    """
    name = str(value).strip().lower()
    if name not in TIME_INTERP_CHOICES:
        raise ValueError(
            "unrecognised Q_lm time-interpolation stencil %r: expected one of %s, or a value "
            "meaning automatic selection (%s)"
            % (value, "|".join(TIME_INTERP_CHOICES), "|".join(AUTO_REQUEST_TOKENS)))
    return name


def effective_srate_for_stencil(srate_helper, srate_internal=None, helper_emits_srate=True):
    """The sampling rate the Q_lm series the stencil interpolates is ACTUALLY on.

    This is deliberately not just the pipeline's `srate`, because two things move it:

      * ``--srate-internal`` re-samples the data the likelihood works on
        (integrate_likelihood_extrinsic_batchmode sets ``deltaT = deltaT_internal``), so when it
        is set it -- not ``--srate`` -- is the grid the stencil steps along.  It is appended to
        the ILE command line by util_RIFT_pseudo_pipe.py without passing through the helper, so
        the helper has to be told about it explicitly.
      * if the helper does not emit ``--srate`` at all, the ILE falls back to its own default
        (ILE_DEFAULT_SRATE), which is 4x the pipeline's usual 4096.

    Getting this wrong does not corrupt anything -- it just picks the stencil using a number the
    run never uses, which is exactly the sort of error that never announces itself.
    """
    if srate_internal:
        return float(srate_internal)
    if helper_emits_srate:
        return srate_helper
    return float(ILE_DEFAULT_SRATE)
