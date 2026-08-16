"""Which sub-sample Q_lm stencil should a given run use?

This is a leaf module on purpose: numpy only, no lal, no numba, no cupy.  The pipeline scripts
(bin/helper_LDG_Events.py) need the answer while building a workflow, and importing
factored_likelihood there would cost ~4 s of numba compilation for a ten-line decision.  Keeping
it here also means the threshold is under a real unit test (test_time_interp_choice.py) rather
than buried in a script that cannot be imported.

THE DECISION.  There is no uniformly better stencil, so the choice is made from the run's own
oversampling factor fNyq/fmax = (srate/2)/fmax.  The two interpolating stencils fail differently:

  'cubic'  4-point Lagrange polynomial.  Error O(h^4): improves FAST with oversampling, poor
           near Nyquist, because a cubic cannot follow the signal there.
  'sinc'   Lanczos windowed sinc, 2a taps (a=8).  Error set by the WINDOW, not by h, so it is
           flat in oversampling: far better than cubic near Nyquist, worse once heavily
           oversampled.

MEASURED crossover (test_q_window_interp.py, max relative error on a synthetic band-limited
signal; medians over 12 seeds; ratio = cubic error / sinc error, so >1 means sinc wins):

    fNyq/fmax      3      4     4.5      5     5.5      6      7      8
    ratio       10.4    3.4     2.3    1.4    0.91   0.77   0.50   0.26

The crossover therefore sits at fNyq/fmax ~= 5.3, and the seed-to-seed spread brackets 1.0 only
over 5-6.  The threshold below is placed at 5, i.e. deliberately on the CUBIC side of the
measured crossover: through the ambiguous 5-6 band the two errors are within ~30% of each other,
while sinc costs ~4x cubic in the Q product (16 taps against 4), so there the cheaper incumbent
should win.  Do not move this without re-measuring -- it is a measured number, not a taste.

Typical production -- srate 4096 with fmax 1700 -- is fNyq/fmax ~ 1.2, deep in sinc's regime,
where sinc is 35-50x more accurate.  A heavily oversampled configuration (the slow-rotation
brute-force tests run fmax 512 at srate 16384, i.e. 16) correctly gets cubic.
"""
from __future__ import division

import numpy as np

INTERP_TIME_OVERSAMPLING_THRESHOLD = 5.0

# Values of --internal-ile-interpolate-time that mean "choose for me" rather than naming a
# stencil.  'True' is the legacy spelling: before automatic selection existed, the helper
# appended a literal '--interpolate-time True', which the ILE driver read as 'cubic'.
AUTO_REQUEST_TOKENS = ('true', '1', 'yes', 'auto')


def choose_time_interp_stencil(srate, fmax):
    """Return (stencil, oversampling) for a run at this sample rate and maximum frequency.

    stencil is 'sinc' below INTERP_TIME_OVERSAMPLING_THRESHOLD and 'cubic' at or above it.
    oversampling is fNyq/fmax, or None if the inputs were unusable -- in which case the stencil
    falls back to 'cubic', the long-standing default, so a missing or malformed srate/fmax can
    never silently select the more expensive stencil.
    """
    try:
        oversampling = (float(srate) / 2.0) / float(fmax)
    except (TypeError, ValueError, ZeroDivisionError):
        return 'cubic', None
    if not np.isfinite(oversampling) or oversampling <= 0:
        return 'cubic', None
    return ('sinc' if oversampling < INTERP_TIME_OVERSAMPLING_THRESHOLD else 'cubic'), oversampling


def is_auto_request(value):
    """True if this --internal-ile-interpolate-time value asks for automatic selection."""
    return str(value).strip().lower() in AUTO_REQUEST_TOKENS
