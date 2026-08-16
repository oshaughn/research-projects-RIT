"""Which sub-sample Q_lm stencil should a run use?  Measured guidance -- and why the pipeline
does NOT decide for you.

Leaf module on purpose: numpy only, no lal, no numba, no cupy, so the pipeline scripts can
import it without paying ~4 s of numba compilation.

THERE IS NO AUTOMATIC SELECTION HERE, AND THAT IS A MEASURED CONCLUSION, NOT AN OMISSION.
Two successive attempts were made and both were disproved by measurement:

  1. Select from fNyq/fmax.  WRONG: that number is identical for every system at fixed settings,
     but the right stencil is not.  Q^a_lm(t) = <h_lm(t)|d> is band-limited by whichever is
     lower, fmax or the TEMPLATE's own highest frequency.
  2. Select from fNyq / (fmax bounded by f_ISCO(M_total)).  ALSO WRONG: mis-selected at 2 of 8
     measured masses, and -- fatally -- the correct stencil depends on **fmin** as strongly as on
     mass.  At M = 5 Msun, srate 4096 / fmax 1700, the winner flips from cubic (fmin 30) to sinc
     (fmin 150) with mass, srate and fmax all identical.  The two cases require disjoint
     threshold ranges, (1.21, 2.33) and (2.33, 4.66), so NO threshold can make a
     (srate, fmax, mass) signature correct.

So the flag takes an explicit stencil name.  A wrong automatic choice here is silent -- it does
not raise, it just makes the likelihood less accurate -- which is exactly the kind of error that
should not be guessed at.

===============================================================================================
MEASURED GUIDANCE -- use this to choose
===============================================================================================

All against an exact FFT-zero-padded reference; paired, K=2000, 3 seeds; each mass normalised to
SNR_lik = 100.  Numbers are max|dlnL| in nats.  srate 4096, fmax 1700, fmin 30, Lmax 2.

    M/Msun    nearest      cubic       sinc     winner
      2.6       295        6.92       3.20      SINC  (2.2x)
      5         479        4.34       6.67      cubic (1.5x)
     10         228        0.544      3.02      cubic (5.6x)
     20         295        0.098      2.95      cubic (30x)
     35         333        0.033      5.43      cubic (163x)
     55         333        0.016      4.74      cubic (295x)
     80         338        0.013      5.32      cubic (414x)
    120        4262       <0.337     60.3       cubic (>179x)

RULE OF THUMB: 'cubic' is right for essentially all binaries above ~4 Msun total.  'sinc' pays
off only for genuinely broadband Q -- low total mass, and/or a high fmin that cuts the long
low-frequency inspiral out of the band.  The crossover in total mass is ~3-4 Msun at fmin 30,
and moves UP with fmin (at fmin 150, sinc still wins at M = 5).

'nearest' is never competitive: it is 2-4 orders of magnitude worse everywhere and crosses 1 nat
of error at SNR 2-6, i.e. it is already unusable at O4 SNRs.

WHAT ACTUALLY SETS THE ANSWER is fNyq divided by the true Q bandwidth.  Scoring 12 measured
(mass, fmin) points that way, a single threshold near 4.2 separates every one of them: sinc wins
below ~4.1, cubic above ~4.4.  The concept is sound; what is missing is a good enough estimator
of the bandwidth at workflow-build time.  f_ISCO is not one -- it drifts by 7.4x across
2.6-120 Msun AND the drift reverses sign (over-predicting the bandwidth by 3.4x at M = 2.6,
under-predicting by 2.2x at M = 120), so it biases toward sinc exactly where the decision is
close.  A PSD-weighted high-frequency quantile of |h|^2/S over [fmin, fmax] is computable from
what the pipeline already has and is the obvious next attempt.

ERROR GROWS AS SNR^2 (measured: fitted exponent 1.999-2.006 over two decades), so a stencil that
looks harmless today matters at 3G sensitivities.  SNR at which each stencil's error first
reaches 1 nat: nearest 2-6; cubic 15 (1.3+1.3 Msun) to 830 (30+25); sinc 36-46.

COST, measured: sinc is ~4.2-4.5x cubic on CPU (16 taps against 4; that path is tap-count bound)
but only ~1.6-3.0x on GPU (bandwidth bound).  End-to-end on CPU at fixed n_max: nearest 9.3 s,
cubic 25.1 s, sinc 85.3 s.  On GPU the difference is not resolvable in wall time.

STANDING LIMITATIONS of the measurements above: zero noise, analytic ZDHP PSD, Lmax 2, TaylorT4
(no merger-ringdown -- the high-mass rows' above-f_ISCO content is termination ringing from the
approximant, not physics, so the high-mass end deserves an IMR check), equal mass except 2.6,
non-spinning, one sky location, 3 seeds.
"""
from __future__ import division

# The stencils this tree knows about.  Kept here rather than imported from factored_likelihood so
# the pipeline can validate a spelling without paying numba's import cost;
# factored_likelihood.TIME_INTERP_CHOICES must agree and test_time_interp_choice asserts it does.
TIME_INTERP_CHOICES = ('nearest', 'cubic', 'sinc')

# Values of --internal-ile-interpolate-time that mean "don't interpolate at all".  These matter
# because the flag takes a VALUE: '--internal-ile-interpolate-time False' passes the STRING
# 'False', which is truthy in Python, so without this it would sail past an `if opts...:` guard
# and then be rejected as an unknown stencil name.
OFF_REQUEST_TOKENS = ('false', '0', 'no', 'off', 'none')

# Spellings that USED to mean "choose automatically", back when this module tried to.  They are
# now rejected with a pointer to the guidance above, rather than silently resolved to some
# default -- a run whose stencil was picked by a rule that no longer exists should not start.
RETIRED_AUTO_TOKENS = ('true', '1', 'yes', 'auto')


def is_off_request(value):
    """True if this --internal-ile-interpolate-time value means "disabled"."""
    return str(value).strip().lower() in OFF_REQUEST_TOKENS


def is_retired_auto_request(value):
    """True if this value is one of the retired "choose for me" spellings."""
    return str(value).strip().lower() in RETIRED_AUTO_TOKENS


def validate_stencil_name(value):
    """Return the canonical stencil name, or raise ValueError.

    The pipeline calls this so a bad value fails while the workflow is being BUILT, rather than
    riding onto every generated ILE command line and killing each job separately after
    submission.
    """
    name = str(value).strip().lower()
    if name in TIME_INTERP_CHOICES:
        return name
    if is_retired_auto_request(value):
        raise ValueError(
            "--internal-ile-interpolate-time %r asked for automatic stencil selection, which has "
            "been REMOVED: it was measured to pick the worse stencil at 2 of 8 total masses, and "
            "the correct choice additionally depends on fmin, which no (srate, fmax, mass) rule "
            "can see. Pass an explicit stencil instead -- 'cubic' is right for essentially all "
            "binaries above ~4 Msun total; 'sinc' only for genuinely broadband Q (low total mass, "
            "or a high fmin). See RIFT.likelihood.time_interp_choice for the measured table."
            % (value,))
    raise ValueError(
        "unrecognised Q_lm time-interpolation stencil %r: expected one of %s, or a value meaning "
        "disabled (%s)"
        % (value, "|".join(TIME_INTERP_CHOICES), "|".join(OFF_REQUEST_TOKENS)))
