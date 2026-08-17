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

Measured with **SEOBNRv4** (an IMR model).  An earlier version of this table used TaylorT4, which
terminates at ISCO and carries no merger-ringdown; it named the WRONG STENCIL at M = 9, 10 and 20
and overstated cubic's high-mass margins by up to 99x.  Do not reintroduce inspiral-only numbers
here.

All against an exact FFT-zero-padded reference; paired, K=2000, 3 seeds; each mass normalised to
SNR_lik = 100.  srate 4096, fmax 1700, fmin 30, Lmax 2.  max|dlnL| in nats:

    M/Msun    nearest      cubic       sinc     winner
      9         369        8.70       3.90      SINC  (2.2x)
     10         286       12.2        4.11      SINC  (3.0x)
     20         284        7.85       3.65      SINC  (2.2x)
     35         200        1.67       3.51      cubic (2.1x)
     55         443        1.31       3.88      cubic (3.0x)
     80         437        0.346      3.15      cubic (9.1x)
    120         433        0.143      7.89      cubic (55x)

and at srate 16384 (SEOBNRv4 cannot be generated at 4096 below M ~ 8):

      5                                          cubic (21x)
      2.6                                        cubic (34x)

FMIN SWEEP, same method, 20 points, 3 seeds each, marginal winners replicated with 3 fresh seeds
(all 12 identical).  Winner and margin; srate 4096, fmax 1700 throughout:

    M \ fmin      20          30          50          100         150
      9        sinc 2.1x   sinc 2.2x   sinc 2.5x   sinc 6.1x   sinc 12.4x
     20        sinc 1.8x   sinc 2.2x   sinc 2.9x   sinc 8.6x   sinc 15.9x
     35        cubic 2.3x  cubic 2.1x  cubic 1.7x  SINC 2.5x   SINC 5.6x
     55        cubic 2.4x  cubic 3.0x  cubic 4.4x  cubic 1.1x  SINC 1.2x

(capitals mark where the fmin-blind rule named the worse stencil).

RULE OF THUMB, and it is TWO-DIMENSIONAL -- fmin matters as much as mass.  The crossover in
total mass RISES with fmin:

    fmin <= 50 Hz    crossover 20-35 Msun    'sinc' below it, 'cubic' above
    fmin  = 100 Hz   crossover 35-55 Msun    'sinc' below it, 'cubic' above
    fmin  = 150 Hz   sinc wins at every mass MEASURED (9-55); crossover is above 55

MEASURED RANGE: 9-55 Msun.  The fmin sweep does NOT cover 80 or 120 Msun, so there is no
high-fmin evidence at those masses -- the fmin-30 ladder puts them firmly in cubic's regime and
nothing here contradicts that.  Do not read the fmin-150 row as "sinc at any mass"; it is "sinc
everywhere we looked, and we stopped at 55".

If you want one conservative rule INSIDE the measured range rather than a boundary: over
fmin >= 100 and M <= 55, always choosing sinc costs at most 1.12x (at M=55, fmin=100, the single
point where cubic still wins), against 5.58x for always choosing cubic.  That asymmetry is why a
flat "prefer sinc" is defensible there -- but it is bounded by the measurement, not universal.

An earlier revision of this file gave only the fmin <= 50 line and it was measurably wrong at
high fmin: it named the worse stencil at (M=35, fmin=100) by 2.5x, (M=35, fmin=150) by **5.6x**,
and (M=55, fmin=150) by 1.2x.

THE MECHANISM, and it is the same property that makes sinc worth having: sinc's error is FLAT --
2.3-5.6 nats across the entire 20-point grid -- while **cubic degrades ~6-8x as fmin goes
20 -> 150** at fixed mass (M=9: 10.7 -> 69.3 nats; M=20: 4.7 -> 45.2).  Raising fmin cuts the long
low-frequency inspiral out of band, which broadens Q relative to Nyquist: exactly sinc's regime.

'nearest' is never competitive: 200-440 nats throughout, and it crosses 1 nat of error at SNR
2-6, i.e. it is already unusable at O4 SNRs.

MARGINS, SCOPED.  **At fmin 30** (the mass ladder above) every margin either way over M = 9-55 is
2.1-3.0x and the worst below 120 is 9.1x.  **Across the fmin sweep** the range is wider, 1.1x to
15.9x, because cubic degrades with fmin while sinc does not.  Quote whichever matches the
configuration you are describing; they are not interchangeable.  The "330x penalty for picking
sinc wrongly" quoted in pre-IMR revisions was a TaylorT4 artifact and is gone either way -- there
is no longer a strong safety reason to break ties toward cubic.

SINC'S ERROR IS FLAT, which is the load-bearing consistency check: 3.1-7.9 nats across the fmin-30
mass ladder and both approximants, and 2.3-5.6 nats across the 20-point fmin sweep.  Flat in BOTH
sweeps is exactly what a window-limited, oversampling-independent error must do.  All the
variation, in both, is cubic's.

WHAT ACTUALLY SETS THE ANSWER is fNyq divided by the true Q bandwidth, and estimating that
bandwidth is the open problem.  f_ISCO is NOT a usable proxy: measured/f_ISCO drifts 15.8x across
2.6-120 Msun with IMR (worse than the 7.4x seen with TaylorT4) and reverses sign near M ~ 10.
Nor is a 99.99%-power quantile of the measured spectrum: with IMR points it is non-monotone
(sinc still wins at fNyq/f_Q = 4.63 while cubic already wins at 4.23), because an IMR spectrum
has a ringdown bump rather than a smooth roll-off.

RIFT.misc.psd_bandwidth does NOT separate them, and this has now been tested properly.  An
earlier revision proposed it as a future selector on the strength of a clean split at quantile
0.99 (sinc <= 2.99, cubic >= 4.33, a 45% gap).  That split was measured at a SINGLE fmin -- all 9
points were fmin 30.  Adding the fmin sweep, the classes OVERLAP over [4.21, 6.01] with 5 points
inside, and one sinc winner ranks above four cubic winners.  A quantile sweep from 0.50 to
0.99999 finds NO separating value; the best is 0.95, still overlapping by 1.18x.  The estimator's
fmin response is simply too weak in the direction that matters: over fmin 20->150 it moves the
M=55 score by only -7% while the physics flips the winner.

That is the THIRD candidate signature to fail -- fNyq/fmax, then f_ISCO, now a PSD-integrated
bandwidth -- which is why the choice is documented rather than automated.

ERROR GROWS AS SNR^2 (measured exponent 1.999-2.006 over two decades), so the choice matters more
at 3G sensitivities.

COST, measured: sinc is ~4.2-4.5x cubic on CPU (16 taps against 4; tap-count bound) but only
~1.6-3.0x on GPU (bandwidth bound).  End-to-end on CPU at fixed n_max: nearest 9.3 s, cubic
25.1 s, sinc 85.3 s.  On GPU the difference is not resolvable in wall time.

STANDING LIMITATIONS: zero noise, analytic ZDHP PSD, Lmax 2, non-spinning, equal mass except 2.6,
one sky location, 3 seeds, one sky/PSD combination.  SEOBNRv4 is unreachable at srate 4096 below
M ~ 8, so the low-fmin crossover is bracketed 20 < M < 35 but not resolved further, and the
high-fmin crossover only as "> 55".

THE AXES THAT HAVE BEEN SWEPT ARE mass and fmin.  BOTH moved the answer, and the second one moved
it AFTER the first had been published as settled.  fmax and Lmax have NOT been swept and should be
presumed load-bearing until they are -- on this heuristic that presumption has now been correct
twice.
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


# The value argparse stores for a BARE '--internal-ile-interpolate-time'.  It must NOT be None:
# with const=None a bare flag is indistinguishable from omitting the flag entirely, so the
# pipeline's truthiness guard skips the block and emits no --interpolate-time at all -- silently
# turning the feature OFF for anyone using the old store_true spelling, while the help text
# claims an explicit stencil is required.  A distinct sentinel makes the bare form reachable so
# it can be rejected with an actionable message.
BARE_FLAG_SENTINEL = '__bare__'

# The one-line crossover statement, duplicated verbatim into every user-facing help string that
# advises on stencil choice.  Defined here so the duplication is CHECKABLE: test_interpolate_time_cli
# asserts each entry point's --help contains this exact text, which is what stops one copy drifting
# (an earlier revision left util_RIFT_pseudo_pipe.py recommending the pre-IMR "cubic unless below
# ~4 Msun", i.e. the measurably worse stencil across roughly 4-20 Msun, while the others were right).
CROSSOVER_GUIDANCE = ("the crossover rises with fmin -- 20-35 Msun at fmin <= 50 Hz, 35-55 Msun "
                      "at fmin 100, and above 55 Msun at fmin 150 (measured over 9-55 Msun only)")



def is_off_request(value):
    """True if this --internal-ile-interpolate-time value means "disabled"."""
    return str(value).strip().lower() in OFF_REQUEST_TOKENS


def resolve_interpolate_time_request(value):
    """Map a raw --internal-ile-interpolate-time value to None (disabled) or a stencil name.

    THE SINGLE DEFINITION of what that flag means, so the two pipeline entry points cannot drift.
    Returns None when the feature is off (flag absent, or an explicit 'False'/'off'/...), and a
    canonical stencil name otherwise.  Raises ValueError, with guidance, for a bare flag, a
    retired "choose for me" spelling, or a typo.
    """
    if value is None:
        return None                      # flag absent
    if is_off_request(value):
        return None                      # explicitly disabled
    if str(value).strip().lower() == BARE_FLAG_SENTINEL:
        raise ValueError(
            "--internal-ile-interpolate-time was given with no value. It used to be a bare "
            "on/off flag that also chose the stencil for you; automatic selection has been "
            "REMOVED as measurably unreliable, so a stencil must now be named explicitly: "
            "nearest|cubic|sinc. Measured with an IMR model the crossover is between 20 and 35 "
            "%s; 'sinc' below the crossover, 'cubic' above. See "
            "RIFT.likelihood.time_interp_choice for the tables." % CROSSOVER_GUIDANCE)
    return validate_stencil_name(value)


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
            "can see. Pass an explicit stencil instead. Measured with an IMR model, %s; "
            "'sinc' below the crossover, 'cubic' above. See RIFT.likelihood.time_interp_choice "
            "for the measured tables." % (value, CROSSOVER_GUIDANCE))
    raise ValueError(
        "unrecognised Q_lm time-interpolation stencil %r: expected one of %s, or a value meaning "
        "disabled (%s)"
        % (value, "|".join(TIME_INTERP_CHOICES), "|".join(OFF_REQUEST_TOKENS)))
