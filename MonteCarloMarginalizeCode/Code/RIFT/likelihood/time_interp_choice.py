"""Which sub-sample Q_lm stencil should a run use?

Leaf module on purpose: numpy only, no lal, no numba, no cupy, so the pipeline scripts can import
it without paying ~4 s of numba compilation.

THE DECISION.  There is no automatic selection, and that is a MEASURED CONCLUSION, not an
omission: three candidate rules were built and all three were disproved, the last fatally --
the right stencil depends on fmin and srate as well as mass, so no (srate, fmax, mass)
signature can be correct.  The flag therefore takes an explicit stencil name, and the retired "choose for me"
spelling raises rather than resolving to a default.

The live recommendation is the CROSSOVER_GUIDANCE constant below.  Every user-facing help string
interpolates it, and test_interpolate_time_cli.py pins it across all entry points, so there is
exactly one place to change if the measurement changes.

THE EVIDENCE LIVES IN DESIGN_q_window_stencil.md, NOT HERE.  That file carries the measured
tables, the three disproved rules, the cost figures, the limitations and the provenance.  It is a
record and is expected to be superseded; this module is code and should not accumulate numbers
that go stale silently.  If the two ever disagree, CROSSOVER_GUIDANCE is authoritative.

    RIFT/likelihood/DESIGN_q_window_stencil.md          (measurements, as of 2026-08-16)
"""
from __future__ import division

# The stencils this tree knows about.  Kept here rather than imported from factored_likelihood so
# the pipeline can validate a spelling without paying numba's import cost;
# factored_likelihood.TIME_INTERP_CHOICES must agree and test_time_interp_choice asserts it does.
TIME_INTERP_CHOICES = ('nearest', 'cubic', 'sinc')

# Taps per SIDE for the 'sinc' stencil (the full stencil is 2a wide).  It lives in this leaf
# module, rather than beside the weight builder in factored_likelihood, because three backends
# now need it -- the CPU window builder, the cupy kernel wrapper, and the JAX gatherer -- and the
# JAX path must not pay factored_likelihood's numba/lal import cost to learn one integer.
# factored_likelihood re-exports it, so `FL.SINC_HALFWIDTH_DEFAULT` keeps working.
SINC_HALFWIDTH_DEFAULT = 8

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
CROSSOVER_GUIDANCE = (
    "at srate 4096 / fmax 1700 the crossover in TOTAL MASS rises with fmin -- 20-35 Msun at "
    "fmin <= 50 Hz, 35-55 Msun at fmin 100, above 55 Msun at fmin 150 -- with 'sinc' BELOW the "
    "crossover and 'cubic' ABOVE it; measured over 9-55 Msun at that srate only, and a higher "
    "srate moves it (at srate 16384 even 2.6-5 Msun measures cubic, by 21-34x)")



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
            "nearest|cubic|sinc. Measured with an IMR model, %s. See "
            "RIFT/likelihood/DESIGN_q_window_stencil.md for the tables." % CROSSOVER_GUIDANCE)
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
            "can see. Pass an explicit stencil instead. Measured with an IMR model, %s. See "
            "RIFT/likelihood/DESIGN_q_window_stencil.md for the measured tables."
            % (value, CROSSOVER_GUIDANCE))
    raise ValueError(
        "unrecognised Q_lm time-interpolation stencil %r: expected one of %s, or a value meaning "
        "disabled (%s)"
        % (value, "|".join(TIME_INTERP_CHOICES), "|".join(OFF_REQUEST_TOKENS)))
