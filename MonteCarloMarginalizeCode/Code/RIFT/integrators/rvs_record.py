"""A sampler's sample record, carrying its own provenance.

DRAFT -- see DESIGN_rvs_naming.md in this directory.  Nothing reads this yet.

WHY THIS EXISTS
---------------
`sampler._rvs` means two things at two times in one function: the RETAINED SET while
`integrate_log` accumulates, and an EXPORT RESAMPLE afterwards, once the fair draw has
rebound every key to ~1.5*eff_samp rows drawn WITH REPLACEMENT proportional to weight.  The
name does not change and neither does the type, so a consumer written against the first
meaning keeps working, silently, against the second.

Nine defects of that shape are on record, and four of them were found reviewing the fix for
the other five -- every one of the four in the BOOLEAN BOOKKEEPING introduced to describe
`_rvs` from outside, rather than in the physics:

  * a fix correct in isolation, wrong once pooling ran after it;
  * one flag answering two questions ("rows resampled" and "record is equal-weight");
  * the CLI option used where "what this pass actually did" was needed;
  * a marker cleared only on the normal return, surviving a raised event.

The common cause is that provenance lived BESIDE the rows instead of WITH them, so every site
that touched the rows had to remember to update something else.  This record puts the two
together, and replaces the booleans with named questions:

    rec.rows_are_resampled()      per-BLOCK property; survives pooling
    rec.is_equal_weight()         whole-RECORD property; pooling destroys it
    rec.posterior_log_weights()   what to weight rows by to represent the posterior

Those first two are the pair that a single boolean kept conflating.  They are deliberately
separate methods with separate names, because the failure mode was not that the answer was
hard to compute -- it was that one name suggested one question while a caller asked another.
"""
from __future__ import absolute_import

import copy

import numpy as np


class RvsProvenance(object):
    """How a sample record came to be -- travels WITH the rows, never beside them.

    `resampled_blocks` is a list, one entry per block, not a single boolean.  A pooled record
    can mix raw and resampled replicas: the fair draw is skipped per pass when it would not
    shrink that pass's record, so a run near the n_extr boundary really does produce both.  A
    scalar cannot express that, and using the CLI option in its place either flattens a
    replica whose importance weights are genuine or leaves a resampled one double-weighted.
    """

    __slots__ = ("resampled_blocks", "block_sizes", "pooled", "n_retained")

    def __init__(self, resampled_blocks=None, block_sizes=None, pooled=False, n_retained=None):
        self.resampled_blocks = list(resampled_blocks or [])
        self.block_sizes = list(block_sizes or [])
        self.pooled = bool(pooled)
        self.n_retained = n_retained          # rows BEFORE the draw, when known

    def __repr__(self):
        return ("RvsProvenance(resampled_blocks={}, block_sizes={}, pooled={}, n_retained={})"
                .format(self.resampled_blocks, self.block_sizes, self.pooled, self.n_retained))


class RvsRecord(object):
    """Sample columns plus the provenance describing them.

    Deliberately NOT a dict subclass.  Consumers that want the old behaviour should reach for
    `.columns`, which makes the read visible in a diff and greppable by the audit script; a
    dict subclass would let every existing `sampler._rvs[...]` keep working against an object
    whose meaning it does not check, which is the whole problem restated.
    """

    __slots__ = ("columns", "provenance", "reserve", "integrand_is_log")

    def __init__(self, columns, provenance=None, reserve=None, integrand_is_log=None):
        self.columns = columns
        self.provenance = provenance if provenance is not None else RvsProvenance()
        # WHAT THE RAW `integrand` COLUMN MEANS ON THIS BACKEND, recorded once by the sampler
        # that wrote it.  True = lnL, False = linear L, None = unknown.
        #
        # This is where `return_lnI` goes to die.  Today that kwarg's value is a RUNTIME
        # property of how mcsamplerEnsemble was called, and no consumer can recover it -- which
        # is why ln_weights_from_rvs has to demand `use_lnL` from every caller and why passing
        # opts.internal_use_lnL instead is a documented bug.  The sampler knows; it now says so
        # once, here, and log_likelihood() below is unambiguous on every backend.
        self.integrand_is_log = integrand_is_log
        # REFERENCE, not a copy.  See retained_* below for why this is a reference and why it
        # is the bounded reserve rather than the raw retained rows.
        self.reserve = reserve

    # -- construction ------------------------------------------------------------------
    @classmethod
    def retained(cls, columns, n_retained=None, reserve=None, integrand_is_log=None):
        """A record whose rows are the pass's own draws, with real importance weights."""
        n = _n_rows(columns)
        return cls(columns, RvsProvenance(resampled_blocks=[False], block_sizes=[n],
                                          pooled=False,
                                          n_retained=n if n_retained is None else n_retained),
                   reserve=reserve, integrand_is_log=integrand_is_log)

    @classmethod
    def fair_draw(cls, columns, n_retained=None, reserve=None, integrand_is_log=None):
        """A record whose rows were drawn WITH REPLACEMENT proportional to weight."""
        n = _n_rows(columns)
        return cls(columns, RvsProvenance(resampled_blocks=[True], block_sizes=[n],
                                          pooled=False, n_retained=n_retained),
                   reserve=reserve, integrand_is_log=integrand_is_log)

    @classmethod
    def pooled(cls, columns, resampled_blocks, block_sizes, reserve=None,
               integrand_is_log=None):
        """A concatenation of replica blocks, weighted between blocks by their evidences."""
        return cls(columns, RvsProvenance(resampled_blocks=list(resampled_blocks),
                                          block_sizes=list(block_sizes), pooled=True),
                   reserve=reserve, integrand_is_log=integrand_is_log)

    # -- the questions -----------------------------------------------------------------
    def rows_are_resampled(self):
        """Were any rows drawn proportional to weight?  A PER-BLOCK property.

        True for a plain fair draw AND for a pooled record built from fair-drawn replicas --
        pooling concatenates blocks, it does not un-resample their rows.  Anything that must
        not re-weight such rows (the .dslice reweight core) asks THIS.

        `any`, not `all`: with a mixture, a consumer that cannot weight rows differently by
        provenance must treat the whole record as unsafe to reweight.
        """
        return any(self.provenance.resampled_blocks)

    def is_equal_weight(self):
        """Does EVERY row carry the same posterior weight?  A WHOLE-RECORD property.

        A single fair draw: yes.  A pooled record: NO, even though each of its blocks is
        internally equal-weight -- blocks differ by exactly the replica evidences Z_k/K, and
        flattening them would mix replicas by row count instead of by evidence.

        This is the question `ln_weights_for_posterior` asks, and the one a single
        `_rvs_is_fairdraw` boolean answered wrongly for a pooled record.
        """
        return (not self.provenance.pooled) and self.rows_are_resampled()

    def blocks_were_flattened(self):
        """Did pooling force equal weights within any block?

        The predicate for "is the Kish n_eff of this record meaningful": a flattened block's
        rows carry its EXPORT SIZE, not its integration quality, so the pooled Kish becomes a
        row count.  Distinct from both questions above -- it is a fact about the pooling STEP.
        """
        return self.provenance.pooled and any(self.provenance.resampled_blocks)

    # -- THE UNIVERSAL OUTPUT API ---------------------------------------------------
    #
    # `_rvs` is INTERNAL.  These are what a consumer should call: one name per quantity, the
    # same meaning on every backend, so nobody has to know that `integrand` holds lnL on three
    # samplers, linear L on two, and either on a sixth depending on a kwarg (the table is in
    # test/expensive_before_merging/integrators/audit_backend_contracts.py).
    #
    # Everything is returned in LOG space, because that is the only convention all six can
    # express without loss -- the linear column underflows to 0 at ~745 nats, which is exactly
    # the regime this whole line of work is about.

    def log_likelihood(self):
        """ln L per row -> float array.  The same thing on every backend.

        Prefers the unambiguous `log_integrand` column.  Falls back to `integrand` ONLY when
        the sampler stated what that column means; when it did not, this RAISES rather than
        guess -- a loud failure beats a plausible wrong number, which is the same rule
        ln_weights_from_rvs already applies one layer down.
        """
        c = self.columns
        if 'log_integrand' in c:
            return np.asarray(_host(c['log_integrand']), dtype=float).ravel()
        if 'integrand' not in c:
            raise KeyError("record has neither 'log_integrand' nor 'integrand'")
        ig = np.asarray(_host(c['integrand']), dtype=float).ravel()
        if self.integrand_is_log is True:
            return ig
        if self.integrand_is_log is False:
            out = np.full(len(ig), -np.inf)
            pos = ig > 0
            out[pos] = np.log(ig[pos])       # non-positive means a rejected/underflowed row
            return out
        raise ValueError(
            "this record has only a raw 'integrand' column and the sampler did not record "
            "whether it holds L or lnL, so its meaning is unrecoverable.  Pass "
            "integrand_is_log= when building the record (see DESIGN_rvs_naming.md).")

    def log_prior(self):
        """ln pi per row -> float array."""
        return self._log_of('log_joint_prior', 'joint_prior')

    def log_sampling_prior(self):
        """ln q per row -> float array."""
        return self._log_of('log_joint_s_prior', 'joint_s_prior')

    def _log_of(self, log_key, lin_key):
        c = self.columns
        if log_key in c:
            return np.asarray(_host(c[log_key]), dtype=float).ravel()
        if lin_key not in c:
            raise KeyError("record has neither {!r} nor {!r}".format(log_key, lin_key))
        v = np.asarray(_host(c[lin_key]), dtype=float).ravel()
        out = np.full(len(v), -np.inf)
        pos = v > 0
        out[pos] = np.log(v[pos])
        return out

    def log_weights(self):
        """THE importance log-weight per row: lnL + ln pi - ln q -> float array.

        No `use_lnL` argument, because the record already knows.  That parameter exists on
        ln_weights_from_rvs only because a bare `_rvs` dict cannot say what its own columns
        mean; a consumer on this API cannot get it wrong.
        """
        return self.log_likelihood() + self.log_prior() - self.log_sampling_prior()

    # -- weights -----------------------------------------------------------------------
    def posterior_log_weights(self, ln_weights_from_columns):
        """Weights to represent the posterior -> float array.

        Uniform when the record is globally equal-weight; otherwise the caller's canonical
        importance weight, derived from the columns.  The derivation is injected rather than
        imported so this module stays free of the ILE's convention handling.
        """
        if self.is_equal_weight():
            return np.zeros(_n_rows(self.columns), dtype=float)
        return np.asarray(ln_weights_from_columns(self.columns), dtype=float)

    # -- the rows the pass actually drew -------------------------------------------------
    def has_retained(self):
        """Is a usable record of the pre-draw rows available?"""
        r = self.reserve
        return isinstance(r, dict) and 'X' in r and 'lnL' in r

    def retained_points(self):
        """(n, ndim) of the points the pass RETAINED, or None.

        A REFERENCE to the bounded warm-seed reserve, deliberately, not the raw retained rows.
        Measured (measure_retained_set_memory.py): holding the raw set costs ~0.9 MB per
        million nmax for AV -- nothing -- but ~92 MB per million for a PORTFOLIO, whose _rvs
        holds every draw, i.e. ~384 MB at nmax=4e6 per ILE process.  And it would be mostly
        ballast: on the collapsed pass this work is about, the portfolio's finite fraction is
        ~1e-5, so almost all of it is -inf rows no consumer can use.

        make_warm_seed_reserve already keeps the affordable thing -- bounded at n_max rows,
        stratified by finite-ness, with the EXACT pre-cap weight total recorded alongside so a
        capped reserve still yields an unbiased lnZ.  Pointing at it costs nothing and is
        already paid for.
        """
        return np.asarray(self.reserve['X'], dtype=float) if self.has_retained() else None

    def retained_lnL(self):
        """lnL of the retained points, or None.  Same reference as retained_points()."""
        return (np.asarray(self.reserve['lnL'], dtype=float).ravel()
                if self.has_retained() else None)

    def n_retained(self):
        """Rows the pass retained BEFORE the draw, when known -- not len(self)."""
        n = self.provenance.n_retained
        if n is None and self.has_retained():
            n = self.reserve.get('n_retained')
        return n

    # -- lifecycle ---------------------------------------------------------------------
    def snapshot(self):
        """A copy that a rejected pass can be restored from, provenance included.

        The rows are copied shallowly (columns are replaced wholesale by the rebind, never
        mutated in place) but the PROVENANCE is copied deeply, because restoring rows while
        leaving provenance describing the rejected pass is one of the four defects this
        record exists to prevent.
        """
        # The reserve rides along BY REFERENCE: it is immutable once built (each pass builds a
        # fresh one), and copying it would reintroduce the memory cost this design avoids.
        return RvsRecord(dict(self.columns), copy.deepcopy(self.provenance),
                         reserve=self.reserve)

    def __len__(self):
        return _n_rows(self.columns)

    def __repr__(self):
        return "RvsRecord({} rows, {})".format(len(self), self.provenance)


class SamplerOutputMixin(object):
    """The public output API every backend gets by inheriting it.

    `_rvs` is an INTERNAL variable: it means different things at different times, and its raw
    columns mean different things on different backends.  Consumers should never reach inside
    it -- they should call this.

    Kept as a mixin because the six MCSampler classes share no base class today (each is
    `class MCSampler(object)`), and giving them one is a bigger change than this draft should
    make.
    """

    def samples(self):
        """This pass's samples, with provenance -> RvsRecord, or None if it never ran.

        THE public accessor.  Everything a consumer needs -- log_likelihood(), log_prior(),
        log_sampling_prior(), log_weights(), rows_are_resampled(), is_equal_weight() -- hangs
        off the returned record and means the same thing on every backend.
        """
        return getattr(self, '_rvs_record', None)

    def set_samples(self, record):
        """Replace this pass's record -> the record, for chaining.

        PUBLIC because the ILE legitimately produces one: replica pooling builds a record the
        sampler cannot (it is a mixture of several passes).  Without this, that code would have
        to assign `sampler._rvs_record` directly -- reaching into another object's private
        attribute, which is the habit this whole design is trying to end.  A writer needs an
        API as much as a reader does.
        """
        self._rvs_record = record
        return record


def _host(v):
    """cupy -> numpy where needed, without importing cupy."""
    try:
        return v.get() if hasattr(v, 'get') and not isinstance(v, np.ndarray) else v
    except Exception:
        return v


def _n_rows(columns):
    for v in (columns or {}).values():
        try:
            return len(np.atleast_1d(np.asarray(v)).ravel())
        except Exception:
            continue
    return 0
