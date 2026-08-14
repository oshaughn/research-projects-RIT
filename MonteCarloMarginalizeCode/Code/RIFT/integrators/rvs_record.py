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

    __slots__ = ("columns", "provenance")

    def __init__(self, columns, provenance=None):
        self.columns = columns
        self.provenance = provenance if provenance is not None else RvsProvenance()

    # -- construction ------------------------------------------------------------------
    @classmethod
    def retained(cls, columns, n_retained=None):
        """A record whose rows are the pass's own draws, with real importance weights."""
        n = _n_rows(columns)
        return cls(columns, RvsProvenance(resampled_blocks=[False], block_sizes=[n],
                                          pooled=False,
                                          n_retained=n if n_retained is None else n_retained))

    @classmethod
    def fair_draw(cls, columns, n_retained=None):
        """A record whose rows were drawn WITH REPLACEMENT proportional to weight."""
        n = _n_rows(columns)
        return cls(columns, RvsProvenance(resampled_blocks=[True], block_sizes=[n],
                                          pooled=False, n_retained=n_retained))

    @classmethod
    def pooled(cls, columns, resampled_blocks, block_sizes):
        """A concatenation of replica blocks, weighted between blocks by their evidences."""
        return cls(columns, RvsProvenance(resampled_blocks=list(resampled_blocks),
                                          block_sizes=list(block_sizes), pooled=True))

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

    # -- lifecycle ---------------------------------------------------------------------
    def snapshot(self):
        """A copy that a rejected pass can be restored from, provenance included.

        The rows are copied shallowly (columns are replaced wholesale by the rebind, never
        mutated in place) but the PROVENANCE is copied deeply, because restoring rows while
        leaving provenance describing the rejected pass is one of the four defects this
        record exists to prevent.
        """
        return RvsRecord(dict(self.columns), copy.deepcopy(self.provenance))

    def __len__(self):
        return _n_rows(self.columns)

    def __repr__(self):
        return "RvsRecord({} rows, {})".format(len(self), self.provenance)


def _n_rows(columns):
    for v in (columns or {}).values():
        try:
            return len(np.atleast_1d(np.asarray(v)).ravel())
        except Exception:
            continue
    return 0
