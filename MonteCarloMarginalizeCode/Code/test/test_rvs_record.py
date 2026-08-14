#!/usr/bin/env python
"""
Contract for RvsRecord (DRAFT -- see RIFT/integrators/DESIGN_rvs_naming.md).

The point of this suite is not coverage for its own sake.  Nine defects of one shape are on
record, and FOUR of them were found while reviewing the fix for the other five -- every one of
those four in the boolean bookkeeping that described `_rvs` from outside.  So each section here
is one of those four failure shapes, written as a test that WOULD HAVE CAUGHT ITS ROUND had the
provenance lived with the rows from the start.

If this design is adopted, these are the tests that justify it.  If it is not, they are the
specification of what any replacement has to get right.
"""

import numpy as np
import pytest

from RIFT.integrators.rvs_record import RvsRecord, RvsProvenance


def _cols(n, seed=0, spread=2.0):
    rng = np.random.default_rng(seed)
    lnL = rng.normal(0.0, spread, size=n)
    return {"log_integrand": lnL,
            "log_joint_prior": np.zeros(n),
            "log_joint_s_prior": np.zeros(n),
            "x": rng.normal(size=n)}


def _ln_w(columns):
    """Stand-in for the ILE's ln_weights_from_rvs."""
    return (np.asarray(columns["log_integrand"], float)
            + np.asarray(columns["log_joint_prior"], float)
            - np.asarray(columns["log_joint_s_prior"], float))


###
### Shape 2 (review round 2): ONE FLAG, TWO QUESTIONS
###
### A single boolean meant both "rows were drawn proportional to w" and "the record is
### globally equal-weight".  A pooled record answers yes to the first and no to the second, so
### whichever way the flag was set, one consumer was wrong.
###

def test_a_fair_draw_answers_yes_to_both_questions():
    rec = RvsRecord.fair_draw(_cols(50), n_retained=1000)
    assert rec.rows_are_resampled() is True
    assert rec.is_equal_weight() is True


def test_a_pooled_record_answers_yes_to_one_and_no_to_the_other():
    """The case a single boolean cannot represent."""
    rec = RvsRecord.pooled(_cols(80), resampled_blocks=[True, True], block_sizes=[40, 40])
    assert rec.rows_are_resampled() is True, \
        'pooling concatenates blocks; it does not un-resample their rows'
    assert rec.is_equal_weight() is False, \
        'blocks differ by their replica evidences, so the record is not globally uniform'


def test_a_retained_record_answers_no_to_both():
    rec = RvsRecord.retained(_cols(500))
    assert rec.rows_are_resampled() is False
    assert rec.is_equal_weight() is False


def test_the_flattening_question_is_a_third_thing_again():
    """`blocks_were_flattened` is a fact about the POOLING STEP, not about the record.

    Keying the pooled-n_eff branch on either of the other two put it below the line that
    changed its own predicate, and it became dead code.
    """
    plain = RvsRecord.fair_draw(_cols(30))
    assert plain.rows_are_resampled() and not plain.blocks_were_flattened(), \
        'an unpooled fair draw was never flattened by pooling'
    pooled_raw = RvsRecord.pooled(_cols(60), resampled_blocks=[False, False], block_sizes=[30, 30])
    assert not pooled_raw.blocks_were_flattened(), 'no block was resampled, so none was flattened'
    pooled_mixed = RvsRecord.pooled(_cols(60), resampled_blocks=[False, True], block_sizes=[30, 30])
    assert pooled_mixed.blocks_were_flattened()


###
### Shape 3 (round 2): THE OPTION IS NOT THE EVENT
###
### `already_resampled=opts.fairdraw_extrinsic_output` is not "did the draw fire": it is
### skipped per pass when it would not shrink that pass's record, so a run can produce a
### MIXTURE of raw and resampled replicas.
###

def test_provenance_is_per_block_not_a_single_boolean():
    rec = RvsRecord.pooled(_cols(90), resampled_blocks=[True, False, True],
                           block_sizes=[30, 30, 30])
    assert rec.provenance.resampled_blocks == [True, False, True]
    assert rec.rows_are_resampled() is True, \
        'a consumer that cannot weight rows differently by provenance must treat the whole ' \
        'record as unsafe to reweight'


def test_a_mixture_is_representable_at_all():
    """The property a scalar cannot have.  Pinned because the scalar version type-checks."""
    mixed = RvsProvenance(resampled_blocks=[True, False], block_sizes=[10, 10], pooled=True)
    assert any(mixed.resampled_blocks) and not all(mixed.resampled_blocks)


###
### Shape 1 (round 1) and shape 4 (round 3): PROVENANCE MUST TRAVEL WITH THE ROWS
###
### Round 1: a rejected warm pass restored its rows but left the reserve and the marker
### describing the pass that had just been thrown away.
### Round 3: a marker cleared only on the normal return survived a raised event and was
### inherited by the next one.
###
### Both are impossible when the provenance is a field of the record being restored, rather
### than a separate attribute someone has to remember.
###

def test_a_snapshot_restores_provenance_along_with_the_rows():
    cold = RvsRecord.fair_draw(_cols(40, seed=1), n_retained=5000)
    saved = cold.snapshot()

    # the warm pass replaces the record in place, with different provenance
    warm = RvsRecord.pooled(_cols(9, seed=2), resampled_blocks=[True, True], block_sizes=[4, 5])

    assert warm.is_equal_weight() is False
    assert saved.is_equal_weight() is True, \
        'the snapshot must still describe the COLD pass, not the warm one that replaced it'
    assert saved.provenance.n_retained == 5000


def test_a_snapshot_cannot_be_mutated_by_the_pass_that_follows_it():
    """Round 1 in miniature: the restored provenance must not alias the live one."""
    rec = RvsRecord.fair_draw(_cols(20), n_retained=100)
    saved = rec.snapshot()
    rec.provenance.pooled = True
    rec.provenance.resampled_blocks.append(True)
    assert saved.provenance.pooled is False
    assert saved.provenance.resampled_blocks == [True], 'the snapshot aliased live provenance'


def test_there_is_no_marker_left_to_leak_across_events():
    """Round 3 could not happen here: 'pooled' is a field of the record, so dropping the
    record drops it.  Nothing survives to be inherited by the next event."""
    rec = RvsRecord.pooled(_cols(20), resampled_blocks=[True], block_sizes=[20])
    assert rec.is_equal_weight() is False
    rec = RvsRecord.fair_draw(_cols(20))          # the next event builds a NEW record
    assert rec.is_equal_weight() is True, \
        'a fresh fair draw inherited "pooled" from the record before it'


###
### The weights, which is what all of this is for
###

def test_posterior_weights_are_uniform_only_for_a_globally_equal_weight_record():
    fair = RvsRecord.fair_draw(_cols(60, seed=3))
    assert np.allclose(fair.posterior_log_weights(_ln_w), 0.0)

    retained = RvsRecord.retained(_cols(60, seed=3))
    lw = retained.posterior_log_weights(_ln_w)
    assert np.allclose(lw, _ln_w(retained.columns))
    assert np.std(lw) > 1.0, 'these weights are not degenerate; flattening them loses the shape'


def test_a_pooled_record_keeps_its_between_block_weights():
    """The round-1 defect: flattening a pooled record mixes replicas by row count."""
    cols = _cols(80, seed=4)
    # blocks offset by 2 nats, as _pool_replica_rvs would leave them
    cols["log_integrand"] = np.concatenate([np.zeros(40), np.full(40, 2.0)])
    rec = RvsRecord.pooled(cols, resampled_blocks=[True, True], block_sizes=[40, 40])
    lw = rec.posterior_log_weights(_ln_w)
    assert not np.allclose(lw, 0.0), 'the replica evidences were flattened away'
    assert lw[40] - lw[0] == pytest.approx(2.0, abs=1e-9)


def test_len_reports_rows_not_columns():
    assert len(RvsRecord.retained(_cols(37))) == 37
    assert len(RvsRecord.retained({})) == 0


###
### The record is deliberately NOT a dict
###

def test_the_record_is_not_a_dict_subclass():
    """A dict subclass would let every existing `sampler._rvs[...]` keep working against an
    object whose meaning it does not check -- the original problem, restated with more steps.
    Consumers must reach for `.columns`, which is visible in a diff and greppable."""
    rec = RvsRecord.retained(_cols(5))
    assert not isinstance(rec, dict)
    with pytest.raises(TypeError):
        rec["log_integrand"]
    assert "log_integrand" in rec.columns
