#!/usr/bin/env python
"""
Contract for RvsRecord (see RIFT/integrators/DESIGN_rvs_naming.md).

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

from RIFT.integrators.rvs_record import RvsRecord, RvsProvenance, SamplerOutputMixin


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
### A COMBINED PARAMETER IS (ndim, N), AND IT COMES FIRST
###
### Every sampler indexes a TUPLE-keyed column as `col[:, idx]` and a plain one as `col[idx]`,
### and `_rvs` is seeded parameters-first -- so counting rows by flattening whichever column
### came first reported ndim*N for an ordinary run with a combined parameter.
###

def _cols_with_combined(n, ndim=3, seed=0):
    """Columns in the order a sampler builds them: the combined parameter FIRST."""
    rng = np.random.default_rng(seed)
    cols = {tuple("p{}".format(i) for i in range(ndim)): rng.normal(size=(ndim, n))}
    cols.update(_cols(n, seed=seed))
    return cols


def test_a_combined_parameter_does_not_multiply_the_row_count():
    rec = RvsRecord.retained(_cols_with_combined(64, ndim=3))
    assert len(rec) == 64, 'the (ndim, N) column was flattened into ndim*N rows'
    assert rec.provenance.block_sizes == [64]
    assert rec.provenance.n_retained == 64


def test_a_fair_draws_uniform_weights_are_one_per_row_not_one_per_entry():
    """The output-length failure: this vector is handed to consumers alongside the rows."""
    rec = RvsRecord.fair_draw(_cols_with_combined(50, ndim=4, seed=5))
    lw = rec.posterior_log_weights(_ln_w)
    assert lw.shape == (50,)
    assert np.allclose(lw, 0.0)


def test_the_row_axis_is_read_from_the_key_even_with_no_scalar_column():
    """A record of parameter columns alone still has to know where its rows are."""
    rng = np.random.default_rng(7)
    assert len(RvsRecord.retained({("m1", "m2"): rng.normal(size=(2, 12))})) == 12
    assert len(RvsRecord.retained({"m1": rng.normal(size=12)})) == 12


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


###
### MIGRATION SAFETY: while the record and the flags both exist, they must agree
###
### This is the one real cost of option A -- two sources of truth during the migration -- so it
### is asserted rather than left as a promise in a design doc.  Four review rounds on #87 were
### all "two descriptions of one thing drifted apart"; this is the guard against doing it again
### at one level up.
###

import os

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')


def _ile_predicates():
    """Exec the ILE's two provenance predicates (it parses argv, so it is not importable)."""
    src = open(_ILE).read()
    # start at the shared LOOKUP, which is defined before the two predicates
    start = src.index("def _rvs_record_for")
    end = src.index("def _pool_replica_rvs")
    ns = {"numpy": np, "np": np, "_rvs_lnL_convention": lambda x=None: bool(x)}
    exec(compile(src[start:end], "ile_predicates", "exec"), ns)
    return ns


class _Sampler(SamplerOutputMixin):
    """A sampler carrying BOTH descriptions, as the tree does mid-migration.

    Inherits the real mixin rather than faking `samples()`, so a change to the public API
    breaks this double instead of leaving it quietly testing something that no longer exists.
    """
    def __init__(self, record, is_fairdraw, is_pooled):
        self.set_samples(record)
        self._rvs_is_fairdraw = is_fairdraw
        self._rvs_is_pooled = is_pooled


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('state,record,flag_fd,flag_pooled', [
    ('retained',      RvsRecord.retained(_cols(20)),                            False, False),
    ('fair draw',     RvsRecord.fair_draw(_cols(20)),                           True,  False),
    ('pooled',        RvsRecord.pooled(_cols(20), [True, True], [10, 10]),      True,  True),
    ('pooled mixed',  RvsRecord.pooled(_cols(20), [True, False], [10, 10]),     True,  True),
    ('pooled raw',    RvsRecord.pooled(_cols(20), [False, False], [10, 10]),    False, True),
])
def test_the_record_and_the_flags_agree_in_every_state(state, record, flag_fd, flag_pooled):
    P = _ile_predicates()
    s = _Sampler(record, flag_fd, flag_pooled)
    assert record.rows_are_resampled() == P["_rvs_is_export_resample"](s), \
        '{}: rows-resampled disagrees between record and flag'.format(state)
    assert record.is_equal_weight() == P["_rvs_is_equal_weight"](s), \
        '{}: equal-weight disagrees between record and flag'.format(state)


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_migrated_consumer_only_trusts_a_record_describing_THESE_columns():
    """The record is a second reference to a mutable dict.  If _rvs has been replaced since the
    record was built, the record describes the wrong rows -- so the consumer checks identity
    and falls back to the flags rather than trusting a stale description."""
    src = open(_ILE).read()
    # the identity check lives in ONE lookup, not repeated per consumer (two copies drift)
    i = src.index('def _rvs_record_for')
    lookup = src[i:i + 1400]
    assert "getattr(rec, 'columns', None) is not rvs" in lookup, \
        'the shared lookup trusts a record without checking it describes these columns'
    assert 'return None' in lookup, 'a stale record must be declined, not returned'

    # and EVERY consumer goes through it rather than reading the attribute directly
    body = src[src.index('def ln_weights_for_posterior'):]
    n_direct = body.count("sampler._rvs_record")
    assert n_direct == 0, \
        '{} consumer(s) touch sampler._rvs_record directly instead of the public API'.format(
            n_direct)
    assert body.count('_rvs_record_for(sampler') >= 3, \
        'expected the weight helper, the .dslice guard and the pooled n_eff to share the lookup'

    # the PRODUCER at the pooling site asks a different question and has its own name: it is
    # about to replace sampler._rvs, so "does a record describe the rows I hold" is wrong there
    assert '_sampler_keeps_records(sampler)' in body, \
        'the pooling producer should ask whether the sampler keeps records at all'
    assert 'sampler.set_samples(' in body, \
        'the pooling producer assigns the private attribute instead of using the setter'
    assert src.count('def _sampler_keeps_records') == 1

    # the flags remain as the fallback until the last consumer is migrated
    assert '_rvs_is_equal_weight(sampler)' in body, 'the flag fallback is gone too early'


###
### THE RESERVE IS REFERENCED, NOT COPIED  (open question 2, answered by measurement)
###

def test_the_record_points_at_the_reserve_rather_than_copying_it():
    reserve = dict(X=np.zeros((7, 6)), lnL=np.zeros(7), n_retained=99999,
                   n_finite=7, ln_sum_w_finite=1.5, params_ordered=list('abcdef'))
    rec = RvsRecord.fair_draw(_cols(3), n_retained=99999, reserve=reserve)
    assert rec.reserve is reserve, 'the reserve was copied; that is the cost this design avoids'
    assert rec.has_retained()
    assert rec.retained_points().shape == (7, 6)
    assert rec.n_retained() == 99999, 'n_retained is the PRE-draw count, not len(record)'
    assert len(rec) == 3


def test_a_record_without_a_reserve_says_so_rather_than_guessing():
    rec = RvsRecord.fair_draw(_cols(3))
    assert rec.has_retained() is False
    assert rec.retained_points() is None and rec.retained_lnL() is None


def test_a_snapshot_keeps_the_reserve_by_reference():
    reserve = dict(X=np.zeros((4, 2)), lnL=np.zeros(4))
    rec = RvsRecord.fair_draw(_cols(3), reserve=reserve)
    assert rec.snapshot().reserve is reserve


def test_a_pooled_record_carries_no_reserve():
    """A pooled record is a mixture of several passes, so there is no single retained set for
    it to point at.  Saying None is correct; pointing at one arbitrary pass's would not be."""
    rec = RvsRecord.pooled(_cols(20), [True, True], [10, 10])
    assert rec.has_retained() is False


###
### END TO END on the sampler that was converted
###

def _av_sampler(n_chunk=20000):
    import RIFT.integrators.mcsamplerAdaptiveVolume as AV
    s = AV.MCSampler(n_chunk=n_chunk)
    s.xpy = AV.xpy_default
    s.identity_convert = AV.identity_convert
    for name in ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    return s


def _av_peaked(rho):
    x0 = 0.5 * np.ones(6)
    w = (0.5 / rho) * np.ones(6)
    lnLmax = 0.5 * rho ** 2

    def lnL(*args, **kwargs):
        x = np.array([np.asarray(a, dtype=float).ravel() for a in args]).T
        out = lnLmax - 0.5 * np.sum(((x - x0) / w) ** 2, axis=-1)
        return np.where(out > lnLmax - 745.0, out, -np.inf)
    return lnL


NAMES6 = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']


def test_a_real_collapsed_pass_records_the_draw_and_points_at_its_reserve():
    np.random.seed(20260813)
    s = _av_sampler()
    s.integrate_log(_av_peaked(100.0), *NAMES6, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    rec = s.samples()
    assert rec is not None and rec.rows_are_resampled() and rec.is_equal_weight()
    assert rec.columns is s._rvs, 'the record must view the live columns'
    assert rec.reserve is s._warm_seed_reserve, 'the reserve was copied rather than referenced'
    assert rec.n_retained() > len(rec), \
        'this pass did not collapse, so it does not exercise the case ({} vs {})'.format(
            rec.n_retained(), len(rec))
    assert rec.retained_points().shape[0] >= len(rec)


def test_a_pass_with_no_fair_draw_still_gets_a_record():
    """"absent" and "not resampled" are different statements; a consumer that has to tell them
    apart is back to combining conditions by hand."""
    np.random.seed(20260813)
    s = _av_sampler()
    s.integrate_log(_av_peaked(100.0), *NAMES6, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False)
    rec = s.samples()
    assert rec is not None, 'no record on the no-fair-draw path'
    assert rec.rows_are_resampled() is False and rec.is_equal_weight() is False
    assert rec.columns is s._rvs


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_migration_changes_no_number():
    """The record path and the flag path must return the SAME weights on a real pass.

    This is what makes the migration safe to land incrementally: converting a consumer is a
    refactor, not a behaviour change, and the two paths can be compared directly until the
    flags are removed.
    """
    src = open(_ILE).read()
    start = src.index("def ln_weights_from_rvs")
    end = src.index("def _pool_replica_rvs")
    ns = {"numpy": np, "np": np, "_rvs_lnL_convention": lambda x=None: bool(x)}
    exec(compile(src[start:end], "ile_w", "exec"), ns)
    ln_w_post = ns["ln_weights_for_posterior"]

    np.random.seed(20260813)
    s = _av_sampler()
    s.integrate_log(_av_peaked(100.0), *NAMES6, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    assert s.samples() is not None and s._rvs_is_fairdraw

    with_record = ln_w_post(s._rvs, s)
    stashed = s.samples()
    s.set_samples(None)                               # force the flag path
    without_record = ln_w_post(s._rvs, s)
    s.set_samples(stashed)
    assert np.array_equal(with_record, without_record), \
        'the record path and the flag path disagree; the migration is not a refactor'

    # ...and the same on a pass with no fair draw, where the answer is the other branch
    np.random.seed(20260813)
    s2 = _av_sampler()
    s2.integrate_log(_av_peaked(100.0), *NAMES6, nmax=400000, neff=8, n=20000,
                     no_protect_names=True, verbose=False)
    a = ln_w_post(s2._rvs, s2)
    s2.set_samples(None)
    b = ln_w_post(s2._rvs, s2)
    assert np.array_equal(a, b)
    assert np.std(a) > 0.0, 'a retained record must keep its varying importance weights'


###
### THE COUNT MUST BE EAGER, because the record references a dict the draw replaces in place
###

def test_n_retained_is_captured_eagerly_not_read_back_from_the_columns():
    """Found while wiring the samplers, and it is this project's own bug class in miniature.

    `RvsRecord.retained(self._rvs)` stores a REFERENCE to the live column dict.  The fair draw
    then rebinds every key of that same dict.  So `len(record)` -- which reads `.columns` --
    returns the POST-draw length, while `provenance.n_retained`, captured at construction,
    still holds the pre-draw count.  Reading the wrong one made a collapsed pass report
    n_retained == rows, i.e. "nothing was discarded", which is the exact opposite of the truth.
    """
    cols = _cols(500)
    rec = RvsRecord.retained(cols)
    assert rec.n_retained() == 500 and len(rec) == 500

    # the draw replaces every column IN PLACE, as integrate_log does
    keep = np.arange(3)
    for k in list(cols):
        cols[k] = np.asarray(cols[k])[keep]

    assert len(rec) == 3, 'len() reads the live columns, by design'
    assert rec.n_retained() == 500, \
        'n_retained was read back from the mutated columns instead of captured eagerly'


def test_a_real_collapsed_pass_reports_more_retained_than_exported():
    """The end-to-end version: on a pass that actually collapses, the record must show the
    discard, not a no-op."""
    np.random.seed(20260813)
    s = _av_sampler()
    s.integrate_log(_av_peaked(100.0), *NAMES6, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    rec = s.samples()
    assert rec.rows_are_resampled()
    assert rec.n_retained() > len(rec), \
        'n_retained={} rows={} -- the record claims the draw discarded nothing'.format(
            rec.n_retained(), len(rec))


###
### EVERY SAMPLER, not just the one that was converted first
###

def _six_samplers():
    """(label, factory, method, target, extra_kwargs) for each sampler with a rebind site.

    mcsampler and mcsamplerEnsemble take a LINEAR integrand; AV/portfolio take log.  Getting
    that wrong makes the fair draw produce negative weights and raise -- verified to fail
    identically on the pristine file, i.e. it is a harness contract, not a defect.
    """
    import RIFT.integrators.mcsampler as MC
    import RIFT.integrators.mcsamplerAdaptiveVolume as AV
    import RIFT.integrators.mcsamplerEnsemble as ENS

    def _log_tgt(rho=8.0):
        x0 = 0.5 * np.ones(6); w = (0.5 / rho) * np.ones(6); m = 0.5 * rho ** 2

        def f(*a, **k):
            x = np.array([np.asarray(v, float).ravel() for v in a]).T
            o = m - 0.5 * np.sum(((x - x0) / w) ** 2, axis=-1)
            return np.where(o > m - 745.0, o, -np.inf)
        return f

    def _lin_tgt(rho=4.0):
        x0 = 0.5 * np.ones(6); w = (0.5 / rho) * np.ones(6)

        def f(*a, **k):
            x = np.array([np.asarray(v, float).ravel() for v in a]).T
            return np.exp(-0.5 * np.sum(((x - x0) / w) ** 2, axis=-1))
        return f

    def _av():
        s = AV.MCSampler(n_chunk=5000)
        s.xpy = AV.xpy_default; s.identity_convert = AV.identity_convert
        for n in NAMES6:
            s.add_parameter(n, pdf=None, left_limit=0.0, right_limit=1.0,
                            prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
        return s

    def _vec(mod):
        def build():
            s = mod.MCSampler()
            v = np.vectorize(lambda x: 1.0)
            for n in NAMES6:
                s.add_parameter(n, v, prior_pdf=v, left_limit=0.0, right_limit=1.0,
                                adaptive_sampling=True)
            return s
        return build

    return [
        ('AV',        _av,        'integrate_log', _log_tgt()),
        ('Ensemble',  _vec(ENS),  'integrate',     _lin_tgt()),
        ('mcsampler', _vec(MC),   'integrate',     _lin_tgt()),
    ]


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('fairdraw', [True, False])
def test_every_wired_sampler_leaves_a_record_that_agrees_with_its_flags(fairdraw):
    """The mechanical step, checked rather than assumed.

    All seven rebind sites were wired by one patcher against PR #87's own markers, so a single
    mistake would be replicated everywhere -- which is exactly the case worth testing rather
    than eyeballing the diff.
    """
    P = _ile_predicates()
    for label, build, meth, target in _six_samplers():
        np.random.seed(11)
        s = build()
        kw = dict(nmax=50000, neff=30, n=5000, no_protect_names=True,
                  verbose=False, save_intg=True)
        if fairdraw:
            kw.update(igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=50)
        getattr(s, meth)(target, *NAMES6, **kw)

        rec = P["_rvs_record_for"](s, s._rvs)
        assert rec is not None, '{}: no record describing the live columns'.format(label)
        assert rec.rows_are_resampled() == P["_rvs_is_export_resample"](s), \
            '{}: rows-resampled disagrees with the flag'.format(label)
        assert rec.is_equal_weight() == P["_rvs_is_equal_weight"](s), \
            '{}: equal-weight disagrees with the flag'.format(label)
        assert rec.rows_are_resampled() is bool(fairdraw), \
            '{}: record does not reflect whether the draw fired'.format(label)
        if fairdraw:
            assert rec.n_retained() >= len(rec), \
                '{}: n_retained {} < exported rows {}'.format(label, rec.n_retained(), len(rec))


def test_all_seven_rebind_sites_are_wired_the_same_way():
    """One patcher wired all seven; pin that none was missed or hand-edited differently."""
    import glob
    total_fd = total_ret = total_reset = 0
    for p in sorted(glob.glob(os.path.join(_INTEGRATORS_DIR, 'mcsampler*.py'))):
        src = open(p).read()
        if 'bFairdraw' not in src:
            continue
        n_sites = src.count('self._rvs_is_fairdraw = True')
        assert src.count('RvsRecord.fair_draw(') == n_sites, \
            '{}: {} rebind sites but {} fair_draw records'.format(
                os.path.basename(p), n_sites, src.count('RvsRecord.fair_draw('))
        assert src.count('RvsRecord.retained(') == n_sites, \
            '{}: a rebind site has no pre-draw retained record'.format(os.path.basename(p))
        assert src.count('self._rvs_record = None') == n_sites, \
            '{}: a rebind site does not reset the record'.format(os.path.basename(p))
        assert 'n_retained=self._rvs_record.n_retained()' in src, \
            '{}: n_retained read back from the mutated columns'.format(os.path.basename(p))
        total_fd += src.count('RvsRecord.fair_draw(')
        total_ret += src.count('RvsRecord.retained(')
        total_reset += n_sites
    assert total_fd == total_ret == total_reset == 7, \
        'expected 7 rebind sites wired, got {}/{}/{}'.format(total_fd, total_ret, total_reset)


_INTEGRATORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'RIFT', 'integrators')


###
### BACKEND CONTRACTS: the differences are real, so make them visible rather than implicit
###

_AUDIT_BE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'expensive_before_merging', 'integrators',
                         'audit_backend_contracts.py')


def _backend_contracts():
    import importlib.util
    spec = importlib.util.spec_from_file_location('audit_backend_contracts', _AUDIT_BE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.path.exists(_AUDIT_BE), reason='backend audit not in this tree')
def test_the_recorded_backend_contracts_match_the_code():
    """The CI gate, as a unit test too: the point is not that the backends agree -- they do
    not, and that is allowed -- but that a change to one shows up as a diff."""
    mod = _backend_contracts()
    import json
    assert os.path.exists(mod.LEDGER), 'no recorded contracts; run --emit-ledger'
    want = json.load(open(mod.LEDGER))
    for b in mod.BACKENDS:
        got = mod.scan(b)
        assert b in want, '{} is not in the recorded contracts'.format(b)
        for k in sorted(set(got) | set(want[b])):
            assert got.get(k) == want[b].get(k), \
                '{}.{}: recorded {!r}, now {!r}'.format(b, k, want[b].get(k), got.get(k))


@pytest.mark.skipif(not os.path.exists(_AUDIT_BE), reason='backend audit not in this tree')
def test_the_integrand_column_really_does_mean_three_different_things():
    """Pinned because it is the specific trap that cost time twice in one afternoon, and
    because a future 'tidy-up' that collapses the three cases would be a behaviour change."""
    mod = _backend_contracts()
    holds = {b: mod.scan(b)['integrand_holds'] for b in mod.BACKENDS}
    assert holds['mcsamplerAdaptiveVolume'] == 'log (aliased)'
    assert holds['mcsamplerPortfolio'] == 'log (aliased)'
    assert holds['mcsampler'] == 'linear'
    assert holds['mcsamplerEnsemble'] == 'L or lnL (kwarg)', \
        'the runtime-dependent case is the dangerous one; it must stay visible'
    assert len(set(holds.values())) == 3, \
        'expected exactly three distinct meanings, got {}'.format(sorted(set(holds.values())))


@pytest.mark.skipif(not os.path.exists(_AUDIT_BE), reason='backend audit not in this tree')
def test_only_two_backends_keep_a_warm_seed_reserve():
    """So RvsRecord.retained_points() must answer None for the other four rather than pretend,
    and the L0 rescue / sequential warm start must keep their fallbacks."""
    mod = _backend_contracts()
    keeps = {b for b in mod.BACKENDS if mod.scan(b)['keeps_warm_seed_reserve']}
    assert keeps == {'mcsamplerAdaptiveVolume', 'mcsamplerPortfolio'}, sorted(keeps)


###
### THE UNIVERSAL OUTPUT API
###
### `_rvs` is internal.  These are what a consumer should call, and the point is that they mean
### the SAME thing on every backend -- so nobody has to know that `integrand` holds lnL on three
### samplers, linear L on two, and either on a sixth depending on a kwarg.
###

@pytest.mark.parametrize('mod_name', ['mcsampler', 'mcsamplerAdaptiveVolume',
                                      'mcsamplerEnsemble', 'mcsamplerGPU',
                                      'mcsamplerNFlow', 'mcsamplerPortfolio'])
def test_every_backend_exposes_the_public_samples_api(mod_name):
    # SOURCE first, so the wiring is checked even for a backend whose optional dependency is
    # absent (mcsamplerNFlow needs `nflows`).  A skip that checked nothing would quietly stop
    # covering a backend the day its dependency dropped out of the environment.
    src = open(os.path.join(_INTEGRATORS_DIR, '{}.py'.format(mod_name))).read()
    assert 'class MCSampler(SamplerOutputMixin' in src, \
        '{}.MCSampler does not inherit the public output API'.format(mod_name)

    import importlib
    try:
        mod = importlib.import_module('RIFT.integrators.{}'.format(mod_name))
    except ImportError as e:
        pytest.skip('{} needs an optional dependency ({}); source wiring checked above'
                    .format(mod_name, e))
    assert issubclass(mod.MCSampler, SamplerOutputMixin), \
        '{}.MCSampler does not expose samples(); consumers would reach into _rvs'.format(mod_name)
    assert callable(getattr(mod.MCSampler, 'samples', None))


def test_log_likelihood_is_lnL_whatever_the_backend_stored():
    """The whole point.  A log backend and a linear backend, same call, same meaning."""
    n = 40
    lnL = np.linspace(-5.0, 5.0, n)

    log_rec = RvsRecord.retained({'log_integrand': lnL,
                                  'log_joint_prior': np.zeros(n),
                                  'log_joint_s_prior': np.zeros(n)})
    lin_rec = RvsRecord.retained({'integrand': np.exp(lnL),
                                  'joint_prior': np.ones(n),
                                  'joint_s_prior': np.ones(n)},
                                 integrand_is_log=False)
    assert np.allclose(log_rec.log_likelihood(), lnL)
    assert np.allclose(lin_rec.log_likelihood(), lnL)
    assert np.allclose(log_rec.log_weights(), lin_rec.log_weights())


def test_a_raw_integrand_column_of_unknown_meaning_raises_rather_than_guessing():
    """The loud failure this codebase prefers.  Without a recorded convention the column's
    meaning is genuinely unrecoverable, and returning a plausible number would be the exact
    defect the backend audit documents."""
    rec = RvsRecord.retained({'integrand': np.array([1.0, 2.0, 3.0]),
                              'joint_prior': np.ones(3), 'joint_s_prior': np.ones(3)})
    assert rec.integrand_is_log is None
    with pytest.raises(ValueError) as e:
        rec.log_likelihood()
    assert 'integrand_is_log' in str(e.value)


def test_a_log_integrand_column_needs_no_convention_at_all():
    """Which is why only mcsampler and Ensemble-in-linear-mode had to be told."""
    n = 5
    rec = RvsRecord.retained({'log_integrand': np.zeros(n),
                              'log_joint_prior': np.zeros(n),
                              'log_joint_s_prior': np.zeros(n)})
    assert rec.integrand_is_log is None
    assert np.allclose(rec.log_likelihood(), 0.0)


def test_non_positive_linear_values_become_minus_inf_not_nan():
    """A rejected or underflowed row is a real zero, not a NaN, and must not poison a sum."""
    rec = RvsRecord.retained({'integrand': np.array([1.0, 0.0, -1.0]),
                              'joint_prior': np.ones(3), 'joint_s_prior': np.ones(3)},
                             integrand_is_log=False)
    lnL = rec.log_likelihood()
    assert lnL[0] == pytest.approx(0.0)
    assert np.isneginf(lnL[1]) and np.isneginf(lnL[2])
    assert not np.any(np.isnan(lnL))


def test_log_weights_needs_no_use_lnL_argument():
    """ln_weights_from_rvs must be told the convention because a bare dict cannot say what its
    own columns mean.  A record can, so the parameter disappears -- and with it the class of
    bug where a caller passes opts.internal_use_lnL instead of the stored convention."""
    import inspect
    sig = inspect.signature(RvsRecord.log_weights)
    # a host-transfer hook is fine; a CONVENTION argument is not -- the record already knows
    assert set(sig.parameters) <= {'self', 'convert'}, \
        'log_weights() grew a convention argument; the record is supposed to already know'
    for banned in ('use_lnL', 'return_lnI', 'integrand_is_log'):
        assert banned not in sig.parameters, \
            'log_weights() takes {}; the whole point is that it does not need one'.format(banned)


def test_the_ensemble_return_lnI_convention_is_recorded_by_the_sampler():
    """The case that made this necessary: for mcsamplerEnsemble the meaning of `integrand` is a
    RUNTIME property of how the pass was called, so only the sampler can record it.

    And the predicate has to be the RIGHT runtime property.  `integrand` is `value_array`,
    which is `cumulative_values` (lnL either way) under return_lnI and exp() of it otherwise;
    use_lnL only decides whether the log columns are written BESIDE it.  An earlier version
    recorded bool(use_lnL) here, which is correct on three of the four combinations and
    silently wrong on return_lnI=True, use_lnL=False -- the mislabelled linear reading sends
    every negative-lnL row to zero weight and logs the positive ones twice.
    """
    src = open(os.path.join(_INTEGRATORS_DIR, 'mcsamplerEnsemble.py')).read()
    assert 'integrand_is_log=bool(return_lnI)' in src, \
        'the Ensemble backend no longer records what its integrand column holds'
    assert 'integrand_is_log=bool(use_lnL)' not in src, \
        'use_lnL says whether log columns were written, NOT what `integrand` holds'
    src_mc = open(os.path.join(_INTEGRATORS_DIR, 'mcsampler.py')).read()
    assert 'integrand_is_log=False' in src_mc, \
        'mcsampler writes only linear columns and must say so'


def test_the_gpu_linear_entry_point_records_that_it_is_linear():
    """mcsamplerGPU.integrate() hands a use_lnL=True call off to integrate_log, so anything
    reaching its record stored a linear `integrand` and no log columns at all.  Leaving the
    convention unrecorded there makes samples().log_likelihood() raise for the DEFAULT mode of
    that backend -- the one case where the record's refusal to guess is a false alarm rather
    than a caught defect."""
    src = open(os.path.join(_INTEGRATORS_DIR, 'mcsamplerGPU.py')).read()
    # both rebind sites of the linear path: the retained record and the fair-draw one
    assert src.count('integrand_is_log=False') == 2, \
        'the GPU linear path must state its convention on BOTH the retained and fairdraw records'


def test_a_mislabelled_log_column_is_not_a_harmless_annotation():
    """Why the two findings above are defects and not bookkeeping: the same rows read under the
    wrong convention are not approximately wrong, they are a different posterior."""
    lnL = np.array([-3.0, -1.0, 2.0, 4.0])
    cols = {'integrand': lnL, 'joint_prior': np.ones(4), 'joint_s_prior': np.ones(4)}
    right = RvsRecord.retained(dict(cols), integrand_is_log=True).log_weights()
    wrong = RvsRecord.retained(dict(cols), integrand_is_log=False).log_weights()
    assert np.allclose(right, lnL)
    assert np.isneginf(wrong[0]) and np.isneginf(wrong[1])   # negative lnL -> zero weight
    assert np.allclose(wrong[2:], np.log(lnL[2:]))           # and log() of a log on the rest


###
### THE BOUNDARY: `_rvs_record` is private to the samplers; everyone else calls samples()
###

_ILE_LISA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'bin', 'integrate_likelihood_extrinsic_batchmode_lisa')


def _attribute_reads(src, attr):
    """How many times this source touches `<not-self>.attr` -> int.

    AST, not text.  Two earlier attempts got this wrong in ways worth recording:

      * a plain substring search counts the COMMENTS that explain the hazard, which in these
        files is most of the occurrences (the same false alarm PR #87 hit);
      * stripping comments and strings then counting tokens MISSES `getattr(sampler,
        '_rvs_record')` entirely -- the attribute name lives in a string literal there, and
        that is precisely the form a consumer reaching inside would use.  That version passed
        against a deliberately reintroduced violation, i.e. it was worse than no test.

    So: attribute access where the object is not `self`, PLUS getattr/setattr/hasattr with the
    name as a string constant and a non-`self` target.
    """
    import ast as _ast
    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return -1                      # never let a parse failure read as "clean"

    def _is_self(node):
        return isinstance(node, _ast.Name) and node.id == 'self'

    n = 0
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Attribute) and node.attr == attr and not _is_self(node.value):
            n += 1
        elif isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name) \
                and node.func.id in ('getattr', 'setattr', 'hasattr') and len(node.args) >= 2:
            a = node.args[1]
            name = a.value if isinstance(a, _ast.Constant) else None
            if name == attr and not _is_self(node.args[0]):
                n += 1
    return n


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_ile_never_touches_the_private_record_attribute():
    """Consumers call samples(); the producer at the pooling site calls set_samples().

    This is the property the whole design is for -- `_rvs` and `_rvs_record` are internal, and
    a consumer reaching inside is how a caller ends up depending on which backend it has.
    """
    n = _attribute_reads(open(_ILE).read(), '_rvs_record')
    assert n == 0, \
        'the ILE touches sampler._rvs_record in {} place(s); use samples()/set_samples()'.format(n)


def test_the_record_tests_use_the_public_api_too():
    """A test that reaches inside is still a consumer written against an internal, and it is
    the one place where doing so looks harmless."""
    n = _attribute_reads(open(os.path.abspath(__file__)).read(), '_rvs_record')
    assert n == 0, \
        'this suite touches ._rvs_record in {} place(s); use samples()/set_samples()'.format(n)


@pytest.mark.skipif(not os.path.exists(_ILE_LISA), reason='LISA driver not in this tree')
def test_the_lisa_driver_is_not_quietly_left_behind():
    """It is a deliberate fork, so it may legitimately have none of this -- but "none" and
    "half" are different, and half is how a fork rots.  See the driver-drift work."""
    src = open(_ILE_LISA).read()
    has_api = 'samples()' in src
    has_private = _attribute_reads(src, '_rvs_record') > 0
    assert not has_private or has_api, \
        'the LISA driver reaches into _rvs_record without using the public API'


@pytest.mark.parametrize('mod_name', ['mcsampler', 'mcsamplerAdaptiveVolume',
                                      'mcsamplerEnsemble', 'mcsamplerGPU',
                                      'mcsamplerNFlow', 'mcsamplerPortfolio'])
def test_only_the_owning_sampler_touches_its_own_record(mod_name):
    """Inside a sampler, `self._rvs_record` is the producer writing its own attribute, which is
    fine.  What must not appear is one sampler reaching into another's."""
    src = open(os.path.join(_INTEGRATORS_DIR, '{}.py'.format(mod_name))).read()
    n = _attribute_reads(src, '_rvs_record')
    assert n == 0, \
        '{} touches a _rvs_record that is not its own, in {} place(s)'.format(mod_name, n)


###
### TIER 1 (VALIDATION_rvs_weight_migration.md): the independent third implementation
###
### shape_recovery.py -- the merge gate -- carries its OWN log_weights_from_rvs(), written
### independently of both ln_weights_from_rvs and RvsRecord.log_weights().  Comparing against it
### is the check that can falsify the migration rather than testing it against itself.
###
### It is a HEURISTIC, deliberately: it guesses the convention with
### `L if np.nanmin(L) < 0 else np.log(L + 1e-300)` and floors instead of masking.  So the
### criterion is agreement on the in-support rows, not bit-identity -- and the fact that the
### gate has to guess at all is the clearest statement of why the record records instead.
###

def _shape_recovery_module():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'expensive_before_merging', 'integrators', 'shape_recovery.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('shape_recovery_for_test', path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _ile_weight_fn():
    src = open(_ILE).read()
    ns = {"numpy": np, "np": np, "_rvs_lnL_convention": lambda x=None: bool(x)}
    exec(compile(src[src.index("def ln_weights_from_rvs"):src.index("def _pool_replica_rvs")],
                 "w", "exec"), ns)
    return ns["ln_weights_from_rvs"]


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
@pytest.mark.parametrize('backend', ['AV', 'Ensemble_log', 'Ensemble_linear', 'mcsampler'])
def test_three_independent_weight_implementations_agree(backend):
    """rec.log_weights() vs ln_weights_from_rvs vs the shape gate's own derivation."""
    sr = _shape_recovery_module()
    if sr is None:
        pytest.skip('shape_recovery.py not importable here')
    import RIFT.integrators.mcsamplerAdaptiveVolume as AV
    import RIFT.integrators.mcsamplerEnsemble as ENS
    import RIFT.integrators.mcsampler as MC

    def log_t(rho=8.0):
        x0 = 0.5 * np.ones(6); w = (0.5 / rho) * np.ones(6); m = 0.5 * rho ** 2

        def f(*a, **k):
            x = np.array([np.asarray(v, float).ravel() for v in a]).T
            o = m - 0.5 * np.sum(((x - x0) / w) ** 2, axis=-1)
            return np.where(o > m - 745.0, o, -np.inf)
        return f

    def lin_t(rho=4.0):
        x0 = 0.5 * np.ones(6); w = (0.5 / rho) * np.ones(6)

        def f(*a, **k):
            x = np.array([np.asarray(v, float).ravel() for v in a]).T
            return np.exp(-0.5 * np.sum(((x - x0) / w) ** 2, axis=-1))
        return f

    np.random.seed(11)
    v = np.vectorize(lambda x: 1.0)
    kw = dict(nmax=50000, neff=30, n=5000, no_protect_names=True, verbose=False, save_intg=True)
    if backend == 'AV':
        s = AV.MCSampler(n_chunk=5000); s.xpy = AV.xpy_default
        s.identity_convert = AV.identity_convert
        for n in NAMES6:
            s.add_parameter(n, pdf=None, left_limit=0.0, right_limit=1.0,
                            prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
        s.integrate_log(log_t(), *NAMES6, **kw); use_lnL = True
    else:
        mod = MC if backend == 'mcsampler' else ENS
        s = mod.MCSampler()
        for n in NAMES6:
            s.add_parameter(n, v, prior_pdf=v, left_limit=0.0, right_limit=1.0,
                            adaptive_sampling=True)
        if backend == 'Ensemble_log':
            s.integrate(log_t(), *NAMES6, use_lnL=True, return_lnI=True, **kw); use_lnL = True
        else:
            s.integrate(lin_t(), *NAMES6, **kw); use_lnL = False

    rec = s.samples()
    assert rec is not None, '{}: no record'.format(backend)
    a = np.asarray(rec.log_weights(), dtype=float)
    b = np.asarray(_ile_weight_fn()(rec.columns, use_lnL=use_lnL), dtype=float)
    c = np.asarray(sr.log_weights_from_rvs(rec.columns), dtype=float)

    # canonical pair: exact
    assert np.array_equal(np.nan_to_num(a, nan=-9e99, neginf=-9e99),
                          np.nan_to_num(b, nan=-9e99, neginf=-9e99)), \
        '{}: rec.log_weights() disagrees with ln_weights_from_rvs'.format(backend)

    # independent heuristic: agree on the rows that carry weight.  Compare SHAPE (differences
    # from the max), since an additive offset would cancel in every downstream normalization.
    good = np.isfinite(a) & np.isfinite(c)
    assert good.sum() >= 5, '{}: too few comparable rows ({})'.format(backend, int(good.sum()))
    da = a[good] - np.max(a[good])
    dc = c[good] - np.max(c[good])
    assert np.allclose(da, dc, atol=1e-8), \
        '{}: the gate\'s independent derivation disagrees (max |diff| {:.3e})'.format(
            backend, float(np.max(np.abs(da - dc))))


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_log_weights_matches_the_canonical_form_including_out_of_support_rows():
    """Randomized equivalence with ln_weights_from_rvs, across all three column families.

    THIS is the test with teeth, and the one above is not.  `log_weights()` was first written as
    `log_likelihood() + log_prior() - log_sampling_prior()`, which is wrong on the linear family:
    the canonical form applies a CONJUNCTIVE keep-mask (`ig>0 & jp>0 & js>0`, whole row -inf),
    while evaluating the terms independently gives `-inf - (-inf) = NaN`.  A NaN weight poisons
    every downstream sum; -inf is a real zero.

    Real sampler records never expose it -- their priors are positive -- so
    `test_three_independent_weight_implementations_agree` PASSES with the bug reintroduced.
    Verified, not assumed: that is why this fuzz exists rather than resting on the end-to-end
    comparison, and why the out-of-support rows are sprinkled in deliberately.
    """
    lwf = _ile_weight_fn()
    rng = np.random.default_rng(3)
    bad = []
    for _ in range(300):
        n = int(rng.integers(3, 40))
        for kind, use, is_log in (('log', None, None),
                                  ('linear', False, False),
                                  ('linear-as-lnL', True, True)):
            if kind == 'log':
                cols = {'log_integrand': rng.normal(0, 5, n),
                        'log_joint_prior': rng.normal(0, 1, n),
                        'log_joint_s_prior': rng.normal(0, 1, n)}
            else:
                cols = {'integrand': rng.normal(0, 3, n),
                        'joint_prior': rng.normal(0, 2, n),      # NEGATIVE priors on purpose
                        'joint_s_prior': rng.normal(0, 2, n)}
            for k in list(cols):                                  # and the nasty values
                v = cols[k].copy()
                v[rng.integers(0, n)] = np.nan
                v[rng.integers(0, n)] = -np.inf
                v[rng.integers(0, n)] = 0.0
                cols[k] = v
            with np.errstate(invalid='ignore', divide='ignore'):
                a = np.asarray(lwf(cols, use_lnL=bool(use)), dtype=float)
                b = np.asarray(RvsRecord.retained(cols, integrand_is_log=is_log).log_weights(),
                               dtype=float)
            f = lambda x: np.nan_to_num(x, nan=-9e99, posinf=9e99, neginf=-9e99)
            if not np.array_equal(f(a), f(b)):
                bad.append(kind)
    assert not bad, 'log_weights() diverges from the canonical form on {} record(s): {}'.format(
        len(bad), sorted(set(bad)))


###
### ADVERSARIAL REVIEW FINDINGS (2026-08-14) -- regressions for each
###

def test_a_pooled_record_from_a_linear_backend_can_still_produce_weights():
    """REVIEW FINDING 1, the one that would have dropped events.

    _pool_replica_rvs keeps only the INTERSECTION of the replica keys, so pooling
    adaptive_cartesian (or Ensemble without use_lnL) replicas yields a bare `integrand` column.
    Built without a convention, log_weights() correctly refuses to guess -- and that ValueError
    escapes the UNWRAPPED .dgrid exporter, out of analyze_event, into the per-event handler,
    which skips the event and writes an empty .dat.  Replicas + a linear backend + .dgrid was a
    dropped event.
    """
    cols = {'integrand': np.array([1.0, 2.0, 3.0, 4.0]),
            'joint_prior': np.ones(4), 'joint_s_prior': np.ones(4)}
    unconventioned = RvsRecord.pooled(cols, [True, True], [2, 2])
    with pytest.raises(ValueError):
        unconventioned.log_weights()          # the record is right to refuse...

    # ...so the ILE must supply the convention, which it takes from the pre-pool record.
    fixed = RvsRecord.pooled(cols, [True, True], [2, 2], integrand_is_log=False)
    lw = fixed.log_weights()
    assert np.all(np.isfinite(lw)) and len(lw) == 4


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_ile_passes_a_convention_when_it_builds_the_pooled_record():
    src = open(_ILE).read()
    i = src.index('_RvsRecord.pooled(')
    block = src[max(0, i - 1600):i + 400]
    assert 'integrand_is_log=' in block, \
        'the pooled record is built with no convention; a linear backend will raise'
    assert 'rvs_integrand_is_lnL' in block, 'no fallback when the pre-pool record is absent'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_pooled_record_provenance_is_filtered_in_lockstep():
    """REVIEW FINDING 3: _pool_replica_rvs drops empty replicas together with their lnZ and
    their resampled flag; the record's block lists must be filtered the same way or they
    describe blocks the record does not contain."""
    src = open(_ILE).read()
    i = src.index('_RvsRecord.pooled(')
    block = src[max(0, i - 1600):i + 500]
    assert '_keep_rec' in block, 'the record\'s block provenance is built from unfiltered lists'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_participation_is_not_confused_with_currently_having_a_record():
    """REVIEW FINDING 4: a replica that raised leaves _rvs_record None while the sampler is
    still a full participant; keying on presence silently skips the pooled record."""
    src = open(_ILE).read()
    i = src.index('def _sampler_keeps_records')
    body = src[i:i + 1400]
    assert 'isinstance(sampler, SamplerOutputMixin)' in body, \
        'participation is still inferred from whether a record happens to be present'


def test_a_snapshotted_record_describes_the_columns_that_get_restored():
    """REVIEW FINDING 5: the restore installs a COPY of the column dict, so a record still
    pointing at the original fails every identity check and does nothing at all."""
    # ONE namespace as globals: the helpers call each other, and functions resolve names in
    # globals, so exec(code, globals, locals) leaves them unable to see one another.
    src = open(_ILE).read()
    start = src.index("def _rebound_record")
    end = src.index("def _warm_seed_geometry")
    ns = {"numpy": np, "np": np}
    exec(compile(src[start:end], "ile_state", "exec"), ns)

    class _S(SamplerOutputMixin):
        pass
    s = _S()
    s._rvs = {'log_integrand': np.zeros(3), 'log_joint_prior': np.zeros(3),
              'log_joint_s_prior': np.zeros(3)}
    s.set_samples(RvsRecord.retained(s._rvs))
    s._rvs_is_fairdraw = False; s._rvs_is_pooled = False
    s._warm_seed_reserve = None; s.portfolio_realizations = []

    cold = dict(s._rvs)
    state = ns["_snapshot_pass_state"](s, 1, 2, 3, {}, rvs=cold)
    s._rvs = {'log_integrand': np.ones(9)}            # the warm pass replaces it
    s.set_samples(RvsRecord.fair_draw(s._rvs))
    ns["_restore_pass_state"](s, state)

    assert s.samples() is not None, 'the record was dropped on restore'
    assert s.samples().columns is s._rvs, \
        'the restored record does not describe the restored columns, so it is inert'
    # which is exactly what _rvs_record_for's identity check asks (it is defined earlier in
    # the file than the slice exec'd above, so the condition is restated rather than imported)
    assert s.samples().columns is s._rvs


###
### The lnZ / Kish estimators, now record-aware
###

def _state_ns():
    src = open(_ILE).read()
    ns = {"numpy": np, "np": np, "_rvs_lnL_convention": lambda x=None: bool(x)}
    exec(compile(src[src.index("def ln_weights_from_rvs"):src.index("def _warm_seed_geometry")],
                 "ile_est", "exec"), ns)
    return ns


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_estimators_give_the_same_answer_from_a_record_or_from_the_columns():
    """A source choice, not a semantics choice -- so both routes must agree exactly."""
    ns = _state_ns()
    n = 60
    rng = np.random.default_rng(17)
    cols = {'log_integrand': rng.normal(0, 4, n),
            'log_joint_prior': rng.normal(0, 1, n),
            'log_joint_s_prior': rng.normal(0, 1, n)}
    rec = RvsRecord.retained(cols)
    for fn, kw in (('_lnZ_of_rvs', dict(already_pooled=False)), ('_kish_neff_of_rvs', {})):
        a = ns[fn](cols, record=rec, **kw)
        b = ns[fn](cols, **kw)
        assert a == pytest.approx(b, rel=1e-12), '{}: record and column routes differ'.format(fn)


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_a_record_describing_other_columns_is_not_believed_by_the_estimators():
    """The identity guard, at the estimators too.  _rvs dicts are copied and replaced all over
    the ILE; a record pointing at a different dict must be ignored, not trusted."""
    ns = _state_ns()
    rng = np.random.default_rng(18)
    mine = {'log_integrand': rng.normal(0, 4, 40),
            'log_joint_prior': np.zeros(40), 'log_joint_s_prior': np.zeros(40)}
    other = {'log_integrand': np.full(40, 99.0),
             'log_joint_prior': np.zeros(40), 'log_joint_s_prior': np.zeros(40)}
    stale = RvsRecord.retained(other)                  # describes SOMETHING ELSE
    got = ns['_lnZ_of_rvs'](mine, already_pooled=False, record=stale)
    want = ns['_lnZ_of_rvs'](mine, already_pooled=False)
    assert got == pytest.approx(want, rel=1e-12), \
        'a record describing other columns was believed; identity guard missing'
    assert got < 90.0, 'the stale record leaked into the estimate'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_estimators_share_one_weight_resolver():
    """Two copies of "prefer the record, else derive" would drift, which is the failure this
    whole branch is about."""
    src = open(_ILE).read()
    assert src.count('def _lw_of(') == 1
    for fn in ('def _lnZ_of_rvs', 'def _kish_neff_of_rvs'):
        i = src.index(fn)
        body = src[i:i + 1500]
        assert '_lw_of(rvs, record, use_lnL)' in body, \
            '{} does not go through the shared resolver'.format(fn)


###
### INTERNAL RECORDS: we had to hand the structure back; that is not the same as publishing it
###

def test_an_internal_record_cannot_be_published_through_samples():
    """The boundary that makes 'internal' mean something rather than being a naming convention.

    Replica pooling has to thread each block's record into _pool_replica_rvs so the block's
    weights are derived with ITS convention. Having had to pass the structure around is not a
    reason for a consumer to reach for it, so set_samples() refuses an internal record and the
    public accessor can therefore never yield one.
    """
    class _S(SamplerOutputMixin):
        pass
    s = _S()
    pub = RvsRecord.retained(_cols(10))
    s.set_samples(pub)
    assert s.samples() is pub

    internal = pub.as_internal()
    assert internal.internal is True
    with pytest.raises(ValueError) as e:
        s.set_samples(internal)
    assert 'INTERNAL' in str(e.value)
    assert s.samples() is pub, 'the refused call must leave the public record untouched'


def test_as_internal_shares_the_data_and_changes_only_the_marker():
    """It is a view for threading, not a copy -- copying every replica's columns would
    reintroduce the memory cost the reserve-by-reference decision avoided."""
    pub = RvsRecord.fair_draw(_cols(12), n_retained=999, reserve={'X': np.zeros((2, 2))})
    it = pub.as_internal()
    assert it.columns is pub.columns
    assert it.provenance is pub.provenance
    assert it.reserve is pub.reserve
    assert it.integrand_is_log == pub.integrand_is_log
    assert pub.internal is False and it.internal is True
    assert it.rows_are_resampled() == pub.rows_are_resampled()
    assert it.n_retained() == 999


def test_the_internal_marker_survives_a_snapshot():
    """Otherwise snapshot/restore would launder an internal record into a publishable one."""
    it = RvsRecord.retained(_cols(6)).as_internal()
    assert it.snapshot().internal is True


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_ile_threads_per_replica_records_and_marks_them_internal():
    src = open(_ILE).read()
    assert 'def _internal_record_of' in src
    i = src.index('_rep_records = [')
    assert '_internal_record_of(sampler)' in src[i:i + 200], \
        'per-replica records are captured without being marked internal'
    j = src.index('_pool_replica_rvs(_rep_rvs')
    assert 'records=_rep_records' in src[j:j + 500], \
        'the per-replica records are not threaded into pooling'
    # and pooling filters them in lockstep with the other per-replica lists
    k = src.index('def _pool_replica_rvs')
    body = src[k:k + 4000]
    assert '_rec_list = [_rec_list[i] for i in _keep' in body, \
        'the records are not filtered in lockstep with rep_rvs/rep_lnZ'
    assert 'def _block_record' in body, 'no per-block identity guard on the threaded records'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_pooling_uses_a_block_record_only_when_it_describes_that_block():
    """The identity guard again, one level down: a record for replica 2 must not be used to
    derive replica 1's lnZ just because the lists line up."""
    ns = {"numpy": np, "np": np, "_rvs_lnL_convention": lambda x=None: bool(x)}
    src = open(_ILE).read()
    exec(compile(src[src.index("def ln_weights_from_rvs"):src.index("def _warm_seed_geometry")],
                 "ile_pool", "exec"), ns)

    class _Conv(object):
        @staticmethod
        def identity_convert(x):
            return x
    rng = np.random.default_rng(21)
    blocks = []
    for sd in (1, 2):
        n = 30
        blocks.append({'log_integrand': rng.normal(0, 2, n),
                       'log_joint_prior': np.zeros(n), 'log_joint_s_prior': np.zeros(n)})
    good = [RvsRecord.retained(b).as_internal() for b in blocks]
    mismatched = [RvsRecord.retained(blocks[1]).as_internal(),
                  RvsRecord.retained(blocks[0]).as_internal()]   # swapped on purpose

    kw = dict(rep_lnZ=[7.0, 9.0], already_resampled=[False, False], use_lnL=False)
    a = ns['_pool_replica_rvs'](list(blocks), _Conv(), records=good, **kw)
    b = ns['_pool_replica_rvs'](list(blocks), _Conv(), records=mismatched, **kw)
    c = ns['_pool_replica_rvs'](list(blocks), _Conv(), records=None, **kw)
    lw = lambda o: ns['ln_weights_from_rvs'](o, use_lnL=False)
    assert np.allclose(lw(a), lw(c)), 'the record route changed the pooled weights'
    assert np.allclose(lw(b), lw(c)), \
        'a record describing ANOTHER block was used; the identity guard is missing'
