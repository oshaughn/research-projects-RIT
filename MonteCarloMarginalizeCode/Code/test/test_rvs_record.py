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


class _Sampler(object):
    """A sampler carrying BOTH descriptions, as the tree does mid-migration."""
    def __init__(self, record, is_fairdraw, is_pooled):
        self._rvs_record = record
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
    n_direct = body.count("getattr(sampler, '_rvs_record', None)")
    assert n_direct == 0, \
        '{} consumer(s) read _rvs_record directly, bypassing the identity check'.format(n_direct)
    assert body.count('_rvs_record_for(sampler') >= 3, \
        'expected the weight helper, the .dslice guard and the pooled n_eff to share the lookup'

    # the PRODUCER at the pooling site asks a different question and has its own name: it is
    # about to replace sampler._rvs, so "does a record describe the rows I hold" is wrong there
    assert '_sampler_keeps_records(sampler)' in body, \
        'the pooling producer should ask whether the sampler keeps records at all'
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
    rec = s._rvs_record
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
    rec = s._rvs_record
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
    assert s._rvs_record is not None and s._rvs_is_fairdraw

    with_record = ln_w_post(s._rvs, s)
    stashed, s._rvs_record = s._rvs_record, None      # force the flag path
    without_record = ln_w_post(s._rvs, s)
    s._rvs_record = stashed
    assert np.array_equal(with_record, without_record), \
        'the record path and the flag path disagree; the migration is not a refactor'

    # ...and the same on a pass with no fair draw, where the answer is the other branch
    np.random.seed(20260813)
    s2 = _av_sampler()
    s2.integrate_log(_av_peaked(100.0), *NAMES6, nmax=400000, neff=8, n=20000,
                     no_protect_names=True, verbose=False)
    a = ln_w_post(s2._rvs, s2)
    s2._rvs_record = None
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
    rec = s._rvs_record
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

from RIFT.integrators.rvs_record import SamplerOutputMixin  # noqa: E402


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
    assert list(sig.parameters) == ['self'], \
        'log_weights() grew a convention argument; the record is supposed to already know'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_ensemble_return_lnI_convention_is_recorded_by_the_sampler():
    """The case that made this necessary: for mcsamplerEnsemble the meaning of `integrand` is a
    RUNTIME property of how the pass was called, so only the sampler can record it."""
    src = open(os.path.join(_INTEGRATORS_DIR, 'mcsamplerEnsemble.py')).read()
    assert 'integrand_is_log=bool(use_lnL)' in src, \
        'the Ensemble backend no longer records what its integrand column holds'
    src_mc = open(os.path.join(_INTEGRATORS_DIR, 'mcsampler.py')).read()
    assert 'integrand_is_log=False' in src_mc, \
        'mcsampler writes only linear columns and must say so'
