#!/usr/bin/env python
"""
Regression tests for the SEQUENTIAL WARM-START seed
(--sampler-sequential-warmstart, bin/integrate_likelihood_extrinsic_batchmode).

This is the same defect PR #78 fixed for the L0 auto-rescue, one code path away, found by
the mechanical `_rvs` audit in test/expensive_before_merging/integrators/audit_rvs_fairdraw.py.
With --n-events-to-analyze > 1, each intrinsic point may seed the next point's extrinsic
integral from the samples it drew.  The capture block read them out of `sampler._rvs`, and by
the time it runs, integrate_log has REBOUND every _rvs key to a fair-draw subset of

    min(n_extr, 1.5*eff_samp, 1.5*neff)

rows taken WITH REPLACEMENT -- a resample built for EXPORT.  Two consequences, both of which
this suite pins:

  1. ON A COLLAPSED PASS THERE IS ALMOST NOTHING LEFT TO SEED FROM.  eff_samp ~ 1 gives one
     row ("Fairdraw size : 1" in the rho_net 146.8 logs) out of a thousand retained, and
     drawing WITH REPLACEMENT is why a "2-point seed" can have affine rank 0 -- two copies of
     one point.  This is not an exotic configuration: every extrinsic stage built by
     create_event_parameter_pipeline_BasicIteration, cepp_basic_htcondor and
     create_event_nr_pipeline_with_cip passes --fairdraw-extrinsic-output unconditionally.

  2. THE GUARD WAS A COUNT (`_lnv.size >= 2`, `np.sum(_keep) >= 2`), which is exactly the
     rule build_warm_seed exists to replace.  n points span at most n-1 affine dimensions, so
     a 2-row seed passes the count and is still rank-deficient in 6.  The next point then
     warm-starts into a degenerate sliver and reports a healthy n_eff over truncated support:
     the QUIET failure mode, which is why this survived while the L0 one was found.

The requirement is the same one PR #78 set: seed from the points the pass RETAINED, and judge
the seed by RANK, repairing it before it reaches the sampler.
"""

import os

import numpy as np
import pytest

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV
from RIFT.integrators.mcsamplerAdaptiveVolume import build_warm_seed, seed_affine_rank

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
NDIM = len(NAMES)
LO = np.zeros(NDIM)
HI = np.ones(NDIM)
AX = list(range(NDIM))

_ILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '..', 'bin', 'integrate_likelihood_extrinsic_batchmode')


def _sampler(n_chunk=10000):
    """Bound to the ACTIVE backend exactly as the ILE does; see test_l0_rescue_seed."""
    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = mcsamplerAV.xpy_default
    s.identity_convert = mcsamplerAV.identity_convert
    for name in NAMES:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    return s


def _peaked(rho, x0=None, widths=None):
    """6-D Gaussian at lnL scale rho^2/2, with the float64 underflow of the real code."""
    x0 = 0.5 * np.ones(NDIM) if x0 is None else np.asarray(x0, dtype=float)
    w = (0.5 / rho) * np.ones(NDIM) if widths is None else np.asarray(widths, dtype=float)
    lnLmax = 0.5 * rho ** 2

    def lnL(*args, **kwargs):
        x = np.array([np.asarray(a, dtype=float).ravel() for a in args]).T
        out = lnLmax - 0.5 * np.sum(((x - x0) / w) ** 2, axis=-1)
        return np.where(out > lnLmax - 745.0, out, -np.inf)
    return lnL


def _load_reserve_lookup():
    """Exec just `_warm_seed_reserve_for` out of the ILE script.

    The ILE is an executable, not an importable module, and importing it would run an
    argument parser and a stack of optional dependencies.  The helper is self-contained
    (builtins only), so lifting its source is enough to test its BEHAVIOUR rather than
    merely asserting that some words appear in the file.
    """
    with open(_ILE) as f:
        src = f.read()
    start = src.index('def _warm_seed_reserve_for')
    end = src.index('def _warm_seed_geometry')
    ns = {}
    exec(compile(src[start:end], 'ile_warm_seed_reserve_for', 'exec'), ns)
    return ns['_warm_seed_reserve_for']


class _FakeSampler(object):
    def __init__(self, params_ordered, reserve=None, members=()):
        self.params_ordered = list(params_ordered)
        if reserve is not None:
            self._warm_seed_reserve = reserve
        self.portfolio_realizations = list(members)


def _reserve(params_ordered, n=10):
    return dict(X=np.zeros((n, len(params_ordered))), lnL=np.zeros(n),
                n_retained=n, n_finite=n, ln_sum_w_finite=0.0,
                params_ordered=list(params_ordered))


###
### 1. the shared reserve lookup -- behaviour, not text
###

def test_the_reserve_lookup_finds_a_reserve_on_the_sampler_itself():
    look = _load_reserve_lookup()
    r = _reserve(NAMES)
    assert look(_FakeSampler(NAMES, reserve=r)) is r


def test_the_reserve_lookup_falls_through_to_a_portfolio_member():
    """A portfolio keeps the reserve on the aggregate, but a bare AV member can be the one
    holding it; the L0 rescue relied on this fallback and the sequential capture must too."""
    look = _load_reserve_lookup()
    r = _reserve(NAMES)
    s = _FakeSampler(NAMES, members=[_FakeSampler(NAMES), _FakeSampler(NAMES, reserve=r)])
    assert look(s) is r


def test_the_reserve_lookup_declines_a_reserve_in_the_wrong_column_order():
    """Silent scrambling is worse than no seed: X is read positionally against
    params_ordered, so a permuted reserve seeds the next point in wrong coordinates."""
    look = _load_reserve_lookup()
    scrambled = list(reversed(NAMES))
    assert look(_FakeSampler(NAMES, reserve=_reserve(scrambled))) is None


def test_the_reserve_lookup_returns_none_when_nobody_kept_one():
    look = _load_reserve_lookup()
    assert look(_FakeSampler(NAMES)) is None
    assert look(_FakeSampler(NAMES, members=[_FakeSampler(NAMES)])) is None


###
### 2. the substance: the fair draw starves this seed, the reserve does not
###

@pytest.mark.parametrize('rho', [60.0, 100.0])
def test_the_fair_draw_leaves_far_fewer_rows_than_the_pass_retained(rho):
    """The precondition for everything below.  Assert it explicitly rather than assume it:
    a test that lands on a healthy pass proves nothing about the collapsed one."""
    np.random.seed(20260813)
    s = _sampler(20000)
    s.integrate_log(_peaked(rho), *NAMES, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    n_rvs = len(np.asarray(mcsamplerAV.identity_convert(s._rvs['log_integrand'])).ravel())
    reserve = s._warm_seed_reserve
    assert reserve is not None, 'no reserve to compare against'
    assert reserve['n_retained'] > n_rvs, \
        'fair draw kept {} of {} retained -- this pass did not collapse, so it cannot ' \
        'exercise the defect'.format(n_rvs, reserve['n_retained'])


def test_a_seed_taken_from_the_fair_draw_is_rank_deficient_where_the_reserve_is_not():
    """The defect and its fix, side by side on one real pass."""
    np.random.seed(20260813)
    s = _sampler(20000)
    s.integrate_log(_peaked(100.0), *NAMES, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    conv = mcsamplerAV.identity_convert
    from_rvs = np.vstack([np.asarray(conv(s._rvs[p]), dtype=float).ravel()
                          for p in s.params_ordered]).T
    rank_rvs, _ = seed_affine_rank(from_rvs, LO, HI, axes=AX)

    reserve = s._warm_seed_reserve
    seed, info = build_warm_seed(np.asarray(reserve['X'], dtype=float),
                                 np.asarray(reserve['lnL'], dtype=float).ravel(),
                                 LO, HI, AX, deltalnL=15.0)
    assert rank_rvs < NDIM, \
        'the fair-drawn rows spanned {}/{} -- no defect to fix on this pass'.format(rank_rvs, NDIM)
    assert info['rank_final'] >= NDIM, \
        'the reserve-based seed is still rank {}/{}'.format(info['rank_final'], NDIM)
    assert len(seed) >= 2


def test_the_old_count_rule_had_no_good_outcome_on_a_collapsed_pass():
    """The count rule fails in BOTH directions, and which one you get is luck.

    Measured on this pass the fair draw keeps ONE row ("Fairdraw size : 1", the rho_net 146.8
    regime), so the count DECLINES and --sampler-sequential-warmstart is silently inert -- the
    user asked for a warm start and got none, with no message.  At the rho_net 102.8 regime it
    keeps ~5, the count ACCEPTS, and the seed is rank 2-4 of 6: the next point warm-starts into
    a sliver and reports a healthy n_eff over truncated support.

    So the assertion is not "the count accepted it".  It is that the count rule cannot produce
    a usable seed here either way, while the reserve can.
    """
    np.random.seed(20260813)
    s = _sampler(20000)
    s.integrate_log(_peaked(100.0), *NAMES, nmax=400000, neff=8, n=20000,
                    no_protect_names=True, verbose=False,
                    igrand_fairdraw_samples=True, igrand_fairdraw_samples_max=200)
    conv = mcsamplerAV.identity_convert
    lnv = np.asarray(conv(s._rvs['log_integrand']), dtype=float).ravel()
    cols = np.vstack([np.asarray(conv(s._rvs[p]), dtype=float).ravel()
                      for p in s.params_ordered]).T
    keep = lnv > (np.nanmax(lnv) - 15.0)
    old_seed = cols[keep] if np.sum(keep) >= 2 else (cols if cols.shape[0] >= 2 else None)
    if old_seed is None:
        declined = True
        rank = 0
    else:
        declined = False
        rank, _ = seed_affine_rank(old_seed, LO, HI, axes=AX)
    assert declined or rank < NDIM, \
        'the fair-drawn rows spanned {}/{} -- this pass does not exercise the defect'.format(rank, NDIM)

    # ...and the reserve, on the same pass, does produce one.
    r = s._warm_seed_reserve
    _, info = build_warm_seed(np.asarray(r['X'], dtype=float),
                              np.asarray(r['lnL'], dtype=float).ravel(),
                              LO, HI, AX, deltalnL=15.0)
    assert info['rank_final'] >= NDIM


def test_duplicate_rows_from_sampling_with_replacement_span_nothing():
    """Why 'n rows' and 'n distinct points' are not the same quantity after a fair draw."""
    pt = np.full((1, NDIM), 0.5)
    assert seed_affine_rank(np.repeat(pt, 5, axis=0), LO, HI, axes=AX)[0] == 0


###
### 3. the ILE actually wires it that way
###

def _read_ile():
    with open(_ILE) as f:
        return f.read()


def _read_ile_code():
    """The ILE source, comments and docstrings removed AND all whitespace squeezed out.

    These assertions are about what the code DOES, and the comments that explain why a rule
    went quote that rule verbatim -- so a naive substring search reports the explanation as a
    regression.  (It did, on the first run of this suite.)

    The result is WHITESPACE-FREE, so every needle below must be written whitespace-free too.
    That is the point: it also makes the assertions immune to reformatting.
    """
    import io
    import tokenize
    src = _read_ile()
    out = []
    prev_end = (1, 0)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError):
        # never let a tokenizer quirk turn into a false pass
        return src
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and tok.start[1] == 0:
            continue          # module/def-level docstring on its own line
        if tok.start[0] != prev_end[0]:
            out.append('\n')
        out.append(tok.string)
        prev_end = tok.end
    return ''.join(out)


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_seq_warmstart_capture_seeds_from_the_reserve():
    src = _read_ile()
    i = src.index('_SEQ_WS_PENDING = _seed_ws')
    block = src[max(0, i - 3000):i]
    assert '_warm_seed_reserve_for(sampler)' in block, \
        'the sequential capture still seeds from _rvs, which the fair draw has truncated'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_seq_warmstart_capture_goes_through_the_rank_tested_builder():
    src = _read_ile()
    i = src.index('_SEQ_WS_PENDING = _seed_ws')
    block = src[max(0, i - 3000):i]
    assert 'build_warm_seed' in block, 'the sequential seed is not rank-tested or puffed'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_seq_warmstart_capture_no_longer_guards_on_a_row_count():
    """The CODE, not the comment that explains why it went."""
    code = _read_ile_code()          # whitespace-free; needles must be too
    assert 'np.sum(_keep)>=2' not in code, 'the count rule is back'
    assert '_keep=_lnv>(np.nanmax(_lnv)' not in code, \
        'the inline deltalnL window is back; build_warm_seed owns that cut'


@pytest.mark.skipif(not os.path.exists(_ILE), reason='ILE executable not in this tree')
def test_the_rescue_and_the_sequential_capture_share_one_reserve_lookup():
    """Five defects of this shape have come from two copies of one idea drifting apart.
    Both callers go through the single helper, so a fix to one cannot miss the other."""
    code = _read_ile_code()          # whitespace-free; needles must be too
    assert code.count('def_warm_seed_reserve_for') == 1
    # one definition + exactly two call sites (the L0 rescue and the sequential capture)
    assert code.count('_warm_seed_reserve_for(sampler)') == 3, \
        'expected the shared lookup to have exactly two callers, found {}'.format(
            code.count('_warm_seed_reserve_for(sampler)') - 1)
    # and the old inline duplicate is gone
    assert "_res_l0=getattr(sampler,'_warm_seed_reserve',None)" not in code, \
        'the inline reserve lookup is back; it will drift from the shared one'
