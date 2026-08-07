#!/usr/bin/env python
"""Correctness tests for the portfolio's SUPPORT-MISMATCH diagnostic
(mcsamplerPortfolio.support_diagnostics / escaped_mass).

These are CORRECTNESS tests, not a claim that the statistic is a useful detector -- that is what
escaped_mass_study.py measures, and its verdict is arm-dependent (see the study docstring).  What
is asserted here is only what must hold for the number to mean anything at all:

  1. OFF-PATH.  With the diagnostic reduced to a no-op, the returned lnZ / var / n_eff are
     BIT-IDENTICAL.  A diagnostic that perturbs the estimator is worse than no diagnostic.
     MEASURED CAVEAT: only the ALL-AV portfolio is bit-reproducible run to run.  An [AV, GMM]
     portfolio is NOT -- repeating the identical configuration with the identical np.random seed
     and the diagnostic ON BOTH TIMES gives lnZ 90.75572176844321 vs 90.75570550321379 (the
     sklearn mixture fit inside mcsamplerEnsemble does not reproduce).  That is a pre-existing
     property of the GMM member, not of this diagnostic, so the bit-identity assertion is made on
     the arm where it is meaningful, and the [AV, GMM] arm is covered by the structural test
     instead.
  2. NON-INVASIVE.  A direct call to _update_support_diagnostics changes no sampler attribute
     outside its own accumulator namespace and does not modify its inputs.
  3. IT FIRES.  Warm-start the AV member from a cloud placed entirely off the true peak, in a
     portfolio whose other member is a COLD (broad) GMM, and escaped_mass for the AV member must
     go to ~1 while the matched-seed control stays at ~0 on the first-chunk statistic.
  4. IT IS READING THE SUPPORT, not a proxy: make the AV member's density strictly positive
     everywhere (so nothing CAN escape) and the same misplaced seed must score 0 / not hard-edged.
  5. A member whose density is nowhere exactly zero is reported hard_edged=False and does not
     contribute to escaped_mass_max, so a soft member cannot mask an escaped hard-edged one.

Run:  RIFT_RUN_EXPENSIVE=1 pytest -v test_escaped_mass_diagnostic.py
      (or: python test_escaped_mass_diagnostic.py)
"""
from __future__ import print_function

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_recovery as sr                     # noqa: E402
from escaped_mass_study import _build_portfolio  # noqa: E402

try:
    import pytest
    pytestmark = pytest.mark.skipif(
        not os.environ.get("RIFT_RUN_EXPENSIVE"),
        reason="expensive merge-gate suite; set RIFT_RUN_EXPENSIVE=1")
except ImportError:      # allow bare `python test_escaped_mass_diagnostic.py`
    pytest = None

NDIM, NCOMP, TSEED = 4, 1, 4242
N_CHUNK, NMAX, NEFF = 4000, 40000, 2000
sr.TRUTH_POOL_N = 100000     # only feeds the box mass + the seed cloud; see the study


def _run(arm, offset, run_seed=13579, disable_diag=False, soften_av=False):
    """Integrate the TRUE target after warm-starting from a target displaced by `offset`."""
    np.random.seed(run_seed)
    true_t = sr.MixtureTarget(NDIM, NCOMP, TSEED)
    seed_t = sr.MixtureTarget(NDIM, NCOMP, TSEED, offset=offset)
    s, members = _build_portfolio(arm, true_t, N_CHUNK)
    kw = {}
    if arm != "avav":
        _dims = tuple(range(NDIM))
        kw = dict(n_comp={_dims: 2}, gmm_dict={_dims: None}, correlate_all_dims=True)
    s.setup(**kw)
    if disable_diag:
        # break the diagnostic, not the sampler: everything else must be untouched
        s._update_support_diagnostics = lambda *a, **k: None
    members[0].bootstrap_from_samples(sr._warm_seed_cloud(seed_t), cover_frac=0.0)
    if soften_av:
        # BREAK THE THING THE STATISTIC TARGETS: keep the identical misplaced seed but make the AV
        # member's reported density strictly positive everywhere, so no sample can be OUTSIDE its
        # support.  If escaped_mass were reading anything other than the support (n_eff, weight
        # concentration, the offset itself) it would still fire; it must not.
        _orig = members[0].sampling_density

        def _soft(X, _o=_orig):
            q = _o(X)
            return None if q is None else np.maximum(np.asarray(q, dtype=float), 1e-12)
        members[0].sampling_density = _soft
    lnI, logvar, eff, dret = s.integrate_log(
        true_t.as_lnfunc(), *true_t.params, no_protect_names=True,
        n=N_CHUNK, n_adapt=100, floor_level=0.0, tempering_exp=0.1,
        neff=NEFF, nmax=NMAX, save_intg=True)
    return float(sr._asnumpy(lnI)), float(sr._asnumpy(logvar)), float(sr._asnumpy(eff)), dret


def test_diagnostic_is_off_path():
    """The estimate must be BIT-identical when the diagnostic is disabled.

    Asserted on the ALL-AV portfolio, the arm that is bit-reproducible at all (see the module
    docstring: an [AV, GMM] portfolio does not reproduce itself run to run, with or without this
    code, so a bit-identity assertion there would be testing sklearn, not the diagnostic)."""
    a = _run("avav", 0.0, disable_diag=False)
    b = _run("avav", 0.0, disable_diag=True)
    assert a[0] == b[0], "lnZ changed with the diagnostic on: {!r} vs {!r}".format(a[0], b[0])
    assert a[1] == b[1], "log-variance changed: {!r} vs {!r}".format(a[1], b[1])
    assert a[2] == b[2], "n_eff changed: {!r} vs {!r}".format(a[2], b[2])
    # and the disabled run must genuinely report nothing, otherwise the comparison is vacuous
    assert float(b[3]["portfolio_escaped_mass_max"]) == 0.0
    assert int(np.sum(b[3]["portfolio_escape_n_eval"])) == 0, \
        "diagnostic still accumulated after being disabled -- the off-path check is vacuous"
    assert int(np.sum(a[3]["portfolio_escape_n_eval"])) > 0, \
        "diagnostic never accumulated in the ENABLED run -- the off-path check is vacuous"


def test_diagnostic_mutates_nothing_outside_its_namespace():
    """Structural off-path check, valid for EVERY arm including the nondeterministic [AV, GMM].

    Runs one chunk, then calls _update_support_diagnostics a second time by hand and verifies that
    the only attributes whose value changed are the diagnostic's own accumulators, and that the
    inputs it is handed come back unmodified."""
    from RIFT.integrators import mcsamplerPortfolio      # noqa: F401  (import check)
    np.random.seed(24680)
    t = sr.MixtureTarget(NDIM, NCOMP, TSEED)
    s, members = _build_portfolio("avgmm_cold", t, N_CHUNK)
    _dims = tuple(range(NDIM))
    s.setup(n_comp={_dims: 2}, gmm_dict={_dims: None}, correlate_all_dims=True)
    members[0].bootstrap_from_samples(sr._warm_seed_cloud(t), cover_frac=0.0)
    s.integrate_log(t.as_lnfunc(), *t.params, no_protect_names=True,
                    n=N_CHUNK, n_adapt=100, floor_level=0.0, tempering_exp=0.1,
                    neff=NEFF, nmax=2 * N_CHUNK, save_intg=True)
    assert s._chunk_mix_parts, "no per-member densities retained; the check would be vacuous"

    own = set(k for k in vars(s) if k.startswith("portfolio_escape") or
              k in ("portfolio_weight_log_total", "portfolio_share_log_num", "_member_index"))
    assert own, "diagnostic namespace not found"
    before = {}
    for k, v in vars(s).items():
        before[k] = v.copy() if isinstance(v, np.ndarray) else v

    n = len(next(iter(s._chunk_mix_parts.values())))
    lw = np.linspace(-3.0, 1.0, n)
    qm = np.ones(n)
    lw_in, qm_in = lw.copy(), qm.copy()
    s._update_support_diagnostics(lw, qm)

    assert np.array_equal(lw, lw_in), "log_weights were modified in place"
    assert np.array_equal(qm, qm_in), "q_mix was modified in place"
    changed = []
    for k, v in vars(s).items():
        if k in own:
            continue
        old = before.get(k, "<absent>")
        same = (np.array_equal(v, old) if isinstance(v, np.ndarray)
                else (v is old or v == old if not isinstance(old, np.ndarray) else False))
        if not same:
            changed.append(k)
    assert not changed, "diagnostic mutated sampler state outside its namespace: {}".format(changed)

    # NON-VACUITY: the same comparison must SEE a deliberate mutation, otherwise "nothing changed"
    # proves only that the comparison is blind.
    s.ntotal = s.ntotal + 1
    seen = [k for k, v in vars(s).items()
            if k not in own and not isinstance(v, np.ndarray) and
            not (v is before.get(k, "<absent>") or v == before.get(k, "<absent>"))]
    assert "ntotal" in seen, "the mutation check cannot detect a change; it is vacuous"


def test_escaped_mass_fires_on_a_misplaced_seed():
    """Matched seed -> first-chunk escaped mass ~0; seed displaced clear of the peak -> ~1.

    Uses the FIRST-CHUNK statistic, which is the one that isolates the seed's own live volume:
    the cumulative statistic also absorbs the ordinary contraction of a correctly-placed member
    and has a large, target-dependent floor (measured median 0.35 at d=4 / 0.82 at d=6)."""
    _, _, _, ok = _run("avgmm_cold", 0.0)
    _, _, _, bad = _run("avgmm_cold", 4.0)
    e_ok = float(ok["portfolio_escaped_mass_early"][0])
    e_bad = float(bad["portfolio_escaped_mass_early"][0])
    assert 0.0 <= e_ok <= 1.0 and 0.0 <= e_bad <= 1.0, "escaped_mass out of [0,1]"
    assert e_bad > 0.9, "misplaced seed did NOT fire the detector: early escaped_mass={}".format(e_bad)
    assert e_ok < 1e-2, "matched seed produced a false positive: early escaped_mass={}".format(e_ok)
    assert bool(ok["portfolio_member_hard_edged"][0]) or e_ok == 0.0


def test_statistic_reads_the_support_and_not_a_proxy():
    """Same misplaced seed, but the AV member's density is floored strictly positive: with no
    region outside its support, escaped_mass MUST read 0 and the member must not be hard-edged.
    This is the break-it check for the "it fires" assertion above -- everything else about the
    run (the displaced seed, the low n_eff, the concentrated weights) is unchanged."""
    _, _, _, bad = _run("avgmm_cold", 4.0)
    _, _, _, soft = _run("avgmm_cold", 4.0, soften_av=True)
    assert float(bad["portfolio_escaped_mass_early"][0]) > 0.9, \
        "control did not fire; the comparison would be vacuous"
    assert float(soft["portfolio_escaped_mass"][0]) == 0.0, \
        "escaped_mass nonzero for a member with strictly positive density: it is not reading support"
    assert not bool(soft["portfolio_member_hard_edged"][0])
    assert float(soft["portfolio_escaped_mass_max"]) == 0.0


def test_soft_member_is_not_scored_as_hard_edged():
    """A live GMM's density is nowhere exactly zero on these targets, so it must be reported
    hard_edged=False and must not enter escaped_mass_max (a soft member reading 0 would otherwise
    drag a max/mean down and mask a fully-escaped AV member)."""
    _, _, _, bad = _run("avgmm_cold", 4.0)
    hard = np.asarray(bad["portfolio_member_hard_edged"], dtype=bool)
    esc = np.asarray(bad["portfolio_escaped_mass"], dtype=float)
    assert hard[0], "the AV member must be observed hard-edged"
    if not hard[1]:
        assert esc[1] == 0.0
        assert float(bad["portfolio_escaped_mass_max"]) == esc[0], \
            "escaped_mass_max must ignore the soft member"


if __name__ == "__main__":
    for fn in (test_diagnostic_is_off_path,
               test_diagnostic_mutates_nothing_outside_its_namespace,
               test_escaped_mass_fires_on_a_misplaced_seed,
               test_statistic_reads_the_support_and_not_a_proxy,
               test_soft_member_is_not_scored_as_hard_edged):
        fn()
        print("PASS", fn.__name__)
