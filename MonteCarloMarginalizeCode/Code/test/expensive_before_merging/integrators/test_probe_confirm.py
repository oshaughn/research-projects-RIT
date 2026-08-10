#!/usr/bin/env python
"""Unit tests for the flag-ON probe: its confirm-on-fail accounting, the settings that would let a
flagged row be cleared with no evidence, the build_sampler patching the two arms depend on, and
main() itself -- the exit code a merge gate actually reads.

Same principle as test_confirm_regressions.py: every check targets the direction that can ship a
bug -- a false CLEAR of a real opt-in regression. A false block only costs a rerun.

These are CHEAP (no sampler runs: the gate seams are stubbed), so unlike the rest of this directory
they belong in the ordinary test path and carry no RIFT_RUN_EXPENSIVE guard.

Run:  pytest -q test_probe_confirm.py
      python test_probe_confirm.py
CI:   .github/workflows/ci.yml, job `integrator-gate-accounting-check` (and the GitLab mirror's
      `integrator_gate_accounting_check`).
"""
import os
import sys

# The probe and the gate live beside this file; pytest invoked from the repo root has not put that
# directory on the path.  Without this the file is COLLECTED and then errors on import, which reads
# as a broken test rather than as the coverage it is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probe_portfolio_optin_flags as PB   # noqa: E402
import shape_recovery as SR                # noqa: E402


def _rec(n_eff=3000.0, js=0.0001, bias=0.001):
    return dict(kind="portfolio", target="t", ndim=4, ncomp=1, target_seed=101, n_eff=n_eff,
                n_ess=n_eff * 3, js=[js, js], js_floor=[0.0005, 0.0005],
                mean_pull=[0.005, 0.005], width_ratio=[1.001, 1.001], corr_diff_max=0.005,
                rel_err=0.01, bias_ln=bias, error=None)


def test_regression_rule_tolerates_starved_but_not_fail():
    """The probe's rule differs from the comparator's on purpose: an opt-in path may trade
    efficiency on a target the default already resolves, but must not break it."""
    assert PB.is_probe_regression("PASS", "FAIL")
    assert not PB.is_probe_regression("PASS", "STARVED")
    assert not PB.is_probe_regression("PASS", "PASS")
    assert not PB.is_probe_regression("STARVED", "FAIL")     # base was not passing either
    # and it accepts the (status, reasons) pair form the probe actually stores
    assert PB.is_probe_regression(("PASS", []), ("FAIL", ["js too big"]))


def _drive(seq, seeds=(1, 2, 3), min_valid=None):
    """Run confirm_flagged with _run_one_cell stubbed to a scripted (default, flag) sequence."""
    calls = {"i": 0}

    def fake(flags, job, nmax_per_dim, neff, run_seed):
        pair = seq[calls["i"] // 2]
        out = pair[0] if (calls["i"] % 2 == 0) else pair[1]
        calls["i"] += 1
        return out

    orig = PB._run_one_cell
    PB._run_one_cell = fake
    try:
        return PB.confirm_flagged([("adaptive_alloc ON", {"portfolio_adaptive_alloc": True},
                                    (4, 1, 303))],
                                  200000, 3000, list(seeds),
                                  len(seeds) if min_valid is None else min_valid)
    finally:
        PB._run_one_cell = orig


def test_noise_clears():
    good = _rec()
    n_conf, n_inconc = _drive([(good, good)] * 3)
    assert (n_conf, n_inconc) == (0, 0), (n_conf, n_inconc)


def test_real_regression_confirms():
    good, bad = _rec(), _rec(js=0.05, bias=0.9)      # flag arm genuinely FAILs the metrics
    n_conf, n_inconc = _drive([(good, bad)] * 3)
    assert n_conf == 1, "a reproducible flag-arm failure was not confirmed"


def test_flag_arm_producing_no_record_counts_against_the_flag():
    """Crashing is worse than passing; discarding those pairs would clear a flag that always dies."""
    good = _rec()
    n_conf, n_inconc = _drive([(good, None)] * 3)
    assert n_conf == 1, "flag arm produced no record on every seed but was cleared"


def test_no_valid_pairs_is_inconclusive_not_a_pass():
    n_conf, n_inconc = _drive([(None, None)] * 3)
    assert n_inconc == 1 and n_conf == 0, (n_conf, n_inconc)


def test_minority_worse_does_not_confirm():
    """One bad seed out of three is the realization sensitivity this exists to absorb."""
    good, bad = _rec(), _rec(js=0.05, bias=0.9)
    n_conf, n_inconc = _drive([(good, bad), (good, good), (good, good)])
    assert (n_conf, n_inconc) == (0, 0)


def test_default_arm_failing_alone_is_inconclusive_not_a_clear():
    """The asymmetry is deliberate and worth pinning down: a missing FLAG record counts against the
    flag, but a missing DEFAULT record makes the pair unusable -- there is nothing to compare the
    flag to.  Discarding those pairs silently would let a row whose baseline never runs (a broken
    default arm, an OOM, a bad target) report 'NOT CONFIRMED' on zero comparisons and clear."""
    good = _rec()
    n_conf, n_inconc = _drive([(None, good)] * 3)
    assert (n_conf, n_inconc) == (0, 1), (n_conf, n_inconc)


def test_a_single_unusable_pair_blocks_a_verdict_at_the_default_requirement():
    """Default min_valid is ALL seeds, so one dead default arm is enough to withhold the clear."""
    good = _rec()
    n_conf, n_inconc = _drive([(good, good), (good, good), (None, good)])
    assert (n_conf, n_inconc) == (0, 1), (n_conf, n_inconc)


def test_partial_min_valid_reaches_a_verdict_on_the_usable_pairs():
    """--confirm-min-valid below the seed count is the documented way to accept a lossy rerun: the
    pairs that DID run must then decide the row, in both directions."""
    good, bad = _rec(), _rec(js=0.05, bias=0.9)
    # 2 usable pairs, both worse -> the relaxed requirement is met and the row is CONFIRMED
    n_conf, n_inconc = _drive([(None, None), (good, bad), (good, bad)], min_valid=2)
    assert (n_conf, n_inconc) == (1, 0), (n_conf, n_inconc)
    # same relaxation, same 2 usable pairs, neither worse -> cleared as noise, not confirmed
    n_conf, n_inconc = _drive([(None, None), (good, good), (good, good)], min_valid=2)
    assert (n_conf, n_inconc) == (0, 0), (n_conf, n_inconc)


def test_partial_min_valid_still_withholds_a_verdict_below_its_own_floor():
    """Relaxing the requirement lowers the bar; it does not remove it."""
    good, bad = _rec(), _rec(js=0.05, bias=0.9)
    n_conf, n_inconc = _drive([(None, None), (None, None), (good, bad)], min_valid=2)
    assert (n_conf, n_inconc) == (0, 1), (n_conf, n_inconc)


# ---------------------------------------------------------------------------------------------
# Evidence requirements.  A confirmation step that can be configured to re-test nothing is worse
# than no confirmation step: it prints a clean verdict over an empty measurement.
# ---------------------------------------------------------------------------------------------

def _rejects(run_seed, repeats, explicit_seeds, min_valid):
    try:
        PB.confirm_plan(run_seed, repeats, explicit_seeds, min_valid)
        return False
    except ValueError:
        return True


def test_zero_repeats_is_rejected_rather_than_clearing_the_row():
    """--confirm-repeats 0 gave seeds=[], min_valid=0, 'NOT CONFIRMED (realization noise)' and
    exit 0: a flagged row cleared by a confirmation that ran nothing."""
    assert _rejects(987654, 0, None, None)
    assert _rejects(987654, -1, None, None)


def test_zero_min_valid_is_rejected():
    """Requiring no valid pair means the INCONCLUSIVE branch can never fire, so a row whose reruns
    all died reads as not-worse."""
    assert _rejects(987654, 3, None, 0)


def test_min_valid_above_the_seed_count_is_rejected():
    """Not dangerous (everything would be INCONCLUSIVE) but unsatisfiable; say so at parse time."""
    assert _rejects(987654, 3, None, 4)


def test_repeated_and_empty_explicit_seeds_are_rejected():
    """Fresh seeds are the whole mechanism: the flagging seed reproduced the same false FAIL four
    times, so re-testing one realization N times is not N pieces of evidence."""
    assert _rejects(987654, 3, "11,11,22", None)
    assert _rejects(987654, 3, "", None)
    assert _rejects(987654, 3, "11,987654", None)     # includes the seed that flagged the row


def test_accepted_plans_are_distinct_and_demand_every_pair_by_default():
    seeds, min_valid = PB.confirm_plan(987654, 3, None, None)
    assert len(set(seeds)) == 3 and 987654 not in seeds, seeds
    assert min_valid == len(seeds)
    seeds, min_valid = PB.confirm_plan(987654, 3, "11,22,33", 2)
    assert (seeds, min_valid) == ([11, 22, 33], 2)


def test_confirm_flagged_refuses_a_verdict_it_has_no_evidence_for():
    """The guard is re-checked inside confirm_flagged, so a caller bypassing the CLI cannot get a
    'NOT CONFIRMED' out of zero seeds."""
    for seeds, min_valid in [((), 0), ((), 1), ((1, 1, 2), 3), ((1, 2, 3), 0), ((1, 2, 3), 4)]:
        try:
            _drive([(_rec(), _rec())] * 3, seeds=seeds, min_valid=min_valid)
        except ValueError:
            continue
        assert False, "confirm_flagged accepted seeds={} min_valid={}".format(seeds, min_valid)


# ---------------------------------------------------------------------------------------------
# The real patching wiring.  The accounting tests above stub _run_one_cell, which is exactly where
# the arm-contamination bug hid: the arms were compared correctly on records produced by samplers
# that had the wrong flags on them.
# ---------------------------------------------------------------------------------------------

class _FakeSampler(object):
    """Stands in for a portfolio sampler; the probe only setattr()s flags onto it."""
    pass


class _gate(object):
    """Stub shape_recovery down to the seams the probe uses, INCLUDING the pristine factory.

    run_one() looks up SR.build_sampler exactly as the real one does, so `seen` records the flags
    the sampler was really built with -- not the flags the probe meant to install.
    """

    def __init__(self, seen, run_one=None):
        self.seen = seen
        self.custom_run_one = run_one

    def _build(self, kind, target, n_chunk):
        return _FakeSampler()

    def _run_one(self, kind, target, nmax, neff, seed=None):
        s = SR.build_sampler(kind, target, 1000)
        flags = dict(vars(s))
        self.seen.append(flags)
        # Carry the flags and seed INTO the record, so a stubbed evaluate() can rule on the arm
        # that actually ran.  That is what lets the main() tests below be end-to-end (a verdict on
        # a real sampler build) rather than a staged sequence of canned verdicts.
        return dict(_rec(), flags=flags, seed=seed)

    def __enter__(self):
        self._saved = (SR.build_sampler, SR.run_one, SR.evaluate, PB._ORIG_BUILD_SAMPLER)
        SR.build_sampler = PB._ORIG_BUILD_SAMPLER = self._build
        SR.run_one = self.custom_run_one or self._run_one
        SR.evaluate = lambda rec: "PASS"
        return self

    def __exit__(self, *exc):
        SR.build_sampler, SR.run_one, SR.evaluate, PB._ORIG_BUILD_SAMPLER = self._saved
        return False


def test_off_arm_is_not_contaminated_by_a_previous_on_arm():
    """THE bug: each config re-wrapped whatever build_sampler happened to be installed, so the
    'flags OFF' baseline ran through the previous arm's wrapper -- comparing a flag with itself."""
    seen = []
    with _gate(seen):
        PB.run_config("adaptive_alloc ON", {"portfolio_adaptive_alloc": True},
                      [(2, 1, 303)], 1000, 100, 5)
        PB.run_config("weight_clip ON", {"portfolio_weight_clip": 1.0},
                      [(2, 1, 303)], 1000, 100, 5)
        PB.run_config("flags OFF (default)", {}, [(2, 1, 303)], 1000, 100, 5)
    assert seen[0] == {"portfolio_adaptive_alloc": True}, seen[0]
    assert seen[1] == {"portfolio_weight_clip": 1.0}, seen[1]      # no adaptive_alloc carried over
    assert seen[2] == {}, "the OFF arm ran with an earlier arm's flags: {}".format(seen[2])


def test_confirmation_arms_do_not_share_wrappers_either():
    """confirm_flagged alternates default/flag arms many times; that is where wrappers piled up."""
    seen = []
    with _gate(seen):
        for _ in range(3):
            PB._run_one_cell({"portfolio_adaptive_alloc": True}, (2, 1, 303), 1000, 100, 7)
            PB._run_one_cell({}, (2, 1, 303), 1000, 100, 7)
    assert seen == [{"portfolio_adaptive_alloc": True}, {}] * 3, seen


def test_patched_build_wraps_the_pristine_factory_not_the_live_global():
    calls = []

    def pristine(kind, target, n_chunk):
        calls.append("pristine")
        return _FakeSampler()

    def leftover(kind, target, n_chunk):     # what a leaked patch would leave installed
        calls.append("leftover")
        return _FakeSampler()

    with _gate([]):
        SR.build_sampler = PB._ORIG_BUILD_SAMPLER = pristine
        SR.build_sampler = leftover
        s = PB.patched_build({"portfolio_weight_clip": 1.0})("portfolio", None, 10)
    assert calls == ["pristine"], calls
    assert vars(s) == {"portfolio_weight_clip": 1.0}


def test_flag_patch_restores_the_original_even_when_the_run_raises():
    orig = SR.build_sampler
    try:
        with PB.flag_patch({"portfolio_adaptive_alloc": True}):
            assert SR.build_sampler is not orig
            raise ValueError("sampler blew up mid-run")
    except ValueError:
        pass
    assert SR.build_sampler is orig, "a failed run left its patch installed for the next arm"


def test_nested_flag_patch_is_refused():
    with PB.flag_patch({"portfolio_adaptive_alloc": True}):
        try:
            with PB.flag_patch({"portfolio_weight_clip": 1.0}):
                raise AssertionError("nesting allowed: wrappers would stack")
        except RuntimeError:
            pass
    assert SR.build_sampler is PB._ORIG_BUILD_SAMPLER


def test_a_cell_that_raises_reports_no_record_and_leaves_no_patch_behind():
    def boom(kind, target, nmax, neff, seed=None):
        raise RuntimeError("integrate failed")

    with _gate([], run_one=boom):
        assert PB._run_one_cell({"portfolio_adaptive_alloc": True},
                                (2, 1, 303), 1000, 100, 7) is None
        assert SR.build_sampler is PB._ORIG_BUILD_SAMPLER


# ---------------------------------------------------------------------------------------------
# The real entry path.  Everything above tests a function the CLI happens to call; these drive
# main() itself, because that is what a merge gate runs and what its exit code comes from.  One
# target, seven configs, the gate seams stubbed -- seconds, no sampler built.
# ---------------------------------------------------------------------------------------------

_ONE_TARGET = ["--dims", "2", "--ncomps", "1", "--seeds", "303",
               "--nmax-per-dim", "100", "--neff", "10"]


_SYNTH_CONFIGS = [
    ("flags OFF (default)", {}),
    ("probe_flag A", {"portfolio_probe_flag_a": True}),
    ("probe_flag B", {"portfolio_probe_flag_b": 2.0}),
]


def _main(argv, evaluate, seen=None, configs=None):
    """Run PB.main() with the gate stubbed; returns (exit_code, flags-per-run)."""
    seen = [] if seen is None else seen
    saved_argv = sys.argv
    sys.argv = ["probe_portfolio_optin_flags.py"] + list(argv)
    try:
        with _gate(seen):
            SR.evaluate = evaluate
            # Inject a synthetic config list: these tests exercise the CONFIRMATION
            # MACHINERY, and must not break when the shipped flag set changes (as it
            # did when adaptive_alloc was excluded -- see FOLLOWUPS.md item 4).
            saved_cfg = PB.FLAG_CONFIGS
            PB.FLAG_CONFIGS = list(configs) if configs is not None else saved_cfg
            try:
                return PB.main(), seen
            finally:
                PB.FLAG_CONFIGS = saved_cfg
    finally:
        sys.argv = saved_argv


def test_main_passes_a_clean_run_and_gives_each_arm_only_its_own_flags():
    """End-to-end on the entry path: exit 0, and the arms are independent where it counts -- the
    baseline is built with NO flags and no arm inherits its predecessor's."""
    code, seen = _main(_ONE_TARGET, evaluate=lambda rec: "PASS", configs=_SYNTH_CONFIGS)
    assert code == 0, code
    assert len(seen) == len(_SYNTH_CONFIGS), seen                 # one run per configured arm
    assert seen[0] == {}, "the flags-OFF baseline was not built clean: {}".format(seen[0])
    assert seen[1] == {"portfolio_probe_flag_a": True}, seen[1]
    assert seen[2] == {"portfolio_probe_flag_b": 2.0}, seen[2]    # arm A's flag not carried over
    assert "portfolio_probe_flag_a" not in seen[2], seen[2]


def test_main_clears_a_row_that_only_fails_at_the_flagging_seed():
    """The row this machinery was built for: FAIL at the original run seed, fine at fresh ones.
    main() must re-test it and exit 0 -- and must do that at seeds it has not already used."""
    seeds_ruled_on = []

    def evaluate(rec):
        seeds_ruled_on.append(rec["seed"])
        if rec["flags"] == {"portfolio_probe_flag_a": True} and rec["seed"] == 987654:
            return "FAIL"
        return "PASS"

    code, seen = _main(_ONE_TARGET + ["--confirm-repeats", "2"], evaluate=evaluate,
                       configs=_SYNTH_CONFIGS)
    assert code == 0, "a row that only fails at its own seed still failed the run"
    confirmation = seeds_ruled_on[len(_SYNTH_CONFIGS):]           # summary runs, then the reruns
    assert 987654 not in confirmation, "re-tested at the seed that flagged it: {}".format(confirmation)
    assert len(set(confirmation)) == 2, confirmation              # two DISTINCT fresh seeds
    assert len(confirmation) == 4, confirmation                   # both arms at each of them


def test_main_still_fails_a_row_that_fails_at_every_fresh_seed():
    """The other direction, on the same path: confirmation must not become a blanket amnesty."""
    def evaluate(rec):
        return "FAIL" if rec["flags"] == {"portfolio_probe_flag_a": True} else "PASS"

    code, _ = _main(_ONE_TARGET + ["--confirm-repeats", "2"], evaluate=evaluate,
                    configs=_SYNTH_CONFIGS)
    assert code == 1, "a reproducible opt-in regression was cleared by the confirmation step"


def test_main_fails_a_row_immediately_under_no_confirm():
    def evaluate(rec):
        return "FAIL" if rec["flags"] == {"portfolio_probe_flag_a": True} else "PASS"

    code, seen = _main(_ONE_TARGET + ["--no-confirm"], evaluate=evaluate,
                       configs=_SYNTH_CONFIGS)
    assert code == 1, code
    assert len(seen) == len(_SYNTH_CONFIGS), \
        "--no-confirm ran reruns anyway: {} runs for {} arms".format(len(seen), len(_SYNTH_CONFIGS))


def test_main_exits_2_on_an_unusable_confirmation_setting_before_running_anything():
    """Rejected at parse time, so an unusable setting costs a usage error rather than an hour of
    sampling followed by a verdict backed by nothing."""
    for bad_opt in (["--confirm-repeats", "0"],
                    ["--confirm-min-valid", "0"],
                    ["--confirm-repeats", "2", "--confirm-min-valid", "3"],
                    ["--confirm-seeds", "11,11,22"],
                    ["--confirm-seeds", "11,987654"]):
        seen = []
        try:
            _main(_ONE_TARGET + bad_opt, evaluate=lambda rec: "PASS", seen=seen)
        except SystemExit as e:
            assert e.code == 2, (bad_opt, e.code)
        else:
            assert False, "accepted {}".format(bad_opt)
        assert seen == [], "{} ran samplers before being rejected".format(bad_opt)




def test_adaptive_alloc_is_excluded_from_the_probe_configs():
    """`--portfolio-adaptive-alloc` is a CONFIRMED regression (FOLLOWUPS.md item 4) and is excluded
    until fixed.  Pinned so the exclusion cannot be undone silently: reinstating those rows is the
    first step of fixing the flag, and this test failing is the reminder that they will fail."""
    import inspect
    src = inspect.getsource(PB.main)
    active = [l for l in src.splitlines()
              if "portfolio_adaptive_alloc" in l and not l.strip().startswith("#")]
    assert not active, "adaptive_alloc re-enabled in the probe configs: {}".format(active)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
    print("probe confirm-on-fail accounting holds")
