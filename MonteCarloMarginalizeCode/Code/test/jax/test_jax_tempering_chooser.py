"""The tempering-cost law and the driver's --adapt-weight-exponent chooser.

WHAT THESE PIN, AND WHY THE SHAPE
---------------------------------
`--adapt-weight-exponent beta` does two structurally different things in RIFT:
in the non-JAX samplers it shapes only the adaptive sampling PRIOR (unbiased at
any beta, no cost in exported samples), while on the JAX flowMC path it is the
exponent of the target the MCMC SAMPLES, so the export must be reweighted by
L^(1-beta) and that reweight has an ESS cost.  The cost obeys

    ESS/N = [beta (2 - beta)]^(n_dim/2)

which is set by the SAMPLED DIMENSION and NOT by lnLmax -- i.e. not by SNR,
which is what the historical non-JAX helper keys on.  Provenance, the measured
sweep and the arm study: RIFT/likelihood/jax_ile/DESIGN_jax_tempering.md.

Several tests below are AST guards on the DRIVER SOURCE rather than assertions
on a helper.  That is deliberate and matches test_jax_fairdraw_export.py: the
defects these pin live at CALL SITES (a chooser that is computed and then not
used; a guard that is computed and then not raised), where a helper-level
assertion cannot see them.  Needs no lal, no GPU and no flowMC.
"""
import ast
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
DRIVER = os.path.join(CODE, "bin", "integrate_likelihood_extrinsic_jax")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from RIFT.likelihood.jax_ile.samplers import (          # noqa: E402
    beta_for_export_ess, export_ess_fraction)


def _driver_tree():
    with open(DRIVER) as f:
        return ast.parse(f.read())


def _load_driver():
    """Import the driver script (no .py suffix) as a module, as
    test_jax_fairdraw_export.py does, so the chooser can be CALLED rather than
    only read.  Source-level guards cannot see whether a branch is reachable --
    a mutation that made the beta>1 branch dead (`if beta >= 1.0 and False:`)
    survived an AST-only version of these tests."""
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader("_ile_jax_driver_temper", DRIVER)
    spec = importlib.util.spec_from_loader("_ile_jax_driver_temper", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


drv = _load_driver()


class _Opts(object):
    """Minimal stand-in for the optparse Values the chooser reads."""
    def __init__(self, **kw):
        self.adapt_adapt = False
        self.auto_adapt_weight_exponent = False
        self.adapt_weight_exponent = 1.0
        self.target_export_ess_frac = drv._TARGET_EXPORT_ESS_FRAC_DEFAULT
        self.allow_degenerate_tempering = False
        self.__dict__.update(kw)


# ----------------------------------------------------------------- the law
def test_law_matches_the_measured_sweep():
    """The closed form against the EXACT sweep on the real BNS likelihood.

    Measured by beta_ess_offline.py on the SNR-23.8 zero-noise BNS (4-D
    phi-marginalised), reference ESS(g=1)=5235 with every g in [0,2] resolved
    (worst-case IS ESS 508).  The law is Gaussian-peak and therefore slightly
    OPTIMISTIC: measured/law runs 0.79 (beta=0.05) to 1.00 (beta=1).  Pinning
    the band both ways is what makes this a test rather than a restatement --
    an over-optimistic law would silently under-budget a real run.
    """
    measured = {                     # beta: measured ESS/N
        0.05: 7.546e-03, 0.10: 3.033e-02, 0.20: 1.042e-01, 0.30: 2.123e-01,
        0.40: 3.481e-01, 0.50: 4.999e-01, 0.60: 6.530e-01, 0.70: 7.923e-01,
        0.80: 9.035e-01, 0.90: 9.752e-01, 1.00: 1.0,
    }
    ratios = []
    for beta, meas in measured.items():
        law = export_ess_fraction(beta, 4)
        ratios.append(meas / law)
    assert min(ratios) > 0.75, "law is far more optimistic than measured: %r" % (ratios,)
    assert max(ratios) <= 1.001, "law UNDER-predicts the measured cost: %r" % (ratios,)


def test_law_is_independent_of_lnLmax_by_construction():
    """The signature carries no lnLmax/SNR argument at all.

    This is the property the whole design rests on, so pin it structurally: if
    someone later adds an SNR term (re-importing the non-JAX helper's rule), the
    signature changes and this fails.
    """
    import inspect
    params = list(inspect.signature(export_ess_fraction).parameters)
    assert params == ["beta", "n_dim"], params
    for bad in ("snr", "lnLmax", "lnl_max", "guess_snr"):
        assert bad not in params


def test_law_depends_on_dimension():
    """A knob that gives the same answer for every dimension is a dead knob."""
    vals = [export_ess_fraction(0.5, d) for d in (3, 4, 5)]
    assert len(set(vals)) == 3
    assert vals[0] > vals[1] > vals[2]        # more dimensions -> costlier


def test_roundtrip_beta_and_target():
    for n_dim in (3, 4, 5):
        for target in (0.3, 0.5, 0.9, 0.99):
            beta = beta_for_export_ess(target, n_dim)
            assert 0.0 < beta <= 1.0
            assert export_ess_fraction(beta, n_dim) == pytest.approx(target, rel=1e-10)


def test_beta_one_is_free_and_is_the_only_free_point():
    for n_dim in (3, 4, 5):
        assert export_ess_fraction(1.0, n_dim) == pytest.approx(1.0)
        assert export_ess_fraction(0.999, n_dim) < 1.0


def test_domain_errors_raise_rather_than_clamp():
    """A silently clamped exponent is the failure this whole change exists to stop."""
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            export_ess_fraction(bad, 4)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            beta_for_export_ess(bad, 4)


def test_historical_helper_beta_would_be_degenerate_here():
    """The number this change exists to prevent someone from porting.

    helper_LDG_Events.py picks beta = 0.1*(22.5/SNR)^2 above SNR 22.5; at the
    SNR-23.8 study event that is 0.0951.  On the 4-D JAX path that leaves ~3% of
    the cloud, i.e. ESS ~140 of 4800 -- below the driver's own "NOT a usable
    posterior sample" warning threshold of 200.
    """
    beta_hist = 0.1 / (23.78 / 22.5) ** 2
    frac = export_ess_fraction(beta_hist, 4)
    assert frac < 0.05
    assert frac * 4800 < 200


# -------------------------------------------------------- driver call sites
def _declared_options():
    """Flag strings passed as the FIRST argument of an add_option call.

    NOT "every string constant beginning with --" anywhere in the file.  That
    weaker version passed while the parser flag was renamed to
    `--auto-adapt-weight-exponent-XX`, because the correct spelling still occurred
    inside two error messages and the inert-flag list -- and a MUTANT was
    committed behind it.
    """
    tree = _driver_tree()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "add_option" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out.add(a.value)
    return out


def test_driver_exposes_the_auto_chooser_options():
    declared = _declared_options()
    for flag in ("--auto-adapt-weight-exponent", "--target-export-ess-frac",
                 "--allow-degenerate-tempering"):
        assert flag in declared, (
            "%s is not DECLARED via add_option (mentioning it in a help or error "
            "string does not make it a flag)" % flag)


def test_no_stray_placeholder_flags():
    """Mutation-harness residue guard.

    A mutation sweep renames flags in place; if one is left applied when the tree
    is staged, it ships.  That happened once (`--auto-adapt-weight-exponent-XX`
    reached a commit).  Cheap, permanent check.
    """
    for flag in _declared_options():
        assert not flag.endswith("-XX"), "placeholder flag left in the parser: %s" % flag


def test_chooser_result_is_actually_assigned_to_the_exponent():
    """A chooser that is computed and then not used is the classic dead knob.

    Pin that the driver's tempering helper ASSIGNS to opts.adapt_weight_exponent,
    not merely that it calls beta_for_export_ess somewhere.
    """
    tree = _driver_tree()
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "resolve_tempering_exponent"), None)
    assert fn is not None, "resolve_tempering_exponent is missing from the driver"
    calls = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "beta_for_export_ess" in calls
    targets = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Attribute):
                    targets.add(t.attr)
    assert "adapt_weight_exponent" in targets, (
        "resolve_tempering_exponent never writes opts.adapt_weight_exponent")


def test_auto_refuses_to_silently_override_an_explicit_exponent():
    """--auto plus an explicit --adapt-weight-exponent must RAISE, not overwrite.

    Pinned on the source because reaching this branch at runtime needs a full
    likelihood build.  Both conflicts (this one and --auto + --adapt-adapt) must
    be present: each was added after noticing the other silently won.
    """
    tree = _driver_tree()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "resolve_tempering_exponent")
    msgs = [c.value for n in ast.walk(fn) if isinstance(n, ast.Raise)
            for c in ast.walk(n)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    joined = " ".join(msgs)
    assert "would overwrite it" in joined, (
        "no guard against --auto silently overriding an explicit exponent")
    assert "--adapt-adapt" in joined, (
        "no guard against --auto being combined with --adapt-adapt")


@pytest.mark.parametrize("bad", [1.5, 2.0, 0.0, -0.2])
def test_beta_outside_the_unit_interval_is_REFUSED_at_runtime(bad, capsys):
    """CALL the chooser.  beta>1 sharpens the target past the posterior
    (samplers.flowmc_sample* take temper = 1/beta, so beta>1 gives inv_T>1) and
    its export reweight has ESS/N = [beta(2-beta)]^(dim/2): 0 at beta=2,
    undefined beyond.  beta<=0 samples the prior.

    An earlier source-only version of this test passed while a mutation made the
    branch dead, and an earlier version of the CODE tested `beta >= 1.0` and
    printed "beta=1 (untempered target)" for every beta>1.  Both are why this
    exercises the function instead of reading it.
    """
    with pytest.raises(SystemExit) as e:
        drv.resolve_tempering_exponent(_Opts(adapt_weight_exponent=bad), 4, 4800)
    assert "untempered" not in str(e.value)
    out = capsys.readouterr().out
    assert "untempered target" not in out, (
        "beta=%r was reported as untempered" % bad)


def test_beta_one_and_a_healthy_beta_are_accepted_at_runtime(capsys):
    """The negative tests above prove nothing if every input raises."""
    drv.resolve_tempering_exponent(_Opts(adapt_weight_exponent=1.0), 4, 4800)
    assert "untempered target" in capsys.readouterr().out
    o = _Opts(adapt_weight_exponent=0.7735)
    drv.resolve_tempering_exponent(o, 4, 4800)
    out = capsys.readouterr().out
    assert "predicted export ESS/N=0.9" in out, out


def test_auto_sets_the_exponent_and_the_value_depends_on_dimension(capsys):
    """The chooser must MOVE the number, and move it differently per dimension.
    Identical output across settings is the signature of a dead knob."""
    got = {}
    for n_dim in (3, 4, 5):
        o = _Opts(auto_adapt_weight_exponent=True)
        drv.resolve_tempering_exponent(o, n_dim, 4800)
        capsys.readouterr()
        assert o.adapt_weight_exponent != 1.0, "auto left the exponent at its default"
        got[n_dim] = o.adapt_weight_exponent
    assert len(set(got.values())) == 3, got
    assert got[3] < got[4] < got[5]


def test_degenerate_exponent_is_refused_and_the_override_lets_it_through(capsys):
    """Both directions.  A guard that never passes anything is not a guard."""
    with pytest.raises(SystemExit) as e:
        drv.resolve_tempering_exponent(_Opts(adapt_weight_exponent=0.09508), 4, 4800)
    assert "would not be a usable posterior sample" in str(e.value)
    capsys.readouterr()
    drv.resolve_tempering_exponent(
        _Opts(adapt_weight_exponent=0.09508, allow_degenerate_tempering=True), 4, 4800)
    assert "predicted export ESS" in capsys.readouterr().out


def test_target_without_auto_is_reported_not_silently_ignored(capsys):
    """Setting a budget and no chooser does nothing; say so.

    Nothing covered this and a mutation deleting the note survived the sweep.
    """
    o = _Opts(target_export_ess_frac=0.5)
    drv.resolve_tempering_exponent(o, 4, 4800)
    out = capsys.readouterr().out
    assert "--target-export-ess-frac" in out and "no effect" in out, out
    # and it must NOT be reported when the chooser is actually on
    o2 = _Opts(auto_adapt_weight_exponent=True, target_export_ess_frac=0.5)
    drv.resolve_tempering_exponent(o2, 4, 4800)
    assert "no effect" not in capsys.readouterr().out


def test_inert_note_lists_only_flags_the_user_ACTUALLY_passed(capsys):
    """On a non-tempered mode the chooser flags are reported as inert -- but the
    report must not name a flag the user never typed.

    It previously appended --target-export-ess-frac whenever --auto was set, even
    with the target at its default, telling the user a flag they had not passed
    was being ignored.
    """
    p = drv.build_parser()

    def mk(**kw):
        class O(object):
            pass
        o = O()
        for opt in p._get_all_options():
            if opt.dest:
                setattr(o, opt.dest, opt.default)
        o.mode = "laplace-is"
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def note(o):
        drv.check_critical_and_report(o, p)
        return "".join(l for l in capsys.readouterr().out.splitlines()
                       if "tempered modes" in l)

    default_target = note(mk(auto_adapt_weight_exponent=True))
    assert "--auto-adapt-weight-exponent" in default_target
    assert "--target-export-ess-frac" not in default_target, default_target

    given_target = note(mk(auto_adapt_weight_exponent=True,
                           target_export_ess_frac=0.5))
    assert "--target-export-ess-frac" in given_target, given_target

    # and on a tempered mode nothing is reported inert at all
    o = mk(auto_adapt_weight_exponent=True)
    o.mode = "flowmc-phimarg"
    assert note(o) == ""


def test_auto_conflicts_raise_at_runtime():
    with pytest.raises(SystemExit) as e1:
        drv.resolve_tempering_exponent(
            _Opts(auto_adapt_weight_exponent=True, adapt_weight_exponent=0.5), 4, 4800)
    assert "would overwrite it" in str(e1.value)
    with pytest.raises(SystemExit) as e2:
        drv.resolve_tempering_exponent(
            _Opts(auto_adapt_weight_exponent=True, adapt_adapt=True), 4, 4800)
    assert "--adapt-adapt" in str(e2.value)


def test_law_refuses_the_same_domain_the_driver_does():
    """One domain, two enforcers -- they must not disagree."""
    for bad in (1.5, 0.0, -0.2):
        with pytest.raises(ValueError):
            export_ess_fraction(bad, 4)


def test_target_frac_default_is_one_named_constant():
    """The parser default and the was-it-passed check must be the same value.

    Two literals would drift, and the drift is silent: --target-export-ess-frac
    set to the old default would stop being reported as inert.
    """
    src = open(DRIVER).read()
    assert "_TARGET_EXPORT_ESS_FRAC_DEFAULT = 0.9" in src
    assert "default=_TARGET_EXPORT_ESS_FRAC_DEFAULT" in src, (
        "the parser hardcodes its own default instead of the named constant")
    tree = _driver_tree()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_target_ess_was_given")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "_TARGET_EXPORT_ESS_FRAC_DEFAULT" in names


def test_chooser_is_actually_CALLED_from_the_dispatch():
    """The wiring, not the helper.

    Every assertion above passes with `resolve_tempering_exponent` defined and
    never invoked -- a perfectly dead chooser.  Pin the CALL SITE: it must appear
    inside analyze_one, guarded by the tempered-mode set, and be handed the
    sampled dimension.  (test_jax_fairdraw_export.py pins its own call sites the
    same way, for the same reason.)
    """
    tree = _driver_tree()
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "analyze_one"), None)
    assert fn is not None, "analyze_one is missing from the driver"
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "resolve_tempering_exponent"]
    assert calls, "analyze_one never calls resolve_tempering_exponent"
    # gated on the tempered-mode set, not on a hand-written mode list
    src = ast.get_source_segment(open(DRIVER).read(), fn) or ""
    assert "_TEMPERED_MODES" in src
    # and told what dimension it is choosing for
    assert any(isinstance(a, ast.Name) and a.id == "dim" for c in calls for a in c.args), \
        "the chooser is called without the sampled dimension"


def test_degenerate_tempering_guard_raises_rather_than_warns():
    """The guard must RAISE.  A printed warning above a 199-row export is exactly
    the silent-degradation mode this change exists to remove.

    Anchored to the ESS BRANCH specifically, not to "the function contains a
    raise": the first version of this test asserted the latter and SURVIVED a
    mutation that turned the guard's raise into a print, because the unrelated
    --auto/--adapt-adapt conflict raise satisfied it.
    """
    tree = _driver_tree()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "resolve_tempering_exponent")
    guard = None
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and any(
                isinstance(x, ast.Name) and x.id == "_USABLE_EXPORT_ESS"
                for x in ast.walk(n.test)):
            guard = n
            break
    assert guard is not None, (
        "no `if ... _USABLE_EXPORT_ESS ...` branch in resolve_tempering_exponent")
    assert any(isinstance(x, ast.Raise) for x in ast.walk(guard)), (
        "the degenerate-export branch does not raise -- a printed warning above a "
        "near-degenerate export is exactly what this guard exists to prevent")


def test_guard_threshold_matches_the_message_the_driver_already_prints():
    """One definition, so the guard and fairdraw_indices' warning cannot drift.

    fairdraw_indices already tells the user at ESS < 200 that the cloud "is NOT a
    usable posterior sample".  If the guard used its own literal, the two could
    disagree and the driver would refuse at one number while warning at another.
    """
    import re
    src = open(DRIVER).read()
    assert re.search(r"^_USABLE_EXPORT_ESS = 200$", src, re.M), \
        "_USABLE_EXPORT_ESS is not defined as a single module-level constant"
    # the pre-existing warning must be expressed through the same constant
    assert "if neff < _USABLE_EXPORT_ESS:" in src, (
        "fairdraw_indices still hardcodes its own threshold; route it through "
        "_USABLE_EXPORT_ESS so the guard and the warning cannot drift apart")


def test_chooser_runs_and_changes_the_exponent_end_to_end():
    """Run the DRIVER, not the helper: --help must render, and the chooser must
    move the number.  A subprocess is worth its seconds -- a parser-level typo is
    invisible to every assertion above."""
    env = dict(os.environ, PYTHONPATH=CODE + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, DRIVER, "--help"], capture_output=True,
                       text=True, env=env, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    # Exact tokens, not substrings: "--auto-adapt-weight-exponent" is a substring
    # of "--auto-adapt-weight-exponent-XX", so the substring form passed against a
    # renamed flag.  Match the whole option token.
    import re
    for flag in ("--auto-adapt-weight-exponent", "--target-export-ess-frac",
                 "--allow-degenerate-tempering"):
        assert re.search(re.escape(flag) + r"(?![-\w])", p.stdout), (
            "%s does not appear as a whole option in --help" % flag)


def test_help_does_not_recommend_the_snr_keyed_rule():
    """RETIRED-claim guard.  Positive assertions cannot catch a superseded claim
    left standing beside the new one, so assert the ABSENCE of the rule this
    change exists to keep off the JAX path.  Append to RETIRED as it changes."""
    with open(DRIVER) as f:
        src = f.read()
    RETIRED = ("snr_fac", "0.1/np.power", "adapt-weight-exponent from the SNR")
    for phrase in RETIRED:
        assert phrase not in src, "retired SNR-keyed rule reappeared: %r" % phrase
