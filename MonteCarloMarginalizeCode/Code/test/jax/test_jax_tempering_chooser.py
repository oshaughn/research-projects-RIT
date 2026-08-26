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
    beta_for_export_ess, export_ess_fraction, export_ess_lower_bound)

# The measured sweep (dim=4) from DESIGN_jax_tempering.md.  Kept as data because
# more than one test needs it, and because the calibration envelope must be
# checked against ALL of it, not against a couple of convenient points.
MEASURED_D4 = {
    0.05: 7.546e-03, 0.09508: 2.763e-02, 0.10: 3.033e-02, 0.20: 1.042e-01,
    0.30: 2.123e-01, 0.40: 3.481e-01, 0.50: 4.999e-01, 0.55: 5.772e-01,
    0.60: 6.530e-01, 0.65: 7.253e-01, 0.70: 7.923e-01, 0.75: 8.522e-01,
    0.80: 9.035e-01, 0.85: 9.449e-01, 0.90: 9.752e-01, 0.95: 9.938e-01,
    1.00: 1.0,
}


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
    """Minimal stand-in for the optparse Values the chooser reads.

    ``supplied`` lists the option TOKENS to pretend were typed.  Setting an
    attribute is NOT the same as passing the flag any more -- that conflation is
    the defect this stub previously baked in -- so any test whose behaviour turns
    on "did the user pass it" must name the token here (or build from argv).
    """
    def __init__(self, supplied=(), **kw):
        self._supplied_options = set(supplied)
        self.adapt_adapt = False
        self.auto_adapt_weight_exponent = False
        self.adapt_weight_exponent = 1.0
        self.target_export_ess_frac = drv._TARGET_EXPORT_ESS_FRAC_DEFAULT
        self.allow_degenerate_tempering = False
        self.smc_puffball = False
        self.__dict__.update(kw)
        self._supplied_options = set(supplied)


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
    measured = MEASURED_D4
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


def test_chosen_beta_meets_the_target_on_the_MEASURED_lower_bound():
    """The chooser must satisfy the target against the CALIBRATED envelope.

    THE DEFECT (review): it used to invert the bare Gaussian law, which the sweep
    shows is optimistic by up to 21%.  Round-tripping that formula is vacuous --
    it only proves the inverse inverts the thing that is known to overstate the
    answer.  At dim=4 with a 0.9 target it returned beta=0.77347, whose measured
    lower bound is 0.866: less than was asked for.
    """
    for n_dim in (3, 4, 5):
        for target in (0.3, 0.5, 0.9, 0.99):
            beta = beta_for_export_ess(target, n_dim)
            assert 0.0 < beta <= 1.0
            assert export_ess_lower_bound(beta, n_dim) >= target - 1e-9, (
                "dim=%d target=%g -> beta=%g retains only %.4f"
                % (n_dim, target, beta, export_ess_lower_bound(beta, n_dim)))
            # and it must be the SMALLEST such beta, to within the solver step
            if beta > 1e-3:
                assert export_ess_lower_bound(beta * 0.99, n_dim) < target + 1e-9


def test_the_old_uncalibrated_answer_would_NOT_pass():
    """Pins that the fix changed the number, not just the wording."""
    assert export_ess_lower_bound(0.77347, 4) < 0.9
    assert beta_for_export_ess(0.9, 4) > 0.77347


def test_calibration_envelope_is_a_true_lower_bound_everywhere_measured():
    """Every measured point of the sweep must sit at or above the envelope.

    This is the property the guard and the chooser both rely on; if a future
    knot edit breaks it anywhere, the conservatism is gone silently.
    """
    for beta, measured in MEASURED_D4.items():
        lb = export_ess_lower_bound(beta, 4)
        assert measured >= lb - 1e-12, (
            "beta=%g: measured %.4e is BELOW the supposed lower bound %.4e"
            % (beta, measured, lb))


def test_lower_bound_is_never_above_the_law_and_meets_it_at_beta_one():
    for n_dim in (3, 4, 5):
        for beta in (0.05, 0.2, 0.5, 0.8, 0.95):
            assert export_ess_lower_bound(beta, n_dim) <= export_ess_fraction(beta, n_dim)
        assert export_ess_lower_bound(1.0, n_dim) == pytest.approx(
            export_ess_fraction(1.0, n_dim))


def test_unreachable_target_raises_rather_than_returning_beta_one():
    """A target the envelope cannot reach must fail loudly, not silently clamp."""
    import RIFT.likelihood.jax_ile.samplers as S
    real = S._ESS_CAL_RATIO
    try:
        S._ESS_CAL_RATIO = tuple(0.5 * r for r in real)   # envelope caps at 0.5
        with pytest.raises(ValueError, match="unreachable"):
            beta_for_export_ess(0.9, 4)
    finally:
        S._ESS_CAL_RATIO = real


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


def test_chooser_result_is_actually_RETURNED_and_consumed():
    """A chooser that is computed and then not used is the classic dead knob.

    RETARGETED: this used to require that the chooser ASSIGN
    opts.adapt_weight_exponent.  That contract was the bug -- opts is per-RUN and
    analyze_one is per-EVENT, so the write made event 1 of a batch read event 0's
    choice.  The chooser now RETURNS the exponent; see
    test_chooser_returns_the_exponent_rather_than_writing_it_back for the other
    half, and test_chooser_is_reusable_across_a_multi_event_BATCH for the
    behaviour this protects.
    """
    tree = _driver_tree()
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "resolve_tempering_exponent"), None)
    assert fn is not None, "resolve_tempering_exponent is missing from the driver"
    calls = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "beta_for_export_ess" in calls
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, "resolve_tempering_exponent returns nothing"


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
    # Derive the exponent from the chooser rather than hardcoding one.  The
    # literal 0.7735 that used to be here was the OLD uncalibrated answer, whose
    # measured lower bound is 0.866 -- so this assertion was quietly encoding the
    # very over-claim the calibration exists to remove.
    beta = beta_for_export_ess(0.9, 4)
    o = _Opts(adapt_weight_exponent=beta)
    drv.resolve_tempering_exponent(o, 4, 4800)
    out = capsys.readouterr().out
    assert "export ESS/N >= 0.9" in out, out


def test_auto_sets_the_exponent_and_the_value_depends_on_dimension(capsys):
    """The chooser must MOVE the number, and move it differently per dimension.
    Identical output across settings is the signature of a dead knob."""
    got = {}
    for n_dim in (3, 4, 5):
        o = _Opts(auto_adapt_weight_exponent=True)
        beta = drv.resolve_tempering_exponent(o, n_dim, 4800)
        capsys.readouterr()
        assert beta != 1.0, "auto returned the default exponent"
        assert o.adapt_weight_exponent == 1.0, "auto mutated opts (see the batch test)"
        got[n_dim] = beta
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
    assert "export ESS/N >=" in capsys.readouterr().out


def test_target_without_auto_is_reported_not_silently_ignored(capsys):
    """Setting a budget and no chooser does nothing; say so.

    Nothing covered this and a mutation deleting the note survived the sweep.
    """
    o = _Opts(target_export_ess_frac=0.5, supplied=["--target-export-ess-frac"])
    drv.resolve_tempering_exponent(o, 4, 4800)
    out = capsys.readouterr().out
    assert "--target-export-ess-frac" in out and "no effect" in out, out
    # and it must NOT be reported when the chooser is actually on
    o2 = _Opts(auto_adapt_weight_exponent=True, target_export_ess_frac=0.5,
               supplied=["--auto-adapt-weight-exponent", "--target-export-ess-frac"])
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

    def mk(supplied=(), **kw):
        class O(object):
            pass
        o = O()
        for opt in p._get_all_options():
            if opt.dest:
                setattr(o, opt.dest, opt.default)
        o.mode = "laplace-is"
        for k, v in kw.items():
            setattr(o, k, v)
        # Setting the attribute is NOT passing the flag: the report keys on the
        # command-line token, so a test that wants a flag reported must name it.
        o._supplied_options = set(supplied)
        return o

    def note(o):
        drv.check_critical_and_report(o, p)
        return "".join(l for l in capsys.readouterr().out.splitlines()
                       if "tempered modes" in l)

    default_target = note(mk(auto_adapt_weight_exponent=True,
                             supplied=["--auto-adapt-weight-exponent"]))
    assert "--auto-adapt-weight-exponent" in default_target
    assert "--target-export-ess-frac" not in default_target, default_target

    given_target = note(mk(auto_adapt_weight_exponent=True,
                           target_export_ess_frac=0.5,
                           supplied=["--auto-adapt-weight-exponent",
                                     "--target-export-ess-frac"]))
    assert "--target-export-ess-frac" in given_target, given_target

    # and on a tempered mode nothing is reported inert at all
    o = mk(auto_adapt_weight_exponent=True,
           supplied=["--auto-adapt-weight-exponent"])
    o.mode = "flowmc-phimarg"
    assert note(o) == ""


def test_chooser_is_reusable_across_a_multi_event_BATCH():
    """analyze_one runs once per intrinsic template with the SAME opts.

    The chooser used to write opts.adapt_weight_exponent, so on event 1 it read
    its own event-0 output as a user-supplied exponent and killed the batch with
    SystemExit -- and ILE_extr.sub runs batches, so this broke every real
    multi-event run of --auto.  Pin BOTH halves: it must not raise, and it must
    not mutate opts.
    """
    o = _Opts(auto_adapt_weight_exponent=True)
    betas = [drv.resolve_tempering_exponent(o, 4, 4800) for _ in range(3)]
    assert len(set(betas)) == 1, betas
    assert o.adapt_weight_exponent == 1.0, (
        "the chooser mutated opts; per-run state must not carry per-event meaning")


def test_chooser_returns_the_exponent_rather_than_writing_it_back():
    """Structural companion: the returned value must be what the caller uses.

    A chooser that returns the right number and is called for its side effect is
    the same dead knob as one that returns nothing.
    """
    tree = _driver_tree()
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "resolve_tempering_exponent")
    assert not any(isinstance(t, ast.Attribute) and t.attr == "adapt_weight_exponent"
                   for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   for t in n.targets), "chooser still assigns opts.adapt_weight_exponent"
    ana = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
               and n.name == "analyze_one")
    src = ast.get_source_segment(open(DRIVER).read(), ana) or ""
    assert "resolved_beta = resolve_tempering_exponent(" in src, (
        "analyze_one ignores the chooser's return value")
    assert "_beta = resolved_beta" in src, (
        "the sampler call sites still read opts instead of the resolved value")


def test_smc_puffball_is_not_refused_because_the_exponent_is_inert_there():
    """--smc-puffball routes to smc_puffball_sample, which swallows `temper` in
    **_ignore and exports uniform weights.  Refusing a run over a number that
    does nothing is a false alarm; the guard skipped this path check and did
    exactly that."""
    for o in (_Opts(smc_puffball=True, adapt_weight_exponent=0.09508,
                    supplied=["--adapt-weight-exponent"]),
              _Opts(smc_puffball=True, auto_adapt_weight_exponent=True,
                    supplied=["--auto-adapt-weight-exponent"])):
        assert drv.resolve_tempering_exponent(o, 4, 4800) == 1.0

    # ... and the no-op is REPORTED, not silent
    o = _Opts(smc_puffball=True, adapt_weight_exponent=0.09508,
              supplied=["--adapt-weight-exponent"])
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        drv.resolve_tempering_exponent(o, 4, 4800)
    assert "--smc-puffball ignores" in buf.getvalue(), buf.getvalue()

    # control: WITHOUT --smc-puffball the same exponent is still refused
    with pytest.raises(SystemExit):
        drv.resolve_tempering_exponent(_Opts(adapt_weight_exponent=0.09508), 4, 4800)


def _opts_from_argv(argv):
    """Parse a real command line AND record which tokens it named."""
    o, _ = drv.build_parser().parse_args(list(argv))
    drv.record_supplied_options(o, list(argv))
    return o


@pytest.mark.parametrize("argv", [
    ["--auto-adapt-weight-exponent", "--adapt-weight-exponent", "1.0"],
    ["--auto-adapt-weight-exponent", "--adapt-weight-exponent=1.0"],
    ["--adapt-weight-exponent", "1.0", "--auto-adapt-weight-exponent"],
])
def test_auto_conflicts_with_an_EXPLICIT_DEFAULT_exponent(argv):
    """--auto plus an explicitly-typed --adapt-weight-exponent 1.0 must RAISE.

    THE DEFECT (review): the conflict was detected by comparing the VALUE against
    the default, so passing the default explicitly was silently accepted and the
    chooser then replaced the user's explicit untempered target -- the opposite of
    the documented behaviour, and a change to the sampled target.  Detection is
    now by command-line TOKEN.  The `--opt=value` form and both orderings are
    covered because a token scan that missed either would reintroduce it.
    """
    with pytest.raises(SystemExit, match="would overwrite it"):
        drv.resolve_tempering_exponent(_opts_from_argv(argv), 4, 4800)


def test_auto_alone_is_still_accepted():
    """The negative test above proves nothing if every command line raises."""
    o = _opts_from_argv(["--auto-adapt-weight-exponent"])
    assert drv.resolve_tempering_exponent(o, 4, 4800) != 1.0


def test_was_supplied_reads_tokens_not_values():
    """Structural: the guard must not infer 'user passed it' from the value."""
    o = _opts_from_argv(["--adapt-weight-exponent", "1.0"])
    assert drv.was_supplied(o, "--adapt-weight-exponent") is True
    assert o.adapt_weight_exponent == 1.0, "value IS the default; only the token differs"
    o2 = _opts_from_argv([])
    assert drv.was_supplied(o2, "--adapt-weight-exponent") is False
    # and an options object built without parsing must not fabricate a conflict
    assert drv.was_supplied(_Opts(), "--adapt-weight-exponent") is False


def test_auto_conflicts_raise_at_runtime():
    with pytest.raises(SystemExit) as e1:
        drv.resolve_tempering_exponent(
            _Opts(auto_adapt_weight_exponent=True, adapt_weight_exponent=0.5,
                  supplied=["--auto-adapt-weight-exponent",
                            "--adapt-weight-exponent"]), 4, 4800)
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


def test_target_frac_given_is_decided_by_the_TOKEN_not_the_value():
    """RETARGETED.  This used to require that _target_ess_was_given compare
    against a named default constant.  That whole mechanism was the defect: a
    user who explicitly passes the default value was indistinguishable from one
    who passed nothing.  It now reads the command line.
    """
    import ast as _ast
    src = open(DRIVER).read()
    fn = next(n for n in _ast.walk(_ast.parse(src)) if isinstance(n, _ast.FunctionDef)
              and n.name == "_target_ess_was_given")
    body = _ast.get_source_segment(src, fn) or ""
    assert "was_supplied" in body, "still inferring from the value"
    assert "_TARGET_EXPORT_ESS_FRAC_DEFAULT" not in body, (
        "still comparing against the default constant")
    # behaviour: explicitly passing the default counts as supplied
    o = _opts_from_argv(["--target-export-ess-frac",
                         str(drv._TARGET_EXPORT_ESS_FRAC_DEFAULT)])
    assert drv._target_ess_was_given(o) is True


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
