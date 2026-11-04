"""Driver-seam tests for the four-axis policy's observability and plan knobs.

These are SUBPROCESS tests on purpose.  Every option here is read in
``build_parser`` and validated in ``check_critical_and_report``, both of which
run before any likelihood is built, so a library-level test of the policy
module cannot see them at all: it would exercise ``PolicyConfig`` directly and
pass no matter what the command line does.  The failure this guards against is
the one this pipeline keeps hitting -- a flag that is accepted and then
silently inert -- so each case asserts that a misuse is REFUSED rather than
ignored, and that the accepting combination is not refused.

A separate file from test_direct_marginalization_policy.py because these need
no JAX device, no synthetic data and no fixture: they are parser and validator
behaviour only, and they run in well under a second each.
"""

import os
import subprocess
import sys

import pytest


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DRIVER = os.path.join(_ROOT, "bin", "integrate_likelihood_extrinsic_jax")

# Enough of a policy request to reach the policy's own validation.  --sim-xml is
# a path that exists so argument parsing does not fail for an unrelated reason;
# the run never gets as far as reading it, because every case below is decided
# at parse time.
_POLICY = ("--mode", "flowmc-phipsimarg", "--distance-marginalization",
           "--direct-marginalization-policy", "auto",
           "--angle-marg-scheme", "exact", "--sim-xml", os.devnull)


def _run(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = _ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = "cpu"
    p = subprocess.run([sys.executable, _DRIVER] + list(args), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=600)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def test_the_help_lists_every_new_policy_and_smc_knob():
    rc, out = _run("--help")
    assert rc == 0, out[-1500:]
    for flag in ("--direct-marginalization-policy-probe-rows",
                 "--direct-marginalization-policy-probe-only",
                 "--direct-marginalization-max-modes",
                 "--direct-marginalization-enriched-max-modes",
                 "--direct-marginalization-base-oversample",
                 "--direct-marginalization-enriched-oversample",
                 "--direct-marginalization-max-starts",
                 "--direct-marginalization-convergence-tol-nats",
                 "--direct-marginalization-time-guard-tol-nats",
                 "--direct-marginalization-reserve-time-refine-max",
                 "--smc-is-samples"):
        assert flag in out, flag


@pytest.mark.parametrize("flag,value", [
    ("--direct-marginalization-policy-probe-rows", "8"),
    ("--direct-marginalization-max-modes", "16"),
    ("--direct-marginalization-enriched-max-modes", "16"),
    ("--direct-marginalization-base-oversample", "2"),
    ("--direct-marginalization-enriched-oversample", "4"),
    ("--direct-marginalization-max-starts", "64"),
    ("--direct-marginalization-convergence-tol-nats", "0.5"),
    ("--direct-marginalization-time-guard-tol-nats", "0.5"),
    ("--direct-marginalization-reserve-time-refine-max", "8"),
])
def test_a_policy_knob_without_the_policy_is_refused_not_ignored(flag, value):
    """Each knob is inert unless the policy is on, so passing one without it is
    a fatal mistake rather than a silently dropped request."""
    rc, out = _run("--mode", "flowmc-phipsimarg", flag, value)
    assert rc != 0, out[-1500:]
    assert "inert" in out, out[-1500:]


def test_a_negative_smc_is_sample_count_is_refused():
    """A negative count reaches the SMC proposal draw, whose bare exception
    handler would swallow it and publish the raw SMC evidence instead of the IS
    evidence, with nothing in the output saying the estimator changed."""
    rc, out = _run("--mode", "flowmc-phipsimarg", "--smc-is-samples", "-1")
    assert rc != 0, out[-1500:]
    assert "smc-is-samples" in out, out[-1500:]


def test_probe_only_without_probe_rows_is_refused():
    """--probe-only with zero rows would exit having measured nothing."""
    rc, out = _run(*(_POLICY + ("--direct-marginalization-policy-probe-only",)))
    assert rc != 0, out[-1500:]
    assert "probe-rows" in out, out[-1500:]


def test_an_enriched_plan_narrower_than_the_base_plan_is_refused():
    """Acceptance compares a base plan against an enriched one that must be
    able to nest it; a narrower enriched cap declines on mode nesting every
    row, which would read as a property of the data."""
    rc, out = _run(*(_POLICY + ("--direct-marginalization-max-modes", "16",
                                "--direct-marginalization-enriched-max-modes",
                                "8")))
    assert rc != 0, out[-1500:]
    assert "nest" in out, out[-1500:]


def test_an_escalation_ceiling_below_its_floor_is_refused():
    rc, out = _run(*(_POLICY + (
        "--direct-marginalization-reserve-time-refine", "8",
        "--direct-marginalization-reserve-time-refine-max", "4")))
    assert rc != 0, out[-1500:]
    assert "reserve-time-refine-max" in out, out[-1500:]


@pytest.mark.parametrize("flag", [
    "--direct-marginalization-convergence-tol-nats",
    "--direct-marginalization-time-guard-tol-nats",
])
def test_a_nonpositive_tolerance_is_refused(flag):
    rc, out = _run(*(_POLICY + (flag, "-1")))
    assert rc != 0, out[-1500:]
    assert "finite and positive" in out, out[-1500:]


def test_the_recorded_accepting_operating_point_is_not_refused():
    """The counterpart to every case above: the combination that reaches the
    four-axis branch must pass validation.  It still exits nonzero, on a
    missing --event-time, which is the point -- the policy's own validation is
    behind it, so a future tightening that rejected this configuration would
    be caught here rather than in a run."""
    rc, out = _run(*(_POLICY + (
        "--direct-marginalization-time-guard", "128",
        "--direct-marginalization-max-modes", "16",
        "--direct-marginalization-enriched-max-modes", "16",
        "--direct-marginalization-base-oversample", "2",
        "--direct-marginalization-enriched-oversample", "4",
        "--direct-marginalization-reserve-time-refine", "4",
        "--direct-marginalization-reserve-time-refine-max", "4",
        "--direct-marginalization-policy-probe-rows", "8",
        "--direct-marginalization-policy-probe-only")))
    assert rc != 0, out[-1500:]
    assert "event-time" in out, out[-1500:]
    assert "direct-marginalization" not in out.split("error:")[-1], out[-1500:]


# --------------------------------------------------- the note's return arity

def _load_driver():
    """Import the driver script as a module (it has no .py extension)."""
    import importlib.util
    spec = importlib.util.spec_from_loader(
        "ile_jax_driver",
        importlib.machinery.SourceFileLoader("ile_jax_driver", _DRIVER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("policy,ledger,theta_rows", [
    ("off", object(), 4),      # policy disabled
    ("auto", None, 4),         # no ledger built
    ("auto", object(), 0),     # no rows to evaluate
])
def test_the_note_returns_three_values_on_every_early_path(policy, ledger,
                                                           theta_rows):
    """``return_values=True`` must return the same NUMBER of values on every
    path, including the ones that give up early.

    The caller unpacks three.  Two early returns handed back two, so any caller
    reaching them died on an unpacking error rather than on the condition the
    early return was written to handle.  None of the three is reachable from
    the probe today, which is exactly why it needs pinning: the guard is
    unexercised, so nothing else would notice it drifting.
    """
    import numpy as np
    mod = _load_driver()

    class _Like(object):
        pass

    like = _Like()
    like.direct_marginalization_policy = policy
    like._batched_ledger = ledger
    theta = np.zeros((theta_rows, 3))

    out = mod.direct_marginalization_policy_note(like, theta,
                                                 return_values=True)
    assert isinstance(out, tuple) and len(out) == 3, out
    assert isinstance(out[0], str)

    plain = mod.direct_marginalization_policy_note(like, theta)
    assert isinstance(plain, str), plain


def test_every_driver_option_string_is_registered_exactly_once():
    """optparse SILENTLY keeps the last registration of a duplicated option.

    A parallel branch merge produced two add_option calls for one flag; --help
    rendered correctly from the first while the parser used the second, so the
    driver ran at a different default from the one documented and nothing
    raised.  That is invisible to every other test in this file, which all go
    through the same parser and would agree with each other.

    Read from the SOURCE rather than the parser for the same reason: the parser
    only knows the winner.
    """
    import collections
    import re

    src = open(_DRIVER, encoding="utf-8").read()
    names = re.findall(r'add_option\(\s*"(--[A-Za-z0-9-]+)"', src)
    dupes = {n: c for n, c in collections.Counter(names).items() if c > 1}
    assert not dupes, "option strings registered more than once: %r" % (dupes,)


def test_each_policy_flag_default_matches_its_PolicyConfig_field():
    """The driver default and the library default must be the SAME number.

    The symptom that started this guard: a run used max_starts 32 from the
    driver while PolicyConfig carried 128, so --help, the config and the
    executed run disagreed and nothing raised.  Any knob whose two defaults
    drift silently changes what a bare command line computes, and a default
    change landing on one side only is the easiest way to produce that.

    Deliberately compares VALUES, not a hardcoded expectation: when a default is
    legitimately changed (RO approved max_starts 32 -> 128), this test keeps
    passing as long as BOTH sides move, and fails the moment only one does.
    """
    from RIFT.likelihood.jax_ile.direct_marginalization_policy import PolicyConfig

    mod = _load_driver()
    parser = mod.build_parser()
    cfg = PolicyConfig()

    pairs = {
        "direct_marginalization_time_guard": "time_guard",
        "direct_marginalization_reserve_time_refine": "reserve_time_refine",
        "direct_marginalization_reserve_time_refine_max": "reserve_time_refine_max",
        "direct_marginalization_error_budget_nats": "total_value_error_budget_nats",
        "direct_marginalization_max_modes": "max_modes",
        "direct_marginalization_enriched_max_modes": "enriched_max_modes",
        "direct_marginalization_base_oversample": "base_oversample",
        "direct_marginalization_enriched_oversample": "enriched_oversample",
        "direct_marginalization_max_starts": "base_max_starts",
        "direct_marginalization_max_time_nodes": "max_time_nodes",
        "direct_marginalization_convergence_tol_nats": "convergence_tol_nats",
        "direct_marginalization_time_guard_tol_nats": "time_guard_tol_nats",
    }
    mismatched = {}
    for dest, field in pairs.items():
        if not hasattr(cfg, field):
            continue
        drv = parser.defaults.get(dest, "<absent>")
        if drv == "<absent>":
            continue
        lib = getattr(cfg, field)
        if drv != lib:
            mismatched[dest] = (drv, lib)
    assert not mismatched, (
        "driver default != PolicyConfig default for %r "
        "(driver, library)" % (mismatched,))


def test_every_policy_flag_help_states_its_real_default():
    """``--help`` must not quote a number the flag no longer uses.

    The value guard above was passing while four help strings still read
    "(default 4)", "(default 8)", "(default 1)", "(default 2)" -- the pre-#280
    portfolio -- and --direct-marginalization-time-guard read "(default 16)"
    against a default of 128.  The values had been repointed at PolicyConfig
    and the prose had not, so the two defaults agreed with each other and
    disagreed with what --help told the operator.  A knob's documented default
    is what someone reads before deciding whether to pass it, so a stale one
    misconfigures a run exactly as a stale value does.

    Reads the parser's rendered help, not the source, so an interpolation that
    silently fails to interpolate is caught too.  A stated default may carry a
    trailing constraint ("16; must be >= 2") -- only the leading token is the
    number, and the rest of the parenthetical may explain the value
    ("1: row at a time", "0 = off").
    """
    import re

    mod = _load_driver()
    parser = mod.build_parser()

    stale = {}
    for opt in parser._get_all_options():
        dest = opt.dest
        if not dest or not str(dest).startswith("direct_marginalization"):
            continue
        for match in re.finditer(r"\(default ([^\s);,:=]+)", opt.help or ""):
            stated = match.group(1).strip().strip("'\"")
            actual = parser.defaults.get(dest)
            try:
                ok = float(stated) == float(actual)
            except (TypeError, ValueError):
                ok = stated == str(actual)
            if not ok:
                stale[opt.get_opt_string()] = (stated, actual)
    assert not stale, (
        "help text states a default the flag does not use %r "
        "(stated, actual)" % (stale,))
