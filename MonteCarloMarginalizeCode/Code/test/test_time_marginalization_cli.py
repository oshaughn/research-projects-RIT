#!/usr/bin/env python3
"""The DRIVER, through a subprocess -- not the library.

A library test cannot see the driver's option at all.  The companion `rift_O4d`
work found an `AttributeError` in exactly this seam (a module referenced by its
submodule name rather than by the alias the import bound), which every one of
its library tests passed straight over and which would have broken *every*
invocation of the driver, with or without the flag.  These tests exist so that
seam is exercised.

They only run `--help` and the startup path, so they need no data and no frames:
the option is validated, and the banner printed, before any data is touched.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
ILE = os.path.join(CODE, "bin", "integrate_likelihood_extrinsic_batchmode")


def _run(*args, timeout=600):
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run([sys.executable, ILE, *args], capture_output=True,
                          text=True, env=env, timeout=timeout)


@pytest.fixture(scope="module")
def help_text():
    p = _run("--help")
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout


def test_the_option_is_registered_on_the_driver(help_text):
    """Weakest of these, and deliberately labelled as such: `--help` exits before
    the validation line, so this passes even with the import-alias bug the
    companion line hit.  The three startup tests below are the ones that catch
    it -- verified by injecting that exact bug, which fails those three and not
    this one."""
    assert "--time-marginalization-quadrature" in help_text


def test_help_states_that_the_option_changes_results_and_costs_something(help_text):
    i = help_text.index("--time-marginalization-quadrature")
    blurb = " ".join(help_text[i:i + 1600].split())
    assert "simpson" in blurb and "bandlimited" in blurb
    assert "CHANGES RESULTS" in blurb
    assert "not free" in blurb.lower() or "cost" in blurb.lower()


def test_a_misspelled_value_is_refused_before_any_data_is_touched():
    p = _run("--time-marginalization-quadrature", "trapezoid")
    assert p.returncode != 0
    combined = p.stdout + p.stderr
    assert "time_quadrature must be one of" in combined, combined[-3000:]


def test_the_banner_names_the_quadrature_actually_in_force():
    """Not a silently-inert flag: the driver says which path is running."""
    p = _run("--time-marginalization-quadrature", "bandlimited")
    out = p.stdout + p.stderr
    assert "Time-marginalization quadrature : bandlimited" in out, out[-3000:]
    assert "RESULT-CHANGING" in out


def test_the_default_is_simpson_when_the_flag_is_absent():
    p = _run()
    out = p.stdout + p.stderr
    assert "Time-marginalization quadrature : simpson" in out, out[-3000:]
    assert "historical" in out


def test_every_marginalizing_driver_call_site_passes_the_option():
    """A ledger over the driver's call sites, checked structurally with `ast`.

    There is no cheap behavioural way to reach the three call sites -- they need
    frames, a PSD and a precompute -- so this pins the invariant instead: every
    call to the likelihood that actually MARGINALIZES must forward
    time_quadrature.  A call that returns the lnL(t) timeseries is exempt,
    because the option does not apply to it.

    The failure this exists for is a new call site added later, or one of the
    three edited, silently reverting to Simpson while the flag still parses and
    the banner still prints.
    """
    import ast
    tree = ast.parse(open(ILE).read())
    target = "DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop"
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == target:
                kw = {k.arg for k in node.keywords if k.arg}
                sites.append((node.lineno, "return_lnLt" in kw, "time_quadrature" in kw))
    assert sites, "no call sites found -- has the function been renamed?"
    unwired = [ln for ln, exports, wired in sites if not exports and not wired]
    assert not unwired, (
        "driver call sites that marginalize but do not pass time_quadrature, "
        "at lines %s" % unwired)
    assert sum(1 for _, exports, wired in sites if wired) >= 3, sites
