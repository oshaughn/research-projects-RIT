#!/usr/bin/env python
"""The gate's two entry points must not disagree about WHICH RIFT they measure.

`run_shape_recovery.sh` exports `PYTHONPATH=<checkout>/MonteCarloMarginalizeCode/Code`; the pytest
entry point documented in `test_shape_recovery.py` exported nothing, so in any environment with
RIFT installed (every IGWN conda env) it measured the INSTALLED RIFT and reported pass/fail as if
it had gated the branch.  Measured on `GMM mix_d4_n2_s101`, quick budget, run seed 987654:
n_eff 42.3 from the branch, 4.6 from /cvmfs/software.igwn.org/conda/envs/igwn.

These tests are CHEAP (no RIFT_RUN_EXPENSIVE, no sampling).  The one that matters is
`test_pytest_entry_point_refuses_a_foreign_rift`: it drives the real entry point in a subprocess
and fails on the pre-fix module, rather than only exercising the helper it calls.

Run:  python -m pytest -v test_rift_provenance_guard.py
"""
import os
import subprocess
import sys

import pytest

import shape_recovery as SR

HERE = os.path.dirname(os.path.abspath(__file__))


def test_conftest_pins_the_cpu_path():
    """Parity with the shell driver's `export CUDA_VISIBLE_DEVICES=""`, which pytest lacked.

    Asserted rather than assumed: it lives in conftest.py, and a conftest that stops being
    collected fails nothing on its own.
    """
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", \
        "conftest.py did not pin the CPU path; the pytest run is not the gate's experiment"


def test_checkout_code_dir_defaults_to_the_enclosing_checkout():
    code = SR.checkout_code_dir()
    assert os.path.basename(code) == "Code"
    assert os.path.basename(os.path.dirname(code)) == "MonteCarloMarginalizeCode"
    # the enclosing one, specifically -- this file sits three levels below it
    assert os.path.realpath(code) == os.path.realpath(os.path.join(HERE, "..", "..", ".."))


def test_checkout_code_dir_honours_a_named_checkout():
    """The base-vs-candidate idiom measures some OTHER checkout on purpose; it must stay sayable."""
    assert SR.checkout_code_dir("/some/other/checkout") == \
        os.path.join("/some/other/checkout", "MonteCarloMarginalizeCode", "Code")


def test_a_foreign_rift_is_refused(monkeypatch):
    monkeypatch.setattr(SR, "rift_package_dir",
                        lambda: "/cvmfs/software.igwn.org/conda/envs/igwn/lib/python3.11"
                                "/site-packages/RIFT")
    with pytest.raises(RuntimeError) as exc:
        SR.assert_rift_under_test()
    msg = str(exc.value)
    assert "WRONG RIFT" in msg
    # both sides named, and the fix spelled out: a guard that only says "no" teaches nothing
    assert "/cvmfs/software.igwn.org" in msg
    assert os.path.realpath(os.path.join(SR.checkout_code_dir(), "RIFT")) in msg
    assert "PYTHONPATH" in msg and "RIFT_SHAPE_CHECKOUT" in msg


def test_an_unimportable_rift_is_refused(monkeypatch):
    """`import RIFT` failing must not read as "nothing to compare, carry on"."""
    monkeypatch.setattr(SR, "rift_package_dir", lambda: None)
    with pytest.raises(RuntimeError) as exc:
        SR.assert_rift_under_test()
    assert "not importable" in str(exc.value)


def test_the_checkouts_own_rift_is_accepted(monkeypatch):
    want = os.path.realpath(os.path.join(SR.checkout_code_dir(), "RIFT"))
    monkeypatch.setattr(SR, "rift_package_dir", lambda: want)
    assert SR.assert_rift_under_test() == want


def test_the_checkouts_own_rift_is_accepted_through_a_symlinked_path(monkeypatch, tmp_path):
    """Compared by realpath: worktrees and NFS homes reach the same tree by several names."""
    link = tmp_path / "checkout"
    link.symlink_to(os.path.dirname(os.path.dirname(SR.checkout_code_dir())))
    monkeypatch.setattr(SR, "rift_package_dir",
                        lambda: os.path.realpath(os.path.join(SR.checkout_code_dir(), "RIFT")))
    assert SR.assert_rift_under_test(str(link))


def test_pytest_entry_point_refuses_a_foreign_rift(tmp_path):
    """THE regression test: drive the documented pytest invocation, pointed at a foreign checkout.

    Collection alone is enough -- the guard runs at import -- so this costs no sampling.  Before
    the fix this exited 0 and reported 4 tests ready to run against whatever RIFT was installed.
    """
    env = dict(os.environ)
    env["RIFT_RUN_EXPENSIVE"] = "1"
    env["RIFT_SHAPE_CHECKOUT"] = str(tmp_path / "not-the-checkout")
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                           os.path.join(HERE, "test_shape_recovery.py")],
                          env=env, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode != 0, "the pytest entry point collected happily against a foreign RIFT:\n" + out
    assert "WRONG RIFT" in out, out


def test_pytest_entry_point_collects_against_its_own_checkout():
    """...and the guard is not simply refusing everything: the correct invocation still works."""
    env = dict(os.environ)
    env["RIFT_RUN_EXPENSIVE"] = "1"
    env.pop("RIFT_SHAPE_CHECKOUT", None)
    code = SR.checkout_code_dir()
    env["PYTHONPATH"] = code + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                           os.path.join(HERE, "test_shape_recovery.py")],
                          env=env, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, out
    assert "test_shape_recovery" in out, out


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
