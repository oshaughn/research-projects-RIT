"""Guard the portfolio member list that util_ConstructIntrinsicPosterior_GenericCoordinates.py
hands to mcsamplerPortfolio, and the per-member arguments that ride with it.

Two defects, both long-standing and both silent at the point of failure:

1. The construction loop reuses one `sampler` variable and only skips on `sampler is None`, so an
   unrecognized --sampler-portfolio name FOLLOWING a recognized one re-appended THE SAME OBJECT.
   Not merely a duplicate: setup() then runs twice on one sampler and the run dies in
   mcsamplerAdaptiveVolume.sample_from_bins with
   "ValueError: operands could not be broadcast together with shapes (4,) (1,2)",
   which names neither the portfolio nor the option that caused it.  Measured on this input:
   AV+GMM exits 0, AV+bogus+GMM exits 1, AV+bogus exits 1.

2. setup() was called with `portolio_args=`, while mcsamplerPortfolio.setup() reads
   kwargs['portfolio_args'].  setup() takes **kwargs, so the misspelt name was accepted and
   ignored -- every --sampler-portfolio-args on this driver was dropped with no message, and the
   driver's own "PRE_EVAL"/"ARGS" lines still printed, so the log looked like it had worked.

Both are asserted against a RUN of the driver.  A static check of the kwarg name would cover (2)
only, and nothing static sees (1).  One subprocess, ~11 s.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
DRIVER = os.path.join(CODE, "bin", "util_ConstructIntrinsicPosterior_GenericCoordinates.py")


def _write_ile(tmp_path, n=300):
    """A tiny ILE .dat: indx m1 m2 a1x a1y a1z a2x a2y a2z lnL sigma_lnL ntot neff, with lnL
    peaked in chirp mass so the RF fit has something to find."""
    rng = np.random.default_rng(7)
    m1 = rng.uniform(25.0, 45.0, n)
    m2 = np.minimum(rng.uniform(15.0, 30.0, n), m1)
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    lnL = -0.5 * ((mc - 28.0) / 1.5) ** 2 + 20.0
    z = np.zeros(n)
    path = os.path.join(str(tmp_path), "ile.dat")
    np.savetxt(path, np.column_stack([np.arange(n), m1, m2, z, z, z, z, z, z,
                                      lnL, np.full(n, 0.01), np.full(n, 1000.0),
                                      np.full(n, 100.0)]))
    return path


@pytest.fixture(scope="module")
def cip_portfolio_run(tmp_path_factory):
    """One portfolio run carrying BOTH provocations: an unrecognized member name sandwiched
    between two good ones, and one --sampler-portfolio-args dict per member."""
    if not os.path.exists(DRIVER):
        pytest.fail("driver missing: %s" % DRIVER)
    tmp_path = tmp_path_factory.mktemp("cip_portfolio")
    fname = _write_ile(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMP_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["XDG_CACHE_HOME"] = os.path.join(str(tmp_path), "cache")
    env["MPLCONFIGDIR"] = os.path.join(str(tmp_path), "mpl")
    cmd = [sys.executable, DRIVER,
           "--fname", fname,
           "--parameter", "mc", "--parameter", "delta_mc",
           "--fit-method", "rf",
           "--sampler-method", "portfolio",
           "--sampler-portfolio", "AV",
           "--sampler-portfolio", "bogus",      # must contribute NOTHING
           "--sampler-portfolio", "GMM",
           "--sampler-portfolio-args", "{}",              # -> AV
           "--sampler-portfolio-args", "{'n_comp':7}",    # -> GMM
           "--n-max", "4000", "--n-eff", "10",
           "--n-output-samples", "20",
           "--no-plots", "--internal-use-lnL"]
    # timeout MUST stay under the core-unit job's timeout-minutes: 20 (1200 s).  At 1800 s it
    # could never fire: GitHub cancels the job first, so a hung driver yields a cancelled job
    # and NO test diagnostic.  900 s matches the rest of the manifest.
    return subprocess.run(cmd, cwd=str(tmp_path), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900)


_MEMBER_ID = re.compile(r"object at (0x[0-9a-fA-F]+)")


def _member_ids(lines):
    """The ADDRESS out of each member setup line.

    NOT `split("object at ")[-1]`.  That keeps the trailing per-member args dict, and the members
    here deliberately carry DIFFERENT args ({} vs {'n_comp':7}) -- so two setups of the SAME
    aliased object compared as distinct and the distinctness assertion passed on a portfolio that
    was one object twice.  Demonstrated with the loop reset removed:
        parsed ids : ['0x7f5914943950> {}', "0x7f5914943950> {'n_comp': 7}"]  -> 2 distinct
        true addrs : ['0x7f5914943950', '0x7f5914943950']                     -> 1 distinct
    A line that does not match is an error rather than a skip: split() would return the whole
    line if MCSampler ever gained a __repr__, silently degrading this into a class-name check.

    Comparing CPython id-reprs is sound here only because every member is alive and referenced by
    portfolio_realizations when these lines print, so no address can be reused.
    """
    out = []
    for ln in lines:
        m = _MEMBER_ID.search(ln)
        assert m, "cannot read a member address out of: %r" % ln
        out.append(m.group(1))
    return out


def _member_lines(proc):
    # setup() prints one line per MEMBER including the object's repr, so this sees ALIASING --
    # which a count of the driver's own "PORTFOLIO: adding" lines cannot.
    return [ln for ln in proc.stdout.splitlines() if "PORTFOLIO setup" in ln]


def test_cip_portfolio_run_completes(cip_portfolio_run):
    proc = cip_portfolio_run
    assert "could not be broadcast together" not in proc.stdout, \
        "an aliased member was set up twice and corrupted its own bin state:\n" \
        + proc.stdout[-3000:]
    assert proc.returncode == 0, \
        "portfolio run exited %d:\n%s" % (proc.returncode, proc.stdout[-3000:])


def test_unrecognized_member_is_skipped_not_aliased(cip_portfolio_run):
    """'bogus' must add no member, and AV and GMM must survive as DISTINCT objects.

    Asserted in both directions on purpose: a reset placed wrongly drops members that ARE
    recognized, which would look like a pass to a test that only counted duplicates.
    """
    lines = _member_lines(cip_portfolio_run)
    ids = _member_ids(lines)
    assert len(ids) == 2, \
        "expected exactly 2 members (AV, GMM); 'bogus' must add none. Got %d:\n%s" \
        % (len(ids), "\n".join(lines))
    assert len(set(ids)) == 2, \
        "the two members are THE SAME OBJECT -- the unknown name was aliased onto the previous " \
        "member:\n" + "\n".join(lines)
    assert any("ignoring unrecognized --sampler-portfolio" in ln and "bogus" in ln
               for ln in cip_portfolio_run.stdout.splitlines()), \
        "the dropped name was not reported:\n" + cip_portfolio_run.stdout[-3000:]
    assert any("mcsamplerAdaptiveVolume" in ln for ln in lines), "the AV member was dropped"
    assert any("mcsamplerEnsemble" in ln for ln in lines), "the GMM member was dropped"


def test_portfolio_args_reach_the_right_member(cip_portfolio_run):
    """--sampler-portfolio-args must arrive at setup(), matched to members IN ORDER.

    Pins the kwarg spelling: setup() reads kwargs['portfolio_args'] and takes **kwargs, so
    `portolio_args=` is accepted and silently ignored.  The empty dict for AV and the n_comp for
    GMM are checked separately, so a change that delivers the args but scrambles the pairing
    fails too.
    """
    lines = _member_lines(cip_portfolio_run)
    assert lines, "member setup() never ran:\n" + cip_portfolio_run.stdout[-3000:]
    gmm = [ln for ln in lines if "mcsamplerEnsemble" in ln]
    av = [ln for ln in lines if "mcsamplerAdaptiveVolume" in ln]
    assert gmm and av, "\n".join(lines)
    assert "'n_comp': 7" in gmm[0], \
        "--sampler-portfolio-args never reached the GMM member; setup() saw:\n" + gmm[0]
    assert "'n_comp'" not in av[0], \
        "the GMM member's arguments leaked onto the AV member:\n" + av[0]


def test_cip_portfolio_args_must_be_dicts(tmp_path):
    """A non-dict --sampler-portfolio-args entry must be refused, not delivered.

    This is a REGRESSION GUARD ON THE SPELLING FIX itself.  While setup() was called with the
    misspelt `portolio_args`, every entry was discarded and a malformed one was harmless: the run
    completed (exit 0) with the args ignored.  Delivering them correctly makes the same input
    fatal inside mcsamplerPortfolio.setup():

        TypeError: cannot convert dictionary update sequence element #0 to a sequence

    Measured on the base vs the spelling fix alone: exit 0 -> exit 1.  So the guard is part of
    the fix, not an extra; without it this driver trades a silently-ignored option for a crash in
    the integrator.
    """
    if not os.path.exists(DRIVER):
        pytest.fail("driver missing: %s" % DRIVER)
    fname = _write_ile(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMP_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["XDG_CACHE_HOME"] = os.path.join(str(tmp_path), "cache")
    env["MPLCONFIGDIR"] = os.path.join(str(tmp_path), "mpl")
    cmd = [sys.executable, DRIVER,
           "--fname", fname,
           "--parameter", "mc", "--parameter", "delta_mc",
           "--fit-method", "rf",
           "--sampler-method", "portfolio",
           "--sampler-portfolio", "AV",
           "--sampler-portfolio", "GMM",
           "--sampler-portfolio-args", "{}",
           "--sampler-portfolio-args", "[1,2]",
           "--n-max", "4000", "--n-eff", "10",
           "--n-output-samples", "20",
           "--no-plots", "--internal-use-lnL"]
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900)
    assert "cannot convert dictionary update sequence" not in proc.stdout, \
        "a malformed args entry reached setup():\n" + proc.stdout[-3000:]
    assert proc.returncode == 99, \
        "expected a clean option-mismatch exit, got %d:\n%s" % (proc.returncode,
                                                                proc.stdout[-3000:])
    assert "OPTION MISMATCH" in proc.stdout, proc.stdout[-2000:]
