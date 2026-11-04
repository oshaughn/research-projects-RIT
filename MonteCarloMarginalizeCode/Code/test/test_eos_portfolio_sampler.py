"""Guard `--sampler-method portfolio` in bin/util_ConstructEOSPosterior.py.

Motivation.  The EOS driver built `mcsamplerPortfolio.MCSampler(portfolio=...)` and then went
straight to `sampler.integrate(...)`, never calling `sampler.setup()`.  `setup()` is the only
place that

  (a) initialises `portfolio_breakpoints`, left at None by __init__, so the very first draw()
      evaluated `None <= iteration` and the run died with
      `TypeError: '<=' not supported between instances of 'NoneType' and 'int'`; and
  (b) calls setup() on each MEMBER, which is what builds AV's my_ranges/dx/V_s and the GMM
      integrator -- a member that never got setup() is cold and cannot draw.

So EVERY `--sampler-method portfolio` invocation of this driver failed, after the fit had already
been paid for.  Which failure you saw depended on one flag, and only one of the two is the
breakpoints defect above:

  * WITH --internal-use-lnL, the breakpoints TypeError in (a), raised in the first draw();
  * WITHOUT it, an earlier `Exception: mcsamplerPortfolio: must integrate lnL` from
    integrate(), because the driver did not force internal_use_lnL for a portfolio the way it
    does for AV.

The fixture below passes the flag explicitly so it lands on the first of those; the forcing has
its own test.  The sibling driver util_ConstructIntrinsicPosterior_GenericCoordinates.py has
always called setup() here.

These tests RUN THE DRIVER.  A static check of "is setup() called" would pass on a setup() call
placed where it cannot fire -- and one such trap is live here: the portfolio branch REBINDS
opts.sampler_method to 'GMM' whenever a GMM member is requested, so a guard written as
`opts.sampler_method == "portfolio"` silently skips setup() for exactly the mixed portfolios
that need it.  test_portfolio_args_reach_members covers that configuration on purpose.

Cost: three driver subprocesses, ~10 s each on the IGWN conda python.  Compare
test_cleanile_intrinsic_precision.py, already in the core-unit gate on the same basis.
"""
import os
import re
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))
DRIVER = os.path.join(CODE, "bin", "util_ConstructEOSPosterior.py")


_MEMBER_ID = re.compile(r"object at (0x[0-9a-fA-F]+)")


def _member_ids(lines):
    """The ADDRESS out of each `PORTFOLIO setup <... object at 0x...> {...}` line.

    Not `split("object at ")[-1]`: that keeps the trailing per-member args dict, so two setups of
    THE SAME object with different args compare as distinct and an aliasing test passes on a
    portfolio that is one object twice.  It also returns the whole line when the substring is
    absent, which would silently degrade this into a no-op if MCSampler ever gains a __repr__ --
    so a line that does not match is an error, not a skip.

    Comparing CPython id-reprs is sound here only because every member is alive and referenced by
    portfolio_realizations when these lines print, so no address can be reused.
    """
    out = []
    for ln in lines:
        m = _MEMBER_ID.search(ln)
        assert m, "cannot read a member address out of: %r" % ln
        out.append(m.group(1))
    return out


def _write_grid(tmp_path, n=200):
    """A tiny separable Gaussian in two dummy coordinates, in the driver's .dat format:
    column 0 lnL, column 1 sigma_lnL, then one column per parameter named in the header."""
    rng = np.random.default_rng(20260915)
    xx = rng.uniform(-1.0, 1.0, n)
    yy = rng.uniform(-1.0, 1.0, n)
    lnL = -0.5 * ((xx / 0.4) ** 2 + (yy / 0.4) ** 2) + 10.0
    path = os.path.join(str(tmp_path), "fake_int_grid.dat")
    np.savetxt(path, np.column_stack([lnL, np.zeros(n), xx, yy]),
               header=" lnL sigma_lnL xx yy")
    return path


def _run(tmp_path, extra):
    """Run the driver in tmp_path.  PYTHONPATH is PREPENDED so the driver imports THIS
    checkout and not an installed RIFT (.travis scripts do not export it for us)."""
    if not os.path.exists(DRIVER):
        pytest.fail("driver missing: %s" % DRIVER)
    fname = _write_grid(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["OMP_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    # Keep the driver's caches inside tmp_path.  Without these it writes
    # ~/.cache/matplotlib/fontlist-v390.json, ~/.config/matplotlib and ~/.cache/arviz into the
    # real HOME -- which on CIT is the NFS home -- on every run of every test in this file.
    env["XDG_CACHE_HOME"] = os.path.join(str(tmp_path), "cache")
    env["MPLCONFIGDIR"] = os.path.join(str(tmp_path), "mpl")
    cmd = [sys.executable, DRIVER,
           "--fname", fname,
           "--parameter", "xx", "--parameter", "yy",
           "--fit-method", "rf",            # 'rf' and 'gp' are the only fits this driver builds
           "--n-max", "4000", "--n-step", "1000", "--n-eff", "10",
           "--n-output-samples", "20",
           "--no-plots", "--ignore-errors-in-data"] + extra
    proc = subprocess.run(cmd, cwd=str(tmp_path), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, timeout=900)
    return proc


@pytest.fixture(scope="module")
def portfolio_run(tmp_path_factory):
    """One successful --sampler-method portfolio run, shared by the assertions below."""
    tmp_path = tmp_path_factory.mktemp("eos_portfolio")
    # --internal-use-lnL is passed EXPLICITLY.  The driver now forces it for a portfolio, but
    # without it on the unfixed base this invocation stops earlier, at
    # "mcsamplerPortfolio: must integrate lnL", and never reaches the breakpoints defect these
    # assertions name.  test_portfolio_forces_internal_use_lnL covers the forcing separately.
    proc = _run(tmp_path, ["--sampler-method", "portfolio", "--sampler-portfolio", "AV",
                           "--internal-use-lnL"])
    return proc, str(tmp_path)


def test_portfolio_completes(portfolio_run):
    """THE regression: the driver used to die inside draw() on the first chunk."""
    proc, _ = portfolio_run
    assert "TypeError: '<=' not supported between instances of 'NoneType' and 'int'" \
        not in proc.stdout, \
        "portfolio_breakpoints was never initialised -- sampler.setup() is not being called:\n" \
        + proc.stdout[-3000:]
    assert proc.returncode == 0, \
        "--sampler-method portfolio exited %d:\n%s" % (proc.returncode, proc.stdout[-3000:])


def test_portfolio_writes_a_finite_evidence(portfolio_run):
    """A non-crashing run is not enough: it must produce the evidence the pipeline reads.

    The peak of the input is lnL=10 over a unit-ish box, so ln Z lands near 10; the bound is
    deliberately loose (RF fit + MC scatter at n_eff=10), it only has to exclude nan/inf and a
    log taken twice.
    """
    proc, wd = portfolio_run
    assert proc.returncode == 0, proc.stdout[-3000:]
    fname = os.path.join(wd, "output-EOS-integral")
    assert os.path.exists(fname), "no evidence file written:\n" + proc.stdout[-2000:]
    lnZ = float(np.atleast_1d(np.loadtxt(fname))[0])
    assert np.isfinite(lnZ), "ln Z is not finite: %r" % lnZ
    # Window measured, not guessed: 6+ unseeded runs of this configuration on ldas-grid (numpy
    # backend) gave 9.88-9.99, spread 0.10 nats, sd ~0.036, against a closed-form 9.98 for the
    # separable Gaussian above with this driver's prior==1 convention.  +/-2 nats is ~55 sigma of
    # margin, wide enough for a different host or backend and still narrow enough to constrain
    # the integral -- the old +/-5 admitted an evidence wrong by a factor of 150, and in
    # particular admitted the ln(prior volume) offset that adaptive_cartesian_gpu exhibits.
    assert 8.0 < lnZ < 12.0, "ln Z = %r, against a closed-form 9.98 for this input" % lnZ

    samples = os.path.join(wd, "output-EOS-samples.dat")
    assert os.path.exists(samples), "no posterior samples written"
    # NOT `len(np.atleast_2d(...)) > 0`: loadtxt on an EMPTY file returns shape (0,), atleast_2d
    # makes that (1, 0), and len() is 1 -- an assertion that cannot fail.  Check .size, and check
    # the row count actually honours --n-output-samples.
    rows = np.atleast_2d(np.loadtxt(samples))
    assert rows.size > 0, "posterior sample file is empty"
    assert rows.shape[0] == 20, \
        "asked for 20 posterior samples, got %d" % rows.shape[0]
    assert np.isfinite(rows).all(), "posterior samples contain nan/inf"


def test_portfolio_args_reach_members(tmp_path):
    """--sampler-portfolio-args must actually arrive at the member's setup().

    Two ways this silently does nothing, both covered here:
      * setup() is never called at all (the bug above); or
      * setup() is called with a MISSPELT keyword.  setup() takes **kwargs, so a wrong name is
        accepted and ignored rather than raising -- mcsamplerPortfolio.setup() reads
        kwargs['portfolio_args'] exactly.  To check whether a given caller has this right, grep
        it for the kwarg it passes and compare against that dict key; do not trust this comment
        for any file but the one it sits next to.

    A GMM member also makes this the case where the portfolio branch rebinds
    opts.sampler_method to 'GMM', so a setup() guarded on --sampler-method would not fire.
    """
    proc = _run(tmp_path, ["--sampler-method", "portfolio",
                           "--sampler-portfolio", "GMM",
                           "--sampler-portfolio-args", "{'n_comp':7}"])
    assert proc.returncode == 0, \
        "portfolio+GMM exited %d:\n%s" % (proc.returncode, proc.stdout[-3000:])
    member_lines = [ln for ln in proc.stdout.splitlines() if "PORTFOLIO setup" in ln]
    assert member_lines, \
        "member setup() never ran -- portfolio members are cold:\n" + proc.stdout[-3000:]
    assert any("'n_comp': 7" in ln for ln in member_lines), \
        "--sampler-portfolio-args never reached the member; setup() saw:\n" \
        + "\n".join(member_lines)


def test_portfolio_without_members_is_refused(tmp_path):
    """--sampler-method portfolio with no --sampler-portfolio used to build an EMPTY portfolio.

    Once setup() is wired up that no longer stops at the breakpoints TypeError: it survives
    construction and setup() and dies deep inside draw() with "index -1 is out of bounds for
    axis 0 with size 0", after a divide-by-zero warning on the member weights.  Refuse up front.
    """
    proc = _run(tmp_path, ["--sampler-method", "portfolio"])
    assert proc.returncode == 99, \
        "expected a clean option-mismatch exit, got %d:\n%s" % (proc.returncode,
                                                                proc.stdout[-3000:])
    assert "OPTION MISMATCH" in proc.stdout, proc.stdout[-2000:]


def test_unrecognized_member_is_skipped_not_aliased(tmp_path):
    """An unrecognized --sampler-portfolio name must be DROPPED, not turned into a duplicate.

    The construction loop reuses one `sampler` variable and only skips on `sampler is None`, so
    before the reset at the top of the loop an unknown name that FOLLOWED a good one re-appended
    the previous member -- the identical object, twice, sharing all its adaptation state.  It is
    silent: the portfolio looks like it has the requested number of members.

    Asserted in both directions on one run, because a reset placed wrongly would instead drop
    members that ARE recognized: 'bogus' must contribute nothing, and AV and GMM must both
    survive as DISTINCT objects.
    """
    proc = _run(tmp_path, ["--sampler-method", "portfolio",
                           "--sampler-portfolio", "AV",
                           "--sampler-portfolio", "bogus",
                           "--sampler-portfolio", "GMM"])
    assert proc.returncode == 0, \
        "exited %d:\n%s" % (proc.returncode, proc.stdout[-3000:])
    # setup() prints one line per MEMBER, including the object's repr -- so this sees aliasing,
    # which a count of "PORTFOLIO: adding" lines would not.
    member_lines = [ln for ln in proc.stdout.splitlines() if "PORTFOLIO setup" in ln]
    members = _member_ids(member_lines)
    assert len(members) == 2, \
        "expected exactly 2 portfolio members (AV, GMM); 'bogus' must add none. Got %d:\n%s" \
        % (len(members), "\n".join(members))
    assert len(set(members)) == 2, \
        "the two portfolio members are THE SAME OBJECT -- an unknown name was aliased onto the " \
        "previous member:\n" + "\n".join(members)
    # ...and the user must be TOLD.  The two OPTION MISMATCH guards only fire when NOTHING
    # matched, so a single typo among several good names would otherwise shrink the portfolio
    # silently -- the likeliest mistake, and the quietest.
    assert any("ignoring unrecognized --sampler-portfolio" in ln and "bogus" in ln
               for ln in proc.stdout.splitlines()), \
        "the dropped name was not reported:\n" + proc.stdout[-3000:]
    assert any("mcsamplerAdaptiveVolume" in ln for ln in proc.stdout.splitlines()
               if "PORTFOLIO setup" in ln), "the AV member was dropped"
    assert any("mcsamplerEnsemble" in ln for ln in proc.stdout.splitlines()
               if "PORTFOLIO setup" in ln), "the GMM member was dropped"


def test_portfolio_forces_internal_use_lnL(tmp_path):
    """A portfolio must run WITHOUT --internal-use-lnL on the command line.

    mcsamplerPortfolio.integrate() raises "must integrate lnL" unless use_lnL is set, and the
    integrand would be the linear likelihood while integrate_log() treats what it is handed as a
    log.  The driver therefore forces opts.internal_use_lnL for a portfolio, the same way the AV
    branch does.  Nothing is masked by this: on the unfixed base there was no configuration in
    which a portfolio produced a number with the flag unset -- it always raised.
    """
    proc = _run(tmp_path, ["--sampler-method", "portfolio", "--sampler-portfolio", "AV"])
    assert "must integrate lnL" not in proc.stdout, \
        "--internal-use-lnL is not being forced for the portfolio:\n" + proc.stdout[-3000:]
    assert proc.returncode == 0, \
        "portfolio without --internal-use-lnL exited %d:\n%s" % (proc.returncode,
                                                                 proc.stdout[-3000:])


def test_portfolio_args_track_names_not_survivors(tmp_path):
    """--sampler-portfolio-args is counted against the NAMES given, and realigned after drops.

    A member can fail to build for reasons that are not the user's fault: the NFlow branch
    `continue`s when nflows is not installed.  Counting the args against the SURVIVING members
    would then reject a portfolio spec that is perfectly well formed -- the same spec would run
    on a host with nflows and exit 99 on a host without it, blaming the argument count for a
    missing optional dependency.

    An unrecognized name stands in for the dropped member here so the test does not itself depend
    on whether nflows is installed.  The surviving GMM member must receive the SECOND args entry,
    which is what proves the realignment indexes by name rather than by position-after-drop.
    """
    proc = _run(tmp_path, ["--sampler-method", "portfolio",
                           "--sampler-portfolio", "bogus",
                           "--sampler-portfolio", "GMM",
                           "--sampler-portfolio-args", "{'n_comp':3}",
                           "--sampler-portfolio-args", "{'n_comp':7}"])
    assert proc.returncode == 0, \
        "a dropped member turned a well-formed spec into an abort:\n" + proc.stdout[-3000:]
    members = [ln for ln in proc.stdout.splitlines() if "PORTFOLIO setup" in ln]
    assert len(members) == 1, "expected only the GMM member:\n" + "\n".join(members)
    assert "'n_comp': 7" in members[0], \
        "the surviving member got the dropped member's arguments:\n" + members[0]
    assert "'n_comp': 3" not in members[0], \
        "args were realigned by position after the drop, not by name:\n" + members[0]


def test_portfolio_args_count_mismatch_is_refused(tmp_path):
    """A genuine user count error must still be refused, loudly.

    mcsamplerPortfolio.setup() only prints "PORTFOLIO - format ERROR" and silently discards ALL
    of them when the count does not match, so the run would otherwise continue with every member
    untuned.
    """
    proc = _run(tmp_path, ["--sampler-method", "portfolio",
                           "--sampler-portfolio", "AV",
                           "--sampler-portfolio-args", "{}",
                           "--sampler-portfolio-args", "{}"])
    assert proc.returncode == 99, \
        "expected a clean option-mismatch exit, got %d:\n%s" % (proc.returncode,
                                                                proc.stdout[-3000:])
    assert "OPTION MISMATCH" in proc.stdout, proc.stdout[-2000:]


def test_portfolio_args_must_be_dicts(tmp_path):
    """A non-dict --sampler-portfolio-args entry must be refused before it reaches setup().

    mcsamplerPortfolio.setup() does `args_here.update(portfolio_extra_args[indx])`, so a list
    entry dies mid-run inside the integrator with
    "TypeError: cannot convert dictionary update sequence element #0 to a sequence" -- a message
    that names neither the option nor the driver.
    """
    proc = _run(tmp_path, ["--sampler-method", "portfolio",
                           "--sampler-portfolio", "AV",
                           "--sampler-portfolio-args", "[1,2]"])
    assert "cannot convert dictionary update sequence" not in proc.stdout, \
        "a malformed args entry reached setup():\n" + proc.stdout[-3000:]
    assert proc.returncode == 99, \
        "expected a clean option-mismatch exit, got %d:\n%s" % (proc.returncode,
                                                               proc.stdout[-3000:])


def test_portfolio_with_no_recognized_member_is_refused(tmp_path):
    """Distinct from the no-members case: names were GIVEN, none matched.

    The likeliest way in is the comma-joined spelling the old help text invited --
    `--sampler-portfolio AV,GMM` is one string that matches no sampler -- which built an EMPTY
    portfolio and died inside draw() with "index -1 is out of bounds for axis 0 with size 0".
    """
    proc = _run(tmp_path, ["--sampler-method", "portfolio", "--sampler-portfolio", "AV,GMM"])
    assert proc.returncode == 99, \
        "expected a clean option-mismatch exit, got %d:\n%s" % (proc.returncode,
                                                               proc.stdout[-3000:])
    assert "matched no known sampler" in proc.stdout, proc.stdout[-2000:]


def test_implemented_fit_method_gp_is_not_refused(tmp_path):
    """The other half of the fit-method contract: 'gp' must still be ACCEPTED.

    Every other test here uses 'rf', so a guard that grew to reject a legitimate method would be
    invisible to them -- refusing valid work is the mirror-image defect of accepting invalid
    work, and exit 99 is just as fatal to a DAG node either way.  ~17 s, the slowest case in this
    file; it is here because 'gp' is otherwise exercised by nothing.
    """
    proc = _run(tmp_path, ["--fit-method", "gp"])
    assert "OPTION MISMATCH" not in proc.stdout, \
        "a supported fit method was refused:\n" + proc.stdout[-3000:]
    assert proc.returncode == 0, \
        "--fit-method gp exited %d:\n%s" % (proc.returncode, proc.stdout[-3000:])


def test_unimplemented_fit_method_is_refused(tmp_path):
    """This driver builds only 'gp' and 'rf'.  Anything else must be refused UP FRONT.

    The --fit-method help was copied from the intrinsic driver, which implements a dozen more.
    Asking for one of those here left `my_fit` at None and the run carried on: nothing referenced
    it until the sampler evaluated the integrand, which died with "TypeError: 'NoneType' object is
    not callable" from inside log_likelihood_function -- after the full setup had been paid for,
    and with nothing in the message naming --fit-method.
    """
    proc = _run(tmp_path, ["--fit-method", "quadratic"])
    assert "TypeError: 'NoneType' object is not callable" not in proc.stdout, \
        "an unbuilt fit reached the integrand:\n" + proc.stdout[-3000:]
    assert proc.returncode == 99, \
        "expected a clean option-mismatch exit, got %d:\n%s" % (proc.returncode,
                                                                proc.stdout[-3000:])
    assert "--fit-method" in proc.stdout, \
        "the message must name the option at fault:\n" + proc.stdout[-2000:]
