"""Run the five integrator quantitative studies as real subprocesses and require exit 0.

WHY A WRAPPER, AND WHY IN ORDINARY CI.  These five scripts carry the only assertions anyone has
written about AV warm-starting and portfolio allocation -- a 4-sigma bias gate, an anti-bias
ordering under a mis-placed proposal, a draw-allocation comparison against standalone AV, safety
under a decoy member, and an oracle finding a needle.  Each ends in `raise SystemExit(1)` on
failure UNDER `--as-test`, so the pass/fail signal is real and machine-readable.  That flag is not
optional here: every one of these scripts keeps its scientific comparisons and its `SystemExit(1)`
behind `if args.as_test`, and without it a biased or otherwise invalid result still prints and exits
0 -- the wrapper would then detect only crashes, not the behaviour it claims to gate.  None of them
ran in CI at all before this: they have
a __main__ and argparse and no test functions, so pytest collects ZERO items and exits 5 -- "no
tests ran", which reads as a pass -- and .travis/ci_roster.txt carried them as HANDRUN.

That roster entry called them "expensive", which is why the suggested fix was an opt-in wrapper
behind RIFT_RUN_EXPENSIVE.  MEASURED, and the premise was wrong: on CIT with the IGWN python
(OMP_NUM_THREADS=1) they take 5, 2, 14, 4 and 4 seconds -- 29 s for all five.  Nothing here needs
to be opt-in.

FLAKE RISK, since these are Monte Carlo studies with tolerance-based gates: all five seed
explicitly (numpy RandomState(0/1/3) and np.random.seed), so they are deterministic rather than
merely lucky, and three consecutive runs of each exited 0.  Three runs is not a flake proof; if
one does prove marginal in CI, tighten ITS seed or widen ITS stated tolerance, and do not
delete the gate.

Subprocess rather than import: each is a __main__ script with argparse, and running it the way a
human runs it is the point -- it is what keeps the wrapper honest about the entry point.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, "..", ".."))

# name -> measured wall seconds on CIT, for whoever wonders what this costs
STUDIES = [
    ("test_AV_bootstrap.py", 5),
    ("test_AV_warmstart_safety.py", 2),
    ("test_portfolio_adaptive_alloc.py", 14),
    ("test_portfolio_balance_heuristic.py", 4),
    ("test_portfolio_oracle.py", 4),
]


@pytest.mark.parametrize("script,_secs", STUDIES)
def test_study_exits_clean(script, _secs):
    path = os.path.join(HERE, script)
    assert os.path.exists(path), (
        "%s is gone. It carried the only assertions on this behaviour; restore it or remove "
        "this entry deliberately." % script)
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    # --as-test is what turns each study from a printout into a gate; see module docstring.
    pr = subprocess.run([sys.executable, path, "--as-test"], env=env, timeout=900,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = pr.stdout.decode("utf-8", "replace")
    assert pr.returncode == 0, "%s --as-test exited %d; its own gate failed.\n%s" % (
        script, pr.returncode, out[-3000:])
