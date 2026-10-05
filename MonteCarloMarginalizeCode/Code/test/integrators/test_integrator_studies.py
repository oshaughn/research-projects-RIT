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
behind RIFT_RUN_EXPENSIVE.  MEASURED, and the premise was wrong: on CIT (ldas-grid) with the IGWN
python (OMP_NUM_THREADS=1) they take 7, 4, 20, 64 and 6 seconds -- 101 s for all five.  The decoy
study is the expensive one only because it replicates itself in nine child processes; that is still
not enough to be opt-in.

FLAKE RISK, since these are Monte Carlo studies with tolerance-based gates: all five seed
explicitly, through RIFT.integrators.seeding.seed_everything, which reaches cupy as well as numpy
(a bare numpy.random.seed is inert on the GPU backend the samplers draw through).  But a seed does
not make a low-effective-sample-size result reliable across platforms, and it does not even fix the
result on ONE platform: scipy's mvnun, which normalizes each GMM component inside gmm.score,
carries an RNG that NEITHER seeding path can reach, so repeating an arm inside one interpreter
changes its answer.  The decoy balance-heuristic study therefore checks the mean log bias over
independent, process-isolated runs against a fixed threshold in nats.  If a study proves marginal
in CI, investigate its sampling and statistical contract; do not delete the gate and do not widen
its threshold.

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
    ("test_AV_bootstrap.py", 7),
    ("test_AV_warmstart_safety.py", 4),
    ("test_portfolio_adaptive_alloc.py", 20),
    ("test_portfolio_balance_heuristic.py", 95),
    ("test_portfolio_oracle.py", 6),
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
