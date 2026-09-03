#!/usr/bin/env python3
"""Repo-wide census: every test file must be reachable from CI, or rostered with a reason.

WHY THIS EXISTS.  CI job membership in this repo is per-file and hand-listed: ci.yml and
.gitlab-ci.yml name individual files, .travis/*.sh name individual files, and three
directories are passed to pytest whole.  Two mechanisms already defend a SINGLE job's list
against silent loss -- .travis/test-slowrot.sh (an explicit FILES manifest, issue #169) and
.travis/test-q-window-stencil.sh (a marker line inside each test file, PR #242).  Both are
SCOPED: slowrot's manifest covers test_slowrot_*.py, and q-window's SCOPE_GLOBS cover eight
filename patterns.  Neither can answer the question one level up -- "is this file registered
with ANY job?" -- and on 2026-09-03 the answer for 86 of 200 test files was no.

That gap is not the same defect as a conflicted job list, and it does not have the same fix.
Most of those 86 files should NOT be gated: they are hand-run studies, plotting demos, and
scripts that import pre-package flat module names (factored_likelihood, lalsimutils, ourio)
which have not existed since RIFT was packaged, so they cannot even be IMPORTED.  A
membership marker has nowhere to record that.  What was missing is a place to record a
DECISION, and a check that fails when a file has none.

So this script does not gate anything.  It asserts that every test file under Code/ is either
reachable from CI configuration, or carries a roster entry stating why it is not.  A new test
file added to a directory no job runs now fails the build instead of sitting unrun.

Reachability is deliberately GENEROUS -- it counts a file as covered if its basename stem
appears anywhere in a CI config (so `python -m RIFT.calmarg.test_selfterm_basis` counts), if
it lives under a directory passed to pytest, or if it carries a RIFT-CI-GATE marker line.  A
generous reachability test makes this check UNDER-report, never over-report: it can miss that
a file is unrun, but it cannot falsely accuse a registered one.  The roster is where the
narrower truth is written down.

Stdlib only -- no numpy, no RIFT import -- so it can run as its own cheap job.
"""

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODEDIR = os.path.join("MonteCarloMarginalizeCode", "Code")
ROSTER = os.path.join(".travis", "ci_roster.txt")

# Directories handed to pytest as a whole, so every file beneath them runs.  Kept explicit
# rather than parsed: a wrong guess here would silently EXCUSE files, which is the failure
# mode this script exists to prevent.  Each must still be a real directory (checked below).
DIR_TARGETS = (
    CODEDIR + "/RIFT/simulation_manager/tests",   # .travis/test-simulation-manager.sh
    CODEDIR + "/test/asimov_integration",         # .travis/test-asimov.sh
    CODEDIR + "/test/jax",                        # .travis/test-jax.sh
)

# Membership markers of the PR #242 kind.  Any file carrying one is registered with that job
# by the job's own script; this census must not then demand a roster entry for it.
MARKER_RE = re.compile(r"^# RIFT-CI-GATE: [a-z0-9-]+$", re.M)

VALID_STATUS = {
    # not gated, and that is the right answer
    "HANDRUN":   "hand-run study or demo; not a pytest target",
    "LEGACY":    "imports pre-package flat modules; cannot be imported at all",
    "OPTDEP":    "needs a dependency CI does not install",
    "GPU":       "needs a GPU; CI runners have none",
    "EXPENSIVE": "opt-in behind an env var by design",
    # not gated, and that is NOT the right answer -- these are debts, stated as such
    "BROKEN":    "collects but fails; needs a fix before it can be gated",
    # tolerated in either state while a companion PR is in flight
    "PENDING":   "registration is in flight in another PR",
}


def _cfg_blob():
    parts = []
    for d, pats in ((".github/workflows", (".yml", ".yaml")),
                    (".travis", (".sh", ".py"))):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(pats):
                parts.append(open(os.path.join(d, f), errors="replace").read())
    for f in (".gitlab-ci.yml", ".travis.yml"):
        if os.path.exists(f):
            parts.append(open(f, errors="replace").read())
    return "\n".join(parts)


def _test_files():
    out = []
    for root, dirs, files in os.walk(CODEDIR):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for f in files:
            if f.endswith(".py") and (f.startswith("test_") or f.endswith("_test.py")):
                out.append(os.path.join(root, f))
    return sorted(out)


def _read_roster():
    """path -> (status, reason).  Duplicate paths are an error: two verdicts, one file."""
    entries, errs = {}, []
    if not os.path.exists(ROSTER):
        return entries, ["%s does not exist" % ROSTER]
    for n, raw in enumerate(open(ROSTER, errors="replace"), 1):
        line = raw.split("#", 1)[0].strip() if raw.lstrip().startswith("#") else raw.rstrip("\n")
        if not line.strip() or raw.lstrip().startswith("#"):
            continue
        bits = line.split(None, 2)
        if len(bits) < 3:
            errs.append("%s:%d: need '<path> <STATUS> <reason>', got %r" % (ROSTER, n, raw.strip()))
            continue
        path, status, reason = bits[0], bits[1], bits[2].strip()
        if status not in VALID_STATUS:
            errs.append("%s:%d: unknown status %r (valid: %s)"
                        % (ROSTER, n, status, ", ".join(sorted(VALID_STATUS))))
        if len(reason) < 12:
            errs.append("%s:%d: reason for %s is too short to be a reason: %r"
                        % (ROSTER, n, path, reason))
        if path in entries:
            errs.append("%s:%d: %s is listed twice" % (ROSTER, n, path))
        entries[path] = (status, reason)
    return entries, errs


def main():
    os.chdir(REPO)
    errs = []

    for d in DIR_TARGETS:
        if not os.path.isdir(d):
            errs.append("DIR_TARGETS names %s, which is not a directory. It was renamed or "
                        "removed; left as is it silently EXCUSES files from this census." % d)

    blob = _cfg_blob()
    files = _test_files()
    if not files:
        print("test-ci-roster: found no test files under %s -- the walk is broken, not the "
              "repo. This is a hard failure, not an empty run." % CODEDIR, file=sys.stderr)
        return 1

    roster, rerrs = _read_roster()
    errs.extend(rerrs)

    reachable = {}
    for f in files:
        stem = os.path.basename(f)[:-3]
        why = None
        if any(f.startswith(d + "/") for d in DIR_TARGETS):
            why = "directory target"
        elif re.search(r"(?<![\w])" + re.escape(stem) + r"(?![\w])", blob):
            why = "named in CI config"
        elif MARKER_RE.search(open(f, errors="replace").read()):
            why = "carries a RIFT-CI-GATE marker"
        reachable[f] = why

    # 1. Every unreachable file needs a verdict.
    for f in files:
        if reachable[f] is None and f not in roster:
            errs.append(
                "%s is reachable from no CI job and has no entry in %s.\n"
                "    An unlisted test never runs here and the job stays green forever.\n"
                "    Either register it with a job, or add a line to the roster saying why not."
                % (f, ROSTER))

    # 2. A roster entry for a file that IS now reachable is stale -- it records a decision that
    #    has been overtaken, and leaving it invites the next reader to trust it.  PENDING is
    #    exempt on purpose: it marks a registration in flight in another PR, so it must be
    #    legal both before and after that PR lands, in either merge order.
    for f, (status, _reason) in sorted(roster.items()):
        if f not in reachable:
            errs.append("%s: %s no longer exists. A roster entry for a deleted file is a "
                        "silent no-op; drop the line." % (ROSTER, f))
        elif reachable[f] is not None and status != "PENDING":
            errs.append("%s: %s is listed as %s but IS now reachable (%s). The entry is stale "
                        "-- delete it." % (ROSTER, f, status, reachable[f]))

    n_reach = sum(1 for v in reachable.values() if v is not None)
    print("test-ci-roster: %d test files under %s" % (len(files), CODEDIR))
    print("  reachable from CI : %d" % n_reach)
    print("  rostered           : %d" % len(roster))
    counts = {}
    for status, _ in roster.values():
        counts[status] = counts.get(status, 0) + 1
    for s in sorted(counts):
        # An unknown status is already an error above; do not crash the summary on it, or the
        # specific message gets buried under a traceback.
        print("      %-10s %3d   (%s)" % (s, counts[s], VALID_STATUS.get(s, "UNKNOWN STATUS")))

    if errs:
        print("\ntest-ci-roster: FAIL", file=sys.stderr)
        for e in errs:
            print("  " + e, file=sys.stderr)
        return 1
    print("test-ci-roster: PASS -- every test file is gated or has a stated reason it is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
