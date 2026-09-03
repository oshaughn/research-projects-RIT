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

REACHABILITY, AND WHAT EACH SIGNAL CAN AND CANNOT DO.  A file counts as covered by one of
three signals, and they are not equally trustworthy.  An earlier version of this docstring
claimed all three could only UNDER-report -- miss that a file is unrun, never falsely clear a
file.  That was wrong for two of them, and both were live bugs:

  * DIRECTORY TARGET -- the file lives under a directory handed to pytest whole.  Exact.
  * NAMED IN A CI CONFIG -- the basename stem appears in a non-comment line, so a module
    invocation (`python -m RIFT.calmarg.test_selfterm_basis`) counts as well as a path.  This
    is textual and therefore loose in one direction: it cannot distinguish an invocation from
    a mention.  Comment-only lines are stripped for exactly that reason (see _cfg_blob) --
    before that, ci.yml's comment explaining why the two cupy parity files are EXCLUDED was
    enough to mark them covered.  It can still be fooled by a filename appearing in a
    non-comment context that does not run it; that residue is accepted and stated.
  * RIFT-CI-GATE MARKER -- the file asserts its own membership.  This one is the file
    claiming coverage for itself, so the claim is checked against KNOWN_GATES rather than
    taken on its shape.  Unvalidated, `# RIFT-CI-GATE: totally-made-up-job` cleared a file
    that no script greps for and nothing runs.

The roster is where the narrower truth is written down.

Stdlib only -- no numpy, no RIFT import -- so it can run as its own cheap job.
"""

import os
import re
import subprocess
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

# Membership markers of the PR #242 kind.  A file carrying one is registered with that job by
# the job's own script, so the census must not then demand a roster entry for it.
#
# BUT THE NAME MUST BE VALIDATED, or this branch is fail-open.  The marker is the only one of
# the three reachability signals that a test file can assert about ITSELF, and the pattern
# below matches shape, not meaning: `# RIFT-CI-GATE: totally-made-up-job` is syntactically
# perfect and no script anywhere greps for it.  Honouring it would count the file as covered
# while nothing runs it -- the inert-guard failure this whole census exists to catch,
# reintroduced inside the census.  It is not hypothetical: before KNOWN_GATES landed, planting
# that exact line on one of the LEGACY files below -- a file that cannot even be IMPORTED,
# because it wants the pre-package `factored_likelihood` -- moved it into the "reachable from
# CI" count and made its roster entry report as STALE.
#
# So a marker counts only if its gate name appears in KNOWN_GATES below, and only if that
# gate's script really greps for the exact literal.  KNOWN_GATES is a shared line, which is
# what PR #242 set out to remove -- but it is one line per JOB, not per test, edited when a
# gate is created rather than when a test is added, so it does not carry the conflict cost
# that motivated the marker in the first place.
MARKER_RE = re.compile(r"^# RIFT-CI-GATE: ([a-z0-9-]+)$", re.M)

# gate name -> the script that discovers files by that marker.  Both directions are checked:
# a name here whose script exists but does not contain the literal is a broken registry, and a
# marker naming anything NOT here is a hard error rather than silent coverage.
#
# A gate whose script does not exist YET is simply not live: markers naming it are not
# honoured, and the files carrying them need roster entries until it lands.  That is what keeps
# this independent of PR #242's merge order -- on rift_O4d today no file carries any marker.
KNOWN_GATES = {
    "q-window-stencil": ".travis/test-q-window-stencil.sh",   # PR #242
}


# PROVING CONSUMPTION, RATHER THAN INFERRING IT.
#
# A gate is honoured only if its script can be ASKED which files it will run.  Set this env var
# and the script must print those paths, one per line, and exit 0:
#
#     if [ -n "${RIFT_CI_GATE_LIST:-}" ]; then printf '%s\n' "${FILES[@]}"; exit 0; fi
#
# PLACEMENT IS PART OF THE CONTRACT, and getting it wrong is not subtle -- it just fails.  The
# discovery and this short-circuit must come BEFORE the script's pytest/numpy/lal probes, which
# means moving the marker/CODEDIR assignment and the discovery line above them.  This census job
# deliberately has no `needs: install`: it is stdlib-only and must stay that way, so a list mode
# sitting behind a numpy probe cannot run here.  Discovery needs only grep; nothing else in the
# gate's preamble is required to answer "which files".
#
# WHY NOTHING WEAKER WILL DO.  Two textual checks were tried here and both were fail-open.
# Requiring the marker literal on a non-comment line passes an ORPHANED `MARKER=` assignment
# whose discovery grep has been deleted.  Adding "and a line matching grep.*MARKER" does not fix
# it: test-q-window-stencil.sh contains `grep -qxF -- "${MARKER}" "${e}"` for a completely
# different purpose -- asserting that an EXCLUDED file does NOT carry the marker -- so deleting
# the real discovery line still left the pattern satisfied.  Checked; it passed green.  No
# regex distinguishes "uses the marker to find files" from "uses the marker"; only running the
# discovery does.
#
# The cost is a small requirement on any gate that wants marker-based membership, and it is
# reported precisely: a marker naming a gate that cannot list is an error against THAT GATE,
# quoting the snippet.  It is not raised speculatively -- a registered gate with no marked files
# costs nothing.
LIST_ENV = "RIFT_CI_GATE_LIST"

LIST_SNIPPET = ('if [ -n "${%s:-}" ]; then printf \'%%s\\n\' "${FILES[@]}"; exit 0; fi'
                % LIST_ENV)


def _gate_listing(script):
    """Ask a gate script which files it would run.  (files, error); files is None if unsupported."""
    if LIST_ENV not in _strip_comment_lines(open(script, errors="replace").read()):
        return None, None
    env = dict(os.environ)
    env[LIST_ENV] = "1"
    try:
        pr = subprocess.run(["bash", script], env=env, timeout=120,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "%s declares %s but could not be run in list mode: %s" % (script, LIST_ENV, exc)
    if pr.returncode != 0:
        return None, ("%s declares %s but exited %d in list mode.\n"
                      "    A list mode that cannot run is worse than none: it looks authoritative.\n"
                      "    stderr: %s"
                      % (script, LIST_ENV, pr.returncode,
                         pr.stderr.decode("utf-8", "replace").strip()[:300]))
    found = [l.strip() for l in pr.stdout.decode("utf-8", "replace").splitlines() if l.strip()]
    if not found:
        return None, ("%s listed no files in list mode.  Its discovery returns nothing, so every "
                      "file carrying its marker is covered by an empty run." % script)
    return found, None


def _live_gates(live_cfg):
    """Gate name -> the set of files that gate's own discovery returns.

    Three conditions, each of which was a hole before it was checked:

      * the gate's script is INVOKED from a workflow entry point.  A script on disk that no job
        runs is not a gate, and honouring its markers repeats the DIR_TARGETS bug.
      * it declares the marker literal outside its comments.  (test-q-window-stencil.sh quotes
        its own marker in the comment block explaining the mechanism, so a bare
        `literal in text` passed with the live MARKER= renamed away.)
      * it can LIST what it will run, and that listing is what the census believes.
    """
    gates, errs = {}, []
    live_paths = set(live_cfg)
    for name, script in sorted(KNOWN_GATES.items()):
        literal = "# RIFT-CI-GATE: %s" % name
        if not os.path.exists(script):
            # Not an error: the gate has not landed (or was retired).  Just not live.
            continue
        if script not in live_paths:
            errs.append("KNOWN_GATES maps %r to %s, which exists but is invoked by no CI job.\n"
                        "    Files carrying that marker would be counted as covered by a script "
                        "nothing runs.\n"
                        "    Restore its job, or drop the registry entry." % (name, script))
            continue
        if literal not in _strip_comment_lines(open(script, errors="replace").read()):
            errs.append("KNOWN_GATES maps %r to %s, but that script does not contain the "
                        "literal %r outside its comments.\n"
                        "    Every file carrying that marker is then counted as covered by a "
                        "gate that never looks for it.\n"
                        "    Fix the script, or drop the registry entry." % (name, script, literal))
            continue
        listing, lerr = _gate_listing(script)
        if lerr:
            errs.append(lerr)
            continue
        if listing is None:
            # Not honoured, and not an error on its own -- only files that actually carry this
            # marker are affected, and they are told below, individually.
            continue
        gates[name] = set(os.path.normpath(x) for x in listing)
    return gates, errs

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
    "PENDING":   "waiting on a named gate that is not live yet; expires when it lands",
}


SELF = os.path.basename(os.path.abspath(__file__))


def _strip_comment_lines(text):
    """Drop lines whose first non-whitespace character is '#'.

    Used twice, and both uses closed a fail-open hole where prose about a name was accepted as
    a use of that name.  A comment is where a CI file EXPLAINS what it does not run, so it is
    precisely the place a name appears without being invoked.
    """
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


# Workflow ENTRY POINTS.  Everything else counts only if something reachable from one of these
# invokes it.  A .travis script that no job runs is not CI, it is a file.
CFG_ROOTS = (".github/workflows", ".gitlab-ci.yml", ".travis.yml")


def _live_configs():
    """CI files reachable from a workflow entry point, by transitive invocation.

    WHY NOT just "every file in .travis/".  Coverage has to depend on a job actually running
    something, not on a file existing.  Deleting the asimov-integration job from ci.yml and
    .gitlab-ci.yml left .travis/test-asimov.sh on disk, still naming test/asimov_integration/ --
    and the census went on reporting those three files as covered, green, with nothing running
    them.  Checked before this closure landed; that is the P2 this answers.

    A script joins the live set when its basename appears on a non-comment line of something
    already live, so `bash .travis/test-asimov.sh` pulls it in and a deleted job drops it out.
    Iterated to a fixed point, because scripts can invoke scripts.
    """
    live = {}
    for root in CFG_ROOTS:
        if os.path.isdir(root):
            for f in sorted(os.listdir(root)):
                if f.endswith((".yml", ".yaml")):
                    live[os.path.join(root, f)] = open(os.path.join(root, f), errors="replace").read()
        elif os.path.exists(root):
            live[root] = open(root, errors="replace").read()

    candidates = {}
    if os.path.isdir(".travis"):
        for f in sorted(os.listdir(".travis")):
            if f.endswith((".sh", ".py")) and f != SELF:
                candidates[os.path.join(".travis", f)] = f

    changed = True
    while changed:
        changed = False
        blob = "\n".join(_strip_comment_lines(t) for t in live.values())
        for path, base in sorted(candidates.items()):
            if path in live:
                continue
            # NOTE the lookbehind allows '/': every invocation is a PATH, `bash
            # .travis/test-asimov.sh`, so excluding '/' matched nothing and emptied the live set.
            if re.search(r"(?<![\w-])" + re.escape(base) + r"(?![\w])", blob):
                live[path] = open(path, errors="replace").read()
                changed = True
    return live


def _cfg_blob():
    """CI configuration with comment-only lines stripped.

    TWO EXCLUSIONS, both of which were live over-reporting bugs, not precautions.

    COMMENT LINES.  A filename mentioned in a comment is not a reference.  ci.yml's
    q-window-stencil-check has a long comment naming the two cupy parity files to explain why
    they are OUT, and that mention alone was enough to count them as covered.  Every genuine
    reference lives in a `run:` block, a pytest argument, or a FILES array -- never on a line
    whose first non-whitespace character is '#' -- so stripping those loses nothing real.

    THIS SCRIPT.  It runs no test, and it discusses test files by name in its own comments.
    Including it let a filename typed into an explanatory comment here mark that file covered
    -- which is exactly what happened while the KNOWN_GATES check above was being written.
    """
    return "\n".join(_strip_comment_lines(t) for t in _live_configs().values())


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

    live_cfg = _live_configs()
    blob = "\n".join(_strip_comment_lines(t) for t in live_cfg.values())

    # A directory target excuses every file beneath it, so BOTH halves must hold: the directory
    # exists, AND some live CI file still hands it to pytest.  Checking only the first meant the
    # asimov-integration job could be deleted outright while its three files kept reporting as
    # covered -- existence of a directory is not evidence that anything runs it.
    live_dirs = set()
    for d in DIR_TARGETS:
        if not os.path.isdir(d):
            errs.append("DIR_TARGETS names %s, which is not a directory. It was renamed or "
                        "removed; left as is it silently EXCUSES files from this census." % d)
            continue
        if not re.search(r"(?<![\w-])" + re.escape(d) + r"(?![\w])", blob):
            errs.append("DIR_TARGETS names %s, but no CI file reachable from a workflow entry "
                        "point passes it to pytest any more.\n"
                        "    Its job or the invocation of its script was removed, so nothing "
                        "runs those files -- yet listing the directory here would go on "
                        "EXCUSING every one of them.\n"
                        "    Restore the invocation, or drop this entry and roster the files."
                        % d)
            continue
        live_dirs.add(d)
    files = _test_files()
    if not files:
        print("test-ci-roster: found no test files under %s -- the walk is broken, not the "
              "repo. This is a hard failure, not an empty run." % CODEDIR, file=sys.stderr)
        return 1

    roster, rerrs = _read_roster()
    errs.extend(rerrs)

    gates, gate_errs = _live_gates(live_cfg)
    errs.extend(gate_errs)

    reachable = {}
    for f in files:
        stem = os.path.basename(f)[:-3]
        why = None
        if any(f.startswith(d + "/") for d in live_dirs):
            why = "directory target"
        elif re.search(r"(?<![\w])" + re.escape(stem) + r"(?![\w])", blob):
            why = "named in CI config"
        else:
            for m in MARKER_RE.finditer(open(f, errors="replace").read()):
                name = m.group(1)
                if name in gates:
                    # The gate's OWN discovery decides.  A file may carry a valid marker and
                    # still not be found -- wrong directory, a narrowed --include -- and then
                    # it is unrun no matter what it says about itself.
                    if os.path.normpath(f) not in gates[name]:
                        errs.append(
                            "%s carries the %s marker, but that gate's own discovery (%s=1) "
                            "does not return it.\n"
                            "    The marker reads as coverage and the gate never runs the file. "
                            "Check the gate's search scope, or roster this file."
                            % (f, name, LIST_ENV))
                        break
                    why = "carries the %s marker" % name
                    break
                # A marker no script reads is WORSE than no marker: it reads as coverage to
                # everyone who opens the file.  Fail on it by name rather than quietly treating
                # the file as unregistered, or the diagnosis becomes "add a roster entry" when
                # the real answer is "fix the marker".
                if name in KNOWN_GATES:
                    script = KNOWN_GATES[name]
                    if not os.path.exists(script):
                        errs.append(
                            "%s carries the marker for gate %r, whose script %s is not present "
                            "in this checkout.\n"
                            "    The marker reads as coverage and buys none.  Either land that "
                            "gate, or roster this file." % (f, name, script))
                    else:
                        errs.append(
                            "%s carries the %s marker, but %s cannot be asked what it runs.\n"
                            "    Marker membership is only honoured for a gate whose own "
                            "discovery can be executed -- no text pattern distinguishes 'uses "
                            "the marker to find files' from 'uses the marker'.\n"
                            "    Move that script's marker discovery ABOVE its pytest/numpy/lal "
                            "probes (discovery needs only grep) and add, right after it builds "
                            "FILES:\n"
                            "      %s\n"
                            "    This census has no `needs: install`, so a list mode behind a "
                            "dependency probe cannot run." % (f, name, script, LIST_SNIPPET))
                else:
                    errs.append(
                        "%s carries '# RIFT-CI-GATE: %s', which is not a known gate.\n"
                        "    No script greps for that literal, so the marker looks like "
                        "coverage and provides none.\n"
                        "    Known gates: %s.  Fix the name, or add the gate to KNOWN_GATES in "
                        "%s." % (f, name, ", ".join(sorted(KNOWN_GATES)) or "(none)",
                                 os.path.relpath(__file__, REPO)))
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
    #    has been overtaken, and leaving it invites the next reader to trust it.
    #
    #    PENDING used to be UNCONDITIONALLY exempt from this, which made it the one status that
    #    could sit in the roster for ever: its whole point was to stay legal before AND after the
    #    companion PR landed, so nothing ever forced its removal.  That bought merge-order
    #    independence at the price of a status with no expiry, which is the rot this file exists
    #    to prevent.  It now carries an ENFORCEABLE condition instead of a promise: the reason
    #    must name the gate it waits on as `gate:<name>`, and the entry is legal only while that
    #    gate is NOT live.  The moment the gate lands, the entry is an error naming itself.
    #
    #    The cost is honest and stated in the PR: merging the companion needs a one-line deletion
    #    here.  That is a forcing function, not a failure.
    for f, (status, reason) in sorted(roster.items()):
        if f not in reachable:
            errs.append("%s: %s no longer exists. A roster entry for a deleted file is a "
                        "silent no-op; drop the line." % (ROSTER, f))
            continue
        if status == "PENDING":
            m = re.search(r"gate:([a-z0-9-]+)", reason)
            if not m:
                errs.append("%s: %s is PENDING but its reason names no gate. Write `gate:<name>` "
                            "in the reason so the entry has a condition that can expire, or use "
                            "a status that does not need one." % (ROSTER, f))
            elif m.group(1) not in KNOWN_GATES:
                errs.append("%s: %s is PENDING on gate %r, which is not in KNOWN_GATES. A "
                            "condition that can never be met never expires."
                            % (ROSTER, f, m.group(1)))
            elif m.group(1) in gates:
                errs.append("%s: %s is PENDING on gate %r, and that gate is now LIVE.\n"
                            "    The wait is over: either the gate registers this file (delete "
                            "this line) or it does not (give the file a real status)."
                            % (ROSTER, f, m.group(1)))
            continue
        if reachable[f] is not None:
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
