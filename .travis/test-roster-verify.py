#!/usr/bin/env python3
"""Check that each roster entry's STATUS is still true, not merely present.

WHY THIS EXISTS.  .travis/test-ci-roster.py enforces that every ungated test file carries a
reason.  It cannot tell whether the reason is CORRECT, and that gap is not theoretical: the
roster asserted "skips cleanly without them" for two jax_gp files while one of them ERRORED
without jax and the other had no guard at all (PR #248).  A reason nobody re-checks is prose,
and prose is what this whole census exists to stop being mistaken for coverage.

So each status carries a FALSIFIABLE, direction-checked predicate, and this job runs it:

  LEGACY     must FAIL to import, and only a collection/import ERROR shows that.  A clean
             collection satisfies nothing -- neither one that finds tests nor one that finds
             none -- because both mean the pre-package module names it supposedly needs are
             resolving, and the file is a candidate for real gating.
  HANDRUN    must collect NO tests.  If it collects some, it is a pytest suite wearing the
             wrong label -- and one that no job runs.
  EXPENSIVE  must collect tests AND, without RIFT_RUN_EXPENSIVE, SKIP them in a run that exits
             cleanly.  Passing none is NOT the predicate: a suite that fails or errors passes
             none too, and a broken suite is not a guarded one.  Catches both a suite that
             stopped collecting and an opt-in guard that stopped guarding.
  OPTDEP     must declare its dependencies as `needs:<mod>[,<mod>]` or `needs:env:VAR`, none of
             which may appear in requirements.txt -- if CI installs it, it is not optional.  The
             behavioural half is keyed off whether those deps are ACTUALLY present in the running
             environment, because that differs between CIT and a runner: with one absent the file
             must not collect-and-fully-pass; with all present it may.  An earlier version of this
             check simply flagged "collects and all pass here", which fired on CIT purely because
             CIT has jax -- a check that reported the environment rather than the claim.

DELIBERATELY NOT CHECKED: which dependency an OPTDEP file wants, and whether a HANDRUN study's
internal gate still holds.  Both need the missing stack or a long run; claiming to check them
would be the same overreach this file exists to catch.  The predicates above are the part that
is decidable HERE, and the docstrings say so.

This is a SEPARATE job from ci-roster-check on purpose: that one is stdlib-only with no
`needs: install`, and must stay that way so it reports even when the install matrix is broken.
This one needs RIFT importable.
"""

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROSTER = os.path.join(".travis", "ci_roster.txt")
CODE = os.path.join("MonteCarloMarginalizeCode", "Code")
TIMEOUT = 300
CLEAN_RCS = (0, 5)   # pytest: 0 = collected/ran without error, 5 = imported fine, no tests here


def _read_roster():
    out = []
    for n, raw in enumerate(open(ROSTER, errors="replace"), 1):
        if raw.lstrip().startswith("#") or not raw.strip():
            continue
        bits = raw.rstrip("\n").split(None, 2)
        if len(bits) >= 3:
            out.append((n, bits[0], bits[1], bits[2].strip()))
    return out



def _declared_deps(reason):
    """Modules / env vars an OPTDEP entry claims, from `needs:a,b` or `needs:env:VAR`."""
    m = re.search(r"needs:([A-Za-z0-9_.,:]+)", reason)
    return [d for d in m.group(1).split(",") if d] if m else []


# Import names that installed metadata cannot resolve to their distribution.  Kept SHORT and
# justified: on CIT, `lalsuite` is a conda metapackage whose dist-info lists no top-level modules
# at all (its files() shows only __pycache__ and the dist-info), so neither packages_distributions
# nor a file scan can learn that `lal` comes from it.  A pip-installed lalsuite wheel does declare
# them, so this table is a fallback for the environment, not a replacement for the lookup.
# Add an entry only when the two mechanisms below genuinely cannot answer.
KNOWN_ALIASES = {
    "lal": "lalsuite", "lalsimulation": "lalsuite", "lalframe": "lalsuite",
    "lalmetaio": "lalsuite", "lalburst": "lalsuite", "lalinspiral": "lalsuite",
    "lalpulsar": "lalsuite", "lalinference": "lalsuite",
}


def _distributions_for(mod):
    """Distribution names that provide this IMPORT name, e.g. sklearn -> {scikit-learn}.

    Three mechanisms, cheapest first: the metadata index, a scan of each distribution's files
    for a top-level `<mod>/` or `<mod>.py`, and finally KNOWN_ALIASES for distributions whose
    metadata declares nothing.
    """
    top = mod.split(".")[0]
    out = set()
    try:
        from importlib.metadata import packages_distributions, distributions, files
    except ImportError:      # pragma: no cover - py<3.10
        return {KNOWN_ALIASES[top]} if top in KNOWN_ALIASES else set()
    try:
        out |= set(packages_distributions().get(top, []))
    except Exception:        # pragma: no cover - defensive
        pass
    if not out:
        try:
            for d in distributions():
                name = (d.metadata["Name"] or "")
                if not name:
                    continue
                for f in (files(name) or []):
                    parts = str(f).split("/")
                    if parts[0] == top or parts[0] == top + ".py":
                        out.add(name)
                        break
        except Exception:    # pragma: no cover - defensive
            pass
    if not out and top in KNOWN_ALIASES:
        out.add(KNOWN_ALIASES[top])
    return out


def _in_requirements(mod):
    """True if requirements.txt installs this module -- in which case it is not optional.

    THE IMPORT NAME IS NOT THE DISTRIBUTION NAME, and comparing them directly made this check
    fail open: `needs:sklearn` sailed past a requirements.txt that says `scikit-learn`, and so
    did `needs:lal` against `lalsuite` -- both importable in CI, both therefore NOT optional,
    both silently accepted as OPTDEP.  Reproduced before this fix for sklearn, lal,
    lalsimulation and skimage.

    So the import name is resolved to the distributions that provide it, and any of those
    matching a requirements line counts.  LIMIT, stated because it is real: the mapping comes
    from installed metadata, so a dependency absent from the CHECKING environment cannot be
    resolved and falls back to the bare name comparison.  In this job requirements.txt is
    installed, which is exactly the case that matters.
    """
    try:
        req = open(os.path.join(REPO, "requirements.txt"), errors="replace").read()
    except OSError:
        return False
    want = {n.lower().replace("-", "_") for n in ({mod} | _distributions_for(mod))}
    for line in req.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=\[]", line)[0].strip().lower().replace("-", "_")
        if name in want:
            return True
    return False


def _dep_present(dep):
    """Is this declared dependency actually available in the environment running the check?"""
    if dep.startswith("env:"):
        return bool(os.environ.get(dep[4:]))
    pr = subprocess.run([sys.executable, "-c", "import %s" % dep],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pr.returncode == 0


def _pytest(path, extra_env=None, collect_only=True):
    """Return (rc, n, n_passed), or (None, None, None) on timeout.

    n is the collected count under --collect-only and the SKIPPED count for a real run, since
    that is what tells a guarded suite apart from a broken one.  n_passed is -1 when not run.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, CODE) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    # A None value REMOVES the variable.  The EXPENSIVE predicate needs a run that genuinely
    # lacks RIFT_RUN_EXPENSIVE; inheriting it from the caller inverts the whole check -- a
    # correct guard then runs its tests and is reported as broken, and an inverted guard skips
    # and is reported as fine.  Reproduced with the variable exported.
    for k, v in (extra_env or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if collect_only:
        cmd.append("--collect-only")
    cmd.append(path)
    try:
        pr = subprocess.run(cmd, env=env, cwd=REPO, timeout=TIMEOUT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        return None, None, None
    text = pr.stdout.decode("utf-8", "replace")
    if collect_only:
        # -q prints one `path::id` line per collected test; count lines, not `::` occurrences,
        # so a class-based id does not count twice.
        return pr.returncode, sum(1 for ln in text.splitlines() if "::" in ln), -1
    m = re.search(r"(\d+) passed", text)
    s = re.search(r"(\d+) skipped", text)
    return pr.returncode, (int(s.group(1)) if s else 0), (int(m.group(1)) if m else 0)


def main():
    os.chdir(REPO)
    errs, checked = [], {}
    for lineno, path, status, reason in _read_roster():
        if not os.path.exists(path):
            continue                      # ci-roster-check owns that error
        if status == "LEGACY":
            rc, n, _ = _pytest(path)
            if rc is None:
                errs.append("%s:%d: %s timed out during collection." % (ROSTER, lineno, path))
            elif n > 0:
                errs.append("%s:%d: %s is LEGACY (\"cannot be imported\") but COLLECTS %d "
                            "tests.\n    Whatever it needed now resolves. Re-check the reason: "
                            "it is probably gateable." % (ROSTER, lineno, path, n))
            elif rc in CLEAN_RCS:
                errs.append("%s:%d: %s is LEGACY (\"cannot be imported\") but pytest collected it "
                            "WITHOUT error (exit %d) and found no tests.\n    Collecting nothing "
                            "is not evidence of a failed import -- only a collection error is -- "
                            "so the reason is stale: whatever it needed now resolves. Re-check it; "
                            "the file is HANDRUN at most, and probably gateable."
                            % (ROSTER, lineno, path, rc))
        elif status == "HANDRUN":
            rc, n, _ = _pytest(path)
            if rc is None:
                errs.append("%s:%d: %s timed out during collection." % (ROSTER, lineno, path))
            elif n > 0:
                errs.append("%s:%d: %s is HANDRUN (\"not a pytest target\") but COLLECTS %d "
                            "tests.\n    It is a real suite that no job runs -- gate it, or "
                            "correct the status." % (ROSTER, lineno, path, n))
        elif status == "EXPENSIVE":
            # Both calls drop RIFT_RUN_EXPENSIVE: collection too, since a module-level skip may
            # key off it and change what is collected at all.
            no_optin = {"RIFT_RUN_EXPENSIVE": None}
            rc, n, _ = _pytest(path, extra_env=no_optin)
            if rc is None or n == 0:
                errs.append("%s:%d: %s is EXPENSIVE but collects nothing.\n    The opt-in "
                            "suite is gone or stopped importing." % (ROSTER, lineno, path))
            else:
                rc2, skipped, passed = _pytest(path, collect_only=False, extra_env=no_optin)
                if rc2 is None:
                    errs.append("%s:%d: %s is EXPENSIVE but the run WITHOUT RIFT_RUN_EXPENSIVE "
                                "timed out after %ds.\n    Opting out should cost nothing, so "
                                "something is executing: the guard is not holding."
                                % (ROSTER, lineno, path, TIMEOUT))
                elif passed > 0:
                    errs.append("%s:%d: %s is EXPENSIVE (\"skips unless RIFT_RUN_EXPENSIVE=1\") "
                                "but %d test(s) PASSED without it.\n    The opt-in guard stopped "
                                "guarding." % (ROSTER, lineno, path, passed))
                elif rc2 != 0 or skipped < n:
                    errs.append("%s:%d: %s is EXPENSIVE (\"skips unless RIFT_RUN_EXPENSIVE=1\") "
                                "but the run WITHOUT it exited %d with %d of %d collected test(s) "
                                "skipped.\n    Passing nothing is not the same as being guarded: "
                                "a suite that fails or errors passes nothing either. Opting out "
                                "must be a CLEAN skip of every collected test."
                                % (ROSTER, lineno, path, rc2, skipped, n))
        elif status == "OPTDEP":
            deps = _declared_deps(reason)
            if not deps:
                errs.append("%s:%d: %s is OPTDEP but names no dependency.\n"
                            "    Write `needs:<module>[,<module>]` or `needs:env:VAR` in the "
                            "reason so the claim can be checked instead of believed.  Two entries "
                            "carrying only prose here turned out to collect and pass completely, "
                            "and belonged in a job." % (ROSTER, lineno, path))
            for d in deps:
                if not d.startswith("env:") and _in_requirements(d):
                    errs.append("%s:%d: %s is OPTDEP on %r, which requirements.txt DOES install.\n"
                                "    Then it is not optional -- gate the file."
                                % (ROSTER, lineno, path, d))
            missing = [d for d in deps if not _dep_present(d)]
            if missing:
                rc, n, _ = _pytest(path)
                if rc is None:
                    # A timeout is a verification that did NOT happen.  Counting it as checked
                    # is the same silent pass this file exists to remove: with TIMEOUT dropped
                    # to 1 s every OPTDEP subprocess timed out and the run still reported
                    # "OPTDEP 8 checked ... PASS".
                    errs.append("%s:%d: %s is OPTDEP and its collection TIMED OUT after %ds.\n"
                                "    Nothing was verified; a timeout is not a pass.  Raise "
                                "TIMEOUT if the file is legitimately slow, or fix the hang."
                                % (ROSTER, lineno, path, TIMEOUT))
                elif n > 0:
                    rc2, _, passed = _pytest(path, collect_only=False)
                    if rc2 is None:
                        errs.append("%s:%d: %s is OPTDEP and its RUN timed out after %ds with "
                                    "%s missing.\n    Nothing was verified; a timeout is not a "
                                    "pass."
                                    % (ROSTER, lineno, path, TIMEOUT, ",".join(missing)))
                    elif rc2 == 0 and passed == n:
                        errs.append("%s:%d: %s is OPTDEP on missing %s, yet collects %d tests and "
                                    "ALL PASS.\n    It does not actually need what it claims; gate "
                                    "it, or correct the reason."
                                    % (ROSTER, lineno, path, ",".join(missing), n))
        else:
            continue
        checked[status] = checked.get(status, 0) + 1

    print("test-roster-verify: predicates checked per status")
    for s in sorted(checked):
        print("  %-10s %3d" % (s, checked[s]))
    if errs:
        print("\ntest-roster-verify: FAIL", file=sys.stderr)
        for e in errs:
            print("  " + e, file=sys.stderr)
        return 1
    print("test-roster-verify: PASS -- every checkable roster reason still holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
