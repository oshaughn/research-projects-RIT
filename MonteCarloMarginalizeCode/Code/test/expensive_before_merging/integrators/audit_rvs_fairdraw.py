#!/usr/bin/env python3
"""Enumerate every read of ``_rvs`` (and of the columns derived from it) and classify
each one against the fair-draw rebind at the end of ``integrate_log``/``integrate``.

WHY THIS EXISTS
---------------
``sampler._rvs`` is not the sample set.  At the end of the integrator the fair-draw
export rebinds *every* ``_rvs`` key to a subset drawn WITH REPLACEMENT, proportional to
weight, of size ``min(n_extr, 1.5*eff_samp, 1.5*neff)``.  Any consumer that reads
``_rvs`` rows, lengths, or statistics after that point is reading an EXPORT RESAMPLE and
is skewed by roughly ``log(n_retained/eff_samp)`` -- catastrophically so on a collapsed
pass, where the resample can be a single row.

That shape has produced five separate defects (CIP posterior export; L0 rescue seed,
PR #78; rescue reject gate, PR #79; warm-seed reserve cap and its logarithm, PR #84).
This script exists so the sixth is found mechanically rather than by whoever happens to
be editing the surrounding code.

This audits a DIFFERENT axis from ``RVS_CACHE_AUDIT.md`` in this directory.  That one
asks "cached column, or canonical components?"; this one asks "before or after the
rebind?".  A site can be wrong on either axis independently.

PHASES
------
``BEFORE`` / ``AFTER``    read lexically inside a function that performs the rebind.
``CALLED_BEFORE/AFTER``   read in a helper, classified by where that helper is called
                          from inside the rebinding function (intra-file call graph).
``POST_INTEGRATE``        read on a sampler outside any rebinding function -- i.e. in
                          bin/ scripts and utilities that run after ``integrate`` has
                          returned.  Every one of these sees the resample.
``NO_REBIND``             read in a file/scope with no rebind at all (helpers, tests).

USAGE
-----
    python3 audit_rvs_fairdraw.py                 # human-readable report
    python3 audit_rvs_fairdraw.py --json          # machine-readable
    python3 audit_rvs_fairdraw.py --summary       # counts per file/phase
    python3 audit_rvs_fairdraw.py --check         # exit 1 on an unclassified site

``--check`` is the CI form.  It does NOT assert that every post-rebind site is a bug --
many are legitimately per-row.  It asserts that every post-rebind site appears in
``VERDICTS`` below with a recorded human judgement.  A new or moved consumer therefore
fails the build and has to be classified by a person, which is the property we want.

Needs Python >= 3.8 for end_lineno; degrades gracefully on 3.6/3.7.
"""
import argparse
import ast
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

TARGETS = [
    "RIFT/integrators/mcsampler.py",
    "RIFT/integrators/mcsamplerAdaptiveVolume.py",
    "RIFT/integrators/mcsamplerEnsemble.py",
    "RIFT/integrators/mcsamplerGPU.py",
    "RIFT/integrators/mcsamplerNFlow.py",
    "RIFT/integrators/mcsamplerPortfolio.py",
    "RIFT/integrators/mcsampler_generic.py",
    "RIFT/misc/distance_slices.py",
    "bin/integrate_likelihood_extrinsic_batchmode",
    "bin/integrate_likelihood_extrinsic_batchmode_lisa",
    "bin/integrate_likelihood_extrinsic",
    "bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py",
    "bin/util_ConstructEOSPosterior.py",
]

# Columns whose per-row VALUES survive the rebind but whose LENGTH and population
# statistics do not.
DERIVED_KEYS = {
    "log_integrand", "log_joint_prior", "log_joint_s_prior", "log_weights",
    "weights", "joint_prior", "joint_s_prior", "integrand", "sample_n",
}

# Calls that turn rows into a population statistic.  Used to raise the severity of a
# report, never to decide it.
POPULATION_CALLS = {
    "sum", "mean", "cov", "std", "var", "median", "average", "argmax", "argmin",
    "max", "min", "len", "logsumexp", "percentile", "quantile", "corrcoef",
    "histogram", "cumsum", "count_nonzero", "vstack", "column_stack", "shape",
}

POST_REBIND_PHASES = ("AFTER", "CALLED_AFTER", "POST_INTEGRATE")


# --------------------------------------------------------------------------- AST compat
def _const_str(node):
    """String value of a subscript slice, across 3.6-3.12 AST shapes."""
    if hasattr(ast, "Index") and isinstance(node, getattr(ast, "Index")):
        node = node.value
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if hasattr(ast, "Str") and isinstance(node, getattr(ast, "Str")):
        return node.s
    return None


def _end_lineno(node, fallback):
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return end
    return max([n.lineno for n in ast.walk(node) if hasattr(n, "lineno")] or [fallback])


# --------------------------------------------------------------------------- analysis
def _rebind_lines(tree):
    """{function name: line of the fair-draw rebind}.  The rebind is the assignment to
    ``self._rvs[key]`` inside the branch guarded by ``bFairdraw``."""
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.If):
                continue
            cond = {n.id for n in ast.walk(sub.test) if isinstance(n, ast.Name)}
            if "bFairdraw" not in cond:
                continue
            writes = [
                s.lineno
                for s in ast.walk(sub)
                if isinstance(s, ast.Assign)
                for t in s.targets
                if isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Attribute)
                and t.value.attr == "_rvs"
            ]
            if writes:
                # innermost rebind wins; first write line is where _rvs stops being the
                # retained set
                out[node.name] = min(out.get(node.name, 10 ** 9), min(writes))
    return out


def _function_spans(tree):
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, _end_lineno(node, node.lineno), node.name))
    spans.sort(key=lambda s: s[1] - s[0])  # innermost first
    return spans


def _call_sites(tree):
    """{callee name: [(line, enclosing function), ...]} for intra-file calls."""
    spans = _function_spans(tree)

    def enclosing(lineno):
        for lo, hi, name in spans:
            if lo <= lineno <= hi:
                return name
        return "<module>"

    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name:
            out.setdefault(name, []).append((node.lineno, enclosing(node.lineno)))
    return out


def scan_file(relpath):
    path = os.path.join(CODE_ROOT, relpath)
    if not os.path.exists(path):
        return [{"file": relpath, "line": 0, "phase": "MISSING", "source": "file not found"}]
    with open(path, "r", errors="replace") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [{"file": relpath, "line": 0, "phase": "UNPARSEABLE", "source": str(e)}]

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node

    rebinds = _rebind_lines(tree)
    spans = _function_spans(tree)
    calls = _call_sites(tree)
    lines = src.splitlines()

    def enclosing(lineno):
        for lo, hi, name in spans:
            if lo <= lineno <= hi:
                return name
        return "<module>"

    write_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)) or type(node).__name__ == "AnnAssign":
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    write_nodes.add(id(sub))

    def phase_of(fn_name, lineno):
        """Classify a read, following one level of intra-file call graph."""
        rebind = rebinds.get(fn_name)
        if rebind is not None:
            return ("BEFORE" if lineno < rebind else "AFTER"), rebind
        # helper: where is it called from?
        verdicts = set()
        rb_line = None
        for call_line, caller in calls.get(fn_name, []):
            caller_rebind = rebinds.get(caller)
            if caller_rebind is None:
                continue
            rb_line = caller_rebind
            verdicts.add("CALLED_BEFORE" if call_line < caller_rebind else "CALLED_AFTER")
        if verdicts == {"CALLED_BEFORE"}:
            return "CALLED_BEFORE", rb_line
        if verdicts == {"CALLED_AFTER"}:
            return "CALLED_AFTER", rb_line
        if verdicts:
            return "CALLED_BOTH", rb_line
        # No intra-file caller inside a rebinding function.  If this file has any
        # rebind at all it is an integrator, and an unlinked helper is ambiguous;
        # otherwise this is a downstream consumer of a returned sampler.
        return ("NO_REBIND" if rebinds else "POST_INTEGRATE"), None

    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == "_rvs"):
            continue
        parent = parents.get(id(node))
        subscript = parent if isinstance(parent, ast.Subscript) and parent.value is node else None
        target = subscript if subscript is not None else node
        if id(target) in write_nodes:
            continue  # a write, not a read

        key = _const_str(subscript.slice) if subscript is not None else None

        pop = None
        p = parents.get(id(target))
        depth = 0
        while p is not None and depth < 5:
            if isinstance(p, ast.Call):
                fn = p.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name in POPULATION_CALLS:
                    pop = name
                    break
            p = parents.get(id(p))
            depth += 1

        fn_name = enclosing(node.lineno)
        phase, rebind = phase_of(fn_name, node.lineno)

        hits.append({
            "file": relpath,
            "line": node.lineno,
            "function": fn_name,
            "key": key,
            "phase": phase,
            "rebind_line": rebind,
            "population_call": pop,
            "derived_key": (key in DERIVED_KEYS) if key else None,
            "source": lines[node.lineno - 1].strip()[:200],
        })
    return hits


# --------------------------------------------------------------------------- ledger
# Human verdicts for post-rebind sites, in rvs_fairdraw_verdicts.json.  Verdict is one of:
#   PER_ROW   -- reads a per-row value it independently owns.  The resample changes WHICH
#                rows are present, never the value carried by a row, so this is correct.
#   FIXED     -- was broken; now reads the retained set or an exact recorded total.
#   BROKEN    -- reads a population statistic of the resample.  Must name its follow-up.
#   BENIGN    -- a population read whose consumer is documented as describing the EXPORT
#                rather than the integral, so the resample is the right input.
#   NO_FAIRDRAW -- this caller never sets igrand_fairdraw_samples, so no rebind happens on
#                its sampler and _rvs is still the retained set.  VERIFY BY GREP, not by
#                assumption: the flag arrives from the caller, not from the sampler.
LEDGER_PATH = os.path.join(HERE, "rvs_fairdraw_verdicts.json")
VALID_VERDICTS = ("PER_ROW", "FIXED", "BROKEN", "BENIGN", "NO_FAIRDRAW")


def site_key(hit):
    """A stable identifier for one post-rebind read.

    NOT the line number -- these files are edited constantly and a line-keyed ledger would be
    stale within a week.  NOT the enclosing function either: analyze_event alone holds 40 of
    these, so a function-keyed ledger would let a NEW consumer be added beside 39 approved
    ones without tripping anything -- exactly the failure mode this gate exists to prevent,
    since every one of the five known defects was added next to correct code.

    So: file, function, and a hash of the READ ITSELF with whitespace squeezed out.  Moving
    or reindenting a line keeps its verdict; changing what it reads does not.
    """
    norm = "".join((hit.get("source") or "").split())
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:10]
    return "{}:{}:{}".format(hit["file"], hit["function"], h)


def load_verdicts():
    """Verdicts live in a sidecar JSON so the ledger can be edited without touching the
    scanner.  Missing file => empty ledger => --check reports everything, which is the
    correct behaviour for a gate: absent evidence is not a pass."""
    if not os.path.exists(LEDGER_PATH):
        return {}
    with open(LEDGER_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit-ledger", action="store_true",
                    help="print a verdict ledger skeleton, preserving existing verdicts")
    ap.add_argument("--phase", help="comma-separated phases to keep")
    ap.add_argument("--post-rebind", action="store_true",
                    help="only sites that see the resample")
    args = ap.parse_args()

    hits = []
    for t in TARGETS:
        hits.extend(scan_file(t))
    if args.post_rebind:
        hits = [h for h in hits if h.get("phase") in POST_REBIND_PHASES]
    if args.phase:
        keep = set(args.phase.split(","))
        hits = [h for h in hits if h.get("phase") in keep]

    if args.json:
        json.dump(hits, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    if args.summary:
        tally = {}
        for h in hits:
            tally.setdefault(h["file"], {}).setdefault(h["phase"], 0)
            tally[h["file"]][h["phase"]] += 1
        phases = sorted({p for v in tally.values() for p in v})
        print("{:<58} {}".format("file", "  ".join("{:>14}".format(p) for p in phases)))
        for f in sorted(tally):
            print("{:<58} {}".format(
                f, "  ".join("{:>14}".format(tally[f].get(p, "")) for p in phases)))
        print("\ntotal sites: {}".format(len(hits)))
        return 0

    if args.emit_ledger:
        ledger = load_verdicts()
        out = {}
        for h in hits:
            if h.get("phase") not in POST_REBIND_PHASES:
                continue
            k = site_key(h)
            out[k] = ledger.get(k, {
                "verdict": "TODO",
                "why": "",
                "source": h.get("source", ""),
            })
            out[k]["source"] = h.get("source", "")
        json.dump(out, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if args.check:
        ledger = load_verdicts()
        unclassified, bad = [], []
        n_post = 0
        for h in hits:
            if h.get("phase") not in POST_REBIND_PHASES:
                continue
            n_post += 1
            k = site_key(h)
            entry = ledger.get(k)
            if entry is None:
                unclassified.append((k, h["line"], h["source"]))
            elif entry.get("verdict") not in VALID_VERDICTS:
                bad.append((k, h["line"], entry.get("verdict")))
        if unclassified or bad:
            if unclassified:
                print("UNCLASSIFIED post-rebind _rvs reads ({}):".format(len(unclassified)))
                for k, line, src in unclassified:
                    print("  {}\n    line {}: {}".format(k, line, src))
            if bad:
                print("\nSites with no usable verdict ({}); must be one of {}:".format(
                    len(bad), ", ".join(VALID_VERDICTS)))
                for k, line, v in bad:
                    print("  {} (line {}) -> {!r}".format(k, line, v))
            print("\n`_rvs` is an EXPORT resample by this point.  For each site above, decide")
            print("whether it reads a per-row value it owns (fine) or a population statistic")
            print("of the resample (broken), then record it:")
            print("  python3 {} --emit-ledger > {}".format(
                os.path.basename(__file__), os.path.basename(LEDGER_PATH)))
            return 1
        print("OK: all {} post-rebind _rvs reads carry a recorded verdict.".format(n_post))
        return 0

    by_file = {}
    for h in hits:
        by_file.setdefault(h["file"], []).append(h)
    n_flag = 0
    for f in sorted(by_file):
        print("\n=== {} ===".format(f))
        for h in sorted(by_file[f], key=lambda x: x["line"]):
            flag = ""
            if h.get("phase") in POST_REBIND_PHASES and h.get("population_call"):
                flag = "   <== post-rebind + {}()".format(h["population_call"])
                n_flag += 1
            print("  {:>5}  {:<34} {:<15} key={:<22}{}".format(
                h["line"], str(h.get("function"))[:34], str(h.get("phase")),
                str(h.get("key")), flag))
            print("         {}".format(h.get("source", "")))
    print("\n{} sites; {} post-rebind reads feed a population-shaped call.".format(
        len(hits), n_flag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
