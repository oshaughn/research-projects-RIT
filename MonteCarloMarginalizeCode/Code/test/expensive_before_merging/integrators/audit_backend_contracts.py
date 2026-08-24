#!/usr/bin/env python3
"""What does each sampler backend actually PUT IN `_rvs`, and what does it expect back?

WHY THIS EXISTS
---------------
The backends are structurally different in ways nothing states, and a consumer that guesses
wrong gets a plausible number rather than an error.  Concretely, this bit twice while wiring
the record in one afternoon:

  * `_rvs['integrand']` holds THREE different things.  It is lnL on AV / NFlow / portfolio
    (aliased from log_integrand), linear L on mcsampler / mcsamplerGPU, and EITHER on
    mcsamplerEnsemble depending on the `return_lnI` kwarg -- i.e. for one backend the column's
    meaning is a RUNTIME property of how the pass was called.  Feed a log callable to a linear
    entry point and the fair draw computes NEGATIVE weights and raises, if you are lucky;
    downstream the same mistake does NOT raise, it takes log() of a log and returns a
    plausible, almost-flat weight vector.  `ln_weights_from_rvs` carries a long comment about
    exactly this, which is why it REQUIRES `use_lnL` to be passed explicitly.
  * only AV and the portfolio keep a `_warm_seed_reserve`; the L0 rescue and the sequential
    warm start have to cope with its absence.
  * the portfolio's `_rvs` holds EVERY draw (including -inf rows); AV's holds only the
    retained subset.  That is a ~90x memory difference and it changes what "n_retained" means.

None of that is discoverable without reading five files.  This prints it as a table, and
`--check` fails when a backend's contract changes without the table being updated -- so the
next developer meets a diff instead of a landmine.

USAGE
-----
    python3 audit_backend_contracts.py            # the table
    python3 audit_backend_contracts.py --json
    python3 audit_backend_contracts.py --check    # CI: contracts match the recorded ledger
"""
import argparse
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

BACKENDS = [
    "mcsampler",
    "mcsamplerAdaptiveVolume",
    "mcsamplerEnsemble",
    "mcsamplerGPU",
    "mcsamplerNFlow",
    "mcsamplerPortfolio",
]

LEDGER = os.path.join(HERE, "backend_contracts.json")

# Columns whose presence distinguishes the log convention from the linear one.
LOG_COLS = ("log_integrand", "log_joint_prior", "log_joint_s_prior")
LIN_COLS = ("integrand", "joint_prior", "joint_s_prior")


def _written_rvs_keys(tree):
    """String keys assigned into self._rvs anywhere in the module."""
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Subscript)
                    and isinstance(t.value, ast.Attribute)
                    and t.value.attr == "_rvs"):
                sl = t.slice
                if hasattr(ast, "Index") and isinstance(sl, getattr(ast, "Index")):
                    sl = sl.value
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    keys.add(sl.value)
    return keys


def _entry_points(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name in ("integrate", "integrate_log"):
            names.add(node.name)
    return names


def _rebind_count(src):
    return src.count("self._rvs_is_fairdraw = True")


def scan(name):
    path = os.path.join(CODE, "RIFT", "integrators", "{}.py".format(name))
    if not os.path.exists(path):
        return {"backend": name, "error": "missing"}
    src = open(path, errors="replace").read()
    tree = ast.parse(src)
    keys = _written_rvs_keys(tree)
    # WHAT DOES _rvs['integrand'] ACTUALLY HOLD?  Not the same question as "which columns
    # exist" -- most backends write both families.  Three distinct answers:
    #   * aliased from log_integrand   -> it holds lnL, always
    #   * a return_lnI/use_lnL kwarg   -> it holds L or lnL depending on how the pass was CALLED
    #   * neither                      -> it holds L, always
    # The middle case is the dangerous one: the column's meaning is a runtime property, so no
    # amount of reading the consumer tells you which it is.
    aliased = ("_rvs['integrand'] = self._rvs['log_integrand']" in src.replace('"', "'"))
    kwarg = "return_lnI" in src
    if aliased:
        integrand_holds = "log (aliased)"
    elif kwarg:
        integrand_holds = "L or lnL (kwarg)"
    else:
        integrand_holds = "linear"
    return {
        "backend": name,
        "entry_points": sorted(_entry_points(tree)),
        "integrand_holds": integrand_holds,
        "has_return_lnI_kwarg": kwarg,
        "rvs_keys": sorted(keys),
        "keeps_warm_seed_reserve": "self._warm_seed_reserve" in src,
        "builds_reserve": "make_warm_seed_reserve(" in src,
        "n_rebind_sites": _rebind_count(src),
        "sets_rvs_record": "RvsRecord.fair_draw(" in src,
        "has_clear_warm_state": "def clear_warm_state" in src,
        "has_reset_sampling": "def reset_sampling" in src,
        "has_bootstrap_from_samples": "def bootstrap_from_samples" in src,
    }


FIELDS = [
    ("entry_points", "entry"),
    ("integrand_holds", "_rvs['integrand']"),
    ("keeps_warm_seed_reserve", "reserve"),
    ("n_rebind_sites", "rebinds"),
    ("sets_rvs_record", "record"),
    ("has_bootstrap_from_samples", "bootstrap"),
    ("has_clear_warm_state", "clear_warm"),
    ("has_reset_sampling", "reset_samp"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--emit-ledger", action="store_true")
    args = ap.parse_args()

    rows = [scan(b) for b in BACKENDS]

    if args.json or args.emit_ledger:
        out = {r["backend"]: r for r in rows}
        if args.emit_ledger:
            with open(LEDGER, "w") as f:
                json.dump(out, f, indent=2, sort_keys=True)
                f.write("\n")
            print("wrote {}".format(os.path.basename(LEDGER)))
            return 0
        json.dump(out, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if args.check:
        if not os.path.exists(LEDGER):
            print("no recorded contracts; run --emit-ledger")
            return 1
        want = json.load(open(LEDGER))
        bad = []
        for r in rows:
            w = want.get(r["backend"])
            if w is None:
                bad.append((r["backend"], "not in the ledger at all"))
                continue
            for k in sorted(set(list(r)) | set(list(w))):
                if r.get(k) != w.get(k):
                    bad.append((r["backend"],
                                "{}: recorded {!r}, now {!r}".format(k, w.get(k), r.get(k))))
        if bad:
            print("BACKEND CONTRACT CHANGED ({} difference(s)):".format(len(bad)))
            for b, msg in bad:
                print("  {:<26} {}".format(b, msg))
            print("\nThese differences are the landmine this file exists to surface: a consumer")
            print("written against one backend meets another and gets a plausible wrong number.")
            print("If the change is intended, re-record it and say why in the PR:")
            print("  python3 {} --emit-ledger".format(os.path.basename(__file__)))
            return 1
        print("OK: all {} backend contracts match the recorded ledger.".format(len(rows)))
        return 0

    print("=" * 108)
    print("SAMPLER BACKEND CONTRACTS -- what each one puts in _rvs and what it expects back")
    print("=" * 108)
    w = {"_rvs['integrand']": 19}
    hdr = "{:<26}".format("backend") + "".join(
        "{:<{}}".format(lbl, w.get(lbl, 13)) for _, lbl in FIELDS)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = "{:<26}".format(r["backend"])
        for key, _ in FIELDS:
            v = r.get(key)
            if isinstance(v, list):
                v = ",".join(x.replace("integrate", "int") for x in v) or "-"
            elif isinstance(v, bool):
                v = "yes" if v else "-"
            line += "{:<{}}".format(str(v), w.get(dict(FIELDS)[key], 13))
        print(line)

    print("\nTHE TRAPS, spelled out:")
    print("  * _rvs['integrand'] HOLDS THREE DIFFERENT THINGS:")
    for kind in ("linear", "log (aliased)", "L or lnL (kwarg)"):
        who = [r["backend"] for r in rows if r.get("integrand_holds") == kind]
        print("      {:<20} {}".format(kind, ", ".join(who) or "none"))
    print("    The kwarg case is the dangerous one: the column's meaning is a RUNTIME property")
    print("    of how the pass was called, so reading the consumer cannot tell you which it is.")
    print("    That is why ln_weights_from_rvs REQUIRES use_lnL to be passed explicitly, and")
    print("    why it must be the stored convention rather than opts.internal_use_lnL.")
    print("  * ENTRY POINT is not the convention either: a backend with only `integrate` takes")
    print("    a LINEAR callable, and feeding it a log one makes the fair draw compute NEGATIVE")
    print("    weights and raise.  Downstream the same mistake does NOT raise -- it takes log()")
    print("    of a log and returns a plausible, wrong, almost-flat weight vector.")
    no_res = [r["backend"] for r in rows if not r["keeps_warm_seed_reserve"]]
    print("  * NO warm-seed reserve: {}".format(", ".join(no_res) or "none"))
    print("    So the L0 rescue and the sequential warm start must cope with its absence, and")
    print("    RvsRecord.retained_points() answers None rather than pretending.")
    print("  * _rvs CONTENTS differ: the portfolio keeps EVERY draw (including -inf rows), AV")
    print("    only the retained subset -- ~92 MB vs ~0.9 MB per million nmax (measured,")
    print("    measure_retained_set_memory.py).  'n_retained' means different things.")
    print("\nPer-backend _rvs keys:")
    for r in rows:
        print("  {:<26} {}".format(r["backend"], ", ".join(r["rvs_keys"]) or "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
