#!/usr/bin/env python3
"""
Audit: what the main ILE driver has that the LISA ILE driver does not.

The two drivers are a DELIBERATE FORK (RO, 2026-08-13: "the overhead of one ring to
rule them all is too high").  This script does not argue with that.  It makes the
consequence -- drift -- mechanically visible, so the fork stays a choice rather than
an accident.

    bin/integrate_likelihood_extrinsic_batchmode        <- main, moves fast
    bin/integrate_likelihood_extrinsic_batchmode_lisa   <- LISA, lags

Both import the SAME integrators (``mcsampler``, ``mcsamplerEnsemble``, ``mcsamplerGPU``,
``mcsamplerAdaptiveVolume``, ``mcsamplerPortfolio``), so anything landed in
``RIFT/integrators/`` already reaches LISA.  The drift measured here is entirely in the
driver: helpers, CLI options, module constants and sampler provenance markers.

WHAT IS EXTRACTED
-----------------
``FUNC``    ``def`` names, qualified by enclosing function (``analyze_event._foo``), so a
            nested helper is not confused with a top-level one of the same name.
``OPTION``  ``--foo`` literals passed to ``add_option``/``add_argument``.  These drivers
            use ``optparse``; both call forms are scanned so a future port to argparse
            does not silently empty this category.
``CONST``   module-level ``UPPER_CASE`` assignments -- the sentinels (``_SEQ_WS_PENDING``)
            and tuning constants that travel with a feature.
``ATTR``    provenance markers set/read on the sampler object (``_rvs_is_fairdraw``,
            ``_warm_seed_reserve``, ...).  These are the fair-draw correctness family
            from PR #87 and are the reason this audit exists.

THE LEDGER
----------
Every gap item needs a recorded decision in ``lisa_drift_ledger.json``:

``PORT``     belongs in LISA and is not there yet -- an open work item.
``PORTED``   carried across; the item should have disappeared from the gap, so a
             ``PORTED`` entry still showing up in the gap is itself an error.
``NA``       does not apply to LISA, WITH A REASON.  "Does not apply" is a fine answer;
             silence is not.
``PHYSICS``  needs a physics decision before it can be answered, with the question
             recorded verbatim.

USAGE
-----
    python3 audit_lisa_driver_drift.py              # human-readable gap report
    python3 audit_lisa_driver_drift.py --summary    # counts per category and decision
    python3 audit_lisa_driver_drift.py --json       # machine-readable
    python3 audit_lisa_driver_drift.py --undecided  # only items with no ledger entry
    python3 audit_lisa_driver_drift.py --check      # CI gate: exit 1 on an undecided item

``--check`` is the CI form, and it is deliberately weak about physics: it does not assert
that the gap is empty, or that any particular item was ported.  Closing the gap is not the
goal -- the fork is intentional.  It asserts only that no item drifted in unnoticed.  A new
helper or option in the main driver fails the build until a person classifies it, which is
the property we want and the one that was missing when 2,357 lines accumulated.

Keyed by NAME, not by source hash (the fair-draw audit next door keys by hash because it
tracks reads of one attribute, which move).  Names are the stable identity here: renaming a
helper in the main driver SHOULD invalidate its verdict, since the thing being tracked is
"does LISA have this", and a rename means nobody has answered that about the new name.

Needs Python >= 3.8.
"""
import argparse
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

MAIN = "bin/integrate_likelihood_extrinsic_batchmode"
LISA = "bin/integrate_likelihood_extrinsic_batchmode_lisa"

LEDGER = os.path.join(HERE, "lisa_drift_ledger.json")

DECISIONS = ("PORT", "PORTED", "NA", "PHYSICS")

# Sampler attributes worth tracking as provenance markers.  Prefix-matched.  Kept narrow
# on purpose: every one of these is a boolean or a record describing MUTABLE SHARED STATE,
# which is the shape that produced six defects in PR #87 (see RVS_FAIRDRAW_AUDIT.md).
ATTR_PREFIXES = ("_rvs_is", "_warm_seed", "_retained", "_export_")


def _is_str(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


class _Collector(ast.NodeVisitor):
    def __init__(self):
        self.funcs = {}      # qualified name -> lineno
        self.options = {}    # "--foo"        -> lineno
        self.consts = {}     # NAME           -> lineno
        self.attrs = {}      # attr name      -> lineno
        self._stack = []

    def visit_FunctionDef(self, node):
        qual = ".".join(self._stack + [node.name])
        self.funcs.setdefault(qual, node.lineno)
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("add_option", "add_argument"):
            for arg in node.args:
                if _is_str(arg) and arg.value.startswith("--"):
                    self.options.setdefault(arg.value, node.lineno)
        # getattr(sampler, '_rvs_is_fairdraw', False) names an attribute just as much as
        # sampler._rvs_is_fairdraw does, and the defensive getattr form is the one the
        # provenance READERS use.  Missing it would let a real port look like a no-op.
        if isinstance(func, ast.Name) and func.id in ("getattr", "setattr", "hasattr"):
            for arg in node.args[1:2]:
                if _is_str(arg) and any(arg.value.startswith(p) for p in ATTR_PREFIXES):
                    self.attrs.setdefault(arg.value, node.lineno)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if not self._stack:
            for tgt in node.targets:
                name = getattr(tgt, "id", None)
                if name and name.upper() == name and any(c.isalpha() for c in name):
                    self.consts.setdefault(name, node.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if any(node.attr.startswith(p) for p in ATTR_PREFIXES):
            self.attrs.setdefault(node.attr, node.lineno)
        self.generic_visit(node)


def collect(relpath):
    path = os.path.join(CODE_ROOT, relpath)
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    c = _Collector()
    c.visit(tree)
    return {"FUNC": c.funcs, "OPTION": c.options, "CONST": c.consts, "ATTR": c.attrs}


def compute_gap():
    """Items present in the main driver and absent from the LISA driver.

    Returns (gap, extras) where gap is a list of dicts and extras lists LISA-only
    items -- reported but never gated, since LISA is allowed its own surface.
    """
    main = collect(MAIN)
    lisa = collect(LISA)
    gap, extras = [], []
    # A FUNC is satisfied by its BARE name as well as its qualified one.  The main driver has
    # ONE analyze_event and nests helpers inside it; this driver has TWO (analyze_event_LISA
    # and analyze_event), so a helper ported here must be hoisted to module level or else
    # duplicated -- and duplicating is the failure mode this audit exists to prevent.  Without
    # this, every correctly-hoisted port would sit in the gap forever as a false positive,
    # which is how a gate gets trained out of people.
    _lisa_bare = {n.rsplit(".", 1)[-1] for n in lisa["FUNC"]}
    for cat in ("FUNC", "OPTION", "CONST", "ATTR"):
        for name in sorted(set(main[cat]) - set(lisa[cat])):
            if cat == "FUNC" and name.rsplit(".", 1)[-1] in _lisa_bare:
                continue
            gap.append({"category": cat, "name": name,
                        "key": "%s:%s" % (cat, name), "main_line": main[cat][name]})
        for name in sorted(set(lisa[cat]) - set(main[cat])):
            extras.append({"category": cat, "name": name, "lisa_line": lisa[cat][name]})
    return gap, extras


def load_ledger():
    """Missing ledger => empty => --check reports every item, which is the safe direction."""
    if not os.path.exists(LEDGER):
        return {}
    with open(LEDGER) as fh:
        raw = json.load(fh)
    return raw.get("entries", raw)


def annotate(gap, ledger):
    for item in gap:
        entry = ledger.get(item["key"])
        item["decision"] = entry.get("decision") if entry else None
        item["reason"] = entry.get("reason") if entry else None
    return gap


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--undecided", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    gap, extras = compute_gap()
    ledger = load_ledger()
    gap = annotate(gap, ledger)

    undecided = [g for g in gap if g["decision"] is None]
    # A PORTED item that is still missing from LISA means the ledger is lying about the
    # tree -- either the port was reverted or it never landed.  Louder than undecided.
    stale = [g for g in gap if g["decision"] == "PORTED"]
    # A ledger entry naming an item no longer in the gap is spent: either it was ported
    # (good) or the main driver dropped it (also fine).  Not a failure, but worth showing
    # so the ledger does not accumulate fiction.
    gap_keys = {g["key"] for g in gap}
    spent = sorted(k for k in ledger if k not in gap_keys)

    if args.json:
        json.dump({"gap": gap, "lisa_only": extras, "undecided": len(undecided),
                   "stale_ported": [s["key"] for s in stale], "spent_entries": spent},
                  sys.stdout, indent=2, sort_keys=True)
        print()
        return 0

    if args.summary:
        print("LISA driver drift: %d items in main and absent from lisa" % len(gap))
        for cat in ("FUNC", "OPTION", "CONST", "ATTR"):
            rows = [g for g in gap if g["category"] == cat]
            if not rows:
                continue
            counts = {}
            for r in rows:
                counts[r["decision"] or "UNDECIDED"] = counts.get(r["decision"] or "UNDECIDED", 0) + 1
            detail = "  ".join("%s=%d" % (k, counts[k]) for k in sorted(counts))
            print("  %-7s %3d   %s" % (cat, len(rows), detail))
        print("  LISA-only surface (never gated): %d" % len(extras))
        if spent:
            print("  spent ledger entries (no longer in gap): %d" % len(spent))
        return 0

    rows = undecided if args.undecided else gap
    if args.undecided and not rows:
        print("no undecided items: every gap item carries a recorded decision")
    for cat in ("FUNC", "OPTION", "CONST", "ATTR"):
        sel = [g for g in rows if g["category"] == cat]
        if not sel:
            continue
        print("=== %s (%d)" % (cat, len(sel)))
        for g in sel:
            print("  %-12s %-52s main:%d" % (g["decision"] or "UNDECIDED", g["name"], g["main_line"]))
            if g["reason"]:
                print("               %s" % g["reason"])
        print()

    if args.check:
        rc = 0
        if undecided:
            print("FAIL: %d gap item(s) carry no decision in %s" % (
                len(undecided), os.path.basename(LEDGER)), file=sys.stderr)
            for g in undecided:
                print("  %s (main:%d)" % (g["key"], g["main_line"]), file=sys.stderr)
            print("\nClassify each as PORT / PORTED / NA / PHYSICS with a reason.",
                  file=sys.stderr)
            rc = 1
        if stale:
            print("FAIL: %d item(s) marked PORTED are still absent from the LISA driver:"
                  % len(stale), file=sys.stderr)
            for g in stale:
                print("  %s" % g["key"], file=sys.stderr)
            rc = 1
        if rc == 0:
            print("OK: all %d gap items carry a recorded decision" % len(gap))
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
