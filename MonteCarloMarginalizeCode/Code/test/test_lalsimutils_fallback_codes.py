"""Approximant fallback codes in lalsimutils must be distinct.

On a LALSuite lacking an approximant, lalsimutils binds it to a negative
sentinel.  Two equal sentinels make hlmoft route one model by another's
branch (issue #488: XPNR at -19 matched check_FD_pending).  The sentinels are
literals in the source, so they are read statically: the check holds whatever
LALSuite is installed.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "RIFT" / "lalsimutils.py"


def _neg_int(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
            and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
        return -node.operand.value
    return None


def fallback_codes():
    tree = ast.parse(SRC.read_text())
    codes = {}  # name -> set of sentinels
    pending_start = pending_names = None
    for node in tree.body:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Assign) or len(sub.targets) != 1 \
                    or not isinstance(sub.targets[0], ast.Name):
                continue
            name, val = sub.targets[0].id, sub.value
            if name == "pending_default_code":
                pending_start = _neg_int(val)
                continue
            if name == "pending_FD_approx":
                pending_names = [e.value for e in val.elts]
                continue
            if not name.startswith("lal"):
                continue
            code = _neg_int(val)
            if code is None and isinstance(val, ast.Call) \
                    and getattr(val.func, "id", None) == "getattr" and len(val.args) == 3:
                code = _neg_int(val.args[2])
            if code is not None:
                codes.setdefault(name, set()).add(code)
    assert pending_start is not None and pending_names
    for i, name in enumerate(pending_names):
        codes.setdefault("pending:" + name, set()).add(pending_start - i)
    return codes


def test_fallback_codes_found():
    codes = fallback_codes()
    assert "lalIMRPhenomXPNR" in codes and "lalIMRPhenomTHM" in codes
    assert "pending:IMRPhenomXP_NRTidalv2" in codes


def test_fallback_codes_distinct():
    owners = {}
    for name, vals in fallback_codes().items():
        assert len(vals) == 1, f"{name} has several sentinels {vals}"
        owners.setdefault(vals.pop(), []).append(name)
    shared = {c: n for c, n in owners.items() if len(n) > 1}
    assert not shared, f"fallback codes shared: {shared}"
