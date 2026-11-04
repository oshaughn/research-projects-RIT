"""Generate RIFT/likelihood/gpu_waveform_xhm_fits.py from LALSimulation C sources.

Usage:  python -I gen_gpu_waveform_xhm_fits.py LALSIM_LIB_DIR OUT.py

Translates the QNM fits of LALSimIMRPhenomXHM_qnm.c and every parameter-space fit ``static double NAME(IMRPhenomXWaveformStruct *pWF,
int Flag)`` in LALSimIMRPhenomXHM_{inspiral,intermediate,ringdown}.c (tag
lalsimulation-v6.2.0) into an elementwise NumPy function ``NAME(w, flag)``.  The
fits contain only declarations, assignments and a switch on the version flag,
so the translation is mechanical: ``pWF->x`` becomes ``w.x``, ``pow``/``fabs``/
``sqrt`` become their NumPy forms, and each ``case`` becomes an ``if`` branch.
"""

import re
import sys

FILES = ("LALSimIMRPhenomXHM_inspiral.c", "LALSimIMRPhenomXHM_intermediate.c",
         "LALSimIMRPhenomXHM_ringdown.c")
SIG = re.compile(r"^static double (IMRPhenomXHM_\w+)\((?:UNUSED )?IMRPhenomXWaveformStruct \*pWF, int (\w+)\)\s*\{?")


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def function_bodies(text):
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = SIG.match(lines[i])
        if not m:
            i += 1
            continue
        name, flag = m.group(1), m.group(2)
        body = [lines[i][m.end():]]
        depth = lines[i].count("{") - lines[i].count("}")
        i += 1
        while depth > 0 or not any("{" in b for b in body):
            body.append(lines[i])
            depth += lines[i].count("{") - lines[i].count("}")
            i += 1
        yield name, flag, "\n".join(body)


def split_top(s, sep=","):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def expr(e):
    e = e.strip()
    e = re.sub(r"pWF->(\w+)", r"w.\1", e)
    e = re.sub(r"\bpow\(", "_pow(", e)
    e = re.sub(r"\bfabs\(", "np.abs(", e)
    for f in ("sqrt", "exp", "log", "tanh", "atan", "cbrt"):
        e = re.sub(r"(?<![\w.])%s\(" % f, "np.%s(" % f, e)
    return e


def statements(code):
    out = []
    for st in code.split(";"):
        st = st.strip().strip("{}").strip()
        if not st or st in ("break",):
            continue
        st = re.sub(r"^\{", "", st).strip()
        if st.startswith("default"):
            continue
        m = re.match(r"^(?:const\s+)?(?:double|REAL8|int|INT4)\s+(.*)$", st, flags=re.S)
        if m:
            for part in split_top(m.group(1)):
                if "=" in part:
                    lhs, rhs = part.split("=", 1)
                    out.append("%s = %s" % (lhs.strip(), expr(rhs)))
            continue
        if "=" in st and not re.match(r"return\b", st):
            lhs, rhs = st.split("=", 1)
            out.append("%s = %s" % (lhs.strip(), expr(rhs)))
            continue
        if re.match(r"return\b", st):
            continue
        raise ValueError("cannot translate: %r" % st)
    return out


def translate(name, flag, body):
    body = strip_comments(body)
    sw = body.index("switch")
    pre = body[:sw]
    rest = body[sw:]
    rest = rest[rest.index("{") + 1:]
    end = rest.rfind("return")
    cases = re.split(r"\bcase\s+(\d+)\s*:", rest[:end])
    lines = ["def %s(w, %s):" % (name, flag)]
    pre_st = [s for s in statements(pre.replace("{", ""))]
    lines += ["    " + s for s in pre_st]
    pending = []
    first = True
    for k in range(1, len(cases), 2):
        ver, code = cases[k], cases[k + 1]
        code = code.split("default")[0]
        pending.append(ver)
        sts = statements(code)
        if not sts:
            continue
        cond = " or ".join("%s == %s" % (flag, v) for v in pending)
        lines.append("    %s %s:" % ("if" if first else "elif", cond))
        lines += ["        " + s for s in sts]
        first = False
        pending = []
    lines.append("    else:")
    lines.append("        raise ValueError('%s: version %%s is not valid' %% %s)" % (name, flag))
    lines.append("    return total")
    return "\n".join(lines)


QNM = re.compile(r"^double (evaluate_QNMfit_\w+)\(double finalDimlessSpin\)\{")


def qnm_fits(text):
    """Translate the QNM and mixing-coefficient fits of LALSimIMRPhenomXHM_qnm.c."""
    text = strip_comments(text)
    out = []
    for m in re.finditer(r"^double (evaluate_QNMfit_\w+)\(double finalDimlessSpin\)\{(.*?)\n\}", text,
                         flags=re.S | re.M):
        name, body = m.group(1), m.group(2)
        body = re.sub(r"if \(fabs\(finalDimlessSpin\) > 1.0\) \{.*?\}", "", body, flags=re.S)
        body = body.replace("return return_val;", "")
        lines = ["def %s(finalDimlessSpin):" % name]
        lines += ["    " + st for st in statements(body)]
        lines.append("    return return_val")
        out.append((name, "\n".join(lines)))
    return out


def main(libdir, out):
    chunks = []
    names = []
    for name, code in qnm_fits(open("%s/LALSimIMRPhenomXHM_qnm.c" % libdir).read()):
        chunks.append(code)
        names.append(name)
    for f in FILES:
        text = open("%s/%s" % (libdir, f)).read()
        for name, flag, body in function_bodies(text):
            chunks.append(translate(name, flag, body))
            names.append(name)
    header = '''"""Parameter-space fits of IMRPhenomXHM (AUTO-GENERATED; do not edit).

Generated by test/waveforms/gen_gpu_waveform_xhm_fits.py from
LALSimIMRPhenomXHM_inspiral.c, LALSimIMRPhenomXHM_intermediate.c and
LALSimIMRPhenomXHM_ringdown.c and LALSimIMRPhenomXHM_qnm.c at tag lalsimulation-v6.2.0 (Copyright (C) 2019-2022
Cecilio Garcia-Quiros, Marta Colleoni, Sascha Husa and the LALSuite authors).
The fit coefficients are copied from those files, so this file is distributed under
the GNU General Public License, version 2 or later (GPL-2.0-or-later).

Each function is elementwise in ``w`` (a namespace of host NumPy arrays).
"""

import numpy as np


def _pow(x, y):
    return np.power(x, y)

'''
    with open(out, "w") as fh:
        fh.write(header + "\n\n" + "\n\n\n".join(chunks) + "\n\n\nFITS = {%s}\n" % ", ".join(
            "%r: %s" % (n, n) for n in names))
    print("wrote %d fits to %s" % (len(names), out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
