#!/usr/bin/env python3
"""Sanitize a LIGO_LW XML grid file for modern (igwn_ligolw / ligo.lw) parsers.

The grid placeholders shipped under ``inputs/`` come from older RIFT data
files which carry columns and types that the latest igwn_ligolw rejects --
specifically:

  * ``Type="ilwd:char"`` columns (deprecated in favor of ``int_8s``)
  * ``Name="sim_inspiral:process_id"`` (no longer in the validated column
    list for the sim_inspiral table)

Both of these can be patched in-place without affecting the rows the
pipeline actually reads.  This script does that, preferring the official
``ligo.lw`` (or fallback ``igwn_ligolw``) API and dropping back to a small
text-based fix-up if neither is available.

Usage::

    python3 sanitize_grid.py input.xml.gz output.xml.gz
"""

import gzip
import io
import os
import re
import sys


def _try_with_ligolw(in_path, out_path):
    """Use the proper ligo.lw / igwn_ligolw API to strip deprecated columns."""
    last_err = None
    for modname in ("ligo.lw", "igwn_ligolw"):
        try:
            mod = __import__(modname, fromlist=["lsctables", "utils", "ligolw"])
        except ImportError as e:
            last_err = e
            continue
        try:
            lsctables = mod.lsctables
            ligolw_utils = mod.utils
            # Both packages expose a content handler we can specialise.
            try:
                LIGOLWContentHandler = ligolw_utils.LIGOLWContentHandler
            except AttributeError:
                from ligo.lw.ligolw import LIGOLWContentHandler  # type: ignore
            handler = lsctables.use_in(LIGOLWContentHandler)

            # ligolw_no_ilwdchar.process_document is the canonical helper for
            # the ilwd:char → int_8s migration.  It's safe to skip if the
            # file is already modern.
            try:
                from ligo.lw.utils.ligolw_no_ilwdchar import (
                    process_document as _no_ilwd,  # type: ignore
                )
            except Exception:
                try:
                    from igwn_ligolw.utils.ligolw_no_ilwdchar import (  # type: ignore
                        process_document as _no_ilwd,
                    )
                except Exception:
                    _no_ilwd = None

            xmldoc = ligolw_utils.load_filename(in_path, contenthandler=handler)

            if _no_ilwd is not None:
                try:
                    _no_ilwd(xmldoc)
                except Exception:
                    pass

            # Drop process_id columns from sim_inspiral if igwn rejects them.
            # We do this defensively for sngl_inspiral too because some old
            # files attach process_id there as well.
            for table_cls_name in ("SimInspiralTable", "SnglInspiralTable"):
                cls = getattr(lsctables, table_cls_name, None)
                if cls is None:
                    continue
                try:
                    table = cls.get_table(xmldoc)
                except Exception:
                    continue
                if "process_id" in getattr(table, "columnnames", []):
                    try:
                        col = table.getColumnByName("process_id")
                        table.removeChild(col)
                    except Exception:
                        # Some bindings name the helper differently
                        for child in list(table.childNodes):
                            name = getattr(child, "Name", None) or getattr(child, "name", None)
                            if name and "process_id" in str(name):
                                table.removeChild(child)

            ligolw_utils.write_filename(xmldoc, out_path)
            return True
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        sys.stderr.write(
            "[sanitize_grid] ligo.lw / igwn_ligolw path failed: {!r}; "
            "falling back to text patch.\n".format(last_err)
        )
    return False


def _text_fallback(in_path, out_path):
    """Stdlib-only fix-up for when no ligo.lw / igwn_ligolw library is available.

    For each ``<Table>`` in the document we:

      1. Identify columns to drop entirely (``process_id``, ``simulation_id``)
         and remember their positions.
      2. Identify columns whose ``Type="ilwd:char"`` -- these need their
         declared type rewritten to ``int_8s`` *and* their data values
         (which look like ``"some:thing:N"``) reduced to the bare integer
         ``N``.
      3. Strip the dropped Column declarations and rewrite ilwd:char types.
      4. Walk the comma-separated ``<Stream>`` body row-by-row, deleting
         the dropped column values and rewriting ilwd-format string values.

    This is enough to satisfy modern (igwn_ligolw / ligo.lw) parsers for
    pipeline placeholders.  It does not attempt structural validation; the
    file is treated as a thin XML wrapper around a CSV body.
    """
    import csv as _csv
    import io as _io

    DROP_COLS = {"process_id", "simulation_id"}

    if in_path.endswith(".gz"):
        with gzip.open(in_path, "rb") as fh:
            data = fh.read().decode("utf-8", errors="replace")
    else:
        with open(in_path) as fh:
            data = fh.read()

    table_re = re.compile(
        r"(<Table\b[^>]*>)(.*?)(</Table>)", re.DOTALL | re.IGNORECASE
    )
    column_re = re.compile(
        r'<Column\s+(?:Type="(?P<type>[^"]+)"\s+Name="(?P<name>[^"]+)"|'
        r'Name="(?P<name2>[^"]+)"\s+Type="(?P<type2>[^"]+)")\s*/>',
        re.IGNORECASE,
    )
    stream_re = re.compile(
        r'(<Stream\b[^>]*>)(.*?)(</Stream>)', re.DOTALL | re.IGNORECASE
    )
    ilwd_value_re = re.compile(r'^[A-Za-z_][\w]*:[A-Za-z_][\w]*:(-?\d+)$')

    def _process_table(match):
        head, body, tail = match.group(1), match.group(2), match.group(3)

        # 1) Walk Column declarations in order; record drop / rewrite plan.
        col_specs = []
        for cm in column_re.finditer(body):
            ctype = cm.group("type") or cm.group("type2")
            cname_full = cm.group("name") or cm.group("name2")
            cname_short = cname_full.split(":", 1)[1] if ":" in cname_full else cname_full
            drop = cname_short in DROP_COLS
            col_specs.append({
                "name": cname_short,
                "type": ctype,
                "drop": drop,
                "rewrite_ilwd": (ctype == "ilwd:char") and not drop,
            })

        # 2) Strip dropped Column declarations and rewrite ilwd:char types.
        def _column_repl(m):
            ctype = m.group("type") or m.group("type2")
            cname_full = m.group("name") or m.group("name2")
            cname_short = cname_full.split(":", 1)[1] if ":" in cname_full else cname_full
            if cname_short in DROP_COLS:
                return ""
            if ctype == "ilwd:char":
                return m.group(0).replace('Type="ilwd:char"', 'Type="int_8s"')
            return m.group(0)
        body_new = column_re.sub(_column_repl, body)
        body_new = re.sub(r"\n[ \t]*\n", "\n", body_new)

        # 3) Process the Stream body row-by-row.
        def _stream_repl(sm):
            shead, sbody, stail = sm.group(1), sm.group(2), sm.group(3)
            leading_ws = re.match(r"\s*", sbody).group(0)
            trailing_ws = re.search(r"\s*$", sbody).group(0)
            inner = sbody.strip()
            if not inner:
                return shead + sbody + stail
            reader = _csv.reader(
                _io.StringIO(inner),
                delimiter=",", quotechar='"', skipinitialspace=False,
            )
            rows = list(reader)
            new_rows = []
            for row in rows:
                if len(row) != len(col_specs):
                    new_rows.append(row)
                    continue
                new_row = []
                for value, spec in zip(row, col_specs):
                    if spec["drop"]:
                        continue
                    if spec["rewrite_ilwd"]:
                        m = ilwd_value_re.match(value.strip().strip('"'))
                        if m:
                            value = m.group(1)
                    new_row.append(value)
                new_rows.append(new_row)
            buf = _io.StringIO()
            writer = _csv.writer(
                buf, delimiter=",", quotechar='"', quoting=_csv.QUOTE_MINIMAL,
                lineterminator="\n",
            )
            for row in new_rows:
                writer.writerow(row)
            new_inner = buf.getvalue().rstrip("\r\n")
            return shead + leading_ws + new_inner + trailing_ws + stail

        body_new = stream_re.sub(_stream_repl, body_new)
        return head + body_new + tail

    out = table_re.sub(_process_table, data)
    # Defensive: any straggling ilwd:char outside of a Table.
    out = out.replace('Type="ilwd:char"', 'Type="int_8s"')

    if out_path.endswith(".gz"):
        with gzip.open(out_path, "wb") as fh:
            fh.write(out.encode("utf-8"))
    else:
        with open(out_path, "w") as fh:
            fh.write(out)
    sys.stderr.write(
        "[sanitize_grid] text fallback applied (column declarations + "
        "ilwd:char type+value rewrites + dropped data columns).\n"
    )


def main():
    if len(sys.argv) != 3:
        print("usage: sanitize_grid.py input.xml(.gz) output.xml(.gz)", file=sys.stderr)
        sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        print("input does not exist: {}".format(in_path), file=sys.stderr)
        sys.exit(2)
    if _try_with_ligolw(in_path, out_path):
        print("[sanitize_grid] wrote {} via ligo.lw / igwn_ligolw".format(out_path))
        return
    _text_fallback(in_path, out_path)
    print("[sanitize_grid] wrote {} via text fallback".format(out_path))


if __name__ == "__main__":
    main()
