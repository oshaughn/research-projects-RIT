#!/usr/bin/env python
"""Consolidate per-event .dgrid (Plan A) or .dslice (Plan B) files.

The ILE extrinsic stage emits one file per intrinsic point (e.g.
``EXTR_out.xml_0_.dgrid``, ``..._1_.dgrid`` ...).  Each is a small ASCII
table with a ``# col1 col2 ...`` header line and one row per distance
sample/slice.  Downstream tools (notably ``util_ConstructEOSPosterior.py``)
want a single concatenated table with one shared header.

This script:
- reads each input file,
- verifies the headers match,
- writes one header line followed by the data rows from all files.

Usage:
    util_ConsolidateDistanceGrids.py --output all_dgrid.dat <files...>
    util_ConsolidateDistanceGrids.py --output all_dgrid.dat --input-glob '*.dgrid'
"""
import argparse
import glob
import os
import sys


def _read_split(fname):
    """Read a file, return (header_line, [data_lines]).

    ``header_line`` is the first line stripped of a leading ``#`` (or None if
    the file has no header).  Blank lines and other comment lines are
    skipped in the data.
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None, []
    header = None
    data = []
    started = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            if not started and header is None:
                header = s.lstrip('#').strip()
            # subsequent comments are ignored
            continue
        started = True
        data.append(line.rstrip('\n'))
    return header, data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs='*', help="input files (.dgrid or .dslice)")
    p.add_argument("--input-glob", default=None,
                   help="glob pattern for inputs (e.g. '*.dgrid')")
    p.add_argument("--output", required=True,
                   help="output consolidated .dat file")
    p.add_argument("--allow-empty", action='store_true',
                   help="exit 0 (writing a header-only output) if no inputs match")
    opts = p.parse_args(argv)

    files = list(opts.inputs)
    if opts.input_glob:
        files += sorted(glob.glob(opts.input_glob))
    # preserve order, drop duplicates
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]
    if not files:
        msg = "no input files provided"
        if opts.allow_empty:
            print("util_ConsolidateDistanceGrids.py: {}; writing empty output".format(msg),
                  file=sys.stderr)
            with open(opts.output, 'w') as f:
                pass
            return 0
        print("util_ConsolidateDistanceGrids.py: ERROR: {}".format(msg),
              file=sys.stderr)
        return 2

    header = None
    all_data = []
    n_files_used = 0
    for fname in files:
        if not os.path.isfile(fname) or os.path.getsize(fname) == 0:
            continue
        h, d = _read_split(fname)
        if h is None:
            print("util_ConsolidateDistanceGrids.py: WARNING: no header in {} "
                  "(skipping)".format(fname), file=sys.stderr)
            continue
        if header is None:
            header = h
        elif h != header:
            print("util_ConsolidateDistanceGrids.py: ERROR: header mismatch in {}\n"
                  "  expected: {}\n  got:      {}".format(fname, header, h),
                  file=sys.stderr)
            return 3
        all_data.extend(d)
        n_files_used += 1

    if header is None:
        if opts.allow_empty:
            with open(opts.output, 'w') as f:
                pass
            return 0
        print("util_ConsolidateDistanceGrids.py: ERROR: no usable inputs found",
              file=sys.stderr)
        return 2

    with open(opts.output, 'w') as f:
        f.write("# " + header + "\n")
        for line in all_data:
            f.write(line + "\n")
    print("util_ConsolidateDistanceGrids.py: wrote {} rows from {} files to {}".format(
        len(all_data), n_files_used, opts.output), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
