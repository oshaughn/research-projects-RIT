#! /usr/bin/env python
"""
util_CleanILE_hyperpipeline.py
==============================

Hyperpipeline-format counterpart to ``util_CleanILE.py``.

Reads any number of per-shard hyperpipeline ``.dat`` files (the kind ILE
emits when ``RIFT_HYPERPIPELINE_FORMAT`` is set), consolidates duplicate
intrinsic-parameter rows by weighted averaging of ``lnL`` (mirroring
``util_CleanILE.py``'s logic via :func:`RIFT.misc.hyperpipeline_io.consolidate`),
sorts the result by ``lnL`` descending, and writes a single composite
hyperpipeline file to either stdout or a named file.

Used by ``util_ILEdagPostprocess.sh`` when ``RIFT_HYPERPIPELINE_FORMAT`` is
truthy; the legacy ``cat | util_CleanILE.py | sort -rg`` chain does not work
on the new format because (a) the per-shard headers would interleave with
data under naive concatenation and (b) the legacy positional-column layout
is different.
"""

from __future__ import absolute_import, print_function

import argparse
import sys

# Import via an explicit file load so this script keeps working even when
# the surrounding RIFT package's ``__init__`` chain is partially broken or
# we're running in a minimal venv.  Falls back to the normal package import
# when that's not available.
try:
    from RIFT.misc import hyperpipeline_io as hpio
except Exception:  # pragma: no cover -- best-effort fallback
    import os, importlib.util as _ilu
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidate = os.path.normpath(os.path.join(
        _here, "..", "RIFT", "misc", "hyperpipeline_io.py"))
    _spec = _ilu.spec_from_file_location("hyperpipeline_io", _candidate)
    hpio = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(hpio)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fname", nargs="+",
                        help="One or more hyperpipeline shard .dat files.")
    parser.add_argument("--output", "-o", default="-",
                        help="Output filename, or '-' for stdout (default).")
    parser.add_argument("--sigma-cut", type=float, default=0.9,
                        help="Drop rows with sigma_lnL above this value "
                             "(default 0.9, mirrors util_CleanILE).")
    parser.add_argument("--digits", type=int, default=5,
                        help="Decimal-place precision used when grouping "
                             "duplicate intrinsic rows (default 5, mirrors "
                             "util_CleanILE).")
    parser.add_argument("--no-consolidate", action="store_true",
                        help="Skip the per-key weighted averaging; just "
                             "concatenate, drop bad-sigma rows, sort by lnL "
                             "and emit.  Useful for debugging or when shards "
                             "are already consolidated.")
    args, _unknown = parser.parse_known_args(argv)

    arr, columns = hpio.read_many(args.fname)
    if args.no_consolidate:
        # Apply the same sigma cut + descending-lnL sort the consolidate path
        # does, to keep downstream behaviour symmetric.
        keep = arr["sigma_lnL"] <= args.sigma_cut
        arr = arr[keep]
        order = arr["lnL"].argsort()[::-1]
        arr = arr[order]
    else:
        arr, columns = hpio.consolidate(arr, columns,
                                        sigma_cut=args.sigma_cut,
                                        digits=args.digits)

    # Materialise as a plain (N,K) float matrix for write_table.
    import numpy as _np
    n = len(arr)
    mat = _np.zeros((n, len(columns)), dtype=float)
    for j, name in enumerate(columns):
        mat[:, j] = arr[name]

    if args.output == "-":
        # write_table only takes a path; emit to stdout via a tempfile then
        # cat.  This is N=1 cost, no worse than the legacy `... > file.composite`.
        import tempfile, os as _os
        tmp = tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False).name
        try:
            hpio.write_table(tmp, columns, mat)
            with open(tmp) as fp:
                sys.stdout.write(fp.read())
        finally:
            _os.unlink(tmp)
    else:
        hpio.write_table(args.output, columns, mat)
        sys.stderr.write(
            "util_CleanILE_hyperpipeline: wrote {} consolidated rows "
            "to {}\n".format(n, args.output))


if __name__ == "__main__":
    main()
