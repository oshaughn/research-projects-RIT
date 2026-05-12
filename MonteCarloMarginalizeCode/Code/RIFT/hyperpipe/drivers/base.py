"""
RIFT.hyperpipe.drivers.base
===========================

Boilerplate-free base class for marg drivers.

The single hard contract a marg driver has with the hyperpipeline is
described in :mod:`RIFT.hyperpipe.drivers`. The work the existing
hand-written drivers had to repeat --- parse a fixed set of CLI args,
sniff the input-grid header for column names, slice the rows by index
range, and write a header-prefixed ``+annotation.dat`` output --- is
encapsulated here so a concrete driver becomes "implement
``log_likelihood`` and call ``MargDriverBase.run()``".
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# CLI parsing
# --------------------------------------------------------------------------


def make_marg_driver_parser(description: Optional[str] = None) -> argparse.ArgumentParser:
    """Construct an ``argparse.ArgumentParser`` with the standard marg-driver flags.

    Concrete drivers can extend this by adding their own arguments to the
    returned parser before calling ``.parse_args()``.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--fname", type=str, default=None,
                   help="Dummy passthrough required by the hyperpipe API.")
    p.add_argument("--using-eos", type=str, required=True,
                   help="Input hyperparameter grid. 'file:<path>' prefix is tolerated.")
    p.add_argument("--using-eos-index", type=int, default=None,
                   help="If set, evaluate this single row (overrides start/end).")
    p.add_argument("--eos_start_index", type=int, default=None,
                   help="First row of the grid to evaluate (inclusive).")
    p.add_argument("--eos_end_index", type=int, default=None,
                   help="Last row of the grid to evaluate (exclusive).")
    p.add_argument("--n-events-to-analyze", type=int, default=None,
                   help="When --using-eos-index is set, evaluate this many rows in total.")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory. Created if absent. Default: cwd.")
    p.add_argument("--outdir-clean", action="store_true",
                   help="If set, delete --outdir before writing.")
    p.add_argument("--fname-output-integral", type=str, default="output-marg-integral",
                   help="Base name for the lnL output file.")
    p.add_argument("--fname-output-samples", type=str, default="output-marg-samples",
                   help="Dummy passthrough required by the hyperpipe API.")
    p.add_argument("--conforming-output-name", action="store_true",
                   help="Append '+annotation.dat' to the integral output name "
                        "(required by create_eos_posterior_pipeline).")
    return p


def parse_marg_driver_args(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Run argparse and normalize the index range.

    Resolves the four overlapping ways the hyperpipeline can ask for an
    index range (``--using-eos-index`` alone, ``--using-eos-index`` plus
    ``--n-events-to-analyze``, or ``--eos_start_index``/``--eos_end_index``)
    into a single ``(eos_start_index, eos_end_index)`` pair.
    """
    opts = parser.parse_args(argv)
    if opts.using_eos_index is not None:
        opts.eos_start_index = opts.using_eos_index
        opts.eos_end_index = opts.using_eos_index + 1
        if opts.n_events_to_analyze:
            opts.eos_end_index = opts.using_eos_index + int(opts.n_events_to_analyze)
    if opts.eos_start_index is None or opts.eos_end_index is None:
        raise SystemExit(
            "marg driver: need either --using-eos-index, or both "
            "--eos_start_index and --eos_end_index."
        )
    if opts.eos_end_index <= opts.eos_start_index:
        raise SystemExit(
            f"marg driver: eos_end_index ({opts.eos_end_index}) must exceed "
            f"eos_start_index ({opts.eos_start_index})."
        )
    return opts


# --------------------------------------------------------------------------
# Grid I/O
# --------------------------------------------------------------------------


def _strip_file_prefix(p: str) -> str:
    return p[5:] if p.startswith("file:") else p


def read_grid(using_eos: str) -> Tuple[np.ndarray, List[str]]:
    """Load a hyperpipe-format grid file.

    Returns ``(rows, column_names)`` where ``rows`` is a 2-D string array
    (so we don't lose precision on the lnL / sigma columns that we will
    overwrite) and ``column_names`` is the list of column names beyond
    the two leading ``lnL  sigma_lnL`` columns.
    """
    path = _strip_file_prefix(using_eos)
    with open(path, "r") as f:
        header = f.readline().rstrip("\n")
    if not header.startswith("#"):
        raise ValueError(
            f"read_grid: expected a '#'-prefixed header in {path!r}; got {header!r}."
        )
    cols = header.lstrip("#").split()
    if len(cols) < 2:
        raise ValueError(
            f"read_grid: header {header!r} must declare at least lnL and sigma_lnL."
        )
    data = np.genfromtxt(path, dtype="str")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Upcast to a wide string dtype so we can rewrite the leading
    # lnL / sigma_lnL columns with arbitrarily-precise float reprs
    # without numpy silently truncating. Without this step,
    # ``rows[i, 0] = repr(some_float)`` would be truncated to whatever
    # the longest string in the input grid happened to be (this was a
    # latent bug in the hand-written example_gaussian.py).
    data = data.astype("<U32")
    return data, cols[2:]


def write_marg_output(
    rows: np.ndarray,
    column_names: Sequence[str],
    *,
    fname_output_integral: str,
    outdir: Optional[str],
    fname: Optional[str],
    conforming_output_name: bool,
) -> str:
    """Write the annotated output file.

    Returns the path written. Mirrors the convention used by
    ``example_gaussian.py`` and the NICER drivers:

        * if ``--fname`` is *not* set, write into ``outdir/`` and use the
          ``--fname-output-integral`` basename;
        * if ``--fname`` *is* set (legacy RIFT-style invocation), write
          to the bare ``--fname-output-integral`` path;
        * if ``--conforming-output-name`` is set, suffix ``+annotation.dat``.
    """
    postfix = "+annotation.dat" if conforming_output_name else ""
    if fname is None:
        outdir = outdir or "."
        Path(outdir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(outdir, fname_output_integral + postfix)
    else:
        out_path = fname_output_integral + postfix
    header = "lnL     sigma_lnL   " + " ".join(column_names)
    np.savetxt(out_path, rows, fmt="%10s", header=header)
    return out_path


# --------------------------------------------------------------------------
# Driver base class
# --------------------------------------------------------------------------


class MargDriverBase:
    """Base class for marg drivers.

    Subclasses implement :meth:`log_likelihood` and (optionally) override
    :meth:`add_arguments` to declare driver-specific CLI flags, then call
    :meth:`run` from their ``__main__`` block.

    Example
    -------
    >>> class MyDriver(MargDriverBase):
    ...     description = "My toy driver."
    ...     def log_likelihood(self, row_values, column_names, opts):
    ...         return -0.5 * sum(float(v) ** 2 for v in row_values), 1e-3
    ...
    >>> if __name__ == "__main__":   # doctest: +SKIP
    ...     MyDriver().run()
    """

    description: Optional[str] = None

    # ----- subclass hooks ------------------------------------------------

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add driver-specific CLI args; default is no-op."""
        pass

    def log_likelihood(
        self,
        row_values: Sequence[str],
        column_names: Sequence[str],
        opts: argparse.Namespace,
    ) -> Tuple[float, float]:
        """Return ``(lnL, sigma_lnL)`` for one grid row.

        ``row_values`` are the raw string values from columns 2.. of the
        grid (i.e. excluding the two lnL columns we'll overwrite).
        ``column_names`` are the matching column names from the header.
        Concrete drivers will typically build a ``dict(zip(column_names,
        map(float, row_values)))`` before computing the likelihood.
        """
        raise NotImplementedError

    # ----- main entry point ---------------------------------------------

    def run(self, argv: Optional[Sequence[str]] = None) -> str:
        """Parse args, evaluate over the requested row range, write output.

        Returns the path of the written annotated file.
        """
        parser = make_marg_driver_parser(description=self.description)
        self.add_arguments(parser)
        opts = parse_marg_driver_args(parser, argv=argv)

        if opts.outdir_clean and opts.outdir:
            import shutil

            try:
                shutil.rmtree(opts.outdir)
            except FileNotFoundError:
                pass

        rows, column_names = read_grid(opts.using_eos)
        logger.info("Loaded grid %r with columns %r", opts.using_eos, column_names)

        start, stop = opts.eos_start_index, opts.eos_end_index
        if start < 0 or stop > rows.shape[0]:
            raise SystemExit(
                f"marg driver: index range [{start},{stop}) exceeds grid "
                f"size {rows.shape[0]}."
            )

        for i in range(start, stop):
            row_values = rows[i, 2:]
            lnL, sigma = self.log_likelihood(row_values, column_names, opts)
            rows[i, 0] = repr(float(lnL))
            rows[i, 1] = repr(float(sigma))

        out_path = write_marg_output(
            rows[start:stop],
            column_names,
            fname_output_integral=opts.fname_output_integral,
            outdir=opts.outdir,
            fname=opts.fname,
            conforming_output_name=opts.conforming_output_name,
        )
        logger.info("Wrote marg output to %r", out_path)
        return out_path
