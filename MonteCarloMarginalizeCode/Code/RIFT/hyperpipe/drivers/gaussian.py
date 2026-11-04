"""
RIFT.hyperpipe.drivers.gaussian
===============================

Toy 3-D Gaussian marg driver --- canonical "smoke test" for the hyperpipeline.

This is a clean, importable port of the hand-written
``example_gaussian.py`` shipped with the Gaussian hyperpipe demo, built on
top of :class:`RIFT.hyperpipe.drivers.MargDriverBase` so the
boilerplate (arg parsing, grid I/O, output formatting) is shared with all
other drivers.

The likelihood is the sum of two 3-D Gaussians symmetric in ``x``::

    p(x, y, z) = N([-x0, 0, 0], 2 I) + N([+x0, 0, 0], 2 I)

with ``x0 = 4`` and identity covariance scaled by 2.  Useful as both a
pedagogical example and a regression test for the post / puff /
convergence-test pieces, because the exact marginals are known.

Run via the in-tree CLI::

    util_HyperMargGaussian.py \\
        --using-eos file:my_grid.dat --outdir out --fname-output-integral lnL.txt \\
        --eos_start_index 0 --eos_end_index 1000 --conforming-output-name
"""

from __future__ import annotations

import argparse
from typing import Sequence, Tuple

import numpy as np

from .base import MargDriverBase


class GaussianMargDriver(MargDriverBase):
    """Sum-of-two-Gaussians 3-D toy likelihood."""

    description = (
        "Toy 3-D bimodal Gaussian marg driver for the RIFT hyperpipeline."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--x-offset",
            type=float,
            default=4.0,
            help="Position of the two modes along x (default: 4).",
        )
        parser.add_argument(
            "--sigma2",
            type=float,
            default=2.0,
            help="Diagonal of the covariance matrix (default: 2).",
        )
        parser.add_argument(
            "--unimodal",
            action="store_true",
            help="If set, drop the second mode at -x_offset.",
        )
        parser.add_argument(
            "--params",
            type=str,
            default="x,y,z",
            help="Comma-separated names of the three grid columns to use "
                 "(default: x,y,z). Must appear in the input-grid header.",
        )

    # Cache the multivariate normals so we don't rebuild per-row.
    _rv_cache = None

    def _build_rvs(self, opts):
        from scipy.stats import multivariate_normal

        cov = np.eye(3) * float(opts.sigma2)
        rv_pos = multivariate_normal(mean=[+opts.x_offset, 0.0, 0.0], cov=cov)
        rv_neg = (
            None
            if opts.unimodal
            else multivariate_normal(mean=[-opts.x_offset, 0.0, 0.0], cov=cov)
        )
        self._rv_cache = (rv_pos, rv_neg)

    def log_likelihood(
        self,
        row_values: Sequence[str],
        column_names: Sequence[str],
        opts: argparse.Namespace,
    ) -> Tuple[float, float]:
        if self._rv_cache is None:
            self._build_rvs(opts)
        rv_pos, rv_neg = self._rv_cache

        # Resolve column indices -> the user-named param columns
        wanted = [p.strip() for p in opts.params.split(",") if p.strip()]
        if len(wanted) != 3:
            raise SystemExit(
                f"GaussianMargDriver: --params must list exactly 3 names "
                f"(got {wanted!r})."
            )
        try:
            idxs = [column_names.index(p) for p in wanted]
        except ValueError as exc:
            raise SystemExit(
                f"GaussianMargDriver: param {exc.args[0]!r} not in grid header "
                f"columns {list(column_names)!r}."
            ) from exc
        vec = np.array([float(row_values[i]) for i in idxs])
        L = rv_pos.pdf(vec)
        if rv_neg is not None:
            L = L + rv_neg.pdf(vec)
        # Guard log(0); the existing demo does not, so we are *gentle*:
        # report a very-negative lnL but keep finite for the GP fit.
        if L <= 0.0 or not np.isfinite(L):
            return -1.0e6, 1.0e-3
        return float(np.log(L)), 1.0e-3


def main(argv=None) -> str:
    """Console-script entry point."""
    return GaussianMargDriver().run(argv=argv)


if __name__ == "__main__":  # pragma: no cover
    main()
