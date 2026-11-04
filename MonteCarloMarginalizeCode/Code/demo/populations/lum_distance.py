#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add detector-frame masses and luminosity distance (Planck15) to an injections table.

Usage:
  python add_detector_masses_and_dl.py \
      --input /mnt/data/injections.dat \
      --output /mnt/data/injections_with_det_and_dl.dat

Requires:
  - pandas
  - astropy
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import pandas as pd
from astropy.cosmology import Planck15
import astropy.units as u


def read_table(path: str, comment: str = "#") -> pd.DataFrame:
    """
    Read a whitespace-delimited table with a header row.

    Uses `sep=r"\\s+"` (preferred over deprecated `delim_whitespace=True`).
    """
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            engine="python",
            comment=comment,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to read table '{path}': {e}") from e
    return df


def ensure_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def add_detector_frame_masses(df: pd.DataFrame) -> pd.DataFrame:
    # m_det = m_source * (1 + z)
    df["mass_1_detector"] = df["mass_1_source"] * (1.0 + df["redshift"])
    df["mass_2_detector"] = df["mass_2_source"] * (1.0 + df["redshift"])
    return df


def add_luminosity_distance(df: pd.DataFrame) -> pd.DataFrame:
    # Use Planck15 luminosity distance in Mpc
    z = df["redshift"].to_numpy()
    # astropy returns Quantity; convert to plain float in Mpc
    dl = Planck15.luminosity_distance(z).to(u.Mpc).value
    df["luminosity_distance"] = dl
    return df


def write_table(df: pd.DataFrame, path: str) -> None:
    # Whitespace-delimited, keep reasonable float precision
    try:
        df.to_csv(path, sep=" ", index=False, float_format="%.9e")
    except Exception as e:
        raise RuntimeError(f"Failed to write output '{path}': {e}") from e


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add detector-frame masses and luminosity distance (Planck15)."
    )
    ap.add_argument("--input", required=True, help="Path to input table")
    ap.add_argument(
        "--output",
        required=True,
        help="Path to output table with added columns",
    )
    args = ap.parse_args()

    df = read_table(args.input)
    ensure_columns(df, ["mass_1_source", "mass_2_source", "redshift"])

    df = add_detector_frame_masses(df)
    df = add_luminosity_distance(df)

    write_table(df, args.output)

    # Brief report to stderr
    print(
        f"Added columns: mass_1_detector, mass_2_detector, luminosity_distance\n"
        f"Rows processed: {len(df)}\n"
        f"Wrote: {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

print(Planck15.luminosity_distance(2.3).to(u.Mpc).value)
