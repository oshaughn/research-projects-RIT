#!/usr/bin/env python

"""Lightweight diagnostics for LISA-RIFT run products."""

from __future__ import print_function

from argparse import ArgumentParser
import json
import os

import numpy as np


LISA_ILE_COLUMNS = (
    "event",
    "m1",
    "m2",
    "s1x",
    "s1y",
    "s1z",
    "s2x",
    "s2y",
    "s2z",
    "ecliptic_longitude",
    "ecliptic_latitude",
    "lnL",
    "sigma_lnL",
    "n_total",
    "n_eff",
)


def load_lisa_ile_table(path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != len(LISA_ILE_COLUMNS):
        raise ValueError(
            "Expected {} LISA ILE columns in {}, found {}".format(
                len(LISA_ILE_COLUMNS), path, data.shape[1]
            )
        )
    return data


def component_masses_to_mc_eta(m1, m2):
    total = m1 + m2
    eta = (m1 * m2) / total**2
    mc = (m1 * m2) ** (3.0 / 5.0) / total ** (1.0 / 5.0)
    return mc, eta


def summarize_lisa_ile(path, lnL_window=15.0, error_threshold=0.4):
    data = load_lisa_ile_table(path)
    lnL = data[:, LISA_ILE_COLUMNS.index("lnL")]
    sigma = data[:, LISA_ILE_COLUMNS.index("sigma_lnL")]
    finite = np.isfinite(lnL)
    if not np.any(finite):
        raise ValueError("No finite lnL values in {}".format(path))

    max_index = int(np.nanargmax(lnL))
    max_lnL = float(lnL[max_index])
    high = finite & (lnL >= max_lnL - lnL_window)
    high_low_error = high & (sigma <= error_threshold)
    m1 = data[:, LISA_ILE_COLUMNS.index("m1")]
    m2 = data[:, LISA_ILE_COLUMNS.index("m2")]
    mc, eta = component_masses_to_mc_eta(m1, m2)

    return {
        "path": os.path.abspath(path),
        "n_rows": int(data.shape[0]),
        "max_lnL": max_lnL,
        "max_index": max_index,
        "high_lnL_points": int(np.count_nonzero(high)),
        "high_lnL_low_error_points": int(np.count_nonzero(high_low_error)),
        "best": {
            "m1": float(m1[max_index]),
            "m2": float(m2[max_index]),
            "mc": float(mc[max_index]),
            "eta": float(eta[max_index]),
            "ecliptic_longitude": float(data[max_index, LISA_ILE_COLUMNS.index("ecliptic_longitude")]),
            "ecliptic_latitude": float(data[max_index, LISA_ILE_COLUMNS.index("ecliptic_latitude")]),
            "sigma_lnL": float(sigma[max_index]),
            "n_eff": float(data[max_index, LISA_ILE_COLUMNS.index("n_eff")]),
        },
    }


def build_parser():
    parser = ArgumentParser(description="Summarize LISA-RIFT ILE/all.net-style output.")
    parser.add_argument("path", help="LISA ILE output, for example lisa_ile_0_.dat or all.net.")
    parser.add_argument("--lnL-window", type=float, default=15.0)
    parser.add_argument("--error-threshold", type=float, default=0.4)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv=None):
    opts = build_parser().parse_args(argv)
    summary = summarize_lisa_ile(
        opts.path,
        lnL_window=opts.lnL_window,
        error_threshold=opts.error_threshold,
    )
    if opts.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print("LISA RIFT diagnostic summary")
    print("  file: {}".format(summary["path"]))
    print("  rows: {}".format(summary["n_rows"]))
    print("  max lnL: {:.6g}".format(summary["max_lnL"]))
    print(
        "  high-lnL points within window: {} ({} with sigma <= {})".format(
            summary["high_lnL_points"],
            summary["high_lnL_low_error_points"],
            opts.error_threshold,
        )
    )
    best = summary["best"]
    print(
        "  best: m1={m1:.6g} m2={m2:.6g} mc={mc:.6g} eta={eta:.6g} "
        "lambda={ecliptic_longitude:.6g} beta={ecliptic_latitude:.6g} "
        "sigma={sigma_lnL:.6g} neff={n_eff:.6g}".format(**best)
    )


if __name__ == "__main__":
    main()
