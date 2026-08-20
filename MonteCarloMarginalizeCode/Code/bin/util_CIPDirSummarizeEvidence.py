#! /usr/bin/env python

"""Consolidate independent CIP evidence estimates.

The historical two-column output is preserved. A terminal pipeline job may
also provide an independently computed L=1 CIP integral; in that case a second
file reports the prior-normalized evidence (Bayes factor for the hypothesis).
"""

import argparse
import glob
import os
import sys

import numpy as np


def _read_scalar_record(fname, required_fields):
    """Read one named, scalar annotation record and validate its fields."""
    try:
        record = np.genfromtxt(fname, names=True)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read {}: {}".format(fname, exc))
    if record.dtype.names is None:
        raise ValueError("{} has no named columns".format(fname))
    missing = [name for name in required_fields if name not in record.dtype.names]
    if missing:
        raise ValueError("{} is missing columns {}".format(fname, missing))
    values = {name: float(np.asarray(record[name]).reshape(-1)[0])
              for name in required_fields}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("{} contains non-finite evidence data".format(fname))
    return values


def find_worker_annotations(cip_dir, cip_prefix=None):
    """Return annotations for exploded workers or one non-exploded CIP run."""
    if cip_prefix is not None:
        fname = cip_prefix + "+annotation.dat"
        return [fname] if os.path.isfile(fname) else []
    pattern = os.path.join(cip_dir, "overlap-grid-*-*[0-9]+annotation.dat")
    return sorted(glob.glob(pattern))


def consolidate_cip_directory(cip_dir, strict=False, cip_prefix=None):
    """Combine worker lnZ values using the established RIFT prescription."""
    base_files = find_worker_annotations(cip_dir, cip_prefix=cip_prefix)
    if not base_files:
        message = "No files for evidence in {}".format(cip_dir)
        if strict:
            raise ValueError(message)
        print(message)
        return None

    rows = []
    for base_name in base_files:
        alt_name = base_name.replace("+annotation.dat",
                                     "_withpriorchange+annotation.dat")
        base = _read_scalar_record(base_name, ("sigmaL",))
        alt = _read_scalar_record(alt_name, ("lnL", "neff"))
        if base["sigmaL"] <= 0:
            raise ValueError("{} has non-positive sigmaL".format(base_name))
        rows.append((alt["lnL"], base["sigmaL"], alt["neff"]))

    net = np.asarray(rows, dtype=float)
    ln_z = np.average(net[:, 0], weights=1.0 / net[:, 1] ** 2)
    sigma_ln_z = max(np.sqrt(np.mean(net[:, 1] ** 2) / len(net)),
                     np.std(net[:, 0]))
    return {
        "lnZ": float(ln_z),
        "sigma_lnZ": float(sigma_ln_z),
        "n_workers": len(rows),
    }


def read_prior_integral(fname):
    """Read the target-prior L=1 integral produced by a dedicated CIP run."""
    prior = _read_scalar_record(fname, ("lnL", "sigmaL", "neff"))
    if prior["sigmaL"] < 0:
        raise ValueError("{} has negative sigmaL".format(fname))
    return {
        "ln_prior": prior["lnL"],
        "sigma_ln_prior": prior["sigmaL"],
        "prior_neff": prior["neff"],
    }


def normalized_evidence(evidence, prior):
    """Return ln B_H = ln Z_H - ln integral(prior), with MC errors."""
    return {
        "lnB": evidence["lnZ"] - prior["ln_prior"],
        "sigma_lnB": np.hypot(evidence["sigma_lnZ"],
                               prior["sigma_ln_prior"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cip-dir", required=True, help="CIP directory")
    parser.add_argument("--cip-prefix", default=None,
                        help="non-exploded CIP output prefix (without .dat)")
    parser.add_argument("--output", default="evidence.out")
    parser.add_argument("--stream-output", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="fail if worker evidence is absent or malformed")
    parser.add_argument("--prior-integral", default=None,
                        help="L=1 CIP annotation for the target prior")
    parser.add_argument("--normalized-output", default=None,
                        help="write prior-normalized evidence to this file")
    # Retained for command-line compatibility. Modern CIP annotations are
    # already in log space; the old switch never altered this utility's result.
    parser.add_argument("--internal-fix-double-log", action="store_true",
                        help=argparse.SUPPRESS)
    opts = parser.parse_args(argv)

    try:
        evidence = consolidate_cip_directory(
            opts.cip_dir, strict=opts.strict, cip_prefix=opts.cip_prefix)
        if evidence is None:
            return 0
        legacy = np.array([[evidence["lnZ"], evidence["sigma_lnZ"]]])
        if opts.stream_output:
            print(*legacy[0])
        else:
            np.savetxt(opts.output, legacy, header=" lnL sigma_lnL")

        if opts.prior_integral:
            if not opts.normalized_output:
                raise ValueError("--prior-integral requires --normalized-output")
            prior = read_prior_integral(opts.prior_integral)
            norm = normalized_evidence(evidence, prior)
            row = np.array([[
                evidence["lnZ"], evidence["sigma_lnZ"],
                prior["ln_prior"], prior["sigma_ln_prior"],
                norm["lnB"], norm["sigma_lnB"],
                evidence["n_workers"], prior["prior_neff"],
            ]])
            np.savetxt(
                opts.normalized_output, row,
                header=(" lnZ sigma_lnZ ln_prior sigma_ln_prior "
                        "lnB_H sigma_lnB_H n_workers prior_neff"),
            )
    except ValueError as exc:
        print("Evidence consolidation failed: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
