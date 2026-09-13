#!/usr/bin/env python3
"""Join JAX-ILE tabular fair-draw sidecars into a RIFT posterior table.

The JAX driver writes one likelihood record ending in ``_.dat`` and one
``_samples.dat`` sidecar per intrinsic point.  Unlike conventional
ILE, it does not write fair draws into LIGO-LW XML, so the XML converter cannot
construct the terminal posterior.  This utility pairs the two tabular products
strictly and emits only coordinates that the JAX driver actually exports.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


HEADER = (
    "m1 m2 a1x a1y a1z a2x a2y a2z mc eta ra dec phiorb incl psi "
    "distance Npts lnL p ps neff mtotal q chi_eff chi_p"
)


def _paired_record(sample_path: Path) -> Path:
    return sample_path.with_name(sample_path.name.replace("_samples.dat", "_.dat"))


def _grid_index(path):
    match = re.match(r"EXTR_out-(\d+)\.xml_(\d+)_(?:samples)?\.dat$", path.name)
    if match is None:
        raise RuntimeError("cannot recover grid index from {}".format(path))
    return int(match.group(1)) + int(match.group(2))


def _spin_summaries(m1, m2, spins):
    """Match the conventional LI chi_eff/chi_p definitions."""
    s1 = np.asarray(spins[:3])
    s2 = np.asarray(spins[3:])
    chi_eff = (m1 * s1[2] + m2 * s2[2]) / (m1 + m2)
    if m2 > m1:
        m1, m2, s1, s2 = m2, m1, s2, s1
    q = m2 / m1
    a1 = 2.0 + 1.5 * q
    a2 = 2.0 + 1.5 / q
    s1_perp = m1 ** 2 * np.linalg.norm(s1[:2])
    s2_perp = m2 ** 2 * np.linalg.norm(s2[:2])
    chi_p = max(a1 * s1_perp, a2 * s2_perp) / (a1 * m1 ** 2)
    return chi_eff, chi_p


def assemble(directory, draws_per_intrinsic=None, expected_intrinsic=None):
    """Return the joined table, refusing incomplete or malformed pairs."""
    samples = sorted(directory.glob("EXTR_out-*_samples.dat"))
    if not samples:
        raise RuntimeError("no JAX-ILE fair-draw sidecars found in {}".format(directory))

    records = {
        path for path in directory.glob("EXTR_out-*_.dat")
        if not path.name.endswith("_samples.dat")
    }
    paired_records = {_paired_record(path) for path in samples}
    if records != paired_records:
        missing_samples = sorted(str(path) for path in records - paired_records)
        missing_records = sorted(str(path) for path in paired_records - records)
        raise RuntimeError("unpaired JAX-ILE products: records_without_samples={}, "
                           "samples_without_records={}".format(
                               missing_samples, missing_records))
    indices = {_grid_index(path) for path in samples}
    if len(indices) != len(samples):
        raise RuntimeError("duplicate intrinsic grid indices in fair-draw sidecars")
    if expected_intrinsic is not None and indices != set(range(expected_intrinsic)):
        missing = sorted(set(range(expected_intrinsic)) - indices)
        excess = sorted(indices - set(range(expected_intrinsic)))
        raise RuntimeError("intrinsic grid is incomplete: expected={}, found={}, "
                           "missing={}, excess={}".format(
                               expected_intrinsic, len(indices), missing, excess))

    rows = []
    seen_records = set()
    for sample_path in samples:
        record_path = _paired_record(sample_path)
        if not record_path.is_file():
            raise FileNotFoundError(record_path)
        if record_path in seen_records:
            raise RuntimeError("duplicate intrinsic record pairing: {}".format(record_path))
        seen_records.add(record_path)

        intrinsic = np.loadtxt(record_path, comments="#", ndmin=2)
        if intrinsic.shape != (1, 13) or not np.isfinite(intrinsic).all():
            raise RuntimeError("invalid intrinsic likelihood record {}: {}".format(
                record_path, intrinsic.shape))

        extrinsic = np.atleast_1d(np.genfromtxt(sample_path, names=True))
        names = set(extrinsic.dtype.names or ())
        required = {
            "right_ascension", "declination", "distance", "inclination", "psi",
            "phi_orb", "loglikelihood",
        }
        if not required <= names:
            raise RuntimeError("missing fair-draw columns in {}: {}".format(
                sample_path, sorted(required - names)))
        if draws_per_intrinsic is not None and len(extrinsic) != draws_per_intrinsic:
            raise RuntimeError("expected {} rows in {}, found {}".format(
                draws_per_intrinsic, sample_path, len(extrinsic)))

        values = intrinsic[0]
        m1, m2 = values[1], values[2]
        mtotal = m1 + m2
        eta = m1 * m2 / mtotal**2
        mc = (m1 * m2) ** 0.6 / mtotal**0.2
        q = min(m1, m2) / max(m1, m2)
        chi_eff, chi_p = _spin_summaries(m1, m2, values[3:9])
        for sample in extrinsic:
            row = np.array([
                *values[1:9], mc, eta,
                sample["right_ascension"], sample["declination"], sample["phi_orb"],
                sample["inclination"], sample["psi"], sample["distance"],
                values[11], sample["loglikelihood"], 1.0, 1.0, values[12],
                mtotal, q, chi_eff, chi_p,
            ], dtype=float)
            if not np.isfinite(row).all():
                raise RuntimeError("nonfinite joined fair draw from {}".format(sample_path))
            rows.append(row)

    if not rows:
        raise RuntimeError("JAX-ILE sidecars contained no fair draws")
    return np.vstack(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_source_sha256(directory):
    digest = hashlib.sha256()
    paths = sorted(set(directory.glob("EXTR_out-*_.dat")) |
                   set(directory.glob("EXTR_out-*_samples.dat")))
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws-per-intrinsic", type=int)
    parser.add_argument("--expected-intrinsic", type=int,
                        help="require exactly contiguous grid IDs 0..N-1")
    parser.add_argument("--shuffle-seed", type=int, default=1986)
    parser.add_argument("--provenance", type=Path,
                        help="JSON ledger (default: OUTPUT.provenance.json)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = args.provenance
    if provenance is None:
        provenance = args.output.with_suffix(".provenance.json")
    provenance.parent.mkdir(parents=True, exist_ok=True)
    for stale in (args.output, provenance):
        if stale.exists():
            stale.unlink()

    table = assemble(args.directory, args.draws_per_intrinsic,
                     args.expected_intrinsic)
    np.random.RandomState(args.shuffle_seed).shuffle(table)
    output_fd, output_name = tempfile.mkstemp(
        prefix=args.output.name + ".tmp.", dir=str(args.output.parent))
    provenance_fd, provenance_name = tempfile.mkstemp(
        prefix=provenance.name + ".tmp.", dir=str(provenance.parent))
    os.close(output_fd)
    os.close(provenance_fd)
    output_tmp = Path(output_name)
    provenance_tmp = Path(provenance_name)
    try:
        np.savetxt(output_tmp, table, header=HEADER, comments="# ", fmt="%.12g")
        payload = {
            "converter": str(Path(__file__).resolve()),
            "converter_sha256": _sha256(Path(__file__).resolve()),
            "source_directory": str(args.directory.resolve()),
            "combined_source_sha256": _combined_source_sha256(args.directory),
            "intrinsic_points": len(list(args.directory.glob(
                "EXTR_out-*_samples.dat"))),
            "draws_per_intrinsic": args.draws_per_intrinsic,
            "expected_intrinsic": args.expected_intrinsic,
            "posterior_rows": len(table),
            "shuffle_seed": args.shuffle_seed,
            "output_sha256": _sha256(output_tmp),
            "equal_weight_columns": {"p": 1.0, "ps": 1.0},
            "omitted_unavailable_coordinates": [
                "time", "redshift", "source_frame_masses"],
        }
        provenance_tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(str(provenance_tmp), str(provenance))
        os.replace(str(output_tmp), str(args.output))
    finally:
        for temporary in (output_tmp, provenance_tmp):
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    main()
