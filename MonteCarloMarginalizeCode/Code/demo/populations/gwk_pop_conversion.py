#!/usr/bin/env python3
"""
Convert .dat files with GW parameters into a new format.

Renames columns:

Detector-frame:
    m1  -> mass_1_detector
    m2  -> mass_2_detector

Spin components:
    a1x -> spin_1x
    a1y -> spin_1y
    a1z -> spin_1z
    a2x -> spin_2x
    a2y -> spin_2y
    a2z -> spin_2z

Source-frame:
    m1_source -> mass_1_source
    m2_source -> mass_2_source

Derived spin quantities added:
    a_1, a_2, cos_tilt_1, cos_tilt_2

Optional assumptions:
    --assume-aligned   treat spins as purely aligned with L
    --n-rows           random downsample rows
    --seed             RNG seed
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# -----------------------------------------------------------
# Column renaming map
# -----------------------------------------------------------
RENAME_MAP = {
    # Detector-frame masses
    "m1": "mass_1_detector",
    "m2": "mass_2_detector",

    # Spin components
    "a1x": "spin_1x",
    "a1y": "spin_1y",
    "a1z": "spin_1z",
    "a2x": "spin_2x",
    "a2y": "spin_2y",
    "a2z": "spin_2z",

    # Source-frame masses
    "m1_source": "mass_1_source",
    "m2_source": "mass_2_source",
}


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def parse_header(line: str) -> list[str]:
    if line.startswith("#"):
        line = line[1:]
    return line.strip().split()


def maybe_subsample(
    data: np.ndarray,
    n_rows: Optional[int],
    seed: Optional[int],
) -> np.ndarray:
    if not n_rows or n_rows <= 0 or n_rows >= data.shape[0]:
        return data

    rng = np.random.default_rng(seed)
    idx = rng.choice(data.shape[0], n_rows, replace=False)
    return data[idx]


def compute_spins(
    data: np.ndarray,
    cmap: dict[str, int],
    assume_aligned: bool = False,
):
    """
    Build:
        a_1, a_2, cos_tilt_1, cos_tilt_2
    from spin vector components.
    """

    a1x = data[:, cmap["a1x"]]
    a1y = data[:, cmap["a1y"]]
    a1z = data[:, cmap["a1z"]]
    a2x = data[:, cmap["a2x"]]
    a2y = data[:, cmap["a2y"]]
    a2z = data[:, cmap["a2z"]]

    if assume_aligned:
        a_1 = np.abs(a1z)
        a_2 = np.abs(a2z)

        cos_tilt_1 = np.sign(a1z)
        cos_tilt_2 = np.sign(a2z)

        cos_tilt_1[cos_tilt_1 == 0] = 1.0
        cos_tilt_2[cos_tilt_2 == 0] = 1.0

    else:
        a_1 = np.sqrt(a1x**2 + a1y**2 + a1z**2)
        a_2 = np.sqrt(a2x**2 + a2y**2 + a2z**2)

        cos_tilt_1 = np.ones_like(a_1)
        cos_tilt_2 = np.ones_like(a_2)

        mask1 = a_1 > 0
        mask2 = a_2 > 0

        cos_tilt_1[mask1] = a1z[mask1] / a_1[mask1]
        cos_tilt_2[mask2] = a2z[mask2] / a_2[mask2]

    return a_1, a_2, cos_tilt_1, cos_tilt_2


# -----------------------------------------------------------
# Main conversion
# -----------------------------------------------------------
def convert_file(
    src: Path,
    dst_dir: Path,
    n_rows: Optional[int],
    seed: Optional[int],
    assume_aligned: bool,
):

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    # Read header
    with src.open("r") as f:
        raw_header = f.readline()

    cols = parse_header(raw_header)
    cmap = {c: i for i, c in enumerate(cols)}

    # Load numeric data
    data = np.loadtxt(src)

    # Subsample rows
    data = maybe_subsample(data, n_rows, seed)

    # Rename columns
    new_cols = [RENAME_MAP.get(c, c) for c in cols]

    # Compute derived spin parameters
    a_1, a_2, cos_tilt_1, cos_tilt_2 = compute_spins(
        data, cmap, assume_aligned
    )

    # Append derived columns
    data_out = np.column_stack(
        [data, a_1, a_2, cos_tilt_1, cos_tilt_2]
    )

    new_cols.extend(["a_1", "a_2", "cos_tilt_1", "cos_tilt_2"])

    # Save WITHOUT '#' prefix in header
    np.savetxt(
        dst,
        data_out,
        header=" ".join(new_cols),
        comments="",
        fmt="%.8e",
    )

    print(f"Saved: {dst}  ({data_out.shape[0]} rows)")


# -----------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-rows", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--assume-aligned", action="store_true")

    args = p.parse_args()

    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)

    files = list(src_dir.glob("*.dat"))
    if not files:
        raise RuntimeError("No .dat files found!")

    print(f"Found {len(files)} files")

    for f in files:
        convert_file(
            f,
            dst_dir,
            args.n_rows,
            args.seed,
            args.assume_aligned,
        )


if __name__ == "__main__":
    main()
