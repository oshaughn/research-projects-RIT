#! /usr/bin/env python
"""
example_gaussian_uvw.py
=======================

Companion likelihood evaluator for the linear-coordinate-convert demo.

This script is a drop-in for ``example_gaussian.py`` -- same CLI, same
output format -- but the likelihood is a Gaussian whose principal axes lie
along (u, v, w), not (x, y, z).  The transformation between the two bases
is loaded from the same plugin file the hyperpipe config hands to
``util_ConstructEOSPosterior.py`` via ``--supplementary-coordinate-code``,
so the likelihood evaluator and the GP/RF fit agree on what the rotation
is by construction -- you can't drift them apart by editing one without
the other.

Why this is worth doing
-----------------------
A Gaussian with equal variances on (x, y, z) is rotation-invariant, so
fitting in either basis would give numerically indistinguishable results
and tell you nothing about whether the plugin is doing anything.  By
choosing unequal principal axes in (u, v, w) (sigma_u != sigma_v != sigma_w)
the GP/RF fit becomes meaningfully easier in the rotated basis -- one of
the length-scales is short, one is long, and the GP's per-axis bandwidth
estimator gets to express that without correlations leaking across all
three coordinates.

CLI surface
-----------
Identical to ``example_gaussian.py``.  See its docstring or the Makefile
in this directory for the argument list.  In short: this script reads
columns (x, y, z) out of the file pointed at by ``--using-eos``, applies
the linear-coordinate-convert plugin to project them to (u, v, w), and
writes lnL + sigma_lnL into the same file's columns 0 and 1.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# CLI -- mirrors example_gaussian.py exactly so the marg-list entry in the
# hyperpipe yaml can be swapped without touching anything else.
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="Dummy argument required by API")
parser.add_argument("--using-eos", type=str, required=True,
                    help="Path (optionally 'file:'-prefixed) to a grid with columns "
                         "(lnL, sigma_lnL, x, y, z).")
parser.add_argument("--using-eos-index", type=int,
                    help="Line number for single calculation.")
parser.add_argument("--n-events-to-analyze", type=int, default=1)
parser.add_argument("--eos_start_index", type=int)
parser.add_argument("--eos_end_index", type=int)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--outdir", type=str)
parser.add_argument("--outdir-clean", type=str)
parser.add_argument("--fname-output-integral", type=str, required=True)
parser.add_argument("--fname-output-samples", type=str,
                    help="Dummy argument required by API; not used.")
parser.add_argument("--conforming-output-name", action="store_true")

# Knobs specific to this demo.  The defaults match
# linear_coordinate_convert.ini sitting next to this script, so the script
# is runnable with no extra flags from inside demo/hyperpipe/.
parser.add_argument("--coord-plugin", default=None,
                    help="Path to linear_coordinate_convert.py.  Defaults to "
                         "the one in this script's directory.")
parser.add_argument("--coord-ini", default=None,
                    help="Path to linear_coordinate_convert.ini.  Defaults to "
                         "the one in this script's directory.")
parser.add_argument("--mu-uvw", default="0.0,4.95,3.5",
                    help="Comma-separated mean of the Gaussian in (u, v, w). "
                         "Default sits near the centroid of the initial "
                         "blind_gaussian_3d_xy_plus.dat grid when rotated.")
parser.add_argument("--sigma-uvw", default="0.5,1.0,1.5",
                    help="Comma-separated standard deviations along (u, v, w). "
                         "Deliberately unequal so the rotation is visible "
                         "in the resulting posterior.")

opts = parser.parse_args()

# Translate the single-event-mode index into the range form expected below.
if opts.using_eos_index is not None:
    opts.eos_start_index = opts.using_eos_index
    opts.eos_end_index = opts.using_eos_index + opts.n_events_to_analyze

if opts.outdir_clean:
    import shutil
    try:
        shutil.rmtree(opts.outdir)
    except FileNotFoundError:
        pass
elif opts.outdir is None:
    opts.outdir = "."
from pathlib import Path
Path(opts.outdir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Load the coordinate-convert plugin and let it parse its own ini.
#
# We deliberately import the plugin module directly (by file path) instead
# of going through RIFT.misc.coordinate_plugin.load_coordinate_converter.
# Reason: this script runs as a marg-driver inside a condor DAG, where
# pulling in the whole RIFT package can be slow and brittle (it imports
# scipy + LAL + ligo.lw eagerly).  The plugin contract guarantees that
# ``prepare`` and ``convert_coordinates`` work standalone, so we exercise
# exactly that path.  If the standalone import works here it will also
# work for the posterior step -- if it doesn't, we want to fail loudly in
# the cheap step, not deep inside the DAG.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
plugin_path = opts.coord_plugin or os.path.join(_HERE, "linear_coordinate_convert.py")
ini_path    = opts.coord_ini    or os.path.join(_HERE, "linear_coordinate_convert.ini")

if not os.path.isfile(plugin_path):
    raise FileNotFoundError(
        f"example_gaussian_uvw: plugin not found at {plugin_path!r}.  "
        "Pass --coord-plugin if the file lives somewhere else."
    )
if not os.path.isfile(ini_path):
    raise FileNotFoundError(
        f"example_gaussian_uvw: ini not found at {ini_path!r}.  "
        "Pass --coord-ini if the file lives somewhere else."
    )

_spec = importlib.util.spec_from_file_location(
    "linear_coordinate_convert_for_likelihood", plugin_path,
)
linear_plugin = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = linear_plugin
_spec.loader.exec_module(linear_plugin)

import configparser
_cfg = configparser.ConfigParser()
_cfg.optionxform = str  # preserve case (matches the loader)
_cfg.read(ini_path)

# We pass the chart name even though there's only one chart in the demo --
# documents intent and stays correct if a user adds more charts later.
linear_plugin.prepare(
    config=_cfg,
    coord_names=["u", "v", "w"],
    low_level_coord_names=["x", "y", "z"],
    chart="uvw_rotated",
)


# ---------------------------------------------------------------------------
# Likelihood.  Gaussian in (u, v, w) with diagonal covariance and
# user-tunable mean / sigmas.  Diagonal-only on purpose: the *point* of
# the demo is that the rotation makes a non-axis-aligned Gaussian in
# (x, y, z) become axis-aligned in (u, v, w), so the diagonal covariance
# here is precisely what the rotated basis is supposed to deliver.
# ---------------------------------------------------------------------------
mu_uvw    = np.array([float(s) for s in opts.mu_uvw.split(",")],    dtype=float)
sigma_uvw = np.array([float(s) for s in opts.sigma_uvw.split(",")], dtype=float)
if mu_uvw.shape != (3,) or sigma_uvw.shape != (3,):
    raise ValueError(
        "example_gaussian_uvw: --mu-uvw and --sigma-uvw must each be exactly "
        "three comma-separated numbers."
    )
if np.any(sigma_uvw <= 0):
    raise ValueError("example_gaussian_uvw: all sigma_uvw entries must be > 0.")
_log_norm = -0.5 * (3 * np.log(2 * np.pi) + 2 * np.sum(np.log(sigma_uvw)))


def _ln_likelihood_uvw(xyz_rows: np.ndarray) -> np.ndarray:
    """Vectorised lnL for a batch of (x, y, z) rows."""
    uvw = linear_plugin.convert_coordinates(
        xyz_rows,
        coord_names=["u", "v", "w"],
        low_level_coord_names=["x", "y", "z"],
        chart="uvw_rotated",
    )
    z2 = ((uvw - mu_uvw) / sigma_uvw) ** 2
    return _log_norm - 0.5 * np.sum(z2, axis=1)


# ---------------------------------------------------------------------------
# Drive the calculation: load the grid, evaluate lnL on the requested row
# range, write back.  Same I/O layout as example_gaussian.py so the rest
# of the hyperpipe wiring doesn't need to know which evaluator is in use.
# ---------------------------------------------------------------------------
fname_eos = opts.using_eos.replace("file:", "", 1)

with open(fname_eos, "r") as f:
    header_str = f.readline().rstrip()
dat_orig_names = header_str.replace("#", "").split()[2:]
print("example_gaussian_uvw: field names ", dat_orig_names)

# We assume the grid columns are (x, y, z) -- the hyperpipe yaml that
# pairs with this script generates the grid in that basis.  Surface a
# clear error if someone wires us up to a grid with different columns.
expected = ["x", "y", "z"]
if dat_orig_names != expected:
    raise ValueError(
        f"example_gaussian_uvw: expected grid columns {expected!r}, got "
        f"{dat_orig_names!r}.  This evaluator only knows how to apply the "
        "(x, y, z) -> (u, v, w) rotation; build a different evaluator for "
        "other layouts."
    )

# Same string-typed numpy load as example_gaussian.py, so the saved
# format matches byte-for-byte and downstream consumers don't care which
# evaluator wrote it.
eoss = np.genfromtxt(fname_eos, dtype="str")
if eoss.ndim == 1:  # only one data row -> promote to 2-D
    eoss = eoss.reshape(1, -1)

if opts.eos_start_index is None or opts.eos_end_index is None:
    raise ValueError(
        "example_gaussian_uvw: pass --using-eos-index (single-event mode) or "
        "--eos_start_index/--eos_end_index (range mode)."
    )
# eos_end_index is exclusive (Pythonic slice semantics).  The original
# example_gaussian.py clamps to len(eoss)-1 and then slices [start:end],
# which silently drops the last row of the grid; we use len(eoss) so
# every requested row is written.
if opts.eos_start_index >= len(eoss):
    sys.exit(0)  # nothing to do
if opts.eos_end_index > len(eoss):
    opts.eos_end_index = len(eoss)

# Pull (x, y, z) out of the string-typed array as floats.
xyz_block = np.asarray(
    eoss[opts.eos_start_index:opts.eos_end_index, 2:5],
    dtype=float,
)
lnL_block = _ln_likelihood_uvw(xyz_block)

for offset, lnL in enumerate(lnL_block):
    row = opts.eos_start_index + offset
    eoss[row, 0] = repr(float(lnL))
    eoss[row, 1] = "0.001"  # nominal integration error -- mirrors example_gaussian.py

postfix = "+annotation.dat" if opts.conforming_output_name else ""
out_block = eoss[opts.eos_start_index:opts.eos_end_index]
out_header = "lnL     sigma_lnL   " + " ".join(dat_orig_names)
if opts.fname is None:
    out_path = os.path.join(opts.outdir, opts.fname_output_integral + postfix)
else:
    out_path = opts.fname_output_integral + postfix
np.savetxt(out_path, out_block, fmt="%10s", header=out_header)
print(f"example_gaussian_uvw: wrote {len(out_block)} rows to {out_path}")
