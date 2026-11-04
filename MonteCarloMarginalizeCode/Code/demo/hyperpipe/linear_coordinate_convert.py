"""
Reference coordinate-convert plugin: y = A @ x + b
==================================================

This is the minimum-viable example of the plugin contract defined in
``RIFT.misc.coordinate_plugin``.  It applies an affine map between the
``grid-N.dat`` input columns and a user-chosen output basis.

Usage (paired with ``util_ConstructEOSPosterior.py``)::

    util_ConstructEOSPosterior.py \\
        --fname my_grid.dat \\
        --parameter u --parameter v --parameter w \\
        --supplementary-coordinate-code   `pwd`/linear_coordinate_convert.py \\
        --supplementary-coordinate-ini    `pwd`/linear_coordinate_convert.ini \\
        --supplementary-coordinate-chart  uvw_rotated \\
        ...

Charts in this plugin
---------------------
This demo declares a single chart, ``uvw_rotated``, with separable uniform
priors on each output coordinate.  Because the chart is unique it is
auto-selected when ``--supplementary-coordinate-chart`` is omitted.  A
production plugin would typically declare several charts -- e.g. one
Cartesian, one cylindrical -- each with its own per-name prior dict; the
chart name then disambiguates the implicit prior (a uniform prior on ``y``
in ``(x, y, z)`` and a uniform prior on ``y`` in ``(r, y, z)`` are not the
same distribution, and they should not share a key in ``prior_map``).

Ini file format
---------------
The plugin reads a single ``[linear]`` section.  ``input_parameters`` and
``output_parameters`` are comma-separated name lists; they MUST exactly
match (modulo whitespace) the driver's ``low_level_coord_names`` and
``coord_names`` respectively, or ``prepare`` raises.  ``A`` and ``b`` are
JSON-encoded so we don't have to invent another little number-grammar::

    [linear]
    input_parameters  = x, y, z
    output_parameters = u, v, w
    A = [[ 0.7071,  0.7071,  0.0],
         [-0.7071,  0.7071,  0.0],
         [ 0.0,     0.0,     1.0]]
    b = [0.0, 0.0, 0.0]

A few sharp edges worth knowing:

* The Jacobian of an affine map is constant, so a uniform prior in ``x``
  maps to a uniform prior in ``y`` -- the chart's separable uniform priors
  declared below are therefore correct.  If you start from a non-uniform
  input prior, write your own plugin that publishes the transformed prior
  per chart.
* The plugin caches ``A``, ``b``, and the column permutation in
  ``prepare`` so the per-sample call in ``convert_coordinates`` is just a
  matmul.
"""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np


NAME = "linear_coordinate_convert"


def _uniform_prior(x):
    # Separable uniform prior on a single coordinate.  Shape-preserving so
    # the integrator can call it on either scalar or vector arguments.
    return np.ones(np.shape(x))


# Atlas: this plugin advertises one chart.  In a multi-chart plugin you'd
# add more entries here, each with its own ``parameters`` list and
# ``priors`` dict.  ``ranges`` is optional and only used to seed
# ``prior_range_map`` for names that weren't pinned by
# --integration-parameter-range on the CLI.
CHARTS = {
    "uvw_rotated": {
        "parameters": ["u", "v", "w"],
        "priors": {
            "u": _uniform_prior,
            "v": _uniform_prior,
            "w": _uniform_prior,
        },
        "ranges": {
            "u": (-7.0, 7.0),
            "v": (-7.0, 7.0),
            "w": (-7.0, 7.0),
        },
        "description": "45-degree rotation of (x, y, z) in the xy-plane.",
    }
}
DEFAULT_CHART = "uvw_rotated"

# These attributes are filled in by ``prepare`` from the ini file.  The
# loader inspects them too, but only as warning sources -- the
# authoritative names live in CHARTS.
INPUT_PARAMETERS: List[str] = []
OUTPUT_PARAMETERS: List[str] = []
_A: Optional[np.ndarray] = None
_b: Optional[np.ndarray] = None


def _parse_list(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def prepare(config=None, coord_names=None, low_level_coord_names=None,
            chart=None, opts=None, **kwargs):
    """Read A, b, and the parameter name lists from the ini file."""
    global INPUT_PARAMETERS, OUTPUT_PARAMETERS, _A, _b

    if config is None:
        raise ValueError(
            "linear_coordinate_convert: pass --supplementary-coordinate-ini "
            "pointing at an ini file with a [linear] section."
        )
    if not config.has_section("linear"):
        raise ValueError(
            "linear_coordinate_convert: ini file is missing the [linear] section."
        )

    section = config["linear"]
    INPUT_PARAMETERS  = _parse_list(section["input_parameters"])
    OUTPUT_PARAMETERS = _parse_list(section["output_parameters"])
    _A = np.asarray(json.loads(section["A"]), dtype=float)
    _b = np.asarray(json.loads(section["b"]), dtype=float)

    # Shape sanity: A must be (#out, #in); b must be (#out,).
    if _A.shape != (len(OUTPUT_PARAMETERS), len(INPUT_PARAMETERS)):
        raise ValueError(
            f"linear_coordinate_convert: A has shape {_A.shape}, "
            f"expected ({len(OUTPUT_PARAMETERS)}, {len(INPUT_PARAMETERS)})"
        )
    if _b.shape != (len(OUTPUT_PARAMETERS),):
        raise ValueError(
            f"linear_coordinate_convert: b has shape {_b.shape}, "
            f"expected ({len(OUTPUT_PARAMETERS)},)"
        )

    # Consistency with the resolved chart's declared parameter list.
    if chart is not None:
        chart_params = CHARTS[chart]["parameters"]
        if list(chart_params) != OUTPUT_PARAMETERS:
            raise ValueError(
                f"linear_coordinate_convert: ini's output_parameters "
                f"{OUTPUT_PARAMETERS!r} disagrees with chart {chart!r}'s "
                f"declared parameters {chart_params!r}"
            )

    # Cross-check against what the driver actually asked for.  This is the
    # whole point of having a ``prepare`` hook -- catch misconfiguration
    # before we start fitting.
    if low_level_coord_names is not None:
        # Direction matters: the plugin needs every one of ITS declared
        # inputs to be available among the driver's columns.  The driver
        # providing EXTRA columns (non-sampled nuisance parameters, global
        # constants, derived quantities carried in the data file) is normal
        # and harmless -- convert_coordinates ignores them via the in_perm
        # column selection.
        missing_in = [p for p in INPUT_PARAMETERS if p not in low_level_coord_names]
        if missing_in:
            raise ValueError(
                "linear_coordinate_convert: ini's input_parameters "
                f"{missing_in!r} not available among the driver's columns "
                f"{list(low_level_coord_names)!r}"
            )
    if coord_names is not None:
        missing_out = [p for p in coord_names if p not in OUTPUT_PARAMETERS]
        if missing_out:
            raise ValueError(
                "linear_coordinate_convert: driver requested fit coords "
                f"{missing_out!r} not declared in ini's output_parameters "
                f"{OUTPUT_PARAMETERS!r}"
            )

    print(
        "  linear_coordinate_convert: ready. "
        f"chart={chart!r}, inputs={INPUT_PARAMETERS}, outputs={OUTPUT_PARAMETERS}, "
        f"A.shape={_A.shape}, b.shape={_b.shape}"
    )


def convert_coordinates(x_in, coord_names, low_level_coord_names, chart=None, **kwargs):
    """Apply y = A @ x + b, then permute columns into ``coord_names`` order."""
    if _A is None or _b is None:
        raise RuntimeError(
            "linear_coordinate_convert: prepare() was not called.  This "
            "means the loader didn't pass an ini file -- supply "
            "--supplementary-coordinate-ini."
        )

    # The chart kwarg is informational for this plugin (we only have one),
    # but a multi-chart plugin would dispatch on it here.
    x = np.asarray(x_in, dtype=float)
    if x.ndim != 2:
        raise ValueError(
            f"linear_coordinate_convert: expected 2D x_in, got shape {x.shape}"
        )

    # Reorder input columns to match the ini's declared input order, since
    # the driver may pass them in some other order.
    in_perm = [low_level_coord_names.index(name) for name in INPUT_PARAMETERS]
    x_aligned = x[:, in_perm]

    y_full = x_aligned @ _A.T + _b  # shape (N, len(OUTPUT_PARAMETERS))

    # Then pick out the columns the driver actually wants, in its order.
    out_perm = [OUTPUT_PARAMETERS.index(name) for name in coord_names]
    return y_full[:, out_perm]


def inverse_convert_coordinates(y_in, coord_names, low_level_coord_names,
                                chart=None, **kwargs):
    """Inverse of ``convert_coordinates``: y -> x via x = A^{-1} (y - b).

    Required by RIFT stages that need to round-trip through the plugin
    basis -- in particular the puff lane (util_HyperparameterPuffball.py
    and util_HyperparameterTracerUpdate.py): they read the grid in the
    file basis, forward-transform to do the displacement step in the
    plugin basis, then inverse-transform the displaced points back to
    the file basis so the output .dat preserves the file's column
    structure.

    Requires A to be square and invertible.  For affine maps that lose
    information (non-square A, or square-but-singular A) there is no
    closed-form inverse and we raise instead of silently using a
    pseudo-inverse: the user is better served by a clear error than by
    a quietly-wrong round-trip.
    """
    if _A is None or _b is None:
        raise RuntimeError(
            "linear_coordinate_convert: prepare() was not called.  This "
            "means the loader didn't pass an ini file -- supply "
            "--supplementary-coordinate-ini."
        )
    if _A.shape[0] != _A.shape[1]:
        raise ValueError(
            "linear_coordinate_convert: cannot invert a non-square A "
            f"(shape={_A.shape!r}).  inverse_convert_coordinates only "
            "supports square (#out == #in) maps."
        )
    # Compute A^{-1} lazily and cache.  numpy.linalg.inv will raise
    # LinAlgError on singular A -- we let it propagate, the caller
    # gets a useful traceback.
    global _A_inv
    try:
        _A_inv  # type: ignore[name-defined]
    except NameError:
        _A_inv = np.linalg.inv(_A)

    y = np.asarray(y_in, dtype=float)
    if y.ndim != 2:
        raise ValueError(
            "linear_coordinate_convert.inverse: expected 2D y_in, got "
            f"shape {y.shape}"
        )

    # Assemble a full-width (N, len(OUTPUT_PARAMETERS)) array in the
    # plugin's canonical output order, padding missing columns with 0
    # only when the caller passed a strict subset.  In practice the
    # puff lane passes all OUTPUT_PARAMETERS, so this is a permutation.
    y_full = np.zeros((y.shape[0], len(OUTPUT_PARAMETERS)), dtype=float)
    seen = set()
    for j, name in enumerate(coord_names):
        if name not in OUTPUT_PARAMETERS:
            raise ValueError(
                "linear_coordinate_convert.inverse: coord_names contains "
                f"{name!r}, not declared in ini's output_parameters "
                f"{OUTPUT_PARAMETERS!r}"
            )
        y_full[:, OUTPUT_PARAMETERS.index(name)] = y[:, j]
        seen.add(name)
    if len(seen) < len(OUTPUT_PARAMETERS):
        missing = set(OUTPUT_PARAMETERS) - seen
        raise ValueError(
            "linear_coordinate_convert.inverse: input matrix does not span "
            f"every output dimension; missing {sorted(missing)!r}.  A "
            "non-square partial inverse is ambiguous; pass all of "
            f"{OUTPUT_PARAMETERS!r} via coord_names."
        )

    # Apply A^{-1} @ (y - b) row-wise.
    x_aligned = (y_full - _b) @ _A_inv.T   # shape (N, len(INPUT_PARAMETERS))

    # Permute into the caller's requested low_level_coord_names order.
    out_perm = [INPUT_PARAMETERS.index(name) for name in low_level_coord_names]
    return x_aligned[:, out_perm]
