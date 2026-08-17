#! /usr/bin/env python
#
# util_HyperparameterTracerUpdate.py
#
# DRAFT - parsimonious-placement project (2026-05-13)
#
# GOAL
#   Drop-in alternative to util_HyperparameterPuffball.py for the RIFT hyperpipeline.
#   Same I/O contract: read a hyperpipe-format .dat grid (header
#   "# lnL sigma_lnL p1 p2 ..."), advance it by tracer SMC + birth-death, write
#   a new .dat grid in the same format.
#
#   Shared engine with util_ParameterTracerUpdate.py (the event-level cousin):
#   both import from RIFT.misc.tracer_placement.
#
# COMPATIBILITY with util_HyperparameterPuffball.py
#   --inj-file, --inj-file-out, --puff-factor, --force-away, --parameter,
#   --no-correlation, --random-parameter, --random-parameter-range,
#   --downselect-parameter, --downselect-parameter-range, --regularize
#       same semantics. --update-method puffball reproduces puffball exactly.
#
# NEW
#   --update-method {smc-mala-bd, smc-mala, birth-death, ucb, puffball}  default smc-mala-bd
#   --tracer-fit-method {rf, rbf, quadratic, polynomial, gp_linmean}   default rf
#       gp_linmean is a linear-mean GP: unlike rf (piecewise-constant, flat
#       outside the training hull) it extrapolates the global lnL trend past
#       the sampled region, so placement can chase a peak clipped at a box
#       edge. It also supplies a real posterior sigma for --update-method ucb.
#   --tracer-lnl-floor-delta FLOAT      default None (OFF; legacy unchanged)
#       Clamp training lnL at max(lnL)-delta instead of cutting outliers, so
#       catastrophic-fit points remain anchors for the surrogate's scale.
#   --inj-file-prev      OPTIONAL previous-iteration .dat (enables SMC bridging)
#   --no-union-refit     opt out of union refit when --inj-file-prev is given
#   --n-mala-steps INT                  default 8
#   --target-ess-frac FLOAT             default 0.5
#   --birth-death-rate FLOAT            default 1.0
#   --ucb-kappa FLOAT                   default 2.0  (UCB exploration weight)
#   --ucb-n-candidates INT              default 20000
#   --rng-seed INT                      deterministic when given
#   --state-in / --state-out            tiny state file (~100 bytes)
#
# --force-away K (inherited from puffball semantics): after the engine returns
# X_out, drop any point within Mahalanobis distance K of an already-kept point
# (covariance from the input grid, ridge-stabilized). K=0 disables.
#
# I/O FORMAT
#   Input/output is the same as util_HyperparameterPuffball.py: a text file with a
#   leading "# lnL sigma_lnL p1 p2 ..." header line. Data rows are numeric.
#   - X (coord array) is extracted from the parameter columns by name (matching
#     --parameter ordering).
#   - Y (lnL) and sigma (sigma_lnL) come from the first two columns.
#   - Output rows preserve all original columns; only the parameter values are
#     updated. The lnL/sigma_lnL columns are zeroed in the output, matching the
#     puffball convention (next iteration's marg driver will overwrite them).
#
# USAGE in hyperpipe (Hydra)
#   puff:
#     exe: util_HyperparameterTracerUpdate.py
#     settings:
#       update-method: smc-mala-bd
#       tracer-fit-method: rf
#
# USAGE in legacy create_eos_posterior_pipeline
#   --puff-exe `which util_HyperparameterTracerUpdate.py`
#   --puff-args `pwd`/args_puff.txt   # contains the flags above
#

import argparse
import os
import pickle
import sys
import numpy as np

try:
    # Canonical path once the engine is merged into RIFT proper.
    from RIFT.misc.tracer_placement import samplers as _tracer_samplers
    from RIFT.misc.tracer_placement import fits as _tracer_fits
    _TRACER_OK = True
except ImportError:
    try:
        # Local-dev fallback: PYTHONPATH points at a directory containing
        # `tracer_placement/` directly (no RIFT/ wrapper). See the demo Makefile.
        from tracer_placement import samplers as _tracer_samplers
        from tracer_placement import fits as _tracer_fits
        _TRACER_OK = True
    except ImportError:
        _TRACER_OK = False


# --------------------------------- CLI ------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    # Mirror util_HyperparameterPuffball.py
    p.add_argument("--inj-file", required=True, help="Input .dat grid (hyperpipe format).")
    p.add_argument("--inj-file-out", default="output-tracer.dat")
    p.add_argument("--puff-factor", default=1.0, type=float)
    p.add_argument("--force-away", default=0.0, type=float)
    p.add_argument("--parameter", action="append", required=True)
    p.add_argument("--no-correlation", action="append", type=str)
    p.add_argument("--random-parameter", action="append")
    p.add_argument("--random-parameter-range", action="append", type=str)
    p.add_argument("--downselect-parameter", action="append")
    p.add_argument("--downselect-parameter-range", action="append", type=str)
    p.add_argument("--regularize", action="store_true")
    # Tracer-specific
    p.add_argument("--update-method",
                   choices=("smc-mala-bd", "smc-mala", "birth-death", "ucb", "puffball"),
                   default="smc-mala-bd")
    p.add_argument("--tracer-fit-method",
                   choices=("rf", "rbf", "quadratic", "polynomial", "gp_linmean"),
                   default="rf")
    p.add_argument("--tracer-lnl-floor-delta", default=None, type=float,
                   help="Clamp training lnL from below at max(lnL) - DELTA "
                        "instead of discarding low points. Keeps catastrophic-fit "
                        "outliers as anchors that pin the surrogate's length "
                        "scale and signal variance. Default off (legacy "
                        "behaviour bit-for-bit unchanged).")
    p.add_argument("--inj-file-prev", default=None,
                   help="Optional previous-iteration .dat for SMC bridging / union refit.")
    p.add_argument("--no-union-refit", action="store_true")
    p.add_argument("--n-mala-steps", default=8, type=int)
    p.add_argument("--target-ess-frac", default=0.5, type=float)
    p.add_argument("--birth-death-rate", default=1.0, type=float)
    # UCB (super-conservative GP-style placement)
    p.add_argument("--ucb-kappa", default=2.0, type=float,
                   help="UCB exploration weight: score = mu(lambda) + kappa * sigma(lambda). "
                        "Larger => more explorative. Used only when --update-method ucb.")
    p.add_argument("--ucb-n-candidates", default=20000, type=int,
                   help="Size of the candidate pool from which UCB greedily selects.")
    p.add_argument("--rng-seed", default=None, type=int)
    p.add_argument("--state-in", default=None)
    p.add_argument("--state-out", default=None)
    # ---- Optional coordinate-convert plugin -------------------------------- #
    # When set, the tracer operates in the PLUGIN basis: forward-transform
    # the file's input-basis columns into the basis named by --parameter,
    # do SMC / birth-death / etc. in that basis, then inverse-transform back
    # to the file basis to write the output .dat.  Legacy code path is
    # byte-identical when --supplementary-coordinate-code is unset.
    # See RIFT.misc.coordinate_plugin for the plugin contract.  The plugin
    # MUST implement inverse_convert_coordinates: the tracer round-trips.
    p.add_argument("--supplementary-coordinate-code", default=None, type=str,
                   help="Coordinate plugin spec: 'rift_default', a .py path, or an importable dotted name.")
    p.add_argument("--supplementary-coordinate-function", default=None, type=str,
                   help="Entry-point callable name. Defaults to 'convert_coordinates'.")
    p.add_argument("--supplementary-coordinate-ini", default=None, type=str,
                   help="Optional ini file handed to the plugin's prepare() hook.")
    p.add_argument("--supplementary-coordinate-chart", default=None, type=str,
                   help="Which chart in the plugin's CHARTS dict to use.")
    p.add_argument("--supplementary-coordinate-input-parameter", action='append', default=None,
                   help="File-column name to feed the plugin as an input dimension. Repeat per column. "
                        "If omitted, the plugin's CHARTS[chart] input_parameters / INPUT_PARAMETERS is used.")
    return p


# ---------------- Mahalanobis self-avoidance (--force-away) --------------- #

def _force_away_decimate(X_kept, X_pool, cov, threshold):
    """Greedy: keep first len(X_kept) rows of X_pool subject to a minimum
    Mahalanobis distance `threshold` from every previously-kept row. Returns
    (X_out, mask). X_kept is the preferred-order list; X_pool is the (possibly
    larger) candidate set to fall back to when a preferred row is rejected.
    Mirrors util_ParameterPuffball.py's --force-away semantics."""
    if threshold <= 0 or len(X_kept) == 0:
        return X_kept, np.ones(len(X_kept), dtype=bool)
    # Regularize cov for stability
    cov_r = cov + 1e-10 * np.eye(cov.shape[0])
    icov = np.linalg.inv(cov_r)
    target_n = len(X_kept)
    queue = list(X_pool)  # candidate list (input order matters: preferred first)
    out = []
    for x in queue:
        if not out:
            out.append(x)
            continue
        diffs = np.asarray(out) - x
        d2 = np.einsum("ij,jk,ik->i", diffs, icov, diffs)
        if (d2 >= threshold ** 2).all():
            out.append(x)
        if len(out) >= target_n:
            break
    if len(out) < target_n:
        # Couldn't fill the budget under the constraint; relax to keep what we have.
        sys.stderr.write(
            f"util_HyperparameterTracerUpdate: --force-away {threshold} could only "
            f"place {len(out)}/{target_n} points; returning the kept subset.\n")
    return np.asarray(out), None


# ----------------------- .dat <-> arrays ----------------------------------- #

def _read_dat(path):
    """Return (column_names, raw_rows ndarray)."""
    with open(path) as f:
        header = f.readline().rstrip("\n")
    if not header.startswith("#"):
        sys.exit(f"util_HyperparameterTracerUpdate: input {path!r} missing '#' header")
    cols = header.lstrip("#").split()
    if len(cols) < 3 or cols[0] != "lnL" or cols[1] not in ("sigma_lnL", "sigma"):
        sys.exit(f"util_HyperparameterTracerUpdate: header must start with "
                 f"'lnL sigma_lnL ...', got {cols!r}")
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None, :]
    return cols, rows


def _extract_X(cols, rows, parameter_order):
    idx = [cols.index(p) for p in parameter_order]
    return rows[:, idx]


def _write_dat(path, cols, rows):
    np.savetxt(path, rows, header=" ".join(cols))


def _build_downselect(opts):
    d = {}
    if opts.downselect_parameter:
        for name, rng in zip(opts.downselect_parameter,
                             opts.downselect_parameter_range or []):
            d[name] = list(eval(rng))
    return d


def _coord_box(parameter_order, downselect_dict, X):
    d = len(parameter_order)
    box = np.zeros((d, 2))
    for i, name in enumerate(parameter_order):
        if name in downselect_dict:
            box[i] = downselect_dict[name]
        else:
            lo = float(X[:, i].min())
            hi = float(X[:, i].max())
            pad = 0.1 * (hi - lo + 1e-9)
            box[i] = (lo - pad, hi + pad)
    return box


# ------------------------------ main --------------------------------------- #

def _load_coord_plugin(opts):
    """Load the coordinate plugin if --supplementary-coordinate-code is set.

    Returns (forward, inverse, in_names) or (None, None, None) if no plugin
    was requested.  Bails out loudly if a plugin was requested but the
    inverse callable isn't present -- the tracer needs to round-trip and
    silently using a pseudo-inverse would produce subtly-wrong placements.
    """
    if not getattr(opts, "supplementary_coordinate_code", None):
        return None, None, None
    from RIFT.misc.coordinate_plugin import load_coordinate_converter
    forward, module = load_coordinate_converter(
        spec=opts.supplementary_coordinate_code,
        function_name=opts.supplementary_coordinate_function,
        ini_path=opts.supplementary_coordinate_ini,
        coord_names=opts.parameter,
        low_level_coord_names=opts.supplementary_coordinate_input_parameter,
        chart=opts.supplementary_coordinate_chart,
        opts=opts,
        prior_map=None,
        prior_range_map=None,
    )
    chart_spec = None
    if opts.supplementary_coordinate_chart:
        chart_spec = getattr(module, "CHARTS", {}).get(opts.supplementary_coordinate_chart)
    if chart_spec is None:
        charts = getattr(module, "CHARTS", None) or {}
        if len(charts) == 1:
            chart_spec = next(iter(charts.values()))
    in_names = list(
        opts.supplementary_coordinate_input_parameter
        or (chart_spec.get("input_parameters") if chart_spec else None)
        or getattr(module, "INPUT_PARAMETERS", [])
    )
    if not in_names:
        sys.exit("util_HyperparameterTracerUpdate: plugin loaded but no "
                 "file-basis input columns are declared; pass "
                 "--supplementary-coordinate-input-parameter or define "
                 "INPUT_PARAMETERS / CHARTS[chart].input_parameters.")
    inverse = getattr(module, "inverse_convert_coordinates", None)
    if not callable(inverse):
        sys.exit("util_HyperparameterTracerUpdate: --supplementary-coordinate-code "
                 "set, but the plugin does not define inverse_convert_coordinates. "
                 "The tracer needs to round-trip through the plugin basis -- add an "
                 "inverse or run without the plugin.")
    print(" util_HyperparameterTracerUpdate: operating in plugin basis {!r} "
          "(file columns {!r}).".format(list(opts.parameter), in_names))
    return forward, inverse, in_names


def _extract_X_via_plugin(cols, rows, parameter_order, forward, in_names):
    """Plugin-aware X extraction: read file-basis columns and forward-transform."""
    missing = [n for n in in_names if n not in cols]
    if missing:
        sys.exit("util_HyperparameterTracerUpdate: plugin input column(s) "
                 "{!r} not in dat header {!r}".format(missing, cols))
    in_idx = [cols.index(n) for n in in_names]
    X_in = rows[:, in_idx].astype(float)
    X = forward(X_in, coord_names=parameter_order, low_level_coord_names=in_names)
    X = np.asarray(X, dtype=float)
    if X.shape != (len(rows), len(parameter_order)):
        sys.exit("util_HyperparameterTracerUpdate: plugin forward returned "
                 "shape {!r}, expected {!r}".format(X.shape, (len(rows), len(parameter_order))))
    return X


def main(argv=None):
    opts = build_parser().parse_args(argv)
    rng = np.random.default_rng(opts.rng_seed)

    # Load the optional coordinate plugin BEFORE any data extraction so the
    # same forward/inverse pair is reused for the input grid, the previous-
    # iteration grid (--inj-file-prev), and the final write-back.
    forward, inverse, in_names = _load_coord_plugin(opts)
    plugin_active = forward is not None

    cols, rows = _read_dat(opts.inj_file)
    if plugin_active:
        X = _extract_X_via_plugin(cols, rows, opts.parameter, forward, in_names)
    else:
        X = _extract_X(cols, rows, opts.parameter)
    Y = rows[:, 0]                 # lnL column
    S = rows[:, 1] if rows.shape[1] >= 2 else None
    downselect = _build_downselect(opts)
    prior_box = _coord_box(opts.parameter, downselect, X)

    method = opts.update_method

    # ---- puffball regression path -------------------------------------- #
    if method == "puffball":
        cov = (np.cov(X.T) * opts.puff_factor**2 if X.shape[1] > 1
               else np.array([[X.std() ** 2]]))
        cov = np.atleast_2d(cov)
        if np.min(np.linalg.eigvalsh(cov)) < 1e-12:
            cov = cov + 1e-8 * np.eye(cov.shape[0])
        delta = rng.multivariate_normal(np.zeros(X.shape[1]), cov, size=len(X))
        X_out = X + delta
        # write back; zero lnL/sigma (puffball convention)
        out_rows = rows.copy()
        if plugin_active:
            # X_out is in the plugin basis -- inverse-transform back to the
            # file basis and write each file-basis column.
            X_in_out = np.asarray(
                inverse(X_out, coord_names=opts.parameter, low_level_coord_names=in_names),
                dtype=float,
            )
            for j, name in enumerate(in_names):
                out_rows[:, cols.index(name)] = X_in_out[:, j]
        else:
            for i, name in enumerate(opts.parameter):
                out_rows[:, cols.index(name)] = X_out[:, i]
        out_rows[:, 0] = 0.0
        out_rows[:, 1] = 0.0
        _write_dat(opts.inj_file_out, cols, out_rows)
        return

    # ---- tracer path --------------------------------------------------- #
    if not _TRACER_OK:
        sys.stderr.write("util_HyperparameterTracerUpdate: RIFT.misc.tracer_placement "
                         "not installed; falling back to puffball.\n")
        opts.update_method = "puffball"
        return main(argv)

    X_train, Y_train, S_train = X, Y, S
    fit_prev = None

    if opts.inj_file_prev is not None and os.path.exists(opts.inj_file_prev):
        cols_p, rows_p = _read_dat(opts.inj_file_prev)
        if plugin_active:
            X_prev = _extract_X_via_plugin(cols_p, rows_p, opts.parameter, forward, in_names)
        else:
            X_prev = _extract_X(cols_p, rows_p, opts.parameter)
        Y_prev = rows_p[:, 0]
        S_prev = rows_p[:, 1] if rows_p.shape[1] >= 2 else None
        fit_prev = _tracer_fits.build(opts.tracer_fit_method,
                                      X_prev, Y_prev, sigma=S_prev,
                                      lnl_floor_delta=opts.tracer_lnl_floor_delta)
        if not opts.no_union_refit:
            X_train = np.vstack([X_prev, X])
            Y_train = np.concatenate([Y_prev, Y])
            if S is not None and S_prev is not None:
                S_train = np.concatenate([S_prev, S])
            else:
                S_train = None

    fit_now = _tracer_fits.build(opts.tracer_fit_method,
                                 X_train, Y_train, sigma=S_train,
                                 lnl_floor_delta=opts.tracer_lnl_floor_delta)

    state = {}
    if opts.state_in and os.path.exists(opts.state_in):
        with open(opts.state_in, "rb") as f:
            state = pickle.load(f)

    sampler_map = {
        "smc-mala-bd": _tracer_samplers.smc_mala_bd,
        "smc-mala":    _tracer_samplers.smc_mala,
        "birth-death": _tracer_samplers.birth_death,
    }
    if hasattr(_tracer_samplers, "ucb_place"):
        sampler_map["ucb"] = _tracer_samplers.ucb_place
    if method not in sampler_map:
        sys.exit(f"util_HyperparameterTracerUpdate: --update-method {method!r} "
                 f"not available in installed engine (have: {sorted(sampler_map)}).")
    sampler = sampler_map[method]

    sampler_kw = dict(
        particles=X,
        surrogate=fit_now,
        surrogate_prev=fit_prev,
        prior_box=prior_box,
        rng=rng,
        n_mala_steps=opts.n_mala_steps,
        target_ess_frac=opts.target_ess_frac,
        birth_death_rate=opts.birth_death_rate,
        state=state,
    )
    if method == "ucb":
        sampler_kw.update(
            kappa=opts.ucb_kappa,
            n_candidates=opts.ucb_n_candidates,
        )
    X_out, info = sampler(**sampler_kw)

    # --- self-avoidance (--force-away) ----------------------------------- #
    if opts.force_away and opts.force_away > 0 and len(X_out) > 1:
        # Use the input grid's covariance as the Mahalanobis metric (stable,
        # independent of where the engine moved particles to).
        if X.shape[1] > 1:
            cov_in = np.cov(X.T)
        else:
            cov_in = np.array([[float(X.std() ** 2) + 1e-12]])
        cov_in = np.atleast_2d(cov_in)
        X_out, _ = _force_away_decimate(
            X_kept=X_out, X_pool=X_out, cov=cov_in, threshold=opts.force_away)

    if opts.state_out:
        with open(opts.state_out, "wb") as f:
            pickle.dump(info.get("state", {}), f)

    # Apply downselect manually (puffball-equivalent behavior)
    if downselect:
        mask = np.ones(len(X_out), dtype=bool)
        for k, (lo, hi) in downselect.items():
            if k in opts.parameter:
                col = opts.parameter.index(k)
                mask &= (X_out[:, col] >= lo) & (X_out[:, col] <= hi)
        X_out = X_out[mask]

    out_rows = np.zeros((len(X_out), rows.shape[1]))
    # carry forward any extra columns from input rows (just zero them; marg driver overwrites)
    if plugin_active:
        # X_out is in the plugin basis -- inverse-transform back to the file
        # basis and write each file-basis column.  --parameter names need
        # not be file columns at all here.
        X_in_out = np.asarray(
            inverse(X_out, coord_names=opts.parameter, low_level_coord_names=in_names),
            dtype=float,
        )
        for j, name in enumerate(in_names):
            out_rows[:, cols.index(name)] = X_in_out[:, j]
    else:
        for i, name in enumerate(opts.parameter):
            out_rows[:, cols.index(name)] = X_out[:, i]
    out_rows[:, 0] = 0.0
    out_rows[:, 1] = 0.0
    _write_dat(opts.inj_file_out, cols, out_rows)


if __name__ == "__main__":
    main()
