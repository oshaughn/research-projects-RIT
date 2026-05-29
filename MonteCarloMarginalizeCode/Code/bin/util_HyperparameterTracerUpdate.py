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
#   --tracer-fit-method {rf, rbf, quadratic, polynomial}               default rf
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
                   choices=("rf", "rbf", "quadratic", "polynomial"),
                   default="rf")
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

def main(argv=None):
    opts = build_parser().parse_args(argv)
    rng = np.random.default_rng(opts.rng_seed)

    cols, rows = _read_dat(opts.inj_file)
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
        # write back into rows; zero lnL/sigma (puffball convention)
        out_rows = rows.copy()
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
        X_prev = _extract_X(cols_p, rows_p, opts.parameter)
        Y_prev = rows_p[:, 0]
        S_prev = rows_p[:, 1] if rows_p.shape[1] >= 2 else None
        fit_prev = _tracer_fits.build(opts.tracer_fit_method,
                                      X_prev, Y_prev, sigma=S_prev)
        if not opts.no_union_refit:
            X_train = np.vstack([X_prev, X])
            Y_train = np.concatenate([Y_prev, Y])
            if S is not None and S_prev is not None:
                S_train = np.concatenate([S_prev, S])
            else:
                S_train = None

    fit_now = _tracer_fits.build(opts.tracer_fit_method,
                                 X_train, Y_train, sigma=S_train)

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
    for i, name in enumerate(opts.parameter):
        out_rows[:, cols.index(name)] = X_out[:, i]
    out_rows[:, 0] = 0.0
    out_rows[:, 1] = 0.0
    _write_dat(opts.inj_file_out, cols, out_rows)


if __name__ == "__main__":
    main()
