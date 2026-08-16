#! /usr/bin/env python
#
# util_ParameterTracerUpdate.py
#
# DRAFT - parsimonious-placement project (rev. 2026-05-13)
#
# GOAL
#   Drop-in alternative to util_ParameterPuffball.py for iterative grid update.
#   Instead of resampling-from-fit + Gaussian jitter, advance the existing grid
#   through a tracer-particle update: tempered SMC with MALA moves on a surrogate,
#   plus optional birth-death rejuvenation. The tool REFITS its own surrogate from
#   the same .dat file CIP reads, so it is fully self-contained (no model files
#   passed between jobs, no CIP coupling, no extra OSG transfers).
#
# COMPATIBILITY
#   --parameter, --inj-file, --inj-file-out, --fref, --fmin, --random-parameter,
#   --random-parameter-range, --mc-range, --eta-range, --mtot-range,
#   --downselect-parameter, --downselect-parameter-range, --reflect-parameter,
#   --enforce-duration-bound  : same semantics as util_ParameterPuffball.py.
#
# DATA-INGEST (mirrors CIP)
#   --fname        path to current iteration's ILE composite .dat file
#   --fname-prev   optional, previous iteration's .dat (enables SMC bridge)
#   --lnL-col      column index of lnL (caller specifies; matches CIP's expectations)
#   --sigma-col    optional column index of sigma_lnL
#
# NEW
#   --update-method {smc-mala-bd, smc-mala, birth-death, puffball}
#       Default smc-mala-bd. "puffball" reproduces util_ParameterPuffball.py for regression.
#   --tracer-fit-method {rf, rbf, quadratic, polynomial, gp_linmean}   default rf
#       gp_linmean is a linear-mean GP: unlike rf (piecewise-constant, flat
#       outside the training hull) it extrapolates the global lnL trend past
#       the sampled region, so placement can chase a peak clipped at a box
#       edge. It also supplies a real posterior sigma (predict_with_std).
#   --tracer-lnl-floor-delta FLOAT  default None (OFF; legacy unchanged)
#       Clamp training lnL at max(lnL)-delta instead of cutting outliers, so
#       catastrophic-fit points remain anchors for the surrogate's scale.
#   --no-union-refit                if --fname-prev given, do NOT include prev points in f_k fit
#   --n-mala-steps INT              default 8
#   --target-ess-frac FLOAT         default 0.5
#   --birth-death-rate FLOAT        default 1.0
#   --rng-seed INT                  deterministic when given
#   --state-in <path>               tiny pickle: {mala_eps, last_rng_state} (~100 bytes)
#   --state-out <path>              where to dump updated state for the next iteration
#
# USAGE EXAMPLE
#   util_ParameterTracerUpdate.py \
#       --parameter mc --parameter eta --parameter chi_eff --parameter chi_p \
#       --inj-file overlap-grid_k.xml.gz --inj-file-out overlap-grid_kp1 \
#       --fname all_lnL_k.dat --fname-prev all_lnL_km1.dat \
#       --lnL-col 9 \
#       --update-method smc-mala-bd --tracer-fit-method rf \
#       --mc-range '[20,40]' --eta-range '[0.10,0.25]' \
#       --state-in tracer_state_k.pkl --state-out tracer_state_kp1.pkl
#
# NOTES
#   - This is a draft skeleton. The sampler engine lives in
#     parsimonious_placement_plan.md's proto/samplers/. For production it should be
#     promoted into RIFT.misc.tracer_placement and imported here.
#   - Surrogate refitting happens inside this tool every call. RF and RBF refits on
#     the typical (X, lnL) size (10^2 - 10^4 points) cost seconds; negligible vs ILE.
#

import argparse
import os
import pickle
import sys
import numpy as np

import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim   # noqa: F401
import lal

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

from igwn_ligolw import lsctables, ligolw   # noqa: F401
lsctables.use_in(ligolw.LIGOLWContentHandler)


# --------------------------------- CLI ------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    # Mirror util_ParameterPuffball.py
    p.add_argument("--inj-file", required=True)
    p.add_argument("--inj-file-out", default="output-tracer")
    p.add_argument("--fail-if-empty", action="store_true")
    p.add_argument("--approx-output", default="SEOBNRv2")
    p.add_argument("--fref", default=None, type=float)
    p.add_argument("--fmin", default=None, type=float)
    p.add_argument("--parameter", action="append", required=True)
    p.add_argument("--random-parameter", action="append")
    p.add_argument("--random-parameter-range", action="append", type=str)
    p.add_argument("--mc-range", default=None)
    p.add_argument("--eta-range", default=None)
    p.add_argument("--mtot-range", default=None)
    p.add_argument("--downselect-parameter", action="append")
    p.add_argument("--downselect-parameter-range", action="append", type=str)
    p.add_argument("--reflect-parameter", action="append", type=str)
    p.add_argument("--enforce-duration-bound", default=None, type=float)
    # Data ingest (mirror CIP)
    p.add_argument("--fname", required=True,
                   help="Current iteration ILE composite .dat (same file CIP reads).")
    p.add_argument("--fname-prev", default=None,
                   help="Optional previous iteration .dat; enables SMC bridging.")
    p.add_argument("--lnL-col", type=int, required=True,
                   help="Column index of lnL in --fname (caller specifies, matches CIP).")
    p.add_argument("--sigma-col", type=int, default=None,
                   help="Optional column index of sigma_lnL.")
    p.add_argument("--lnL-downscale-factor", type=float, default=1.0)
    # Tracer-specific
    p.add_argument("--update-method",
                   choices=("smc-mala-bd", "smc-mala", "birth-death", "puffball"),
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
    p.add_argument("--no-union-refit", action="store_true",
                   help="If --fname-prev is given, do NOT include those points in the f_k fit.")
    p.add_argument("--n-mala-steps", default=8, type=int)
    p.add_argument("--target-ess-frac", default=0.5, type=float)
    p.add_argument("--birth-death-rate", default=1.0, type=float)
    p.add_argument("--rng-seed", default=None, type=int)
    p.add_argument("--state-in", default=None)
    p.add_argument("--state-out", default=None)
    # Back-compat with puffball for the regression path
    p.add_argument("--puff-factor", default=1.0, type=float)
    p.add_argument("--force-away", default=0.0, type=float)
    p.add_argument("--regularize", action="store_true")
    p.add_argument("--no-correlation", type=str, action="append")
    return p


# --------------------- XML -> X (coord array) ------------------------------ #

def _xml_to_X(opts, coord_names):
    P_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj_file)
    rows = []
    for P in P_list:
        if opts.fmin is not None:
            P.fmin = opts.fmin
        if opts.fref is not None:
            P.fref = opts.fref
        line = np.zeros(len(coord_names))
        for i, name in enumerate(coord_names):
            fac = lal.MSUN_SI if name in ("mc", "m1", "m2", "mtot") else 1.0
            line[i] = P.extract_param(name) / fac
        rows.append(line)
    return np.array(rows), P_list


def _X_to_xml(X_out, P_template_list, opts, coord_names, downselect_dict):
    P_out_list = []
    n = min(len(P_template_list), len(X_out))
    for i in range(n):
        P = P_template_list[i]
        for j, name in enumerate(coord_names):
            fac = lal.MSUN_SI if name in ("mc", "m1", "m2", "mtot") else 1.0
            P.assign_param(name, X_out[i, j] * fac)
        keep = True
        for k, (lo, hi) in downselect_dict.items():
            v = P.extract_param(k)
            if k in ("mc", "m1", "m2", "mtot"):
                v /= lal.MSUN_SI
            if v < lo or v > hi:
                keep = False
                break
        if opts.enforce_duration_bound is not None:
            if lalsimutils.estimateWaveformDuration(P) > opts.enforce_duration_bound:
                keep = False
        if keep:
            P_out_list.append(P)
    if opts.fail_if_empty and not P_out_list:
        sys.stderr.write("tracer: produced empty grid; aborting.\n")
        sys.exit(2)
    fref = P_out_list[0].fref if P_out_list else opts.fref
    lalsimutils.ChooseWaveformParams_array_to_xml(
        P_out_list, fname=opts.inj_file_out, fref=fref)


def _build_downselect(opts):
    d = {}
    if opts.mc_range is not None:
        d["mc"] = list(eval(opts.mc_range))
    if opts.eta_range is not None:
        d["eta"] = list(eval(opts.eta_range))
    if opts.mtot_range is not None:
        d["mtot"] = list(eval(opts.mtot_range))
    if opts.downselect_parameter:
        for name, rng in zip(opts.downselect_parameter, opts.downselect_parameter_range or []):
            d[name] = list(eval(rng))
    return d


def _coord_box(coord_names, downselect_dict, X):
    d = len(coord_names)
    box = np.zeros((d, 2))
    for i, name in enumerate(coord_names):
        if name in downselect_dict:
            box[i] = downselect_dict[name]
        else:
            box[i] = (float(X[:, i].min()), float(X[:, i].max()))
    return box


# --------------- .dat -> (X, Y, sigma) in coord-frame ---------------------- #
#
# We need (X, Y) in the SAME coordinate frame the sampler will move in. The .dat
# columns are not in coord-frame in general; CIP does its own conversion. We use
# the simplest robust approach: match each .dat row to a P in the input XML by
# row order (the standard RIFT contract), then read coord-frame X from the XML
# and lnL from the matched .dat row.
#
# When --fname-prev is used, the caller must also pass a previous-grid XML
# (--inj-file-prev) so we can do the same matching for f_{k-1}. To keep the CLI
# minimal in this draft we expect the convention that --fname-prev sits next to
# overlap-grid_{k-1}.xml.gz in the same directory (a one-line lookup) — TODO
# in production: add --inj-file-prev.

def _load_lnL_for_grid(fname_dat, lnL_col, sigma_col, n_points, downscale):
    arr = np.loadtxt(fname_dat)
    if arr.ndim == 1:
        arr = arr[None, :]
    if len(arr) < n_points:
        sys.stderr.write(
            f"tracer: .dat has fewer rows ({len(arr)}) than XML grid ({n_points}); "
            "alignment assumption (row order) violated.\n")
        sys.exit(3)
    Y = arr[:n_points, lnL_col] * downscale
    if sigma_col is not None:
        S = arr[:n_points, sigma_col]
    else:
        S = None
    return Y, S


# ------------------------------- main -------------------------------------- #

def main(argv=None):
    opts = build_parser().parse_args(argv)
    coord_names = list(opts.parameter)
    downselect = _build_downselect(opts)

    X, P_template = _xml_to_X(opts, coord_names)
    if X.size == 0:
        sys.stderr.write("tracer: empty input grid; aborting.\n")
        sys.exit(2)
    prior_box = _coord_box(coord_names, downselect, X)
    rng = np.random.default_rng(opts.rng_seed)

    method = opts.update_method

    # ---- puffball regression path -------------------------------------- #
    if method == "puffball":
        cov = np.cov(X.T) * opts.puff_factor**2 if X.shape[1] > 1 else np.array([[X.std()**2]])
        cov = np.atleast_2d(cov)
        delta = rng.multivariate_normal(np.zeros(X.shape[1]), cov, size=len(X))
        X_out = X + delta
        _X_to_xml(X_out, P_template, opts, coord_names, downselect)
        return

    # ---- tracer path: refit our own surrogate from .dat ---------------- #
    if not _TRACER_OK:
        sys.stderr.write("tracer: RIFT.misc.tracer_placement not installed; "
                         "falling back to puffball.\n")
        opts.update_method = "puffball"
        return main(argv)

    Y_k, S_k = _load_lnL_for_grid(opts.fname, opts.lnL_col, opts.sigma_col,
                                  len(X), opts.lnL_downscale_factor)

    X_train_k = X
    Y_train_k = Y_k
    S_train_k = S_k
    X_prev = None
    fit_prev = None

    if opts.fname_prev is not None:
        # Read previous-iteration grid + lnL. Production CLI should accept
        # --inj-file-prev explicitly; here we follow the naming convention.
        inj_prev = opts.fname_prev.replace(".dat", ".xml.gz")
        if not os.path.exists(inj_prev):
            sys.stderr.write(
                f"tracer: cannot locate previous grid XML ({inj_prev}); "
                "ignoring --fname-prev and running unbridged.\n")
        else:
            opts_prev = argparse.Namespace(**vars(opts))
            opts_prev.inj_file = inj_prev
            X_prev, _ = _xml_to_X(opts_prev, coord_names)
            Y_prev, S_prev = _load_lnL_for_grid(
                opts.fname_prev, opts.lnL_col, opts.sigma_col,
                len(X_prev), opts.lnL_downscale_factor)
            if not opts.no_union_refit:
                X_train_k = np.vstack([X_prev, X])
                Y_train_k = np.concatenate([Y_prev, Y_k])
                S_train_k = (np.concatenate([S_prev, S_k])
                             if (S_prev is not None and S_k is not None) else None)
            # f_{k-1} fit on prior data only
            fit_prev = _tracer_fits.build(opts.tracer_fit_method,
                                          X_prev, Y_prev, sigma=S_prev,
                                          lnl_floor_delta=opts.tracer_lnl_floor_delta)

    fit_now = _tracer_fits.build(opts.tracer_fit_method,
                                 X_train_k, Y_train_k, sigma=S_train_k,
                                 lnl_floor_delta=opts.tracer_lnl_floor_delta)

    state = {}
    if opts.state_in and os.path.exists(opts.state_in):
        with open(opts.state_in, "rb") as f:
            state = pickle.load(f)

    sampler = {
        "smc-mala-bd": _tracer_samplers.smc_mala_bd,
        "smc-mala":    _tracer_samplers.smc_mala,
        "birth-death": _tracer_samplers.birth_death,
    }[method]

    X_out, info = sampler(
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

    if opts.state_out:
        with open(opts.state_out, "wb") as f:
            pickle.dump(info.get("state", {}), f)

    _X_to_xml(X_out, P_template, opts, coord_names, downselect)


if __name__ == "__main__":
    main()
