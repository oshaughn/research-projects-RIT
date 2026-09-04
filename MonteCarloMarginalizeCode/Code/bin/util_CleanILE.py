#! /usr/bin/env python
# util_CleanILE.py
#
#  Reads FILE (not stdin). Consolidates ILE entries for the same physical system.
#  Compare to: util_MassGriCoalesce.py


import sys
import os
import RIFT.misc.xmlutils as xmlutils
#from optparse import OptionParser
from igwn_ligolw import lsctables, table, utils

import numpy as np
import RIFT.misc.weight_simulations as weight_simulations

import re
import fileinput
#import StringIO

data_at_intrinsic = {}
models_at_intrinsic = {}   # same keys; parallel list of model labels, model-aware mode only
models_seen = []

my_digits=5  # safety for high-SNR BNS

import argparse
parser = argparse.ArgumentParser(usage="util_CleanILE.py fname1.dat fname2.dat ... ")
parser.add_argument("fname",action='append',nargs='+')
parser.add_argument("--a6c", action="store_true")
parser.add_argument("--hyperbolic", action="store_true")
parser.add_argument("--eccentricity", action="store_true")
parser.add_argument("--meanPerAno", action="store_true")
#Askold: adding specification for tabular eos file
parser.add_argument("--tabular-eos-file", action="store_true") 
parser.add_argument("--model-group-regex", default=None, help="Regex matched against each input file's BASENAME; capture group 1 is the waveform-model label.  Enables model-aware combination: replicas are averaged within a model, then models are marginalized over with --model-prior weights.  Without this flag every evaluation at a given intrinsic point is pooled flat, which is correct for replicas of ONE model and wrong across models.")
parser.add_argument("--model-prior", action="append", default=None, help="LABEL=WEIGHT prior weight for one model (repeatable).  Default: uniform over the labels actually seen.  Weights are renormalized over the models present at each intrinsic point.")
parser.add_argument("--expect-models", default=None, help="Comma-separated list of the models this run CONFIGURED.  Coverage is judged against this list, not against the labels that happen to appear: a model whose composites are all empty or missing is skipped before its label is ever recorded, so without this a total failure of one approximant looks like a complete run and even --require-all-models accepts every point.  The builder passes its full --approx list.")
parser.add_argument("--require-all-models", action="store_true", help="Drop intrinsic points not evaluated under EVERY model.  Without it, a point covered by a subset is marginalized over that subset, which silently changes the estimator point by point.")
opts = parser.parse_args()

model_mode = opts.model_group_regex is not None
model_rx = re.compile(opts.model_group_regex) if model_mode else None
expected_models = [m.strip() for m in opts.expect_models.split(",") if m.strip()] if opts.expect_models else None
model_prior_arg = {}
if opts.model_prior:
    for item in opts.model_prior:
        if "=" not in item:
            sys.exit("--model-prior wants LABEL=WEIGHT, got {}".format(item))
        label, _, wt = item.partition("=")
        model_prior_arg[label.strip()] = float(wt)
    if any(w < 0 for w in model_prior_arg.values()):
        sys.exit("--model-prior weights must be non-negative")


def expected_row_lengths(opts):
    """Column counts consistent with the enabled advanced-physics groups.

    An ILE row is composed as

        event_id m1 m2 s1x s1y s1z s2x s2y s2z
        [distance] [lambda1 lambda2 [eos_table_index]] [a6c] [E0 p_phi0]
        [eccentricity [meanPerAno]]
        lnL sigmaOverL ntotal neff

    (the ordering of the optional groups matches the ``col_lnL`` increment
    chain in util_ConstructIntrinsicPosterior_GenericCoordinates.py).  Each
    enabled flag contributes a KNOWN number of columns, so the groups compose:
    a run with --a6c --hyperbolic --eccentricity --meanPerAno writes all four.
    Tides / EOS index / pinned distance have no command-line flag here, so the
    row WIDTH is what distinguishes them; the allowed widths below are the
    flag-implied base plus each of those possibilities.
    """
    n_flag = 0
    if opts.a6c:
        n_flag += 1
    if opts.hyperbolic:
        n_flag += 2
    if opts.eccentricity:
        n_flag += 1
        if opts.meanPerAno:
            n_flag += 1
    lengths = set()
    lengths.add(13 + n_flag)      # no tides, no pinned distance
    lengths.add(13 + n_flag + 2)  # lambda1, lambda2
    lengths.add(13 + n_flag + 3)  # lambda1, lambda2, eos_table_index
    if n_flag == 0:
        lengths.add(14)           # pinned distance (written only on its own)
    return lengths


allowed_lengths = expected_row_lengths(opts)

#print opts.fname
from pathlib import Path
for fname in opts.fname[0]: #sys.argv[1:]:
    fname  = Path(fname).resolve()
    if not( os.path.exists(fname)):  # skip symbolic links that don't resolve : important for .composite files
        continue
    if os.stat(fname).st_size==0:  # skip files of zero length
        continue
    sys.stderr.write(str(fname)+"\n")
    this_model = None
    if model_mode:
        match = model_rx.search(os.path.basename(str(fname)))
        if match is None:
            sys.exit("--model-group-regex {!r} does not match {}; refusing to "
                     "pool it as an unlabelled model".format(
                         opts.model_group_regex, os.path.basename(str(fname))))
        this_model = match.group(1)
        if this_model not in models_seen:
            models_seen.append(this_model)
#    data = np.loadtxt(fname)  # this will FAIL if we have a heterogeneous data source!  BE CAREFUL
    data = np.genfromtxt(fname,invalid_raise=False)  #  Protect against inhomogeneous data
    if len(data.shape) ==1:
        data = np.array([data]) # force proper treatment for single-line file
    for line in data:
      try:
        line = np.around(line, decimals=my_digits)
        if len(line) not in allowed_lengths:  # strip lines with the wrong length
            raise ValueError("Unsupported ILE row layout: {} columns (expected one of {})".format(len(line), sorted(allowed_lengths)))
        # Whatever the enabled groups, the last four columns are
        # lnL sigmaOverL ntotal neff, so everything between the event id and
        # them is the intrinsic key used to consolidate repeated evaluations.
        col_intrinsic = len(line) - 4
        lnL, sigmaOverL, ntot, neff = line[col_intrinsic:]
        if sigmaOverL>0.9:
            continue    # do not allow poorly-resolved cases (e.g., dominated by one point). These are often useless
        if tuple(line[1:col_intrinsic]) in data_at_intrinsic:
#            print " repeated occurrence ", line[1:9]
            data_at_intrinsic[tuple(line[1:col_intrinsic])].append(line[col_intrinsic:])
            models_at_intrinsic[tuple(line[1:col_intrinsic])].append(this_model)
        else:
#            print " new key ", line[1:9]
            data_at_intrinsic[tuple(line[1:col_intrinsic])] = [line[col_intrinsic:]]
            models_at_intrinsic[tuple(line[1:col_intrinsic])] = [this_model]
      except Exception as exc:
          sys.stderr.write("Skipping malformed ILE row in {}: {}\n".format(fname, exc))
          continue

def _pool_linear(lnL, sigmaOverL, ntot, weights=None):
    """Combine evaluations of the SAME quantity by their weighted linear mean in L.

    Returns (lnLbar, sigmaOverL).  The linear arithmetic is performed relative
    to this pool's own maximum, so the largest member is exactly one and Lbar
    cannot underflow to zero merely because some *other* model is much better.

    DO NOT inverse-variance weight with the reported sigmas: each sigma is
    computed from the same importance weights as its lnL, so a replica that
    silently missed the likelihood peak reports BOTH a low lnL AND a small
    sigma -- 1/sigma^2 weighting then overweights the worst replica, giving a
    systematically low combined lnL with an overconfident combined error.
    The pooled (ntot-weighted) linear mean is unbiased in L regardless.

    Error: max(propagated per-run sigmas, between-replica scatter).  Only the
    scatter term can see the replica lottery (correlated underreporting); with
    K replicas it has K-1 dof, so treat the result as a t-interval downstream.
    """
    lnLscale = np.max(lnL)
    L = np.exp(lnL - lnLscale)
    K = len(lnL)
    if weights is None:
        wts = np.asarray(ntot, dtype=float)
        if np.any(wts <= 0) or not np.all(np.isfinite(wts)):
            wts = np.ones(K)
    else:
        wts = np.asarray(weights, dtype=float)
    wts = wts/np.sum(wts)
    Lbar = np.sum(wts*L)
    sigma_prop = np.sqrt(np.sum((wts*sigmaOverL*L)**2))/Lbar
    if K > 1:
        sigma_scatter = np.sqrt( np.sum(wts**2 * (L - Lbar)**2) * K/(K-1.) )/Lbar
    else:
        sigma_scatter = 0.
    return lnLscale + np.log(Lbar), max(sigma_prop, sigma_scatter)


if model_mode:
    if expected_models:
        absent = [m for m in expected_models if m not in models_seen]
        if absent:
            sys.stderr.write(
                "util_CleanILE: WARNING: configured models contributed NOTHING: {}. "
                "Their composites were empty or missing, so they are invisible to the "
                "per-point coverage check and this is a {}-model mixture, not a {}-model "
                "one.\n".format(", ".join(absent), len(models_seen), len(expected_models)))
        # judge coverage against what was CONFIGURED
        for m in expected_models:
            if m not in models_seen:
                models_seen.append(m)
    if model_prior_arg:
        missing = [m for m in models_seen if m not in model_prior_arg]
        if missing:
            sys.exit("--model-prior given but missing weights for {}; specify "
                     "every model or none".format(missing))
    sys.stderr.write("util_CleanILE: model-aware combination over {} models: {}\n".format(
        len(models_seen), ", ".join(models_seen)))

n_partial = 0
n_dropped_partial = 0
n_points = 0
model_spread = []   # between-model scatter, reported but NOT put in sigmaOverL

for key in data_at_intrinsic:
    lnL, sigmaOverL, ntot,neff =   np.transpose(data_at_intrinsic[key])
    lnL = np.atleast_1d(lnL); sigmaOverL = np.atleast_1d(sigmaOverL); ntot = np.atleast_1d(ntot); neff = np.atleast_1d(neff)
    sigmaOverL = np.maximum(sigmaOverL, 1e-7*np.ones(len(lnL)))   # prevent accidental underflow during debugging/using synthetic data with no error
    if not model_mode:
        # One model (or replicas of one model): pool everything flat.
        lnLmean, sigmaNetOverL = _pool_linear(lnL, sigmaOverL, ntot)
    else:
        # Two levels, because replicas and models are not the same thing.
        # Within a model, replicas estimate ONE number -> ntot-weighted mean.
        # Across models, the quantity itself differs -> marginalize,
        #   L_marg(lambda) = sum_m p(m) L_m(lambda),
        # which is linear in L, not in lnL.  Averaging lnL instead would give
        # the geometric mean (a logarithmic opinion pool), which is not
        # Bayesian model marginalization.
        labels = models_at_intrinsic[key]
        present = [m for m in models_seen if m in labels]
        if len(present) < len(models_seen):
            if opts.require_all_models:
                n_dropped_partial += 1
                continue
            n_partial += 1
        lnL_m = []; sig_m = []; w_m = []
        for m in present:
            sel = np.array([lab == m for lab in labels])
            lnLm, sm = _pool_linear(lnL[sel], sigmaOverL[sel], ntot[sel])
            lnL_m.append(lnLm); sig_m.append(sm)
            w_m.append(model_prior_arg[m] if model_prior_arg else 1.0)
        lnL_m = np.atleast_1d(np.array(lnL_m)); sig_m = np.atleast_1d(np.array(sig_m))
        w_m = np.atleast_1d(np.array(w_m, dtype=float))
        if np.sum(w_m) <= 0:
            sys.exit("intrinsic point {} is covered only by zero-prior models; "
                     "cannot define the subset mixture".format(key))
        # Renormalized over the models PRESENT here: with partial coverage the
        # estimator is a marginal over a subset, which is why n_partial is
        # reported and --require-all-models exists.
        w_m = w_m/np.sum(w_m)
        # Combine the model means on a new common scale.  Each model was pooled
        # on its own scale above; converting only the final model means relative
        # to their maximum is the log-sum-exp construction and remains finite
        # even when the models differ by thousands of nats.
        positive = w_m > 0
        # A zero-prior model must not set the numerical scale: if it is 1000
        # nats louder than every positive-prior model, scaling by it makes all
        # contributing likelihoods underflow and returns -inf/nan.
        lnL_pos = lnL_m[positive]
        sig_pos = sig_m[positive]
        w_pos = w_m[positive]
        w_pos = w_pos/np.sum(w_pos)
        lnL_model_scale = np.max(lnL_pos)
        L_m_scaled = np.exp(lnL_pos - lnL_model_scale)
        Lbar_scaled = np.sum(w_pos*L_m_scaled)
        lnLmean = lnL_model_scale + np.log(Lbar_scaled)
        # ACROSS MODELS, report ONLY the propagated integration uncertainty.
        #
        # An earlier version also took the between-model scatter, reasoning that
        # it is the waveform-systematic contribution and should widen the
        # downstream fit.  It does the opposite.  sigmaOverL is an INTEGRATION
        # error, and CIP drops every row above --sigma-cut (default 0.6,
        # util_ConstructIntrinsicPosterior_GenericCoordinates.py).  Model
        # disagreement large enough to matter therefore exceeds the cut and
        # DELETES precisely the intrinsic points where the models disagree --
        # exactly the points this workflow exists to fit.
        #
        # The model variation is already carried by Lbar, which is the
        # marginalized likelihood.  It does not belong in the error bar too.
        sigmaNetOverL = np.sqrt(np.sum(
            (w_pos*sig_pos*L_m_scaled)**2))/Lbar_scaled
        M = len(lnL_pos)
        if M > 1:
            # kept as a diagnostic only -- never folded into sigmaNetOverL
            spread = np.sqrt(np.sum(
                w_pos**2 * (L_m_scaled - Lbar_scaled)**2)
                * M/(M-1.))/Lbar_scaled
            model_spread.append(spread)

    n_points += 1


    # The key already holds every intrinsic column that was present in the
    # input rows, in input order, so the composite preserves whatever
    # combination of advanced-physics groups the run enabled.
    print(-1, *key, lnLmean, sigmaNetOverL, np.sum(ntot), -1)


# Coverage report.  stdout is the data stream, so this goes to stderr.
if model_mode:
    sys.stderr.write(
        "util_CleanILE: {} intrinsic points written, {} models\n".format(
            n_points, len(models_seen)))
    if n_dropped_partial:
        sys.stderr.write(
            "util_CleanILE: DROPPED {} intrinsic points not evaluated under all "
            "{} models (--require-all-models)\n".format(
                n_dropped_partial, len(models_seen)))
    if model_spread:
        sys.stderr.write(
            "util_CleanILE: between-model scatter (diagnostic only, NOT folded "
            "into sigmaOverL): median {:.4f}, max {:.4f} over {} multi-model "
            "points\n".format(float(np.median(model_spread)),
                              float(np.max(model_spread)), len(model_spread)))
    if n_partial:
        sys.stderr.write(
            "util_CleanILE: WARNING: {} of {} intrinsic points were evaluated "
            "under only a SUBSET of the {} models, and were marginalized over "
            "that subset.  The estimator therefore differs point to point -- a "
            "model-dependent, lambda-dependent change in the effective prior. "
            "Common causes: the sigmaOverL>0.9 resolution cut firing for one "
            "model but not another, or a model's ILE job failing. Use "
            "--require-all-models to drop these instead.\n".format(
                n_partial, n_points, len(models_seen)))
