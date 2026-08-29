#! /usr/bin/env python
"""Combine per-waveform-model posteriors into one, weighted by p(m) Z_m.

This is the recombination step of the multi-approximant workflow.  The
iteration loop fits a cross-model MARGINALIZED likelihood (see
util_CleanILE.py --model-group-regex), so every model shares one intrinsic
grid; the terminal stage then forks, fitting each model separately to get its
own posterior samples and its own evidence.  This script closes the fork.

The posterior of the model-marginalized hypothesis is the mixture

    p(theta|d) = sum_m  w_m  p_m(theta|d),      w_m propto p(m) Z_m

so a model that fits the data better contributes proportionally more samples.
Sampling -- rather than carrying weights -- is what keeps the output a drop-in
replacement for a single-model extrinsic_posterior_samples.dat.

NOTE the two weights are different things and both are needed: p(m) is the
prior belief in a waveform model, Z_m is what the data say about it.  Passing
--model-prior alone does NOT give a prior-weighted mixture, because Z_m still
multiplies it; that is the intended Bayesian behaviour.
"""

import argparse
import os
import sys

import numpy as np

from RIFT.misc.cip_pipeline import systematic_resample, unique_draw_bound


def read_ln_evidence(fname):
    """Read ln Z from a CIP '+annotation.dat' file.

    Format written by util_ConstructIntrinsicPosterior_GenericCoordinates.py:
    a '# lnL sigma_lnL ...' header then one whitespace-separated row whose
    first field is ln_integrand_value_absolute.
    """
    with open(fname) as f:
        rows = [ln for ln in f if ln.strip() and not ln.strip().startswith("#")]
    if not rows:
        raise ValueError("{}: no evidence row".format(fname))
    value = float(rows[0].split()[0])
    if not np.isfinite(value):
        raise ValueError("{}: ln Z is not finite ({})".format(fname, value))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True,
                        help="LABEL:POSTERIOR.dat:ANNOTATION.dat (repeatable)")
    parser.add_argument("--model-prior", action="append", default=None,
                        help="LABEL=WEIGHT prior p(m) (repeatable). Default uniform.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-output-samples", type=int, default=None,
                        help="Requested output count (default: total input rows); "
                             "capped at the duplicate-free fair-draw frontier.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Set for a reproducible draw; default is unseeded.")
    opts = parser.parse_args(argv)

    priors = {}
    if opts.model_prior:
        for item in opts.model_prior:
            label, _, wt = item.partition("=")
            label = label.strip()
            if label in priors:
                parser.error("duplicate --model-prior for {}".format(label))
            value = float(wt)
            if not np.isfinite(value) or value < 0:
                parser.error("--model-prior weights must be finite and non-negative")
            priors[label] = value
        if sum(priors.values()) <= 0:
            parser.error("--model-prior weights must have positive total mass")

    labels, samples, header, ln_z = [], [], None, []
    for spec in opts.model:
        parts = spec.split(":")
        if len(parts) != 3:
            parser.error("--model wants LABEL:POSTERIOR.dat:ANNOTATION.dat, got {}".format(spec))
        label, post_file, annot_file = parts
        if label in labels:
            parser.error("duplicate --model label {}".format(label))
        for f in (post_file, annot_file):
            if not os.path.exists(f):
                sys.exit("util_CombineApproximantPosteriors: missing {}".format(f))
        with open(post_file) as f:
            first = f.readline()
        this_header = first.rstrip("\n") if first.startswith("#") else None
        if header is None:
            header = this_header
        elif this_header != header:
            # Silently column-mismatched posteriors would produce a garbage
            # mixture, so refuse rather than guess an alignment.
            sys.exit("util_CombineApproximantPosteriors: {} has a different column "
                     "header than the first model; refusing to mix".format(post_file))
        dat = np.atleast_2d(np.genfromtxt(post_file, comments="#"))
        if dat.size == 0:
            sys.exit("util_CombineApproximantPosteriors: {} has no samples".format(post_file))
        n_before = len(dat)
        dat = np.unique(dat, axis=0)
        if len(dat) < n_before:
            sys.stderr.write(
                "util_CombineApproximantPosteriors: WARNING: removed {} duplicate "
                "rows from {} before model allocation.\n".format(
                    n_before-len(dat), post_file))
        labels.append(label); samples.append(dat); ln_z.append(read_ln_evidence(annot_file))

    if priors:
        missing = [l for l in labels if l not in priors]
        if missing:
            sys.exit("--model-prior given but missing weights for {}".format(missing))
        ln_prior = np.array([np.log(priors[l]) if priors[l] > 0 else -np.inf for l in labels])
    else:
        ln_prior = np.zeros(len(labels))

    # w_m propto p(m) Z_m, in logs so a large ln Z spread cannot overflow.
    ln_w = np.array(ln_z) + ln_prior
    positive = np.isfinite(ln_w)
    if not np.any(positive):
        parser.error("model weights have no positive finite mass")
    scale = np.max(ln_w[positive])
    w = np.zeros(len(labels))
    w[positive] = np.exp(ln_w[positive] - scale)
    w = w/np.sum(w)

    n_requested = (opts.n_output_samples if opts.n_output_samples is not None
                   else int(sum(len(d) for d in samples)))
    if n_requested < 1:
        parser.error("--n-output-samples must be positive")
    # Treat every component row as an atom in the empirical mixture.  A row in
    # model m has mass w_m/N_m.  This is the same systematic fair-draw contract
    # as the final CIP export (PR #180), and avoids the overly conservative
    # per-model capacity floor used by an earlier implementation.
    row_weights = np.concatenate([
        np.full(len(dat), wt/len(dat)) for dat, wt in zip(samples, w)])
    all_samples = np.vstack(samples)
    n_total = min(n_requested, unique_draw_bound(row_weights))
    if n_total < n_requested:
        sys.stderr.write(
            "util_CombineApproximantPosteriors: WARNING: the unique fair-draw "
            "frontier supports only {} of {} requested mixture rows; reducing output rather "
            "than drawing duplicates.\n".format(n_total, n_requested))
    rng = np.random.default_rng(opts.seed)
    selected = systematic_resample(row_weights, n_total, rng=rng)
    offsets = np.cumsum([0] + [len(dat) for dat in samples])
    counts = np.array([
        np.sum((selected >= offsets[i]) & (selected < offsets[i+1]))
        for i in range(len(samples))])

    sys.stderr.write("util_CombineApproximantPosteriors: mixture over {} models\n".format(len(labels)))
    for label, lnz, wt, cnt, dat in zip(labels, ln_z, w, counts, samples):
        sys.stderr.write("   {:<20s} lnZ={:12.4f}  weight={:8.5f}  draws={:7d}  (of {} samples)\n".format(
            label, lnz, wt, cnt, len(dat)))
    dominant = np.max(w)
    if dominant > 0.99:
        sys.stderr.write(
            "util_CombineApproximantPosteriors: WARNING: one model carries {:.4f} of the "
            "mixture; the combined posterior is effectively single-model. A large ln Z "
            "gap between waveform models is usually a sign the models disagree far more "
            "than the statistical error, not that one is 'right'.\n".format(dominant))

    out = all_samples[selected]
    np.savetxt(opts.output, out, header=header[1:].strip() if header else "")
    sys.stderr.write("util_CombineApproximantPosteriors: wrote {} samples to {}\n".format(
        len(out), opts.output))


if __name__ == "__main__":
    main()
