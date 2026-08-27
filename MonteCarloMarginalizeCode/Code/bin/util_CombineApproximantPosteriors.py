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
                        help="Default: the total number of input samples.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Set for a reproducible draw; default is unseeded.")
    opts = parser.parse_args(argv)

    priors = {}
    if opts.model_prior:
        for item in opts.model_prior:
            label, _, wt = item.partition("=")
            priors[label.strip()] = float(wt)

    labels, samples, header, ln_z = [], [], None, []
    for spec in opts.model:
        parts = spec.split(":")
        if len(parts) != 3:
            parser.error("--model wants LABEL:POSTERIOR.dat:ANNOTATION.dat, got {}".format(spec))
        label, post_file, annot_file = parts
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
    ln_w -= np.max(ln_w)
    w = np.exp(ln_w)
    w = w/np.sum(w)

    n_total = opts.n_output_samples or int(sum(len(d) for d in samples))
    rng = np.random.default_rng(opts.seed)
    counts = rng.multinomial(n_total, w)

    sys.stderr.write("util_CombineApproximantPosteriors: mixture over {} models\n".format(len(labels)))
    for label, lnz, wt, cnt, dat in zip(labels, ln_z, w, counts, samples):
        sys.stderr.write("   {:<20s} lnZ={:12.4f}  weight={:8.5f}  draws={:7d}  (of {} samples)\n".format(
            label, lnz, wt, cnt, len(dat)))
    # Drawing more samples from a model than it has is legal (with replacement)
    # but degrades the effective sample size, and does so invisibly: the output
    # file still has the requested number of rows.
    starved = [(l, int(c), len(d)) for l, c, d in zip(labels, counts, samples) if c > len(d)]
    if starved:
        sys.stderr.write(
            "util_CombineApproximantPosteriors: WARNING: drawing more samples than "
            "available for {}; those rows are duplicates and the effective sample "
            "size is smaller than the row count. Give the favoured model more CIP "
            "output samples, or lower --n-output-samples.\n".format(
                ", ".join("{} ({} draws from {})".format(*x) for x in starved)))

    dominant = np.max(w)
    if dominant > 0.99:
        sys.stderr.write(
            "util_CombineApproximantPosteriors: WARNING: one model carries {:.4f} of the "
            "mixture; the combined posterior is effectively single-model. A large ln Z "
            "gap between waveform models is usually a sign the models disagree far more "
            "than the statistical error, not that one is 'right'.\n".format(dominant))

    drawn = [dat[rng.integers(0, len(dat), size=cnt)] for dat, cnt in zip(samples, counts) if cnt > 0]
    out = np.vstack(drawn)
    rng.shuffle(out)
    np.savetxt(opts.output, out, header=header[1:].strip() if header else "")
    sys.stderr.write("util_CombineApproximantPosteriors: wrote {} samples to {}\n".format(
        len(out), opts.output))


if __name__ == "__main__":
    main()
