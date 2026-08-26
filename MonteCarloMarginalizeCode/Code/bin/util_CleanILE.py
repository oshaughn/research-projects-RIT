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

import fileinput
#import StringIO

data_at_intrinsic = {}

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
opts = parser.parse_args()


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
        else:
#            print " new key ", line[1:9]
            data_at_intrinsic[tuple(line[1:col_intrinsic])] = [line[col_intrinsic:]]
      except Exception as exc:
          sys.stderr.write("Skipping malformed ILE row in {}: {}\n".format(fname, exc))
          continue

for key in data_at_intrinsic:
    lnL, sigmaOverL, ntot,neff =   np.transpose(data_at_intrinsic[key])
    lnL = np.atleast_1d(lnL); sigmaOverL = np.atleast_1d(sigmaOverL); ntot = np.atleast_1d(ntot); neff = np.atleast_1d(neff)
    sigmaOverL = np.maximum(sigmaOverL, 1e-7*np.ones(len(lnL)))   # prevent accidental underflow during debugging/using synthetic data with no error
    lnLmax = np.max(lnL)
    L = np.exp(lnL - lnLmax)  # remove overall Lmax factor, which factors out of the combination
    K = len(lnL)
    # Combine repeated evaluations by their SAMPLE-COUNT-weighted LINEAR mean.
    # DO NOT inverse-variance weight with the reported sigmas: each sigma is
    # computed from the same importance weights as its lnL, so a replica that
    # silently missed the likelihood peak reports BOTH a low lnL AND a small
    # sigma -- 1/sigma^2 weighting then overweights the worst replica, giving a
    # systematically low combined lnL with an overconfident combined error.
    # The pooled (ntot-weighted) linear mean is unbiased in L regardless.
    wts = np.asarray(ntot, dtype=float)
    if np.any(wts <= 0) or not np.all(np.isfinite(wts)):
        wts = np.ones(K)
    wts = wts/np.sum(wts)
    Lbar = np.sum(wts*L)
    lnLmeanMinusLmax = np.log(Lbar)
    # Error: max(propagated per-run sigmas, between-replica scatter).  Only the
    # scatter term can see the replica lottery (correlated underreporting); with
    # K replicas it has K-1 dof, so treat the result as a t-interval downstream.
    sigma_prop = np.sqrt(np.sum((wts*sigmaOverL*L)**2))/Lbar
    if K > 1:
        sigma_scatter = np.sqrt( np.sum(wts**2 * (L - Lbar)**2) * K/(K-1.) )/Lbar
    else:
        sigma_scatter = 0.
    sigmaNetOverL = max(sigma_prop, sigma_scatter)


    # The key already holds every intrinsic column that was present in the
    # input rows, in input order, so the composite preserves whatever
    # combination of advanced-physics groups the run enabled.
    print(-1, *key, lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
