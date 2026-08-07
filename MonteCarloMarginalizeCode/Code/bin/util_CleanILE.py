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

tides_on = False
distance_on = False  
col_intrinsic = 9

import argparse
parser = argparse.ArgumentParser(usage="util_CleanILE.py fname1.dat fname2.dat ... ")
parser.add_argument("fname",action='append',nargs='+')
parser.add_argument("--eccentricity", action="store_true")
parser.add_argument("--meanPerAno", action="store_true")
#Askold: adding specification for tabular eos file
parser.add_argument("--tabular-eos-file", action="store_true") 
opts = parser.parse_args()

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
        lambda1=lambda2=0
        eos_index = 0
        if opts.eccentricity:
            if opts.meanPerAno:
                indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,ecc,meanPerAno, lnL, sigmaOverL, ntot, neff = line
                col_intrinsic = 11
            else:
                indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,ecc, lnL, sigmaOverL, ntot, neff = line
                col_intrinsic = 10
        elif len(line) == 13 and (not tides_on) and (not distance_on):  # strip lines with the wrong length
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,lnL, sigmaOverL, ntot, neff = line
        elif  len(line) == 14:
            distance_on=True
            col_intrinsic=10
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,dist, lnL, sigmaOverL, ntot, neff = line
        elif len(line)==15:
            tides_on  = True
            col_intrinsic =11
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z, lambda1,lambda2,lnL, sigmaOverL, ntot, neff = line

        #Askold: adding the option for tabular eos file
        elif opts.tabular_eos_file and len(line) == 16: #checking if the tabular eos file is defined in the parser and if the line actually has all the columns
            #no eccentricity assumed here, since export_eos_index option doesn't output eccentricity, also it doesn't apply to neutron stars
            col_intrinsic = 12 #I assume eos_index to be intrinsic parameter
            indx, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, lambda1, lambda2, eos_index, lnL, sigmaOverL, ntot, neff = line 

        if sigmaOverL>0.9:
            continue    # do not allow poorly-resolved cases (e.g., dominated by one point). These are often useless
        if tuple(line[1:col_intrinsic]) in data_at_intrinsic:
#            print " repeated occurrence ", line[1:9]
            data_at_intrinsic[tuple(line[1:col_intrinsic])].append(line[col_intrinsic:])
        else:
#            print " new key ", line[1:9]
            data_at_intrinsic[tuple(line[1:col_intrinsic])] = [line[col_intrinsic:]]
      except:
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


    if opts.eccentricity:
        if opts.meanPerAno:
            print(-1, key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8], key[9], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
        else:
            print(-1, key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    elif tides_on:
        print(-1, key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8],key[9], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    elif distance_on:
        print(-1, key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    
    #Askold: new option for tabular eos file
    elif opts.tabular_eos_file: #written similarly to the previous ones
        print(-1, key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8],key[9], key[10],  lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    
    else:
        print(-1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
