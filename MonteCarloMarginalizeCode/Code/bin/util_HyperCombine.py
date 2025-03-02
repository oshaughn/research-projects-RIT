#! /usr/bin/env python
# util_CleanILE.py
#
#  Reads FILE (not stdin). Consolidates ILE entries for the same physical system.
#  Compare to: util_MassGriCoalesce.py


import sys
import os

import numpy as np
import RIFT.misc.weight_simulations as weight_simulations
import scipy.special

import fileinput
#import StringIO

data_at_intrinsic = {}

my_digits=7  # safety for high-SNR BNS


import argparse
parser = argparse.ArgumentParser(usage="util_HyperCombine.py fname1.dat fname2.dat ... ")
parser.add_argument("fname",action='append',nargs='+')
parser.add_argument("--combination",default='average',help='average|sum|product, depends on if we treat them as independent trials of the same quantity or independent')
opts = parser.parse_args()

n_terms = 0
enforce_length=False
if opts.combination != 'average':
    enforce_length=True

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
    if len(data.shape)==1:
        data = np.array([data])
    for line in data:
      try:
        line = np.around(line, decimals=my_digits)
        if tuple(line[2:]) in data_at_intrinsic:
#            print " repeated occurrence ", line[1:9]
            data_at_intrinsic[tuple(line[2:])].append(line[:2])
        else:
#            print " new key ", line[1:9]
            data_at_intrinsic[tuple(line[2:])] = [line[:2]]
        n_terms = np.max([n_terms, len(data_at_intrinsic[tuple(line[2:])]) ])
      except:
          continue

header_str=''
with open(opts.fname[0][0],'r') as f:
    header_str = f.readline()
    header_str = header_str.rstrip()

if header_str.startswith('#'):
    print(header_str)  # repeat the header, so we recapture al the syntax
for key in data_at_intrinsic:
    if len(key) < 1: 
        # sometimes we get spaces showing up, which is weird. Skip these. Should not happen
        continue
    if len(data_at_intrinsic[key]) < 1:
        # skip empty rows. This should not happen
        continue
    if enforce_length:
        if len(data_at_intrinsic[key]) < n_terms:
            print(" Incomplete data for {} ".format(key),file=sys.stderr)
            continue
    lnL, sigmaOverL =   np.transpose(data_at_intrinsic[key])
    lnLmax = np.max(lnL)
    if opts.combination =='product':
        lnLnet = np.sum(lnL)
        sigma = np.sqrt(np.sum(sigmaOverL**2))
        print(" {} {} ".format(lnLnet, sigma) + ' '.join(map(str,key)) )
    else:
        sigma = sigmaOverL*np.exp(lnL-lnLmax)  # remove overall Lmax factor, which factors out from the weights constructed from \sigma

        # This is an average, treating them as independent measurements
        wts = weight_simulations.AverageSimulationWeights(None, None,sigma)   
        lnLmeanMinusLmax = scipy.special.logsumexp((lnL - lnLmax)*wts)
        sigmaNetOverL = (np.sqrt(1./np.sum(1./sigma/sigma)))/np.exp(lnLmeanMinusLmax)

        if opts.combination =='sum':
            lnLmeanMinusLmax *= len(wts)   # not an average !
            sigmaNetOverL *= np.sqrt(len(wts))
                              
        print(" {} {} ".format(lnLmeanMinusLmax+lnLmax, sigmaNetOverL) + ' '.join(map(str,key)) )
