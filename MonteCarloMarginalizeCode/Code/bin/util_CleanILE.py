#! /usr/bin/env python
# util_CleanILE.py
#
#  Reads FILE (not stdin). Consolidates ILE entries for the same physical system.
#  Compare to: util_MassGriCoalesce.py


import sys
import os
import RIFT.misc.xmlutils as xmlutils
#from optparse import OptionParser
from glue.ligolw import lsctables, table, utils

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
opts = parser.parse_args()

#print opts.fname

for fname in opts.fname[0]: #sys.argv[1:]:
    if os.stat(fname).st_size==0:  # skip files of zero length
        continue
    sys.stderr.write(fname)
#    data = np.loadtxt(fname)  # this will FAIL if we have a heterogeneous data source!  BE CAREFUL
    data = np.genfromtxt(fname,invalid_raise=False)  #  Protect against inhomogeneous data
    if len(data.shape) ==1:
        data = np.array([data]) # force proper treatment for single-line file
    for line in data:
      try:
        line = np.around(line, decimals=my_digits)
        lambda1=lambda2=0
        if len(line) == 13 and (not tides_on) and (not distance_on):  # strip lines with the wrong length
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,lnL, sigmaOverL, ntot, neff = line
        elif  len(line) == 14:
            distance_on=True
            col_intrinsic=10
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,dist, lnL, sigmaOverL, ntot, neff = line
        elif len(line)==15:
            tides_on  = True
            col_intrinsic =11
            indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z, lambda1,lambda2,lnL, sigmaOverL, ntot, neff = line
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
    lnLmax = np.max(lnL)
    sigma = sigmaOverL*np.exp(lnL-lnLmax)  # remove overall Lmax factor, which factors out from the weights constructed from \sigma
    wts = weight_simulations.AverageSimulationWeights(None, None,sigma)   
    lnLmeanMinusLmax = np.log(np.sum(np.exp(lnL - lnLmax)*wts))
    sigmaNetOverL = (np.sqrt(1./np.sum(1./sigma/sigma)))/np.exp(lnLmeanMinusLmax)


    if tides_on:
        print(-1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8],key[9], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    elif distance_on:
        print(-1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], key[8], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
    else:
        print(-1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1)
