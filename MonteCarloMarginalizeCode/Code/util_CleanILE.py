#! /usr/bin/env python
# util_CleanILE.py
#
#  Reads FILE (not stdin). Consolidates ILE entries for the same physical system.
#  Compare to: util_MassGriCoalesce.py


import sys
import os
import xmlutils
from optparse import OptionParser
from glue.ligolw import lsctables, table, utils

import numpy as np
import weight_simulations

import fileinput
import StringIO

data_at_intrinsic = {}

for fname in sys.argv[1:]:

    data = np.loadtxt(fname)
    for line in data:
        indx, m1,m2, s1x,s1y,s1z,s2x,s2y,s2z,lnL, sigmaOverL, ntot, neff = line
        if data_at_intrinsic.has_key(tuple(line[1:9])):
#            print " repeated occurrence ", line[1:9]
            data_at_intrinsic[tuple(line[1:9])].append( line[9:])
        else:
#            print " new key ", line[1:9]
            data_at_intrinsic[tuple(line[1:9])] = [line[9:]]

for key in data_at_intrinsic:
    lnL, sigmaOverL, ntot,neff =   np.transpose(data_at_intrinsic[key])
    lnLmax = np.max(lnL)
    sigma = sigmaOverL*np.exp(lnL-lnLmax)  # remove overall Lmax factor, which factors out from the weights constructed from \sigma
    wts = weight_simulations.AverageSimulationWeights(None, None,sigma)   
    lnLmeanMinusLmax = np.log(np.sum(np.exp(lnL - lnLmax)*wts))
    sigmaNetOverL = (np.sqrt(1./np.sum(1./sigma/sigma)))/np.exp(lnLmeanMinusLmax)


                                                    
    print -1,  key[0],key[1], key[2], key[3],key[4], key[5],key[6], key[7], lnLmeanMinusLmax+lnLmax, sigmaNetOverL, np.sum(ntot), -1
