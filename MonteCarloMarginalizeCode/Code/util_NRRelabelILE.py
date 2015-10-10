#! /usr/bin/env python
#
# util_NRRelabelILE.py
#
# GOAL
#    Accepts ILE default .dat output (indx m1 m2 s1x s1y s1z s2x s2y s2z XXX) and converts to group param XXX
#    If option specified, picks a *single* element which has the highest lnL
#
#


import argparse
import sys
import numpy as np
import lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools

import effectiveFisher  as eff   # for the mesh grid generation
import PrecessingFisherMatrix   as pcf   # Superior tools to perform overlaps. Will need to standardize with Evans' approach in effectiveFisher.py

from multiprocessing import Pool
try:
    import os
    n_threads = int(os.environ['OMP_NUM_THREADS'])
    print " Pool size : ", n_threads
except:
    n_threads=1
    print " - No multiprocessing - "

try:
	import NRWaveformCatalogManager as nrwf
	hasNR =True
except:
	hasNR=False
try:
    hasEOB=True
    import EOBTidalExternal as eobwf
except:
    hasEOB=False


###
### Load options
###

parser = argparse.ArgumentParser()
parser.add_argument("--group",default=None)
parser.add_argument("--fname", default=None, help="Base output file for ascii text (.dat) and xml (.xml.gz)")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
opts=  parser.parse_args()

if opts.verbose:
    True
    #lalsimutils.rosDebugMessagesContainer[0]=True   # enable error logging inside lalsimutils

if not opts.fname:
    sys.exit(0)

dat = np.loadtxt(opts.fname)
# check shape, fix if just one line
if len(dat.shape)<2:
    dat = np.array([dat])

best_matches = {}
for line in dat:
#    print line
    m1 = line[1]
    m2 = line[2]
    s1x,s1y,s1z= line[3:6]
    s2x,s2y,s2z= line[6:9]
    params_to_test = {}
    params_to_test['q'] = m2/m1
    params_to_test['s1x'] = s1x
    params_to_test['s1y'] = s1y
    params_to_test['s1z'] = s1z
    params_to_test['s2x'] = s2x
    params_to_test['s2y'] = s2y
    params_to_test['s2z'] = s2z
    matches = nrwf.NRSimulationLookup(params_to_test)
    if len(matches)>0:
        print matches[0][0], matches[0][1], line[9], line[10], line[11], line[12]
        if best_matches.has_key((matches[0])):
            if best_matches[matches[0]] < line[9]:
                best_matches[matches[0]] = line[9]
        else:
            best_matches[matches[0]] = line[9]

print " -----  BEST MATCHES ------ "  # unsorted
for key in best_matches:
    print key,  best_matches[key]
