#! /usr/bin/env python
#
# util_NRBestOfIndex.py
#
# GOAL
#    Accepts ILE default .dat output (indx m1 m2 group param XXX).
#    Finds the *single best* case in that set.
#    ---> FIXME, SHOULD USE LOCAL QUADRATIC FIT <--
#
#
#   python ~/research-projects/MonteCarloMarginalizeCode/Code/util_NRBestOfIndex.py --fname RIT-Generic-Lmax2-vC2.indexed --group Sequence-RIT-Generic


import argparse
import sys
import numpy as np
import lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools

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


###
### Load options
###

parser = argparse.ArgumentParser()
#parser.add_argument("--group",default=None)
parser.add_argument("--fname", default=None, help="Base output file for ascii text (.dat) and xml (.xml.gz)")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
opts=  parser.parse_args()

if opts.verbose:
    True
    #lalsimutils.rosDebugMessagesContainer[0]=True   # enable error logging inside lalsimutils

if not opts.fname:
    sys.exit(0)



best_matches = {}
best_matches_masses ={}
full_spoke = {}

with open(opts.fname) as f:
 for line_str in f:
     # string split
     line = line_str.split()
     if opts.verbose:
         print "  Input ", line
     # Get NR data
     group = line[3]
     if not nrwf.internal_ParametersAreExpressions.has_key(group):
         pass
     param = None
     if nrwf.internal_ParametersAreExpressions[group]:
         param = eval(line[4])
     else:
         param = line[4]

     key = (group,param)
     lnLhere = float(line[5])
     if best_matches.has_key(key):
         if best_matches[key] < lnLhere:
                best_matches[key] = lnLhere
                best_matches_masses[key] = (float(line[1]),float(line[2]))
     else:
         best_matches[key] = line[5]
         best_matches_masses[key] = (float(line[1]),float(line[2]))

     # Add to spoke
     mtot = float(line[1])+ float(line[2])
     if full_spoke.has_key(key):
        full_spoke[key].append([mtot, lnLhere])
     else:
         full_spoke[key] = [[mtot,lnLhere]]

print " -----  BEST MATCHES ------ "  # unsorted
for key in best_matches:
    tmax =0
    if nrwf.internal_EstimatePeakL2M2Emission[key[0]].has_key(key[1]):
        tmax = nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]]
    af = -1000
    Mf = -1000
    if nrwf.internal_WaveformMetadata[key[0]][key[1]].has_key('ChiFMagnitude'):
        af = nrwf.internal_WaveformMetadata[key[0]][key[1]]['ChiFMagnitude']
    if nrwf.internal_WaveformMetadata[key[0]][key[1]].has_key('MF'):
        Mf = nrwf.internal_WaveformMetadata[key[0]][key[1]]['MF']
    wfP = nrwf.WaveformModeCatalog(key[0],key[1], metadata_only=True)
    xi = wfP.P.extract_param('xi')
    print best_matches[key], key,   best_matches_masses[key][0], best_matches_masses[key][1], tmax, xi, Mf, af
