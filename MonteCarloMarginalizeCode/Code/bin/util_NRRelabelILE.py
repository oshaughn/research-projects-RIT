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
import RIFT.lalsimutils as lalsimutils


from multiprocessing import Pool
try:
    import os
    n_threads = int(os.environ['OMP_NUM_THREADS'])
    print(" Pool size : ", n_threads)
except:
    n_threads=1
    print(" - No multiprocessing - ")

try:
	import NRWaveformCatalogManager3 as nrwf
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
parser.add_argument("--eccentricity",action="store_true",default=False)
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
best_matches_masses ={}
best_matches_xi = {}
for line in dat:
#    print line
    m1 = line[1]
    m2 = line[2]
    s1x,s1y,s1z= line[3:6]
    s2x,s2y,s2z= line[6:9]
    if opts.eccentricity:
        eccentricity = line[9]
    params_to_test = {}
    params_to_test['q'] = m2/m1
    params_to_test['s1x'] = s1x
    params_to_test['s1y'] = s1y
    params_to_test['s1z'] = s1z
    params_to_test['s2x'] = s2x
    params_to_test['s2y'] = s2y
    params_to_test['s2z'] = s2z
    if opts.eccentricity:
        params_to_test['eccentricity'] = eccentricity
    if opts.group:
        matches = nrwf.NRSimulationLookup(params_to_test,valid_groups=[opts.group])
    else:
        matches = nrwf.NRSimulationLookup(params_to_test)
    if not len(matches)>0:
        pass
    # Pick the longest simulation"
    try:
      if len(matches)> 2:
            
        print("   Attempting to pick the longest simulation matching  the simulation from ", matches)
        MOmega0  = 1
        tmax = -1
        good_sim = None
        for key in matches:
#            print "    ... trying ", key
            # First choice is to look up by Momega0
            if 'Momega0' in nrwf.internal_WaveformMetadata[key[0]][key[1]]  and  tmax == -1:
                if nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0'] < MOmega0:
                    good_sim = key
                    MOmega0 = nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0']
            # second choice is to look up by NR peak value.  This may not be available (i.e., no preprocessed data)  , so the entire system could crash
            else:
                if nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]] > tmax:
                    good_sim = key
                    tmax = nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]]
#        print " Picked  ",key,  " with MOmega0 ", MOmega0, " and peak duration ", nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]]
        matches = [good_sim]
    except:
        matches  = [matches[0]] # pick the first one.  Note we will want to reduce /downselect the lookup process

    if len(matches)>0:
        lnL_indx = 9
        if opts.eccentricity:
            lnL_indx = 10
        if opts.verbose:
            print(matches[0][0], matches[0][1], line[lnL_indx], line[lnL_indx+1], line[lnL_indx+2], line[lnL_indx+3])
        if matches[0] in best_matches:
            if best_matches[matches[0]] < line[lnL_indx]:
#                print " Replacing "
                best_matches[matches[0]] = line[lnL_indx]
                best_matches_masses[matches[0]] = (line[1],line[2])
                q = m2/m1
                best_matches_xi[matches[0]] = (s1z + q*s2z)/(1+q)  # assumes simulation L is parallel to z
        else:
            best_matches[matches[0]] = line[lnL_indx]
            best_matches_masses[matches[0]] = (line[1],line[2])
            q = m2/m1
            best_matches_xi[matches[0]] = (s1z + q*s2z)/(1+q)  # assumes simulation L is parallel to z

print(" -----  BEST MATCHES ------ ")  # unsorted
for key in best_matches:
    tmax =0
    if key[1] in nrwf.internal_EstimatePeakL2M2Emission[key[0]]:
        tmax = nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]]
    af = -1000
    Mf = -1000
    if 'ChiFMagnitude' in nrwf.internal_WaveformMetadata[key[0]][key[1]]:
        af = nrwf.internal_WaveformMetadata[key[0]][key[1]]['ChiFMagnitude']
    if 'MF' in nrwf.internal_WaveformMetadata[key[0]][key[1]]:
        Mf = nrwf.internal_WaveformMetadata[key[0]][key[1]]['MF']
    print(best_matches[key], key,   best_matches_masses[key][0], best_matches_masses[key][1], tmax, best_matches_xi[key], Mf, af)
