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
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools
import RIFT.interpolators.gp as gp

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


###
### Load options
###

parser = argparse.ArgumentParser()
#parser.add_argument("--group",default=None)
parser.add_argument("--fname", default=None, help="Base output file for ascii text (.dat) and xml (.xml.gz)")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--M-max-cut",type=float,default=1e5,help="Maximum mass to consider (e.g., if there is a cut on distance, this matters)")
parser.add_argument("--lnL-cut-up",default=5,type=float,help="Maximum amount (in lnL) a fit can increase lnL, based on what is calculated. Use a tighter threshold for a denser grid, and a wide threshold if you want to live dangerously.")
parser.add_argument("--sigma-cut",type=float,default=0.95,help="Eliminate points with large error from the fit.")
parser.add_argument("--fit", action="store_true",default=False, help="Local quadratic fit on best points")
parser.add_argument("--no-gp",action="store_true")
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
#     if opts.verbose:
#         print "  Input ", line
     # Get NR data
     if float(line[1])+float(line[2]) > opts.M_max_cut:
         continue
     group = line[3]
     if not nrwf.internal_ParametersAreExpressions.has_key(group):
         continue
     param = None
     if nrwf.internal_ParametersAreExpressions[group]:
         param = eval(line[4])
     else:
         param = line[4]

     if len(line)<6:
         continue

     key = (group,param)

     failure_mode=False; sigma_here = 1e6
     # in case line abbreviated for some reason
     try:
         lnLhere = float(line[5])  
         sigma_here = float(line[6])
         npts_here = float(line[7])
     except:
         lnLhere = -10
         sigma_here = 0.1
         npts_here = 1
         failure_mode=True
         continue
     if sigma_here > opts.sigma_cut:
         failure_mode=True
         continue  # DO NOT RECORD ITEMS which are completely unconverged (one point). Insane answers likely.  (should ALSO have tunable cutoff on accuracy)
     if not failure_mode:
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
        full_spoke[key].append([mtot, lnLhere,sigma_here])
      else:
         full_spoke[key] = [[mtot,lnLhere, sigma_here]]

print(" -----  BEST SINGLE POINT MATCHES ------ ")  # unsorted
if  opts.fit:
  print("  +++ Using fit code +++ ")
  for key in best_matches:
      full_spoke[key] = np.array(sorted(np.array(full_spoke[key]),key=lambda p: p[1]))
      lnLmaxHere = np.max(full_spoke[key][:,1])
      indx_peak = np.argmax(full_spoke[key][:,1])
      sigma_crit = full_spoke[key][indx_peak][2]
      n_required =np.max([4,np.min([10, 0.1*len(full_spoke[key])])])
      indx_crit =int(np.max([n_required-1,np.argmin( np.abs(full_spoke[key][:,1]-10))]))
      if opts.verbose:
	print(" Spoke needs ", key, indx_crit+1)
      reduced_spoke = full_spoke[key][-indx_crit:]
      mMin = np.min(reduced_spoke[:,0])
      mMax = np.max(reduced_spoke[:,0])
      weights = reduced_spoke[:,2]
      reduced_spoke[np.isnan(weights),2] = 0  # do not use NAN entries with errors
      reduced_spoke[np.isnan(weights),1] = 0  # do not use NAN entries with errors
#      print " Fitting ", key, reduced_spoke
      z=[]; mBestGuess = 0; lnLBestGuess = 0;
      try:
          z = np.polyfit(reduced_spoke[:,0], reduced_spoke[:,1],2,w=1./(reduced_spoke[:,2])) # Note TERRIBLE documentation/convention in numpy https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
          mBestGuess = -0.5*z[1]/z[0]
          lnLBestGuess = z[2] -0.25*z[1]**2/z[0] 
      except:
          print(" Interpolation failure (internal to polyfit, VERY UNUSUAL) for spoke ", key, " reverting to pointwise best ")
          indxMax = np.argmax(reduced_spoke[:,2])
          mBestGuess = reduced_spoke[indxMax,0]
          lnLBestGuess = reduced_spoke[indxMax,2]
          z = [0,0,lnLBestGuess]
      print(key, z[0], mBestGuess, lnLBestGuess, best_matches[key], sigma_crit)
      if z[2]<0 and mBestGuess> mMin and mBestGuess < mMax:
        if lnLBestGuess < lnLmaxHere+opts.lnL_cut_up and lnLBestGuess > lnLmaxHere-5*sigma_crit:  # do not allow arbitrary extrapolation
          if opts.verbose:
              print(" Replacing peak ", key, best_matches[key], " -> ", lnLBestGuess, " at mass ", mBestGuess)
              if lnLBestGuess > lnLmaxHere+5:
                  print(" VERY LARGE CHANGE FOR", key, lnLmaxHere, "->", lnLBestGuess)
          best_matches[key] = lnLBestGuess 
          orig_mtot = best_matches_masses[key][0]+best_matches_masses[key][1]
          orig_m1 = best_matches_masses[key][0]
          orig_m2 = best_matches_masses[key][1]
          best_matches_masses[key]=(mBestGuess* orig_m1/orig_mtot, mBestGuess* orig_m2/orig_mtot)
        else:
	  print(" Replacement rejected as out of range ", key, " reject ", lnLmaxHere, "->", lnLBestGuess, " : you probably need to rerun this spoke")
      if z[2]<0 and not ( mBestGuess> mMin and mBestGuess < mMax):
          print(" PLACEMENT FAILURE: ", key, mBestGuess, " outside of ", [mMin,mMax])
      

      ## REVISE BEST FIT VIA GAUSSIAN PROCESS
      #   GP is much less problematic for very crazy values that are far above trend due to bad MC luck
      #   Has VERY significant implications for rankings overall - often downranking
      try:
       if not opts.no_gp:
          my_gp = gp.GaussianProcess1d(reduced_spoke, sigma0=0.1,sigmab=0.2,h=1)
          xvals_dense = np.linspace(mMin,mMax,1000)
          yvals_dense = my_gp.predict(xvals_dense)[:,0]
          indx_gp_best = np.argmax(yvals_dense)
          print(" GP best fit ", xvals_dense[indx_gp_best], yvals_dense[indx_gp_best][0,0], "versus", mBestGuess, lnLmaxHere)
          if  mBestGuess-2 < xvals_dense[indx_gp_best] < mBestGuess+2 and yvals_dense[indx_gp_best][0,0] < lnLBestGuess+5*sigma_crit:
              mBestGuess = xvals_dense[indx_gp_best]
              lnLmaxHere= yvals_dense[indx_gp_best][0,0]
              best_matches[key] = lnLmaxHere
              orig_mtot = best_matches_masses[key][0]+best_matches_masses[key][1]
              orig_m1 = best_matches_masses[key][0]
              orig_m2 = best_matches_masses[key][1]
              best_matches_masses[key]=(mBestGuess* orig_m1/orig_mtot, mBestGuess* orig_m2/orig_mtot)
              print("  ....using GP best fit  for ", key, best_matches[key], best_matches_masses[key])

      except:
          print(" GP failure")


for key in best_matches:
  try:
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
    print(best_matches[key], key[0], str(key[1]).replace(' ',''),   best_matches_masses[key][0], best_matches_masses[key][1], tmax, xi, Mf, af)
  except:
     print("Skipping ", key)
