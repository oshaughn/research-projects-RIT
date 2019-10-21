#! /usr/bin/env python
#
# Converts an ILE output (indexed) into STANDARD form, but applying an fmin cut, a sigma cut, and a number of points cut
# Can also use REFERENCE SPIN VALUES
#
# COMPARE TO
#   util_NRBestOfIndex.py
#
# EXAMPLE
#   python ~/research-projects/MonteCarloMarginalizeCode/Code/util_NRIndexedFminCutToILE.py --fname SXS-SingleSim_Ossokine_0234-Lmax2-v2_refine.indexed  --verbose


import argparse
import numpy as np
import RIFT.lalsimutils as lalsimutils
import NRWaveformCatalogManager3 as nrwf

import lal
MsunInSec = lal.MSUN_SI*lal.G_SI/lal.C_SI**3


###
### Load options
###

parser = argparse.ArgumentParser()
parser.add_argument("--fname", default=None, help="Base output file for ascii text (.dat) and xml (.xml.gz)")
parser.add_argument("--flow",default=30,type=float,help="Use a low frequency cutoff to eliminate points that don't satisfy the necessary condition")
parser.add_argument("--sigma-cut",default=0.95, type=float)
parser.add_argument("--min-npts",default=5e5,type=float)
parser.add_argument("--reference-spins", action='store_true')
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
opts=  parser.parse_args()



lnLhere  = 0
sigma_here=0
npts_here=0


dat_out = []
with open(opts.fname) as f:
 for line_str in f:
     # string split
     line = line_str.split()
     # Get NR data
     group = line[3]
     if not nrwf.internal_ParametersAreExpressions.has_key(group):
         if opts.verbose:
             print(" Cannot parse ",group)
         continue
     param = None
     if nrwf.internal_ParametersAreExpressions[group]:
         param = eval(line[4])
     else:
         param = line[4]
     if len(line)<6:
         if opts.verbose:
             print(" Line too short")
         continue
     m1 = eval(line[1])
     m2 = eval(line[2])
     try:
         meta_here = nrwf.internal_WaveformMetadata[group][param]
     except:
         print(" Skipping ", group, param)
         continue
     # in case line abbreviated for some reason
     try:
         lnLhere = float(line[5])  
         sigma_here = float(line[6])
         npts_here = float(line[7])
     except:
         if opts.verbose:
             print(" Failed on ", line)
         continue
     if sigma_here > opts.sigma_cut:  
         if opts.verbose:
             print(" Skipping large error ", line)
         continue
     if npts_here < opts.min_npts:
         continue

     try:
         if group == "Sequence-SXS-All":
             Momega0= nrwf.internal_WaveformMetadata[group][param]["Momega0"]
         else:
             Momega0 = nrwf.internal_EstimateStartingMOmega0[group][param]
     except:
         Momega0 = 0
     f0 = Momega0/(m1+m2)/MsunInSec/np.pi  # quadrupole frequency
     
     if True:
     #try:
         wfP = nrwf.WaveformModeCatalog(group,param, metadata_only=True)

         s1x,s1y,s1z = [wfP.P.s1x,wfP.P.s1y,wfP.P.s1z]
         s2x,s2y,s2z = [wfP.P.s2x,wfP.P.s2y,wfP.P.s2z]

         if opts.reference_spins:
             if not 'LAtReference' in nrwf.internal_WaveformMetadata[group][param]:
                 continue
             L = nrwf.internal_WaveformMetadata[group][param]['LAtReference']
             Lhat = L/np.sqrt(np.dot(L,L))
             frmL = lalsimutils.VectorToFrame(Lhat)
             hatX, hatY, hatZ = frmL

             if 'OrbitalPhaseAtReference' in nrwf.internal_WaveformMetadata[group][param]:
                 Phi_ref = nrwf.internal_WaveformMetadata[group][param]['OrbitalPhaseAtReference']
                 hatXp = hatX * np.cos(Phi_ref) + hatY*np.sin(Phi_ref)
                 hatYp = -hatX*np.sin(Phi_ref) + hatY*np.cos(Phi_ref)
                 hatX = hatXp
                 hatY = hatYp
                 # if np.dot(hatX,hatX)>1+1e-5:
                 #     print " Argh, units ", hatX, np.sqrt(np.dot(hatX,hatX)), Phi_ref
                 # if np.dot(hatY,hatY)>1+1e-5:
                 #     print " Argh, units ", hatY, np.sqrt(np.dot(hatY,hatY)), Phi_ref

             if 'Chi1AtReference' in nrwf.internal_WaveformMetadata[group][param]:
                 chi1 = nrwf.internal_WaveformMetadata[group][param]['Chi1AtReference']
                 if np.dot(chi1,chi1) > 1:
                     print(" ARRGH ", group,param)
                 s1x = np.dot(hatX,chi1)
                 s1y = np.dot(hatY,chi1)
                 s1z = np.dot(hatZ,chi1)
                 if s1x**2 + s1y**2+s1z**2>1:
                     print(" ARRGH ", group,param, np.sqrt(s1x**2 + s1y**2+s1z**2), np.sqrt(np.dot(chi1,chi1)))
             if 'Chi2AtReference' in nrwf.internal_WaveformMetadata[group][param]:
                 chi2 = nrwf.internal_WaveformMetadata[group][param]['Chi2AtReference']
                 if np.dot(chi2,chi2) > 1:
                     print(" ARRGH 2", group,param)
                 s2x = np.dot(hatX,chi2)
                 s2y = np.dot(hatY,chi2)
                 s2z = np.dot(hatZ,chi2)
                 if s2x**2 + s2y**2+s2z**2>1:
                     print(" ARRGH 2", group,param, np.sqrt(s2x**2 + s2y**2+s2z**2), np.sqrt(np.dot(chi2,chi2)))

         if f0 < opts.flow:
             line_out = [ -1, m1, m2, s1x,s1y,s1z, s2x,s2y,s2z, lnLhere, sigma_here,npts_here, float(line[-2])]
             if opts.verbose:
                 print(' '.join(map(str,line_out)))
             dat_out.append(line_out)
         else:
             if opts.verbose:
                 print(" Skipping line as too high frequency ", f0, line)
#     except:
     else:
         print(" FAILED TO PARSE ", line)
         continue


np.savetxt(opts.fname.replace('indexed', 'composite_cleaned'), np.array(dat_out))
