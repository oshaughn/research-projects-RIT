#!/usr/bin/python
# Monica Rizzo

#  python ~/research-projects/MonteCarloMarginalizeCode/Code/util_GenerateMaxlnLWaveform_NRFromIndex.py --nr-group Sequence-SXS-All --nr-param 2 --run-dir G184098-v_test_hybrid-SingleSimNR-1-Sequence-SXS-All-Lmax2-fmin-10 --fname-indexed G184098-v_test_hybrid-SingleSimNR-1-Sequence-SXS-All-Lmax2-fmin-10_wrapup.indexed --verbose
#
# FIXME
#   - SXS: Switch to use-provided-strain, if it was used
#   - if maxpt not available, run ILE in single-point mode to find best fit parameters, adding --maximize-only
#   - omit memory

import numpy as np
import sys
import argparse
import os
import subprocess
from shutil import copyfile

import NRWaveformCatalogManager3 as nrwf
import lal
import RIFT.lalsimutils as lalsimutils

parser = argparse.ArgumentParser()
parser.add_argument("--l-max", type=int, default=2, help="Include all (l,m) modes with l less than or equal to this value.")
parser.add_argument("--run-dir",default=None, help="directory code was run in")
parser.add_argument("--nr-group",default=None,help="group")
parser.add_argument("--nr-param",default=None,help="param")
parser.add_argument("--fname-indexed",default=None,help="File name of *.indexed file")
parser.add_argument("--n-max",default=1e4,type=int,help="Override settings in command-single.sh")
parser.add_argument("--event-time", type=np.float128,  default=None,help="Event time. If not present, will use event.log")
parser.add_argument("--save-plots", default=False, action='store_true',help="saves waveform plot and hoft data")
parser.add_argument("--use-NR", default=False, action='store_true', help="generate plots using NR data")
parser.add_argument("--no-memory",action='store_true')
parser.add_argument("--sigma-cut",default=0.8,type=float)
parser.add_argument("--M-max-cut",default=200,type=float)
parser.add_argument("--verbose",action='store_true')
opts=parser.parse_args()

Lmax = opts.l_max


#look through run directory for file containing max lnl

#arrays to store lnl values and file numbers
lnl_vals = []
fnums = []

if not opts.fname_indexed:
    sys.exit(0)


#util_dir = "/home/oshaughn/research-projects/MonteCarloMarginalizeCode/Code/"

###
### # Parse  *.indexed (see util_NRBestOfIndex.py)
###
best_matches = {}
best_matches_masses ={}
full_spoke = {}
with open(opts.fname_indexed) as f:
 for line_str in f:
     # string split
     line = line_str.split()
     if opts.verbose:
         print("  Input ", line)
     # Get NR data
     if float(line[1])+float(line[2]) > opts.M_max_cut:
         continue
     group = line[3]
     if not (group == opts.nr_group):
         if opts.verbose:
             print(" Bad group ", group)
         continue
     if not nrwf.internal_ParametersAreExpressions.has_key(group):
         continue
     param = None
     if nrwf.internal_ParametersAreExpressions[group]:
         param = eval(line[4])
     else:
         param = line[4]

     if not (param == opts.nr_param):
         if opts.verbose:
             print(" Bad param", param)    # Problem: this can happen because of the lookup code! Equivalent physical parameters
         continue

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




###
### Create ILE job to follow it up
###

# Create xml file with correct mass and NR parameters 
m1,m2 = best_matches_masses[(opts.nr_group,opts.nr_param)]
wfP =  nrwf.WaveformModeCatalog(opts.nr_group, opts.nr_param, metadata_only=True)
wfP.P.assign_param('mtot',(m1+m2)*lal.MSUN_SI)
wfP.P.print_params()
P_list = [wfP.P]
lalsimutils.ChooseWaveformParams_array_to_xml(P_list, fname="single-pt")  # choice of fref irrelevant




        

with open(str(opts.run_dir)+"/command-single.sh",'r') as runfile:
    rf=str(runfile.readlines()[1])
    rf=rf.replace('create_event_dag_via_grid', 'integrate_likelihood_extrinsic')
    rf+=" --maximize-only"
    rf=rf.split()
    rf[rf.index("--sim-xml")+1]="single-pt.xml.gz"
    rf[rf.index("--output-file")+1]="ILE-single.xml.gz"  
    if "--n-copies" in rf:
        rf[rf.index("--n-copies")+1]=""
    if "--n-max" in rf:
        rf[rf.index("--n-max")+1]=" 10000 "
    rf_submit = ' '.join(rf)
    if "--n-copies" in rf:
        rf_submit=rf_submit.replace("--n-copies","")
        
    print(rf_submit)
    os.system(rf_submit)
       


    #acquire event time
    if opts.event_time == None:
        with open(str(opts.run_dir)+"/event.log", 'r') as log:
            for line in log:
                if "Time" in line: #line.startswith("End"):
                    event_time=line.split()[-1]
    else:
        event_time = opts.event_time
    
    infile = 'maxpt_ILE-single.xml.gz.xml.gz'

    cmd ="util_NRDumpDetectorResponse.py --inj "+infile+" --event 0 --t-ref "+str(event_time)+" --group "+opts.nr_group+' --param  '+str(opts.nr_param).replace(' ', '')+' --l ' + str(opts.l_max) + " --use-perturbative-extraction" 
    if opts.no_memory:
        cmd = cmd+ " --no-memory "
    
    if opts.save_plots:
        cmd+=" --save-plots --verbose"

os.chdir(opts.run_dir)
print(cmd)
os.system(cmd)

