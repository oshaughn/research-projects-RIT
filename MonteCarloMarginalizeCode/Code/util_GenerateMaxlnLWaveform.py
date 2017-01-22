#!/usr/bin/python
# Monica Rizzo

# ./generateMaxlnLWaveform.py --run-dir G268556-C01-SEOBNRv3-fmin20-fromPE-v0/ (name of run directory) --save-plots --use-NR (if using an NR run, otherwise uses util_LALDumpDetectorResponse.py) 

import numpy as np
import sys
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", default=None, help="directory code was run in")
parser.add_argument("--save-plots", default=False, action='store_true',help="saves waveform plot and hoft data")
parser.add_argument("--use-NR", default=False, action='store_true', help="generate plots using NR data")
opts=parser.parse_args()

#look through run directory for file containing max lnl

#arrays to store lnl values and file numbers
lnl_vals = []
fnums = []

if opts.run_dir:
    for file in os.listdir(str(opts.run_dir)):
       if str(file).endswith(".dat") and str(file).startswith("CME"):
	   with open(str(opts.run_dir)+"/"+str(file), 'r') as f:
	       fnum=str(file).split('-')[2].replace('EVENT_','')
               fnums.append(fnum)
	       lnl = f.read().split(' ')[9]
               lnl_vals.append(float(lnl))  
                    
else:
   print "Please specify run directory"

# print max lnl value and generate plots 
print "Max lnL value: "+ str(max(lnl_vals))
indx=fnums[lnl_vals.index(max(lnl_vals))]

        
if opts.use_NR:
    print "Using NR data: "
  
    infile = "" 
    #get name of maxpt file
    for file in os.listdir(str(opts.run_dir)):
       if str(file).startswith("maxpt") and "_"+str(indx) in str(file):
            infile=str(opts.run_dir)+"/"+str(file)

    #acquire event time
    with open(str(opts.run_dir)+"/event.log", 'r') as log:
       for line in log:
          if "Time" in line: #line.startswith("End"):
              event_time=line.split()[-1]

    #get nr group and params
    with open(str(opts.run_dir)+"/command-single.sh", 'r') as ile_opts:
         opts_list=ile_opts.read().split()
	 if "--nr-group" in opts_list:
         	nr_group=opts_list[opts_list.index("--nr-group")+1]
         	nr_params=opts_list[opts_list.index("--nr-param")+1]
	 if "--nr-lookup-group" in opts_list:
                nr_group=opts_list[opts_list.index("--nr-lookup-group")+1]
		import lalsimutils
		import NRWaveformCatalogManager as nrwf
	        nr_group = opts_list[opts_list.index("--nr-lookup-group")+1]
	        print " Looking up NR parameters from best fit parameters"
	        P_list = lalsimutils.xml_to_ChooseWaveformParams_array(infile)
	        P = P_list[0]
                compare_dict = {}
                compare_dict['q'] = P.m2/P.m1 # Need to match the template parameter. NOTE: VERY IMPORTANT that P is updated with the event params
                compare_dict['s1z'] = P.s1z
                compare_dict['s1x'] = P.s1x
                compare_dict['s1y'] = P.s1y
                compare_dict['s2z'] = P.s2z
                compare_dict['s2x'] = P.s2x
                compare_dict['s2y'] = P.s2y
                print " Parameter matching condition ", compare_dict
                good_sim_list = nrwf.NRSimulationLookup(compare_dict,valid_groups=[nr_group])
                if len(good_sim_list)< 1:
                        print " ------- NO MATCHING SIMULATIONS FOUND ----- "
                        import sys
                        sys.exit(0)
                        print " Identified set of matching NR simulations ", good_sim_list
                try:
                        print  "   Attempting to pick longest simulation matching  the simulation  "
                        MOmega0  = 1
                        good_sim = None
                        for key in good_sim_list:
                                print key, nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]]
                                if nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0'] < MOmega0:
                                        good_sim = key
                                        MOmega0 = nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0']
                                print " Picked  ",key,  " with MOmega0 ", MOmega0, " and peak duration ", nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]]
                except:
                        good_sim  = good_sim_list[0] # pick the first one.  Note we will want to reduce /downselect the lookup process
                group = good_sim[0]
                nr_params = good_sim[1]

    cmd ="util_NRDumpDetectorResponse.py --inj "+infile+" --event 0 --t-ref "+str(event_time)+" --group "+nr_group+" --param "+nr_params+" --use-perturbative-extraction"  #-full"
    
    if opts.save_plots:
	cmd+=" --save-plots --verbose"

else:
   #get name of maxpt file
    for file in os.listdir(str(opts.run_dir)):
       if str(file).startswith("maxpt") and "_"+str(indx) in str(file):
            infile=str(opts.run_dir)+str(file)

    #acquire event time
    with open(str(opts.run_dir)+"/event.log", 'r') as log:
       for line in log:
          if line.startswith("End"):
              event_time=line.split()[2]

    cmd = "util_LALDumpDetectorResponse.py --inj "+infile+" --event 0 --t-ref "+str(event_time)
    if opts.save_plots:
	cmd+=" --save-plots --verbose"

os.system(cmd)

