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
    with open("./"+str(opts.run_dir)+"/command-single.sh", 'r') as ile_opts:
         opts_list=ile_opts.read().split()
         nr_group=opts_list[opts_list.index("--nr-group")+1]
         nr_params=opts_list[opts_list.index("--nr-param")+1]

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

    cmd = "./util_LALDumpDetectorResponse.py --inj "+infile+" --event 0 --t-ref "+str(event_time)
    if opts.save_plots:
	cmd+=" --save-plots --verbose"

os.system(cmd)

