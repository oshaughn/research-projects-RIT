#!/usr/bin/python
# Monica Rizzo

# util_GenerateMaxlnLWaveform.py --run-dir G268556-C01-SEOBNRv3-fmin20-fromPE-v0/ (name of run directory) --save-plots --use-NR (if using an NR run, otherwise uses util_LALDumpDetectorResponse.py) 
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

parser = argparse.ArgumentParser()
parser.add_argument("--l-max", type=int, default=2, help="Include all (l,m) modes with l less than or equal to this value.")
parser.add_argument("--run-dir",default=None, help="directory code was run in")
parser.add_argument("--event-time", type=np.float128,  default=None,help="Event time. If not present, will use event.log")
parser.add_argument("--save-plots", default=False, action='store_true',help="saves waveform plot and hoft data")
parser.add_argument("--use-NR", default=False, action='store_true', help="generate plots using NR data")
parser.add_argument("--no-memory",action='store_true')
parser.add_argument("--sigma-cut",default=0.8,type=float)
opts=parser.parse_args()

#look through run directory for file containing max lnl

#arrays to store lnl values and file numbers
lnl_vals = []
fnums = []

#util_dir = "/home/oshaughn/research-projects/MonteCarloMarginalizeCode/Code/"


bestline = []; lnl_vals=[]; fnums=[]
if opts.run_dir:
    for file in os.listdir(str(opts.run_dir)):
       if str(file).endswith(".dat") and str(file).startswith("CME"):
           with open(str(opts.run_dir)+"/"+str(file), 'r') as f:
               cme_file=str(opts.run_dir)+"/"+str(file)
               fnum=str(file).split('-')[2].replace('EVENT_','')
               fnums.append(fnum)
               line = f.read().split(' ')
               lnl = float(line[9])
               sigma_lnL = float(line[10])
               if np.isnan(sigma_lnL):
                   print(" Skipping ")
                   fnums.pop()
                   continue
               if len(lnl_vals) > 0 and  lnl > np.max(lnl_vals) and sigma_lnL<opts.sigma_cut:
                  bestline =line
                  print("Updating bestline ", lnl, max(lnl_vals), bestline)
               lnl_vals.append(float(lnl)) 
	      # print lnl, sigma_lnL, float(line[1])+float(line[2]), fnum
               if float(sigma_lnL) > opts.sigma_cut:
                  print(" skipping ")
                  fnums.pop(); lnl_vals.pop()
                  continue
                    
else:
   print("Please specify run directory")

# print max lnl value 
print("Max lnL value: "+ str(max(lnl_vals)))
indx=fnums[lnl_vals.index(max(lnl_vals))]
print(indx, bestline)


Lmax = opts.l_max
        
if opts.use_NR:
    print("Using NR data: ")
  
    infile = "" 
    #get name of maxpt file
    for file in os.listdir(str(opts.run_dir)):
       if str(file).startswith("maxpt") and "_"+str(indx) in str(file):
            infile=str(opts.run_dir)+"/"+str(file)
       elif str(file).startswith("maxpt") and "ILE-single" in str(file):
            infile=str(opts.run_dir)+"/"+str(file)

    # create single event to find maxpt if not already in existance
    if infile=='':
        cme_file = ""
        for file in os.listdir(str(opts.run_dir)):
            if ("EVENT_"+indx+"-") in file and file.endswith(".xml.gz.dat"):
                cme_file = file

        print(" Loading ", cme_file)
        param = np.loadtxt(cme_file)
        param = np.array(list(map(int,param*1000)))/1000.  # prevent scientific notation from appearing in arguments!
        print(param)
#        with open(cme_file, 'r') as params:
#		param=params.read().split()
#                param=[float(i) for i in param]
#		param=["%.20f" % i for i in param]
        write_xml = "util_WriteInjectionFile.py --parameter m1 --parameter-value "+str(param[1])+" --parameter m2 --parameter-value "+str(param[2])+" --parameter s1x --parameter-value "+str(param[3])+" --parameter s1y --parameter-value "+str(param[4])+" --parameter s1z --parameter-value "+str(param[5])+" --parameter s2x --parameter-value "+str(param[6])+" --parameter s2y --parameter-value "+str(param[7])+" --parameter s2z --parameter-value "+str(param[8])+" --approximant SEOBNRv4 --fname single-pt" 
        os.chdir(opts.run_dir)
        os.system(write_xml)
        
        with open(str(opts.run_dir)+"/command-single.sh",'r') as runfile:
           rf=str(runfile.readlines()[1])
           rf=rf.replace('create_event_dag_via_grid', 'integrate_likelihood_extrinsic')
           rf+=" --maximize-only"
           rf=rf.split()
           rf[rf.index("--sim-xml")+1]="single-pt.xml.gz"
           rf[rf.index("--output-file")+1]="ILE-single.xml.gz"  
           if "--n-copies" in rf:
              rf[rf.index("--n-copies")+1]=""
           rf_submit = ' '.join(rf)
           if "--n-copies" in rf:
              rf_submit=rf_submit.replace("--n-copies","")
        
        print(rf_submit)
        os.system(rf_submit)
       
        for file in os.listdir(str(opts.run_dir)): 
          if str(file).startswith("maxpt"):
             infile=str(opts.run_dir)+"/"+str(file)


    #acquire event time
    if opts.event_time == None:
     with open(str(opts.run_dir)+"/event.log", 'r') as log:
       for line in log:
          if "Time" in line: #line.startswith("End"):
              event_time=line.split()[-1]
    else:
        event_time = opts.event_time
    
    #get nr group and params
    with open(str(opts.run_dir)+"/command-single.sh", 'r') as ile_opts:
         opts_list=ile_opts.read().split()
         if "--l-max" in opts_list:
             Lmax_used = int(opts_list[opts_list.index("--l-max")+1])
             if (Lmax_used != Lmax):
                    print(" ---- WARNING, YOU WILL OVERRIDE THE MAXIMUM L USED  --")
         if "--nr-group" in opts_list:
             nr_group=opts_list[opts_list.index("--nr-group")+1]
             nr_params=opts_list[opts_list.index("--nr-param")+1]
         if "--nr-lookup-group" in opts_list:
                nr_group=opts_list[opts_list.index("--nr-lookup-group")+1]
                import RIFT.lalsimutils as lalsimutils
                import NRWaveformCatalogManager3 as nrwf
                nr_group = opts_list[opts_list.index("--nr-lookup-group")+1]
                print(" Looking up NR parameters from best fit parameters")
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
                print(" Parameter matching condition ", compare_dict)
                good_sim_list = nrwf.NRSimulationLookup(compare_dict,valid_groups=[nr_group])
                if len(good_sim_list)< 1:
                        print(" ------- NO MATCHING SIMULATIONS FOUND ----- ")
                        import sys
                        sys.exit(0)
                        print(" Identified set of matching NR simulations ", good_sim_list)
                try:
                        print("   Attempting to pick longest simulation matching  the simulation  ")
                        MOmega0  = 1
                        good_sim = None
                        for key in good_sim_list:
                                print(key, nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]])
                                if nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0'] < MOmega0:
                                        good_sim = key
                                        MOmega0 = nrwf.internal_WaveformMetadata[key[0]][key[1]]['Momega0']
                                print(" Picked  ",key,  " with MOmega0 ", MOmega0, " and peak duration ", nrwf.internal_EstimatePeakL2M2Emission[key[0]][key[1]])
                except:
                        good_sim  = good_sim_list[0] # pick the first one.  Note we will want to reduce /downselect the lookup process
                group = good_sim[0]
                nr_params = good_sim[1]

    cmd ="util_NRDumpDetectorResponse.py --inj "+infile+" --event 0 --t-ref "+str(event_time)+" --group "+nr_group+' --param  "'+str(nr_params).replace(' ', '')+'" --l ' + str(opts.l_max) + " --use-perturbative-extraction"  #-full"
    if opts.no_memory:
        cmd = cmd+ " --no-memory "
    
    if opts.save_plots:
        cmd+=" --save-plots --verbose"

else:
   #get name of maxpt file
    
    infile = ""
    #get name of maxpt file
    for file in os.listdir(str(opts.run_dir)):
       if str(file).startswith("maxpt") and "_"+str(indx) in str(file):
            infile=str(opts.run_dir)+"/"+str(file)
       elif str(file).startswith("maxpt") and "ILE-single" in str(file):
            infile=str(opts.run_dir)+"/"+str(file)

    # create single event to find maxpt if not already in existance
    if infile=='':
        with open(cme_file, 'r') as params:
                param=params.read().split()
                param=[float(i) for i in param]
                param=["%.20f" % i for i in param]
        write_xml = "util_WriteInjectionFile.py --parameter m1 --parameter-value "+str(param[1])+" --parameter m2 --parameter-value "+str(param[2])+" --parameter s1x --parameter-value "+str(param[3])+" --parameter s1y --parameter-value "+str(param[4])+" --parameter s1z --parameter-value "+str(param[5])+" --parameter s2x --parameter-value "+str(param[6])+" --parameter s2y --parameter-value "+str(param[7])+" --parameter s2z --parameter-value "+str(param[8])+" --approximant SEOBNRv4 --fname single-pt"
        os.chdir(opts.run_dir)
        os.system(write_xml)

        with open(str(opts.run_dir)+"/command-single.sh",'r') as runfile:
           rf=str(runfile.readlines()[1])
           rf=rf.replace('create_event_dag_via_grid', 'integrate_likelihood_extrinsic')
           rf+=" --maximize-only"
           rf=rf.split()
           rf[rf.index("--sim-xml")+1]="single-pt.xml.gz"
           rf[rf.index("--output-file")+1]="ILE-single.xml.gz"
           if "--n-copies" in rf:
              rf[rf.index("--n-copies")+1]=""
           rf_submit = ' '.join(rf)
           if "--n-copies" in rf:
              rf_submit=rf_submit.replace("--n-copies","")

        print(rf_submit)
        os.system(rf_submit)

        for file in os.listdir(str(opts.run_dir)):
          if str(file).startswith("maxpt"):
             infile=str(opts.run_dir)+"/"+str(file)



    #acquire event time
    with open(str(opts.run_dir)+"/event.log", 'r') as log:
       for line in log:
          if line.startswith("End"):
              event_time=line.split()[2]

    approx = "SEOBNRv4"
    #get nr group and params
    with open(str(opts.run_dir)+"/command-single.sh", 'r') as ile_opts:
         opts_list=ile_opts.read().split()
         if "--approximant" in opts_list:
                approx=opts_list[opts_list.index("--approximant")+1]



    cmd = "util_LALDumpDetectorResponse.py --inj "+infile+" --event 0 --t-ref "+str(event_time)+ " --approximant " + approx
    if opts.save_plots:
        cmd+=" --save-plots --verbose"

os.chdir(opts.run_dir)
print(cmd)
os.system(cmd)

