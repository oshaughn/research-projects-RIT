#!/usr/bin/python
#Monica Rizzo
# iterate.py --composite ../OfficialResultDatabase/*SEOBNRv3*.composite
#will create directory in location the code is run in 
#

import numpy as np
import numpy.linalg as la
import sys
import argparse
import os
import subprocess
from shutil import copyfile
import time


parser = argparse.ArgumentParser()
parser.add_argument("--l-max", type=int, default=2, help="Include all (l,m) modes with l less than or equal to this value.")
parser.add_argument("--run-dir",default=None, help="directory code was run in")
parser.add_argument("--max-iter",type=int,default=10,help="maximum iteration")
parser.add_argument("--approx",default="SEOBNRv2",help="approximant for output samples")
#parser.add_argument("--composite",default=None, help="takes one or multiple composite files")
#parser.add_argument("--save-plots", default=False, action='store_true',help="saves waveform plot and hoft data")
parser.add_argument("--postproc-opts",type=str,default=None, help="string containing postprocessing options")
parser.add_argument("--event-time",default=None)
opts=parser.parse_args()

def calc_kl(mu_1, mu_2, sigma_1, sigma_2, sigma_1_inv, sigma_2_inv):
    return 0.5*(np.trace(np.dot(sigma_2_inv,sigma_1))+np.dot(np.dot((mu_2-mu_1).T, sigma_2_inv), (mu_2-mu_1))-len(mu_1)+np.log(la.det(sigma_2)/la.det(sigma_1)))


#if ile directory is specified
if opts.run_dir:
   os.chdir(str(opts.run_dir))

   run_dir_full=os.getcwd()
   run_dir=run_dir_full.split('/')[-1]
 
   comp_file=None
   #find composite file 
   os.chdir(run_dir_full+"/../")
   for f in os.listdir(os.getcwd()):
       if str(f).endswith("composite") and str(run_dir) in str(f):
           comp_file=f
  
   #create composite file if not already exisitng
   if comp_file==None:
      wrapup="util_ILEdagPostprocess.sh "+run_dir+" "+run_dir+"_wrapup"
      comp_file=run_dir+"_wrapup.composite"
      os.system(wrapup)

   comp_file_full=os.getcwd()+"/"+comp_file
   #initialize iteration 
   
   it_count=np.array([0])
   for f in os.listdir(run_dir_full):
      if str(f).startswith("iteration"):
          it_count=np.append(it_count,int(f[-1]))
   if max(it_count)>0:
      it=max(it_count)
      comp_file_full=run_dir_full+"/iteration"+str(it)+"/iterate.composite"
      it+=1
   else:
      it=1

   iterate=True
   while iterate==True and it<opts.max_iter:

   #make directory to store iteration
      os.chdir(run_dir_full)
      iteration_dir="iteration"+str(it)
      os.mkdir(iteration_dir)
      os.chdir(run_dir_full+"/"+iteration_dir)

      with open(str(run_dir_full)+"/command-single.sh",'r') as runfile: 
         rf=str(runfile.readlines()[1])
         if "approximant" in rf:
           rf=rf.split()
           approx=str(rf[rf.index("--approximant")+1])
         else:
           approx=opts.approx
      if opts.approx!="SEOBNRv2":
         approx=opts.approx

      post_proc="util_ConstructIntrinsicPosterior_GenericCoordinates.py --fname "+comp_file_full+" "+opts.postproc_opts+" --approx-output "+approx
      os.system(post_proc)
     
      if it>2:  # 
          prev_iter="iteration"+str(it-1)
          #load gamma matrices
          gamma1=np.loadtxt(run_dir_full+"/"+prev_iter+"/lnL_gamma.dat", delimiter=" ")
          gamma2=np.loadtxt(run_dir_full+"/"+iteration_dir+"/lnL_gamma.dat", delimiter=" ")
          
          sigma1=la.inv(gamma1)
          sigma2=la.inv(gamma2)

          #load bestpts
          bestpt1=np.loadtxt(run_dir_full+"/"+prev_iter+"/lnL_bestpt.dat")
          bestpt2=np.loadtxt(run_dir_full+"/"+iteration_dir+"/lnL_bestpt.dat")

          kl=calc_kl(bestpt1, bestpt2, sigma1, sigma2, gamma1, gamma2)

          #diff=np.dot(np.dot(abs(bestpt1-bestpt2),mtx1),abs(bestpt1-bestpt2))

          print "KL Divergence:"
          print kl

          if kl<0.1:
              iterate=False

      if iterate==False:
         break
      else:
         #use samples to run again
         if opts.event_time:
            event_time=opts.event_time
         else:
            with open(str(run_dir_full)+"/event.log", 'r') as log:
               for line in log:
                  if line.startswith("End"):
                     event_time=line.split()[2]

         copyfiles="cp ../*.cache ../*psd* ./"
         os.system(copyfiles)
         try:
           copyfile("../event.log","event.log")
         except:
           pass

         with open(str(run_dir_full)+"/command-single.sh",'r') as runfile:
            rf=str(runfile.readlines()[1])
            rf=rf.replace('integrate_likelihood_extrinsic', 'create_event_dag_via_grid')
            rf=rf.split()
            rf[rf.index("--sim-xml")+1]="output-ILE-samples.xml.gz"
            rf_submit = ' '.join(rf)
            rf_submit+=" --n-copies 2"

         os.system(rf_submit)

         with open("integrate.sub", 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write("accounting_group = " +os.environ["LIGO_ACCOUNTING"] + '\n' + "accounting_group_user = "+os.environ["LIGO_USER_NAME"] + '\n' + content)
         
         os.system("condor_submit_dag -maxpre 50 -maxidle 50 -maxjobs 1000 marginalize_extrinsic_parameters_grid.dag")

         dag_complete=False
         while dag_complete==False:
           with open("marginalize_extrinsic_parameters_grid.dag.dagman.log") as dag_log:
              for l in dag_log:
                 if "terminated" in l:
                    dag_complete=True
              if dag_complete==False:
                 time.sleep(60)

         compile1="find ./ -name 'CME*.dat' -exec cat {} \; > iterate_tmp.dat"
         os.system(compile1)

         compile2="util_CleanILE.py iterate_tmp.dat | sort -rg -k10 > iterate.composite"
         os.system(compile2)
         comp_file_full=run_dir_full+"/"+iteration_dir+"/iterate.composite"
         it+=1


#elif opts.composite:
#   if len(opts.composite)>1:
#      #combine all composite files into one  
#      cmd="cat `ls "+str(opts.composite)+" | grep pp `| sort -rg -k10 > all.composite"
#      os.system(cmd)
#      comp_file="all.composite"
#    else:
#      comp_file=opts.composite


else:
  print "Please specify run directory"








