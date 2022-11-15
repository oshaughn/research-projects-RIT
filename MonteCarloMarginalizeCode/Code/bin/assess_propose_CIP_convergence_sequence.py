#! /bin/bash
#
#  GOAL
#     * assess work done in a directory with lots of CIP standard output
#     * if not achieving goals, write subdag
import argparse
import sys
import os
import glob
import shutil
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import functools
import itertools

from glue import pipeline # https://github.com/lscsoft/lalsuite-archive/blob/5a47239a877032e93b1ca34445640360d6c3c990/glue/glue/pipeline.py

import RIFT.misc.dag_utils as dag_utils
from RIFT.misc.dag_utils import mkdir
from RIFT.misc.dag_utils import which


parser = argparse.ArgumentParser()
parser.add_argument("--working-directory",default="./")
parser.add_argument("--iteration-number",type=int,default=0,help="Just to keep track")
parser.add_argument("--n-eff-goal",default=10000,type=int,help="Target n-eff goal")
parser.add_argument("--n-eff-per-worker",default=200,type=int,help="n-eff goal per worker CIP ")
parser.add_argument("--cip-exe",default=None,help="filename of CIP or equivalent executable. Will default to `which util_ConstructIntrinsicPosterior_GenericCoordinates` in low-level code")
parser.add_argument("--cip-args",default=None,help="filename of args_cip.txt file  which holds CIP arguments.  Should NOT conflict with arguments auto-set by this DAG ... in particular, i/o arguments will be modified.   We will NOT change --n-max, so user is assumed to have set this intelligently before trying this.  ASSUME INITIAL ITERATION COUNT ALREADY STRIPPED, and do NOT have --n-eff present")
parser.add_argument('--n-iterations',default=10,type=int,help="Maximum number of iterations to perform before terminating")
parser.add_argument("--cip-explode-jobs-min",default=3,type=int,help="Min number of jobs to use in parallel.  ")
parser.add_argument("--cip-explode-jobs-max",default=3,type=int,help="Min number of jobs to use in parallel.  ")
parser.add_argument("--transfer-file-list",default=None,help="File containing list of *input* filenames to transfer, one name per file. Copied into transfer_files for condor directly.  If provided, also enables attempts to deduce files that need to be transferred for the pipeline to operate, as needed for OSG, etc")
parser.add_argument("--request-memory-CIP",default=16384,type=int,help="Memory request for condor (in Mb) for fitting jobs.")
parser.add_argument("--use-singularity",action='store_true',help="Attempts to use a singularity image in SINGULARITY_RIFT_IMAGE")
parser.add_argument("--use-osg",action='store_true',help="Attempts to set up an OSG workflow.  Must submit from an osg allowed submit machine")
parser.add_argument("--use-osg-simple-requirements",action='store_true',help="Uses an aggressive simplified requirements string for worker jobs")
parser.add_argument("--condor-local-nonworker",action='store_true',help="Uses local universe for non-worker condor jobs. Important to run in non-NFS location, as other jobs don't have file transfer set up.")
parser.add_argument("--condor-nogrid-nonworker",action='store_true',help="Uses local flocking for non-worker condor jobs. Important to run in non-NFS location, as other jobs don't have file transfer set up.")
parser.add_argument("--general-retries",default=0,type=int,help="Number of retry attempts for internal jobs (convert, CIP, ...). (These can fail, albeit more rarely, usually due to filesystem problems)")
parser.add_argument("--general-request-disk",default="4M",type=str,help="Request disk passed to condor. Must be done for all jobs now")
opts=  parser.parse_args()




###
### Change to working directory, assess work done towards goal
###
os.chdir(opts.working_directory)
fnames_output = list(glob.glob("overlap-grid-*+annotation.dat"))
# match only desired string: end with +annotation, remove withpriorchange
fnames_output  = [x for x in fnames_output if 'annotation' in x]
fnames_output  = [x for x in  fnames_output if not ('lnL' in x)]
print(" Output so far  ", fnames_output)
print(" Number of instances spawned :", len(fnames_output))  # hopefully not an overcount due to pipeline fails - hopefully only existing at end.

# # Get current number of workers
# if not(os.path.exists("pipe_n_workers_now.dat")):
#     n_workers_now =opts.cip_explode_jobs_min
# else:
#     n_workers_now = np.loadtxt("pipe_n_workers_now.dat") # single number, all 

# Get n_eff record from last iteration, total completed to date
if not(os.path.exists("pipe_n_eff_now.dat")):
    n_workers_now =opts.cip_explode_jobs_min
else:
    n_workers_now =np.loadtxt("pipe_n_eff_now.dat") # single number

# Get n_eff
n_eff_list = []
for fname in fnames_output:
    dat = np.loadtxt(fname)
    n_eff_list.append(dat[-1]) # single-line file
n_eff_so_far = np.sum(np.array(n_eff_list))
# Update file
n_eff_observed_per_worker = opts.n_eff_per_worker
if len(n_eff_list)>5:  # if we have enough experience, use the average
    n_eff_observed_per_worker = np.mean(np.array(n_eff_list))
n_eff_remaining = opts.n_eff_goal  - n_eff_so_far
if n_eff_remaining <=0:
    sys.exit(1)  # return failure (halt dag) because success !  We are done.
n_workers_proposed = np.min([opts.cip_explode_jobs_max, int(n_eff_remaining/n_eff_observed_per_worker)+1])


###
### PARSE REST OF ARGS

local_worker_universe="vanilla"
no_worker_grid=False
if opts.condor_local_nonworker:
    local_worker_universe="local"
if opts.condor_nogrid_nonworker:
    no_worker_grid=True


working_dir_inside_local = working_dir_inside = opts.working_directory
os.chdir(opts.working_directory)
# working_dir_inside_local is for jobs on local nodes
if opts.use_singularity or opts.use_osg:
    working_dir_inside = "./" # all files on the remote machine are in the current directory

singularity_image = None
if opts.use_singularity:
    print(" === USING SINGULARITY === ")
    singularity_image = os.environ["SINGULARITY_RIFT_IMAGE"]  # must be present to use singularity
    # SINGULARITY IMAGES ARE ON CVMFS, SO WE CAN AVOID THE SINGULARITY EXEC CALL
    # hardcoding a fiducial copy of lalapps_path2cache; beware about the executable name change
    os.environ['LALAPPS_PATH2CACHE'] = "/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/envs/igwn-py39/bin/lalapps_path2cache" #"singularity exec {singularity_image} lalapps_path2cache".format(singularity_image=singularity_image)
    print(singularity_image)

if (opts.cip_args is None):
    print(" No arguments provided for low-level job")
    sys.exit(0)

if True:
    with open(opts.cip_args) as f:
        cip_args_list = f.readlines()
    cip_args = ' '.join( [x.replace('\\','') for x in cip_args_list] )
    cip_args = ' '.join(cip_args.split(' ')[1:])
    # Some argument protection for later
    cip_args = cip_args.replace('[', ' \'[')
    cip_args = cip_args.replace(']', ']\'')
    cip_args=cip_args.rstrip()
    cip_args += ' --no-plots '  

print("CIP", cip_args)

transfer_file_names = []
if not (opts.transfer_file_list is None):
    transfer_file_names=[]
    # Load args.txt. Remove first item.  Store
    with open(opts.transfer_file_list) as f:
        for  line in f.readlines():
            transfer_file_names.append(line.rstrip())
    print(" Input files to transfer to job working directory (note!)", transfer_file_names)


# Write dag with just those jobs present
# ASSUMES log files exist

dag = pipeline.CondorDAG(log=os.getcwd())


##   Fit job: default case
cip_args_base = cip_args
out_dir_base = opts.working_directory
cip_exe = opts.cip_exe
cip_job, cip_job_name = dag_utils.write_CIP_sub(tag='CIP',log_dir=None,arg_str=cip_args_base,request_memory=opts.request_memory_CIP,input_net=opts.working_directory+'/all.net',output='overlap-grid-subdag-{}-$(process)'.format(opts.iteration_number),out_dir=out_dir_base,exe=cip_exe,universe=local_worker_universe,no_grid=no_worker_grid)
# Modify: set 'initialdir'
cip_job.add_condor_cmd("initialdir",opts.working_directory)
# Modify output argument: change logs and working directory to be subdirectory for the run
cip_job.set_log_file(opts.working_directory+"/logs/cip-subdag-{}-$(cluster)-$(process).log".format(opts.iteration_number))
cip_job.set_stderr_file(opts.working_directory+"/logs/cip-subdag-{}-$(cluster)-$(process).err".format(opts.iteration_number))
cip_job.set_stdout_file(opts.working_directory+"/logs/cip-subdag-{}-$(cluster)-$(process).out".format(opts.iteration_number))
cip_job.add_condor_cmd('request_disk',opts.general_request_disk)
cip_job.write_sub_file()


parent_node = None

for indx in np.arange(n_workers_proposed):
    cip_node = pipeline.CondorDAGNode(cip_job)
    cip_node.set_retry(opts.general_retries)
    dag.add_node(cip_node)

dag_name="convergence_cip_{}_subdag".format(opts.iteration_number)
dag.set_dag_file(dag_name)
dag.write_concrete_dag()
