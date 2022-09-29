#! /usr/bin/env python
#
#  GOAL
#   Iterate CIP until convergence
#     * Create many stages of calling a subsag creating job


import argparse
import sys
import os
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
parser.add_argument("--n-eff-goal",default=10000,type=int)
parser.add_argument("--n-eff-per-worker",default=200,type=int,help="n-eff goal per worker CIP ")
parser.add_argument("--cip-exe",default=None,help="filename of CIP or equivalent executable. Will default to `which util_ConstructIntrinsicPosterior_GenericCoordinates` in low-level code")
parser.add_argument("--cip-args",default=None,help="filename of args_cip.txt file  which holds CIP arguments.  Should NOT conflict with arguments auto-set by this DAG ... in particular, i/o arguments will be modified.   We will NOT change --n-max, so user is assumed to have set this intelligently before trying this.  ASSUME INITIAL ITERATION COUNT ALREADY STRIPPED")
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
parser.add_argument("--general-retries",default=1,type=int,help="Number of retry attempts for internal jobs (convert, CIP, ...). (These can fail, albeit more rarely, usually due to filesystem problems)")
parser.add_argument("--general-request-disk",default="4M",type=str,help="Request disk passed to condor. Must be done for all jobs now")
parser.add_argument("--use-full-submit-paths",action='store_true',help="DAG created has full paths to submit files generated. Note this is implemented on a per-file/as-needed basis, mainly to facilitate using this dag as an external subdag")
opts=  parser.parse_args()


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



###
### DAG generation
###

dag = pipeline.CondorDAG(log=os.getcwd())

# Make directories for all iterations
mkdir(opts.working_directory+"/logs")



##
## Command used for assessment job (except for iteration number) will always be the same
cmd_subdag = "  --iteration-number $(macroiteration)  --n-iterations {}".format(opts.n_iterations)
cmd_subdag += " --cip-exe {} --cip-args {} ".format(opts.cip_exe,opts.cip_args)
cmd_subdag += " --n-eff-goal {} --n-eff-per-worker {} ".format(opts.n_eff_goal, opts.n_eff_per_worker)
cmd_subdag += " --cip-explode-jobs-min {} --cip-explode-jobs-max {} ".format(opts.cip_explode_jobs_min,opts.cip_explode_jobs_max)
cmd_subdag += " --general-retries {} --general-request-disk {} --request-memory-CIP  ".format(opts.general_retries,opts.general_request_disk,opts.request_memory_CIP)
if opts.transfer_file_list:
    cmd_subdag += " --transfer-file-list {} ".format(opts.transfer_file_list)
if opts.use_osg:
    cmd_subdag += " --use-osg "
if opts.use_osg_simple_requirements:
    cmd_subdag += " --use-osg-simple-requirements "
if opts.use_singularity:
    cmd_subdag += " --use-singularity "
if opts.condor_local_nonworker:
    cmd_subdag += " --condor-local-nonworker "
if opts.condor_nogrid_nonworker:
    cmd_subdag += " --condor-nogrid-nonworker "
print(" Iteration command arguments:  ",cmd_subdag)

# Write a submit file which creates the subdag needed
#  REUSE the generic 'con' submit from misc.dag_utils
con_job, con_job_name = dag_utils.write_consolidate_sub_simple(tag='assess_propose',exe='assess_propose_CIP_convergence_sequence.py',log_dir=None,arg_str='',base=opts.working_directory, target='',universe=local_worker_universe,no_grid=no_worker_grid)
#print(dir(con_job))
con_job._CondorJob__arguments.remove(opts.working_directory)
con_job.add_arg(cmd_subdag)
# Modify: set 'initialdir' : should run in top-level direcory
#con_job.add_condor_cmd("initialdir",opts.working_directory+"/iteration_$(macroiteration)_con")
# Modify output argument: change logs and working directory to be subdirectory for the run
con_job.set_log_file(opts.working_directory+"//logs/ap-$(macroiteration)-$(cluster)-$(process).log")
con_job.set_stderr_file(opts.working_directory+"/logs/ap-$(macroiteration)-$(cluster)-$(process).err")
con_job.set_stdout_file(opts.working_directory+"//logs/ap-$(macroiteration)-$(cluster)-$(process).out")
con_job.add_condor_cmd('request_disk',opts.general_request_disk)
if opts.use_full_submit_paths:
    fname = opts.working_directory+"/"+con_job.get_sub_file()
    con_job.set_sub_file(fname)
con_job.write_sub_file()




# Write subdag placeholders
parent_fit_node = None


for it in np.arange(0,opts.n_iterations):
    ###
    ### ASSESSMENT JOB
    con_node = pipeline.CondorDAGNode(con_job)
    con_node.add_macro("macroiteration",it)
    con_node.set_retry(opts.general_retries)
    if (parent_fit_node):
        con_node.add_parent(parent_fit_node)

    ###
    ### WORKING FROM ASSESSMENT JOB
    main_subdag =  pipeline.CondorDAGManJob("{}/convergence_cip_{}_subdag.dag".format(opts.working_directory,it))  # assumes subdag is created
    main_analysis_node = main_subdag.create_node()
    main_analysis_node.set_retry(opts.general_retries)
    main_analysis_node.add_parent(con_node)

    parent_fit_node = main_analysis_node
    dag.add_node(con_node)
    dag.add_node(main_analysis_node)
    


dag_name="cip_converge_workflow"
dag.set_dag_file(dag_name)
dag.write_concrete_dag()
