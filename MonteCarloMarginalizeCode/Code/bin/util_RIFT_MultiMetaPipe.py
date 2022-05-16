#! /usr/bin/env python
#
#  a lazy user syntax for constructing *multiple* RIFT pseudo pipe commands, based on a core command.  
#  Designed for a single event analysis, with multiple related (possibly concurrent) analyses feeding one another
#
#  Use pipeline.py for now since familiar, should move to htcondor raw
#
#  https://git.ligo.org/lscsoft/glue/-/blob/master/glue/pipeline.py
#   https://htcondor.readthedocs.io/en/latest/apis/python-bindings/tutorials/DAG-Creation-And-Submission.html

# EXAMPLE
#   python util_RIFT_MultiMetaPipe.py --workflow dummy_metafile.meta

# dummy_metafile.meta
# common_args arg1 arg2
# A argA1 argA2
# B argB1 argB2
# C
# D
# parent A child B
# parent B child C
# parent B child D
# flow C child D
# flow D chid C

# we can even have a synchronous_child option, for children that we know are slower, so they can start at the same time and not wait

# this then builds a submit DAG, and runs the job building pipeline for all the specified jobs
#
# note that a dag to build the jobs is SILLY -- it makes more sense to inline them.
# the more important part about assembling the workflow this way is that it allows us to specify RIFT DEPENDENCIES to pass composite and other information between nodes

# UNFINISHED
#   - 'flow' relationships are bi-directional, BUT
#  - right now pseudo_pipe only implements ONE fetch, not a list of them!  Need to update the fetch capability

import argparse
import numpy as np
import os
# try:
#     import htcondor
# except:
#     print(" - no htcondor - ")
#     exit(0)
#from htcondor import dags

from glue import pipeline
import RIFT.misc.dag_utils as dag_utils
#from RIFT.misc.dag_utils import which



# def pseudo_job(arg_line):
#     mysub = htcondor.Submit(executable='util_RIFT_pseudo_pipe.py',arguments=arg_line)
#     return mysub


parser = argparse.ArgumentParser()
parser.add_argument("--workflow",help="input file, pseudo-dag specification. ")
parser.add_argument("--fetch-all-grids",action='store_true',help="always fetch latest grid files from parents.  Note this forces run directory creation")
opts= parser.parse_args()


exe = dag_utils.which('util_RIFT_pseudo_pipe.py')
base_dir = os.getcwd()

#dag = dags.DAG()
dag = pipeline.CondorDAG(log=os.getcwd())


# read lines
common_args=''
my_nodes = {}
with open(opts.workflow,'r') as f:
    for line in f:
#        print(" INPUT : ", line)
        line=line.strip()
        if len(line)==0: 
            continue
        if line[0] == "#":
            continue
        word0 = line.split(None,1)[0]
        rest0 = ' '.join(line.split(None,1)[1:])
#        print(" word || rest = {} || {} ".format(word0,rest0))
        # fork on options
        if word0 == 'common_args':
            print(" Common arguments : ", rest0)
            common_args+= rest0 + " "
            continue
        elif word0 == 'parent':
            print(" Parent/child specification : ", rest0)
            a,relation,b = rest0.split()
            a_node = my_nodes[a][1]
            b_node = my_nodes[b][1]
            b_node.add_parent(a_node)
            # fetch option
            if opts.fetch_all_grids:
                b_job = my_nodes[b][0]
                b_job._CondorJob__arguments += [ " --external-fetch-native-from {}/{} ".format(base_dir,a) ] 
            continue
        elif word0 == 'flow':
            print(" flow parent/child specification : ", rest0)
            a,relation,b = rest0.split()
            # fetch option
            if opts.fetch_all_grids:
                b_job = my_nodes[b][0]
                b_job._CondorJob__arguments += [ " --external-fetch-native-from {}/{} ".format(base_dir,a) ] 


            continue
        # otherwise defining job task
        this_job = pipeline.CondorDAGJob(universe='vanilla',executable=exe)
        other_args = []
        job_dir = base_dir
        if opts.fetch_all_grids:
            other_args = [" --use-rundir {}/{} ".format(base_dir, word0)]  # need fixed run location so we can find jobs later!
            job_dir = "{}/{}".format(base_dir, word0)
        this_job._CondorJob__arguments = [common_args+ rest0 ]+other_args

        # boilerplate
        # note this is not the JOB directory, because we have not made those yet!
        this_job.set_sub_file("{}/workflow-{}.sub".format(base_dir,word0))
        this_job.set_log_file("{}/workflow-{}.log".format(base_dir,word0))
        this_job.set_stderr_file("{}/workflow-{}.err".format(base_dir,word0))
        this_job.set_stdout_file("{}/workflow-{}.out".format(base_dir,word0))

        # standard needed things
        this_job.add_condor_cmd('getenv', 'True')
        this_job.add_condor_cmd("request_memory",'2048')
        this_job.add_condor_cmd('request_disk',"50M")
        try:
            this_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
            this_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
        except:
            print(" No accounting information available")


        this_node = pipeline.CondorDAGNode(this_job)
        my_nodes[word0] = [this_job, this_node]
        dag.add_node(this_node)

for name in my_nodes:
    this_job, this_node = my_nodes[name]
    this_job.write_sub_file()
        

dag.set_dag_file("workflow")
dag.write_concrete_dag()

# much more useful to have a script to run directly -- dag is generally too slow to launch jobs due to condor queuing
dag.write_script()
