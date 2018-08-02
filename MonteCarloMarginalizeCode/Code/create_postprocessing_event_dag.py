#! /usr/bin/env python
#
#  create_postprocessing_event_dag.py 
#
#  DAG for postprocessing jobs
#  Uses
#     - single-node job, to postprocess existing .composite file -> fit
#     - multi-node job, to perform many posterior generation calculations from a fit
#     - EOS ranking calculations
#     - EOS likelihood on grid generation (e.g., for posterior)
#
#  INGREDIENT
#     Essentially all nodes in this DAG are util_ConstructIntrinsicPosterior_GenericCoordinates.py
#     Arguments of the DAG mirror arguments of that function
#
# EXAMPLES
#    python create_postprocessing_event_dag.py --cip-args args.txt --eos-params eos_names.txt  --workflow eos_rank
#        # https://git.ligo.org/lscsoft/lalsuite/tree/master/lalsimulation/src  has list of EOS names
#        # Must prefix them with lal_  for code to work

import argparse
import sys
import os
import numpy as np
import lalsimutils
import lalsimulation as lalsim
import lal
import functools
import itertools

from glue import pipeline

# Taken from
# http://pythonadventures.wordpress.com/2011/03/13/equivalent-of-the-which-command-in-python/
def is_exe(fpath):
    return os.path.exists(fpath) and os.access(fpath, os.X_OK)

def which(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file): return exe_file

    return None

def mkdir(dir_name):
    try :
        os.mkdir(dir_name)
    except OSError:
        pass


def write_CIP_sub(tag='integrate', exe=None, log_dir=None, use_eos=False,ncopies=1,arg_str=None,arg_vals=None, **kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    exe = exe or which("util_ConstructIntrinsicPosterior_GenericCoordinates.py")
    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    #
    # Macro based options.
    #     - select EOS from list (done via macro)
    #     - pass spectral parameters
    #
#    ile_job.add_var_opt("event")
    if use_eos:
        ile_job.add_var_opt("using-eos")


    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if kwargs.has_key("fname_output_samples") and kwargs["fname_output_samples"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["fname_output_samples"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
    if kwargs.has_key("fname_output_integral") and kwargs["fname_output_integral"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["fname_output_integral"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))

    #
    # Add normal arguments
    # FIXME: Get valid options from a module
    #
    for opt, param in kwargs.iteritems():
        if isinstance(param, list) or isinstance(param, tuple):
            # NOTE: Hack to get around multiple instances of the same option
            for p in param:
                ile_job.add_arg("--%s %s" % (opt.replace("_", "-"), str(p)))
        elif param is True:
            ile_job.add_opt(opt.replace("_", "-"), None)
        elif param is None or param is False:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    ile_job.add_condor_cmd('getenv', 'True')
    ile_job.add_condor_cmd('request_memory', '2048')

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print " LIGO accounting information not available.  You must add this manually to integrate.sub !"
        
    

    ###
    ### SUGGESTION FROM STUART (for later)
    # request_memory = ifthenelse( (LastHoldReasonCode=!=34 && LastHoldReasonCode=!=26), InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage) )
    # periodic_release = ((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26))
    # This will automatically release a job that is put on hold for using too much memory with a 50% increased memory request each tim.e


    return ile_job, ile_sub_name




parser = argparse.ArgumentParser()
parser.add_argument("--working-directory",default="./")
parser.add_argument("--cip-args",default=None,help="filename of args.txt file  which holds CIP arguments.  Should NOT conflict with arguments auto-set by this DAG ... in particular, i/o arguments will be modified")
parser.add_argument("--eos-params",default=None,help="filename of eos_params.dat, which is either a list of LAL EOS names *or* a list of parameters. Header identifies types")
parser.add_argument("--workflow",default='single',help="[single|fit|posterior|eos_rank|eos_marg] describes workflow layout used.  'Single' is a single node, running the fit and posterior.  'fit' only generates the fit and saves it; posterior only generates the posterior (with multiple jobs); eos_rank uses a saved fit to rank EOS; and eos_marg uses a saved fit for EOS marginalization")
opts=  parser.parse_args()



if opts.cip_args is None:
    print " No arguments provided for low-level job"
    sys.exit(0)

# Load args.txt. Remove first item.  Store
with open(opts.cip_args) as f:
    cip_args_list = f.readlines()
cip_args = ' '.join( map( lambda x: x.replace('\\',''),cip_args_list) )
cip_args = ' '.join(cip_args.split(' ')[1:])
# Some argument protection for later
cip_args = cip_args.replace('[', '\'[')
cip_args = cip_args.replace(']', ']\'')
cip_args=cip_args.rstrip()
print cip_args

###
### DAG generation
###
log_dir="%s/logs/" % opts.working_directory # directory to hold

dag = pipeline.CondorDAG(log=os.getcwd())

mkdir(log_dir) # Make a directory to hold log files of jobs



###
###  Configuration 0: Fit job
###
if opts.workflow == 'single':
    single_job, single_job_name = write_CIP_sub(tag='CIP',log_dir=log_dir,arg_str=cip_args)
    single_job.write_sub_file()

    cip_node = pipeline.CondorDAGNode(single_job)
    cip_node.add_macro("macroevent", 0)
    cip_node.set_category("CIP")
    dag.add_node(cip_node)
elif opts.workflow=='eos_rank' and not (opts.eos_params is None):
    eos_job, eos_job_name = write_CIP_sub(tag='CIP',log_dir=log_dir,arg_str=cip_args,use_eos=True)
    eos_job.write_sub_file()

    # Look up EOS names
    names_eos = list(np.loadtxt(opts.eos_params,dtype=str).flat)
    print names_eos

    for name in names_eos:
        cip_node = pipeline.CondorDAGNode(eos_job)
        cip_node.add_macro("macrousing_eos", name)
        cip_node.set_category("CIP")
        dag.add_node(cip_node)
        

dag_name="marginalize_intrinsic_parameters_postprocessing"
dag.set_dag_file(dag_name)
dag.write_concrete_dag()

