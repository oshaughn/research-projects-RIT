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
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import functools
import itertools

from glue import pipeline # https://github.com/lscsoft/lalsuite-archive/blob/5a47239a877032e93b1ca34445640360d6c3c990/glue/glue/pipeline.py

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


def write_CIP_sub(tag='integrate', exe=None, log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=8192,arg_vals=None, **kwargs):
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

    if "fname_output_samples" in kwargs and kwargs["fname_output_samples"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["fname_output_samples"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
    if "fname_output_integral" in kwargs and kwargs["fname_output_integral"] is not None:
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
    for opt, param in list(kwargs.items()):
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
    ile_job.add_condor_cmd('request_memory', str(request_memory)) 
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")
        
    

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
parser.add_argument("--request-memory",default=8192,type=int,help="Memory request for condor (in Mb).")
parser.add_argument("--n-post-jobs",default=1,type=int,help="Number of posterior jobs. Used in posterior and fit+posterior workflows")
parser.add_argument("--workflow",default='single',help="[single|fit|fit+posterior|eos_rank|eos_marg] describes workflow layout used.  'Single' is a single node, running the fit and posterior.  'fit' only generates the fit and saves it; posterior only generates the posterior (with multiple jobs); eos_rank uses a saved fit to rank EOS; and eos_marg uses a saved fit for EOS marginalization")
opts=  parser.parse_args()



if opts.cip_args is None:
    print(" No arguments provided for low-level job")
    sys.exit(0)

# Load args.txt. Remove first item.  Store
with open(opts.cip_args) as f:
    cip_args_list = f.readlines()
cip_args = ' '.join( [x.replace('\\','') for x in cip_args_list] )
cip_args = ' '.join(cip_args.split(' ')[1:])
# Some argument protection for later
cip_args = cip_args.replace('[', ' \'[')
cip_args = cip_args.replace(']', ']\'')
cip_args=cip_args.rstrip()
cip_args += ' --no-plots '
print(cip_args)

###
### Fiducial fit job (=sanity check that code will run)
###
cmdname="%s/command-single_fit.sh" % opts.working_directory
cmd = open(cmdname, 'w')
arg_list = cip_args
exe = which("util_ConstructIntrinsicPosterior_GenericCoordinates.py")
cmd.write('#!/usr/bin/env bash\n')
cmd.write(exe + ' ' + arg_list )
cmd.close()
st = os.stat(cmdname)
import stat
os.chmod(cmdname, st.st_mode | stat.S_IEXEC)


###
### DAG generation
###
log_dir="%s/logs/" % opts.working_directory # directory to hold

dag = pipeline.CondorDAG(log=os.getcwd())

mkdir(log_dir) # Make a directory to hold log files of jobs



###
###  Configuration 0: Fit job
###
if opts.workflow == 'single' or opts.workflow=='fit':
    if opts.workflow=='fit':
        cip_args += ' --fit-save-gp my_fit.pkl'
    single_job, single_job_name = write_CIP_sub(tag='CIP',log_dir=log_dir,arg_str=cip_args,request_memory=opts.request_memory)
    single_job.write_sub_file()

    cip_node = pipeline.CondorDAGNode(single_job)
    cip_node.add_macro("macroevent", 0)
    cip_node.set_category("CIP")
    dag.add_node(cip_node)
if opts.workflow == 'posterior' or opts.workflow=='fit+posterior':
    if opts.workflow=='fit+posterior':
        cip_args_fit = cip_args + ' --fit-save-gp my_fit.pkl'
        cip_args_fit += ' --fname-output-integral integral_fit'   # insure output filenames unique if multiple runs performed
        cip_args_fit += ' --fname-output-samples integral_fit'   # insure output filenames unique if multiple runs performed

        fit_job, fit_job_name = write_CIP_sub(tag='CIP_fit',log_dir=log_dir,arg_str=cip_args_fit,request_memory=opts.request_memory)
        fit_job.write_sub_file()

    cip_args_load=cip_args
    if opts.workflow == 'fit+posterior':
        cip_args_load =  ' --fit-load-gp my_fit.pkl'  # we are saving the fit
    cip_args_load += ' --fname-output-integral integral_$(macroevent)'   # insure output filenames unique if multiple runs performed
    cip_args_load += ' --fname-output-samples integral_$(macroevent)'   # insure output filenames unique if multiple runs performed
    single_job, single_job_name = write_CIP_sub(tag='CIP_post',log_dir=log_dir,arg_str=cip_args_load,request_memory=opts.request_memory)
    single_job.write_sub_file()

    if opts.workflow =='fit+posterior':
        fit_node = pipeline.CondorDAGNode(fit_job)
        fit_node.add_macro("macroevent", 0)
        fit_node.set_category("CIP")
        dag.add_node(fit_node)

    for event_id in np.arange(opts.n_post_jobs):
        cip_node = pipeline.CondorDAGNode(single_job)
        cip_node.add_macro("macroevent", event_id)
        cip_node.set_category("CIP")
        if opts.workflow == 'fit+posterior':
            cip_node.add_parent(fit_node)
        dag.add_node(cip_node)
        

elif opts.workflow=='eos_rank' and not (opts.eos_params is None):
    cip_args += ' --fname-output-integral integral_$(macrousingeos)'   # insure output filenames unique if multiple runs performed
    cip_args += ' --fname-output-samples integral_$(macrousingeos)'   # insure output filenames unique if multiple runs performed
    eos_job, eos_job_name = write_CIP_sub(tag='CIP',log_dir=log_dir,arg_str=cip_args,use_eos=True,request_memory=opts.request_memory)
    eos_job.write_sub_file()

    cip_args_nomatter = cip_args + " --no-matter1 --no-matter2 "  # redundant, should not be needed
    bbh_job, bbh_job_name = write_CIP_sub(tag='CIP_bbh',log_dir=log_dir,arg_str=cip_args_nomatter,use_eos=False,request_memory=opts.request_memory)
    bbh_job.write_sub_file()

    # Look up EOS names
    names_eos = list(np.loadtxt(opts.eos_params,dtype=str).flat)
    print(names_eos)

    for name in names_eos:
        if name == 'lal_BBH':
            # Do a BBH run, without an EOS parameterization added. Lambdas will default to zero.
            cip_node = pipeline.CondorDAGNode(bbh_job)
            cip_node.add_macro("macrousing_eos", "SLY4")   # not used
            cip_node.set_category("CIP_BH")
            dag.add_node(cip_node)
            continue
        cip_node = pipeline.CondorDAGNode(eos_job)
        cip_node.add_macro("macrousing_eos", name)
        cip_node.set_category("CIP")
        dag.add_node(cip_node)

elif opts.workflow=='eos_rank_param' and not (opts.eos_params is None):
    params_eos = np.genfromtxt(opts.eos_params,names=True)
    param_names = params_eos.dtype.names

    cmdname="%s/command-single_fit.sh" % opts.working_directory
    cmd = open(cmdname, 'w')
    arg_list = cip_args
    exe = which("util_ConstructIntrinsicPosterior_GenericCoordinates.py")
    cmd.write('#!/usr/bin/env bash\n')
    cmd.write(exe + ' ' + arg_list + " --using-eos spec --eos-param spectral --eos-param-values [[0,0,0],[1,1,0,0]] ")  # just to have something to parse
    cmd.close()

    cip_args += ' --fname-output-integral integral_$(macroindex)-$(cluster)-$(process)'   # insure output filenames unique if multiple runs performed
    cip_args += ' --fname-output-samples integral_$(macroindex)-$(cluster)-$(process)'   # insure output filenames unique if multiple runs performed
    cip_args += ' --eos-param spectral'
    if 'epsilon0' in param_names:
        # the first few arguments are not used in the LAL case anyways
        cip_args += ' --eos-param-values [[$(macrop0),$(macroeps0),$(macroxmax)],[$(macrogamma1),$(macrogamma2),$(macrogamma3),$(macrogamma4)]] '
    else:
        # Thee are not used in the LAL case anyways
        cip_args += ' --eos-param-values [[0,0,0],[$(macrogamma1),$(macrogamma2),$(macrogamma3),$(macrogamma4)]] '
    eos_job, eos_job_name = write_CIP_sub(tag='CIP',log_dir=log_dir,arg_str=cip_args,use_eos=True,request_memory=opts.request_memory)
    eos_job.write_sub_file()


    # Look up EOS names
    # DEFAULT: spectral parameterization (for now)
    for indx in np.arange(len(params_eos[param_names[0]])):
        cip_node = pipeline.CondorDAGNode(eos_job)
        cip_node.add_macro("macroindex", indx)
        cip_node.add_macro("macrousingeos", 'spec')  # not used in this format, but we need to pass something
        if 'epsilon0' in param_names:
            cip_node.add_macro("macroepsilon0", params_eos["epsilon0"][indx])
            cip_node.add_macro("macrop0", params_eos["p0"][indx])
            cip_node.add_macro("macroxmax", params_eos["xmax"][indx])
        cip_node.add_macro("macrogamma1", params_eos["gamma1"][indx])
        cip_node.add_macro("macrogamma2", params_eos["gamma2"][indx])
        cip_node.add_macro("macrogamma3", params_eos["gamma3"][indx])
        cip_node.add_macro("macrogamma4", params_eos["gamma4"][indx])
        cip_node.set_category("CIP_spectral")
        dag.add_node(cip_node)
        
        

dag_name="marginalize_intrinsic_parameters_postprocessing"
dag.set_dag_file(dag_name)
dag.write_concrete_dag()

