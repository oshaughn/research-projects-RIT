# Copyright (C) 2013  Evan Ochsner
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
A collection of routines to manage Condor workflows (DAGs).
"""

import os, sys
import numpy as np
from time import time
from hashlib import md5

from glue import pipeline

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>"

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


def generate_job_id():
    """
    Generate a unique md5 hash for use as a job ID.
    Borrowed and modified from the LAL code in glue/glue/pipeline.py
    """
    t = str( int( time() * 1000 ) )
    r = str( int( np.random.random() * 100000000000000000 ) )
    return md5(t + r).hexdigest()


# From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py


def write_integrate_likelihood_extrinsic_grid_sub(tag='integrate', exe=None, log_dir=None, ncopies=1, **kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over
    extrinsic parameters.
    Like the other case (below), but modified to use the sim_xml
    and loop over 'event'

    Inputs:
        - 'tag' is a string to specify the base name of output files. The output
          submit file will be named tag.sub, and the jobs will write their
          output to tag-ID.out, tag-ID.err, tag.log, where 'ID' is a unique
          identifier for each instance of a job run from the sub file.
        - 'cache' is the path to a cache file which gives the location of the
          data to be analyzed.
        - 'sim' is the path to the XML file with the grid
        - 'channelH1/L1/V1' is the channel name to be read for each of the
          H1, L1 and V1 detectors.
        - 'psdH1/L1/V1' is the path to an XML file specifying the PSD of
          each of the H1, L1, V1 detectors.
        - 'ncopies' is the number of runs with identical input parameters to
          submit per condor 'cluster'

    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    assert len(kwargs["psd_file"]) == len(kwargs["channel_name"])

    exe = exe or which("integrate_likelihood_extrinsic")
    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]

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

    #
    # Macro based options
    #
    ile_job.add_var_opt("event")

    ile_job.add_condor_cmd('getenv', 'True')
    ile_job.add_condor_cmd('request_memory', '2048')

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


# FIXME: Keep in sync with arguments of integrate_likelihood_extrinsic
def write_integrate_likelihood_extrinsic_sub(tag='integrate', exe=None, log_dir=None, ncopies=1, **kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over
    extrinsic parameters.

    Inputs:
        - 'tag' is a string to specify the base name of output files. The output
          submit file will be named tag.sub, and the jobs will write their
          output to tag-ID.out, tag-ID.err, tag.log, where 'ID' is a unique
          identifier for each instance of a job run from the sub file.
        - 'cache' is the path to a cache file which gives the location of the
          data to be analyzed.
        - 'coinc' is the path to a coincident XML file, from which masses and
          times will be drawn FIXME: remove this once it's no longer needed.
        - 'channelH1/L1/V1' is the channel name to be read for each of the
          H1, L1 and V1 detectors.
        - 'psdH1/L1/V1' is the path to an XML file specifying the PSD of
          each of the H1, L1, V1 detectors.
        - 'ncopies' is the number of runs with identical input parameters to
          submit per condor 'cluster'

    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    assert len(kwargs["psd_file"]) == len(kwargs["channel_name"])

    exe = exe or which("integrate_likelihood_extrinsic")
    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]

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
        elif param is None:
            continue
        else:
            ile_job.add_opt(opt.replace("_", "-"), str(param))

    #
    # Macro based options
    #
    ile_job.add_var_opt("mass1")
    ile_job.add_var_opt("mass2")

    ile_job.add_condor_cmd('getenv', 'True')
    ile_job.add_condor_cmd('request_memory', '2048')
    
    return ile_job, ile_sub_name

def write_result_coalescence_sub(tag='coalesce', exe=None, log_dir=None, output_dir="./", use_default_cache=True):
    """
    Write a submit file for launching jobs to coalesce ILE output
    """

    exe = exe or which("ligolw_sqlite")
    sql_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    sql_sub_name = tag + '.sub'
    sql_job.set_sub_file(sql_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    sql_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    sql_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    sql_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if use_default_cache:
        sql_job.add_opt("input-cache", "ILE_$(macromassid).cache")
    else:
        sql_job.add_arg("$(macrofiles)")
    #sql_job.add_arg("*$(macromassid)*.xml.gz")
    sql_job.add_opt("database", "ILE_$(macromassid).sqlite")
    #if os.environ.has_key("TMPDIR"):
        #tmpdir = os.environ["TMPDIR"]
    #else:
        #print >>sys.stderr, "WARNING, TMPDIR environment variable not set. Will default to /tmp/, but this could be dangerous."
        #tmpdir = "/tmp/"
    tmpdir = "/dev/shm/"
    sql_job.add_opt("tmp-space", tmpdir)
    sql_job.add_opt("verbose", None)

    sql_job.add_condor_cmd('getenv', 'True')
    sql_job.add_condor_cmd('request_memory', '1024')
    
    return sql_job, sql_sub_name

def write_posterior_plot_sub(tag='plot_post', exe=None, log_dir=None, output_dir="./"):
    """
    Write a submit file for launching jobs to coalesce ILE output
    """

    exe = exe or which("plot_like_contours")
    plot_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    plot_sub_name = tag + '.sub'
    plot_job.set_sub_file(plot_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    plot_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    plot_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    plot_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    plot_job.add_opt("show-points", None)
    plot_job.add_opt("dimension1", "mchirp")
    plot_job.add_opt("dimension2", "eta")
    plot_job.add_opt("input-cache", "ILE_all.cache")
    plot_job.add_opt("log-evidence", None)

    plot_job.add_condor_cmd('getenv', 'True')
    plot_job.add_condor_cmd('request_memory', '1024')
    
    return plot_job, plot_sub_name

def write_tri_plot_sub(tag='plot_tri', injection_file=None, exe=None, log_dir=None, output_dir="./"):
    """
    Write a submit file for launching jobs to coalesce ILE output
    """

    exe = exe or which("make_triplot")
    plot_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    plot_sub_name = tag + '.sub'
    plot_job.set_sub_file(plot_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    plot_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    plot_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    plot_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    plot_job.add_opt("output", "ILE_triplot_$(macromassid).png")
    if injection_file is not None:
        plot_job.add_opt("injection", injection_file)
    plot_job.add_arg("ILE_$(macromassid).sqlite")

    plot_job.add_condor_cmd('getenv', 'True')
    #plot_job.add_condor_cmd('request_memory', '2048')
    
    return plot_job, plot_sub_name

def write_1dpos_plot_sub(tag='1d_post_plot', exe=None, log_dir=None, output_dir="./"):
    """
    Write a submit file for launching jobs to coalesce ILE output
    """

    exe = exe or which("postprocess_1d_cumulative")
    plot_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    plot_sub_name = tag + '.sub'
    plot_job.set_sub_file(plot_sub_name)

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    plot_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    plot_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    plot_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    plot_job.add_opt("save-sampler-file", "ILE_$(macromassid).sqlite")
    plot_job.add_opt("disable-triplot", None)
    plot_job.add_opt("disable-1d-density", None)

    plot_job.add_condor_cmd('getenv', 'True')
    plot_job.add_condor_cmd('request_memory', '2048')
    
    return plot_job, plot_sub_name



def write_CIP_sub(tag='integrate', exe=None, input_net='all.net',output='output-ILE-samples',universe="vanilla",out_dir=None,log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=8192,arg_vals=None, no_grid=False,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    exe = exe or which("util_ConstructIntrinsicPosterior_GenericCoordinates.py")
    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies


    # no grid
    if no_grid:
        ile_job.add_condor_cmd("+DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("+flock_local",'true')

    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    ile_job.add_opt("fname", input_net)
    ile_job.add_opt("fname-output-samples", out_dir+"/"+output)
    ile_job.add_opt("fname-output-integral", out_dir+"/"+output)

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


    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    ile_job.add_condor_cmd("stream_error",'True')
    ile_job.add_condor_cmd("stream_output",'True')

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


def write_puff_sub(tag='puffball', exe=None, input_net='output-ILE-samples',output='puffball',universe="vanilla",out_dir=None,log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=1024,arg_vals=None, no_grid=False,**kwargs):
    """
    Perform puffball calculation 
    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    exe = exe or which("util_ParameterPuffball.py")
    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("+DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("+flock_local",'true')

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    ile_job.add_opt("inj-file", input_net)
    ile_job.add_opt("inj-file-out", output)


    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

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

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")
        
    
    return ile_job, ile_sub_name


def write_ILE_sub_simple(tag='integrate', exe=None, log_dir=None, use_eos=False,simple_unique=False,ncopies=1,arg_str=None,request_memory=4096,request_gpu=False,request_disk=False,arg_vals=None, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,use_simple_osg_requirements=False,singularity_image=None,use_cvmfs_frames=False,frames_dir=None,cache_file=None,fragile_hold=False,max_runtime_minutes=None,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """
    if use_singularity and (singularity_image == None)  :
        print(" FAIL : Need to specify singularity_image to use singularity ")
        sys.exit(0)
    if use_singularity and (frames_dir == None)  and (cache_file == None) :
        print(" FAIL : Need to specify frames_dir or cache_file to use singularity (at present) ")
        sys.exit(0)
    if use_singularity and (transfer_files == None)  :
        print(" FAIL : Need to specify transfer_files to use singularity at present!  (we will append the prescript; you should transfer any PSDs as well as the grid file ")
        sys.exit(0)

    exe = exe or which("integrate_likelihood_extrinsic")
    frames_local = None
    if use_singularity:
        path_split = exe.split("/")
        print((" Executable: name breakdown ", path_split, " from ", exe))
        singularity_base_exe_path = "/opt/lscsoft/rift/MonteCarloMarginalizeCode/Code/"  # should not hardcode this ...!
        if 'SINGULARITY_BASE_EXE_DIR' in list(os.environ.keys()) :
            singularity_base_exe_path = os.environ['SINGULARITY_BASE_EXE_DIR']
        else:
#            singularity_base_exe_path = "/opt/lscsoft/rift/MonteCarloMarginalizeCode/Code/"  # should not hardcode this ...!
            singularity_base_exe_path = "/usr/bin/"  # should not hardcode this ...!
        exe=singularity_base_exe_path + path_split[-1]
        if not(frames_dir is None):
            frames_local = frames_dir.split("/")[-1]
    elif use_osg:  # NOT using singularity!
        if not(frames_dir is None):
            frames_local = frames_dir.split("/")[-1]
        path_split = exe.split("/")
        exe=path_split[-1]  # pull out basename
        exe_here = 'my_wrapper.sh'
        if transfer_files is None:
            transfer_files = []
        transfer_files += ['../my_wrapper.sh']
        with open(exe_here,'w') as f:
            f.write("#! /bin/bash  \n")
            f.write(r"""
#!/bin/bash
# Modules and scripts run directly from repository
# Note the repo and branch are self-referential ! Not a robust solution long-term
# Exit on failure:
# set -e
export INSTALL_DIR=research-projects-RIT
export ILE_DIR=${INSTALL_DIR}/MonteCarloMarginalizeCode/Code
export PATH=${PATH}:${ILE_DIR}
export PYTHONPATH=${PYTHONPATH}:${ILE_DIR}
export GW_SURROGATE=gwsurrogate
git clone https://git.ligo.org/richard-oshaughnessy/research-projects-RIT.git
pushd ${INSTALL_DIR} 
git checkout temp-RIT-Tides-port_master-GPUIntegration 
popd

ls 
cat local.cache
echo Starting ...
./research-projects-RIT/MonteCarloMarginalizeCode/Code/""" + exe + " $@ \n")
            os.system("chmod a+x "+exe_here)
            exe = exe_here  # update executable


    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    #
    # Macro based options.
    #     - select EOS from list (done via macro)
    #     - pass spectral parameters
    #
#    ile_job.add_var_opt("event")
    if use_eos:
        ile_job.add_var_opt("using-eos")


    requirements =[]
    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    if simple_unique:
        uniq_str = "$(macroevent)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    # Add lame initial argument

    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


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

    if cache_file:
        ile_job.add_opt("cache-file",cache_file)

    ile_job.add_var_opt("event")

    if not use_osg:
        ile_job.add_condor_cmd('getenv', 'True')
    ile_job.add_condor_cmd('request_memory', str(request_memory)) 
    if not(request_disk is False):
        ile_job.add_condor_cmd('request_disk', str(request_disk)) 
    nGPUs =0
    if request_gpu:
        nGPUs=1
        ile_job.add_condor_cmd('request_GPUs', str(nGPUs)) 
    if use_singularity:
        # Compare to https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
            ile_job.add_condor_cmd('request_CPUs', str(1))
            ile_job.add_condor_cmd('transfer_executable', 'False')
            ile_job.add_condor_cmd("+SingularityBindCVMFS", 'True')
            ile_job.add_condor_cmd("+SingularityImage", '"' + singularity_image + '"')
            requirements = []
            requirements.append("HAS_SINGULARITY=?=TRUE")
#            if not(use_simple_osg_requirements):
#                requirements.append("HAS_CVMFS_LIGO_CONTAINERS=?=TRUE")
            #ile_job.add_condor_cmd("requirements", ' (IS_GLIDEIN=?=True) && (HAS_LIGO_FRAMES=?=True) && (HAS_SINGULARITY=?=TRUE) && (HAS_CVMFS_LIGO_CONTAINERS=?=TRUE)')

    if use_cvmfs_frames:
        requirements.append("HAS_LIGO_FRAMES=?=TRUE")
        ile_job.add_condor_cmd('use_x509userproxy','True')
        if 'X509_USER_PROXY' in list(os.environ.keys()):
            print(" Storing copy of X509 user proxy -- beware expiration! ")
            cwd = os.getcwd()
            fname_proxy = cwd +"/my_proxy"  # this can get overwritten, that's fine - just renews, feature not bug
            os.system("cp ${X509_USER_PROXY} "  + fname_proxy)
#            ile_job.add_condor_cmd('x509userproxy',os.environ['X509_USER_PROXY'])
            ile_job.add_condor_cmd('x509userproxy',fname_proxy)


    if use_osg:
           if not(use_simple_osg_requirements):
               requirements.append("IS_GLIDEIN=?=TRUE")
           # avoid black-holing jobs to specific machines that consistently fail. Uses history attribute for ad
           ile_job.add_condor_cmd('periodic_release','(HoldReasonCode == 45) && (HoldReasonSubCode == 0)')
           ile_job.add_condor_cmd('job_machine_attrs','Machine')
           ile_job.add_condor_cmd('job_machine_attrs_history_length','4')
#           for indx in [1,2,3,4]:
#               requirements.append("TARGET.GLIDEIN_ResourceName=!=MY.MachineAttrGLIDEIN_ResourceName{}".format(indx))
           if "OSG_DESIRED_SITES" in os.environ:
               ile_job.add_condor_cmd('+DESIRED_SITES',os.environ["OSG_DESIRED_SITES"])
           if "OSG_UNDESIRED_SITES" in os.environ:
               ile_job.add_condor_cmd('+UNDESIRED_SITES',os.environ["OSG_UNDESIRED_SITES"])
           # Some options to automate restarts, acts on top of RETRY in dag
           if fragile_hold:
               ile_job.add_condor_cmd("periodic_release","(NumJobStarts < 5) && ((CurrentTime - EnteredCurrentStatus) > 600)")
               ile_job.add_condor_cmd("on_exit_hold","(ExitBySignal == True) || (ExitCode != 0)")
    if use_singularity or use_osg:
            # Set up file transfer options
           ile_job.add_condor_cmd("when_to_transfer_output",'ON_EXIT')

           # Stream log info
           ile_job.add_condor_cmd("stream_error",'True')
           ile_job.add_condor_cmd("stream_output",'True')

    # Create prescript command to set up local.cache, only if frames are needed
    # if we have CVMFS frames, we should be copying local.cache over directly, with it already populated !
    if not(frames_local is None) and not(use_cvmfs_frames):   # should be required for singularity or osg
        try:
            lalapps_path2cache=os.environ['LALAPPS_PATH2CACHE']
        except KeyError:
            print("Variable LALAPPS_PATH2CACHE is unset, assume default lalapps_path2cache is appropriate")
            lalapps_path2cache="lalapps_path2cache"
        cmdname = 'ile_pre.sh'
        if transfer_files is None:
            transfer_files = []
        transfer_files += ["../ile_pre.sh", frames_dir]  # assuming default working directory setup
        with open(cmdname,'w') as f:
            f.write("#! /bin/bash -xe \n")
            f.write( "ls "+frames_local+" | {lalapps_path2cache} 1> local.cache \n".format(lalapps_path2cache=lalapps_path2cache))  # Danger: need user to correctly specify local.cache directory
            # Rewrite cache file to use relative paths, not a file:// operation
            f.write(" cat local.cache | awk '{print $1, $2, $3, $4}' > local_stripped.cache \n")
            f.write("for i in `ls " + frames_local + "`; do echo "+ frames_local + "/$i; done  > base_paths.dat \n")
            f.write("paste local_stripped.cache base_paths.dat > local_relative.cache \n")
            f.write("cp local_relative.cache local.cache \n")
            os.system("chmod a+x ile_pre.sh")
        ile_job.add_condor_cmd('+PreCmd', '"ile_pre.sh"')


#    if use_osg:
#        ile_job.add_condor_cmd("+OpenScienceGrid",'True')
    if use_cvmfs_frames:
        transfer_files += ["../local.cache"]
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    # Write requirements
    # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    if not transfer_files is None:
        if not isinstance(transfer_files, list):
            fname_str=transfer_files
        else:
            fname_str = ','.join(transfer_files)
        fname_str=fname_str.strip()
        ile_job.add_condor_cmd('transfer_input_files', fname_str)
        ile_job.add_condor_cmd('should_transfer_files','YES')

    if not transfer_output_files is None:
        if not isinstance(transfer_output_files, list):
            fname_str=transfer_output_files
        else:
            fname_str = ','.join(transfer_output_files)
        fname_str=fname_str.strip()
        ile_job.add_condor_cmd('transfer_output_files', fname_str)
 
    # Periodic remove: kill jobs running longer than max runtime
    # https://stackoverflow.com/questions/5900400/maximum-run-time-in-condor
    if not(max_runtime_minutes is None):
        remove_str = 'JobStatus =?= 2 && (CurrentTime - JobStartDate) > ( {})'.format(60*max_runtime_minutes)
        ile_job.add_condor_cmd('periodic_remove', remove_str)
    

    ###
    ### SUGGESTION FROM STUART (for later)
    # request_memory = ifthenelse( (LastHoldReasonCode=!=34 && LastHoldReasonCode=!=26), InitialRequestMemory, int(1.5 * NumJobStarts * MemoryUsage) )
    # periodic_release = ((HoldReasonCode =?= 34) || (HoldReasonCode =?= 26))
    # This will automatically release a job that is put on hold for using too much memory with a 50% increased memory request each tim.e


    return ile_job, ile_sub_name



def write_consolidate_sub_simple(tag='consolidate', exe=None, base=None,target=None,universe="vanilla",arg_str=None,log_dir=None, use_eos=False,ncopies=1,no_grid=False, **kwargs):
    """
    Write a submit file for launching a consolidation job
       util_ILEdagPostprocess.sh   # suitable for ILE consolidation.  
       arg_str   # add argument (used for NR postprocessing, to identify group)


    """

    exe = exe or which("util_ILEdagPostprocess.sh")
    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")


    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # Add manual options for input, output
    ile_job.add_arg(base) # what directory to load
    ile_job.add_arg(target) # where to put the output (label), in CWD

    #
    # NO OPTIONS
    #
#    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
#    arg_str = arg_str.lstrip('-')
#    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line


    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))



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
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("+DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("+flock_local",'true')



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



def write_unify_sub_simple(tag='unify', exe=None, base=None,target=None,universe="vanilla",arg_str=None,log_dir=None, use_eos=False,ncopies=1,no_grid=False, **kwargs):
    """
    Write a submit file for launching a consolidation job
       util_ILEdagPostprocess.sh   # suitable for ILE consolidation.  
       arg_str   # add argument (used for NR postprocessing, to identify group)


    """

    exe = exe or which("util_CleanILE.py")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors

    # Write unify.sh
    #    - problem of globbing inside condor commands
    #    - problem that *.composite files from intermediate results will generally NOT be present 
    cmdname ='unify.sh'
    base_str = ''
    if not (base is None):
        base_str = ' ' + base +"/"
    with open(cmdname,'w') as f:        
        f.write("#! /usr/bin/env bash\n")
        f.write( "ls " + base_str+"*.composite  1>&2 \n")  # write filenames being concatenated to stderr
        f.write( exe +  base_str+ "*.composite \n")
    st = os.stat(cmdname)
    import stat
    os.chmod(cmdname, st.st_mode | stat.S_IEXEC)


    ile_job = pipeline.CondorDAGJob(universe=universe, executable=base_str+cmdname) # force full prefix
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # Add manual options for input, output
#    ile_job.add_arg('*.composite') # what to do

    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(target)

    ile_job.add_condor_cmd('getenv', 'True')
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("+DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("+flock_local",'true')

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name

def write_convert_sub(tag='convert', exe=None, file_input=None,file_output=None,universe="vanilla",arg_str='',log_dir=None, use_eos=False,ncopies=1, no_grid=False,**kwargs):
    """
    Write a submit file for launching a 'convert' job
       convert_output_format_ile2inference

    """

    exe = exe or which("convert_output_format_ile2inference")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors

    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    if not(arg_str is None or len(arg_str)<2):
        arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
        arg_str = arg_str.lstrip('-')
        ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#        ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
    ile_job.add_arg(file_input)
    
    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(file_output)

    ile_job.add_condor_cmd('getenv', 'True')
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("+DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("+flock_local",'true')

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name


def write_test_sub(tag='converge', exe=None,samples_files=None, base=None,target=None,universe="target",arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a convergence test job

    """

    exe = exe or which("convergence_test_samples.py") 

    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    # Add options for two parameter files
    for name in samples_files:
#        ile_job.add_opt("samples",name)  # do not add in usual fashion, because otherwise the key's value is overwritten
        ile_job.add_opt("samples " + name,'')  

    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(target)

    ile_job.add_condor_cmd('getenv', 'True')
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name



def write_plot_sub(tag='converge', exe=None,samples_files=None, base=None,target=None,arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a final plot.  Note the user can in principle specify several samples (e.g., several iterations, if we want to diagnose them)

    """

    exe = exe or which("plot_posterior_corner.py") 

    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    # Add options for two parameter files
    for name in samples_files:
#        ile_job.add_opt("samples",name)  # do not add in usual fashion, because otherwise the key's value is overwritten
        ile_job.add_opt("posterior-file " + name,'')  

    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(target)

    ile_job.add_condor_cmd('getenv', 'True')
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name




def write_init_sub(tag='gridinit', exe=None,arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a grid initialization job.
    Note this routine MUST create whatever files are needed by the ILE iteration

    """

    exe = exe or which("util_ManualOverlapGrid.py") 

    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
    arg_str = arg_str.lstrip('-')
    ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    ile_job.add_condor_cmd('getenv', 'True')
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 


    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name



def write_psd_sub_BW_monoblock(tag='PSD_BW_mono', exe=None, log_dir=None, ncopies=1,arg_str=None,request_memory=4096,arg_vals=None, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,singularity_image=None,frames_dir=None,cache_file=None,psd_length=4,srate=4096,data_start_time=None,event_time=None,universe='local',no_grid=False,**kwargs):
    """
    Write a submit file for constructing the PSD using BW
    Modern argument syntax for BW
    Note that *all ifo-specific results must be set outside this loop*, to work sensibly, and passed as an argument

    Inputs:
      - channel_dict['H1']  = [channel_name, flow_ifo]
    Outputs:
        - An instance of the CondorDAGJob that was generated for BW
    """
    exe = exe or which("BayesWave")
    if exe is None:
        print(" BayesWave not available, hard fail ")
        sys.exit(0)
    frames_local = None

    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("+DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("+flock_local",'true')



    requirements =[]
    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))


    #
    # Loop over IFOs
    # You should only have one, in the workflow for which this is intended
    # Problem: 
    ile_job.add_arg("$(macroargument0)")


    #
    # Add mandatory options
    ile_job.add_opt('Niter', '1000100')
    ile_job.add_opt('Nchain', '20')
    ile_job.add_opt('Dmax', '200')  # limit number of dimensions in model
    ile_job.add_opt('resume', '')
    ile_job.add_opt('progress', '')
    ile_job.add_opt('checkpoint', '')
    ile_job.add_opt('bayesLine', '')
    ile_job.add_opt('cleanOnly', '')
    ile_job.add_opt('updateGeocenterPSD', '')
    ile_job.add_opt('dataseed', '1234')  # make reproducible

    ile_job.add_opt('trigtime', str(event_time))
    ile_job.add_opt('psdstart', str(event_time-(psd_length-2)))
    ile_job.add_opt('segment-start', str(event_time-(psd_length-2)))
    ile_job.add_opt('seglen', str(psd_length))
    ile_job.add_opt('psdlength', str(psd_length))
    ile_job.add_opt('srate', str(srate))
    ile_job.add_opt('outputDir', 'output_$(ifo)')





    # Add lame initial argument
    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


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

    # Write requirements
    # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name


def write_psd_sub_BW_step1(tag='PSD_BW_post', exe=None, log_dir=None, ncopies=1,arg_str=None,request_memory=4096,arg_vals=None, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,singularity_image=None,frames_dir=None,cache_file=None,channel_dict=None,psd_length=4,srate=4096,data_start_time=None,event_time=None,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
      - channel_dict['H1']  = [channel_name, flow_ifo]
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """
    exe = exe or which("BayesWavePost")
    if exe is None:
        print(" BayesWavePost not available, hard fail ")
        import sys
        sys.exit(0)
    frames_local = None

    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)


    requirements =[]
    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    #
    # Add mandatory options
    ile_job.add_opt('checkpoint', '')
    ile_job.add_opt('bayesLine', '')
    ile_job.add_opt('cleanOnly', '')
    ile_job.add_opt('updateGeocenterPSD', '')
    ile_job.add_opt('Nchain', '20')
    ile_job.add_opt('Niter', '4000000')
    ile_job.add_opt('Nbayesline', '2000')
    ile_job.add_opt('dataseed', '1234')  # make reproducible

    ile_job.add_opt('trigtime', str(event_time))
    ile_job.add_opt('psdstart', str(event_time-(psd_length-2)))
    ile_job.add_opt('segment-start', str(event_time-(psd_length-2)))
    ile_job.add_opt('seglen', str(psd_length))
    ile_job.add_opt('srate', str(srate))



    #
    # Loop over IFOs
    # Not needed, can do one job per PSD
#    ile_job.add_opt("ifo","$(ifo)")
#    ile_job.add_opt("$(ifo)-cache",cache_file)
    for ifo in channel_dict:
        channel_name, channel_flow = channel_dict[ifo]
        ile_job.add_arg("--ifo "+ ifo)  # need to prevent overwriting!
        ile_job.add_opt(ifo+"-channel", ifo+":"+channel_name)
        ile_job.add_opt(ifo+"-cache", cache_file)
        ile_job.add_opt(ifo+"-flow", str(channel_flow))

    # Add lame initial argument
    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


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

    # Write requirements
    # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name


def write_psd_sub_BW_step0(tag='PSD_BW', exe=None, log_dir=None, ncopies=1,arg_str=None,request_memory=4096,arg_vals=None, transfer_files=None,transfer_output_files=None,use_singularity=False,use_osg=False,singularity_image=None,frames_dir=None,cache_file=None,channel_dict=None,psd_length=4,srate=4096,data_start_time=None,event_time=None,**kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
      - channel_dict['H1']  = [channel_name, flow_ifo]
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """
    exe = exe or which("BayesWave")
    if exe is None:
        print(" BayesWave not available, hard fail ")
        sys.exit(0)
    frames_local = None

    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)


    requirements =[]
    #
    # Logging options
    #
    uniq_str = "$(macroevent)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    #
    # Add mandatory options
    ile_job.add_opt('checkpoint', '')
    ile_job.add_opt('bayesLine', '')
    ile_job.add_opt('cleanOnly', '')
    ile_job.add_opt('updateGeocenterPSD', '')
    ile_job.add_opt('Nchain', '20')
    ile_job.add_opt('Niter', '4000000')
    ile_job.add_opt('Nbayesline', '2000')
    ile_job.add_opt('dataseed', '1234')  # make reproducible

    ile_job.add_opt('trigtime', str(event_time))
    ile_job.add_opt('psdstart', str(event_time-(psd_length-2)))
    ile_job.add_opt('segment-start', str(event_time-(psd_length-2)))
    ile_job.add_opt('seglen', str(psd_length))
    ile_job.add_opt('srate', str(srate))



    #
    # Loop over IFOs
    for ifo in channel_dict:
        channel_name, channel_flow = channel_dict[ifo]
        ile_job.add_arg("--ifo " + ifo)
        ile_job.add_opt(ifo+"-channel", ifo+":"+channel_name)
        ile_job.add_opt(ifo+"-cache", cache_file)
        ile_job.add_opt(ifo+"-flow", str(channel_flow))
        ile_job.add_opt(ifo+"-timeslide", str(0.0))


    # Add lame initial argument
    if "output_file" in kwargs and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if "save_samples" in kwargs and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


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

    # Write requirements
    # From https://github.com/lscsoft/lalsuite/blob/master/lalinference/python/lalinference/lalinference_pipe_utils.py
    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name


def write_resample_sub(tag='resample', exe=None, file_input=None,file_output=None,universe="vanilla",arg_str='',log_dir=None, use_eos=False,ncopies=1, no_grid=False,**kwargs):
    """
    Write a submit file for launching a 'resample' job
       util_ResampleILEOutputWithExtrinsic.py

    """

    exe = exe or which("util_ResampleILEOutputWithExtrinsic.py")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors

    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    if not(arg_str is None or len(arg_str)<2):
        arg_str = arg_str.lstrip() # remove leading whitespace and minus signs
        arg_str = arg_str.lstrip('-')
        ile_job.add_opt(arg_str,'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
#        ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
    ile_job.add_opt('fname',file_input)
    ile_job.add_opt('fname-out',file_output)
    
    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file(file_output)

    ile_job.add_condor_cmd('getenv', 'True')
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    ile_job.add_condor_cmd('requirements', '&&'.join('({0})'.format(r) for r in requirements))

    # no grid
    if no_grid:
        ile_job.add_condor_cmd("+DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("+flock_local",'true')


    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name



def write_cat_sub(tag='cat', exe=None, file_prefix=None,file_postfix=None,file_output=None,universe="vanilla",arg_str='',log_dir=None, use_eos=False,ncopies=1, no_grid=False,**kwargs):
    """
    Write a submit file for launching a 'resample' job
       util_ResampleILEOutputWithExtrinsic.py

    """

    exe = exe or which("find")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors
    exe_switch = which("switcheroo")  # tool for patterend search-replace, to fix first line of output file

    cmdname = 'catjob.sh'
    with open(cmdname,'w') as f:
        f.write("#! /bin/bash\n")
        f.write(exe+"  . -name '"+file_prefix+"*"+file_postfix+"' -exec cat {} \; | sort -r | uniq > "+file_output+";\n")
        f.write(exe_switch + " 'm1 ' '# m1 ' "+file_output)  # add standard prefix
        os.system("chmod a+x "+cmdname)

    ile_job = pipeline.CondorDAGJob(universe=universe, executable='catjob.sh')
    requirements=[]
    if universe=='local':
        requirements.append("IS_GLIDEIN=?=undefined")
    # no grid
    if no_grid:
        ile_job.add_condor_cmd("+DESIRED_SITES",'"nogrid"')
        ile_job.add_condor_cmd("+flock_local",'true')


    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)


#    ile_job.add_arg(" . -name '" + file_prefix + "*" +file_postfix+"' -exec cat {} \; ")
    
    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    ile_job.add_condor_cmd('getenv', 'True')
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")


    return ile_job, ile_sub_name



def write_convertpsd_sub(tag='convert_psd', exe=None, ifo=None,file_input=None,target_dir=None,arg_str='',log_dir=None,  universe='local',**kwargs):
    """
    Write script to convert PSD from one format to another.  Needs to be called once per PSD file being used.
    """

    exe = exe or which("convert_psd_ascii2xml")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors
    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    ile_job.add_opt("fname-psd-ascii",file_input)
    ile_job.add_opt("ifo",ifo)
    ile_job.add_arg("--conventional-postfix")
    
    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    if not (target_dir is None):
        # Copy output PSD into place
        ile_job.add_condor_cmd("+PostCmd", '" cp '+ifo+'-psd.xml.gz ' + target_dir +'"')

    ile_job.add_condor_cmd('getenv', 'True')
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name


def write_joingrids_sub(tag='join_grids', exe=None, universe='vanilla', input_pattern=None,target_dir=None,output_base=None,log_dir=None,n_explode=1, gzip="/usr/bin/gzip", old_add=True, **kwargs):
    """
    Write script to convert PSD from one format to another.  Needs to be called once per PSD file being used.
    """
    exe = exe or which("ligolw_add")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors

#     exe_here = "my_join.sh"
#     with open(exe_here,'w') as f:
#             f.write("#! /bin/bash  \n")
#             f.write(r"""
# #!/bin/bash
# # Modules and scripts run directly from repository
# # Note the repo and branch are self-referential ! Not a robust solution long-term
# # Exit on failure:
# # set -e
# {} {}  > {}/{}.xml
# gzip {}.{}.xml""".format(exe,input_pattern,target_dir,output_base,target_dir,output_base) )
#     os.system("chmod a+x "+exe_here)
#     exe = exe_here  # update executable

    ile_job = pipeline.CondorDAGJob(universe=universe, executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    fname_out =target_dir + "/" +output_base + ".xml.gz"
    ile_job.add_arg("--output="+fname_out)

    working_dir = log_dir.replace("/logs", '') # assumption about workflow/naming! Danger!

    #
    # Logging options
    #
    uniq_str = "$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))
#    ile_job.set_stdout_file(fname_out)

#    ile_job.add_condor_cmd("+PostCmd",  ' "' + gzip + ' ' +fname_out + '"')

    explode_str = ""
    for indx in np.arange(n_explode):
        explode_str+= " {}/{}-{}.xml.gz ".format(working_dir,output_base,indx)
    explode_str += " {}/{}.xml.gz ".format(working_dir,output_base)
    ile_job.add_arg(explode_str)
#    ile_job.add_arg("overlap-grid*.xml.gz")  # working in our current directory
    
    if old_add:
        ile_job.add_opt("ilwdchar-compat",'')  # needed?

    ile_job.add_condor_cmd('getenv', 'True')
    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print(" LIGO accounting information not available.  You must add this manually to integrate.sub !")

    return ile_job, ile_sub_name

