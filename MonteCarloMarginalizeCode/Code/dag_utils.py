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

import os
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
    t = str( long( time() * 1000 ) )
    r = str( long( np.random.random() * 100000000000000000L ) )
    return md5(t + r).hexdigest()

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

    if kwargs.has_key("output_file") and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if kwargs.has_key("save_samples") and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]

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
        print " LIGO accounting information not available.  You must add this manually to integrate.sub !"
        
    

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

    if kwargs.has_key("output_file") and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if kwargs.has_key("save_samples") and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]

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



def write_CIP_sub(tag='integrate', exe=None, input_net='all.net',output='output-ILE-samples',out_dir=None,log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=8192,arg_vals=None, **kwargs):
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
    ile_job.add_condor_cmd('request_memory', str(request_memory)) 
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

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


def write_puff_sub(tag='puffball', exe=None, input_net='output-ILE-samples',output='puffball',out_dir=None,log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=1024,arg_vals=None, **kwargs):
    """
    Perform puffball calculation 
    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    exe = exe or which("util_ParameterPuffball.py")
    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)
 
    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    #
    # Add options en mass, by brute force
    #
    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

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
    ile_job.add_condor_cmd('request_memory', str(request_memory)) 
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print " LIGO accounting information not available.  You must add this manually to integrate.sub !"
        
    
    return ile_job, ile_sub_name


def write_ILE_sub_simple(tag='integrate', exe=None, log_dir=None, use_eos=False,ncopies=1,arg_str=None,request_memory=4096,request_gpu=False,arg_vals=None, transfer_files=None, **kwargs):
    """
    Write a submit file for launching jobs to marginalize the likelihood over intrinsic parameters.

    Inputs:
    Outputs:
        - An instance of the CondorDAGJob that was generated for ILE
    """

    exe = exe or which("integrate_likelihood_extrinsic")
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

    if not transfer_files is None:
        fname_str = ','.join(transfer_files)
        fname_str=fname_str.strip()
        ile_job.add_condor_cmd('transfer_input_files', fname_str)

    #
    # Logging options
    #
    uniq_str = "$(macromassid)-$(cluster)-$(process)"
    ile_job.set_log_file("%s%s-%s.log" % (log_dir, tag, uniq_str))
    ile_job.set_stderr_file("%s%s-%s.err" % (log_dir, tag, uniq_str))
    ile_job.set_stdout_file("%s%s-%s.out" % (log_dir, tag, uniq_str))

    # Add lame initial argument

    if kwargs.has_key("output_file") and kwargs["output_file"] is not None:
        #
        # Need to modify the output file so it's unique
        #
        ofname = kwargs["output_file"].split(".")
        ofname, ext = ofname[0], ".".join(ofname[1:])
        ile_job.add_file_opt("output-file", "%s-%s.%s" % (ofname, uniq_str, ext))
        del kwargs["output_file"]
        if kwargs.has_key("save_samples") and kwargs["save_samples"] is True:
            ile_job.add_opt("save-samples", None)
            del kwargs["save_samples"]


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

    ile_job.add_var_opt("event")

    ile_job.add_condor_cmd('getenv', 'True')
    ile_job.add_condor_cmd('request_memory', str(request_memory)) 
    nGPUs =0
    if request_gpu:
        nGPUs=1
    ile_job.add_condor_cmd('request_GPUs', str(nGPUs)) 
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

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



def write_consolidate_sub_simple(tag='consolidate', exe=None, base=None,target=None,arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a consolidation job
       util_ILEdagPostprocess.sh   # suitable for ILE consolidation.  
       arg_str   # add argument (used for NR postprocessing, to identify group)


    """

    exe = exe or which("util_ILEdagPostprocess.sh")
    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)
    # This is a hack since CondorDAGJob hides the queue property
    ile_job._CondorJob__queue = ncopies

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    # Add manual options for input, output
    ile_job.add_arg(base) # what directory to load
    ile_job.add_arg(target) # where to put the output (label), in CWD

    #
    # Add options en mass, by brute force
    #
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
    # To change interactively:
    #   condor_qedit
    # for example: 
    #    for i in `condor_q -hold  | grep oshaughn | awk '{print $1}'`; do condor_qedit $i RequestMemory 30000; done; condor_release -all 

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



def write_unify_sub_simple(tag='unify', exe=None, base=None,target=None,arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
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


    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable="./"+cmdname)

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

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print " LIGO accounting information not available.  You must add this manually to integrate.sub !"

    return ile_job, ile_sub_name

def write_convert_sub(tag='convert', exe=None, file_input=None,file_output=None,arg_str='',log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a 'convert' job
       convert_output_format_ile2inference

    """

    exe = exe or which("convert_output_format_ile2inference")  # like cat, but properly accounts for *independent* duplicates. (Danger if identical). Also strips large errors

    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    if not(arg_str is None or len(arg_str)<2):
        ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line
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

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print " LIGO accounting information not available.  You must add this manually to integrate.sub !"

    return ile_job, ile_sub_name


def write_test_sub(tag='converge', exe=None,samples_files=None, base=None,target=None,arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a convergence test job

    """

    exe = exe or which("convergence_test_samples.py") 

    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

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

    try:
        ile_job.add_condor_cmd('accounting_group',os.environ['LIGO_ACCOUNTING'])
        ile_job.add_condor_cmd('accounting_group_user',os.environ['LIGO_USER_NAME'])
    except:
        print " LIGO accounting information not available.  You must add this manually to integrate.sub !"

    return ile_job, ile_sub_name



def write_plot_sub(tag='converge', exe=None,samples_files=None, base=None,target=None,arg_str=None,log_dir=None, use_eos=False,ncopies=1, **kwargs):
    """
    Write a submit file for launching a final plot.  Note the user can in principle specify several samples (e.g., several iterations, if we want to diagnose them)

    """

    exe = exe or which("plot_posterior_corner.py") 

    ile_job = pipeline.CondorDAGJob(universe="vanilla", executable=exe)

    ile_sub_name = tag + '.sub'
    ile_job.set_sub_file(ile_sub_name)

    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

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
        print " LIGO accounting information not available.  You must add this manually to integrate.sub !"

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

    ile_job.add_opt(arg_str[2:],'')  # because we must be idiotic in how we pass arguments, I strip off the first two elements of the line

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
        print " LIGO accounting information not available.  You must add this manually to integrate.sub !"

    return ile_job, ile_sub_name
