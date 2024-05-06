#! /usr/bin/env python
#
# GOAL: single-button injection setup from ini file
#
#

################################################################
#### Code Description ####
################################################################

# This code serves to create a full RIFT injection workflow, pulling all relevant information (including waveform parameters) from a config.ini file.
# Please see the example ini file (BBH_inject_example.ini)


## Basic Code Process: ##

# The code process is as follows, starting from reading in the ini file: (example: single_inject_RIFT.py -- /home/albert.einstein/path/to/config/BBH_inject_example.ini)
# Create a MDC (mock data challenge) SimInspiral table xml from the waveform parameters
# uses LALWriteFrame.py to create frame files for the specified detectors, stored in the signal_frames dir (also combined_frames if there is no noise)
# if specified with --use-noise, adds noise frames and dumps the combined into the combined_frames dir
# calculates SNR with util_FrameZeroNoiseSNR.py, uses network SNR as the initial guess
# writes the coinc.xml table
# runs util_RIFT_pseudo_pipe.py, which sets up the RIFT analysis workflow. In the resultant analysis directory the user needs to only submit the dag via condor_submit_dag


## Options: ##

# --use-osg will allow you to run on the IGWN pool of distributed computing resources, for which you need to have the environment variables:
# SINGULARITY_RIFT_IMAGE: path to the RIFT Singularity image (container) that will run RIFT
# SINGULARITY_BASE_EXE_DIR=/usr/local/bin/

# --add-extrinsic will enable calculation of the extrinsic posteriors in the final iteration

# --force-cpu will disable the GPU code pathway, this is useful for running smaller or custom jobs

# --use-noise will add noise to the created signal frames

# --bypass frames will skip to the util_RIFT_pseudo_pipe.py step, assuming that all the file structure is otherwise in place.
# this is useful for recreating an analysis off of existing frames without waiting for them to generate.

# --just-frames will exit the program after the SNR report is completed.
# when combined with --bypass frames, allows for two-stage injection creation

# --use-hyperbolic passes further options to util_RIFT_pseudo_pipe.py, enabling analysis of hyperbolic systems


## Other notes: ##

# if you're using a distance marginalization table, it will take a while to create the injection. You'll see:
#  IntegrationWarning: The integral is probably divergent, or slowly convergent.
# but it is working, give it a few minutes



################################################################
#### Preamble ####
################################################################

import numpy as np
import argparse
import os
import sys
sys.path.append(os.getcwd())
import shutil
import ast
import glob
import re
import configparser

from RIFT.misc.dag_utils import mkdir
import RIFT.lalsimutils as lalsimutils
import lal
import lalsimulation as lalsim

from gwpy.timeseries import TimeSeries
from matplotlib import pyplot as plt

# Backward compatibility
from RIFT.misc.dag_utils import which
lalapps_path2cache = which('lal_path2cache')
if lalapps_path2cache == None:
    lalapps_path2cache =  which('lalapps_path2cache')
    
################################################################
#### Functions & Definitions ####
################################################################


def unsafe_config_get(config,args,verbose=False):
    if verbose:
        print(" Retrieving ", args, end=' ') 
        print(" Found ",eval(config.get(*args)))
    return eval( config.get(*args))

def add_frames(channel,input_frame, noise_frame,combined_frame):
    # note - this needs to be updated to work with the current install regardless of user
    exe = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pp/add_frames.py'))
    if not(os.path.dirname(exe) in os.environ['PATH'].split(os.pathsep)):
        print('add_frames.py not found, adding to PATH...')
        os.environ['PATH'] += os.pathsep + os.path.dirname(exe)
    cmd = exe + " " + channel + " " + input_frame + " " + noise_frame + " " + combined_frame 
    print(cmd)
    os.system(cmd)

lsu_MSUN=lal.MSUN_SI
lsu_PC=lal.PC_SI        

################################################################
#### Options Parse ####
################################################################
        
parser = argparse.ArgumentParser()
parser.add_argument("--use-ini", default=None, type=str, required=True, help="REQUIRED. Pass ini file for parsing. Intended to reproduce lalinference_pipe functionality. Overrides most other arguments.")
parser.add_argument("--use-osg",action='store_true',help="Attempt OSG operation. Command-line level option, not at ini level to make portable results")
parser.add_argument("--add-extrinsic",action='store_true',help="Add extrinsic posterior.  Corresponds to --add-extrinsic --add-extrinsic-time-resampling --batch-extrinsic for pipeline")
parser.add_argument("--force-cpu",action='store_true',help="Forces avoidance of GPUs")
parser.add_argument("--use-noise",action='store_true',help="Combine clean signal injection with fiducial noise frames. Note that you must have pycbc installed.")
parser.add_argument("--bypass-frames",action='store_true',help="Skip making mdc and frame files, use with caution")
parser.add_argument("--just-frames",action='store_true',help="Stop after making frame files, use with caution")
parser.add_argument("--use-hyperbolic",action='store_true',help="Adds hyperbolic options, requires TEOBResumS approx. NOTE: development still in progress.")
opts =  parser.parse_args()

config = configparser.ConfigParser(allow_no_value=True) #SafeConfigParser deprecated from py3.2

ini_path = os.path.abspath(opts.use_ini)

# read in config
config.read(ini_path)

bypass_frames=opts.bypass_frames 
just_frames=opts.just_frames
use_noise=opts.use_noise 

dist_marg_arg = config.get('rift-pseudo-pipe','internal-marginalize-distance') 
dist_marg = True if dist_marg_arg.lower() == 'true' else False

################################################################
#### Initialize ####
################################################################

# Create, go to working directory
working_dir = config.get('init','working_directory')
print(" Working dir ", working_dir)
mkdir(working_dir)
os.chdir(working_dir)
working_dir_full = os.getcwd()

# load IFO list from config
ifos = unsafe_config_get(config,['analysis','ifos'])

################################################################
#### Load Injection Parameters from config into lalsimutils ####
################################################################

P_list = []
P = lalsimutils.ChooseWaveformParams()

## Waveform Model ##
approx_str = config.get('engine','approx')
P.approx=lalsim.GetApproximantFromString(approx_str)

## Intrinsic Parameters ##
P.m1 = float(config.get('injection-parameters','m1'))*lsu_MSUN
P.m2 = float(config.get('injection-parameters','m2'))*lsu_MSUN
P.s1x = float(config.get('injection-parameters','s1x'))
P.s1y = float(config.get('injection-parameters','s1y'))
P.s1z = float(config.get('injection-parameters','s1z'))
P.s2x = float(config.get('injection-parameters','s2x'))
P.s2y = float(config.get('injection-parameters','s2y'))
P.s2z = float(config.get('injection-parameters','s2z'))
if opts.use_hyperbolic:
    P.E0 = float(config.get('injection-parameters','E0'))
    P.p_phi0 = float(config.get('injection-parameters','p_phi0'))
## Extrinsic Parameters ##
P.dist = float(config.get('injection-parameters','dist'))*1e6*lsu_PC
fiducial_event_time = float(config.get('injection-parameters','event_time')) # for frame addition
P.tref = fiducial_event_time
P.theta = float(config.get('injection-parameters','dec'))
P.phi = float(config.get('injection-parameters','ra'))
P.incl = float(config.get('injection-parameters','incl'))
P.psi = float(config.get('injection-parameters','psi'))
P.phiref = float(config.get('injection-parameters','phase'))
# Other waveform settings
P.fmin = float(config.get('injection-parameters','fmin'))

P_list.append(P)

# create mdc.xml.gz - contains all injection params
if bypass_frames:
    print("Skipping mdc creation")
    pass
else:
    lalsimutils.ChooseWaveformParams_array_to_xml(P_list,"mdc")

    
    
################################################################    
#### Create frame files and cache file ####
################################################################

if bypass_frames:
    print("Skipping frame creation")
    indx = 0
    pass
else:
    # create clean signal frames #
    print(" === Writing signal files === ")

    mkdir('signal_frames')
    # must be compatible with duration of the noise frames
    t_start = int(fiducial_event_time)-150
    t_stop = int(fiducial_event_time)+150

    indx = 0 # need an event number, meaningless otherwise
    target_subdir = 'signal_frames/event_{}'.format(indx) # this is where the signal frames will go
    mkdir(working_dir_full+"/"+target_subdir)
    os.chdir(working_dir_full+"/"+target_subdir)
    # here we're in the signal_frames/event_0 directory.
    # Loop over instruments, write frame files and cache
    for ifo in ifos:
        cmd = "util_LALWriteFrame.py --inj " + working_dir_full+"/mdc.xml.gz --event {} --start {}  --stop {}  --instrument {} --approx {}".format(indx, t_start,t_stop,ifo, approx_str)
        print(cmd)
        os.system(cmd)
        
    # Here we plot the signal frames    
    gwf_files = [os.path.abspath(file) for file in os.listdir(os.getcwd()) if file.endswith('.gwf')]
    
    for index, path in enumerate(gwf_files):
        file = os.path.basename(path)
        # assign correct channel name
        if file.startswith('H'):
            channel = 'H1:FAKE-STRAIN'
        elif file.startswith('L'):
            channel = 'L1:FAKE-STRAIN'
        elif file.startswith('V'):
            channel = 'V1:FAKE-STRAIN'
            
        data = TimeSeries.read(path, channel)
        strain = np.array(data.value)
        sample_times = np.array(data.times)
        
        # plot domain - don't do this for combined frames
        max_amp = np.argmax(strain)        
        plot_domain = [sample_times[max_amp] - 1.0, sample_times[max_amp] + 0.5]
        
        plt.plot(sample_times, data, label=f'{channel}')
        plt.xlim(plot_domain)
        plt.legend(loc='best')
        plt.savefig(f'{channel}_plot.png', bbox_inches='tight')
        plt.close()
        
    if not use_noise:
        # copy the contents of signal_frames/event_0 to combined_frames/event_0
        os.chdir(working_dir_full)
        mkdir('combined_frames')
        os.chdir(working_dir_full)
        target_subdir='combined_frames/event_{}'.format(indx)
        shutil.copytree(working_dir_full+'/signal_frames/event_{}'.format(indx), target_subdir)
        
    else:

        print(" === Using fiducial noise frames === ")

        noise_frame_dir = config.get("make_data","fiducial_noise_frames")

        print(" === Joining synthetic signals to reference noise === ")

        seglen_actual = t_stop - t_start
        os.chdir(working_dir_full)
        mkdir('combined_frames')
        os.chdir(working_dir_full)
        target_subdir='combined_frames/event_{}'.format(indx)
        mkdir(target_subdir)
        os.chdir(working_dir_full+"/"+target_subdir)
        for ifo in ifos:
            fname_input = working_dir_full+"/signal_frames/event_{}/{}-fake_strain-{}-{}.gwf".format(indx,ifo[0],int(t_start),seglen_actual)
            fname_noise= noise_frame_dir + "/" + ifo + ("/%08d.gwf" % indx)
            fname_output = working_dir_full+"/{}/{}-combined-{}-{}.gwf".format(target_subdir,ifo[0],int(t_start),seglen_actual)
            print(ifo+":FAKE-STRAIN")
            print(fname_input)
            print(fname_noise)
            print(fname_output)
            add_frames(ifo+":FAKE-STRAIN",fname_input, fname_noise,fname_output)
            
        # Here we plot the combined frames    
        gwf_files = [os.path.abspath(file) for file in os.listdir(os.getcwd()) if file.endswith('.gwf')]

        for index, path in enumerate(gwf_files):
            file = os.path.basename(path)
            # assign correct channel name
            if file.startswith('H'):
                channel = 'H1:FAKE-STRAIN'
            elif file.startswith('L'):
                channel = 'L1:FAKE-STRAIN'
            elif file.startswith('V'):
                channel = 'V1:FAKE-STRAIN'
            
            data = TimeSeries.read(path, channel)

            plt.plot(data, label=f'{channel}')
            plt.legend(loc='best')
            plt.savefig(f'{channel}_plot.png', bbox_inches='tight')
            plt.close()
            
    # write frame paths to cache
    os.chdir(working_dir_full)
    target_subdir='combined_frames/event_{}'.format(indx)
    os.chdir(working_dir_full+"/"+target_subdir)
    cmd = "/bin/find .  -name '*.gwf' | {} > signals.cache".format(lalapps_path2cache)
    os.system(cmd)
    
    
if not opts.use_noise:
    # Calculate SNRs for zero-noise frames
    target_subdir='combined_frames/event_{}'.format(indx)
    target_subdir_full = working_dir_full+"/"+target_subdir
    os.chdir(target_subdir_full)
    if os.path.exists(target_subdir_full + '/snr-report.txt'):
        print('SNR report already exists, skipping')
        pass
    else:
        cmd = "util_FrameZeroNoiseSNR.py --cache signals.cache --psd lalsim.SimNoisePSDaLIGOZeroDetHighPower"
        os.system(cmd)
    with open('snr-report.txt', 'r') as file:
        snr_dict = ast.literal_eval(file.read())
    file.close()
    snr_guess = snr_dict['Network'] # passed to pseudo_pipe as --force-hint-snr
else:
    # need a way to estimate SNR for the noise frames, skipping for now
    print('Skipping SNR guess with noise frames')
    pass
    
if opts.just_frames:
    print('Exiting after frame creation!')
    sys.exit(0)

################################################################
#### RIFT Workflow Generation ####
################################################################ 


# check for distance marginalization lookup table - primarily for testing
if bypass_frames:
    os.chdir(working_dir_full)
    if os.path.exists(working_dir_full+'/'+'distance_marginalization_lookup.npz'):
        print("Using existing distance marginalization table")
        dist_marg_exists = True
        dist_marg_path = working_dir_full + "/distance_marginalization_lookup.npz"
    else:
        if dist_marg:
            print("Distance marginalization table not found, it will be made during the workflow")
        dist_marg_exists = False
else:
    dist_marg_exists = False
    

os.chdir(working_dir_full)


# write coinc file
cmd = "util_SimInspiralToCoinc.py --sim-xml mdc.xml.gz --event {}".format(indx)
for ifo in ifos:
    cmd += "  --ifo {} ".format(ifo)
os.system(cmd)

# Designate rundir 
rundir = config.get('init','rundir')
    
#point to signals.cache
target_file = 'combined_frames/event_{}/signals.cache'.format(indx)

# RIFT analysis setup
cmd = 'util_RIFT_pseudo_pipe.py --use-coinc `pwd`/coinc.xml --use-ini {} --use-rundir `pwd`/{} --fake-data-cache `pwd`/{}'.format(ini_path, rundir, target_file)
if opts.add_extrinsic:
    cmd += ' --add-extrinsic --add-extrinsic-time-resampling --batch-extrinsic '
if not opts.use_noise:
    cmd += ' --force-hint-snr {} '.format(snr_guess)
if dist_marg_exists:
    cmd += ' --internal-marginalize-distance-file {} '.format(dist_marg_path)
if opts.force_cpu:
    if opts.use_osg:
        print("Don't force CPUs on the OSG, exiting.")
        sys.exit(1)
    else:
        print('Avoiding GPUs')
        cmd += ' --ile-no-gpu '
if opts.use_osg:
    print('Configuring for OSG operation')
    cmd += ' --use-osg '
    cmd += ' --use-osg-file-transfer '
    cmd += ' --ile-retries 10 '
    # note: --use-osg-cip not used by default - can set this in the ini file
if opts.use_hyperbolic:
    print('Adding hyperbolic options')
    cmd += ' --assume-hyperbolic '

    
os.system(cmd)   
          
################################################################
#### Post-Workflow Cleanup ####
################################################################

# copy over PSDs
psd_dict = unsafe_config_get(config,["make_psd",'fiducial_psds'])
for psd in psd_dict:
    psd_path = psd_dict[psd]
    shutil.copy2(psd_path, rundir)

# copy config file into rundir/reproducibility
repro_path = working_dir_full+"/"+rundir+"/"+"reproducibility"
shutil.copy2(ini_path, repro_path)

if opts.use_osg:
    # make frames_dir and copy frame files for transfer
    os.chdir(working_dir_full+"/"+rundir)
    mkdir('frames_dir')
    os.chdir(working_dir_full+"/combined_frames/event_{}".format(indx))
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.gwf'):
            pathname = os.path.join(os.getcwd(), filename)
            if os.path.isfile(pathname):
                shutil.copy2(pathname, working_dir_full+"/"+rundir+"/frames_dir")
                
    
# for ease of testing, copy the distance marginalization table to the top-level dir
if dist_marg:
    if os.path.exists(working_dir_full+'/'+'distance_marginalization_lookup.npz'):
        pass
    else:
        print('Copying distance marginalization')
        os.chdir(working_dir_full+"/"+rundir)
        shutil.copy2('distance_marginalization_lookup.npz', working_dir_full)
    

    
# change name of dag
os.chdir(working_dir_full+"/"+rundir)
os.rename('marginalize_intrinsic_parameters_BasicIterationWorkflow.dag', working_dir + '.dag')


## Update ILE requirements to avoid bad hosts ##

excluded_hosts = []

host_file = '/home/richard.oshaughnessy/igwn_feedback/rift_avoid_hosts.txt'

with open(host_file, 'r') as file:
    for line in file:
        hostname = line.strip()
        excluded_hosts.append(f"(TARGET.Machine =!= '{hostname}')")

avoid_string = "&&".join(excluded_hosts)

# List of file paths to modify
file_paths = ["ILE.sub","ILE_puff.sub","iteration_4_cip/ILE.sub","iteration_4_cip/ILE_puff.sub"]

if opts.add_extrinsic:
    file_paths.append("ILE_extr.sub")
    
if opts.use_hyperbolic:
    file_paths = ["ILE.sub","ILE_puff.sub","iteration_2_cip/ILE.sub","iteration_2_cip/ILE_puff.sub"]

for file_path in file_paths:
    # Construct the full file path
    full_path = os.path.join(os.getcwd(), file_path)

    # Read the contents of the file
    with open(full_path, "r") as file:
        lines = file.readlines()
        
    # Find the index of the line containing the 'requirements' command
    requirements_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("requirements"):
            requirements_index = i
            break
            
    # Find the index of the line containing the 'queue' command
    queue_index = len(lines) - 1
    for i, line in enumerate(reversed(lines)):
        if line.strip().startswith("queue"):
            queue_index = len(lines) - 1 - i
            break    
    
    # Modify lines
    if requirements_index is not None:
        #requirements_line = lines[requirements_index]
        if opts.use_osg:
            requirements_line = "requirements = (HAS_SINGULARITY=?=TRUE)" + "&&" + avoid_string + "\n"
            lines[requirements_index] = requirements_line
            # Insert the 'require_gpus' line before the 'queue' command
            lines.insert(queue_index, "require_gpus = Capability >= 3.5\n")
        else:
            requirements_line = "requirements = " + avoid_string + "\n"
            lines[requirements_index] = requirements_line
            
    # Write the modified contents back to the file
    with open(full_path, "w") as file:
        file.writelines(lines)

#
#
