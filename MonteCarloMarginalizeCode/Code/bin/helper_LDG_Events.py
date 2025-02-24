#! /usr/bin/env python
#
# GOAL
#   - builds pipeline arguments needed for real events, ONLY INCLUDING THOSE RELATED TO DATAFIND
#      - finds and/or builds PSDs
#   - provides configparser i/o if needed (e.g., to standardize PSD settings)
#   - does gracedb event lookup, knows about standard channel names / stores history about analysis choices
#   - intended to simplify offline analysis and followup
#
# REFERENCES
#
# EXAMPLES
#   python helper_LDG_Events.py --gracedb-id G299775 --use-online-psd

import os, sys, shutil
import numpy as np
import argparse

import lal
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim

from igwn_ligolw import lsctables, table, utils
from glue.lal import CacheEntry

import configparser as ConfigParser
import gzip

# Backward compatibility
from RIFT.misc.dag_utils import which
lalapps_path2cache = which('lal_path2cache')
if lalapps_path2cache == None:
    lalapps_path2cache =  which('lalapps_path2cache')



def is_int_power_of_2(a_in):
    a =int(a_in)
    if (a & (a-1)):
        return False
    else:
        return True

def unsafe_config_get(config,args,verbose=False):
    if verbose:
        print( " Retrieving ", args) 
        print( " Found ",eval(config.get(*args)))
    return eval( config.get(*args))

def query_available_ifos(ifos_all,types,server,data_start,data_end,datafind_exe='gw_data_find'):
    ifos_out = []
    for ifo in ifos_all:
        cmd = datafind_exe + ' -u file --gaps -o ' + ifo[0] + ' -t ' + types + ' --server ' + server + ' -s ' + str(data_start) + ' -e ' + str(data_end) + " > test_retrieve.dat"
        os.system(cmd)
        lines=np.loadtxt("test_retrieve.dat",dtype=str)
        if len(lines)>0:
            ifos_out.append(ifo)
    return ifos_out


# The following code only works on LIGO data .. is there a better way?
#   V1 : ITF_SCIENCEMODE
#   L1 : LDS-SCIENCE
# Alternative LIGO options: DMT-ANALYSIS_READY:1'
# Alternative LIGO options:
#     H1:HOFT_OK
#     H1:OBSERVATION_INTENT
# Alternative Virgo options
#     V1:GOOD_QUALITY_DATA_CAT1
def query_available_ifos_viadq(ifos_all,data_start,data_end):
    ifos_out = []
    from gwpy.segments import DataQualityFlag
    ifos_out = []
    for ifo in ifos_all:
        segs = None
        try:
            if ifo in ["H1","L1"]:
                segs = DataQualityFlag.query(ifo+":LDS-SCIENCE:1",data_start,data_end)
#	        segs = DataQualityFlag.query(ifo+":OBSERVATION_INTENT:1",data_start,data_end)
            if ifo in ["V1"]:
                segs = DataQualityFlag.query(ifo+":ITF_SCIENCEMODE:1",data_start,data_end)
#                segs = DataQualityFlag.query(ifo+":OBSERVATION_INTENT:1",data_start,data_end)
            # If we reach this point, it hasn't crashed, so
            ifos_out.append(ifo)
        except:
            True
    return ifos_out

def ldg_datafind(ifo_base, types, server, data_start,data_end,datafind_exe='gw_data_find', retrieve=False,machine_with_files="ldas-pcdev2.ligo.caltech.edu"):
    fname_out_raw = ifo_base[0]+"_raw.cache"
    fname_out = ifo_base[0]+"_local.cache"
    print([ifo_base, types, server, data_start, data_end])
    cmd = datafind_exe + ' -u file --gaps -o ' + ifo_base[0] + ' -t ' + types + ' --server ' + server + ' -s ' + str(data_start) + ' -e ' + str(data_end) + " > " +fname_out_raw
    os.system(cmd)

    if not retrieve:
        # If we are not retrieving, we are on a cluster, and we need to load and convert this information
        # see util_CacheFileConvert.py  (here)
        with open(fname_out_raw,'r') as f:
            lines = f.readlines()
            lines = list(map(lambda x: str(CacheEntry.from_T050017(x)), lines))
        # Add carriage return if not present
        for indx in np.arange(len(lines)):
            if lines[indx][-1] != "\n":
                lines[indx]+= "\n"
        with open(fname_out,'w') as f:
            for line in lines:
                f.write(line)

    else:
        print(" Trying to transfer files necessary from ", machine_with_files)
        os.system("mkdir -f frames")
        with open(fname_out_raw,'r') as f:
            lines = f.readlines()
            for line in lines:
                fname_remote = line[16:]  # drop first few characters
                cmd = " gsiscp " + machine_with_files +":"+fname_remote   + " frames/"
                os.system(cmd)
    return True

def ldg_make_cache(retrieve=False):
    if not retrieve:
        os.system("find frames -name '*.gwf' | {} > local.cache".format(lalapps_path2cache))
    else:
        os.system("cat *_local.cache > local.cache")
    return True

def ldg_make_psd(ifo, channel_name,psd_start_time,psd_end_time,srate=4096,use_gwpy=False, force_regenerate=False,working_directory="."):
    psd_fname = ifo + "-psd.xml.gz"
    if (not force_regenerate) and os.path.isfile(working_directory+"/"+psd_fname):
        print(" File exists : ", psd_fname)
        return True
    cmd = "gstlal_reference_psd --verbose --channel-name " + ifo + "=" + channel_name + " --gps-start-time " + str(int(psd_start_time)) + " --gps-end-time " + str(int(psd_end_time)) + " --write-psd " + psd_fname + " --data-source frames --frame-cache local.cache --sample-rate " + str(srate)
    print(cmd)
    os.system(cmd)
    return True


observing_run_time ={}
observing_run_time["O1"] = [1126051217,1137254417] # https://www.gw-openscience.org/O1/
observing_run_time["O2"] = [1164556817,1187733618] # https://www.gw-openscience.org/O2/
observing_run_time["O3"] = [1230000000,1282953618] # end O3
observing_run_time["O4"] = [1282953618,1430000000] # Completely made up boundaries, for now
def get_observing_run(t):
    for run in observing_run_time:
        if  t > observing_run_time[run][0] and t < observing_run_time[run][1]:
            return run
    print(" No run available for time ", t, " in ", observing_run_time)
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--gracedb-id",default=None,type=str)
parser.add_argument("--internal-use-gracedb-bayestar",action='store_true',help="Retrieve BS skymap from gracedb (bayestar.fits), and use it internally in integration with --use-skymap bayestar.fits.")
parser.add_argument("--force-data-lookup",action='store_true',help='Use this flag if you want to use real data.')
parser.add_argument("--force-mc-range",default=None,type=str,help="For PP plots, or other analyses requiring a specific mc range (eg ini file). Enforces initial grid placement inside this region. Passed directly to MOG and CIP.")
parser.add_argument("--limit-mc-range",default=None,type=str,help="For PP plots, or other analyses requiring a specific mc range (eg ini file), bounding the limit *above*.  Allows the code to auto-select its mc range as usual, then takes the intersection with this limit")
parser.add_argument("--scale-mc-range",type=float,default=None,help="If using the auto-selected mc, scale the ms range proposed by a constant factor. Recommend > 1. . ini file assignment will override this.")
parser.add_argument("--force-eta-range",default=None,type=str,help="For PP plots. Enforces initial grid placement inside this region")
parser.add_argument("--allow-subsolar", action='store_true', help="Override limits which otherwise prevent subsolar mass PE")
parser.add_argument("--use-legacy-gracedb",action='store_true')
parser.add_argument("--event-time",type=float,default=None)
parser.add_argument("--sim-xml",default=None)
parser.add_argument("--use-coinc",default=None)
parser.add_argument("--event",type=int,default=None)
parser.add_argument("--check-ifo-availability",action='store_true',help="if true, attempt to use frame availability or DQ information to choose ")
parser.add_argument("--manual-ifo-list",default=None,type=str,help="Overrides IFO list normally retrieve by event ID.  Use with care (e.g., glitch studies) or for events specified with --event-time.")
parser.add_argument("--observing-run",default=None,help="Use the observing run settings to choose defaults for channel names, etc. Not yet implemented using lookup from event time")
parser.add_argument("--calibration-version",default=None,help="Calibration version to be used.")
parser.add_argument("--playground-data",default=None,help="Playground data. Modifies channel names used.")
parser.add_argument("--datafind-server",default=None,help="LIGO_DATAFIND_SERVER (will override environment variable, which is used as default)")
parser.add_argument("--fmin",default=None,type=float,help="Minimum frequency for integration. Used to estimate signal duration")
parser.add_argument("--manual-mc-min",default=None,type=float,help="Manually input minimum in chirp mass")
parser.add_argument("--manual-mc-max",default=None,type=float,help="Manually input maximum in chrip mass")
parser.add_argument("--force-chi-max",default=None,type=float,help="Provde this value to override the value of chi-max provided") 
parser.add_argument("--force-chi-small-max",default=None,type=float,help="Provde this value to override the value of chi-max provided")
parser.add_argument("--force-lambda-max",default=None,type=float,help="Provde this value to override the value of lambda-max provided")
parser.add_argument("--force-lambda-small-max",default=None,type=float,help="Provde this value to override the value of lambda-small-max provided")
parser.add_argument("--fmin-template",default=20,type=float,help="Minimum frequency for template. Used to estimate signal duration. If fmin not specified, also the minimum frequency for integration")
parser.add_argument("--fmax",default=None,type=float,help="fmax. Use this ONLY if you want to override the default settings, which are set based on the PSD used")
parser.add_argument("--data-start-time",default=None,type=float)
parser.add_argument("--data-end-time",default=None,type=float,help="If both data-start-time and data-end-time are provided, this interval will be used.")
parser.add_argument("--data-LI-seglen",default=None,type=float,help="If provided, use a buffer this long, placing the signal 2s after this, and try to use 0.4s tukey windowing on each side, to be consistent with LI.  Note next argument to change windowing")
parser.add_argument("--data-tukey-window-time",default=0.4,type=float,help="The default amount of time (in seconds) during the turn-on phase of the tukey window. Note that for massive signals, the amount of unfiltered data is (seglen - 2s  - window_time), and can be as short as 1.6 s by default")
parser.add_argument("--no-enforce-duration-bound",action='store_true',help="Allow user to perform LI-style behavior and reequest signals longer than the seglen. Argh, by request.")
#parser.add_argument("--enforce-q-min",default=None,type=float,help='float.  If provided ,the grid will go down to this mass ratio. SEGMENT LENGTH WILL BE ADJUSTED')
parser.add_argument("--working-directory",default=".")
parser.add_argument("--datafind-exe",default="gw_data_find")
parser.add_argument("--gracedb-exe",default="gracedb")
parser.add_argument("--fake-data",action='store_true',help="If this argument is present, the channel names are overridden to FAKE_STRAIN")
parser.add_argument("--cache",type=str,default=None,help="If this argument is present, the various routines will use the frame files in this cache. The user is responsible for setting this up")
parser.add_argument("--psd-file", action="append", help="instrument=psd-file, e.g. H1=H1_PSD.xml.gz. Can be given multiple times for different instruments.  Required if using --fake-data option")
parser.add_argument("--psd-assume-common-window", action='store_true', help="Assume window used to generate PSD is the same as the window used to analyze the data - on-source (BW) not off-source")
parser.add_argument("--assume-fiducial-psd-files", action="store_true", help="Will populate the arguments --psd-file IFO=IFO-psd.xml.gz for all IFOs being used, based on data availability.   Intended for user to specify PSD files later, or for DAG to build BW PSDs. ")
parser.add_argument("--use-online-psd",action='store_true',help='Use PSD from gracedb, if available')
parser.add_argument("--assume-matter",action='store_true',help="If present, the code will add options necessary to manage tidal arguments. The proposed fit strategy and initial grid will allow for matter")
parser.add_argument("--assume-matter-conservatively",action='store_true',help="If present, the code will use the full prior range for exploration and sampling. [Without this option, the initial grid is limited to a physically plausible range in lambda-i")
parser.add_argument("--assume-matter-eos",type=str,default=None,help="If present, AND --assume-matter is present, the code will adopt this specific EOS.  CIP will generate tidal parameters according to (exactly) that EOS.  Not recommended -- better to do this by postprocessing")
parser.add_argument("--assume-matter-but-primary-bh",action='store_true',help="If present, the code will add options necessary to manage tidal arguments for the smaller body ONLY. (Usually pointless)")
parser.add_argument("--internal-tabular-eos-file",type=str,default=None,help="Tabular file of EOS to use.  The default prior will be UNIFORM in this table!. NOT YET IMPLEMENTED (initial grids, etc)")
parser.add_argument("--assume-eccentric",action='store_true',help="If present, the code will add options necessary to manage eccentric arguments. The proposed fit strategy and initial grid will allow for eccentricity")
parser.add_argument("--use-meanPerAno",action='store_true',help="The proposed fit strategy and initial grid will allow for meanPerAno")
parser.add_argument("--assume-nospin",action='store_true',help="If present, the code will not add options to manage precessing spins (the default is aligned spin)")
parser.add_argument("--assume-precessing-spin",action='store_true',help="If present, the code will add options to manage precessing spins (the default is aligned spin)")
parser.add_argument("--assume-volumetric-spin",action='store_true',help="If present, the code will assume a volumetric spin prior in its last iterations. If *not* present, the code will adopt a uniform magnitude spin prior in its last iterations. If not present, generally more iterations are taken.")
parser.add_argument("--assume-highq",action='store_true',help="If present, the code will adopt a strategy that drops spin2. Also the precessing strategy will allow perpendicular spin to play a role early on (rather than as a subdominant parameter later)")
parser.add_argument("--assume-well-placed",action='store_true',help="If present, the code will adopt a strategy that assumes the initial grid is very well placed, and will minimize the number of early iterations performed. Not as extrme as --propose-flat-strategy")
parser.add_argument("--propose-ile-convergence-options",action='store_true',help="If present, the code will try to adjust the adaptation options, Nmax, etc based on experience")
parser.add_argument("--internal-propose-ile-convergence-freezeadapt",action='store_true',help="If present, uses the --no-adapt-after-first --no-adapt-distance options (at one point default)")
parser.add_argument("--internal-propose-ile-adapt-log",action='store_true',help="If present, uses the --adapt-log argument. Useful for very loud signals. Note only lnL information is used for adapting, not prior, so samples will be *uniform* in prior range if lnL is low")
parser.add_argument("--internal-ile-rotate-phase", action='store_true')
parser.add_argument("--internal-ile-auto-logarithm-offset",action='store_true',help="Passthrough to ILE")
parser.add_argument("--internal-ile-use-lnL",action='store_true',help="Passthrough to ILE.  Will DISABLE auto-logarithm-offset and manual-logarithm-offset")
parser.add_argument("--internal-cip-use-lnL",action='store_true')
parser.add_argument("--ile-n-eff",default=50,type=int,help="Target n_eff passed to ILE.  Try to keep above 2")
parser.add_argument("--test-convergence",action='store_true',help="If present, the code will terminate if the convergence test  passes. WARNING: if you are using a low-dimensional model the code may terminate during the low-dimensional model!")
parser.add_argument("--internal-test-convergence-threshold",type=float,default=0.02,help="The value of the threshold. 0.02 has been default ")
parser.add_argument("--lowlatency-propose-approximant",action='store_true', help="If present, based on the object masses, propose an approximant. Typically TaylorF2 for mc < 6, and SEOBNRv4_ROM for mc > 6.")
parser.add_argument("--online", action='store_true', help="Use online settings")
parser.add_argument("--propose-initial-grid",action='store_true',help="If present, the code will either write an initial grid file or (optionally) add arguments to the workflow so the grid is created by the workflow.  The proposed grid is designed for ground-based LIGO/Virgo/Kagra-scale instruments")
parser.add_argument("--propose-initial-grid-fisher",action='store_true',help="If present, overrides propose-initial-grid.  Uses the SEMIANALYTIC fisher matrix to propose an initial grid: very fast, well targeted.")
parser.add_argument("--force-grid-stretch-mc-factor",default=None,type=float,help="A factor to multiply the mc grid width by. By default 1, or 1.5 if search active. Use larger values mainly in synthetic data tests with noise, to insure sufficiently wide coverage around the *true* parameter values.")
#parser.add_argument("--propose-initial-grid-includes-search-error",action='store_true',help="Searches have paraemter offsets, but injections have known parameters.  You need a wider grid if you are running from a search grid, since they are usually substantiallyoffset from the maximumlikelihood ")
parser.add_argument("--force-notune-initial-grid",action='store_true',help="Prevent tuning of grid")
parser.add_argument("--force-initial-grid-size",default=None,type=int,help="Force grid size for initial grid (hopefully)")
parser.add_argument("--propose-fit-strategy",action='store_true',help="If present, the code will propose a fit strategy (i.e., cip-args or cip-args-list).  The strategy will take into account the mass scale, presence/absence of matter, and the spin of the component objects.  If --lowlatency-propose-approximant is active, the code will use a strategy suited to low latency (i.e., low cost, compatible with search PSDs, etc)")
parser.add_argument("--propose-flat-strategy",action="store_true",help="If present AND propose-fit-strategy is present, the strategy proposed will have puffball and convergence tests for every iteration, and the same CIP")
parser.add_argument("--propose-converge-last-stage",action="store_true",help="If present, the last pre-extrinsic stage is 'iterate to convergence' form")
parser.add_argument("--force-fit-method",type=str,default=None,help="Force specific fit method")
#parser.add_argument("--internal-fit-strategy-enforces-cut",action='store_true',help="Fit strategy enforces lnL-offset (default 15) after the first batch of iterations. ACTUALLY DEFAULT - SHOULD BE REDUNDANT")
parser.add_argument("--last-iteration-extrinsic",action='store_true',help="Does nothing!  extrinsic implemented with CEP call, user must do this elsewhere")
parser.add_argument("--no-propose-limits",action='store_true',help="If a fit strategy is proposed, the default strategy will propose limits on mc and eta.  This option disables those limits, so the user can specify their own" )
parser.add_argument("--hint-snr",default=None,type=float,help="If provided, use as a hint for the signal SNR when choosing ILE and CIP options (e.g., to avoid overflow or underflow).  Mainly important for synthetic sources with very high SNR")
parser.add_argument("--ile-distance-prior",default=None,help="If present, passed through to the distance prior option.  If dmarg active, passed to dmarg so the correct prior used when building the marginalization table. ")
parser.add_argument("--internal-marginalize-distance",action='store_true',help='Create options to marginalize over distance in the pipeline. Also create any necessary marginalization files at runtime, based on the maximum distance assumed')
parser.add_argument("--internal-marginalize-distance-file",help="Filename for marginalization file.  You MUST make sure the max distance is set correctly")
parser.add_argument("--internal-distance-max",type=float,default=None,help='If present, the code will use this as the upper limit on distance (overriding the distance maximum in the ini file, or any other setting). *required* to use internal-marginalize-distance in most circumstances')
parser.add_argument("--internal-use-amr",action='store_true',help='If present,the code will set up to use AMR.  Currently not much implemented here, and most of the heavy lifting is elsewhere')
parser.add_argument("--internal-use-aligned-phase-coordinates", action='store_true', help="If present, instead of using mc...chi-eff coordinates for aligned spin, will use SM's phase-based coordinates. Requires spin for now")
parser.add_argument("--internal-use-rescaled-transverse-spin-coordinates",action='store_true',help="If present, use coordinates which rescale the unit sphere with special transverse sampling")
parser.add_argument("--use-downscale-early",action='store_true', help="If provided, the first block of iterations are performed with lnL-downscale-factor passed to CIP, such that rho*2/2 * lnL-downscale-factor ~ (15)**2/2, if rho_hint > 15 ")
parser.add_argument("--use-gauss-early",action='store_true',help="If provided, use gaussian resampling in early iterations ('G'). Note this is a different CIP instance than using a quadratic likelihood!")
parser.add_argument("--use-quadratic-early",action='store_true',help="If provided, use a quadratic fit in the early iterations'")
parser.add_argument("--use-gp-early",action='store_true',help="If provided, use a gp fit in the early iterations'")
parser.add_argument("--use-cov-early",action='store_true',help="If provided, use cov fit in the early iterations'")
parser.add_argument("--use-osg",action='store_true',help="If true, use pathnames consistent with OSG")
parser.add_argument("--use-cvmfs-frames",action='store_true',help="If true, require LIGO frames are present (usually via CVMFS). User is responsible for generating cache file compatible with it.  This option insures that the cache file is properly transferred (because you have generated it)")
parser.add_argument("--use-ini",default=None,type=str,help="Attempt to parse LI ini file to set corresponding options. WARNING: MAY OVERRIDE SOME OTHER COMMAND-LINE OPTIONS")
parser.add_argument("--verbose",action='store_true')
opts=  parser.parse_args()

if opts.assume_matter_but_primary_bh:
    opts.assume_matter=True

#internal_dmax = None
internal_dmax = opts.internal_distance_max # default is None

fit_method='gp'
if not(opts.force_fit_method is None):
    fit_method=opts.force_fit_method


fmax = 1700 # default
psd_names = {}
event_dict = {}


if opts.online:
    opts.calibration_version = "C00"  # will define online variants of C00

datafind_exe = opts.datafind_exe
gracedb_exe = opts.gracedb_exe
download_request = " get file "
if opts.use_legacy_gracedb:
    gracedb_exe = "gracedb_legacy"
    download_request = " download "

###
### Hardcoded lookup tables, for production data analysis 
###

data_types = {}
standard_channel_names = {}

# Initialize O2
data_types["O1"] = {}
standard_channel_names["O1"] = {}
# Initialize O2
data_types["O2"] = {}
standard_channel_names["O2"] = {}
# Initialize O3
data_types["O3"] = {}
standard_channel_names["O3"] = {}
# Initialize O4
data_types["O4"] = {}
standard_channel_names["O4"] = {}


typical_bns_range_Mpc = {}
typical_bns_range_Mpc["O1"] = 100 
typical_bns_range_Mpc["O2"] = 100 
typical_bns_range_Mpc["O3"] = 130
typical_bns_range_Mpc["O4"] = 130

## O1 definitions
cal_versions = {"C00", "C01", "C02"}
for cal in cal_versions:
    for ifo in "H1", "L1":
        if cal == "C00":
            standard_channel_names["O1"][(cal,ifo)] = "GDS-CALIB_STRAIN" # _"+cal
            data_types["O1"][(cal,ifo)] = ifo+"_HOFT"
        else:
            standard_channel_names["O1"][(cal,ifo)] = "DCS-CALIB_STRAIN_"+cal 
            data_types["O1"][(cal,ifo)] = ifo+"_HOFT_" + cal

## O2 definitions
cal_versions = {"C00", "C01", "C02"}
for cal in cal_versions:
    for ifo in "H1", "L1":
        data_types["O2"][(cal,ifo)] = ifo+"_HOFT_" + cal
        if cal == "C00":
            standard_channel_names["O2"][(cal,ifo)] = "GDS-CALIB_STRAIN" # _"+cal
        elif cal == "C02":
            standard_channel_names["O2"][(cal,ifo)] = "DCH-CLEAN_STRAIN_C02"
            data_types["O2"][(cal,ifo)] = ifo+"_CLEANED_HOFT_C02"
        else:
            standard_channel_names["O2"][(cal,ifo)] = "DCS-CALIB_STRAIN_"+cal 
#Virgo
data_types["O2"][("C00", "V1")] = "V1Online"
data_types["O2"][("C02", "V1")] = "V1O2Repro2A"
standard_channel_names["O2"][("C02", "V1")] = "Hrec_hoft_V1O2Repro2A_16384Hz"
if opts.verbose:
    print(standard_channel_names["O2"])

# Replay data
if opts.playground_data:
    data_types["O2"][("C00","H1")] = "GDS-GATED_STRAIN_O2Replay"
    data_types["O2"][("C00","L1")] = "GDS-GATED_STRAIN_O2Replay"

## O3 definition (see Gregg Mendell email)
# https://github.com/lpsinger/gwcelery/blob/master/gwcelery/conf/production.py
#  - note that in exceptional circumstances we may want to use gated strain
# [H1|L1]:GDS-CALIB_STRAIN
# [H1|L1]:GDS-CALIB_STRAIN_CLEAN
# [H1|L1]:GDS-GATED_STRAIN
# https://github.com/lpsinger/gwcelery/blob/master/gwcelery/conf/production.py
cal_versions = {"C00","C01", "X01","X02","C01_nonlinear"}
for cal in cal_versions:
    for ifo in "H1", "L1":
        data_types["O3"][(cal,ifo)] = ifo+"_HOFT_" + cal
        if opts.online:
            data_types["O3"][(cal,ifo)] = ifo+"_llhoft"
        if cal == "C00":
            standard_channel_names["O3"][(cal,ifo)] = "GDS-CALIB_STRAIN_CLEAN" 
            standard_channel_names["O3"][(cal,ifo,"BeforeMay1")] = "GDS-CALIB_STRAIN" 
            # Correct channel name is for May 1 onward : need to use *non-clean* before May 1; see Alan W email and https://wiki.ligo.org/Calibration/CalReview_20190502
            if opts.online:
                standard_channel_names["O3"][(cal,ifo)] = "GDS-CALIB_STRAIN" # Do not assume cleaning is available in low latency
        if cal == "X01":  # experimental version of C01 calibration
            standard_channel_names["O3"][(cal,ifo)] = "DCS-CALIB_STRAIN_CLEAN_X01"
        if cal == "X02":
            standard_channel_names["O3"][(cal,ifo)] = "DCS-CALIB_STRAIN_CLEAN_X02" 
        if cal == 'C01':
            standard_channel_names["O3"][(cal,ifo)] = "DCS-CALIB_STRAIN_CLEAN_C01" 
        if cal == 'C01_nonlinear':
            standard_channel_names["O3"][(cal,ifo)] = "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01" 
            data_types["O3"][(cal,ifo)] = ifo+"_HOFT_CLEAN_SUB60HZ_C01"
data_types["O3"][("C00", "V1")] = "V1Online"
data_types["O3"][("X01", "V1")] = "V1Online"
data_types["O3"][("X02", "V1")] = "V1Online"
data_types["O3"][("C01", "V1")] = "V1Online"
data_types["O3"][("C01_nonlinear", "V1")] = "V1Online"
# https://wiki.ligo.org/LSC/JRPComm/ObsRun3#Virgo_AN1
data_types["O3"][("C01", "V1","September")] = "V1O3Repro1A"
data_types["O3"][("C01_nonlinear", "V1","September")] = "V1O3ReproA"
standard_channel_names["O3"][("C01", "V1","September")] = "Hrec_hoft_V1O3ReproA_16384Hz"
standard_channel_names["O3"][("C01_nonlinear", "V1","September")] = "Hrec_hoft_V1O3ReproA_16384Hz"
standard_channel_names["O3"][("C00", "V1")] = "Hrec_hoft_16384Hz"
standard_channel_names["O3"][("X01", "V1")] = "Hrec_hoft_16384Hz"
standard_channel_names["O3"][("X02", "V1")] = "Hrec_hoft_16384Hz"
standard_channel_names["O3"][("C01", "V1")] = "Hrec_hoft_16384Hz"
standard_channel_names["O3"][("C01_nonlinear", "V1")] = "Hrec_hoft_16384Hz"
if opts.online:
    data_types["O3"][("C00", "V1")] = "V1_llhoft"
    standard_channel_names["O3"][("C00", "V1")] = "Hrec_hoft_16384Hz"

if opts.verbose:
    print(standard_channel_names["O3"])


# O4, analysis-ready frames: 
#  https://dcc.ligo.org/DocDB/0186/T2300083/007/O4-Analysis-Ready-Frames-v7.pdf
cal_versions = {"C00","online"}
for cal in cal_versions:
    for ifo in "H1", "L1":
        data_types["O4"][(cal,ifo)] = ifo+"_HOFT_" + cal+"_AR"
        if opts.online:
            data_types["O4"][(cal,ifo)] = ifo+"_llhoft"
            standard_channel_names["O4"][(cal,ifo)] = "GDS-CALIB_STRAIN_CLEAN"
        if cal == "C00":
            standard_channel_names["O4"][(cal,ifo)] = "GDS-CALIB_STRAIN_CLEAN_AR" 
            # Correct channel name is for May 1 onward : need to use *non-clean* before May 1; see Alan W email and https://wiki.ligo.org/Calibration/CalReview_20190502
            if opts.online:
                data_types["O4"][(cal,ifo)] = ifo+"_HOFT_" + cal
                # Unlike O3, cleaning *is* available in low latency
                standard_channel_names["O4"][(cal,ifo)] = "GDS-CALIB_STRAIN_CLEAN" 
data_types["O4"][("C00", "V1")] = "HoftOnline"
# https://wiki.ligo.org/LSC/JRPComm/ObsRun3#Virgo_AN1
standard_channel_names["O4"][("C00", "V1")] = "Hrec_hoft_16384Hz"
if opts.online:
    data_types["O4"][("C00", "V1")] = "V1_llhoft"
    standard_channel_names["O4"][("C00", "V1")] = "Hrec_hoft_16384Hz"

if opts.verbose:
    print(standard_channel_names["O4"])






datafind_server = None
try:
   datafind_server = os.environ['LIGO_DATAFIND_SERVER']
   print(" LIGO_DATAFIND_SERVER ", datafind_server)
except:
  print(" No LIGO_DATAFIND_SERVER ")
  datafind_server = "datafind.ligo.org:443"
if opts.datafind_server:
    datafind_server = opts.datafind_server
if (datafind_server is None) and not (opts.fake_data):
    print(" FAIL: No data !")

###
### Import event and PSD: Manual branch
###

use_gracedb_event = False
if not(opts.gracedb_id is None):
    use_gracedb_event = True
elif opts.sim_xml:  # right now, configured to do synthetic data only...should be able to mix/match
    print("====Loading injection XML:", opts.sim_xml, opts.event, " =======")
    P = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.sim_xml))[opts.event]
    P.radec =False  # do NOT propagate the epoch later
    P.fref = opts.fmin_template
    P.fmin = opts.fmin_template
    event_dict["tref"]=P.tref = opts.event_time
    event_dict["m1"] = P.m1/lal.MSUN_SI
    event_dict["m2"] = P.m2/lal.MSUN_SI
    event_dict["MChirp"] = P.extract_param('mc')/lal.MSUN_SI  # used in strategy downselection
    event_dict["s1z"] = P.s1z
    event_dict["s2z"] = P.s2z
    event_dict["P"] = P
    event_dict["epoch"]  = 0 # no estimate for now
elif opts.use_coinc: # If using a coinc through injections and not a GraceDB event.
    # Same code as used before for gracedb
    coinc_file = opts.use_coinc
    samples = lsctables.SnglInspiralTable.get_table(utils.load_filename(coinc_file,contenthandler=lalsimutils.cthdler))
    event_duration=4  # default
    ifo_list = []
    snr_list = []
    tref_list = []
    for row in samples:
        m1 = row.mass1
        m2 = row.mass2
        ifo_list.append(row.ifo)
        snr_list.append(row.snr)
        tref_list.append(row.end_time + 1e-9*row.end_time_ns)
        try:
            event_duration = row.event_duration # may not exist
        except:
            print(" event_duration field not in XML ")
            event_duration=4 # fallback
    event_dict["m1"] = row.mass1
    event_dict["m2"] = row.mass2
    event_dict["s1z"] = row.spin1z
    event_dict["s2z"] = row.spin2z
    event_dict["IFOs"] = list(set(ifo_list))
    max_snr_idx = snr_list.index(max(snr_list))
    event_dict['SNR'] = snr_list[max_snr_idx]
    event_dict['tref'] = tref_list[max_snr_idx]
    P=lalsimutils.ChooseWaveformParams()
    P.m1 = event_dict["m1"]*lal.MSUN_SI; P.m2=event_dict["m2"]*lal.MSUN_SI; P.s1z = event_dict["s1z"]; P.s2z = event_dict["s2z"]
    P.fmin = opts.fmin_template  #  fmin we will use internally
    P.tref = event_dict["tref"]
    event_dict["P"] = P
    event_dict["epoch"]  = event_duration

# PSDs must be provided by hand, IF this is done by this code!
ifo_list=[]
if not (opts.psd_file is None):
    for inst, psdf in map(lambda c: c.split("="), opts.psd_file):
            psd_names[inst] = psdf
            ifo_list.append(inst)
    if not(opts.manual_ifo_list):
        event_dict["IFOs"] = ifo_list
    else:
        event_dict["IFOs"] = opts.manual_ifo_list.replace('[','').replace(']','').split(',')
if not ("IFOs" in event_dict.keys()):  
    if not(opts.manual_ifo_list is None):
#        import ast
        event_dict["IFOs"] = opts.manual_ifo_list.replace('[','').replace(']','').split(',')
#        event_dict["IFOs"] = ast.literal_eval(opts.manual_ifo_list)
    else:
        event_dict["IFOs"] = []   # Make sure this is populated, for later


###
### Import event and PSD: GraceDB branch
###

if use_gracedb_event:
  cmd_event = gracedb_exe + download_request + opts.gracedb_id + " event.log"
  if not(opts.use_legacy_gracedb):
        cmd_event += " > event.log "
  os.system(cmd_event)
  # Parse gracedb. Note very annoying heterogeneity in event.log files
  with open("event.log",'r') as f:
        lines = f.readlines()
        for  line  in lines:
            line = line.split(':')
            param = line[0]
            if opts.verbose:
                print(" Parsing line ", line)
            if param in ['MChirp', 'MTot', "SNR","Frequency"]: # add a cwb parameter
                event_dict[ line[0]]  = float(line[1])
            elif 'ime' in param: # event time
                event_dict["tref"] = float(line[1])
            elif param == 'IFOs':
                line[1] = line[1].replace(' ','').rstrip()
                ifo_list = line[1].split(",")
                event_dict["IFOs"] = list(set(event_dict["IFOs"]  +ifo_list))
            elif param == "Pipeline":
                event_dict["search_pipeline"] = line[1].lstrip().rstrip()
                print(" event.log: PIPELINE -{}-".format( event_dict['search_pipeline']))
  try:
    # Read in event parameters. Use masses as quick estimate
    coinc_name = 'coinc.xml'
    if not(opts.use_coinc):
        cmd_event = gracedb_exe + download_request + opts.gracedb_id + " coinc.xml"
        if not(opts.use_legacy_gracedb):
            cmd_event += " > coinc.xml "
        os.system(cmd_event)
        cmd_fix_ilwdchar = "ligolw_no_ilwdchar coinc.xml"; os.system(cmd_fix_ilwdchar) # sigh, need to make sure we are compatible
    else:
        coinc_name = opts.use_coinc
    samples = lsctables.SnglInspiralTable.get_table(utils.load_filename(coinc_name,contenthandler=lalsimutils.cthdler))
    event_duration=4  # default
    for row in samples:
        m1 = row.mass1
        m2 = row.mass2
        try:
            event_duration = row.event_duration # may not exist
        except:
            print(" event_duration field not in XML ")
            event_duration=4 # fallback
    event_dict["m1"] = row.mass1
    event_dict["m2"] = row.mass2
    event_dict["s1z"] = row.spin1z
    event_dict["s2z"] = row.spin2z
    P=lalsimutils.ChooseWaveformParams()
    P.m1 = event_dict["m1"]*lal.MSUN_SI; P.m2=event_dict["m2"]*lal.MSUN_SI; P.s1z = event_dict["s1z"]; P.s2z = event_dict["s2z"]
    P.fmin = opts.fmin_template  #  fmin we will use internally
    P.tref = event_dict["tref"]
    event_dict["P"] = P
    event_dict["epoch"]  = event_duration

    # Get PSD
    # and add PSD arguments to psd_names array.  NEEDS TO BE SIMPLIFIED, redundant code
    if opts.use_online_psd:
        fmax = 1000.
        # After O3, switch to using psd embedded in coinc file!
        if P.tref > 1369054567 - 24*60*60*365: # guesstimate of changeover in gracedb
            do_fallback=True
            if 'search_pipeline' in event_dict:
                # pycbc-specific logic
                if event_dict['search_pipeline'] == 'pycbc':
                    do_fallback=False
                    shutil.copyfile(coinc_name,opts.working_directory+'/psd.xml.gz')
#                    for ifo in event_dict["IFOs"]:
#                        fname_psd_now = "{}-psd.xml".format(ifo)
#                        shutil.copyfile(coinc_name, fname_psd_now)
#                        os.system('gzip {}.gz'.format(fname_psd_now))
                        # Same code as below
                    cmd = "helper_OnlinePSDCleanup.py --psd-file psd.xml.gz "
                        # Label PSD file names in argument strings
                    for ifo in event_dict["IFOs"]:
                            if not opts.use_osg:
                                psd_names[ifo] = opts.working_directory+"/" + ifo + "-psd.xml.gz"
                            else:
                                psd_names[ifo] =  ifo + "-psd.xml.gz"
                            cmd += " --ifo " + ifo
                    os.system(cmd)
            # gstlal-style coinc: psd embedded in a way we can retrieve with this command
            if do_fallback:
              for ifo  in event_dict["IFOs"]:
                if event_dict['search_pipeline'] == 'gstlal':
                  cmd_event = "ligolw_print -t {}:array -d ' '  {}  > {}_psd_ascii.dat".format(ifo,coinc_name,ifo)
                  os.system(cmd_event)
                  cmd_event = "convert_psd_ascii2xml  --fname-psd-ascii {}_psd_ascii.dat --conventional-postfix --ifo {}  ".format(ifo,ifo)
                  os.system(cmd_event)
                else: # for spiir,mbta just copy over coinc.xml as PSD
                  fname_psd_now = "{}-psd.xml".format(ifo)
                  shutil.copyfile(coinc_name, fname_psd_now)
                  os.system('gzip {}'.format(fname_psd_now))
                if not opts.use_osg:
                    psd_names[ifo] = opts.working_directory+"/" + ifo + "-psd.xml.gz"
                else:
                    psd_names[ifo] =  ifo + "-psd.xml.gz"
        else:
            fmax = 1000.
            cmd_event = gracedb_exe + download_request + opts.gracedb_id + " psd.xml.gz"
            if not(opts.use_legacy_gracedb):
                cmd_event += " ./psd.xml.gz " # output file must be specified for new framework
            os.system(cmd_event)
            cmd = "helper_OnlinePSDCleanup.py --psd-file psd.xml.gz "
            # Convert PSD to a useful format
            for ifo in event_dict["IFOs"]:
                if not opts.use_osg:
                    psd_names[ifo] = opts.working_directory+"/" + ifo + "-psd.xml.gz"
                else:
                    psd_names[ifo] =  ifo + "-psd.xml.gz"
                cmd += " --ifo " + ifo
            os.system(cmd)
  except:
      print(" ==> probably not a CBC event, attempting to proceed anyways, FAKING central value <=== ")
      P=lalsimutils.ChooseWaveformParams()
      # For CWB triggers, should use event.log file to pull out a central frequency
      P.m1=P.m2= 50*lal.MSUN_SI  # make this up completely, just so code will run, goal is higher mass than this, watch out for mc range
      event_dict["P"]=P
      event_dict["MChirp"] = P.extract_param('mc')/lal.MSUN_SI
      event_dict["epoch"]  = 0 # no estimate for now
      if not "SNR" in event_dict:
          event_dict["SNR"] = 10  # again made up so code will run


  # Get bayestar.fits 
  if opts.internal_use_gracedb_bayestar:
      cmd_event = gracedb_exe + download_request + opts.gracedb_id + " bayestar.fits "
      if not(opts.use_legacy_gracedb):
          cmd_event += " > bayestar.fits "
      os.system(cmd_event)

if not (opts.hint_snr is None) and not ("SNR" in event_dict.keys()):
    event_dict["SNR"] = np.max([opts.hint_snr,6])  # hinting a low SNR isn't helpful

print(" Event analysis ", event_dict)
if (opts.event_time is None) or opts.sim_xml or "P" in event_dict:
    print( " == candidate event parameters (as passed to helper) == ")
    event_dict["P"].print_params()
    if not(opts.event_time is None):
        event_dict["tref"] = opts.event_time
elif not(opts.event_time is None):
    print(  " == Using event time only; please specify a grid! == ")
    event_dict["tref"]  = opts.event_time
    event_dict["epoch"] = 4
    if not("IFOs" in event_dict.keys()):
        event_dict["IFOs"] = [] # empty list, user must provide later
#    if not("P" in event_dict.keys()):
    event_dict["P"] = lalsimutils.ChooseWaveformParams() # default!

    if "MChirp" not in event_dict.keys():
        event_dict["MChirp"] = event_dict["P"].extract_param('mc')/lal.MSUN_SI  # note this is RANDOM
    else:
        event_dict["P"].assign_param('mc', event_dict["MChirp"]*lal.MSUN_SI)
    print( event_dict["MChirp"])

# Use event GPS time to set observing run, if not provided.  Insures automated operation with a trivial settings file does good things.
if (opts.observing_run is None) and not opts.fake_data:
    tref = event_dict["tref"]
    opts.observing_run = get_observing_run(tref)
    if opts.calibration_version is None and (opts.use_ini is None):
        # This should be a dictionary lookup.
        if opts.observing_run == "O2":
            opts.calibration_version = "C02"
        if opts.observing_run == "O1":
            opts.calibration_version = "C02"
        if opts.observing_run == "O3":
            opts.calibration_version = "C00"   # for now! change as needed



###
### General logic 
###

snr_fac = 1.
if "SNR" in event_dict.keys():
    lnLmax_true = event_dict['SNR']**2 / 2.
    lnLoffset_all = 2*lnLmax_true  # very large : should be enough to keep all points
    lnLoffset_early = 0.8*lnLmax_true  # default value early on : should be good enough
    snr_fac = np.max([snr_fac, event_dict["SNR"]/15.])  # scale down regions accordingly
else:
    lnLoffset_all = 1000
    lnLoffset_early = 500  # a fiducial value, good enough for a wide range of SNRs 

if fit_method =='quadratic' or fit_method =='polynomial' or opts.use_quadratic_early or opts.use_cov_early:
    if 'SNR' in event_dict.keys():
        lnLoffset_all = 0.25*2*lnLmax_true  # not that big, only keep some of th epoints
        lnLoffset_early = np.max([0.1*lnLmax_true,10])  # decent enough
        lnL_start = lnLoffset_all
        lnL_end = np.max([0.05*lnLmax_true,10])  # decent enough
    else:
        lnLoffset_early = 50   # for reasonable fits, not great for low ampltiude sources
        lnLoffset_all =50 # for reasonable sources, not great for low-amplitude sources, should provide override

# Estimate signal duration
t_event = event_dict["tref"]
P=event_dict["P"]
#lalsimutils.ChooseWaveformParams()
#P.m1= event_dict['m1']*lal.MSUN_SI;  P.m2= event_dict['m2']*lal.MSUN_SI;
data_start_time = opts.data_start_time
if (event_dict["epoch"]) is None:
   event_dict["epoch"]=0  # protect against insanity that should never happen
if (opts.data_start_time is None) or (opts.data_end_time is None):
    # onlyu calculate waveform duration (for data retrieval) if we are NOT provided a duration
    # important: sometimes we pass P.fmin == 0 (e.g., NRSur), and estimateWaveofmrDuration will FAIL.
    t_duration  = np.max([ event_dict["epoch"], lalsimutils.estimateWaveformDuration(P)])
    t_before = np.max([4,t_duration])*1.1+8+2  # buffer for inverse spectrum truncation
    data_start_time_orig = data_start_time = t_event - int(t_before)
    data_end_time = t_event + int(t_before) # for inverse spectrum truncation. Overkill

    # Estimate data needed for PSD
    #   - need at least 8 times the duration of the signal!
    #   - important to get very accurate PSD estimation for long signals
    t_psd_window_size = np.max([1024, int(8*t_duration)])
    psd_data_start_time = t_event - 32-t_psd_window_size - t_before
    psd_data_end_time = t_event - 32 - t_before
    # set the start time to be the time needed for the PSD, if we are generating a PSD
    if (opts.psd_file is None) and  use_gracedb_event and not opts.use_online_psd:
        data_start_time = psd_data_start_time
else:
    # arguments override any attempt to calculate duration.  Note these time intervals are used for retrieval, so we can add safety!
    data_start_time_orig  = opts.data_start_time
    data_start_time = int(data_start_time_orig -2)
    data_end_time = int(opts.data_end_time +1)

# reset IFO list if needed. Do NOT do with online_psd
#
if opts.check_ifo_availability and not opts.use_online_psd:  # online PSD only available for some IFOs
    # Do not override the original IFO list, build from the list of PSD files or otherwise
    original_list = event_dict["IFOs"] 
    event_dict["IFOs"] = list( set( original_list+query_available_ifos_viadq(["H1","L1","V1"],data_start_time_orig,data_end_time)))
    print( " IFO list to use ", event_dict["IFOs"])

# define channel names
ifos = event_dict["IFOs"]
channel_names = {}
if opts.use_ini is None:
  for ifo in ifos:
    if opts.fake_data:
        channel_names[ifo] = "FAKE-STRAIN"
    else:
        channel_names[ifo] = standard_channel_names[opts.observing_run][(opts.calibration_version,ifo)]
        # Channel names to use before May 1 in O3: need better lookup logic
        if opts.observing_run == "O3" and  event_dict["tref"] < 1240750000 and opts.calibration_version == 'C00':
            if ifo in ['H1', 'L1']:
                channel_names[ifo] = standard_channel_names[opts.observing_run][(opts.calibration_version,ifo,"BeforeMay1")]
        if opts.observing_run == "O3" and ('C01' in opts.calibration_version) and   event_dict["tref"] > 1252540000 and event_dict["tref"]< 1253980000 and ifo =='V1':
            if ifo == 'V1':
                channel_names[ifo] = standard_channel_names[opts.observing_run][(opts.calibration_version, ifo, "September")]

# Parse LI ini
use_ini=False
srate=4096
if not(opts.use_ini is None):
    use_ini=True
    config = ConfigParser.ConfigParser()
    config.read(opts.use_ini)

    # Overwrite general settings
    ifos = unsafe_config_get(config,['analysis','ifos'])
    event_dict["IFOs"] = ifos # overwrite this
    if not(opts.psd_file): # only do the below  if we are not specifying the PSD fies by passing arguments directly (e.g., as used for online use)
        opts.assume_fiducial_psd_files=True # for now.  
        for ifo in ifos:
            if not ifo in psd_names:
            # overwrite PSD names
                if not opts.use_osg:
                    psd_names[ifo] = opts.working_directory+"/" + ifo + "-psd.xml.gz"
                else:
                    psd_names[ifo] =  ifo + "-psd.xml.gz"

    
    # opts.use_osg = config.get('analysis','osg')


    # Overwrite executables ? Not for this script

    # overwrite channel names
    fmin_vals ={}
    fmin_fiducial = -1
    for ifo in ifos:
        raw_channel_name =  unsafe_config_get(config,['data','channels'])[ifo]
        if ':' in raw_channel_name:
            raw_channel_name = raw_channel_name.split(':')[-1]
        channel_names[ifo] =raw_channel_name
        fmin_vals[ifo] = unsafe_config_get(config,['lalinference','flow'])[ifo]
        fmin_fiducial = fmin_vals[ifo]

    # overwrite arguments associated with seglen
    if 'seglen' in dict(config['engine']):
        opts.choose_LI_data_seglen=False
        opts.data_LI_seglen = unsafe_config_get(config,['engine','seglen'])

    # overwrite arguments used with srate/2, OR fmax if provided
    opts.fmax = unsafe_config_get(config,['engine','srate'])/2 -1  # LI default is not to set srate as an independent variable. Occasional danger with maximum frequency limit in PSD
    # overwrite arguments used with fmax, if provided. ASSUME same for all!
    if config.has_option('lalinference', 'fhigh'):
        fhigh_dict = unsafe_config_get(config,['lalinference','fhigh'])
        for name in fhigh_dict:
            opts.fmax = float(fhigh_dict[name])

    srate = int(unsafe_config_get(config,['engine','srate']))  # raise the srate
    if not(is_int_power_of_2(srate)):
        print("srate must be power of 2!")
        sys.exit(0)
    # Print warning if srate < 4096
    if srate < 4096:
        print("""WARNING WARNING WARNING : 
You are requesting srate < 4096 with your ini file.  Use at your own risk - all our comprehensive testing is at higher sampling rates.
""")
    
    opts.fmin = fmin_fiducial # used only to estimate length; overridden later

    # Use ini file arguments, unless override

    # Force safe PSD

# Set up, perform datafind (if not fake data)
made_ifo_cache_files=False
if not (opts.fake_data):
    made_ifo_cache_files =True
    for ifo in ifos:
        # LI-style parsing
        if use_ini:
            data_type_here = unsafe_config_get(config,['datafind','types'])[ifo]
            channel_name_here = unsafe_config_get(config,['data','channels'])[ifo]
        else:
            data_type_here = data_types[opts.observing_run][(opts.calibration_version,ifo)]
            # Special lookup for later in O3
            # https://wiki.ligo.org/LSC/JRPComm/ObsRun3#Virgo_AN1
            if opts.observing_run == "O3" and ('C01' in opts.calibration_version) and   event_dict["tref"] > 1252540000 and event_dict["tref"]< 1253980000 and ifo =='V1':
                data_type_here=data_types["O3"][(opts.calibration_version, ifo,"September")]        
        ldg_datafind(ifo, data_type_here, datafind_server,int(data_start_time), int(data_end_time), datafind_exe=datafind_exe)
        if opts.online:
            import gwpy.timeseries
            print(np.floor(event_dict["tref"]))
            start_time= np.floor(event_dict["tref"]) - opts.data_LI_seglen - 4   #8s before analysis-start (4s prev) 
            end_time= np.floor(event_dict["tref"]) + 6              #4s after analysis-end (2s prev) 
            duration=round(end_time-start_time)
            try:
                data = gwpy.timeseries.TimeSeries.get(channel=channel_name_here,frametype=data_type_here,start=start_time,end=end_time,verbose=True)
                datapath = os.path.join(opts.working_directory,f"{ifo}-{data_type_here}-{start_time}-{duration}.gwf")
                data.write(datapath)
            except:
                print(" ===> FAILED gwpy.timeseries via NDS:  User must supply own frames! <=== ")

if not opts.cache:  # don't make a cache file if we have one!
    real_data = not(opts.gracedb_id is None)
    real_data = real_data or  opts.check_ifo_availability
    real_data = real_data or opts.force_data_lookup
    real_data = real_data or made_ifo_cache_files
    ldg_make_cache(retrieve=real_data) # we are using the ifo_local.cache files, generated in the previous step, almost without fail.
    opts.cache = "local.cache" # standard filename populated
    if opts.online:
        cmd = "/bin/find .  -name '*.gwf' | {} > local.cache".format(lalapps_path2cache)
        os.system(cmd)

# If needed, build PSDs
transfer_files=[]
if (opts.psd_file is None) and (not opts.use_online_psd) and not (opts.assume_fiducial_psd_files):
    print(" PSD construction ")
    for ifo in event_dict["IFOs"]:
        print(" Building PSD  for ", ifo)
        try:
            ldg_make_psd(ifo, channel_names[ifo], psd_data_start_time, psd_data_end_time, working_directory=opts.working_directory)
            if not opts.use_osg:
                psd_names[ifo] = opts.working_directory+"/" + ifo + "-psd.xml.gz"
            else:
                psd_names[ifo] =  ifo + "-psd.xml.gz"
        except:
            print("  ... PSD generation failed! ")
            sys.exit(1)
elif (opts.assume_fiducial_psd_files):
    for ifo in event_dict["IFOs"]:
            if not opts.use_osg:
                psd_names[ifo] = opts.working_directory+"/" + ifo + "-psd.xml.gz"
            else:
                psd_names[ifo] =  ifo + "-psd.xml.gz"
for ifo in psd_names.keys():
    transfer_files.append(opts.working_directory+"/" + ifo + "-psd.xml.gz")



# Estimate mc range, eta range
#   - UPDATE: need to add scaling with SNR too

mc_center = event_dict["MChirp"]
v_PN_param = (np.pi* mc_center*opts.fmin*lalsimutils.MsunInSec)**(1./3.)  # 'v' parameter
v_PN_param_max = 0.2
v_PN_param = np.min([v_PN_param,v_PN_param_max])
# Estimate width. Note this must *also* account for search error (if we are using search triggers), so it is double-counted and super-wide
# Note I have TWO factors to set: the absolute limits on the CIP, and the grid spacing (which needs to be narrower) for PE placement
fac_search_correct=1.
if opts.gracedb_id: #opts.propose_initial_grid_includes_search_error:
    fac_search_correct = 1.5   # if this is too large we can get duration effects / seglen limit problems when mimicking LI
if opts.force_grid_stretch_mc_factor:
    fac_search_correct =  opts.force_grid_stretch_mc_factor
ln_mc_error_pseudo_fisher = 1.5*np.array([1,fac_search_correct])*0.3*(v_PN_param/v_PN_param_max)**(7.)/snr_fac  # this ignores range due to redshift / distance, based on a low-order estimate
print("  Logarithmic mass error interval base ", ln_mc_error_pseudo_fisher)
if ln_mc_error_pseudo_fisher[0] >1:
    ln_mc_errors_pseudo_fisher =np.array([0.8,0.8])   # stabilize
mc_min_tight, mc_min = np.exp( - ln_mc_error_pseudo_fisher)*mc_center  # conservative !  Should depend on mc, use a Fisher formula. Does not scale to BNS
mc_max_tight, mc_max =np.exp( ln_mc_error_pseudo_fisher)*mc_center   # conservative ! 

# eta <->  delta
#   Start out with a grid out to eta = 0.1 ( *tight, passed to the grid code)
#   Do more than this with puffball and other tools
#   Use other tools to set CIP limits
eta_max = 0.24999999
eta_val =P.extract_param('eta')
tune_grid = False
eta_max_tight = eta_max
eta_min_tight  = eta_min = 0.1  # default for now, will fix this later
tmp1,tmp2 = lalsimutils.m1m2(1,eta_min)
delta_max_tight= delta_max =(tmp1-tmp2)/(tmp1+tmp2)  # about 0.8
delta_min_tight = delta_min =1e-4  # Some approximants like SEOBNRv3 can hard fail if m1=m2
if not(opts.force_eta_range is None):
    tmp = opts.force_eta_range 
    eta_range_parsed = list(map(float,tmp.replace('[','').replace(']','').split(',')))
    delta_min_tight = np.sqrt(1 - 4*eta_range_parsed[1]) + 1e-4
    delta_max_tight = np.sqrt(1 - 4*eta_range_parsed[0])
if mc_center < 2.6 and opts.propose_initial_grid and not(opts.allow_subsolar):  
    # BNS scale, need to constraint eta to satisfy mc > 1
    # do NOT always want to do this ... override if we are using force-eta-range
    import scipy.optimize
    # solution to equation with m2 -> 1 is  1 == mc delta 2^(1/5)/(1-delta^2)^(3/5), which is annoying to solve
    def crit_m2(delta):
        eta_val = 0.25*(1-delta*delta)
        return 0.5*mc_center*(eta_val**(-3./5.))*(1-delta) - 1.
    res = scipy.optimize.brentq(crit_m2, 0.001,0.999) # critical value of delta: largest possible for this mc value
    delta_max =np.min([1.1*res,0.99])
    eta_min = 0.25*(1-delta_max*delta_max)
# Need logic for BH-NS scale objects to be reasonable
#   Typical problem for following up these triggers: segment length grows unreasonably long
elif  mc_center < 18 and P.extract_param('q') < 0.6 and opts.propose_initial_grid and not(opts.allow_subsolar):  # BH-NS scale, want to make sure we do a decent job at covering high-mass-ratio end
   import scipy.optimize
   # solution to equation with m2 -> 1 is  1 == mc delta 2^(1/5)/(1-delta^2)^(3/5), which is annoying to solve
   def crit_m2(delta):
       eta_val = 0.25*(1-delta*delta)
       return 0.5*mc_center*(eta_val**(-3./5.))*(1-delta) - 3.
   res = scipy.optimize.brentq(crit_m2, 0.001,0.999) # critical value of delta: largest possible for this mc value
   delta_max =np.min([1.1*res,0.99])
   eta_min = 0.25*(1-delta_max*delta_max)
   if "SNR" in event_dict.keys():
       if event_dict["SNR"]>15 :  # If loud, allow the upper limit to deviate from the maximum
           q_max = np.mean( [P.extract_param('q'),1])   # a guess, trying to exclude a significant chunk of space
           eta_max = q_max/(1.+q_max)**2
# High mass ratio configuration.  PROTOTYPE, NEEDS LOTS OF WORK FOR BH-NS, should restore use of  fisher grid!
elif opts.propose_initial_grid and eta_val < 0.1: # this will override the previous work
    eta_min =0.25*eta_val
    eta_max= np.min([0.249999,4*eta_val])
    delta_max = np.sqrt(1. - 4*eta_min)
    delta_min = np.sqrt(1. - 4*eta_max)
    tune_grid = True
    if opts.force_notune_initial_grid:
        tune_grid=False

chieff_center = P.extract_param('xi')
chieff_min = np.max([chieff_center -0.3/snr_fac,-1])
chieff_max = np.min([chieff_center +0.4/snr_fac,1])  # bias high, given natural q ellipsoids
if chieff_min >0 and use_gracedb_event:
    chieff_min = -0.1   # make sure to cover spin zero, most BBH have zero spin and missing zero is usually an accident of the search recovered params

if opts.scale_mc_range:
    mc_center = (mc_min + mc_max)/2
    mc_width = (mc_max - mc_min)
    mc_min = mc_center -0.5*opts.scale_mc_range*mc_width
    mc_max = mc_center +0.5*opts.scale_mc_range*mc_width

if use_ini:
    engine_dict = dict(config['engine'])
    if 'chirpmass-min' in engine_dict:
        mc_min = float(engine_dict['chirpmass-min'])
    if 'chirpmass-max' in engine_dict:
        mc_max = float(engine_dict['chirpmass-max'])
    if 'q-min'  in engine_dict:
        q_min = float(engine_dict['q-min'])
        eta_min = q_min/(1.+q_min)**2
    if 'q-max'  in engine_dict:
        q_max = float(engine_dict['q-max'])
        eta_max = q_max/(1.+q_max)**2
        if eta_max >=0.25:
            eta_max = 0.24999999  # rounding/finite-precision issues may cause nan problems 
    if 'ecc_min' in engine_dict:
        ecc_range_str = "  ["+str(engine_dict['ecc_min'])+","+str(engine_dict['ecc_max'])+"]"
    if 'meanPerAno_min' in engine_dict:
        meanPerAno_range_str = "  ["+str(engine_dict['meanPerAno_min'])+","+str(engine_dict['meanPerAno_max'])+"]"
        
mc_range_str = "  ["+str(mc_min_tight)+","+str(mc_max_tight)+"]"  # Use a tight placement grid for CIP
if not(opts.manual_mc_min is None):
    mc_min = opts.manual_mc_min
if not(opts.manual_mc_max is None):
    mc_max = opts.manual_mc_max
mc_range_str_cip = " --mc-range ["+str(mc_min)+","+str(mc_max)+"]"
if not(opts.force_mc_range is None):
    mc_range_str_cip = " --mc-range " + opts.force_mc_range
elif opts.limit_mc_range:
    line_here = list(map(float,opts.limit_mc_range.replace('[','').replace(']','').split(',') ))
    mc_min_lim,mc_max_lim = line_here
    mc_min = np.max([mc_min,mc_min_lim])
    mc_max = np.min([mc_max,mc_max_lim])
    mc_range_str_cip = " --mc-range ["+str(mc_min)+","+str(mc_max)+"]"
eta_range_str = "  ["+str(eta_min_tight) +","+str(eta_max_tight)+"]"  # default will include  1, as we work with BBHs
eta_range_str_cip = " --eta-range ["+str(eta_min) +","+str(eta_max)+"]"  # default will include  1, as we work with BBHs
if not (opts.force_eta_range is None):
    eta_range_str_cip = " --eta-range " + opts.force_eta_range


###
### Write arguments
###
helper_ile_args ="X "
helper_test_args="X "
helper_cip_args = "X "
helper_cip_arg_list = []

chieff_str = '' # Scoping issue fix

helper_test_args += "  --method lame  --parameter mc --parameter eta  --iteration $(macroiteration) "
if not opts.assume_nospin:
    helper_test_args += " --parameter xi "  # require chi_eff distribution to be stable
if not opts.test_convergence:
    helper_test_args+= " --always-succeed "

helper_ile_args += " --save-P 0.1 "   # truncate internal data structures (should do better memory management/avoid need for this if --save-samples is not on)
if not (opts.fmax is None):
    helper_ile_args += " --fmax " + str(opts.fmax)  # pass actual fmax
else:
    helper_ile_args += " --fmax " + str(fmax)
rescaled_base_ile = False
helper_cip_args_extra=''
if "SNR" in event_dict.keys():
    snr_here = event_dict["SNR"]
    if snr_here > 25:
        lnL_expected = snr_here**2 /2. - np.log(10)*50  # remember, 10^308 is typical overflow scale, giving range of 10^100 above
        if not(opts.internal_ile_auto_logarithm_offset) and not opts.internal_ile_use_lnL:
            helper_ile_args += " --manual-logarithm-offset " + str(lnL_expected)
        if not(opts.use_downscale_early):
            # Blocks in ALL iterations, which is not recommended
            helper_cip_args +=   " --lnL-protect-overflow " #" --lnL-shift-prevent-overflow " + str(lnL_expected)   # warning: this can have side effects if the shift makes lnL negative, as the default value of the fit is 0 !
        elif snr_here <35:
            helper_cip_args_extra +=  " --lnL-protect-overflow " # " --lnL-shift-prevent-overflow " + str(lnL_expected)   # warning: this can have side effects if the shift makes lnL negative, as the default value of the fit is 0 !
        else:
            helper_cip_args_extra = " --internal-use-lnL "   # always use lnL scaling for loud signals at late times, overflow issue for most integrators
        rescaled_base_ile = True
if opts.internal_ile_auto_logarithm_offset and not opts.internal_ile_use_lnL:
    helper_ile_args += " --auto-logarithm-offset "
    rescaled_base_ile = True
if opts.internal_ile_use_lnL:
    helper_ile_args += ' --internal-use-lnL '
if opts.internal_cip_use_lnL:
    helper_cip_args += " --internal-use-lnL "

if not opts.use_osg:
    if '/' in opts.cache:
        helper_ile_args += " --cache " + opts.cache
    else:
        helper_ile_args += " --cache " + opts.working_directory+ "/" + opts.cache
else:
    helper_ile_args += " --cache local.cache "
helper_ile_args += " --event-time " + str(event_dict["tref"])
for ifo in ifos:
    helper_ile_args += " --channel-name "+ifo+"="+channel_names[ifo]
    helper_ile_args += " --psd-file "+ifo+"="+psd_names[ifo]
    if (not use_ini):
      if not (opts.fmin is None):
        helper_ile_args += " --fmin-ifo "+ifo+"="+str(opts.fmin)
    elif not(opts.use_ini is None):
        helper_ile_args += " --fmin-ifo "+ifo+"="+str(unsafe_config_get(config,['lalinference','flow'])[ifo])
#helper_ile_args += " --fmax " + str(fmax)
helper_ile_args += " --fmin-template " + str(opts.fmin_template)
if not use_ini:
    helper_ile_args += " --reference-freq " + str(opts.fmin_template)  # in case we are using a code which allows this to be specified
else:
    helper_ile_args += " --reference-freq " + str(unsafe_config_get(config,['engine','fref']))
approx_str= "SEOBNRv4"  # default, should not be used.  See also cases where grid is tuned

dmin = 1 # default
if use_ini:
    # See above, provided by ini file
    engine_dict = dict(config['engine'])
    if 'distance-min' in engine_dict:
        dmin = float(engine_dict['distance-min'])
    if 'distance-max' in engine_dict:
        dmax = float(engine_dict['distance-max'])
    else:
        mc_Msun = P.extract_param('mc')/lal.MSUN_SI
        dmax_guess =(1./snr_fac)* 2.5*2.26*typical_bns_range_Mpc[opts.observing_run]* (mc_Msun/1.2)**(5./6.)
        dmax = np.min([dmax_guess,10000]) # place ceiling
    if internal_dmax is None: # overrride ini file if already set.  Note this will override the lowlatency propose approx
        internal_dmax = dmax
    
if opts.lowlatency_propose_approximant:
#    approx  = lalsim.TaylorF2
    approx_str = "SpinTaylorT4"
    mc_Msun = P.extract_param('mc')/lal.MSUN_SI
    if mc_Msun > 6:
#        approx = lalsim.SEOBNRv4_ROM
        approx_str = "SEOBNRv4"
    helper_ile_args += " --approx " + approx_str
    helper_cip_args += " --approx-output " + approx_str  # insure the two codes are talking about the same approximant. Avoid silly things.

    # Also choose d-max. Relies on archival and fixed network sensitvity estimates.
    dmax_guess =(1./snr_fac)* 2.5*2.26*typical_bns_range_Mpc[opts.observing_run]* (mc_Msun/1.2)**(5./6.)
    dmax_guess = np.min([dmax_guess,10000]) # place ceiling
    if internal_dmax is None:
        internal_dmax = dmax_guess
    # if opts.use_ini is None:
    #     helper_ile_args +=  " --d-max " + str(int(internal_dmax))  # note also used below

    if (opts.data_LI_seglen is None) and  (opts.data_start_time is None) and not(use_ini):
        # Also choose --data-start-time, --data-end-time and disable inverse spectrum truncation (use tukey)
        #   ... note that data_start_time was defined BEFORE with the datafind job
        T_window_raw = 1.1/lalsimutils.estimateDeltaF(P)  # includes going to next power of 2, AND a bonus factor of a few
        T_window_raw = np.max([T_window_raw,4])  # can't be less than 4 seconds long
        print(" Time window : ", T_window_raw, " based on fmin  = ", P.fmin)
        data_start_time = np.max([int(P.tref - T_window_raw -2 )  , data_start_time_orig])  # don't request data we don't have! 
        data_end_time = int(P.tref + 2)
        helper_ile_args += " --data-start-time " + str(data_start_time) + " --data-end-time " + str(data_end_time)  + " --inv-spec-trunc-time 0 --window-shape 0.01"
    if use_ini:
        T_window_raw = unsafe_config_get(config,['engine','seglen'])
        data_start_time = np.max([int(P.tref - T_window_raw -2 )  , data_start_time_orig])  # don't request data we don't have! 
        data_end_time = int(P.tref + 2)
        T_window = data_start_time - data_end_time
        window_shape = opts.data_tukey_window_time*2/T_window  # make it clear that this is a one-sided interval
        helper_ile_args += " --data-start-time " + str(data_start_time) + " --data-end-time " + str(data_end_time)  + " --inv-spec-trunc-time 0 --window-shape " + str(window_shape)
        if opts.psd_assume_common_window:
            helper_ile_args += " --psd-window-shape {} ".format(window_shape)

if not(internal_dmax is None):
    helper_ile_args +=  " --d-max " + str(int(internal_dmax))
    if dmin != 1: # if not default value, add argument
        helper_ile_args += " --d-min {} ".format(dmin)
    if opts.ile_distance_prior:
        helper_ile_args += " --d-prior {} ".format(opts.ile_distance_prior)   # moving here from pseudo_pipe
    if opts.internal_marginalize_distance and not(opts.internal_marginalize_distance_file):
        # Generate marginalization file (should probably be in DAG? But we may also want to override it with internal file)
        cmd_here = " util_InitMargTable --d-max {} ".format(internal_dmax)
        if dmin != 1:
            cmd_here += " --d-min {} ".format(dmin)
        if opts.ile_distance_prior:
            cmd_here += " --d-prior {} ".format(opts.ile_distance_prior)
        os.system(cmd_here)
        prefix_file = "{}/".format(opts.working_directory)
        if opts.use_osg:
            prefix_file =''
            transfer_files += [ opts.working_directory+"/distance_marginalization_lookup.npz" ]
        helper_ile_args += " --distance-marginalization  --distance-marginalization-lookup-table {}distance_marginalization_lookup.npz ".format(prefix_file)
    elif opts.internal_marginalize_distance_file:
        helper_ile_args += " --distance-marginalization "
        if not(opts.use_osg):
            helper_ile_args += " --distance-marginalization-lookup-table {} ".format(opts.internal_marginalize_distance_file)
        else:
            transfer_files += [ opts.internal_marginalize_distance_file ]
            fname_short = opts.internal_marginalize_distance_file.split('/')[-1]
            helper_ile_args += " --distance-marginalization-lookup-table {} ".format(fname_short)

if not ( (opts.data_start_time is None) and (opts.data_end_time is None)):
    # Manually set the data start and end time.
    T_window = opts.data_end_time - opts.data_start_time
    # Use LI-style tukey windowing
    window_shape = opts.data_tukey_window_time*2/T_window
    data_start_time =opts.data_start_time
    data_end_time =opts.data_end_time
    helper_ile_args += " --data-start-time " + str(data_start_time) + " --data-end-time " + str(data_end_time)  + " --inv-spec-trunc-time 0 --window-shape " + str(window_shape)
    if opts.psd_assume_common_window:
        helper_ile_args += " --psd-window-shape {} ".format(window_shape)

elif opts.data_LI_seglen:
    seglen = opts.data_LI_seglen
    # Use LI-style positioning of trigger relative to 2s before end of buffer
    # Use LI-style tukey windowing
    window_shape = opts.data_tukey_window_time*2/seglen
    data_end_time = event_dict["tref"]+2
    data_start_time = event_dict["tref"] +2 - seglen
    helper_ile_args += " --data-start-time " + str(data_start_time) + " --data-end-time " + str(data_end_time)  + " --inv-spec-trunc-time 0  --window-shape " + str(window_shape)
    if opts.psd_assume_common_window:
            helper_ile_args += " --psd-window-shape {} ".format(window_shape)

if opts.assume_eccentric:
    helper_ile_args += " --save-eccentricity "
    if opts.use_meanPerAno:
        helper_ile_args += " --save-meanPerAno "
if opts.propose_initial_grid_fisher: # and (P.extract_param('mc')/lal.MSUN_SI < 10.):
    cmd  = "util_AnalyticFisherGrid.py  --inj-file-out  proposed-grid  "
    # Add standard downselects : do not have m1, m2 be less than 1
    if not(opts.force_mc_range is None):
        # force downselect based on this range
        cmd += " --downselect-parameter mc --downselect-parameter-range " + opts.force_mc_range 
    if not(opts.force_eta_range is None):
        cmd += " --downselect-parameter eta --downselect-parameter-range " + opts.force_eta_range 
    cmd += " --fmin " + str(opts.fmin_template)
    if opts.data_LI_seglen and not (opts.no_enforce_duration_bound):  
        cmd += " --enforce-duration-bound " + str(opts.data_LI_seglen)
    if not(opts.allow_subsolar):
        cmd += "  --downselect-parameter m1 --downselect-parameter-range [1,10000]   --downselect-parameter m2 --downselect-parameter-range [1,10000]  "
    if opts.assume_nospin:
        grid_size = 1000   # 500 was too small with noise
    else:
        chieff_range = "[{},{}]".format( float(chieff_min), float(chieff_max))
        chieff_range= chieff_range.replace(' ', '')   # assumes all chieff are possible
        if opts.propose_fit_strategy:
            # If we don't have a fit plan, use the NS spin maximum as the default
            if (P.extract_param('mc')/lal.MSUN_SI < 2.6):   # assume a maximum NS mass of 3 Msun
                chi_max = 0.1   # propose a maximum NS spin
                chi_range = "[{},{}]".format(-chi_max,chi_max).replace(' ','')
                chieff_range = chi_range  # force to be smaller
                cmd += " --downselect-parameter s1z --downselect-parameter-range " + chi_range + "   --downselect-parameter s2z --downselect-parameter-range " + chi_range 

        cmd += " --random-parameter chieff_aligned  --random-parameter-range " + chieff_range
        grid_size =2500
    if opts.assume_eccentric:
        cmd += " --random-parameter eccentricity --random-parameter-range " + ecc_range_str
        if opts.use_meanPerAno:
            cmd += " --random-parameter meanPerAno --random-parameter-range " + meanPerAno_range_str
    if "SNR" in event_dict:
        grid_size *= np.max([1,event_dict["SNR"]/15])  # more grid points at higher amplitude. Yes, even though we also contract the paramete range
    if not (opts.force_initial_grid_size is None):
        grid_size = opts.force_initial_grid_size
    cmd += " --grid-cartesian-npts  " + str(int(grid_size))
    print(" Executing grid command ", cmd)
    os.system(cmd)


elif opts.propose_initial_grid:
    delta_grid_min = delta_min_tight
    delta_grid_max = delta_max_tight 

    # add basic mass parameters
    cmd  = "util_ManualOverlapGrid.py  --fname proposed-grid --skip-overlap "
    mass_string_init = " --random-parameter mc --random-parameter-range   " + mc_range_str + "  --random-parameter delta_mc --random-parameter-range '[" + str(delta_grid_min) +"," + str(delta_grid_max) + "]'  "
    cmd+= mass_string_init
    # Add standard downselects : do not have m1, m2 be less than 1
    if not(opts.force_mc_range is None):
        # force downselect based on this range
        cmd += " --downselect-parameter mc --downselect-parameter-range " + opts.force_mc_range 
    if not(opts.force_eta_range is None):
        cmd += " --downselect-parameter eta --downselect-parameter-range " + opts.force_eta_range 
    cmd += " --fmin " + str(opts.fmin_template)
    if opts.data_LI_seglen and not (opts.no_enforce_duration_bound):  
        cmd += " --enforce-duration-bound " + str(opts.data_LI_seglen)
    if not(opts.allow_subsolar):
        cmd += "  --downselect-parameter m1 --downselect-parameter-range [1,10000]   --downselect-parameter m2 --downselect-parameter-range [1,10000]  "
    if tune_grid:
        cmd += " --reset-grid-via-match --match-value 0.85 --use-fisher  --use-fisher-resampling --approx  " + approx_str # ow, but useful
    if opts.assume_nospin:
        grid_size = 1000   # 500 was too small with zero noise
    else:
        chieff_range = "[{},{}]".format( float(chieff_min), float(chieff_max))
        chieff_range= chieff_range.replace(' ', '')   # assumes all chieff are possible
        if opts.propose_fit_strategy:
            # If we don't have a fit plan, use the NS spin maximum as the default
            if (P.extract_param('mc')/lal.MSUN_SI < 2.6):   # assume a maximum NS mass of 3 Msun
                chi_max = 0.1   # propose a maximum NS spin
                chi_range = str([-chi_max,chi_max]).replace(' ','')
                chieff_range = chi_range  # force to be smaller
                cmd += " --downselect-parameter s1z --downselect-parameter-range " + chi_range + "   --downselect-parameter s2z --downselect-parameter-range " + chi_range 

        chieff_str= " --random-parameter chieff_aligned  --random-parameter-range " + chieff_range
        cmd += chieff_str
        # if opts.internal_use_aligned_phase_coordinates:
        #     grid_size=2000
        # else:
        grid_size =2500

        if opts.assume_precessing_spin:
            # Handle problems with SEOBNRv3 failing for aligned binaries -- add small amount of misalignment in the initial grid
            cmd += " --parameter s1x --parameter-range [0.00001,0.00003] "
    if opts.assume_eccentric:
        cmd += " --random-parameter eccentricity --random-parameter-range " + ecc_range_str
        if opts.use_meanPerAno:
            cmd += " --random-parameter meanPerAno --random-parameter-range " + meanPerAno_range_str
    if opts.internal_tabular_eos_file:
        cmd += " --tabular-eos-file {} ".format(opts.internal_tabular_eos_file)
        grid_size *=2  # larger grids needed for discrete realization scenarios
    elif opts.assume_matter and not opts.assume_matter_eos:
        # Do the initial grid assuming matter, with tidal parameters set by the AP4 EOS provided by lalsuite
        # We will leverage working off this to find the lambdaTilde dependence
#        cmd += " --use-eos AP4 "  
        # Choose the lambda range based on chirp mass!  If chirp mass is large, we need to use very low lambda.
        # based on lambda(m) estimate 3000*((mc_center-2.2)/(1.2))**2
        # FIXME: Get real EOS limit?
        def lambda_m_estimate(m):
            if m>2.2:
                return 10
            else:
                return 3000*((2.2-m)/(1.2))**4
        lambda_grid_min=50
        if not(opts.assume_matter_but_primary_bh):
            P.lambda1 = lambda_m_estimate(P.m1/lal.MSUN_SI)
            lambda1_min = np.min([50,P.lambda1*0.2])
            lambda1_max = np.min([1500,P.lambda1*2])
            if opts.assume_matter_conservatively:
                lambda1_min =10
                lambda1_max = 5000
        else:
            lambda1_min = 0
            lambda1_max=0
        P.lambda2 = lambda_m_estimate(P.m2/lal.MSUN_SI)
        lambda2_min = np.min([50,P.lambda2*0.2])
        lambda2_max = np.min([1500,P.lambda2*2])
        if opts.assume_matter_conservatively:
            lambda2_min =10
            lambda2_max = 5000
        cmd += " --random-parameter lambda1 --random-parameter-range [{},{}] --random-parameter lambda2 --random-parameter-range [{},{}] ".format(lambda1_min,lambda1_max,lambda2_min,lambda2_max)
        grid_size *=2   # denser grid
    elif opts.assume_matter and opts.assume_matter_eos:
        cmd += " --use-eos {} ".format(opts.assume_matter_eos.replace('lal_', ''))

    if opts.propose_fit_strategy and not opts.internal_use_aligned_phase_coordinates:
        if (P.extract_param('mc')/lal.MSUN_SI < 10):   # assume a maximum NS mass of 3 Msun
            grid_size *=1.5  # denser grid at low mass, because of tight correlations

    if  not ( opts.internal_use_aligned_phase_coordinates )  and (('quadratic' in fit_method) or ('polynomial' in fit_method) or 'rf' in fit_method) :
        grid_size *= 1.5  # denser initial grid for these methods, since they need more training to stabilize at times. But not for new coordinates

    if "SNR" in event_dict:
        grid_size *= np.max([1,event_dict["SNR"]/15])  # more grid points at higher amplitude. Yes, even though we also contract the paramete range

    if not (opts.force_initial_grid_size is None):
        grid_size = opts.force_initial_grid_size
    cmd += " --grid-cartesian-npts  " + str(int(grid_size))
    print(" Executing grid command ", cmd)
    os.system(cmd)

    # special grids for rf, which tend to flop around unless they have broad-spectrum training data. If loud, use sparse wider grid in chieff *at low mass* (eg NSBH, BNS)
    if fit_method=='rf' and (mc_center < 4) and not(opts.assume_nospin):
       # create a second grid for rf low mass, because the chieff range can sometimes be quite narrow if the SNR is loud
        chieff_str_new = " --random-parameter chieff_aligned  --random-parameter-range  [-0.5,0.5] "
        cmd_alt = cmd.replace(chieff_str, chieff_str_new)
        cmd_alt = cmd_alt.replace(" --grid-cartesian-npts  " + str(int(grid_size)), " --grid-cartesian-npts  " + str(int(grid_size/3))) # small perturbation
        cmd_alt = cmd_alt.replace("fname proposed-grid",  "fname proposed-grid-extra")
        print(" Executing supplementary grid command for rf, to stabilize spin fits ", cmd_alt)
        os.system(cmd_alt)
        cmd_add = "ligolw_add proposed-grid.xml.gz proposed-grid-extra.xml.gz --output tmp.xml.gz"
        os.system(cmd_add)
        os.system("mv tmp.xml.gz proposed-grid.xml.gz")

    # retarget if we are using force_eta_range: for things like GW190814, put more grid at high q
    # try to avoid sampling too much close by
    if  (mc_center < 8 and P.extract_param('q') < 0.5):
        # create a SECOND grid and join to first, to flesh out high q specifically
        q_grid_max = np.mean( [P.extract_param('q'),0.7])   # a guess, trying to exclude a significant chunk of space
        delta_grid_min = (1-q_grid_max)/(1+q_grid_max)
        qref = P.extract_param('q')*0.5
        delta_grid_max = (1-qref)/(1+qref)

        mass_string_init_new = " --random-parameter mc --random-parameter-range   " + mc_range_str + "  --random-parameter delta_mc --random-parameter-range '[" + str(delta_grid_min) +"," + str(delta_grid_max) + "]'  "
        if not(opts.assume_nospin):
            cmd = cmd.replace(mass_string_init, mass_string_init_new)
            chieff_str_new = " --random-parameter chieff_aligned  --random-parameter-range '[-0.5,0.5]' "
            cmd.replace(chieff_str, chieff_str_new)

        cmd = cmd.replace("fname proposed-grid",  "fname proposed-grid-extra")
        print(" Executing supplementary grid command for high q ", cmd)
        os.system(cmd)
        cmd_add = "ligolw_add proposed-grid.xml.gz proposed-grid-extra.xml.gz --output tmp.xml.gz"
        os.system(cmd_add)
        os.system("mv tmp.xml.gz proposed-grid.xml.gz")



    # if opts.assume_matter:
    #     # Now perform a puffball in lambda1 and lambda2
    #     cmd_puff = " util_ParameterPuffball.py --parameter LambdaTilde  --inj-file proposed-grid.xml.gz --inj-file-out proposed-grid_puff_lambda --downselect-parameter lambda1 --downselect-parameter-range [0.1,5000] --downselect-parameter lambda2 --downselect-parameter-range [0.1,5000]"
    #     os.system(cmd_puff)
    #     # Now add these two together
    #     # ideally, ligolw_add will work... except it fails
    #     P_A = lalsimutils.xml_to_ChooseWaveformParams_array("proposed-grid.xml.gz")
    #     P_B = lalsimutils.xml_to_ChooseWaveformParams_array("proposed-grid_puff_lambda.xml.gz")
    #     lalsimutils.ChooseWaveformParams_array_to_xml(P_A+P_B, "proposed-grid.xml.gz")

puff_factor=2
if opts.propose_fit_strategy and (not opts.gracedb_id is None):
    # use a puff factor that depends on mass.  Use a larger puff factor below around 10.
    if (P.extract_param('mc')/lal.MSUN_SI < 10):   # assume a maximum NS mass of 3 Msun
        puff_factor =6  # high q, use more aggressive puff


if opts.propose_ile_convergence_options:
    helper_ile_args += " --time-marginalization  --inclination-cosine-sampler   --n-max 4000000 --n-eff {} ".format(opts.ile_n_eff)
    if not(opts.internal_use_gracedb_bayestar):
        helper_ile_args += " --declination-cosine-sampler  "  # skymap coordinates all fixed
    else:
        helper_ile_args += " --n-chunk 500 " # much smaller chunk size for integration for ILE if we are using an input skymap! Slow, but does the hard dimension
    # Modify someday to use the SNR to adjust some settings
    # Proposed option will use GPUs
    # Note that number of events to analyze is controlled by a different part of the workflow !
    helper_ile_args += " --vectorized --gpu   --srate {} ".format(srate)
    if opts.internal_propose_ile_convergence_freezeadapt:
        helper_ile_args += "  --no-adapt-after-first --no-adapt-distance  "
                    

    prefactor = 0.1 # typical value. 0.3 fine for low amplitude, 0.1 for GMM
    if (opts.internal_propose_ile_adapt_log):
        helper_ile_args += " --adapt-log "
    else:
        if snr_fac > 1.5:  # this is a pretty loud signal, so we need to tune the adaptive exponent too!
#            if not(rescaled_base_ile):
            helper_ile_args += " --adapt-weight-exponent " + str(prefactor/np.power(snr_fac/1.5,2))
#            else:
#                # if we are adjusting the logarithm scale based on signal strength, we don't want to smash it too much, so only use the default prefactor
#                helper_ile_args += " --adapt-weight-exponent " + str(prefactor)
        else:
            helper_ile_args += " --adapt-weight-exponent  {} ".format(prefactor)  

if opts.internal_use_gracedb_bayestar:
    helper_ile_args += " --skymap-file {}/bayestar.fits ".format(opts.working_directory)

if opts.internal_ile_rotate_phase:
    helper_ile_args += " --internal-rotate-phase "


puff_max_it=0
helper_puff_args = " --parameter mc --parameter eta --fmin {} --fref {} ".format(opts.fmin_template,opts.fmin_template)
if opts.assume_eccentric:
    helper_puff_args += " --parameter eccentricity "
    if opts.use_meanPerAno:
        helper_puff_args += " --parameter meanPerAno "

if event_dict["MChirp"] >25:
    # at high mass, mc/eta correlation weak, don't want to have eta coordinate degeneracy at q=1 to reduce puff proposals  near there
    helper_puff_args = " --parameter mc --parameter delta_mc "  
if opts.propose_fit_strategy:
    puff_max_it= 0
    # Strategy: One iteration of low-dimensional, followed by other dimensions of high-dimensional
    print(" Fit strategy NOT IMPLEMENTED -- currently just provides basic parameterization options. Need to work in real strategies (e.g., cip-arg-list)")
    lnLoffset_late = 15 # default
    helper_cip_args += ' --no-plots --fit-method {}  '.format(fit_method)
    if not opts.internal_use_aligned_phase_coordinates:
        helper_cip_args += '   --parameter mc --parameter delta_mc '
    else:
        helper_cip_args += " --parameter-implied mu1 --parameter-implied mu2 --parameter-nofit mc --parameter delta_mc "  
    if 'gp' in fit_method:
        helper_cip_args += " --cap-points 12000 "
    if not opts.no_propose_limits:
        helper_cip_args += mc_range_str_cip + eta_range_str_cip
    if opts.force_chi_max:
        helper_cip_args += " --chi-max {} ".format(opts.force_chi_max)
    if opts.force_chi_small_max:
        helper_cip_args += " --chi-small-max {} ".format(opts.force_chi_small_max)
    if opts.force_lambda_max:
        helper_cip_args += " --lambda-max {} ".format(opts.force_lambda_max)
    if opts.force_lambda_small_max:
        helper_cip_args += " --lambda-small-max {} ".format(opts.force_lambda_small_max)

    helper_cip_arg_list_common = str(helper_cip_args)[1:] # drop X
    n_it_early =3
    n_it_mid = 4
    if opts.assume_highq:
        n_it_early =2
        qmin_puff = 0.05 # 20:1
    if opts.assume_well_placed:
        n_it_early = 1
    if opts.internal_use_aligned_phase_coordinates:
        n_it_early = 2
        n_it_mid =2
    helper_cip_arg_list = [str(n_it_early) + " " + helper_cip_arg_list_common, "{} ".format(n_it_mid) +  helper_cip_arg_list_common +helper_cip_args_extra ]

    # downscale factor, if requested
    if opts.use_downscale_early and event_dict["SNR"]>15:
        scale_fac = (15/event_dict["SNR"])**2
        helper_cip_arg_list[0] += " --lnL-downscale-factor {} ".format(scale_fac)
            
    # Impose a cutoff on the range of parameter used, IF the fit is a gp fit
    if 'gp' in fit_method:
        helper_cip_arg_list[0] += " --lnL-offset " + str(lnLoffset_all) 
        for indx in np.arange(1,len(helper_cip_arg_list)):  # do NOT constrain the first CIP, as it has so few points!
            helper_cip_arg_list[indx] += " --lnL-offset " + str(lnLoffset_early)

    extended_early=False
    if opts.use_quadratic_early or opts.use_cov_early:
        extended_early=True
        helper_cip_arg_list = [helper_cip_arg_list[0]] + helper_cip_arg_list  # augment the number of levels with an early quadratic or 'cov' stage
        if opts.use_quadratic_early:
            helper_cip_arg_list[0] = helper_cip_arg_list[0].replace('fit-method '+fit_method, 'fit-method quadratic ')
        else:
            helper_cip_arg_list[0] = helper_cip_arg_list[0].replace('fit-method '+fit_method, 'fit-method cov ')
        if 'gp' in fit_method: # lnL offset only enabled for gp so far
            helper_cip_arg_list[0] = helper_cip_arg_list[0].replace(" --lnL-offset " + str(lnLoffset_all)," --lnL-offset " + str(lnL_start) )  # more sane initial range for quadratic; see later
        elif 'rf' in fit_method:
            helper_cip_arg_list[0] += " --lnL-offset " +   str(lnL_start)
    elif opts.use_gp_early:
        extended_early=True
        helper_cip_arg_list = [helper_cip_arg_list[0]] + helper_cip_arg_list  # augment the number of levels with an early gp stage
        helper_cip_arg_list[0] = helper_cip_arg_list[0].replace('fit-method '+fit_method, 'fit-method gp')
        helper_cip_arg_list[0] += " --lnL-offset " + str(lnLoffset_all)

    if not opts.assume_nospin:
        helper_puff_args += " --parameter chieff_aligned "
        if not opts.assume_precessing_spin:
            # aligned spin branch
            if opts.internal_use_aligned_phase_coordinates:
                # mu1,mu2,q,s2z are coordinates, with mu1,mu2,delta_mc already implemented
                helper_cip_args += ' --parameter-nofit s1z --parameter-nofit s2z ' # --parameter-implied chiMinus  # keep chiMinus out, until we add flexible tools                
                helper_cip_arg_list[0] += " --parameter-nofit s1z --parameter-nofit s2z  "
                for indx in np.arange(1,len(helper_cip_arg_list)): # allow for variable numbers of subsequent steps, with different settings
                    helper_cip_arg_list[indx] += ' --parameter-implied chiMinus --parameter-nofit s1z --parameter-nofit s2z '
            elif not opts.assume_highq:
                # normal aligned spin
                helper_cip_args += ' --parameter-implied xi  --parameter-nofit s1z --parameter-nofit s2z ' # --parameter-implied chiMinus  # keep chiMinus out, until we add flexible tools
                helper_cip_arg_list[0] +=  ' --parameter-implied xi  --parameter-nofit s1z --parameter-nofit s2z ' 
                for indx in np.arange(1,len(helper_cip_arg_list)): # allow for variable numbers of subsequent steps, with different settings
                    helper_cip_arg_list[indx] += ' --parameter-implied xi  --parameter-implied chiMinus --parameter-nofit s1z --parameter-nofit s2z '
            else: # highq
                helper_cip_args += ' --parameter-implied xi  --parameter-nofit s1z ' # --parameter-implied chiMinus  # keep chiMinus out, until we add flexible tools
                helper_cip_arg_list[0] +=  ' --parameter-implied xi  --parameter-nofit s1z  ' 
                for indx in np.arange(1,len(helper_cip_arg_list)): # allow for variable numbers of subsequent steps, with different settings
                    helper_cip_arg_list[indx] += ' --parameter-implied xi   --parameter-nofit s1z   ' 
                
            puff_max_it=4

        else: #  opts.assume_precessing_spin:
            if not(opts.internal_use_aligned_phase_coordinates):
                helper_cip_args += ' --parameter-implied xi  ' # --parameter-implied chiMinus  # keep chiMinus out, until we add flexible tools
            # REMOVE intermediate stage where we used to do chiMinus ... it is NOT the dominant issue for precession, by far
            # so only 3 stages (0,1,2).  0 is aligned; 1 has chi_p; 2 has standard uniform spin prior (or volumetric)
            helper_cip_arg_list = [str(n_it_early) + " " + helper_cip_arg_list_common, "2 " +  helper_cip_arg_list_common,"3 " +  helper_cip_arg_list_common + helper_cip_args_extra  ]
            # downscale factor, if requested
            if opts.use_downscale_early and event_dict["SNR"]>15:
                scale_fac = (15/event_dict["SNR"])**2
                helper_cip_arg_list[0] += " --lnL-downscale-factor {} ".format(scale_fac)
                helper_cip_arg_list[1] += " --lnL-downscale-factor {} ".format(scale_fac)

            if not opts.assume_highq:
                # First three batches use cartersian coordinates
                helper_cip_arg_list[0] += "  --parameter-nofit s1z --parameter-nofit s2z "
                helper_cip_arg_list[1] +=  '    --parameter-nofit s1z --parameter-nofit s2z  ' 
                # element 2 can have an optional prior change, so do NOT apply it here
                if not(opts.internal_use_aligned_phase_coordinates):
                    helper_cip_arg_list[0] +=  '  --parameter-implied xi   ' 
                    helper_cip_arg_list[1] +=  '  --parameter-implied xi   ' 
                if not(opts.internal_use_rescaled_transverse_spin_coordinates):
                    helper_cip_arg_list[0] +=   ' --use-precessing --parameter-nofit s1x --parameter-nofit s1y --parameter-nofit s2x  --parameter-nofit s2y  --no-adapt-parameter s1x --no-adapt-parameter s1y --no-adapt-parameter s2x --no-adapt-parameter s2y --transverse-prior taper-down '
                    helper_cip_arg_list[1] +=   ' --parameter-implied chi_p --use-precessing  --parameter-nofit s1x --parameter-nofit s1y --parameter-nofit s2x  --parameter-nofit s2y   --transverse-prior taper-down '
                else:
                    helper_cip_arg_list[0] = helper_cip_arg_list[0].replace('--parameter-nofit s1z --parameter-nofit s2z','')
                    helper_cip_arg_list[1] = helper_cip_arg_list[1].replace('--parameter-nofit s1z --parameter-nofit s2z','')
                    helper_cip_arg_list[0] +=  " --use-precessing  --parameter-nofit s1z_bar --parameter-nofit s2z_bar --parameter-nofit chi1_perp_u --parameter-nofit chi2_perp_u --parameter-nofit phi1 --parameter-nofit phi2 --no-adapt-parameter phi1 --no-adapt-parameter phi2 --no-adapt-parameter chi1_perp_u --no-adapt-parameter chi2_perp_u "
                    helper_cip_arg_list[1] +=  "  --use-precessing  --parameter-nofit s1z_bar --parameter-nofit s2z_bar --parameter-nofit chi1_perp_u --parameter-nofit chi2_perp_u --parameter-nofit phi1 --parameter-nofit phi2 --no-adapt-parameter phi1 --no-adapt-parameter phi2 --no-adapt-parameter chi1_perp_u --no-adapt-parameter chi2_perp_u "
                    helper_cip_arg_list[1] +=   ' --parameter-implied chi_p  '
                
                if not(opts.internal_use_aligned_phase_coordinates):
                    helper_cip_arg_list[2] += ' --parameter-implied xi  --parameter-implied chiMinus  ' 
                else:
                    helper_cip_arg_list[2] += '   --parameter-implied chiMinus  ' 
                if not(opts.assume_volumetric_spin) and not(opts.internal_use_rescaled_transverse_spin_coordinates):
                    helper_cip_arg_list[2] +=  '  --use-precessing  --parameter-nofit chi1 --parameter-nofit chi2 --parameter-nofit cos_theta1 --parameter-nofit cos_theta2 --parameter-nofit phi1 --parameter-nofit phi2   --parameter-implied s1x --parameter-implied s1y --parameter-implied s2x --parameter-implied s2y '
                elif opts.internal_use_rescaled_transverse_spin_coordinates:
                    helper_cip_arg_list[2] +=  '  --use-precessing    --parameter-implied s1x --parameter-implied s1y --parameter-implied s2x --parameter-implied s2y '
                    helper_cip_arg_list[2] +=  "   --prior-in-integrand-correction  'uniform_over_rbar_singular' --parameter-nofit s1z_bar --parameter-nofit s2z_bar --parameter-nofit chi1_perp_u --parameter-nofit chi2_perp_u --parameter-nofit phi1 --parameter-nofit  phi2 "
                else:
                    # if we are doing a FLAT structure, we are volumeric or not
                    helper_cip_args += '  --parameter-nofit s1z --parameter-nofit s2z  '
                    # this will be perfectly adequate volumetric result ...but note the spin priors above are using more concentrated spins near the origin
                    helper_cip_arg_list[2] +=  '    --parameter-nofit s1z --parameter-nofit s2z  ' 
                    helper_cip_arg_list[2] +=   ' --use-precessing --parameter s1x --parameter s1y --parameter s2x  --parameter s2y   '
                # # Default prior is *volumetric*
                # helper_cip_args += ' --parameter-nofit s1x --parameter-nofit s1y --parameter-nofit s2x  --parameter-nofit s2y --use-precessing '
                # helper_cip_arg_list[1] +=   ' --parameter s1x --parameter s1y --parameter s2x  --parameter s2y --use-precessing '

                # if extended_early:
                #     for indx in np.arange(3,len(helper_cip_arg_list)-1): # allow for variable numbers of subsequent steps, with different settings
                #         helper_cip_arg_list[indx] += ' --parameter-implied xi  --parameter-implied chiMinus --parameter-nofit s1z --parameter-nofit s2z ' 
                #         helper_cip_arg_list[indx] +=   ' --use-precessing --parameter s1x --parameter s1y --parameter s2x  --parameter s2y  '

                # Last iterations are with a polar spin, to get spin prior  (as usually requested). Fir is the same as before, but sampling is more efficient
                # if not opts.assume_volumetric_spin and len(helper_cip_arg_list)>3:
                # else:
                #     print(" IMPLEMENT VOLUMETRIC SPINS - no longer automatic")
                #     helper_cip_arg_list.pop() # remove last stage of iterations as superfluous
        
                # Change convergence threshold at late times
                #            helper_cip_arg_list[2].replace( " --lnL-offset " + str(lnLoffset_early), " --lnL-offset " + str(lnLoffset_late)
                helper_cip_arg_list[-1].replace( " --lnL-offset " + str(lnLoffset_early), " --lnL-offset " + str(lnLoffset_late))
            else:  # strategy for high q
                helper_cip_arg_list[0] +=  '  --parameter-implied xi  --parameter-nofit s1z  '  # note PURE ALIGNED SPIN SO FAR
                helper_cip_arg_list[0] +=   ' --use-precessing  '
            
                helper_cip_arg_list[1] += ' --parameter-implied xi  --parameter-nofit s1z --parameter-implied chi1_perp --parameter-nofit s1x --parameter-nofit s1y ' 
                helper_cip_arg_list[1] +=   ' --use-precessing   '

                # this will be perfectly adequate volumetric result
                helper_cip_arg_list[2] += ' --parameter-implied xi  --parameter-nofit s1z  ' 
                helper_cip_arg_list[2] +=   ' --use-precessing  --parameter s1x --parameter s2x --parameter-implied chi2_perp --parameter-nofit s2x --parameter-nofit s2y'
                # # Default prior is *volumetric*
                # helper_cip_args += ' --parameter-nofit s1x --parameter-nofit s1y --parameter-nofit s2x  --parameter-nofit s2y --use-precessing '
                # helper_cip_arg_list[1] +=   ' --parameter s1x --parameter s1y --parameter s2x  --parameter s2y --use-precessing '
            
                # Last iterations are with a polar spin, to get spin prior  (as usually requested). Fir is the same as before, but sampling is more efficient
                helper_cip_arg_list[3] +=  '  --use-precessing --parameter-implied xi   --parameter-nofit chi1  --parameter-nofit theta1  --parameter-nofit phi1    --parameter-implied s1x --parameter-implied s1y --parameter-implied s2x --parameter-implied s2y '
        
                # Change convergence threshold at late times
                #            helper_cip_arg_list[2].replace( " --lnL-offset " + str(lnLoffset_early), " --lnL-offset " + str(lnLoffset_late)
                helper_cip_arg_list[3]=helper_cip_arg_list[3].replace( " --lnL-offset " + str(lnLoffset_early), " --lnL-offset " + str(lnLoffset_late))

            n_its = list(map(lambda x: float(x.split()[0]), helper_cip_arg_list))
            puff_max_it= n_its[0] + n_its[1] # puff for first 2 types, to insure good coverage in narrow-q cases
            try:
                if event_dict["m2"]/event_dict["m1"] < 0.4: # High q, do even through the full aligned spin model case
                    puff_max_it += n_its[2]
            except:
                print("No mass information, can't add extra stages")

            # G case: rewrite first case
            if opts.use_gauss_early:
                helper_cip_arg_list[0] = 'G2 --fit-method quadratic --parameter mc --parameter eta --parameter xi --n-output-samples 5000 --sigma-cut 0.9 --lnL-cut {} '.format(np.max([0.2*lnLmax_true,15]))


    if not(opts.use_gauss_early) and (('quadratic' in fit_method) or ('polynomial' in fit_method)):
        helper_cip_arg_list[0] += " --lnL-offset " + str(lnL_start)
        if not(opts.assume_nospin) and not( opts.propose_converge_last_stage):  # don't add more iterations in the zero-spin test cases, or if we are iterating to convergence
            helper_cip_arg_list += helper_cip_arg_list[-1]  # add another set of iterations : these are super fast, and we want to get narrow
        n_levels = len(helper_cip_arg_list)
        for indx in np.arange(1,n_levels):  # do NOT constrain the first CIP, as it has so few points!
            helper_cip_arg_list[indx] += " --lnL-offset " + str( lnL_start*(1.- 1.*indx/(n_levels-1.))  + lnL_end*indx/(n_levels-1.) )

    if opts.assume_matter and not(opts.internal_tabular_eos_file) and not(opts.assume_matter_eos):
        helper_puff_args += " --parameter LambdaTilde  --downselect-parameter s1z --downselect-parameter-range [-0.9,0.9] --downselect-parameter s2z --downselect-parameter-range [-0.9,0.9]  "  # Probably should also aggressively force sampling of low-lambda region
        helper_cip_args += " --input-tides --parameter-implied LambdaTilde  --parameter-nofit lambda2 " # For early fitting, just fit LambdaTilde
        if not(opts.assume_matter_but_primary_bh):
            helper_cip_args+= " --parameter-nofit lambda1 "
        else:
            helper_puff_args = helper_puff_args.replace(" --parameter LambdaTilde ", " --parameter lambda2 ")  # if primary a BH, only varying lambda2
        # Add LambdaTilde on top of the aligned spin runs
        for indx in np.arange(len(helper_cip_arg_list)):
            helper_cip_arg_list[indx]+= " --input-tides --parameter-implied LambdaTilde  --parameter-nofit lambda2 " 
            if not(opts.assume_matter_but_primary_bh):
                helper_cip_arg_list[indx] += " --parameter-nofit lambda1 "
        # add --prior-lambda-linear to first iteration, to sample better at low lambda
        if not (opts.assume_matter_conservatively):
            helper_cip_arg_list[0] += " --prior-lambda-linear "
        # Remove LambdaTilde from *first* batch of iterations .. too painful ? NO, otherwise we wander off into wilderness
#        helper_cip_arg_list[0] = helper_cip_arg_list[0].replace('--parameter-implied LambdaTilde','')
        # Add one line with deltaLambdaTilde
        helper_cip_arg_list.append(helper_cip_arg_list[-1]) 
        helper_cip_arg_list[-1] +=  " --parameter-implied DeltaLambdaTilde "

        n_its = map(lambda x: float(x.split()[0]), helper_cip_arg_list)
        puff_max_it= np.sum(n_its) # puff all the way to the end
    elif opts.internal_tabular_eos_file:
        helper_cip_args += " --tabular-eos-file {} ".format(opts.internal_tabular_eos_file)
        helper_ile_args +=  " --export-eos-index "

        # Askold: remove --parameter-implied LambdaTilde and DeltaLambdaTilde for tabular EOS, add --input-tides and --input-eos-index
        helper_cip_args += " --input-tides --input-eos-index"  
        for indx in np.arange(len(helper_cip_arg_list)):
            helper_cip_arg_list[indx] += " --input-tides --input-eos-index --tabular-eos-file {} ".format(opts.internal_tabular_eos_file)
        # helper_cip_arg_list[-1] += "  --parameter-implied DeltaLambdaTilde "

    elif opts.assume_matter_eos:
        helper_cip_args += "  --input-tides --using-eos {} ".format(opts.assume_matter_eos)
        helper_cip_args+= " --parameter-implied LambdaTilde "
        if opts.assume_matter_but_primary_bh:
            helper_cip_args += " --assume-eos-but-primary-bh "
        for indx in np.arange(len(helper_cip_arg_list)):
            helper_cip_arg_list[indx] += " --input-tides  --parameter-implied LambdaTilde --using-eos {} ".format(opts.assume_matter_eos)
            if opts.assume_matter_but_primary_bh:
                helper_cip_arg_list[indx] += " --assume-eos-but-primary-bh "
        helper_cip_arg_list[-1] += "  --parameter-implied DeltaLambdaTilde "
# lnL-offset was already enforced
#    if opts.internal_fit_strategy_enforces_cut:
#        for indx in np.arange(len(helper_cip_arg_list))[1:]:
#            helper_cip_arg_list[indx] += " --lnL-offset 20 "  # enforce lnL cutoff past the first iteration. Focuses fit on high-likelihood points as in O1/O2


# editing ILE args based on strategy above, so only writing now
with open("helper_ile_args.txt",'w') as f:
    f.write(helper_ile_args)
if not opts.lowlatency_propose_approximant:
    print(" helper_ile_args.txt  does *not* include --d-max, --approximant, --l-max ")

with open("helper_cip_args.txt",'w') as f:
    f.write(helper_cip_args)


if opts.propose_flat_strategy:
    # All iterations of CIP use the same as the last
    instructions_cip = map(lambda x: x.rstrip().split(' '), helper_cip_arg_list)#np.loadtxt("helper_cip_arg_list.txt", dtype=str)
    n_iterations =0
    lines  = []
    for indx in np.arange(len(instructions_cip)):
        n_iterations += int(instructions_cip[indx][0])
        lines.append(' '.join(instructions_cip[indx][1:]))   # merge back together
    helper_cip_arg_list  = [str(n_iterations) + " " + lines[-1]]  # overwrite with new setup


if opts.propose_converge_last_stage:
    helper_cip_last_it = '1 ' +  ' '.join(helper_cip_arg_list[-1].split()[1:])
    lastline = helper_cip_arg_list[-1].lstrip()
    lastline_split = lastline.split(' ')
    lastline_split[0] = 'Z'
    helper_cip_arg_list[-1]  = ' '.join(lastline_split)

    if opts.last_iteration_extrinsic:
        # create a new item, which is like the last one, except ... we will assume we have more workers, and just one iteration
        # NOTE: assume util_RIFT_pseudo_pipe will handle setting n-eff and workers correctly for that iteration, since we can't control it here.
        helper_cip_arg_list += [helper_cip_last_it]


with open("helper_cip_arg_list.txt",'w+') as f:
    f.write("\n".join(helper_cip_arg_list))


# Impose test in last phase only
#   - for convenience, transform last Z 
#   - note Z will override iteration thresholding anyways
#if opts.propose_converge_last_stage:
for indx in np.arange(len(helper_cip_arg_list)):
        firstword = helper_cip_arg_list[indx].split(' ')[0]
        print(helper_cip_arg_list[indx],firstword)
        if firstword == 'Z':
            helper_cip_arg_list[indx]  = '1 ' +  ' '.join(helper_cip_arg_list[indx].split(' ')[1:])
        if 'G' == firstword[0]:
            print(helper_cip_arg_list[indx][1:])
            helper_cip_arg_list[indx] = helper_cip_arg_list[indx][1:]
n_its = list(map(lambda x: float(x.split()[0]), helper_cip_arg_list))
n_its_to_not_test = np.sum(n_its) - n_its[-1]
helper_test_args += " --iteration-threshold {} ".format(int(n_its_to_not_test))
helper_test_args += " --threshold {} ".format(opts.internal_test_convergence_threshold)

with open("helper_test_args.txt",'w+') as f:
    f.write(helper_test_args)


if opts.assume_matter:
    with open("helper_convert_args.txt",'w+') as f:
        f.write(" --export-tides ")
else:
    with open("helper_convert_args.txt",'w+') as f:
        f.write(" --fref {} ".format(opts.fmin_template))   # needed to insure correct precession angles derived from internal conversion
# Askold: add --export-eos for tabular EOS file
if opts.assume_matter and opts.internal_tabular_eos_file:
    with open("helper_convert_args.txt", 'a') as f:
        f.write(" --export-eos ")
        
if opts.assume_eccentric:
    with open("helper_convert_args.txt",'a') as f:
        f.write(" --export-eccentricity ")

if opts.use_meanPerAno:
    with open("helper_convert_args.txt",'a') as f:
        f.write(" --export-meanPerAno ")

if opts.propose_fit_strategy:
    with open("helper_puff_max_it.txt",'w') as f:
        f.write(str(puff_max_it))

if opts.propose_fit_strategy:
    with open("helper_puff_factor.txt",'w') as f:
        f.write(str(puff_factor))

if opts.propose_fit_strategy:
    helper_puff_args += " --downselect-parameter eta --downselect-parameter-range ["+str(eta_min) +","+str(eta_max)+"]"
    helper_puff_args += " --puff-factor " + str(puff_factor)
    force_away_val=0.05
    if mc_center < 3:
        force_away_val = 0.01
    if opts.assume_nospin:
        force_away_val=0.01  # lower dimension, need to avoid ripples on boundary
    if not(opts.internal_use_amr):
        helper_puff_args += " --force-away " + str(force_away_val)  # prevent duplicate points. Don't do this for AMR, since they are already quite sparse
    with open("helper_puff_args.txt",'w') as f:
        f.write(helper_puff_args)

if opts.use_osg:
    # Write log of extra files to transfer
    #   - psd files
    #   - cache file (local.cache), IF using cvmfs frames
    if opts.use_cvmfs_frames:
        transfer_files.append(opts.working_directory+"/local.cache")
    with open("helper_transfer_files.txt","w") as f:
        for name in transfer_files:
            f.write(name+"\n")
        
