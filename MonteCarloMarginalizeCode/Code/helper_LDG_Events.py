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

import os, sys
import numpy as np
import argparse

import lal
import lalsimutils
import lalsimulation as lalsim

from glue.ligolw import lsctables, table, utils
from glue.lal import CacheEntry


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
def query_available_ifos_viadq(ifos_all,data_start,data_end):
    ifos_out = []
    from gwpy.segments import DataQualityFlag
    ifos_out = []
    for ifo in ifos_all:
        segs = None
        try:
            if ifo in ["H1","L1"]:
                segs = DataQualityFlag.query(ifo+":LDS-SCIENCE:1",data_start,data_end)
            if ifo in ["V1"]:
                segs = DataQualityFlag.query(ifo+":ITF_SCIENCEMODE:1",data_start,data_end)
            # If we reach this point, it hasn't crashed, so
            ifos_out.append(ifo)
        except:
            True
    return ifos_out

def ldg_datafind(ifo_base, types, server, data_start,data_end,datafind_exe='gw_data_find', retrieve=False,machine_with_files="ldas-pcdev1.ligo.caltech.edu"):
    fname_out_raw = ifo_base[0]+"_raw.cache"
    fname_out = ifo_base[0]+"_local.cache"
    print [ifo_base, types, server, data_start, data_end]
    cmd = datafind_exe + ' -u file --gaps -o ' + ifo_base[0] + ' -t ' + types + ' --server ' + server + ' -s ' + str(data_start) + ' -e ' + str(data_end) + " > " +fname_out_raw
    os.system(cmd)

    if not retrieve:
        # If we are not retrieving, we are on a cluster, and we need to load and convert this information
        # see util_CacheFileConvert.py  (here)
        with open(fname_out_raw,'r') as f:
            lines = f.readlines()
            lines = map(lambda x: str(CacheEntry.from_T050017(x)), lines)
        with open(fname_out,'w') as f:
            for line in lines:
                f.write(line)

    else:
        print " Trying to transfer files necessary from ", machine_with_files
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
        os.system("find frames -name '*.gwf' | lalapps_path2cache > local.cache")
    else:
        os.system("cat *_local.cache > local.cache")
    return True

def ldg_make_psd(ifo, channel_name,psd_start_time,psd_end_time,srate=4096,use_gwpy=False, force_regenerate=False,working_directory="."):
    psd_fname = ifo + "-psd.xml.gz"
    if (not force_regenerate) and os.path.isfile(working_directory+"/"+psd_fname):
        print " File exists : ", psd_fname
        return True
    cmd = "gstlal_reference_psd --verbose --channel-name " + ifo + "=" + channel_name + " --gps-start-time " + str(psd_start_time) + " --gps-end-time " + str(psd_end_time) + " --write-psd " + psd_fname + " --data-source -frames --frame-cache local.cache --srate " + str(srate)
    print cmd
    os.system(cmd)
    return True


observing_run_time ={}
observing_run_time["O1"] = [1126051217,1137254417] # https://www.gw-openscience.org/O1/
observing_run_time["O2"] = [1164556817,1187733618] # https://www.gw-openscience.org/O2/
observing_run_time["O3"] = [1230000000,1430000000] # Completely made up boundaries, for now
def get_observing_run(t):
    for run in observing_run_time:
        if  t > observing_run_time[run][0] and t < observing_run_time[run][1]:
            return run
    print " No run available for time ", t, " in ", observing_run_time
    return None


parser = argparse.ArgumentParser()
parser.add_argument("--gracedb-id",default=None,type=str)
parser.add_argument("--force-data-lookup",action='store_true',help='Use this flag if you want to use real data.')
parser.add_argument("--use-legacy-gracedb",action='store_true')
parser.add_argument("--event-time",type=float,default=None)
parser.add_argument("--sim-xml",default=None)
parser.add_argument("--event",type=int,default=None)
parser.add_argument("--check-ifo-availability",action='store_true',help="if true, attempt to use frame availability or DQ information to choose ")
parser.add_argument("--observing-run",default=None,help="Use the observing run settings to choose defaults for channel names, etc. Not yet implemented using lookup from event time")
parser.add_argument("--calibration-version",default=None,help="Calibration version to be used.")
parser.add_argument("--playground-data",default=None,help="Playground data. Modifies channel names used.")
parser.add_argument("--datafind-server",default=None,help="LIGO_DATAFIND_SERVER (will override environment variable, which is used as default)")
parser.add_argument("--fmin",default=None,type=float,help="Minimum frequency for integration. Used to estimate signal duration")
parser.add_argument("--fmin-template",default=20,type=float,help="Minimum frequency for template. Used to estimate signal duration. If fmin not specified, also the minimum frequency for integration")
parser.add_argument("--fmax",default=None,type=float,help="fmax. Use this ONLY if you want to override the default settings, which are set based on the PSD used")
parser.add_argument("--data-start-time",default=None)
parser.add_argument("--data-end-time",default=None,help="If both data-start-time and data-end-time are provided, this interval will be used.")
parser.add_argument("--data-LI-seglen",default=None,type=float,help="If provided, use a buffer this long, placing the signal 2s after this, and try to use 0.4s tukey windowing on each side, to be consistent with LI.  ")
#parser.add_argument("--enforce-q-min",default=None,type=float,help='float.  If provided ,the grid will go down to this mass ratio. SEGMENT LENGTH WILL BE ADJUSTED')
parser.add_argument("--working-directory",default=".")
parser.add_argument("--datafind-exe",default="gw_data_find")
parser.add_argument("--gracedb-exe",default="gracedb")
parser.add_argument("--fake-data",action='store_true',help="If this argument is present, the channel names are overridden to FAKE_STRAIN")
parser.add_argument("--cache",type=str,default=None,help="If this argument is present, the various routines will use the frame files in this cache. The user is responsible for setting this up")
parser.add_argument("--psd-file", action="append", help="instrument=psd-file, e.g. H1=H1_PSD.xml.gz. Can be given multiple times for different instruments.  Required if using --fake-data option")
parser.add_argument("--use-online-psd",action='store_true',help='Use PSD from gracedb, if available')
parser.add_argument("--assume-matter",action='store_true',help="If present, the code will add options necessary to manage tidal arguments. The proposed fit strategy and initial grid will allow for matter")
parser.add_argument("--assume-nospin",action='store_true',help="If present, the code will not add options to manage precessing spins (the default is aligned spin)")
parser.add_argument("--assume-precessing-spin",action='store_true',help="If present, the code will add options to manage precessing spins (the default is aligned spin)")
parser.add_argument("--propose-ile-convergence-options",action='store_true',help="If present, the code will try to adjust the adaptation options, Nmax, etc based on experience")
parser.add_argument("--lowlatency-propose-approximant",action='store_true', help="If present, based on the object masses, propose an approximant. Typically TaylorF2 for mc < 6, and SEOBNRv4_ROM for mc > 6.")
parser.add_argument("--online", action='store_true', help="Use online settings")
parser.add_argument("--propose-initial-grid",action='store_true',help="If present, the code will either write an initial grid file or (optionally) add arguments to the workflow so the grid is created by the workflow.  The proposed grid is designed for ground-based LIGO/Virgo/Kagra-scale instruments")
#parser.add_argument("--propose-initial-grid-includes-search-error",action='store_true',help="Searches have paraemter offsets, but injections have known parameters.  You need a wider grid if you are running from a search grid, since they are usually substantiallyoffset from the maximumlikelihood ")
parser.add_argument("--propose-fit-strategy",action='store_true',help="If present, the code will propose a fit strategy (i.e., cip-args or cip-args-list).  The strategy will take into account the mass scale, presence/absence of matter, and the spin of the component objects.  If --lowlatency-propose-approximant is active, the code will use a strategy suited to low latency (i.e., low cost, compatible with search PSDs, etc)")
parser.add_argument("--no-propose-limits",action='store_true',help="If a fit strategy is proposed, the default strategy will propose limits on mc and eta.  This option disables those limits, so the user can specify their own" )
parser.add_argument("--hint-snr",default=None,type=float,help="If provided, use as a hint for the signal SNR when choosing ILE and CIP options (e.g., to avoid overflow or underflow).  Mainly important for synthetic sources with very high SNR")
parser.add_argument("--use-quadratic-early",action='store_true',help="If provided, use a quadratic fit in the early iterations'")
parser.add_argument("--verbose",action='store_true')
opts=  parser.parse_args()


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


typical_bns_range_Mpc = {}
typical_bns_range_Mpc["O1"] = 100 
typical_bns_range_Mpc["O2"] = 100 
typical_bns_range_Mpc["O3"] = 130

## O1 definitions
cal_versions = {"C00", "C01", "C02"}
for cal in cal_versions:
    for ifo in "H1", "L1":
        if cal is "C00":
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
        if cal is "C00":
            standard_channel_names["O2"][(cal,ifo)] = "GDS-CALIB_STRAIN" # _"+cal
        elif cal is "C02":
            standard_channel_names["O2"][(cal,ifo)] = "DCH-CLEAN_STRAIN_C02"
            data_types["O2"][(cal,ifo)] = ifo+"_CLEANED_HOFT_C02"
        else:
            standard_channel_names["O2"][(cal,ifo)] = "DCS-CALIB_STRAIN_"+cal 
#Virgo
data_types["O2"][("C00", "V1")] = "V1Online"
data_types["O2"][("C02", "V1")] = "V1O2Repro2A"
standard_channel_names["O2"][("C02", "V1")] = "Hrec_hoft_V1O2Repro2A_16384Hz"
if opts.verbose:
    print standard_channel_names["O2"]

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
cal_versions = {"C00"}
for cal in cal_versions:
    for ifo in "H1", "L1":
        data_types["O3"][(cal,ifo)] = ifo+"_HOFT_" + cal
        if opts.online:
            data_types["O3"][(cal,ifo)] = ifo+"_llhoft"
        if cal is "C00":
            standard_channel_names["O3"][(cal,ifo)] = "GDS-CALIB_STRAIN_CLEAN" 
            if opts.online:
                standard_channel_names["O3"][(cal,ifo)] = "GDS-CALIB_STRAIN" # Do not assume cleaning is available in low latency
data_types["O3"][("C00", "V1")] = "V1Online"
standard_channel_names["O3"][("C00", "V1")] = "Hrec_hoft_16384Hz"
if opts.online:
    data_types["O3"][("C00", "V1")] = "V1_llhoft"
    standard_channel_names["O3"][("C00", "V1")] = "Hrec_hoft_16384Hz"

if opts.verbose:
    print standard_channel_names["O3"]




datafind_server = None
try:
   datafind_server = os.environ['LIGO_DATAFIND_SERVER']
   print " LIGO_DATAFIND_SERVER ", datafind_server
except:
  print " No LIGO_DATAFIND_SERVER "
if opts.datafind_server:
    datafind_server = opts.datafind_server
if (datafind_server is None) and not (opts.fake_data):
    print " FAIL: No data !"

###
### Import event and PSD: Manual branch
###

use_gracedb_event = False
if not(opts.gracedb_id is None):
    use_gracedb_event = True
elif opts.sim_xml:  # right now, configured to do synthetic data only...should be able to mix/match
    print "====Loading injection XML:", opts.sim_xml, opts.event, " ======="
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

# PSDs must be provided by hand, IF this is done by this code!
ifo_list=[]
if not (opts.psd_file is None):
    for inst, psdf in map(lambda c: c.split("="), opts.psd_file):
            psd_names[inst] = psdf
            ifo_list.append(inst)
    event_dict["IFOs"] = ifo_list

###
### Import event and PSD: GraceDB branch
###

if use_gracedb_event:
    cmd_event = gracedb_exe + download_request + opts.gracedb_id + " event.log"
    os.system(cmd_event)
    # Parse gracedb. Note very annoying heterogeneity in event.log files
    with open("event.log",'r') as f:
        lines = f.readlines()
        for  line  in lines:
            line = line.split(':')
            param = line[0]
            if opts.verbose:
                print " Parsing line ", line
            if param in ['MChirp', 'MTot', "SNR"]:
                event_dict[ line[0]]  = float(line[1])
            elif 'ime' in param: # event time
                event_dict["tref"] = float(line[1])
            elif param == 'IFOs':
                line[1] = line[1].replace(' ','').rstrip()
                ifo_list = line[1].split(",")
                event_dict["IFOs"] = ifo_list

    # Read in event parameters. Use masses as quick estimate
    cmd_event = gracedb_exe + download_request + opts.gracedb_id + " coinc.xml"
    os.system(cmd_event)
    samples = table.get_table(utils.load_filename("coinc.xml",contenthandler=lalsimutils.cthdler), lsctables.SnglInspiralTable.tableName)
    event_duration=4  # default
    for row in samples:
        m1 = row.mass1
        m2 = row.mass2
        try:
            event_duration = row.event_duration # may not exist
        except:
            print " event_duration field not in XML "
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
    fmax = 1000.
    cmd_event = gracedb_exe + download_request + opts.gracedb_id + " psd.xml.gz"
    os.system(cmd_event)
    if opts.use_online_psd:
        cmd = "helper_OnlinePSDCleanup.py --psd-file psd.xml.gz "
        # Convert PSD to a useful format
        for ifo in event_dict["IFOs"]:
            psd_names[ifo] = opts.working_directory+"/"+ifo+"-psd.xml.gz"
            cmd += " --ifo " + ifo
        os.system(cmd)


if not (opts.hint_snr is None) and not ("SNR" in event_dict.keys()):
    event_dict["SNR"] = np.max([opts.hint_snr,6])  # hinting a low SNR isn't helpful

print " Event analysis ", event_dict
print " == candidate event parameters (as passed to helper) == "
event_dict["P"].print_params()

# Use event GPS time to set observing run, if not provided.  Insures automated operation with a trivial settings file does good things.
if (opts.observing_run is None) and not opts.fake_data:
    tref = event_dict["tref"]
    opts.observing_run = get_observing_run(tref)
    if opts.calibration_version is None:
        # This should be a dictionary lookup.
        if opts.observing_run is "O2":
            opts.calibration_version = "C02"
        if opts.observing_run is "O1":
            opts.calibration_version = "C02"
        if opts.observing_run is "O3":
            opts.calibration_version = "C00"   # for now! change as needed



###
### General logic 
###

snr_fac = 1
if "SNR" in event_dict.keys():
    lnLmax_true = event_dict['SNR']**2 / 2.
    lnLoffset_early = 0.8*lnLmax_true  # default value early on : should be good enough
    snr_fac = np.max([snr_fac, event_dict["SNR"]/15.])  # scale down regions accordingly
else:
    lnLoffset_early = 500  # a fiducial value, good enough for a wide range of SNRs 

# Estimate signal duration
t_event = event_dict["tref"]
P=event_dict["P"]
#lalsimutils.ChooseWaveformParams()
#P.m1= event_dict['m1']*lal.MSUN_SI;  P.m2= event_dict['m2']*lal.MSUN_SI; 
t_duration  = np.max([ event_dict["epoch"], lalsimutils.estimateWaveformDuration(P)])
t_before = np.max([4,t_duration])*1.1+8+2  # buffer for inverse spectrum truncation
data_start_time_orig = data_start_time = t_event - int(t_before)
data_end_time = t_event + int(t_before) # for inverse spectrum truncation. Overkill

# Estimate data needed for PSD
psd_data_start_time = t_event - 2048 - t_before
psd_data_end_time = t_event - 1024 - t_before
# set the start time to be the time needed for the PSD, if we are generating a PSD
if (opts.psd_file is None) and  use_gracedb_event and not opts.use_online_psd:
    data_start_time = psd_data_start_time

# reset IFO list if needed. Do NOT do with online_psd
if opts.check_ifo_availability and not opts.use_online_psd:  # online PSD only available for some IFOs
        event_dict["IFOs"] = query_available_ifos_viadq(["H1","L1","V1"],data_start_time_orig,data_end_time)

# define channel names
ifos = event_dict["IFOs"]
channel_names = {}
for ifo in ifos:
    if opts.fake_data:
        channel_names[ifo] = "FAKE-STRAIN"
    else:
        channel_names[ifo] = standard_channel_names[opts.observing_run][(opts.calibration_version,ifo)]

# Set up, perform datafind (if not fake data)
if not (opts.fake_data):
    for ifo in ifos:
        data_type_here = data_types[opts.observing_run][(opts.calibration_version,ifo)]
        ldg_datafind(ifo, data_type_here, datafind_server,int(data_start_time), int(data_end_time), datafind_exe=datafind_exe)
if not opts.cache:  # don't make a cache file if we have one!
    real_data = not(opts.gracedb_id is None)
    real_data = real_data or  opts.check_ifo_availability
    real_data = real_data or opts.force_data_lookup
    ldg_make_cache(retrieve=real_data) # we are using the ifo_local.cache files
    opts.cache = "local.cache" # standard filename populated

# If needed, build PSDs
if (opts.psd_file is None) and not opts.use_online_psd:
    print " PSD construction "
    for ifo in event_dict["IFOs"]:
        print " Building PSD  for ", ifo
        try:
            ldg_make_psd(ifo, channel_names[ifo], psd_start_time, psd_end_time, working_directory=opts.working_directory)
            psd_names[ifo] = opts.working_directory+"/" + ifo + "-psd.xml.gz"
        except:
            print "  ... PSD generation failed! "
            sys.exit(1)

# Estimate mc range, eta range
#   - UPDATE: need to add scaling with SNR too

mc_center = event_dict["MChirp"]
v_PN_param = (np.pi* mc_center*opts.fmin*lalsimutils.MsunInSec)**(1./3.)  # 'v' parameter
v_PN_param = np.min([v_PN_param,1])
# Estimate width. Note this must *also* account for search error (if we are using search triggers), so it is double-counted and super-wide
# Note I have TWO factors to set: the absolute limits on the CIP, and the grid spacing (which needs to be narrower) for PE placement
fac_search_correct=1.
if opts.gracedb_id: #opts.propose_initial_grid_includes_search_error:
    fac_search_correct = 2
ln_mc_error_pseudo_fisher = 1.3*np.array([1,fac_search_correct])*0.3*(v_PN_param/0.2)**(7.)/snr_fac  # this ignores range due to redshift / distance, based on a low-order estimate
if ln_mc_error_pseudo_fisher[0] >1:
    ln_mc_errors_pseudo_fisher =np.array([0.8,0.8])   # stabilize
mc_min, mc_min_tight = np.exp( - ln_mc_error_pseudo_fisher)*mc_center  # conservative !  Should depend on mc, use a Fisher formula. Does not scale to BNS
mc_max, mc_max_tight=np.exp( ln_mc_error_pseudo_fisher)*mc_center   # conservative ! 

# eta <->  delta
#   Start out with a grid out to eta = 0.1 ( *tight, passed to the grid code)
#   Do more than this with puffball and other tools
#   Use other tools to set CIP limits
eta_max = 0.249999
eta_val =P.extract_param('eta')
tune_grid = False
eta_max_tight = eta_max
eta_min_tight  = eta_min = 0.1  # default for now, will fix this later
tmp1,tmp2 = lalsimutils.m1m2(1,eta_min)
delta_max_tight= delta_max =(tmp1-tmp2)/(tmp1+tmp2)  # about 0.8
delta_min_tight = delta_min =1e-4  # Some approximants like SEOBNRv3 can hard fail if m1=m2
if mc_center < 2.6 and opts.propose_initial_grid:  # BNS scale, need to constraint eta to satisfy mc > 1
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
elif mc_center < 18 and P.extract_param('q') < 0.6 and opts.propose_initial_grid:  # BH-NS scale, want to make sure we do a decent job at covering high-mass-ratio end
   import scipy.optimize
   # solution to equation with m2 -> 1 is  1 == mc delta 2^(1/5)/(1-delta^2)^(3/5), which is annoying to solve
   def crit_m2(delta):
       eta_val = 0.25*(1-delta*delta)
       return 0.5*mc_center*(eta_val**(-3./5.))*(1-delta) - 3.
   res = scipy.optimize.brentq(crit_m2, 0.001,0.999) # critical value of delta: largest possible for this mc value
   delta_max =np.min([1.1*res,0.99])
   eta_min = 0.25*(1-delta_max*delta_max)
# High mass ratio configuration.  PROTOTYPE, NEEDS LOTS OF WORK FOR BH-NS, should restore use of  fisher grid!
elif opts.propose_initial_grid and eta_val < 0.1: # this will override the previous work
    eta_min =0.25*eta_val
    eta_max= np.min([0.249999,4*eta_val])
    delta_max = np.sqrt(1. - 4*eta_min)
    delta_min = np.sqrt(1. - 4*eta_max)
    tune_grid = True

chieff_center = P.extract_param('xi')
chieff_min = np.max([chieff_center -0.3,-1])/snr_fac
chieff_max = np.max([chieff_center +0.3,1])/snr_fac
if chieff_min >0 and use_gracedb_event:
    chieff_min = -0.1   # make sure to cover spin zero, most BBH have zero spin and missing zero is usually an accident of the search recovered params

mc_range_str = "  ["+str(mc_min_tight)+","+str(mc_max_tight)+"]"  # Use a tight placement grid for CIP
mc_range_str_cip = " --mc-range ["+str(mc_min)+","+str(mc_max)+"]"
eta_range_str = "  ["+str(eta_min_tight) +","+str(eta_max_tight)+"]"  # default will include  1, as we work with BBHs
eta_range_str_cip = " --eta-range ["+str(eta_min) +","+str(eta_max)+"]"  # default will include  1, as we work with BBHs

###
### Write arguments
###
helper_ile_args ="X "
helper_test_args="X "
helper_cip_args = "X "
helper_cip_arg_list = []

helper_test_args += " --always-succeed --method lame  --parameter mc"

helper_ile_args += " --save-P 0.1 "   # truncate internal data structures (should do better memory management/avoid need for this if --save-samples is not on)
if "SNR" in event_dict.keys():
    snr_here = event_dict["SNR"]
    if snr_here > 25:
        lnL_expected = snr_here**2 /2. - 10  # 10 is rule of thumb, depends on distance prior
        helper_ile_args += " --manual-logarithm-offset " + str(lnL_expected)
        helper_cip_args += " --lnL-shift-prevent-overflow " + str(lnL_expected)   # warning: this can have side effects if the shift makes lnL negative, as the default value of the fit is 0 !

helper_ile_args += " --cache " + opts.working_directory+ "/" + opts.cache
helper_ile_args += " --event-time " + str(event_dict["tref"])
for ifo in ifos:
    helper_ile_args += " --channel-name "+ifo+"="+channel_names[ifo]
    helper_ile_args += " --psd-file "+ifo+"="+psd_names[ifo]
    if not (opts.fmin is None):
        helper_ile_args += " --fmin-ifo "+ifo+"="+str(opts.fmin)
helper_ile_args += " --fmax " + str(fmax)
helper_ile_args += " --fmin-template " + str(opts.fmin_template)
helper_ile_args += " --reference-freq " + str(opts.fmin_template)  # in case we are using a code which allows this to be specified
approx_str= "SEOBNRv4"  # default, should not be used.  See also cases where grid is tuned
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
    helper_ile_args +=  " --d-max " + str(int(dmax_guess))

    if (opts.data_LI_seglen is None) and  (opts.data_start_time is None):
        # Also choose --data-start-time, --data-end-time and disable inverse spectrum truncation (use tukey)
        #   ... note that data_start_time was defined BEFORE with the datafind job
        T_window_raw = 1.1/lalsimutils.estimateDeltaF(P)  # includes going to next power of 2, AND a bonus factor of a few
        T_window_raw = np.max([T_window_raw,4])  # can't be less than 4 seconds long
        print " Time window : ", T_window_raw, " based on fmin  = ", P.fmin
        data_start_time = np.max([int(P.tref - T_window_raw -2 )  , data_start_time_orig])  # don't request data we don't have! 
        data_end_time = int(P.tref + 2)
        helper_ile_args += " --data-start-time " + str(data_start_time) + " --data-end-time " + str(data_end_time)  + " --inv-spec-trunc-time 0 --window-shape 0.01"

if not ( (opts.data_start_time is None) and (opts.data_end_time is None)):
    # Manually set the data start and end time.
    T_window = opts.data_end_time - opts.data_start_time
    # Use LI-style tukey windowing
    window_shape = 0.4*2/T_window
    data_start_time =opts.data_start_time
    data_end_time =opts.data_end_time
    helper_ile_args += " --data-start-time " + str(data_start_time) + " --data-end-time " + str(data_end_time)  + " --inv-spec-trunc-time 0 --window-shape " + str(window_shape)
elif opts.data_LI_seglen:
    seglen = opts.data_LI_seglen
    # Use LI-style positioning of trigger relative to 2s before end of buffer
    # Use LI-style tukey windowing
    window_shape = 0.4*2/seglen
    data_end_time = event_dict["tref"]+2
    data_start_time = event_dict["tref"] +2 - seglen
    helper_ile_args += " --data-start-time " + str(data_start_time) + " --data-end-time " + str(data_end_time)  + " --inv-spec-trunc-time 0  --window-shape " + str(window_shape)

if opts.propose_initial_grid:
    # add basic mass parameters
    cmd  = "util_ManualOverlapGrid.py  --fname proposed-grid --skip-overlap --parameter mc --parameter-range   " + mc_range_str + "  --parameter delta_mc --parameter-range '[" + str(delta_min_tight) +"," + str(delta_max_tight) + "]'  "
    # Add standard downselects : do not have m1, m2 be less than 1
    cmd += " --fmin " + str(opts.fmin_template)
    if opts.data_LI_seglen:  
        cmd += " --enforce-duration-bound " + str(opts.data_LI_seglen)
    cmd += "  --downselect-parameter m1 --downselect-parameter-range [1,10000]   --downselect-parameter m2 --downselect-parameter-range [1,10000]  "
    if tune_grid:
        cmd += " --reset-grid-via-match --match-value 0.85 --use-fisher  --use-fisher-resampling --approx  " + approx_str # ow, but useful
    if opts.assume_nospin:
        cmd += " --grid-cartesian-npts 500 " 
    else:
        chieff_range = str([chieff_min,chieff_max]).replace(' ', '')   # assumes all chieff are possible
        if opts.propose_fit_strategy:
            # If we don't have a fit plan, use the NS spin maximum as the default
            if (P.extract_param('mc')/lal.MSUN_SI < 2.6):   # assume a maximum NS mass of 3 Msun
                chi_max = 0.05   # propose a maximum NS spin
                chi_range = str([-chi_max,chi_max]).replace(' ','')
                chieff_range = chi_range  # force to be smaller
                cmd += " --downselect-parameter s1z --downselect-parameter-range " + chi_range + "   --downselect-parameter s2z --downselect-parameter-range " + chi_range 

        cmd += " --parameter chieff_aligned  --parameter-range " + chieff_range+  " --grid-cartesian-npts 4000 "

        if opts.assume_precessing_spin:
            # Handle problems with SEOBNRv3 failing for aligned binaries -- add small amount of misalignment in the initial grid
            cmd += " --parameter s1x --parameter-range [0.01,0.03] "

    if opts.assume_matter:
        # Do the initial grid assuming matter, with tidal parameters set by the AP4 EOS provided by lalsuite
        # We will leverage working off this to find the lambdaTilde dependence
        cmd += " --use-eos AP4 "  

    print " Executing grid command ", cmd
    os.system(cmd)

    if opts.assume_matter:
        # Now perform a puffball in lambda1 and lambda2
        cmd_puff = " util_ParameterPuffball.py --parameter lambda1 --parameter lambda2 --inj-file proposed-grid.xml.gz --inj-file-out proposed-grid_puff_lambda --downselect-parameter lambda1 --downselect-parameter-range [0.1,5000] --downselect-parameter lambda2 --downselect-parameter-range [0.1,5000]"
        os.system(cmd_puff)
        # Now add these two together
        # ideally, ligolw_add will work... except it fails
        P_A = lalsimutils.xml_to_ChooseWaveformParams_array("proposed-grid.xml.gz")
        P_B = lalsimutils.xml_to_ChooseWaveformParams_array("proposed-grid_puff_lambda.xml.gz")
        lalsimutils.ChooseWaveformParams_array_to_xml(P_A+P_B, "proposed-grid.xml.gz")

if opts.propose_ile_convergence_options:
    helper_ile_args += " --time-marginalization  --inclination-cosine-sampler --declination-cosine-sampler   --n-max 2000000 --n-eff 50 "
    # Modify someday to use the SNR to adjust some settings
    # Proposed option will use GPUs
    # Note that number of events to analyze is controlled by a different part of the workflow !
    helper_ile_args += " --vectorized --gpu  --no-adapt-after-first --no-adapt-distance --srate 4096 "

    if snr_fac > 1.5:  # this is a pretty loud signal, so we need to tune the adaptive exponent too!
        helper_ile_args += " --adapt-weight-exponent " + str(0.3/np.power(snr_fac/1.5,2))
    else:
        helper_ile_args += " --adapt-weight-exponent  0.3 "  # typical value

with open("helper_ile_args.txt",'w') as f:
    f.write(helper_ile_args)
if not opts.lowlatency_propose_approximant:
    print " helper_ile_args.txt  does *not* include --d-max, --approximant, --l-max "


if opts.propose_fit_strategy:
    # Strategy: One iteration of low-dimensional, followed by other dimensions of high-dimensional
    print " Fit strategy NOT IMPLEMENTED -- currently just provides basic parameterization options. Need to work in real strategies (e.g., cip-arg-list)"
    helper_cip_args += " --lnL-offset " + str(lnLoffset_early)
    helper_cip_args += ' --cap-points 12000 --no-plots --fit-method gp  --parameter mc --parameter delta_mc '
    if not opts.no_propose_limits:
        helper_cip_args += mc_range_str_cip + eta_range_str_cip

    helper_cip_arg_list_common = str(helper_cip_args)[1:] # drop X
    helper_cip_arg_list = ["3 " + helper_cip_arg_list_common, "4 " +  helper_cip_arg_list_common ]
    if opts.use_quadratic_early:
        helper_cip_arg_list[0] = helper_cip_arg_list[0].replace('fit-method gp', 'fit-method quadratic')

    if not opts.assume_nospin:
        helper_cip_args += ' --parameter-implied xi  --parameter-nofit s1z --parameter-nofit s2z ' # --parameter-implied chiMinus  # keep chiMinus out, until we add flexible tools
        helper_cip_arg_list[0] +=  ' --parameter-implied xi  --parameter-nofit s1z --parameter-nofit s2z ' 
        helper_cip_arg_list[1] += ' --parameter-implied xi  --parameter-implied chiMinus --parameter-nofit s1z --parameter-nofit s2z ' 
        

        if opts.assume_precessing_spin:
            # Use cartesian coordinates for now.  Polar is more flexible
            # Default prior is *volumetric*
            helper_cip_args += ' --parameter-nofit s1x --parameter-nofit s1y --parameter-nofit s2x  --parameter-nofit s2y --use-precessing '
            helper_cip_arg_list[0] +=   ' --parameter-nofit s1x --parameter-nofit s1y --parameter-nofit s2x  --parameter-nofit s2y --use-precessing '
            helper_cip_arg_list[1] +=   ' --parameter s1x --parameter s1y --parameter s2x  --parameter s2y --use-precessing '
    if opts.assume_matter:
        helper_cip_args += " --input-tides --parameter-implied LambdaTilde --parameter-nofit lambda1 --parameter-nofit lambda2 " # For early fitting, just fit LambdaTilde
        # Add LambdaTilde on top of the aligned spin runs
        for indx in np.arange(len(helper_cip_arg_list)):
            helper_cip_arg_list[indx]+= " --input-tides --parameter-implied LambdaTilde --parameter-nofit lambda1 --parameter-nofit lambda2 " 
        # Add one line with deltaLambdaTilde
        helper_cip_arg_list.append(helper_cip_arg_list[-1]) 
        # Make the second to last line include tides
        #    - first iterations add lambdatilde
        #    - second iterations add deltalambda
#        helper_arg_list[-2] +=  " --input-tides --parameter-implied LambdaTilde --parameter-nofit lambda1 --parameter-nofit lambda2 "
        helper_cip_arg_list[-1] +=  " --input-tides --parameter-implied LambdaTilde --parameter-implied LambdaTilde --parameter-nofit lambda1 --parameter-nofit lambda2 "

with open("helper_cip_args.txt",'w') as f:
    f.write(helper_cip_args)

with open("helper_cip_arg_list.txt",'w+') as f:
    f.write("\n".join(helper_cip_arg_list))

with open("helper_test_args.txt",'w+') as f:
    f.write(helper_test_args)


if opts.assume_matter:
    with open("helper_convert_args.txt",'w+') as f:
        f.write(" --export-tides ")
