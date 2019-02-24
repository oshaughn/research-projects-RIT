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


def ldg_datafind(ifo_base, types, server, data_start,data_end,datafind_exe='gw_data_find', retrieve=False,machine_with_files="ldas-pcdev1.ligo.caltech.edu"):
    fname_out_raw = ifo_base[0]+"_raw.cache"
    fname_out = ifo_base[0]+"_local.cache"
    print [ifo_base, types, server, data_start, data_end]
    cmd = 'gw_data_find -u file --gaps -o ' + ifo_base[0] + ' -t ' + types + ' --server ' + server + ' -s ' + str(data_start) + ' -e ' + str(data_end) + " > " +fname_out_raw
    os.system(cmd)

    if not retrieve:
        # If we are not retrieving, we are on a cluster, and we need to load and convert this information
        # see util_CacheFileConvert.py  (here)
        with open(fname_out_raw,'r') as f:
            lines = f.readlines()
            lines = map(lambda x: str(CacheEntry.from_T050017(x)), lines)
        with open(fname_out,'w') as f:
            for line in lines:
                f.write(line+"\n")

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


parser = argparse.ArgumentParser()
parser.add_argument("--gracedb-id",default=None,type=str)
parser.add_argument("--event-time",default=None)
parser.add_argument("--observing-run",default="O2",help="Use the observing run settings to choose defaults for channel names, etc. Not yet implemented using lookup from event time")
parser.add_argument("--calibration-version",default="C02",help="Calibration version to be used.")
parser.add_argument("--datafind-server",default=None,help="LIGO_DATAFIND_SERVER (will override environment variable, which is used as default)")
parser.add_argument("--fmin",default=20,type=float,help="Minimum frequency for integration. Used to estimate signal duration")
parser.add_argument("--fmin-template",default=20,type=float,help="Minimum frequency for template. Used to estimate signal duration")
parser.add_argument("--fmax",default=None,type=float,help="fmax. Use this ONLY if you want to override the default settings, which are set based on the PSD used")
parser.add_argument("--data-start-time",default=None)
parser.add_argument("--data-end-time",default=None,help="If both data-start-time and data-end-time are provided, this interval will be used.")
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
parser.add_argument("--propose-initial-grid",action='store_true',help="If present, the code will either write an initial grid file or (optionally) add arguments to the workflow so the grid is created by the workflow.  The proposed grid is designed for ground-based LIGO/Virgo/Kagra-scale instruments")
parser.add_argument("--propose-fit-strategy",action='store_true',help="If present, the code will propose a fit strategy (i.e., cip-args or cip-args-list).  The strategy will take into account the mass scale, presence/absence of matter, and the spin of the component objects")
opts=  parser.parse_args()

datafind_exe = opts.datafind_exe
gracedb_exe = opts.gracedb_exe

datafind_server = None
try:
   datafind_server = os.environ['LIGO_DATAFIND_SERVER']
except:
  print " No LIGO_DATAFIND_SERVER "
if opts.datafind_server:
    datafind_server = opts.datafind_server
else:
    datafind_server="ldr.ldas.cit:80"
use_gracedb_event = False
if not(opts.gracedb_id is None):
    use_gracedb_event = True


data_types = {}
standard_channel_names = {}

# Initialize O2
data_types["O2"] = {}
standard_channel_names["O2"] = {}
cal_versions = {"C00", "C01", "C02"}
for cal in cal_versions:
    for ifo in "H1", "L1":
        data_types["O2"][(cal,ifo)] = ifo+"_HOFT_" + cal
        if cal is "C00":
            standard_channel_names["O2"][(cal,ifo)] = "GDS-CALIB_STRAIN_"+cal
        else:
            standard_channel_names["O2"][(cal,ifo)] = "DCS-CALIB_STRAIN_"+cal
print standard_channel_names["O2"]


fmax = 1700 # default
psd_names = {}
event_dict = {}


###
### GraceDB branch
###

if True: #use_gracedb_event:
    cmd_event = gracedb_exe + " download " + opts.gracedb_id + " event.log"
    os.system(cmd_event)
    # Parse gracedb. Note very annoying heterogeneity in event.log files
    with open("event.log",'r') as f:
        lines = f.readlines()
        for  line  in lines:
            line = line.split(':')
            param = line[0]
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
    cmd_event = gracedb_exe + " download " + opts.gracedb_id + " coinc.xml"
    os.system(cmd_event)
    samples = table.get_table(utils.load_filename("coinc.xml"), lsctables.SnglInspiralTable.tableName)
    for row in samples:
        m1 = row.mass1
        m2 = row.mass2
        event_duration = row.event_duration
    event_dict["m1"] = row.mass1
    event_dict["m2"] = row.mass2
    event_dict["epoch"]  = event_duration

    # Get PSD
    fmax = 1000.
    cmd_event = gracedb_exe + " download " + opts.gracedb_id + " psd.xml.gz"
    if opts.use_online_psd:
        for ifo in event_dict["IFOs"]:
            psd_names[ifo] = opts.working_directory+"/psd.xml.gz"


###
### General logic 
###

# Estimate signal duration
t_event = event_dict["tref"]
P=lalsimutils.ChooseWaveformParams()
P.m1= event_dict['m1']*lal.MSUN_SI;  P.m2= event_dict['m2']*lal.MSUN_SI; 
t_duration  = np.max([ event_dict["epoch"], lalsimutils.estimateWaveformDuration(P)])
t_before = np.max([4,t_duration])*1.1+8+2  # buffer for inverse spectrum truncation
data_start_time = t_event - int(t_before)
data_end_time = t_event + int(t_before) # for inverse spectrum truncation. Overkill

# Estimate data needed for PSD
psd_data_start_time = t_event - 2048 - t_before
psd_data_end_time = t_event - 1024 - t_before
# set the start time to be the time needed for the PSD, if we are generating a PSD
if use_gracedb_event and not opts.use_online_psd:
    data_start_time = psd_data_start_time

# define channel names
ifos = event_dict["IFOs"]
channel_names = {}
for ifo in ifos:
    if opts.fake_data:
        channel_names[ifo] = "FAKE-STRAIN"
    else:
        channel_names[ifo] = standard_channel_names[opts.observing_run][(opts.calibration_version,ifo)]

# Set up, perform datafind 
for ifo in ifos:
    data_type_here = data_types[opts.observing_run][(opts.calibration_version,ifo)]
    ldg_datafind(ifo, data_type_here, datafind_server,data_start_time, data_end_time, datafind_exe=datafind_exe)
ldg_make_cache()

# If needed, build PSDs
print " PSD construction not yet implemented ... must use GraceDB-provided PSDs"

###
### Write arguments
###
helper_ile_args =""
helper_ile_args += " --cache " + opts.working_directory+ "/local.cache"
for ifo in ifos:
    helper_ile_args += " --channel-name "+ifo+"="+channel_names[ifo]
    helper_ile_args += " --psd-file "+ifo+"="+psd_names[ifo]
helper_ile_args += " --fmax " + str(fmax)
helper_ile_args += " --fmin-template " + str(opts.fmin_template)
helper_ile_args += " --fmin " + str(opts.fmin)

with open("helper_ile_args.txt",'w') as f:
    f.write(helper_ile_args)



helper_cip_args = ""

with open("helper_cip_args.txt",'w') as f:
    f.write(helper_ile_args)
