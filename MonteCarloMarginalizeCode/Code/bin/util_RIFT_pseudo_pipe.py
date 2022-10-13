#! /usr/bin/env python
#
# GOAL
#
#
# HISTORY
#   - Based on testing_archival_and_pseudo_online/scripts/setup_analysis_gracedb_event.py  in  richard-oshaughnessy/rapid_pe_nr_review_o3.git
#
# EXAMPLES
#    Here, <EXE> refers to the name given to this code
#  - Reproduce argument sequence of lalinference_pipe
#       <EXE>  --use-ini `pwd`/test.ini --use-coinc `pwd`/coinc.xml --use-rundir `pwd`/test --use-online-psd-file `pwd`/psd.xml.gz
#  - Run on events with full automation 
#       <EXE> --gracedb-id G329483 --approx NRHybSur3dq8 --l-max 4


import numpy as np
import argparse
import os
import sys
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
import configparser as ConfigParser

if ( 'RIFT_LOWLATENCY'  in os.environ):
    assume_lowlatency = True
else:
    assume_lowlatency=False

import shutil

# Default setup assumes the underlying sampling will be *cartesian* 
# for a precessing binary.  Change as appropriate if the underlying helper changes to be more sensible.
prior_args_lookup={}
prior_args_lookup["default"] =""
prior_args_lookup["volumetric"] =""
prior_args_lookup["uniform_mag_prec"] =" --pseudo-uniform-magnitude-prior "
prior_args_lookup["uniform_aligned"] =""
prior_args_lookup["zprior_aligned"] =" --aligned-prior alignedspin-zprior"

typical_bns_range_Mpc = {}
typical_bns_range_Mpc["O1"] = 100 
typical_bns_range_Mpc["O2"] = 100 
typical_bns_range_Mpc["O3"] = 130
observing_run_time ={}
observing_run_time["O1"] = [1126051217,1137254417] # https://www.gw-openscience.org/O1/
observing_run_time["O2"] = [1164556817,1187733618] # https://www.gw-openscience.org/O2/
observing_run_time["O3"] = [1230000000,1430000000] # Completely made up boundaries, for now
def get_observing_run(t):
    for run in observing_run_time:
        if  t > observing_run_time[run][0] and t < observing_run_time[run][1]:
            return run
    print( " No run available for time ", t, " in ", observing_run_time)
    return None

def unsafe_config_get(config,args,verbose=False):
    if verbose:
        print( " Retrieving ", args)
        print( " Found ",eval(config.get(*args)))
    return eval( config.get(*args))


def format_gps_time(tval):
    if isinstance(tval,str):
        return tval
    if tval is None:
        return "0"
    str_out = "{:.5f}".format(float(tval))
    return str_out

def retrieve_event_from_coinc(fname_coinc):
    from ligo.lw import lsctables, table, utils
    from RIFT import lalsimutils
    event_dict ={}
    samples = lsctables.SnglInspiralTable.get_table(utils.load_filename(fname_coinc,contenthandler=lalsimutils.cthdler))
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
            print( " event_duration field not in XML ")
    event_dict["m1"] = row.mass1
    event_dict["m2"] = row.mass2
    event_dict["s1z"] = row.spin1z
    event_dict["s2z"] = row.spin2z
    event_dict["IFOs"] = list(set(ifo_list))
    max_snr_idx = snr_list.index(max(snr_list))
    event_dict['SNR'] = snr_list[max_snr_idx]
    event_dict['tref'] = tref_list[max_snr_idx]
    return event_dict


def unsafe_parse_arg_string(my_argstr,match):
    arglist  = [x for x in my_argstr.split("--") if len(x)>0]
    for x in arglist:
        if match in x:
            return x
    return None
        


parser = argparse.ArgumentParser()
parser.add_argument("--use-production-defaults",action='store_true',help="Use production defaults. Intended for use with tools like asimov or by nonexperts who just want something to run on a real event.  Will require manual setting of other arguments!")
parser.add_argument("--use-subdags",action='store_true',help="Use CEPP_Alternate instead of CEPP_BasicIteration")
parser.add_argument("--use-ini",default=None,type=str,help="Pass ini file for parsing. Intended to reproduce lalinference_pipe functionality. Overrides most other arguments. Full path recommended")
parser.add_argument("--use-rundir",default=None,type=str,help="Intended to reproduce lalinference_pipe functionality. Must be absolute path.")
parser.add_argument("--use-online-psd-file",default=None,type=str,help="Provides specific online PSD file, so no downloads are needed")
parser.add_argument("--use-coinc",default=None,type=str,help="Intended to reproduce lalinference_pipe functionality")
parser.add_argument("--manual-ifo-list",default=None,type=str,help="Overrides IFO list normally retrieve by event ID.  Use with care (e.g., glitch studies) or for events specified with --event-time.")
parser.add_argument("--online",action='store_true')
parser.add_argument("--extra-args-helper",action=None, help="Filename with arguments for the helper. Use to provide alternative channel names and other advanced configuration (--channel-name, data type)!")
parser.add_argument("--manual-postfix",default='',type=str)
parser.add_argument("--gracedb-id",default=None,type=str)
parser.add_argument("--gracedb-exe",default="gracedb")
parser.add_argument("--use-legacy-gracedb",action='store_true')
parser.add_argument("--internal-use-gracedb-bayestar",action='store_true',help="Retrieve BS skymap from gracedb (bayestar.fits), and use it internally in integration with --use-skymap bayestar.fits.")
parser.add_argument("--event-time",default=None,type=float,help="Event time. Intended to override use of GracedbID. MUST provide --manual-initial-grid ")
parser.add_argument("--calibration",default="C00",type=str)
parser.add_argument("--playground-data",action='store_true', help="Passed through to helper_LDG_events, and changes name prefix")
parser.add_argument("--approx",default=None,type=str,help="Approximant. REQUIRED")
parser.add_argument("--use-gwsurrogate",action='store_true',help="Attempt to use gwsurrogate instead of lalsuite.")
parser.add_argument("--l-max",default=2,type=int)
parser.add_argument("--no-matter",action='store_true', help="Force analysis without matter. Really only matters for BNS")
parser.add_argument("--assume-nospin",action='store_true', help="Force analysis with zero spin")
parser.add_argument("--assume-precessing",action='store_true', help="Force analysis *with* transverse spins")
parser.add_argument("--assume-nonprecessing",action='store_true', help="Force analysis *without* transverse spins")
parser.add_argument("--assume-matter",action='store_true', help="Force analysis *with* matter. Really only matters for BNS")
parser.add_argument("--assume-matter-but-primary-bh",action='store_true',help="If present, the code will add options necessary to manage tidal arguments for the smaller body ONLY. (Usually pointless)")
parser.add_argument("--internal-tabular-eos-file",type=str,default=None,help="Tabular file of EOS to use.  The default prior will be UNIFORM in this table!")
parser.add_argument("--assume-eccentric",action='store_true', help="Add eccentric options for each part of analysis")
parser.add_argument("--assume-lowlatency-tradeoffs",action='store_true', help="Force analysis with various low-latency tradeoffs (e.g., drop spin 2, use aligned, etc)")
parser.add_argument("--assume-highq",action='store_true', help="Force analysis with the high-q strategy, neglecting spin2. Passed to 'helper'")
parser.add_argument("--assume-well-placed",action='store_true',help="If present, the code will adopt a strategy that assumes the initial grid is very well placed, and will minimize the number of early iterations performed. Not as extrme as --propose-flat-strategy")
parser.add_argument("--ile-distance-prior",default=None,help="If present, passed through to the distance prior option.   If provided, BLOCKS distance marginalization")
parser.add_argument("--internal-ile-request-disk",help="Use if you are transferring large files, or if you otherwise expect a lot of data ")
parser.add_argument("--internal-marginalize-distance",action='store_true',help="If present, the code will marginalize over the distance variable. Passed diretly to helper script. Default will be to generate d_marg script *on the fly*")
parser.add_argument("--internal-marginalize-distance-file",help="Filename for marginalization file.  You MUST make sure the max distance is set correctly")
parser.add_argument("--internal-distance-max",type=float,help="If present, the code will use this as the upper limit on distance (overriding the distance maximum in the ini file, or any other setting). *required* to use internal-marginalize-distance in most circumstances")
parser.add_argument("--internal-correlate-default",action='store_true',help='Force joint sampling in mc,delta_mc, s1z and possibly s2z')
parser.add_argument("--internal-force-iterations",type=int,default=None,help="If inteeger provided, overrides internal guidance on number of iterations, attempts to force prolonged run. By default puts convergence tests on")
parser.add_argument("--internal-flat-strategy",action='store_true',help="Use the same CIP options for every iteration, with convergence tests on.  Passes --test-convergence, ")
parser.add_argument("--internal-use-amr",action='store_true',help="Changes refinement strategy (and initial grid) to use. PRESENTLY WE CAN'T MIX AND MATCH AMR, CIP ITERATIONS, so this is fixed for the whole run right now; use continuation and 'fetch' to augment")
parser.add_argument("--internal-use-amr-bank",default="",type=str,help="Bank used for template")
parser.add_argument("--internal-use-amr-puff",action='store_true',help="Use puffball with AMR (as usual).  May help with stalling")
parser.add_argument("--internal-use-aligned-phase-coordinates", action='store_true', help="If present, instead of using mc...chi-eff coordinates for aligned spin, will use SM's phase-based coordinates. Requires spin for now")
parser.add_argument("--internal-use-rescaled-transverse-spin-coordinates",action='store_true',help="If present, use coordinates which rescale the unit sphere with special transverse sampling")
parser.add_argument("--external-fetch-native-from",type=str,help="Directory name of run where grids will be retrieved.  Recommend this is for an ACTIVE run, or otherwise producing a large grid so the retrieved grid changes/isn't fixed")
parser.add_argument("--internal-propose-converge-last-stage",action='store_true',help="Pass through to helper")
parser.add_argument("--add-extrinsic",action='store_true')
parser.add_argument("--add-extrinsic-time-resampling",action='store_true',help="adds the time resampling option.  Only deployed for vectorized calculations (which should be all that end-users can access)")
parser.add_argument("--batch-extrinsic",action='store_true')
parser.add_argument("--fmin",default=20,type=int,help="Mininum frequency for integration. template minimum frequency (we hope) so all modes resolved at this frequency")  # should be 23 for the BNS
parser.add_argument("--fmin-template",default=None,type=float,help="Mininum frequency for template. If provided, then overrides automated settings for fmin-template = fmin/Lmax")  # should be 23 for the BNS
parser.add_argument("--data-LI-seglen",default=None,type=int,help="If specified, passed to the helper. Uses data selection appropriate to LI. Must specify the specific LI seglen used.")
parser.add_argument("--choose-data-LI-seglen",action='store_true')
parser.add_argument("--fix-bns-sky",action='store_true')
parser.add_argument("--ile-sampler-method",type=str,default=None)
parser.add_argument("--ile-n-eff",type=int,default=None,help="ILE n_eff passed to helper/downstream. Default internally is 50; lower is faster but less accurate, going much below 10 could be dangerous ")
parser.add_argument("--cip-sampler-method",type=str,default=None)
parser.add_argument("--cip-fit-method",type=str,default=None)
parser.add_argument("--cip-internal-use-eta-in-sampler",action='store_true', help="Use 'eta' as a sampling parameter. Designed to make GMM sampling behave particularly nicely for objects which could be equal mass")
parser.add_argument("--ile-jobs-per-worker",type=int,default=None,help="Default will be 20 per worker usually for moderate-speed approximants, and more for very fast configurations")
parser.add_argument("--ile-no-gpu",action='store_true')
parser.add_argument("--ile-force-gpu",action='store_true')
parser.add_argument("--fake-data-cache",type=str)
parser.add_argument("--spin-magnitude-prior",default='default',type=str,help="options are default [uniform mag for precessing, zprior for aligned], volumetric, uniform_mag_prec, uniform_mag_aligned, zprior_aligned")
parser.add_argument("--force-lambda-max",default=None,type=float,help="Provde this value to override the value of lambda-max provided") 
parser.add_argument("--force-lambda-small-max",default=None,type=float,help="Provde this value to override the value of lambda-small-max provided") 
parser.add_argument("--force-chi-max",default=None,type=float,help="Provde this value to override the value of chi-max provided") 
parser.add_argument("--force-chi-small-max",default=None,type=float,help="Provde this value to override the value of chi-max provided") 
parser.add_argument("--force-ecc-max",default=None,type=float,help="Provde this value to override the value of ecc-max provided")
parser.add_argument("--force-ecc-min",default=None,type=float,help="Provde this value to override the value of ecc-min provided")
parser.add_argument("--force-mc-range",default=None,type=str,help="Pass this argumen through to the helper to set the mc range")
parser.add_argument("--force-eta-range",default=None,type=str,help="Pass this argumen through to the helper to set the eta range")
parser.add_argument("--force-hint-snr",default=None,type=str,help="Pass this argumen through to the helper to control source amplitude effects")
parser.add_argument("--force-initial-grid-size",default=None,type=float,help="Only used for automated grids.  Passes --force-initial-grid-size down to helper")
parser.add_argument("--hierarchical-merger-prior-1g",action='store_true',help="As in 1903.06742")
parser.add_argument("--hierarchical-merger-prior-2g",action='store_true',help="As in 1903.06742")
parser.add_argument("--link-reference-pe",action='store_true',help="If present, creates a directory 'reference_pe' and adds symbolic links to fiducial samples. These can be used by the automated plotting code.  Requires LVC_PE_SAMPLES environment variable defined!")
parser.add_argument("--link-reference-psds",action='store_true',help="If present, uses the varialbe LVC_PE_CONFIG to find a 'reference_pe_config_map.dat' file, which provides the location for reference PSDs.  Will override PSDs used / setup by default")
parser.add_argument("--make-bw-psds",action='store_true',help='If present, adds nodes to create BW PSDs to the dag.  If at all possible, avoid this and re-use existing PSDs')
parser.add_argument("--link-bw-psds",action='store_true',help='If present, uses the script retrieve_bw_psd_for_event.sh  to find a precomputed BW psd, and convert it to our format')
parser.add_argument("--use-online-psd",action='store_true', help="If present, will use the online PSD estimates")
parser.add_argument("--ile-retries",default=3,type=int)
parser.add_argument("--general-retries",default=3,type=int)
parser.add_argument("--ile-runtime-max-minutes",default=None,type=int,help="If not none, kills ILE jobs that take longer than the specified integer number of minutes. Do not use unless an expert")
parser.add_argument("--fit-save-gp",action="store_true",help="If true, pass this argument to CIP. GP plot for each iteration will be saved. Useful for followup investigations or reweighting. Warning: lots of disk space (1G or so per iteration)")
parser.add_argument("--cip-explode-jobs",type=int,default=None)
parser.add_argument("--cip-explode-jobs-last",type=int,default=None,help="Number of jobs to use in last stage.  Hopefully in future auto-set")
parser.add_argument("--cip-quadratic-first",action='store_true')
parser.add_argument("--n-output-samples",type=int,default=5000,help="Number of output samples generated in the final iteration")
parser.add_argument("--internal-cip-cap-neff",type=int,default=500,help="Largest value for CIP n_eff to use for *non-final* iterations. ALWAYS APPLIED. ")
parser.add_argument("--internal-cip-temper-log",action='store_true',help="Use temper_log in CIP.  Helps stabilize adaptation for high q for example")
parser.add_argument("--internal-ile-sky-network-coordinates",action='store_true',help="Passthrough to ILE ")
parser.add_argument("--internal-ile-rotate-phase", action='store_true')
parser.add_argument("--internal-loud-signal-mitigation-suite",action='store_true',help="Enable more aggressive adaptation - make sure we adapt in distance, sky location, etc rather than use uniform sampling, because we are constraining normally subdominant parameters")
parser.add_argument("--internal-ile-freezeadapt",action='store_true',help="Passthrough to ILE ")
parser.add_argument("--internal-ile-adapt-log",action='store_true',help="Passthrough to ILE ")
parser.add_argument("--internal-ile-auto-logarithm-offset",action='store_true',help="Passthrough to ILE")
parser.add_argument("--manual-initial-grid",default=None,type=str,help="Filename (full path) to initial grid. Copied into proposed-grid.xml.gz, overwriting any grid assignment done here")
parser.add_argument("--manual-extra-ile-args",default=None,type=str,help="Avenue to adjoin extra ILE arguments.  Needed for unusual configurations (e.g., if channel names are not being selected, etc)")
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--use-quadratic-early",action='store_true',help="If provided, use a quadratic fit in the early iterations'")
parser.add_argument("--use-gp-early",action='store_true',help="If provided, use a gp fit in the early iterations'")
parser.add_argument("--use-cov-early",action='store_true',help="If provided, use cov fit in the early iterations'")
parser.add_argument("--use-osg",action='store_true',help="Restructuring for ILE on OSG. The code by default will use CVMFS")
parser.add_argument("--use-osg-cip",action='store_true',help="Restructuring for ILE on OSG. The code by default will use CVMFS")
parser.add_argument("--use-osg-file-transfer",action='store_true',help="Restructuring for ILE on OSG. The code will NOT use CVMFS, and instead will try to transfer the frame files.")
parser.add_argument("--internal-truncate-files-for-osg-file-transfer",action='store_true',help="If use-osg-file-transfer, will use FrCopy plus the start/end time to build the frame directory.")
parser.add_argument("--condor-local-nonworker",action='store_true',help="Provide this option if job will run in non-NFS space. ")
parser.add_argument("--condor-nogrid-nonworker",action='store_true',help="NOW STANDARD, auto-set if you pass use-osg   Causes flock_local for 'internal' jobs")
parser.add_argument("--use-osg-simple-requirements",action='store_true',help="Provide this option if job should use a more aggressive setting for OSG matching ")
parser.add_argument("--archive-pesummary-label",default=None,help="If provided, creates a 'pesummary' directory and fills it with this run's final output at the end of the run")
parser.add_argument("--archive-pesummary-event-label",default="this_event",help="Label to use on the pesummary page itself")
opts=  parser.parse_args()



if (opts.use_ini):
    # Attempt to lazy-parse all command line arguments from ini file
    config = ConfigParser.ConfigParser()
    config.optionxform=str # force preserve case! Important for --choose-data-LI-seglen
    config.read(opts.use_ini)
    if 'rift-pseudo-pipe' in config:
        # get the list of items
        rift_items = dict(config["rift-pseudo-pipe"])
        config_dict = vars(opts) # access dictionry of options
#        print(config_dict)
#        print(list(rift_items))

        # acounting groups/users: if presnet and NOT DEFINED IN ENV (which dominates!), define them
        if not('LIGO_USER_NAME'  in os.environ) and 'accounting_group_user' in rift_items:
            os.environ["LIGO_USER_NAME"] = rift_items['accounting_group_user']
        if not('LIGO_ACCOUNTING'  in os.environ) and 'accounting_group' in rift_items:
            os.environ["LIGO_ACCOUNTING"] = rift_items['accounting_group']
        
        # attempt to lazy-select the command-line that are present in the ini file section
        for item in rift_items:
            item_renamed = item.replace('-','_')
            if (item_renamed in config_dict):
                val = rift_items[item].strip()
#                if not(config_dict[item_renamed]):   # needs to be set to some value. Don't *disable* what is enabled on command line
                print(" ini file parser (overrides command line, except booleans): ",item, rift_items[item])
                if val != "":
                    config_dict[item_renamed] = eval(rift_items[item])
                else:
                    config_dict[item_renamed] = True
        print(config_dict)

if opts.internal_loud_signal_mitigation_suite:
    opts.internal_ile_freezeadapt=False  # make sure to adapt every iteration, and adapt in distance if present
    opts.internal_ile_sky_network_coordinates=True # skymap is better
    opts.internal_ile_rotate_phase = True  # phase coordinates can be sharper

# Default prior for aligned analysis should be z prior !
if opts.assume_nonprecessing or opts.approx == "IMRPhenomD":
    prior_args_lookup["default"] = prior_args_lookup["zprior_aligned"]

if opts.use_osg:
    opts.condor_nogrid_nonworker = True  # note we ALSO have to check this if we set use_osg in the ini file! Moved statement so flagged



if not(opts.ile_jobs_per_worker):
    opts.ile_jobs_per_worker=20
    if opts.assume_nospin or opts.assume_nonprecessing or (opts.approx == "IMRPhenomD" or opts.approx == "SEOBNRv4"):
        if opts.internal_marginalize_distance:
            # if we are using distance marginalization, use many more jobs per worker, to reduce startup transient relative cost (and queuing time latency). Jobs are too fast.
            opts.ile_jobs_per_worker =100 

if opts.use_production_defaults:
    opts.condor_nogrid_nonworker =True
    opts.use_cov_early =True
    opts.internal_marginalize_distance =True
    opts.cip_explode_jobs = 5 # will be overriden later
    if opts.use_osg:
        opts.use_nogrid_nonworker = True
        opts.ile_retries=10  # very unstable environment

if opts.internal_use_amr:
    # Require subdags!  Makes sure we evaluate all subgrid points
    opts.use_subdags = True
    # Disable incompatible settings
    opts.external_fetch_native_from = None
    opts.cip_explode_jobs= None

    amr_q_coord = "delta"
    amr_q_coord_range="0.0,0.95"
#    amr_q_coord = "eta"
#    amr_q_coord_range="0.05,0.249999"

if opts.internal_force_iterations and opts.internal_propose_converge_last_stage:
    print("==> Inconsistent options --internal-force-iterations and --internal-propose-converge-last-stage, overriding former")
    opts.internal_force_iterations= None # Can't force iteration number if we are using arbitrary iterate to convergence!

download_request = " get file "
gracedb_exe =opts.gracedb_exe
if opts.use_legacy_gracedb:
    gracedb_exe = "gracedb_legacy"
    download_request = " download "


if opts.assume_highq:
    opts.internal_correlate_default=True
event_dict={}

if (opts.approx is None) and not (opts.use_ini is None):
#    config = ConfigParser.ConfigParser()
#    config.read(opts.use_ini)
    approx_name_ini = config.get('engine','approx')
    approx_name_cleaned = lalsim.GetStringFromApproximant(lalsim.GetApproximantFromString(approx_name_ini))
    opts.approx = approx_name_cleaned
    print( " Approximant provided in ini file: ",approx_name_cleaned)
elif opts.approx is None:
    print( " Approximant required! ")
    sys.exit(1)

if opts.use_osg:
    os.environ["LIGO_DATAFIND_SERVER"]="datafind.ligo.org:443"   #  enable lookup of data

if opts.make_bw_psds:
    if not(opts.choose_data_LI_seglen) and (opts.data_LI_seglen is None):
        print( " To use the BW PSD, you MUST provide a default analysis seglen ")
        sys.exit(1)

if opts.online:
        opts.use_online_psd =True
        if opts.link_bw_psds:
            print( " Inconsistent options for PSDs ")
            sys.exit(1)

fmin = opts.fmin
fmin_template  = opts.fmin
if opts.l_max > 2:
    print( " ==> Reducing minimum template frequency because of HM <== ")
    fmin_template = opts.fmin * 2./opts.l_max
if not(opts.fmin_template is None):
    fmin_template = opts.fmin_template
gwid = opts.gracedb_id if (not opts.gracedb_id is None) else '';
if opts.gracedb_id is None:
    gwid="manual_"+ format_gps_time(opts.event_time)
    if not (opts.use_ini is None):
        gwid = ''
else:
# checks X509_USER_PROXY env variable
# if empty, checks grid-proxy-info -path
# if empty, fails and tells you to run ligo-proxy-init
    if not("X509_USER_PROXY" in os.environ.keys()):
        import subprocess
        str_proxy =subprocess.check_output(['grid-proxy-info','-path']).rstrip()
        if len(str_proxy) < 1:
            print( " Run ligo-proxy-init or otherwise have a method to query gracedb / use CVMFS frames as you need! ! ")
            sys.exit(1)
print(" Event ", gwid)
base_dir = os.getcwd()
if opts.use_rundir:
    base_dir =''
#if opts.use_ini:
#    base_dir =''  # all directories are provided as full path names


if opts.choose_data_LI_seglen:
    coinc_file = "coinc.xml"
    if not(opts.use_coinc):
        cmd_event = gracedb_exe + download_request + opts.gracedb_id  + " coinc.xml"
        if not(opts.use_legacy_gracedb):
            cmd_event += " > coinc.xml "
        os.system(cmd_event)
        cmd_fix_ilwdchar = "ligolw_no_ilwdchar coinc.xml"; os.system(cmd_fix_ilwdchar) # sigh, need to make sure we are compatible
    elif opts.use_coinc:
        coinc_file = opts.use_coinc
    event_dict = retrieve_event_from_coinc(coinc_file)
    P=lalsimutils.ChooseWaveformParams()
    P.m1 = event_dict["m1"]*lal.MSUN_SI; P.m2=event_dict["m2"]*lal.MSUN_SI; P.s1z = event_dict["s1z"]; P.s2z = event_dict["s2z"]
    P.fmin = opts.fmin  #  fmin we will use internally
    T_wave = lalsimutils.estimateWaveformDuration(P) +2  # 2 second buffer on end; note that with next power of 2, will go up to 4s
    T_wave_round = lalsimutils.nextPow2( T_wave)

    # For frequency-domain approximants, I need another factor of 2!
    # We have an extra buffer
    if lalsim.SimInspiralImplementedFDApproximants(P.approx)==1:
            print( " FD approximant, needs extra buffer for RIFT at present ")
            T_wave_round *=2 

    print( " Assigning auto-selected segment length ", T_wave_round)
    opts.data_LI_seglen  = T_wave_round

    # Problem with SEOBNRv3 starting frequencies
    mtot_msun = event_dict["m1"]+event_dict["m2"] 
    if ('SEOB' in opts.approx) and mtot_msun > 90*(20./opts.fmin):
            fmin_template = int(14*(90/mtot_msun))   # should also decrease this due to lmax!
            print( "  SEOB starting frequencies need to be reduced for this event; trying ", fmin_template)


is_analysis_precessing =False
is_analysis_eccentric =False
if opts.approx == "SEOBNRv3" or opts.approx == "NRSur7dq2" or opts.approx == "NRSur7dq4" or (opts.approx == 'SEOBNv3_opt') or (opts.approx == 'IMRPhenomPv2') or (opts.approx =="SEOBNRv4P" ) or (opts.approx == "SEOBNRv4PHM") or ('SpinTaylor' in opts.approx) or ('IMRPhenomTP' in opts.approx or ('IMRPhenomXP' in opts.approx)):
        is_analysis_precessing=True
if opts.assume_precessing:
        is_analysis_precessing = True
if opts.assume_nonprecessing:
        is_analysis_precessing = False
if opts.assume_eccentric:
        is_analysis_eccentric = True


dirname_run = gwid+ "_" + opts.calibration+ "_"+ opts.approx+"_fmin" + str(fmin) +"_fmin-template"+str(fmin_template) +"_lmax"+str(opts.l_max) + "_"+opts.spin_magnitude_prior
if opts.online:
    dirname_run += "_onlineLLframes"
elif opts.use_online_psd:
    dirname_run += "_onlinePSD"
elif opts.link_bw_psds:
    dirname_run += "_fiducialBWpsd"
elif opts.make_bw_psds:
    dirname_run += "_manualBWpsd"
if opts.data_LI_seglen:
    dirname_run += "_LIseglen"+str(opts.data_LI_seglen)
if opts.assume_matter:
    dirname_run += "_with_matter"
if opts.assume_eccentric:
    dirname_run += "_with_eccentricity"
if opts.no_matter:
    dirname_run += "_no_matter"
if opts.assume_highq:
    dirname_run+="_highq"
if opts.assume_well_placed:
    dirname_run+="_placed"
if opts.playground_data:
    dirname_run = "playground_" + dirname_run
if not(opts.cip_sampler_method is None):
    dirname_run += "_" + opts.cip_sampler_method
if not(opts.cip_fit_method is None):
    dirname_run += "_" + opts.cip_fit_method
if opts.use_osg:
    dirname_run += '_OSG'
if opts.manual_postfix:
    dirname_run += opts.manual_postfix
# Override run directory name
if opts.use_rundir:
    dirname_run = opts.use_rundir
os.mkdir(dirname_run)
os.chdir(dirname_run)


if not(opts.use_ini is None):
    if opts.use_coinc is None:
        print( " coinc required for ini file operation at present ")
        sys.exit(1)
    # Load in event dictionary
    event_dict = retrieve_event_from_coinc(opts.use_coinc)
    # Create relevant sim_xml file to hold parameters (does not parse coinc)
    P=lalsimutils.ChooseWaveformParams()
    P.m1 = event_dict["m1"]*lal.MSUN_SI; P.m2=event_dict["m2"]*lal.MSUN_SI; P.s1z = event_dict["s1z"]; P.s2z = event_dict["s2z"]
    # Load in ini file to select relevant fmin, fref [latter usually unused]
#    config = ConfigParser.ConfigParser()
#    config.read(opts.use_ini)
    fmin_vals ={}
    fmin_fiducial = -1
    ifo_list = eval(config.get('analysis','ifos'))
    for ifo in ifo_list:
        fmin_vals[ifo] = unsafe_config_get(config,['lalinference','flow'])[ifo]
        fmin_fiducial = fmin_vals[ifo]
    event_dict["IFOs"] = ifo_list
    print( "IFO list from ini ", ifo_list)
    P.fmin = fmin_fiducial
    P.fref = unsafe_config_get(config,['engine','fref'])
    # Write 'target_params.xml.gz' file
    lalsimutils.ChooseWaveformParams_array_to_xml([P], "target_params")

    if opts.use_production_defaults:
        # use more workers for high-q triggers
        # worker scale = (1+2/q), max of 50
        q = P.m2/P.m1
        opts.cip_explode_jobs = np.min([int(2+3./q),50])


helper_psd_args = ''
srate=4096  # default, built into helper, unwise to go lower, LI will almost never do higher
if opts.make_bw_psds:
    helper_psd_args += " --assume-fiducial-psd-files --fmax " + str(srate/2-1)

# Create provenance info : we want run to be reproducible
# for low-latency analysis, we can assume we have provenance.
if not(assume_lowlatency):
        os.mkdir("reproducibility")
        # Write this script and its arguments
        import shutil, json
#        thisfile = os.path.realpath(__file__)
#        shutil.copyfile(thisfile, "reproducibility/the_script_used.py")
        argparse_dict = vars(opts)
        with open("reproducibility/the_arguments_used.json",'w') as f:
                json.dump(argparse_dict,f)
        # Write commits
#        cmd = "(cd ${ILE_CODE_PATH}; git rev-parse HEAD) > reproducibility/RIFT.commit"
#        os.system(cmd)
        module_list = ['gwsurrogate',  'NRSur7dq2', 'scipy', 'numpy', 'sklearn', 'lalsimulation','lal']
        with open("reproducibility/module_versions", 'w') as f:
                for name in module_list:
                    try:
                        exec("import "+ name+"; val = "+name+".__version__")
                        f.write(name +" " +val+"\n")
                    except:
                        print( " No provenance for ", name)


# Run helper command
npts_it = 500
cmd = " helper_LDG_Events.py --force-notune-initial-grid   --propose-fit-strategy --propose-ile-convergence-options  --fmin " + str(fmin) + " --fmin-template " + str(fmin_template) + " --working-directory " + base_dir + "/" + dirname_run  + helper_psd_args  + " --no-enforce-duration-bound --test-convergence "
if opts.internal_use_gracedb_bayestar:
    cmd += " --internal-use-gracedb-bayestar "
if opts.internal_use_amr:
    cmd += " --internal-use-amr " # minimal support performed in this routine, mainly for puff
if opts.internal_use_aligned_phase_coordinates:
    cmd += " --internal-use-aligned-phase-coordinates "
if opts.internal_use_rescaled_transverse_spin_coordinates:
    cmd += " --internal-use-rescaled-transverse-spin-coordinates "
if not(opts.internal_use_amr) and not(opts.manual_initial_grid):
    cmd+= " --propose-initial-grid "
if opts.force_initial_grid_size:
    cmd += " --force-initial-grid-size {} ".format(opts.force_initial_grid_size)
if opts.assume_matter:
        cmd += " --assume-matter "
        npts_it = 1000
        if opts.assume_matter_but_primary_bh:
            cmd+= " --assume-matter-but-primary-bh "
        if opts.internal_tabular_eos_file:
            cmd += " --internal-tabular-eos-file {} ".format(opts.internal_tabular_eos_file)
if  opts.assume_nospin:
    cmd += " --assume-nospin "
else:  
  if is_analysis_precessing:
        cmd += " --assume-precessing-spin "
        npts_it = 1500
if is_analysis_eccentric:
    cmd += " --assume-eccentric "
if opts.assume_highq:
    cmd+= ' --assume-highq  --force-grid-stretch-mc-factor 2'  # the mc range, tuned to equal-mass binaries, is probably too narrow. Workaround until fixed in helper
    npts_it =1000
if opts.internal_propose_converge_last_stage:
    cmd += " --propose-converge-last-stage "
if not(opts.cip_fit_method is None):
    cmd += " --force-fit-method {} ".format(opts.cip_fit_method)
    if opts.cip_fit_method == 'rf':
        npts_it*=2 # more iteration points if we use RF ... not sane otherwise. Note for precession this is a large iteration size
    elif opts.cip_fit_method == 'quadratic' or opts.cip_fit_method == 'polynomial' or opts.use_quadratic_early or opts.use_cov_early:
        npts_it*=2 # more iteration points if we use some initial quadratic iterations ... they also benefit from more samples overall. Default description is for GP



if not(opts.ile_n_eff is None):
    cmd += " --ile-n-eff {} ".format(opts.ile_n_eff)
if not(opts.force_mc_range is None):
    cmd+= " --force-mc-range {} ".format(opts.force_mc_range)
if not(opts.force_eta_range is None):
    cmd+= " --force-eta-range {} ".format(opts.force_eta_range)
if not(opts.gracedb_id is None) and (opts.use_ini is None):
    cmd +="  --gracedb-id " + gwid 
    if  opts.use_legacy_gracedb:
        cmd+= " --use-legacy-gracedb "
elif  not(opts.event_time is None):
    cmd += " --event-time " + format_gps_time(opts.event_time)
if opts.online:
        cmd += " --online "
if opts.playground_data:
        cmd += " --playground-data "
if opts.use_online_psd:
        cmd += " --use-online-psd "
if opts.data_LI_seglen:
        cmd += " --data-LI-seglen "+str(opts.data_LI_seglen)
if opts.assume_well_placed:
    cmd += " --assume-well-placed "
#if is_event_bns and not opts.no_matter:
#        cmd += " --assume-matter "
#        npts_it = 1000
if opts.internal_flat_strategy:
    cmd +=  " --test-convergence --propose-flat-strategy "
if opts.use_quadratic_early:
    cmd += " --use-quadratic-early "
elif opts.use_gp_early:
    cmd += " --use-gp-early "
elif opts.use_cov_early:
    cmd += " --use-cov-early "
if opts.use_osg:
    cmd += " --use-osg "
    if not(opts.use_osg_file_transfer):
        cmd += " --use-cvmfs-frames "  # only run with CVMFS data, otherwise very very painful
if opts.use_ini:
    cmd += " --use-ini " + opts.use_ini
    cmd += " --sim-xml {}/target_params.xml.gz --event 0 ".format(base_dir + "/"+ dirname_run)  # full path to target_params.xml.gz
    if (opts.event_time is None):
        cmd += " --event-time " + str(event_dict["tref"])
    #
else:
    cmd += " --calibration-version " + opts.calibration 
if opts.use_online_psd_file:
    # Get IFO list from ini file
##    import ConfigParser
#    config = ConfigParser.ConfigParser()
#    config.read(opts.use_ini)
    ifo_list = eval(config.get('analysis','ifos'))
    # Create command line arguments for those IFOs, so helper can correctly pass then downward
    for ifo in ifo_list:
        cmd+= " --psd-file {}={}".format(ifo,opts.use_online_psd_file)
if "SNR" in event_dict:
    cmd += " --hint-snr {} ".format(event_dict["SNR"])
if not(opts.force_hint_snr is None):
    cmd += " --hint-snr {} ".format(opts.force_hint_snr)
if not(opts.event_time is None) and not(opts.manual_ifo_list is None):
    cmd += " --manual-ifo-list {} ".format(opts.manual_ifo_list)
if (opts.internal_marginalize_distance) and not opts.ile_distance_prior:
    cmd += " --internal-marginalize-distance "
if (opts.internal_marginalize_distance_file ):
    cmd += " --internal-marginalize-distance-file {} ".format(opts.internal_marginalize_distance_file)
if not(opts.internal_distance_max is None):
    cmd += ' --internal-distance-max {} '.format(opts.internal_distance_max)
if opts.add_extrinsic:
    cmd += " --last-iteration-extrinsic "
if opts.internal_ile_freezeadapt:
    cmd += " --internal-propose-ile-convergence-freezeadapt "  # old-style O3: adaptation frozen after first point, no distance adapt (!)
if opts.internal_ile_adapt_log:
    cmd += " --internal-propose-ile-adapt-log "  # old-style O3: adaptation frozen after first point, no distance adapt (!)
if opts.internal_ile_auto_logarithm_offset:
    cmd += " --internal-ile-auto-logarithm-offset "
if opts.internal_ile_rotate_phase:
    cmd += " --internal-ile-rotate-phase "
# If user provides ini file *and* ini file has fake-cache field, generate a local.cache file, and pass it as argument
if opts.use_ini:
#    config = ConfigParser.ConfigParser()
#    config.read(opts.use_ini)
    if config.has_option("lalinference", "fake-cache"):
        # dictionary, entries are individual lcf files; we just need to concatenate their contents
        fake_cache_dict = unsafe_config_get(config,["lalinference","fake-cache"])
        fake_cache_fnames = [fake_cache_dict[x] for x in fake_cache_dict.keys()]
        cmd_cat = 'cat ' + ' '.join(fake_cache_fnames) + ' > local.cache'
        os.system(cmd_cat)
        cmd += " --cache local.cache --fake-data  "
if opts.fake_data_cache:
    cmd += " --cache {} --fake-data  ".format(opts.fake_data_cache)
    if len(event_dict["IFOs"]) >0 :
        short_list = " {} ".format(event_dict['IFOs'])        
        cmd += " --manual-ifo-list {} ".format(short_list.replace(' ',''))
print( cmd)
os.system(cmd)
#sys.exit(0)

# Create distance maximum (since that is NOT always chosen by the helper, and makes BNS runs needlessly much more painful!)
observing_run = 'O3'
if (opts.use_ini is None):
 try:
  with open("event.log",'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'ime:' in line:  # look for Event time, Event Time, etc
            tref = float(line.split(' ')[-1])
            observing_run = get_observing_run(tref)
        if 'hirp' in line:
            mc_Msun = float(line.split(' ')[-1])
 except:
   print( " Failure parsing event.log")
else:
    # use sim_xml produced above to generate necessary parameters
    t_ref = P.tref
    mc_Msun = P.extract_param('mc')/lal.MSUN_SI
snr_fac=1
#mc_Msun = P.extract_param('mc')/lal.MSUN_SI
try:
    dmax_guess =(1./snr_fac)* 2.5*2.26*typical_bns_range_Mpc[observing_run]* (mc_Msun/1.2)**(5./6.)
    dmax_guess = np.min([dmax_guess,10000]) # place ceiling
except:
    print( " ===> Defaulting to maximum distance <=== ")
    dmax_guess = 10000
# Last stage of commands done by other tools: too annoying to copy stuff over and run the next generation of the pipeline
instructions_ile = np.loadtxt("helper_ile_args.txt", dtype=str)  # should be one line
line = ' '.join(instructions_ile)
line += " --l-max " + str(opts.l_max) 
if (opts.use_ini is None) and not('--d-max' in line):
    line += " --d-max " + str(dmax_guess)
if opts.ile_distance_prior:
    line += " --d-prior {} ".format(opts.ile_distance_prior)
if opts.ile_force_gpu:
    line +=" --force-gpu-only "
sur_location_prefix = "my_surrogates/nr_surrogates/"
if 'GW_SURROGATE' in os.environ:
    sur_location_prefix=''
if opts.use_osg:
    sur_location_prefix = "/"
if not 'NR' in opts.approx:
        line += " --approx " + opts.approx
elif opts.use_gwsurrogate and 'NRHybSur' in opts.approx:
        line += " --rom-group {} --rom-param NRHybSur3dq8.h5 --approx {} ".format(sur_location_prefix,opts.approx)
elif opts.use_gwsurrogate and "NRSur7dq2" in opts.approx:
        line += " --rom-group {} --rom-param NRSur7dq2.h5 --approx {}  ".format(sur_location_prefix,opts.approx)
elif opts.use_gwsurrogate and "NRSur7dq4" in opts.approx:
        line += " --rom-group {} --rom-param NRSur7dq4.h5  --approx {}".format(sur_location_prefix,opts.approx)
elif ("SEOBNR" in opts.approx) or ("NRHybSur" in opts.approx) or ("NRSur7d" in opts.approx) or ("NRTidal" in opts.approx): 
        line += " --approx " + opts.approx
else:
        print( " Unknown approx ", opts.approx)
        sys.exit(1)
if not(opts.manual_extra_ile_args is None):
    line += opts.manual_extra_ile_args
if not(opts.ile_sampler_method is None):
    line += " --sampler-method {} ".format(opts.ile_sampler_method)
if opts.internal_ile_sky_network_coordinates:
    line += " --internal-sky-network-coordinates "
with open('args_ile.txt','w') as f:
        f.write(line)


#os.system("cp helper_test_args.txt args_test.txt")
with open ("helper_test_args.txt",'r') as f:
    line = f.readline()
    if opts.add_extrinsic: 
        # We NEVER want to terminate if we're doing extrinsic at the end.  Block termination, so extrinsic occurs on schedule
        line += " --always-succeed "
    with open("args_test.txt",'w') as g:
        g.write(line)

# CIP
#   - modify priors to be consistent with the spin priors used in the paper
#   - for the BNS, set chi_max
with open("helper_cip_arg_list.txt",'r') as f:
        raw_lines = f.readlines()


# Add arguments to the file we will use
instructions_cip = list(map(lambda x: x.rstrip().split(' '), raw_lines))#np.loadtxt("helper_cip_arg_list.txt", dtype=str)
n_iterations =0
lines  = []
for indx in np.arange(len(instructions_cip)):
    if instructions_cip[indx][0] == 'Z':
        n_iterations += 1
    else:
        n_iterations += int(instructions_cip[indx][0])
    line = ' ' .join(instructions_cip[indx])
    n_max_cip = 100000000;  # 1e8; doing more than this requires special memory management within the integrators in general. This lets us get a decent number of samples even with one worker for hard problems
    # if (opts.cip_sampler_method == "GMM") or (opts.cip_sampler_method == 'adaptive_cartesian_gpu'):
    #     n_max_cip *=3   # it is faster, so run longer; helps with correlated-sampling cases
    n_sample_target=opts.n_output_samples
    if indx < len(instructions_cip)-1: # on all but last iteration, cap the number of points coming out : this drives the total amount of work for AMR, etc!
        n_sample_target= np.min([opts.n_output_samples,10*opts.internal_cip_cap_neff])
    n_workers = 1
    if opts.cip_explode_jobs:
        n_workers = opts.cip_explode_jobs
    n_eff_cip_here = int(n_sample_target/n_workers)
    if indx < len(instructions_cip)-1: # on all but 
        n_eff_cip_here = np.min([opts.internal_cip_cap_neff, n_eff_cip_here]) # n_eff: make sure to do *less* than the limit. Lowering this saves immensely on internal/exploration runtime
    n_sample_min_per_worker = int(n_eff_cip_here/100)+2  # need at least 2 samples, and don't have any worker fall down on the job too much compared to the target

    # Analyze the iteration report
    n_eff_expected_max_easy = 1e-2 * n_max_cip
    n_eff_expected_max_hard = 1e-7 * n_max_cip
    print( " cip iteration group {} : n_eff likely will be between {} and {}, you are asking for at least {} and targeting {}".format(indx,n_eff_expected_max_easy, n_eff_expected_max_hard, n_sample_min_per_worker,n_eff_cip_here))

    line +=" --n-output-samples {}  --n-eff {} --n-max {}  --fail-unless-n-eff {}  --downselect-parameter m2 --downselect-parameter-range [1,1000] ".format(int(n_sample_target/n_workers), n_eff_cip_here, n_max_cip,n_sample_min_per_worker)
    if not(opts.cip_fit_method is None):
        line = line.replace('--fit-method gp ', '--fit-method ' + opts.cip_fit_method)  # should not be called, see --force-fit-method argument to helper
    if not (opts.cip_sampler_method is None):
        line += " --sampler-method "+opts.cip_sampler_method
    if opts.internal_cip_temper_log:
        line += " --internal-temper-log "
    line += prior_args_lookup[opts.spin_magnitude_prior]
    if opts.cip_internal_use_eta_in_sampler:
        line = line.replace('parameter delta_mc','parameter eta')
    if opts.cip_fit_method == 'quadratic' or opts.cip_fit_method == 'polynomial':
        line = line.replace('parameter delta_mc', 'parameter-implied eta --parameter-nofit delta_mc')     # quadratic fit needs eta coordinate. Should be done by helper ideally
    if opts.use_quadratic_early or opts.use_cov_early and indx < 1:
        line = line.replace('parameter delta_mc', 'parameter-implied eta --parameter-nofit delta_mc')     # quadratic or cov fit needs eta coordinate
    if opts.hierarchical_merger_prior_1g:
        # Must use mtotal, q coordinates!  Change defaults
        line = line.replace('parameter mc', 'parameter mtot')
        line = line.replace('parameter delta_mc', 'parameter q')
        line += " --prior-tapered-mass-ratio "
    elif opts.hierarchical_merger_prior_2g:
        # Must use mtotal, q coordinates! Change defaults
        line = line.replace('parameter mc', 'parameter mtot')
        line = line.replace('parameter delta_mc', 'parameter q')
        line += " --prior-gaussian-mass-ratio --prior-gaussian-spin1-magnitude "   # should require precessing analysis
    elif opts.assume_highq and ('s1z' in line):
        line += " --sampler-method GMM --internal-correlate-parameters 'mc,delta_mc,s1z' "
    elif opts.internal_correlate_default and ('s1z' in line):
        addme = " --sampler-method GMM --internal-correlate-parameters 'mc,delta_mc,s1z,s2z' "
        if 's1z_bar' in line:
            addme = addme.replace('s1z,', 's1z_bar')
            addme = addme.replace('s2z,', 's2z_bar')
        if opts.assume_precessing and ('cos_theta1' in line): # if we are in a polar coordinates step, change the correlated parameters. This is suboptimal.
            addme = addme.replace(',s1z,s2z', ',chi1,cos_theta1')
        # For high-q triggers, don't waste time correlating s2z
        if 'm2' in event_dict:
            if event_dict['m2']/event_dict['m1']< 0.4:
                addme = " --sampler-method GMM --internal-correlate-parameters 'mc,delta_mc,s1z' "
                if 's1z_bar' in line:
                    addme = addme.replace('s1z,', 's1z_bar')
            if opts.assume_precessing and ('cos_theta1' in line): # if we are in a polar coordinates step, change the correlated parameters. This is suboptimal.
                addme = addme.replace(',s1z' ',chi1,cos_theta1')
        line += addme

    # on last iteration, usually don't want to use correlated sampling if precessing, need to change coordinates
    if opts.approx in lalsimutils.waveform_approx_limit_dict:
        chi_max = lalsimutils.waveform_approx_limit_dict[opts.approx]["chi-max"]
        if not(opts.force_chi_max is None):
            chi_max = opts.force_chi_max
        q_min = lalsimutils.waveform_approx_limit_dict[opts.approx]["q-min"]
        eta_min = q_min/(1+q_min)**2
        line += " --chi-max {}  ".format(chi_max)
        # Secondary body can also have spin, allow us to force its range
        if opts.force_chi_small_max:
            line += " --chi-small-max {} ".format(chi_small_max)
        # Parse arguments, impose limit based on the approximant used, as described above
#        import StringIO
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument("--eta-range")
        my_opts, unknown_opts =my_parser.parse_known_args(line.split())
        eta_range_orig = eval(my_opts.eta_range)
        eta_range_revised = [np.max([eta_min,eta_range_orig[0]]),np.min([1,eta_range_orig[1]])]
        line=line.replace("--eta-range "+my_opts.eta_range,"--eta-range "+str(eta_range_revised))
        # Ideally, load in initial grid, and remove points outside the targeted range
        # IMPLEMENT THIS
        
        # Lambda range
        if opts.force_lambda_max:
            line += " --lambda-max  {} ".format(opts.force_lambda_max)
        if opts.force_lambda_small_max:
            line += " --lambda-small-max  {} ".format(opts.force_lambda_small_max)

    if opts.fit_save_gp:
        line += " --fit-save-gp my_gp "  # fiducial filename, stored in each iteration
    if opts.assume_eccentric:
        line = line.replace('parameter mc', 'parameter mc --parameter eccentricity --use-eccentricity')
        if not(opts.force_ecc_max is None):
            ecc_max = opts.force_ecc_max
            line += " --ecc-max {}  ".format(ecc_max)
        if not(opts.force_ecc_min is None):
            ecc_min = opts.force_ecc_min
            line += " --ecc-min {}  ".format(ecc_min)
    line += "\n"
    lines.append(line)

if opts.cip_quadratic_first:
    lines[0]=lines[0].replace(' --fit-method gp ', ' --fit-method quadratic ')
    lines[0]=lines[0].replace(' --parameter delta_mc ', ' --parameter eta ')   # almost without fail we are using mc, delta_mc, xi  as zeroth layer


if opts.internal_use_amr:
    lines =[ ] 
    # Manually implement aligned spin.  Should parse some of this from ini file ...
    print(" AMR prototype: Using hardcoded aligned-spin settings, setting arguments")
    internal_overlap_threshold = 0.001 # smallest it should be
    # if "SNR" in event_dict:
    #     internal_overlap_threshold = np.max([internal_overlap_threshold, 0.5*(6./event_dict["SNR"])**2])  # try to 
    internal_overlap_threshold = 1- internal_overlap_threshold
    amr_coord_dist  = "mchirp_eta"
    if opts.internal_use_aligned_phase_coordinates:
        amr_coord_dist = "mu1_mu2_q_s2z"
    lines += ["10 --no-exact-match --overlap-threshold {} ".format(internal_overlap_threshold) + " --distance-coordinates {} --verbose   --refine ".format(amr_coord_dist)+base_dir + "/" + dirname_run + "/intrinsic_grid_all_iterations.hdf --max-n-points 1000 --n-max-output 5000 " ]
    if opts.internal_use_amr_bank:
        lines[0] +=" --intrinsic-param mass1 --intrinsic-param mass2 "  # output by default written this way for bank files
    else:
        lines[0] +=" --intrinsic-param mchirp --intrinsic-param {} ".format(amr_q_coord)     # if we built the bank, we used mc, eta/q coordinates
    if not(opts.assume_nospin):
        lines[0] += " --intrinsic-param spin1z "
        if not(opts.assume_lowlatency_tradeoffs):
            lines[0] += " --intrinsic-param spin2z "

with open("args_cip_list.txt",'w') as f: 
   for line in lines:
           f.write(line)

# Write test file
# with open("args_test.txt",'w') as f:
#     test_args = " --method lame  --parameter m1 "
#     if not(opts.internal_use_amr):   # ALWAYS run the test with AMR
#         test_args +=  " --always-succeed  "
#     else:
#         test_args += " --threshold 0.02 "
#     f.write("X  "+test_args)


# Write puff file
#puff_params = " --parameter mc --parameter delta_mc --parameter chieff_aligned "
puff_max_it =4
#  Read puff args from file, if present
try:
    with open("helper_puff_max_it.txt",'r') as f:
        puff_max_it = int(f.readline())
except:
    print( " No puff file ")

instructions_puff = np.loadtxt("helper_puff_args.txt", dtype=str)  # should be one line
puff_params = ' '.join(instructions_puff)
if opts.assume_matter:
#    puff_params += " --parameter LambdaTilde "  # should already be present
    puff_max_it +=5   # make sure we resolve the correlations
if opts.assume_eccentric:
        puff_params += " --parameter eccentricity --downselect-parameter eccentricity --downselect-parameter-range '[0,0.9]' "
if opts.assume_highq:
    puff_params = puff_params.replace(' delta_mc ', ' eta ')  # use natural coordinates in the high q strategy. May want to do this always
    puff_max_it +=3
with open("args_puff.txt",'w') as f:
        puff_args = puff_params + " --downselect-parameter chi1 --downselect-parameter-range [0,1] --downselect-parameter chi2 --downselect-parameter-range [0,1] "
        if opts.assume_matter:
            lambda_max = 5000
            lambda_small_max=5000
            if opts.force_lambda_max:
                lambda_max = opts.force_lambda_max
            if opts.force_lambda_small_max:
                lambda_small_max = opts.force_lambda_small_max
            # Prevent negative lambda accidentally from puff
            puff_args += " --downselect-parameter lambda1 --downselect-parameter-range [0,{}] --downselect-parameter lambda2 --downselect-parameter-range [0,{}] ".format(lambda_max, lambda_small_max)
        if False: #opts.cip_fit_method == 'rf':
            # RF can majorly overfit and create 'voids' early on, eliminate the force-away
            # Should only do this in the INITIAL puff, not all, to avoid known problems later
            puff_args = puff_args.replace(unsafe_parse_arg_string(puff_args,'force-away'),'')
        if opts.data_LI_seglen:
                puff_args+= " --enforce-duration-bound " +str(opts.data_LI_seglen)
        f.write("X " + puff_args)

# Create archive dag.  Based on Udall's experience/code
#    * if ini file, use it
#    * PSD files: will need to convert from XML.  Will need wrapper to generate this (not raw pesummary call).. Not now.
if opts.archive_pesummary_label:
    os.mkdir("pesummary")
    rundir = base_dir+"/"+dirname_run
    if opts.add_extrinsic:
        samplestr = " --samples " + rundir +"/extrinsic_posterior_samples.dat "
    else:
        samplestr = " --samples " + rundir + "/posterior_samples-$(macroiteration).dat "
    labelstr = " --labels {} ".format(opts.archive_pesummary_label)
    configstr=""
    if opts.use_ini:
        configstr = " -c " +opts.use_ini
    approxstr = " -a "+opts.approx
    psdstr = ""
    plot_args = "--v --gw --webdir {}/pesummary".format(rundir)+ opts.archive_pesummary_event_label+samplestr+labelstr+approxstr+configstr+psdstr
    with open("args_plot.txt",'w') as f:
        f.write(plot_args)

# Overwrite iteration number
if opts.internal_force_iterations:
    n_iterations = opts.internal_force_iterations

# Overwrite grid if needed
if not (opts.manual_initial_grid is None):
    shutil.copyfile(opts.manual_initial_grid, "proposed-grid.xml.gz")

# Build DAG
cip_mem  = 30000
n_jobs_per_worker=opts.ile_jobs_per_worker
if opts.cip_fit_method == 'rf':
    cip_mem = 15000  # more typical for long-duration single-worker runs
if opts.cip_fit_method =='quadratic' or opts.cip_fit_method =='polynomial':  # much lower memory requirement
    cip_mem = 4000
cepp = "create_event_parameter_pipeline_BasicIteration"
if opts.use_subdags:
    cepp = "create_event_parameter_pipeline_AlternateIteration"
cmd =cepp+ "  --ile-n-events-to-analyze {} --input-grid proposed-grid.xml.gz --ile-exe  `which integrate_likelihood_extrinsic_batchmode`   --ile-args `pwd`/args_ile.txt --cip-args-list args_cip_list.txt --test-args args_test.txt --request-memory-CIP {} --request-memory-ILE 4096 --n-samples-per-job ".format(n_jobs_per_worker,cip_mem) + str(npts_it) + " --working-directory `pwd` --n-iterations " + str(n_iterations) + " --n-copies 1" + "   --ile-retries "+ str(opts.ile_retries) + " --general-retries " + str(opts.general_retries)
if not(opts.ile_runtime_max_minutes is None):
    cmd += " --ile-runtime-max-minutes {} ".format(opts.ile_runtime_max_minutes)
if not(opts.internal_use_amr) or opts.internal_use_amr_puff:
    cmd+= " --puff-exe `which util_ParameterPuffball.py` --puff-cadence 1 --puff-max-it " + str(puff_max_it)+ " --puff-args `pwd`/args_puff.txt "
if opts.assume_eccentric:
    cmd += " --use-eccentricity "
if opts.internal_use_amr:
    print(" AMR prototype: Using hardcoded aligned-spin settings, assembling grid, requires coinc!")
    cmd += " --cip-exe `which util_AMRGrid.py ` "
    coinc_file = "coinc.xml"
    if not(os.path.exists("coinc.xml")) and not(opts.use_coinc):
        # re-download coinc if not already present
        cmd_event = gracedb_exe + download_request + opts.gracedb_id  + " coinc.xml"
        if not(opts.use_legacy_gracedb):
            cmd_event += " > coinc.xml "
        os.system(cmd_event)
        cmd_fix_ilwdchar = "ligolw_no_ilwdchar coinc.xml"; os.system(cmd_fix_ilwdchar) # sigh, need to make sure we are compatible
    elif opts.use_coinc:
        coinc_file = opts.use_coinc
    event_dict = retrieve_event_from_coinc(coinc_file)
    if opts.internal_use_amr_bank:
        with open("toy.ini","w") as f:
            f.write("""
[General]

#The name of the directory you want results output to
output_parent_directory=output

[GridRefine]
no-exact-match=
distance-coordinates=mchirp_eta
overlap-thresh=0.99
verbose=
intrinsic-param=[mass1,mass2]

[InitialGridOnly]
overlap-threshold = 0.4
points-per-side=8
""")
        cmd_amr_init = "util_GridSubsetOfTemplateBank.py --use-ini {}  --use-bank {} --mass1 {} --mass2 {}  ".format("toy.ini",opts.internal_use_amr_bank,event_dict["m1"],event_dict["m2"]) #,event_dict["s1z"],event_dict["s2z"])  # --s1z {} --s2z {}
        if opts.assume_nospin:
            cmd_amr_init += " --assume-nospin "
        print(" INIT ",cmd_amr_init)
        os.system(cmd_amr_init)
        shutil.copyfile("intrinsic_grid_iteration_0.xml.gz", "proposed-grid.xml.gz")  # Actually put the grid in the right place
    else:
        # don't use bank files, instead use manually-prescribed mc, eta, spin range. SHOULD FIX TO BE TIGHTER
        mc_min,mc_max = lalsimutils.guess_mc_range(event_dict,force_mc_range=opts.force_mc_range)
        amr_coord_dist  = "mchirp_eta"
        if opts.internal_use_aligned_phase_coordinates:
            amr_coord_dist = "mu1_mu2_q_s2z"
        cmd_amr_init = "util_AMRGrid.py --mc-min {} --mc-max {} --distance-coordinates {} --initial-region mchirp={},{} --initial-region {}={} --initial-region spin1z=-0.8,0.8  --points-per-side 8 --fname-output-samples proposed-grid  --setup intrinsic_grid_all_iterations   ".format(mc_min,mc_max,amr_coord_dist,mc_min,mc_max,amr_q_coord,amr_q_coord_range)
        if not(opts.assume_lowlatency_tradeoffs):
            cmd_amr_init += "  --initial-region spin2z=-0.8,0.8  " # for lowlatency tradeoffs, drop spin2 as superfluous
        print(" INIT ", cmd_amr_init)
        os.system(cmd_amr_init)
    
if opts.external_fetch_native_from:
    import json
    # Write json file 
    fetch_dict = {}
    fetch_dict['method'] = 'native'
    fetch_dict['source'] = opts.external_fetch_native_from
    fetch_dict['n_max'] = 1000  # should tune this to grid structure needs; 1000 is probably safe; not yet implemented
    with open("my_dict.json",'w') as f:
        json.dump(fetch_dict,f)
    with open("fetch_args.txt",'w') as f:
        f.write("  --input-json {}/my_dict.json ".format(base_dir + "/"+ dirname_run))
    # Add command linke arguments
    cmd += " --fetch-ext-grid-exe `which util_FetchExternalGrid.py`  --fetch-ext-grid-args `pwd`/fetch_args.txt "
if not(opts.ile_no_gpu):
    cmd +=" --request-gpu-ILE "
if opts.add_extrinsic:
    cmd += " --last-iteration-extrinsic --last-iteration-extrinsic-nsamples {} ".format(opts.n_output_samples)
    if opts.add_extrinsic_time_resampling:
        cmd+= " --last-iteration-extrinsic-time-resampling "
if opts.batch_extrinsic:
    cmd += " --last-iteration-extrinsic-batched-convert "
if opts.internal_ile_request_disk:
    cmd += " --ile-request-disk {} ".format(opts.internal_ile_request_disk)
if opts.cip_explode_jobs:
   cmd+= " --cip-explode-jobs  " + str(opts.cip_explode_jobs) + " --cip-explode-jobs-dag "  # use dag workers
   if opts.cip_fit_method and not(opts.cip_fit_method == 'gp'):
       # if we are not using default GP fit, so all fit instances are equal
       cmd += " --cip-explode-jobs-flat "  
   if opts.cip_explode_jobs_last:
       cmd += " --cip-explode-jobs-last {} ".format(opts.cip_explode_jobs_last)
if opts.make_bw_psds:
    cmd+= " --use-bw-psd --bw-exe `which BayesWave` --bw-post-exe `which BayesWavePost` "
if opts.use_osg:
    cmd += " --use-osg --use-singularity  --cache-file local.cache  "   # run on the OSG, make sure to get frames (rather than try to transfer them).  Note with CVMFS frames we need to provide the cache, but that SHOULD be added to the arg list by the helper already.  However, the argument is needed to avoid failure.
    if opts.use_osg_cip:
        cmd += " --use-osg-cip "
    if not(opts.use_osg_file_transfer):
        cmd += " --use-cvmfs-frames "
    elif not(opts.internal_truncate_files_for_osg_file_transfer):  # attempt to make copies of frame files, and set up to transfer them with *every* job (!)
        os.system("util_ForOSG_MakeTruncatedLocalFramesDir.sh .")
#        os.system("echo ../frames_dir >> helper_transfer_files.txt")
        cmd += " --frames-dir `pwd`/frames_dir "
    cmd+= " --transfer-file-list  "+base_dir+"/"+dirname_run+"/helper_transfer_files.txt"
if opts.condor_local_nonworker:
    cmd += " --condor-local-nonworker "
if opts.condor_nogrid_nonworker:
    cmd += " --condor-nogrid-nonworker "
if opts.use_osg_simple_requirements:
    cmd += " --use-osg-simple-reqirements "
if opts.archive_pesummary_label:
#    cmd += " --plot-exe `which summarypages` --plot-args  args_plot.txt "
    cmd += " --plot-exe summarypages --plot-args  args_plot.txt "
print(cmd)
os.system(cmd)

if opts.use_osg_file_transfer and opts.internal_truncate_files_for_osg_file_transfer:
    # build truncated frames.  Note this parses ILE arguments, so must be done last
    os.system("util_ForOSG_MakeTruncatedLocalFramesDir.sh .")

## RUNMON
try:
    from runmonitor import store_tools as sto
    if opts.use_ini != None: # making an assumption that opts.use_ini corresponds to prod_O3b file structures, and that opts.use_ini == None corresponds to standard setup with opts.gracedb_id passed. Maybe not a robust assumptio
        level = 2
        event = os.getcwd.split("/")[-2].split("_")[0]
    else:
        level = 1
        event = opts.gracedb_id
    sto.store(event,level)
except Exception as fail:
    print(fail)
    print("Unable to initialize run monitoring automatically. If you wish to use this feature please do so manually")
