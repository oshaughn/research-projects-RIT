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
    from glue.ligolw import lsctables, table, utils
    from RIFT import lalsimutils
    event_dict ={}
    samples = table.get_table(utils.load_filename(fname_coinc,contenthandler=lalsimutils.cthdler), lsctables.SnglInspiralTable.tableName)
    event_duration=4  # default
    ifo_list = []
    for row in samples:
        m1 = row.mass1
        m2 = row.mass2
        ifo_list.append(row.ifo)
        try:
            event_duration = row.event_duration # may not exist
        except:
            print( " event_duration field not in XML ")
    event_dict["m1"] = row.mass1
    event_dict["m2"] = row.mass2
    event_dict["s1z"] = row.spin1z
    event_dict["s2z"] = row.spin2z
    event_dict["tref"] = row.end_time + 1e-9*row.end_time_ns
    event_dict["IFOs"] = list(set(ifo_list))
    event_dict["SNR"] = row.snr
    return event_dict

def unsafe_parse_arg_string(my_argstr,match):
    arglist  = [x for x in my_argstr.split("--") if len(x)>0]
    for x in arglist:
        if match in x:
            return x
    return None
        

parser = argparse.ArgumentParser()
parser.add_argument("--use-ini",default=None,type=str,help="Pass ini file for parsing. Intended to reproduce lalinference_pipe functionality. Overrides most other arguments. Full path recommended")
parser.add_argument("--use-rundir",default=None,type=str,help="Intended to reproduce lalinference_pipe functionality. Must be absolute path.")
parser.add_argument("--use-online-psd-file",default=None,type=str,help="Provides specific online PSD file, so no downloads are needed")
parser.add_argument("--use-coinc",default=None,type=str,help="Intended to reproduce lalinference_pipe functionality")
parser.add_argument("--online",action='store_true')
parser.add_argument("--extra-args-helper",action=None, help="Filename with arguments for the helper. Use to provide alternative channel names and other advanced configuration (--channel-name, data type)!")
parser.add_argument("--manual-postfix",default='',type=str)
parser.add_argument("--gracedb-id",default=None,type=str)
parser.add_argument("--event-time",default=None,type=float,help="Event time. Intended to override use of GracedbID. MUST provide --manual-initial-grid ")
parser.add_argument("--calibration",default="C00",type=str)
parser.add_argument("--playground-data",action='store_true', help="Passed through to helper_LDG_events, and changes name prefix")
parser.add_argument("--approx",default=None,type=str,help="Approximant. REQUIRED")
parser.add_argument("--use-gwsurrogate",action='store_true',help="Attempt to use gwsurrogate instead of lalsuite.")
parser.add_argument("--l-max",default=2,type=int)
parser.add_argument("--no-matter",action='store_true', help="Force analysis without matter. Really only matters for BNS")
parser.add_argument("--assume-matter",action='store_true', help="Force analysis *with* matter. Really only matters for BNS")
parser.add_argument("--assume-highq",action='store_true', help="Force analysis with the high-q strategy, neglecting spin2. Passed to 'helper'")
parser.add_argument("--assume-well-placed",action='store_true',help="If present, the code will adopt a strategy that assumes the initial grid is very well placed, and will minimize the number of early iterations performed. Not as extrme as --propose-flat-strategy")
parser.add_argument("--internal-correlate-default",action='store_true',help='Force joint sampling in mc,delta_mc, s1z and possibly s2z')
parser.add_argument("--internal-flat-strategy",action='store_true',help="Use the same CIP options for every iteration, with convergence tests on.  Passes --test-convergence, ")
parser.add_argument("--add-extrinsic",action='store_true')
parser.add_argument("--fmin",default=20,type=int,help="Mininum frequency for integration. template minimum frequency (we hope) so all modes resolved at this frequency")  # should be 23 for the BNS
parser.add_argument("--fmin-template",default=None,type=float,help="Mininum frequency for template. If provided, then overrides automated settings for fmin-template = fmin/Lmax")  # should be 23 for the BNS
parser.add_argument("--data-LI-seglen",default=None,type=int,help="If specified, passed to the helper. Uses data selection appropriate to LI. Must specify the specific LI seglen used.")
parser.add_argument("--choose-data-LI-seglen",action='store_true')
parser.add_argument("--fix-bns-sky",action='store_true')
parser.add_argument("--cip-sampler-method",type=str,default=None)
parser.add_argument("--cip-fit-method",type=str,default=None)
parser.add_argument("--ile-jobs-per-worker",type=int,default=20)
parser.add_argument("--ile-no-gpu",action='store_true')
parser.add_argument("--spin-magnitude-prior",default='default',type=str,help="options are default [volumetric for precessing,uniform for aligned], volumetric, uniform_mag_prec, uniform_mag_aligned, zprior_aligned")
parser.add_argument("--force-chi-max",default=None,type=float,help="Provde this value to override the value of chi-max provided") 
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
parser.add_argument("--cip-quadratic-first",action='store_true')
parser.add_argument("--manual-initial-grid",default=None,type=str,help="Filename (full path) to initial grid. Copied into proposed-grid.xml.gz, overwriting any grid assignment done here")
parser.add_argument("--manual-extra-ile-args",default=None,type=str,help="Avenue to adjoin extra ILE arguments.  Needed for unusual configurations (e.g., if channel names are not being selected, etc)")
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--use-osg",action='store_true',help="Restructuring for ILE on OSG. The code by default will use CVMFS")
parser.add_argument("--use-osg-file-transfer",action='store_true',help="Restructuring for ILE on OSG. The code will NOT use CVMFS, and instead will try to transfer the frame files.")
parser.add_argument("--condor-local-nonworker",action='store_true',help="Provide this option if job will run in non-NFS space. ")
parser.add_argument("--use-osg-simple-requirements",action='store_true',help="Provide this option if job should use a more aggressive setting for OSG matching ")
parser.add_argument("--archive-pesummary-label",default=None,help="If provided, creates a 'pesummary' directory and fills it with this run's final output at the end of the run")
parser.add_argument("--archive-pesummary-event-label",default="this_event",help="Label to use on the pesummary page itself")
opts=  parser.parse_args()


if opts.assume_highq:
    opts.internal_correlate_default=True
event_dict={}

if (opts.approx is None) and not (opts.use_ini is None):
    config = ConfigParser.ConfigParser()
    config.read(opts.use_ini)
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
if opts.use_ini:
    base_dir =''  # all directories are provided as full path names


if opts.choose_data_LI_seglen:
    cmd_event = "gracedb_legacy download " + opts.gracedb_id + " coinc.xml"
    os.system(cmd_event)
    event_dict = retrieve_event_from_coinc("coinc.xml")
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
if opts.approx == "SEOBNRv3" or opts.approx == "NRSur7dq2" or opts.approx == "NRSur7dq4" or (opts.approx == 'SEOBNv3_opt') or (opts.approx == 'IMRPhenomPv2') or (opts.approx =="SEOBNRv4P" ) or (opts.approx == "SEOBNRv4PHM") or ('SpinTaylor' in opts.approx):
        is_analysis_precessing=True

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
    config = ConfigParser.ConfigParser()
    config.read(opts.use_ini)
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



helper_psd_args = ''
srate=4096  # default, built into helper, unwise to go lower, LI will almost never do higher
if opts.make_bw_psds:
    helper_psd_args += " --assume-fiducial-psd-files --fmax " + str(srate/2-1)


# Create provenance info : we want run to be reproducible
if True:
        os.mkdir("reproducibility")
        # Write this script and its arguments
        import shutil, json
#        thisfile = os.path.realpath(__file__)
#        shutil.copyfile(thisfile, "reproducibility/the_script_used.py")
        argparse_dict = vars(opts)
        with open("reproducibility/the_arguments_used.json",'w') as f:
                json.dump(argparse_dict,f)
        # Write commits
        cmd = "(cd ${ILE_CODE_PATH}; git rev-parse HEAD) > reproducibility/RIFT.commit"
        os.system(cmd)
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
cmd = " helper_LDG_Events.py --force-notune-initial-grid   --propose-fit-strategy --propose-ile-convergence-options --propose-initial-grid --fmin " + str(fmin) + " --fmin-template " + str(fmin_template) + " --working-directory " + base_dir + "/" + dirname_run  + helper_psd_args  + " --no-enforce-duration-bound "
if not(opts.gracedb_id is None) and (opts.use_ini is None):
    cmd +=" --use-legacy-gracedb --gracedb-id " + gwid 
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
if opts.assume_matter:
        cmd += " --assume-matter "
        npts_it = 1000
if opts.assume_well_placed:
    cmd += " --assume-well-placed "
if opts.assume_highq:
    cmd+= ' --assume-highq  --force-grid-stretch-mc-factor 2'  # the mc range, tuned to equal-mass binaries, is probably too narrow. Workaround until fixed in helper
    npts_it =1000
#if is_event_bns and not opts.no_matter:
#        cmd += " --assume-matter "
#        npts_it = 1000
if is_analysis_precessing:
        cmd += " --assume-precessing-spin "
        npts_it = 1500
if opts.internal_flat_strategy:
    cmd +=  " --test-convergence --propose-flat-strategy "
if opts.use_osg:
    cmd += " --use-osg "
    cmd += " --use-cvmfs-frames "  # only run with CVMFS data, otherwise very very painful
if opts.use_ini:
    cmd += " --use-ini " + opts.use_ini
    cmd += " --sim-xml target_params.xml.gz --event 0 "
    cmd += " --event-time " + str(event_dict["tref"])
    #
else:
    cmd += " --calibration-version " + opts.calibration 
if opts.use_online_psd_file:
    # Get IFO list from ini file
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(opts.use_ini)
    ifo_list = eval(config.get('analysis','ifos'))
    # Create command line arguments for those IFOs, so helper can correctly pass then downward
    for ifo in ifo_list:
        cmd+= " --psd-file {}={}".format(ifo,opts.use_online_psd_file)
if "SNR" in event_dict:
    cmd += " --hint-snr {} ".format(event_dict["SNR"])

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
if (opts.use_ini is None):
    line += " --d-max " + str(dmax_guess)
sur_location_prefix = "my_surrogates/nr_surrogates/"
if opts.use_osg:
    sur_location_prefix = "/"
if not 'NR' in opts.approx:
        line += " --approx " + opts.approx
elif opts.use_gwsurrogate and 'NRHybSur' in opts.approx:
        line += " --rom-group {} --rom-param NRHybSur3dq8.h5 --approx {} ".format(sur_location_prefix,opts.approx)
elif opts.use_gwsurrogate and "NRSur7d2" in opts.approx:
        line += " --rom-group {} --rom-param NRSur7dq2.h5 --approx {}  ".format(sur_location_prefix,opts.approx)
elif opts.use_gwsurrogate and "NRSur7d4" in opts.approx:
        line += " --rom-group {} --rom-param NRSur7dq4.h5  --approx {}".format(sur_location_prefix,opts.approx)
elif ("SEOBNR" in opts.approx) or ("NRHybSur" in opts.approx) or ("NRSur7d" in opts.approx): 
        line += " --approx " + opts.approx
else:
        print( " Unknown approx ", opts.approx)
        sys.exit(1)
if not(opts.manual_extra_ile_args is None):
    line += opts.manual_extra_ile_args
if not(opts.ile_runtime_max_minutes is None):
    lines += " --ile-runtime-max-minutes {} ".format(opts.ile_runtime_max_minutes)
with open('args_ile.txt','w') as f:
        f.write(line)
os.system("cp helper_test_args.txt args_test.txt")

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
    n_iterations += int(instructions_cip[indx][0])
    line = ' ' .join(instructions_cip[indx])
    n_max_cip = 10000000;
    if opts.cip_sampler_method is "GMM":
        n_max_cip *=10   # it is faster, so run longer; helps with correlated-sampling cases
    line +=" --n-output-samples 10000 --n-eff 10000 --n-max {}   --downselect-parameter m2 --downselect-parameter-range [1,1000] ".format(n_max_cip)
    if not(opts.cip_fit_method is None):
        line = line.replace('--fit-method gp', '--fit-method ' + opts.cip_fit_method)
    if not (opts.cip_sampler_method is None):
        line += " --sampler-method "+opts.cip_sampler_method
    line += prior_args_lookup[opts.spin_magnitude_prior]
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
        line += " --sampler-method GMM --internal-correlate-parameters 'mc,delta_mc,s1z,s2z' "
    if opts.approx in lalsimutils.waveform_approx_limit_dict:
        chi_max = lalsimutils.waveform_approx_limit_dict[opts.approx]["chi-max"]
        if not(opts.force_chi_max is None):
            chi_max = opts.force_chi_max
        q_min = lalsimutils.waveform_approx_limit_dict[opts.approx]["q-min"]
        eta_min = q_min/(1+q_min)**2
        line += " --chi-max {}  "
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
    if opts.fit_save_gp:
        line += " --fit-save-gp my_gp "  # fiducial filename, stored in each iteration
    line += "\n"
    lines.append(line)

if opts.cip_quadratic_first:
    lines[0]=lines[0].replace(' --fit-method gp ', ' --fit-method quadratic ')
    lines[0]=lines[0].replace(' --parameter delta_mc ', ' --parameter eta ')   # almost without fail we are using mc, delta_mc, xi  as zeroth layer

with open("args_cip_list.txt",'w') as f: 
   for line in lines:
           f.write(line)

# Write test file
with open("args_test.txt",'w') as f:
        f.write("X --always-succeed --method lame  --parameter m1")


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
if opts.assume_highq:
    puff_params = puff_params.replace(' delta_mc ', ' eta ')  # use natural coordinates in the high q strategy. May want to do this always
    puff_max_it +=3
with open("args_puff.txt",'w') as f:
        puff_args = puff_params + " --downselect-parameter chi1 --downselect-parameter-range [0,1] --downselect-parameter chi2 --downselect-parameter-range [0,1] "
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

# Overwrite grid if needed
if not (opts.manual_initial_grid is None):
    shutil.copyfile(opts.manual_initial_grid, "proposed-grid.xml.gz")

# Build DAG
cip_mem  = 30000
n_jobs_per_worker=opts.ile_jobs_per_worker
if opts.cip_fit_method == 'rf':  # much lower memory requirement
    cip_mem = 4000
cmd ="create_event_parameter_pipeline_BasicIteration  --ile-n-events-to-analyze {} --input-grid proposed-grid.xml.gz --ile-exe  `which integrate_likelihood_extrinsic_batchmode`   --ile-args args_ile.txt --cip-args-list args_cip_list.txt --test-args args_test.txt --request-memory-CIP {} --request-memory-ILE 4096 --n-samples-per-job ".format(n_jobs_per_worker,cip_mem) + str(npts_it) + " --working-directory `pwd` --n-iterations " + str(n_iterations) + " --n-copies 1" + " --puff-exe `which util_ParameterPuffball.py` --puff-cadence 1 --puff-max-it " + str(puff_max_it)+ " --puff-args args_puff.txt  --ile-retries "+ str(opts.ile_retries) + " --general-retries " + str(opts.general_retries)
if not(opts.ile_no_gpu):
    cmd +=" --request-gpu-ILE "
if opts.add_extrinsic:
    cmd += " --last-iteration-extrinsic --last-iteration-extrinsic-nsamples 8000 "
if opts.cip_explode_jobs:
   cmd+= " --cip-explode-jobs  " + str(opts.cip_explode_jobs)
   if not(opts.cip_fit_method is None) and not(opts.cip_fit_method == 'gp'):
       # if we are not using default GP fit, so all fit instances are equal
       cmd += " --cip-explode-jobs-flat "  
if opts.make_bw_psds:
    cmd+= " --use-bw-psd --bw-exe `which BayesWave` --bw-post-exe `which BayesWavePost` "
if opts.use_osg:
    cmd += " --use-osg --use-singularity  --cache-file local.cache  "   # run on the OSG, make sure to get frames (rather than try to transfer them).  Note with CVMFS frames we need to provide the cache, but that SHOULD be added to the arg list by the helper already.  However, the argument is needed to avoid failure.
    if not(opts.use_osg_file_transfer):
        cmd += " --use-cvmfs-frames "
    else:  # attempt to make copies of frame files, and set up to transfer them with *every* job (!)
        os.system("util_ForOSG_MakeLocalFramesDir.sh local.cache")
        os.system("echo frames_dir >> helper_transfer_files.txt")
    cmd+= " --transfer-file-list  "+base_dir+"/"+dirname_run+"/helper_transfer_files.txt"
if opts.condor_local_nonworker:
    cmd += " --condor-local-nonworker "
if opts.use_osg_simple_requirements:
    cmd += " --use-osg-simple-reqirements "
if opts.archive_pesummary_label:
#    cmd += " --plot-exe `which summarypages` --plot-args  args_plot.txt "
    cmd += " --plot-exe summarypages --plot-args  args_plot.txt "
print(cmd)
os.system(cmd)
