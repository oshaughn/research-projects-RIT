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
import shlex
import subprocess
import sys
import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
import configparser as ConfigParser

if ( 'RIFT_LOWLATENCY'  in os.environ):
    assume_lowlatency = True
else:
    assume_lowlatency=False

# ----------------------------------------------------------------------
# Hyperpipeline ASCII grid format (opt-in via env var).  When set, every
# grid file pseudo_pipe touches (target_params, proposed-grid, the
# --input-grid handed to BasicIteration, and the --sim-xml command-single
# sanity-check invocation) is .dat instead of .xml.gz, and the
# downstream pipeline runs in hyperpipeline mode end-to-end.  When unset,
# behaviour is identical to the legacy XML pipeline.
#
# By design, pseudo_pipe does NOT convert formats internally -- the
# entire process operates cohesively in one mode or the other.  In
# hyperpipeline mode the user is responsible for staging any external
# inputs (e.g. --manual-initial-grid) as hyperpipeline .dat; the
# auto-generated AMR / template-bank seed-grid paths still emit XML and
# will fail downstream unless --manual-initial-grid is supplied.
# ----------------------------------------------------------------------
_use_hpip_pp = str(os.environ.get("RIFT_HYPERPIPELINE_FORMAT", "")).strip().lower() in ("1", "true", "yes", "on")
grid_suffix_pp = "dat" if _use_hpip_pp else "xml.gz"
sim_grid_flag_pp = "--sim-grid" if _use_hpip_pp else "--sim-xml"
if _use_hpip_pp:
    print(" === pseudo_pipe: hyperpipeline ASCII grid format active (RIFT_HYPERPIPELINE_FORMAT) ===")
    print("     Inter-stage grids will be .{}, command-single will use {}".format(grid_suffix_pp, sim_grid_flag_pp))

# Backward compatibility
from RIFT.misc.dag_utils_generic import which
from RIFT.misc.cip_pipeline import flag_final_group_unique
# leaf module: numpy only, so this does not drag numba/cupy into the pipeline script
from RIFT.likelihood.time_interp_choice import (
    BARE_FLAG_SENTINEL, CROSSOVER_GUIDANCE, resolve_interpolate_time_request)
ligolw_prefix = 'igwn_'
if not(which(ligolw_prefix + "ligolw_add")):
    ligolw_prefix = ''


    
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
    from igwn_ligolw import lsctables, table, utils
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
    if hasattr(row, 'alpha4'):
        event_dict["eccentricity"] = row.alpha4
        if hasattr(row, 'alpha'):
            event_dict["meanPerAno"] = row.alpha
        else:
            event_dict["meanPerAno"] = None
    else:
        event_dict["eccentricity"] = None
        event_dict["meanPerAno"] = None
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
def unsafe_parse_arg_string_dict(my_argstr):
    arglist  = [x for x in my_argstr.split("--") if len(x)>0]
    dict_return = {}
    for x in arglist:
        net = x.split(' ')
        if len(net)>0:
            dict_return[net[0]] = net[1]
    return dict_return


def _lisa_data_products_from_ini(opts):
    """Fill the LISA data-product opts (channels / PSD files) from the conventional
    production-ini sections so a LISA run can be driven by --use-ini like the
    ground-based path.  Scalars and lisa-* algorithm options are already populated
    by the generic [rift-pseudo-pipe] parser (any CLI arg, by name); here we only
    translate the per-channel *dict* products that don't map cleanly to a flat key:

      [data]         channels = {'A': 'fake_strain', 'E': ..., 'T': ...}
      [lalinference] psds     = {'A': 'A_psd.xml.gz', ...}

    Values already set (e.g. via [rift-pseudo-pipe] lisa-channel-name) win, so the
    flat CLI surface still overrides.  Read-only; no LDG data-find is invoked.
    """
    import configparser as _CfgP
    cfg = _CfgP.ConfigParser()
    cfg.optionxform = str
    cfg.read(opts.use_ini)

    def _dict_to_assignments(section, key):
        if not cfg.has_option(section, key):
            return None
        mapping = eval(cfg.get(section, key))
        return ["{}={}".format(ifo, val) for ifo, val in mapping.items()]

    if not opts.lisa_channel_name:
        opts.lisa_channel_name = _dict_to_assignments("data", "channels")
    if not opts.lisa_psd_file:
        opts.lisa_psd_file = _dict_to_assignments("lalinference", "psds")


def run_lisa_known_sky_surface(opts):
    if opts.approx is None:
        print(" --lisa-known-sky requires --approx ")
        sys.exit(1)
    if opts.use_ini is not None:
        # LISA production-ini path: scalars/algorithm options come from the
        # generic [rift-pseudo-pipe] parser; fill the per-channel data products
        # (channels, PSDs) from the conventional [data]/[lalinference] sections.
        _lisa_data_products_from_ini(opts)

    bin_dir = os.path.dirname(os.path.abspath(__file__))
    helper = os.path.join(bin_dir, "helper_LISA_Events.py")
    cepp = os.path.join(bin_dir, "create_event_parameter_pipeline_BasicIteration")
    ile = os.path.join(bin_dir, "integrate_likelihood_extrinsic_batchmode_lisa")

    if opts.use_rundir:
        workdir = os.path.abspath(opts.use_rundir)
    else:
        event_label = "manual_" + format_gps_time(opts.event_time)
        sky_label = "variable_sky" if opts.lisa_vary_sky else "known_sky"
        workdir = os.path.abspath(
            event_label + "_LISA_" + opts.approx + "_" + sky_label + opts.manual_postfix
        )
    os.makedirs(workdir, exist_ok=False)

    helper_cmd = [
        sys.executable,
        helper,
        "--working-directory",
        workdir,
        "--input-grid",
        "proposed-grid.dat",
        "--approximant",
        opts.approx,
        "--l-max",
        str(opts.l_max),
        "--event-time",
        format_gps_time(opts.event_time),
        "--cache-file",
        opts.lisa_cache_file,
        "--ecliptic-longitude",
        str(opts.ecliptic_longitude),
        "--ecliptic-latitude",
        str(opts.ecliptic_latitude),
        "--fmin-template",
        str(opts.lisa_fmin_template),
        "--fmax",
        str(opts.lisa_fmax),
        "--reference-freq",
        str(opts.lisa_reference_freq),
        "--srate",
        str(opts.lisa_srate),
        "--data-integration-window-half",
        str(opts.lisa_data_integration_window_half),
        "--grid-size",
        str(opts.lisa_grid_size),
        "--grid-fractional-width",
        str(opts.lisa_grid_fractional_width),
        "--sky-grid-width",
        str(opts.lisa_sky_grid_width),
        "--n-iterations",
        str(opts.lisa_n_iterations),
        "--n-samples-per-job",
        str(opts.lisa_n_samples_per_job),
        "--request-memory-ILE",
        str(opts.internal_ile_request_memory),
        "--request-memory-CIP",
        str(opts.internal_cip_request_memory or 4096),
    ]
    if opts.lisa_vary_sky:
        helper_cmd.append("--vary-sky")
    if opts.lisa_zero_likelihood:
        helper_cmd.append("--zero-likelihood")
    for assignment in opts.lisa_channel_name or []:
        helper_cmd.extend(["--channel-name", assignment])
    for assignment in opts.lisa_psd_file or []:
        helper_cmd.extend(["--psd-file", assignment])
    if opts.extra_args_helper:
        with open(opts.extra_args_helper) as extra:
            helper_cmd.extend(shlex.split(extra.read()))

    print(" LISA known-sky helper command: ", " ".join(shlex.quote(x) for x in helper_cmd))
    subprocess.run(helper_cmd, check=True)

    if opts.lisa_skip_cepp_render:
        print(" LISA helper bundle written in {}".format(workdir))
        return

    env = os.environ.copy()
    env["RIFT_HYPERPIPELINE_FORMAT"] = "1"
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.path.abspath(os.path.join(bin_dir, "..")) + os.pathsep + env.get("PYTHONPATH", "")
    cepp_cmd = [
        sys.executable,
        cepp,
        "--ile-n-events-to-analyze",
        "1",
        "--input-grid",
        os.path.join(workdir, "proposed-grid.dat"),
        "--ile-exe",
        ile,
        "--ile-args",
        os.path.join(workdir, "args_ile.txt"),
        "--cip-args-list",
        os.path.join(workdir, "args_cip_list.txt"),
        "--test-args",
        os.path.join(workdir, "args_test.txt"),
        "--working-directory",
        workdir,
        "--n-iterations",
        str(opts.lisa_n_iterations),
        "--n-samples-per-job",
        str(opts.lisa_n_samples_per_job),
        "--n-copies",
        str(opts.ile_copies),
        "--request-memory-ILE",
        str(opts.internal_ile_request_memory),
        "--request-memory-CIP",
        str(opts.internal_cip_request_memory or 4096),
        "--transfer-file-list",
        os.path.join(workdir, "helper_transfer_files.txt"),
    ]
    # Container: let write_ILE_sub_simple emit the singularity + file-transfer
    # wiring (the LDG path's native mechanism) rather than any LISA-specific code.
    # Needs SINGULARITY_RIFT_IMAGE (+ SINGULARITY_BASE_EXE_DIR) in the env.
    if opts.lisa_use_singularity:
        # write_ILE_sub_simple's singularity path requires the CEPP's --cache-file
        # to be set (else "Need to specify frames_dir or cache_file to use
        # singularity"); the LISA cache is otherwise only inside the ILE args.
        cepp_cmd += ["--use-singularity", "--cache-file", opts.lisa_cache_file]
    # Puffball between iterations: perturb the (very tight) CIP posterior so the
    # next grid is not a near-degenerate cluster (else the CIP refit diverges).
    if opts.lisa_n_iterations > 1 and not opts.lisa_no_puff:
        cepp_cmd += [
            "--puff-exe", os.path.join(bin_dir, "util_ParameterPuffball.py"),
            "--puff-args", os.path.join(workdir, "args_puff.txt"),
            "--puff-cadence", "1",
            "--puff-max-it", str(opts.lisa_n_iterations),
        ]
    # LISA reflected-sky-mode (vary-sky): reflect the grid at one iteration to
    # explore the secondary sky mode (latitude bimodality).
    if opts.lisa_search_reflected_sky_mode and opts.lisa_n_iterations > 1:
        cepp_cmd += [
            "--search-reflected-sky-mode",
            "--reflected-sky-mode-exe", os.path.join(bin_dir, "convert_primary_sky_mode_to_secondary"),
            "--lisa-reference-time", str(opts.lisa_reference_time),
        ]
        if opts.lisa_search_reflected_sky_mode_iteration is not None:
            cepp_cmd += ["--search-reflected-sky-mode-iteration", str(opts.lisa_search_reflected_sky_mode_iteration)]
    print(" LISA known-sky CEPP command: ", " ".join(shlex.quote(x) for x in cepp_cmd))
    subprocess.run(cepp_cmd, check=True, cwd=workdir, env=env)
    print(" LISA known-sky CEPP surface rendered in {}".format(workdir))



parser = argparse.ArgumentParser()
parser.add_argument("--skip-reproducibility",action='store_true')
parser.add_argument("--use-production-defaults",action='store_true',help="Use production defaults. Intended for use with tools like asimov or by nonexperts who just want something to run on a real event.  Will require manual setting of other arguments!")
parser.add_argument("--use-subdags",action='store_true',help="Use CEPP_Alternate instead of CEPP_BasicIteration. Note this writes an adaptively-sized DAG each iteration, but doesn't otherwise optimize yet.")
parser.add_argument("--pipeline-builder",default=None,choices=["BasicIteration","AlternateIteration"],help="Explicitly select the create_event_parameter_pipeline_* iteration builder, as a drop-in hot-swap for side-by-side A/B testing. Overrides the implicit --use-subdags routing. If unset, the builder is chosen by --use-subdags (Alternate) vs. the default (Basic).")
parser.add_argument("--use-ile-subdags",action='store_true',help="Use ILE subdag system (new)")
parser.add_argument("--bilby-ini-file",default=None,type=str,help="Pass ini file for parsing. Intended to use for calibration reweighting. Full path recommended")
parser.add_argument("--bilby-pickle-file",default=None,type=str,help="Bilby Pickle file with event settings. Intended to use for calibration reweighting. Full path recommended")
parser.add_argument("--use-ini",default=None,type=str,help="Pass ini file for parsing. Intended to reproduce lalinference_pipe functionality. Overrides most other arguments. Full path recommended")
parser.add_argument("--use-rundir",default=None,type=str,help="Intended to reproduce lalinference_pipe functionality. Must be absolute path.")
parser.add_argument("--use-online-psd-file",default=None,type=str,help="Provides specific online PSD file, so no downloads are needed")
parser.add_argument("--use-coinc",default=None,type=str,help="Intended to reproduce lalinference_pipe functionality")
parser.add_argument("--manual-ifo-list",default=None,type=str,help="Overrides IFO list normally retrieve by event ID.  Use with care (e.g., glitch studies) or for events specified with --event-time.")
parser.add_argument("--online",action='store_true')
parser.add_argument("--calibration-reweighting",action='store_true',help="Option to add job to DAG to reweight posterior samples due to calibration uncertainty.")
parser.add_argument("--calibration-reweighting-batchsize",type=int,default=None,help="If not 'None', tries to group the final set of points based on jobs of a fixed size")
parser.add_argument("--calibration-reweighting-count",type=int,default=None,help="If not 'None', the number of calibration curves to request when marginalizing. Default is 100")
parser.add_argument("--calibration-reweighting-initial-extra-args",type=str,default=None,help="If not 'None', pass through. One argument targets effective sample size, other duplicates inoutput")
parser.add_argument("--calibration-reweighting-extra-args",type=str,default=None,help="If not 'None', pass through. One argument targets effective sample size, other duplicates inoutput")
parser.add_argument("--calibration-reweighting-osg",action='store_true',help="Attempt to use settings for OSG for cal reweighting. Remove after developed")
# In-loop calibration marginalization (inside the ILE GPU loop), as opposed to the
# postprocessing --calibration-reweighting path above.  Setting the envelope directory
# enables it and threads the corresponding flags into the ILE arguments (args_ile.txt).
parser.add_argument("--calmarg-envelope-directory",default=None,type=str,help="Enable IN-LOOP calibration marginalization in ILE. Directory with per-IFO calibration envelope files named <IFO>.txt (e.g. H1.txt, L1.txt, V1.txt). Threaded to ILE as --calibration-envelope-directory (absolute path).")
parser.add_argument("--calmarg-n-realizations",default=100,type=int,help="Number of calibration realizations for in-loop calmarg. Threaded to ILE as --calibration-n-realizations.")
parser.add_argument("--calmarg-spline-count",default=10,type=int,help="Number of spline nodes for in-loop calmarg envelopes. Threaded to ILE as --calibration-spline-count.")
parser.add_argument("--calmarg-fused-kernel",action='store_true',help="Use the fused GPU kernel (Option C) for in-loop calmarg. GPU only; ILE falls back to the loop method otherwise. Threaded to ILE as --calibration-fused-kernel.")
parser.add_argument("--calmarg-pilot",action='store_true',help="Option C adaptive calibration: add per-iteration cal PILOT jobs that learn a cal proposal (harvest top-lnL composite points -> ILE --calibration-dump-responsibilities -> fit+consolidate) and SEED the next iteration's wide ILE jobs via --calibration-proposal-breadcrumb. Requires --calmarg-envelope-directory.")
parser.add_argument("--calmarg-pilot-cadence",default=1,type=int,help="Run a cal pilot every n iterations (default 1).")
parser.add_argument("--calmarg-pilot-max-it",default=3,type=int,help="Stop launching cal pilots after this iteration (cal is boring; freeze once learned). Default 3.")
parser.add_argument("--calmarg-pilot-top-fraction",default=0.05,type=float,help="Fraction of highest-lnL composite points the pilot harvests. Default 0.05.")
parser.add_argument("--calmarg-pilot-max-points",default=32,type=int,help="Cap on harvested pilot points per iteration. Default 32.")
parser.add_argument("--calmarg-first-cip-sigma-cut",default=100.0,type=float,help="With --calmarg-pilot: relax the first CIP stage's --sigma-cut to this value, so cold-start (prior-cal) iteration-0 points -- which have large MC error -- are not all stripped by CIP's default 0.6.  Threaded to helper_LDG_Events.py. Default 100 (effectively keep all cold-start points).")
parser.add_argument("--calmarg-burn-in-neff",default=None,type=float,help="In-loop calmarg: burn the extrinsic sampler in on the cheap zero-cal likelihood to this n_eff before the full cal-marginalized integration (warm start; the extrinsic posterior is ~cal-independent). Threaded to ILE as --calibration-burn-in-neff.")
parser.add_argument("--calmarg-export-posterior",action='store_true',help="In-loop calmarg: at the final fairdraw export, also write the RECOVERED calibration posterior -- for each fair-draw sample, draw one cal realization in proportion to its posterior weight and write a self-contained sibling <output>_<event>_cal.dat with the full draw (intrinsic + extrinsic + cal_<IFO>_amp_<k>/cal_<IFO>_phase_<k> node columns). Threaded to ILE as --calibration-export-posterior (fires only at the extrinsic/fairdraw stage).")
parser.add_argument("--extrinsic-handoff",action='store_true',help="Extrinsic handoff (GMM sampler only): each iteration's wide ILE jobs write a per-event extrinsic GMM proposal (--extrinsic-proposal-output) of their extrinsic posterior; a per-iteration consolidation picks the most representative one and SEEDS the next iteration's wide ILE jobs via --extrinsic-proposal-breadcrumb, so the extrinsic sampler starts on the answer instead of cold.  Requires --ile-sampler-method GMM.  See RIFT/calmarg/DESIGN_extrinsic_handoff.md.")
parser.add_argument("--extrinsic-handoff-select",default="lnL",help="Metric the extrinsic consolidation ranks per-event proposals by (lnL|neff|n_samples). Default lnL (most peak-representative).")
parser.add_argument("--distance-reweighting",action='store_true',help="Option to add job to DAG to reweight posterior samples due to different distance prior (LVK prod prior)")
parser.add_argument("--extra-args-helper",action=None, help="Filename with arguments for the helper. Use to provide alternative channel names and other advanced configuration (--channel-name, data type)!")
parser.add_argument("--manual-postfix",default='',type=str)
parser.add_argument("--gracedb-id",default=None,type=str)
parser.add_argument("--gracedb-exe",default="gracedb")
parser.add_argument("--use-legacy-gracedb",action='store_true')
parser.add_argument("--internal-use-gracedb-bayestar",action='store_true',help="Retrieve BS skymap from gracedb (bayestar.fits), and use it internally in integration with --use-skymap bayestar.fits.")
parser.add_argument("--event-time",default=None,type=float,help="Event time. Intended to override use of GracedbID. MUST provide --manual-initial-grid ")
parser.add_argument("--lisa-known-sky",action='store_true',help="Use the LISA helper to build a known-sky LISA CEPP surface and exit. Avoids the LDG event helper path.")
parser.add_argument("--lisa-vary-sky",action='store_true',help="With --lisa-known-sky, treat ecliptic sky location as intrinsic rather than pinning --lisa-fixed-sky.")
parser.add_argument("--lisa-skip-cepp-render",action='store_true',help="With --lisa-known-sky, only write the helper bundle; do not render the CEPP DAG.")
parser.add_argument("--lisa-cache-file",default="lisa.cache",help="With --lisa-known-sky, cache file passed to the LISA ILE.")
parser.add_argument("--lisa-channel-name",action="append",default=None,help="With --lisa-known-sky, channel assignment such as A=fake_strain. May be repeated.")
parser.add_argument("--lisa-psd-file",action="append",default=None,help="With --lisa-known-sky, PSD assignment such as A=A_psd.xml.gz. May be repeated.")
parser.add_argument("--ecliptic-longitude",default=1.0,type=float,help="With --lisa-known-sky, fixed ecliptic longitude.")
parser.add_argument("--ecliptic-latitude",default=0.3,type=float,help="With --lisa-known-sky, fixed ecliptic latitude.")
parser.add_argument("--lisa-fmin-template",default=1.0e-3,type=float,help="With --lisa-known-sky, template low-frequency cutoff.")
parser.add_argument("--lisa-fmax",default=0.125,type=float,help="With --lisa-known-sky, high-frequency cutoff.")
parser.add_argument("--lisa-reference-freq",default=5.0e-3,type=float,help="With --lisa-known-sky, waveform reference frequency.")
parser.add_argument("--lisa-srate",default=0.25,type=float,help="With --lisa-known-sky, sample rate. Kept as float for long-duration LISA data.")
parser.add_argument("--lisa-data-integration-window-half",default=300.0,type=float,help="With --lisa-known-sky, half-width of the ILE data integration window.")
parser.add_argument("--lisa-no-puff",action="store_true",help="Disable the inter-iteration puffball for the known-sky LISA path.")
parser.add_argument("--lisa-search-reflected-sky-mode",action="store_true",help="LISA vary-sky: at one iteration, reflect the grid to the secondary sky mode (handles the LISA latitude bimodality).")
parser.add_argument("--lisa-search-reflected-sky-mode-iteration",default=None,type=int,help="Iteration to reflect the sky (default n_iterations-2).")
parser.add_argument("--lisa-reference-time",default=0.0,type=float,help="LISA coalescence/reference time (for the reflected-sky transform).")
parser.add_argument("--lisa-use-singularity",action="store_true",help="Forward --use-singularity to the CEPP for the known-sky LISA path. The container wiring + transfer is then emitted by write_ILE_sub_simple, exactly as for the LDG path; set SINGULARITY_RIFT_IMAGE (osdf:// staged image preferred, so dag_utils file-transfers it) and SINGULARITY_BASE_EXE_DIR (dir of the LISA ILE *inside* the image).")
parser.add_argument("--lisa-grid-size",default=3,type=int,help="With --lisa-known-sky, number of synthetic initial-grid points.")
parser.add_argument("--lisa-grid-fractional-width",default=1.0e-3,type=float,help="With --lisa-known-sky, fractional mass width for the initial grid.")
parser.add_argument("--lisa-sky-grid-width",default=0.02,type=float,help="With --lisa-known-sky --lisa-vary-sky, ecliptic sky half-step scale for the initial grid.")
parser.add_argument("--lisa-n-iterations",default=1,type=int,help="With --lisa-known-sky, CEPP iteration count.")
parser.add_argument("--lisa-n-samples-per-job",default=1,type=int,help="With --lisa-known-sky, CEPP samples per job.")
parser.add_argument("--lisa-zero-likelihood",action='store_true',help="With --lisa-known-sky, pass --zero-likelihood through to the LISA ILE args.")
parser.add_argument("--calibration",default="C00",type=str)
parser.add_argument("--playground-data",action='store_true', help="Passed through to helper_LDG_events, and changes name prefix")
parser.add_argument("--approx",default=None,type=str,help="Approximant. REQUIRED")
parser.add_argument("--use-gwsurrogate",action='store_true',help="Attempt to use gwsurrogate instead of lalsuite.")
parser.add_argument("--use-gwsignal",action='store_true',help="Attempt to use gwsignal interface.")
parser.add_argument("--l-max",default=2,type=int)
parser.add_argument("--no-matter",action='store_true', help="Force analysis without matter. Really only matters for BNS")
parser.add_argument("--assume-nospin",action='store_true', help="Force analysis with zero spin")
parser.add_argument("--assume-precessing",action='store_true', help="Force analysis *with* transverse spins")
parser.add_argument("--assume-nonprecessing",action='store_true', help="Force analysis *without* transverse spins")
parser.add_argument("--assume-matter",action='store_true', help="Force analysis *with* matter. Really only matters for BNS")
parser.add_argument("--assume-matter-eos",default=None,type=str, help="Force analysis *with* matter. Really only matters for BNS")
parser.add_argument("--assume-matter-conservatively",action='store_true',help="If present, the code will use the full prior range for exploration and sampling. [Without this option, the initial grid is limited to a physically plausible range in lambda-i")
parser.add_argument("--assume-matter-but-primary-bh",action='store_true',help="If present, the code will add options necessary to manage tidal arguments for the smaller body ONLY. (Usually pointless)")
parser.add_argument("--internal-tabular-eos-file",type=str,default=None,help="Tabular file of EOS to use.  The default prior will be UNIFORM in this table!")
parser.add_argument("--sample-eccentricity-squared",action='store_true', help="Option for sampling as well as fitting in eccentricity_squared instead of fitting in eccentricity_squared and sampling in eccentricity (also need option --use-eccentricity-squared")
parser.add_argument("--use-eccentricity-squared",action='store_true', help="Allows for fitting and sampling in eccentricity_squared instead of eccentricity")
parser.add_argument("--sample-eccentricity-ln",action='store_true', help="Option for sampling as well as fitting in eccentricity_ln instead of fitting in eccentricity_ln and sampling in eccentricity (also need option --use-eccentricity-ln")
parser.add_argument("--use-eccentricity-ln",action='store_true', help="Allows for fitting and sampling in eccentricity_ln instead of eccentricity")
parser.add_argument("--assume-eccentric",action='store_true', help="Add eccentric options for each part of analysis")
parser.add_argument("--use-meanPerAno",action='store_true', help="Add meanPerAno options for each part of analysis")
parser.add_argument("--internal-cip-use-periodic-ecc-vars",action='store_true', help="use e cos ell, e sin ell as fitting variables ")
parser.add_argument("--assume-lowlatency-tradeoffs",action='store_true', help="Force analysis with various low-latency tradeoffs (e.g., drop spin 2, use aligned, etc)")
parser.add_argument("--assume-highq",action='store_true', help="Force analysis with the high-q strategy, neglecting spin2. Passed to 'helper'")
parser.add_argument("--assume-well-placed",action='store_true',help="If present, the code will adopt a strategy that assumes the initial grid is very well placed, and will minimize the number of early iterations performed. Not as extrme as --propose-flat-strategy")
parser.add_argument("--ile-distance-prior",default=None,help="If present, passed through to the distance prior option.   If provided, BLOCKS distance marginalization")
parser.add_argument("--internal-ile-buffer-after-trigger",default=2,type=float,help="Provided to allow user to change time after trigger. NOT FULLY IMPLEMENTED")
parser.add_argument("--internal-ile-request-disk",help="Use if you are transferring large files, or if you otherwise expect a lot of data ")
parser.add_argument("--internal-cip-request-disk",help="Use if you are transferring large files, or if you otherwise expect a lot of data ")
parser.add_argument("--internal-general-request-disk",help="Use if you are transferring large files, or if you otherwise expect a lot of data. Specifically for things like calmarg/surrogate h5 files ")
parser.add_argument("--internal-ile-request-memory",default=4096,type=int,help="ILE memory request in Mb. Only experts should change this.")
parser.add_argument("--internal-ile-n-max",default=None,type=int,help="Set maximum number of evaluations each ILE worker uses. EXPERTS ONLY")
parser.add_argument("--internal-ile-inv-spec-trunc-time",default=None,type=float,help="Timescale of inverse spectrum truncation time. Default in pipeline is zero. Should be no more than 1/2 the segment length")
parser.add_argument("--internal-ile-data-tukey-window-time",default=None,type=float,help="Timescale of the tukey window (total, both sides)")
parser.add_argument("--internal-ile-psd-common-window",action='store_true',help="Default is to use the window shape correction on the input PSD (assumed to be scaled), and NOT to try to scale PSD.  Adding this option means we assume the PSD is not being window-corrected on input, so does not need rescaling. ")
parser.add_argument("--internal-ile-modify-taper",default=None,help="String provided modifies taper. If not provided TAPER_START will be used for all waveforms. Future-protecting for nonstandard tapering")
parser.add_argument("--internal-marginalize-distance",action='store_true',help="If present, the code will marginalize over the distance variable. Passed diretly to helper script. Default will be to generate d_marg script *on the fly*")
parser.add_argument("--internal-marginalize-distance-file",help="Filename for marginalization file.  You MUST make sure the max distance is set correctly")
parser.add_argument("--internal-distance-max",type=float,help="If present, the code will use this as the upper limit on distance (overriding the distance maximum in the ini file, or any other setting). *required* to use internal-marginalize-distance in most circumstances")
parser.add_argument("--internal-ile-check-good-enough",action='store_true', help=" IN PROGRESS: force creation of 'ile_good_enough' files in all ILE run directories, and adding to transfer_file_list")
parser.add_argument("--internal-correlate-default",action='store_true',help='Force joint sampling in mc,delta_mc, s1z and possibly s2z')
parser.add_argument("--internal-force-iterations",type=int,default=None,help="If integer provided, overrides internal guidance on number of iterations, attempts to force prolonged run. By default puts convergence tests on")
parser.add_argument("--internal-truncate-cip-arg-list",type=int, default=None, help="If integer provided, write only the last N lines of the cip_arg_list file. Recommended value is 1, to create extrinsic+calmarg only output. Other values can be used to disable the first few iterations of manual tuning, if initial grid is well-adapted")
parser.add_argument("--internal-test-convergence-threshold",type=float,default=None,help="The value of the threshold. 0.02 has been default. If not specified, left out of helper command line (where default is maintained) ")
parser.add_argument("--internal-flat-strategy",action='store_true',help="Use the same CIP options for every iteration, with convergence tests on.  Passes --test-convergence, ")
parser.add_argument("--internal-use-amr",action='store_true',help="Changes refinement strategy (and initial grid) to use. PRESENTLY WE CAN'T MIX AND MATCH AMR, CIP ITERATIONS, so this is fixed for the whole run right now; use continuation and 'fetch' to augment")
parser.add_argument("--internal-use-amr-bank",default="",type=str,help="Bank used for template")
parser.add_argument("--internal-use-amr-puff",action='store_true',help="Use puffball with AMR (as usual).  May help with stalling")
parser.add_argument("--internal-use-force-away",type=float,default=None,help="Specific force-away value")
parser.add_argument("--internal-use-aligned-phase-coordinates", action='store_true', help="If present, instead of using mc...chi-eff coordinates for aligned spin, will use SM's phase-based coordinates. Requires spin for now")
parser.add_argument("--internal-use-rescaled-transverse-spin-coordinates",action='store_true',help="If present, use coordinates which rescale the unit sphere with special transverse sampling")
parser.add_argument("--external-fetch-native-from",type=str,help="Directory name of run where grids will be retrieved.  Recommend this is for an ACTIVE run, or otherwise producing a large grid so the retrieved grid changes/isn't fixed")
parser.add_argument("--internal-propose-converge-last-stage",action='store_true',help="Pass through to helper")
parser.add_argument("--internal-n-iterations-subdag-max",default=10,type=int,help="Subdag convergence proposal max iterations option")
parser.add_argument("--internal-n-evaluations-per-iteration",default=None,type=int,help="Number of ILE evaluation points per iteration, if not set then pipeline selects experience-based default.  Each ILE worker will do a fraction of this total workload.")
parser.add_argument("--add-extrinsic",action='store_true')
parser.add_argument("--add-extrinsic-time-resampling",action='store_true',help="adds the time resampling option.  Only deployed for vectorized calculations (which should be all that end-users can access)")
parser.add_argument("--internal-ile-srate-time-resampling",default=None, help=" Adds --srate-resample-time-marginalization to ILE for  output, to provide higher-resolution time output ")
parser.add_argument("--internal-ile-srate-internal",default=None, help=" Adds --srate-internal to ILE, modifying how calculations are performed internally to use a higher sampling rate ")
parser.add_argument("--internal-ile-interpolate-time",nargs='?',const=BARE_FLAG_SENTINEL,default=None,type=str,help="Enable sub-sample interpolation of Q_lm at fractional detector arrival times in the maintained NoLoop likelihood. REQUIRES AN EXPLICIT STENCIL: nearest|cubic|sinc -- automatic selection was removed as measurably unreliable, and a bare flag is rejected rather than silently doing nothing. MEASURED GUIDANCE (SEOBNRv4, an IMR model): %s. Forwarded verbatim to helper_LDG_Events.py, which validates it. Full tables, limitations and provenance: RIFT/likelihood/DESIGN_q_window_stencil.md." % CROSSOVER_GUIDANCE)
parser.add_argument("--internal-ile-n-chunk",default=None,type=int,help="Override the extrinsic chunk size (--n-chunk) passed to ILE, via the helper. Default behaviour (helper): 40000, scaled linearly with SNR above 40 and capped at 160000, because at high SNR the posterior is a vanishing fraction of the prior volume and a small chunk gives few informative samples per adaptation step. Larger chunks cost GPU memory but measured HOST memory (what RequestMemory governs) is flat, so no memory-request change is normally needed. EXPERTS ONLY.")
parser.add_argument("--batch-extrinsic",action='store_true')
parser.add_argument("--fmin",default=20,type=int,help="Mininum frequency for integration. template minimum frequency (we hope) so all modes resolved at this frequency")  # should be 23 for the BNS
parser.add_argument("--fmin-template",default=None,type=float,help="Mininum frequency for template. If provided, then overrides automated settings for fmin-template = fmin/Lmax")  # should be 23 for the BNS
parser.add_argument("--data-LI-seglen",default=None,type=int,help="If specified, passed to the helper. Uses data selection appropriate to LI. Must specify the specific LI seglen used.")
parser.add_argument("--choose-data-LI-seglen",action='store_true')
parser.add_argument("--fix-bns-sky",action='store_true')
parser.add_argument("--declination",default=0.1,type=float)
parser.add_argument("--right-ascension",default=0.57,type=float)
parser.add_argument("--ile-sampler-method",type=str,default=None)
parser.add_argument("--ile-n-eff",type=int,default=None,help="ILE n_eff passed to helper/downstream. Default internally is 50; lower is faster but less accurate, going much below 10 could be dangerous ")
parser.add_argument("--cip-sampler-method",type=str,default=None)
parser.add_argument("--cip-sampler-portfolio-list",type=str,default=None,help="if sampler-method==portfolio, string-separated list of options. Goes into --sampler-portfolio array in CIP argument list ")
parser.add_argument("--cip-sampler-oracle-list",type=str,default=None,help="if sampler-method==portfolio, string-separated list of options from [RS,Climb]. Goes into --sampller-oracle array in CIP argument list. Note if you have supplementary arguments like --sampler-oracle-args, -oracle-reference-sample-file, --oracle-reference-sample-params, you need to pass these with manual-extra-cip-args ")
parser.add_argument("--cip-fit-method",type=str,default=None)
parser.add_argument("--cip-internal-use-eta-in-sampler",action='store_true', help="Use 'eta' as a sampling parameter. Designed to make GMM sampling behave particularly nicely for objects which could be equal mass")
parser.add_argument("--ile-jobs-per-worker",type=int,default=None,help="Default will be 20 per worker usually for moderate-speed approximants, and more for very fast configurations")
parser.add_argument("--ile-jobs-per-worker-first",type=int,default=None,help="Default size for initial iteration, usually 2* number used for others")
parser.add_argument("--ile-no-gpu",action='store_true')
parser.add_argument("--ile-xpu",action='store_true',help='Request ILE run on both GPU and CPU. Disables ile_force_gpu, if provided!')
parser.add_argument("--ile-force-gpu",action='store_true')
parser.add_argument("--ile-gpu-fanout",default=None,help="Multi-GPU ILE fan-out: split each ILE batch's intrinsic-grid range across N GPUs on the node (one shard per GPU).  Integer N (also requests N GPUs+CPUs) or 'auto' (split across whatever GPUs are visible at runtime).  Baked into the generated ile_pre.sh, so it needs no runtime environment.  Equivalent to setting RIFT_ILE_GPU_FANOUT.  Requires --ile-force-gpu.")
parser.add_argument("--fake-data-cache",type=str)
parser.add_argument("--spin-magnitude-prior",default='default',type=str,help="options are default [uniform mag for precessing, zprior for aligned], volumetric, uniform_mag_prec, uniform_mag_aligned, zprior_aligned")
parser.add_argument("--eccentricity-prior",default='uniform',type=str,choices=['uniform','log_uniform'],help="options are uniform in e ('uniform') and uniform in log(e) ('log_uniform')")  # constrained: the value is forwarded verbatim to CIP, which only branches on the exact string 'log_uniform', so an unrecognized value here would silently run the uniform prior instead of failing
parser.add_argument("--force-lambda-max",default=None,type=float,help="Provide this value to override the value of lambda-max provided") 
parser.add_argument("--force-lambda-small-max",default=None,type=float,help="Provide this value to override the value of lambda-small-max provided") 
parser.add_argument("--force-lambda-no-linear-init",action='store_true',help="Disables use of priors focused towards small lambda for initial iterations. Designed for PP plot tests with wide/uniform priors.")
parser.add_argument("--force-chi-max",default=None,type=float,help="Provide this value to override the value of chi-max provided") 
parser.add_argument("--force-chi-small-max",default=None,type=float,help="Provide this value to override the value of chi-max provided") 
parser.add_argument("--force-ecc-max",default=None,type=float,help="Provide this value to override the value of ecc-max provided")
parser.add_argument("--force-ecc-min",default=None,type=float,help="Provide this value to override the value of ecc-min provided")
parser.add_argument("--force-comp-max",default=1000,type=float,help="Provide this value to override the value of the max component mass in CIP provided")
parser.add_argument("--force-comp-min",default=1,type=float,help="Provide this value to override the value of min component mass in CIP provided")
parser.add_argument("--force-meanPerAno-max",default=None,type=float,help="Provide this value to override the value of meanPerAno-max provided")
parser.add_argument("--force-meanPerAno-min",default=None,type=float,help="Provide this value to override the value of meanPerAno-min provided")
parser.add_argument("--scale-mc-range",type=float,default=None,help="If using the auto-selected mc, scale the ms range proposed by a constant factor. Recommend > 1. . ini file assignment will override this.")
parser.add_argument("--limit-mc-range",default=None,type=str,help="Pass this argumen through to the helper to set the mc range")
parser.add_argument("--force-mc-range",default=None,type=str,help="Pass this argumen through to the helper to set the mc range")
parser.add_argument("--force-eta-range",default=None,type=str,help="Pass this argumen through to the helper to set the eta range")
parser.add_argument("--allow-subsolar", action='store_true', help="Override limits which otherwise prevent subsolar mass PE")
parser.add_argument("--force-hint-snr",default=None,type=str,help="Pass this argumen through to the helper to control source amplitude effects")
parser.add_argument("--force-initial-grid-size",default=None,type=float,help="Only used for automated grids.  Passes --force-initial-grid-size down to helper")
parser.add_argument("--hierarchical-merger-prior-1g",action='store_true',help="As in 1903.06742")
parser.add_argument("--hierarchical-merger-prior-2g",action='store_true',help="As in 1903.06742")
parser.add_argument("--link-reference-pe",action='store_true',help="If present, creates a directory 'reference_pe' and adds symbolic links to fiducial samples. These can be used by the automated plotting code.  Requires LVC_PE_SAMPLES environment variable defined!")
parser.add_argument("--link-reference-psds",action='store_true',help="If present, uses the varialbe LVC_PE_CONFIG to find a 'reference_pe_config_map.dat' file, which provides the location for reference PSDs.  Will override PSDs used / setup by default")
parser.add_argument("--make-bw-psds",action='store_true',help='If present, adds nodes to create BW PSDs to the dag.  If at all possible, avoid this and re-use existing PSDs')
parser.add_argument("--link-bw-psds",action='store_true',help='If present, uses the script retrieve_bw_psd_for_event.sh  to find a precomputed BW psd, and convert it to our format')
parser.add_argument("--use-online-psd",action='store_true', help="If present, will use the online PSD estimates")
parser.add_argument("--ile-copies",default=1,type=int)
parser.add_argument("--ile-retries",default=3,type=int)
parser.add_argument("--general-retries",default=3,type=int)
parser.add_argument("--ile-runtime-max-minutes",default=None,type=int,help="If not none, kills ILE jobs that take longer than the specified integer number of minutes. Do not use unless an expert")
parser.add_argument("--fit-save-gp",action="store_true",help="If true, pass this argument to CIP. GP plot for each iteration will be saved. Useful for followup investigations or reweighting. Warning: lots of disk space (1G or so per iteration)")
parser.add_argument("--cip-explode-jobs",type=int,default=None)
parser.add_argument("--cip-explode-jobs-last",type=int,default=None,help="Number of jobs to use in last stage.  Hopefully in future auto-set")
parser.add_argument("--cip-explode-jobs-auto",action='store_true',help="Auto-select --cip-explode-jobs based on SNR. Changes both cip-explode-jobs and cip-explode-jobs-last")
parser.add_argument("--cip-explode-jobs-auto-scale",type=float,default=None,help="Scales up number of jobs requested by cip-explode-jobs-auto")
parser.add_argument("--cip-explode-jobs-dag",type=float,default=None,help="Uses subdag for CIP, with many retries - adaptively will terminate at target work level")
parser.add_argument("--cip-quadratic-first",action='store_true')
parser.add_argument("--cip-sigma-cut",default=None,type=float,help="sigma-cut is an error threshold for CIP.  Passthrough")
parser.add_argument("--n-output-samples",type=int,default=5000,help="Number of output samples generated in the interim iteration")
parser.add_argument("--n-output-samples-last",type=int,default=20000,help="Number of output samples generated in the final iteration")
parser.add_argument("--internal-last-iteration-extrinsic-samples-per-ile",default=5,type=int,help="Draw this many samples from each ILE job")
parser.add_argument("--internal-last-iteration-extrinsic-samples-per-ile-internal",default=10,type=int,help="Draw this many samples from each ILE job")
parser.add_argument("--internal-cip-cap-neff",type=int,default=500,help="Largest value for CIP n_eff to use for *non-final* iterations. ALWAYS APPLIED. ")
# --- Alt config: resolve transverse-spin (chi1_perp) tails, esp. low mass (transverse-spin study) ---
# The interim CIP posterior is the COMBINATION of the cip-explode-jobs worker cohort; its NET
# effective-sample count (not any single worker's n_eff) is what resolves the transverse tails.
# The shipped default caps that net count via --internal-cip-cap-neff=500 and n-output-samples=5000,
# and stops on the tail-blind Gaussian 'lame' convergence test -> chi1_perp under-extends vs bilby.
# This opt-in bundle lifts the NET samples-out and switches to a tail-sensitive stop.
parser.add_argument("--internal-cip-transverse-tails",action='store_true',help="OPT-IN alt config for resolving transverse-spin (chi1_perp) tails, esp. at low mass. Bundles: (a) tail-sensitive convergence test (passes --internal-test-convergence-method js_lame to helper_LDG_Events.py, unless overridden); (b) raises the NET interim posterior samples across the CIP worker cohort by lifting --internal-cip-cap-neff and --n-output-samples and scaling up --cip-explode-jobs (MORE WORKERS -> more net samples-out, NOT larger per-worker n_eff) -- the raised interim sample count is what makes js_lame's quantile-drift tolerance statistically meaningful; (c) transverse TAIL-GUARD in the puffball: --append-with-random-parameter chi1_perp appends+shuffles uniformly-random transverse draws into every puff, so the proposed grid keeps offering chi1_perp tail coverage even after the posterior contracts (the measured tail-starvation feedback), and puff is kept active through all iterations. REQUIRES A PRECESSING ANALYSIS (precessing approximant or --assume-precessing): the tail guard proposes nonzero transverse spin, so combining this with --assume-nospin/--assume-nonprecessing or an aligned-spin approximant is REJECTED rather than silently changing the spin model analyzed. Tune with the --internal-cip-transverse-tails-* flags. Default OFF (behavior unchanged). See results_triage/CONVERGENCE_PROTOCOL_2026-07-23.md.")
parser.add_argument("--internal-cip-transverse-tails-cap-neff",type=int,default=4000,help="With --internal-cip-transverse-tails: raise --internal-cip-cap-neff to at least this (the interim net-n_eff throttle; shipped base is 500).")
parser.add_argument("--internal-cip-transverse-tails-nout",type=int,default=20000,help="With --internal-cip-transverse-tails: raise interim --n-output-samples to at least this (net samples out, combined across workers).")
parser.add_argument("--internal-cip-transverse-tails-worker-scale",type=float,default=3.0,help="With --internal-cip-transverse-tails: multiply cip-explode-jobs (and -last) by this, so the raised net sample count is produced by MORE WORKERS while each worker's n_eff stays modest.")
parser.add_argument("--internal-cip-transverse-tails-puff-fraction",type=float,default=0.3,help="With --internal-cip-transverse-tails: fraction of the puff output appended as uniformly-random chi1_perp tail-guard points (puffball --append-with-random-fraction).")
parser.add_argument("--internal-test-convergence-method",type=str,default=None,help="Convergence-test method passed to helper_LDG_Events.py (lame|ks1d|KL_1d|js_additive|js_lame). If js_lame is requested, --internal-cip-transverse-tails is AUTO-ENABLED (the raised interim sample count is required for js_lame's drift tolerance) and therefore js_lame REQUIRES A PRECESSING ANALYSIS -- it is rejected with --assume-nospin/--assume-nonprecessing or an aligned-spin approximant, where there is no transverse tail to score. If unset: helper default (lame), or js_lame when --internal-cip-transverse-tails is on.")
parser.add_argument('--internal-cip-tripwire',type=float,help="Passed to CIP")
parser.add_argument("--internal-cip-temper-log",action='store_true',help="Use temper_log in CIP.  Helps stabilize adaptation for high q for example")
parser.add_argument("--internal-cip-request-memory",default=None,type=int,help="ILE memory request in Mb. Only experts should change this.")
parser.add_argument("--internal-ile-sky-network-coordinates",action='store_true',help="Passthrough to ILE ")
parser.add_argument("--internal-ile-sky-network-coordinates-raw",action='store_true',help="Passthrough to ILE ")
parser.add_argument("--internal-ile-rotate-phase", action='store_true')
parser.add_argument("--internal-loud-signal-mitigation-suite",action='store_true',help="Enable more aggressive adaptation - make sure we adapt in distance, sky location, etc rather than use uniform sampling, because we are constraining normally subdominant parameters")
parser.add_argument("--internal-ile-freezeadapt",action='store_true',help="Passthrough to ILE ")
parser.add_argument("--internal-ile-reset-adapt",action='store_true',help="Force reset of adaptation")
parser.add_argument("--internal-ile-force-noreset-adapt",action='store_true',help="Undo any attempt to --force-reset-all")
parser.add_argument("--internal-ile-adapt-log",action='store_true',help="Passthrough to ILE ")
parser.add_argument("--internal-ile-auto-logarithm-offset",action='store_true',help="Passthrough to ILE")
parser.add_argument("--internal-ile-use-lnL",action='store_true',help="Passthrough to ILE via helper.  Will DISABLE auto-logarithm-offset and manual-logarithm-offset for ILE")
parser.add_argument("--export-marginal-distance-grid",action='store_true',help="Ask the ILE extrinsic stage to export per-intrinsic likelihood density grids in luminosity distance. Forces ILE lnL mode and disables distance marginalization. Requires the extrinsic stage (--add-extrinsic).")
parser.add_argument("--export-distance-slices",default=0,type=int,help="If >0, ask the ILE extrinsic stage to export K-row .dslice files (Plan-B fixed-distance extrinsic-marginalized likelihoods). Forces ILE lnL mode and disables distance marginalization. Requires the extrinsic stage (--add-extrinsic).")
parser.add_argument("--export-distance-slices-n-core",default=0,type=int,help="Passthrough: --n-distance-slice-core for the .dslice export.")
parser.add_argument("--export-distance-slices-n-wing",default=0,type=int,help="Passthrough: --n-distance-slice-wing for the .dslice export.")
parser.add_argument("--export-distance-slices-all-fresh",action='store_true',default=False,help="Passthrough: --distance-slice-all-fresh. All K slices are fresh fixed-d integrations (no importance-reweight core). Use at low main-loop n_eff, where the reweight core is starved.")
parser.add_argument("--export-distance-slices-randomize",action='store_true',default=False,help="Passthrough: --distance-slice-randomize. (all-fresh) Draw each intrinsic's fresh-slice distances at random posterior-d quantiles, so K=1 gives a fair-draw of d per intrinsic -- cheap dense (intrinsic,d) coverage for the AD surrogate.")
parser.add_argument("--export-distance-slices-wing-neff",default=None,type=int,help="Passthrough: --distance-slice-wing-neff (n_eff per fresh fixed-d slice integration -- the precision of each L(d) row).")
parser.add_argument("--export-distance-slices-wing-nmax",default=None,type=int,help="Passthrough: --distance-slice-wing-nmax (max samples per fresh fixed-d slice integration).")
parser.add_argument("--export-distance-slices-wing-delta-lnL",default=None,type=float,help="Passthrough: --distance-slice-wing-delta-lnL for the .dslice export (target lnL drop below peak for wing placement).")
parser.add_argument("--export-distance-slices-skip-threshold",default=None,type=float,help="Passthrough: --distance-slice-skip-threshold for the .dslice export (absolute peak-lnL detectability cut).")
parser.add_argument("--ile-additional-files-to-transfer",default=None,help="Comma-separated list of filenames. To append to the transfer file list for ILE jobs (only). Intended for surrogates in LAL_DATA_PATH for wide-ranging use")
parser.add_argument("--internal-cip-use-lnL",action='store_true')
parser.add_argument("--manual-initial-grid",default=None,type=str,help="Filename (full path) to initial grid. Copied into proposed-grid.<suffix>, overwriting any grid assignment done here. Suffix is .xml.gz by default and .dat when RIFT_HYPERPIPELINE_FORMAT is set; the source file's format must match the active mode.")
parser.add_argument("--manual-initial-grid-supplements",action='store_true', help="Manual inital grid used to SUPPLEMENT output of the default helper grid.")
parser.add_argument("--manual-extra-ile-args",default=None,type=str,help="Avenue to adjoin extra ILE arguments.  Needed for unusual configurations (e.g., if channel names are not being selected, etc)")
parser.add_argument("--internal-ile-force-adapt-all",action='store_true', help="Syntactic sugar to prevent need to add manual-extra-ile-args for this: easier on user")
parser.add_argument("--internal-puff-transverse",action='store_true', help=" appends the following arguments: --parameter phi1 --parameter phi2 --parameter chi1_perp_u --parameter chi2_perp_u ")
parser.add_argument("--manual-extra-puff-args",default=None,type=str,help="Avenue to adjoin extra PUFF arguments.  ")
parser.add_argument("--manual-extra-test-args",default=None,type=str,help="Avenue to adjoin extra TEST arguments.  ")
parser.add_argument("--manual-extra-cip-args",default=None,type=str,help="Avenue to adjoin extra CIP arguments.  Needed for external priors or likelihoods in CIP stage")
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--use-downscale-early",action='store_true', help="If provided, the first block of iterations are performed with lnL-downscale-factor passed to CIP, such that rho*2/2 * lnL-downscale-factor ~ (15)**2/2, if rho_hint > 15 ")
parser.add_argument("--use-gauss-early",action='store_true',help="If provided, use gaussian resampling in early iterations ('G'). Note this is a different CIP instance than using a quadratic likelihood!")
parser.add_argument("--use-quadratic-early",action='store_true',help="If provided, use a quadratic fit in the early iterations'")
parser.add_argument("--use-gp-early",action='store_true',help="If provided, use a gp fit in the early iterations'")
parser.add_argument("--use-cov-early",action='store_true',help="If provided, use cov fit in the early iterations'")
parser.add_argument("--use-osg",action='store_true',help="Restructuring for ILE on OSG. The code by default will use CVMFS")
parser.add_argument("--use-osg-cip",action='store_true',help="Restructuring for ILE on OSG. The code by default will use CVMFS")
parser.add_argument("--use-osg-file-transfer",action='store_true',help="Restructuring for ILE on OSG. The code will NOT use CVMFS, and instead will try to transfer the frame files.")
parser.add_argument("--internal-use-oauth-files",default=None,type=str,help="Option for low level pipeline writer to use scitokens. Useful if files on osdf need to be transferred, like containers ")
parser.add_argument("--internal-truncate-files-for-osg-file-transfer",action='store_true',help="If use-osg-file-transfer, will use FrCopy plus the start/end time to build the frame directory.")
parser.add_argument("--condor-local-nonworker",action='store_true',help="Provide this option if job will run in non-NFS space. ")
parser.add_argument("--condor-local-nonworker-igwn-prefix",action='store_true', help="Adds some prefix text to start up cvmfs igwn environment, so local jobs have access to standard RIFT operators. Required for public OSG.")
parser.add_argument("--condor-nogrid-nonworker",action='store_true',help="NOW STANDARD, auto-set if you pass use-osg   Causes flock_local for 'internal' jobs, UNLESS using --use-osg-public")
parser.add_argument("--use-osg-simple-requirements",action='store_true',help="Provide this option if job should use a more aggressive setting for OSG matching ")
parser.add_argument("--use-osg-public",action='store_true',help="Activate public osg settings. Enforces use-osg, condor-local-nonworker, and condor_local+nonworker_igwn_prefix")
parser.add_argument("--archive-pesummary-label",default=None,help="If provided, creates a 'pesummary' directory and fills it with this run's final output at the end of the run")
parser.add_argument("--archive-pesummary-event-label",default="this_event",help="Label to use on the pesummary page itself")
parser.add_argument("--internal-mitigate-fd-J-frame",default="L_frame",help="L_frame|rotate, choose method to deal with ChooseFDWaveform being in wrong frame. Default is to request L frame for inputs")
parser.add_argument("--internal-force-puff-iterations", default=None, type=int, help="Number of iterations to be puffed. If None, will use the algorithm to determine the number of iterations to puff.")
opts=  parser.parse_args()

# Resolve the sub-sample stencil request IMMEDIATELY, so a bare flag / retired 'True' / typo
# fails here rather than being forwarded into a workflow build.  Value unused at this point --
# the call is for its validation side effect; the helper resolves it again for the emission.
resolve_interpolate_time_request(opts.internal_ile_interpolate_time)

# Multi-GPU ILE fan-out: --ile-gpu-fanout funnels through RIFT_ILE_GPU_FANOUT, which
# create_event_parameter_pipeline_BasicIteration (run via os.system, inheriting this
# environment) and dag_utils read at DAG-build time to size request_GPUs/CPUs and bake
# the value into ile_pre.sh.  A CLI value wins over any inherited environment value.
if opts.ile_gpu_fanout is not None:
    os.environ['RIFT_ILE_GPU_FANOUT'] = str(opts.ile_gpu_fanout)

config_stored=None; config_dict=None
ile_condor_commands = None
if (opts.use_ini):
    # Attempt to lazy-parse all command line arguments from ini file
    config = ConfigParser.ConfigParser()
    config.optionxform=str # force preserve case! Important for --choose-data-LI-seglen
    config.read(opts.use_ini)
    config_stored=config
    # Command line arguments
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

        if not('RIFT_REQUIRE_GPUS' in os.environ) and 'ile_require_gpus' in rift_items:
            os.environ['RIFT_REQUIRE_GPUS'] = rift_items['ile_require_gpus']

        # Container family (multi-container per-machine image selection): let the ini
        # carry the manifest + base exe dir, so a single ini is self-contained.  These
        # are read from os.environ by create_event_parameter_pipeline_BasicIteration /
        # write_ILE_sub_simple; the environment still dominates if already set.
        if not('SINGULARITY_RIFT_IMAGE' in os.environ) and 'singularity_rift_image' in rift_items:
            os.environ['SINGULARITY_RIFT_IMAGE'] = rift_items['singularity_rift_image']
        if not('SINGULARITY_BASE_EXE_DIR' in os.environ) and 'singularity_base_exe_dir' in rift_items:
            os.environ['SINGULARITY_BASE_EXE_DIR'] = rift_items['singularity_base_exe_dir']

        # attempt to lazy-select the command-line that are present in the ini file section
        for item in rift_items:
            item_renamed = item.replace('-','_')
            if (item_renamed in config_dict):
                val = rift_items[item].strip()
#                if not(config_dict[item_renamed]):   # needs to be set to some value. Don't *disable* what is enabled on command line
                print(" ini file parser (overrides command line, except booleans): ",item, rift_items[item])
                if val != "":
                    if 'manual_extra' in item_renamed: # manual-extra-ile-args, manual-extra-cip-args, etc. Do not parse, pass through!
                        config_dict[item_renamed] = val
                    else:
                        config_dict[item_renamed] = eval(rift_items[item])
                else:
                    config_dict[item_renamed] = True
        print(config_dict)
    # Condor commands for ile
    if 'rift-ile-condor' in config:
        rift_items = dict(config["rift-ile-condor"])
        config_ile_condor_dict = vars(opts) # access dictionry of options
        ile_condor_commands = []
        for item in rift_items:
            val = rift_items[item].strip()
            ile_condor_commands.append([item, val])

# Multi-GPU ILE fan-out: funnel --ile-gpu-fanout (CLI) OR ile-gpu-fanout=... (ini
# [rift-pseudo-pipe]) through RIFT_ILE_GPU_FANOUT.  Done AFTER the ini is parsed so an
# ini-only value also lands.  create_event_parameter_pipeline_BasicIteration (run via
# os.system, inheriting this environment) and dag_utils read it at DAG-build time to
# size request_GPUs/CPUs and bake the value into ile_pre.sh.
if opts.ile_gpu_fanout is not None:
    os.environ['RIFT_ILE_GPU_FANOUT'] = str(opts.ile_gpu_fanout)

if opts.lisa_known_sky:
    run_lisa_known_sky_surface(opts)
    sys.exit(0)



if opts.use_osg:
    opts.condor_nogrid_nonworker = True  # note we ALSO have to check this if we set use_osg in the ini file! Moved statement so flagged

if opts.use_osg_public:
    opts.use_osg=True
    opts.condor_local_nonworker=True
    opts.condor_local_nonworker_igwn_prefix=False
    opts.condor_nogrid_nonworker=False

if opts.ile_copies <=0:
    raise Exception(" Must have 1 or more ILE instances per intrinsic point")

if not(opts.internal_mitigate_fd_J_frame in ['L_frame', 'rotate']):
    raise Exception(" Unknown option for internal_mitigate_fd_J_frame")
if (opts.approx in ['IMRPhenomXPHM' or 'IMRPhenomXO4']) and opts.assume_precessing:
    print(" NOTE NOTE NOTE : Mitigation of ChooseFDWaveform frame being applied : {} ".format(opts.internal_mitigate_fd_J_frame))

if opts.internal_loud_signal_mitigation_suite:
    opts.internal_ile_freezeadapt=False  # make sure to adapt every iteration, and adapt in distance if present
    if opts.ile_sampler_method == 'adaptive_cartesian_gpu' or opts.ile_sampler_method == 'GMM':
        opts.internal_ile_use_lnL = True
        # For coordinate-tied systems, some special options
        opts.internal_ile_sky_network_coordinates=True # skymap is better
        opts.internal_ile_rotate_phase = True  # phase coordinates can be sharper

# Default prior for aligned analysis should be z prior !
if opts.assume_nonprecessing or opts.approx == "IMRPhenomD":
    prior_args_lookup["default"] = prior_args_lookup["zprior_aligned"]
    opts.internal_puff_transverse=False


if opts.ile_xpu:
    opts.ile_force_gpu = False

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

if opts.use_osg and not(opts.use_osg_file_transfer): 
    print(" datafind: changing LIGO_DATAFIND_SERVER internally to find CVMFS-disseminated data")
    os.environ["LIGO_DATAFIND_SERVER"]="datafind.ligo.org:443"   #  enable lookup of data for public cvmfs

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
elif opts.use_coinc and opts.fake_data_cache:
    # if gracedb id is NOT none, but if we have a coinc and cache file, do nothing/no warnings
   print("  pseudo_pipe: no authenticated lookup needed, coinc and cache file provided as ", opts.use_coinc, opts.fake_data_cache)     
else:
    from RIFT.misc.dag_utils_generic import which
    # https://computing.docs.ligo.org/guide/htcondor/credentials/#scitokens
    print(" ===> WARNING <=== ")
    print(" gracedb id provided but either missing coinc file or cache file; lookup necessary, possibly requiring authentication ")
    cmd_httoken_info = which('httokendecode')
    if not (cmd_httoken_info):
        print("   - no httokendecode - ")
    else:
        print(" Token info , if any ")
        os.system(cmd_httoken_info)
        # checks X509_USER_PROXY env variable
        # if empty, checks grid-proxy-info -path
        # if empty, fails and tells you to run ligo-proxy-init
    # if not("X509_USER_PROXY" in os.environ.keys()):
    #     cmd_grid = which("ecp-cert-info")  # current default
    #     if not cmd_grid:
    #         cmd_grid = which('grid-proxy-info')  # old behavior
    #     str_proxy =subprocess.check_output([cmd_grid,'-path']).rstrip()
    #     if len(str_proxy) < 1:
    #         print( " Run ligo-proxy-init or otherwise have a method to query gracedb / use CVMFS frames as you need! ! ")
    #         sys.exit(1)
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
        cmd_fix_ilwdchar = "{}ligolw_no_ilwdchar coinc.xml".format(ligolw_prefix); os.system(cmd_fix_ilwdchar) # sigh, need to make sure we are compatible
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
if opts.approx == "SEOBNRv3" or opts.approx == "NRSur7dq2" or opts.approx == "NRSur7dq4" or (opts.approx == 'SEOBNv3_opt') or (opts.approx == 'IMRPhenomPv2') or (opts.approx =="SEOBNRv4P" ) or (opts.approx == "SEOBNRv4PHM") or (opts.approx == "SEOBNRv5PHM") or ('SpinTaylor' in opts.approx) or ('IMRPhenomTP' in opts.approx or ('IMRPhenomXP' in opts.approx)):
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
    if opts.use_meanPerAno:
        dirname_run += "_and_meanPerAno"
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
    # default value for eccentricity is 0 for 'P'!  Only change this value from default if eccentricity is present, do NOT want to fill it with None in particular
    if not(event_dict['eccentricity'] is None):   
        P.eccentricity = event_dict["eccentricity"]
    if not(event_dict['meanPerAno'] is None):
        P.meanPerAno = event_dict["meanPerAno"]
    # Write 'target_params' file -- hyperpipeline .dat or legacy XML.
    if _use_hpip_pp:
        from RIFT.misc import hyperpipeline_io as _hpio
        _cols = _hpio.build_column_list(
            use_eccentricity=(P.eccentricity != 0),
            use_meanPerAno=(P.meanPerAno != 0))
        _hpio.write_grid_from_P_list("target_params", [P], _cols,
                                     lal_module=lal,
                                     lalsimutils_module=lalsimutils)
    else:
        lalsimutils.ChooseWaveformParams_array_to_xml([P], "target_params")

    if opts.use_production_defaults:
        # use more workers for high-q triggers
        # worker scale = (1+2/q), max of 50
        q = P.m2/P.m1
        opts.cip_explode_jobs = int(np.min([int(2+3./q),50]))


helper_psd_args = ''
srate=4096  # default, built into helper, unwise to go lower, LI will almost never do higher
if opts.make_bw_psds:
    helper_psd_args += " --assume-fiducial-psd-files --fmax " + str(srate/2-1)

# Create provenance info : we want run to be reproducible
# for low-latency analysis, we can assume we have provenance?
if not(opts.skip_reproducibility): # not(assume_lowlatency):
        import shutil, json
        if opts.use_ini:
            shutil.copyfile(opts.use_ini, "local.ini") # copy into current directory
        if not(os.path.exists("reproducibility")):
            os.mkdir("reproducibility")
        # Write this script and its arguments
#        thisfile = os.path.realpath(__file__)
#        shutil.copyfile(thisfile, "reproducibility/the_script_used.py")
        argparse_dict = vars(opts)
        # arguments in json form
        with open("reproducibility/the_arguments_used.json",'w') as f:
                json.dump(argparse_dict,f)
        # config parsing
        if opts.use_ini and not(config_dict is None):
            for name in argparse_dict:
                config_stored['rift-pseudo-pipe'][name] = str(argparse_dict[name]) # add info
            with open("reproducibility/local_real.ini",'w') as f:
                config.write(f) # the actual arguments used!  (Not yet tested it works as input)
            
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
# Alt config for transverse-spin (chi1_perp) tails (opt-in; transverse-spin study). Lift the NET
# interim posterior-sample count (combined across CIP workers) and switch to a tail-sensitive stop.
# Worker COUNT is scaled later (after the auto-explode block); here we lift the per-iteration net
# throttles (cap-neff, n-output) and record that helper must use the tail-sensitive convergence test.
# js_lame REQUIRES the raised interim sample count (its quantile-drift tolerance is at the
# split-half noise floor at the shipped n~5e3): requesting js_lame auto-enables the tails bundle.
#
# The whole bundle only has meaning for an analysis that HAS transverse spin: the tail guard
# appends uniformly-random chi1_perp (i.e. nonzero s1x/s1y) to every puff, and js_lame scores the
# chi1_perp tail.  Under --assume-nospin/--assume-nonprecessing, or with an aligned-spin
# approximant, those grid points are not representable by the waveform being used: ILE either
# rejects them or silently analyzes a different spin model, and the convergence test is handed no
# transverse parameter at all (helper only passes chi1_perp when the analysis is precessing).
# Refuse the combination rather than quietly changing the physics of the run.
def approx_supports_precession(approx_name):
    """Authoritative precession classification of an approximant, from lalsimulation's own spin
    support flag.  The is_analysis_precessing test above is a hand-maintained name list that omits
    supported precessing models (e.g. IMRPhenomXO4a), so it must not be the sole gate on options
    that require transverse spin.  Returns None when the name is not a lalsimulation approximant
    (external/NR waveforms), in which case the caller should fall back to the name list."""
    try:
        support = lalsim.SimInspiralGetSpinSupportFromApproximant(lalsim.GetApproximantFromString(approx_name))
    except Exception:
        return None
    # CASEBYCASE (e.g. NR/surrogate entries) allows precession, decided per waveform; both the
    # current and the older LAL_-prefixed spellings of these constants are accepted.
    precessing_support = [getattr(lalsim, name) for name in
                          ('SIM_INSPIRAL_PRECESSINGSPIN', 'LAL_SIM_INSPIRAL_PRECESSINGSPIN',
                           'SIM_INSPIRAL_CASEBYCASE', 'LAL_SIM_INSPIRAL_CASEBYCASE')
                          if hasattr(lalsim, name)]
    if not precessing_support:
        return None
    return support in precessing_support

if opts.internal_cip_transverse_tails or opts.internal_test_convergence_method == 'js_lame':
    # "precessing approximant or --assume-precessing" is the contract, so ask lalsimulation about
    # the approximant rather than trusting only the name list.  The forced flags still win, since
    # they change what the analysis actually samples.
    analysis_has_transverse_spin = is_analysis_precessing or bool(approx_supports_precession(opts.approx))
    if opts.assume_nospin or opts.assume_nonprecessing:
        analysis_has_transverse_spin = False
    if not analysis_has_transverse_spin:
        raise Exception(" --internal-cip-transverse-tails (and --internal-test-convergence-method js_lame, which enables it) require a PRECESSING analysis: the puff tail-guard proposes nonzero chi1_perp and the convergence test scores its tail, neither of which an aligned-spin/zero-spin waveform can represent.  Current settings give a nonprecessing analysis (approx {}{}{}).  Use a precessing approximant or --assume-precessing, or drop these options.".format(opts.approx, ' with --assume-nospin' if opts.assume_nospin else '', ' with --assume-nonprecessing' if opts.assume_nonprecessing else ''))
    if not is_analysis_precessing:
        # A precessing approximant the name list does not recognize.  The bundle needs the analysis
        # itself to carry transverse spin -- the helper only proposes the precessing fit strategy and
        # the chi1_perp convergence parameter for a precessing analysis -- so turn it on here instead
        # of making the user restate the approximant's own physics with --assume-precessing.
        is_analysis_precessing = True
        print("  [transverse-tails] approximant {} supports precession; using the precessing analysis options this bundle requires".format(opts.approx))
if opts.internal_test_convergence_method == 'js_lame' and not opts.internal_cip_transverse_tails:
    opts.internal_cip_transverse_tails = True
    print("  [transverse-tails] AUTO-ENABLED by --internal-test-convergence-method js_lame (drift test needs the raised interim n-output-samples)")
if opts.internal_cip_transverse_tails:
    opts.internal_cip_cap_neff = int(np.max([opts.internal_cip_cap_neff, opts.internal_cip_transverse_tails_cap_neff]))
    opts.n_output_samples      = int(np.max([opts.n_output_samples,      opts.internal_cip_transverse_tails_nout]))
    if opts.internal_test_convergence_method is None:
        opts.internal_test_convergence_method = 'js_lame'
    print("  [transverse-tails] raising NET interim sampling: cip-cap-neff -> {}, n-output-samples -> {} (worker count scaled x{} below); convergence test -> {}; puff tail-guard chi1_perp fraction {}".format(opts.internal_cip_cap_neff, opts.n_output_samples, opts.internal_cip_transverse_tails_worker_scale, opts.internal_test_convergence_method, opts.internal_cip_transverse_tails_puff_fraction))

cmd = " helper_LDG_Events.py --force-notune-initial-grid   --propose-fit-strategy --propose-ile-convergence-options  --fmin " + str(fmin) + " --fmin-template " + str(fmin_template) + " --working-directory " + base_dir + "/" + dirname_run  + helper_psd_args  + " --no-enforce-duration-bound --test-convergence "
if opts.internal_test_convergence_method:
    cmd += " --internal-test-convergence-method {} ".format(opts.internal_test_convergence_method)
if opts.internal_use_gracedb_bayestar:
    cmd += " --internal-use-gracedb-bayestar "
if opts.internal_use_amr:
    cmd += " --internal-use-amr " # minimal support performed in this routine, mainly for puff
if opts.internal_use_aligned_phase_coordinates:
    cmd += " --internal-use-aligned-phase-coordinates "
if opts.internal_use_rescaled_transverse_spin_coordinates:
    cmd += " --internal-use-rescaled-transverse-spin-coordinates "
if not(opts.internal_use_amr) and not(opts.manual_initial_grid and not(opts.manual_initial_grid_supplements)):
    cmd+= " --propose-initial-grid "
if opts.force_initial_grid_size:
    cmd += " --force-initial-grid-size {} ".format(int(opts.force_initial_grid_size))
if opts.assume_matter:
        cmd += " --assume-matter "
        npts_it = 1000
        if opts.assume_matter_eos:
            cmd += " --assume-matter-eos {} ".format(opts.assume_matter_eos)
        if opts.assume_matter_but_primary_bh:
            cmd+= " --assume-matter-but-primary-bh "
        if opts.internal_tabular_eos_file:
            cmd += " --internal-tabular-eos-file {} ".format(opts.internal_tabular_eos_file)
        if opts.assume_matter_conservatively:
            cmd += " --assume-matter-conservatively "
if  opts.assume_nospin:
    cmd += " --assume-nospin "
else:  
  if is_analysis_precessing:
        cmd += " --assume-precessing-spin "
        npts_it = 1500
if is_analysis_eccentric:
    cmd += " --assume-eccentric "
    npts_it = int(npts_it*1.5)
    if opts.use_meanPerAno:
        cmd += " --use-meanPerAno "
        npts_it = int(npts_it*1.5)
if opts.assume_highq:
    cmd+= ' --assume-highq  --force-grid-stretch-mc-factor 2'  # the mc range, tuned to equal-mass binaries, is probably too narrow. Workaround until fixed in helper
    npts_it =1000
if opts.internal_propose_converge_last_stage:
    cmd += " --propose-converge-last-stage "
if opts.internal_test_convergence_threshold: # pass argument if provided
    cmd += " --internal-test-convergence-threshold {}  ".format(opts.internal_test_convergence_threshold)
if not(opts.cip_fit_method is None):
    cmd += " --force-fit-method {} ".format(opts.cip_fit_method)
    if opts.cip_fit_method == 'rf':
        npts_it*=2 # more iteration points if we use RF ... not sane otherwise. Note for precession this is a large iteration size
    elif opts.cip_fit_method == 'quadratic' or opts.cip_fit_method == 'polynomial' or opts.use_quadratic_early or opts.use_cov_early:
        npts_it*=2 # more iteration points if we use some initial quadratic iterations ... they also benefit from more samples overall. Default description is for GP

if opts.internal_ile_use_lnL:
    cmd+= " --internal-ile-use-lnL "
if opts.export_marginal_distance_grid or (opts.export_distance_slices and opts.export_distance_slices > 0):
    # Per-distance likelihood export needs ILE lnL mode (forced here for the
    # whole run, giving clean lnL-scaled helper args) and, *only at the export
    # stage*, no distance marginalization.  We deliberately do NOT disable
    # distance marginalization globally: the intrinsic iterations keep it (it
    # is a large speedup).  Only the final extrinsic stage that emits the
    # per-distance output has --distance-marginalization stripped, and that
    # stripping is done by create_event_parameter_pipeline_* on the ILE_extr
    # argument string -- not here.
    opts.internal_ile_use_lnL = True
    if "--internal-ile-use-lnL" not in cmd:
        cmd += " --internal-ile-use-lnL "
    if not opts.add_extrinsic:
        print(" ==> WARNING: distance grid/slice export is emitted by the ILE extrinsic stage, but --add-extrinsic is not set; no per-distance output will be produced. <== ")
if opts.internal_cip_use_lnL:
    cmd += " --internal-cip-use-lnL "
if opts.internal_ile_data_tukey_window_time:
    cmd += " --data-tukey-window-time {} ".format(opts.internal_ile_data_tukey_window_time)
if (opts.internal_ile_psd_common_window):
    cmd += " --psd-assume-common-window "
if not(opts.ile_n_eff is None):
    cmd += " --ile-n-eff {} ".format(opts.ile_n_eff)
if opts.limit_mc_range:
    cmd+= " --limit-mc-range  " + str(opts.limit_mc_range).replace(' ','')
if not(opts.force_mc_range is None):
    cmd+= " --force-mc-range  " + str(opts.force_mc_range).replace(' ','')
elif opts.scale_mc_range:
    cmd += " --scale-mc-range  " + str(opts.scale_mc_range).replace(' ','')
if not(opts.force_eta_range is None):
    cmd+= " --force-eta-range  " + str(opts.force_eta_range).replace(' ','')
if opts.allow_subsolar:
    cmd += " --allow-subsolar "
if opts.force_chi_max:
    cmd+= " --force-chi-max {} ".format(opts.force_chi_max)
if opts.force_chi_small_max:
    cmd+= " --force-chi-small-max {} ".format(opts.force_chi_small_max)
if opts.force_lambda_max:
    cmd+= " --force-lambda-max {} ".format(opts.force_lambda_max)
if opts.force_lambda_small_max:
    cmd+= " --force-lambda-small-max {} ".format(opts.force_lambda_small_max)    
if not(opts.gracedb_id is None): #  and (opts.use_ini is None):
    # --gracedb-id downloads coinc.xml, and allows use of PSD files in coinc.xml
    # Note providing coinc.xml will prevent attempting to download coinc from gracedb, but it is STILL needed to retrieve PSDs from it
    cmd +="  --gracedb-id " + gwid 
    if  opts.use_legacy_gracedb:
        cmd+= " --use-legacy-gracedb "
elif  not(opts.event_time is None):
    cmd += " --event-time " + format_gps_time(opts.event_time)
    if opts.use_ini:
        seglen = float(config['engine']['seglen'])
        data_start_time = opts.event_time - (seglen - 2)
        data_end_time = opts.event_time + 2
        cmd += " --data-start-time {}  --data-end-time {} ".format(data_start_time, data_end_time)
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
if opts.calmarg_pilot:
    # cold-start cal pilots draw cal from the broad PRIOR -> large MC error on iteration 0;
    # relax the first CIP stage's sigma-cut so those points are not all stripped.
    cmd += " --calmarg-first-cip-sigma-cut {} ".format(opts.calmarg_first_cip_sigma_cut)
#if is_event_bns and not opts.no_matter:
#        cmd += " --assume-matter "
#        npts_it = 1000
if opts.internal_flat_strategy:
    cmd +=  " --test-convergence --propose-flat-strategy "
if opts.use_downscale_early:
    cmd += " --use-downscale-early "
if opts.use_gauss_early:
    cmd += " --use-gauss-early "
elif opts.use_quadratic_early:
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
    cmd += " {} {}/target_params.{} --event 0 ".format(sim_grid_flag_pp, base_dir + "/"+ dirname_run, grid_suffix_pp)  # full path to target_params (xml.gz or .dat)
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
if opts.ile_distance_prior:
    cmd += " --ile-distance-prior {} ".format(opts.ile_distance_prior)
if (opts.internal_marginalize_distance): #  and not opts.ile_distance_prior:
    cmd += "  --internal-marginalize-distance "  # note distance marginalization only in one code path (otherwise errors)
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
if resolve_interpolate_time_request(opts.internal_ile_interpolate_time) is not None:
    # resolve_interpolate_time_request rather than a truthiness test: the flag takes a VALUE, so
    # '--internal-ile-interpolate-time False' passes the STRING 'False' (truthy in Python) and a
    # BARE flag passes a sentinel.  Both must be distinguished from "a stencil was named", and a
    # bare flag must raise rather than silently forward nothing.
    # HELPER passthrough (not a raw ILE arg): the helper owns ILE argument construction, and it
    # also knows whether the maintained NoLoop path that --interpolate-time requires is in use --
    # which needs --time-marginalization AND --vectorized AND one of --gpu/--rotation-slow/
    # --freqresponse; the ILE driver refuses rather than ignoring if any is missing.  It also owns the stencil choice, because srate and fmax are
    # resolved there -- so forward the request verbatim rather than resolving it here, and let the
    # helper's log line be the single record of what was chosen.
    cmd += " --internal-ile-interpolate-time " + str(opts.internal_ile_interpolate_time) + " "
if not(opts.internal_ile_n_chunk is None):
    cmd += " --internal-ile-n-chunk {} ".format(int(opts.internal_ile_n_chunk))
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
# we MUST make helper_ile_args.txt
if not(os.path.exists('helper_ile_args.txt')):
    print(" FAILURE: helper call failed to generate required file helper_ile_args.txt")
    sys.exit(1)
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
if opts.internal_ile_n_max:
    line = line.replace('--n-max 4000000 ', '--n-max ' + str(opts.internal_ile_n_max)+" ")
line += " --l-max " + str(opts.l_max) 
if 'data-start-time' in line and 's1z' in event_dict:  # only call this if we have (a) fixed time interval and (b) CBC parameters for event
    # Print warnings based on duration and fmin
    line_dict = unsafe_parse_arg_string_dict(line)
    data_start_time = float(line_dict['data-start-time'])
    data_end_time = float(line_dict['data-end-time'])
    P.m1 = event_dict["m1"]*lal.MSUN_SI; P.m2=event_dict["m2"]*lal.MSUN_SI; P.s1z = event_dict["s1z"]; P.s2z = event_dict["s2z"]
    P.fmin = opts.fmin  #  fmin we will use internally
    if opts.fmin_template:
        P.fmin = opts.fmin_template
    if opts.l_max > 2 and (("IMRPhenomXP" in opts.approx) or ('XO4a' in opts.approx)):
        # fmin is start for all modes.   If Lmax>2, use fmin*(2/Lmax) to estimate starting frequecy
        P_temp = P.copy()
        P_temp.fmin *= 2./opts.l_max
        t_HM = lalsimutils.estimateWaveformDuration(P_temp)
        if  data_end_time - data_start_time < t_HM/2:
            print("""  WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING 
Your choice of fmin, lmax, and approximant suggests waveform wraparound will occur.  We recommend a longer segment length
"""
)
    elif opts.l_max > 2 and opts.fmin_template:
        # all modes start at the same time, possibly with different frequencies. Starting frequency needs to be reduced 
        # User has specified fmin_template, and therefore overridden our default plan to lower the starting frequency
        if opts.l_max/2 * opts.fmin_template > opts.fmin:
            print(" WARNING WARNING WARNING: You have modes starting in band. You should probably reduce your starting frequency")
if (opts.use_ini is None) and not('--d-max' in line):
    line += " --d-max " + str(dmax_guess)
if opts.ile_distance_prior:
    line += " --d-prior {} ".format(opts.ile_distance_prior)
if opts.fix_bns_sky:
    line +=" --declination " + str(opts.declination) + " --right-ascension " + str(opts.right_ascension)
    line = line.replace('--declination-cosine-sampler', '') # if we are pinning dec, we aren't using a cosine coordinate. Don't mess up.
if opts.ile_force_gpu:
    line +=" --force-gpu-only "
sur_location_prefix = "my_surrogates/nr_surrogates/"
if 'GW_SURROGATE' in os.environ:
    sur_location_prefix='surrogate_downloads/'
if opts.use_osg:
    sur_location_prefix = "/"
if opts.use_gwsignal:
    line += " --use-gwsignal  --approx " + opts.approx
elif not 'NR' in opts.approx:
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
if opts.internal_ile_reset_adapt or ((opts.ile_sampler_method =='adaptive_cartesian_gpu' or not(opts.ile_sampler_method)) and not(opts.internal_ile_freezeadapt) ):
    # force reset if
    #   - requested or
    #   - AC + not freezeadapt
    line += " --force-reset-all "
if not(opts.manual_extra_ile_args is None):
    line += " {} ".format(opts.manual_extra_ile_args)  # embed with space on each side, avoid collisions
    if '--declination ' in opts.manual_extra_ile_args:   # if we are pinning dec, we aren't using a cosine coordinate. Don't mess up.
        line = line.replace('--declination-cosine-sampler', '')
if opts.internal_ile_force_adapt_all:
    line += " --force-adapt-all "
# NOTE on per-distance export (grid/slices): the --last-iteration-export-*
# flags are *pipeline-builder* flags (consumed by
# create_event_parameter_pipeline_*), NOT ILE flags, so they are added to the
# CEPP command below -- not to this ILE argument string (args_ile.txt).  The
# args_ile.txt here is the *intrinsic* ILE configuration and intentionally
# keeps --distance-marginalization (a speedup); lnL mode was already forced
# above, so --internal-use-lnL is present.  create_event_parameter_pipeline_*
# strips --distance-marginalization only from the extrinsic (ILE_extr) stage
# that emits the per-distance output.
if not(opts.ile_sampler_method is None):
    line += " --sampler-method {} ".format(opts.ile_sampler_method)
if opts.internal_ile_sky_network_coordinates:
    line += " --internal-sky-network-coordinates "
if opts.internal_ile_sky_network_coordinates_raw:
    line += " --internal-sky-network-coordinates-raw "
if opts.ile_no_gpu or opts.ile_sampler_method ==  "AV":  # make sure we are using the standard code path if not using GPUs
    line += " --force-xpy " 
if opts.internal_ile_force_noreset_adapt:
    line = line.replace(' --force-reset-all ', ' ')
if opts.internal_mitigate_fd_J_frame == 'L_frame':
    line += " --internal-waveform-fd-L-frame "
if opts.internal_ile_inv_spec_trunc_time:
    line = line.replace("inv-spec-trunc-time 0 ","inv-spec-trunc-time {} ".format(opts.internal_ile_inv_spec_trunc_time))
if (opts.internal_ile_modify_taper):
    line += " --internal-waveform-taper SIM_INSPIRAL_TAPER_START " # taper start of waveform by default, overrides any settings in the grids
if opts.internal_ile_srate_internal:
    line += " --srate-internal {} ".format(opts.internal_ile_srate_internal)
# strictly the next argument only does anything at the extrinsic step, otherwis it is ignored
if opts.internal_ile_srate_time_resampling:
    line += " --srate-resample-time-marginalization {} ".format(opts.internal_ile_srate_time_resampling)
# In-loop calibration marginalization (inside the ILE GPU loop).  Engages on the
# distance-marginalization code path (kept in args_ile.txt); the fused kernel
# additionally requires GPU and falls back to the loop method otherwise.
if opts.calmarg_envelope_directory:
    cal_dir = os.path.abspath(opts.calmarg_envelope_directory)
    cal_dir_arg = cal_dir
    if opts.use_osg_file_transfer:
        # OSG file transfer: the worker has no shared filesystem, so an absolute
        # --calibration-envelope-directory path is unreachable.  The per-IFO <IFO>.txt
        # envelope files are transferred FLAT into the job scratch dir, so reference them
        # relative to '.', and auto-append them to the ILE transfer list (the user should
        # not have to remember --ile-additional-files-to-transfer for these).
        cal_dir_arg = '.'
        _cal_files = ",".join("{}/{}.txt".format(cal_dir, ifo) for ifo in event_dict["IFOs"])
        opts.ile_additional_files_to_transfer = (opts.ile_additional_files_to_transfer + "," + _cal_files) if opts.ile_additional_files_to_transfer else _cal_files
    line += " --calibration-envelope-directory {} --calibration-n-realizations {} --calibration-spline-count {} ".format(cal_dir_arg, opts.calmarg_n_realizations, opts.calmarg_spline_count)
    if opts.calmarg_fused_kernel:
        line += " --calibration-fused-kernel "
    if opts.calmarg_burn_in_neff:
        line += " --calibration-burn-in-neff {} ".format(opts.calmarg_burn_in_neff)
    if opts.calmarg_export_posterior:
        # recovered cal posterior columns; harmless on the wide stage (only fires at the
        # fairdraw/extrinsic stage, which has --save-samples + --resample-time-marginalization).
        line += " --calibration-export-posterior "
    if opts.calmarg_pilot:
        # Option C: wide ILE jobs are SEEDED from the previous iteration's consolidated cal
        # proposal.  The $(macroiterationprev) condor macro resolves per node; ILE falls
        # back to the broad prior when the file is absent/invalid (the first iterations).
        if opts.use_osg_file_transfer:
            # OSG: no shared FS -> reference the breadcrumb by BASENAME (it is transferred
            # in from the submit node, produced at runtime by calpilot_{N-1}), and add it to
            # the ILE transfer list.  Also create a placeholder cal_consolidated_-1.npz so
            # condor's transfer for the FIRST iteration (prev=-1, never produced) does not
            # fail.  Write a VALID 'prior' breadcrumb (proposal == prior -> seeding from it ==
            # cold prior draws, zero weights) rather than a 0-byte file: that way iteration 0
            # LOADS cleanly even on an older ILE binary that does not guard against an empty
            # placeholder (belt-and-suspenders; the size-guard in the ILE is the other half).
            line += " --calibration-proposal-breadcrumb cal_consolidated_$(macroiterationprev).npz "
            _bc_xfer = os.getcwd() + "/cal_consolidated_$(macroiterationprev).npz"
            opts.ile_additional_files_to_transfer = (opts.ile_additional_files_to_transfer + "," + _bc_xfer) if opts.ile_additional_files_to_transfer else _bc_xfer
            _cal_ph_path = os.getcwd() + "/cal_consolidated_-1.npz"
            try:
                import RIFT.calmarg.generate_realizations as _genr_ph, RIFT.calmarg.breadcrumbs as _bcr_ph
                _cal_ph = _genr_ph.prior_cal_breadcrumb_dict(cal_dir, list(event_dict["IFOs"]),
                              fmin_template, srate/2. - 1., opts.calmarg_spline_count)
                _bcr_ph.save(_cal_ph_path, cal=_cal_ph, meta=dict(placeholder=True, iteration=-1))
            except Exception as _e_calph:
                print("  WARNING: could not build prior cal placeholder ({}); writing 0-byte placeholder (needs the ILE empty-breadcrumb guard).".format(_e_calph))
                open(_cal_ph_path, "a").close()
        else:
            line += " --calibration-proposal-breadcrumb {}/cal_consolidated_$(macroiterationprev).npz ".format(os.getcwd())

# Extrinsic handoff (independent of calmarg).  Each wide ILE job WRITES its run's extrinsic
# GMM proposal, and is SEEDED from the previous iteration's consolidated proposal.  Only the
# GMM sampler builds the seedable gmm_dict, so warn if a different sampler is selected.
if opts.extrinsic_handoff:
    if opts.ile_sampler_method != 'GMM':
        print("  WARNING: --extrinsic-handoff seeds the ensemble (GMM) sampler's gmm_dict, but --ile-sampler-method is {}; the seed is a no-op for that sampler.  Pass --ile-sampler-method GMM.".format(opts.ile_sampler_method))
    # output: per-event proposal breadcrumb (basename; written relative to the ILE initialdir
    # on a shared FS, or to job scratch + transferred back on OSG).  $(macroevent) is the
    # per-node event macro, so each wide ILE job gets a distinct file.
    line += " --extrinsic-proposal-output extr_proposal_$(macroiteration)_$(macroevent).npz "
    # seed: from iteration N-1's consolidated proposal.  Mirror the cal breadcrumb path
    # (OSG basename + transfer + iteration-0 placeholder vs shared-FS absolute path).
    if opts.use_osg_file_transfer:
        line += " --extrinsic-proposal-breadcrumb extr_consolidated_$(macroiterationprev).npz "
        _ext_bc_xfer = os.getcwd() + "/extr_consolidated_$(macroiterationprev).npz"
        opts.ile_additional_files_to_transfer = (opts.ile_additional_files_to_transfer + "," + _ext_bc_xfer) if opts.ile_additional_files_to_transfer else _ext_bc_xfer
        # valid EMPTY breadcrumb placeholder (loads cleanly -> extrinsic=None -> no seed/cold),
        # rather than a 0-byte file that np.load chokes on.
        _ext_ph_path = os.getcwd() + "/extr_consolidated_-1.npz"
        try:
            import RIFT.calmarg.breadcrumbs as _bcr_ph
            _bcr_ph.save(_ext_ph_path, meta=dict(placeholder=True, iteration=-1))
        except Exception:
            open(_ext_ph_path, "a").close()
    else:
        line += " --extrinsic-proposal-breadcrumb {}/extr_consolidated_$(macroiterationprev).npz ".format(os.getcwd())

with open('args_ile.txt','w') as f:
        f.write(line)

# ILE transfer file list
#  if arguments provided, append (usually empty file/nonexistent)
if opts.ile_additional_files_to_transfer or opts.internal_ile_check_good_enough:
    extra_files = ''
    if opts.ile_additional_files_to_transfer:
        extra_files = opts.ile_additional_files_to_transfer
        if opts.internal_ile_check_good_enough:
            extra_files += ','
    if opts.internal_ile_check_good_enough:
        extra_files += 'ile_check_good_enough'
    print(" Supplementary transfer request ",extra_files) 
    my_files = list(map(lambda x: x.split(),extra_files.split(','))) # split on , remove whitespace
    my_files = sum(my_files, []) # flatten the list
    my_files = [x for x in my_files if x]  # remove empty elements
    print("  File transfer request resolves to ", my_files)
    with open("helper_transfer_files.txt",'a') as f:
        for line in my_files:
            f.write(line + "\n")


#os.system("cp helper_test_args.txt args_test.txt")
with open ("helper_test_args.txt",'r') as f:
    line = f.readline()
    if opts.add_extrinsic: 
        # We NEVER want to terminate if we're doing extrinsic at the end.  Block termination, so extrinsic occurs on schedule
        line += " --always-succeed "
    if opts.manual_extra_test_args:
        line += " {} ".format(opts.manual_extra_test_args)  # avenue to add extra tests or change test settings
    with open("args_test.txt",'w') as g:
        g.write(line)

# CIP
#   - modify priors to be consistent with the spin priors used in the paper
#   - for the BNS, set chi_max
with open("helper_cip_arg_list.txt",'r') as f:
        raw_lines = f.readlines()

# MODIFY EXPLODE REQUEST
if opts.cip_explode_jobs_auto and event_dict["SNR"]:
    snr = event_dict["SNR"]
    q = P.m2/P.m1
    n_max_jobs=1000
    n_jobs_normal_guess =  2+2*int( (1./q)*np.power(np.max([(snr/15),1]), 1.3) )  # increase number of workers linearly with SNR**1.3 and with mass ratio
    n_jobs_normal_actual = np.min([n_jobs_normal_guess,n_max_jobs])
    n_jobs_final_actual = np.min([2*n_jobs_normal_guess,n_max_jobs])
    if opts.assume_matter:   # more workers for matter physics jobs
        n_jobs_normal_actual *=2; 
        n_jobs_final_actual *=2
    if opts.cip_explode_jobs_auto_scale:
        n_jobs_normal_actual *= opts.cip_explode_jobs_auto_scale; n_jobs_normal_actual = int(n_jobs_normal_actual)
        n_jobs_final_actual *=  opts.cip_explode_jobs_auto_scale; n_jobs_final_actual =int(n_jobs_final_actual )
    print("  AUTO-EXPLODE GUESS {} {} {} ", n_jobs_normal_guess, n_jobs_normal_actual,n_jobs_final_actual)
    opts.cip_explode_jobs = n_jobs_normal_actual
    opts.cip_explode_jobs_last = n_jobs_final_actual
if opts.cip_explode_jobs_auto:
     n_eff_last_target_orig = opts.n_output_samples_last/opts.cip_explode_jobs_last
     if n_eff_last_target_orig > 300:
          opts.cip_explode_job_last = int(opts.n_output_samples_last/300)
          print("  LARGE OUTPUT SAMPLES, CHANGING FINAL EXPLODE to keep n_eff in CIP reasonable ", opts.cip_explode_job_last)


# Alt config for transverse-spin tails (opt-in): scale up the CIP worker cohort AFTER the
# auto-explode block has set the baseline worker count. More workers produce the raised NET
# sample count (cap-neff/n-output lifted above) while each worker's n_eff stays modest -- the
# net (combined) posterior is what resolves chi1_perp tails, not any single worker. (transverse-spin study)
if opts.internal_cip_transverse_tails:
    _sc = opts.internal_cip_transverse_tails_worker_scale
    _base = opts.cip_explode_jobs if opts.cip_explode_jobs else 1
    _base_last = opts.cip_explode_jobs_last if opts.cip_explode_jobs_last else _base
    opts.cip_explode_jobs = int(np.ceil(_base * _sc))
    opts.cip_explode_jobs_last = int(np.ceil(_base_last * _sc))
    print("  [transverse-tails] scaled CIP worker cohort x{}: cip-explode-jobs {} -> {}, -last {} -> {}".format(_sc, _base, opts.cip_explode_jobs, _base_last, opts.cip_explode_jobs_last))

# Add arguments to the file we will use
instructions_cip = list(map(lambda x: x.rstrip().split(' '), raw_lines))#np.loadtxt("helper_cip_arg_list.txt", dtype=str)
n_iterations =0
lines  = []
for indx in np.arange(len(instructions_cip)):
    print(instructions_cip[indx])
    if instructions_cip[indx][0] == 'Z':
        n_iterations += 1
    elif instructions_cip[indx][0][0] == 'G':
        n_G = int(instructions_cip[indx][0][1:])
        n_iterations += n_G
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
    n_workers_last =n_workers
    if opts.cip_explode_jobs_last:
        n_workers_last = opts.cip_explode_jobs_last
    n_eff_cip_last = int(n_sample_target/n_workers_last)
    if indx < len(instructions_cip)-1: # on all but 
        n_eff_cip_here= int(n_sample_target/n_workers)
        n_eff_cip_here = np.amin([opts.internal_cip_cap_neff/n_workers + 1, n_eff_cip_here]) # n_eff: make sure to do *less* than the limit. Lowering this saves immensely on internal/exploration runtime
    else:
        n_eff_cip_here = n_eff_cip_last
    n_sample_min_per_worker = int(n_eff_cip_here/100)+2  # need at least 2 samples, and don't have any worker fall down on the job too much compared to the target

    # Analyze the iteration report
    n_eff_expected_max_easy = 1e-2 * n_max_cip
    n_eff_expected_max_hard = 1e-7 * n_max_cip
    print( " cip iteration group {} : n_eff likely will be between {} and {}, you are asking for at least {} and targeting {}".format(indx,n_eff_expected_max_easy, n_eff_expected_max_hard, n_sample_min_per_worker,n_eff_cip_here))

    line +=" --n-output-samples {}  --n-eff {} --n-max {}  --fail-unless-n-eff {}  --downselect-parameter m2 --downselect-parameter-range [{},{}] ".format(int(n_sample_target/n_workers), n_eff_cip_here, n_max_cip,n_sample_min_per_worker, opts.force_comp_min,opts.force_comp_max)
    if not(opts.allow_subsolar or opts.force_comp_min or opts.force_comp_max):
        line += "  --downselect-parameter m2 --downselect-parameter-range [1,1000] "
    if not(opts.cip_fit_method is None):
        line = line.replace('--fit-method gp ', '--fit-method ' + opts.cip_fit_method)  # should not be called, see --force-fit-method argument to helper
    if not (opts.cip_sampler_method is None):
        line += " --sampler-method "+opts.cip_sampler_method
        if  (opts.cip_sampler_method == 'portfolio'):
            if opts.cip_sampler_portfolio_list is None:
                print(" FAILURE: portfolio requires options. No default!")
                sys.exit(1)
            port_names = opts.cip_sampler_portfolio_list.split(',')
            for name in port_names:
                line += " --sampler-portfolio {} ".format(name.strip())
            if opts.cip_sampler_oracle_list:
                oracle_names = opts.cip_sampler_oracle_list.split(',')
                for name in oracle_names:
                    line += " --sampler-oracle {} ".format(name.strip())                
    if opts.internal_cip_temper_log:
        line += " --internal-temper-log "
    if opts.internal_cip_tripwire:
        line += " --tripwire-fraction {} ".format(opts.internal_cip_tripwire)
    line += prior_args_lookup[opts.spin_magnitude_prior]

    if opts.cip_internal_use_eta_in_sampler:
        line = line.replace('parameter delta_mc','parameter eta')
    if opts.cip_fit_method == 'quadratic' or opts.cip_fit_method == 'polynomial':
        line = line.replace('parameter delta_mc', 'parameter-implied eta --parameter-nofit delta_mc')     # quadratic fit needs eta coordinate. Should be done by helper ideally
    if opts.use_quadratic_early or opts.use_cov_early and indx < 1:
        line = line.replace('parameter delta_mc', 'parameter-implied eta --parameter-nofit delta_mc')     # quadratic or cov fit needs eta coordinate
    if opts.force_lambda_no_linear_init:
        line = line.replace("--prior-lambda-linear", "")  # remove this line, usually used in iteration0
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
        if opts.cip_sampler_method not in {'GMM', 'AV', 'portfolio'}:
            print("  ASSUME HIGHQ FAIL  - currently only GMM/AV/portfolio ")
        else:
            line += " --sampler-method {} --internal-correlate-parameters 'mc,delta_mc,s1z' ".format(opts.cip_sampler_method)
            if 's1z_bar' in line:
                # FIRST attempt to replace with commas, note previous line
                line = line.replace("mc,s1z'", "mc,s1z_bar'")
    elif opts.internal_correlate_default and ('s1z' in line) and opts.cip_sampler_method == 'GMM':
        # currently ONLY implementing correlations for GMM
        my_sampler_method='GMM'  # Warning can override default sampler setting if not careful!
        if opts.cip_sampler_method:
            my_sampler_method = opts.cip_sampler_method
        addme = " --sampler-method {} --internal-correlate-parameters 'mc,delta_mc,s1z,s2z' ".format(my_sampler_method)
        if 's1z_bar' in line:
            # FIRST attempt to replace with commas, note previous line
            addme = addme.replace('s1z,', 's1z_bar,')
            addme = addme.replace('s2z', 's2z_bar')
        if opts.assume_precessing and ('cos_theta1' in line): # if we are in a polar coordinates step, change the correlated parameters. This is suboptimal.
            addme = addme.replace(',s1z,s2z', ',chi1,cos_theta1')
        # For high-q triggers, don't waste time correlating s2z
        if 'm2' in event_dict:
            if event_dict['m2']/event_dict['m1']< 0.4:
                addme = " --sampler-method {} --internal-correlate-parameters 'mc,delta_mc,s1z' ".format(my_sampler_method)
                if 's1z_bar' in line:
                    addme = addme.replace("mc,s1z'", "mc,s1z_bar'")
            if opts.assume_precessing and ('cos_theta1' in line): # if we are in a polar coordinates step, change the correlated parameters. This is suboptimal.
                addme = addme.replace(',s1z' ',chi1,cos_theta1')
        line += addme

    if opts.cip_sigma_cut:
        line += " --sigma-cut {} ".format(opts.cip_sigma_cut)

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
            line += " --chi-small-max {} ".format(opts.force_chi_small_max)
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
    line += " --eccentricity-prior {}".format(opts.eccentricity_prior)
    if opts.assume_eccentric:
        if opts.use_meanPerAno:
            line += " --parameter meanPerAno --use-meanPerAno "
        if opts.use_eccentricity_squared:
            line += " --use-eccentricity --parameter eccentricity_squared "
        elif opts.use_eccentricity_ln:
            line += " --use-eccentricity --parameter eccentricity_ln "
        else:
            line += " --use-eccentricity --parameter eccentricity "
        # if opts.use_eccentricity_squared:
        #     if opts.use_meanPerAno:
        #         if not(opts.internal_use_aligned_phase_coordinates):
        #             line = line.replace('parameter mc', 'parameter mc --parameter eccentricity_squared --use-eccentricity --parameter meanPerAno --use-meanPerAno')
        #         else:
        #             line = line.replace('parameter-nofit mc', 'parameter-nofit mc --parameter eccentricity_squared --use-eccentricity --parameter meanPerAno --use-meanPerAno')
        #     else:
        #         if not(opts.internal_use_aligned_phase_coordinates):
        #             line = line.replace('parameter mc', 'parameter mc --parameter eccentricity_squared --use-eccentricity')
        #         else:
        #             line = line.replace('parameter-nofit mc', 'parameter-nofit mc --parameter eccentricity_squared --use-eccentricity')
        # else:
        #     if opts.use_meanPerAno:
        #         if not(opts.internal_use_aligned_phase_coordinates):
        #              line = line.replace('parameter mc', 'parameter mc --parameter eccentricity --use-eccentricity --parameter meanPerAno --use-meanPerAno')
        #         else:
        #              line = line.replace('parameter mc', 'parameter mc --parameter eccentricity --use-eccentricity --parameter meanPerAno --use-meanPerAno')
        #     else:
        #         if not(opts.internal_use_aligned_phase_coordinates):
        #             line = line.replace('parameter mc', 'parameter mc --parameter eccentricity --use-eccentricity')
        #         else:
        #             line = line.replace('parameter-nofit mc', 'parameter-nofit mc --parameter eccentricity --use-eccentricity')
        if not(opts.force_ecc_max is None):
            ecc_max = opts.force_ecc_max
            line += " --ecc-max {}  ".format(ecc_max)
        if not(opts.force_ecc_min is None):
            ecc_min = opts.force_ecc_min
            line += " --ecc-min {}  ".format(ecc_min)
        if not(opts.force_meanPerAno_max is None):
            meanPerAno_max = opts.force_meanPerAno_max
            line += " --meanPerAno-max {}  ".format(meanPerAno_max)
        if not(opts.force_meanPerAno_min is None):
            meanPerAno_min = opts.force_meanPerAno_min
            line += " --meanPerAno-min {}  ".format(meanPerAno_min)
    if not(opts.manual_extra_cip_args is None):
        line += " {} ".format(opts.manual_extra_cip_args)  # embed with space on each side, avoid collisions
    line += "\n"
    lines.append(line)

if opts.cip_quadratic_first:
    lines[0]=lines[0].replace(' --fit-method gp ', ' --fit-method quadratic ')
    lines[0]=lines[0].replace(' --parameter delta_mc ', ' --parameter eta ')   # almost without fail we are using mc, delta_mc, xi  as zeroth layer

if opts.assume_eccentric:
    # iteration 0 is eccentricity_squared and nofit meanPerAno
    lines[0] = lines[0].replace('--parameter eccentricity ','--parameter eccentricity_squared ')
    if opts.use_meanPerAno:
        lines[0] = lines[0].replace('--parameter meanPerAno ','--parameter-nofit meanPerAno ')
        if opts.internal_cip_use_periodic_ecc_vars:
            extra_line  =lines[0].replace('parameter eccentricity_squared', 'parameter-implied ecc_cos_meanPerAno --parameter-implied ecc_sin_meanPerAno --parameter-nofit eccentricity')
            lines.insert(1,extra_line)
            n_iterations +=1 
            #for indx in range(2, len(lines)):
            #lines[indx] = lines[indx].replace('parameter eccentricity ', 'parameter-implied ecc_cos_meanPerAno --parameter-implied ecc_sin_meanPerAno --parameter-nofit eccentricity ')
            #lines[indx] = lines[indx].replace('parameter meanPerAno' 'parameter-nofit meanPerAno')

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
   if not(opts.internal_truncate_cip_arg_list is None):
       lines = lines[-opts.internal_truncate_cip_arg_list:]  # truncate the cip arg list file
   # The final CIP group produces both the published posterior and the downstream grid,
   # so it gets the duplicate-free fair draw (capped at sum(w)/max(w)).  Internal
   # iterations keep the fair draw with duplicates allowed, so successive iterations feed
   # an unbiased convergence test.  AMR arg lines drive a different executable that does
   # not accept the flag, so that path is left untouched.
   if not(opts.internal_use_amr):
       lines = flag_final_group_unique(lines)
   for line in lines:
           f.write(line.rstrip("\n") + "\n")

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
puff_max_it = 4 
#  Read puff args from file, if present
try:
    with open("helper_puff_max_it.txt",'r') as f:
        puff_max_it = int(f.readline())
except:
    print( " No puff file ")

instructions_puff = np.loadtxt("helper_puff_args.txt", dtype=str)  # should be one line
puff_params = ' '.join(instructions_puff)
if opts.internal_puff_transverse:
    puff_params = puff_params.replace('--parameter chieff_aligned', '--parameter s1z_bar --parameter s2z_bar ')
    puff_params +=  ' --parameter phi1 --parameter phi2 --parameter chi1_perp_u --parameter chi2_perp_u --reflect-parameter chi1_perp_u --downselect-parameter chi1_perp_u  --downselect-parameter-range [0,1]  --reflect-parameter chi2_perp_u --downselect-parameter chi2_perp_u  --downselect-parameter-range [0,1] '
if opts.internal_cip_transverse_tails:
    # transverse TAIL-GUARD (transverse-spin study 2026-07): every puff APPENDS (and shuffles in)
    # uniformly-random chi1_perp draws (range defaults to [0, chi1-downselect-cap], azimuth also
    # randomized), so the proposed grid keeps offering transverse-tail coverage even after the
    # posterior/grid contracts -- the measured tail-starvation feedback that narrows chi1_perp.
    # Keep the puff active through ALL outer iterations (the nested refinement subdag already
    # puffs every sub-iteration): a guard that turns off at puff-max-it stops guarding.
    # These points carry nonzero s1x/s1y, so they are only physical for a precessing analysis;
    # the bundle is refused above for nonprecessing/zero-spin settings, which is what makes it
    # safe to append them unconditionally here.
    puff_params += ' --append-with-random-parameter chi1_perp --append-with-random-fraction {} '.format(opts.internal_cip_transverse_tails_puff_fraction)
    puff_max_it = max(puff_max_it, 30)
if opts.assume_matter:
#    puff_params += " --parameter LambdaTilde "  # should already be present
    puff_max_it +=5   # make sure we resolve the correlations
if opts.assume_eccentric:
        puff_params += " --downselect-parameter eccentricity --downselect-parameter-range [{},{}] ".format(opts.force_ecc_min,opts.force_ecc_max)
if opts.assume_highq:
        puff_params = puff_params.replace(' delta_mc ', ' eta ')  # use natural coordinates in the high q strategy. May want to do this always
        puff_max_it +=3

if opts.internal_force_puff_iterations is not None:
    puff_max_it = int(opts.internal_force_puff_iterations)

with open("args_puff.txt",'w') as f:
        puff_args =''  # note used below
        if opts.assume_nospin:
            puff_args=puff_params
        elif opts.force_chi_max and not(opts.force_chi_small_max):
            puff_args = puff_params + " --downselect-parameter chi1 --downselect-parameter-range [0,{}]  ".format(opts.force_chi_max)
        elif not(opts.force_chi_max) and (opts.force_chi_small_max):
            puff_args = puff_params + " --downselect-parameter chi2 --downselect-parameter-range [0,{}]  ".format(opts.force_chi_small_max)
        elif opts.force_chi_max and opts.force_chi_small_max:
            puff_args = puff_params + " --downselect-parameter chi1 --downselect-parameter-range [0,{}] --downselect-parameter chi2 --downselect-parameter-range [0,{}] ".format(opts.force_chi_max, opts.force_chi_small_max)
        elif not(opts.force_chi_max) and not(opts.force_chi_small_max):  # nothing set, default, forcce downselect on both spins
            puff_args = puff_params + " --downselect-parameter chi1 --downselect-parameter-range [0,1] --downselect-parameter chi2 --downselect-parameter-range [0,1] "
        else:
            puff_args = puff_params # passthrough case, should not happen ...
        if opts.assume_matter  and not(opts.assume_matter_but_primary_bh):
            lambda_max = 5000
            lambda_small_max=5000
            if opts.force_lambda_max:
                lambda_max = opts.force_lambda_max
            if opts.force_lambda_small_max:
                lambda_small_max = opts.force_lambda_small_max
            # Prevent negative lambda accidentally from puff
            puff_args += " --downselect-parameter lambda1 --downselect-parameter-range [0,{}] --downselect-parameter lambda2 --downselect-parameter-range [0,{}] ".format(lambda_max, lambda_small_max)
        if opts.assume_matter  and opts.assume_matter_but_primary_bh:
#            lambda_max = 0
            lambda_small_max=5000
#            if opts.force_lambda_max:
#                lambda_max = opts.force_lambda_max
            if opts.force_lambda_small_max:
                lambda_small_max = opts.force_lambda_small_max
            # Prevent negative lambda accidentally from puff
            puff_args += " --downselect-parameter lambda2 --downselect-parameter-range [0,{}] ".format(lambda_small_max)
        if False: #opts.cip_fit_method == 'rf':
            # RF can majorly overfit and create 'voids' early on, eliminate the force-away
            # Should only do this in the INITIAL puff, not all, to avoid known problems later
            puff_args = puff_args.replace(unsafe_parse_arg_string(puff_args,'force-away'),'')
        if opts.data_LI_seglen:
                puff_args+= " --enforce-duration-bound " +str(opts.data_LI_seglen)
        if opts.internal_use_force_away:
            puff_args = puff_args.replace(unsafe_parse_arg_string(puff_args,'force-away')," --force-away {} ".format(str(opts.internal_use_force_away)))
        if not(opts.manual_extra_puff_args is None):
            puff_args += " {} ".format(opts.manual_extra_puff_args)  # embed with space on each side, avoid collisions
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
    if opts.manual_initial_grid_supplements:
        if _use_hpip_pp:
            # ligolw_add is XML-only -- the equivalent for hyperpipeline is
            # the head-line + cat | sort | uniq | shuf shell pattern from
            # the BasicIteration join_grids.sh.  We refuse rather than
            # silently produce a broken proposed-grid.
            raise SystemExit(
                "pseudo_pipe: --manual-initial-grid-supplements is XML-only "
                "(uses ligolw_add); incompatible with RIFT_HYPERPIPELINE_FORMAT. "
                "Pre-merge your supplements into a single .dat grid and pass it "
                "via --manual-initial-grid instead.")
        cmd_add = '{}ligolw_add {} proposed-grid.xml.gz --output tmp.xml.gz'.format(ligolw_prefix,opts.manual_initial_grid)
        os.system(cmd_add)
        shutil.copyfile('tmp.xml.gz', "proposed-grid.xml.gz")
    else:
        # shutil.copyfile is format-agnostic: works for either .xml.gz or
        # .dat as long as the source matches the active mode.
        shutil.copyfile(opts.manual_initial_grid, "proposed-grid." + grid_suffix_pp)

# override npts_it if needed
if opts.internal_n_evaluations_per_iteration:
    npts_it = opts.internal_n_evaluations_per_iteration

# Build DAG
cip_mem  = 30000
ile_mem = opts.internal_ile_request_memory
n_jobs_per_worker=opts.ile_jobs_per_worker
if opts.cip_fit_method == 'rf':
    cip_mem = 15000  # more typical for long-duration single-worker runs
if opts.cip_fit_method =='quadratic' or opts.cip_fit_method =='polynomial':  # much lower memory requirement
    cip_mem = 4000
if opts.internal_cip_request_memory:
    cip_mem = opts.internal_cip_request_memory
cepp = "create_event_parameter_pipeline_BasicIteration"
if opts.use_subdags:
    cepp = "create_event_parameter_pipeline_AlternateIteration"
if opts.pipeline_builder:  # explicit override wins, for clean side-by-side A/B testing of the two builders
    if opts.use_subdags and opts.pipeline_builder != "AlternateIteration":
        # use_subdags is set either by the user or force-set by --internal-use-amr (which REQUIRES the Alternate builder)
        print(" WARNING: --pipeline-builder {} overrides --use-subdags routing; AMR/subdag runs require AlternateIteration ".format(opts.pipeline_builder))
    cepp = "create_event_parameter_pipeline_" + opts.pipeline_builder
print(" Pipeline builder (create_event_parameter_pipeline_*): ", cepp)
cmd =cepp+ "  --ile-n-events-to-analyze {} --input-grid proposed-grid.{} --ile-exe  `which integrate_likelihood_extrinsic_batchmode`   --ile-args `pwd`/args_ile.txt --cip-args-list args_cip_list.txt --test-args args_test.txt --request-memory-CIP {} --request-memory-ILE {} --n-samples-per-job ".format(n_jobs_per_worker,grid_suffix_pp,cip_mem,ile_mem) + str(npts_it) + " --working-directory `pwd` --n-iterations " + str(n_iterations) + " --n-iterations-subdag-max {} ".format(opts.internal_n_iterations_subdag_max) + "  --n-copies {} ".format(opts.ile_copies) + "   --ile-retries "+ str(opts.ile_retries) + " --general-retries " + str(opts.general_retries)
if opts.ile_jobs_per_worker_first:
    cmd += " --ile-n-events-to-analyze-first {} ".format(opts.ile_jobs_per_worker_first)
if opts.assume_matter or opts.assume_eccentric:
    cmd +=  " --convert-args `pwd`/helper_convert_args.txt "
if not(opts.ile_runtime_max_minutes is None):
    cmd += " --ile-runtime-max-minutes {} ".format(opts.ile_runtime_max_minutes)
if not(opts.internal_use_amr) or opts.internal_use_amr_puff:
    cmd+= " --puff-exe `which util_ParameterPuffball.py` --puff-cadence 1 --puff-max-it " + str(puff_max_it)+ " --puff-args `pwd`/args_puff.txt "
if opts.calmarg_pilot:
    cmd += " --calmarg-pilot --calmarg-pilot-cadence {} --calmarg-pilot-max-it {} --calmarg-pilot-top-fraction {} --calmarg-pilot-max-points {} ".format(
        opts.calmarg_pilot_cadence, opts.calmarg_pilot_max_it, opts.calmarg_pilot_top_fraction, opts.calmarg_pilot_max_points)
    if opts.use_osg_file_transfer:
        # Graceful degradation for the OSG file-transfer regime.  The wide ILE jobs (and the
        # last-iteration EXTRINSIC ILE jobs) list cal_consolidated_$(macroiterationprev).npz in
        # transfer_input_files; condor HARD-HOLDS (HoldReasonCode 13) if that source file is
        # absent on the submit node.  A calpilot only produces cal_consolidated_<it>.npz for
        # iterations it<=--calmarg-pilot-max-it on-cadence, so any wide/extrinsic iteration
        # whose seed was never produced (e.g. --calmarg-pilot-max-it 1 but 5 wide iterations)
        # would dead-hold.  Pre-seed a VALID prior-breadcrumb placeholder (a copy of the always
        # -present cal_consolidated_-1.npz iteration-0 seed) for EVERY iteration index a wide or
        # extrinsic job can reference.  A real calpilot OVERWRITES its placeholder at runtime via
        # transfer_output_files (the DAG seed barrier guarantees ordering), so behavior is
        # unchanged whenever the learned seed IS produced; a missing seed now falls back to the
        # prior (the placeholder == proposal==prior -> zero-weight prior cal draws) instead of
        # dead-holding.  skip-if-exists preserves real seeds across a DAG rescue/resume.
        _cal_ph_seed = os.getcwd() + "/cal_consolidated_-1.npz"
        if os.path.exists(_cal_ph_seed):
            # wide it in [it_start, n_iterations-1] references prev=it-1; the extrinsic stage
            # (it=n_iterations) references prev=n_iterations-1 -> indices 0 .. n_iterations-1.
            for _kit in range(0, int(n_iterations)):
                _cal_ph_dst = os.getcwd() + "/cal_consolidated_{}.npz".format(_kit)
                if not os.path.exists(_cal_ph_dst):
                    shutil.copyfile(_cal_ph_seed, _cal_ph_dst)
        else:
            print("  WARNING: cal_consolidated_-1.npz placeholder absent; cannot pre-seed missing cal proposal breadcrumbs (wide/extrinsic ILE may hard-hold if a calpilot stage is skipped).")
if opts.extrinsic_handoff:
    cmd += " --extrinsic-handoff --extrinsic-handoff-select {} ".format(opts.extrinsic_handoff_select)
if opts.assume_eccentric:
    cmd += " --use-eccentricity "
    if opts.sample_eccentricity_squared:
        cmd += " --use-eccentricity-squared-sampling "
    if opts.sample_eccentricity_ln:
        cmd += " --use-eccentricity-ln-sampling "
    if opts.use_meanPerAno:
        cmd += " --use-meanPerAno "
if opts.calibration_reweighting and (not opts.bilby_pickle_file):
    cmd += " --calibration-reweighting --calibration-reweighting-exe `which calibration_reweighting.py` --bilby-ini-file {} --bilby-pickle-exe `which bilby_pipe_generation` ".format(str(opts.bilby_ini_file))
    if opts.calibration_reweighting_count:
        cmd+= " --calibration-reweighting-count {} ".format(opts.calibration_reweighting_count)
    if opts.calibration_reweighting_batchsize:
        cmd += " --calibration-reweighting-batchsize {} ".format(opts.calibration_reweighting_batchsize)
    if opts.calibration_reweighting_extra_args:
        cmd += " --calibration-reweighting-extra-args '{}' ".format(opts.calibration_reweighting_extra_args)
    if opts.calibration_reweighting_osg:
        cmd += " --calibration-reweighting-osg "
        opts.calibration_reweighting_initial_extra_args += " --use_local_cal_files "
    if opts.calibration_reweighting_initial_extra_args:
        cmd += " --calibration-reweighting-initial-extra-args '{}' ".format(opts.calibration_reweighting_initial_extra_args)
elif opts.calibration_reweighting and opts.bilby_pickle_file:
    cmd += " --calibration-reweighting --calibration-reweighting-exe `which calibration_reweighting.py` --bilby-pickle-file {} ".format(str(opts.bilby_pickle_file))
    if opts.calibration_reweighting_count:
        cmd+= " --calibration-reweighting-count {} ".format(opts.calibration_reweighting_count)
    if opts.calibration_reweighting_extra_args:
        cmd += " --calibration-reweighting-extra-args '{}' ".format(opts.calibration_reweighting_extra_args)
    if opts.calibration_reweighting_osg:
        cmd += " --calibration-reweighting-osg "
        opts.calibration_reweighting_initial_extra_args += " --use_local_cal_files "
    if opts.calibration_reweighting_initial_extra_args:
        cmd += " --calibration-reweighting-initial-extra-args '{}' ".format(opts.calibration_reweighting_initial_extra_args)
if opts.internal_tabular_eos_file:
    cmd += " --use-tabular-eos-file "
if opts.distance_reweighting:
    cmd += " --comov-distance-reweighting --comov-distance-reweighting-exe `which make_uni_comov_skymap.py` --convert-ascii2h5-exe `which convert_output_format_ascii2h5.py` "
if opts.use_gauss_early:
    cmd += " --cip-exe-G `which util_ConstructIntrinsicPosterior_GaussianResampling.py ` "
if opts.internal_use_amr:
    print(" AMR prototype: Using hardcoded aligned-spin settings, assembling grid, requires coinc!")
    if _use_hpip_pp and opts.manual_initial_grid is None:
        # The AMR seed-grid generators (util_AMRGrid.py /
        # util_GridSubsetOfTemplateBank.py) emit XML and have not been
        # converted to hyperpipeline.  Per the project policy of
        # operating cohesively in one mode or the other, we refuse
        # rather than producing a proposed-grid.xml.gz the rest of the
        # hyperpipeline workflow can't consume.
        raise SystemExit(
            "pseudo_pipe: --internal-use-amr seed-grid auto-generation is "
            "XML-only and incompatible with RIFT_HYPERPIPELINE_FORMAT.  "
            "Stage your initial grid as a hyperpipeline .dat and pass it "
            "via --manual-initial-grid, or run without RIFT_HYPERPIPELINE_FORMAT.")
    cmd += " --cip-exe `which util_AMRGrid.py ` "
    coinc_file = "coinc.xml"
    if not(os.path.exists("coinc.xml")) and not(opts.use_coinc):
        # re-download coinc if not already present
        cmd_event = gracedb_exe + download_request + opts.gracedb_id  + " coinc.xml"
        if not(opts.use_legacy_gracedb):
            cmd_event += " > coinc.xml "
        os.system(cmd_event)
        cmd_fix_ilwdchar = "{}ligolw_no_ilwdchar coinc.xml"; os.system(ligolw_prefix,cmd_fix_ilwdchar) # sigh, need to make sure we are compatible
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
if opts.ile_xpu:
    cmd += " --request-xpu-ILE "
if opts.add_extrinsic:
    cmd += " --last-iteration-extrinsic --last-iteration-extrinsic-nsamples {} ".format(opts.n_output_samples_last)
    if opts.internal_last_iteration_extrinsic_samples_per_ile:
        cmd += " --last-iteration-extrinsic-samples-per-ile {}".format(opts.internal_last_iteration_extrinsic_samples_per_ile)
    if opts.internal_last_iteration_extrinsic_samples_per_ile_internal:
        cmd += " --last-iteration-extrinsic-samples-per-ile-internal {}".format(opts.internal_last_iteration_extrinsic_samples_per_ile_internal)        
    if opts.add_extrinsic_time_resampling:
        cmd+= " --last-iteration-extrinsic-time-resampling "
if opts.batch_extrinsic:
    cmd += " --last-iteration-extrinsic-batched-convert "
if opts.internal_ile_request_disk:
    cmd += " --ile-request-disk {} ".format(opts.internal_ile_request_disk)
if opts.internal_cip_request_disk:
    cmd += " --cip-request-disk {} ".format(opts.internal_ile_request_disk)
if opts.internal_general_request_disk:
    cmd += " --general-request-disk {} ".format(opts.internal_general_request_disk)
if opts.use_ile_subdags:
    cmd += " --ile-group-subdag "
if opts.cip_explode_jobs_dag:  # note name does not match name used in next level below ! Beware!
    cmd += " --cip-explode-jobs-subdag --cip-explode-jobs-dag --cip-explode-jobs 2 "  
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
    elif (opts.internal_truncate_files_for_osg_file_transfer):  # attempt to make copies of frame files, and set up to transfer them with *every* job (!)
        if os.path.exists('local.cache'):
            os.system("util_ForOSG_MakeTruncatedLocalFramesDir.sh .")
        else:
            print(" --- WARNING --- ")
            print(" File truncation not yet performed")
        # if environment variable active, check that frames were created! Fail otherwise
        if 'RIFT_TRUNCATE_CHECK' in os.environ:
            fnames_gwf = os.listdir('./frames_dir/')
            if len(fnames_gwf)< len(event_dict["IFOs"]):
                raise Exception(" Pipeline build failure: Problem generating truncated frames for OSG")
            
#        os.system("echo ../frames_dir >> helper_transfer_files.txt")
        cmd += " --frames-dir `pwd`/frames_dir "
    elif opts.use_osg_file_transfer:
        cmd += " --frames-dir `pwd`/frames_dir "  # assume this will be built by the end user for us, for now
    cmd+= " --transfer-file-list  "+base_dir+"/"+dirname_run+"/helper_transfer_files.txt"
elif opts.ile_additional_files_to_transfer:
    # also transfer files if we request by hand!
    cmd+= " --transfer-file-list  "+base_dir+"/"+dirname_run+"/helper_transfer_files.txt"
if opts.internal_use_oauth_files:
    cmd += " --use-oauth-files {} ".format(opts.internal_use_oauth_files)
if opts.condor_local_nonworker:
    cmd += " --condor-local-nonworker "
if opts.condor_nogrid_nonworker:
    cmd += " --condor-nogrid-nonworker "
if opts.use_osg_simple_requirements:
    cmd += " --use-osg-simple-reqirements "
if opts.archive_pesummary_label:
#    cmd += " --plot-exe `which summarypages` --plot-args  args_plot.txt "
    cmd += " --plot-exe summarypages --plot-args  args_plot.txt "
# Horribly annoying XPHM/XO4a fix because ChooseFDWaveform called.  Seems to be UNIVERSAL for the approximant name, but only if precessing
if opts.internal_mitigate_fd_J_frame == 'rotate' and (opts.approx == 'IMRPhenomXPHM' or 'XO4a' in opts.approx) and opts.assume_precessing:
    cmd += " --frame-rotation "
#if opts.internal_mitigate_fd_J_frame =="L_frame" and not(opts.manual_extra_ile_args) and not(opts.use_gwsignal):
#    cmd +=" --calibration-reweighting-initial-extra-args='--internal-waveform-fd-L-frame' "
if opts.calibration_reweighting:
    my_extra_string = ''
    if opts.use_gwsignal:
        my_extra_string = ' --use-gwsignal '
    if opts.assume_eccentric:
        my_extra_string += " --use-eccentricity "
    if opts.manual_extra_ile_args:
         print(" calmarg: Parsing  ", opts.manual_extra_ile_args)
         my_str_list = opts.manual_extra_ile_args.lstrip().split("--")
         my_revised_args = []
         # MANUAL PARSING, SO STUPID, but argparse does not do what I want
         for arg_item in my_str_list:
             if 'internal-waveform-extra-lalsuite-args' in arg_item:
                 my_revised_args += ['--internal-waveform-extra-lalsuite-args', arg_item.replace('internal-waveform-extra-lalsuite-args', '')]
             if 'internal-waveform-extra-kwargs' in arg_item:
                 my_revised_args += ['--internal-waveform-extra-kwargs', arg_item.replace('internal-waveform-extra-kwargs', '')]
        
         # Parse string for waveform arguments
         # Currently: fork off the lmax_nyquist (the most common scenario), leave the rest to a unified dictionary to pass on
         # ISSUE: argparse parsing does not seem to work, fall back to optparse
         my_parser=argparse.ArgumentParser()
         my_parser.add_argument("--internal-waveform-extra-lalsuite-args",type=str,default=None)
         my_parser.add_argument("--internal-waveform-extra-kwargs",type=str, default=None)
         my_opts, unknown_opts =my_parser.parse_known_args(my_revised_args )
         print(' calmarg: parsed args ', my_opts, " and others ", unknown_opts)
         my_extra_args = {}
         if my_opts.internal_waveform_extra_kwargs:
             my_arg_dict = eval(my_opts.internal_waveform_extra_kwargs)
             # due to quoting, might not evaluate to a dictionary
             if not(isinstance(my_arg_dict, dict)):
                 my_arg_dict = eval(my_arg_dict)
             if 'lmax_nyquist' in my_arg_dict:
                 my_extra_string+= " --use-gwsignal-lmax-nyquist {} ".format(my_arg_dict['lmax_nyquist'])
                 del my_arg_dict['lmax_nyquist'] # remove key
             my_extra_args.update(my_arg_dict)
         if my_opts.internal_waveform_extra_lalsuite_args:
             my_arg_dict = eval(my_opts.internal_waveform_extra_lalsuite_args)
             if not(isinstance(my_arg_dict, dict)):
                 my_arg_dict = eval(my_arg_dict)
             my_extra_args.update(my_arg_dict)
         if my_extra_args:
            my_extra_string += ' --extra-waveform-kwargs "{}" '.format(my_extra_args)
#         my_extra_string += ' ' + opts.manual_extra_ile_args + ' '
    if opts.use_ini:
        fref = unsafe_config_get(config,['engine','fref'])
        my_extra_string += ' --fref {} '.format(fref)
    if (opts.internal_mitigate_fd_J_frame =="L_frame"):
        my_extra_string += ' --internal-waveform-fd-L-frame '
    if opts.calibration_reweighting_initial_extra_args:
        my_extra_string+= ' {} '.format(opts.calibration_reweighting_initial_extra_args) # make sure to add spaces/padding
    cmd +=" --calibration-reweighting-initial-extra-args='  {}' ".format(my_extra_string)
#if opts.internal_mitigate_fd_J_frame =="L_frame" and opts.use_gwsignal and not(opts.manual_extra_ile_args):
#    cmd +=" --calibration-reweighting-initial-extra-args='--internal-waveform-fd-L-frame --use-gwsignal' "
if opts.condor_local_nonworker_igwn_prefix:
    cmd += " --condor-local-nonworker-igwn-prefix "

# Make copy of local.cache for use in file transfer
if opts.use_osg_file_transfer and opts.internal_truncate_files_for_osg_file_transfer and os.path.exists('local.cache'):
    shutil.copyfile('local.cache', 'local_orig.cache')
    # Move contents of ile_pre.sh here
    os.system("cat local.cache > awk '{print $1, $2, $3, $4}' > local_stripped.cache")
    os.system('for i in `ls frames_dir/*.gwf`; do echo frames_local/${i} ; done > base_paths.dat') # yes probably easier to do the ls myself
    os.system("paste local_stripped.cache base_paths.dat > local_relative.cache ")
    os.system("cp local_relative.cache local.cache")

if not(ile_condor_commands is None):
    # create file
    with open("ile_condor_commands.txt", 'w') as f:
        for key, val in ile_condor_commands:
            f.write(key+ '  ' + val + '\n')
    cmd += " --ile-condor-commands `pwd`/ile_condor_commands.txt "

# Per-distance likelihood export on the extrinsic stage. These are
# pipeline-builder flags consumed by create_event_parameter_pipeline_*, so
# they are added to the CEPP command (not the ILE args). They only take effect
# when the extrinsic stage exists (--add-extrinsic).
if opts.export_marginal_distance_grid:
    cmd += " --last-iteration-export-marginal-distance-grid "
if opts.export_distance_slices and opts.export_distance_slices > 0:
    cmd += " --last-iteration-export-distance-slices {} ".format(opts.export_distance_slices)
    if opts.export_distance_slices_all_fresh:
        cmd += " --last-iteration-export-distance-slices-all-fresh "
    if opts.export_distance_slices_randomize:
        cmd += " --last-iteration-export-distance-slices-randomize "
    if opts.export_distance_slices_wing_neff is not None:
        cmd += " --last-iteration-export-distance-slices-wing-neff {} ".format(opts.export_distance_slices_wing_neff)
    if opts.export_distance_slices_wing_nmax is not None:
        cmd += " --last-iteration-export-distance-slices-wing-nmax {} ".format(opts.export_distance_slices_wing_nmax)
    if opts.export_distance_slices_n_core:
        cmd += " --last-iteration-export-distance-slices-n-core {} ".format(opts.export_distance_slices_n_core)
    if opts.export_distance_slices_n_wing:
        cmd += " --last-iteration-export-distance-slices-n-wing {} ".format(opts.export_distance_slices_n_wing)
    if opts.export_distance_slices_wing_delta_lnL is not None:
        cmd += " --last-iteration-export-distance-slices-wing-delta-lnL {} ".format(opts.export_distance_slices_wing_delta_lnL)
    if opts.export_distance_slices_skip_threshold is not None:
        cmd += " --last-iteration-export-distance-slices-skip-threshold {} ".format(opts.export_distance_slices_skip_threshold)

print(cmd)
os.system(cmd)

if opts.internal_ile_check_good_enough:
    # Populate 'ile_check_good_enough' through all subdirectories
    cmd_enough = r"find . -name 'iter*ile' -type d -exec touch {}/ile_good_enough \; "
    os.system(cmd)

if opts.use_osg_file_transfer and opts.internal_truncate_files_for_osg_file_transfer:
    if opts.fake_data_cache:
        shutil.copyfile(opts.fake_data_cache, 'local.cache')
    # build truncated frames.  Note this parses ILE arguments, so must be done last
    if os.path.exists('local.cache'):
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
