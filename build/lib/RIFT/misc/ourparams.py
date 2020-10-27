"""
ourparams.py
   - Intended to hold all standard arguments, either in 
      - default command line settings OR 
      - via  hardcoded defaults
   - Provides command line parsing
   - Provides explicit settings for sampler parameters
"""

import argparse
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
import lal

from glue.ligolw import utils, lsctables, table, ligolw

try:
    hasNR=True
    import NRWaveformCatalogManager3 as nrwf
except:
    hasNR=False
    print(" - no NR waveforms - ")

rosDebugMessagesDictionary = {}   # Mutable after import (passed by reference). Not clear if it can be used by caling routines
                                                  # BUT if every module has a `PopulateMessagesDictionary' module, I can set their internal copies


def ParseStandardArguments():
    global rosDebugMessagesDictionary

    rosDebugMessagesDictionary["Test"] = True
    rosDebugMessagesDictionary["DebugMessages"] = False
    rosDebugMessagesDictionary["DebugMessagesLong"] = False
    

    parser = argparse.ArgumentParser()
#    print " === Loading options === "
#    print "  ++ NOTE OPTIONS MUST BE IMPLEMENTED BY EACH ROUTINE INDIVIDUALLY - NOT YET STANDARDIZED  ++ "
    # Options.  Try to be consistent with lalinference

    # General parameters
    parser.add_argument("--Niter", dest='nmax',default=100000,type=int,help="Number of iterations")
    parser.add_argument("--Neff", dest='neff', default=100,type=int,help="Target number of effective samples")
    parser.add_argument("--Nskip", dest='nskip', default=2000,type=int,help="How often MC progress is reported (in verbose mode)")
    parser.add_argument("--convergence-tests-on", default=False,action='store_true')
    parser.add_argument("--d-max",default=10000,type=float,help="Maximum distance in Mpc")

    # Likelihood functions
    parser.add_argument("--LikelihoodType_raw",default=False,action='store_true')
    parser.add_argument("--LikelihoodType_MargPhi",default=False,action='store_true',help="Deprecated/disabled")
    parser.add_argument("--LikelihoodType_MargT",default=False,action='store_true',help="Deprecated/disabled")
    parser.add_argument("--LikelihoodType_MargTdisc",default=False,action='store_true',help="Deprecated/disabled")
    parser.add_argument("--LikelihoodType_MargTdisc_array",default=True,action='store_true',help="Default")
    parser.add_argument("--LikelihoodType_MargTdisc_array_vector",default=False,action='store_true',help="Use matrix operations to compute likelihood. Discrete....ViaArray is called. Inputs are scalars")
    parser.add_argument("--LikelihoodType_vectorized",default=False,action='store_true',help="Use matrix operations to compute likelihood. Discrete....ViaArrayVector is called. Inputs are vectors")
    parser.add_argument("--LikelihoodType_vectorized_noloops",default=False,action='store_true',help="Use matrix operations to compute likelihood. Discrete....ViaArrayVectorNoLoop is called. Inputs are vectors.  GPU version on this branch")
    parser.add_argument("--adapt-parameter", action='append',help = "Adapt in this parameter (ra, dec, tref, incl,dist,phi,psi)")
    parser.add_argument( "--adapt-beta", type=float,default=1)
    parser.add_argument("--adapt-adapt",action='store_true',help="Adapt the tempering exponent")
    parser.add_argument("--adapt-log",action='store_true',help="Use a logarithmic tempering exponent")
    parser.add_argument("--adapt-mix", type=float,default=0.1)
    parser.add_argument("--no-adapt-distance",  default=False, action='store_true')
    parser.add_argument("--no-adapt-sky", default=False,action='store_true')
    # Infrastructure choices:
    parser.add_argument("--skip-interpolation",dest="opt_SkipInterpolation",default=False,action='store_true')  # skip interpolation : saves time, but some things will crash later
    parser.add_argument("--skip-modes-less-than",dest="opt_SkipModeThreshold",default=1e-5,type=float, help="Skip modes with relative effect less than this factor times the 22 mode, in the overlap matrix 'U'. So a choice of 1e-2 roughly  modes greater 10%% of the 22 mode amplitude are included. Should be smaller than 1/snr^2.")

    # Noise model
    parser.add_argument("--psd-name", type=str, default="", help="psd name ('eval'). lal.LIGOIPsd, lalsim.SimNoisePSDiLIGOSRD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, ... ")
    parser.add_argument("--psd-name-V", type=str, default="", help="psd name ('eval'). lalsim.SimNoisePSDVirgo")
    parser.add_argument("--psd-file",dest='psd_file',default=None, help="psd-file for all instruments. Assumes PSD for all instruments is provided in the file.  If None, uses an analytic PSD")
    parser.add_argument("--psd-file-singleifo",dest="psd_file_singleifo",action="append", help="psd file for one instrument. Not available for all cases")
    parser.add_argument("--psd-truncate-inverse",dest="psd_TruncateInverse",default=True,action='store_true')
    parser.add_argument("--psd-truncate-inverse-time",dest="psd_TruncateInverseTime",default=8,type=float)

#    parser.add_argument("--psd-file",dest='psd_file',default=None, help="instrument=psd-file, e.g. H1=H1_PSD.xml.gz. Can be given multiple times for different instruments. If None, uses an analytic PSD.  Implemented for compatibility with Pankow")

    # Numerical relativity file (will override approximant used as *template*)
    parser.add_argument("--NR-signal-group", default=None,help="Specific NR simulation group to use")
    parser.add_argument("--NR-signal-param", default=None,help="Parameter value")
    parser.add_argument("--NR-template-group", default=None,help="Specific NR simulation group to use")
    parser.add_argument("--NR-template-param", default=None,help="Parameter value")

    parser.add_argument("--ROM-template-group",default=None)
    parser.add_argument("--ROM-template-param",default=None)

    # Use external EOB matlab code
    parser.add_argument("--use-external-EOB",default=False,action='store_true')

    # Data or injection
    parser.add_argument("-c","--cache-file",dest='cache_file', default=None, help="LIGO cache file containing all data needed.")
    parser.add_argument("-C","--channel-name", action="append", help="instrument=channel-name, e.g. H1=FAKE-STRAIN. Can be given multiple times for different instruments.")
    parser.add_argument("--force-read-full-frames", dest='opt_ReadWholeFrameFilesInCache', action='store_true')
    parser.add_argument("--inj-xml", dest='inj', default=None,help="inspiral XML file containing injection information.")
    parser.add_argument("--event",type=int, dest="event_id", default=0,help="event ID of injection XML to use.")
    parser.add_argument("--coinc-xml", dest='coinc', help="gstlal_inspiral XML file containing coincidence information.")
    parser.add_argument("--seglen", dest='seglen', type=int, default=32, help="Minimum segment duration surrounding coing to be analyzed. (Will be rounded up to next power of 2).  ")
    parser.add_argument( "--padding", dest='padding',  type=int, default=2, help="Time window after the trigger to be included in the data segment.")
    parser.add_argument("--srate", dest='srate', type=int, default=4096, help="Sampling rate to use. Data filtered down to this rate, PSD extended up to this rate. Can conflict with stored PSD file.")

    # Sampling prior
    parser.add_argument("--sampling-prior-use-inj-sky-location",dest='opt_UseKnownSkyPosition',action='store_true')
    parser.add_argument("--sampling-prior-use-skymap",dest='opt_UseSkymap')
    parser.add_argument("--rotate-sky-coordinates", action='store_true')

    # Lower frequencies 
    parser.add_argument("--fmin-snr", dest='fmin_SNR', type=float, default=30)
    parser.add_argument("--fmin-template", dest='fmin_Template', type=float, default=35)  # slightly higher than signal, so shorter and event is offset (satisfy Evan's t_shift>0 assert)
    parser.add_argument("--fmax-snr", dest='fmax_SNR', type=float, default=2000)

    # Options to set template physics
    parser.add_argument("--fref", dest='fref', type=float, default=100.0, help="Waveform reference frequency [template]. Required, default is 100 Hz.")
    parser.add_argument("--amporder", dest='amporder', type=int, default=0, help="Amplitude order of PN waveforms")
    parser.add_argument("--Lmax", dest="Lmax", type=int, default=2, help="Lmax (>=2) to use")
    parser.add_argument( "--order",dest='order', type=int, default=-1, help="Phase order of PN waveforms")
    parser.add_argument("--spinOrder", dest='spinOrder', type=int, default=-1, help="Specify twice the PN order (e.g. 5 <==> 2.5PN) of spin effects to use, only for LALSimulation (default: -1 <==> Use all spin effects).")
    parser.add_argument("--noSpin",  action="store_false", dest="SpinQ",default=False, help="template will assume no spins.")
    parser.add_argument("--spinAligned", action="store_true", dest="SpinAlignedQ",default=False, help="template will assume spins aligned with the orbital angular momentum.")
    parser.add_argument("--approx",type=str, default="TaylorT4", help="Specify a template approximant and phase order to use.") # A tighter integration with GetStringFromApproximant and vice versa would be nice

    # Fix parameters at injected values.  Assumes injected values present
    parser.add_argument("--fix-rightascension", action='append_const', dest='fixparams',const='right_ascension',default=[])
    parser.add_argument("--fix-declination", action='append_const', dest='fixparams',const="declination")
    parser.add_argument("--fix-polarization", action='append_const', dest='fixparams',const='psi')
    parser.add_argument("--fix-distance", action='append_const', dest='fixparams',const='dist')
    parser.add_argument("--fix-time", action='append_const', dest='fixparams',const='tref')
    parser.add_argument("--fix-inclination", action='append_const', dest='fixparams',const='incl')
    parser.add_argument("--fix-phase", action='append_const', dest='fixparams',const='phi')

    parser.add_argument("--force-gps-time", dest='force_gps_time', default=None,type=numpy.float64)

    # Extra options to set verbosity level
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False)
    parser.add_argument("--super-verbose", action="store_true", dest="super_verbose", default=False)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose", default=False)

    # Test options
    parser.add_argument("--mass1", dest="template_mass1", default=False, type=float)
    parser.add_argument("--mass2", dest="template_mass2", default=False, type=float)
    parser.add_argument("--signal-mass1", dest="signal_mass1", default=False, type=float)
    parser.add_argument("--signal-mass2", dest="signal_mass2", default=False, type=float)
    parser.add_argument("--eff-lambda", type=float, help="Value of effective tidal parameter. Optional, ignored if not given")
    parser.add_argument("--deff-lambda", type=float, help="Value of second effective tidal parameter. Optional, ignored if not given")
    parser.add_argument("--indicate-mass1", dest="indicate_mass1", default=False, type=float)
    parser.add_argument("--indicate-mass2", dest="indicate_mass2", default=False, type=float)
    parser.add_argument("--signal-fmin",dest="signal_fmin",default=30,type=float)
    parser.add_argument("--signal-fref",dest="signal_fref",default=0.,type=float, help="Override the default fref(=0) of XML injections ONLY")
    parser.add_argument("--signal-distance",dest="signal_distMpc",default=25,type=float)
    parser.add_argument("--signal-inclination", dest="signal_incl",default=False,type=float)
    parser.add_argument("--signal-polarization", dest="signal_psi",default=False,type=float)
    parser.add_argument("--signal-phase", dest="signal_phiref",default=False,type=float)
    parser.add_argument("--signal-time", dest="signal_tref",default=False,type=float,help="If a small floating point number (<1000), added to the fiducial epoch to shift the injection time. Otherwise, the GPS time of the injection; the reference time is set to that event time.")

    # Interactive plots
    parser.add_argument("--show-input-h",dest='plot_ShowH', action='store_true', default=False,help ="Show interactive plots of h(t). ")
    parser.add_argument("--show-sampler-results",dest='plot_ShowSampler', action='store_true', default=False,help ="Show interactive plots from sampler results. ")
    parser.add_argument("--show-sampler-inputs",dest='plot_ShowSamplerInputs', action='store_true', default=False,help ="Show interactive plots from sampler *inputs*. ")
    parser.add_argument("--show-likelihood-versus-time",dest="plot_ShowLikelihoodVersusTime", action='store_true',default=False,help="Show single-IFO likelihoods versus time. If injection parameters are known, also plot L(t) in the geocenter. Useful to confirm a signal has been correctly windowed at all!")
    parser.add_argument("--show-psd",dest='plot_ShowPSD',action='store_true',default=False,help="Plot PSDs, as imported and stored.")
    parser.add_argument("--no-interactive-plots",action='store_true', default=False)

    # File output and conditions controlling output
    parser.add_argument("--save-sampler-file", dest="points_file_base",default="sampler-output-file")
    parser.add_argument("--save-threshold-fraction", dest="points_threshold_match",type=float,default=0.0,help="Roughly speaking, target match for points to be returned.  In practice, all points with L> \sqrt{P}L_{max} are returned")
    parser.add_argument("--save-P",default=0.,type=float)
    parser.add_argument("--save-deltalnL",type=float, default=float("Inf"), 
                        help="Threshold on deltalnL for points preserved in output file.  Requires --output-file to be defined")
    parser.add_argument("--save-metadata", dest="force_store_metadata",action="store_true")
#    parser.add_argument("--save-metadata-file", dest="fname_metadata",default=None)
    parser.add_argument("--use-metadata", dest="force_use_metadata",action="store_true")
    parser.add_argument("--use-metadata-file", dest="fname_metadata",default=None)

    # DAG generation (only for dag-generating scripts)
    parser.add_argument("--code-dir", default=None,help="dag: Where to look for source (needed for safe submit files)")
    parser.add_argument("--n-queue", default=1,type=int, help="dag: How many jobs to queue")
    parser.add_argument("--n-intrinsic", default=100,type=int, help="dag: How many intrinsic points to sample")

    # Unused: To suck up parameters by letting me do search/replace on text files
    parser.add_argument("--unused-argument", default=None, help="Used to sop up arguments I want to replace")

    parser.add_argument("--save-no-samples",action='store_true')

    args = parser.parse_args()
    # Connect debugging options to argument parsing
    rosDebugMessagesDictionary["DebugMessages"] = args.verbose
    rosDebugMessagesDictionary["DebugMessagesLong"] = args.super_verbose
    if (rosDebugMessagesDictionary["DebugMessagesLong"]):
        rosDebugMessagesDictionary["DebugMessages"]=True

    # Error check: confirm that the desired template exists
#    print nrwf.internal_ParametersAvailable.keys()
    if hasNR and not ( args.NR_template_group in nrwf.internal_ParametersAvailable.keys()):
#        raise( nrwf.NRNoSimulation,args.NR_template_group)
        if args.NR_template_group:
            print(" ===== UNKNOWN NR PARAMETER ====== ")
            print(args.NR_template_group, args.NR_template_param)
    elif hasNR:
        if args.NR_template_param:
            args.NR_template_param = eval(args.NR_template_param) # needs to be evaluated
        if not ( args.NR_template_param in nrwf.internal_ParametersAvailable[args.NR_template_group]):
            print(" ===== UNKNOWN NR PARAMETER ====== ")
            print(args.NR_template_group, args.NR_template_param)

    return args, rosDebugMessagesDictionary

###
### Populate detector network and single-detector SNRs
###
def PopulateTriggerSNRs(opts):
    rhoExpected ={}
    # Read in *coincidence* XML (overridden by injection, if present)
    if opts.coinc:
        xmldoc = utils.load_filename(opts.coinc,contenthandler =lalsimutils.cthdler)
        coinc_table = table.get_table(xmldoc, lsctables.CoincInspiralTable.tableName)
        assert len(coinc_table) == 1
        coinc_row = coinc_table[0]
       # Populate the SNR sequence and mass sequence
        sngl_inspiral_table = table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
        m1, m2 = None, None
        for sngl_row in sngl_inspiral_table:
            # NOTE: gstlal is exact match, but other pipelines may not be
            assert m1 is None or (sngl_row.mass1 == m1 and sngl_row.mass2 == m2)
            m1, m2 = sngl_row.mass1, sngl_row.mass2
            rhoExpected[str(sngl_row.ifo)] = sngl_row.snr  # record for comparisons later

    return rhoExpected

###
### Populate Psig from either inj xml, coinc, command-line options, or default choice:
###
def PopulatePrototypeSignal(opts):
    approxSignal = lalsim.GetApproximantFromString(opts.approx)
    approxTemplate = approxSignal
    ampO =opts.amporder # sets which modes to include in the physical signal
    Lmax = opts.Lmax # sets which modes to include
    fref = opts.fref
    fminWavesSignal = opts.fmin_Template  # too long can be a memory and time hog, particularly at 16 kHz
    fminSNR =opts.fmin_SNR
    fSample = opts.srate

    Psig = None

    # Read in *coincidence* XML (overridden by injection, if present)
    if opts.coinc:
        xmldoc = utils.load_filename(opts.coinc)
        coinc_table = table.get_table(xmldoc, lsctables.CoincInspiralTable.tableName)
        assert len(coinc_table) == 1
        coinc_row = coinc_table[0]
       # Populate the SNR sequence and mass sequence
        sngl_inspiral_table = table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
        m1, m2 = None, None
        for sngl_row in sngl_inspiral_table:
            # NOTE: gstlal is exact match, but other pipelines may not be
            assert m1 is None or (sngl_row.mass1 == m1 and sngl_row.mass2 == m2)
            m1, m2 = sngl_row.mass1, sngl_row.mass2
        m1 = m1*lal.MSUN_SI
        m2 = m2*lal.MSUN_SI
       # Create a 'best recovered signal'
        Psig = lalsimutils.ChooseWaveformParams(
            m1=m1,m2=m2,approx=approxSignal,
            fmin = fminWavesSignal, 
            dist=factored_likelihood.distMpcRef*1e6*lal.PC_SI,    # default distance
            fref=fref, 
            ampO=ampO)  # FIXME: Parameter mapping from trigger space to search space

    # Read in *injection* XML
    if opts.inj:
        Psig = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))[opts.event_id]  # Load in the physical parameters of the injection.  
        Psig.deltaT = 1./fSample   # needed if we will generate fake data from the injection xml *by this program, internally* 
        timeWaveform = lalsimutils.estimateWaveformDuration(Psig)
        Psig.deltaF = 1./lalsimutils.nextPow2(opts.seglen)       # Frequency binning needs to account for target segment length


    if  not Psig and opts.channel_name:  # If data loaded but no signal generated
            if (not opts.template_mass1) or (not opts.template_mass2) or (not opts.force_gps_time):
                print(" CANCEL: Specifying parameters via m1, m2, and time on the command line ")
            Psig = lalsimutils.ChooseWaveformParams(approx=approxSignal,
                                                    fmin = fminWavesSignal, 
                                                    dist=factored_likelihood.distMpcRef*1e6*lal.PC_SI,    # default distance
                                                    fref=fref)
            Psig.m1 = lal.MSUN_SI*opts.template_mass1
            Psig.m2 = lal.MSUN_SI*opts.template_mass2
            Psig.tref = lal.LIGOTimeGPS(0.000000000)  # Initialize as GPSTime object
            Psig.tref += opts.force_gps_time # Pass value of float into it

            
    if not(Psig):
            m1 = 4*lal.MSUN_SI
            m2 = 3*lal.MSUN_SI
            Psig = lalsimutils.ChooseWaveformParams(
                m1 = m1,m2 =m2,
                fmin = fminWavesSignal, 
                fref=fref, 
                approx=approxSignal,
                ampO=ampO
                )
    # Use forced parameters, if provided
    if opts.template_mass1:
        Psig.m1 = opts.template_mass1*lal.LAL_MSUN_SI
    if opts.template_mass2:
        Psig.m2 = opts.template_mass2*lal.LAL_MSUN_SI

    return Psig

###
### Populate standard sampler arguments
###
import RIFT.integrators.mcsampler as mcsampler
import numpy 
import functools

import RIFT.likelihood.factored_likelihood as factored_likelihood
tWindowExplore  = factored_likelihood.tWindowExplore
Tmax = numpy.max([0.05,tWindowExplore[1]]) # max ref. time
Tmin = numpy.min([-0.05,tWindowExplore[0]]) # min ref. time
Dmax = 10000 #* 1.e6 * lal.LAL_PC_SI # max distance. CAN CAUSE PROBLEMS with the CDF integrator if this limit is large (SI units). SHOULD BE OVERRIDDED
Dmin = 1. # * 1.e6 * lal.LAL_PC_SI      # min distance.  Needs to be nonzero to avoid integrator stupid problems with reconstructing the CDF

# set up bounds on parameters
# Polarization angle
psi_min, psi_max = 0, numpy.pi
# RA and dec
ra_min, ra_max = 0, 2*numpy.pi
dec_min, dec_max = -numpy.pi/2, numpy.pi/2
# Reference time
tref_min, tref_max = Tmin, Tmax
# Inclination angle
inc_min, inc_max = 0, numpy.pi
# orbital phi
phi_min, phi_max = 0, 2*numpy.pi
# distance
dist_min, dist_max = Dmin, Dmax


pFocusEachDimension = 0.03

def PopulateSamplerParameters(sampler, theEpochFiducial, tEventFiducial,distBoundGuess, Psig, opts):

    pinned_params ={}

    print(" Trying to pin parameters ", opts.fixparams)

    
    adapt_ra = (not opts.no_adapt_sky) #and ('ra' in opts.adapt_paramter)
    adapt_dec = (not opts.no_adapt_sky)# and ('dec' in opts.adapt_parameter)
    adapt_dist = (not opts.no_adapt_distance) # and ('dist' in opts.adapt_parameter)
   
    adapt_incl=False;adapt_phi=False; adapt_psi=False
    if opts.adapt_parameter:
        adapt_incl = ('incl' in opts.adapt_parameter)
        adapt_phi = 'phi' in opts.adapt_parameter
        adapt_psi = 'psi' in opts.adapt_parameter


    # Uniform sampling (in area) but nonuniform sampling in distance (*I hope*).  Auto-cdf inverse
    if not(opts.opt_UseSkymap):
        if opts.opt_UseKnownSkyPosition:
            print(" +++ USING INJECTED SKY POSITION TO NARROW THE SKY SEARCH REGION ")
            sampler.add_parameter("right_ascension", functools.partial(mcsampler.gauss_samp, Psig.phi,0.03), None, ra_min, ra_max, 
                              prior_pdf = mcsampler.uniform_samp_phase)
            sampler.add_parameter("declination", functools.partial(mcsampler.gauss_samp, Psig.theta,0.03), None, dec_min, dec_max, 
                              prior_pdf= mcsampler.uniform_samp_dec)
        else:
            sampler.add_parameter("right_ascension", mcsampler.uniform_samp_phase, None, ra_min, ra_max, 
                                      prior_pdf = mcsampler.uniform_samp_phase,
                                  adaptive_sampling= adapt_ra
                                  )
            sampler.add_parameter("declination",mcsampler.uniform_samp_dec, None, dec_min, dec_max, 
                          prior_pdf= mcsampler.uniform_samp_dec,
                                  adaptive_sampling= adapt_dec)
            if opts.fixparams.count('right_ascension') and Psig != None:
                print("  ++++ Fixing ra to injected value +++ ")
                pinned_params['right_ascension'] = Psig.phi
                #sampler.add_pinned_parameter('ra', Psig.phi )
            if opts.fixparams.count('declination')  and Psig:
                    print("  ++++ Fixing dec to injected value +++ ")
                    pinned_params['declination'] = Psig.theta
                    #sampler.add_pinned_parameter('dec', Psig.theta )
    # Override *everything* if I have a skymap used
    else:
        print("  ++++ USING SKYMAP +++ ")
        print(" ==Loading skymap==")
        print("   skymap file ", opts.opt_UseSkymap)

        # Import modules used for skymaps only here (they will be available by reference later)
        import healpy
        from lalinference.bayestar import fits as bfits
        from lalinference.bayestar import plot as bplot
    
        smap, smap_meta = bfits.read_sky_map(opts.opt_UseSkymap)
        # Modify skymap to regularize it: don't want zero probability anywhere!
        smap = 0.999 *smap + 0.001*numpy.ones(len(smap))  # 0.11% probability of being somewhere randomly on the sky
        sides = healpy.npix2nside(len(smap))
        cum_smap = numpy.cumsum(smap)

        import bisect
        def bayestar_cdf_inv(x,y):                 # it will be passed two arguments, use one
            indx = bisect.bisect(cum_smap,x) 
            th,ph = healpy.pix2ang(sides, indx)
            if rosDebugMessagesDictionary["DebugMessagesLong"]:
                print(" skymap used x->(th,ph) :", x,th,ph)
            return ph,th-numpy.pi/2.

        def bayestar_cdf_inv_vector(x,y):   # Manually vectorize, so I can insert pdb breaks
            indxVec = map(lambda z: bisect.bisect(cum_smap,z), x) 
            th, ph = healpy.pix2ang(sides,indxVec)
            return numpy.array([ph, th-numpy.pi/2.])                            # convert to RA, DEC from equatorial. Return must be pair of arrays

        def bayestar_pdf_radec(x,y):               # look up value at targted pixel 
            ra,dec = x,y
            indx = bplot._healpix_lookup(smap,  ra, dec) # note bplot takes lon (=ra, really), lat (dec). Not sure why
            return smap[indx]

        sampler.add_parameter(("right_ascension", "declination"), 
                              None, 
                              bayestar_cdf_inv_vector, (ra_min, dec_min), (ra_max, dec_max))
        sampler.prior_pdf[("right_ascension", "declination")] = numpy.vectorize(lambda x,y: 1./len(smap)) # 1/number of pixels
        sampler.pdf[("right_ascension", "declination")] =          numpy.vectorize(bayestar_pdf_radec)              # look up value at targteed pixel 
        sampler._pdf_norm[("right_ascension", "declination")] = 1


    sampler.add_parameter("psi", functools.partial(mcsampler.uniform_samp_vector, psi_min, psi_max), None, psi_min, psi_max,
                      prior_pdf =mcsampler.uniform_samp_psi,
                          adaptive_sampling=adapt_psi)
    if opts.fixparams.count('psi') and Psig:
        print("  ++++ Fixing psi to injected value +++ ")
        pinned_params['psi'] = Psig.psi


    def gauss_samp_withfloor(mu, std, myfloor, x):
        return 1.0/numpy.sqrt(2*numpy.pi*std**2)*numpy.exp(-(x-mu)**2/2/std**2) + myfloor
    
    sampler.add_parameter("t_ref", functools.partial(mcsampler.gauss_samp_withfloor, tEventFiducial, 0.01,0.001), None, tref_min, tref_max, 
                      prior_pdf = functools.partial(mcsampler.uniform_samp_vector, tWindowExplore[0],tWindowExplore[1]))
    if opts.fixparams.count('tref') and Psig:
        print("  ++++ Fixing time to injected value +++ ")
        pinned_params['t_ref']  = float(Psig.tref - theEpochFiducial)
        #sampler.add_pinned_parameter("tref", float(Psig.tref - theEpochFiducial) )

    # Phase (angle of L)
    sampler.add_parameter("phi_orb", functools.partial(mcsampler.uniform_samp_vector, phi_min, phi_max), None, phi_min, phi_max,
                      prior_pdf = mcsampler.uniform_samp_phase,
                          adaptive_sampling = adapt_phi)
    if opts.fixparams.count('phi') and Psig:
            print("  ++++ Fixing phi to injected value +++ ")
            pinned_params['phi_orb'] = Psig.phiref
#            sampler.add_pinned_parameter("phi", Psig.phiref )


    # Inclination (angle of L; nonspinning)
    sampler.add_parameter("inclination", functools.partial(mcsampler.cos_samp_vector), None, inc_min, inc_max,
                      prior_pdf = mcsampler.uniform_samp_theta,
                          adaptive_sampling = adapt_incl)
    if opts.fixparams.count('incl') and Psig:
        print("  ++++ Fixing incl to injected value +++ ")
        pinned_params['inclination'] = Psig.incl
#        sampler.add_pinned_parameter("incl", Psig.incl )

    # Distance
    def quadratic_samp_withfloor(rmaxQuad,rmaxFlat,pFlat,x):
            ret = 0.
            if x<rmaxQuad:
                ret+= (1-pFlat)* x*x/(3*numpy.power(rmaxQuad,3))
            if x<rmaxFlat:
                ret +=pFlat/rmaxFlat
            return  ret
    def uniform_samp_withfloor(rmaxQuad,rmaxFlat,pFlat,x):
            ret = 0.
            if x<rmaxQuad:
                ret+= (1-pFlat)/rmaxQuad
            if x<rmaxFlat:
                ret +=pFlat/rmaxFlat
            return  ret
    quadratic_samp_withfloor_vector = numpy.vectorize(quadratic_samp_withfloor, otypes=[numpy.float])
#    uniform_samp_withfloor_vector = numpy.vectorize(uniform_samp_withfloor, otypes=[numpy.float])
    uniform_samp_withfloor_vector = mcsampler.uniform_samp_withfloor_vector
    # Use an option for maximum distance
    dist_max_to_use = opts.d_max
    sampler.add_parameter("distance",
#                          functools.partial(mcsampler.quadratic_samp_vector, distBoundGuess), None, dist_min, dist_max,
#                          functools.partial(mcsampler.uniform_samp_vector,0, distBoundGuess), None, dist_min, dist_max,
                          functools.partial( uniform_samp_withfloor_vector, numpy.min([distBoundGuess,dist_max_to_use]), dist_max_to_use, 0.001), None, dist_min, dist_max_to_use,
                         prior_pdf = functools.partial(mcsampler.quadratic_samp_vector, dist_max_to_use
                                                        ),
                          adaptive_sampling = adapt_dist
                         )
#        sampler.add_parameter("dist", functools.partial(mcsampler.quadratic_samp_vector,  distBoundGuess ), None, dist_min, dist_max, prior_pdf = numpy.vectorize(lambda x: x**2/(3.*numpy.power(dist_max,3))))
    if opts.fixparams.count('dist') and Psig:
            print("  ++++ Fixing distance to injected value +++ ")
            pinned_params['distance'] = Psig.dist/(1e6*lal.LAL_PC_SI)
            #sampler.add_pinned_parameter("dist", Psig.dist/(1e6*lal.LAL_PC_SI) )
        
    return pinned_params
