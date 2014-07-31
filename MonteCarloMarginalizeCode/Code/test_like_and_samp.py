#! /usr/bin/env python
"""
test_like_and_samp.py:  Testing the likelihood evaluation and sampler, working in conjunction
   - Creates *zero-noise* signal by default, at fixed parameters OR from injection xml, 
     OR loads data from frame files
   - Creates candidate template, with exactly the same parameters
   - Monte Carlo over the likeihood

Options will (eventually) include
   - use of numerical PSD
   - use of template XML  [potentially different parameters]


Examples:
  # Run with default parameters for injection, with different approximants 
        python test_like_and_samp.py  --show-sampler-inputs --show-sampler-results --show-likelihood-versus-time
        python test_like_and_samp.py --approx TaylorT4 --amporder -1 --Lmax 3 --srate 16384
        python test_like_and_samp.py --approx EOBNRv2HM --Lmax 2 --srate 16384 --skip-interpolation 
  # Use a numerical psd or analytic PSD.  Plot what you are using
        python test_like_and_samp.py --show-psd
        python test_like_and_samp.py --psd-file psd.xml.gz --show-psd
  # Run using several likelihood approximations
        python test_like_and_samp.py  --show-sampler-inputs --show-sampler-results --LikelihoodType_MargTdisc   # NOT DEBUGGED
  
  # Run with a fixed sky location (at injection values)
        python test_like_and_samp.py --fix-polarization --fix-distance --fix-rightascension --fix-time --fix-inclination
   # Run using a skymap. [Should be consistent with injection values]. Note the skymap is modified to conserve probability; see notes
        python test_like_and_samp.py --sampling-prior-use-skymap skymap.fits.gz

  # Run with a known injection  (template has same masses by default)
        python test_like_and_samp.py --inj-xml mdc.xml.gz   
  # Run with real data, using an injection or a coinc
        python test_like_and_samp.py --inj-xml mdc.xml.gz   --cache-file test1.cache --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE_h_16384Hz_4R
        python test_like_and_samp.py  --cache-file test1.cache --channel-name H1=FAKE-STRAIN --channel-name L1=FAKE-STRAIN --channel-name V1=FAKE_h_16384Hz_4R  

  # Run with a synthetic injected signal with a nontrivial polarization, inclination (edge on), time, and phase
        python test_like_and_samp.py --signal-inclination 1.5708 --signal-time 0.02 --signal-polarization 0.3 --signal-phase -0.7  # test inclination, polarization propagated consistently

  # Run with convergence tests and adaptation
"""
try:
    import matplotlib
    #matplotlib.use("Agg")
    if matplotlib.get_backend is not 'TkAgg':  # on cluster
        matplotlib.use("GDK")
    from matplotlib import pylab as plt
    bNoInteractivePlots = True  # Move towards saved fig plots, for speed
except:
    print "- no matplotlib -"
    bNoInteractivePlots = True




import sys
import pickle

import numpy as np

from glue.lal import Cache
from glue.ligolw import utils, lsctables, table, ligolw,  git_version
from glue.ligolw.utils import process


import lal
import lalsimulation as lalsim
import lalsimutils


try:
    import healpy
    from lalinference.bayestar import fits as bfits
    from lalinference.bayestar import plot as bplot
except:
    print " -no skymaps - "



__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>, R. O'Shaughnessy"

import factored_likelihood
import factored_likelihood_test

import xmlutils
import common_cl
import ourparams
import ourio
opts, rosDebugMessagesDictionary = ourparams.ParseStandardArguments()
print opts
print rosDebugMessagesDictionary

if opts.verbose:
    try:
        from subprocess import call
        print " ---- LAL Version ----"
        call(["lal-version"])
        print " ---- GLUE Version ----"
        call(["ligolw_print",  "--version"])
        print " ---- pylal Version ----"
        call(["pylal_version"])
    except:
        print "  ... trouble printing version numbers "
    print " Glue tag ", git_version.id
    try:
        import scipy   # used in lalsimutils; also printed in pylal-version
        print " Scipy: ", scipy.__version__
    except:
        print " trouble printing scipy version"

def mean_and_dev(arr, wt):
    av = np.average(arr, weights=wt)
    var = np.average(arr*arr, weights=wt)
    return [av, var - av*av]

def pcum_at(val,arr, wt):
    nm = np.sum(wt)
    return np.sum(wt[np.where(val < arr)]/nm)

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>, R. O'Shaughnessy"


checkInputs = opts.plot_ShowLikelihoodVersusTime

rosUseDifferentWaveformLengths = False    
rosUseRandomTemplateStartingFrequency = False

rosUseTargetedDistance = True
rosUseStrongPriorOnParameters = False
rosShowSamplerInputDistributions = False #opts.plot_ShowSamplerInputs
rosShowRunningConvergencePlots = True
rosShowTerminalSampleHistograms = True
rosUseMultiprocessing= False
rosDebugCheckPriorIntegral = False
nMaxEvals = int(opts.nmax)
print " Running at most ", nMaxEvals, " iterations"

fracThreshold = opts.points_threshold_match


theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 
tEventFiducial =   0                                                                # relative to GPS reference

det_dict = {}
rhoExpected ={}

tWindowReference =   factored_likelihood_test.tWindowReference
tWindowExplore =     factored_likelihood.tWindowExplore

approxSignal = lalsim.GetApproximantFromString(opts.approx)
approxTemplate = approxSignal
ampO =opts.amporder # sets which modes to include in the template (and signal, if injected)
Lmax = opts.Lmax # sets which modes to include in the template.  Print warning if inconsistent.
fref = opts.fref
fminWavesTemplate = opts.fmin_Template  # too long can be a memory and time hog, particularly at 16 kHz
fminWavesSignal = opts.signal_fmin  # too long can be a memory and time hog, particularly at 16 kHz
fminSNR =opts.fmin_SNR
fmaxSNR =opts.fmax_SNR
fSample = opts.srate
window_beta = 0.01

if ampO ==-1 and Lmax < 5:
    print " +++ WARNING ++++ "
    print "  : Lmax is ", Lmax, " which may be insufficient to resolve all higher harmonics in the signal! "

if (ampO+2)> Lmax:
    print " +++ WARNING ++++ "
    print "  : Lmax is ", Lmax, " which may be insufficient to resolve all higher harmonics in the signal! "

if opts.channel_name is not None and opts.cache_file is None:
    print >>sys.stderr, "Cache file required when requesting channel data."	
    exit(-1)
elif opts.channel_name is not None:
    det_dict = dict(map(lambda cname: cname.split("="), opts.channel_name))


#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
Psig = None

# Read in *coincidence* XML (overridden by injection, if present)
if opts.coinc:
    print "Loading coinc XML:", opts.coinc
    xmldoc = utils.load_filename(opts.coinc)
    coinc_table = table.get_table(xmldoc, lsctables.CoincInspiralTable.tableName)
    assert len(coinc_table) == 1
    coinc_row = coinc_table[0]
    event_time = float(coinc_row.get_end())  # 
    event_time_gps = lal.GPSTimeNow()    # Pack as GPSTime *explicitly*, so all time operations are type-consistent
    event_time_gps.gpsSeconds = int(event_time)
    event_time_gps.gpsNanoSeconds = int(1e9*(event_time -event_time_gps.gpsSeconds))
    theEpochFiducial = event_time_gps       # really should avoid duplicate names
    print "Coinc XML loaded, event time: %s" % str(coinc_row.get_end())
    # Populate the SNR sequence and mass sequence
    sngl_inspiral_table = table.get_table(xmldoc, lsctables.SnglInspiralTable.tableName)
    m1, m2 = None, None
    for sngl_row in sngl_inspiral_table:
        # NOTE: gstlal is exact match, but other pipelines may not be
        assert m1 is None or (sngl_row.mass1 == m1 and sngl_row.mass2 == m2)
        m1, m2 = sngl_row.mass1, sngl_row.mass2
        rhoExpected[str(sngl_row.ifo)] = sngl_row.snr  # record for comparisons later
        if rosDebugMessagesDictionary["DebugMessages"]:
            det = str(sngl_row.ifo)
            print det, rhoExpected[det]
    m1 = m1*lal.LAL_MSUN_SI
    m2 = m2*lal.LAL_MSUN_SI
    rho2Net = 0
    for det in rhoExpected:
        rho2Net += rhoExpected[det]**2
    if rosDebugMessagesDictionary["DebugMessages"]:
        print " Network :", np.sqrt(rho2Net)
    # Create a 'best recovered signal'
    Psig = lalsimutils.ChooseWaveformParams(
        m1=m1,m2=m2,approx=approxSignal,
        fmin = fminWavesSignal, 
        dist=factored_likelihood.distMpcRef*1e6*lal.LAL_PC_SI,    # default distance
        fref=fref, 
        tref = event_time_gps,
        ampO=ampO      
        )  # FIXME: Parameter mapping from trigger space to search space
    if rosDebugMessagesDictionary["DebugMessages"]:
        print " === Coinc table : estimated signal [overridden if injection] ==="
        Psig.print_params()

# Read in *injection* XML
if opts.inj:
    print "Loading injection XML:", opts.inj
    Psig = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))[opts.event_id]  # Load in the physical parameters of the injection.  
    m1 = Psig.m1
    m2 = Psig.m2
    timeWaveform = float(-lalsimutils.hoft(Psig).epoch)
    Psig.deltaF = 1./lalsimutils.nextPow2(opts.seglen)       # Frequency binning needs to account for target segment length
    theEpochFiducial = Psig.tref  # Reset
    tEventFiducial = 0               # Reset
    print " ++ Targeting event at time ++ ", lalsimutils.stringGPSNice(Psig.tref)
    print " +++ WARNING: ADOPTING STRONG PRIORS +++ "
    rosUseStrongPriorOnParameters= True
    Psig.print_params()
    print "---- End injeciton parameters ----"

# Use forced parameters, if provided
if opts.template_mass1:
    Psig.m1 = opts.template_mass1*lal.LAL_MSUN_SI
if opts.template_mass2:
    Psig.m2 = opts.template_mass2*lal.LAL_MSUN_SI

# Reset origin of time, if required. (This forces different parts of data to be read- important! )
if opts.force_gps_time:
    print " +++ USER HAS OVERRIDDEN FIDUCIAL EPOCH +++ "
    print "  The zero of time (and the region to be windowed) will be changed; you had better know what you are doing.  "
    print "    original " ,lalsimutils.stringGPSNice(theEpochFiducial)
    print "    new      ", opts.force_gps_time
    theEpochFiducial = lal.GPSTimeNow()
    theEpochFiducial.gpsSeconds = int(opts.force_gps_time)
    theEpochFiducial.gpsNanoSeconds =  int(1e9*(opts.force_gps_time - int(opts.force_gps_time)))

# Create artificial "signal".  Needed to minimize duplicate code when I
#  -  consistently test waveform duration
#  - copy template parameters 
#  - [test code] : generate plots vs time [test code], expected SNR, etc
#       WARNING: Test code plots will not look perfect, because we don't know the true sky location (or phase, polarization, ...)
if  not Psig and opts.channel_name:  # If data loaded but no signal generated
    if (not opts.template_mass1) or (not opts.template_mass2) or (not opts.force_gps_time):
        print " CANCEL: Specifying parameters via m1, m2, and time on the command line "
        sys.exit(0)
    Psig = lalsimutils.ChooseWaveformParams(approx=approxSignal,
        fmin = fminWavesSignal, 
        dist=factored_likelihood.distMpcRef*1e6*lal.LAL_PC_SI,    # default distance
        fref=fref)
    Psig.m1 = lal.LAL_MSUN_SI*opts.template_mass1
    Psig.m2 = lal.LAL_MSUN_SI*opts.template_mass2
    Psig.tref = lal.LIGOTimeGPS(0.000000000)  # Initialize as GPSTime object
    Psig.tref += opts.force_gps_time # Pass value of float into it


# TEST THE SEGMENT LENGTH TARGET
if Psig:
    timeSegmentLength  = -float(lalsimutils.hoft(Psig).epoch)
    if rosDebugMessagesDictionary["DebugMessages"]:
        print " Template duration : ", timeSegmentLength
    if timeSegmentLength > opts.seglen:
        print " +++ CATASTROPHE : You are requesting less data than your template target needs!  +++"
        print "    Requested data size: ", opts.seglen
        print "    Template duration  : ", timeSegmentLength
        sys.exit(0)


# TRY TO READ IN DATA: if data specified, use it and construct the detector list from it. Otherwise...
if opts.channel_name and    (opts.opt_ReadWholeFrameFilesInCache):
    for inst, chan in map(lambda c: c.split("="), opts.channel_name):
        print "Reading channel %s from cache %s" % (inst+":"+chan, opts.cache_file)
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan,window_shape=window_beta)
        fSample = len(data_dict[inst].data.data)*data_dict[inst].deltaF
        df = data_dict[inst].deltaF
        if Psig:
            Psig.deltaF = df
        print "Frequency binning: %f, length %d" % (data_dict[inst].deltaF, len(data_dict[inst].data.data))
        print "Sampling rate ", fSample
if opts.channel_name and not (opts.opt_ReadWholeFrameFilesInCache):
    if Psig:
        event_time = Psig.tref
    else:
        event_time = theEpochFiducial  # For now...get from XML if that is the option
    start_pad, end_pad = opts.seglen-opts.padding, opts.padding 
    for inst, chan in map(lambda c: c.split("="), opts.channel_name):
        print "Reading channel %s from cache %s" % (inst+":"+chan, opts.cache_file)
        # FIXME: Assumes a frame file exists covering EXACTLY the needed interval!
        taper = lalsim.LAL_SIM_INSPIRAL_TAPER_STARTEND
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan, start=int(event_time)-start_pad, stop=int(event_time)+end_pad,window_shape=window_beta)
        fSample = len(data_dict[inst].data.data)*data_dict[inst].deltaF
        df = data_dict[inst].deltaF
        if Psig:
            Psig.deltaF =df
        print "Frequency binning: %f, length %d" % (data_dict[inst].deltaF, len(data_dict[inst].data.data))
        print "Sampling rate ", fSample

#        print " Sampling rate of data ", fSample

# CREATE A DEFAULT "signal", if none made to this point.  
analytic_signal = False
if len(data_dict) is 0:
    analytic_signal = True

    if not(Psig):
        m1 = 4*lal.LAL_MSUN_SI
        m2 = 3*lal.LAL_MSUN_SI

        Psig = lalsimutils.ChooseWaveformParams(
            m1 = m1,m2 =m2,
            fmin = fminWavesSignal, 
            fref=fref, 
            tref = theEpochFiducial + tEventFiducial,
            approx=approxSignal,
            ampO=ampO,
            radec=True, theta=1.2, phi=2.4,
            detector='H1', 
            dist=opts.signal_distMpc*1.e6*lal.LAL_PC_SI,    # move back to aLIGO distances
            deltaT=1./fSample
                                )
        timeSegmentLength  = -float(lalsimutils.hoft(Psig).epoch)
        if timeSegmentLength > opts.seglen:
            print " +++ CATASTROPHE : You are requesting less data than your template target needs!  +++"
            print "    Requested data size: ", opts.seglen
            print "    Template duration  : ", timeSegmentLength
            sys.exit(0)

    if opts.signal_incl:
        Psig.incl = opts.signal_incl
    if opts.signal_psi:
        Psig.psi = opts.signal_psi
    if opts.signal_phiref:
        Psig.phi = opts.signal_phiref
    if opts.signal_tref:
        if np.abs(opts.signal_tref) < 1000:
            Psig.tref = theEpochFiducial+opts.signal_tref  
        else:
            Psig.tref = opts.signal_tref
            theEpochFiducial=opts.signal_tref

    df = lalsimutils.findDeltaF(Psig)
    if 1/df < opts.seglen:   # Allows user to change seglen of data for *analytic* models, on the command line. Particularly useful re testing PSD truncation
        df = 1./lalsimutils.nextPow2(opts.seglen)
    Psig.print_params()
    Psig.deltaF = df
    data_dict['H1'] = lalsimutils.non_herm_hoff(Psig)
    Psig.detector = 'L1'
    data_dict['L1'] = lalsimutils.non_herm_hoff(Psig)
    Psig.detector = 'V1'
    data_dict['V1'] = lalsimutils.non_herm_hoff(Psig)

# Reset origin of time, if required
if opts.force_gps_time:
    print " +++ USER HAS OVERRIDDEN FIDUCIAL EPOCH +++ "
    print "  The zero of time (and the region to be windowed) will be changed; you had better know what you are doing.  "
    print "    original " ,theEpochFiducial
    print "    new      ", opts.force_gps_time
    theEpochFiducial = lal.GPSTimeNow()
    theEpochFiducial.gpsSeconds = int(opts.force_gps_time)
    theEpochFiducial.gpsNanoSeconds =  int(1e9*(opts.force_gps_time - int(opts.force_gps_time)))


# PSD reading
psd_dict = {}
if not(opts.psd_file) and not(opts.psd_file_singleifo):
    analyticPSD_Q = True # For simplicity, using an analytic PSD
    for det in data_dict.keys():
#        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower   #lal.LIGOIPsd    # Preserves key equality in data_dict , psd_dict
#        psd_dict[det] = lal.LIGOIPsd    # Preserves key equality in data_dict , psd_dict
        psd_dict[det] = lalsim.SimNoisePSDiLIGOSRD    # Preserves key equality in data_dict , psd_dict.  this is Chris' 'iLIGO' PSD, for test data

else:
    analyticPSD_Q = False # For simplicity, using an analytic PSD
    detectors = data_dict.keys()
    detectors_singlefile_dict = common_cl.parse_cl_key_value(opts.psd_file_singleifo)
    df = data_dict[detectors[0]].deltaF
    fNyq = (len(data_dict[detectors[0]].data.data)/2)*df
    print " == Loading numerical PSDs =="
    for det in detectors:
        psd_fname = ""
        if detectors_singlefile_dict.has_key(det):
            psd_fname = detectors_singlefile_dict[det]
        else:
            psd_fname = opts.psd_fname
        print "Reading PSD for instrument %s from %s" % (det, psd_fname)

        # "Standard" PSD parsing code used on master.
        psd_dict[det] = lalsimutils.get_psd_series_from_xmldoc(psd_fname, det)  # pylal type!
        tmp = psd_dict[det].data
        print "Sanity check reporting : pre-extension, min is ", np.min(tmp), " and maximum is ", np.max(tmp)
        deltaF = data_dict[det].deltaF
        fmin = psd_dict[det].f0
        fmax = fmin + psd_dict[det].deltaF*len(psd_dict[det].data)-deltaF
        print "PSD deltaF before interpolation %f" % psd_dict[det].deltaF
        psd_dict[det] = lalsimutils.resample_psd_series(psd_dict[det], deltaF)
        print "PSD deltaF after interpolation %f" % psd_dict[det].deltaF
        print "Post-extension the new PSD has 1/df = ", 1./psd_dict[det].deltaF, " (data 1/df = ", 1./deltaF, ") and length ", len(psd_dict[det].data.data)
        tmp = psd_dict[det].data.data
        nBad = np.argmin(tmp[np.nonzero(tmp)])
        fBad = nBad*deltaF
        print "Post-extension sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), "(at n=", np.argmin(tmp[np.nonzero(tmp)])," or f=", fBad, ")  and maximum is ", np.max(psd_dict[det].data.data)
        print

        # psd_dict[det] = lalsimutils.pylal_psd_to_swig_psd(lalsimutils.get_psd_series_from_xmldoc(opts.psd_file, det))
        # psd_dict[det] = lalsimutils.regularize_swig_psd_series_near_nyquist(psd_dict[det], fNyq-opts.fmax_SNR) # zero out 80 hz window near nyquist
        # psd_dict[det] =  lalsimutils.enforce_swig_psd_fmin(psd_dict[det], fminSNR)           # enforce fmin at the psd level, HARD CUTOFF
        # tmp = psd_dict[det].data.data
        # print "Sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), " and maximum is ", np.max(psd_dict[det].data.data)
        # deltaF = data_dict[det].deltaF
        # # remember the PSD is one-sided, but h(f) is two-sided. The lengths are not equal.
        # psd_dict[det] = lalsimutils.extend_swig_psd_series_to_sampling_requirements(psd_dict[det], df, df*(len(data_dict[det].data.data)/2))
        # print "Post-extension the new PSD has 1/df = ", 1./psd_dict[det].deltaF, " (data 1/df = ", 1./deltaF, ") and length ", len(psd_dict[det].data.data)
        # tmp = psd_dict[det].data.data
        # print "Post-extension sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), " and maximum is ", np.max(psd_dict[det].data.data)

# This code is a DUPLICATE TEST, used to initialize the peak log likelihood.
# I use this threshold to identify points for further investigation.
print " == Data report == "
detectors = data_dict.keys()
rho2Net = 0
print  " Amplitude report :"
fminSNR =opts.fmin_SNR
for det in detectors:
    if analyticPSD_Q:
        IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det],fMax=fmaxSNR)
    else:
        IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det].data.data,fMax=fmaxSNR,analyticPSD_Q=False)
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print det, " rho = ", rhoDet
print "Network : ", np.sqrt(rho2Net)

if checkInputs and not bNoInteractivePlots:
    print " == Plotting detector data (time domain; requires regeneration, MANUAL TIMESHIFTS,  and seperate code path! Argh!) == "
    P = Psig.copy()
    P.tref = Psig.tref
    for det in detectors:
        P.detector=det   # we do 
        hT = lalsimutils.hoft(P)
        tvals = float(P.tref - theEpochFiducial) + hT.deltaT*np.arange(len(hT.data.data))
        plt.figure(1)
        plt.plot(tvals, hT.data.data,label=det)

    tlen = hT.deltaT*len(np.nonzero(np.abs(hT.data.data)))
    tRef = np.abs(float(hT.epoch))
    plt.xlim( tRef-0.5,tRef+0.1)  # Not well centered, based on epoch to identify physical merger time
    plt.savefig("test_like_and_samp-input-hoft.pdf")


# Load skymap, if present
if opts.opt_UseSkymap:
    print " ==Loading skymap=="
    print "   skymap file ", opts.opt_UseSkymap
    smap, smap_meta = bfits.read_sky_map(opts.opt_UseSkymap)
    sides = healpy.npix2nside(len(smap))

    if opts.plot_ShowSamplerInputs and not bNoInteractivePlots:
        try:
            from lalinference.bayestar import plot
            plt.subplot(111, projection='astro mollweide')
            plot.healpix_heatmap(smap)
            plt.show()
        except:
            print " No skymap for you "



# Struct to hold template parameters
P = lalsimutils.ChooseWaveformParams(fmin=fminWavesTemplate, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         approx=approxTemplate,
         fref=fref,
         tref=theEpochFiducial,
         deltaT=1./fSample,
         dist=100*1.e6*lal.LAL_PC_SI,
         deltaF=df)


#
# Perform the Precompute stage
#   WARNING: Using default values for inverse spectrum truncation (True) and inverse spectrun truncation time (8s) from ourparams.py
#                     ILE adopts a different convention.  ROS old development branch has yet another approach (=set during PSD reading).
#
rholms_intp, crossTerms, rholms = factored_likelihood.PrecomputeLikelihoodTerms(theEpochFiducial,tWindowReference[1], P, data_dict,psd_dict, Lmax, fmaxSNR, analyticPSD_Q,inv_spec_trunc_Q=opts.psd_TruncateInverse,T_spec=opts.psd_TruncateInverseTime)
epoch_post = theEpochFiducial # Suggested change.  BE CAREFUL: Now that we trim the series, this is NOT what I used to be
print "Finished Precomputation..."
print "====Generating metadata from precomputed results ====="
distBoundGuess = factored_likelihood.estimateUpperDistanceBoundInMpc(rholms, crossTerms)
print " distance probably less than ", distBoundGuess, " Mpc"

try:
    print "====Loading metadata from previous runs (if any): <base>-seed-data.dat ====="
    if not opts.force_use_metadata:
        print " ... this data will NOT be used to change the samplers... "
    metadata ={}
    fnameBase = opts.points_file_base
    if opts.fname_metadata:
        fname = opts.fname_metadata
    else:
        fname =  fnameBase+"-seed-data.pkl"
    print " ... trying to open ", fname
    with open(fname,'r') as f:
        metadata = pickle.load(f)
        print " Loaded metadata file :", metadata
except:
    print " === Skipping metadata step ==="

TestDictionary = factored_likelihood_test.TestDictionaryDefault
# TestDictionary["DataReport"]             = True
# TestDictionary["DataReportTime"]             = False
TestDictionary["UVReport"]              =  analytic_signal  # this report is very confusing for real data
# TestDictionary["UVReflection"]          = True
# TestDictionary["QReflection"]          = False
TestDictionary["Rho22Timeseries"]      = True
TestDictionary["lnLModelAtKnown"]  =  analytic_signal  # this report is very confusing for real data
TestDictionary["lnLDataAtKnownPlusOptimalTimePhase"] = False
TestDictionary["lnLAtKnown"]           = True
TestDictionary["lnLAtKnownMarginalizeTime"]  = False  # cluster python has problems with integrate.quad
TestDictionary["lnLDataPlot"]            = opts.plot_ShowLikelihoodVersusTime
TestDictionary["lnLDataPlotVersusPsi"]            = False # opts.plot_ShowLikelihoodVersusTime
TestDictionary["lnLDataPlotVersusPhi"]            = False # opts.plot_ShowLikelihoodVersusTime
TestDictionary["lnLDataPlotVersusPhiPsi"]            = False # opts.plot_ShowLikelihoodVersusTime

#opts.fmin_SNR=40

factored_likelihood_test.TestLogLikelihoodInfrastructure(TestDictionary,theEpochFiducial,  data_dict, psd_dict, fmaxSNR,analyticPSD_Q, Psig, rholms,rholms_intp, crossTerms, detectors,Lmax)

if opts.rotate_sky_coordinates:  # FIXME: should also test that both theta, phi are coordinates *and* adaptation is on
    det0 = data_dict.keys()[0]
    det1 = data_dict.keys()[1]
    print " ======= ROTATING COORDINATES ====== " 
    print "Rotation based on current position of detectors detectors", det0, det1
    import lalsimulation as lalsim
    # Detector seperation relative to the earth, *not* sky fixed coordinates
    theDetectorSeparation = lalsim.DetectorPrefixToLALDetector(det0).location - lalsim.DetectorPrefixToLALDetector(det1).location
    vecZ = np.array(theDetectorSeparation/np.sqrt( np.dot(theDetectorSeparation,theDetectorSeparation)))
    vecZth, vecZph = lalsimutils.polar_angles_in_frame(lalsimutils.unit_frame(), vecZ)
    # Rotate to sky-fixed coordinates (azimuth)  [Could also do with polar angles, just adding gmst]
    time_angle =  np.mod( lal.GreenwichMeanSiderealTime(theEpochFiducial), 2*np.pi)
    vecZnew = np.dot(lalsimutils.rotation_matrix(np.array([0,0,1]), time_angle), vecZ)
    print "Rotation 'z' vector in sidereal coordinates ", vecZnew
#    print lalsimutils.polar_angles_in_frame(lalsimutils.unit_frame(), vecZ), lalsimutils.polar_angles_in_frame(lalsimutils.unit_frame(), vecZnew)

    # Create a frame associated with the rotated angle
    frm = lalsimutils.VectorToFrame(vecZnew)   # Create an orthonormal frame related to this particular choice of z axis. (Used as 'rotation' object)
    frmInverse= np.asarray(np.matrix(frm).I)                                    # Create an orthonormal frame to undo the transform above
    def rotate_sky_forwards(theta,phi):   # When theta=0 we are describing the coordinats of the zhat direction in the vecZ frame
        global frm
        return lalsimutils.polar_angles_in_frame_alt(frm, theta,phi)

    def rotate_sky_backwards(theta,phi): # When theta=0, the vector should be along the vecZ direction and the polar angles returned its polar angles
        global frmInverse
        return lalsimutils.polar_angles_in_frame_alt(frmInverse, theta,phi)




#
# Call the likelihood function for various extrinsic parameter values
# Uses the (already-allocated) template structure "P" structure *only* to pass parameters.  All parameters used should be specified.
#
nEvals = 0
if not opts.LikelihoodType_MargTdisc_array:
    def likelihood_function(right_ascension, declination, t_ref, phi_orb, inclination, psi, distance): # right_ascension, declination, t_ref, phi_orb, inclination, psi, distance):
        global nEvals
        global pdfFullPrior

#        if opts.rotate_sky_coordinates:
#            print "   -Sky ring width ", np.std(declination), " note contribution from floor is of order p_floor*(pi)/sqrt(12) ~ 0.9 pfloor"
#            print "   -Distance width", np.std(distance)

        lnL = np.zeros(right_ascension.shape)
        i = 0
        for ph, th, tr, phr, ic, ps, di in zip(right_ascension, declination, t_ref, phi_orb, inclination, psi, distance):
            if opts.rotate_sky_coordinates: 
                th,ph = rotate_sky_backwards(np.pi/2 - th,ph)
                th = np.pi/2 - th
                ph = np.mod(ph, 2*np.pi)
            P.phi = ph # right ascension
            P.theta = th # declination
            P.tref = theEpochFiducial + tr # ref. time (rel to epoch for data taking)
            P.phiref = phr # ref. orbital phase
            P.incl = ic # inclination
            P.psi = ps # polarization angle
            P.dist = di*1e6*lal.LAL_PC_SI # luminosity distance.  The sampler assumes Mpc; P requires SI
            lnL[i] = factored_likelihood.FactoredLogLikelihood(P, rholms_intp, crossTerms, Lmax)#+ np.log(pdfFullPrior(ph, th, tr, ps, ic, ps, di))
            i+=1


        nEvals+=i 
        return np.exp(lnL)
else: # Sum over time for every point in other extrinsic params
    def likelihood_function(right_ascension, declination,t_ref, phi_orb, inclination,
            psi, distance):
        # use EXTREMELY many bits
        lnL = np.zeros(right_ascension.shape,dtype=np.float128)
        i = 0
#        if opts.rotate_sky_coordinates:
#            print "   -Sky ring width ", np.std(declination), " note contribution from floor is of order p_floor*(pi)/sqrt(12) ~ 0.9 pfloor"
#            print "   -RA width", np.std(right_ascension)
#            print "   -Distance width", np.std(distance)

        tvals = np.linspace(tWindowExplore[0],tWindowExplore[1],int((tWindowExplore[1]-tWindowExplore[0])/P.deltaT))  # choose an array at the target sampling rate. P is inherited globally
        for ph, th, phr, ic, ps, di in zip(right_ascension, declination,
                phi_orb, inclination, psi, distance):
            if opts.rotate_sky_coordinates: 
                th,ph = rotate_sky_backwards(np.pi/2 - th,ph)
                th = np.pi/2 - th
                ph = np.mod(ph, 2*np.pi)

            P.phi = ph # right ascension
            P.theta = th # declination
            P.tref = theEpochFiducial  # see 'tvals', above
            P.phiref = phr # ref. orbital phase
            P.incl = ic # inclination
            P.psi = ps # polarization angle
            P.dist = di* 1.e6 * lal.LAL_PC_SI # luminosity distance

            lnL[i] = factored_likelihood.FactoredLogLikelihoodTimeMarginalized(tvals,
                    P, rholms_intp,rholms, crossTerms,                   
                    Lmax)
            i+=1
    
        return np.exp(lnL)

import mcsampler
sampler = mcsampler.MCSampler()

# Populate sampler 
## WARNING: CP has changed pinning interface again.  Need to rework
pinned_params = ourparams.PopulateSamplerParameters(sampler, theEpochFiducial,tEventFiducial, distBoundGuess, Psig, opts)
unpinned_params = set(sampler.params) - set(pinned_params)

if opts.plot_ShowPSD and not bNoInteractivePlots:
    for det in psd_dict.keys():
        if analyticPSD_Q:
            fvals =  np.arange(opts.fmin_SNR,2000,1.)              # the upper limit is kind of random
            Sh   =  map(psd_dict[det],fvals)
        else:
            fvals = np.arange(len(psd_dict[det].data.data))*deltaF  # stored as one-sided PSD
            Sh  = psd_dict[det].data.data
            nSkip = int(len(fvals)/(4096))
            fvals = fvals[::nSkip]   # downsample! PSDs are often too huge to plot!
            Sh = Sh[::nSkip]        # downsample!
        plt.figure(0)
        plt.plot(np.log10(fvals),np.log10(Sh),label="Sh:"+det)
        plt.xlabel('f (Hz)')
        plt.ylabel('Sh $Hz^{-1}$')
        plt.title('PSDs used')
    # Comparison plot: iLIGO
    Sh = map(lal.LIGOIPsd,fvals)
    plt.plot(np.log10(fvals),np.log10(Sh),label="Sh:iLIGO")
    Sh = map(lalsim.SimNoisePSDaLIGOZeroDetHighPower,fvals)
    plt.plot(np.log10(fvals),np.log10(Sh),label="Sh:aLIGO")
    plt.legend()
    plt.xlim(0,4)
    plt.ylim(-50,-30)
    plt.savefig("FLT-psd.jpg")  # really not in FLT, but the same kind of plot

if rosShowSamplerInputDistributions:
    print " ====== Plotting prior and sampling distributions ==== "
    print "  PROBLEM: Build in/hardcoded via uniform limits on each parameter! Need to add measure factors "
    nFig = 0
    for param in sampler.params:
      if  not(isinstance(param,tuple)): # not(sampler.pinned[param]) and
        nFig+=1
        plt.figure(nFig)
        plt.clf()
        xLow = sampler.llim[param]
        xHigh = sampler.rlim[param]
        xvals = np.linspace(xLow,xHigh,500)
        pdfPrior = sampler.prior_pdf[param]  # Force type conversion in case we have non-float limits for some reasona
        pdfvalsPrior = np.array(map(pdfPrior, xvals))  # do all the np operations by hand: no vectorization
        pdf = sampler.pdf[param]
        cdf = sampler.cdf[param]
        pdfvals = pdf(xvals)
        cdfvals = cdf(xvals)
        plt.plot(xvals,pdfvalsPrior,label="prior:"+str(param),linestyle='--')
        plt.plot(xvals,pdfvals,label=str(param))
        plt.plot(xvals,cdfvals,label='cdf:'+str(param))
        plt.xlabel(str(param))
        plt.legend()
        plt.savefig("test_like_and_samp-"+str(param)+".jpg")


if  (rosShowSamplerInputDistributions or opts.plot_ShowPSD) and not bNoInteractivePlots:  # minimize number of pauses
    plt.show()


#
# Provide convergence tests
# FIXME: Currently using hardcoded thresholds, poorly hand-tuned
#
import functools
test_converged = {}
if opts.convergence_tests_on:
    test_converged['neff'] = functools.partial(mcsampler.convergence_test_MostSignificantPoint,0.01)  # most significant point less than 1% of probability
    test_converged["normal_integral"] = functools.partial(mcsampler.convergence_test_NormalSubIntegrals, 25, 0.01, 0.1)   # 20 sub-integrals are gaussian distributed *and* relative error < 10%, based on sub-integrals . Should use # of intervals << neff target from above.  Note this sets our target error tolerance on  lnLmarg


tGPSStart = lal.GPSTimeNow()
print " Unpinned : ", unpinned_params
print " Pinned : ",  pinned_params
pinned_params.update({"n": opts.nskip, "nmax": opts.nmax, "neff": opts.neff, "full_output": True, "verbose":True, "extremely_verbose": opts.super_verbose,"igrand_threshold_fraction": fracThreshold, "igrandmax":rho2Net/2, "save_intg":True,
    "convergence_tests" : test_converged,    # Dictionary of convergence tests

    "tempering_exp":opts.adapt_beta,
    "floor_level": opts.adapt_mix, # The new sampling distribution at the end of each chunk will be floor_level-weighted average of a uniform distribution and the (L^tempering_exp p/p_s)-weighted histogram of sampled points.
    "history_mult": 10, # Multiplier on 'n' - number of samples to estimate marginalized 1-D histograms
    "n_adapt": 100, # Number of chunks to allow adaption over
    "igrand_threshold_deltalnL": opts.save_deltalnL, # Threshold on distance from max L to save sample
    "igrand_threshold_p": opts.save_P # Threshold on cumulative probability contribution to cache sample

})
print " Params ", pinned_params
res, var,  neff , dict_return = sampler.integrate(likelihood_function, *unpinned_params, **pinned_params)

if opts.rotate_sky_coordinates:
        tmpTheta = np.zeros(len(sampler._rvs["declination"]))
        tmpPhi = np.zeros(len(sampler._rvs["declination"]))
        tmpThetaOut = np.zeros(len(sampler._rvs["declination"]))
        tmpPhiOut = np.zeros(len(sampler._rvs["declination"]))
        tmpTheta = sampler._rvs["declination"]
        tmpPhi = sampler._rvs["right_ascension"]
        for indx in np.arange(len(tmpTheta)):
            tmpThetaOut[indx],tmpPhiOut[indx] = rotate_sky_backwards(np.pi/2 - tmpTheta[indx],tmpPhi[indx])
        sampler._rvs["declination"] = np.pi/2 - tmpThetaOut
        sampler._rvs["right_ascension"] = np.mod(tmpPhiOut, 2*np.pi)


print sampler._rvs.keys()
retNew = [sampler._rvs["right_ascension"], sampler._rvs['declination'],sampler._rvs['t_ref'], sampler._rvs['phi_orb'],sampler._rvs['inclination'], sampler._rvs['psi'], sampler._rvs['psi'], sampler._rvs['distance'], sampler._rvs["joint_prior"], sampler._rvs["joint_s_prior"],np.log(sampler._rvs["integrand"])]
retNew = map(list, zip(*retNew))
ret = np.array(retNew)

tGPSEnd = lal.GPSTimeNow()
print "Parameters returned by this integral ",  sampler._rvs.keys(), len(sampler._rvs)
ntotal = opts.nmax  # Not true in general
print " Evaluation time  = ", float(tGPSEnd - tGPSStart), " seconds"
print " lnLmarg is ", np.log(res), " with nominal relative sampling error ", np.sqrt(var)/res, " but a more reasonable estimate based on the lnL history is " #, np.std(lnLmarg - np.log(res))
print " expected largest value is ", rho2Net/2, "and observed largest lnL is ", np.max(np.transpose(ret)[-1])
print " note neff is ", neff, "; compare neff^(-1/2) = ", 1/np.sqrt(neff)

print "==Returned dictionary==="
print dict_return


print "==Profiling info (assuming MAXIMUM evals hit)==="
print "   - Time per L evaluation ", float(tGPSEnd-tGPSStart)/ntotal
print "   - Time per neff             ", float(tGPSEnd-tGPSStart)/neff


# Save the sampled points to a file
# Only store some
fnameBase = opts.points_file_base
retSorted = ret[ np.argsort(ret[:,-1])]
ourio.dumpSamplesToFile(fnameBase+"-points.dat", retSorted, ['ra','dec', 'tref', 'phi', 'incl', 'psi', 'dist', 'p', 'ps', 'lnL']) 
#sampArray = Psig.list_params()  # Eventually, make this used. Note odd structure in list
#np.savetxt(fnameBase+"-params.dat", np.array(sampArray))
#print " Parameters : ", sampArray
ourio.dumpSamplesToFile(fnameBase+'-result.dat', np.array([[res, np.sqrt(var), np.max(ret[:,-1]),ntotal,neff, P.m1/lal.LAL_MSUN_SI,P.m2/lal.LAL_MSUN_SI]]), ['Z', 'sigmaZ', 'lnLmax','N', 'Neff','m1','m2'])  # integral, std dev,  total number of points
#np.savetxt(fnameBase+'-result.dat', [res, np.sqrt(var), ntotal])   # integral, std dev,  total number of points. Be SURE we do not lose precision!
#np.savetxt(fnameBase+'-dump-lnLmarg.dat',lnLmarg[::opts.nskip])  # only print output periodically -- otherwise far too large files!

if neff > 5 or opts.force_store_metadata:  # A low threshold but not completely implausible.  Often we are not clueless 
    print "==== Computing and saving metadata for future runs: <base>-seed-data.dat ====="
    print " Several effective points producted; generating metadata file "
    if neff < 20:
        print "  +++ WARNING +++ : Very few effective samples were found. Be VERY careful about using this as input to subsequent searches! "
    metadata={}
    weights = np.exp(ret[:,-1])*ret[:,-3]/ret[:,-2]
    metadata["ra"] =  mean_and_dev(ret[:,0], weights)
    metadata["dec"] = mean_and_dev(ret[:,1], weights)
    metadata["tref"] =  mean_and_dev(ret[:,2], weights)
    metadata["phi"] =  mean_and_dev(ret[:,3], weights)
    metadata["incl"] =  mean_and_dev(ret[:,4], weights)
    metadata["psi"] =  mean_and_dev(ret[:,5], weights)
    metadata["dist"] =  mean_and_dev(ret[:,6], weights)
    with open(fnameBase+"-seed-data.dat",'w') as f:
        for key in ['ra','dec', 'tref', 'phi', 'incl', 'psi', 'dist']:
            f.write(key + " " + str(metadata[key][0]) + ' '+ str(metadata[key][1]) + '\n')
    fnameBase = opts.points_file_base
    if opts.fname_metadata:
        fname = opts.fname_metadata
    else:
        fname =  fnameBase+"-seed-data.pkl"
    with open(fname,'w') as f:
        pickle.dump(metadata,f)

if opts.inj:
    print "==== PP data: <base>-pp-instance.dat ====="
    lnLAt = factored_likelihood.FactoredLogLikelihood(Psig, rholms_intp, crossTerms, Lmax)
    # Evaluate best data point
    ppdata = {}
    weights = np.exp(ret[:,-1])*ret[:,-3]/ret[:,-2]
    ppdata['ra'] = [Psig.phi,pcum_at(Psig.phi,ret[:,0],weights)]
    ppdata['dec'] = [Psig.theta,pcum_at(Psig.theta,ret[:,1], weights)]
    ppdata['tref'] = [Psig.tref-theEpochFiducial,pcum_at(Psig.tref-theEpochFiducial,ret[:,2], weights)]
    ppdata['phi'] = [Psig.phiref,pcum_at(Psig.phiref,ret[:,3], weights)]
    ppdata['incl'] = [Psig.incl,pcum_at(Psig.incl,ret[:,4], weights)]
    ppdata['psi'] = [Psig.psi,pcum_at(Psig.psi,ret[:,5], weights)]
    ppdata['dist'] = [Psig.dist/(1e6*lal.LAL_PC_SI),pcum_at(Psig.dist/(1e6*lal.LAL_PC_SI),ret[:,6], weights)]
    ppdata['lnL'] =  [lnLAt, pcum_at(lnLAt, ret[:,-1], weights)]

    # Dump data: p(<x)
    with open(fnameBase+"-pp-data.dat",'w') as f:
        for key in ['ra','dec', 'tref', 'phi', 'incl', 'psi', 'dist', 'lnL']: 
            f.write(key + " " + str(ppdata[key][0]) + ' '+ str(ppdata[key][1]) + '\n')


# Save the outputs in CP's format, for comparison.  NOT YET ACTIVE CODE -- xmlutils has a bug on master (lacking terms in dictionary)
if  True: # opts.points_file_base:
    print "==== Exporting to xml: <base>.xml.gz ====="
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    process.register_to_xmldoc(xmldoc, sys.argv[0], opts.__dict__)
    samples = {}
    samples["distance"]= ret[:,6]
    samples["t_ref"] = ret[:,2]
    samples["polarization"]= ret[:,5]
    samples["coa_phase"]= ret[:,3]
    samples["latitude"]= ret[:,1]
    samples["longitude"]= ret[:,0]
    samples["inclination"]= ret[:,4]
    samples["loglikelihood"]= ret[:,-1]  #alpha1
    samples["joint_prior"]= ret[:,-3]     #alpha2
    samples["joint_s_prior"]= ret[:,-2]  #alpha3
    # samples = sampler._rvs
    # samples["distance"] = samples["distance"]
    # samples["polarization"] = samples["psi"]
    # samples["coa_phase"] = samples["phi"]
    # samples["latitude"] = samples["dec"]
    # samples["inclination"] = samples["incl"]
    # samples["polarization"] = samples["psi"]
    # samples["longitude"] = samples["ra"]
    # samples["loglikelihood"] = np.log(samples["integrand"])
    m1 = P.m1/lal.LAL_MSUN_SI
    m2 = P.m2/lal.LAL_MSUN_SI
    samples["mass1"] = np.ones(samples["polarization"].shape)*m1
    samples["mass2"] = np.ones(samples["polarization"].shape)*m2
#    utils.write_fileobj(xmldoc,sys.stdout)
    xmlutils.append_samples_to_xmldoc(xmldoc, samples)
#    utils.write_fileobj(xmldoc,sys.stdout)
    xmlutils.append_likelihood_result_to_xmldoc(xmldoc, np.log(res), **{"mass1": m1, "mass2": m2})
    utils.write_filename(xmldoc, opts.points_file_base+".xml.gz", gz=True)


# Plot terminal histograms from the sampled points and log likelihoods
# FIXME: Needs to be rewritten to work with the new sampler names
if False: #rosShowTerminalSampleHistograms:
    print " ==== CONVERGENCE PLOTS (**beware: potentially truncated data!**) === "
    plt.figure(99)
    plt.clf()
    lnL = np.transpose(ret)[-1]
#    plt.plot(np.arange(len(lnLmarg)), lnLmarg,label="lnLmarg")
#    nExtend = np.max([len(lnL),len(lnLmarg)])
    plt.plot(np.arange(nExtend), np.ones(nExtend)*rho2Net/2,label="rho^2/2")
    plt.xlim(0,len(lnL))
    plt.xlabel('iteration')
    plt.ylabel('lnL')
    plt.legend()
    ourio.plotParameterDistributionsFromSamples("results", sampler, ret,  ['ra','dec', 'tref', 'phi', 'incl', 'psi', 'dist', 'lnL'])
    print " ==== TERMINAL 2D HISTOGRAMS: Sampling and posterior === "
        # Distance-inclination
    ra,dec,tref,phi,incl, psi,dist,lnL = np.transpose(ret)  # unpack. This can include all or some of the data set. The default configuration returns *all* points
    plt.figure(1)
    plt.clf()
    H, xedges, yedges = np.histogram2d(dist/(1e6*lal.LAL_PC_SI),incl, weights=np.exp(lnL),bins=(10,10))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    plt.imshow(H, extent=extent, interpolation='nearest', aspect=0.618)
    plt.colorbar()
    plt.xlim(0, 50)
    plt.ylim(0, np.pi)
    plt.title("Posterior distribution: d-incl ")
        # phi-psi
    plt.figure(2)
    plt.clf()
    H, xedges, yedges = np.histogram2d(phi,psi, bins=(10,10),weights=np.exp(lnL))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    plt.imshow(H, extent=extent, interpolation='nearest', aspect=0.618)
    plt.colorbar()
    plt.xlim(0,2*np.pi)
    plt.ylim(0,np.pi)
    plt.title("Posterior distribution: phi-psi ")
        # ra-dec
    plt.figure(3)
    plt.clf()
    H, xedges, yedges = np.histogram2d(ra,dec, bins=(10,10),weights=np.exp(lnL))
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    plt.imshow(H, extent=extent, interpolation='nearest', aspect=0.618)
    plt.xlim(0,2*np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.colorbar()
    plt.title("Posterior distribution: ra-dec ")
    plt.show()
