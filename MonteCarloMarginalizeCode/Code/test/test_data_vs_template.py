#! /usr/bin/env python
"""
test_like_and_samp.py:  

"""

import lalsimutils
import numpy as np
import lal
import lalsimulation as lalsim

import factored_likelihood
import factored_likelihood_test

import sys
import functools

import numpy
try:
    from matplotlib import pylab as plt
except:
    print("- no matplotlib -")

try:
    import healpy
    from lalinference.bayestar import fits as bfits
    from lalinference.bayestar import plot as bplot
except:
    print(" -no skymaps - ")


from glue.lal import Cache
from glue.ligolw import utils, lsctables, table

import ourparams
import ourio
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()
factored_likelihood.rosDebugMessagesDictionary = rosDebugMessagesDictionary
lalsimutils.rosDebugMessagesDictionary            = rosDebugMessagesDictionary
print(opts)
print(rosDebugMessagesDictionary)


checkInputs = opts.plot_ShowLikelihoodVersusTime

rosUseDifferentWaveformLengths = False    
rosUseRandomTemplateStartingFrequency = False

rosUseTargetedDistance = True
rosUseStrongPriorOnParameters = False
rosShowSamplerInputDistributions = opts.plot_ShowSamplerInputs
rosShowRunningConvergencePlots = True
rosShowTerminalSampleHistograms = True
rosUseMultiprocessing= False
rosDebugCheckPriorIntegral = False
nMaxEvals = int(opts.nmax)
print(" Running at most ", nMaxEvals, " iterations")

fracThreshold = opts.points_threshold_match

theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 
tEventFiducial = 0                                                                 # relative to GPS reference


det_dict = {}
rhoExpected ={}

tWindowReference = factored_likelihood.tWindowReference
tWindowExplore =     factored_likelihood.tWindowExplore

approxSignal = lalsim.GetApproximantFromString(opts.approx)
approxTemplate = approxSignal
ampO =opts.amporder # sets which modes to include in the physical signal
Lmax = opts.Lmax # sets which modes to include
fref = opts.fref
fminWavesSignal = opts.fmin_Template  # too long can be a memory and time hog, particularly at 16 kHz
fminSNR =opts.fmin_SNR
fSample = opts.srate

if rosUseDifferentWaveformLengths: 
    fminWavesTemplate = fminWavesSignal+0.005
else:
    if rosUseRandomTemplateStartingFrequency:
         print("   --- Generating a random template starting frequency  ---- ")
         fminWavesTemplate = fminWavesSignal+5.*np.random.random_sample()
    else:
        fminWavesTemplate = fminWavesSignal

#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
data_dict_time = {}
Psig = None

# Read in *coincidence* XML (overridden by injection, if present)
if opts.coinc:
    print("Loading coinc XML:", opts.coinc)
    xmldoc = utils.load_filename(opts.coinc)
    coinc_table = table.get_table(xmldoc, lsctables.CoincInspiralTable.tableName)
    assert len(coinc_table) == 1
    coinc_row = coinc_table[0]
    event_time = float(coinc_row.get_end())  # 
    event_time_gps = lal.GPSTimeNow()    # Pack as GPSTime *explicitly*, so all time operations are type-consistent
    event_time_gps.gpsSeconds = int(event_time)
    event_time_gps.gpsNanoSeconds = int(1e9*(event_time -event_time_gps.gpsSeconds))
    theEpochFiducial = event_time_gps       # really should avoid duplicate names
    print("Coinc XML loaded, event time: %s" % str(coinc_row.get_end()))
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
            print(det, rhoExpected[det])
    m1 = m1*lal.LAL_MSUN_SI
    m2 = m2*lal.LAL_MSUN_SI
    rho2Net = 0
    for det in rhoExpected:
        rho2Net += rhoExpected[det]**2
    if rosDebugMessagesDictionary["DebugMessages"]:
        print(" Network :", np.sqrt(rho2Net))
    # Create a 'best recovered signal'
    Psig = lalsimutils.ChooseWaveformParams(
        m1=m1,m2=m2,approx=approxSignal,
        fmin = fminWavesSignal, 
        dist=factored_likelihood.distMpcRef*1e6*lal.LAL_PC_SI,    # default distance
        fref=fref, 
        tref = event_time_gps,
        ampO=ampO,
                                )  # FIXME: Parameter mapping from trigger space to search space
    if rosDebugMessagesDictionary["DebugMessages"]:
        print(" === Coinc table : estimated signal [overridden if injection] ===")
        Psig.print_params()

# Read in *injection* XML
if opts.inj:
    print("Loading injection XML:", opts.inj)
    Psig = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))[opts.event_id]  # Load in the physical parameters of the injection.  
    timeWaveform = float(-lalsimutils.hoft(Psig).epoch)
    Psig.deltaF = 1./lalsimutils.nextPow2(opts.seglen)       # Frequency binning needs to account for target segment length
    theEpochFiducial = Psig.tref  # Reset
    tEventFiducial = 0               # Reset
    print(" ++ Targeting event at time ++ ", lalsimutils.stringGPSNice(Psig.tref))
    print(" +++ WARNING: ADOPTING STRONG PRIORS +++ ")
    rosUseStrongPriorOnParameters= True

# TEST THE SEGMENT LENGTH TARGET
if Psig:
    timeSegmentLength  = float(-lalsimutils.hoft(Psig).epoch)
    if rosDebugMessagesDictionary["DebugMessages"]:
        print(" Template duration : ", timeSegmentLength)
    if timeSegmentLength > opts.seglen:
        print(" +++ CATASTROPHE : You are requesting less data than your template target needs!  +++")
        print("    Requested data size: ", opts.seglen)
        print("    Template duration  : ", timeSegmentLength)
        sys.exit(0)

# TRY TO READ IN DATA: if data specified, use it and construct the detector list from it. Otherwise...
if opts.channel_name and    (opts.opt_ReadWholeFrameFilesInCache):
    for inst, chan in map(lambda c: c.split("="), opts.channel_name):
        print("Reading channel %s from cache %s" % (inst+":"+chan, opts.cache_file))
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan,window="Tukey",window_beta=0.1)
        data_dict_time[inst] = lalsimutils.frame_data_to_hoft(opts.cache_file, inst+":"+chan)
        fSample = len(data_dict[inst].data.data)*data_dict[inst].deltaF
        df = data_dict[inst].deltaF
        if Psig:
            Psig.deltaF = df
        print("Frequency binning: %f, length %d" % (data_dict[inst].deltaF, len(data_dict[inst].data.data)))
        print("Sampling rate ", fSample)
if opts.channel_name and not (opts.opt_ReadWholeFrameFilesInCache):
    if Psig:
        event_time = Psig.tref
    else:
        event_time = theEpochFiducial  # For now...get from XML if that is the option
    start_pad, end_pad = opts.seglen-opts.padding, opts.padding 
    for inst, chan in map(lambda c: c.split("="), opts.channel_name):
        print("Reading channel %s from cache %s" % (inst+":"+chan, opts.cache_file))
        # FIXME: Assumes a frame file exists covering EXACTLY the needed interval!
        taper = lalsim.LAL_SIM_INSPIRAL_TAPER_STARTEND
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan, start=int(event_time)-start_pad, stop=int(event_time)+end_pad,window="Tukey",window_beta=0.1)
        data_dict_time[inst] = lalsimutils.frame_data_to_hoft(opts.cache_file, inst+":"+chan, start=int(event_time)-start_pad, stop=int(event_time)+end_pad)
        fSample = len(data_dict[inst].data.data)*data_dict[inst].deltaF
        df = data_dict[inst].deltaF
        if Psig:
            Psig.deltaF =df
        print("Frequency binning: %f, length %d" % (data_dict[inst].deltaF, len(data_dict[inst].data.data)))
        print("Sampling rate ", fSample)

#        print " Sampling rate of data ", fSample

# CREATE A DEFAULT INJECTION, IF NONEMADE TO THIS POINT
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
            dist=250.*1.e6*lal.LAL_PC_SI,    # move back to aLIGO distances
            deltaT=1./fSample
                                )
        timeSegmentLength  = lalsimutils.estimateWaveformDuration(Psig)
        if timeSegmentLength > opts.seglen:
            print(" +++ CATASTROPHE : You are requesting less data than your template target needs!  +++")
            print("    Requested data size: ", opts.seglen)
            print("    Template duration  : ", timeSegmentLength)
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
    Psig.print_params()
    Psig.deltaF = df
    data_dict['H1'] = lalsimutils.non_herm_hoff(Psig)
    data_dict_time['H1'] = lalsimutils.hoft(Psig)
    Psig.detector = 'L1'
    data_dict['L1'] = lalsimutils.non_herm_hoff(Psig)
    data_dict_time['L1'] = lalsimutils.hoft(Psig)
    Psig.detector = 'V1'
    data_dict['V1'] = lalsimutils.non_herm_hoff(Psig)
    data_dict_time['V1'] = lalsimutils.hoft(Psig)

# Reset origin of time, if required
if opts.force_gps_time:
    print(" +++ USER HAS OVERRIDDEN FIDUCIAL EPOCH +++ ")
    print("  The zero of time (and the region to be windowed) will be changed; you had better know what you are doing.  ")
    print("    original " ,theEpochFiducial)
    print("    new      ", opts.force_gps_time)
    theEpochFiducial = lal.GPSTimeNow()
    theEpochFiducial.gpsSeconds = int(opts.force_gps_time)
    theEpochFiducial.gpsNanoSeconds =  int(1e9*(opts.force_gps_time - int(opts.force_gps_time)))

# TODO: Read PSD from XML
psd_dict = {}
if not(opts.psd_file):
    analyticPSD_Q = True # For simplicity, using an analytic PSD
    for det in data_dict.keys():
        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower   #lal.LIGOIPsd    # Preserves key equality in data_dict , psd_dict

else:
    analyticPSD_Q = False # For simplicity, using an analytic PSD
    detectors = data_dict.keys()
    df = data_dict[detectors[0]].deltaF
    for det in detectors:
        print("Reading PSD for instrument %s from %s" % (det, opts.psd_file))
        psd_dict[det] = lalsimutils.pylal_psd_to_swig_psd(lalsimutils.get_psd_series_from_xmldoc(opts.psd_file, det))
        psd_dict[det] = lalsimutils.regularize_swig_psd_series_near_nyquist(psd_dict[det], 80) # zero out 80 hz window near nyquist
        psd_dict[det] =  lalsimutils.enforce_swig_psd_fmin(psd_dict[det], fminSNR)           # enforce fmin at the psd level, HARD CUTOFF
        tmp = psd_dict[det].data.data
        print("Sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), " and maximum is ", np.max(psd_dict[det].data.data))
        deltaF = data_dict[det].deltaF
        # remember the PSD is one-sided, but h(f) is two-sided. The lengths are not equal.
        psd_dict[det] = lalsimutils.extend_swig_psd_series_to_sampling_requirements(psd_dict[det], df, df*(len(data_dict[det].data.data)/2))
        print("Post-extension the new PSD has 1/df = ", 1./psd_dict[det].deltaF, " (data 1/df = ", 1./deltaF, ") and length ", len(psd_dict[det].data.data))
        tmp = psd_dict[det].data.data
        print("Post-extension sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), " and maximum is ", np.max(psd_dict[det].data.data))


### MAIN POINT OF TEST
#  Plot data vs time
detectors = data_dict.keys()
print(detectors, data_dict_time.keys())
if True: # checkInputs:   # Disable until I fix the timing issue
    print(" == Plotting detector data (time domain; requires regeneration, MANUAL TIMESHIFTS,  and seperate code path! Argh!) == ")
    P=Psig.copy()
    P.tref = Psig.tref
    print("Template target time ", float(P.tref))
    print("Data start time", float(data_dict_time[det].epoch))
    print("Zero of time ", float(theEpochFiducial))
    for det in detectors:
        P.detector=det 
        hT = lalsimutils.hoft(P)
        tvals = float(hT.epoch - theEpochFiducial) + hT.deltaT*np.arange(len(hT.data.data))  
        plt.figure(1)
        nSkip = 1
        plt.plot(tvals[::nSkip], np.real(hT.data.data[::nSkip]),label='template:'+det)

        tvals = float(data_dict_time[det].epoch - theEpochFiducial) + data_dict_time[det].deltaT*np.arange(len(data_dict_time[det].data.data))  # Not correct timing relative to zero - FIXME
        plt.plot(tvals[::nSkip],  np.real(data_dict_time[det].data.data[::nSkip]),label='data:'+det,linestyle='--')

    plt.legend()
    plt.xlabel("t(s), based on zero at "+lalsimutils.stringGPSNice(theEpochFiducial))
    plt.ylabel('h(t)')
    plt.xlim(-0.5, 0.5)  # focus on a narrow region near the peak
    plt.show()
