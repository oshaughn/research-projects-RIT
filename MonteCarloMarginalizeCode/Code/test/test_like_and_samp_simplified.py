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
  # Test adaptive sampling with running adaptive exponent
     ./test_like_and_samp.py --Nskip 2000 --LikelihoodType_MargTdisc_array --adapt-beta 0.5  --Niter 100000 --adapt-adapt --convergence-tests-on 
  # NR testing
      ./test_like_and_samp.py --NR-signal-group 'Sequence-GT-Aligned-UnequalMass' --NR-signal-param '(0., 2.)' --signal-mass1 100 --signal-mass2 100 --seglen 32 --approx EOBNRv2HM --fref 0 --signal-distance 1000 --fref 0 --show-input-h --show-likelihood-versus-time
      ./test_like_and_samp.py --NR-signal-group 'Sequence-GT-Aligned-UnequalMass' --NR-signal-param '(0., 2.)' --NR-template-group 'Sequence-GT-Aligned-UnequalMass' --NR-template-param '(0., 1.)'  --signal-mass1 100 --signal-mass2 100 --seglen 32 --approx EOBNRv2HM --fref 0 --signal-distance 1000 --fref 0 --LikelihoodType_MargTdisc_array --show-input-h  --show-likelihood-versus-time

      clear; ./test_like_and_samp.py  --signal-mass1 100 --signal-mass2 100 --seglen 32  --approx EOBNRv2HM --verbose
  # Run with default parameters for injection, with different approximants 
        python test_like_and_samp.py  --show-sampler-inputs --show-sampler-results --show-likelihood-versus-time
        python test_like_and_samp.py --approx TaylorT4 --amporder -1 --Lmax 3 --srate 16384
        python test_like_and_samp.py --approx EOBNRv2HM --Lmax 2 --srate 16384 --skip-interpolation 
  # Use a numerical psd or analytic PSD.  Plot what you are using
        python test_like_and_samp.py --show-psd
        python test_like_and_samp.py --psd-file psd.xml.gz --show-psd
        python test_like_and_samp.py --psd-file-singleifo H1=HLV.xml.gz --psd-file-singleifo V1=HLV.xml.gz
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
  # Run neglecting almost all modes (threshold normally strips out worst cases)
    ./test_like_and_samp.py --Nskip 2000 --approx EOBNRv2HM --srate 16384 --Lmax 5 --LikelihoodType_MargTdisc_array --skip-modes-less-than 1e-2

  # EOB tidal implementation: Make sure source and template are both of consistent mass
     python test_like_and_samp.py  --mass1 1.5 --mass2 1.35  --signal-mass1 1.5 --signal-mass2 1.35 --seglen 128 --verbose
     python test_like_and_samp.py --use-external-EOB  --mass1 1.5 --mass2 1.35  --signal-mass1 1.5 --signal-mass2 1.35 --seglen 128 --verbose
"""

from __future__ import print_function

print("- no matplotlib -")
bNoInteractivePlots = True
bNoMatplotlib=True




import sys

import numpy as np

from glue.lal import Cache
from glue.ligolw import utils, lsctables, table, ligolw,  git_version
from glue.ligolw.utils import process


import lal
import lalsimulation as lalsim
import lalsimutils



__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>, R. O'Shaughnessy"

import factored_likelihood

import xmlutils
import common_cl
import ourparams
import ourio
opts, rosDebugMessagesDictionary = ourparams.ParseStandardArguments()
print(opts)
print(rosDebugMessagesDictionary)

if bNoInteractivePlots:  # change the state only if possible
    bNoInteractivePlots = opts.no_interactive_plots

if opts.verbose:
    try:
        from subprocess import call
        print(" ---- LAL Version ----")
        call(["lal-version"])
        print(" ---- GLUE Version ----")
        call(["ligolw_print",  "--version"])
        print(" ---- pylal Version ----")
        call(["pylal_version"])
    except:
        print("  ... trouble printing version numbers ")
    print(" Glue tag ", git_version.id)
    try:
        import scipy   # used in lalsimutils; also printed in pylal-version
        print(" Scipy: ", scipy.__version__)
    except:
        print(" trouble printing scipy version")

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
print(" Running at most ", nMaxEvals, " iterations")

fracThreshold = opts.points_threshold_match


theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 
tEventFiducial =   0                                                                # relative to GPS reference

det_dict = {}
rhoExpected ={}

tWindowExplore =     factored_likelihood.tWindowExplore

approxSignal = lalsim.GetApproximantFromString(opts.approx)
approxTemplate = approxSignal
ampO =opts.amporder # sets which modes to include in the template (and signal, if injected)
Lmax = opts.Lmax # sets which modes to include in the template.  Print warning if inconsistent.
fref = opts.fref
fref_signal = opts.signal_fref
fminWavesTemplate = opts.fmin_Template  # too long can be a memory and time hog, particularly at 16 kHz
fminWavesSignal = opts.signal_fmin   # If I am using synthetic data, be consistent? #opts.signal_fmin  # too long can be a memory and time hog, particularly at 16 kHz
if fminWavesSignal > fminWavesTemplate:
    print(" WARNING : a choice of fminWavesSignal greater than 0 will cause problems with the cutting and timeshifting code, which requires waveforms to start at the start of the window.")
fminSNR =opts.fmin_SNR
fmaxSNR =opts.fmax_SNR
fSample = opts.srate
window_beta = 0.01

if ampO ==-1 and Lmax < 5:
    print(" +++ WARNING ++++ ")
    print("  : Lmax is ", Lmax, " which may be insufficient to resolve all higher harmonics in the signal! ")

if (ampO+2)> Lmax:
    print(" +++ WARNING ++++ ")
    print("  : Lmax is ", Lmax, " which may be insufficient to resolve all higher harmonics in the signal! ")

if opts.channel_name is not None and opts.cache_file is None:
    print("Cache file required when requesting channel data.", file=sys.stderr)
    exit(-1)
elif opts.channel_name is not None:
    det_dict = dict(map(lambda cname: cname.split("="), opts.channel_name))


#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
Psig = None

# Read in *injection* XML
if opts.inj:
    print("====Loading injection XML:", opts.inj, " =======")
    Psig = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))[opts.event_id]  # Load in the physical parameters of the injection.  
    m1 = Psig.m1
    m2 = Psig.m2
    f_temp = Psig.fmin
    if Psig.fmin < 1e-2:
        Psig.fmin=opts.fmin_SNR  # necessary for duration estimate
    timeWaveform = lalsimutils.estimateWaveformDuration(Psig) #float(-lalsimutils.hoft(Psig).epoch)
    Psig.fmin=f_temp
    Psig.deltaT = 1./fSample  # default sampling rate
    Psig.deltaF = 1./lalsimutils.nextPow2(opts.seglen)       # Frequency binning needs to account for target segment length
    Psig.fref = opts.signal_fref
    theEpochFiducial = Psig.tref  # Reset
    tEventFiducial = 0               # Reset
    print(" ++ Targeting event at time ++ ", lalsimutils.stringGPSNice(Psig.tref))
    print(" +++ WARNING: ADOPTING STRONG PRIORS +++ ")
    rosUseStrongPriorOnParameters= True
#    Psig.print_params()

# Use forced parameters, if provided
if opts.template_mass1 and Psig:
    Psig.m1 = opts.template_mass1*lalsimutils.lsu_MSUN
if opts.template_mass2 and Psig:
    Psig.m2 = opts.template_mass2*lalsimutils.lsu_MSUN
# Use forced parameters, if provided. Note these will override the first set
if opts.signal_mass1 and Psig:
    Psig.m1 = opts.signal_mass1*lalsimutils.lsu_MSUN
if opts.signal_mass2 and Psig:
    Psig.m2 = opts.signal_mass2*lalsimutils.lsu_MSUN
if opts.eff_lambda and Psig:
    lambda1, lambda2 = 0, 0
    if opts.eff_lambda is not None:
        lambda1, lambda2 = lalsimutils.tidal_lambda_from_tilde(m1, m2, opts.eff_lambda, opts.deff_lambda or 0)
        Psig.lambda1 = lambda1
        Psig.lambda2 = lambda2
if Psig and not opts.cache_file:  # Print parameters of fake data
    Psig.print_params()
    print("---- End injection parameters ----")

# Reset origin of time, if required. (This forces different parts of data to be read- important! )
if opts.force_gps_time:
    print(" +++ USER HAS OVERRIDDEN FIDUCIAL EPOCH +++ ")
    print("  The zero of time (and the region to be windowed) will be changed; you had better know what you are doing.  ")
    print("    original " ,lalsimutils.stringGPSNice(theEpochFiducial))
    print("    new      ", opts.force_gps_time)
    theEpochFiducial = lal.GPSTimeNow()
    theEpochFiducial = opts.force_gps_time
#    theEpochFiducial.gpsSeconds = int(opts.force_gps_time)
#    theEpochFiducial.gpsNanoSeconds =  int(1e9*(opts.force_gps_time - int(opts.force_gps_time)))

# Create artificial "signal".  Needed to minimize duplicate code when I
#  -  consistently test waveform duration
#  - copy template parameters 
#  - [test code] : generate plots vs time [test code], expected SNR, etc
#       WARNING: Test code plots will not look perfect, because we don't know the true sky location (or phase, polarization, ...)
if  not Psig and opts.channel_name:  # If data loaded but no signal generated
    if (not opts.template_mass1) or (not opts.template_mass2) or (not opts.force_gps_time):
        print(" CANCEL: For frame-file reading, arguments --mass1 --mass2 --event-time  all required ")
#        print opts.template_mass1, opts.template_mass2,opts.force_gps_time
        sys.exit(0)
    Psig = lalsimutils.ChooseWaveformParams(approx=approxSignal,
        fmin = fminWavesSignal, 
        dist=factored_likelihood.distMpcRef*1e6*lalsimutils.lsu_PC,    # default distance
        fref=fref)
    Psig.m1 = lalsimutils.lsu_MSUN*opts.template_mass1
    Psig.m2 = lalsimutils.lsu_MSUN*opts.template_mass2
    Psig.tref = lal.LIGOTimeGPS(0.000000000)  # Initialize as GPSTime object
    Psig.tref += opts.force_gps_time # Pass value of float into it


# TEST THE SEGMENT LENGTH TARGET
if Psig:
    f_temp = Psig.fmin
    if Psig.fmin < 1e-2:
        Psig.fmin=opts.fmin_SNR  # necessary for duration estimate
    print(" Min frequency for duration ... ", Psig.fmin)
    Psig.print_params()
    timeSegmentLength  = lalsimutils.estimateWaveformDuration(Psig) #-float(lalsimutils.hoft(Psig).epoch)  # needs to work if h(t) unavailable (FD waveform). And this is faster.
    Psig.fmin=f_temp
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
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan,window_shape=window_beta)
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
        taper = lalsimutils.lsu_TAPER_STARTEND
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan, start=int(event_time)-start_pad, stop=int(event_time)+end_pad,window_shape=window_beta)
        fSample = len(data_dict[inst].data.data)*data_dict[inst].deltaF
        df = data_dict[inst].deltaF
        if Psig:
            Psig.deltaF =df
        print("Frequency binning: %f, length %d" % (data_dict[inst].deltaF, len(data_dict[inst].data.data)))
        print("Sampling rate ", fSample)

#        print " Sampling rate of data ", fSample

# CREATE A DEFAULT "signal", if none made to this point.  
analytic_signal = False
if len(data_dict) is 0:
    analytic_signal = True
    print(" Generating signal in memory (no frames or inj)")
    if not(Psig):
        if opts.signal_mass1:
            m1 = opts.signal_mass1*lalsimutils.lsu_MSUN
        else:
            m1 = 4*lalsimutils.lsu_MSUN
        if opts.signal_mass2:
            m2 = opts.signal_mass2*lalsimutils.lsu_MSUN
        else:
            m2 = 3*lalsimutils.lsu_MSUN


        Psig = lalsimutils.ChooseWaveformParams(
            m1 = m1,m2 =m2,
            fmin = fminWavesSignal, 
            fref=fref, 
            tref = theEpochFiducial + tEventFiducial,
            approx=approxSignal,
            ampO=ampO,
            radec=True, theta=1.2, phi=2.4,
            detector='H1', 
            dist=opts.signal_distMpc*1.e6*lalsimutils.lsu_PC,    # move back to aLIGO distances
            deltaT=1./fSample
                                )
        print(" Effective lambda ", opts.eff_lambda)
        if opts.eff_lambda and m1/lal.MSUN_SI<3 and m2/lal.MSUN_SI<3:
            lambda1, lambda2 = 0, 0
            if opts.eff_lambda is not None:
                lambda1, lambda2 = lalsimutils.tidal_lambda_from_tilde(m1, m2, opts.eff_lambda, opts.deff_lambda or 0)
            Psig.lambda1 = lambda1
            Psig.lambda2 = lambda2
        timeSegmentLength  = float(lalsimutils.estimateWaveformDuration(Psig))
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

    df = lalsimutils.estimateDeltaF(Psig)
    if 1/df < opts.seglen:   # Allows user to change seglen of data for *analytic* models, on the command line. Particularly useful re testing PSD truncation
        df = 1./lalsimutils.nextPow2(opts.seglen)
        Psig.deltaF = df  # change the df
    if not opts.NR_signal_group and not opts.use_external_EOB:
        print(" ---  Using synthetic signal --- ")
        Psig.print_params(); print(" ---  Writing synthetic signal to memory --- ")
        Psig.deltaF = df
        Psig.detector='H1'
        data_dict['H1'] = lalsimutils.non_herm_hoff(Psig)
        Psig.detector='L1'
        data_dict['L1'] = lalsimutils.non_herm_hoff(Psig)
        Psig.detector='V1'
        data_dict['V1'] = lalsimutils.non_herm_hoff(Psig)
#        for det in ['H1', 'L1', 'V1']:
#            Psig.detector = det
#            data_dict[det] = lalsimutils.non_herm_hoff(Psig)

    elif opts.use_external_EOB:
        print(" ---  Using external EOB signal --- ")
        Psig.print_params()
        Psig.deltaF = df
        Psig.taper = lalsim.SIM_INSPIRAL_TAPER_START
        # Load the underlying hlm sequence
        wfP = eobwf.WaveformModeCatalog(Psig,lmax=Lmax)
        # Generate the data in each detector
        # Confirm data maximum is not at the edge of the window
        wfP.P.detector='H1'
        data_dict['H1'] = wfP.non_herm_hoff()
        wfP.P.detector='L1'
        data_dict['L1'] = wfP.non_herm_hoff()
        wfP.P.detector='V1'
        data_dict['V1'] = wfP.non_herm_hoff()


    elif   opts.NR_signal_group: # and (Psig.m1+Psig.m2)/lal.MSUN_SI > 50:   # prevent sources < 50 Msun from being generated -- let's not be stupid 
        print(" ---  Using synthetic NR injection file --- ")
        print(opts.NR_signal_group, opts.NR_signal_param)   # must be valid: ourparams.py will thrown an error
        # Load the catalog
        wfP = nrwf.WaveformModeCatalog(opts.NR_signal_group, opts.NR_signal_param, \
                                           clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, 
                                       lmax=Lmax,align_at_peak_l2_m2_emission=True)
        mtot = Psig.m1 + Psig.m2
        q = wfP.P.m2/wfP.P.m1
        wfP.P.m1 = mtot/(1+q)
        wfP.P.m2 = mtot*q/(1+q)
        wfP.P.fref = 0
        wfP.P.tref = theEpochFiducial  # Initialize as GPSTime object  . IMPORTANT - segfault otherwise
        if opts.signal_distMpc:
            wfP.P.dist = opts.signal_distMpc*1e6*lal.PC_SI # default
        else:
            wfP.P.dist = 2500(mtot/200/lal.MSUN_SI)*1e6*lal.PC_SI     # put at a reasonable distance for this mass, to get a reasonable SNR 
        Psig = wfP.P 


        # Estimate the duration needed and generate accordingly
        T_window_needed = max([16, opts.seglen, 2**int(np.log(wfP.estimateDurationSec())/np.log(2)+1)])
        df= Psig.deltaF = 1./T_window_needed   # redefine df!
        print("NR duration window to be used ", 1./Psig.deltaF)
        wfP.P.print_params()

        for det in ['H1', 'L1', 'V1']:
            wfP.P.detector = det
            wfP.P.radec =True  # physical source
            data_dict[det] = wfP.non_herm_hoff()
       
    else:
            print("Not valid NR simulation or injection parameter")
            sys.exit(0)

# Report on signal injected
if opts.verbose:
    print("  ---- Report on detector data ----- ")
    print("  det  length    duration ")
    for det in data_dict:
        print(det, data_dict[det].data.length, data_dict[det].data.length*Psig.deltaT)


# Reset origin of time, if required
if opts.force_gps_time:
    print(" +++ USER HAS OVERRIDDEN FIDUCIAL EPOCH +++ ")
    print("  The zero of time (and the region to be windowed) will be changed; you had better know what you are doing.  ")
    print("    original " ,theEpochFiducial)
    print("    new      ", opts.force_gps_time)
    theEpochFiducial = lal.GPSTimeNow()
#    theEpochFiducial.gpsSeconds = int(opts.force_gps_time)
#    theEpochFiducial.gpsNanoSeconds =  int(1e9*(opts.force_gps_time - int(opts.force_gps_time)))
    theEpochFiducial = opts.force_gps_time  # seconds etc not manually specifiable


# PSD reading
psd_dict = {}
if not(opts.psd_file) and not(opts.psd_file_singleifo):
    analyticPSD_Q = True # For simplicity, using an analytic PSD
    for det in data_dict.keys():
#        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower   #lal.LIGOIPsd    # Preserves key equality in data_dict , psd_dict
#        psd_dict[det] = lal.LIGOIPsd    # Preserves key equality in data_dict , psd_dict
        if (opts.psd_name == "" and det != "V") or (opts.psd_name_V=="" and det == "V"):
            psd_dict[det] = lalsim.SimNoisePSDiLIGOSRD    # Preserves key equality in data_dict , psd_dict.  this is Chris' 'iLIGO' PSD, for test data
        if opts.psd_name_V!= "" and det =='V1':
            if opts.verbose:
                print(" Assigning PSD for", det, opts.psd_name_V)
            psd_dict[det] = eval(opts.psd_name_V)  # Better not screw up!
        if opts.psd_name != ""and (det =='H1'  or det == 'L1'):
            if opts.verbose:
                print(" Assigning PSD for  ", det,  opts.psd_name)
            psd_dict[det] = eval(opts.psd_name)  # Better not screw up!

else:
    analyticPSD_Q = False # For simplicity, using an analytic PSD
    detectors = data_dict.keys()
    if opts.psd_file_singleifo:
        detectors_singlefile_dict = common_cl.parse_cl_key_value(opts.psd_file_singleifo)
    else:
        detectors_singlefile_dict ={}
    df = data_dict[detectors[0]].deltaF
    fNyq = (len(data_dict[detectors[0]].data.data)/2)*df
    print(" == Loading numerical PSDs ==")
    for det in detectors:
        psd_fname = ""
        if detectors_singlefile_dict.has_key(det):
            psd_fname = detectors_singlefile_dict[det]
        else:
            psd_fname = opts.psd_file
        print("Reading PSD for instrument %s from %s" % (det, psd_fname))

        # "Standard" PSD parsing code used on master.
        psd_dict[det] = lalsimutils.get_psd_series_from_xmldoc(psd_fname, det)  # pylal type!
        tmp = psd_dict[det].data
        print("Sanity check reporting : pre-extension, min is ", np.min(tmp), " and maximum is ", np.max(tmp))
        deltaF = data_dict[det].deltaF
        fmin = psd_dict[det].f0
        fmax = fmin + psd_dict[det].deltaF*len(psd_dict[det].data)-deltaF
        print("PSD deltaF before interpolation %f" % psd_dict[det].deltaF)
        psd_dict[det] = lalsimutils.resample_psd_series(psd_dict[det], deltaF)
        print("PSD deltaF after interpolation %f" % psd_dict[det].deltaF)
        print("Post-extension the new PSD has 1/df = ", 1./psd_dict[det].deltaF, " (data 1/df = ", 1./deltaF, ") and length ", len(psd_dict[det].data.data))
        tmp = psd_dict[det].data.data
        nBad = np.argmin(tmp[np.nonzero(tmp)])
        fBad = nBad*deltaF
        print("Post-extension sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), "(at n=", np.argmin(tmp[np.nonzero(tmp)])," or f=", fBad, ")  and maximum is ", np.max(psd_dict[det].data.data))
        print()

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
print(" == Data report == ")
detectors = data_dict.keys()
rho2Net = 0
print(" Amplitude report :")
fminSNR =opts.fmin_SNR
for det in detectors:
    if analyticPSD_Q:
        IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det],fMax=fmaxSNR)
    else:
        IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det].data.data,fMax=fmaxSNR,analyticPSD_Q=False)
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print(det, " rho = ", rhoDet)
print("Network : ", np.sqrt(rho2Net))

if opts.plot_ShowH and not bNoMatplotlib: # and not bNoInteractivePlots:
    print(" == Plotting FRAME DATA == ")
    plt.figure(2)
    for det in detectors:
        hT = lalsimutils.DataInverseFourier(data_dict[det])  # complex inverse fft, for 2-sided data
        # Roll so we are centered
        hT = lalsimutils.DataRollBins(hT,-len(data_dict[det].data.data)/2)
        print("  : Confirm nonzero data! : ",det, np.max(np.abs(data_dict[det].data.data)))
        tvals = float(hT.epoch - theEpochFiducial) + hT.deltaT*np.arange(len(hT.data.data))
        plt.plot(tvals, hT.data.data,label=det)
    plt.title(" data_dict h(t)")
    plt.legend()
    plt.savefig("test_like_and_samp-frames-hoft."+fExtension)
    plt.xlim(-0.5,0.5)  # usually centered around t=0
    plt.savefig("test_like_and_samp-frames-zoomed-hoft."+fExtension)
    print(" == Plotting TEMPLATE (time domain; requires regeneration, MANUAL TIMESHIFTS,  and seperate code path! Argh!) == ")
    if Psig:
        P = Psig
        for det in detectors:
            P.detector=det   # we do 
            hT = lalsimutils.hoft(P)
            tvals = float(hT.epoch -theEpochFiducial)+ hT.deltaT*np.arange(len(hT.data.data))
            plt.figure(1)
            plt.plot(tvals, hT.data.data,label=det)
        plt.title(" Template data h(t)")
        plt.legend()
        plt.savefig("test_like_and_samp-injection-hoft."+fExtension)
    if not bNoInteractivePlots:
        plt.show()

# Load skymap, if present
if opts.opt_UseSkymap:
    print(" ==Loading skymap==")
    print("   skymap file ", opts.opt_UseSkymap)
    smap, smap_meta = bfits.read_sky_map(opts.opt_UseSkymap)
    sides = healpy.npix2nside(len(smap))

    if opts.plot_ShowSamplerInputs and not bNoInteractivePlots:
        try:
            from lalinference.bayestar import plot
            plt.subplot(111, projection='astro mollweide')
            plot.healpix_heatmap(smap)
            plt.show()
        except:
            print(" No skymap for you ")


# Get masses: Note if no signal, these need to be defined

# Struct to hold template parameters
P = lalsimutils.ChooseWaveformParams(fmin=fminWavesTemplate, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         approx=approxTemplate,
         fref=fref,
         tref=theEpochFiducial,
         deltaT=1./fSample,
         dist=100*1.e6*lalsimutils.lsu_PC,
         deltaF=df)
try:
  if lambda1 or lambda2:
    P.lambda1 = lambda1
    P.lambda2 = lambda2
except:
    True
#
# Perform the Precompute stage
#   WARNING: Using default values for inverse spectrum truncation (True) and inverse spectrun truncation time (8s) from ourparams.py
#                     ILE adopts a different convention.  ROS old development branch has yet another approach (=set during PSD reading).
#
rholms_intp, crossTerms, crossTermsV, rholms, rest = factored_likelihood.PrecomputeLikelihoodTerms(theEpochFiducial,tWindowExplore[1], P, data_dict,psd_dict, Lmax, fmaxSNR, analyticPSD_Q,ignore_threshold=opts.opt_SkipModeThreshold,inv_spec_trunc_Q=opts.psd_TruncateInverse,T_spec=opts.psd_TruncateInverseTime,NR_group=opts.NR_template_group,NR_param=opts.NR_template_param,use_external_EOB=opts.use_external_EOB,ROM_group=opts.ROM_template_group,ROM_param=opts.ROM_template_group)


epoch_post = theEpochFiducial # Suggested change.  BE CAREFUL: Now that we trim the series, this is NOT what I used to be
print("Finished Precomputation...")
print("====Estimating distance range =====")
distBoundGuess = factored_likelihood.estimateUpperDistanceBoundInMpc(rholms, crossTerms)
print(" distance probably less than ", distBoundGuess, " Mpc")

# set the lnL overflow value using the peak U value, scaled
print("===Setting lnL offset to minimize overflow===")
lnLOffsetValue = 0
for det in crossTerms.keys():
    for mode in crossTerms[det].keys():  # actually a loop over pairs
        if np.abs(crossTerms[det][mode]) > lnLOffsetValue:
            lnLOffsetValue = 0.2*np.abs(crossTerms[det][mode])*(factored_likelihood.distMpcRef/distBoundGuess)**2
lnLOffsetValue =0
print("+++using lnL offset value++", lnLOffsetValue)
if lnLOffsetValue > 200:
    print(" ARE YOU SURE THIS LARGE OF AN OFFSET IS A GOOD IDEA?")
    print(" ARE YOU REMOTELY CLOSE TO THE CORRECT PART OF PARAMETER SPACE?")





#
# Call the likelihood function for various extrinsic parameter values
# Uses the (already-allocated) template structure "P" structure *only* to pass parameters.  All parameters used should be specified.
#
nEvals = 0
if True:
    def likelihood_function(right_ascension, declination,t_ref, phi_orb, inclination,
            psi, distance):
        global nEvals
        global lnLOffsetValue
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

            P.phi = float(ph) # right ascension
            P.theta = float(th) # declination
            P.tref = float(theEpochFiducial)  # see 'tvals', above
            P.phiref = float(phr) # ref. orbital phase
            P.incl = float(ic) # inclination
            P.psi = float(ps) # polarization angle
            P.dist = float(di* 1.e6 * lalsimutils.lsu_PC) # luminosity distance

            lnL[i] = factored_likelihood.FactoredLogLikelihoodTimeMarginalized(tvals,
                    P, rholms_intp,rholms, crossTerms, crossTermsV,                   
                    Lmax)
            i+=1
        
        nEvals +=i # len(tvals)  # go forward using length of tvals
        return np.exp(lnLOffsetValue)*np.exp(lnL-lnLOffsetValue)

import RIFT.integrators.mcsampler as mcsampler
sampler = mcsampler.MCSampler()

# Populate sampler 
## WARNING: CP has changed pinning interface again.  Need to rework
pinned_params = ourparams.PopulateSamplerParameters(sampler, theEpochFiducial,tEventFiducial, distBoundGuess, Psig, opts)
unpinned_params = set(sampler.params) - set(pinned_params)


import time
tManualStart = time.clock()
tGPSStart = lal.GPSTimeNow()
print(" Unpinned : ", unpinned_params)
print(" Pinned : ",  pinned_params)
pinned_params.update({"n": opts.nskip, "nmax": opts.nmax, "neff": opts.neff, "full_output": True, "verbose":True, "extremely_verbose": opts.super_verbose,"igrand_threshold_fraction": fracThreshold, "igrandmax":rho2Net/2, "save_intg":True,
    "convergence_tests" : test_converged,    # Dictionary of convergence tests

    "tempering_exp":opts.adapt_beta,
    "tempering_log": opts.adapt_log,
    "tempering_adapt": opts.adapt_adapt,
    "floor_level": opts.adapt_mix, # The new sampling distribution at the end of each chunk will be floor_level-weighted average of a uniform distribution and the (L^tempering_exp p/p_s)-weighted histogram of sampled points.
    "history_mult": 10, # Multiplier on 'n' - number of samples to estimate marginalized 1-D histograms
    "n_adapt": 100, # Number of chunks to allow adaption over
    "igrand_threshold_deltalnL": opts.save_deltalnL, # Threshold on distance from max L to save sample
    "igrand_threshold_p": opts.save_P # Threshold on cumulative probability contribution to cache sample

})
print(" Params ", pinned_params)
res, var,  neff , dict_return = sampler.integrate(likelihood_function, *unpinned_params, **pinned_params)

print(sampler._rvs.keys())
field_names = ['m1', 'm2', 'ra','dec', 'tref', 'phi', 'incl', 'psi', 'dist', 'p', 'ps', 'lnL']   # FIXME: Modify to use record array, so not hardcoding fields
retNew = [P.m1/lalsimutils.lsu_MSUN*np.ones(len(sampler._rvs['right_ascension'])), P.m2/lalsimutils.lsu_MSUN*np.ones(len(sampler._rvs['right_ascension'])), sampler._rvs["right_ascension"], sampler._rvs['declination'],sampler._rvs['t_ref'], sampler._rvs['phi_orb'],sampler._rvs['inclination'], sampler._rvs['psi'],  sampler._rvs['distance'], sampler._rvs["joint_prior"], sampler._rvs["joint_s_prior"],np.log(sampler._rvs["integrand"])]
retNew = map(list, zip(*retNew))
ret = np.array(retNew)
retNiceIndexed = np.array(np.reshape(ret,-1)).view(dtype=zip(field_names, ['float64']*len(field_names))).copy()  # Nice record array, use for recording outuput without screwing up column indexing.  Reshape makes sure the arrays returned are 1d.  Silly syntax.
#ret = np.array(retNew)

tGPSEnd = lal.GPSTimeNow()
tManualEnd = time.clock()
print("Parameters returned by this integral ",  sampler._rvs.keys(), len(sampler._rvs))
ntotal = nEvals # opts.nmax  # Not true in general
print(" Evaluation time  = ", float(tGPSEnd - tGPSStart), " seconds")
print(" lnLmarg is ", np.log(res), " with nominal relative sampling error ", np.sqrt(var)/res, " but a more reasonable estimate based on the lnL history is ") #, np.std(lnLmarg - np.log(res))
print(" expected largest value is ", rho2Net/2, "and observed largest lnL is ", np.max(np.transpose(ret)[-1]))
print(" note neff is ", neff, "; compare neff^(-1/2) = ", 1/np.sqrt(neff))

print("==Returned dictionary===")
print(dict_return)


print("==Profiling info===")
print("   - Time per L evaluation ", float(tGPSEnd-tGPSStart)/ntotal,   (tManualEnd-tManualStart)/ntotal)
print("   - Time per neff             ", float(tGPSEnd-tGPSStart)/neff)


# Save the sampled points to a file
# Only store some
fnameBase = opts.points_file_base
retSorted = ret[ np.argsort(ret[:,-1])]
ourio.dumpSamplesToFile(fnameBase+"-points.dat", retSorted, field_names) 
#sampArray = Psig.list_params()  # Eventually, make this used. Note odd structure in list
#np.savetxt(fnameBase+"-params.dat", np.array(sampArray))
#print " Parameters : ", sampArray
ourio.dumpSamplesToFile(fnameBase+'-result.dat', np.array([[res, np.sqrt(var), np.max(ret[:,-1]),ntotal,neff, P.m1/lalsimutils.lsu_MSUN,P.m2/lalsimutils.lsu_MSUN]]), ['Z', 'sigmaZ', 'lnLmax','N', 'Neff','m1','m2'])  # integral, std dev,  total number of points
#np.savetxt(fnameBase+'-result.dat', [res, np.sqrt(var), ntotal])   # integral, std dev,  total number of points. Be SURE we do not lose precision!
#np.savetxt(fnameBase+'-dump-lnLmarg.dat',lnLmarg[::opts.nskip])  # only print output periodically -- otherwise far too large files!

if neff > 5 or opts.force_store_metadata:  # A low threshold but not completely implausible.  Often we are not clueless 
    print("==== Computing and saving metadata for future runs: <base>-seed-data.dat =====")
    print(" Several effective points producted; generating metadata file ")
    if neff < 20:
        print("  +++ WARNING +++ : Very few effective samples were found. Be VERY careful about using this as input to subsequent searches! ")
    metadata={}
    weights = np.exp(ret[:,-1])*ret[:,-3]/ret[:,-2]
    metadata["ra"] =  mean_and_dev(ret[:,-3-7], weights)
    metadata["dec"] = mean_and_dev(ret[:,-3-6], weights)
    metadata["tref"] =  mean_and_dev(ret[:,-3-5], weights)
    metadata["phi"] =  mean_and_dev(ret[:,-3-4], weights)
    metadata["incl"] =  mean_and_dev(ret[:,-3-3], weights)
    metadata["psi"] =  mean_and_dev(ret[:,-3-2], weights)
    metadata["dist"] =  mean_and_dev(ret[:,-3-1], weights)
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
    print("==== PP data: <base>-pp-instance.dat =====")
    lnLAt = factored_likelihood.FactoredLogLikelihood(Psig, rholms_intp, crossTerms, crossTermsV, Lmax)
    # Evaluate best data point
    ppdata = {}
    weights = np.exp(ret[:,-1])*ret[:,-3]/ret[:,-2]
    ppdata['ra'] = [Psig.phi,pcum_at(Psig.phi,ret[:,2],weights)]
    ppdata['dec'] = [Psig.theta,pcum_at(Psig.theta,ret[:,3], weights)]
    if not opts.LikelihoodType_MargTdisc_array:
        ppdata['tref'] = [Psig.tref-theEpochFiducial,pcum_at(Psig.tref-theEpochFiducial,ret[:,4], weights)]
    else:
        ppdata['tref'] = [0,0,0]
    ppdata['phi'] = [Psig.phiref,pcum_at(Psig.phiref,ret[:,5], weights)]
    ppdata['incl'] = [Psig.incl,pcum_at(Psig.incl,ret[:,6], weights)]
    ppdata['psi'] = [Psig.psi,pcum_at(Psig.psi,ret[:,7], weights)]
    ppdata['dist'] = [Psig.dist/(1e6*lalsimutils.lsu_PC),pcum_at(Psig.dist/(1e6*lalsimutils.lsu_PC),ret[:,8], weights)]
    ppdata['lnL'] =  [lnLAt, pcum_at(lnLAt, ret[:,-1], weights)]

    # Dump data: p(<x)
    with open(fnameBase+"-pp-data.dat",'w') as f:
        for key in ['ra','dec', 'tref', 'phi', 'incl', 'psi', 'dist', 'lnL']: 
            f.write(key + " " + str(ppdata[key][0]) + ' '+ str(ppdata[key][1]) + '\n')


# Save the outputs in CP's format, for comparison.  NOT YET ACTIVE CODE -- xmlutils has a bug on master (lacking terms in dictionary)
if  True: # opts.points_file_base:
    print("==== Exporting to xml: <base>.xml.gz =====")
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    opts.NR_template_param = ""
#    process.register_to_xmldoc(xmldoc, sys.argv[0], opts.__dict__)    # the process-params generation has not been reliable
    process.register_to_xmldoc(xmldoc, sys.argv[0], {})    # the process-params generation has not been reliable
    samples = {}
    samples["distance"]= ret[:,-3-1] #retNiceIndexed['dist']
    samples["t_ref"] = ret[:,-3-5] #retNiceIndexed['tref']
    samples["polarization"]= ret[:,-3-1] #retNiceIndexed['psi']
    samples["coa_phase"]= ret[:,-3-4] #retNiceIndexed['phi']
    samples["latitude"]= retNiceIndexed['dec']
    samples["longitude"]= retNiceIndexed['ra']
    samples["inclination"]= retNiceIndexed['incl']
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
    m1 = P.m1/lalsimutils.lsu_MSUN
    m2 = P.m2/lalsimutils.lsu_MSUN
    samples["mass1"] = np.ones(samples["polarization"].shape)*m1
    samples["mass2"] = np.ones(samples["polarization"].shape)*m2
#    utils.write_fileobj(xmldoc,sys.stdout)
    xmlutils.append_samples_to_xmldoc(xmldoc, samples)
#    utils.write_fileobj(xmldoc,sys.stdout)
    xmlutils.append_likelihood_result_to_xmldoc(xmldoc, np.log(res), **{"mass1": m1, "mass2": m2})
    utils.write_filename(xmldoc, opts.points_file_base+".xml.gz", gz=True)

