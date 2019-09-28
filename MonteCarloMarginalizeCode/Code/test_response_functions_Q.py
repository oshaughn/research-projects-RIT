#! /usr/bin/env python
# test_response_functions_Q.py
#    - Plot Qlm(t) [specifically, Q22] on response to a 'delta' function h(t)
#      By default, the 'delta' function is a gaussian
#    - Uses full test_like_and_samp.py infrastructure, so can read in data as well.
#
# USER MUST MANUALLY CHANGE
#   To turn on and off the discontinuity and windowing
# EXAMPLE
#  -  Plot response functions for the 'discontinuous' function plus signal, using a smooth iLIGO scalePSD
#         test_response_functions_Q.py --psd-file mypsd.xml.gz  
#         test_response_functions_Q.py --psd-file mypsd.xml.gz   --seglen 64  # repeat with more data
#
#   - Change the template durations, to move the signal around in time
#         test_response_functions_Q.py --psd-file mypsd.xml.gz   --seglen 64  # repeat with more data
#
#  - Repeat with an aLIGO-scale noisy (numerical) PSD
#         test_response_functions_Q.py --psd-file mypsd-noisy.xml.gz  --fmin-template 40 --signal-fmin 22 --show-psd --signal-dist 500     # use a noisy, numerical PSD

#  - Repeat for truncated inverse psd (e.g., noisy V1)
#    python test_response_functions_Q.py  --psd-file mypsd-noisy.xml.gz  --seglen 64 --fmin-template 30 --signal-fmin 22 --show-psd --psd-truncate-inverse --psd-truncate-inverse-time 8 --signal-dist 500
#    python test_response_functions_Q.py  --psd-file mypsd-noisy.xml.gz  --seglen 64  --show-psd --signal-fmin 22 --psd-truncate-inverse --psd-truncate-inverse-time 16 --signal-dist 500

#
#
#  - Reading in real data. [Typically, inj xml or coinc needed. Make sure seglen chosen *and* default instrument channel provided!]
#    Doing so, make *absolutely certain* the correct fref, Lmax, amporder are used!
#          python ../../Code/test_response_functions_Q.py --cache zero_noise.cache   --psd-file HLV-ILIGO_PSD.xml.gz --inj mdc.xml.gz  --fref 0 --Lmax 3 --amporder -1
#          python ../../Code/test_response_functions_Q.py --cache zero_noise.cache  --psd-file HLV-ILIGO_PSD.xml.gz --psd-truncate-inverse --psd-truncate-inverse-time 8 --inj mdc.xml.gz --fref 0 --Lmax 3 --amporder -1
#          python ../../Code/test_response_functions_Q.py --cache iligo_noise.cache  --psd-file HLV-ILIGO_PSD.xml.gz --psd-truncate-inverse --psd-truncate-inverse-time 8 --inj mdc.xml.gz --fref 0 --Lmax 3 --amporder -1
#          python ../../Code/test_response_functions_Q.py --cache iligo_noise.cache  --psd-file HLV-ILIGO_PSD.xml.gz --psd-truncate-inverse --psd-truncate-inverse-time 8 --inj mdc.xml.gz  --seglen 31 --padding 2  --fref 0 --Lmax 3 --amporder -1  #  basically HUGELY zero-pad data, to push ringing/response off. Not perfect
#         python ../../Code/test_response_functions_Q.py --cache iligo_noise.cache  --psd-file HLV-ILIGO_PSD.xml.gz  --channel V1=FAKE-STRAIN  --inj mdc.xml.gz  --seglen 31 --fref 0 --Lmax 3 --amporder -1 --show-likelihood-versus-time; open test-Q-* FLT-*
#        python ../../Code/test_response_functions_Q.py --cache iligo_noise.cache  --psd-file HLV-ILIGO_PSD.xml.gz   --channel V1=FAKE-STRAIN --channel H1=FAKE-STRAIN --channel L1=FAKE-STRAIN  --inj mdc.xml.gz  --seglen 31 --fref 0 --Lmax 3 --amporder -1 --psd-truncate-inverse --show-likelihood-versus-time --show-psd; open test-Q-* FLT-*

#        python ../../Code/test_response_functions_Q.py --cache s5_noise.cache  --psd-file HLV-ILIGO_PSD.xml.gz   --channel V1=FAKE-STRAIN --channel H1=FAKE-STRAIN --channel L1=FAKE-STRAIN  --inj mdc.xml.gz  --seglen 31 --fref 0 --Lmax 3 --amporder -1 --psd-truncate-inverse --show-likelihood-versus-time --show-psd; open test-Q-* FLT-*

from __future__ import print_function

try:
    import matplotlib
    #matplotlib.use("Agg")
    matplotlib.use("GDK")
    from matplotlib import pylab as plt
except:
    print("- no matplotlib -")


import lalsimutils
import numpy as np
import lal
import lalsimulation as lalsim

import factored_likelihood
import factored_likelihood_test

import sys
import functools
import pickle
import numpy

try:
    import healpy
    from lalinference.bayestar import fits as bfits
    from lalinference.bayestar import plot as bplot
except:
    print(" -no skymaps - ")


from glue.lal import Cache
from glue.ligolw import utils, lsctables, table, ligolw, git_version
from glue.ligolw.utils import process

import xmlutils
import ourparams
import ourio
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()
factored_likelihood.rosDebugMessagesDictionary = rosDebugMessagesDictionary
lalsimutils.rosDebugMessagesDictionary            = rosDebugMessagesDictionary
print(opts)
print(rosDebugMessagesDictionary)

print(" ======= TESTING RESPONSE FUNCTION AND DISCONTINITY ====")
print("   Edit code variables 'bUseWindow' and 'bAddDiscontinity' to insert a plausible discontinuity ")
bUseWindow = True
bAddDiscontinuity = False
window_beta = 0.01
detDefault=  'V1'

def fakeGaussianDataTime(amp,sigma,deltaT, npts, window='Tukey', window_beta=0.01):
    # Create h(t)
    htGauss =lal.CreateCOMPLEX16TimeSeries("h", lal.LIGOTimeGPS(0.), 0., deltaT, lal.lalDimensionlessUnit, npts)
    htGauss.data.data[:] = 0.j  # initialize the data

    # Populate first few samples with a half-gaussian , +
    tvals = np.arange(npts)*deltaT
    htGauss.data.data += amp*np.exp(-tvals*tvals/(sigma*sigma)/2.)

    # Populate the last few samples with a half-gaussian, -
    tvalsMinus = np.arange(npts)[::-1]*deltaT
    htGauss.data.data += -amp*np.exp(-tvalsMinus*tvalsMinus/(sigma*sigma)/2.)

    if window:
        windowArray  = lal.CreateNamedREAL8Window(window, window_beta, len(htGauss.data.data))  # Window (start)
        htGauss.data.data *= windowArray.data.data

    return htGauss 
    
def DataFourier(ht):   # Complex fft wrapper (COMPLEX16Time ->COMPLEX16Freq. No error checking or padding!
    TDlen = ht.data.length
    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)
    hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            ht.epoch, ht.f0, 1./ht.deltaT/TDlen, lal.lalHertzUnit, 
            TDlen)
    lal.COMPLEX16TimeFreqFFT(hf, ht, fwdplan)
    # assume memory freed by swig python
    return hf
def DataInverseFourier(hf):   # Complex fft wrapper (COMPLEX16Freq ->COMPLEX16Time. No error checking or padding!
    FDlen = hf.data.length
    dt = 1./hf.deltaF/FDlen
    revplan=lal.CreateReverseCOMPLEX16FFTPlan(FDlen,0)
    ht = lal.CreateCOMPLEX16TimeSeries("Template h(t)", 
            hf.epoch, hf.f0, dt, lal.lalDimensionlessUnit, 
            FDlen)
    lal.COMPLEX16FreqTimeFFT( ht, hf, revplan)  
    # assume memory freed by swig python
    return ht
## TEST and DEMO of truncate_inverse_psd
# fNyq = 2048
# df = 1/64
# psd = lalsimutils.get_psd_series_from_xmldoc("psd.xml.gz")
# psd = lalsimutils.resample_psd_series(psd, df, 0, fNyq)
# psd = lal.ResizeREAL8FrequencySeries(psd,0, fNyq/df+1)
# psdNew = lalsimutils.truncate_inverse_psd(psd,8,40,2000)   # impose hard limits at 40,2000 Hz
def truncate_inverse_psd(swig_psd,Tkeep,fmin,fmax):
    """
    Takes swig_psd, Tkeep,fmax.  Conputes inverse 1/psd^(1/2) (iFFT of sqrt(weights2side)) in the time domain, zeroing all except [-Tkeep/2, Tkeep/2]
    around t=0, inverse-fft's, and returns a 'revised' PSD, corresponding to a revised (finite-length) time duration filter.   
   *Requires* PSD  initialized up to its nyquist frequency, in swig format, with pow2+1  entries.

    *fmin* and *fmax* are supplied explicitly by the user, to *ensure* any hard cutoffs are explicitly included in the PSD model,
    using *exactly* the same code paths used elsewhere/previously
        [Even without lines, the low-frequency cutoff introduces long-duration 'ringing' into the whitening filter, which this program is intended to truncate]
        [*WARNING*: This means that in *subsequent* uses of this PSD, you need to set flow = 0 ! ]

    Implemented using 'IP' infrastructure  (IP.intgd = freq; IP.ovlp = time) to allocate memory and map the PSD arrays,
    to insure consistent length requirements and bin alignment (i.e., code path) as implementation used in inner product.
    (Ideally, should probably use IP to return the inverse FFT too)

    For comparison: https://www.lsc-group.phys.uwm.edu/daswg/projects/lal/nightly/docs/html/_find_chirp_s_p_data_8c_source.html#l00431
    Reference: Section 4.7 in Duncan's thesis http://arxiv.org/pdf/0705.1514v1.pdf 
    """

    # Create frequencies : df, fNyq
    df = swig_psd.deltaF
    fNyq = df * (len(swig_psd.data.data) -1)   # Implicit assumption about odd length of raw psd file
    T = 1/df  # = N deltaT
    # We cannot truncate if the timescale is too short. Return untruncated PSD (not safe; perhaps die?)
    if T<1.5*Tkeep:
        return swig_psd
    # Proceed: make safety copy of swig_psd, to prevent accidentally changing it by numpy magic (e.g., side effects in IP)
    swig_psd_copy = lal.CutREAL8FrequencySeries(swig_psd,0, len(swig_psd.data.data)) # swig passes pointers, don't change the original array by accident

    # Create, extract 'weights2side' [PSD in array form] and two pre-allocated buffers
#    print " Creating IP structure, truncating the PSD over the range ", [fmin, fmax]
    # Master branch arguments
    IP = lalsimutils.ComplexOverlap(fLow=fmin, fMax=fmax,fNyq=fNyq, deltaF=swig_psd_copy.deltaF,psd=swig_psd_copy.data.data, analyticPSD_Q=False)
    npts = len(IP.intgd.data.data)  # = IP.len2side
    wt = lal.CutCOMPLEX16FrequencySeries(IP.intgd,0,npts)  # instantiate 2-sided freq series, copied from IP.  [associated timeseries in 'IP.ovlp'-> xt]
    wt.data.data = np.sqrt(IP.weights2side) # populate with weights (=1/sqrt(PSD)].  This includes any truncation (flow, fmax, etc)

    # SCALE FACTOR
    #   - concern we did not keep enough numerical precision to preserve zeros
    wtScale = np.abs(np.max(wt.data.data[np.nonzero(wt.data.data)]))
    wt.data.data = wt.data.data/wtScale
    # Inverse FFT to create timeseries (ROS wrapper)
    xt = DataInverseFourier(wt)

    # Truncate the series, targeting a specific duration.  (Up to user to make sure these durations are sane)
    dt =xt.deltaT
    nSamplesToZero = int((T-Tkeep)/dt/2)  # The ifft array in *time* has t=0 at the first sample. Zero out the middle 2*nSamplesToZero points
    nSamples = xt.data.length
    xt.data.data[nSamplesToZero+1:nSamples-nSamplesToZero] = 0.   # Be careful re potential fencepost error here.
    # FFT the truncated weights
    wtNew = DataFourier(xt)


    # Allocate and populate a *one-sided* REAL8FrequencySeries, of correct length. [Remember, we assume fully allocated PSD at start]
    # Avoid direct numpy assignment, since mixing memory models produces segfaults
    swig_psd_out = lal.CutREAL8FrequencySeries(swig_psd,0, swig_psd.data.length)  # make *another* copy, to write into
    #swig_psd_out.data.data = np.zeros(len(swig_psd_out.data.data))
    tmp =  wtNew.data.data[0:npts/2+1]  # Assign data from first half, including f=0 bin. BE CAREFUL re potential fenceposts
    
    tmp = tmp[::-1]                               # Reverse so f=0 is first bin. Remember f=0 is one past midpoint in original array
    tmpSquare = np.abs(tmp)**2            #  create 1/S^2 array. Because the FFT is 2-sided, there is a small complex part created. Use np.abs() to force real.
    for i in np.arange(swig_psd_out.data.length - 1):
        swig_psd_out.data.data[i]=0
        if tmpSquare[i]:   # avoid underflow
            swig_psd_out.data.data[i] = 1./tmpSquare[i]  # assign
    # Set the zero and nyquist bin
    swig_psd_out.data.data[0] = np.inf
    swig_psd_out.data.data[-1] =0.
    # Apply scale factor at end
    swig_psd_out.data.data *= 1/np.power(wtScale,2)  # re-insert scale factor. Remember wt scales as longweights. This tries to keep numbers in the FFT's of order unity, to avoid precision loss

    # Return truncated PSD
    print("   Truncated PSD created ")
    return swig_psd_out


def fakeGaussianDataFrequency(amp, sigma,deltaT,npts,window='Tukey', window_beta=0.01):
    return DataFourier(fakeGaussianDataTime(amp,sigma,deltaT,npts,window=window,window_beta=window_beta))

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

__author__ = "R. O'Shaughnessy"


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
        ampO=ampO      
        )  # FIXME: Parameter mapping from trigger space to search space
    if rosDebugMessagesDictionary["DebugMessages"]:
        print(" === Coinc table : estimated signal [overridden if injection] ===")
        Psig.print_params()

# Read in *injection* XML
if opts.inj:
    print("Loading injection XML:", opts.inj)
    Psig = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))[opts.event_id]  # Load in the physical parameters of the injection.  
    m1 = Psig.m1
    m2 = Psig.m2
    timeWaveform = float(-lalsimutils.hoft(Psig).epoch)
    Psig.deltaF = 1./lalsimutils.nextPow2(opts.seglen)       # Frequency binning needs to account for target segment length
    theEpochFiducial = Psig.tref  # Reset
    tEventFiducial = 0               # Reset
    print(" ++ Targeting event at time ++ ", lalsimutils.stringGPSNice(Psig.tref))
    print(" +++ WARNING: ADOPTING STRONG PRIORS +++ ")
    rosUseStrongPriorOnParameters= True
    Psig.print_params()
    print("---- End injeciton parameters ----")

# Use forced parameters, if provided
if opts.template_mass1:
    Psig.m1 = opts.template_mass1*lal.LAL_MSUN_SI
if opts.template_mass2:
    Psig.m2 = opts.template_mass2*lal.LAL_MSUN_SI

# Reset origin of time, if required. (This forces different parts of data to be read- important! )
if opts.force_gps_time:
    print(" +++ USER HAS OVERRIDDEN FIDUCIAL EPOCH +++ ")
    print("  The zero of time (and the region to be windowed) will be changed; you had better know what you are doing.  ")
    print("    original " ,lalsimutils.stringGPSNice(theEpochFiducial))
    print("    new      ", opts.force_gps_time)
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
        print(" CANCEL: Specifying parameters via m1, m2, and time on the command line ")
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
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan,window_shape=window_beta)
        fSample = len(data_dict[inst].data.data)*data_dict[inst].deltaF
        df = data_dict[inst].deltaF
        if Psig:
            Psig.deltaF = df
        print("Frequency binning: %f, length %d" % (data_dict[inst].deltaF, len(data_dict[inst].data.data)))
        print("Sampling rate ", fSample)
    detDefault = (data_dict.keys())[0]
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
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan, start=int(event_time)-start_pad, stop=int(event_time)+end_pad,window_shape=window_beta)
        fSample = len(data_dict[inst].data.data)*data_dict[inst].deltaF
        df = data_dict[inst].deltaF
        if Psig:
            Psig.deltaF =df
        print("Frequency binning: %f, length %d" % (data_dict[inst].deltaF, len(data_dict[inst].data.data)))
        print("Sampling rate ", fSample)

    detDefault = (data_dict.keys())[0]
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
    if 1/df < opts.seglen:   # Allows user to change seglen of data for *analytic* models, on the command line. Particularly useful re testing PSD truncation
        df = 1./lalsimutils.nextPow2(opts.seglen)
    Psig.print_params()
    Psig.deltaF = df

    # Create gaussian fake data
    fNyqTarget = 2048.
    npts = 2*int(fNyqTarget/df)
    amp = 1e-15
    window_beta = window_beta
    if not bAddDiscontinuity:
        amp *=0
    if bUseWindow:
        # create signal first, to set epoch
        data_dict[detDefault] = lalsimutils.non_herm_hoff(Psig)
        sig = fakeGaussianDataFrequency(amp,2, 1/(2*fNyqTarget),npts,window='Tukey',window_beta=window_beta)
    else:
        data_dict[detDefault] = lalsimutils.non_herm_hoff(Psig)
        sig = fakeGaussianDataFrequency(amp,2, 1/(2*fNyqTarget),npts)

    data_dict[detDefault].data.data += sig.data.data


    # Report on fake data
    print(" ============")
    print(" Fake data report : (n,df,fNyq) ", data_dict[detDefault].data.length, data_dict[detDefault].deltaF, data_dict[detDefault].deltaF*data_dict[detDefault].data.length/2)



# PSD reading
psd_dict = {}
if not(opts.psd_file):
    analyticPSD_Q = True # For simplicity, using an analytic PSD
    for det in data_dict.keys():
        psd_dict[det] = lal.LIGOIPsd # lalsim.SimNoisePSDaLIGOZeroDetHighPower   #lal.LIGOIPsd    # Preserves key equality in data_dict , psd_dict

else:
    analyticPSD_Q = False # For simplicity, using an analytic PSD
    detectors = data_dict.keys()
    df = data_dict[detectors[0]].deltaF
    fNyq = (len(data_dict[detectors[0]].data.data)/2)*df
    print(" == Loading numerical PSD ==")
    for det in detectors:
        print("Reading PSD for instrument %s from %s" % (det, opts.psd_file))

        # "Standard" PSD parsing code used on master.
        psd_dict[det] = lalsimutils.get_psd_series_from_xmldoc(opts.psd_file, det)  # pylal type!
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

        # Inverse PSD truncation (optional)
        if opts.psd_TruncateInverse:
            Tleft = opts.psd_TruncateInverseTime
            print("  Attempting to truncate inverse PSD, using target length ", Tleft)
            psd_dict[det] = truncate_inverse_psd(psd_dict[det],Tleft,opts.fmin_SNR,opts.fmax_SNR)
            print("  Confirm returned PSD is sane " , psd_dict[det].data.data[0], psd_dict[det].data.data[-1])
            
            print(" To use this PSD *precisely* will require changing fminSNR and fmaxSNR ")
            fminSNR=5
            print("   SNR integrand ", [fminSNR, fNyq])


        #ARCHIVAL: Original non-interpolated method. Seems to fail asymptomatically (memory management?) for some PSD files
        # DANGER: Assumes PSD file starts at 0 hz
        # psd_dict[det] = lalsimutils.pylal_psd_to_swig_psd(lalsimutils.get_psd_series_from_xmldoc(opts.psd_file, det))
        # psd_dict[det] = lalsimutils.regularize_swig_psd_series_near_nyquist(psd_dict[det], fNyq-opts.fmax_SNR) # zero out 80 hz window near nyquist
        # psd_dict[det] =  lalsimutils.enforce_swig_psd_fmin(psd_dict[det], fminSNR)           # enforce fmin at the psd level, HARD CUTOFF
        # tmp = psd_dict[det].data.data
        # print "Sanity check reporting : pre-extension, min is ", np.min(tmp[np.nonzero(tmp)]), " and maximum is ", np.max(psd_dict[det].data.data)
        # deltaF = data_dict[det].deltaF
        # # remember the PSD is one-sided, but h(f) is two-sided. The lengths are not equal.
        # psd_dict[det] = lalsimutils.extend_swig_psd_series_to_sampling_requirements(psd_dict[det], df, df*(len(data_dict[det].data.data)/2))
        # print "Post-extension the new PSD has 1/df = ", 1./psd_dict[det].deltaF, " (data 1/df = ", 1./deltaF, ") and length ", len(psd_dict[det].data.data)
        # tmp = psd_dict[det].data.data
        # nBad = np.argmin(tmp[np.nonzero(tmp)])
        # fBad = nBad*deltaF
        # print "Post-extension sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), "(at n=", np.argmin(tmp[np.nonzero(tmp)])," or f=", fBad, ")  and maximum is ", np.max(psd_dict[det].data.data)
        # print

# This code is a DUPLICATE TEST, used to initialize the peak log likelihood.
# I use this threshold to identify points for further investigation.
detectors = data_dict.keys()
if False: #not (opts.coinc) :
    print(" == Data report [USELESS IF NOISY DATA] == ")
    rho2Net = 0
    print(" Amplitude report *from data*:")
    for det in detectors:
        if analyticPSD_Q:
            IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det])
        else:
            IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det].data.data,analyticPSD_Q=False)
        rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
        rho2Net += rhoDet*rhoDet
        print(det, " rho = ", rhoDet)
    print("Network : ", np.sqrt(rho2Net))

if opts.plot_ShowH: 
    print(" == Plotting *raw* detector data data (time domain) via MANUAL INVERSE FFT  == ")
    print("     [This plot includes any windowing and data padding]   ")
    for det in detectors:
        revplan = lal.CreateReverseCOMPLEX16FFTPlan(len(data_dict[det].data.data), 0)
        ht = lal.CreateCOMPLEX16TimeSeries("ht", 
                lal.LIGOTimeGPS(0.), 0., Psig.deltaT, lal.lalDimensionlessUnit,
                len(data_dict[det].data.data))
        lal.COMPLEX16FreqTimeFFT(ht, data_dict[det], revplan)

        tvals = float(ht.epoch - theEpochFiducial) + ht.deltaT*np.arange(len(ht.data.data))  # Not correct timing relative to zero - FIXME
        plt.figure(1)
        plt.plot(tvals, np.abs(ht.data.data),label=det)
    plt.legend()
    plt.show()

    print(" == Plotting detector data (time domain; requires regeneration, MANUAL TIMESHIFTS,  and seperate code path! Argh!) == ")
    P = Psig.copy()
    P.tref = Psig.tref
    for det in detectors:
        P.detector=det 
        hT = lalsimutils.complex_hoft(P)   # Note detector time propagation is NOT performed by this function. This is NOT a fair comparison!
        tvals = float(P.tref - theEpochFiducial) + hT.deltaT*np.arange(len(hT.data.data))  # Not correct timing relative to zero - FIXME
        plt.figure(1)
        plt.plot(tvals, np.abs(hT.data.data),label=det)

    tlen = hT.deltaT*len(np.nonzero(np.abs(hT.data.data)))
#    tRef = np.abs(float(hT.epoch))
#    plt.xlim( tRef-0.5,tRef+0.1)  # Not well centered, based on epoch to identify physical merger time
#    plt.savefig("test_like_and_samp-input-hoft.pdf")
    plt.show()

# Load skymap, if present
if opts.opt_UseSkymap:
    print(" ==Loading skymap==")
    print("   skymap file ", opts.opt_UseSkymap)
    smap, smap_meta = bfits.read_sky_map(opts.opt_UseSkymap)
    sides = healpy.npix2nside(len(smap))

    if opts.plot_ShowSamplerInputs:
        try:
            from lalinference.bayestar import plot
            plt.subplot(111, projection='astro mollweide')
            plot.healpix_heatmap(smap)
            plt.show()
        except:
            print(" No skymap for you ")


#     import bisect

#     # WARNING: smap defined here and 'globally' stored
#     # FOR THE RECORD:http://healpix.sourceforge.net/pdf/intro.pdf for angle conversions
#     cum_smap = np.cumsum(smap)
#     def bayestar_cdf_inv(x,y):                 # it will be passed two arguments, use one
#         indx = bisect.bisect(cum_smap,x) 
#         th,ph = healpy.pix2ang(sides, indx)
#         if rosDebugMessagesDictionary["DebugMessagesLong"]:
#             print " skymap used x->(th,ph) :", x,th,ph
#         return ph,th-np.pi/2.

# #    bayestar_cdf_inv_vector = np.vectorize(bayestar_cdf_inv)
#     def bayestar_cdf_inv_vector(x,y):   # Manually vectorize, so I can insert pdb breaks
# #        pdb.set_trace()
#         indxVec = map(lambda z: bisect.bisect(cum_smap,z), x) 
#         th, ph = healpy.pix2ang(sides,indxVec)
#         return np.array([ph, th-np.pi/2.])                            # convert to RA, DEC from equatorial. Return must be pair of arrays

#     def bayestar_pdf_radec(x,y):               # look up value at targted pixel 
#         ra,dec = x,y
# #        pdb.set_trace()
#         indx = bplot._healpix_lookup(smap,  ra, dec) # note bplot takes lon (=ra, really), lat (dec). Not sure why
#         return smap[indx]


# Struct to hold template parameters
P = lalsimutils.ChooseWaveformParams(fmin=fminWavesTemplate, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         approx=approxTemplate,
         fref=fref,
         tref=theEpochFiducial,
         deltaT=1./fSample,
         deltaF=df)

if opts.template_mass1:
    P.m1 = lal.LAL_MSUN_SI*opts.template_mass1
if opts.template_mass2:
    P.m2 = lal.LAL_MSUN_SI*opts.template_mass2

timeWaveformTemplate = float(-lalsimutils.hoft(P).epoch)
timeWaveform             = float(-lalsimutils.hoft(Psig).epoch)
print(" Reminder: signal duration, template duration, PSD truncQ, psdTruncT ", timeWaveform, timeWaveformTemplate) #, opts.psd_TruncateInverse, opts.psd_TruncateInverseTime


#
# Perform the Precompute stage
#    - use a large tWindow : I want to retain the *entire* timeseries
#    - will it work if tWindowReference[1] is large
#
#tWindowReference[1] = Psig.deltaT*len(data_dict[detectors[0]].data.data)
#print tWindowReference[1]
print("  ---- > Problem : Master automatically clips the timeseries duration; use a *hardcoded* large window, for now, to retain as much as plausible ")
print("          Note: if the window is too large, you hit t_shift < 0 ")
det0 = detectors[0]
tWindowReference[1]= timeWaveformTemplate # data_dict[det0].data.epoch - 
rholms_intp, crossTerms, rholms = factored_likelihood.PrecomputeLikelihoodTerms(theEpochFiducial,tWindowReference[1], P, data_dict,psd_dict, Lmax, fmaxSNR, analyticPSD_Q)
print("Done with precompute")
print(detectors, rholms.keys())

# Plot h(t)
plt.figure(0)
for det in detectors:
    ht = DataInverseFourier(data_dict[det])  # this had better be windowed, if the input is!
    tvals = np.arange(ht.data.length)*ht.deltaT + float(ht.epoch - theEpochFiducial)
    plt.plot(tvals,np.real(ht.data.data),label='h(t):'+det)
plt.xlabel('t(s): 0 at ' +  lalsimutils.stringGPSNice(theEpochFiducial))
plt.ylabel('h (data)')
plt.legend()
plt.savefig("test-Q-response-ht.jpeg")

#
# Plot rho22(t)
#
plt.figure(1)
plt.clf()
for det in detectors:
    rho22 =rholms[det][(2,2)]
    tvals = np.arange(rho22.data.length)*rho22.deltaT + float(rho22.epoch -theEpochFiducial) # use time alignment
    plt.plot(tvals,np.abs(rho22.data.data),label='rho22:'+det)
plt.xlabel('t(s) [relative]:  0 at ' +  lalsimutils.stringGPSNice(theEpochFiducial))
plt.ylabel('Q22')
plt.legend()
plt.savefig("test-Q-response-rho22.jpeg")


plt.clf()
plt.figure(2)
rho22 =rholms[detDefault][(2,2)]
tvals = np.arange(rho22.data.length)*rho22.deltaT + float(rho22.epoch -theEpochFiducial) # use time alignment
plt.plot(tvals,np.log10(np.abs(rho22.data.data)),label='logrho22')
plt.xlabel('t(s)')
plt.ylabel('log Q22')
plt.legend()
plt.savefig("test-Q-response-log-rho22.jpeg")

#
# Plot lnLData(t). BE CAREFUL: it is rolled, make sure time identifications correct
#
plt.clf()
tStartOffsetDiscrete = tvals[0]
print(" Plotting lnLData from ", float(theEpochFiducial+tStartOffsetDiscrete), " onward ")
for det in detectors:
    nBinsDiscrete = len(data_dict[det].data.data)
    lnLDataDiscrete = factored_likelihood.DiscreteSingleDetectorLogLikelihoodData(theEpochFiducial,rholms, theEpochFiducial+tStartOffsetDiscrete, nBinsDiscrete, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, Lmax, det)
    plt.plot(tvals[:len(lnLDataDiscrete)],lnLDataDiscrete,label='lnLData:'+det)
plt.xlabel('t [fixme]  0 at ' +  lalsimutils.stringGPSNice(theEpochFiducial))
plt.ylabel('lnLData')
plt.savefig("test-Q-response-lnLData.jpeg")
plt.xlim(-0.5, 0.5)
plt.savefig("test-Q-response-lnLData-Zoom.jpeg")

#
# Plot lnL(t) around the event (interpolated, remember)
#
nptsMore = int(fSample*(tWindowExplore[1]-tWindowExplore[0]))
print(nptsMore)
tvals = np.linspace(tWindowExplore[0],tWindowExplore[1], nptsMore)
lnL = np.zeros(len(tvals))
for indx in np.arange(len(tvals)):
    # Make sure to populate the 'passed' parameters consistent with the injected value (sky location, etc)
    P.theta = Psig.theta
    P.phi   = Psig.phi
    P.dist   = Psig.dist
    P.incl   = Psig.incl
    P.psi    = Psig.psi  
    P.phiref = Psig.phiref
    P.tref =  theEpochFiducial+tvals[indx]
    lnL[indx] =  factored_likelihood.FactoredLogLikelihood(P, rholms_intp, crossTerms, Lmax)
plt.clf()
plt.plot(tvals,lnL,label='lnL')
plt.xlabel('t - tEvent (s): relative to '+lalsimutils.stringGPSNice(theEpochFiducial))
plt.ylabel('lnL')
plt.savefig("test-Q-response-lnL.jpg")

#
# Plot PSD
#
if opts.plot_ShowPSD:
    nPSD=0
    plt.clf()
    for det in psd_dict.keys():
        nPSD+=0  # Useful offset condition, if the PSDs are too closely spaced
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
        plt.plot(np.log10(fvals),np.log10(Sh)+nPSD,label="Sh:"+det)
        plt.xlabel('f (Hz)')
        plt.ylabel('Sh $Hz^{-1}$')
        plt.title('PSDs used')
    # Comparison plot: iLIGO
    Sh = map(lal.LIGOIPsd,fvals)
    plt.plot(np.log10(fvals),np.log10(Sh),label="Sh:iLIGO")
    Sh = map(lalsim.SimNoisePSDaLIGOZeroDetHighPower,fvals)
    plt.plot(np.log10(fvals),np.log10(Sh),label="Sh:aLIGO")
    plt.legend()
    plt.ylim(-50,-30)
    plt.xlim(0,3.5)
    plt.savefig("test-Q-response-psd.pdf")


TestDictionary = factored_likelihood_test.TestDictionaryDefault
TestDictionary["UVReport"]              =  analytic_signal  # this report is very confusing for real data
TestDictionary["QSquaredTimeseries"] = False # opts.plot_ShowLikelihoodVersusTime    # should be command-line option to control this plot specifically
TestDictionary["Rho22Timeseries"]      = True
TestDictionary["lnLModelAtKnown"]  =  analytic_signal  # this report is very confusing for real data
TestDictionary["lnLDataAtKnownPlusOptimalTimePhase"] = False
TestDictionary["lnLAtKnown"]           = True
TestDictionary["lnLAtKnownMarginalizeTime"]  = False
TestDictionary["AbsolutelyNoPlots"]   = not opts.plot_ShowLikelihoodVersusTime and not opts.plot_ShowSamplerInputs and not opts.plot_ShowSampler
TestDictionary["lnLDataPlot"]            = opts.plot_ShowLikelihoodVersusTime    # Plot individual geocentered L_k(t) and total L(t) [interpolated code]; plot discrete-data L_k(t)
TestDictionary["lnLDataPlotVersusPsi"] = opts.plot_ShowLikelihoodVersusTime
TestDictionary["lnLDataPlotVersusPhi"] = opts.plot_ShowLikelihoodVersusTime
TestDictionary["lnLDataPlotVersusPhiPsi"] = opts.plot_ShowLikelihoodVersusTime
print(" Detectors ", detectors)
factored_likelihood_test.TestLogLikelihoodInfrastructure(TestDictionary,theEpochFiducial,  data_dict, psd_dict,fmaxSNR, analyticPSD_Q, Psig, rholms,rholms_intp, crossTerms, detectors,Lmax)
