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
        python test_like_and_samp.py  --show-sampler-inputs --show-sampler-results --LikelihoodType_MargPhi
        python test_like_and_samp.py  --show-sampler-inputs --show-sampler-results --LikelihoodType_MargT  \
                 --Niter 10 --Nskip 1       # VERY SLOW
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

"""
import sys

import numpy as np

from glue.lal import Cache
from glue.ligolw import utils, lsctables, table

import lal
import lalsimulation as lalsim
import lalsimutils

try:
    from matplotlib import pylab as plt
except:
    print "- no matplotlib -"

try:
    import healpy
    from lalinference.bayestar import fits as bfits
    from lalinference.bayestar import plot as bplot
except:
    print " -no skymaps - "



__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>, R. O'Shaughnessy"

import factored_likelihood
import factored_likelihood_test
import ourio
import ourparams

opts, rosDebugMessagesDictionary = ourparams.ParseStandardArguments()
print opts

checkInputs = False

rosUseDifferentWaveformLengths = False    
rosUseRandomTemplateStartingFrequency = False

rosDebugMessages = opts.verbose
rosShowSamplerInputDistributions = opts.plot_ShowSamplerInputs
rosShowRunningConvergencePlots = True
rosShowTerminalSampleHistograms = opts.plot_ShowSampler
rosSaveHighLikelihoodPoints = True
rosUseThresholdForReturn = False
rosUseMultiprocessing= False
rosDebugCheckPriorIntegral = False

nMaxEvals = int(opts.nmax)
print " Running at most ", nMaxEvals, " iterations"

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

if opts.channel_name is not None and opts.cache_file is None:
    print >>sys.stderr, "Cache file required when requesting channel data."	
    exit(-1)
elif opts.channel_name is not None:
    det_dict = dict(map(lambda cname: cname.split("="), opts.channel_name))



if rosUseDifferentWaveformLengths: 
    fminWavesTemplate = fminWavesSignal+0.005
else:
    if rosUseRandomTemplateStartingFrequency:
         print "   --- Generating a random template starting frequency  ---- " 
         fminWavesTemplate = fminWavesSignal+5.*np.random.random_sample()
    else:
        fminWavesTemplate = fminWavesSignal

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

# TRY TO READ IN DATA: if data specified, use it and construct the detector list from it. Otherwise...
if opts.channel_name and    (opts.opt_ReadWholeFrameFilesInCache):
    for inst, chan in map(lambda c: c.split("="), opts.channel_name):
        print "Reading channel %s from cache %s" % (inst+":"+chan, opts.cache_file)
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan)
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
        data_dict[inst] = lalsimutils.frame_data_to_non_herm_hoff(opts.cache_file, inst+":"+chan, start=int(event_time)-start_pad, stop=int(event_time)+end_pad)
        fSample = len(data_dict[inst].data.data)*data_dict[inst].deltaF
        df = data_dict[inst].deltaF
        if Psig:
            Psig.deltaF =df
        print "Frequency binning: %f, length %d" % (data_dict[inst].deltaF, len(data_dict[inst].data.data))
        print "Sampling rate ", fSample

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
        timeSegmentLength  = -float(lalsimutils.hoft(Psig).epoch)
        if timeSegmentLength > opts.seglen:
            print " +++ CATASTROPHE : You are requesting less data than your template target needs!  +++"
            print "    Requested data size: ", opts.seglen
            print "    Template duration  : ", timeSegmentLength
            sys.exit(0)

    df = lalsimutils.findDeltaF(Psig)
    Psig.print_params()
    Psig.deltaF = df
    data_dict['H1'] = factored_likelihood.non_herm_hoff(Psig)
    Psig.detector = 'L1'
    data_dict['L1'] = factored_likelihood.non_herm_hoff(Psig)
    Psig.detector = 'V1'
    data_dict['V1'] = factored_likelihood.non_herm_hoff(Psig)

# Reset origin of time, if required
if opts.force_gps_time:
    print " +++ USER HAS OVERRIDDEN FIDUCIAL EPOCH +++ "
    print "  The zero of time (and the region to be windowed) will be changed; you had better know what you are doing.  "
    print "    original " ,theEpochFiducial
    print "    new      ", opts.force_gps_time
    theEpochFiducial = lal.GPSTimeNow()
    theEpochFiducial.gpsSeconds = int(opts.force_gps_time)
    theEpochFiducial.gpsNanoSeconds =  int(1e9*(opts.force_gps_time - int(opts.force_gps_time)))

# TODO: Read PSD from XML
psd_dict = {}
if not(opts.psd_file):
    analyticPSD_Q = True # For simplicity, using an analytic PSD
    for det in data_dict.keys():
#        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower   #lal.LIGOIPsd    # Preserves key equality in data_dict , psd_dict
#        psd_dict[det] = lal.LIGOIPsd    # Preserves key equality in data_dict , psd_dict
        psd_dict[det] = lalsim.SimNoisePSDiLIGOSRD    # Preserves key equality in data_dict , psd_dict.  this is Chris' 'iLIGO' PSD, for test data

else:
    analyticPSD_Q = False # For simplicity, using an analytic PSD
    detectors = data_dict.keys()
    df = data_dict[detectors[0]].deltaF
    for det in detectors:
        print "Reading PSD for instrument %s from %s" % (det, opts.psd_file)
        psd_dict[det] = lalsimutils.pylal_psd_to_swig_psd(lalsimutils.get_psd_series_from_xmldoc(opts.psd_file, det))
        psd_dict[det] = lalsimutils.regularize_swig_psd_series_near_nyquist(psd_dict[det], 80) # zero out 80 hz window near nyquist
        psd_dict[det] =  lalsimutils.enforce_swig_psd_fmin(psd_dict[det], fminSNR)           # enforce fmin at the psd level, HARD CUTOFF
        tmp = psd_dict[det].data.data
        print "Sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), " and maximum is ", np.max(psd_dict[det].data.data)
        deltaF = data_dict[det].deltaF
        # remember the PSD is one-sided, but h(f) is two-sided. The lengths are not equal.
        psd_dict[det] = lalsimutils.extend_swig_psd_series_to_sampling_requirements(psd_dict[det], df, df*(len(data_dict[det].data.data)/2))
        print "Post-extension the new PSD has 1/df = ", 1./psd_dict[det].deltaF, " (data 1/df = ", 1./deltaF, ") and length ", len(psd_dict[det].data.data)
        tmp = psd_dict[det].data.data
        print "Post-extension sanity check reporting  : min is ", np.min(tmp[np.nonzero(tmp)]), " and maximum is ", np.max(psd_dict[det].data.data)

# This code is a DUPLICATE TEST, used to initialize the peak log likelihood.
# I use this threshold to identify points for further investigation.
print " == Data report == "
detectors = data_dict.keys()
rho2Net = 0
print  " Amplitude report :"
fminSNR =opts.fmin_SNR
for det in detectors:
    if analyticPSD_Q:
        IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det])
    else:
        IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det].data.data,analyticPSD_Q=False)
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print det, " rho = ", rhoDet
print "Network : ", np.sqrt(rho2Net)

if checkInputs:
    print " == Plotting detector data (time domain; requires regeneration, MANUAL TIMESHIFTS,  and seperate code path! Argh!) == "
    P = Psig.copy()
    P.tref = Psig.tref
    for det in detectors:
        P.detector=det   # we do 
        hT = hoft(P)
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

    if opts.plot_ShowSamplerInputs:
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
         fref=fref,
         tref=theEpochFiducial,
         deltaT=1./fSample,
         dist=100*1.e6*lal.LAL_PC_SI,
         deltaF=df)


#
# Perform the Precompute stage
#
rholms_intp, crossTerms, rholms, epoch_post = factored_likelihood.PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict,psd_dict, Lmax, analyticPSD_Q)
print "Finished Precomputation..."
print "====Generating metadata from precomputed results ====="
distBoundGuess = factored_likelihood.estimateUpperDistanceBoundInMpc(rholms, crossTerms)
print " distance probably less than ", distBoundGuess, " Mpc"

print "====Loading metadata from previous runs (if any): sampler-seed-data.dat ====="



TestDictionary = factored_likelihood_test.TestDictionaryDefault
TestDictionary["DataReport"]             = True
TestDictionary["DataReportTime"]       = True
TestDictionary["UVReport"]              = True
TestDictionary["UVReflection"]          = True
TestDictionary["QReflection"]          = False
TestDictionary["lnLModelAtKnown"]  = True
TestDictionary["lnLDataAtKnownPlusOptimalTimePhase"] = False
TestDictionary["lnLAtKnown"]           = True
TestDictionary["lnLAtKnownMarginalizeTime"]  = False
TestDictionary["lnLDataPlot"]            = True

#opts.fmin_SNR=40

factored_likelihood_test.TestLogLikelihoodInfrastructure(TestDictionary,theEpochFiducial, epoch_post, data_dict, psd_dict, analyticPSD_Q, Psig, rholms,rholms_intp, crossTerms, detectors,Lmax)



#
# Call the likelihood function for various extrinsic parameter values
#
nEvals = 0
def likelihood_function(phi, theta, tref, phiref, incl, psi, dist):
    global nEvals
    global pdfFullPrior

    lnL = np.zeros(phi.shape)
    i = 0
    for ph, th, tr, phr, ic, ps, di in zip(phi, theta, tref, phiref, incl, psi, dist):
        P.phi = ph # right ascension
        P.theta = th # declination
        P.tref = theEpochFiducial + tr # ref. time (rel to epoch for data taking)
        P.phiref = phr # ref. orbital phase
        P.incl = ic # inclination
        P.psi = ps # polarization angle
        P.dist = di*lal.LAL_PC_SI # luminosity distance.  The sampler assumes Mpc; P requires SI

        lnL[i] = factored_likelihood.FactoredLogLikelihood(theEpochFiducial,P, rholms_intp, crossTerms, Lmax)#+ np.log(pdfFullPrior(ph, th, tr, ps, ic, ps, di))
        i+=1


    nEvals+=i 
    return np.exp(lnL)

import mcsampler
sampler = mcsampler.MCSampler()

# Populate sampler 
ourparams.PopulateSamplerParameters(sampler, theEpochFiducial,tEventFiducial, distBoundGuess, Psig, opts)

if opts.plot_ShowPSD:
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



if rosShowSamplerInputDistributions:
    print " ====== Plotting prior and sampling distributions ==== "
    print "  PROBLEM: Build in/hardcoded via uniform limits on each parameter! Need to add measure factors "
    nFig = 0
    for param in sampler.params:
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
        plt.savefig("test_like_and_samp-"+str(param)+".pdf")


if  rosShowSamplerInputDistributions or opts.plot_ShowPSD:  # minimize number of pauses
    plt.show()



tGPSStart = lal.GPSTimeNow()
res, var, ret, lnLmarg, neff = sampler.integrate(likelihood_function, "ra", "dec", "tref", "phi", "incl", "psi", "dist", n=opts.nskip,nmax=opts.nmax,igrandmax=rho2Net/2,full_output=True,neff=opts.neff,igrand_threshold_fraction=fracThreshold,use_multiprocessing=rosUseMultiprocessing,verbose=True,extremely_verbose=opts.super_verbose)
tGPSEnd = lal.GPSTimeNow()
print " Evaluation time  = ", float(tGPSEnd - tGPSStart), " seconds"
print " lnLmarg is ", np.log(res), " with nominal relative sampling error ", np.sqrt(var)/res, " but a more reasonable estimate based on the lnL history is ", np.std(lnLmarg - np.log(res))
print " expected largest value is ", rho2Net/2, "and observed largest lnL is ", np.max(np.transpose(ret)[-1])
print " note neff is ", neff, "; compare neff^(-1/2) = ", 1/np.sqrt(neff)

# Save the sampled points to a file
# Only store some
ourio.dumpSamplesToFile("test_like_and_samp-dump.dat", ret, ['ra','dec', 'tref', 'phi', 'incl', 'psi', 'dist', 'lnL'])
np.savetxt('test_like_and_samp-result.dat', [res])
np.savetxt('test_like_and_samp-dump-lnLmarg.dat',lnLmarg)

if checkInputs:
    plt.show()


# Plot terminal histograms from the sampled points and log likelihoods
if rosShowTerminalSampleHistograms:
    print " ==== CONVERGENCE PLOTS (**beware: potentially truncated data!**) === "
    plt.figure(99)
    plt.clf()
    lnL = np.transpose(ret)[-1]
    plt.plot(np.arange(len(lnLmarg)), lnLmarg,label="lnLmarg")
    nExtend = np.max([len(lnL),len(lnLmarg)])
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
