import sys
from optparse import OptionParser

import numpy as np
from matplotlib import pylab as plt

from glue.lal import Cache
import lal
import lalsimulation as lalsim
import lalsimutils


"""
test_like_and_samp.py:  Testing the likelihood evaluation and sampler, working in conjunction

"""

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
nMaxEvals = 1e4

if rosUseThresholdForReturn and nMaxEvals > 5e4:  # I usually only want to restrict the return set when I am testing
    fracThreshold = opts.points_threshold_match
else:
    fracThreshold = 0.0

theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 
tEventFiducial = 0                                                                 # relative to GPS reference

# optp = OptionParser()
# optp.add_option("-c", "--cache-file", default=None, help="LIGO cache file containing all data needed.")
# optp.add_option("-C", "--channel-name", action="append", help="instrument=channel-name, e.g. H1=FAKE-STRAIN. Can be given multiple times for different instruments.")
# opts, args = optp.parse_args()

det_dict = {}
rhoExpected ={}
if opts.channel_name is not None and opts.cache_file is None:
    print >>sys.stderr, "Cache file required when requesting channel data."	
    exit(-1)
elif opts.channel_name is not None:
    det_dict = dict(map(lambda cname: cname.split("="), opts.channel_name))


ampO =opts.amporder # sets which modes to include in the physical signal
Lmax = opts.Lmax # sets which modes to include
fref = opts.fref
fminWavesSignal = opts.fmin_Template  # too long can be a memory and time hog, particularly at 16 kHz
fSample = opts.srate

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
if len(det_dict) > 0:
    for d, chan in det_dict.iteritems():
        data_dict[d] = lalsimutils.frame_data_to_hoff(opts.cache_file, chan)
else:
    m1 = 4*lal.LAL_MSUN_SI
    m2 = 3*lal.LAL_MSUN_SI

    Psig = lalsimutils.ChooseWaveformParams(
        m1 = m1,m2 =m2,
        fmin = fminWavesSignal, 
        fref=fref, ampO=ampO,
                                radec=True, theta=1.2, phi=2.4,
                                detector='H1', 
                                dist=25.*1.e6*lal.LAL_PC_SI,
                                tref = theEpochFiducial,
        deltaT=1./fSample
                                )
    df = lalsimutils.findDeltaF(Psig)
    Psig.print_params()
    Psig.deltaF = df
    data_dict['H1'] = factored_likelihood.non_herm_hoff(Psig)
    Psig.detector = 'L1'
    data_dict['L1'] = factored_likelihood.non_herm_hoff(Psig)
    Psig.detector = 'V1'
    data_dict['V1'] = factored_likelihood.non_herm_hoff(Psig)

# TODO: Read PSD from XML
psd_dict = {}
analyticPSD_Q = True # For simplicity, using an analytic PSD
psd_dict['H1'] = lal.LIGOIPsd
psd_dict['L1'] = lal.LIGOIPsd
psd_dict['V1'] = lal.LIGOIPsd

print " == Data report == "
detectors = data_dict.keys()
rho2Net = 0
print  " Amplitude report :"
fminSNR =opts.fmin_SNR
for det in detectors:
    IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=Psig.deltaF,psd=psd_dict[det])
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
TestDictionary["QReflection"]          = True
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
