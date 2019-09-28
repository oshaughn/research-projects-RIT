from __future__ import print_function

import sys
from optparse import OptionParser

import numpy
from matplotlib import pylab as plt

from glue.lal import Cache
import lalsimutils

"""
test_like_and_samp.py:  Testing the likelihood evaluation and sampler, working in conjunction

"""

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>, R. O'Shaughnessy"

from factored_likelihood import *
import ourio

tStartEverything = lal.GPSTimeNow()

checkInputs = True
checkResults = True # Turn on to print/plot output; Turn off for testing speed
checkInputPlots = False
checkResultsPlots = True
checkResultsSlowChecks = False
rosUseRandomEventTime = True
rosUseDifferentWaveformLengths = False    # Very important test: lnL should be independent of the template length
rosUseRandomTemplateStartingFrequency = True
nMaxEvals = 1e6


rosUseTargetedDistance = True
rosUseStrongPriorOnParameters = False
rosDebugMessages = False
rosShowSamplerInputDistributions = True
rosShowRunningConvergencePlots = True
rosShowTerminalSampleHistograms = True
rosSaveHighLikelihoodPoints = True

fminSNR = 25
fSample = 4096*4

ifoName = "H1"
theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 


Niter = 5 # Number of times to call likelihood function
Tmax = 0.05 # max ref. time (prior)
Tmin = -0.05 # min ref. time (prior)
Dmax = 510. * 1.e6 * lal.LAL_PC_SI # max distance (prior)
Dmin = 1. * 1.e6 * lal.LAL_PC_SI     # min distance (prior)

#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
psd_dict = {}
rhoExpected ={}
rhoExpectedAlt ={}
analyticPSD_Q = True # For simplicity, using an analytic PSD

fminWavesSignal = 25
if rosUseDifferentWaveformLengths: 
    fminWavesTemplate = fminWavesSignal+0.005
else:
    if rosUseRandomTemplateStartingFrequency:
         print("   --- Generating a random template starting frequency  ---- ")
         fminWavesTemplate = fminWavesSignal+5.*np.random.random_sample()
    else:
        fminWavesTemplate = fminWavesSignal


distanceFiducial = 25.  # Make same as reference
psd_dict[ifoName] =  lalsim.SimNoisePSDiLIGOSRD
m1 = 4*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
tEventFiducial = 0.000 # 10./fSample
if rosUseRandomEventTime:
    print("   --- Generating a random event (barycenter) time  ---- ")
    tEventFiducial+= 0.01*(2*np.random.random_sample()-1.)
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100
Psig = ChooseWaveformParams(fmin = fminWavesSignal, radec=False, incl=0.0,phiref=0.0, theta=-np.pi/2, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         tref=theEpochFiducial+tEventFiducial,
         deltaT=1./fSample,
         dist=distanceFiducial*1.e6*lal.LAL_PC_SI)
tEventFiducialGPS = Psig.tref             # the 'trigger time' we will localize on
df = findDeltaF(Psig)
Psig.deltaF = df
Psig.print_params()
data_dict[ifoName] = lalsimutils.non_herm_hoff(Psig)
print("Timing spacing in data vs expected : ", df, data_dict[ifoName].deltaF)

print(" == Data report == ")
detectors = data_dict.keys()
rho2Net = 0
for det in detectors:
    IP = ComplexIP(fLow=fminSNR, fNyq=fSample/2.,deltaF=df,psd=psd_dict[det])
    IPOverlap = ComplexOverlap(fLow=fminSNR, fNyq=fSample/2.,deltaF=df,psd=psd_dict[det],analyticPSD_Q=True,full_output=True)  # Use for debugging later
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rhoExpectedAlt[det] = rhoDet2 = IPOverlap.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print(det, rhoDet, rhoDet2, " at epoch ", float(data_dict[det].epoch))
print("Network : ", np.sqrt(rho2Net))


print(" ======= Template specified: precomputing all quantities ==========")
# Struct to hold template parameters
# Fiducial distance provided but will not be used
Lmax = 2 # sets which modes to include
P = ChooseWaveformParams(fmin=fminWavesTemplate, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         tref=theEpochFiducial,
         deltaT=1./fSample,
         dist=100*1.e6*lal.LAL_PC_SI,
         deltaF=df) #ChooseWaveformParams(m1=m1,m2=m2,fmin = fminWaves, dist=100.*1.e6*lal.LAL_PC_SI, deltaF=df,ampO=ampO,fref=fref)
rholms_intp, crossTerms, rholms, epoch_post = PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict, psd_dict, Lmax, analyticPSD_Q)
tMiddle = lal.GPSTimeNow()

#
print("====Generating metadata from precomputed results =====")
distBoundGuess = estimateUpperDistanceBoundInMpc(rholms, crossTerms)
print(" distance probably less than ", distBoundGuess, " Mpc")

if checkInputs == True:



    print(" ======= UV test: Recover the SNR of the injection  ==========")
    print(" Detector lnLmodel  (-2lnLmodel)^(1/2)  rho(directly)  [last two entries should be equal!] ")
    for det in detectors:
        lnLModel = SingleDetectorLogLikelihoodModel(crossTerms, Psig.tref, Psig.phi, Psig.theta, Psig.incl, Psig.phiref, Psig.psi, Psig.dist, 2, det)
        print(det, lnLModel, np.sqrt(-2*lnLModel), rhoExpected[det], "      [last two equal?]")
    print(" ======= End to end LogL: Recover the SNR of the injection at the injection parameters  ==========")
    lnL = FactoredLogLikelihood(Psig, rholms_intp, crossTerms, Lmax)
    print("  : Evan's code : ", lnL, " versus rho^2/2 ", rho2Net/2)
    print("  : Timing issues (checkme!) : fiducial = ", stringGPSNice(theEpochFiducial))

    print(" ======= rholm test: Plot the lnLdata timeseries at the injection parameters (* STILL TIME OFFSET *)  ==========")
    tmin = np.max(float(epoch_post - theEpochFiducial),tWindowReference[0]+0.03)   # the minimum time used is set by the rolling condition
#    tvals = np.linspace(tmin,tWindowReference[1],4*fSample*(tWindowReference[1]-tmin))
    tvals = np.linspace(tWindowExplore[0]+tEventFiducial,tWindowExplore[1]+tEventFiducial,fSample*(tWindowExplore[1]-tWindowExplore[0]))
    for det in detectors:
        lnLData = map( lambda x: SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, theEpochFiducial+x, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, det), tvals)
        lnLDataEstimate = np.ones(len(tvals))*rhoExpected[det]*rhoExpected[det]
        plt.figure(1)
        tvalsPlot = tvals 
        plt.plot(tvalsPlot, lnLData,label='Ldata(t)+'+det)
        plt.plot(tvalsPlot, lnLDataEstimate,label="$rho^2("+det+")$")
    tEventRelative =float( Psig.tref - theEpochFiducial)
    print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is ", tEventRelative)
    plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
    plt.title("lnLdata (interpolated) vs narrow time interval")
    plt.xlabel('t(s)')
    plt.ylabel('lnLdata')

    print(" ======= rholm test: Plot the lnL timeseries at the injection parameters (* STILL TIME)  ==========")
    tmin = np.max(float(epoch_post - theEpochFiducial),tWindowReference[0]+0.03)   # the minimum time used is set by the rolling condition
    tvals = np.linspace(tmin,tWindowReference[1],4*fSample*(tWindowReference[1]-tmin))
    P = Psig.copy()
    lnL = np.zeros(len(tvals))
    for indx in np.arange(len(tvals)):
            P.tref =  theEpochFiducial+tvals[indx]
            lnL[indx] =  FactoredLogLikelihood(P, rholms_intp, crossTerms, 2)
    lnLEstimate = np.ones(len(tvals))*rho2Net/2
    plt.figure(1)
    tvalsPlot = tvals 
    plt.plot(tvalsPlot, lnL,label='lnL(t)')
    plt.plot(tvalsPlot, lnLEstimate,label="$rho^2/2(net)$")
    tEventRelative =float( Psig.tref - theEpochFiducial)
    print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is the same ", tEventRelative)
    plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
    plt.title("lnL (interpolated) vs narrow time interval")
    plt.xlim(Tmin,Tmax)  # the window we actually use
    plt.legend()
    plt.savefig("test_like_and_samp-lnLvsTime.pdf")


#
# Call the likelihood function for various extrinsic parameter values
#
nEvals = 0
def likelihood_function(phi, theta, tref, phiref, incl, psi, dist):
    global nEvals
    global sampler
    global rho2Net
    global priorPDF

    lnL = numpy.zeros(phi.shape)
    i = 0
    for ph, th, tr, phr, ic, ps, di in zip(phi, theta, tref, phiref, incl, psi, dist):
        P.phi = ph # right ascension
        P.theta = th # declination
        P.tref = theEpochFiducial + tr # ref. time (rel to epoch for data taking)
        P.phiref = phr # ref. orbital phase
        P.incl = ic # inclination
        P.psi = ps # polarization angle
        P.dist = di # luminosity distance
        
        lnL[i] = FactoredLogLikelihood(P, rholms_intp, crossTerms, Lmax)
        i+=1

    tEnd = lal.GPSTimeNow()
    return numpy.exp(lnL)

tMiddleAfterPlots = lal.GPSTimeNow()

import RIFT.integrators.mcsampler as mcsampler
sampler = mcsampler.MCSampler()

# Sampling distribution
def uniform_samp(a, b, x):
   if type(x) is float:
       return 1/(b-a)
   else:
       return numpy.ones(x.shape[0])/(b-a)

# set up bounds on parameters
# Polarization angle
psi_min, psi_max = 0, 2*numpy.pi
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


import functools
# Define the true prior PDF
# The likelihood function assumes this function exists.
# The integrator actually does a monte carlo integral of L*p using a sampling prior p_s.
def pdfFullPrior(phi, theta, tref, phiref, incl, psi, dist): # remember theta is dec, 
    return (np.cos(theta)/(4*np.pi)) * 1./(tWindowExplore[1]-tWindowExplore[0])*1/(3*Dmax**3)*np.cos(incl)/(4*np.pi)*1./(np.pi)


# Uniform sampling (in area) but nonuniform sampling in distance (*I hope*).  Auto-cdf inverse
if rosUseStrongPriorOnParameters:
    sampler.add_parameter("psi", functools.partial(mcsampler.gauss_samp, Psig.psi, 0.5), None, psi_min, psi_max, 
                          prior_pdf =mcsampler.uniform_samp_psi  )
    # Use few degree square prior : at SNR 20 in 3 detetors.  This is about 10 deg square
    sampler.add_parameter("ra", functools.partial(mcsampler.gauss_samp, Psig.phi,0.05), None, ra_min, ra_max, 
                          prior_pdf = mcsampler.uniform_samp_phase)
    sampler.add_parameter("dec", functools.partial(mcsampler.gauss_samp, Psig.theta,0.05), None, dec_min, dec_max, 
                          prior_pdf= mcsampler.uniform_samp_dec)
    sampler.add_parameter("tref", functools.partial(mcsampler.gauss_samp, tEventFiducial, 0.005), None, tref_min, tref_max, 
                          prior_pdf = functools.partial(mcsampler.uniform_samp_vector, tWindowExplore[0],tWindowExplore[1]))
    sampler.add_parameter("phi", functools.partial(mcsampler.gauss_samp, Psig.phiref,0.5), None, phi_min, phi_max, 
                          prior_pdf = mcsampler.uniform_samp_phase)
    sampler.add_parameter("incl", functools.partial(mcsampler.gauss_samp, Psig.incl,0.3), None, inc_min, inc_max, 
                          prior_pdf = mcsampler.uniform_samp_theta)
    if rosUseTargetedDistance:
        sampler.add_parameter("dist", functools.partial(mcsampler.quadratic_samp_vector,  distBoundGuess*1e6*lal.LAL_PC_SI ), None, dist_min, dist_max, prior_pdf = np.vectorize(lambda x: x**2/(3*Dmax**3)))
    else:
        sampler.add_parameter("dist", functools.partial(mcsampler.gauss_samp_withfloor, Psig.dist, Psig.dist*0.1, 0.001/Psig.dist), None, dist_min, dist_max,prior_pdf =np.vectorize( lambda x: x**2/(3*Dmax**3)))
else:
    # PROBLEM: Underlying prior samplers are not uniform.  We need two stages
    sampler.add_parameter("psi", functools.partial(mcsampler.uniform_samp_vector, psi_min, psi_max), None, psi_min, psi_max,
                          prior_pdf =mcsampler.uniform_samp_psi )
    # Single IFO, no info
    sampler.add_parameter("ra", functools.partial(mcsampler.uniform_samp_vector, ra_min, ra_max), None, ra_min, ra_max, 
                          prior_pdf = mcsampler.uniform_samp_phase)
    sampler.add_parameter("dec", functools.partial(mcsampler.dec_samp_vector, dec_min, dec_max), None, dec_min, dec_max, 
                          prior_pdf= mcsampler.uniform_samp_dec)
    sampler.add_parameter("tref", functools.partial(mcsampler.gauss_samp, tEventFiducial, 0.01), None, tref_min, tref_max, 
                          prior_pdf = functools.partial(mcsampler.uniform_samp_vector, tWindowExplore[0],tWindowExplore[1]))
    sampler.add_parameter("phi", functools.partial(mcsampler.uniform_samp_vector, phi_min, phi_max), None, phi_min, phi_max,
                          prior_pdf = mcsampler.uniform_samp_phase)
    sampler.add_parameter("incl", functools.partial(mcsampler.cos_samp_vector), None, inc_min, inc_max,
                          prior_pdf = mcsampler.uniform_samp_theta)
    if rosUseTargetedDistance:
        sampler.add_parameter("dist", functools.partial(mcsampler.quadratic_samp_vector,  distBoundGuess*1e6*lal.LAL_PC_SI ), None, dist_min, dist_max, prior_pdf = np.vectorize(lambda x: x**2/(3*Dmax**3)))
    else:
        sampler.add_parameter("dist", functools.partial(mcsampler.gauss_samp_withfloor, Psig.dist, Psig.dist*0.1, 0.001/Psig.dist), None, dist_min, dist_max,prior_pdf =np.vectorize( lambda x: x**2/(3*Dmax**3)))


if rosShowSamplerInputDistributions:
    print(" ====== Plotting prior and sampling distributions ==== ")
    print("  PROBLEM: Build in/hardcoded via uniform limits on each parameter! Need to add measure factors ")
    nFig = 0
    for param in sampler.params:
        nFig+=1
        plt.figure(nFig)
        plt.clf()
        xLow = sampler.llim[param]
        xHigh = sampler.rlim[param]
        xvals = np.linspace(xLow,xHigh,500)
        pdfPrior = sampler.prior_pdf[param]  # Force type conversion in case we have non-float limits for some reasona
        pdfvalsPrior = np.array(map(pdfPrior, xvals))  # do all the numpy operations by hand: no vectorization
        pdf = sampler.pdf[param]
        cdf = sampler.cdf[param]
        pdfvals = pdf(xvals)
        cdfvals = cdf(xvals)
        if str(param) ==  "dist":
            xvvals = xvals/(1e6*lal.LAL_PC_SI)       # plot in Mpc, not m.  Note PDF has to change
            pdfvalsPrior = pdfvalsPrior * (1e6*lal.LAL_PC_SI) # rescale units
            pdfvals = pdfvals * (1e6*lal.LAL_PC_SI) # rescale units
        plt.plot(xvals,pdfvalsPrior,label="prior:"+str(param),linestyle='--')
        plt.plot(xvals,pdfvals,label=str(param))
        plt.plot(xvals,cdfvals,label='cdf:'+str(param))
        plt.xlabel(str(param))
        plt.legend()
        plt.savefig("test_like_and_samp-"+str(param)+".pdf")
#    plt.show()

tGPSStart = lal.GPSTimeNow()
res, var, ret, lnLmarg, neff = sampler.integrate(likelihood_function, "ra", "dec", "tref", "phi", "incl", "psi", "dist", n=200,nmax=nMaxEvals,igrandmax=rho2Net/2,full_output=True,neff=100,igrand_threshold_fraction=0.95)
tGPSEnd = lal.GPSTimeNow()
print(" Evaluation time  = ", float(tGPSEnd - tGPSStart), " seconds")
print(" lnLmarg is ", np.log(res), " with expected relative error ", np.sqrt(var)/res)
print(" expected largest value is ", rho2Net/2, end=' ')
print(" note neff is ", neff, "; compare neff^(-1/2) = ", 1/np.sqrt(neff))

# Save the sampled points to a file
# Only store some
ourio.dumpSamplesToFile("test_like_and_samp_singleifo-dump.dat", ret, ['ra','dec', 'tref', 'phi', 'incl', 'psi', 'dist', 'lnL'])

if checkInputs:
    plt.show()
# Plot terminal histograms from the sampled points and log likelihoods
if rosShowTerminalSampleHistograms:
    ra,dec,tref,phi,incl, psi,dist,lnL = np.transpose(ret)  # unpack. This can include all or some of the data set. The default configuration returns *all* points
    print(" ==== CONVERGENCE PLOTS (**beware: potentially truncated data!**) === ")
    plt.figure(0)
    plt.clf()
    plt.plot(np.arange(len(lnLmarg)), lnLmarg,label="lnLmarg")
    plt.plot(np.arange(len(lnL)), np.ones(len(lnL))*rho2Net/2,label="rho^2/2")
    plt.xlim(0,len(lnL))
    plt.xlabel('iteration')
    plt.ylabel('lnL')
    plt.legend()
    print(" ==== TERMINAL 1D HISTOGRAMS: Sampling and posterior === ")
    plt.figure(1)
    plt.clf()
    hist, bins  = np.histogram(dist/(1e6*lal.LAL_PC_SI),bins=50,density=True)
    center = (bins[:-1]+bins[1:])/2
    plt.plot(center,hist,label="dist:sampled")
    hist,bins = np.histogram(dist/(1e6*lal.LAL_PC_SI),bins=50,weights=np.exp(lnL),density=True)
    center = (bins[:-1]+bins[1:])/2
    plt.plot(center,hist,label="dist:post")
    plt.xlabel("d (Mpc)")
    plt.title("Sampling and posterior distribution: d ")
    plt.legend()
    plt.figure(2)
    plt.clf()
    hist, bins  = np.histogram(tref,bins=50,density=True)
    center = (bins[:-1]+bins[1:])/2
    plt.plot(center,hist,label="tref:sampled")
    hist, bins  = np.histogram(tref,bins=50,density=True,weights=np.exp(lnL))
    center = (bins[:-1]+bins[1:])/2
    plt.plot(center,hist,label="tref:post")
    plt.xlim(-0.01+tEventFiducial,0.01+tEventFiducial)
    plt.xlabel("t (s)")
    plt.title("Sampling and posterior distribution: t ")
    plt.legend()
    plt.figure(3)
    plt.clf()
    hist, bins  = np.histogram(incl,bins=50,normed=True)
    center = (bins[:-1]+bins[1:])/2
    plt.plot(center,hist,label="incl:sampled")
    hist, bins  = np.histogram(incl,bins=50,normed=True,weights=np.exp(lnL))
    center = (bins[:-1]+bins[1:])/2
    plt.plot(center,hist,label="incl:post")
    plt.xlabel("incl")
    plt.title("Sampling and posterior distribution: incl ")
    plt.legend()
    plt.figure(4)
    plt.clf()
    hist, bins  = np.histogram(psi,bins=50,normed=True)
    center = (bins[:-1]+bins[1:])/2
    plt.plot(center,hist,label="psi:sampled")
    hist, bins  = np.histogram(psi,bins=50,normed=True,weights=np.exp(lnL))
    center = (bins[:-1]+bins[1:])/2
    plt.plot(center,hist,label="psi:post")
    plt.xlabel("psi")
    plt.title("Sampling and posterior distribution: psi ")
    plt.legend()
    plt.show()
    print(" ==== TERMINAL 2D HISTOGRAMS: Sampling and posterior === ")
        # Distance-inclination
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
    plt.ylim(0,np.pi)
    plt.colorbar()
    plt.title("Posterior distribution: ra-dec ")
    plt.show()

tMiddleBeforePlots2 = lal.GPSTimeNow()

tMiddleAfterPlots2 = lal.GPSTimeNow()

