import sys
from optparse import OptionParser

import numpy

from glue.lal import Cache
import lalsimutils

"""
test_like_and_samp.py:  Testing the likelihood evaluation and sampler, working in conjunction

"""

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>, Chris Pankow <pankow@gravity.phys.uwm.edu>, R. O'Shaughnessy"

from factored_likelihood import *

checkResults = True # Turn on to print/plot output; Turn off for testing speed
checkInputPlots = False
checkResultsPlots = True
checkResultsSlowChecks = False
rosUseRandomEventTime = True
rosUseDifferentWaveformLengths = False    # Very important test: lnL should be independent of the template length
rosUseRandomTemplateStartingFrequency = True


fminSNR = 25
fSample = 4096*4

ifoName = "H1"
theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 


Niter = 5 # Number of times to call likelihood function
Tmax = 0.01 # max ref. time
Tmin = -0.01 # min ref. time
Dmax = 110. * 1.e6 * lal.LAL_PC_SI # max ref. time
Dmin = 1. * 1.e6 * lal.LAL_PC_SI   # min ref. time

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
         print "   --- Generating a random template starting frequency  ---- " 
         fminWavesTemplate = fminWavesSignal+5.*np.random.random_sample()
    else:
        fminWavesTemplate = fminWavesSignal





distanceFiducial = 25.  # Make same as reference
psd_dict[ifoName] =  lalsim.SimNoisePSDiLIGOSRD
m1 = 4*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
tEventFiducial = 0.000 # 10./fSample
if rosUseRandomEventTime:
    print "   --- Generating a random event (barycenter) time  ---- " 
    tEventFiducial+= 0.05*np.random.random_sample()
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100
Psig = ChooseWaveformParams(fmin = fminWavesSignal, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
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
data_dict[ifoName] = non_herm_hoff(Psig)
print "Timing spacing in data vs expected : ", df, data_dict[ifoName].deltaF

print " == Data report == "
detectors = data_dict.keys()
rho2Net = 0
for det in detectors:
    IP = ComplexIP(fLow=fminSNR, fNyq=fSample/2.,deltaF=df,psd=psd_dict[det])
    IPOverlap = ComplexOverlap(fLow=fminSNR, fNyq=fSample/2.,deltaF=df,psd=psd_dict[det],analyticPSD_Q=True,full_output=True)  # Use for debugging later
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rhoExpectedAlt[det] = rhoDet2 = IPOverlap.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print det, rhoDet, rhoDet2, " at epoch ", float(data_dict[det].epoch)
print "Network : ", np.sqrt(rho2Net)


print " ======= Template specified: precomputing all quantities =========="
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

#
# Perform the Precompute stage
#
rholms_intp, crossTerms, rholms, epoch_post = PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict,psd_dict, Lmax, analyticPSD_Q)
print "Finished Precomputation..."


#
# Call the likelihood function for various extrinsic parameter values
#
nEvals = 0
def likelihood_function(phi, theta, tref, phiref, incl, psi, dist):
    global nEvals
    lnL = numpy.zeros(phi.shape)
    i = 0
    print " Likelihood results :  "
    print "  lnL   sqrt(2max(lnL))  sqrt(2 lnLmarg)   <lnL> "
    for ph, th, tr, phr, ic, ps, di in zip(phi, theta, tref, phiref, incl, psi, dist):
        P.phi = ph # right ascension
        P.theta = th # declination
        P.tref = theEpochFiducial + tr # ref. time (rel to epoch for data taking)
        P.phiref = phr # ref. orbital phase
        P.incl = ic # inclination
        P.psi = ps # polarization angle
        P.dist = di # luminosity distance

        lnL[i] = FactoredLogLikelihood(theEpochFiducial,P, rholms_intp, crossTerms, Lmax)
        if (numpy.mod(i,100)==0):
#            print "Evaluation # ", i, " : For (RA, DEC, tref, phiref, incl, psi, dist) ="
#            print "\t", P.phi, P.theta, float(P.tref-theEpochFiducial), P.phiref, P.incl, P.psi, P.dist/(1e6*lal.LAL_PC_SI)
            logLmarg =np.log(np.mean(np.exp(lnL[:i])))
#            print "\tlog likelihood is ",  lnL[i], ";   log-integral L_{marg} =", logLmarg , " with sqrt(2Lmarg)= ", np.sqrt(2*logLmarg), "; and  <lnL>=  ", np.mean(lnL[:i])
            print   lnL[i], np.sqrt(np.max(lnL[:i])), logLmarg , np.sqrt(2*logLmarg), np.mean(lnL[:i])
        i+=1
    return numpy.exp(lnL)

import mcsampler
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
inc_min, inc_max = -numpy.pi/2, numpy.pi/2
# orbital phi
phi_min, phi_max = 0, 2*numpy.pi
# distance
dist_min, dist_max = Dmin, Dmax

import functools
# Uniform sampling, auto-cdf inverse
sampler.add_parameter("psi", functools.partial(uniform_samp, psi_min, psi_max), None, psi_min, psi_max)
sampler.add_parameter("ra", functools.partial(uniform_samp, ra_min, ra_max), None, ra_min, ra_max)
sampler.add_parameter("dec", functools.partial(uniform_samp, dec_min, dec_max), None, dec_min, dec_max)
sampler.add_parameter("tref", functools.partial(uniform_samp, tref_min, tref_max), None, tref_min, tref_max)
sampler.add_parameter("phi", functools.partial(uniform_samp, phi_min, phi_max), None, phi_min, phi_max)
sampler.add_parameter("inc", functools.partial(uniform_samp, inc_min, inc_max), None, inc_min, inc_max)
sampler.add_parameter("dist", functools.partial(uniform_samp, dist_min, dist_max), None, dist_min, dist_max)

res, var = sampler.integrate(likelihood_function, 1e6, "ra", "dec", "tref", "phi", "inc", "psi", "dist")
print res, numpy.sqrt(var)
