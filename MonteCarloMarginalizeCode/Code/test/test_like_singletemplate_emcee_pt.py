"""test_like_singletemplate_emcee.py
   a test to see how to glue emcee to our likelihood evaluation.
"""

from __future__ import print_function

import sys
from optparse import OptionParser

import numpy
from matplotlib import pylab as plt

from glue.lal import Cache
import lalsimutils

from factored_likelihood import *
from ourio import *
import emcee

checkInputs = False

rosUseDifferentWaveformLengths = False    
rosUseRandomTemplateStartingFrequency = False

rosUseTargetedDistance = True
rosUseStrongPriorOnParameters = True
rosDebugMessages = True
rosShowSamplerInputDistributions = True
rosShowRunningConvergencePlots = True
rosShowTerminalSampleHistograms = True
rosSaveHighLikelihoodPoints = True

theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 
tEventFiducial = 0                                                                 # relative to GPS reference

optp = OptionParser()
optp.add_option("-c", "--cache-file", default=None, help="LIGO cache file containing all data needed.")
optp.add_option("-C", "--channel-name", action="append", help="instrument=channel-name, e.g. H1=FAKE-STRAIN. Can be given multiple times for different instruments.")
opts, args = optp.parse_args()

det_dict = {}
rhoExpected ={}
if opts.channel_name is not None and opts.cache_file is None:
    print("Cache file required when requesting channel data.", file=sys.stderr)
    exit(-1)
elif opts.channel_name is not None:
    det_dict = dict(map(lambda cname: cname.split("="), opts.channel_name))

Niter = 5 # Number of times to call likelihood function
Tmax = 0.05 # max ref. time
Tmin = -0.05 # min ref. time
Dmax = 110. * 1.e6 * lal.LAL_PC_SI # max distance
Dmin = 1. * 1.e6 * lal.LAL_PC_SI   # min distance

ampO =0 # sets which modes to include in the physical signal
Lmax = 2 # sets which modes to include
fref = 100
fminWavesSignal = 25  # too long can be a memory and time hog, particularly at 16 kHz
fSample = 4096*4

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
if len(det_dict) > 0:
    for d, chan in det_dict.items():
        data_dict[d] = lalsimutils.frame_data_to_hoff(opts.cache_file, chan)
else:
    m1 = 4*lal.LAL_MSUN_SI
    m2 = 3*lal.LAL_MSUN_SI

    Psig = ChooseWaveformParams(
        m1 = m1,m2 =m2,
        fmin = fminWavesSignal, 
        fref=fref, ampO=ampO,
                                radec=True, theta=1.2, phi=2.4,
                                detector='H1', 
                                dist=25.*1.e6*lal.LAL_PC_SI,
                                tref = theEpochFiducial,
        deltaT=1./fSample
                                )
    df = findDeltaF(Psig)
    Psig.print_params()
    Psig.deltaF = df
    data_dict['H1'] = lalsimutils.non_herm_hoff(Psig)
    Psig.detector = 'L1'
    data_dict['L1'] = lalsimutils.non_herm_hoff(Psig)
    Psig.detector = 'V1'
    data_dict['V1'] = lalsimutils.non_herm_hoff(Psig)

# TODO: Read PSD from XML
psd_dict = {}
analyticPSD_Q = True # For simplicity, using an analytic PSD
psd_dict['H1'] = lal.LIGOIPsd
psd_dict['L1'] = lal.LIGOIPsd
psd_dict['V1'] = lal.LIGOIPsd

print(" == Data report == ")
detectors = data_dict.keys()
rho2Net = 0
print(" Amplitude report :")
fminSNR =30
for det in detectors:
    IP = ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=Psig.deltaF,psd=psd_dict[det])
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print(det, " rho = ", rhoDet)
print("Network : ", np.sqrt(rho2Net))

if checkInputs:
    print(" == Plotting detector data (time domain; requires regeneration, MANUAL TIMESHIFTS,  and seperate code path! Argh!) == ")
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
P = ChooseWaveformParams(fmin=fminWavesTemplate, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
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
rholms_intp, crossTerms, rholms, epoch_post = PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict,psd_dict, Lmax, analyticPSD_Q)
print("Finished Precomputation...")
print("====Generating metadata from precomputed results =====")
distBoundGuess = estimateUpperDistanceBoundInMpc(rholms, crossTerms)
print(" distance probably less than ", distBoundGuess, " Mpc")

nEvals = 0
def likelihood_function( x):
    global nEvals
    global rholms_intp
    global crossTerms
    global Lmax
    phi, theta, tref, phiref, incl, psi, dist = x

    # Eliminate  implausible values
    # These don't count as evaluations
    if np.abs(tref)>tWindowExplore[1] or dist < 0 or dist > distBoundGuess or psi<0 or psi>np.pi or phiref<0 or phiref>2*np.pi or phi<0 or phi>2*np.pi:
        return -np.inf

    P.phi = phi # right ascension
    P.theta = theta # declination
    P.tref = theEpochFiducial + tref # ref. time (rel to epoch for data taking)
    P.phiref = phiref # ref. orbital phase
    P.incl = incl # inclination
    P.psi = psi # polarization angle
    P.dist = dist*(1e6*lal.LAL_PC_SI) # luminosity distance

    lnL = FactoredLogLikelihood(P, rholms_intp, crossTerms, Lmax)
    if rosDebugMessages:
        if (numpy.mod(nEvals,400)==10 and nEvals>100):
            print("\t Params ", nEvals, " (RA, DEC, tref, phiref, incl, psi, dist) =")
            print("\t", nEvals, P.phi, P.theta, float(P.tref-theEpochFiducial), P.phiref, P.incl, P.psi, P.dist/(1e6*lal.LAL_PC_SI), lnL)

    nEvals+=1
    return np.exp(lnL)




## ALTERNATIVE: Just in case we are not being very efficient,let's compare an MCMC.
# emcee
ndim = 7
nwalkers = 100
ntemps = 20
p0 = [[[np.random.random_sample(),np.random.random_sample(),0.002*np.random.random_sample(),np.random.random_sample(),np.random.random_sample(),np.random.random_sample(),np.random.random_sample()*distBoundGuess] for i in range(nwalkers)] for i in range(ntemps)]

def logp(x):
	return 0.0

sampler = emcee.PTSampler(ntemps,nwalkers, ndim, likelihood_function,logp,threads=2)
for p, lnprob, lnlike in sampler.sample(p0, iterations=1000):
    pass

for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                           lnlike0=lnlike,
                                           iterations=10000, thin=10):
    pass

print("ACL: ", np.max(sampler.acor))

np.savetxt('test-emcee.dat', sampler.chain[0,...])

