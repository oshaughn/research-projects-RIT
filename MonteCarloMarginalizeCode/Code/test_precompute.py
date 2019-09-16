"""
test_precompute.py:  Testing the likelihood evaluation.
    - Generates a synthetic signal (at fixed or random sky location; at a fixed or random source orientation; at a fixed or random time)
      For simplicity, the source has fixed masses and distance.
    - The synthetic signal is introduced into a 3-detector network 'data' with *zero noise* and *an assumed known PSD*
    - Generates a 'template' signal at a reference distance
    - Precomputes all factors appearing in the likelihood
      Compares those factors for self-consistency (U and Q)
    - Evaluates the likelihood at the injection parameters and versus time

"""

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu> and R. O'Shaughnessy"

import lalsimutils
import numpy as np
import lal
import lalsimulation as lalsim

import factored_likelihood
import factored_likelihood_test
from matplotlib import pylab as plt
import sys
import scipy.optimize

#
# Set by user
#
checkResults = True
checkResultsSlowChecks = True
checkInputPlots = False
checkResultsPlots = True
checkRhoIngredients = False
rosUseRandomSkyLocation = True
rosUseRandomSourceOrientation = True
rosUseRandomEventTime = False
rosUseRandomTemplateStartingFrequency = False

approxSignal = lalsim.TaylorT4
approxTemplate = lalsim.TaylorT4

#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
psd_dict = {}
rhoExpected ={}
rhoExpectedAlt ={}
analyticPSD_Q = True # For simplicity, using an analytic PSD

fminWaves = 25
fminWavesTemplate = 25
if (rosUseRandomTemplateStartingFrequency):
    print("   --- Generating a random template starting frequency  ---- ")
    fminWavesTemplate += 5*np.random.random_sample()
fminSNR = 25
fmaxSNR = 2000
fSample = 4096*4

theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 

psd_dict['H1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['L1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['V1'] = lalsim.SimNoisePSDiLIGOSRD


distanceFiducial = 25.  # Make same as reference
m1 = 4*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
tEventFiducial = 0.0
if rosUseRandomEventTime:
    print("   --- Generating a random event (barycenter) time  ---- ")
    tEventFiducial+= 0.05*np.random.random_sample()
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100
    
Psig = lalsimutils.ChooseWaveformParams(fmin = fminWaves, radec=True, incl=0.0,phiref=0.0, theta=0.2, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         approx=approxSignal,
         fref=fref,
         tref=theEpochFiducial+tEventFiducial,
         deltaT=1./fSample,
        detector='H1', dist=distanceFiducial*1.e6*lal.LAL_PC_SI)
if rosUseRandomSkyLocation:
    print("   --- Generating a random sky location  ---- ")
    Psig.theta = np.arccos( 2*(np.random.random_sample())-1)
    Psig.phi = (np.random.random_sample())*2*lal.LAL_PI
    Psig.psi = (np.random.random_sample())*lal.LAL_PI
if rosUseRandomSourceOrientation:
    print("   --- Generating a random source orientation  ---- ")
    Psig.incl = np.arccos( 2*(np.random.random_sample())-1)
    Psig.phiref = (np.random.random_sample())*2*lal.LAL_PI
if rosUseRandomEventTime:
    print("   --- Generating a random event (barycenter) time  ---- ")
    Psig.tref += np.random.random_sample()

df = lalsimutils.findDeltaF(Psig)
Psig.deltaF = df
Psig.print_params()
print(" ======= Generating synthetic data in each interferometer (manual timeshifts) ==========")
t0 = Psig.tref
Psig.detector = 'H1'
data_dict['H1'] = lalsimutils.non_herm_hoff(Psig)  # already takes care of propagating to a detector, using the 'detector' field
Psig.detector = 'L1'
data_dict['L1'] = lalsimutils.non_herm_hoff(Psig)
Psig.detector = 'V1'
data_dict['V1'] = lalsimutils.non_herm_hoff(Psig)


print(" == Data report == ")
detectors = data_dict.keys()
rho2Net = 0
for det in detectors:
    IP = lalsimutils.ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det])
    IPOverlap = lalsimutils.ComplexOverlap(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det],analyticPSD_Q=True,full_output=True)  # Use for debugging later
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rhoExpectedAlt[det] = rhoDet2 = IPOverlap.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print(det, rhoDet, rhoDet2, " has arrival time relative to fiducial of ", float(factored_likelihood.ComputeArrivalTimeAtDetector(det, Psig.phi,Psig.theta,Psig.tref) - theEpochFiducial))
    tarrive = factored_likelihood.ComputeArrivalTimeAtDetector(det, Psig.phi,Psig.theta,Psig.tref)
    print(" and has  epoch ", float(data_dict[det].epoch), " with arrival time ",  lalsimutils.stringGPSNice(tarrive))
print("Network : ", np.sqrt(rho2Net))

if checkInputPlots:
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
    plt.xlim( 9.5,11)  # Not well centered
    plt.show()



print(" ======= Template specified: precomputing all quantities ==========")
# Struct to hold template parameters
# Fiducial distance provided but will not be used
P =  lalsimutils.ChooseWaveformParams(fmin=fminWavesTemplate, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         approx=approxTemplate,
         fref=fref,
         deltaT=1./fSample,
         tref=theEpochFiducial,
         deltaF=df)
rholms_intp, crossTerms, rholms, epoch_post = factored_likelihood.PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict, psd_dict, Lmax, fmaxSNR, analyticPSD_Q)


TestDictionary = factored_likelihood_test.TestDictionaryDefault
TestDictionary["DataReport"]             = True
TestDictionary["DataReportTime"]       = False
TestDictionary["UVReport"]              = True
TestDictionary["UVReflection"]          = True
TestDictionary["QReflection"]          = False
TestDictionary["lnLModelAtKnown"]  = True
TestDictionary["lnLDataAtKnownPlusOptimalTimePhase"] = False
TestDictionary["lnLAtKnown"]           = True
TestDictionary["lnLAtKnownMarginalizeTime"]  = False
TestDictionary["lnLDataPlot"]            = True

#opts.fmin_SNR=40

factored_likelihood_test.TestLogLikelihoodInfrastructure(TestDictionary,theEpochFiducial, epoch_post, data_dict, psd_dict, fmaxSNR, analyticPSD_Q, Psig, rholms,rholms_intp, crossTerms, detectors,Lmax)
sys.exit(0)

