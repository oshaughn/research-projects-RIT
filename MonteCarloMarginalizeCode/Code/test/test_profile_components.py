
"""
test_profile_components.py: 
   - simple timing tests on different elements in the likelihood
"""
from pylal import Fr

import factored_likelihood 
import numpy as np
import lal
import lalsimulation as lalsim

import lalsimutils
#from factored_likelihood import * # gets most things
from matplotlib import pylab as plt
import sys
import ourparams
opts,  rosDebugMessagesDictionary = ourparams.ParseStandardArguments()
factored_likelihood.rosDebugMessagesDictionary = rosDebugMessagesDictionary
lalsimutils.rosDebugMessagesDictionary            = rosDebugMessagesDictionary

#
# Set by user
#
checkResults = True
rosUseZeroNoiseCache  = True
approxTemplate = lalsim.TaylorT4

cost_dict = {}
#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
data_dict_time = {}
psd_dict = {}
rhoExpected ={}
rhoManual ={}
rhoExpectedAlt ={}
analyticPSD_Q = True # For simplicity, using an analytic PSD

# Note: Injected signal has fmin = 25. We should choose a different value.
fminWaves = fminWavesSignal = fminWavesTemplate = opts.fmin_Template  # too long can be a memory and time hog, particularly at 16 kHz
fminSNR =opts.fmin_SNR
fSample = 16384 #opts.srate


theEpochFiducial = lal.LIGOTimeGPS(1000000014.000000000)   # Use actual injection GPS time (assumed from trigger)
#theEpochFiducial = lal.LIGOTimeGPS(1000000000.000000000)     # Use epoch of the data
tEventFiducial = 0   #  time relative to fiducial epoch, used to identify window to look in.  Checked empirically.

detectors = ['H1', "L1", "V1"]
psd_dict['H1'] = lalsim.SimNoisePSDiLIGOSRD
#psd_dict['L1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['V1'] = lalsim.SimNoisePSDiLIGOSRD

Psig = lalsimutils.xml_to_ChooseWaveformParams_array("mdc.xml.gz")[0]  # Load in the physical parameters of the injection (=first element)
Psig.approx=lalsim.SEOBNRv4
Psig.print_params()


# Load IFO FFTs.
tS = lal.GPSTimeNow()
# ASSUME data has same sampling rate!
df=1./32
dt = 1./fSample
Psig.deltaF = df
Psig.detector='H1'
data_dict['H1'] = lalsimutils.non_herm_hoff(Psig)
#Psig.detector='L1'
#data_dict['L1'] = lalsimutils.non_herm_hoff(Psig)
Psig.detector='V1'
data_dict['V1'] = lalsimutils.non_herm_hoff(Psig)


tE = lal.GPSTimeNow()
cost_dict['readin'] = tE-tS
#print data_dict['H1'].data.data[10]  # confirm data loaded
df = data_dict['H1'].deltaF  
fSample = len(data_dict['H1'].data.data)*data_dict['H1'].deltaF  # Note two-sided
print(" sampling rate of data = ", fSample)




print(" ======= Template specified: precomputing all quantities ==========")
# Struct to hold template parameters
# Fiducial distance provided but will not be used
m1 = 4*lal.MSUN_SI
m2 = 4*lal.MSUN_SI
ampO =opts.amporder # sets which modes to include in the physical signal
Lmax = opts.Lmax # sets which modes to include
fref = opts.fref
P =  lalsimutils.ChooseWaveformParams(fmin=fminWaves, radec=False,
         m1=m1,m2=m2,
         ampO=ampO,
         approx=approxTemplate,
         fref=fref,
         deltaT=1./fSample,
         tref=theEpochFiducial,
         dist=100*1.e6*lal.PC_SI,  # critical that this is fiducial (will be reset later)
         deltaF=df)
tStartPrecompute = lal.GPSTimeNow()
rholms_intp, crossTerms, crossTermsV, rholms, rest = factored_likelihood.PrecomputeLikelihoodTerms(theEpochFiducial,0.1, P, data_dict,psd_dict, Lmax, 2000, analyticPSD_Q,ignore_threshold=opts.opt_SkipModeThreshold,inv_spec_trunc_Q=opts.psd_TruncateInverse,T_spec=opts.psd_TruncateInverseTime,NR_group=opts.NR_template_group,NR_param=opts.NR_template_param,use_external_EOB=opts.use_external_EOB,ROM_group=opts.ROM_template_group,ROM_param=opts.ROM_template_group)

# Pack operation does it for each detector, so I need a loop
lookupNKdict = {}
lookupKNdict={}
lookupKNconjdict={}
ctUArrayDict = {}
ctVArrayDict={}
rholmArrayDict={}
rholms_intpArrayDict={}
for det in rholms_intp.keys():
    lookupNKdict[det],lookupKNdict[det], lookupKNconjdict[det], ctUArrayDict[det], ctVArrayDict[det], rholmArrayDict[det], rholms_intpArrayDict[det] = factored_likelihood.PackLikelihoodDataStructuresAsArrays( rholms[det].keys(), rholms_intp[det], rholms[det], crossTerms[det])
#rholms_intp, crossTerms, rholms, epoch_post, lookupNKdict,lookupKNdict, ctUArrayDict, ctVArrayDict, rholmArrayDict, rholms_intpArrayDict = factored_likelihood.PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict, psd_dict, Lmax, analyticPSD_Q,ignore_threshold=opts.opt_SkipModeThreshold)
tEndPrecompute = lal.GPSTimeNow()
cost_dict['precompute'] = float(tEndPrecompute-tStartPrecompute)

tS = lal.GPSTimeNow()
nEvals = 7000
for i in np.arange(nEvals):
#    P = Psig.copy()
#    P.tref +=0.001*np.random.random_sample()  # mimic setting parameters in the structure
    lnL = factored_likelihood.FactoredLogLikelihood( Psig, rholms,  rholms_intp, crossTerms,crossTermsV, 2)
tE = lal.GPSTimeNow()
cost_dict['lnL'] = float(tE-tS)/nEvals

tS = lal.GPSTimeNow()
nEvals = 7000
for i in np.arange(nEvals):
    lnL = factored_likelihood.SingleDetectorLogLikelihoodData(theEpochFiducial, rholms_intp,Psig.tref, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, 'H1')
tE = lal.GPSTimeNow()
cost_dict['lnLdata1'] = float(tE-tS)/nEvals



tS = lal.GPSTimeNow()
nEvals = 70000
for i in np.arange(nEvals):
    lnL = factored_likelihood.SingleDetectorLogLikelihoodDataViaArray(theEpochFiducial,lookupNKdict, rholms_intpArrayDict,Psig.tref, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist,  'H1')
tE = lal.GPSTimeNow()
cost_dict['lnLdata1b'] = float(tE-tS)/nEvals
print("  +++ Fast lnLData eval (array-ized)++")
print(" Per evaluation, not vectorizing over time, cost is ", cost_dict['lnLdata1b'])


tS = lal.GPSTimeNow()
nEvals = int(fSample*5)   # should be a controlled number of evals, in case data set gets large
for i in np.arange(nEvals):
    lnL = factored_likelihood.DiscreteSingleDetectorLogLikelihoodDataViaArray(theEpochFiducial,lookupNKdict, rholmArrayDict, 1./fSample, Psig.tref, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist,  'H1')
tE = lal.GPSTimeNow()
cost_dict['lnLdataDiscArrayPerTime'] = float(tE-tS)/nEvals / len(rholmArrayDict['H1'][0])  # Cost *per time bin* being evaluated
cost_dict['lnLdataDiscArray'] = float(tE-tS)/nEvals 
print("  +++ Fast lnLdata eval ++")
print("  In total ", cost_dict['lnLdataDiscArray'],  " per evaluation of the entire time array, using nEvals = ", nEvals)
print("  Per time point (npts =  ", len(rholmArrayDict['H1'][0]), " )  the cost is ", cost_dict['lnLdataDiscArrayPerTime'])



tS = lal.GPSTimeNow()
nEvals = 100000
for i in np.arange(nEvals):
    lnL = factored_likelihood.SingleDetectorLogLikelihoodModelViaArray(lookupNKdict, ctUArrayDict, ctVArrayDict,  Psig.tref, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist,  'H1')
tE = lal.GPSTimeNow()
cost_dict['lnLModelArray'] = float(tE-tS)/nEvals  # Cost *per time bin* being evaluated
print("  +++ Fast lnLModel eval ++")
print(" Per evaluation, cost is ", cost_dict['lnLModelArray'])


if False:
    tS = lal.GPSTimeNow()
    nEvals = 10
    npts = 10
    P = Psig.copy()
    phi = np.ones(npts)*P.phi
    theta = np.ones(npts)*P.theta
    tref = np.ones(npts)*(P.tref-theEpochFiducial)
    phiref = np.ones(npts)*P.phiref
    incl = np.ones(npts)*P.incl
    psi = np.ones(npts)*P.psi
    dist = np.ones(npts)*P.dist
    for i in np.arange(nEvals):
        lnL =factored_likelihood.FactoredLogLikelihoodVectorized(theEpochFiducial,phi,theta,tref, phiref, incl, psi, dist, rholms_intp, crossTerms,Psig.tref,  Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, detectors)
    tE = lal.GPSTimeNow()
    cost_dict['lnLVector'] = float(tE-tS)/nEvals/npts
    print("  +++ Fast lnLVector eval ++")
    print("  In total ", cost_dict['lnLVector'],  " per evaluation of each point, using nEvals*npts = ", nEvals*npts)



tS = lal.GPSTimeNow()
nEvals = 7000
for i in np.arange(nEvals):
    lnL = factored_likelihood.SingleDetectorLogLikelihoodModel(crossTerms,crossTermsV, Psig.tref, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, 'H1')
tE = lal.GPSTimeNow()
cost_dict['lnLmodel1'] = float(tE-tS)/nEvals

tS = lal.GPSTimeNow()
nEvals = 100000
for i in np.arange(nEvals):
        Ylms = factored_likelihood.ComputeYlms(Lmax, 0., 0.)
tE = lal.GPSTimeNow()
cost_dict['Ylms'] = float(tE-tS)/nEvals

tS = lal.GPSTimeNow()
nEvals = 100000
for i in np.arange(nEvals):
        rho22 = rholms_intp['H1'][(2,2)](0.)
tE = lal.GPSTimeNow()
cost_dict['rholms'] = float(tE-tS)/nEvals

tS = lal.GPSTimeNow()
nEvals = 10000
for i in np.arange(nEvals):
        np.roll(data_dict['H1'],500)
tE = lal.GPSTimeNow()
cost_dict['roll'] = float(tE-tS)/nEvals

if False:
    tS = lal.GPSTimeNow()
    nEvals = 1000
    for i in np.arange(nEvals):
        lnL =factored_likelihood.NetworkLogLikelihoodPolarizationMarginalized(theEpochFiducial, rholms_intp, crossTerms,Psig.tref,  Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, detectors)
    tE = lal.GPSTimeNow()
    cost_dict['psi'] = float(tE-tS)/nEvals



# tS = lal.GPSTimeNow()
# nEvals = 20
# for i in np.arange(nEvals):
#     lnLmargTimeD = factored_likelihood.NetworkLogLikelihoodTimeMarginalizedDiscrete(theEpochFiducial, rholms, crossTerms,Psig.tref,factored_likelihood.tWindowExplore, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, detectors)
# tE = lal.GPSTimeNow()
# cost_dict['lnLmargTimeDiscrete'] = float(tE-tS)/nEvals

if False:
    tS = lal.GPSTimeNow()
    nEvals = 20
    for i in np.arange(nEvals):
        lnLmargTime = factored_likelihood.NetworkLogLikelihoodTimeMarginalized(theEpochFiducial, rholms_intp, crossTerms,Psig.tref, factored_likelihood.tWindowExplore,Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, detectors)
    tE = lal.GPSTimeNow()
    cost_dict['lnLmargTime'] = float(tE-tS)/nEvals


print(" ======= Report ==========")
print("Sampling rate: ", fSample , " and data duration ", len(data_dict['H1'].data.data)/fSample)
print("Reading data :", float(cost_dict['readin']))
print("Precomputation time (per intrinsic parameters!) : ", cost_dict['precompute'])
print("Precomputation ingredient: rolling the data : ", cost_dict['roll'])
print("Per-evaluation FactoredLogLikelihood : ", cost_dict['lnL'])
print("Per-evaluation SingleDetectorLogLikelihoodData  *3: ", cost_dict['lnLdata1']*3)
print("Per-evaluation DiscreteSingleDetectorLogLikelihoodDataViaArray  *3: ", cost_dict['lnLdataDiscArrayPerTime']*3)
print("Per-evaluation SingleDetectorLogLikelihoodModel *3 : ", cost_dict['lnLmodel1']*3)
print("Per-evaluation SingleDetectorLogLikelihoodModelViaArray *3 : ", cost_dict['lnLModelArray']*3)
print("  ... sum of previous two should equal FactoredLogLikelihood. ")
print("Per-evaluation : generate all Ylms : ", cost_dict['Ylms'])
print("Per-evaluation : evaluate one rholm * (2Lmax+1) *3: ", cost_dict['rholms']* (2*Lmax+1)*3)
print("  ... this should agree with SingleDetectorLogLikelihoodData  *3 ")
print(" Special test : polarization log likelihood cost per evaluation = ", cost_dict['psi'],  " which is significantly higher than ", cost_dict['lnL'], "because of one (python-implemented) integration, but should be the same order of time if the integral is done in C ")
#print "Per evaluation NetworkLogLikelihoodTimeMarginalizedDiscrete (an alternative) ", cost_dict['lnLmargTimeDiscrete'], " with answer ", lnLmargTimeD, " which had better compare with the unmarginalized lnL (i.e., about 200) ", lnL
#print "    ... use if convergence of a blind Monte Carlo requires many times the following number of iterations ",  cost_dict['lnLmargTimeDiscrete']/cost_dict['lnL']
print("Per evaluation NetworkLogLikelihoodTimeMarginalized (an alternative) ", cost_dict['lnLmargTime'], " with answer ", lnLmargTime, " which had better compare with the unmarginalized lnL (i.e., about 200) ", lnL)
print("    ... use if convergence of a blind Monte Carlo requires many times the following number of iterations ",  cost_dict['lnLmargTime']/cost_dict['lnL'])




TestDictionary = factored_likelihood_test.TestDictionaryDefault
TestDictionary["UVReport"]              =  analytic_signal  # this report is very confusing for real data
TestDictionary["QSquaredTimeseries"] = False # opts.plot_ShowLikelihoodVersusTime    # should be command-line option to control this plot specifically
TestDictionary["Rho22Timeseries"]      = opts.plot_ShowLikelihoodVersusTime and opts.verbose
TestDictionary["lnLModelAtKnown"]  =  analytic_signal  # this report is very confusing for real data
TestDictionary["lnLDataAtKnownPlusOptimalTimePhase"] = False
TestDictionary["lnLAtKnown"]           = True
TestDictionary["lnLAtKnownMarginalizeTime"]  = False
TestDictionary["AbsolutelyNoPlots"]   = not opts.plot_ShowLikelihoodVersusTime and not opts.plot_ShowSamplerInputs and not opts.plot_ShowSampler
TestDictionary["lnLDataPlot"]            = opts.plot_ShowLikelihoodVersusTime    # Plot individual geocentered L_k(t) and total L(t) [interpolated code]; plot discrete-data L_k(t)
TestDictionary["lnLDataPlotVersusPsi"] = opts.plot_ShowLikelihoodVersusTime
TestDictionary["lnLDataPlotVersusPhi"] = opts.plot_ShowLikelihoodVersusTime
TestDictionary["lnLDataPlotVersusPhiPsi"] = opts.plot_ShowLikelihoodVersusTime
factored_likelihood_test.TestLogLikelihoodInfrastructure(TestDictionary,theEpochFiducial, epoch_post, data_dict, psd_dict, analyticPSD_Q, Psig, rholms,rholms_intp, crossTerms, lookupNK, lookupKN,  ctUArrayDict,ctVArrayDict, rholmArrayDict, detectors,Lmax,opts)

