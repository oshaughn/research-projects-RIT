
"""
test_precompute_noisydata.py:  Testing the likelihood evaluation with noisy, 3-ifo data.
  - Loads frame data that Chris generated (assumed in signal_hoft) into 3-detector data sets
  - Uses an assumed PSD
  - Constructs a template which *exactly* matches the known source
  - Plots the individual detector lnL_k(t) timeseries and lnL timeseries
    for the injected parameters (i.e., sky location).  Does it work?
"""
from pylal import Fr

from factored_likelihood import * # gets most things
from matplotlib import pylab as plt
import sys

#
# Set by user
#
checkResults = True
rosUseZeroNoiseCache  = True

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
fminWaves = 30
fminSNR = 30
fSample = 4096  # will be reset by data sampling rate - I do not have freedom here.

theEpochFiducial = lal.LIGOTimeGPS(1000000014.000000000)   # Use actual injection GPS time (assumed from trigger)
#theEpochFiducial = lal.LIGOTimeGPS(1000000000.000000000)     # Use epoch of the data
tEventFiducial = 0   #  time relative to fiducial epoch, used to identify window to look in.  Checked empirically.

detectors = {'H1', "L1", "V1"}
psd_dict['H1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['L1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['V1'] = lalsim.SimNoisePSDiLIGOSRD

# Load IFO FFTs.
tS = lal.GPSTimeNow()
# ASSUME data has same sampling rate!
if not rosUseZeroNoiseCache:
    fnameCache = 'test1.cache'
else:
    fnameCache = 'test1-0noise.cache'
data_dict['H1'] =frame_data_to_non_herm_hoff(fnameCache, "H1"+":FAKE-STRAIN")
data_dict['V1'] =frame_data_to_non_herm_hoff(fnameCache, "V1"+":FAKE-STRAIN")
data_dict['L1'] =frame_data_to_non_herm_hoff(fnameCache, "L1"+":FAKE-STRAIN")
tE = lal.GPSTimeNow()
cost_dict['readin'] = tE-tS
#print data_dict['H1'].data.data[10]  # confirm data loaded
df = data_dict['H1'].deltaF  
fSample = len(data_dict['H1'].data.data)*data_dict['H1'].deltaF  # Note two-sided
print " sampling rate of data = ", fSample


Psig = xml_to_ChooseWaveformParams_array("mdc.xml.gz")[0]  # Load in the physical parameters of the injection (=first element)

print " ======= Template specified: precomputing all quantities =========="
# Struct to hold template parameters
# Fiducial distance provided but will not be used
m1 = 4*lal.LAL_MSUN_SI
m2 = 4*lal.LAL_MSUN_SI
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100
P =  ChooseWaveformParams(fmin=fminWaves, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         deltaT=1./fSample,
         tref=theEpochFiducial,
         dist=100*1.e6*lal.LAL_PC_SI,  # critical that this is fiducial (will be reset later)
         deltaF=df)
tStartPrecompute = lal.GPSTimeNow()
rholms_intp, crossTerms, rholms, epoch_post = PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict, psd_dict, Lmax, analyticPSD_Q)
tEndPrecompute = lal.GPSTimeNow()
cost_dict['precompute'] = float(tEndPrecompute-tStartPrecompute)

tS = lal.GPSTimeNow()
nEvals = 7000
for i in np.arange(nEvals):
#    P = Psig.copy()
#    P.tref +=0.001*np.random.random_sample()  # mimic setting parameters in the structure
    lnL = FactoredLogLikelihood(theEpochFiducial, Psig, rholms_intp, crossTerms, 2)
tE = lal.GPSTimeNow()
cost_dict['lnL'] = float(tE-tS)/nEvals

tS = lal.GPSTimeNow()
nEvals = 7000
for i in np.arange(nEvals):
    lnL = SingleDetectorLogLikelihoodData(theEpochFiducial, rholms_intp,Psig.tref, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, 'H1')
tE = lal.GPSTimeNow()
cost_dict['lnLdata1'] = float(tE-tS)/nEvals

tS = lal.GPSTimeNow()
nEvals = 7000
for i in np.arange(nEvals):
    lnL = SingleDetectorLogLikelihoodModel(crossTerms,Psig.tref, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, 'H1')
tE = lal.GPSTimeNow()
cost_dict['lnLmodel1'] = float(tE-tS)/nEvals

tS = lal.GPSTimeNow()
nEvals = 100000
for i in np.arange(nEvals):
        Ylms = ComputeYlms(Lmax, 0., 0.)
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


# tS = lal.GPSTimeNow()
# nEvals = 20
# for i in np.arange(nEvals):
#     lnLmargTime = NetworkLogLikelihoodTimeMarginalized(theEpochFiducial, rholms_intp, crossTerms,Psig.tref, Psig.phi,Psig.theta, P.incl, P.phiref,P.psi, P.dist, 2, detectors)
# tE = lal.GPSTimeNow()
# cost_dict['lnLmargTime'] = float(tE-tS)/nEvals


print " ======= Report =========="
print "Sampling rate: ", fSample , " and data duration ", len(data_dict['H1'].data.data)/fSample
print "Reading data :", float(cost_dict['readin'])
print "Precomputation time (per intrinsic parameters!) : ", cost_dict['precompute']
print "Precomputation ingredient: rolling the data : ", cost_dict['roll']
print "Per-evaluation FactoredLogLikelihood : ", cost_dict['lnL']
print "Per-evaluation SingleDetectorLogLikelihoodData  *3: ", cost_dict['lnLdata1']*3
print "Per-evaluation SingleDetectorLogLikelihoodModel *3 : ", cost_dict['lnLmodel1']*3
print "  ... sum of previous two should equal FactoredLogLikelihood. "
print "Per-evaluation : generate all Ylms : ", cost_dict['Ylms']
print "Per-evaluation : evaluate one rholm * (2Lmax+1) *3: ", cost_dict['rholms']* (2*Lmax+1)*3
print "  ... this should agree with SingleDetectorLogLikelihoodData  *3 "
#print "Per evaluation NetworkLogLikelihoodTimeMarginalized (an alternative) ", cost_dict['lnLmargTime'], " with answer ", lnLmargTime, " which had better compare with the unmarginalized lnL (i.e., about 200) ", lnL
#print "    ... use if convergence of a blind Monte Carlo requires many times the following number of iterations ",  cost_dict['lnLmargTime']/cost_dict['lnL']
