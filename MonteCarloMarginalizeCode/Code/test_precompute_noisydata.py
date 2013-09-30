
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

#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
psd_dict = {}
rhoExpected ={}
rhoExpectedAlt ={}
analyticPSD_Q = True # For simplicity, using an analytic PSD

fminWaves = 25
fminSNR = 25
fSample = 4096

theEpochFiducial = lal.LIGOTimeGPS(1000000014.000000000)   # Use actual injection GPS time (assumed from trigger)

psd_dict['H1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['L1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['V1'] = lalsim.SimNoisePSDiLIGOSRD


# Load H1 data FFT
data_dict['H1'] =frame_data_to_non_herm_hoff("test1.cache", "H1"+":FAKE-STRAIN")
data_dict['V1'] =frame_data_to_non_herm_hoff("test1.cache", "V1"+":FAKE-STRAIN")
data_dict['L1'] =frame_data_to_non_herm_hoff("test1.cache", "L1"+":FAKE-STRAIN")
print data_dict['H1'].data.data[10]  # confirm data loaded
df = data_dict['H1'].deltaF  

# Plot the H1 data (some time)
# fvals = data_dict['H1'].deltaF* np.arange(len(data_dict['H1'].data.data))  # remember frequencies are padded from the center out
# plt.plot(fvals, np.abs(data_dict['H1'].data.data))
# plt.show()



m1 = 4*lal.LAL_MSUN_SI
m2 = 4*lal.LAL_MSUN_SI
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100


print " ======= Template specified: precomputing all quantities =========="
# Struct to hold template parameters
# Fiducial distance provided but will not be used
P =  ChooseWaveformParams(fmin=fminWaves, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         deltaT=1./fSample,
         tref=theEpochFiducial,
         dist=100*1.e6*lal.LAL_PC_SI,
         deltaF=df)
rholms_intp, crossTerms, rholms, epoch_post = PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict, psd_dict, Lmax, analyticPSD_Q)



if checkResults == True:
    print " ======= rholm test: Plot the lnLdata timeseries at the injection parameters (* STILL TIME OFFSET *)  =========="
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
    print " Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is ", tEventRelative
    plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
    plt.title("lnLdata (interpolated) vs narrow time interval")
    plt.xlabel('t(s)')
    plt.ylabel('lnLdata')

    print " ======= rholm test: Plot the lnL timeseries at the injection parameters (* STILL TIME)  =========="
    tmin = np.max(float(epoch_post - theEpochFiducial),tWindowReference[0]+0.03)   # the minimum time used is set by the rolling condition
    tvals = np.linspace(tmin,tWindowReference[1],4*fSample*(tWindowReference[1]-tmin))
    P = Psig.copy()
    lnL = np.zeros(len(tvals))
    for indx in np.arange(len(tvals)):
            P.tref =  theEpochFiducial+tvals[indx]
            lnL[indx] =  FactoredLogLikelihood(theEpochFiducial, P, rholms_intp, crossTerms, 2)
    lnLEstimate = np.ones(len(tvals))*rho2Net/2
    plt.figure(1)
    tvalsPlot = tvals 
    plt.plot(tvalsPlot, lnL,label='lnL(t)')
    plt.plot(tvalsPlot, lnLEstimate,label="$rho^2/2(net)$")
    tEventRelative =float( Psig.tref - theEpochFiducial)
    print " Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is the same ", tEventRelative
    plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
    plt.title("lnL (interpolated) vs narrow time interval")
    plt.legend()
    plt.show()
