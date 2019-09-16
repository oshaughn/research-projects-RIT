
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
fSample = 4096*4  # will be reset by data sampling rate.

theEpochFiducial = lal.LIGOTimeGPS(1000000014.000000000)   # Use actual injection GPS time (assumed from trigger)
#theEpochFiducial = lal.LIGOTimeGPS(1000000000.000000000)     # Use epoch of the data
tEventFiducial = 0   #  time relative to fiducial epoch, used to identify window to look in.  Checked empirically.

detectors = ['H1', "L1", "V1"]
psd_dict['H1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['L1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['V1'] = lalsim.SimNoisePSDiLIGOSRD

# Load IFO FFTs.
# ASSUME data has same sampling rate!
if not rosUseZeroNoiseCache:
    fnameCache = 'test1.cache'
else:
    fnameCache = 'test1-0noise.cache'
data_dict['H1'] =frame_data_to_non_herm_hoff(fnameCache, "H1"+":FAKE-STRAIN")
data_dict['V1'] =frame_data_to_non_herm_hoff(fnameCache, "V1"+":FAKE-STRAIN")
data_dict['L1'] =frame_data_to_non_herm_hoff(fnameCache, "L1"+":FAKE-STRAIN")
#print data_dict['H1'].data.data[10]  # confirm data loaded
df = data_dict['H1'].deltaF  
fSample = len(data_dict['H1'].data.data)*data_dict['H1'].deltaF  # Note two-sided
print(" sampling rate of data = ", fSample)
print(" ===  Repeating SNR calculation (only valid for noiseless data) ===")
rho2NetManual = 0
for det in detectors:
    IP = ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det])
    rhoManual[det] = IP.norm(data_dict[det])
    rho2NetManual+= rhoManual[det]*rhoManual[det]
    print(det, rhoManual[det], " via raw data")


# TARGET INJECTED SIGNAL (for reference and calibration of results)
Psig = xml_to_ChooseWaveformParams_array("mdc.xml.gz")[0]  # Load in the physical parameters of the injection (=first element)
dfSig = Psig.deltaF = findDeltaF(Psig)  # NOT the full deltaf to avoid padding problems
Psig.print_params()
rho2Net = 0
for det in detectors:
    Psig.detector = det
    hT = lsu.non_herm_hoff(Psig)
    fSampleSig = len(hT.data.data)*hT.deltaF
    IP = ComplexIP(fLow=fminSNR, fNyq=fSampleSig/2,deltaF=dfSig,psd=psd_dict[det])
    rhoExpected[det] = rhoDet = IP.norm(hT)
    rho2Net += rhoDet*rhoDet
    print(det, rhoDet, "compare to", rhoExpected[det],  "; note it arrival time relative to fiducial of ", float(ComputeArrivalTimeAtDetector(det, Psig.phi,Psig.theta,Psig.tref) - theEpochFiducial))
print("Network : ", np.sqrt(rho2Net))





# PLOTTING INPUT DATA
print(" ====== Plotting input data (time domain) downsampled by x16 ==== ")
data_dict_time['H1'] = frame_data_to_hoft(fnameCache, "H1"+":FAKE-STRAIN")
dat = data_dict_time['H1'].data.data
print(len(dat), np.max(np.abs(dat)))
tvals = data_dict_time['H1'].deltaT*np.arange(len(dat))
dat = signal.decimate(dat,16)
tvals = signal.decimate(tvals,16)
plt.figure(5)
plt.plot(tvals,dat,label='h(t) [H1]')
plt.legend()
plt.xlim(0,64)
plt.draw()


# Plot the H1 data (some time)
# fvals = data_dict['H1'].deltaF* np.arange(len(data_dict['H1'].data.data))  # remember frequencies are padded from the center out
# plt.plot(fvals, np.abs(data_dict['H1'].data.data))
# plt.show()



m1 = 4*lal.LAL_MSUN_SI
m2 = 4*lal.LAL_MSUN_SI
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100


print(" ======= Template specified: precomputing all quantities ==========")
# Struct to hold template parameters
# Fiducial distance provided but will not be used
P =  ChooseWaveformParams(fmin=fminWaves, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         deltaT=1./fSample,
         tref=theEpochFiducial,
         dist=100*1.e6*lal.LAL_PC_SI,  # critical that this is fiducial (will be reset later)
         deltaF=df)
rholms_intp, crossTerms, rholms, epoch_post = PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict, psd_dict, Lmax, analyticPSD_Q)




#rho2Net = 400 # complete guess -- just needed to make a line
print(rho2Net)
if checkResults == True:
    print(" ======= End to end test : lnL at the injection parameters  ==========")
    lnL = FactoredLogLikelihood(Psig, rholms_intp, crossTerms, 2)
    print(" Result ", lnL, " compared to (expected network) ", rho2Net/2 , " and ", rho2NetManual/2)

    print(" ======= rholm test: Plot the lnLdata timeseries at the injection parameters (* STILL TIME OFFSET *)  ==========")
    tCenter = float(Psig.tref - theEpochFiducial)
    print(" Printing around ", tCenter, " using zero of time at ", stringGPSNice(theEpochFiducial))
    tvals = np.linspace(tCenter +tWindowReference[0],tCenter+ tWindowReference[1], fSample*(tWindowReference[1]-tWindowReference[0]))
    for det in detectors:
        lnLData = map( lambda x: SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, theEpochFiducial+x, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, det), tvals)
        lnLDataEstimate = np.ones(len(tvals))*rhoExpected[det]*rhoExpected[det]
        plt.figure(1)
        tvalsPlot = tvals 
        plt.plot(tvalsPlot, lnLData,label='Ldata(t)+'+det)
        plt.plot(tvalsPlot, lnLDataEstimate,label="$rho^2("+det+")$")
    tEventRelative =float( Psig.tref - theEpochFiducial)
    print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is ", tEventRelative)
    plt.plot([tEventRelative,tEventRelative],[0,rho2Net], color='k',linestyle='--')  # actual time of event (vs 'triggering' time), relative to the 'zero of time' theEpochFiducial
    plt.title("lnLdata (interpolated) vs narrow time interval")
    plt.xlabel('t(s)')
    plt.ylabel('lnLdata')
    


    print(" ======= rholm test: Plot the lnL timeseries at the injection parameters ==========")
    tCenter = float(Psig.tref - theEpochFiducial)
    print(" Printing in the neighborhood of a reference time ", tCenter, "centered on ", stringGPSNice(theEpochFiducial))
    tvals = np.linspace(tCenter +tWindowReference[0],tCenter+ tWindowReference[1], fSample*(tWindowReference[1]-tWindowReference[0]))
    P = Psig.copy()
    lnL = np.zeros(len(tvals))
    for indx in np.arange(len(tvals)):
            P.tref =  theEpochFiducial+tvals[indx]
            lnL[indx] =  FactoredLogLikelihood(P, rholms_intp, crossTerms, 2)
    print(" log L scale ",np.max(lnL))
    plt.figure(1)
    tvalsPlot = tvals 
    lnLEstimate = np.ones(len(tvals))*rho2Net/2
    plt.plot(tvalsPlot, lnL,label='lnL(t)')
    plt.plot(tvalsPlot, lnLEstimate,label="$rho^2/2(net)$")
    tEventRelative =float( Psig.tref - theEpochFiducial)
    print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is the same ", tEventRelative)
    plt.plot([tEventRelative,tEventRelative],[0,rho2Net], color='k',linestyle='--')  # actual time of event (vs 'triggering' time), relative to the 'zero of time' theEpochFiducial
    plt.title("lnL (interpolated) vs narrow time interval")
    plt.legend()
    plt.show()
