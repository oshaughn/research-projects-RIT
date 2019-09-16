"""
test_precompute.py:  Testing the likelihood evaluation.
    - Generates a fiducial signal
    - The synthetic signal is introduced directly overhead a single detector at the earth's barycenter.
      I assume *zero noise* and *an assumed known PSD*
    - Generates a 'template' signal at a reference distance
    - Precomputes all factors appearing in the likelihood
      Compares those factors for self-consistency (U and Q)
    - Evaluates the likelihood at the injection parameters and versus time

"""

__author__ = "R. O'Shaughnessy"

# GOAL
#   Like TestPrecompute, but using *extremely controlled* conditions so I can *calibrate a priori*
#   The source is directly overhead a single interferometer.  


from factored_likelihood import *
from matplotlib import pylab as plt
import sys

checkResults = True # Turn on to print/plot output; Turn off for testing speed
checkInputPlots = False
checkResultsPlots = True
checkResultsSlowChecks = False
rosUseRandomEventTime = True
rosUseDifferentWaveformLengths = False    # Very important test: lnL should be independent of the template length
rosUseRandomTemplateStartingFrequency = True

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

fminSNR = 25
fSample = 4096*4

ifoName = "Fake"
theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 

# Create complex FD data that does not assume Hermitianity - i.e.
# contains positive and negative freq. content
def non_herm_hoff_fake(P):
    hp, hc = lalsim.SimInspiralChooseTDWaveform(P.phiref, P.deltaT, P.m1, P.m2, 
            P.s1x, P.s1y, P.s1z, P.s2x, P.s2y, P.s2z, P.fmin, P.fref, P.dist, 
            P.incl, P.lambda1, P.lambda2, P.waveFlags, P.nonGRparams,
            P.ampO, P.phaseO, P.approx)
    hp.epoch = hp.epoch + P.tref
    hc.epoch = hc.epoch + P.tref
    hoft = hp
    if P.taper != lalsim.LAL_SIM_INSPIRAL_TAPER_NONE: # Taper if requested
        lalsim.SimInspiralREAL8WaveTaper(hoft.data, P.taper)
    if P.deltaF == None:
        TDlen = nextPow2(hoft.data.length)
    else:
        TDlen = int(1./P.deltaF * 1./P.deltaT)
        assert TDlen >= hoft.data.length

    fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)
    hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlen)
    hoftC = lal.CreateCOMPLEX16TimeSeries("hoft", hoft.epoch, hoft.f0,
            hoft.deltaT, hoft.sampleUnits, TDlen)
    # copy h(t) into a COMPLEX16 array which happens to be purely real
    for i in range(TDlen):
        hoftC.data.data[i] = hoft.data.data[i]
    FDlen = TDlen
    hoff = lal.CreateCOMPLEX16FrequencySeries("Template h(f)", 
            hoft.epoch, hoft.f0, 1./hoft.deltaT/TDlen, lal.lalHertzUnit, 
            FDlen)
    lal.COMPLEX16TimeFreqFFT(hoff, hoftC, fwdplan)
    return hoff


distanceFiducial = 25.  # Make same as reference
psd_dict[ifoName] =  lalsim.SimNoisePSDiLIGOSRD
m1 = 4*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
tEventFiducial = 0.000 # 10./fSample
if rosUseRandomEventTime:
    print("   --- Generating a random event (barycenter) time  ---- ")
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
data_dict[ifoName] = non_herm_hoff_fake(Psig)
print("Timing spacing in data vs expected : ", df, data_dict[ifoName].deltaF)

print(" == Data report == ")
detectors = data_dict.keys()
rho2Net = 0
fvalsDump = np.linspace(20,2000,1000)
psdVecDump = map(lambda x: psd_dict[ifoName](x),fvalsDump )
#print psdVecDump
np.savetxt('psd.dat', (fvalsDump,psdVecDump))  # to calibrate the SNR calculations below

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
P = ChooseWaveformParams(fmin=fminWavesTemplate, radec=False, incl=0.0,phiref=0.0, theta=0.0, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         tref=theEpochFiducial,
         deltaT=1./fSample,
         dist=100*1.e6*lal.LAL_PC_SI,
         deltaF=df) #ChooseWaveformParams(m1=m1,m2=m2,fmin = fminWaves, dist=100.*1.e6*lal.LAL_PC_SI, deltaF=df,ampO=ampO,fref=fref)
rholms_intp, crossTerms, rholms, epoch_post = PrecomputeLikelihoodTerms(theEpochFiducial,P, data_dict, psd_dict, Lmax, analyticPSD_Q)

if checkResults == True:
    # Print values of cross terms
    detectors = data_dict.keys()
    assert rholms_intp.keys() == crossTerms.keys()
    for det in detectors:
        for pair1 in rholms_intp[ifoName]:
            for pair2 in rholms_intp[ifoName]:
                if np.abs(crossTerms[det][pair1,pair2]) > 1e-5:
                    print(det, pair1, pair2, crossTerms[det][pair1,pair2], " compare (2,\pm 2) in scale to ", rhoExpected[det]**2 * 8.*lal.LAL_PI/5. *np.power( distanceFiducial/distMpcRef,2))
    
    print(" ======= UV symmetry check (reflection symmetric) ==========")
    constraint1 = 0
    for det in detectors:
        for pair1 in rholms_intp[ifoName]:
            for pair2 in rholms_intp[ifoName]:
#                print pair1, pair2, crossTerms[det][pair1,pair2], " - ",  ((-1)**(pair1[0]+pair2[0])*np.conj(crossTerms[det][(pair1[0],-pair1[1]),(pair2[0],-pair2[1])])
                constraint1 += np.abs( crossTerms[det][pair1,pair2] - ((-1)**pair1[0])*np.conj(crossTerms[det][(pair1[0],-pair1[1]), (pair2[0],-pair2[1])]) )**2
    print("   : Reflection symmetry constraint (UV) : 0 ~=  ", constraint1)

    print(" ======= UV test: Recover the SNR of the injection  ==========")
    print(" Detector lnLmodel  (-2lnLmodel)^(1/2)  rho(directly)  [last two entries should be equal! (if source and template start at the same frequency)] ")
    for det in detectors:
        lnLModel = SingleDetectorLogLikelihoodModel(crossTerms, Psig.tref, Psig.phi, Psig.theta, Psig.incl, Psig.phiref, Psig.psi, Psig.dist, 2, det)
        print(det, lnLModel, np.sqrt(-2*lnLModel), rhoExpected[det])


    print(" ======= rholm complex conjugation check (22 and 2-2 modes only) ==========")
    constraint1 = 0
    for det in detectors:
        hxx = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        hyy = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, -2)
        for i in np.arange(len(hxx.data.data)):
            constraint1+= np.abs(hxx.data.data[i]-np.conj(hyy.data.data[i]))**2
    print("   : Reflection symmetry constraint (Q22,Q2-2) with raw data: : 0 ~= ", constraint1/len(hxx.data.data))    # error per point

    # constraint1 = 0
    # for det in detectors:
    #     npts = len(hxx.data.data)
    #     t= hxx.deltaT*np.arange(npts)
    #     for i in np.arange(len(hxx.data.data)):
    #         constraint1+= np.abs(rholms_intp[det][(2,2)](t[i])-np.conj(rholms_intp[det][(2,-2)](t[i])))**2
    # print "   : Reflection symmetry constraint (Q22,Q2-2) with interpolation", constraint1/len(t)    # error per point 
    # print "   : Example  of complex conjugate quantities in interpolation ", rholms_intp['H1'][(2,2)](0.), rholms_intp['H1'][(2,-2)](0.)


    print(" ======= rholm test: Recover the SNR of the injection at the injection parameters  ==========")
    print("  WARNING: the interpolation is done with its own epoch ")
    for det in detectors:
        lnLData = SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, Psig.tref, Psig.phi, Psig.theta, Psig.incl, Psig.phiref, Psig.psi, Psig.dist, 2, det)
        print(det, lnLData, np.sqrt(lnLData), rhoExpected[det])


    print(" ======= End to end LogL: Recover the SNR of the injection at the injection parameters (*)  ==========")
    print("    *** CANNOT RUN WITH FAKE DETECTORS (as in this test) **** ")
#    lnL = FactoredLogLikelihood(P, rholms_intp, crossTerms, Lmax)
#    print "  : Evan's code : ", lnL, " versus rho^2/2 ", rho2Net/2




    print(" ======= rholm test: Plot the lnLdata timeseries at the injection parameters (*)  ==========")
    # tvals = np.linspace(-0.1,0.1,5000)
    # for det in detectors:
    #     lnLData = map( (lambda x: SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, theEpochFiducial+x, Psig.phi,Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, det)), tvals)
    #     lnLDataEstimate = np.ones(len(tvals))*rho2Net
    #     plt.figure(1)
    #     plt.plot(tvals, lnLData,label='Ldata(t)+'+det)
    #     plt.plot(tvals, lnLDataEstimate,label="$rho^2$")
    #     plt.ylim(0, rho2Net*1.1)
    #     tEventRelative =float( Psig.tref - theEpochFiducial)
    # plt.plot([tEventRelative,0],[tEventRelative,rho2Net], color='k',linestyle='-')
    # plt.title("lnL (interpolated) vs all time")
    # plt.legend()

    tvals = np.linspace(tWindowExplore[0]+tEventFiducial,tWindowExplore[1]+tEventFiducial,int(0.25*fSample*(tWindowExplore[1]-tWindowExplore[0])))
    for det in detectors:
        lnLData = map( lambda x: SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, theEpochFiducial+x, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, det), tvals)
#        print lnLData
        lnLDataEstimate = np.ones(len(tvals))*rho2Net
        plt.figure(2)
        tvalsPlot = tvals  # time is done relative to a fiducial epoch so should just work
        plt.plot(tvalsPlot, lnLData,label='Ldata(t)+'+det)
        plt.plot(tvalsPlot, lnLDataEstimate,label="$rho^2$")
        plt.ylim(-rho2Net, rho2Net*1.1)
    tEventRelative =float( Psig.tref - theEpochFiducial)
    print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is the same ", tEventRelative)
    plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='-')
    plt.plot([tEventRelative,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
    plt.title("lnLData (interpolated) vs narrow time interval")
    plt.xlabel("t(s)")
    plt.ylabel("lnL")
    plt.xlim(-0.1,0.1)
    plt.legend()
    plt.show()



    if checkResultsSlowChecks:

        print(" ======= Plotting rholm timeseries (NOT timeeshifted; are rho's offset correctly?)  ==========")
        # plot the raw rholms
        plt.figure(5)
        for det in detectors:
            QestimateScalar  = crossTerms[det][((2,2),(2,2))]*(100/distanceFiducial)/2. * np.sqrt(5./(4*lal.LAL_PI))
            rhonow = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
            print("  rho timeseries :(det, max, expected) = ",  det,np.max(np.abs(rhonow.data.data)),QestimateScalar, "   [should be equal]")
            npts = len(rhonow.data.data)
            t = rhonow.deltaT*np.arange(npts)
            plt.plot(t,np.abs(rhonow.data.data),label='rholm(2,2):'+det)
            Qestimate = np.ones(len(t))*np.abs(QestimateScalar)
            plt.plot(t,Qestimate)
            plt.title("rholm(2,2) vs all time")
            plt.legend()
        plt.show()
        print(" ======= rholm test: interpolation check (2,2) mode: data vs timesampling ==========")
        constraint1 = 0
        for det in detectors:
            hxx = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
            npts = len(hxx.data.data)
            t= hxx.deltaT*np.arange(npts)
        #        t = map(lambda x: x if x<npts*hxx.deltaT/2 else x-npts*hxx.deltaT, hxx.deltaT*np.arange(npts))  # center at t=0
            for i in np.arange(len(hxx.data.data)):
                constraint1+= np.abs(hxx.data.data[i]-rholms_intp[det][(2,2)](t[i]))**2
            print("   : Quality of interpolation per point : 0 ~= ", constraint1/len(hxx.data.data))

