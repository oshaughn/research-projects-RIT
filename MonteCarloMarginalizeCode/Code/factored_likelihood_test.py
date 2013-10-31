
""""
factored_likelihood_test.py
Standard test suite for the likelihood infrastructure.
"""


import lalsimutils
import numpy as np
import lal
import lalsimulation as lalsim

import factored_likelihood
try:
    from matplotlib import pylab as plt
except:
    print "No plots for you!"
import sys
import scipy.optimize
from scipy import integrate

TestDictionaryDefault = {}
TestDictionaryDefault["DataReport"]             = False
TestDictionaryDefault["DataReportTime"]       = False
TestDictionaryDefault["UVReport"]              = False
TestDictionaryDefault["UVReflection"]          = False
TestDictionaryDefault["QReflection"]          = False
TestDictionaryDefault["QSquaredTimeseries"] = False
TestDictionaryDefault["lnLModelAtKnown"]  = False
TestDictionaryDefault["lnLDataAtKnownPlusOptimalTimePhase"] = False
TestDictionaryDefault["lnLAtKnown"]           = False
TestDictionaryDefault["lnLAtKnownMarginalizeTime"]  = False
TestDictionaryDefault["lnLDataPlot"]            = False


def TestLogLikelihoodInfrastructure(TestDictionary,theEpochFiducial,epoch_post, data_dict, psd_dict, fmaxSNR, analyticPSD_Q,Psig,rholms,rholms_intp, crossTerms, detectors, Lmax):

    fmin_SNR=30
    keysPairs = lalsimutils.constructLMIterator(Lmax)

    df = data_dict[detectors[0]].deltaF
#    fSample = opts.srate  # this may be reset by the data -- be careful.  SHOULD recalculate from deltaF and length of data
    fSample = data_dict[detectors[0]].deltaF*len(data_dict[detectors[0]].data.data)

    rhoExpected = {}
    rhoExpectedAlt = {}
    rhoFake = {}
    tWindowReference = factored_likelihood.tWindowReference
    tWindowExplore =     factored_likelihood.tWindowExplore
    tEventFiducial    = float(Psig.tref - theEpochFiducial)

    rho2Net =0
    print " ++ WARNING : Some tests depend on others.  Not made robust yet ++ "
    # Data: what is the SNR of the injected signal?
    # Only useful for *zero noise* signals.
    if TestDictionary["DataReport"]:
        print " == Data report == "
        detectors = data_dict.keys()
        rho2Net = 0
        for det in detectors:
            if analyticPSD_Q:
                IP = lalsimutils.ComplexIP(fLow=fmin_SNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det],fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q)
                IPOverlap = lalsimutils.ComplexOverlap(fLow=fmin_SNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det],fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q,full_output=True)  # Use for debugging later
            else:
                IP = lalsimutils.ComplexIP(fLow=fmin_SNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det].data.data,fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q)
                IPOverlap = lalsimutils.ComplexOverlap(fLow=fmin_SNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det].data.data,fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q,full_output=True)              
            rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
            rhoExpectedAlt[det] = rhoDet2 = IPOverlap.norm(data_dict[det])
            rho2Net += rhoDet*rhoDet
            print det, rhoDet, rhoDet2, " [via IP and Overlap]; both should agree with analytic expectations (if zero noise)."
        print "Network : ", np.sqrt(rho2Net)


        print " .... Generating the zero-noise template  (in case the real data is noisy), to estimate its amplitude at the signal  ..... "
        print "      [for some signals (coincs) the distance is not set, so the amplitude will be set to a fiducial distance. The value will be off] "
        data_fake_dict ={}
        rho2Net = 0
        for det in detectors:
            data_fake_dict[det] = lal.ResizeCOMPLEX16FrequencySeries(factored_likelihood.non_herm_hoff(Psig), 0, len(data_dict[det].data.data))  # Pad if needed!
            if analyticPSD_Q:
                IP = lalsimutils.ComplexIP(fLow=fmin_SNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det],fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q)
            else:
                IP = lalsimutils.ComplexIP(fLow=fmin_SNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det].data.data,fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q)
            rhoFake[det] = IP.norm(data_fake_dict[det])   # Reset
            rho2Net += rhoFake[det]*rhoFake[det]
            print " Fake data :", det, rhoFake[det]
        print " Fake network :", rho2Net

    if TestDictionary["DataReportTime"]:
        print " == Timing report == "
        for det in detectors:
            print det, " Time offset between data and fiducial : ", float(data_dict[det].epoch-theEpochFiducial)
            if not(det is "Fake"):
                print det," Time offset from time of flight (known parms): ",  float(factored_likelihood.ComputeArrivalTimeAtDetector(det, Psig.phi, Psig.theta, theEpochFiducial) - theEpochFiducial)
        
    # U report
    if TestDictionary["UVReport"]:
        print " ======= UV report =========="
        for det in detectors:
            for pair1 in keysPairs:
                for pair2 in keysPairs:
                    if np.abs(crossTerms[det][pair1,pair2]) > 1e-5:
                        print det, pair1, pair2, crossTerms[det][pair1,pair2]


    # UV reflection symmetry
    if (TestDictionary["UVReflection"]): # Only valid for nonprecessing
        print " ======= UV symmetry check (reflection symmetric) =========="
        constraint1 = 0
        for det in detectors:
            for pair1 in keysPairs:
                for pair2 in keysPairs:
                    constraint1 += np.abs( crossTerms[det][pair1,pair2] - ((-1)**(pair1[0]+pair2[0]))*np.conj(crossTerms[det][(pair1[0],-pair1[1]), (pair2[0],-pair2[1])]) )**2
        print "   : Reflection symmetry constraint (UV) : 0 ~=  ", constraint1
        if np.abs(constraint1) > 1e-15:
            print " ++ WARNING ++"
            print "   If you see this message, UV reflection symmetry does not hold.  If you have run with a nonprecessing binary "
            print "   then this symmetry *must* hold, preferably to machine precision.. \n  PLEASE CHECK  ANY RECENT CHANGES TO THE LOW-LEVEL INFRASTRUCTURE (e.g., inner products, psd import, etc)"

    # Q(t) reflection symmetry [discrete]
    if TestDictionary["QReflection"]:   # Only valid for nonprecessing
            constraint1 = 0
            for det in detectors:
                hxx = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
                hyy = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, -2)
                for i in np.arange(len(hxx.data.data)):
                    constraint1+= np.abs(hxx.data.data[i]-np.conj(hyy.data.data[i]))**2
            print "   : Reflection symmetry constraint (Q22,Q2-2) with raw data: : 0 ~= ", constraint1/len(hxx.data.data)    # error per point 

    # if TestDictionary["QSquaredTimeseries"]:
    #     print " ======= Q^2 test: Plot versus time  =========="
    #     plt.clf()
    #     plt.figure(2)   # plot not geocentered
    #     # Plot
    #     for det in detectors:
    #         q = factored_likelihood.QSumOfSquaresDiscrete(rholms[det],crossTerms[det])
    #         tvals = float(q.epoch-theEpochFiducial) + np.arange(len(q.data.data))*q.deltaT
    #         # restrict bin range, if length too long
    #         if len(q.data.data)>6000:
    #             nmin = int(float(Psig.tref - theEpochFiducial)/q.deltaT) - 500   # 1/16 s at 16 kHz, 1/4s at 4 kHz
    #             nmax = nmin+500
    #             plt.plot(tvals[nmin:nmax],np.abs(q.data.data)[nmin:nmax],label='q(t):'+det)
    #         else:
    #             plt.plot(tvals,np.abs(q.data.data),label='q(t):'+det)
    #         plt.xlabel('t(s) [not geocentered]')
    #         plt.ylabel('q')
    #         plt.title('q:'+factored_likelihood.stringGPSNice(q.epoch))
    #     plt.legend()

    # lnLmodel (known parameters). 
    #   Using conventional interpolated likelihood, so skip if not available
    if TestDictionary["lnLModelAtKnown"]:
            print " ======= UV test: Recover the SNR of the injection  =========="
            print " Detector lnLmodel  (-2lnLmodel)^(1/2)  rho(directly)  [last two entries should be equal!] "
            for det in detectors:
                lnLModel = factored_likelihood.SingleDetectorLogLikelihoodModel(crossTerms, Psig.tref, Psig.phi, Psig.theta, Psig.incl, Psig.phiref, Psig.psi, Psig.dist, 2, det)
                print det, lnLModel, np.sqrt(-2*lnLModel), rhoExpected[det], "      [last two equal (in zero noise)?]"

    # lnL (known parameters)
    if TestDictionary["lnLAtKnown"]:
            print " ======= End to end LogL: Recover the SNR of the injection at the injection parameters  =========="
            lnL = factored_likelihood.FactoredLogLikelihood(theEpochFiducial,Psig, rholms_intp, crossTerms, Lmax)
            print "  : Default code : ", lnL, " versus rho^2/2 ", rho2Net/2 , " [last two equal (in zero noise)?]"
            print "     [should agree in zero noise. Some disagreement expected because *recovered* (=best-fit-to-data) time and phase parameters are slightly different than injected]"

    # lnLMarginalizeTime
    if TestDictionary["lnLAtKnownMarginalizeTime"]:
            print " ======= \int L dt/T: Consistency across multiple methods  =========="
#            lnLmargT1 = factored_likelihood.NetworkLogLikelihoodTimeMarginalized(theEpochFiducial,rholms_intp, crossTerms, Psig.tref,  tWindowExplore, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, detectors)
            lnLmargT2 = factored_likelihood.NetworkLogLikelihoodTimeMarginalizedDiscrete(theEpochFiducial,rholms, crossTerms, Psig.tref, tWindowExplore, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, detectors)
            def fn(x):
                P2 = Psig.copy()
                P2.tref = theEpochFiducial+x  
                return np.exp(np.max([factored_likelihood.FactoredLogLikelihood(theEpochFiducial, P2, rholms_intp, crossTerms,Lmax),-15]))   # control for roundoff
            lnLmargT3 = np.log(integrate.quad(fn,  tWindowExplore[0], tWindowExplore[1],points=[0],limit=500)[0])
            print "Validating ln \int L dt/T over a window (manual,interp,discrete)= ", lnLmargT3,  lnLmargT2, " note time window has length ", tWindowExplore[1]-tWindowExplore[0]



    # lnLdata (plot)
    if TestDictionary["lnLDataPlot"]:

        # Plot the interpolated lnLData
        print " ======= lnLdata timeseries at the injection parameters =========="
        tmin = np.max(float(epoch_post - theEpochFiducial),tWindowReference[0]+0.03)   # the minimum time used is set by the rolling condition
        tvals = np.linspace(tWindowExplore[0]+tEventFiducial,tWindowExplore[1]+tEventFiducial,fSample*(tWindowExplore[1]-tWindowExplore[0]))
        for det in detectors:
            lnLData = map( lambda x: factored_likelihood.SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, theEpochFiducial+x, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, det), tvals)
            lnLDataEstimate = np.ones(len(tvals))*rhoExpected[det]*rhoExpected[det]
            plt.figure(1)
            plt.xlabel('t(s) [geocentered]')
            plt.ylabel('lnLdata')
            plt.title("lnLdata (interpolated) vs narrow time interval")
            tvalsPlot = tvals 
            plt.plot(tvalsPlot, lnLData,label='Ldata(t)+'+det)
            plt.plot(tvalsPlot, lnLDataEstimate,label="$rho^2("+det+")$")
            nBinsDiscrete = int(fSample*0.1)
            tStartOffsetDiscrete = tWindowExplore[0]   # timeshift correction *should* already performed by DiscreteSingleDetectorLogLikelihood
            tvalsDiscrete = tStartOffsetDiscrete +np.arange(nBinsDiscrete) *1.0/fSample
            lnLDataDiscrete = factored_likelihood.DiscreteSingleDetectorLogLikelihoodData(theEpochFiducial,rholms, theEpochFiducial+tStartOffsetDiscrete, nBinsDiscrete, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, det)
            plt.figure(2)
            plt.xlabel('t(s) [not geocentered]')
            plt.ylabel('lnLdata')
#            plt.plot(tvalsDiscrete, lnLDataDiscrete,label='Ldata(t):discrete+'+det)
            plt.title('lnLdata(t) discrete, MANUAL TIME SHIFT to geocenter')
            plt.legend()
        tEventRelative =float( Psig.tref - theEpochFiducial)
        print " Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is ", tEventRelative
        plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
        plt.title("lnLdata (interpolated) vs narrow time interval")

        print " ======= rholm test: Plot the lnL timeseries at the injection parameters =========="
        tmin = np.max(float(epoch_post - theEpochFiducial),tWindowExplore[0])   # the minimum time used is set by the rolling condition
        tvals = np.linspace(tmin,tWindowReference[1],4*fSample*(tWindowExplore[1]-tmin))
        P = Psig.copy()
        lnL = np.zeros(len(tvals))
        for indx in np.arange(len(tvals)):
            P.tref =  theEpochFiducial+tvals[indx]
            lnL[indx] =  factored_likelihood.FactoredLogLikelihood(theEpochFiducial, P, rholms_intp, crossTerms, 2)
        lnLEstimate = np.ones(len(tvals))*rho2Net/2
        plt.figure(1)
        tvalsPlot = tvals 
        plt.plot(tvalsPlot, lnL,label='lnL(t)')
        plt.plot(tvalsPlot, lnLEstimate,label="$rho^2/2(net)$")
        plt.ylim(-2*rho2Net,2*rho2Net)   # ends of the sequence can be crap. This just sets the plot range for user convenience, so it's ok to hack it.
        tEventRelative =float( Psig.tref - theEpochFiducial)
        print " Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is the same ", tEventRelative
        plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
        plt.title("lnL (interpolated) vs narrow time interval")
        plt.legend()
        plt.show()

    return True
