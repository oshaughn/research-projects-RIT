
""""
factored_likelihood_test.py
Standard test suite for the likelihood infrastructure.
Figure index:
    1: likelihood vs time (windowed; geocentered; time zero at theEpochFiducial)
    2: ingredients vs time (windowed; *not* geocentered; time zero at theEpcochFiducial)
    3: ingredients vs time (*not* windowed; *not* geocentered)
    4:  ... as above, but time zero at first time sample (PENDING)
    5: likelihood vs psi
    6: likelihood vs psi
    7: likelihood vs psi,phi
    10: h(f)
"""


import lalsimutils
import numpy as np
import lal
import lalsimulation as lalsim

import factored_likelihood
try:
    import matplotlib
    print(" matplotlib backend ", matplotlib.get_backend())
    if matplotlib.get_backend() is "MacOSX":
        bSavePlots  = False
    else:
        bSavePlots = True
    if matplotlib.get_backend() is 'TkAgg':  # on cluster
        print("On cluster")
        fExtensionHighDensity = "png"
        fExtensionLowDensity = "png"
    else:
        fExtensionHighDensity = "jpeg"
        fExtensionLowDensity = "png"
    from matplotlib import pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    bNoInteractivePlots = False  # Move towards saved fig plots, for speed
    bNoMatplotlib =False
except:
    bNoMatplotlib =True
    bNoInteractivePlots = True
    print("  factored_likelihood_test: No plots for you!")
import sys
import scipy.optimize
from scipy import integrate

tWindowReference = [-0.2, 0.2]  # need to be consistent with tWindowExplore from factored_likelihood.py

TestDictionaryDefault = {}
TestDictionaryDefault["DataReport"]             = True
TestDictionaryDefault["DataPlot"]             = False
TestDictionaryDefault["DataReportTime"]       = True
TestDictionaryDefault["UVReport"]              = True
TestDictionaryDefault["UVReflection"]          = True
TestDictionaryDefault["QReflection"]          = False
TestDictionaryDefault["QSquaredTimeseries"] = True
TestDictionaryDefault["Rho22Timeseries"]     = True
TestDictionaryDefault["lnLModelAtKnown"]  = True
TestDictionaryDefault["lnLDataAtKnownPlusOptimalTimePhase"] = False
TestDictionaryDefault["lnLAtKnown"]           = True
TestDictionaryDefault["lnLAtKnownMarginalizeTime"]  = True
TestDictionaryDefault["lnLDataPlot"]            = True
TestDictionaryDefault["lnLDataPlotVersusPsi"]            = True
TestDictionaryDefault["lnLDataPlotVersusPhi"]            = True
TestDictionaryDefault["lnLDataPlotVersusPhiPsi"]            = True


def TestLogLikelihoodInfrastructure(TestDictionary,theEpochFiducial, data_dict, psd_dict, fmaxSNR, analyticPSD_Q,Psig,rholms,rholms_intp, crossTerms, crossTermsV, detectors, Lmax):
    global tWindowReference

    fmin_SNR=30
#    keysPairs = lalsimutils.constructLMIterator(Lmax)
    print(detectors)
    keysPairs = rholms_intp[detectors[0]].keys()

    df = data_dict[detectors[0]].deltaF
#    fSample = opts.srate  # this may be reset by the data -- be careful.  SHOULD recalculate from deltaF and length of data
    fSample = data_dict[detectors[0]].deltaF*len(data_dict[detectors[0]].data.data)


    rhoExpected = {}
    rhoExpectedAlt = {}
    rhoFake = {}
#    tWindowReference =  tWindowReference
    tWindowExplore =     factored_likelihood.tWindowExplore
    tEventFiducial    = float(Psig.tref - theEpochFiducial)
    if not bNoMatplotlib:
            plt.figure(0)  # Make sure not overwritten
            plt.title("Placeholder - reset to this screen")

    rho2Net =0
    print(" ++ WARNING : Some tests depend on others.  Not made robust yet ++ ")
    # Data: what is the SNR of the injected signal?
    # Only useful for *zero noise* signals.
    if TestDictionary["DataReport"]:
        print(" == Data report == ")
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
            print(det, rhoDet, rhoDet2, " [via IP and Overlap]; both should agree with analytic expectations (if zero noise).")
        print("Network : ", np.sqrt(rho2Net))


        print(" .... Generating the zero-noise template  (in case the real data is noisy), to estimate its amplitude at the signal  ..... ")
        print("      [for some signals (coincs) the distance is not set, so the amplitude will be set to a fiducial distance. The value will be off] ")
        data_fake_dict ={}
        rho2Net = 0
        for det in detectors:
            Psig.detector= det
            data_fake_dict[det] = lal.ResizeCOMPLEX16FrequencySeries(
                    lalsimutils.non_herm_hoff(Psig), 0,
                    len(data_dict[det].data.data))  # Pad if needed!
            if analyticPSD_Q:
                IP = lalsimutils.ComplexIP(fLow=fmin_SNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det],fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q)
            else:
                IP = lalsimutils.ComplexIP(fLow=fmin_SNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det].data.data,fMax=fmaxSNR, analyticPSD_Q=analyticPSD_Q)
            rhoFake[det] = IP.norm(data_fake_dict[det])   # Reset
            rho2Net += rhoFake[det]*rhoFake[det]
            print(" Fake data :", det, rhoFake[det])
        print(" Fake network:", np.sqrt(rho2Net))

    if TestDictionary["DataReportTime"]:
        print(" == Timing report == ")
        for det in detectors:
            print(det, " Time offset between data and fiducial : ", float(data_dict[det].epoch-theEpochFiducial))
            if not(det is "Fake"):
                print(det," Time offset from time of flight (known parms): ",  float(factored_likelihood.ComputeArrivalTimeAtDetector(det, Psig.phi, Psig.theta, theEpochFiducial) - theEpochFiducial))
        
    # U report
    if TestDictionary["UVReport"]:
        print(" ======= UV report ==========")
        for det in detectors:
            for pair1 in keysPairs:
                for pair2 in keysPairs:
                    if np.abs(crossTerms[det][pair1,pair2]) > 1e-5:
                        print("U", det, pair1, pair2, crossTerms[det][pair1,pair2])
                        print("V", det, pair1, pair2, crossTermsV[det][pair1,pair2])


    # UV reflection symmetry
    if (TestDictionary["UVReflection"]): # Only valid for nonprecessing
        print(" ======= UV symmetry check (reflection symmetric) ==========")
        constraint1 = 0
        for det in detectors:
            for pair1 in keysPairs:
                for pair2 in keysPairs:
                    constraint1 += np.abs( crossTerms[det][pair1,pair2] - ((-1)**(pair1[0]+pair2[0]))*np.conj(crossTerms[det][(pair1[0],-pair1[1]), (pair2[0],-pair2[1])]) )**2
        print("   : Reflection symmetry constraint (UV) : 0 ~=  ", constraint1)
        if np.abs(constraint1) > 1e-15:
            print(" ++ WARNING ++")
            print("   If you see this message, UV reflection symmetry does not hold.  If you have run with a nonprecessing binary ")
            print("   then this symmetry *must* hold, preferably to machine precision.. \n  PLEASE CHECK  ANY RECENT CHANGES TO THE LOW-LEVEL INFRASTRUCTURE (e.g., inner products, psd import, etc)")

    # Q(t) reflection symmetry [discrete]
    if TestDictionary["QReflection"]:   # Only valid for nonprecessing
            constraint1 = 0
            for det in detectors:
                hxx = rholms[det][( 2, 2)]
                hyy = rholms[det][( 2, -2)]
                for i in np.arange(len(hxx.data.data)):
                    constraint1+= np.abs(hxx.data.data[i]-np.conj(hyy.data.data[i]))**2
            print("   : Reflection symmetry constraint (Q22,Q2-2) with raw data: : 0 ~= ", constraint1/len(hxx.data.data))    # error per point 

    if TestDictionary["Rho22Timeseries"] and not bNoMatplotlib:
        print(" ======= rho22: Plot versus time  ==========")
        print("    Note in Evan's implementation, they are functions of t in GPS units (i.e., 10^9) ")
        plt.clf()
        plt.figure(2)   # plot not geocentered
        # Plot
        for det in detectors:
            q = rholms[det][(2,2)] # factored_likelihood.QSumOfSquaresDiscrete(rholms[det],crossTerms[det])
            print(" rho22 plot ", det, lalsimutils.stringGPSNice(q.epoch), lalsimutils.stringGPSNice(theEpochFiducial))
            tvals = float(q.epoch-theEpochFiducial) + np.arange(len(q.data.data))*q.deltaT  # rho timeseries are truncated, so short
            plt.plot(tvals,np.abs(q.data.data),label='rho22(t):'+det)
            plt.xlabel('t(s) [not geocentered] : relative to '+lalsimutils.stringGPSNice(theEpochFiducial))
            plt.ylabel('rho22')
            plt.title('q:'+lalsimutils.stringGPSNice(theEpochFiducial))

            qf = rholms_intp[det][(2,2)] # factored_likelihood.QSumOfSquaresDiscrete(rholms[det],crossTerms[det]
            tvals = np.linspace(tWindowExplore[0],tWindowExplore[1], fSample*(tWindowExplore[1]-tWindowExplore[0]))  # rho timeseries are truncated, so short
            # Evan's implementation: large time scale for rholms(t)
            tvals = map(lambda x: float(theEpochFiducial+x), tvals)
            qvals = map(qf, tvals)
            plt.plot(tvals,np.abs(qvals),label='rho22intp(t):'+det)
            plt.xlabel('t(s) [not geocentered] : relative to '+lalsimutils.stringGPSNice(theEpochFiducial))
            plt.ylabel('rho22')
            plt.title('q:'+lalsimutils.stringGPSNice(theEpochFiducial))

        plt.legend()
        plt.savefig("FLT-rho22."+fExtensionHighDensity)

    # lnLmodel (known parameters). 
    #   Using conventional interpolated likelihood, so skip if not available
    if TestDictionary["lnLModelAtKnown"]:
            print(" ======= UV test: Recover the SNR of the injection  ==========")
            print(" Detector lnLmodel  (-2lnLmodel)^(1/2)  rho(directly)  [last two entries should be equal!] ")
            for det in detectors:
                lnLModel = factored_likelihood.SingleDetectorLogLikelihoodModel(crossTerms, crossTermsV, Psig.tref, Psig.phi, Psig.theta, Psig.incl, Psig.phiref, Psig.psi, Psig.dist, Lmax, det)
                print(det, lnLModel, np.sqrt(-2*lnLModel), rhoExpected[det], "      [last two equal (in zero noise)?]")

    # lnL (known parameters)
    if TestDictionary["lnLAtKnown"]:
            print(" ======= End to end LogL: Recover the SNR of the injection at the injection parameters  ==========")
            lnL = factored_likelihood.FactoredLogLikelihood(Psig, rholms, rholms_intp, crossTerms, crossTermsV, Lmax)
            print("  : Default code : ", lnL, " versus rho^2/2 ", rho2Net/2 , " [last two equal (in zero noise)?]")
            print("     [should agree in zero noise. Some disagreement expected because *recovered* (=best-fit-to-data) time and phase parameters are slightly different than injected]")

    # lnLMarginalizeTime
    if TestDictionary["lnLAtKnownMarginalizeTime"]:
            print(" ======= \int L dt/T: Consistency across multiple methods  ==========")
            tvals = np.linspace(tWindowExplore[0], tWindowExplore[1], int(fSample*(tWindowExplore[1]-tWindowExplore[0])))
            lnLmargT1 = factored_likelihood.FactoredLogLikelihoodTimeMarginalized(tvals,Psig, rholms_intp, rholms, crossTerms, crossTermsV, Lmax)
            lnLmargT1b = factored_likelihood.FactoredLogLikelihoodTimeMarginalized(tvals,Psig, rholms_intp, rholms, crossTerms, crossTermsV, Lmax,interpolate=True)
#            lnLmargT1 = factored_likelihood.NetworkLogLikelihoodTimeMarginalized(theEpochFiducial,rholms_intp, crossTerms, Psig.tref,  tWindowExplore, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, detectors)
            lnLmargT2 = factored_likelihood.NetworkLogLikelihoodTimeMarginalizedDiscrete(theEpochFiducial,rholms, crossTerms, crossTermsV,Psig.tref, tWindowExplore, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, Lmax, detectors)
            def fn(x):
                P2 = Psig.copy()
                P2.tref = theEpochFiducial+x  
                return np.exp(np.max([factored_likelihood.FactoredLogLikelihood(P2, rholms, rholms_intp, crossTerms,crossTermsV,Lmax),-15]))   # control for roundoff
            lnLmargT3 = np.log(integrate.quad(fn,  tWindowExplore[0], tWindowExplore[1],points=[0],limit=500)[0])
            print("Validating ln \int L dt/T over a window (manual,interp,discrete)= ", lnLmargT3, lnLmargT1, lnLmargT1b,   " note time window has length ", tWindowExplore[1]-tWindowExplore[0])



    # lnLdata (plot)
    if TestDictionary["lnLDataPlot"] and not bNoMatplotlib:

        # Plot the interpolated lnLData versus *time*
        print(" ======= lnLdata timeseries at the injection parameters ==========")
        tvals = np.linspace(tWindowExplore[0],tWindowExplore[1],fSample*(tWindowExplore[1]-tWindowExplore[0]))
        for det in detectors:
            lnLData = map( lambda x: factored_likelihood.SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, theEpochFiducial+x, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, Lmax, det), tvals)
            lnLDataEstimate = np.ones(len(tvals))*rhoExpected[det]*rhoExpected[det]
            plt.figure(1)
            plt.xlabel('t(s) [geocentered]relative to '+lalsimutils.stringGPSNice(theEpochFiducial))
            plt.ylabel('lnLdata')
            plt.title("lnLdata (interpolated) vs narrow time interval")
            indx = [k for  k,value in enumerate((tvals>tWindowExplore[0])  * (tvals<tWindowExplore[1])) if value] # gets results if true
            lnLfac = 4*np.max([np.abs(lnLData[k]) for k in indx])  # Max in window
            if lnLfac < 100:
                lnLfac = 100
            plt.ylim(-lnLfac,lnLfac)   # sometimes we get yanked off the edges.  Larger than this isn't likely
            tvalsPlot = tvals 
            plt.plot(tvalsPlot, lnLData,label='Ldata(t)+'+det)
            plt.plot(tvalsPlot, lnLDataEstimate,label="$rho^2("+det+")$")
            nBinsDiscrete =  len(data_dict[det].data.data)#int(fSample*1)                      # plot all of data, straight up!
            tStartOffsetDiscrete = 0 #tWindowExplore[0]-0.5   # timeshift correction *should* already performed by DiscreteSingleDetectorLogLikelihood
            tvalsDiscrete = tStartOffsetDiscrete +np.arange(nBinsDiscrete) *1.0/fSample
            lnLDataDiscrete = factored_likelihood.DiscreteSingleDetectorLogLikelihoodData(theEpochFiducial,rholms, theEpochFiducial+tStartOffsetDiscrete, nBinsDiscrete, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, Lmax, det)
            tvalsDiscrete = tvalsDiscrete[:len(lnLDataDiscrete)]
            plt.figure(2)
            plt.xlabel('t(s) [not geocentered] relative to '+lalsimutils.stringGPSNice(theEpochFiducial))
            plt.ylabel('lnLdata')
            nSkip = 1 # len(tvalsDiscrete)/4096   # Go to fixed number of points
            lnLDataEstimate = np.ones(len(tvalsDiscrete))*rhoExpected[det]*rhoExpected[det]
            plt.plot(tvalsDiscrete, lnLDataEstimate,label="$rho^2("+det+")$")
            plt.plot(tvalsDiscrete[::nSkip], lnLDataDiscrete[::nSkip],label='Ldata(t):discrete+'+det)
            plt.title('lnLdata(t) discrete, NO TIME SHIFTS')
            plt.legend()
        tEventRelative =float( Psig.tref - theEpochFiducial)
        print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is ", tEventRelative)
        plt.figure(1)
        plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
        plt.title("lnLdata (interpolated) vs narrow time interval")
        plt.xlim(-0.05,0.05)
        if bSavePlots:
            plt.savefig("FLT-lnLData."+fExtensionLowDensity)

        print(" ======= rholm test: Plot the lnL timeseries at the injection parameters ==========")
        tvals = np.linspace(tWindowExplore[0],tWindowExplore[1],fSample*(tWindowExplore[1]-tWindowExplore[0]))
        print("  ... plotting over range ", [min(tvals), max(tvals)], " with npts = ", len(tvals))
        P = Psig.copy()
        lnL = np.zeros(len(tvals))
        for indx in np.arange(len(tvals)):
            P.tref =  theEpochFiducial+tvals[indx]
            lnL[indx] =  factored_likelihood.FactoredLogLikelihood(P, rholms,  rholms_intp, crossTerms,crossTermsV, Lmax)
        lnLEstimate = np.ones(len(tvals))*rho2Net/2
        plt.figure(1)
        tvalsPlot = tvals 
        plt.plot(tvalsPlot, lnL,label='lnL(t)')
        plt.plot(tvalsPlot, lnLEstimate,label="$rho^2/2(net)$")

        indx = [k for  k,value in enumerate((tvals>tWindowExplore[0])  * (tvals<tWindowExplore[1])) if value] # gets results if true
        lnLfac = 4*np.max([np.abs(lnL[k]) for k in indx])  # Max in window
        if lnLfac < 100:
            lnLfac = 100
        plt.ylim(-lnLfac,lnLfac)   # sometimes we get yanked off the edges.  Larger than this isn't likely
        tEventRelative =float( Psig.tref - theEpochFiducial)
        print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is the same ", tEventRelative)
        plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
        plt.title("lnL (interpolated) vs narrow time interval")
        plt.xlim(-0.05,0.05)
        plt.legend()
        if bSavePlots:
            plt.savefig("FLT-lnL."+fExtensionLowDensity)

    # lnLdata (plot), using vectorized packing
    if TestDictionary["lnLDataPlotAlt"] and not bNoMatplotlib:
        lookupNKDict = {}
        lookupKNDict={}
        lookupKNconjDict={}
        ctUArrayDict = {}
        ctVArrayDict={}
        rholmArrayDict={}
        rholms_intpArrayDict={}
        epochDict={}

        for det in rholms_intp.keys():
                lookupNKDict[det],lookupKNDict[det], lookupKNconjDict[det], ctUArrayDict[det], ctVArrayDict[det], rholmArrayDict[det], rholms_intpArrayDict[det], epochDict[det] = factored_likelihood.PackLikelihoodDataStructuresAsArrays( rholms[det].keys(), rholms_intp[det], rholms[det], crossTerms[det],crossTermsV[det])

        # Plot the interpolated lnLData versus *time*
        print(" ======= lnLdata timeseries at the injection parameters ==========")
        tvals = np.linspace(tWindowExplore[0],tWindowExplore[1],fSample*(tWindowExplore[1]-tWindowExplore[0]))
        for det in detectors:
            lnLData = map( lambda x: factored_likelihood.SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, theEpochFiducial+x, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, Lmax, det), tvals)
            lnLDataEstimate = np.ones(len(tvals))*rhoExpected[det]*rhoExpected[det]
            plt.figure(1)
            plt.xlabel('t(s) [geocentered]relative to '+lalsimutils.stringGPSNice(theEpochFiducial))
            plt.ylabel('lnLdata')
            plt.title("lnLdata (interpolated) vs narrow time interval")
            indx = [k for  k,value in enumerate((tvals>tWindowExplore[0])  * (tvals<tWindowExplore[1])) if value] # gets results if true
            lnLfac = 4*np.max([np.abs(lnLData[k]) for k in indx])  # Max in window
            if lnLfac < 100:
                lnLfac = 100
            plt.ylim(-lnLfac,lnLfac)   # sometimes we get yanked off the edges.  Larger than this isn't likely
            tvalsPlot = tvals 
            plt.plot(tvalsPlot, lnLData,label='Ldata(t)+'+det)
            plt.plot(tvalsPlot, lnLDataEstimate,label="$rho^2("+det+")$")
            nBinsDiscrete =  len(data_dict[det].data.data)#int(fSample*1)                      # plot all of data, straight up!
            tStartOffsetDiscrete = 0 #tWindowExplore[0]-0.5   # timeshift correction *should* already performed by DiscreteSingleDetectorLogLikelihood
            tvalsDiscrete = tStartOffsetDiscrete +np.arange(nBinsDiscrete) *1.0/fSample
            lnLDataDiscrete = factored_likelihood.DiscreteSingleDetectorLogLikelihoodDataViaArray(tvalsPlot,Psig, lookupNKDict, rholmArrayDict, Lmax=Lmax,det=det)
            tvalsDiscrete = tvalsDiscrete[:len(lnLDataDiscrete)]
            plt.figure(2)
            plt.xlabel('t(s) [not geocentered] relative to '+lalsimutils.stringGPSNice(theEpochFiducial))
            plt.ylabel('lnLdata')
            nSkip = 1 # len(tvalsDiscrete)/4096   # Go to fixed number of points
            lnLDataEstimate = np.ones(len(tvalsDiscrete))*rhoExpected[det]*rhoExpected[det]
            plt.plot(tvalsDiscrete, lnLDataEstimate,label="$rho^2("+det+")$")
            plt.plot(tvalsDiscrete[::nSkip], lnLDataDiscrete[::nSkip],label='Ldata(t):discrete+'+det)
            plt.title('lnLdata(t) discrete, NO TIME SHIFTS')
            plt.legend()
        tEventRelative =float( Psig.tref - theEpochFiducial)
        print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is ", tEventRelative)
        plt.figure(1)
        plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
        plt.title("lnLdata (interpolated) vs narrow time interval")
        plt.xlim(-0.05,0.05)
        if bSavePlots:
            plt.savefig("FLT-lnLaltData."+fExtensionLowDensity)

        print(" ======= rholm test: Plot the lnL timeseries at the injection parameters ==========")
        tvals = np.linspace(tWindowExplore[0],tWindowExplore[1],fSample*(tWindowExplore[1]-tWindowExplore[0]))
        print("  ... plotting over range ", [min(tvals), max(tvals)], " with npts = ", len(tvals))
        P = Psig.copy()
        lnL = np.zeros(len(tvals))
        for indx in np.arange(len(tvals)):
            P.tref =  theEpochFiducial+tvals[indx]
            lnL[indx] =  factored_likelihood.FactoredLogLikelihood(P, rholms,  rholms_intp, crossTerms,crossTermsV, Lmax)
        lnLEstimate = np.ones(len(tvals))*rho2Net/2
        plt.figure(1)
        tvalsPlot = tvals 
        plt.plot(tvalsPlot, lnL,label='lnL(t)')
        plt.plot(tvalsPlot, lnLEstimate,label="$rho^2/2(net)$")

        indx = [k for  k,value in enumerate((tvals>tWindowExplore[0])  * (tvals<tWindowExplore[1])) if value] # gets results if true
        lnLfac = 4*np.max([np.abs(lnL[k]) for k in indx])  # Max in window
        if lnLfac < 100:
            lnLfac = 100
        plt.ylim(-lnLfac,lnLfac)   # sometimes we get yanked off the edges.  Larger than this isn't likely
        tEventRelative =float( Psig.tref - theEpochFiducial)
        print(" Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is the same ", tEventRelative)
        plt.plot([tEventFiducial,tEventFiducial],[0,rho2Net], color='k',linestyle='--')
        plt.title("lnL (interpolated) vs narrow time interval")
        plt.xlim(-0.05,0.05)
        plt.legend()
        if bSavePlots:
            plt.savefig("FLT-lnL_alt."+fExtensionLowDensity)

    # lnLdata (plot)
    if TestDictionary["lnLDataPlotVersusPsi"] and not bNoMatplotlib:
        print(" ======= Code test: Plot the lnL versus psi, at the injection parameters ==========")
        psivals = np.linspace(0, 2*np.pi,500)
        P = Psig.copy()
        P.tref =Psig.tref    #Probably already created. Be careful re recreating, some memory management issues
        lnL = np.zeros(len(psivals))
        for indx in np.arange(len(psivals)):
            P.psi =  psivals[indx]
            lnL[indx] =  factored_likelihood.FactoredLogLikelihood(P, rholms, rholms_intp, crossTerms,crossTermsV, 2)
        lnLEstimate = np.ones(len(psivals))*rho2Net/2
        plt.figure(5)
        plt.plot(psivals, lnL,label='lnL(phi)')
        plt.plot(psivals, lnLEstimate,label="$rho^2/2(net)$")
        psiEvent = Psig.psi 
        plt.ylabel('lnL')
        plt.xlabel('$\psi$')
        plt.plot([psiEvent,psiEvent],[0,rho2Net], color='k',linestyle='--')
        plt.plot([psiEvent+np.pi,psiEvent+np.pi],[0,rho2Net], color='k',linestyle='--')   # add second line
        plt.plot([psiEvent+2*np.pi,psiEvent+2*np.pi],[0,rho2Net], color='k',linestyle='--')   # add third line
        plt.title("lnL (interpolated) vs phase (psi)")
        plt.legend()

    # lnL (plot)
    if TestDictionary["lnLDataPlotVersusPhi"]:  # here phi means *phiref*, not *phiS*
        print(" ======= Code test: Plot the lnL versus phi, at the injection parameters ==========")
        phivals = np.linspace(0, 2*np.pi,500)
        P = Psig.copy()
        P.tref =Psig.tref    #Probably already created. Be careful re recreating, some memory management issues
        lnL = np.zeros(len(phivals))
        lnLdata = {}
        for indx in np.arange(len(phivals)):
            P.phiref =  phivals[indx]
            lnL[indx] =  factored_likelihood.FactoredLogLikelihood(P, rholms, rholms_intp, crossTerms, crossTermsV, 2)
        for det in detectors:
            lnLdata[det] = np.zeros(len(phivals))
            for indx in np.arange(len(phivals)):
                P.phiref =  phivals[indx]
                lnLdata[det][indx] =  factored_likelihood.SingleDetectorLogLikelihoodData(theEpochFiducial, rholms_intp,P.tref,P.phi,P.theta,P.incl,-P.phiref,P.psi,P.dist, Lmax,det)
            
        lnLEstimate = np.ones(len(phivals))*rho2Net/2
        plt.figure(6)
        plt.plot(phivals, lnL,label='lnL(phi)')
        plt.plot(phivals, lnLEstimate,label="$rho^2/2(net)$")
        for det in detectors:
            plt.plot(phivals, lnLdata[det],label='lnLdata(phi):'+det) 
        phiEvent = Psig.phiref
        plt.ylabel('lnL')
        plt.xlabel('$\phi$')
        plt.plot([phiEvent,phiEvent],[0,rho2Net], color='k',linestyle='-')
        plt.plot([phiEvent+np.pi,phiEvent+np.pi],[0,rho2Net], color='k',linestyle='--')         # not a physically required extrema, but often a good approx
        plt.plot([phiEvent+2*np.pi,phiEvent+2*np.pi],[0,rho2Net], color='k',linestyle='-')   # add second line
        plt.title("lnL (interpolated) vs phase (phi)")
        plt.legend()

    # lnLdata (plot)
    if TestDictionary["lnLDataPlotVersusPhiPsi"]:
        print(" ======= Code test: Plot the lnL versus phi,psi, at the injection parameters ==========")
        psivals = np.linspace(0, 2*np.pi,50)
        phivals = np.linspace(0, 2*np.pi,50)
        psivals, phivals = np.meshgrid(psivals,phivals)
        P = Psig.copy()
        P.tref =Psig.tref    #Probably already created. Be careful re recreating, some memory management issues
        lnL = np.zeros(psivals.shape)
        for indx in np.arange(psivals.shape[0]):
            for y in np.arange(psivals.shape[1]):
                P.psi =  psivals[indx,y]
                P.phiref =  phivals[indx,y]
                lnL[indx,y] =  factored_likelihood.FactoredLogLikelihood(P, rholms, rholms_intp, crossTerms,crossTermsV, 2)

        myfig = plt.figure(7)
        ax = myfig.add_subplot(111, projection='3d')
        ax.plot_wireframe(psivals,phivals,lnL)
#        ax.plot_wireframe(psivals,phivals,phivals)  # Confirm I am plotting what I think I am
        ax.set_xlabel('psi')
        ax.set_ylabel('phi')
        ax.set_zlabel('lnL')

    if (not bNoMatplotlib) and (not bNoInteractivePlots)  and (TestDictionary["lnLDataPlotVersusPsi"] or TestDictionary["lnLDataPlot"] or  TestDictionary["lnLDataPlotVersusPhiPsi"]): # TestDictionary["DataReport"] or
        print(" Making plots ")
        print(TestDictionary)
        plt.show()


    return True
