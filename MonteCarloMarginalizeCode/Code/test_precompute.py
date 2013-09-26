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

from factored_likelihood import *
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
rosUseRandomSkyLocation = False
rosUseRandomSourceOrientation = False
rosUseRandomEventTime = False

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

theEpochFiducial = lal.LIGOTimeGPS(1064023405.000000000)   # 2013-09-24 early am 
#theEpochFiducial = 1064023405.000000000
#theEpochFiducial =   1000000000
#theEpochFiducial =   0

# psd_dict['H1'] = lal.LIGOIPsd
# psd_dict['L1'] = lal.LIGOIPsd
# psd_dict['V1'] = lal.LIGOIPsd
psd_dict['H1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['L1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['V1'] = lalsim.SimNoisePSDiLIGOSRD


distanceFiducial = 25.  # Make same as reference
m1 = 4*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
tEventFiducial = 0.02
if rosUseRandomEventTime:
    print "   --- Generating a random event (barycenter) time  ---- " 
    tEventFiducial+= 0.05*np.random.random_sample()
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100
    
Psig = ChooseWaveformParams(fmin = fminWaves, radec=True, incl=0.0,phiref=0.0, theta=0.2, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         tref=theEpochFiducial+tEventFiducial,
         deltaT=1./fSample,
        detector='H1', dist=distanceFiducial*1.e6*lal.LAL_PC_SI)
if rosUseRandomSkyLocation:
    print "   --- Generating a random sky location  ---- " 
    Psig.theta = np.arccos( 2*(np.random.random_sample())-1)
    Psig.phi = (np.random.random_sample())*2*lal.LAL_PI
    Psig.psi = (np.random.random_sample())*lal.LAL_PI
if rosUseRandomSourceOrientation:
    print "   --- Generating a random source orientation  ---- " 
    Psig.incl = np.arccos( 2*(np.random.random_sample())-1)
    Psig.phiref = (np.random.random_sample())*2*lal.LAL_PI
if rosUseRandomEventTime:
    print "   --- Generating a random event (barycenter) time  ---- " 
    Psig.tref += np.random.random_sample()

df = findDeltaF(Psig)
Psig.deltaF = df
Psig.print_params()
print " ======= Generating synthetic data in each interferometer (manual timeshifts) =========="
t0 = Psig.tref
Psig.tref =  ComputeArrivalTimeAtDetector('H1', Psig.phi,Psig.theta,Psig.tref)
data_dict['H1'] = non_herm_hoff(Psig)
Psig.detector = 'L1'
Psig.tref =ComputeArrivalTimeAtDetector('L1', Psig.phi,Psig.theta,Psig.tref)
data_dict['L1'] = non_herm_hoff(Psig)
Psig.detector = 'V1'
Psig.tref =  ComputeArrivalTimeAtDetector('V1', Psig.phi,Psig.theta,Psig.tref)
data_dict['V1'] = non_herm_hoff(Psig)


print " == Data report == "
detectors = data_dict.keys()
rho2Net = 0
for det in detectors:
    IP = ComplexIP(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det])
    IPOverlap = ComplexOverlap(fLow=fminSNR, fNyq=fSample/2,deltaF=df,psd=psd_dict[det],analyticPSD_Q=True,full_output=True)  # Use for debugging later
    rhoExpected[det] = rhoDet = IP.norm(data_dict[det])
    rhoExpectedAlt[det] = rhoDet2 = IPOverlap.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print det, rhoDet, rhoDet2, " has arrival time relative to fiducial of ", float(ComputeArrivalTimeAtDetector(det, Psig.phi,Psig.theta,Psig.tref) - theEpochFiducial)
    tarrive = ComputeArrivalTimeAtDetector(det, Psig.phi,Psig.theta,Psig.tref)
    print " and has  epoch ", float(data_dict[det].epoch), " with arrival time ",  tarrive.gpsSeconds, ".", tarrive.gpsNanoSeconds
print "Network : ", np.sqrt(rho2Net)

if checkInputPlots:
    print " == Plotting detector data (time domain; requires regeneration, MANUAL TIMESHIFTS,  and seperate code path! Argh!) == "
    P = Psig.copy()
    P.tref = Psig.tref
    for det in detectors:
        P.detector=det
        P.tref = ComputeArrivalTimeAtDetector(det, P.phi, P.theta,P.tref)
        hT = hoft(P)
        tvals = float(P.tref - theEpochFiducial) + hT.deltaT*np.arange(len(hT.data.data))
        plt.figure(1)
        plt.plot(tvals, hT.data.data,label=det)

    tlen = hT.deltaT*len(np.nonzero(np.abs(hT.data.data)))
    plt.xlim( 9.5,11)  # Not well centered
    plt.show()

    # # print " == Plotting detector data (frequency domain) == "
    # fakepsdFunction = lalsim.SimNoisePSDiLIGOSRD
    # for det in detectors:
    #     nbins = len(data_dict[det].data.data)
    #     fvals = np.arange(-IP.fNyq, IP.fNyq, df )
    #     hTildeAmpVals = np.abs(data_dict[det].data.data) 
    #     plt.figure(2)
    #     plt.plot(fvals, hTildeAmpVals, label=det)
    #     plt.xlim(-500,500)
    #     plt.figure(3)
    #     plt.xlim(1,3)
    #     plt.ylim(-25,-20)
    #     fakepsdData = map(lambda x: fakepsdFunction(max(x, 1e-2)), np.abs(fvals))
    #     plt.plot(np.log10(np.abs(fvals)), np.log10(np.sqrt(np.abs(fvals))*hTildeAmpVals), label=det)
    #     plt.plot(np.log10(np.abs(fvals)), np.log10(np.sqrt(fakepsdData)),label=det+'IFO')
    # plt.show()



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


print " ======= Reporting on results =========="
#
# Examine and sanity check the output
#
if checkResults == True:
    # Print values of cross terms
    detectors = data_dict.keys()
    for det in detectors:
        for pair1 in rholms_intp['V1']:
            for pair2 in rholms_intp['V1']:
                if np.abs(crossTerms[det][pair1,pair2]) > 1e-5:
                    print det, pair1, pair2, crossTerms[det][pair1,pair2]
    
    print " ======= UV symmetry check (reflection symmetric) =========="
    constraint1 = 0
    for det in detectors:
        for pair1 in rholms_intp['V1']:
            for pair2 in rholms_intp['V1']:
#                print pair1, pair2, crossTerms[det][pair1,pair2], " - ",  ((-1)**(pair1[0]+pair2[0])*np.conj(crossTerms[det][(pair1[0],-pair1[1]),(pair2[0],-pair2[1])])
                constraint1 += np.abs( crossTerms[det][pair1,pair2] - ((-1)**pair1[0])*np.conj(crossTerms[det][(pair1[0],-pair1[1]), (pair2[0],-pair2[1])]) )**2
    print "   : Reflection symmetry constraint (UV) : 0 ~=  ", constraint1

    print " ======= UV test: Recover the SNR of the injection  =========="
    print " Detector lnLmodel  (-2lnLmodel)^(1/2)  rho(directly)  [last two entries should be equal!] "
    for det in detectors:
        lnLModel = SingleDetectorLogLikelihoodModel(crossTerms, Psig.tref, Psig.phi, Psig.theta, Psig.incl, Psig.phiref, Psig.psi, Psig.dist, 2, det)
        print det, lnLModel, np.sqrt(-2*lnLModel), rhoExpected[det], "      [last two equal?]"

    print " ======= rholm test: Recover the SNR of the injection at the injection parameters (*)  =========="
    for det in detectors:
        lnLData = SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, Psig.tref, Psig.phi,  Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, det)
        print det, lnLData, np.sqrt(lnLData), rhoExpected[det]

    print " ======= End to end LogL: Recover the SNR of the injection at the injection parameters (*)  =========="
    lnL = FactoredLogLikelihood(theEpochFiducial,Psig, rholms_intp, crossTerms, Lmax)
    print "  : Evan's code : ", lnL, " versus rho^2/2 ", rho2Net/2


if checkRhoIngredients == True:    
    print " ======= rholm ingredients: the integrand (for 22 vs H1) =========="
    hlms = hlmoff(Psig, Lmax)
    h22 = lalsim.SphHarmFrequencySeriesGetMode(hlms, 2, 2)
    h20 = lalsim.SphHarmFrequencySeriesGetMode(hlms, 2, 0)
    fvals = np.linspace(-IP.fNyq,IP.fNyq, len(h22.data.data))
    rhoIntegrand = np.conj(h22.data.data)*data_dict['H1'].data.data*IP.longweights
    rhoIntegrand0 = np.conj(h20.data.data)*data_dict['H1'].data.data*IP.longweights
    plt.figure(1)
    plt.xlim(-500,500)
    plt.plot(fvals,np.abs(rhoIntegrand),label="integrand h*data/Sh")
    plt.plot(fvals,np.real(rhoIntegrand),label='re')
    plt.plot(fvals,np.imag(rhoIntegrand),label='im')
    plt.legend()
    plt.show()

    plt.figure(2)
    myfft = np.fft.ifft(rhoIntegrand)
    tvals = np.linspace(0,1/h22.deltaF,len(myfft))
    plt.plot(tvals,np.abs(myfft),label='Q22(t)')
    plt.legend()
    plt.show()

    # plt.figure(2)
    # myfft = np.fft.ifft(rhoIntegrand0)
    # tvals = np.linspace(0,1/h22.deltaF,len(myfft))
    # plt.plot(tvals,np.abs(myfft),label='Q20(t)')
    # plt.legend()
    # plt.show()  # Show at the same time as the others


if checkResultsSlowChecks:
    print " ======= rholm test: interpolation check (2,2) mode: data vs timesampling =========="
    constraint1 = 0
    for det in detectors:
        hxx = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        npts = len(hxx.data.data)
        if (rosInterpolateOnlyTimeWindow):
            t= hxx.deltaT*np.arange(int(( tWindowReference[1]-tWindowReference[0])/hxx.deltaT))
        else:
            t= hxx.deltaT*np.arange(npts)
        rhovals =map( rholms_intp[det][(2,2)], t)   # slow interpolation over all time! argh!
        constraint1 = np.sum(np.abs(hxx.data.data[:len(rhovals)] - rhovals)**2)
        print "   : Quality of interpolation per point (vector method): 0 ~= ", constraint1/len(hxx.data.data),  " [this can be very bad if the interpolator is only tuned to a smal subset of points]"
        # for i in np.arange(len(hxx.data.data)):
        #     constraint1+= np.abs(hxx.data.data[i]-rholms_intp[det][(2,2)](t[i]))**2
        # print "   : Quality of interpolation per point : 0 ~= ", constraint1/len(hxx.data.data)

if checkRhoIngredients == True:

    print " ======= rholm complex conjugation check (22 and 2-2 modes only) =========="
    constraint1 = 0
    for det in detectors:
        hxx = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        hyy = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, -2)
        for i in np.arange(len(hxx.data.data)):
            constraint1+= np.abs(hxx.data.data[i]-np.conj(hyy.data.data[i]))**2
    print "   : Reflection symmetry constraint (Q22,Q2-2) with raw data: : 0 ~= ", constraint1/len(hxx.data.data)    # error per point 


    print " ======= rholm operation: calling ComplexOverlap (for 22 vs H1) =========="

    plt.figure(3)
    hlms = hlmoff(Psig, Lmax)
    for pair1 in rholms_intp['V1']:
            hxx = lalsim.SphHarmFrequencySeriesGetMode(hlms, pair1[0], pair2[0])
            rho, rhoTS, rhoIdx, rhoPhase = IPOverlap.ip(hxx, data_dict['H1'])
            tvals =  rhoTS.deltaT*np.arange(len(rhoTS.data.data))
            plt.plot(tvals,np.abs(rhoTS.data.data),label="Q(t)[IP]"+H1)
    plt.legend()
    plt.show()

    print " ======= Plotting rholm timeseries (NOT timeeshifted; are rho's offset correctly?)  =========="
    # plot the raw rholms
    plt.figure(1)
    for det in detectors:
        rhonow = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        npts = len(rhonow.data.data)
#        t = map(lambda x: x if x<npts*rhonow.deltaT/2 else x-npts*rhonow.deltaT, rhonow.deltaT*np.arange(npts))  # center at t=0
        t = rhonow.deltaT*np.arange(npts)
        plt.plot(t,np.abs(rhonow.data.data),label=det)
    # Plot the interpolated rholms
#    tt = np.arange(float(data_dict['H1'].epoch),P.tref ,1/500.) # Create a finer array of time steps. BE VERY CAREFUL - resampling generates huge arrays. BE CAREFUL not to extrapolate too far outside range
    tt = np.arange(0,rhonow.deltaT*npts ,1/5000.) # Create a finer array of time steps. BE VERY CAREFUL - resampling generates huge arrays. BE CAREFUL not to extrapolate too far outside range
    plt.figure(1)
    rhointpH = rholms_intp['H1'][(2,2)]
    rhointpL = rholms_intp['L1'][(2,2)]
    rhointpV = rholms_intp['V1'][(2,2)]
    plt.plot(tt,np.abs(rhointpH(tt)), label='H(2,2)')
    plt.plot(tt,np.abs(rhointpL(tt)), label='L(2,2)')
    plt.plot(tt,np.abs(rhointpV(tt)), label='V(2,2)')
    plt.legend()

    plt.figure(2)
    rhointpH = rholms_intp['H1'][(2,-2)]
    rhointpL = rholms_intp['L1'][(2,-2)]
    rhointpV = rholms_intp['V1'][(2,-2)]
    plt.plot(tt,np.abs(rhointpH(tt)), label='H(2,-2)')
    plt.plot(tt,np.abs(rhointpL(tt)), label='L(2,-2)')
    plt.plot(tt,np.abs(rhointpV(tt)), label='V(2,-2)')
    plt.legend()

    plt.show()


    # constraint1 = 0
    # for det in detectors:
    #     npts = len(hxx.data.data)
    #     t= hxx.deltaT*np.arange(npts)
    #     for i in np.arange(len(hxx.data.data)):
    #         constraint1+= np.abs(rholms_intp[det][(2,2)](t[i])-np.conj(rholms_intp[det][(2,-2)](t[i])))**2
    # print "   : Reflection symmetry constraint (Q22,Q2-2) with interpolation", constraint1/len(t)    # error per point 
    # print "   : Example  of complex conjugate quantities in interpolation ", rholms_intp['H1'][(2,2)](0.), rholms_intp['H1'][(2,-2)](0.)





if checkResults == True:
    print " ======= rholm test: Plot the lnLdata timeseries at the injection parameters (* STILL TIME OFFSET *)  =========="
    tmin = np.max(float(epoch_post - theEpochFiducial),tWindowReference[0]+0.03)   # the minimum time used is set by the rolling condition
    tvals = np.linspace(tmin,tWindowReference[1],4*fSample*(tWindowReference[1]-tmin))
    for det in detectors:
        lnLData = map( lambda x: SingleDetectorLogLikelihoodData(theEpochFiducial,rholms_intp, theEpochFiducial+x, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,Psig.psi, Psig.dist, 2, det), tvals)
        lnLDataEstimate = np.ones(len(tvals))*rhoExpected[det]*rhoExpected[det]
        plt.figure(1)
        tvalsPlot = tvals 
        plt.plot(tvalsPlot, lnLData,label='Ldata(t)+'+det)
        plt.plot(tvalsPlot, lnLDataEstimate,label="$rho^2("+det+")$")
    tEventRelative =float( Psig.tref - theEpochFiducial)
    print " Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is ", tEventRelative
    plt.plot([tEventFiducial,0],[tEventFiducial,rho2Net], color='k',linestyle='--')
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
    plt.plot(tvalsPlot, lnL,label='lnL(t)+'+det)
    plt.plot(tvalsPlot, lnLEstimate,label="$rho^2("+det+")$")
    tEventRelative =float( Psig.tref - theEpochFiducial)
    print " Real time (relative to fiducial start time) ", tEventFiducial,  " and our triggering time is the same ", tEventRelative
    plt.plot([tEventFiducial,0],[tEventFiducial,rho2Net], color='k',linestyle='--')
    plt.title("lnL (interpolated) vs narrow time interval")
#    plt.legend()
    plt.show()


    # print " ======= End to end LogL: Maximize in polarization and time (*)  =========="
    # lnLmax = scipy.optimize.fmin( (lambda (x,y) : SingleDetectorLogLikelihoodData(rholms_intp, x+theEpochFiducial, Psig.phi, Psig.theta, Psig.incl, Psig.phiref,y, Psig.dist, 2, det) if y>0 and y<lal.LAL_PI and x>0 and x<1/Psig.deltaF else 0), [0.001, Psig.psi],maxiter=1000)
    # print "  : Evan after optimizing in phase and time ", lnLmax


        

    


