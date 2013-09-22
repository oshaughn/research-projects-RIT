"""
Simple program to test the precomputation of inner product factors
appearing in the likelihood
"""

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>"

from factored_likelihood import *
from matplotlib import pylab as plt


#
# Set by user
#
checkResults = True # Turn on to print/plot output; Turn off for testing speed
checkInputPlots = False
checkResultsPlots = True

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
distanceFiducial = 25.  # Make same as reference



# psd_dict['H1'] = lal.LIGOIPsd
# psd_dict['L1'] = lal.LIGOIPsd
# psd_dict['V1'] = lal.LIGOIPsd
psd_dict['H1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['L1'] = lalsim.SimNoisePSDiLIGOSRD
psd_dict['V1'] = lalsim.SimNoisePSDiLIGOSRD



m1 = 4*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
ampO =0 # sets which modes to include in the physical signal
Lmax = 2  # sets which modes to include in the output
fref = 100
Psig = ChooseWaveformParams(fmin = fminWaves, radec=True, incl=0.0,phiref=0.0, theta=0.2, phi=0,psi=0.0,
         m1=m1,m2=m2,
         ampO=ampO,
         fref=fref,
         deltaT=1./fSample,
        detector='H1', dist=distanceFiducial*1.e6*lal.LAL_PC_SI)
df = findDeltaF(Psig)
Psig.deltaF = df
Psig.print_params()
print " ======= Generating synthetic data in each interferometer (manual timeshifts) =========="
t0 = Psig.tref
Psig.tref = t0 + ComputeTimeDelay('H1', Psig.theta,Psig.phi,Psig.tref)
data_dict['H1'] = non_herm_hoff(Psig)
Psig.detector = 'L1'
Psig.tref = t0 + ComputeTimeDelay('L1', Psig.theta,Psig.phi,Psig.tref)
data_dict['L1'] = non_herm_hoff(Psig)
Psig.detector = 'V1'
Psig.tref = t0 + ComputeTimeDelay('V1', Psig.theta,Psig.phi,Psig.tref)
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
    print det, rhoDet, rhoDet2, " at epoch ", float(data_dict[det].epoch), " with time delay ", ComputeTimeDelay(det, Psig.theta,Psig.phi,Psig.tref)
print "Network : ", np.sqrt(rho2Net)

if checkInputPlots:
    print " == Plotting detector data (time domain; requires regeneration, MANUAL TIMESHIFTS,  and seperate code path! Argh!) == "
    P = Psig.copy()
    for det in detectors:
        P.detector=det
        P.tref += ComputeTimeDelay(det, P.theta,P.phi,P.tref)
        hT = hoft(P)
        tvals = P.tref + hT.deltaT*np.arange(len(hT.data.data))
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
         dist=100*1.e6*lal.LAL_PC_SI,
         deltaF=df)
rholms_intp, crossTerms, rholms = PrecomputeLikelihoodTerms(P, data_dict, psd_dict, Lmax, analyticPSD_Q)


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
        lnLModel = SingleDetectorLogLikelihoodModel(crossTerms, P.tref, Psig.phi, Psig.theta, P.incl, P.phiref, Psig.psi, Psig.dist, 2, det)
        print det, lnLModel, np.sqrt(-2*lnLModel), rhoExpected[det]


    print " ======= rholm complex conjugation check (22 and 2-2 modes only) =========="
    constraint1 = 0
    for det in detectors:
        hxx = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        hyy = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, -2)
        for i in np.arange(len(hxx.data.data)):
            constraint1+= np.abs(hxx.data.data[i]-np.conj(hyy.data.data[i]))**2
    print "   : Reflection symmetry constraint (Q22,Q2-2) with raw data: : 0 ~= ", constraint1/len(hxx.data.data)    # error per point 

    # constraint1 = 0
    # for det in detectors:
    #     npts = len(hxx.data.data)
    #     t= hxx.deltaT*np.arange(npts)
    #     for i in np.arange(len(hxx.data.data)):
    #         constraint1+= np.abs(rholms_intp[det][(2,2)](t[i])-np.conj(rholms_intp[det][(2,-2)](t[i])))**2
    # print "   : Reflection symmetry constraint (Q22,Q2-2) with interpolation", constraint1/len(t)    # error per point 
    # print "   : Example  of complex conjugate quantities in interpolation ", rholms_intp['H1'][(2,2)](0.), rholms_intp['H1'][(2,-2)](0.)

    print " ======= rholm test: interpolation check (2,2) mode: data vs timesampling =========="
    constraint1 = 0
    for det in detectors:
        hxx = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
        npts = len(hxx.data.data)
        t= hxx.deltaT*np.arange(npts)
#        t = map(lambda x: x if x<npts*hxx.deltaT/2 else x-npts*hxx.deltaT, hxx.deltaT*np.arange(npts))  # center at t=0
        for i in np.arange(len(hxx.data.data)):
            constraint1+= np.abs(hxx.data.data[i]-rholms_intp[det][(2,2)](t[i]))**2
        print "   : Quality of interpolation per point : 0 ~= ", constraint1/len(hxx.data.data)

        
    # print " ======= rholm test: Epochs and timing =========="
    # for det in detectors:
    #     for pair1 in rholms_intp['V1']:
    #         hxx = lalsim.SphHarmTimeSeriesGetMode(rholms[det], 2, 2)
    #         print det, pair1, float(hxx.epoch), float(hxx.deltaT)



    print " ======= rholm test: Recover the SNR of the injection at the injection parameters (*)  =========="
    for det in detectors:
        lnLData = SingleDetectorLogLikelihoodData(rholms_intp, P.tref, P.theta,P.phi, P.incl, P.phiref,P.psi, P.dist, 2, det)
        print det, lnLData, np.sqrt(lnLData), rhoExpected[det]

    print " ======= rholm test: Plot the lnLdata timeseries at the injection parameters (*)  =========="
    tvals = np.linspace(0,10,3000)
    for det in detectors:
        lnLData = map( lambda x: SingleDetectorLogLikelihoodData(rholms_intp, x, P.theta,P.phi, P.incl, P.phiref,P.psi, P.dist, 2, det), tvals)
        plt.figure(1)
        plt.plot(tvals, lnLData,label='Ldata(t)+'+det)
    plt.legend()
    plt.show()

    

if checkResultsPlots == True:

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

    plt.figure(2)
    myfft = np.fft.ifft(rhoIntegrand0)
    tvals = np.linspace(0,1/h22.deltaF,len(myfft))
    plt.plot(tvals,np.abs(myfft),label='Q20(t)')
    plt.legend()
    plt.show()  # Show at the same time as the others


    print " ======= rholm operation: calling ComplexOverlap (for 22 vs H1) =========="

    plt.figure(3)
    for pair1 in rholms_intp['V1']:
            hxx = lalsim.SphHarmFrequencySeriesGetMode(hlms, pair1[0], pair2[0])
            rho, rhoTS, rhoIdx, rhoPhase = IPOverlap.ip(hxx, data_dict['H1'])
            plt.plot(tvals,np.abs(rhoTS.data.data),label="Q(t)[IP]")
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


    # plt.figure(3)
    # plt.plot(fvals,np.abs(h22.data.data),label='h22(f)')
    # plt.legend()

    plt.figure(2)
    rhointpH = rholms_intp['H1'][(2,-2)]
    rhointpL = rholms_intp['L1'][(2,-2)]
    rhointpV = rholms_intp['V1'][(2,-2)]
    plt.plot(tt,np.abs(rhointpH(tt)), label='H(2,-2)')
    plt.plot(tt,np.abs(rhointpL(tt)), label='L(2,-2)')
    plt.plot(tt,np.abs(rhointpV(tt)), label='V(2,-2)')
    plt.legend()

    plt.show()

    # plt.figure(1)
    # plt.title("H1 $rho{lm}$s")
    # det = 'H1'
    # rho22intp = rholms_intp[det][(2,2)]
    # rho2m2intp = rholms_intp[det][(2,-2)]
    # rho21intp = rholms_intp[det][(2,1)]
    # rho2m1intp = rholms_intp[det][(2,-1)]
    # rho20intp = rholms_intp[det][(2,0)]
    # plt.plot(tt, np.abs(rho22intp(tt)), 'b--', label='(2,2)')
    # plt.plot(tt, np.abs(rho2m2intp(tt)), 'b.', label='(2,-2)')
    # plt.plot(tt, np.abs(rho21intp(tt)), 'r--', label='(2,1)')
    # plt.plot(tt, np.abs(rho2m1intp(tt)), 'r.', label='(2,-1)')
    # plt.plot(tt, np.abs(rho20intp(tt)), 'k-', label='(2,0)')
    # plt.legend()
    # plt.show()

    # plt.figure(2)
    # plt.title('L1 $\rho_{lm}$s')
    # det = 'L1'
    # rho22intp = rholms_intp[det][(2,2)]
    # rho2m2intp = rholms_intp[det][(2,-2)]
    # rho21intp = rholms_intp[det][(2,1)]
    # rho2m1intp = rholms_intp[det][(2,-1)]
    # rho20intp = rholms_intp[det][(2,0)]
    # plt.plot(tt, np.abs(rho22intp(tt)), 'b--', label='(2,2)')
    # plt.plot(tt, np.abs(rho2m2intp(tt)), 'b.', label='(2,-2)')
    # plt.plot(tt, np.abs(rho21intp(tt)), 'r--', label='(2,1)')
    # plt.plot(tt, np.abs(rho2m1intp(tt)), 'r.', label='(2,-1)')
    # plt.plot(tt, np.abs(rho20intp(tt)), 'k-', label='(2,0)')
    # plt.legend()

    # plt.figure(3)
    # plt.title('V1 $\rho_{lm}$s')
    # det = 'V1'
    # rho22intp = rholms_intp[det][(2,2)]
    # rho2m2intp = rholms_intp[det][(2,-2)]
    # rho21intp = rholms_intp[det][(2,1)]
    # rho2m1intp = rholms_intp[det][(2,-1)]
    # rho20intp = rholms_intp[det][(2,0)]
    # plt.plot(tt, np.abs(rho22intp(tt)), 'b--', label='(2,2)')
    # plt.plot(tt, np.abs(rho2m2intp(tt)), 'b.', label='(2,-2)')
    # plt.plot(tt, np.abs(rho21intp(tt)), 'r--', label='(2,1)')
    # plt.plot(tt, np.abs(rho2m1intp(tt)), 'r.', label='(2,-1)')
    # plt.plot(tt, np.abs(rho20intp(tt)), 'k-', label='(2,0)')
    # plt.legend()

#    plt.show()
