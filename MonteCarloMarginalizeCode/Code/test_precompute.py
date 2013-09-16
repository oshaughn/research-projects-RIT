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

#
# Produce data with a coherent signal in H1, L1, V1
#
data_dict = {}
psd_dict = {}
analyticPSD_Q = True # For simplicity, using an analytic PSD
psd_dict['H1'] = lal.LIGOIPsd
psd_dict['L1'] = lal.LIGOIPsd
psd_dict['V1'] = lal.LIGOIPsd

m1 = 5*lal.LAL_MSUN_SI
m2 = 3*lal.LAL_MSUN_SI
Psig = ChooseWaveformParams(fmin = 30., radec=True, theta=1.2, phi=2.4,
         m1=m1,m2=m2,
        detector='H1', dist=25.*1.e6*lal.LAL_PC_SI)
df = findDeltaF(Psig)
Psig.deltaF = df
Psig.print_params()
print " ======= Generating synthetic data in each interferometer =========="
data_dict['H1'] = non_herm_hoff(Psig)
Psig.detector = 'L1'
data_dict['L1'] = non_herm_hoff(Psig)
Psig.detector = 'V1'
data_dict['V1'] = non_herm_hoff(Psig)

print " == Data report == "
detectors = data_dict.keys()
rho2Net = 0
for det in detectors:
    IP = ComplexIP(fLow=30, fNyq=2048,deltaF=df,psd=psd_dict[det])
    rhoDet = IP.norm(data_dict[det])
    rho2Net += rhoDet*rhoDet
    print det, rhoDet
print "Network : ", np.sqrt(rho2Net)

print " ======= Template specified: precomputing all quantities =========="
# Struct to hold template parameters
P = ChooseWaveformParams(fmin = 40., dist=100.*1.e6*lal.LAL_PC_SI, deltaF=df)
Lmax = 2 # sets which modes to include
#
# Perform the Precompute stage
#
rholms_intp, crossTerms, rholms = PrecomputeLikelihoodTerms(P, data_dict,
        psd_dict, Lmax, analyticPSD_Q)


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
                print det, pair1, pair2, crossTerms[det][pair1,pair2]

    print " ======= Plotting  results =========="
    # Plot the interpolated rholms
    tt = np.arange(0.,1./df ,1/500.) # Create a finer array of time steps. BE VERY CAREFUL - resampling generates huge arrays. BE CAREFUL not to extrapolate too far outside range
    plt.figure(1)
    rhointpH = rholms_intp['H1'][(2,2)]
    rhointpL = rholms_intp['L1'][(2,2)]
    rhointpV = rholms_intp['V1'][(2,2)]
    plt.plot(tt,np.abs(rhointpH(tt)), label='H(2,2)')
    plt.plot(tt,np.abs(rhointpL(tt)), label='L(2,2)')
    plt.plot(tt,np.abs(rhointpV(tt)), label='V(2,2)')
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
