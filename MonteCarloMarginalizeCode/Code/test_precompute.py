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

Psig = ChooseWaveformParams(fmin = 10., radec=True, theta=1.2, phi=2.4,
        detector='H1', dist=100.*1.e6*lal.LAL_PC_SI)
df = findDeltaF(Psig)
Psig.deltaF = df
data_dict['H1'] = non_herm_hoff(Psig)
Psig.detector = 'L1'
data_dict['L1'] = non_herm_hoff(Psig)
Psig.detector = 'V1'
data_dict['V1'] = non_herm_hoff(Psig)

# Struct to hold template parameters
P = ChooseWaveformParams(fmin = 40., dist=100.*1.e6*lal.LAL_PC_SI, deltaF=df)
Lmax = 2 # sets which modes to include


#
# Perform the Precompute stage
#
rholms_intp, crossTerms, rholms = PrecomputeLikelihoodTerms(P, data_dict,
        psd_dict, Lmax, analyticPSD_Q)

#
# Examine and sanity check the output
#

if checkResults == True:
    # Print values of cross terms
    detectors = data_dict.keys()
    for det in detectors:
        for l in range(2,Lmax+1):
            for m in range(-l,l+1):
                for lp in range(2,Lmax+1):
                    for mp in range(-lp,lp+1):
                        print det, "(",l,",",m,") (",lp,",",mp,") term:",\
                                crossTerms[det][ ((l,m),(lp,mp)) ]

    # Plot the interpolated rholms
    tt = np.arange(0.,64.,5.e-6) # Create a finer array of time steps
    plt.figure(1)
    plt.title('H1 $\rho_{lm}$s')
    det = 'H1'
    rho22intp = rholms_intp[det][(2,2)]
    rho2m2intp = rholms_intp[det][(2,-2)]
    rho21intp = rholms_intp[det][(2,1)]
    rho2m1intp = rholms_intp[det][(2,-1)]
    rho20intp = rholms_intp[det][(2,0)]
    plt.plot(tt, np.abs(rho22intp(tt)), 'b--', label='(2,2)')
    plt.plot(tt, np.abs(rho2m2intp(tt)), 'b.', label='(2,-2)')
    plt.plot(tt, np.abs(rho21intp(tt)), 'r--', label='(2,1)')
    plt.plot(tt, np.abs(rho2m1intp(tt)), 'r.', label='(2,-1)')
    plt.plot(tt, np.abs(rho20intp(tt)), 'k-', label='(2,0)')
    plt.legend()

    plt.figure(2)
    plt.title('L1 $\rho_{lm}$s')
    det = 'L1'
    rho22intp = rholms_intp[det][(2,2)]
    rho2m2intp = rholms_intp[det][(2,-2)]
    rho21intp = rholms_intp[det][(2,1)]
    rho2m1intp = rholms_intp[det][(2,-1)]
    rho20intp = rholms_intp[det][(2,0)]
    plt.plot(tt, np.abs(rho22intp(tt)), 'b--', label='(2,2)')
    plt.plot(tt, np.abs(rho2m2intp(tt)), 'b.', label='(2,-2)')
    plt.plot(tt, np.abs(rho21intp(tt)), 'r--', label='(2,1)')
    plt.plot(tt, np.abs(rho2m1intp(tt)), 'r.', label='(2,-1)')
    plt.plot(tt, np.abs(rho20intp(tt)), 'k-', label='(2,0)')
    plt.legend()

    plt.figure(3)
    plt.title('V1 $\rho_{lm}$s')
    det = 'V1'
    rho22intp = rholms_intp[det][(2,2)]
    rho2m2intp = rholms_intp[det][(2,-2)]
    rho21intp = rholms_intp[det][(2,1)]
    rho2m1intp = rholms_intp[det][(2,-1)]
    rho20intp = rholms_intp[det][(2,0)]
    plt.plot(tt, np.abs(rho22intp(tt)), 'b--', label='(2,2)')
    plt.plot(tt, np.abs(rho2m2intp(tt)), 'b.', label='(2,-2)')
    plt.plot(tt, np.abs(rho21intp(tt)), 'r--', label='(2,1)')
    plt.plot(tt, np.abs(rho2m1intp(tt)), 'r.', label='(2,-1)')
    plt.plot(tt, np.abs(rho20intp(tt)), 'k-', label='(2,0)')
    plt.legend()

    plt.show()
