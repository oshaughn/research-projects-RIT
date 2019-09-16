"""
Simple program illustrating the precomputation step and calls to the
likelihood function.
"""

__author__ = "Evan Ochsner <evano@gravity.phys.uwm.edu>"

from factored_likelihood import *

Niter = 5 # Number of times to call likelihood function
Tmax = 38. # max ref. time
Tmin = 36. # min ref. time
Dmax = 110. * 1.e6 * lal.LAL_PC_SI # max ref. time
Dmin = 90. * 1.e6 * lal.LAL_PC_SI # min ref. time

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
data_dict['H1'] = lsu.non_herm_hoff(Psig)
Psig.detector = 'L1'
data_dict['L1'] = lsu.non_herm_hoff(Psig)
Psig.detector = 'V1'
data_dict['V1'] = lsu.non_herm_hoff(Psig)

# Struct to hold template parameters
P = ChooseWaveformParams(fmin = 40., dist=100.*1.e6*lal.LAL_PC_SI, deltaF=df)
Lmax = 2 # sets which modes to include

#
# Perform the Precompute stage
#
rholms_intp, crossTerms, rholms = PrecomputeLikelihoodTerms(P, data_dict,
        psd_dict, Lmax, analyticPSD_Q)
print("Finished Precomputation...")

#
# Call the likelihood function for various extrinsic parameter values
#
for i in range(Niter):
    P.phi = 2. * np.pi * np.random.rand() # right ascension
    P.theta = np.pi * np.random.rand() # declination
    P.tref = np.random.rand() * (Tmax - Tmin) # ref. time
    P.phiref = 2. * np.pi * np.random.rand() # ref. orbital phase
    P.incl = np.pi * np.random.rand() # inclination
    P.psi = np.pi * np.random.rand() # polarization angle
    P.dist = np.random.rand() * (Dmax - Dmin) # luminosity distance

    lnL = FactoredLogLikelihood(P, rholms_intp, crossTerms, Lmax)

    print("For (RA, DEC, tref, phiref, incl, psi, dist) =")
    print("\t", P.phi, P.theta, P.tref, P.phiref, P.incl, P.psi, P.dist)
    print("\tlog likelihood is:", lnL)
