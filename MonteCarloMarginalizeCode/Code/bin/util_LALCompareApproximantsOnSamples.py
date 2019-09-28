#! /usr/bin/env python
#
# GOAL
#  - Load in LI posterior samples
#  - generate waveforms using all the real parameters, at H and L
#  - compute overlaps due to two approximations
#
# ALSO USEFUL FOR COMPARISON
#  util_TestPrecessingProperties.py   # plots h(t), h(f) for precessing binaries, to illustrate their similarity
#  util_NRCompareSimulations.py  # overlaps


import argparse
import sys
import numpy as np
import scipy
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal

try:
    from matplotlib import pyplot as plt
except:
    print(" - no plots - ")

parser = argparse.ArgumentParser()
parser.add_argument("--approx",default="SEOBNRv2")
parser.add_argument("--approx2",default="SpinTaylorT4")
parser.add_argument("--use-precessing",action='store_true')
parser.add_argument("--fname-lalinference",help="filename of posterior_samples.dat file [standard LI output], to overlay on corner plots")
#parser.add_argument("--fmin",type=float,default=None)
parser.add_argument("--fisher-psd",type=str,default="SimNoisePSDaLIGOZeroDetHighPower",help="psd name (attribute in lalsimulation).  SimNoisePSDiLIGOSRD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--psd-file",  help="File name for PSD (assumed hanford). Overrides --fisher-psd if provided")
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=8., help="Default window size for processing.")
parser.add_argument("--fmax",default=1700,type=float)
parser.add_argument("--save-plots",action='store_true')
parser.add_argument("--verbose",action='store_true')
opts=  parser.parse_args()

# Handle PSD
# FIXME: Change to getattr call, instead of 'eval'
eff_fisher_psd = lalsim.SimNoisePSDiLIGOSRD
if not opts.psd_file:
    #eff_fisher_psd = eval(opts.fisher_psd)
    eff_fisher_psd = getattr(lalsim, opts.fisher_psd)   # --fisher-psd SimNoisePSDaLIGOZeroDetHighPower   now
    analyticPSD_Q=True
else:
    print(" Importing PSD file ", opts.psd_file)
    eff_fisher_psd = lalsimutils.load_resample_and_clean_psd(opts.psd_file, 'H1', 1./opts.seglen)
    analyticPSD_Q = False


print(" Loading samples ")
samples_in = np.genfromtxt(opts.fname_lalinference,names=True)
print(" Done loading samples ")
deltaT = 1./4096
T_window =int(opts.seglen)

# Create overlap 
P=lalsimutils.ChooseWaveformParams()
P.deltaT = deltaT
P.deltaF = 1./T_window
P.fmin = samples_in["flow"][0]
P.approx=lalsim.SpinTaylorT2
P.print_params()
print(lalsimutils.estimateDeltaF(P), P.deltaF)
if P.deltaF > lalsimutils.estimateDeltaF(P):
    sys.exit(0)
hfBase = lalsimutils.complex_hoff(P) # Does not matter what this is.
print(" Done generating waveform")
IP = lalsimutils.CreateCompatibleComplexOverlap(hfBase,analyticPSD_Q=analyticPSD_Q,psd=eff_fisher_psd,fMax=opts.fmax,interpolate_max=True)
print(" Done populating inner product")


# Similar code: convert_output_format_inference2ile
fac_reduce=1
for indx in np.arange(len(samples_in["m1"])):
#    print indx
    ###
    ###  Define the system
    ###
    m1 = samples_in["m1"][indx]*lal.MSUN_SI
    m2 = samples_in["m2"][indx]*lal.MSUN_SI
    d = samples_in["distance"][indx]*lal.PC_SI*1e6
    P = lalsimutils.ChooseWaveformParams(m1=m1,m2=m2,dist=d)
    P.taper =lalsim.SIM_INSPIRAL_TAPER_START
    P.radec=True
    P.theta = samples_in["dec"][indx]
    P.phi = samples_in["ra"][indx]
    if "time_maxl" in samples_in.dtype.names:
     P.time = samples_in["time_maxl"][fac_reduce*indx]
    else:
     P.time = samples_in["time"][indx]
    P.fmin = samples_in["flow"][indx]
    P.fref = samples_in["f_ref"][indx]  # a field now !  MAY NOT BE RELIABLE
    if "phi_orb" in samples_in.dtype.names:
     P.phiref = samples_in["phi_orb"][fac_reduce*indx]
    elif "phase_maxl" in samples_in.dtype.names:
     P.phiref =samples_in["phase_maxl"][fac_reduce*indx]
    elif "phase" in samples_in.dtype.names:
     P.phiref = samples_in["phase"][fac_reduce*indx]
    else:
     print(samples_in.dtype.names)
     P.phiref = 0  # does not actually matter
    P.approx = lalsim.GetApproximantFromString(opts.approx)
    if "phi_jl" in samples_in.dtype.names and 'theta1' in samples_in.dtype.names:
      P.init_via_system_frame( 
         thetaJN=samples_in["theta_jn"][fac_reduce*indx],
         phiJL=samples_in["phi_jl"][fac_reduce*indx],
         theta1=samples_in["tilt1"][fac_reduce*indx],
         theta2=samples_in["tilt2"][fac_reduce*indx],
         phi12=samples_in["phi12"][fac_reduce*indx],
         chi1=samples_in["a1"][fac_reduce*indx],
         chi2=samples_in["a2"][fac_reduce*indx],
         psiJ=samples_in["psi"][fac_reduce*indx]   # THIS IS NOT BEING SET CONSISTENTLY...but we marginalize over it, so that's ok
         )
    elif P.approx == lalsim.SEOBNRv2 or  P.approx == lalsimutils.lalSEOBv4 or P.approx == lalsimutils.lalIMRPhenomD or P.approx == lalsim.IMRPhenomC:
        # Aligned spin model
        P.s1z = samples_in["a1z"][fac_reduce*indx]
        P.s2z = samples_in["a2z"][fac_reduce*indx]
        P.psi = samples_in["psi"][fac_reduce*indx]
        if "theta_jn" in samples_in.dtype.names:
                P.incl = samples_in["theta_jn"][fac_reduce*indx]

    elif 'theta1' in 'theta1' in samples_in.dtype.names:
      P.init_via_system_frame( 
         thetaJN=samples_in["theta_jn"][fac_reduce*indx],
         phiJL=0, # does not matter
         theta1=samples_in["theta1"][fac_reduce*indx],
         theta2=samples_in["theta2"][fac_reduce*indx],
         phi12=samples_in["phi12"][fac_reduce*indx],
         chi1=samples_in["a1"][fac_reduce*indx],
         chi2=samples_in["a2"][fac_reduce*indx],
         psiJ=samples_in["psi"][fac_reduce*indx]   # THIS IS NOT BEING SET CONSISTENTLY...but we marginalize over it, so that's ok
         )
    else:
        print(" Don't know how to handle this orientation for", opts.approx)

      ###
      ### Create two approximants
      ###
    P.deltaT = deltaT
    P.deltaF = 1./T_window
    if opts.verbose:
        P.print_params()

    P.approx = lalsim.GetApproximantFromString(opts.approx)
    P.fmax = opts.fmax
    if P.approx == lalsim.SEOBNRv2 or P.approx == lalsim.TaylorF2 or P.approx == lalsim.IMRPhenomC or P.approx==lalsim.IMRPhenomD :
        P.s1x = 0
        P.s2x = 0
        P.s1y=0
        P.s2y=0
    hF_1 = lalsimutils.complex_hoff(P)

    P.approx = lalsim.GetApproximantFromString(opts.approx2)
    hF_2 = lalsimutils.complex_hoff(P)

    nm_1 = IP.norm(hF_1)
    nm_2 = IP.norm(hF_2)
    match = IP.ip(hF_1,hF_2)/nm_1/nm_2
    print(indx, match)

    if opts.verbose and match < 0.95:
        P.print_params()

    ###
    ### Create time-domain plots of the two approximants
    ### [util_NRCompareSimulations.py is similar]
    ###
    if opts.save_plots:
        hT_1 = lalsimutils.DataInverseFourier(hF_1)
        hT_2 = lalsimutils.DataInverseFourier(hF_2)
        tvals = lalsimutils.evaluate_tvals(hT_1) - float(P.tref)
        
        # if I have an FD approximant, the times are not necessarily set correctly
        # if lalsim.SimInspiralImplementedFDApproximants(lalsim.GetApproximantFromString(opts.approx))==1:
        #     indx = np.argmax(hT_1.data.data)
        #     indx_0 = int(-tvals[0]/P.deltaT)
        #     print " Rolling by ", indx-indx_0, indx, indx_0
        #     hT_1.data.data = np.roll(hT_1.data.data, -indx+indx_0)
        # if lalsim.SimInspiralImplementedFDApproximants(lalsim.GetApproximantFromString(opts.approx2))==1:
        #     indx = np.argmax(hT_2.data.data)
        #     indx_0 = int(-tvals[0]/P.deltaT)
        #     hT_2.data.data = np.roll(hT_2.data.data, -indx+indx_0)
        #     print " Rolling by ", indx-indx_0,indx,indx_0

        plt.plot(tvals, np.real(hT_1.data.data),label=opts.approx)
        plt.plot(tvals, np.abs(hT_1.data.data),label=opts.approx)
        plt.plot(tvals, np.real(hT_2.data.data),label=opts.approx2)
#        plt.xlim(-1, 0.1)
        plt.legend()
        plt.savefig("post_compare_"+opts.approx+"_"+opts.approx2+"_"+str(indx)+".png")
        plt.clf()
