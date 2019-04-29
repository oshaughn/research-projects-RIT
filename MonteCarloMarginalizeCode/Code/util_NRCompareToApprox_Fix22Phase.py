#!/usr/bin/env python
#
# GOAL
#    - plot strain h_{lm}(f) per mode from *two* approximants or two simulations
#    - scale to specific masses (or use dimensionless if M=1).
#    - compute overlaps pair by pair
#
# BONUS USES
#    - validate tapering (as needed)
#
# USAGE
#   util_NRCompareSimulations.py --lmax 3
#   util_NRCompareSimulations.py --lmax 3 --no-maximize
#
#  python util_NRCompareSimulations.py --parameter s1z --parameter-value 0 --use-rom3 --use-provided-strain2
#
#  python util_NRCompareSimulations.py --group Sequence-SXS-All --param 1 --group2 Sequence-RIT-Generic --param2 U0_D9.53_q1.00_a0.0_n100 --use-perturbative-extraction --mass 70 --parameter s1z --parameter-value 0 --fmin 30 --save-plots
#
#  python util_NRCompareSimulations.py --use-perturbative-extraction-full --use-provided-strain --use-spec-lev 6 --group Sequence-SXS-All --param BBH_SKS_d14.3_q1.22_sA_0_0_0.330_sB_0_0_-0.440 --group2 Sequence-RIT-Generic  --param2 D12.25_q0.82_a-0.44_0.33_n120 --use-interpolated-overlap --approx SEOBNRv2 --parameter s1z --parameter-value 0.33 --parameter s2z --parameter-value -0.44 --mass 70 --fmin 20 --save-plots --plot-dimensionless-strain --use-hybrid2
#
# util_NRCompareSimulations.py --group Sequence-SXS-All --param 1 --group2 Sequence-RIT-Generic --param2 U0_D9.53_q1.00_a0.0_n100 --use-perturbative-extraction --mass 70 --parameter s1z --parameter-value 0 --fmin 30 --save-plots --plot-dimensionless-strain  --approx SEOBNRv2 --use-provided-strain
#
#


import argparse
import numpy as np

import lalsimutils
import lalsimulation as lalsim
import lal
import sys

import NRWaveformCatalogManager as nrwf

try:
    import ROMWaveformManager as romwf
    useROM=True
except:
    print " - no ROM - "
    useROM=False


parser = argparse.ArgumentParser()
parser.add_argument("--group", default="Sequence-SXS-All",help="inspiral XML file containing injection information.")
parser.add_argument("--param", default="1",help="Parameter value")
parser.add_argument("--align-22",action='store_true',help='Find the optimal time and phase for alignment for the 22 mode [then halt]')
parser.add_argument("--use-interpolated-overlap",action='store_true', help="The overlap uses a local quadratic approximation near the peak, not just the maximum value")
parser.add_argument("--use-perturbative-extraction", action='store_true')
parser.add_argument("--use-perturbative-extraction-full", action='store_true')
parser.add_argument("--use-provided-strain", action='store_true')
parser.add_argument("--use-provided-strain2", action='store_true')
parser.add_argument("--use-hybrid",action='store_true')
parser.add_argument("--use-hybrid-method",default='taper_add')
parser.add_argument("--use-hybrid2-method",default='taper_add')
parser.add_argument("--use-hybrid2",action='store_true')
parser.add_argument("--use-rom3", action='store_true')  # show ROM as well.  Default ROM being shown.
parser.add_argument("--use-spec-lev",type=int,default=5)
parser.add_argument("--approx", default="SEOBNRv2", help="approximant to use for comparison")
parser.add_argument("--manual-MOmega0",type=float,default=None,help="Manual MOmega0, used to standardize reconstruction across simulations of different length.")
parser.add_argument("--fmin",default=30,type=float)
parser.add_argument("--mass", type=float,default=70.0,help="Total mass in solar masses")  # 150 turns out to be ok for Healy et al sims
parser.add_argument("--parameter", action='append', help='Explicit list of parameters to use')
parser.add_argument("--parameter-value", action="append",help="Explicit list of parameter values to use")
parser.add_argument("--srate-factor",type=int,default=1,help="Factor to increase srate: 1=16384kHz, 2=2*16384kHz, ...")
parser.add_argument("--lmax", default=2, type=int)
parser.add_argument("--rextr1", default=None,type=int)
parser.add_argument("--rextr2", default=None,type=int)
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--print-group-list",default=False,action='store_true')
parser.add_argument("--print-param-list",default=False,action='store_true')
parser.add_argument("--plot-dimensionless-strain",action='store_true')
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
parser.add_argument("--no-maximize", default=False, action='store_true')
parser.add_argument("--psd",default="SimNoisePSDaLIGOZeroDetHighPower")
parser.add_argument("--psd-file",default=None,help="PSD file (assumed for hanford)")
opts = parser.parse_args()



bNoInteractivePlots=True # default
fig_extension = '.png'
try:
    import matplotlib
    print " Matplotlib backend ", matplotlib.get_backend()
    if matplotlib.get_backend() is 'MacOSX':
        if opts.save_plots:
            print "  OSX without interactive plots"
            bNoInteractivePlots=True
            fig_extension='.jpg'
        else:  #  Interactive plots
            print "  OSX with interactive plots"
            bNoInteractivePlots=False
    elif matplotlib.get_backend() is 'agg':
        fig_extension = '.png'
        bNoInteractivePlots=True
        print " No OSX; no interactive plots "
    else:
        print " Unknown configuration "
        fig_extension = '.png'
        bNoInteractivePlots =True
    from matplotlib import pyplot as plt
    bNoPlots=False
except:
    from matplotlib import pyplot as plt
    fig_extension = '.png'
    print " - no matplotlib - "
    bNoInteractivePlots = False
    bNoPlots = True



group = opts.group
if not nrwf.internal_ParametersAreExpressions[group]:
    param = opts.param
else:
    param = eval(str(opts.param))

# Check for SXS 
if opts.use_spec_lev:
  if group == "Sequence-SXS-All":
    nrwf.internal_FilenamesForParameters[group][param] =nrwf.internal_FilenamesForParameters[group][param].replace("Lev5", "Lev" +str(opts.use_spec_lev))
    print " OVERRIDE OF SXS LEVEL : ", nrwf.internal_FilenamesForParameters[group][param]


if opts.print_group_list:
    print "Simulations available"
    for key in  nrwf.internal_ParametersAvailable.keys():
        print  "  ", key
    sys.exit(0)

if opts.print_param_list:
    print "Parameters available for ", group
    for key in  nrwf.internal_ParametersAvailable[group]:
        print  "  ", key
    sys.exit(0)


lmax = opts.lmax
T_window = 16.
df = 1./T_window
fmin =opts.fmin
fmaxSNR=1700
analyticPSD_Q=True
psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower
if opts.psd_file:
    analyticPSD_Q=False
    print "Reading PSD for instrument %s from %s" % ("H1", opts.psd_file)
    psd = lalsimutils.load_resample_and_clean_psd(opts.psd_file, "H1", df)
elif opts.psd and hasattr(lalsim, opts.psd):
    psd = getattr(lalsim, opts.psd)

###
### Generate waveforms
###

wfP = nrwf.WaveformModeCatalog(opts.group, param, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, extraction_radius=opts.rextr1,lmax=lmax,align_at_peak_l2_m2_emission=True,perturbative_extraction=opts.use_perturbative_extraction,perturbative_extraction_full=opts.use_perturbative_extraction_full,use_provided_strain=opts.use_provided_strain,reference_phase_at_peak=True,manual_MOmega0=opts.manual_MOmega0)
mtotOrig = (wfP.P.m1+wfP.P.m2)/lal.MSUN_SI
mtotParam = opts.mass
wfP.P.m1 *= mtotParam/mtotOrig
wfP.P.m2 *= mtotParam/mtotOrig
wfP.P.dist =100*1e6*lal.PC_SI
wfP.P.approx =  lalsim.GetApproximantFromString(opts.approx) # lalsim.EOBNRv2HM
wfP.P.deltaF = 1./T_window
wfP.P.deltaT = 1./(16384*opts.srate_factor)
#wfP.P.deltaT = 1./4096
wfP.P.fmin=opts.fmin  # 5 Hz, because I will be dealing with long waveforms at high mass
wfP.P.print_params()
hlmF_NR1 = wfP.hlmoff(  deltaT=wfP.P.deltaT,force_T=T_window,hybrid_use=opts.use_hybrid,hybrid_method=opts.use_hybrid_method)  # force 8 second window -- fine for NR
fvals_NR1 = lalsimutils.evaluate_fvals(hlmF_NR1[(2,2)])




# ROM case, if needed
if useROM and opts.use_rom3:
    acatHere = romwf.WaveformModeCatalog('gwsurrogate-0.9.3/surrogate_downloads/', 'NRHybSur3dq8')
    print " Default ROM (EOB), using NR parameters from NR1"
    hlmT_ROM = acatHere.hlmoft(wfP.P, use_basis=False,force_T=T_window)
    hlmF_ROM = acatHere.hlmoff(wfP.P, use_basis=False,force_T=T_window)  # Must force duration consistency, very annoying
    print " Duration test ", 1./hlmF_NR1[(2,2)].deltaF, 1./hlmF_ROM[(2,2)].deltaF
    print " Number of points test ", hlmF_NR1[(2,2)].data.length, hlmF_ROM[(2,2)].data.length  # Problem: padding is not consistent
else:
    hlmF_ROM = None




#print "These pointers should be distinct", hlmF_NR1[(2,2)],  hlmF_NR2[(2,2)]


if opts.approx == 'SEOBNRv2' or opts.approx == 'EOBNRv2HM':
    wfP.P.s1x=0
    wfP.P.s1y=0
    wfP.P.s2x=0
    wfP.P.s2y=0
if wfP.P.extract_param('eta') > 0.249999:
    wfP.P.m2 *= 0.9999
wfP.P.fmin = 10  # want this to be low for sanity
wfP.P.approx = lalsim.GetApproximantFromString(opts.approx)
wfP.P.print_params()
print " based on approx ", opts.approx

# LAL fourier f (based on case 1)
hlmF_lal = lalsimutils.hlmoff(wfP.P,lmax)
fvals_lal = lalsimutils.evaluate_fvals(hlmF_lal[(2,2)])

# Reference calculation with approximant, NO PADDING, to get true IMR duration
print "  Extracting true signal duration, without padding, for plotting purposes "
P1 = wfP.P.manual_copy()
P1.deltaF = None
tmpC = lalsimutils.complex_hoft(P1)
T_duration_true = -float(tmpC.epoch)


###
### Norms and overlaps
###



# Create overlaps
if not opts.no_maximize:
    IP = nrwf.CreateCompatibleComplexOverlap(hlmF_NR1,psd=psd,fLow=opts.fmin,fMax=fmaxSNR,analyticPSD_Q=analyticPSD_Q)
else:
    IP = nrwf.CreateCompatibleComplexIP(hlmF_NR1,psd=psd,fLow=opts.fmin,fMax=fmaxSNR,analyticPSD_Q=analyticPSD_Q)




# Per-mode SNR and Overlaps
print " Overlap table. WARNING; All modes optimized independently in time! "

# FIND APPROPRIATE 22 MAXIMUM TIME AND PHASE
IP.full_output=True
rho_ref, rhoData_ref, rhoIdx_ref, rhoPhase_ref = IP.ip(hlmF_NR1[(2,2)], hlmF_lal[(2,2)])
print "Reference : ", rhoData_ref[rhoIdx_ref]/rho_ref, rhoPhase_ref

print " mode  SNR_NR SNR_NR2  SNR_lal   overlap(NR1,NR2)  |overlap(NR1,NR2)| "
for mode in hlmF_NR1.keys():
  if mode in hlmF_lal.keys():
        rho_NR1 = IP.norm(hlmF_NR1[mode])
        rho_lal = IP.norm(hlmF_lal[mode])
        rho, rhoData, rhoIdx, rhoPhase = IP.ip(hlmF_NR1[mode],hlmF_lal[mode])
        inner_12 = rhoData[rhoIdx_ref] * np.exp(-1j*rhoPhase_ref*mode[1]/2)/rho_NR1/rho_lal
#        inner_12 = IP.ip(hlmF_NR1[mode],hlmF_NR2[mode])/rho_NR2/rho_NR1
        print mode, rho_NR1,rho_lal,  inner_12, np.abs(inner_12)





###
### Plot dimensionless hlm(t)
###     note that HYBRIDIZATION IS NOT SHOWN HERE
###
if not opts.plot_dimensionless_strain:
    sys.exit(0)


fig_indx =6
fig_list = []

plt.figure(3); #plt.legend()
#plt.figure(6); #plt.legend(); 
#plt.figure(7);# plt.legend();
if not bNoPlots:
    if not bNoInteractivePlots:
        plt.show()
    else:
        for indx in fig_list:
            print "Writing figure ", indx
            plt.figure(indx); plt.legend(); plt.savefig("nr-dimensionless-comparison-" +str(indx)+fig_extension)
            plt.xlim(tPeak-500,tPeak+200); plt.savefig("nr-dimensionless-comparison-detail-" +str(indx)+fig_extension)

###
### Plot unit-specific hlm(t)
###

print " Writing hlm(t) "
fig_indx =60
fig_list = []
for mode in [(2,2), (2,-2), (2,1),(3,2),(3,0),(2,0), (3,3), (4,4), (4,3), (4,2), (4,1), (4,0), (5,5), (5,4), (5,3), (5,2), (5,1), (5,0)]:
 if mode in hlmF_NR1.keys()  and mode in hlmF_lal.keys():
    hlmF_NR1_now = hlmF_NR1[mode]
    hlmF_lal_now = hlmF_lal[mode]
    fvals = lalsimutils.evaluate_fvals(hlmF_NR1_now)
    plt.figure(fig_indx)
    fig_indx+=1
    plt.plot(fvals,np.abs(hlmF_NR1_now.data.data))
    plt.plot(fvals,np.abs(hlmF_lal_now.data.data))
    plt.xlim(-300,300)

    print "Writing figure ", fig_indx
    plt.savefig("nr-strain-comparison-fourier-"+str(mode)+".png")


    hlmT_NR1_now = lalsimutils.DataInverseFourier(hlmF_NR1_now)
    hlmT_lal_now = lalsimutils.DataInverseFourier(hlmF_lal_now)
    tvals = lalsimutils.evaluate_tvals(hlmT_NR1_now) - float(wfP.P.tref)
    tvals_lal = lalsimutils.evaluate_tvals(hlmT_lal_now) - float(wfP.P.tref)

    # Apply time and phase shift
    # Phase shift is easy
    tRef22_NR = tvals[np.argmax(hlmT_NR1_now.data.data)]
    tRef22_lal = tvals_lal[np.argmax(hlmT_lal_now.data.data)]
    z_NR = hlmT_NR1_now.data.data[np.argmax(hlmT_NR1_now.data.data)]
    z_lal = hlmT_lal_now.data.data[np.argmax(hlmT_lal_now.data.data)]

    hlmT_lal_now.data.data *= z_NR/z_lal * np.abs(z_lal/z_NR)
    # Time shift is annoying
    tvals_lal += tRef22_NR - tRef22_lal
                                  

    plt.figure(fig_indx)
    plt.clf()
    fig_list.append( fig_indx)
    fig_indx+=1
#    plt.plot(tvals , np.abs(hlmT_NR1_now.data.data),'r', label=str(mode))
    plt.plot(tvals , np.real(hlmT_NR1_now.data.data),'k', label=str(mode))
#    plt.plot(tvals_lal , np.abs(hlmT_lal_now.data.data),'b', label=str(mode))
    plt.plot(tvals_lal , np.real(hlmT_lal_now.data.data),'b', label=str(mode))
    plt.xlim(-T_duration_true,0.1)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain ($10^{-21}$)")

    print "Writing figure ", fig_indx
    plt.savefig("nr-strain-comparison-"+str(mode)+".png")


    plt.xlim(-0.5,0.1)

    print "Writing figure ", fig_indx
    plt.savefig("nr-strain-comparison-detail-"+str(mode)+".png")
