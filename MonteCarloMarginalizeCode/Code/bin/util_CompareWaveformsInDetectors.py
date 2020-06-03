#! /usr/bin/env python
#
# GOAL
#    - evaluate overlap between two waveforms (two inj xml) in H, L
#    - prints out mismatch in H, L.
#    - no time maximization by default.  (We can do it)
#
# COMPARE TO
#   - util_NRCompareSimulations.py
#   - util_ManualOverlapGrid
#   - util_LALCompareApproximantsOnSamples.py
#
# DEMOS
#     * Compare two approximants with each other, on the same parameters
#     python util_CompareWaveformsInDetectors.py --inj1 inj.xml.gz --inj2 inj.xml.gz --approx SEOBNRv2 --approx2 SEOBNRv4
#     
#    * Compare one single waveform (e.g., NR best fits) with many others (e.g., output of precessing parameters)
#      python util_CompareWaveformsInDetectors.py --inj inj1.xml.gz --inj2 maxpt.xml.gz --approx SEOBNRv2 --group2 Sequence-SXS-All --param2 1 --use-provided-strain2
#
#   * Compare two NR waveforms that have the same parameters

from __future__ import print_function

import argparse
import numpy as np

import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import sys

import NRWaveformCatalogManager3 as nrwf
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--maximize",action='store_true',help="Perform overlap instead of faithfulness test. Important if inconsistent times")
parser.add_argument("--inj",default=None,type=str,help="Required. Arguments for 1. If used, NR will override intrinsic parameters")
parser.add_argument("--inj2",default=None,type=str,help="Required. Arguments for 2. If used, NR will override intrinsic parameters")
parser.add_argument("--group", default=None,help="inspiral XML file containing injection information.")
parser.add_argument("--param", default="1",help="Parameter value")
parser.add_argument("--group2", default=None,help="inspiral XML file containing injection information.")
parser.add_argument("--param2", default="1",help="Parameter value")
parser.add_argument("--align-22",action='store_true',help='Find the optimal time and phase for alignment for the 22 mode [then halt]')
parser.add_argument("--use-interpolated-overlap",action='store_true', help="The overlap uses a local quadratic approximation near the peak, not just the maximum value")
parser.add_argument("--use-perturbative-extraction", action='store_true')
parser.add_argument("--use-perturbative-extraction-full", action='store_true')
parser.add_argument("--use-provided-strain", action='store_true')
parser.add_argument("--use-provided-strain2", action='store_true')
parser.add_argument("--use-spec-lev",type=int,default=5)
parser.add_argument("--use-hybrid",action='store_true')
parser.add_argument("--use-hybrid-method",default='taper_add')
parser.add_argument("--use-hybrid2",action='store_true')
parser.add_argument("--use-hybrid2-method",default='taper_add')
parser.add_argument("--approx", default="EOBNRv2HM", help="approximant to use for comparison")
parser.add_argument("--approx2", default="EOBNRv2HM", help="approximant to use for comparison")
parser.add_argument("--lmax",default=2,type=int)
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen",type=int,default=16,help="Window time")
parser.add_argument("--fref",default=None,type=float,help="Reference frqeuency (assumed equal to fmin)")
parser.add_argument("--fmin",default=20,type=float,help="Minimum frequency for overlap integral -- NOT necessarily tied to tempalte starting frequencies!")
parser.add_argument("--fmin-template1",default=None,type=float,help="Override template frequency 1 (e.g., for NR XML files)")
parser.add_argument("--fmin-template2",default=None,type=float,help="Override template frequency 1 (e.g., for NR XML files)")
parser.add_argument("--fmax",default=2000,type=float,help="Maximum frequency in Hz, used for PSD integral.")
parser.add_argument("--psd-file",default=None,action='append',help="PSD file")
parser.add_argument("--psd",type=str,default="SimNoisePSDaLIGOZeroDetHighPower",help="psd name (attribute in lalsimulation).  SimNoisePSDiLIGOSRD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--fname-output", default="comparison_output.dat",type=str)
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",action="store_true")
parser.add_argument("--tref",default=None,type=float)

opts = parser.parse_args()


###
### Spacing settings, for overlaps
###


lmax = opts.lmax
T_window = opts.seglen
df = 1./T_window
fmin =opts.fmin  # default for now
# if opts.fmin_template1:
#     fmin = np.min([fmin, opts.fmin_template1])
# if opts.fmin_template2:
#     fmin = np.min([fmin, opts.fmin_template2])
fmaxSNR=1700
analyticPSD_Q=True

psd_dict = {}
ifo_list = []

psd=lalsim.SimNoisePSDaLIGOZeroDetHighPower
if opts.psd_file:
    analyticPSD_Q=False

    for inst, psdf in map(lambda c: c.split("="), opts.psd_file):
        if opts.verbose: 
            print("Reading PSD for instrument %s from %s" % (inst, psdf))
        psd_dict[inst] = lalsimutils.load_resample_and_clean_psd(psdf, inst, df)

    ifo_list = psd_dict.keys()

elif opts.psd and hasattr(lalsim, opts.psd):
    ifo_list = ['H1','L1']
    psd = getattr(lalsim, opts.psd)


fNyq = opts.srate/2.
IP=None
IP_list = {}
if opts.maximize:
    for ifo in ifo_list:
        IP_list[ifo]= lalsimutils.ComplexOverlap(fNyq=fNyq,deltaF=df,analyticPSD_Q=analyticPSD_Q,psd=psd_dict[ifo],fMax=opts.fmax,fLow=fmin)
else:
    for ifo in ifo_list:
        IP_list[ifo] = lalsimutils.ComplexIP(fNyq=fNyq,deltaF=df,analyticPSD_Q=analyticPSD_Q,psd=psd_dict[ifo],fMax=opts.fmax,fLow=fmin)

###
### Load injection XML
###

if opts.inj is None:
    print(" --inj required")
    sys.exit(0)
print(" Reading injection file 1 for comparison ", opts.inj)
P1_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj)
nlines1  = len(P1_list)
print(" Read  ", nlines1, " injections")

if nlines1 < 1:
    print(" No data in ", opts.inj)

tref = float( (P1_list[0]).tref ) # default
if not(opts.tref is None):
    tref = opts.tref

if opts.inj2 is None:
    print(" --inj2 required")
    sys.exit(0)
print(" Reading injection file 1 for comparison ", opts.inj2)
P2_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj2)
nlines2  = len(P2_list)

if nlines2 < 1:
    print(" No data in ", opts.inj2)




###
### Define functions to generate waveforms
###

def get_hF1(indx,ifo):
    global opts
    P = P1_list[indx % nlines1].manual_copy() # looping
    if not (opts.fmin_template1 is None):
        P.fmin = opts.fmin_template1
    if not (opts.fref is None):
        P.fref = opts.fref
    else:
        P.ref = P.fmin
    P.radec =True
    P.tref = P1_list[indx%nlines1].tref # copy, this is an allocated object
    P.deltaF = df
    P.deltaT = 1./opts.srate
    P.detector = ifo
    P.approx = lalsim.GetApproximantFromString(opts.approx)  # override the XML. Can screw you up (ref spins)
    # if P.approx == lalsim.IMRPhenomPv2:
    #     phiJL_now = P.extract_param('phiJL')
    #     P.assign_param('phiJL', phiJL_now-np.pi/2)  # Estimate
#        P.fref = 100
    if opts.verbose:
        P.print_params()
    hF = lalsimutils.non_herm_hoff(P)
    return hF

def get_hF2(indx,ifo):
    global opts
    P = P2_list[indx % nlines2].manual_copy() # looping
    if not (opts.fmin_template2 is None):
        P.fmin = opts.fmin_template2
    if not (opts.fref is None):
        P.fref = opts.fref
    else:
        P.ref = P.fmin
    P.radec =True
    P.tref = P2_list[indx%nlines2].tref # copy, this is an allocated object
    P.deltaF = df
    P.deltaT = 1./opts.srate
    P.detector = ifo
    P.approx = lalsim.GetApproximantFromString(opts.approx2)  # override the XML. Can screw you up
    if P.approx == lalsim.IMRPhenomPv2:
        cosbeta =np.cos(P.extract_param('beta'))
        my_phase = np.pi -np.pi/2
        dt = 20*(P.m1+P.m2)/lal.MSUN_SI * lalsimutils.MsunInSec  # ad hoc factor .. there is apparently a timeshift of Pv2 relative to PD of about 20 M
        P.tref += dt
        phiJL_now = P.extract_param('phiJL')
        psi_now = P.psi
        P.assign_param('phiJL', phiJL_now-my_phase)  # Estimate
        P.psi  = psi_now   # fix L alignment, not J
        P.phiref += my_phase 
        P.phiref += np.pi/2 
    if opts.verbose:
        P.print_params()
    hF = lalsimutils.non_herm_hoff(P)
    return hF


return_hF1 = get_hF1
return_hF2 = get_hF2

###
### Load NR (as needed) and redefine generation functions
###


group = opts.group
param = None
wfP =  None
if group in nrwf.internal_ParametersAreExpressions.keys():
    if not nrwf.internal_ParametersAreExpressions[group]:
        param = opts.param
    else:
        param = eval(str(opts.param))
    if opts.use_spec_lev:
            if group == "Sequence-SXS-All":
                nrwf.internal_FilenamesForParameters[group][param] =nrwf.internal_FilenamesForParameters[group][param].replace("Lev5", "Lev" +str(opts.use_spec_lev))
                print(" OVERRIDE OF SXS LEVEL : ", nrwf.internal_FilenamesForParameters[group][param])
    wfP = nrwf.WaveformModeCatalog(opts.group, param, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, lmax=lmax,align_at_peak_l2_m2_emission=True,perturbative_extraction=opts.use_perturbative_extraction,perturbative_extraction_full=opts.use_perturbative_extraction_full,use_provided_strain=opts.use_provided_strain,reference_phase_at_peak=True,quiet=True)
    wfP.P.fmin  = opts.fmin

    print(" WARNING: NR orientation parameters differ, so the waveform CANNOT be identical and will generally differ in at least orbital phase ")

    def get_hF1_NR(indx,ifo):
        P_here = P1_list[indx % nlines1]

        wfP.P.radec=True
        wfP.P.m1 = P_here.m1
        wfP.P.m2 = P_here.m2

        wfP.P.dist = P_here.dist
        wfP.P.incl = P_here.incl
        wfP.P.theta = P_here.theta
        wfP.P.phi = P_here.phi
        wfP.P.tref = P_here.tref
        wfP.P.psi = P_here.psi
        wfP.P.phiref = P_here.phiref


        wfP.P.fmin = opts.fmin
        wfP.P.deltaF = df
        wfP.P.deltaT = 1./opts.srate
        wfP.P.detector = ifo
        wfP.P.approx = lalsim.GetApproximantFromString(opts.approx)  # override the XML. Can screw you up (ref spins)

        if opts.verbose:
            wfP.P.print_params()

        hF = wfP.non_herm_hoff()
        return hF
    
    return_hF1 = get_hF1_NR


group2 = opts.group2
if group2 in nrwf.internal_ParametersAreExpressions.keys():
    if not nrwf.internal_ParametersAreExpressions[group2]:
        param2 = opts.param2
    else:
        param2 = eval(str(opts.param2))
    # Check for SXS 
    if group2 == "Sequence-SXS-All":
        nrwf.internal_FilenamesForParameters[group2][param2] =nrwf.internal_FilenamesForParameters[group2][param2].replace("Lev5", "Lev" +str(opts.use_spec_lev))
        print(" OVERRIDE OF SXS LEVEL : ", nrwf.internal_FilenamesForParameters[group2][param2])
    if opts.verbose:
        print("Importing ", group, param , " and ", group2, param2)
    wfP2 = nrwf.WaveformModeCatalog(opts.group2, param2, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, lmax=lmax,align_at_peak_l2_m2_emission=True,perturbative_extraction=opts.use_perturbative_extraction,use_provided_strain=opts.use_provided_strain2,reference_phase_at_peak=True,quiet=True)
    wfP2.P.fmin  = opts.fmin

    print(" WARNING: NR orientation parameters differ, so the waveform CANNOT be identical and will generally differ in at least orbital phase ")
    def get_hF2_NR(indx,ifo):
        P_here = P2_list[indx % nlines2]

        wfP2.P.radec=True
        wfP2.P.m1 = P_here.m1
        wfP2.P.m2 = P_here.m2

        wfP2.P.dist = P_here.dist
        wfP2.P.incl = P_here.incl
        wfP2.P.theta = P_here.theta
        wfP2.P.phi = P_here.phi
        wfP2.P.tref = P_here.tref
        wfP2.P.psi = P_here.psi
        wfP2.P.phiref = P_here.phiref


        wfP2.P.fmin = opts.fmin
        wfP2.P.deltaF = df
        wfP2.P.deltaT = 1./opts.srate
        wfP2.P.detector = ifo
        wfP2.P.approx = lalsim.GetApproximantFromString(opts.approx2)  # override the XML. Can screw you up (ref spins)
        if opts.verbose:
            wfP2.P.print_params()

        hF = wfP2.non_herm_hoff()

        return hF

    
    return_hF2 = get_hF2_NR








###
### Loop and carry out overlap
###

n_evals = np.max([nlines1,nlines2])

dat_out =[]

for indx in np.arange(n_evals):
  if opts.verbose:
      print(indx)
  if True:
#  try:
    line = []
    print(P1_list[indx].extract_param('thetaJN'), P1_list[indx].phiref, P1_list[indx].extract_param('beta'), end=' ')
    for ifo in ifo_list: #,'V1']:
        IP = IP_list[ifo]
        hF1 = return_hF1(indx,ifo)
        nm1 = IP.norm(hF1)
        hF2 = return_hF2(indx,ifo)
        nm2 = IP.norm(hF2)
        val = np.abs(IP.ip(hF1,hF2,include_epoch_differences=True)/nm1/nm2)  # correct for timeshift issues, if any
        print(" Epoch test ", hF1.epoch, hF2.epoch)
        line.append(val)
#        print ifo, IP.ip(hF1,hF2)/nm1/nm2, IP.ip(hF1,hF2),nm1, nm2
        print(val, end=' ')
        if opts.save_plots:
            if opts.verbose:
                print(" --- Saving plot for ", ifo, " ----")
            label1 = opts.approx
            label2 = opts.approx2
            hT1 = lalsimutils.DataInverseFourier(hF1)
            hT2 = lalsimutils.DataInverseFourier(hF2)
            if not (opts.group is None):
                npts = hT1.data.length
                T_wave =npts*hT1.deltaT
                hT1 = lalsimutils.DataRollTime(hT1,-0.5*T_wave/2)
                if opts.verbose:
                    print(" Epoch1 ", float(hT1.epoch))
                label1 = group+":"+param
            if not (opts.group2 is None):
                if opts.verbose:
                    print(" ---> Rolling to fix FT centering <-- ")
                npts = hT2.data.length
                T_wave =npts*hT2.deltaT
                hT2 = lalsimutils.DataRollTime(hT2,-0.5*T_wave/2)
                if opts.verbose:
                    print(" Epoch2 ", hT2.epoch)
                label2 = group2+":"+param2
            tvals = lalsimutils.evaluate_tvals(hT1) 
            tvals2 = lalsimutils.evaluate_tvals(hT2) 
#            tref = float(hT1.epoch)
            plt.plot(tvals -tref,np.real(hT1.data.data),'r',label=label1)
            plt.plot(tvals2 -tref,np.real(hT2.data.data),'g',label=label2)
            plt.legend()
            plt.savefig(opts.fname_output+"_fig_"+str(indx)+"_"+ifo+".png"); 
            
            # Estimate peak location, so plot can be focused on detail
            tmax_offset =(tvals- tref)[ np.argmax( np.abs(hT1.data.data))]  
                
            plt.xlim(tmax_offset-0.15, tmax_offset+0.05)  # NR scale, focus on merger
            plt.savefig(opts.fname_output+"_fig_"+str(indx)+"_"+ifo+"_detail.png"); 
            plt.clf()
    print()
    dat_out.append(line)

#  except:
  else:
      print(" Skipping ", indx)

np.savetxt(opts.fname_output, np.array(dat_out))
