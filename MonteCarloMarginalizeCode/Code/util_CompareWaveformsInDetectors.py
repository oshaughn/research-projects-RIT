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

import argparse
import numpy as np

import lalsimutils
import lalsimulation as lalsim
import lal
import sys

import NRWaveformCatalogManager as nrwf


parser = argparse.ArgumentParser()
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
parser.add_argument("--fmin",default=10,type=float)
parser.add_argument("--fmax",default=2000,type=float,help="Maximum frequency in Hz, used for PSD integral.")
parser.add_argument("--psd-file",default=None,help="PSD file (assumed for hanford)")
parser.add_argument("--psd",type=str,default="SimNoisePSDaLIGOZeroDetHighPower",help="psd name (attribute in lalsimulation).  SimNoisePSDiLIGOSRD, lalsim.SimNoisePSDaLIGOZeroDetHighPower, lalsimutils.Wrapper_AdvLIGOPsd, .SimNoisePSDiLIGOSRD... ")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")


opts = parser.parse_args()


###
### Spacing settings, for overlaps
###


lmax = opts.lmax
T_window = 64.
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

fNyq = opts.srate/2.
IP = lalsimutils.ComplexIP(fNyq=fNyq,deltaF=df,analyticPSD_Q=analyticPSD_Q,psd=psd,fMax=opts.fmax)



###
### Load injection XML
###

P1_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj)
nlines1  = len(P1_list)

P2_list = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj2)
nlines2  = len(P2_list)

###
### Define functions to generate waveforms
###

def get_hF1(indx,ifo):
    P = P1_list[indx % nlines1].manual_copy() # looping
    P.fmin = opts.fmin
    P.radec =True
    P.tref = P1_list[indx%nlines1].tref # copy, this is an allocated object
    P.deltaF = df
#    P.fmin  = opts.fmin
    P.deltaT = 1./opts.srate
    P.detector = ifo
    P.approx = lalsim.GetApproximantFromString(opts.approx)  # override the XML. Can screw you up (ref spins)
#    P.print_params()
    hF = lalsimutils.non_herm_hoff(P)
    return hF

def get_hF2(indx,ifo):
    P = P2_list[indx % nlines2].manual_copy() # looping
    P.fmin = opts.fmin
    P.radec =True
    P.tref = P1_list[indx%nlines1].tref # copy, this is an allocated object
    P.deltaF = df
#    P.fmin  = opts.fmin
    P.deltaT = 1./opts.srate
    P.detector = ifo
    P.approx = lalsim.GetApproximantFromString(opts.approx2)  # override the XML. Can screw you up
#    P.print_params()
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
                print " OVERRIDE OF SXS LEVEL : ", nrwf.internal_FilenamesForParameters[group][param]
    wfP = nrwf.WaveformModeCatalog(opts.group, param, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, lmax=lmax,align_at_peak_l2_m2_emission=True,perturbative_extraction=opts.use_perturbative_extraction,perturbative_extraction_full=opts.use_perturbative_extraction_full,use_provided_strain=opts.use_provided_strain,reference_phase_at_peak=True)
    wfP.P.fmin  = opts.fmin

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

#        wfP.P.print_params()

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
        print " OVERRIDE OF SXS LEVEL : ", nrwf.internal_FilenamesForParameters[group2][param2]
    if opts.verbose:
        print "Importing ", group, param , " and ", group2, param2
    wfP2 = nrwf.WaveformModeCatalog(opts.group2, param2, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, lmax=lmax,align_at_peak_l2_m2_emission=True,perturbative_extraction=opts.use_perturbative_extraction,use_provided_strain=opts.use_provided_strain2,reference_phase_at_peak=True)
    wfP2.P.fmin  = opts.fmin

    def get_hF2_NR(indx,ifo):
        P_here = P2_list[indx % nlines1]

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
#        wfP2.P.print_params()

        hF = wfP2.non_herm_hoff()

        return hF

    
    return_hF2 = get_hF2_NR








###
### Loop and carry out overlap
###

n_evals = np.max([nlines1,nlines2])

for indx in np.arange(n_evals):
    for ifo in ['H1','L1','V1']:
        hF1 = return_hF1(indx,ifo)
        nm1 = IP.norm(hF1)
        hF2 = return_hF2(indx,ifo)
        nm2 = IP.norm(hF2)
#        print ifo, IP.ip(hF1,hF2)/nm1/nm2, IP.ip(hF1,hF2),nm1, nm2
        print  np.abs(IP.ip(hF1,hF2)/nm1/nm2),  
    print
