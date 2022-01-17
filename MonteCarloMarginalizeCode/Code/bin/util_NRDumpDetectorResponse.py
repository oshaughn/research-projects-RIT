#! /usr/bin/env python
#
# util_NRDumpDetectorResponse.py
#
# EXAMPLE
#    util_NRDumpDetectorResponse.py --inj inj.xml.gz --event 0 --group GROUP --param PARAM --ifo H
#
# Intended to reproduce events found by ILE.  Will use real sky locations, total masses, etc


import argparse
import numpy as np

import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import sys

import NRWaveformCatalogManager3 as nrwf

parser = argparse.ArgumentParser()
parser.add_argument("--inj",default=None)
parser.add_argument("--event",default=None,type=int)
parser.add_argument("--group", default="Sequence-GT-Aligned-UnequalMass",help="inspiral XML file containing injection information.")
parser.add_argument("--param", default=(0.0, 2.),help="Parameter value")
parser.add_argument("--hybrid-use",action='store_true',help="Enable use of NR (or ROM!) hybrid, using --approx as the default approximant and with a frequency fmin")
parser.add_argument("--hybrid-method",default="taper_add",help="Hybridization method for NR (or ROM!).  Passed through to LALHybrid. pseudo_aligned_from22 will provide ad-hoc higher modes, if the early-time hybridization model only includes the 22 mode")
parser.add_argument("--no-memory",action='store_true')
parser.add_argument("--seglen",default=8,type=int)
parser.add_argument("--t-ref",default=None,type=float,help="Zero of time to use on exported data files. (default is to use the time in the XML")
parser.add_argument("--use-perturbative-extraction",default=False,action='store_true')  # this should be the default with ILE operation
parser.add_argument("--use-provided-strain",default=False,action='store_true')  # this should NEVER be used, provided for testing only
parser.add_argument("--l", default=2, type=int)
parser.add_argument("--rextr", default=None,type=int)
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--approx",default='SEOBNRv2',help="Approximant to use for LAL comparisons. Can be EOBNRv2HM or SEOBNRv2")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
opts = parser.parse_args()

print(" WARNING: Use this package only when you record TIME INFORMATION (i.e., --maximization-only). ")

T_window = opts.seglen
deltaT = 1./4096

bNoInteractivePlots=True # default
fig_extension = '.png'
try:
    import matplotlib
    print(" Matplotlib backend ", matplotlib.get_backend())
    if matplotlib.get_backend() is 'MacOSX':
        if opts.save_plots:
            print("  OSX without interactive plots")
            bNoInteractivePlots=True
            fig_extension='.png'
        else:  #  Interactive plots
            print("  OSX with interactive plots")
            bNoInteractivePlots=False
    elif matplotlib.get_backend() is 'agg':
        fig_extension = '.png'
        bNoInteractivePlots=True
        print(" No OSX; no interactive plots ")
    else:
        print(" Unknown configuration ")
        fig_extension = '.png'
        bNoInteractivePlots =True
    from matplotlib import pyplot as plt
    bNoPlots=False
except:
    from matplotlib import pyplot as plt
    fig_extension = '.png'
    print(" - no matplotlib - ")
    bNoInteractivePlots = False
    bNoPlots = True

group = opts.group
if not nrwf.internal_ParametersAreExpressions[group]:
    param = opts.param
else:
    param = eval(str(opts.param))
if opts.verbose:
    print("Importing ", group, param)


## LOAD INJECTION FILE
#   - complete garbage as waveform entry, to make sure the code to load waveforms operates PERIOD
print(" Loading injection file ... ")
# force_waveform='TaylorT4threePointFivePN',,force_taper='TAPER_NONE'
P = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))[opts.event]  # Load in the physical parameters of the inj

## LOAD WEIGHT REPORT (just in case)
# from ligo.lw import lsctables, utils, table
# samples = table.get_table(utils.load_filename(opts.inj), lsctables.SimInspiralTable.tableName)
# lnLmax = np.max([s.alpha1 for s in samples])
# def sorted_indexes(seq):
#     return [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]
# samples_indx_sorted =sorted_indexes( [np.exp(s.alpha-lnLmax)*s.alpha2/s.alpha3 for s in samples])
# print samples[opts.event]
# the_sample = samples[opts.event]
# print " Weight for this sample ", the_sample.alpha  + np.log(the_sample.alpha2/the+sample.alpha3)

## LOAD NR SIMULATION
print(" Loading NR simulation ... ")
no_memory = False
wfP = nrwf.WaveformModeCatalog(opts.group, param, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, extraction_radius=opts.rextr,lmax=opts.l,align_at_peak_l2_m2_emission=True,perturbative_extraction=opts.use_perturbative_extraction,use_provided_strain=opts.use_provided_strain)
if opts.no_memory and wfP.P.SoftAlignedQ():
    print(" STRIPPING MEMORY")
    no_memory = True


## COPY INJECTION FILE CORE SETTINGS TO wfP.P: only mass and 7 extrinsic parameters matter for NR
#   - could use P.manual_copy, but want to be CAREFUL.
# total mass
wfP.P.m1 = P.m1
wfP.P.m2 =P.m2
# seven intrinsic parameters
wfP.P.dist = P.dist
wfP.P.incl = P.incl
wfP.P.phiref=P.phiref
wfP.P.theta = P.theta
wfP.P.phi = P.phi
wfP.P.psi = P.psi
wfP.P.tref = P.tref  # this is a GEOCENTRIC TIME for now
wfP.P.detector = 'H1'  # for now
wfP.P.radec =True
wfP.P.deltaF = 1./T_window  # 1
wfP.P.deltaT = deltaT

wfP.P.print_params()

data_dict_T ={}
fig_list = {}
indx = 0
t_ref = wfP.P.tref   # This SHOULD NOT BE USED, if you want to time-align your signal
if opts.t_ref:
    t_ref = float(opts.t_ref)
else:
    print(" Warning: no reference time, it will be very difficult to time-align your signals ")
for ifo in ['H1', 'L1', 'V1']:
    indx += 1
    fig_list[ifo] = indx
    plt.figure(indx)
    wfP.P.detector = ifo
    data_dict_T[ifo] = wfP.real_hoft(no_memory=opts.no_memory,hybrid_use=opts.hybrid_use,hybrid_method=opts.hybrid_method)
    tvals = data_dict_T[ifo].deltaT*np.arange(data_dict_T[ifo].data.length) + float(data_dict_T[ifo].epoch)
    det = lalsim.DetectorPrefixToLALDetector(ifo)
    print(ifo, " T_peak =", P.tref + lal.TimeDelayFromEarthCenter(det.location, P.phi, P.theta, P.tref) - t_ref)
    if opts.verbose:
        plt.plot(tvals - wfP.P.tref, data_dict_T[ifo].data.data)

    np.savetxt(ifo+"_nr_"+group+"_"+str(param)+"_event_"+str(opts.event)+".dat", np.array([tvals -t_ref, data_dict_T[ifo].data.data]).T)

if opts.verbose and not opts.save_plots:
    plt.show()
if opts.save_plots:
    for ifo in  fig_list:
        plt.figure(fig_list[ifo])
        plt.savefig("response-"+str(ifo)+group+"_"+str(param)+fig_extension)
        



