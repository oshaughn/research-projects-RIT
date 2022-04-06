#! /usr/bin/env python
#  
#
# Tool: dumps a candidate waveform to a frame file.
# Default GPS time used is boring
# Tries to be as close to C as possible -- no interface via pylal/glue
#
# EXAMPLE
#   python util_NRWriteFrame.py --group 'Sequence-SXS-All' --param 1 --verbose
#   python util_NRWriteFrame.py --incl 1.5 --verbose     # edge on
#
#  WARNING: My version does NOT interpolate the signal to a synchronized set of sample times.
#                       This may cause problems for some applications, particularly at low sample rates.


import argparse
import sys
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal

import NRWaveformCatalogManager3 as nrwf


parser = argparse.ArgumentParser()
parser.add_argument("--group", default="Sequence-GT-Aligned-UnequalMass",help="inspiral XML file containing injection information.")
parser.add_argument("--param", default=(0.0, 2.),help="Parameter value")
parser.add_argument("--rextr", default=None,type=int)
parser.add_argument("--fname", default=None, help = "Base name for output frame file. Otherwise auto-generated ")
parser.add_argument("--instrument", default="H1",help="Use H1, L1,V1")
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing injection information. Used for extrinsic information only")
parser.add_argument("--nr-perturbative-extraction",default=False,action='store_true')
parser.add_argument("--nr-use-hybrid",action='store_true')
parser.add_argument("--mass", default=150.0,type=float,help="Total mass in solar masses")  # 150 turns out to be ok for Healy et al sims
parser.add_argument("--lmax", default=2, type=int)
parser.add_argument("--event",type=int, dest="event_id", default=None,help="event ID of injection XML to use.")
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=16., help="Default window size for processing.")
parser.add_argument("--incl",default=0,type=float,help="Set the inclination of the simuation. Helpful for aligned spin tests")
parser.add_argument("--start", type=int,default=None)
parser.add_argument("--stop", type=int,default=None)
parser.add_argument("--verbose", action="store_true",default=False)
parser.add_argument("--print-group-list",default=False,action='store_true')
parser.add_argument("--print-param-list",default=False,action='store_true')
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
opts=  parser.parse_args()


group = opts.group
if nrwf.internal_ParametersAreExpressions[group]:
    param = eval(str(opts.param))
else:
    param = opts.param
if opts.verbose:
    print("Importing ", group, param) 


if opts.print_group_list:
    print("Simulations available")
    for key in  list(nrwf.internal_ParametersAvailable.keys()):
        print("  ", key)
    sys.exit(0)

if opts.print_param_list:
    print("Parameters available for ", group)
    for key in  nrwf.internal_ParametersAvailable[group]:
        print("  ", key)
    sys.exit(0)

# Window size : 8 s is usually more than enough, though we will fill a 16s buffer to be sure.
T_window = 16. # default 

# Load catalog
wfP = nrwf.WaveformModeCatalog(opts.group, param, clean_initial_transient=True,clean_final_decay=True, shift_by_extraction_radius=True, extraction_radius=opts.rextr,lmax=opts.lmax,align_at_peak_l2_m2_emission=True,build_strain_and_conserve_memory=True,perturbative_extraction=opts.nr_perturbative_extraction)


# Generate signal
wfP.P.deltaT = 1./opts.srate
wfP.P.deltaF = 1./T_window
wfP.P.radec = True  # use a real source with a real instrument
wfP.P.fmin = 10
mtotOrig = (wfP.P.m1+wfP.P.m2)/lal.MSUN_SI
mtotParam = opts.mass
wfP.P.m1 *= mtotParam/mtotOrig
wfP.P.m2 *= mtotParam/mtotOrig
wfP.P.incl = opts.incl

if not opts.inj:
    wfP.P.taper = lalsimutils.lsu_TAPER_START
    wfP.P.tref = 1000000000 #1000000000            # default
    wfP.P.dist = 100*1e6*lal.PC_SI # default
else:
    from ligo.lw import lsctables, table, utils # check all are needed

    filename = opts.inj
    event = opts.event_id
    xmldoc = utils.load_filename(filename, verbose = True,contenthandler=lalsimutils.cthdler)
    sim_inspiral_table = lsctables.SimInspiralTable.get_table(xmldoc)
    wfP.P.copy_sim_inspiral(sim_inspiral_table[int(event)])
wfP.P.detector = opts.instrument
wfP.P.print_params()
mtotMsun = (wfP.P.m1+wfP.P.m2)/lal.MSUN_SI


# Rescale window if needed. Higher masses need longer, to get right start frequency
print(" NR duration (in s) of simulation at this mass = ", wfP.estimateDurationSec())
print(" NR starting 22 mode frequency at this mass = ", wfP.estimateFminHz())
#T_window = max([16., 2**int(2+np.log2(np.power(mtotMsun/150, 1+3./8.)))])
T_window = max([16, 2**int(np.log(wfP.estimateDurationSec())/np.log(2)+1)])
wfP.P.deltaF = 1./T_window
print(" Final T_window ", T_window)


# Generate signal
hoft = wfP.real_hoft(hybrid_use=opts.nr_use_hybrid)   # include translation of source, but NOT interpolation onto regular time grid
print(" Original signal (min, max) ", np.min(hoft.data.data) ,np.max(hoft.data.data))
# zero pad to be opts.seglen long
TDlenGoal = int(opts.seglen/hoft.deltaT)
nptsOrig = hoft.data.length
hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlenGoal)
hoft.data.data[nptsOrig:TDlenGoal] = 0 # np.zeros(TDlenGoal-nptsOrig) # zero out the tail
print(" Resized signal (min, max) ", np.min(hoft.data.data) ,np.max(hoft.data.data))
# zero pad some more on either side, to make sure the segment covers start to stop
if opts.start and hoft.epoch > opts.start:
    nToAddBefore = int((hoft.epoch-opts.start)/hoft.deltaT)
    print("Padding start ", nToAddBefore, hoft.data.length)
    ht = lal.CreateREAL8TimeSeries("Template h(t)", 
            hoft.epoch - nToAddBefore*hoft.deltaT, 0, hoft.deltaT, lalsimutils.lsu_DimensionlessUnit, 
            hoft.data.length+nToAddBefore)
    ht.data.data[0:ht.data.length] = np.zeros(ht.data.length)  # initialize to zero for safety
    ht.data.data[nToAddBefore:nToAddBefore+hoft.data.length] = hoft.data.data
    hoft = ht

if opts.stop and hoft.epoch+hoft.data.length*hoft.deltaT < opts.stop:
    nToAddAtEnd = int( (-(hoft.epoch+hoft.data.length*hoft.deltaT)+opts.stop)/hoft.deltaT)
else:
    nToAddAtEnd=0
if nToAddAtEnd <=0:
    nToAddAtEnd = int(1/hoft.deltaT)  # always at at least 1s of padding at end
print("Padding end ", nToAddAtEnd, hoft.data.length)
nptsNow = hoft.data.length
hoft = lal.ResizeREAL8TimeSeries(hoft,0, int(hoft.data.length+nToAddAtEnd))
hoft.data.data[nptsNow:hoft.data.length] = 0
print(" Padded signal (min, max) ", np.min(hoft.data.data) ,np.max(hoft.data.data))


channel = opts.instrument+":FAKE-STRAIN"

tstart = int(hoft.epoch)
duration = int(hoft.data.length*hoft.deltaT)
if not opts.fname:
    fname = opts.instrument.replace("1","")+"-fake_strain-"+str(tstart)+"-"+str(duration)+".gwf"

print("Writing signal with ", hoft.data.length*hoft.deltaT, " to file ", fname)
print(" Maximum original ", np.max(hoft.data.data), " corresponding to time ", np.argmax(hoft.data.data)*hoft.deltaT+hoft.epoch)
lalsimutils.hoft_to_frame_data(fname,channel,hoft)

bNoInteractivePlots=True # default
fig_extension = '.jpg'
try:
    import matplotlib
    print(" Matplotlib backend ", matplotlib.get_backend())
    if matplotlib.get_backend() is 'MacOSX':
        if opts.save_plots:
            print("  OSX without interactive plots")
            bNoInteractivePlots=True
            fig_extension='.jpg'
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
    fig_extension = '.jpeg'
    print(" - no matplotlib - ")
    bNoInteractivePlots = True
    bNoPlots = False


# TEST: Confirm it works by reading the frame
if opts.verbose and not bNoPlots:
    import os
#    from matplotlib import pyplot as plt
    # First must create corresponding cache file
    os.system("echo "+ fname+ " | lalapps_path2cache   > test.cache")
    # Now I can read it
    # Beware that the results are OFFSET FROM ONE ANOTHER due to PADDING,
    #    but that the time associations are correct
    hoft2 = lalsimutils.frame_data_to_hoft("test.cache", channel)
    tvals2 = (float(hoft2.epoch) - float(wfP.P.tref)) +  np.arange(hoft2.data.length)*hoft2.deltaT
    tvals = (float(hoft.epoch) - float(wfP.P.tref)) +  np.arange(hoft.data.length)*hoft.deltaT

    print(" Maximum original ", np.max(hoft.data.data), " size ", len(tvals), len(hoft.data.data), " corresponding to time ", np.argmax(hoft.data.data)*hoft.deltaT+hoft.epoch)
    print(" Maximum frames ", np.max(hoft2.data.data), " size ", len(tvals2), len(hoft2.data.data), " corresponding to time ", np.argmax(hoft2.data.data)*hoft2.deltaT+hoft2.epoch)

    plt.plot(tvals2,hoft2.data.data,label='Fr')
    plt.xlim(-0.75* (mtotMsun/150),0.25*mtotMsun/150)
    plt.plot(tvals,hoft.data.data,label='orig')
    plt.legend(); 

    if not bNoInteractivePlots:
        plt.show()
    else:
        for indx in [1]:
            print("Writing figure ", indx)
            plt.xlim(-0.75* (mtotMsun/150),0.25*mtotMsun/150)
            plt.figure(indx); plt.savefig("nr-framedump-" +str(indx)+fig_extension)
            plt.xlim(min(tvals2),max(tvals2)) # full range with pad
            plt.figure(indx); plt.savefig("nr-framedump-full-" +str(indx)+fig_extension)


