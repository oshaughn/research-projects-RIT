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

from __future__ import print_function

import argparse
import sys
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal

import RIFT.physics.EOBTidalExternalC as eobwf


parser = argparse.ArgumentParser()
parser.add_argument("--fname", default=None, help = "Base name for output frame file. Otherwise auto-generated ")
parser.add_argument("--instrument", default="H1",help="Use H1, L1,V1")
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing injection information. Used for extrinsic information only")
parser.add_argument("--mass1", default=1.50,type=float,help="Mass in solar masses")  # 150 turns out to be ok for Healy et al sims
parser.add_argument("--mass2", default=1.35,type=float,help="Mass in solar masses")
parser.add_argument("--lambda1",default=590,type=float)
parser.add_argument("--lambda2", default=590,type=float)
parser.add_argument("--fmin", default=35,type=float,help="Mininmum frequency in Hz, default is 40Hz to make short enough waveforms. Focus will be iLIGO to keep comutations short")
parser.add_argument("--lmax", default=2, type=int)
parser.add_argument("--event",type=int, dest="event_id", default=None,help="event ID of injection XML to use.")
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=256*2., help="Default window size for processing.")
parser.add_argument("--incl",default=0,type=float,help="Set the inclination of the simuation. Helpful for aligned spin tests")
parser.add_argument("--start", type=int,default=None)
parser.add_argument("--stop", type=int,default=None)
parser.add_argument("--approx",type=str,default=None,help="Unused")
parser.add_argument("--single-ifo",action='store_true',default=None,help="Unused")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
opts=  parser.parse_args()



# Window size : 8 s is usually more than enough, though we will fill a 16s buffer to be sure.
T_window = 128. # default.  Note in practice most NS-NS waveforms will need to be many tens of minutes long 



# Generate signal
P=lalsimutils.ChooseWaveformParams()
P.m1 = opts.mass1 *lal.MSUN_SI
P.m2 = opts.mass2 *lal.MSUN_SI
P.dist = 150*1e6*lal.PC_SI
P.lambda1  = opts.lambda1
P.lambda2  = opts.lambda2
P.fmin=opts.fmin   # Just for comparison!  Obviously only good for iLIGO
P.ampO=-1  # include 'full physics'
P.deltaT=1./16384
P.taper = lalsim.SIM_INSPIRAL_TAPER_START
# This must be done BEFORE changing the duration
P.scale_to_snr(20,lalsim.SimNoisePSDaLIGOZeroDetHighPower,['H1', 'L1'])
if opts.start and opts.stop:
    opts.seglen = opts.stop-opts.start # override
P.deltaF = 1./opts.seglen #lalsimutils.findDeltaF(P)
if P.deltaF > 1./T_window:
    print(" time too short ")


if not opts.inj:
    P.taper = lalsimutils.lsu_TAPER_START
    P.tref = 1000000000 #1000000000            # default
    P.dist = 150*1e6*lal.PC_SI # default
else:
    from glue.ligolw import lsctables, table, utils # check all are needed

    filename = opts.inj
    event = opts.event_id
    xmldoc = utils.load_filename(filename, verbose = True,contenthandler =lalsimutils.cthdler)
    sim_inspiral_table = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
    P.copy_sim_inspiral(sim_inspiral_table[int(event)])
P.detector = opts.instrument

P.print_params()

# FAIL if masses are not viable
if P.m1/lal.MSUN_SI > 3 or P.m2/lal.MSUN_SI > 3:
    print(" Invalid NS mass ")
    sys.exit(0)

wfP = eobwf.WaveformModeCatalog(P,lmax=opts.lmax)
print(" Loaded modes ", wfP.waveform_modes_complex.keys())
print(" Duration of stored signal ", wfP.estimateDurationSec())
mtotMsun = (wfP.P.m1+wfP.P.m2)/lal.MSUN_SI


# Generate signal
hoft = wfP.real_hoft()   # include translation of source, but NOT interpolation onto regular time grid
print(" Original signal (min, max) ", np.min(hoft.data.data) ,np.max(hoft.data.data))
print(" Original signal duration ", hoft.deltaT*hoft.data.length)
# zero pad to be opts.seglen long
TDlenGoal = int(opts.seglen/hoft.deltaT)
if TDlenGoal < hoft.data.length:
    print(" seglen too short -- signal truncation would be required")
    sys.exit(0)
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
print("Maximum original ", np.max(hoft.data.data))
print("Start time", hoft.epoch)
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
    fig_extension = '.png'
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

    ncrit = np.argmax(hoft2.data.data)
    tcrit = float(hoft2.epoch) - float(wfP.P.tref) + ncrit*hoft2.deltaT    # zero time

    print(" Maximum original ", np.max(hoft.data.data), " size ", len(tvals), len(hoft.data.data))
    print(" Maximum frames ", np.max(hoft2.data.data), " size ", len(tvals2), len(hoft2.data.data))
    print(" Location of maximum in samples. relative time ", ncrit, tcrit)
    print(" Location of maximum in samples, compared to tref", tcrit+P.tref, end=' ')
    print(" Location of maximum as GPS time ", ncrit*hoft2.deltaT+ float(hof2.epoch))

    plt.plot(tvals2,hoft2.data.data,label='Fr')
    plt.xlim(tcrit-1,tcrit+1)
    plt.plot(tvals,hoft.data.data,label='orig')
    plt.legend(); 

    if not bNoInteractivePlots:
        plt.show()
    else:
        for indx in [1]:
            print("Writing figure ", indx)
            plt.xlim(tcrit-0.1,tcrit+0.01)
            plt.figure(indx); plt.savefig("eob-framedump-" +str(indx)+fig_extension)
#            plt.xlim(min(tvals2),max(tvals2)) # full range with pad
#            plt.figure(indx); plt.savefig("eob-framedump-full-" +str(indx)+fig_extension)


