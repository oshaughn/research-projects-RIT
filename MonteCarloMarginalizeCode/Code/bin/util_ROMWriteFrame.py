#! /usr/bin/env python
#  
#
# Tool: dumps a candidate waveform to a frame file.
# Default GPS time used is boring
# Tries to be as close to C as possible -- no interface via pylal/glue
#
# EXAMPLE
#    python util_LALWriteFrame.py; FrDump -i dummy.gwf
#
#  WARNING: My version does NOT interpolate the signal to a synchronized set of sample times.
#                       This may cause problems for some applications, particularly at low sample rates.


import argparse
import numpy as np
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal

import RIFT.physics.ROMWaveformManager as romwf

parser = argparse.ArgumentParser()
parser.add_argument("--group", default="my_surrogates/nr_surrogates/",help="Surrogate example tutorial/TutorialSurrogate")
parser.add_argument("--param", default='/SpEC_q1_9_NoSpin_SingleModes_REF.h5',help="Parameter value:  /SpEC_q1_10_NoSpin_nu5thDegPoly_exclude_2_0_FastSplined.h5, ...")
parser.add_argument("--lmax", type=int, default=2,help="Set max modes")
parser.add_argument("--fname", default=None, help = "Base name for output frame file. Otherwise auto-generated ")
parser.add_argument("--instrument", default="H1",help="Use H1, L1,V1")
parser.add_argument("--inj", dest='inj', default=None,help="inspiral XML file containing injection information.")
parser.add_argument("--event",type=int, dest="event_id", default=None,help="event ID of injection XML to use.")
parser.add_argument("--single-ifo",default=False,action='store_true')
parser.add_argument("--approx",type=str,default=None)
parser.add_argument("--srate",type=int,default=16384,help="Sampling rate")
parser.add_argument("--seglen", type=float,default=16., help="Default window size for processing.")
parser.add_argument("--start", type=int,default=None)
parser.add_argument("--stop", type=int,default=None)
parser.add_argument("--fref", dest='fref', type=float, default=0.0, help="Waveform reference frequency [template]. Required, default is 0 (coalescence).")
parser.add_argument("--incl",default=None,help="Set the inclination of L (at fref). Particularly helpful for aligned spin tests")
parser.add_argument("--mass1",default=35,type=float,help='Mass 1 (solar masses)')
parser.add_argument("--mass2",default=34,type=float,help='Mass 2 (solar masses)')
parser.add_argument("--verbose", action="store_true",default=False)
opts=  parser.parse_args()


# Generate signal
P = lalsimutils.ChooseWaveformParams()
P.deltaT = 1./opts.srate
P.radec = True  # use a real source with a real instrument
if not opts.inj:
    P.randomize(aligned_spin_Q=True,default_inclination=opts.incl)
    P.m1 = opts.mass1*lalsimutils.lsu_MSUN
    P.m2 = opts.mass2*lalsimutils.lsu_MSUN
    P.taper = lalsimutils.lsu_TAPER_START
    P.tref =1000000000  # default
    if opts.approx:
        P.approx = lalsim.GetApproximantFromString(str(opts.approx))
    else:
        P.approx = lalsim.GetApproximantFromString("SpinTaylorT2")
else:
    from glue.ligolw import lsctables, table, utils # check all are needed

    filename = opts.inj
    event = opts.event_id
    xmldoc = utils.load_filename(filename, verbose = True, contenthandler =lalsimutils.cthdler)
    sim_inspiral_table = table.get_table(xmldoc, lsctables.SimInspiralTable.tableName)
    P.copy_sim_inspiral(sim_inspiral_table[int(event)])
    if opts.approx:
        P.approx=lalsimutils.StringToLALApproximant(opts.approx)
P.taper = lalsimutils.lsu_TAPER_START
P.detector = opts.instrument
P.print_params()


T_est = lalsimutils.estimateWaveformDuration(P)
T_est = P.deltaT*lalsimutils.nextPow2(T_est/P.deltaT)
T_est = np.max([4,T_est])
P.deltaF = 1./T_est
print(" Duration ", T_est)
if T_est < opts.seglen:
    print(" Buffer length too short, automating retuning forced ")

# Generate ROM
acatHere = romwf.WaveformModeCatalog(opts.group, opts.param, opts.lmax)

# Generate signal
hoft = acatHere.real_hoft(P)   # include translation of source, but NOT interpolation onto regular time grid
# zero pad to be opts.seglen long, if necessary
if opts.seglen/hoft.deltaT > hoft.data.length:
    TDlenGoal = int(opts.seglen/hoft.deltaT)
    hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlenGoal)

# zero pad some more on either side, to make sure the segment covers start to stop
if opts.start and hoft.epoch > opts.start:
    nToAddBefore = int((hoft.epoch-opts.start)/hoft.deltaT)
    print(nToAddBefore, hoft.data.length)
    ht = lal.CreateREAL8TimeSeries("Template h(t)", 
            hoft.epoch - nToAddBefore*hoft.deltaT, 0, hoft.deltaT, lalsimutils.lsu_DimensionlessUnit, 
            hoft.data.length+nToAddBefore)
    ht.data.data = np.zeros(ht.data.length)  # clear
    ht.data.data[nToAddBefore:nToAddBefore+hoft.data.length] = hoft.data.data
    hoft = ht

if opts.stop and hoft.epoch+hoft.data.length*hoft.deltaT < opts.stop:
    nToAddAtEnd = int( (-(hoft.epoch+hoft.data.length*hoft.deltaT)+opts.stop)/hoft.deltaT)
    print("Padding end ", nToAddAtEnd, hoft.data.length)
    hoft = lal.ResizeREAL8TimeSeries(hoft,0, int(hoft.data.length+nToAddAtEnd))

channel = opts.instrument+":FAKE-STRAIN"

tstart = int(hoft.epoch)
duration = int(hoft.data.length*hoft.deltaT)
if not opts.fname:
    fname = opts.instrument.replace("1","")+"-fake_strain-"+str(tstart)+"-"+str(duration)+".gwf"

print("Writing signal with ", hoft.data.length*hoft.deltaT, " to file ", fname)
lalsimutils.hoft_to_frame_data(fname,channel,hoft)

# TEST: Confirm it works by reading the frame
if opts.verbose:
    import os
    from matplotlib import pyplot as plt
    plt.switch_backend('agg')  # fix backend
    # First must create corresponding cache file
    os.system("echo "+ fname+ " | lalapps_path2cache   > test.cache")
    # Now I can read it
    # Beware that the results are OFFSET FROM ONE ANOTHER due to PADDING,
    #    but that the time associations are correct
    hoft2 = lalsimutils.frame_data_to_hoft("test.cache", channel)
    tvals2 = (float(hoft2.epoch) - float(P.tref)) +  np.arange(hoft2.data.length)*hoft2.deltaT
    tvals = (float(hoft.epoch) - float(P.tref)) +  np.arange(hoft.data.length)*hoft.deltaT
    plt.plot(tvals2,hoft2.data.data,label='Fr')
    plt.plot(tvals,hoft.data.data,label='orig')
    plt.xlim(-1,0.1)  # should scale with mass
    plt.legend(); 
    plt.savefig("frdump_rom_"+opts.instrument+".png")
