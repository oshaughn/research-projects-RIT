#!/usr/bin/env python
#
# GOAL
#    - plot strain h_{lm}(f) per mode
#    - scale to specific masses (or use dimensionless if M=1).
#    - show SRD
#
# BONUS USES
#    - validate tapering (as needed)
#
# USAGE
#    util_TidalPlotSimulationStrain --simulate-network  --seglen 64 # only plots individual detectors
#    util_TidalPlotSimulationStrain --simulate-network  --seglen 256 # only plots individual detectors

import argparse
import numpy as np

import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import sys

import RIFT.physics.EOBTidalExternalC as eobT

parser = argparse.ArgumentParser()
parser.add_argument("--group", default="Sequence-GT-Aligned-UnequalMass",help="inspiral XML file containing injection information.")
parser.add_argument("--param", default=(0.0, 2.),help="Parameter value")
parser.add_argument("--mass1", default=1.5,help="Total mass in solar masses")  # 150 turns out to be ok for Healy et al sims
parser.add_argument("--mass2", default=1.4,help="Total mass in solar masses")  # 150 turns out to be ok for Healy et al sims
parser.add_argument("--l", default=3, type=int)
parser.add_argument("--fmin",default=30, type=float)
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--incl", type=float,default=np.pi/4)
parser.add_argument("--print-group-list",default=False,action='store_true')
parser.add_argument("--print-param-list",default=False,action='store_true')
parser.add_argument("--simulate-network",default=False, action='store_true')
parser.add_argument("--seglen",default=64,type=int)
parser.add_argument("--show-plots",default=False,action='store_true')
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
opts = parser.parse_args()



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
        fig_extension = '.jpg'
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
param = eval(str(opts.param))
if opts.verbose:
    print("Importing ", group, param)
    

if opts.print_group_list:
    print("Simulations available")
    for key in  nrwf.internal_ParametersAvailable.keys():
        print("  ", key)
    sys.exit(0)

if opts.print_param_list:
    print("Parameters available for ", group)
    for key in  nrwf.internal_ParametersAvailable[group]:
        print("  ", key)
    sys.exit(0)


l = opts.l
T_window = opts.seglen
fmin =10

P=lalsimutils.ChooseWaveformParams()
P.m1 = opts.mass1 *lal.MSUN_SI
P.m2 = opts.mass2 *lal.MSUN_SI
P.dist = 150*1e6*lal.PC_SI
P.lambda1  = 500
P.lambda2  = 500
P.fmin=opts.fmin   # Just for comparison!  Obviously only good for iLIGO
P.ampO=-1  # include 'full physics'
P.deltaT=1./16384
P.taper = lalsim.SIM_INSPIRAL_TAPER_START
P.deltaF = lalsimutils.findDeltaF(P)
P.scale_to_snr(20,lalsim.SimNoisePSDaLIGOZeroDetHighPower,['H1', 'L1'])
if P.deltaF > 1./T_window:
    print(" time too short ")
P.deltaF = 1./opts.seglen
P.print_params()

wfP = eobT.WaveformModeCatalog(P,lmax=l,align_at_peak_l2_m2_emission=True)
print(" Loaded modes ", wfP.waveform_modes_complex.keys())
wfP.P.incl = opts.incl

###
### Rest of code: Plot modes. Use PRECISELY the same code path as ILE
###
data_dict={}
if opts.simulate_network:
    detectors = ['H1', 'L1','V1']

    # Convert to time domain
    for det in detectors:
        # Generate data
        wfP.P.detector=det
        data_dict[det] = wfP.non_herm_hoff()

        hT = lalsimutils.DataInverseFourier(data_dict[det])  # complex inverse fft, for 2-sided data
        # Roll so we are centered
        ncrit  = np.argmax(np.abs(hT.data.data))
        print("  : Maximum at (discrete) time: ",det, ncrit*hT.deltaT + hT.epoch, " relative to tref ")
        tvals = lalsimutils.evaluate_tvals(hT)  - P.tref
        plt.plot(tvals, hT.data.data,label=det)
    plt.title(" data_dict h(t)")
    plt.legend()
    plt.savefig("tidal_plot_strain-hoft"+fig_extension)
    plt.xlim(-0.5,0.5)  # usually centered around t=0
    plt.savefig("tidal_plot_strain-hoft-zoomed"+fig_extension)
    
    plt.show()
    sys.exit(0)


###
### Rest of code: Plot modes
###

# LAL hlm(t)
hlmT_lal = lalsimutils.SphHarmTimeSeries_to_dict(lalsimutils.hlmoft(wfP.P,opts.l),opts.l)
tvals_lal = lalsimutils.evaluate_tvals(hlmT_lal[(2,2)])  #float(hlmT_lal[(2,2)].epoch) + np.arange(len(hlmT_lal[(2,2)].data.data))*hlmT_lal[(2,2)].deltaT

# EOB hlm(t)
hlmT_eob = wfP.hlmoft(force_T=1./P.deltaF)
tvals_eob = lalsimutils.evaluate_tvals(hlmT_eob[(2,2)])

###
### Time domain : (a) rescaled EOB_tidal [here], (b) LAL, and (c) rescaled NR [internal]
###
print(" TIME DOMAIN PLOTS")
# Manually plot raw data
figindex =0
tmax = np.abs(wfP.waveform_modes_complex[(2,2)][-1,0])
for mode in wfP.waveform_modes_complex.keys():
 if opts.show_plots:
    figindex+=1; plt.figure(figindex)
    plt.xlim(tmax-0.1, tmax)
    plt.xlabel("$t (s)$")
    plt.ylabel("$|r/M \\tilde{h}_{lm}(t)|$")
    plt.plot(wfP.waveform_modes_complex[mode][:,0], np.abs(wfP.waveform_modes_complex[mode][:,1]),'r-',label=str(mode))
    plt.plot(wfP.waveform_modes_complex[mode][:,0], np.real(wfP.waveform_modes_complex[mode][:,1]),'r--',label=str(mode))
    plt.legend()
#    plt.plot(tvals_eob,np.abs(hlmT_eob[mode].data.data), label=str(mode)+'_eob')
#    plt.plot(tvals_eob,np.real(hlmT_eob[mode].data.data), label=str(mode)+'_eob')

#    plt.figure(99);
#    if np.abs(mode[1])>0:
    figindex+=1; plt.figure(figindex)
    plt.xlabel("$t (s)$")
    plt.ylabel("$arg \\tilde{h}_{lm}(t)$")
    datPhase = lalsimutils.unwind_phase(np.angle(wfP.waveform_modes_complex[mode][:,1]))    #needs to be unwound to be continuous
    plt.plot(wfP.waveform_modes_complex[mode][:,0], datPhase,label=str(mode))
    

figindex=20
for mode in hlmT_lal.keys():
  if opts.show_plots:
    figindex+=1
    plt.figure(figindex)
    plt.xlim(-1, 0.1)
    plt.xlabel("$t (s)$")
    plt.ylabel("$| \\tilde{h}_{lm}(t)|$")
    tvals = lalsimutils.evaluate_tvals(hlmT_lal[mode])
    plt.plot(tvals, np.abs(hlmT_lal[mode].data.data),'r-' ,label=str(mode)+"_lal")
    plt.plot(tvals, np.real(hlmT_lal[mode].data.data),'r--' ,label=str(mode)+"_lal")
    if mode in hlmT_eob:
        plt.plot(tvals_eob, np.abs(hlmT_eob[mode].data.data),'b-' ,label=str(mode)+"_eob")
        plt.plot(tvals_eob, np.real(hlmT_eob[mode].data.data),'b--' ,label=str(mode)+"_eob")

        hlm_NR = wfP.waveform_modes_complex[mode]
        scaleFactorM = eobT.MsunInSec*(wfP.P.m1+wfP.P.m2)/lal.MSUN_SI
        scaleFactorDinSeconds =  wfP.P.dist/lal.C_SI
        hlm_NR[:,1] *=  scaleFactorM/scaleFactorDinSeconds  # Nominally we report r h in units of M
        plt.plot(hlm_NR[:,0], np.abs(hlm_NR[:,1]),'k', label=str(mode)+"_eob_manual")

    plt.legend()

 
print("FREQUENCY VERSUS TIME")
if opts.show_plots:
    datPhase= lalsimutils.unwind_phase(np.angle(hlmT_eob[(2,2)].data.data))
    nStride=4
    freq = (np.roll(datPhase,nStride/2) - np.roll(datPhase,-nStride/2))/(P.deltaT*nStride)
    datPhase_lal= lalsimutils.unwind_phase(np.angle(hlmT_lal[(2,2)].data.data))
    freq_lal = (datPhase_lal - np.roll(datPhase_lal,-1))/P.deltaT
#print len(datPhase), len(freq), len(tvals_eob), len(tvals)
    plt.figure(50)
    plt.plot(tvals_eob, freq,label='eob')
    plt.plot(tvals, freq_lal,label='lal')
    plt.plot(tvals, np.ones(len(tvals))*P.fmin) # should be tangent to starting point
    plt.ylim(0,2000)
    plt.xlim(-50,0)
    plt.legend()

    plt.show()


###
### Frequency domain : (a) rescaled NR [here], (b) LAL, and (c) rescaled NR [internal]
###

# NR fourier: Via rescaling (cleaned) dimensionless FFT
plt.figure(1)
#plt.rc('text',usetex=True)
plt.xlabel("$\log_{10} f (Hz)$")
plt.ylabel('$\log_{10}|\\tilde{h}_{lm}(f)|$')
plt.ylim(-38,-20)  # astrophysical scale
plt.xlim(0.5,4)    # plausible frequency range
plt.figure(2)
plt.xlabel("$f (Hz)$")
plt.ylabel("$|\\tilde{h}_{lm}(f)|$")
plt.xlim(-200,200)  # plausible frequency range for high-mass sources
plt.figure(3)
plt.xlabel("$f (Hz)$")
plt.ylabel("$|\\tilde{h}_{lm}(f)|^2/S_h(f)$")
plt.xlim(-200,200)  # plausible frequency range for high-mass sources
plt.figure(6)

# NR fourier via hlmoff
hlmF_NR_lal = wfP.hlmoff(  deltaT=wfP.P.deltaT,force_T=1./P.deltaF)  
fvals_NR_lal = lalsimutils.evaluate_fvals(hlmF_NR_lal[(2,2)])
# lal 
hlm_lal = lalsimutils.SphHarmFrequencySeries_to_dict(lalsimutils.hlmoff(wfP.P,opts.l),opts.l)
fvals_lal = lalsimutils.evaluate_fvals(hlm_lal[(2,2)])


for mode in hlm_lal.keys():
 if opts.show_plots:
    if mode[1]>0 and mode in hlmF_NR_lal.keys():
        print(" Handling mode ", mode)
        # Re-evaluate the frequency sampling each time
        fvals_lal =lalsimutils.evaluate_fvals(hlm_lal[mode])
        fvals_NR_lal =lalsimutils.evaluate_fvals(hlmF_NR_lal[mode])
        print(mode, len(fvals_lal), len(fvals_NR_lal))


        plt.figure(1); plt.xlim(1, 4); 
        plt.plot(np.log10(fvals_lal), np.log10(np.abs(hlm_lal[mode].data.data)),eobT.mode_line_style[mode], label=str(mode)+"_lal")
        plt.plot(np.log10(fvals_NR_lal), np.log10(np.abs(hlmF_NR_lal[mode].data.data)),eobT.mode_line_style[mode], label=str(mode)+"_eob",lw=2) 
        plt.figure(2); plt.xlim(20, 200);
        plt.plot(fvals_lal, np.abs(hlm_lal[mode].data.data),eobT.mode_line_style[mode], label=str(mode)+"_lal")
        plt.plot(fvals_NR_lal, np.abs(hlmF_NR_lal[mode].data.data),eobT.mode_line_style[mode], label=str(mode)+"_eob",lw=2)
        plt.figure(3); plt.xlim(20, 200);
        datSh_lal = map(lalsim.SimNoisePSDaLIGOZeroDetHighPower, np.abs(fvals_lal))
        datSh_eob = map(lalsim.SimNoisePSDaLIGOZeroDetHighPower, np.abs(fvals_NR_lal))
        plt.plot(fvals_lal, np.abs(hlm_lal[mode].data.data)**2/datSh_lal,eobT.mode_line_style[mode], label=str(mode)+"_lal")
        plt.plot(fvals_NR_lal, np.abs(hlmF_NR_lal[mode].data.data)**2/datSh_eob,eobT.mode_line_style[mode], label=str(mode)+"_eob",lw=2)


if opts.show_plots:
    plt.figure(1);plt.legend()
    plt.figure(2);plt.legend()
    plt.figure(3);plt.legend()
    plt.show()


###
### Inner products
###

print(hlm_lal[(2,2)].deltaF, len(hlm_lal[(2,2)].data.data))
print(hlmF_NR_lal[(2,2)].deltaF, len(hlmF_NR_lal[(2,2)].data.data))
IP = lalsimutils.CreateCompatibleComplexOverlap(hlm_lal)

for mode in hlm_lal:
 if mode in hlmF_NR_lal:
    print(mode, IP.norm(hlm_lal[mode]), IP.norm(hlmF_NR_lal[mode]), IP.ip(hlm_lal[mode],hlmF_NR_lal[mode])/(IP.norm(hlm_lal[mode]) * IP.norm(hlmF_NR_lal[mode])))
