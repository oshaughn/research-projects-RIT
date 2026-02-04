#! /usr/bin/env python
#
#  EXAMPLES
#  python check_waveform_taper.py --approx SEOBNRv4PHM  # important ! 
#
# RESULTS
#
# WATCH OUT FOR
#   - tapering that depends on the segment length (!)
#   - tapering that is notably different depending on the waveform model (and hence interface: gwsignal vs lalsimulation vs FD)

import numpy as np
from matplotlib import pyplot as plt
import argparse
import lal

import RIFT
import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as factored_likelihood  # direct hoft call
import RIFT.physics.GWSignal as rift_gws

parser = argparse.ArgumentParser()
parser.add_argument("--approximant",type=str,default="IMRPhenomPv2")
parser.add_argument("--fd-standoff-factor",type=float,default=0.9,help="NOT STANDARD yet")
parser.add_argument("--fiducial",action='store_true')
parser.add_argument("--extreme",action='store_true')
parser.add_argument("--mtot",default=40,type=float)
parser.add_argument("--fmin",default=20,type=float)
parser.add_argument("--use_gwsignal",action='store_true')
parser.add_argument("--use-extra-fd-args",action='store_true')
parser.add_argument("--use-xphm-spintaylor",action='store_true') # see https://git.ligo.org/asimov/data/-/blob/main/analyses/bilby-bbh/analysis_bilby_IMRPhenomXPHM-SpinTaylor.yaml?ref_type=heads
parser.add_argument("--use-same-fref",action='store_true')
parser.add_argument("--rom-group",default=None)
parser.add_argument("--rom-param",default=None)
parser.add_argument("--Lmax",default=5,type=int)
parser.add_argument("--tmin",default=-5,type=float)
parser.add_argument("--tmax",default=0.1,type=float)
parser.add_argument("--force-psi",default=None,type=float)
parser.add_argument("--force-phase-shift-factor-of-pi",default=0,type=float)
parser.add_argument("--force-aligned",action='store_true')
parser.add_argument("--force-zero-inclination",action='store_true')
parser.add_argument("--force-no-condition",action='store_true')
parser.add_argument("--plot-modes-extensively",action='store_true')
parser.add_argument("--use-lalsim-condition",action='store_true')
parser.add_argument("--verbose",action='store_true')
opts=  parser.parse_args()

P = lalsimutils.ChooseWaveformParams()
P.ampO=-1  # need this otherwise we don't get SpinTaylor HM output
P.phaseO = 7 # so we have less insane outputs
if not(opts.force_no_condition):
   P.taper = lalsimutils.lsu_TAPER_START
if False: #not(opts.fiducial):
   P.randomize()   
else:
   P.m2 = P.m1/1.9
   P.theta = 0.1  # irrelevant/unused
   P.phi = 1.3    # irrelevant/unused
   P.incl = 0.2
   P.phiref = 5
   P.psi = 0.9
   P.s1x = 0.0
   P.s1y = -0.7
   P.s1z = 0.0
   P.s2x = 0.4
   P.s2z = -0.3
   if opts.extreme:
       P.s1x=0.99
       P.s1y=P.s1z=0

P.dist = RIFT.likelihood.factored_likelihood.distMpcRef*1e6*lal.PC_SI  # fiducial reference distance
P.assign_param('mtot',opts.mtot*lal.MSUN_SI)
if opts.force_aligned:
    P.s1x = P.s1y=P.s2x=P.s2y=0
if not(opts.use_gwsignal):
    if opts.approximant != "TaylorT4": # not default setting
        P.approx = lalsimutils.lalsim.GetApproximantFromString(opts.approximant)  # allow user to override the approx setting. Important for NR followup, where no approx set in sim_xml!
else:
    P.approx = opts.approximant

P.print_params()

# hoft via hlm, using exactly the function call we use in production
extra_args ={}
extra_waveform_args ={}
extra_waveform_args['fd_centering_factor']= 0.9
extra_waveform_args['fd_standoff_factor']=opts.fd_standoff_factor
if opts.use_extra_fd_args:
    extra_waveform_args['fd_L_frame'] = True
if opts.force_no_condition:
    extra_waveform_args['no_condition']=True
if opts.use_xphm_spintaylor:
    extra_waveform_args['FinalSpinMod'] =2
    extra_waveform_args['PhenomXPHMReleaseVersion'] = 122022
    extra_waveform_args['PrecVersion'] = 320
if opts.use_lalsim_condition:
    extra_waveform_args['condition'] = True # in current documentation as laldict option


def analyze_waveform(seglen, srate, **kwargs):
 P.deltaT=1./srate
 P.deltaF = 1./seglen
 P.fref = 22
 P.fmin=opts.fmin
 if opts.use_same_fref:
    P.fref = P.fmin

 if not(opts.force_psi is None):
    P.psi = opts.force_psi
 if opts.force_zero_inclination:
    P.incl = 0


 P_copy = P.manual_copy() # beware, call may change P!
 P_copy.print_params()
 hlmF_1, _= factored_likelihood.internal_hlm_generator(P_copy, opts.Lmax, use_gwsignal=opts.use_gwsignal, use_gwsignal_approx=opts.approximant,ROM_group=opts.rom_group,ROM_param=opts.rom_param, extra_waveform_kwargs=extra_waveform_args, **extra_args)
 hlmT_1  = {}
 for mode in hlmF_1:
    #print(mode,hlmF_1[mode].data.data[0])
    hlmT_1[mode] = lalsimutils.DataInverseFourier(hlmF_1[mode])
    print(mode, np.max(np.abs(hlmT_1[mode].data.data))*np.abs(lal.SpinWeightedSphericalHarmonic(P.incl,- P.phiref,-2,mode[0],mode[1])) ) #,"\t\t", 1./hlmF_1[mode].deltaF, hlmT_1[mode].deltaT*hlmT_1[mode].data.length)
 P_copy  = P.manual_copy()
 hTc_1 = lalsimutils.hoft_from_hlm(hlmT_1, P_copy,return_complex=True,extra_phase_shift=np.pi*opts.force_phase_shift_factor_of_pi)
 if opts.verbose:
    print('net ', np.max(np.abs(hTc_1.data.data)))

 # hoft direct. Currently lalsuite only
 if not(opts.use_gwsignal):
  hTc_2  = lalsimutils.complex_hoft(P,extra_waveform_args=extra_waveform_args)
 else:
  hTc_2 = rift_gws.complex_hoft(P, approx_string=opts.approximant, extra_waveform_args=extra_waveform_args)
 if opts.verbose:
  print('net2 ', np.max(np.abs(hTc_2.data.data)))

# now confirm complex_hoft dependence on psi is as desired
#    NOT THE SAME PSI DEPENDENCE AS WE ASSUME ELSEWHERE
# psi_ref = float(P.psi)
# P.psi = 0
# hTc_3 = lalsimutils.complex_hoft(P,extra_waveform_args=extra_waveform_args); P.psi = psi_ref
# hTc_3.data.data *= np.exp(-2j*P.psi)

# if opts.verbose:
#    dh =np.max(np.abs(hTc_3.data.data - hTc_2.data.data))
#    print(" Max diff psi - confirm psi coding correct ", dh )
#    print(" Duration check ", len(hTc_1.data.data), len(hTc_2.data.data), len(hTc_3.data.data))

 tvals1 = lalsimutils.evaluate_tvals(hTc_1)
 tvals2 = lalsimutils.evaluate_tvals(hTc_2)
 indx1 = np.argmax(np.abs(hTc_1.data.data))
 indx2 = np.argmax(np.abs(hTc_2.data.data))

 plt.title(opts.approximant)
 if opts.verbose:
    print( tvals1[indx1], np.angle(hTc_1.data.data[indx1]), P.psi )
    print( tvals2[indx2], np.angle(hTc_2.data.data[indx2]), P.psi )
 plt.plot(tvals1, np.abs(hTc_1.data.data),c='k',label="internal_{}_{}".format(seglen,srate))
 plt.plot(tvals2, np.abs(hTc_2.data.data),c='r',label="hoft")


 #plt.xlim(np.min([tvals1[0],tvals2[0] ]), 0.1)

analyze_waveform(256,4096)  # what happens for injections, if we are not careful!
analyze_waveform(32,4096)
analyze_waveform(16,4096)
analyze_waveform(8,4096)
analyze_waveform(4,4096)


plt.xlim(opts.tmin,opts.tmax)
plt.title(opts.approximant)
plt.legend()
plt.savefig("comp_{}.png".format(opts.approximant))
