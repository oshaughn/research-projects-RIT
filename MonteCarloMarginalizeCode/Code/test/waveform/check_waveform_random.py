#! /usr/bin/env python
#
#  EXAMPLES
#   python check_waveform_random.py --approx SpinTaylorT4 --Lmax 4
#   python check_waveform_random.py  --force-psi 0.1
#   python ./check_waveform_random.py --approx SpinTaylorT4 --force-psi 0 --use-same-fref --force-aligned
#   python check_waveform_random.py --approx SEOBNRv5EHM --force-aligned --use-eccentric --use-gwsignal --inj mdc.xml.gz --event 15
#   python check_waveform_random.py --approximant IMRPhenomXPHM --fiducial --use-extra-fd-args --assert-overlap 0.998
#   python check_waveform_random.py --approximant IMRPhenomXPHM --stress-q4-edge --seglen 16 --use-extra-fd-args --assert-overlap 0.995
#
# RESULTS
#   - Pv2: perfect
#   - XPHM: some small phase/alignment issues.  - fixed
#   - gwsignal
#      - SEOB: perfect, now
#       - NRSur
#
# WATCH OUT FOR
#   - starting too close too isco
#   - wraparound issues (segment length)
#   - different WF interfaces: eg, some use the IMRPv2 interface fallback, some use ChooseTDModes

# NRSur: use LAL_DATA_PATH, eg from below, and use a larger total mass
#   scp ldas:/scratch/lalsimulation/NRSur7dq4_v1.0.h5 .
#    export LAL_DATA_PATH=`pwd`
#

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import argparse
import lal
import sys

import RIFT
import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as factored_likelihood  # direct hoft call
import RIFT.physics.GWSignal as rift_gws

parser = argparse.ArgumentParser()
parser.add_argument("--approximant",type=str,default="IMRPhenomPv2")
parser.add_argument("--fiducial",action='store_true')
parser.add_argument("--stress-q4-edge", action='store_true')
parser.add_argument("--stress-near-aligned", action='store_true')
parser.add_argument("--mtot",default=40,type=float)
parser.add_argument("--fmin",default=20,type=float)
parser.add_argument("--seglen", default=8, type=int)
parser.add_argument("--use-gwsignal",action='store_true')
parser.add_argument("--use-extra-fd-args",action='store_true')
parser.add_argument("--use-xphm-spintaylor",action='store_true') # see https://git.ligo.org/asimov/data/-/blob/main/analyses/bilby-bbh/analysis_bilby_IMRPhenomXPHM-SpinTaylor.yaml?ref_type=heads
parser.add_argument("--use-same-fref",action='store_true')
parser.add_argument("--rom-group",default=None)
parser.add_argument("--rom-param",default=None)
parser.add_argument("--Lmax",default=5,type=int)
parser.add_argument("--tmin",default=-0.5,type=float)
parser.add_argument("--tmax",default=0.1,type=float)
parser.add_argument("--force-psi",default=None,type=float)
parser.add_argument("--force-phase-shift-factor-of-pi",default=0,type=float)
parser.add_argument("--force-aligned",action='store_true')
parser.add_argument("--force-zero-inclination",action='store_true')
parser.add_argument("--use-eccentric",action='store_true')
parser.add_argument("--inj", default=None,help="inspiral XML file containing injection information.")
parser.add_argument("--event",type=int, default=None,help="event ID of injection XML to use.")
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--assert-overlap", type=float, default=None,
                    help="fail unless the mode-sum/direct flat-noise overlap, maximized over integer time and constant phase, reaches this value")
opts=  parser.parse_args()


def best_flat_overlap(a, b, delta_t, f_low, f_high):
    """Flat-noise complex overlap maximized over integer time and phase."""
    fa = np.fft.fft(a)
    fb = np.fft.fft(b)
    freqs = np.fft.fftfreq(len(a), delta_t)
    keep = np.logical_and(np.abs(freqs) >= f_low, np.abs(freqs) <= f_high)
    fa = np.where(keep, fa, 0)
    fb = np.where(keep, fb, 0)
    a_band = np.fft.ifft(fa)
    b_band = np.fft.ifft(fb)
    corr = signal.correlate(a_band, b_band, mode="full", method="fft")
    lags = signal.correlation_lags(len(a_band), len(b_band), mode="full")
    lag = int(lags[np.argmax(np.abs(corr))])
    b_shift = np.zeros_like(b_band)
    if lag >= 0:
        b_shift[lag:] = b_band[:len(b_band)-lag]
    else:
        b_shift[:lag] = b_band[-lag:]
    overlap = abs(np.vdot(a_band, b_shift))/(np.linalg.norm(a_band)*np.linalg.norm(b_shift))
    scale = np.vdot(b_shift, a_band)/np.vdot(b_shift, b_shift)
    return float(overlap), lag, scale

P = lalsimutils.ChooseWaveformParams()
P.ampO=-1  # need this otherwise we don't get SpinTaylor HM output
P.phaseO = 7 # so we have less insane outputs
P.taper = lalsimutils.lsu_TAPER_START
if not(opts.fiducial or opts.stress_q4_edge or opts.stress_near_aligned) and not(opts.inj):
   print("Creating random event to use for plot comparison.")
   P.randomize()
   # move inside conditional for use inj purposes
   P.dist = RIFT.likelihood.factored_likelihood.distMpcRef*1e6*lal.PC_SI  # fiducial reference distance
   P.assign_param('mtot',opts.mtot*lal.MSUN_SI)

   if opts.use_eccentric:
      P.eccentricity = np.random.uniform(0.0,0.4) #for safety, for now
      P.meanPerAno = np.random.uniform(0.0,2*np.pi)

elif opts.inj:
   if not(opts.event):
      print("ERROR: must specify event to use.")
      sys.exit(0)
   else:
      ## as in lalwriteframe
      from igwn_ligolw import lsctables, table, utils # check all are needed
      filename = opts.inj
      event = opts.event
      print(f"Using event {event} from {filename}.")
      xmldoc = utils.load_filename(filename, verbose = True, contenthandler =lalsimutils.cthdler)
      sim_inspiral_table = lsctables.SimInspiralTable.get_table(xmldoc)
      P.copy_sim_inspiral(sim_inspiral_table[int(event)])
      P.tref = 0.0 ## force this for plotting purposes only - safe?
else:
   print("Using fiducial event parameters.")
   P.m2 = P.m1/1.5
   P.theta = 0.1  # irrelevant/unused
   P.phi = 1.3    # irrelevant/unused
   P.incl = 0.2
   P.phiref = 5
   P.psi = 0.9
   P.s1x = 0.1
   P.s1y = -0.6
   P.s1z = 0.4
   P.s2x = 0.4
   P.s2z = -0.3
   if opts.use_eccentric:
      P.eccentricity = 0.15
      P.meanPerAno = np.pi
   # move inside conditional for use inj purposes
   P.dist = RIFT.likelihood.factored_likelihood.distMpcRef*1e6*lal.PC_SI  # fiducial reference distance
   P.assign_param('mtot',opts.mtot*lal.MSUN_SI)

   if opts.stress_q4_edge:
      P.m1, P.m2 = 60*lal.MSUN_SI, 15*lal.MSUN_SI
      P.incl, P.phiref, P.psi = 1.2, 1.1, 0.4
      P.s1x, P.s1y, P.s1z = 0.75, 0.0, 0.25
      P.s2x, P.s2y, P.s2z = -0.20, 0.25, -0.10
   elif opts.stress_near_aligned:
      P.m1, P.m2 = 40*lal.MSUN_SI, 20*lal.MSUN_SI
      P.incl, P.phiref, P.psi = 1.2, 1.1, 0.4
      P.s1x, P.s1y, P.s1z = 1e-6, 0.0, 0.4
      P.s2x, P.s2y, P.s2z = 0.0, 0.0, -0.2

if opts.force_aligned:
    P.s1x = P.s1y=P.s2x=P.s2y=0
if not(opts.use_gwsignal):
    if opts.approximant != "TaylorT4": # not default setting
        P.approx = lalsimutils.lalsim.GetApproximantFromString(opts.approximant)  # allow user to override the approx setting. Important for NR followup, where no approx set in sim_xml!
else:
    P.approx = opts.approximant

if opts.approximant == "SEOBNRv5EHM":
   # temp workaround - needs a solution
   P.taper = lalsimutils.lsu_TAPER_NONE
   
P.deltaT=1./4096
P.deltaF = 1./opts.seglen
P.fref = 22
if opts.stress_q4_edge:
    P.fref = 30
P.fmin=opts.fmin
if opts.use_same_fref:
    P.fref = P.fmin

if not(opts.force_psi is None):
    P.psi = opts.force_psi
if opts.force_zero_inclination:
    P.incl = 0
P.print_params()

# hoft via hlm, using exactly the function call we use in production
extra_waveform_args ={}
extra_waveform_args['fd_centering_factor']= 0.9
if opts.use_extra_fd_args:
    extra_waveform_args['fd_L_frame'] = True
if opts.use_xphm_spintaylor:
    extra_waveform_args['FinalSpinMod'] =2
    extra_waveform_args['PhenomXPHMReleaseVersion'] = 122022
    extra_waveform_args['PrecVersion'] = 320


P_copy = P.manual_copy() # beware, call may change P!
hlmF_1, _= factored_likelihood.internal_hlm_generator(P_copy, opts.Lmax, use_gwsignal=opts.use_gwsignal, use_gwsignal_approx=opts.approximant,ROM_group=opts.rom_group,ROM_param=opts.rom_param, extra_waveform_kwargs=extra_waveform_args)
hlmT_1  = {}
for mode in hlmF_1:
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
  # import astropy.units as u
  # python_dict = {'mass1' : P.m1/lal.MSUN_SI * u.solMass,
  #             'mass2' : P.m2/lal.MSUN_SI * u.solMass,
  #             'spin1x' : P.s1x*u.dimensionless_unscaled,
  #             'spin1y' : P.s1y*u.dimensionless_unscaled,
  #             'spin1z' : P.s1z*u.dimensionless_unscaled,
  #             'spin2x' : P.s2x*u.dimensionless_unscaled,
  #             'spin2y' : P.s2y*u.dimensionless_unscaled,
  #             'spin2z' : P.s2z*u.dimensionless_unscaled,
  #             'deltaT' : P.deltaT*u.s,
  #             'f22_start' : P.fmin*u.Hz,
  #             'f22_ref': P.fref*u.Hz,
  #             'phi_ref' : P.phiref*u.rad,
  #             'distance' : P.dist/(1e6*lal.PC_SI)*u.Mpc,
  #             'inclination' : P.incl*u.rad,
  #             'eccentricity' : P.eccentricity*u.dimensionless_unscaled,
  #             'longAscNodes' : P.psi*u.rad,
  #             'meanPerAno' : P.meanPerAno*u.rad,
  #             'condition' : 1}
  # hp, hc = gws.core.waveform.GenerateTDWaveform(python_dict, gen)
  #hTc_2 = 
if opts.verbose:
  print('net2 ', np.max(np.abs(hTc_2.data.data)))

if opts.assert_overlap is not None:
    overlap, lag, scale = best_flat_overlap(
        np.asarray(hTc_1.data.data), np.asarray(hTc_2.data.data),
        float(hTc_1.deltaT), P.fmin, min(1024., 0.5/P.deltaT),
    )
    print("Mode-sum/direct overlap", overlap, "at lag", lag, "samples")
    if not np.isfinite(overlap) or overlap < opts.assert_overlap:
        raise SystemExit("FAIL: overlap {} is below {}".format(overlap, opts.assert_overlap))
    print("Best aligned complex scale", scale)
    if not np.isfinite(scale) or abs(np.angle(scale)) > 0.05 or not 0.8 < abs(scale) < 1.2:
        raise SystemExit("FAIL: mode-sum/direct phase or amplitude convention is inconsistent")

    # fd_L_frame modes are intrinsic.  Polarization is applied only when the
    # modes are summed into strain, so changing psi must not change any mode.
    P_psi = P.manual_copy()
    P_psi.psi += 0.37
    hlmF_psi, _ = factored_likelihood.internal_hlm_generator(
        P_psi, opts.Lmax, use_gwsignal=opts.use_gwsignal,
        use_gwsignal_approx=opts.approximant, ROM_group=opts.rom_group,
        ROM_param=opts.rom_param, extra_waveform_kwargs=extra_waveform_args,
    )
    relative_mode_difference = max(
        np.linalg.norm(hlmF_1[mode].data.data - hlmF_psi[mode].data.data)
        / max(np.linalg.norm(hlmF_1[mode].data.data), np.finfo(float).tiny)
        for mode in hlmF_1
    )
    print("Maximum relative mode change under psi shift", relative_mode_difference)
    if relative_mode_difference > 1e-12:
        raise SystemExit("FAIL: intrinsic modes depend on polarization angle")

# now confirm complex_hoft dependence on psi is as desired
#    NOT THE SAME PSI DEPENDENCE AS WE ASSUME ELSEWHERE
psi_ref = float(P.psi)
P.psi = 0
if opts.use_gwsignal:
   hTc_3 = rift_gws.complex_hoft(P,approx_string=opts.approximant); P.psi = psi_ref
else:
   hTc_3 = lalsimutils.complex_hoft(P); P.psi = psi_ref
hTc_3.data.data *= np.exp(-2j*P.psi)

if opts.verbose:
    dh =np.max(np.abs(hTc_3.data.data - hTc_2.data.data))
    print(" Max diff psi - confirm psi coding correct ", dh )
    print(" Duration check ", len(hTc_1.data.data), len(hTc_2.data.data), len(hTc_3.data.data))

tvals1 = lalsimutils.evaluate_tvals(hTc_1)
tvals2 = lalsimutils.evaluate_tvals(hTc_2)
indx1 = np.argmax(np.abs(hTc_1.data.data))
indx2 = np.argmax(np.abs(hTc_2.data.data))

insert_string = str(P.approx)
if not(isinstance(P.approx, str)):
   import lalsimulation as lalsim
   insert_string = lalsim.GetStringFromApproximant(P.approx)


plt.title(opts.approximant)
if opts.verbose:
    print( tvals1[indx1], np.angle(hTc_1.data.data[indx1]), P.psi )
    print( tvals2[indx2], np.angle(hTc_2.data.data[indx2]), P.psi )
plt.plot(tvals1, np.abs(hTc_1.data.data),c='k')
plt.plot(tvals2, np.abs(hTc_2.data.data),c='r')
plt.savefig(f"wf_{opts.approximant}_long_check.png")
plt.plot(tvals1, np.real(hTc_1.data.data),c='k',lw=1)
plt.plot(tvals2, np.real(hTc_2.data.data),c='r',lw=1)
#if dh > 1e-4 *np.max
plt.plot(tvals2, np.real(hTc_3.data.data),c='g',lw=1)
#plt.xlim(np.min([tvals1[0],tvals2[0] ]), 0.1)
plt.xlim(opts.tmin,opts.tmax)
plt.title(opts.approximant)

plt.savefig(f"wf_{opts.approximant}_check.png")
