#! /usr/bin/env python

import argparse
import numpy as np
from pesummary.gw.file.strain import StrainData
from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design
import matplotlib.pyplot as plt
import RIFT.lalsimutils as lalsimutils
import RIFT.physics.GWSignal as gws
import lal
import lalsimulation as lalsim
import os
import re
import sys
from pesummary.core.plots.interpolate import Bounded_interp1d
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False

## USAGE
# python plot-waveform-ci.py --approx 'SEOBNRv5EHM' --samples 'extrinsic-1000-draws.dat' --event-name 'GW150914'

## TODO
# - if extrinsic file is default, take a downsample for speed??
# - can this be turned into a dag? waveform gen is slow

#################################################### 
### setup - user must supply event name, and change approx/samples file if desired
#################################################### 

parser = argparse.ArgumentParser()
parser.add_argument("--approx",default='SEOBNRv4',type=str,help="Approximant. REQUIRED")
parser.add_argument("--samples",default='extrinsic_posterior_samples.dat',type=str,help="Posterior samples used to generate 90% CI")
parser.add_argument("--downselect", action="store_true", help="Downselect number of waveforms to generate. Try 1000")
parser.add_argument("--draws",default=None,type=int,help="MUST be set if using downselect. Try 1000")
parser.add_argument("--event-name", default='GW',type=str,help="event name for plot titles and filenames")
opts=  parser.parse_args()

# defaults - can this be read in from anywhere??
event_name = opts.event_name

set_approx_str = opts.approx
samples = opts.samples

# assuming this is run in a standard RIFT directory
with open("ILE.sub", "r") as f:
    content = f.read()
match = re.search(r'--event-time\s+([\d.]+)', content)
if match:
    tevent = float(match.group(1))
    print(f"Extracted event time: {tevent}")
else:
    print("ERROR: event time not found.")
    sys.exit()

def get_random_lines(filename, num_lines):
    with open(filename) as f:
        lines = f.readlines()
    header = lines[0]
    data_lines = lines[1:]
    return header, random.sample(data_lines, num_lines)

## needs to be tested!!!
if opts.downselect:
    if not opts.draws > 0:
        print("ERROR: you must set draws is using downselect")
        sys.exit()
    header, random_lines = get_random_lines(samples, draws)
    with open(f'{samples}-downsample.dat',w) as f:
        f.write(header)
        for line in random_lines:
            f.write(line)
    samples = f'{samples}-downsample.dat'

# default waveform storage - avoid remaking this after initial build (slow)
waveform_filename = "waveform_realizations.npz"
    
####################################################
### Define necessary funcs - whiten, waveforms, CI
####################################################

# as in pesummary, bandpass/filter the data
# see gwosc tutorial
def whiten_strain(data, bp_freqs=[50,250], srate=4096.0, notches=[60., 120.,180.]):
    bp = filter_design.bandpass(*bp_freqs, srate)
    zpks = [filter_design.notch(line, srate) for line in notches]
    zpk = filter_design.concatenate_zpks(bp, *zpks)
    hfilt = data.filter(zpk, filtfilt=True)
    _strain = hfilt.crop(*hfilt.span.contract(1))
    return _strain

# make param list from sample file AND generate tvals/hoft PER detector - store them
def generate_waveform_realizations(samples_file, tevent, fmin=10., fref=10., approx=set_approx_str):
    samples = np.genfromtxt(samples_file, names=True, replace_space=None)
    P_list = []
    # so manual :( but it works
    for indx in np.arange(len(samples["m1"])):
        P = lalsimutils.ChooseWaveformParams()
        P.m1 = samples["m1"][indx] * lal.MSUN_SI
        P.m2 = samples["m2"][indx] * lal.MSUN_SI
        P.s1x = samples["a1x"][indx]
        P.s1y = samples["a1y"][indx]
        P.s1z = samples["a1z"][indx]
        P.s2x = samples["a2x"][indx]
        P.s2y = samples["a2y"][indx]
        P.s2z = samples["a2z"][indx]

        # ecc/mpa not always present
        if 'eccentricity' in samples.dtype.names:
            P.eccentricity = samples["eccentricity"][indx]
        if 'meanPerAno' in samples.dtype.names:
            P.meanPerAno = samples["meanPerAno"][indx]

        P.ampO = -1
        P.dist = samples["distance"][indx] * lal.PC_SI * 1.e6
        P.incl = samples["incl"][indx]
        P.theta = samples["dec"][indx]
        P.phi = samples["ra"][indx]
        P.psi = samples["psi"][indx]
        P.phiref = samples["phiorb"][indx] - np.pi/2
        P.tref = samples["time"][indx]
        P.radec = True
        P.detector = "H1"
        P.approx = approx
        P.deltaF = 1./16
        P.fmin = fmin
        P.fref = fref
        P_list.append(P)

    # generate waveforms for each set of parameters
    # double check, some redundancy here
    tvals_H_list, ht_H_list, tvals_L_list, ht_L_list = [], [], [], []
    for param in P_list:

        if param.approx.startswith('SEOBNR'):
            hoft_H = gws.hoft(param, approx_string=param.approx)
            param.detector = "L1"
            hoft_L = gws.hoft(param, approx_string=param.approx)
        else:
            param.approx = lalsim.GetApproximantFromString(param.approx)
            hoft_H = lalsimutils.hoft(param)
            param.detector = "L1"
            hoft_L = lalsimutils.hoft(param)
            
        tvals_H = lalsimutils.evaluate_tvals(hoft_H) - tevent
        indx_ok = np.logical_and(tvals_H > -0.5, tvals_H < 0.2)
        tvals_H = tvals_H[indx_ok]
        hoft_H = TimeSeries(hoft_H.data.data[indx_ok], name='Strain', t0=tvals_H[0], unit='dimensionless', dt=tvals_H[1] - tvals_H[0])
        hoft_H = whiten_strain(hoft_H)
        tvals_H_list.append(tvals_H)
        ht_H_list.append(hoft_H)

        tvals_L = lalsimutils.evaluate_tvals(hoft_L) - tevent
        indx_ok = np.logical_and(tvals_L > -0.5, tvals_L < 0.2)
        tvals_L = tvals_L[indx_ok]
        hoft_L = TimeSeries(hoft_L.data.data[indx_ok], name='Strain', t0=tvals_L[0], unit='dimensionless', dt=tvals_L[1] - tvals_L[0])
        hoft_L = whiten_strain(hoft_L)
        tvals_L_list.append(tvals_L)
        ht_L_list.append(hoft_L)

    return tvals_H_list, ht_H_list, tvals_L_list, ht_L_list

# generic function to compute the 90% CIs
# done as pesummary/gwosc tutorial
def compute_ci(tvals, waveform_realizations):
    new_t = np.arange(min(tvals[0]), max(tvals[0]), 1/4096.)

    td_waveform_array = np.array([
        Bounded_interp1d(times, amplitude, xlow=min(tvals[0]), xhigh=max(tvals[0]))(new_t)
        for times, amplitude in zip(tvals, waveform_realizations)
    ])
    
    upper = np.percentile(td_waveform_array, 95, axis=0)
    lower = np.percentile(td_waveform_array, 5, axis=0)

    return new_t, upper, lower

####################################################
### get data and call functions to get waveforms ###
####################################################

# fetch open data, pesummary
H1_data = StrainData.fetch_open_data('H1', tevent - 20, tevent + 5)
L1_data = StrainData.fetch_open_data('L1', tevent - 20, tevent + 5)
# whiten
_strain_H = whiten_strain(H1_data)
times_H = [t-tevent for t in np.array(_strain_H.times)]
_strain_L = whiten_strain(L1_data)
times_L = [t-tevent for t in np.array(_strain_L.times)]

# do not rebuild waveforms for CI unless you have to
if os.path.exists(waveform_filename):
    print(f"File {waveform_filename} found. Loading existing waveforms.")

else:
    print(f"File {waveform_filename} not found. Generating waveforms.")
    tvals_H_list, ht_H_list, tvals_L_list, ht_L_list = generate_waveform_realizations(samples, tevent)
    np.savez(waveform_filename, 
             times_H=np.array(tvals_H_list), amplitude_H=np.array(ht_H_list),
             times_L=np.array(tvals_L_list), amplitude_L=np.array(ht_L_list))

# for consistency, load and use file either way
data = np.load(waveform_filename)
tvals_H = data["times_H"]
waveform_realizations_H = data["amplitude_H"]
tvals_L = data["times_L"]
waveform_realizations_L = data["amplitude_L"]

# get 90% CIs
new_t_H, upper_H, lower_H = compute_ci(tvals_H, waveform_realizations_H)
new_t_L, upper_L, lower_L = compute_ci(tvals_L, waveform_realizations_L)

####################################################
###  Make the plots ###
####################################################

# plot strain data
plt.figure(figsize=(20,5))
plt.plot(times_H,_strain_H,'gray',label='Real H1 strain',alpha=0.7,lw=2.5)
plt.plot(times_L,_strain_L,'gray',label='Real L1 strain',alpha=0.4,lw=2.5)
plt.xlim([-0.3,0.075])
plt.xlabel('Time (seconds) from '+str(tevent))
plt.ylabel('Strain')
plt.legend(loc='lower left')
plt.title(f'LIGO strain data near {event_name}')
plt.savefig(f"{event_name}_strain_data.png")

# plot h(t) waveforms, H as an example
plt.figure(figsize=(15,5))
for i in range(0,len(waveform_realizations_H)):
    plt.plot(np.array(tvals_H[i]), waveform_realizations_H[i],'gray',alpha=0.25)
plt.xlabel('Time (seconds) from '+str(tevent))
plt.ylabel('Strain')
plt.title('Waveform Comparison');
plt.xlim(-0.3,0.075);
plt.savefig(f"{event_name}_ht_waveforms.png", bbox_inches='tight')

## ci + data h1/l1 sep panels
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

axes[0].plot(times_H, _strain_H, 'gray', label='Real H1 strain', alpha=0.4, lw=2.5)
axes[0].fill_between(new_t_H, upper_H, lower_H, color='lightgreen', alpha=0.4, label=f'{set_approx_str} 90% CI')
axes[0].set_xlim([-0.3, 0.075])
axes[0].set_ylabel('strain')
axes[0].legend(loc='lower left')
axes[0].set_title(f'LIGO strain data near {event_name}',fontsize=24)
axes[0].text(.5,.9,'LIGO Hanford',horizontalalignment='center',transform=axes[0].transAxes)

axes[1].plot(times_L, _strain_L, 'gray', label='Real L1 strain', alpha=0.4, lw=2.5)
axes[1].fill_between(new_t_L, upper_L, lower_L, color='lightgreen', alpha=0.4, label=f'{set_approx_str} 90% CI')
axes[1].set_xlim([-0.3, 0.075])
axes[1].set_xlabel('time (seconds) from ' + str(tevent))
axes[1].set_ylabel('strain')
axes[1].legend(loc='lower left')
axes[1].text(.5,.9,'LIGO Livingston',horizontalalignment='center',transform=axes[1].transAxes)

plt.tight_layout()
plt.savefig(f"{event_name}_CI_H1L1.png", bbox_inches='tight')
