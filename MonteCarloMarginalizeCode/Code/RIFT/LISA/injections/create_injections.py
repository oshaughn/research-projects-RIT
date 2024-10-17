#/usr/bin/env python 
# TODO: 
# Use argparser
# use mdc.xml.gz


import sys
import numpy as np
import h5py
import RIFT.lalsimutils as lalsimutils
from RIFT.LISA.response.LISA_response import *
import lal
import lalsimulation
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

__author__ = "A. Jan"
############################################################################################
# Arguments
###########################################################################################
parser = ArgumentParser()
parser.add_argument("--save-path", default=os.getcwd(), help="Path where you want to save the h5 files")
parser.add_argument("--psd-path", default=None, help="Path to a xml.gz PSD needed to calculate SNR.")
parser.add_argument("--inj", default=None, help="Inspiral XML file containing injection information.")
parser.add_argument("--fNyq", default=0.125, help="fNyq for generating waveforms")
parser.add_argument("--deltaF", default=1/(64*32768), help="DeltaF of the injectons")
parser.add_argument("--modes", default= "[(2,2),(2,1),(3,3),(3,2),(3,1),(4,4),(4,3),(4,2),(5,5)]", help="list of modes to use in injection.")
parser.add_argument("--path-to-NR-hdf5", default=None, help="path to NRhdf5 (LVK format) if using NR hdf5 for injection.")
parser.add_argument("--snr-fmin", default=0.0001, help="fmin while calculating SNR")
# parser.add_argument("--calculate-snr", default=False, help="Calculate SNR of the fake signal.")
# parser.add_argument("--event", default=0, help="Event ID of injection XML to use.")
opts = parser.parse_args()

###########################################################################################
# Injection parameters
###########################################################################################
P_list = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))
P_inj = P_list[0]
if not(opts.save_path):
    print("Save path not provided")
    opts.save_path = os.getcwd()
print(f"Saving frames in {opts.save_path}")

P = lalsimutils.ChooseWaveformParams()

P.m1 = P_inj.m1 
P.m2 = P_inj.m2
P.s1z = P_inj.s1z
P.s2z = P_inj.s2z
P.dist = P_inj.dist
P.fmin = P_inj.fmin
P.fmax = float(opts.fNyq)
P.deltaF = float(opts.deltaF)

P.deltaT = 0.5/P.fmax
P.phiref = 0.0   # add phiref later (confirm with ROS)!
P.inclination = 0.0 # add inclination later (confirm with ROS)!
P.psi = 0.0 # is not used when generating a waveform
P.fref = P_inj.fref # what happens?
P.tref = 0.0


modes = np.array(eval(opts.modes))
lmax  = np.max(modes[:,0])
beta  = P_inj.theta
lamda = P_inj.phi
psi = P_inj.psi
phi_ref = P_inj.phiref
inclination = P_inj.incl
tref = float(P_inj.tref)

fref = None
P.approx = P_inj.approx
path_to_NR_hdf5=opts.path_to_NR_hdf5

snr_fmin = float(opts.snr_fmin)
snr_fmax = float(opts.fNyq)
print("###############")
if 1/P.deltaF/60/60/24 >0.5:
    print(f"Data length = {1/P.deltaF/60/60/24} days.")
else:
    print(f"Data length = {1/P.deltaF/60/60} hrs.")


print(f"\nWaveform is being generated with m1 = {P.m1/lalsimutils.lsu_MSUN}, m2 = {P.m2/lalsimutils.lsu_MSUN}, s1z = {P.s1z}, s2z = {P.s2z}")
print(f"deltaF = {P.deltaF}, fmin  = {P.fmin}, fmax = {P.fmax}, deltaT = {P.deltaT}, modes = {list(modes)}, lmax = {lmax}, tref = {tref}")
print(f"phiref = {phi_ref}, psi = {psi}, inclination = {inclination}, beta ={beta}, lambda = {lamda}")
print(f"path_to_NR_hdf5 = {path_to_NR_hdf5}, approx = {lalsimulation.GetStringFromApproximant(P.approx)}\n")
print("###############")

###########################################################################################
# Functions to calculate SNR and plot 
###########################################################################################
def calculate_snr(data_dict, fmin, fmax, fNyq, psd):
    """This function calculates zero-noise snr of a LISA signal."""
    assert data_dict["A"].deltaF == data_dict["E"].deltaF == data_dict["T"].deltaF
    print(f"Integrating from {fmin} to {fmax} Hz.")
    # create instance of inner product
    IP_A = lalsimutils.ComplexIP(fmin, fmax, fNyq, data_dict["A"].deltaF, psd["A"], False, False, 0.0,)
    IP_E = lalsimutils.ComplexIP(fmin, fmax, fNyq, data_dict["A"].deltaF, psd["E"], False, False, 0.0,)
    IP_T = lalsimutils.ComplexIP(fmin, fmax, fNyq, data_dict["A"].deltaF, psd["T"], False, False, 0.0,)
    
    # calculate SNR of each channel 
    A_snr, E_snr, T_snr = np.sqrt(IP_A.ip(data_dict["A"], data_dict["A"])), np.sqrt(IP_E.ip(data_dict["E"], data_dict["E"])), np.sqrt(IP_T.ip(data_dict["T"], data_dict["T"]))
    # combine SNR
    snr = np.real(np.sqrt(A_snr**2 + E_snr**2 + T_snr**2)) # SNR (zero noise) = sqrt(<h|h>)
    print(f"A-channel snr = {A_snr.real:0.3f}, E-channel snr = {E_snr.real:0.3f}, T-channel snr = {T_snr.real:0.3f},\n\tTotal SNR = {snr:0.3f}.")
    return snr

def create_PSD_injection_figure(data_dict, psd, injection_save_path, snr):
    """This function create a FD injection figure with PSD."""
    channels = list(data_dict.keys())
    fmax = data_dict[channels[0]].data.length * 0.5 * data_dict[channels[0]].deltaF #assumed double sided
    fvals = np.arange(-fmax, fmax, data_dict[channels[0]].deltaF)

    plt.title(f"Injection vs PSD (SNR = {snr:0.2f})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characterstic strain")
    # plot data
    for channel in channels:
        # For m > 0, hlm is define for f < 0 in lalsimulation. That's why abs is over fvals too.
        data = np.abs(2*fvals*data_dict[channel].data.data) # we need get both -m and m modes, right now this only has positive modes present.
        plt.loglog(-fvals, data, label = channel, linewidth = 1.2)
    # plot PSD
    psd_fvals = psd.f0 + data_dict[channels[0]].deltaF*np.arange(psd.data.length)
    plt.loglog(psd_fvals, np.sqrt(psd_fvals * psd.data.data), label = "PSD", linewidth = 1.5, color = "cornflowerblue")
    plt.legend(loc="upper left")
    # place x-y limits
    plt.gca().set_ylim([10**(-24), 10**(-17)])
    plt.gca().set_xlim([10**(-4), 1])
    plt.grid(alpha = 0.5)
    # save
    plt.savefig(injection_save_path + "/injection-psd.png", bbox_inches = "tight")

###########################################################################################
# Generating injection
###########################################################################################
# generate modes
hlmf = lalsimutils.hlmoff_for_LISA(P, Lmax=lmax, modes=modes, path_to_NR_hdf5=path_to_NR_hdf5) 
modes = list(hlmf.keys())

# create injections
data_dict = create_lisa_injections(hlmf, P.fmax, fref, beta, lamda, psi, inclination, phi_ref, tref) 

# save them in h5 format
A_h5_file = h5py.File(f'{opts.save_path}/A-fake_strain-1000000-10000.h5', 'w')
A_h5_file.create_dataset('data', data=data_dict["A"].data.data)
A_h5_file.attrs["deltaF"], A_h5_file.attrs["epoch"], A_h5_file.attrs["length"], A_h5_file.attrs["f0"] = hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0 
A_h5_file.close()

E_h5_file = h5py.File(f'{opts.save_path}/E-fake_strain-1000000-10000.h5', 'w')
E_h5_file.create_dataset('data', data=data_dict["E"].data.data)
E_h5_file.attrs["deltaF"], E_h5_file.attrs["epoch"], E_h5_file.attrs["length"], E_h5_file.attrs["f0"] =  hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0
E_h5_file.close()

T_h5_file = h5py.File(f'{opts.save_path}/T-fake_strain-1000000-10000.h5', 'w')
T_h5_file.create_dataset('data', data=data_dict["T"].data.data)
T_h5_file.attrs["deltaF"], T_h5_file.attrs["epoch"], T_h5_file.attrs["length"], T_h5_file.attrs["f0"] = hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0
T_h5_file.close()

# calculate SNR
print(f"Reading PSD to calculate SNR for LISA instrument from {opts.psd_path}.")
psd = {}
psd["A"] = lalsimutils.get_psd_series_from_xmldoc(opts.psd_path + "/A-psd.xml.gz", "A")
psd["A"] = lalsimutils.resample_psd_series(psd["A"], P.deltaF)
psd_fvals = psd["A"].f0 + P.deltaF*np.arange(psd["A"].data.length)
psd["A"].data.data[ psd_fvals < snr_fmin] = 0 

psd["E"] = lalsimutils.get_psd_series_from_xmldoc(opts.psd_path + "/E-psd.xml.gz", "E")
psd["E"] = lalsimutils.resample_psd_series(psd["E"], P.deltaF)
psd_fvals = psd["E"].f0 + P.deltaF*np.arange(psd["E"].data.length)
psd["E"].data.data[ psd_fvals < snr_fmin] = 0

psd["T"] = lalsimutils.get_psd_series_from_xmldoc(opts.psd_path + "/T-psd.xml.gz", "T")
psd["T"] = lalsimutils.resample_psd_series(psd["T"], P.deltaF)
psd_fvals = psd["T"].f0 + P.deltaF*np.arange(psd["T"].data.length)
psd["T"].data.data[ psd_fvals < snr_fmin] = 0
snr = calculate_snr(data_dict, snr_fmin, snr_fmax, 0.5/P.deltaT, psd)

# plot figure
create_PSD_injection_figure(data_dict, psd, opts.save_path, snr)

