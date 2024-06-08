## 
# TODO: 
# Use argparser
# use mdc.xml.gz


import sys
RIFT = "RIFT-LISA-3G-O4c"
sys.path.append(f"/Users/aasim/Desktop/Research/Mcodes/{RIFT}/MonteCarloMarginalizeCode/Code")


import numpy as np
import h5py
import RIFT.lalsimutils as lalsimutils
from scripts_LISA_test.response.LISA_response import *
import lal
import lalsimulation
import matplotlib.pyplot as plt

# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument("--inj", default=None, help="Inspiral XML file containing injection information.")
# parser.add_argument("--calculate-snr", default=False, help="Calculate SNR of the fake signal.")
# parser.add_argument("--psd-path", default=None, help="Path to a xml.gz PSD needed to calculate SNR.")
# parser.add_argument("--event", default=0, help="Event ID of injection XML to use.")
# opts = parser.parse_args()
# P_list = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))

#############################################
injection_save_path = "/Users/aasim/Desktop/Research/Projects/RIFT_LISA/Development/injections"
psd_path = "/Users/aasim/Desktop/Research/Mcodes/RIFT-LISA-3G-O4c/MonteCarloMarginalizeCode/Code/scripts_LISA_test/psd_generation/A-psd.xml.gz"

P = lalsimutils.ChooseWaveformParams()

P.m1 = 1e6 * lal.MSUN_SI
P.m2 = 5e5 * lal.MSUN_SI
P.s1z = 0.2
P.s2z = 0.4
P.dist = 120e3  * lal.PC_SI * 1e6 
P.fmin = 8*10**(-5)
P.fmax = 0.125
P.deltaF = 1/(32*32768)
P.deltaT = 0.5/P.fmax
P.approx = lalsimulation.GetApproximantFromString("IMRPhenomHM")

P.phiref = 0.0   # add phiref later (confirm with ROS)!
P.inclination = 0.0 # add inclination later (confirm with ROS)!
P.fref = 8*10**(-5) # what happens?
P.tref = 0.0

lmax = 2
modes = [(2,2), (3,3), (3,2)]
beta  = np.pi/6
lamda = np.pi/5
psi = np.pi/4
phi_ref = np.pi/3
inclination = np.pi/2

snr_fmin = 10**(-4)
snr_fmax = 0.125
##############################################
# Functions
def calculate_snr(data_dict, fmin, fmax, fNyq, psd):
    """This function calculates zero-noise snr of a LISA signal."""
    assert data_dict["A"].deltaF == data_dict["E"].deltaF == data_dict["T"].deltaF
    print(f"Integrating from {fmin} to {fmax} Hz.")
    # create instance of inner product
    IP = lalsimutils.ComplexIP(fmin, fmax, fNyq, data_dict["A"].deltaF, psd, False, False, 0.0,)
    # calculate SNR of each channel 
    A_snr, E_snr, T_snr = np.sqrt(IP.ip(data_dict["A"], data_dict["A"])), np.sqrt(IP.ip(data_dict["E"], data_dict["E"])), np.sqrt(IP.ip(data_dict["T"], data_dict["T"]))
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
    plt.legend()
    # place x-y limits
    plt.gca().set_ylim([10**(-24), 10**(-17)])
    plt.gca().set_xlim([10**(-4), 1])
    plt.grid(alpha = 0.5)
    # save
    plt.savefig(injection_save_path + "/injection-psd.png", bbox_inches = "tight")
##############################################

# generate modes
print(f"ChooseWaveformParams set to with approx {lalsimulation.GetStringFromApproximant(P.approx)}: \n {P.__dict__}")
hlmf = lalsimutils.hlmoff_for_LISA(P, Lmax=lmax, modes=modes) 
modes = list(hlmf.keys())

# create injections
data_dict = create_lisa_injections(hlmf, P.fmax, beta, lamda, psi, inclination, phi_ref, P.tref) 

# save them in h5 format
A_h5_file = h5py.File(f'{injection_save_path}/A-fake_strain-1000000-10000.h5', 'w')
A_h5_file.create_dataset('data', data=data_dict["A"].data.data)
A_h5_file.attrs["deltaF"], A_h5_file.attrs["epoch"], A_h5_file.attrs["length"], A_h5_file.attrs["f0"] = hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0 
A_h5_file.close()

E_h5_file = h5py.File(f'{injection_save_path}/E-fake_strain-1000000-10000.h5', 'w')
E_h5_file.create_dataset('data', data=data_dict["E"].data.data)
E_h5_file.attrs["deltaF"], E_h5_file.attrs["epoch"], E_h5_file.attrs["length"], E_h5_file.attrs["f0"] =  hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0
E_h5_file.close()

T_h5_file = h5py.File(f'{injection_save_path}/T-fake_strain-1000000-10000.h5', 'w')
T_h5_file.create_dataset('data', data=data_dict["T"].data.data)
T_h5_file.attrs["deltaF"], T_h5_file.attrs["epoch"], T_h5_file.attrs["length"], T_h5_file.attrs["f0"] = hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0
T_h5_file.close()

# calculate SNR
print(f"Reading PSD to calculate SNR for LISA instrument from {psd_path}.")
psd = lalsimutils.get_psd_series_from_xmldoc(psd_path, "A")
psd = lalsimutils.resample_psd_series(psd, P.deltaF)
psd_fvals = psd.f0 + P.deltaF*np.arange(psd.data.length)
psd.data.data[ psd_fvals < snr_fmin] = 0 
snr = calculate_snr(data_dict, snr_fmin, snr_fmax, 0.5/P.deltaT, psd)

# plot figure
create_PSD_injection_figure(data_dict, psd, injection_save_path, snr)

