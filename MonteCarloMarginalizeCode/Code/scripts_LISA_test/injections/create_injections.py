## 
# TODO: Make this a proper code to write injections for LISA (use hlmoft, FFT it and the again in time domain?)(include options to read in NR-h5files)
# Choose waveform params needs to print LISA related parameters
# Confirm if phiref needs to be zero in P or not
# Add functionality so that we include tref at fref


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
P.dist = 80e3  * lal.PC_SI * 1e6 
P.phiref = 0.0   # add phiref later (confirm with ROS)!
P.inclination = 0.0 # add inclination later (confirm with ROS)!
P.fref = 8*10**(-5)
P.fmin = 8*10**(-5)
P.fmax = 1
P.deltaF = 1/(8*32768)
P.deltaT = 0.5/P.fmax
P.approx = lalsimulation.GetApproximantFromString("IMRPhenomD")
P.tref = 0.0

beta  = np.pi/6
lamda = np.pi/5
psi = np.pi/4
phi_ref = np.pi/3
inclination = np.pi/2

snr_fmin = 10**(-4)
snr_fmax = 1
##############################################

def calculate_snr(data_dict, fmin, fmax, psd):
    """This function calculates zero-noise snr of a LISA signal."""
    
    assert data_dict["A"].deltaF == data_dict["E"].deltaF == data_dict["T"].deltaF
    print(f"Integrating from {fmin} to {fmax} Hz.")

    IP = lalsimutils.ComplexIP(fmin, fmax, fmax, data_dict["A"].deltaF, psd, False, False, 0.0,)  # second fmax is fNyq
    A_snr, E_snr, T_snr = np.sqrt(IP.ip(data_dict["A"], data_dict["A"])), np.sqrt(IP.ip(data_dict["E"], data_dict["E"])), np.sqrt(IP.ip(data_dict["T"], data_dict["T"]))

    snr = np.real(np.sqrt(A_snr**2 + E_snr**2 + T_snr**2)) # SNR = sqrt(<h|h>)
    print(f"A-channel snr = {A_snr.real:0.3f}, E-channel snr = {E_snr.real:0.3f}, T-channel snr = {T_snr.real:0.3f},\n\tTotal SNR = {snr:0.3f}.")
    return snr

def create_PSD_injection_figure(data_dict, psd, injection_save_path, snr):
    plt.title(f"Injection vs PSD (SNR = {snr:0.2f})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characterstic strain")
    channels = list(data_dict.keys())
    fmax = data_dict[channels[0]].data.length * 0.5 * data_dict[channels[0]].deltaF #assumed double sided
    fvals = np.arange(-fmax, fmax, data_dict[channels[0]].deltaF)
    for channel in channels:
        plt.loglog(fvals, 2*fvals*np.abs(data_dict[channel].data.data), label = channel, linewidth = 1.2)
    psd_fvals = psd.f0 + data_dict[channels[0]].deltaF*np.arange(psd.data.length)
    plt.loglog(psd_fvals, np.sqrt(psd_fvals * psd.data.data), label = "PSD", linewidth = 1.5, color = "cornflowerblue")
    plt.legend()
    plt.grid(alpha = 0.5)
    plt.savefig(injection_save_path + "/injection-psd.png", bbox_inches = "tight")
    
print(f"Choose waveform params set to with approx {lalsimulation.GetStringFromApproximant(P.approx)}: \n {P.__dict__}")
hlmf, hlmf_conj = lalsimutils.std_and_conj_hlmoff(P) 
modes = list(hlmf.keys())

data_dict = create_lisa_injections(hlmf, P.fmax, beta, lamda, psi, inclination, phi_ref, P.tref)

A_h5_file = h5py.File(f'{injection_save_path}/A-fake_strain-1000000-10000.h5', 'w')
A_h5_file.create_dataset('data', data=data_dict["A"].data.data)
A_h5_file.attrs["deltaF"], A_h5_file.attrs["epoch"], A_h5_file.attrs["length"], A_h5_file.attrs["f0"] = hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0 
A_h5_file.close()

E_h5_file = h5py.File(f'{injection_save_path}/E-fake_strain-1000000-10000.h5', 'w')
E_h5_file.create_dataset('data', data=data_dict["E"].data.data)
E_h5_file.attrs["deltaF"], E_h5_file["epoch"], E_h5_file["length"], E_h5_file["f0"] =  hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0
E_h5_file.close()

T_h5_file = h5py.File(f'{injection_save_path}/T-fake_strain-1000000-10000.h5', 'w')
T_h5_file.create_dataset('data', data=data_dict["T"].data.data)
T_h5_file["deltaF"], T_h5_file["epoch"], T_h5_file["length"], T_h5_file["f0"] = hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0
T_h5_file.close()

print(f"Reading PSD to calculate SNR for LISA instrument from {psd_path}")
psd = lalsimutils.get_psd_series_from_xmldoc(psd_path, "A")
psd = lalsimutils.resample_psd_series(psd, P.deltaF)
psd_fvals = psd.f0 + P.deltaF*np.arange(psd.data.length)
psd.data.data[ psd_fvals < P.fmin] = 0 # 
snr = calculate_snr(data_dict, snr_fmin, snr_fmax, psd)
create_PSD_injection_figure(data_dict, psd, injection_save_path, snr)

