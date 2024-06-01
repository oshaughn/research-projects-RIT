## 
# TODO: Make this a proper code to write injections for LISA (use hoft, FFT it and the again in time domain?)(include options to read in NR-h5files)
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

# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument("--inj", default=None, help="Inspiral XML file containing injection information.")
# parser.add_argument("--calculate-snr", default=False, help="Calculate SNR of the fake signal.")
# parser.add_argument("--psd-path", default=None, help="Path to a xml.gz PSD needed to calculate SNR.")
# parser.add_argument("--event", default=0, help="Event ID of injection XML to use.")
# opts = parser.parse_args()
# P_list = lalsimutils.xml_to_ChooseWaveformParams_array(str(opts.inj))



# psd_path = ""
injection_save_path = "/Users/aasim/Desktop/Research/Projects/RIFT_LISA/Development/injections"

P = lalsimutils.ChooseWaveformParams()

P.m1 = 1e6 * lal.MSUN_SI
P.m2 = 5e5 * lal.MSUN_SI
P.s1z = 0.2
P.s2z = 0.4
P.dist = 18e3  * lal.PC_SI * 1e6 
P.phiref = 0.0   # add phiref later (confirm with ROS)!
P.inclination = 0.0 # add inclination later (confirm with ROS)!
P.fref = 0.0001
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

print(f"Choose waveform params set to with approx {lalsimulation.GetStringFromApproximant(P.approx)}: \n {P.__dict__}")
hlmf, hlmf_conj = lalsimutils.std_and_conj_hlmoff(P) 
modes = list(hlmf.keys())


tf_dict, f_dict, amp_dict, phase_dict = get_tf_from_phase_dict(hlmf, P.fmax)
A = 0
E = 0
T = 0
for mode in modes:
    l, m = mode[0], mode[1]
    H_0 = transformed_Hplus_Hcross(beta, lamda, psi, inclination, phi_ref, mode[0], mode[1])  # my function is define for marginalization
    L1, L2, L3 = Evaluate_Gslr(tf_dict[mode] + P.tref, f_dict[mode], H_0, beta, lamda)
    time_shifted_phase = phase_dict[mode] + 2*np.pi*P.tref*f_dict[mode]
    tmp_data = amp_dict[mode] * np.exp(1j*time_shifted_phase)  
    # I belive BBHx conjugates because the formalism is define for A*exp(-1jphase), but I need to check with ROS and Mike Katz.
    A += np.conj(tmp_data * L1)
    E += np.conj(tmp_data * L2)
    T += np.conj(tmp_data * L3)


A_h5_file = h5py.File(f'{injection_save_path}/A-fake_strain.h5', 'w')
A_h5_file.create_dataset('data', data=A)
A_h5_file.attrs["deltaF"], A_h5_file.attrs["epoch"], A_h5_file.attrs["length"], A_h5_file.attrs["f0"] = hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0 

E_h5_file = h5py.File(f'{injection_save_path}/E-fake_strain.h5', 'w')
E_h5_file.create_dataset('data', data=E)
E_h5_file.attrs["deltaF"], E_h5_file["epoch"], E_h5_file["length"], E_h5_file["f0"] =  hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0

T_h5_file = h5py.File(f'{injection_save_path}/T-fake_strain.h5', 'w')
T_h5_file.create_dataset('data', data=T)
T_h5_file["deltaF"], T_h5_file["epoch"], T_h5_file["length"], T_h5_file["f0"] = hlmf[modes[0]].deltaF, float(hlmf[modes[0]].epoch), hlmf[modes[0]].data.length, hlmf[modes[0]].f0
