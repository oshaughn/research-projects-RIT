import numpy as np
import RIFT.lalsimutils as lsu
from RIFT.LISA.response.LISA_response import *
import lal
import lalsimulation
import matplotlib.pyplot as plt
import os

__author__ = "A. Jan"


def load_psd(param_dict):
    """
    Load Power Spectral Density (PSD) data for the LISA instrument. 

    Parameters:
    param_dict (dict): A dictionary containing parameters for loading the PSD, 
                       should contain 'psd_path', 'deltaF', and 'snr_fmin'.

    Returns:
    dict: A dictionary containing PSD for A, E, and T channels.
    """
    print(f"Reading PSD to calculate SNR for LISA instrument from {param_dict['psd_path']}.")
    psd = {}
    psd["A"] = lsu.get_psd_series_from_xmldoc(param_dict["psd_path"] + "/A-psd.xml.gz", "A")
    psd["A"] = lsu.resample_psd_series(psd["A"],  param_dict['deltaF'])
    psd_fvals = psd["A"].f0 + param_dict['deltaF']*np.arange(psd["A"].data.length)
    psd["A"].data.data[ psd_fvals < param_dict['snr_fmin']] = 0 

    psd["E"] = lsu.get_psd_series_from_xmldoc(param_dict["psd_path"]+ "/E-psd.xml.gz", "E")
    psd["E"] = lsu.resample_psd_series(psd["E"],  param_dict['deltaF'])
    psd_fvals = psd["E"].f0 + param_dict['deltaF']*np.arange(psd["E"].data.length)
    psd["E"].data.data[ psd_fvals < param_dict['snr_fmin']] = 0

    psd["T"] = lsu.get_psd_series_from_xmldoc(param_dict["psd_path"]+ "/T-psd.xml.gz", "T")
    psd["T"] = lsu.resample_psd_series(psd["T"],  param_dict['deltaF'])
    psd_fvals = psd["T"].f0 +  param_dict['deltaF']*np.arange(psd["T"].data.length)
    psd["T"].data.data[ psd_fvals < param_dict['snr_fmin']] = 0
    return psd

def calculate_snr(data_dict, fmin, fmax, fNyq, psd):
    """
    Calculate the zero-noise Signal-to-Noise Ratio (SNR) for LISA signals.

    Parameters:
    data_dict (dict): A dictionary containing the A, E, and T signal data,
    fmin (float): The minimum frequency for integration in Hz,
    fmax (float): The maximum frequency for integration in Hz,
    fNyq (float): The Nyquist frequency in Hz,
    psd (dict): A dictionary containing the PSD for A, E, and T channels.

    Returns:
    float: The total zero-noise SNR calculated across all channels.
    """

    assert data_dict["A"].deltaF == data_dict["E"].deltaF == data_dict["T"].deltaF
    print(f"Integrating from {fmin} to {fmax} Hz.")

    # create instance of inner product 
    IP_A = lsu.ComplexIP(fmin, fmax, fNyq, psd["A"].deltaF, psd["A"], False, False, 0.0,)
    IP_E = lsu.ComplexIP(fmin, fmax, fNyq, psd["A"].deltaF, psd["E"], False, False, 0.0,)
    IP_T = lsu.ComplexIP(fmin, fmax, fNyq, psd["A"].deltaF, psd["T"], False, False, 0.0,)

    # calculate SNR of each channel  (injections created are one sided f-series, but they should be two sided, so extra factor of 2) 
    A_snr, E_snr, T_snr = np.sqrt(2*IP_A.ip(data_dict["A"], data_dict["A"])), np.sqrt(2*IP_E.ip(data_dict["E"], data_dict["E"])), np.sqrt(2*IP_T.ip(data_dict["T"], data_dict["T"]))
    
    # combine SNR
    snr = np.real(np.sqrt(A_snr**2 + E_snr**2 + T_snr**2)) # SNR (zero noise) = sqrt(<h|h>)
    
    print(f"A-channel snr = {A_snr.real:0.3f}, E-channel snr = {E_snr.real:0.3f}, T-channel snr = {T_snr.real:0.3f},\n\tTotal SNR = {snr:0.3f}.")
    return snr

def create_PSD_injection_figure(data_dict, psd, injection_save_path, snr):
    """
    Create a frequency-domain injection figure with PSD plotted against A, E and T data..

    Parameters:
    data_dict (dict): A dictionary containing signal data for each channel,
    psd (dict):  A dictionary containing the PSD for A, E, and T channels,
    injection_save_path (str): The file path where the generated figure will be saved.
    snr (float): The SNR to display in the figure title.

    Returns:
    None: This function saves the figure to the specified path.
    """
    channels = list(data_dict.keys())
    fvals = get_fvals(data_dict[channels[0]])

    # plot data
    plt.title(f"Injection vs PSD (SNR = {snr:0.2f})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characterstic strain")
    psd_fvals = psd[channels[0]].f0 + data_dict[channels[0]].deltaF*np.arange(psd[channels[0]].data.length)

    for channel in channels:
        # For m > 0, hlm is define for f < 0 in lalsimulation. That's why abs is over fvals too.
        data = np.abs(2*fvals*data_dict[channel].data.data) # we need get both -m and m modes, right now this only has positive modes present.
        plt.loglog(-fvals, data, label = channel, linewidth = 1.2)
        plt.loglog(psd_fvals, np.sqrt(psd_fvals * psd[channel].data.data), label = channel + "-psd", linewidth = 0.8)
    
    plt.legend(loc="upper right")

    # place x-y limits
    plt.gca().set_ylim([10**(-24), 10**(-17)])
    plt.gca().set_xlim([10**(-4), 1])
    plt.grid(alpha = 0.5)

    # save
    plt.savefig(injection_save_path + "/injection-psd.png", bbox_inches = "tight")


def generate_lisa_TDI_dict(param_dict):
    print(param_dict)
    P = lsu.ChooseWaveformParams()
    P.m1 = param_dict["m1"] * lal.MSUN_SI
    P.m2 = param_dict["m2"] * lal.MSUN_SI
    P.s1x, P.s1y, P.s1z = 0.0, 0.0, param_dict["s1z"]
    P.s2x, P.s2y, P.s2z = 0.0, 0.0, param_dict["s2z"]
    P.dist = param_dict["dist"] * 1e6 * lal.PC_SI

    P.deltaT, P.deltaF = param_dict["deltaT"], param_dict["deltaF"]
    P.fref = param_dict["wf-fref"]
    P.approx = lalsimulation.GetApproximantFromString(param_dict["approx"])
    P.fmin, P.fmax = param_dict["fmin"], 0.5/P.deltaT
    P.psi, P.phiref, P.inclination, P.tref, P.theta, P.phi = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    P.eccentricity, P.meanPerAno = 0.0, 0.0

    modes = np.array(param_dict["modes"])
    lmax = np.max(modes[:,0])

    path_to_NR_hdf5 = param_dict["path_to_NR_hdf5"] if 'path_to_NR_hdf5' in param_dict else None

    number_of_bins = 1/(P.deltaF*P.deltaT)
    power_of_number_of_bins = np.log2(number_of_bins)
    assert power_of_number_of_bins == np.ceil(power_of_number_of_bins), f'Number of bins needs to be a power of 2, increase 1/deltaF from {1/P.deltaF} to {1/ (2**np.ceil(power_of_number_of_bins)*P.deltaT)}.'

    print("###############")
    if 1/P.deltaF/60/60/24 >0.5:
        print(f"Data length = {1/P.deltaF/60/60/24:2f} days.")
    else:
        print(f"Data length = {1/P.deltaF/60/60:2f} hrs.")


    print(f"\nWaveform is being generated with m1 = {P.m1/lsu.lsu_MSUN}, m2 = {P.m2/lsu.lsu_MSUN}, s1z = {P.s1z}, s2z = {P.s2z}, distance = {P.dist/1e6/lal.PC_SI}")
    print(f"deltaF = {P.deltaF}, fmin  = {P.fmin}, fmax = {P.fmax}, deltaT = {P.deltaT}, modes = {list(modes)}, lmax = {lmax}, tref = {param_dict['tref']}")
    print(f"phiref = {param_dict['phi_ref']}, psi = {param_dict['psi']}, inclination = {param_dict['inclination']}, beta = {param_dict['beta']}, lambda = {param_dict['lambda']}")
    print(f"path_to_NR_hdf5 = {path_to_NR_hdf5}, approx = {lalsimulation.GetStringFromApproximant(P.approx)}\n")
    print("###############")

    hlmf = lsu.hlmoff_for_LISA(P, Lmax=lmax, modes=modes, path_to_NR_hdf5=path_to_NR_hdf5) 
    modes = list(hlmf.keys())

    # create injections
    data_dict = create_lisa_injections(hlmf, P.fmax, param_dict["fref"], param_dict["beta"], param_dict["lambda"], param_dict["psi"], param_dict["inclination"], param_dict["phi_ref"], param_dict["tref"]) 
    return data_dict

def generate_lisa_injections(data_dict, param_dict, get_snr = True):
    if not(os.path.exists(param_dict['save_path'])):
        print(f"Provided path doesn't exist {param_dict['save_path']}, creating it.")
        os.mkdir(param_dict["save_path"])
    create_h5_files_from_data_dict(data_dict, param_dict["save_path"])
    cmd = f"util_WriteInjectionFile.py --parameter m1 --parameter-value  {param_dict['m1']} \
              --parameter m2 --parameter-value {param_dict['m2']} \
              --parameter s1x --parameter-value 0.0 --parameter s1y --parameter-value 0.0 --parameter s1z --parameter-value {param_dict['s1z']} \
              --parameter s2x --parameter-value 0.0 --parameter s2y --parameter-value 0.0 --parameter s2z --parameter-value {param_dict['s2z']}  \
              --parameter eccentricity --parameter-value 0 --approx {param_dict['approx']}  --parameter dist --parameter-value {param_dict['dist']}  \
              --parameter fmin --parameter-value {param_dict['fmin']}  --parameter incl --parameter-value {param_dict['inclination']}  \
              --parameter tref --parameter-value {param_dict['tref']}  --parameter phiref --parameter-value {param_dict['phi_ref']}  \
              --parameter theta --parameter-value {param_dict['beta']}  --parameter phi --parameter-value  {param_dict['lambda']}   \
              --parameter psi --parameter-value {param_dict['psi']} "
    print(f"Executing command to create mdc.xml.gz\n{cmd}")
    os.system(cmd)
    os.system(f"mv mdc.xml.gz {param_dict['save_path']}/mdc.xml.gz")
    os.system(f"ls {param_dict['save_path']}/*h5 | lal_path2cache > {param_dict['save_path']}/local.cache")
    os.system(f" util_SimInspiralToCoinc.py --sim-xml {param_dict['save_path']}/mdc.xml.gz --event 0 --ifo A --ifo E --ifo T ; mv coinc.xml {param_dict['save_path']}/coinc.xml")
    if get_snr and 'psd_path' in param_dict:
        psd = load_psd(param_dict)
        snr = calculate_snr(data_dict, param_dict['snr_fmin'], param_dict['snr_fmax'], 0.5/param_dict['deltaT'], psd)
        create_PSD_injection_figure(data_dict, psd, param_dict["save_path"], snr)
