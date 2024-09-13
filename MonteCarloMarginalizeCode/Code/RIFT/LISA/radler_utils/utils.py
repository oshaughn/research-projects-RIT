import matplotlib.pyplot as plt
import lal
import numpy as np
import h5py
import sys
from scipy.interpolate import interp1d

RIFT = "RIFT-LISA-3G-O4c"
sys.path.append(f"/Users/aasim/Desktop/Research/Mcodes/{RIFT}/MonteCarloMarginalizeCode/Code")

import RIFT.lalsimutils as lsu
from RIFT.LISA.response.LISA_response import *

__author__ = "A. Jan"
###########################################################################################
# Functions
###########################################################################################
def create_resampled_lal_COMPLEX16TimeSeries(tvals, data_dict, new_tvals=None):
    """A helper function to create lal COMPLEX16TimeSeries.
        Args:
            tvals (numpy.array)    : time values over which the data is defined,
            data_dict (dictionary) : dictionary containing data stored in numpy array,
            new_tvals (numpy.array): resampled time values. (set to None if resampling is not needed).      
        Returns:
            data_dict              : dictionary containing resampled data stored as lal.COMPLEX16TimeSeries objects."""
    data_dict_new = {}
    for channel in data_dict.keys():
        print(f"Reading channel {channel}")
        if not(new_tvals is None):
        	# new_tvals passed as arguments, check if interpolation is needed.
            equal_length = np.equal(len(tvals), len(new_tvals))
            old_deltaT, new_deltaT = np.diff(tvals)[0], np.diff(new_tvals)[0]
            equal_deltaT = np.equal(old_deltaT, new_deltaT)
        else:
        	# new_tvals not passed as argument, set these to True to bypass interpolation.
            equal_length=True
            equal_deltaT=True
        if equal_length and equal_deltaT:
            new_deltaT = np.diff(tvals)[0]
            print("Resampling not requested.")
            new_data = data_dict[channel]
        else:
            new_deltaT = np.diff(new_tvals)[0]
            print(f"Resampling from {old_deltaT} s to {new_deltaT} s")
            func = interp1d(tvals, data_dict[channel], fill_value=tuple([0,0]), bounds_error=False)
            new_data = func(new_tvals)

        ht_lal = lal.CreateCOMPLEX16TimeSeries("ht_lal", 0.0, 0, new_deltaT, lal.DimensionlessUnit, len(new_data))    
        ht_lal.data.data = new_data + 0j
        print(f" Delta T = {ht_lal.deltaT} s, size = {ht_lal.data.length}, time = {ht_lal.data.length*ht_lal.deltaT/3600/24:2f} days") 
        data_dict_new[channel] = ht_lal
        
    return data_dict_new


def get_ldc_psds(save_path=None, fvals=None, channels = ["A", "E", "T"], model = "SciRDv1"):
    """This function generates LISA psds using the lisa data challenge package.
        Args:
            save_path (string)     : path where to save the psds as txt files,
            fvals (boolean)        : frequency values on which you want to evaluate the PSD, if None then it will generate frequency valuee,
            channel (list)         : list of channels ["A", "E", "T", "X", "XY"],
            model (string)         : "Proposal", "SciRDv1", "SciRDdeg1", "MRDv1","MRD_MFR","mldc", "newdrs", "LCESAcall", "redbook".
        Returns:
            psd_dictionary"""
    try:
        import ldc.lisa.noise as noise
    except:
        print("ldc is not installed.")
        sys.exit()
    if fvals is None:
        fmin = 0.0
        deltaT = 8
        fNyq = 0.5/deltaT
        deltaF = 1/(4194304*deltaT)
        fvals = np.arange(fmin, fNyq, deltaF)
        print(f"Generating psds using model = {model}, fmin = {fmin}Hz, fmax = {fNyq}Hz, 1/deltaF = {4194304*deltaT}s, deltaT = {deltaT}s.")
    noise_model = noise.get_noise_model(model, fvals)
    Sn = {}
    Sn["fvals"] = fvals
    for channel in channels:
        print(f"Channel = {channel}")
        Sn[channel] = noise_model.psd(fvals, channel)
        if save_path:
             np.savetxt(f"{save_path}/{channel}_psd.txt", np.vstack([Sn["fvals"], Sn[channel]]).T)
                    
    return Sn

def generate_data_from_radler(h5_path, output_as_AET = False, new_tvals =  None, output_as_FD = False, condition=True):
    """This function takes in a radler h5 file and outputs a data dictionary.
        Args:
            h5_path (string)       : path to radler h5 file,
            output_as_AET (boolean): set to True if you want the data as A, E, T,
            new_tvals (numpy array): pass the new tvals to truncate and/or resample the data,
            output_as_FD (boolean) : set to True if you want the data in frequency domain,
        Returns:
            data_dictionary"""
    # Load data
    data = h5py.File(h5_path)
    # Extract X, Y, Z data
    XYZ_data = np.array(data["H5LISA"]['PreProcess']['TDIdata'])
    # DeltaT
    cadence  = np.array(data["H5LISA"]["GWSources"]['MBHB-0']['Cadence'])
    # Save as dictionary
    data_dict = {}
    data_dict["X"], data_dict["Y"], data_dict["Z"] = XYZ_data[:,1], XYZ_data[:,2], XYZ_data[:,3]
    # tvals for this data
    old_tvals = XYZ_data[:,0]

    # Convert into AET if requested
    if output_as_AET:
        tmp_dict = data_dict
        data_dict = {}
        data_dict["A"] = 1/np.sqrt(2) * (tmp_dict["Z"] - tmp_dict["X"])
        data_dict["E"] = 1/np.sqrt(6) * (tmp_dict["X"] - 2*tmp_dict["Y"] + tmp_dict["Z"])
        data_dict["T"] = 1/np.sqrt(3) * (tmp_dict["X"] + tmp_dict["Y"] + tmp_dict["Z"])

    # new_tvals are none by default, so if they are not provided no interpolatin will occur and this function will just create lal tseries.
    data_dict = create_resampled_lal_COMPLEX16TimeSeries(old_tvals, data_dict, new_tvals)
    
    # Condition if requested
    if condition:
        print("\tTapering requested")
        for channel in data_dict:
            TDlen = (data_dict[channel].data.length)
            ntaper = int(0.01*TDlen) 
            # taper start of the time series
            vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
            print(f"\t\t Tapering from index 0 ({vectaper[0]}) to {ntaper}.")
            data_dict[channel].data.data[:ntaper] *= vectaper # 0 at 0 and slowly peak 
            # taper end of the time series
            index_front = TDlen-ntaper
            print(f"\t\t Tapering from index {index_front} ({vectaper[::-1][0]}) to {TDlen-1}.")
            data_dict[channel].data.data[index_front:] *= vectaper[::-1]  # slowly drop and then 0 at -1
    
    # Convert into FD if requested
    if output_as_FD:
        if new_tvals is None:
            power = np.log2(len(old_tvals))
        else:
            power = np.log2(len(new_tvals))
        assert power == np.ceil(power), "The data bins need to be power of 2 for lal FFT routines, make sure len(new_tvals) is a power of 2."
        tmp_dict = data_dict
        data_dict = {}
        for channel in tmp_dict:
            data_dict[channel] = lsu.DataFourier(tmp_dict[channel])

    return data_dict


def get_radler_mbhb_params(h5_path):
    """This function takes in a radler h5 file and outputs the parameter of the MBHB injection as a dictionary.
        Args:
            h5_path (string)       : path to radler h5 file,
        Returns:
            parameter dictionary"""
    # Load data
    data = h5py.File(h5_path)
    # Extract params
    pGW = data["H5LISA"]["GWSources"]['MBHB-0']
    params = {}
    params["m1"] = np.array(pGW.get('Mass1'))
    params["m2"] = np.array(pGW.get('Mass2'))
    params["chi1"] = np.array(pGW.get('Spin1')*np.cos(pGW.get('PolarAngleOfSpin1')))
    params["chi2"] = np.array(pGW.get('Spin2')*np.cos(pGW.get('PolarAngleOfSpin2')))

    theL = np.array(pGW.get('InitialPolarAngleL'))
    phiL = np.array(pGW.get('InitialAzimuthalAngleL'))
    longt = np.array(pGW.get('EclipticLongitude'))
    lat = np.array(pGW.get('EclipticLatitude'))

    params["tc"] = np.array(pGW.get('CoalescenceTime'))
    params["phi0"] = np.array(pGW.get('PhaseAtCoalescence'))
    params["DL"]  = np.array(pGW.get('Distance'))

    dist  = params["DL"]  * 1.e6 * lal.PC_SI
    # print ("DL = ", DL*1.e-3, "Gpc")
    params["beta"] = np.array(lat)
    params["lambda"] = np.array(longt)
    params["incl"] = np.array(np.arccos( np.cos(theL)*np.sin(lat) + np.cos(lat)*np.sin(theL)*np.cos(longt-phiL)))

    up_psi = np.array(np.sin(params["beta"])*np.sin(theL)*np.cos(params["lambda"] - phiL) - np.cos(theL)*np.cos(params["beta"]))
    down_psi = np.array(np.sin(theL)*np.sin(params["lambda"] - phiL))
    params["psi"] = np.array(np.arctan2(up_psi, down_psi))
    params["z"] = np.array(pGW.get("Redshift"))

    return params
