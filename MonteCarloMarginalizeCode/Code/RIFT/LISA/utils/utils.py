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

YRSID_SI = 31558149.763545603
C_SI = lal.C_SI
ConstOmega = 1.99098659277e-7
OrbitR =149597870700.0

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
            save_path (string): path where to save the psds as txt files,
            fvals (boolean)   : frequency values on which you want to evaluate the PSD, if None then it will generate frequency valuee,
            channel (list)    : list of channels ["A", "E", "T", "X", "XY"],
            model (string)    : "Proposal", "SciRDv1", "SciRDdeg1", "MRDv1","MRD_MFR","mldc", "newdrs", "LCESAcall", "redbook".
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

def generate_data_from_radler(h5_path, output_as_AET = False, new_tvals =  None, output_as_FD = False, condition=True, taper_percent=0.0001):
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
            ntaper = int(taper_percent*TDlen) 
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


def generate_data_from_sangria(h5_path, output_as_AET = False, new_tvals = None, output_as_FD = False, condition=True, resize = False, add_noise = False, taper_percent=0.0001):
    """This function takes in a sangria h5 file and outputs a data dictionary.
        Args:
            h5_path (string)       : path to radler h5 file,
            output_as_AET (boolean): set to True if you want the data as A, E, T,
            new_tvals (numpy array): pass the new tvals to truncate and/or resample the data,
            output_as_FD (boolean) : set to True if you want the data in frequency domain,
        Returns:
            data_dictionary"""
    # Load data
    data = h5py.File(h5_path)
    # Extract X, Y, Z data for mbhb
    XYZ_data_mbhb = data['sky']['mbhb']['tdi']
    # Save as dictionary
    data_dict_mbhb =  {}
    data_dict_mbhb.update({'X':np.array(XYZ_data_mbhb['X']).squeeze(1),
                           'Y':np.array(XYZ_data_mbhb['Y']).squeeze(1),
                           'Z':np.array(XYZ_data_mbhb['Z']).squeeze(1)})
    old_tvals = np.array(XYZ_data_mbhb['t']).squeeze(1)

    # data_dict is our return dictionary, save as no-noise right now, add noise based on user input.
    data_dict = data_dict_mbhb

    # noise goes here
    # full data with noise and gbs
    if not(add_noise is False):
        print(f"Adding noise {add_noise}")
        full_data_dict = {}
        XYZ_data_full = data['obs']['tdi']
        full_data_dict.update({"X":np.array(XYZ_data_full['X']).squeeze(1),
                        "Y":np.array(XYZ_data_full['Y']).squeeze(1),
                        "Z":np.array(XYZ_data_full['Z']).squeeze(1)})
        if add_noise=='with_gbs':
            data_dict = full_data_dict
        elif add_noise=='without_gbs':
            # subtract mbhb signals
            noise_with_gb = {}
            for channel in ["X", "Y", "Z"]:
                noise_with_gb[channel] = full_data_dict[channel] - data_dict_mbhb[channel]
            # collect gbs
            noise_just_gb = {}
            noise_just_gb.update({"X":0, "Y":0, "Z":0})
            for i in ['v','d','i']:
                print(i)
                XYZ_data_gb = data['sky'][f'{i}gb']["tdi"]
                for channel in ["X", "Y", "Z"]:
                    noise_just_gb[channel] += np.array(XYZ_data_gb[channel]).squeeze(1)
            # just get noise
            noise_without_gb = {}
            for channel in ["X", "Y", "Z"]:
                noise_without_gb[channel] =  noise_with_gb[channel] - noise_just_gb[channel]
                data_dict[channel] = data_dict_mbhb[channel] + noise_without_gb[channel]

    # Convert into AET if requested
    if output_as_AET:
        tmp_dict = data_dict
        data_dict = {}
        data_dict["A"] = 1/np.sqrt(2) * (tmp_dict["Z"] - tmp_dict["X"])
        data_dict["E"] = 1/np.sqrt(6) * (tmp_dict["X"] - 2*tmp_dict["Y"] + tmp_dict["Z"])
        data_dict["T"] = 1/np.sqrt(3) * (tmp_dict["X"] + tmp_dict["Y"] + tmp_dict["Z"])

    # Condition if requested
    if condition:
        print("\tTapering requested")
        for channel in data_dict:
            TDlen = len(data_dict[channel])
            ntaper = int(taper_percent*TDlen) 
            # taper start of the time series
            vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
            print(f"\t\t Tapering from index 0 ({vectaper[0]}) to {ntaper}.")
            data_dict[channel][:ntaper] *= vectaper # 0 at 0 and slowly peak 
            # taper end of the time series
            index_front = TDlen-ntaper
            print(f"\t\t Tapering from index {index_front} ({vectaper[::-1][0]}) to {TDlen-1}.")
            data_dict[channel][index_front:] *= vectaper[::-1]  # slowly drop and then 0 at -1

    # Resizing
    if resize:
        old_power = np.log2(len(old_tvals))
        new_power = np.ceil(old_power)
        print(f"Resizing from {2**old_power*5/3600/24} to {2**new_power*5/3600/24}.")
        for channel in data_dict:
            tmp = np.zeros(int(2**new_power))
            tmp[:len(old_tvals)] = data_dict[channel]
            data_dict[channel] = tmp
        old_tvals = np.arange(0, len(data_dict[channel]), 1) * 5
    # new_tvals are none by default, so if they are not provided no interpolatin will occur and this function will just create lal tseries.
    data_dict = create_resampled_lal_COMPLEX16TimeSeries(old_tvals, data_dict, new_tvals)

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

def get_radler_mbhb_params(h5_path, dataset = "radler", sangria_signal=0):
    """This function takes in a radler h5 file and outputs the parameter of the MBHB injection as a dictionary.
        Args:
            h5_path (string): path to radler h5 file,
        Returns:
            parameter dictionary"""
    # Load data
    data = h5py.File(h5_path)
    if dataset == 'radler':
        pGW = data["H5LISA"]["GWSources"]['MBHB-0']
        # Extract params
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
    if dataset == 'sangria':
        pGW = data['sky']['mbhb']['cat'][sangria_signal]
        # Extract params
        params = {}
        params["m1"] = np.array(pGW['Mass1'])
        params["m2"] = np.array(pGW['Mass2'])
        params["chi1"] = np.array(pGW['Spin1']*np.cos(pGW['PolarAngleOfSpin1']))
        params["chi2"] = np.array(pGW['Spin2']*np.cos(pGW['PolarAngleOfSpin2']))

        theL = np.array(pGW['InitialPolarAngleL'])
        phiL = np.array(pGW['InitialAzimuthalAngleL'])
        longt = np.array(pGW['EclipticLongitude'])
        lat = np.array(pGW['EclipticLatitude'])

        params["tc"] = np.array(pGW['CoalescenceTime'])
        params["phi0"] = np.array(pGW['PhaseAtCoalescence'])
        params["DL"]  = np.array(pGW['Distance'])

        dist  = params["DL"]  * 1.e6 * lal.PC_SI
        # print ("DL = ", DL*1.e-3, "Gpc")
        params["beta"] = np.array(lat)
        params["lambda"] = np.array(longt)
        params["incl"] = np.array(np.arccos( np.cos(theL)*np.sin(lat) + np.cos(lat)*np.sin(theL)*np.cos(longt-phiL)))

        up_psi = np.array(np.sin(params["beta"])*np.sin(theL)*np.cos(params["lambda"] - phiL) - np.cos(theL)*np.cos(params["beta"]))
        down_psi = np.array(np.sin(theL)*np.sin(params["lambda"] - phiL))
        params["psi"] = np.array(np.arctan2(up_psi, down_psi))
        params["z"] = np.array(pGW["Redshift"])
    


    return params

def modpi(phase):
    """Modulus with pi as the period

    This function was originally in BBHx. 

    Args:
        phase (scalar or np.ndarray): Phase angle.

    Returns:
        scalar or np.ndarray: Phase angle modulus by pi.

    """
    # from sylvain
    return phase - np.floor(phase / np.pi) * np.pi

def tSSBfromLframe(tL, lambdaSSB, betaSSB, t0=0.0):
    """Get time in SSB frame from time in LISA-frame.

    Compute Solar System Barycenter time ``tSSB`` from retarded time at the center
    of the LISA constellation ``tL``. **NOTE**: depends on the sky position
    given in solar system barycenter (SSB) frame.

    This function was originally in BBHx. 

    Args:
        tL (scalar or np.ndarray): Time in LISA constellation reference frame.
        lambdaSSB (scalar or np.ndarray): Ecliptic longitude in
            SSB reference frame.
        betaSSB (scalar or np.ndarray): Ecliptic latitude in SSB reference frame.
        t0 (double, optional): Initial start time point away from zero.
            (Default: ``0.0``)

    Returns:
        scalar or np.ndarray: Time in the SSB frame.

    """
    ConstPhi0 = ConstOmega * t0
    phase = ConstOmega * tL + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return (
        tL
        + RoC * np.cos(betaSSB) * np.cos(phase)
        - 1.0 / 2 * ConstOmega * pow(RoC * np.cos(betaSSB), 2) * np.sin(2.0 * phase)
    )


# Compute retarded time at the center of the LISA constellation tL from Solar System Barycenter time tSSB */
def tLfromSSBframe(tSSB, lambdaSSB, betaSSB, t0=0.0):
    """Get time in LISA frame from time in SSB-frame.

    Compute retarded time at the center of the LISA constellation frame ``tL`` from
    the time in the SSB frame ``tSSB``. **NOTE**: depends on the sky position
    given in solar system barycenter (SSB) frame.

    This function was originally in BBHx. 

    Args:
        tSSB (scalar or np.ndarray): Time in LISA constellation reference frame.
        lambdaSSB (scalar or np.ndarray): Ecliptic longitude in
            SSB reference frame.
        betaSSB (scalar or np.ndarray): Time in LISA constellation reference frame.
        t0 (double, optional): Initial start time point away from zero.
            (Default: ``0.0``)

    Returns:
        scalar or np.ndarray: Time in the LISA frame.

    """
    ConstPhi0 = ConstOmega * t0
    phase = ConstOmega * tSSB + ConstPhi0 - lambdaSSB
    RoC = OrbitR / C_SI
    return tSSB - RoC * np.cos(betaSSB) * np.cos(phase)

def LISA_to_SSB(tL, lambdaL, betaL, psiL, t0=0.0):
    """Convert sky/orientation from LISA frame to SSB frame.

    Convert the sky and orientation parameters from the center of the LISA
    constellation reference to the SSB reference frame.

    The parameters that are converted are the reference time, ecliptic latitude,
    ecliptic longitude, and polarization angle.

    This function was originally in BBHx. 

    Args:
        tL (scalar or np.ndarray): Time in LISA constellation reference frame.
        lambdaL (scalar or np.ndarray): Ecliptic longitude in
            LISA reference frame.
        betaL (scalar or np.ndarray): Ecliptic latitude in LISA reference frame.
        psiL (scalar or np.ndarray): Polarization angle in LISA reference frame.
        t0 (double, optional): Initial start time point away from zero.
            (Default: ``0.0``)

    Returns:
        Tuple: (``tSSB``, ``lambdaSSB``, ``betaSSB``, ``psiSSB``)


    """

    t0 = t0 * YRSID_SI

    ConstPhi0 = ConstOmega * t0
    coszeta = np.cos(np.pi / 3.0)
    sinzeta = np.sin(np.pi / 3.0)
    coslambdaL = np.cos(lambdaL)
    sinlambdaL = np.sin(lambdaL)
    cosbetaL = np.cos(betaL)
    sinbetaL = np.sin(betaL)

    lambdaSSB_approx = 0.0
    betaSSB_approx = 0.0
    # Initially, approximate alpha using tL instead of tSSB - then iterate */
    tSSB_approx = tL
    for k in range(3):
        alpha = ConstOmega * tSSB_approx + ConstPhi0
        cosalpha = np.cos(alpha)
        sinalpha = np.sin(alpha)
        lambdaSSB_approx = np.arctan2(
            cosalpha * cosalpha * cosbetaL * sinlambdaL
            - sinalpha * sinbetaL * sinzeta
            + cosbetaL * coszeta * sinalpha * sinalpha * sinlambdaL
            - cosalpha * cosbetaL * coslambdaL * sinalpha
            + cosalpha * cosbetaL * coszeta * coslambdaL * sinalpha,
            cosbetaL * coslambdaL * sinalpha * sinalpha
            - cosalpha * sinbetaL * sinzeta
            + cosalpha * cosalpha * cosbetaL * coszeta * coslambdaL
            - cosalpha * cosbetaL * sinalpha * sinlambdaL
            + cosalpha * cosbetaL * coszeta * sinalpha * sinlambdaL,
        )
        betaSSB_approx = np.arcsin(
            coszeta * sinbetaL
            + cosalpha * cosbetaL * coslambdaL * sinzeta
            + cosbetaL * sinalpha * sinzeta * sinlambdaL
        )
        tSSB_approx = tSSBfromLframe(tL, lambdaSSB_approx, betaSSB_approx, t0)

    lambdaSSB_approx = lambdaSSB_approx % (2 * np.pi)
    #  /* Polarization */
    psiSSB = modpi(
        psiL
        + np.arctan2(
            cosalpha * sinzeta * sinlambdaL - coslambdaL * sinalpha * sinzeta,
            cosbetaL * coszeta
            - cosalpha * coslambdaL * sinbetaL * sinzeta
            - sinalpha * sinbetaL * sinzeta * sinlambdaL,
        )
    )

    return (tSSB_approx, lambdaSSB_approx, betaSSB_approx, psiSSB)

def SSB_to_LISA(tSSB, lambdaSSB, betaSSB, psiSSB, t0=0.0):
    """Convert sky/orientation from SSB frame to LISA frame.

    Convert the sky and orientation parameters from the SSB reference frame to the center of the LISA
    constellation reference frame.

    The parameters that are converted are the reference time, ecliptic latitude,
    ecliptic longitude, and polarization angle.

    This function was originally in BBHx. 

    Args:
        tSSB (scalar or np.ndarray): Time in SSB reference frame.
        lambdaSSB (scalar or np.ndarray): Ecliptic longitude in
            SSB reference frame.
        betaSSB (scalar or np.ndarray): Ecliptic latitude in SSB reference frame.
        psiSSB (scalar or np.ndarray): Polarization angle in SSB reference frame.
        t0 (double, optional): Initial start time point away from zero in years.
            (Default: ``0.0``)

    Returns:
        Tuple: (``tL``, ``lambdaL``, ``betaL``, ``psiL``)

    """
    t0 = t0 * YRSID_SI

    ConstPhi0 = ConstOmega * t0
    alpha = 0.0
    cosalpha = 0
    sinalpha = 0.0
    coslambda = 0
    sinlambda = 0.0
    cosbeta = 0.0
    sinbeta = 0.0

    coszeta = np.cos(np.pi / 3.0)
    sinzeta = np.sin(np.pi / 3.0)
    coslambda = np.cos(lambdaSSB)
    sinlambda = np.sin(lambdaSSB)
    cosbeta = np.cos(betaSSB)
    sinbeta = np.sin(betaSSB)

    alpha = ConstOmega * tSSB + ConstPhi0
    cosalpha = np.cos(alpha)
    sinalpha = np.sin(alpha)
    tL = tLfromSSBframe(tSSB, lambdaSSB, betaSSB, t0)
    lambdaL = np.arctan2(
        cosalpha * cosalpha * cosbeta * sinlambda
        + sinalpha * sinbeta * sinzeta
        + cosbeta * coszeta * sinalpha * sinalpha * sinlambda
        - cosalpha * cosbeta * coslambda * sinalpha
        + cosalpha * cosbeta * coszeta * coslambda * sinalpha,
        cosalpha * sinbeta * sinzeta
        + cosbeta * coslambda * sinalpha * sinalpha
        + cosalpha * cosalpha * cosbeta * coszeta * coslambda
        - cosalpha * cosbeta * sinalpha * sinlambda
        + cosalpha * cosbeta * coszeta * sinalpha * sinlambda,
    )
    betaL = np.arcsin(
        coszeta * sinbeta
        - cosalpha * cosbeta * coslambda * sinzeta
        - cosbeta * sinalpha * sinzeta * sinlambda
    )
    psiL = modpi(
        psiSSB
        + np.arctan2(
            coslambda * sinalpha * sinzeta - cosalpha * sinzeta * sinlambda,
            cosbeta * coszeta
            + cosalpha * coslambda * sinbeta * sinzeta
            + sinalpha * sinbeta * sinzeta * sinlambda,
        )
    )

    return np.vstack([tL, lambdaL, betaL, psiL]).T


def get_secondary_mode_for_skylocation(coalesence_time, lamda, beta, psi = 0.0, t0=0.0):
    """
    Calculate the secondary mode location for a given sky location in the SSB frame.

    Parameters:
    coalescence_time (float): The time of coalescence.
    lambda_param (float): The lambda parameter for the SSB frame.
    beta (float): The beta parameter, which is bimodal in the LISA frame.
    psi (float, optional): The psi parameter (default is 0.0, not necessary).
    t0 (float, optional): The initial time (default is 0.0).

    Returns:
    numpy.ndarray: The secondary mode parameters in the SSB frame, needed for grid generation.
    """
    lisa_params = SSB_to_LISA(coalesence_time, lamda, beta, psi, t0)
    lisa_params_new = lisa_params
    # in lisa frame only beta is bimodal (for full detector response) and its location is -beta_L
    lisa_params_new[0, 2] = -1.0*lisa_params_new[0, 2]
    SSB_params_new = LISA_to_SSB(lisa_params_new[0, 0], lisa_params_new[0, 1], lisa_params_new[0, 2], lisa_params_new[0, 3])
    return SSB_params_new
