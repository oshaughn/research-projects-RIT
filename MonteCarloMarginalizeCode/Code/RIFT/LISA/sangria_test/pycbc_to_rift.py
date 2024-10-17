from pycbc.frame import read_frame
import matplotlib.pyplot as plt
import lal
import numpy as np
import h5py

import sys
RIFT = "RIFT-LISA-3G-O4c"
sys.path.append(f"/Users/aasim/Desktop/Research/Mcodes/{RIFT}/MonteCarloMarginalizeCode/Code")

import RIFT.lalsimutils as lsu
from RIFT.LISA.response.LISA_response import *
print(lsu.__file__)
from scipy.interpolate import interp1d

###########################################################################################
# Functions
###########################################################################################
def create_lal_COMPLEX16TimeSeries(pycbc_tseries):
    #ht_lal_real = lal.CreateREAL8TimeSeries("ht_lal", pycbc_tseries._epoch, 0, pycbc_tseries.delta_t, lal.DimensionlessUnit, len(pycbc_tseries.data))
    #ht_lal_real.data.data = pycbc_tseries.data
    # Resample
    # lal.ResampleREAL8TimeSeries(ht_lal_real, 10)
    #ht_lal = lal.CreateCOMPLEX16TimeSeries("ht_lal", ht_lal_real.epoch, 0, ht_lal_real.delta_t, lal.DimensionlessUnit, len(ht_lal_real.data))
    # Resize to 242 days, a power of 2 in seconds
    #ht_lal = lal.ResizeCOMPLEX16TimeSeries(ht_lal, 0, 4194304)
    old_tvals = np.arange(0, pycbc_tseries.delta_t*len(pycbc_tseries.data), pycbc_tseries.delta_t)
    new_tvals = np.arange(0, 31536000, 8)
    func = interp1d(old_tvals, pycbc_tseries.data)
    new_data = func(new_tvals)

    ht_lal = lal.CreateCOMPLEX16TimeSeries("ht_lal", pycbc_tseries._epoch, 0, 8, lal.DimensionlessUnit, len(new_data))    
    ht_lal.data.data = new_data + 0j
    ht_lal = lal.ResizeCOMPLEX16TimeSeries(ht_lal, 0, 4194304)
    print(f" Delta T = {ht_lal.deltaT} s, size = {ht_lal.data.length}, time = {ht_lal.data.length*ht_lal.deltaT/3600/24:2f} days") 
    return ht_lal

def create_injection_from_pycbc(pycbc_tseries, save_path):
    htA_lal, htE_lal, htT_lal =  create_lal_COMPLEX16TimeSeries(pycbc_tseries["A"]), create_lal_COMPLEX16TimeSeries(pycbc_tseries["E"]), create_lal_COMPLEX16TimeSeries(pycbc_tseries["T"])
    
    tvals = np.arange(0, htA_lal.data.length * htA_lal.deltaT, htA_lal.deltaT)

    plt.plot(tvals, htA_lal.data.data)
    plt.xlabel("Time [s]")
    plt.savefig(f"{save_path}/A_time.png")
    plt.cla()

    plt.plot(tvals, htE_lal.data.data)
    plt.xlabel("Time [s]")
    plt.savefig(f"{save_path}/E_time.png")
    plt.cla()

    plt.plot(tvals, htT_lal.data.data)
    plt.xlabel("Time [s]")
    plt.savefig(f"{save_path}/T_time.png")
    plt.cla()

    data_dict = {}
    data_dict["A"], data_dict["E"], data_dict["T"] =  lsu.DataFourier(htA_lal), lsu.DataFourier(htE_lal), lsu.DataFourier(htT_lal)

    fvals = -data_dict["A"].deltaF*np.arange(data_dict["A"].data.length//2, -data_dict["A"].data.length//2, -1)

    plt.loglog(fvals, 2*fvals*np.abs(data_dict["A"].data.data))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characteristic Strain")
    plt.savefig(f"{save_path}/A_frequency.png")
    plt.cla()

    plt.loglog(fvals, 2*fvals*np.abs(data_dict["E"].data.data))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characteristic Strain")
    plt.savefig(f"{save_path}/E_frequency.png")
    plt.cla()

    plt.loglog(fvals, 2*fvals*np.abs(data_dict["T"].data.data))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Characteristic Strain")
    plt.savefig(f"{save_path}/T_frequency.png")
    plt.cla()

    A_h5_file = h5py.File(f'{save_path}/A-fake_strain-1000000-10000.h5', 'w')
    A_h5_file.create_dataset('data', data=data_dict["A"].data.data)
    A_h5_file.attrs["deltaF"], A_h5_file.attrs["epoch"], A_h5_file.attrs["length"], A_h5_file.attrs["f0"] = data_dict["A"].deltaF, float(data_dict["A"].epoch), data_dict["A"].data.length, data_dict["A"].f0 
    A_h5_file.close()

    E_h5_file = h5py.File(f'{save_path}/E-fake_strain-1000000-10000.h5', 'w')
    E_h5_file.create_dataset('data', data=data_dict["E"].data.data)
    E_h5_file.attrs["deltaF"], E_h5_file.attrs["epoch"], E_h5_file.attrs["length"], E_h5_file.attrs["f0"] =  data_dict["E"].deltaF, float(data_dict["E"].epoch), data_dict["E"].data.length, data_dict["E"].f0
    E_h5_file.close()

    T_h5_file = h5py.File(f'{save_path}/T-fake_strain-1000000-10000.h5', 'w')
    T_h5_file.create_dataset('data', data=data_dict["T"].data.data)
    T_h5_file.attrs["deltaF"], T_h5_file.attrs["epoch"], T_h5_file.attrs["length"], T_h5_file.attrs["f0"] = data_dict["T"].deltaF, float(data_dict["T"].epoch), data_dict["T"].data.length, data_dict["T"].f0
    T_h5_file.close()
    

    
