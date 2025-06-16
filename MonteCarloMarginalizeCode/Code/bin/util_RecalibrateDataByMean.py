#! /usr/bin/env python

# EXAMPLE
#   ./util_FixProblemsAndRecenter.py --ifo H1 --start 999999852 --end 999999854 --channel H1:FAKE-STRAIN  H*.gwf 
#   util_FixProblemsAndRecenter.py --ifo H1 --start 1370292658 --end 1370292668 --channel GDS_CALIB_STRAIN_CLEAN_AR --calibration-file-path ~/ H1-GDS_CALIB_STRAIN_CLEAN_AR-1370292658-10.gwf 

from gwpy.timeseries import TimeSeries
import RIFT.lalsimutils as lalsimutils
#import glob
import sys
import bilby
import gwpy.time
from scipy.interpolate import make_interp_spline

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ifo",type=str)
parser.add_argument("--start",type=float)  # do not use! safer to pull in whole file, avoid start/end trouble with buggy gwpy time interface
parser.add_argument("--end",type=float)
parser.add_argument("--channel")
parser.add_argument("--validate-time-samples")
parser.add_argument("--calibration-file-path")
parser.add_argument("--fmin")
parser.add_argument("--fmax")
parser.add_argument("--output",default="output.gwf")
parser.add_argument("fnames",nargs="+")
opts = parser.parse_args()

dat = TimeSeries.read(opts.fnames, opts.channel) #,start=opts.start,end=opts.end)

if opts.validate_time_samples:
    # Test 0: fix noninteger part
    deltaT = dat.dt.to_value('s')
    srate = int(1./deltaT)
    t0 = dat.t0.to_value('s')
    t0_int = int(t0)
    t0_offset = t0-t0_int
    t0_offset_bad = t0_offset - int(t0_offset*srate)/srate
    if np.abs(t0_offset_bad)*srate > 0: # shift!
        print(' Sampling rate ', srate)
        print(' Bad offset ', t0_offset_bad)
        dat.epoch = gwpy.time.from_gps(t0_int + int(t0_offset*srate)/srate)


        
if opts.calibration_file_path:
    dat_lal = dat.to_lal()
    # Step 0: some bilby bookkeeping
    ifo  = bilby.gw.detector.get_empty_interferometer(opts.ifo)
    ifo.minimum_frequency = opts.fmin
    ifo.maximum_frequency = opts.fmax

    # Data format:
    #    log_frequency_array 0
    #    ampklitude_median 1
    #    phase_median 2
    # see : bilby/pw.prior.py parsing for this
    fname_cal = opts.calibration_file_path + "/" + opts.ifo + ".txt"  # assume standard envelope file names, eg output asimov retrieves
    calibration_data = np.genfromtxt(fname_cal).T
    log_frequency_array = np.log(calibration_data[0])
    amplitude_median = calibration_data[1] - 1
    phase_median = calibration_data[2]

    fn_A = make_interp_spline(log_frequency_array, amplitude_median)
    fn_Ph = make_interp_spline(log_frequency_array, phase_median)

    # Fourier transform data
    #  https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_freq_f_f_t__h.html
    datF = lalsimutils.DataFourierREAL8(dat_lal)
    npts = datF.data.length
    fvals = np.arange(datF.data.length)*datF.deltaF  # always start from 0
    # Don't bother outside of range
    indx_relevant = np.logical_and(fvals > np.min(calibration_data[0]), fvals <np.max(calibration_data[0]) )
    
    # Evaluate, apply mean response on frequency array, using grid of frequencies from above
    #    Default convention is 'data', which is the correct one of course
    datF.data.data[indx_relevant] *= fn_A( np.log( fvals[indx_relevant] ) )
    datF.data.data[indx_relevant] *= np.exp(1j*fn_Ph( np.log( fvals[indx_relevant]  ) ) )

    # Inverse fourier transform
    datT = lalsimutils.DataInverseFourierREAL8(datF)

    # Overwrite
    dat  = TimeSeries.from_lal(datT)

    
dat.write(opts.output)
