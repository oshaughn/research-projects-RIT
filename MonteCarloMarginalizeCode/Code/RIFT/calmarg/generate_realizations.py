#
#
# OBJECTIVE: access to calibration realizations
#

# REFERENCE: https://github.com/bilby-dev/bilby/blob/main/bilby/gw/detector/calibration.py
#         see https://dcc.ligo.org/DocDB/0116/T1400682/001/calnote
#
# PROBLEM
#    - calibration factors are one-sided by default?  Need to be careful about 2-sided things
# EXAMPLE:
#      python  ./generate_realizations.py --fname H1.txt

#import bilby.gw.detector.calibration

import numpy as np
import h5py
import scipy.interpolate


def retrieve_envelope_from_file(fname, frequency_array=None,**kwargs):
    """
    retrieve_envelope_from_file
         fname : assumed currently to be ascii file.  Provide options for h5
             ascii format:
                   frequency  median_mag median_phase  16_mag  16_phase 84_mag 84_phase
             data convention: applying to data, see bilby file above.
             amplitude data is centered around *1*
         frequency_array : list of *positive* frequencies to interpolate to, for efficiency
    """
    dat = np.loadtxt(fname)

    
    # default frequencyes
    if frequency_array is None:
        out_amp = np.zeros( (len(dat),3))
        out_amp[:,0] = dat[:,0]
        out_amp[:,1] = dat[:,1]
        out_amp[:,2] = (dat[:,-2] - dat[:,3])/2   # 84-16 / 2 to get 1 sigma estimate
        out_phase =np.zeros( (len(dat),2))
        out_phase[:,0] = dat[:,0]
        out_phase[:,1] = dat[:,2]
        out_phase[:,2] = (dat[:,-1] - dat[:,4])/2
    else:
        out_amp = np.zeros( (len(frequency_array),3))
        out_phase = np.zeros( (len(frequency_array),3))
        out_amp[:,0 ] = frequency_array
        out_amp[:,1] = np.interp(frequency_array, dat[:,0], dat[:,1])
        out_amp[:,2] = np.interp(frequency_array, dat[:,0],(dat[:,-2] - dat[:,3])/2)
        out_phase[:,0 ] = frequency_array
        out_phase[:,1] = np.interp(frequency_array, dat[:,0], dat[:,2])
        out_phase[:,2] = np.interp(frequency_array, dat[:,0],(dat[:,-1] - dat[:,4])/2)

    return out_amp, out_phase


def nodes_to_spline_coefficients_matrix(n_points):
    # Follow calibrarion,py, in turn following      https://dcc.ligo.org/LIGO-T230
    # See Soichiro https://dcc.ligo.org/DocDB/0187/T2300140/001/interpolation_evenly_spaced.pdf for long hardcoded implementation of cubic splines inline

        tmp1 = np.zeros(shape=(n_points, n_points))
        tmp1[0, 0] = -1
        tmp1[0, 1] = 2
        tmp1[0, 2] = -1
        tmp1[-1, -3] = -1
        tmp1[-1, -2] = 2
        tmp1[-1, -1] = -1
        for i in range(1, n_points - 1):
            tmp1[i, i - 1] = 1 / 6
            tmp1[i, i] = 2 / 3
            tmp1[i, i + 1] = 1 / 6
        tmp2 = np.zeros(shape=(n_points, n_points))
        for i in range(1, n_points - 1):
            tmp2[i, i - 1] = 1
            tmp2[i, i] = -2
            tmp2[i, i + 1] =1
        return  np.linalg.solve(tmp1, tmp2)


def create_realizations(fname, T_segment,dT, fmin,  fmax, n_spline_points, n_realizations):
    # NOTE
    #    - the bilby tool (because it needs high computational efficiency, being done many times) is much harder to read. We will use library code, because we only call it ONCE PER RUN
    #    - similarly, the LI/bilby tool uses a slightly different representation, because they are trying to avoid transcendental operations to improve efficiency
    # Conversion tool
    # spline_matrix = nodes_to_spline_coefficients_matrix(n_spline_points)
    # STEP 0: logarithmic frequency spacing in positive freequency
#    print(fname, T_segment, dT, fmin, fmax, n_spline_points, n_realizations)
    
    log_freq_spline_locations = np.linspace(np.log10(fmin), np.log10(fmax), n_spline_points)
    
    # Localize data to location
    dat_amp, dat_phase = retrieve_envelope_from_file(fname, frequency_array=10**log_freq_spline_locations)
    # Create random spline realizations
    amp_rand_array = np.zeros((n_spline_points, n_realizations))
    phase_rand_array = np.zeros((n_spline_points, n_realizations))
#    print(amp_rand_array.shape, phase_rand_array.shape)
    # Create random amplitudes, phases
    #   - not efficient, for loop : use matrix operations to speed up!
    for indx in np.arange(n_spline_points):
        amp_rand_array[indx,:] = np.random.normal(loc=dat_amp[indx,1], scale=dat_amp[indx,2], size=n_realizations)
        phase_rand_array[indx,:] = np.random.normal(loc=dat_phase[indx,1], scale=dat_phase[indx,2], size=n_realizations)

    # Create realizations (complex-valued array for TWO_SIDED system
    deltaF_seg = 1./T_segment
    npts_seg    =  int(T_segment/dT)
    # Match array locations from lalsimutils.evaluate_fvals!
    freq_locations_physical =  deltaF_seg*np.array([ npts_seg/2 -k if  k<=npts_seg/2 else -k+npts_seg/2 for k in np.arange(npts_seg)])  # How lal packs its fft
    mask_positive = freq_locations_physical > 0
    mask_negative = freq_locations_physical < 0
    mask_in_range = np.logical_and(np.abs(freq_locations_physical) >= fmin ,  np.abs(freq_locations_physical) <= fmax)
    
    dat_out = np.ones((npts_seg, n_realizations),dtype=complex)  # default factor is unity

    # Loop over realizations, build up spline
    #   - should be able to vectorize this as well, using Soichiro trick noted above
    for indx in np.arange(n_realizations):
        mask_plus = mask_positive & mask_in_range
        mask_minus = mask_negative & mask_in_range
        cs_amp = scipy.interpolate.CubicSpline(log_freq_spline_locations, amp_rand_array[:,indx])
        cs_phase = scipy.interpolate.CubicSpline(log_freq_spline_locations, phase_rand_array[:,indx])
        log10_freq_pos_in_range = np.log10(freq_locations_physical[mask_plus])
        log10_minus_freq_neg_in_range = np.log10(-freq_locations_physical[mask_minus])
        # Apply interpolated coefficients. Note negative frequency handling
        dat_out[mask_plus, indx] = cs_amp(  log10_freq_pos_in_range )*np.exp(1j*cs_phase(log10_freq_pos_in_range))
        dat_out[mask_minus, indx] = cs_amp( log10_minus_freq_neg_in_range  )*np.exp(-1j*cs_phase(log10_minus_freq_neg_in_range))

#    print(log10_freq_pos_in_range, cs_amp(log10_freq_pos_in_range), cs_phase(log10_freq_pos_in_range) )
#    print(dat_out[mask_plus])
    return dat_out



if __name__ == "__main__":
    from matplotlib import pyplot as plt

    import sys
    import argparse

    n_spline_points = 10
    n_realizations = 100
    dT = 1./1024
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname",default=None,help="File name of interferometer cal file")
    parser.add_argument("--fmin",type=float,default=20)
    parser.add_argument("--fmax",type=float,default=500)
    parser.add_argument("--seglen",default=4, type=int)
    opts=  parser.parse_args()

    deltaF_seg = 1./opts.seglen
    npts_seg    =  int(opts.seglen/dT)
    # Match array locations from lalsimutils.evaluate_fvals!
    freq_locations_physical =  deltaF_seg*np.array([ npts_seg/2 -k if  k<=npts_seg/2 else -k+npts_seg/2 for k in np.arange(npts_seg)])  # How lal packs its fft

    
    if opts.fname is None:
        raise Exception("Fail: no input")

    dat_out = create_realizations(opts.fname, opts.seglen, dT, opts.fmin, opts.fmax, n_spline_points, n_realizations)

    # amplitude plot
    lower_vals = np.percentile(np.abs(dat_out), 16, 1)
    upper_vals = np.percentile(np.abs(dat_out), 84, 1)
    plt.plot(freq_locations_physical,lower_vals)
    plt.plot(freq_locations_physical,upper_vals)
    plt.savefig("fig_calmarg_amp.png")
    plt.clf()


    # phase plot
    lower_vals = np.percentile(np.angle(dat_out),16, 1)
    upper_vals = np.percentile(np.angle(dat_out), 84, 1)
    plt.plot(freq_locations_physical,lower_vals)
    plt.plot(freq_locations_physical,upper_vals)
    plt.savefig("fig_calmarg_phase.png")
