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


def node_prior(fname, fmin, fmax, n_spline_points):
    """Return the calibration PRIOR over spline nodes for one detector, as the
    diagonal Gaussian implied by the envelope file.

    The per-detector node vector is laid out as
        [amp_0 .. amp_{N-1}, phase_0 .. phase_{N-1}]   (N = n_spline_points)
    with amp node i ~ N(median_amp_i, sigma_amp_i) and phase node i ~
    N(median_phase_i, sigma_phase_i), independent (this is exactly the prior that
    create_realizations() draws from).

    Returns dict(mean, sigma, node_log_f, n_nodes_amp) -- mean/sigma length 2N.
    """
    log_freq_spline_locations = np.linspace(np.log10(fmin), np.log10(fmax), n_spline_points)
    dat_amp, dat_phase = retrieve_envelope_from_file(fname, frequency_array=10**log_freq_spline_locations)
    mean = np.concatenate([dat_amp[:, 1], dat_phase[:, 1]])
    sigma = np.concatenate([dat_amp[:, 2], dat_phase[:, 2]])
    return dict(mean=mean, sigma=sigma, node_log_f=log_freq_spline_locations,
                n_nodes_amp=int(n_spline_points))


def prior_cal_breadcrumb_dict(env_dir, dets, fmin, fmax, n_spline_points, fmin_ifo=None):
    """Build the 'cal' breadcrumb dict for the broad PRIOR, with proposal == prior.

    Suitable as an iteration-0 placeholder breadcrumb: seeding from it
    (seed_realizations_from_breadcrumb) draws cal realizations from the prior with ZERO
    importance weights (log prior - log proposal = 0), i.e. it is equivalent to the cold
    prior draws -- but, unlike a 0-byte placeholder, it LOADS cleanly (so an older ILE binary
    that does not guard against an empty placeholder will not crash on it).

    Layout matches seed_realizations_from_breadcrumb: the full node vector is concatenated per
    detector in `dets` order as [det0_amp_0..,det0_phase_0..,det1_amp..,...]; dim = 2N*len(dets).
    """
    import os
    means = []; sigmas = []; node_log_f = None
    for ifo in dets:
        fmin_here = fmin
        if fmin_ifo and ifo in fmin_ifo:
            fmin_here = fmin_ifo[ifo]
        pr = node_prior(os.path.join(env_dir, ifo + ".txt"), fmin_here, fmax, n_spline_points)
        means.append(pr["mean"]); sigmas.append(pr["sigma"])
        if node_log_f is None:
            node_log_f = pr["node_log_f"]
    prior_mean = np.concatenate(means); prior_sigma = np.concatenate(sigmas)
    return dict(proposal_mean=prior_mean, proposal_cov=np.diag(prior_sigma ** 2),
                prior_mean=prior_mean, prior_sigma=prior_sigma,
                node_log_f=node_log_f, n_nodes_amp=int(n_spline_points), dets=list(dets))


def _draw_amp_phase_nodes(dat_amp, dat_phase, n_spline_points, n_realizations):
    """Draw amp/phase spline nodes from the prior, in the EXACT random-number order
    create_realizations() has always used (so seeded behavior is byte-identical)."""
    amp_rand_array = np.zeros((n_spline_points, n_realizations))
    phase_rand_array = np.zeros((n_spline_points, n_realizations))
    for indx in np.arange(n_spline_points):
        amp_rand_array[indx, :] = np.random.normal(loc=dat_amp[indx, 1], scale=dat_amp[indx, 2], size=n_realizations)
        phase_rand_array[indx, :] = np.random.normal(loc=dat_phase[indx, 1], scale=dat_phase[indx, 2], size=n_realizations)
    return amp_rand_array, phase_rand_array


def build_realizations_from_nodes(amp_rand_array, phase_rand_array, T_segment, dT,
                                  fmin, fmax, log_freq_spline_locations):
    """Build the complex two-sided per-realization calibration factor array from
    spline-node values.  Factored out of create_realizations() so the SAME spline
    construction is reused both for prior draws and for proposal-seeded draws
    (seed_realizations_from_breadcrumb).

    amp_rand_array, phase_rand_array : (n_spline_points, n_realizations).
    Returns dat_out (npts_seg, n_realizations) complex, unity outside [fmin,fmax].
    """
    n_realizations = amp_rand_array.shape[1]
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

    return dat_out


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
    # Create random spline realizations (prior draws -- same RNG order as always)
    amp_rand_array, phase_rand_array = _draw_amp_phase_nodes(dat_amp, dat_phase, n_spline_points, n_realizations)

    # Create realizations (complex-valued array for TWO_SIDED system
    return build_realizations_from_nodes(amp_rand_array, phase_rand_array, T_segment, dT,
                                         fmin, fmax, log_freq_spline_locations)


def draw_prior_realizations_with_nodes(env_dir, dets, T_segment, dT, fmin, fmax,
                                       n_spline_points, n_realizations,
                                       fmin_ifo=None, rng=None):
    """Draw PRIOR calibration realizations AND keep the spline-node draws.

    Same prior as create_realizations(), but it returns the node vectors (which
    create_realizations discards) so a pilot can fit a proposal over them.  Used by
    the ILE --calibration-dump-responsibilities path.

    env_dir : directory of per-detector envelope files <ifo>.txt.
    dets    : detector order; the returned node vector is concatenated per det as
              [det0_amp, det0_phase, det1_amp, det1_phase, ...] (breadcrumb layout).

    Returns dict with:
       realizations : {ifo: (npts_seg, n_realizations) complex}
       nodes        : (n_realizations, 2*n_spline_points*len(dets))  prior draws
       prior_mean   : (dim,)   diagonal-Gaussian prior mean over the full node vector
       prior_sigma  : (dim,)   prior sigma
       node_log_f   : (n_spline_points,)  log10 spline node frequencies (det 0)
       n_nodes_amp  : n_spline_points
       dets         : list
    """
    import os
    if rng is None:
        rng = np.random.default_rng()
    priors = []
    for ifo in dets:
        fmin_here = fmin
        if fmin_ifo and ifo in fmin_ifo:
            fmin_here = fmin_ifo[ifo]
        priors.append(node_prior(os.path.join(env_dir, ifo + ".txt"), fmin_here, fmax, n_spline_points))
    prior_mean = np.concatenate([p["mean"] for p in priors])
    prior_sigma = np.concatenate([p["sigma"] for p in priors])
    dim = prior_mean.shape[0]
    # diagonal-Gaussian prior draws over the full node vector
    nodes = prior_mean[None, :] + prior_sigma[None, :] * rng.standard_normal((n_realizations, dim))

    n_amp = int(n_spline_points)
    dim_per_det = 2 * n_amp
    realizations = {}
    for i_det, ifo in enumerate(dets):
        fmin_here = fmin
        if fmin_ifo and ifo in fmin_ifo:
            fmin_here = fmin_ifo[ifo]
        log_freq_spline_locations = np.linspace(np.log10(fmin_here), np.log10(fmax), n_spline_points)
        block = nodes[:, i_det*dim_per_det:(i_det+1)*dim_per_det]
        realizations[ifo] = build_realizations_from_nodes(
            block[:, :n_amp].T, block[:, n_amp:].T, T_segment, dT, fmin_here, fmax,
            log_freq_spline_locations)
    return dict(realizations=realizations, nodes=nodes, prior_mean=prior_mean,
                prior_sigma=prior_sigma, node_log_f=priors[0]["node_log_f"],
                n_nodes_amp=n_amp, dets=list(dets))


def seed_realizations_from_breadcrumb(bc, T_segment, dT, fmin, fmax, n_spline_points,
                                      n_realizations, fmin_ifo=None, rng=None):
    """Draw n_realizations calibration factors per detector from a LEARNED Gaussian
    proposal (a breadcrumb), and return the Phase-0 importance weights.

    This is the production seed path (Option C): instead of drawing cal nodes from
    the broad prior, draw them from the consolidated pilot proposal (concentrated on
    the high-likelihood cal region), and carry log_w = log prior - log proposal so the
    marginalization stays unbiased.

    bc : breadcrumb dict (breadcrumbs.load(...)) OR its ["cal"] sub-dict.  The proposal
         is a joint Gaussian over the FULL multi-detector node vector, concatenated over
         bc["dets"] in order: [det0_amp, det0_phase, det1_amp, det1_phase, ...].
    fmin : scalar template fmin (used for all dets) OR ignored per-det if fmin_ifo given.
    fmin_ifo : optional {ifo: fmin} for per-detector low-frequency cutoffs.
    rng : numpy Generator (default: a fresh default_rng()).

    Returns (dat_out_dict, cal_log_weights, nodes):
       dat_out_dict   : {ifo: (npts_seg, n_realizations) complex}
       cal_log_weights: (n_realizations,) = log prior - log proposal (shared across dets,
                        since it is ONE joint draw per realization).
       nodes          : (n_realizations, dim) the proposal-drawn node vectors (so a pilot
                        can refit a proposal from a seeded run).
    """
    from RIFT.calmarg import adaptive
    cal = bc["cal"] if (isinstance(bc, dict) and "cal" in bc) else bc
    if rng is None:
        rng = np.random.default_rng()
    mean = np.asarray(cal["proposal_mean"], dtype=float)
    cov = np.asarray(cal["proposal_cov"], dtype=float)
    prior_mean = np.asarray(cal["prior_mean"], dtype=float)
    prior_sigma = np.asarray(cal["prior_sigma"], dtype=float)
    dets = list(cal["dets"])
    n_amp = int(cal["n_nodes_amp"])
    dim_per_det = 2 * n_amp
    assert mean.shape == (dim_per_det * len(dets),), \
        "breadcrumb proposal dim %s != 2*n_nodes_amp*len(dets)=%d" % (mean.shape, dim_per_det*len(dets))

    # Draw the full multi-detector node vector from the proposal; importance weights.
    nodes = rng.multivariate_normal(mean, cov, size=n_realizations)          # (n_real, dim)
    log_q = adaptive._mvn_logpdf(nodes, mean, cov)
    log_p = adaptive.log_prior(nodes, prior_mean, prior_sigma)
    cal_log_weights = log_p - log_q

    dat_out_dict = {}
    for i_det, ifo in enumerate(dets):
        fmin_here = fmin
        if fmin_ifo and ifo in fmin_ifo:
            fmin_here = fmin_ifo[ifo]
        log_freq_spline_locations = np.linspace(np.log10(fmin_here), np.log10(fmax), n_spline_points)
        block = nodes[:, i_det*dim_per_det:(i_det+1)*dim_per_det]             # (n_real, 2N)
        amp_rand_array = block[:, :n_amp].T                                   # (N, n_real)
        phase_rand_array = block[:, n_amp:].T
        dat_out_dict[ifo] = build_realizations_from_nodes(
            amp_rand_array, phase_rand_array, T_segment, dT, fmin_here, fmax,
            log_freq_spline_locations)
    return dat_out_dict, cal_log_weights, nodes



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
