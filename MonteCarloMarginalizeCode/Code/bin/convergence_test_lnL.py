#! /usr/bin/env python
#
# python convergence_test_lnL.py --composite-file all.net --flag-tides-in-composite --d-eff 4
# for i in rest*/run*/all.net; do echo $i; python ~/convergence_test_lnL.py --composite-file $i --d-eff 8 --eccentricity --meanPerAno; done


import numpy as np
import argparse
import scipy.stats
from scipy.stats import gaussian_kde, chi2
import numpy.linalg as la
import sys

import RIFT.lalsimutils as lalsimutils
import RIFT.misc.samples_utils as samples_utils
from RIFT.misc.samples_utils import add_field
from RIFT.misc.samples_utils import add_field, extract_combination_from_LI, standard_expand_samples

parser = argparse.ArgumentParser()
parser.add_argument("--composite-file", action='append', help="Samples used in convergence test")
parser.add_argument("--composite-file-has-labels",action='store_true',help="Assume header for composite file")
parser.add_argument("--flag-tides-in-composite",action='store_true',help='Required, if you want to parse files with tidal parameters')
parser.add_argument("--flag-eos-index-in-composite",action='store_true',help='Required, if you want to parse files with EOS index in composite (and tides)')
parser.add_argument("--lnL-cut",default=15,type=float)
parser.add_argument("--sigma-cut",default=0.4,type=float)
parser.add_argument("--d-eff",default=3,type=int,help="Effective dimension, used to assess if we have enough nearby samples close to the peak. ")
parser.add_argument("--n-close-min",default=1,type=int,help="Minimum number of samples close to the peak, *per effective dimension* (within 1)")
parser.add_argument("--n-cut-min",default=2000,type=int,help="Minimum number of samples passing the likelihood cut")
parser.add_argument("--eccentricity", action="store_true", help="Read sample files in format including eccentricity")
parser.add_argument("--meanPerAno", action="store_true", help="Read sample files in format including meanPerAno - assumes eccentricity also present")
parser.add_argument("--posterior-file",default=[], action='append', help="Samples used in convergence test")
parser.add_argument("--parameter", action='append', help="Parameters used in convergence test")
parser.add_argument("--threshold",default=0.01,type=float,  help="Manual threshold for the test being performed. (If not specified, the success condition is determined by default for that diagnostic, based on the samples size and properties).  Try 0.01")
parser.add_argument("--tests", action='append',default=[],  help="What tests to perform")
parser.add_argument("--test-output",  help="Filename to return output. Result is a scalar >=0 and ideally <=1.  Closer to 0 should be good. Second column is the diagnostic, first column is 0 or 1 (success or failure)")
parser.add_argument("--always-succeed",action='store_true',help="Test output is always success.  Use for plotting convergence diagnostics so jobs insured to run for many iterations.")
parser.add_argument("--iteration-threshold",default=0,type=int,help="Test is applied if iteration >= iteration-threshold. Default is 0")
parser.add_argument("--iteration",default=0,type=int,help="Current reported iteration. Default is 0.")
parser.add_argument("--write-file-on-success",type=str,default="INTRINSIC_CONVERGED",help="Produces an (empty) file with this name if the convergence tests passes.  Note you should pass the FULL PATH to this file if you want it to occur in the run directory for example")
opts=  parser.parse_args()

if len(opts.composite_file)<1:
    print(" Need at least one composite file ")
    sys.exit(1)

if opts.iteration < opts.iteration_threshold:
    sys.exit(0)

field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lnL", "sigmaOverL", "ntot", "neff")
if opts.flag_tides_in_composite:
    if opts.flag_eos_index_in_composite:
        print(" Reading composite file, assumingtide/eos-index-based format ")
        field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lambda1", "lambda2", "eos_indx","lnL", "sigmaOverL", "ntot", "neff")
    else:
        print(" Reading composite file, assuming tide-based format ")
        field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lambda1", "lambda2", "lnL", "sigmaOverL", "ntot", "neff")
if opts.eccentricity:
    print(" Reading composite file, assuming eccentricity-based format ")
    if opts.meanPerAno:
        print(" Reading composite file, assuming mpa-based format ")
        field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","eccentricity", "meanPerAno", "lnL", "sigmaOverL", "ntot", "neff")
    else:
        field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","eccentricity", "lnL", "sigmaOverL", "ntot", "neff")
field_formats = [np.float32 for x in field_names]
composite_dtype = [ (x,float) for x in field_names] #np.dtype(names=field_names ,formats=field_formats)

    
samples_list = []
composite_list = []
for fname in opts.posterior_file:
    samples = np.genfromtxt(fname, names=True)
    samples = stan
    samples = standard_expand_samples(samples)
    for name in samples.dtype.names:
        if name in lalsimutils.periodic_params:
            samples[name] = np.mod(samples[name], lalsimutils.periodic_params[name])
    samples_list.append(samples)
# Composite file parsing: from composite list
for fname in opts.composite_file:
    if not(opts.composite_file_has_labels):
        samples = np.loadtxt(fname,dtype=composite_dtype)  # Names are not always available
    else:
        samples = np.genfromtxt(fname,names=True)
        samples = rfn.rename_fields(samples, {'sigmalnL': 'sigmaOverL', 'sigma_lnL': 'sigmaOverL'})   # standardize names, some drift in labels
    # enforce periodicity
    for name in samples.dtype.names:
        if name in lalsimutils.periodic_params:
            samples[name] = np.mod(samples[name], lalsimutils.periodic_params[name])
    if 'lnL' in samples.dtype.names:
        samples = samples[ ~np.isnan(samples["lnL"])] # remove nan likelihoods -- they can creep in with poor settings/overflows
    name_ref = samples.dtype.names[0]

    if 'm1' in samples.dtype.names:
        samples=add_field(samples,[('mtotal',float)]); samples["mtotal"]= samples["m1"]+samples["m2"]; 
        samples=add_field(samples,[('q',float)]); samples["q"]= samples["m2"]/samples["m1"]; 
        samples=add_field(samples,[('mc',float)]); samples["mc"] = lalsimutils.mchirp(samples["m1"], samples["m2"])
        samples=add_field(samples,[('eta',float)]); samples["eta"] = lalsimutils.symRatio(samples["m1"], samples["m2"])
        samples=add_field(samples,[('chi_eff',float)]); samples["chi_eff"]= (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["mtotal"]); 
        chi1_perp = np.sqrt(samples['a1x']*samples["a1x"] + samples['a1y']**2)
        chi2_perp = np.sqrt(samples['a2x']**2 + samples['a2y']**2)
        samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
        samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp

        phi1 = np.arctan2(samples['a1x'], samples['a1y']);
        phi2 = np.arctan2(samples['a2x'], samples['a2y']);
        samples = add_field(samples, [('phi1',float), ('phi2',float), ('phi12',float)])
        samples['phi1'] = phi1
        samples['phi2'] = phi2
        samples['phi12'] = phi2 - phi1

        if ('lambda1' in samples.dtype.names):
            Lt,dLt = lalsimutils.tidal_lambda_tilde(samples['m1'], samples['m2'],  samples['lambda1'], samples['lambda2'])
            samples= add_field(samples, [('LambdaTilde',float), ('DeltaLambdaTilde',float),('lambdat',float),('dlambdat',float)])
            samples['LambdaTilde'] = samples['lambdat']= Lt
            samples['DeltaLambdaTilde'] = samples['dlambdat']= dLt

    
    if opts.sigma_cut >0:
        npts = len(np.atleast_1d(samples[name_ref]))
        # strip NAN
        sigma_vals = samples["sigmaOverL"]
        good_sigma = sigma_vals < opts.sigma_cut
        npts_out = np.sum(good_sigma)
        if npts_out < npts:
            new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
            for name in samples.dtype.names:
                new_samples[name] = samples[name][good_sigma]
            samples = new_samples
    if opts.lnL_cut and 'lnL' in samples.dtype.names:
        npts = len(np.atleast_1d(samples[name_ref]))
        # strip NAN
        lnL_vals = samples["lnL"]
        not_nan = np.logical_not(np.isnan(lnL_vals))
        npts_out = np.sum(not_nan)
        if npts_out < npts:
            new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
            for name in samples.dtype.names:
                new_samples[name] = samples[name][not_nan]
            samples = new_samples
        
        # apply cutoff
        indx_ok =np.arange(npts)
        lnL_max = np.max(samples["lnL"])
        indx_ok = samples["lnL"]>lnL_max  -opts.lnL_cut
        npts_out = np.sum(indx_ok)
        new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
        for name in samples.dtype.names:
            new_samples[name] = samples[name][indx_ok]
        samples = new_samples
    print(" Reading ", len(samples), " from {} : lnLmax = {}".format(fname, lnL_max))
    composite_list.append(samples)


#print(samples_list, composite_list)
composite_preferred = composite_list[0]
thresh_max = -opts.d_eff
lnLmax =  np.max(composite_preferred['lnL'])  # assume anything passing is good
n_close = np.sum(composite_preferred['lnL'] > lnLmax + thresh_max)  # typical dimension
thresh_levels = np.array([-1,-2,-3,-4,-5, -6, thresh_max])
expected_cdf = scipy.stats.chi2.cdf( -  2*thresh_levels,df=opts.d_eff)
expected_cdf *= 1./expected_cdf[-1]
n_thresh = np.array([np.sum( composite_preferred['lnL'] > lnLmax  + offset) for offset in thresh_levels])
print(" Number of nearby samples " , n_thresh)
print(" Fraction of nearby samples ", n_thresh/n_close, expected_cdf)
print(" Ratio ", n_thresh/n_close / expected_cdf)

if len(composite_preferred) < opts.n_cut_min:
    print("  Failure: too few points contribute to estimating posterior , absolute. ")
    sys.exit(1)
    
if n_thresh[0] < opts.n_close_min*opts.d_eff:
    print("  Failure: too few points near peak , absolute. FIXME MODEL ")
    sys.exit(1)
if (n_thresh/n_close/expected_cdf)[0] < 1e-2:
    print("  Failure: too few points near peak , vs expected ")
    sys.exit(1)
sys.exit(0)    
