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
from RIFT.misc.samples_utils import add_field, extract_combination_from_LI, standard_expand_samples, alt_expand_samples

from scipy.stats import gaussian_kde

parser = argparse.ArgumentParser()
parser.add_argument("--composite-file", action='append', help="Samples used in convergence test")
parser.add_argument("--composite-file-has-labels",action='store_true',help="Assume header for composite file")
parser.add_argument("--flag-tides-in-composite",action='store_true',help='Required, if you want to parse files with tidal parameters')
parser.add_argument("--flag-eos-index-in-composite",action='store_true',help='Required, if you want to parse files with EOS index in composite (and tides)')
parser.add_argument("--lnL-cut",default=None,type=float)
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



# Import
posterior_list = []
posteriorP_list = []
label_list = []
# Load posterior files
if opts.posterior_file:
  for fname in opts.posterior_file:
    samples = np.genfromtxt(fname,names=True,replace_space=None)  # don't replace underscores in names
    if 'm1' in samples.dtype.names:
        samples = standard_expand_samples(samples)
#    if not(opts.no_mod_psi) and 'psi' in samples.dtype.names:
#        samples['psi'] = np.mod(samples['psi'],np.pi)
    for name in samples.dtype.names:
        if name in lalsimutils.periodic_params:
            samples[name] = np.mod(samples[name], lalsimutils.periodic_params[name])



    # Save samples
    posterior_list.append(samples)

    # Continue ... rest not used at present
    continue

composite_list=[]
if opts.composite_file:
 print(opts.composite_file)
 for fname in opts.composite_file[:1]:  # Only load the first one!
    print(" Loading ... ", fname)
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

#    samples = np.recarray(samples.T,names=field_names,dtype=field_formats) #,formats=field_formats)
    # If no record names
    # Add mtotal, q,
    samples = alt_expand_samples(samples) # update

    samples_orig = samples

    # DOWNSELECTION AS NEEDED
    #    - user lnL cuts (hould not use)
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
        print(" lnL_max = ", lnL_max)
        indx_ok = samples["lnL"]>lnL_max  -opts.lnL_cut
        npts_out = np.sum(indx_ok)
        new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
        for name in samples.dtype.names:
            new_samples[name] = samples[name][indx_ok]
        samples = new_samples


    print(" Loaded samples from ", fname , len(np.atleast_1d(samples[name_ref])))

    composite_list.append(samples)

    continue


# Extract data for KDE
# Extract data for KDE
coord_names = opts.parameter # Used  in fit

lnL= None
if composite_list and posterior_list:
    # composite file i/o
    samples = composite_list[0]
    samples2 = posterior_list[0]
    samples_ref_name = samples.dtype.names[0]
    samples2_ref_name = samples2.dtype.names[0]
    # Create data for corner plot
    dat_mass = np.zeros( (len(np.atleast_1d(samples[samples_ref_name])), len(coord_names)) )
    dat_mass2 = np.zeros( (len(np.atleast_1d(samples2[samples2_ref_name])), len(coord_names)) )
    indx_sorted = np.arange(len(samples))
    if "lnL" in samples.dtype.names:
        lnL = samples["lnL"]
        indx_sorted = lnL.argsort()
    else:
        sys.exit("FAIL")


    for indx in np.arange(len(opts.parameter)):
        param = opts.parameter[indx]
        # composite file extraction
        if param in field_names:
            dat_mass[:,indx] = samples[param]
        else:
            print(" Trying alternative access for ", param)
            dat_mass[:,indx] = extract_combination_from_LI(samples, param)
        # posterior file extraction
        if param in samples2.dtype.names:
            dat_mass2[:,indx] = samples2[param]
        else:
            print(" Trying alternative access for ", param)
            dat_mass2[:,indx] = extract_combination_from_LI(samples2, param)

        

    # reverse order ... make sure largest plotted last
    dat_mass = dat_mass[indx_sorted]   # Sort by lnL
    lnL = lnL[indx_sorted]



#print(samples_list, composite_list)

# threshold lnL points
if (opts.lnL_cut):
    dat_mass = dat_mass[ lnL > np.max(lnL - opts.lnL_cut)   ]
    lnL = lnL[ lnL > np.max(lnL - opts.lnL_cut) ]
else:
    opts.lnL_cut = 0

# HARDCODED THRESHOLD BASED ON DIMENSION: use 95% probability
delta_lnL = 2*chi2.ppf(0.95,len(coord_names))
print(" Threshold offset due to dimension: ", delta_lnL, len(coord_names))
is_nearby =  lnL > np.max(lnL-  delta_lnL)

delta_lnL_close = 2*chi2.ppf(0.90,len(coord_names))
print(" Close Threshold offset due to dimension: ", delta_lnL_close, len(coord_names))
is_very_nearby =  lnL > np.max(lnL-  delta_lnL_close)

for pIndex in np.arange(len(posterior_list)):
    # Build KDE on posterior
    kde = gaussian_kde(dat_mass2.T)
    post_kde_eval = kde(dat_mass2.T)
    contour_threshold_90 = np.percentile(post_kde_eval, 10)

    # Evaluate KDE on likelihood points
    is_within_threshold = kde(dat_mass.T) > contour_threshold_90
    A=  np.sum(is_within_threshold)
    B = np.sum(np.logical_and(is_within_threshold, is_nearby))
    C = np.sum( np.logical_and(np.logical_not(is_within_threshold), is_nearby))
    D = np.sum( np.logical_and(np.logical_not(is_within_threshold), is_very_nearby))
    print(" Total , nearby, within, both ", len(dat_mass), np.sum(is_nearby), np.sum(is_within_threshold), np.sum(np.logical_and(is_within_threshold, is_nearby)) )
    print(" OUTSIDE ", C , C/B, D/B)
