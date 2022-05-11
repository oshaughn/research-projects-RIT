#! /usr/bin/env python
#
# GOAL
#   - read latest lnLmarg file from target working directory
#   - assess its distribution (modulo errors)
#       * Warning if lots of high-errorr points (esp at top
#       * Plot distribution of lnL, probbly warn if long tail but peak lnL is high
#   - probably provide option to run as a convergence test!
#   - WARNING; 'composite' includes the PUFFBALL points, and so it's not a fair test!


import RIFT.lalsimutils as lalsimutils
import RIFT.misc.bounded_kde as bounded_kde
from  RIFT.misc.samples_utils import  add_field

import scipy.stats
import glob

import lal
import numpy as np
import argparse


try:
    import matplotlib
#    print(" Matplotlib backend ", matplotlib.get_backend())
    if matplotlib.get_backend() == 'agg':
        fig_extension = '.png'
        bNoInteractivePlots=True
    else:
        matplotlib.use('agg')
        fig_extension = '.png'
        bNoInteractivePlots =True
    from matplotlib import pyplot as plt
    bNoPlots=False
except:
    print(" Error setting backend")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import corner

dpi_base=200
legend_font_base=16
rc_params = {'backend': 'ps',
             'axes.labelsize': 11,
             'axes.titlesize': 10,
             'font.size': 11,
             'legend.fontsize': legend_font_base,
             'xtick.labelsize': 11,
             'ytick.labelsize': 11,
             #'text.usetex': True,
             'font.family': 'Times New Roman'}#,
             #'font.sans-serif': ['Bitstream Vera Sans']}#,
plt.rcParams.update(rc_params)




parser = argparse.ArgumentParser()
parser.add_argument("--run-directory",help="run directory (preferred method).  Will auto-identify file")
parser.add_argument("--test",action='store_true')
parser.add_argument("--assume-dof",default=None,type=int,help="Number of DOF to assume for chisquared comparison")
parser.add_argument("--composite-file",help="Specific composite file to use (single), assumes fair draw")
parser.add_argument("--flag-tides-in-composite",action='store_true',help='Required, if you want to parse files with tidal parameters')
parser.add_argument("--eccentricity", action="store_true", help="Read sample files in format including eccentricity")
parser.add_argument("--sigma-cut",default=0.4,type=float)
parser.add_argument("--chi-max",default=1,type=float)
opts=  parser.parse_args()


# Identify files
if opts.run_directory:
    mynames = list(glob.glob(opts.run_directory+"/*.composite"))
    mynames.sort(key=lambda f: int(filter(str.isdigit, f)))
    print(" Composite files : ", mynames)
    opts.composite_file = mynames[-1]


# Import
composite_list = []
composite_full_list = []
field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lnL", "sigmaOverL", "ntot", "neff")
if opts.flag_tides_in_composite:
    print(" Reading composite file, assuming tide-based format ")
    field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","lambda1", "lambda2", "lnL", "sigmaOverL", "ntot", "neff")
if opts.eccentricity:
    print(" Reading composite file, assuming eccentricity-based format ")
    field_names=("indx","m1", "m2",  "a1x", "a1y", "a1z", "a2x", "a2y", "a2z","eccentricity", "lnL", "sigmaOverL", "ntot", "neff")
field_formats = [np.float32 for x in field_names]
composite_dtype = [ (x,float) for x in field_names] #np.dtype(names=field_names ,formats=field_formats)
# Load composite file
if opts.composite_file:
    fname = opts.composite_file
    print(" Loading ... ", fname)
    samples = np.loadtxt(fname,dtype=composite_dtype)  # Names are not always available
    # DO NOT CUT if we are testing!
    if not(opts.test) and opts.sigma_cut >0:
        npts = len(samples["m1"])
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
    samples=add_field(samples,[('mtotal',float)]); samples["mtotal"]= samples["m1"]+samples["m2"]; 
    samples=add_field(samples,[('q',float)]); samples["q"]= samples["m2"]/samples["m1"]; 
    samples=add_field(samples,[('mc',float)]); samples["mc"] = lalsimutils.mchirp(samples["m1"], samples["m2"])
    samples=add_field(samples,[('eta',float)]); samples["eta"] = lalsimutils.symRatio(samples["m1"], samples["m2"])
    samples=add_field(samples,[('chi_eff',float)]); samples["chi_eff"]= (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["mtotal"]); 
    chi1_perp = np.sqrt(samples['a1x']*samples["a1x"] + samples['a1y']**2)
    chi2_perp = np.sqrt(samples['a2x']**2 + samples['a2y']**2)
    samples = add_field(samples, [('chi1_perp',float)]); samples['chi1_perp'] = chi1_perp
    samples = add_field(samples, [('chi2_perp',float)]); samples['chi2_perp'] = chi2_perp

    if ('lambda1' in samples.dtype.names):
        Lt,dLt = lalsimutils.tidal_lambda_tilde(samples['m1'], samples['m2'],  samples['lambda1'], samples['lambda2'])
        samples= add_field(samples, [('LambdaTilde',float), ('DeltaLambdaTilde',float),('lambdat',float),('dlambdat',float)])
        samples['LambdaTilde'] = samples['lambdat']= Lt
        samples['DeltaLambdaTilde'] = samples['dlambdat']= dLt

    samples_orig = samples

    npts = len(samples["m1"])
        # strip NAN
    lnL_vals = samples["lnL"]
    not_nan = np.logical_not(np.isnan(lnL_vals))
    npts_out = np.sum(not_nan)
    if npts_out < npts:
        new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
        for name in samples.dtype.names:
            new_samples[name] = samples[name][not_nan]
        samples = new_samples
        

    print(" Loaded samples from ", fname , len(samples["m1"]))
    if True:
        # impose Kerr limit
        npts = len(samples["m1"])
        indx_ok =np.arange(npts)
        chi1_squared = samples["chi1_perp"]**2 + samples["a1z"]**2
        chi2_squared = samples["chi2_perp"]**2 + samples["a2z"]**2
        indx_ok = np.logical_and(chi1_squared < opts.chi_max ,chi2_squared < opts.chi_max)
        npts_out = np.sum(indx_ok)
        if npts_out < npts:
            print(" Ok systems ", npts_out)
            new_samples = np.recarray( (npts_out,), dtype=samples.dtype)
            for name in samples.dtype.names:
                new_samples[name] = samples[name][indx_ok]
            samples = new_samples
            print(" Stripped samples  from ", fname , len(samples["m1"]))


    composite_list.append(samples)
    composite_full_list.append(samples_orig)

#    continue


lnLmax = np.max(samples["lnL"])

###
### WARNINGS
###

if opts.test:
    # Test 0: Lots of high variance points within top 10%
    npts = len(samples['lnL'])
    npts_test = int(0.1*npts)+1
    vals = np.array(samples["lnL"])
    lnL_crit = vals[-npts_test]
    indx_top = samples["lnL"]>lnL_crit
    indx_top_bad = np.logical_and(samples["lnL"] > lnL_crit, samples["sigmaOverL"] > opts.sigma_cut) # cannot pass if we remove!
    if np.sum(indx_top_bad)/ np.sum(indx_top) >  0.01:
        print(" WARNING: High error points are more than 1% of the final sample; consider increasing n_eff or n_max for ILE ")


###
### PLOTS
###

# Generally we do NOT want to make these if we are in test mode, since we want to reject points

# Step 1: lnL cdf
#   - overlay expected DOF
vals = np.array(samples["lnL"])
n_dof =2
if np.any(np.abs(samples["a1z"])>0):
    n_dof +=2
if np.any(np.abs(samples["a1x"])>0):
    n_dof +=4
vals.sort()
plt.plot( vals, np.arange(len(vals))/(1.*len(vals)))
plt.plot( vals, 1-scipy.stats.chi2.cdf( (lnLmax-vals)*2,df=n_dof))
plt.xlabel(r'$\ln L$')
plt.ylabel(r'$P(<\ln L)$')
plt.savefig("cdf_lnL.png")
