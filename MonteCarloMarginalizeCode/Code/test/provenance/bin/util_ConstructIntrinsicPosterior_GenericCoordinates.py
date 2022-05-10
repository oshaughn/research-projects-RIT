#! /usr/bin/env python
#
# GOAL
#   - load in lnL data
#   - fit peak to quadratic (standard), GP, etc. 
#   - pass as input to mcsampler, to generate posterior samples
#
# FORMAT
#   - pankow simplification of standard format
#
# COMPARE TO
#   util_NRQuadraticFit.py
#   postprocess_1d_cumulative
#   util_QuadraticMassPosterior.py
#



import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import functools
import itertools


# GPU acceleration: NOT YET, just do usual
xpy_default=numpy  # just in case, to make replacement clear and to enable override
identity_convert = lambda x: x  # trivial return itself
cupy_success=False


no_plots = True
internal_dtype = np.float32  # only use 32 bit storage! Factor of 2 memory savings for GP code in high dimensions

C_CGS=2.997925*10**10 # Argh, Monica!
 


parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--input-tides",action='store_true',help="Use input format with tidal fields included.")
parser.add_argument("--input-distance",action='store_true',help="Use input format with distance fields (but not tidal fields?) enabled.")
parser.add_argument("--fname-lalinference",help="filename of posterior_samples.dat file [standard LI output], to overlay on corner plots")
parser.add_argument("--fname-output-samples",default="output-ILE-samples",help="output posterior samples (default output-ILE-samples -> output-ILE)")
parser.add_argument("--fname-output-integral",default="integral_result",help="output filename for integral result. Postfixes appended")
parser.add_argument("--approx-output",default="SEOBNRv2", help="approximant to use when writing output XML files.")
parser.add_argument("--amplitude-order",default=-1,type=int,help="Set ampO for grid. Used in PN")
parser.add_argument("--phase-order",default=7,type=int,help="Set phaseO for grid. Used in PN")
parser.add_argument("--fref",default=20,type=float, help="Reference frequency used for spins in the ILE output.  (Since I usually use SEOBNRv3, the best choice is 20Hz)")
parser.add_argument("--fmin",type=float,default=20)
parser.add_argument("--fname-rom-samples",default=None,help="*.rom_composite output. Treated identically to set of posterior samples produced by mcsampler after constructing fit.")
parser.add_argument("--n-output-samples",default=3000,type=int,help="output posterior samples (default 3000)")
parser.add_argument("--desc-lalinference",type=str,default='',help="String to adjoin to legends for LI")
parser.add_argument("--desc-ILE",type=str,default='',help="String to adjoin to legends for ILE")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
parser.add_argument("--no-adapt-parameter",action='append',help="Disable adaptive sampling in a parameter. Useful in cases where a parameter is not well-constrained, and the a prior sampler is well-chosen.")
parser.add_argument("--mc-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--eta-range",default=None,help="Eta range. Important if we have a BNS or other item that has a strong constraint.")
parser.add_argument("--mtot-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--trust-sample-parameter-box",action='store_true', help="If used, sets the prior range to the SAMPLE range for any parameters. NOT IMPLEMENTED. This should be automatically done for mc!")
parser.add_argument("--plots-do-not-force-large-range",action='store_true', help = "If used, the plots do NOT automatically set the chieff range to [-1,1], the eta range to [0,1/4], etc")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--no-downselect",action='store_true',help='Prevent using downselection on output points' )
parser.add_argument("--no-downselect-grid",action='store_true',help='Prevent using downselection on input points. Applied only to mc range' )
parser.add_argument("--downselect-enforce-kerr",action='store_true',help="Provides limits that enforce the kerr limit. Also imposed in coordinate transformations.")
parser.add_argument("--aligned-prior", default="uniform",help="Options are 'uniform', 'volumetric', and 'alignedspin-zprior'. Only influences s1z, s2z")
parser.add_argument("--transverse-prior", default="uniform",help="Options are 'volumetric' (default) and 'alignedspin-zprior'. Only influences s1x,s1y,s2x,s2y")
parser.add_argument("--spin-prior-chizplusminus-alternate-sampling",default='alignedspin_zprior',help="Use gaussian sampling when using chizplus, chizminus, to make reweighting more efficient.")
parser.add_argument("--import-prior-dictionary-file",default=None,type=str,help="File with dictionary stored_param_dict = 'name':func and stored_param_ranges = 'name':[left,right].  Use to overwrite priors with user-specified function")
parser.add_argument("--output-prior-dictionary-file",default=None,type=str,help="File with dictionary 'name':func. ")
parser.add_argument("--prior-gaussian-mass-ratio",action='store_true',help="Applies a gaussian mass ratio prior (mean=0.5, width=0.2 by default). Only viable in mtot, q coordinates. Not properly normalized, so will break bayes factors by about 2%%")
parser.add_argument("--prior-tapered-mass-ratio",action='store_true',help="Applies a tapered mass ratio prior (transition 0.8, kappa=20). Only viable in mtot, q coordinates. Not properly normalized, a tapering factor instread")
parser.add_argument("--prior-gaussian-spin1-magnitude",action='store_true',help="Applies a gaussian spin magnitude prior (mean=0.7, width=0.1 by default) for FIRST spin. Only viable in polar spin coordinates. Not properly normalized, so will break bayes factors by a small amount (depending on chi_max).  Used for 2g+1g merger arguments")
parser.add_argument("--prior-tapered-spin1-magnitude",action='store_true',help="Applies a tapered prior to spin1 magnitude")
parser.add_argument("--prior-tapered-spin1z",action='store_true',help="Applies a tapered prior to spin1's z component")
parser.add_argument("--pseudo-uniform-magnitude-prior", action='store_true',help="Applies volumetric prior internally, and then reweights at end step to get uniform spin magnitude prior")
parser.add_argument("--pseudo-uniform-magnitude-prior-alternate-sampling", action='store_true',help="Changes the internal sampling to be gaussian, not volumetric")
parser.add_argument("--pseudo-gaussian-mass-prior",action='store_true', help="Applies a gaussian mass prior in postprocessing. Done via reweighting so we can use arbitrary mass sampling coordinates.")
parser.add_argument("--pseudo-gaussian-mass-prior-mean",default=1.33,type=float, help="Mean value for reweighting")
parser.add_argument("--pseudo-gaussian-mass-prior-std",default=0.09, type=float,help="Width for reweighting")
parser.add_argument("--prior-lambda-linear",action='store_true',help="Use p(lambda) ~ lambdamax -lambda. Intended for first few iterations, to insure strong coverage of the low-lambda_k corner")
parser.add_argument("--prior-lambda-power",type=float,default=1,help="Use p(lambda) ~ (lambdamax -lambda)^p. Intended for first few iterations, to insure strong coverage of the low-lambda_k corner")
parser.add_argument("--mirror-points",action='store_true',help="Use if you have many points very near equal mass (BNS). Doubles the number of points in the fit, each of which has a swapped m1,m2")
parser.add_argument("--cap-points",default=-1,type=int,help="Maximum number of points in the sample, if positive. Useful to cap the number of points ued for GP. See also lnLoffset. Note points are selected AT RANDOM")
parser.add_argument("--chi-max", default=1,type=float,help="Maximum range of 'a' allowed.  Use when comparing to models that aren't calibrated to go to the Kerr limit.")
parser.add_argument("--chi-small-max", default=None,type=float,help="Maximum range of 'a' allowed on the smaller body.  If not specified, defaults to chi_max")
parser.add_argument("--chiz-plus-range", default=None,help="USE WITH CARE: If you are using chiz_minus, chiz_plus for a near-equal-mass system, then setting the chiz-plus-range can improve convergence (e.g., for aligned-spin systems), loosely by setting a chi_eff range that is allowed")
parser.add_argument("--lambda-max", default=4000,type=float,help="Maximum range of 'Lambda' allowed.  Minimum value is ZERO, not negative.")
parser.add_argument("--lambda-small-max", default=None,type=float,help="Maximum range of 'Lambda' allowed for smaller body. If provided and smaller than lambda_max, used ")
parser.add_argument("--lambda-plus-max", default=None,type=float,help="Maximum range of 'Lambda_plus' allowed.  Used for sampling. Pick small values to accelerate sampling! Otherwise, use lambda-max.")
parser.add_argument("--parameter-nofit", action='append', help="Parameter used to initialize the implied parameters, and varied at a low level, but NOT the fitting parameters")
parser.add_argument("--use-precessing",action='store_true')
parser.add_argument("--lnL-shift-prevent-overflow",default=None,type=float,help="Define this quantity to be a large positive number to avoid overflows. Note that we do *not* define this dynamically based on sample values, to insure reproducibility and comparable integral results. BEWARE: If you shift the result to be below zero, because the GP relaxes to 0, you will get crazy answers.")
parser.add_argument("--lnL-protect-overflow",action='store_true',help="Before fitting, subtract lnLmax - 100.  Add this quantity back at the end.")
parser.add_argument("--lnL-offset",type=float,default=np.inf,help="lnL offset. ONLY POINTS within lnLmax - lnLoffset are used in the calculation!  VERY IMPORTANT - default value chosen to include all points, not viable for production with some fit techniques like gp")
parser.add_argument("--lnL-offset-n-random",type=int,default=0,help="Add this many random points past the threshold")
parser.add_argument("--lnL-cut",type=float,default=None,help="lnL cut [MANUAL]. Remove points below this likelihood value from consideration.  Generally should not use")
parser.add_argument("--M-max-cut",type=float,default=1e5,help="Maximum mass to consider (e.g., if there is a cut on distance, this matters)")
parser.add_argument("--sigma-cut",type=float,default=0.6,help="Eliminate points with large error from the fit.")
parser.add_argument("--ignore-errors-in-data",action='store_true',help='Ignore reported error in lnL. Helpful for testing purposes (i.e., if the error is zero)')
parser.add_argument("--lnL-peak-insane-cut",type=float,default=np.inf,help="Throw away lnL greater than this value. Should not be necessary")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
parser.add_argument("--inj-file", help="Name of injection file")
parser.add_argument("--event-num", type=int, default=0,help="Zero index of event in inj_file")
parser.add_argument("--report-best-point",action='store_true')
parser.add_argument("--force-no-adapt",action='store_true',help="Disable adaptation, both of the tempering exponent *and* the individual sampling prior(s)")
parser.add_argument("--fit-uses-reported-error",action='store_true')
parser.add_argument("--fit-uses-reported-error-factor",type=float,default=1,help="Factor to add to standard deviation of fit, before adding to lnL. Multiplies number fitting dimensions")
parser.add_argument("--n-max",default=3e8,type=float)
parser.add_argument("--n-eff",default=3e3,type=int)
parser.add_argument("--internal-bound-factor-if-n-eff-small",default=None,type=float,help="If n_eff < n_ouptut_samples, we truncate the output size based on n_eff*(factor)")
parser.add_argument("--n-chunk",default=1e5,type=int)
parser.add_argument("--contingency-unevolved-neff",default=None,help="Contingency planning for when n_eff produced by CIP is small, and user doesn't want to have hard failures.  Note --fail-unless-n-eff will prevent this from happening. Options: quadpuff, ...")
parser.add_argument("--not-worker",action='store_true',help="Nonworker jobs, IF we have workers present, don't have the 'fail unless' statement active")
parser.add_argument("--fail-unless-n-eff",default=None,type=int,help="If nonzero, places a minimum requirement on n_eff. Code will exit if not achieved, with no sample generation")
parser.add_argument("--fit-method",default="quadratic",help="quadratic|polynomial|gp|gp_hyper|gp_lazy|cov|kde")
parser.add_argument("--fit-load-quadratic",default=None,help="Filename of hdf5 file to load quadratic fit from. ")
parser.add_argument("--fit-load-quadratic-path",default="GW190814/annealing_mc_source_eta_chieff",help="Path in hdf5 file to specific covariance matrix to be used")
parser.add_argument("--pool-size",default=3,type=int,help="Integer. Number of GPs to use (result is averaged)")
parser.add_argument("--fit-load-gp",default=None,type=str,help="Filename of GP fit to load. Overrides fitting process, but user MUST correctly specify coordinate system to interpret the fit with.  Does not override loading and converting the data.")
parser.add_argument("--fit-save-gp",default=None,type=str,help="Filename of GP fit to save. ")
parser.add_argument("--fit-order",type=int,default=2,help="Fit order (polynomial case: degree)")
parser.add_argument("--fit-uncertainty-added",default=False, action='store_true', help="Reported likelihood is lnL+(fit error). Use for placement and use of systematic errors.")
parser.add_argument("--no-plots",action='store_true')
parser.add_argument("--using-eos", type=str, default=None, help="Name of EOS.  Fit parameter list should physically use lambda1, lambda2 information (but need not) ")
parser.add_argument("--no-use-lal-eos",action='store_true',help="Do not use LAL EOS interface. Used for spectral EOS. Do not use this.")
parser.add_argument("--no-matter1", action='store_true', help="Set the lambda parameters to zero (BBH) but return them")
parser.add_argument("--no-matter2", action='store_true', help="Set the lambda parameters to zero (BBH) but return them")
parser.add_argument("--protect-coordinate-conversions", action='store_true', help="Adds an extra layer to coordinate conversions with range tests. Slows code down, but adds layer of safety for out-of-range EOS parameters for example")
parser.add_argument("--source-redshift",default=0,type=float,help="Source redshift (used to convert from source-frame mass [integration limits] to arguments of fitting function.  Note that if nonzero, integration done in SOURCE FRAME MASSES, but the fit is calculated using DETECTOR FRAME")
parser.add_argument("--eos-param", type=str, default=None, help="parameterization of equation of state")
parser.add_argument("--eos-param-values", default=None, help="Specific parameter list for EOS")
parser.add_argument("--sampler-method",default="adaptive_cartesian",help="adaptive_cartesian|GMM|adaptive_cartesian_gpu")
parser.add_argument("--internal-use-lnL",action='store_true',help="integrator internally manipulates lnL. ONLY VIABLE FOR GMM AT PRESENT")
parser.add_argument("--internal-correlate-parameters",default=None,type=str,help="comman-separated string indicating parameters that should be sampled allowing for correlations. Must be sampling parameters. Only implemented for gmm.  If string is 'all', correlate *all* parameters")
parser.add_argument("--internal-n-comp",default=1,type=int,help="number of components to use for GMM sampling. Default is 1, because we expect a unimodal posterior in well-adapted coordinates.  If you have crappy coordinates, use more")
parser.add_argument("--internal-gmm-memory-chisquared-factor",default=None,type=float,help="Multiple of the number of degrees of freedom to save. 5 is a part in 10^6, 4 is 10^{-4}, and None keeps all up to lnL_offset.  Note that low-weight points can contribute notably to n_eff, and it can be dangerous to assume a simple chisquared likelihood!  Provided in case we need very long runs")
parser.add_argument("--use-eccentricity", action="store_true")

ECC_MAX = 0.25 # maximum value of eccentricity, hard-coding here for ease of editing

# FIXME hacky options added by me (Liz) to try to get my capstone project to work.
# I needed a way to fix the component masses and nothing else seemed to work.
parser.add_argument("--fixed-parameter", action="append")
parser.add_argument("--fixed-parameter-value", action="append")

opts=  parser.parse_args()
if not(opts.no_adapt_parameter):
    opts.no_adapt_parameter =[] # needs to default to empty list
no_plots = no_plots |  opts.no_plots
lnL_shift = 0
lnL_default_large_negative = -500
if opts.lnL_shift_prevent_overflow:
    lnL_shift  = opts.lnL_shift_prevent_overflow
if not(opts.force_no_adapt):
    opts.force_no_adapt=False  # force explicit boolean false

print(" Arguments: ",' '.join(sys.argv[1:]))
print(" Input: ",opts.fname)
print(" Output: ",opts.fname_output_samples+".xml.gz")

import os
os.system("touch "+opts.fname_output_samples + ".xml.gz")

print(" ==  Input contents (at execute time) == ")
os.system("cat {} ".format(opts.fname))


os.system("wc -l {} > {} ".format(opts.fname,opts.fname_output_samples + ".xml.gz")) # put input contents into the output file, so I can see the inputs

