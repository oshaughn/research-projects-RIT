#! /usr/bin/env python
#
# GOAL
#   - load in lnL data
#   - fit peak to GAUSSIAN
#   - draw from this gaussian, using reweighting to make proposed samples
#   - can reject points that are close to existing samples, if desired ? 
#
#   A drop-in replacement for util_ConstructIntrinsicPosterior_GenericCoordinates
#
#
# ISSUES
#  - coordinates used for fit are ALSO used for draws.  


import RIFT.interpolators.BayesianLeastSquares as BayesianLeastSquares
import RIFT.interpolators.ConstrainedQuadraticLikelihood as ConstrainedQuadraticLikelihood

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import scipy.linalg as linalg
import scipy.stats
import scipy.special
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools
import itertools

import joblib  # http://scikit-learn.org/stable/modules/model_persistence.html

# GPU acceleration: NOT YET, just do usual
xpy_default=numpy  # just in case, to make replacement clear and to enable override
identity_convert = lambda x: x  # trivial return itself
cupy_success=False


no_plots = True
internal_dtype = np.float32  # only use 32 bit storage! Factor of 2 memory savings for GP code in high dimensions

C_CGS=2.997925*10**10 # Argh, Monica!
 
try:
    import matplotlib
    matplotlib.use('agg')  # prevent requests for DISPLAY
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.lines as mlines
    import corner

    no_plots=False
except ImportError:
    print(" - no matplotlib - ")


from sklearn.preprocessing import PolynomialFeatures
if True:
#try:
    import RIFT.misc.ModifiedScikitFit as msf  # altenative polynomialFeatures
else:
#except:
    print(" - Faiiled ModifiedScikitFit : No polynomial fits - ")
from sklearn import linear_model

from ligo.lw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)

import RIFT.integrators.mcsampler as mcsampler
try:
    import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
    mcsampler_gmm_ok = True
except:
    print(" No mcsamplerEnsemble ")
    mcsampler_gmm_ok = False
try:
    import RIFT.integrators.mcsamplerGPU as mcsamplerGPU
    mcsampler_gpu_ok = True
    mcsamplerGPU.xpy_default =xpy_default  # force consistent, in case GPU present
    mcsamplerGPU.identity_convert = identity_convert
except:
    print( " No mcsamplerGPU ")
    mcsampler_gpu_ok = False
try:
    import RIFT.interpolators.senni as senni
    senni_ok = True
except:
    print( " No senni ")
    senni_ok = False

try:
    import RIFT.interpolators.internal_GP
    internalGP_ok = True
except:
    print( " - no internal_GP -  ")
    internalGP_ok = False

try:
    import RIFT.interpolators.gpytorch_wrapper as gpytorch_wrapper
    gpytorch_ok = True
except:
    print( " No gpytorch_wrapper ")
    gpytorch_ok = False



def render_coord(x):
    if x in lalsimutils.tex_dictionary.keys():
        return lalsimutils.tex_dictionary[x]
    if 'product(' in x:
        a=x.replace(' ', '') # drop spaces
        a = a[:len(a)-1] # drop last
        a = a[8:]
        terms = a.split(',')
        exprs =list(map(render_coord, terms))
        exprs = list(map( lambda x: x.replace('$', ''), exprs))
        my_label = ' '.join(exprs)
        return '$'+my_label+'$'
    else:
        return x

def render_coordinates(coord_names):
    return list(map(render_coord, coord_names))




parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--internal-no-scale",action='store_true',help="If true, does NOT attempt to rescale coordinates to have mean zero, variance 1 in each dimension. Makes debugging output more human-accessible")
parser.add_argument("--internal-no-priors",action='store_true',help="Return only draws from the likelihood, without any prior reweighting.  Makes debugging easier.")
parser.add_argument("--input-tides",action='store_true',help="Use input format with tidal fields included.")
parser.add_argument("--input-distance",action='store_true',help="Use input format with distance fields (but not tidal fields?) enabled.")
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
parser.add_argument("--ecc-max", default=0.9,type=float,help="Maximum range of 'eccentricity' allowed.")
parser.add_argument("--ecc-min", default=0.0,type=float,help="Minimum range of 'eccentricity' allowed.")
parser.add_argument("--chiz-plus-range", default=None,help="USE WITH CARE: If you are using chiz_minus, chiz_plus for a near-equal-mass system, then setting the chiz-plus-range can improve convergence (e.g., for aligned-spin systems), loosely by setting a chi_eff range that is allowed")
parser.add_argument("--lambda-max", default=4000,type=float,help="Maximum range of 'Lambda' allowed.  Minimum value is ZERO, not negative.")
parser.add_argument("--lambda-small-max", default=None,type=float,help="Maximum range of 'Lambda' allowed for smaller body. If provided and smaller than lambda_max, used ")
parser.add_argument("--lambda-plus-max", default=None,type=float,help="Maximum range of 'Lambda_plus' allowed.  Used for sampling. Pick small values to accelerate sampling! Otherwise, use lambda-max.")
parser.add_argument("--parameter-nofit", action='append', help="Parameter used to initialize the implied parameters, and varied at a low level, but NOT the fitting parameters")
parser.add_argument("--use-precessing",action='store_true')
parser.add_argument("--lnL-downscale-factor",type=float,default=None,help="Multiply log likelihood by this number.  Intended for early stages of iterative analyses. Broadens the posterior. Assumes lnL is usual scale. Applied by MULTIPLYING INPUT DATA BY THIS FACTOR, before anything else applied.  Note also applied BEFORE MANUAL OFFSETS")
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
parser.add_argument("--internal-temper-log",action='store_true',help="integrator internally uses lnL as sampling weights (only).  Designed to reduce insane contrast and overfitting for high-amplitude cases")
parser.add_argument("--internal-correlate-parameters",default=None,type=str,help="comman-separated string indicating parameters that should be sampled allowing for correlations. Must be sampling parameters. Only implemented for gmm.  If string is 'all', correlate *all* parameters")
parser.add_argument("--internal-n-comp",default=1,type=int,help="number of components to use for GMM sampling. Default is 1, because we expect a unimodal posterior in well-adapted coordinates.  If you have crappy coordinates, use more")
parser.add_argument("--internal-gmm-memory-chisquared-factor",default=None,type=float,help="Multiple of the number of degrees of freedom to save. 5 is a part in 10^6, 4 is 10^{-4}, and None keeps all up to lnL_offset.  Note that low-weight points can contribute notably to n_eff, and it can be dangerous to assume a simple chisquared likelihood!  Provided in case we need very long runs")
parser.add_argument("--use-eccentricity", action="store_true")
parser.add_argument("--tripwire-fraction",default=0.05,type=float,help="Fraction of nmax of iterations after which n_eff needs to be greater than 1+epsilon for a small number epsilon")

# FIXME hacky options added by me (Liz) to try to get my capstone project to work.
# I needed a way to fix the component masses and nothing else seemed to work.
parser.add_argument("--fixed-parameter", action="append")
parser.add_argument("--fixed-parameter-value", action="append")

opts=  parser.parse_args()
if not(opts.no_adapt_parameter):
    opts.no_adapt_parameter =[] # needs to default to empty list
ECC_MAX = opts.ecc_max
ECC_MIN = opts.ecc_min
no_plots = no_plots |  opts.no_plots
lnL_shift = 0
lnL_default_large_negative = -500
if opts.lnL_shift_prevent_overflow:
    lnL_shift  = opts.lnL_shift_prevent_overflow
if not(opts.force_no_adapt):
    opts.force_no_adapt=False  # force explicit boolean false

source_redshift=opts.source_redshift


my_eos=None
#option to be used if gridded values not calculated assuming EOS
if opts.using_eos!=None:
    import RIFT.physics.EOSManager as EOSManager
    eos_name=opts.using_eos
    if opts.verbose:
        print(" Using EOS ", eos_name, opts.eos_param, opts.eos_param_values)

    if opts.eos_param == 'spectral':
        # Will not work yet -- need to modify to parse command-line arguments
        spec_param_packed=eval(opts.eos_param_values) # two lists: first are 'fixed' and second are specific
        fixed_param_array=spec_param_packed[0]
        spec_param_array=spec_param_packed[1]
        spec_params ={}
        spec_params['gamma1']=spec_param_array[0]
        spec_params['gamma2']=spec_param_array[1]
        # not used anymore: p0, epsilon0 set by the LI interface
        spec_params['p0']=fixed_param_array[0]   
        spec_params['epsilon0']=fixed_param_array[1]
        spec_params['xmax']=fixed_param_array[2]
        if len(spec_param_array) <3:
            spec_params['gamma3']=spec_params['gamma4']=0
        else:
            spec_params['gamma3']=spec_param_array[2]
            spec_params['gamma4']=spec_param_array[3]
        eos_base = EOSManager.EOSLindblomSpectral(name=eos_name,spec_params=spec_params,use_lal_spec_eos=not opts.no_use_lal_eos)
#        eos_vals = eos_base.make_spec_param_eos(npts=500)
#        lalsim_spec_param = eos_vals/(C_CGS**2)*7.42591549*10**(-25) # argh, Monica!
#        np.savetxt("lalsim_eos/"+eos_name+"_spec_param_geom.dat", np.c_[lalsim_spec_param[:,1], lalsim_spec_param[:,0]])
#        my_eos=lalsim.SimNeutronStarEOSFromFile(path+"/lalsim_eos/"+eos_name+"_spec_param_geom.dat")
        my_eos=eos_base
    elif 'lal_' in eos_name:
        eos_name = eos_name.replace('lal_','')
        my_eos = EOSManager.EOSLALSimulation(name=eos_name)
    else:
        my_eos = EOSManager.EOSFromDataFile(name=eos_name,fname =EOSManager.dirEOSTablesBase+"/" + eos_name+".dat")


with open('args.txt','w') as fp:
    import sys
    fp.write(' '.join(sys.argv))

if opts.fit_method == "quadratic":
    opts.fit_order = 2  # overrride

###
### Comparison data (from LI)
###
remap_ILE_2_LI = {
 "s1z":"a1z", "s2z":"a2z", 
 "s1x":"a1x", "s1y":"a1y",
 "s2x":"a2x", "s2y":"a2y",
 "chi1_perp":"chi1_perp",
 "chi2_perp":"chi2_perp",
 "chi1":'a1',
 "chi2":'a2',
 "cos_phiJL": 'cos_phiJL',
 "sin_phiJL": 'sin_phiJL',
 "cos_theta1":'costilt1',
 "cos_theta2":'costilt2',
 "theta1":"tilt1",
 "theta2":"tilt2",
  "xi":"chi_eff", 
  "chiMinus":"chi_minus", 
  "delta":"delta", 
  "delta_mc":"delta", 
 "mtot":'mtotal', "mc":"mc", "eta":"eta","m1":"m1","m2":"m2",
  "cos_beta":"cosbeta",
  "beta":"beta",
  "LambdaTilde":"lambdat",
  "DeltaLambdaTilde": "dlambdat"}


downselect_dict = {}
dlist = []
dlist_ranges=[]
if opts.downselect_parameter:
    dlist = opts.downselect_parameter
    dlist_ranges  = list(map(eval,opts.downselect_parameter_range))
else:
    dlist = []
    dlist_ranges = []
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]


chi_max = opts.chi_max
chi_small_max = chi_max
if not opts.chi_small_max is None:
    chi_small_max = opts.chi_small_max
lambda_max=opts.lambda_max
lambda_small_max  = lambda_max
if not  (opts.lambda_small_max is None):
    lambda_small_max = opts.lambda_small_max
lambda_plus_max = opts.lambda_max
if opts.lambda_plus_max:
    lambda_plus_max  = opts.lambda_max
downselect_dict['chi1'] = [0,chi_max]
downselect_dict['chi2'] = [0,chi_small_max]
if opts.input_tides and ('lambda1' in opts.parameter or 'LambdaTilde' in opts.parameter):
    downselect_dict['lambda1'] = [0,lambda_max]
    downselect_dict['lambda2'] = [0,lambda_small_max]
for param in ['s1z', 's2z', 's1x','s2x', 's1y', 's2y']:
    downselect_dict[param] = [-chi_max,chi_max]
# Enforce definition of eta
downselect_dict['eta'] = [0,0.25]

if opts.no_downselect:
    downselect_dict={}


test_converged={}
#test_converged['neff'] = functools.partial(mcsampler.convergence_test_MostSignificantPoint,0.01)  # most significant point less than 1/neff of probability.  Exactly equivalent to usual neff threshold.
#test_converged["normal_integral"] = functools.partial(mcsampler.convergence_test_NormalSubIntegrals, 25, 0.01, 0.1)   # 20 sub-integrals are gaussian distributed [weakly; mainly to rule out outliers] *and* relative error < 10%, based on sub-integrals . Should use # of intervals << neff target from above.  Note this sets our target error tolerance on  lnLmarg.  Note the specific test requires >= 20 sub-intervals, which demands *very many* samples (each subintegral needs to be converged).
###
### Parameters in use
###

coord_names = opts.parameter # Used  in fit
if coord_names is None:
    coord_names = []
if 'chi_pavg' in coord_names:
    low_level_coord_names += ['chi_pavg']
error_factor = len(coord_names)
if error_factor ==0 :
    raise Exception(" Coordinate list for fit empty; exiting ")
if opts.fit_uses_reported_error:
    error_factor=len(coord_names)*opts.fit_uses_reported_error_factor
# TeX dictionary
tex_dictionary = lalsimutils.tex_dictionary
print(" Coordinate names for fit :, ", coord_names)
print(" Rendering coordinate names : ",  render_coordinates(coord_names))  # map(lambda x: tex_dictionary[x], coord_names)
print(" Symmetry for these fitting coordinates :", lalsimutils.symmetry_sign_exchange(coord_names))


###
### Prior functions : a dictionary
###

# mcmin, mcmax : to be defined later
def M_prior(x):  # not normalized; see section II.C of https://arxiv.org/pdf/1701.01137.pdf
    return 2*x/(mc_max**2-mc_min**2)
def q_prior(x):
    return 1./(1+x)**2  # not normalized; see section II.C of https://arxiv.org/pdf/1701.01137.pdf
def m1_prior(x):
    return 1./200
def m2_prior(x):
    return 1./200
def s1z_prior(x):
    return 1./(2*chi_max)
def s2z_prior(x):
    return 1./(2*chi_max)
def mc_prior(x):
    return 2*x/(mc_max**2-mc_min**2)
def unscaled_eta_prior_cdf(eta_min):
    """
    cumulative for integration of x^(-6/5)(1-4x)^(-1/2) from eta_min to 1/4.
    Used to normalize the eta prior
    Derivation in mathematica:
       Integrate[ 1/\[Eta]^(6/5) 1/Sqrt[1 - 4 \[Eta]], {\[Eta], \[Eta]min, 1/4}]
    """
    return  2**(2./5.) *np.sqrt(np.pi)*scipy.special.gamma(-0.2)/scipy.special.gamma(0.3) + 5*scipy.special.hyp2f1(-0.2,0.5,0.8, 4*eta_min)/(eta_min**(0.2))
def eta_prior(x,norm_factor=1.44):
    """
    eta_prior returns the eta prior. 
    Change norm_factor by the output 
    """
    return 1./np.power(x,6./5.)/np.power(1-4.*x, 0.5)/norm_factor
def delta_mc_prior(x,norm_factor=1.44):
    """
    delta_mc = sqrt(1-4eta)  <-> eta = 1/4(1-delta^2)
    Transform the prior above
    """
    eta_here = 0.25*(1 -x*x)
    return 2./np.power(eta_here, 6./5.)/norm_factor

def m_prior(x):
    return 1/(1e3-1.)  # uniform in mass, use a square.  Should always be used as m1,m2 in pairs. Note this does NOT restrict m1>m2.


def triangle_prior(x,R=chi_max):
    return (np.ones(x.shape)-np.abs(x/R))/R  # triangle from -R to R centered on zero
def xi_uniform_prior(x):
    return np.ones(x.shape)
def s_component_uniform_prior(x,R=chi_max):  # If all three are used, a volumetric prior
    return np.ones(x.shape)/(2.*R)
def s_component_sqrt_prior(x,R=chi_max):  # If all three are used, a volumetric prior
    return 1./(4.*R*np.sqrt(np.abs(x)/R))  # -R,R range
def s_component_gaussian_prior(x,R=chi_max/3.):
    """
    (proportinal to) prior on range in one-dimensional components, in a cartesian domain.
    Could be useful to sample densely near zero spin.
    [Note: we should use 'truncnorm' instead...]
    """
    xp = np.array(x,dtype=float)
    val= scipy.stats.truncnorm(-chi_max/R,chi_max/R,scale=R).pdf(xp)  # stupid casting problem : x is dtype 'object'
    return val

def s_component_zprior(x,R=chi_max):
    # assume maximum spin =1. Should get from appropriate prior range
    # Integrate[-1/2 Log[Abs[x]], {x, -1, 1}] == 1
    val = -1./(2*R) * np.log( (np.abs(x)/R+1e-7).astype(float))
    return val


def s_component_volumetricprior(x,R=1.):
    # assume maximum spin =1. Should get from appropriate prior range
    # for SPIN MAGNITUDE OF PRECESSING SPINS only
    return (1./3.* np.power(x/R,2))

def s_component_aligned_volumetricprior(x,R=1.):
    # assume maximum spin =1. Should get from appropriate prior range
    # for SPIN COMPONENT ALIGNED (s1z,s2z) for aligned spins only
    #This is a probability that is defined on x\in[-R,R], such that \int_a^b dx p(x)  is the volume of a sphere between horizontal slices at height a,b:
    #p(x)dx =  pi R^2 (1- (x/R)^2)/ (4 pi R^3 /3) = 3/4 * (1 - (x/R)^2) /R
    return (3./4.*(1- np.power(x/R,2))/R)

def lambda_prior(x):
    return np.ones(x.shape)/lambda_max   # assume arbitrary
def lambda_small_prior(x):
    return np.ones(x.shape)/lambda_small_max   # assume arbitrary


# DO NOT USE UNLESS REQUIRED FOR COMPATIBILITY
def lambda_tilde_prior(x):
    return np.ones(x.shape)/opts.lambda_max   # 0,4000
def delta_lambda_tilde_prior(x):
    return np.ones(x.shape)/1000.   # -500,500

def gaussian_mass_prior(x,mu=0.,sigma=1.):   # actually viable for *any* prior.  
    y = np.array(x,dtype=np.float32)
    return np.exp( - 0.5*(y-mu)**2/sigma**2)/np.sqrt(2*np.pi*sigma**2)

def tapered_magnitude_prior(x,loc=0.65,kappa=19.):   # 
    """ 
    tapered_magnitude_prior is 1 inside a region and tapers to 0 outside
    The scale factor is designed so the taper is very strong and has no effect away from the region of significance
    Equivalent to
        (1 - 1/(1+f1)) / (1+f2) = f1/(1+f1)(1+f2)
    """
    y = np.array(x,dtype=np.float32) # problem of object type data
    f1 = np.exp( - (y-loc)*kappa)
    f2 = np.exp( - (y+loc)*kappa)
    
    return f1/(1+f1)/(1+f2)

def tapered_magnitude_prior_alt(x,loc=0.8,kappa=20.):   # 
    """ 
    tapered_magnitude_prior is 1 above the scale factor and 0 below it
        1/ (1+f) =
    """
    y = np.array(x,dtype=np.float32) # problem of object type data
    f1 = np.exp( - (y-loc)*kappa)
    
    return 1/(1+f1)

def eccentricity_prior(x):
    return np.ones(x.shape) / (ECC_MAX-ECC_MIN) # uniform over the interval [0.0, ECC_MAX]

def precession_prior(x):
    return 0.5*np.ones(x.shape) # uniform over the interval [0.0, 2.0]

def unnormalized_uniform_prior(x):
    return np.ones(x.shape)
def unnormalized_log_prior(x):
    return 1./x

prior_map  = { "mtot": M_prior, "q":q_prior, "s1z":s_component_uniform_prior, "s2z":functools.partial(s_component_uniform_prior, R=chi_small_max), "mc":mc_prior, "eta":eta_prior, 'delta_mc':delta_mc_prior, 'xi':xi_uniform_prior,'chi_eff':xi_uniform_prior,'delta': (lambda x: 1./2),
    's1x':s_component_uniform_prior,
    's2x':functools.partial(s_component_uniform_prior, R=chi_small_max),
    's1y':s_component_uniform_prior,
    's2y': functools.partial(s_component_uniform_prior, R=chi_small_max),
    'chiz_plus':s_component_uniform_prior,
    'chiz_minus':s_component_uniform_prior,
    'm1':m_prior,
    'm2':m_prior,
    'lambda1':lambda_prior,
    'lambda2':lambda_small_prior,
    'lambda_plus': lambda_prior,
    'lambda_minus': lambda_prior,
    'LambdaTilde':lambda_tilde_prior,
    'DeltaLambdaTilde':delta_lambda_tilde_prior,
    # Polar spin components (uniform magnitude by default)
    'chi1':s_component_uniform_prior,  
    'chi2':functools.partial(s_component_uniform_prior, R=chi_small_max),
    'theta1': mcsampler.uniform_samp_theta,
    'theta2': mcsampler.uniform_samp_theta,
    'cos_theta1': mcsampler.uniform_samp_cos_theta,
    'cos_theta2': mcsampler.uniform_samp_cos_theta,
    'phi1':mcsampler.uniform_samp_phase,
    'phi2':mcsampler.uniform_samp_phase,
    'eccentricity':eccentricity_prior,
    'chi_pavg':precession_prior,
    'mu1': unnormalized_log_prior,
    'mu2': unnormalized_uniform_prior
}
prior_range_map = {"mtot": [1, 300], "q":[0.01,1], "s1z":[-0.999*chi_max,0.999*chi_max], "s2z":[-0.999*chi_small_max,0.999*chi_small_max], "mc":[0.9,250], "eta":[0.01,0.2499999],'delta_mc':[0,0.9], 'xi':[-chi_max,chi_max],'chi_eff':[-chi_max,chi_max],'delta':[-1,1],
   's1x':[-chi_max,chi_max],
   's2x':[-chi_small_max,chi_small_max],
   's1y':[-chi_max,chi_max],
   's2y':[-chi_small_max,chi_small_max],
  'chiz_plus':[-chi_max,chi_max],   # BEWARE BOUNDARIES
  'chiz_minus':[-chi_max,chi_max],
  'm1':[0.9,1e3],
  'm2':[0.9,1e3],
  'lambda1':[0.01,lambda_max],
  'lambda2':[0.01,lambda_small_max],
  'lambda_plus':[0.01,lambda_plus_max],
  'lambda_minus':[-lambda_max,lambda_max],  # will include the true region always...lots of overcoverage for small lambda, but adaptation will save us.
  'eccentricity':[ECC_MIN, ECC_MAX],
  'chi_pavg':[0.0,2.0],  
  # strongly recommend you do NOT use these as parameters!  Only to insure backward compatibility with LI results
  'LambdaTilde':[0.01,5000],
  'DeltaLambdaTilde':[-500,500],
  'chi1':[0,chi_max],
  'chi2':[0,chi_small_max],
  'theta1':[0,np.pi],
  'theta2':[0,np.pi],
  'cos_theta1':[-1,1],
  'cos_theta2':[-1,1],
  'phi1':[0,2*np.pi],
  'phi2':[0,2*np.pi],
  'mu1':[0.0001,1e3],    # suboptimal, but something  
  'mu2':[-300,1e3]
}
if not (opts.chiz_plus_range is None):
    print(" Warning: Overriding default chiz_plus range. USE WITH CARE", opts.chiz_plus_range)
    prior_range_map['chiz_plus']=eval(opts.chiz_plus_range)

if not (opts.eta_range is None):
    print(" Warning: Overriding default eta range. USE WITH CARE")
    eta_range=prior_range_map['eta'] = eval(opts.eta_range)  # really only useful if eta is a coordinate.  USE WITH CARE
    prior_range_map['delta_mc'] = np.sqrt(1-4*np.array(prior_range_map['eta']))[::-1]  # reverse

    # change eta range normalization factors to match prior range on eta
    norm_factor = unscaled_eta_prior_cdf(eta_range[0]) - unscaled_eta_prior_cdf(eta_range[1])
    prior_map['eta'] = functools.partial(eta_prior, norm_factor=norm_factor)
    prior_map['delta_mc'] = functools.partial(delta_mc_prior, norm_factor=norm_factor)

###
### Modify priors, as needed
###
#  https://bugs.ligo.org/redmine/issues/5020
#  https://github.com/lscsoft/lalsuite/blob/master/lalinference/src/LALInferencePrior.c
if opts.aligned_prior == 'alignedspin-zprior':
    # prior on s1z constructed to produce the standard distribution
    prior_map["s1z"] = s_component_zprior
    prior_map["s2z"] = functools.partial(s_component_zprior,R=chi_small_max)

if opts.transverse_prior == 'alignedspin-zprior':
    prior_map["s1x"] = s_component_zprior
    prior_map["s1y"] = s_component_zprior
    prior_map["s2x"] = functools.partial(s_component_zprior,R=chi_small_max)
    prior_map["s2y"] = functools.partial(s_component_zprior,R=chi_small_max)
elif opts.transverse_prior == 'sqrt-prior':
    prior_map["s1x"] = s_component_sqrt_prior
    prior_map["s1y"] = s_component_sqrt_prior
    prior_map["s2x"] = functools.partial(s_component_sqrt_prior,R=chi_small_max)
    prior_map["s2y"] = functools.partial(s_component_sqrt_prior,R=chi_small_max)
elif opts.transverse_prior == 'taper-down':
    prior_map["s1x"] = triangle_prior
    prior_map["s1y"] = triangle_prior
    prior_map["s2x"] = functools.partial(triangle_prior,R=chi_small_max)
    prior_map["s2y"] = functools.partial(triangle_prior,R=chi_small_max)
    

if opts.aligned_prior == 'volumetric':
    prior_map["s1z"] = s_component_aligned_volumetricprior
    prior_map["s2z"] = s_component_aligned_volumetricprior

if opts.pseudo_uniform_magnitude_prior and opts.pseudo_uniform_magnitude_prior_alternate_sampling:
    prior_map['s1x'] = s_component_gaussian_prior
    prior_map['s2x'] = s_component_gaussian_prior
    prior_map['s1y'] = s_component_gaussian_prior
    prior_map['s2y'] = s_component_gaussian_prior


if opts.prior_lambda_linear:
    if opts.prior_lambda_power == 1:
        prior_map['lambda1'] = functools.partial(mcsampler.linear_down_samp,xmin=0,xmax=lambda_max)
        prior_map['lambda2'] = functools.partial(mcsampler.linear_down_samp,xmin=0,xmax=lambda_small_max)
    else:
        prior_map['lambda1'] = functools.partial(mcsampler.power_down_samp,xmin=0,xmax=lambda_max,alpha=opts.prior_lambda_power+1)
        prior_map['lambda2'] = functools.partial(mcsampler.power_down_samp,xmin=0,xmax=lambda_small_max,alpha=opts.prior_lambda_power+1)




###
### Linear fits. Resampling a quadratic. (Export me)
###

def fit_quadratic_stored(fname_h5,loc,L_offset=200):
    import h5py
    with h5py.File(fname_h5,'r') as F:
        event = F(loc)  # assumed to be mc_source ,eta, chi_eff for now!
        mean = np.array(event["mean"])
        cov = np.matrix(event["cov"])
    cov_det = np.linalg.det(cov)
    icov = np.linalg.pinv(cov)
    n_params = len(mean)
    return mean, cov


def fit_quadratic_alt(x,y,y_err=None,gamma_x=None,x0=None,symmetry_list=None,verbose=False,hard_regularize_negative=True):
#    gamma_x = None
    if not (y_err is None):
        gamma_x =np.diag(1./np.power(y_err,2))

    the_quadratic_results = BayesianLeastSquares.fit_quadratic( x, y,gamma_x=gamma_x,verbose=verbose,hard_regularize_negative=hard_regularize_negative)#x0=None)#x0_val_here)
    peak_val_est, best_val_est, my_fisher_est, linear_term_est,fn_estimate = the_quadratic_results

    # ESTIMATED cov, what should be true if condition number reasonable
    cov = linalg.pinv( my_fisher_est)
    print(np.linalg.eig(my_fisher_est)[0])
    if np.linalg.cond(my_fisher_est) > 1e3:
        print("  : WARNING: Ill-conditioned quadratic form ")
        print(np.linalg.eig(my_fisher_est))
        # Try to undo the impact of the most extreme scale, usually the chirp mass, before taking the inverse:
        #    Gamma_new == S Gamma S    # undo scale factor,
        #        Gamma = S^{-1} Gamma_{new}  S^{-1}
        #    Sigma = Gamma^{-1}  = S  Gamma_{new}^{-1} S
        diag_fish = np.diagonal(my_fisher_est)
        scale_fac = np.max(diag_fish)
        scale_indx = list(diag_fish).index(scale_fac)
        rescale = np.diag(np.ones(len(my_fisher_est)))
        rescale[scale_indx,scale_indx] = np.sqrt(scale_fac)
        i_rescale = np.diag( 1./np.diagonal(rescale))
        print(" Rescale ",i_rescale)
        my_fisher_est_alt = np.einsum('ij,jk,kl',i_rescale,my_fisher_est,i_rescale)
        print(my_fisher_est, my_fisher_est_alt)
        print("  Rescaled condition ", np.linalg.cond(my_fisher_est_alt))
        cov = np.linalg.pinv( my_fisher_est_alt)
        cov = np.einsum('ij,jk,kl',i_rescale,cov,i_rescale)  # re-apply the scale factor

    return best_val_est, cov


def fit_quadratic_nonneg(x,y,y_err=None,x0=None,symmetry_list=None,verbose=False):
    lnLmax, mu_est, cov_est = ConstrainedQuadraticLikelihood.fit_grid( x, y,verbose=verbose)#x0=None)#x0_val_here)

    return mu_est, cov_est



# initialize
dat_mass  = [] 
weights = []
n_params = -1

###
### Retrieve data
###
#  id m1 m2  lnL sigma/L  neff
col_lnL = 9
if opts.input_tides:
    print(" Tides input")
    col_lnL +=2
elif opts.use_eccentricity:
    print(" Eccentricity input: [",ECC_MIN, ", ",ECC_MAX, "]")
    col_lnL += 1
if opts.input_distance:
    print(" Distance input")
    col_lnL +=1
dat_orig = dat = np.loadtxt(opts.fname)
dat_orig = dat[dat[:,col_lnL].argsort()] # sort  http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
print(" Original data size = ", len(dat), dat.shape)

# Rescale lnL data, if requested.  Note requires user have sensible understanding of zero points of likelihood, etc  Appl
if opts.lnL_downscale_factor:
    dat[:,col_lnL] *= opts.lnL_downscale_factor
    dat_orig[:,col_lnL] *= opts.lnL_downscale_factor

 ###
 ### Convert data.  Use lalsimutils for flexibility
 ###
P_list = []
dat_out =[]
 
# simplify/recast None -> [] so I cna use 'in' below
if opts.parameter==None:
    opts.parameter = [] # force list, so avoid 'is iterable' below


symmetry_list =lalsimutils.symmetry_sign_exchange(coord_names)  # identify symmetry due to exchange
mc_min = 1e10
mc_max = -1

mc_index = -1 # index of mchirp in parameter index. To help with nonstandard GP
mc_cut_range = [-np.inf, np.inf] 
if opts.mc_range:
    mc_cut_range = eval(opts.mc_range)  # throw out samples outside this range.
    mc_min = mc_cut_range[0]
    mc_max = mc_cut_range[1]
    if opts.source_redshift>0:
        mc_cut_range =np.array(mc_cut_range)*(1+opts.source_redshift)  # prevent stupidity in grid selection
print(" Stripping samples outside of ", mc_cut_range, " in mc")
P= lalsimutils.ChooseWaveformParams()
P_list_in = []
for line in dat:
  # Skip precessing binaries unless explicitly requested not to!
  if not opts.use_precessing and (line[3]**2 + line[4]**2 + line[6]**2 + line[7]**2)>0.01:
      print(" Skipping precessing binaries ")
      continue
  if line[1]+line[2] > opts.M_max_cut:
      if opts.verbose:
          print(" Skipping ", line, " as too massive, with mass ", line[1]+line[2])
      continue
  if line[col_lnL+1] > opts.sigma_cut:
#      if opts.verbose:
#          print " Skipping ", line
      continue
  if not (opts.lnL_cut is None):
    if line[col_lnL] < opts.lnL_cut:
      continue  # strip worthless points.  DANGEROUS
  mc_here = lalsimutils.mchirp(line[1],line[2])
  if  (not opts.no_downselect_grid) and (mc_here < mc_cut_range[0] or mc_here > mc_cut_range[1]):
      if False and opts.verbose:
          print("Stripping because sample outside of target  mc range ", line)
      continue
  if line[col_lnL] < opts.lnL_peak_insane_cut:
    P.fref = opts.fref  # IMPORTANT if you are using a quantity that depends on J
    P.fmin = opts.fmin
    P.m1 = line[1]*lal.MSUN_SI
    P.m2 = line[2]*lal.MSUN_SI
    P.s1x = line[3]
    P.s1y = line[4]
    P.s1z = line[5]
    P.s2x = line[6]
    P.s2y = line[7]
    P.s2z = line[8]

    if opts.input_tides:
        P.lambda1 = line[9]
        P.lambda2 = line[10]
    if opts.use_eccentricity:
        P.eccentricity = line[9]
    if opts.input_distance:
        P.dist = lal.PC_SI*1e6*line[9]  # Incompatible with tides, note!
    
    if opts.contingency_unevolved_neff == "quadpuff":
        P_copy = P.manual_copy()  # prevent duplication
        P_list_in.append(P_copy) # store  it, make sure distinct

    # INPUT GRID: Evaluate binary parameters on fitting coordinates
    line_out = np.zeros(len(coord_names)+2)
    chi_pavg_out = 0 #initialize
    for x in np.arange(len(coord_names)):
        if coord_names[x] == 'chi_pavg':
            chi_pavg_out = P.extract_param('chi_pavg')
            line_out[x] = chi_pavg_out
        else:
            line_out[x] = P.extract_param(coord_names[x])
 #        line_out[x] = getattr(P, coord_names[x])
    line_out[-2] = line[col_lnL]
    line_out[-1] = line[col_lnL+1]  # adjoin error estimate
    dat_out.append(line_out)




    # Update mc range
    if not(opts.mc_range):
        mc_here = lalsimutils.mchirp(line[1],line[2])
        if mc_here < mc_min:
            mc_min = mc_here
        if mc_here > mc_max:
            mc_max = mc_here

    # Mirror!
    if opts.mirror_points:
        P.swap_components()
        # INPUT GRID: Evaluate binary parameters on fitting coordinates
        line_out = np.zeros(len(coord_names)+2)
        for x in np.arange(len(coord_names)):
            line_out[x] = P.extract_param(coord_names[x])
        line_out[-2] = line[col_lnL]
        line_out[-1] = line[col_lnL+1]  # adjoin error estimate
        dat_out.append(line_out)


Pref_default = P.copy()  # keep this around to fix the masses, if we don't have an inj

# Force 32 bit dype
dat_out = np.array(dat_out,dtype=internal_dtype)
print(" Stripped size  = ", dat_out.shape,  " with memory usage (bytes) ", sys.getsizeof(dat_out))
 # scale out mass units
for p in ['mc', 'm1', 'm2', 'mtot']:
    if p in coord_names:
        indx = coord_names.index(p)
        dat_out[:,indx] /= lal.MSUN_SI
            


# Repack data
X =dat_out[:,0:len(coord_names)]
Y = dat_out[:,-2]
Y_err = dat_out[:,-1]

# Save copies for later (plots)
X_orig = X.copy()
Y_orig = Y.copy()


# rescale X coordinatres
if not(opts.internal_no_scale):
    X_raw_mean = X[np.argmax(Y)]    # use peak value, instead of mean of input
#    X_raw_mean = np.mean(X,axis=0)
    X_raw_scale =   np.std(X, axis=0)
#    X_raw_scale = np.ones(len(X[0]))
    print(" Transforming variables ")
    print( " : Reference point {} ".format(X_raw_mean))
    print( " : Variable scales {} ".format(X_raw_scale))
    X = (X - X_raw_mean)/X_raw_scale  # hope broadcasting works here




# Eliminate values with Y too small
max_lnL = np.max(Y)
indx_ok = Y>np.max(Y)-opts.lnL_offset
# Provide some random points, to insure reasonable tapering behavior away from the sample
if opts.lnL_offset_n_random>0:
    p_select = np.min([opts.lnL_offset_n_random*1.0/len(Y),1])
    indx_ok = np.logical_or(indx_ok, np.random.choice([True,False],size=len(indx_ok),p=[p_select,1-p_select]))
print(" Points used in fit : ", sum(indx_ok), " given max lnL ", max_lnL)
if max_lnL < 10 and np.mean(Y) > -10: # second condition to allow synthetic tests not to fail, as these often have maxlnL not large
    print(" Resetting to use ALL input data -- beware ! ")
    # nothing matters, we will reject it anyways
    indx_ok = np.ones(len(Y),dtype=bool)
elif sum(indx_ok) < 5*len(X[0])**2: # and max_lnL > 30:
    n_crit_needed = np.min([5*len(X[0])**2,len(X)])
    print("  Likely below threshold needed to fit a maximum , beware: {} {} ".format(n_crit_needed,len(X[0])))
    # mark the top elements and use them for fits
    # this may be VERY VERY DANGEROUS if the peak is high and poorly sampled
    idx_sorted_index = np.lexsort((np.arange(len(Y)), Y))  # Sort the array of Y, recovering index values
    indx_list = np.array( [[k, Y[k]] for k in idx_sorted_index])     # pair up with the weights again
    indx_list = indx_list[::-1]  # reverse, so most significant are first
    indx_ok = list(map(int,indx_list[:n_crit_needed,0]))
    print(" Revised number of points for fit: ", sum(indx_ok), indx_ok, indx_list[:n_crit_needed])
    Y = Y[indx_ok]
    Y_err = Y_err[indx_ok]
    X = X[indx_ok]
    # Reset indx_ok, so it operates correctly later
    indx_ok = np.ones(len(Y), dtype=bool)
X_raw = X.copy()

my_fit= None
if not(opts.fit_load_quadratic is None):
    print("FIT METHOD IS STORED QUADRATIC; no data used! ")
    my_mean, my_cov = fit_quadratic_stored(opts.fit_load_quadratic, opts.fit_load_quadratic_path)
elif opts.fit_method == "quadratic":
    print(" FIT METHOD ", opts.fit_method, " IS QUADRATIC")
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]

    # construct prior map
    diag_prior = np.ones(len(coord_names))
    for indx in np.arange(len(coord_names)):
        rng = prior_range_map[coord_names[indx]]
        diag_prior[indx] = 1./(rng[1]-rng[0])**2  # prior range
    diag_prior= np.diag(diag_prior)

    my_mean, my_cov = fit_quadratic_alt(X,Y,y_err=Y_err,gamma_x=None,symmetry_list=symmetry_list,verbose=opts.verbose)
elif opts.fit_method == "quadratic_nonneg":
    print(" FIT METHOD ", opts.fit_method, " IS QUADRATIC nonnegative")
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
    my_mean, my_cov = fit_quadratic_nonneg(X,Y,y_err=Y_err,symmetry_list=symmetry_list,verbose=opts.verbose)
else:
    print(" NO KNOWN FIT METHOD ")
    sys.exit(55)

print(" Gaussian ", my_mean, my_cov)
if not(opts.internal_no_scale):
    my_mean_true = X_raw_mean+X_raw_scale*my_mean 
    my_cov_true = X_raw_scale*my_cov*X_raw_scale.T
    print(" Convert to physical coordinates ", my_mean_true, my_cov_true)

# Sort for later convenience (scatterplots, etc)
#indx = Y.argsort()#[::-1]
#X=X[indx]
#Y=Y[indx]


###
### Create mcsampler.  Just to make drawing coordinates easier
###

sampler = mcsampler.MCSampler()

##
## Loop over param names
##
for p in coord_names:
    if p in prior_map:
        prior_here = prior_map[p]
        range_here = prior_range_map[p]
    else:
        prior_here = lambda x: np.ones(x.shape)
        range_here = [-1,1]

    ## Special case: mc : provide ability to override range
    if p == 'mc' and opts.mc_range:
        range_here = eval(opts.mc_range)
    elif p=='mc' and opts.trust_sample_parameter_box:
        range_here = [np.max([range_here[0],mc_min]), np.min([range_here[1],mc_max])]
    if p =='mtot' and opts.mtot_range:
        range_here = eval(opts.mtot_range)
    # special cases mu1,mu2: rather than try to choose intelligently, just use training box data for limits
    if (p=='ln_mu1' or p == 'mu1' or p=='mu2') and opts.trust_sample_parameter_box:
        # extremely special/unusual use case, looking up from *input* data : mu1 only used as coordinate if also used for fitting
        indx_lnmu = coord_names.index(p)
        lnmu_max = np.max(dat_out[:,indx_lnmu])
        lnmu_min = np.min(dat_out[:,indx_lnmu])
        range_here = [lnmu_min,lnmu_max]

    adapt_me = False
    sampler.add_parameter(p, pdf=np.vectorize(lambda x:1), prior_pdf=prior_here,left_limit=range_here[0],right_limit=range_here[1],adaptive_sampling=adapt_me)



###
### Draw from the multivariate normal, then remove samples which violate limits
###

# Draw 1e6 samples
#my_mean = np.zeros(my_mean.shape)   # why is thsis wrong/offset?
dat_samples = np.random.multivariate_normal(my_mean, my_cov,size=int(3e7))

if not(opts.internal_no_scale):
    # Undeo transformation
    dat_samples = X_raw_mean + X_raw_scale*dat_samples


# Reject based on limits
for indx in np.arange(len(coord_names)):
    param = coord_names[indx]
    indx_ok = np.logical_and(dat_samples[:,indx] > sampler.llim[param] ,dat_samples[:,indx] < sampler.rlim[param] )
    dat_samples = dat_samples[indx_ok]

# Construct weights based on priors
prior_weights = np.ones(len(dat_samples))
print(" Retaining {} samples ".format(len(prior_weights)))
if not(opts.internal_no_priors):
    for indx in np.arange(len(coord_names)):
        param = coord_names[indx]
        prior_weights*= sampler.prior_pdf[param](dat_samples[:,indx])
prior_weights*= 1./np.sum(prior_weights)    

# Fairdraw remainder of samples
npts_out = np.min([int(1e-2 * len(dat_samples)), opts.n_output_samples*10000 ])
indx_choose = np.random.choice(len(prior_weights), size=npts_out,p=prior_weights)
dat_samples = dat_samples[indx_choose]
print(" Fairdraw sample size ", npts_out)
#print(dat_samples)


# First apply cuts on parameters in the param list (raw) 
# These cuts are vectorized and MUCH faster
pnames = list(coord_names)
print(downselect_dict)
for indx in np.arange(len(pnames)):
    param = pnames[indx]
    if param in downselect_dict:
        indx_ok = np.logical_and(dat_samples[:,indx] > downselect_dict[param][0], dat_samples[:,indx] < downselect_dict[param][1])
        dat_samples = dat_samples[indx_ok]
        del downselect_dict[param]
        print(" Downselecting : {} {}".format(param, len(dat_samples)))
print(" First cut reduction net : {} ".format(len(dat_samples)))


###
### Vector convert coordinates to the minimal needed
###

# Aligned case (default)
if not('s1x' in coord_names) and not('chi1' in coord_names):
    low_level_coord_names = ['m1','m2','s1z', 's2z']
    # yes, the naming here is kind of funky, we are trying to reconstruct 'standard' names from usually derived quantities
    X_new = lalsimutils.convert_waveform_coordinates(dat_samples, coord_names=low_level_coord_names,low_level_coord_names=coord_names)
#    print(X_new)

    # remove time-wasting tests we don't need
    del downselect_dict['s1x']
    del downselect_dict['s2x']
    del downselect_dict['s1y']
    del downselect_dict['s2y']
    del downselect_dict['chi1']
    del downselect_dict['chi2']

# Precessing case




###
### Export
###

print(" Preparing for export " )

# First apply cuts on parameters in the 'standard' coord list
# These cuts are vectorized and MUCH faster
pnames = list(low_level_coord_names)
print(downselect_dict)
for indx in np.arange(len(pnames)):
    param = pnames[indx]
    if param in downselect_dict:
        indx_ok = np.logical_and(X_new[:,indx] > downselect_dict[param][0], X_new[:,indx] < downselect_dict[param][1])
        X_new = X_new[indx_ok]
        del downselect_dict[param]
        print(" Downselecting : {} {}".format(param, len(X_new)))
print(" Second cut reduction net : {} ".format(len(X_new)))

# Now resize to final target size
if len(X_new) > opts.n_output_samples:
    print(" Final resize ")
    X_new  = X_new[:opts.n_output_samples]

# Now assemble the final list of exported quantities.  Any leftover cuts not already applied can be performed here.
P_out_list = []
# Loop over points 
for indx_P in np.arange(len(X_new)):   # make sure no past-limits errors
    P=lalsimutils.ChooseWaveformParams()
    for indx in np.arange(len(low_level_coord_names)):
        param  = low_level_coord_names[indx]
        fac = 1
        if low_level_coord_names[indx] in ['mc', 'mtot', 'm1', 'm2']:
            fac = lal.MSUN_SI
            if X_new[indx_P,indx]>1e10:  # correct for mu1,mu2 in coords
                fac=1
        P.assign_param(param, (X_new[indx_P,indx]*fac))
    
    include_item=True
    for p in downselect_dict.keys():
        val = P.extract_param(p) 
        if p in ['mc','m1','m2','mtot']:
            val = val/lal.MSUN_SI
        if val < downselect_dict[p][0] or val > downselect_dict[p][1]:
            include_item = False
    if include_item:
        P_out_list.append(P)

print(" Exporting {} samples ".format(len(P_out_list) ) )
# Save output
lalsimutils.ChooseWaveformParams_array_to_xml(P_out_list,fname=opts.fname_output_samples,fref=P.fref)





