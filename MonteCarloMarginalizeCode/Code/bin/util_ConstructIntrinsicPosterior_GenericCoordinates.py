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


import RIFT.interpolators.BayesianLeastSquares as BayesianLeastSquares

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import scipy.stats
import scipy.special
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools
import itertools

import joblib  # http://scikit-learn.org/stable/modules/model_persistence.html

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

from glue.ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)

import RIFT.integrators.mcsampler as mcsampler
try:
    import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
    mcsampler_gmm_ok = True
except:
    print(" No mcsamplerEnsemble ")
    mcsampler_gmm_ok = False
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


def extract_combination_from_LI(samples_LI, p):
    """
    extract_combination_from_LI
      - reads in known columns from posterior samples
      - for selected known combinations not always available, it will compute them from standard quantities
    """
    if p in samples_LI.dtype.names:  # e.g., we have precomputed it
        return samples_LI[p]
    if p in remap_ILE_2_LI.keys():
       if remap_ILE_2_LI[p] in samples_LI.dtype.names:
         return samples_LI[ remap_ILE_2_LI[p] ]
    # Return cartesian components of spin1, spin2.  NOTE: I may already populate these quantities in 'Add important quantities'
    if p == 'chiz_plus':
        print(" Transforming ")
        if 'a1z' in samples_LI.dtype.names:
            return (samples_LI['a1z']+ samples_LI['a2z'])/2.
        if 'theta1' in samples_LI.dtype.names:
            return (samples_LI['a1']*np.cos(samples_LI['theta1']) + samples_LI['a2']*np.cos(samples_LI['theta2']) )/2.
#        return (samples_LI['a1']+ samples_LI['a2'])/2.
    if p == 'chiz_minus':
        print(" Transforming ")
        if 'a1z' in samples_LI.dtype.names:
            return (samples_LI['a1z']- samples_LI['a2z'])/2.
        if 'theta1' in samples_LI.dtype.names:
            return (samples_LI['a1']*np.cos(samples_LI['theta1']) - samples_LI['a2']*np.cos(samples_LI['theta2']) )/2.
#        return (samples_LI['a1']- samples_LI['a2'])/2.
    if  'theta1' in samples_LI.dtype.names:
        if p == 's1x':
            return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) * np.cos( samples_LI['phi1'])
        if p == 's1y' :
            return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) * np.sin( samples_LI['phi1'])
        if p == 's2x':
            return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) * np.cos( samples_LI['phi2'])
        if p == 's2y':
            return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) * np.sin( samples_LI['phi2'])
        if p == 'chi1_perp' :
            return samples_LI["a1"]*np.sin(samples_LI[ 'theta1']) 
        if p == 'chi2_perp':
            return samples_LI["a2"]*np.sin(samples_LI[ 'theta2']) 
    if 'lambdat' in samples_LI.dtype.names:  # LI does sampling in these tidal coordinates
        lambda1, lambda2 = lalsimutils.tidal_lambda_from_tilde(samples_LI["m1"], samples_LI["m2"], samples_LI["lambdat"], samples_LI["dlambdat"])
        if p == "lambda1":
            return lambda1
        if p == "lambda2":
            return lambda2
    if p == 'delta' or p=='delta_mc':
        return (samples_LI['m1']  - samples_LI['m2'])/((samples_LI['m1']  + samples_LI['m2']))
    # Return cartesian components of Lhat
    if p == 'product(sin_beta,sin_phiJL)':
        return np.sin(samples_LI[ remap_ILE_2_LI['beta'] ]) * np.sin(  samples_LI['phi_jl'])
    if p == 'product(sin_beta,cos_phiJL)':
        return np.sin(samples_LI[ remap_ILE_2_LI['beta'] ]) * np.cos(  samples_LI['phi_jl'])

    print(" No access for parameter ", p)
    return np.zeros(len(samples_LI['m1']))  # to avoid causing a hard failure

def add_field(a, descr):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = numpy.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    return b


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
#parser.add_argument("--no-adapt-parameter",action='append',help="Disable adaptive sampling in a parameter. Useful in cases where a parameter is not well-constrained, and the a prior sampler is well-chosen.")
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
parser.add_argument("--adapt",action='store_true')
parser.add_argument("--fit-uses-reported-error",action='store_true')
parser.add_argument("--fit-uses-reported-error-factor",type=float,default=1,help="Factor to add to standard deviation of fit, before adding to lnL. Multiplies number fitting dimensions")
parser.add_argument("--n-max",default=3e8,type=float)
parser.add_argument("--n-eff",default=3e3,type=int)
parser.add_argument("--contingency-unevolved-neff",default=None,help="Contingency planning for when n_eff produced by CIP is small, and user doesn't want to have hard failures.  Note --fail-unless-n-eff will prevent this from happening. Options: quadpuff, ...")
parser.add_argument("--fail-unless-n-eff",default=None,type=int,help="If nonzero, places a minimum requirement on n_eff. Code will exit if not achieved, with no sample generation")
parser.add_argument("--fit-method",default="quadratic",help="quadratic|polynomial|gp|gp_hyper")
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
parser.add_argument("--internal-correlate-parameters",default=None,type=str,help="comman-separated string indicating parameters that should be sampled allowing for correlations. Must be sampling parameters. Only implemented for gmm.  If string is 'all', correlate *all* parameters")
opts=  parser.parse_args()
no_plots = no_plots |  opts.no_plots
lnL_shift = 0
lnL_default_large_negative = -500
if opts.lnL_shift_prevent_overflow:
    lnL_shift  = opts.lnL_shift_prevent_overflow

source_redshift=opts.source_redshift


my_eos=None
#option to be used if gridded values not calculated assuming EOS
if opts.using_eos!=None:
    import EOSManager
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

if opts.fname_lalinference:
    print(" Loading lalinference samples for direct comparison ", opts.fname_lalinference)
    samples_LI = np.genfromtxt(opts.fname_lalinference,names=True)

    print(" Checking consistency between fref in samples and fref assumed here ")
    try:
        print(set(samples_LI['f_ref']), opts.fref)
    except:
        print(" No fref")

    print(" Checking LI samples have desired parameters ")
    try:
      for p in opts.parameter:
        if p in remap_ILE_2_LI:
            print(p , " -> ", remap_ILE_2_LI[p])
        else:
            print(p, " NOT LISTED IN KEYS")
    except:
        print("remap check failed")

    ###
    ### Add important quantities easily derived from the samples but not usually provided
    ###

    delta_dat = (samples_LI["m1"] - samples_LI["m2"])/(samples_LI["m1"]+samples_LI["m2"])
    samples_LI = add_field(samples_LI, [('delta', float)]); samples_LI['delta'] = delta_dat
    if "a1z" in samples_LI.dtype.names:
        chi_minus = (samples_LI["a1z"]*samples_LI["m1"] - samples_LI["a2z"]*samples_LI["m2"])/(samples_LI["m1"]+samples_LI["m2"])
        samples_LI = add_field(samples_LI, [('chi_minus', float)]); samples_LI['chi_minus'] = chi_minus

    if "tilt1" in samples_LI.dtype.names:
        a1x_dat = samples_LI["a1"]*np.sin(samples_LI["theta1"])*np.cos(samples_LI["phi1"])
        a1y_dat = samples_LI["a1"]*np.sin(samples_LI["theta1"])*np.sin(samples_LI["phi1"])
        chi1_perp = samples_LI["a1"]*np.sin(samples_LI["theta1"])

        a2x_dat = samples_LI["a2"]*np.sin(samples_LI["theta2"])*np.cos(samples_LI["phi2"])
        a2y_dat = samples_LI["a2"]*np.sin(samples_LI["theta2"])*np.sin(samples_LI["phi2"])
        chi2_perp = samples_LI["a2"]*np.sin(samples_LI["theta2"])

                                      
        samples_LI = add_field(samples_LI, [('a1x', float)]);  samples_LI['a1x'] = a1x_dat
        samples_LI = add_field(samples_LI, [('a1y', float)]); samples_LI['a1y'] = a1y_dat
        samples_LI = add_field(samples_LI, [('a2x', float)]);  samples_LI['a2x'] = a2x_dat
        samples_LI = add_field(samples_LI, [('a2y', float)]);  samples_LI['a2y'] = a2y_dat
        samples_LI = add_field(samples_LI, [('chi1_perp', float)]); samples_LI['chi1_perp'] = chi1_perp
        samples_LI = add_field(samples_LI, [('chi2_perp', float)]); samples_LI['chi2_perp'] = chi2_perp

        samples_LI = add_field(samples_LI, [('cos_phiJL',float)]); samples_LI['cos_phiJL'] = np.cos(samples_LI['phi_jl'])
        samples_LI = add_field(samples_LI, [('sin_phiJL',float)]); samples_LI['sin_phiJL'] = np.sin(samples_LI['phi_jl'])
        
        samples_LI = add_field(samples_LI, [('product(sin_beta,sin_phiJL)', float)])
        samples_LI['product(sin_beta,sin_phiJL)'] =np.sin(samples_LI[ remap_ILE_2_LI['beta'] ]) * np.sin(  samples_LI['phi_jl'])


        samples_LI = add_field(samples_LI, [('product(sin_beta,cos_phiJL)', float)])
        samples_LI['product(sin_beta,cos_phiJL)'] =np.sin(samples_LI[ remap_ILE_2_LI['beta'] ]) * np.cos(  samples_LI['phi_jl'])

    # Add all keys in samples_LI to the remapper
    for key in samples_LI.dtype.names:
        if not (key in remap_ILE_2_LI.keys()):
            remap_ILE_2_LI[key] = key  # trivial remapping

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
low_level_coord_names = coord_names # Used for Monte Carlo
if opts.parameter_implied:
    coord_names = coord_names+opts.parameter_implied
if opts.parameter_nofit:
    if opts.parameter is None:
        low_level_coord_names = opts.parameter_nofit # Used for Monte Carlo
    else:
        low_level_coord_names = opts.parameter+opts.parameter_nofit # Used for Monte Carlo
error_factor = len(coord_names)
if opts.fit_uses_reported_error:
    error_factor=len(coord_names)*opts.fit_uses_reported_error_factor
# TeX dictionary
tex_dictionary = lalsimutils.tex_dictionary
print(" Coordinate names for fit :, ", coord_names)
print(" Rendering coordinate names : ",  render_coordinates(coord_names))  # map(lambda x: tex_dictionary[x], coord_names)
print(" Symmetry for these fitting coordinates :", lalsimutils.symmetry_sign_exchange(coord_names))
print(" Coordinate names for Monte Carlo :, ", low_level_coord_names)
print(" Rendering coordinate names : ", list(map(lambda x: tex_dictionary[x], low_level_coord_names)))


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

def xi_uniform_prior(x):
    return np.ones(x.shape)
def s_component_uniform_prior(x,R=chi_max):  # If all three are used, a volumetric prior
    return np.ones(x.shape)/(2.*R)
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
    'phi2':mcsampler.uniform_samp_phase
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
    if  'chiz_plus' in low_level_coord_names:
        if opts.spin_prior_chizplusminus_alternate_sampling is 'alignedspin_zprior':
            # just a  trick to make reweighting more efficient.
            prior_map['chiz_plus'] = s_component_zprior
            prior_map['chiz_minus'] = s_component_zprior
        else:
            prior_map['chiz_plus'] = s_component_gaussian_prior
            prior_map['chiz_minus'] = s_component_gaussian_prior

if opts.transverse_prior == 'alignedspin-zprior':
    prior_map["s1x"] = s_component_zprior
    prior_map["s1y"] = s_component_zprior
    prior_map["s2x"] = functools.partial(s_component_zprior,R=chi_small_max)
    prior_map["s2y"] = functools.partial(s_component_zprior,R=chi_small_max)

if opts.aligned_prior == 'volumetric':
    prior_map["s1z"] = s_component_aligned_volumetricprior
    prior_map["s2z"] = s_component_aligned_volumetricprior

if opts.pseudo_uniform_magnitude_prior and opts.pseudo_uniform_magnitude_prior_alternate_sampling:
    prior_map['s1x'] = s_component_gaussian_prior
    prior_map['s2x'] = s_component_gaussian_prior
    prior_map['s1y'] = s_component_gaussian_prior
    prior_map['s2y'] = s_component_gaussian_prior
    if 's1z' in low_level_coord_names:
        prior_map['s1z'] = s_component_gaussian_prior
        prior_map['s2z'] = s_component_gaussian_prior
    elif 'chiz_plus' in low_level_coord_names:  # because of rotated coordinate system. This matches in interior
        print(" CODE PATH NOT YET WORKING ")
        sys.exit(0)
        prior_map['chiz_plus'] = s_component_gaussian_prior #lambda x: s_component_gaussian_prior(x, R=chi_max/3.)
        prior_map['chiz_minus'] = s_component_gaussian_prior #lambda x: s_component_gaussian_prior(x, R=chi_max/3.) 
#        prior_map['s1z'] = s_component_gaussian_prior
#        prior_map['s2z'] = s_component_gaussian_prior

if opts.prior_gaussian_spin1_magnitude:
    if not  'chi1' in low_level_coord_names:
        print(" Incompatible options: gaussian spin1 prior requires polar coordinates")
        sys.exit(0)
    prior_map['chi1'] =functools.partial(gaussian_mass_prior,mu=0.7,sigma=0.1)  # not fully normalized particularly if chimax <1! Dangerous, fixme eventually

if opts.prior_tapered_spin1_magnitude:
    if not  'chi1' in low_level_coord_names:
        print(" Incompatible options: tapered spin1 prior requires polar coordinates")
        sys.exit(0)
    prior_map['s1z'] =tapered_magnitude_prior

if opts.prior_tapered_spin1z:
    if not  's1z' in low_level_coord_names:
        print(" Incompatible options: tapered spin1z prior requires cartesian coordinates")
        sys.exit(0)
    prior_map['s1z'] =tapered_magnitude_prior

if opts.prior_gaussian_mass_ratio:
    if not  'q' in low_level_coord_names:
        print(" Incompatible options: gaussian q prior requires q in coordinates (e.g., mtot,q coordinates)")
        sys.exit(0)
    prior_map['q'] = functools.partial(gaussian_mass_prior,mu=0.5,sigma=0.2)  # not fully normalized, and very ad-hoc

if opts.prior_tapered_mass_ratio:
    if not  'q' in low_level_coord_names:
        print(" Incompatible options: gaussian q prior requires q in coordinates (e.g., mtot,q coordinates)")
        sys.exit(0)
    prior_map['q'] = functools.partial(tapered_magnitude_prior_alt,loc=0.8,kappa=20.)  # not fully normalized, and very ad-hoc

if opts.prior_lambda_linear:
    if opts.prior_lambda_power == 1:
        prior_map['lambda1'] = functools.partial(mcsampler.linear_down_samp,xmin=0,xmax=lambda_max)
        prior_map['lambda2'] = functools.partial(mcsampler.linear_down_samp,xmin=0,xmax=lambda_small_max)
    else:
        prior_map['lambda1'] = functools.partial(mcsampler.power_down_samp,xmin=0,xmax=lambda_max,alpha=opts.prior_lambda_power+1)
        prior_map['lambda2'] = functools.partial(mcsampler.power_down_samp,xmin=0,xmax=lambda_small_max,alpha=opts.prior_lambda_power+1)


# tex_dictionary  = {
#  "mtot": '$M$',
#  "mc": '${\cal M}_c$',
#  "m1": '$m_1$',
#  "m2": '$m_2$',
#   "q": "$q$",
#   "delta" : "$\delta$",
#   "DeltaOverM2_perp" : "$\Delta_\perp$",
#   "DeltaOverM2_L" : "$\Delta_{||}$",
#   "SOverM2_perp" : "$S_\perp$",
#   "SOverM2_L" : "$S_{||}$",
#   "eta": "$\eta$",
#   "chi_eff": "$\chi_{eff}$",
#   "xi": "$\chi_{eff}$",
#   "s1z": "$\chi_{1,z}$",
#   "s2z": "$\chi_{2,z}$"
# }


###
### Linear fits. Resampling a quadratic. (Export me)
###

def fit_quadratic_alt(x,y,y_err=None,x0=None,symmetry_list=None,verbose=False):
    gamma_x = None
    if not (y_err is None):
        gamma_x =1./np.power(y_err,2)
    the_quadratic_results = BayesianLeastSquares.fit_quadratic( x, y,gamma_x=gamma_x,verbose=verbose)#x0=None)#x0_val_here)
    peak_val_est, best_val_est, my_fisher_est, linear_term_est,fn_estimate = the_quadratic_results

    np.savetxt("lnL_peakval.dat",[peak_val_est])   # generally not very useful
    np.savetxt("lnL_bestpt.dat",best_val_est)  
    np.savetxt("lnL_gamma.dat",my_fisher_est,header=' '.join(coord_names))
        

    bic  =-2*( -0.5*np.sum(np.power((y - fn_estimate(x)),2))/2 - 0.5* len(y)*np.log(len(x[0])) )

    print("  Fit: std :" , np.std( y-fn_estimate(x)))
    print("  Fit: BIC :" , bic)

    return fn_estimate


# https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/preprocessing/data.py#L1139
def fit_polynomial(x,y,x0=None,symmetry_list=None,y_errors=None):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

    clf_list = []
    bic_list = []
    for indx in np.arange(opts.fit_order+1):
        poly = msf.PolynomialFeatures(degree=indx,symmetry_list=symmetry_list)
        X_  = poly.fit_transform(x)

        if opts.verbose:
            print(" Fit : poly: RAW :", poly.get_feature_names())
            print(" Fit : ", poly.powers_)

        # Strip things with inappropriate symmetry: IMPOSSIBLE
        # powers_new = []
        # if not(symmetry_list is None):
        #  for line in poly.powers_:
        #     signature = np.prod(np.power( np.array(symmetry_list), line))
        #     if signature >0:
        #         powers_new.append(line)
        #  poly.powers_ = powers_new

        #  X_  = poly.fit_transform(x) # refit, with symmetry-constrained structure

        #  print " Fit : poly: After symmetry constraint :", poly.get_feature_names()
        #  print " Fit : ", poly.powers_


        clf = linear_model.LinearRegression()
        if y_errors is None or opts.ignore_errors_in_data:
            clf.fit(X_,y)
        else:
            assert len(y_errors) == len(y)
            clf.fit(X_,y,sample_weight=1./y_errors**2)  # fit with usual weights

        clf_list.append(clf)

        print(" Fit: Testing order ", indx)
        print(" Fit: std: ", np.std(y - clf.predict(X_)),  "using number of features ", len(y))  # should NOT be perfect
        if not (y_errors is None):
            print(" Fit: weighted error ", np.std( (y - clf.predict(X_))/y_errors))
        bic = -2*( -0.5*np.sum(np.power(y - clf.predict(X_),2))  - 0.5*len(y)*np.log(len(x[0])))
        print(" Fit: BIC:", bic)
        bic_list.append(bic)

    clf = clf_list[np.argmin(np.array(bic_list) )]

    return lambda x: clf.predict(poly.fit_transform(x))


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

def adderr(y):
    val,err = y
    return val+error_factor*err

def fit_gp(x,y,x0=None,symmetry_list=None,y_errors=None,hypercube_rescale=False,fname_export="gp_fit"):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

    # If we are loading a fit, override everything else
    if opts.fit_load_gp:
        print(" WARNING: Do not re-use fits across architectures or versions : pickling is not transferrable ")
        my_gp=joblib.load(opts.fit_load_gp)
        if opts.protect_coordinate_conversions:
            return lalsimutils.RangeProtectReduce(lambda x: my_gp.predict(x), -np.inf)
        return lambda x:my_gp.predict(x)

    # Amplitude: 
    #   - We are fitting lnL.  
    #   - We know the scale more or less: more than 2 in the log is bad
    # Scale
    #   - because of strong correlations with chirp mass, the length scales can be very short
    #   - they are rarely very long, but at high mass can be long
    #   - I need to allow for a RANGE

    length_scale_est = []
    length_scale_bounds_est = []
    for indx in np.arange(len(x[0])):
        # These length scales have been tuned by expereience
        length_scale_est.append( 2*np.std(x[:,indx])  )  # auto-select range based on sampling retained
        length_scale_min_here= np.max([1e-3,0.2*np.std(x[:,indx]/np.sqrt(len(x)))])
        if indx == mc_index:
            length_scale_min_here= 0.2*np.std(x[:,indx]/np.sqrt(len(x)))
            print(" Setting mc range: retained point range is ", np.std(x[:,indx]), " and target min is ", length_scale_min_here)
        length_scale_bounds_est.append( (length_scale_min_here , 5*np.std(x[:,indx])   ) )  # auto-select range based on sampling *RETAINED* (i.e., passing cut).  Note that for the coordinates I usually use, it would be nonsensical to make the range in coordinate too small, as can occasionally happens

    print(" GP: Input sample size ", len(x), len(y))
    print(" GP: Estimated length scales ")
    print(length_scale_est)
    print(length_scale_bounds_est)

    if not (hypercube_rescale):
        # These parameters have been hand-tuned by experience to try to set to levels comparable to typical lnL Monte Carlo error
        kernel = WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-2,1))+C(0.5, (1e-3,1e1))*RBF(length_scale=length_scale_est, length_scale_bounds=length_scale_bounds_est)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8)

        gp.fit(x,y)

        print(" Fit: std: ", np.std(y - gp.predict(x)),  "using number of features ", len(y))

        if opts.fit_save_gp:
            print(" Attempting to save fit ", opts.fit_save_gp+".pkl")
            joblib.dump(gp,opts.fit_save_gp+".pkl")
        
        if not (opts.fit_uncertainty_added):
            if opts.protect_coordinate_conversions:
                return lalsimutils.RangeProtectReduce( (lambda x: gp.predict(x) ), -np.inf)
            return lambda x: gp.predict(x)
        else:
            return lambda x: adderr(gp.predict(x,return_std=True))
    else:
        x_scaled = np.zeros(x.shape)
        x_center = np.zeros(len(length_scale_est))
        x_center = np.mean(x)
        print(" Scaling data to central point ", x_center)
        for indx in np.arange(len(x)):
            x_scaled[indx] = (x[indx] - x_center)/length_scale_est # resize

        kernel = WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-2,1))+C(0.5, (1e-3,1e1))*RBF( len(x_center), (1e-3,1e1))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8)
        
        gp.fit(x_scaled,y)
        print(" Fit: std: ", np.std(y - gp.predict(x_scaled)),  "using number of features ", len(y))  # should NOT be perfect

        return lambda x,x0=x_center,scl=length_scale_est: gp.predict( (x-x0 )/scl)

def map_funcs(func_list,obj):
    return [func(obj) for func in func_list]
def fit_gp_pool(x,y,n_pool=10,**kwargs):
    """
    Split the data into 10 parts, and return a GP that averages them
    """
    x_copy = np.array(x)
    y_copy = np.array(y)
    indx_list =np.arange(len(x_copy))
    np.random.shuffle(indx_list) # acts in place
    partition_list = np.array_split(indx_list,n_pool)
    gp_fit_list =[]
    for part in partition_list:
        print(" Fitting partition ")
        gp_fit_list.append(fit_gp(x[part],y[part],**kwargs))
    fn_out =  lambda x: np.mean( map_funcs( gp_fit_list,x), axis=0)
    print(" Testing ", fn_out([x[0]]))
    return fn_out

def fit_nn(x,y,y_errors=None,fname_export='nn_fit',adaptive=True):
    y_packed = y[:,np.newaxis]
    if not (y_errors is None):
        errors_packed = y_errors[:,np.newaxis]
    else:
        errors_packed = None
    import os
    working_dir = os.getcwd()
#    for indx in np.arange(len(x[0])):
#        print np.min(x[:,indx]), np.max(x[:,indx]), (np.max(x[:,indx])-np.mean(x[:,indx]))/np.std(x[:,indx])
    if adaptive:
        nn_interpolator = senni.AdaptiveInterpolator(x,y_packed,errors_packed,epochs=60, frac=0.2, hlayer_size=2**(1+len(x[0])), test_frac=0,working_dir=working_dir,loss_func='chi2',p_drop=0.02)  # May want to adjust size of network based on data size?
    else:
        nn_interpolator = senni.Interpolator(x,y_packed,errors_packed,epochs=100, frac=0.2, test_frac=0,working_dir=working_dir,loss_func='chi2',p_drop=0.03,regularize=False,weight_decay=1e-2)  # May want to adjust size of network based on data size?
    nn_interpolator.train()
    if opts.fit_save_gp:
        print( " Attempting to save NN fit ", opts.fit_save_gp+".network")
        nn_interpolator.save(opts.fit_save_gp+".network")

    def fn_return(x):
        x_in = np.copy(x)  # need to make a copy to avoid altering input/changing response
        return nn_interpolator.evaluate(x_in)

    print( " Demonstrating NN")   # debugging
#    print x, fn_return(x),nn_interpolator.evaluate(x),y
    residuals2 = fn_return(x) - y
    residuals = nn_interpolator.evaluate(x)-y
    print( "    std ", np.std(residuals), np.std(residuals2), np.max(y), np.max(fn_return(x)))
    return fn_return



def fit_rf(x,y,y_errors=None,fname_export='nn_fit'):
#    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    # Instantiate model. Usually not that many structures to find, don't overcomplicate
    #   - should scale like number of samples
    rf = ExtraTreesRegressor(n_estimators=100, verbose=True,n_jobs=-1) # no more than 5% of samples in a leaf
    if y_errors is None:
        rf.fit(x,y)
    else:
        rf.fit(x,y,sample_weight=1./y_errors**2)

    ### reject points with infinities : problems for inputs
    def fn_return(x_in,rf=rf):
        f_out = -lnL_default_large_negative*np.ones(len(x_in))
        # remove infinity or Nan
        indx_ok = np.all(np.isfinite(x_in),axis=-1)
        # rf internally uses float32, so we need to remove points > 10^37 or so ! 
        #    ... this *should* never happen due to bounds constraints, but ...
        indx_ok_size = np.all( np.logical_not(np.greater(np.abs(x_in),1e37)), axis=-1)
        indx_ok = np.logical_and(indx_ok, indx_ok_size)
        f_out[indx_ok] = rf.predict(x_in[indx_ok])
        return f_out
#    fn_return = lambda x_in: rf.predict(x_in) 

    print( " Demonstrating RF")   # debugging
    residuals = rf.predict(x)-y
    print( "    std ", np.std(residuals), np.max(y), np.max(fn_return(x)))
    return fn_return

def fit_nn_rfwrapper(x,y,y_errors=None,fname_export='nn_fit'):
    from sklearn.ensemble import RandomForestRegressor
    # Instantiate model. Usually not that many structures to find, don't overcomplicate
    #   - should scale like number of samples
    rf = RandomForestRegressor(n_estimators=100, random_state = 42,verbose=True,n_jobs=-1)
    if y_errors is None:
        rf.fit(x,y)
    else:
        rf.fit(x,y,sample_weight=1./y_errors**2)

    y_packed = y[:,np.newaxis]
    if not (y_errors is None):
        errors_packed = y_errors[:,np.newaxis]
    else:
        errors_packed = None
    import os
    working_dir = os.getcwd()
#    for indx in np.arange(len(x[0])):
#        print np.min(x[:,indx]), np.max(x[:,indx]), (np.max(x[:,indx])-np.mean(x[:,indx]))/np.std(x[:,indx])
    # train first with one loss, then the next?
    nn_interpolator = senni.Interpolator(x,y_packed,errors_packed,epochs=10, frac=0.2, test_frac=0,working_dir=working_dir,loss_func='mape')  # May want to adjust size of network based on data size?
    nn_interpolator.train()
    nn_interpolator.loss_func='chi2'; nn_interpolator.epochs = 50
    nn_interpolator.train()

    y_max = np.max(y)
    def fn_return(x):
        vals_rf = rf.predict(x)
        vals_nn = nn_interpolator.evaluate(x)
        return np.where(vals_rf > y_max-15, vals_nn, vals_rf)

    print( " Demonstrating NN")   # debugging
    residuals = fn_return(x)-y
    print( "    std ", np.std(residuals), np.max(y), np.max(fn_return(x)))
    return fn_return

if not(gpytorch_ok):
    def fit_gpytorch(x):
        sys.exit(1)
else:
  def fit_gpytorch(x,y,y_errors=None,fname_export='nn_fit',adaptive=True):
    y_packed = y[:,np.newaxis]
    if not (y_errors is None):
        errors_packed = y_errors[:,np.newaxis]
    else:
        errors_packed = None
    import os
    working_dir = os.getcwd()
    gp_interpolator = gpytorch_wrapper.Interpolator(x,y_packed,epochs=60) 
    gp_interpolator.train()
    if opts.fit_save_gp:
        print( " FAIL save gp fit - not yet implemented ")
#        gp_interpolator.save(opts.fit_save_gp+".network")

    def fn_return(x):
        x_in = np.copy(x)  # need to make a copy to avoid altering input/changing response
        return gp_interpolator.evaluate(x_in)

    print( " Demonstrating gpytorch fit ")   # debugging
    residuals2 = fn_return(x) - y
    residuals = nn_interpolator.evaluate(x)-y
    print( "    std ", np.std(residuals), np.std(residuals2), np.max(y), np.max(fn_return(x)))
    return fn_return


if internalGP_ok:
    from RIFT.interpolators.internal_GP import fit_gp as fit_gp_sparse
else:
    def fit_gp_sparse(x):
        sys.exit(1)






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
if opts.input_distance:
    print(" Distance input")
    col_lnL +=1
dat_orig = dat = np.loadtxt(opts.fname)
dat_orig = dat[dat[:,col_lnL].argsort()] # sort  http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
print(" Original data size = ", len(dat), dat.shape)

 ###
 ### Convert data.  Use lalsimutils for flexibility
 ###
P_list = []
dat_out =[]
 
extra_plot_coord_names = [ ['mtot', 'q', 'xi'], ['mc', 'eta'], ['m1', 'm2'], ['s1z','s2z'] ] # replot
if 's1x' in opts.parameter:  # in practice, we always use the transverse cartesian components 
    print(" Plotting coordinates include spin magnitude and transverse spins ")
    extra_plot_coord_names += [['chi1_perp', 's1z'], ['chi2_perp','s2z'],['chi1','chi2'],['cos_theta1','cos_theta2']]
if 'lambda1' in opts.parameter or 'LambdaTilde' in opts.parameter or (not( opts.parameter_implied is None) and  ( 'LambdaTilde' in opts.parameter_implied)):
    print(" Plotting coordinates include tides")
    extra_plot_coord_names += [['mc', 'eta', 'LambdaTilde'],['lambda1', 'lambda2'], ['LambdaTilde', 'DeltaLambdaTilde'], ['m1','lambda1'], ['m2','lambda2']]
dat_out_low_level_coord_names = []
dat_out_extra = []
for item in extra_plot_coord_names:
    dat_out_extra.append([])


symmetry_list =lalsimutils.symmetry_sign_exchange(coord_names)  # identify symmetry due to exchange
mc_min = 1e10
mc_max = -1

mc_index = -1 # index of mchirp in parameter index. To help with nonstandard GP
mc_cut_range = [-np.inf, np.inf] 
if opts.mc_range:
    mc_cut_range = eval(opts.mc_range)  # throw out samples outside this range.
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
    if opts.input_distance:
        P.dist = lal.PC_SI*1e6*line[9]  # Incompatible with tides, note!
    
    if opts.contingency_unevolved_neff == "quadpuff":
        P_copy = P.manual_copy()  # prevent duplication
        P_list_in.append(P_copy) # store  it, make sure distinct

    # INPUT GRID: Evaluate binary parameters on fitting coordinates
    line_out = np.zeros(len(coord_names)+2)
    for x in np.arange(len(coord_names)):
        line_out[x] = P.extract_param(coord_names[x])
 #        line_out[x] = getattr(P, coord_names[x])
    line_out[-2] = line[col_lnL]
    line_out[-1] = line[col_lnL+1]  # adjoin error estimate
    dat_out.append(line_out)

    # Alternate grids: Evaluate input binary parameters on other coordinates requested, for comparison
    for indx in np.arange(len(extra_plot_coord_names)):
        line_out = np.zeros(len(extra_plot_coord_names[indx]))
        for x in np.arange(len(line_out)):
            line_out[x] = P.extract_param( extra_plot_coord_names[indx][x])
        dat_out_extra[indx].append(line_out)

    # results using sampling coordinates (low_level_coord_names) 
    line_out = np.zeros(len(low_level_coord_names))
    for x in np.arange(len(line_out)):
        fac = 1
        if low_level_coord_names[x] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        line_out[x] = P.extract_param(low_level_coord_names[x])/fac
        if low_level_coord_names[x] in ['mc']:
            mc_index = x
    dat_out_low_level_coord_names.append(line_out)


    # Update mc range
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

        # Alternate grids: Evaluate input binary parameters on other coordinates requested, for comparison
        for indx in np.arange(len(extra_plot_coord_names)):
            line_out = np.zeros(len(extra_plot_coord_names[indx]))
            for x in np.arange(len(line_out)):
                line_out[x] = P.extract_param( extra_plot_coord_names[indx][x])
            dat_out_extra[indx].append(line_out)

        # results using sampling coordinates (low_level_coord_names) 
        line_out = np.zeros(len(low_level_coord_names))
        for x in np.arange(len(line_out)):
            fac = 1
            if low_level_coord_names[x] in ['mc','m1','m2','mtot']:
                fac = lal.MSUN_SI
            line_out[x] = P.extract_param(low_level_coord_names[x])/fac
            if low_level_coord_names[x] in ['mc']:
                mc_index = x
        dat_out_low_level_coord_names.append(line_out)

Pref_default = P.copy()  # keep this around to fix the masses, if we don't have an inj

# Force 32 bit dype
dat_out = np.array(dat_out,dtype=internal_dtype)
print(" Stripped size  = ", dat_out.shape,  " with memory usage (bytes) ", sys.getsizeof(dat_out))
dat_out_low_level_coord_names = np.array(dat_out_low_level_coord_names)
 # scale out mass units
for p in ['mc', 'm1', 'm2', 'mtot']:
    if p in coord_names:
        indx = coord_names.index(p)
        dat_out[:,indx] /= lal.MSUN_SI
    for x in np.arange(len(extra_plot_coord_names)):
        dat_out_extra[x] = np.array(dat_out_extra[x],dtype=internal_dtype)  # put into np form
        if p in extra_plot_coord_names[x]:
            indx = extra_plot_coord_names[x].index(p)
            dat_out_extra[x][:,indx] /= lal.MSUN_SI
            


# Repack data
X =dat_out[:,0:len(coord_names)]
Y = dat_out[:,-2]
Y_err = dat_out[:,-1]
# Save copies for later (plots)
X_orig = X.copy()
Y_orig = Y.copy()

# Plot cumulative distribution in lnL, for all points.  Useful sanity check for convergence.  Start with RAW
if not opts.no_plots:
    Yvals_copy = Y_orig.copy()
    Yvals_copy = Yvals_copy[Yvals_copy.argsort()[::-1]]
    pvals = np.arange(len(Yvals_copy))*1.0/len(Yvals_copy)
    plt.plot(Yvals_copy, pvals)
    plt.xlabel(r"$\ln{\cal L}$")
    plt.ylabel(r"evaluation fraction $(<\ln{\cal L})$")
    plt.savefig("lnL_cumulative_distribution_of_input_points.png"); plt.clf()


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
elif sum(indx_ok) < 10: # and max_lnL > 30:
    # mark the top 10 elements and use them for fits
    # this may be VERY VERY DANGEROUS if the peak is high and poorly sampled
    idx_sorted_index = np.lexsort((np.arange(len(Y)), Y))  # Sort the array of Y, recovering index values
    indx_list = np.array( [[k, Y[k]] for k in idx_sorted_index])     # pair up with the weights again
    indx_list = indx_list[::-1]  # reverse, so most significant are first
    indx_ok = list(map(int,indx_list[:10,0]))
    print(" Revised number of points for fit: ", sum(indx_ok), indx_ok, indx_list[:10])
X_raw = X.copy()

my_fit= None
if opts.fit_method == "quadratic":
    print(" FIT METHOD ", opts.fit_method, " IS QUADRATIC")
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    if opts.report_best_point:
        my_fit = fit_quadratic_alt(X,Y,symmetry_list=symmetry_list,verbose=opts.verbose)
        pt_best_X = np.loadtxt("lnL_bestpt.dat")
        for indx in np.arange(len(coord_names)):
            fac = 1
            if coord_names[indx] in ['mc','m1','m2','mtot']:
                fac = lal.MSUN_SI
            p_to_assign = coord_names[indx]
            if p_to_assign == 'xi':
                p_to_assign = "chieff_aligned"
            P.assign_param(p_to_assign,pt_best_X[indx]*fac) 
           
        print(" ====BEST BINARY ====")
        print(" Parameters from fit ", pt_best_X)
        P.print_params()
        sys.exit(0)
    my_fit = fit_quadratic_alt(X,Y,symmetry_list=symmetry_list,verbose=opts.verbose)
elif opts.fit_method == "polynomial":
    print(" FIT METHOD ", opts.fit_method, " IS POLYNOMIAL")
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    my_fit = fit_polynomial(X,Y,symmetry_list=symmetry_list,y_errors=Y_err)
elif opts.fit_method == 'gp_hyper':
    print(" FIT METHOD ", opts.fit_method, " IS GP with hypercube rescaling")
    # some data truncation IS used for the GP, but beware
    print(" Truncating data set used for GP, to reduce memory usage needed in matrix operations")
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    my_fit = fit_gp(X,Y,y_errors=Y_err,hypercube_rescale=True)
elif opts.fit_method == 'gp':
    print(" FIT METHOD ", opts.fit_method, " IS GP")
    # some data truncation IS used for the GP, but beware
    print(" Truncating data set used for GP, to reduce memory usage needed in matrix operations")
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    my_fit = fit_gp(X,Y,y_errors=Y_err)
elif opts.fit_method == 'gp-pool':
    print(" FIT METHOD ", opts.fit_method, " IS GP (pooled) with pool size ", opts.pool_size)
    # some data truncation IS used for the GP, but beware
    print(" Truncating data set used for GP, to reduce memory usage needed in matrix operations")
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    if opts.pool_size == None:
        opts.pool_size = np.max([2,np.round(4000/len(X))])  # pick a pool size that has no more than 4000 members per pool
    my_fit = fit_gp_pool(X,Y,y_errors=Y_err,n_pool=opts.pool_size)
elif opts.fit_method == 'gp-torch':
    print( " FIT METHOD ", opts.fit_method, " IS gpytorch ")
    # NO data truncation for NN needed?  To be *consistent*, have the code function the same way as the others
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    my_fit = fit_gpytorch(X,Y,y_errors=Y_err)
elif opts.fit_method == 'nn':
    print( " FIT METHOD ", opts.fit_method, " IS NN ")
    # NO data truncation for NN needed?  To be *consistent*, have the code function the same way as the others
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    my_fit = fit_nn(X,Y,y_errors=Y_err)
elif opts.fit_method == 'rf':
    print( " FIT METHOD ", opts.fit_method, " IS RF ")
    # NO data truncation for NN needed?  To be *consistent*, have the code function the same way as the others
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    my_fit = fit_rf(X,Y,y_errors=Y_err)
elif opts.fit_method == 'nn_rfwrapper':
    print( " FIT METHOD ", opts.fit_method, " IS NN with RF wrapper ")
    # NO data truncation for NN needed?  To be *consistent*, have the code function the same way as the others
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    my_fit = fit_nn_rfwrapper(X,Y,y_errors=Y_err)
elif opts.fit_method == 'gp-sparse':
    print( " FIT METHOD ", opts.fit_method, " IS gp-sparse ")
    if not internalGP_ok:
        print( " FAILED ")
        sys.exit(1)
    # NO data truncation for NN needed?  To be *consistent*, have the code function the same way as the others
    X=X[indx_ok]
    Y=Y[indx_ok] - lnL_shift
    Y_err = Y_err[indx_ok]
    dat_out_low_level_coord_names =     dat_out_low_level_coord_names[indx_ok]
    # Cap the total number of points retained, AFTER the threshold cut
    if opts.cap_points< len(Y) and opts.cap_points> 100:
        n_keep = opts.cap_points
        indx = np.random.choice(np.arange(len(Y)),size=n_keep,replace=False)
        Y=Y[indx]
        X=X[indx]
        Y_err=Y_err[indx]
        dat_out_low_level_coord_names = dat_out_low_level_coord_names[indx]
    my_fit = fit_gp_sparse(X,Y,y_errors=Y_err)




# Sort for later convenience (scatterplots, etc)
indx = Y.argsort()#[::-1]
X=X[indx]
Y=Y[indx]
dat_out_low_level_coord_names =dat_out_low_level_coord_names[indx]

# Make grid plots for all pairs of points, to facilitate direct validation of where posterior support lies
if not no_plots:
 import itertools
 for i, j in itertools.product( np.arange(len(coord_names)),np.arange(len(coord_names)) ):
  if i < j:
    plt.scatter( X[:,i],X[:,j],label='rapid_pe:'+opts.desc_ILE,c=Y); plt.legend(); plt.colorbar()
    x_name = render_coord(coord_names[i])
    y_name = render_coord(coord_names[j])
    plt.xlabel( x_name)
    plt.ylabel( y_name )
    plt.title("rapid_pe evaluations (=inputs); no fits")
    plt.savefig("scatter_"+coord_names[i]+"_"+coord_names[j]+".png"); plt.clf()


###
### Coordinate conversion tool
###
if not opts.using_eos:
 def convert_coords(x_in):
    return lalsimutils.convert_waveform_coordinates(x_in, coord_names=coord_names,low_level_coord_names=low_level_coord_names,source_redshift=source_redshift,enforce_kerr=opts.downselect_enforce_kerr)
else:
 def convert_coords(x_in):
    x_out = lalsimutils.convert_waveform_coordinates_with_eos(x_in, coord_names=coord_names,low_level_coord_names=low_level_coord_names,eos_class=my_eos,no_matter1=opts.no_matter1, no_matter2=opts.no_matter2,source_redshift=source_redshift,enforce_kerr=opts.downselect_enforce_kerr)
    return x_out


###
### Integrate posterior
###


sampler = mcsampler.MCSampler()
if opts.sampler_method == "GMM":
    sampler = mcsamplerEnsemble.MCSampler()


##
## Loop over param names
##
for p in low_level_coord_names:
    if not(opts.parameter_implied is None):
       if p in opts.parameter_implied:
        # We do not need to sample parameters that are implied by other parameters, so we can overparameterize 
        continue
    prior_here = prior_map[p]
    range_here = prior_range_map[p]

    ## Special case: mc : provide ability to override range
    if p == 'mc' and opts.mc_range:
        range_here = eval(opts.mc_range)
    elif p=='mc' and opts.trust_sample_parameter_box:
        range_here = [np.max([range_here[0],mc_min]), np.min([range_here[1],mc_max])]
    if p =='mtot' and opts.mtot_range:
        range_here = eval(opts.mtot_range)

    sampler.add_parameter(p, pdf=np.vectorize(lambda x:1), prior_pdf=prior_here,left_limit=range_here[0],right_limit=range_here[1],adaptive_sampling=True)


# Import prior
if not(opts.import_prior_dictionary_file is None):
    dat  =     joblib.load(opts.import_prior_dictionary_file)
#    print dat
    stored_param_prior_pdf = dat[0]
    stored_param_ranges =dat[1]
    for param in stored_param_prior_pdf.keys():
        if param in sampler.prior_pdf.keys():
            print(" OVER-RIDE OF PRIOR FOR ", param, stored_param_prior_pdf[param], stored_param_ranges[param])
#        stored_param_pdf[param] = sampler.pdf[param]
            sampler.prior_pdf[param] = stored_param_prior_pdf[param] 
            sampler.llim[param], sampler.rlim[param] = stored_param_ranges[param]
        

# Export prior
if not(opts.output_prior_dictionary_file is None):
    stored_param_prior_pdf = {}
    stored_param_ranges = {}
    for param in sampler.params:
#        stored_param_pdf[param] = sampler.pdf[param]
        stored_param_prior_pdf[param] = sampler.prior_pdf[param]
        stored_param_ranges[param] = [sampler.llim[param],sampler.rlim[param]]
    joblib.dump([stored_param_prior_pdf,stored_param_ranges],opts.output_prior_dictionary_file)
#    with open(opts.output_prior_dictionary_file,'w') as f:
#        pickle.dump([stored_param_prior_pdf,stored_param_ranges],f)


likelihood_function = None
if len(low_level_coord_names) ==1:
    def likelihood_function(x):  
        if isinstance(x,float):
            return np.exp(my_fit([x]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x],dtype=internal_dtype).T) ))
            return np.exp(my_fit(convert_coords(np.c_[x])))
if len(low_level_coord_names) ==2:
    def likelihood_function(x,y):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y])))
if len(low_level_coord_names) ==3:
    def likelihood_function(x,y,z):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z])))
if len(low_level_coord_names) ==4:
    def likelihood_function(x,y,z,a):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a])))
if len(low_level_coord_names) ==5:
    def likelihood_function(x,y,z,a,b):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b])))
if len(low_level_coord_names) ==6:
    def likelihood_function(x,y,z,a,b,c):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b,c],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c])))
if len(low_level_coord_names) ==7:
    def likelihood_function(x,y,z,a,b,c,d):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d])))
if len(low_level_coord_names) ==8:
    def likelihood_function(x,y,z,a,b,c,d,e):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e])))
if len(low_level_coord_names) ==9:
    def likelihood_function(x,y,z,a,b,c,d,e,f):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e,f]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e,f])))
if len(low_level_coord_names) ==10:
    def likelihood_function(x,y,z,a,b,c,d,e,f,g):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e,f,g]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e,f,g])))


n_step = 1e5
my_exp = np.min([1,0.8*np.log(n_step)/np.max(Y)])   # target value : scale to slightly sublinear to (n_step)^(0.8) for Ymax = 200. This means we have ~ n_step points, with peak value wt~ n_step^(0.8)/n_step ~ 1/n_step^(0.2), limiting contrast
#my_exp = np.max([my_exp,  1/np.log(n_step)]) # do not allow extreme contrast in adaptivity, to the point that one iteration will dominate
print(" Weight exponent ", my_exp, " and peak contrast (exp)*lnL = ", my_exp*np.max(Y), "; exp(ditto) =  ", np.exp(my_exp*np.max(Y)), " which should ideally be no larger than of order the number of trials in each epoch, to insure reweighting doesn't select a single preferred bin too strongly.  Note also the floor exponent also constrains the peak, de-facto")


extra_args={}
if opts.sampler_method == "GMM":
    n_max_blocks = ((1.0*int(opts.n_max))/n_step) 
    def parse_corr_params(my_str):
        """
        Takes a string with no spaces, and returns a tuple
        """
        corr_param_names = my_str.replace(',',' ').split()
        corr_param_indexes = []
        for param in corr_param_names:
            try:
                indx = low_level_coord_names.index(param)
                corr_param_indexes.append(indx)
            except:
                continue
        return tuple(corr_param_indexes)
    if opts.internal_correlate_parameters == 'all':
        gmm_dict = {tuple(range(len(low_level_coord_names))):None} # integrate *jointly* in all parameters together
    elif not (opts.internal_correlate_parameters is None):
        # Correlate identified parameters
        my_blocks = opts.internal_correlate_parameters.split()
        my_tuples = list(map( parse_corr_params, my_blocks))
        gmm_dict = {x:None for x in my_tuples}
        # What about un-labelled parameters? Make a null tuple for them as well
        correlated_params = set(()); correlated_params = correlated_params.union( *list(map(set,my_tuples)))
        uncorrelated_params = set(np.arange(len(low_level_coord_names))); 
        uncorrelated_params = uncorrelated_params.difference(correlated_params)
        for x in uncorrelated_params:
            gmm_dict[(x,)] = None
        print( " Using correlated GMM sampling on sampling variable indexes " , gmm_dict, " out of ", low_level_coord_names)
    else:
        param_indexes = range(len(low_level_coord_names))
        gmm_dict  = {(k,):None for k in param_indexes} # no correlations
    extra_args = {'n_comp':3,'max_iter':n_max_blocks,'L_cutoff': (np.exp(max_lnL-lnL_shift - opts.lnL_offset)),'gmm_dict':gmm_dict}  # made up for now, should adjust
    
# Result shifted by lnL_shift
res, var, neff, dict_return = sampler.integrate(likelihood_function, *low_level_coord_names,  verbose=True,nmax=int(opts.n_max),n=n_step,neff=opts.n_eff, save_intg=True,tempering_adapt=True, floor_level=1e-3,igrand_threshold_p=1e-3,convergence_tests=test_converged,adapt_weight_exponent=my_exp,no_protect_names=True, **extra_args)  # weight ecponent needs better choice. We are using arbitrary-name functions

# Test n_eff threshold
if not (opts.fail_unless_n_eff is None):
    if neff < opts.fail_unless_n_eff:
        print(" FAILURE: n_eff too small")
        sys.exit(1)
if neff < opts.n_eff:   
    print(" ==> neff (={}) is low <==".format(neff))
    if opts.contingency_unevolved_neff == 'quadpuff'  and neff < np.min([500,opts.n_eff]): # we can usually get by with about 500 points
        # Add errors
        # Note we only want to add errors to RETAINED points
        print(" Contingency: quadpuff: take covariance of points, draw from it again, add to existing points as offsets (i.e. a puffball) ")
        n_output_size = np.min([len(P_list_in),opts.n_output_samples])
        print(" Preparing to write ", n_output_size , " samples ")

        my_cov = np.cov(X.T)  # covariance of data points
        rv = scipy.stats.multivariate_normal(mean=np.zeros(len(X[0])), cov=my_cov,allow_singular=True)  # they are just complaining about dynamic range
        delta_X = rv.rvs(size=len(X))
        X_new = X+delta_X
        P_out_list = []
        # Loop over points 
        # Jitter using the parameters we use to fit with
        for indx_P in np.arange(np.min([len(P_list_in),len(X)])):   # make sure no past-limits errors
            include_item=True
            P = P_list_in[indx_P]
            for indx in np.arange(len(coord_names)):
                param  = coord_names[indx]
                fac = 1
                if coord_names[indx] in ['mc', 'mtot', 'm1', 'm2']:
                    fac = lal.MSUN_SI
                P.assign_param(param, (X_new[indx_P,indx]*fac))
            for p in downselect_dict.keys():
                val = P.extract_param(p) 
                if p in ['mc','m1','m2','mtot']:
                    val = val/lal.MSUN_SI
                if val < downselect_dict[p][0] or val > downselect_dict[p][1]:
                    include_item = False
            if include_item:
                P_out_list.append(P)

        # Save output
        lalsimutils.ChooseWaveformParams_array_to_xml(P_out_list[:n_output_size],fname=opts.fname_output_samples,fref=P.fref)
        sys.exit(0)



# Save result -- needed for odds ratios, etc.
#   Warning: integral_result.dat uses *original* prior, before any reweighting
np.savetxt(opts.fname_output_integral+".dat", [np.log(res)+lnL_shift])
eos_extra = []
annotation_header = "lnL sigmaL neff "
if opts.using_eos:
    eos_extra = [opts.using_eos]
    annotation_header += 'eos_name '
    if opts.eos_param == 'spectral':
        # Should also 
        my_eos_params = my_eos.spec_params
        eos_extra += list(map( lambda x: str(my_eos_params[x]), ["gamma1", "gamma2", "gamma3", "gamma4", "p0", "epsilon0", "xmax"]))
#        eos_extra += opts.eos_param
        annotation_header += "gamma1 gamma2 gamma3 gamma4 p0 epsilon0 xmax"
with open(opts.fname_output_integral+"+annotation.dat", 'w') as file_out:
    str_out =list( map(str,[np.log(res), np.sqrt(var)/res, neff]))
    file_out.write("# " + annotation_header + "\n")
    file_out.write(' '.join( str_out + eos_extra + ["\n"]))
#np.savetxt(opts.fname_output_integral+"+annotation.dat", np.array([[np.log(res), np.sqrt(var)/res, neff]]), header=eos_extra)

if neff < len(low_level_coord_names):
    print(" PLOTS WILL FAIL ")
    print(" Not enough independent Monte Carlo points to generate useful contours")




samples = sampler._rvs
print(samples.keys())
n_params = len(coord_names)
dat_mass = np.zeros((len(samples[low_level_coord_names[0]]),n_params+3))
dat_logL = np.log(samples["integrand"])
lnLmax = np.max(dat_logL[np.isfinite(dat_logL)])
print(" Max lnL ", np.max(dat_logL))
if opts.lnL_protect_overflow:
    lnL_shift = lnLmax - 100.

# Throw away stupid points that don't impact the posterior
indx_ok = np.logical_and(dat_logL > lnLmax-opts.lnL_offset ,samples["joint_s_prior"]>0)
for p in low_level_coord_names:
    samples[p] = samples[p][indx_ok]
dat_logL  = dat_logL[indx_ok]
samples["joint_prior"] =samples["joint_prior"][indx_ok]
samples["joint_s_prior"] =samples["joint_s_prior"][indx_ok]



###
### 1d posteriors of the coordinates used for sampling  [EQUALLY WEIGHTED, BIASED because physics cuts aren't applied]
###

p = samples["joint_prior"]
ps =samples["joint_s_prior"]
lnL = dat_logL
lnLmax = np.max(lnL)
weights = np.exp(lnL-lnLmax)*p/ps


# If we are using pseudo uniform spin magnitude, reweight
#     ONLY done if we use s1x, s1y, s1z, s2x, s2y, s2z
# volumetric prior scales as a1^2 a2^2 da1 da2; we need to undo it
if opts.pseudo_uniform_magnitude_prior and 's1x' in samples.keys() and 's1z' in samples.keys():
    prior_weight = np.prod([prior_map[x](samples[x]) for x in ['s1x','s1y','s1z'] ],axis=0)
    val = np.array(samples["s1z"]**2+samples["s1y"]**2 + samples["s1x"]**2,dtype=internal_dtype)
    chi1 = np.sqrt(val)  # weird typecasting problem
    weights *= 3.*chi_max*chi_max/(chi1*chi1*prior_weight)   # prior_weight accounts for the density, in cartesian coordinates
    weights[ chi1>chi_max] =0
    if 's2z' in samples.keys():
        prior_weight = np.prod([prior_map[x](samples[x]) for x in ['s2x','s2y','s2z'] ],axis=0)
        val = np.array(samples["s2z"]**2+samples["s2y"]**2 + samples["s2x"]**2,dtype=internal_dtype)
        chi2= np.sqrt(val)
        weights[ chi2>chi_small_max] =0
        weights *= 3.*chi_small_max*chi_small_max/(chi2*chi2*prior_weight)
elif opts.pseudo_uniform_magnitude_prior and  'chiz_plus' in samples.keys() and not opts.pseudo_uniform_magnitude_prior_alternate_sampling:
    # Uniform sampling: simple volumetric reweight
    s1z  = samples['chiz_plus'] + samples['chiz_minus']
    s2z  = samples['chiz_plus'] - samples['chiz_minus']
    val1 = np.array(s1z**2+samples["s1y"]**2 + samples["s1x"]**2,dtype=internal_dtype); chi1 = np.sqrt(val1)
    val2 = np.array(s2z**2+samples["s2y"]**2 + samples["s2x"]**2,dtype=internal_dtype); chi2= np.sqrt(val2)
    indx_ok = np.logical_and(chi1<=chi_max , chi2<=chi_small_max)
    weights[ np.logical_not(indx_ok)] = 0  # Zero out failing samples. Has effect of fixing prior range!
    weights[indx_ok] *= 9.*(chi_max**2 * chi_small_max**2)/(chi1*chi1*chi2*chi2)[indx_ok]
elif opts.pseudo_uniform_magnitude_prior and  'chiz_plus' in samples.keys() and not opts.pseudo_uniform_magnitude_prior_alternate_sampling:
    s1z  = samples['chiz_plus'] + samples['chiz_minus']
    s2z  = samples['chiz_plus'] - samples['chiz_minus']
    val1 = np.array(s1z**2+samples["s1y"]**2 + samples["s1x"]**2,dtype=internal_dtype); chi1 = np.sqrt(val1)
    val2 = np.array(s2z**2+samples["s2y"]**2 + samples["s2x"]**2,dtype=internal_dtype); chi2= np.sqrt(val2)
    indx_ok = np.logical_and(chi1<=chi_max , chi2<=chi_small_max)
    weights[ np.logical_not(indx_ok)] = 0  # Zero out failing samples. Has effect of fixing prior range!
    prior_weight = np.prod([prior_map[x](samples[x]) for x in ['s1x','s1y', 's2x', 's2y','chiz_plus','chiz_minus'] ],axis=0)
    weights[indx_ok] *= 9.*(chi_max**2  * chi_small_max**2)/(chi1*chi1*chi2*chi2)[indx_ok]/prior_weight[indx_ok]  # undo chizplus, chizminus prior
    

# If we are using alignedspin-zprior AND chiz+, chiz-, then we need to reweight .. that prior cannot be evaluated internally
# Prevent alignedspin-zprior from being used when transverse spins are present ... no sense!
# Note we need to downslelect early in this case
if opts.aligned_prior =="alignedspin-zprior" and 'chiz_plus' in samples.keys()  and (not 's1x' in samples.keys()):
    prior_weight = np.prod([prior_map[x](samples[x]) for x in ['chiz_plus','chiz_minus'] ],axis=0)
    s1z  = samples['chiz_plus'] + samples['chiz_minus']
    s2z  =samples['chiz_plus'] - samples['chiz_minus']
    indx_ok = np.logical_and(np.abs(s1z)<=chi_max , np.abs(s2z)<=chi_max)
    weights[ np.logical_not(indx_ok)] = 0  # Zero out failing samples. Has effect of fixing prior range!
    weights[indx_ok] *= s_component_zprior( s1z[indx_ok])*s_component_zprior(s2z[indx_ok])/(prior_weight[indx_ok])  # correct for uniform

if opts.pseudo_gaussian_mass_prior:
    # mass normalization (assuming mc, eta limits are bounds - as is invariably the case)
    mass_area = 0.5*(mc_max**2 - mc_min**2)*(unscaled_eta_prior_cdf(eta_range[0]) - unscaled_eta_prior_cdf(eta_range[1]))
    # Extract m1 and m2, i solar mass units
    m1 = np.zeros(len(weights))
    m2 = np.zeros(len(weights))
    for indx in np.arange(len(weights)):
        P=lalsimutils.ChooseWaveformParams()
        for indx_name in np.arange(len(low_level_coord_names)):
            p = low_level_coord_names[indx_name]
            # Do not bother to scale by solar masses, only to undo it later
            P.assign_param(p, samples[p][indx])
        m1[indx] = P.extract_param('m1')
        m2[indx] = P.extract_param('m2')
    # For speed, do this transformation to mass coordinates by hand rather than the usual loop
    # m1=None
    # m2=None
    # if 'm1' in samples.keys():  # will never happen
    #     m1 = samples['m1']
    #     m2 = samples['m2']
    # elif 'mc' in samples.keys():  #almost always true
    #     mc = samples['mc']
    #     eta = None
    #     if 'eta' in samples.keys():
    #         eta = samples['eta']
    #     elif 'delta_mc' in samples.keys():
    #         eta = np.array(0.25*(1-samples['delta_mc']**2)) # see definition
    #     else:
    #         print " Failed transformation "
    #         sys.exit(0)
    #     print type(mc), type(eta)
    #     m1 = lalsimutils.mass1(mc,eta)
    #     m2 = lalsimutils.mass2(mc,eta)
    # else:
    #     print " Failed transformation"
    #     sys.exit(0)
    # Reormalize mass region. Note normalizatoin issue introduced: no boundaries in mass region used to rescale.
    weights *= mass_area*gaussian_mass_prior( m1, opts.pseudo_gaussian_mass_prior_mean,opts.pseudo_gaussian_mass_prior_std)*gaussian_mass_prior( m2, opts.pseudo_gaussian_mass_prior_mean,opts.pseudo_gaussian_mass_prior_std)


# Integral result v2: using modified prior. 
# Note also downselects NOT applied: no range cuts, unless applied as part of aligned_prior, etc.  
#   - use for Bayes factors with GREAT CARE for this reason; should correct for with indx_ok
log_res_reweighted = lnLmax + np.log(np.mean(weights))
sigma_reweighted= np.std(weights,dtype=np.float128)/np.mean(weights)
neff_reweighted = np.sum(weights)/np.max(weights)
np.savetxt(opts.fname_output_integral+"_withpriorchange.dat", [log_res_reweighted])  # should agree with the usual result, if no prior changes
with open(opts.fname_output_integral+"_withpriorchange+annotation.dat", 'w') as file_out:
    str_out = list(map(str,[log_res_reweighted, sigma_reweighted, neff]))
    file_out.write("# " + annotation_header + "\n")
    file_out.write(' '.join( str_out + eos_extra + ["\n"]))
#np.savetxt(opts.fname_output_integral+"_withpriorchange+annotation.dat", np.array([[log_res_reweighted,sigma_reweighted, neff]]),header=eos_extra)

# Load in reference parameters
Pref = lalsimutils.ChooseWaveformParams()
if  opts.inj_file is not None:
    Pref = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj_file)[opts.event_num]
else:
    Pref.m1 = Pref_default.m1
    Pref.m2 = Pref_default.m2
Pref.fref = opts.fref  # not encoded in the XML!
Pref.print_params()

if not no_plots:
 for indx in np.arange(len(low_level_coord_names)):
   try:
    dat_out = []; dat_out_LI=[]
    p = low_level_coord_names[indx]
    print(" -- 1d cumulative "+ str(indx)+ ":"+ low_level_coord_names[indx]+" ----")
    dat_here = samples[low_level_coord_names[indx]]
    range_x = [np.min(dat_here), np.max(dat_here)]
    if opts.fname_lalinference:
        dat_LI =  extract_combination_from_LI( samples_LI, low_level_coord_names[indx])
        if not(dat_LI is None):
            range_x[0] = np.min([range_x[0], np.min(dat_LI)])
            range_x[1] = np.max([range_x[1], np.max(dat_LI)])
    for x in np.linspace(range_x[0],range_x[1],200):
         dat_out.append([x, np.sum( weights[ dat_here< x])/np.sum(weights)])
         # if opts.fname_lalinference:
         #     tmp = extract_combination_from_LI(p)
         #     if not (tmp == None) :
         #        dat_out_LI.append([x, 1.0*sum( tmp<x)/len(tmp) ]) 
         if opts.fname_lalinference and (p in remap_ILE_2_LI.keys()): 
             dat_out_LI.append([x, (1.0*np.sum( samples_LI[ remap_ILE_2_LI[p] ]< x))/len(samples_LI) ])
    
    np.savetxt(p+"_cdf_nocut_beware.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    plt.plot(dat_out[:,0],dat_out[:,1],label="rapid_pe:"+opts.desc_ILE,color='b')
    if opts.fname_lalinference: # and  (p in remap_ILE_2_LI.keys()):
        plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],label="LI:"+opts.desc_lalinference,color='r')
   
    # Add vertical line
    here_val = Pref.extract_param(p)
    fac = 1
    if p in ['mc','m1','m2','mtot']:
        fac = lal.MSUN_SI
    here_val = here_val/fac
    print(" Vertical line ", p, " ", here_val)
    plt.axvline(here_val,color='k',linestyle='dashed')

    x_name = render_coord(p)
    plt.xlabel(x_name); plt.legend()
    y_name  = x_name.replace('$','')
    y_name = "$P(<"+y_name + ")$"
    plt.ylabel(y_name)
    plt.title("CDF: "+x_name)
    plt.savefig(p+"_cdf_nocut_beware.png"); plt.clf()
   except:
      plt.clf()  # clear plot, just in case
      print(" No 1d plot for variable")


###
### Corner 1 [BIASED, does not account for sanity cuts on physical variables]
###

# Labels for corner plots
if not no_plots:
    black_line = mlines.Line2D([], [], color='black', label='rapid_pe:'+opts.desc_ILE)
    red_line =mlines.Line2D([], [], color='red', label='LI:'+opts.desc_lalinference)
    green_line =mlines.Line2D([], [], color='green', label='rapid_pe (evaluation points)' )
    blue_line =mlines.Line2D([], [], color='blue', label='rapid_pe (evaluation points, good fit)' )
    line_handles = [black_line,green_line,blue_line]
#corner_legend_location=(0., 1.0, 1., .7)
    corner_legend_location=(0.7, 1.0)
    corner_legend_prop = {'size':6}
# https://stackoverflow.com/questions/7125009/how-to-change-legend-size-with-matplotlib-pyplot
#params = {'legend.fontsize': 20, 'legend.linewidth': 2}
#plt.rcParams.update(params)
    if opts.fname_lalinference:
        line_handles = [black_line,red_line,green_line,blue_line]


print(" ---- Corner 1: Sampling coordinates (NO CONSTRAINTS APPLIED HERE: BIASED) ---- ")
dat_mass = np.zeros( (len(lnL),len(low_level_coord_names)),dtype=np.float64)
dat_mass_LI = []
if opts.fname_lalinference:
    dat_mass_LI = np.zeros( (len(samples_LI), len(low_level_coord_names)), dtype=np.float64)
if not no_plots:
  for indx in np.arange(len(low_level_coord_names)):
    dat_mass[:,indx] = samples[low_level_coord_names[indx]]
    if opts.fname_lalinference and low_level_coord_names[indx] in remap_ILE_2_LI.keys() :
#        tmp = extract_combination_from_LI[samples_LI, low_level_coord_names[indx]]
#        if not (tmp==None):
#            dat_mass_LI[:,indx]
     if remap_ILE_2_LI[low_level_coord_names[indx]] in samples_LI.dtype.names:
        dat_mass_LI[:,indx] = samples_LI[ remap_ILE_2_LI[low_level_coord_names[indx]] ]
    if opts.fname_lalinference and low_level_coord_names[indx] in ["lambda1", "lambda2"]:
        print(" populating ", low_level_coord_names[indx], " via _extract ")
        dat_mass_LI[:,indx] = extract_combination_from_LI(samples_LI, low_level_coord_names[indx])  # requires special extraction technique, since it needs to be converted

truth_here = []
for indx in np.arange(len(low_level_coord_names)):
    fac = 1
    if low_level_coord_names[indx] in ['mc','m1','m2','mtot']:
        fac = lal.MSUN_SI
    truth_here.append(Pref.extract_param(low_level_coord_names[indx])/fac)


CIs = [0.95,0.9, 0.68]
quantiles_1d = [0.05,0.95]
range_here = []
if not no_plots:
  for p in low_level_coord_names:
#    print p, prior_range_map[p]
    range_here.append(prior_range_map[p])
    if (range_here[-1][1] < np.mean(samples[p])+2*np.std(samples[p])  ):
         range_here[-1][1] = np.mean(samples[p])+2*np.std(samples[p])
    if (range_here[-1][0] > np.mean(samples[p])-2*np.std(samples[p])  ):
         range_here[-1][0] = np.mean(samples[p])-2*np.std(samples[p])
    # Don't let lower limit get too extreme
    if (range_here[-1][0] < np.mean(samples[p])-3*np.std(samples[p])  ):
         range_here[-1][0] = np.mean(samples[p])-3*np.std(samples[p])
    # Don't let upper limit get too extreme
    if (range_here[-1][1] > np.mean(samples[p])+5*np.std(samples[p])  ):
         range_here[-1][1] = np.mean(samples[p])+5*np.std(samples[p])
    if range_here[-1][0] < prior_range_map[p][0]:
        range_here[-1][0] = prior_range_map[p][0]
    if range_here[-1][1] > prior_range_map[p][1]:
        range_here[-1][1] = prior_range_map[p][1]
    if opts.fname_lalinference:
        # If LI samples are shown, make sure their full posterior range is plotted.
      try:
        print(p)
        p_LI  = remap_ILE_2_LI[p]
        if range_here[-1][0] > np.min(samples_LI[p_LI]):
            range_here[-1][0] = np.min(samples_LI[p_LI])
        if range_here[-1][1] < np.max(samples_LI[p_LI]):
            range_here[-1][1] = np.max(samples_LI[p_LI])
      except:
          print(" Parameter failure with LI, trying extract_combination_... ")
          tmp = extract_combination_from_LI(samples_LI, p)
          if range_here[-1][0] > np.min(tmp):
            range_here[-1][0] = np.min(tmp)
          if range_here[-1][1] < np.max(tmp):
            range_here[-1][1] = np.max(tmp)
    print(p, range_here[-1])  # print out range to be used in plots.

if not no_plots:
    labels_tex = list(map(lambda x: tex_dictionary[x], low_level_coord_names))
    fig_base = corner.corner(dat_mass[:,:len(low_level_coord_names)], weights=(weights/np.sum(weights)).astype(np.float64),labels=labels_tex, quantiles=quantiles_1d,plot_datapoints=False,plot_density=False,no_fill_contours=True,fill_contours=False,levels=CIs,truths=truth_here,range=range_here)
    my_cmap_values = 'g' # default color
    if True:
    #try:
# Plot simulation points (X array): MAY NOT BE POSSIBLE if dimensionality is inconsistent
        cm = plt.cm.get_cmap('RdYlBu_r')
        y_span = Y.max() - Y.min()
        y_min = Y.min()
    #    print y_span, y_min
        my_cmap_values = map(tuple,cm( (Y-y_min)/y_span) )
        my_cmap_values ='g'

        fig_base = corner.corner(dat_out_low_level_coord_names,weights=np.ones(len(X))/len(X), plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base, data_kwargs={'c':my_cmap_values},hist_kwargs={'color':'g', 'linestyle':'dashed'},range=range_here)

        # TRUNCATED data set used here
        indx_ok = Y > Y.max() - scipy.stats.chi2.isf(0.1,len(low_level_coord_names))/2  # approximate threshold for significant points,from inverse cdf 90%
        n_ok = np.sum(indx_ok)
        fig_base  = corner.corner(dat_out_low_level_coord_names[indx_ok],weights=np.ones(n_ok)*1.0/n_ok, plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base, data_kwargs={'c':'b'},hist_kwargs={'color':'b', 'linestyle':'dashed'},range=range_here)

    #except:
    else:
        print(" Some ridiculous range error with the corner plots, again")

    if opts.fname_lalinference:
      try:
        corner.corner( dat_mass_LI,color='r',labels=labels_tex,weights=np.ones(len(dat_mass_LI))*1.0/len(dat_mass_LI),fig=fig_base,quantiles=quantiles_1d,no_fill_contours=True,plot_datapoints=False,plot_density=False,fill_contours=False,levels=CIs) #,range=range_here)
      except:
          print(" Failed !")
    plt.legend(handles=line_handles, bbox_to_anchor=corner_legend_location, prop=corner_legend_prop,loc=4)
    plt.savefig("posterior_corner_nocut_beware.png"); plt.clf()

print(" ---- Subset for posterior samples (and further corner work) --- ")

# pick random numbers
p_threshold_size = np.min([5*opts.n_output_samples,len(weights)])
p_thresholds =  np.random.uniform(low=0.0,high=1.0,size=p_threshold_size)#opts.n_output_samples)
if opts.verbose:
    print(" output size: selected thresholds N=", len(p_thresholds))
# find sample indexes associated with the random numbers
#    - FIXME: first truncate the bad ones
# idx_sorted_index = numpy.lexsort((numpy.arange(len(weights)), weights))  # Sort the array of weights, recovering index values
# indx_list = numpy.array( [[k, weights[k]] for k in idx_sorted_index])     # pair up with the weights again
# cum_weights = np.cumsum(indx_list[:,1)
# cum_weights = cum_weights/cum_weights[-1]
# indx_list = [indx_list[k, 0] for k, value in enumerate(cum_sum > deltaP) if value]  # find the indices that preserve > 1e-7 of total probabilit
cum_sum  = np.cumsum(weights)
cum_sum = cum_sum/cum_sum[-1]
indx_list = list(map(lambda x : np.sum(cum_sum < x),  p_thresholds))  # this can lead to duplicates
if opts.verbose:
    print(" output size: selected random indices N=", len(indx_list))
lnL_list = []
P_list =[]
P = lalsimutils.ChooseWaveformParams()
P.approx = lalsim.GetApproximantFromString(opts.approx_output)
#P.approx = lalsim.SEOBNRv2  # DEFAULT
P.fmin = opts.fmin # DEFAULT
P.fref = opts.fref
for indx_here in indx_list:
        line = [samples[p][indx_here] for p in low_level_coord_names]
        Pgrid = P.manual_copy()
        Pgrid.fref = opts.fref  # Just to make SURE
        include_item =True
        # Set attributes that are being changed as necessary, leaving all others fixed
        for indx in np.arange(len(low_level_coord_names)):
            # Skip crazy configurations (e.g., violate Kerr bound)
            # if parameter involes a mass parameter, scale it to sensible units
            fac = 1
            if low_level_coord_names[indx] in ['mc', 'mtot', 'm1', 'm2']:
                fac = lal.MSUN_SI
            # do assignment of parameters anyways, as we will skip it momentarily
            coord_to_assign = low_level_coord_names[indx]
            if coord_to_assign == 'xi':
                coord_to_assign= 'chieff_aligned'
            Pgrid.assign_param(coord_to_assign, line[indx]*fac)
#            print indx_here, coord_to_assign, line[indx]
        # Test for downselect
        for p in downselect_dict.keys():
            val = Pgrid.extract_param(p) 
            if p in ['mc','m1','m2','mtot']:
                val = val/lal.MSUN_SI
            if val < downselect_dict[p][0] or val > downselect_dict[p][1]:
                    include_item = False
                    if opts.verbose:
                        print(" Sample: Skipping " , line, ' due to ', p, val, downselect_dict[p])

        # Set some superfluous quantities, needed only for PN approximants, so the result is generated sensibly
        Pgrid.ampO =opts.amplitude_order
        Pgrid.phaseO =opts.phase_order

        # Downselect.
        # for param in downselect_dict:
        #     if Pgrid.extract_param(param) < downselect_dict[param][0] or Pgrid.extract_param(param) > downselect_dict[param][1]:
        #         print " Skipping " , line
        #         include_item =False
        if include_item:
         if Pgrid.m2 <= Pgrid.m1:  # do not add grid elements with m2> m1, to avoid possible code pathologies !
            P_list.append(Pgrid)
            lnL_list.append(np.log(samples["integrand"][indx_here]))
         else:
            Pgrid.swap_components()  # IMPORTANT.  This should NOT change the physical functionality FOR THE PURPOSES OF OVERLAP (but will for PE - beware phiref, etc!)
            P_list.append(Pgrid)
            lnL_list.append(np.log(samples["integrand"][indx_here]))
        else:
            True



 ###
 ### Export data
 ###
n_output_size = np.min([len(P_list),opts.n_output_samples])
lalsimutils.ChooseWaveformParams_array_to_xml(P_list[:n_output_size],fname=opts.fname_output_samples,fref=P.fref)
lnL_list = np.array(lnL_list,dtype=internal_dtype)
np.savetxt(opts.fname_output_samples+"_lnL.dat", lnL_list)


if not opts.no_plots:
    Yvals_copy = lnL_list.copy()
    Yvals_copy = Yvals_copy[Yvals_copy.argsort()[::-1]]
    pvals = np.arange(len(Yvals_copy))*1.0/len(Yvals_copy)
    plt.plot(Yvals_copy, pvals)
    plt.xlabel(r"$\ln{\cal L}$")
    plt.ylabel(r"$\hat{P}(<\ln{\cal L})$")
    plt.savefig("lnL_cumulative_distribution_posterior_estimate.png"); plt.clf()



###
### Identify, save best point
###

P_best = P_list[ np.argmax(lnL_list)  ]
lalsimutils.ChooseWaveformParams_array_to_xml([P_best], "best_point_by_lnL")
lnL_best = lnL_list[np.argmax(lnL_list)]
np.savetxt("best_point_by_lnL_value.dat", np.array([lnL_best]));


###
### STOP IF NO MORE PLOTS
###
if no_plots:
    sys.exit(0)

###
### Extract data from samples, in array form. INCLUDES any cuts (e.g., kerr limit)
###
dat_mass_post = np.zeros( (len(P_list),len(coord_names)),dtype=np.float64)
for indx_line  in np.arange(len(P_list)):
    for indx in np.arange(len(coord_names)):
        fac=1
        if coord_names[indx] in ['mc', 'mtot', 'm1', 'm2']:
                fac = lal.MSUN_SI
        dat_mass_post[indx_line,indx] = P_list[indx_line].extract_param(coord_names[indx])/fac


dat_extra_post = []
for x in np.arange(len(extra_plot_coord_names)):
    coord_names_here = extra_plot_coord_names[x]
    feature_here = np.zeros( (len(P_list),len(coord_names_here)),dtype=np.float64)
    for indx_line  in np.arange(len(P_list)):
        for indx in np.arange(len(coord_names_here)):
            fac=1
            if coord_names_here[indx] in ['mc', 'mtot', 'm1', 'm2']:
                fac = lal.MSUN_SI
            feature_here[indx_line,indx] = P_list[indx_line].extract_param(coord_names_here[indx])/fac
    dat_extra_post.append(feature_here)



range_here=[]
for indx in np.arange(len(coord_names)):    
    range_here.append( [np.min(dat_mass_post[:, indx]),np.max(dat_mass_post[:, indx])])
    # Manually reset some ranges to be more useful for plotting
    if coord_names[indx] in ['xi', 'chi_eff']:
        range_here[-1] = [-1,1]
    if coord_names[indx] in ['eta']:
        range_here[-1] = [0,0.25]
        if opts.eta_range:
            range_here[-1] = eval(opts.eta_range)
    if coord_names[indx] in ['q']:
        range_here[-1] = [0,1]
    if coord_names[indx] in ['s1z', 's2z']:
        range_here[-1] = [-1,1]
    if coord_names[indx] in ['chi1_perp', 'chi2_perp']:
        range_here[-1] = [0,1]
    print(coord_names[indx], range_here[-1])

if opts.fname_lalinference:
    dat_mass_LI = np.zeros( (len(samples_LI), len(coord_names)), dtype=np.float64)
    for indx in np.arange(len(coord_names)):
        if coord_names[indx] in remap_ILE_2_LI.keys():
            tmp = extract_combination_from_LI(samples_LI, coord_names[indx])
            if not (tmp is None):
                dat_mass_LI[:,indx] = tmp
        if coord_names[indx] in ["lambda1", "lambda2"]:
            print(" populating ", coord_names[indx], " via _extract ")
            dat_mass_LI[:,indx] = extract_combination_from_LI(samples_LI, coord_names[indx])  # requires special extraction technique, since it needs to be converted

        if range_here[indx][0] > np.min(dat_mass_LI[:,indx]):
            range_here[indx][0] = np.min(dat_mass_LI[:,indx])
        if range_here[indx][1] < np.max(dat_mass_LI[:,indx]):
            range_here[indx][1] = np.max(dat_mass_LI[:,indx])

print(" ---- 1d cumulative on fitting coordinates (NOT biased: FROM SAMPLES, including downselect) --- ")
for indx in np.arange(len(coord_names)):
    p = coord_names[indx]
    dat_out = []; dat_out_LI=[]
    print(" -- 1d cumulative "+ str(indx)+ ":"+ coord_names[indx]+" ----")
    dat_here = dat_mass_post[:,indx]
    wt_here = np.ones(len(dat_here))
    range_x = [np.min(dat_here), np.max(dat_here)]
    if opts.fname_lalinference:
        dat_LI = extract_combination_from_LI(samples_LI,p)
        if not(dat_LI is None):
            range_x[0] = np.min([range_x[0], np.min(dat_LI)])
            range_x[1] = np.max([range_x[1], np.max(dat_LI)])

    for x in np.linspace(range_x[0],range_x[1],200):
         dat_out.append([x, np.sum(  wt_here[dat_here< x] )/len(dat_here)])    # NO WEIGHTS for these resampled points
#         dat_out.append([x, np.sum( weights[ dat_here< x])/np.sum(weights)])
         if opts.fname_lalinference and not (dat_LI is None) :
                dat_out_LI.append([x, 1.0*sum( dat_LI<x)/len(dat_LI) ]) 
#         if opts.fname_lalinference and (p in remap_ILE_2_LI.keys()) :
#             dat_out_LI.append([x, (1.0*np.sum( samples_LI[ remap_ILE_2_LI[p] ]< x))/len(samples_LI) ])
    np.savetxt(p+"_alt_cdf.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    plt.plot(dat_out[:,0],dat_out[:,1],label="rapid_pe:"+opts.desc_ILE,color='b')
    if opts.fname_lalinference and (p in remap_ILE_2_LI.keys()) and not (dat_out_LI is None):
        dat_out_LI = np.array(dat_out_LI)
        try:
            plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],label="LI:"+opts.desc_lalinference,color='r')
        except:
            print("  - plot failure - ")
    # Add vertical line
    here_val = Pref.extract_param(p)
    fac = 1
    if p in ['mc','m1','m2','mtot']:
        fac = lal.MSUN_SI
    here_val = here_val/fac
    print(" Vertical line ", p, " ", here_val)
    plt.axvline(here_val,color='k',linestyle='dashed')


    x_name = render_coord(p)
    plt.xlabel(x_name); plt.legend()
    y_name  = x_name.replace('$','')
    y_name = "$P(<"+y_name + ")$"
    plt.ylabel(y_name)
    plt.savefig(p+"_alt_cdf.png"); plt.clf()


print(" ---- Corner 2: Fitting coordinates (+ original sample point overlay) ---- ")

###
### Corner plot.  Also overlay sample points
###

truth_here = []
for indx in np.arange(len(coord_names)):
    fac = 1
    if coord_names[indx] in ['mc','m1','m2','mtot']:
        fac = lal.MSUN_SI
    truth_here.append(Pref.extract_param(coord_names[indx])/fac)


try:
#if True:
 labels_tex = render_coordinates(coord_names)#map(lambda x: tex_dictionary[x], coord_names)
 fig_base = corner.corner(dat_mass_post[:,:len(coord_names)], weights=np.ones(len(dat_mass_post))*1.0/len(dat_mass_post),labels=labels_tex, quantiles=quantiles_1d,plot_datapoints=False,plot_density=False,no_fill_contours=True,fill_contours=False,levels=CIs, range=range_here,truths=truth_here)

 if opts.fname_lalinference:
    fig_base=corner.corner( dat_mass_LI,color='r',labels=labels_tex,weights=np.ones(len(dat_mass_LI))*1.0/len(dat_mass_LI),fig=fig_base,quantiles=quantiles_1d,no_fill_contours=True,plot_datapoints=False,plot_density=False,fill_contours=False,levels=CIs,range=range_here)

 # BEFORE truncation, note, to highlight region explored. ONLY for this plot
 fig_base = corner.corner(X_orig, weights=np.ones(len(X_orig))/len(X_orig),plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base, data_kwargs={'c':'g'},hist_kwargs={'color':'g', 'linestyle':'dashed'},range=range_here)
 # A subset of the truncated data set
 indx_ok = Y > Y.max() - scipy.stats.chi2.isf(0.1,len(low_level_coord_names))/2  # approximate threshold for significant points,from inverse cdf 90%
 n_ok = np.sum(indx_ok)
 fig_base  = corner.corner(X[indx_ok],weights=np.ones(n_ok)*1.0/n_ok, plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base, data_kwargs={'c':'r'},hist_kwargs={'color':'b', 'linestyle':'dashed'},range=range_here)


 plt.legend(handles=line_handles, bbox_to_anchor=corner_legend_location, prop=corner_legend_prop,loc=4)
 plt.savefig("posterior_corner_fit_coords.png"); plt.clf()

except:
#else:
    print(" No corner 2")


###
### Corner plot 3
###
print(" ---- Corner 3: Bonus corner plots ---- ")
for indx in np.arange(len(extra_plot_coord_names)):
 if True:
# try:
    fig_base =None
    coord_names_here = extra_plot_coord_names[indx]
    str_name = '_'.join(coord_names_here)
    print(" Generating corner for ", str_name)
    dat_here = dat_extra_post[indx]
    dat_points_here  = dat_out_extra[indx]
    labels_tex = render_coordinates(coord_names_here)#map(lambda x: tex_dictionary[x], coord_names_here)
    range_here=[]
    dat_mass_LI = None

    can_render_LI = opts.fname_lalinference and  ( set(coord_names_here) < (set(samples_LI.dtype.names) | set(remap_ILE_2_LI.keys()) | set(['lambda1','lambda2','LambdaTilde','DeltaLambdaTilde'])) )

    if  can_render_LI:
        print("   - LI parameters available for ", coord_names_here)
        dat_mass_LI = np.zeros( (len(samples_LI), len(coord_names_here)), dtype=np.float64)
        for x in np.arange(len(coord_names_here)):
            print("    ... extracting ", coord_names_here[x])
            tmp = extract_combination_from_LI(samples_LI, coord_names_here[x])
            if not (tmp is None):
                dat_mass_LI[:,x] = tmp
#                print "   .....   ", tmp[:3]
            else:
                print("   ... warning, extraction failed for ", coord_names_here[x])
    for z in np.arange(len(coord_names_here)):    
        range_here.append( [np.min(dat_points_here[:, z]),np.max(dat_points_here[:, z])])
        if not (opts.plots_do_not_force_large_range):
         # Manually reset some ranges to be more useful for plotting
         if coord_names_here[z] in ['xi', 'chi_eff']:
            range_here[-1] = [-1,1]             # can be horrible if we are very informative
         if coord_names_here[z] in ['eta']:
            range_here[-1] = [0,0.25]         # can be horrible if we are very informative
         if coord_names_here[z] in ['q']:
            range_here[-1] = [0,1]
         if coord_names_here[z] in ['s1z', 's2z']:
            range_here[-1] = [-1,1]
         if coord_names_here[z] in ['chi1_perp', 'chi2_perp']:
            range_here[-1] = [0,1]
        if opts.fname_lalinference and  ( set(coord_names_here) < set(remap_ILE_2_LI.keys())):
                range_here[-1][0] =  np.min( [np.min( dat_mass_LI[:,z]), range_here[-1][0]  ])
                range_here[-1][1] =  np.max( [np.max( dat_mass_LI[:,z]), range_here[-1][1]  ])
        if coord_names_here[z] in prior_range_map.keys():
            range_here[-1][1] = np.min( [range_here[-1][1], prior_range_map[coord_names_here[z]][1]])  # override the limits, if I have a prior.

        print('   - Range ', coord_names_here[z], range_here[-1])
      
    truth_here = []
    for z in np.arange(len(coord_names_here)):
        fac=1
        if coord_names_here[z] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        truth_here.append(Pref.extract_param(coord_names_here[z])/fac)

    print(" Truth here for ", coord_names_here, truth_here)

    print(" Generating figure for ", extra_plot_coord_names[indx], " using ", len(dat_here), " from the posterior and ",  len(dat_points_here) , len(Y_orig), " from the original data set ")
    fig_base = corner.corner(dat_here, weights=np.ones(len(dat_here))*1.0/len(dat_here), labels=labels_tex, quantiles=quantiles_1d,plot_datapoints=False,plot_density=False,no_fill_contours=True,fill_contours=False,levels=CIs,range=range_here,truths=truth_here)
                
    if can_render_LI:
        corner.corner( dat_mass_LI, weights=np.ones(len(dat_mass_LI))*1.0/len(dat_mass_LI), color='r',labels=labels_tex,fig=fig_base,quantiles=quantiles_1d,no_fill_contours=True,plot_datapoints=False,plot_density=False,fill_contours=False,levels=CIs,range=range_here)


    print(" Rendering past samples for ",  extra_plot_coord_names[indx], " based on ", len(dat_points_here))
    fig_base = corner.corner(dat_points_here,weights=np.ones(len(dat_points_here))*1.0/len(dat_points_here), plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base, data_kwargs={'color':'g'},hist_kwargs={'color':'g', 'linestyle':'dashed'},range=range_here)
    # Render points available. Note we use the ORIGINAL data set, and truncate it
    indx_ok = Y_orig > Y_orig.max() - scipy.stats.chi2.isf(0.1,len(low_level_coord_names))/2  # approximate threshold for significant points,from inverse cdf 90%
    n_ok = np.sum(indx_ok)
    print(" Adding points for figure ", n_ok, extra_plot_coord_names[indx], " drawn from original  ")
    fig_base  = corner.corner(dat_points_here[indx_ok],weights=np.ones(n_ok)*1.0/n_ok, plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base, data_kwargs={'c':'b'},hist_kwargs={'color':'b', 'linestyle':'dashed'},range=range_here)


    plt.legend(handles=line_handles, bbox_to_anchor=corner_legend_location, prop=corner_legend_prop,loc=4)
    print(" Writing coord ", str_name)
    plt.savefig("posterior_corner_extra_coords_"+str_name+".png"); plt.clf()

# except:
 else:
     print(" Failed to generate corner for ", extra_plot_coord_names[indx])

sys.exit(0)


