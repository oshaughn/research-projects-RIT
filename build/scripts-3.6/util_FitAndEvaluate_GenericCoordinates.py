#! /usr/bin/env python
#
# GOAL
#   - load in lnL data
#   - fit peak to quadratic (standard), GP, etc. 
#   - evaluate,  based on some parameter grid
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
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools
import itertools

no_plots = True


try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.lines as mlines
    import corner

    no_plots=False
except ImportError:
    print(" - no matplotlib - ")


from sklearn.preprocessing import PolynomialFeatures
import RIFT.misc.ModifiedScikitFit as msf  # altenative polynomialFeatures
from sklearn import linear_model

from glue.ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)

import RIFT.integrators.mcsampler as mcsampler


def render_coord(x):
    if x in lalsimutils.tex_dictionary.keys():
        return tex_dictionary[x]
    if 'product(' in x:
        a=x.replace(' ', '') # drop spaces
        a = a[:len(a)-1] # drop last
        a = a[8:]
        terms = a.split(',')
        exprs =map(render_coord, terms)
        exprs = map( lambda x: x.replace('$', ''), exprs)
        my_label = ' '.join(exprs)
        return '$'+my_label+'$'
    else:
        return x

def render_coordinates(coord_names):
    return map(render_coord, coord_names)


parser = argparse.ArgumentParser()
parser.add_argument("--maximize-mass",action='store_true', help="If true, maximize the likelihood for each value of the total mass. Ignore any grid placement in total mass")
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--input-tides",action='store_true',help="Use input format with tidal fields included.")
parser.add_argument("--fname-xml-base",help="filename of xml file to use as base (e.g., to specify m1,m2, chi1, chi2, ... to minimize burden on ascii file")
parser.add_argument("--fname-parameter-grid",help="filename of ascii parameters to use to evaluate the fit")
parser.add_argument("--fname-out",default="eval.dat")
parser.add_argument("--fref",default=20,type=float, help="Reference frequency used for spins in the ILE output.  (Since I usually use SEOBNRv3, the best choice is 20Hz)")
parser.add_argument("--fmin",type=float,default=20)
parser.add_argument("--fname-rom-samples",default=None,help="*.rom_composite output. Treated identically to set of posterior samples produced by mcsampler after constructing fit.")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
parser.add_argument("--mc-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--eta-range",default=None,help="Eta range. Important if we have a BNS or other item that has a strong constraint.")
parser.add_argument("--mtot-range",default=None,help="Chirp mass range [mc1,mc2]. Important if we have a low-mass object, to avoid wasting time sampling elsewhere.")
parser.add_argument("--trust-sample-parameter-box",action='store_true', help="If used, sets the prior range to the SAMPLE range for any parameters. NOT IMPLEMENTED. This should be automatically done for mc!")
parser.add_argument("--plots-do-not-force-large-range",action='store_true', help = "If used, the plots do NOT automatically set the chieff range to [-1,1], the eta range to [0,1/4], etc")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--chi-max", default=1,type=float,help="Maximum range of 'a' allowed.  Use when comparing to models that aren't calibrated to go to the Kerr limit.")
parser.add_argument("--parameter-nofit", action='append', help="Parameter used to initialize the implied parameters, and varied at a low level, but NOT the fitting parameters")
parser.add_argument("--use-precessing",action='store_true')
parser.add_argument("--lnL-offset",type=float,default=10,help="lnL offset")
parser.add_argument("--lnL-cut",type=float,default=None,help="lnL cut [MANUAL]")
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
parser.add_argument("--n-max",default=3e5,type=float)
parser.add_argument("--n-eff",default=3e3,type=int)
parser.add_argument("--fit-method",default="quadratic",help="quadratic|polynomial|gp|gp_hyper")
parser.add_argument("--fit-order",type=int,default=2,help="Fit order (polynomial case: degree)")
parser.add_argument("--fit-uncertainty-added",default=False, action='store_true', help="Reported likelihood is lnL+(fit error). Use for placement and use of systematic errors.")
parser.add_argument("--no-plots",action='store_true')
opts=  parser.parse_args()
no_plots = no_plots |  opts.no_plots

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
    dlist_ranges  = map(eval,opts.downselect_parameter_range)
else:
    dlist = []
    dlist_ranges = []
if len(dlist) != len(dlist_ranges):
    print(" downselect parameters inconsistent", dlist, dlist_ranges)
for indx in np.arange(len(dlist_ranges)):
    downselect_dict[dlist[indx]] = dlist_ranges[indx]


chi_max = opts.chi_max
downselect_dict['chi1'] = [0,chi_max]
downselect_dict['chi2'] = [0,chi_max]
for param in ['s1z', 's2z', 's1x','s2x', 's1y', 's2y']:
    downselect_dict[param] = [-chi_max,chi_max]
# Enforce definition of eta
downselect_dict['eta'] = [0,0.25]



test_converged={}
#test_converged['neff'] = functools.partial(mcsampler.convergence_test_MostSignificantPoint,0.01)  # most significant point less than 1/neff of probability.  Exactly equivalent to usual neff threshold.
#test_converged["normal_integral"] = functools.partial(mcsampler.convergence_test_NormalSubIntegrals, 25, 0.01, 0.1)   # 20 sub-integrals are gaussian distributed [weakly; mainly to rule out outliers] *and* relative error < 10%, based on sub-integrals . Should use # of intervals << neff target from above.  Note this sets our target error tolerance on  lnLmarg.  Note the specific test requires >= 20 sub-intervals, which demands *very many* samples (each subintegral needs to be converged).


prior_range_map = {"mtot": [1, 300], "q":[0.01,1], "s1z":[-0.999*chi_max,0.999*chi_max], "s2z":[-0.999*chi_max,0.999*chi_max], "mc":[0.9,250], "eta":[0.01,0.2499999], 'xi':[-chi_max,chi_max],'chi_eff':[-chi_max,chi_max],'delta':[-1,1],
   's1x':[-chi_max,chi_max],
   's2x':[-chi_max,chi_max],
   's1y':[-chi_max,chi_max],
   's2y':[-chi_max,chi_max],
  'm1':[0.9,1e3],
  'm2':[0.9,1e3],
  'lambda1':[0.01,4000],
  'lambda2':[0.01,4000],
  # strongly recommend you do NOT use these as parameters!  Only to insure backward compatibility with LI results
  'LambdaTilde':[0.01,5000],
  'DeltaLambdaTilde':[-500,500],
}

if not (opts.eta_range is None):
    print(" Warning: Overriding default eta range. USE WITH CARE")
    prior_range_map['eta'] = eval(opts.eta_range)  # really only useful if eta is a coordinate.  USE WITH CARE




# TeX dictionary
tex_dictionary = lalsimutils.tex_dictionary


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

def fit_gp(x,y,x0=None,symmetry_list=None,y_errors=None,hypercube_rescale=False):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

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

    print(" GP: Estimated length scales ")
    print(length_scale_est)
    print(length_scale_bounds_est)

    if not (hypercube_rescale):
        # These parameters have been hand-tuned by experience to try to set to levels comparable to typical lnL Monte Carlo error
        kernel = WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-2,1))+C(0.5, (1e-3,1e1))*RBF(length_scale=length_scale_est, length_scale_bounds=length_scale_bounds_est)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8)

        gp.fit(x,y)

        print(" Fit: std: ", np.std(y - gp.predict(x)),  "using number of features ", len(y))

        if not (opts.fit_uncertainty_added):
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
print(" Coordinate names for fit :, ", coord_names)
print(" Rendering coordinate names : ",  render_coordinates(coord_names))  # map(lambda x: tex_dictionary[x], coord_names)
print(" Symmetry for these fitting coordinates :", lalsimutils.symmetry_sign_exchange(coord_names))
print(" Coordinate names for Monte Carlo :, ", low_level_coord_names)
print(" Rendering coordinate names : ", map(lambda x: tex_dictionary[x], low_level_coord_names))

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
dat_orig = dat = np.loadtxt(opts.fname)
dat_orig = dat[dat[:,col_lnL].argsort()] # sort  http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
print(" Original data size = ", len(dat), dat.shape)

 ###
 ### Convert data.  Use lalsimutils for flexibility
 ###
P_list = []
dat_out =[]
 

symmetry_list =lalsimutils.symmetry_sign_exchange(coord_names)  # identify symmetry due to exchange
mc_min = 1e10
mc_max = -1

mc_index = -1 # index of mchirp in parameter index. To help with nonstandard GP
mc_cut_range = [-np.inf, np.inf] 
if opts.mc_range:
    mc_cut_range = eval(opts.mc_range)  # throw out samples outside this range
print(" Stripping samples outside of ", mc_cut_range, " in mc")
P= lalsimutils.ChooseWaveformParams()
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
  if line[col_lnL] < opts.lnL_cut:
      continue  # strip worthless points.  DANGEROUS
  mc_here = lalsimutils.mchirp(line[1],line[2])
  if mc_here < mc_cut_range[0] or mc_here > mc_cut_range[1]:
      if opts.verbose:
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

    # INPUT GRID: Evaluate binary parameters on fitting coordinates
    line_out = np.zeros(len(coord_names)+2)
    for x in np.arange(len(coord_names)):
        line_out[x] = P.extract_param(coord_names[x])
 #        line_out[x] = getattr(P, coord_names[x])
    line_out[-2] = line[col_lnL]
    line_out[-1] = line[col_lnL+1]  # adjoin error estimate
    dat_out.append(line_out)


    # results using sampling coordinates (low_level_coord_names) 
    line_out = np.zeros(len(low_level_coord_names))
    for x in np.arange(len(line_out)):
        fac = 1
        if low_level_coord_names[x] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        line_out[x] = P.extract_param(low_level_coord_names[x])/fac
        if low_level_coord_names[x] in ['mc','mtot']:  # only use one overall mass index
            mc_index = x


    # Update mc range
    mc_here = lalsimutils.mchirp(line[1],line[2])
    if mc_here < mc_min:
        mc_min = mc_here
    if mc_here > mc_max:
        mc_max = mc_here

Pref_default = P.copy()  # keep this around to fix the masses, if we don't have an inj

dat_out = np.array(dat_out)
print(" Stripped size  = ", dat_out.shape)
 # scale out mass units
for p in ['mc', 'm1', 'm2', 'mtot']:
    if p in coord_names:
        indx = coord_names.index(p)
        dat_out[:,indx] /= lal.MSUN_SI
            


# Repack data
X =dat_out[:,0:len(coord_names)]
Y = dat_out[:,-2]
Y_err = dat_out[:,-1]

# Eliminate values with Y too small
max_lnL = np.max(Y)
indx_ok = Y>np.max(Y)-opts.lnL_offset
print(" Points used in fit : ", sum(indx_ok), " given max lnL ", max_lnL)
if max_lnL < 10:
    # nothing matters, we will reject it anyways
    indx_ok = np.ones(len(Y),dtype=bool)
elif sum(indx_ok) < 10: # and max_lnL > 30:
    # mark the top 10 elements and use them for fits
    # this may be VERY VERY DANGEROUS if the peak is high and poorly sampled
    idx_sorted_index = np.lexsort((np.arange(len(Y)), Y))  # Sort the array of Y, recovering index values
    indx_list = np.array( [[k, Y[k]] for k in idx_sorted_index])     # pair up with the weights again
    indx_list = indx_list[::-1]  # reverse, so most significant are first
    indx_ok = map(int,indx_list[:10,0])
    print(" Revised number of points for fit: ", sum(indx_ok), indx_ok, indx_list[:10])
X_raw = X.copy()

my_fit= None
if opts.fit_method == "quadratic":
    print(" FIT METHOD ", opts.fit_method, " IS QUADRATIC")
    X=X[indx_ok]
    Y=Y[indx_ok]
    Y_err = Y_err[indx_ok]
    my_fit = fit_quadratic_alt(X,Y,symmetry_list=symmetry_list,verbose=opts.verbose)
elif opts.fit_method == "polynomial":
    print(" FIT METHOD ", opts.fit_method, " IS POLYNOMIAL")
    X=X[indx_ok]
    Y=Y[indx_ok]
    Y_err = Y_err[indx_ok]
    my_fit = fit_polynomial(X,Y,symmetry_list=symmetry_list,y_errors=Y_err)
elif opts.fit_method == 'gp_hyper':
    print(" FIT METHOD ", opts.fit_method, " IS GP with hypercube rescaling")
    # some data truncation IS used for the GP, but beware
    print(" Truncating data set used for GP, to reduce memory usage needed in matrix operations")
    X=X[indx_ok]
    Y=Y[indx_ok]
    Y_err = Y_err[indx_ok]
    my_fit = fit_gp(X,Y,y_errors=Y_err,hypercube_rescale=True)
elif opts.fit_method == 'gp':
    print(" FIT METHOD ", opts.fit_method, " IS GP")
    # some data truncation IS used for the GP, but beware
    print(" Truncating data set used for GP, to reduce memory usage needed in matrix operations")
    X=X[indx_ok]
    Y=Y[indx_ok]
    Y_err = Y_err[indx_ok]
    my_fit = fit_gp(X,Y,y_errors=Y_err)

# Sort for later convenience (scatterplots, etc)
indx = Y.argsort()#[::-1]
X=X[indx]
Y=Y[indx]






###
### Coordinate conversion tool
###
def convert_coords(x_in):
    return lalsimutils.convert_waveform_coordinates(x_in, coord_names=coord_names,low_level_coord_names=low_level_coord_names)



likelihood_function = None
if len(low_level_coord_names) ==1:
    def likelihood_function(x):  
        if isinstance(x,float):
            return np.exp(my_fit([x]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x]).T)  ))
if len(low_level_coord_names) ==2:
    def likelihood_function(x,y):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y]).T)))
if len(low_level_coord_names) ==3:
    def likelihood_function(x,y,z):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y,z]).T)))
if len(low_level_coord_names) ==4:
    def likelihood_function(x,y,z,a):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y,z,a]).T)))
if len(low_level_coord_names) ==5:
    def likelihood_function(x,y,z,a,b):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b]).T)))
if len(low_level_coord_names) ==6:
    def likelihood_function(x,y,z,a,b,c):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b,c]).T)))
if len(low_level_coord_names) ==7:
    def likelihood_function(x,y,z,a,b,c,d):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b,c,d]).T)))
if len(low_level_coord_names) ==8:
    def likelihood_function(x,y,z,a,b,c,d,e):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b,c,d,e]).T)))
if len(low_level_coord_names) ==9:
    def likelihood_function(x,y,z,a,b,c,d,e,f):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e,f]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b,c,d,e,f]).T)))
if len(low_level_coord_names) ==10:
    def likelihood_function(x,y,z,a,b,c,d,e,f,g):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e,f,g]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b,c,d,e,f,g]).T)))




# PROCEDURE
#   - Identify grid of desired parameters (e.g., ascii table) 
#   - Create

# Base
P_base = lalsimutils.xml_to_ChooseWaveformParams_array(opts.fname_xml_base)[0]

# Grid
samples_rec = np.genfromtxt(opts.fname_parameter_grid,names=True)
params_rec = samples_rec.dtype.names

# Conversion
P_list  =[]
grid_list = []
lnL_list = []
for indx in np.arange(len(samples_rec[params_rec[0]])):
    P = P_base.manual_copy()
    for param in params_rec:
        val = samples_rec[param][indx]
        fac=1
        if param in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        P.assign_param(param,fac*val)

    if opts.verbose:
        P.print_params()
    line_out = np.zeros(len(coord_names))
    for x in np.arange(len(line_out)):
        fac = 1
        if coord_names[x] in ['mc','m1','m2','mtot']:
            fac = lal.MSUN_SI
        line_out[x] = P.extract_param(coord_names[x])/fac

    # If opts.maximize_mass, we are reporting the likelihood maximized in total mass (all other parameters held fixed)
    # Remember, mc_index tells us the variable we need to scale
    arg=-1
    if (not opts.maximize_mass) or mc_index <0:
        arg = my_fit(line_out)[0]
    else:
        scalevec = np.ones(len(coord_names));
        def scaledfunc(x):
            scalevec[mc_index] = x
            val = -my_fit(line_out*scalevec)
            return -my_fit(line_out*scalevec)[0]
        res= scipy.optimize.minimize(scaledfunc,1,bounds=[(0.01,100)],options={'maxiter':50})  # unlikely to have mass range scale of a factor of 10^4
        arg = -scaledfunc(res.x)
    grid_list.append(line_out)
    lnL_list.append(arg)
    print(line_out, arg)


n_params = len(grid_list[0])
dat_out = np.zeros( (len(grid_list), n_params+1))
dat_out[:,:n_params] = np.array(grid_list)
dat_out[:,-1] = np.array(lnL_list)
np.savetxt(opts.fname_out, dat_out)
