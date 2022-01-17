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
#
# EXAMPLE:
#   python `which util_ConstructEOSPosterior.py` --fname fake_int_grid.dat  --parameter gamma1 --parameter gamma2 --lnL-offset 50

import RIFT.interpolators.BayesianLeastSquares as BayesianLeastSquares

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import scipy.stats
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



def render_coord(x):
    if x in lalsimutils.tex_dictionary.keys():
        return lalsimutils.tex_dictionary[x]
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
parser.add_argument("--fname",help="filename of *.dat file [EOS table: int_res gamma1 gamma2 ...]")
parser.add_argument("--fname-output-samples",default="output-EOS-samples",help="output posterior samples (default output-ILE-samples -> output-ILE)")
parser.add_argument("--n-output-samples",default=2000,type=int,help="output posterior samples (default 3000)")
parser.add_argument("--eos-param", type=str, default=None, help="parameterization of equation of state [spectral only, for now]")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior. Currently can only specify gamma1,gamma2, ..., and these MUST be columns in --fname")
parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
#parser.add_argument("--no-adapt-parameter",action='append',help="Disable adaptive sampling in a parameter. Useful in cases where a parameter is not well-constrained, and the a prior sampler is well-chosen.")
parser.add_argument("--parameter-nofit", action='append', help="Parameter used to initialize the implied parameters, and varied at a low level, but NOT the fitting parameters")
parser.add_argument("--trust-sample-parameter-box",action='store_true', help="If used, sets the prior range to the SAMPLE range for any parameters. NOT IMPLEMENTED. This should be automatically done for mc!")
parser.add_argument("--plots-do-not-force-large-range",action='store_true', help = "If used, the plots do NOT automatically set the chieff range to [-1,1], the eta range to [0,1/4], etc")
parser.add_argument("--downselect-parameter",action='append', help='Name of parameter to be used to eliminate grid points ')
parser.add_argument("--downselect-parameter-range",action='append',type=str)
parser.add_argument("--no-downselect",action='store_true')
parser.add_argument("--aligned-prior", default="uniform",help="Options are 'uniform', 'volumetric', and 'alignedspin-zprior'")
parser.add_argument("--cap-points",default=-1,type=int,help="Maximum number of points in the sample, if positive. Useful to cap the number of points ued for GP. See also lnLoffset. Note points are selected AT RANDOM")
parser.add_argument("--lambda-max", default=4000,type=float,help="Maximum range of 'Lambda' allowed.  Minimum value is ZERO, not negative.")
parser.add_argument("--lnL-shift-prevent-overflow",default=None,type=float,help="Define this quantity to be a large positive number to avoid overflows. Note that we do *not* define this dynamically based on sample values, to insure reproducibility and comparable integral results. BEWARE: If you shift the result to be below zero, because the GP relaxes to 0, you will get crazy answers.")
parser.add_argument("--lnL-offset",type=float,default=10,help="lnL offset")
parser.add_argument("--lnL-cut",type=float,default=None,help="lnL cut [MANUAL]")
parser.add_argument("--sigma-cut",type=float,default=0.6,help="Eliminate points with large error from the fit.")
parser.add_argument("--ignore-errors-in-data",action='store_true',help='Ignore reported error in lnL. Helpful for testing purposes (i.e., if the error is zero)')
parser.add_argument("--lnL-peak-insane-cut",type=float,default=np.inf,help="Throw away lnL greater than this value. Should not be necessary")
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
parser.add_argument("--n-max",default=3e5,type=float)
parser.add_argument("--n-eff",default=3e3,type=int)
parser.add_argument("--pool-size",default=3,type=int,help="Integer. Number of GPs to use (result is averaged)")
parser.add_argument("--fit-load-gp",default=None,type=str,help="Filename of GP fit to load. Overrides fitting process, but user MUST correctly specify coordinate system to interpret the fit with.  Does not override loading and converting the data.")
parser.add_argument("--fit-save-gp",default=None,type=str,help="Filename of GP fit to save. ")
parser.add_argument("--fit-order",type=int,default=2,help="Fit order (polynomial case: degree)")
parser.add_argument("--no-plots",action='store_true')
parser.add_argument("--using-eos", type=str, default=None, help="Name of EOS if not already determined in lnL")
opts=  parser.parse_args()
no_plots = no_plots |  opts.no_plots
lnL_shift = 0
if opts.lnL_shift_prevent_overflow:
    lnL_shift  = opts.lnL_shift_prevent_overflow



with open('args.txt','w') as fp:
    import sys
    fp.write(' '.join(sys.argv))

###
### Comparison data (from LI)
###

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

if opts.no_downselect:
    downselect_dict={}


test_converged={}

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
# TeX dictionary
#tex_dictionary = lalsimutils.tex_dictionary
print(" Coordinate names for fit :, ", coord_names)
print(" Coordinate names for Monte Carlo :, ", low_level_coord_names)
#print " Rendering coordinate names : ", map(lambda x: tex_dictionary[x], low_level_coord_names)


###
### Prior functions : a dictionary
###

def eos_param_uniform_prior(x):
    return np.ones(x.shape)


prior_map  = { 'gamma1':eos_param_uniform_prior, 'gamma2':eos_param_uniform_prior,
}
# Les: somewhat more aggressive: 
#    gamma1: 0.2,2
#    gamma2: -1.67, 1.7
prior_range_map = { 'gamma1':  [0.707899,1.31], 'gamma2':[-1.6,1.7], 'gamma3':[-0.6,0.6], 'gamma4':[-0.02,0.02]
}


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
        
        return lambda x: gp.predict(x)
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




# initialize
dat_mass  = [] 
weights = []
n_params = -1

###
### Retrieve data
###
#  int_sig sigma/L gamma1 gamma2 ...
col_lnL = 0
dat_orig = dat = np.loadtxt(opts.fname)
dat_orig = dat[dat[:,col_lnL].argsort()] # sort  http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
print(" Original data size = ", len(dat), dat.shape)

 ###
 ### Convert data.  
 ###

dat_out = []
for line in dat:
  dat_here= np.zeros(len(coord_names)+2)
  if line[col_lnL+1] > opts.sigma_cut:
      print("skipping", line)
      continue
  dat_here[:-2] = line[2:len(coord_names)+2]  # modify to use names!
  dat_here[-2] = line[0]
  dat_here[-1] = line[1]
  dat_out.append(dat_here)
dat_out= np.array(dat_out)
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
    indx_ok = map(int,indx_list[:10,0])
    print(" Revised number of points for fit: ", sum(indx_ok), indx_ok, indx_list[:10])
X_raw = X.copy()

my_fit= None
if True:
    print(" FIT METHOD : GP")
    # some data truncation IS used for the GP, but beware
    print(" Truncating data set used for GP, to reduce memory usage needed in matrix operations")
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
    my_fit = fit_gp(X,Y,y_errors=Y_err)


# Sort for later convenience (scatterplots, etc)
indx = Y.argsort()#[::-1]
X=X[indx]
Y=Y[indx]

# Make grid plots for all pairs of points, to facilitate direct validation of where posterior support lies
if not no_plots:
 import itertools
 for i, j in itertools.product( np.arange(len(coord_names)),np.arange(len(coord_names)) ):
  if i < j:
    plt.scatter( X[:,i],X[:,j],label='rapid_pe',c=Y); plt.legend(); plt.colorbar()
    x_name = str(coord_names[i])
    y_name = str(coord_names[j])
    plt.xlabel( x_name)
    plt.ylabel( y_name )
    plt.title("rapid_pe evaluations (=inputs); no fits")
    plt.savefig("scatter_"+coord_names[i]+"_"+coord_names[j]+".png"); plt.clf()



###
### Integrate posterior
###


sampler = mcsampler.MCSampler()


##
## Loop over param names
##
for p in coord_names:
    prior_here = prior_map[p]
    range_here = prior_range_map[p]

    sampler.add_parameter(p, pdf=np.vectorize(lambda x:1), prior_pdf=prior_here,left_limit=range_here[0],right_limit=range_here[1],adaptive_sampling=True)

likelihood_function = None
def convert_coords(x):
    return x
if len(coord_names) ==1:
    def likelihood_function(x):  
        if isinstance(x,float):
            return np.exp(my_fit([x]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x],dtype=internal_dtype).T) ))
            return np.exp(my_fit(convert_coords(np.c_[x])))
if len(coord_names) ==2:
    def likelihood_function(x,y):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y])))
if len(coord_names) ==3:
    def likelihood_function(x,y,z):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z])))
if len(coord_names) ==4:
    def likelihood_function(x,y,z,a):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z,a],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a])))
if len(coord_names) ==5:
    def likelihood_function(x,y,z,a,b):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b])))
if len(coord_names) ==6:
    def likelihood_function(x,y,z,a,b,c):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c]))
        else:
#            return np.exp(my_fit(convert_coords(np.array([x,y,z,a,b,c],dtype=internal_dtype).T)))
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c])))
if len(coord_names) ==7:
    def likelihood_function(x,y,z,a,b,c,d):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d])))
if len(coord_names) ==8:
    def likelihood_function(x,y,z,a,b,c,d,e):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e])))
if len(coord_names) ==9:
    def likelihood_function(x,y,z,a,b,c,d,e,f):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e,f]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e,f])))
if len(coord_names) ==10:
    def likelihood_function(x,y,z,a,b,c,d,e,f,g):  
        if isinstance(x,float):
            return np.exp(my_fit([x,y,z,a,b,c,d,e,f,g]))
        else:
            return np.exp(my_fit(convert_coords(np.c_[x,y,z,a,b,c,d,e,f,g])))


n_step = 1e5
my_exp = np.min([1,0.8*np.log(n_step)/np.max(Y)])   # target value : scale to slightly sublinear to (n_step)^(0.8) for Ymax = 200. This means we have ~ n_step points, with peak value wt~ n_step^(0.8)/n_step ~ 1/n_step^(0.2), limiting contrast
#my_exp = np.max([my_exp,  1/np.log(n_step)]) # do not allow extreme contrast in adaptivity, to the point that one iteration will dominate
print(" Weight exponent ", my_exp, " and peak contrast (exp)*lnL = ", my_exp*np.max(Y), "; exp(ditto) =  ", np.exp(my_exp*np.max(Y)), " which should ideally be no larger than of order the number of trials in each epoch, to insure reweighting doesn't select a single preferred bin too strongly.  Note also the floor exponent also constrains the peak, de-facto")

res, var, neff, dict_return = sampler.integrate(likelihood_function, *coord_names,  verbose=True,nmax=int(opts.n_max),n=n_step,neff=opts.n_eff, save_intg=True,tempering_adapt=True, floor_level=1e-3,igrand_threshold_p=1e-3,convergence_tests=test_converged,adapt_weight_exponent=my_exp,no_protect_names=True)  # weight ecponent needs better choice. We are using arbitrary-name functions


# Save result -- needed for odds ratios, etc.
np.savetxt("integral_result.dat", [np.log(res)])

if neff < len(coord_names):
    print(" PLOTS WILL FAIL ")
    print(" Not enough independent Monte Carlo points to generate useful contours")




samples = sampler._rvs
print(samples.keys())
n_params = len(coord_names)
dat_mass = np.zeros((len(samples[coord_names[0]]),n_params+3))
dat_logL = np.log(samples["integrand"])
print(" Max lnL ", np.max(dat_logL))

# Throw away stupid points that don't impact the posterior
indx_ok = np.logical_and(dat_logL > np.max(dat_logL)-opts.lnL_offset ,samples["joint_s_prior"]>0)
for p in coord_names:
    samples[p] = samples[p][indx_ok]
dat_logL  = dat_logL[indx_ok]
print(samples.keys())
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




if not no_plots:
 for indx in np.arange(len(coord_names)):
   if True:
    dat_out = []; dat_out_LI=[]
    p = coord_names[indx]
    print(" -- 1d cumulative "+ str(indx)+ ":"+ coord_names[indx]+" ----")
    dat_here = samples[coord_names[indx]]
    range_x = [np.min(dat_here), np.max(dat_here)]
    for x in np.linspace(range_x[0],range_x[1],200):
         dat_out.append([x, np.sum( weights[ dat_here< x])/np.sum(weights)])
    
    np.savetxt(p+"_cdf_nocut_beware.dat", np.array(dat_out))
    dat_out = np.array(dat_out); 
    plt.plot(dat_out[:,0],dat_out[:,1],label="rapid_pe")
   
    x_name = p
    plt.xlabel(x_name); plt.legend()
    y_name  = x_name.replace('$','')
    y_name = "$P(<"+y_name + ")$"
    plt.ylabel(y_name)
    plt.title("CDF: "+x_name)
    plt.savefig(p+"_cdf_nocut_beware.png"); plt.clf()
   else:
      plt.clf()  # clear plot, just in case
      print(" No 1d plot for variable")



print(" ---- Subset for posterior samples (and further corner work) --- ")

# pick random numbers
p_thresholds =  np.random.uniform(low=0.0,high=1.0,size=opts.n_output_samples)
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
indx_list = np.array(map(lambda x : np.sum(cum_sum < x),  p_thresholds))  # this can lead to duplicates

dat_out = np.zeros( (len(p_thresholds),len(coord_names)))
for indx in np.arange(len(coord_names)):
    dat_out[:,indx] = samples[coord_names[indx]][indx_list]

print(" Saving to ", opts.fname_output_samples+".dat")
np.savetxt(opts.fname_output_samples+".dat",dat_out,header=' '.join(coord_names))
