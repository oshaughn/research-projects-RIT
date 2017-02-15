#!/usr/bin/env python
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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import corner

import BayesianLeastSquares

import argparse
import sys
import numpy as np
import scipy
import lalsimutils
import lalsimulation as lalsim
import lalframe
import lal
import functools
import itertools

from matplotlib import pyplot as plt


from sklearn.preprocessing import PolynomialFeatures
import ModifiedScikitFit as msf  # altenative polynomialFeatures
from sklearn import linear_model

from glue.ligolw import lsctables, utils, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)

import mcsampler




parser = argparse.ArgumentParser()
parser.add_argument("--fname",help="filename of *.dat file [standard ILE output]")
parser.add_argument("--fname-lalinference",help="filename of posterior_samples.dat file [standard LI output], to overlay on corner plots")
parser.add_argument("--fname-output-samples",default="output-ILE-samples",help="output posterior samples (default output-ILE-samples -> output-ILE)")
parser.add_argument("--fref",default=20,help="Reference frequency used for spins in the ILE output.  (Since I usually use SEOBNRv3, the best choice is 20Hz)")
parser.add_argument("--fname-rom-samples",default=None,help="*.rom_composite output. Treated identically to set of posterior samples produced by mcsampler after constructing fit.")
parser.add_argument("--n-output-samples",default=3000,type=int,help="output posterior samples (default 3000)")
parser.add_argument("--desc-lalinference",type=str,default='',help="String to adjoin to legends for LI")
parser.add_argument("--desc-ILE",type=str,default='',help="String to adjoin to legends for ILE")
parser.add_argument("--parameter", action='append', help="Parameters used as fitting parameters AND varied at a low level to make a posterior")
parser.add_argument("--parameter-implied", action='append', help="Parameter used in fit, but not independently varied for Monte Carlo")
parser.add_argument("--parameter-nofit", action='append', help="Parameter used to initialize the implied parameters, and varied at a low level, but NOT the fitting parameters")
parser.add_argument("--use-precessing",action='store_true')
parser.add_argument("--lnL-offset",type=float,default=10,help="lnL offset")
parser.add_argument("--lnL-cut",type=float,default=None,help="lnL cut [MANUAL]")
parser.add_argument("--M-max-cut",type=float,default=1e5,help="Maximum mass to consider (e.g., if there is a cut on distance, this matters)")
parser.add_argument("--sigma-cut",type=float,default=0.6,help="Eliminate points with large error from the fit.")
parser.add_argument("--lnL-peak-insane-cut",type=float,default=np.inf,help="Throw away lnL greater than this value. Should not be necessary")
parser.add_argument("--fmin",type=float,default=None)
parser.add_argument("--verbose", action="store_true",default=False, help="Required to build post-frame-generating sanity-test plots")
parser.add_argument("--save-plots",default=False,action='store_true', help="Write plots to file (only useful for OSX, where interactive is default")
parser.add_argument("--inj-file", help="Name of injection file")
parser.add_argument("--event-num", type=int, default=0,help="Zero index of event in inj_file")
parser.add_argument("--report-best-point",action='store_true')
parser.add_argument("--adapt",action='store_true')
parser.add_argument("--fit-uses-reported-error",action='store_true')
parser.add_argument("--n-max",default=3e5,type=float)
parser.add_argument("--n-eff",default=3e3,type=int)
parser.add_argument("--fit-method",default="quadratic",help="quadratic|polynomial|gp")
parser.add_argument("--fit-order",type=int,default=2,help="Fit order (polynomial case: degree)")
opts=  parser.parse_args()


if opts.fit_method == "quadratic":
    opts.fit_order = 2  # overrride

###
### Comparison data (from LI)
###
remap_ILE_2_LI = {"s1z":"a1z", "s2z":"a2z", "xi":"chi_eff", 
 "mc":"mc", "eta":"eta","m1":"m1","m2":"m2"}
if opts.fname_lalinference:
    print " Loading lalinference samples for direct comparison ", opts.fname_lalinference
    samples_LI = np.genfromtxt(opts.fname_lalinference,names=True)

    print " Checking LI samples have desired parameters "
    for p in opts.parameter:
        if p in remap_ILE_2_LI:
            print p , " -> ", remap_ILE_2_LI[p]
        else:
            print p, " NOT LISTED IN KEYS"



test_converged={}
#test_converged['neff'] = functools.partial(mcsampler.convergence_test_MostSignificantPoint,0.01)  # most significant point less than 1/neff of probability.  Exactly equivalent to usual neff threshold.
#test_converged["normal_integral"] = functools.partial(mcsampler.convergence_test_NormalSubIntegrals, 25, 0.01, 0.1)   # 20 sub-integrals are gaussian distributed [weakly; mainly to rule out outliers] *and* relative error < 10%, based on sub-integrals . Should use # of intervals << neff target from above.  Note this sets our target error tolerance on  lnLmarg.  Note the specific test requires >= 20 sub-intervals, which demands *very many* samples (each subintegral needs to be converged).


###
### Prior functions : a dictionary
###

# mcmin, mcmax : to be defined later
def M_prior(x):
    return x/(mc_max-mc_min)
def q_prior(x):
    return x/(1+x)**2  # not normalized
def m1_prior(x):
    return 1./200
def m2_prior(x):
    return 1./200
def s1z_prior(x):
    return 1./2
def s2z_prior(x):
    return 1./2
def mc_prior(x):
    return x/(mc_max-mc_min)
def eta_prior(x):
    return 1./np.power(x,6./5.)/np.power(1-4.*x, 0.5)/1.44

def xi_uniform_prior(x):
    return np.ones(x.shape)
def s_component_uniform_prior(x):
    return np.ones(x.shape)/2.

prior_map  = { "mtot": M_prior, "q":q_prior, "s1z":s1z_prior, "s2z":s2z_prior, "mc":mc_prior, "eta":eta_prior, 'xi':xi_uniform_prior,'chi_eff':xi_uniform_prior,'delta': (lambda x: 1./2),
    's1x':s_component_uniform_prior,
    's2x':s_component_uniform_prior,
    's1y':s_component_uniform_prior,
    's2y':s_component_uniform_prior,
}
prior_range_map = {"mtot": [1, 200], "q":[0.01,1], "s1z":[-0.99,0.99], "s2z":[-0.99,0.99], "mc":[0.9,90], "eta":[0.01,0.2499999], 'xi':[-1,1],'chi_eff':[-1,1],'delta':[-1,1],
   's1x':[-1,1],
   's2x':[-1,1],
   's1y':[-1,1],
   's2y':[-1,1]
}


# TeX dictionary
tex_dictionary = lalsimutils.tex_dictionary
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

def fit_quadratic_alt(x,y,x0=None,symmetry_list=None,verbose=False):
    the_quadratic_results = BayesianLeastSquares.fit_quadratic( x, y,verbose=verbose)#x0=None)#x0_val_here)
    peak_val_est, best_val_est, my_fisher_est, linear_term_est,fn_estimate = the_quadratic_results

    np.savetxt("lnL_peakval.dat",[peak_val_est])   # generally not very useful
    np.savetxt("lnL_bestpt.dat",best_val_est)  
    np.savetxt("lnL_gamma.dat",my_fisher_est,header=' '.join(coord_names))
        

    bic  =-2*( -0.5*np.sum(np.power((y - fn_estimate(x)),2))/2 - 0.5* len(y)*np.log(len(x[0])) )

    print "  Fit: std :" , np.std( y-fn_estimate(x))
    print "  Fit: BIC :" , bic

    return fn_estimate


# https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/preprocessing/data.py#L1139
def fit_polynomial(x,y,x0=None,symmetry_list=None):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

    clf_list = []
    bic_list = []
    for indx in np.arange(opts.fit_order+1):
        poly = msf.PolynomialFeatures(degree=indx,symmetry_list=symmetry_list)
        X_  = poly.fit_transform(x)

        if opts.verbose:
            print " Fit : poly: RAW :", poly.get_feature_names()
            print " Fit : ", poly.powers_

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
        clf.fit(X_,y)

        clf_list.append(clf)

        print  " Fit: Testing order ", indx
        print  " Fit: std: ", np.std(y - clf.predict(X_)),  "using number of features ", len(y)  # should NOT be perfect
        bic = -2*( -0.5*np.sum(np.power(y - clf.predict(X_),2))  - 0.5*len(y)*np.log(len(x[0])))
        print  " Fit: BIC:", bic
        bic_list.append(bic)

    clf = clf_list[np.argmin(np.array(bic_list) )]

    return lambda x: clf.predict(poly.fit_transform(x))


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def fit_gp(x,y,x0=None,symmetry_list=None):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

#    kernel = C([1.0,0.05],[ (1e-3, 1e2), (1e-3, 1)]) * RBF([1,0.05], [ (1e-3, 1e2), (1e-3, 1)])
    kernel = C(1, (1e-3,1e1))*RBF(1, (1e-3,1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    gp.fit(x,y)

    print  " Fit: std: ", np.std(y - gp.predict(x)),  "using number of features ", len(y)  # should NOT be perfect

    return lambda x: gp.predict(x)


coord_names = opts.parameter # Used  in fit
if opts.parameter_implied:
    coord_names = coord_names+opts.parameter_implied
low_level_coord_names = opts.parameter # Used for Monte Carlo
if opts.parameter_nofit:
    low_level_coord_names = opts.parameter+opts.parameter_nofit # Used for Monte Carlo
print " Coordinate names for fit :, ", coord_names
print " Rendering coordinate names : ", map(lambda x: tex_dictionary[x], coord_names)
print " Symmetry for these fitting coordinates :", lalsimutils.symmetry_sign_exchange(coord_names)
print " Coordinate names for Monte Carlo :, ", low_level_coord_names
print " Rendering coordinate names : ", map(lambda x: tex_dictionary[x], low_level_coord_names)

# initialize
dat_mass  = [] 
weights = []
n_params = -1

###
### Retrieve data
###
#  id m1 m2  lnL sigma/L  neff
col_lnL = 9
dat_orig = dat = np.loadtxt(opts.fname)
dat_orig = dat[dat[:,col_lnL].argsort()] # sort  http://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
print " Original data size = ", len(dat), dat.shape


 ###
 ### Convert data.  Use lalsimutils for flexibility
 ###
P_list = []
dat_out =[]
 

symmetry_list =lalsimutils.symmetry_sign_exchange(coord_names)  # identify symmetry due to exchange
mc_min = 1e10
mc_max = -1

P= lalsimutils.ChooseWaveformParams()
for line in dat:
  # Skip precessing binaries unless explicitly requested not to!
  if not opts.use_precessing and (line[3]**2 + line[4]**2 + line[6]**2 + line[7]**2)>0.01:
      continue
  if line[1]+line[2] > opts.M_max_cut:
      print " Skipping ", line, " as too massive, with mass ", line[1]+line[2]
      continue
  if line[10] > opts.sigma_cut:
      print " Skipping ", line
      continue
  if line[col_lnL] < opts.lnL_peak_insane_cut:
    P.fref = opts.fref  # IMPORTANT if you are using a quantity that depends on J
    P.m1 = line[1]*lal.MSUN_SI
    P.m2 = line[2]*lal.MSUN_SI
    P.s1x = line[3]
    P.s1y = line[4]
    P.s1z = line[5]
    P.s2x = line[6]
    P.s2y = line[7]
    P.s2z = line[8]
 #    print line,  P.extract_param('xi')
    line_out = np.zeros(len(coord_names)+1)
    for x in np.arange(len(coord_names)):
        line_out[x] = P.extract_param(coord_names[x])
 #        line_out[x] = getattr(P, coord_names[x])
    line_out[-1] = line[col_lnL]
    dat_out.append(line_out)

    # Update mc range
    mc_here = lalsimutils.mchirp(line[1],line[2])
    if mc_here < mc_min:
        mc_min = mc_here
    if mc_here > mc_max:
        mc_max = mc_here

dat_out = np.array(dat_out)
 # scale out mass units
for p in ['mc', 'm1', 'm2', 'mtot']:
    if p in coord_names:
        indx = coord_names.index(p)
        dat_out[:,indx] /= lal.MSUN_SI


# Repack data
X =dat_out[:,0:len(coord_names)]
Y = dat_out[:,-1]

# Eliminate values with Y too small
max_lnL = np.max(Y)
indx_ok = Y>np.max(Y)-opts.lnL_offset
print " Points used in fit : ", sum(indx_ok), " given max lnL ", max_lnL
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
    print " Revised number of points for fit: ", sum(indx_ok), indx_ok, indx_list[:10]
X_raw = X.copy()

my_fit= None
if opts.fit_method == "quadratic":
    X=X[indx_ok]
    Y=Y[indx_ok]
    if opts.report_best_point:
        my_fit = fit_quadratic_alt(X,Y,symmetry_list=symmetry_list)
        pt_best_X = np.loadtxt("lnL_bestpt.dat")
        for indx in np.arange(len(coord_names)):
            fac = 1
            if coord_names[indx] in ['mc','m1','m2','mtot']:
                fac = lal.MSUN_SI
            p_to_assign = coord_names[indx]
            if p_to_assign == 'xi':
                p_to_assign = "chieff_aligned"
            P.assign_param(p_to_assign,pt_best_X[indx]*fac) 
           
        print " ====BEST BINARY ===="
        print " Parameters from fit ", pt_best_X
        P.print_params()
        sys.exit(0)
    my_fit = fit_quadratic_alt(X,Y,symmetry_list=symmetry_list,verbose=opts.verbose)
elif opts.fit_method == "polynomial":
    X=X[indx_ok]
    Y=Y[indx_ok]
    my_fit = fit_polynomial(X,Y,symmetry_list=symmetry_list)
else:
    my_fit = fit_gp(X,Y)

# Sort for later convenience (scatterplots, etc)
indx = Y.argsort()#[::-1]
X=X[indx]
Y=Y[indx]


# Make grid plots for all pairs of points, to facilitate direct validation of where posterior support lies
import itertools
for i, j in itertools.product( np.arange(len(coord_names)),np.arange(len(coord_names)) ):
  if i < j:
    plt.scatter( X[:,i],X[:,j],label='rapid_pe:'+opts.desc_ILE,c=Y); plt.legend(); plt.colorbar()
    plt.xlabel( tex_dictionary[coord_names[i]])
    plt.ylabel( tex_dictionary[coord_names[j]])
    plt.title("rapid_pe evaluations (=inputs); no fits")
    plt.savefig("scatter_"+coord_names[i]+"_"+coord_names[j]+".png"); plt.clf()




###
### Coordinate conversion tool
###
def convert_coords(x_in):
    return lalsimutils.convert_waveform_coordinates(x_in, coord_names=coord_names,low_level_coord_names=low_level_coord_names)


###
### Integrate posterior
###


sampler = mcsampler.MCSampler()


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

    sampler.add_parameter(p, pdf=np.vectorize(lambda x:1), prior_pdf=prior_here,left_limit=range_here[0],right_limit=range_here[1],adaptive_sampling=True)

likelihood_function = None
if len(low_level_coord_names) ==1:
    def likelihood_function(x):  
        if isinstance(x,float):
            return np.exp(my_fit([x]))
        else:
            return np.exp(my_fit(convert_coords(np.array([x]).T)))
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


res, var, neff, dict_return = sampler.integrate(likelihood_function, *low_level_coord_names,  verbose=True,nmax=int(opts.n_max),n=1e5,save_intg=True,tempering_adapt=True, floor_level=1e-3,igrand_threshold_p=1e-3,convergence_tests=test_converged,adapt_weight_exponent=0.1,no_protect_names=True)  # weight ecponent needs better choice. We are using arbitrary-name functions



samples = sampler._rvs
print samples.keys()
n_params = len(coord_names)
dat_mass = np.zeros((len(samples[coord_names[0]]),n_params+3))
dat_logL = np.log(samples["integrand"])
print " Max lnL ", np.max(dat_logL)

# Throw away stupid points that don't impact the posterior
indx_ok = np.logical_and(dat_logL > np.max(dat_logL)-opts.lnL_offset ,samples["joint_s_prior"]>0)
for p in low_level_coord_names:
    samples[p] = samples[p][indx_ok]
dat_logL  = dat_logL[indx_ok]
print samples.keys()
samples["joint_prior"] =samples["joint_prior"][indx_ok]
samples["joint_s_prior"] =samples["joint_s_prior"][indx_ok]



###
### 1d posteriors of the coordinates used for sampling  [EQUALLY WEIGHTED]
###

p = samples["joint_prior"]
ps =samples["joint_s_prior"]
lnL = dat_logL
lnLmax = np.max(lnL)
weights = np.exp(lnL-lnLmax)*p/ps

# Load in reference parameters
Pref = lalsimutils.ChooseWaveformParams()
if  opts.inj_file is not None:
    Pref = lalsimutils.xml_to_ChooseWaveformParams_array(opts.inj_file)[opts.event_num]
Pref.print_params()

for indx in np.arange(len(low_level_coord_names)):
    dat_out = []; dat_out_LI=[]
    p = low_level_coord_names[indx]
    print " -- 1d cumulative "+ str(indx)+ ":"+ low_level_coord_names[indx]+" ----"
    dat_here = samples[low_level_coord_names[indx]]
    for x in np.linspace(np.min(dat_here),np.max(dat_here),200):
         dat_out.append([x, np.sum( weights[ dat_here< x])/np.sum(weights)])
         if opts.fname_lalinference and (p in remap_ILE_2_LI.keys()):
             dat_out_LI.append([x, (1.0*np.sum( samples_LI[ remap_ILE_2_LI[p] ]< x))/len(samples_LI) ])
    np.savetxt(p+"_cdf.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    plt.plot(dat_out[:,0],dat_out[:,1],label="rapid_pe:"+opts.desc_ILE,color='b')
    if opts.fname_lalinference and  (p in remap_ILE_2_LI.keys()):
        plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],label="LI:"+opts.desc_lalinference,color='r')

    # Add vertical line
    here_val = Pref.extract_param(p)
    fac = 1
    if p in ['mc','m1','m2','mtot']:
        fac = lal.MSUN_SI
    here_val = here_val/fac
    print " Vertical line ", p, " ", here_val
    plt.axvline(here_val,color='k',linestyle='--')


    plt.xlabel(tex_dictionary[p]); plt.legend()
    plt.savefig(p+"_cdf.png"); plt.clf()


###
### 1d posteriors of the coordinates used for sampling
###




###
### Corner
###


print " ---- Corner 1: Sampling coordinates ---- "
dat_mass = np.zeros( (len(lnL),len(low_level_coord_names)),dtype=np.float64)
dat_mass_LI = []
if opts.fname_lalinference:
    dat_mass_LI = np.zeros( (len(samples_LI), len(low_level_coord_names)), dtype=np.float64)
for indx in np.arange(len(low_level_coord_names)):
    dat_mass[:,indx] = samples[low_level_coord_names[indx]]
    if opts.fname_lalinference and low_level_coord_names[indx] in remap_ILE_2_LI.keys():
        dat_mass_LI[:,indx] = samples_LI[ remap_ILE_2_LI[low_level_coord_names[indx]] ]

CIs = [0.95,0.9, 0.68]
quantiles_1d = [0.05,0.95]
range_here = []
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
    if (range_here[-1][1] < np.mean(samples[p])+5*np.std(samples[p])  ):
         range_here[-1][1] = np.mean(samples[p])+5*np.std(samples[p])
    if range_here[-1][0] < prior_range_map[p][0]:
        range_here[-1][0] = prior_range_map[p][0]
    if range_here[-1][1] < prior_range_map[p][1]:
        range_here[-1][1] = prior_range_map[p][1]
    print p, range_here[-1]  # print out range to be used in plots.

labels_tex = map(lambda x: tex_dictionary[x], low_level_coord_names)
fig_base = corner.corner(dat_mass[:,:len(low_level_coord_names)], weights=(weights/np.sum(weights)).astype(np.float64),labels=labels_tex, quantiles=quantiles_1d,plot_datapoints=False,plot_density=False,no_fill_contours=True,fill_contours=False,levels=CIs,range=range_here)

# Plot simulation points (X array): MAY NOT BE POSSIBLE if dimensionality is inconsistent
#fig_base = corner.corner(X,plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base,weights = 1*np.ones(len(X))/len(X), data_kwargs={'color':'g'},hist_kwargs={'color':'g', 'linestyle':'--'},range_here=range_here)

if opts.fname_lalinference:
    corner.corner( dat_mass_LI,color='r',labels=labels_tex,weights=np.ones(len(dat_mass_LI))*1.0/len(dat_mass_LI),fig=fig_base,quantiles=quantiles_1d,no_fill_contours=True,plot_datapoints=False,plot_density=False,fill_contours=False,levels=CIs,range=range_here)


plt.savefig("posterior_corner.png"); plt.clf()

print " ---- Subset for posterior samples (and further corner work) --- " 

# pick random numbers
p_thresholds =  np.random.uniform(low=0.0,high=1.0,size=opts.n_output_samples)
# find sample indexes associated with the random numbers
#    - FIXME: first truncate the bad ones
# idx_sorted_index = numpy.lexsort((numpy.arange(len(weights)), weights))  # Sort the array of weights, recovering index values
# indx_list = numpy.array( [[k, weights[k]] for k in idx_sorted_index])     # pair up with the weights again
# cum_weights = np.cumsum(indx_list[:,1)
# cum_weights = cum_weights/cum_weights[-1]
# indx_list = [indx_list[k, 0] for k, value in enumerate(cum_sum > deltaP) if value]  # find the indices that preserve > 1e-7 of total probabilit
cum_sum  = np.cumsum(weights)
cum_sum = cum_sum/cum_sum[-1]
indx_list = map(lambda x : np.sum(cum_sum < x),  p_thresholds)  # this can lead to duplicates
P_list =[]
P = lalsimutils.ChooseWaveformParams()
P.approx = lalsim.SEOBNRv2  # DEFAULT
P.fmin = opts.fref # DEFAULT
P.fref = opts.fref
for indx_here in indx_list:
        line = [samples[p][indx_here] for p in low_level_coord_names]
        Pgrid = P.manual_copy()
        include_item =True
        # Set attributes that are being changed as necessary, leaving all others fixed
        for indx in np.arange(len(low_level_coord_names)):
#            if param_names[indx] in downselect_dict:
#                if line[indx] < downselect_dict[param_names[indx]][0] or line[indx] > downselect_dict[param_names[indx]][1]:
#                    include_item = False
#                    if opts.verbose:
#                        print " Skipping " , line
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
            
        if opts.verbose:
            Pgrid.print_params()
        # Downselect.
        # for param in downselect_dict:
        #     if Pgrid.extract_param(param) < downselect_dict[param][0] or Pgrid.extract_param(param) > downselect_dict[param][1]:
        #         print " Skipping " , line
        #         include_item =False
        if include_item:
         if Pgrid.m2 <= Pgrid.m1:  # do not add grid elements with m2> m1, to avoid possible code pathologies !
            P_list.append(Pgrid)
         else:
            Pgrid.swap_components()  # IMPORTANT.  This should NOT change the physical functionality FOR THE PURPOSES OF OVERLAP (but will for PE - beware phiref, etc!)
            P_list.append(Pgrid)
        else:
            True

###
### Extract data from samples, in array form
###
dat_mass_post = np.zeros( (len(P_list),len(coord_names)),dtype=np.float64)
for indx_line  in np.arange(len(P_list)):
    for indx in np.arange(len(coord_names)):
        fac=1
        if coord_names[indx] in ['mc', 'mtot', 'm1', 'm2']:
                fac = lal.MSUN_SI
        dat_mass_post[indx_line,indx] = P_list[indx_line].extract_param(coord_names[indx])/fac




range_here=[]
for indx in np.arange(len(coord_names)):    
    range_here.append( [np.min(dat_mass_post[:, indx]),np.max(dat_mass_post[:, indx])])
    # Manually reset some ranges to be more useful for plotting
    if coord_names[indx] in ['xi', 'chi_eff']:
        range_here[-1] = [-1,1]
    if coord_names[indx] in ['eta']:
        range_here[-1] = [0,0.25]
    if coord_names[indx] in ['q']:
        range_here[-1] = [0,1]
    print coord_names[indx], range_here[-1]

if opts.fname_lalinference:
    dat_mass_LI = np.zeros( (len(samples_LI), len(coord_names)), dtype=np.float64)
    for indx in np.arange(len(coord_names)):
        if coord_names[indx] in remap_ILE_2_LI.keys():
            dat_mass_LI[:,indx] = samples_LI[ remap_ILE_2_LI[coord_names[indx]] ]

        if range_here[indx][0] > np.min(dat_mass_LI[:,indx]):
            range_here[indx][0] = np.min(dat_mass_LI[:,indx])
        if range_here[indx][1] < np.max(dat_mass_LI[:,indx]):
            range_here[indx][1] = np.max(dat_mass_LI[:,indx])

print " ---- 1d cumulative on fitting coordinates --- " 
for indx in np.arange(len(coord_names)):
    p = coord_names[indx]
    dat_out = []; dat_out_LI=[]
    print " -- 1d cumulative "+ str(indx)+ ":"+ coord_names[indx]+" ----"
    dat_here = dat_mass_post[:,indx]
    wt_here = np.ones(len(dat_here))
    for x in np.linspace(np.min(dat_here),np.max(dat_here),200):
         dat_out.append([x, np.sum(  wt_here[dat_here< x] )/len(dat_here)])    # NO WEIGHTS for these resampled points
#         dat_out.append([x, np.sum( weights[ dat_here< x])/np.sum(weights)])
         if opts.fname_lalinference and (p in remap_ILE_2_LI.keys()) :
             dat_out_LI.append([x, (1.0*np.sum( samples_LI[ remap_ILE_2_LI[p] ]< x))/len(samples_LI) ])
    np.savetxt(p+"_alt_cdf.dat", np.array(dat_out))
    dat_out = np.array(dat_out); dat_out_LI=np.array(dat_out_LI)
    plt.plot(dat_out[:,0],dat_out[:,1],label="rapid_pe:"+opts.desc_ILE,color='b')
    if opts.fname_lalinference and (p in remap_ILE_2_LI.keys()) :
        plt.plot(dat_out_LI[:,0],dat_out_LI[:,1],label="LI:"+opts.desc_lalinference,color='r')

    # Add vertical line
    here_val = Pref.extract_param(p)
    fac = 1
    if p in ['mc','m1','m2','mtot']:
        fac = lal.MSUN_SI
    here_val = here_val/fac
    print " Vertical line ", p, " ", here_val
    plt.axvline(here_val,color='k',linestyle='--')


    plt.xlabel(tex_dictionary[p]); plt.legend()
    plt.savefig(p+"_alt_cdf.png"); plt.clf()


print " ---- Corner 2: Fitting coordinates (+ original sample point overlay) ---- "

###
### Corner plot.  Also overlay sample points
###

labels_tex = map(lambda x: tex_dictionary[x], coord_names)
fig_base = corner.corner(dat_mass_post[:,:len(coord_names)], weights=np.ones(len(dat_mass_post))*1.0/len(dat_mass_post),labels=labels_tex, quantiles=quantiles_1d,plot_datapoints=False,plot_density=False,no_fill_contours=True,fill_contours=False,levels=CIs, range=range_here)

if opts.fname_lalinference:
    fig_base=corner.corner( dat_mass_LI,color='r',labels=labels_tex,weights=np.ones(len(dat_mass_LI))*1.0/len(dat_mass_LI),fig=fig_base,quantiles=quantiles_1d,no_fill_contours=True,plot_datapoints=False,plot_density=False,fill_contours=False,levels=CIs,range=range_here)


fig_base = corner.corner(X,plot_datapoints=True,plot_density=False,plot_contours=False,quantiles=None,fig=fig_base,weights = 1*np.ones(len(X))/len(X), data_kwargs={'color':'g'},hist_kwargs={'color':'g', 'linestyle':'--'},range=range_here)


plt.savefig("posterior_corner_fit_coords.png"); plt.clf()




sys.exit(0)


