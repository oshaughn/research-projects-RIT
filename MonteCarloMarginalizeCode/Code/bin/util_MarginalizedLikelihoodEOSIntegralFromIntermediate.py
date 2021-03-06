#! /usr/bin/env python
#
# GOAL
#   - compute I(gamma), based on intermediate calculations of G(eta, LambdaTilde)
#     [User may want to smooth/validate intermediate integral before using it!]
#   - relies on 
#      - calculating EOS
#      - 
#
# ARGUMENTS
#   - redshift (assumed known)
#   - redshifted chirp mass
#   - EOS parameters 
#
#
# EXAMPLE
#   - python `which util_MarginalizedLikelihoodEOSIntegralFromIntermediate.py`  --intermediate-file analyze_27/qt/intermediate_nospin.dat 
#   python `which util_MarginalizedLikelihoodEOSIntegralFromIntermediate.py`  --intermediate-file analyze_27/qt/intermediate_nospin.dat  --parameter gamma1 --parameter-value 0.707 --parameter gamma2 --parameter-value 0.707 --mcz 1.26940994223 --redshift `cat redshift_of_27.dat`


import RIFT.physics.EOSManager as EOSManager

import argparse
import sys
import numpy as np
import numpy.lib.recfunctions
import scipy
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import functools
import itertools

import os



from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


def fit_gp(x,y,x0=None,symmetry_list=None,y_errors=None,hypercube_rescale=False,fname_export="gp_fit"):
    """
    x = array so x[0] , x[1], x[2] are points.
    """

    # If we are loading a fit, override everything else
    if opts.fit_save_gp and os.path.isfile(opts.fit_save_gp):
        print(" WARNING: Do not re-use fits across architectures or versions : pickling is not transferrable ")
        my_gp=joblib.load(opts.fit_save_gp)
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

        # These parameters have been hand-tuned by experience to try to set to levels comparable to typical lnL Monte Carlo error
    kernel = WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-2,1))+C(0.5, (1e-3,1e1))*RBF(length_scale=length_scale_est, length_scale_bounds=length_scale_bounds_est)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8)

    gp.fit(x,y)

    print(" Fit: std: ", np.std(y - gp.predict(x)),  "using number of features ", len(y))

    if opts.fit_save_gp:
        print(" Attempting to save fit ", opts.fit_save_gp)
        joblib.dump(gp,opts.fit_save_gp)
        
    return lambda x: gp.predict(x)


parser = argparse.ArgumentParser()
parser.add_argument("--intermediate-file", help="Name of intermediate evaluation grid")
parser.add_argument("--integral-output",default="integral_output.dat", help="Output file for this set of parameters")
parser.add_argument("--mcz", type=float,default=None,help="Redshifted chirp mass. MUST be specified ")
parser.add_argument("--redshift",default=0,type=float, help="Redshift, assumed known ")
parser.add_argument("--pkl-file", help="Fit.  If already present, can re-use a version of the fit generated by this code")
parser.add_argument("--fit-save-gp",default=None, help="pkl file to save to. If already present, use this fit. Please use postfix .pkl")
parser.add_argument("--verbose",action='store_true')
parser.add_argument("--parameter", action='append', help="Parameters used to construct the EOS. Assume spectral parameterization for now")
parser.add_argument("--parameter-value", type=float,action='append', help="Value of parameter")
opts=  parser.parse_args()


eos_params ={}
for indx  in np.arange(len(opts.parameter)):
    param = opts.parameter[indx]
    eos_params[param] = opts.parameter_value[indx]

if not opts.intermediate_file and not opts.pkl_file:
    print(" FAILURE: need input data")
    sys.exit(0)


###
### Compute source-frame chirp mass
###

if opts.mcz is None:
    print("FAILURE: Need redshifted chirp mass. (Should be able to retrieve from injection xml with detector-frame masses)")
    sys.exit(0)
z=opts.redshift
mc_source = opts.mcz/(1+z)

###
### Load grid, create fit
###

my_gp=None
if opts.fit_save_gp:
    my_gp=joblib.load(opts.pkl_file)
else:
    dat = np.loadtxt(opts.intermediate_file)  # delta LambdaTilde lnG

print("TEMPORARY HACK: Use small data set, to get code to run")
dat=dat[:500]  # temporary

X =dat[:,0:1]
Y = dat[:,-1]
my_fit = fit_gp(X,Y)

print(my_fit(np.c_[[0,500]])[0])

###
### Build EOS: lambda(m) curves
###

p0_ref = 2.272499999999999930e+33
epsilon0_ref = 2.050000000000000000e+14
xmax_ref =  7.25
if not 'gamma0' in eos_params:
    eos_params['gamma0']=0.707
if not 'gamma1' in eos_params:
    eos_params['gamma1']=0.707
if not 'p0' in eos_params:
    eos_params['p0'] = p0_ref
if not 'epsilon0' in eos_params:
    eos_params['epsilon0'] = epsilon0_ref
if not 'xmax' in eos_params:
    eos_params['xmax'] = xmax_ref

print(eos_params)
my_eos = EOSManager.EOSLindblomSpectral(name="internal",spec_params=eos_params)
dat_mr = EOSManager.make_mr_lambda(my_eos.eos)  # r m(Msun) lambda
lam_fit = scipy.interpolate.interp1d(dat_mr[:,1], dat_mr[:,2])

#print dat_mr, lam_fit(1.4)


###
### Build lambda_tilde (eta) function
###

def lambda_tilde_here(delta):
    # compute m1,m2
    eta = 0.25*(1.0 - delta*delta)
    m1,m2 = lalsimutils.m1m2(mc_source, eta)
    lam1 = lam_fit(m1)
    lam2 = lam_fit(m2)
    lambda_tilde, dLt = lalsimutils.tidal_lambda_tilde(m1,m2,lam1,lam2)
    return lambda_tilde


###
### Construct (one-dimensional) likelihood
###

fac_standard = 16 * np.power(2,2./5.)/np.power(3-1,2)
def likelihood_function(d):  
    lam_here = lambda_tilde_here(d)
    if isinstance(d,float):
        lnG = my_fit(np.c_[[d,lam_here]])[0]
        return np.exp(lnG)* mc_source *fac_standard/np.power(1-d*d,6./5.)
    else:
        return np.exp(my_fit(np.c_[d,lam_here])) * mc_source *fac_standard/np.power(1-d*d,6./5.)


###
### Perform one-dimensional integral
###
res = scipy.integrate.quad(likelihood_function,0, 0.5)

###
### Save result
###
I = res[0]
print(np.log(I), I)
vals = [np.log(I), eos_params["gamma1"],eos_params["gamma2"],eos_params["gamma3"], np.log10(float(eos_params["p0"])), np.log10(float(eos_params["epsilon0"])), eos_params["xmax"]]
print(vals)
np.savetxt(opts.integral_output,np.array(vals).T)
