#Monica Rizzo, 2018...minor edits by ROS
#
# USAGE
#   python util_ConstructBNSMarginalEOSRank.py --composite G298048/production_C00_cleaned_TaylorT4/all.composite --parameter mc --parameter eta --lnL-cutoff 10 --using-eos ap4

import numpy as np
import matplotlib.pyplot as plt
import argparse

#from lalinference.rapid_pe import RIFT.lalsimutils as lalsimutils
import RIFT.lalsimutils as lalsimutils
import RIFT.physics.EOSManager as EOSManager
#import lalsim_EOS_tools as let
from scipy.integrate import nquad
#import EOS_param as ep
import os

import RIFT.physics.MonotonicSpline as ms 

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


###
### Conversion Function
###

#lambda_tilde (this is not implemented how i need it to be implemeted in lalsimutils)
def tidal_lambda_tilde(m1, m2, lam1, lam2):
    
    eta = (m1*m2)/(m1+m2)**2

    lam_til = (8./13.) * ((1 + 7*eta - 31*eta**2) * (lam1 + lam2) \
    + np.sqrt(1 - 4*eta) * (1 + 9*eta - 11*eta**2) * (lam1 - lam2))

    return lam_til

def calc_mc(red_data):
    return lalsimutils.mchirp(red_data[:,1], red_data[:,2])

def calc_eta(red_data):
    return lalsimutils.symRatio(red_data[:,1], red_data[:,2])

def calc_lambda_tilde(red_data):
    return tidal_lambda_tilde(red_data[:,1], red_data[:,2], red_data[:,9], red_data[:,10])

def calc_lambda_from_eos(red_data):

    #calculate lambda values (lambda_of_m defined in )
    lam1 = lambda_of_m(red_data[:,1])
    lam2 = lambda_of_m(red_data[:,2])

    return tidal_lambda_tilde(red_data[:,1], red_data[:,2], lam1, lam2)

#dictionary mapping parameters to functions used to calculate them
param_dict = {'mc': calc_mc, 
              'eta': calc_eta,
              'lambda_tilde': calc_lambda_tilde, 
              'eos_lambda': calc_lambda_from_eos
             }      

###
### Priors
###

#prior functions (most taken from util_ConstructIntrinsincPosterior_GenericCoordinates.py)
def mc_prior(mc):
    return mc/(max_mc - min_mc)

def eta_prior(eta):
    return 1./(eta**(6./5.) * (1.- 4.*eta)**(1/2) * 1.44)

def lambda_tilde_prior(lambda_tilde):
    return 1./5000.

def lambda_from_eos_prior(lam):
    return 1.

def s1z_prior(x):
    return 1./2.

def s2z_prior(x):
    return 1./2.

#dictionary of available priors
prior_dict = {'mc': mc_prior,
              'eta': eta_prior,
              'lambda_tilde': lambda_tilde_prior, 
              'eos_lambda' : lambda_from_eos_prior, 
              's1z': s1z_prior,
              's2z': s2z_prior
             }


###
### Fit Functions
###

#gp fit (basically taken from util_ConstructIntrinsincPosterior_GenericCoordinates.py)
def gp_fit(x, y, mc_index=0):
    
    print("Fitting", len(y), "points: ")

    #length scale tuning taken from util_ConstructIntrinsicPosterior_GenericCoordinates.py
    length_scale_est = []
    length_scale_bounds_est = []

    for indx in np.arange(len(x[0])):
    # These length scales have been tuned by experience
        length_scale_est.append(2*np.std(x[:,indx]))  # auto-select range based on sampling retained
        length_scale_min_here= np.max([1e-3,0.2*np.std(x[:,indx]/np.sqrt(len(x)))])
        if indx == mc_index:
            length_scale_min_here= 0.2*np.std(x[:,indx]/np.sqrt(len(x)))
            print(" Setting mc range: retained point range is ", np.std(x[:,indx]), " and target min is ", length_scale_min_here)
    length_scale_bounds_est.append( (length_scale_min_here, 5*np.std(x[:,indx])) )

    #set up kernel
    kernel = WhiteKernel(noise_level=0.1,noise_level_bounds=(1e-2,1))+C(0.5, (1e-3,1e1))*RBF(length_scale=length_scale_est, length_scale_bounds=length_scale_bounds_est)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8)

    #fit and estimate
    gp.fit(x, y)

    def fit_func(coord):
       return gp.predict(coord)[0]
 
    return fit_func


###
### Integral
###

#likelihood integral
def integrate_likelihood(fit_function, bounds, params):

    #inputs: variables corresponding to bounds
    def func(*args):
  
        input_array = []
        for arg in args:
            input_array.append(arg)

        #grab index of mc and eta from params
        if 'eos_lambda' in params:
            mc_indx = params.index('mc')
            eta_indx = params.index('eta')
         
            m1,m2 = lalsimutils.m1m2(args[mc_indx], args[eta_indx])

            lam1 = lambda_of_m(m1)
            lam2 = lambda_of_m(m2)
             
            l = tidal_lambda_tilde(m1, m2, lam1, lam2)

            input_array.append(l)


        #evaluate fit function at input values
        function_value = fit_function([input_array])
     
        #multiply all priors   
        prior_value = 1.
        for p, arg in zip(params, args):
            prior_value *= prior_dict[p](arg)
                                        
        return np.exp(function_value) * prior_value
    
    int_val = nquad(func, bounds)
    
    return int_val

#arguments (obv will need to add more)
parser=argparse.ArgumentParser()
parser.add_argument("--composite-file", type=str, help="file used to fit gp")
parser.add_argument("--parameter", action='append')
parser.add_argument("--using-eos", type=str, default=None, help="Name of EOS if not already determined in lnL")
parser.add_argument("--eos-param", type=str, default=None, help="parameterization of equation of state")
parser.add_argument("--parameter-eos",action='append',default=None,help="Parameters to pass to the EOS model named in 'eos-param'. Specify them as a list, in order: --parameter 1.5 --parameter 0.2 --parameter 3.7 ...")
parser.add_argument("--lnL-cutoff", type=float, default=80.0)
parser.add_argument("--fname-out", type=str, default='integral_result.dat')
opts=parser.parse_args()


#option to be used if gridded values not calculated assuming EOS
if opts.using_eos!=None:
    eos_name=opts.using_eos

    if opts.eos_param == 'spectral':
        # Does not seem to be general -- need to modify 
#        spec_param=ep.make_spec_param_eos(eos_name)
        
#        path=os.getcwd()

        #save for further use        
#        if 'lalsim_eos' not in os.listdir("."):
#            os.mkdir("lalsim_eos")      
        lalsim_spec_param=spec_param/(C_CGS**2)*7.42591549*10**(-25)
        np.savetxt("lalsim_eos/"+eos_name+"_spec_param_geom.dat", np.c_[lalsim_spec_param[:,1], lalsim_spec_param[:,0]])
      
        spec_param_eos=lalsim.SimNeutronStarEOSFromFile(path+"/lalsim_eos/"+eos_name+"_spec_param_geom.dat")
                
        mr_lambda=let.make_mr_lambda(spec_param_eos)

    else:
        my_eos = EOSManager.EOSFromDataFile(name=eos_name,fname =EOSManager.dirEOSTablesBase+"/" + eos_name+".dat")
        mr_lambda=EOSManager.make_mr_lambda(my_eos.eos) 

    lambda_const=ms.interpolate(mr_lambda[:,1], mr_lambda[:,2])    # Precompute constants for interpolation

    #calculate lambda(m)
    def lambda_of_m(mass):  

        if hasattr(mass, '__iter__'):
            lam = np.array([])    
        
            for m in mass:
                # should make ms.interp_func vectorized
                l = ms.interp_func(m, mr_lambda[:,1], mr_lambda[:,2], lambda_const)
                try:
                    if l > 5e3 or  l == None or np.isnan(l):
                        l = 0.
                except:
#                    print " Very unusual lambda(m) situation, probably because of outside the EOS range: ", m, l
                    l=0
                lam = np.append(lam, l)

            return lam
        else:
            lam = ms.interp_func(mass, mr_lambda[:,1], mr_lambda[:,2], lambda_const)
            try:
                if lam > 5e3 or np.isnan(lam) or lam == None:
                    lam = 0.
            except:
 #               print " Very unusual lambda(m) situation, probably because of outside the EOS range: ", mass, lam
                lam=0
            return lam

#    print " Testing lambda(m) function "
#    print lambda_of_m(1.4), lambda_of_m(0.9), lambda_of_m([0.9,1,1.4])

#assume standard composite file (including tides) format
if opts.composite_file:
   
    param_array = []    

    print("Fitting to params:")
    for param in opts.parameter:
        print(param)
        param_array.append(param)

    #append extra argument to calculate 
    if opts.using_eos:
       param_array.append('eos_lambda')

    #load data 
    comp_data = np.loadtxt(opts.composite_file)

    #determine mc range
    mc_comp = lalsimutils.mchirp(comp_data[:,1], comp_data[:,2]) 
    max_mc = max(mc_comp)
    min_mc = min(mc_comp)

    #set max_lnL
    max_lnL = max(comp_data[:,11])
    lnL_cutoff = opts.lnL_cutoff
 

    #reduce data according to lnL cutoff
    comp_data_reduced = np.array([])

    #apply likelihood cutoff
    for i in range(0, len(comp_data)):
        if comp_data[i,11] >= (max_lnL - lnL_cutoff):
            if len(comp_data_reduced) == 0:
                comp_data_reduced = np.hstack((comp_data_reduced, comp_data[i,:]))
            else:
                comp_data_reduced = np.vstack((comp_data_reduced, comp_data[i,:]))

    #values to fit
    x = np.zeros((len(comp_data_reduced), len(param_array)))
    
    #populate array of fit values
    for i in range(0, len(param_array)):
        x[:,i] = param_dict[param_array[i]](comp_data_reduced)

    lnL = comp_data_reduced[:,11]

    #fit data
    gp_fit_function = gp_fit(x, lnL)

    #print gp_fit_function([[mc[1], eta[1], lambda_tilde[1]]])

    #dictionary of integral bounds corresponding to parameters
    integral_bound_dict = {'mc': [min_mc, max_mc],
                       'eta': [0.14, 0.249999],   # For BNS, we can't go to higher than about 5:1 mass ratio
                       's1z': [-1.0, 1.0],
                       's2z': [-1.0, 1.0]
                      }

    #eta diverges at 0, set a lower bound of 0.01
    #also diverges at 0.25, set to 0.2499999999
 
    #array of integral limits to pass to integral function
    integral_bounds = []
   
    for p in param_array:
        if p != 'eos_lambda':
           integral_bounds.append(integral_bound_dict[p])

    #do the integral
    print("Integrating ...")
    print(integral_bounds)
    print(param_array)
    integral_result = integrate_likelihood(gp_fit_function, integral_bounds, param_array)
    print(integral_result)
    
    np.savetxt(opts.fname_out, integral_result)
