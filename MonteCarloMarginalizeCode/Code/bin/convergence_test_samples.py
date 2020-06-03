#! /usr/bin/env python
#
#
# GOAL
#   - takes two sets of samples, and some parameter(s)
#       - should be able to interchange samples provided with ILE *.xml.gz, *.composite, or posterior samples (preferred).  FLEXIBILITY NOT YET IMPLEMENTED. 
#         Postfix determines behavior
#   - performs specified test, with specified tolerance, to see if they are 'similar enough'
#   - returns FAILURE if test is a success (!), so a condor DAG will terminate
#
# EXAMPLES
# convergence_test_samples.py --samples GW170823_pure_NR_and_NRSur7dq2_lmax3_fmin20_C02_cleaned_alignedspin_zprior.dat --samples GW170823_pure_NR_and_NRSur7dq2_lmax3_fmin20_C02_cleaned_alignedspin_zprior.dat --parameter m1 --parameter m2   # test samples against themselves, must return 0!
#
# RESOURCES
#   Based on code in util_DriverIterateILEFitPosterior*.py

import numpy as np
import argparse
import scipy.stats
import numpy.linalg as la
import sys

from RIFT.misc.samples_utils import add_field

parser = argparse.ArgumentParser()
parser.add_argument("--samples", action='append', help="Samples used in convergence test")
parser.add_argument("--parameter", action='append', help="Parameters used in convergence test")
parser.add_argument("--parameter-range", action='append', help="Parameter ranges used in convergence test (used if KDEs or similar knowledge of the PDF is needed). If used, must specify for ALL variables, in order")
parser.add_argument("--method", default='lame',  help="Test to perform: lame|ks1d|...")
parser.add_argument("--threshold",default=0.01,type=float,  help="Manual threshold for the test being performed. (If not specified, the success condition is determined by default for that diagnostic, based on the samples size and properties).  Try 0.01")
parser.add_argument("--test-output",  help="Filename to return output. Result is a scalar >=0 and ideally <=1.  Closer to 0 should be good. Second column is the diagnostic, first column is 0 or 1 (success or failure)")
parser.add_argument("--always-succeed",action='store_true',help="Test output is always success.  Use for plotting convergence diagnostics so jobs insured to run for many iterations.")
parser.add_argument("--iteration-threshold",default=0,type=int,help="Test is applied if iteration >= iteration-threshold. Default is 0")
parser.add_argument("--iteration",default=0,type=int,help="Current reported iteration. Default is 0.")
opts=  parser.parse_args()

if len(opts.samples)<2:
    print(" Need at least two sets of samples")
    sys.exit(1)

if opts.iteration < opts.iteration_threshold:
    sys.exit(0)



# Test options
#
#   (a) lame: Compute a multivariate gaussian estimate (sample mean and variance), and then use KL divergence between them !
#   (b) KS_1d: One-dimensional KS test on cumulative distribution  
#   (c) KL_1d: One-dimensional KL divergence, using KDE estimate.  Requires bounded domain; parameter bounds can be passed 


def calc_kl(mu_1, mu_2, sigma_1, sigma_2, sigma_1_inv, sigma_2_inv):
    """
    calc_kl : KL divergence for two gaussians.  sigma_1, and sigma_2 are the covariance matricies.
    """
    return 0.5*(np.trace(np.dot(sigma_2_inv,sigma_1))+np.dot(np.dot((mu_2-mu_1).T, sigma_2_inv), (mu_2-mu_1))-len(mu_1)+np.log(la.det(sigma_2)/la.det(sigma_1)))

def calc_kl_scalar(mu_1, mu_2, sigma_1, sigma_2):
    """
    calc_kl : KL divergence for two gaussians.  sigma_1, and sigma_2 are the covariance matricies.
    """
    return np.log(sigma_2/sigma_1) -0.5 +( (mu_1-mu_2)**2 + sigma_1**2)/(2*sigma_2**2)


def test_lame(dat1,dat2):
    """
    Compute a multivariate gaussian estimate (sample mean and variance), and then use KL divergence between them !
    """
    mu_1 = np.mean(dat1,axis=0)
    mu_2 = np.mean(dat2,axis=0)
    sigma_1 = np.cov(dat1.T)
    sigma_2 = np.cov(dat2.T)
    if np.isscalar(mu_1) or len(mu_1)==1:
        return np.asscalar(calc_kl_scalar(mu_1, mu_2, sigma_1, sigma_2))
    else:
        sigma_1_inv = np.linalg.inv(sigma_1)
        sigma_2_inv = np.linalg.inv(sigma_2)
    return calc_kl(mu_1,mu_2, sigma_1, sigma_2, sigma_1_inv, sigma_2_inv)

def test_ks1d(dat1_1d, dat2_1d):
    """
    KS test based on two sample sets.  Uses the KS D value as threshold
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ks_2samp.html
    """
    return scipy.stats.ks_2samp(dat1_1d,dat2_1d)[0]  # return KS statistic

def test_KL1d(dat1_1d,dat2_1d,range1=None, range2=None):
    return None




# Procedure

samples1 = np.genfromtxt(opts.samples[0], names=True)
samples2 = np.genfromtxt(opts.samples[1], names=True)

# Add missing fields needed for some tests
if not('xi' in samples1.dtype.names):
            samples1 = add_field(samples1, [('chi_eff',float)]); samples1['chi_eff'] = (samples1["m1"]*samples1["a1z"]+samples1["m2"]*samples1["a2z"])/(samples1["m1"]+samples1["m2"])
            samples1 = add_field(samples1, [('xi',float)]); samples1['xi'] = (samples1["m1"]*samples1["a1z"]+samples1["m2"]*samples1["a2z"])/(samples1["m1"]+samples1["m2"])
if not('xi' in samples2.dtype.names):
            samples2 = add_field(samples2, [('chi_eff',float)]); samples2['chi_eff'] = (samples2["m1"]*samples2["a1z"]+samples2["m2"]*samples2["a2z"])/(samples2["m1"]+samples2["m2"])
            samples2 = add_field(samples2, [('xi',float)]); samples2['xi'] = (samples2["m1"]*samples2["a1z"]+samples2["m2"]*samples2["a2z"])/(samples2["m1"]+samples2["m2"])


param_names1 = samples1.dtype.names; param_names2 = samples2.dtype.names
npts1 = len(samples1[param_names1[0]])
npts2 = len(samples2[param_names2[0]])  

# Read in data into array.  For now, assume the specific parameters requested are provided.
dat1 = np.empty( (npts1,len(opts.parameter)))
dat2 = np.empty( (npts2,len(opts.parameter)))
indx=0
for param in opts.parameter:
    dat1[:,indx] = samples1[param]
    dat2[:,indx] = samples2[param]
    indx+=1


# Perform test
val_test = np.inf
if opts.method == 'lame':
    val_test = test_lame(dat1,dat2)
elif opts.method == 'KS_1d':
    val_test = test_ks1d(dat1[:,0],dat2[:,0])
elif opts.method == 'KL_1d':
    val_test = test_KL1d(dat1[:,0],dat2[:,0])
else:
    print(" No known method ", opts.method)
print(val_test)

if opts.always_succeed or (opts.threshold is None):
    sys.exit(0)

if (val_test < opts.threshold):  
    sys.exit(1)
else:
    sys.exit(0)
