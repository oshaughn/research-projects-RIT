# Test script for comparing GMM integrator to existing mcsampler integrator in
# RIFT. A simple n-dimensional integrand consisting of a highly-correlated
# Gaussian is used.

from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from RIFT.integrators import mcsampler, mcsamplerEnsemble, mcsamplerGPU, mcsamplerAdaptiveVolume
has_nflow = False
try:
    from RIFT.integrators import mcsamplerNFlow
    has_nflow=True
except:
    print(" No nflow ")


import optparse
parser = optparse.OptionParser()
parser.add_option("--n-max",type=int,default=40000)
parser.add_option("--n-eff",type=int,default=1000)
parser.add_option("--use-lnL",action='store_true')
parser.add_option("--save-plot",action='store_true')
parser.add_option("--as-test",action='store_true')
parser.add_option("--no-adapt",action='store_true')
parser.add_option("--floor-level",default=0.0,type=float)
parser.add_option("--n-chunk",default=10000,type=int)
parser.add_option("--verbose",action='store_true')
opts, args = parser.parse_args()

verbose=opts.verbose

tempering_exp =0.1

### test parameters

# width of domain of integration, same for all dimensions
width = 10.0                                                    
# number of dimensions
ndim = 3                                                        
# mean of the Gaussian, allowed to occupy middle half of each dimension
mu = np.random.uniform(-1 * width / 4.0, width / 4.0, ndim)    
# max number of samples for mcsampler
nmax = opts.n_max                                         
# number of iterations for mcsamplerEnsemble
n_iters = int(nmax/opts.n_chunk)

llim = -1 * width / 2
rlim = width / 2

### generate list of named parameters
params = [str(i) for i in range(ndim)]

### generate the covariance matrix
cov = np.identity(ndim)
cov[ndim - 1][ndim - 1] = 0.05 # make it narrower in one dimension

### add some covariance (to test handling of strongly-correlated likelihoods)
cov[0][ndim - 1] = -0.1
cov[ndim - 1][0] = -0.1

### define integrand as a weighted sum of Gaussians
scale_factor = 100

def f(x1, x2, x3):
    x = np.array([x1, x2, x3]).T
    return scale_factor*multivariate_normal.pdf(x, mu, cov)
def ln_f(x1, x2, x3):
    x = np.array([x1, x2, x3]).T
    return np.log(scale_factor*multivariate_normal.pdf(x, mu, cov)+1e-100)

### initialize samplers
sampler = mcsampler.MCSampler()
samplerEnsemble = mcsamplerEnsemble.MCSampler()
samplerAC = mcsamplerGPU.MCSampler()
samplerAV = mcsamplerAdaptiveVolume.MCSampler()
samplerNF=None
if has_nflow:
    samplerNF = mcsamplerNFlow.MCSampler()

### add parameters
for p in params:
    sampler.add_parameter(p, np.vectorize(lambda x:1/(rlim-llim)), 
            prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
            left_limit=llim, right_limit=rlim,
            adaptive_sampling=not opts.no_adapt)
    samplerEnsemble.add_parameter(p, 
                                  pdf=np.vectorize(lambda x:1/(rlim-llim)),
                                  prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
                                  left_limit=llim, right_limit=rlim,adaptive_sampling=not opts.no_adapt)
    # for AC sampler, make sure pdf and prior pdfs are *normalized* *initially*
    samplerAC.add_parameter(p, pdf=np.vectorize(lambda x:1/(rlim-llim)),
            prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
            left_limit=llim, right_limit=rlim,adaptive_sampling=not opts.no_adapt)
    samplerAV.add_parameter(p, pdf=np.vectorize(lambda x:1/(rlim-llim)),
            prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
            left_limit=llim, right_limit=rlim,adaptive_sampling=True)
    if has_nflow:
        samplerNF.add_parameter(p, pdf=np.vectorize(lambda x:1/(rlim-llim)),
            prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
            left_limit=llim, right_limit=rlim,adaptive_sampling=True)

# number of Gaussian components to use in GMM
n_comp = 1

### integrate
extra_args = {"n": opts.n_chunk,"n_adapt":100, "floor_level":opts.floor_level,"tempering_exp" :tempering_exp,"neff":opts.n_eff}  # don't terminate
integral_1, var_1, eff_samp_1, _ = sampler.integrate(f, *params, 
        no_protect_names=True, nmax=nmax, save_intg=True,verbose=verbose,**extra_args)
print(" default {} {} {} ".format(integral_1, np.sqrt(var_1)/integral_1, eff_samp_1))
print(" --- finished default --")
use_lnL = opts.use_lnL
return_lnI=opts.use_lnL
if use_lnL:
    infunc = ln_f
else:
    infunc = f
integral_1b, var_1b, eff_samp_1b, _ = samplerAC.integrate(infunc, *params, 
        no_protect_names=True, nmax=nmax, save_intg=True,verbose=verbose,use_lnL=use_lnL,**extra_args)
if use_lnL:
    rel_error = np.exp(var_1b/2 - integral_1b) # I think ...
    integral_1b = np.exp(integral_1b)
else:
    rel_error = np.sqrt(var_1b)/integral_1b
print(" AC {} {} {} ".format(integral_1b, rel_error, eff_samp_1b))
print(" --- finished AC --")
integral_2, var_2, eff_samp_2, _ = samplerEnsemble.integrate(infunc, *params, 
        min_iter=n_iters, max_iter=n_iters, correlate_all_dims=True, n_comp=n_comp,super_verbose=False,verbose=verbose,use_lnL=use_lnL,return_lnI=return_lnI,**extra_args)
if return_lnI and use_lnL:
    rel_error_2 = np.exp(var_2/2 - integral_2)
    integral_2 = np.exp(integral_2)
else:
    rel_error_2 = np.sqrt(var_2)/integral_2
print(" GMM {} {} {} ".format(integral_2, rel_error_2, eff_samp_2))
print(" --- finished GMM --")
samplerAV.setup()
integral_3, var_3, eff_samp_3, _ = samplerAV.integrate_log(ln_f, *params, no_protect_names=True,
        nmax=nmax,
        min_iter=n_iters, max_iter=n_iters, **extra_args)
integral_3 = np.exp(integral_3)
var_3 = np.exp(var_3)
print(" -- finished AV -- ")
if samplerNF:
    samplerNF.setup()
    extra_args_here = {}
    extra_args_here.update(extra_args)
    extra_args_here['n_adapt'] = 10 # smaller count
    extra_args_here['tempering_exp']=1 # don't use 
    print(" INJECTIONS ", mu)
    integral_4, var_4, eff_samp_4, _ = samplerNF.integrate_log(ln_f, *params, no_protect_names=False,
        nmax=nmax,
        min_iter=n_iters, max_iter=n_iters, super_verbose=False,**extra_args_here)
    integral_4 = np.exp(integral_4)
    var_4 = np.exp(var_4)
    print(" -- finished NF -- ")


print(np.array([integral_1,integral_1b,integral_2,integral_3])*width**3)  # remove prior factor, should get result of normal over domain
print(np.array([np.sqrt(var_1)/integral_1,np.sqrt(var_1b)/integral_1b,np.sqrt(var_2)/integral_2,np.sqrt(var_3)/integral_3]))  # relative error in each integral
print(" AC/default ",  integral_1b/integral_1, np.sqrt(var_1)/integral_1)  # off by width**3
print(" GMM/default ",integral_2/integral_1, np.sqrt(var_1)/integral_1, np.sqrt(var_2)/integral_2)
print(" AV/default ",integral_3/integral_1, np.sqrt(var_1)/integral_1, np.sqrt(var_3)/integral_3)
if has_nflow:
    print(" NF/default ",integral_4/integral_1, np.sqrt(var_1)/integral_1, np.sqrt(var_4)/integral_4)
print("mu",mu)
### CDFs

sigma_fail =4
if opts.as_test:
    # remember var is a VARIANCE
    if np.log(np.abs(integral_1b/integral_1)) > sigma_fail*np.sqrt(var_1)/integral_1:
        print(" FAIL AC")
        exit(1)
    if np.log(np.abs(integral_2/integral_1)) > sigma_fail*np.sqrt(var_1+var_2)/integral_1:
        print(" FAIL GMM ")
        exit(1)
    if np.log(np.abs(integral_3/integral_1)) > sigma_fail*np.sqrt(var_1+var_3)/integral_1:
        print(" FAIL AV ")
        exit(1)
        

if not(opts.save_plot):
    exit(0)

### get our posterior samples as a single array
arr_1 = np.empty((len(sampler._rvs["0"]), ndim))
arr_1b = np.empty((len(samplerAC._rvs["0"]), ndim))
arr_2 = np.empty((len(samplerEnsemble._rvs["0"]), ndim))
for i in range(ndim):
    arr_1[:,i] = sampler._rvs[str(i)].flatten()
    arr_1b[:,i] = samplerAC._rvs[str(i)].flatten()
    arr_2[:,i] = samplerEnsemble._rvs[str(i)].flatten()

colors = ["black", "red", "blue", "green", "orange"]

plt.figure(figsize=(10, 8))

for i in range(ndim):
    s = np.sqrt(cov[i][i])
    ### get sorted samples (for the current dimension)
    x_1 = arr_1[:,i][np.argsort(arr_1[:,i])]
    x_1b = arr_1b[:,i][np.argsort(arr_1b[:,i])]
    x_2 = arr_2[:,i][np.argsort(arr_2[:,i])]
    ### plot true cdf
    plt.plot(x_1, truncnorm.cdf(x_1, llim, rlim, mu[i], s), label="True CDF",
            color=colors[i], linewidth=0.5)
    # NOTE: old mcsampler stores L, mcsamplerEnsemble stores lnL
    L = sampler._rvs["integrand"]
    p = sampler._rvs["joint_prior"]
    ps = sampler._rvs["joint_s_prior"]
    ### compute weights of samples
    weights_1 = (L * p / ps)[np.argsort(arr_1[:,i])]
    L = samplerEnsemble._rvs["integrand"]
    if return_lnI:
        L = np.exp(L - np.max(L))
    p = samplerEnsemble._rvs["joint_prior"]
    ps = samplerEnsemble._rvs["joint_s_prior"]
    ### compute weights of samples
    weights_2 = (L * p / ps)[np.argsort(arr_2[:,i])]

    if "integrand" in samplerAC._rvs:
        L = samplerAC._rvs["integrand"]
        p = samplerAC._rvs["joint_prior"]
        ps = samplerAC._rvs["joint_s_prior"]
    else:
        L = np.exp(samplerAC._rvs["log_integrand"])
        p = np.exp(samplerAC._rvs["log_joint_prior"])
        ps = np.exp(samplerAC._rvs["log_joint_s_prior"])
    ### compute weights of samples
    weights_1b = (L * p / ps)[np.argsort(arr_1b[:,i])]

    y_1 = np.cumsum(weights_1)
    y_1 /= y_1[-1] # normalize
    y_2 = np.cumsum(weights_2)
    y_2 /= y_2[-1] # normalize
    y_1b = np.cumsum(weights_1b)
    y_1b /= y_1b[-1]
    ### plot recovered cdf
    plt.plot(x_1, y_1, "--", label="Recovered CDF, mcsampler", color=colors[i],
            linewidth=2)
    plt.plot(x_2, y_2, ":", label="Recovered CDF, mcsamplerEnsemble",
            color=colors[i], linewidth=2)
    plt.plot(x_1b, y_1b, ".", label="Recovered CDF, mcsamplerAC",
            color=colors[i], linewidth=1)

#plt.legend()

fname = "cdf.pdf"

print("Saving CDF figure as " + fname + "...")
plt.xlabel(r'$x$')
plt.ylabel(r'$P(<x)$')
plt.savefig(fname)


# TEST OUTPUT
