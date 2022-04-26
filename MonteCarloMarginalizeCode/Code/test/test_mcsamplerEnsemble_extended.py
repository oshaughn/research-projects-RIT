# Test script for comparing GMM integrator to existing mcsampler integrator in
# RIFT. A simple n-dimensional integrand consisting of a highly-correlated
# Gaussian is used.

from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from RIFT.integrators import mcsampler, mcsamplerEnsemble, mcsamplerGPU

verbose=False

tempering_exp =0.01

### test parameters

# width of domain of integration, same for all dimensions
width = 10.0                                                    
# number of dimensions
ndim = 3                                                        
# mean of the Gaussian, allowed to occupy middle half of each dimension
mu = np.random.uniform(-1 * width / 4.0, width / 4.0, ndim)    
# number of iterations for mcsamplerEnsemble
n_iters = 40                                                    
# max number of samples for mcsampler
nmax = 40000                                                    

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
def f(x1, x2, x3):
    x = np.array([x1, x2, x3]).T
    return multivariate_normal.pdf(x, mu, cov)
def ln_f(x1, x2, x3):
    x = np.array([x1, x2, x3]).T
    return np.log(multivariate_normal.pdf(x, mu, cov)+1e-100)

### initialize samplers
sampler = mcsampler.MCSampler()
samplerEnsemble = mcsamplerEnsemble.MCSampler()
samplerAC = mcsamplerGPU.MCSampler()

### add parameters
for p in params:
    sampler.add_parameter(p, np.vectorize(lambda x:1/(rlim-llim)), 
            prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
            left_limit=llim, right_limit=rlim,
            adaptive_sampling=True)
    samplerEnsemble.add_parameter(p, 
                                  pdf=np.vectorize(lambda x:1/(rlim-llim)),
                                  prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
                                  left_limit=llim, right_limit=rlim,adaptive_sampling=True)
    # for AC sampler, make sure pdf and prior pdfs are *normalized* *initially*
    samplerAC.add_parameter(p, pdf=np.vectorize(lambda x:1/(rlim-llim)),
            prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
            left_limit=llim, right_limit=rlim,adaptive_sampling=True)

# number of Gaussian components to use in GMM
n_comp = 1

### integrate
integral_1, var_1, eff_samp_1, _ = sampler.integrate(f, *params, 
        no_protect_names=True, nmax=20000, save_intg=True,verbose=verbose)
print(" --- finished default --")
integral_1b, var_1b, eff_samp_1b, _ = samplerAC.integrate(f, *params, 
        no_protect_names=True, nmax=20000, save_intg=True,verbose=verbose)
print(" --- finished AC --")
use_lnL = False
return_lnI=False
if use_lnL:
    infunc = ln_f
else:
    infunc = f
integral_2, var_2, eff_samp_2, _ = samplerEnsemble.integrate(infunc, *params, 
        min_iter=n_iters, max_iter=n_iters, correlate_all_dims=True, n_comp=n_comp,super_verbose=verbose,verbose=verbose,tempering_exp=tempering_exp,use_lnL=use_lnL,return_lnI=return_lnI)
if return_lnI and use_lnL:
    integral_2 = np.exp(integral_2)
print(" --- finished GMM --")
print(np.array([integral_1,integral_1b,integral_2])*width**3)  # remove prior factor, should get result of normal over domain
print(" AC/default ",  integral_1b/integral_1, np.sqrt(var_1)/integral_1)  # off by width**3
print(" GMM/default ",integral_2/integral_1, np.sqrt(var_1)/integral_1, np.sqrt(var_2)/integral_2)
print("mu",mu)
### CDFs

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

    L = samplerAC._rvs["integrand"]
    p = samplerAC._rvs["joint_prior"]
    ps = samplerAC._rvs["joint_s_prior"]
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

plt.savefig(fname)
