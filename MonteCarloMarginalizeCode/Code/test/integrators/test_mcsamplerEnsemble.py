# Test script for comparing GMM integrator to existing mcsampler integrator in
# RIFT. A simple n-dimensional integrand consisting of a highly-correlated
# Gaussian is used.

from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from RIFT.integrators import mcsampler, mcsamplerEnsemble

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
    return (multivariate_normal.pdf(x, mu, cov))

### initialize samplers
sampler = mcsampler.MCSampler()
samplerEnsemble = mcsamplerEnsemble.MCSampler()

### add parameters
for p in params:
    sampler.add_parameter(p, np.vectorize(lambda x:1), 
            prior_pdf=np.vectorize(lambda x:1),
            left_limit=llim, right_limit=rlim,
            adaptive_sampling=True)
    samplerEnsemble.add_parameter(p, left_limit=llim, right_limit=rlim,adaptive_sampling=True)

# number of Gaussian components to use in GMM
n_comp = 1

### integrate
integral_1, var_1, eff_samp_1, _ = sampler.integrate(f, *params, 
        no_protect_names=True, nmax=20000, save_intg=True)
print(" --- finished default --")
integral_2, var_2, eff_samp_2, _ = samplerEnsemble.integrate(f, *params, 
        min_iter=n_iters, max_iter=n_iters, correlate_all_dims=True, n_comp=n_comp,super_verbose=True)
print(" --- finished GMM --")
print("mu",mu)
### CDFs

### get our posterior samples as a single array
arr_1 = np.empty((len(sampler._rvs["0"]), ndim))
arr_2 = np.empty((len(samplerEnsemble._rvs["0"]), ndim))
for i in range(ndim):
    arr_1[:,i] = sampler._rvs[str(i)].flatten()
    arr_2[:,i] = samplerEnsemble._rvs[str(i)].flatten()

colors = ["black", "red", "blue", "green", "orange"]

plt.figure(figsize=(10, 8))

for i in range(ndim):
    s = np.sqrt(cov[i][i])
    ### get sorted samples (for the current dimension)
    x_1 = arr_1[:,i][np.argsort(arr_1[:,i])]
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
    p = samplerEnsemble._rvs["joint_prior"]
    ps = samplerEnsemble._rvs["joint_s_prior"]
    ### compute weights of samples
    weights_2 = (L * p / ps)[np.argsort(arr_2[:,i])]
    y_1 = np.cumsum(weights_1)
    y_1 /= y_1[-1] # normalize
    y_2 = np.cumsum(weights_2)
    y_2 /= y_2[-1] # normalize
    ### plot recovered cdf
    plt.plot(x_1, y_1, "--", label="Recovered CDF, mcsampler", color=colors[i],
            linewidth=2)
    plt.plot(x_2, y_2, ":", label="Recovered CDF, mcsamplerEnsemble",
            color=colors[i], linewidth=2)

plt.legend()

fname = "cdf.png"

print("Saving CDF figure as " + fname + "...")

plt.savefig(fname)
