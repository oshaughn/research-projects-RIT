# Test script for comparing GMM integrator to existing mcsampler integrator in
# RIFT. A simple n-dimensional integrand consisting of a highly-correlated
# Gaussian is used.

from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from RIFT.integrators import mcsampler, mcsamplerEnsemble

### test parameters

# width of domain of integration, same for all dimensions
width = 10.0                                                    
llim = -1 * width / 2
rlim = width / 2
ndim = 1                                                        
mu = np.random.uniform(rlim*0.75,rlim)
sigma = 1
cov = sigma**2
# number of iterations for mcsamplerEnsemble
n_iters = 40                                                    
nmax = 40000                                                    


### define integrand as a single gaussian
def f(x1):
    x = np.array(x1,dtype=float)
    return (norm.pdf(x, loc=mu, scale=sigma))

#print(f(np.random.uniform(llim,rlim,size=100)))

### initialize samplers
sampler = mcsampler.MCSampler()
samplerEnsemble = mcsamplerEnsemble.MCSampler()

### add parameters
params = [str(i) for i in range(ndim)]
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
integral_2, var_2, eff_samp_2, dict_return = samplerEnsemble.integrate(f, *params, 
        min_iter=n_iters, max_iter=n_iters, correlate_all_dims=True, n_comp=n_comp,super_verbose=True,dict_return=True)
print(" --- finished GMM --")
print("mu sigma in ",mu, sigma)
my_integrator = dict_return["integrator"]
gmm_dict = my_integrator.gmm_dict
my_gmm = gmm_dict[(0,)]  # pull out the specific thing we just optimized
scale_from_unnorm = 0.5*(rlim-llim) # scale factor, does not also account for offset 
mu_sampler =my_gmm.means[0][0]*scale_from_unnorm
sigma_sampler =  np.sqrt(my_gmm.covariances[0])[0,0]*scale_from_unnorm
print("mu, sigma for sampler ",mu_sampler,sigma_sampler) # print mean and covariance

# Validate: draw from sampler
npts_here = 500
my_demo_samples = my_gmm.sample(npts_here)[:,0]
my_demo_samples.sort()
plt.scatter(my_demo_samples,np.arange(npts_here)/(1.*npts_here))
plt.plot(my_demo_samples, truncnorm( (llim-mu)/sigma, (rlim-mu)/sigma, loc=mu,scale=sigma).cdf(my_demo_samples),label='truncnorm(crap)')
plt.xlabel("x")
plt.ylabel("CDF of sampling distribution")
plt.savefig("my_cdf_sampler.png")

### Now pull out the integrator's model
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
    s = sigma
    ### get sorted samples (for the current dimension)
    x_1 = arr_1[:,i][np.argsort(arr_1[:,i])]
    x_2 = arr_2[:,i][np.argsort(arr_2[:,i])]
    ### plot true cdf
    plt.plot(x_1, truncnorm.cdf(x_1, llim, rlim, mu, s), label="True CDF",
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

fname = "cdf_adapt_test.png"

print("Saving CDF figure as " + fname + "...")

plt.savefig(fname)
