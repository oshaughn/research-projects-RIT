# Test script for comparing GMM integrator to existing mcsampler integrator in
# RIFT. A simple n-dimensional integrand consisting of a highly-correlated
# Gaussian is used.

from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from scipy.special import erf

from RIFT.integrators import mcsampler, mcsamplerEnsemble, mcsamplerGPU

tempering_exp =0.01
# max number of samples for mcsampler
nmax = 1000000  
n_iters = nmax/1000

#

Z_rosenbrock = -5.804

### test parameters

### define integrand 
### some typecasting needed
def f(x1, x2):
    minus_lnL = np.array(np.power((1.-x1), 2) + 100.* np.power((x2-x1**2),2),dtype=float)
    return np.exp( - (minus_lnL))
def ln_f(x1, x2, x3): 
    minus_lnL = np.array(np.power((1.-x1), 2) + 100.* np.power((x2-x1**2),2),dtype=float)
    return - minus_lnL

### initialize samplers
sampler = mcsampler.MCSampler()
samplerEnsemble = mcsamplerEnsemble.MCSampler()
samplerAC = mcsamplerGPU.MCSampler()

### add parameters
llim=-5
rlim=5
params = ['x1','x2']
for p in params:
    sampler.add_parameter(p, np.vectorize(lambda x:1), 
            prior_pdf=np.vectorize(lambda x:1./(rlim-llim)),
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
        no_protect_names=True, nmax=nmax, save_intg=True,verbose=False)
print(np.log(integral_1), Z_rosenbrock)
print(" --- finished default --")
integral_1b, var_1b, eff_samp_1b, _ = samplerAC.integrate(f, *params, 
        no_protect_names=True, nmax=nmax, save_intg=True,verbose=False)
print(np.log(integral_1b), Z_rosenbrock)
print(" --- finished AC --")
print(" NEED TO ADD OPTION TO TEST CORRELATED SAMPLING ")
use_lnL = False
return_lnI=False
if use_lnL:
    infunc = ln_f
else:
    infunc = f
integral_2, var_2, eff_samp_2, _ = samplerEnsemble.integrate(infunc, *params, 
        min_iter=n_iters, max_iter=n_iters, correlate_all_dims=True, n_comp=n_comp,super_verbose=False,verbose=False,tempering_exp=tempering_exp,use_lnL=use_lnL,return_lnI=return_lnI)
if return_lnI and use_lnL:
    integral_2 = np.exp(integral_2)
print(" --- finished GMM --")
print(integral_1,integral_1b,integral_2,np.exp(Z_rosenbrock))
print(np.array([integral_1,integral_1b,integral_2])/np.exp(Z_rosenbrock))
print(" AC/default ", integral_1b/integral_1, np.sqrt(var_1)/integral_1)  # off by width**3
print(" GMM/default ",integral_2/integral_1, np.sqrt(var_1)/integral_1, np.sqrt(var_2)/integral_2)
### CDFs

### get our posterior samples as a single array
ndim=2
arr_1 = np.empty((len(sampler._rvs["x1"]), ndim))
arr_1b = np.empty((len(samplerAC._rvs["x1"]), ndim))
arr_2 = np.empty((len(samplerEnsemble._rvs["x1"]), ndim))
for i in range(ndim):
    arr_1[:,i] = sampler._rvs["x"+str(i+1)].flatten()
    arr_1b[:,i] = samplerAC._rvs["x"+str(i+1)].flatten()
    arr_2[:,i] = samplerEnsemble._rvs["x"+str(i+1)].flatten()

colors = ["black", "red", "blue", "green", "orange"]

plt.figure(figsize=(10, 8))

for i in range(ndim):
    ### get sorted samples (for the current dimension)
    x_1 = arr_1[:,i][np.argsort(arr_1[:,i])]
    x_1b = arr_1b[:,i][np.argsort(arr_1b[:,i])]
    x_2 = arr_2[:,i][np.argsort(arr_2[:,i])]
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


# Semianalytic plot
def fn_marginal_x(x):
    return 1/(10) * np.sqrt(np.pi)/2000 * (erf( 10*(5-x**2)) + erf(10*(5+x**2))) * np.exp( - (1-x)**2)
def dP_cdf(p,x):
    return fn_marginal_x(x)

from scipy import integrate, interpolate

x_i = np.linspace(-5,5,1000)
cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.01*(10)).T[0]
cdf/=cdf[-1]
plt.plot(x_i,cdf,label="true")



plt.legend()

fname = "cdf_rosenbrock.png"

print("Saving CDF figure as " + fname + "...")

plt.savefig(fname)
