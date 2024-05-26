# Test for evaluating rosenbrock 2d likelihood, where the 1d marginal and evidence can be computed.
#
# Suggested:
#   python test_mcsampler_rosenbrock.py;  plot_posterior_corner.py --posterior-file fairdraw_rosenbrock_1.dat --posterior-file fairdraw_rosenbrock_1b.dat --posterior-file fairdraw_rosenbrock_2.dat --parameter x1 --parameter x2  --quantiles None --ci-list [0.9]

from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from scipy.special import erf

from RIFT.integrators import mcsampler, mcsamplerEnsemble, mcsamplerGPU, mcsamplerAdaptiveVolume

import optparse
parser = optparse.OptionParser()
parser.add_option("--n-max",type=int,default=1000000)
parser.add_option("--save-plot",action='store_true')
parser.add_option("--production",action='store_true')
parser.add_option("--use-lnL",action='store_true')
parser.add_option("--lnL-shift",default=100,type=float,help="Our integrators assume lnL >0 when designing adaptation, so this shift helps us adapt, and is more consistent with our real problems.  Choose adapt-weight-exponent consistent with this value to be most realistic")
#parser.add_option("--as-test",action='store_true')
#parser.add_option("--no-adapt",action='store_true')
parser.add_option("--floor-level",default=0.4,type=float)  # for this problem, a higher floor level helps
parser.add_option("--adapt-weight-exponent",default=0.1,type=float)
parser.add_option("--n-chunk",default=10000,type=int)
parser.add_option("--verbose",action='store_true')
opts, args = parser.parse_args()


tempering_exp =opts.adapt_weight_exponent
# max number of samples for mcsampler
nmax = opts.n_max
n_block=opts.n_chunk
n_iters = nmax/n_block

save_fairdraws=True
save_fairdraw_prefix="fairdraw_rosenbrock"
#

Z_rosenbrock = -5.804

### test parameters

### define integrand 
### some typecasting needed
lnL_offset = opts.lnL_shift    # lnL > 1 used in adaptation
def f(x1, x2):
    minus_lnL = np.array(np.power((1.-x1), 2) + 100.* np.power((x2-x1**2),2),dtype=float)
    return np.exp( lnL_offset - (minus_lnL))
def ln_f(x1, x2): 
    minus_lnL = np.array(np.power((1.-x1), 2) + 100.* np.power((x2-x1**2),2),dtype=float)
    return lnL_offset - minus_lnL

### initialize samplers
sampler = mcsampler.MCSampler()
samplerEnsemble = mcsamplerEnsemble.MCSampler()
samplerAC = mcsamplerGPU.MCSampler()
samplerAV = mcsamplerAdaptiveVolume.MCSampler()

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
    samplerAV.add_parameter(p, pdf=np.vectorize(lambda x:1/(rlim-llim)),
            prior_pdf=np.vectorize(lambda x:1/(rlim-llim)),
            left_limit=llim, right_limit=rlim,adaptive_sampling=True)

# number of Gaussian components to use in GMM
n_comp = 2

extra_args = {"n": opts.n_chunk,"n_adapt":100, "floor_level":opts.floor_level,"tempering_exp" :tempering_exp}


### integrate
integral_1, var_1, eff_samp_1, _ = sampler.integrate(f, *params, 
        no_protect_names=True, nmax=nmax, save_intg=True,verbose=opts.verbose,**extra_args)
print(np.log(integral_1) - lnL_offset, Z_rosenbrock)
print(" --- finished default --")
integral_1b, var_1b, eff_samp_1b, _ = samplerAC.integrate(f, *params, 
        no_protect_names=True, nmax=nmax, save_intg=True,verbose=opts.verbose,**extra_args)
print(np.log(integral_1b) - lnL_offset, Z_rosenbrock)
print(" --- finished AC --")
print(" NEED TO ADD OPTION TO TEST CORRELATED SAMPLING ")
use_lnL = opts.use_lnL
return_lnI=opts.use_lnL
if use_lnL:
    infunc = ln_f
else:
    infunc = f
integral_2, var_2, eff_samp_2, _ = samplerEnsemble.integrate(infunc, *params, 
        min_iter=n_iters, max_iter=n_iters, correlate_all_dims=True, n_comp=n_comp,super_verbose=False,verbose=opts.verbose,use_lnL=use_lnL,return_lnI=return_lnI,**extra_args)
if return_lnI and use_lnL:
    integral_2 = np.exp(integral_2)
print(np.log(integral_2) - lnL_offset, Z_rosenbrock)
print(" --- finished GMM --")
#def my_infunc(*args):
#    return ln_f(*np.array([*args]).T)
samplerAV.setup()
integral_3, var_3, eff_samp_3, _ = samplerAV.integrate_log(ln_f, *params, no_protect_names=True,
        nmax=nmax,
        min_iter=n_iters, max_iter=n_iters, **extra_args)
integral_3 = np.exp(integral_3)
var_3 = np.exp(var_3)



print(integral_1/np.exp(lnL_offset),integral_1b/np.exp(lnL_offset),integral_2/np.exp(lnL_offset),integral_3/np.exp(lnL_offset),np.exp(Z_rosenbrock))
print(np.array([integral_1,integral_1b,integral_2,integral_3])/np.exp(Z_rosenbrock+lnL_offset))
print(" AC/default ", integral_1b/integral_1, np.sqrt(var_1)/integral_1)  # off by width**3
print(" GMM/default ",integral_2/integral_1, np.sqrt(var_1)/integral_1, np.sqrt(var_2)/integral_2)
print(" AV/default ",integral_3/integral_1, np.sqrt(var_1)/integral_1, np.sqrt(var_3)/integral_3)
### CDFs

### get our posterior samples as a single array
ndim=2
arr_1 = np.empty((len(sampler._rvs["x1"]), ndim))
arr_1b = np.empty((len(samplerAC._rvs["x1"]), ndim))
arr_2 = np.empty((len(samplerEnsemble._rvs["x1"]), ndim))
arr_3 = np.empty((len(samplerAV._rvs["x1"]), ndim))
for i in range(ndim):
    arr_1[:,i] = sampler._rvs["x"+str(i+1)].flatten()
    arr_1b[:,i] = samplerAC._rvs["x"+str(i+1)].flatten()
    arr_2[:,i] = samplerEnsemble._rvs["x"+str(i+1)].flatten()
    arr_3[:,i] = samplerAV._rvs["x"+str(i+1)].flatten()

colors = ["black", "red", "blue", "green", "orange"]

plt.figure(figsize=(10, 8))

if True:
    # NOTE: old mcsampler stores L, mcsamplerEnsemble stores lnL
    L = sampler._rvs["integrand"]
    p = sampler._rvs["joint_prior"]
    ps = sampler._rvs["joint_s_prior"]
    ### compute weights of samples
    weights_1 = (L * p / ps)
    n_ess_1 = np.sum(weights_1)**2/np.sum(weights_1**2)
    print("default  n_eff, n_ess ", eff_samp_1,n_ess_1)

    L = samplerEnsemble._rvs["integrand"]
    if return_lnI:
        L = np.exp(L - np.max(L))
    p = samplerEnsemble._rvs["joint_prior"]
    ps = samplerEnsemble._rvs["joint_s_prior"]
    ### compute weights of samples
    weights_2 = (L * p / ps)
    n_ess_2 = np.sum(weights_2)**2/np.sum(weights_2**2)
    print("AC  n_eff, n_ess ", eff_samp_2,n_ess_2)


    L = samplerAC._rvs["integrand"]
    p = samplerAC._rvs["joint_prior"]
    ps = samplerAC._rvs["joint_s_prior"]
    ### compute weights of samples
    weights_1b = (L * p / ps)
    n_ess_1b = np.sum(weights_1b)**2/np.sum(weights_1b**2)
    print("GMM  n_eff, n_ess ", eff_samp_1b,n_ess_1b)

    lnL = samplerAV._rvs["log_integrand"]  
    lnp = samplerAV._rvs["log_joint_prior"]  
    lnps = samplerAV._rvs["log_joint_s_prior"]  
    ### compute weights of samples
    weights_3 = np.exp(lnL + lnp - lnps )
    n_ess_3 = np.sum(weights_3)**2/np.sum(weights_3**2)
    print("AV  n_eff, n_ess ", eff_samp_3,n_ess_3)


if save_fairdraws:
    npts_out_1 = int(n_ess_1)
    p = np.array(weights_1/np.sum(weights_1),dtype=np.float64)
    indx_save_1 = np.random.choice(np.arange(len(p)), size=npts_out_1, p=p)
    np.savetxt(save_fairdraw_prefix+"_1.dat", arr_1[indx_save_1],header=" x1 x2")

    npts_out_2 = int(n_ess_2)
    p = np.array(weights_2/np.sum(weights_2),dtype=np.float64)
    indx_save_2 = np.random.choice(np.arange(len(p)), size=npts_out_2, p=p)
    np.savetxt(save_fairdraw_prefix+"_2.dat", arr_2[indx_save_2],header=" x1 x2")

    npts_out_1b = int(n_ess_1b)
    p = np.array(weights_1b/np.sum(weights_1b),dtype=np.float64)
    indx_save_1b = np.random.choice(np.arange(len(p)), size=npts_out_1b, p=p)
    np.savetxt(save_fairdraw_prefix+"_1b.dat", arr_1b[indx_save_1b],header=" x1 x2")

    npts_out_3 = np.min([int(n_ess_3*0.8),5000])
    p = np.array(weights_3/np.sum(weights_3),dtype=np.float64)
    indx_save_3 = np.random.choice(np.arange(len(p)), size=npts_out_3, p=p)
    np.savetxt(save_fairdraw_prefix+"_3.dat", arr_3[indx_save_3],header=" x1 x2")


##
## HOW TO MAKE JS TEST
# for param in x1 x2 ; do ls fairdraw_*.dat | ../scripts/tool_pairs.py --prefix="convergence_test_samples.py --method JS --parameter ${param}" > tmpfile;./tmpfile > jsvals_${param}.dat; done

## HOW TO MAKE A PLOT
# plot_posterior_corner.py --posterior-file fairdraw_rosenbrock_1.dat --posterior-file fairdraw_rosenbrock_1b.dat --posterior-file fairdraw_rosenbrock_2.dat --parameter x1 --parameter x2  --quantiles None --ci-list [0.9]


# Make copies, so they have the same orders relative to the sample values
weights_1_orig = np.array(weights_1)
weights_1b_orig = np.array(weights_1b)
weights_2_orig = np.array(weights_2)
weights_3_orig = np.array(weights_3)
# Note: the fairdraw export is better
for i in range(ndim):
    ### get sorted samples (for the current dimension)
    indx_sort = np.argsort(arr_1[:,i])
    x_1 = arr_1[:,i][indx_sort]
    weights_1 = weights_1_orig[indx_sort]

    indx_sort = np.argsort(arr_1b[:,i])
    x_1b = arr_1b[:,i][indx_sort]
    weights_1b = weights_1b_orig[indx_sort]

    indx_sort = np.argsort(arr_2[:,i])
    x_2 = arr_2[:,i][indx_sort]
    weights_2 = weights_2_orig[indx_sort]

    indx_sort = np.argsort(arr_3[:,i])
    x_3 = arr_3[:,i][indx_sort]
    weights_3 = weights_3_orig[indx_sort]


    y_1 = np.cumsum(weights_1)
    y_1 /= y_1[-1] # normalize
    y_2 = np.cumsum(weights_2)
    y_2 /= y_2[-1] # normalize
    y_1b = np.cumsum(weights_1b)
    y_1b /= y_1b[-1]
    y_3 = np.cumsum(weights_3)
    y_3 /= y_3[-1]
    ### plot recovered cdf
    plt.plot(x_1, y_1, "--", label="CDF, mcsampler", color=colors[i],
            linewidth=2)
    plt.plot(x_2, y_2, ":", label="CDF, mcsamplerEnsemble",
            color=colors[i], linewidth=2)
    plt.plot(x_1b, y_1b, ".", label="CDF, mcsamplerAC",
            color=colors[i], linewidth=1)
    plt.plot(x_3, y_3, "--", label="CDF, mcsamplerAV",
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



if not(opts.production):
    plt.legend()

fname = "cdf_rosenbrock.png"

print("Saving CDF figure as " + fname + "...")
if opts.production:
    plt.xlabel(r'$x$')
    plt.ylabel(r'$P(<x)$')

plt.savefig(fname)
