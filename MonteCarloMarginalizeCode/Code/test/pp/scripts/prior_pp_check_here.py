#! /usr/bin/env python
# prior_pp_check:  produces PP-like output from injection files, compares to prior.
# Note currently does NOT change the distance prior to be cosmological

import argparse
import RIFT.lalsimutils as lalsimutils
import numpy as np
import glob
import RIFT.likelihood.priors_utils as priors_utils
import RIFT.integrators.mcsampler  as mcsampler # for quick CDF inverse
import functools
import lal

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir",default='my_set_16s',type=str)
parser.add_argument("--mc-range",default='[10,20]',type=str)
parser.add_argument("--eta-range", default='[0.2,0.2499999]',type=str)
parser.add_argument("--d-max",default=750,type=float)
parser.add_argument("--parameter",action='append',type=str)
parser.add_argument("--plot-sanity",action='store_true')
opts = parser.parse_args()
mc_min,mc_max = eval(opts.mc_range)
eta_min,eta_max = eval(opts.eta_range)
q_min = (1-np.sqrt(1-4*eta_min))/(1+np.sqrt(1-4*eta_min))

import scipy.integrate
#def q_prior(x,nm=1):
#    return 1./(1+x)**2/nm
#nm_q = scipy.integrate.quad(q_prior,1./20, 1)[0]
def delta_mc_prior(x,norm_factor=1.44):
    """
    delta_mc = sqrt(1-4eta)  <-> eta = 1/4(1-delta^2)
    Transform the prior above
    """
    eta_here = 0.25*(1 -x*x)
    return 2./np.power(eta_here, 6./5.)/norm_factor


# Distance prior: use mcsampler to make cdf inverse
sampler = mcsampler.MCSampler()
nm = priors_utils.dist_prior_pseudo_cosmo_eval_norm(0,opts.d_max)
dist_prior_pdf =functools.partial( priors_utils.dist_prior_pseudo_cosmo, nm=nm)
sampler.add_parameter('dist', pdf=dist_prior_pdf, cdf_inv= None, left_limit=0.0, right_limit=opts.d_max)
#q_prior_pdf = functools.partial( q_prior, nm=nm_q)
sampler.add_parameter('delta_mc',pdf=delta_mc_prior, left_limit=0, right_limit=np.sqrt(1-4*eta_min))

npts = 10000
# Generate prior data
P_list  =[]
P_out = lalsimutils.ChooseWaveformParams()
for name in ['m1','m2', 's1x','s1y','s1z','s2x', 's2y','s2z', 'theta', 'phi','psi','phiref','dist', 'incl']:
    setattr(P_out,name, np.zeros(npts))

random_mc = np.sqrt(mc_min**2 + np.random.uniform(size=2*npts)*(mc_max**2-mc_min**2))
random_dL = sampler.cdf_inv['dist'](np.random.uniform(size=2*npts))
random_delta = sampler.cdf_inv['delta_mc'](np.random.uniform(size=2*npts))
#print(random_mc,random_dL, random_q)
n_kept = 0
for indx in np.arange(2*npts):
    if n_kept >= npts:
        continue
    P = lalsimutils.ChooseWaveformParams()
    P.randomize() # randomize spins
    P.assign_param('mc', random_mc[indx])
    P.assign_param('delta_mc',random_delta[indx])
    P.assign_param('dist',random_dL[indx])
    if P.m2 > 1 and n_kept < npts:
        P_list.append(P)
        P_out.m1[n_kept] = P.m1
        P_out.m2[n_kept] = P.m2 
        P_out.s1x[n_kept] = P.s1x
        P_out.s1y[n_kept] = P.s1y
        P_out.s1z[n_kept] = P.s1z
        P_out.s2x[n_kept] = P.s2x
        P_out.s2y[n_kept] = P.s2y
        P_out.s2z[n_kept] = P.s2z
        P_out.theta[n_kept] = P.theta
        P_out.phi[n_kept] = P.phi
        P_out.psi[n_kept] = P.psi
        P_out.phiref[n_kept] = P.phiref
        P_out.dist[n_kept] = P.dist
        P_out.incl[n_kept] = P.incl
        n_kept += 1




# Pull in MDC files
fname = 'mdc.xml.gz'
P_list = lalsimutils.xml_to_ChooseWaveformParams_array(fname)
mvals = np.zeros((len(P_list),4))
indx_name =0
for indx, P in enumerate(P_list):
    dat_out =np.zeros(len(opts.parameter))
    if opts.plot_sanity:
        mvals[indx_name] = [P.extract_param('mc')/lal.MSUN_SI,P.extract_param('q'),P.m1/lal.MSUN_SI,P.m2/lal.MSUN_SI]
        if P.m1<P.m2:
            mvals[indx_name] =  [P.extract_param('mc')/lal.MSUN_SI,1./P.extract_param('q'),P.m2/lal.MSUN_SI,P.m1/lal.MSUN_SI]
        indx_name+=1
    indx = 0
    for name in opts.parameter:
        fac =1
        if name in ['mc']:
            fac = lal.MSUN_SI
        if name in ['dist']:
            fac = lal.PC_SI*1e6
        dat_out[indx] = np.sum(P_out.extract_param(name) < P.extract_param(name)/fac)*1.0/npts
        #print(P.extract_param(name),P_out.extract_param(name))
        indx+=1
    print(' '.join(map(str,dat_out)))

if opts.plot_sanity:
    from matplotlib import pyplot as plt
    plt.scatter(mvals[:,0], mvals[:,1])
    plt.savefig("prior_fig_mc_q.png")
    plt.clf()
    plt.scatter(mvals[:,2], mvals[:,3])
    xvals = np.linspace(np.min(mvals[:,3]), np.max(mvals[:,2]), 100)
    yvals = xvals
    plt.plot(xvals,yvals)
    yvals =  q_min*xvals
    plt.plot(xvals, yvals)
    plt.savefig("prior_fig_m1_m2.png")
