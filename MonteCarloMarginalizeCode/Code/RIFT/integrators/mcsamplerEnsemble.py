import sys
import math
import bisect
from collections import defaultdict

import numpy as np
from RIFT.precision import RiftFloat  # platform-portable replacement for np.float128

try:
    import cupy
    import cupyx.scipy.special
    xpy_default = cupy
    xpy_special_default = cupyx.scipy.special
    identity_convert = cupy.asnumpy
    identity_convert_togpu = cupy.asarray
    cupy_ok = True
except ImportError:
    xpy_default = np
    xpy_special_default = None
    identity_convert = lambda x: x
    identity_convert_togpu = lambda x: x
    cupy_ok = False

import itertools
import functools

import scipy.special

#from statutils import cumvar

from multiprocessing import Pool


# Mirror healpy stuff
from RIFT.integrators.mcsampler import HealPixSampler

from . import MonteCarloEnsemble as monte_carlo

__author__ = "Ben Champion"

rosDebugMessages = True

class NanOrInf(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class MCSampler(object):
    """
    Class to define a set of parameter names, limits, and probability densities.
    """

    @staticmethod
    def match_params_from_args(args, params):
        not_common = set(args) ^ set(params)
        if len(not_common) == 0:
            return True
        if all([not isinstance(i, tuple) for i in not_common]):
            return False

        to_match = [i for i in not_common if not isinstance(i, tuple)]
        against = [i for i in not_common if isinstance(i, tuple)]
        
        matched = []
        import itertools
        for i in range(2, max(list(map(len, against)))+1):
            matched.extend([t for t in itertools.permutations(to_match, i) if t in against])
        return (set(matched) ^ set(against)) == set()


    def __init__(self):
        self.ntotal = 0
        self.n = 0
        self.params = set()
        self.params_ordered = []
        self.pdf = {}
        self._pdf_norm = defaultdict(lambda: 1)
        self._rvs = {}
        self.cdf = {}
        self.cdf_inv = {}
        self.llim, self.rlim = {}, {}
        self.adaptive = []
        self._hist = {}
        self.prior_pdf = {}
        self.func = None
        self.sample_format = None
        self.curr_args = None
        self.gmm_dict ={} 
        self.integrator = None 

        self.xpy = xpy_default
        self.identity_convert = identity_convert
        self.identity_convert_togpu = identity_convert_togpu


    def clear(self):
        self.params = set()
        self.params_ordered = []
        self.pdf = {}
        self._pdf_norm = defaultdict(lambda: 1.0)
        self._rvs = {}
        self._hist = {}
        self.cdf = {}
        self.cdf_inv = {}
        self.llim = {}
        self.rlim = {}
        self.adaptive = []
        self.integrator=None

    def add_parameter(self, params, pdf=None,  cdf_inv=None, left_limit=None, right_limit=None, 
                        prior_pdf=None, adaptive_sampling=False):
        self.params.add(params)
        self.params_ordered.append(params)
        if rosDebugMessages:
            print(" mcsampler: Adding parameter ", params, " with limits ", [left_limit, right_limit])
        if isinstance(params, tuple):
            assert all([lim[0] < lim[1] for lim in zip(left_limit, right_limit)])
            if left_limit is None:
                self.llim[params] = list(float("-inf"))*len(params)
            else:
                self.llim[params] = left_limit
            if right_limit is None:
                self.rlim[params] = list(float("+inf"))*len(params)
            else:
                self.rlim[params] = right_limit
            if prior_pdf is not None:
                for p in params:
                    self.prior_pdf[p] = prior_pdf
        else:
            assert left_limit < right_limit
            if left_limit is None:
                self.llim[params] = float("-inf")
            else:
                self.llim[params] = left_limit
            if right_limit is None:
                self.rlim[params] = float("+inf")
            else:
                self.rlim[params] = right_limit
            if prior_pdf is not None:
                self.prior_pdf[params] = prior_pdf

    def evaluate(self, samples):
        # The user integrand is a host (numpy/scipy) function in general, so move
        # samples to the CPU before calling it and push the result back to the
        # active backend (cupy on GPU). This is a no-op when xpy is numpy.
        samples = self.identity_convert(samples)
        temp = []
        for index in range(len(self.curr_args)):
            temp.append(samples[:,index])
        temp_ret = self.identity_convert_togpu(self.func(*temp))
        # column vector (n,1); cupy.rot90 does not accept array-likes/lists, and
        # reshape is backend-agnostic and order-preserving (equiv. to the old
        # np.rot90([temp_ret], -1)).
        return temp_ret.reshape((-1, 1))


    def calc_pdf(self, samples):
        n, _ = samples.shape
        temp_ret = self.xpy.ones((n, 1))
        # Prior pdfs are host functions in general; evaluate them on CPU samples
        # and convert the result back to the active backend.
        samples_cpu = self.identity_convert(samples)
        for index in range(len(self.curr_args)):
            if self.curr_args[index] in self.prior_pdf:
                pdf_func = self.prior_pdf[self.curr_args[index]]
                temp_samples = samples_cpu[:,index]
                pdf_vals = self.identity_convert_togpu(pdf_func(temp_samples))
                temp_ret *= pdf_vals.reshape( temp_ret.shape)
        return temp_ret

    def setup(self,n_comp=None,**kwargs):
      integrator_func  = kwargs['integrator_func'] if "integrator_func" in kwargs  else None
      mcsamp_func  = kwargs['mcsamp_func'] if "mcsamp_func" in kwargs  else None
      proc_count = kwargs['proc_count'] if "proc_count" in kwargs else None
      direct_eval = kwargs['direct_eval'] if "direct_eval" in kwargs else False
      min_iter = kwargs['min_iter'] if "min_iter" in kwargs else 10
      max_iter = kwargs['max_iter'] if "max_iter" in kwargs else 20
      var_thresh = kwargs['var_thres'] if "var_thresh" in kwargs else 0.05
      write_to_file = kwargs['write_to_file'] if "write_to_file" in kwargs else False
      correlate_all_dims = kwargs['correlate_all_dims'] if  "correlate_all_dims" in kwargs else False
      gmm_adapt = kwargs['gmm_adapt'] if "gmm_adapt" in kwargs else None
      gmm_epsilon = kwargs['gmm_epsilon'] if "gmm_epsilon" in kwargs else None
      L_cutoff = kwargs["L_cutoff"] if "L_cutoff" in kwargs else None
      tempering_exp = kwargs["tempering_exp"] if "tempering_exp" in kwargs else 1.0
      tempering_adapt = kwargs["tempering_adapt"] if "tempering_adapt" in kwargs else False
      ess_target = kwargs["ess_target"] if "ess_target" in kwargs else None
      ess_floor = kwargs["ess_floor"] if "ess_floor" in kwargs else None
      lnw_failure_cut = kwargs["lnw_failure_cut"] if "lnw_failure_cut" in kwargs else None
      nmax = kwargs["nmax"] if "nmax" in kwargs else 1e6
      neff = kwargs["neff"] if "neff" in kwargs else 1000
      n = kwargs["n"] if "n" in kwargs else min(1000, nmax)

      self.n = n
      self.curr_args = self.params_ordered

      if 'gmm_dict' in list(kwargs.keys()):
          gmm_dict = kwargs['gmm_dict']
      else:
          gmm_dict = None
      dim = len(self.params_ordered)
      bounds=[]
      for param in self.params_ordered:
            bounds.append([self.llim[param], self.rlim[param]])
      raw_bounds = self.xpy.array(bounds)
          
      if gmm_dict is None:
            # See note in integrate(): dict keys must be host ints, not 0-d
            # cupy arrays (which are unhashable).
            bounds = {}
            for indx in np.arange(len(raw_bounds)):
                bounds[(indx,)] = raw_bounds[indx]
            bounds=raw_bounds
            if correlate_all_dims:
                gmm_dict = {tuple(range(dim)):None}
                bounds = {tuple(range(dim)): raw_bounds}
            else:
                gmm_dict = {}
                for i in range(dim):
                    gmm_dict[(i,)] = None
      else:
            bounds ={}
            for dims in gmm_dict:
                n_dims = len(dims)
                bounds_here = self.xpy.empty((n_dims,2))
                for indx in range(n_dims):
                    bounds_here[indx] = raw_bounds[dims[indx]]
                bounds[dims]=bounds_here

      self.integrator = monte_carlo.integrator(dim, bounds, gmm_dict, n_comp, n=self.n, prior=self.calc_pdf,
                         user_func=integrator_func, proc_count=proc_count,L_cutoff=L_cutoff,gmm_adapt=gmm_adapt,gmm_epsilon=gmm_epsilon,tempering_exp=tempering_exp,
                         tempering_adapt=tempering_adapt, ess_target=ess_target, ess_floor=ess_floor)

    def update_sampling_prior(self,ln_weights, n_history,tempering_exp=1,log_scale_weights=True,floor_integrated_probability=0,external_rvs=None,**kwargs):
      rvs_here = self._rvs
      if external_rvs:
        rvs_here = external_rvs

      ln_weights  = self.xpy.array(self.identity_convert(ln_weights))
      ln_weights *= tempering_exp

      gmm_dict = self.integrator.gmm_dict

      n_history_to_use = self.xpy.min([n_history, len(ln_weights), len(rvs_here[self.params_ordered[0]])] )

      sample_array = self.xpy.empty( (len(self.params_ordered), n_history_to_use))
      for indx, p in enumerate(self.params_ordered):
          sample_array[indx] = rvs_here[p][-n_history_to_use:]
      sample_array = sample_array.T

      for dim_group in gmm_dict:
            if self.integrator.gmm_adapt:
                if (dim_group in self.integrator.gmm_adapt):
                    if not(self.integrator.gmm_adapt[dim_group]):
                        continue
            new_bounds = self.xpy.empty((len(dim_group), 2))
            new_bounds = self.integrator.bounds[dim_group]
            model = self.integrator.gmm_dict[dim_group]
            temp_samples = self.xpy.empty((n_history_to_use, len(dim_group)))
            index = 0
            for dim in dim_group:
                temp_samples[:,index] = self.identity_convert(sample_array[:,dim])
                index += 1

            if self.xpy.any(self.xpy.isnan(ln_weights)):
                ok_indx = ~self.xpy.isnan(ln_weights)
                temp_samples = temp_samples[ok_indx]
                ln_weights = ln_weights[ok_indx]
            
            if model is None:
                if isinstance(self.integrator.n_comp, int) and self.integrator.n_comp != 0:
                    model = GMM.gmm(self.integrator.n_comp, new_bounds,epsilon=self.integrator.gmm_epsilon)
                    model.fit(temp_samples, log_sample_weights=ln_weights)
                elif isinstance(self.integrator.n_comp, dict) and self.integrator.n_comp[dim_group] != 0:
                    model = GMM.gmm(self.integrator.n_comp[dim_group], new_bounds,epsilon=self.integrator.gmm_epsilon)
                    model.fit(temp_samples, log_sample_weights=ln_weights)
            else:
                model.update(temp_samples, log_sample_weights=ln_weights)
            self.integrator.gmm_dict[dim_group] = model


    def draw_simplified(self,n,*args,**kwargs):
        n_samples = int(n)
        self.integrator.n = n

        if len(args) == 0:
            args = self.params
        n_params = int(len(args))

        save_no_samples= False
        if 'save_no_samples' in list(kwargs.keys()):
            save_no_samples = kwargs['save_no_samples']

        rv = self.xpy.empty((n_params, n_samples), dtype=np.float64)
        joint_p_s = self.xpy.ones(n_samples, dtype=np.float64)
        joint_p_prior = self.xpy.ones(n_samples, dtype=np.float64)

        self.integrator._sample()
        for indx, p in enumerate(self.params_ordered):
            if isinstance(type(rv), type(self.integrator.sample_array)):
                rv[indx,:]  = self.integrator.sample_array[:,indx]
            else:
                rv[indx,:]  = self.identity_convert_togpu(self.integrator.sample_array[:,indx])
        joint_p_s = self.integrator.sampling_prior_array
        joint_p_prior = self.calc_pdf(rv.T).flatten()

        return joint_p_s, joint_p_prior, rv


    def integrate_log(self, func, *args,**kwargs):
        args_passed = {}
        args_passed.update(kwargs)
        args_passed['use_lnL']=True
        args_passed['return_lnI']=True
        return integrate(func, *args, args_passed)

    def integrate(self, func, *args,**kwargs):
        nmax = kwargs["nmax"] if "nmax" in kwargs else 1e6
        neff = kwargs["neff"] if "neff" in kwargs else 1000
        n = kwargs["n"] if "n" in kwargs else min(1000, nmax)
        n_comp = kwargs["n_comp"] if "n_comp" in kwargs else 1
        if 'gmm_dict' in list(kwargs.keys()):
            gmm_dict = kwargs['gmm_dict']
        else:
            gmm_dict = None
        reflect = kwargs['reflect'] if "reflect" in kwargs else False
        integrator_func  = kwargs['integrator_func'] if "integrator_func" in kwargs  else None
        mcsamp_func  = kwargs['mcsamp_func'] if "mcsamp_func" in kwargs  else None
        proc_count = kwargs['proc_count'] if "proc_count" in kwargs else None
        direct_eval = kwargs['direct_eval'] if "direct_eval" in kwargs else False
        min_iter = kwargs['min_iter'] if "min_iter" in kwargs else 10
        max_iter = kwargs['max_iter'] if "max_iter" in kwargs else 20
        var_thresh = kwargs['var_thres'] if "var_thresh" in kwargs else 0.05
        write_to_file = kwargs['write_to_file'] if "write_to_file" in kwargs else False
        correlate_all_dims = kwargs['correlate_all_dims'] if  "correlate_all_dims" in kwargs else False
        gmm_adapt = kwargs['gmm_adapt'] if "gmm_adapt" in kwargs else None
        gmm_epsilon = kwargs['gmm_epsilon'] if "gmm_epsilon" in kwargs else None
        L_cutoff = kwargs["L_cutoff"] if "L_cutoff" in kwargs else None
        tempering_exp = kwargs["tempering_exp"] if "tempering_exp" in kwargs else 1.0
        # --adapt-adapt: ESS-self-tuned refit exponent (previously silently
        # dropped by this sampler; only mcsampler/mcsamplerGPU honored it)
        tempering_adapt = kwargs["tempering_adapt"] if "tempering_adapt" in kwargs else False
        ess_target = kwargs["ess_target"] if "ess_target" in kwargs else None
        ess_floor = kwargs["ess_floor"] if "ess_floor" in kwargs else None
        lnw_failure_cut = kwargs["lnw_failure_cut"] if "lnw_failure_cut" in kwargs else None

        max_err = kwargs["max_err"] if "max_err" in kwargs else 10

        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        super_verbose = kwargs["super_verbose"] if "super_verbose" in kwargs else False
        dict_return_q = kwargs["dict_return"] if "dict_return" in kwargs else False

        tripwire_fraction = kwargs["tripwire_fraction"] if "tripwire_fraction" in kwargs else 2
        tripwire_epsilon = kwargs["tripwire_epsilon"] if "tripwire_epsilon" in kwargs else 0.001

        use_lnL = kwargs["use_lnL"] if "use_lnL" in kwargs else False 
        return_lnI = kwargs["return_lnI"] if "return_lnI" in kwargs else False

        bFairdraw  = kwargs["igrand_fairdraw_samples"] if "igrand_fairdraw_samples" in kwargs else False
        n_extr = kwargs["igrand_fairdraw_samples_max"] if "igrand_fairdraw_samples_max" in kwargs else None

        self.func = func
        self.curr_args = args
        if n_comp is None:
            print('No n_comp given, assuming 1 component per dimension')
            n_comp = 1
        dim = len(args)
        bounds=[]
        for param in args:
            bounds.append([self.llim[param], self.rlim[param]])
        raw_bounds = self.xpy.array(bounds)

        bounds=None
        if gmm_dict is None:
            # NOTE: dim-group / bounds dict keys must be *host* integers. Building
            # them with self.xpy.arange would produce unhashable 0-d cupy arrays
            # on GPU; keep this bookkeeping on the CPU with range/np.arange.
            bounds = {}
            for indx in np.arange(len(raw_bounds)):
                bounds[(indx,)] = raw_bounds[indx]
            bounds=raw_bounds
            if correlate_all_dims:
                gmm_dict = {tuple(range(dim)):None}
                bounds = {tuple(range(dim)): raw_bounds}
            else:
                gmm_dict = {}
                for i in range(dim):
                    gmm_dict[(i,)] = None
        else:
            bounds ={}
            for dims in gmm_dict:
                n_dims = len(dims)
                bounds_here = self.xpy.empty((n_dims,2))
                for indx in range(n_dims):
                    bounds_here[indx] = raw_bounds[dims[indx]]
                bounds[dims]=bounds_here

        integrator = monte_carlo.integrator(dim, bounds, gmm_dict, n_comp, n=n, prior=self.calc_pdf,
                         user_func=integrator_func, proc_count=proc_count,L_cutoff=L_cutoff,gmm_adapt=gmm_adapt,gmm_epsilon=gmm_epsilon,tempering_exp=tempering_exp,
                         tempering_adapt=tempering_adapt, ess_target=ess_target, ess_floor=ess_floor)
        if not direct_eval:
            func = self.evaluate
        if use_lnL:
            print(" ==> input assumed as lnL ")
        if return_lnI:
            print(" ==> internal calculations and return values are lnI ")
        integrator.integrate(func, min_iter=min_iter, max_iter=max_iter, var_thresh=var_thresh, neff=neff, nmax=nmax,max_err=max_err,verbose=verbose,progress=super_verbose,tripwire_fraction=tripwire_fraction,tripwire_epsion=tripwire_epsilon,use_lnL=use_lnL,return_lnI=return_lnI,lnw_failure_cut=lnw_failure_cut)

        self.n = int(integrator.n)
        self.ntotal = int(integrator.ntotal)
        integral = integrator.integral
        print("Result ",integrator.scaled_error_squared, integrator.integral)
        if not(return_lnI):
            error_squared = integrator.scaled_error_squared * self.xpy.exp(integrator.log_error_scale_factor)/ (self.ntotal/self.n)
        else:
            error_squared = integrator.scaled_error_squared  - self.xpy.log(self.ntotal/self.n)
        eff_samp = integrator.eff_samp
        sample_array = integrator.cumulative_samples
        if not(return_lnI):
            value_array = self.xpy.exp(integrator.cumulative_values)
        else:
            value_array = integrator.cumulative_values
        p_array = integrator.cumulative_p_s
        prior_array = integrator.cumulative_p

        if mcsamp_func is not None:
            mcsamp_func(self, integrator)

        # Store sample history on the host so downstream (CPU) consumers --
        # weights, CDFs, posterior plots -- work regardless of backend.
        index = 0
        for param in args:
            self._rvs[param] = self.identity_convert(sample_array[:,index])
            index += 1
        self._rvs['joint_prior'] = self.identity_convert(prior_array)
        self._rvs['joint_s_prior'] = self.identity_convert(p_array)
        self._rvs['integrand'] = self.identity_convert(value_array)

        if bFairdraw and not(n_extr is None):
           # scalars: use Python min on floats.  self.xpy.min([list]) fails on cupy
           # (cupy.min has no list overload -> "'list' object has no attribute 'min'"),
           # which crashed the GMM sampler's fairdraw export on GPU.
           n_extr = int(min(float(n_extr), 1.5*float(eff_samp), 1.5*float(neff)))
           print(" Fairdraw size : ", n_extr)
           if return_lnI:
               ln_wt =  integrator.cumulative_values
           else:
               ln_wt = self.xpy.log(value_array)
           ln_wt += self.xpy.log(prior_array/p_array)
           ln_wt += - scipy.special.logsumexp(self.identity_convert(ln_wt))
           wt = self.xpy.exp(ln_wt)
           if n_extr < len(value_array):
               indx_list = self.identity_convert(self.xpy.random.choice(self.xpy.arange(len(wt)), size=n_extr,replace=True,p=wt))
               for key in list(self._rvs.keys()):
                   if isinstance(key, tuple):
                       self._rvs[key] = self._rvs[key][:,indx_list]
                   else:
                       self._rvs[key] = self._rvs[key][indx_list]

        dict_return = {}
        if dict_return_q:
            dict_return["integrator"] = integrator

        if write_to_file:
            dat_out = self.xpy.c_[sample_array, value_array, p_array]
            np.savetxt('mcsampler_data.txt', self.identity_convert(dat_out),
                        header=" ".join(['sample_array', 'value_array', 'p_array']))

        # Return scalars on the host so callers can do plain numpy arithmetic
        # (np.sqrt, np.log, np.array([...])) on the results.
        return self.identity_convert(integral), self.identity_convert(error_squared), self.identity_convert(eff_samp), dict_return


def inv_uniform_cdf(a, b, x):
    return (b-a)*x+a

def gauss_samp(mu, std, x):
    return 1.0/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mu)**2/2/std**2)

def gauss_samp_withfloor(mu, std, myfloor, x):
    return 1.0/np.sqrt(2*np.pi*std**2)*np.exp(-(x-mu)**2/2/std**2) + myfloor

gauss_samp_withfloor_vector = np.vectorize(gauss_samp_withfloor,otypes=[np.float64])


def q_samp_vector(qmin,qmax,x):
    scale = 1./(1+qmin) - 1./(1+qmax)
    return 1/np.power((1+x),2)/scale
def q_cdf_inv_vector(qmin,qmax,x):
    return np.array((qmin + qmax*qmin + qmax*x - qmin*x)/(1 + qmax - qmax*x + qmin*x),dtype=RiftFloat)

def M_samp_vector(Mmin,Mmax,x):
    scale = 2./(Mmax**2 - Mmin**2)
    return x*scale


def cos_samp(x):
        return np.sin(x)/2

def dec_samp(x):
        return np.sin(x+np.pi/2)/2

cos_samp_vector = np.vectorize(cos_samp,otypes=[np.float64])
dec_samp_vector = np.vectorize(dec_samp,otypes=[np.float64])
def cos_samp_cdf_inv_vector(p):
    return np.arccos( 2*p-1)
def dec_samp_cdf_inv_vector(p):
    return np.arccos(2*p-1) - np.pi/2


def pseudo_dist_samp(r0,r):
        return r*r*np.exp( - (r0/r)*(r0/r)/2. + r0/r)+0.01

pseudo_dist_samp_vector = np.vectorize(pseudo_dist_samp,otypes=[np.float64])

def delta_func_pdf(x_0, x):
    return 1.0 if x == x_0 else 0.0

delta_func_pdf_vector = np.vectorize(delta_func_pdf, otypes=[np.float64])

def delta_func_samp(x_0, x):
    return x_0

delta_func_samp_vector = np.vectorize(delta_func_samp, otypes=[np.float64])

class HealPixSampler(object):
    @staticmethod
    def thph2decra(th, ph):
        return np.pi/2-th, ph

    @staticmethod
    def decra2thph(dec, ra):
        return np.pi/2-dec, ra

    def __init__(self, skymap, massp=1.0):
        self.skymap = skymap
        self._massp = massp
        self.renormalize()

    @property
    def massp(self):
        return self._massp

    @massp.setter
    def massp(self, value):
        assert 0 <= value <= 1
        self._massp = value
        norm = self.renormalize()

    def renormalize(self):
        res = healpy.npix2nside(len(self.skymap))
        self.pdf_sorted = sorted([(p, i) for i, p in enumerate(self.skymap)], reverse=True)
        self.valid_points_decra = []
        cdf, np_count = 0, 0
        for p, i in self.pdf_sorted:
            if p == 0:
                continue
            self.valid_points_decra.append(HealPixSampler.thph2decra(*healpy.pix2ang(res, i)))
            cdf += p
            if cdf > self._massp:
                break
        self._renorm = cdf
        self.valid_points_hist = None
        return self._renorm

    def __expand_valid(self, min_p=1e-7):
        if self._massp == 1.0:
            min_p = min(min_p, max(self.skymap))
        else:
            min_p = self.pseudo_pdf(*self.valid_points_decra[-1])

        self.valid_points_hist = []
        ns = healpy.npix2nside(len(self.skymap))

        self._renorm = 0
        for i, v in enumerate(self.skymap >= min_p):
            self._renorm += self.skymap[i] if v else 0

        for pt in self.valid_points_decra:
            th, ph = HealPixSampler.decra2thph(pt[0], pt[1])
            pix = healpy.ang2pix(ns, th, ph)
            if self.skymap[pix] < min_p:
                continue
            self.valid_points_hist.extend([pt]*int(round(self.pseudo_pdf(*pt)/min_p)))
        self.valid_points_hist = np.array(self.valid_points_hist).T

    def pseudo_pdf(self, dec_in, ra_in):
        th, ph = HealPixSampler.decra2thph(dec_in, ra_in)
        res = healpy.npix2nside(len(self.skymap))
        return self.skymap[healpy.ang2pix(res, th, ph)]/self._renorm

    def pseudo_cdf_inverse(self, dec_in=None, ra_in=None, ndraws=1, stype='vecthist'):
        if ra_in is not None:
            ndraws = len(ra_in)
        if ra_in is None:
            ra_in, dec_in = np.zeros((2, ndraws))

        if stype == 'rejsamp':
            ceiling = max(self.skymap)
            i, np_count = 0, len(self.valid_points_decra)
            while i < len(ra_in):
                rnd_n = np.random.randint(0, np_count)
                trial = np.random.uniform(0, ceiling)
                if trial <= self.pseudo_pdf(*self.valid_points_decra[rnd_n]):
                    dec_in[i], ra_in[i] = self.valid_points_decra[rnd_n]
                    i += 1
            return np.array([dec_in, ra_in])
        elif stype == 'vecthist':
            if self.valid_points_hist is None:
                self.__expand_valid()
            np_count = self.valid_points_hist.shape[1]
            rnd_n = np.random.randint(0, np_count, len(ra_in))
            dec_in, ra_in = self.valid_points_hist[:,rnd_n]
            return np.array([dec_in, ra_in])
        else:
            raise ValueError("%s is not a recgonized sampling type" % stype)

pseudo_dist_samp_vector = np.vectorize(pseudo_dist_samp,otypes=[np.float64])


def sanityCheckSamplerIntegrateUnity(sampler,*args,**kwargs):
        return sampler.integrate(lambda *args: 1,*args,**kwargs)

def convergence_test_MostSignificantPoint(pcut, rvs, params):
    weights = rvs["weights"]
    indxmax = np.argmax(weights)
    wtSum = np.sum(weights)
    return  weights[indxmax]/wtSum < pcut

def convergence_test_NormalSubIntegrals(ncopies, pcutNormalTest, sigmaCutRelativeErrorThreshold, rvs, params):
    weights = rvs["integrand"]* rvs["joint_prior"]/rvs["joint_s_prior"]
    igrandValues = np.zeros(ncopies)
    len_part = int(len(weights)/ncopies)
    for indx in np.arange(ncopies):
        igrandValues[indx] = np.log(np.mean(weights[indx*len_part:(indx+1)*len_part]))
    igrandValues= np.sort(igrandValues)
    valTest = stats.normaltest(igrandValues)[1]
    igrandSigma = (np.std(igrandValues))/np.sqrt(ncopies)
    print(" Test values on distribution of log evidence:  (gaussianity p-value; standard deviation of ln evidence) ", valTest, igrandSigma)
    print(" Ln(evidence) sub-integral values, as used in tests  : ", igrandValues)
    return valTest> pcutNormalTest and igrandSigma < sigmaCutRelativeErrorThreshold

from . import gaussian_mixture_model as GMM
def create_wide_single_component_prior(bounds, epsilon=None):
    model = GMM.gmm(1, bounds, epsilon=epsilon)
    widths = np.array([ bounds[k][1] - bounds[k][0] for k in np.arange(len(bounds))])  
    model.means = [np.array([np.mean(bounds[k]) for k in np.arange(len(bounds))]) ]
    model.covariances = [np.diag( widths**2)]
    model.weights = [1]
    model.adapt = [False]
    model.d = len(bounds)
