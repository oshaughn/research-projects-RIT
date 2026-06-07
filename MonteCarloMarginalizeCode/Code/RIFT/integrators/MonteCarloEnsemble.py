# -*- coding: utf-8 -*-
'''
Monte Carlo Integrator
----------------------
Perform an adaptive monte carlo integral.
'''
from __future__ import print_function
import numpy as np
from . import gaussian_mixture_model as GMM
import traceback
import time
from scipy.special import logsumexp

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

regularize_log_scale = 1e-64  # before taking np.log, add this, so we don't propagate infinities


def _xpy_logsumexp(a, axis=None):
    """Portable logsumexp (mirror of gaussian_mixture_model._xpy_logsumexp).

    cupyx.scipy.special.logsumexp is absent in the CUDA 10.2 cupy build needed
    for older (sm_30) GPUs, so implement the reduction with cupy primitives and
    fall back to scipy on CPU.
    """
    if cupy_ok:
        a = cupy.asarray(a)
        a_max = cupy.amax(a, axis=axis, keepdims=True)
        a_max = cupy.where(cupy.isfinite(a_max), a_max, cupy.zeros_like(a_max))
        out = cupy.log(cupy.sum(cupy.exp(a - a_max), axis=axis, keepdims=True)) + a_max
        if axis is None:
            return out.reshape(())
        return cupy.squeeze(out, axis=axis)
    return logsumexp(a, axis=axis)


try:
    from multiprocess import Pool
except:
    print('no multiprocess')


class integrator:
    '''
    Class to iteratively perform an adaptive Monte Carlo integral where the integrand
    is a combination of one or more Gaussian curves, in one or more dimensions.

    Parameters
    ----------
    d : int
        Total number of dimensions.

    bounds : dictionary with array bounds, with keys matching gmm_dict
        Limits of integration, where each row represents [left_lim, right_lim]
        for its corresponding dimension.

    gmm_dict : dict
        Dictionary where each key is a tuple of one or more dimensions
        that are to be modeled together. If the integrand has strong correlations between
        two or more dimensions, they should be grouped. Each value is by default initialized
        to None, and is replaced with the GMM object for its dimension(s).

    n_comp : int or {tuple:int}
        The number of Gaussian components per group of dimensions. If its type is int,
        this number of components is used for all dimensions. If it is a dict, it maps
        each key in gmm_dict to an integer number of mixture model components.

    n : int
        Number of samples per iteration

    prior : function
        Function to evaluate prior for samples

    user_func : function
        Function to run each iteration

    L_cutoff : float
        Likelihood cutoff for samples to store

    use_lnL : bool
        Whether or not lnL or L will be returned by the integrand
    '''

    def __init__(self, d, bounds, gmm_dict, n_comp, n=None, prior=None,
                user_func=None, proc_count=None, L_cutoff=None, use_lnL=False,return_lnI=False,gmm_adapt=None,gmm_epsilon=None,tempering_exp=1,temper_log=False,lnw_failure_cut=None):
        # if 'return_lnI' is active, 'integral' holds the *logarithm* of the integral.
        # user-specified parameters
        self.d = d
        self.bounds = bounds
        self.gmm_dict = gmm_dict
        self.gmm_adapt = gmm_adapt
        self.gmm_epsilon= gmm_epsilon
        self.n_comp = n_comp
        self.user_func=user_func
        self.prior = prior
        self.proc_count = proc_count
        self.use_lnL = use_lnL
        self.return_lnI = return_lnI
        
        self.xpy = xpy_default
        self.identity_convert = identity_convert
        self.identity_convert_togpu = identity_convert_togpu
        
        # constants
        self.t = 0.02 # percent estimated error threshold
        if n is None:
            self.n = int(5000 * self.d) # number of samples per batch
        else:
            self.n = int(n)
        self.ntotal = 0
        # integrator object parameters
        self.sample_array = None
        self.value_array = None
        self.sampling_prior_array = None
        self.prior_array = None
        self.scaled_error_squared = 0.
        if self.return_lnI:
            self.scaled_error_squared = None
        if not(lnw_failure_cut):
            self.terrible_lnw_threshold = -1000
        else:
            self.terrible_lnw_threshold = lnw_failure_cut
        self.log_error_scale_factor = 0.
        self.integral = 0
        if self.return_lnI:
            self.integral=None
        self.eff_samp = 0
        self.iterations = 0 # for weighted averages and count
        self.log_scale_factor = 0  # to handle very large answers
        self.max_value = float('-inf') # for calculating eff_samp
        self.total_value = 0 # for calculating eff_samp
        if self.return_lnI:
            self.total_value = None
        self.n_max = float('inf')
        # saved values
        self.cumulative_samples = self.xpy.empty((0, d))
        self.cumulative_values = self.xpy.empty(0)
        self.cumulative_p = self.xpy.empty(0)
        self.cumulative_p_s = self.xpy.empty(0)
        self.tempering_exp=tempering_exp
        self.temper_log=temper_log
        if L_cutoff is None:
            self.L_cutoff = -1
        else:
            self.L_cutoff = L_cutoff
        
    def _calculate_prior(self):
        if self.prior is None:
            self.prior_array = self.xpy.ones(self.n)
        else:
            self.prior_array = self.prior(self.sample_array).flatten()

    def _sample(self):
        self.sampling_prior_array = self.xpy.ones(self.n)
        self.sample_array = self.xpy.empty((self.n, self.d))
        for dim_group in self.gmm_dict: # iterate over grouped dimensions
            # create a matrix of the left and right limits for this set of dimensions
            new_bounds = self.xpy.empty((len(dim_group), 2))
            new_bounds = self.bounds[dim_group]
            if len(new_bounds.shape) < 2:
                new_bounds = self.xpy.array([new_bounds])
            model = self.gmm_dict[dim_group]
            if model is None:
                # sample uniformly for this group of dimensions
                llim = new_bounds[:,0]
                rlim = new_bounds[:,1]
                temp_samples = self.xpy.random.uniform(llim, rlim, (self.n, len(dim_group)))
                # update responsibilities
                vol = self.xpy.prod(rlim - llim)
                self.sampling_prior_array *= 1.0 / vol
            else:
                # sample from the gmm
                temp_samples = model.sample(self.n)
                # update responsibilities
                self.sampling_prior_array *= model.score(temp_samples)
            index = 0
            for dim in dim_group:
                self.sample_array[:,dim] = temp_samples[:,index]
                index += 1

    def _train(self):
        sample_array, value_array, sampling_prior_array = self.xpy.copy(self.sample_array), self.xpy.copy(self.value_array), self.xpy.copy(self.sampling_prior_array)
        if self.use_lnL:
            lnL = value_array
        else:
            lnL = self.xpy.log(value_array+regularize_log_scale)
        
        log_weights = self.tempering_exp*lnL + self.xpy.log(self.prior_array) - sampling_prior_array
        if self.temper_log:
            log_weights = self.xpy.log(self.xpy.maximum(lnL,1e-5))
            
        for dim_group in self.gmm_dict: # iterate over grouped dimensions
            if self.gmm_adapt:
                if (dim_group in self.gmm_adapt):
                    if not(self.gmm_adapt[dim_group]):
                        continue
            new_bounds = self.xpy.empty((len(dim_group), 2))
            new_bounds = self.bounds[dim_group]
            model = self.gmm_dict[dim_group]
            temp_samples = self.xpy.empty((self.n, len(dim_group)))
            index = 0
            for dim in dim_group:
                temp_samples[:,index] = sample_array[:,dim]
                index += 1
            if model is None:
                if isinstance(self.n_comp, int) and self.n_comp != 0:
                    model = GMM.gmm(self.n_comp, new_bounds,epsilon=self.gmm_epsilon)
                    model.fit(temp_samples, log_sample_weights=log_weights)
                elif isinstance(self.n_comp, dict) and self.n_comp[dim_group] != 0:
                    model = GMM.gmm(self.n_comp[dim_group], new_bounds,epsilon=self.gmm_epsilon)
                    model.fit(temp_samples, log_sample_weights=log_weights)
            else:
                model.update(temp_samples, log_sample_weights=log_weights)
            try:
                model.score(temp_samples[:5])
                self.gmm_dict[dim_group] = model
            except:
                print(" Failed to update ", dim_group)


    def _calculate_results(self):
        if self.use_lnL:
            lnL = self.xpy.copy(self.value_array)
        else:
            lnL = self.xpy.log(self.value_array+regularize_log_scale)
        mask = self.xpy.ones(lnL.shape,dtype=bool)
        if not(self.L_cutoff is None):
            if not(self.xpy.isinf(self.L_cutoff)):
                mask = lnL > (self.xpy.log(self.L_cutoff) if self.L_cutoff > 0 else -self.xpy.inf)
        lnL = lnL[mask]
        prior = self.prior_array[mask]
        sampling_prior = self.sampling_prior_array[mask]
        
        self.cumulative_samples = self.xpy.append(self.cumulative_samples, self.sample_array[mask], axis=0)
        self.cumulative_values = self.xpy.append(self.cumulative_values, lnL, axis=0)
        self.cumulative_p = self.xpy.append(self.cumulative_p, prior, axis=0)
        self.cumulative_p_s = self.xpy.append(self.cumulative_p_s, sampling_prior, axis=0)
        
        log_weights = lnL + self.xpy.log(prior) - self.xpy.log(sampling_prior)
        if self.xpy.any(self.xpy.isnan(log_weights)):
            print(" NAN weight ")
            raise ValueError
        if self.terrible_lnw_threshold: 
            if self.xpy.max(log_weights) < self.terrible_lnw_threshold:
                print(" TERRIBLE FIT ")
                raise ValueError

        log_scale_factor = self.xpy.max(log_weights)
        if not(self.return_lnI):
            scale_factor = self.xpy.exp(log_scale_factor)
            log_weights -= log_scale_factor
            summed_vals = scale_factor * self.xpy.sum(self.xpy.exp(log_weights))
            integral_value = summed_vals / self.n
        
            scaled_error_squared = self.xpy.var(self.xpy.exp(log_weights)) / self.n
            log_error_scale_factor = 2. * log_scale_factor
        
            self.integral = (self.iterations * self.integral + integral_value) / (self.iterations + 1)
            self.scaled_error_squared = (self.iterations * self.xpy.exp(self.log_error_scale_factor - log_error_scale_factor) * self.scaled_error_squared + scaled_error_squared) / (self.iterations + 1)
            self.log_error_scale_factor = log_error_scale_factor
        
            self.total_value += summed_vals
            self.max_value = self.xpy.maximum(scale_factor, self.max_value)
            self.eff_samp = self.total_value / self.max_value
        else:
            log_sum_weights = _xpy_logsumexp(log_weights)

            log_integral_here = log_sum_weights - self.xpy.log(self.n)
            if not(self.integral ):
                self.integral = log_integral_here
                self.total_value = log_sum_weights
                self.max_value = log_scale_factor
            else:
                self.integral = _xpy_logsumexp([ self.integral + self.xpy.log(self.iterations), log_integral_here]) - self.xpy.log(self.iterations+1)
                self.total_value = _xpy_logsumexp([self.total_value, log_sum_weights])
                self.max_value = self.xpy.maximum(self.max_value, self.xpy.max(log_weights))
            self.eff_samp = self.xpy.exp(self.total_value - (self.max_value  ))

            tmp_max = self.xpy.max(log_weights)
            log_scaled_error_squared = self.xpy.log(self.xpy.var(self.xpy.exp(log_weights - tmp_max))) + 2*tmp_max - self.xpy.log(self.n)
            if not(self.scaled_error_squared):
                self.scaled_error_squared = log_scaled_error_squared
            else:
                self.scaled_error_squared = _xpy_logsumexp([ self.scaled_error_squared + self.xpy.log(self.iterations), log_scaled_error_squared]) - self.xpy.log(self.iterations+1)

    def _reset(self):
        for k in self.gmm_dict:
            self.gmm_dict[k] = None
        

    def integrate(self, func, min_iter=10, max_iter=20, var_thresh=0.0, max_err=10,
            neff=float('inf'), nmax=None, progress=False, epoch=None,verbose=True,force_no_adapt=False,use_lnL=False,return_lnI=False,**kwargs):
        n_adapt = int(kwargs["n_adapt"]) if "n_adapt" in kwargs else 100
        tripwire_fraction = kwargs["tripwire_fraction"] if "tripwire_fraction" in kwargs else 2
        tripwire_epsilon = kwargs["tripwire_epsilon"] if "tripwire_epsilon" in kwargs else 0.001
        self.use_lnL = use_lnL
        self.return_lnI = return_lnI

        err_count = 0
        cumulative_eval_time = 0
        adapting=True
        if nmax is None:
            nmax = max_iter * self.n
        while self.iterations < max_iter and self.ntotal < nmax and self.eff_samp < neff:
            if (self.ntotal > nmax*tripwire_fraction) and (self.eff_samp < 1+tripwire_epsilon):
                print(" Tripwire: n_eff too low ")
                raise Exception("Tripwire on n_eff")

            if force_no_adapt or self.iterations >= n_adapt:
                adapting=False
            if err_count >= max_err:
                print('Exiting due to errors...')
                break
            try:
                self._sample()
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                break
            except Exception as e:
                print(traceback.format_exc())
                print('Error sampling, resetting...')
                err_count += 1
                self._reset()
                continue
            t1 = time.time()
            if self.proc_count is None:
                # Ensure input to func is numpy for CPU-based user functions if needed, 
                # but the user is encouraged to support xpy.
                self.value_array = func(self.xpy.copy(self.sample_array)).flatten()
            else:
                split_samples = self.xpy.array_split(self.sample_array, self.proc_count)
                p = Pool(self.proc_count)
                self.value_array = self.xpy.concatenate(p.map(func, split_samples), axis=0)
                p.close()
            cumulative_eval_time += time.time() - t1
            self._calculate_prior()
            try:
                self._calculate_results()
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                break
            except Exception as e:
                print(traceback.format_exc())
                print('Error calculating results, resetting...')
                err_count += 1
                self._reset()
                continue
            self.iterations += 1
            self.ntotal += self.n
            testval = self.scaled_error_squared
            if not(self.return_lnI):
                testval = self.xpy.log(self.scaled_error_squared) + self.log_error_scale_factor
            if self.iterations >= min_iter and testval < self.xpy.log(var_thresh):
                break
            try:
                if adapting:
                    self._train()
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                break
            except Exception as e:
                print(traceback.format_exc())
                print('Error training, resetting...')
                err_count += 1
                self._reset()
            if self.user_func is not None:
                self.user_func(self)
            if progress:
                for k in self.gmm_dict:
                    if self.gmm_dict[k] is not None:
                        self.gmm_dict[k].print_params()
            if epoch is not None and self.iterations % epoch == 0:
                self._reset()
            if verbose:
                print(self.scaled_error_squared)
                if not(self.return_lnI):
                    print(" : {} {} {} {} {} ".format((self.iterations-1)*self.n, self.eff_samp, self.xpy.sqrt(2*self.xpy.max(self.cumulative_values)), self.xpy.sqrt(2*(self.xpy.log(self.integral))),  self.xpy.sqrt(self.scaled_error_squared )/self.integral/self.xpy.sqrt(self.iterations ) ) )
                else:
                    print(" : {} {} {} {} {} ".format((self.iterations-1)*self.n, self.eff_samp, self.xpy.sqrt(2*self.xpy.max(self.cumulative_values)), self.xpy.sqrt(2*self.integral), self.xpy.exp(0.5*(self.scaled_error_squared - self.integral*2) )/self.xpy.sqrt(self.iterations)))
        print('cumulative eval time: ', cumulative_eval_time)
        print('integrator iterations: ', self.iterations)
