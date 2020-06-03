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

    bounds : np.ndarray
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
                user_func=None, proc_count=None, L_cutoff=None, use_lnL=False):
        # user-specified parameters
        self.d = d
        self.bounds = bounds
        self.gmm_dict = gmm_dict
        self.n_comp = n_comp
        self.user_func=user_func
        self.prior = prior
        self.proc_count = proc_count
        self.use_lnL = use_lnL
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
        self.p_array = None
        self.prior_array = None
        self.integral = 0
        self.var = 0
        self.eff_samp = 0
        self.iterations = 0 # for weighted averages and count
        self.max_value = float('-inf') # for calculating eff_samp
        self.total_value = 0 # for calculating eff_samp
        self.n_max = float('inf')
        # saved values
        self.cumulative_samples = np.empty((0, d))
        self.cumulative_values = np.empty((0, 1))
        self.cumulative_p = np.empty((0, 1))
        self.cumulative_p_s = np.empty((0, 1))
        if L_cutoff is None:
            self.L_cutoff = -1
        else:
            self.L_cutoff = L_cutoff

    def _calculate_prior(self):
        if self.prior is None:
            self.prior_array = np.ones((self.n, 1))
        else:
            self.prior_array = self.prior(self.sample_array)

    def _sample(self):
        self.p_array = np.ones((self.n, 1))
        self.sample_array = np.empty((self.n, self.d))
        for dim_group in self.gmm_dict: # iterate over grouped dimensions
            # create a matrix of the left and right limits for this set of dimensions
            new_bounds = np.empty((len(dim_group), 2))
            index = 0
            for dim in dim_group:
                new_bounds[index] = self.bounds[dim]
                index += 1
            model = self.gmm_dict[dim_group]
            if model is None:
                # sample uniformly for this group of dimensions
                llim = new_bounds[:,0]
                rlim = new_bounds[:,1]
                temp_samples = np.random.uniform(llim, rlim, (self.n, len(dim_group)))
                # update responsibilities
                vol = np.prod(rlim - llim)
                self.p_array *= 1.0 / vol
            else:
                # sample from the gmm
                temp_samples = model.sample(self.n, new_bounds)
                # update responsibilities
                self.p_array *= model.score(temp_samples, new_bounds)
            index = 0
            for dim in dim_group:
                # put columns of temp_samples in final places in sample_array
                self.sample_array[:,[dim]] = temp_samples[:,[index]]
                index += 1

    def _train(self):
        sample_array, value_array, p_array = np.copy(self.sample_array), np.copy(self.value_array), np.copy(self.p_array)
        if self.use_lnL:
            value_array += abs(np.max(value_array))
            value_array = np.exp(value_array)
        weights = abs((value_array * self.prior_array) / p_array) # training weights for samples
        for dim_group in self.gmm_dict: # iterate over grouped dimensions
            # create a matrix of the left and right limits for this set of dimensions
            new_bounds = np.empty((len(dim_group), 2))
            index = 0
            for dim in dim_group:
                new_bounds[index] = self.bounds[dim]
                index += 1
            model = self.gmm_dict[dim_group] # get model for this set of dimensions
            temp_samples = np.empty((self.n, len(dim_group)))
            index = 0
            for dim in dim_group:
                # get samples corresponding to the current model
                temp_samples[:,[index]] = sample_array[:,[dim]]
                index += 1
            if model is None:
                # model doesn't exist yet
                if isinstance(self.n_comp, int) and self.n_comp != 0:
                    model = GMM.gmm(self.n_comp)
                    model.fit(temp_samples, sample_weights=weights)
                elif isinstance(self.n_comp, dict) and self.n_comp[dim_group] != 0:
                    model = GMM.gmm(self.n_comp[dim_group])
                    model.fit(temp_samples, sample_weights=weights)
            else:
                model.update(temp_samples, sample_weights=weights)
            self.gmm_dict[dim_group] = model


    def _calculate_results(self):
        # cumulative samples
        if self.use_lnL:
            value_array = np.exp(self.value_array)
            lnL = self.value_array
        else:
            value_array = np.copy(self.value_array)
            lnL = np.log(self.value_array)
        #mask = value_array >= self.L_cutoff
        mask = value_array >= self.L_cutoff
        mask = mask.flatten()
        self.cumulative_samples = np.append(self.cumulative_samples, self.sample_array[mask], axis=0)
        self.cumulative_values = np.append(self.cumulative_values, lnL[mask], axis=0)
        self.cumulative_p = np.append(self.cumulative_p, self.prior_array[mask], axis=0)
        self.cumulative_p_s = np.append(self.cumulative_p_s, self.p_array[mask], axis=0)
        # make local copies
        value_array = np.copy(value_array) * self.prior_array
        p_array = np.copy(self.p_array)
        value_array /= p_array
        # calculate variance
        curr_var = np.var(value_array) / self.n
        # calculate eff_samp
        max_value = np.max(value_array)
        if max_value > self.max_value:
            self.max_value = max_value
        self.total_value += np.sum(value_array)
        self.eff_samp = self.total_value / self.max_value
        if np.isnan(self.eff_samp):
            self.eff_samp = 0
        # calculate integral
        curr_integral = (1.0 / self.n) * np.sum(value_array)
        # update results
        self.integral = ((self.integral * self.iterations) + curr_integral) / (self.iterations + 1)
        self.var = ((self.var * self.iterations) + curr_var) / (self.iterations + 1)

    def _reset(self):
        ### reset GMMs
        for k in self.gmm_dict:
            self.gmm_dict[k] = None
        

    def integrate(self, func, min_iter=10, max_iter=20, var_thresh=0.0, max_err=10,
            neff=float('inf'), nmax=None, progress=False, epoch=None,verbose=True):
        '''
        Evaluate the integral

        Parameters
        ----------
        func : function
            Integrand function
        min_iter : int
            Minimum number of integrator iterations
        max_iter : int
            Maximum number of integrator iterations
        var_thresh : float
            Variance threshold for terminating integration
        max_err : int
            Maximum number of errors to catch before terminating integration
        neff : float
            Effective samples threshold for terminating integration
        nmax : int
            Maximum number of samples to draw
        progress : bool
            Print GMM parameters each iteration
        '''
        err_count = 0
        cumulative_eval_time = 0
        if nmax is None:
            nmax = max_iter * self.n
        while self.iterations < max_iter and self.ntotal < nmax and self.eff_samp < neff:
#            print('Iteration:', self.iterations)
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
                print('Error sampling, retrying...')
                err_count += 1
                continue
            t1 = time.time()
            if self.proc_count is None:
                self.value_array = func(np.copy(self.sample_array))
            else:
                split_samples = np.array_split(self.sample_array, self.proc_count)
                p = Pool(self.proc_count)
                self.value_array = np.concatenate(p.map(func, split_samples), axis=0)
                p.close()
            cumulative_eval_time += time.time() - t1
            self._calculate_prior()
            self._calculate_results()
            self.iterations += 1
            self.ntotal += self.n
            if self.iterations >= min_iter and self.var < var_thresh:
                break
            try:
                self._train()
            except KeyboardInterrupt:
                print('KeyboardInterrupt, exiting...')
                break
            except Exception as e:
                print(traceback.format_exc())
                print('Error training, retrying...')
                err_count += 1
            if self.user_func is not None:
                self.user_func(self)
            if progress:
                for k in self.gmm_dict:
                    if self.gmm_dict[k] is not None:
                        self.gmm_dict[k].print_params()
            if epoch is not None and self.iterations % epoch == 0:
                self._reset()
            if verbose:
                # Standard mcsampler message, to monitor convergence
                print(" : {} {} {} {} {} ".format((self.iterations-1)*self.n, self.eff_samp, np.sqrt(2*np.max(self.cumulative_values)), np.sqrt(2*np.log(self.integral)), "-" ) )
        print('cumulative eval time: ', cumulative_eval_time)
        print('integrator iterations: ', self.iterations)
