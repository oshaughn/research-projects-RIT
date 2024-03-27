

import sys
import math

import numpy
np=numpy #import numpy as np


from RIFT.integrators import multivariate_truncnorm as truncnorm



from RIFT.integrators.mcsampler_generic import MCSamplerGeneric

class Puffballracle(MCSamplerGeneric):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.reference_mean = None
        self.reference_cov = None

    def add_parameter(self, params, pdf,   **kwargs):
        super().add_parameter(params,pdf, **kwargs)

    def setup(self, reference_mean=None, reference_cov=None,**kwargs):
        super().setup(**kwargs)
        # can be empty
        self.reference_mean = reference_mean 
        self.reference_cov = reference_cov


    def update_sampling_prior(self,ln_weights, n_history,lnw_cut = -10, external_rvs=None,**kwargs):
        # Allow updates provided from outside sources, assuming weights provided
        rvs_here = self._rvs
        if external_rvs:
            rvs_here = external_rvs

        if lnw_cut: # we can override and trainon all data
            rv_train = rvs_here[ln_weights > np.max(ln_weights) + lnw_cut ]  # training range
        else:
            rv_train = rvs_here 

        # compute mean and cov
        self.reference_mean = np.mean(rv_train, axis=0)
        self.reference_cov =  np.cov(rv_train)


    def draw_simplified(self, n_samples, *args, **kwargs):
        rv_out = np.empty( (n_samples, len(self.params_ordered)))

        # create bounds formatting
        bounds = np.empty((len(self.params_ordered),2))
        for indx, p in enumerate(self.params_ordered):
            bounds[:,indx] = [self.llim, self.rlim]

        # draw, iterating until we fill our target array
        n_out =0
        while n_out < n_samples:
            # from multivariate_truncnorm ... but directly
            samples = np.random.multivariate_normal(self.mean, self.cov, size=n_samples*2)
            llim = np.rot90(bounds[:,[0]])
            rlim = np.rot90(bounds[:,[1]])
            replace1 = np.greater(samples, llim).all(axis=1)
            replace2 = np.less(samples, rlim).all(axis=1)
            replace = np.array(np.logical_and(replace1, replace2)).flatten()
            to_append = samples[replace]
            if len(to_append) + n_out > n_samples:
                to_append = to_append[:(n_samples - n_out)]
            n_end = n_out + len(to_append)
            rv_out[n_out:n_end] = to_append
            n_out = n_end


        return None, None,rv_out  # compatible interface with mcsampler
        
