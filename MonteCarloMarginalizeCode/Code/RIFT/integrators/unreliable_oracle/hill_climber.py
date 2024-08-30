

import sys
import math

import numpy
np=numpy #import numpy as np

import scipy.optimize

from RIFT.integrators import multivariate_truncnorm as truncnorm



from RIFT.integrators.mcsampler_generic import MCSamplerGeneric

class ClimbingOracle(MCSamplerGeneric):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.reference_samples = None
        self.lnL = None # function to climb
        self.n_climbers = None

    def add_parameter(self, params, pdf,   **kwargs):
        super().add_parameter(params,pdf, **kwargs)

    def setup(self, lnL=None,n_climbers = 10,**kwargs):
        super().setup(**kwargs)
        self.lnL= lnL
        self.n_climbers = n_climbers

        # create bounds formatting.  Need to follow syntax for scipy.optimize.minimize
        bounds = []
        for indx, p in enumerate(self.params_ordered):
            bounds.append(  (self.llim[p], self.rlim[p]) )

        self._bounds = bounds


    def update_sampling_prior(self,ln_weights, n_history,lnw_cut = -10, external_rvs=None,verbose=False,**kwargs):
        if verbose:
            print(" ---- ORACLE UPDATE - CLIMB --- ")
        # Allow updates provided from outside sources, assuming weights provided
        rvs_here = self._rvs
        if external_rvs:
            rvs_here = external_rvs

        # if no history
        if rvs_here is None:
            raise Exception(" oracle  - update  -update_sampling_prior requires initial history")

        n_history_to_use = np.min([n_history,  len(rvs_here[self.params_ordered[0]])] )
        if not(ln_weights is None):
            n_history_to_use = np.min([n_history,len(ln_weights)])
        
        sample_array = self.xpy.empty( (len(self.params_ordered), n_history_to_use))
        for indx, p in enumerate(self.params_ordered):
            sample_array[indx] = rvs_here[p][-n_history_to_use:]


        if lnw_cut and not(ln_weights is None): # we can override and trainon all data
            sample_array = sample_array[:, ln_weights > np.max(ln_weights) + lnw_cut ]  # training range

        # Pick points to climb
        drawn_indx = np.random.choice(range(len(sample_array)), replace=True,size=self.n_climbers) # random samples
        sample_array = sample_array[:,drawn_indx].T

        # Climb 
        rv_out = []
        def fn_here(x):
            xp = x.reshape(1,-1).T
            # lnL takes a *x as arguments, each an array. fn_here takes a single vector-valued argument of scalars
            try: 
                return - self.lnL(*xp)
            except:
                return -np.inf
        
        args_minimize = kwargs['minimize_args'] if 'minimize_args' in kwargs else {'tol':0.1, 'options': {'maxiter':5}}
        for indx in range(self.n_climbers):
            res = scipy.optimize.minimize(fn_here, sample_array[indx], **args_minimize)
            rv_out.append(res.x)

        # create samples based on draws
        self.reference_samples = np.array(rv_out)

    def draw_simplified(self, n_samples, *args, **kwargs):
        print(" ---- ORACLE DRAW --- ")
        rv_out = np.empty( (n_samples, len(self.params_ordered)))

        drawn_indx = np.random.choice(range(len(self.reference_samples)), size=n_samples) # random samples
        rv_out = self.reference_samples[drawn_indx]

        return None, None,rv_out  # compatible interface with mcsampler
        
