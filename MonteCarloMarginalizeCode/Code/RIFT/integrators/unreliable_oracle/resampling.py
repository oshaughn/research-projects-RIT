

import sys
import math
#import bisect
from collections import defaultdict
from types import ModuleType

import numpy
np=numpy #import numpy as np
from scipy import integrate, interpolate, special
import itertools
import functools



from RIFT.integrators.mcsampler_generic import MCSamplerGeneric

class ResamplingOracle(MCSamplerGeneric):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.reference_samples = None
        self.reference_params = None

    def add_parameter(self, params, pdf,   **kwargs):
        super().add_parameter(params,pdf, **kwargs)

    def setup(self, reference_samples=None, reference_params=None,**kwargs):
        super().setup(**kwargs)
        self.reference_samples = reference_samples
        self.reference_params = reference_params

        # parameters that match our list.  
        # Note: we may need to initialize this LATER, if we have not yet been passed our params!
        if self.params_ordered and self.reference_params:
            print(self.params_ordered, self.reference_params)
            self.valid_params = [p for p in self.reference_params  if p in self.params_ordered] # valid parameters to sample from
            # remainder, will be uniform.  ORDER MATTERS, so this is NOT a set difference:
            # draw_simplified consumes a block of numpy's (seeded) global stream per entry, so
            # set order decides which block lands on which parameter -- and str hashing is
            # salted per process (PYTHONHASHSEED), so two runs with the SAME --seed drew
            # different distances/inclinations.  Measured: five processes at --seed 101 gave
            # four distinct distance blocks.  The oracle trains the sampling prior (and seeds
            # the AV live volume), so that reaches lnZ.  params_ordered order is stable.
            _valid = set(self.valid_params)
            self.other_params = [p for p in self.params_ordered if p not in _valid]

    def update_sampling_prior(self, *args, **kwargs):
        True


    def draw_simplified(self, n_samples, *args, **kwargs):
        rv_out = np.empty( (n_samples, len(self.params_ordered)))
        drawn_indx = np.random.choice(range(len(self.reference_samples)), size=n_samples) # random samples
        # Fill in raw data
        for p in self.valid_params:
            indx_out = self.params_ordered.index(p)
            indx_in = self.reference_params.index(p)
            rv_out[:,indx_out]  =  self.reference_samples[drawn_indx][:,indx_in] # copy column
        # fill in other columns
        for p in self.other_params:
            indx_out = self.params_ordered.index(p)
            rv_out[:,indx_out] = np.random.uniform(low=self.llim[p], high=self.rlim[p], size=n_samples)

        return None, None,rv_out  # compatible interface with mcsampler
        
