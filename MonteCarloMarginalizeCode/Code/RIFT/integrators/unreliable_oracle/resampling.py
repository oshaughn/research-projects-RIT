

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

        # parameters that match our list
        self.valid_params = [p for x in self.reference_params  if p in self.params_ordered] # valid parameters to sample from
        self.other_params = list( set(self.params_ordered) - set(valid_params)) # remainder, will be uniform
            


    def draw_simplified(self, n_samples, *args, **kwargs):
        rv_out = np.empty( (n_samples, len(self.params_ordered)))
        drawn_indx = np.random.choice(range(len(reference_samples)), size=n_samples) # random samples
        # Fill in raw data
        for p in self.valid_params:
            indx_out = self.params_ordered.index(p)
            indx_in = self.reference_params.index(p)
            rv_out[indx_out]  =  drawn_indx[indx_in] # copy column
        # fill in other columns
        for p in self.other_params:
            indx_out = self.params_ordered.index(p)
            rv_out[indx_out] = np.random.uniform(low=self.llim(p), high=self.rrlim(p), size=n_samples)

        return None, None,rv_out  # compatible interface with mcsampler
        
