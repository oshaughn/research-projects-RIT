import numpy as np



# idea: be able to create oracle from reference samples
class ReferenceSamples(object):
    def __init__(self,fname=None):
        self.reference_samples = None
        self.reference_params =None

    def from_ascii(self, fname=None, reference_params=None,npts_out=None):
        if not(fname) or (reference_params is None):
            raise Exception(" ReferenceSamples : requires fname or reference_params")
        
        self.reference_params = reference_params

        # load data
        dat = np.genfromtxt(fname,names=True)

        # check field names are all present
        for name in reference_params:
            if not(name in dat.dtype.names):
                raise Exception(" ReferenceSamples: Cannot find in file : {}".format(name))

        # create array
        if not(npts_out):
            npts_out = len(dat[reference_params[0]])
        npts_out = np.min([len(dat[reference_params[0]]), npts_out]) # don't take more
        
        # copy data and return
        dat_out = np.empty((npts_out, len(reference_params)))
        for indx, p in enumerate(reference_params):
            dat_out[:,indx] = dat[p]

        self.reference_samples = dat_out
