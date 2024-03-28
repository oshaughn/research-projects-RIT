import numpy as np
import RIFT.lalsimutils as lalsimutils
import lal

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

    def from_sim_xml(self, fname=None, reference_params=None,npts_out=None):
        if not(fname) or (reference_params is None):
            raise Exception(" ReferenceSamples : requires fname or reference_params")

        self.reference_params = reference_params

        P_list = lalsimutils.xml_to_ChooseWaveformParams_array(fname)

        coord_names = reference_params
        # code verbatim from CIP
        dat_mass_post = np.zeros( (len(P_list),len(coord_names)),dtype=np.float64)
        for indx_line  in np.arange(len(P_list)):
            for indx in np.arange(len(coord_names)):
                fac=1
                if coord_names[indx] in ['mc', 'mtot', 'm1', 'm2']:
                    fac = lal.MSUN_SI
                dat_mass_post[indx_line,indx] = P_list[indx_line].extract_param(coord_names[indx])/fac

        self.reference_samples = dat_mass_post
