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


    def from_skymap_fits(self, fname=None, reference_params=None,npts_out=None, cos_dec=False):
        # see https://emfollow.docs.ligo.org/userguide/tutorial/multiorder_skymaps.html
        from astropy.table import QTable
        from astropy import units as u
        import astropy_healpix as ah

        if not(fname):
            raise Exception(" ReferenceSamples : requires fname ")

        self.reference_params = reference_params
        skymap = QTable.read(fname)

        # identify pertinent samples
        skymap.sort('PROBDENSITY', reverse=True)
        level, ipix = ah.uniq_to_level_ipix(skymap['UNIQ'])
        pixel_area = ah.nside_to_pixel_area(ah.level_to_nside(level))
        prob = pixel_area * skymap['PROBDENSITY']
        cumprob = np.cumsum(prob)
        i = cumprob.searchsorted(0.9)  # critical 90% area threshold
        skymap=skymap[:i] # delete everything except this part

        # convert to array of RA/DEC values (pixel centers)
        ra = np.zeros(i)
        dec = np.zeros(i)
        for indx in range(i):
            uniq = skymap[indx]['UNIQ']
            level, ipix = ah.uniq_to_level_ipix(uniq)
            nside = ah.level_to_nside(level)
            ra_val, dec_val = ah.healpix_to_lonlat(ipix, nside, order='nested') # warning, units
            ra[indx]  = ra_val.value
            dec[indx] = dec_val.value

        if cos_dec:
            dec = np.arccos(dec)
        self.reference_samples = np.c_[ra,dec]
        self.reference_params = ['right_ascension', 'declination']
