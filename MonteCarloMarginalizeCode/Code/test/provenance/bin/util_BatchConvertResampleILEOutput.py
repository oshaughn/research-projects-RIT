#! /usr/bin/env python
#
# GOAL
#   - take output xml files (several files)
#   - for each file **indepedently**, create a fair draw sample and print it out.  Produces fairdraw extrinsic, intrinsic samples if input xml is fairdraw!
#   - if you want to resample *jointly*, use the older tools 
#   
#
# EXAMPLES
#    util_BatchConvertResampleILEOutput.py --n-output-samples 5 EXTR_out.xml.gz > extrinsic_posterior_samples.dat


# Setup. 
import numpy as np
import lal
import RIFT.lalsimutils as lalsimutils
from igwn_ligolw import utils, table, lsctables, ligolw

import bisect


# Contenthandlers : argh
#   - http://software.ligo.org/docs/glue/
lsctables.use_in(ligolw.LIGOLWContentHandler)

def mc(m1,m2):
    return np.power(m1*m2, 3./5.)/np.power(m1+m2, 1./5.)
def eta(m1,m2):
    return m1*m2/(np.power(m1+m2, 2))



import optparse
parser = optparse.OptionParser()
parser.add_option("--n-output-samples-per-file",default=5,type=int)
parser.add_option("--fref",default=20,type=float,help="Reference frequency. Depending on approximant and age of implementation, may be ignored")
parser.add_option("--export-extra-spins",action='store_true',help="Reference frequency. Depending on approximant and age of implementation, may be ignored")
parser.add_option("--export-tides",action='store_true',help="Include tidal parameters")
parser.add_option("--export-cosmology",action='store_true',help="Include source frame masses and redshift")
parser.add_option("--export-weights",action='store_true',help="Include a field 'weights' equal to L p/ps")
parser.add_option("--export-eccentricity", action="store_true", help="Include eccentricity")
parser.add_option("--with-cosmology",default="Planck15",help="Specific cosmology to use")
parser.add_option("--use-interpolated-cosmology",action='store_true',help="Specific cosmology to use")
parser.add_option("--convention",default="RIFT",help="RIFT|LI")
opts, args = parser.parse_args()

if opts.export_cosmology:
    import astropy, astropy.cosmology
    from astropy.cosmology import default_cosmology
    import astropy.units as u
    default_cosmology.set(opts.with_cosmology)
    cosmo = default_cosmology.get()

    # Set up fast cosmology
    def cosmo_func(dist_Mpc):
        return astropy.cosmology.z_at_value(cosmo.luminosity_distance, dist_Mpc*u.Mpc)
    if opts.use_interpolated_cosmology:
        from scipy import interpolate
        zvals = np.arange(0,20,0.0025)   # Note hardcoded maximum redshift
        dvals = np.zeros(len(zvals))
        for indx in np.arange(len(zvals)):
            dvals[indx]  = float(cosmo.luminosity_distance(zvals[indx])/u.Mpc)
        # Interpolate. should use monotonic spline code, but ...
        interp_cosmo_func = interpolate.interp1d(dvals,zvals)
        cosmo_func = interp_cosmo_func
        


print(" Inputs: ",  ' '.join(args))
print(" Outputs: stdout ")
