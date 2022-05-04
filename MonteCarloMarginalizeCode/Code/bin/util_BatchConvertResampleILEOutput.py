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
from ligo.lw import utils, table, lsctables, ligolw

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
        


# Add LI-style export
if True: #opts.convention == 'LI':
  print("# m1 m2 a1x a1y a1z a2x a2y a2z mc eta  ra dec time phiorb incl psi  distance Npts lnL p ps neff  mtotal q chi_eff chi_p",end=' ')
  if opts.export_extra_spins:
      print( 'theta_jn phi_jl tilt1 tilt2 costilt1 costilt2 phi12 a1 a2 psiJ',end=' ')
  if opts.export_tides:
      print( "lambda1 lambda2 lam_tilde",end=' ')
  if opts.export_cosmology:
      print( " m1_source m2_source mc_source mtotal_source redshift ",end=' ')
  if opts.export_weights:
      print( " weights ", )
  print('')


for fname in args:
    points = lsctables.SimInspiralTable.get_table(utils.load_filename(fname,contenthandler=ligolw.LIGOLWContentHandler), lsctables.SimInspiralTable.tableName)

    # Stage 1: downselect samples (reduces overhead to do this first)
    lnL = np.array([row.alpha1 for row in points])  # hardcoded name
    p = np.array([row.alpha2 for row in points])
    ps = np.array([row.alpha3 for row in points])
    Nmax = np.max([int(row.simulation_id) for row in points])+1    # Nmax. Assumes NOT mixed samples.

    lnLmax = np.max(lnL)
    weights = np.exp(lnL - lnLmax) * (p/ps)
    # replace weights if nan
    weights=np.nan_to_num(weights,nan=1e-2)
    neff_here = np.sum(weights)/np.max(weights)  # neff for this file.  Assumes NOT mixed samples: dangerous
    p_threshold_size = opts.n_output_samples_per_file
    indx_list = np.random.choice(np.arange(len(weights)), size=p_threshold_size, p=weights/np.sum(weights))

    points_reduced = [points[indx] for indx in indx_list]
    points = points_reduced

    # Stage 2: convert, copied directly from convert_output_format_ile2inference
    for indx in np.arange(len(points)):
        pt = points[indx]
        if not(hasattr(pt,'spin1x')):  # no spins were provided. That means zero spin. Initialize to avoid an error
            pt.spin1x = pt.spin1y=pt.spin1z = 0
            pt.spin2x = pt.spin2y=pt.spin2z = 0
        
        # Compute derived quantities
        P=lalsimutils.ChooseWaveformParams()
        P.m1 =pt.mass1
        P.m2 =pt.mass2
        P.s1x = pt.spin1x
        P.s1y = pt.spin1y
        P.s1z = pt.spin1z
        P.s2x = pt.spin2x
        P.s2y = pt.spin2y
        P.s2z = pt.spin2z
        P.fmin=opts.fref
        try:
            P.fmin = pt.f_lower  # should use this first
        except:
            True
        chieff_here =P.extract_param('xi')
        chip_here = P.extract_param('chi_p')
        mc_here = mc(pt.mass1,pt.mass2)
        eta_here = eta(pt.mass1,pt.mass2)
        mtot_here = pt.mass1 + pt.mass2
        
        print( pt.mass1, pt.mass2, pt.spin1x, pt.spin1y, pt.spin1z, pt.spin2x, pt.spin2y, pt.spin2z, mc_here, eta_here, \
            pt.longitude, \
            pt.latitude, \
            pt.geocent_end_time + 1e-9* pt.geocent_end_time_ns, \
            pt.coa_phase,  \
            pt.inclination, \
            pt.polarization, \
            pt.distance, \
            Nmax, lnL[indx], p[indx],ps[indx], neff_here, \
            mtot_here, pt.mass2/pt.mass1, \
            chieff_here,  \
            chip_here,end=' ')
        if opts.export_extra_spins:
            P.incl = pt.inclination  # need inclination to calculate theta_jn
            P.phiref=pt.coa_phase  # need coa_phase to calculate theta_jn ... this determines viewing angle
            thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, psiJ = P.extract_system_frame()
            print( thetaJN, phiJL, theta1, theta2, np.cos(theta1), np.cos(theta2), phi12, chi1, chi2, psiJ,end=' ')
        if opts.export_tides:
            P.lambda1 = pt.alpha5
            P.lambda2 = pt.alpha6
            lam_tilde = P.extract_param("LambdaTilde")
            print( pt.alpha5, pt.alpha6, lam_tilde,end=' ')
        if opts.export_cosmology:
#            z = astropy.cosmology( default_cosmology.luminosity_distance, pt.distance *u.Mpc)
            #z =astropy.cosmology.z_at_value(cosmo.luminosity_distance, pt.distance*u.Mpc)
            z = cosmo_func(pt.distance)
            m1_source = pt.mass1/(1+z)
            m2_source = pt.mass2/(1+z)
            print( m1_source, m2_source, mc_here/(1+z), mtot_here/(1+z), float(z), end=' ')
        if opts.export_weights:
            print(wt[indx],end=' ')
        if opts.export_eccentricity:
            print(ecc[indx])
        print('')


    
