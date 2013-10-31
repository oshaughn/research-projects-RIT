"""
ourparams.py
   - Intended to hold all standard arguments, either in 
      - default command line settings OR 
      - via  hardcoded defaults
   - Provides command line parsing
   - Provides explicit settings for sampler parameters
"""

import argparse
import lalsimulation as lalsim
import lal

rosDebugMessagesDictionary = {}   # Mutable after import (passed by reference). Not clear if it can be used by caling routines
                                                  # BUT if every module has a `PopulateMessagesDictionary' module, I can set their internal copies


def ParseStandardArguments():
    global rosDebugMessagesDictionary

    rosDebugMessagesDictionary["Test"] = True
    rosDebugMessagesDictionary["DebugMessages"] = False
    rosDebugMessagesDictionary["DebugMessagesLong"] = False
    

    parser = argparse.ArgumentParser()
    print " === Loading options === "
    print "  ++ NOTE OPTIONS MUST BE IMPLEMENTED BY EACH ROUTINE INDIVIDUALLY - NOT YET STANDARDIZED  ++ "
    # Options.  Try to be consistent with lalinference

    # General parameters
    parser.add_argument("--Niter", dest='nmax',default=10000,type=int,help="Number of iterations")
    parser.add_argument("--Neff", dest='neff', default=100,type=int,help="Target number of effective samples")
    parser.add_argument("--Nskip", dest='nskip', default=200,type=int,help="How often MC progress is reported (in verbose mode)")

    # Likelihood functions
    parser.add_argument("--LikelihoodType_raw",default=True,action='store_true')
    parser.add_argument("--LikelihoodType_MargPhi",default=False,action='store_true')
    parser.add_argument("--LikelihoodType_MargT",default=False,action='store_true')
    parser.add_argument("--LikelihoodType_MargTdisc",default=False,action='store_true')
    # Infrastructure choices:
    parser.add_argument("--skip-interpolation",dest="opt_SkipInterpolation",default=False,action='store_true')  # skip interpolation : saves time, but some things will crash later

    # Noise model
    parser.add_argument("--psd-file",dest='psd_file',default=None, help="psd-file for all instruments. Assumes PSD for all instruments is provided in the file.  If None, uses an analytic PSD")
#    parser.add_argument("--psd-file",dest='psd_file',default=None, help="instrument=psd-file, e.g. H1=H1_PSD.xml.gz. Can be given multiple times for different instruments. If None, uses an analytic PSD.  Implemented for compatibility with Pankow")

    # Data or injection
    parser.add_argument("-c","--cache-file",dest='cache_file', default=None, help="LIGO cache file containing all data needed.")
    parser.add_argument("-C","--channel-name", action="append", help="instrument=channel-name, e.g. H1=FAKE-STRAIN. Can be given multiple times for different instruments.")
    parser.add_argument("--force-read-full-frames", dest='opt_ReadWholeFrameFilesInCache', action='store_true')
    parser.add_argument("--inj-xml", dest='inj', default=None,help="inspiral XML file containing injection information.")
    parser.add_argument("--event", dest="event_id", default=0,help="event ID of injection XML to use.")
    parser.add_argument("--coinc-xml", dest='coinc', help="gstlal_inspiral XML file containing coincidence information.")
    parser.add_argument("--seglen", dest='seglen', type=int, default=32, help="Minimum segment duration surrounding coing to be analyzed. (Will be rounded up to next power of 2).  ")
    parser.add_argument( "--padding", dest='padding',  type=int, default=10, help="Time window after the trigger to be included in the data segment")
    parser.add_argument("--srate", dest='srate', type=int, default=4096, help="Sampling rate to use. Data filtered down to this rate, PSD extended up to this rate. Can conflict with stored PSD file.")

    # Sampling prior
    parser.add_argument("--sampling-prior-use-inj-sky-location",dest='opt_UseKnownSkyPosition',action='store_true')
    parser.add_argument("--sampling-prior-use-skymap",dest='opt_UseSkymap')

    # Lower frequencies 
    parser.add_argument("--fmin-snr", dest='fmin_SNR', type=float, default=30)
    parser.add_argument("--fmin-template", dest='fmin_Template', type=float, default=30)
    parser.add_argument("--fmax-snr", dest='fmax_SNR', type=float, default=2000)

    # Options to set template physics
    parser.add_argument("--fref", dest='fref', type=float, default=100.0, help="Waveform reference frequency. Required, default is 100 Hz.")
    parser.add_argument("--amporder", dest='amporder', type=int, default=0, help="Amplitude order of PN waveforms")
    parser.add_argument("--Lmax", dest="Lmax", type=int, default=2, help="Lmax (>=2) to use")
    parser.add_argument( "--order",dest='order', type=int, default=-1, help="Phase order of PN waveforms")
    parser.add_argument("--spinOrder", dest='spinOrder', type=int, default=-1, help="Specify twice the PN order (e.g. 5 <==> 2.5PN) of spin effects to use, only for LALSimulation (default: -1 <==> Use all spin effects).")
    parser.add_argument("--noSpin",  action="store_false", dest="SpinQ",default=False, help="template will assume no spins.")
    parser.add_argument("--spinAligned", action="store_true", dest="SpinAlignedQ",default=False, help="template will assume spins aligned with the orbital angular momentum.")
    parser.add_argument("--approx",type=str, default="TaylorT4", help="Specify a template approximant and phase order to use.") # A tighter integration with GetStringFromApproximant and vice versa would be nice

    # Fix parameters at injected values.  Assumes injected values present
    parser.add_argument("--fix-rightascension", action='append_const', dest='fixparams',const='ra',default=[])
    parser.add_argument("--fix-declination", action='append_const', dest='fixparams',const="dec")
    parser.add_argument("--fix-polarization", action='append_const', dest='fixparams',const='psi')
    parser.add_argument("--fix-distance", action='append_const', dest='fixparams',const='dist')
    parser.add_argument("--fix-time", action='append_const', dest='fixparams',const='tref')
    parser.add_argument("--fix-inclination", action='append_const', dest='fixparams',const='incl')
    parser.add_argument("--fix-phase", action='append_const', dest='fixparams',const='phi')

    parser.add_argument("--force-gps-time", dest='force_gps_time', default=None,type=float)

    # Extra options to set verbosity level
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False)
    parser.add_argument("--super-verbose", action="store_true", dest="super_verbose", default=False)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose", default=False)

    # Interactive plots
    parser.add_argument("--show-sampler-results",dest='plot_ShowSampler', action='store_true', default=False,help ="Show interactive plots from sampler results. ")
    parser.add_argument("--show-sampler-inputs",dest='plot_ShowSamplerInputs', action='store_true', default=False,help ="Show interactive plots from sampler *inputs*. ")
    parser.add_argument("--show-likelihood-versus-time",dest="plot_ShowLikelihoodVersusTime", action='store_true',default=False,help="Show single-IFO likelihoods versus time. If injection parameters are known, also plot L(t) in the geocenter. Useful to confirm a signal has been correctly windowed at all!")
    parser.add_argument("--show-psd",dest='plot_ShowPSD',action='store_true',default=False,help="Plot PSDs, as imported and stored.")

    # File output and conditions controlling output
    parser.add_argument("--save-sampler-file", dest="points_file_base",default="sampler-output-file")
    parser.add_argument("--save-threshold-fraction", dest="points_threshold_match",type=float,default=0.0,help="Roughly speaking, target match for points to be returned.  In practice, all points with L> \sqrt{P}L_{max} are returned")

    args = parser.parse_args()
    # Connect debugging options to argument parsing
    rosDebugMessagesDictionary["DebugMessages"] = args.verbose
    rosDebugMessagesDictionary["DebugMessagesLong"] = args.super_verbose
    if (rosDebugMessagesDictionary["DebugMessagesLong"]):
        rosDebugMessagesDictionary["DebugMessages"]=True

    return args, rosDebugMessagesDictionary


###
### Populate standard sampler arguments
###
import mcsampler
import numpy 
import functools

import factored_likelihood
tWindowExplore  = factored_likelihood.tWindowExplore
Tmax = numpy.max([0.05,tWindowExplore[1]]) # max ref. time
Tmin = numpy.min([-0.05,tWindowExplore[0]]) # min ref. time
Dmax = 2000. #* 1.e6 * lal.LAL_PC_SI # max distance. CAN CAUSE PROBLEMS with the CDF integrator if this limit is large (SI units)
Dmin = 1. # * 1.e6 * lal.LAL_PC_SI      # min distance.  Needs to be nonzero to avoid integrator stupid problems with reconstructing the CDF

# set up bounds on parameters
# Polarization angle
psi_min, psi_max = 0, numpy.pi
# RA and dec
ra_min, ra_max = 0, 2*numpy.pi
dec_min, dec_max = -numpy.pi/2, numpy.pi/2
# Reference time
tref_min, tref_max = Tmin, Tmax
# Inclination angle
inc_min, inc_max = 0, numpy.pi
# orbital phi
phi_min, phi_max = 0, 2*numpy.pi
# distance
dist_min, dist_max = Dmin, Dmax



def PopulateSamplerParameters(sampler, theEpochFiducial, tEventFiducial,distBoundGuess, Psig, opts):

    # Uniform sampling (in area) but nonuniform sampling in distance (*I hope*).  Auto-cdf inverse
    if not(opts.opt_UseSkymap):
        if opts.opt_UseKnownSkyPosition:
            print " +++ USING INJECTED SKY POSITION TO NARROW THE SKY SEARCH REGION "
            sampler.add_parameter("ra", functools.partial(mcsampler.gauss_samp, Psig.phi,0.03), None, ra_min, ra_max, 
                              prior_pdf = mcsampler.uniform_samp_phase)
            sampler.add_parameter("dec", functools.partial(mcsampler.gauss_samp, Psig.theta,0.03), None, dec_min, dec_max, 
                              prior_pdf= mcsampler.uniform_samp_dec)
        else:
            if opts.fixparams.count('ra') and Psig:
                print "  ++++ Fixing ra to injected value +++ "
                sampler.add_pinned_parameter('ra', Psig.phi )
            else:
                sampler.add_parameter("ra", mcsampler.uniform_samp_phase, None, ra_min, ra_max, 
                                      prior_pdf = mcsampler.uniform_samp_phase)
            if opts.fixparams.count('dec') and Psig:
                    print "  ++++ Fixing dec to injected value +++ "
                    sampler.add_pinned_parameter('dec', Psig.theta )
            else:
                sampler.add_parameter("dec",mcsampler.uniform_samp_dec, None, dec_min, dec_max, 
                          prior_pdf= mcsampler.uniform_samp_dec)
    # Override *everything* if I have a skymap used
    else:
        print "  ++++ USING SKYMAP +++ "
        print " ==Loading skymap=="
        print "   skymap file ", opts.opt_UseSkymap

        # Import modules used for skymaps only here (they will be available by reference later)
        import healpy
        from lalinference.bayestar import fits as bfits
        from lalinference.bayestar import plot as bplot
    
        smap, smap_meta = bfits.read_sky_map(opts.opt_UseSkymap)
        # Modify skymap to regularize it: don't want zero probability anywhere!
        smap = 0.999 *smap + 0.001*numpy.ones(len(smap))  # 0.11% probability of being somewhere randomly on the sky
        sides = healpy.npix2nside(len(smap))
        cum_smap = numpy.cumsum(smap)

        import bisect
        def bayestar_cdf_inv(x,y):                 # it will be passed two arguments, use one
            indx = bisect.bisect(cum_smap,x) 
            th,ph = healpy.pix2ang(sides, indx)
            if rosDebugMessagesDictionary["DebugMessagesLong"]:
                print " skymap used x->(th,ph) :", x,th,ph
            return ph,th-numpy.pi/2.

        def bayestar_cdf_inv_vector(x,y):   # Manually vectorize, so I can insert pdb breaks
            indxVec = map(lambda z: bisect.bisect(cum_smap,z), x) 
            th, ph = healpy.pix2ang(sides,indxVec)
            return numpy.array([ph, th-numpy.pi/2.])                            # convert to RA, DEC from equatorial. Return must be pair of arrays

        def bayestar_pdf_radec(x,y):               # look up value at targted pixel 
            ra,dec = x,y
            indx = bplot._healpix_lookup(smap,  ra, dec) # note bplot takes lon (=ra, really), lat (dec). Not sure why
            return smap[indx]

        sampler.add_parameter(("ra", "dec"), 
                              None, 
                              bayestar_cdf_inv_vector, (ra_min, dec_min), (ra_max, dec_max))
        sampler.prior_pdf[("ra", "dec")] = numpy.vectorize(lambda x,y: 1./len(smap)) # 1/number of pixels
        sampler.pdf[("ra", "dec")] =          numpy.vectorize(bayestar_pdf_radec)              # look up value at targteed pixel 
        sampler._pdf_norm[("ra", "dec")] = 1


    if opts.fixparams.count('psi') and Psig:
        print "  ++++ Fixing psi to injected value +++ "
        sampler.add_pinned_parameter('psi', Psig.psi )
    else:
        sampler.add_parameter("psi", functools.partial(mcsampler.uniform_samp_vector, psi_min, psi_max), None, psi_min, psi_max,
                      prior_pdf =mcsampler.uniform_samp_psi )
    if opts.fixparams.count('tref') and Psig:
        print "  ++++ Fixing time to injected value +++ "
        sampler.add_pinned_parameter("tref", float(Psig.tref - theEpochFiducial) )
    else:
        sampler.add_parameter("tref", functools.partial(mcsampler.gauss_samp, tEventFiducial, 0.01), None, tref_min, tref_max, 
                      prior_pdf = functools.partial(mcsampler.uniform_samp_vector, tWindowExplore[0],tWindowExplore[1]))
    # Phase (angle of L)
    if opts.fixparams.count('phi') and Psig:
            print "  ++++ Fixing phi to injected value +++ "
            sampler.add_pinned_parameter("phi", Psig.phiref )
    else:
        sampler.add_parameter("phi", functools.partial(mcsampler.uniform_samp_vector, phi_min, phi_max), None, phi_min, phi_max,
                      prior_pdf = mcsampler.uniform_samp_phase)
    # Inclination (angle of L; nonspinning)
    if opts.fixparams.count('inc') and Psig:
        print "  ++++ Fixing incl to injected value +++ "
        sampler.add_pinned_parameter("incl", Psig.incl )
    else:
        sampler.add_parameter("incl", functools.partial(mcsampler.cos_samp_vector), None, inc_min, inc_max,
                      prior_pdf = mcsampler.uniform_samp_theta)

    # Distance
    if opts.fixparams.count('dist') and Psig:
            print "  ++++ Fixing distance to injected value +++ "
            sampler.add_pinned_parameter("dist", Psig.dist/(1e6*lal.LAL_PC_SI) )
    else:
        sampler.add_parameter("dist",
                          functools.partial(mcsampler.quadratic_samp_vector, distBoundGuess), None, dist_min, dist_max,
#                          functools.partial(mcsampler.uniform_samp_vector,0, distBoundGuess), None, dist_min, dist_max,
#                          functools.partial(mcsampler.quadratic_samp_withfloor_vector, distBoundGuess, dist_max, 0.001), None, dist_min, dist_max,
                         prior_pdf = functools.partial(mcsampler.quadratic_samp_vector, dist_max)
                         )
#        sampler.add_parameter("dist", functools.partial(mcsampler.quadratic_samp_vector,  distBoundGuess ), None, dist_min, dist_max, prior_pdf = numpy.vectorize(lambda x: x**2/(3.*numpy.power(dist_max,3))))
        
    return True
