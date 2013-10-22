import sys
import bisect
from collections import defaultdict

import numpy
from scipy import integrate, interpolate
import itertools
import functools

import healpy

from statutils import cumvar

from multiprocessing import Pool

__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

rosDebugMessages = True

class MCSampler(object):
    """
    Class to define a set of parameter names, limits, and probability densities.
    """
    def __init__(self):
        # Parameter names
        self.params = set()
        # parameter -> pdf function object
        self.pdf = {}
        # If the pdfs aren't normalized, this will hold the normalization 
        # constant
        self._pdf_norm = defaultdict(lambda: 1)
        # Cache for the sampling points
        self._rvs = {}
        # Sample point cache
        self._cache = []
        # parameter -> cdf^{-1} function object
        self.cdf = {}
        self.cdf_inv = {}
        # params for left and right limits
        self.llim, self.rlim = {}, {}

        # MEASURES (=priors): ROS needs these at the sampler level, to clearly separate their effects
        # ASSUMES the user insures they are normalized
        self.prior_pdf = {}

    def clear(self):
        """
        Clear out the parameters and their settings, as well as clear the sample cache.
        """
        self.params = set()
        self.pdf = {}
        self._pdf_norm = defaultdict(lambda: 1)
        self._rvs = {}
        self._cache = []
        self.cdf = {}
        self.cdf_inv = {}
        self.llim = {}
        self.rlim = {}

    def add_parameter(self, params, pdf,  cdf_inv=None, left_limit=None, right_limit=None, prior_pdf=None):
        """
        Add one (or more) parameters to sample dimensions. params is either a string describing the parameter, or a tuple of strings. The tuple will indicate to the sampler that these parameters must be sampled together. left_limit and right_limit are on the infinite interval by default, but can and probably should be specified. If several params are given, left_limit, and right_limit must be a set of tuples with corresponding length. Sampling PDF is required, and if not provided, the cdf inverse function will be determined numerically from the sampling PDF.
        """
        self.params.add(params)
        if rosDebugMessages: 
            print " Adding parameter ", params, " with limits ", [left_limit, right_limit]
        if isinstance(params, tuple):
            if left_limit is None:
                self.llim[params] = list(float("-inf"))*len(params)
            else:
                self.llim[params] = left_limit
            if right_limit is None:
                self.rlim[params] = list(float("+inf"))*len(params)
            else:
                self.rlim[params] = right_limit
        else:
            if left_limit is None:
                self.llim[params] = float("-inf")
            else:
                self.llim[params] = left_limit
            if right_limit is None:
                self.rlim[params] = float("+inf")
            else:
                self.rlim[params] = right_limit
        self.pdf[params] = pdf
        # FIXME: This only works automagically for the 1d case currently
        self.cdf_inv[params] = cdf_inv or self.cdf_inverse(params)
        if not isinstance(params, tuple):
            self.cdf[params] =  self.cdf_function(params)
            if prior_pdf is None:
                self.prior_pdf[params] = lambda x:1
            else:
                self.prior_pdf[params] = prior_pdf


    def cdf_function(self, param):
        """
        Numerically determine the  CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF.
        """
        # Solve P'(x) == p(x), with P[lower_boun] == 0
        def dP_cdf(p, x):
            return self.pdf[param](x)
        x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000)
        # Integrator needs to have a step size which doesn't step over the
        # probability mass
        # TODO: Determine h_max.
        cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.01*(self.rlim[param]-self.llim[param])).T[0]
        if cdf[-1] != 1.0: # original pdf wasn't normalized
            self._pdf_norm[param] = cdf[-1]
            cdf /= cdf[-1]
        # Interpolate the inverse
        return interpolate.interp1d( x_i,cdf)

    def cdf_inverse(self, param):
        """
        Numerically determine the inverse CDF from a given sampling PDF. If the PDF itself is not normalized, the class will keep an internal record of the normalization and adjust the PDF values as necessary. Returns a function object which is the interpolated CDF inverse.
        """
        # Solve P'(x) == p(x), with P[lower_boun] == 0
        def dP_cdf(p, x):
            return self.pdf[param](x)
        x_i = numpy.linspace(self.llim[param], self.rlim[param], 1000)
        # Integrator needs to have a step size which doesn't step over the
        # probability mass
        # TODO: Determine h_max.
        cdf = integrate.odeint(dP_cdf, [0], x_i, hmax=0.01*(self.rlim[param]-self.llim[param])).T[0]
        if cdf[-1] != 1.0: # original pdf wasn't normalized
            self._pdf_norm[param] = cdf[-1]
            cdf /= cdf[-1]
        # Interpolate the inverse
        return interpolate.interp1d(cdf, x_i)

    def draw(self, rvs, *args, **kwargs):
        """
        Draw a set of random variates for parameter(s) args. Left and right limits are handed to the function. If args is None, then draw *all* parameters. 'rdict' parameter is a boolean. If true, returns a dict matched to param name rather than list. rvs must be either a list of uniform random variates to transform for sampling, or an integer number of samples to draw.
        """
        if len(args) == 0:
            args = self.params

        if isinstance(rvs, int) or isinstance(rvs, float):
            #
            # Convert all arguments to tuples
            #
            # FIXME: UGH! Really? This was the most elegant thing you could come
            # up with?
            rvs_tmp = [numpy.random.uniform(0,1,(len(p), rvs)) for p in map(lambda i: (i,) if not isinstance(i, tuple) else i, args)]
            rvs_tmp = numpy.array([self.cdf_inv[param](*rv) for (rv, param) in zip(rvs_tmp, args)])
        else:
            rvs_tmp = numpy.array(rvs)


        # FIXME: ELegance; get some of that...
        # This is mainly to ensure that the array can be "splatted", e.g.
        # separated out into its components for matching with args. The case of
        # one argument has to be handled specially.
        res = []
        for (cdf_rv, param) in zip(rvs_tmp, args):
            if len(cdf_rv.shape) == 1:
                res.append((self.pdf[param](cdf_rv)/self._pdf_norm[param], self.prior_pdf[param](cdf_rv), cdf_rv))
            else:
                res.append((self.pdf[param](*cdf_rv)/self._pdf_norm[param], self.prior_pdf[param](*cdf_rv), cdf_rv))

        #
        # Cache the samples we chose
        #
        if len(self._rvs) == 0:
            self._rvs = dict(zip(args, rvs_tmp))
        else:
            rvs_tmp = dict(zip(args, rvs_tmp))
            for p, ar in self._rvs.iteritems():
                self._rvs[p] = numpy.hstack( (ar, rvs_tmp[p]) )

        #
        # Pack up the result if the user wants a dictonary instead
        #
        if kwargs.has_key("rdict"):
            return dict(zip(args, res))
        return zip(*res)

    def save_points(self, intg, prior):
        # NOTE: Will save points from other integrations before this if used more than once.
        self._cache.extend( [ rvs for rvs, ratio, rnd in zip(numpy.array(self._rvs).T, intg/prior, numpy.random.uniform(0, 1, len(prior))) if ratio < 1 or 1.0/ratio < rnd ] )

    # FIXME: Remove *args -- we'll use the function signature instead
    def integrate(self, func, *args, **kwargs):
        """
        Integrate func, by using n sample points. Right now, all params defined must be passed to args must be provided, but this will change soon.

        Limitations:
            func's signature must contain all parameters currently defined by the sampler, and with the same names. This is required so that the sample values can be passed consistently.

        kwargs:
        nmax -- total allowed number of sample points, will throw a warning if this number is reached before neff.
        neff -- Effective samples to collect before terminating. If not given, assume infinity
        n -- Number of samples to integrate in a 'chunk' -- default is 1000

        Pinning a value: By specifying a kwarg with the same of an existing parameter, it is possible to "pin" it. The sample draws will always be that value, and the sampling prior will use a delta function at that value.
        """

        #
        # Pin values
        #
        tempcdfdict, temppdfdict, temppriordict = {}, {}, {}
        for p, val in kwargs.iteritems():
            if p in self.params:
                # Store the previous pdf/cdf in case it's already defined
                tempcdfdict[p] = self.cdf_inv[p]
                temppdfdict[p] = self.pdf[p]
                temppriordict[p] = self.prior_pdf[p]
                # Set a new one to always return the same value
                self.pdf[p] = functools.partial(delta_func_pdf_vector, val)
                self.prior_pdf[p] = functools.partial(delta_func_pdf_vector, val)
                self.cdf_inv[p] = functools.partial(delta_func_samp_vector, val)

        # put it back in the args
        #args = tuple(list(args) + filter(lambda p: p in self.params, kwargs.keys()))
        # This is a semi-hack to ensure that the integrand is called with
        # the arguments in the right order
        # FIXME: How dangerous is this?
        args = func.func_code.co_varnames[:func.func_code.co_argcount]
        if set(args) & self.params != set(args):
            raise ValueError("All integrand variables must be represented by integral parameters.")
        
        #
        # Determine stopping conditions
        #
        nmax = kwargs["nmax"] if kwargs.has_key("nmax") else float("inf")
        neff = kwargs["neff"] if kwargs.has_key("neff") else numpy.float128("inf")
        n = kwargs["n"] if kwargs.has_key("n") else min(1000, nmax)
        peakExpected = kwargs["igrandmax"] if kwargs.has_key("igrandmax") else 0   # Do integral as L/e^peakExpected, if possible
        fracCrit = kwargs['igrand_threshold_fraction'] if kwargs.has_key('igrand_threshold_fraction') else 0 # default is to return all
        bReturnPoints = kwargs['full_output'] if kwargs.has_key('full_output') else False
        bUseMultiprocessing = kwargs['use_multiprocessing'] if kwargs.has_key('use_multiprocessing') else False
        nProcesses = kwargs['nprocesses'] if kwargs.has_key('nprocesses') else 2
        bShowEvaluationLog = kwargs['verbose'] if kwargs.has_key('verbose') else False
        bShowEveryEvaluation = kwargs['extremely_verbose'] if kwargs.has_key('extremely_verbose') else False

        if bShowEvaluationLog:
            print " .... mcsampler : providing verbose output ..... "
        if bUseMultiprocessing:
            if rosDebugMessages:
                print " Initiating multiprocessor pool : ", nProcesses
            p = Pool(nProcesses)

        int_val1 = numpy.float128(0)
        ntotal = 0
        maxval = -float("Inf")
        maxlnL = -float("Inf")
        eff_samp = 0
        mean, std = None, None 

        #
        # TODO: Allocate memory to return values
        #
        if bReturnPoints:
            theGoodPoints = numpy.zeros((nmax,len(args)))
            theGoodlnL = numpy.zeros(nmax)

        # Need FULL history to calculate neff!  No substitutes!
        # Be careful to allocate the larger of n and nmax
        if nmax < float("inf"):
            nbinsToStore = int(numpy.max([nmax,n]))
        else:
            nBinsToStore = 1e7    # don't store everything! stop!
            nmax = nBinsToStore
        theIntegrandFull = numpy.zeros(nmax,dtype=numpy.float128)
        theMaxFull = numpy.zeros(nmax,dtype=numpy.float128)


        if bShowEvaluationLog:
            print "iteration Neff  rhoMax rhoExpected  sqrt(2*Lmarg)  Lmarg"
        nEval =0

        while eff_samp < neff and ntotal < nmax:
            # Draw our sample points
            p_s, p_prior, rv = self.draw(n, *args)
                        
            # Calculate the overall p_s assuming each pdf is independent
            joint_p_s = numpy.prod(p_s, axis=0)
            joint_p_prior = numpy.prod(p_prior, axis=0)
            # ROS fix: Underflow issue: prevent probability from being zero!  This only weakly distorts our result in implausible regions
            # Be very careful: distance prior is in SI units,so the natural scale is 1/(10)^6 * 1/(10^24)
            # FIXME: Non-portable change, breaks universality of the integration.
            joint_p_s  = numpy.maximum(numpy.ones(len(joint_p_s))*1e-50,joint_p_s)

            numpy.testing.assert_array_less(0,joint_p_s)        # >0!  (CANNOT be zero or negative for any sample point, else disaster. Human errors happen.)
            numpy.testing.assert_array_less(0,joint_p_prior)   # >0!  (could be zero if needed.)
            if len(rv[0].shape) != 1:
                rv = rv[0]
            if bUseMultiprocessing:
                fval = p.map(lambda x : func(*x), numpy.transpose(rv))
            else:
                fval = func(*rv)
            int_val = fval*joint_p_prior /joint_p_s
            if bShowEveryEvaluation:
                for i in range(n):
                    print " Evaluation details: p,ps, L = ", joint_p_prior[i], joint_p_s[i], fval[i]

            # Calculate max L (a useful convergence feature) for debug reporting.  Not used for integration
            # Try to avoid nan's
            maxlnL = numpy.log(numpy.max([numpy.exp(maxlnL), numpy.max(fval),numpy.exp(-100)]))   # note if f<0, this will return nearly 0
            # Calculate the effective samples via max over the current evaluations
            # Requires populationg theIntegrandFull, a history of all integrand evaluations.
            maxval = [max(maxval, int_val[0]) if int_val[0] != 0 else maxval]
            for v in int_val[1:]:
                maxval.append( v if v > maxval[-1] and v != 0 else maxval[-1] )
                for i in range(0, int(n)-1):
                    theIntegrandFull[nEval+i] = int_val[i]  # FIXME: Could do this by using maxval[-1] intelligently, rather than storing and resumming all
#                        theIntegrandMaxSoFar = numpy.maximum.accumulate(theIntegrandFull) # For debugging only.  FIXME: should split into max over new data and old

#           eff_samp = (int_val.cumsum()/maxval)[-1] + eff_samp   # ROS: This is wrong (monotonic over blocks of size 'n').  neff can reset to 1 at any time.
            eff_samp = numpy.sum((theIntegrandFull/maxval[-1])[:(ntotal+n-1)])
            # FIXME: Need to bring in the running stddev here
            var = cumvar(int_val, mean, std, ntotal)[-1]
            # FIXME: Reenable caching
            #self.save_points(int_val, joint_p_s)
            #print "%d samples saved" % len(self._cache)
            int_val1 += int_val.sum()
            ntotal += n
            mean = int_val1
            maxval = maxval[-1]
            if bShowEvaluationLog:
                print " :",  ntotal, eff_samp, numpy.sqrt(2*maxlnL), numpy.sqrt(2*peakExpected), numpy.sqrt(2*numpy.log(int_val1/ntotal)), int_val1/ntotal

            if ntotal >= nmax and neff != float("inf"):
                print >>sys.stderr, "WARNING: User requested maximum number of samples reached... bailing."

            # Store our sample points
            if bReturnPoints:
                for i in range(0, int(n)):
                    theGoodPoints[nEval+i] = numpy.transpose(rv)[i]
                    theGoodlnL[nEval+i] = numpy.log(fval[i])
            nEval +=n  # duplicate variable to ntotal.  Need to disentangle


        # If we were pinning any values, undo the changes we did before
        self.cdf_inv.update(tempcdfdict)
        self.pdf.update(temppdfdict)
        self.prior_pdf.update(temppriordict)

        # Select points to be returned.
        # Downselect the points passed back: only use high likelihood values. (Hardcoded threshold specific to our problem. Return of these points should probably be optional)
        if bReturnPoints:
            if fracCrit > 0:
                lnLcrit = numpy.power(fracCrit*numpy.sqrt(2*maxlnL),2)/2  # fraction of the SNR being returned
            else:
                lnLcrit = -100  # return everything
            datReduced = numpy.array([ list(theGoodPoints[i])+ [theGoodlnL[i]] for i in range(nEval) if theGoodlnL[i] > lnLcrit ])

            #  Note size is TRUNCATED: only re-evaluated every n points!
            # Need to stretch the buffer, so I have one Lmarg per evaluation
#           LmargArrayRaw = numpy.cumsum(int_val)/(numpy.arange(1,len(int_val)+1)) # array of partial sums.
            LmargArrayRaw = numpy.cumsum(theIntegrandFull)/(numpy.arange(1,len(theIntegrandFull)+1)) # array of partial sums.
            LmargArray = LmargArrayRaw
            # numpy.zeros(nEval)
            # LmargArray[0] = LmargArrayRaw[0]
            # for i in numpy.arange(1,nEval-1):
            #         LmargArray[i+1] == LmargArray[i]
            #         if numpy.mod(i , n ==0):
            #                 LmargArray[i+1] == LmargArrayRaw[(i-1)/n]

            return int_val1/ntotal, var/ntotal, datReduced, numpy.log(LmargArray), eff_samp
        else:
            return int_val1/ntotal, var/ntotal
                

### UTILITIES: Predefined distributions
#  Be careful: vectorization is not always implemented consistently in new versions of numpy
def uniform_samp(a, b, x):   # I prefer to vectorize with the same call for all functions, rather than hardcode vectorization
        if  x>a and x<b:
                return 1/(b-a)
        else:
                return 0
#uniform_samp_vector = numpy.vectorize(uniform_samp,excluded=['a','b'],otypes=[numpy.float])
uniform_samp_vector = numpy.vectorize(uniform_samp,otypes=[numpy.float])

# syntatic sugar : predefine the most common distributions
uniform_samp_phase = numpy.vectorize(lambda x: 1/(2*numpy.pi))
uniform_samp_psi = numpy.vectorize(lambda x: 1/(numpy.pi))
uniform_samp_theta = numpy.vectorize(lambda x: numpy.sin(x)/(2))
uniform_samp_dec = numpy.vectorize(lambda x: numpy.cos(x)/(2))

def quadratic_samp(rmax,x):
        if x<rmax:
                return x**2/(3*rmax**3)
        else:
                return 0

quadratic_samp_vector = numpy.vectorize(quadratic_samp, otypes=[numpy.float])

def inv_uniform_cdf(a, b, x):
    return (b-a)*x+a

def gauss_samp(mu, std, x):
    return 1.0/numpy.sqrt(2*numpy.pi*std**2)*numpy.exp(-(x-mu)**2/2/std**2)

def gauss_samp_withfloor(mu, std, myfloor, x):
    return 1.0/numpy.sqrt(2*numpy.pi*std**2)*numpy.exp(-(x-mu)**2/2/std**2) + myfloor

#gauss_samp_withfloor_vector = numpy.vectorize(gauss_samp_withfloor,excluded=['mu','std','myfloor'],otypes=[numpy.float])
gauss_samp_withfloor_vector = numpy.vectorize(gauss_samp_withfloor,otypes=[numpy.float])


def cos_samp(x):
        return numpy.sin(x)/2   # x from 0, pi

def dec_samp(x):
        return numpy.sin(x+numpy.pi/2)/2   # x from 0, pi

cos_samp_vector = numpy.vectorize(cos_samp,otypes=[numpy.float])
dec_samp_vector = numpy.vectorize(dec_samp,otypes=[numpy.float])

def pseudo_dist_samp(r0,r):
        return r*r*numpy.exp( - (r0/r)*(r0/r)/2. + r0/r)+0.01  # put a floor on probability, so we converge. Note this floor only cuts out NEARBY distances

#pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,excluded=['r0'],otypes=[numpy.float])
pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,otypes=[numpy.float])

def delta_func_pdf(x_0, x):
    return 1.0 if x == x_0 else 0.0

delta_func_pdf_vector = numpy.vectorize(delta_func_pdf, otypes=[numpy.float])

def delta_func_samp(x_0, x):
    return x_0

delta_func_samp_vector = numpy.vectorize(delta_func_samp, otypes=[numpy.float])

def sky_rejection(skymap, ra_in, dec_in, massp=1.0):
    """
    Do rejection sampling of the skymap PDF, restricted to the greatest XX % of the mass, ra_in and dec_in will be returned, replaced with the new sample points.
    """

    res = healpy.npix2nside(len(skymap))
    pdf_sorted = sorted([(p, i) for i, p in enumerate(skymap)], reverse=True)
    valid_points = []
    cdf, np = 0, 0
    for p, i in pdf_sorted:
        valid_points.append( healpy.pix2ang(res, i) )
        cdf += p
        np += 1
        if cdf > massp:
            break

    i = 0
    while i < len(ra_in):
        rnd_n = numpy.random.randint(0, np)
        trial = numpy.random.uniform(0, pdf_sorted[0][0])
        #print i, trial, pdf_sorted[rnd_n] 
        # TODO: Ensure (ra, dec) within bounds
        if trial < pdf_sorted[rnd_n][0]:
            dec_in[i], ra_in[i] = valid_points[rnd_n]
            i += 1
    dec_in -= numpy.pi/2
    # FIXME: How does this get reversed?
    dec_in *= -1
    return numpy.array([ra_in, dec_in])
#pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,excluded=['r0'],otypes=[numpy.float])
pseudo_dist_samp_vector = numpy.vectorize(pseudo_dist_samp,otypes=[numpy.float])


def sanityCheckSamplerIntegrateUnity(sampler,*args,**kwargs):
        return sampler.integrate(lambda *args: 1,*args,**kwargs)
