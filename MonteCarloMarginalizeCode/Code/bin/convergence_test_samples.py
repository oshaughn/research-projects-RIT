#! /usr/bin/env python
#
#
# GOAL
#   - takes two sets of samples, and some parameter(s)
#       - should be able to interchange samples provided with ILE *.xml.gz, *.composite, or posterior samples (preferred).  FLEXIBILITY NOT YET IMPLEMENTED. 
#         Postfix determines behavior
#   - performs specified test, with specified tolerance, to see if they are 'similar enough'
#   - returns FAILURE if test is a success (!), so a condor DAG will terminate
#
# EXAMPLES
# convergence_test_samples.py --samples GW170823_pure_NR_and_NRSur7dq2_lmax3_fmin20_C02_cleaned_alignedspin_zprior.dat --samples GW170823_pure_NR_and_NRSur7dq2_lmax3_fmin20_C02_cleaned_alignedspin_zprior.dat --parameter m1 --parameter m2   # test samples against themselves, must return 0!
#
# RESOURCES
#   Based on code in util_DriverIterateILEFitPosterior*.py

import numpy as np
import argparse
import scipy.stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
import numpy.linalg as la
import sys

from RIFT.misc.samples_utils import add_field
import RIFT.misc.samples_utils
from RIFT.misc import hyperpipeline_io as hpio

parser = argparse.ArgumentParser()
parser.add_argument("--samples", action='append', help="Samples used in convergence test")
parser.add_argument("--parameter", action='append', help="Parameters used in convergence test")
parser.add_argument("--parameter-range", action='append', help="Parameter ranges used in convergence test (used if KDEs or similar knowledge of the PDF is needed). If used, must specify for ALL variables, in order")
parser.add_argument("--method", default='lame',  help="Test to perform: lame|KS_1d|KL_1d|JS|js_lame.  js_lame = lame on the unbounded parameters (mc,eta,xi) AND, on each bounded transverse parameter (chi1_perp,...), a bounded-domain-aware JS plus an upper-quantile DRIFT test over a lag window of previous iterations.  Converged only if ALL components pass (transverse-spin tail-contraction diagnostic).")
parser.add_argument("--threshold",default=0.01,type=float,  help="Manual threshold for the test being performed. (If not specified, the success condition is determined by default for that diagnostic, based on the samples size and properties).  Try 0.01")
parser.add_argument("--js-threshold",default=0.002,type=float, help="[js_lame] threshold on the bounded-domain JS (base-2, squared) of each transverse parameter.  Split-half noise floor at n~5e3 is ~1e-4 (p90 ~4e-4); active tail motion is >~2e-3.  Default 0.002.")
parser.add_argument("--quantile-tolerance",default=0.02,type=float, help="[js_lame] relative tolerance on the drift of upper quantiles (see --drift-quantiles) of each transverse parameter, tested against every available lagged iteration in --drift-window.  NOTE: split-half noise on q95 at n=5e3 is ~1.2-1.7 percent (median), so this tolerance is only clean if interim posteriors have >~2e4 samples; at 5e3 it is deliberately conservative (extra iterations, never premature stop).  Default 0.02.")
parser.add_argument("--drift-window",default=3,type=int, help="[js_lame] how many previous iterations to test quantile drift against (files located by the posterior_samples-N.dat naming convention next to the first --samples argument).  Slow monotone tail drift is invisible in one-step statistics but accumulates over the window.  Default 3.")
parser.add_argument("--drift-quantiles",default="90,95", help="[js_lame] comma-separated upper percentiles whose relative drift is tested.  Default '90,95'.")
parser.add_argument("--transverse-parameter", action='append', help="[js_lame] parameters (subset of --parameter) treated as bounded transverse parameters.  Default: any of chi1_perp,chi2_perp,chi_p,a1,a2,chi1,chi2 present in --parameter.")
parser.add_argument("--test-output",  help="Filename to return output. Result is a scalar >=0 and ideally <=1.  Closer to 0 should be good. Second column is the diagnostic, first column is 0 or 1 (success or failure)")
parser.add_argument("--always-succeed",action='store_true',help="Test output is always success.  Use for plotting convergence diagnostics so jobs insured to run for many iterations.")
parser.add_argument("--iteration-threshold",default=0,type=int,help="Test is applied if iteration >= iteration-threshold. Default is 0")
parser.add_argument("--iteration",default=0,type=int,help="Current reported iteration. Default is 0.")
parser.add_argument("--write-file-on-success",type=str,default="INTRINSIC_CONVERGED",help="Produces an (empty) file with this name if the convergence tests passes.  Note you should pass the FULL PATH to this file if you want it to occur in the run directory for example")
parser.add_argument("--verbose", action='store_true')
opts=  parser.parse_args()

if len(opts.samples)<2:
    print(" Need at least two sets of samples")
    sys.exit(1)

if opts.iteration < opts.iteration_threshold:
    sys.exit(0)



# Test options
#
#   (a) lame: Compute a multivariate gaussian estimate (sample mean and variance), and then use KL divergence between them !
#   (b) KS_1d: One-dimensional KS test on cumulative distribution  
#   (c) KL_1d: One-dimensional KL divergence, using KDE estimate.  Requires bounded domain; parameter bounds can be passed 


def calc_kl(mu_1, mu_2, sigma_1, sigma_2, sigma_1_inv, sigma_2_inv):
    """
    calc_kl : KL divergence for two gaussians.  sigma_1, and sigma_2 are the covariance matricies.
    """
    return 0.5*(np.trace(np.dot(sigma_2_inv,sigma_1))+np.dot(np.dot((mu_2-mu_1).T, sigma_2_inv), (mu_2-mu_1))-len(mu_1)+np.log(la.det(sigma_2)/la.det(sigma_1)))

def calc_kl_scalar(mu_1, mu_2, sigma_1, sigma_2):
    """
    calc_kl : KL divergence for two gaussians.  sigma_1, and sigma_2 are the covariance matricies.
    """
    return np.log(sigma_2/sigma_1) -0.5 +( (mu_1-mu_2)**2 + sigma_1**2)/(2*sigma_2**2)


def test_lame(dat1,dat2):
    """
    Compute a multivariate gaussian estimate (sample mean and variance), and then use KL divergence between them !
    """
    mu_1 = np.mean(dat1,axis=0)
    mu_2 = np.mean(dat2,axis=0)
    sigma_1 = np.cov(dat1.T)
    sigma_2 = np.cov(dat2.T)
    if np.isscalar(mu_1) or len(mu_1)==1:
        return np.asscalar(calc_kl_scalar(mu_1, mu_2, sigma_1, sigma_2))
    else:
        sigma_1_inv = np.linalg.inv(sigma_1)
        sigma_2_inv = np.linalg.inv(sigma_2)
    return calc_kl(mu_1,mu_2, sigma_1, sigma_2, sigma_1_inv, sigma_2_inv)

def test_ks1d(dat1_1d, dat2_1d):
    """
    KS test based on two sample sets.  Uses the KS D value as threshold
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ks_2samp.html
    """
    return scipy.stats.ks_2samp(dat1_1d,dat2_1d)[0]  # return KS statistic

def test_KL1d(dat1_1d,dat2_1d,range1=None, range2=None):
    return None



def calculate_js(samplesA, samplesB, ntests=100, xsteps=100):
    """
    JS (1d) from https://git.ligo.org/pe/O4/bilby_o4_review/-/blob/main/GW150914/run_comparison.py

    Notes:  (a) does 100 tests with random resampling, (b) uses KDE (!) as density estimate, (c) truncates range to smaller of all samples present, does not allow for large range differences
    """
    js_array = np.zeros(ntests)
    for j in range(ntests):
        nsamples = min([len(samplesA), len(samplesB)])
        A = np.random.choice(samplesA, size=nsamples, replace=False)
        B = np.random.choice(samplesB, size=nsamples, replace=False)
        xmin = np.min([np.min(A), np.min(B)])
        xmax = np.max([np.max(A), np.max(B)])
        x = np.linspace(xmin, xmax, xsteps)
        A_pdf = gaussian_kde(A)(x)
        B_pdf = gaussian_kde(B)(x)

        js_array[j] = np.nan_to_num(np.power(jensenshannon(A_pdf, B_pdf,base=2), 2)) # other papers use base 2, not base e

    return np.median(js_array)

def calculate_js_bounded(A, B, lo=0.0, hi=1.0, ntests=20, xsteps=100):
    """
    Bounded-domain-aware JS (base-2, squared) for a parameter on [lo,hi] (e.g. chi1_perp >= 0):
    reflect the samples about both boundaries before the KDE and evaluate only inside [lo,hi],
    so edge-piling at the boundary does not leak probability mass and bias the estimate.
    (The plain calculate_js KDE is biased for edge-piled bounded variables.)
    """
    js_array = np.zeros(ntests)
    n = min(len(A), len(B))
    x = np.linspace(lo, hi, xsteps)
    for j in range(ntests):
        a = np.random.choice(A, size=n, replace=False)
        b = np.random.choice(B, size=n, replace=False)
        aa = np.concatenate([a, 2*lo - a, 2*hi - a])
        bb = np.concatenate([b, 2*lo - b, 2*hi - b])
        pa = gaussian_kde(aa)(x); pb = gaussian_kde(bb)(x)
        js_array[j] = np.nan_to_num(np.power(jensenshannon(pa, pb, base=2), 2))
    return np.median(js_array)


# js_lame parameter classes.  Bounded transverse parameters get the JS+drift treatment;
# circular parameters are EXCLUDED from the Gaussian-moment (lame) block entirely, since
# Gaussian moments of a circular variable are not meaningful.
JS_LAME_BOUNDED_DEFAULT = ['chi1_perp', 'chi2_perp', 'chi_p', 'a1', 'a2', 'chi1', 'chi2']
JS_LAME_CIRCULAR = ['phi1', 'phi2', 'phi12', 'phiJL', 'psiJ', 'phiorb', 'psi']


def test_js_lame(dat1, dat2, param_list, opts, samples_path_current=None):
    """
    Transverse-tail-sharp convergence test (transverse-spin study 2026-07):
      component 1: 'lame' (multivariate-Gaussian KL) on the non-bounded, non-circular
                   parameters (typically mc, eta, xi) vs opts.threshold;
      component 2: bounded-domain JS on each transverse parameter vs opts.js_threshold;
      component 3: relative drift of upper quantiles (opts.drift_quantiles) of each
                   transverse parameter vs opts.quantile_tolerance -- tested not only
                   against the previous iteration but against every available lagged
                   iteration within opts.drift_window (posterior_samples-N.dat naming),
                   because the observed failure mode is SLOW MONOTONE tail drift that is
                   inside the noise floor of any one-step statistic ('lame' passed at
                   0.016<0.02 while chi1_perp's 90% CI was still moving ~4.5%/iteration).
    Returns a value scaled so the standard 'val < opts.threshold' semantics apply:
      val = opts.threshold * max_over_components(component/its_threshold).
    """
    idx = {p: i for i, p in enumerate(param_list)}
    if opts.transverse_parameter:
        bounded = [p for p in opts.transverse_parameter if p in idx]
    else:
        bounded = [p for p in JS_LAME_BOUNDED_DEFAULT if p in idx]
    gaussian_params = [p for p in param_list if p not in bounded and p not in JS_LAME_CIRCULAR]

    components = {}
    if gaussian_params:
        cols = [idx[p] for p in gaussian_params]
        val_lame = test_lame(dat1[:, cols], dat2[:, cols])
        components['lame(%s)' % ','.join(gaussian_params)] = val_lame / opts.threshold
    for p in bounded:
        A = dat1[:, idx[p]]; B = dat2[:, idx[p]]
        hi = max(1.0, np.max(A), np.max(B))
        val_js = calculate_js_bounded(A, B, lo=0.0, hi=hi)
        components['js(%s)' % p] = val_js / opts.js_threshold
        for qq in [float(x) for x in opts.drift_quantiles.split(',')]:
            qa = np.percentile(A, qq); qb = np.percentile(B, qq)
            drift = np.abs(qa - qb) / max(np.abs(qa), np.abs(qb), 1e-10)
            components['dq%g(%s,lag1)' % (qq, p)] = drift / opts.quantile_tolerance

    # lagged drift: locate previous-iteration posterior files by naming convention
    if samples_path_current and opts.drift_window > 1:
        import re, os
        m = re.search(r'^(.*posterior_samples-)(\d+)(\.dat)$', samples_path_current)
        if m:
            it_now = int(m.group(2))
            for lag in range(2, opts.drift_window + 1):
                f_lag = "%s%d%s" % (m.group(1), it_now - lag, m.group(3))
                if it_now - lag < 1 or not os.path.exists(f_lag):
                    continue
                try:
                    s_lag = read_and_prepare(f_lag)
                    for p in bounded:
                        if p not in s_lag.dtype.names:
                            continue
                        A = dat1[:, idx[p]]; C = np.asarray(s_lag[p], dtype=float)
                        for qq in [float(x) for x in opts.drift_quantiles.split(',')]:
                            qa = np.percentile(A, qq); qc = np.percentile(C, qq)
                            drift = np.abs(qa - qc) / max(np.abs(qa), np.abs(qc), 1e-10)
                            components['dq%g(%s,lag%d)' % (qq, p, lag)] = drift / opts.quantile_tolerance
                except Exception as e:
                    print("   js_lame: could not use lagged file %s : %s" % (f_lag, e))

    worst = max(components, key=components.get)
    print(" js_lame components (value/threshold; converged needs ALL < 1):")
    for k in sorted(components, key=components.get, reverse=True):
        print("    %-28s %.4f" % (k, components[k]))
    print("  js_lame worst: %s" % worst)
    return opts.threshold * components[worst]


def test_js_additive(dat1,dat2):
    """
    For all fields in sample, calculate 1d js

    js test from
    https://git.ligo.org/pe/O4/bilby_o4_review/-/blob/main/GW150914/run_comparison.py
    """

    n_dim = len(dat1[0])
    js_net = 0
    for indx in np.arange(n_dim):
        js_net += calculate_js(dat1[:,indx],dat2[:,indx])
    return js_net

# Procedure
def read_samples(fname):
    if hpio.sniff(fname):
        samples, _columns = hpio.read_table(fname)
        return samples
    return np.genfromtxt(fname, names=True)

# Ensure mc/eta exist: hyperpipeline posteriors (RIFT_HYPERPIPELINE_FORMAT) carry
# only m1/m2, and standard_expand_samples does not always add the chirp-mass
# coordinates the convergence test fits -> "no field of name mc".  Derive them.
def _ensure_mc_eta(samples):
    if 'm1' not in samples.dtype.names or 'm2' not in samples.dtype.names:
        return samples
    m1 = samples['m1']; m2 = samples['m2']
    if 'mc' not in samples.dtype.names:
        samples = add_field(samples, [('mc', float)])
        samples['mc'] = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    if 'eta' not in samples.dtype.names:
        samples = add_field(samples, [('eta', float)])
        samples['eta'] = (m1 * m2) / (m1 + m2) ** 2
    return samples

def read_and_prepare(fname):
    """read_samples + the standard field expansion/derivations (mc, eta, xi, chi1/chi2).
    Used identically for the two --samples files and for js_lame's lagged-iteration files."""
    samples = read_samples(fname)
    if 'm1' in samples.dtype.names:
        samples = RIFT.misc.samples_utils.standard_expand_samples(samples)
        if opts.verbose:
            print(" Samples (%s) expanded fields " % fname, samples.dtype.names)
    samples = _ensure_mc_eta(samples)
    if 'm1' in samples.dtype.names:
        if not('xi' in samples.dtype.names):
            if not 'chi_eff' in samples.dtype.names:
                samples = add_field(samples, [('chi_eff',float)]); samples['chi_eff'] = (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["m1"]+samples["m2"])
            samples = add_field(samples, [('xi',float)]); samples['xi'] = (samples["m1"]*samples["a1z"]+samples["m2"]*samples["a2z"])/(samples["m1"]+samples["m2"])
        if not 'chi1' in samples.dtype.names and 'a1x' in samples.dtype.names: # RIFT internal output
            samples = add_field(samples, [('chi1',float),('chi2',float)])
            samples['chi1'] = np.sqrt(samples['a1x']**2+samples['a1y']**2 + samples['a1z']**2)
            samples['chi2'] = np.sqrt(samples['a2x']**2+samples['a2y']**2 + samples['a2z']**2)
    return samples

samples1 = read_and_prepare(opts.samples[0])
samples2 = read_and_prepare(opts.samples[1])


param_names1 = samples1.dtype.names; param_names2 = samples2.dtype.names
npts1 = len(samples1[param_names1[0]])
npts2 = len(samples2[param_names2[0]])  

# Read in data into array.  For now, assume the specific parameters requested are provided.
dat1 = np.empty( (npts1,len(opts.parameter)))
dat2 = np.empty( (npts2,len(opts.parameter)))
indx=0
for param in opts.parameter:
    dat1[:,indx] = samples1[param]
    dat2[:,indx] = samples2[param]
    indx+=1


# Perform test.  Method-name ALIASES: the pipeline wiring (helper_LDG_Events
# --internal-test-convergence-method) documents lowercase names; accept both spellings.
# (Previously '--method ks1d' silently hit the unknown-method branch -> val=inf -> never
# converge, with no loud diagnostic.)
_METHOD_ALIASES = {'ks1d': 'KS_1d', 'kl1d': 'KL_1d', 'kl_1d': 'KL_1d', 'js': 'JS', 'js_additive': 'JS'}
method = _METHOD_ALIASES.get(opts.method, opts.method)

val_test = np.inf
if method == 'lame':
    val_test = test_lame(dat1,dat2)
elif method == 'KS_1d':
    val_test = test_ks1d(dat1[:,0],dat2[:,0])
elif method == 'KL_1d':
    val_test = test_KL1d(dat1[:,0],dat2[:,0])
elif method == 'JS':
    val_test = test_js_additive(dat1,dat2)
elif method == 'js_lame':
    val_test = test_js_lame(dat1, dat2, list(opts.parameter), opts,
                            samples_path_current=opts.samples[0])
else:
    print(" UNKNOWN METHOD '%s' (known: lame KS_1d/ks1d KL_1d JS/js_additive js_lame) -- test value inf, will NEVER report convergence" % opts.method)
if val_test is None:   # e.g. KL_1d is unimplemented; treat as 'no information -> keep going'
    print(" Method '%s' returned no value; treating as not converged" % opts.method)
    val_test = np.inf
print(val_test)

if opts.always_succeed or (opts.threshold is None):
    sys.exit(0)

if (val_test < opts.threshold):
    np.savetxt(opts.write_file_on_success,np.array([]))
    sys.exit(1)
else:
    sys.exit(0)
