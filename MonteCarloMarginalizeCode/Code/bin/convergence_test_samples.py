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
import hashlib
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
parser.add_argument("--js-threshold",default=0.002,type=float, help="[js_lame] threshold on the bounded-domain JS (base-2, squared) of each transverse parameter.  The null floor of this statistic depends on the KDE bandwidth and so on the shape of the posterior, not on the sample count alone: any fixed value is right only for the samples it was tuned on.  The test measures the floor per parameter (split-half on the current iteration) and warns when this value sits below it; prefer --js-lame-auto-threshold.  Default 0.002.")
parser.add_argument("--quantile-tolerance",default=0.02,type=float, help="[js_lame] relative tolerance on the drift of upper quantiles (see --drift-quantiles) of each transverse parameter, tested against every available lagged iteration in --drift-window.  NOTE: split-half noise on q95 at n=5e3 is ~1.2-1.7 percent (median), so this tolerance is only clean if interim posteriors have >~2e4 samples; at 5e3 it is deliberately conservative (extra iterations, never premature stop).  Default 0.02.")
parser.add_argument("--drift-window",default=3,type=int, help="[js_lame] how many previous iterations to test quantile drift against (files located by the posterior_samples-N.dat naming convention next to the first --samples argument).  Slow monotone tail drift is invisible in one-step statistics but accumulates over the window.  Default 3.")
parser.add_argument("--drift-quantiles",default="90,95", help="[js_lame] comma-separated upper percentiles whose relative drift is tested.  Default '90,95'.")
parser.add_argument("--transverse-parameter", action='append', help="[js_lame] parameters (subset of --parameter) treated as bounded transverse parameters.  Default: any of chi1_perp,chi2_perp,chi_p,a1,a2,chi1,chi2 present in --parameter.")
parser.add_argument("--js-lame-auto-threshold",action='store_true', help="[js_lame] derive --threshold/--js-threshold/--quantile-tolerance from the noise floor at the number of DISTINCT samples actually supplied, instead of using the fixed values.  The lame and quantile floors come from the measured 1/n and 1/sqrt(n) fits; the JS floor is measured directly for each transverse parameter by splitting the current posterior in half, because it tracks the KDE bandwidth and hence the shape of the posterior.  The floor moves with n (JS and lame as 1/n, quantile drift as 1/sqrt(n)), so a fixed threshold is right at exactly one sample size and wrong everywhere else: the shipped defaults sit 15-50x below the floor at the honest per-worker supply (~800 distinct), where they fire on pure noise and the gate never converges.  Strongly recommended.")
parser.add_argument("--js-lame-noise-safety",default=1.5,type=float, help="[js_lame] multiple of the p95 noise floor used as the threshold under --js-lame-auto-threshold.  Default 1.5, which reproduces the thresholds recommended in the measured-noise study.")
parser.add_argument("--js-lame-n-distinct",default=None,type=int, help="[js_lame] override the distinct-sample count used to set the noise floor.  Use this when the test is handed a POOLED posterior whose distinct count is known from the export sidecars (+annotation_export.dat); otherwise it is counted from the supplied rows.")
parser.add_argument("--js-lame-require-lags",action='store_true', help="[js_lame] refuse to report convergence until the --drift-window lag history actually exists.  Without this, the first couple of (sub-)iterations have no lagged posteriors and the test quietly falls back to a one-step statistic that cannot see the tail drift -- which is how the shipped 'lame' gate stopped the nested loop at sub-iteration 2-3 of ~50.")
parser.add_argument("--js-lame-reference-drift",default=0.045,type=float, help="[js_lame] the per-iteration upper-quantile drift the gate is REQUIRED to be able to see, used only to report whether it can.  Default 0.045, the measured S240629by chi1_perp widening rate.  Drift compounds over the lag window, so the gate sees the reference signal only if the quantile threshold is below (1+d)**drift_window - 1; when it is not, the test is warned to be blind and the fix is more pooled workers or a longer window.")
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

def js_lame_rng(*arrays):
    """Deterministic RNG for the js_lame estimators, seeded from the sample VALUES themselves.

    Nothing in the stop decision may depend on process-global RNG state.  Both the JS statistic
    (resampled draws) and, under --js-lame-auto-threshold, the threshold it is compared against
    (split-half null) are Monte Carlo estimates; scored from np.random they come out differently
    every time the job runs, so the same posterior files can be converged on one attempt and not
    converged on the next -- a Condor retry of an identical iteration is enough to flip it, and the
    workflow then terminates at a different iteration than an identical rerun.  A fixed constant
    seed would make each call reproducible but would also reuse one partition/draw pattern for
    every parameter and every iteration; keying on the data keeps those estimates independent of
    each other while making each of them a function of its inputs alone.

    Uses the legacy RandomState because its stream is guaranteed stable across numpy versions,
    so the same inputs score the same way on a resumed or relocated run.
    """
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=float))
        h.update(repr(a.shape).encode('utf-8'))
        h.update(a.tobytes())
    return np.random.RandomState(int.from_bytes(h.digest()[:4], 'big'))


def reflected_kde_pdf(x, samples, lo, hi):
    """Reflected-boundary KDE for samples on [lo,hi], evaluated at x.

    The bandwidth is taken from the UNREFLECTED samples.  Fitting gaussian_kde to the
    concatenated block [a, 2*lo-a, 2*hi-a] instead lets Scott's rule see the spread of that
    block, which spans [2*lo-hi, 2*hi-lo]: for a posterior concentrated well inside the domain
    -- exactly the transverse-spin case this test is for -- that inflates the bandwidth several
    fold and smooths away the between-iteration shape change the JS component exists to detect.
    Since bounded parameters are excluded from the Gaussian 'lame' block and the drift component
    only watches q90/q95, an over-smoothed JS is a direct route to premature convergence.

    Because the Gaussian kernel is symmetric, reflecting the DATA about a boundary is identical
    to reflecting the EVALUATION point, so the mirrored contributions are just added here; the
    result is the usual reflection estimator at the sample-scale bandwidth.
    Returns None when the draw has no usable spread (KDE undefined) -- see calculate_js_bounded.
    """
    samples = np.asarray(samples, dtype=float)
    if len(samples) < 2 or not np.isfinite(samples).all() or np.std(samples) <= 0:
        return None
    try:
        kde = gaussian_kde(samples)
    except np.linalg.LinAlgError:   # singular covariance (numerically degenerate draw)
        return None
    return kde(x) + kde(2*lo - x) + kde(2*hi - x)


def calculate_js_bounded(A, B, lo=0.0, hi=1.0, ntests=20, xsteps=100, rng=None):
    """
    Bounded-domain-aware JS (base-2, squared) for a parameter on [lo,hi] (e.g. chi1_perp >= 0):
    reflect the samples about both boundaries before the KDE and evaluate only inside [lo,hi],
    so edge-piling at the boundary does not leak probability mass and bias the estimate.
    (The plain calculate_js KDE is biased for edge-piled bounded variables.)
    The reflection is applied at the bandwidth of the unreflected samples: see reflected_kde_pdf.
    The resampling draws come from a deterministic RNG keyed to A and B (js_lame_rng) unless a
    caller supplies one, so re-scoring the same two posteriors always returns the same value.
    """
    if rng is None:
        rng = js_lame_rng(A, B)
    js_array = np.zeros(ntests)
    n = min(len(A), len(B))
    x = np.linspace(lo, hi, xsteps)
    for j in range(ntests):
        a = rng.choice(A, size=n, replace=False)
        b = rng.choice(B, size=n, replace=False)
        pa = reflected_kde_pdf(x, a, lo, hi); pb = reflected_kde_pdf(x, b, lo, hi)
        if pa is None or pb is None:
            # An exactly-constant draw (e.g. a column with no transverse spin at all) has no
            # KDE.  Call it identical only when both sides are the SAME point mass; otherwise
            # report the maximum (1 bit), i.e. not converged.  Never certify convergence from
            # a density estimate that could not be formed.
            same = (np.std(a) <= 0 and np.std(b) <= 0 and np.isclose(np.mean(a), np.mean(b)))
            js_array[j] = 0.0 if same else 1.0
            continue
        js_array[j] = np.nan_to_num(np.power(jensenshannon(pa, pb, base=2), 2))
    return np.median(js_array)


# js_lame parameter classes.  Bounded transverse parameters get the JS+drift treatment;
# circular parameters are EXCLUDED from the Gaussian-moment (lame) block entirely, since
# Gaussian moments of a circular variable are not meaningful.
JS_LAME_BOUNDED_DEFAULT = ['chi1_perp', 'chi2_perp', 'chi_p', 'a1', 'a2', 'chi1', 'chi2']
JS_LAME_CIRCULAR = ['phi1', 'phi2', 'phi12', 'phiJL', 'psiJ', 'phiorb', 'psi']

# p95 noise floors of the js_lame components under the null (two independent draws of the SAME
# converged posterior), fitted to a K=400-pair bootstrap over n_distinct = 300..20000:
#
#   n_distinct |   js p95 |  lame p95 | dq90 p95 | dq95 p95
#         300  |  0.0798  |  0.0830   |  0.1538  |  0.1597
#         800  |  0.0313  |  0.0319   |  0.0981  |  0.1013
#        2000  |  0.0126  |  0.0138   |  0.0621  |  0.0645
#        5000  |  0.0048  |  0.0054   |  0.0417  |  0.0441
#       20000  |  0.0012  |  0.0012   |  0.0197  |  0.0217
#
# JS and lame are both squared distances between densities, so their null scales as 1/n
# (n*js p95 is 24-25 across the whole range); a quantile position scales as 1/sqrt(n)
# (sqrt(n)*dq95 p95 is 2.8-3.1).  Constants are taken at the conservative end of each fit.
#
# CAVEAT on the js column: those numbers were fitted with the older bounded-JS estimator, which
# took its KDE bandwidth from the reflected block and was therefore over-smoothed (see
# reflected_kde_pdf).  A sharper kernel has a HIGHER null, and by an amount that depends on the
# bandwidth and hence on the shape of the posterior -- not on n alone -- so no single refitted
# constant would be right for the corrected estimator either.  The js floor is therefore MEASURED
# at run time by js_bounded_null_floor, and JS_LAME_NOISE_JS_COEFF survives only as a fallback for
# when there are too few distinct values to split; treat it as a LOWER bound on the true floor.
# The lame and dq estimators are unchanged, so their constants still apply as measured.
JS_LAME_NOISE_JS_COEFF   = 25.0   # js p95   ~ COEFF / n   (superseded estimator; fallback only)
JS_LAME_NOISE_LAME_COEFF = 25.5   # lame p95 ~ COEFF / n
JS_LAME_NOISE_DQ_COEFF   = 3.1    # dq p95   ~ COEFF / sqrt(n)

# Run-time measurement of the js null: how many disjoint split-half pairs to score, which
# quantile of them to report (p95, to match the tabulated components), and the smallest half
# that is worth measuring at all.
JS_LAME_NULL_SPLITS   = 20
JS_LAME_NULL_QUANTILE = 95
JS_LAME_NULL_MIN_HALF = 50


def js_lame_noise_floor(n_distinct):
    """p95 null noise floor of each js_lame component at n_distinct samples.  See the table above.
    The 'js' entry is the fallback constant only; prefer the measured js_bounded_null_floor."""
    n = max(float(n_distinct), 1.0)
    return {'js':   JS_LAME_NOISE_JS_COEFF / n,
            'lame': JS_LAME_NOISE_LAME_COEFF / n,
            'dq':   JS_LAME_NOISE_DQ_COEFF / np.sqrt(n)}


def js_bounded_null_floor(samples, lo, hi, n_target,
                          n_splits=JS_LAME_NULL_SPLITS, quantile=JS_LAME_NULL_QUANTILE,
                          xsteps=100):
    """MEASURED p95 null floor of calculate_js_bounded for this parameter, scaled to n_target
    samples per side.  Returns None if there are too few distinct values to split.

    The null is 'two independent draws of the same distribution', so it is measured here the same
    way the tabulated study measured it: split one posterior into two halves and score them
    against each other, repeatedly.  Measuring rather than tabulating keeps the threshold
    consistent with whatever the estimator actually does -- the JS null depends on the KDE
    bandwidth, so it moves when the estimator or the posterior shape changes, and a constant
    fitted against one estimator on one set of samples silently stops describing either.

    Two details matter:
      - split the DISTINCT values, not the rows.  When CIP's export has padded a request with
        duplicate rows, a row-wise split puts the same point on both sides, the halves stop being
        independent, and the measured floor collapses toward zero -- which would put the threshold
        below the true noise and hang the gate forever on pure noise.
      - the halves hold len(unique)//2 points each, while the real comparison is scored at
        n_target; the null scales as 1/n per side, so rescale by (half / n_target).
      - the partitions are drawn from a deterministic RNG keyed to these samples (js_lame_rng),
        never from np.random.  This measurement IS the pass threshold under
        --js-lame-auto-threshold, so an unseeded partition would score the same posterior files
        against a different threshold on every attempt -- including a Condor retry of the same
        iteration -- and identical runs could stop at different iterations.
    """
    u = np.unique(np.asarray(samples, dtype=float))
    m = len(u) // 2
    if m < JS_LAME_NULL_MIN_HALF:
        return None
    rng = js_lame_rng(u)
    vals = np.zeros(n_splits)
    for j in range(n_splits):
        perm = rng.permutation(len(u))
        vals[j] = calculate_js_bounded(u[perm[:m]], u[perm[m:2*m]], lo=lo, hi=hi,
                                       ntests=1, xsteps=xsteps, rng=rng)
    return np.percentile(vals, quantile) * float(m) / max(float(n_target), 1.0)


def js_lame_count_distinct(*arrays):
    """Smallest number of DISTINCT rows among the supplied sample blocks.

    The noise floor is set by INFORMATION, not by row count.  CIP's export pads its request with
    duplicates once the requested count exceeds the honest supply, so a 20000-row posterior can
    carry only ~800 distinct points; keying the thresholds to the row count would then put them
    ~25x below the true floor.  Counting distinct rows here makes the gate honest whether or not
    --posterior-unique-draw was used upstream.
    """
    counts = []
    for a in arrays:
        a = np.atleast_2d(np.asarray(a, dtype=float))
        try:
            counts.append(len(np.unique(a, axis=0)))
        except TypeError:   # numpy too old for axis= on unique
            counts.append(len(a))
    return min(counts) if counts else 0


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
    Each component's threshold is either the fixed CLI value or, under
    --js-lame-auto-threshold, a multiple of the p95 null noise floor: tabulated at the DISTINCT
    sample count supplied for lame and dq (js_lame_noise_floor), and measured per parameter for
    the bandwidth-dependent JS component (js_bounded_null_floor).
    Returns a value scaled so the standard 'val < opts.threshold' semantics apply:
      val = opts.threshold * max_over_components(component/its_threshold).
    """
    idx = {p: i for i, p in enumerate(param_list)}
    if opts.transverse_parameter:
        bounded = [p for p in opts.transverse_parameter if p in idx]
    else:
        bounded = [p for p in JS_LAME_BOUNDED_DEFAULT if p in idx]
    gaussian_params = [p for p in param_list if p not in bounded and p not in JS_LAME_CIRCULAR]

    # Thresholds keyed to the noise floor at the DISTINCT sample count actually supplied.
    n_distinct = opts.js_lame_n_distinct if opts.js_lame_n_distinct else js_lame_count_distinct(dat1, dat2)
    floor = js_lame_noise_floor(n_distinct)
    # The js floor is measured per parameter below (it depends on the KDE bandwidth, so it is not
    # a function of n alone); thr_js here is only the fallback used when that measurement cannot
    # be made.
    if opts.js_lame_auto_threshold:
        thr_lame = opts.js_lame_noise_safety * floor['lame']
        thr_js   = opts.js_lame_noise_safety * floor['js']
        thr_dq   = opts.js_lame_noise_safety * floor['dq']
        print(" js_lame: n_distinct %d -> p95 noise floor lame %.2e dq %.2e (js floor measured per parameter)" % (
            n_distinct, floor['lame'], floor['dq']))
        print("          thresholds (%.2f x floor): lame %.2e dq %.2e" % (
            opts.js_lame_noise_safety, thr_lame, thr_dq))
    else:
        thr_lame, thr_js, thr_dq = opts.threshold, opts.js_threshold, opts.quantile_tolerance
        below = [name for name, thr, fl in (('--threshold', thr_lame, floor['lame']),
                                            ('--quantile-tolerance', thr_dq, floor['dq'])) if thr < fl]
        if below:
            print(" js_lame WARNING: at n_distinct %d the p95 null noise floor is lame %.2e dq %.2e,"
                  % (n_distinct, floor['lame'], floor['dq']))
            print("   which is ABOVE the fixed threshold(s) %s -- those components will fire on pure noise and"
                  % ', '.join(below))
            print("   this gate can never report convergence.  Pass --js-lame-auto-threshold, or pool more CIP")
            print("   workers (distinct scales linearly with worker count; ~25 workers reach ~2e4 distinct).")
    # Can this gate actually SEE the failure mode it exists to catch?  Noise is lag-independent
    # while a monotone tail drift compounds, so a reference drift of d per iteration accumulates to
    # (1+d)**L - 1 over a lag of L.  If even the longest available lag stays under the quantile
    # threshold, the gate is blind and will stop the loop early -- the original bug, reintroduced.
    # This is the direction that costs a run, so report it whichever way the thresholds were set.
    lag_max = max(1, opts.drift_window)
    signal_max = (1.0 + opts.js_lame_reference_drift) ** lag_max - 1.0
    if signal_max < thr_dq:
        lag_needed = int(np.ceil(np.log1p(thr_dq) / np.log1p(opts.js_lame_reference_drift)))
        print(" js_lame WARNING: BLIND to the reference drift.  At n_distinct %d the quantile threshold is"
              % n_distinct)
        print("   %.4f, but a %.1f%%/iteration drift only reaches %.4f over the %d-iteration window."
              % (thr_dq, 100 * opts.js_lame_reference_drift, signal_max, lag_max))
        print("   This gate can stop the loop while the tail is still widening.  Fix with more pooled CIP")
        print("   workers (the floor falls as 1/sqrt(n_distinct)) or --drift-window %d." % lag_needed)
    elif opts.verbose:
        print(" js_lame: a %.1f%%/iteration drift reaches %.4f over the %d-iteration window vs threshold %.4f -- detectable."
              % (100 * opts.js_lame_reference_drift, signal_max, lag_max, thr_dq))

    components = {}
    if gaussian_params:
        cols = [idx[p] for p in gaussian_params]
        val_lame = test_lame(dat1[:, cols], dat2[:, cols])
        components['lame(%s)' % ','.join(gaussian_params)] = val_lame / thr_lame
    for p in bounded:
        A = dat1[:, idx[p]]; B = dat2[:, idx[p]]
        hi = max(1.0, np.max(A), np.max(B))
        # Measure this parameter's own JS null floor (split-half on the current iteration) rather
        # than reading it off the tabulated 1/n fit, which was made against a different, more
        # heavily smoothed estimator.
        floor_js = js_bounded_null_floor(A, lo=0.0, hi=hi, n_target=n_distinct)
        floor_js_source = 'measured'
        if floor_js is None:
            floor_js = floor['js']
            floor_js_source = 'fallback'
            print(" js_lame WARNING: %s has fewer than %d distinct values, too few to split and measure a JS null"
                  % (p, 2 * JS_LAME_NULL_MIN_HALF))
            print("   floor; falling back to the tabulated coefficient %.2e, which was fitted against the"
                  % floor_js)
            print("   older over-smoothed estimator and is therefore a LOWER bound on the real floor here.")
        thr_js_p = opts.js_lame_noise_safety * floor_js if opts.js_lame_auto_threshold else thr_js
        if opts.js_lame_auto_threshold or opts.verbose:
            print(" js_lame: p95 JS null floor for %s is %.2e (%s) at n_distinct %d -> threshold %.2e"
                  % (p, floor_js, floor_js_source, n_distinct, thr_js_p))
        if not opts.js_lame_auto_threshold and thr_js_p < floor_js:
            print(" js_lame WARNING: --js-threshold %.2e is BELOW the p95 JS null floor %.2e (%s) for %s,"
                  % (thr_js_p, floor_js, floor_js_source, p))
            print("   so that component fires on pure sampling noise and this gate can never report")
            print("   convergence.  Pass --js-lame-auto-threshold, or pool more CIP workers.")
        val_js = calculate_js_bounded(A, B, lo=0.0, hi=hi)
        components['js(%s)' % p] = val_js / thr_js_p
        for qq in [float(x) for x in opts.drift_quantiles.split(',')]:
            qa = np.percentile(A, qq); qb = np.percentile(B, qq)
            drift = np.abs(qa - qb) / max(np.abs(qa), np.abs(qb), 1e-10)
            components['dq%g(%s,lag1)' % (qq, p)] = drift / thr_dq

    # lagged drift: locate previous-iteration posterior files by naming convention
    n_lags_used = 0
    if samples_path_current and opts.drift_window > 1:
        import re, os
        m = re.search(r'^(.*posterior_samples-)(\d+)(\.dat)$', samples_path_current)
        if not m:
            print(" js_lame WARNING: --samples '%s' does not match the posterior_samples-N.dat naming that"
                  % samples_path_current)
            print("   locates lagged iterations, so NO lag window is in effect and this is a one-step test.")
        if m:
            it_now = int(m.group(2))
            for lag in range(2, opts.drift_window + 1):
                f_lag = "%s%d%s" % (m.group(1), it_now - lag, m.group(3))
                if it_now - lag < 1 or not os.path.exists(f_lag):
                    continue
                try:
                    s_lag = read_and_prepare(f_lag)
                    n_lags_used += 1
                    for p in bounded:
                        if p not in s_lag.dtype.names:
                            continue
                        A = dat1[:, idx[p]]; C = np.asarray(s_lag[p], dtype=float)
                        for qq in [float(x) for x in opts.drift_quantiles.split(',')]:
                            qa = np.percentile(A, qq); qc = np.percentile(C, qq)
                            drift = np.abs(qa - qc) / max(np.abs(qa), np.abs(qc), 1e-10)
                            components['dq%g(%s,lag%d)' % (qq, p, lag)] = drift / thr_dq
                except Exception as e:
                    print("   js_lame: could not use lagged file %s : %s" % (f_lag, e))

    # Refuse to certify convergence before the lag window is actually populated.  With a window of
    # W, sub-iteration 2 or 3 has no lagged files at all, so the test silently degrades to the
    # one-step statistic that the noise-floor study shows cannot see the drift -- and stopping at
    # sub-iteration 2-3 of ~50 is precisely the failure this method exists to prevent.
    if opts.drift_window > 1 and n_lags_used < opts.drift_window - 1:
        msg = ("only %d of %d lagged iterations available -- the drift window is not populated, so"
               " this is effectively a one-step test" % (n_lags_used, opts.drift_window - 1))
        if opts.js_lame_require_lags:
            print(" js_lame: %s; reporting NOT CONVERGED by --js-lame-require-lags." % msg)
            return np.inf
        print(" js_lame WARNING: %s." % msg)
        print("   A one-step test at this supply cannot see the tail drift; consider --js-lame-require-lags.")

    worst = max(components, key=components.get)
    print(" js_lame components (value/threshold; converged needs ALL < 1):")
    for k in sorted(components, key=components.get, reverse=True):
        print("    %-28s %.4f" % (k, components[k]))
    print("  js_lame worst: %s (n_distinct %d, lags used %d)" % (worst, n_distinct, n_lags_used))
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
