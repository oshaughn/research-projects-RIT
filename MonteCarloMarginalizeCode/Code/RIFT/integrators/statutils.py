import numpy
from RIFT.precision import RiftFloat  # platform-portable replacement for np.float128
import scipy.special

__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>, R. O'Shaughnessy"

#
# Stat utilities
#

def welford(x_array, mean=None,var=None,n=0):
    """
    https://www.embeddedrelated.com/showarticle/785.php
    see also https://brenocon.com/blog/2008/11/calculating-running-variance-in-python-and-c/

    No reallocations, unlike 'cumvar' below!
    """
    k = 0 
    M = 0
    S = 0
    if mean and var:
            k+=1+n
            M=mean
            S=var*(n-1)
    for x in x_array:
        k += 1
        Mnext = M + (x - M) / k
        S = S + (x - M)*(x - Mnext)
        M = Mnext
#    return (M, S/(k-1))
    return S/(k-1)

def cumvar(arr, mean=None, var=None, n=0):
	"""
	Numerically stable running sample variance measure. If mean and var are supplied, they will be used as the history values. See 

    http://www.johndcook.com/standard_deviation.html

    for algorithm details.
	"""
	if mean and var:
		m, s = numpy.zeros(len(arr)+1), numpy.zeros(len(arr)+1,dtype=RiftFloat)
		m[0] = mean
		s[0] = var*(n-1)
		buf = numpy.array([0])
	else:
		m, s = numpy.zeros(arr.shape), numpy.zeros(arr.shape,dtype=RiftFloat)
		m[0] = arr[0]
		buf = numpy.array([])

	for i, x in enumerate(numpy.concatenate((buf, arr))):
		if mean is None:
			k = i+1+n
		else:
			k = i+n
		if i == 0: continue
		m[i] = m[i-1] + (x-m[i-1])/k
		s[i] = s[i-1] + (x-m[i-1])*(x-m[i])

	if mean and var:
		return s[1:]/numpy.arange(n, n + len(s)-1)
	else:
		norm = numpy.arange(n, n + len(s))
		norm[0] = 1 # avoid a warning about zero division
		return s/norm

def int_var(samples):
    mean = numpy.mean(samples)
    sq_mean = numpy.mean(samples**2)
    return (sq_mean-mean**2)/(len(samples)-1)


# Alternative implementation that uses a state variable, rather than recomputing every step (as the algorithm above does!)
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
# https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# CONFIRM CORRECTNESS for batched update : we want the 'parallel algorithm' noted there

# What we really want is to make two aggregates, and merge them by the 'parallel variance' expression:
#   - have existing result
#   - compute for new set
#   - update aggregate

def update(existingAggregate, newValues,xpy=numpy):
    if isinstance(newValues, (int, float, complex)):
        # Handle single digits.
        newValues = [newValues]
    (nA, xAmean, M2A) = existingAggregate
    nB = len(newValues)
    xBmean = xpy.mean(newValues)
    M2B = xpy.sum((newValues - xBmean)**2)   # classical problem of overflow ... sum of squares of these quantities, usually integrands, and large.
    
    delta = xBmean - xAmean
    mean = xAmean + delta* nB/(nA+nB)
    M2AB = M2A + M2B + delta**2 * (nA*nB)/(nA+nB)
    return (nA+nB, mean, M2AB)

#     count += len(newValues) 
#     # newvalues - oldMean
#     delta = np.subtract(newValues, [mean] * len(newValues))
#     mean += np.sum(delta / count)
#     # newvalues - newMeant
#     delta2 = np.subtract(newValues, [mean] * len(newValues))
#     M2 += np.sum(delta * delta2)

#     return (count, mean, M2)

def finalize(existingAggregate):
     (count, mean, M2) = existingAggregate
     (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1)) 
     if count < 2:
         return float('nan')
     else:
         return (mean,  sampleVariance)

	
def init_log(newLogValues_orig,special=scipy.special,xpy=numpy):
    """

    fallback mode: if special is fallback, then the aggregate is internally done with the normal numpy calculation!
    """
    logsumexp = special.logsumexp
    xpy_here=xpy
    newLogValues = newLogValues_orig
    if special==scipy.special and xpy != numpy:
        newLogValues = xpy.copy(newLogValues_orig) # copy, so we don't just edit the pointer contents
        newLogValues = xpy.asnumpy(newLogValues)
        xpy_here=numpy
  
    n = len(newLogValues)
    lnL_max = xpy_here.max(newLogValues)
    ratio = newLogValues - lnL_max
    dat = xpy_here.exp(ratio)
    log_mean = xpy_here.log(xpy_here.mean(dat))
#    log_M2 = xpy_here.log(xpy_here.sum( (dat-xpy_here.exp(log_mean))**2))
    log_M2 = logsumexp( 2*xpy_here.log(xpy_here.abs(dat - xpy_here.exp(log_mean) )))
#    dat_raw = xpy_here.exp(newLogValues)
#    print(log_M2 + lnL_max*2, xpy_here.log( xpy_here.var(xpy_here.exp(newLogValues))*(n-1)) , xpy_here.sqrt(xpy_here.var(dat_raw))/xpy_here.mean(dat_raw)  )
#    log_M2 = xpy_here.log(xpy_here.var(dat))+xpy_here.log(n-1)

    return (n, log_mean, log_M2 , lnL_max)
def update_log(existingLogAggregate, newLogValues_orig,special=scipy.special,xpy=numpy):
    """
    logsumexp : warning it is implemented but has a different function name, need to wrap it carefully and detect which is used
    """
    logsumexp = special.logsumexp
    if isinstance(newLogValues_orig, (int, float, complex)):
        # Handle single digits.
        newLogValues_orig = [newLogValues_orig]
    xpy_here=xpy
    newLogValues = newLogValues_orig
    if special==scipy.special and xpy != numpy:
        newLogValues = xpy.copy(newLogValues_orig) # copy, so we don't just edit the pointer contents
        newLogValues = xpy.asnumpy(newLogValues)
        xpy_here=numpy

    # https://docs.cupy.dev/en/latest/reference/generated/cupyx.scipy.special.logsumexp.html
    (nA, log_xAmean, log_M2A,log_refA) = existingLogAggregate

    # Evaluate reference scale, B for mean
    nB = len(newLogValues)
    log_refB = xpy_here.max(newLogValues)
    log_xBmean = logsumexp(newLogValues - log_refB) - xpy_here.log(nB)
    # compute M2AB after removing scale factor from all the terms
#    log_M2B = xpy_here.log(xpy_here.var(newLogValues - log_refB)) + xpy_here.log(nB-1)
    log_M2B = logsumexp( 2*xpy_here.log(xpy_here.abs(xpy_here.exp(newLogValues-log_refB) - xpy_here.exp(log_xBmean) )))

    # Find new common scale factor, and apply it
    #   Warning: cupy.max does not work recently, must cast
    logRef = xpy_here.max(xpy_here.array([log_refA,log_refB]))
    log_xAmean += -(logRef - log_refA)
    log_xBmean += -(logRef - log_refB)
    log_M2A += -2*(logRef-log_refA)  # scale is quadratic
    log_M2B += -2*(logRef-log_refB)

    # Update mean and second moment
    log_xNewMean = logsumexp(xpy_here.array([log_xAmean + xpy_here.log(nA),log_xBmean + xpy_here.log(nB)])) - xpy_here.log(nA+nB)
    log_delta = xpy_here.log(xpy_here.abs(xpy_here.exp(log_xAmean)- xpy_here.exp(log_xBmean))) # sign irrelevant
    log_M2New = logsumexp(xpy_here.array([log_M2A,log_M2B,2*log_delta + xpy_here.log(nA)+ xpy_here.log(nB) - xpy_here.log(nA+nB)]))

    # return new aggregate
    return (nA+nB, log_xNewMean, log_M2New, logRef)
def finalize_log(existingAggregate,xpy=numpy):
    """

    fallback mode: if special is fallback, then the aggregate is internally done with the normal numpy calculation!
    """
    
    (count, log_mean_orig, log_M2, log_ref) = existingAggregate
    (log_mean,  log_sampleVariance) = (log_mean_orig+log_ref, log_M2 + 2*log_ref - xpy.log((count - 1)))
#     print( log_mean, log_sampleVariance)
    if count < 2:
         return float('nan')
    else:
         return (log_mean,  log_sampleVariance)


#
# MC-error stabilization helpers (host-side, numpy only).
#
# Motivation: the sample variance of the importance weights, computed from the
# SAME draws as the integral, is algebraically 1/ESS_hat - 1/n restated.  It is
# tail-blind: a run that has not sampled the dominant weight region reports BOTH
# a low integral AND a small error bar, so the naive estimate fails conditionally
# on the run being wrong.  These helpers provide (a) a generalized-Pareto tail
# diagnostic (PSIS k-hat, Vehtari et al. JMLR 2024), (b) an ESS statistic,
# (c) a between-chunk jackknife scatter that sees adaptation nonstationarity,
# and (d) bootstrap quantiles of lnZ for honest (asymmetric) intervals when the
# relative error is large.  All are cheap relative to likelihood evaluations and
# must be called on CPU (numpy) arrays.
#

def pareto_khat_from_log(log_wt, tail_frac=0.2, min_tail=20):
    """Generalized-Pareto tail index k of the importance-weight distribution,
    fit to the largest weights (Zhang & Stephens 2009 posterior-mean estimator,
    as used by Pareto-smoothed importance sampling).  Input is LOG weights on
    any scale (k is invariant under overall rescaling).  Interpretation:
    k < 0.5 : weight variance finite, the naive error estimate is meaningful;
    0.5-0.7 : variance marginal, treat the naive sigma as optimistic;
    k > 0.7 : the weight tail is unresolved -- the naive sigma is a lower bound
              and the integral itself may be dominated by unseen tail mass.
    Returns float k, or None if there are too few finite weights to fit."""
    lw = numpy.asarray(log_wt, dtype=float)
    lw = lw[numpy.isfinite(lw)]
    n = len(lw)
    if n < 5 * min_tail:
        return None
    lw = numpy.sort(lw)
    w = numpy.exp(lw - lw[-1])   # rescale by max: tail values are O(1), rest may underflow harmlessly
    M = int(min(tail_frac * n, numpy.ceil(3 * numpy.sqrt(n))))
    M = max(M, min_tail)
    if M >= n:
        M = n - 1
    tail = w[-M:]
    mu = w[-M - 1]               # threshold = largest non-tail weight
    x = tail - mu
    if x[-1] <= 0:
        return 0.0               # massive ties at the top: no resolvable tail
    x = x[x > 0]
    nt = len(x)
    if nt < min_tail:
        return 0.0
    xstar = x[int(nt / 4 + 0.5) - 1]
    if xstar <= 0:
        return 0.0
    m = 30 + int(numpy.sqrt(nt))
    jj = numpy.arange(1, m + 1)
    theta = 1.0 / x[-1] + (1 - numpy.sqrt(m / (jj - 0.5))) / (3.0 * xstar)
    theta[theta == 0] = 1e-12    # avoid the (measure-zero) singular point
    # profile log-likelihood of the GPD for each candidate theta
    k_of = -numpy.mean(numpy.log1p(-numpy.outer(theta, x)), axis=1)
    k_of[numpy.abs(k_of) < 1e-12] = 1e-12
    with numpy.errstate(divide='ignore', invalid='ignore'):
        lp = nt * (numpy.log(theta / k_of) + k_of - 1)
    lp[~numpy.isfinite(lp)] = -numpy.inf
    lp -= lp.max()
    wts = numpy.exp(lp)
    s = wts.sum()
    if not numpy.isfinite(s) or s <= 0:
        return None
    theta_hat = numpy.sum(theta * wts) / s
    if theta_hat == 0:
        return 0.0
    # Zhang & Stephens parameterize the GPD with k_ZS = -xi (their k>0 is a
    # BOUNDED tail); return the PSIS/Vehtari tail index xi = -k_ZS, so that
    # heavy tails give POSITIVE k-hat and the 0.5/0.7 thresholds apply.
    return float(numpy.mean(numpy.log1p(-theta_hat * x)))


def ess_from_log_weights(log_wt):
    """Kish effective sample size (sum w)^2 / sum w^2 from LOG weights."""
    lw = numpy.asarray(log_wt, dtype=float)
    lw = lw[numpy.isfinite(lw)]
    if len(lw) == 0:
        return 0.0
    lse = scipy.special.logsumexp
    return float(numpy.exp(2 * lse(lw) - lse(2 * lw)))


def block_scatter_sigma(lnZ_blocks, n_blocks):
    """sigma(lnZ) from the between-chunk scatter of per-chunk mean estimates,
    via a delete-one jackknife of the n-weighted pooled mean.  Each chunk of an
    adaptive run used a different proposal, so this sees the nonstationarity the
    pooled within-run variance averages away.  (It still cannot see modes that
    EVERY chunk missed -- only independent replicas can.)  Returns float sigma
    or None if fewer than 2 usable chunks."""
    lnZ = numpy.asarray(lnZ_blocks, dtype=float)
    nb = numpy.asarray(n_blocks, dtype=float)
    good = numpy.isfinite(lnZ) & (nb > 0)
    lnZ = lnZ[good]
    nb = nb[good]
    K = len(lnZ)
    if K < 2:
        return None
    ref = lnZ.max()
    Z = numpy.exp(lnZ - ref)
    tot = numpy.sum(nb * Z)
    N = nb.sum()
    loo = (tot - nb * Z) / (N - nb)      # leave-one-out pooled means (relative to ref)
    if tot <= 0 or numpy.any(loo <= 0):
        return None
    ln_loo = numpy.log(loo)
    var_jk = (K - 1) / K * numpy.sum((ln_loo - ln_loo.mean()) ** 2)
    return float(numpy.sqrt(var_jk))


def bootstrap_lnZ_quantiles(log_wt, n_total=None, n_boot=200, quantiles=(0.05, 0.5, 0.95), rng_seed=None):
    """Bootstrap quantiles of lnZ_hat = ln( sum_i w_i / n_total ) by resampling
    the stored LOG weights with replacement.  When the relative error is O(1)
    the delta-method +-sigma interval on lnZ is meaningless (the distribution is
    strongly skewed); these quantiles are an honest same-sample interval.  They
    remain blind to tail mass never sampled -- pair with pareto_khat_from_log.
    n_total: divisor if the stored weights are a pruned subset of a larger run
    (the pruned-away weights contribute negligibly to the sum).  Returns a
    numpy array of lnZ quantiles, or None if too few weights."""
    lw = numpy.asarray(log_wt, dtype=float)
    lw = lw[numpy.isfinite(lw)]
    n = len(lw)
    if n < 10:
        return None
    if n_total is None:
        n_total = n
    # Reproducibility: default_rng(None) takes fresh OS entropy, so the printed
    # interval moved between two invocations that agreed on lnZ to the last bit.
    # Derive the stream from --seed instead.  It MUST be a stream of its own and
    # must not consume numpy's global RNG: the samplers draw from that global
    # stream, so spending draws here would shift every subsequent sampler draw and
    # this diagnostic -- which is not allowed to touch the answer -- would change
    # lnL.  The counter keeps the per-point/per-replica bootstraps from all
    # resampling with the same indices.  Unseeded runs keep fresh entropy.
    if rng_seed is None:
        from RIFT.integrators.seeding import next_derived_rng
        rng = next_derived_rng('statutils.bootstrap_lnZ_quantiles')
    else:
        rng = numpy.random.default_rng(rng_seed)
    ref = lw.max()
    w = numpy.exp(lw - ref)
    out = numpy.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = numpy.log(numpy.sum(w[idx]))
    out += ref - numpy.log(n_total)
    return numpy.quantile(out, quantiles)
