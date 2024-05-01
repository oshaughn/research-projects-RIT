import numpy
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
		m, s = numpy.zeros(len(arr)+1), numpy.zeros(len(arr)+1,dtype=numpy.float128)
		m[0] = mean
		s[0] = var*(n-1)
		buf = numpy.array([0])
	else:
		m, s = numpy.zeros(arr.shape), numpy.zeros(arr.shape,dtype=numpy.float128)
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
