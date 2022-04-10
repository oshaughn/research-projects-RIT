import numpy

__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

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
