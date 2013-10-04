import numpy

__author__ = "Chris Pankow <pankow@gravity.phys.uwm.edu>"

#
# Stat utilities
#
def cumvar(arr, mean=None, var=None, n=0):
	"""
	Numerically stable running variance measure. See http://www.johndcook.com/standard_deviation.html for algorithm details. If mean and std are supplied, they will be used as the history values.
	"""
	if mean and var:
		m, s = numpy.zeros(len(arr)+1), numpy.zeros(len(arr)+1)
		m[0] = mean
		s[0] = var*(n-1)
		buf = numpy.array([0])
	else:
		m, s = numpy.zeros(arr.shape), numpy.zeros(arr.shape)
		m[0] = arr[0]
		buf = numpy.array([])

	for i, x in enumerate(numpy.concatenate((buf, arr))):
		k = i+1+n
		if i == 0: continue
		m[i] = m[i-1] + (x-m[i-1])/k
		s[i] = s[i-1] + (x-m[i-1])*(x-m[i])

	if mean and var:
		return s[1:]/numpy.arange(n, n + len(s)-1)
	else:
		return s/numpy.arange(n + 1, n + len(s)+1)
