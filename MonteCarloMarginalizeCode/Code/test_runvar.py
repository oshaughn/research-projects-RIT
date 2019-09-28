#!/usr/bin/env python
import sys
import numpy
import scipy.stats
from statutils import cumvar, int_var

print("Assert variance of zeros is zero.")
assert cumvar(numpy.ones(1000))[-1] == 0.0
print("...good")

print("Testing the variance of 100000 Gaussian random variables")
rvs = numpy.random.normal(0, 1, 100000)

print("Numpy sample variance %.13f, cumvar (no chunking): %.13f" % (rvs.std(ddof=1)**2, cumvar(rvs)[-1]))

print("Testing chunk by chunk...")
var, mean = None, None
for i in range(1000):
    chunk = rvs[i*100:(i+1)*100]
    var = cumvar(chunk, mean, var, i*100)[-1]
    #print mean, numpy.average(rvs[:(i+1)*100])
    mean = numpy.average(rvs[:(i+1)*100])
    print("Numpy sample variance %.13f, cumvar (chunk %d): %.13f" % (rvs[:(i+1)*100].std(ddof=1)**2, i, var))

print("cumvar/N (all chunks): %.13f, Lepage integral variance %.13f" % (var/len(rvs), int_var(rvs)))
print("numpy var/N (all chunks): %.13f, Lepage integral variance %.13f" % (rvs.std(ddof=1)**2/len(rvs), int_var(rvs)))
